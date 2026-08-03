use std::{
    fs::{File, OpenOptions},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
};

use super::types::{FalconError, FalconResult};

// ─── WAL entry format ─────────────────────────────────────────────────────────
//
// Each record on disk:
//   [4 bytes: magic 0xFA1C0DB5]
//   [4 bytes: entry type]
//   [8 bytes: sequence number]
//   [4 bytes: payload length]
//   [N bytes: payload]
//   [4 bytes: CRC32 of header + payload]

const MAGIC: u32 = 0xFA1C_0DB5;

#[derive(Debug, Clone, PartialEq)]
pub enum WalEntryType {
    Insert      = 1,
    Delete      = 2,
    Checkpoint  = 3,
    Truncate    = 4,
    Update      = 5,
}

impl WalEntryType {
    fn from_u32(n: u32) -> Option<Self> {
        match n {
            1 => Some(Self::Insert),
            2 => Some(Self::Delete),
            3 => Some(Self::Checkpoint),
            4 => Some(Self::Truncate),
            5 => Some(Self::Update),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub seq:     u64,
    pub kind:    WalEntryType,
    pub table:   String,
    pub payload: Vec<u8>,
}

// ─── WAL writer ───────────────────────────────────────────────────────────────

pub struct Wal {
    path:     PathBuf,
    file:     File,
    next_seq: u64,
}

impl Wal {
    pub fn open(path: impl AsRef<Path>) -> FalconResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;

        let next_seq = Self::scan_next_seq(&path)?;

        Ok(Self { path, file, next_seq })
    }

    /// Append an INSERT record for a table.
    pub fn write_insert(&mut self, table: &str, data: &[u8]) -> FalconResult<u64> {
        self.write_entry(WalEntryType::Insert, table, data)
    }

    /// Append a DELETE record.
    pub fn write_delete(&mut self, table: &str, data: &[u8]) -> FalconResult<u64> {
        self.write_entry(WalEntryType::Delete, table, data)
    }

    /// Append a CHECKPOINT — marks that all prior entries are flushed to storage.
    pub fn write_checkpoint(&mut self, table: &str) -> FalconResult<u64> {
        self.write_entry(WalEntryType::Checkpoint, table, &[])
    }

    /// Append a TRUNCATE record.
    pub fn write_truncate(&mut self, table: &str) -> FalconResult<u64> {
        self.write_entry(WalEntryType::Truncate, table, &[])
    }

    /// Append an UPDATE record.
    pub fn write_update(&mut self, table: &str, data: &[u8]) -> FalconResult<u64> {
        self.write_entry(WalEntryType::Update, table, data)
    }

    /// Path this WAL is backed by — needed to re-read the log during recovery.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of entries written since this WAL was opened (or last truncated).
    pub fn entry_count(&self) -> u64 {
        self.next_seq
    }

    /// Truncate the WAL file to zero bytes.
    ///
    /// Call this after a successful checkpoint: all state is in `.fdb` files and
    /// the WAL is no longer needed for recovery.  The sequence counter is kept so
    /// future entries get unique numbers; `entry_count()` resets to 0.
    pub fn truncate(&mut self) -> FalconResult<()> {
        // On Windows an open file handle blocks truncation of the same file.
        // Drop our handle first, open a new one with truncation, then reopen for append.
        drop(std::mem::replace(
            &mut self.file,
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&self.path)?,
        ));
        self.file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&self.path)?;
        self.next_seq = 0;
        Ok(())
    }

    fn write_entry(
        &mut self,
        kind:    WalEntryType,
        table:   &str,
        payload: &[u8],
    ) -> FalconResult<u64> {
        let seq     = self.next_seq;
        let table_b = table.as_bytes();

        // Payload = [u16 table name len][table name bytes][payload bytes]
        let mut data = Vec::with_capacity(2 + table_b.len() + payload.len());
        data.extend_from_slice(&(table_b.len() as u16).to_le_bytes());
        data.extend_from_slice(table_b);
        data.extend_from_slice(payload);

        let mut header = Vec::with_capacity(20);
        header.extend_from_slice(&MAGIC.to_le_bytes());
        header.extend_from_slice(&(kind as u32).to_le_bytes());
        header.extend_from_slice(&seq.to_le_bytes());
        header.extend_from_slice(&(data.len() as u32).to_le_bytes());

        let crc = crc32(&header, &data);

        self.file.write_all(&header)?;
        self.file.write_all(&data)?;
        self.file.write_all(&crc.to_le_bytes())?;
        self.file.flush()?;

        self.next_seq += 1;
        Ok(seq)
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    fn scan_next_seq(path: &Path) -> FalconResult<u64> {
        let entries = read_wal(path)?;
        Ok(entries.last().map(|e| e.seq + 1).unwrap_or(0))
    }
}

// ─── WAL reader ───────────────────────────────────────────────────────────────

pub fn read_wal(path: impl AsRef<Path>) -> FalconResult<Vec<WalEntry>> {
    let file = match File::open(path.as_ref()) {
        Ok(f)  => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(FalconError::Io(e)),
    };

    let mut reader  = BufReader::new(file);
    let mut entries = Vec::new();

    loop {
        let mut hdr = [0u8; 20];
        match reader.read_exact(&mut hdr) {
            Ok(())  => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(FalconError::Io(e)),
        }

        let magic   = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(FalconError::ParseError("WAL: bad magic number".into()));
        }

        let kind_u32   = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        let seq        = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
        let data_len   = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;

        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)
            .map_err(|_| FalconError::ParseError("WAL: truncated entry".into()))?;

        let mut crc_buf = [0u8; 4];
        reader.read_exact(&mut crc_buf)
            .map_err(|_| FalconError::ParseError("WAL: missing CRC".into()))?;

        let stored_crc = u32::from_le_bytes(crc_buf);
        let computed   = crc32(&hdr, &data);
        if stored_crc != computed {
            return Err(FalconError::ParseError(format!(
                "WAL: CRC mismatch at seq {seq}: stored {stored_crc:#010x}, computed {computed:#010x}"
            )));
        }

        let kind = WalEntryType::from_u32(kind_u32)
            .ok_or_else(|| FalconError::ParseError(format!("WAL: unknown entry type {kind_u32}")))?;

        let table_len = u16::from_le_bytes(data[0..2].try_into().unwrap()) as usize;
        let table = std::str::from_utf8(&data[2..2+table_len])
            .map_err(|_| FalconError::ParseError("WAL: invalid table name utf8".into()))?
            .to_string();
        let payload = data[2 + table_len..].to_vec();

        entries.push(WalEntry { seq, kind, table, payload });
    }

    Ok(entries)
}

/// Find the sequence number of the last CHECKPOINT for a table.
pub fn last_checkpoint_seq(entries: &[WalEntry], table: &str) -> Option<u64> {
    entries.iter().rev()
        .find(|e| e.kind == WalEntryType::Checkpoint && e.table == table)
        .map(|e| e.seq)
}

/// Return entries after the last checkpoint (the ones needing replay).
pub fn entries_to_replay<'a>(entries: &'a [WalEntry], table: &str) -> &'a [WalEntry] {
    match last_checkpoint_seq(entries, table) {
        None      => entries,
        Some(seq) => {
            let pos = entries.iter().position(|e| e.seq == seq).unwrap_or(0);
            &entries[pos + 1..]
        }
    }
}

// ─── CRC32 (simple Castagnoli) ────────────────────────────────────────────────

fn crc32(header: &[u8], data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in header.iter().chain(data.iter()) {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 == 1 { crc = (crc >> 1) ^ POLY; }
            else             { crc >>= 1; }
        }
    }
    !crc
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tmp_wal() -> (TempDir, Wal) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");
        (dir, Wal::open(&path).unwrap())
    }

    #[test]
    fn write_and_read_back() {
        let (dir, mut wal) = tmp_wal();
        let path = dir.path().join("test.wal");

        wal.write_insert("users", b"row data").unwrap();
        wal.write_insert("users", b"more data").unwrap();
        wal.write_checkpoint("users").unwrap();

        let entries = read_wal(&path).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].kind, WalEntryType::Insert);
        assert_eq!(entries[0].table, "users");
        assert_eq!(entries[0].payload, b"row data");
        assert_eq!(entries[2].kind, WalEntryType::Checkpoint);
    }

    #[test]
    fn sequence_numbers_increment() {
        let (_dir, mut wal) = tmp_wal();
        let s0 = wal.write_insert("t", b"a").unwrap();
        let s1 = wal.write_insert("t", b"b").unwrap();
        let s2 = wal.write_checkpoint("t").unwrap();
        assert!(s0 < s1 && s1 < s2);
    }

    #[test]
    fn entries_to_replay_returns_post_checkpoint() {
        let (dir, mut wal) = tmp_wal();
        let path = dir.path().join("test.wal");

        wal.write_insert("t", b"old").unwrap();
        wal.write_checkpoint("t").unwrap();
        wal.write_insert("t", b"new").unwrap();

        let entries = read_wal(&path).unwrap();
        let replay  = entries_to_replay(&entries, "t");
        assert_eq!(replay.len(), 1);
        assert_eq!(replay[0].payload, b"new");
    }

    #[test]
    fn survives_reopen() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.write_insert("t", b"hello").unwrap();
        }
        {
            let mut wal = Wal::open(&path).unwrap();
            assert_eq!(wal.next_seq(), 1);
            wal.write_insert("t", b"world").unwrap();
        }

        let entries = read_wal(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].payload, b"world");
    }

    #[test]
    fn crc_detects_corruption() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wal");

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.write_insert("t", b"data").unwrap();
        }

        // Corrupt a byte in the middle of the file.
        let mut raw = std::fs::read(&path).unwrap();
        let mid = raw.len() / 2;
        raw[mid] ^= 0xFF;
        std::fs::write(&path, raw).unwrap();

        assert!(read_wal(&path).is_err());
    }
}
