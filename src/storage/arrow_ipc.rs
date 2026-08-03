// FalconDB — Arrow IPC reader / writer
//
// Arrow IPC is the zero-copy interchange format the Arrow ecosystem reads
// natively: pandas, Polars, DuckDB and Spark all load it without a conversion
// pass. Parquet is the storage format; IPC is the handoff format, so this is the
// cheap way to move a result set into another tool.
//
// Two framings exist and they are not interchangeable:
//   * **stream** (`.arrows`) — a flat sequence of batches, readable while it is
//     still being written. No index, so no random access.
//   * **file** (`.arrow`) — the same batches plus a footer holding block offsets,
//     which lets a reader seek to a batch without scanning the ones before it.
//
// Both are written here. A file-format reader cannot read a stream and vice
// versa, so `read_ipc_file` and `read_ipc_stream` are kept separate rather than
// guessing from the extension.

use std::io::{Read, Write};
use std::path::Path;

use arrow::ipc::{
    reader::{FileReader, StreamReader},
    writer::{FileWriter, StreamWriter},
};
use arrow::record_batch::RecordBatch as ArrowBatch;

use super::{
    table::RecordBatch,
    types::{FalconError, FalconResult},
};

fn io_err(e: impl std::fmt::Display) -> FalconError {
    FalconError::InvalidQuery(e.to_string())
}

// ─── Writing ──────────────────────────────────────────────────────────────────

/// Write batches in Arrow IPC **file** format to any sink.
///
/// Every batch must share the first batch's schema, which the Arrow writer
/// enforces — a mismatch is reported rather than producing a file that readers
/// would reject later.
pub fn write_ipc_file_to<W: Write>(batches: &[RecordBatch], sink: W) -> FalconResult<()> {
    let arrow_batches = to_arrow(batches)?;
    let schema = arrow_batches[0].schema();

    let mut writer = FileWriter::try_new(sink, &schema).map_err(io_err)?;
    for ab in &arrow_batches {
        writer.write(ab).map_err(io_err)?;
    }
    writer.finish().map_err(io_err)?;
    Ok(())
}

/// Write batches in Arrow IPC **stream** format to any sink.
pub fn write_ipc_stream_to<W: Write>(batches: &[RecordBatch], sink: W) -> FalconResult<()> {
    let arrow_batches = to_arrow(batches)?;
    let schema = arrow_batches[0].schema();

    let mut writer = StreamWriter::try_new(sink, &schema).map_err(io_err)?;
    for ab in &arrow_batches {
        writer.write(ab).map_err(io_err)?;
    }
    writer.finish().map_err(io_err)?;
    Ok(())
}

/// Write batches to a `.arrow` file (IPC file format).
pub fn write_ipc_file(batches: &[RecordBatch], path: &Path) -> FalconResult<()> {
    let file = std::fs::File::create(path).map_err(io_err)?;
    write_ipc_file_to(batches, file)
}

/// Serialise batches to an in-memory IPC stream buffer.
///
/// Useful for handing a result set to a caller over FFI or a socket without
/// going through the filesystem.
pub fn write_ipc_stream_to_vec(batches: &[RecordBatch]) -> FalconResult<Vec<u8>> {
    let mut buf = Vec::new();
    write_ipc_stream_to(batches, &mut buf)?;
    Ok(buf)
}

// ─── Reading ──────────────────────────────────────────────────────────────────

/// Read a `.arrow` file (IPC file format) into FalconDB batches.
pub fn read_ipc_file(path: &Path, table_name: &str) -> FalconResult<Vec<RecordBatch>> {
    let file = std::fs::File::open(path).map_err(io_err)?;
    let reader = FileReader::try_new(file, None).map_err(io_err)?;
    from_arrow(reader, table_name)
}

/// Read an IPC stream from any source into FalconDB batches.
pub fn read_ipc_stream<R: Read>(source: R, table_name: &str) -> FalconResult<Vec<RecordBatch>> {
    let reader = StreamReader::try_new(source, None).map_err(io_err)?;
    from_arrow(reader, table_name)
}

// ─── Conversion helpers ───────────────────────────────────────────────────────

fn to_arrow(batches: &[RecordBatch]) -> FalconResult<Vec<ArrowBatch>> {
    if batches.is_empty() {
        return Err(FalconError::InvalidQuery("no batches to write".into()));
    }
    batches.iter().map(super::parquet_io::batch_to_arrow).collect()
}

fn from_arrow(
    reader:     impl Iterator<Item = Result<ArrowBatch, arrow::error::ArrowError>>,
    table_name: &str,
) -> FalconResult<Vec<RecordBatch>> {
    let mut out = Vec::new();
    for batch in reader {
        let ab = batch.map_err(io_err)?;
        out.push(super::parquet_io::batch_from_arrow(&ab, table_name)?);
    }
    Ok(out)
}
