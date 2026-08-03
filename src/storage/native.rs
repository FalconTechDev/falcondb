use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};

use super::{
    column::Column,
    compression::{compress_column, decompress_column, CompressedColumn, Encoding},
    schema::{Field, Schema},
    table::{RecordBatch, Table},
    types::{DataType, FalconError, FalconResult},
};

// ─── FalconDB native format (.fdb) ────────────────────────────────────────────
//
// File layout:
//   [8  bytes] magic: b"FALCONDB"
//   [2  bytes] format version (little-endian u16)
//   [4  bytes] number of columns
//   [4  bytes] number of rows
//   Schema section:
//     per column:
//       [2  bytes] name length
//       [N  bytes] name (utf-8)
//       [1  byte ] data type id
//       [1  byte ] nullable flag
//   Column section:
//     per column:
//       [1  byte ] encoding id
//       [4  bytes] null_mask length
//       [N  bytes] null_mask
//       [4  bytes] null_count
//       [4  bytes] row_count
//       [4  bytes] data length
//       [N  bytes] data (lz4 compressed)
//       [1  byte ] has_dict flag
//       if has_dict:
//         [4  bytes] dict length
//         [N  bytes] dict (lz4 compressed)

const MAGIC: &[u8; 8] = b"FALCONDB";
const VERSION: u16 = 1;

// ─── Write ────────────────────────────────────────────────────────────────────

pub fn write_table(table: &Table, path: impl AsRef<Path>) -> FalconResult<()> {
    // Merge all batches into one for simplicity (we compress anyway).
    let total_rows = table.row_count();
    if total_rows == 0 {
        return Err(FalconError::InvalidQuery("cannot write empty table".into()));
    }

    let schema    = &table.schema;
    let col_count = schema.field_count();

    // Gather all values per column across batches.
    let mut columns: Vec<Column> = (0..col_count)
        .map(|ci| gather_column(table, ci))
        .collect::<FalconResult<_>>()?;

    let file = File::create(path.as_ref())?;
    let mut w = BufWriter::new(file);

    // Header.
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(col_count as u32).to_le_bytes())?;
    w.write_all(&(total_rows as u32).to_le_bytes())?;

    // Schema.
    for field in &schema.fields {
        let name_bytes = field.name.as_bytes();
        w.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
        w.write_all(name_bytes)?;
        w.write_all(&[dtype_id(&field.data_type)])?;
        write_dtype_extra(&mut w, &field.data_type)?;
        w.write_all(&[field.nullable as u8])?;
    }

    // Columns (compressed).
    for col in &mut columns {
        let comp = compress_column(col)?;
        write_compressed_column(&mut w, &comp)?;
    }

    w.flush()?;
    Ok(())
}

fn gather_column(table: &Table, col_idx: usize) -> FalconResult<Column> {
    use super::{
        column::{BooleanColumn, Decimal128Column, PrimitiveColumn, Utf8Column},
        types::ScalarValue,
    };

    let total = table.row_count();
    let dt    = &table.schema.fields[col_idx].data_type;

    // Collect all ScalarValues for this column.
    let mut vals: Vec<ScalarValue> = Vec::with_capacity(total);
    for batch in &table.batches {
        if col_idx >= batch.column_count() { continue; }
        let col = batch.column(col_idx);
        for row in 0..batch.row_count() {
            vals.push(col.get(row)?);
        }
    }

    Ok(match dt {
        DataType::Boolean => Column::Boolean(BooleanColumn::new(
            vals.iter().map(|v| v.as_bool()).collect(),
        )),
        DataType::Int32 => Column::Int32(PrimitiveColumn::new(
            vals.iter().map(|v| v.as_i64().map(|n| n as i32)).collect(),
        )),
        DataType::Int64 => Column::Int64(PrimitiveColumn::new(
            vals.iter().map(|v| v.as_i64()).collect(),
        )),
        DataType::Float32 => Column::Float32(PrimitiveColumn::new(
            vals.iter().map(|v| v.as_f64().map(|f| f as f32)).collect(),
        )),
        DataType::Float64 => Column::Float64(PrimitiveColumn::new(
            vals.iter().map(|v| v.as_f64()).collect(),
        )),
        DataType::Utf8 => Column::Utf8(Utf8Column::new(
            vals.into_iter().map(|v| match v {
                ScalarValue::Utf8(s) => s,
                _ => None,
            }).collect(),
        )),
        DataType::Date32 => Column::Date32(PrimitiveColumn::new(
            vals.iter().map(|v| v.as_date32()).collect(),
        )),
        DataType::Timestamp => Column::Timestamp(PrimitiveColumn::new(
            vals.iter().map(|v| v.as_timestamp()).collect(),
        )),
        DataType::Decimal128(p, s) => Column::Decimal128(Decimal128Column::new(
            vals.iter().map(|v| match v {
                ScalarValue::Decimal128(opt, _, _) => *opt,
                _ => None,
            }).collect(),
            *p, *s,
        )),
        DataType::Null => Column::Int64(PrimitiveColumn::new(vec![None; total])),
    })
}

fn write_compressed_column<W: Write>(w: &mut W, comp: &CompressedColumn) -> FalconResult<()> {
    w.write_all(&[encoding_id(&comp.encoding)])?;
    w.write_all(&(comp.null_mask.len() as u32).to_le_bytes())?;
    w.write_all(&comp.null_mask)?;
    w.write_all(&(comp.null_count as u32).to_le_bytes())?;
    w.write_all(&(comp.row_count as u32).to_le_bytes())?;
    w.write_all(&(comp.data.len() as u32).to_le_bytes())?;
    w.write_all(&comp.data)?;

    match &comp.dict {
        Some(dict) => {
            w.write_all(&[1u8])?;
            w.write_all(&(dict.len() as u32).to_le_bytes())?;
            w.write_all(dict)?;
        }
        None => {
            w.write_all(&[0u8])?;
        }
    }

    Ok(())
}

// ─── Read ─────────────────────────────────────────────────────────────────────

pub fn read_table(path: impl AsRef<Path>, table_name: &str) -> FalconResult<Table> {
    let bytes = fs::read(path.as_ref())?;
    let mut pos = 0usize;

    // Header.
    check_magic(&bytes, &mut pos)?;
    let _version  = read_u16(&bytes, &mut pos)?;
    let col_count = read_u32(&bytes, &mut pos)? as usize;
    let row_count = read_u32(&bytes, &mut pos)? as usize;

    // Schema.
    let mut fields = Vec::with_capacity(col_count);
    for _ in 0..col_count {
        let name_len  = read_u16(&bytes, &mut pos)? as usize;
        let name      = read_str(&bytes, &mut pos, name_len)?;
        let dt        = read_dtype(&bytes, &mut pos)?;
        let nullable  = read_u8(&bytes, &mut pos)? != 0;
        fields.push(Field::new(name, dt, nullable));
    }

    let schema = Schema::new(fields);

    // Columns.
    let mut columns = Vec::with_capacity(col_count);
    for ci in 0..col_count {
        let comp = read_compressed_column(&bytes, &mut pos, &schema.fields[ci].data_type)?;
        let col  = decompress_column(&comp)?;

        if col.len() != row_count {
            return Err(FalconError::ParseError(format!(
                "column {ci} has {} rows, expected {row_count}",
                col.len(),
            )));
        }

        columns.push(col);
    }

    let batch = RecordBatch::new(schema.clone(), columns)?;
    let mut table = Table::new(table_name, schema);
    table.append_batch(batch)?;
    Ok(table)
}

fn read_compressed_column(
    bytes: &[u8],
    pos:   &mut usize,
    dt:    &DataType,
) -> FalconResult<CompressedColumn> {
    let encoding_byte = read_u8(bytes, pos)?;
    let encoding = encoding_from_id(encoding_byte)?;

    let mask_len  = read_u32(bytes, pos)? as usize;
    let null_mask = bytes[*pos..*pos + mask_len].to_vec();
    *pos += mask_len;

    let null_count = read_u32(bytes, pos)? as usize;
    let row_count  = read_u32(bytes, pos)? as usize;
    let data_len   = read_u32(bytes, pos)? as usize;
    let data       = bytes[*pos..*pos + data_len].to_vec();
    *pos += data_len;

    let has_dict = read_u8(bytes, pos)?;
    let dict = if has_dict != 0 {
        let dict_len = read_u32(bytes, pos)? as usize;
        let d = bytes[*pos..*pos + dict_len].to_vec();
        *pos += dict_len;
        Some(d)
    } else {
        None
    };

    Ok(CompressedColumn {
        data_type: dt.clone(),
        encoding,
        null_mask,
        null_count,
        row_count,
        data,
        dict,
    })
}

// ─── Primitive I/O helpers ────────────────────────────────────────────────────

fn read_u8(bytes: &[u8], pos: &mut usize) -> FalconResult<u8> {
    if *pos >= bytes.len() {
        return Err(FalconError::ParseError("unexpected EOF reading u8".into()));
    }
    let v = bytes[*pos];
    *pos += 1;
    Ok(v)
}

fn read_u16(bytes: &[u8], pos: &mut usize) -> FalconResult<u16> {
    let b = read_bytes::<2>(bytes, pos)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> FalconResult<u32> {
    let b = read_bytes::<4>(bytes, pos)?;
    Ok(u32::from_le_bytes(b))
}

fn read_bytes<const N: usize>(bytes: &[u8], pos: &mut usize) -> FalconResult<[u8; N]> {
    if *pos + N > bytes.len() {
        return Err(FalconError::ParseError(format!("unexpected EOF reading {N} bytes")));
    }
    let arr: [u8; N] = bytes[*pos..*pos + N].try_into().unwrap();
    *pos += N;
    Ok(arr)
}

fn read_str(bytes: &[u8], pos: &mut usize, len: usize) -> FalconResult<String> {
    if *pos + len > bytes.len() {
        return Err(FalconError::ParseError("unexpected EOF reading string".into()));
    }
    let s = std::str::from_utf8(&bytes[*pos..*pos + len])
        .map_err(|_| FalconError::ParseError("invalid utf-8 in field name".into()))?
        .to_string();
    *pos += len;
    Ok(s)
}

fn check_magic(bytes: &[u8], pos: &mut usize) -> FalconResult<()> {
    if bytes.len() < 8 || &bytes[..8] != MAGIC {
        return Err(FalconError::ParseError("not a FalconDB file".into()));
    }
    *pos += 8;
    Ok(())
}

// ─── Type encoding ────────────────────────────────────────────────────────────

fn dtype_id(dt: &DataType) -> u8 {
    match dt {
        DataType::Boolean        => 0,
        DataType::Int32          => 1,
        DataType::Int64          => 2,
        DataType::Float32        => 3,
        DataType::Float64        => 4,
        DataType::Utf8           => 5,
        DataType::Null           => 6,
        DataType::Date32         => 7,
        DataType::Timestamp      => 8,
        DataType::Decimal128(_, _) => 9,
    }
}

/// Read a DataType from the schema section. For Decimal128 (id=9) reads 2 extra bytes.
fn read_dtype(bytes: &[u8], pos: &mut usize) -> FalconResult<DataType> {
    let id = read_u8(bytes, pos)?;
    match id {
        0 => Ok(DataType::Boolean),
        1 => Ok(DataType::Int32),
        2 => Ok(DataType::Int64),
        3 => Ok(DataType::Float32),
        4 => Ok(DataType::Float64),
        5 => Ok(DataType::Utf8),
        6 => Ok(DataType::Null),
        7 => Ok(DataType::Date32),
        8 => Ok(DataType::Timestamp),
        9 => {
            let precision = read_u8(bytes, pos)?;
            let scale     = read_u8(bytes, pos)? as i8;
            Ok(DataType::Decimal128(precision, scale))
        }
        _ => Err(FalconError::ParseError(format!("unknown DataType id {id}"))),
    }
}

/// Write extra schema bytes for types that need them (currently only Decimal128).
fn write_dtype_extra<W: Write>(w: &mut W, dt: &DataType) -> FalconResult<()> {
    if let DataType::Decimal128(p, s) = dt {
        w.write_all(&[*p, *s as u8])?;
    }
    Ok(())
}

fn encoding_id(e: &Encoding) -> u8 {
    match e {
        Encoding::Lz4Raw         => 0,
        Encoding::DictionaryLz4  => 1,
        Encoding::BitPackedLz4   => 2,
        Encoding::DeltaLz4       => 3,
        Encoding::RleLz4         => 4,
    }
}

fn encoding_from_id(id: u8) -> FalconResult<Encoding> {
    match id {
        0 => Ok(Encoding::Lz4Raw),
        1 => Ok(Encoding::DictionaryLz4),
        2 => Ok(Encoding::BitPackedLz4),
        3 => Ok(Encoding::DeltaLz4),
        4 => Ok(Encoding::RleLz4),
        _ => Err(FalconError::ParseError(format!("unknown Encoding id {id}"))),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::{RecordBatch, Table},
        types::{DataType, ScalarValue},
    };
    use tempfile::TempDir;

    fn make_table() -> Table {
        let schema = Schema::new(vec![
            Field::required("id",    DataType::Int64),
            Field::nullable("name",  DataType::Utf8),
            Field::nullable("score", DataType::Float64),
            Field::nullable("flag",  DataType::Boolean),
        ]);
        let batch = RecordBatch::new(schema.clone(), vec![
            ColumnBuilder::int64(vec![Some(1), Some(2), Some(3)]),
            ColumnBuilder::utf8_from_str(vec![Some("alice"), None, Some("carol")]),
            ColumnBuilder::float64(vec![Some(9.5), Some(8.0), None]),
            ColumnBuilder::boolean(vec![Some(true), Some(false), None]),
        ]).unwrap();

        let mut t = Table::new("users", schema);
        t.append_batch(batch).unwrap();
        t
    }

    #[test]
    fn roundtrip() {
        let dir  = TempDir::new().unwrap();
        let path = dir.path().join("users.fdb");

        let original = make_table();
        write_table(&original, &path).unwrap();
        let loaded = read_table(&path, "users").unwrap();

        assert_eq!(loaded.row_count(), original.row_count());
        assert_eq!(loaded.schema.field_count(), original.schema.field_count());

        let ob = &original.batches[0];
        let lb = &loaded.batches[0];

        for ci in 0..ob.column_count() {
            for ri in 0..ob.row_count() {
                assert_eq!(
                    ob.column(ci).get(ri).unwrap(),
                    lb.column(ci).get(ri).unwrap(),
                );
            }
        }
    }

    #[test]
    fn bad_magic_rejected() {
        let dir  = TempDir::new().unwrap();
        let path = dir.path().join("bad.fdb");
        fs::write(&path, b"NOTFDBXX...").unwrap();
        assert!(read_table(&path, "t").is_err());
    }

    #[test]
    fn null_preservation() {
        let dir   = TempDir::new().unwrap();
        let path  = dir.path().join("nulls.fdb");
        let table = make_table();
        write_table(&table, &path).unwrap();
        let loaded = read_table(&path, "users").unwrap();

        let name_col = loaded.batches[0].column_by_name("name").unwrap();
        assert_eq!(name_col.get(1).unwrap(), ScalarValue::Utf8(None));

        let score_col = loaded.batches[0].column_by_name("score").unwrap();
        assert_eq!(score_col.get(2).unwrap(), ScalarValue::Float64(None));
    }
}
