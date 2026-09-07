// FalconDB — Parquet reader / writer
//
// Converts between FalconDB's RecordBatch / Column types and Apache Arrow /
// Parquet.  Uses the official `arrow` + `parquet` crates.

use std::path::Path;
use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, BooleanArray, Float32Array, Float64Array,
            Int32Array, Int64Array, StringArray},
    datatypes::{DataType as ArrowDT, Field as ArrowField, Schema as ArrowSchema},
    record_batch::RecordBatch as ArrowBatch,
};
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};
use parquet::file::properties::WriterProperties;

use super::{
    column::{Column, ColumnBuilder},
    schema::{Field, Schema},
    table::RecordBatch,
    types::{DataType, FalconError, FalconResult},
};

// ─── FalconDB → Arrow ─────────────────────────────────────────────────────────

fn falcon_col_to_arrow(col: &Column) -> FalconResult<ArrayRef> {
    Ok(match col {
        Column::Boolean(c) => {
            let vals: Vec<Option<bool>> = c.values.iter().copied().collect();
            Arc::new(BooleanArray::from(vals)) as ArrayRef
        }
        Column::Int32(c) => {
            let vals: Vec<Option<i32>> = c.iter().collect();
            Arc::new(Int32Array::from(vals)) as ArrayRef
        }
        Column::Int64(c) => {
            let vals: Vec<Option<i64>> = c.iter().collect();
            Arc::new(Int64Array::from(vals)) as ArrayRef
        }
        Column::Float32(c) => {
            let vals: Vec<Option<f32>> = c.iter().collect();
            Arc::new(Float32Array::from(vals)) as ArrayRef
        }
        Column::Float64(c) => {
            let vals: Vec<Option<f64>> = c.iter().collect();
            Arc::new(Float64Array::from(vals)) as ArrayRef
        }
        Column::Utf8(c) => {
            let owned: Vec<Option<String>> = c.iter()
                .map(|v| v.map(|s| s.to_string()))
                .collect();
            let vals: Vec<Option<&str>> = owned.iter()
                .map(|v| v.as_deref())
                .collect();
            Arc::new(StringArray::from(vals)) as ArrayRef
        }
        Column::Date32(c) => {
            let vals: Vec<Option<i32>> = c.iter().collect();
            Arc::new(arrow::array::Date32Array::from(vals)) as ArrayRef
        }
        Column::Timestamp(c) => {
            let vals: Vec<Option<i64>> = c.iter().collect();
            Arc::new(
                arrow::array::TimestampMicrosecondArray::from(vals)
                    .with_timezone_opt(None::<String>),
            ) as ArrayRef
        }
        Column::Decimal128(c) => {
            let vals: Vec<Option<i128>> = c.inner.iter().collect();
            Arc::new(
                arrow::array::Decimal128Array::from(vals)
                    .with_precision_and_scale(c.precision, c.scale)
                    .map_err(|e| FalconError::InvalidQuery(e.to_string()))?,
            ) as ArrayRef
        }
    })
}

fn falcon_dt_to_arrow(dt: &DataType) -> ArrowDT {
    match dt {
        DataType::Boolean       => ArrowDT::Boolean,
        DataType::Int32         => ArrowDT::Int32,
        DataType::Int64         => ArrowDT::Int64,
        DataType::Float32       => ArrowDT::Float32,
        DataType::Float64       => ArrowDT::Float64,
        DataType::Utf8          => ArrowDT::Utf8,
        DataType::Null          => ArrowDT::Null,
        DataType::Date32        => ArrowDT::Date32,
        DataType::Timestamp     => ArrowDT::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
        DataType::Decimal128(p, s) => ArrowDT::Decimal128(*p, *s),
    }
}

/// Convert a FalconDB batch to Arrow. Shared with the Arrow IPC writer.
pub(crate) fn batch_to_arrow(batch: &RecordBatch) -> FalconResult<ArrowBatch> {
    let fields: Vec<ArrowField> = batch.schema.fields.iter()
        .map(|f| ArrowField::new(
            &f.name,
            falcon_dt_to_arrow(&f.data_type),
            f.nullable,
        ))
        .collect();
    let schema = Arc::new(ArrowSchema::new(fields));

    let arrays: Vec<ArrayRef> = batch.columns.iter()
        .map(|c| falcon_col_to_arrow(c.as_ref()))
        .collect::<FalconResult<_>>()?;

    ArrowBatch::try_new(schema, arrays)
        .map_err(|e| FalconError::InvalidQuery(e.to_string()))
}

// ─── Arrow → FalconDB ─────────────────────────────────────────────────────────

fn arrow_dt_to_falcon(dt: &ArrowDT) -> FalconResult<DataType> {
    Ok(match dt {
        ArrowDT::Boolean   => DataType::Boolean,
        // Narrow integer types widen to Int32/Int64 (no data loss for stored values).
        ArrowDT::Int8 | ArrowDT::Int16 | ArrowDT::Int32 => DataType::Int32,
        ArrowDT::Int64  => DataType::Int64,
        // Unsigned ints: UInt8/16/32 → Int32 (safe within SQL range), UInt64 → Int64.
        ArrowDT::UInt8 | ArrowDT::UInt16 | ArrowDT::UInt32 => DataType::Int32,
        ArrowDT::UInt64 => DataType::Int64,
        ArrowDT::Float32   => DataType::Float32,
        ArrowDT::Float64   => DataType::Float64,
        ArrowDT::Utf8 | ArrowDT::LargeUtf8 => DataType::Utf8,
        ArrowDT::Null      => DataType::Null,
        ArrowDT::Date32    => DataType::Date32,
        ArrowDT::Timestamp(_, _) => DataType::Timestamp,
        ArrowDT::Decimal128(p, s) => DataType::Decimal128(*p, *s),
        other => return Err(FalconError::InvalidQuery(
            format!("unsupported Arrow type in Parquet: {other}")
        )),
    })
}

fn arrow_array_to_falcon(arr: &dyn Array, dt: &DataType) -> FalconResult<Column> {
    use arrow::array::*;
    let len = arr.len();
    if matches!(dt, DataType::Null) {
        return Ok(ColumnBuilder::int64(vec![None; len]));
    }
    Ok(match dt {
        DataType::Boolean => {
            let a = arr.as_any().downcast_ref::<BooleanArray>()
                .ok_or_else(|| FalconError::InvalidQuery("expected BooleanArray".into()))?;
            ColumnBuilder::boolean(
                (0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i)) }).collect()
            )
        }
        DataType::Int32 => {
            use arrow::array::{Int8Array, Int16Array, UInt8Array, UInt16Array, UInt32Array};
            if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
                ColumnBuilder::int32((0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i)) }).collect())
            } else if let Some(a) = arr.as_any().downcast_ref::<Int16Array>() {
                ColumnBuilder::int32((0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i) as i32) }).collect())
            } else if let Some(a) = arr.as_any().downcast_ref::<Int8Array>() {
                ColumnBuilder::int32((0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i) as i32) }).collect())
            } else if let Some(a) = arr.as_any().downcast_ref::<UInt8Array>() {
                ColumnBuilder::int32((0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i) as i32) }).collect())
            } else if let Some(a) = arr.as_any().downcast_ref::<UInt16Array>() {
                ColumnBuilder::int32((0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i) as i32) }).collect())
            } else if let Some(a) = arr.as_any().downcast_ref::<UInt32Array>() {
                ColumnBuilder::int32((0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i) as i32) }).collect())
            } else {
                return Err(FalconError::InvalidQuery("expected Int32-compatible array".into()));
            }
        }
        DataType::Int64 => {
            use arrow::array::UInt64Array;
            if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
                ColumnBuilder::int64((0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i)) }).collect())
            } else if let Some(a) = arr.as_any().downcast_ref::<UInt64Array>() {
                ColumnBuilder::int64((0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i) as i64) }).collect())
            } else {
                return Err(FalconError::InvalidQuery("expected Int64-compatible array".into()));
            }
        }
        DataType::Float32 => {
            let a = arr.as_any().downcast_ref::<Float32Array>()
                .ok_or_else(|| FalconError::InvalidQuery("expected Float32Array".into()))?;
            ColumnBuilder::float32(
                (0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i)) }).collect()
            )
        }
        DataType::Float64 => {
            let a = arr.as_any().downcast_ref::<Float64Array>()
                .ok_or_else(|| FalconError::InvalidQuery("expected Float64Array".into()))?;
            ColumnBuilder::float64(
                (0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i)) }).collect()
            )
        }
        DataType::Utf8 => {
            use arrow::array::LargeStringArray;
            let vals: Vec<Option<String>> = if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
                (0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i).to_string()) }).collect()
            } else if let Some(a) = arr.as_any().downcast_ref::<LargeStringArray>() {
                (0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i).to_string()) }).collect()
            } else {
                return Err(FalconError::InvalidQuery("expected StringArray".into()));
            };
            ColumnBuilder::utf8(vals)
        }
        DataType::Date32 => {
            let a = arr.as_any().downcast_ref::<Date32Array>()
                .ok_or_else(|| FalconError::InvalidQuery("expected Date32Array".into()))?;
            ColumnBuilder::date32(
                (0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i)) }).collect()
            )
        }
        DataType::Timestamp => {
            let a = arr.as_any().downcast_ref::<TimestampMicrosecondArray>()
                .ok_or_else(|| FalconError::InvalidQuery("expected TimestampMicrosecondArray".into()))?;
            ColumnBuilder::timestamp(
                (0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i)) }).collect()
            )
        }
        DataType::Decimal128(p, s) => {
            let a = arr.as_any().downcast_ref::<Decimal128Array>()
                .ok_or_else(|| FalconError::InvalidQuery("expected Decimal128Array".into()))?;
            ColumnBuilder::decimal128(
                (0..len).map(|i| if arr.is_null(i) { None } else { Some(a.value(i)) }).collect(),
                *p, *s,
            )
        }
        DataType::Null => ColumnBuilder::int64(vec![None; len]),
    })
}

/// Convert an Arrow batch to FalconDB. Shared with the Arrow IPC reader.
pub(crate) fn batch_from_arrow(batch: &ArrowBatch, _table_name: &str) -> FalconResult<RecordBatch> {
    let schema_fields: Vec<Field> = batch.schema().fields().iter()
        .map(|f| {
            let dt = arrow_dt_to_falcon(f.data_type())?;
            Ok(Field {
                name:      f.name().clone(),
                data_type: dt,
                nullable:  f.is_nullable(),
            })
        })
        .collect::<FalconResult<_>>()?;

    let schema = Schema::new(schema_fields);
    let falcon_types: Vec<DataType> = schema.fields.iter().map(|f| f.data_type.clone()).collect();

    let columns: Vec<Column> = batch.columns().iter().zip(falcon_types.iter())
        .map(|(arr, dt)| arrow_array_to_falcon(arr.as_ref(), dt))
        .collect::<FalconResult<_>>()?;

    RecordBatch::new(schema, columns)
        .map_err(|e| FalconError::InvalidQuery(format!("arrow→falcon batch: {e}")))
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Write `batches` to a Parquet file at `path`.
pub fn write_parquet(batches: &[RecordBatch], path: &Path) -> FalconResult<()> {
    if batches.is_empty() {
        return Err(FalconError::InvalidQuery("no batches to write".into()));
    }

    // Convert first batch to get the Arrow schema.
    let arrow_batches: Vec<ArrowBatch> = batches.iter()
        .map(batch_to_arrow)
        .collect::<FalconResult<_>>()?;

    let schema = arrow_batches[0].schema();
    let props  = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::SNAPPY)
        .build();

    let file = std::fs::File::create(path)
        .map_err(|e| FalconError::InvalidQuery(e.to_string()))?;

    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| FalconError::InvalidQuery(e.to_string()))?;

    for ab in &arrow_batches {
        writer.write(ab).map_err(|e| FalconError::InvalidQuery(e.to_string()))?;
    }
    writer.close().map_err(|e| FalconError::InvalidQuery(e.to_string()))?;
    Ok(())
}

/// Read all row groups from a Parquet file into FalconDB RecordBatches.
pub fn read_parquet(path: &Path, table_name: &str) -> FalconResult<Vec<RecordBatch>> {
    read_parquet_projected(path, table_name, None)
}

/// Expand a `*`/`?` pattern into the matching files, sorted by name.
///
/// Only the final path component may contain wildcards — `data/*.parquet` works,
/// `data/*/file.parquet` does not. Recursive matching would need a directory
/// walk with its own symlink and permission questions; this covers the shape
/// that actually gets typed.
///
/// A path with no wildcard is returned unchanged, so callers can pass either.
/// Sorting makes the row order of a multi-file read reproducible, which
/// filesystem enumeration order alone does not guarantee.
pub fn expand_glob(pattern: &str) -> FalconResult<Vec<std::path::PathBuf>> {
    if !pattern.contains('*') && !pattern.contains('?') {
        return Ok(vec![std::path::PathBuf::from(pattern)]);
    }

    let path   = Path::new(pattern);
    let dir    = path.parent().filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let needle = path.file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| FalconError::InvalidQuery(format!("bad glob pattern: {pattern}")))?;

    if dir.to_string_lossy().contains('*') || dir.to_string_lossy().contains('?') {
        return Err(FalconError::InvalidQuery(
            "wildcards are only supported in the file name, not in directories".into(),
        ));
    }

    let entries = std::fs::read_dir(dir)
        .map_err(|e| FalconError::InvalidQuery(format!("cannot read {}: {e}", dir.display())))?;

    let mut out: Vec<std::path::PathBuf> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter(|e| {
            e.file_name().to_str().is_some_and(|n| glob_matches(n, needle))
        })
        .map(|e| e.path())
        .collect();

    if out.is_empty() {
        return Err(FalconError::InvalidQuery(format!("no files match {pattern}")));
    }
    out.sort();
    Ok(out)
}

/// Match a file name against a `*`/`?` pattern.
///
/// `*` matches any run including empty, `?` matches exactly one character.
fn glob_matches(name: &str, pattern: &str) -> bool {
    let n: Vec<char> = name.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    glob_rec(&n, &p)
}

fn glob_rec(n: &[char], p: &[char]) -> bool {
    match (n, p) {
        (_, []) => n.is_empty(),
        (_, ['*', rest @ ..]) => {
            if glob_rec(n, rest) {
                return true;
            }
            !n.is_empty() && glob_rec(&n[1..], p)
        }
        ([], _) => false,
        ([_, nr @ ..], ['?', pr @ ..]) => glob_rec(nr, pr),
        ([nc, nr @ ..], [pc, pr @ ..]) => nc == pc && glob_rec(nr, pr),
    }
}

/// Read every file matching `pattern` into one batch list.
///
/// Files are read in sorted name order and concatenated, so a dataset split
/// across `part-000.parquet`, `part-001.parquet`, … reads back in order. Every
/// file must share the first one's schema.
pub fn read_parquet_glob(pattern: &str, table_name: &str) -> FalconResult<Vec<RecordBatch>> {
    let files = expand_glob(pattern)?;
    let mut out = Vec::new();
    let mut schema: Option<crate::storage::schema::Schema> = None;

    for f in files {
        let batches = read_parquet(&f, table_name)?;
        if let Some(first) = batches.first() {
            match &schema {
                None => schema = Some(first.schema.clone()),
                Some(s) if s != &first.schema => {
                    return Err(FalconError::InvalidQuery(format!(
                        "{} has a different schema from the first file matched",
                        f.display()
                    )));
                }
                Some(_) => {}
            }
        }
        out.extend(batches);
    }
    Ok(out)
}

/// Read a Parquet file, optionally reading only the named columns.
///
/// Projection pushdown belongs here rather than after the load: a column the
/// query never mentions should not be decompressed, decoded, or allocated at
/// all. On a wide table that is most of the file.
///
/// Row groups are decoded in parallel. Each is independent — that is the unit
/// Parquet is built around — so this is the natural place to use every core, and
/// the previous reader used exactly one.
pub fn read_parquet_projected(
    path:       &Path,
    table_name: &str,
    columns:    Option<&std::collections::HashSet<String>>,
) -> FalconResult<Vec<RecordBatch>> {
    use parquet::arrow::ProjectionMask;
    use rayon::prelude::*;

    let open = || {
        std::fs::File::open(path).map_err(|e| FalconError::InvalidQuery(e.to_string()))
    };

    let builder = ParquetRecordBatchReaderBuilder::try_new(open()?)
        .map_err(|e| FalconError::InvalidQuery(e.to_string()))?;

    let n_groups = builder.metadata().num_row_groups();
    let parquet_schema = builder.parquet_schema();

    // Map the requested column names onto leaf indices. Names the file does not
    // have are ignored: the caller's set is a union across query clauses and may
    // mention columns from elsewhere.
    let mask = match columns {
        None => ProjectionMask::all(),
        Some(want) => {
            let leaves: Vec<usize> = (0..parquet_schema.num_columns())
                .filter(|&i| {
                    let name = parquet_schema.columns()[i].path().parts().join(".");
                    want.iter().any(|w| w.eq_ignore_ascii_case(&name))
                })
                .collect();
            // Nothing matched — fall back to reading everything rather than
            // silently producing a batch with no columns.
            if leaves.is_empty() {
                ProjectionMask::all()
            } else {
                ProjectionMask::leaves(parquet_schema, leaves)
            }
        }
    };

    if n_groups == 0 {
        return Ok(Vec::new());
    }

    // Each row group is opened and decoded independently so the work can spread
    // across cores; results are re-joined in file order to keep row order stable.
    let per_group: FalconResult<Vec<Vec<RecordBatch>>> = (0..n_groups)
        .into_par_iter()
        .map(|g| -> FalconResult<Vec<RecordBatch>> {
            // Batch size is deliberately left at the arrow-rs default of 1024, which
            // makes a Parquet load land as many small batches (1M rows -> 977).
            //
            // That looks wrong — the engine's per-batch thresholds are written
            // against DEFAULT_BATCH_SIZE (65_536), so at 1024 the intra-batch
            // parallel kernels never engage — and setting `.with_batch_size(
            // Table::DEFAULT_BATCH_SIZE)` here was tried. It is 8x slower:
            //
            //   full_scan/1M     170us -> 1.58ms   (+828%)
            //   group_by/1M     1.00ms -> 3.23ms   (+221%)
            //   full_scan/100k  63.4us -> 131us    (+107%)
            //   filter_where/1M 1.48ms -> 2.68ms    (+81%)
            //
            // The engine gets its parallelism from having *many batches to fan out
            // over* (`AGG_PARALLEL_THRESHOLD` = 32 batches), not from splitting one
            // big batch. 1M rows as 977 batches spreads across every core; as 16
            // batches it falls under the threshold and runs serially. Small batches
            // also keep each column slice inside L2 for the whole pass.
            //
            // Raise this only together with the inter-batch strategy in
            // `query::aggregate`, and measure `full_scan/1M` before believing it.
            let reader = ParquetRecordBatchReaderBuilder::try_new(open()?)
                .map_err(|e| FalconError::InvalidQuery(e.to_string()))?
                .with_projection(mask.clone())
                .with_row_groups(vec![g])
                .build()
                .map_err(|e| FalconError::InvalidQuery(e.to_string()))?;

            let mut out = Vec::new();
            for batch in reader {
                let ab = batch.map_err(|e| FalconError::InvalidQuery(e.to_string()))?;
                out.push(batch_from_arrow(&ab, table_name)?);
            }
            Ok(out)
        })
        .collect();

    Ok(per_group?.into_iter().flatten().collect())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{column::ColumnBuilder, schema::Field, table::RecordBatch, types::DataType};

    fn make_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::required("id",    DataType::Int64),
            Field::required("score", DataType::Float64),
            Field::nullable("name",  DataType::Utf8),
        ]);
        RecordBatch::new(schema, vec![
            ColumnBuilder::int64(vec![Some(1), Some(2), Some(3)]),
            ColumnBuilder::float64(vec![Some(1.1), Some(2.2), Some(3.3)]),
            ColumnBuilder::utf8(vec![Some("a".into()), None, Some("c".into())]),
        ]).unwrap()
    }

    #[test]
    fn parquet_roundtrip() {
        let batch   = make_batch();
        let tmp     = tempfile::NamedTempFile::new().unwrap();
        write_parquet(&[batch.clone()], tmp.path()).unwrap();
        let batches = read_parquet(tmp.path(), "test").unwrap();
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert_eq!(b.row_count(), 3);
        // id column
        assert_eq!(b.column_by_name("id").unwrap().get(0).unwrap(),
                   crate::storage::types::ScalarValue::Int64(Some(1)));
        // null in name column
        assert_eq!(b.column_by_name("name").unwrap().get(1).unwrap(),
                   crate::storage::types::ScalarValue::Utf8(None));
    }

    #[test]
    fn parquet_multiple_batches() {
        let b1 = make_batch();
        let b2 = make_batch();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_parquet(&[b1, b2], tmp.path()).unwrap();
        let batches = read_parquet(tmp.path(), "test").unwrap();
        let total: usize = batches.iter().map(|b| b.row_count()).sum();
        assert_eq!(total, 6);
    }
}
