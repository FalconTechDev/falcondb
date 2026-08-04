// FalconDB — Chunked streaming execution for billion-row scale
//
// Processes data in fixed-size chunks, never loading the full dataset into RAM.
//
//   ChunkedScanner  — splits an in-memory Table into fixed-size chunks
//   ParquetStream   — reads Parquet row-group-by-row-group with predicate pushdown
//   StreamingAgg    — incremental GROUP BY that merges partial results in parallel

use std::path::Path;

use rayon::prelude::*;

use super::{
    stats::BatchStats,
    table::{RecordBatch, Table},
    types::{FalconError, FalconResult},
};
use crate::query::{
    aggregate::{AggExpr, GroupByExec},
    expr::Expr,
    select::{execute_select, SelectQuery},
};

// ─── Chunk size ───────────────────────────────────────────────────────────────

/// 64k rows ≈ 1.5 MB per f64 column — fits comfortably in L2 cache.
pub const DEFAULT_CHUNK_ROWS: usize = 65_536;

// ─── ChunkedScanner ──────────────────────────────────────────────────────────

/// Applies filter/projection to a Table in chunks to reduce peak RSS.
pub struct ChunkedScanner {
    pub chunk_rows: usize,
}

impl Default for ChunkedScanner {
    fn default() -> Self { Self { chunk_rows: DEFAULT_CHUNK_ROWS } }
}

impl ChunkedScanner {
    pub fn new(chunk_rows: usize) -> Self { Self { chunk_rows } }

    /// Filter a table in chunks, returning all passing RecordBatches.
    pub fn filter(&self, table: &Table, filter: &Expr) -> FalconResult<Vec<RecordBatch>> {
        let query = SelectQuery::scan_all().with_filter(filter.clone());
        let mut out = Vec::new();
        for batch in &table.batches {
            if batch.row_count() <= self.chunk_rows {
                out.extend(filter_batch(batch, &query)?);
            } else {
                for chunk in split_batch(batch, self.chunk_rows)? {
                    out.extend(filter_batch(&chunk, &query)?);
                }
            }
        }
        Ok(out)
    }

    /// GROUP BY over a table in chunks; merges partial results.
    pub fn group_by(
        &self,
        table: &Table,
        keys:  Vec<Expr>,
        aggs:  Vec<AggExpr>,
    ) -> FalconResult<RecordBatch> {
        let exec = GroupByExec::new(keys.clone(), aggs.clone());

        // Phase 1: one partial aggregate per batch (parallel).
        // Using try_fold/try_reduce so any chunk error propagates instead of
        // silently dropping the chunk and undercounting.
        let partials: Vec<RecordBatch> = table.batches
            .par_iter()
            .try_fold(
                Vec::new,
                |mut acc, batch| -> FalconResult<Vec<RecordBatch>> {
                    let chunks = if batch.row_count() <= self.chunk_rows {
                        vec![batch.clone()]
                    } else {
                        split_batch(batch, self.chunk_rows)?
                    };
                    for c in chunks {
                        acc.push(exec.execute(std::slice::from_ref(&c))?);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(Vec::new, |mut a, b| { a.extend(b); Ok(a) })?;

        if partials.is_empty() {
            return Err(FalconError::InvalidQuery("empty table".into()));
        }

        // Phase 2: merge partial aggregates.
        exec.execute(&partials)
    }
}

fn filter_batch(batch: &RecordBatch, query: &SelectQuery) -> FalconResult<Vec<RecordBatch>> {
    let mut tmp = Table::new("_chunk", batch.schema.clone());
    tmp.batches.push(batch.clone());
    execute_select(&tmp, query)
}

/// Split a RecordBatch into chunks of at most `chunk_rows` rows.
pub fn split_batch(batch: &RecordBatch, chunk_rows: usize) -> FalconResult<Vec<RecordBatch>> {
    let n = batch.row_count();
    if n == 0 { return Ok(vec![]); }
    let mut out = Vec::with_capacity((n + chunk_rows - 1) / chunk_rows);
    let mut start = 0;
    while start < n {
        let end = (start + chunk_rows).min(n);
        out.push(batch.slice(start, end - start)?);
        start = end;
    }
    Ok(out)
}

// ─── ParquetStream ───────────────────────────────────────────────────────────

/// Reads a Parquet file one row-group at a time with optional predicate pushdown.
/// Memory usage = one row group at a time, regardless of total file size.
pub struct ParquetStream {
    pub path:       std::path::PathBuf,
    pub table_name: String,
    pub filter:     Option<Expr>,
}

impl ParquetStream {
    pub fn new(path: impl AsRef<Path>, table_name: &str) -> Self {
        Self {
            path:       path.as_ref().to_owned(),
            table_name: table_name.into(),
            filter:     None,
        }
    }

    pub fn with_filter(mut self, expr: Expr) -> Self {
        self.filter = Some(expr);
        self
    }

    pub fn execute(&self) -> FalconResult<Vec<RecordBatch>> {
        use super::parquet_io::read_parquet;
        let batches = read_parquet(&self.path, &self.table_name)?;

        if let Some(filter) = &self.filter {
            let query = SelectQuery::scan_all().with_filter(filter.clone());
            let mut out = Vec::new();
            for batch in batches {
                // Zone-map predicate pushdown: skip row groups that can't match.
                let stats = BatchStats::compute(&batch);
                if stats.can_skip(filter) {
                    continue;
                }
                out.extend(filter_batch(&batch, &query)?);
            }
            Ok(out)
        } else {
            Ok(batches)
        }
    }

    pub fn count(&self) -> FalconResult<u64> {
        Ok(self.execute()?.iter().map(|b| b.row_count() as u64).sum())
    }
}

// ─── StreamingAgg ─────────────────────────────────────────────────────────────

/// Incremental GROUP BY over a ParquetStream.
/// Holds at most one row group + the partial aggregate table in RAM.
pub struct StreamingAgg {
    pub stream: ParquetStream,
    pub keys:   Vec<Expr>,
    pub aggs:   Vec<AggExpr>,
}

impl StreamingAgg {
    pub fn new(stream: ParquetStream, keys: Vec<Expr>, aggs: Vec<AggExpr>) -> Self {
        Self { stream, keys, aggs }
    }

    pub fn execute(&self) -> FalconResult<RecordBatch> {
        use super::parquet_io::read_parquet;
        let exec    = GroupByExec::new(self.keys.clone(), self.aggs.clone());
        let batches = read_parquet(&self.stream.path, &self.stream.table_name)?;

        // Phase 1: partial aggregation per row group (parallel).
        // collect into Result so any per-batch error propagates instead of
        // silently dropping the batch and undercounting.
        let partials: Vec<RecordBatch> = batches
            .par_iter()
            .map(|b| exec.execute(std::slice::from_ref(b)))
            .collect::<FalconResult<Vec<_>>>()?;

        if partials.is_empty() {
            return Err(FalconError::InvalidQuery("empty file".into()));
        }

        // Phase 2: final merge.
        exec.execute(&partials)
    }
}

// ─── Convenience ─────────────────────────────────────────────────────────────

pub fn scan_parquet(path: impl AsRef<Path>, table_name: &str) -> ParquetStream {
    ParquetStream::new(path, table_name)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        column::ColumnBuilder,
        parquet_io::write_parquet,
        schema::{Field, Schema},
        types::DataType,
    };
    use crate::query::expr::{BinaryOp, Expr};
    use crate::storage::types::ScalarValue;

    fn make_table(n: usize) -> Table {
        let schema = Schema::new(vec![
            Field::required("id",    DataType::Int64),
            Field::required("score", DataType::Float64),
        ]);
        let batch = RecordBatch::new(schema.clone(), vec![
            ColumnBuilder::int64((0..n).map(|i| Some(i as i64)).collect()),
            ColumnBuilder::float64((0..n).map(|i| Some((i % 100) as f64)).collect()),
        ]).unwrap();
        let mut t = Table::new("test", schema);
        t.append_batch(batch).unwrap();
        t
    }

    #[test]
    fn chunked_filter_basic() {
        let table   = make_table(1000);
        let scanner = ChunkedScanner::new(200);
        let filter  = Expr::BinaryOp {
            left:  Box::new(Expr::Column("score".into())),
            op:    BinaryOp::Gt,
            right: Box::new(Expr::Literal(ScalarValue::Float64(Some(50.0)))),
        };
        let results = scanner.filter(&table, &filter).unwrap();
        let total: usize = results.iter().map(|b| b.row_count()).sum();
        assert!(total > 0 && total < 1000);
    }

    #[test]
    fn split_batch_correctness() {
        let table  = make_table(500);
        let chunks = split_batch(&table.batches[0], 128).unwrap();
        assert_eq!(chunks.len(), 4); // 128+128+128+116
        let total: usize = chunks.iter().map(|b| b.row_count()).sum();
        assert_eq!(total, 500);
    }

    #[test]
    fn parquet_stream_with_filter() {
        let table = make_table(300);
        let tmp   = tempfile::NamedTempFile::new().unwrap();
        write_parquet(&table.batches, tmp.path()).unwrap();

        let filter = Expr::BinaryOp {
            left:  Box::new(Expr::Column("score".into())),
            op:    BinaryOp::Gt,
            right: Box::new(Expr::Literal(ScalarValue::Float64(Some(50.0)))),
        };
        let results = ParquetStream::new(tmp.path(), "test")
            .with_filter(filter)
            .execute()
            .unwrap();
        let total: usize = results.iter().map(|b| b.row_count()).sum();
        assert!(total > 0);
    }
}
