use std::sync::Arc;


use rayon::prelude::*;

/// Minimum total rows before concatenating columns in parallel is worth a rayon
/// hand-off. Below this the copy is short enough that scheduling costs more.
/// Measured at 262_144 on a machine reporting 20 threads, so ~13.1k each.
fn par_concat_rows() -> usize {
    crate::query::tuning::par_min_rows(13_107)
}

use super::{
    column::Column,
    schema::Schema,
    types::{FalconError, FalconResult},
};


/// A RecordBatch is a slice of a table: a fixed-size, schema-aligned collection of columns.
/// It is the unit that flows through the execution engine.
///
/// Columns are held behind `Arc` so that cloning a batch — which the engine does on
/// every unfiltered scan and every projection — costs a refcount bump instead of
/// deep-copying the data. That matters most for `Utf8`, where a deep clone means one
/// heap allocation *per row*.
#[derive(Debug, Clone)]
pub struct RecordBatch {
    pub schema:  Schema,
    pub columns: Vec<Arc<Column>>,
}

impl RecordBatch {
    pub fn new(schema: Schema, columns: Vec<Column>) -> FalconResult<Self> {
        Self::from_shared(schema, columns.into_iter().map(Arc::new).collect())
    }

    /// Build from columns that are already shared, so no data is copied.
    /// Use this whenever the columns come from an existing batch.
    pub fn from_shared(schema: Schema, columns: Vec<Arc<Column>>) -> FalconResult<Self> {
        if schema.field_count() != columns.len() {
            return Err(FalconError::InvalidQuery(format!(
                "schema has {} fields but {} columns were provided",
                schema.field_count(),
                columns.len(),
            )));
        }

        let row_count = columns.first().map(|c| c.len()).unwrap_or(0);
        for (i, col) in columns.iter().enumerate() {
            if col.len() != row_count {
                return Err(FalconError::InvalidQuery(format!(
                    "column '{}' has {} rows but expected {}",
                    schema.fields[i].name,
                    col.len(),
                    row_count,
                )));
            }
        }

        Ok(Self { schema, columns })
    }

    /// Concatenate batches that share a schema into one batch.
    ///
    /// Empty input yields an empty batch with an empty schema, which is the only
    /// sensible answer when there is no schema to inherit.
    ///
    /// This is deliberately the only implementation. `query::sort` and `query::join`
    /// each grew a private copy of it, and the join's copy pushed every cell through
    /// a `ScalarValue` — O(rows x cols) enum boxing for what is a memcpy per column.
    /// Nothing caught it because both copies were *correct*; the join was simply
    /// 100x slower, and nothing measured joins. Route new callers here.
    pub fn concat(batches: &[RecordBatch]) -> FalconResult<RecordBatch> {
        let Some(first) = batches.first() else {
            return RecordBatch::new(Schema::new(vec![]), vec![]);
        };
        if batches.len() == 1 {
            return Ok(first.clone());
        }

        for b in batches {
            if b.schema != first.schema {
                return Err(FalconError::InvalidQuery(
                    "cannot concatenate batches with different schemas".into(),
                ));
            }
        }

        // Columns concatenate independently. The copy is bandwidth-bound and one
        // thread does not saturate memory on its own, so wide-enough inputs spread
        // it over cores; small ones would only pay for the hand-off.
        let concat_col = |ci: usize| -> FalconResult<Column> {
            let parts: Vec<&Column> = batches.iter().map(|b| b.column(ci)).collect();
            Column::concat(&parts)
        };
        let n_cols = first.column_count();
        let total_rows: usize = batches.iter().map(|b| b.row_count()).sum();
        let columns: FalconResult<Vec<Column>> = if n_cols > 1 && total_rows >= par_concat_rows() {
            (0..n_cols).into_par_iter().map(concat_col).collect()
        } else {
            (0..n_cols).map(concat_col).collect()
        };

        RecordBatch::new(first.schema.clone(), columns?)
    }

    pub fn row_count(&self) -> usize {
        self.columns.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn column(&self, index: usize) -> &Column {
        &self.columns[index]
    }

    pub fn column_by_name(&self, name: &str) -> FalconResult<&Column> {
        let idx = self.schema.column_index(name)?;
        Ok(&self.columns[idx])
    }

    /// Packed-bit variant of filter_by_mask — 8× smaller mask, better cache behaviour.
    ///
    /// Drives gather directly from the bitmask without materialising an intermediate
    /// index list. The popcount is computed once and shared across every column.
    ///
    /// Serial on purpose. Gathering the columns in parallel was tried and measured
    /// nothing: on a pinned harness, the one case where it engaged
    /// (`filter_where/1M`) came out at +0.02%, p=0.98. It did, however, cost 6x on
    /// chunked scans that filter many small batches, because each batch then paid
    /// for a rayon spawn. A gather is memory-latency bound, so more threads
    /// contend for the same bus instead of adding throughput.
    pub fn filter_by_bitmask(&self, mask: &[u64]) -> FalconResult<RecordBatch> {
        let row_count = self.row_count();
        let selected: usize = mask.iter().map(|w| w.count_ones() as usize).sum();
        let filtered: FalconResult<Vec<Column>> = self.columns.iter()
            .map(|col| col.gather_by_bitmask(mask, row_count, selected))
            .collect();
        RecordBatch::new(self.schema.clone(), filtered?)
    }

    /// Filter rows by `mask`, materialising only the columns named in `keep`.
    ///
    /// This is projection pushdown: a query that reads two columns of a wide table
    /// should not pay to rebuild the other columns just to discard them at the
    /// projection step. Column order follows the schema, not `keep`.
    ///
    /// Names in `keep` that the batch does not have are ignored — the caller's
    /// needed-column set is a union across clauses and may mention columns from
    /// another table in a join.
    pub fn filter_by_bitmask_cols(
        &self,
        mask: &[u64],
        keep: &std::collections::HashSet<String>,
    ) -> FalconResult<RecordBatch> {
        let row_count = self.row_count();
        let selected: usize = mask.iter().map(|w| w.count_ones() as usize).sum();

        let mut fields  = Vec::with_capacity(keep.len());
        let mut columns = Vec::with_capacity(keep.len());
        for (i, field) in self.schema.fields.iter().enumerate() {
            if keep.contains(&field.name) {
                fields.push(field.clone());
                columns.push(self.column(i).gather_by_bitmask(mask, row_count, selected)?);
            }
        }
        RecordBatch::new(Schema::new(fields), columns)
    }

    /// Keep only the columns named in `keep`, without filtering rows.
    pub fn project_columns(
        &self,
        keep: &std::collections::HashSet<String>,
    ) -> FalconResult<RecordBatch> {
        let mut fields  = Vec::with_capacity(keep.len());
        let mut columns = Vec::with_capacity(keep.len());
        for (i, field) in self.schema.fields.iter().enumerate() {
            if keep.contains(&field.name) {
                fields.push(field.clone());
                columns.push(self.columns[i].clone());
            }
        }
        RecordBatch::from_shared(Schema::new(fields), columns)
    }

    pub fn filter_by_indices(&self, indices: &[usize]) -> FalconResult<RecordBatch> {
        let filtered: FalconResult<Vec<Column>> = self.columns.iter()
            .map(|col| col.filter_by_indices(indices))
            .collect();
        RecordBatch::new(self.schema.clone(), filtered?)
    }

    pub fn filter_by_mask(&self, mask: &[bool]) -> FalconResult<RecordBatch> {
        if mask.len() != self.row_count() {
            return Err(FalconError::InvalidQuery(format!(
                "mask length {} does not match row count {}",
                mask.len(),
                self.row_count(),
            )));
        }

        // The index list is built once and shared. `Column::filter_by_mask` derives
        // it from the mask itself, so calling that per column rebuilt it per column —
        // three times over 90k surviving rows for an ANTI join, which keeps most of
        // what it scans. That extra allocation is also why the same join measured
        // 2.2 ms alone and 3.6 ms inside a full suite: the cost moves with how tired
        // the allocator is, and a stable engine should not have that dependency.
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| keep.then_some(i))
            .collect();

        let filtered: FalconResult<Vec<Column>> = self
            .columns
            .iter()
            .map(|col| col.filter_by_indices(&indices))
            .collect();

        RecordBatch::new(self.schema.clone(), filtered?)
    }

    /// Select a subset of columns by name. Shares the underlying column data —
    /// projecting is a pointer copy, not a data copy.
    pub fn project(&self, names: &[&str]) -> FalconResult<RecordBatch> {
        let mut columns = Vec::with_capacity(names.len());
        let mut fields = Vec::with_capacity(names.len());

        for &name in names {
            let idx = self.schema.column_index(name)?;
            columns.push(Arc::clone(&self.columns[idx]));
            fields.push(self.schema.fields[idx].clone());
        }

        RecordBatch::from_shared(Schema::new(fields), columns)
    }

    pub fn slice(&self, offset: usize, length: usize) -> FalconResult<RecordBatch> {
        let sliced: FalconResult<Vec<Column>> = self
            .columns
            .iter()
            .map(|col| col.slice(offset, length))
            .collect();

        RecordBatch::new(self.schema.clone(), sliced?)
    }
}

// ─── Constraints ─────────────────────────────────────────────────────────────

/// The kind of a table-level integrity constraint.
///
/// NOT NULL is expressed directly on `Field.nullable`; only multi-row constraints
/// (UNIQUE, PRIMARY KEY) and expression constraints (CHECK) live here.
#[derive(Debug, Clone)]
pub enum ConstraintKind {
    /// All listed columns together form a primary key: must be unique and non-NULL.
    PrimaryKey(Vec<String>),
    /// The listed columns (combined) must be unique; NULLs do not violate uniqueness.
    Unique(Vec<String>),
    /// The SQL boolean expression must hold for every row.  Stored as the raw SQL
    /// text so the storage layer stays free of parser/expression-tree imports.
    Check(String),
    /// Every non-NULL value in `columns` must appear in `ref_columns` of `ref_table`.
    ForeignKey {
        columns:     Vec<String>,
        ref_table:   String,
        ref_columns: Vec<String>,
    },
}

#[derive(Debug, Clone)]
pub struct TableConstraint {
    pub name: Option<String>,
    pub kind: ConstraintKind,
}

// ─── Index metadata ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexType {
    BTree,
    Hash,
}

/// Metadata for a declared index.  The live data structure lives in
/// `Database::indices`; this just records what was declared (for SHOW INDEX,
/// DROP INDEX validation, and index rebuilds after DML).
#[derive(Debug, Clone)]
pub struct IndexDef {
    pub name:       String,
    pub column:     String,
    pub index_type: IndexType,
    pub unique:     bool,
}

// ─── Table ────────────────────────────────────────────────────────────────────

/// A Table holds the full dataset as a vector of RecordBatches.
/// Splitting into batches allows parallel processing and bounded memory usage.
pub struct Table {
    pub name:        String,
    pub schema:      Schema,
    pub batches:     Vec<RecordBatch>,
    /// Zone maps, one per batch and index-aligned with `batches`.
    ///
    /// Computed once when a batch is appended. Recomputing them per query would
    /// cost a full pass over the data — exactly what the pruning is meant to avoid.
    pub stats:       Vec<super::stats::BatchStats>,
    /// Integrity constraints attached to this table.
    pub constraints: Vec<TableConstraint>,
    /// Declared indices.  The live `IndexRegistry` is stored separately in
    /// `Database::indices` and rebuilt after every DML that touches this table.
    pub indices:     Vec<IndexDef>,
}

impl Table {
    pub const DEFAULT_BATCH_SIZE: usize = 65_536;

    pub fn new(name: impl Into<String>, schema: Schema) -> Self {
        Self {
            name: name.into(),
            schema,
            batches: Vec::new(),
            stats: Vec::new(),
            constraints: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn with_constraints(
        name: impl Into<String>,
        schema: Schema,
        constraints: Vec<TableConstraint>,
    ) -> Self {
        Self {
            name: name.into(),
            schema,
            batches: Vec::new(),
            stats: Vec::new(),
            constraints,
            indices: Vec::new(),
        }
    }

    pub fn append_batch(&mut self, batch: RecordBatch) -> FalconResult<()> {
        if batch.schema != self.schema {
            return Err(FalconError::InvalidQuery(
                "batch schema does not match table schema".to_string(),
            ));
        }
        self.stats.push(super::stats::BatchStats::compute(&batch));
        self.batches.push(batch);
        Ok(())
    }

    pub fn row_count(&self) -> usize {
        self.batches.iter().map(|b| b.row_count()).sum()
    }

    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Merge fragmented batches back into full-sized ones, returning the number
    /// of batches removed.
    ///
    /// DELETE and UPDATE rebuild only the batches they touch, so a table that has
    /// been mutated a lot ends up as many small batches. That costs on every
    /// subsequent scan: per-batch overhead is paid regardless of how few rows the
    /// batch holds, and zone maps get less selective as batches shrink.
    ///
    /// Batches already at or above `DEFAULT_BATCH_SIZE` are passed through
    /// untouched — rewriting them would copy every value to no benefit. Runs of
    /// smaller batches are concatenated up to that size.
    pub fn vacuum(&mut self) -> FalconResult<usize> {
        let before = self.batches.len();
        if before <= 1 {
            return Ok(0);
        }

        let old = std::mem::take(&mut self.batches);
        self.stats.clear();

        // Accumulate small batches until they reach the target size, then flush.
        let mut pending: Vec<RecordBatch> = Vec::new();
        let mut pending_rows = 0usize;

        // Declared here so both the loop and the tail can flush.
        fn flush(
            table:   &mut Table,
            pending: &mut Vec<RecordBatch>,
            rows:    &mut usize,
        ) -> FalconResult<()> {
            match pending.len() {
                0 => {}
                // A single pending batch needs no concatenation.
                1 => table.append_batch(pending.pop().unwrap())?,
                _ => {
                    let merged = RecordBatch::concat(pending)?;
                    pending.clear();
                    table.append_batch(merged)?;
                }
            }
            *rows = 0;
            Ok(())
        }

        for batch in old {
            let n = batch.row_count();
            if n == 0 {
                continue; // empty batches carry no rows and no information
            }
            if n >= Self::DEFAULT_BATCH_SIZE {
                // Already full: flush what is pending, then pass it through.
                flush(self, &mut pending, &mut pending_rows)?;
                self.append_batch(batch)?;
                continue;
            }
            if pending_rows + n > Self::DEFAULT_BATCH_SIZE {
                flush(self, &mut pending, &mut pending_rows)?;
            }
            pending_rows += n;
            pending.push(batch);
        }
        flush(self, &mut pending, &mut pending_rows)?;

        Ok(before - self.batches.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        column::ColumnBuilder,
        schema::Field,
        types::DataType,
    };

    fn make_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::required("id", DataType::Int64),
            Field::nullable("name", DataType::Utf8),
        ]);
        let columns = vec![
            ColumnBuilder::int64(vec![Some(1), Some(2), Some(3)]),
            ColumnBuilder::utf8_from_str(vec![Some("alice"), Some("bob"), None]),
        ];
        RecordBatch::new(schema, columns).unwrap()
    }

    #[test]
    fn batch_row_count() {
        let batch = make_batch();
        assert_eq!(batch.row_count(), 3);
    }

    #[test]
    fn batch_filter() {
        let batch = make_batch();
        let mask = vec![true, false, true];
        let filtered = batch.filter_by_mask(&mask).unwrap();
        assert_eq!(filtered.row_count(), 2);
    }

    #[test]
    fn batch_project() {
        let batch = make_batch();
        let projected = batch.project(&["name"]).unwrap();
        assert_eq!(projected.column_count(), 1);
        assert_eq!(projected.schema.fields[0].name, "name");
    }

    #[test]
    fn batch_slice() {
        let batch = make_batch();
        let sliced = batch.slice(1, 2).unwrap();
        assert_eq!(sliced.row_count(), 2);
    }

    #[test]
    fn mismatched_schema_rejects_batch() {
        let schema = Schema::new(vec![Field::required("x", DataType::Int32)]);
        let col = ColumnBuilder::int32(vec![Some(1), Some(2)]);
        let batch = RecordBatch::new(schema, vec![col]).unwrap();

        let mut table = Table::new("t", Schema::new(vec![
            Field::required("id", DataType::Int64),
        ]));
        assert!(table.append_batch(batch).is_err());
    }
}
