use rayon::prelude::*;

use crate::storage::{
    schema::{Field, Schema},
    table::{RecordBatch, Table},
    types::{FalconResult, ScalarValue},
};

use super::expr::Expr;

// в”Ђв”Ђв”Ђ Projection в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// What columns to return.
#[derive(Clone, Debug)]
pub enum Projection {
    /// SELECT *
    All,
    /// SELECT col1, col2, expr AS alias, ...
    Columns(Vec<SelectExpr>),
}

#[derive(Clone, Debug)]
pub struct SelectExpr {
    pub expr:  Expr,
    pub alias: String,
}

impl SelectExpr {
    pub fn column(name: &str) -> Self {
        Self { expr: Expr::Column(name.into()), alias: name.into() }
    }

    pub fn named(expr: Expr, alias: &str) -> Self {
        Self { expr, alias: alias.into() }
    }
}

// в”Ђв”Ђв”Ђ SelectQuery в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// A fully structured SELECT query.
#[derive(Clone, Debug)]
pub struct SelectQuery {
    pub projection: Projection,
    pub filter:     Option<Expr>,
    pub limit:      Option<usize>,
    pub offset:     usize,
    pub distinct:   bool,
    /// Projection pushdown: when set, the scan materialises only these columns.
    ///
    /// `None` means "every column", which is always correct. A caller that sets
    /// this promises the set covers every column any later stage will read —
    /// GROUP BY, HAVING, ORDER BY and window clauses included, not just the
    /// SELECT list. Getting that wrong surfaces as a missing-column error, not
    /// as silently wrong data.
    pub scan_columns: Option<std::collections::HashSet<String>>,
}

impl SelectQuery {
    pub fn scan_all() -> Self {
        Self {
            projection:   Projection::All,
            filter:       None,
            limit:        None,
            offset:       0,
            distinct:     false,
            scan_columns: None,
        }
    }

    /// Restrict the scan to `cols` (projection pushdown).
    pub fn with_scan_columns(mut self, cols: std::collections::HashSet<String>) -> Self {
        self.scan_columns = Some(cols);
        self
    }

    pub fn with_filter(mut self, expr: Expr) -> Self {
        self.filter = Some(expr);
        self
    }

    pub fn with_limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    pub fn with_offset(mut self, n: usize) -> Self {
        self.offset = n;
        self
    }

    pub fn with_columns(mut self, cols: Vec<SelectExpr>) -> Self {
        self.projection = Projection::Columns(cols);
        self
    }

    pub fn with_distinct(mut self, d: bool) -> Self {
        self.distinct = d;
        self
    }
}

// в”Ђв”Ђв”Ђ Executor в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// Execute a SELECT query over a table and return a flat list of result batches.
pub fn execute_select(table: &Table, query: &SelectQuery) -> FalconResult<Vec<RecordBatch>> {
    // Each batch is processed independently вЂ” parallelism here is safe.
    let processed: FalconResult<Vec<Option<RecordBatch>>> = table
        .batches
        .par_iter()
        .enumerate()
        .map(|(i, batch)| {
            // Zone-map pruning: if the batch's precomputed bounds prove no row can
            // match, skip it without reading a single value.
            if let Some(filter) = &query.filter {
                if table.stats.get(i).is_some_and(|s| s.can_skip(filter)) {
                    return Ok(None);
                }
            }
            process_batch(batch, query)
        })
        .collect();

    // Filter out empty batches, then apply global offset + limit.
    let mut result: Vec<RecordBatch> = processed?
        .into_iter()
        .flatten()
        .filter(|b| b.row_count() > 0)
        .collect();

    if query.distinct {
        result = dedup_batches(result);
    }

    if query.offset > 0 || query.limit.is_some() {
        result = apply_offset_limit(result, query.offset, query.limit);
    }

    Ok(result)
}

/// Hash one row into a single `u64`, without allocating.
///
/// The previous key was a `Vec<String>` built with `to_string()` per cell, which
/// cost one heap allocation per cell — three million of them for a million-row,
/// three-column DISTINCT. Hashing the raw typed values costs none.
///
/// Hashing rather than storing the key means two distinct rows could collide and
/// one would be dropped as a duplicate. `row_eq` re-checks equality on collision,
/// so the hash only decides *where* to look, never what is equal.
pub(crate) fn row_hash(batch: &RecordBatch, row: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    use crate::storage::column::Column;

    let mut h = super::hash::FxHasher::default();
    for ci in 0..batch.column_count() {
        // Discriminate the column position so that shifting a value between
        // columns changes the hash.
        ci.hash(&mut h);
        match batch.column(ci) {
            Column::Boolean(c)   => c.get(row).hash(&mut h),
            Column::Int32(c)     => c.get(row).hash(&mut h),
            Column::Int64(c)     => c.get(row).hash(&mut h),
            Column::Date32(c)    => c.get(row).hash(&mut h),
            Column::Timestamp(c) => c.get(row).hash(&mut h),
            Column::Utf8(c)      => c.get_str(row).hash(&mut h),
            // Floats have no `Hash`: NaN != NaN and 0.0 == -0.0, so hash the bit
            // pattern with those two cases normalised, matching `row_eq`.
            Column::Float32(c) => c.get(row).map(norm_f32).hash(&mut h),
            Column::Float64(c) => c.get(row).map(norm_f64).hash(&mut h),
            Column::Decimal128(c) => c.inner.get(row).hash(&mut h),
        }
    }
    h.finish()
}

#[inline]
fn norm_f32(v: f32) -> u32 {
    if v.is_nan() { f32::NAN.to_bits() } else if v == 0.0 { 0 } else { v.to_bits() }
}

#[inline]
fn norm_f64(v: f64) -> u64 {
    if v.is_nan() { f64::NAN.to_bits() } else if v == 0.0 { 0 } else { v.to_bits() }
}

/// Are two rows equal across every column? Used to settle hash collisions.
pub(crate) fn row_eq(a: &RecordBatch, ra: usize, b: &RecordBatch, rb: usize) -> bool {
    use crate::storage::column::Column;

    if a.column_count() != b.column_count() {
        return false;
    }
    for ci in 0..a.column_count() {
        let same = match (a.column(ci), b.column(ci)) {
            (Column::Boolean(x),   Column::Boolean(y))   => x.get(ra) == y.get(rb),
            (Column::Int32(x),     Column::Int32(y))     => x.get(ra) == y.get(rb),
            (Column::Int64(x),     Column::Int64(y))     => x.get(ra) == y.get(rb),
            (Column::Date32(x),    Column::Date32(y))    => x.get(ra) == y.get(rb),
            (Column::Timestamp(x), Column::Timestamp(y)) => x.get(ra) == y.get(rb),
            (Column::Utf8(x),      Column::Utf8(y))      => x.get_str(ra) == y.get_str(rb),
            (Column::Float32(x),   Column::Float32(y))   =>
                x.get(ra).map(norm_f32) == y.get(rb).map(norm_f32),
            (Column::Float64(x),   Column::Float64(y))   =>
                x.get(ra).map(norm_f64) == y.get(rb).map(norm_f64),
            (Column::Decimal128(x), Column::Decimal128(y)) => x.inner.get(ra) == y.inner.get(rb),
            // Different column types cannot hold equal values.
            _ => false,
        };
        if !same {
            return false;
        }
    }
    true
}

/// Remove duplicate rows across batches (for SELECT DISTINCT and UNION).
pub fn dedup_batches(batches: Vec<RecordBatch>) -> Vec<RecordBatch> {
    use super::hash::FastMap;

    // hash -> positions in the *input* batches that hashed there. Recording input
    // positions rather than output ones matters: a row has to be comparable
    // against earlier rows of the batch currently being scanned, which has not
    // been filtered or emitted yet.
    let mut seen: FastMap<u64, Vec<(usize, usize)>> = FastMap::default();
    let mut keeps: Vec<Vec<bool>> = Vec::with_capacity(batches.len());

    for (bi, batch) in batches.iter().enumerate() {
        let mut keep = Vec::with_capacity(batch.row_count());
        for row in 0..batch.row_count() {
            let h = row_hash(batch, row);
            let bucket = seen.entry(h).or_default();
            // The hash only narrows the search; equality decides.
            let duplicate = bucket
                .iter()
                .any(|&(pb, pr)| row_eq(&batches[pb], pr, batch, row));

            if duplicate {
                keep.push(false);
            } else {
                bucket.push((bi, row));
                keep.push(true);
            }
        }
        keeps.push(keep);
    }

    let mut result = Vec::with_capacity(batches.len());
    for (batch, keep) in batches.iter().zip(keeps) {
        if keep.iter().any(|&k| k) {
            if let Ok(filtered) = batch.filter_by_mask(&keep) {
                result.push(filtered);
            }
        }
    }
    result
}

fn process_batch(batch: &RecordBatch, query: &SelectQuery) -> FalconResult<Option<RecordBatch>> {
    // 1. Filter (WHERE) вЂ” packed bitmask: 8Г— smaller than Vec<bool>, better cache behaviour.
    //
    // With projection pushdown active, only the needed columns are rebuilt. The
    // mask is still computed against the full batch, since the filter may read a
    // column the query does not otherwise return.
    let filtered = match (&query.filter, &query.scan_columns) {
        (Some(filter), Some(keep)) => {
            let mask = filter.eval_mask_bits(batch)?;
            batch.filter_by_bitmask_cols(&mask, keep)?
        }
        (Some(filter), None) => {
            let mask = filter.eval_mask_bits(batch)?;
            batch.filter_by_bitmask(&mask)?
        }
        (None, Some(keep)) => batch.project_columns(keep)?,
        (None, None) => batch.clone(),
    };

    if filtered.row_count() == 0 {
        return Ok(None);
    }

    // 2. Project (SELECT columns / expressions).
    let projected = match &query.projection {
        Projection::All => filtered,
        Projection::Columns(exprs) => project_batch(&filtered, exprs)?,
    };

    Ok(Some(projected))
}

pub fn project_batch(batch: &RecordBatch, exprs: &[SelectExpr]) -> FalconResult<RecordBatch> {
    use crate::storage::column::{BooleanColumn, Column, PrimitiveColumn, Utf8Column};
    use crate::storage::types::DataType;

    let mut columns  = Vec::with_capacity(exprs.len());
    let mut fields   = Vec::with_capacity(exprs.len());
    let row_count    = batch.row_count();

    for sel in exprs {
        // Fast path: a bare column reference is already the column we want.
        //
        // The general path below evaluates the expression into a `Vec<ScalarValue>`
        // and rebuilds a typed column from it. `ScalarValue` is 48 bytes, so for a
        // plain `SELECT a, b` that boxes and unboxes every cell — 48 MB of
        // allocation per million rows — to reproduce data the batch already holds.
        // Columns live behind `Arc`, so the same result costs a refcount bump.
        if let Expr::Column(name) = &sel.expr {
            let idx = batch.schema.column_index(name)?;
            let mut field = batch.schema.fields[idx].clone();
            // The projection may rename the column: `SELECT a AS b`.
            field.name = sel.alias.clone();
            fields.push(field);
            columns.push(std::sync::Arc::clone(&batch.columns[idx]));
            continue;
        }

        let values = sel.expr.eval_batch(batch)?;

        // Determine result type from first non-null value.
        let sample = values.iter().find(|v| !v.is_null());
        let dt = sample.map(|v| v.data_type()).unwrap_or(DataType::Null);

        let col = match dt {
            DataType::Boolean => Column::Boolean(BooleanColumn::new(
                values.iter().map(|v| v.as_bool()).collect(),
            )),
            DataType::Int32 | DataType::Int64 => Column::Int64(PrimitiveColumn::new(
                values.iter().map(|v| v.as_i64()).collect(),
            )),
            DataType::Float32 | DataType::Float64 => Column::Float64(PrimitiveColumn::new(
                values.iter().map(|v| v.as_f64()).collect(),
            )),
            DataType::Utf8 => Column::Utf8(Utf8Column::new(
                values.into_iter().map(|v| match v {
                    ScalarValue::Utf8(s) => s,
                    _ => None,
                }).collect(),
            )),
            DataType::Date32 => Column::Date32(PrimitiveColumn::new(
                values.iter().map(|v| match v {
                    ScalarValue::Date32(d) => *d,
                    // An integer in a date-typed projection is a day offset,
                    // which is how date arithmetic surfaces here.
                    ScalarValue::Int64(Some(n)) => Some(*n as i32),
                    _ => None,
                }).collect(),
            )),
            DataType::Timestamp => Column::Timestamp(PrimitiveColumn::new(
                values.iter().map(|v| match v {
                    ScalarValue::Timestamp(t)   => *t,
                    ScalarValue::Int64(Some(n)) => Some(*n),
                    _ => None,
                }).collect(),
            )),
            DataType::Decimal128(p, s) => Column::Decimal128(
                crate::storage::column::Decimal128Column::new(
                    values.iter().map(|v| match v {
                        ScalarValue::Decimal128(n, _, _) => *n,
                        _ => None,
                    }).collect(),
                    p,
                    s,
                ),
            ),
            // Every value was NULL, so there is no type to infer. An all-NULL Int64
            // column is the neutral choice: it renders as NULL whatever the caller
            // expected.
            //
            // This match is deliberately exhaustive — no catch-all arm. A new
            // DataType must fail to compile here rather than fall through to an
            // all-NULL column, which is how projecting a DATE silently returned
            // NULL before these arms existed.
            DataType::Null => Column::Int64(PrimitiveColumn::new(vec![None; row_count])),
        };

        let field = Field::nullable(sel.alias.as_str(), col.data_type());
        columns.push(std::sync::Arc::new(col));
        fields.push(field);
    }

    // `from_shared`, not `new`: the fast path above hands back columns that are
    // already shared, and re-wrapping them would undo the point of it.
    RecordBatch::from_shared(Schema::new(fields), columns)
}

pub fn apply_offset_limit(
    batches: Vec<RecordBatch>,
    offset:  usize,
    limit:   Option<usize>,
) -> Vec<RecordBatch> {
    let mut rows_skipped = 0usize;
    let mut rows_taken   = 0usize;
    let max_rows         = limit.unwrap_or(usize::MAX);
    let mut result       = Vec::new();

    for batch in batches {
        if rows_taken >= max_rows { break; }

        let n = batch.row_count();

        if rows_skipped + n <= offset {
            rows_skipped += n;
            continue;
        }

        let start = if rows_skipped < offset { offset - rows_skipped } else { 0 };
        rows_skipped = offset;

        let available = n - start;
        let take      = available.min(max_rows - rows_taken);

        if take == 0 { break; }

        if let Ok(sliced) = batch.slice(start, take) {
            rows_taken += take;
            result.push(sliced);
        }
    }

    result
}

// в”Ђв”Ђв”Ђ Result helpers в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// Flatten result batches into a pretty-printed table string.
pub fn display_results(batches: &[RecordBatch]) -> String {
    if batches.is_empty() {
        return "(0 rows)\n".into();
    }

    let schema     = &batches[0].schema;
    let col_names  = schema.column_names();
    let col_count  = col_names.len();

    // Collect all rows as strings.
    let mut rows: Vec<Vec<String>> = Vec::new();
    for batch in batches {
        for row_idx in 0..batch.row_count() {
            let row: Vec<String> = (0..col_count)
                .map(|col_idx| {
                    batch.column(col_idx).get(row_idx)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|_| "ERR".into())
                })
                .collect();
            rows.push(row);
        }
    }

    // Compute column widths.
    let mut widths: Vec<usize> = col_names.iter().map(|n| n.len()).collect();
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    let separator: String = widths.iter()
        .map(|&w| "-".repeat(w + 2))
        .collect::<Vec<_>>()
        .join("+");
    let separator = format!("+{separator}+");

    let mut out = String::new();
    out.push_str(&separator);
    out.push('\n');

    // Header.
    let header: String = col_names.iter().zip(widths.iter())
        .map(|(name, &w)| format!(" {name:<w$} "))
        .collect::<Vec<_>>()
        .join("|");
    out.push_str(&format!("|{header}|\n"));
    out.push_str(&separator);
    out.push('\n');

    // Rows.
    for row in &rows {
        let line: String = row.iter().zip(widths.iter())
            .map(|(cell, &w)| format!(" {cell:<w$} "))
            .collect::<Vec<_>>()
            .join("|");
        out.push_str(&format!("|{line}|\n"));
    }

    out.push_str(&separator);
    out.push('\n');
    out.push_str(&format!("({} rows)\n", rows.len()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        query::expr::{BinaryOp, Expr},
        storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::{RecordBatch, Table},
            types::{DataType, ScalarValue},
        },
    };

    fn make_table() -> Table {
        let schema = Schema::new(vec![
            Field::required("id",    DataType::Int64),
            Field::nullable("name",  DataType::Utf8),
            Field::nullable("score", DataType::Float64),
        ]);

        let batch = RecordBatch::new(schema.clone(), vec![
            ColumnBuilder::int64(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]),
            ColumnBuilder::utf8_from_str(vec![
                Some("alice"), Some("bob"), Some("carol"), Some("dave"), Some("eve"),
            ]),
            ColumnBuilder::float64(vec![
                Some(9.5), Some(8.0), Some(7.2), Some(6.8), Some(10.0),
            ]),
        ]).unwrap();

        let mut table = Table::new("users", schema);
        table.append_batch(batch).unwrap();
        table
    }

    #[test]
    fn select_all() {
        let table   = make_table();
        let query   = SelectQuery::scan_all();
        let results = execute_select(&table, &query).unwrap();
        let total: usize = results.iter().map(|b| b.row_count()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn select_with_filter() {
        let table = make_table();
        let query = SelectQuery::scan_all().with_filter(Expr::BinaryOp {
            left:  Box::new(Expr::Column("score".into())),
            op:    BinaryOp::Gt,
            right: Box::new(Expr::Literal(ScalarValue::Float64(Some(8.0)))),
        });
        let results = execute_select(&table, &query).unwrap();
        let total: usize = results.iter().map(|b| b.row_count()).sum();
        assert_eq!(total, 2); // alice (9.5) and eve (10.0)
    }

    #[test]
    fn select_with_limit() {
        let table   = make_table();
        let query   = SelectQuery::scan_all().with_limit(2);
        let results = execute_select(&table, &query).unwrap();
        let total: usize = results.iter().map(|b| b.row_count()).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn select_with_offset() {
        let table   = make_table();
        let query   = SelectQuery::scan_all().with_offset(3);
        let results = execute_select(&table, &query).unwrap();
        let total: usize = results.iter().map(|b| b.row_count()).sum();
        assert_eq!(total, 2); // dave and eve
    }

    #[test]
    fn select_projection() {
        let table = make_table();
        let query = SelectQuery::scan_all().with_columns(vec![
            SelectExpr::column("id"),
            SelectExpr::column("name"),
        ]);
        let results = execute_select(&table, &query).unwrap();
        assert_eq!(results[0].column_count(), 2);
        assert_eq!(results[0].schema.fields[0].name, "id");
        assert_eq!(results[0].schema.fields[1].name, "name");
    }

    #[test]
    fn display_results_format() {
        let table   = make_table();
        let query   = SelectQuery::scan_all().with_limit(2);
        let results = execute_select(&table, &query).unwrap();
        let output  = display_results(&results);
        assert!(output.contains("alice"));
        assert!(output.contains("bob"));
        assert!(output.contains("(2 rows)"));
    }
}
