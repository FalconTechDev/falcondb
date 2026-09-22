//! INSERT / UPDATE / DELETE.
//!
//! Tables are updated copy-on-write: a mutation builds a new `Table` and the
//! `Database` swaps it in. That is cheap because `RecordBatch` holds its columns
//! behind `Arc` — batches the statement does not touch are carried over as pointer
//! copies, and only the affected batches are rebuilt.

use std::sync::Arc;

use crate::storage::{
    column::{BooleanColumn, Column, PrimitiveColumn, Utf8Column},
    schema::Schema,
    table::{RecordBatch, Table},
    types::{DataType, FalconError, FalconResult, ScalarValue},
};

use super::{expr::Expr, select::SelectExpr};

// ─── Statements ──────────────────────────────────────────────────────────────

/// Source data for an INSERT statement.
#[derive(Debug, Clone)]
pub enum InsertSource {
    /// `INSERT INTO t VALUES (…), (…)`
    Values(Vec<Vec<Expr>>),
    /// `INSERT INTO t SELECT …` — stored as an opaque parsed query so the
    /// execution engine (which owns the table snapshot) can run it.
    Select(Box<crate::query::sql::ParsedQuery>),
}

#[derive(Debug, Clone)]
pub struct InsertStmt {
    pub table:     String,
    /// Target columns. Empty means "every column, in schema order".
    pub columns:   Vec<String>,
    pub source:    InsertSource,
    /// `RETURNING` expressions; empty if not present.
    pub returning: Vec<SelectExpr>,
}

#[derive(Debug, Clone)]
pub struct UpdateStmt {
    pub table:       String,
    pub assignments: Vec<(String, Expr)>,
    pub filter:      Option<Expr>,
    /// `RETURNING` expressions; empty if not present.
    pub returning:   Vec<SelectExpr>,
}

#[derive(Debug, Clone)]
pub struct DeleteStmt {
    pub table:     String,
    pub filter:    Option<Expr>,
    /// `RETURNING` expressions; empty if not present.
    pub returning: Vec<SelectExpr>,
}

// ─── INSERT ──────────────────────────────────────────────────────────────────

/// Append rows to `table`.
///
/// Returns the new table, the number of rows inserted, and (when RETURNING is
/// present) the inserted batch projected through the RETURNING expressions.
///
/// Only `InsertSource::Values` is handled here.  `InsertSource::Select` must be
/// resolved to a `Values` insert by the caller before this is called.
pub fn execute_insert(
    table: &Table,
    stmt:  &InsertStmt,
) -> FalconResult<(Table, usize, Option<RecordBatch>)> {
    let rows = match &stmt.source {
        InsertSource::Values(rows) => rows,
        InsertSource::Select(_) => {
            return Err(FalconError::InvalidQuery(
                "INSERT … SELECT must be resolved before execute_insert".into(),
            ));
        }
    };

    if rows.is_empty() {
        return Ok((clone_table(table), 0, None));
    }

    let schema = &table.schema;

    // Resolve which schema column each supplied value maps to.
    let target: Vec<usize> = if stmt.columns.is_empty() {
        (0..schema.field_count()).collect()
    } else {
        stmt.columns
            .iter()
            .map(|name| schema.column_index(name))
            .collect::<FalconResult<_>>()?
    };

    for (row_idx, row) in rows.iter().enumerate() {
        if row.len() != target.len() {
            return Err(FalconError::InvalidQuery(format!(
                "INSERT row {} has {} values but {} columns were named",
                row_idx + 1,
                row.len(),
                target.len(),
            )));
        }
    }

    // Evaluate every literal into a per-column value list. Columns the statement
    // does not name are filled with NULL.
    let n_new = rows.len();
    let mut per_column: Vec<Vec<ScalarValue>> =
        vec![vec![ScalarValue::Null; n_new]; schema.field_count()];

    for (row_idx, row) in rows.iter().enumerate() {
        for (slot, expr) in target.iter().zip(row.iter()) {
            per_column[*slot][row_idx] = eval_constant(expr)?;
        }
    }

    // Build one new batch and append it — existing batches are untouched.
    let new_columns: FalconResult<Vec<Column>> = per_column
        .iter()
        .enumerate()
        .map(|(i, vals)| scalars_to_column(vals, &schema.fields[i].data_type))
        .collect();

    let batch = RecordBatch::new(schema.clone(), new_columns?)?;

    let returning = if stmt.returning.is_empty() {
        None
    } else if is_wildcard_returning(&stmt.returning) {
        Some(batch.clone())
    } else {
        use crate::query::select::project_batch;
        Some(project_batch(&batch, &stmt.returning)?)
    };

    let mut out = clone_table(table);
    out.append_batch(batch)?;
    Ok((out, n_new, returning))
}

// ─── DELETE ──────────────────────────────────────────────────────────────────

/// Remove rows matching `stmt.filter`.
///
/// Returns the new table, rows removed, and (when RETURNING is present) the
/// deleted rows projected through the RETURNING expressions.
/// A statement with no filter empties the table.
pub fn execute_delete(
    table: &Table,
    stmt:  &DeleteStmt,
) -> FalconResult<(Table, usize, Option<RecordBatch>)> {
    use crate::query::select::project_batch;

    let mut out = clone_table_empty(table);
    let mut removed = 0usize;
    let mut deleted_batches: Vec<RecordBatch> = Vec::new();
    let need_returning = !stmt.returning.is_empty();

    for batch in &table.batches {
        match &stmt.filter {
            None => {
                removed += batch.row_count();
                if need_returning {
                    deleted_batches.push(batch.clone());
                }
            }
            Some(filter) => {
                let mask = filter.eval_mask_bits(batch)?;
                let keep = invert_mask(&mask, batch.row_count());

                if need_returning {
                    // Collect the deleted rows (the ones we do NOT keep).
                    let deleted = batch.filter_by_bitmask(&mask)?;
                    if deleted.row_count() > 0 {
                        deleted_batches.push(deleted);
                    }
                }

                let kept = batch.filter_by_bitmask(&keep)?;
                removed += batch.row_count() - kept.row_count();
                if kept.row_count() > 0 {
                    out.append_batch(kept)?;
                }
            }
        }
    }

    let returning = if need_returning && !deleted_batches.is_empty() {
        let merged = merge_batches_simple(&deleted_batches)?;
        if is_wildcard_returning(&stmt.returning) {
            Some(merged)
        } else {
            Some(project_batch(&merged, &stmt.returning)?)
        }
    } else {
        None
    };

    Ok((out, removed, returning))
}

// ─── UPDATE ──────────────────────────────────────────────────────────────────

/// Apply `stmt.assignments` to rows matching `stmt.filter`.
///
/// Returns the new table, rows changed, and (when RETURNING is present) the
/// updated rows (post-update values) projected through the RETURNING expressions.
pub fn execute_update(
    table: &Table,
    stmt:  &UpdateStmt,
) -> FalconResult<(Table, usize, Option<RecordBatch>)> {
    use crate::query::select::project_batch;

    let schema = &table.schema;

    // Resolve assignment targets up front so a bad column name fails before any work.
    let targets: Vec<(usize, &Expr)> = stmt
        .assignments
        .iter()
        .map(|(name, expr)| Ok((schema.column_index(name)?, expr)))
        .collect::<FalconResult<_>>()?;

    let mut out = clone_table_empty(table);
    let mut changed = 0usize;
    let mut updated_batches: Vec<RecordBatch> = Vec::new();
    let need_returning = !stmt.returning.is_empty();

    for batch in &table.batches {
        let n_rows = batch.row_count();

        // Which rows does this statement touch?
        let selected: Vec<bool> = match &stmt.filter {
            None => vec![true; n_rows],
            Some(filter) => {
                let mask = filter.eval_mask_bits(batch)?;
                (0..n_rows).map(|i| (mask[i >> 6] >> (i & 63)) & 1 == 1).collect()
            }
        };

        let hits = selected.iter().filter(|&&s| s).count();
        if hits == 0 {
            // Nothing to do — carry the batch over without rebuilding it.
            out.append_batch(batch.clone())?;
            continue;
        }
        changed += hits;

        // Evaluate each assignment across the whole batch, then splice in the new
        // values only at the selected rows.
        let mut columns: Vec<Arc<Column>> = batch.columns.clone();
        for (col_idx, expr) in &targets {
            // Fully typed path: the assignment evaluates straight into f64s and is
            // spliced into a dense copy, so neither side boxes a value per row.
            if schema.fields[*col_idx].data_type == DataType::Float64 {
                if let Some(vals) = expr.eval_dense_f64(batch) {
                    if let Column::Float64(src) = batch.column(*col_idx) {
                        if src.null_count() == 0 {
                            let mut dense = src.dense().to_vec();
                            for (r, keep) in selected.iter().enumerate() {
                                if *keep {
                                    dense[r] = vals[r];
                                }
                            }
                            columns[*col_idx] = Arc::new(Column::Float64(
                                crate::storage::column::PrimitiveColumn::from_non_null(dense),
                            ));
                            continue;
                        }
                    }
                }
            }

            let dt = &schema.fields[*col_idx].data_type;
            let src = batch.column(*col_idx);

            // A literal is the common case and does not need materialising once
            // per row before it can be written once per selected row.
            if let crate::query::expr::Expr::Literal(v) = expr {
                if let Some(c) = splice_constant(src, v, &selected) {
                    columns[*col_idx] = Arc::new(c);
                    continue;
                }
            }

            let computed = expr.eval_batch(batch)?;
            columns[*col_idx] = Arc::new(match splice_typed(src, &computed, &selected) {
                Some(c) => c,
                None => {
                    let mut values: Vec<ScalarValue> = (0..n_rows)
                        .map(|r| src.get(r))
                        .collect::<FalconResult<_>>()?;
                    for (r, keep) in selected.iter().enumerate() {
                        if *keep {
                            values[r] = computed[r].clone();
                        }
                    }
                    scalars_to_column(&values, dt)?
                }
            });
        }

        let new_batch = RecordBatch::from_shared(schema.clone(), columns)?;

        if need_returning {
            // Collect only the rows that were updated.
            let mask: Vec<u64> = selected.chunks(64).map(|chunk| {
                chunk.iter().enumerate().fold(0u64, |acc, (i, &sel)| {
                    if sel { acc | (1u64 << i) } else { acc }
                })
            }).collect();
            let updated_rows = new_batch.filter_by_bitmask(&mask)?;
            if updated_rows.row_count() > 0 {
                updated_batches.push(updated_rows);
            }
        }

        out.append_batch(new_batch)?;
    }

    let returning = if need_returning && !updated_batches.is_empty() {
        let merged = merge_batches_simple(&updated_batches)?;
        if is_wildcard_returning(&stmt.returning) {
            Some(merged)
        } else {
            Some(project_batch(&merged, &stmt.returning)?)
        }
    } else {
        None
    };

    Ok((out, changed, returning))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Create an empty table preserving schema, constraints, and index metadata.
fn clone_table_empty(table: &Table) -> Table {
    Table {
        name:        table.name.clone(),
        schema:      table.schema.clone(),
        batches:     Vec::new(),
        stats:       Vec::new(),
        constraints: table.constraints.clone(),
        indices:     table.indices.clone(),
    }
}

/// Copy a table's structure. Batches are `Arc`-shared, so this copies pointers only.
fn clone_table(table: &Table) -> Table {
    Table {
        name:        table.name.clone(),
        schema:      table.schema.clone(),
        batches:     table.batches.clone(),
        stats:       table.stats.clone(),
        constraints: table.constraints.clone(),
        indices:     table.indices.clone(),
    }
}

/// Public wrapper used by callers that hold an `Arc<Table>` (e.g. `run_alter_table`).
pub fn clone_table_arc(table: &Table) -> Table {
    clone_table(table)
}

/// Flip every bit up to `row_count`, leaving padding bits clear.
fn invert_mask(mask: &[u64], row_count: usize) -> Vec<u64> {
    let mut out: Vec<u64> = mask.iter().map(|w| !w).collect();
    // Clear bits past the end so they cannot select phantom rows.
    let tail = row_count & 63;
    if tail != 0 {
        if let Some(last) = out.get_mut(row_count >> 6) {
            *last &= (1u64 << tail) - 1;
        }
    }
    for word in out.iter_mut().skip((row_count + 63) >> 6) {
        *word = 0;
    }
    out
}

/// INSERT values must be constants — a column reference has no row to read from.
fn eval_constant(expr: &Expr) -> FalconResult<ScalarValue> {
    match expr {
        Expr::Literal(v) => Ok(v.clone()),
        other => Err(FalconError::InvalidQuery(format!(
            "INSERT values must be literals, got: {other:?}"
        ))),
    }
}

/// Materialise scalars into a column of the schema's declared type.
///
/// Numeric widening and narrowing are allowed (SQL does the same when writing a
/// literal into a typed column), but a value that cannot be represented at all is a
/// hard error. Returning NULL instead would silently destroy data — an earlier
/// version of this function did exactly that, turning `SET score = score * 2` into
/// a column of NULLs because the multiplication produced `Float64` and the column
/// was declared `Int64`.
fn scalars_to_column(vals: &[ScalarValue], dt: &DataType) -> FalconResult<Column> {
    /// Map each value, treating NULL as NULL and failing on anything unrepresentable.
    fn convert<T>(
        vals: &[ScalarValue],
        dt:   &DataType,
        f:    impl Fn(&ScalarValue) -> Option<T>,
    ) -> FalconResult<Vec<Option<T>>> {
        vals.iter()
            .map(|v| {
                if v.is_null() {
                    return Ok(None);
                }
                f(v).map(Some).ok_or_else(|| {
                    FalconError::InvalidQuery(format!(
                        "value {v} cannot be stored in a column of type {dt}"
                    ))
                })
            })
            .collect()
    }

    Ok(match dt {
        // Booleans stay strict: coercing 1/0 here is how integer data ends up
        // silently flattened into flags.
        DataType::Boolean => Column::Boolean(BooleanColumn::new(convert(
            vals, dt, |v| v.as_bool(),
        )?)),

        DataType::Int32 => Column::Int32(PrimitiveColumn::new(convert(
            vals, dt, |v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)).map(|n| n as i32),
        )?)),

        DataType::Int64 => Column::Int64(PrimitiveColumn::new(convert(
            vals, dt, |v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)),
        )?)),

        DataType::Float32 => Column::Float32(PrimitiveColumn::new(convert(
            vals, dt, |v| v.as_f64().map(|f| f as f32),
        )?)),

        DataType::Float64 => Column::Float64(PrimitiveColumn::new(convert(
            vals, dt, |v| v.as_f64(),
        )?)),

        // Any scalar has a textual form, so this conversion cannot fail.
        DataType::Utf8 => Column::Utf8(Utf8Column::new(
            vals.iter()
                .map(|v| if v.is_null() { None } else { Some(v.to_string()) })
                .collect(),
        )),

        DataType::Date32 => Column::Date32(PrimitiveColumn::new(convert(
            vals, dt, |v| v.as_date32(),
        )?)),

        DataType::Timestamp => Column::Timestamp(PrimitiveColumn::new(convert(
            vals, dt, |v| v.as_timestamp(),
        )?)),

        DataType::Decimal128(p, s) => {
            let (p, s) = (*p, *s);
            use crate::storage::column::Decimal128Column;
            Column::Decimal128(Decimal128Column::new(
                convert(vals, dt, |v| match v {
                    ScalarValue::Decimal128(d) => d.value,
                    other => other.as_f64().map(|f| (f * 10f64.powi(s as i32)) as i128),
                })?,
                p, s,
            ))
        }

        DataType::Null => Column::Int64(PrimitiveColumn::new(vec![None; vals.len()])),
    })
}

/// Merge multiple batches into one by concatenating row-by-row (scalar slow-path).
/// Used only for RETURNING result sets, which are typically small.

/// Overwrite the selected rows of a primitive column in place, densely.
///
/// The general path boxes *every* row of the target column into a `ScalarValue` —
/// the untouched ones included — and rebuilds the column from that. It is the same
/// shape that made joins 100x slow, and here it is paid per updated column: at 100k
/// rows UPDATE measured 7.5 ms against DuckDB's 3.8 ms.
///
/// Returns `None` for anything it cannot handle — a nullable source, a type it does
/// not list, or a computed value that is NULL or of another type — so the caller
/// falls back and the semantics stay in one place. A NULL assignment is rare and
/// changes the column's validity, which the dense copy here does not model.
/// Splice one constant into the selected rows.
///
/// `UPDATE t SET col = <literal>` is the common shape, and the general path
/// handles it by first asking `eval_batch` for the value of that literal once per
/// row: `Expr::Literal(v) => vec![v.clone(); row_count]`. For a string that is a
/// heap allocation per row of the table before any row is written, and for a
/// dictionary column it then looks the same string up once per row as well.
///
/// Here the value is converted once. A text column resolves its code once too,
/// so the work per selected row is a `u32` store.
///
/// Returns `None` for anything it cannot model, exactly as [`splice_typed`] does,
/// and the caller falls through to the general path.
fn splice_constant(src: &Column, value: &ScalarValue, selected: &[bool]) -> Option<Column> {
    use crate::storage::column::PrimitiveColumn;

    debug_assert_eq!(src.len(), selected.len(), "mask must cover the column");

    // Writing a NULL changes the column's validity, which the dense forms below
    // do not model. The general path knows how.
    if value.is_null() {
        return None;
    }

    macro_rules! dense {
        ($c:expr, $variant:ident, $extract:expr) => {{
            let c = $c;
            if c.has_nulls() {
                return None;
            }
            let v = $extract(value)?;
            let mut values = c.dense().to_vec();
            for (row, keep) in selected.iter().enumerate() {
                if *keep {
                    values[row] = v;
                }
            }
            Some(Column::$variant(PrimitiveColumn::from_non_null(values)))
        }};
    }

    match src {
        Column::Int64(c) => dense!(c, Int64, |v: &ScalarValue| match v {
            ScalarValue::Int64(Some(x)) => Some(*x),
            _ => None,
        }),
        Column::Float64(c) => dense!(c, Float64, |v: &ScalarValue| match v {
            ScalarValue::Float64(Some(x)) => Some(*x),
            _ => None,
        }),
        Column::Int32(c) => dense!(c, Int32, |v: &ScalarValue| match v {
            ScalarValue::Int32(Some(x)) => Some(*x),
            _ => None,
        }),
        Column::Float32(c) => dense!(c, Float32, |v: &ScalarValue| match v {
            ScalarValue::Float32(Some(x)) => Some(*x),
            _ => None,
        }),

        Column::Utf8(c) => {
            let ScalarValue::Utf8(Some(text)) = value else {
                return None;
            };
            // One lookup for the whole statement, not one per row.
            let mut dict: Vec<String> = c.dict().to_vec();
            let code = match dict.iter().position(|d| d == text) {
                Some(i) => i as u32,
                None => {
                    dict.push(text.clone());
                    (dict.len() - 1) as u32
                }
            };
            let mut codes: Vec<u32> = c.codes().to_vec();
            for (row, keep) in selected.iter().enumerate() {
                if *keep {
                    codes[row] = code;
                }
            }
            Some(Column::Utf8(Utf8Column::from_parts(std::sync::Arc::new(dict), codes)))
        }

        _ => None,
    }
}

fn splice_typed(
    src:      &Column,
    computed: &[ScalarValue],
    selected: &[bool],
) -> Option<Column> {
    use crate::storage::column::PrimitiveColumn;

    if src.null_count() > 0 {
        return None;
    }

    macro_rules! splice {
        ($col:expr, $variant:ident, $extract:expr) => {{
            let mut values = $col.dense().to_vec();
            for (r, keep) in selected.iter().enumerate() {
                if *keep {
                    values[r] = $extract(&computed[r])?;
                }
            }
            Some(Column::$variant(PrimitiveColumn::from_non_null(values)))
        }};
    }

    match src {
        Column::Int64(c) => splice!(c, Int64, |v: &ScalarValue| match v {
            ScalarValue::Int64(Some(x)) => Some(*x),
            _ => None,
        }),
        Column::Float64(c) => splice!(c, Float64, |v: &ScalarValue| match v {
            ScalarValue::Float64(Some(x)) => Some(*x),
            _ => None,
        }),
        Column::Int32(c) => splice!(c, Int32, |v: &ScalarValue| match v {
            ScalarValue::Int32(Some(x)) => Some(*x),
            _ => None,
        }),
        Column::Float32(c) => splice!(c, Float32, |v: &ScalarValue| match v {
            ScalarValue::Float32(Some(x)) => Some(*x),
            _ => None,
        }),

        // A dictionary column splices by code, not by string. Without this arm a
        // string UPDATE fell to the general path, which reads every row as a
        // `ScalarValue::Utf8` -- a heap allocation per row of the table, whether
        // the row is being changed or not. Measured on 100k rows with half of
        // them updated: 23.99ms against 2.34ms for the same UPDATE on a Float64
        // column, so 10x, and the benchmark suite never reached it because its
        // DML group updates a float.
        Column::Utf8(c) => {
            let mut dict: Vec<String> = c.dict().to_vec();
            let mut codes: Vec<u32> = c.codes().to_vec();
            // The dictionary is small -- distinct values, not rows -- and an
            // UPDATE usually writes one repeated value, so this map is built once
            // and hit thereafter.
            let mut lookup: crate::query::hash::FastMap<String, u32> = dict
                .iter()
                .enumerate()
                .map(|(i, sval)| (sval.clone(), i as u32))
                .collect();

            for (row, keep) in selected.iter().enumerate() {
                if !keep {
                    continue;
                }
                match computed.get(row) {
                    Some(ScalarValue::Utf8(Some(sval))) => {
                        let code = match lookup.get(sval) {
                            Some(&code) => code,
                            None => {
                                let code = dict.len() as u32;
                                dict.push(sval.clone());
                                lookup.insert(sval.clone(), code);
                                code
                            }
                        };
                        codes[row] = code;
                    }
                    Some(ScalarValue::Utf8(None)) | Some(ScalarValue::Null) => {
                        codes[row] = Utf8Column::NULL;
                    }
                    // Anything else means the expression did not produce a
                    // string; the general path knows how to report that.
                    _ => return None,
                }
            }
            Some(Column::Utf8(Utf8Column::from_parts(std::sync::Arc::new(dict), codes)))
        }

        _ => None,
    }
}

/// Concatenate the batches a DML statement is returning.
///
/// Delegates to [`RecordBatch::concat`]. What stood here was a third private
/// copy of the merge, and the worst of them: it pushed every cell through a
/// `ScalarValue`, which is an enum allocation per value, and then rebuilt each
/// column from that. O(rows x columns) boxing for what the shared version does
/// as a memcpy per column.
///
/// That exact shape once made this engine's joins 100x slower than they needed
/// to be, and nothing caught it because the copy was *correct*. It was still
/// here, on the path `UPDATE ... RETURNING` and `DELETE ... RETURNING` take.
///
/// The `table` argument is gone with it: `RecordBatch::concat` takes the schema
/// from the batches, which is where it came from in the first place.
fn merge_batches_simple(batches: &[RecordBatch]) -> FalconResult<RecordBatch> {
    RecordBatch::concat(batches)
}

/// True when `returning` is a single `RETURNING *` wildcard sentinel.
fn is_wildcard_returning(returning: &[SelectExpr]) -> bool {
    matches!(returning, [se] if se.alias == "*")
}

/// Build an empty table that keeps the given schema.
pub fn empty_like(name: &str, schema: &Schema) -> Table {
    Table::new(name, schema.clone())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod merge_perf {
    use super::*;
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        types::DataType,
    };
    use std::time::Instant;

    /// What does `UPDATE` cost on a column the typed splice does not cover?
    ///
    /// `splice_typed` handles `Int64`, `Float64`, `Int32` and `Float32`. Anything
    /// else -- `Utf8`, `Boolean`, `Date32`, `Timestamp`, `Decimal128` -- falls to
    /// the general path, which reads every row as a `ScalarValue` and rebuilds the
    /// column from that. For `Utf8` each of those reads allocates a `String`.
    ///
    /// The benchmark suite does not reach it: its `UPDATE` sets `score`, a
    /// `Float64`, which the fast path takes. So a string update is untested for
    /// speed as well as unoptimised.
    ///
    /// Measured before the Utf8 arm existed: 23.99ms against 2.34ms, so 10.3x.
    /// After it: 8.14-8.27ms, so 3.6x. What is left is not the splice -- it is
    /// `eval_batch`, which returns a `ScalarValue` per row even when the
    /// expression is the constant `'x'`, so a string constant is cloned once per
    /// row of the table before the splice ever sees it.
    ///
    /// Run with
    /// `cargo test --release --lib update_by_column_type -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn update_by_column_type() {
        use crate::db::Database;
        use crate::storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::RecordBatch,
            types::DataType,
        };
        use std::time::Instant;

        const ROWS: usize = 100_000;

        let make_db = || {
            let schema = Schema::new(vec![
                Field::required("id", DataType::Int64),
                Field::required("dept", DataType::Utf8),
                Field::required("score", DataType::Float64),
            ]);
            let batch = RecordBatch::new(
                schema,
                vec![
                    ColumnBuilder::int64((0..ROWS as i64).map(Some).collect()),
                    ColumnBuilder::utf8_from_str(
                        (0..ROWS)
                            .map(|i| Some(["eng", "sales", "hr", "ops", "legal"][i % 5]))
                            .collect(),
                    ),
                    ColumnBuilder::float64((0..ROWS).map(|i| Some((i % 100) as f64)).collect()),
                ],
            )
            .unwrap();
            let db = Database::in_memory();
            db.create_table_from_batches("t", vec![batch]).unwrap();
            db
        };

        let time = |label: &str, sql: &str| {
            // A fresh table each time: an UPDATE changes what the next one sees.
            for _ in 0..2 {
                let db = make_db();
                db.sql(sql).unwrap();
            }
            let mut best = std::time::Duration::MAX;
            for _ in 0..7 {
                let db = make_db();
                let t = Instant::now();
                db.sql(sql).unwrap();
                best = best.min(t.elapsed());
            }
            println!("  {label:>34}: {best:>9.2?}");
            best
        };

        println!("UPDATE half of {ROWS} rows:");
        let f = time("score (Float64, typed splice)", "UPDATE t SET score = score + 1 WHERE score > 50");
        let u = time("dept (Utf8, ScalarValue path)", "UPDATE t SET dept = 'x' WHERE score > 50");
        println!(
            "  the string column costs {:.1}x the float one",
            u.as_secs_f64() / f.as_secs_f64(),
        );
    }

    /// What the `ScalarValue` merge cost the RETURNING path.
    ///
    /// `merge_batches_simple` used to rebuild every column by pushing each cell
    /// through a `ScalarValue` -- an enum allocation per value -- and this stayed
    /// in place long after the same shape was found and fixed in the join, where
    /// it had made joins 100x slower.
    ///
    /// The benchmark suite never covered it: its DML group runs `UPDATE` and
    /// `DELETE` without `RETURNING`, and the merge only happens when rows are
    /// returned. So the speedup has to be measured here or not at all.
    ///
    /// Run with
    /// `cargo test --release --lib returning_merge_shapes -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn returning_merge_shapes() {
        const ROWS: usize = 100_000;
        const BATCH: usize = 65_536;

        let schema = Schema::new(vec![
            Field::required("id", DataType::Int64),
            Field::required("dept", DataType::Utf8),
            Field::required("score", DataType::Float64),
        ]);
        let mut batches = Vec::new();
        let mut start = 0;
        while start < ROWS {
            let n = BATCH.min(ROWS - start);
            batches.push(
                RecordBatch::new(
                    schema.clone(),
                    vec![
                        ColumnBuilder::int64((0..n).map(|i| Some((start + i) as i64)).collect()),
                        ColumnBuilder::utf8_from_str(
                            (0..n)
                                .map(|i| Some(["eng", "sales", "hr", "ops", "legal"][(start + i) % 5]))
                                .collect(),
                        ),
                        ColumnBuilder::float64(
                            (0..n).map(|i| Some(((start + i) % 100) as f64 + 0.5)).collect(),
                        ),
                    ],
                )
                .unwrap(),
            );
            start += n;
        }

        /// The shape that was here: every cell through a ScalarValue.
        fn boxed(batches: &[RecordBatch], schema: &Schema) -> FalconResult<RecordBatch> {
            let total = batches.iter().map(|b| b.row_count()).sum::<usize>();
            let mut per_col: Vec<Vec<ScalarValue>> = (0..schema.field_count())
                .map(|_| Vec::with_capacity(total))
                .collect();
            for batch in batches {
                for row in 0..batch.row_count() {
                    for col in 0..schema.field_count() {
                        per_col[col].push(batch.column(col).get(row)?);
                    }
                }
            }
            let columns: FalconResult<Vec<Column>> = per_col
                .iter()
                .enumerate()
                .map(|(i, vals)| scalars_to_column(vals, &schema.fields[i].data_type))
                .collect();
            RecordBatch::new(schema.clone(), columns?)
        }

        // Both must produce the same rows; the old one was correct, only slow.
        let a = RecordBatch::concat(&batches).unwrap();
        let b = boxed(&batches, &schema).unwrap();
        assert_eq!(a.row_count(), b.row_count());
        for c in 0..a.column_count() {
            for r in (0..a.row_count()).step_by(4999) {
                assert_eq!(a.column(c).get(r).unwrap(), b.column(c).get(r).unwrap(), "col {c} row {r}");
            }
        }

        let bench = |label: &str, f: &dyn Fn() -> RecordBatch| {
            for _ in 0..3 { std::hint::black_box(f()); }
            let t = Instant::now();
            for _ in 0..20 { std::hint::black_box(f()); }
            let per = t.elapsed() / 20;
            println!("  {label:>24}: {per:>9.2?}");
            per
        };

        println!("merging {ROWS} rows, 3 columns, for RETURNING:");
        let old = bench("via ScalarValue", &|| boxed(&batches, &schema).unwrap());
        let new = bench("typed concat", &|| RecordBatch::concat(&batches).unwrap());
        println!("  typed is {:.2}x the boxed merge", new.as_secs_f64() / old.as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        query::expr::BinaryOp,
        storage::{
            column::ColumnBuilder,
            schema::Field,
            types::{DataType, ScalarValue},
        },
    };

    /// The typed splice must produce exactly what the boxed path produces.
    ///
    /// It is a second implementation of the same operation, and this codebase has
    /// already paid for two of those drifting apart, so it is pinned to the general
    /// path rather than to a hand-written expectation.
    #[test]
    fn typed_splice_agrees_with_the_boxed_path() {
        let src = ColumnBuilder::float64((0..64).map(|i| Some(i as f64)).collect());
        let computed: Vec<ScalarValue> =
            (0..64).map(|i| ScalarValue::Float64(Some(i as f64 * 10.0))).collect();
        let selected: Vec<bool> = (0..64).map(|i| i % 3 == 0).collect();

        let typed = splice_typed(&src, &computed, &selected)
            .expect("a null-free Float64 column should take the typed path");

        let mut values: Vec<ScalarValue> = (0..64).map(|r| src.get(r).unwrap()).collect();
        for (r, keep) in selected.iter().enumerate() {
            if *keep {
                values[r] = computed[r].clone();
            }
        }
        let boxed = scalars_to_column(&values, &DataType::Float64).unwrap();

        assert_eq!(typed.len(), boxed.len());
        for r in 0..typed.len() {
            assert_eq!(typed.get(r).unwrap(), boxed.get(r).unwrap(), "row {r} differs");
        }
    }

    /// Cases the typed splice must decline rather than guess at.
    #[test]
    fn typed_splice_declines_what_it_cannot_model() {
        let selected = vec![true, false];

        // Assigning NULL changes the column's validity, which the dense copy does
        // not model — it has to fall back, not write a placeholder.
        let src = ColumnBuilder::int64(vec![Some(1), Some(2)]);
        let to_null = vec![ScalarValue::Int64(None), ScalarValue::Int64(Some(9))];
        assert!(splice_typed(&src, &to_null, &selected).is_none());

        // A source that already has NULLs, for the same reason.
        let nullable = ColumnBuilder::int64(vec![None, Some(2)]);
        let vals = vec![ScalarValue::Int64(Some(7)), ScalarValue::Int64(Some(9))];
        assert!(splice_typed(&nullable, &vals, &selected).is_none());

        // A computed value of another type must not be silently coerced.
        let mismatched = vec![ScalarValue::Utf8(Some("x".into())), ScalarValue::Int64(Some(9))];
        assert!(splice_typed(&src, &mismatched, &selected).is_none());

        // Boolean has no dense splice, so it still declines.
        let flags = ColumnBuilder::boolean(vec![Some(true), Some(false)]);
        let flag_vals = vec![ScalarValue::Boolean(Some(false)), ScalarValue::Boolean(Some(true))];
        assert!(splice_typed(&flags, &flag_vals, &selected).is_none());
    }

    /// Text is spliced by dictionary code, and the result has to match what the
    /// general path would have produced.
    ///
    /// This used to be part of `typed_splice_declines_what_it_cannot_model`:
    /// text was on the declined list, and a string UPDATE cost 10x what the same
    /// update on a float column cost, because the fallback reads every row of the
    /// table as a `ScalarValue::Utf8` -- one heap allocation per row, changed or
    /// not.
    #[test]
    fn typed_splice_handles_text_by_code() {
        let selected = vec![true, false, true, false];
        let text = ColumnBuilder::utf8_from_str(vec![Some("a"), Some("b"), Some("c"), Some("d")]);

        // One value already in the dictionary, one that is not.
        let vals = vec![
            ScalarValue::Utf8(Some("b".into())),
            ScalarValue::Utf8(Some("ignored".into())),
            ScalarValue::Utf8(Some("brand new".into())),
            ScalarValue::Utf8(Some("ignored".into())),
        ];
        let got = splice_typed(&text, &vals, &selected).expect("text should splice");
        let expect = ["b", "b", "brand new", "d"];
        for (row, want) in expect.iter().enumerate() {
            assert_eq!(
                got.get(row).unwrap(),
                ScalarValue::Utf8(Some((*want).into())),
                "row {row}",
            );
        }

        // Writing a NULL into a column that had none must be visible as a NULL,
        // not as whatever string the code happens to point at.
        let to_null = vec![
            ScalarValue::Utf8(None),
            ScalarValue::Utf8(Some("x".into())),
            ScalarValue::Null,
            ScalarValue::Utf8(Some("x".into())),
        ];
        let got = splice_typed(&text, &to_null, &selected).expect("text should splice");
        assert!(got.get(0).unwrap().is_null(), "row 0 should be NULL");
        assert!(got.get(2).unwrap().is_null(), "row 2 should be NULL");
        assert_eq!(got.get(1).unwrap(), ScalarValue::Utf8(Some("b".into())), "row 1 untouched");
        assert_eq!(got.null_count(), 2, "and the column should know it has NULLs");

        // A non-string value must still be declined rather than coerced.
        let wrong = vec![
            ScalarValue::Int64(Some(1)),
            ScalarValue::Utf8(Some("x".into())),
            ScalarValue::Utf8(Some("y".into())),
            ScalarValue::Utf8(Some("z".into())),
        ];
        assert!(splice_typed(&text, &wrong, &selected).is_none());
    }

    fn make_table() -> Table {
        let schema = Schema::new(vec![
            Field::required("id", DataType::Int64),
            Field::nullable("name", DataType::Utf8),
            Field::nullable("score", DataType::Float64),
        ]);
        let batch = RecordBatch::new(
            schema.clone(),
            vec![
                ColumnBuilder::int64(vec![Some(1), Some(2), Some(3)]),
                ColumnBuilder::utf8_from_str(vec![Some("alice"), Some("bob"), Some("carol")]),
                ColumnBuilder::float64(vec![Some(9.5), Some(7.0), Some(8.5)]),
            ],
        )
        .unwrap();
        let mut t = Table::new("t", schema);
        t.append_batch(batch).unwrap();
        t
    }

    fn lit_i(n: i64) -> Expr { Expr::Literal(ScalarValue::Int64(Some(n))) }
    fn lit_f(n: f64) -> Expr { Expr::Literal(ScalarValue::Float64(Some(n))) }
    fn lit_s(s: &str) -> Expr { Expr::Literal(ScalarValue::Utf8(Some(s.into()))) }

    fn gt(col: &str, v: f64) -> Expr {
        Expr::BinaryOp {
            left:  Box::new(Expr::Column(col.into())),
            op:    BinaryOp::Gt,
            right: Box::new(lit_f(v)),
        }
    }

    // ── INSERT ───────────────────────────────────────────────────────────────

    #[test]
    fn insert_appends_rows() {
        let t = make_table();
        let stmt = InsertStmt {
            table:     "t".into(),
            columns:   vec![],
            source:    crate::query::dml::InsertSource::Values(vec![vec![lit_i(4), lit_s("dave"), lit_f(6.0)]]),
            returning: vec![],
        };
        let (out, n, _) = execute_insert(&t, &stmt).unwrap();
        assert_eq!(n, 1);
        assert_eq!(out.row_count(), 4);
    }

    #[test]
    fn insert_named_columns_fills_rest_with_null() {
        let t = make_table();
        let stmt = InsertStmt {
            table:     "t".into(),
            columns:   vec!["id".into()],
            source:    crate::query::dml::InsertSource::Values(vec![vec![lit_i(9)]]),
            returning: vec![],
        };
        let (out, n, _) = execute_insert(&t, &stmt).unwrap();
        assert_eq!(n, 1);
        assert_eq!(out.row_count(), 4);

        // The appended batch is last; its single row must have a NULL name.
        let last = out.batches.last().unwrap();
        assert!(last.column_by_name("name").unwrap().get(0).unwrap().is_null());
    }

    #[test]
    fn insert_arity_mismatch_errors() {
        let t = make_table();
        let stmt = InsertStmt {
            table:     "t".into(),
            columns:   vec!["id".into(), "name".into()],
            source:    crate::query::dml::InsertSource::Values(vec![vec![lit_i(4)]]),
            returning: vec![],
        };
        assert!(execute_insert(&t, &stmt).is_err());
    }

    #[test]
    fn insert_rejects_non_literal() {
        let t = make_table();
        let stmt = InsertStmt {
            table:     "t".into(),
            columns:   vec!["id".into()],
            source:    crate::query::dml::InsertSource::Values(vec![vec![Expr::Column("id".into())]]),
            returning: vec![],
        };
        assert!(execute_insert(&t, &stmt).is_err());
    }

    // ── DELETE ───────────────────────────────────────────────────────────────

    #[test]
    fn delete_with_filter_removes_matching() {
        let t = make_table();
        let stmt = DeleteStmt { table: "t".into(), filter: Some(gt("score", 8.0)), returning: vec![] };
        let (out, n, _) = execute_delete(&t, &stmt).unwrap();
        assert_eq!(n, 2, "9.5 and 8.5 should go");
        assert_eq!(out.row_count(), 1);
        assert_eq!(
            out.batches[0].column_by_name("name").unwrap().get(0).unwrap(),
            ScalarValue::Utf8(Some("bob".into()))
        );
    }

    #[test]
    fn delete_without_filter_empties_table() {
        let t = make_table();
        let stmt = DeleteStmt { table: "t".into(), filter: None, returning: vec![] };
        let (out, n, _) = execute_delete(&t, &stmt).unwrap();
        assert_eq!(n, 3);
        assert_eq!(out.row_count(), 0);
    }

    #[test]
    fn delete_matching_nothing_keeps_all() {
        let t = make_table();
        let stmt = DeleteStmt { table: "t".into(), filter: Some(gt("score", 1000.0)), returning: vec![] };
        let (out, n, _) = execute_delete(&t, &stmt).unwrap();
        assert_eq!(n, 0);
        assert_eq!(out.row_count(), 3);
    }

    // ── UPDATE ───────────────────────────────────────────────────────────────

    #[test]
    fn update_with_filter_changes_only_matching() {
        let t = make_table();
        let stmt = UpdateStmt {
            table:       "t".into(),
            assignments: vec![("score".into(), lit_f(0.0))],
            filter:      Some(gt("score", 8.0)),
            returning:   vec![],
        };
        let (out, n, _) = execute_update(&t, &stmt).unwrap();
        assert_eq!(n, 2);

        let scores = out.batches[0].column_by_name("score").unwrap();
        assert_eq!(scores.get(0).unwrap(), ScalarValue::Float64(Some(0.0))); // was 9.5
        assert_eq!(scores.get(1).unwrap(), ScalarValue::Float64(Some(7.0))); // untouched
        assert_eq!(scores.get(2).unwrap(), ScalarValue::Float64(Some(0.0))); // was 8.5
    }

    #[test]
    fn update_without_filter_changes_every_row() {
        let t = make_table();
        let stmt = UpdateStmt {
            table:       "t".into(),
            assignments: vec![("name".into(), lit_s("x"))],
            filter:      None,
            returning:   vec![],
        };
        let (out, n, _) = execute_update(&t, &stmt).unwrap();
        assert_eq!(n, 3);
        let names = out.batches[0].column_by_name("name").unwrap();
        for row in 0..3 {
            assert_eq!(names.get(row).unwrap(), ScalarValue::Utf8(Some("x".into())));
        }
    }

    #[test]
    fn update_can_reference_other_columns() {
        let t = make_table();
        let stmt = UpdateStmt {
            table:       "t".into(),
            assignments: vec![(
                "score".into(),
                Expr::BinaryOp {
                    left:  Box::new(Expr::Column("score".into())),
                    op:    BinaryOp::Add,
                    right: Box::new(lit_f(1.0)),
                },
            )],
            filter:    None,
            returning: vec![],
        };
        let (out, n, _) = execute_update(&t, &stmt).unwrap();
        assert_eq!(n, 3);
        let scores = out.batches[0].column_by_name("score").unwrap();
        assert_eq!(scores.get(0).unwrap(), ScalarValue::Float64(Some(10.5)));
        assert_eq!(scores.get(1).unwrap(), ScalarValue::Float64(Some(8.0)));
    }

    #[test]
    fn update_unknown_column_errors() {
        let t = make_table();
        let stmt = UpdateStmt {
            table:       "t".into(),
            assignments: vec![("nope".into(), lit_i(1))],
            filter:      None,
            returning:   vec![],
        };
        assert!(execute_update(&t, &stmt).is_err());
    }

    // ── invert_mask ──────────────────────────────────────────────────────────

    #[test]
    fn invert_mask_clears_padding_bits() {
        // 3 rows, none selected → inverting must select exactly rows 0..3.
        let inverted = invert_mask(&[0u64], 3);
        assert_eq!(inverted[0], 0b111);
    }

    #[test]
    fn invert_mask_round_trips() {
        let mask = [0b1010u64];
        let inv = invert_mask(&mask, 4);
        assert_eq!(inv[0], 0b0101);
    }
}
