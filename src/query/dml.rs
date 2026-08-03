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

use super::expr::Expr;

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
    pub table:   String,
    /// Target columns. Empty means "every column, in schema order".
    pub columns: Vec<String>,
    pub source:  InsertSource,
}

#[derive(Debug, Clone)]
pub struct UpdateStmt {
    pub table:       String,
    pub assignments: Vec<(String, Expr)>,
    pub filter:      Option<Expr>,
}

#[derive(Debug, Clone)]
pub struct DeleteStmt {
    pub table:  String,
    pub filter: Option<Expr>,
}

// ─── INSERT ──────────────────────────────────────────────────────────────────

/// Append rows to `table`, returning the new table and the number of rows inserted.
///
/// Only `InsertSource::Values` is handled here.  `InsertSource::Select` must be
/// resolved to a `Values` insert by the caller before this is called.
pub fn execute_insert(table: &Table, stmt: &InsertStmt) -> FalconResult<(Table, usize)> {
    let rows = match &stmt.source {
        InsertSource::Values(rows) => rows,
        InsertSource::Select(_) => {
            return Err(FalconError::InvalidQuery(
                "INSERT … SELECT must be resolved before execute_insert".into(),
            ));
        }
    };

    if rows.is_empty() {
        return Ok((clone_table(table), 0));
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

    let mut out = clone_table(table);
    out.append_batch(batch)?;
    Ok((out, n_new))
}

// ─── DELETE ──────────────────────────────────────────────────────────────────

/// Remove rows matching `stmt.filter`, returning the new table and rows removed.
/// A statement with no filter empties the table.
pub fn execute_delete(table: &Table, stmt: &DeleteStmt) -> FalconResult<(Table, usize)> {
    let mut out = clone_table_empty(table);
    let mut removed = 0usize;

    for batch in &table.batches {
        match &stmt.filter {
            None => {
                removed += batch.row_count();
            }
            Some(filter) => {
                // Keep rows where the predicate is *not* true, so NULLs survive —
                // matching SQL's three-valued logic for DELETE.
                let mask = filter.eval_mask_bits(batch)?;
                let keep = invert_mask(&mask, batch.row_count());
                let kept = batch.filter_by_bitmask(&keep)?;
                removed += batch.row_count() - kept.row_count();
                if kept.row_count() > 0 {
                    out.append_batch(kept)?;
                }
            }
        }
    }

    Ok((out, removed))
}

// ─── UPDATE ──────────────────────────────────────────────────────────────────

/// Apply `stmt.assignments` to rows matching `stmt.filter`.
/// Returns the new table and the number of rows changed.
pub fn execute_update(table: &Table, stmt: &UpdateStmt) -> FalconResult<(Table, usize)> {
    let schema = &table.schema;

    // Resolve assignment targets up front so a bad column name fails before any work.
    let targets: Vec<(usize, &Expr)> = stmt
        .assignments
        .iter()
        .map(|(name, expr)| Ok((schema.column_index(name)?, expr)))
        .collect::<FalconResult<_>>()?;

    let mut out = clone_table_empty(table);
    let mut changed = 0usize;

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
            let computed = expr.eval_batch(batch)?;
            let mut values: Vec<ScalarValue> = (0..n_rows)
                .map(|r| batch.column(*col_idx).get(r))
                .collect::<FalconResult<_>>()?;
            for (r, keep) in selected.iter().enumerate() {
                if *keep {
                    values[r] = computed[r].clone();
                }
            }
            let dt = &schema.fields[*col_idx].data_type;
            columns[*col_idx] = Arc::new(scalars_to_column(&values, dt)?);
        }

        out.append_batch(RecordBatch::from_shared(schema.clone(), columns)?)?;
    }

    Ok((out, changed))
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

/// Build an empty table that keeps the given schema.
pub fn empty_like(name: &str, schema: &Schema) -> Table {
    Table::new(name, schema.clone())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

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
            table:   "t".into(),
            columns: vec![],
            source:  crate::query::dml::InsertSource::Values(vec![vec![lit_i(4), lit_s("dave"), lit_f(6.0)]]),
        };
        let (out, n) = execute_insert(&t, &stmt).unwrap();
        assert_eq!(n, 1);
        assert_eq!(out.row_count(), 4);
    }

    #[test]
    fn insert_named_columns_fills_rest_with_null() {
        let t = make_table();
        let stmt = InsertStmt {
            table:   "t".into(),
            columns: vec!["id".into()],
            source:  crate::query::dml::InsertSource::Values(vec![vec![lit_i(9)]]),
        };
        let (out, n) = execute_insert(&t, &stmt).unwrap();
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
            table:   "t".into(),
            columns: vec!["id".into(), "name".into()],
            source:  crate::query::dml::InsertSource::Values(vec![vec![lit_i(4)]]),
        };
        assert!(execute_insert(&t, &stmt).is_err());
    }

    #[test]
    fn insert_rejects_non_literal() {
        let t = make_table();
        let stmt = InsertStmt {
            table:   "t".into(),
            columns: vec!["id".into()],
            source:  crate::query::dml::InsertSource::Values(vec![vec![Expr::Column("id".into())]]),
        };
        assert!(execute_insert(&t, &stmt).is_err());
    }

    // ── DELETE ───────────────────────────────────────────────────────────────

    #[test]
    fn delete_with_filter_removes_matching() {
        let t = make_table();
        let stmt = DeleteStmt { table: "t".into(), filter: Some(gt("score", 8.0)) };
        let (out, n) = execute_delete(&t, &stmt).unwrap();
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
        let stmt = DeleteStmt { table: "t".into(), filter: None };
        let (out, n) = execute_delete(&t, &stmt).unwrap();
        assert_eq!(n, 3);
        assert_eq!(out.row_count(), 0);
    }

    #[test]
    fn delete_matching_nothing_keeps_all() {
        let t = make_table();
        let stmt = DeleteStmt { table: "t".into(), filter: Some(gt("score", 1000.0)) };
        let (out, n) = execute_delete(&t, &stmt).unwrap();
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
        };
        let (out, n) = execute_update(&t, &stmt).unwrap();
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
        };
        let (out, n) = execute_update(&t, &stmt).unwrap();
        assert_eq!(n, 3);
        let names = out.batches[0].column_by_name("name").unwrap();
        for row in 0..3 {
            assert_eq!(names.get(row).unwrap(), ScalarValue::Utf8(Some("x".into())));
        }
    }

    #[test]
    fn update_can_reference_other_columns() {
        let t = make_table();
        // SET score = score + 1
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
            filter: None,
        };
        let (out, n) = execute_update(&t, &stmt).unwrap();
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
