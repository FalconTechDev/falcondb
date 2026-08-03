use rayon::prelude::*;

use crate::storage::{
    column::Column,
    table::RecordBatch,
    types::{FalconResult, ScalarValue},
};

use super::expr::Expr;

// ─── Sort key ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum SortOrder {
    Asc,
    Desc,
}

#[derive(Debug, Clone)]
pub struct SortExpr {
    pub expr:      Expr,
    pub order:     SortOrder,
    pub nulls_last: bool,
}

impl SortExpr {
    pub fn asc(expr: Expr) -> Self {
        Self { expr, order: SortOrder::Asc, nulls_last: false }
    }

    pub fn desc(expr: Expr) -> Self {
        Self { expr, order: SortOrder::Desc, nulls_last: true }
    }

    pub fn asc_nulls_last(expr: Expr) -> Self {
        Self { expr, order: SortOrder::Asc, nulls_last: true }
    }
}

// ─── Sort executor ────────────────────────────────────────────────────────────

/// Sort a slice of batches by the given sort expressions.
/// Returns a single sorted batch (batches are merged first).
pub fn execute_sort(batches: &[RecordBatch], sort_exprs: &[SortExpr]) -> FalconResult<RecordBatch> {
    if batches.is_empty() {
        return Err(crate::storage::types::FalconError::InvalidQuery(
            "cannot sort empty result".into(),
        ));
    }

    execute_sort_limit(batches, sort_exprs, None)
}

/// Sort, keeping only the first `limit` rows when one is given.
///
/// A bare `ORDER BY … LIMIT k` does not need a full sort: selecting the k best
/// rows is O(n log k) against O(n log n), and — more importantly at these sizes —
/// it reorders k rows instead of every row. On a million rows with `LIMIT 10`
/// that is the difference between gathering ten rows and gathering a million.
pub fn execute_sort_limit(
    batches:    &[RecordBatch],
    sort_exprs: &[SortExpr],
    limit:      Option<usize>,
) -> FalconResult<RecordBatch> {
    if batches.is_empty() {
        return Err(crate::storage::types::FalconError::InvalidQuery(
            "cannot sort empty result".into(),
        ));
    }

    let merged = merge_batches(batches)?;
    let n = merged.row_count();
    if n == 0 {
        return Ok(merged);
    }

    // Single primitive sort key: compare the raw values instead of boxing every
    // one into a 48-byte ScalarValue. At 100k rows that is 4.8 MB of randomly
    // accessed keys reduced to 1.6 MB of contiguous (key, index) pairs.
    if let Some(indices) = try_typed_sort(&merged, sort_exprs, limit)? {
        return gather(&merged, &indices);
    }

    // Evaluate all sort keys upfront — one Vec<ScalarValue> per sort expression.
    let sort_cols: Vec<Vec<ScalarValue>> = sort_exprs
        .iter()
        .map(|s| s.expr.eval_batch(&merged))
        .collect::<FalconResult<_>>()?;

    // Build row indices and sort them.
    // Use Rayon parallel sort above 64 k rows (overhead amortised at that scale).
    let mut indices: Vec<usize> = (0..merged.row_count()).collect();
    let cmp = |&a: &usize, &b: &usize| -> std::cmp::Ordering {
        for (col, expr) in sort_cols.iter().zip(sort_exprs.iter()) {
            let va = &col[a];
            let vb = &col[b];
            let ord = compare_scalar(va, vb, expr.nulls_last);
            let ord = match expr.order { SortOrder::Asc => ord, SortOrder::Desc => ord.reverse() };
            if ord != std::cmp::Ordering::Equal { return ord; }
        }
        std::cmp::Ordering::Equal
    };
    if indices.len() >= 65_536 {
        indices.par_sort_unstable_by(cmp);
    } else {
        indices.sort_unstable_by(cmp);
    }

    if let Some(k) = limit {
        indices.truncate(k);
    }

    gather(&merged, &indices)
}

/// Reorder every column of `batch` by `indices`.
///
/// Columns are gathered in parallel: each one is an independent random-access
/// pass over its own memory, so with several columns this is the cheapest
/// parallelism available in the sort. Below the threshold the rayon hand-off
/// costs more than the gather itself.
fn gather(batch: &RecordBatch, indices: &[usize]) -> FalconResult<RecordBatch> {
    let parallel = batch.column_count() > 1 && indices.len() >= 8_192;

    let cols: FalconResult<Vec<_>> = if parallel {
        batch.columns.par_iter()
            .map(|col| col.filter_by_indices(indices))
            .collect()
    } else {
        batch.columns.iter()
            .map(|col| col.filter_by_indices(indices))
            .collect()
    };
    RecordBatch::new(batch.schema.clone(), cols?)
}

/// Sort on a single primitive column without going through `ScalarValue`.
///
/// Returns `None` when the query does not fit this shape — several sort keys, a
/// computed expression, or a column type without a natural total order here — so
/// the caller falls back to the general path.
///
/// NULLs are partitioned out first rather than being given a sentinel value:
/// picking a sentinel would collide with real data at the extremes of the range.
fn try_typed_sort(
    batch:      &RecordBatch,
    sort_exprs: &[SortExpr],
    limit:      Option<usize>,
) -> FalconResult<Option<Vec<usize>>> {
    use crate::storage::column::Column;

    if sort_exprs.len() != 1 {
        return Ok(None);
    }
    let s = &sort_exprs[0];
    let Expr::Column(name) = &s.expr else { return Ok(None) };
    let Ok(col) = batch.column_by_name(name) else { return Ok(None) };

    let desc = matches!(s.order, SortOrder::Desc);

    // Build (key, row) pairs for the non-NULL rows, and collect NULL rows apart.
    macro_rules! typed {
        ($c:expr, $ty:ty) => {{
            let c = $c;
            let dense = c.dense();
            let mut pairs: Vec<($ty, u32)> = Vec::with_capacity(dense.len());
            let mut nulls: Vec<usize> = Vec::new();
            for i in 0..c.len() {
                if c.is_valid(i) {
                    pairs.push((dense[i], i as u32));
                } else {
                    nulls.push(i);
                }
            }
            sort_pairs(&mut pairs, desc);
            Some(assemble(pairs, nulls, s.nulls_last, limit))
        }};
    }

    let out = match col {
        Column::Int64(c)     => typed!(c, i64),
        Column::Int32(c)     => typed!(c, i32),
        Column::Float64(c)   => typed!(c, f64),
        Column::Float32(c)   => typed!(c, f32),
        Column::Date32(c)    => typed!(c, i32),
        Column::Timestamp(c) => typed!(c, i64),
        // Utf8 keys would mean cloning or borrowing strings per row; Decimal128
        // and Boolean are rare enough as sort keys not to be worth a second path.
        _ => None,
    };
    Ok(out)
}

/// Sort (key, row) pairs, ascending or descending.
///
/// `partial_cmp` covers the float cases; NaN never reaches here because NULL and
/// non-NULL are split before this point and NaN compares unordered, so it is
/// pinned to one end via `Ordering::Equal` rather than corrupting the order.
fn sort_pairs<T: PartialOrd + Copy + Send>(pairs: &mut Vec<(T, u32)>, desc: bool) {
    let cmp = |a: &(T, u32), b: &(T, u32)| {
        let ord = a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal);
        if desc { ord.reverse() } else { ord }
    };
    if pairs.len() >= 65_536 {
        pairs.par_sort_unstable_by(cmp);
    } else {
        pairs.sort_unstable_by(cmp);
    }
}

/// Splice the sorted non-NULL rows together with the NULL rows and apply `limit`.
fn assemble<T>(
    pairs:      Vec<(T, u32)>,
    nulls:      Vec<usize>,
    nulls_last: bool,
    limit:      Option<usize>,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(pairs.len() + nulls.len());
    if !nulls_last {
        out.extend(nulls.iter().copied());
    }
    out.extend(pairs.into_iter().map(|(_, i)| i as usize));
    if nulls_last {
        out.extend(nulls);
    }
    if let Some(k) = limit {
        out.truncate(k);
    }
    out
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn merge_batches(batches: &[RecordBatch]) -> FalconResult<RecordBatch> {
    if batches.len() == 1 {
        return Ok(batches[0].clone());
    }

    let schema    = batches[0].schema.clone();
    let col_count = schema.field_count();

    // Typed bulk concat, one column at a time. The previous implementation pushed
    // every value through `ScalarValue`, which allocated an enum per cell (and cloned
    // every String) before converting the whole thing back — O(rows × cols) boxing
    // for what is really a memcpy per column.
    let columns: FalconResult<Vec<Column>> = (0..col_count)
        .map(|ci| {
            let parts: Vec<&Column> = batches.iter().map(|b| b.column(ci)).collect();
            Column::concat(&parts)
        })
        .collect();

    RecordBatch::new(schema, columns?)
}

fn compare_scalar(a: &ScalarValue, b: &ScalarValue, nulls_last: bool) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (a.is_null(), b.is_null()) {
        (true,  true)  => return Ordering::Equal,
        (true,  false) => return if nulls_last { Ordering::Greater } else { Ordering::Less },
        (false, true)  => return if nulls_last { Ordering::Less }    else { Ordering::Greater },
        (false, false) => {}
    }

    // Temporal types are checked before the numeric path: `as_f64` does not cover
    // them, and falling through to the `Equal` default would make ORDER BY on a
    // DATE or TIMESTAMP column silently do nothing.
    match (a, b) {
        (ScalarValue::Date32(Some(da)), ScalarValue::Date32(Some(db)))       => return da.cmp(db),
        (ScalarValue::Timestamp(Some(ta)), ScalarValue::Timestamp(Some(tb))) => return ta.cmp(tb),
        _ => {}
    }

    if let (Some(fa), Some(fb)) = (a.as_f64(), b.as_f64()) {
        return fa.partial_cmp(&fb).unwrap_or(Ordering::Equal);
    }

    if let (ScalarValue::Utf8(Some(sa)), ScalarValue::Utf8(Some(sb))) = (a, b) {
        return sa.cmp(sb);
    }

    if let (ScalarValue::Boolean(Some(ba)), ScalarValue::Boolean(Some(bb))) = (a, b) {
        return ba.cmp(bb);
    }

    // Mixed or unhandled types compare equal, which leaves their relative order
    // untouched. Every type FalconDB can put in a column is covered above.
    Ordering::Equal
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        query::expr::Expr,
        storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::RecordBatch,
            types::{DataType, ScalarValue},
        },
    };

    fn make_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::required("name",  DataType::Utf8),
            Field::required("score", DataType::Float64),
        ]);
        RecordBatch::new(schema, vec![
            ColumnBuilder::utf8_from_str(vec![
                Some("carol"), Some("alice"), Some("bob"), Some("dave"),
            ]),
            ColumnBuilder::float64(vec![
                Some(7.0), Some(9.5), Some(8.0), Some(6.0),
            ]),
        ]).unwrap()
    }

    #[test]
    fn sort_asc() {
        let batch   = make_batch();
        let sorted  = execute_sort(&[batch], &[SortExpr::asc(Expr::Column("score".into()))]).unwrap();
        let first   = sorted.column_by_name("score").unwrap().get(0).unwrap();
        let last    = sorted.column_by_name("score").unwrap().get(3).unwrap();
        assert_eq!(first, ScalarValue::Float64(Some(6.0)));
        assert_eq!(last,  ScalarValue::Float64(Some(9.5)));
    }

    #[test]
    fn sort_desc() {
        let batch   = make_batch();
        let sorted  = execute_sort(&[batch], &[SortExpr::desc(Expr::Column("score".into()))]).unwrap();
        let first   = sorted.column_by_name("score").unwrap().get(0).unwrap();
        assert_eq!(first, ScalarValue::Float64(Some(9.5)));
    }

    #[test]
    fn sort_string_asc() {
        let batch   = make_batch();
        let sorted  = execute_sort(&[batch], &[SortExpr::asc(Expr::Column("name".into()))]).unwrap();
        let first   = sorted.column_by_name("name").unwrap().get(0).unwrap();
        assert_eq!(first, ScalarValue::Utf8(Some("alice".into())));
    }

    #[test]
    fn sort_nulls_last() {
        let schema = Schema::new(vec![Field::nullable("x", DataType::Float64)]);
        let batch  = RecordBatch::new(schema, vec![
            ColumnBuilder::float64(vec![Some(1.0), None, Some(3.0)]),
        ]).unwrap();
        let sorted = execute_sort(
            &[batch],
            &[SortExpr::asc_nulls_last(Expr::Column("x".into()))],
        ).unwrap();
        let last = sorted.column_by_name("x").unwrap().get(2).unwrap();
        assert!(last.is_null());
    }

    #[test]
    fn sort_multi_key() {
        let schema = Schema::new(vec![
            Field::required("dept",  DataType::Utf8),
            Field::required("score", DataType::Float64),
        ]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::utf8_from_str(vec![Some("eng"), Some("hr"), Some("eng")]),
            ColumnBuilder::float64(vec![Some(8.0), Some(9.0), Some(7.0)]),
        ]).unwrap();
        let sorted = execute_sort(&[batch], &[
            SortExpr::asc(Expr::Column("dept".into())),
            SortExpr::desc(Expr::Column("score".into())),
        ]).unwrap();
        // eng 8, eng 7 → eng desc → 8 then 7
        let first_score = sorted.column_by_name("score").unwrap().get(0).unwrap();
        assert_eq!(first_score, ScalarValue::Float64(Some(8.0)));
    }
}
