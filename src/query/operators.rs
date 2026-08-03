// Concrete operators for the push-based pipeline.
//
// Each one delegates to the same routine the existing executor calls, so the two
// paths cannot drift in what they compute — only in when and how much they
// materialise. That is deliberate: the point of this work is to remove
// intermediate allocation, not to change any answer.
//
// Streaming operators (Filter, Project, Limit) turn one chunk into at most one
// chunk. Blocking ones (Sort, Aggregate, Distinct) have to see every row before
// they can answer, so they accumulate in `process` and produce in `flush`.

use crate::storage::{
    table::RecordBatch,
    types::FalconResult,
};

use super::{
    aggregate::GroupByExec,
    expr::Expr,
    pipeline::{OpResult, Operator},
    select::{project_batch, SelectExpr},
    sort::{execute_sort_limit, SortExpr},
};

// ─── Filter ───────────────────────────────────────────────────────────────────

/// `WHERE` — drops rows that fail the predicate.
pub struct FilterOp {
    predicate: Expr,
}

impl FilterOp {
    pub fn new(predicate: Expr) -> Self {
        Self { predicate }
    }
}

impl Operator for FilterOp {
    fn process(&mut self, chunk: RecordBatch) -> FalconResult<OpResult> {
        let mask = self.predicate.eval_mask_bits(&chunk)?;
        let out  = chunk.filter_by_bitmask(&mask)?;
        // An empty chunk carries no rows, so there is nothing to pass on.
        Ok(if out.row_count() == 0 { OpResult::Empty } else { OpResult::Emit(out) })
    }

    fn name(&self) -> &'static str { "Filter" }
}

// ─── Project ──────────────────────────────────────────────────────────────────

/// `SELECT` column list — evaluates expressions and renames columns.
pub struct ProjectOp {
    exprs: Vec<SelectExpr>,
}

impl ProjectOp {
    pub fn new(exprs: Vec<SelectExpr>) -> Self {
        Self { exprs }
    }
}

impl Operator for ProjectOp {
    fn process(&mut self, chunk: RecordBatch) -> FalconResult<OpResult> {
        Ok(OpResult::Emit(project_batch(&chunk, &self.exprs)?))
    }

    fn name(&self) -> &'static str { "Project" }
}

// ─── Limit / Offset ───────────────────────────────────────────────────────────

/// `LIMIT` / `OFFSET`.
///
/// Reports `Done` once the limit is met, which is what lets the source stop
/// reading instead of scanning a table it will discard.
pub struct LimitOp {
    /// Rows still to skip.
    skip: usize,
    /// Rows still to emit; `None` means unbounded.
    take: Option<usize>,
}

impl LimitOp {
    pub fn new(offset: usize, limit: Option<usize>) -> Self {
        Self { skip: offset, take: limit }
    }
}

impl Operator for LimitOp {
    fn process(&mut self, chunk: RecordBatch) -> FalconResult<OpResult> {
        if self.take == Some(0) {
            return Ok(OpResult::Done);
        }

        let n = chunk.row_count();

        // Still inside the OFFSET window.
        if self.skip >= n {
            self.skip -= n;
            return Ok(OpResult::Empty);
        }

        let start     = self.skip;
        self.skip     = 0;
        let available = n - start;

        let take = match self.take {
            None    => available,
            Some(k) => k.min(available),
        };
        if let Some(k) = self.take.as_mut() {
            *k -= take;
        }

        if take == 0 {
            return Ok(OpResult::Done);
        }
        Ok(OpResult::Emit(chunk.slice(start, take)?))
    }

    fn name(&self) -> &'static str { "Limit" }
}

// ─── Sort ─────────────────────────────────────────────────────────────────────

/// `ORDER BY` — blocking: the smallest row may arrive last.
pub struct SortOp {
    keys:  Vec<SortExpr>,
    /// Top-N bound, when the query has a LIMIT the sort can honour itself.
    limit: Option<usize>,
    held:  Vec<RecordBatch>,
}

impl SortOp {
    pub fn new(keys: Vec<SortExpr>, limit: Option<usize>) -> Self {
        Self { keys, limit, held: Vec::new() }
    }
}

impl Operator for SortOp {
    fn process(&mut self, chunk: RecordBatch) -> FalconResult<OpResult> {
        self.held.push(chunk);
        Ok(OpResult::Empty)
    }

    fn flush(&mut self) -> FalconResult<Vec<RecordBatch>> {
        if self.held.is_empty() {
            return Ok(Vec::new());
        }
        let held = std::mem::take(&mut self.held);
        Ok(vec![execute_sort_limit(&held, &self.keys, self.limit)?])
    }

    fn name(&self) -> &'static str { "Sort" }
}

// ─── Aggregate ────────────────────────────────────────────────────────────────

/// `GROUP BY` with aggregates — blocking.
///
/// Chunks are held and aggregated together at flush rather than folded
/// incrementally. Partial-state merging across chunks is the next step here; the
/// interface already allows it, and doing it now would mean two implementations
/// of the same arithmetic to keep in agreement.
pub struct AggregateOp {
    exec: GroupByExec,
    held: Vec<RecordBatch>,
}

impl AggregateOp {
    pub fn new(exec: GroupByExec) -> Self {
        Self { exec, held: Vec::new() }
    }
}

impl Operator for AggregateOp {
    fn process(&mut self, chunk: RecordBatch) -> FalconResult<OpResult> {
        self.held.push(chunk);
        Ok(OpResult::Empty)
    }

    fn flush(&mut self) -> FalconResult<Vec<RecordBatch>> {
        if self.held.is_empty() {
            return Ok(Vec::new());
        }
        let held = std::mem::take(&mut self.held);
        Ok(vec![self.exec.execute(&held)?])
    }

    fn name(&self) -> &'static str { "Aggregate" }
}

// ─── Distinct ─────────────────────────────────────────────────────────────────

/// `SELECT DISTINCT` — blocking, because a duplicate may arrive in any chunk.
pub struct DistinctOp {
    held: Vec<RecordBatch>,
}

impl DistinctOp {
    pub fn new() -> Self {
        Self { held: Vec::new() }
    }
}

impl Default for DistinctOp {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for DistinctOp {
    fn process(&mut self, chunk: RecordBatch) -> FalconResult<OpResult> {
        self.held.push(chunk);
        Ok(OpResult::Empty)
    }

    fn flush(&mut self) -> FalconResult<Vec<RecordBatch>> {
        let held = std::mem::take(&mut self.held);
        Ok(super::select::dedup_batches(held))
    }

    fn name(&self) -> &'static str { "Distinct" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{
        expr::BinaryOp,
        pipeline::Pipeline,
    };
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        types::{DataType, ScalarValue},
    };

    fn batch(vals: Vec<i64>) -> RecordBatch {
        let schema = Schema::new(vec![Field::required("v", DataType::Int64)]);
        RecordBatch::new(schema, vec![ColumnBuilder::int64(vals.into_iter().map(Some).collect())])
            .unwrap()
    }

    fn values(batches: &[RecordBatch]) -> Vec<i64> {
        let mut out = Vec::new();
        for b in batches {
            let c = b.column(0);
            for r in 0..b.row_count() {
                out.push(c.get(r).unwrap().as_i64().unwrap());
            }
        }
        out
    }

    fn gt(n: i64) -> Expr {
        Expr::BinaryOp {
            left:  Box::new(Expr::Column("v".into())),
            op:    BinaryOp::Gt,
            right: Box::new(Expr::Literal(ScalarValue::Int64(Some(n)))),
        }
    }

    #[test]
    fn filter_keeps_matching_rows() {
        let mut p = Pipeline::new(vec![Box::new(FilterOp::new(gt(4_990)))]);
        let got = p.run(&[batch((0..5_000).collect())]).unwrap();
        assert_eq!(values(&got), (4_991..5_000).collect::<Vec<_>>());
    }

    #[test]
    fn filter_that_matches_nothing_yields_nothing() {
        let mut p = Pipeline::new(vec![Box::new(FilterOp::new(gt(999_999)))]);
        assert!(p.run(&[batch((0..5_000).collect())]).unwrap().is_empty());
    }

    #[test]
    fn limit_takes_the_first_rows() {
        let mut p = Pipeline::new(vec![Box::new(LimitOp::new(0, Some(7)))]);
        let got = p.run(&[batch((0..5_000).collect())]).unwrap();
        assert_eq!(values(&got), (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn offset_skips_across_a_chunk_boundary() {
        // 3000 > CHUNK_ROWS, so the offset has to be carried between chunks.
        let mut p = Pipeline::new(vec![Box::new(LimitOp::new(3_000, Some(5)))]);
        let got = p.run(&[batch((0..5_000).collect())]).unwrap();
        assert_eq!(values(&got), (3_000..3_005).collect::<Vec<_>>());
    }

    #[test]
    fn offset_past_the_end_yields_nothing() {
        let mut p = Pipeline::new(vec![Box::new(LimitOp::new(10_000, None))]);
        assert!(p.run(&[batch((0..5_000).collect())]).unwrap().is_empty());
    }

    #[test]
    fn limit_zero_yields_nothing() {
        let mut p = Pipeline::new(vec![Box::new(LimitOp::new(0, Some(0)))]);
        assert!(p.run(&[batch((0..100).collect())]).unwrap().is_empty());
    }

    #[test]
    fn filter_then_limit_compose() {
        let mut p = Pipeline::new(vec![
            Box::new(FilterOp::new(gt(100))),
            Box::new(LimitOp::new(0, Some(3))),
        ]);
        let got = p.run(&[batch((0..5_000).collect())]).unwrap();
        assert_eq!(values(&got), vec![101, 102, 103]);
    }

    #[test]
    fn sort_orders_across_chunks() {
        let mut p = Pipeline::new(vec![Box::new(SortOp::new(
            vec![SortExpr::asc(Expr::Column("v".into()))],
            None,
        ))]);
        let got = p.run(&[batch((0..5_000).rev().collect())]).unwrap();
        assert_eq!(values(&got), (0..5_000).collect::<Vec<_>>());
    }

    #[test]
    fn sort_with_a_top_n_bound() {
        let mut p = Pipeline::new(vec![Box::new(SortOp::new(
            vec![SortExpr::asc(Expr::Column("v".into()))],
            Some(4),
        ))]);
        let got = p.run(&[batch((0..5_000).rev().collect())]).unwrap();
        assert_eq!(values(&got), vec![0, 1, 2, 3]);
    }

    #[test]
    fn filter_then_sort_compose() {
        let mut p = Pipeline::new(vec![
            Box::new(FilterOp::new(gt(4_995))),
            Box::new(SortOp::new(vec![SortExpr::asc(Expr::Column("v".into()))], None)),
        ]);
        let got = p.run(&[batch((0..5_000).rev().collect())]).unwrap();
        assert_eq!(values(&got), vec![4_996, 4_997, 4_998, 4_999]);
    }

    #[test]
    fn distinct_removes_duplicates_across_chunks() {
        let mut p = Pipeline::new(vec![Box::new(DistinctOp::new())]);
        // Values repeat every 100, well inside a chunk and also across chunks.
        let got = p.run(&[batch((0..5_000).map(|i| i % 100).collect())]).unwrap();
        let mut vals = values(&got);
        vals.sort();
        assert_eq!(vals, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn sort_then_limit_after_a_blocking_stage() {
        // Guards the driver's flush ordering: the sort's output has to travel
        // through the limit before the limit is itself flushed.
        let mut p = Pipeline::new(vec![
            Box::new(SortOp::new(vec![SortExpr::asc(Expr::Column("v".into()))], None)),
            Box::new(LimitOp::new(0, Some(3))),
        ]);
        let got = p.run(&[batch((0..5_000).rev().collect())]).unwrap();
        assert_eq!(values(&got), vec![0, 1, 2]);
    }
}
