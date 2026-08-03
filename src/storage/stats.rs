//! Zone maps: per-batch min/max/null summaries used to skip whole batches.
//!
//! A selective predicate usually rules out most batches outright. Keeping the
//! bounds of every column lets the scan prove "no row in this batch can match" and
//! move on without reading a single value — the classic columnar win, and the
//! reason analytical engines stay fast on filtered scans.
//!
//! Pruning is *conservative*: it may fail to skip a batch it could have skipped,
//! but it must never skip one that holds a matching row. Every rule below is
//! written to return `false` when anything is unclear.

use crate::{
    query::expr::{BinaryOp, Expr},
    storage::{
        column::Column,
        table::RecordBatch,
        types::ScalarValue,
    },
};

use std::collections::HashMap;

/// Bounds for one column within one batch.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub min:        Option<ScalarValue>,
    pub max:        Option<ScalarValue>,
    pub null_count: usize,
    pub row_count:  usize,
}

impl ColumnStats {
    /// Are every value in this column NULL?
    pub fn all_null(&self) -> bool {
        self.null_count == self.row_count
    }
}

/// Bounds for every column of a batch, keyed by column name.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub columns: HashMap<String, ColumnStats>,
}

impl BatchStats {
    pub fn compute(batch: &RecordBatch) -> Self {
        let mut columns = HashMap::with_capacity(batch.column_count());
        for (i, field) in batch.schema.fields.iter().enumerate() {
            columns.insert(field.name.clone(), column_stats(batch.column(i)));
        }
        Self { columns }
    }

    /// Can this batch be skipped entirely for `predicate`?
    ///
    /// Returns `true` only when no row can possibly satisfy it.
    pub fn can_skip(&self, predicate: &Expr) -> bool {
        match predicate {
            Expr::BinaryOp { left, op, right } => match op {
                // AND: ruling out either side rules out the conjunction.
                BinaryOp::And => self.can_skip(left) || self.can_skip(right),
                // OR: only skippable when both sides are impossible.
                BinaryOp::Or => self.can_skip(left) && self.can_skip(right),
                _ => self.can_skip_comparison(left, op, right),
            },
            _ => false,
        }
    }

    fn can_skip_comparison(&self, left: &Expr, op: &BinaryOp, right: &Expr) -> bool {
        // Normalise to `column <op> literal`, flipping the operator if the literal
        // is on the left.
        let (name, op, lit) = match (left, right) {
            (Expr::Column(n), Expr::Literal(v)) => (n, op.clone(), v),
            (Expr::Literal(v), Expr::Column(n)) => match flip(op) {
                Some(flipped) => (n, flipped, v),
                None => return false,
            },
            _ => return false,
        };

        let Some(stats) = self.columns.get(name.as_str()) else {
            return false;
        };
        // An all-NULL column never satisfies a comparison, but leave that to the
        // evaluator rather than adding a second rule here.
        let (Some(min), Some(max)) = (&stats.min, &stats.max) else {
            return false;
        };
        // Comparisons against NULL are never true, but again: not our job.
        if lit.is_null() {
            return false;
        }

        let (Some(lit_n), Some(min_n), Some(max_n)) = (cmp_key(lit), cmp_key(min), cmp_key(max))
        else {
            // Mixed or non-comparable types — refuse to prune.
            return false;
        };

        match op {
            // No value exceeds max, so `col > lit` is impossible once max <= lit.
            BinaryOp::Gt   => max_n <= lit_n,
            BinaryOp::GtEq => max_n <  lit_n,
            BinaryOp::Lt   => min_n >= lit_n,
            BinaryOp::LtEq => min_n >  lit_n,
            // The literal sits outside [min, max], so nothing equals it.
            BinaryOp::Eq   => lit_n < min_n || lit_n > max_n,
            // `!=` only fails when every value is the same and equal to the literal.
            BinaryOp::NotEq => min_n == max_n && min_n == lit_n && stats.null_count == 0,
            _ => false,
        }
    }
}

/// Mirror a comparison so the column ends up on the left.
fn flip(op: &BinaryOp) -> Option<BinaryOp> {
    Some(match op {
        BinaryOp::Gt    => BinaryOp::Lt,
        BinaryOp::Lt    => BinaryOp::Gt,
        BinaryOp::GtEq  => BinaryOp::LtEq,
        BinaryOp::LtEq  => BinaryOp::GtEq,
        BinaryOp::Eq    => BinaryOp::Eq,
        BinaryOp::NotEq => BinaryOp::NotEq,
        _ => return None,
    })
}

/// Order-preserving numeric key. Strings are handled separately because they are
/// not mutually comparable with numbers.
fn cmp_key(v: &ScalarValue) -> Option<OrdKey> {
    match v {
        ScalarValue::Utf8(Some(s))       => Some(OrdKey::Text(s.clone())),
        ScalarValue::Boolean(Some(b))    => Some(OrdKey::Num(if *b { 1.0 } else { 0.0 })),
        ScalarValue::Date32(Some(d))     => Some(OrdKey::Num(*d as f64)),
        ScalarValue::Timestamp(Some(ts)) => Some(OrdKey::Num(*ts as f64)),
        other => other.as_f64().map(OrdKey::Num),
    }
}

/// A comparable key that refuses to order text against numbers.
#[derive(Debug, PartialEq, PartialOrd)]
enum OrdKey {
    Num(f64),
    Text(String),
}

fn column_stats(col: &Column) -> ColumnStats {
    let row_count = col.len();

    macro_rules! numeric_stats {
        ($values:expr, $wrap:expr) => {{
            let mut min: Option<f64> = None;
            let mut max: Option<f64> = None;
            let mut nulls = 0usize;
            for v in $values {
                match v {
                    Some(x) => {
                        let x = x as f64;
                        min = Some(min.map_or(x, |m: f64| m.min(x)));
                        max = Some(max.map_or(x, |m: f64| m.max(x)));
                    }
                    None => nulls += 1,
                }
            }
            ColumnStats {
                min: min.map($wrap),
                max: max.map($wrap),
                null_count: nulls,
                row_count,
            }
        }};
    }

    match col {
        Column::Int64(c)   => numeric_stats!(c.iter(), |x: f64| ScalarValue::Int64(Some(x as i64))),
        Column::Int32(c)   => numeric_stats!(c.iter(), |x: f64| ScalarValue::Int64(Some(x as i64))),
        Column::Float64(c) => numeric_stats!(c.iter(), |x: f64| ScalarValue::Float64(Some(x))),
        Column::Float32(c) => numeric_stats!(c.iter(), |x: f64| ScalarValue::Float64(Some(x))),

        Column::Boolean(c) => {
            let mut any_true = false;
            let mut any_false = false;
            let mut nulls = 0usize;
            for v in &c.values {
                match v {
                    Some(true)  => any_true = true,
                    Some(false) => any_false = true,
                    None        => nulls += 1,
                }
            }
            ColumnStats {
                min: any_false.then(|| ScalarValue::Boolean(Some(false)))
                    .or_else(|| any_true.then(|| ScalarValue::Boolean(Some(true)))),
                max: any_true.then(|| ScalarValue::Boolean(Some(true)))
                    .or_else(|| any_false.then(|| ScalarValue::Boolean(Some(false)))),
                null_count: nulls,
                row_count,
            }
        }

        Column::Utf8(c) => {
            let mut min: Option<&str> = None;
            let mut max: Option<&str> = None;
            let mut nulls = 0usize;
            for v in c.iter() {
                match v {
                    Some(s) => {
                        if min.is_none_or(|m| s < m) { min = Some(s); }
                        if max.is_none_or(|m| s > m) { max = Some(s); }
                    }
                    None => nulls += 1,
                }
            }
            ColumnStats {
                min: min.map(|s| ScalarValue::Utf8(Some(s.to_string()))),
                max: max.map(|s| ScalarValue::Utf8(Some(s.to_string()))),
                null_count: nulls,
                row_count,
            }
        }

        Column::Date32(c) => {
            let mut mn: Option<i32> = None;
            let mut mx: Option<i32> = None;
            let mut nulls = 0usize;
            for v in c.iter() {
                match v {
                    Some(x) => {
                        mn = Some(mn.map_or(x, |m: i32| m.min(x)));
                        mx = Some(mx.map_or(x, |m: i32| m.max(x)));
                    }
                    None => nulls += 1,
                }
            }
            ColumnStats {
                min: mn.map(|v| ScalarValue::Date32(Some(v))),
                max: mx.map(|v| ScalarValue::Date32(Some(v))),
                null_count: nulls,
                row_count,
            }
        }

        Column::Timestamp(c) => {
            let mut mn: Option<i64> = None;
            let mut mx: Option<i64> = None;
            let mut nulls = 0usize;
            for v in c.iter() {
                match v {
                    Some(x) => {
                        mn = Some(mn.map_or(x, |m: i64| m.min(x)));
                        mx = Some(mx.map_or(x, |m: i64| m.max(x)));
                    }
                    None => nulls += 1,
                }
            }
            ColumnStats {
                min: mn.map(|v| ScalarValue::Timestamp(Some(v))),
                max: mx.map(|v| ScalarValue::Timestamp(Some(v))),
                null_count: nulls,
                row_count,
            }
        }

        Column::Decimal128(c) => {
            let mut mn: Option<i128> = None;
            let mut mx: Option<i128> = None;
            let mut nulls = 0usize;
            for v in c.inner.iter() {
                match v {
                    Some(x) => {
                        mn = Some(mn.map_or(x, |m: i128| m.min(x)));
                        mx = Some(mx.map_or(x, |m: i128| m.max(x)));
                    }
                    None => nulls += 1,
                }
            }
            let (p, s) = (c.precision, c.scale);
            ColumnStats {
                min: mn.map(|v| ScalarValue::decimal(Some(v), p, s)),
                max: mx.map(|v| ScalarValue::decimal(Some(v), p, s)),
                null_count: nulls,
                row_count,
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        types::DataType,
    };

    fn batch(scores: Vec<Option<f64>>) -> RecordBatch {
        let schema = Schema::new(vec![Field::nullable("score", DataType::Float64)]);
        RecordBatch::new(schema, vec![ColumnBuilder::float64(scores)]).unwrap()
    }

    fn lit(v: f64) -> Expr { Expr::Literal(ScalarValue::Float64(Some(v))) }

    fn cmp(op: BinaryOp, v: f64) -> Expr {
        Expr::BinaryOp {
            left:  Box::new(Expr::Column("score".into())),
            op,
            right: Box::new(lit(v)),
        }
    }

    #[test]
    fn computes_min_max_and_nulls() {
        let s = BatchStats::compute(&batch(vec![Some(3.0), None, Some(7.0)]));
        let c = &s.columns["score"];
        assert_eq!(c.min, Some(ScalarValue::Float64(Some(3.0))));
        assert_eq!(c.max, Some(ScalarValue::Float64(Some(7.0))));
        assert_eq!(c.null_count, 1);
    }

    #[test]
    fn skips_when_range_is_below_predicate() {
        let s = BatchStats::compute(&batch(vec![Some(1.0), Some(5.0)]));
        assert!(s.can_skip(&cmp(BinaryOp::Gt, 10.0)), "max 5 cannot exceed 10");
        assert!(!s.can_skip(&cmp(BinaryOp::Gt, 3.0)), "5 > 3 — must not skip");
    }

    #[test]
    fn skips_when_range_is_above_predicate() {
        let s = BatchStats::compute(&batch(vec![Some(20.0), Some(30.0)]));
        assert!(s.can_skip(&cmp(BinaryOp::Lt, 10.0)));
        assert!(!s.can_skip(&cmp(BinaryOp::Lt, 25.0)));
    }

    #[test]
    fn equality_outside_range_is_skippable() {
        let s = BatchStats::compute(&batch(vec![Some(1.0), Some(5.0)]));
        assert!(s.can_skip(&cmp(BinaryOp::Eq, 99.0)));
        assert!(!s.can_skip(&cmp(BinaryOp::Eq, 3.0)), "3 is in range — cannot rule out");
    }

    #[test]
    fn literal_on_the_left_is_handled() {
        let s = BatchStats::compute(&batch(vec![Some(1.0), Some(5.0)]));
        // 10 < score  ⇔  score > 10  → impossible
        let e = Expr::BinaryOp {
            left:  Box::new(lit(10.0)),
            op:    BinaryOp::Lt,
            right: Box::new(Expr::Column("score".into())),
        };
        assert!(s.can_skip(&e));
    }

    #[test]
    fn and_skips_if_either_side_is_impossible() {
        let s = BatchStats::compute(&batch(vec![Some(1.0), Some(5.0)]));
        let e = Expr::BinaryOp {
            left:  Box::new(cmp(BinaryOp::Gt, 0.0)),   // satisfiable
            op:    BinaryOp::And,
            right: Box::new(cmp(BinaryOp::Gt, 100.0)), // impossible
        };
        assert!(s.can_skip(&e));
    }

    #[test]
    fn or_skips_only_if_both_sides_are_impossible() {
        let s = BatchStats::compute(&batch(vec![Some(1.0), Some(5.0)]));
        let one_ok = Expr::BinaryOp {
            left:  Box::new(cmp(BinaryOp::Gt, 0.0)),
            op:    BinaryOp::Or,
            right: Box::new(cmp(BinaryOp::Gt, 100.0)),
        };
        assert!(!s.can_skip(&one_ok));

        let both_bad = Expr::BinaryOp {
            left:  Box::new(cmp(BinaryOp::Gt, 100.0)),
            op:    BinaryOp::Or,
            right: Box::new(cmp(BinaryOp::Lt, -100.0)),
        };
        assert!(s.can_skip(&both_bad));
    }

    #[test]
    fn unknown_column_never_prunes() {
        let s = BatchStats::compute(&batch(vec![Some(1.0)]));
        let e = Expr::BinaryOp {
            left:  Box::new(Expr::Column("missing".into())),
            op:    BinaryOp::Gt,
            right: Box::new(lit(100.0)),
        };
        assert!(!s.can_skip(&e));
    }

    #[test]
    fn text_is_not_compared_against_numbers() {
        let schema = Schema::new(vec![Field::nullable("name", DataType::Utf8)]);
        let b = RecordBatch::new(
            schema,
            vec![ColumnBuilder::utf8_from_str(vec![Some("alice"), Some("bob")])],
        )
        .unwrap();
        let s = BatchStats::compute(&b);
        let e = Expr::BinaryOp {
            left:  Box::new(Expr::Column("name".into())),
            op:    BinaryOp::Gt,
            right: Box::new(lit(5.0)),
        };
        assert!(!s.can_skip(&e), "comparing text to a number must not prune");
    }

    #[test]
    fn string_range_prunes() {
        let schema = Schema::new(vec![Field::nullable("name", DataType::Utf8)]);
        let b = RecordBatch::new(
            schema,
            vec![ColumnBuilder::utf8_from_str(vec![Some("alice"), Some("bob")])],
        )
        .unwrap();
        let s = BatchStats::compute(&b);
        let e = Expr::BinaryOp {
            left:  Box::new(Expr::Column("name".into())),
            op:    BinaryOp::Eq,
            right: Box::new(Expr::Literal(ScalarValue::Utf8(Some("zoe".into())))),
        };
        assert!(s.can_skip(&e), "'zoe' sorts past 'bob'");
    }

    #[test]
    fn all_null_column_does_not_prune_incorrectly() {
        let s = BatchStats::compute(&batch(vec![None, None]));
        // No bounds exist, so nothing can be proven — must not skip.
        assert!(!s.can_skip(&cmp(BinaryOp::Gt, 10.0)));
        assert!(s.columns["score"].all_null());
    }
}
