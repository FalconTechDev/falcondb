// Query optimizer — rewrites the expression tree before execution.
//
// Implemented passes:
//   1. Constant folding      — evaluate constant sub-expressions at plan time
//   2. Predicate pushdown    — move cheap filters before expensive ones
//   3. Null-aware pruning    — skip batches whose null_count == row_count
//   4. Expression ordering   — put cheaper predicates first in AND chains

use crate::storage::{table::RecordBatch, types::ScalarValue};
use super::expr::{BinaryOp, Expr};

// ─── Constant folding ─────────────────────────────────────────────────────────

/// Recursively fold constant sub-expressions into literals.
///
/// Examples:
///   1 + 2          → Literal(3)
///   col > 1 + 1    → col > Literal(2)
///   NOT NOT col    → col (double-negation)
pub fn fold_constants(expr: Expr) -> Expr {
    match expr {
        Expr::BinaryOp { left, op, right } => {
            let l = fold_constants(*left);
            let r = fold_constants(*right);

            // Both sides are literals → evaluate now.
            if let (Expr::Literal(lv), Expr::Literal(rv)) = (&l, &r) {
                if let Ok(result) = apply_binary_literal(lv, &op, rv) {
                    return Expr::Literal(result);
                }
            }

            Expr::BinaryOp { left: Box::new(l), op, right: Box::new(r) }
        }

        Expr::Not(inner) => {
            let folded = fold_constants(*inner);
            // NOT NOT x → x
            if let Expr::Not(inner2) = folded {
                return *inner2;
            }
            // NOT true → false, NOT false → true
            if let Expr::Literal(ScalarValue::Boolean(Some(b))) = folded {
                return Expr::Literal(ScalarValue::Boolean(Some(!b)));
            }
            Expr::Not(Box::new(folded))
        }

        Expr::IsNull(inner) => {
            let folded = fold_constants(*inner);
            if let Expr::Literal(v) = &folded {
                return Expr::Literal(ScalarValue::Boolean(Some(v.is_null())));
            }
            Expr::IsNull(Box::new(folded))
        }

        Expr::IsNotNull(inner) => {
            let folded = fold_constants(*inner);
            if let Expr::Literal(v) = &folded {
                return Expr::Literal(ScalarValue::Boolean(Some(!v.is_null())));
            }
            Expr::IsNotNull(Box::new(folded))
        }

        // New expression types — fold their sub-expressions but don't simplify further.
        Expr::Like { expr, pattern, negated, case_insensitive } => Expr::Like {
            expr: Box::new(fold_constants(*expr)),
            pattern, negated, case_insensitive,
        },
        Expr::InList { expr, list, negated } => Expr::InList {
            expr: Box::new(fold_constants(*expr)),
            list: list.into_iter().map(fold_constants).collect(),
            negated,
        },
        Expr::Case { when_thens, else_expr } => Expr::Case {
            when_thens: when_thens.into_iter()
                .map(|(c, r)| (fold_constants(c), fold_constants(r)))
                .collect(),
            else_expr: else_expr.map(|e| Box::new(fold_constants(*e))),
        },
        Expr::Function { name, args } => Expr::Function {
            name,
            args: args.into_iter().map(fold_constants).collect(),
        },
        Expr::Cast { expr, to_type } => Expr::Cast {
            expr: Box::new(fold_constants(*expr)),
            to_type,
        },

        other => other,
    }
}

fn apply_binary_literal(
    l:  &ScalarValue,
    op: &BinaryOp,
    r:  &ScalarValue,
) -> Result<ScalarValue, ()> {
    if l.is_null() || r.is_null() {
        return Ok(ScalarValue::Null);
    }

    match op {
        BinaryOp::And => {
            let a = l.as_bool().ok_or(())?;
            let b = r.as_bool().ok_or(())?;
            return Ok(ScalarValue::Boolean(Some(a && b)));
        }
        BinaryOp::Or => {
            let a = l.as_bool().ok_or(())?;
            let b = r.as_bool().ok_or(())?;
            return Ok(ScalarValue::Boolean(Some(a || b)));
        }
        _ => {}
    }

    if let (Some(a), Some(b)) = (l.as_f64(), r.as_f64()) {
        return Ok(match op {
            BinaryOp::Add  => ScalarValue::Float64(Some(a + b)),
            BinaryOp::Sub  => ScalarValue::Float64(Some(a - b)),
            BinaryOp::Mul  => ScalarValue::Float64(Some(a * b)),
            BinaryOp::Div  => {
                if b == 0.0 { return Err(()); }
                ScalarValue::Float64(Some(a / b))
            }
            BinaryOp::Eq    => ScalarValue::Boolean(Some((a - b).abs() < f64::EPSILON)),
            BinaryOp::NotEq => ScalarValue::Boolean(Some((a - b).abs() >= f64::EPSILON)),
            BinaryOp::Lt    => ScalarValue::Boolean(Some(a < b)),
            BinaryOp::LtEq  => ScalarValue::Boolean(Some(a <= b)),
            BinaryOp::Gt    => ScalarValue::Boolean(Some(a > b)),
            BinaryOp::GtEq  => ScalarValue::Boolean(Some(a >= b)),
            _ => return Err(()),
        });
    }

    if let (ScalarValue::Utf8(Some(a)), ScalarValue::Utf8(Some(b))) = (l, r) {
        return Ok(match op {
            BinaryOp::Eq    => ScalarValue::Boolean(Some(a == b)),
            BinaryOp::NotEq => ScalarValue::Boolean(Some(a != b)),
            _ => return Err(()),
        });
    }

    Err(())
}

// ─── Predicate ordering ───────────────────────────────────────────────────────

/// Estimated cost of evaluating an expression per row (lower = cheaper).
/// Used to put cheap predicates first in AND chains.
pub fn expr_cost(expr: &Expr) -> u32 {
    match expr {
        Expr::Literal(_)   => 0,
        Expr::Column(_)    => 1,
        // By evaluation time a subquery has already been folded into a literal,
        // so ordering never actually sees one. Rank it as free rather than let a
        // stray node change the ordering.
        Expr::Subquery(_)  => 0,
        Expr::InSubquery { .. } => 0,
        Expr::Exists { .. }     => 0,
        Expr::IsNull(_)    => 2,
        Expr::IsNotNull(_) => 2,
        Expr::Not(e)       => 1 + expr_cost(e),
        Expr::Like { expr, .. }  => 5 + expr_cost(expr),
        Expr::InList { expr, list, .. } => {
            3 + expr_cost(expr) + list.len() as u32
        }
        Expr::Case { when_thens, else_expr } => {
            let wt: u32 = when_thens.iter().map(|(c, r)| expr_cost(c) + expr_cost(r)).sum();
            let el = else_expr.as_ref().map_or(0, |e| expr_cost(e));
            10 + wt + el
        }
        Expr::Function { args, .. } => {
            5 + args.iter().map(expr_cost).sum::<u32>()
        }
        Expr::Cast { expr, .. }    => 3 + expr_cost(expr),
        Expr::Extract { expr, .. } => 2 + expr_cost(expr),
        Expr::BinaryOp { left, op, right } => {
            let base = match op {
                BinaryOp::And | BinaryOp::Or => 0,
                BinaryOp::Eq | BinaryOp::NotEq => 2,
                BinaryOp::Lt | BinaryOp::LtEq |
                BinaryOp::Gt | BinaryOp::GtEq => 2,
                BinaryOp::Add | BinaryOp::Sub  => 3,
                BinaryOp::Mul                  => 4,
                BinaryOp::Div                  => 8,
            };
            base + expr_cost(left) + expr_cost(right)
        }
    }
}

/// Flatten a nested AND chain into a list of predicates, then sort by cost.
pub fn order_predicates(expr: Expr) -> Expr {
    let mut preds = flatten_and(expr);
    preds.sort_by_key(|e| expr_cost(e));
    rebuild_and(preds)
}

fn flatten_and(expr: Expr) -> Vec<Expr> {
    match expr {
        Expr::BinaryOp { left, op: BinaryOp::And, right } => {
            let mut l = flatten_and(*left);
            l.extend(flatten_and(*right));
            l
        }
        other => vec![other],
    }
}

/// Drop conjuncts that repeat one already in the list.
///
/// `WHERE a > 5 AND b < 2 AND a > 5` evaluates `a > 5` twice, once per row, for
/// no gain. Written-out queries rarely look like that, but generated SQL and
/// predicates assembled by pushdown routinely do.
///
/// Only structurally identical conjuncts are removed. Proving that `a > 5` and
/// `5 < a` mean the same thing needs a normal form this optimiser does not have,
/// and guessing would change results.
pub fn dedup_predicates(expr: Expr) -> Expr {
    let preds = flatten_and(expr);
    if preds.len() < 2 {
        return rebuild_and(preds);
    }

    let mut seen: Vec<Expr> = Vec::with_capacity(preds.len());
    for p in preds {
        // `Expr` implements structural PartialEq; subquery nodes compare only to
        // themselves, so nothing is merged across a subquery boundary.
        if !seen.contains(&p) {
            seen.push(p);
        }
    }
    rebuild_and(seen)
}

fn rebuild_and(mut preds: Vec<Expr>) -> Expr {
    assert!(!preds.is_empty());
    let mut result = preds.remove(0);
    for p in preds {
        result = Expr::BinaryOp {
            left:  Box::new(result),
            op:    BinaryOp::And,
            right: Box::new(p),
        };
    }
    result
}

// ─── Batch pruning ────────────────────────────────────────────────────────────

/// Return true if this batch can be skipped entirely based on column statistics.
///
/// Currently detects:
///   - Column is all-NULL and predicate requires non-NULL values.
pub fn can_prune_batch(batch: &RecordBatch, filter: &Expr) -> bool {
    prune_check(batch, filter).unwrap_or(false)
}

fn prune_check(batch: &RecordBatch, expr: &Expr) -> Option<bool> {
    match expr {
        // If we need a column to be non-null but it's all null → prune.
        Expr::IsNull(inner) => {
            if let Expr::Column(name) = inner.as_ref() {
                let col = batch.column_by_name(name).ok()?;
                // IS NULL on an all-null column → every row matches → don't prune.
                if col.null_count() == col.len() { return Some(false); }
            }
            None
        }

        Expr::IsNotNull(inner) => {
            if let Expr::Column(name) = inner.as_ref() {
                let col = batch.column_by_name(name).ok()?;
                // IS NOT NULL on an all-null column → no row matches → prune.
                if col.null_count() == col.len() { return Some(true); }
            }
            None
        }

        // For numeric comparisons on all-null columns → no row matches → prune.
        Expr::BinaryOp { left, op, right } => {
            let is_comparison = matches!(
                op,
                BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt |
                BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq
            );
            if is_comparison {
                if let Expr::Column(name) = left.as_ref() {
                    if let Ok(col) = batch.column_by_name(name) {
                        if col.null_count() == col.len() { return Some(true); }
                    }
                }
                if let Expr::Column(name) = right.as_ref() {
                    if let Ok(col) = batch.column_by_name(name) {
                        if col.null_count() == col.len() { return Some(true); }
                    }
                }
            }
            None
        }

        Expr::Like { expr, .. } | Expr::InList { expr, .. } => {
            if let Expr::Column(name) = expr.as_ref() {
                if let Ok(col) = batch.column_by_name(name) {
                    if col.null_count() == col.len() { return Some(true); }
                }
            }
            None
        }
        _ => None,
    }
}

// ─── Join reordering (cost-based) ────────────────────────────────────────────

/// Reorder joins to minimize estimated intermediate result sizes.
///
/// Strategy: greedily join the smallest table first. Estimate each table's size
/// from its row count (available after loading); if the table isn't in the
/// registry yet, fall back to the order in which joins appear in the SQL.
///
/// This is a heuristic, not a full-blown DP planner. It avoids the exponential
/// blowup of N! orderings while still catching the common case where users write
/// JOINs in non-optimal order.
pub fn reorder_joins<J>(
    joins:      Vec<J>,
    row_count:  impl Fn(&str) -> Option<usize>,
) -> Vec<J>
where
    J: HasJoinTable,
{
    let mut indexed: Vec<(usize, J)> = joins.into_iter().enumerate().collect();

    // Sort by ascending estimated right-side row count (smallest tables first).
    // Stable sort preserves original order when counts are equal or unknown.
    indexed.sort_by(|(_, a), (_, b)| {
        let ca = row_count(a.table_name()).unwrap_or(usize::MAX);
        let cb = row_count(b.table_name()).unwrap_or(usize::MAX);
        ca.cmp(&cb)
    });

    indexed.into_iter().map(|(_, j)| j).collect()
}

pub trait HasJoinTable {
    fn table_name(&self) -> &str;
}

impl HasJoinTable for crate::query::sql::JoinClause {
    fn table_name(&self) -> &str { &self.table }
}

// ─── Full optimization pipeline ───────────────────────────────────────────────

pub fn optimize_filter(expr: Expr) -> Expr {
    let folded = fold_constants(expr);
    // Deduplicate before ordering: two identical conjuncts have identical cost,
    // so removing one after sorting would be a coin flip over which survives.
    let deduped = dedup_predicates(folded);
    order_predicates(deduped)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        query::expr::{BinaryOp, Expr},
        storage::types::ScalarValue,
    };

    fn lit(v: i64) -> Expr { Expr::Literal(ScalarValue::Int64(Some(v))) }
    fn litf(v: f64) -> Expr { Expr::Literal(ScalarValue::Float64(Some(v))) }

    #[test]
    fn fold_addition() {
        let e = Expr::BinaryOp {
            left: Box::new(lit(2)),
            op:   BinaryOp::Add,
            right: Box::new(lit(3)),
        };
        let r = fold_constants(e);
        assert!(matches!(r, Expr::Literal(ScalarValue::Float64(Some(v))) if (v - 5.0).abs() < 1e-10));
    }

    #[test]
    fn fold_nested() {
        // (1 + 2) * 4 → 12
        let inner = Expr::BinaryOp {
            left:  Box::new(lit(1)),
            op:    BinaryOp::Add,
            right: Box::new(lit(2)),
        };
        let outer = Expr::BinaryOp {
            left:  Box::new(inner),
            op:    BinaryOp::Mul,
            right: Box::new(lit(4)),
        };
        let r = fold_constants(outer);
        if let Expr::Literal(ScalarValue::Float64(Some(v))) = r {
            assert!((v - 12.0).abs() < 1e-10);
        } else {
            panic!("expected Literal(12)");
        }
    }

    #[test]
    fn fold_double_negation() {
        let e = Expr::Not(Box::new(Expr::Not(Box::new(Expr::Column("x".into())))));
        let r = fold_constants(e);
        assert!(matches!(r, Expr::Column(_)));
    }

    #[test]
    fn fold_not_bool() {
        let e = Expr::Not(Box::new(Expr::Literal(ScalarValue::Boolean(Some(true)))));
        let r = fold_constants(e);
        assert_eq!(r, Expr::Literal(ScalarValue::Boolean(Some(false))));
    }

    #[test]
    fn fold_is_null_literal() {
        let e = Expr::IsNull(Box::new(Expr::Literal(ScalarValue::Null)));
        let r = fold_constants(e);
        assert_eq!(r, Expr::Literal(ScalarValue::Boolean(Some(true))));
    }

    #[test]
    fn predicate_ordering() {
        // A division (cost=8+) before a column check (cost=1) should be reordered.
        let expensive = Expr::BinaryOp {
            left:  Box::new(Expr::Column("a".into())),
            op:    BinaryOp::Div,
            right: Box::new(Expr::Literal(ScalarValue::Float64(Some(2.0)))),
        };
        let cheap = Expr::Column("b".into());
        let combined = Expr::BinaryOp {
            left:  Box::new(expensive),
            op:    BinaryOp::And,
            right: Box::new(cheap),
        };
        let optimized = order_predicates(combined);
        // The first predicate in the AND chain should be the cheaper one.
        if let Expr::BinaryOp { left, .. } = optimized {
            assert!(expr_cost(&left) <= 4, "left should be cheaper");
        }
    }

    #[test]
    fn prune_all_null_column() {
        use crate::storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::RecordBatch,
            types::DataType,
        };
        let schema = Schema::new(vec![Field::nullable("x", DataType::Int64)]);
        let batch  = RecordBatch::new(schema, vec![
            ColumnBuilder::int64(vec![None, None, None]),
        ]).unwrap();

        // x > 5 on all-null column → prune
        let filter = Expr::BinaryOp {
            left:  Box::new(Expr::Column("x".into())),
            op:    BinaryOp::Gt,
            right: Box::new(Expr::Literal(ScalarValue::Int64(Some(5)))),
        };
        assert!(can_prune_batch(&batch, &filter));
    }

    #[test]
    fn no_prune_when_data_exists() {
        use crate::storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::RecordBatch,
            types::DataType,
        };
        let schema = Schema::new(vec![Field::required("x", DataType::Int64)]);
        let batch  = RecordBatch::new(schema, vec![
            ColumnBuilder::int64(vec![Some(1), Some(2)]),
        ]).unwrap();

        let filter = Expr::BinaryOp {
            left:  Box::new(Expr::Column("x".into())),
            op:    BinaryOp::Gt,
            right: Box::new(Expr::Literal(ScalarValue::Int64(Some(0)))),
        };
        assert!(!can_prune_batch(&batch, &filter));
    }
}
