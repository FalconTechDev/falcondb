use std::fmt;

use rayon::prelude::*;

use crate::storage::{
    table::RecordBatch,
    types::{FalconError, FalconResult, ScalarValue},
};

/// Minimum number of u64 mask words (= rows / 64) to justify parallel mask computation.
/// 4096 words = 262,144 rows — below this, rayon scheduling overhead dominates.
const PAR_MASK_WORDS: usize = 4096;

/// Wrapper so `Expr` can keep deriving `Debug`/`PartialEq` without requiring the
/// whole plan tree to implement them.
#[derive(Clone)]
pub struct SubqueryPlan(pub Box<crate::query::sql::ParsedQuery>);

impl std::fmt::Debug for SubqueryPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<scalar subquery on '{}'>", self.0.table)
    }
}

impl PartialEq for SubqueryPlan {
    /// Plans are not structurally comparable, so a subquery node is only equal to
    /// itself. This is deliberately conservative: passes that use `==` to fold or
    /// dedupe expressions will simply never merge across subqueries.
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0.as_ref(), other.0.as_ref())
    }
}

/// The expression tree — the core of the query engine.
/// Every WHERE clause, projection, and computed column is an Expr.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(ScalarValue),
    Column(String),

    /// A scalar subquery: `(SELECT AVG(x) FROM t)`.
    ///
    /// It is *not* evaluated by `eval_batch` — the executor runs it once up front
    /// and rewrites the node into a `Literal`, because a subquery has no per-row
    /// meaning and re-running it for every batch would be wasteful. Evaluating one
    /// directly is a bug, and `eval_batch` says so rather than guessing.
    Subquery(SubqueryPlan),

    /// `x IN (SELECT …)` / `x NOT IN (SELECT …)`.
    /// Folded into an `InList` of literals before evaluation.
    InSubquery {
        expr:    Box<Expr>,
        plan:    SubqueryPlan,
        negated: bool,
    },

    /// `EXISTS (SELECT …)` / `NOT EXISTS (…)`.
    /// Folded into a boolean literal before evaluation — with no correlation the
    /// answer is the same for every row.
    Exists {
        plan:    SubqueryPlan,
        negated: bool,
    },
    BinaryOp { left: Box<Expr>, op: BinaryOp, right: Box<Expr> },
    Not(Box<Expr>),
    IsNull(Box<Expr>),
    IsNotNull(Box<Expr>),

    /// SQL LIKE / ILIKE pattern match.  `%` = any run, `_` = one char.
    Like {
        expr:              Box<Expr>,
        pattern:           String,
        negated:           bool,
        case_insensitive:  bool,
    },

    /// `expr IN (v1, v2, ...)` or `expr NOT IN (...)`.
    InList {
        expr:    Box<Expr>,
        list:    Vec<Expr>,
        negated: bool,
    },

    /// `CASE WHEN c1 THEN r1 … ELSE e END`.
    Case {
        when_thens: Vec<(Expr, Expr)>,
        else_expr:  Option<Box<Expr>>,
    },

    /// Scalar function call: `UPPER(col)`, `ROUND(x, 2)`, etc.
    Function {
        name: String,
        args: Vec<Expr>,
    },

    /// `CAST(expr AS type)`.
    Cast {
        expr:     Box<Expr>,
        to_type:  crate::storage::types::DataType,
    },

    /// `EXTRACT(field FROM expr)`.
    Extract {
        field: DateTimeField,
        expr:  Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum DateTimeField {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Epoch,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,

    // Comparison
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,

    // Logical
    And,
    Or,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BinaryOp::Add   => "+",
            BinaryOp::Sub   => "-",
            BinaryOp::Mul   => "*",
            BinaryOp::Div   => "/",
            BinaryOp::Eq    => "=",
            BinaryOp::NotEq => "!=",
            BinaryOp::Lt    => "<",
            BinaryOp::LtEq  => "<=",
            BinaryOp::Gt    => ">",
            BinaryOp::GtEq  => ">=",
            BinaryOp::And   => "AND",
            BinaryOp::Or    => "OR",
        })
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Literal(v)   => write!(f, "{v}"),
            Expr::Column(n)    => write!(f, "{n}"),
            Expr::BinaryOp { left, op, right } => write!(f, "({left} {op} {right})"),
            Expr::Not(e)       => write!(f, "NOT {e}"),
            Expr::IsNull(e)    => write!(f, "{e} IS NULL"),
            Expr::IsNotNull(e) => write!(f, "{e} IS NOT NULL"),
            Expr::Like { expr, pattern, negated, case_insensitive } => {
                let kw = if *case_insensitive { "ILIKE" } else { "LIKE" };
                let neg = if *negated { "NOT " } else { "" };
                write!(f, "{expr} {neg}{kw} '{pattern}'")
            }
            Expr::InList { expr, list, negated } => {
                let neg = if *negated { "NOT " } else { "" };
                let items: Vec<_> = list.iter().map(|e| e.to_string()).collect();
                write!(f, "{expr} {neg}IN ({})", items.join(", "))
            }
            Expr::Case { when_thens, else_expr } => {
                write!(f, "CASE")?;
                for (w, t) in when_thens { write!(f, " WHEN {w} THEN {t}")?; }
                if let Some(e) = else_expr { write!(f, " ELSE {e}")?; }
                write!(f, " END")
            }
            Expr::Function { name, args } => {
                let parts: Vec<_> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{name}({})", parts.join(", "))
            }
            Expr::Cast { expr, to_type } => write!(f, "CAST({expr} AS {to_type})"),
            Expr::Extract { field, expr } => write!(f, "EXTRACT({field:?} FROM {expr})"),
            Expr::Subquery(p)  => write!(f, "(SELECT FROM {})", p.0.table),
            Expr::InSubquery { expr, negated, plan } => {
                let neg = if *negated { "NOT " } else { "" };
                write!(f, "{expr} {neg}IN (SELECT FROM {})", plan.0.table)
            }
            Expr::Exists { negated, plan } => {
                let neg = if *negated { "NOT " } else { "" };
                write!(f, "{neg}EXISTS (SELECT FROM {})", plan.0.table)
            }
        }
    }
}

impl Expr {
    /// Evaluate this expression against a batch, returning one ScalarValue per row.
    /// Evaluate into a dense column, without a `ScalarValue` per row.
    ///
    /// `eval_batch` boxes every value into a 48-byte enum. That has turned up as the
    /// dominant cost in four separate operators — the join, DISTINCT, window ORDER
    /// BY and UPDATE — each of which grew its own local workaround. This is the
    /// shared version.
    ///
    /// Returns `None` for anything it does not cover, so callers keep `eval_batch`
    /// as the fallback and there is exactly one definition of what an expression
    /// means. Handled here: a plain column reference, a literal, and arithmetic over
    /// two f64 operands — the shapes that actually appear in hot paths.
    ///
    /// NULLs bail out rather than being modelled: arithmetic on a NULL yields NULL,
    /// which needs a validity mask the dense form here does not carry, and guessing
    /// at it would change query results rather than just timings.
    pub fn eval_dense_f64(&self, batch: &RecordBatch) -> Option<Vec<f64>> {
        use crate::storage::column::Column;

        match self {
            Expr::Column(name) => {
                let col = batch.column_by_name(name).ok()?;
                if col.null_count() > 0 {
                    return None;
                }
                match col {
                    Column::Float64(c) => Some(c.dense().to_vec()),
                    Column::Int64(c) => Some(c.dense().iter().map(|&v| v as f64).collect()),
                    Column::Int32(c) => Some(c.dense().iter().map(|&v| v as f64).collect()),
                    Column::Float32(c) => Some(c.dense().iter().map(|&v| v as f64).collect()),
                    _ => None,
                }
            }

            Expr::Literal(ScalarValue::Float64(Some(v))) => {
                Some(vec![*v; batch.row_count()])
            }
            Expr::Literal(ScalarValue::Int64(Some(v))) => {
                Some(vec![*v as f64; batch.row_count()])
            }

            Expr::BinaryOp { left, op, right } => {
                let f: fn(f64, f64) -> f64 = match op {
                    BinaryOp::Add => |a, b| a + b,
                    BinaryOp::Sub => |a, b| a - b,
                    BinaryOp::Mul => |a, b| a * b,
                    // Division by zero gives an infinity here, matching what the
                    // boxed path produces for f64 rather than inventing an error.
                    BinaryOp::Div => |a, b| a / b,
                    _ => return None,
                };
                let mut l = left.eval_dense_f64(batch)?;
                let r = right.eval_dense_f64(batch)?;
                if l.len() != r.len() {
                    return None;
                }
                for (a, b) in l.iter_mut().zip(r) {
                    *a = f(*a, b);
                }
                Some(l)
            }

            _ => None,
        }
    }

    pub fn eval_batch(&self, batch: &RecordBatch) -> FalconResult<Vec<ScalarValue>> {
        match self {
            Expr::Literal(val) => {
                Ok(vec![val.clone(); batch.row_count()])
            }

            // The executor folds these into literals before evaluation; reaching
            // here means a code path forgot to call `resolve_subqueries`.
            Expr::Subquery(plan) => Err(FalconError::InvalidQuery(format!(
                "internal: unresolved scalar subquery on '{}'",
                plan.0.table
            ))),
            Expr::InSubquery { plan, .. } => Err(FalconError::InvalidQuery(format!(
                "internal: unresolved IN (SELECT …) on '{}'",
                plan.0.table
            ))),
            Expr::Exists { plan, .. } => Err(FalconError::InvalidQuery(format!(
                "internal: unresolved EXISTS on '{}'",
                plan.0.table
            ))),

            Expr::Column(name) => {
                let col = batch.column_by_name(name)?;
                (0..batch.row_count())
                    .map(|i| col.get(i))
                    .collect()
            }

            Expr::BinaryOp { left, op, right } => {
                let lhs = left.eval_batch(batch)?;
                let rhs = right.eval_batch(batch)?;
                lhs.into_iter()
                    .zip(rhs)
                    .map(|(l, r)| apply_binary_op(&l, op, &r))
                    .collect()
            }

            Expr::Not(inner) => {
                inner.eval_batch(batch)?
                    .into_iter()
                    .map(|v| match v {
                        ScalarValue::Boolean(Some(b)) => Ok(ScalarValue::Boolean(Some(!b))),
                        ScalarValue::Boolean(None)    => Ok(ScalarValue::Boolean(None)),
                        _ => Err(FalconError::InvalidQuery("NOT requires a boolean".into())),
                    })
                    .collect()
            }

            Expr::IsNull(inner) => {
                Ok(inner.eval_batch(batch)?
                    .into_iter()
                    .map(|v| ScalarValue::Boolean(Some(v.is_null())))
                    .collect())
            }

            Expr::IsNotNull(inner) => {
                Ok(inner.eval_batch(batch)?
                    .into_iter()
                    .map(|v| ScalarValue::Boolean(Some(!v.is_null())))
                    .collect())
            }

            Expr::Like { expr, pattern, negated, case_insensitive } => {
                let vals = expr.eval_batch(batch)?;
                Ok(vals.into_iter().map(|v| {
                    let matched = match &v {
                        ScalarValue::Utf8(Some(s)) => {
                            let (s2, p2) = if *case_insensitive {
                                (s.to_lowercase(), pattern.to_lowercase())
                            } else {
                                (s.clone(), pattern.clone())
                            };
                            like_matches(&s2, &p2)
                        }
                        ScalarValue::Null | ScalarValue::Utf8(None) => false,
                        _ => false,
                    };
                    ScalarValue::Boolean(Some(if *negated { !matched } else { matched }))
                }).collect())
            }

            Expr::InList { expr, list, negated } => {
                let vals  = expr.eval_batch(batch)?;
                let n     = batch.row_count();
                // Pre-evaluate list members (they may be literals or columns).
                let list_cols: Vec<Vec<ScalarValue>> = list
                    .iter()
                    .map(|e| e.eval_batch(batch))
                    .collect::<FalconResult<_>>()?;
                Ok((0..n).map(|i| {
                    let v = &vals[i];
                    if v.is_null() {
                        return ScalarValue::Boolean(None);
                    }
                    let found = list_cols.iter().any(|col| col[i] == *v);
                    ScalarValue::Boolean(Some(if *negated { !found } else { found }))
                }).collect())
            }

            Expr::Case { when_thens, else_expr } => {
                let n = batch.row_count();
                // Evaluate all condition and result columns up front.
                let cond_cols: Vec<Vec<ScalarValue>> = when_thens
                    .iter()
                    .map(|(c, _)| c.eval_batch(batch))
                    .collect::<FalconResult<_>>()?;
                let res_cols: Vec<Vec<ScalarValue>> = when_thens
                    .iter()
                    .map(|(_, r)| r.eval_batch(batch))
                    .collect::<FalconResult<_>>()?;
                let else_col = else_expr.as_ref()
                    .map(|e| e.eval_batch(batch))
                    .transpose()?;

                Ok((0..n).map(|i| {
                    for (ci, (cond_col, res_col)) in cond_cols.iter().zip(res_cols.iter()).enumerate() {
                        let _ = ci;
                        if matches!(cond_col[i], ScalarValue::Boolean(Some(true))) {
                            return res_col[i].clone();
                        }
                    }
                    else_col.as_ref()
                        .map(|c| c[i].clone())
                        .unwrap_or(ScalarValue::Null)
                }).collect())
            }

            Expr::Function { name, args } => {
                // Clock functions are constant within a query: SQL requires every row
                // to see the same instant, so read the clock once and broadcast it
                // rather than calling it per row and straddling a tick boundary.
                if let Some(v) = eval_clock_function(name) {
                    return Ok(vec![v; batch.row_count()]);
                }

                let arg_cols: Vec<Vec<ScalarValue>> = args
                    .iter()
                    .map(|a| a.eval_batch(batch))
                    .collect::<FalconResult<_>>()?;
                let n = batch.row_count();
                Ok((0..n).map(|i| {
                    let row_args: Vec<&ScalarValue> = arg_cols.iter().map(|c| &c[i]).collect();
                    eval_function(name, &row_args)
                }).collect::<FalconResult<_>>()?)
            }

            Expr::Cast { expr, to_type } => {
                use crate::storage::types::{DataType, parse_date, parse_timestamp, parse_decimal};
                let vals = expr.eval_batch(batch)?;
                vals.into_iter().map(|v| {
                    if v.is_null() { return Ok(ScalarValue::Null); }
                    Ok(match to_type {
                        DataType::Int64 | DataType::Int32 => {
                            let n = match &v {
                                ScalarValue::Int64(Some(i))   => *i,
                                ScalarValue::Int32(Some(i))   => *i as i64,
                                ScalarValue::Float64(Some(f)) => *f as i64,
                                ScalarValue::Utf8(Some(s))    => s.parse::<i64>().map_err(|_|
                                    FalconError::InvalidQuery(format!("cannot cast '{s}' to INTEGER"))
                                )?,
                                ScalarValue::Boolean(Some(b)) => if *b { 1 } else { 0 },
                                ScalarValue::Date32(Some(d))  => *d as i64,
                                other => return Err(FalconError::InvalidQuery(
                                    format!("cannot cast {other} to INTEGER")
                                )),
                            };
                            ScalarValue::Int64(Some(n))
                        }
                        DataType::Float64 | DataType::Float32 => {
                            let f = match &v {
                                ScalarValue::Float64(Some(f))          => *f,
                                ScalarValue::Int64(Some(i))            => *i as f64,
                                ScalarValue::Utf8(Some(s))             => s.parse::<f64>().map_err(|_|
                                    FalconError::InvalidQuery(format!("cannot cast '{s}' to FLOAT"))
                                )?,
                                ScalarValue::Decimal128(d) if d.value.is_some() =>
                                    d.value.unwrap() as f64 / 10f64.powi(d.scale as i32),
                                other => return Err(FalconError::InvalidQuery(
                                    format!("cannot cast {other} to FLOAT")
                                )),
                            };
                            ScalarValue::Float64(Some(f))
                        }
                        DataType::Utf8 => ScalarValue::Utf8(Some(v.to_string())),
                        DataType::Boolean => {
                            let b = match &v {
                                ScalarValue::Boolean(Some(b)) => *b,
                                ScalarValue::Int64(Some(i))   => *i != 0,
                                ScalarValue::Utf8(Some(s))    => matches!(s.to_lowercase().as_str(), "true" | "1" | "yes"),
                                other => return Err(FalconError::InvalidQuery(
                                    format!("cannot cast {other} to BOOLEAN")
                                )),
                            };
                            ScalarValue::Boolean(Some(b))
                        }
                        DataType::Date32 => {
                            let d = match &v {
                                ScalarValue::Date32(Some(d)) => *d,
                                ScalarValue::Int64(Some(i))  => *i as i32,
                                ScalarValue::Utf8(Some(s))   => parse_date(s).ok_or_else(|| {
                                    FalconError::InvalidQuery(format!("cannot cast '{s}' to DATE"))
                                })?,
                                other => return Err(FalconError::InvalidQuery(
                                    format!("cannot cast {other} to DATE")
                                )),
                            };
                            ScalarValue::Date32(Some(d))
                        }
                        DataType::Timestamp => {
                            let ts = match &v {
                                ScalarValue::Timestamp(Some(t)) => *t,
                                ScalarValue::Int64(Some(i))     => *i,
                                ScalarValue::Utf8(Some(s))      => parse_timestamp(s).ok_or_else(|| {
                                    FalconError::InvalidQuery(format!("cannot cast '{s}' to TIMESTAMP"))
                                })?,
                                other => return Err(FalconError::InvalidQuery(
                                    format!("cannot cast {other} to TIMESTAMP")
                                )),
                            };
                            ScalarValue::Timestamp(Some(ts))
                        }
                        DataType::Decimal128(p, s) => {
                            let n = match &v {
                                ScalarValue::Decimal128(d) if d.value.is_some() => d.value.unwrap(),
                                ScalarValue::Int64(Some(i))   => (*i as i128) * 10i128.pow(*s as u32),
                                ScalarValue::Float64(Some(f)) => (*f * 10f64.powi(*s as i32)) as i128,
                                ScalarValue::Utf8(Some(st))   => parse_decimal(st, *s).ok_or_else(|| {
                                    FalconError::InvalidQuery(format!("cannot cast '{st}' to DECIMAL"))
                                })?,
                                other => return Err(FalconError::InvalidQuery(
                                    format!("cannot cast {other} to DECIMAL")
                                )),
                            };
                            ScalarValue::decimal(Some(n), *p, *s)
                        }
                        _ => v,
                    })
                }).collect()
            }

            Expr::Extract { field, expr } => {
                use crate::storage::types::days_to_ymd;
                let vals = expr.eval_batch(batch)?;
                vals.into_iter().map(|v| {
                    if v.is_null() { return Ok(ScalarValue::Null); }
                    let result = match (&v, field) {
                        (ScalarValue::Date32(Some(d)), f) => {
                            let (y, mo, day) = days_to_ymd(*d);
                            Some(match f {
                                DateTimeField::Year   => y as i64,
                                DateTimeField::Month  => mo as i64,
                                DateTimeField::Day    => day as i64,
                                DateTimeField::Epoch  => (*d as i64) * 86_400,
                                _ => return Err(FalconError::InvalidQuery(
                                    "DATE only supports YEAR/MONTH/DAY/EPOCH".into()
                                )),
                            })
                        }
                        (ScalarValue::Timestamp(Some(ts)), f) => {
                            let total_secs = ts.div_euclid(1_000_000);
                            let secs_day   = total_secs.rem_euclid(86_400);
                            let days       = total_secs.div_euclid(86_400) as i32;
                            let (y, mo, day) = days_to_ymd(days);
                            let h = secs_day / 3600;
                            let m = (secs_day % 3600) / 60;
                            let s = secs_day % 60;
                            Some(match f {
                                DateTimeField::Year   => y as i64,
                                DateTimeField::Month  => mo as i64,
                                DateTimeField::Day    => day as i64,
                                DateTimeField::Hour   => h,
                                DateTimeField::Minute => m,
                                DateTimeField::Second => s,
                                DateTimeField::Epoch  => total_secs,
                            })
                        }
                        _ => return Err(FalconError::InvalidQuery(
                            "EXTRACT requires a DATE or TIMESTAMP value".into()
                        )),
                    };
                    Ok(result.map(|n| ScalarValue::Int64(Some(n))).unwrap_or(ScalarValue::Null))
                }).collect()
            }
        }
    }

    /// Add every column this expression reads to `out`.
    ///
    /// Returns `false` if the expression contains a node whose column references
    /// cannot be enumerated statically — today that is any subquery, whose plan is
    /// opaque here. Callers using this for projection pushdown must treat `false`
    /// as "needs every column" rather than trusting a partial set, or pushdown
    /// would prune a column the query actually reads.
    pub fn collect_columns(&self, out: &mut std::collections::HashSet<String>) -> bool {
        match self {
            Expr::Column(name) => {
                out.insert(name.clone());
                true
            }
            Expr::Literal(_) => true,

            // Subquery plans are not walked here, so their references are unknown.
            Expr::Subquery(_) | Expr::InSubquery { .. } | Expr::Exists { .. } => false,

            Expr::BinaryOp { left, op: _, right } => {
                let l = left.collect_columns(out);
                let r = right.collect_columns(out);
                l && r
            }
            Expr::Not(e) | Expr::IsNull(e) | Expr::IsNotNull(e) => e.collect_columns(out),
            Expr::Like { expr, .. } => expr.collect_columns(out),
            Expr::Cast { expr, .. } => expr.collect_columns(out),
            Expr::Extract { expr, .. } => expr.collect_columns(out),
            Expr::InList { expr, list, .. } => {
                let mut ok = expr.collect_columns(out);
                for e in list {
                    ok &= e.collect_columns(out);
                }
                ok
            }
            Expr::Case { when_thens, else_expr } => {
                let mut ok = true;
                for (w, t) in when_thens {
                    ok &= w.collect_columns(out);
                    ok &= t.collect_columns(out);
                }
                if let Some(e) = else_expr {
                    ok &= e.collect_columns(out);
                }
                ok
            }
            Expr::Function { args, .. } => {
                let mut ok = true;
                for a in args {
                    ok &= a.collect_columns(out);
                }
                ok
            }
        }
    }

    /// Evaluate to a boolean mask — used for WHERE filters.
    pub fn eval_mask(&self, batch: &RecordBatch) -> FalconResult<Vec<bool>> {
        let values = self.eval_batch(batch)?;
        values
            .into_iter()
            .map(|v| match v {
                ScalarValue::Boolean(Some(b)) => Ok(b),
                ScalarValue::Boolean(None)    => Ok(false),
                ScalarValue::Null             => Ok(false),
                _ => Err(FalconError::InvalidQuery(
                    "filter expression must return a boolean".into(),
                )),
            })
            .collect()
    }

    /// Packed-bit bitmask filter — tries a zero-alloc fast path first.
    ///
    /// Fast path (no ScalarValue boxing):
    ///   `col op literal`, `AND`/`OR` of fast sub-expressions.
    ///   Operates directly on `&[Option<T>]`, 8× less memory than `Vec<bool>`.
    ///
    /// Falls back to `eval_batch` → pack for any complex expression.
    pub fn eval_mask_bits(&self, batch: &RecordBatch) -> FalconResult<Vec<u64>> {
        let n = batch.row_count();
        let n_words = (n + 63) / 64;

        if let Some(r) = self.try_fast_mask(batch, n_words) {
            return r;
        }

        // General fallback: eval_batch → pack booleans into u64 words.
        let values = self.eval_batch(batch)?;
        let mut mask = vec![0u64; n_words];
        for (i, v) in values.iter().enumerate() {
            if matches!(v, ScalarValue::Boolean(Some(true))) {
                mask[i >> 6] |= 1u64 << (i & 63);
            }
        }
        Ok(mask)
    }

    /// Returns `Some(result)` if the expression is a simple predicate we can evaluate
    /// directly on typed column slices (no `ScalarValue` allocation).
    fn try_fast_mask(&self, batch: &RecordBatch, n_words: usize) -> Option<FalconResult<Vec<u64>>> {
        

        match self {
            // AND: bitwise AND of two fast masks; falls back if either side is complex.
            Expr::BinaryOp { left, op: BinaryOp::And, right } => {
                let lm = left.try_fast_mask(batch, n_words)?;
                let rm = right.try_fast_mask(batch, n_words)?;
                Some(lm.and_then(|mut l| { let r = rm?; zip_bits(&mut l, &r, |a, b| a & b); Ok(l) }))
            }
            // OR: bitwise OR.
            Expr::BinaryOp { left, op: BinaryOp::Or, right } => {
                let lm = left.try_fast_mask(batch, n_words)?;
                let rm = right.try_fast_mask(batch, n_words)?;
                Some(lm.and_then(|mut l| { let r = rm?; zip_bits(&mut l, &r, |a, b| a | b); Ok(l) }))
            }
            // col op literal — direct typed comparison.
            Expr::BinaryOp { left, op, right } => {
                match (left.as_ref(), right.as_ref()) {
                    (Expr::Column(name), Expr::Literal(lit)) => {
                        let col = batch.column_by_name(name).ok()?;
                        Some(fast_cmp_mask(col, op, lit, n_words))
                    }
                    // literal op col — swap the operator.
                    (Expr::Literal(lit), Expr::Column(name)) => {
                        let col = batch.column_by_name(name).ok()?;
                        let swapped = swap_cmp_op(op)?;
                        Some(fast_cmp_mask(col, &swapped, lit, n_words))
                    }
                    _ => None,
                }
            }
            // `col IN (literal, ...)` — the union of the equality masks.
            //
            // This shape had no typed path, so it fell to `eval_batch`, which
            // broadcasts *each list member* into a `ScalarValue` per row before
            // comparing: `IN (1,2,3)` over 100k rows built 400k boxed values to
            // answer 300k comparisons. Measured at 8.09ms against 98.6us for the
            // single equality `bucket = 1` on the same column -- 82x the cost of
            // one comparison to do three.
            //
            // Routing each member through `fast_cmp_mask` is not only faster, it
            // is the only way the two agree. The general path compares boxed
            // values with a derived `PartialEq`, which is strict about the
            // variant, so `WHERE f IN (1, 2)` on a `Float64` column matched *no
            // rows at all* while `WHERE f = 1` matched one: the literals parse as
            // integers and `Int64(1) != Float64(1.0)`. `fast_cmp_mask` has the
            // widening arms, so `IN` now coerces exactly as `=` does.
            Expr::InList { expr, list, negated } => {
                let Expr::Column(name) = expr.as_ref() else { return None };
                let col = batch.column_by_name(name).ok()?;
                // Every member must be a literal; a correlated one is a different
                // question and stays on the general path.
                let lits: Vec<&ScalarValue> = list
                    .iter()
                    .map(|e| match e {
                        Expr::Literal(v) => Some(v),
                        _ => None,
                    })
                    .collect::<Option<_>>()?;

                let mut acc = vec![0u64; n_words];
                for lit in lits {
                    match fast_cmp_mask(col, &BinaryOp::Eq, lit, n_words) {
                        Ok(m) => zip_bits(&mut acc, &m, |a, b| a | b),
                        Err(e) => return Some(Err(e)),
                    }
                }
                if *negated {
                    // NOT IN is the complement, minus the rows that are NULL:
                    // a NULL value is neither in nor out, and the general path
                    // gave those rows `Boolean(None)`, which filters as false.
                    for w in acc.iter_mut() {
                        *w = !*w;
                    }
                    mask_off_padding(&mut acc, batch.row_count());
                    clear_null_rows(col, &mut acc);
                }
                Some(Ok(acc))
            }

            _ => None,
        }
    }
}

// ─── Fast predicate helpers ───────────────────────────────────────────────────

/// Scalar fallback: compare a dense slice against a value, writing 1 bit per row.
/// Operator is matched once outside the loop so LLVM can autovectorise each arm.
fn cmp_dense<T: Copy + PartialOrd>(
    dense: &[T],
    v:     T,
    op:    &BinaryOp,
    mask:  &mut [u64],
) -> FalconResult<()> {
    macro_rules! run {
        ($pred:expr) => {{
            for (w, chunk) in dense.chunks(64).enumerate() {
                let mut bits = 0u64;
                for (j, &x) in chunk.iter().enumerate() {
                    bits |= (($pred)(x) as u64) << j;
                }
                mask[w] = bits;
            }
        }};
    }
    match op {
        BinaryOp::Gt    => run!(|x: T| x >  v),
        BinaryOp::GtEq  => run!(|x: T| x >= v),
        BinaryOp::Lt    => run!(|x: T| x <  v),
        BinaryOp::LtEq  => run!(|x: T| x <= v),
        BinaryOp::Eq    => run!(|x: T| x == v),
        BinaryOp::NotEq => run!(|x: T| x != v),
        _ => return Err(FalconError::InvalidQuery("non-comparison op in WHERE".into())),
    }
    Ok(())
}

/// Vectorised f64 comparison — writes directly to a u64 bitmask using AVX2 movemask.
///
/// Uses `_mm256_cmp_pd` (4-wide) + `_mm256_movemask_pd` to extract 4 comparison bits
/// per instruction, accumulating directly into each u64 word. Eliminates the
/// intermediate `Vec<bool>` allocation and the separate pack pass.
fn cmp_dense_f64_simd(dense: &[f64], v: f64, op: &BinaryOp, mask: &mut [u64]) -> FalconResult<()> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: we just checked for AVX2 above.
        return unsafe { cmp_f64_avx2(dense, v, op, mask) };
    }
    // Scalar fallback (non-x86 or no AVX2).
    cmp_f64_scalar(dense, v, op, mask)
}

/// Same approach for i64 (no AVX2 movemask for integers in the same way, so keep
/// the two-pass approach but avoid the intermediate Vec allocation by writing inline).
fn cmp_dense_i64_simd(dense: &[i64], v: i64, op: &BinaryOp, mask: &mut [u64]) -> FalconResult<()> {
    cmp_i64_scalar(dense, v, op, mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cmp_f64_avx2(dense: &[f64], v: f64, op: &BinaryOp, mask: &mut [u64]) -> FalconResult<()> {
    use std::arch::x86_64::*;
    macro_rules! run_avx2 {
        ($cmp_imm:expr, $scalar_pred:expr) => {{
            let thresh = _mm256_set1_pd(v);
            for (wi, chunk64) in dense.chunks(64).enumerate() {
                let n = chunk64.len();
                let n_full4 = n & !3;
                let ptr = chunk64.as_ptr();
                let mut word = 0u64;
                let mut i = 0;
                while i < n_full4 {
                    let vals = _mm256_loadu_pd(ptr.add(i));
                    let cmp  = _mm256_cmp_pd::<$cmp_imm>(vals, thresh);
                    word |= (_mm256_movemask_pd(cmp) as u64) << i;
                    i += 4;
                }
                // scalar tail (< 4 remaining elements)
                for j in n_full4..n {
                    if ($scalar_pred)(*ptr.add(j)) { word |= 1u64 << j; }
                }
                mask[wi] = word;
            }
            Ok(())
        }};
    }
    match op {
        BinaryOp::Gt    => run_avx2!({ _CMP_GT_OQ  as i32 }, |x: f64| x >  v),
        BinaryOp::GtEq  => run_avx2!({ _CMP_GE_OQ  as i32 }, |x: f64| x >= v),
        BinaryOp::Lt    => run_avx2!({ _CMP_LT_OQ  as i32 }, |x: f64| x <  v),
        BinaryOp::LtEq  => run_avx2!({ _CMP_LE_OQ  as i32 }, |x: f64| x <= v),
        BinaryOp::Eq    => run_avx2!({ _CMP_EQ_OQ  as i32 }, |x: f64| x == v),
        BinaryOp::NotEq => run_avx2!({ _CMP_NEQ_UQ as i32 }, |x: f64| x != v),
        _ => Err(FalconError::InvalidQuery("non-comparison op in WHERE".into())),
    }
}

/// Scalar fallback: single-pass direct bitmask write, no Vec<bool> allocation.
#[inline]
fn cmp_f64_scalar(dense: &[f64], v: f64, op: &BinaryOp, mask: &mut [u64]) -> FalconResult<()> {
    macro_rules! run {
        ($pred:expr) => {{
            for (wi, chunk) in dense.chunks(64).enumerate() {
                let mut bits = 0u64;
                for (j, &x) in chunk.iter().enumerate() {
                    bits |= (($pred)(x) as u64) << j;
                }
                mask[wi] = bits;
            }
            Ok(())
        }};
    }
    match op {
        BinaryOp::Gt    => run!(|x: f64| x >  v),
        BinaryOp::GtEq  => run!(|x: f64| x >= v),
        BinaryOp::Lt    => run!(|x: f64| x <  v),
        BinaryOp::LtEq  => run!(|x: f64| x <= v),
        BinaryOp::Eq    => run!(|x: f64| x == v),
        BinaryOp::NotEq => run!(|x: f64| x != v),
        _ => Err(FalconError::InvalidQuery("non-comparison op in WHERE".into())),
    }
}

/// i64 single-pass direct bitmask write, no Vec<bool> allocation.
#[inline]
fn cmp_i64_scalar(dense: &[i64], v: i64, op: &BinaryOp, mask: &mut [u64]) -> FalconResult<()> {
    macro_rules! run {
        ($pred:expr) => {{
            for (wi, chunk) in dense.chunks(64).enumerate() {
                let mut bits = 0u64;
                for (j, &x) in chunk.iter().enumerate() {
                    bits |= (($pred)(x) as u64) << j;
                }
                mask[wi] = bits;
            }
            Ok(())
        }};
    }
    match op {
        BinaryOp::Gt    => run!(|x: i64| x >  v),
        BinaryOp::GtEq  => run!(|x: i64| x >= v),
        BinaryOp::Lt    => run!(|x: i64| x <  v),
        BinaryOp::LtEq  => run!(|x: i64| x <= v),
        BinaryOp::Eq    => run!(|x: i64| x == v),
        BinaryOp::NotEq => run!(|x: i64| x != v),
        _ => Err(FalconError::InvalidQuery("non-comparison op in WHERE".into())),
    }
}

/// Clear mask bits for NULL rows.
///
/// `cmp_dense` deliberately compares every slot, including the placeholder values
/// sitting at NULL positions. Sweeping those bits afterwards keeps the comparison
/// loop branch-free; a comparison against NULL is never true in SQL.
fn clear_null_bits<T: Copy + Default>(
    col:  &crate::storage::column::PrimitiveColumn<T>,
    mask: &mut [u64],
) {
    if !col.has_nulls() {
        return;
    }
    for i in 0..col.len() {
        if !col.is_valid(i) {
            mask[i >> 6] &= !(1u64 << (i & 63));
        }
    }
}

/// Clear the bits past the last row.
///
/// Complementing a mask sets the padding bits in the final word, which stand for
/// rows that do not exist. Nothing downstream should ever see them set.
fn mask_off_padding(mask: &mut [u64], rows: usize) {
    let used = rows & 63;
    if used != 0 {
        if let Some(last) = mask.last_mut() {
            *last &= (1u64 << used) - 1;
        }
    }
}

/// Clear the bit of every row whose value is NULL, whatever the column's type.
fn clear_null_rows(col: &crate::storage::column::Column, mask: &mut [u64]) {
    use crate::storage::column::Column;
    macro_rules! clear {
        ($c:expr) => {{
            if $c.has_nulls() {
                for i in 0..$c.len() {
                    if !$c.is_valid(i) {
                        mask[i >> 6] &= !(1u64 << (i & 63));
                    }
                }
            }
        }};
    }
    match col {
        Column::Int32(c)     => clear!(c),
        Column::Int64(c)     => clear!(c),
        Column::Float32(c)   => clear!(c),
        Column::Float64(c)   => clear!(c),
        Column::Boolean(c)   => {
            for (i, v) in c.values.iter().enumerate() {
                if v.is_none() {
                    mask[i >> 6] &= !(1u64 << (i & 63));
                }
            }
        }
        Column::Date32(c)    => clear!(c),
        Column::Timestamp(c) => clear!(c),
        Column::Utf8(c)      => {
            if c.has_nulls() {
                for (i, &code) in c.codes().iter().enumerate() {
                    if crate::storage::column::Utf8Column::is_null_code(code) {
                        mask[i >> 6] &= !(1u64 << (i & 63));
                    }
                }
            }
        }
        _ => {}
    }
}

fn zip_bits(dst: &mut [u64], src: &[u64], f: impl Fn(u64, u64) -> u64) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) { *d = f(*d, s); }
}

/// Parallel version of `cmp_dense_f64_simd` — splits mask into N coarse chunks
/// (one per rayon thread, minimum 1024 words = 65,536 rows per chunk) and runs
/// each chunk with the existing AVX2 function. One rayon task per chunk.
fn par_cmp_dense_f64(dense: &[f64], v: f64, op: &BinaryOp, mask: &mut [u64]) -> FalconResult<()> {
    let n_words    = mask.len();
    let chunk_words = (n_words / 8).max(1024); // ≤ 8 tasks, each ≥ 65k rows
    mask.par_chunks_mut(chunk_words)
        .enumerate()
        .try_for_each(|(ci, word_chunk)| {
            let start_row = ci * chunk_words * 64;
            let end_row   = (start_row + word_chunk.len() * 64).min(dense.len());
            if start_row >= dense.len() { return Ok(()); }
            cmp_dense_f64_simd(&dense[start_row..end_row], v, op, word_chunk)
        })
}

/// Parallel version of `cmp_dense_i64_simd`.
fn par_cmp_dense_i64(dense: &[i64], v: i64, op: &BinaryOp, mask: &mut [u64]) -> FalconResult<()> {
    let n_words     = mask.len();
    let chunk_words = (n_words / 8).max(1024);
    mask.par_chunks_mut(chunk_words)
        .enumerate()
        .try_for_each(|(ci, word_chunk)| {
            let start_row = ci * chunk_words * 64;
            let end_row   = (start_row + word_chunk.len() * 64).min(dense.len());
            if start_row >= dense.len() { return Ok(()); }
            cmp_dense_i64_simd(&dense[start_row..end_row], v, op, word_chunk)
        })
}

fn swap_cmp_op(op: &BinaryOp) -> Option<BinaryOp> {
    Some(match op {
        BinaryOp::Gt    => BinaryOp::Lt,
        BinaryOp::GtEq  => BinaryOp::LtEq,
        BinaryOp::Lt    => BinaryOp::Gt,
        BinaryOp::LtEq  => BinaryOp::GtEq,
        BinaryOp::Eq    => BinaryOp::Eq,
        BinaryOp::NotEq => BinaryOp::NotEq,
        _ => return None,
    })
}

/// Zero-alloc typed comparison: set bit i if `col[i] op lit` is true.
fn fast_cmp_mask(
    col:     &crate::storage::column::Column,
    op:      &BinaryOp,
    lit:     &ScalarValue,
    n_words: usize,
) -> FalconResult<Vec<u64>> {
    use crate::storage::column::Column;
    let mut mask = vec![0u64; n_words];

    macro_rules! cmp_loop {
        ($col:expr, $cmp_val:expr, $ty:ty) => {{
            let v: $ty = $cmp_val;
            cmp_dense($col.dense(), v, op, &mut mask)?;
            clear_null_bits($col, &mut mask);
        }};
    }

    macro_rules! cmp_f64 {
        ($c:expr, $v:expr) => {{
            if n_words >= PAR_MASK_WORDS {
                par_cmp_dense_f64($c.dense(), $v, op, &mut mask)?;
            } else {
                cmp_dense_f64_simd($c.dense(), $v, op, &mut mask)?;
            }
            clear_null_bits($c, &mut mask);
        }};
    }
    macro_rules! cmp_i64 {
        ($c:expr, $v:expr) => {{
            if n_words >= PAR_MASK_WORDS {
                par_cmp_dense_i64($c.dense(), $v, op, &mut mask)?;
            } else {
                cmp_dense_i64_simd($c.dense(), $v, op, &mut mask)?;
            }
            clear_null_bits($c, &mut mask);
        }};
    }

    match (col, lit) {
        // Float64 — SIMD path, optionally parallel for large batches
        (Column::Float64(c), ScalarValue::Float64(Some(v))) => { cmp_f64!(c, *v); }
        (Column::Float64(c), ScalarValue::Int64(Some(v)))   => { cmp_f64!(c, *v as f64); }
        (Column::Float64(c), ScalarValue::Int32(Some(v)))   => { cmp_f64!(c, *v as f64); }
        // Int64 — SIMD path, optionally parallel
        (Column::Int64(c), ScalarValue::Int64(Some(v)))  => { cmp_i64!(c, *v); }
        (Column::Int64(c), ScalarValue::Int32(Some(v)))  => { cmp_i64!(c, *v as i64); }
        (Column::Int64(c), ScalarValue::Float64(Some(v))) => {
            // Widen i64 → f64, then use SIMD f64 comparison.
            let dense = c.dense();
            let widened: Vec<f64> = dense.iter().map(|&x| x as f64).collect();
            if n_words >= PAR_MASK_WORDS {
                par_cmp_dense_f64(&widened, *v, op, &mut mask)?;
            } else {
                cmp_dense_f64_simd(&widened, *v, op, &mut mask)?;
            }
            clear_null_bits(c, &mut mask);
        }
        (Column::Int32(c),     ScalarValue::Int32(Some(v)))     => cmp_loop!(c, *v, i32),
        (Column::Int32(c),     ScalarValue::Int64(Some(v)))     => cmp_loop!(c, *v as i32, i32),
        (Column::Date32(c),    ScalarValue::Date32(Some(v)))    => cmp_loop!(c, *v, i32),
        (Column::Date32(c),    ScalarValue::Int64(Some(v)))     => cmp_loop!(c, *v as i32, i32),
        (Column::Timestamp(c), ScalarValue::Timestamp(Some(v))) => cmp_loop!(c, *v, i64),
        (Column::Timestamp(c), ScalarValue::Int64(Some(v)))     => cmp_loop!(c, *v, i64),
        (Column::Utf8(c),  ScalarValue::Utf8(Some(v)))  => {
            // Decide the predicate once per distinct string, then read codes.
            //
            // The column stores a code per row and the strings once, so asking
            // the question of every row asks it of the same handful of strings
            // over and over. The dictionary has five entries where the benchmark
            // column has a hundred thousand rows; after this table is built the
            // scan compares nothing but bytes of a `Vec<bool>`.
            let pass: Vec<bool> = c
                .dict()
                .iter()
                .map(|x| match op {
                    BinaryOp::Eq    => x.as_str() == v.as_str(),
                    BinaryOp::NotEq => x.as_str() != v.as_str(),
                    BinaryOp::Lt    => x.as_str() <  v.as_str(),
                    BinaryOp::LtEq  => x.as_str() <= v.as_str(),
                    BinaryOp::Gt    => x.as_str() >  v.as_str(),
                    BinaryOp::GtEq  => x.as_str() >= v.as_str(),
                    _ => false,
                })
                .collect();
            // A NULL cell fails every comparison, as it did row by row above.
            for (w, chunk) in c.codes().chunks(64).enumerate() {
                let mut bits = 0u64;
                for (j, &code) in chunk.iter().enumerate() {
                    let hit = !crate::storage::column::Utf8Column::is_null_code(code)
                        && pass[code as usize];
                    bits |= (hit as u64) << j;
                }
                mask[w] = bits;
            }
        }
        _ => {} // null literal or unsupported type → all bits 0 (no rows pass)
    }

    Ok(mask)
}

// ─── SQL LIKE pattern matching ────────────────────────────────────────────────

/// Match `s` against SQL LIKE `pattern` (`%` = any run, `_` = one char).
fn like_matches(s: &str, pattern: &str) -> bool {
    let s = s.as_bytes();
    let p = pattern.as_bytes();
    like_recursive(s, p)
}

fn like_recursive(s: &[u8], p: &[u8]) -> bool {
    match (s, p) {
        (_, []) => s.is_empty(),
        (_, [b'%', rest @ ..]) => {
            // % can match zero or more characters.
            if like_recursive(s, rest) { return true; }
            if !s.is_empty() { return like_recursive(&s[1..], p); }
            false
        }
        ([], _) => false,
        ([_sc, sr @ ..], [b'_', pr @ ..]) => like_recursive(sr, pr),
        ([sc, sr @ ..], [pc, pr @ ..]) => sc == pc && like_recursive(sr, pr),
    }
}

// ─── Scalar function dispatch ─────────────────────────────────────────────────

/// Wall-clock functions, evaluated once per batch rather than once per row.
///
/// Returns `None` for anything that is not a clock function, so the caller falls
/// through to ordinary per-row dispatch.
fn eval_clock_function(name: &str) -> Option<ScalarValue> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let upper = name.to_uppercase();
    if !matches!(upper.as_str(), "NOW" | "CURRENT_TIMESTAMP" | "CURRENT_DATE" | "TODAY") {
        return None;
    }

    // A clock before the epoch would mean the system time is badly wrong; treat it
    // as zero rather than panicking inside a query.
    let micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);

    Some(match upper.as_str() {
        "CURRENT_DATE" | "TODAY" => {
            ScalarValue::Date32(Some(micros.div_euclid(1_000_000 * 86_400) as i32))
        }
        _ => ScalarValue::Timestamp(Some(micros)),
    })
}

fn eval_function(name: &str, args: &[&ScalarValue]) -> FalconResult<ScalarValue> {
    match name.to_uppercase().as_str() {
        "UPPER" => {
            let s = str_arg(args, 0)?;
            Ok(s.map(|v| ScalarValue::Utf8(Some(v.to_uppercase())))
               .unwrap_or(ScalarValue::Null))
        }
        "LOWER" => {
            let s = str_arg(args, 0)?;
            Ok(s.map(|v| ScalarValue::Utf8(Some(v.to_lowercase())))
               .unwrap_or(ScalarValue::Null))
        }
        "LENGTH" | "LEN" => {
            let s = str_arg(args, 0)?;
            Ok(s.map(|v| ScalarValue::Int64(Some(v.chars().count() as i64)))
               .unwrap_or(ScalarValue::Null))
        }
        "TRIM" => {
            let s = str_arg(args, 0)?;
            Ok(s.map(|v| ScalarValue::Utf8(Some(v.trim().to_string())))
               .unwrap_or(ScalarValue::Null))
        }
        "LTRIM" => {
            let s = str_arg(args, 0)?;
            Ok(s.map(|v| ScalarValue::Utf8(Some(v.trim_start().to_string())))
               .unwrap_or(ScalarValue::Null))
        }
        "RTRIM" => {
            let s = str_arg(args, 0)?;
            Ok(s.map(|v| ScalarValue::Utf8(Some(v.trim_end().to_string())))
               .unwrap_or(ScalarValue::Null))
        }
        "SUBSTR" | "SUBSTRING" => {
            let s   = str_arg(args, 0)?;
            let pos = num_arg(args, 1)? as i64;
            let len = if args.len() > 2 { Some(num_arg(args, 2)? as usize) } else { None };
            Ok(match s {
                None => ScalarValue::Null,
                Some(v) => {
                    let chars: Vec<char> = v.chars().collect();
                    let start = ((pos - 1).max(0)) as usize;
                    let slice: String = match len {
                        Some(l) => chars.iter().skip(start).take(l).collect(),
                        None    => chars.iter().skip(start).collect(),
                    };
                    ScalarValue::Utf8(Some(slice))
                }
            })
        }
        "CONCAT" => {
            let mut out = String::new();
            for arg in args {
                match arg {
                    ScalarValue::Utf8(Some(s)) => out.push_str(s),
                    ScalarValue::Null | ScalarValue::Utf8(None) => {}
                    other => out.push_str(&other.to_string()),
                }
            }
            Ok(ScalarValue::Utf8(Some(out)))
        }
        "COALESCE" => {
            Ok(args.iter()
                .find(|v| !v.is_null())
                .map(|v| (*v).clone())
                .unwrap_or(ScalarValue::Null))
        }
        "NULLIF" => {
            if args.len() < 2 {
                return Err(FalconError::InvalidQuery("NULLIF requires 2 args".into()));
            }
            if args[0] == args[1] { Ok(ScalarValue::Null) } else { Ok((*args[0]).clone()) }
        }
        "ABS" => {
            match args.first() {
                Some(ScalarValue::Float64(Some(v))) => Ok(ScalarValue::Float64(Some(v.abs()))),
                Some(ScalarValue::Int64(Some(v)))   => Ok(ScalarValue::Int64(Some(v.abs()))),
                Some(ScalarValue::Null) | None       => Ok(ScalarValue::Null),
                Some(other) => Err(FalconError::InvalidQuery(
                    format!("ABS: unsupported type {:?}", other.data_type())
                )),
            }
        }
        "ROUND" => {
            let v = float_arg(args, 0)?;
            let decimals = if args.len() > 1 { num_arg(args, 1)? as u32 } else { 0 };
            Ok(match v {
                None => ScalarValue::Null,
                Some(x) => {
                    let factor = 10f64.powi(decimals as i32);
                    ScalarValue::Float64(Some((x * factor).round() / factor))
                }
            })
        }
        "CEIL" | "CEILING" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.ceil()))).unwrap_or(ScalarValue::Null))
        }
        "FLOOR" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.floor()))).unwrap_or(ScalarValue::Null))
        }
        "SQRT" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.sqrt()))).unwrap_or(ScalarValue::Null))
        }
        "POW" | "POWER" => {
            let base = float_arg(args, 0)?;
            let exp  = float_arg(args, 1)?;
            Ok(match (base, exp) {
                (Some(b), Some(e)) => ScalarValue::Float64(Some(b.powf(e))),
                _ => ScalarValue::Null,
            })
        }
        "SIGN" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x: f64| ScalarValue::Int64(Some(x.signum() as i64))).unwrap_or(ScalarValue::Null))
        }
        "IF" => {
            if args.len() < 3 {
                return Err(FalconError::InvalidQuery("IF requires 3 args".into()));
            }
            match args[0] {
                ScalarValue::Boolean(Some(true)) => Ok((*args[1]).clone()),
                _ => Ok((*args[2]).clone()),
            }
        }
        "YEAR" => {
            use crate::storage::types::days_to_ymd;
            match args.first() {
                Some(ScalarValue::Date32(Some(d))) => Ok(ScalarValue::Int64(Some(days_to_ymd(*d).0 as i64))),
                Some(ScalarValue::Timestamp(Some(ts))) => {
                    let days = ts.div_euclid(1_000_000 * 86_400) as i32;
                    Ok(ScalarValue::Int64(Some(days_to_ymd(days).0 as i64)))
                }
                _ => Ok(ScalarValue::Null),
            }
        }
        "MONTH" => {
            use crate::storage::types::days_to_ymd;
            match args.first() {
                Some(ScalarValue::Date32(Some(d))) => Ok(ScalarValue::Int64(Some(days_to_ymd(*d).1 as i64))),
                Some(ScalarValue::Timestamp(Some(ts))) => {
                    let days = ts.div_euclid(1_000_000 * 86_400) as i32;
                    Ok(ScalarValue::Int64(Some(days_to_ymd(days).1 as i64)))
                }
                _ => Ok(ScalarValue::Null),
            }
        }
        "DAY" | "DAYOFMONTH" => {
            use crate::storage::types::days_to_ymd;
            match args.first() {
                Some(ScalarValue::Date32(Some(d))) => Ok(ScalarValue::Int64(Some(days_to_ymd(*d).2 as i64))),
                Some(ScalarValue::Timestamp(Some(ts))) => {
                    let days = ts.div_euclid(1_000_000 * 86_400) as i32;
                    Ok(ScalarValue::Int64(Some(days_to_ymd(days).2 as i64)))
                }
                _ => Ok(ScalarValue::Null),
            }
        }
        "DATEDIFF" => {
            if args.len() < 2 {
                return Err(FalconError::InvalidQuery("DATEDIFF requires 2 args".into()));
            }
            match (args[0], args[1]) {
                (ScalarValue::Date32(Some(a)), ScalarValue::Date32(Some(b))) =>
                    Ok(ScalarValue::Int64(Some((*a - *b) as i64))),
                (ScalarValue::Timestamp(Some(a)), ScalarValue::Timestamp(Some(b))) =>
                    Ok(ScalarValue::Int64(Some((a - b) / (86_400 * 1_000_000)))),
                _ => Ok(ScalarValue::Null),
            }
        }
        "DATE_TRUNC" => {
            use crate::storage::types::{days_to_ymd, ymd_to_days};
            if args.len() < 2 {
                return Err(FalconError::InvalidQuery("DATE_TRUNC requires 2 args".into()));
            }
            let unit = match args[0] {
                ScalarValue::Utf8(Some(s)) => s.to_uppercase(),
                _ => return Err(FalconError::InvalidQuery("DATE_TRUNC: first arg must be a string".into())),
            };
            match args[1] {
                ScalarValue::Timestamp(Some(ts)) => {
                    let total_secs = ts.div_euclid(1_000_000);
                    let days = total_secs.div_euclid(86_400) as i32;
                    let (y, mo, _day) = days_to_ymd(days);
                    let truncated = match unit.as_str() {
                        "YEAR"  => ymd_to_days(y, 1, 1) as i64 * 86_400 * 1_000_000,
                        "MONTH" => ymd_to_days(y, mo, 1) as i64 * 86_400 * 1_000_000,
                        "DAY"   => days as i64 * 86_400 * 1_000_000,
                        "HOUR"  => (total_secs / 3600) * 3600 * 1_000_000,
                        _ => return Err(FalconError::InvalidQuery(format!("DATE_TRUNC: unknown unit '{unit}'"))),
                    };
                    Ok(ScalarValue::Timestamp(Some(truncated)))
                }
                ScalarValue::Date32(Some(d)) => {
                    let (y, mo, _) = days_to_ymd(*d);
                    let truncated = match unit.as_str() {
                        "YEAR"  => ymd_to_days(y, 1, 1),
                        "MONTH" => ymd_to_days(y, mo, 1),
                        "DAY"   => *d,
                        _ => return Err(FalconError::InvalidQuery(format!("DATE_TRUNC: unknown unit '{unit}'"))),
                    };
                    Ok(ScalarValue::Date32(Some(truncated)))
                }
                _ => Ok(ScalarValue::Null),
            }
        }
        "GREATEST" => {
            let non_null: Vec<_> = args.iter().filter(|v| !v.is_null()).collect();
            if non_null.is_empty() { return Ok(ScalarValue::Null); }
            Ok(non_null.into_iter()
                .max_by(|a, b| {
                    let fa = a.as_f64().unwrap_or(f64::NEG_INFINITY);
                    let fb = b.as_f64().unwrap_or(f64::NEG_INFINITY);
                    fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|v| (*v).clone())
                .unwrap_or(ScalarValue::Null))
        }
        "LEAST" => {
            let non_null: Vec<_> = args.iter().filter(|v| !v.is_null()).collect();
            if non_null.is_empty() { return Ok(ScalarValue::Null); }
            Ok(non_null.into_iter()
                .min_by(|a, b| {
                    let fa = a.as_f64().unwrap_or(f64::INFINITY);
                    let fb = b.as_f64().unwrap_or(f64::INFINITY);
                    fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|v| (*v).clone())
                .unwrap_or(ScalarValue::Null))
        }
        "REPLACE" => {
            let s    = str_arg(args, 0)?;
            let from = str_arg(args, 1)?;
            let to   = str_arg(args, 2)?;
            Ok(match (s, from, to) {
                (Some(s), Some(f), Some(t)) => ScalarValue::Utf8(Some(s.replace(f, t))),
                _ => ScalarValue::Null,
            })
        }
        "LPAD" => {
            let s   = str_arg(args, 0)?;
            let len = if args.len() > 1 { num_arg(args, 1)? as usize } else { 0 };
            let pad = if args.len() > 2 { str_arg(args, 2)? } else { Some(" ") };
            Ok(match (s, pad) {
                (Some(s), Some(p)) => {
                    let chars: Vec<char> = s.chars().collect();
                    if chars.len() >= len {
                        ScalarValue::Utf8(Some(chars[..len].iter().collect()))
                    } else {
                        let deficit = len - chars.len();
                        let pad_chars: Vec<char> = p.chars().collect();
                        if pad_chars.is_empty() {
                            ScalarValue::Utf8(Some(s.to_string()))
                        } else {
                            let prefix: String = pad_chars.iter().cycle().take(deficit).collect();
                            ScalarValue::Utf8(Some(format!("{prefix}{s}")))
                        }
                    }
                }
                _ => ScalarValue::Null,
            })
        }
        "RPAD" => {
            let s   = str_arg(args, 0)?;
            let len = if args.len() > 1 { num_arg(args, 1)? as usize } else { 0 };
            let pad = if args.len() > 2 { str_arg(args, 2)? } else { Some(" ") };
            Ok(match (s, pad) {
                (Some(s), Some(p)) => {
                    let chars: Vec<char> = s.chars().collect();
                    if chars.len() >= len {
                        ScalarValue::Utf8(Some(chars[..len].iter().collect()))
                    } else {
                        let deficit = len - chars.len();
                        let pad_chars: Vec<char> = p.chars().collect();
                        if pad_chars.is_empty() {
                            ScalarValue::Utf8(Some(s.to_string()))
                        } else {
                            let suffix: String = pad_chars.iter().cycle().take(deficit).collect();
                            ScalarValue::Utf8(Some(format!("{s}{suffix}")))
                        }
                    }
                }
                _ => ScalarValue::Null,
            })
        }
        "REPEAT" => {
            let s = str_arg(args, 0)?;
            let n = num_arg(args, 1)? as usize;
            Ok(match s {
                Some(v) => ScalarValue::Utf8(Some(v.repeat(n))),
                None => ScalarValue::Null,
            })
        }
        "STRPOS" | "POSITION" => {
            // STRPOS(string, substring) — returns 1-based position, 0 if not found
            let s   = str_arg(args, 0)?;
            let sub = str_arg(args, 1)?;
            Ok(match (s, sub) {
                (Some(s), Some(sub)) => {
                    let pos = s.find(sub).map(|b| s[..b].chars().count() + 1).unwrap_or(0);
                    ScalarValue::Int64(Some(pos as i64))
                }
                _ => ScalarValue::Null,
            })
        }
        "REVERSE" => {
            let s = str_arg(args, 0)?;
            Ok(match s {
                Some(v) => ScalarValue::Utf8(Some(v.chars().rev().collect())),
                None => ScalarValue::Null,
            })
        }
        "LN" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.ln()))).unwrap_or(ScalarValue::Null))
        }
        "LOG" | "LOG10" => {
            let base = if name.eq_ignore_ascii_case("log") && args.len() > 1 {
                float_arg(args, 0)?
            } else {
                Some(10.0f64)
            };
            let v = if name.eq_ignore_ascii_case("log") && args.len() > 1 {
                float_arg(args, 1)?
            } else {
                float_arg(args, 0)?
            };
            Ok(match (base, v) {
                (Some(b), Some(x)) => ScalarValue::Float64(Some(x.log(b))),
                _ => ScalarValue::Null,
            })
        }
        "LOG2" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.log2()))).unwrap_or(ScalarValue::Null))
        }
        "EXP" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.exp()))).unwrap_or(ScalarValue::Null))
        }
        "MOD" => {
            let a = float_arg(args, 0)?;
            let b = float_arg(args, 1)?;
            Ok(match (a, b) {
                (Some(x), Some(y)) if y != 0.0 => ScalarValue::Float64(Some(x % y)),
                _ => ScalarValue::Null,
            })
        }
        "CHR" => {
            let n = num_arg(args, 0)?;
            let c = char::from_u32(n as u32).unwrap_or('?');
            Ok(ScalarValue::Utf8(Some(c.to_string())))
        }
        "ASCII" => {
            let s = str_arg(args, 0)?;
            Ok(match s {
                Some(v) => ScalarValue::Int64(Some(v.chars().next().map(|c| c as i64).unwrap_or(0))),
                None => ScalarValue::Null,
            })
        }
        "SPLIT_PART" => {
            let s   = str_arg(args, 0)?;
            let sep = str_arg(args, 1)?;
            let n   = num_arg(args, 2)? as usize;
            Ok(match (s, sep) {
                (Some(s), Some(sep)) => {
                    let part = s.split(sep).nth(n.saturating_sub(1))
                        .map(|p| ScalarValue::Utf8(Some(p.to_string())))
                        .unwrap_or(ScalarValue::Utf8(Some(String::new())));
                    part
                }
                _ => ScalarValue::Null,
            })
        }
        "PI" => Ok(ScalarValue::Float64(Some(std::f64::consts::PI))),
        "E"  => Ok(ScalarValue::Float64(Some(std::f64::consts::E))),
        "SIN" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.sin()))).unwrap_or(ScalarValue::Null))
        }
        "COS" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.cos()))).unwrap_or(ScalarValue::Null))
        }
        "TAN" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.tan()))).unwrap_or(ScalarValue::Null))
        }
        "ASIN" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.asin()))).unwrap_or(ScalarValue::Null))
        }
        "ACOS" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.acos()))).unwrap_or(ScalarValue::Null))
        }
        "ATAN" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.atan()))).unwrap_or(ScalarValue::Null))
        }
        "ATAN2" => {
            let y = float_arg(args, 0)?;
            let x = float_arg(args, 1)?;
            Ok(match (y, x) {
                (Some(y), Some(x)) => ScalarValue::Float64(Some(y.atan2(x))),
                _ => ScalarValue::Null,
            })
        }
        "DEGREES" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.to_degrees()))).unwrap_or(ScalarValue::Null))
        }
        "RADIANS" => {
            let v = float_arg(args, 0)?;
            Ok(v.map(|x| ScalarValue::Float64(Some(x.to_radians()))).unwrap_or(ScalarValue::Null))
        }
        "TRUNC" | "TRUNCATE" => {
            let v = float_arg(args, 0)?;
            let decimals = if args.len() > 1 { num_arg(args, 1)? as i32 } else { 0 };
            Ok(match v {
                None => ScalarValue::Null,
                Some(x) => {
                    let factor = 10f64.powi(decimals);
                    ScalarValue::Float64(Some((x * factor).trunc() / factor))
                }
            })
        }
        "INITCAP" => {
            let s = str_arg(args, 0)?;
            Ok(match s {
                Some(v) => {
                    let result: String = v.split_whitespace()
                        .map(|word| {
                            let mut c = word.chars();
                            match c.next() {
                                None => String::new(),
                                Some(f) => f.to_uppercase().to_string() + &c.as_str().to_lowercase(),
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    ScalarValue::Utf8(Some(result))
                }
                None => ScalarValue::Null,
            })
        }
        "LEFT" => {
            let s = str_arg(args, 0)?;
            let n = num_arg(args, 1)? as usize;
            Ok(match s {
                Some(v) => ScalarValue::Utf8(Some(v.chars().take(n).collect())),
                None => ScalarValue::Null,
            })
        }
        "RIGHT" => {
            let s = str_arg(args, 0)?;
            let n = num_arg(args, 1)? as usize;
            Ok(match s {
                Some(v) => {
                    let chars: Vec<char> = v.chars().collect();
                    let start = chars.len().saturating_sub(n);
                    ScalarValue::Utf8(Some(chars[start..].iter().collect()))
                }
                None => ScalarValue::Null,
            })
        }
        "CHAR_LENGTH" | "CHARACTER_LENGTH" => {
            let s = str_arg(args, 0)?;
            Ok(s.map(|v| ScalarValue::Int64(Some(v.chars().count() as i64)))
               .unwrap_or(ScalarValue::Null))
        }
        "OCTET_LENGTH" | "BYTE_LENGTH" => {
            let s = str_arg(args, 0)?;
            Ok(s.map(|v| ScalarValue::Int64(Some(v.len() as i64)))
               .unwrap_or(ScalarValue::Null))
        }
        // Not a built-in — try the user-defined function registry before giving up.
        other => match crate::query::udf::call(other, args) {
            Some(result) => result,
            None => Err(FalconError::InvalidQuery(format!("unknown function: {other}"))),
        },
    }
}

fn str_arg<'a>(args: &[&'a ScalarValue], idx: usize) -> FalconResult<Option<&'a str>> {
    match args.get(idx) {
        Some(ScalarValue::Utf8(Some(s))) => Ok(Some(s.as_str())),
        Some(ScalarValue::Utf8(None)) | Some(ScalarValue::Null) | None => Ok(None),
        Some(other) => Err(FalconError::InvalidQuery(
            format!("expected string argument, got {:?}", other.data_type())
        )),
    }
}

fn float_arg(args: &[&ScalarValue], idx: usize) -> FalconResult<Option<f64>> {
    match args.get(idx) {
        Some(v) => Ok(v.as_f64()),
        None => Ok(None),
    }
}

fn num_arg(args: &[&ScalarValue], idx: usize) -> FalconResult<f64> {
    float_arg(args, idx)?
        .ok_or_else(|| FalconError::InvalidQuery(format!("function argument {idx} is NULL")))
}

fn apply_binary_op(
    left:  &ScalarValue,
    op:    &BinaryOp,
    right: &ScalarValue,
) -> FalconResult<ScalarValue> {
    // Propagate NULLs.
    if left.is_null() || right.is_null() {
        return Ok(ScalarValue::Null);
    }

    match op {
        BinaryOp::And => {
            let l = left.as_bool().unwrap_or(false);
            let r = right.as_bool().unwrap_or(false);
            return Ok(ScalarValue::Boolean(Some(l && r)));
        }
        BinaryOp::Or => {
            let l = left.as_bool().unwrap_or(false);
            let r = right.as_bool().unwrap_or(false);
            return Ok(ScalarValue::Boolean(Some(l || r)));
        }
        _ => {}
    }

    // Date arithmetic.
    match (left, right, op) {
        (ScalarValue::Date32(Some(d)), ScalarValue::Int64(Some(n)), BinaryOp::Add) =>
            return Ok(ScalarValue::Date32(Some(*d + *n as i32))),
        (ScalarValue::Int64(Some(n)), ScalarValue::Date32(Some(d)), BinaryOp::Add) =>
            return Ok(ScalarValue::Date32(Some(*d + *n as i32))),
        (ScalarValue::Date32(Some(d)), ScalarValue::Int64(Some(n)), BinaryOp::Sub) =>
            return Ok(ScalarValue::Date32(Some(*d - *n as i32))),
        (ScalarValue::Date32(Some(a)), ScalarValue::Date32(Some(b)), BinaryOp::Sub) =>
            return Ok(ScalarValue::Int64(Some((*a - *b) as i64))),
        (ScalarValue::Timestamp(Some(ts)), ScalarValue::Int64(Some(n)), BinaryOp::Add) =>
            return Ok(ScalarValue::Timestamp(Some(ts + n))),
        (ScalarValue::Int64(Some(n)), ScalarValue::Timestamp(Some(ts)), BinaryOp::Add) =>
            return Ok(ScalarValue::Timestamp(Some(ts + n))),
        (ScalarValue::Timestamp(Some(ts)), ScalarValue::Int64(Some(n)), BinaryOp::Sub) =>
            return Ok(ScalarValue::Timestamp(Some(ts - n))),
        (ScalarValue::Timestamp(Some(a)), ScalarValue::Timestamp(Some(b)), BinaryOp::Sub) =>
            return Ok(ScalarValue::Int64(Some(a - b))),
        // Date comparisons
        (ScalarValue::Date32(Some(a)), ScalarValue::Date32(Some(b)), cmp_op) => {
            return Ok(ScalarValue::Boolean(Some(match cmp_op {
                BinaryOp::Eq    => a == b,
                BinaryOp::NotEq => a != b,
                BinaryOp::Lt    => a <  b,
                BinaryOp::LtEq  => a <= b,
                BinaryOp::Gt    => a >  b,
                BinaryOp::GtEq  => a >= b,
                _ => return Err(FalconError::InvalidQuery("invalid op on DATE".into())),
            })));
        }
        (ScalarValue::Timestamp(Some(a)), ScalarValue::Timestamp(Some(b)), cmp_op) => {
            return Ok(ScalarValue::Boolean(Some(match cmp_op {
                BinaryOp::Eq    => a == b,
                BinaryOp::NotEq => a != b,
                BinaryOp::Lt    => a <  b,
                BinaryOp::LtEq  => a <= b,
                BinaryOp::Gt    => a >  b,
                BinaryOp::GtEq  => a >= b,
                _ => return Err(FalconError::InvalidQuery("invalid op on TIMESTAMP".into())),
            })));
        }
        _ => {}
    }

    // Numeric operations.
    if let (Some(l), Some(r)) = (left.as_f64(), right.as_f64()) {
        return match op {
            BinaryOp::Add  => Ok(ScalarValue::Float64(Some(l + r))),
            BinaryOp::Sub  => Ok(ScalarValue::Float64(Some(l - r))),
            BinaryOp::Mul  => Ok(ScalarValue::Float64(Some(l * r))),
            BinaryOp::Div  => {
                if r == 0.0 {
                    Err(FalconError::DivisionByZero)
                } else {
                    Ok(ScalarValue::Float64(Some(l / r)))
                }
            }
            BinaryOp::Eq    => Ok(ScalarValue::Boolean(Some((l - r).abs() < f64::EPSILON))),
            BinaryOp::NotEq => Ok(ScalarValue::Boolean(Some((l - r).abs() >= f64::EPSILON))),
            BinaryOp::Lt    => Ok(ScalarValue::Boolean(Some(l < r))),
            BinaryOp::LtEq  => Ok(ScalarValue::Boolean(Some(l <= r))),
            BinaryOp::Gt    => Ok(ScalarValue::Boolean(Some(l > r))),
            BinaryOp::GtEq  => Ok(ScalarValue::Boolean(Some(l >= r))),
            _ => unreachable!(),
        };
    }

    // String operations.
    if let (ScalarValue::Utf8(Some(l)), ScalarValue::Utf8(Some(r))) = (left, right) {
        return match op {
            BinaryOp::Eq    => Ok(ScalarValue::Boolean(Some(l == r))),
            BinaryOp::NotEq => Ok(ScalarValue::Boolean(Some(l != r))),
            BinaryOp::Lt    => Ok(ScalarValue::Boolean(Some(l < r))),
            BinaryOp::LtEq  => Ok(ScalarValue::Boolean(Some(l <= r))),
            BinaryOp::Gt    => Ok(ScalarValue::Boolean(Some(l > r))),
            BinaryOp::GtEq  => Ok(ScalarValue::Boolean(Some(l >= r))),
            _ => Err(FalconError::InvalidQuery(
                "arithmetic not supported on strings".into(),
            )),
        };
    }

    Err(FalconError::TypeMismatch {
        expected: left.data_type(),
        got:      right.data_type(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dense evaluator must agree with `eval_batch`, value for value.
    ///
    /// It is a second definition of what an expression means, and this codebase has
    /// already paid twice for two implementations of one thing drifting apart, so it
    /// is pinned to the original rather than to a hand-written expectation.
    #[test]
    fn dense_eval_agrees_with_eval_batch() {
        use crate::storage::{column::ColumnBuilder, schema::{Field, Schema}, types::DataType};

        let schema = Schema::new(vec![
            Field::required("a", DataType::Float64),
            Field::required("b", DataType::Int64),
        ]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::float64((0..97).map(|i| Some(i as f64 - 40.0)).collect()),
            ColumnBuilder::int64((0..97).map(|i| Some(i as i64 * 3)).collect()),
        ]).unwrap();

        let exprs = vec![
            Expr::Column("a".into()),
            Expr::Column("b".into()),
            Expr::BinaryOp {
                left:  Box::new(Expr::Column("a".into())),
                op:    BinaryOp::Add,
                right: Box::new(Expr::Literal(ScalarValue::Float64(Some(1.5)))),
            },
            Expr::BinaryOp {
                left:  Box::new(Expr::Column("a".into())),
                op:    BinaryOp::Mul,
                right: Box::new(Expr::Column("b".into())),
            },
        ];

        for e in &exprs {
            let dense = e.eval_dense_f64(&batch).expect("should take the dense path");
            let boxed = e.eval_batch(&batch).unwrap();
            assert_eq!(dense.len(), boxed.len());
            for (i, (d, b)) in dense.iter().zip(&boxed).enumerate() {
                let bv = b.as_f64().expect("numeric");
                assert!((d - bv).abs() < 1e-9, "row {i}: dense {d} vs boxed {bv} for {e:?}");
            }
        }
    }

    /// Shapes the dense evaluator must decline instead of approximating.
    #[test]
    fn dense_eval_declines_nulls_and_non_numeric() {
        use crate::storage::{column::ColumnBuilder, schema::{Field, Schema}, types::DataType};

        let schema = Schema::new(vec![
            Field::nullable("a", DataType::Float64),
            Field::required("s", DataType::Utf8),
        ]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::float64(vec![Some(1.0), None, Some(3.0)]),
            ColumnBuilder::utf8_from_str(vec![Some("x"), Some("y"), Some("z")]),
        ]).unwrap();

        // A NULL makes the result NULL, which needs a validity mask this form has not got.
        assert!(Expr::Column("a".into()).eval_dense_f64(&batch).is_none());
        // Text is not numeric.
        assert!(Expr::Column("s".into()).eval_dense_f64(&batch).is_none());
        // A comparison is not arithmetic; predicates have their own mask evaluator.
        assert!(Expr::BinaryOp {
            left:  Box::new(Expr::Column("s".into())),
            op:    BinaryOp::Gt,
            right: Box::new(Expr::Literal(ScalarValue::Float64(Some(0.0)))),
        }.eval_dense_f64(&batch).is_none());
    }
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::RecordBatch,
        types::DataType,
    };

    fn make_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::required("age", DataType::Int64),
            Field::nullable("name", DataType::Utf8),
        ]);
        RecordBatch::new(schema, vec![
            ColumnBuilder::int64(vec![Some(10), Some(25), Some(40)]),
            ColumnBuilder::utf8_from_str(vec![Some("alice"), Some("bob"), Some("carol")]),
        ])
        .unwrap()
    }

    #[test]
    fn literal_expr() {
        let batch = make_batch();
        let expr = Expr::Literal(ScalarValue::Int64(Some(42)));
        let result = expr.eval_batch(&batch).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v == &ScalarValue::Int64(Some(42))));
    }

    #[test]
    fn filter_gt() {
        let batch = make_batch();
        let expr = Expr::BinaryOp {
            left:  Box::new(Expr::Column("age".into())),
            op:    BinaryOp::Gt,
            right: Box::new(Expr::Literal(ScalarValue::Int64(Some(20)))),
        };
        let mask = expr.eval_mask(&batch).unwrap();
        assert_eq!(mask, vec![false, true, true]);
    }

    #[test]
    fn division_by_zero() {
        let batch = make_batch();
        let expr = Expr::BinaryOp {
            left:  Box::new(Expr::Literal(ScalarValue::Float64(Some(1.0)))),
            op:    BinaryOp::Div,
            right: Box::new(Expr::Literal(ScalarValue::Float64(Some(0.0)))),
        };
        assert!(matches!(
            expr.eval_batch(&batch).unwrap_err(),
            FalconError::DivisionByZero
        ));
    }

    #[test]
    fn is_null() {
        let schema = Schema::new(vec![Field::nullable("x", DataType::Int64)]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::int64(vec![Some(1), None, Some(3)]),
        ])
        .unwrap();

        let expr = Expr::IsNull(Box::new(Expr::Column("x".into())));
        let mask = expr.eval_mask(&batch).unwrap();
        assert_eq!(mask, vec![false, true, false]);
    }
}

#[cfg(test)]
mod in_list_semantics {
    /// Does `IN` agree with `=` when the literal's type differs from the column's?
    ///
    /// `IN` compares boxed `ScalarValue`s with a derived `PartialEq`, which is
    /// strict about the variant, while `=` goes through `fast_cmp_mask`, which has
    /// arms for widening an integer literal to a float column. If those two
    /// disagree, that is a correctness question to settle before making either
    /// faster.
    #[test]
    fn in_agrees_with_equality_across_numeric_types() {
        use crate::db::Database;
        use crate::storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::RecordBatch,
            types::DataType,
        };
        let batch = RecordBatch::new(
            Schema::new(vec![
                Field::required("f", DataType::Float64),
                Field::required("i", DataType::Int64),
            ]),
            vec![
                ColumnBuilder::float64((1..=5).map(|i| Some(i as f64)).collect()),
                ColumnBuilder::int64((1..=5).map(Some).collect()),
            ],
        )
        .unwrap();
        let db = Database::in_memory();
        db.create_table_from_batches("t", vec![batch]).unwrap();

        let n = |sql: &str| -> usize {
            db.sql(sql).unwrap().batches.iter().map(|b| b.row_count()).sum()
        };

        println!("f = 1        -> {}", n("SELECT f FROM t WHERE f = 1"));
        println!("f IN (1,2)   -> {}", n("SELECT f FROM t WHERE f IN (1, 2)"));
        println!("i = 1        -> {}", n("SELECT i FROM t WHERE i = 1"));
        println!("i IN (1,2)   -> {}", n("SELECT i FROM t WHERE i IN (1, 2)"));
        println!("f IN (1.0,2.0) -> {}", n("SELECT f FROM t WHERE f IN (1.0, 2.0)"));

        // The point of the fix: `IN` coerces the way `=` does.
        assert_eq!(n("SELECT f FROM t WHERE f IN (1, 2)"), 2);
        assert_eq!(n("SELECT f FROM t WHERE f IN (1.0, 2.0)"), 2);
        assert_eq!(n("SELECT i FROM t WHERE i IN (1, 2)"), 2);
    }

    /// Which other forms of "is this value that value" disagree with `=`?
    ///
    /// `IN` did, and the cause was general: comparing boxed `ScalarValue`s with a
    /// derived `PartialEq`, which is strict about the variant, while `=` goes
    /// through a path with widening arms. Any form that reaches the boxed
    /// comparison has the same defect available to it, so this asks all of them
    /// the same question at once: an integer literal against a `Float64` column
    /// holding 1.0, 2.0, 3.0.
    ///
    /// Every row of the report below should agree with the `=` row. A form that
    /// does not is a bug of the same family, not a quirk of that syntax.
    #[test]
    fn every_comparison_form_coerces_like_equality() {
        use crate::db::Database;
        use crate::storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::RecordBatch,
            types::DataType,
        };

        let batch = RecordBatch::new(
            Schema::new(vec![
                Field::required("f", DataType::Float64),
                Field::required("i", DataType::Int64),
            ]),
            vec![
                ColumnBuilder::float64((1..=3).map(|i| Some(i as f64)).collect()),
                ColumnBuilder::int64((1..=3).map(Some).collect()),
            ],
        )
        .unwrap();
        let db = Database::in_memory();
        db.create_table_from_batches("t", vec![batch]).unwrap();

        let n = |sql: &str| -> usize {
            db.sql(sql).unwrap().batches.iter().map(|b| b.row_count()).sum()
        };

        // Each of these asks "which rows of f equal 1 (or are 1 or 2)", written a
        // different way. The integer literal is the point: the column is Float64.
        let forms: Vec<(&str, usize, usize)> = vec![
            ("f = 1",                    n("SELECT f FROM t WHERE f = 1"), 1),
            ("f IN (1, 2)",              n("SELECT f FROM t WHERE f IN (1, 2)"), 2),
            ("f BETWEEN 1 AND 2",        n("SELECT f FROM t WHERE f BETWEEN 1 AND 2"), 2),
            ("f >= 1 AND f <= 2",        n("SELECT f FROM t WHERE f >= 1 AND f <= 2"), 2),
            ("NOT (f <> 1)",             n("SELECT f FROM t WHERE NOT (f <> 1)"), 1),
            ("CASE WHEN f = 1",          n("SELECT f FROM t WHERE CASE WHEN f = 1 THEN true ELSE false END"), 1),
            ("f = 1 OR f = 2",           n("SELECT f FROM t WHERE f = 1 OR f = 2"), 2),
        ];

        let mut wrong = Vec::new();
        for (form, got, want) in &forms {
            println!("  {form:<24} -> {got} rows (expected {want})");
            if got != want {
                wrong.push(format!("{form}: got {got}, expected {want}"));
            }
        }
        assert!(wrong.is_empty(), "forms disagreeing with `=`:
  {}", wrong.join("
  "));
    }

    /// The same question, asked where row *membership* is decided rather than a
    /// filter: join keys, `GROUP BY`, `UNION`, `IN (SELECT ...)`.
    ///
    /// These all match values against values, and each one that reaches the boxed
    /// comparison can disagree with `=` the way `IN` did. `WHERE` turned out to be
    /// clean once `IN` was fixed; this checks the rest of the engine rather than
    /// assuming the family ends there.
    ///
    /// An Int64 column and a Float64 column holding the same three numbers should
    /// join on three rows, group to three groups, and union to three values.
    #[test]
    fn membership_across_numeric_types() {
        use crate::db::Database;
        use crate::storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::RecordBatch,
            types::DataType,
        };

        let ints = RecordBatch::new(
            Schema::new(vec![Field::required("i", DataType::Int64)]),
            vec![ColumnBuilder::int64((1..=3).map(Some).collect())],
        )
        .unwrap();
        let floats = RecordBatch::new(
            Schema::new(vec![Field::required("f", DataType::Float64)]),
            vec![ColumnBuilder::float64((1..=3).map(|i| Some(i as f64)).collect())],
        )
        .unwrap();
        let db = Database::in_memory();
        db.create_table_from_batches("ints", vec![ints]).unwrap();
        db.create_table_from_batches("floats", vec![floats]).unwrap();

        let n = |sql: &str| -> usize {
            match db.sql(sql) {
                Ok(r) => r.batches.iter().map(|b| b.row_count()).sum(),
                Err(e) => {
                    println!("      (error: {e})");
                    usize::MAX
                }
            }
        };

        let cases: Vec<(&str, usize, usize)> = vec![
            ("JOIN ON i = f",
             n("SELECT i FROM ints JOIN floats ON ints.i = floats.f"), 3),
            ("JOIN, qualified select",
             n("SELECT ints.i FROM ints JOIN floats ON ints.i = floats.f"), 3),
            ("i IN (SELECT f)",
             n("SELECT i FROM ints WHERE i IN (SELECT f FROM floats)"), 3),
            ("EXISTS correlated",
             n("SELECT i FROM ints WHERE EXISTS (SELECT 1 FROM floats WHERE floats.f = ints.i)"), 3),
        ];

        let mut wrong = Vec::new();
        for (form, got, want) in &cases {
            let shown = if *got == usize::MAX { "error".to_string() } else { got.to_string() };
            println!("  {form:<24} -> {shown} rows (expected {want})");
            if got != want {
                wrong.push(format!("{form}: got {shown}, expected {want}"));
            }
        }
        assert!(wrong.is_empty(), "membership forms that do not match across numeric types:
  {}", wrong.join("
  "));
    }

    /// The typed `IN` path must answer exactly what the general path answered,
    /// including for NULLs, `NOT IN`, text, and a list with no match in it.
    #[test]
    fn typed_in_matches_the_general_path() {
        use crate::db::Database;
        use crate::storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::RecordBatch,
            types::DataType,
        };

        let batch = RecordBatch::new(
            Schema::new(vec![
                Field::nullable("i", DataType::Int64),
                Field::nullable("s", DataType::Utf8),
            ]),
            vec![
                ColumnBuilder::int64(vec![Some(1), Some(2), None, Some(3), Some(2)]),
                ColumnBuilder::utf8_from_str(vec![
                    Some("a"), Some("b"), None, Some("c"), Some("b"),
                ]),
            ],
        )
        .unwrap();
        let db = Database::in_memory();
        db.create_table_from_batches("t", vec![batch]).unwrap();

        let n = |sql: &str| -> usize {
            db.sql(sql).unwrap().batches.iter().map(|b| b.row_count()).sum()
        };

        // A NULL row is in neither the IN set nor its complement.
        assert_eq!(n("SELECT i FROM t WHERE i IN (1, 2)"), 3);
        assert_eq!(n("SELECT i FROM t WHERE i NOT IN (1, 2)"), 1);
        assert_eq!(n("SELECT s FROM t WHERE s IN ('a', 'b')"), 3);
        assert_eq!(n("SELECT s FROM t WHERE s NOT IN ('a', 'b')"), 1);
        // Nothing in the list occurs.
        assert_eq!(n("SELECT i FROM t WHERE i IN (7, 8)"), 0);
        assert_eq!(n("SELECT s FROM t WHERE s IN ('z')"), 0);
        // NOT IN over a list that matches nothing keeps every non-NULL row.
        assert_eq!(n("SELECT i FROM t WHERE i NOT IN (7)"), 4);
        // An empty-ish edge: one member.
        assert_eq!(n("SELECT i FROM t WHERE i IN (2)"), 2);
        // AND/OR compose with the new arm.
        assert_eq!(n("SELECT i FROM t WHERE i IN (1, 2) AND i > 1"), 2);
        assert_eq!(n("SELECT i FROM t WHERE i IN (1) OR i IN (3)"), 2);
    }

    /// What `IN` costs now, against the single equality it is built from.
    ///
    /// Run with
    /// `cargo test --release --lib in_list_speed -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn in_list_speed() {
        use crate::db::Database;
        use crate::storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::RecordBatch,
            types::DataType,
        };
        use std::time::Instant;

        const ROWS: usize = 100_000;
        let batch = RecordBatch::new(
            Schema::new(vec![
                Field::required("dept", DataType::Utf8),
                Field::required("bucket", DataType::Int64),
            ]),
            vec![
                ColumnBuilder::utf8_from_str(
                    (0..ROWS)
                        .map(|i| Some(["eng", "sales", "hr", "ops", "legal"][i % 5]))
                        .collect(),
                ),
                ColumnBuilder::int64((0..ROWS).map(|i| Some((i % 5) as i64)).collect()),
            ],
        )
        .unwrap();
        let db = Database::in_memory();
        db.create_table_from_batches("t", vec![batch]).unwrap();

        let time = |label: &str, sql: &str| {
            for _ in 0..3 { db.sql(sql).unwrap(); }
            let mut best = std::time::Duration::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                std::hint::black_box(db.sql(sql).unwrap());
                best = best.min(t.elapsed());
            }
            println!("  {label:>36}: {best:>9.2?}");
            best
        };

        println!("over {ROWS} rows:");
        let eq = time("bucket = 1", "SELECT COUNT(*) FROM t WHERE bucket = 1");
        let ii = time("bucket IN (1,2,3)", "SELECT COUNT(*) FROM t WHERE bucket IN (1,2,3)");
        let et = time("dept = 'eng'", "SELECT COUNT(*) FROM t WHERE dept = 'eng'");
        let it = time("dept IN ('eng','hr','ops')", "SELECT COUNT(*) FROM t WHERE dept IN ('eng','hr','ops')");
        let ni = time("bucket NOT IN (1,2,3)", "SELECT COUNT(*) FROM t WHERE bucket NOT IN (1,2,3)");
        println!("    integer IN costs {:.1}x one equality", ii.as_secs_f64() / eq.as_secs_f64());
        println!("    text IN costs {:.1}x one text equality", it.as_secs_f64() / et.as_secs_f64());
        println!("    NOT IN costs {:.1}x IN", ni.as_secs_f64() / ii.as_secs_f64());
    }
}
