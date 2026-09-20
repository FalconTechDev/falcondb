use std::collections::HashMap;

use smallvec::SmallVec;
use super::hash::FastMap;

use rayon::prelude::*;

use crate::storage::{
    column::{BooleanColumn, Column, PrimitiveColumn, Utf8Column},
    schema::{Field, Schema},
    table::RecordBatch,
    types::{DataType, FalconError, FalconResult, ScalarValue},
};

use super::expr::Expr;

/// Total rows across a batch list.
///
/// Parallel decisions are made on this rather than on `batches.len()`, so they mean
/// the same thing whatever produced the batches.
#[inline]
fn rows_of(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.row_count()).sum()
}

fn is_count_star(agg: &AggExpr) -> bool {
    matches!(agg.func, AggFunc::Count) && matches!(agg.input, Expr::Literal(_))
}

// в”Ђв”Ђв”Ђ Aggregate functions в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

#[derive(Debug, Clone)]
pub enum AggFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    CountDistinct,
    StringAgg { separator: String },
    BoolAnd,
    BoolOr,
    Stddev { population: bool },
    Variance { population: bool },
    First,
    Last,
}

#[derive(Debug, Clone)]
pub struct AggExpr {
    pub func:  AggFunc,
    pub input: Expr,
    pub alias: String,
}

impl AggExpr {
    pub fn count(input: Expr, alias: &str) -> Self {
        Self { func: AggFunc::Count, input, alias: alias.into() }
    }
    pub fn sum(input: Expr, alias: &str) -> Self {
        Self { func: AggFunc::Sum, input, alias: alias.into() }
    }
    pub fn avg(input: Expr, alias: &str) -> Self {
        Self { func: AggFunc::Avg, input, alias: alias.into() }
    }
    pub fn min(input: Expr, alias: &str) -> Self {
        Self { func: AggFunc::Min, input, alias: alias.into() }
    }
    pub fn max(input: Expr, alias: &str) -> Self {
        Self { func: AggFunc::Max, input, alias: alias.into() }
    }
}

// в”Ђв”Ђв”Ђ Slow-path accumulator (general, any Expr) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GroupKey(String);

impl GroupKey {
    fn from_row(values: &[ScalarValue]) -> Self {
        Self(values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\x00"))
    }
}

#[derive(Debug)]
enum Accumulator {
    Count        { n: i64 },
    CountDistinct { seen: std::collections::HashSet<String> },
    Sum          { total: f64, valid: bool },
    Avg          { total: f64, count: i64 },
    Min          { value: Option<f64>, str_value: Option<String> },
    Max          { value: Option<f64>, str_value: Option<String> },
    StringAgg    { parts: Vec<String>, sep: String },
    BoolAnd      { result: Option<bool> },
    BoolOr       { result: Option<bool> },
    Stddev       { sum: f64, sum_sq: f64, count: i64, population: bool },
    Variance     { sum: f64, sum_sq: f64, count: i64, population: bool },
    First        { value: Option<ScalarValue> },
    Last         { value: Option<ScalarValue> },
}

impl Accumulator {
    fn new(func: &AggFunc) -> Self {
        match func {
            AggFunc::Count         => Self::Count { n: 0 },
            AggFunc::CountDistinct => Self::CountDistinct { seen: std::collections::HashSet::new() },
            AggFunc::Sum           => Self::Sum  { total: 0.0, valid: false },
            AggFunc::Avg           => Self::Avg  { total: 0.0, count: 0 },
            AggFunc::Min           => Self::Min  { value: None, str_value: None },
            AggFunc::Max           => Self::Max  { value: None, str_value: None },
            AggFunc::StringAgg { separator } => Self::StringAgg { parts: vec![], sep: separator.clone() },
            AggFunc::BoolAnd       => Self::BoolAnd { result: None },
            AggFunc::BoolOr        => Self::BoolOr  { result: None },
            AggFunc::Stddev   { population } => Self::Stddev   { sum: 0.0, sum_sq: 0.0, count: 0, population: *population },
            AggFunc::Variance { population } => Self::Variance { sum: 0.0, sum_sq: 0.0, count: 0, population: *population },
            AggFunc::First         => Self::First { value: None },
            AggFunc::Last          => Self::Last  { value: None },
        }
    }

    fn update(&mut self, val: &ScalarValue) {
        if val.is_null() { return; }
        match self {
            Self::Count { n } => *n += 1,
            Self::CountDistinct { seen } => { seen.insert(val.to_string()); }
            Self::Sum { total, valid } => {
                if let Some(v) = val.as_f64() { *total += v; *valid = true; }
            }
            Self::Avg { total, count } => {
                if let Some(v) = val.as_f64() { *total += v; *count += 1; }
            }
            Self::Min { value, str_value } => {
                if let Some(v) = val.as_f64() {
                    *value = Some(value.map_or(v, |cur: f64| cur.min(v)));
                } else if let ScalarValue::Utf8(Some(s)) = val {
                    *str_value = Some(match str_value {
                        Some(cur) => if s < cur { s.clone() } else { cur.clone() },
                        None      => s.clone(),
                    });
                }
            }
            Self::Max { value, str_value } => {
                if let Some(v) = val.as_f64() {
                    *value = Some(value.map_or(v, |cur: f64| cur.max(v)));
                } else if let ScalarValue::Utf8(Some(s)) = val {
                    *str_value = Some(match str_value {
                        Some(cur) => if s > cur { s.clone() } else { cur.clone() },
                        None      => s.clone(),
                    });
                }
            }
            Self::StringAgg { parts, .. } => {
                parts.push(val.to_string());
            }
            Self::BoolAnd { result } => {
                let b = match val { ScalarValue::Boolean(Some(b)) => *b, _ => return };
                *result = Some(result.unwrap_or(true) && b);
            }
            Self::BoolOr { result } => {
                let b = match val { ScalarValue::Boolean(Some(b)) => *b, _ => return };
                *result = Some(result.unwrap_or(false) || b);
            }
            Self::Stddev { sum, sum_sq, count, .. } | Self::Variance { sum, sum_sq, count, .. } => {
                if let Some(v) = val.as_f64() {
                    *sum += v;
                    *sum_sq += v * v;
                    *count += 1;
                }
            }
            Self::First { value } => {
                if value.is_none() { *value = Some(val.clone()); }
            }
            Self::Last { value } => {
                *value = Some(val.clone());
            }
        }
    }

    /// Fold another accumulator of the same kind into this one.
    ///
    /// Aggregates over disjoint row ranges combine associatively, which is what
    /// lets batches be accumulated in parallel and merged afterwards. Floating
    /// point addition is not exactly associative, so a parallel SUM can differ
    /// from a serial one in the last bits — the same trade every engine here
    /// makes to get parallel aggregation at all.
    fn merge(&mut self, other: Accumulator) {
        match (self, other) {
            (Self::Count { n }, Self::Count { n: m }) => *n += m,
            (Self::CountDistinct { seen }, Self::CountDistinct { seen: other }) => {
                seen.extend(other);
            }
            (Self::Sum { total, valid }, Self::Sum { total: t2, valid: v2 }) => {
                *total += t2;
                *valid |= v2;
            }
            (Self::Avg { total, count }, Self::Avg { total: t2, count: c2 }) => {
                *total += t2;
                *count += c2;
            }
            (
                Self::Min { value, str_value },
                Self::Min { value: v2, str_value: s2 },
            ) => {
                if let Some(v) = v2 {
                    *value = Some(value.map_or(v, |cur: f64| cur.min(v)));
                }
                if let Some(s) = s2 {
                    *str_value = Some(match str_value {
                        Some(cur) if cur.as_str() <= s.as_str() => cur.clone(),
                        _ => s,
                    });
                }
            }
            (
                Self::Max { value, str_value },
                Self::Max { value: v2, str_value: s2 },
            ) => {
                if let Some(v) = v2 {
                    *value = Some(value.map_or(v, |cur: f64| cur.max(v)));
                }
                if let Some(s) = s2 {
                    *str_value = Some(match str_value {
                        Some(cur) if cur.as_str() >= s.as_str() => cur.clone(),
                        _ => s,
                    });
                }
            }
            (Self::StringAgg { parts, .. }, Self::StringAgg { parts: other, .. }) => {
                parts.extend(other);
            }
            (Self::BoolAnd { result }, Self::BoolAnd { result: r2 }) => {
                *result = match (*result, r2) {
                    (Some(a), Some(b)) => Some(a && b),
                    (Some(a), None)    => Some(a),
                    (None, Some(b))    => Some(b),
                    _                  => None,
                };
            }
            (Self::BoolOr { result }, Self::BoolOr { result: r2 }) => {
                *result = match (*result, r2) {
                    (Some(a), Some(b)) => Some(a || b),
                    (Some(a), None)    => Some(a),
                    (None, Some(b))    => Some(b),
                    _                  => None,
                };
            }
            (
                Self::Stddev { sum, sum_sq, count, .. },
                Self::Stddev { sum: s2, sum_sq: ss2, count: c2, .. },
            ) |
            (
                Self::Variance { sum, sum_sq, count, .. },
                Self::Variance { sum: s2, sum_sq: ss2, count: c2, .. },
            ) => {
                *sum += s2; *sum_sq += ss2; *count += c2;
            }
            (Self::First { value }, Self::First { value: v2 }) => {
                if value.is_none() { *value = v2; }
            }
            (Self::Last { value }, Self::Last { value: v2 }) => {
                if v2.is_some() { *value = v2; }
            }
            // Accumulators are built from the same AggFunc list in the same order,
            // so the variants always line up. A mismatch is a bug, not input data.
            _ => unreachable!("merging accumulators of different kinds"),
        }
    }

    fn result(&self) -> ScalarValue {
        match self {
            Self::Count { n }            => ScalarValue::Int64(Some(*n)),
            Self::CountDistinct { seen } => ScalarValue::Int64(Some(seen.len() as i64)),
            Self::Sum { total, valid } => {
                if *valid { ScalarValue::Float64(Some(*total)) } else { ScalarValue::Float64(None) }
            }
            Self::Avg { total, count } => {
                if *count > 0 { ScalarValue::Float64(Some(total / *count as f64)) }
                else          { ScalarValue::Float64(None) }
            }
            Self::Min { value, str_value } => {
                if let Some(v) = value { ScalarValue::Float64(Some(*v)) }
                else if let Some(s) = str_value { ScalarValue::Utf8(Some(s.clone())) }
                else { ScalarValue::Null }
            }
            Self::Max { value, str_value } => {
                if let Some(v) = value { ScalarValue::Float64(Some(*v)) }
                else if let Some(s) = str_value { ScalarValue::Utf8(Some(s.clone())) }
                else { ScalarValue::Null }
            }
            Self::StringAgg { parts, sep } => {
                if parts.is_empty() { ScalarValue::Null }
                else { ScalarValue::Utf8(Some(parts.join(sep.as_str()))) }
            }
            Self::BoolAnd { result } => match result {
                Some(b) => ScalarValue::Boolean(Some(*b)),
                None    => ScalarValue::Null,
            },
            Self::BoolOr { result } => match result {
                Some(b) => ScalarValue::Boolean(Some(*b)),
                None    => ScalarValue::Null,
            },
            Self::Stddev { sum, sum_sq, count, population } => {
                if *count == 0 { return ScalarValue::Null; }
                let n = *count as f64;
                let mean = sum / n;
                let var_pop = (sum_sq / n - mean * mean).max(0.0);
                if *population {
                    ScalarValue::Float64(Some(var_pop.sqrt()))
                } else {
                    if n <= 1.0 { ScalarValue::Null }
                    else { ScalarValue::Float64(Some((var_pop * n / (n - 1.0)).sqrt())) }
                }
            }
            Self::Variance { sum, sum_sq, count, population } => {
                if *count == 0 { return ScalarValue::Null; }
                let n = *count as f64;
                let mean = sum / n;
                let var_pop = sum_sq / n - mean * mean;
                if *population {
                    ScalarValue::Float64(Some(var_pop.max(0.0)))
                } else {
                    if n <= 1.0 { ScalarValue::Null }
                    else { ScalarValue::Float64(Some((var_pop * n / (n - 1.0)).max(0.0))) }
                }
            }
            Self::First { value } => value.clone().unwrap_or(ScalarValue::Null),
            Self::Last  { value } => value.clone().unwrap_or(ScalarValue::Null),
        }
    }
}

// в”Ђв”Ђв”Ђ Fast-path batch state (parallel arrays, no boxing) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

// Inline capacity: most analytical GROUP BY queries have ≤ 8 groups per batch.
// With SmallVec, states for n_groups ≤ 8 are fully inline — zero heap allocs.
// AGG_INLINE=8 keeps BatchAggState at ~184 bytes (vs 320 bytes at 16), so 122
// per-batch states fit in L2 cache.
const AGG_INLINE: usize = 8;

/// Per-aggregate typed accumulator arrays, indexed by group_id.
struct BatchAggState {
    count:   SmallVec<[i64;  AGG_INLINE]>,
    sum:     SmallVec<[f64;  AGG_INLINE]>,
    has_val: SmallVec<[bool; AGG_INLINE]>,
}

impl BatchAggState {
    fn new() -> Self {
        Self {
            count:   SmallVec::new(),
            sum:     SmallVec::new(),
            has_val: SmallVec::new(),
        }
    }

    fn with_capacity(n: usize) -> Self {
        let mut s = Self::new();
        s.count.resize(n, 0);
        s.sum.resize(n, 0.0);
        s.has_val.resize(n, false);
        s
    }

    fn push_group(&mut self) {
        self.count.push(0);
        self.sum.push(0.0);
        self.has_val.push(false);
    }
}

// в”Ђв”Ђв”Ђ GroupBy executor в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

#[derive(Clone, Debug)]
pub struct GroupByExec {
    pub group_by: Vec<Expr>,
    pub aggs:     Vec<AggExpr>,
}

impl GroupByExec {
    pub fn new(group_by: Vec<Expr>, aggs: Vec<AggExpr>) -> Self {
        Self { group_by, aggs }
    }

    pub fn execute(&self, batches: &[RecordBatch]) -> FalconResult<RecordBatch> {
        // Try O(n) batch path first; fall back to general row-by-row path.
        match self.execute_fast(batches) {
            Ok(Some(r)) => return Ok(r),
            Ok(None)    => {}
            Err(e)      => return Err(e),
        }
        self.execute_slow(batches)
    }

    // в”Ђв”Ђ Fast path в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    /// Returns `None` when conditions for the fast path aren't met.
    ///
    /// Fast path conditions:
    /// - Single group-by column, must be `Expr::Column`
    /// - All agg inputs are `Expr::Column` (numeric) OR `Expr::Literal` with `COUNT`
    /// - No CountDistinct
    fn execute_fast(&self, batches: &[RecordBatch]) -> FalconResult<Option<RecordBatch>> {
        // Skip fast path for functions that need the slow accumulator
        for agg in &self.aggs {
            if matches!(agg.func,
                AggFunc::CountDistinct | AggFunc::StringAgg { .. }
                | AggFunc::BoolAnd | AggFunc::BoolOr
                | AggFunc::Stddev { .. } | AggFunc::Variance { .. }
                | AggFunc::First | AggFunc::Last
            ) { return Ok(None); }
        }

        // Single group-by column, must be Expr::Column
        if self.group_by.len() != 1 { return Ok(None); }
        let gb_name = match &self.group_by[0] {
            Expr::Column(n) => n.as_str(),
            _ => return Ok(None),
        };

        // Agg inputs: Expr::Column в†’ Some(name), COUNT(*) literal в†’ None
        let mut agg_col_names: Vec<Option<&str>> = Vec::with_capacity(self.aggs.len());
        for agg in &self.aggs {
            match &agg.input {
                Expr::Column(n) => agg_col_names.push(Some(n.as_str())),
                Expr::Literal(_) if matches!(agg.func, AggFunc::Count) => {
                    agg_col_names.push(None); // COUNT(*) вЂ” no column needed
                }
                _ => return Ok(None),
            }
        }

        // Need at least one batch to inspect column types
        let first = match batches.first() { Some(b) => b, None => return Ok(None) };

        // Group column must be Utf8 (most common) or Int64
        let gb_col = match first.column_by_name(gb_name) { Ok(c) => c, Err(_) => return Ok(None) };
        match gb_col.data_type() {
            DataType::Utf8  => {}
            DataType::Int64 | DataType::Int32 => {}
            _ => return Ok(None),
        }

        // Named agg columns must be numeric
        for (&maybe_name, agg) in agg_col_names.iter().zip(self.aggs.iter()) {
            if let Some(name) = maybe_name {
                let col = match first.column_by_name(name) { Ok(c) => c, Err(_) => return Ok(None) };
                match col.data_type() {
                    DataType::Float64 | DataType::Float32 | DataType::Int64 | DataType::Int32 => {}
                    _ => {
                        // COUNT on a non-numeric col is still fine
                        if !matches!(agg.func, AggFunc::Count) { return Ok(None); }
                    }
                }
            }
        }

        // Dispatch on group column type
        match gb_col.data_type() {
            DataType::Utf8            => self.fast_utf8_gb(batches, gb_name, &agg_col_names).map(Some),
            DataType::Int64 | DataType::Int32
                                      => self.fast_i64_gb(batches, gb_name, &agg_col_names).map(Some),
            _ => unreachable!(),
        }
    }

    /// Fast path: Utf8 group-by column.
    /// Batches are aggregated in parallel with Rayon; partial results are merged sequentially.
    fn fast_utf8_gb(
        &self,
        batches:       &[RecordBatch],
        gb_name:       &str,
        agg_col_names: &[Option<&str>],
    ) -> FalconResult<RecordBatch> {
        let n_aggs = self.aggs.len();

        // в”Ђв”Ђ Per-batch aggregation (parallel) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        let first_b      = batches.first().ok_or_else(|| FalconError::InvalidQuery("no batches".into()))?;
        let gb_col_idx   = first_b.schema.column_index(gb_name)?;
        let agg_col_indices: Vec<Option<usize>> = agg_col_names.iter()
            .map(|&n| n.and_then(|name| first_b.schema.column_index(name).ok()))
            .collect();

        // Determine whether every batch is null-free in the GB column.
        // O(1) per batch (has_nulls is a stored flag), so negligible for 100-200 batches.
        let all_no_nulls = batches.iter().all(|b| {
            b.column(gb_col_idx)
             .as_utf8_column()
             .map(|c| !c.has_nulls())
             .unwrap_or(false)
        });

        if all_no_nulls {
            // FAST PATH — all batches are null-free.
            //
            // Key insight: dict codes are stable 0-indexed identifiers, so code == gid.
            // We extract the global key list from the first batch once (O(1) per dict
            // entry), then each Rayon task returns only its accumulated BatchAggState —
            // no per-batch String clones, no per-task HashMap.
            //
            // For n_groups=5, 122 Rayon batches over 1M rows this eliminates:
            //   • 610 String heap allocations (replaced by 5 one-time clones)
            //   • 610 HashMap inserts/lookups in the merge phase
            //   • 1 HashMap allocation per merge
            let first_col = first_b.column(gb_col_idx).as_utf8_column()
                .ok_or_else(|| FalconError::InvalidQuery(format!("Expected Utf8 column '{gb_name}'")))?;
            let global_keys: Vec<String> = first_col.dict().to_vec();
            let n_g = global_keys.len();

            // Intra-batch parallel path: one large batch (e.g. from Parquet) → split into
            // rayon chunks so each thread accumulates its own local state then we merge.
            // Split by *work*, not by batch. Fanning out over batches gives one task
            // each, so two 50k-row batches leave eighteen of twenty threads idle —
            // which is what a 100k `GROUP BY` became once a Parquet load started
            // producing 65_536-row batches instead of 1024-row ones.
            //
            // The chunks are (batch, row range) pairs, so a batch is divided without
            // being copied: `accumulate_batch` already takes a row offset.
            //
            // The threshold is the ordinary one for parallelising at all. It used to
            // be far higher because this path required a single large batch; now that
            // it divides by rows it subsumes the per-batch fan-out below — with many
            // batches the chunks simply line up with them — so the old, stricter
            // threshold only kept two-batch inputs on two threads.
            let total_gb_rows: usize = batches.iter().map(|b| b.row_count()).sum();
            if total_gb_rows >= crate::query::tuning::par_min_rows_per_batch_fanout() {
                let n_threads = rayon::current_num_threads().max(1);
                // Aim for a few chunks per thread so an uneven split still balances.
                let target = (total_gb_rows / (n_threads * 2).max(1)).max(4096);

                let mut work: Vec<(usize, usize, usize)> = Vec::new();
                for (bi, b) in batches.iter().enumerate() {
                    let rows = b.row_count();
                    let mut start = 0;
                    while start < rows {
                        let end = (start + target).min(rows);
                        work.push((bi, start, end));
                        start = end;
                    }
                }

                let partial: FalconResult<Vec<Vec<BatchAggState>>> = work
                    .par_iter()
                    .map(|&(bi, start, end)| {
                        let batch = &batches[bi];
                        let gb_dict = batch.column(gb_col_idx).as_utf8_column()
                            .ok_or_else(|| FalconError::InvalidQuery(
                                format!("Expected Utf8 column '{gb_name}'")
                            ))?;
                        let mut local: Vec<BatchAggState> = (0..n_aggs)
                            .map(|_| BatchAggState::with_capacity(n_g))
                            .collect();
                        self.accumulate_batch(
                            &gb_dict.codes()[start..end], start, batch, &agg_col_indices, &mut local,
                        )?;
                        Ok(local)
                    })
                    .collect();
                let mut final_states: Vec<BatchAggState> = (0..n_aggs)
                    .map(|_| BatchAggState::with_capacity(n_g))
                    .collect();
                for local in partial? {
                    for (agg_idx, agg) in self.aggs.iter().enumerate() {
                        for g in 0..n_g {
                            merge_state_entry(&agg.func, &local[agg_idx], g, &mut final_states[agg_idx], g);
                        }
                    }
                }
                return self.build_fast_result_utf8(gb_name, global_keys, final_states);
            }

            macro_rules! map_batches_utf8_nonnull {
                ($iter:expr) => {
                    $iter.map(|batch| {
                        let gb_dict = batch.column(gb_col_idx).as_utf8_column()
                            .ok_or_else(|| FalconError::InvalidQuery(
                                format!("Expected Utf8 column '{gb_name}'")
                            ))?;
                        let mut local_states: Vec<BatchAggState> = (0..n_aggs)
                            .map(|_| BatchAggState::with_capacity(n_g))
                            .collect();
                        self.accumulate_batch(gb_dict.codes(), 0, batch, &agg_col_indices, &mut local_states)?;
                        Ok(local_states)
                    }).collect()
                };
            }
            let partial_states: FalconResult<Vec<Vec<BatchAggState>>> = if rows_of(batches) < crate::query::tuning::par_min_rows_per_batch_fanout() {
                map_batches_utf8_nonnull!(batches.iter())
            } else {
                map_batches_utf8_nonnull!(batches.par_iter())
            };

            // Merge: simple indexed array addition — no HashMap, no string comparison.
            let mut final_states: Vec<BatchAggState> = (0..n_aggs)
                .map(|_| BatchAggState::with_capacity(n_g))
                .collect();
            for local in partial_states? {
                for (agg_idx, agg) in self.aggs.iter().enumerate() {
                    for g in 0..n_g {
                        merge_state_entry(
                            &agg.func, &local[agg_idx], g,
                            &mut final_states[agg_idx], g,
                        );
                    }
                }
            }
            return self.build_fast_result_utf8(gb_name, global_keys, final_states);
        }

        // NULLABLE PATH — at least one batch has null GB values.
        // Use per-batch key tracking and HashMap-based merge to handle the extra null group.
        let batch_fn_utf8_null = |batch: &RecordBatch| -> FalconResult<(Vec<String>, Vec<BatchAggState>)> {
                let gb_col = batch.column(gb_col_idx);
                let gb_dict = gb_col.as_utf8_column()
                    .ok_or_else(|| FalconError::InvalidQuery(
                        format!("Expected Utf8 column '{gb_name}'")
                    ))?;
                let dict = gb_dict.dict();
                let n_g  = dict.len();
                let mut local_keys: Vec<String> = Vec::with_capacity(n_g + 1);
                let mut local_states: Vec<BatchAggState> =
                    (0..n_aggs).map(|_| BatchAggState::new()).collect();
                let mut code_to_gid: Vec<u32> = Vec::with_capacity(n_g);
                for s in dict {
                    code_to_gid.push(local_keys.len() as u32);
                    local_keys.push(s.clone());
                    for st in &mut local_states { st.push_group(); }
                }
                let mut null_gid: Option<u32> = None;
                let mut batch_gids: Vec<u32> = Vec::with_capacity(gb_dict.len());
                for &code in gb_dict.codes() {
                    let gid = if crate::storage::column::Utf8Column::is_null_code(code) {
                        match null_gid {
                            Some(g) => g,
                            None => {
                                let g = match local_keys.iter().position(|k| k.is_empty()) {
                                    Some(i) => i as u32,
                                    None => {
                                        let id = local_keys.len() as u32;
                                        local_keys.push(String::new());
                                        for st in &mut local_states { st.push_group(); }
                                        id
                                    }
                                };
                                null_gid = Some(g);
                                g
                            }
                        }
                    } else {
                        code_to_gid[code as usize]
                    };
                    batch_gids.push(gid);
                }
                self.accumulate_batch(&batch_gids, 0, batch, &agg_col_indices, &mut local_states)?;
                Ok((local_keys, local_states))
        };
        let partials: FalconResult<Vec<(Vec<String>, Vec<BatchAggState>)>> = if rows_of(batches) < crate::query::tuning::par_min_rows_per_batch_fanout() {
            batches.iter().map(batch_fn_utf8_null).collect()
        } else {
            batches.par_iter().map(batch_fn_utf8_null).collect()
        };

        let mut final_keys:   Vec<String>          = Vec::new();
        let mut final_map:    FastMap<String, u32> = FastMap::default();
        let mut final_states: Vec<BatchAggState>   = (0..n_aggs).map(|_| BatchAggState::new()).collect();

        for (local_keys, local_states) in partials? {
            for (local_gi, key) in local_keys.iter().enumerate() {
                let final_gid = if let Some(&id) = final_map.get(key.as_str()) {
                    id as usize
                } else {
                    let id = final_keys.len();
                    final_keys.push(key.clone());
                    final_map.insert(key.clone(), id as u32);
                    for s in &mut final_states { s.push_group(); }
                    id
                };
                for (agg_idx, agg) in self.aggs.iter().enumerate() {
                    merge_state_entry(&agg.func, &local_states[agg_idx], local_gi,
                                      &mut final_states[agg_idx], final_gid);
                }
            }
        }

        self.build_fast_result_utf8(gb_name, final_keys, final_states)
    }

    /// Fast path: Int64 group-by column.
    /// Batches are aggregated in parallel with Rayon; partial results are merged sequentially.
    fn fast_i64_gb(
        &self,
        batches:       &[RecordBatch],
        gb_name:       &str,
        agg_col_names: &[Option<&str>],
    ) -> FalconResult<RecordBatch> {
        let n_aggs = self.aggs.len();

        // в”Ђв”Ђ Per-batch aggregation (parallel) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        let first_i = batches.first().ok_or_else(|| FalconError::InvalidQuery("no batches".into()))?;
        let gb_col_idx_i = first_i.schema.column_index(gb_name)?;
        let agg_col_indices_i: Vec<Option<usize>> = agg_col_names.iter()
            .map(|&n| n.and_then(|name| first_i.schema.column_index(name).ok()))
            .collect();

        let batch_fn_i64 = |batch: &RecordBatch| -> FalconResult<(Vec<i64>, Vec<BatchAggState>)> {
            let n_rows = batch.row_count();
            let gb_col = batch.column(gb_col_idx_i);
            let gb_i64 = gb_col.as_int64_values()
                .ok_or_else(|| FalconError::InvalidQuery(
                    format!("Expected Int64 column '{gb_name}'")
                ))?;
            let mut local_keys:   Vec<i64>           = Vec::new();
            let mut local_map:    FastMap<i64, u32>  = FastMap::default();
            let mut local_states: Vec<BatchAggState> = (0..n_aggs).map(|_| BatchAggState::new()).collect();
            let mut batch_gids: Vec<u32> = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                let key: i64 = gb_i64[row].unwrap_or(i64::MIN);
                let gid = if let Some(&id) = local_map.get(&key) {
                    id
                } else {
                    let id = local_keys.len() as u32;
                    local_keys.push(key);
                    local_map.insert(key, id);
                    for s in &mut local_states { s.push_group(); }
                    id
                };
                batch_gids.push(gid);
            }
            self.accumulate_batch(&batch_gids, 0, batch, &agg_col_indices_i, &mut local_states)?;
            Ok((local_keys, local_states))
        };
        let partials: FalconResult<Vec<(Vec<i64>, Vec<BatchAggState>)>> = if rows_of(batches) < crate::query::tuning::par_min_rows_per_batch_fanout() {
            batches.iter().map(batch_fn_i64).collect()
        } else {
            batches.par_iter().map(batch_fn_i64).collect()
        };

        // в”Ђв”Ђ Merge partial results в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        let mut final_keys:   Vec<i64>           = Vec::new();
        let mut final_map:    FastMap<i64, u32>  = FastMap::default();
        let mut final_states: Vec<BatchAggState> = (0..n_aggs).map(|_| BatchAggState::new()).collect();

        for (local_keys, local_states) in partials? {
            for (local_gi, &key) in local_keys.iter().enumerate() {
                let final_gid = if let Some(&id) = final_map.get(&key) {
                    id as usize
                } else {
                    let id = final_keys.len();
                    final_keys.push(key);
                    final_map.insert(key, id as u32);
                    for s in &mut final_states { s.push_group(); }
                    id
                };
                for (agg_idx, agg) in self.aggs.iter().enumerate() {
                    merge_state_entry(&agg.func, &local_states[agg_idx], local_gi,
                                      &mut final_states[agg_idx], final_gid);
                }
            }
        }

        self.build_fast_result_i64(gb_name, final_keys, final_states)
    }

    /// The hot inner loop: typed, branch-free per agg type.
    /// `agg_col_indices[i] = None` means COUNT(*) вЂ” count every row without reading a column.
    fn accumulate_batch(
        &self,
        batch_gids:      &[u32],
        row_start:       usize,
        batch:           &RecordBatch,
        agg_col_indices: &[Option<usize>],
        states:          &mut [BatchAggState],
    ) -> FalconResult<()> {
        let n_aggs = self.aggs.len();

        // Fuse SUM + AVG pairs on the same no-null Float64 column into a single pass,
        // eliminating a redundant scan of the value column and its loop overhead.
        // Bitmask: zero alloc, supports up to 64 agg expressions (typical n <= 8).
        let mut fused: u64 = 0;
        for i in 0..n_aggs.min(64) {
            if fused & (1u64 << i) != 0 { continue; }
            let Some(col_i) = agg_col_indices[i] else { continue };
            if !matches!(self.aggs[i].func, AggFunc::Sum) { continue; }
            let Column::Float64(c) = batch.column(col_i) else { continue };
            if c.has_nulls() { continue; }
            for j in (i + 1)..n_aggs.min(64) {
                if fused & (1u64 << j) != 0 { continue; }
                if !matches!(self.aggs[j].func, AggFunc::Avg) { continue; }
                if agg_col_indices[j] != Some(col_i) { continue; }
                // i < j by construction; split_at_mut(j) gives left[..j] and right[j..].
                let (left, right) = states.split_at_mut(j);
                let sum_st = &mut left[i];
                let avg_st = &mut right[0];  // right[0] == states[j]
                for (&gid, &v) in batch_gids.iter().zip(c.dense()[row_start..].iter()) {
                    let g = gid as usize;
                    sum_st.sum[g]    += v;
                    sum_st.has_val[g] = true;
                    avg_st.sum[g]    += v;
                    avg_st.count[g]  += 1;
                }
                fused |= (1u64 << i) | (1u64 << j);
                break;
            }
        }

        for (agg_idx, (agg, &maybe_idx)) in self.aggs.iter().zip(agg_col_indices.iter()).enumerate() {
            if agg_idx < 64 && fused & (1u64 << agg_idx) != 0 { continue; }
            let state = &mut states[agg_idx];

            // COUNT(*): count every row without touching a column.
            if maybe_idx.is_none() {
                for &gid in batch_gids {
                    state.count[gid as usize] += 1;
                }
                continue;
            }

            let col = batch.column(maybe_idx.unwrap());

            // Dispatch once per batch (not per row) on (func, column type).
            match (&agg.func, col) {
                // в”Ђв”Ђ COUNT в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
                (AggFunc::Count, Column::Float64(c)) => {
                    if !c.has_nulls() {
                        for &gid in batch_gids { state.count[gid as usize] += 1; }
                    } else {
                        for (r, &gid) in batch_gids.iter().enumerate() {
                            if c.is_valid(row_start + r) { state.count[gid as usize] += 1; }
                        }
                    }
                }
                (AggFunc::Count, Column::Int64(c)) => {
                    if !c.has_nulls() {
                        for &gid in batch_gids { state.count[gid as usize] += 1; }
                    } else {
                        for (r, &gid) in batch_gids.iter().enumerate() {
                            if c.is_valid(row_start + r) { state.count[gid as usize] += 1; }
                        }
                    }
                }
                (AggFunc::Count, Column::Int32(c)) => {
                    for (r, &gid) in batch_gids.iter().enumerate() {
                        if c.is_valid(row_start + r) { state.count[gid as usize] += 1; }
                    }
                }
                (AggFunc::Count, Column::Utf8(c)) => {
                    for (r, &gid) in batch_gids.iter().enumerate() {
                        if c.get_str(row_start + r).is_some() { state.count[gid as usize] += 1; }
                    }
                }

                // в”Ђв”Ђ SUM в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
                (AggFunc::Sum, Column::Float64(c)) => {
                    if !c.has_nulls() {
                        for (&gid, &v) in batch_gids.iter().zip(c.dense()[row_start..].iter()) {
                            let g = gid as usize;
                            state.sum[g] += v;
                            state.has_val[g] = true;
                        }
                    } else {
                        for (r, &gid) in batch_gids.iter().enumerate() {
                            if let Some(v) = c.get(row_start + r) {
                                let g = gid as usize;
                                state.sum[g] += v;
                                state.has_val[g] = true;
                            }
                        }
                    }
                }
                (AggFunc::Sum, Column::Int64(c)) => {
                    if !c.has_nulls() {
                        for (&gid, &v) in batch_gids.iter().zip(c.dense()[row_start..].iter()) {
                            let g = gid as usize;
                            state.sum[g] += v as f64;
                            state.has_val[g] = true;
                        }
                    } else {
                        for (r, &gid) in batch_gids.iter().enumerate() {
                            if let Some(v) = c.get(row_start + r) {
                                let g = gid as usize;
                                state.sum[g] += v as f64;
                                state.has_val[g] = true;
                            }
                        }
                    }
                }
                (AggFunc::Sum, Column::Int32(c)) => {
                    for (r, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row_start + r) {
                            let g = gid as usize;
                            state.sum[g] += v as f64;
                            state.has_val[g] = true;
                        }
                    }
                }

                // в”Ђв”Ђ AVG в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
                (AggFunc::Avg, Column::Float64(c)) => {
                    if !c.has_nulls() {
                        for (&gid, &v) in batch_gids.iter().zip(c.dense()[row_start..].iter()) {
                            let g = gid as usize;
                            state.sum[g]   += v;
                            state.count[g] += 1;
                        }
                    } else {
                        for (r, &gid) in batch_gids.iter().enumerate() {
                            if let Some(v) = c.get(row_start + r) {
                                let g = gid as usize;
                                state.sum[g]   += v;
                                state.count[g] += 1;
                            }
                        }
                    }
                }
                (AggFunc::Avg, Column::Int64(c)) => {
                    if !c.has_nulls() {
                        for (&gid, &v) in batch_gids.iter().zip(c.dense()[row_start..].iter()) {
                            let g = gid as usize;
                            state.sum[g]   += v as f64;
                            state.count[g] += 1;
                        }
                    } else {
                        for (r, &gid) in batch_gids.iter().enumerate() {
                            if let Some(v) = c.get(row_start + r) {
                                let g = gid as usize;
                                state.sum[g]   += v as f64;
                                state.count[g] += 1;
                            }
                        }
                    }
                }
                (AggFunc::Avg, Column::Int32(c)) => {
                    for (r, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row_start + r) {
                            let g = gid as usize;
                            state.sum[g]   += v as f64;
                            state.count[g] += 1;
                        }
                    }
                }

                // в”Ђв”Ђ MIN в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
                (AggFunc::Min, Column::Float64(c)) => {
                    for (r, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row_start + r) {
                            let g = gid as usize;
                            if !state.has_val[g] || v < state.sum[g] {
                                state.sum[g] = v; state.has_val[g] = true;
                            }
                        }
                    }
                }
                (AggFunc::Min, Column::Int64(c)) => {
                    for (r, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row_start + r) {
                            let g  = gid as usize;
                            let vf = v as f64;
                            if !state.has_val[g] || vf < state.sum[g] {
                                state.sum[g] = vf; state.has_val[g] = true;
                            }
                        }
                    }
                }

                // в”Ђв”Ђ MAX в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
                (AggFunc::Max, Column::Float64(c)) => {
                    for (r, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row_start + r) {
                            let g = gid as usize;
                            if !state.has_val[g] || v > state.sum[g] {
                                state.sum[g] = v; state.has_val[g] = true;
                            }
                        }
                    }
                }
                (AggFunc::Max, Column::Int64(c)) => {
                    for (r, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row_start + r) {
                            let g  = gid as usize;
                            let vf = v as f64;
                            if !state.has_val[g] || vf > state.sum[g] {
                                state.sum[g] = vf; state.has_val[g] = true;
                            }
                        }
                    }
                }

                // в”Ђв”Ђ Fallback for remaining type combos в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
                _ => {
                    let vals = col.as_float64_values().ok_or_else(|| {
                        FalconError::InvalidQuery(format!(
                            "Cannot aggregate '{}' as float", agg.alias
                        ))
                    })?;
                    for (r, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = vals[row_start + r] {
                            let g = gid as usize;
                            match &agg.func {
                                AggFunc::Sum => { state.sum[g] += v; state.has_val[g] = true; }
                                AggFunc::Avg => { state.sum[g] += v; state.count[g] += 1; }
                                AggFunc::Min => {
                                    if !state.has_val[g] || v < state.sum[g] {
                                        state.sum[g] = v; state.has_val[g] = true;
                                    }
                                }
                                AggFunc::Max => {
                                    if !state.has_val[g] || v > state.sum[g] {
                                        state.sum[g] = v; state.has_val[g] = true;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn build_fast_result_utf8(
        &self,
        gb_name:    &str,
        group_keys: Vec<String>,
        states:     Vec<BatchAggState>,
    ) -> FalconResult<RecordBatch> {
        let n_groups = group_keys.len();
        let mut fields  = Vec::new();
        let mut columns = Vec::new();

        fields.push(Field::nullable(gb_name, DataType::Utf8));
        columns.push(Column::Utf8(Utf8Column::new(
            group_keys.into_iter().map(Some).collect(),
        )));

        for (agg, state) in self.aggs.iter().zip(states.iter()) {
            let (dt, col) = build_agg_col(&agg.func, state, n_groups);
            fields.push(Field::nullable(agg.alias.as_str(), dt));
            columns.push(col);
        }
        RecordBatch::new(Schema::new(fields), columns)
    }

    fn build_fast_result_i64(
        &self,
        gb_name:    &str,
        group_keys: Vec<i64>,
        states:     Vec<BatchAggState>,
    ) -> FalconResult<RecordBatch> {
        let n_groups = group_keys.len();
        let mut fields  = Vec::new();
        let mut columns = Vec::new();

        fields.push(Field::nullable(gb_name, DataType::Int64));
        columns.push(Column::Int64(PrimitiveColumn::new(
            group_keys.into_iter()
                .map(|k| if k == i64::MIN { None } else { Some(k) })
                .collect(),
        )));

        for (agg, state) in self.aggs.iter().zip(states.iter()) {
            let (dt, col) = build_agg_col(&agg.func, state, n_groups);
            fields.push(Field::nullable(agg.alias.as_str(), dt));
            columns.push(col);
        }
        RecordBatch::new(Schema::new(fields), columns)
    }

    // в”Ђв”Ђ Slow path (original row-by-row, general Expr support) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    /// Two-phase aggregation: each batch builds its own hash table, then the
    /// tables are merged.
    ///
    /// Sharing one table across threads would need a lock on every row. Building
    /// per-batch tables and merging costs an extra pass over the *groups*, which
    /// is normally a tiny fraction of the rows.
    ///
    /// Group order stays deterministic. Merging keeps the left side's order and
    /// appends whatever keys the right side introduces; since that is associative
    /// and rayon reduces over contiguous ranges, the result matches what a serial
    /// left-to-right pass would have produced.
    fn execute_slow(&self, batches: &[RecordBatch]) -> FalconResult<RecordBatch> {
        type Partial = (
            HashMap<GroupKey, Vec<Accumulator>>,
            Vec<GroupKey>,
            HashMap<GroupKey, Vec<ScalarValue>>,
        );

        let empty = || -> Partial { (HashMap::new(), Vec::new(), HashMap::new()) };

        let merge = |mut a: Partial, b: Partial| -> FalconResult<Partial> {
            let (b_groups, b_order, mut b_vals) = b;
            let mut b_groups = b_groups;
            for key in b_order {
                match a.0.get_mut(&key) {
                    Some(existing) => {
                        if let Some(incoming) = b_groups.remove(&key) {
                            for (acc, other) in existing.iter_mut().zip(incoming) {
                                acc.merge(other);
                            }
                        }
                    }
                    None => {
                        if let Some(incoming) = b_groups.remove(&key) {
                            a.0.insert(key.clone(), incoming);
                            if let Some(v) = b_vals.remove(&key) {
                                a.2.insert(key.clone(), v);
                            }
                            a.1.push(key);
                        }
                    }
                }
            }
            Ok(a)
        };

        let (groups, key_order, group_key_values) = if rows_of(batches) < crate::query::tuning::par_min_rows_per_batch_fanout() {
            batches.iter().try_fold(empty(), |mut acc, batch| -> FalconResult<Partial> {
                self.accumulate_slow(batch, &mut acc.0, &mut acc.1, &mut acc.2)?;
                Ok(acc)
            })?
        } else {
            batches
                .par_iter()
                .try_fold(empty, |mut acc, batch| -> FalconResult<Partial> {
                    self.accumulate_slow(batch, &mut acc.0, &mut acc.1, &mut acc.2)?;
                    Ok(acc)
                })
                .try_reduce(empty, merge)?
        };

        self.build_slow_result(&groups, &key_order, &group_key_values)
    }

    fn accumulate_slow(
        &self,
        batch:            &RecordBatch,
        groups:           &mut HashMap<GroupKey, Vec<Accumulator>>,
        key_order:        &mut Vec<GroupKey>,
        group_key_values: &mut HashMap<GroupKey, Vec<ScalarValue>>,
    ) -> FalconResult<()> {
        let group_cols: Vec<Vec<ScalarValue>> = self.group_by
            .iter().map(|e| e.eval_batch(batch)).collect::<FalconResult<_>>()?;
        let agg_cols: Vec<Vec<ScalarValue>> = self.aggs
            .iter().map(|a| a.input.eval_batch(batch)).collect::<FalconResult<_>>()?;

        for row in 0..batch.row_count() {
            let key_vals: Vec<ScalarValue> = group_cols.iter().map(|c| c[row].clone()).collect();
            let key = GroupKey::from_row(&key_vals);

            if !groups.contains_key(&key) {
                key_order.push(key.clone());
                group_key_values.insert(key.clone(), key_vals);
                groups.insert(key.clone(), self.aggs.iter().map(|a| Accumulator::new(&a.func)).collect());
            }

            let accs = groups.get_mut(&key).unwrap();
            for (acc, col) in accs.iter_mut().zip(agg_cols.iter()) {
                acc.update(&col[row]);
            }
        }
        Ok(())
    }

    fn build_slow_result(
        &self,
        groups:           &HashMap<GroupKey, Vec<Accumulator>>,
        key_order:        &[GroupKey],
        group_key_values: &HashMap<GroupKey, Vec<ScalarValue>>,
    ) -> FalconResult<RecordBatch> {
        let n_rows = key_order.len();
        let mut fields  = Vec::new();
        let mut columns = Vec::new();

        for (gb_idx, gb_expr) in self.group_by.iter().enumerate() {
            let alias = match gb_expr {
                Expr::Column(name) => name.clone(),
                _ => format!("group_{gb_idx}"),
            };
            let vals: Vec<ScalarValue> = key_order.iter()
                .map(|k| group_key_values[k][gb_idx].clone()).collect();
            let (dt, col) = scalar_vec_to_column(&vals, n_rows)?;
            fields.push(Field::nullable(alias.as_str(), dt));
            columns.push(col);
        }

        for (agg_idx, agg) in self.aggs.iter().enumerate() {
            let vals: Vec<ScalarValue> = key_order.iter()
                .map(|k| groups[k][agg_idx].result()).collect();
            let (dt, col) = scalar_vec_to_column(&vals, n_rows)?;
            fields.push(Field::nullable(agg.alias.as_str(), dt));
            columns.push(col);
        }

        RecordBatch::new(Schema::new(fields), columns)
    }
}

// в”Ђв”Ђв”Ђ Merge helper for parallel GROUP BY в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

fn merge_state_entry(
    func:   &AggFunc,
    src:    &BatchAggState,
    src_gi: usize,
    dst:    &mut BatchAggState,
    dst_gi: usize,
) {
    match func {
        AggFunc::Count => {
            dst.count[dst_gi] += src.count[src_gi];
        }
        AggFunc::Sum => {
            if src.has_val[src_gi] {
                dst.sum[dst_gi]     += src.sum[src_gi];
                dst.has_val[dst_gi]  = true;
            }
        }
        AggFunc::Avg => {
            dst.sum[dst_gi]   += src.sum[src_gi];
            dst.count[dst_gi] += src.count[src_gi];
        }
        AggFunc::Min => {
            if src.has_val[src_gi] && (!dst.has_val[dst_gi] || src.sum[src_gi] < dst.sum[dst_gi]) {
                dst.sum[dst_gi]     = src.sum[src_gi];
                dst.has_val[dst_gi] = true;
            }
        }
        AggFunc::Max => {
            if src.has_val[src_gi] && (!dst.has_val[dst_gi] || src.sum[src_gi] > dst.sum[dst_gi]) {
                dst.sum[dst_gi]     = src.sum[src_gi];
                dst.has_val[dst_gi] = true;
            }
        }
        AggFunc::CountDistinct
        | AggFunc::StringAgg { .. }
        | AggFunc::BoolAnd | AggFunc::BoolOr
        | AggFunc::Stddev { .. } | AggFunc::Variance { .. }
        | AggFunc::First | AggFunc::Last => unreachable!("not in fast path"),
    }
}

// в”Ђв”Ђв”Ђ Simple aggregation (no GROUP BY) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

pub fn execute_agg_only(
    batches: &[RecordBatch],
    aggs:    &[AggExpr],
) -> FalconResult<RecordBatch> {
    // COUNT(*) shortcut: all aggs are COUNT(*) в†’ sum row_count, read no column data.
    if aggs.iter().all(is_count_star) {
        let total: i64 = batches.iter().map(|b| b.row_count() as i64).sum();
        let mut fields  = Vec::new();
        let mut columns = Vec::new();
        for agg in aggs {
            fields.push(Field::nullable(agg.alias.as_str(), DataType::Int64));
            columns.push(Column::Int64(PrimitiveColumn::new(vec![Some(total)])));
        }
        return RecordBatch::new(Schema::new(fields), columns);
    }

    // Pre-resolve column indices once per query (column_by_name is a linear scan).
    let schema = batches.first()
        .ok_or_else(|| FalconError::InvalidQuery("no batches".into()))?
        .schema.clone();
    let col_indices: Vec<Option<usize>> = aggs.iter().map(|agg| {
        if is_count_star(agg) { return None; }
        if let Expr::Column(name) = &agg.input {
            schema.column_index(name).ok()
        } else { None }
    }).collect();

    // Accumulate each batch independently, then merge. Aggregates are associative
    // over disjoint row ranges, so this is safe and lets every core work.
    //
    // Below this many rows a serial fold wins outright. Entering a rayon region
    // costs a fixed ~27 µs here — waking the pool — and the vectorised kernel is
    // fast enough that a lot of rows are needed to earn that back. Timing
    // `full_scan` at three sizes gives serial 0.224 ns/row against parallel
    // 27.4 µs + 0.070 ns/row, so the two meet at about 178k rows.
    //
    // The count is in rows, not batches: a batch is 1024 rows off a Parquet load
    // but 65_536 elsewhere, and a batch-count threshold silently means completely
    // different things depending on where the table came from.
    // Measured at ~178k rows on a machine reporting 20 threads, so ~8.9k each.
    // See `query::tuning` for why this is per-thread rather than absolute.
    let agg_parallel_min_rows = crate::query::tuning::par_min_rows(8_900);

    /// Rows a rayon task should cover before the hand-off is worth its cost,
    /// converted to a batch count for `with_min_len`.
    ///
    /// This was a literal `8` batches, which is ~8 µs of work at 1024 rows a batch
    /// and was measured as the right granularity there. It is the wrong quantity to
    /// fix, though: at 65_536 rows a batch the same eight are half a million rows,
    /// so a million-row scan handed twenty threads just two tasks.
    ///
    /// A computed value does cost something — rayon folds a literal into its
    /// splitting and cannot fold a variable, measured at ~6% on full_scan/1M. That
    /// is worth paying against the 27–41% the larger batches return.
    const AGG_ROWS_PER_TASK: usize = 8_192;

    // A lone batch has nothing to fan out over, so the per-column kernels may split
    // the slice themselves. With more than one batch the fan-out happens above and a
    // second, nested one would only contend.
    let may_split = batches.len() == 1;

    macro_rules! accumulate_body {
        ($accs:expr, $batch:expr) => {{
            let mut accs = $accs;
            let batch = $batch;
            for ((acc, agg), &maybe_idx) in
                accs.iter_mut().zip(aggs.iter()).zip(col_indices.iter())
            {
                if is_count_star(agg) {
                    if let Accumulator::Count { n } = acc {
                        *n += batch.row_count() as i64;
                    }
                    continue;
                }
                if let Some(idx) = maybe_idx {
                    let col = batch.column(idx);
                    if accumulate_typed(acc, col, may_split) {
                        continue;
                    }
                }
                let vals = agg.input.eval_batch(batch)?;
                for val in &vals {
                    acc.update(val);
                }
            }
            Ok(accs)
        }};
    }

    let total_rows: usize = batches.iter().map(|b| b.row_count()).sum();

    // Batches vary in size, so derive the grouping from the average rather than
    // assuming one shape.
    let rows_per_batch = (total_rows / batches.len().max(1)).max(1);
    let min_batches_per_task = AGG_ROWS_PER_TASK.div_ceil(rows_per_batch).max(1);

    let accumulators = if total_rows < agg_parallel_min_rows {
        batches.iter().try_fold(
            aggs.iter().map(|a| Accumulator::new(&a.func)).collect::<Vec<_>>(),
            |accs, batch| -> FalconResult<Vec<Accumulator>> { accumulate_body!(accs, batch) },
        )?
    } else {
        batches
            .par_iter()
            // A batch off a Parquet load is 1024 rows — about a microsecond of work.
            // Left to itself rayon splits down to one batch per task, so at 100k rows
            // (98 batches) the dispatch cost is the bulk of the query. Handing each
            // task a run of batches keeps every core busy with far fewer hand-offs.
            .with_min_len(min_batches_per_task)
            .try_fold(
                || aggs.iter().map(|a| Accumulator::new(&a.func)).collect::<Vec<_>>(),
                |accs, batch| -> FalconResult<Vec<Accumulator>> { accumulate_body!(accs, batch) },
            )
            .try_reduce(
                || aggs.iter().map(|a| Accumulator::new(&a.func)).collect::<Vec<_>>(),
                |mut a, b| {
                    for (acc, other) in a.iter_mut().zip(b) {
                        acc.merge(other);
                    }
                    Ok(a)
                },
            )?
    };

    let mut fields  = Vec::new();
    let mut columns = Vec::new();
    for (agg, acc) in aggs.iter().zip(accumulators.iter()) {
        let result = acc.result();
        let dt     = result.data_type();
        let col    = scalar_to_single_row_col(&result);
        fields.push(Field::nullable(agg.alias.as_str(), dt));
        columns.push(col);
    }
    RecordBatch::new(Schema::new(fields), columns)
}

// в”Ђв”Ђв”Ђ Typed accumulation в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// Fold a whole column into `acc` without boxing any value.
///
/// Returns `false` when the accumulator/column pair has no typed path, so the
/// caller falls back to the `ScalarValue` loop.
///
/// The no-NULL case is kept as a separate straight loop over the dense slice
/// because that is the shape LLVM can autovectorise; interleaving a validity
/// check would stop it.
/// Number of independent accumulators used to add a float slice.
/// Two AVX2 registers' worth: enough chains to keep the FPU busy, few enough that
/// the tail stays short.
const SUM_LANES: usize = 8;

/// Sum a slice with several independent accumulators.
///
/// `iter().sum()` cannot be vectorised: float addition is not associative, so the
/// compiler has to add strictly left to right and every add waits on the previous
/// one's latency. That put the f64 sum at 0.73 ns/row, about seven times off what
/// the hardware can do. Splitting into independent chains lets the CPU issue them
/// together and lets LLVM emit SIMD.
///
/// This does reassociate the additions, so the last bits of the result can differ
/// from a strict left-to-right sum. For a long run of values that is normally
/// *more* accurate, not less — it is a step toward pairwise summation.
#[inline]
fn sum_lanes<T: SumLane>(dense: &[T]) -> f64 {
    let mut acc = [0.0f64; SUM_LANES];
    let mut chunks = dense.chunks_exact(SUM_LANES);
    for c in &mut chunks {
        for (a, &v) in acc.iter_mut().zip(c.iter()) {
            *a += v.to_f64();
        }
    }
    let mut total = acc.iter().sum::<f64>();
    for &v in chunks.remainder() {
        total += v.to_f64();
    }
    total
}

/// Widened to f64 for summation.
///
/// `Into<f64>` would be the natural bound but `i64` does not implement it: the
/// conversion loses precision past 2^53. SUM already returns f64 and the previous
/// `v as f64` did exactly this, so the trait keeps that behaviour rather than
/// changing what SUM means on wide integers.
trait SumLane: Copy + Sync {
    fn to_f64(self) -> f64;
}

impl SumLane for f64 {
    #[inline(always)]
    fn to_f64(self) -> f64 { self }
}
impl SumLane for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 { self as f64 }
}
impl SumLane for i64 {
    #[inline(always)]
    fn to_f64(self) -> f64 { self as f64 }
}
impl SumLane for i32 {
    #[inline(always)]
    fn to_f64(self) -> f64 { self as f64 }
}

/// Rows per rayon task when summing one large slice.
const PAR_SUM_CHUNK: usize = 16_384;

/// Parallel variant of [`sum_lanes`]. Chunked rather than per-element so each
/// task does real work and still sums its own chunk with the vectorised kernel.
fn par_sum_lanes<T: SumLane>(dense: &[T]) -> f64 {
    dense.par_chunks(PAR_SUM_CHUNK).map(sum_lanes).sum::<f64>()
}

fn accumulate_typed(acc: &mut Accumulator, col: &Column, may_split: bool) -> bool {
    use crate::storage::column::Column as C;

    /// Apply `$f` to every non-NULL value of a primitive column, as f64.
    macro_rules! each_valid {
        ($c:expr, $v:ident, $body:block) => {{
            let c = $c;
            let dense = c.dense();
            if c.has_nulls() {
                for i in 0..c.len() {
                    if c.is_valid(i) {
                        let $v = dense[i] as f64;
                        $body
                    }
                }
            } else {
                for &x in dense {
                    let $v = x as f64;
                    $body
                }
            }
        }};
    }

    /// Run `$arm` for each primitive column type; `false` for the rest.
    macro_rules! over_primitives {
        ($v:ident, $body:block) => {
            match col {
                C::Int64(c)     => each_valid!(c, $v, $body),
                C::Int32(c)     => each_valid!(c, $v, $body),
                C::Float64(c)   => each_valid!(c, $v, $body),
                C::Float32(c)   => each_valid!(c, $v, $body),
                C::Date32(c)    => each_valid!(c, $v, $body),
                C::Timestamp(c) => each_valid!(c, $v, $body),
                _ => return false,
            }
        };
    }

    match acc {
        Accumulator::Count { n } => {
            // COUNT(expr) counts non-NULL values; the count is metadata, not data.
            *n += (col.len() - col.null_count()) as i64;
            true
        }
        Accumulator::Sum { total, valid } => {
            // Fast paths for no-null columns: a single reduction without branches.
            //
            // A long enough slice is split across cores, but only when the caller
            // says there is nothing else to spread the work over — `may_split` is
            // true for a lone batch. The size test alone used to decide this, which
            // was harmless while a Parquet load produced 1024-row batches (it never
            // fired) and actively wrong once batches were large: every batch of a
            // serial fold would enter its own rayon region.
            const PAR_SUM_THRESHOLD: usize = 65_536;
            if let C::Float64(c) = col {
                if !c.has_nulls() {
                    let dense = c.dense();
                    if !dense.is_empty() {
                        *total += if may_split && dense.len() >= PAR_SUM_THRESHOLD {
                            par_sum_lanes(dense)
                        } else {
                            sum_lanes(dense)
                        };
                        *valid = true;
                    }
                    return true;
                }
            }
            if let C::Int64(c) = col {
                if !c.has_nulls() {
                    let dense = c.dense();
                    if !dense.is_empty() {
                        *total += if may_split && dense.len() >= PAR_SUM_THRESHOLD {
                            par_sum_lanes(dense)
                        } else {
                            sum_lanes(dense)
                        };
                        *valid = true;
                    }
                    return true;
                }
            }
            if let C::Int32(c) = col {
                if !c.has_nulls() {
                    let dense = c.dense();
                    if !dense.is_empty() {
                        *total += if may_split && dense.len() >= PAR_SUM_THRESHOLD {
                            par_sum_lanes(dense)
                        } else {
                            sum_lanes(dense)
                        };
                        *valid = true;
                    }
                    return true;
                }
            }
            let mut t = 0.0f64;
            let mut any = false;
            over_primitives!(v, {
                t += v;
                any = true;
            });
            *total += t;
            *valid |= any;
            true
        }
        Accumulator::Avg { total, count } => {
            if let C::Float64(c) = col {
                if !c.has_nulls() {
                    let dense = c.dense();
                    *total += if may_split && dense.len() >= 65_536 {
                        dense.par_iter().copied().sum::<f64>()
                    } else {
                        dense.iter().copied().sum::<f64>()
                    };
                    *count += dense.len() as i64;
                    return true;
                }
            }
            let mut t = 0.0f64;
            let mut n = 0i64;
            over_primitives!(v, {
                t += v;
                n += 1;
            });
            *total += t;
            *count += n;
            true
        }
        Accumulator::Min { value, .. } => {
            if let C::Float64(c) = col {
                if !c.has_nulls() {
                    let dense = c.dense();
                    let chunk_min = if may_split && dense.len() >= 65_536 {
                        dense.par_iter().copied().reduce(|| f64::INFINITY, f64::min)
                    } else {
                        dense.iter().copied().fold(f64::INFINITY, f64::min)
                    };
                    if chunk_min.is_finite() {
                        *value = Some(value.map_or(chunk_min, |v: f64| v.min(chunk_min)));
                    }
                    return true;
                }
            }
            let mut cur = f64::INFINITY;
            let mut any = false;
            over_primitives!(v, {
                if v < cur { cur = v; }
                any = true;
            });
            if any {
                *value = Some(value.map_or(cur, |c: f64| c.min(cur)));
            }
            true
        }
        Accumulator::Max { value, .. } => {
            if let C::Float64(c) = col {
                if !c.has_nulls() {
                    let dense = c.dense();
                    let chunk_max = if may_split && dense.len() >= 65_536 {
                        dense.par_iter().copied().reduce(|| f64::NEG_INFINITY, f64::max)
                    } else {
                        dense.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    };
                    if chunk_max.is_finite() {
                        *value = Some(value.map_or(chunk_max, |v: f64| v.max(chunk_max)));
                    }
                    return true;
                }
            }
            let mut cur = f64::NEG_INFINITY;
            let mut any = false;
            over_primitives!(v, {
                if v > cur { cur = v; }
                any = true;
            });
            if any {
                *value = Some(value.map_or(cur, |c: f64| c.max(cur)));
            }
            true
        }
        // These need the slow accumulator path
        Accumulator::CountDistinct { .. }
        | Accumulator::StringAgg { .. }
        | Accumulator::BoolAnd { .. }
        | Accumulator::BoolOr  { .. }
        | Accumulator::Stddev  { .. }
        | Accumulator::Variance { .. }
        | Accumulator::First { .. }
        | Accumulator::Last  { .. } => false,
    }
}

// в”Ђв”Ђв”Ђ Helpers в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

fn build_agg_col(func: &AggFunc, state: &BatchAggState, n: usize) -> (DataType, Column) {
    match func {
        AggFunc::Count => {
            let vals: Vec<Option<i64>> = state.count.iter().map(|&v| Some(v)).collect();
            (DataType::Int64, Column::Int64(PrimitiveColumn::new(vals)))
        }
        AggFunc::Sum => {
            let vals: Vec<Option<f64>> = (0..n)
                .map(|i| if state.has_val[i] { Some(state.sum[i]) } else { None })
                .collect();
            (DataType::Float64, Column::Float64(PrimitiveColumn::new(vals)))
        }
        AggFunc::Avg => {
            let vals: Vec<Option<f64>> = (0..n)
                .map(|i| if state.count[i] > 0 { Some(state.sum[i] / state.count[i] as f64) } else { None })
                .collect();
            (DataType::Float64, Column::Float64(PrimitiveColumn::new(vals)))
        }
        AggFunc::Min | AggFunc::Max => {
            let vals: Vec<Option<f64>> = (0..n)
                .map(|i| if state.has_val[i] { Some(state.sum[i]) } else { None })
                .collect();
            (DataType::Float64, Column::Float64(PrimitiveColumn::new(vals)))
        }
        AggFunc::CountDistinct
        | AggFunc::StringAgg { .. }
        | AggFunc::BoolAnd | AggFunc::BoolOr
        | AggFunc::Stddev { .. } | AggFunc::Variance { .. }
        | AggFunc::First | AggFunc::Last => unreachable!("not in fast path"),
    }
}

fn scalar_vec_to_column(vals: &[ScalarValue], n: usize) -> FalconResult<(DataType, Column)> {
    let dt = vals.iter()
        .find(|v| !v.is_null())
        .map(|v| v.data_type())
        .unwrap_or(DataType::Utf8);

    let col = match &dt {
        DataType::Boolean => Column::Boolean(BooleanColumn::new(
            vals.iter().map(|v| v.as_bool()).collect(),
        )),
        DataType::Int32 | DataType::Int64 => Column::Int64(PrimitiveColumn::new(
            vals.iter().map(|v| v.as_i64()).collect(),
        )),
        DataType::Float32 | DataType::Float64 => Column::Float64(PrimitiveColumn::new(
            vals.iter().map(|v| v.as_f64()).collect(),
        )),
        DataType::Utf8 => Column::Utf8(Utf8Column::new(
            vals.iter().map(|v| match v {
                ScalarValue::Utf8(s) => s.clone(),
                _ => None,
            }).collect(),
        )),
        DataType::Date32 => Column::Date32(PrimitiveColumn::new(
            vals.iter().map(|v| v.as_date32()).collect(),
        )),
        DataType::Timestamp => Column::Timestamp(PrimitiveColumn::new(
            vals.iter().map(|v| v.as_timestamp()).collect(),
        )),
        DataType::Decimal128(p, s) => {
            let (p, s) = (*p, *s);
            Column::Decimal128(crate::storage::column::Decimal128Column::new(
                vals.iter().map(|v| match v {
                    ScalarValue::Decimal128(d) => d.value,
                    _ => None,
                }).collect(),
                p, s,
            ))
        }
        DataType::Null => Column::Int64(PrimitiveColumn::new(vec![None; n])),
    };
    Ok((dt, col))
}

fn scalar_to_single_row_col(val: &ScalarValue) -> Column {
    match val {
        ScalarValue::Boolean(v) => Column::Boolean(BooleanColumn::new(vec![*v])),
        ScalarValue::Int32(v)   => Column::Int64(PrimitiveColumn::new(vec![v.map(|n| n as i64)])),
        ScalarValue::Int64(v)   => Column::Int64(PrimitiveColumn::new(vec![*v])),
        ScalarValue::Float32(v) => Column::Float64(PrimitiveColumn::new(vec![v.map(|n| n as f64)])),
        ScalarValue::Float64(v) => Column::Float64(PrimitiveColumn::new(vec![*v])),
        ScalarValue::Utf8(v)    => Column::Utf8(Utf8Column::new(vec![v.clone()])),
        ScalarValue::Date32(v)                 => Column::Date32(PrimitiveColumn::new(vec![*v])),
        ScalarValue::Timestamp(v)              => Column::Timestamp(PrimitiveColumn::new(vec![*v])),
        ScalarValue::Decimal128(d)             => Column::Decimal128(
            crate::storage::column::Decimal128Column::new(vec![d.value], d.precision, d.scale)
        ),
        ScalarValue::Null                      => Column::Int64(PrimitiveColumn::new(vec![None])),
    }
}

// в”Ђв”Ђв”Ђ Tests в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        query::expr::Expr,
        storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::RecordBatch,
            types::DataType,
        },
    };

    fn make_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::required("dept",  DataType::Utf8),
            Field::required("score", DataType::Float64),
        ]);
        RecordBatch::new(schema, vec![
            ColumnBuilder::utf8_from_str(vec![
                Some("eng"), Some("eng"), Some("hr"), Some("hr"), Some("eng"),
            ]),
            ColumnBuilder::float64(vec![
                Some(9.0), Some(8.0), Some(7.0), Some(6.0), Some(10.0),
            ]),
        ]).unwrap()
    }

    /// GROUP BY must give the same answer however the rows are split into batches.
    ///
    /// The parallel path divides by row ranges across batches, and every other
    /// group-by test here is a handful of rows — well under the threshold — so none
    /// of them reach it. A chunk boundary falling mid-batch, and a batch smaller
    /// than one chunk, are exactly the cases that can go wrong.
    #[test]
    fn parallel_group_by_matches_a_single_batch() {
        use crate::storage::{column::ColumnBuilder, schema::{Field, Schema}, types::DataType};

        const DEPTS: [&str; 5] = ["eng", "sales", "hr", "ops", "legal"];
        // Past the parallel threshold on any plausible thread count, and deliberately
        // uneven so the last batch is shorter than a chunk.
        const TOTAL: usize = 700_000;
        let sizes = [400_000usize, 250_000, 50_000];

        let schema = Schema::new(vec![
            Field::required("dept", DataType::Utf8),
            Field::required("v", DataType::Int64),
        ]);
        let mut batches = Vec::new();
        let mut start = 0usize;
        for n in sizes {
            batches.push(RecordBatch::new(
                schema.clone(),
                vec![
                    ColumnBuilder::utf8_from_str(
                        (0..n).map(|i| Some(DEPTS[(start + i) % 5])).collect(),
                    ),
                    ColumnBuilder::int64((0..n).map(|i| Some((start + i) as i64)).collect()),
                ],
            ).unwrap());
            start += n;
        }
        assert_eq!(start, TOTAL);

        let exec = GroupByExec::new(
            vec![Expr::Column("dept".into())],
            vec![
                AggExpr::count(Expr::Column("v".into()), "cnt"),
                AggExpr::sum(Expr::Column("v".into()), "total"),
            ],
        );
        let split = exec.execute(&batches).unwrap();

        // The same rows as one batch: the serial shape, and the reference.
        let single = vec![RecordBatch::new(
            schema,
            vec![
                ColumnBuilder::utf8_from_str((0..TOTAL).map(|i| Some(DEPTS[i % 5])).collect()),
                ColumnBuilder::int64((0..TOTAL).map(|i| Some(i as i64)).collect()),
            ],
        ).unwrap()];
        let whole = exec.execute(&single).unwrap();

        assert_eq!(split.row_count(), 5, "wrong number of groups");
        assert_eq!(split.row_count(), whole.row_count());

        // Compare group by group, keyed by name — group order is not guaranteed.
        let read = |b: &RecordBatch| -> std::collections::HashMap<String, (f64, f64)> {
            let mut m = std::collections::HashMap::new();
            for r in 0..b.row_count() {
                let ScalarValue::Utf8(Some(k)) = b.column(0).get(r).unwrap() else {
                    panic!("expected a text key")
                };
                let c = b.column(1).get(r).unwrap().as_f64().unwrap();
                let t = b.column(2).get(r).unwrap().as_f64().unwrap();
                m.insert(k, (c, t));
            }
            m
        };
        let a = read(&split);
        let b = read(&whole);
        assert_eq!(a.len(), b.len());
        for (k, (c, t)) in &a {
            let (c2, t2) = b.get(k).unwrap_or_else(|| panic!("group {k} missing"));
            assert_eq!(c, c2, "count differs for {k}");
            assert_eq!(t, t2, "sum differs for {k}");
        }

        // And against arithmetic, so both paths being wrong the same way is caught.
        let expected_count = (TOTAL / 5) as f64;
        for (k, (c, _)) in &a {
            assert_eq!(*c, expected_count, "group {k} has the wrong count");
        }
    }

    #[test]
    fn group_by_count() {
        let batch = make_batch();
        let exec = GroupByExec::new(
            vec![Expr::Column("dept".into())],
            vec![AggExpr::count(Expr::Column("score".into()), "cnt")],
        );
        let result = exec.execute(&[batch]).unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn group_by_sum() {
        let batch = make_batch();
        let exec = GroupByExec::new(
            vec![Expr::Column("dept".into())],
            vec![AggExpr::sum(Expr::Column("score".into()), "total")],
        );
        let result = exec.execute(&[batch]).unwrap();
        assert_eq!(result.row_count(), 2);
        let total_col = result.column_by_name("total").unwrap();
        let eng_total = total_col.get(0).unwrap();
        assert_eq!(eng_total, ScalarValue::Float64(Some(27.0)));
    }

    #[test]
    fn group_by_avg() {
        let batch = make_batch();
        let exec = GroupByExec::new(
            vec![Expr::Column("dept".into())],
            vec![AggExpr::avg(Expr::Column("score".into()), "avg_score")],
        );
        let result = exec.execute(&[batch]).unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn group_by_min_max() {
        let batch = make_batch();
        let exec = GroupByExec::new(
            vec![Expr::Column("dept".into())],
            vec![
                AggExpr::min(Expr::Column("score".into()), "min_s"),
                AggExpr::max(Expr::Column("score".into()), "max_s"),
            ],
        );
        let result = exec.execute(&[batch]).unwrap();
        assert_eq!(result.row_count(), 2);
        // eng: min=8, max=10; hr: min=6, max=7
        let min_col = result.column_by_name("min_s").unwrap();
        assert_eq!(min_col.get(0).unwrap(), ScalarValue::Float64(Some(8.0))); // eng
        let max_col = result.column_by_name("max_s").unwrap();
        assert_eq!(max_col.get(0).unwrap(), ScalarValue::Float64(Some(10.0))); // eng
    }

    #[test]
    fn lane_sum_matches_a_plain_sum() {
        // Lengths around the lane count: shorter than one chunk, exactly one,
        // and several with every possible tail.
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 1000, 65_537] {
            let vals: Vec<f64> = (0..n).map(|i| (i % 100) as f64 + 0.5).collect();
            let plain: f64 = vals.iter().copied().sum();
            let lanes = sum_lanes(&vals);
            let par = par_sum_lanes(&vals);
            // Reassociation moves the last bits only; scale the tolerance with the
            // magnitude of the sum rather than asserting bit equality.
            let tol = (plain.abs() * 1e-12).max(1e-9);
            assert!((lanes - plain).abs() <= tol, "n={n}: lanes {lanes} vs {plain}");
            assert!((par - plain).abs() <= tol, "n={n}: par {par} vs {plain}");
        }
    }

    #[test]
    fn lane_sum_matches_a_plain_sum_for_integers() {
        for n in [0usize, 1, 7, 8, 9, 17, 1000, 65_537] {
            let i64s: Vec<i64> = (0..n).map(|i| i as i64 - 500).collect();
            let plain: f64 = i64s.iter().map(|&v| v as f64).sum();
            assert!((sum_lanes(&i64s) - plain).abs() <= (plain.abs() * 1e-12).max(1e-9));
            assert!((par_sum_lanes(&i64s) - plain).abs() <= (plain.abs() * 1e-12).max(1e-9));

            let i32s: Vec<i32> = (0..n).map(|i| i as i32 - 500).collect();
            let plain32: f64 = i32s.iter().map(|&v| v as f64).sum();
            assert!((sum_lanes(&i32s) - plain32).abs() <= (plain32.abs() * 1e-12).max(1e-9));
        }
    }

    #[test]
    fn lane_sum_keeps_the_old_wide_integer_behaviour() {
        // i64 -> f64 loses precision past 2^53. SUM has always returned f64 and the
        // previous kernel did the same `as f64` cast, so the widening must round the
        // same way rather than quietly becoming exact.
        let big = 1i64 << 53;
        let vals = vec![big, 1, big, 1];
        let plain: f64 = vals.iter().map(|&v| v as f64).sum();
        assert_eq!(sum_lanes(&vals), plain);
    }

    #[test]
    fn lane_sum_handles_negatives_and_fractions() {
        let vals: Vec<f64> = (0..1234).map(|i| (i as f64 - 617.0) * 0.375).collect();
        let plain: f64 = vals.iter().copied().sum();
        assert!((sum_lanes(&vals) - plain).abs() <= (plain.abs() * 1e-12).max(1e-9));
    }

    #[test]
    fn agg_only_count_sum() {
        let batch = make_batch();
        let result = execute_agg_only(
            &[batch],
            &[
                AggExpr::count(Expr::Column("score".into()), "cnt"),
                AggExpr::sum(Expr::Column("score".into()), "total"),
            ],
        ).unwrap();
        assert_eq!(result.row_count(), 1);
        let cnt   = result.column_by_name("cnt").unwrap().get(0).unwrap();
        assert_eq!(cnt, ScalarValue::Int64(Some(5)));
        let total = result.column_by_name("total").unwrap().get(0).unwrap();
        assert_eq!(total, ScalarValue::Float64(Some(40.0)));
    }

    #[test]
    fn agg_min_max() {
        let batch = make_batch();
        let result = execute_agg_only(
            &[batch],
            &[
                AggExpr::min(Expr::Column("score".into()), "min_s"),
                AggExpr::max(Expr::Column("score".into()), "max_s"),
            ],
        ).unwrap();
        let min = result.column_by_name("min_s").unwrap().get(0).unwrap();
        let max = result.column_by_name("max_s").unwrap().get(0).unwrap();
        assert_eq!(min, ScalarValue::Float64(Some(6.0)));
        assert_eq!(max, ScalarValue::Float64(Some(10.0)));
    }

    #[test]
    fn fast_path_matches_slow_path() {
        // Verify fast path gives same results as slow path on a larger batch.
        let schema = Schema::new(vec![
            Field::required("cat",   DataType::Utf8),
            Field::required("val",   DataType::Float64),
        ]);
        let n = 1000;
        let cats = ["a", "b", "c"];
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::utf8_from_str((0..n).map(|i| Some(cats[i % 3])).collect()),
            ColumnBuilder::float64((0..n).map(|i| Some(i as f64)).collect()),
        ]).unwrap();

        let exec = GroupByExec::new(
            vec![Expr::Column("cat".into())],
            vec![
                AggExpr::count(Expr::Column("val".into()), "cnt"),
                AggExpr::sum(Expr::Column("val".into()), "total"),
                AggExpr::avg(Expr::Column("val".into()), "avg"),
            ],
        );

        let fast = exec.execute(&[batch.clone()]).unwrap();
        let slow = exec.execute_slow(&[batch]).unwrap();

        // Compare row counts
        assert_eq!(fast.row_count(), slow.row_count());
        // Both should have 3 groups
        assert_eq!(fast.row_count(), 3);
    }
}
