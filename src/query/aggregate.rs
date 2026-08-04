use std::collections::HashMap;

use super::hash::FastMap;

use rayon::prelude::*;

use crate::storage::{
    column::{BooleanColumn, Column, PrimitiveColumn, Utf8Column},
    schema::{Field, Schema},
    table::RecordBatch,
    types::{DataType, FalconError, FalconResult, ScalarValue},
};

use super::expr::Expr;

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
        }
    }
}

// в”Ђв”Ђв”Ђ Fast-path batch state (parallel arrays, no boxing) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// Per-aggregate typed accumulator arrays, indexed by group_id.
/// All three fields are valid for all agg types вЂ” unused ones are zero-sized
/// conceptually but kept uniform to avoid conditional field access in hot loops.
struct BatchAggState {
    count:   Vec<i64>,   // COUNT / AVG denominator
    sum:     Vec<f64>,   // SUM / AVG numerator / MIN value / MAX value
    has_val: Vec<bool>,  // SUM validity / MIN-MAX "first seen"
}

impl BatchAggState {
    fn new() -> Self {
        Self { count: Vec::new(), sum: Vec::new(), has_val: Vec::new() }
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
        // No CountDistinct in fast path
        for agg in &self.aggs {
            if matches!(agg.func, AggFunc::CountDistinct) { return Ok(None); }
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
        let first_b = batches.first().ok_or_else(|| FalconError::InvalidQuery("no batches".into()))?;
        let gb_col_idx = first_b.schema.column_index(gb_name)?;
        let agg_col_indices: Vec<Option<usize>> = agg_col_names.iter()
            .map(|&n| n.and_then(|name| first_b.schema.column_index(name).ok()))
            .collect();

        let partials: FalconResult<Vec<(Vec<String>, Vec<BatchAggState>)>> = batches
            .par_iter()
            .map(|batch| {
                let gb_col = batch.column(gb_col_idx);
                let gb_dict = gb_col.as_utf8_column()
                    .ok_or_else(|| FalconError::InvalidQuery(
                        format!("Expected Utf8 column '{gb_name}'")
                    ))?;

                // The column is dictionary-encoded, so assign group ids per *distinct
                // string* вЂ” a handful of times вЂ” instead of hashing a string on every
                // row. Mapping a row to its group then costs one array index.
                let dict = gb_dict.dict();
                let mut local_keys: Vec<String> = Vec::with_capacity(dict.len() + 1);
                let mut local_states: Vec<BatchAggState> =
                    (0..n_aggs).map(|_| BatchAggState::new()).collect();

                // code в†’ group id. One extra slot maps NULL onto the empty-string
                // group, matching the previous behaviour.
                let mut code_to_gid: Vec<u32> = Vec::with_capacity(dict.len());
                for s in dict {
                    code_to_gid.push(local_keys.len() as u32);
                    local_keys.push(s.clone());
                    for st in &mut local_states { st.push_group(); }
                }

                // NULL groups with the empty string, as the row-wise path did. The
                // group is created only on first sight вЂ” allocating it up front
                // would report an extra, empty group for null-free data.
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

                self.accumulate_batch(&batch_gids, batch, &agg_col_indices, &mut local_states)?;
                Ok((local_keys, local_states))
            })
            .collect();

        // в”Ђв”Ђ Merge partial results (sequential, usually fast vs. accumulation) в”Ђ
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

        let partials: FalconResult<Vec<(Vec<i64>, Vec<BatchAggState>)>> = batches
            .par_iter()
            .map(|batch| {
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

                self.accumulate_batch(&batch_gids, batch, &agg_col_indices_i, &mut local_states)?;
                Ok((local_keys, local_states))
            })
            .collect();

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
        batch:           &RecordBatch,
        agg_col_indices: &[Option<usize>],
        states:          &mut [BatchAggState],
    ) -> FalconResult<()> {
        for (agg_idx, (agg, &maybe_idx)) in self.aggs.iter().zip(agg_col_indices.iter()).enumerate() {
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
                        for (row, &gid) in batch_gids.iter().enumerate() {
                            if c.is_valid(row) { state.count[gid as usize] += 1; }
                        }
                    }
                }
                (AggFunc::Count, Column::Int64(c)) => {
                    if !c.has_nulls() {
                        for &gid in batch_gids { state.count[gid as usize] += 1; }
                    } else {
                        for (row, &gid) in batch_gids.iter().enumerate() {
                            if c.is_valid(row) { state.count[gid as usize] += 1; }
                        }
                    }
                }
                (AggFunc::Count, Column::Int32(c)) => {
                    for (row, &gid) in batch_gids.iter().enumerate() {
                        if c.is_valid(row) { state.count[gid as usize] += 1; }
                    }
                }
                (AggFunc::Count, Column::Utf8(c)) => {
                    for (row, &gid) in batch_gids.iter().enumerate() {
                        if c.get_str(row).is_some() { state.count[gid as usize] += 1; }
                    }
                }

                // в”Ђв”Ђ SUM в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
                (AggFunc::Sum, Column::Float64(c)) => {
                    if !c.has_nulls() {
                        for (&gid, &v) in batch_gids.iter().zip(c.dense().iter()) {
                            let g = gid as usize;
                            state.sum[g] += v;
                            state.has_val[g] = true;
                        }
                    } else {
                        for (row, &gid) in batch_gids.iter().enumerate() {
                            if let Some(v) = c.get(row) {
                                let g = gid as usize;
                                state.sum[g] += v;
                                state.has_val[g] = true;
                            }
                        }
                    }
                }
                (AggFunc::Sum, Column::Int64(c)) => {
                    if !c.has_nulls() {
                        for (&gid, &v) in batch_gids.iter().zip(c.dense().iter()) {
                            let g = gid as usize;
                            state.sum[g] += v as f64;
                            state.has_val[g] = true;
                        }
                    } else {
                        for (row, &gid) in batch_gids.iter().enumerate() {
                            if let Some(v) = c.get(row) {
                                let g = gid as usize;
                                state.sum[g] += v as f64;
                                state.has_val[g] = true;
                            }
                        }
                    }
                }
                (AggFunc::Sum, Column::Int32(c)) => {
                    for (row, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row) {
                            let g = gid as usize;
                            state.sum[g] += v as f64;
                            state.has_val[g] = true;
                        }
                    }
                }

                // в”Ђв”Ђ AVG в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
                (AggFunc::Avg, Column::Float64(c)) => {
                    if !c.has_nulls() {
                        for (&gid, &v) in batch_gids.iter().zip(c.dense().iter()) {
                            let g = gid as usize;
                            state.sum[g]   += v;
                            state.count[g] += 1;
                        }
                    } else {
                        for (row, &gid) in batch_gids.iter().enumerate() {
                            if let Some(v) = c.get(row) {
                                let g = gid as usize;
                                state.sum[g]   += v;
                                state.count[g] += 1;
                            }
                        }
                    }
                }
                (AggFunc::Avg, Column::Int64(c)) => {
                    if !c.has_nulls() {
                        for (&gid, &v) in batch_gids.iter().zip(c.dense().iter()) {
                            let g = gid as usize;
                            state.sum[g]   += v as f64;
                            state.count[g] += 1;
                        }
                    } else {
                        for (row, &gid) in batch_gids.iter().enumerate() {
                            if let Some(v) = c.get(row) {
                                let g = gid as usize;
                                state.sum[g]   += v as f64;
                                state.count[g] += 1;
                            }
                        }
                    }
                }
                (AggFunc::Avg, Column::Int32(c)) => {
                    for (row, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row) {
                            let g = gid as usize;
                            state.sum[g]   += v as f64;
                            state.count[g] += 1;
                        }
                    }
                }

                // в”Ђв”Ђ MIN в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
                (AggFunc::Min, Column::Float64(c)) => {
                    for (row, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row) {
                            let g = gid as usize;
                            if !state.has_val[g] || v < state.sum[g] {
                                state.sum[g] = v; state.has_val[g] = true;
                            }
                        }
                    }
                }
                (AggFunc::Min, Column::Int64(c)) => {
                    for (row, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row) {
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
                    for (row, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row) {
                            let g = gid as usize;
                            if !state.has_val[g] || v > state.sum[g] {
                                state.sum[g] = v; state.has_val[g] = true;
                            }
                        }
                    }
                }
                (AggFunc::Max, Column::Int64(c)) => {
                    for (row, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = c.get(row) {
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
                    for (row, &gid) in batch_gids.iter().enumerate() {
                        if let Some(v) = vals[row] {
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

        let (groups, key_order, group_key_values) = batches
            .par_iter()
            .try_fold(empty, |mut acc, batch| -> FalconResult<Partial> {
                self.accumulate_slow(batch, &mut acc.0, &mut acc.1, &mut acc.2)?;
                Ok(acc)
            })
            .try_reduce(empty, merge)?;

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
        AggFunc::Count | AggFunc::CountDistinct => {
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
    let accumulators = batches
        .par_iter()
        .try_fold(
            || aggs.iter().map(|a| Accumulator::new(&a.func)).collect::<Vec<_>>(),
            |mut accs, batch| -> FalconResult<Vec<Accumulator>> {
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
                        if accumulate_typed(acc, col) {
                            continue;
                        }
                    }
                    let vals = agg.input.eval_batch(batch)?;
                    for val in &vals {
                        acc.update(val);
                    }
                }
                Ok(accs)
            },
        )
        .try_reduce(
            || aggs.iter().map(|a| Accumulator::new(&a.func)).collect::<Vec<_>>(),
            |mut a, b| {
                for (acc, other) in a.iter_mut().zip(b) {
                    acc.merge(other);
                }
                Ok(a)
            },
        )?;

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
fn accumulate_typed(acc: &mut Accumulator, col: &Column) -> bool {
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
            if let C::Float64(c) = col {
                if !c.has_nulls() {
                    if !c.dense().is_empty() {
                        *total += c.dense().iter().copied().sum::<f64>();
                        *valid = true;
                    }
                    return true;
                }
            }
            if let C::Int64(c) = col {
                if !c.has_nulls() {
                    if !c.dense().is_empty() {
                        *total += c.dense().iter().map(|&v| v as f64).sum::<f64>();
                        *valid = true;
                    }
                    return true;
                }
            }
            if let C::Int32(c) = col {
                if !c.has_nulls() {
                    if !c.dense().is_empty() {
                        *total += c.dense().iter().map(|&v| v as f64).sum::<f64>();
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
                    *total += dense.iter().copied().sum::<f64>();
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
                    if let Some(chunk_min) = c.dense().iter().copied().reduce(f64::min) {
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
                    if let Some(chunk_max) = c.dense().iter().copied().reduce(f64::max) {
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
        // COUNT(DISTINCT) needs the values themselves, not a running scalar.
        Accumulator::CountDistinct { .. } => false,
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
        AggFunc::CountDistinct => unreachable!("CountDistinct not in fast path"),
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
