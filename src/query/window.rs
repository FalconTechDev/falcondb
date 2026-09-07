use std::collections::hash_map::Entry;

use super::hash::FastMap;

use crate::storage::{
    column::{Column, PrimitiveColumn, Utf8Column},
    schema::Field,
    table::RecordBatch,
    types::{DataType, FalconError, FalconResult, ScalarValue},
};

use super::{expr::Expr, sort::SortOrder};

// ─── Window function kinds ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum WindowFunc {
    RowNumber,
    Rank,
    DenseRank,
    Lag(usize),
    Lead(usize),
    Sum,
    Avg,
    Min,
    Max,
    Count,
    FirstValue,
    LastValue,
    NthValue(usize),
    PercentRank,
    CumeDist,
    Ntile(usize),
}

// ─── Window frame ─────────────────────────────────────────────────────────────

/// Which rows of a partition an aggregate window function sees at each row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrame {
    /// `UNBOUNDED PRECEDING AND CURRENT ROW` — the SQL default. The frame grows
    /// as the scan advances, giving a running total.
    Cumulative,
    /// `UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` — the whole partition at every row.
    WholePartition,
    /// `<preceding> PRECEDING AND UNBOUNDED FOLLOWING` — frame from N rows back to
    /// the end of the partition. Use `preceding = 0` for CURRENT ROW AND UNBOUNDED FOLLOWING.
    UnboundedFollowing { preceding: usize },
    /// `ROWS BETWEEN <preceding> PRECEDING AND <following> FOLLOWING` — a frame of
    /// bounded width that slides along the partition. This is what makes a rolling
    /// average rolling rather than cumulative.
    Rows { preceding: usize, following: usize },
}

impl WindowFrame {
    /// Frame bounds for the row at `pos` in a partition of `len` rows, as an
    /// inclusive `[lo, hi]` range of positions.
    fn bounds(&self, pos: usize, len: usize) -> (usize, usize) {
        match self {
            WindowFrame::Cumulative              => (0, pos),
            WindowFrame::WholePartition          => (0, len.saturating_sub(1)),
            WindowFrame::UnboundedFollowing { preceding } => {
                (pos.saturating_sub(*preceding), len.saturating_sub(1))
            }
            WindowFrame::Rows { preceding, following } => (
                pos.saturating_sub(*preceding),
                (pos + following).min(len - 1),
            ),
        }
    }
}

impl std::fmt::Display for WindowFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowFrame::Cumulative => {
                f.write_str("ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW")
            }
            WindowFrame::WholePartition => {
                f.write_str("ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING")
            }
            WindowFrame::UnboundedFollowing { preceding } => write!(
                f,
                "ROWS BETWEEN {preceding} PRECEDING AND UNBOUNDED FOLLOWING"
            ),
            WindowFrame::Rows { preceding, following } => write!(
                f,
                "ROWS BETWEEN {preceding} PRECEDING AND {following} FOLLOWING"
            ),
        }
    }
}

// ─── Window expression ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WindowExpr {
    pub func:         WindowFunc,
    /// Input column expression (None for ROW_NUMBER / RANK / DENSE_RANK / COUNT(*)).
    pub input:        Option<Expr>,
    pub partition_by: Vec<Expr>,
    pub order_by:     Vec<super::sort::SortExpr>,
    /// Frame for the aggregate functions (SUM/AVG/MIN/MAX/COUNT). Ranking and
    /// offset functions ignore it — their meaning is fixed by the ordering.
    pub frame:        WindowFrame,
    pub alias:        String,
}

impl WindowExpr {
    /// Compute the window column for `batch` (all rows must be in one batch).
    pub fn execute(&self, batch: &RecordBatch) -> FalconResult<Column> {
        let n = batch.row_count();
        if n == 0 {
            return Ok(Column::Int64(PrimitiveColumn::new(vec![])));
        }

        // 1. Evaluate partition keys (one Vec<ScalarValue> per PARTITION BY expression).
        let part_vals: Vec<Vec<ScalarValue>> = self.partition_by.iter()
            .map(|e| e.eval_batch(batch))
            .collect::<FalconResult<_>>()?;

        // 2. Evaluate ORDER BY columns inside the window.
        let order_vals: Vec<Vec<ScalarValue>> = self.order_by.iter()
            .map(|s| s.expr.eval_batch(batch))
            .collect::<FalconResult<_>>()?;

        // A single numeric ORDER BY key can be read as raw f64s, skipping the
        // `ScalarValue` box per row that the sort below would otherwise unpack again.
        // `None` here just means the general path is used, as it was before.
        let order_dense: Option<Vec<f64>> = (self.order_by.len() == 1)
            .then(|| self.order_by[0].expr.eval_dense_f64(batch))
            .flatten();

        // 3. Evaluate input column (for SUM/AVG/LAG/LEAD etc.).
        let input_vals: Option<Vec<ScalarValue>> = self.input.as_ref()
            .map(|e| e.eval_batch(batch))
            .transpose()?;

        // 4. Group rows into partitions (preserve insertion order for determinism).
        //
        // Without PARTITION BY there is exactly one partition holding every row, so
        // the map is skipped entirely. Building it anyway meant formatting a key and
        // doing a hash-map insert per row — 100k of them, all on the same empty key.
        let partitions: Vec<Vec<usize>> = if part_vals.is_empty() {
            vec![(0..n).collect()]
        } else {
            let mut part_map: FastMap<Vec<String>, Vec<usize>> = FastMap::default();
            let mut part_order: Vec<Vec<String>> = Vec::new();
            for row in 0..n {
                let key: Vec<String> =
                    part_vals.iter().map(|col| col[row].to_string()).collect();
                match part_map.entry(key.clone()) {
                    Entry::Vacant(e) => { part_order.push(key); e.insert(vec![row]); }
                    Entry::Occupied(mut e) => { e.get_mut().push(row); }
                }
            }
            part_order.iter().map(|k| part_map[k].clone()).collect()
        };

        // 5. Apply window function per partition.
        // ROW_NUMBER, RANK and DENSE_RANK all produce a non-NULL i64 per row, so
        // their result is written densely instead of through a `ScalarValue` per
        // row and a conversion pass afterwards. Every other function keeps the
        // general buffer, which also carries NULLs (LAG past the start, an empty
        // frame) that a dense i64 array cannot express.
        if matches!(
            self.func,
            WindowFunc::RowNumber | WindowFunc::Rank | WindowFunc::DenseRank
        ) {
            let mut ranks: Vec<i64> = vec![0; n];
            for indices in &partitions {
                let mut sorted = indices.clone();
                if !order_vals.is_empty()
                    && !try_dense_partition_sort(order_dense.as_deref(), &self.order_by, &mut sorted)
                    && !try_typed_partition_sort(&order_vals, &self.order_by, &mut sorted)
                {
                    sorted.sort_unstable_by(|&a, &b| {
                        for (col, expr) in order_vals.iter().zip(self.order_by.iter()) {
                            let ord = cmp_scalar(&col[a], &col[b]);
                            let ord = match expr.order {
                                SortOrder::Asc => ord,
                                SortOrder::Desc => ord.reverse(),
                            };
                            if ord != std::cmp::Ordering::Equal { return ord; }
                        }
                        std::cmp::Ordering::Equal
                    });
                }
                rank_dense(&self.func, &sorted, &order_vals, &mut ranks);
            }
            return Ok(Column::Int64(PrimitiveColumn::from_non_null(ranks)));
        }

        let mut result: Vec<ScalarValue> = vec![ScalarValue::Null; n];

        for indices in &partitions {

            // Sort within partition by ORDER BY keys.
            let mut sorted = indices.clone();
            if !order_vals.is_empty()
                && !try_dense_partition_sort(order_dense.as_deref(), &self.order_by, &mut sorted)
                && !try_typed_partition_sort(&order_vals, &self.order_by, &mut sorted)
            {
                sorted.sort_unstable_by(|&a, &b| {
                    for (col, expr) in order_vals.iter().zip(self.order_by.iter()) {
                        let ord = cmp_scalar(&col[a], &col[b]);
                        let ord = match expr.order { SortOrder::Asc => ord, SortOrder::Desc => ord.reverse() };
                        if ord != std::cmp::Ordering::Equal { return ord; }
                    }
                    std::cmp::Ordering::Equal
                });
            }

            apply_window_func(
                &self.func,
                &sorted,
                &order_vals,
                input_vals.as_deref(),
                self.frame,
                &mut result,
            )?;
        }

        Ok(scalars_to_col(&result))
    }
}


/// Dense ranking for the three functions whose output is always a non-NULL i64.
///
/// Mirrors the `RowNumber` / `Rank` / `DenseRank` arms of [`apply_window_func`];
/// the tie rules are the same, which is what the tests pin. Kept separate rather
/// than generalising that function because the others produce values of the input's
/// type and can be NULL.
fn rank_dense(
    func:       &WindowFunc,
    sorted:     &[usize],
    order_vals: &[Vec<ScalarValue>],
    out:        &mut [i64],
) {
    let tied = |a: usize, b: usize| -> bool {
        order_vals.iter().all(|col| cmp_scalar(&col[a], &col[b]) == std::cmp::Ordering::Equal)
    };

    match func {
        WindowFunc::RowNumber => {
            for (pos, &row) in sorted.iter().enumerate() {
                out[row] = pos as i64 + 1;
            }
        }
        WindowFunc::Rank => {
            let mut rank = 1i64;
            for (pos, &row) in sorted.iter().enumerate() {
                if pos > 0 && !tied(sorted[pos - 1], row) {
                    rank = pos as i64 + 1;
                }
                out[row] = rank;
            }
        }
        WindowFunc::DenseRank => {
            let mut rank = 1i64;
            for (pos, &row) in sorted.iter().enumerate() {
                if pos > 0 && !tied(sorted[pos - 1], row) {
                    rank += 1;
                }
                out[row] = rank;
            }
        }
        _ => unreachable!("rank_dense is only called for the three ranking functions"),
    }
}

/// Sort one partition by a single numeric ORDER BY key read straight from the
/// column, with no `ScalarValue` in the way.
///
/// The same ordering as [`try_typed_partition_sort`], which stays for the cases the
/// dense evaluator declines — NULLs, text, several keys. Both build the same
/// total-order `u64` keys and hand them to the same radix sort, so they cannot
/// order rows differently.
fn try_dense_partition_sort(
    dense:    Option<&[f64]>,
    order_by: &[super::sort::SortExpr],
    sorted:   &mut Vec<usize>,
) -> bool {
    let Some(vals) = dense else { return false };
    if order_by.len() != 1 {
        return false;
    }
    let desc = matches!(order_by[0].order, SortOrder::Desc);

    let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(sorted.len());
    for (pos, &row) in sorted.iter().enumerate() {
        let key = super::sort::f64_sort_key(vals[row]);
        pairs.push((if desc { !key } else { key }, pos as u32));
    }
    super::sort::sort_u64_pairs(&mut pairs);

    let original = sorted.clone();
    for (out, &(_, pos)) in sorted.iter_mut().zip(pairs.iter()) {
        *out = original[pos as usize];
    }
    true
}

/// Sort one partition by a single numeric ORDER BY key, without `cmp_scalar`.
///
/// The general path compares two `ScalarValue`s per comparison — a dynamic enum
/// match, roughly 1.7M times at 100k rows. Converting the column to total-order
/// `u64` keys once turns that into the same radix sort `ORDER BY` already uses.
///
/// Returns `false` when it does not apply — several ORDER BY keys, a NULL, or a
/// non-numeric type — so the caller falls back and the semantics stay in one place.
/// DESC flips the key bits, which is what `query::sort` does for the same reason:
/// it keeps a single ascending sort rather than a second comparison path.
fn try_typed_partition_sort(
    order_vals: &[Vec<ScalarValue>],
    order_by:   &[super::sort::SortExpr],
    sorted:     &mut Vec<usize>,
) -> bool {
    if order_vals.len() != 1 {
        return false;
    }
    let col = &order_vals[0];
    let desc = matches!(order_by[0].order, SortOrder::Desc);

    let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(sorted.len());
    for (pos, &row) in sorted.iter().enumerate() {
        let key = match &col[row] {
            ScalarValue::Float64(Some(v)) => super::sort::f64_sort_key(*v),
            ScalarValue::Float32(Some(v)) => super::sort::f64_sort_key(*v as f64),
            // Shift the sign bit so the unsigned ordering matches the signed one.
            ScalarValue::Int64(Some(v))  => (*v as u64) ^ (1u64 << 63),
            ScalarValue::Int32(Some(v))  => ((*v as i64) as u64) ^ (1u64 << 63),
            ScalarValue::Date32(Some(v)) => ((*v as i64) as u64) ^ (1u64 << 63),
            ScalarValue::Timestamp(Some(v)) => (*v as u64) ^ (1u64 << 63),
            _ => return false,
        };
        pairs.push((if desc { !key } else { key }, pos as u32));
    }

    super::sort::sort_u64_pairs(&mut pairs);

    // `pairs` holds positions within `sorted`, so reorder through a copy of it.
    let original = sorted.clone();
    for (out, &(_, pos)) in sorted.iter_mut().zip(pairs.iter()) {
        *out = original[pos as usize];
    }
    true
}

// ─── Per-partition function application ──────────────────────────────────────

fn apply_window_func(
    func:       &WindowFunc,
    sorted:     &[usize],
    order_vals: &[Vec<ScalarValue>],
    input_vals: Option<&[ScalarValue]>,
    frame:      WindowFrame,
    result:     &mut [ScalarValue],
) -> FalconResult<()> {
    match func {
        WindowFunc::RowNumber => {
            for (rank, &row) in sorted.iter().enumerate() {
                result[row] = ScalarValue::Int64(Some(rank as i64 + 1));
            }
        }

        WindowFunc::Rank => {
            let mut rank = 1i64;
            for (pos, &row) in sorted.iter().enumerate() {
                if pos > 0 {
                    let prev = sorted[pos - 1];
                    let tied = order_vals.iter().all(|col| col[row] == col[prev]);
                    if !tied { rank = pos as i64 + 1; }
                }
                result[row] = ScalarValue::Int64(Some(rank));
            }
        }

        WindowFunc::DenseRank => {
            let mut rank = 1i64;
            for (pos, &row) in sorted.iter().enumerate() {
                if pos > 0 {
                    let prev = sorted[pos - 1];
                    let tied = order_vals.iter().all(|col| col[row] == col[prev]);
                    if !tied { rank += 1; }
                }
                result[row] = ScalarValue::Int64(Some(rank));
            }
        }

        WindowFunc::Lag(offset) => {
            let vals = need_input(input_vals, "LAG")?;
            for (pos, &row) in sorted.iter().enumerate() {
                result[row] = if pos >= *offset {
                    vals[sorted[pos - *offset]].clone()
                } else {
                    ScalarValue::Null
                };
            }
        }

        WindowFunc::Lead(offset) => {
            let vals = need_input(input_vals, "LEAD")?;
            let len = sorted.len();
            for (pos, &row) in sorted.iter().enumerate() {
                result[row] = if pos + *offset < len {
                    vals[sorted[pos + *offset]].clone()
                } else {
                    ScalarValue::Null
                };
            }
        }

        WindowFunc::Sum => {
            let vals = need_input(input_vals, "SUM")?;
            frame_sum_avg(sorted, vals, frame, result, false);
        }

        WindowFunc::Avg => {
            let vals = need_input(input_vals, "AVG")?;
            frame_sum_avg(sorted, vals, frame, result, true);
        }

        WindowFunc::Min => {
            let vals = need_input(input_vals, "MIN")?;
            frame_min_max(sorted, vals, frame, result, true);
        }

        WindowFunc::Max => {
            let vals = need_input(input_vals, "MAX")?;
            frame_min_max(sorted, vals, frame, result, false);
        }

        WindowFunc::Count => {
            // COUNT(*) counts frame rows; COUNT(expr) counts non-NULL ones.
            let len = sorted.len();
            for (pos, &row) in sorted.iter().enumerate() {
                let (lo, hi) = frame.bounds(pos, len);
                let n = match input_vals {
                    None => (hi - lo + 1) as i64,
                    Some(vals) => sorted[lo..=hi]
                        .iter()
                        .filter(|&&r| !vals[r].is_null())
                        .count() as i64,
                };
                result[row] = ScalarValue::Int64(Some(n));
            }
        }

        WindowFunc::FirstValue => {
            let vals = need_input(input_vals, "FIRST_VALUE")?;
            let len = sorted.len();
            for (pos, &row) in sorted.iter().enumerate() {
                let (lo, _hi) = frame.bounds(pos, len);
                result[row] = vals[sorted[lo]].clone();
            }
        }

        WindowFunc::LastValue => {
            let vals = need_input(input_vals, "LAST_VALUE")?;
            let len = sorted.len();
            for (pos, &row) in sorted.iter().enumerate() {
                let (_lo, hi) = frame.bounds(pos, len);
                result[row] = vals[sorted[hi]].clone();
            }
        }

        WindowFunc::NthValue(n) => {
            let vals = need_input(input_vals, "NTH_VALUE")?;
            let len = sorted.len();
            for (pos, &row) in sorted.iter().enumerate() {
                let (lo, hi) = frame.bounds(pos, len);
                let idx = lo + n.saturating_sub(1);
                result[row] = if idx <= hi {
                    vals[sorted[idx]].clone()
                } else {
                    ScalarValue::Null
                };
            }
        }

        WindowFunc::PercentRank => {
            let len = sorted.len();
            if len <= 1 {
                for &row in sorted { result[row] = ScalarValue::Float64(Some(0.0)); }
            } else {
                let mut rank = 1i64;
                for (pos, &row) in sorted.iter().enumerate() {
                    if pos > 0 {
                        let prev = sorted[pos - 1];
                        let tied = order_vals.iter().all(|col| col[row] == col[prev]);
                        if !tied { rank = pos as i64 + 1; }
                    }
                    result[row] = ScalarValue::Float64(Some(
                        (rank - 1) as f64 / (len - 1) as f64
                    ));
                }
            }
        }

        WindowFunc::CumeDist => {
            let len = sorted.len();
            let mut pos = 0usize;
            while pos < len {
                // Find the end of the peer group (rows with equal ORDER BY values).
                let mut end = pos + 1;
                while end < len {
                    let tied = order_vals.iter().all(|col| col[sorted[end]] == col[sorted[pos]]);
                    if !tied { break; }
                    end += 1;
                }
                let cume = end as f64 / len as f64;
                for i in pos..end {
                    result[sorted[i]] = ScalarValue::Float64(Some(cume));
                }
                pos = end;
            }
        }

        WindowFunc::Ntile(buckets) => {
            let len = sorted.len();
            let buckets = (*buckets).max(1);
            for (pos, &row) in sorted.iter().enumerate() {
                // SQL NTILE: ceil-first distribution. First (len % buckets) buckets get
                // floor(len/buckets)+1 rows, the rest get floor(len/buckets) rows.
                let large = len % buckets;
                let large_size = len / buckets + 1;
                let bucket = if pos < large * large_size {
                    pos / large_size + 1
                } else {
                    large + (pos - large * large_size) / (len / buckets) + 1
                };
                result[row] = ScalarValue::Int64(Some(bucket as i64));
            }
        }
    }
    Ok(())
}

// ─── Frame-aware aggregates ───────────────────────────────────────────────────

/// Rolling SUM (or AVG when `want_avg`) over the frame, in O(n).
///
/// Both frame shapes advance monotonically — `lo` and `hi` never move backwards as
/// `pos` increases — so the running total is maintained by adding entering rows and
/// subtracting leaving ones rather than re-summing the frame at every row.
///
/// Sums are kept in f64 and are therefore subject to floating-point drift; adding
/// and subtracting is not exactly associative, so a long rolling sum can differ in
/// the last bits from summing each frame independently.
fn frame_sum_avg(
    sorted:   &[usize],
    vals:     &[ScalarValue],
    frame:    WindowFrame,
    result:   &mut [ScalarValue],
    want_avg: bool,
) {
    let len = sorted.len();
    let mut total = 0.0f64;
    let mut count = 0i64;
    // Half-open [cur_lo, cur_hi) tracking which positions are currently summed.
    let mut cur_lo = 0usize;
    let mut cur_hi = 0usize;

    for (pos, &row) in sorted.iter().enumerate() {
        let (lo, hi) = frame.bounds(pos, len);
        let hi_ex = hi + 1;

        while cur_hi < hi_ex {
            if let Some(v) = vals[sorted[cur_hi]].as_f64() { total += v; count += 1; }
            cur_hi += 1;
        }
        while cur_lo < lo {
            if let Some(v) = vals[sorted[cur_lo]].as_f64() { total -= v; count -= 1; }
            cur_lo += 1;
        }

        result[row] = if count > 0 {
            let out = if want_avg { total / count as f64 } else { total };
            ScalarValue::Float64(Some(out))
        } else {
            ScalarValue::Null
        };
    }
}

/// Rolling MIN (or MAX when `!want_min`) over the frame, in O(n) amortised.
///
/// Uses a monotonic deque of positions: the front always holds the extreme of the
/// current frame. A value that enters while worse than the incumbent can never
/// become the answer before that incumbent leaves, so it is dropped on arrival —
/// which is what keeps the total work linear rather than O(n × frame width).
fn frame_min_max(
    sorted:   &[usize],
    vals:     &[ScalarValue],
    frame:    WindowFrame,
    result:   &mut [ScalarValue],
    want_min: bool,
) {
    use std::collections::VecDeque;

    let len = sorted.len();
    let mut deque: VecDeque<usize> = VecDeque::new(); // positions, values monotonic
    let mut cur_hi = 0usize;

    for (pos, &row) in sorted.iter().enumerate() {
        let (lo, hi) = frame.bounds(pos, len);
        let hi_ex = hi + 1;

        while cur_hi < hi_ex {
            if let Some(v) = vals[sorted[cur_hi]].as_f64() {
                while let Some(&back) = deque.back() {
                    let bv = vals[sorted[back]].as_f64().unwrap_or(f64::NAN);
                    let dominated = if want_min { bv >= v } else { bv <= v };
                    if dominated { deque.pop_back(); } else { break; }
                }
                deque.push_back(cur_hi);
            }
            cur_hi += 1;
        }
        // Drop entries that have fallen out of the left edge of the frame.
        while let Some(&front) = deque.front() {
            if front < lo { deque.pop_front(); } else { break; }
        }

        result[row] = match deque.front() {
            Some(&p) => ScalarValue::Float64(vals[sorted[p]].as_f64()),
            None     => ScalarValue::Null,
        };
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn need_input<'a>(vals: Option<&'a [ScalarValue]>, func: &str) -> FalconResult<&'a [ScalarValue]> {
    vals.ok_or_else(|| FalconError::InvalidQuery(format!("{func} requires an input expression")))
}

fn cmp_scalar(a: &ScalarValue, b: &ScalarValue) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.is_null(), b.is_null()) {
        (true, true)   => Ordering::Equal,
        (true, false)  => Ordering::Greater,
        (false, true)  => Ordering::Less,
        (false, false) => {
            // Temporal types first: `as_f64` does not cover them, so without these
            // arms a window ORDER BY on a date would rank every row as a tie.
            match (a, b) {
                (ScalarValue::Date32(Some(da)), ScalarValue::Date32(Some(db))) => da.cmp(db),
                (ScalarValue::Timestamp(Some(ta)), ScalarValue::Timestamp(Some(tb))) => ta.cmp(tb),
                _ => {
                    if let (Some(fa), Some(fb)) = (a.as_f64(), b.as_f64()) {
                        fa.partial_cmp(&fb).unwrap_or(Ordering::Equal)
                    } else if let (ScalarValue::Utf8(Some(sa)), ScalarValue::Utf8(Some(sb))) = (a, b) {
                        sa.cmp(sb)
                    } else {
                        Ordering::Equal
                    }
                }
            }
        }
    }
}

fn scalars_to_col(vals: &[ScalarValue]) -> Column {
    let dt = vals.iter().find(|v| !v.is_null())
        .map(|v| v.data_type())
        .unwrap_or(DataType::Int64);
    match &dt {
        DataType::Int64 | DataType::Int32 =>
            Column::Int64(PrimitiveColumn::new(vals.iter().map(|v| v.as_i64()).collect())),
        DataType::Float64 | DataType::Float32 =>
            Column::Float64(PrimitiveColumn::new(vals.iter().map(|v| v.as_f64()).collect())),
        DataType::Utf8 =>
            Column::Utf8(Utf8Column::new(vals.iter().map(|v| match v {
                ScalarValue::Utf8(s) => s.clone(), _ => None,
            }).collect())),
        DataType::Boolean =>
            Column::Boolean(crate::storage::column::BooleanColumn::new(
                vals.iter().map(|v| v.as_bool()).collect(),
            )),
        // Temporal values carry their payload in a variant `as_i64` does not read,
        // so without these arms LAG/LEAD/FIRST_VALUE over a date yield all NULLs.
        DataType::Date32 =>
            Column::Date32(PrimitiveColumn::new(vals.iter().map(|v| match v {
                ScalarValue::Date32(d) => *d, _ => None,
            }).collect())),
        DataType::Timestamp =>
            Column::Timestamp(PrimitiveColumn::new(vals.iter().map(|v| match v {
                ScalarValue::Timestamp(t) => *t, _ => None,
            }).collect())),
        DataType::Decimal128(p, s) =>
            Column::Decimal128(crate::storage::column::Decimal128Column::new(
                vals.iter().map(|v| match v {
                    ScalarValue::Decimal128(d) => d.value, _ => None,
                }).collect(),
                *p,
                *s,
            )),
        _ =>
            Column::Int64(PrimitiveColumn::new(vals.iter().map(|v| v.as_i64()).collect())),
    }
}

/// Add extra columns (window results) to a RecordBatch.
pub fn add_window_columns(
    batch:  RecordBatch,
    fields: Vec<Field>,
    cols:   Vec<Column>,
) -> FalconResult<RecordBatch> {
    let mut all_fields  = batch.schema.fields.clone();
    let mut all_columns = batch.columns;
    all_fields.extend(fields);
    all_columns.extend(cols.into_iter().map(std::sync::Arc::new));
    RecordBatch::from_shared(crate::storage::schema::Schema::new(all_fields), all_columns)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        query::{expr::Expr, sort::SortExpr},
        storage::{column::ColumnBuilder, schema::{Field, Schema}, table::RecordBatch, types::DataType},
    };

    /// The typed partition sort must order rows exactly as `cmp_scalar` does.
    ///
    /// Two orderings of the same data is how a window would silently start
    /// numbering rows differently, so this compares the fast path against the
    /// comparator it replaces rather than against a hand-written expectation.
    /// Keys are distinct here: with ties the two differ legitimately, because the
    /// general path sorts unstably while the radix path is stable, and SQL does not
    /// define which of two tied rows is numbered first.
    #[test]
    fn typed_partition_sort_matches_the_comparator() {
        for desc in [false, true] {
            let vals: Vec<ScalarValue> = (0..5_000)
                .map(|i| ScalarValue::Float64(Some(((i * 7919) % 5_000) as f64 - 2_500.0)))
                .collect();
            let order_vals = vec![vals];
            let expr = if desc {
                SortExpr::desc(Expr::Column("v".into()))
            } else {
                SortExpr::asc(Expr::Column("v".into()))
            };
            let order_by = vec![expr];

            let mut typed: Vec<usize> = (0..order_vals[0].len()).collect();
            assert!(
                try_typed_partition_sort(&order_vals, &order_by, &mut typed),
                "distinct Float64 keys should take the typed path"
            );

            let mut general: Vec<usize> = (0..order_vals[0].len()).collect();
            general.sort_unstable_by(|&a, &b| {
                let ord = cmp_scalar(&order_vals[0][a], &order_vals[0][b]);
                if desc { ord.reverse() } else { ord }
            });

            assert_eq!(typed, general, "desc={desc}: typed sort disagreed with cmp_scalar");
        }
    }

    /// Dense ranking must match the general implementation, ties included.
    ///
    /// RANK and DENSE_RANK differ only in how they treat equal keys, so a second
    /// implementation that got ties subtly wrong would still pass a row count and a
    /// spot check. This runs both over the same input, with deliberate ties.
    #[test]
    fn dense_ranking_matches_apply_window_func() {
        // Keys with runs of equal values, so every tie rule is exercised.
        let keys: Vec<ScalarValue> = [3, 1, 1, 2, 2, 2, 5, 1]
            .iter()
            .map(|&v| ScalarValue::Int64(Some(v)))
            .collect();
        let n = keys.len();
        let order_vals = vec![keys];

        // Ascending by key, ties left in input order — a stable sort, so both paths
        // see exactly the same ordering and only the ranking logic differs.
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by(|&a, &b| cmp_scalar(&order_vals[0][a], &order_vals[0][b]));

        for func in [WindowFunc::RowNumber, WindowFunc::Rank, WindowFunc::DenseRank] {
            let mut dense = vec![0i64; n];
            rank_dense(&func, &sorted, &order_vals, &mut dense);

            let mut boxed: Vec<ScalarValue> = vec![ScalarValue::Null; n];
            apply_window_func(
                &func, &sorted, &order_vals, None, WindowFrame::Cumulative, &mut boxed,
            )
            .unwrap();

            let boxed_i64: Vec<i64> = boxed
                .iter()
                .map(|v| match v {
                    ScalarValue::Int64(Some(x)) => *x,
                    other => panic!("{func:?} produced {other:?}"),
                })
                .collect();
            assert_eq!(dense, boxed_i64, "{func:?} ranked differently");
        }
    }

    /// The dense partition sort must order rows identically to the typed one.
    ///
    /// Three sorts of the same thing now exist — dense, typed-from-ScalarValue, and
    /// the comparator — and each was added because the one before it was too slow.
    /// They have to agree, or a window would start numbering rows differently
    /// depending on which path a column happened to take.
    #[test]
    fn dense_and_typed_partition_sorts_agree() {
        for desc in [false, true] {
            let raw: Vec<f64> = (0..3_000).map(|i| ((i * 7919) % 3_000) as f64 - 1_500.0).collect();
            let boxed: Vec<ScalarValue> =
                raw.iter().map(|&v| ScalarValue::Float64(Some(v))).collect();
            let order_by = vec![if desc {
                SortExpr::desc(Expr::Column("v".into()))
            } else {
                SortExpr::asc(Expr::Column("v".into()))
            }];

            let mut from_dense: Vec<usize> = (0..raw.len()).collect();
            assert!(try_dense_partition_sort(Some(&raw), &order_by, &mut from_dense));

            let mut from_typed: Vec<usize> = (0..raw.len()).collect();
            assert!(try_typed_partition_sort(&[boxed], &order_by, &mut from_typed));

            assert_eq!(from_dense, from_typed, "desc={desc}: the two sorts disagreed");
        }
    }

    /// Anything the typed path cannot order must fall back, not guess.
    #[test]
    fn typed_partition_sort_declines_what_it_cannot_order() {
        let order_by = vec![SortExpr::asc(Expr::Column("v".into()))];
        let mut idx = vec![0usize, 1];

        // Text has no numeric key here.
        let text = vec![vec![
            ScalarValue::Utf8(Some("a".into())),
            ScalarValue::Utf8(Some("b".into())),
        ]];
        assert!(!try_typed_partition_sort(&text, &order_by, &mut idx));

        // A NULL has no total-order key either.
        let with_null = vec![vec![ScalarValue::Float64(Some(1.0)), ScalarValue::Null]];
        assert!(!try_typed_partition_sort(&with_null, &order_by, &mut idx));

        // Two ORDER BY keys are outside its remit.
        let two = vec![
            vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(2))],
            vec![ScalarValue::Int64(Some(3)), ScalarValue::Int64(Some(4))],
        ];
        let two_exprs = vec![
            SortExpr::asc(Expr::Column("a".into())),
            SortExpr::asc(Expr::Column("b".into())),
        ];
        assert!(!try_typed_partition_sort(&two, &two_exprs, &mut idx));
    }

    fn make_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::required("dept",   DataType::Utf8),
            Field::required("salary", DataType::Float64),
        ]);
        RecordBatch::new(schema, vec![
            ColumnBuilder::utf8_from_str(vec![
                Some("eng"), Some("eng"), Some("hr"), Some("hr"), Some("eng"),
            ]),
            ColumnBuilder::float64(vec![
                Some(100.0), Some(120.0), Some(80.0), Some(90.0), Some(110.0),
            ]),
        ]).unwrap()
    }

    #[test]
    fn row_number_partition_by_dept() {
        let batch = make_batch();
        let expr = WindowExpr {
            func:         WindowFunc::RowNumber,
            input:        None,
            partition_by: vec![Expr::Column("dept".into())],
            order_by:     vec![SortExpr::desc(Expr::Column("salary".into()))],
            frame:        WindowFrame::Cumulative,
            alias:        "rn".into(),
        };
        let col = expr.execute(&batch).unwrap();
        // eng rows: salaries 100, 120, 110 → sorted desc → 120(rn=1), 110(rn=2), 100(rn=3)
        // eng is at batch rows 0, 1, 4 → values 100, 120, 110
        // after sort desc: row1(120)→rn=1, row4(110)→rn=2, row0(100)→rn=3
        assert_eq!(col.get(0).unwrap(), ScalarValue::Int64(Some(3))); // 100 → rank 3
        assert_eq!(col.get(1).unwrap(), ScalarValue::Int64(Some(1))); // 120 → rank 1
        assert_eq!(col.get(4).unwrap(), ScalarValue::Int64(Some(2))); // 110 → rank 2
    }

    #[test]
    fn rank_with_ties() {
        let schema = Schema::new(vec![
            Field::required("score", DataType::Float64),
        ]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::float64(vec![Some(10.0), Some(10.0), Some(8.0)]),
        ]).unwrap();
        let expr = WindowExpr {
            func:         WindowFunc::Rank,
            input:        None,
            partition_by: vec![],
            order_by:     vec![SortExpr::desc(Expr::Column("score".into()))],
            frame:        WindowFrame::Cumulative,
            alias:        "r".into(),
        };
        let col = expr.execute(&batch).unwrap();
        // 10, 10 → rank 1, 1; then 8 → rank 3
        assert_eq!(col.get(0).unwrap(), ScalarValue::Int64(Some(1)));
        assert_eq!(col.get(1).unwrap(), ScalarValue::Int64(Some(1)));
        assert_eq!(col.get(2).unwrap(), ScalarValue::Int64(Some(3)));
    }

    #[test]
    fn dense_rank_with_ties() {
        let schema = Schema::new(vec![Field::required("score", DataType::Float64)]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::float64(vec![Some(10.0), Some(10.0), Some(8.0)]),
        ]).unwrap();
        let expr = WindowExpr {
            func:         WindowFunc::DenseRank,
            input:        None,
            partition_by: vec![],
            order_by:     vec![SortExpr::desc(Expr::Column("score".into()))],
            frame:        WindowFrame::Cumulative,
            alias:        "dr".into(),
        };
        let col = expr.execute(&batch).unwrap();
        assert_eq!(col.get(0).unwrap(), ScalarValue::Int64(Some(1)));
        assert_eq!(col.get(1).unwrap(), ScalarValue::Int64(Some(1)));
        assert_eq!(col.get(2).unwrap(), ScalarValue::Int64(Some(2)));
    }

    #[test]
    fn lag_basic() {
        let schema = Schema::new(vec![Field::required("v", DataType::Float64)]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::float64(vec![Some(1.0), Some(2.0), Some(3.0)]),
        ]).unwrap();
        let expr = WindowExpr {
            func:         WindowFunc::Lag(1),
            input:        Some(Expr::Column("v".into())),
            partition_by: vec![],
            order_by:     vec![SortExpr::asc(Expr::Column("v".into()))],
            frame:        WindowFrame::Cumulative,
            alias:        "lag_v".into(),
        };
        let col = expr.execute(&batch).unwrap();
        assert!(col.get(0).unwrap().is_null()); // first row has no previous
        assert_eq!(col.get(1).unwrap(), ScalarValue::Float64(Some(1.0)));
        assert_eq!(col.get(2).unwrap(), ScalarValue::Float64(Some(2.0)));
    }

    #[test]
    fn running_sum() {
        let schema = Schema::new(vec![Field::required("v", DataType::Float64)]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::float64(vec![Some(1.0), Some(2.0), Some(3.0)]),
        ]).unwrap();
        let expr = WindowExpr {
            func:         WindowFunc::Sum,
            input:        Some(Expr::Column("v".into())),
            partition_by: vec![],
            order_by:     vec![SortExpr::asc(Expr::Column("v".into()))],
            frame:        WindowFrame::Cumulative,
            alias:        "running_sum".into(),
        };
        let col = expr.execute(&batch).unwrap();
        assert_eq!(col.get(0).unwrap(), ScalarValue::Float64(Some(1.0)));
        assert_eq!(col.get(1).unwrap(), ScalarValue::Float64(Some(3.0)));
        assert_eq!(col.get(2).unwrap(), ScalarValue::Float64(Some(6.0)));
    }
}
