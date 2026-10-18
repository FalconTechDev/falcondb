use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::storage::{
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
        return gather_narrow(&merged, &indices);
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
/// Deliberately serial. Gathering the columns in parallel looks like free
/// parallelism — each column is an independent pass — but measuring it (see
/// tests/diag_sort.rs) showed it losing on three of four shapes, and by 2x on a
/// 10k-row sort:
///
///   rows / batches      parallel     serial
///   10k  / 10            0.958 ms    0.472 ms
///   100k / 98            5.554 ms    4.199 ms
///   10k  / 1             0.825 ms    0.301 ms
///   100k / 1             2.877 ms    2.963 ms
///
/// Two reasons. A typical result has a handful of columns, so rayon is asked to
/// schedule across a three-element slice and the hand-off costs more than the
/// work. And a gather is random access bound by memory latency rather than by
/// CPU, so extra threads contend for the same bus instead of adding throughput.
///
/// A very wide result might flip that, but nothing here measured one, and a
/// threshold chosen on a guess is how the regression got in.
fn gather_narrow(batch: &RecordBatch, indices: &[u32]) -> FalconResult<RecordBatch> {
    let cols: FalconResult<Vec<_>> = batch.columns.iter()
        .map(|col| col.gather_u32(indices))
        .collect();
    RecordBatch::new(batch.schema.clone(), cols?)
}

fn gather(batch: &RecordBatch, indices: &[usize]) -> FalconResult<RecordBatch> {
    let cols: FalconResult<Vec<_>> = batch.columns.iter()
        .map(|col| col.filter_by_indices(indices))
        .collect();
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
) -> FalconResult<Option<Vec<u32>>> {
    use crate::storage::column::{Column, Utf8Column};

    // Row numbers are `u32` throughout the typed path. A batch with more rows
    // than that cannot be addressed by it, so hand those to the general path.
    if batch.row_count() >= u32::MAX as usize {
        return Ok(None);
    }

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
            // A column with no NULLs asks `is_valid` nothing: the branch is the
            // only thing standing between this loop and a vectorised one, and it
            // can never be taken. Same shape of waste that `Column::gather_u32`
            // was carrying until it got its dense path.
            if c.has_nulls() {
                for i in 0..c.len() {
                    if c.is_valid(i) {
                        pairs.push((dense[i], i as u32));
                    } else {
                        nulls.push(i);
                    }
                }
            } else {
                pairs.extend(dense.iter().enumerate().map(|(i, &v)| (v, i as u32)));
            }
            sort_pairs(&mut pairs, desc);
            Some(assemble(pairs, nulls, s.nulls_last, limit))
        }};
    }

    // Float64/Float32: convert to integer sort key for branchless Ord comparison.
    // This eliminates the partial_cmp NaN branch on every comparison in the sort.
    macro_rules! float_typed {
        ($c:expr, $float:ty, $uint:ty, $key_fn:expr) => {{
            let c = $c;
            let dense = c.dense();
            // A small LIMIT never needs a pair per row. See `top_k_ascending`.
            if let Some(k) = top_k_limit(limit, dense.len()) {
                let mut sink = NullSink::new(k);
                let pairs = if c.has_nulls() {
                    top_k_ascending(
                        (0..c.len()).filter_map(|i| {
                            if c.is_valid(i) {
                                let key = $key_fn(dense[i]);
                                Some((if desc { !key } else { key }, i as u32))
                            } else {
                                sink.push(i);
                                None
                            }
                        }),
                        k,
                    )
                } else {
                    top_k_ascending(
                        dense.iter().enumerate().map(|(i, &v)| {
                            let key = $key_fn(v);
                            (if desc { !key } else { key }, i as u32)
                        }),
                        k,
                    )
                };
                Some(assemble(pairs, sink.rows, s.nulls_last, limit))
            } else {
            let mut pairs: Vec<($uint, u32)> = Vec::with_capacity(dense.len());
            let mut nulls: Vec<usize> = Vec::new();
            // See the note in `typed!`: no NULLs means the validity branch is
            // dead weight on every row.
            if c.has_nulls() {
                for i in 0..c.len() {
                    if c.is_valid(i) {
                        let key = $key_fn(dense[i]);
                        // Flip all bits for DESC so we can always sort ascending.
                        pairs.push((if desc { !key } else { key }, i as u32));
                    } else {
                        nulls.push(i);
                    }
                }
            } else {
                pairs.extend(dense.iter().enumerate().map(|(i, &v)| {
                    let key = $key_fn(v);
                    (if desc { !key } else { key }, i as u32)
                }));
            }
            sort_pairs_ord_limited(&mut pairs, limit, nulls.len(), s.nulls_last);
            Some(assemble(pairs, nulls, s.nulls_last, limit))
            }
        }};
    }

    let out = match col {
        Column::Int64(c)     => typed!(c, i64),
        Column::Int32(c)     => typed!(c, i32),
        Column::Float64(c)   => float_typed!(c, f64, u64, f64_ord_key),
        Column::Float32(c)   => float_typed!(c, f32, u32, f32_ord_key),
        Column::Date32(c)    => typed!(c, i32),
        Column::Timestamp(c) => typed!(c, i64),

        // A dictionary column sorts by rank, not by string.
        //
        // Without this, ordering by a text column fell to the generic path: a
        // `ScalarValue` per row -- a heap-allocated `String` each -- and then
        // string comparisons all the way through the sort. Measured over 100k
        // rows, `ORDER BY dept DESC` took 11.45ms against 1.89ms for the same
        // sort on a `Float64` column, so 6.1x.
        //
        // The ordering of the codes only has to be decided once, and the
        // dictionary holds distinct values rather than rows -- five entries for
        // the column above, against a hundred thousand rows. After that this is
        // the integer sort the numeric arms already do, on pairs that are half
        // the size of the float ones: `(u32, u32)` is eight bytes where
        // `(u64, u32)` is sixteen.
        //
        // Worst case is a column where every value is distinct, and then sorting
        // the dictionary is the same work as sorting the rows would have been --
        // but done once, on strings that are already deduplicated, rather than
        // repeatedly inside a comparison sort.
        Column::Utf8(c) => {
            let dict = c.dict();
            let mut order: Vec<u32> = (0..dict.len() as u32).collect();
            order.sort_unstable_by(|&a, &b| dict[a as usize].cmp(&dict[b as usize]));
            let mut rank = vec![0u32; dict.len()];
            for (position, &code) in order.iter().enumerate() {
                rank[code as usize] = position as u32;
            }

            let codes = c.codes();
            // As in the float arms: a small LIMIT is answered by a bounded heap
            // over the codes, with no pair materialised for the rows it rejects.
            if let Some(k) = top_k_limit(limit, codes.len()) {
                let mut sink = NullSink::new(k);
                let key = |r: u32| if desc { !r } else { r };
                let pairs = if c.has_nulls() {
                    top_k_ascending(
                        codes.iter().enumerate().filter_map(|(row, &code)| {
                            if Utf8Column::is_null_code(code) {
                                sink.push(row);
                                None
                            } else {
                                Some((key(rank[code as usize]), row as u32))
                            }
                        }),
                        k,
                    )
                } else {
                    top_k_ascending(
                        codes.iter().enumerate().map(|(row, &code)| {
                            (key(rank[code as usize]), row as u32)
                        }),
                        k,
                    )
                };
                Some(assemble(pairs, sink.rows, s.nulls_last, limit))
            } else {
            let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(codes.len());
            let mut nulls: Vec<usize> = Vec::new();
            // Flip the rank's bits for DESC so the sort is always ascending,
            // exactly as the float arms do with their ordering keys.
            let key = |r: u32| if desc { !r } else { r };
            if c.has_nulls() {
                for (row, &code) in codes.iter().enumerate() {
                    if Utf8Column::is_null_code(code) {
                        nulls.push(row);
                    } else {
                        pairs.push((key(rank[code as usize]), row as u32));
                    }
                }
            } else {
                pairs.extend(
                    codes.iter().enumerate().map(|(row, &code)| {
                        (key(rank[code as usize]), row as u32)
                    }),
                );
            }
            sort_pairs_ord_limited(&mut pairs, limit, nulls.len(), s.nulls_last);
            Some(assemble(pairs, nulls, s.nulls_last, limit))
            }
        }
        // Utf8 keys would mean cloning or borrowing strings per row; Decimal128
        // and Boolean are rare enough as sort keys not to be worth a second path.
        _ => None,
    };
    Ok(out)
}

/// IEEE 754 → total-order u64: maps all f64 values to u64 preserving sort order.
/// Negatives flip all bits; positives flip only the sign bit.
/// NaN sorts above +∞ (any representation).
#[inline(always)]
fn f64_ord_key(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits >> 63 == 0 { bits ^ (1u64 << 63) } else { !bits }
}

#[inline(always)]
fn f32_ord_key(v: f32) -> u32 {
    let bits = v.to_bits();
    if bits >> 31 == 0 { bits ^ (1u32 << 31) } else { !bits }
}

/// Sort `(key, row)` pairs by the unsigned key, using the radix sort above.
///
/// Exposed so `query::window` can reuse this instead of growing its own sort. Its
/// ORDER BY ran through `cmp_scalar` on boxed `ScalarValue`s — a dynamic enum match
/// per comparison, about 1.7M of them at 100k rows, which is why a window cost 15.9 ms
/// where `execute_sort` on the same column cost 2.67 ms.
pub(crate) fn sort_u64_pairs(pairs: &mut Vec<(u64, u32)>) {
    sort_pairs_ord(pairs);
}

/// Total-order `u64` key for an `f64`, for callers building their own pairs.
pub(crate) fn f64_sort_key(v: f64) -> u64 {
    f64_ord_key(v)
}

/// Unsigned key that an LSD radix sort can decompose into 8-bit digits.
#[cfg(test)]
trait RadixKey: Ord + Copy + Send + Default {
    const PASSES: usize;
    fn digit(self, pass: usize) -> usize;
}

#[cfg(test)]
impl RadixKey for u64 {
    const PASSES: usize = 8;
    #[inline(always)]
    fn digit(self, pass: usize) -> usize { ((self >> (pass * 8)) & 0xFF) as usize }
}

#[cfg(test)]
impl RadixKey for u32 {
    const PASSES: usize = 4;
    #[inline(always)]
    fn digit(self, pass: usize) -> usize { ((self >> (pass * 8)) & 0xFF) as usize }
}

/// Below this, the radix scratch allocation costs more than pdqsort's comparisons.
#[cfg(test)]
const RADIX_MIN_ROWS: usize = 8_192;

/// Sort (Ord-key, row) pairs ascending — branchless, no closure, no NaN special case.
/// Minimum pairs before the sort is worth handing to rayon.
///
/// Measured on scattered `f64` keys: parallel is 1.34x serial at 8k rows, 0.80x
/// at 16k, 0.34x at 100k. The crossover sits between 8k and 16k, so 12k, divided
/// by the thread count so the constant travels to a machine with a different one.
fn par_sort_min_rows() -> usize {
    crate::query::tuning::par_min_rows(600)
}

/// Sort (key, row) pairs.
///
/// Sorts the whole tuple, so equal keys fall back to the row index and the result
/// is the same order a stable sort would give. That is why an unstable sort is
/// safe here, and why the radix sort it replaced could be dropped without
/// changing any query's output -- `radix_against_pdqsort` asserts the two agree.
///
/// This used to be an LSD radix sort above 8192 rows. Measured against the
/// standard library's sort on 100k pairs, across three distributions:
///
/// | keys                        | radix  | sort_unstable | par_sort_unstable |
/// |-----------------------------|--------|---------------|-------------------|
/// | 100 distinct (the benchmark)| 1.58ms | 2.31ms        | **1.14ms**        |
/// | all distinct, ordered       | 2.47ms | **133us**     | 153us             |
/// | all distinct, scattered     | 3.26ms | 2.35ms        | **864us**         |
///
/// The radix sort lost on every one of them. It looked defensible only on the
/// benchmark's column, which holds a hundred distinct values -- and even there it
/// was beaten. On already-ordered input it is 18x slower, because pdqsort detects
/// sortedness and a radix sort makes all eight passes regardless; ordered and
/// nearly-ordered columns are ordinary in analytical data, not a corner case.

/// The `k` smallest `(key, row)` pairs, in order, taken without building a pair
/// per row.
///
/// [`sort_pairs_ord_limited`] already refuses to *sort* more than the limit
/// needs, but it operates on a vector that was built with one entry per row.
/// That is the cost which survived it: at 100M rows a `(u64, u32)` pair is
/// sixteen bytes, so `ORDER BY score DESC LIMIT 10` allocated and wrote 1.6 GB
/// in order to return ten rows. The commit that added top-N selection called
/// removing that "a different change"; this is it.
///
/// A bounded max-heap keeps `k` pairs and touches every row exactly once. Its
/// root is the worst of the `k` kept so far, so a row is rejected by a single
/// comparison -- and for ten rows out of a hundred million that comparison fails
/// almost every time, which is why this is close to the cost of a bare scan.
///
/// Keys arrive already normalised so that *ascending* is always what is wanted:
/// the float and dictionary arms flip the bits of their ordering key for DESC,
/// exactly as they do before handing pairs to the radix sort.
///
/// Ties at the boundary are arbitrary, as they already were -- these sorts are
/// unstable and the engine has never promised a particular tied row.
fn top_k_ascending<T, I>(rows: I, k: usize) -> Vec<(T, u32)>
where
    T: Ord + Copy,
    I: Iterator<Item = (T, u32)>,
{
    let mut heap: BinaryHeap<(T, u32)> = BinaryHeap::with_capacity(k + 1);
    for pair in rows {
        if heap.len() < k {
            heap.push(pair);
        } else if let Some(worst) = heap.peek() {
            if pair < *worst {
                heap.pop();
                heap.push(pair);
            }
        }
    }
    let mut out = heap.into_vec();
    out.sort_unstable();
    out
}

/// Whether the bounded scan is worth taking instead of building every pair.
///
/// Two conditions, both guesses rather than measured crossovers, and said so
/// here so nobody reads them as findings:
///
///   - `k` small in absolute terms. A heap of a hundred thousand entries is no
///     longer a cache-resident structure, and the comparison that rejects a row
///     stops being free.
///   - `k` a small fraction of `n`. When the limit approaches the row count the
///     heap does the whole sort through its slowest possible interface.
///
/// The shapes that matter -- ten rows out of ten or a hundred million -- are
/// clear of both by several orders of magnitude.
fn top_k_limit(limit: Option<usize>, rows: usize) -> Option<usize> {
    let k = limit?;
    if k == 0 || k > 4096 || k.saturating_mul(64) > rows {
        return None;
    }
    Some(k)
}

/// Collect at most `cap` NULL row numbers.
///
/// Only `cap` of them can ever be emitted: `assemble` places the NULL rows
/// before or after the keyed ones and then truncates to the limit, so a NULL
/// beyond the `k`th is unreachable whichever way `NULLS FIRST` falls.
struct NullSink {
    rows: Vec<usize>,
    cap:  usize,
}

impl NullSink {
    fn new(cap: usize) -> Self {
        Self { rows: Vec::new(), cap }
    }

    #[inline]
    fn push(&mut self, row: usize) {
        if self.rows.len() < self.cap {
            self.rows.push(row);
        }
    }
}

/// Sort `(key, row)` pairs, keeping only as many as the `LIMIT` can use.
///
/// `ORDER BY ... LIMIT 10` used to sort every row and then throw away all but
/// ten: `assemble` truncated the finished list. Over 100k rows that measured
/// 1.70ms against 2.33ms for the same sort with no limit -- 0.73x, where the
/// work is `k` rows out of `n` and the ratio should be nowhere near one.
///
/// `select_nth_unstable` partitions in O(n) so the smallest `k` sit at the front,
/// and then only those `k` are sorted. Everything past them is never ordered at
/// all.
///
/// How many pairs the limit can use depends on where the NULLs go, because
/// `assemble` emits them before or after the keyed rows: with NULLs first they
/// consume part of the limit, with NULLs last they cannot.
///
/// Only applied when `k` is a small fraction of `n`. Partitioning and then
/// sorting is two passes, so for a limit close to the row count it is slower than
/// sorting once.
///
/// Ties at the boundary: which of several equal keys lands in the top `k` is
/// arbitrary here, as it already was -- these sorts are unstable, so the engine
/// has never promised a particular tied row. The *set* can differ from what a
/// full sort would have truncated to, which is why this is said out loud rather
/// than left for someone to discover.
fn sort_pairs_ord_limited<T: Ord + Copy + Send>(
    pairs:      &mut Vec<(T, u32)>,
    limit:      Option<usize>,
    nulls:      usize,
    nulls_last: bool,
) {
    if let Some(k) = limit {
        let needed = if nulls_last { k } else { k.saturating_sub(nulls) };
        if needed == 0 {
            pairs.clear();
            return;
        }
        // 8x is a guess at where two passes stop being cheaper than one, not a
        // measured crossover; it is well clear of the shapes that matter
        // (ten rows out of a hundred thousand).
        if needed < pairs.len() && needed.saturating_mul(8) <= pairs.len() {
            pairs.select_nth_unstable(needed - 1);
            pairs.truncate(needed);
        }
    }
    sort_pairs_ord(pairs);
}

fn sort_pairs_ord<T: Ord + Copy + Send>(pairs: &mut [(T, u32)]) {
    if pairs.len() >= par_sort_min_rows() {
        pairs.par_sort_unstable();
    } else {
        pairs.sort_unstable();
    }
}

/// LSD radix sort, 8 bits per pass, stable.
///
/// All pass histograms are built in a single scan so that passes whose digit is
/// constant across every key can be skipped — for real float columns the high
/// exponent bytes rarely vary, which typically removes half the passes.
#[cfg(test)]
fn radix_sort_pairs<T: RadixKey>(pairs: &mut Vec<(T, u32)>) {
    let n = pairs.len();
    let mut counts = vec![[0usize; 256]; T::PASSES];
    for &(k, _) in pairs.iter() {
        for pass in 0..T::PASSES {
            counts[pass][k.digit(pass)] += 1;
        }
    }

    let active: Vec<usize> = (0..T::PASSES)
        .filter(|&pass| counts[pass].iter().filter(|&&c| c != 0).count() > 1)
        .collect();
    if active.is_empty() {
        return;
    }

    let mut buf: Vec<(T, u32)> = vec![(T::default(), 0u32); n];
    let mut in_buf = false;
    for &pass in &active {
        let mut offsets = [0usize; 256];
        let mut acc = 0usize;
        for d in 0..256 {
            offsets[d] = acc;
            acc += counts[pass][d];
        }
        let (src, dst) = if in_buf {
            (&mut buf[..], &mut pairs[..])
        } else {
            (&mut pairs[..], &mut buf[..])
        };
        for &entry in src.iter() {
            let d = entry.0.digit(pass);
            dst[offsets[d]] = entry;
            offsets[d] += 1;
        }
        in_buf = !in_buf;
    }
    if in_buf {
        pairs.copy_from_slice(&buf);
    }
}

/// Sort (key, row) pairs for integer types (PartialOrd == Ord for non-float).
fn sort_pairs<T: PartialOrd + Copy + Send>(pairs: &mut Vec<(T, u32)>, desc: bool) {
    let cmp = |a: &(T, u32), b: &(T, u32)| {
        let ord = a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal);
        if desc { ord.reverse() } else { ord }
    };
    if pairs.len() >= par_sort_min_rows() {
        pairs.par_sort_unstable_by(cmp);
    } else {
        pairs.sort_unstable_by(cmp);
    }
}

/// Splice the sorted non-NULL rows together with the NULL rows and apply `limit`.
/// Splice the sorted non-NULL rows together with the NULL rows and apply `limit`.
///
/// Row numbers stay `u32`, the width the sort already worked in. Widening them
/// to `usize` here cost a pass over every index and handed the gather an array
/// twice the size to walk alongside the columns it reads -- 800 KB instead of
/// 400 at 100k rows.
///
/// That only became worth doing once `Column::gather_u32` grew the dense fast
/// path its `usize` twin always had. Measured before that fix, narrow indices
/// looked 1.96x *slower*; measured after, the gather is 0.88x and dropping the
/// widening pass takes the pair from 464us to 393us.
fn assemble<T>(
    pairs:      Vec<(T, u32)>,
    nulls:      Vec<usize>,
    nulls_last: bool,
    limit:      Option<usize>,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(pairs.len() + nulls.len());
    if !nulls_last {
        out.extend(nulls.iter().map(|&i| i as u32));
    }
    out.extend(pairs.into_iter().map(|(_, i)| i));
    if nulls_last {
        out.extend(nulls.iter().map(|&i| i as u32));
    }
    if let Some(k) = limit {
        out.truncate(k);
    }
    out
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Concatenate the batches to be sorted.
///
/// Delegates to [`RecordBatch::concat`]. This used to be a private copy of it
/// that walked the columns one at a time, which meant it never saw the parallel
/// path that lives there -- so correcting `par_concat_rows` had no effect on a
/// sort until this was pointed at the shared implementation. The same shape of
/// mistake cost this engine a 100x join regression once: two implementations of
/// one operation, only one of which gets fixed.
fn merge_batches(batches: &[RecordBatch]) -> FalconResult<RecordBatch> {
    RecordBatch::concat(batches)
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
mod top_n {
    use crate::db::Database;
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::RecordBatch,
        types::{DataType, ScalarValue},
    };

    fn db_with(ints: Vec<Option<i64>>) -> Database {
        let batch = RecordBatch::new(
            Schema::new(vec![Field::nullable("n", DataType::Int64)]),
            vec![ColumnBuilder::int64(ints)],
        )
        .unwrap();
        let db = Database::in_memory();
        db.create_table_from_batches("t", vec![batch]).unwrap();
        db
    }

    fn vals(db: &Database, sql: &str) -> Vec<Option<i64>> {
        let r = db.sql(sql).unwrap();
        let mut out = Vec::new();
        for b in &r.batches {
            for i in 0..b.row_count() {
                out.push(match b.column(0).get(i).unwrap() {
                    ScalarValue::Null => None,
                    v => v.as_i64(),
                });
            }
        }
        out
    }

    /// `ORDER BY ... LIMIT k` must return the same rows the full sort would, now
    /// that only `k` of them are sorted.
    ///
    /// Partitioning before sorting is the part that could go wrong: take the wrong
    /// `k`, mishandle the boundary, or lose the NULL placement.
    #[test]
    fn limited_sort_matches_the_full_sort() {
        // Deliberately not in order, and large enough that the partition path
        // engages rather than being skipped as too small a saving.
        let n = 10_000i64;
        let db = db_with((0..n).map(|i| Some((i * 7919) % n)).collect());

        for k in [1usize, 2, 10, 100, 1000] {
            let got = vals(&db, &format!("SELECT n FROM t ORDER BY n LIMIT {k}"));
            let want: Vec<Option<i64>> = (0..k as i64).map(Some).collect();
            assert_eq!(got, want, "ascending, LIMIT {k}");

            let got = vals(&db, &format!("SELECT n FROM t ORDER BY n DESC LIMIT {k}"));
            let want: Vec<Option<i64>> = (0..k as i64).map(|i| Some(n - 1 - i)).collect();
            assert_eq!(got, want, "descending, LIMIT {k}");
        }

        // A limit at and past the row count must not lose rows.
        assert_eq!(vals(&db, "SELECT n FROM t ORDER BY n LIMIT 10000").len(), 10_000);
        assert_eq!(vals(&db, "SELECT n FROM t ORDER BY n LIMIT 20000").len(), 10_000);
        assert_eq!(vals(&db, "SELECT n FROM t ORDER BY n LIMIT 0").len(), 0);
    }

    /// NULLs are emitted before or after the keyed rows, so they consume part of
    /// the limit in one case and not the other. Getting that backwards would
    /// return too few rows, or the wrong ones.
    #[test]
    fn limited_sort_places_nulls_correctly() {
        let mut v: Vec<Option<i64>> = (0..200).map(|i| Some(200 - i)).collect();
        for i in (0..v.len()).step_by(20) {
            v[i] = None;
        }
        let nulls = v.iter().filter(|x| x.is_none()).count();
        let db = db_with(v);

        // Default placement, whichever it is, must agree with the unlimited sort.
        for k in [1usize, 5, 12, 50] {
            let limited = vals(&db, &format!("SELECT n FROM t ORDER BY n LIMIT {k}"));
            let full: Vec<Option<i64>> = vals(&db, "SELECT n FROM t ORDER BY n")
                .into_iter()
                .take(k)
                .collect();
            assert_eq!(limited, full, "LIMIT {k} must be a prefix of the full sort");
        }
        assert!(nulls > 0, "the fixture must actually contain NULLs");
    }

    /// With every key equal, the top `k` is arbitrary but must still be `k` rows
    /// of the right value -- the case where a partition could drop rows.
    #[test]
    fn limited_sort_handles_all_ties() {
        let db = db_with(vec![Some(42); 5_000]);
        for k in [1usize, 3, 100, 4999, 5000] {
            let got = vals(&db, &format!("SELECT n FROM t ORDER BY n LIMIT {k}"));
            assert_eq!(got.len(), k, "LIMIT {k} over all-equal keys");
            assert!(got.iter().all(|v| *v == Some(42)));
        }
    }
}

mod utf8_sort {
    use super::*;
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        types::DataType,
    };

    fn batch(values: Vec<Option<&str>>) -> RecordBatch {
        RecordBatch::new(
            Schema::new(vec![Field::nullable("s", DataType::Utf8)]),
            vec![ColumnBuilder::utf8_from_str(values)],
        )
        .unwrap()
    }

    fn sorted(values: Vec<Option<&str>>, by: SortExpr) -> Vec<Option<String>> {
        let b = batch(values);
        let out = execute_sort(std::slice::from_ref(&b), &[by]).unwrap();
        let col = out.column(0);
        (0..out.row_count())
            .map(|r| match col.get(r).unwrap() {
                ScalarValue::Utf8(s) => s,
                ScalarValue::Null => None,
                other => panic!("unexpected {other:?}"),
            })
            .collect()
    }

    /// Sorting a dictionary column by rank must order it by string.
    ///
    /// The ranks come from sorting the dictionary, which is in first-seen order,
    /// so a column whose values appear out of order is the case that catches a
    /// rank table built the wrong way round.
    #[test]
    fn orders_by_string_not_by_dictionary_position() {
        let vals = vec![Some("pear"), Some("apple"), Some("fig"), Some("apple")];
        assert_eq!(
            sorted(vals.clone(), SortExpr::asc(Expr::Column("s".into()))),
            vec![
                Some("apple".to_string()),
                Some("apple".to_string()),
                Some("fig".to_string()),
                Some("pear".to_string()),
            ],
        );
        assert_eq!(
            sorted(vals, SortExpr::desc(Expr::Column("s".into()))),
            vec![
                Some("pear".to_string()),
                Some("fig".to_string()),
                Some("apple".to_string()),
                Some("apple".to_string()),
            ],
        );
    }

    /// NULLs are not in the dictionary and have to be placed, not ranked.
    #[test]
    fn places_nulls_rather_than_ranking_them() {
        let vals = vec![Some("b"), None, Some("a"), None];

        // `asc` puts them first, `asc_nulls_last` at the end.
        assert_eq!(
            sorted(vals.clone(), SortExpr::asc(Expr::Column("s".into()))),
            vec![None, None, Some("a".to_string()), Some("b".to_string())],
        );
        assert_eq!(
            sorted(vals.clone(), SortExpr::asc_nulls_last(Expr::Column("s".into()))),
            vec![Some("a".to_string()), Some("b".to_string()), None, None],
        );
        // `desc` is nulls-last by construction.
        assert_eq!(
            sorted(vals, SortExpr::desc(Expr::Column("s".into()))),
            vec![Some("b".to_string()), Some("a".to_string()), None, None],
        );
    }

    /// Equal strings keep their original row order, as every other type does.
    #[test]
    fn equal_strings_keep_row_order() {
        let b = RecordBatch::new(
            Schema::new(vec![
                Field::required("s", DataType::Utf8),
                Field::required("id", DataType::Int64),
            ]),
            vec![
                ColumnBuilder::utf8_from_str(vec![Some("x"), Some("x"), Some("x"), Some("x")]),
                ColumnBuilder::int64(vec![Some(10), Some(20), Some(30), Some(40)]),
            ],
        )
        .unwrap();
        for by in [
            SortExpr::asc(Expr::Column("s".into())),
            SortExpr::desc(Expr::Column("s".into())),
        ] {
            let out = execute_sort(std::slice::from_ref(&b), &[by]).unwrap();
            let ids: Vec<i64> = (0..out.row_count())
                .map(|r| out.column(1).get(r).unwrap().as_i64().unwrap())
                .collect();
            assert_eq!(ids, vec![10, 20, 30, 40], "ties must not be reordered");
        }
    }

    /// Every value distinct is the worst case for ranking, and still has to be
    /// right: the dictionary is then as large as the column.
    #[test]
    fn handles_an_all_distinct_column() {
        let owned: Vec<String> = (0..64).map(|i| format!("{:03}", (i * 37) % 64)).collect();
        let vals: Vec<Option<&str>> = owned.iter().map(|s| Some(s.as_str())).collect();
        let got = sorted(vals, SortExpr::asc(Expr::Column("s".into())));
        let mut want: Vec<Option<String>> = owned.iter().map(|s| Some(s.clone())).collect();
        want.sort();
        assert_eq!(got, want);
    }
}

#[cfg(test)]
mod perf {
    use super::*;
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        types::DataType,
    };
    use std::time::Instant;

    /// Radix against pdqsort, and both against building the pairs.
    ///
    /// Producing the sorted indices is 75% of a 100k `ORDER BY`. This splits that
    /// into the scan that builds the (key, row) pairs and the sort itself, and
    /// compares the two sorts on identical input. `RADIX_MIN_ROWS` sends anything
    /// over 8192 rows to the radix sort; whether that is the right call at 100k is
    /// a measurement, and it has not been made at this size with this key.
    ///
    /// An LSD radix over a `u64` key is up to eight passes, each reading and
    /// scattering 1.6 MB of pairs. `sort_unstable` is one pass of comparisons over
    /// data that stays in cache. Which wins depends on how many passes the keys
    /// actually need, which is why the skip-constant-digits optimisation exists
    /// and why it has to be measured rather than assumed.
    ///
    /// Run with
    /// `cargo test --release --lib radix_against_pdqsort -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn radix_against_pdqsort() {
        const ROWS: usize = 100_000;

        // Three distributions, because both algorithms are data-sensitive and
        // picking one from a single artificial column would be fitting the engine
        // to the benchmark. Radix skips passes whose digit never varies; pdqsort
        // handles heavy duplication well. They are strongest on opposite inputs.
        for (label, keys) in [
            (
                "100 distinct (the benchmark's column)",
                (0..ROWS).map(|i| !f64_sort_key((i % 100) as f64 + 0.5)).collect::<Vec<u64>>(),
            ),
            (
                "all distinct, already ordered",
                (0..ROWS).map(|i| !f64_sort_key(i as f64 + 0.5)).collect::<Vec<u64>>(),
            ),
            (
                "all distinct, scattered",
                (0..ROWS)
                    .map(|i| {
                        // Cheap deterministic scatter; no rand dependency.
                        let h = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                        !f64_sort_key((h >> 11) as f64 * 1.0e-6)
                    })
                    .collect::<Vec<u64>>(),
            ),
        ] {
        println!("
--- {label} ---");
        let fresh = || -> Vec<(u64, u32)> {
            keys.iter().enumerate().map(|(i, &k)| (k, i as u32)).collect()
        };

        // Both sorts must agree, or the faster one is irrelevant.
        let mut a = fresh();
        let mut b = fresh();
        radix_sort_pairs(&mut a);
        b.sort_unstable();
        assert_eq!(a, b, "radix and pdqsort disagree on the sorted order");

        let time_radix = || {
            let mut v = fresh();
            let t = Instant::now();
            radix_sort_pairs(&mut v);
            let e = t.elapsed();
            std::hint::black_box(&v);
            e
        };
        let time_pdq = || {
            let mut v = fresh();
            let t = Instant::now();
            v.sort_unstable();
            let e = t.elapsed();
            std::hint::black_box(&v);
            e
        };
        let time_build = || {
            let t = Instant::now();
            let v = fresh();
            let e = t.elapsed();
            std::hint::black_box(&v);
            e
        };

        for _ in 0..3 {
            time_radix();
            time_pdq();
            time_build();
        }
        let mut r = Vec::new();
        let mut q = Vec::new();
        let mut bl = Vec::new();
        for _ in 0..9 {
            r.push(time_radix());
            q.push(time_pdq());
            bl.push(time_build());
        }
        let (r, q, bl) = (
            *r.iter().min().unwrap(),
            *q.iter().min().unwrap(),
            *bl.iter().min().unwrap(),
        );

        println!("{ROWS} (u64, u32) pairs, 100 distinct keys:");
        println!("  build the pairs = {bl:?}");
        println!("  radix sort      = {r:?}");
        println!("  sort_unstable   = {q:?}");
        println!("  radix is {:.2}x pdqsort", r.as_secs_f64() / q.as_secs_f64());

        // The radix sort runs on one thread. Twelve threads doing more total work
        // may still finish sooner -- the only way to know is to ask.
        let time_par = || {
            let mut v = fresh();
            let t = Instant::now();
            v.par_sort_unstable();
            let e = t.elapsed();
            std::hint::black_box(&v);
            e
        };
        let mut check = fresh();
        check.par_sort_unstable();
        assert_eq!(check, b, "parallel sort disagrees on the sorted order");
        for _ in 0..3 { time_par(); }
        let pr = (0..9).map(|_| time_par()).min().unwrap();
        println!("  par_sort_unstable = {pr:?}  ({:.2}x the serial radix)",
                 pr.as_secs_f64() / r.as_secs_f64());
        }

        // Where does handing the sort to rayon start paying? A threshold guessed
        // rather than measured is how this file ended up with several that
        // described how the code used to work.
        println!("
--- serial vs parallel crossover (scattered keys) ---");
        for n in [1_000usize, 4_000, 8_000, 16_000, 32_000, 64_000, 100_000] {
            let ks: Vec<u64> = (0..n)
                .map(|i| {
                    let h = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    !f64_sort_key((h >> 11) as f64 * 1.0e-6)
                })
                .collect();
            let mk = || -> Vec<(u64, u32)> {
                ks.iter().enumerate().map(|(i, &k)| (k, i as u32)).collect()
            };
            let ser = || { let mut v = mk(); let t = Instant::now(); v.sort_unstable(); let e = t.elapsed(); std::hint::black_box(&v); e };
            let par = || { let mut v = mk(); let t = Instant::now(); v.par_sort_unstable(); let e = t.elapsed(); std::hint::black_box(&v); e };
            for _ in 0..3 { ser(); par(); }
            let sm = (0..9).map(|_| ser()).min().unwrap();
            let pm = (0..9).map(|_| par()).min().unwrap();
            println!("  {n:>7} rows: serial {sm:>10.2?}  parallel {pm:>10.2?}  ({:.2}x)",
                     pm.as_secs_f64() / sm.as_secs_f64());
        }
    }

    /// What does the validity branch cost when there is nothing to check?
    ///
    /// Building the sort's (key, row) pairs asked `is_valid` on every row, even
    /// for a column that holds no NULLs and cannot answer anything but yes. The
    /// branch is also what stops the loop vectorising.
    ///
    /// Run with
    /// `cargo test --release --lib dense_key_extraction -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn dense_key_extraction() {
        const ROWS: usize = 100_000;
        let col = ColumnBuilder::float64((0..ROWS).map(|i| Some((i % 100) as f64 + 0.5)).collect());
        let c = match &col {
            crate::storage::column::Column::Float64(c) => c,
            _ => unreachable!(),
        };

        // The shape this loop had: ask, branch, push.
        let branching = || {
            let dense = c.dense();
            let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(dense.len());
            let mut nulls: Vec<usize> = Vec::new();
            for i in 0..c.len() {
                if c.is_valid(i) {
                    pairs.push((!f64_sort_key(dense[i]), i as u32));
                } else {
                    nulls.push(i);
                }
            }
            (pairs, nulls)
        };
        // The shape it has now when the column is dense.
        let dense_path = || {
            let dense = c.dense();
            let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(dense.len());
            pairs.extend(
                dense.iter().enumerate().map(|(i, &v)| (!f64_sort_key(v), i as u32)),
            );
            (pairs, Vec::<usize>::new())
        };

        assert_eq!(branching().0, dense_path().0, "the two must build the same pairs");

        let time = |label: &str, f: &dyn Fn() -> (Vec<(u64, u32)>, Vec<usize>)| {
            for _ in 0..3 { std::hint::black_box(f()); }
            let t = std::time::Instant::now();
            for _ in 0..200 { std::hint::black_box(f()); }
            let per = t.elapsed() / 200;
            println!("  {label:>28}: {per:>9.2?}");
            per
        };

        println!("building {ROWS} (key, row) pairs from a column with no NULLs:");
        let b = time("asking is_valid per row", &branching);
        let d = time("dense path", &dense_path);
        println!("  dense is {:.2}x the branching form", d.as_secs_f64() / b.as_secs_f64());
    }

    /// What does `ORDER BY` cost on a column the typed sort does not cover?
    ///
    /// `try_typed_sort` handles `Int64`, `Int32`, `Float64`, `Float32`, `Date32`
    /// and `Timestamp`. `Utf8` is not among them, so sorting by a string column
    /// falls to the generic path: a `ScalarValue` per row -- a heap-allocated
    /// `String` each -- and then string comparisons all the way through the sort.
    ///
    /// The column is dictionary-encoded, so the ordering of its *codes* can be
    /// decided once by sorting the dictionary, which holds distinct values rather
    /// than rows. After that the sort is the same integer sort the numeric path
    /// already does.
    ///
    /// The benchmark suite sorts `score`, a `Float64`, so it never measures this.
    ///
    /// Run with
    /// `cargo test --release --lib order_by_column_type -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn order_by_column_type() {
        use std::time::Instant;
        const ROWS: usize = 100_000;

        let batch = RecordBatch::new(
            Schema::new(vec![
                Field::required("dept", DataType::Utf8),
                Field::required("score", DataType::Float64),
            ]),
            vec![
                ColumnBuilder::utf8_from_str(
                    (0..ROWS)
                        .map(|i| Some(["eng", "sales", "hr", "ops", "legal"][i % 5]))
                        .collect(),
                ),
                ColumnBuilder::float64((0..ROWS).map(|i| Some((i % 100) as f64)).collect()),
            ],
        )
        .unwrap();

        let time = |label: &str, column: &str| {
            let exprs = vec![SortExpr::desc(Expr::Column(column.into()))];
            let run = || {
                let out = execute_sort(std::slice::from_ref(&batch), &exprs).unwrap();
                assert_eq!(out.row_count(), ROWS);
            };
            for _ in 0..3 { run(); }
            let mut best = std::time::Duration::MAX;
            for _ in 0..9 {
                let t = Instant::now();
                run();
                best = best.min(t.elapsed());
            }
            println!("  {label:>32}: {best:>9.2?}");
            best
        };

        println!("ORDER BY DESC over {ROWS} rows:");
        let f = time("score (Float64, typed sort)", "score");
        let u = time("dept (Utf8, ScalarValue path)", "dept");
        println!("  the string column costs {:.1}x the float one", u.as_secs_f64() / f.as_secs_f64());
    }

    /// Is the pre-sort merge worth spreading over cores at this size?
    ///
    /// `execute_sort` concatenates its input into one batch before sorting, and
    /// that merge is 22-25% of a 100k-row ORDER BY even with only two batches --
    /// a full copy of every column before any comparison happens.
    ///
    /// `RecordBatch::concat` can copy columns in parallel, but only above
    /// `par_concat_rows()`, which is 262k on this machine. At 100k it copies three
    /// columns one after another. Three columns is three tasks, which is not much
    /// parallelism, but it is free if it wins.
    ///
    /// Run with
    /// `cargo test --release --lib merge_parallel_columns -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn merge_parallel_columns() {
        use rayon::prelude::*;
        use std::time::Instant;
        const BATCH: usize = 65_536;
        for &ROWS in &[5_000usize, 20_000, 50_000, 100_000, 300_000] {

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
        // 5k fits one batch; the rest take two or more.

        let n_cols = schema.fields.len();
        let concat_col = |ci: usize| {
            let parts: Vec<&crate::storage::column::Column> =
                batches.iter().map(|b| b.column(ci)).collect();
            crate::storage::column::Column::concat(&parts).unwrap()
        };

        let serial = || {
            let t = Instant::now();
            let cols: Vec<_> = (0..n_cols).map(concat_col).collect();
            let e = t.elapsed();
            std::hint::black_box(&cols);
            e
        };
        let parallel = || {
            let t = Instant::now();
            let cols: Vec<_> = (0..n_cols).into_par_iter().map(concat_col).collect();
            let e = t.elapsed();
            std::hint::black_box(&cols);
            e
        };

        // Same columns either way.
        let a = (0..n_cols).map(concat_col).collect::<Vec<_>>();
        let b: Vec<_> = (0..n_cols).into_par_iter().map(concat_col).collect();
        for c in 0..n_cols {
            assert_eq!(a[c].len(), b[c].len());
            for r in (0..a[c].len()).step_by(4999) {
                assert_eq!(a[c].get(r).unwrap(), b[c].get(r).unwrap(), "column {c} row {r}");
            }
        }

        for _ in 0..3 { serial(); parallel(); }
        let s = (0..15).map(|_| serial()).min().unwrap();
        let p = (0..15).map(|_| parallel()).min().unwrap();
        println!(
            "  {ROWS:>7} rows, {} batches: serial {s:>9.2?}  parallel {p:>9.2?}  {:.2}x",
            batches.len(),
            p.as_secs_f64() / s.as_secs_f64(),
        );
        }
    }

    /// Should the sorted indices stay `u32` all the way to the gather?
    ///
    /// The sort works on `(key, u32)` pairs -- the row number is four bytes,
    /// because a batch cannot have more rows than that. `assemble` then widens
    /// every one to `usize` so `gather` can take `&[usize]`, which costs a pass
    /// over the indices and doubles the array the gather then walks: 800 KB
    /// instead of 400 KB at 100k rows, competing for the same cache as the
    /// columns being gathered.
    ///
    /// `Column::gather_u32` already exists and takes the narrow form. This
    /// measures whether using it is worth the change, on both halves: the widening
    /// pass that would disappear, and the gather that would read half as much
    /// index.
    ///
    /// Run with
    /// `cargo test --release --lib u32_indices_to_the_gather -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn u32_indices_to_the_gather() {
        const ROWS: usize = 100_000;

        let batch = RecordBatch::new(
            Schema::new(vec![
                Field::required("id", DataType::Int64),
                Field::required("dept", DataType::Utf8),
                Field::required("score", DataType::Float64),
            ]),
            vec![
                ColumnBuilder::int64((0..ROWS as i64).map(Some).collect()),
                ColumnBuilder::utf8_from_str(
                    (0..ROWS).map(|i| Some(["eng", "sales", "hr", "ops", "legal"][i % 5])).collect(),
                ),
                ColumnBuilder::float64((0..ROWS).map(|i| Some((i % 100) as f64 + 0.5)).collect()),
            ],
        )
        .unwrap();

        // A realistic permutation: what a descending sort on a 100-value column
        // produces, so the gather is as scattered as the real one.
        let mut pairs: Vec<(u64, u32)> = (0..ROWS)
            .map(|i| (!f64_sort_key((i % 100) as f64 + 0.5), i as u32))
            .collect();
        pairs.sort_unstable();
        let narrow: Vec<u32> = pairs.iter().map(|&(_, i)| i).collect();
        let wide: Vec<usize> = narrow.iter().map(|&i| i as usize).collect();

        // The two gathers must produce the same rows, or the faster one is moot.
        for c in 0..batch.column_count() {
            let a = batch.column(c).gather_u32(&narrow).unwrap();
            let b = batch.column(c).filter_by_indices(&wide).unwrap();
            assert_eq!(a.len(), b.len(), "column {c}: length");
            for r in (0..a.len()).step_by(997) {
                assert_eq!(a.get(r).unwrap(), b.get(r).unwrap(), "column {c} row {r}");
            }
        }

        let time = |label: &str, f: &dyn Fn(), reps: u32| {
            for _ in 0..3 { f(); }
            let t = std::time::Instant::now();
            for _ in 0..reps { f(); }
            let per = t.elapsed() / reps;
            println!("  {label:>32}: {per:>9.2?}");
            per
        };

        println!("ORDER BY, {ROWS} rows, 3 columns:");
        let widen = time(
            "widening u32 -> usize",
            &|| { std::hint::black_box(narrow.iter().map(|&i| i as usize).collect::<Vec<usize>>()); },
            200,
        );
        let g_wide = time(
            "gather by usize (today)",
            &|| {
                for c in 0..batch.column_count() {
                    std::hint::black_box(batch.column(c).filter_by_indices(&wide).unwrap());
                }
            },
            60,
        );
        let g_narrow = time(
            "gather by u32",
            &|| {
                for c in 0..batch.column_count() {
                    std::hint::black_box(batch.column(c).gather_u32(&narrow).unwrap());
                }
            },
            60,
        );

        println!("  gather alone: {:.2}x", g_narrow.as_secs_f64() / g_wide.as_secs_f64());
        println!(
            "  with the widening pass removed too: {:?} -> {:?}",
            widen + g_wide,
            g_narrow,
        );
    }

    /// Where does `ORDER BY` spend its time at the size we lose on?
    ///
    /// `order_by/100k` is 2.29 ms against Polars' 1.62 ms -- a 0.67 ms gap, larger
    /// in absolute terms than anything the join build work was chasing. The
    /// existing `tests/diag_sort_breakdown.rs` splits merge from sort, but it
    /// builds 1024-row batches while the Parquet reader produces 65 536-row ones,
    /// so at 100k rows the benchmark has two batches and the merge it measures is
    /// not the merge the benchmark pays. This splits the part that remains.
    ///
    /// Measured in one process, alternating: this machine drifts further between
    /// runs than the effect being looked for.
    ///
    /// Run with
    /// `cargo test --release --lib sort_phase_breakdown -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn sort_phase_breakdown() {
        const ROWS: usize = 100_000;

        // The benchmark's shape: three columns, sorted on one Float64 descending,
        // and the whole row gathered afterwards.
        let batch = RecordBatch::new(
            Schema::new(vec![
                Field::required("id", DataType::Int64),
                Field::required("dept", DataType::Utf8),
                Field::required("score", DataType::Float64),
            ]),
            vec![
                ColumnBuilder::int64((0..ROWS as i64).map(Some).collect()),
                ColumnBuilder::utf8_from_str(
                    (0..ROWS).map(|i| Some(["eng", "sales", "hr", "ops", "legal"][i % 5])).collect(),
                ),
                ColumnBuilder::float64((0..ROWS).map(|i| Some((i % 100) as f64 + 0.5)).collect()),
            ],
        )
        .unwrap();
        let exprs = vec![SortExpr::desc(Expr::Column("score".into()))];

        let indices_only = || {
            let t = Instant::now();
            let idx = try_typed_sort(&batch, &exprs, None).unwrap();
            let e = t.elapsed();
            assert!(idx.is_some(), "the typed path should take this column");
            (e, idx.unwrap())
        };
        let (_, idx) = indices_only();
        assert_eq!(idx.len(), ROWS);

        let gather_only = || {
            let t = Instant::now();
            let out = gather_narrow(&batch, &idx).unwrap();
            let e = t.elapsed();
            assert_eq!(out.row_count(), ROWS);
            e
        };
        let whole = || {
            let t = Instant::now();
            let out = execute_sort(std::slice::from_ref(&batch), &exprs).unwrap();
            let e = t.elapsed();
            assert_eq!(out.row_count(), ROWS);
            e
        };

        for _ in 0..3 {
            indices_only();
            gather_only();
            whole();
        }
        let mut i = Vec::new();
        let mut g = Vec::new();
        let mut w = Vec::new();
        for _ in 0..9 {
            i.push(indices_only().0);
            g.push(gather_only());
            w.push(whole());
        }
        let (i, g, w) = (
            *i.iter().min().unwrap(),
            *g.iter().min().unwrap(),
            *w.iter().min().unwrap(),
        );

        println!("ORDER BY score DESC, {ROWS} rows, 3 columns:");
        println!("  produce the sorted indices = {i:?}");
        println!("  gather the rows            = {g:?}");
        println!("  whole execute_sort         = {w:?}");
        println!(
            "  indices are {:.0}% of it, gather {:.0}%",
            i.as_secs_f64() / w.as_secs_f64() * 100.0,
            g.as_secs_f64() / w.as_secs_f64() * 100.0,
        );
    }
}

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
    fn radix_sort_matches_comparison_sort() {
        let n = RADIX_MIN_ROWS * 2 + 37;
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut pairs: Vec<(u64, u32)> = (0..n)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state, i as u32)
            })
            .collect();
        let mut expected = pairs.clone();
        expected.sort_unstable();
        sort_pairs_ord(&mut pairs);
        assert_eq!(pairs, expected);
    }

    #[test]
    fn radix_sort_handles_constant_and_duplicate_keys() {
        let n = RADIX_MIN_ROWS + 5;
        let mut pairs: Vec<(u64, u32)> = (0..n).map(|i| (42u64, i as u32)).collect();
        sort_pairs_ord(&mut pairs);
        assert_eq!(pairs, (0..n).map(|i| (42u64, i as u32)).collect::<Vec<_>>());

        let mut dup: Vec<(u32, u32)> = (0..n).map(|i| ((i % 3) as u32, i as u32)).collect();
        let mut expected = dup.clone();
        expected.sort_unstable();
        sort_pairs_ord(&mut dup);
        assert_eq!(dup, expected);
    }

    #[test]
    fn sort_large_float_column_is_ordered() {
        let n = RADIX_MIN_ROWS * 2;
        let schema = Schema::new(vec![Field::required("score", DataType::Float64)]);
        let vals: Vec<Option<f64>> = (0..n)
            .map(|i| Some(((i * 7919) % n) as f64 - (n / 2) as f64))
            .collect();
        let batch = RecordBatch::new(schema, vec![ColumnBuilder::float64(vals)]).unwrap();
        let sorted = execute_sort(&[batch], &[SortExpr::asc(Expr::Column("score".into()))]).unwrap();
        let col = sorted.column_by_name("score").unwrap();
        let mut prev = f64::NEG_INFINITY;
        for i in 0..col.len() {
            let ScalarValue::Float64(Some(v)) = col.get(i).unwrap() else { panic!("expected f64") };
            assert!(v >= prev, "out of order at {i}: {v} < {prev}");
            prev = v;
        }
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
