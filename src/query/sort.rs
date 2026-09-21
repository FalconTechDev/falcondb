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

    // Float64/Float32: convert to integer sort key for branchless Ord comparison.
    // This eliminates the partial_cmp NaN branch on every comparison in the sort.
    macro_rules! float_typed {
        ($c:expr, $float:ty, $uint:ty, $key_fn:expr) => {{
            let c = $c;
            let dense = c.dense();
            let mut pairs: Vec<($uint, u32)> = Vec::with_capacity(dense.len());
            let mut nulls: Vec<usize> = Vec::new();
            for i in 0..c.len() {
                if c.is_valid(i) {
                    let key = $key_fn(dense[i]);
                    // Flip all bits for DESC so we can always sort ascending.
                    pairs.push((if desc { !key } else { key }, i as u32));
                } else {
                    nulls.push(i);
                }
            }
            sort_pairs_ord(&mut pairs);
            Some(assemble(pairs, nulls, s.nulls_last, limit))
        }};
    }

    let out = match col {
        Column::Int64(c)     => typed!(c, i64),
        Column::Int32(c)     => typed!(c, i32),
        Column::Float64(c)   => float_typed!(c, f64, u64, f64_ord_key),
        Column::Float32(c)   => float_typed!(c, f32, u32, f32_ord_key),
        Column::Date32(c)    => typed!(c, i32),
        Column::Timestamp(c) => typed!(c, i64),
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
trait RadixKey: Ord + Copy + Send + Default {
    const PASSES: usize;
    fn digit(self, pass: usize) -> usize;
}

impl RadixKey for u64 {
    const PASSES: usize = 8;
    #[inline(always)]
    fn digit(self, pass: usize) -> usize { ((self >> (pass * 8)) & 0xFF) as usize }
}

impl RadixKey for u32 {
    const PASSES: usize = 4;
    #[inline(always)]
    fn digit(self, pass: usize) -> usize { ((self >> (pass * 8)) & 0xFF) as usize }
}

/// Below this, the radix scratch allocation costs more than pdqsort's comparisons.
const RADIX_MIN_ROWS: usize = 8_192;

/// Sort (Ord-key, row) pairs ascending — branchless, no closure, no NaN special case.
fn sort_pairs_ord<T: RadixKey>(pairs: &mut Vec<(T, u32)>) {
    if pairs.len() >= RADIX_MIN_ROWS {
        radix_sort_pairs(pairs);
    } else {
        pairs.sort_unstable();
    }
}

/// LSD radix sort, 8 bits per pass, stable.
///
/// All pass histograms are built in a single scan so that passes whose digit is
/// constant across every key can be skipped — for real float columns the high
/// exponent bytes rarely vary, which typically removes half the passes.
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

        // The benchmark's key: `score` is (i % 100) + 0.5 as f64, sorted
        // descending, so the keys are bit-flipped before sorting.
        let keys: Vec<u64> = (0..ROWS)
            .map(|i| !f64_sort_key((i % 100) as f64 + 0.5))
            .collect();
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
            let out = gather(&batch, &idx).unwrap();
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
