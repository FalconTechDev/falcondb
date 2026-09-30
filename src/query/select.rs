use rayon::prelude::*;

use crate::storage::{
    schema::{Field, Schema},
    table::{RecordBatch, Table},
    types::{FalconResult, ScalarValue},
};

use super::expr::Expr;

// в”Ђв”Ђв”Ђ Projection в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// What columns to return.
#[derive(Clone, Debug)]
pub enum Projection {
    /// SELECT *
    All,
    /// SELECT col1, col2, expr AS alias, ...
    Columns(Vec<SelectExpr>),
}

#[derive(Clone, Debug)]
pub struct SelectExpr {
    pub expr:  Expr,
    pub alias: String,
}

impl SelectExpr {
    pub fn column(name: &str) -> Self {
        Self { expr: Expr::Column(name.into()), alias: name.into() }
    }

    pub fn named(expr: Expr, alias: &str) -> Self {
        Self { expr, alias: alias.into() }
    }
}

// в”Ђв”Ђв”Ђ SelectQuery в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// A fully structured SELECT query.
#[derive(Clone, Debug)]
pub struct SelectQuery {
    pub projection: Projection,
    pub filter:     Option<Expr>,
    pub limit:      Option<usize>,
    pub offset:     usize,
    pub distinct:   bool,
    /// Projection pushdown: when set, the scan materialises only these columns.
    ///
    /// `None` means "every column", which is always correct. A caller that sets
    /// this promises the set covers every column any later stage will read —
    /// GROUP BY, HAVING, ORDER BY and window clauses included, not just the
    /// SELECT list. Getting that wrong surfaces as a missing-column error, not
    /// as silently wrong data.
    pub scan_columns: Option<std::collections::HashSet<String>>,
}

impl SelectQuery {
    pub fn scan_all() -> Self {
        Self {
            projection:   Projection::All,
            filter:       None,
            limit:        None,
            offset:       0,
            distinct:     false,
            scan_columns: None,
        }
    }

    /// Restrict the scan to `cols` (projection pushdown).
    pub fn with_scan_columns(mut self, cols: std::collections::HashSet<String>) -> Self {
        self.scan_columns = Some(cols);
        self
    }

    pub fn with_filter(mut self, expr: Expr) -> Self {
        self.filter = Some(expr);
        self
    }

    pub fn with_limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    pub fn with_offset(mut self, n: usize) -> Self {
        self.offset = n;
        self
    }

    pub fn with_columns(mut self, cols: Vec<SelectExpr>) -> Self {
        self.projection = Projection::Columns(cols);
        self
    }

    pub fn with_distinct(mut self, d: bool) -> Self {
        self.distinct = d;
        self
    }
}

// в”Ђв”Ђв”Ђ Executor в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// Execute a SELECT query over a table and return a flat list of result batches.
pub fn execute_select(table: &Table, query: &SelectQuery) -> FalconResult<Vec<RecordBatch>> {
    // Each batch is processed independently — parallelism here is safe.
    // Small tables are done serially: rayon's scheduling would cost more than it
    // saves. The size is counted in rows, not batches, so the decision does not
    // change meaning with the batch shape the loader happened to produce.
    let total_rows: usize = table.batches.iter().map(|b| b.row_count()).sum();
    let processed: FalconResult<Vec<Option<RecordBatch>>> =
        if total_rows < crate::query::tuning::par_min_rows_per_batch_fanout() {
        table
            .batches
            .iter()
            .enumerate()
            .map(|(i, batch)| {
                if let Some(filter) = &query.filter {
                    if table.stats.get(i).is_some_and(|s| s.can_skip(filter)) {
                        return Ok(None);
                    }
                }
                process_batch(batch, query)
            })
            .collect()
    } else {
        table
            .batches
            .par_iter()
            .enumerate()
            .map(|(i, batch)| {
                // Zone-map pruning: if the batch's precomputed bounds prove no row can
                // match, skip it without reading a single value.
                if let Some(filter) = &query.filter {
                    if table.stats.get(i).is_some_and(|s| s.can_skip(filter)) {
                        return Ok(None);
                    }
                }
                process_batch(batch, query)
            })
            .collect()
    };

    // Filter out empty batches, then apply global offset + limit.
    let mut result: Vec<RecordBatch> = processed?
        .into_iter()
        .flatten()
        .filter(|b| b.row_count() > 0)
        .collect();

    if query.distinct {
        result = dedup_batches(result);
    }

    if query.offset > 0 || query.limit.is_some() {
        result = apply_offset_limit(result, query.offset, query.limit);
    }

    Ok(result)
}

/// Hash one row into a single `u64`, without allocating.
///
/// The previous key was a `Vec<String>` built with `to_string()` per cell, which
/// cost one heap allocation per cell — three million of them for a million-row,
/// three-column DISTINCT. Hashing the raw typed values costs none.
///
/// Hashing rather than storing the key means two distinct rows could collide and
/// one would be dropped as a duplicate. `row_eq` re-checks equality on collision,
/// so the hash only decides *where* to look, never what is equal.
pub(crate) fn row_hash(batch: &RecordBatch, row: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    use crate::storage::column::Column;

    let mut h = super::hash::FxHasher::default();
    for ci in 0..batch.column_count() {
        // Discriminate the column position so that shifting a value between
        // columns changes the hash.
        ci.hash(&mut h);
        match batch.column(ci) {
            Column::Boolean(c)   => c.get(row).hash(&mut h),
            Column::Int32(c)     => c.get(row).hash(&mut h),
            Column::Int64(c)     => c.get(row).hash(&mut h),
            Column::Date32(c)    => c.get(row).hash(&mut h),
            Column::Timestamp(c) => c.get(row).hash(&mut h),
            Column::Utf8(c)      => c.get_str(row).hash(&mut h),
            // Floats have no `Hash`: NaN != NaN and 0.0 == -0.0, so hash the bit
            // pattern with those two cases normalised, matching `row_eq`.
            Column::Float32(c) => c.get(row).map(norm_f32).hash(&mut h),
            Column::Float64(c) => c.get(row).map(norm_f64).hash(&mut h),
            Column::Decimal128(c) => c.inner.get(row).hash(&mut h),
        }
    }
    h.finish()
}

#[inline]
fn norm_f32(v: f32) -> u32 {
    if v.is_nan() { f32::NAN.to_bits() } else if v == 0.0 { 0 } else { v.to_bits() }
}

#[inline]
fn norm_f64(v: f64) -> u64 {
    if v.is_nan() { f64::NAN.to_bits() } else if v == 0.0 { 0 } else { v.to_bits() }
}

/// Are two rows equal across every column? Used to settle hash collisions.
pub(crate) fn row_eq(a: &RecordBatch, ra: usize, b: &RecordBatch, rb: usize) -> bool {
    use crate::storage::column::Column;

    if a.column_count() != b.column_count() {
        return false;
    }
    for ci in 0..a.column_count() {
        let same = match (a.column(ci), b.column(ci)) {
            (Column::Boolean(x),   Column::Boolean(y))   => x.get(ra) == y.get(rb),
            (Column::Int32(x),     Column::Int32(y))     => x.get(ra) == y.get(rb),
            (Column::Int64(x),     Column::Int64(y))     => x.get(ra) == y.get(rb),
            (Column::Date32(x),    Column::Date32(y))    => x.get(ra) == y.get(rb),
            (Column::Timestamp(x), Column::Timestamp(y)) => x.get(ra) == y.get(rb),
            (Column::Utf8(x),      Column::Utf8(y))      => x.get_str(ra) == y.get_str(rb),
            (Column::Float32(x),   Column::Float32(y))   =>
                x.get(ra).map(norm_f32) == y.get(rb).map(norm_f32),
            (Column::Float64(x),   Column::Float64(y))   =>
                x.get(ra).map(norm_f64) == y.get(rb).map(norm_f64),
            (Column::Decimal128(x), Column::Decimal128(y)) => x.inner.get(ra) == y.inner.get(rb),
            // Different column types cannot hold equal values.
            _ => false,
        };
        if !same {
            return false;
        }
    }
    true
}

/// Remove duplicate rows across batches (for SELECT DISTINCT and UNION).
pub fn dedup_batches(batches: Vec<RecordBatch>) -> Vec<RecordBatch> {
    use super::hash::FastMap;

    if let Some(keeps) = dedup_single_primitive(&batches) {
        return apply_keeps(&batches, keeps);
    }

    // hash -> positions in the *input* batches that hashed there. Recording input
    // positions rather than output ones matters: a row has to be comparable
    // against earlier rows of the batch currently being scanned, which has not
    // been filtered or emitted yet.
    let mut seen: FastMap<u64, Vec<(usize, usize)>> = FastMap::default();
    let mut keeps: Vec<Vec<bool>> = Vec::with_capacity(batches.len());

    for (bi, batch) in batches.iter().enumerate() {
        let mut keep = Vec::with_capacity(batch.row_count());
        for row in 0..batch.row_count() {
            let h = row_hash(batch, row);
            let bucket = seen.entry(h).or_default();
            // The hash only narrows the search; equality decides.
            let duplicate = bucket
                .iter()
                .any(|&(pb, pr)| row_eq(&batches[pb], pr, batch, row));

            if duplicate {
                keep.push(false);
            } else {
                bucket.push((bi, row));
                keep.push(true);
            }
        }
        keeps.push(keep);
    }

    apply_keeps(&batches, keeps)
}

/// Materialise the surviving rows of each batch.
fn apply_keeps(batches: &[RecordBatch], keeps: Vec<Vec<bool>>) -> Vec<RecordBatch> {
    let mut result = Vec::with_capacity(batches.len());
    for (batch, keep) in batches.iter().zip(keeps) {
        if keep.iter().any(|&k| k) {
            if let Ok(filtered) = batch.filter_by_mask(&keep) {
                result.push(filtered);
            }
        }
    }
    result
}

/// Minimum rows before splitting the dedup scan across cores.
///
/// At 131_072 a 100k-row DISTINCT fell just under the line and ran serially, which
/// left it about 2x behind Polars while the 1M case — which did split — led every
/// rival. The merge pass costs one hash per distinct value per chunk, so it stays
/// cheap well below this.
/// Measured at 32_768 on a machine reporting 20 threads, so ~1.6k each.
fn par_dedup_rows() -> usize {
    crate::query::tuning::par_min_rows(1_638)
}

/// DISTINCT over a single primitive column, without per-row type dispatch.
///
/// The general path matches on the column enum, calls a validity-checking getter
/// and consults a `FastMap<u64, Vec<(usize, usize)>>` — for *every row*. At a
/// million rows that measured 20.5 ms against DuckDB's 4.7 ms, and it grew exactly
/// linearly where the rivals grew sub-linearly. Here the type is resolved once per
/// batch and the raw value goes straight into a typed set.
///
/// Returns `None` for anything it does not handle — several columns, NULLs, or a
/// type not listed — so the caller falls back rather than this having to restate
/// the general semantics. First occurrence wins, exactly as before, so the output
/// and its order are unchanged.
fn dedup_single_primitive(batches: &[RecordBatch]) -> Option<Vec<Vec<bool>>> {
    use super::hash::FastSet;
    use crate::storage::column::Column;

    let first = batches.first()?;
    if first.column_count() != 1 {
        return None;
    }
    if batches.iter().any(|b| b.column(0).null_count() > 0) {
        return None;
    }

    let total_rows: usize = batches.iter().map(|b| b.row_count()).sum();

    // Text: number the distinct strings once, then dedup numbers.
    //
    // A dictionary column already knows its distinct values, but each batch has
    // its own dictionary, so a code means nothing across batches. Numbering the
    // strings globally -- one pass over the dictionaries, which hold distinct
    // values rather than rows -- turns the whole question into the integer one
    // the arms below already answer.
    //
    // The number is an identity, not a hash. A 64-bit hash of the string would
    // fit the same machinery and be slightly cheaper, but two different strings
    // that collided would be silently deduplicated, and returning the wrong rows
    // is not a trade worth making for a few nanoseconds.
    //
    // Without this, `SELECT DISTINCT` on a text column took the general path --
    // a hash map of buckets holding row positions, with a full row comparison per
    // candidate. Measured at 100k rows: 3.90ms against 772us for the same query
    // over a float column, while the text column had five distinct values to the
    // float's hundred.
    if matches!(first.column(0), Column::Utf8(_)) {
        let mut ids: super::hash::FastMap<&str, u64> = super::hash::FastMap::default();
        let mut next_id = 0u64;
        let mut keys: Vec<Vec<u64>> = Vec::with_capacity(batches.len());

        for b in batches {
            let Column::Utf8(c) = b.column(0) else { return None };
            // Local code -> global id, built from this batch's dictionary.
            let mut local = vec![0u64; c.dict().len()];
            for (code, text) in c.dict().iter().enumerate() {
                let id = *ids.entry(text.as_str()).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                local[code] = id;
            }
            keys.push(c.codes().iter().map(|&code| local[code as usize]).collect());
        }

        // First occurrence wins, which is what DISTINCT keeps.
        let mut seen: FastSet<u64> = FastSet::default();
        let keeps = keys
            .iter()
            .map(|batch_keys| batch_keys.iter().map(|&id| seen.insert(id)).collect())
            .collect();
        return Some(keeps);
    }

    macro_rules! run {
        ($variant:ident, $key:expr) => {{
            // Hashing every row is the bulk of the work and each row is independent,
            // so it splits over cores. What cannot be split is *which* occurrence
            // wins: DISTINCT keeps the first, which is a global fact.
            //
            // So each chunk marks the rows that are first *within the chunk*, then a
            // short serial pass walks those in chunk order and clears any that an
            // earlier chunk already had. The second pass touches only locally-kept
            // rows — at most one per distinct value per chunk — rather than all of
            // them, so the serial part shrinks with cardinality instead of row count.
            let par = total_rows >= par_dedup_rows() && batches.len() > 1;
            if par {
                let per_chunk: Vec<(FastSet<u64>, Vec<Vec<bool>>)> = batches
                    .par_chunks(batches.len().div_ceil(rayon::current_num_threads().max(1)).max(1))
                    .map(|chunk| {
                        let mut seen: FastSet<u64> = FastSet::default();
                        let mut keeps = Vec::with_capacity(chunk.len());
                        for batch in chunk {
                            let mut keep = Vec::with_capacity(batch.row_count());
                            if let Column::$variant(c) = batch.column(0) {
                                for &v in c.dense() {
                                    keep.push(seen.insert($key(v)));
                                }
                            }
                            keeps.push(keep);
                        }
                        (seen, keeps)
                    })
                    .collect();

                // A chunk that saw no rows of the expected type produced no keeps;
                // that only happens on a mixed-type input, which the caller rejects.
                if per_chunk.iter().map(|(_, k)| k.iter().map(|v| v.len()).sum::<usize>()).sum::<usize>()
                    != total_rows
                {
                    return None;
                }

                let mut global: FastSet<u64> = FastSet::default();
                let mut keeps: Vec<Vec<bool>> = Vec::with_capacity(batches.len());
                let mut bi = 0usize;
                for (_, chunk_keeps) in per_chunk {
                    for mut keep in chunk_keeps {
                        let Column::$variant(c) = batches[bi].column(0) else { return None };
                        let dense = c.dense();
                        for (row, k) in keep.iter_mut().enumerate() {
                            if *k && !global.insert($key(dense[row])) {
                                *k = false;
                            }
                        }
                        keeps.push(keep);
                        bi += 1;
                    }
                }
                return Some(keeps);
            }

            let mut seen: FastSet<u64> = FastSet::default();
            let mut keeps = Vec::with_capacity(batches.len());
            for batch in batches {
                let Column::$variant(c) = batch.column(0) else { return None };
                let dense = c.dense();
                let mut keep = Vec::with_capacity(dense.len());
                for &v in dense {
                    keep.push(seen.insert($key(v)));
                }
                keeps.push(keep);
            }
            Some(keeps)
        }};
    }

    match first.column(0) {
        Column::Int64(_)  => run!(Int64, |v: i64| v as u64),
        Column::Int32(_)  => run!(Int32, |v: i32| v as i64 as u64),
        Column::Date32(_) => run!(Date32, |v: i32| v as i64 as u64),
        // Normalised the same way `row_hash` does it, so NaN and -0.0 behave as
        // they did on the general path.
        Column::Float64(_) => run!(Float64, |v: f64| norm_f64(v)),
        Column::Float32(_) => run!(Float32, |v: f32| norm_f32(v) as u64),
        _ => None,
    }
}

fn process_batch(batch: &RecordBatch, query: &SelectQuery) -> FalconResult<Option<RecordBatch>> {
    // 1. Filter (WHERE) вЂ” packed bitmask: 8Г— smaller than Vec<bool>, better cache behaviour.
    //
    // With projection pushdown active, only the needed columns are rebuilt. The
    // mask is still computed against the full batch, since the filter may read a
    // column the query does not otherwise return.
    let filtered = match (&query.filter, &query.scan_columns) {
        (Some(filter), Some(keep)) => {
            let mask = filter.eval_mask_bits(batch)?;
            batch.filter_by_bitmask_cols(&mask, keep)?
        }
        (Some(filter), None) => {
            let mask = filter.eval_mask_bits(batch)?;
            batch.filter_by_bitmask(&mask)?
        }
        (None, Some(keep)) => batch.project_columns(keep)?,
        (None, None) => batch.clone(),
    };

    if filtered.row_count() == 0 {
        return Ok(None);
    }

    // 2. Project (SELECT columns / expressions).
    let projected = match &query.projection {
        Projection::All => filtered,
        Projection::Columns(exprs) => project_batch(&filtered, exprs)?,
    };

    Ok(Some(projected))
}

/// Apply a scalar function to a dictionary column's *entries* rather than its rows.
///
/// `Expr::Function`'s general path allocates a `Vec` of argument references per
/// row and calls `eval_function`, whose first act is `name.to_uppercase()` -- a
/// `String` allocated per row just to choose a match arm. On top of that the
/// column is boxed into `ScalarValue` per row and the result boxed again.
///
/// Measured over 100k rows holding five distinct strings, cost per row:
/// `CONCAT` 358ns, `LOWER` 256ns, `UPPER` 254ns, `LENGTH` 191ns. Five distinct
/// values means the answer needs five transformations, not a hundred thousand.
///
/// For the functions that return text the output is the same column with a
/// different dictionary, so the rows cost a `u32` copy each.
///
/// The transformed dictionary is deduplicated and the codes remapped, which the
/// structure does not require -- `from_parts` only checks that codes index the
/// dictionary -- but `UPPER` maps `sales` and `Sales` onto one string, and a
/// dictionary holding the same string twice would give anything that groups by
/// code two groups for one value. Proving nothing does that is harder than
/// keeping the invariant, and the dedup costs one hash per distinct string.
///
/// Returns `None` for every shape it does not handle, which then takes the
/// general path exactly as before.
fn dict_mapped_function(
    batch: &RecordBatch,
    expr: &Expr,
) -> Option<crate::storage::column::Column> {
    use crate::storage::column::{Column, PrimitiveColumn, Utf8Column};

    let Expr::Function { name, args } = expr else { return None };

    // The column argument, and any literal that goes with it.
    enum Shape<'a> { One, ConcatAfter(&'a str), ConcatBefore(&'a str) }
    // The shapes worth a fast path: one column, or one column and one literal.
    //
    // The second arm once required the literal to be text, which was written for
    // CONCAT and silently excluded `ROUND(f, 2)` -- the numeric arms below never
    // saw it, and ROUND stayed at 183ns per row while ABS dropped to 7ns. The
    // literal's type is the callee's business, not this match's.
    let (col_name, shape) = match args.as_slice() {
        [Expr::Column(c)] => (c, Shape::One),
        [Expr::Column(c), Expr::Literal(ScalarValue::Utf8(Some(lit)))] => {
            (c, Shape::ConcatAfter(lit.as_str()))
        }
        [Expr::Literal(ScalarValue::Utf8(Some(lit))), Expr::Column(c)] => {
            (c, Shape::ConcatBefore(lit.as_str()))
        }
        [Expr::Column(c), Expr::Literal(_)] => (c, Shape::One),
        _ => return None,
    };

    let idx = batch.schema.column_index(col_name).ok()?;

    // Numeric functions take the dense values straight from the column.
    //
    // There is no dictionary to exploit here -- numbers do not repeat the way a
    // department name does -- so the saving is only in not boxing each value into
    // a `ScalarValue` and back, and in choosing the match arm once instead of once
    // per row. Measured per row over 100k rows: ROUND 183ns, COALESCE 157, ABS 137.
    //
    // The arms below mirror `eval_function` exactly, including its choice of
    // result type: ABS keeps an integer column integral, while ROUND answers in
    // floats whatever it was given. A fast path is not the place to change either.
    if let Some(col) = dense_numeric_function(batch.columns[idx].as_ref(), &upper_name(name), args) {
        return Some(col);
    }

    let Column::Utf8(c) = batch.columns[idx].as_ref() else { return None };

    let upper = upper_name(name);
    let is_concat = matches!(upper.as_str(), "CONCAT");
    if !is_concat && !matches!(shape, Shape::One) {
        return None;
    }
    // A single-argument CONCAT, or one whose literal is not text, is not a shape
    // this path models. `Shape::One` now also covers `f(col, <non-text literal>)`,
    // which reaches here only when the numeric arms above declined it.
    if is_concat && (matches!(shape, Shape::One) || args.len() != 2) {
        return None;
    }
    if !is_concat && args.len() != 1 {
        return None;
    }

    // LENGTH answers with a number, so it builds a different column type.
    if matches!(upper.as_str(), "LENGTH" | "LEN") && matches!(shape, Shape::One) {
        let lens: Vec<i64> = c.dict().iter().map(|e| e.chars().count() as i64).collect();
        let values: Vec<Option<i64>> = c
            .codes()
            .iter()
            .map(|&code| {
                if Utf8Column::is_null_code(code) { None } else { Some(lens[code as usize]) }
            })
            .collect();
        return Some(Column::Int64(PrimitiveColumn::new(values)));
    }

    let transform = |e: &str| -> Option<String> {
        Some(match upper.as_str() {
            "UPPER" => e.to_uppercase(),
            "LOWER" => e.to_lowercase(),
            "TRIM"  => e.trim().to_string(),
            "LTRIM" => e.trim_start().to_string(),
            "RTRIM" => e.trim_end().to_string(),
            "CONCAT" => match shape {
                Shape::ConcatAfter(lit)  => format!("{e}{lit}"),
                Shape::ConcatBefore(lit) => format!("{lit}{e}"),
                Shape::One => return None,
            },
            _ => return None,
        })
    };

    // Transform each entry once, then fold duplicates back together.
    let mut new_dict: Vec<String> = Vec::with_capacity(c.dict().len());
    let mut seen: super::hash::FastMap<String, u32> = super::hash::FastMap::default();
    let mut remap: Vec<u32> = Vec::with_capacity(c.dict().len());
    for entry in c.dict().iter() {
        let out = transform(entry)?;
        let code = match seen.get(&out) {
            Some(&code) => code,
            None => {
                let code = new_dict.len() as u32;
                seen.insert(out.clone(), code);
                new_dict.push(out);
                code
            }
        };
        remap.push(code);
    }

    // What a NULL row becomes depends on the function, and the two answers differ.
    //
    // UPPER, LOWER and the TRIMs return NULL for a NULL input. CONCAT does not: it
    // skips NULL arguments and always returns a string, so `CONCAT(NULL, '!')` is
    // `"!"`. Keeping NULL for those rows was the first version of this function and
    // it silently disagreed with the path it replaced -- caught by the test that
    // runs a shape this fast path declines and compares.
    //
    // `transform("")` is exactly the answer for a row whose string contributes
    // nothing, which is what CONCAT does with a NULL.
    let null_code = if is_concat {
        let out = transform("")?;
        Some(match seen.get(&out) {
            Some(&code) => code,
            None => {
                let code = new_dict.len() as u32;
                seen.insert(out.clone(), code);
                new_dict.push(out);
                code
            }
        })
    } else {
        None
    };

    let codes: Vec<u32> = c
        .codes()
        .iter()
        .map(|&code| {
            if Utf8Column::is_null_code(code) {
                null_code.unwrap_or(code)
            } else {
                remap[code as usize]
            }
        })
        .collect();

    Some(Column::Utf8(Utf8Column::from_parts(std::sync::Arc::new(new_dict), codes)))
}

/// Uppercase a function name once, for matching.
fn upper_name(name: &str) -> String {
    name.to_uppercase()
}

/// Apply a numeric function to a dense column without boxing each value.
///
/// Returns `None` for anything it does not handle, including any column that has
/// NULLs in a position these arms would have to think about -- those take the
/// general path, which already answers them correctly.
fn dense_numeric_function(
    col:  &crate::storage::column::Column,
    name: &str,
    args: &[Expr],
) -> Option<crate::storage::column::Column> {
    use crate::storage::column::{Column, PrimitiveColumn};

    // COALESCE(col, <literal>) fills the NULLs and leaves everything else alone.
    //
    // Only when the literal's type matches the column's. With a mismatch the
    // general path hands back whichever argument won per row, so the column ends
    // up typed from whatever its first non-NULL value happened to be; that is a
    // question about type coercion, not about speed, and it is left where it is.
    if name == "COALESCE" && args.len() == 2 {
        if let Expr::Literal(lit) = &args[1] {
            match (col, lit) {
                (Column::Int64(c), ScalarValue::Int64(Some(fill))) => {
                    let values: Vec<Option<i64>> =
                        (0..c.len()).map(|i| Some(c.get(i).unwrap_or(*fill))).collect();
                    return Some(Column::Int64(PrimitiveColumn::new(values)));
                }
                (Column::Float64(c), ScalarValue::Float64(Some(fill))) => {
                    let values: Vec<Option<f64>> =
                        (0..c.len()).map(|i| Some(c.get(i).unwrap_or(*fill))).collect();
                    return Some(Column::Float64(PrimitiveColumn::new(values)));
                }
                _ => return None,
            }
        }
        return None;
    }

    // A second argument is only meaningful to ROUND, and only as a literal.
    //
    // This guard sits *below* the COALESCE block on purpose. It was above it at
    // first, and since it returns `None` for any two-argument call that is not
    // ROUND, it rejected COALESCE before the code handling COALESCE could run --
    // leaving that code unreachable and the function at its old 155ns per row.
    //
    // That is the second time in this work a guard written for one function has
    // silently swallowed another: the argument match above once admitted a second
    // literal only when it was text, written for CONCAT, which excluded
    // `ROUND(f, 2)` the same way. Both were caught by measuring each function
    // separately and noticing that one did not move.
    let digits = match args {
        [_] => None,
        [_, Expr::Literal(v)] => Some(v.as_i64()? as i32),
        _ => return None,
    };
    if digits.is_some() && name != "ROUND" {
        return None;
    }

    match col {
        Column::Float64(c) => {
            let f: Box<dyn Fn(f64) -> f64> = match name {
                "ABS"   => Box::new(f64::abs),
                "CEIL" | "CEILING" => Box::new(f64::ceil),
                "FLOOR" => Box::new(f64::floor),
                "SQRT"  => Box::new(f64::sqrt),
                "SIGN"  => Box::new(|x: f64| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }),
                "ROUND" => {
                    let factor = 10f64.powi(digits.unwrap_or(0));
                    Box::new(move |x: f64| (x * factor).round() / factor)
                }
                _ => return None,
            };
            let values: Vec<Option<f64>> = (0..c.len())
                .map(|i| c.get(i).map(&*f))
                .collect();
            Some(Column::Float64(PrimitiveColumn::new(values)))
        }
        Column::Int64(c) => {
            // ABS and SIGN keep the column integral, as `eval_function` does.
            // ROUND of an integer answers in floats, which is a different column
            // type, so it goes through the float arm by widening first.
            match name {
                "ABS" | "SIGN" => {
                    let f: fn(i64) -> i64 = if name == "ABS" { i64::abs } else { i64::signum };
                    let values: Vec<Option<i64>> = (0..c.len())
                        .map(|i| c.get(i).map(f))
                        .collect();
                    Some(Column::Int64(PrimitiveColumn::new(values)))
                }
                "ROUND" | "CEIL" | "CEILING" | "FLOOR" | "SQRT" => {
                    let factor = 10f64.powi(digits.unwrap_or(0));
                    let g: Box<dyn Fn(f64) -> f64> = match name {
                        "ROUND" => Box::new(move |x: f64| (x * factor).round() / factor),
                        "CEIL" | "CEILING" => Box::new(f64::ceil),
                        "FLOOR" => Box::new(f64::floor),
                        _ => Box::new(f64::sqrt),
                    };
                    let values: Vec<Option<f64>> = (0..c.len())
                        .map(|i| c.get(i).map(|v| g(v as f64)))
                        .collect();
                    Some(Column::Float64(PrimitiveColumn::new(values)))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// `CASE WHEN <condition> THEN <literal> ELSE <literal> END`, chosen by bitmask.
///
/// The general path evaluates the condition into a `Vec<ScalarValue>` and then
/// clones a `ScalarValue` per row to pick the arm. The condition already has a
/// packed-bitmask evaluator -- the one `WHERE` uses -- and when both arms are
/// literals the result is two constants selected by a bit. Measured at 89ns per
/// row over 100k rows before this.
///
/// Deliberately narrow: one `WHEN`, both arms literal, an `ELSE` present. A
/// missing `ELSE` means NULL for unmatched rows, which is a different column
/// shape, and several `WHEN`s need the first-match rule the general path
/// implements. Both decline here and are answered exactly as before.
fn dense_case(
    batch: &RecordBatch,
    expr:  &Expr,
) -> Option<crate::storage::column::Column> {
    use crate::storage::column::{BooleanColumn, Column, PrimitiveColumn, Utf8Column};

    let Expr::Case { when_thens, else_expr } = expr else { return None };
    let [(cond, then)] = when_thens.as_slice() else { return None };
    let Expr::Literal(a) = then else { return None };
    let Some(else_e) = else_expr.as_ref() else { return None };
    let Expr::Literal(b) = else_e.as_ref() else { return None };

    let n = batch.row_count();
    let mask = cond.eval_mask_bits(batch).ok()?;
    let hit = |i: usize| mask[i >> 6] & (1u64 << (i & 63)) != 0;

    // Both arms must be the same kind, or the column has no single type.
    match (a, b) {
        (ScalarValue::Int64(Some(x)), ScalarValue::Int64(Some(y))) => {
            Some(Column::Int64(PrimitiveColumn::from_non_null(
                (0..n).map(|i| if hit(i) { *x } else { *y }).collect(),
            )))
        }
        (ScalarValue::Float64(Some(x)), ScalarValue::Float64(Some(y))) => {
            Some(Column::Float64(PrimitiveColumn::from_non_null(
                (0..n).map(|i| if hit(i) { *x } else { *y }).collect(),
            )))
        }
        (ScalarValue::Boolean(Some(x)), ScalarValue::Boolean(Some(y))) => {
            Some(Column::Boolean(BooleanColumn::new(
                (0..n).map(|i| Some(if hit(i) { *x } else { *y })).collect(),
            )))
        }
        (ScalarValue::Utf8(Some(x)), ScalarValue::Utf8(Some(y))) => {
            // Two distinct strings: a two-entry dictionary and one code per row.
            let dict = std::sync::Arc::new(vec![x.clone(), y.clone()]);
            let codes: Vec<u32> = (0..n).map(|i| if hit(i) { 0 } else { 1 }).collect();
            Some(Column::Utf8(Utf8Column::from_parts(dict, codes)))
        }
        _ => None,
    }
}

pub fn project_batch(batch: &RecordBatch, exprs: &[SelectExpr]) -> FalconResult<RecordBatch> {
    use crate::storage::column::{BooleanColumn, Column, PrimitiveColumn, Utf8Column};
    use crate::storage::types::DataType;

    let mut columns  = Vec::with_capacity(exprs.len());
    let mut fields   = Vec::with_capacity(exprs.len());
    let row_count    = batch.row_count();

    for sel in exprs {
        // Fast path: a bare column reference is already the column we want.
        //
        // The general path below evaluates the expression into a `Vec<ScalarValue>`
        // and rebuilds a typed column from it. `ScalarValue` is 48 bytes, so for a
        // plain `SELECT a, b` that boxes and unboxes every cell — 48 MB of
        // allocation per million rows — to reproduce data the batch already holds.
        // Columns live behind `Arc`, so the same result costs a refcount bump.
        if let Expr::Column(name) = &sel.expr {
            let idx = batch.schema.column_index(name)?;
            let mut field = batch.schema.fields[idx].clone();
            // The projection may rename the column: `SELECT a AS b`.
            field.name = sel.alias.clone();
            fields.push(field);
            columns.push(std::sync::Arc::clone(&batch.columns[idx]));
            continue;
        }

        // Fast path: `CASE WHEN <cond> THEN <a> ELSE <b> END` with literal arms.
        if let Some(col) = dense_case(batch, &sel.expr) {
            let mut field = crate::storage::schema::Field::nullable(&sel.alias, col.data_type());
            field.name = sel.alias.clone();
            fields.push(field);
            columns.push(std::sync::Arc::new(col));
            continue;
        }

        // Fast path: a function of one text column is answered per distinct
        // string, not per row.
        if let Some(col) = dict_mapped_function(batch, &sel.expr) {
            let mut field = crate::storage::schema::Field::nullable(&sel.alias, col.data_type());
            field.name = sel.alias.clone();
            fields.push(field);
            columns.push(std::sync::Arc::new(col));
            continue;
        }

        let values = sel.expr.eval_batch(batch)?;

        // Determine result type from first non-null value.
        let sample = values.iter().find(|v| !v.is_null());
        let dt = sample.map(|v| v.data_type()).unwrap_or(DataType::Null);

        let col = match dt {
            DataType::Boolean => Column::Boolean(BooleanColumn::new(
                values.iter().map(|v| v.as_bool()).collect(),
            )),
            DataType::Int32 | DataType::Int64 => Column::Int64(PrimitiveColumn::new(
                values.iter().map(|v| v.as_i64()).collect(),
            )),
            DataType::Float32 | DataType::Float64 => Column::Float64(PrimitiveColumn::new(
                values.iter().map(|v| v.as_f64()).collect(),
            )),
            DataType::Utf8 => Column::Utf8(Utf8Column::new(
                values.into_iter().map(|v| match v {
                    ScalarValue::Utf8(s) => s,
                    _ => None,
                }).collect(),
            )),
            DataType::Date32 => Column::Date32(PrimitiveColumn::new(
                values.iter().map(|v| match v {
                    ScalarValue::Date32(d) => *d,
                    // An integer in a date-typed projection is a day offset,
                    // which is how date arithmetic surfaces here.
                    ScalarValue::Int64(Some(n)) => Some(*n as i32),
                    _ => None,
                }).collect(),
            )),
            DataType::Timestamp => Column::Timestamp(PrimitiveColumn::new(
                values.iter().map(|v| match v {
                    ScalarValue::Timestamp(t)   => *t,
                    ScalarValue::Int64(Some(n)) => Some(*n),
                    _ => None,
                }).collect(),
            )),
            DataType::Decimal128(p, s) => Column::Decimal128(
                crate::storage::column::Decimal128Column::new(
                    values.iter().map(|v| match v {
                        ScalarValue::Decimal128(d) => d.value,
                        _ => None,
                    }).collect(),
                    p,
                    s,
                ),
            ),
            // Every value was NULL, so there is no type to infer. An all-NULL Int64
            // column is the neutral choice: it renders as NULL whatever the caller
            // expected.
            //
            // This match is deliberately exhaustive — no catch-all arm. A new
            // DataType must fail to compile here rather than fall through to an
            // all-NULL column, which is how projecting a DATE silently returned
            // NULL before these arms existed.
            DataType::Null => Column::Int64(PrimitiveColumn::new(vec![None; row_count])),
        };

        let field = Field::nullable(sel.alias.as_str(), col.data_type());
        columns.push(std::sync::Arc::new(col));
        fields.push(field);
    }

    // `from_shared`, not `new`: the fast path above hands back columns that are
    // already shared, and re-wrapping them would undo the point of it.
    RecordBatch::from_shared(Schema::new(fields), columns)
}

pub fn apply_offset_limit(
    batches: Vec<RecordBatch>,
    offset:  usize,
    limit:   Option<usize>,
) -> Vec<RecordBatch> {
    let mut rows_skipped = 0usize;
    let mut rows_taken   = 0usize;
    let max_rows         = limit.unwrap_or(usize::MAX);
    let mut result       = Vec::new();

    for batch in batches {
        if rows_taken >= max_rows { break; }

        let n = batch.row_count();

        if rows_skipped + n <= offset {
            rows_skipped += n;
            continue;
        }

        let start = if rows_skipped < offset { offset - rows_skipped } else { 0 };
        rows_skipped = offset;

        let available = n - start;
        let take      = available.min(max_rows - rows_taken);

        if take == 0 { break; }

        if let Ok(sliced) = batch.slice(start, take) {
            rows_taken += take;
            result.push(sliced);
        }
    }

    result
}

// в”Ђв”Ђв”Ђ Result helpers в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// Flatten result batches into a pretty-printed table string.
pub fn display_results(batches: &[RecordBatch]) -> String {
    if batches.is_empty() {
        return "(0 rows)\n".into();
    }

    let schema     = &batches[0].schema;
    let col_names  = schema.column_names();
    let col_count  = col_names.len();

    // Collect all rows as strings.
    let mut rows: Vec<Vec<String>> = Vec::new();
    for batch in batches {
        for row_idx in 0..batch.row_count() {
            let row: Vec<String> = (0..col_count)
                .map(|col_idx| {
                    batch.column(col_idx).get(row_idx)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|_| "ERR".into())
                })
                .collect();
            rows.push(row);
        }
    }

    // Compute column widths.
    let mut widths: Vec<usize> = col_names.iter().map(|n| n.len()).collect();
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    let separator: String = widths.iter()
        .map(|&w| "-".repeat(w + 2))
        .collect::<Vec<_>>()
        .join("+");
    let separator = format!("+{separator}+");

    let mut out = String::new();
    out.push_str(&separator);
    out.push('\n');

    // Header.
    let header: String = col_names.iter().zip(widths.iter())
        .map(|(name, &w)| format!(" {name:<w$} "))
        .collect::<Vec<_>>()
        .join("|");
    out.push_str(&format!("|{header}|\n"));
    out.push_str(&separator);
    out.push('\n');

    // Rows.
    for row in &rows {
        let line: String = row.iter().zip(widths.iter())
            .map(|(cell, &w)| format!(" {cell:<w$} "))
            .collect::<Vec<_>>()
            .join("|");
        out.push_str(&format!("|{line}|\n"));
    }

    out.push_str(&separator);
    out.push('\n');
    out.push_str(&format!("({} rows)\n", rows.len()));
    out
}

#[cfg(test)]
mod dict_functions {
    use crate::db::Database;
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::RecordBatch,
        types::{DataType, ScalarValue},
    };

    fn db_with(values: Vec<Option<&str>>) -> Database {
        let batch = RecordBatch::new(
            Schema::new(vec![Field::nullable("s", DataType::Utf8)]),
            vec![ColumnBuilder::utf8_from_str(values)],
        )
        .unwrap();
        let db = Database::in_memory();
        db.create_table_from_batches("t", vec![batch]).unwrap();
        db
    }

    fn strings(db: &Database, sql: &str) -> Vec<Option<String>> {
        let r = db.sql(sql).unwrap();
        let mut out = Vec::new();
        for b in &r.batches {
            for row in 0..b.row_count() {
                out.push(match b.column(0).get(row).unwrap() {
                    ScalarValue::Utf8(v) => v,
                    ScalarValue::Null => None,
                    other => panic!("expected text, got {other:?}"),
                });
            }
        }
        out
    }

    /// Which `HAVING` forms work?
    ///
    /// `HAVING COUNT(*) > 10` reports "unsupported function arg", which is
    /// ordinary SQL failing outright. Before fixing it, this maps what does and
    /// does not work, so the fix is aimed at the real boundary rather than at the
    /// one example that happened to be tried.
    #[test]
    #[ignore]
    fn having_forms() {
        use crate::storage::types::DataType;
        let batch = RecordBatch::new(
            Schema::new(vec![
                Field::required("dept", DataType::Utf8),
                Field::required("n", DataType::Int64),
            ]),
            vec![
                ColumnBuilder::utf8_from_str(
                    (0..100).map(|i| Some(["a", "b", "c"][i % 3])).collect(),
                ),
                ColumnBuilder::int64((0..100).map(Some).collect()),
            ],
        )
        .unwrap();
        let db = Database::in_memory();
        db.create_table_from_batches("t", vec![batch]).unwrap();

        for sql in [
            "SELECT dept, COUNT(*) c FROM t GROUP BY dept HAVING c > 10",
            "SELECT dept, COUNT(*) c FROM t GROUP BY dept HAVING COUNT(*) > 10",
            "SELECT dept, COUNT(n) c FROM t GROUP BY dept HAVING COUNT(n) > 10",
            "SELECT dept, SUM(n) s FROM t GROUP BY dept HAVING SUM(n) > 100",
            "SELECT dept, AVG(n) a FROM t GROUP BY dept HAVING AVG(n) > 10",
            "SELECT dept, COUNT(*) c FROM t GROUP BY dept HAVING COUNT(*) > 10 AND dept <> 'a'",
            "SELECT COUNT(*) FROM t HAVING COUNT(*) > 1",
        ] {
            match db.sql(sql) {
                Ok(r) => {
                    let rows: usize = r.batches.iter().map(|b| b.row_count()).sum();
                    println!("  ok   ({rows} rows)  {sql}");
                }
                Err(e) => println!("  FAIL  {e}
        {sql}"),
            }
        }
    }

    /// The shapes the benchmark suite still does not execute.
    ///
    /// `LIKE` and the scalar functions came out of this same question and were
    /// 29x to 400x off. These are what is left: `HAVING`, `EXTRACT`, `UNION`,
    /// `LIMIT`, `COUNT(DISTINCT)` and `STRING_AGG`, none of which appear in the
    /// suite.
    ///
    /// Each is paired with the closest thing that *is* measured, so a number means
    /// something. `HAVING` against the same `GROUP BY` without it; `LIMIT` against
    /// the same `ORDER BY` without it; `COUNT(DISTINCT)` against plain `COUNT`.
    ///
    /// Run with
    /// `cargo test --release --lib unbenchmarked_shapes -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn unbenchmarked_shapes() {
        use crate::storage::types::DataType;
        use std::time::Instant;

        const ROWS: usize = 100_000;
        let words = ["engineering", "sales", "human resources", "operations", "legal"];
        let batch = RecordBatch::new(
            Schema::new(vec![
                Field::required("dept", DataType::Utf8),
                Field::required("n", DataType::Int64),
                Field::required("score", DataType::Float64),
                Field::required("d", DataType::Date32),
            ]),
            vec![
                ColumnBuilder::utf8_from_str(
                    (0..ROWS).map(|i| Some(words[i % words.len()])).collect(),
                ),
                ColumnBuilder::int64((0..ROWS as i64).map(Some).collect()),
                ColumnBuilder::float64((0..ROWS).map(|i| Some((i % 1000) as f64)).collect()),
                ColumnBuilder::date32((0..ROWS).map(|i| Some(19000 + (i % 365) as i32)).collect()),
            ],
        )
        .unwrap();
        let db = Database::in_memory();
        db.create_table_from_batches("t", vec![batch]).unwrap();

        let time = |label: &str, sql: &str| -> Option<std::time::Duration> {
            for _ in 0..3 {
                if let Err(e) = db.sql(sql) {
                    println!("  {label:<40}unsupported: {e}");
                    return None;
                }
            }
            let mut best = std::time::Duration::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                std::hint::black_box(db.sql(sql).unwrap());
                best = best.min(t.elapsed());
            }
            println!("  {label:<40}{best:>9.2?}");
            Some(best)
        };

        println!("over {ROWS} rows, 5 distinct departments:
");

        println!("-- HAVING, against the same GROUP BY without it --");
        let gb = time("GROUP BY dept", "SELECT dept, COUNT(*) c FROM t GROUP BY dept").unwrap();
        if let Some(h) = time("... HAVING COUNT(*) > 10", "SELECT dept, COUNT(*) c FROM t GROUP BY dept HAVING COUNT(*) > 10") {
            println!("      {:.2}x the plain GROUP BY", h.as_secs_f64() / gb.as_secs_f64());
        }

        println!("
-- LIMIT, against the same ORDER BY without it --");
        let ob = time("ORDER BY score DESC", "SELECT score FROM t ORDER BY score DESC").unwrap();
        if let Some(l) = time("... LIMIT 10", "SELECT score FROM t ORDER BY score DESC LIMIT 10") {
            println!("      {:.2}x the full sort (should be far cheaper)", l.as_secs_f64() / ob.as_secs_f64());
        }

        println!("
-- COUNT(DISTINCT), against plain COUNT --");
        let c = time("COUNT(*)", "SELECT COUNT(*) FROM t").unwrap();
        for (label, sql) in [
            ("COUNT(DISTINCT dept)  5 values",  "SELECT COUNT(DISTINCT dept) FROM t"),
            ("COUNT(DISTINCT n)   100k values", "SELECT COUNT(DISTINCT n) FROM t"),
        ] {
            if let Some(d) = time(label, sql) {
                println!("      {:.0}x COUNT(*)", d.as_secs_f64() / c.as_secs_f64());
            }
        }

        println!("
-- the rest, against reading a column --");
        let base = time("SELECT n (reference)", "SELECT n FROM t").unwrap();
        for (label, sql) in [
            ("EXTRACT(YEAR FROM d)",  "SELECT EXTRACT(YEAR FROM d) FROM t"),
            ("YEAR(d)",               "SELECT YEAR(d) FROM t"),
            ("UNION of two halves",   "SELECT n FROM t WHERE n < 50000 UNION SELECT n FROM t WHERE n >= 50000"),
            ("UNION ALL",             "SELECT n FROM t WHERE n < 50000 UNION ALL SELECT n FROM t WHERE n >= 50000"),
            ("STRING_AGG(dept)",      "SELECT STRING_AGG(dept, ',') FROM t"),
        ] {
            if let Some(d) = time(label, sql) {
                println!("      {:.0}x the reference, {:.0} ns per row",
                    d.as_secs_f64() / base.as_secs_f64(),
                    d.as_nanos() as f64 / ROWS as f64);
            }
        }
    }

    /// `CASE` and `COALESCE` on the fast path must answer what the general path
    /// answered, and the shapes the fast path declines must still work.
    ///
    /// The declining cases matter more than the accepted ones here: a fast path
    /// that quietly handles a shape it models wrongly -- several `WHEN`s, a
    /// missing `ELSE`, a type mismatch -- is worse than one that never fires.
    #[test]
    fn dense_case_and_coalesce_match_the_general_path() {
        use crate::storage::types::DataType;

        let batch = RecordBatch::new(
            Schema::new(vec![
                Field::nullable("n", DataType::Int64),
                Field::nullable("s", DataType::Utf8),
            ]),
            vec![
                ColumnBuilder::int64(vec![Some(10), Some(60), None, Some(100)]),
                ColumnBuilder::utf8_from_str(vec![Some("a"), Some("b"), None, Some("c")]),
            ],
        )
        .unwrap();
        let db = Database::in_memory();
        db.create_table_from_batches("t", vec![batch]).unwrap();

        let ints = |sql: &str| -> Vec<Option<i64>> {
            let r = db.sql(sql).unwrap();
            let b = &r.batches[0];
            (0..b.row_count())
                .map(|i| match b.column(0).get(i).unwrap() {
                    ScalarValue::Null => None,
                    v => v.as_i64(),
                })
                .collect()
        };
        let texts = |sql: &str| -> Vec<Option<String>> {
            let r = db.sql(sql).unwrap();
            let b = &r.batches[0];
            (0..b.row_count())
                .map(|i| match b.column(0).get(i).unwrap() {
                    ScalarValue::Utf8(v) => v,
                    ScalarValue::Null => None,
                    other => panic!("expected text, got {other:?}"),
                })
                .collect()
        };

        // The shape the fast path takes. The NULL row fails `n > 50`, so it gets
        // the ELSE arm -- which is what the general path does too.
        assert_eq!(
            ints("SELECT CASE WHEN n > 50 THEN 1 ELSE 0 END FROM t"),
            vec![Some(0), Some(1), Some(0), Some(1)],
        );
        assert_eq!(
            texts("SELECT CASE WHEN n > 50 THEN 'big' ELSE 'small' END FROM t"),
            vec![Some("small".into()), Some("big".into()), Some("small".into()), Some("big".into())],
        );

        // COALESCE fills only the NULLs.
        assert_eq!(ints("SELECT COALESCE(n, 0) FROM t"), vec![Some(10), Some(60), Some(0), Some(100)]);
        assert_eq!(ints("SELECT COALESCE(n, -1) FROM t"), vec![Some(10), Some(60), Some(-1), Some(100)]);

        // Shapes the fast path declines, which must answer identically.
        //
        // No ELSE: unmatched rows are NULL, a different column shape.
        assert_eq!(
            ints("SELECT CASE WHEN n > 50 THEN 1 END FROM t"),
            vec![None, Some(1), None, Some(1)],
        );
        // Two WHENs, where first-match matters: 100 satisfies both.
        assert_eq!(
            ints("SELECT CASE WHEN n > 50 THEN 1 WHEN n > 5 THEN 2 ELSE 3 END FROM t"),
            vec![Some(2), Some(1), Some(3), Some(1)],
        );
        // A non-literal arm.
        assert_eq!(
            ints("SELECT CASE WHEN n > 50 THEN n ELSE 0 END FROM t"),
            vec![Some(0), Some(60), Some(0), Some(100)],
        );
        // COALESCE over two columns rather than a literal.
        assert_eq!(
            texts("SELECT COALESCE(s, 'none') FROM t"),
            vec![Some("a".into()), Some("b".into()), Some("none".into()), Some("c".into())],
        );
    }

    /// The dense numeric path must answer what the per-row path answered, and in
    /// the same *type*.
    ///
    /// `eval_function` keeps `ABS` integral for an integer column but answers
    /// `ROUND` in floats whatever it was given. A fast path that quietly widened
    /// `ABS(n)` to a float would pass any test that only compared magnitudes, so
    /// the column's own type is asserted here.
    #[test]
    fn dense_numeric_functions_match_the_general_path() {
        use crate::storage::types::DataType;

        let batch = RecordBatch::new(
            Schema::new(vec![
                Field::nullable("n", DataType::Int64),
                Field::nullable("f", DataType::Float64),
            ]),
            vec![
                ColumnBuilder::int64(vec![Some(-7), Some(0), None, Some(7)]),
                ColumnBuilder::float64(vec![Some(-2.345), Some(0.0), None, Some(2.345)]),
            ],
        )
        .unwrap();
        let db = Database::in_memory();
        db.create_table_from_batches("t", vec![batch]).unwrap();

        let col = |sql: &str| -> (DataType, Vec<Option<f64>>) {
            let r = db.sql(sql).unwrap();
            let b = &r.batches[0];
            let dt = b.schema.fields[0].data_type.clone();
            let vals = (0..b.row_count())
                .map(|i| match b.column(0).get(i).unwrap() {
                    ScalarValue::Null => None,
                    v => v.as_f64(),
                })
                .collect();
            (dt, vals)
        };

        // ABS keeps the integer column integral.
        let (dt, v) = col("SELECT ABS(n) FROM t");
        assert_eq!(dt, DataType::Int64, "ABS must not widen an integer column");
        assert_eq!(v, vec![Some(7.0), Some(0.0), None, Some(7.0)]);

        // SIGN likewise.
        let (dt, v) = col("SELECT SIGN(n) FROM t");
        assert_eq!(dt, DataType::Int64);
        assert_eq!(v, vec![Some(-1.0), Some(0.0), None, Some(1.0)]);

        // ROUND answers in floats even for an integer column.
        let (dt, _) = col("SELECT ROUND(n) FROM t");
        assert_eq!(dt, DataType::Float64, "ROUND answers in floats");

        // Float arms, including the digits argument.
        assert_eq!(col("SELECT ABS(f) FROM t").1, vec![Some(2.345), Some(0.0), None, Some(2.345)]);
        assert_eq!(col("SELECT ROUND(f, 2) FROM t").1, vec![Some(-2.35), Some(0.0), None, Some(2.35)]);
        assert_eq!(col("SELECT ROUND(f) FROM t").1, vec![Some(-2.0), Some(0.0), None, Some(2.0)]);
        assert_eq!(col("SELECT CEIL(f) FROM t").1, vec![Some(-2.0), Some(0.0), None, Some(3.0)]);
        assert_eq!(col("SELECT FLOOR(f) FROM t").1, vec![Some(-3.0), Some(0.0), None, Some(2.0)]);

        // A shape the fast path declines goes to the general path unchanged --
        // including where that path errors. `ROUND(f, n)` with a NULL in `n`
        // reports "function argument 1 is NULL" rather than answering NULL, which
        // is not what SQL says but is what this engine has always done. Pinned so
        // the fast path cannot quietly start answering it differently.
        match db.sql("SELECT ROUND(f, n) FROM t") {
            Err(e) => assert!(
                e.to_string().contains("NULL"),
                "expected the existing NULL-argument error, got: {e}"
            ),
            Ok(_) => panic!("this now answers; decide deliberately what it should answer"),
        }

        // The same shape without a NULL does answer, and still avoids the fast
        // path because the second argument is a column.
        let r = db.sql("SELECT ROUND(f, 1) FROM t WHERE n IS NOT NULL").unwrap();
        assert_eq!(r.batches[0].row_count(), 3);
    }

    /// The dictionary path must answer exactly what the per-row path answered.
    #[test]
    fn dict_mapped_functions_match_the_general_path() {
        let db = db_with(vec![Some("  Padded  "), Some("sales"), None, Some("SALES"), Some("")]);

        assert_eq!(
            strings(&db, "SELECT UPPER(s) FROM t"),
            vec![Some("  PADDED  ".into()), Some("SALES".into()), None, Some("SALES".into()), Some("".into())],
        );
        assert_eq!(
            strings(&db, "SELECT LOWER(s) FROM t"),
            vec![Some("  padded  ".into()), Some("sales".into()), None, Some("sales".into()), Some("".into())],
        );
        assert_eq!(
            strings(&db, "SELECT TRIM(s) FROM t"),
            vec![Some("Padded".into()), Some("sales".into()), None, Some("SALES".into()), Some("".into())],
        );
        assert_eq!(
            strings(&db, "SELECT LTRIM(s) FROM t"),
            vec![Some("Padded  ".into()), Some("sales".into()), None, Some("SALES".into()), Some("".into())],
        );
        assert_eq!(
            strings(&db, "SELECT CONCAT(s, '!') FROM t"),
            // CONCAT skips a NULL argument rather than propagating it, so the NULL
            // row is "!" and not NULL. That is this engine's existing answer.
            vec![Some("  Padded  !".into()), Some("sales!".into()), Some("!".into()), Some("SALES!".into()), Some("!".into())],
        );

        // LENGTH answers with numbers, so it leaves the dictionary behind.
        let r = db.sql("SELECT LENGTH(s) FROM t").unwrap();
        let lens: Vec<Option<i64>> = (0..r.batches[0].row_count())
            .map(|row| r.batches[0].column(0).get(row).unwrap().as_i64())
            .collect();
        assert_eq!(lens, vec![Some(10), Some(5), None, Some(5), Some(0)]);
    }

    /// `UPPER` maps two different strings onto one, and the result must not carry
    /// a dictionary holding that string twice.
    ///
    /// Nothing structurally forbids duplicates -- `from_parts` only checks that
    /// codes index the dictionary -- but anything that groups by code rather than
    /// by string would then report two groups for one value. This is what the
    /// dedup in `dict_mapped_function` is for, and the test states it as a result
    /// rather than as an implementation detail.
    #[test]
    fn a_collapsing_transform_still_groups_as_one_value() {
        let db = db_with(vec![Some("sales"), Some("SALES"), Some("Sales"), Some("legal")]);

        let r = db.sql("SELECT UPPER(s) AS u, COUNT(*) AS n FROM t GROUP BY UPPER(s)").unwrap();
        let mut groups: Vec<(String, i64)> = Vec::new();
        for b in &r.batches {
            for row in 0..b.row_count() {
                let name = match b.column(0).get(row).unwrap() {
                    ScalarValue::Utf8(Some(v)) => v,
                    other => panic!("expected text, got {other:?}"),
                };
                groups.push((name, b.column(1).get(row).unwrap().as_i64().unwrap()));
            }
        }
        groups.sort();
        assert_eq!(groups, vec![("LEGAL".to_string(), 1), ("SALES".to_string(), 3)]);
    }

    /// Shapes the fast path declines still work, through the general path.
    #[test]
    fn unsupported_shapes_fall_through() {
        let db = db_with(vec![Some("sales"), None]);
        // Two columns, a nested call, and a numeric function: none are the
        // single-text-column shape, and all must still answer.
        assert_eq!(strings(&db, "SELECT UPPER(LOWER(s)) FROM t"), vec![Some("SALES".into()), None]);
        assert_eq!(strings(&db, "SELECT CONCAT(s, s) FROM t"), vec![Some("salessales".into()), Some("".into())]);
    }
}

#[cfg(test)]
mod distinct_perf {
    use crate::db::Database;
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::RecordBatch,
        types::DataType,
    };
    use std::time::Instant;

    /// Two shapes the benchmark suite never runs: a partitioned window, and
    /// `IN (...)`.
    ///
    /// The window benchmark is `ROW_NUMBER() OVER (ORDER BY score DESC)` with no
    /// `PARTITION BY` at all, and there is no `IN` benchmark of any kind. Both are
    /// ordinary SQL, and the pattern of this session is that an unmeasured shape
    /// stays on whatever path it was first given.
    ///
    /// Numeric and text versions of each, so a gap between them points at a
    /// missing typed path rather than at the feature being expensive.
    ///
    /// Run with
    /// `cargo test --release --lib unmeasured_shapes -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn unmeasured_shapes() {
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
                Field::required("score", DataType::Float64),
            ]),
            vec![
                ColumnBuilder::utf8_from_str(
                    (0..ROWS)
                        .map(|i| Some(["eng", "sales", "hr", "ops", "legal"][i % 5]))
                        .collect(),
                ),
                ColumnBuilder::int64((0..ROWS).map(|i| Some((i % 5) as i64)).collect()),
                ColumnBuilder::float64((0..ROWS).map(|i| Some((i % 100) as f64)).collect()),
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
            println!("  {label:>44}: {best:>9.2?}");
            best
        };

        println!("over {ROWS} rows:");
        let wn = time(
            "ROW_NUMBER, no PARTITION BY (benchmarked)",
            "SELECT ROW_NUMBER() OVER (ORDER BY score DESC) AS rn FROM t",
        );
        let wi = time(
            "ROW_NUMBER, PARTITION BY bucket (Int64)",
            "SELECT ROW_NUMBER() OVER (PARTITION BY bucket ORDER BY score DESC) AS rn FROM t",
        );
        let wt = time(
            "ROW_NUMBER, PARTITION BY dept (Utf8)",
            "SELECT ROW_NUMBER() OVER (PARTITION BY dept ORDER BY score DESC) AS rn FROM t",
        );
        println!("    partitioning by an integer costs {:.1}x the unpartitioned form", wi.as_secs_f64() / wn.as_secs_f64());
        println!("    partitioning by text costs {:.1}x the integer", wt.as_secs_f64() / wi.as_secs_f64());

        let ii = time("WHERE bucket IN (1,2,3)", "SELECT COUNT(*) FROM t WHERE bucket IN (1,2,3)");
        let it = time(
            "WHERE dept IN ('eng','hr','ops')",
            "SELECT COUNT(*) FROM t WHERE dept IN ('eng','hr','ops')",
        );
        let ie = time("WHERE bucket = 1 (for reference)", "SELECT COUNT(*) FROM t WHERE bucket = 1");
        println!("    IN over integers costs {:.1}x a single equality", ii.as_secs_f64() / ie.as_secs_f64());
        println!("    IN over text costs {:.1}x the integer IN", it.as_secs_f64() / ii.as_secs_f64());
    }

    /// What does `DISTINCT` cost on a text column against a numeric one?
    ///
    /// The benchmark suite runs `SELECT DISTINCT score`, a `Float64`. `GROUP BY`
    /// in that suite *is* on a text column and is one of this engine's strongest
    /// points, so the dictionary machinery exists -- the question is whether
    /// `DISTINCT` reaches it.
    ///
    /// Run with
    /// `cargo test --release --lib distinct_by_column_type -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn distinct_by_column_type() {
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
        let db = Database::in_memory();
        db.create_table_from_batches("t", vec![batch]).unwrap();

        let time = |label: &str, sql: &str| {
            for _ in 0..3 { db.sql(sql).unwrap(); }
            let mut best = std::time::Duration::MAX;
            for _ in 0..9 {
                let t = Instant::now();
                std::hint::black_box(db.sql(sql).unwrap());
                best = best.min(t.elapsed());
            }
            println!("  {label:>30}: {best:>9.2?}");
            best
        };

        println!("DISTINCT over {ROWS} rows:");
        let f = time("score (Float64)", "SELECT DISTINCT score FROM t");
        let u = time("dept (Utf8)", "SELECT DISTINCT dept FROM t");
        println!("  text costs {:.2}x the float", u.as_secs_f64() / f.as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        query::expr::{BinaryOp, Expr},
        storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::{RecordBatch, Table},
            types::{DataType, ScalarValue},
        },
    };

    /// The typed DISTINCT path must agree with the general one.
    ///
    /// Two implementations of the same semantics is how a 100x-slow join went
    /// unnoticed in this codebase, so the fast path is pinned to the slow one
    /// rather than just to a hand-written expectation.
    /// Text dedup numbers the distinct strings globally, so batches with
    /// different dictionaries have to agree on what is a duplicate.
    ///
    /// The second batch's dictionary is deliberately in a different order from
    /// the first's, which is what catches treating a local code as an identity.
    #[test]
    fn text_distinct_spans_dictionaries() {
        use crate::storage::{column::ColumnBuilder, schema::{Field, Schema}, types::DataType};

        let schema = Schema::new(vec![Field::required("s", DataType::Utf8)]);
        let mk = |vals: Vec<&str>| {
            RecordBatch::new(
                schema.clone(),
                vec![ColumnBuilder::utf8_from_str(vals.into_iter().map(Some).collect())],
            )
            .unwrap()
        };
        // First batch sees a, b; second sees b, a, c -- so "b" is code 0 there
        // and code 1 here.
        let batches = vec![mk(vec!["a", "b", "a"]), mk(vec!["b", "a", "c"])];

        let keeps = dedup_single_primitive(&batches).expect("text should take the typed path");
        assert_eq!(keeps[0], vec![true, true, false], "a, b kept; second a dropped");
        assert_eq!(
            keeps[1],
            vec![false, false, true],
            "b and a already seen in the first batch; c is new",
        );

        // And the whole thing end to end keeps first occurrences in order.
        let out = dedup_batches(batches);
        let seen: Vec<String> = out
            .iter()
            .flat_map(|b| {
                (0..b.row_count()).map(move |r| match b.column(0).get(r).unwrap() {
                    ScalarValue::Utf8(Some(v)) => v,
                    other => panic!("unexpected {other:?}"),
                })
            })
            .collect();
        assert_eq!(seen, vec!["a", "b", "c"]);
    }

    #[test]
    fn typed_distinct_agrees_with_the_general_path() {
        let schema = Schema::new(vec![Field::required("v", DataType::Int64)]);
        // Split across batches so a duplicate must be caught against an earlier one.
        let batches: Vec<RecordBatch> = [vec![3i64, 1, 3, 2], vec![1, 4, 2], vec![5, 5]]
            .iter()
            .map(|vals| {
                RecordBatch::new(
                    schema.clone(),
                    vec![ColumnBuilder::int64(vals.iter().map(|&v| Some(v)).collect())],
                )
                .unwrap()
            })
            .collect();

        let fast = dedup_single_primitive(&batches).expect("Int64 should take the typed path");
        let fast_rows: Vec<i64> = collect_kept(&batches, &fast);

        // First occurrence wins, in scan order.
        assert_eq!(fast_rows, vec![3, 1, 2, 4, 5]);

        let out = dedup_batches(batches.clone());
        let via_dedup: Vec<i64> = out
            .iter()
            .flat_map(|b| {
                (0..b.row_count()).map(move |r| match b.column(0).get(r).unwrap() {
                    ScalarValue::Int64(Some(v)) => v,
                    other => panic!("unexpected {other:?}"),
                })
            })
            .collect();
        assert_eq!(via_dedup, fast_rows);
    }

    /// The parallel dedup must keep the same rows, in the same order, as the serial one.
    ///
    /// Every other DISTINCT test here is a handful of rows, so none of them reach
    /// par_dedup_rows() and the chunked path went unexercised. Which occurrence
    /// survives is a global fact, so a chunk boundary falling between two copies of
    /// a value is exactly the case that can go wrong.
    #[test]
    fn parallel_dedup_matches_the_serial_result() {
        let schema = Schema::new(vec![Field::required("v", DataType::Int64)]);
        // Well over par_dedup_rows(), split into many batches so chunks land mid-run.
        const BATCHES: usize = 200;
        const PER_BATCH: usize = 1_000;
        const DISTINCT: i64 = 97; // coprime with the batch size, so repeats straddle chunks

        let batches: Vec<RecordBatch> = (0..BATCHES)
            .map(|b| {
                let vals: Vec<Option<i64>> = (0..PER_BATCH)
                    .map(|i| Some(((b * PER_BATCH + i) as i64) % DISTINCT))
                    .collect();
                RecordBatch::new(schema.clone(), vec![ColumnBuilder::int64(vals)]).unwrap()
            })
            .collect();
        assert!(BATCHES * PER_BATCH > par_dedup_rows(), "test must reach the parallel path");

        let parallel = dedup_single_primitive(&batches).expect("Int64 takes the typed path");

        // Same computation, forced through one chunk by handing it a single batch.
        let flat: Vec<Option<i64>> = (0..BATCHES * PER_BATCH)
            .map(|i| Some((i as i64) % DISTINCT))
            .collect();
        let single = vec![RecordBatch::new(
            schema,
            vec![ColumnBuilder::int64(flat)],
        )
        .unwrap()];
        let serial = dedup_single_primitive(&single).expect("Int64 takes the typed path");

        let kept_parallel: Vec<bool> = parallel.into_iter().flatten().collect();
        let kept_serial: Vec<bool> = serial.into_iter().flatten().collect();
        assert_eq!(
            kept_parallel, kept_serial,
            "chunked dedup kept different rows than the single-chunk one"
        );
        assert_eq!(
            kept_serial.iter().filter(|&&k| k).count(),
            DISTINCT as usize,
            "wrong number of distinct values"
        );
    }

    /// Floats need the same NaN and signed-zero handling the general path has.
    #[test]
    fn typed_distinct_normalises_floats_like_row_hash() {
        let schema = Schema::new(vec![Field::required("v", DataType::Float64)]);
        let batch = RecordBatch::new(
            schema,
            vec![ColumnBuilder::float64(vec![
                Some(0.0),
                Some(-0.0),      // same value as 0.0
                Some(f64::NAN),
                Some(f64::NAN),  // NaN must dedupe against itself here
                Some(1.5),
            ])],
        )
        .unwrap();
        let batches = vec![batch];
        let keeps = dedup_single_primitive(&batches).expect("Float64 should take the typed path");
        assert_eq!(keeps[0], vec![true, false, true, false, true]);
    }

    /// NULLs and multi-column inputs stay on the general path.
    #[test]
    fn typed_distinct_declines_what_it_cannot_handle() {
        let nullable = Schema::new(vec![Field::nullable("v", DataType::Int64)]);
        let with_null = vec![RecordBatch::new(
            nullable,
            vec![ColumnBuilder::int64(vec![Some(1), None, Some(1)])],
        )
        .unwrap()];
        assert!(dedup_single_primitive(&with_null).is_none());

        let two_cols = Schema::new(vec![
            Field::required("a", DataType::Int64),
            Field::required("b", DataType::Int64),
        ]);
        let wide = vec![RecordBatch::new(
            two_cols,
            vec![
                ColumnBuilder::int64(vec![Some(1)]),
                ColumnBuilder::int64(vec![Some(2)]),
            ],
        )
        .unwrap()];
        assert!(dedup_single_primitive(&wide).is_none());
    }

    fn collect_kept(batches: &[RecordBatch], keeps: &[Vec<bool>]) -> Vec<i64> {
        let mut out = Vec::new();
        for (b, keep) in batches.iter().zip(keeps) {
            for (r, &k) in keep.iter().enumerate() {
                if k {
                    match b.column(0).get(r).unwrap() {
                        ScalarValue::Int64(Some(v)) => out.push(v),
                        other => panic!("unexpected {other:?}"),
                    }
                }
            }
        }
        out
    }

    fn make_table() -> Table {
        let schema = Schema::new(vec![
            Field::required("id",    DataType::Int64),
            Field::nullable("name",  DataType::Utf8),
            Field::nullable("score", DataType::Float64),
        ]);

        let batch = RecordBatch::new(schema.clone(), vec![
            ColumnBuilder::int64(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]),
            ColumnBuilder::utf8_from_str(vec![
                Some("alice"), Some("bob"), Some("carol"), Some("dave"), Some("eve"),
            ]),
            ColumnBuilder::float64(vec![
                Some(9.5), Some(8.0), Some(7.2), Some(6.8), Some(10.0),
            ]),
        ]).unwrap();

        let mut table = Table::new("users", schema);
        table.append_batch(batch).unwrap();
        table
    }

    #[test]
    fn select_all() {
        let table   = make_table();
        let query   = SelectQuery::scan_all();
        let results = execute_select(&table, &query).unwrap();
        let total: usize = results.iter().map(|b| b.row_count()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn select_with_filter() {
        let table = make_table();
        let query = SelectQuery::scan_all().with_filter(Expr::BinaryOp {
            left:  Box::new(Expr::Column("score".into())),
            op:    BinaryOp::Gt,
            right: Box::new(Expr::Literal(ScalarValue::Float64(Some(8.0)))),
        });
        let results = execute_select(&table, &query).unwrap();
        let total: usize = results.iter().map(|b| b.row_count()).sum();
        assert_eq!(total, 2); // alice (9.5) and eve (10.0)
    }

    #[test]
    fn select_with_limit() {
        let table   = make_table();
        let query   = SelectQuery::scan_all().with_limit(2);
        let results = execute_select(&table, &query).unwrap();
        let total: usize = results.iter().map(|b| b.row_count()).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn select_with_offset() {
        let table   = make_table();
        let query   = SelectQuery::scan_all().with_offset(3);
        let results = execute_select(&table, &query).unwrap();
        let total: usize = results.iter().map(|b| b.row_count()).sum();
        assert_eq!(total, 2); // dave and eve
    }

    #[test]
    fn select_projection() {
        let table = make_table();
        let query = SelectQuery::scan_all().with_columns(vec![
            SelectExpr::column("id"),
            SelectExpr::column("name"),
        ]);
        let results = execute_select(&table, &query).unwrap();
        assert_eq!(results[0].column_count(), 2);
        assert_eq!(results[0].schema.fields[0].name, "id");
        assert_eq!(results[0].schema.fields[1].name, "name");
    }

    #[test]
    fn display_results_format() {
        let table   = make_table();
        let query   = SelectQuery::scan_all().with_limit(2);
        let results = execute_select(&table, &query).unwrap();
        let output  = display_results(&results);
        assert!(output.contains("alice"));
        assert!(output.contains("bob"));
        assert!(output.contains("(2 rows)"));
    }
}
