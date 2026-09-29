use std::collections::HashSet;
use std::sync::Arc;

use smallvec::SmallVec;

use rayon::prelude::*;

use super::hash::{FastMap, FastSet};

use crate::storage::{
    column::{BooleanColumn, Column, PrimitiveColumn, Utf8Column},
    schema::{Field, Schema},
    table::RecordBatch,
    types::{DataType, FalconError, FalconResult, ScalarValue},
};

/// Minimum left rows before a probe splits over cores.
///
/// Shared by the inner-join probe and the semi/anti one: both decide each left row
/// independently, so one threshold covers both rather than each growing its own.
/// Measured at 65_536 on a machine reporting 20 threads, so ~3.3k each.
fn par_probe_rows() -> usize {
    crate::query::tuning::par_min_rows(3_277)
}

// ─── Join type ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    /// All matched rows + unmatched rows from both sides (NULLs on the missing side).
    Full,
    /// Cartesian product — every left row paired with every right row, no ON condition.
    Cross,
    /// Return left rows that have at least one match on the right.
    Semi,
    /// Return left rows that have NO match on the right.
    Anti,
}

// ─── Hash join ────────────────────────────────────────────────────────────────

/// Execute a semi-join or anti-join.
///
/// Returns only left columns. Semi keeps left rows that match the right;
/// Anti keeps left rows with no right match.
pub fn execute_hash_semi_anti_join(
    left_batches:  &[RecordBatch],
    right_batches: &[RecordBatch],
    left_keys:     &[&str],
    right_keys:    &[&str],
    anti:          bool,
) -> FalconResult<RecordBatch> {
    if left_batches.is_empty() {
        // No left rows → result is always empty.
        return merge_batches(left_batches);
    }
    if right_batches.is_empty() {
        // No right rows → semi returns nothing, anti returns all left.
        return if anti { merge_batches(left_batches) } else {
            // Return empty batch with left schema.
            let left = merge_batches(left_batches)?;
            left.filter_by_mask(&vec![false; left.row_count()])
        };
    }

    let right_merged = merge_batches(right_batches)?;
    let left_merged  = merge_batches(left_batches)?;
    let typed = int_keys_usable(&right_merged, right_keys, &left_merged, left_keys);
    let (hash_table, bloom) = build_hash_table(&right_merged, right_keys, typed)?;

    let left_key_cols: Vec<&Column> = left_keys.iter()
        .map(|k| left_merged.column_by_name(k))
        .collect::<FalconResult<_>>()?;
    let right_key_cols: Vec<&Column> = right_keys.iter()
        .map(|k| right_merged.column_by_name(k))
        .collect::<FalconResult<_>>()?;

    // Same guard as the inner-join probe: only filter when the key types match.
    let bloom = bloom
        .as_ref()
        .filter(|_| filter_is_safe(&right_key_cols, &left_key_cols));

    // Each left row is decided independently — semi/anti keep or drop it and record
    // nothing shared — so the probe splits over row ranges. The inner join's probe
    // was parallelised earlier; this one was left serial and is the same shape.
    let n_left = left_merged.row_count();
    let decide = |l_row: usize| -> FalconResult<bool> {
        // A definite miss needs no string key: semi drops the row, anti keeps it.
        if let Some(f) = bloom {
            let h = match &hash_table {
                JoinKeyMap::Int(_) => int_key_hash(int_key(left_key_cols[0], l_row)),
                JoinKeyMap::Str(_) => row_key_hash(&left_key_cols, l_row),
            };
            if !f.might_contain(h) {
                return Ok(anti);
            }
        }
        let matched = match &hash_table {
            JoinKeyMap::Int(t) => t.contains(int_key(left_key_cols[0], l_row)),
            JoinKeyMap::Str(t) => match single_text_key(&left_key_cols, l_row) {
                Some(key) => t.contains_key(key),
                None if matches!(left_key_cols.as_slice(), [Column::Utf8(_)]) => false,
                None => t.contains_key(&row_key(&left_key_cols, l_row)?),
            },
        };
        Ok(if anti { !matched } else { matched })
    };

    let keep: Vec<bool> = if n_left >= par_probe_rows() {
        let chunk = (n_left / rayon::current_num_threads().max(1)).max(8_192);
        let ranges: Vec<(usize, usize)> = (0..n_left)
            .step_by(chunk)
            .map(|s| (s, (s + chunk).min(n_left)))
            .collect();
        let parts: FalconResult<Vec<Vec<bool>>> = ranges
            .par_iter()
            .map(|&(start, end)| (start..end).map(&decide).collect())
            .collect();
        // Ranges are visited in order, so the mask lines up with the input rows.
        parts?.concat()
    } else {
        (0..n_left).map(&decide).collect::<FalconResult<_>>()?
    };

    left_merged.filter_by_mask(&keep)
}

/// Execute a hash join between two sets of batches on the given key columns.
///
/// Algorithm:
///   1. Build phase — hash the *right* side into a HashMap<key → Vec<row_idx>>.
///   2. Probe phase — iterate *left* rows, look up matches in the hash table.
/// Execute a CROSS JOIN (cartesian product) — no key columns required.
pub fn execute_cross_join(
    left_batches:  &[RecordBatch],
    right_batches: &[RecordBatch],
) -> FalconResult<RecordBatch> {
    if left_batches.is_empty() || right_batches.is_empty() {
        return if left_batches.is_empty() {
            merge_batches(left_batches)
        } else {
            merge_batches(right_batches)
        };
    }
    let left  = merge_batches(left_batches)?;
    let right = merge_batches(right_batches)?;
    let n = left.row_count() * right.row_count();
    let mut left_indices:  Vec<u32> = Vec::with_capacity(n);
    let mut right_indices: Vec<u32> = Vec::with_capacity(n);
    for l in 0..left.row_count() {
        for r in 0..right.row_count() {
            left_indices.push(l as u32);
            right_indices.push(r as u32);
        }
    }
    build_output(&left, &right, &left_indices, &right_indices, None)
}

pub fn execute_hash_join(
    left_batches:  &[RecordBatch],
    right_batches: &[RecordBatch],
    left_keys:     &[&str],
    right_keys:    &[&str],
    join_type:     JoinType,
) -> FalconResult<RecordBatch> {
    execute_hash_join_projected(
        left_batches, right_batches, left_keys, right_keys, join_type, None,
    )
}

/// Hash join that materialises only the output columns in `keep`.
///
/// `keep` must list every column any later stage reads, under either its source
/// name or its post-join name; `None` means "build them all", which is what every
/// caller that cannot prove the set should pass. Building a column costs a cache
/// miss per output row against the larger input, so for `SELECT COUNT(*)` over a
/// join — which reads no columns at all — this is most of the query.
pub fn execute_hash_join_projected(
    left_batches:  &[RecordBatch],
    right_batches: &[RecordBatch],
    left_keys:     &[&str],
    right_keys:    &[&str],
    join_type:     JoinType,
    keep:          Option<&HashSet<String>>,
) -> FalconResult<RecordBatch> {
    if left_batches.is_empty() || right_batches.is_empty() {
        // FULL OUTER: even if one side is empty, return the other with NULLs.
        if join_type == JoinType::Full {
            let (left, right) = if left_batches.is_empty() {
                (merge_batches(right_batches)?, merge_batches(right_batches)?)
            } else {
                (merge_batches(left_batches)?, merge_batches(left_batches)?)
            };
            let n = if left_batches.is_empty() { right.row_count() } else { left.row_count() };
            let (li, ri): (Vec<u32>, Vec<u32>) = if left_batches.is_empty() {
                (vec![NO_ROW; n], (0..n as u32).collect())
            } else {
                ((0..n as u32).collect(), vec![NO_ROW; n])
            };
            return build_output(&left, &right, &li, &ri, None);
        }
        return Err(FalconError::InvalidQuery("join requires non-empty inputs".into()));
    }
    if left_keys.len() != right_keys.len() || left_keys.is_empty() {
        return Err(FalconError::InvalidQuery("join key lists must have the same length".into()));
    }

    // Concatenating a side copies every column it has. Columns that the join will
    // not emit are copied and then thrown away — at 1M rows that was two thirds of
    // the left-hand merge, and it is why `SUM(k)` measured *faster* than `COUNT(*)`
    // on the same join: both built one output column, but both also merged three.
    //
    // The key columns have to survive because the probe reads them.
    // Output names depend on which columns each side *has*: a right column keeps its
    // name unless the left side also has one by that name, in which case it becomes
    // `right.x`. Trimming a side could therefore rename a column that the query then
    // fails to find, so when the two schemas share any name, nothing is trimmed.
    let names_collide = match (left_batches.first(), right_batches.first()) {
        (Some(l), Some(r)) => {
            let left_names: HashSet<&str> =
                l.schema.fields.iter().map(|f| f.name.as_str()).collect();
            r.schema.fields.iter().any(|f| left_names.contains(f.name.as_str()))
        }
        _ => false,
    };
    let trim = keep.filter(|_| !names_collide);

    let left_slim  = project_for_join(left_batches, left_keys, trim);
    let right_slim = project_for_join(right_batches, right_keys, trim);

    let right_merged = merge_batches(&right_slim)?;

    // Checked per batch rather than on a merged left side, so the hash table can be
    // built before deciding whether the left side needs merging at all. Requiring
    // every batch to qualify is the same condition: `null_count` sums over a concat.
    let typed = left_slim
        .iter()
        .all(|b| int_keys_usable(&right_merged, right_keys, b, left_keys));
    let (hash_table, bloom) = build_hash_table(&right_merged, right_keys, typed)?;

    // INNER and LEFT decide each left row independently and never look at a row
    // outside its own batch, so the left side does not have to be one batch. Merging
    // it is pure overhead, and measurement says the overhead is not the copy:
    // concatenating 16 batches into a fresh 8 MB Int64 column costs 2.16 ms, the
    // same copy into an already-faulted buffer costs 509 us. Three quarters of the
    // merge is the operating system handing out pages, at roughly 2.5 GB/s against
    // a machine that reads at 57 GB/s.
    //
    // Probing in place skips that allocation and its copy. What is concatenated
    // instead is the *output*, which for a selective join is far smaller than the
    // probe side — 100k rows where the merge moved 1M.
    if matches!(join_type, JoinType::Inner | JoinType::Left) && left_slim.len() > 1 {
        return probe_in_place(
            &left_slim,
            &right_merged,
            left_keys,
            right_keys,
            &hash_table,
            bloom.as_ref(),
            join_type,
            keep,
        );
    }

    let left_merged = merge_batches(&left_slim)?;
    probe(
        &left_merged,
        &right_merged,
        left_keys,
        right_keys,
        &hash_table,
        bloom.as_ref(),
        join_type,
        keep,
    )
}

/// Probe the left side without concatenating it.
///
/// Work is split into row ranges that never cross a batch boundary, so the split
/// is independent of how the reader happened to batch the input: one huge batch
/// still spreads over every core, and many small ones do not each pay for a
/// rayon hand-off.
///
/// Output order matches the merged path exactly — units are produced in batch
/// order and, within a batch, in row order, which is the order a concatenated
/// left side would have had. `probe_in_place_matches_the_merged_contract` pins
/// that, because two implementations of one semantics have already cost this
/// engine a 100x regression that nothing caught.
#[allow(clippy::too_many_arguments)]
fn probe_in_place(
    left_batches: &[RecordBatch],
    right:        &RecordBatch,
    left_keys:    &[&str],
    right_keys:   &[&str],
    hash_table:   &JoinKeyMap,
    bloom:        Option<&BuildFilter>,
    join_type:    JoinType,
    keep:         Option<&HashSet<String>>,
) -> FalconResult<RecordBatch> {
    let right_key_cols: Vec<&Column> = right_keys.iter()
        .map(|k| right.column_by_name(k))
        .collect::<FalconResult<_>>()?;

    let total: usize = left_batches.iter().map(|b| b.row_count()).sum();
    let target = (total / rayon::current_num_threads().max(1)).max(8_192);
    let mut units: Vec<(usize, usize, usize)> = Vec::new();
    for (bi, b) in left_batches.iter().enumerate() {
        let n = b.row_count();
        let mut start = 0;
        while start < n {
            let end = (start + target).min(n);
            units.push((bi, start, end));
            start = end;
        }
    }
    // Every batch was empty. Fall back so the empty result still carries the
    // schema the merged path would have produced.
    if units.is_empty() {
        let left_merged = merge_batches(left_batches)?;
        return probe(
            &left_merged, right, left_keys, right_keys, hash_table, bloom, join_type, keep,
        );
    }

    // Indices are row numbers inside the unit's own batch, so nothing needs a
    // global numbering and the left side is never merged.
    let probe_unit =
        |&(bi, start, end): &(usize, usize, usize)| -> FalconResult<(usize, Vec<u32>, Vec<u32>)> {
            let batch = &left_batches[bi];
            let left_key_cols: Vec<&Column> = left_keys.iter()
                .map(|k| batch.column_by_name(k))
                .collect::<FalconResult<_>>()?;
            let bloom_ok = bloom.filter(|_| filter_is_safe(&right_key_cols, &left_key_cols));

            let mut li = Vec::new();
            let mut ri = Vec::new();
            let mut scratch = Vec::new();
            for l_row in start..end {
                probe_one(
                    l_row, &left_key_cols, hash_table, bloom_ok, &join_type,
                    &mut li, &mut ri, None, &mut scratch,
                )?;
            }
            Ok((bi, li, ri))
        };

    let matches: Vec<(usize, Vec<u32>, Vec<u32>)> = if units.len() > 1 {
        units.par_iter().map(probe_unit).collect::<FalconResult<_>>()?
    } else {
        units.iter().map(probe_unit).collect::<FalconResult<_>>()?
    };

    // Preferred: one allocation per output column. The per-unit builder below is
    // correct for every case but allocates a batch per unit and then copies the
    // whole result again to concatenate them — two passes over the output, and an
    // allocation pattern that measurably degrades other code in the same process.
    if join_type == JoinType::Inner {
        if let Some(out) = build_output_dense(left_batches, right, &matches, keep)? {
            return Ok(out);
        }
    }

    let outs: FalconResult<Vec<RecordBatch>> = matches
        .par_iter()
        .map(|(bi, li, ri)| build_output(&left_batches[*bi], right, li, ri, keep))
        .collect();
    RecordBatch::concat(&outs?)
}

/// Can this column be gathered straight into a caller-provided slice?
fn is_dense_gatherable(c: &Column) -> bool {
    matches!(
        c,
        Column::Int32(_)
            | Column::Int64(_)
            | Column::Float32(_)
            | Column::Float64(_)
            | Column::Date32(_)
            | Column::Timestamp(_)
    )
}

/// Build an INNER join's output with one allocation per column.
///
/// Each unit fills a disjoint slice of a buffer allocated once, so the output is
/// written exactly once. The alternative — a `RecordBatch` per unit followed by a
/// concatenation — writes it twice and, more importantly, turns one large
/// allocation per column into one mid-sized allocation per unit, allocated on a
/// worker thread and freed on another. That pattern cost Polars 3.61 ms -> 8.24 ms
/// when it ran after FalconDB in the same benchmark process; see Methodology in
/// `README.md`.
///
/// Returns `None` when the shape is not one this can handle — a non-primitive
/// column, or a source with NULLs — and the caller falls back. INNER only: LEFT
/// emits `NO_ROW` markers, which are NULLs in the output.
fn build_output_dense(
    left_batches: &[RecordBatch],
    right:        &RecordBatch,
    matches:      &[(usize, Vec<u32>, Vec<u32>)],
    keep:         Option<&HashSet<String>>,
) -> FalconResult<Option<RecordBatch>> {
    let Some(first) = left_batches.first() else { return Ok(None) };
    let plan  = output_plan(&first.schema, &right.schema, keep);
    let total: usize = matches.iter().map(|(_, li, _)| li.len()).sum();

    // Every source this will read must be a dense primitive with no NULLs, and the
    // left batches must agree on type. Checked up front so the fallback is chosen
    // before any column has been built.
    let usable = plan.iter().all(|(side, ci, _, _)| match side {
        Side::Right => {
            let c = right.column(*ci);
            is_dense_gatherable(c) && c.null_count() == 0
        }
        Side::Left => {
            let head = first.column(*ci);
            is_dense_gatherable(head)
                && left_batches.iter().all(|b| {
                    let c = b.column(*ci);
                    std::mem::discriminant(c) == std::mem::discriminant(head)
                        && c.null_count() == 0
                })
        }
    });
    if !usable {
        return Ok(None);
    }

    let mut fields  = Vec::with_capacity(plan.len());
    let mut columns = Vec::with_capacity(plan.len());

    for (side, ci, name, dt) in &plan {
        macro_rules! dense {
            ($variant:ident, $t:ty) => {{
                let mut values: Vec<$t> = vec![<$t>::default(); total];

                // Hand each unit the slice its own rows occupy. Units appear in
                // batch order and, within a batch, in row order, so the slices land
                // in the same order the merged path would have produced.
                let mut rest: &mut [$t] = values.as_mut_slice();
                let mut slots: Vec<&mut [$t]> = Vec::with_capacity(matches.len());
                for (_, li, _) in matches {
                    let (head, tail) = std::mem::take(&mut rest).split_at_mut(li.len());
                    slots.push(head);
                    rest = tail;
                }

                slots.par_iter_mut().zip(matches.par_iter()).for_each(
                    |(out, (bi, li, ri))| {
                        let (src, idx) = match side {
                            Side::Left  => (left_batches[*bi].column(*ci), li),
                            Side::Right => (right.column(*ci), ri),
                        };
                        // A mismatch here would leave the output full of default
                        // values -- a wrong answer that looks like data. `usable`
                        // has already established the variant, but the check and
                        // the use are far apart, so say so loudly rather than
                        // trusting a future edit to keep them in step. A silently
                        // dropped mismatch is exactly how `table_from_batches`
                        // lost every row of a recursive CTE.
                        match src {
                            Column::$variant(p) => p.gather_u32_into(idx, out),
                            other => unreachable!(
                                "build_output_dense: `usable` accepted {:?} but the                                  gather expected {}",
                                other.data_type(),
                                stringify!($variant),
                            ),
                        }
                    },
                );
                drop(slots);
                Column::$variant(PrimitiveColumn::from_non_null(values))
            }};
        }

        let probe_src = match side {
            Side::Left  => first.column(*ci),
            Side::Right => right.column(*ci),
        };
        let col = match probe_src {
            Column::Int32(_)     => dense!(Int32, i32),
            Column::Int64(_)     => dense!(Int64, i64),
            Column::Float32(_)   => dense!(Float32, f32),
            Column::Float64(_)   => dense!(Float64, f64),
            Column::Date32(_)    => dense!(Date32, i32),
            Column::Timestamp(_) => dense!(Timestamp, i64),
            // `usable` already excluded these; returning None keeps the two in
            // agreement rather than trusting them to stay in agreement.
            _ => return Ok(None),
        };
        fields.push(Field::nullable(name.as_str(), dt.clone()));
        columns.push(col);
    }

    Ok(Some(RecordBatch::new(Schema::new(fields), columns)?))
}

// ─── Sort-merge join ─────────────────────────────────────────────────────────

/// Inner join by sorting both sides on the key and walking them in step.
///
/// The hash join builds a table of every distinct right-hand key, which for a
/// large build side is a large allocation and a random-access probe per left
/// row. Sort-merge trades that for two sorts and a single linear pass, so its
/// memory is bounded by the sort rather than by key cardinality, and both scans
/// are sequential.
///
/// Which is faster depends on the shape of the data, so this is offered
/// alongside the hash join rather than replacing it — see `should_sort_merge`.
/// Only INNER is implemented; outer variants need unmatched-row bookkeeping that
/// the hash path already does well.
pub fn execute_sort_merge_join(
    left_batches:  &[RecordBatch],
    right_batches: &[RecordBatch],
    left_keys:     &[&str],
    right_keys:    &[&str],
) -> FalconResult<RecordBatch> {
    if left_keys.len() != right_keys.len() || left_keys.is_empty() {
        return Err(FalconError::InvalidQuery(
            "join key lists must have the same length".into(),
        ));
    }

    let left  = merge_batches(left_batches)?;
    let right = merge_batches(right_batches)?;

    let left_key_cols: Vec<&Column> = left_keys.iter()
        .map(|k| left.column_by_name(k))
        .collect::<FalconResult<_>>()?;
    let right_key_cols: Vec<&Column> = right_keys.iter()
        .map(|k| right.column_by_name(k))
        .collect::<FalconResult<_>>()?;

    // Sort row indices by key rather than the rows themselves: the payload
    // columns are gathered once at the end, so they are never moved twice.
    let mut l_idx: Vec<usize> = (0..left.row_count()).collect();
    let mut r_idx: Vec<usize> = (0..right.row_count()).collect();

    let l_keys: Vec<String> = (0..left.row_count())
        .map(|r| row_key(&left_key_cols, r))
        .collect::<FalconResult<_>>()?;
    let r_keys: Vec<String> = (0..right.row_count())
        .map(|r| row_key(&right_key_cols, r))
        .collect::<FalconResult<_>>()?;

    l_idx.sort_unstable_by(|&a, &b| l_keys[a].cmp(&l_keys[b]));
    r_idx.sort_unstable_by(|&a, &b| r_keys[a].cmp(&r_keys[b]));

    let mut left_out:  Vec<u32> = Vec::new();
    let mut right_out: Vec<u32> = Vec::new();

    let (mut i, mut j) = (0usize, 0usize);
    while i < l_idx.len() && j < r_idx.len() {
        let lk = &l_keys[l_idx[i]];
        let rk = &r_keys[r_idx[j]];

        match lk.cmp(rk) {
            std::cmp::Ordering::Less    => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                // Equal keys form a run on each side; every pairing matches, so
                // emit the cross product of the two runs.
                let i_end = (i..l_idx.len()).find(|&x| &l_keys[l_idx[x]] != lk).unwrap_or(l_idx.len());
                let j_end = (j..r_idx.len()).find(|&x| &r_keys[r_idx[x]] != rk).unwrap_or(r_idx.len());

                for &l in &l_idx[i..i_end] {
                    for &r in &r_idx[j..j_end] {
                        left_out.push(l as u32);
                        right_out.push(r as u32);
                    }
                }
                i = i_end;
                j = j_end;
            }
        }
    }

    build_output(&left, &right, &left_out, &right_out, None)
}

/// Is sort-merge the better choice for these inputs?
///
/// Hash join wins when the build side is small: the table fits in cache and the
/// probe is one lookup per row. It loses when the build side is large enough
/// that the table stops fitting, at which point every probe is a cache miss and
/// the two sorts become the cheaper option.
///
/// The threshold is a judgement call, not a measured optimum — it is deliberately
/// high so the default stays with the well-tested hash path.
pub fn should_sort_merge(left_rows: usize, right_rows: usize) -> bool {
    const BUILD_SIDE_LIMIT: usize = 1_000_000;
    right_rows > BUILD_SIDE_LIMIT && left_rows > BUILD_SIDE_LIMIT
}

// ─── Build phase ─────────────────────────────────────────────────────────────

/// Smallest build side worth filtering. Below this the filter costs more to
/// construct than the probe-side lookups it saves.
const BLOOM_MIN_ROWS: usize = 1_024;

fn build_hash_table(
    batch: &RecordBatch,
    keys:  &[&str],
    typed: bool,
) -> FalconResult<(JoinKeyMap, Option<BuildFilter>)> {
    let key_cols: Vec<&Column> = keys.iter()
        .map(|k| batch.column_by_name(k))
        .collect::<FalconResult<_>>()?;

    let n = batch.row_count();
    let mut bloom = (n >= BLOOM_MIN_ROWS).then(|| BuildFilter::with_capacity(n));

    if typed {
        let col = key_cols[0];
        // The same column the probe will read, shared not copied.
        let key_idx = batch.schema.column_index(keys[0])?;
        let mut table = IntHashTable::with_rows(n, Arc::clone(&batch.columns[key_idx]));
        for row in 0..n {
            let k = int_key(col, row);
            table.insert(k, row);
            if let Some(f) = bloom.as_mut() {
                f.insert(int_key_hash(k));
            }
        }
        return Ok((JoinKeyMap::Int(table), bloom));
    }

    let mut table: FastMap<String, SmallVec<[usize; 1]>> = FastMap::default();
    for row in 0..n {
        let key = row_key(&key_cols, row)?;
        table.entry(key).or_default().push(row);
        if let Some(f) = bloom.as_mut() {
            f.insert(row_key_hash(&key_cols, row));
        }
    }

    Ok((JoinKeyMap::Str(table), bloom))
}

// ─── Probe phase ─────────────────────────────────────────────────────────────

/// Match one left row and append whatever it contributes to the output.
///
/// Shared by the serial and parallel drivers so the two cannot drift: this join
/// was 100x behind precisely because one copy of a routine got fixed and another
/// did not. `right_matched` is `None` for the parallel path, which only runs for
/// INNER and LEFT — the join types that never need it.
#[allow(clippy::too_many_arguments)]
#[inline]
fn probe_one(
    l_row:         usize,
    left_key_cols: &[&Column],
    hash_table:    &JoinKeyMap,
    bloom:         Option<&BuildFilter>,
    join_type:     &JoinType,
    left_indices:  &mut Vec<u32>,
    right_indices: &mut Vec<u32>,
    right_matched: Option<&mut Vec<bool>>,
    scratch:       &mut Vec<usize>,
) -> FalconResult<()> {
    // A definite miss skips the lookup, and on the string path the key build too.
    if let Some(f) = bloom {
        let h = match hash_table {
            JoinKeyMap::Int(_) => int_key_hash(int_key(left_key_cols[0], l_row)),
            JoinKeyMap::Str(_) => row_key_hash(left_key_cols, l_row),
        };
        if !f.might_contain(h) {
            if *join_type == JoinType::Left {
                left_indices.push(l_row as u32);
                right_indices.push(NO_ROW);
            }
            return Ok(());
        }
    }

    // Reused across rows so a match costs no allocation.
    scratch.clear();
    match hash_table {
        JoinKeyMap::Int(t) => t.collect_matches(int_key(left_key_cols[0], l_row), scratch),
        JoinKeyMap::Str(t) => {
            // Borrowed lookup when the key is a single text column; see
            // `single_text_key`. `HashMap<String, _>` accepts `&str`, so this
            // costs no allocation at all.
            let found = match single_text_key(left_key_cols, l_row) {
                Some(key) => t.get(key),
                None if matches!(left_key_cols, [Column::Utf8(_)]) => None, // NULL key
                None => t.get(&row_key(left_key_cols, l_row)?),
            };
            if let Some(rows) = found {
                scratch.extend_from_slice(rows);
            }
        }
    }

    if scratch.is_empty() {
        if *join_type == JoinType::Left || *join_type == JoinType::Full {
            left_indices.push(l_row as u32);
            right_indices.push(NO_ROW);
        }
        return Ok(());
    }

    for &r_row in scratch.iter() {
        left_indices.push(l_row as u32);
        right_indices.push(r_row as u32);
    }
    if let Some(m) = right_matched {
        for &r_row in scratch.iter() {
            m[r_row] = true;
        }
    }
    Ok(())
}

fn probe(
    left:       &RecordBatch,
    right:      &RecordBatch,
    left_keys:  &[&str],
    right_keys: &[&str],
    hash_table: &JoinKeyMap,
    bloom:      Option<&BuildFilter>,
    join_type:  JoinType,
    keep:       Option<&HashSet<String>>,
) -> FalconResult<RecordBatch> {
    let left_key_cols: Vec<&Column> = left_keys.iter()
        .map(|k| left.column_by_name(k))
        .collect::<FalconResult<_>>()?;
    let right_key_cols: Vec<&Column> = right_keys.iter()
        .map(|k| right.column_by_name(k))
        .collect::<FalconResult<_>>()?;

    // Only use the filter when the two sides agree on key types; see
    // `row_key_hash` for why a mismatch could drop matches.
    let bloom_ok = bloom.filter(|_| filter_is_safe(&right_key_cols, &left_key_cols));

    // INNER and LEFT decide each left row independently, so the probe splits over
    // row ranges. RIGHT and FULL also have to record which right rows were hit,
    // which is shared state, so they stay on the serial path rather than paying
    // for synchronisation on every match.
    //
    // Everything else in the engine — scan, filter, group-by — is parallel; the
    // join was the one operator still running on a single core.
    if matches!(join_type, JoinType::Inner | JoinType::Left)
        && left.row_count() >= par_probe_rows()
    {
        let n = left.row_count();
        let chunk = (n / rayon::current_num_threads().max(1)).max(8_192);
        let ranges: Vec<(usize, usize)> = (0..n)
            .step_by(chunk)
            .map(|s| (s, (s + chunk).min(n)))
            .collect();

        let parts: FalconResult<Vec<(Vec<u32>, Vec<u32>)>> = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut li = Vec::new();
                let mut ri = Vec::new();
                let mut scratch = Vec::new();
                for l_row in start..end {
                    probe_one(
                        l_row, &left_key_cols, hash_table, bloom_ok, &join_type,
                        &mut li, &mut ri, None, &mut scratch,
                    )?;
                }
                Ok((li, ri))
            })
            .collect();

        // Ranges are visited in order, so concatenating them keeps the same output
        // row order the serial path produces.
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for (li, ri) in parts? {
            left_indices.extend(li);
            right_indices.extend(ri);
        }
        return build_output(left, right, &left_indices, &right_indices, keep);
    }

    let mut left_indices:  Vec<u32> = Vec::new();
    let mut right_indices: Vec<u32> = Vec::new();
    let mut right_matched: Vec<bool>          = vec![false; right.row_count()];

    let mut scratch = Vec::new();
    for l_row in 0..left.row_count() {
        probe_one(
            l_row, &left_key_cols, hash_table, bloom_ok, &join_type,
            &mut left_indices, &mut right_indices, Some(&mut right_matched),
            &mut scratch,
        )?;
    }

    // RIGHT / FULL OUTER JOIN: add unmatched right rows.
    if join_type == JoinType::Right || join_type == JoinType::Full {
        for (r_row, &matched) in right_matched.iter().enumerate() {
            if !matched {
                left_indices.push(NO_ROW);
                right_indices.push(r_row as u32);
            }
        }
    }

    build_output(left, right, &left_indices, &right_indices, keep)
}

// ─── Output builder ───────────────────────────────────────────────────────────

/// Which side an output column comes from.
enum Side { Left, Right }

/// Decide the join's output columns: which side each comes from, its index
/// there, its name after the join, and its type.
///
/// Shared by the two output builders. It used to live inside `build_output`,
/// which was fine while there was one builder; a second one that re-derived the
/// same list would be a second definition of what a join returns, and this
/// engine has already paid for two implementations of one semantics.
fn output_plan(
    left:  &Schema,
    right: &Schema,
    keep:  Option<&HashSet<String>>,
) -> Vec<(Side, usize, String, DataType)> {
    // Right columns keep their name unless the left side already used it.
    let left_names: FastSet<&str> = left.column_names().into_iter().collect();

    let mut plan: Vec<(Side, usize, String, DataType)> = Vec::new();
    // A column nothing downstream reads is never built. Gathering one costs a cache
    // miss per output row against the larger input, which for `SELECT COUNT(*)` over
    // a join was most of the query. `keep` is `None` whenever the caller could not
    // prove which columns are read, and then everything is built as before.
    //
    // Both the source name and the post-join name are accepted, so a reference that
    // survived a rename is not pruned by accident: erring toward keeping a column
    // costs time, dropping one the query reads is a wrong answer.
    //
    // A table-qualified reference is matched on the part after the dot. The parser
    // hands `ints.i` through as the literal string "ints.i", and comparing that to
    // the column's own name -- "i" -- matched nothing, so
    //
    //     SELECT ints.i FROM ints JOIN floats ON ints.i = floats.f
    //
    // pruned away every column it asked for, fell to the one-column fallback
    // below, and then failed with "column 'ints.i' not found" while the same query
    // selecting plain `i` worked. Only the select list of a *join* was affected;
    // `Schema::column_index` had already learned this rule for everywhere else.
    //
    // Matching on the bare name can keep a column the query did not want -- both
    // sides' `i`, when only one was asked for -- which is the direction this
    // function already errs in, and for the same reason.
    let wanted = |src: &str, out: &str| -> bool {
        keep.is_none_or(|k| {
            k.contains(src)
                || k.contains(out)
                || k.iter().any(|w| {
                    w.rsplit_once('.').is_some_and(|(_, bare)| {
                        bare.eq_ignore_ascii_case(src) || bare.eq_ignore_ascii_case(out)
                    })
                })
        })
    };

    for (i, field) in left.fields.iter().enumerate() {
        if wanted(&field.name, &field.name) {
            plan.push((Side::Left, i, field.name.clone(), field.data_type.clone()));
        }
    }
    for (i, field) in right.fields.iter().enumerate() {
        let name = if left_names.contains(field.name.as_str()) {
            format!("right.{}", field.name)
        } else {
            field.name.clone()
        };
        if wanted(&field.name, &name) {
            plan.push((Side::Right, i, name, field.data_type.clone()));
        }
    }

    // A batch with no columns reports zero rows, which would turn COUNT(*) into 0.
    // Keep one column from the smaller side so the row count survives; the right
    // side is the build side and is normally the smaller of the two.
    if plan.is_empty() {
        if let Some(field) = right.fields.first() {
            plan.push((Side::Right, 0, field.name.clone(), field.data_type.clone()));
        } else if let Some(field) = left.fields.first() {
            plan.push((Side::Left, 0, field.name.clone(), field.data_type.clone()));
        }
    }
    plan
}

fn build_output(
    left:          &RecordBatch,
    right:         &RecordBatch,
    left_indices:  &[u32],
    right_indices: &[u32],
    keep:          Option<&HashSet<String>>,
) -> FalconResult<RecordBatch> {
    // Match pairs are `u32` row numbers with `NO_ROW` reserved as the NULL marker,
    // so a side at or past that many rows cannot be addressed. Refusing is the only
    // safe answer: silently truncating would join the wrong rows together.
    for (side, batch) in [("left", left), ("right", right)] {
        if batch.row_count() >= NO_ROW as usize {
            return Err(FalconError::InvalidQuery(format!(
                "join {side} side has {} rows, more than this join can address ({})",
                batch.row_count(),
                NO_ROW as usize - 1,
            )));
        }
    }

    let n_rows = left_indices.len();
    let mut fields  = Vec::new();
    let mut columns = Vec::new();

    let plan = output_plan(&left.schema, &right.schema, keep);

    let gather_one = |(side, i, _name, dt): &(Side, usize, String, DataType)| -> FalconResult<Column> {
        let (col, indices) = match side {
            Side::Left  => (left.column(*i),  left_indices),
            Side::Right => (right.column(*i), right_indices),
        };
        Ok(match gather_u32_typed(col, indices, dt) {
            Some((_, c)) => c,
            None => {
                let vals = gather_nullable(col, indices, n_rows)?;
                scalar_vec_to_col(&vals, dt).1
            }
        })
    };

    // Serial on purpose. Gathering the output columns in parallel looks free —
    // they are independent passes over a shared index list — and was tried here at
    // 100k output rows: LEFT JOIN went from 1.70x DuckDB to 1.76x, i.e. nothing.
    // That is the same answer `RecordBatch::filter_by_bitmask` records for the same
    // idea: a gather is bound by memory, so more threads contend for one bus
    // instead of adding throughput.
    let built: FalconResult<Vec<Column>> = plan.iter().map(gather_one).collect();

    for ((_, _, name, dt), c) in plan.iter().zip(built?) {
        fields.push(Field::nullable(name.as_str(), dt.clone()));
        columns.push(c);
    }

    RecordBatch::new(Schema::new(fields), columns)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// The key of one row, borrowed, when the join has a single text key column.
///
/// `row_key` renders every key column into a `String` and joins them, so a probe
/// row costs two allocations before the hash table is consulted -- which is what
/// the Bloom filter was added to avoid paying for rows that cannot match.
///
/// A `HashMap<String, _>` can be looked up by `&str`, so for the single-text-key
/// case none of that is needed. Measured on a 100k-row build against 1M probe
/// rows, an INNER JOIN on a text key took 95.86ms against 6.22ms for the same
/// join on an integer key.
///
/// `None` means either "not this shape" or "this row's key is NULL"; both send
/// the caller down the general path, and a NULL key matches nothing anyway.
#[inline]
fn single_text_key<'a>(key_cols: &[&'a Column], row: usize) -> Option<&'a str> {
    let [Column::Utf8(c)] = key_cols else { return None };
    c.get_str(row)
}

fn row_key(key_cols: &[&Column], row: usize) -> FalconResult<String> {
    let parts: FalconResult<Vec<String>> = key_cols.iter()
        .map(|col| col.get(row).map(|v| v.to_string()))
        .collect();
    Ok(parts?.join("\x00"))
}

// ─── Typed integer keys ───────────────────────────────────────────────────────

/// The hash table, keyed either by a raw integer or by the formatted string.
///
/// `row_key` formats every key into a `String`: for a join on an `i64` column that
/// is a `ScalarValue`, an integer formatted into a heap string, a `Vec<String>`
/// and a final joined `String` — four allocations per row, on both sides. At a
/// million rows a side it was the bulk of a 312 ms join. Hashing the integer
/// itself allocates nothing.
enum JoinKeyMap {
    Int(IntHashTable),
    /// One entry per distinct key, holding the build rows that carry it.
    ///
    /// `SmallVec<[usize; 1]>` rather than `Vec<usize>`: a join key is usually
    /// unique on the build side, and a `Vec` for every one of them is a heap
    /// allocation per distinct key -- 100k of them for a 100k-row build side,
    /// each holding a single number, each a pointer to chase on every probe that
    /// hits it. The integer table solved the same problem with chains in a flat
    /// array; this keeps the common case inline instead.
    Str(FastMap<String, SmallVec<[usize; 1]>>),
}

/// Chained hash table over integer keys, in three flat arrays.
///
/// The map this replaces was `FastMap<i64, Vec<usize>>`, which allocates a `Vec`
/// per distinct key: a 100k-row build side meant 100k heap allocations, and
/// measured at ~30 ms of a 41 ms join — far more than the probe it feeds. Here a
/// build touches three preallocated arrays and allocates nothing per key.
///
/// `heads` is indexed by bucket and holds the first row in that bucket, `next`
/// chains the rest, and `keys` lets the probe confirm a real match rather than a
/// bucket collision. Row numbers are `u32`, which is why `int_keys_usable` refuses
/// build sides at or above `u32::MAX` rows.
struct IntHashTable {
    mask:  usize,
    heads: Vec<u32>,
    next:  Vec<u32>,
    /// The build side's key column, shared rather than copied.
    ///
    /// This used to be a `Vec<i64>` of one key per build row -- 800 KB at 100k
    /// rows -- duplicating what the column already held. The chain walk reads it
    /// to check whether a node really matches; it can read the column instead,
    /// and an `Arc` clone costs a refcount bump.
    ///
    /// Measured on both sides, four runs, 100k build rows against 1M probes:
    ///
    ///   build   with 603-1170us   without 553-589us
    ///   probe   with 11.20-12.37ms  without 10.95-11.35ms
    ///
    /// Faster on both, which was not obvious -- an extra indirection per chain
    /// node could have cost more than the allocation saved. It does not, because
    /// the column is resident anyway and 800 KB of duplicate stops competing with
    /// the buckets for cache. The numbers without it are also markedly steadier,
    /// the copy having made the build sensitive to allocator state.
    ///
    /// Unlike the bucket array this one had no second job: `bucket_density`
    /// records why shrinking *that* one loses. `keys_array_worth_it` records
    /// this.
    key_col: Arc<Column>,
}

/// Marks the end of a chain / an empty bucket.
const NO_ROW: u32 = u32::MAX;

impl IntHashTable {
    fn with_rows(n: usize, key_col: Arc<Column>) -> Self {
        // Two buckets per row keeps chains short without wasting much memory.
        // One per row was measured and loses: it saves 300us of build and costs
        // 4.6ms of probe, because the probe walks the longer chains a million
        // times. Four per row wins on a serial probe and loses on a parallel one,
        // which is the one production has. See `bucket_density`.
        let buckets = (n.max(1) * 2).next_power_of_two();
        Self {
            mask:  buckets - 1,
            heads: vec![NO_ROW; buckets],
            next:  vec![NO_ROW; n],
            key_col,
        }
    }

    /// Mixes an i64 into a bucket. Join keys are very often dense runs like
    /// 0, 10, 20, ..., which would pile into a handful of buckets if the low bits
    /// were used directly.
    #[inline(always)]
    fn bucket(&self, key: i64) -> usize {
        bucket_with(self.mask, key)
    }

    /// Build the table in parallel, producing byte-for-byte what the serial
    /// insert loop produces -- and measurably slower than it.
    ///
    /// Kept, behind `cfg(test)`, because the idea is an obvious one to have and
    /// the measurement says not to: 0.93x serial at two partitions, 0.97x at
    /// four, and worse above that, against a serial build of 767us. A 7% best
    /// case on a build that is a quarter of the join is about 1.5% of a join,
    /// for atomics and a second code path. `parallel_build_speed` prints the
    /// table.
    ///
    /// Why it loses is the shape, not the tuning. Partitioning by bucket makes
    /// every thread examine every row to find the ones it owns, so total work is
    /// O(partitions x rows) where the serial loop is O(rows). Per-row work here
    /// is a multiply and a compare, far too little to pay for being looked at
    /// twelve times. Something that wins has to keep total work at O(rows) -- a
    /// radix partition of the rows in one pass, then chains built per partition
    /// -- which is a real piece of engineering rather than a tweak.
    ///
    /// What the exercise did establish is that identity is achievable: see
    /// below.
    ///
    /// The build runs on one thread while the probe runs on all of them, so it
    /// weighs on a join more than its cost suggests: 1.03 ms of a 4.43 ms 1M-row
    /// join, none of it parallel.
    ///
    /// Partitioned by bucket rather than by row. Every bucket belongs to exactly
    /// one partition and every row to exactly one bucket, so no two threads ever
    /// touch the same `heads` slot or the same `next` slot. The atomics carry no
    /// contention; they are here only because the compiler cannot see that
    /// disjointness, and on x86 a relaxed load or store is an ordinary one.
    ///
    /// Rows are visited in ascending order within a partition -- the order the
    /// serial loop used -- so the chains come out identical rather than merely
    /// equivalent. That matters beyond tidiness: chain order within a bucket
    /// decides the order of a join's output rows, and an engine that reorders
    /// results between runs is a worse thing than a slow one.
    /// `parallel_build_is_identical_to_serial` asserts the arrays match exactly.
    ///
    /// The price is that every thread computes `bucket` for every row and keeps
    /// only the rows it owns: arithmetic on a value already in a register, traded
    /// against a scattered write it would otherwise have to synchronise on.
    /// Build the table in parallel with a radix partition, keeping total work at
    /// O(rows) and reproducing the serial chains byte for byte -- and still
    /// slower than the serial loop, at 1.60x it.
    ///
    /// That result is the useful part. `build_parallel` lost because its total
    /// work was O(partitions x rows); this one fixes exactly that and does not
    /// win, which says the insert loop was never the bottleneck. What a build
    /// actually spends its time on is allocating and filling roughly 2.2 MB per
    /// join -- 1 MB of bucket heads, 400 KB of chain links, 800 KB of keys -- plus,
    /// for either parallel version, copying the atomic arrays back into plain ones.
    /// Parallelism cannot help with memory that has to be touched once either way.
    ///
    /// So the next thing worth trying on the build side is allocating less, not
    /// spreading the work wider: one bucket per row instead of two, and dropping
    /// the `keys` array, which duplicates data the key column already holds.
    /// Both are measurable with the harness in this module.
    ///
    /// `build_parallel` below partitions by bucket, which makes every thread look
    /// at every row to find the ones it owns -- O(partitions x rows), and measured
    /// slower than the serial loop. This instead makes one pass over the rows to
    /// decide where each belongs, then gives each thread a contiguous run of rows
    /// that are already its own.
    ///
    /// Five phases, none of them O(partitions x rows):
    ///   1. keys, and from them each row's bucket partition (parallel over rows)
    ///   2. count rows per (row chunk, bucket partition) -- a P x P matrix, not
    ///      one counter per bucket, so it stays in cache
    ///   3. prefix-sum those counts, partition-major and chunk-minor
    ///   4. scatter row ids into one scratch array at those offsets
    ///   5. each thread walks its partition's slice of scratch and links the
    ///      chains for its own buckets
    ///
    /// Order survives all of it. Phase 3 lays chunks out in ascending order within
    /// a partition, and phase 4 preserves row order within a chunk, so phase 5
    /// sees rows in ascending order exactly as the serial loop did. That is what
    /// `parallel_build_is_identical_to_serial` checks -- not that the matches
    /// agree, but that `heads` and `next` are the same arrays, because chain order
    /// decides the order of a join's output rows.
    ///
    /// The atomics carry no contention: a row belongs to one bucket, a bucket to
    /// one partition, and a partition to one thread, so no two threads ever touch
    /// the same slot. They are here only because the compiler cannot see that.
    #[cfg(test)]
    fn build_radix(col: Arc<Column>, n: usize) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

        let buckets = (n.max(1) * 2).next_power_of_two();
        let mask = buckets - 1;

        let keys: Vec<i64> = (0..n).into_par_iter().map(|r| int_key(&col, r)).collect();
        if n == 0 {
            return Self {
                mask,
                heads: vec![NO_ROW; buckets],
                next:  Vec::new(),
                key_col: col,
            };
        }

        let parts  = rayon::current_num_threads().max(1);
        let span   = buckets.div_ceil(parts);
        let chunk  = n.div_ceil(parts);
        let chunks = n.div_ceil(chunk);

        // Phase 2: how many rows each chunk sends to each partition.
        let counts: Vec<Vec<u32>> = (0..chunks)
            .into_par_iter()
            .map(|c| {
                let lo = c * chunk;
                let hi = ((c + 1) * chunk).min(n);
                let mut local = vec![0u32; parts];
                for &k in &keys[lo..hi] {
                    local[bucket_with(mask, k) / span] += 1;
                }
                local
            })
            .collect();

        // Phase 3: partition-major, chunk-minor. A partition's rows end up
        // contiguous, and within it the chunks stay in ascending order.
        let mut offsets    = vec![vec![0u32; parts]; chunks];
        let mut part_start = vec![0u32; parts + 1];
        let mut running    = 0u32;
        for p in 0..parts {
            part_start[p] = running;
            for (c, off) in offsets.iter_mut().enumerate() {
                off[p] = running;
                running += counts[c][p];
            }
        }
        part_start[parts] = running;
        debug_assert_eq!(running as usize, n, "radix partition lost rows");

        // Phase 4: scatter.
        let scratch: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(NO_ROW)).collect();
        (0..chunks).into_par_iter().for_each(|c| {
            let lo = c * chunk;
            let hi = ((c + 1) * chunk).min(n);
            let mut cur = offsets[c].clone();
            for (r, &k) in (lo..hi).zip(&keys[lo..hi]) {
                let p = bucket_with(mask, k) / span;
                scratch[cur[p] as usize].store(r as u32, Relaxed);
                cur[p] += 1;
            }
        });

        // Phase 5: link the chains, one partition per thread.
        let heads: Vec<AtomicU32> = (0..buckets).map(|_| AtomicU32::new(NO_ROW)).collect();
        let next:  Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(NO_ROW)).collect();
        (0..parts).into_par_iter().for_each(|p| {
            let (lo, hi) = (part_start[p] as usize, part_start[p + 1] as usize);
            for slot in &scratch[lo..hi] {
                let row = slot.load(Relaxed) as usize;
                let b = bucket_with(mask, keys[row]);
                next[row].store(heads[b].load(Relaxed), Relaxed);
                heads[b].store(row as u32, Relaxed);
            }
        });

        Self {
            mask,
            heads: heads.into_iter().map(|a| a.into_inner()).collect(),
            next:  next.into_iter().map(|a| a.into_inner()).collect(),
            key_col: col,
        }
    }

    #[cfg(test)]
    fn build_parallel(col: Arc<Column>, n: usize) -> Self {
        Self::build_parallel_with(col, n, rayon::current_num_threads().max(1))
    }

    #[cfg(test)]
    fn build_parallel_with(col: Arc<Column>, n: usize, parts: usize) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

        let buckets = (n.max(1) * 2).next_power_of_two();
        let mask = buckets - 1;

        let keys: Vec<i64> = (0..n).into_par_iter().map(|r| int_key(&col, r)).collect();

        let heads: Vec<AtomicU32> = (0..buckets).map(|_| AtomicU32::new(NO_ROW)).collect();
        let next:  Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(NO_ROW)).collect();

        let span = buckets.div_ceil(parts);
        (0..parts).into_par_iter().for_each(|p| {
            let lo = p * span;
            let hi = ((p + 1) * span).min(buckets);
            if lo >= hi {
                return;
            }
            for (row, &k) in keys.iter().enumerate() {
                let b = bucket_with(mask, k);
                if b >= lo && b < hi {
                    next[row].store(heads[b].load(Relaxed), Relaxed);
                    heads[b].store(row as u32, Relaxed);
                }
            }
        });

        Self {
            mask,
            heads: heads.into_iter().map(|a| a.into_inner()).collect(),
            next:  next.into_iter().map(|a| a.into_inner()).collect(),
            key_col: col,
        }
    }

    #[inline]
    fn insert(&mut self, key: i64, row: usize) {
        debug_assert!(row < self.next.len(), "build row out of range");
        let b = self.bucket(key);
        self.next[row] = self.heads[b];
        self.heads[b] = row as u32;
    }

    /// Append every build-side row matching `key`.
    #[inline]
    fn collect_matches(&self, key: i64, out: &mut Vec<usize>) {
        let mut cur = self.heads[self.bucket(key)];
        while cur != NO_ROW {
            let row = cur as usize;
            if int_key(&self.key_col, row) == key {
                out.push(row);
            }
            cur = self.next[row];
        }
    }

    #[inline]
    fn contains(&self, key: i64) -> bool {
        let mut cur = self.heads[self.bucket(key)];
        while cur != NO_ROW {
            let row = cur as usize;
            if int_key(&self.key_col, row) == key {
                return true;
            }
            cur = self.next[row];
        }
        false
    }
}

/// Integer-like columns that fit an `i64` key without loss.
fn is_int_key_col(col: &Column) -> bool {
    matches!(
        col,
        Column::Int64(_) | Column::Int32(_) | Column::Date32(_) | Column::Timestamp(_)
    )
}

/// Bloom hash for a raw integer key.
///
/// The generic `row_key_hash` matches on the column enum for every row, which on
/// the typed path is dispatch the caller has already resolved — and it runs once
/// per probed row, so it is the hottest thing in the join. Build and probe must
/// agree on this function or the filter would reject real matches.
#[inline(always)]
fn int_key_hash(k: i64) -> u64 {
    let mut h = k as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h
}

#[inline(always)]
fn int_key(col: &Column, row: usize) -> i64 {
    match col {
        Column::Int64(c) => c.dense()[row],
        Column::Int32(c) => c.dense()[row] as i64,
        Column::Date32(c) => c.dense()[row] as i64,
        Column::Timestamp(c) => c.dense()[row],
        _ => unreachable!("guarded by is_int_key_col"),
    }
}

/// May this join hash raw integers instead of formatted strings?
///
/// Requires one key column per side, the same integer type on both, and no NULLs.
/// The NULL rule is not an optimisation detail: `row_key` renders a NULL as the
/// text "NULL", so on the string path two NULL keys compare equal and join to each
/// other, which SQL says must not happen. Restricting the typed path to null-free
/// columns keeps both paths giving the same answer instead of quietly disagreeing.
fn int_keys_usable(
    build: &RecordBatch,
    build_keys: &[&str],
    probe: &RecordBatch,
    probe_keys: &[&str],
) -> bool {
    if build_keys.len() != 1 || probe_keys.len() != 1 {
        return false;
    }
    let (Ok(b), Ok(p)) = (
        build.column_by_name(build_keys[0]),
        probe.column_by_name(probe_keys[0]),
    ) else {
        return false;
    };
    is_int_key_col(b)
        && b.data_type() == p.data_type()
        && b.null_count() == 0
        && p.null_count() == 0
        // IntHashTable numbers build rows as u32.
        && build.row_count() < NO_ROW as usize
}

// ─── Bloom filter over the build side ────────────────────────────────────────

/// Mixes an i64 into a bucket. Join keys are very often dense runs like
/// 0, 10, 20, ..., which would pile into a handful of buckets if the low bits
/// were used directly.
#[inline(always)]
fn bucket_with(mask: usize, key: i64) -> usize {
    let mut h = key as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as usize) & mask
}

/// A Bloom filter of the build side's join keys.
///
/// The probe key is a `String` built by joining the key columns, so every probe
/// row costs an allocation before the hash table is even consulted. Most rows in
/// a selective join match nothing, so that allocation is usually wasted work.
///
/// This filter answers "definitely absent" without allocating: a probe row that
/// the filter rejects skips the `row_key` build entirely. False positives are
/// allowed — they simply fall through to the hash table, which is authoritative.
/// False negatives are not possible, so the join result is unchanged.
struct BuildFilter {
    bits: Vec<u64>,
    mask: usize,
}

impl BuildFilter {
    /// Sized to roughly 16 bits per distinct key, rounded to a power of two, which
    /// keeps the false-positive rate low enough that the filter pays for itself.
    fn with_capacity(n: usize) -> Self {
        let want = (n.max(1) * 16).next_power_of_two().max(64);
        let words = want / 64;
        Self { bits: vec![0u64; words], mask: words - 1 }
    }

    /// Two probes derived from one hash — the standard double-hashing trick, so
    /// only one hash has to be computed per row.
    #[inline]
    fn slots(&self, h: u64) -> (usize, u64, usize, u64) {
        let h1 = h;
        let h2 = h.rotate_left(32) | 1;
        (
            (h1 as usize >> 6) & self.mask,
            1u64 << (h1 & 63),
            (h2 as usize >> 6) & self.mask,
            1u64 << (h2 & 63),
        )
    }

    #[inline]
    fn insert(&mut self, h: u64) {
        let (w1, b1, w2, b2) = self.slots(h);
        self.bits[w1] |= b1;
        self.bits[w2] |= b2;
    }

    /// `false` means the key is definitely not in the build side.
    #[inline]
    fn might_contain(&self, h: u64) -> bool {
        let (w1, b1, w2, b2) = self.slots(h);
        (self.bits[w1] & b1) != 0 && (self.bits[w2] & b2) != 0
    }
}

/// Hash a row's join key from the typed column values, without allocating.
///
/// Correctness rests on one asymmetry: the filter may say "maybe" when the
/// answer is no (a false positive costs one wasted hash-table lookup), but it
/// must never say "no" when the answer is yes — that would silently drop join
/// matches.
///
/// So this has to agree with `row_key`, which compares the *rendered text* of
/// each value. For columns of the same type, equal values render equally, so
/// hashing the typed values introduces no false negatives. Across different
/// types it could — `1` as Int64 and `1.0` as Float64 render alike but hash
/// apart — which is why `filter_is_safe` refuses the filter unless both sides
/// have matching key types.
///
/// The float normalisation only merges values (NaN with NaN, -0.0 with 0.0),
/// which can add false positives and never removes a match.
fn row_key_hash(key_cols: &[&Column], row: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = crate::query::hash::FxHasher::default();

    for (i, col) in key_cols.iter().enumerate() {
        i.hash(&mut h);
        match col {
            Column::Boolean(c)   => c.get(row).hash(&mut h),
            Column::Int32(c)     => c.get(row).hash(&mut h),
            Column::Int64(c)     => c.get(row).hash(&mut h),
            Column::Date32(c)    => c.get(row).hash(&mut h),
            Column::Timestamp(c) => c.get(row).hash(&mut h),
            Column::Utf8(c)      => c.get_str(row).hash(&mut h),
            Column::Float32(c)   => c.get(row).map(|v| {
                if v.is_nan() { f32::NAN.to_bits() } else if v == 0.0 { 0 } else { v.to_bits() }
            }).hash(&mut h),
            Column::Float64(c)   => c.get(row).map(|v| {
                if v.is_nan() { f64::NAN.to_bits() } else if v == 0.0 { 0 } else { v.to_bits() }
            }).hash(&mut h),
            Column::Decimal128(c) => c.inner.get(row).hash(&mut h),
        }
    }
    h.finish()
}

/// Is a Bloom filter safe for this pair of key column sets?
///
/// Only when the two sides agree on type for every key column — see
/// `row_key_hash` for why a type mismatch could turn into a dropped match.
fn filter_is_safe(build: &[&Column], probe: &[&Column]) -> bool {
    build.len() == probe.len()
        && build.iter().zip(probe).all(|(b, p)| b.data_type() == p.data_type())
}

/// Build a column from boxed values, widening narrow numerics as the join's output
/// schema has always done.
fn scalar_vec_to_col(vals: &[ScalarValue], dt: &DataType) -> (DataType, Column) {
    match dt {
        DataType::Boolean => (
            DataType::Boolean,
            Column::Boolean(BooleanColumn::new(vals.iter().map(|v| v.as_bool()).collect())),
        ),
        DataType::Int32 | DataType::Int64 => (
            DataType::Int64,
            Column::Int64(PrimitiveColumn::new(vals.iter().map(|v| v.as_i64()).collect())),
        ),
        DataType::Float32 | DataType::Float64 => (
            DataType::Float64,
            Column::Float64(PrimitiveColumn::new(vals.iter().map(|v| v.as_f64()).collect())),
        ),
        DataType::Utf8 => (
            DataType::Utf8,
            Column::Utf8(Utf8Column::new(vals.iter().map(|v| match v {
                ScalarValue::Utf8(s) => s.clone(),
                _ => None,
            }).collect())),
        ),
        _ => (
            DataType::Int64,
            Column::Int64(PrimitiveColumn::new(vec![None; vals.len()])),
        ),
    }
}

/// Gather one output column through `ScalarValue`, for types the typed path skips.
fn gather_nullable(
    col:     &Column,
    indices: &[u32],
    n:       usize,
) -> FalconResult<Vec<ScalarValue>> {
    let mut vals = Vec::with_capacity(n);
    for &idx in indices {
        vals.push(if idx == NO_ROW {
            ScalarValue::Null
        } else {
            col.get(idx as usize)?
        });
    }
    Ok(vals)
}

/// Gather one output column straight from the source, by `u32` index.
///
/// `NO_ROW` marks an outer join's unmatched side. The path this replaces went
/// through `Vec<ScalarValue>` — one boxed enum per output cell, and a `String`
/// clone per row for text — before converting the whole thing back into a column.
///
/// Returns `None` for types the slow path handles, so the caller keeps its existing
/// behaviour for those. The widening of Int32 to Int64 and Float32 to Float64
/// matches `scalar_vec_to_col` exactly: the output schema must not change.
fn gather_u32_typed(
    col:     &Column,
    indices: &[u32],
    dt:      &DataType,
) -> Option<(DataType, Column)> {
    match (dt, col) {
        (DataType::Int64, Column::Int64(c)) => {
            Some((DataType::Int64, Column::Int64(c.gather_u32(indices))))
        }
        (DataType::Float64, Column::Float64(c)) => {
            Some((DataType::Float64, Column::Float64(c.gather_u32(indices))))
        }
        (DataType::Utf8, Column::Utf8(c)) => {
            Some((DataType::Utf8, Column::Utf8(c.gather_u32(indices))))
        }
        (DataType::Date32, Column::Date32(c)) => {
            Some((DataType::Date32, Column::Date32(c.gather_u32(indices))))
        }
        (DataType::Timestamp, Column::Timestamp(c)) => {
            Some((DataType::Timestamp, Column::Timestamp(c.gather_u32(indices))))
        }
        // Int32/Float32 widen on this path, so they cannot reuse a gather that
        // preserves the source type. They fall through to the boxed path.
        _ => None,
    }
}


/// Drop the columns of one join side that neither the probe nor the output needs.
///
/// Returns the batches unchanged when `keep` is `None` — that is the caller saying
/// it could not prove what is read — or when nothing can be dropped, so the common
/// case costs one scan of the schema and no copying. A `RecordBatch` clone is a
/// refcount bump, so building the trimmed batches copies no data either; the saving
/// is in the concatenation that follows.
fn project_for_join(
    batches: &[RecordBatch],
    keys:    &[&str],
    keep:    Option<&HashSet<String>>,
) -> Vec<RecordBatch> {
    let Some(keep) = keep else { return batches.to_vec() };
    let Some(first) = batches.first() else { return batches.to_vec() };

    let wanted: Vec<bool> = first
        .schema
        .fields
        .iter()
        .map(|f| {
            keys.contains(&f.name.as_str())
                || keep.contains(&f.name)
                // A right-hand column may be renamed on the way out; keep it if the
                // renamed form is what the query asked for.
                || keep.contains(&format!("right.{}", f.name))
        })
        .collect();

    if wanted.iter().all(|w| *w) {
        return batches.to_vec();
    }

    let fields: Vec<_> = first
        .schema
        .fields
        .iter()
        .zip(&wanted)
        .filter(|(_, w)| **w)
        .map(|(f, _)| f.clone())
        .collect();
    let schema = Schema::new(fields);

    batches
        .iter()
        .map(|b| {
            let cols: Vec<_> = b
                .columns
                .iter()
                .zip(&wanted)
                .filter(|(_, w)| **w)
                .map(|(c, _)| std::sync::Arc::clone(c))
                .collect();
            RecordBatch::from_shared(schema.clone(), cols)
                .expect("column count follows the filtered schema")
        })
        .collect()
}

/// Concatenate the batches of one join side.
///
/// Delegates to [`RecordBatch::concat`]. This used to be a local copy that pushed
/// every cell through a `ScalarValue`; with 977 batches per side off a Parquet load
/// it dominated every join, and the typed concat it now calls is the same one
/// `query::sort` uses.
fn merge_batches(batches: &[RecordBatch]) -> FalconResult<RecordBatch> {
    if batches.is_empty() {
        // Callers rely on an empty side producing an empty batch rather than an
        // error, so keep that shape instead of forwarding the empty-input error.
        return RecordBatch::new(Schema::new(vec![]), vec![]);
    }
    RecordBatch::concat(batches)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::RecordBatch,
        types::{DataType, ScalarValue},
    };

    /// The typed integer path and the string path must agree exactly.
    ///
    /// Two implementations of the same thing is how the join ended up 100x behind
    /// in the first place — `merge_batches` had been fixed in `query::sort` and
    /// left slow here. This pins the two paths together for every join type.
    #[test]
    fn typed_and_string_key_paths_agree() {
        let l = orders();
        let r = users();
        let lk = ["user_id"];
        let rk = ["user_id"];

        assert!(
            int_keys_usable(&r, &rk, &l, &lk),
            "null-free Int64 keys should take the typed path"
        );

        for jt in [JoinType::Inner, JoinType::Left, JoinType::Right, JoinType::Full] {
            let mut out = Vec::new();
            for typed in [true, false] {
                let (map, bloom) = build_hash_table(&r, &rk, typed).unwrap();
                out.push(probe(&l, &r, &lk, &rk, &map, bloom.as_ref(), jt.clone(), None).unwrap());
            }
            let (a, b) = (&out[0], &out[1]);
            assert_eq!(a.row_count(), b.row_count(), "{jt:?}: row count differs");
            assert_eq!(a.schema.column_names(), b.schema.column_names(), "{jt:?}: schema differs");
            for ci in 0..a.schema.field_count() {
                for row in 0..a.row_count() {
                    assert_eq!(
                        a.column(ci).get(row).unwrap(),
                        b.column(ci).get(row).unwrap(),
                        "{jt:?}: value differs at column {ci}, row {row}"
                    );
                }
            }
        }
    }

    /// The parallel probe must agree with the serial one — including row order.
    ///
    /// Every other join test here is three rows long, so none of them reach the
    /// 65_536-row threshold and the parallel path went unexercised. Ranges are
    /// concatenated in order, so output order must still follow the left side.
    #[test]
    fn parallel_probe_matches_the_serial_contract() {
        const LEFT_ROWS: usize = 70_000; // over PAR_PROBE_ROWS
        const STRIDE: i64 = 5;

        let l_schema = Schema::new(vec![Field::required("k", DataType::Int64)]);
        let left = RecordBatch::new(
            l_schema,
            vec![ColumnBuilder::int64((0..LEFT_ROWS).map(|i| Some(i as i64)).collect())],
        ).unwrap();

        let r_schema = Schema::new(vec![
            Field::required("k", DataType::Int64),
            Field::required("tag", DataType::Int64),
        ]);
        let n_right = LEFT_ROWS / STRIDE as usize;
        let right = RecordBatch::new(
            r_schema,
            vec![
                ColumnBuilder::int64((0..n_right).map(|i| Some(i as i64 * STRIDE)).collect()),
                ColumnBuilder::int64((0..n_right).map(|i| Some(i as i64)).collect()),
            ],
        ).unwrap();

        let inner = execute_hash_join(
            std::slice::from_ref(&left), std::slice::from_ref(&right),
            &["k"], &["k"], JoinType::Inner,
        ).unwrap();
        assert_eq!(inner.row_count(), n_right, "inner join lost or duplicated rows");

        // Left keys ascend, so a correctly ordered result has an ascending key too.
        let k = inner.column_by_name("k").unwrap();
        let mut prev = i64::MIN;
        for row in 0..inner.row_count() {
            let ScalarValue::Int64(Some(v)) = k.get(row).unwrap() else {
                panic!("expected Int64 key")
            };
            assert!(v > prev, "output not in left order at row {row}: {v} after {prev}");
            assert_eq!(v % STRIDE, 0, "row {row} has key {v}, which has no match");
            prev = v;
        }

        let left_join = execute_hash_join(
            std::slice::from_ref(&left), std::slice::from_ref(&right),
            &["k"], &["k"], JoinType::Left,
        ).unwrap();
        assert_eq!(left_join.row_count(), LEFT_ROWS, "left join dropped unmatched rows");
    }

    /// The parallel semi/anti probe must agree with the serial one.
    ///
    /// The existing semi/anti coverage is a handful of rows, well under
    /// PAR_PROBE_ROWS, so the chunked path would otherwise go unexercised — the
    /// same gap that left the radix sort and the inner-join probe untested when
    /// they were first parallelised.
    #[test]
    fn parallel_semi_anti_probe_matches_the_serial_contract() {
        const LEFT_ROWS: usize = 70_000; // over PAR_PROBE_ROWS
        const STRIDE: i64 = 5;

        let schema = Schema::new(vec![Field::required("k", DataType::Int64)]);
        let left = RecordBatch::new(
            schema.clone(),
            vec![ColumnBuilder::int64((0..LEFT_ROWS).map(|i| Some(i as i64)).collect())],
        )
        .unwrap();
        let n_right = LEFT_ROWS / STRIDE as usize;
        let right = RecordBatch::new(
            schema,
            vec![ColumnBuilder::int64(
                (0..n_right).map(|i| Some(i as i64 * STRIDE)).collect(),
            )],
        )
        .unwrap();

        let semi = execute_hash_semi_anti_join(
            std::slice::from_ref(&left), std::slice::from_ref(&right),
            &["k"], &["k"], false,
        ).unwrap();
        assert_eq!(semi.row_count(), n_right, "semi join kept the wrong number of rows");

        let anti = execute_hash_semi_anti_join(
            std::slice::from_ref(&left), std::slice::from_ref(&right),
            &["k"], &["k"], true,
        ).unwrap();
        assert_eq!(
            anti.row_count(),
            LEFT_ROWS - n_right,
            "anti join kept the wrong number of rows"
        );

        // Semi keeps exactly the multiples of STRIDE, in left order.
        let k = semi.column_by_name("k").unwrap();
        let mut prev = i64::MIN;
        for row in 0..semi.row_count() {
            let ScalarValue::Int64(Some(v)) = k.get(row).unwrap() else {
                panic!("expected Int64 key")
            };
            assert!(v > prev, "semi output not in left order at row {row}");
            assert_eq!(v % STRIDE, 0, "semi kept {v}, which has no match");
            prev = v;
        }
    }

    /// Build and probe must hash bloom keys the same way.
    ///
    /// The filter may only ever say "maybe" wrongly, never "no". If the two sides
    /// disagreed on the hash it would reject real matches, and the join would come
    /// back short with nothing failing loudly — so this drives a typed join past the
    /// bloom threshold and checks every match is still found.
    #[test]
    fn typed_bloom_hash_agrees_between_build_and_probe() {
        // Well past BLOOM_MIN_ROWS so the filter is actually built.
        const BUILD_ROWS: usize = 20_000;
        const PROBE_ROWS: usize = 60_000;
        const STRIDE: i64 = 3;

        let schema = Schema::new(vec![Field::required("k", DataType::Int64)]);
        let left = RecordBatch::new(
            schema.clone(),
            vec![ColumnBuilder::int64((0..PROBE_ROWS).map(|i| Some(i as i64)).collect())],
        )
        .unwrap();
        let right = RecordBatch::new(
            schema,
            vec![ColumnBuilder::int64(
                (0..BUILD_ROWS).map(|i| Some(i as i64 * STRIDE)).collect(),
            )],
        )
        .unwrap();

        let out = execute_hash_join(
            std::slice::from_ref(&left),
            std::slice::from_ref(&right),
            &["k"], &["k"], JoinType::Inner,
        )
        .unwrap();

        // Every left key that is a multiple of STRIDE and within the build range.
        let expected = (0..PROBE_ROWS)
            .filter(|i| *i as i64 % STRIDE == 0 && (*i as i64 / STRIDE) < BUILD_ROWS as i64)
            .count();
        assert_eq!(
            out.row_count(), expected,
            "bloom dropped matches — build and probe hashes disagree"
        );
    }

    /// A nullable key column must stay on the string path.
    ///
    /// Not a performance choice: `row_key` renders NULL as the text "NULL", so the
    /// string path joins NULL keys to each other. That is wrong per SQL, but it is
    /// the behaviour callers have today, and a fast path that silently disagreed
    /// with it would be worse than a slow one.
    #[test]
    fn nullable_keys_do_not_take_the_typed_path() {
        let schema = Schema::new(vec![Field::nullable("k", DataType::Int64)]);
        let with_null = RecordBatch::new(
            schema.clone(),
            vec![ColumnBuilder::int64(vec![Some(1), None, Some(3)])],
        ).unwrap();
        let no_null = RecordBatch::new(
            schema,
            vec![ColumnBuilder::int64(vec![Some(1), Some(2), Some(3)])],
        ).unwrap();

        assert!(!int_keys_usable(&with_null, &["k"], &no_null, &["k"]));
        assert!(!int_keys_usable(&no_null, &["k"], &with_null, &["k"]));
        assert!(int_keys_usable(&no_null, &["k"], &no_null, &["k"]));
    }

    /// The in-place probe must return exactly what the merged probe returns.
    ///
    /// Same rows, same order, same columns. The only difference between the two
    /// paths is whether the left side was concatenated first, and that is an
    /// implementation detail the caller must not be able to observe.
    #[test]
    fn probe_in_place_matches_the_merged_contract() {
        use crate::storage::{column::ColumnBuilder, schema::Field, types::DataType};

        let schema = Schema::new(vec![
            Field::required("k", DataType::Int64),
            Field::nullable("payload", DataType::Utf8),
        ]);
        // Keys repeat so some left rows match several right rows, and some match
        // none — both are cases where order and cardinality could drift.
        let rows = 5_000usize;
        let make = |batch_rows: usize| -> Vec<RecordBatch> {
            let mut out = Vec::new();
            let mut start = 0;
            while start < rows {
                let n = batch_rows.min(rows - start);
                out.push(
                    RecordBatch::new(
                        schema.clone(),
                        vec![
                            ColumnBuilder::int64(
                                (0..n).map(|i| Some(((start + i) % 700) as i64)).collect(),
                            ),
                            ColumnBuilder::utf8_from_str(
                                (0..n)
                                    .map(|i| if (start + i) % 11 == 0 { None } else { Some("p") })
                                    .collect(),
                            ),
                        ],
                    )
                    .unwrap(),
                );
                start += n;
            }
            out
        };

        let right_schema = Schema::new(vec![Field::required("rk", DataType::Int64)]);
        let right = vec![RecordBatch::new(
            right_schema,
            vec![ColumnBuilder::int64((0..400).map(|i| Some(i as i64 * 3)).collect())],
        )
        .unwrap()];

        for join_type in [JoinType::Inner, JoinType::Left] {
            let many = make(137); // many batches -> in-place path
            let one  = make(rows); // single batch -> merged path
            assert!(many.len() > 1 && one.len() == 1);

            let a = execute_hash_join(&many, &right, &["k"], &["rk"], join_type.clone()).unwrap();
            let b = execute_hash_join(&one, &right, &["k"], &["rk"], join_type.clone()).unwrap();

            assert_eq!(a.row_count(), b.row_count(), "{join_type:?}: row count");
            assert_eq!(
                a.schema.column_names(),
                b.schema.column_names(),
                "{join_type:?}: columns",
            );
            for c in 0..a.column_count() {
                for r in 0..a.row_count() {
                    assert_eq!(
                        a.column(c).get(r).unwrap(),
                        b.column(c).get(r).unwrap(),
                        "{join_type:?}: column {c} row {r}",
                    );
                }
            }
        }
    }

    /// The dense output builder must agree with the general one.
    ///
    /// The test above uses a nullable `Utf8` payload, which `build_output_dense`
    /// declines — so it exercises the fallback, not the fast path. This one is
    /// all-primitive and NULL-free, which is exactly the shape the fast path
    /// claims, and compares it against the merged path that does not use it.
    ///
    /// Worth its own test because the two builders write the output differently:
    /// one fills disjoint slices of a single buffer from several threads, the
    /// other concatenates a batch per unit. An off-by-one in the slice hand-out
    /// would silently shift rows rather than fail.
    #[test]
    fn dense_output_matches_the_general_builder() {
        use crate::storage::{column::ColumnBuilder, schema::Field, types::DataType};

        let schema = Schema::new(vec![
            Field::required("k", DataType::Int64),
            Field::required("v", DataType::Float64),
        ]);
        let rows = 5_000usize;
        let make = |batch_rows: usize| -> Vec<RecordBatch> {
            let mut out = Vec::new();
            let mut start = 0;
            while start < rows {
                let n = batch_rows.min(rows - start);
                out.push(
                    RecordBatch::new(
                        schema.clone(),
                        vec![
                            ColumnBuilder::int64(
                                (0..n).map(|i| Some(((start + i) % 700) as i64)).collect(),
                            ),
                            ColumnBuilder::float64(
                                (0..n).map(|i| Some((start + i) as f64 * 0.5)).collect(),
                            ),
                        ],
                    )
                    .unwrap(),
                );
                start += n;
            }
            out
        };

        // Repeated keys on both sides, so some left rows match several right rows
        // and the output is longer than either input. Uneven batch sizes so the
        // last unit is a different length from the rest.
        let right = vec![RecordBatch::new(
            Schema::new(vec![Field::required("rk", DataType::Int64)]),
            vec![ColumnBuilder::int64(
                (0..800).map(|i| Some((i % 400) as i64 * 3)).collect(),
            )],
        )
        .unwrap()];

        let many = make(137);
        let one = make(rows);
        assert!(many.len() > 1 && one.len() == 1);

        let dense = execute_hash_join(&many, &right, &["k"], &["rk"], JoinType::Inner).unwrap();
        let plain = execute_hash_join(&one, &right, &["k"], &["rk"], JoinType::Inner).unwrap();

        // Every matching right key appears twice, so a correct result is twice the
        // number of matching left rows -- fan-out, not a filter. Only keys that are
        // multiples of 3 and below 700 match, so this is well under `rows`.
        assert!(
            dense.row_count() > 3_000,
            "expected fan-out over ~234 shared keys, got {}",
            dense.row_count(),
        );
        assert_eq!(dense.row_count(), plain.row_count());
        assert_eq!(dense.schema.column_names(), plain.schema.column_names());
        for c in 0..dense.column_count() {
            for r in 0..dense.row_count() {
                assert_eq!(
                    dense.column(c).get(r).unwrap(),
                    plain.column(c).get(r).unwrap(),
                    "column {c} row {r}",
                );
            }
        }
    }

    /// Which output builder is faster, measured without leaving the process.
    ///
    /// Ignored by default: it is a measurement, not an assertion, and it takes a
    /// second. Run it with
    /// `cargo test --release --lib dense_vs_general -- --ignored --nocapture`.
    ///
    /// It exists because this machine drifts. Two criterion runs taken twenty
    /// minutes apart disagreed by 50-90% on benchmarks the change cannot touch --
    /// SEMI and ANTI do not go through this code at all, and they moved as much as
    /// the ones that do. Comparing a run against another run was measuring the
    /// laptop. Both builders are called here from one process, alternately, on
    /// identical input, so whatever the clock is doing it is doing to both.
    #[test]
    #[ignore]
    fn dense_vs_general_output_builder() {
        use crate::storage::{column::ColumnBuilder, schema::Field, types::DataType};
        use std::time::Instant;

        const BATCHES:    usize = 16;
        const BATCH_ROWS: usize = 65_536;
        const RIGHT_ROWS: usize = 100_000;
        const PER_UNIT:   usize = 6_250; // ~100k output rows, the benchmark's shape

        let left_schema = Schema::new(vec![
            Field::required("k", DataType::Int64),
            Field::required("v", DataType::Int64),
        ]);
        let left: Vec<RecordBatch> = (0..BATCHES)
            .map(|b| {
                RecordBatch::new(
                    left_schema.clone(),
                    vec![
                        ColumnBuilder::int64(
                            (0..BATCH_ROWS).map(|i| Some((b * BATCH_ROWS + i) as i64)).collect(),
                        ),
                        ColumnBuilder::int64((0..BATCH_ROWS).map(|i| Some(i as i64)).collect()),
                    ],
                )
                .unwrap()
            })
            .collect();
        let right = RecordBatch::new(
            Schema::new(vec![Field::required("rk", DataType::Int64)]),
            vec![ColumnBuilder::int64((0..RIGHT_ROWS).map(|i| Some(i as i64)).collect())],
        )
        .unwrap();

        // One unit per batch, indices scattered so the gather pays realistic cache
        // misses rather than walking memory in order.
        let matches: Vec<(usize, Vec<u32>, Vec<u32>)> = (0..BATCHES)
            .map(|b| {
                let li: Vec<u32> = (0..PER_UNIT)
                    .map(|i| ((i * 7919) % BATCH_ROWS) as u32)
                    .collect();
                let ri: Vec<u32> = (0..PER_UNIT)
                    .map(|i| ((i * 104_729) % RIGHT_ROWS) as u32)
                    .collect();
                (b, li, ri)
            })
            .collect();

        let dense = || {
            let t = Instant::now();
            let out = build_output_dense(&left, &right, &matches, None).unwrap();
            let e = t.elapsed();
            assert!(out.is_some(), "dense builder declined the shape it is meant for");
            assert_eq!(out.unwrap().row_count(), BATCHES * PER_UNIT);
            e
        };
        let general = || {
            let t = Instant::now();
            let parts: FalconResult<Vec<RecordBatch>> = matches
                .par_iter()
                .map(|(bi, li, ri)| build_output(&left[*bi], &right, li, ri, None))
                .collect();
            let out = RecordBatch::concat(&parts.unwrap()).unwrap();
            let e = t.elapsed();
            assert_eq!(out.row_count(), BATCHES * PER_UNIT);
            e
        };

        for _ in 0..3 {
            dense();
            general();
        }
        // Alternate, then take each side's minimum: the fastest observation is the
        // one least polluted by whatever else the machine did during it.
        let mut d = Vec::new();
        let mut g = Vec::new();
        for _ in 0..15 {
            d.push(dense());
            g.push(general());
        }
        let dmin = d.iter().min().unwrap();
        let gmin = g.iter().min().unwrap();

        println!("output build, {} rows over {BATCHES} units:", BATCHES * PER_UNIT);
        println!("  one allocation per column = {dmin:?}");
        println!("  batch per unit + concat   = {gmin:?}");
        println!(
            "  dense is {:.2}x the general builder",
            dmin.as_secs_f64() / gmin.as_secs_f64(),
        );
    }

    /// Does the Bloom filter earn its place on the typed (integer) path?
    ///
    /// Its documented purpose is the *string* path, where every probe row builds
    /// a `String` key before the table is consulted and a rejection skips that
    /// allocation. The typed path has no such allocation: a miss there is one
    /// read of `heads`. The filter adds two reads, so on paper it could be pure
    /// overhead -- plus roughly 200 KB allocated and 100k insertions on the build
    /// side of every join.
    ///
    /// `probe_one` already takes `bloom: Option<&BuildFilter>`, so both answers
    /// can be measured from one process by passing `Some` or `None`. That matters
    /// more than usual here: this machine drifted 1.6x between two runs of the
    /// same diagnostic minutes apart, which is larger than the effect being
    /// looked for. Anything compared across process runs on it is noise.
    ///
    /// Run with
    /// `cargo test --release --lib bloom_filter_on_the_typed_path -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bloom_filter_on_the_typed_path() {
        use crate::storage::{column::ColumnBuilder, schema::Field, types::DataType};
        use std::time::Instant;

        const RIGHT_ROWS: usize = 100_000;
        const LEFT_ROWS:  usize = 1_000_000;

        let key = |name: &str, vals: Vec<i64>| {
            RecordBatch::new(
                Schema::new(vec![Field::required(name, DataType::Int64)]),
                vec![ColumnBuilder::int64(vals.into_iter().map(Some).collect())],
            )
            .unwrap()
        };

        // Build side keys 0,10,20,... so a tenth of the probe side matches: the
        // benchmark's selectivity. A run where nothing matches flatters the filter,
        // because then it answers every row without the table being touched.
        let right = key("rk", (0..RIGHT_ROWS as i64).map(|i| i * 10).collect());
        let left  = key("k", (0..LEFT_ROWS as i64).collect());

        let right_key_cols: Vec<&Column> = vec![right.column(0)];
        let left_key_cols:  Vec<&Column> = vec![left.column(0)];

        let (table, filter) = build_hash_table(&right, &["rk"], true).unwrap();
        let filter = filter.expect("100k rows should be over BLOOM_MIN_ROWS");

        let probe_all = |bloom: Option<&BuildFilter>| -> (std::time::Duration, usize) {
            let mut li = Vec::new();
            let mut ri = Vec::new();
            let mut scratch = Vec::new();
            let t = Instant::now();
            for row in 0..LEFT_ROWS {
                probe_one(
                    row, &left_key_cols, &table, bloom, &JoinType::Inner,
                    &mut li, &mut ri, None, &mut scratch,
                )
                .unwrap();
            }
            (t.elapsed(), li.len())
        };

        // Both answers must be identical -- the filter is an accelerator, and one
        // that changed the result would be a correctness bug, not a slow path.
        let (_, with_n)    = probe_all(Some(&filter));
        let (_, without_n) = probe_all(None);
        assert_eq!(with_n, without_n, "the filter changed the join result");
        assert_eq!(with_n, RIGHT_ROWS, "expected one match per build row");

        let mut on  = Vec::new();
        let mut off = Vec::new();
        for _ in 0..5 {
            on.push(probe_all(Some(&filter)).0);
            off.push(probe_all(None).0);
        }
        let on_min  = on.iter().min().unwrap();
        let off_min = off.iter().min().unwrap();

        // Build cost of the filter itself, measured the same alternating way.
        let mut b_on  = Vec::new();
        let mut b_off = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            let f = BuildFilter::with_capacity(RIGHT_ROWS);
            std::hint::black_box(&f);
            b_on.push(t.elapsed());

            let t = Instant::now();
            let (tb, _) = build_hash_table(&right, &["rk"], true).unwrap();
            std::hint::black_box(&tb);
            b_off.push(t.elapsed());
        }

        println!("probe of {LEFT_ROWS} rows, 1 in 10 matching:");
        println!("  filter on  = {on_min:?}");
        println!("  filter off = {off_min:?}");
        println!(
            "  filter makes the probe {:.2}x",
            on_min.as_secs_f64() / off_min.as_secs_f64(),
        );
        println!("  allocating the filter alone = {:?}", b_on.iter().min().unwrap());
        println!("  whole build incl. filter    = {:?}", b_off.iter().min().unwrap());
    }

    /// What does building the hash table actually spend its time on?
    ///
    /// The build runs on one thread while the probe runs on all of them, so it is
    /// a larger share of a join than its cost suggests: about 1.2 ms of a 4.4 ms
    /// 1M-row join, none of it parallel. Before parallelising it, find out how
    /// much of it is even the insert loop -- `IntHashTable::with_rows` allocates
    /// the bucket heads and fills them with a sentinel, and allocation on this
    /// machine was measured at roughly 0.27 ms per fresh megabyte.
    ///
    /// Measured in one process, alternating, for the reason given on
    /// `bloom_filter_on_the_typed_path`.
    ///
    /// Run with
    /// `cargo test --release --lib hash_table_build_breakdown -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn hash_table_build_breakdown() {
        use crate::storage::{column::ColumnBuilder, schema::Field, types::DataType};
        use std::time::Instant;

        const ROWS: usize = 100_000;

        let right = RecordBatch::new(
            Schema::new(vec![Field::required("rk", DataType::Int64)]),
            vec![ColumnBuilder::int64((0..ROWS as i64).map(|i| Some(i * 10)).collect())],
        )
        .unwrap();
        let col = Arc::clone(&right.columns[0]);

        let alloc_only = || {
            let t = Instant::now();
            let table = IntHashTable::with_rows(ROWS, Arc::clone(&col));
            let e = t.elapsed();
            std::hint::black_box(&table);
            e
        };
        let alloc_and_insert = || {
            let t = Instant::now();
            let mut table = IntHashTable::with_rows(ROWS, Arc::clone(&col));
            for row in 0..ROWS {
                table.insert(int_key(&col, row), row);
            }
            let e = t.elapsed();
            std::hint::black_box(&table);
            e
        };
        let whole_build = || {
            let t = Instant::now();
            let built = build_hash_table(&right, &["rk"], true).unwrap();
            let e = t.elapsed();
            std::hint::black_box(&built);
            e
        };

        for _ in 0..3 {
            alloc_only();
            alloc_and_insert();
            whole_build();
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        for _ in 0..9 {
            a.push(alloc_only());
            b.push(alloc_and_insert());
            c.push(whole_build());
        }
        let (a, b, c) = (
            *a.iter().min().unwrap(),
            *b.iter().min().unwrap(),
            *c.iter().min().unwrap(),
        );

        println!("build side, {ROWS} rows:");
        println!("  allocate the table       = {a:?}");
        println!("  allocate + insert        = {b:?}   (insert alone {:?})", b.saturating_sub(a));
        println!("  whole build incl. filter = {c:?}   (filter adds {:?})", c.saturating_sub(b));
        println!(
            "  allocation is {:.0}% of the build",
            a.as_secs_f64() / c.as_secs_f64() * 100.0,
        );
    }

    /// Is 16 bits per key the right size for the filter?
    ///
    /// It is the hottest randomly-addressed structure in a join: touched twice per
    /// build row and twice per probe row. At 16 bits per key, 100k keys make a
    /// 256 KB filter; at 8 they make 128 KB, which halves the footprint on both
    /// sides at the cost of a higher false-positive rate. A false positive costs
    /// one hash-table lookup -- exactly what the filter was avoiding -- so the
    /// trade is only worth taking if the cache win is bigger than the extra
    /// lookups, which is a question for a measurement rather than for arithmetic.
    ///
    /// `with_capacity` derives the size from a key count, so halving the count
    /// passed to it is how 8-bits-per-key is obtained without touching the
    /// production constant.
    ///
    /// Run with
    /// `cargo test --release --lib bloom_filter_sizing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bloom_filter_sizing() {
        use crate::storage::{column::ColumnBuilder, schema::Field, types::DataType};
        use std::time::Instant;

        const RIGHT_ROWS: usize = 100_000;
        const LEFT_ROWS:  usize = 1_000_000;

        let right = RecordBatch::new(
            Schema::new(vec![Field::required("rk", DataType::Int64)]),
            vec![ColumnBuilder::int64((0..RIGHT_ROWS as i64).map(|i| Some(i * 10)).collect())],
        )
        .unwrap();
        let left = RecordBatch::new(
            Schema::new(vec![Field::required("k", DataType::Int64)]),
            vec![ColumnBuilder::int64((0..LEFT_ROWS as i64).map(Some).collect())],
        )
        .unwrap();
        let left_key_cols: Vec<&Column> = vec![left.column(0)];
        let col = right.column(0);

        let (table, _) = build_hash_table(&right, &["rk"], true).unwrap();

        let make_filter = |bits_per_key: usize| {
            let mut f = BuildFilter::with_capacity(RIGHT_ROWS * bits_per_key / 16);
            let t = Instant::now();
            for row in 0..RIGHT_ROWS {
                f.insert(int_key_hash(int_key(col, row)));
            }
            (f, t.elapsed())
        };

        let probe_all = |bloom: &BuildFilter| -> (std::time::Duration, usize) {
            let mut li = Vec::new();
            let mut ri = Vec::new();
            let mut scratch = Vec::new();
            let t = Instant::now();
            for row in 0..LEFT_ROWS {
                probe_one(
                    row, &left_key_cols, &table, Some(bloom), &JoinType::Inner,
                    &mut li, &mut ri, None, &mut scratch,
                )
                .unwrap();
            }
            (t.elapsed(), li.len())
        };

        let (wide, _)   = make_filter(16);
        let (narrow, _) = make_filter(8);

        // A smaller filter may say "maybe" more often, but it must never say "no"
        // to a key that is present -- the result has to be identical either way.
        let (_, n_wide)   = probe_all(&wide);
        let (_, n_narrow) = probe_all(&narrow);
        assert_eq!(n_wide, RIGHT_ROWS);
        assert_eq!(n_narrow, RIGHT_ROWS, "narrower filter dropped matches");

        let mut pw = Vec::new();
        let mut pn = Vec::new();
        let mut bw = Vec::new();
        let mut bn = Vec::new();
        for _ in 0..5 {
            pw.push(probe_all(&wide).0);
            pn.push(probe_all(&narrow).0);
            bw.push(make_filter(16).1);
            bn.push(make_filter(8).1);
        }
        let pw = *pw.iter().min().unwrap();
        let pn = *pn.iter().min().unwrap();
        let bw = *bw.iter().min().unwrap();
        let bn = *bn.iter().min().unwrap();

        println!("filter sizing, {RIGHT_ROWS} keys / {LEFT_ROWS} probes:");
        println!("  16 bits/key (256 KB): build {bw:?}, probe {pw:?}");
        println!("   8 bits/key (128 KB): build {bn:?}, probe {pn:?}");
        println!(
            "  narrow is {:.2}x on build, {:.2}x on probe",
            bn.as_secs_f64() / bw.as_secs_f64(),
            pn.as_secs_f64() / pw.as_secs_f64(),
        );
    }

    /// Does the join table need its own copy of the build keys? It did not.
    ///
    /// `IntHashTable` used to hold one `i64` per build row -- 800 KB at 100k --
    /// duplicating the key column so the chain walk could check a node. It now
    /// shares the column through an `Arc` instead. Measured on both sides before
    /// the change, four runs, 100k build rows against 1M probes:
    ///
    ///   build   with 603-1170us     without 553-589us
    ///   probe   with 11.20-12.37ms  without 10.95-11.35ms
    ///
    /// Faster on both. That was not obvious: an extra indirection per chain node
    /// could easily have cost more than the allocation saved. It does not,
    /// because the column is resident anyway and the duplicate was competing with
    /// the buckets for cache. The figures without it are also far steadier -- the
    /// copy had made the build sensitive to allocator state.
    ///
    /// This keeps the comparison runnable: production has only the shared-column
    /// shape now, so the copying one lives here.
    ///
    /// Run with
    /// `cargo test --release --lib keys_array_worth_it -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn keys_array_worth_it() {
        use std::time::Instant;

        const BUILD: usize = 100_000;
        const PROBE: usize = 1_000_000;

        let right = int_key_batch("rk", (0..BUILD as i64).map(|i| i * 10).collect());
        let left = int_key_batch("k", (0..PROBE as i64).collect());
        let rcol = Arc::clone(&right.columns[0]);
        let lcol = left.column(0);

        // What production does now.
        let build_shared = || {
            let t = Instant::now();
            let mut table = IntHashTable::with_rows(BUILD, Arc::clone(&rcol));
            for row in 0..BUILD {
                table.insert(int_key(&rcol, row), row);
            }
            let e = t.elapsed();
            std::hint::black_box(&table);
            e
        };

        // What it did before: the same chains plus a per-row copy of the keys.
        let build_copying = || {
            let t = Instant::now();
            let mut table = IntHashTable::with_rows(BUILD, Arc::clone(&rcol));
            let mut keys: Vec<i64> = Vec::with_capacity(BUILD);
            for row in 0..BUILD {
                let k = int_key(&rcol, row);
                keys.push(k);
                table.insert(k, row);
            }
            let e = t.elapsed();
            std::hint::black_box((&table, &keys));
            e
        };

        let table = {
            let mut t = IntHashTable::with_rows(BUILD, Arc::clone(&rcol));
            for row in 0..BUILD {
                t.insert(int_key(&rcol, row), row);
            }
            t
        };
        let keys: Vec<i64> = (0..BUILD).map(|r| int_key(&rcol, r)).collect();

        let probe_via_column = || {
            let mut hits = 0usize;
            let t = Instant::now();
            for row in 0..PROBE {
                let key = int_key(lcol, row);
                let mut cur = table.heads[bucket_with(table.mask, key)];
                while cur != NO_ROW {
                    let r = cur as usize;
                    if int_key(&rcol, r) == key {
                        hits += 1;
                    }
                    cur = table.next[r];
                }
            }
            (t.elapsed(), hits)
        };
        let probe_via_keys = || {
            let mut hits = 0usize;
            let t = Instant::now();
            for row in 0..PROBE {
                let key = int_key(lcol, row);
                let mut cur = table.heads[bucket_with(table.mask, key)];
                while cur != NO_ROW {
                    let r = cur as usize;
                    if keys[r] == key {
                        hits += 1;
                    }
                    cur = table.next[r];
                }
            }
            (t.elapsed(), hits)
        };

        let (_, a) = probe_via_column();
        let (_, b) = probe_via_keys();
        assert_eq!(a, b, "the two chain walks must find the same matches");
        assert_eq!(a, BUILD, "every build row should match exactly once");

        let bench = |label: &str, f: &dyn Fn() -> std::time::Duration, reps: usize| {
            for _ in 0..2 { f(); }
            let best = (0..reps).map(|_| f()).min().unwrap();
            println!("  {label:>26}: {best:>9.2?}");
            best
        };

        println!("100k build rows, {PROBE} probes:");
        let bo = bench("build, shared column", &build_shared, 9);
        let bw = bench("build, copied keys", &build_copying, 9);
        let po = bench("probe, reads the column", &|| probe_via_column().0, 3);
        let pw = bench("probe, reads copied keys", &|| probe_via_keys().0, 3);
        println!("  build saves {:?}", bw.saturating_sub(bo));
        println!("  probe saves {:?}", pw.saturating_sub(po));
        // The probe runs on every core and the build on one, so the probe column
        // is worth roughly a tenth of itself in production.
        println!(
            "  with the probe scaled by 10x for parallelism: {:.0}us -> {:.0}us",
            bw.as_secs_f64() * 1e6 + pw.as_secs_f64() * 1e5,
            bo.as_secs_f64() * 1e6 + po.as_secs_f64() * 1e5,
        );
    }

    /// What does a join on a text key cost against the same join on an integer?
    ///
    /// `int_keys_usable` gates the typed path on integer keys. A text key falls
    /// to `row_key`, which builds a `String` per probe row before the hash table
    /// is consulted -- the Bloom filter exists precisely to skip that allocation
    /// for rows that cannot match, which is an admission of how expensive it is.
    ///
    /// Joining on a string column is ordinary SQL and the benchmark suite joins
    /// on `id`, an integer, so this is unmeasured.
    ///
    /// Run with
    /// `cargo test --release --lib join_by_key_type -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn join_by_key_type() {
        use crate::storage::{column::ColumnBuilder, schema::Field, types::DataType};
        use std::time::Instant;

        const BUILD: usize = 100_000;
        const PROBE: usize = 1_000_000;

        let int_batch = |name: &str, vals: Vec<i64>| {
            RecordBatch::new(
                Schema::new(vec![Field::required(name, DataType::Int64)]),
                vec![ColumnBuilder::int64(vals.into_iter().map(Some).collect())],
            )
            .unwrap()
        };
        let text_batch = |name: &str, vals: Vec<String>| {
            RecordBatch::new(
                Schema::new(vec![Field::required(name, DataType::Utf8)]),
                vec![ColumnBuilder::utf8(vals.into_iter().map(Some).collect())],
            )
            .unwrap()
        };

        // Same shape either way: a tenth of the probe side matches.
        let right_i = int_batch("rk", (0..BUILD as i64).map(|i| i * 10).collect());
        let left_i = int_batch("k", (0..PROBE as i64).collect());
        let right_s = text_batch("rk", (0..BUILD).map(|i| format!("k{:08}", i * 10)).collect());
        let left_s = text_batch("k", (0..PROBE).map(|i| format!("k{i:08}")).collect());

        let time = |label: &str, left: &RecordBatch, right: &RecordBatch| {
            let run = || {
                let out = execute_hash_join(
                    std::slice::from_ref(left),
                    std::slice::from_ref(right),
                    &["k"],
                    &["rk"],
                    JoinType::Inner,
                )
                .unwrap();
                assert_eq!(out.row_count(), BUILD, "a tenth of the probe side should match");
            };
            for _ in 0..2 { run(); }
            let mut best = std::time::Duration::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                run();
                best = best.min(t.elapsed());
            }
            println!("  {label:>28}: {best:>9.2?}");
            best
        };

        println!("INNER JOIN, {BUILD} build rows against {PROBE} probe rows:");
        let i = time("integer key", &left_i, &right_i);
        let s = time("text key", &left_s, &right_s);
        println!("  the text key costs {:.1}x the integer one", s.as_secs_f64() / i.as_secs_f64());
    }

    /// How many buckets per row should the join table have?
    ///
    /// `with_rows` allocates two buckets per row, so 100k build rows take a 1 MB
    /// `heads` array on top of 400 KB of chain links and 800 KB of keys. The
    /// build was measured as allocation-bound rather than compute-bound -- two
    /// attempts at parallelising it both came out slower -- so 2.2 MB per join is
    /// the thing to attack, and `heads` is the largest piece of it.
    ///
    /// One bucket per row halves it. The cost is collisions: a load factor of 1.0
    /// instead of 0.5 makes chains longer, and the probe walks those chains. Both
    /// sides have to be measured, because a build that saves less than the probe
    /// loses is not a saving.
    ///
    /// Run with
    /// `cargo test --release --lib bucket_density -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bucket_density() {
        use std::time::Instant;

        const BUILD: usize = 100_000;
        const PROBE: usize = 1_000_000;

        /// `with_rows`, parameterised by how many buckets each row gets.
        fn build_with(col: Arc<Column>, n: usize, per_row: usize) -> IntHashTable {
            let buckets = (n.max(1) * per_row).next_power_of_two();
            let mut t = IntHashTable {
                mask: buckets - 1,
                heads: vec![NO_ROW; buckets],
                next: vec![NO_ROW; n],
                key_col: Arc::clone(&col),
            };
            for row in 0..n {
                t.insert(int_key(&col, row), row);
            }
            t
        }

        // Keys 0,10,20,... on the build side and 0..1M on the probe side: a tenth
        // of the probe matches, which is the benchmark's selectivity.
        let right = int_key_batch("rk", (0..BUILD as i64).map(|i| i * 10).collect());
        let left = int_key_batch("k", (0..PROBE as i64).collect());
        let rcol = Arc::clone(&right.columns[0]);
        let left_cols: Vec<&Column> = vec![left.column(0)];

        for per_row in [1usize, 2, 4] {
            let buckets = (BUILD * per_row).next_power_of_two();

            let build = || {
                let t = Instant::now();
                let table = build_with(Arc::clone(&rcol), BUILD, per_row);
                let e = t.elapsed();
                std::hint::black_box(&table);
                e
            };
            for _ in 0..3 { build(); }
            let b = (0..9).map(|_| build()).min().unwrap();

            let map = JoinKeyMap::Int(build_with(Arc::clone(&rcol), BUILD, per_row));
            let probe = || {
                let mut li = Vec::new();
                let mut ri = Vec::new();
                let mut scratch = Vec::new();
                let t = Instant::now();
                for row in 0..PROBE {
                    probe_one(
                        row, &left_cols, &map, None,
                        &JoinType::Inner, &mut li, &mut ri, None, &mut scratch,
                    )
                    .unwrap();
                }
                (t.elapsed(), li.len())
            };
            let (_, matches) = probe();
            assert_eq!(matches, BUILD, "every build row should match exactly once");
            for _ in 0..2 { probe(); }
            let pr = (0..3).map(|_| probe().0).min().unwrap();

            println!(
                "  {per_row} bucket(s)/row ({} buckets, {} KB heads): build {b:>9.2?}, probe {pr:>9.2?}, total {:>9.2?}",
                buckets,
                buckets * 4 / 1024,
                b + pr,
            );
        }
    }

    fn int_key_batch(name: &str, vals: Vec<i64>) -> RecordBatch {
        use crate::storage::{column::ColumnBuilder, schema::Field, types::DataType};
        RecordBatch::new(
            Schema::new(vec![Field::required(name, DataType::Int64)]),
            vec![ColumnBuilder::int64(vals.into_iter().map(Some).collect())],
        )
        .unwrap()
    }

    /// The parallel build must produce the same table, not an equivalent one.
    ///
    /// Chain order within a bucket decides the order of a join's output rows, so
    /// "the same matches in some order" is not good enough -- it would make query
    /// results depend on how work happened to be split. Asserting the arrays
    /// themselves is the strongest statement available and costs nothing.
    ///
    /// Duplicate keys are the case that matters: a bucket with one row cannot
    /// have its order changed.
    #[test]
    fn parallel_build_is_identical_to_serial() {
        for &(n, period) in &[(1usize, 1i64), (7, 3), (1000, 97), (50_000, 977)] {
            let batch = int_key_batch("k", (0..n as i64).map(|i| (i % period) * 10).collect());
            let col = Arc::clone(&batch.columns[0]);

            let mut serial = IntHashTable::with_rows(n, Arc::clone(&col));
            for row in 0..n {
                serial.insert(int_key(&col, row), row);
            }
            for (label, built) in [
                ("bucket-partitioned", IntHashTable::build_parallel(Arc::clone(&col), n)),
                ("radix", IntHashTable::build_radix(Arc::clone(&col), n)),
            ] {
                let parallel = built;

            assert_eq!(serial.mask, parallel.mask, "n={n} {label}: mask");
            assert!(
                Arc::ptr_eq(&serial.key_col, &parallel.key_col),
                "n={n} {label}: both should share the same key column",
            );
            assert_eq!(serial.heads, parallel.heads, "n={n} {label}: bucket heads");
            assert_eq!(serial.next, parallel.next, "n={n} {label}: chains");

            // And the thing those arrays exist for.
            for key in [0i64, 10, 30, 12_345, -1] {
                let mut a = Vec::new();
                let mut b = Vec::new();
                serial.collect_matches(key, &mut a);
                parallel.collect_matches(key, &mut b);
                assert_eq!(a, b, "n={n} {label}: matches for {key}");
                assert_eq!(
                    serial.contains(key),
                    parallel.contains(key),
                    "n={n} {label}: contains {key}",
                );
            }
            }
        }
    }

    /// Is building the table in parallel actually faster?
    ///
    /// Measured in one process, alternating, for the reason given on
    /// `bloom_filter_on_the_typed_path`: this machine drifts further between
    /// runs than the effect being looked for.
    ///
    /// Run with
    /// `cargo test --release --lib parallel_build_speed -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn parallel_build_speed() {
        use std::time::Instant;
        const ROWS: usize = 100_000;

        let batch = int_key_batch("k", (0..ROWS as i64).map(|i| i * 10).collect());
        let col = Arc::clone(&batch.columns[0]);

        let serial = || {
            let t = Instant::now();
            let mut table = IntHashTable::with_rows(ROWS, Arc::clone(&col));
            for row in 0..ROWS {
                table.insert(int_key(&col, row), row);
            }
            let e = t.elapsed();
            std::hint::black_box(&table);
            e
        };
        let parallel_with = |parts: usize| {
            // Clone the handle per call, so the outer closure only borrows and
            // can be called more than once.
            let col = Arc::clone(&col);
            move || {
                let t = Instant::now();
                let table = IntHashTable::build_parallel_with(Arc::clone(&col), ROWS, parts);
                let e = t.elapsed();
                std::hint::black_box(&table);
                e
            }
        };
        let parallel = parallel_with(rayon::current_num_threads().max(1));

        for _ in 0..3 {
            serial();
            parallel();
        }
        let mut sv = Vec::new();
        let mut pv = Vec::new();
        for _ in 0..9 {
            sv.push(serial());
            pv.push(parallel());
        }
        let sm = *sv.iter().min().unwrap();
        let pm = *pv.iter().min().unwrap();

        println!("hash table build, {ROWS} rows:");
        println!("  serial   = {sm:?}");
        println!("  parallel = {pm:?}");
        println!("  parallel is {:.2}x serial", pm.as_secs_f64() / sm.as_secs_f64());

        let radix = || {
            let t = Instant::now();
            let table = IntHashTable::build_radix(Arc::clone(&col), ROWS);
            let e = t.elapsed();
            std::hint::black_box(&table);
            e
        };
        for _ in 0..3 {
            radix();
            serial();
        }
        let mut rv = Vec::new();
        for _ in 0..9 {
            rv.push(radix());
        }
        let rm = *rv.iter().min().unwrap();
        println!("  radix    = {rm:?}  ({:.2}x serial)", rm.as_secs_f64() / sm.as_secs_f64());

        // Is the loss the partition count or the approach? Fewer partitions means
        // less rescanning but less parallelism.
        for parts in [2usize, 4, 6, 8, 12] {
            let f = parallel_with(parts);
            for _ in 0..3 { f(); }
            let best = (0..9).map(|_| f()).min().unwrap();
            println!("  {parts:>2} partitions = {best:?}  ({:.2}x serial)",
                     best.as_secs_f64() / sm.as_secs_f64());
        }
    }

    fn orders() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::required("order_id", DataType::Int64),
            Field::required("user_id",  DataType::Int64),
            Field::required("amount",   DataType::Float64),
        ]);
        RecordBatch::new(schema, vec![
            ColumnBuilder::int64(vec![Some(1), Some(2), Some(3)]),
            ColumnBuilder::int64(vec![Some(10), Some(20), Some(10)]),
            ColumnBuilder::float64(vec![Some(100.0), Some(200.0), Some(150.0)]),
        ]).unwrap()
    }

    fn users() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::required("user_id", DataType::Int64),
            Field::required("name",    DataType::Utf8),
        ]);
        RecordBatch::new(schema, vec![
            ColumnBuilder::int64(vec![Some(10), Some(20), Some(30)]),
            ColumnBuilder::utf8_from_str(vec![Some("alice"), Some("bob"), Some("carol")]),
        ]).unwrap()
    }

    #[test]
    fn inner_join() {
        let result = execute_hash_join(
            &[orders()], &[users()],
            &["user_id"], &["user_id"],
            JoinType::Inner,
        ).unwrap();
        // user 10 has 2 orders, user 20 has 1, user 30 has none → 3 rows
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn left_join() {
        // orders left join users — all 3 orders, even if no matching user
        let result = execute_hash_join(
            &[orders()], &[users()],
            &["user_id"], &["user_id"],
            JoinType::Left,
        ).unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn right_join_includes_unmatched() {
        // users right join orders — user 30 (carol) has no orders → NULL row
        let result = execute_hash_join(
            &[orders()], &[users()],
            &["user_id"], &["user_id"],
            JoinType::Right,
        ).unwrap();
        // 3 matched + 1 unmatched (carol) = 4
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn inner_join_correct_values() {
        let result = execute_hash_join(
            &[orders()], &[users()],
            &["user_id"], &["user_id"],
            JoinType::Inner,
        ).unwrap();
        // Both alice rows should appear
        let name_col = result.column_by_name("name").unwrap();
        let names: Vec<ScalarValue> = (0..result.row_count())
            .map(|i| name_col.get(i).unwrap())
            .collect();
        assert!(names.contains(&ScalarValue::Utf8(Some("alice".into()))));
        assert!(names.contains(&ScalarValue::Utf8(Some("bob".into()))));
    }
}
