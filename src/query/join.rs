use super::hash::{FastMap, FastSet};

use crate::storage::{
    column::{BooleanColumn, Column, PrimitiveColumn, Utf8Column},
    schema::{Field, Schema},
    table::RecordBatch,
    types::{DataType, FalconError, FalconResult, ScalarValue},
};

// ─── Join type ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
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
    let (hash_table, bloom) = build_hash_table(&right_merged, right_keys)?;

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

    let mut keep = Vec::with_capacity(left_merged.row_count());
    for l_row in 0..left_merged.row_count() {
        // A definite miss needs no string key: semi drops the row, anti keeps it.
        if let Some(f) = bloom {
            if !f.might_contain(row_key_hash(&left_key_cols, l_row)) {
                keep.push(anti);
                continue;
            }
        }
        let key     = row_key(&left_key_cols, l_row)?;
        let matched = hash_table.contains_key(&key);
        keep.push(if anti { !matched } else { matched });
    }

    left_merged.filter_by_mask(&keep)
}

/// Execute a hash join between two sets of batches on the given key columns.
///
/// Algorithm:
///   1. Build phase — hash the *right* side into a HashMap<key → Vec<row_idx>>.
///   2. Probe phase — iterate *left* rows, look up matches in the hash table.
pub fn execute_hash_join(
    left_batches:  &[RecordBatch],
    right_batches: &[RecordBatch],
    left_keys:     &[&str],
    right_keys:    &[&str],
    join_type:     JoinType,
) -> FalconResult<RecordBatch> {
    if left_batches.is_empty() || right_batches.is_empty() {
        return Err(FalconError::InvalidQuery("join requires non-empty inputs".into()));
    }
    if left_keys.len() != right_keys.len() || left_keys.is_empty() {
        return Err(FalconError::InvalidQuery("join key lists must have the same length".into()));
    }

    let right_merged = merge_batches(right_batches)?;
    let left_merged  = merge_batches(left_batches)?;

    let (hash_table, bloom) = build_hash_table(&right_merged, right_keys)?;
    probe(
        &left_merged,
        &right_merged,
        left_keys,
        right_keys,
        &hash_table,
        bloom.as_ref(),
        join_type,
    )
}

// ─── Build phase ─────────────────────────────────────────────────────────────

/// Smallest build side worth filtering. Below this the filter costs more to
/// construct than the probe-side lookups it saves.
const BLOOM_MIN_ROWS: usize = 1_024;

fn build_hash_table(
    batch: &RecordBatch,
    keys:  &[&str],
) -> FalconResult<(FastMap<String, Vec<usize>>, Option<BuildFilter>)> {
    let key_cols: Vec<&Column> = keys.iter()
        .map(|k| batch.column_by_name(k))
        .collect::<FalconResult<_>>()?;

    let mut table: FastMap<String, Vec<usize>> = FastMap::default();
    let n = batch.row_count();
    let mut bloom = (n >= BLOOM_MIN_ROWS).then(|| BuildFilter::with_capacity(n));

    for row in 0..n {
        let key = row_key(&key_cols, row)?;
        table.entry(key).or_default().push(row);
        if let Some(f) = bloom.as_mut() {
            f.insert(row_key_hash(&key_cols, row));
        }
    }

    Ok((table, bloom))
}

// ─── Probe phase ─────────────────────────────────────────────────────────────

fn probe(
    left:       &RecordBatch,
    right:      &RecordBatch,
    left_keys:  &[&str],
    right_keys: &[&str],
    hash_table: &FastMap<String, Vec<usize>>,
    bloom:      Option<&BuildFilter>,
    join_type:  JoinType,
) -> FalconResult<RecordBatch> {
    let left_key_cols: Vec<&Column> = left_keys.iter()
        .map(|k| left.column_by_name(k))
        .collect::<FalconResult<_>>()?;
    let right_key_cols: Vec<&Column> = right_keys.iter()
        .map(|k| right.column_by_name(k))
        .collect::<FalconResult<_>>()?;

    let mut left_indices:  Vec<Option<usize>> = Vec::new();
    let mut right_indices: Vec<Option<usize>> = Vec::new();
    let mut right_matched: Vec<bool>          = vec![false; right.row_count()];

    // Only use the filter when the two sides agree on key types; see
    // `row_key_hash` for why a mismatch could drop matches.
    let bloom = bloom.filter(|_| filter_is_safe(&right_key_cols, &left_key_cols));

    for l_row in 0..left.row_count() {
        // Definitely-absent rows skip the `row_key` allocation entirely.
        if let Some(f) = bloom {
            if !f.might_contain(row_key_hash(&left_key_cols, l_row)) {
                if join_type == JoinType::Left {
                    left_indices.push(Some(l_row));
                    right_indices.push(None);
                }
                continue;
            }
        }

        let key     = row_key(&left_key_cols, l_row)?;
        let matches = hash_table.get(&key);

        match matches {
            Some(r_rows) => {
                for &r_row in r_rows {
                    left_indices.push(Some(l_row));
                    right_indices.push(Some(r_row));
                    right_matched[r_row] = true;
                }
            }
            None => {
                if join_type == JoinType::Left {
                    left_indices.push(Some(l_row));
                    right_indices.push(None);
                }
            }
        }
    }

    // RIGHT JOIN: add unmatched right rows.
    if join_type == JoinType::Right {
        for (r_row, &matched) in right_matched.iter().enumerate() {
            if !matched {
                left_indices.push(None);
                right_indices.push(Some(r_row));
            }
        }
    }

    build_output(left, right, &left_indices, &right_indices)
}

// ─── Output builder ───────────────────────────────────────────────────────────

fn build_output(
    left:          &RecordBatch,
    right:         &RecordBatch,
    left_indices:  &[Option<usize>],
    right_indices: &[Option<usize>],
) -> FalconResult<RecordBatch> {
    let n_rows = left_indices.len();
    let mut fields  = Vec::new();
    let mut columns = Vec::new();

    // All left columns.
    for (i, field) in left.schema.fields.iter().enumerate() {
        let col    = left.column(i);
        let vals   = gather_nullable(col, left_indices, n_rows)?;
        let (_, c) = scalar_vec_to_col(&vals, &field.data_type);
        fields.push(Field::nullable(field.name.as_str(), field.data_type.clone()));
        columns.push(c);
    }

    // Right columns — skip key columns to avoid duplicates.
    let left_names: FastSet<&str> =
        left.schema.column_names().into_iter().collect();

    for (i, field) in right.schema.fields.iter().enumerate() {
        let name = if left_names.contains(field.name.as_str()) {
            format!("right.{}", field.name)
        } else {
            field.name.clone()
        };

        let col    = right.column(i);
        let vals   = gather_nullable(col, right_indices, n_rows)?;
        let (_, c) = scalar_vec_to_col(&vals, &field.data_type);
        fields.push(Field::nullable(name.as_str(), field.data_type.clone()));
        columns.push(c);
    }

    RecordBatch::new(Schema::new(fields), columns)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn row_key(key_cols: &[&Column], row: usize) -> FalconResult<String> {
    let parts: FalconResult<Vec<String>> = key_cols.iter()
        .map(|col| col.get(row).map(|v| v.to_string()))
        .collect();
    Ok(parts?.join("\x00"))
}

// ─── Bloom filter over the build side ────────────────────────────────────────

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

fn gather_nullable(
    col:     &Column,
    indices: &[Option<usize>],
    n:       usize,
) -> FalconResult<Vec<ScalarValue>> {
    let mut vals = Vec::with_capacity(n);
    for &idx in indices {
        vals.push(match idx {
            Some(i) => col.get(i)?,
            None    => ScalarValue::Null,
        });
    }
    Ok(vals)
}

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

fn merge_batches(batches: &[RecordBatch]) -> FalconResult<RecordBatch> {
    if batches.len() == 1 {
        return Ok(batches[0].clone());
    }

    let schema    = batches[0].schema.clone();
    let col_count = schema.field_count();
    let total: usize = batches.iter().map(|b| b.row_count()).sum();

    let mut merged: Vec<Vec<ScalarValue>> = vec![Vec::with_capacity(total); col_count];
    for batch in batches {
        for ci in 0..col_count {
            let col = batch.column(ci);
            for row in 0..batch.row_count() {
                merged[ci].push(col.get(row)?);
            }
        }
    }

    let columns: Vec<Column> = merged.iter().enumerate()
        .map(|(i, vals)| scalar_vec_to_col(vals, &schema.fields[i].data_type).1)
        .collect();

    RecordBatch::new(schema, columns)
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
