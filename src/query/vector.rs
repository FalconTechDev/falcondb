// Vectorized column operations.
//
// Instead of evaluating predicates row-by-row and producing Vec<bool>,
// we operate on chunks of 64 rows using u64 bitmasks.
// Each bit in the mask corresponds to one row; bit=1 means "keep".
//
// This allows the compiler to auto-vectorize inner loops and reduces
// branch pressure by 64×.

use crate::storage::{
    column::Column,
    table::RecordBatch,
    types::FalconResult,
};

// ─── Selection vector ─────────────────────────────────────────────────────────

/// Compact representation of which rows are "alive" after filtering.
///
/// The selection vector stores a bitmask: one `u64` per 64 rows.
/// Bit `i % 64` of word `i / 64` is set when row `i` passes the filter.
#[derive(Debug, Clone)]
pub struct SelectionVector {
    words:    Vec<u64>,
    row_count: usize,
}

impl SelectionVector {
    /// All rows selected.
    pub fn all(row_count: usize) -> Self {
        let n_words = words_for(row_count);
        let mut words = vec![u64::MAX; n_words];
        // Zero out the tail bits beyond row_count.
        if row_count % 64 != 0 {
            let tail = row_count % 64;
            words[n_words - 1] = (1u64 << tail) - 1;
        }
        Self { words, row_count }
    }

    /// No rows selected.
    pub fn none(row_count: usize) -> Self {
        Self { words: vec![0u64; words_for(row_count)], row_count }
    }

    /// Build from a plain `Vec<bool>`.
    pub fn from_bools(mask: &[bool]) -> Self {
        let mut words = vec![0u64; words_for(mask.len())];
        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        Self { words, row_count: mask.len() }
    }

    /// Convert back to a `Vec<bool>` (for interop with existing code).
    pub fn to_bools(&self) -> Vec<bool> {
        (0..self.row_count)
            .map(|i| (self.words[i / 64] >> (i % 64)) & 1 == 1)
            .collect()
    }

    /// Number of selected (alive) rows.
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// AND two selection vectors (row passes both filters).
    pub fn and(&self, other: &SelectionVector) -> SelectionVector {
        debug_assert_eq!(self.row_count, other.row_count);
        let words = self.words.iter().zip(&other.words).map(|(a, b)| a & b).collect();
        SelectionVector { words, row_count: self.row_count }
    }

    /// OR two selection vectors (row passes at least one filter).
    pub fn or(&self, other: &SelectionVector) -> SelectionVector {
        debug_assert_eq!(self.row_count, other.row_count);
        let words = self.words.iter().zip(&other.words).map(|(a, b)| a | b).collect();
        SelectionVector { words, row_count: self.row_count }
    }

    /// NOT — invert selection.
    pub fn not(&self) -> SelectionVector {
        let n_words = self.words.len();
        let mut words: Vec<u64> = self.words.iter().map(|w| !w).collect();
        // Fix tail bits.
        if self.row_count % 64 != 0 && n_words > 0 {
            let tail = self.row_count % 64;
            words[n_words - 1] &= (1u64 << tail) - 1;
        }
        SelectionVector { words, row_count: self.row_count }
    }

    /// Iterate over the indices of selected rows.
    pub fn selected_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(word_idx, &w)| {
            BitIterator(w, word_idx * 64)
        })
    }
}

struct BitIterator(u64, usize);

impl Iterator for BitIterator {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 { return None; }
        let bit = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1; // clear lowest set bit
        Some(self.1 + bit)
    }
}

fn words_for(n: usize) -> usize {
    n.div_ceil(64)
}

// ─── Vectorized comparison kernels ────────────────────────────────────────────
//
// These functions operate directly on typed slices, enabling auto-vectorization.
// Each returns a SelectionVector representing rows that satisfy the predicate.

// ─── SIMD-accelerated kernels (AVX2, x86_64 only) ────────────────────────────
//
// These kernels operate on the validity bitmap and value arrays separately,
// mirroring the PrimitiveColumn layout.  We process 4 i64 / f64 values per
// AVX2 register and accumulate results into 64-bit bitmask words.
//
// On non-x86 platforms (or when AVX2 is absent at runtime) we fall through
// to the scalar implementations below.

/// Build a bitmask for `col > threshold` using AVX2 when available.
#[cfg(target_arch = "x86_64")]
pub fn vec_gt_i64_simd(values: &[i64], validity: &[u64], threshold: i64) -> Vec<u64> {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: we just checked for AVX2 at runtime.
        unsafe { vec_gt_i64_avx2(values, validity, threshold) }
    } else {
        vec_gt_i64_scalar(values, validity, threshold)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_gt_i64_avx2(values: &[i64], validity: &[u64], threshold: i64) -> Vec<u64> {
    use std::arch::x86_64::*;

    let n         = values.len();
    let n_words   = n.div_ceil(64);
    let mut out   = vec![0u64; n_words];
    let thresh    = _mm256_set1_epi64x(threshold);

    let mut i = 0usize;
    while i + 4 <= n {
        let word_idx = i / 64;
        let bit_off  = i % 64;

        // Load 4 i64 values.
        let v = _mm256_loadu_si256(values.as_ptr().add(i) as *const __m256i);

        // AVX2 has no direct gt_epi64; emulate with _mm256_cmpgt_epi64.
        let cmp = _mm256_cmpgt_epi64(v, thresh);
        let mask4 = _mm256_movemask_epi8(cmp) as u32;

        // movemask operates on bytes (32 bytes in a 256-bit reg → 32 bits).
        // Each i64 spans 8 bytes: bits 0-7 for lane 0, 8-15 for lane 1, etc.
        // Extract one bit per lane (the MSB of each lane is the sign bit of
        // the comparison result, which is all-1s when the comparison is true).
        let lane0 = ((mask4 >> 0)  & 1) as u64;
        let lane1 = ((mask4 >> 8)  & 1) as u64;
        let lane2 = ((mask4 >> 16) & 1) as u64;
        let lane3 = ((mask4 >> 24) & 1) as u64;

        // Validity: load the relevant 4 bits from the validity bitmask.
        let v0 = (validity[word_idx] >> bit_off) & 1;
        let v1 = (validity[word_idx] >> (bit_off + 1)) & 1;
        let v2 = (validity[word_idx] >> (bit_off + 2)) & 1;
        let v3 = (validity[word_idx] >> (bit_off + 3)) & 1;

        out[word_idx] |= ((lane0 & v0) << bit_off)
            | ((lane1 & v1) << (bit_off + 1))
            | ((lane2 & v2) << (bit_off + 2))
            | ((lane3 & v3) << (bit_off + 3));

        i += 4;
    }

    // Scalar tail.
    while i < n {
        let word_idx = i / 64;
        let bit_off  = i % 64;
        let valid    = (validity[word_idx] >> bit_off) & 1;
        if valid == 1 && values[i] > threshold {
            out[word_idx] |= 1u64 << bit_off;
        }
        i += 1;
    }

    out
}

#[cfg(target_arch = "x86_64")]
fn vec_gt_i64_scalar(values: &[i64], validity: &[u64], threshold: i64) -> Vec<u64> {
    let n_words = values.len().div_ceil(64);
    let mut out = vec![0u64; n_words];
    for (i, &v) in values.iter().enumerate() {
        let valid = (validity[i / 64] >> (i % 64)) & 1;
        if valid == 1 && v > threshold {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

pub fn vec_gt_f64(col: &[Option<f64>], threshold: f64) -> SelectionVector {
    let mut words = vec![0u64; words_for(col.len())];
    for (i, v) in col.iter().enumerate() {
        if v.map_or(false, |x| x > threshold) {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    SelectionVector { words, row_count: col.len() }
}

pub fn vec_lt_f64(col: &[Option<f64>], threshold: f64) -> SelectionVector {
    let mut words = vec![0u64; words_for(col.len())];
    for (i, v) in col.iter().enumerate() {
        if v.map_or(false, |x| x < threshold) {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    SelectionVector { words, row_count: col.len() }
}

pub fn vec_eq_f64(col: &[Option<f64>], value: f64) -> SelectionVector {
    let mut words = vec![0u64; words_for(col.len())];
    for (i, v) in col.iter().enumerate() {
        if v.map_or(false, |x| (x - value).abs() < f64::EPSILON) {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    SelectionVector { words, row_count: col.len() }
}

pub fn vec_gt_i64(col: &[Option<i64>], threshold: i64) -> SelectionVector {
    let mut words = vec![0u64; words_for(col.len())];
    for (i, v) in col.iter().enumerate() {
        if v.map_or(false, |x| x > threshold) {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    SelectionVector { words, row_count: col.len() }
}

pub fn vec_lt_i64(col: &[Option<i64>], threshold: i64) -> SelectionVector {
    let mut words = vec![0u64; words_for(col.len())];
    for (i, v) in col.iter().enumerate() {
        if v.map_or(false, |x| x < threshold) {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    SelectionVector { words, row_count: col.len() }
}

pub fn vec_eq_i64(col: &[Option<i64>], value: i64) -> SelectionVector {
    let mut words = vec![0u64; words_for(col.len())];
    for (i, v) in col.iter().enumerate() {
        if v.map_or(false, |x| x == value) {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    SelectionVector { words, row_count: col.len() }
}

pub fn vec_eq_str<'a>(col: &[Option<String>], value: &str) -> SelectionVector {
    let mut words = vec![0u64; words_for(col.len())];
    for (i, v) in col.iter().enumerate() {
        if v.as_deref().map_or(false, |s| s == value) {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    SelectionVector { words, row_count: col.len() }
}

pub fn vec_is_null(col: &Column) -> SelectionVector {
    let n = col.len();
    let mut words = vec![0u64; words_for(n)];
    for i in 0..n {
        if col.get(i).map(|v| v.is_null()).unwrap_or(false) {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    SelectionVector { words, row_count: n }
}

// ─── Vectorized column arithmetic ─────────────────────────────────────────────

/// Add a scalar to every non-null element of an f64 column.
pub fn vec_add_scalar_f64(col: &[Option<f64>], scalar: f64) -> Vec<Option<f64>> {
    col.iter().map(|v| v.map(|x| x + scalar)).collect()
}

/// Multiply every non-null element by a scalar.
pub fn vec_mul_scalar_f64(col: &[Option<f64>], scalar: f64) -> Vec<Option<f64>> {
    col.iter().map(|v| v.map(|x| x * scalar)).collect()
}

/// Element-wise sum of two columns.
pub fn vec_add_f64(a: &[Option<f64>], b: &[Option<f64>]) -> Vec<Option<f64>> {
    a.iter().zip(b).map(|(x, y)| match (x, y) {
        (Some(a), Some(b)) => Some(a + b),
        _ => None,
    }).collect()
}

// ─── Apply selection vector to a RecordBatch ──────────────────────────────────

/// Filter a batch using a SelectionVector without materializing a bool Vec.
pub fn apply_selection(batch: &RecordBatch, sel: &SelectionVector) -> FalconResult<RecordBatch> {
    let indices: Vec<usize> = sel.selected_indices().collect();
    let columns: FalconResult<Vec<Column>> = batch.columns.iter()
        .map(|col| col.filter_by_indices(&indices))
        .collect();
    RecordBatch::new(batch.schema.clone(), columns?)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn f64_col(vals: Vec<Option<f64>>) -> Vec<Option<f64>> { vals }
    fn i64_col(vals: Vec<Option<i64>>) -> Vec<Option<i64>> { vals }

    #[test]
    fn selection_vector_all() {
        let sv = SelectionVector::all(5);
        assert_eq!(sv.count(), 5);
        let bools = sv.to_bools();
        assert!(bools.iter().all(|&b| b));
    }

    #[test]
    fn selection_vector_none() {
        let sv = SelectionVector::none(5);
        assert_eq!(sv.count(), 0);
    }

    #[test]
    fn from_bools_roundtrip() {
        let mask = vec![true, false, true, true, false];
        let sv   = SelectionVector::from_bools(&mask);
        assert_eq!(sv.to_bools(), mask);
        assert_eq!(sv.count(), 3);
    }

    #[test]
    fn bitmask_and() {
        let a = SelectionVector::from_bools(&[true, true,  false, false]);
        let b = SelectionVector::from_bools(&[true, false, true,  false]);
        let r = a.and(&b);
        assert_eq!(r.to_bools(), vec![true, false, false, false]);
    }

    #[test]
    fn bitmask_or() {
        let a = SelectionVector::from_bools(&[true, true,  false, false]);
        let b = SelectionVector::from_bools(&[true, false, true,  false]);
        let r = a.or(&b);
        assert_eq!(r.to_bools(), vec![true, true, true, false]);
    }

    #[test]
    fn bitmask_not() {
        let sv = SelectionVector::from_bools(&[true, false, true]);
        let r  = sv.not();
        assert_eq!(r.to_bools(), vec![false, true, false]);
    }

    #[test]
    fn bit_iterator_correct() {
        let sv = SelectionVector::from_bools(&[false, true, false, true, true]);
        let indices: Vec<usize> = sv.selected_indices().collect();
        assert_eq!(indices, vec![1, 3, 4]);
    }

    #[test]
    fn vec_gt_f64_basic() {
        let col = f64_col(vec![Some(1.0), Some(5.0), None, Some(3.0)]);
        let sv  = vec_gt_f64(&col, 2.0);
        assert_eq!(sv.to_bools(), vec![false, true, false, true]);
    }

    #[test]
    fn vec_eq_i64_basic() {
        let col = i64_col(vec![Some(10), Some(20), None, Some(10)]);
        let sv  = vec_eq_i64(&col, 10);
        assert_eq!(sv.to_bools(), vec![true, false, false, true]);
    }

    #[test]
    fn selection_vector_large() {
        // Test crossing the 64-bit boundary.
        let n    = 130;
        let mask: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let sv   = SelectionVector::from_bools(&mask);
        assert_eq!(sv.to_bools(), mask);
        assert_eq!(sv.count(), mask.iter().filter(|&&b| b).count());
    }

    #[test]
    fn selected_indices_large() {
        let n    = 200;
        let mask: Vec<bool> = (0..n).map(|i| i % 7 == 0).collect();
        let sv   = SelectionVector::from_bools(&mask);
        let got: Vec<usize>      = sv.selected_indices().collect();
        let expected: Vec<usize> = (0..n).filter(|i| i % 7 == 0).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn vec_arithmetic_add_scalar() {
        let col = f64_col(vec![Some(1.0), None, Some(3.0)]);
        let res = vec_add_scalar_f64(&col, 10.0);
        assert_eq!(res, vec![Some(11.0), None, Some(13.0)]);
    }

    #[test]
    fn apply_selection_to_batch() {
        use crate::storage::{
            column::ColumnBuilder,
            schema::{Field, Schema},
            table::RecordBatch,
            types::DataType,
        };
        let schema = Schema::new(vec![
            Field::required("x", DataType::Int64),
        ]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::int64(vec![Some(1), Some(2), Some(3), Some(4)]),
        ]).unwrap();

        let sv = SelectionVector::from_bools(&[true, false, true, false]);
        let filtered = apply_selection(&batch, &sv).unwrap();
        assert_eq!(filtered.row_count(), 2);
    }
}
