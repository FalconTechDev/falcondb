use std::sync::Arc;

use super::types::{DataType, FalconError, FalconResult, ScalarValue};

/// A column is a typed, contiguous array of values — the core unit of falconDB.
/// Each variant owns its data as a Vec, which gives us cache-friendly sequential access.
#[derive(Debug, Clone)]
pub enum Column {
    Boolean(BooleanColumn),
    Int32(PrimitiveColumn<i32>),
    Int64(PrimitiveColumn<i64>),
    Float32(PrimitiveColumn<f32>),
    Float64(PrimitiveColumn<f64>),
    Utf8(Utf8Column),
    /// Days since 1970-01-01.
    Date32(PrimitiveColumn<i32>),
    /// Microseconds since epoch.
    Timestamp(PrimitiveColumn<i64>),
    /// Fixed-point: value = integer / 10^scale.
    Decimal128(Decimal128Column),
}

impl Column {
    pub fn data_type(&self) -> DataType {
        match self {
            Column::Boolean(_)     => DataType::Boolean,
            Column::Int32(_)       => DataType::Int32,
            Column::Int64(_)       => DataType::Int64,
            Column::Float32(_)     => DataType::Float32,
            Column::Float64(_)     => DataType::Float64,
            Column::Utf8(_)        => DataType::Utf8,
            Column::Date32(_)      => DataType::Date32,
            Column::Timestamp(_)   => DataType::Timestamp,
            Column::Decimal128(c)  => DataType::Decimal128(c.precision, c.scale),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Column::Boolean(c)    => c.len(),
            Column::Int32(c)      => c.len(),
            Column::Int64(c)      => c.len(),
            Column::Float32(c)    => c.len(),
            Column::Float64(c)    => c.len(),
            Column::Utf8(c)       => c.len(),
            Column::Date32(c)     => c.len(),
            Column::Timestamp(c)  => c.len(),
            Column::Decimal128(c) => c.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> FalconResult<ScalarValue> {
        if index >= self.len() {
            return Err(FalconError::IndexOutOfBounds { index, len: self.len() });
        }
        Ok(match self {
            Column::Boolean(c)    => ScalarValue::Boolean(c.get(index)),
            Column::Int32(c)      => ScalarValue::Int32(c.get(index)),
            Column::Int64(c)      => ScalarValue::Int64(c.get(index)),
            Column::Float32(c)    => ScalarValue::Float32(c.get(index)),
            Column::Float64(c)    => ScalarValue::Float64(c.get(index)),
            Column::Utf8(c)       => ScalarValue::Utf8(c.get(index)),
            Column::Date32(c)     => ScalarValue::Date32(c.get(index)),
            Column::Timestamp(c)  => ScalarValue::Timestamp(c.get(index)),
            Column::Decimal128(c) => ScalarValue::decimal(c.inner.get(index), c.precision, c.scale),
        })
    }

    /// Returns a new column containing only the rows at the given indices.
    pub fn filter_by_indices(&self, indices: &[usize]) -> FalconResult<Column> {
        Ok(match self {
            Column::Boolean(c)    => Column::Boolean(c.filter(indices)?),
            Column::Int32(c)      => Column::Int32(c.filter(indices)?),
            Column::Int64(c)      => Column::Int64(c.filter(indices)?),
            Column::Float32(c)    => Column::Float32(c.filter(indices)?),
            Column::Float64(c)    => Column::Float64(c.filter(indices)?),
            Column::Utf8(c)       => Column::Utf8(c.filter(indices)?),
            Column::Date32(c)     => Column::Date32(c.filter(indices)?),
            Column::Timestamp(c)  => Column::Timestamp(c.filter(indices)?),
            Column::Decimal128(c) => Column::Decimal128(Decimal128Column {
                inner:     c.inner.filter(indices)?,
                precision: c.precision,
                scale:     c.scale,
            }),
        })
    }

    /// Returns a new column containing only the rows where the boolean mask is true.
    pub fn filter_by_mask(&self, mask: &[bool]) -> FalconResult<Column> {
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();
        self.filter_by_indices(&indices)
    }

    /// Packed-bit variant: `mask` is `ceil(row_count/64)` words, LSB-first.
    /// Faster than `filter_by_mask` because the mask occupies 8× less cache.
    ///
    /// When filtering a whole batch prefer `gather_by_bitmask` (no index list alloc)
    /// or at minimum build indices once and share across columns.
    pub fn filter_by_bitmask(&self, mask: &[u64], row_count: usize) -> FalconResult<Column> {
        let indices = bitmask_to_indices(mask, row_count);
        self.filter_by_indices(&indices)
    }

    /// Gather rows from a packed bitmask without materialising an intermediate index list.
    ///
    /// `selected` is the popcount of `mask` — pass it in so the caller can compute it
    /// once and share across all columns in a batch.
    pub fn gather_by_bitmask(&self, mask: &[u64], row_count: usize, selected: usize) -> FalconResult<Column> {
        Ok(match self {
            Column::Int32(c)   => Column::Int32(c.gather_by_bitmask(mask, row_count, selected)),
            Column::Int64(c)   => Column::Int64(c.gather_by_bitmask(mask, row_count, selected)),
            Column::Float32(c) => Column::Float32(c.gather_by_bitmask(mask, row_count, selected)),
            Column::Float64(c) => Column::Float64(c.gather_by_bitmask(mask, row_count, selected)),
            Column::Date32(c)  => Column::Date32(c.gather_by_bitmask(mask, row_count, selected)),
            Column::Timestamp(c) => Column::Timestamp(c.gather_by_bitmask(mask, row_count, selected)),
            Column::Utf8(c)    => Column::Utf8(c.gather_by_bitmask(mask, row_count, selected)),
            Column::Decimal128(c) => Column::Decimal128(Decimal128Column {
                inner:     c.inner.gather_by_bitmask(mask, row_count, selected),
                precision: c.precision,
                scale:     c.scale,
            }),
            // BooleanColumn: fall back to index-based path (rare in analytical queries)
            Column::Boolean(_) => self.filter_by_bitmask(mask, row_count)?,
        })
    }

    pub fn slice(&self, offset: usize, length: usize) -> FalconResult<Column> {
        Ok(match self {
            Column::Boolean(c)    => Column::Boolean(c.slice(offset, length)?),
            Column::Int32(c)      => Column::Int32(c.slice(offset, length)?),
            Column::Int64(c)      => Column::Int64(c.slice(offset, length)?),
            Column::Float32(c)    => Column::Float32(c.slice(offset, length)?),
            Column::Float64(c)    => Column::Float64(c.slice(offset, length)?),
            Column::Utf8(c)       => Column::Utf8(c.slice(offset, length)?),
            Column::Date32(c)     => Column::Date32(c.slice(offset, length)?),
            Column::Timestamp(c)  => Column::Timestamp(c.slice(offset, length)?),
            Column::Decimal128(c) => Column::Decimal128(Decimal128Column {
                inner:     c.inner.slice(offset, length)?,
                precision: c.precision,
                scale:     c.scale,
            }),
        })
    }

    pub fn null_count(&self) -> usize {
        match self {
            Column::Boolean(c)    => c.null_count(),
            Column::Int32(c)      => c.null_count(),
            Column::Int64(c)      => c.null_count(),
            Column::Float32(c)    => c.null_count(),
            Column::Float64(c)    => c.null_count(),
            Column::Utf8(c)       => c.null_count(),
            Column::Date32(c)     => c.null_count(),
            Column::Timestamp(c)  => c.null_count(),
            Column::Decimal128(c) => c.inner.null_count(),
        }
    }

    /// Downcast to Int64, promoting Int32 if needed.
    pub fn as_int64_values(&self) -> Option<Vec<Option<i64>>> {
        match self {
            Column::Int32(c) => Some(c.iter().map(|v| v.map(|n| n as i64)).collect()),
            Column::Int64(c) => Some(c.to_options()),
            _ => None,
        }
    }

    /// Downcast to Float64, promoting integer types if needed.
    pub fn as_float64_values(&self) -> Option<Vec<Option<f64>>> {
        match self {
            Column::Int32(c)      => Some(c.iter().map(|v| v.map(|n| n as f64)).collect()),
            Column::Int64(c)      => Some(c.iter().map(|v| v.map(|n| n as f64)).collect()),
            Column::Float32(c)    => Some(c.iter().map(|v| v.map(|n| n as f64)).collect()),
            Column::Float64(c)    => Some(c.to_options()),
            Column::Decimal128(c) => {
                let s = c.scale;
                Some(c.inner.iter().map(|v| v.map(|n| n as f64 / 10f64.powi(s as i32))).collect())
            }
            _ => None,
        }
    }

    /// Borrow the dictionary-encoded string column, so callers can work on codes
    /// instead of materialising text.
    pub fn as_utf8_column(&self) -> Option<&Utf8Column> {
        match self {
            Column::Utf8(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_bool_values(&self) -> Option<&[Option<bool>]> {
        match self {
            Column::Boolean(c) => Some(&c.values),
            _ => None,
        }
    }
}

// ─── Bitmask expansion ───────────────────────────────────────────────────────

/// Expand a packed bitmask (LSB-first, one bit per row) into the list of set row indices.
///
/// Uses a bit-scan rather than testing every row: `trailing_zeros` jumps straight to
/// the next set bit and Kernighan's `bits &= bits - 1` clears it. A selective predicate
/// that keeps 10 % of rows therefore does ~10 % of the work instead of a full scan.
/// The output is pre-sized with popcount so the gather never reallocates.
pub fn bitmask_to_indices(mask: &[u64], row_count: usize) -> Vec<usize> {
    let selected: usize = mask.iter().map(|w| w.count_ones() as usize).sum();
    let mut indices = Vec::with_capacity(selected);

    for (word_idx, &word) in mask.iter().enumerate() {
        let base = word_idx << 6;
        if base >= row_count {
            break;
        }
        let mut bits = word;
        while bits != 0 {
            let row = base + bits.trailing_zeros() as usize;
            if row >= row_count {
                break;
            }
            indices.push(row);
            bits &= bits - 1; // clear the lowest set bit
        }
    }

    indices
}

// ─── Concatenation ───────────────────────────────────────────────────────────

impl Column {
    /// Concatenate columns into one without round-tripping values through `ScalarValue`.
    ///
    /// When every input shares a variant this is a typed bulk copy. Mixed numeric
    /// variants are promoted to `Float64`; anything else is an error.
    pub fn concat(columns: &[&Column]) -> FalconResult<Column> {
        if columns.is_empty() {
            return Ok(Column::Int64(PrimitiveColumn::new(Vec::new())));
        }

        let first: &Column = columns[0];
        let total: usize   = columns.iter().map(|c| c.len()).sum();

        let uniform = columns
            .iter()
            .all(|c| std::mem::discriminant(*c) == std::mem::discriminant(first));

        if uniform {
            // Boolean still holds `Vec<Option<bool>>`, so a slice copy works.
            macro_rules! concat_boxed {
                ($variant:ident, $wrapper:ident) => {{
                    let mut values = Vec::with_capacity(total);
                    for &col in columns {
                        if let Column::$variant(c) = col {
                            values.extend_from_slice(&c.values);
                        }
                    }
                    Column::$variant($wrapper::new(values))
                }};
            }

            // Numeric columns are split into values + validity. When no input has a
            // NULL the whole concat is a memcpy of dense slices; otherwise fall back
            // to rebuilding through `Option`.
            macro_rules! concat_primitive {
                ($variant:ident) => {{
                    let parts: Vec<_> = columns
                        .iter()
                        .filter_map(|c| match c {
                            Column::$variant(p) => Some(p),
                            _ => None,
                        })
                        .collect();
                    Column::$variant(PrimitiveColumn::concat_all(&parts))
                }};
            }

            return Ok(match first {
                Column::Boolean(_) => concat_boxed!(Boolean, BooleanColumn),
                Column::Utf8(_) => {
                    let parts: Vec<_> = columns
                        .iter()
                        .filter_map(|c| match c {
                            Column::Utf8(u) => Some(u),
                            _ => None,
                        })
                        .collect();
                    Column::Utf8(Utf8Column::concat_all(&parts))
                }
                Column::Int32(_)   => concat_primitive!(Int32),
                Column::Int64(_)   => concat_primitive!(Int64),
                Column::Float32(_) => concat_primitive!(Float32),
                Column::Float64(_) => concat_primitive!(Float64),
                Column::Date32(_)  => concat_primitive!(Date32),
                Column::Timestamp(_) => concat_primitive!(Timestamp),
                Column::Decimal128(first_d) => {
                    let (p, s) = (first_d.precision, first_d.scale);
                    let parts: Vec<_> = columns
                        .iter()
                        .filter_map(|c| match c {
                            Column::Decimal128(d) => Some(&d.inner),
                            _ => None,
                        })
                        .collect();
                    Column::Decimal128(Decimal128Column {
                        inner: PrimitiveColumn::concat_all(&parts),
                        precision: p,
                        scale: s,
                    })
                }
            });
        }

        // Mixed variants: a numeric mix is the only meaningful case — promote to Float64.
        if columns.iter().all(|c| c.as_float64_values().is_some()) {
            let mut values = Vec::with_capacity(total);
            for &col in columns {
                values.extend(col.as_float64_values().expect("checked above"));
            }
            return Ok(Column::Float64(PrimitiveColumn::new(values)));
        }

        Err(FalconError::InvalidQuery(
            "cannot concatenate columns of incompatible types".into(),
        ))
    }
}

#[cfg(test)]
mod dense_layout_tests {
    use super::*;

    #[test]
    fn roundtrips_values_and_nulls() {
        let c = PrimitiveColumn::new(vec![Some(1i64), None, Some(3), None]);
        assert_eq!(c.len(), 4);
        assert_eq!(c.get(0), Some(1));
        assert_eq!(c.get(1), None);
        assert_eq!(c.get(2), Some(3));
        assert_eq!(c.get(3), None);
        assert_eq!(c.null_count(), 2);
        assert_eq!(c.to_options(), vec![Some(1), None, Some(3), None]);
    }

    #[test]
    fn null_free_column_skips_the_bitmap() {
        let c = PrimitiveColumn::new(vec![Some(1i64), Some(2)]);
        assert!(!c.has_nulls(), "no bitmap should be allocated when nothing is null");
        assert_eq!(c.dense(), &[1, 2]);
    }

    #[test]
    fn dense_slice_is_contiguous_and_full_length() {
        // The placeholder at a NULL slot keeps positions aligned, which is what
        // lets `dense()` be indexed by row number.
        let c = PrimitiveColumn::new(vec![Some(10i64), None, Some(30)]);
        assert_eq!(c.dense().len(), 3);
        assert_eq!(c.dense()[0], 10);
        assert_eq!(c.dense()[2], 30);
        assert!(c.has_nulls());
    }

    #[test]
    fn placeholder_never_leaks_through_get() {
        let c = PrimitiveColumn::new(vec![None, Some(5i64)]);
        // Slot 0 holds a default in `dense()`, but the public API must report NULL.
        assert_eq!(c.dense()[0], 0);
        assert_eq!(c.get(0), None);
        assert_eq!(c.get(1), Some(5));
    }

    #[test]
    fn is_valid_matches_get() {
        let c = PrimitiveColumn::new(vec![Some(1i64), None, Some(3)]);
        for i in 0..c.len() {
            assert_eq!(c.is_valid(i), c.get(i).is_some(), "row {i}");
        }
    }

    #[test]
    fn filter_preserves_nulls() {
        let c = PrimitiveColumn::new(vec![Some(1i64), None, Some(3), Some(4)]);
        let f = c.filter(&[1, 2]).unwrap();
        assert_eq!(f.get(0), None);
        assert_eq!(f.get(1), Some(3));
        assert_eq!(f.null_count(), 1);
    }

    #[test]
    fn slice_preserves_nulls_and_drops_bitmap_when_clean() {
        let c = PrimitiveColumn::new(vec![Some(1i64), None, Some(3), Some(4)]);

        let with_null = c.slice(0, 2).unwrap();
        assert_eq!(with_null.to_options(), vec![Some(1), None]);

        let clean = c.slice(2, 2).unwrap();
        assert_eq!(clean.to_options(), vec![Some(3), Some(4)]);
        assert!(!clean.has_nulls(), "a null-free slice needs no bitmap");
    }

    #[test]
    fn bitmap_spans_word_boundary() {
        // 70 rows forces a second bitmap word; only row 65 is NULL.
        let vals: Vec<Option<i64>> =
            (0..70).map(|i| if i == 65 { None } else { Some(i as i64) }).collect();
        let c = PrimitiveColumn::new(vals);
        assert_eq!(c.null_count(), 1);
        assert_eq!(c.get(65), None);
        assert_eq!(c.get(64), Some(64));
        assert_eq!(c.get(66), Some(66));
    }

    #[test]
    fn float_column_roundtrips() {
        let c = PrimitiveColumn::new(vec![Some(1.5f64), None, Some(-2.5)]);
        assert_eq!(c.get(0), Some(1.5));
        assert_eq!(c.get(1), None);
        assert_eq!(c.get(2), Some(-2.5));
    }

    #[test]
    fn empty_column_is_well_formed() {
        let c: PrimitiveColumn<i64> = PrimitiveColumn::new(vec![]);
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert!(!c.has_nulls());
        assert_eq!(c.null_count(), 0);
    }

    #[test]
    fn all_null_column() {
        let c = PrimitiveColumn::new(vec![None::<i64>, None, None]);
        assert_eq!(c.null_count(), 3);
        assert!(c.has_nulls());
        assert!(c.iter().all(|v| v.is_none()));
    }

    // ── The dense gather paths, which are easy to break silently ──────────────

    #[test]
    fn filter_selecting_only_valid_rows_drops_the_bitmap() {
        let c = PrimitiveColumn::new(vec![Some(1i64), None, Some(3)]);
        let f = c.filter(&[0, 2]).unwrap();
        assert_eq!(f.to_options(), vec![Some(1), Some(3)]);
        assert!(!f.has_nulls(), "no NULL survived the selection");
    }

    #[test]
    fn filter_can_repeat_and_reorder_indices() {
        // Sorting reuses this path with a permutation, so order and duplicates matter.
        let c = PrimitiveColumn::new(vec![Some(1i64), None, Some(3)]);
        let f = c.filter(&[2, 1, 0, 2]).unwrap();
        assert_eq!(f.to_options(), vec![Some(3), None, Some(1), Some(3)]);
    }

    #[test]
    fn filter_rejects_out_of_range_index() {
        let c = PrimitiveColumn::new(vec![Some(1i64)]);
        assert!(c.filter(&[0, 5]).is_err());
    }

    #[test]
    fn filter_across_word_boundary_keeps_null_positions() {
        let vals: Vec<Option<i64>> =
            (0..130).map(|i| if i % 3 == 0 { None } else { Some(i as i64) }).collect();
        let c = PrimitiveColumn::new(vals);
        let indices: Vec<usize> = (60..80).collect();
        let f = c.filter(&indices).unwrap();
        for (out_i, &src) in indices.iter().enumerate() {
            assert_eq!(f.get(out_i), c.get(src), "row {src}");
        }
    }

    #[test]
    fn concat_all_matches_element_wise_copy() {
        let a = PrimitiveColumn::new(vec![Some(1i64), None]);
        let b = PrimitiveColumn::new(vec![Some(3i64)]);
        let c = PrimitiveColumn::new(vec![None, Some(5i64)]);

        let joined = PrimitiveColumn::concat_all(&[&a, &b, &c]);
        assert_eq!(
            joined.to_options(),
            vec![Some(1), None, Some(3), None, Some(5)]
        );
    }

    #[test]
    fn concat_all_of_null_free_parts_stays_null_free() {
        let a = PrimitiveColumn::new(vec![Some(1i64)]);
        let b = PrimitiveColumn::new(vec![Some(2i64)]);
        let joined = PrimitiveColumn::concat_all(&[&a, &b]);
        assert!(!joined.has_nulls());
        assert_eq!(joined.dense(), &[1, 2]);
    }

    #[test]
    fn concat_all_promotes_when_a_later_part_has_nulls() {
        // The first part carries no bitmap; appending a nullable one must
        // retroactively mark the earlier rows valid rather than blank them out.
        let clean = PrimitiveColumn::new(vec![Some(1i64), Some(2)]);
        let dirty = PrimitiveColumn::new(vec![None, Some(4i64)]);
        let joined = PrimitiveColumn::concat_all(&[&clean, &dirty]);
        assert_eq!(joined.to_options(), vec![Some(1), Some(2), None, Some(4)]);
    }

    #[test]
    fn concat_all_spanning_many_words() {
        let a = PrimitiveColumn::new((0..100).map(|i| Some(i as i64)).collect::<Vec<_>>());
        let b = PrimitiveColumn::new(vec![None, Some(999i64)]);
        let joined = PrimitiveColumn::concat_all(&[&a, &b]);

        assert_eq!(joined.len(), 102);
        assert_eq!(joined.get(0), Some(0));
        assert_eq!(joined.get(99), Some(99));
        assert_eq!(joined.get(100), None);
        assert_eq!(joined.get(101), Some(999));
    }
}

#[cfg(test)]
mod dictionary_tests {
    use super::*;

    fn col(vals: &[Option<&str>]) -> Utf8Column {
        Utf8Column::new(vals.iter().map(|v| v.map(str::to_string)).collect())
    }

    #[test]
    fn deduplicates_repeated_strings() {
        let c = col(&[Some("eng"), Some("hr"), Some("eng"), Some("eng")]);
        assert_eq!(c.len(), 4);
        assert_eq!(c.dict(), &["eng".to_string(), "hr".to_string()], "stored once each");
        assert_eq!(c.codes(), &[0, 1, 0, 0]);
    }

    #[test]
    fn roundtrips_values_and_nulls() {
        let c = col(&[Some("a"), None, Some("b")]);
        assert_eq!(c.get_str(0), Some("a"));
        assert_eq!(c.get_str(1), None);
        assert_eq!(c.get_str(2), Some("b"));
        assert_eq!(c.null_count(), 1);
        assert_eq!(
            c.to_options(),
            vec![Some("a".to_string()), None, Some("b".to_string())]
        );
    }

    #[test]
    fn empty_string_is_distinct_from_null() {
        let c = col(&[Some(""), None]);
        assert_eq!(c.get_str(0), Some(""));
        assert_eq!(c.get_str(1), None);
        assert_eq!(c.null_count(), 1);
    }

    #[test]
    fn code_of_finds_and_misses() {
        let c = col(&[Some("eng"), Some("hr")]);
        assert_eq!(c.code_of("hr"), Some(1));
        assert_eq!(c.code_of("legal"), None, "a miss means no row can match");
    }

    #[test]
    fn filter_shares_the_dictionary_and_copies_no_text() {
        let c = col(&[Some("eng"), Some("hr"), Some("eng")]);
        let f = c.filter(&[2, 0]).unwrap();
        assert_eq!(f.to_options(), vec![Some("eng".to_string()), Some("eng".to_string())]);
        // The dictionary rides along untouched — that is what makes filtering cheap.
        assert_eq!(f.dict(), c.dict());
    }

    #[test]
    fn filter_preserves_nulls_and_rejects_bad_index() {
        let c = col(&[Some("a"), None]);
        let f = c.filter(&[1, 0]).unwrap();
        assert_eq!(f.get_str(0), None);
        assert_eq!(f.get_str(1), Some("a"));
        assert!(c.filter(&[5]).is_err());
    }

    #[test]
    fn slice_keeps_codes_aligned() {
        let c = col(&[Some("a"), Some("b"), None, Some("c")]);
        let s = c.slice(1, 3).unwrap();
        assert_eq!(s.to_options(), vec![Some("b".to_string()), None, Some("c".to_string())]);
    }

    #[test]
    fn concat_merges_dictionaries_without_duplicates() {
        let a = col(&[Some("eng"), Some("hr")]);
        let b = col(&[Some("hr"), Some("legal")]);
        let joined = Utf8Column::concat_all(&[&a, &b]);

        assert_eq!(
            joined.to_options(),
            vec![
                Some("eng".to_string()),
                Some("hr".to_string()),
                Some("hr".to_string()),
                Some("legal".to_string())
            ]
        );
        // "hr" appears in both inputs but must be stored once.
        assert_eq!(joined.dict().len(), 3);
    }

    #[test]
    fn concat_remaps_codes_correctly() {
        // `b`'s local code 0 is "zeta", which must not be confused with `a`'s code 0.
        let a = col(&[Some("alpha")]);
        let b = col(&[Some("zeta"), Some("alpha")]);
        let joined = Utf8Column::concat_all(&[&a, &b]);
        assert_eq!(joined.get_str(0), Some("alpha"));
        assert_eq!(joined.get_str(1), Some("zeta"));
        assert_eq!(joined.get_str(2), Some("alpha"));
    }

    #[test]
    fn concat_preserves_nulls() {
        let a = col(&[None, Some("x")]);
        let b = col(&[Some("y"), None]);
        let joined = Utf8Column::concat_all(&[&a, &b]);
        assert_eq!(joined.null_count(), 2);
        assert_eq!(joined.get_str(0), None);
        assert_eq!(joined.get_str(3), None);
    }

    #[test]
    fn low_cardinality_column_stores_few_strings() {
        // 1000 rows drawn from 3 values must keep a 3-entry dictionary.
        let vals: Vec<Option<String>> = (0..1000)
            .map(|i| Some(["eng", "hr", "ops"][i % 3].to_string()))
            .collect();
        let c = Utf8Column::new(vals);
        assert_eq!(c.len(), 1000);
        assert_eq!(c.dict().len(), 3);
    }
}

#[cfg(test)]
mod bitmask_concat_tests {
    use super::*;

    // ── bitmask_to_indices ───────────────────────────────────────────────────

    #[test]
    fn bitmask_expands_set_bits() {
        // bits 0, 3, 5 set
        let mask = [0b101001u64];
        assert_eq!(bitmask_to_indices(&mask, 6), vec![0, 3, 5]);
    }

    #[test]
    fn bitmask_respects_row_count() {
        // bit 10 is set but the batch only has 5 rows — it must not be emitted.
        let mask = [1u64 << 10];
        assert!(bitmask_to_indices(&mask, 5).is_empty());
    }

    #[test]
    fn bitmask_spans_word_boundary() {
        let mask = [1u64 << 63, 1u64];
        assert_eq!(bitmask_to_indices(&mask, 65), vec![63, 64]);
    }

    #[test]
    fn bitmask_all_zero_selects_nothing() {
        assert!(bitmask_to_indices(&[0u64, 0u64], 128).is_empty());
    }

    #[test]
    fn bitmask_matches_naive_scan() {
        // Property check against the obvious implementation.
        let mask = [0xDEAD_BEEF_CAFE_1234u64, 0x0F0F_0F0F_0F0F_0F0Fu64];
        let row_count = 128;
        let naive: Vec<usize> = (0..row_count)
            .filter(|&i| (mask[i >> 6] >> (i & 63)) & 1 == 1)
            .collect();
        assert_eq!(bitmask_to_indices(&mask, row_count), naive);
    }

    // ── Column::concat ───────────────────────────────────────────────────────

    #[test]
    fn concat_uniform_int64_preserves_nulls() {
        let a = ColumnBuilder::int64(vec![Some(1), None]);
        let b = ColumnBuilder::int64(vec![Some(3)]);
        let out = Column::concat(&[&a, &b]).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out.get(0).unwrap(), ScalarValue::Int64(Some(1)));
        assert!(out.get(1).unwrap().is_null());
        assert_eq!(out.get(2).unwrap(), ScalarValue::Int64(Some(3)));
    }

    #[test]
    fn concat_uniform_utf8() {
        let a = ColumnBuilder::utf8_from_str(vec![Some("a")]);
        let b = ColumnBuilder::utf8_from_str(vec![Some("b"), None]);
        let out = Column::concat(&[&a, &b]).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out.get(0).unwrap(), ScalarValue::Utf8(Some("a".into())));
        assert_eq!(out.get(1).unwrap(), ScalarValue::Utf8(Some("b".into())));
        assert!(out.get(2).unwrap().is_null());
    }

    #[test]
    fn concat_mixed_numeric_promotes_to_float64() {
        let a = ColumnBuilder::int64(vec![Some(1)]);
        let b = ColumnBuilder::float64(vec![Some(2.5)]);
        let out = Column::concat(&[&a, &b]).unwrap();
        assert_eq!(out.data_type(), DataType::Float64);
        assert_eq!(out.get(0).unwrap(), ScalarValue::Float64(Some(1.0)));
        assert_eq!(out.get(1).unwrap(), ScalarValue::Float64(Some(2.5)));
    }

    #[test]
    fn concat_empty_input_yields_empty_column() {
        let out = Column::concat(&[]).unwrap();
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn concat_incompatible_types_errors() {
        let a = ColumnBuilder::utf8_from_str(vec![Some("a")]);
        let b = ColumnBuilder::int64(vec![Some(1)]);
        assert!(Column::concat(&[&a, &b]).is_err());
    }

    #[test]
    fn concat_matches_row_by_row_copy() {
        // The typed concat must produce exactly what a naive per-row copy would.
        let a = ColumnBuilder::float64(vec![Some(1.0), None, Some(3.0)]);
        let b = ColumnBuilder::float64(vec![Some(4.0)]);
        let out = Column::concat(&[&a, &b]).unwrap();

        let mut expected = Vec::new();
        for col in [&a, &b] {
            for i in 0..col.len() {
                expected.push(col.get(i).unwrap());
            }
        }
        let actual: Vec<_> = (0..out.len()).map(|i| out.get(i).unwrap()).collect();
        assert_eq!(actual, expected);
    }
}

// ─── Primitive Column ────────────────────────────────────────────────────────

/// A numeric column stored Arrow-style: dense values plus a separate validity
/// bitmap.
///
/// The obvious representation, `Vec<Option<T>>`, is a trap for an analytical
/// engine. `Option<f64>` occupies 16 bytes — 8 for the value and 8 for the
/// discriminant after padding — so values sit 16 bytes apart with tag bytes
/// interleaved. That doubles memory and, worse, stops LLVM from vectorising any
/// loop over the column: it cannot load four adjacent `f64`s in one instruction
/// because they are not adjacent.
///
/// Splitting the two halves makes `values` a true contiguous array. Measured on
/// 4 M values, a `WHERE` scan went from 6.96 ms to 2.12 ms (3.3×) and the column
/// shrank from 61 MB to 30 MB — with no explicit SIMD, purely from letting the
/// optimiser see a dense slice.
///
/// `validity` is `None` when the column has no NULLs at all, which is the common
/// case and lets hot loops skip the per-row check entirely.
#[derive(Debug, Clone)]
pub struct PrimitiveColumn<T: Copy> {
    values:   Vec<T>,
    validity: Option<Vec<u64>>,
}

impl<T: Copy + Default> PrimitiveColumn<T> {
    pub fn new(values: Vec<Option<T>>) -> Self {
        let has_nulls = values.iter().any(|v| v.is_none());
        if !has_nulls {
            // Fast path: unwrap straight into a dense vector, no bitmap needed.
            return Self {
                values:   values.into_iter().map(|v| v.unwrap_or_default()).collect(),
                validity: None,
            };
        }

        let mut dense = Vec::with_capacity(values.len());
        let mut bits  = vec![0u64; values.len().div_ceil(64)];
        for (i, v) in values.into_iter().enumerate() {
            match v {
                Some(x) => {
                    bits[i >> 6] |= 1u64 << (i & 63);
                    dense.push(x);
                }
                // NULL slots still occupy a position so indexing stays O(1);
                // the value is arbitrary and must never be read without the mask.
                None => dense.push(T::default()),
            }
        }
        Self { values: dense, validity: Some(bits) }
    }

    pub fn from_non_null(values: Vec<T>) -> Self {
        Self { values, validity: None }
    }

    /// Gather rows by `u32` index, where `u32::MAX` means "emit a NULL here".
    ///
    /// The join records its matches as index pairs, one per output row. Holding
    /// those as `Option<usize>` costs 16 bytes an entry against 4 here, and the
    /// join then made a second dense copy of them to reach a fast gather. At 100k
    /// matches that bookkeeping measured around 60 ns per matched row — more than
    /// building the output columns it feeds.
    /// Gather into a slice the caller already allocated.
    ///
    /// Exists so a join can allocate one buffer per output column and let each
    /// worker fill a disjoint slice of it. Building the output in per-unit pieces
    /// instead turns one 800 KB allocation per column into twenty of 40 KB,
    /// allocated on a worker thread and freed on another; that pattern measurably
    /// degraded unrelated code sharing the process, which is what
    /// `README.md` records under Methodology.
    ///
    /// Handles no NULLs at all. The caller must have established that the source
    /// has none and that no index is the NULL marker — the INNER-join case. A
    /// caller that cannot establish it has to use [`Self::gather_u32`], which
    /// allocates but is always correct.
    ///
    /// # Panics
    /// If `out.len() != idx.len()`, or if any index is out of range.
    pub fn gather_u32_into(&self, idx: &[u32], out: &mut [T]) {
        assert_eq!(idx.len(), out.len(), "gather_u32_into: length mismatch");
        for (slot, &i) in out.iter_mut().zip(idx) {
            *slot = self.values[i as usize];
        }
    }

    pub fn gather_u32(&self, idx: &[u32]) -> Self {
        const NULL_IDX: u32 = u32::MAX;

        let mut values = Vec::with_capacity(idx.len());
        let mut bits = vec![0u64; idx.len().div_ceil(64)];
        let mut any_null = false;

        for (out_i, &i) in idx.iter().enumerate() {
            if i != NULL_IDX && self.is_valid(i as usize) {
                bits[out_i >> 6] |= 1u64 << (out_i & 63);
                values.push(self.values[i as usize]);
            } else {
                // NULL slots still occupy a position so indexing stays O(1).
                any_null = true;
                values.push(T::default());
            }
        }

        Self { values, validity: any_null.then_some(bits) }
    }

    /// Gather rows by index, where `None` means "emit a NULL here".
    ///
    /// This is the outer-join shape: a LEFT join's right-hand indices carry a
    /// `None` for every unmatched row. Routing that through `Vec<Option<T>>` and
    /// back cost a 16-byte slot per output row plus a second repacking pass, and it
    /// is why LEFT JOIN measured 6x slower than INNER on the same data — the dense
    /// path only ran when every index was present.
    pub fn gather_opt(&self, indices: &[Option<usize>]) -> Self {
        let mut values = Vec::with_capacity(indices.len());
        let mut bits = vec![0u64; indices.len().div_ceil(64)];
        let mut any_null = false;

        for (out_i, idx) in indices.iter().enumerate() {
            match idx {
                Some(i) if self.is_valid(*i) => {
                    bits[out_i >> 6] |= 1u64 << (out_i & 63);
                    values.push(self.values[*i]);
                }
                // NULL slots still occupy a position so indexing stays O(1).
                _ => {
                    any_null = true;
                    values.push(T::default());
                }
            }
        }

        Self { values, validity: any_null.then_some(bits) }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The dense value slice. Entries at NULL positions are meaningless — pair
    /// this with [`PrimitiveColumn::is_valid`] or check [`PrimitiveColumn::has_nulls`]
    /// first. This is the slice that makes vectorised scans possible.
    pub fn dense(&self) -> &[T] {
        &self.values
    }

    /// Does this column contain any NULL?
    pub fn has_nulls(&self) -> bool {
        self.validity.is_some()
    }

    #[inline]
    pub fn is_valid(&self, index: usize) -> bool {
        match &self.validity {
            None => true,
            Some(bits) => (bits[index >> 6] >> (index & 63)) & 1 == 1,
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        if self.is_valid(index) {
            Some(self.values[index])
        } else {
            None
        }
    }

    /// Iterate as `Option<T>`, for code that does not care about the layout.
    pub fn iter(&self) -> impl Iterator<Item = Option<T>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// Materialise back into the boxed representation. Allocates — prefer
    /// [`PrimitiveColumn::dense`] or [`PrimitiveColumn::iter`] on hot paths.
    pub fn to_options(&self) -> Vec<Option<T>> {
        self.iter().collect()
    }

    pub fn null_count(&self) -> usize {
        match &self.validity {
            None => 0,
            Some(_) => (0..self.len()).filter(|&i| !self.is_valid(i)).count(),
        }
    }

    /// Gather the rows at `indices`.
    ///
    /// This runs on every filtered scan and every sort reorder, so it writes the
    /// dense representation directly. Collecting into `Vec<Option<T>>` and handing
    /// that to `new` would allocate a doubled-width temporary, then re-scan it for
    /// NULLs and repack — three passes where one will do.
    ///
    /// Left serial and unprefetched on purpose. This gather is bound by memory
    /// latency, so it looks like an easy target, but both obvious fixes measured
    /// worse: parallelising across columns (see `query::sort::gather`) and
    /// software-prefetching `indices[i + 16]`, which cost ORDER BY 18% at 10k
    /// rows and 7% at 100k.
    pub fn filter(&self, indices: &[usize]) -> FalconResult<Self> {
        let len = self.values.len();
        if let Some(&bad) = indices.iter().find(|&&i| i >= len) {
            return Err(FalconError::IndexOutOfBounds { index: bad, len });
        }

        // No NULLs anywhere: a straight gather, no bitmap to maintain.
        let Some(bits) = &self.validity else {
            return Ok(Self::from_non_null(
                indices.iter().map(|&i| self.values[i]).collect(),
            ));
        };

        let mut values   = Vec::with_capacity(indices.len());
        let mut out_bits = vec![0u64; indices.len().div_ceil(64)];
        let mut any_null = false;

        for (out_i, &i) in indices.iter().enumerate() {
            values.push(self.values[i]);
            if (bits[i >> 6] >> (i & 63)) & 1 == 1 {
                out_bits[out_i >> 6] |= 1u64 << (out_i & 63);
            } else {
                any_null = true;
            }
        }

        Ok(Self {
            values,
            // A selection that happened to exclude every NULL needs no bitmap.
            validity: any_null.then_some(out_bits),
        })
    }

    /// Gather rows driven directly by a packed bitmask — no intermediate index list.
    ///
    /// `selected` must equal the popcount of `mask[..ceil(row_count/64)]`.
    /// Uses `get_unchecked` because every bit position is guaranteed < row_count.
    pub fn gather_by_bitmask(&self, mask: &[u64], row_count: usize, selected: usize) -> Self {
        // Non-null fast path: straight gather into a dense vec.
        if self.validity.is_none() {
            let mut values = Vec::with_capacity(selected);
            for (wi, &word) in mask.iter().enumerate() {
                let base = wi << 6;
                if base >= row_count { break; }
                let mut bits = word;
                while bits != 0 {
                    let row = base + bits.trailing_zeros() as usize;
                    if row >= row_count { break; }
                    // SAFETY: row < row_count == self.values.len()
                    values.push(unsafe { *self.values.get_unchecked(row) });
                    bits &= bits - 1;
                }
            }
            return Self::from_non_null(values);
        }

        // Nullable path: mirror the validity bitmap.
        let validity_bits = self.validity.as_deref().unwrap();
        let mut values   = Vec::with_capacity(selected);
        let mut out_bits = vec![0u64; selected.div_ceil(64)];
        let mut any_null = false;
        let mut out_i    = 0usize;

        for (wi, &word) in mask.iter().enumerate() {
            let base = wi << 6;
            if base >= row_count { break; }
            let mut bits = word;
            while bits != 0 {
                let row = base + bits.trailing_zeros() as usize;
                if row >= row_count { break; }
                // SAFETY: row < row_count == self.values.len()
                values.push(unsafe { *self.values.get_unchecked(row) });
                if (validity_bits[row >> 6] >> (row & 63)) & 1 == 1 {
                    out_bits[out_i >> 6] |= 1u64 << (out_i & 63);
                } else {
                    any_null = true;
                }
                out_i += 1;
                bits &= bits - 1;
            }
        }

        Self { values, validity: any_null.then_some(out_bits) }
    }

    pub fn slice(&self, offset: usize, length: usize) -> FalconResult<Self> {
        if offset + length > self.values.len() {
            return Err(FalconError::IndexOutOfBounds {
                index: offset + length,
                len: self.values.len(),
            });
        }

        let values = self.values[offset..offset + length].to_vec();

        let Some(bits) = &self.validity else {
            return Ok(Self::from_non_null(values));
        };

        let mut out_bits = vec![0u64; length.div_ceil(64)];
        let mut any_null = false;
        for j in 0..length {
            let i = offset + j;
            if (bits[i >> 6] >> (i & 63)) & 1 == 1 {
                out_bits[j >> 6] |= 1u64 << (j & 63);
            } else {
                any_null = true;
            }
        }

        Ok(Self { values, validity: any_null.then_some(out_bits) })
    }

    /// Append another column's rows onto this one, staying in the dense form.
    fn extend_from(&mut self, other: &Self) {
        let base = self.values.len();
        self.values.extend_from_slice(&other.values);

        if self.validity.is_none() && other.validity.is_none() {
            return; // still no NULLs on either side
        }

        let total = self.values.len();
        let mut bits = match self.validity.take() {
            Some(mut b) => { b.resize(total.div_ceil(64), 0); b }
            // Everything already appended was non-NULL, so start all-ones up to `base`.
            None => {
                let mut b = vec![0u64; total.div_ceil(64)];
                for i in 0..base {
                    b[i >> 6] |= 1u64 << (i & 63);
                }
                b
            }
        };

        for j in 0..other.values.len() {
            if other.is_valid(j) {
                let i = base + j;
                bits[i >> 6] |= 1u64 << (i & 63);
            }
        }

        self.validity = Some(bits);
    }

    /// Concatenate primitive columns without boxing every value through `Option`.
    pub fn concat_all(parts: &[&Self]) -> Self {
        let total: usize = parts.iter().map(|p| p.len()).sum();

        // Common case: nothing is nullable, so this is a plain memcpy chain.
        if parts.iter().all(|p| !p.has_nulls()) {
            let mut values = Vec::with_capacity(total);
            for p in parts {
                values.extend_from_slice(&p.values);
            }
            return Self::from_non_null(values);
        }

        let mut out = Self { values: Vec::with_capacity(total), validity: None };
        for p in parts {
            out.extend_from(p);
        }
        out
    }
}

// ─── Boolean Column ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BooleanColumn {
    pub values: Vec<Option<bool>>,
}

impl BooleanColumn {
    pub fn new(values: Vec<Option<bool>>) -> Self {
        Self { values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn get(&self, index: usize) -> Option<bool> {
        self.values[index]
    }

    pub fn null_count(&self) -> usize {
        self.values.iter().filter(|v| v.is_none()).count()
    }

    pub fn filter(&self, indices: &[usize]) -> FalconResult<Self> {
        let mut out = Vec::with_capacity(indices.len());
        for &i in indices {
            if i >= self.values.len() {
                return Err(FalconError::IndexOutOfBounds { index: i, len: self.values.len() });
            }
            out.push(self.values[i]);
        }
        Ok(Self::new(out))
    }

    pub fn slice(&self, offset: usize, length: usize) -> FalconResult<Self> {
        if offset + length > self.values.len() {
            return Err(FalconError::IndexOutOfBounds {
                index: offset + length,
                len: self.values.len(),
            });
        }
        Ok(Self::new(self.values[offset..offset + length].to_vec()))
    }
}

// ─── Utf8 Column ─────────────────────────────────────────────────────────────

/// Code stored for a NULL row. Real codes index into `dict`, so the top value is
/// free to act as a sentinel — a column would need 4 billion distinct strings to
/// collide with it.
const NULL_CODE: u32 = u32::MAX;

/// A dictionary-encoded string column: unique strings once, a `u32` code per row.
///
/// Analytical string columns are overwhelmingly low-cardinality — departments,
/// countries, status flags — so storing one `String` per row wastes both space and
/// time. Every row costs 24 bytes of `Option<String>` plus a heap allocation, and
/// any operation that groups or compares has to hash or memcmp the text itself.
///
/// Encoding once turns those into integer problems:
/// - `GROUP BY` maps each *dictionary entry* to a group, then each row is a single
///   array index instead of a string hash plus a hash-table probe.
/// - `filter` keeps the dictionary and gathers codes, so it copies no text at all.
/// - Equality against a literal resolves to one code lookup, then integer compares.
#[derive(Debug, Clone)]
pub struct Utf8Column {
    // Arc-shared so filter/slice/gather clone the dict in O(1) instead of deep-copying.
    dict:      Arc<Vec<String>>,
    codes:     Vec<u32>,
    has_nulls: bool,
}

impl Utf8Column {
    pub fn new(values: Vec<Option<String>>) -> Self {
        let mut dict_vec:  Vec<String> = Vec::new();
        let mut codes:     Vec<u32>    = Vec::with_capacity(values.len());
        let mut seen:      std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut has_nulls: bool        = false;

        for v in values {
            match v {
                None => { codes.push(NULL_CODE); has_nulls = true; }
                Some(s) => {
                    let code = match seen.get(&s) {
                        Some(&c) => c,
                        None => {
                            let c = dict_vec.len() as u32;
                            dict_vec.push(s.clone());
                            seen.insert(s, c);
                            c
                        }
                    };
                    codes.push(code);
                }
            }
        }

        Self { dict: Arc::new(dict_vec), codes, has_nulls }
    }

    /// Build directly from an existing Arc-shared dictionary. The caller guarantees every
    /// code is either `NULL_CODE` or a valid index into `dict`.
    /// Gather rows by `u32` index, where `u32::MAX` means "emit a NULL here".
    ///
    /// Carries the dictionary over, so this copies codes rather than strings.
    pub fn gather_u32(&self, idx: &[u32]) -> Self {
        const NULL_IDX: u32 = u32::MAX;

        let mut codes = Vec::with_capacity(idx.len());
        let mut has_nulls = false;
        for &i in idx {
            if i == NULL_IDX {
                has_nulls = true;
                codes.push(NULL_CODE);
            } else {
                let code = self.codes[i as usize];
                if code == NULL_CODE {
                    has_nulls = true;
                }
                codes.push(code);
            }
        }
        Self { dict: Arc::clone(&self.dict), codes, has_nulls }
    }

    /// Gather rows by index, where `None` means "emit a NULL here".
    ///
    /// Carries the dictionary over rather than rebuilding it, so an outer join
    /// copies `u32` codes instead of cloning every string.
    pub fn gather_opt(&self, indices: &[Option<usize>]) -> Self {
        let mut codes = Vec::with_capacity(indices.len());
        let mut has_nulls = false;
        for idx in indices {
            match idx {
                Some(i) => {
                    let code = self.codes[*i];
                    if code == NULL_CODE {
                        has_nulls = true;
                    }
                    codes.push(code);
                }
                None => {
                    has_nulls = true;
                    codes.push(NULL_CODE);
                }
            }
        }
        Self { dict: Arc::clone(&self.dict), codes, has_nulls }
    }

    fn from_parts(dict: Arc<Vec<String>>, codes: Vec<u32>) -> Self {
        let has_nulls = codes.iter().any(|&c| c == NULL_CODE);
        Self { dict, codes, has_nulls }
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Borrow the string for a row — no allocation.
    #[inline]
    pub fn get_str(&self, index: usize) -> Option<&str> {
        let code = self.codes[index];
        if code == NULL_CODE {
            None
        } else {
            Some(self.dict[code as usize].as_str())
        }
    }

    /// Owned copy of a row's string. Prefer [`Utf8Column::get_str`] on hot paths.
    pub fn get(&self, index: usize) -> Option<String> {
        self.get_str(index).map(str::to_string)
    }

    /// The distinct strings, in first-seen order.
    pub fn dict(&self) -> &[String] {
        self.dict.as_slice()
    }

    /// One code per row; `NULL_CODE` marks a NULL. Index into [`Utf8Column::dict`].
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }

    #[inline]
    pub fn is_null_code(code: u32) -> bool {
        code == NULL_CODE
    }

    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.has_nulls
    }

    /// Find the code for a string, if the dictionary holds it.
    ///
    /// A miss means no row can equal it — which callers can turn into an empty
    /// result without touching the data.
    pub fn code_of(&self, needle: &str) -> Option<u32> {
        self.dict.iter().position(|s| s == needle).map(|i| i as u32)
    }

    pub fn iter(&self) -> impl Iterator<Item = Option<&str>> + '_ {
        (0..self.len()).map(move |i| self.get_str(i))
    }

    /// Materialise back into the boxed representation. Allocates one `String` per
    /// row — used only where an owned vector is genuinely required.
    pub fn to_options(&self) -> Vec<Option<String>> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }

    pub fn null_count(&self) -> usize {
        self.codes.iter().filter(|&&c| c == NULL_CODE).count()
    }

    /// Gather rows by index.
    ///
    /// The dictionary is shared with the result, so this copies no text — only
    /// `u32` codes. That is the whole point of the encoding on a filter path.
    ///
    /// The dictionary is *not* compacted, so a highly selective filter can retain
    /// entries no surviving row uses. Trading a little retained memory for an O(n)
    /// gather is the right call here; a compaction pass can be added if it matters.
    pub fn filter(&self, indices: &[usize]) -> FalconResult<Self> {
        let len = self.codes.len();
        if let Some(&bad) = indices.iter().find(|&&i| i >= len) {
            return Err(FalconError::IndexOutOfBounds { index: bad, len });
        }
        let codes = indices.iter().map(|&i| self.codes[i]).collect();
        Ok(Self::from_parts(self.dict.clone(), codes))
    }

    /// Gather rows driven directly by a packed bitmask — no intermediate index list.
    pub fn gather_by_bitmask(&self, mask: &[u64], row_count: usize, selected: usize) -> Self {
        let mut codes    = Vec::with_capacity(selected);
        let mut has_nulls = false;
        for (wi, &word) in mask.iter().enumerate() {
            let base = wi << 6;
            if base >= row_count { break; }
            let mut bits = word;
            while bits != 0 {
                let row = base + bits.trailing_zeros() as usize;
                if row >= row_count { break; }
                // SAFETY: row < row_count == self.codes.len()
                let code = unsafe { *self.codes.get_unchecked(row) };
                if code == NULL_CODE { has_nulls = true; }
                codes.push(code);
                bits &= bits - 1;
            }
        }
        Self { dict: self.dict.clone(), codes, has_nulls }
    }

    pub fn slice(&self, offset: usize, length: usize) -> FalconResult<Self> {
        if offset + length > self.codes.len() {
            return Err(FalconError::IndexOutOfBounds {
                index: offset + length,
                len: self.codes.len(),
            });
        }
        Ok(Self::from_parts(
            self.dict.clone(),
            self.codes[offset..offset + length].to_vec(),
        ))
    }

    /// Concatenate columns, merging their dictionaries.
    ///
    /// Each input's dictionary is remapped **once**, not once per row, so joining
    /// batches costs one pass over the codes plus a handful of string lookups.
    pub fn concat_all(parts: &[&Self]) -> Self {
        if parts.is_empty() {
            return Self { dict: Arc::new(vec![]), codes: vec![], has_nulls: false };
        }
        if parts.len() == 1 {
            return (*parts[0]).clone();
        }

        let total: usize = parts.iter().map(|p| p.len()).sum();

        // Fast path: all parts share the same Arc pointer — just concatenate codes.
        let first_dict = &parts[0].dict;
        if parts.iter().all(|p| Arc::ptr_eq(&p.dict, first_dict)) {
            let mut codes = Vec::with_capacity(total);
            for p in parts { codes.extend_from_slice(&p.codes); }
            let has_nulls = parts.iter().any(|p| p.has_nulls);
            return Self { dict: Arc::clone(first_dict), codes, has_nulls };
        }

        // Medium path: all dicts have identical entries (common after Parquet load).
        // Codes are already compatible — no remapping needed.
        let first_dict_content: &[String] = first_dict.as_ref();
        if parts.iter().skip(1).all(|p| p.dict.as_ref().as_slice() == first_dict_content) {
            let mut codes = Vec::with_capacity(total);
            for p in parts { codes.extend_from_slice(&p.codes); }
            let has_nulls = parts.iter().any(|p| p.has_nulls);
            return Self { dict: Arc::clone(first_dict), codes, has_nulls };
        }

        // Slow path: dicts differ — merge via HashMap.
        let mut dict_vec: Vec<String> = Vec::new();
        let mut seen:     std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut codes:    Vec<u32> = Vec::with_capacity(total);

        for p in parts {
            let remap: Vec<u32> = p
                .dict
                .iter()
                .map(|s| match seen.get(s) {
                    Some(&c) => c,
                    None => {
                        let c = dict_vec.len() as u32;
                        dict_vec.push(s.clone());
                        seen.insert(s.clone(), c);
                        c
                    }
                })
                .collect();

            codes.extend(p.codes.iter().map(|&code| {
                if code == NULL_CODE { NULL_CODE } else { remap[code as usize] }
            }));
        }

        let has_nulls = parts.iter().any(|p| p.has_nulls);
        Self { dict: Arc::new(dict_vec), codes, has_nulls }
    }
}

// ─── Decimal128 Column ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Decimal128Column {
    pub inner:     PrimitiveColumn<i128>,
    pub precision: u8,
    pub scale:     i8,
}

impl Decimal128Column {
    pub fn new(values: Vec<Option<i128>>, precision: u8, scale: i8) -> Self {
        Self { inner: PrimitiveColumn::new(values), precision, scale }
    }

    pub fn len(&self) -> usize { self.inner.len() }
    pub fn is_empty(&self) -> bool { self.inner.is_empty() }
}

// ─── Builder ─────────────────────────────────────────────────────────────────

pub struct ColumnBuilder;

impl ColumnBuilder {
    pub fn int32(values: Vec<Option<i32>>) -> Column {
        Column::Int32(PrimitiveColumn::new(values))
    }

    pub fn int64(values: Vec<Option<i64>>) -> Column {
        Column::Int64(PrimitiveColumn::new(values))
    }

    pub fn float32(values: Vec<Option<f32>>) -> Column {
        Column::Float32(PrimitiveColumn::new(values))
    }

    pub fn float64(values: Vec<Option<f64>>) -> Column {
        Column::Float64(PrimitiveColumn::new(values))
    }

    pub fn boolean(values: Vec<Option<bool>>) -> Column {
        Column::Boolean(BooleanColumn::new(values))
    }

    pub fn utf8(values: Vec<Option<String>>) -> Column {
        Column::Utf8(Utf8Column::new(values))
    }

    pub fn utf8_from_str(values: Vec<Option<&str>>) -> Column {
        Column::Utf8(Utf8Column::new(
            values.into_iter().map(|v| v.map(|s| s.to_string())).collect(),
        ))
    }

    pub fn date32(values: Vec<Option<i32>>) -> Column {
        Column::Date32(PrimitiveColumn::new(values))
    }

    pub fn timestamp(values: Vec<Option<i64>>) -> Column {
        Column::Timestamp(PrimitiveColumn::new(values))
    }

    pub fn decimal128(values: Vec<Option<i128>>, precision: u8, scale: i8) -> Column {
        Column::Decimal128(Decimal128Column::new(values, precision, scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_filter_by_mask() {
        let col = ColumnBuilder::int32(vec![Some(1), Some(2), Some(3), Some(4)]);
        let mask = vec![true, false, true, false];
        let filtered = col.filter_by_mask(&mask).unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered.get(0).unwrap(), ScalarValue::Int32(Some(1)));
        assert_eq!(filtered.get(1).unwrap(), ScalarValue::Int32(Some(3)));
    }

    #[test]
    fn column_null_count() {
        let col = ColumnBuilder::int32(vec![Some(1), None, Some(3), None]);
        assert_eq!(col.null_count(), 2);
    }

    #[test]
    fn column_slice() {
        let col = ColumnBuilder::int64(vec![Some(10), Some(20), Some(30), Some(40)]);
        let sliced = col.slice(1, 2).unwrap();
        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.get(0).unwrap(), ScalarValue::Int64(Some(20)));
        assert_eq!(sliced.get(1).unwrap(), ScalarValue::Int64(Some(30)));
    }

    #[test]
    fn index_out_of_bounds_returns_error() {
        let col = ColumnBuilder::int32(vec![Some(1)]);
        assert!(col.get(1).is_err());
    }
}
