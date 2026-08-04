use std::collections::{BTreeMap, HashMap};

use super::{
    column::Column,
    types::{FalconResult, ScalarValue},
};

// ─── Index trait ──────────────────────────────────────────────────────────────

pub trait Index: Send + Sync {
    fn column_name(&self) -> &str;

    /// Return all row indices where the column equals `value`.
    fn lookup_eq(&self, value: &ScalarValue) -> Vec<usize>;

    /// Return all row indices where the column is in the range [lo, hi].
    fn lookup_range(&self, lo: &ScalarValue, hi: &ScalarValue) -> Vec<usize>;

    /// Return all row indices where the column is in a given set.
    fn lookup_in(&self, values: &[ScalarValue]) -> Vec<usize> {
        values.iter().flat_map(|v| self.lookup_eq(v)).collect()
    }
}

// ─── Shared key type ─────────────────────────────────────────────────────────

/// Normalized, orderable key derived from a ScalarValue.
///
/// All numeric types (integer, float, date, timestamp, decimal) map to the
/// same `Num` variant via their f64 representation. This is the only safe
/// choice: a SQL literal is typed by the parser independently of the column
/// type, so `WHERE score = 5` on a DOUBLE column produces an Int64 probe key.
/// If integers and floats lived in separate enum variants the probe would be
/// a different key than every entry in the tree and the lookup would silently
/// return zero rows.
///
/// f64 has 53 bits of mantissa, so integers up to 2^53 round-trip exactly.
/// Larger i64 values lose precision; those columns should not be indexed on
/// equality in SQL that uses floating-point literals anyway.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum IndexKey {
    Null,
    Num(u64),   // all numeric types — f64 bit-pattern ordered by f64_to_ord_bits
    Str(String),
    Bool(u8),
}

impl IndexKey {
    fn from_scalar(v: &ScalarValue) -> Self {
        match v {
            ScalarValue::Null               => Self::Null,
            ScalarValue::Boolean(None)      => Self::Null,
            ScalarValue::Boolean(Some(b))   => Self::Bool(*b as u8),
            ScalarValue::Int32(None)        => Self::Null,
            ScalarValue::Int32(Some(n))     => Self::Num(f64_to_ord_bits(*n as f64)),
            ScalarValue::Int64(None)        => Self::Null,
            ScalarValue::Int64(Some(n))     => Self::Num(f64_to_ord_bits(*n as f64)),
            ScalarValue::Float32(None)      => Self::Null,
            ScalarValue::Float32(Some(f))   => Self::Num(f64_to_ord_bits(*f as f64)),
            ScalarValue::Float64(None)      => Self::Null,
            ScalarValue::Float64(Some(f))   => Self::Num(f64_to_ord_bits(*f)),
            ScalarValue::Utf8(None)              => Self::Null,
            ScalarValue::Utf8(Some(s))           => Self::Str(s.clone()),
            ScalarValue::Date32(None)            => Self::Null,
            ScalarValue::Date32(Some(d))         => Self::Num(f64_to_ord_bits(*d as f64)),
            ScalarValue::Timestamp(None)         => Self::Null,
            ScalarValue::Timestamp(Some(ts))     => Self::Num(f64_to_ord_bits(*ts as f64)),
            ScalarValue::Decimal128(d) if d.value.is_none() => Self::Null,
            ScalarValue::Decimal128(d) => Self::Num(f64_to_ord_bits(
                d.value.unwrap_or(0) as f64 / 10f64.powi(d.scale as i32),
            )),
        }
    }
}

/// Convert f64 to a u64 that sorts in the same order as the float.
fn f64_to_ord_bits(f: f64) -> u64 {
    let bits = f.to_bits();
    if bits >> 63 == 0 {
        bits | (1 << 63)
    } else {
        !bits
    }
}

// ─── B-tree Index ─────────────────────────────────────────────────────────────

/// Sorted index backed by a BTreeMap. Supports O(log n) point lookups
/// and efficient range queries.
pub struct BTreeIndex {
    name: String,
    tree: BTreeMap<IndexKey, Vec<usize>>,
}

impl BTreeIndex {
    pub fn build(column_name: &str, col: &Column) -> FalconResult<Self> {
        let mut tree: BTreeMap<IndexKey, Vec<usize>> = BTreeMap::new();
        for row in 0..col.len() {
            let key = IndexKey::from_scalar(&col.get(row)?);
            tree.entry(key).or_default().push(row);
        }
        Ok(Self { name: column_name.to_string(), tree })
    }

    pub fn entry_count(&self) -> usize {
        self.tree.len()
    }
}

impl Index for BTreeIndex {
    fn column_name(&self) -> &str { &self.name }

    fn lookup_eq(&self, value: &ScalarValue) -> Vec<usize> {
        let key = IndexKey::from_scalar(value);
        self.tree.get(&key).cloned().unwrap_or_default()
    }

    fn lookup_range(&self, lo: &ScalarValue, hi: &ScalarValue) -> Vec<usize> {
        let lo_key = IndexKey::from_scalar(lo);
        let hi_key = IndexKey::from_scalar(hi);
        self.tree
            .range(lo_key..=hi_key)
            .flat_map(|(_, rows)| rows.iter().copied())
            .collect()
    }
}

// ─── Hash Index ───────────────────────────────────────────────────────────────

/// Hash index for O(1) exact-match lookups. Does not support range queries.
pub struct HashIndex {
    name: String,
    map:  HashMap<IndexKey, Vec<usize>>,
}

impl HashIndex {
    pub fn build(column_name: &str, col: &Column) -> FalconResult<Self> {
        let mut map: HashMap<IndexKey, Vec<usize>> = HashMap::new();
        for row in 0..col.len() {
            let key = IndexKey::from_scalar(&col.get(row)?);
            map.entry(key).or_default().push(row);
        }
        Ok(Self { name: column_name.to_string(), map })
    }

    pub fn entry_count(&self) -> usize {
        self.map.len()
    }
}

impl Index for HashIndex {
    fn column_name(&self) -> &str { &self.name }

    fn lookup_eq(&self, value: &ScalarValue) -> Vec<usize> {
        let key = IndexKey::from_scalar(value);
        self.map.get(&key).cloned().unwrap_or_default()
    }

    fn lookup_range(&self, _lo: &ScalarValue, _hi: &ScalarValue) -> Vec<usize> {
        // Hash index cannot answer range queries — callers must fall back to full scan.
        Vec::new()
    }
}

// ─── Index registry ───────────────────────────────────────────────────────────

/// Holds all indices for a single table.
pub struct IndexRegistry {
    btree:  HashMap<String, BTreeIndex>,
    hash:   HashMap<String, HashIndex>,
}

impl IndexRegistry {
    pub fn new() -> Self {
        Self { btree: HashMap::new(), hash: HashMap::new() }
    }

    pub fn add_btree(&mut self, name: &str, col: &Column) -> FalconResult<()> {
        let idx = BTreeIndex::build(name, col)?;
        self.btree.insert(name.to_string(), idx);
        Ok(())
    }

    pub fn add_hash(&mut self, name: &str, col: &Column) -> FalconResult<()> {
        let idx = HashIndex::build(name, col)?;
        self.hash.insert(name.to_string(), idx);
        Ok(())
    }

    /// Look up rows by exact value. Prefers hash index, falls back to btree.
    pub fn lookup_eq(&self, column: &str, value: &ScalarValue) -> Option<Vec<usize>> {
        if let Some(idx) = self.hash.get(column) {
            return Some(idx.lookup_eq(value));
        }
        if let Some(idx) = self.btree.get(column) {
            return Some(idx.lookup_eq(value));
        }
        None
    }

    /// Look up rows in a range. Only btree supports this.
    pub fn lookup_range(
        &self,
        column: &str,
        lo:     &ScalarValue,
        hi:     &ScalarValue,
    ) -> Option<Vec<usize>> {
        self.btree.get(column).map(|idx| idx.lookup_range(lo, hi))
    }

    pub fn has_index(&self, column: &str) -> bool {
        self.btree.contains_key(column) || self.hash.contains_key(column)
    }

    pub fn index_count(&self) -> usize {
        self.btree.len() + self.hash.len()
    }
}

impl Default for IndexRegistry {
    fn default() -> Self { Self::new() }
}

// ─── Multi-batch table index ─────────────────────────────────────────────────

/// An index that maps values to (batch_index, row_index) pairs, suitable for
/// tables that span multiple RecordBatches.
pub struct TableIndex {
    column_name: String,
    btree:       Option<BTreeMap<IndexKey, Vec<(usize, usize)>>>,
    hash:        Option<HashMap<IndexKey, Vec<(usize, usize)>>>,
    use_hash:    bool,
}

impl TableIndex {
    pub fn build_btree(
        column_name: &str,
        table: &super::table::Table,
    ) -> FalconResult<Self> {
        let col_idx = table.schema.column_index(column_name)?;
        let mut tree: BTreeMap<IndexKey, Vec<(usize, usize)>> = BTreeMap::new();
        for (batch_idx, batch) in table.batches.iter().enumerate() {
            let col = batch.column(col_idx);
            for row in 0..col.len() {
                let key = IndexKey::from_scalar(&col.get(row)?);
                tree.entry(key).or_default().push((batch_idx, row));
            }
        }
        Ok(Self {
            column_name: column_name.to_string(),
            btree: Some(tree),
            hash: None,
            use_hash: false,
        })
    }

    pub fn build_hash(
        column_name: &str,
        table: &super::table::Table,
    ) -> FalconResult<Self> {
        let col_idx = table.schema.column_index(column_name)?;
        let mut map: HashMap<IndexKey, Vec<(usize, usize)>> = HashMap::new();
        for (batch_idx, batch) in table.batches.iter().enumerate() {
            let col = batch.column(col_idx);
            for row in 0..col.len() {
                let key = IndexKey::from_scalar(&col.get(row)?);
                map.entry(key).or_default().push((batch_idx, row));
            }
        }
        Ok(Self {
            column_name: column_name.to_string(),
            btree: None,
            hash: Some(map),
            use_hash: true,
        })
    }

    pub fn column_name(&self) -> &str { &self.column_name }

    /// Return all (batch_idx, row_idx) pairs for rows where column = value.
    pub fn lookup_eq(&self, value: &ScalarValue) -> Vec<(usize, usize)> {
        let key = IndexKey::from_scalar(value);
        if self.use_hash {
            self.hash.as_ref()
                .and_then(|m| m.get(&key))
                .cloned()
                .unwrap_or_default()
        } else {
            self.btree.as_ref()
                .and_then(|m| m.get(&key))
                .cloned()
                .unwrap_or_default()
        }
    }

    /// Return all (batch_idx, row_idx) pairs where lo <= column <= hi.
    pub fn lookup_range(&self, lo: &ScalarValue, hi: &ScalarValue) -> Vec<(usize, usize)> {
        let lo_key = IndexKey::from_scalar(lo);
        let hi_key = IndexKey::from_scalar(hi);
        match &self.btree {
            Some(tree) => tree.range(lo_key..=hi_key)
                .flat_map(|(_, rows)| rows.iter().copied())
                .collect(),
            None => Vec::new(),
        }
    }
}

/// Per-table registry of multi-batch indices, keyed by column name.
pub struct MultiTableRegistry {
    indices: HashMap<String, TableIndex>,
}

impl MultiTableRegistry {
    pub fn new() -> Self { Self { indices: HashMap::new() } }

    pub fn add(&mut self, idx: TableIndex) {
        self.indices.insert(idx.column_name.clone(), idx);
    }

    pub fn remove(&mut self, column_name: &str) {
        self.indices.remove(column_name);
    }

    pub fn get(&self, column_name: &str) -> Option<&TableIndex> {
        self.indices.get(column_name)
    }

    pub fn has_index(&self, column_name: &str) -> bool {
        self.indices.contains_key(column_name)
    }

    pub fn is_empty(&self) -> bool { self.indices.is_empty() }

    pub fn index_names(&self) -> Vec<&str> {
        self.indices.values().map(|i| i.column_name.as_str()).collect()
    }
}

impl Default for MultiTableRegistry {
    fn default() -> Self { Self::new() }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::column::ColumnBuilder;

    fn int_col() -> Column {
        ColumnBuilder::int64(vec![Some(10), Some(20), Some(10), Some(30), None])
    }

    fn str_col() -> Column {
        ColumnBuilder::utf8_from_str(vec![
            Some("alice"), Some("bob"), Some("alice"), Some("carol"), None,
        ])
    }

    #[test]
    fn btree_lookup_eq_int() {
        let col = int_col();
        let idx = BTreeIndex::build("id", &col).unwrap();
        let rows = idx.lookup_eq(&ScalarValue::Int64(Some(10)));
        assert_eq!(rows, vec![0, 2]);
    }

    #[test]
    fn btree_lookup_eq_missing() {
        let col  = int_col();
        let idx  = BTreeIndex::build("id", &col).unwrap();
        let rows = idx.lookup_eq(&ScalarValue::Int64(Some(99)));
        assert!(rows.is_empty());
    }

    #[test]
    fn btree_range_query() {
        let col  = int_col();
        let idx  = BTreeIndex::build("id", &col).unwrap();
        let rows = idx.lookup_range(
            &ScalarValue::Int64(Some(10)),
            &ScalarValue::Int64(Some(20)),
        );
        let mut sorted = rows.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn btree_lookup_str() {
        let col  = str_col();
        let idx  = BTreeIndex::build("name", &col).unwrap();
        let rows = idx.lookup_eq(&ScalarValue::Utf8(Some("alice".into())));
        assert_eq!(rows, vec![0, 2]);
    }

    #[test]
    fn hash_lookup_eq() {
        let col  = int_col();
        let idx  = HashIndex::build("id", &col).unwrap();
        let rows = idx.lookup_eq(&ScalarValue::Int64(Some(20)));
        assert_eq!(rows, vec![1]);
    }

    #[test]
    fn registry_prefers_hash_for_eq() {
        let col  = int_col();
        let mut reg = IndexRegistry::new();
        reg.add_hash("id", &col).unwrap();
        reg.add_btree("id", &col).unwrap();
        let rows = reg.lookup_eq("id", &ScalarValue::Int64(Some(10))).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn registry_range_uses_btree() {
        let col  = int_col();
        let mut reg = IndexRegistry::new();
        reg.add_btree("id", &col).unwrap();
        let rows = reg.lookup_range(
            "id",
            &ScalarValue::Int64(Some(10)),
            &ScalarValue::Int64(Some(20)),
        ).unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn null_values_are_indexed() {
        let col  = int_col();
        let idx  = BTreeIndex::build("id", &col).unwrap();
        let rows = idx.lookup_eq(&ScalarValue::Int64(None));
        assert_eq!(rows, vec![4]);
    }

    #[test]
    fn float_ordering_correct() {
        let col = ColumnBuilder::float64(vec![
            Some(-1.0), Some(0.0), Some(1.0), Some(f64::INFINITY),
        ]);
        let idx  = BTreeIndex::build("x", &col).unwrap();
        let rows = idx.lookup_range(
            &ScalarValue::Float64(Some(-1.0)),
            &ScalarValue::Float64(Some(1.0)),
        );
        assert_eq!(rows.len(), 3);
    }
}
