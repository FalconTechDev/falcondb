//! A fast, non-cryptographic hasher for internal query state.
//!
//! `std`'s default `RandomState` uses SipHash-1-3, which is DoS-resistant because
//! hash maps are often keyed by untrusted input. Aggregation and join state is
//! never exposed that way — the keys come from data we already own — so we can
//! trade that resistance for speed.
//!
//! This is the FxHash algorithm used by `rustc` itself: a multiply-xor-rotate over
//! 8-byte words. On the short string and integer keys that dominate `GROUP BY` it
//! is several times cheaper than SipHash, and `GROUP BY` hashes once per row.

use std::{
    collections::{HashMap, HashSet},
    hash::{BuildHasherDefault, Hasher},
};

/// Multiplier from rustc's FxHasher — chosen so every input bit influences the
/// high bits of the product.
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add_to_hash(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            // `chunks_exact(8)` guarantees the length, so the conversion cannot fail.
            self.add_to_hash(u64::from_ne_bytes(chunk.try_into().unwrap()));
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut buf = [0u8; 8];
            buf[..remainder.len()].copy_from_slice(remainder);
            // Fold in the length so "ab" and "ab\0" do not collide.
            self.add_to_hash(u64::from_ne_bytes(buf) ^ (remainder.len() as u64));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) { self.add_to_hash(i as u64); }
    #[inline]
    fn write_u16(&mut self, i: u16) { self.add_to_hash(i as u64); }
    #[inline]
    fn write_u32(&mut self, i: u32) { self.add_to_hash(i as u64); }
    #[inline]
    fn write_u64(&mut self, i: u64) { self.add_to_hash(i); }
    #[inline]
    fn write_usize(&mut self, i: usize) { self.add_to_hash(i as u64); }
    #[inline]
    fn write_i64(&mut self, i: i64) { self.add_to_hash(i as u64); }
    #[inline]
    fn write_i32(&mut self, i: i32) { self.add_to_hash(i as u32 as u64); }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// Drop-in replacement for `HashMap` in query-internal state.
pub type FastMap<K, V> = HashMap<K, V, FxBuildHasher>;

/// Drop-in replacement for `HashSet` in query-internal state.
pub type FastSet<T> = HashSet<T, FxBuildHasher>;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut h = FxHasher::default();
        value.hash(&mut h);
        h.finish()
    }

    #[test]
    fn same_input_same_hash() {
        assert_eq!(hash_of(&"engineering"), hash_of(&"engineering"));
        assert_eq!(hash_of(&42i64), hash_of(&42i64));
    }

    #[test]
    fn different_input_different_hash() {
        assert_ne!(hash_of(&"eng"), hash_of(&"hr"));
        assert_ne!(hash_of(&1i64), hash_of(&2i64));
    }

    #[test]
    fn length_is_folded_in() {
        // Without folding the length, trailing-zero padding would collide these.
        assert_ne!(hash_of(&"ab"), hash_of(&"ab\0"));
    }

    #[test]
    fn works_as_hashmap_hasher() {
        let mut map: FastMap<String, u32> = FastMap::default();
        map.insert("eng".into(), 1);
        map.insert("hr".into(), 2);
        assert_eq!(map.get("eng"), Some(&1));
        assert_eq!(map.get("hr"), Some(&2));
        assert_eq!(map.get("missing"), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn distributes_short_keys() {
        // A weak hash would bucket sequential integers together; check we get a
        // wide spread of distinct values.
        let hashes: FastSet<u64> = (0..1000i64).map(|i| hash_of(&i)).collect();
        assert_eq!(hashes.len(), 1000, "sequential keys must not collide");
    }

    #[test]
    fn long_keys_use_all_bytes() {
        let a = "a".repeat(64);
        let mut b = a.clone();
        b.push('x');
        assert_ne!(hash_of(&a), hash_of(&b));

        // A change in the final chunk must be visible.
        let c = format!("{}z", "a".repeat(63));
        assert_ne!(hash_of(&a), hash_of(&c));
    }
}
