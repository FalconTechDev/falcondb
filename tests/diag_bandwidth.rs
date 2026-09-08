//! What is the floor? A roofline for this machine.
//!
//! "We are at the memory limit" is a claim, not a measurement, and the limit is
//! not one number: 1M f64s is 8 MB, which fits this CPU's 24 MB L3, so a scan of
//! it never touches DRAM at all. Comparing an engine's throughput against DRAM
//! bandwidth would flatter it; comparing against cache bandwidth is the honest
//! test.
//!
//! This measures the simplest possible kernel — sum a slice of f64 — at sizes
//! that sit in L2, in L3, and past L3, single-threaded and across cores. Whatever
//! the engine does per row, it cannot beat these numbers, so the gap between them
//! and a benchmark is the headroom that actually exists.
//!
//! Run pinned, the same way scripts/bench.ps1 runs the benchmarks, or the numbers
//! are not comparable to them.

use std::time::Instant;

use rayon::prelude::*;

/// Sum with independent accumulators — the same shape `query::aggregate` uses, so
/// this is a floor the engine could actually reach rather than a fantasy.
fn sum_lanes(v: &[f64]) -> f64 {
    const LANES: usize = 8;
    let mut acc = [0.0f64; LANES];
    let mut chunks = v.chunks_exact(LANES);
    for c in &mut chunks {
        for (a, &x) in acc.iter_mut().zip(c) {
            *a += x;
        }
    }
    let mut total: f64 = acc.iter().sum();
    for &x in chunks.remainder() {
        total += x;
    }
    total
}

fn best<F: FnMut() -> std::time::Duration>(mut f: F, runs: usize) -> std::time::Duration {
    (0..runs).map(|_| f()).min().unwrap()
}

fn gb_per_s(bytes: usize, d: std::time::Duration) -> f64 {
    bytes as f64 / d.as_secs_f64() / 1e9
}

#[test]
fn read_bandwidth_by_working_set() {
    // 128 KB fits L2; 8 MB is the benchmark's 1M-row column and fits L3;
    // 256 MB is well past L3 and must come from DRAM.
    let sizes = [16_384usize, 1_000_000, 32_000_000];

    println!("\n{:>12}  {:>9}  {:>14}  {:>14}", "elements", "size", "1 thread", "all threads");
    for n in sizes {
        let v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let bytes = n * std::mem::size_of::<f64>();

        // Warm the cache for the small cases; for the large one this does nothing,
        // which is the point.
        for _ in 0..3 {
            std::hint::black_box(sum_lanes(&v));
        }

        let serial = best(
            || {
                let t = Instant::now();
                std::hint::black_box(sum_lanes(&v));
                t.elapsed()
            },
            9,
        );

        let chunk = (n / rayon::current_num_threads().max(1)).max(4096);
        let parallel = best(
            || {
                let t = Instant::now();
                let s: f64 = v.par_chunks(chunk).map(sum_lanes).sum();
                std::hint::black_box(s);
                t.elapsed()
            },
            9,
        );

        println!(
            "{:>12}  {:>7.1} MB  {:>8.1} GB/s  {:>8.1} GB/s",
            n,
            bytes as f64 / 1e6,
            gb_per_s(bytes, serial),
            gb_per_s(bytes, parallel),
        );
        println!(
            "{:>12}  {:>9}  {:>8.1} µs  {:>10.1} µs",
            "", "", serial.as_secs_f64() * 1e6, parallel.as_secs_f64() * 1e6
        );
    }

    println!("\nthreads = {}", rayon::current_num_threads());
}
