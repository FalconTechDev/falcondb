//! Where the Parquet load time goes, by column.
//!
//! Loading is about 1.5x slower than DuckDB at both 10M and 100M -- stable with
//! scale, so a gap rather than a scaling defect. Before changing anything, this
//! asks which column costs what, by reading each one on its own through the
//! projection the reader already supports.
//!
//! The columns are deliberately different in kind: an Int64 and a Float64 are a
//! copy of fixed-width values, while the Utf8 column has to become a dictionary,
//! and the file holds five distinct strings repeated over every row.
//!
//!   cargo test --release --test load_profile -- --ignored --nocapture

use falcondb::storage::parquet_io::read_parquet_projected;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

fn data_dir() -> PathBuf {
    std::env::var("FALCON_BENCH_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("CARGO_TARGET_DIR")
                .map(|t| PathBuf::from(t).join("bench-data"))
                .unwrap_or_else(|_| PathBuf::from("target/bench-data"))
        })
}

fn profile(rows: usize) {
    let path = data_dir().join(format!("bench_{rows}.parquet"));
    if !path.exists() {
        println!("no cached file at {path:?} -- run one_shot_large first");
        return;
    }
    println!("\n=== {}M rows ===", rows / 1_000_000);

    for cols in [
        vec!["id"],
        vec!["score"],
        vec!["dept"],
        vec!["id", "score", "dept"],
    ] {
        let want: HashSet<String> = cols.iter().map(|c| c.to_string()).collect();
        let t = Instant::now();
        let batches = read_parquet_projected(&path, "bench", Some(&want)).unwrap();
        let elapsed = t.elapsed();
        let n: usize = batches.iter().map(|b| b.row_count()).sum();
        println!("  {:<20}{elapsed:>10.2?}   ({n} rows)", cols.join(","));
    }
}

#[test]
#[ignore]
fn load_profile_10m() {
    profile(10_000_000);
}

#[test]
#[ignore]
fn load_profile_100m() {
    profile(100_000_000);
}
