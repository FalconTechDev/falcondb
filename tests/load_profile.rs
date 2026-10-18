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

/// The phases of a load, timed separately.
///
/// The by-column numbers above add up to much less than what `Database::load_parquet`
/// reports, so something after the read costs as much as the read. This splits
/// the two instead of guessing: reading the file into batches, then handing those
/// batches to a table.
#[test]
#[ignore]
fn load_phases_100m() {
    use falcondb::storage::table::Table;

    let path = data_dir().join("bench_100000000.parquet");
    if !path.exists() {
        println!("no cached file -- run one_shot_large first");
        return;
    }

    let t = Instant::now();
    let batches = falcondb::storage::parquet_io::read_parquet(&path, "bench").unwrap();
    let read = t.elapsed();
    let n_batches = batches.len();
    let rows: usize = batches.iter().map(|b| b.row_count()).sum();

    let schema = batches[0].schema.clone();
    let mut table = Table::new("bench", schema);
    let t = Instant::now();
    table.append_batches(batches.clone()).unwrap();
    let append = t.elapsed();

    // The loop this replaced, measured against it in the same process on the same
    // batches, because the whole-load timing is too noisy to attribute half a
    // second to one of its phases.
    let mut one_at_a_time = Table::new("bench", table.schema.clone());
    let t = Instant::now();
    for b in batches {
        one_at_a_time.append_batch(b).unwrap();
    }
    let serial = t.elapsed();

    println!("
=== 100M load phases ===");
    println!("  read_parquet      {read:>10.2?}");
    println!("  append_batches    {append:>10.2?}");
    println!("  append_batch loop {serial:>10.2?}");
    println!("  {n_batches} batches, {rows} rows");
}
