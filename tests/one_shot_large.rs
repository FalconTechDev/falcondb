//! One single run of each query at 10M and 100M rows, FalconDB only.
//!
//! This is the iteration harness for the two shapes that do not scale past 1M.
//! It is deliberately *not* Criterion and deliberately does not run the rival
//! engines:
//!
//!   - **One run, no median.** At these sizes a single reading is the realistic
//!     measure: nobody runs a hundred-million-row query a hundred times, and the
//!     run a user waits for is the first. The cost is that one reading cannot
//!     separate the query from whatever the machine was doing that second, so a
//!     difference under about 20% is not resolved by it.
//!   - **FalconDB only.** When the question is "did my change help", the rivals
//!     add minutes per iteration and answer nothing. Use
//!     `tests/one_shot_vs_rivals.rs` when the question is "where do we stand".
//!   - **The data file is cached.** Writing the 100M-row Parquet took 32 s, which
//!     dominated everything else. It is written once to `FALCON_BENCH_DATA` (or a
//!     default beside the target directory) and reused, so a second run starts
//!     measuring immediately. Delete the file to regenerate it.
//!
//! Run with:
//!   cargo test --release --test one_shot_large -- --ignored --nocapture
//!
//! 100M rows needs about 3 GB resident and a 414 MB file on disk.

use falcondb::db::Database;
use falcondb::storage::column::ColumnBuilder;
use falcondb::storage::schema::{Field, Schema};
use falcondb::storage::table::{RecordBatch, Table};
use falcondb::storage::types::DataType;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const DEPTS: [&str; 5] = ["engineering", "sales", "hr", "ops", "legal"];

/// The queries. Each is run once.
///
/// The first two are the shapes that do not scale; the rest are here so a change
/// that helps one and hurts another cannot hide.
const QUERIES: [(&str, &str); 6] = [
    ("COUNT(*) WHERE score > 50",  "SELECT COUNT(*) FROM bench WHERE score > 50"),
    ("ORDER BY score LIMIT 10",    "SELECT score FROM bench ORDER BY score DESC LIMIT 10"),
    ("SUM(score)",                 "SELECT SUM(score) FROM bench"),
    ("SUM(id) integer",            "SELECT SUM(id) FROM bench"),
    ("GROUP BY dept",              "SELECT dept, COUNT(*), AVG(score) FROM bench GROUP BY dept"),
    ("COUNT(DISTINCT dept)",       "SELECT COUNT(DISTINCT dept) FROM bench"),
];

/// Where the cached Parquet files live.
fn data_dir() -> PathBuf {
    std::env::var("FALCON_BENCH_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("CARGO_TARGET_DIR")
                .map(|t| PathBuf::from(t).join("bench-data"))
                .unwrap_or_else(|_| PathBuf::from("target/bench-data"))
        })
}

/// Build the file if it is not already there.
///
/// Written in chunks so the whole table never exists twice in memory: at 100M
/// rows a single batch is about 2 GB of `Vec`s before the writer sees any of it.
fn ensure_data(rows: usize) -> (PathBuf, Option<Duration>) {
    let dir = data_dir();
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join(format!("bench_{rows}.parquet"));
    if path.exists() {
        return (path, None);
    }

    const CHUNK: usize = 2_000_000;
    let schema = Schema::new(vec![
        Field::required("id", DataType::Int64),
        Field::required("dept", DataType::Utf8),
        Field::required("score", DataType::Float64),
    ]);
    let t = Instant::now();
    let mut table = Table::new("bench", schema.clone());
    let mut start = 0usize;
    while start < rows {
        let end = (start + CHUNK).min(rows);
        let batch = RecordBatch::new(
            schema.clone(),
            vec![
                ColumnBuilder::int64((start..end).map(|i| Some(i as i64)).collect()),
                ColumnBuilder::utf8_from_str((start..end).map(|i| Some(DEPTS[i % 5])).collect()),
                ColumnBuilder::float64((start..end).map(|i| Some((i % 100) as f64 + 0.5)).collect()),
            ],
        )
        .unwrap();
        table.append_batch(batch).unwrap();
        start = end;
    }
    falcondb::storage::parquet_io::write_parquet(&table.batches, &path).unwrap();
    (path, Some(t.elapsed()))
}

/// Print what the machine was doing, because this harness has no rival engine to
/// serve as a reference.
///
/// The four-engine suite can tell a slow machine from slow code: if Polars moved
/// with us, the machine moved. Running FalconDB alone iterates far faster and
/// gives that up -- and the first run after doing so proved why. Every query,
/// including the ones whose code was untouched, read 2-3.5x slower than the run
/// before: GROUP BY 5.92ms -> 20.4ms, COUNT(DISTINCT) 1.83ms -> 6.03ms, load
/// 448ms -> 1.01s. The battery had reached 14% and Windows was throttling.
/// Without this line the obvious reading was that the change under test had made
/// everything worse.
fn machine_state() -> String {
    // PowerShell rather than a crate: this is a diagnostic, not a dependency.
    let script = "Add-Type -AssemblyName System.Windows.Forms; \
                  $p=[System.Windows.Forms.SystemInformation]::PowerStatus; \
                  $c=(Get-CimInstance Win32_Processor).CurrentClockSpeed; \
                  Write-Output (\"{0} {1}% {2}MHz\" -f $p.PowerLineStatus, \
                  [math]::Round($p.BatteryLifePercent*100), $c)";
    match std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", script])
        .output()
    {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "unknown".to_string(),
    }
}


fn run_size(rows: usize) {
    let human = format!("{}M", rows / 1_000_000);
    println!("\n=== {human} rows, FalconDB only, one run each ===");
    let state = machine_state();
    println!("  machine: {state}");
    if state.starts_with("Offline") || state.contains(" 1") && state.contains("% ") {
        println!("  WARNING: battery or low charge. Numbers below are not comparable");
        println!("           with any other run -- see the comment on machine_state.");
    }

    let (path, written) = ensure_data(rows);
    match written {
        Some(d) => println!("  data generated in {d:.2?} (cached for next time)"),
        None => println!("  data reused from cache"),
    }

    let t = Instant::now();
    let db = Database::in_memory();
    db.load_parquet(&path, "bench").unwrap();
    println!("  load: {:.2?}", t.elapsed());
    println!();

    for (label, sql) in QUERIES {
        let t = Instant::now();
        let r = db.sql(sql);
        let elapsed = t.elapsed();
        match r {
            Ok(_) => println!("  {label:<28}{elapsed:>12.2?}"),
            Err(e) => println!("  {label:<28}{:>12}  ERROR: {e}", "-"),
        }
    }
}

#[test]
#[ignore]
fn one_shot_10m() {
    run_size(10_000_000);
}

/// Separate test so 10M can be run alone, and so a failure at 100M does not cost
/// the 10M numbers.
#[test]
#[ignore]
fn one_shot_100m() {
    run_size(100_000_000);
}
