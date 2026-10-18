//! One single run of each query at 10M and 100M rows, FalconDB against DuckDB.
//!
//! Deliberately *not* Criterion. The suite in `benches/` takes the median of 100
//! samples because at 10k-1M rows a single reading is dominated by noise. At these
//! sizes one run is the realistic measure: nobody runs a hundred-million-row query
//! a hundred times, and the first run is the one a user actually waits for.
//!
//! What that buys and costs:
//!   - No warm-up, no repetition, no median. Each number is one wall-clock reading.
//!   - A single reading cannot separate the query from whatever the machine was
//!     doing that second. Treat differences under about 20% as unresolved.
//!   - The FalconDB-against-DuckDB comparison within one run is the sound part:
//!     both read the same Parquet file in the same process moments apart.
//!
//! Loading is timed separately and excluded from the query times, as the benchmark
//! suite does, because loading is a different question from querying.
//!
//! Run with:
//!   cargo test --release --test one_shot_large -- --ignored --nocapture
//!
//! 100M rows needs roughly 6-8 GB of RAM across the two engines and about a
//! minute of Parquet writing before anything is measured.

use duckdb::Connection;
use falcondb::db::Database;
use falcondb::storage::column::ColumnBuilder;
use falcondb::storage::schema::{Field, Schema};
use falcondb::storage::table::{RecordBatch, Table};
use falcondb::storage::types::DataType;
use std::time::{Duration, Instant};

const DEPTS: [&str; 5] = ["engineering", "sales", "hr", "ops", "legal"];

/// The five queries, in the spelling both engines accept.
const QUERIES: [(&str, &str); 5] = [
    ("SUM(score)",            "SELECT SUM(score) FROM bench"),
    ("WHERE score > 50",      "SELECT COUNT(*) FROM bench WHERE score > 50"),
    ("GROUP BY dept",         "SELECT dept, COUNT(*), AVG(score) FROM bench GROUP BY dept"),
    ("ORDER BY score LIMIT 10", "SELECT score FROM bench ORDER BY score DESC LIMIT 10"),
    ("COUNT(DISTINCT dept)",  "SELECT COUNT(DISTINCT dept) FROM bench"),
];

/// Build the Parquet file in chunks so the whole table never exists twice in
/// memory at once. At 100M rows a single batch would be about 2 GB of `Vec`s
/// before the writer sees any of it.
fn write_parquet(rows: usize, path: &std::path::Path) -> Duration {
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
    falcondb::storage::parquet_io::write_parquet(&table.batches, path).unwrap();
    t.elapsed()
}

fn run_size(rows: usize, dir: &tempfile::TempDir) {
    let human = if rows >= 1_000_000 {
        format!("{}M", rows / 1_000_000)
    } else {
        format!("{rows}")
    };
    println!("\n=== {human} rows ===");

    let path = dir.path().join(format!("bench_{rows}.parquet"));
    let write = write_parquet(rows, &path);
    let size_mb = std::fs::metadata(&path).map(|m| m.len() / 1_048_576).unwrap_or(0);
    println!("  parquet written in {write:.2?} ({size_mb} MB on disk)");

    // Loading, timed but reported separately: a different question from querying.
    let t = Instant::now();
    let fdb = Database::in_memory();
    fdb.load_parquet(&path, "bench").unwrap();
    let fdb_load = t.elapsed();

    let t = Instant::now();
    let duck = Connection::open_in_memory().unwrap();
    duck.execute_batch(&format!(
        "CREATE TABLE bench AS SELECT * FROM read_parquet('{}')",
        path.display().to_string().replace('\\', "/")
    ))
    .unwrap();
    let duck_load = t.elapsed();

    println!("  load:  FalconDB {fdb_load:>10.2?}   DuckDB {duck_load:>10.2?}");
    println!();
    println!("  {:<26}{:>12}{:>12}{:>10}", "query", "FalconDB", "DuckDB", "ratio");

    for (label, sql) in QUERIES {
        // FalconDB: one run.
        let t = Instant::now();
        let r = fdb.sql(sql);
        let f_time = t.elapsed();
        let f_ok = r.is_ok();

        // DuckDB: one run. `query_row` forces the result to be produced rather
        // than left as a lazily-evaluated statement.
        let t = Instant::now();
        let d_ok = duck
            .prepare(sql)
            .and_then(|mut stmt| {
                let mut n = 0usize;
                let mut rows_iter = stmt.query([])?;
                while rows_iter.next()?.is_some() {
                    n += 1;
                }
                Ok(n)
            })
            .is_ok();
        let d_time = t.elapsed();

        let ratio = d_time.as_secs_f64() / f_time.as_secs_f64();
        let note = if !f_ok {
            "  FalconDB errored"
        } else if !d_ok {
            "  DuckDB errored"
        } else {
            ""
        };
        println!(
            "  {label:<26}{f_time:>12.2?}{d_time:>12.2?}{ratio:>9.1}x{note}",
        );
    }
}

/// 10M rows.
#[test]
#[ignore]
fn one_shot_10m() {
    let dir = tempfile::tempdir().unwrap();
    run_size(10_000_000, &dir);
}

/// 100M rows. Separate test so 10M can be run on its own, and so a failure at
/// 100M does not cost the 10M numbers.
#[test]
#[ignore]
fn one_shot_100m() {
    let dir = tempfile::tempdir().unwrap();
    run_size(100_000_000, &dir);
}
