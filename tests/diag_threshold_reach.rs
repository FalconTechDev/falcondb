//! Which per-batch thresholds can actually fire, and under which way of building
//! a table?
//!
//! Every parallel kernel in the engine is gated on a *per-batch* size, but batch
//! size is decided by how the table was built, not by how many rows it holds. A
//! Parquet load yields 1024-row batches, so a threshold of 65_536 is unreachable
//! there no matter how large the table. Before deleting or lowering any of them,
//! measure reachability on every construction path.
//!
//! Run with: cargo test --test diag_threshold_reach -- --nocapture

use falcondb::{
    db::Database,
    storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::{RecordBatch, Table},
        types::DataType,
    },
};

/// Mirrors the constants the engine gates on, in rows.
const THRESHOLDS: [(&str, usize); 3] = [
    ("PAR_MASK_WORDS (expr.rs)", 4096 * 64),
    ("PAR_SUM_THRESHOLD (aggregate.rs)", 65_536),
    ("PAR_INTRA_GB_ROWS (aggregate.rs)", 524_288),
];

fn schema() -> Schema {
    Schema::new(vec![
        Field::required("id", DataType::Int64),
        Field::required("score", DataType::Float64),
    ])
}

fn make_batch(start: usize, n: usize) -> RecordBatch {
    RecordBatch::new(
        schema(),
        vec![
            ColumnBuilder::int64((0..n).map(|i| Some((start + i) as i64)).collect()),
            ColumnBuilder::float64((0..n).map(|i| Some((start + i) as f64)).collect()),
        ],
    )
    .unwrap()
}

fn report(label: &str, sizes: &[usize]) {
    let largest = sizes.iter().copied().max().unwrap_or(0);
    let total: usize = sizes.iter().sum();
    println!("\n{label}");
    println!("  rows={total}  batches={}  largest batch={largest}", sizes.len());
    for (name, need) in THRESHOLDS {
        let fires = largest >= need;
        println!("    {name}: needs {need} rows in one batch -> {}", if fires { "FIRES" } else { "never fires" });
    }
}

#[test]
fn threshold_reachability_by_construction_path() {
    const ROWS: usize = 1_000_000;

    // Path 1: Parquet load — what the benchmark and any file-backed table uses.
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.parquet");
        let mut src = Table::new("t", schema());
        src.append_batch(make_batch(0, ROWS)).unwrap();
        falcondb::storage::parquet_io::write_parquet(&src.batches, &path).unwrap();

        let db = Database::in_memory();
        db.load_parquet(&path, "t").unwrap();
        let t = db.get_table("t").unwrap();
        let sizes: Vec<usize> = t.batches.iter().map(|b| b.row_count()).collect();
        report("Parquet load", &sizes);
    }

    // Path 2: built in memory through append_batch, which re-chunks to
    // Table::DEFAULT_BATCH_SIZE.
    {
        let mut t = Table::new("t", schema());
        t.append_batch(make_batch(0, ROWS)).unwrap();
        let sizes: Vec<usize> = t.batches.iter().map(|b| b.row_count()).collect();
        report("append_batch (DEFAULT_BATCH_SIZE)", &sizes);
    }

    // Path 3: one batch handed in whole, as an operator that merges produces.
    {
        let sizes = vec![make_batch(0, ROWS).row_count()];
        report("single unsplit batch (e.g. post-merge in sort)", &sizes);
    }
}
