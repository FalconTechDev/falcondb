//! Where does ORDER BY actually spend its time?
//!
//! `execute_sort` merges every batch into one before sorting, then gathers the
//! result — two full materialisations of the table. Removing the merge is only
//! worth doing if the merge is a real share of the total, and eliminating it would
//! cost a global-index -> (batch, offset) lookup per output row, so it can easily
//! be a net loss. Measure first.
//!
//! Run with: cargo test --release --test diag_sort_breakdown -- --nocapture

use std::time::Instant;

use falcondb::{
    query::{
        expr::Expr,
        sort::{execute_sort, SortExpr},
    },
    storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::RecordBatch,
        types::DataType,
    },
};

const DEPTS: [&str; 5] = ["eng", "sales", "hr", "ops", "legal"];

/// Same shape as the benchmark table, split into `batch_rows`-sized batches to
/// match what a Parquet load produces.
fn build_batches(rows: usize, batch_rows: usize) -> Vec<RecordBatch> {
    let schema = Schema::new(vec![
        Field::required("id", DataType::Int64),
        Field::required("dept", DataType::Utf8),
        Field::required("score", DataType::Float64),
    ]);
    let mut out = Vec::new();
    let mut start = 0;
    while start < rows {
        let n = batch_rows.min(rows - start);
        out.push(
            RecordBatch::new(
                schema.clone(),
                vec![
                    ColumnBuilder::int64((0..n).map(|i| Some((start + i) as i64)).collect()),
                    ColumnBuilder::utf8_from_str(
                        (0..n).map(|i| Some(DEPTS[(start + i) % 5])).collect(),
                    ),
                    ColumnBuilder::float64(
                        (0..n).map(|i| Some(((start + i) % 100) as f64 + 0.5)).collect(),
                    ),
                ],
            )
            .unwrap(),
        );
        start += n;
    }
    out
}

#[test]
fn sort_time_against_merge_time() {
    const ROWS: usize = 100_000;
    // What the Parquet reader produces. This said 1024 for a long time, which is
    // what the reader used to produce; at 100k rows that is 98 batches against
    // the 2 the benchmark actually has, and the merge it measured was not the
    // merge the benchmark pays.
    const BATCH: usize = 65_536;
    let batches = build_batches(ROWS, BATCH);

    // `merge_batches` short-circuits to a clone when handed a single batch, so
    // sorting one pre-merged batch does everything except the merge. The gap
    // between the two is the merge, measured through the public API and without
    // guessing at what an equivalent copy would cost.
    let single = build_batches(ROWS, ROWS);
    assert_eq!(single.len(), 1);

    let keys = vec![SortExpr::desc(Expr::Column("score".into()))];
    let time = |input: &[RecordBatch]| {
        let t = Instant::now();
        let out = execute_sort(input, &keys).unwrap();
        assert_eq!(out.row_count(), ROWS);
        t.elapsed()
    };

    // Warm up, then take the best of several runs — the minimum is the least
    // noisy estimator here, the same reason criterion reports one.
    for _ in 0..3 {
        time(&batches);
        time(&single);
    }
    let many = (0..9).map(|_| time(&batches)).min().unwrap();
    let one = (0..9).map(|_| time(&single)).min().unwrap();

    println!("batches           = {}", batches.len());
    println!("sort of {:>3} batches = {:?}", batches.len(), many);
    println!("sort of   1 batch   = {:?}", one);
    let share = (many.as_secs_f64() - one.as_secs_f64()) / many.as_secs_f64() * 100.0;
    println!("merge share        = {share:.1}%");
}
