//! Where does a join actually spend its time?
//!
//! Parallelising the probe over twelve cores took 1M-row INNER JOIN from 52.3 ms
//! to 48.9 ms — a 6.5% saving, so the probe is a small share of the total and the
//! remaining ~45 ms is somewhere else. Rather than guess, time the join through
//! the SQL layer against the join operator called directly: if they differ, the
//! cost is not in the join at all.
//!
//! Run with: cargo test --release --test diag_join_breakdown -- --nocapture

use std::time::Instant;

use falcondb::{
    db::Database,
    query::join::{execute_hash_join, JoinType},
    storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::{RecordBatch, Table},
        types::DataType,
    },
};

const DEPTS: [&str; 5] = ["eng", "sales", "hr", "ops", "legal"];

fn write_side(
    dir: &tempfile::TempDir,
    name: &str,
    rows: usize,
    key_stride: i64,
) -> std::path::PathBuf {
    let path = dir.path().join(format!("{name}.parquet"));
    let schema = Schema::new(vec![
        Field::required("k", DataType::Int64),
        Field::required("label", DataType::Utf8),
    ]);
    let batch = RecordBatch::new(
        schema.clone(),
        vec![
            ColumnBuilder::int64((0..rows).map(|i| Some(i as i64 * key_stride)).collect()),
            ColumnBuilder::utf8_from_str((0..rows).map(|i| Some(DEPTS[i % 5])).collect()),
        ],
    )
    .unwrap();
    let mut t = Table::new(name, schema);
    t.append_batch(batch).unwrap();
    falcondb::storage::parquet_io::write_parquet(&t.batches, &path).unwrap();
    path
}

fn best<F: FnMut() -> std::time::Duration>(mut f: F, runs: usize) -> std::time::Duration {
    (0..runs).map(|_| f()).min().unwrap()
}

#[test]
fn join_operator_against_the_sql_path() {
    const ROWS: usize = 1_000_000;
    let dir = tempfile::tempdir().unwrap();
    let left_path = write_side(&dir, "big", ROWS, 1);
    let right_path = write_side(&dir, "small", ROWS / 10, 10);

    let db = Database::in_memory();
    db.load_parquet(&left_path, "big").unwrap();
    db.load_parquet(&right_path, "small").unwrap();

    let big = db.get_table("big").unwrap();
    let small = db.get_table("small").unwrap();
    println!("left batches={}  right batches={}", big.batches.len(), small.batches.len());

    // Warm up both paths before timing either.
    for _ in 0..2 {
        db.sql("SELECT COUNT(*) AS n FROM big JOIN small ON big.k = small.k").unwrap();
        execute_hash_join(&big.batches, &small.batches, &["k"], &["k"], JoinType::Inner).unwrap();
    }

    let sql = best(
        || {
            let t = Instant::now();
            let r = db
                .sql("SELECT COUNT(*) AS n FROM big JOIN small ON big.k = small.k")
                .unwrap();
            let e = t.elapsed();
            assert_eq!(r.batches[0].row_count(), 1);
            e
        },
        5,
    );

    let direct = best(
        || {
            let t = Instant::now();
            let out =
                execute_hash_join(&big.batches, &small.batches, &["k"], &["k"], JoinType::Inner)
                    .unwrap();
            let e = t.elapsed();
            assert_eq!(out.row_count(), ROWS / 10);
            e
        },
        5,
    );

    // The join merges each side into one batch before building or probing.
    // `concat_batches` is exactly that step, so timing it splits the operator into
    // "flatten the inputs" and "everything else".
    let concat_left = best(
        || {
            let t = Instant::now();
            let b = RecordBatch::concat(&big.batches).unwrap();
            let e = t.elapsed();
            assert_eq!(b.row_count(), ROWS);
            e
        },
        5,
    );
    let concat_right = best(
        || {
            let t = Instant::now();
            let b = RecordBatch::concat(&small.batches).unwrap();
            let e = t.elapsed();
            assert_eq!(b.row_count(), ROWS / 10);
            e
        },
        5,
    );

    println!("SQL   SELECT COUNT(*) ... JOIN = {sql:?}");
    println!("direct execute_hash_join      = {direct:?}");
    println!(
        "overhead outside the join operator = {:.1}%",
        (sql.as_secs_f64() - direct.as_secs_f64()) / sql.as_secs_f64() * 100.0
    );
    // COUNT(*) reads no column, so projection pruning builds none of them. SUM over
    // a left column forces one to be gathered. The gap is what materialising a
    // column of the join output actually costs — as opposed to the bookkeeping of
    // recording 100k matches, which both queries pay.
    for _ in 0..2 {
        db.sql("SELECT SUM(k) AS s FROM big JOIN small ON big.k = small.k").unwrap();
    }
    let sql_sum = best(
        || {
            let t = Instant::now();
            db.sql("SELECT SUM(k) AS s FROM big JOIN small ON big.k = small.k").unwrap();
            t.elapsed()
        },
        5,
    );
    println!("  SQL COUNT(*) (no column built)         = {sql:?}");
    println!("  SQL SUM(k)   (one column gathered)     = {sql_sum:?}");

    // Same probe side, a build side 100x smaller. The probe still walks 1M rows,
    // so whatever falls away is the cost of building the hash table.
    let tiny_path = write_side(&dir, "tiny", 1_000, 1_000);
    db.load_parquet(&tiny_path, "tiny").unwrap();
    let tiny = db.get_table("tiny").unwrap();
    for _ in 0..2 {
        execute_hash_join(&big.batches, &tiny.batches, &["k"], &["k"], JoinType::Inner).unwrap();
    }
    let small_build = best(
        || {
            let t = Instant::now();
            execute_hash_join(&big.batches, &tiny.batches, &["k"], &["k"], JoinType::Inner)
                .unwrap();
            t.elapsed()
        },
        5,
    );
    // Same build side (100k keys) and same 1M probe, but the key ranges do not
    // overlap, so nothing matches and no output rows are assembled. The gap to the
    // matching join is what building the result costs.
    let miss_path = write_side(&dir, "miss", ROWS / 10, 10);
    db.load_parquet(&miss_path, "miss").unwrap();
    let miss = db.get_table("miss").unwrap();
    // Shift the probe side out of range by joining on a key column that cannot hit.
    let shifted = RecordBatch::new(
        Schema::new(vec![Field::required("k", DataType::Int64)]),
        vec![ColumnBuilder::int64(
            (0..ROWS).map(|i| Some(i as i64 + 100_000_000)).collect(),
        )],
    )
    .unwrap();
    let shifted = vec![shifted];
    for _ in 0..2 {
        execute_hash_join(&shifted, &miss.batches, &["k"], &["k"], JoinType::Inner).unwrap();
    }
    let no_output = best(
        || {
            let t = Instant::now();
            let out =
                execute_hash_join(&shifted, &miss.batches, &["k"], &["k"], JoinType::Inner)
                    .unwrap();
            let e = t.elapsed();
            assert_eq!(out.row_count(), 0);
            e
        },
        5,
    );
    println!("  100k build + 1M probe, zero matches   = {no_output:?}");
    println!("  same 1M probe, 1k-row build side      = {small_build:?}");
    println!("  concat left  (977 batches, 1M rows)   = {concat_left:?}");
    println!("  concat right ( 98 batches, 100k rows) = {concat_right:?}");
    let merges = concat_left + concat_right;
    println!(
        "  merges are {:.1}% of the join; build+probe+output = {:?}",
        merges.as_secs_f64() / direct.as_secs_f64() * 100.0,
        direct.saturating_sub(merges)
    );
}
