// Bare aggregates (`SELECT SUM(x)` with no GROUP BY): correctness and speed.
//
// This path used to build a `Vec<ScalarValue>` per batch and fold it serially,
// which measured 40x slower than DuckDB on the same query. It now reads the raw
// column slice and accumulates batches in parallel, so the tests here pin both
// the answers and the fact that parallel merging did not break them.

use std::time::Instant;

use falcondb::{
    db::Database,
    storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::{RecordBatch, Table},
        types::DataType,
    },
};

/// Table of `rows` values where score = (i % 100) + 0.5, split into many batches
/// so the parallel merge path is actually exercised.
fn db_with_scores(rows: usize, batch: usize) -> Database {
    let schema = Schema::new(vec![
        Field::required("id", DataType::Int64),
        Field::nullable("score", DataType::Float64),
    ]);
    let mut table = Table::new("t", schema.clone());
    let mut i = 0usize;
    while i < rows {
        let n = batch.min(rows - i);
        let b = RecordBatch::new(
            schema.clone(),
            vec![
                ColumnBuilder::int64((i..i + n).map(|k| Some(k as i64)).collect()),
                ColumnBuilder::float64((i..i + n).map(|k| Some((k % 100) as f64 + 0.5)).collect()),
            ],
        )
        .unwrap();
        table.append_batch(b).unwrap();
        i += n;
    }
    let db = Database::in_memory();
    db.create_table_from_batches("t", table.batches).unwrap();
    db
}

fn scalar(db: &Database, sql: &str) -> f64 {
    db.sql(sql).unwrap().to_rows()[0][0].parse::<f64>().unwrap()
}

#[test]
fn sum_is_correct_across_many_batches() {
    let db = db_with_scores(100_000, 1024);
    // Each block of 100 rows sums to 5000.
    assert_eq!(scalar(&db, "SELECT SUM(score) AS s FROM t"), 1000.0 * 5000.0);
}

#[test]
fn count_min_max_avg_are_correct() {
    let db = db_with_scores(100_000, 1024);
    assert_eq!(scalar(&db, "SELECT COUNT(*) AS s FROM t"), 100_000.0);
    assert_eq!(scalar(&db, "SELECT MIN(score) AS s FROM t"), 0.5);
    assert_eq!(scalar(&db, "SELECT MAX(score) AS s FROM t"), 99.5);
    // Values run 0.5 .. 99.5, so the mean is (0.5 + 99.5) / 2.
    assert_eq!(scalar(&db, "SELECT AVG(score) AS s FROM t"), 50.0);
}

#[test]
fn aggregates_skip_nulls() {
    let schema = Schema::new(vec![Field::nullable("v", DataType::Float64)]);
    let mut table = Table::new("t", schema.clone());
    // Two batches, each with a NULL, so the merge path sees partial states.
    for chunk in [vec![Some(1.0), None, Some(3.0)], vec![Some(5.0), None]] {
        table
            .append_batch(RecordBatch::new(schema.clone(), vec![ColumnBuilder::float64(chunk)]).unwrap())
            .unwrap();
    }
    let db = Database::in_memory();
    db.create_table_from_batches("t", table.batches).unwrap();

    assert_eq!(scalar(&db, "SELECT SUM(v) AS s FROM t"), 9.0);
    assert_eq!(scalar(&db, "SELECT MIN(v) AS s FROM t"), 1.0);
    assert_eq!(scalar(&db, "SELECT MAX(v) AS s FROM t"), 5.0);
    assert_eq!(scalar(&db, "SELECT AVG(v) AS s FROM t"), 3.0);
    // COUNT(v) counts non-NULL values; COUNT(*) counts rows.
    assert_eq!(scalar(&db, "SELECT COUNT(v) AS s FROM t"), 3.0);
    assert_eq!(scalar(&db, "SELECT COUNT(*) AS s FROM t"), 5.0);
}

#[test]
fn all_null_column_aggregates_to_null() {
    let schema = Schema::new(vec![Field::nullable("v", DataType::Float64)]);
    let mut table = Table::new("t", schema.clone());
    table
        .append_batch(
            RecordBatch::new(schema.clone(), vec![ColumnBuilder::float64(vec![None, None])]).unwrap(),
        )
        .unwrap();
    let db = Database::in_memory();
    db.create_table_from_batches("t", table.batches).unwrap();

    assert_eq!(db.sql("SELECT SUM(v) AS s FROM t").unwrap().to_rows()[0][0], "NULL");
    assert_eq!(db.sql("SELECT MIN(v) AS s FROM t").unwrap().to_rows()[0][0], "NULL");
}

#[test]
fn computed_input_still_works() {
    // Not a bare column reference, so this takes the general ScalarValue path.
    let db = db_with_scores(1_000, 128);
    assert_eq!(scalar(&db, "SELECT SUM(score * 2) AS s FROM t"), 10.0 * 5000.0 * 2.0);
}

#[test]
fn sum_over_a_million_rows_is_fast_enough() {
    let db = db_with_scores(1_000_000, 1024);

    // Warm up, then time one pass.
    let _ = db.sql("SELECT SUM(score) AS s FROM t").unwrap();
    let t0 = Instant::now();
    let got = scalar(&db, "SELECT SUM(score) AS s FROM t");
    let elapsed = t0.elapsed();

    assert_eq!(got, 10_000.0 * 5000.0);
    println!("SUM over 1M rows: {elapsed:?}");

    // The serial ScalarValue version took ~22ms here. This is a generous ceiling:
    // it is not a performance target, only a guard against silently regressing
    // back to boxing every value.
    assert!(
        elapsed.as_millis() < 15,
        "SUM over 1M rows took {elapsed:?}; the typed path has probably regressed"
    );
}
