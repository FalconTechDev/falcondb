//! Arithmetic on integer columns stays integral.
//!
//! The defect this pins: every binary operator went through `as_f64`, so
//! `n * 1` on an `Int64` column came back as a `Float64` -- and above 2^53 an
//! `f64` cannot hold consecutive integers, so it came back as a *different
//! number*. Multiplying by one is not supposed to lose anything.

use falcondb::db::Database;
use falcondb::storage::column::ColumnBuilder;
use falcondb::storage::schema::{Field, Schema};
use falcondb::storage::table::RecordBatch;
use falcondb::storage::types::{DataType, ScalarValue};

/// 2^53 + 1: the smallest odd integer an `f64` cannot represent.
const BEYOND_F64: i64 = 9_007_199_254_740_993;

fn db() -> Database {
    let batch = RecordBatch::new(
        Schema::new(vec![
            Field::required("n", DataType::Int64),
            Field::required("small", DataType::Int32),
            Field::required("f", DataType::Float64),
        ]),
        vec![
            ColumnBuilder::int64(vec![Some(BEYOND_F64), Some(1), Some(-5)]),
            ColumnBuilder::int32(vec![Some(7), Some(1), Some(-5)]),
            ColumnBuilder::float64(vec![Some(2.5), Some(1.0), Some(-5.0)]),
        ],
    )
    .unwrap();
    let db = Database::in_memory();
    db.create_table_from_batches("t", vec![batch]).unwrap();
    db
}

fn first(db: &Database, sql: &str) -> ScalarValue {
    db.sql(sql).unwrap().batches[0].column(0).get(0).unwrap()
}

/// The value survives, exactly, at a magnitude where `f64` would round it.
#[test]
fn integer_arithmetic_keeps_every_bit() {
    let db = db();
    for (sql, want) in [
        ("SELECT n * 1 FROM t", BEYOND_F64),
        ("SELECT n + 0 FROM t", BEYOND_F64),
        ("SELECT n - 0 FROM t", BEYOND_F64),
        ("SELECT n + 1 FROM t", BEYOND_F64 + 1),
        ("SELECT n - 1 FROM t", BEYOND_F64 - 1),
        ("SELECT 1 * n FROM t", BEYOND_F64),
        ("SELECT n * 2 - n FROM t", BEYOND_F64),
    ] {
        assert_eq!(
            first(&db, sql),
            ScalarValue::Int64(Some(want)),
            "{sql}",
        );
    }
}

/// `Int32` widens to `Int64` rather than to `Float64`.
#[test]
fn small_integers_widen_to_integers() {
    let db = db();
    assert_eq!(first(&db, "SELECT small * 3 FROM t"), ScalarValue::Int64(Some(21)));
    assert_eq!(first(&db, "SELECT small + small FROM t"), ScalarValue::Int64(Some(14)));
}

/// A float anywhere in the expression makes the result a float, as before.
#[test]
fn mixing_in_a_float_still_gives_a_float() {
    let db = db();
    match first(&db, "SELECT small * f FROM t") {
        ScalarValue::Float64(Some(v)) => assert_eq!(v, 17.5),
        other => panic!("expected Float64, got {other:?}"),
    }
}

/// Division stays on the float path.
///
/// This is a deliberate deviation from integer division, not an oversight: it is
/// what DuckDB and SQLite return for `/` on integers, and what this engine has
/// always returned. Pinned so a later change to it is a decision rather than an
/// accident.
#[test]
fn division_is_float_division() {
    let db = db();
    match first(&db, "SELECT small / 2 FROM t") {
        ScalarValue::Float64(Some(v)) => assert_eq!(v, 3.5),
        other => panic!("expected Float64, got {other:?}"),
    }
}

/// Overflow is an error, because a wrapped product is a wrong answer that looks
/// like an answer.
#[test]
fn overflow_is_refused_rather_than_wrapped() {
    let batch = RecordBatch::new(
        Schema::new(vec![Field::required("n", DataType::Int64)]),
        vec![ColumnBuilder::int64(vec![Some(i64::MAX)])],
    )
    .unwrap();
    let db = Database::in_memory();
    db.create_table_from_batches("t", vec![batch]).unwrap();

    assert!(db.sql("SELECT n + 1 FROM t").is_err(), "i64::MAX + 1");
    assert!(db.sql("SELECT n * 2 FROM t").is_err(), "i64::MAX * 2");
    // The same expression one step below the edge is fine.
    assert_eq!(first(&db, "SELECT n - 1 FROM t"), ScalarValue::Int64(Some(i64::MAX - 1)));
}

/// WHERE still works on integer arithmetic, which is the other place the change
/// could have broken something.
#[test]
fn filters_over_integer_arithmetic_still_select() {
    let db = db();
    let rows: usize = db
        .sql("SELECT n FROM t WHERE n * 1 > 0")
        .unwrap()
        .batches
        .iter()
        .map(|b| b.row_count())
        .sum();
    assert_eq!(rows, 2, "two of the three values are positive");
}
