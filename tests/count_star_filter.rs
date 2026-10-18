//! `COUNT(*)` with a `WHERE` counts the mask rather than building the rows.
//!
//! The answer must be identical to what the materialising path gave. The way this
//! can go wrong is specific: the count is a population count over a packed
//! bitmask, so any bit set past the last row inflates it. Row counts that are not
//! multiples of 64 are therefore the cases that matter, and a row count of
//! exactly 64 or 128 would pass even with the bug.

use falcondb::db::Database;
use falcondb::storage::column::ColumnBuilder;
use falcondb::storage::schema::{Field, Schema};
use falcondb::storage::table::RecordBatch;
use falcondb::storage::types::{DataType, ScalarValue};

fn db_with(rows: usize) -> Database {
    let batch = RecordBatch::new(
        Schema::new(vec![
            Field::required("n", DataType::Int64),
            Field::nullable("maybe", DataType::Int64),
            Field::required("s", DataType::Utf8),
        ]),
        vec![
            ColumnBuilder::int64((0..rows as i64).map(Some).collect()),
            ColumnBuilder::int64(
                (0..rows).map(|i| if i % 3 == 0 { None } else { Some(i as i64) }).collect(),
            ),
            ColumnBuilder::utf8_from_str(
                (0..rows).map(|i| Some(["a", "b", "c"][i % 3])).collect(),
            ),
        ],
    )
    .unwrap();
    let db = Database::in_memory();
    db.create_table_from_batches("t", vec![batch]).unwrap();
    db
}

fn count(db: &Database, sql: &str) -> i64 {
    let r = db.sql(sql).unwrap();
    match r.batches[0].column(0).get(0).unwrap() {
        ScalarValue::Int64(Some(v)) => v,
        other => panic!("expected a count, got {other:?}"),
    }
}

/// The counted answer must equal the number of rows the same filter selects.
///
/// `SELECT n ... WHERE` still goes through the materialising path, so it is the
/// control: if the two disagree, the fast path is wrong.
#[test]
fn count_star_with_filter_matches_the_general_path() {
    // Row counts chosen around the 64-bit word boundary, because the failure mode
    // is a stray bit in the final partial word. 64 and 128 are included as the
    // cases that would pass even with that bug.
    for rows in [1usize, 63, 64, 65, 127, 128, 129, 1000, 4095, 10_000] {
        let db = db_with(rows);

        let cases = [
            "n >= 0",                 // every row
            "n < 0",                  // no rows
            "n >= 0 AND n < 100000",  // a prefix, or everything when smaller
            "n > 10",                 // all but a prefix
            "maybe IS NULL",          // a third, via the null mask
            "maybe IS NOT NULL",      // the complement of that
            "s = 'a'",                // text equality, dictionary path
            "s <> 'a'",               // its negation
            "s LIKE 'b%'",            // the LIKE mask
            "n IN (1, 2, 3)",         // the IN mask
            "n NOT IN (1, 2, 3)",     // complemented, where padding bits are set
                                      // and then masked off
            "n > 5 AND n < 20",       // AND of two masks
            "n < 2 OR n > 9995",      // OR of two masks
        ];

        for pred in cases {
            let counted = count(&db, &format!("SELECT COUNT(*) FROM t WHERE {pred}"));
            let materialised: i64 = db
                .sql(&format!("SELECT n FROM t WHERE {pred}"))
                .unwrap()
                .batches
                .iter()
                .map(|b| b.row_count() as i64)
                .sum();
            assert_eq!(
                counted, materialised,
                "{rows} rows, WHERE {pred}: counted {counted}, materialised {materialised}",
            );
        }
    }
}

/// Shapes the fast path must decline, because counting the mask would answer the
/// wrong question.
#[test]
fn count_star_fast_path_declines_what_it_cannot_answer() {
    let db = db_with(300);

    // COUNT(col) counts non-NULL values, not rows: 200 of the 300 `maybe` values
    // are non-NULL, and the filter keeps all 300.
    assert_eq!(count(&db, "SELECT COUNT(maybe) FROM t WHERE n >= 0"), 200);

    // A second aggregate alongside the count must still be computed.
    let r = db.sql("SELECT COUNT(*), SUM(n) FROM t WHERE n < 10").unwrap();
    assert_eq!(
        r.batches[0].column(0).get(0).unwrap(),
        ScalarValue::Int64(Some(10))
    );
    assert_eq!(
        r.batches[0].column(1).get(0).unwrap().as_f64(),
        Some(45.0),
        "SUM of 0..10"
    );

    // GROUP BY turns one number into several, so the scalar path cannot serve it.
    let r = db.sql("SELECT s, COUNT(*) FROM t WHERE n >= 0 GROUP BY s").unwrap();
    let groups: i64 = r.batches.iter().map(|b| b.row_count() as i64).sum();
    assert_eq!(groups, 3);

    // LIMIT applies to the result, and the result of a scalar aggregate is one row.
    assert_eq!(
        db.sql("SELECT COUNT(*) FROM t WHERE n >= 0 LIMIT 1")
            .unwrap()
            .batches[0]
            .row_count(),
        1
    );

    // HAVING filters the aggregated row, so the count can legitimately vanish.
    let kept: usize = db
        .sql("SELECT COUNT(*) c FROM t WHERE n >= 0 HAVING COUNT(*) > 1000")
        .unwrap()
        .batches
        .iter()
        .map(|b| b.row_count())
        .sum();
    assert_eq!(kept, 0, "300 rows is not more than 1000");
}
