//! A small `LIMIT` is answered by a bounded heap rather than by a pair per row.
//!
//! The thing to pin is that it answers the *same question*. The limited path and
//! the unlimited one are different code, so for every shape the first `k` values
//! of the full sort are the control, and `LIMIT k` must reproduce them.
//!
//! Row counts here are deliberately above the threshold that enables the scan
//! (the limit must be a small fraction of the rows), because a test under it
//! measures the old path and passes without touching the new one.

use falcondb::db::Database;
use falcondb::storage::column::ColumnBuilder;
use falcondb::storage::schema::{Field, Schema};
use falcondb::storage::table::RecordBatch;
use falcondb::storage::types::{DataType, ScalarValue};

const ROWS: usize = 1000;

fn db_with(scores: Vec<Option<f64>>, texts: Vec<Option<&str>>) -> Database {
    let batch = RecordBatch::new(
        Schema::new(vec![
            Field::nullable("score", DataType::Float64),
            Field::nullable("name", DataType::Utf8),
        ]),
        vec![
            ColumnBuilder::float64(scores),
            ColumnBuilder::utf8_from_str(texts),
        ],
    )
    .unwrap();
    let db = Database::in_memory();
    db.create_table_from_batches("t", vec![batch]).unwrap();
    db
}

/// The column of a result, as nullable strings so floats and text share one
/// comparison.
fn column(db: &Database, sql: &str) -> Vec<Option<String>> {
    db.sql(sql)
        .unwrap()
        .batches
        .iter()
        .flat_map(|b| {
            (0..b.row_count())
                .map(|i| {
                    // A NULL arrives as the typed `Float64(None)` rather than as
                    // `ScalarValue::Null`, so both spellings are read as absent.
                    let v = b.column(0).get(i).unwrap();
                    let text = format!("{v:?}");
                    if text == "Null" || text.ends_with("(None)") {
                        None
                    } else {
                        Some(text)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `ORDER BY ... LIMIT k` must equal the first `k` of the same sort unlimited.
#[test]
fn a_limited_sort_returns_the_head_of_the_full_sort() {
    // Values are distinct, so there is exactly one right answer and a tie at the
    // boundary cannot excuse a difference. Ties get their own test below.
    let scores: Vec<Option<f64>> = (0..ROWS)
        .map(|i| {
            if i % 97 == 0 {
                None
            } else {
                // Scattered rather than monotonic: an ordered column would let a
                // heap that only ever accepted its first `k` rows pass for ASC.
                Some(((i * 7919) % ROWS) as f64 + 0.5)
            }
        })
        .collect();
    let names: Vec<Option<&str>> = (0..ROWS)
        .map(|i| if i % 89 == 0 { None } else { Some(WORDS[i % WORDS.len()]) })
        .collect();
    let db = db_with(scores, names);

    for col in ["score", "name"] {
        for order in ["ASC", "DESC"] {
            for nulls in ["", " NULLS FIRST", " NULLS LAST"] {
                for k in [1usize, 2, 10] {
                    let tail = format!("ORDER BY {col} {order}{nulls}");
                    let limited = column(&db, &format!("SELECT {col} FROM t {tail} LIMIT {k}"));
                    let full = column(&db, &format!("SELECT {col} FROM t {tail}"));
                    assert_eq!(
                        limited,
                        full[..k.min(full.len())].to_vec(),
                        "{tail} LIMIT {k}",
                    );
                }
            }
        }
    }
}

/// With every key equal, the heap's comparison never accepts a replacement.
///
/// Which rows come back is arbitrary -- these sorts are unstable -- so what is
/// checked is the count and the value, not the identity.
#[test]
fn a_limited_sort_handles_a_column_of_one_value() {
    let db = db_with(
        (0..ROWS).map(|_| Some(1.5)).collect(),
        (0..ROWS).map(|_| Some("same")).collect(),
    );
    for col in ["score", "name"] {
        let got = column(&db, &format!("SELECT {col} FROM t ORDER BY {col} DESC LIMIT 7"));
        assert_eq!(got.len(), 7, "{col}");
        assert!(got.iter().all(|v| v == &got[0]), "{col}: {got:?}");
    }
}

/// More NULLs than the limit, placed first: the answer is NULLs only, and no
/// keyed row may appear.
#[test]
fn a_limited_sort_can_be_all_nulls() {
    let scores: Vec<Option<f64>> = (0..ROWS)
        .map(|i| if i < ROWS - 3 { None } else { Some(i as f64) })
        .collect();
    let db = db_with(scores, (0..ROWS).map(|_| Some("x")).collect());
    let got = column(&db, "SELECT score FROM t ORDER BY score ASC NULLS FIRST LIMIT 5");
    assert_eq!(got, vec![None, None, None, None, None]);
}


/// The same agreement, with the rows spread over many batches.
///
/// This is the path that reduces each batch to `k` rows before splicing, so that
/// the whole table is never copied into one contiguous batch. A single-batch
/// test never reaches it, and the mistake it exists to catch is a batch whose
/// own best rows are discarded because the reduction looked only at the first
/// one.
#[test]
fn a_limited_sort_over_many_batches_agrees_too() {
    // The best rows are put in the *last* batch, where a reduction that stopped
    // early would not see them.
    let per_batch = 100usize;
    let batches: Vec<_> = (0..20)
        .map(|b| {
            RecordBatch::new(
                Schema::new(vec![
                    Field::nullable("score", DataType::Float64),
                    Field::nullable("name", DataType::Utf8),
                ]),
                vec![
                    ColumnBuilder::float64(
                        (0..per_batch)
                            .map(|i| {
                                if (b * per_batch + i) % 97 == 0 {
                                    None
                                } else {
                                    Some((b * per_batch + i) as f64)
                                }
                            })
                            .collect(),
                    ),
                    ColumnBuilder::utf8_from_str(
                        (0..per_batch)
                            .map(|i| Some(WORDS[(b + i) % WORDS.len()]))
                            .collect(),
                    ),
                ],
            )
            .unwrap()
        })
        .collect();
    let db = Database::in_memory();
    db.create_table_from_batches("t", batches).unwrap();

    for col in ["score", "name"] {
        for order in ["ASC", "DESC"] {
            for nulls in ["", " NULLS FIRST", " NULLS LAST"] {
                for k in [1usize, 3, 10] {
                    let tail = format!("ORDER BY {col} {order}{nulls}");
                    let limited = column(&db, &format!("SELECT {col} FROM t {tail} LIMIT {k}"));
                    let full = column(&db, &format!("SELECT {col} FROM t {tail}"));
                    assert_eq!(
                        limited,
                        full[..k.min(full.len())].to_vec(),
                        "{tail} LIMIT {k}, 20 batches",
                    );
                }
            }
        }
    }
}

const WORDS: [&str; 7] = ["delta", "alpha", "echo", "bravo", "golf", "charlie", "foxtrot"];
