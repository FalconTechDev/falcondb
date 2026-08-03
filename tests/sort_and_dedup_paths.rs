// The typed sort fast path, top-N fusion, and hash-based dedup.
//
// These three optimisations replaced code that went through `ScalarValue` for
// every cell. Each introduced a way to be subtly wrong — NULL placement, ties,
// float identity, LIMIT with OFFSET — so the tests below pin the behaviour that
// has to match the slow path it replaced.

use std::io::Write;

use falcondb::db::Database;
use tempfile::NamedTempFile;

fn db_from(csv: &str) -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{csv}").unwrap();
    f.flush().unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();
    (db, f)
}

fn col(db: &Database, sql: &str) -> Vec<String> {
    db.sql(sql).unwrap().to_rows().into_iter().map(|r| r[0].clone()).collect()
}

// ─── Typed sort fast path ─────────────────────────────────────────────────────

#[test]
fn ascending_integer_sort() {
    let (db, _f) = db_from("v\n5\n1\n4\n2\n3\n");
    assert_eq!(col(&db, "SELECT v FROM t ORDER BY v"), ["1", "2", "3", "4", "5"]);
}

#[test]
fn descending_integer_sort() {
    let (db, _f) = db_from("v\n5\n1\n4\n2\n3\n");
    assert_eq!(col(&db, "SELECT v FROM t ORDER BY v DESC"), ["5", "4", "3", "2", "1"]);
}

#[test]
fn float_sort_orders_by_value() {
    let (db, _f) = db_from("v\n2.5\n0.1\n10.75\n1.0\n");
    assert_eq!(col(&db, "SELECT v FROM t ORDER BY v"), ["0.1", "1", "2.5", "10.75"]);
}

#[test]
fn negative_values_sort_below_zero() {
    let (db, _f) = db_from("v\n3\n-7\n0\n-1\n");
    assert_eq!(col(&db, "SELECT v FROM t ORDER BY v"), ["-7", "-1", "0", "3"]);
}

#[test]
fn nulls_sort_first_ascending() {
    // The typed path splits NULLs out rather than giving them a sentinel value;
    // a sentinel would collide with real data at the extremes.
    let (db, _f) = db_from("v\n3\n\n1\n");
    let got = col(&db, "SELECT v FROM t ORDER BY v");
    assert_eq!(got, ["NULL", "1", "3"], "got {got:?}");
}

#[test]
fn nulls_sort_last_descending() {
    let (db, _f) = db_from("v\n3\n\n1\n");
    let got = col(&db, "SELECT v FROM t ORDER BY v DESC");
    assert_eq!(got, ["3", "1", "NULL"], "got {got:?}");
}

#[test]
fn all_null_column_sorts_without_losing_rows() {
    let (db, _f) = db_from("v\n\n\n\n");
    assert_eq!(col(&db, "SELECT v FROM t ORDER BY v").len(), 3);
}

#[test]
fn ties_keep_every_row() {
    let (db, _f) = db_from("v\n2\n1\n2\n1\n2\n");
    assert_eq!(col(&db, "SELECT v FROM t ORDER BY v"), ["1", "1", "2", "2", "2"]);
}

#[test]
fn sort_carries_the_other_columns_along() {
    let (db, _f) = db_from("k,v\na,3\nb,1\nc,2\n");
    let rows = db.sql("SELECT k, v FROM t ORDER BY v").unwrap().to_rows();
    let ks: Vec<&str> = rows.iter().map(|r| r[0].as_str()).collect();
    assert_eq!(ks, ["b", "c", "a"], "row payload lost its pairing with the key");
}

#[test]
fn large_sort_crosses_the_parallel_threshold() {
    // Above 65_536 rows the sort switches to the parallel path.
    let mut csv = String::from("v\n");
    for i in (0..80_000i64).rev() {
        csv.push_str(&format!("{}\n", i % 1000));
    }
    let (db, _f) = db_from(&csv);
    let got = col(&db, "SELECT v FROM t ORDER BY v");
    assert_eq!(got.len(), 80_000);
    assert_eq!(got[0], "0");
    assert_eq!(got[got.len() - 1], "999");
}

// ─── Top-N (ORDER BY + LIMIT) ─────────────────────────────────────────────────

#[test]
fn limit_returns_the_smallest_rows() {
    let (db, _f) = db_from("v\n9\n3\n7\n1\n5\n");
    assert_eq!(col(&db, "SELECT v FROM t ORDER BY v LIMIT 3"), ["1", "3", "5"]);
}

#[test]
fn limit_with_desc_returns_the_largest() {
    let (db, _f) = db_from("v\n9\n3\n7\n1\n5\n");
    assert_eq!(col(&db, "SELECT v FROM t ORDER BY v DESC LIMIT 2"), ["9", "7"]);
}

#[test]
fn offset_is_honoured_with_limit() {
    // The sort keeps offset + limit rows, so OFFSET must still be applied after.
    let (db, _f) = db_from("v\n9\n3\n7\n1\n5\n");
    assert_eq!(col(&db, "SELECT v FROM t ORDER BY v LIMIT 2 OFFSET 2"), ["5", "7"]);
}

#[test]
fn limit_larger_than_the_table_returns_everything() {
    let (db, _f) = db_from("v\n2\n1\n");
    assert_eq!(col(&db, "SELECT v FROM t ORDER BY v LIMIT 100"), ["1", "2"]);
}

#[test]
fn top_n_matches_a_full_sort() {
    let mut csv = String::from("v\n");
    for i in 0..5_000i64 {
        csv.push_str(&format!("{}\n", (i * 7919) % 1000));
    }
    let (db, _f) = db_from(&csv);

    let full = col(&db, "SELECT v FROM t ORDER BY v");
    let top = col(&db, "SELECT v FROM t ORDER BY v LIMIT 25");
    assert_eq!(top, full[..25], "top-N disagreed with a full sort");
}

// ─── Hash-based dedup ─────────────────────────────────────────────────────────

#[test]
fn distinct_on_integers() {
    let (db, _f) = db_from("v\n1\n2\n1\n3\n2\n");
    let mut got = col(&db, "SELECT DISTINCT v FROM t");
    got.sort();
    assert_eq!(got, ["1", "2", "3"]);
}

#[test]
fn distinct_dedups_within_a_single_batch() {
    // Regression: dedup recorded output positions for a batch that had not been
    // emitted yet, so comparing a row against an earlier row of the same batch
    // indexed past the end of the result.
    let (db, _f) = db_from("v\n7\n7\n7\n");
    assert_eq!(col(&db, "SELECT DISTINCT v FROM t"), ["7"]);
}

#[test]
fn distinct_compares_whole_rows() {
    let (db, _f) = db_from("a,b\n1,x\n1,y\n1,x\n");
    assert_eq!(db.sql("SELECT DISTINCT a, b FROM t").unwrap().row_count(), 2);
}

#[test]
fn distinct_treats_nulls_as_equal() {
    let (db, _f) = db_from("v\n\n\n1\n");
    assert_eq!(db.sql("SELECT DISTINCT v FROM t").unwrap().row_count(), 2);
}

#[test]
fn distinct_on_strings() {
    let (db, _f) = db_from("s\nalpha\nbeta\nalpha\n");
    let mut got = col(&db, "SELECT DISTINCT s FROM t");
    got.sort();
    assert_eq!(got, ["alpha", "beta"]);
}

#[test]
fn distinct_on_floats_identifies_equal_values() {
    let (db, _f) = db_from("v\n1.5\n2.5\n1.5\n");
    assert_eq!(db.sql("SELECT DISTINCT v FROM t").unwrap().row_count(), 2);
}

#[test]
fn distinct_across_many_batches() {
    let (db, _f) = db_from("v\n1\n");
    for i in 0..40 {
        db.sql(&format!("INSERT INTO t VALUES ({})", i % 5)).unwrap();
    }
    let mut got = col(&db, "SELECT DISTINCT v FROM t");
    got.sort();
    assert_eq!(got, ["0", "1", "2", "3", "4"]);
}

#[test]
fn distinct_result_matches_a_manual_count() {
    let mut csv = String::from("v\n");
    for i in 0..3_000i64 {
        csv.push_str(&format!("{}\n", i % 137));
    }
    let (db, _f) = db_from(&csv);
    assert_eq!(db.sql("SELECT DISTINCT v FROM t").unwrap().row_count(), 137);
}
