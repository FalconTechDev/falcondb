// Set operations: UNION, UNION ALL, INTERSECT, EXCEPT.
//
// UNION (without ALL) used to be silently executed as UNION ALL, so these tests
// pin the deduplicating behaviour as much as they cover the new operators.

use std::io::Write;

use falcondb::db::Database;
use tempfile::NamedTempFile;

/// Build a two-table database from CSV bodies.
fn two_tables(a_csv: &str, b_csv: &str) -> (Database, NamedTempFile, NamedTempFile) {
    let mut fa = NamedTempFile::new().unwrap();
    let mut fb = NamedTempFile::new().unwrap();
    write!(fa, "{a_csv}").unwrap();
    write!(fb, "{b_csv}").unwrap();

    let db = Database::in_memory();
    db.load_csv(fa.path(), "a").unwrap();
    db.load_csv(fb.path(), "b").unwrap();
    (db, fa, fb)
}

/// Sorted single-column result, so tests do not depend on batch ordering.
fn sorted_col(db: &Database, sql: &str) -> Vec<String> {
    let mut vals: Vec<String> = db
        .sql(sql)
        .unwrap()
        .to_rows()
        .into_iter()
        .map(|r| r[0].clone())
        .collect();
    vals.sort();
    vals
}

// ─── UNION ────────────────────────────────────────────────────────────────────

#[test]
fn union_removes_duplicates() {
    let (db, _fa, _fb) = two_tables("n\n1\n2\n2\n", "n\n2\n3\n");
    assert_eq!(
        sorted_col(&db, "SELECT n FROM a UNION SELECT n FROM b"),
        vec!["1", "2", "3"]
    );
}

#[test]
fn union_all_keeps_duplicates() {
    let (db, _fa, _fb) = two_tables("n\n1\n2\n2\n", "n\n2\n3\n");
    assert_eq!(
        sorted_col(&db, "SELECT n FROM a UNION ALL SELECT n FROM b"),
        vec!["1", "2", "2", "2", "3"]
    );
}

#[test]
fn union_dedups_within_one_side() {
    // Duplicates inside the left side alone must also collapse.
    let (db, _fa, _fb) = two_tables("n\n7\n7\n7\n", "n\n9\n");
    assert_eq!(
        sorted_col(&db, "SELECT n FROM a UNION SELECT n FROM b"),
        vec!["7", "9"]
    );
}

// ─── INTERSECT ────────────────────────────────────────────────────────────────

#[test]
fn intersect_keeps_common_rows() {
    let (db, _fa, _fb) = two_tables("n\n1\n2\n3\n", "n\n2\n3\n4\n");
    assert_eq!(
        sorted_col(&db, "SELECT n FROM a INTERSECT SELECT n FROM b"),
        vec!["2", "3"]
    );
}

#[test]
fn intersect_result_is_a_set() {
    // 2 appears twice on the left but must appear once in the result.
    let (db, _fa, _fb) = two_tables("n\n2\n2\n5\n", "n\n2\n");
    assert_eq!(
        sorted_col(&db, "SELECT n FROM a INTERSECT SELECT n FROM b"),
        vec!["2"]
    );
}

#[test]
fn intersect_disjoint_is_empty() {
    let (db, _fa, _fb) = two_tables("n\n1\n2\n", "n\n8\n9\n");
    assert!(sorted_col(&db, "SELECT n FROM a INTERSECT SELECT n FROM b").is_empty());
}

// ─── EXCEPT ───────────────────────────────────────────────────────────────────

#[test]
fn except_removes_right_side_rows() {
    let (db, _fa, _fb) = two_tables("n\n1\n2\n3\n", "n\n2\n");
    assert_eq!(
        sorted_col(&db, "SELECT n FROM a EXCEPT SELECT n FROM b"),
        vec!["1", "3"]
    );
}

#[test]
fn except_result_is_a_set() {
    let (db, _fa, _fb) = two_tables("n\n1\n1\n1\n4\n", "n\n4\n");
    assert_eq!(
        sorted_col(&db, "SELECT n FROM a EXCEPT SELECT n FROM b"),
        vec!["1"]
    );
}

#[test]
fn except_is_not_commutative() {
    let (db, _fa, _fb) = two_tables("n\n1\n2\n", "n\n2\n3\n");
    assert_eq!(sorted_col(&db, "SELECT n FROM a EXCEPT SELECT n FROM b"), vec!["1"]);
    assert_eq!(sorted_col(&db, "SELECT n FROM b EXCEPT SELECT n FROM a"), vec!["3"]);
}

// ─── Composition ──────────────────────────────────────────────────────────────

#[test]
fn chained_ops_apply_left_to_right() {
    // (a UNION b) EXCEPT c, not a UNION (b EXCEPT c).
    let mut fc = NamedTempFile::new().unwrap();
    write!(fc, "n\n1\n").unwrap();
    let (db, _fa, _fb) = two_tables("n\n1\n2\n", "n\n3\n");
    db.load_csv(fc.path(), "c").unwrap();

    assert_eq!(
        sorted_col(
            &db,
            "SELECT n FROM a UNION SELECT n FROM b EXCEPT SELECT n FROM c"
        ),
        vec!["2", "3"]
    );
}

#[test]
fn branch_runs_its_own_where_clause() {
    let (db, _fa, _fb) = two_tables("n\n1\n2\n3\n", "n\n1\n2\n3\n");
    assert_eq!(
        sorted_col(
            &db,
            "SELECT n FROM a WHERE n < 3 EXCEPT SELECT n FROM b WHERE n = 1"
        ),
        vec!["2"]
    );
}

#[test]
fn multi_column_rows_compare_by_whole_row() {
    let (db, _fa, _fb) = two_tables(
        "k,v\nx,1\ny,2\n",
        "k,v\nx,9\n",
    );
    // (x,1) differs from (x,9), so nothing is removed.
    let rows = db
        .sql("SELECT k, v FROM a EXCEPT SELECT k, v FROM b")
        .unwrap()
        .row_count();
    assert_eq!(rows, 2);
}
