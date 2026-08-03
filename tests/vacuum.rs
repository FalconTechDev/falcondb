// VACUUM — compacting fragmented batches.
//
// DELETE and UPDATE rebuild only the batches they touch, so a heavily mutated
// table degrades into many small batches. VACUUM merges them back. The property
// that matters is that only the layout changes: every row, value and ordering
// must survive untouched.

use std::io::Write;

use falcondb::db::Database;
use tempfile::NamedTempFile;

fn db_with_rows(n: usize) -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "id,v\n").unwrap();
    for i in 0..n {
        writeln!(f, "{i},{}", i * 10).unwrap();
    }
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();
    (db, f)
}

fn batch_count(db: &Database) -> usize {
    db.get_table("t").unwrap().batch_count()
}

fn all_rows(db: &Database) -> Vec<Vec<String>> {
    db.sql("SELECT id, v FROM t ORDER BY id").unwrap().to_rows()
}

/// Fragment the table by inserting rows one at a time.
fn fragment(db: &Database, from: usize, to: usize) {
    for i in from..to {
        db.sql(&format!("INSERT INTO t VALUES ({i}, {})", i * 10)).unwrap();
    }
}

#[test]
fn vacuum_reduces_batch_count() {
    let (db, _f) = db_with_rows(5);
    fragment(&db, 5, 25);

    let before = batch_count(&db);
    assert!(before > 1, "test setup did not fragment the table");

    db.sql("VACUUM t").unwrap();
    let after = batch_count(&db);
    assert!(after < before, "VACUUM did not compact: {before} -> {after}");
    assert_eq!(after, 1, "small batches should merge into one");
}

#[test]
fn vacuum_preserves_every_row() {
    let (db, _f) = db_with_rows(5);
    fragment(&db, 5, 30);

    let before = all_rows(&db);
    db.sql("VACUUM t").unwrap();
    let after = all_rows(&db);

    assert_eq!(before, after, "VACUUM changed the data");
}

#[test]
fn vacuum_preserves_row_count() {
    let (db, _f) = db_with_rows(10);
    fragment(&db, 10, 40);

    let before = db.sql("SELECT id FROM t").unwrap().row_count();
    db.sql("VACUUM t").unwrap();
    let after = db.sql("SELECT id FROM t").unwrap().row_count();
    assert_eq!(before, after);
    assert_eq!(after, 40);
}

#[test]
fn vacuum_survives_deletes() {
    let (db, _f) = db_with_rows(10);
    fragment(&db, 10, 30);
    db.sql("DELETE FROM t WHERE id < 5").unwrap();

    let before = all_rows(&db);
    db.sql("VACUUM t").unwrap();
    assert_eq!(all_rows(&db), before);
    assert_eq!(db.sql("SELECT id FROM t").unwrap().row_count(), 25);
}

#[test]
fn vacuum_reports_batches_reclaimed() {
    let (db, _f) = db_with_rows(5);
    fragment(&db, 5, 20);

    let before = batch_count(&db);
    let r = db.sql("VACUUM t").unwrap();
    let reclaimed: usize = r.to_rows()[0][0].parse().unwrap();
    assert_eq!(reclaimed, before - batch_count(&db));
}

#[test]
fn vacuum_is_idempotent() {
    let (db, _f) = db_with_rows(5);
    fragment(&db, 5, 20);

    db.sql("VACUUM t").unwrap();
    let after_first = batch_count(&db);
    let rows = all_rows(&db);

    db.sql("VACUUM t").unwrap();
    assert_eq!(batch_count(&db), after_first, "second VACUUM changed the layout");
    assert_eq!(all_rows(&db), rows);
}

#[test]
fn vacuum_without_a_table_name_covers_every_table() {
    let (db, _f) = db_with_rows(5);
    fragment(&db, 5, 20);

    let mut f2 = NamedTempFile::new().unwrap();
    write!(f2, "id,v\n1,1\n").unwrap();
    db.load_csv(f2.path(), "u").unwrap();
    for i in 2..15 {
        db.sql(&format!("INSERT INTO u VALUES ({i}, {i})")).unwrap();
    }

    db.sql("VACUUM").unwrap();
    assert_eq!(db.get_table("t").unwrap().batch_count(), 1);
    assert_eq!(db.get_table("u").unwrap().batch_count(), 1);
}

#[test]
fn vacuum_on_a_missing_table_is_an_error() {
    let (db, _f) = db_with_rows(3);
    assert!(db.sql("VACUUM no_such_table").is_err());
}

#[test]
fn vacuum_on_an_untouched_table_is_a_no_op() {
    let (db, _f) = db_with_rows(20);
    let rows = all_rows(&db);
    db.sql("VACUUM t").unwrap();
    assert_eq!(all_rows(&db), rows);
    assert_eq!(batch_count(&db), 1);
}

#[test]
fn queries_still_work_after_vacuum() {
    let (db, _f) = db_with_rows(5);
    fragment(&db, 5, 30);
    db.sql("VACUUM t").unwrap();

    // Filter, aggregate and sort all have to work on the rebuilt batches.
    assert_eq!(db.sql("SELECT id FROM t WHERE id >= 20").unwrap().row_count(), 10);
    let total = db.sql("SELECT SUM(v) AS s FROM t").unwrap().to_rows()[0][0].clone();
    let expected: i64 = (0..30i64).map(|i| i * 10).sum();
    assert_eq!(total.parse::<f64>().unwrap() as i64, expected);
}
