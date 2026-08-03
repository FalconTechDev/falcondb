// NOW() / CURRENT_TIMESTAMP / CURRENT_DATE.
//
// The load-bearing property is that a clock function is constant within a query:
// every row must see the same instant, or a query that compares two rows against
// "now" can disagree with itself.

use std::io::Write;

use falcondb::db::Database;
use tempfile::NamedTempFile;

fn db_with_rows(n: usize) -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "id\n").unwrap();
    for i in 0..n {
        writeln!(f, "{i}").unwrap();
    }
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();
    (db, f)
}

fn col0(db: &Database, sql: &str) -> Vec<String> {
    db.sql(sql).unwrap().to_rows().into_iter().map(|r| r[0].clone()).collect()
}

#[test]
fn now_returns_a_value_for_every_row() {
    let (db, _f) = db_with_rows(5);
    let vals = col0(&db, "SELECT NOW() FROM t");
    assert_eq!(vals.len(), 5);
    assert!(vals.iter().all(|v| !v.is_empty()), "NOW() produced an empty value");
}

#[test]
fn now_is_constant_within_a_query() {
    // 20k rows is enough that a per-row clock read would almost certainly tick.
    let (db, _f) = db_with_rows(20_000);
    let vals = col0(&db, "SELECT NOW() FROM t");
    let first = &vals[0];
    assert!(
        vals.iter().all(|v| v == first),
        "NOW() varied within one query — it must be evaluated once per query"
    );
}

#[test]
fn current_timestamp_without_parens_parses() {
    let (db, _f) = db_with_rows(3);
    let vals = col0(&db, "SELECT CURRENT_TIMESTAMP FROM t");
    assert_eq!(vals.len(), 3);
    assert!(vals.iter().all(|v| v == &vals[0]));
}

#[test]
fn current_date_is_constant_and_dateish() {
    let (db, _f) = db_with_rows(4);
    let vals = col0(&db, "SELECT CURRENT_DATE FROM t");
    assert!(vals.iter().all(|v| v == &vals[0]));
    // Rendered as a date, so it should carry the YYYY-MM-DD separators.
    assert_eq!(vals[0].matches('-').count(), 2, "got {:?}", vals[0]);
}

#[test]
fn now_advances_between_queries() {
    let (db, _f) = db_with_rows(1);
    let a = col0(&db, "SELECT NOW() FROM t");
    std::thread::sleep(std::time::Duration::from_millis(5));
    let b = col0(&db, "SELECT NOW() FROM t");
    assert_ne!(a[0], b[0], "NOW() should move forward across separate queries");
}

#[test]
fn current_date_usable_in_a_filter() {
    let (db, _f) = db_with_rows(3);
    // Every row passes: today is not before itself.
    let n = db
        .sql("SELECT id FROM t WHERE CURRENT_DATE >= CURRENT_DATE")
        .unwrap()
        .row_count();
    assert_eq!(n, 3);
}
