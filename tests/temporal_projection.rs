// Projecting DATE / TIMESTAMP values.
//
// Regression coverage for a bug in `project_batch`: it matched only Boolean,
// Int, Float and Utf8, and every other type fell through to a catch-all that
// built an all-NULL Int64 column. Any query that projected a date through an
// expression silently returned NULL instead of the value — no error, no warning.

use std::io::Write;

use falcondb::db::Database;
use tempfile::NamedTempFile;

fn db_with_dates() -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "id,d\n1,2024-03-15\n2,2024-07-04\n3,2023-12-31\n").unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();
    (db, f)
}

fn col0(db: &Database, sql: &str) -> Vec<String> {
    db.sql(sql).unwrap().to_rows().into_iter().map(|r| r[0].clone()).collect()
}

#[test]
fn projecting_a_date_column_keeps_its_value() {
    let (db, _f) = db_with_dates();
    let vals = col0(&db, "SELECT d FROM t");
    assert!(
        vals.iter().all(|v| v != "NULL"),
        "date projection produced NULLs: {vals:?}"
    );
    assert!(vals.contains(&"2024-03-15".to_string()), "got {vals:?}");
}

#[test]
fn date_arithmetic_survives_projection() {
    let (db, _f) = db_with_dates();
    // 2023-12-31 + 1 day = 2024-01-01
    let vals = col0(&db, "SELECT d + 1 FROM t WHERE id = 3");
    assert_eq!(vals, vec!["2024-01-01"]);
}

#[test]
fn date_trunc_projects_a_date_not_null() {
    let (db, _f) = db_with_dates();
    let vals = col0(&db, "SELECT DATE_TRUNC('MONTH', d) FROM t WHERE id = 1");
    assert_eq!(vals, vec!["2024-03-01"]);
}

#[test]
fn extract_still_yields_an_integer() {
    let (db, _f) = db_with_dates();
    let vals = col0(&db, "SELECT EXTRACT(YEAR FROM d) FROM t WHERE id = 2");
    assert_eq!(vals, vec!["2024"]);
}

#[test]
fn cast_to_date_projects_correctly() {
    let (db, _f) = db_with_dates();
    let vals = col0(&db, "SELECT CAST('2020-02-29' AS DATE) FROM t WHERE id = 1");
    assert_eq!(vals, vec!["2020-02-29"]);
}

#[test]
fn date_ordering_is_preserved_through_projection() {
    let (db, _f) = db_with_dates();
    let vals = col0(&db, "SELECT d FROM t ORDER BY d");
    assert_eq!(vals, vec!["2023-12-31", "2024-03-15", "2024-07-04"]);
}
