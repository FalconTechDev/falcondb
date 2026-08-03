// EXPLAIN and EXPLAIN ANALYZE.
//
// Plain EXPLAIN describes the plan and must not touch data. ANALYZE runs the
// statement and reports measured time and the real row count.

use std::io::Write;

use falcondb::db::Database;
use tempfile::NamedTempFile;

fn db_with_rows(n: usize) -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "id,dept\n").unwrap();
    for i in 0..n {
        writeln!(f, "{i},{}", if i % 2 == 0 { "eng" } else { "hr" }).unwrap();
    }
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();
    (db, f)
}

fn plan_text(db: &Database, sql: &str) -> String {
    db.sql(sql)
        .unwrap()
        .to_rows()
        .into_iter()
        .map(|r| r[0].clone())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn explain_describes_the_plan() {
    let (db, _f) = db_with_rows(10);
    let plan = plan_text(&db, "EXPLAIN SELECT * FROM t WHERE id > 5");
    assert!(plan.contains("t"), "plan does not name the table: {plan}");
}

#[test]
fn explain_analyze_reports_timing_and_row_count() {
    let (db, _f) = db_with_rows(10);
    let plan = plan_text(&db, "EXPLAIN ANALYZE SELECT * FROM t WHERE id > 5");

    assert!(plan.contains("Execution time:"), "no timing line: {plan}");
    // ids 6..9 pass the filter.
    assert!(
        plan.contains("Rows produced: 4"),
        "wrong or missing row count: {plan}"
    );
}

#[test]
fn explain_analyze_still_shows_the_plan() {
    let (db, _f) = db_with_rows(10);
    let plan = plan_text(&db, "EXPLAIN ANALYZE SELECT dept, COUNT(*) FROM t GROUP BY dept");
    assert!(plan.contains("Execution time:"), "{plan}");
    assert!(
        plan.to_lowercase().contains("group"),
        "plan body missing from ANALYZE output: {plan}"
    );
}

#[test]
fn explain_analyze_row_count_tracks_the_data() {
    let (db, _f) = db_with_rows(100);
    let plan = plan_text(&db, "EXPLAIN ANALYZE SELECT * FROM t WHERE dept = 'eng'");
    assert!(plan.contains("Rows produced: 50"), "{plan}");
}

#[test]
fn plain_explain_does_not_execute() {
    let (db, _f) = db_with_rows(10);
    // If EXPLAIN ran the DELETE, the table would lose rows.
    let _ = plan_text(&db, "EXPLAIN DELETE FROM t WHERE id > 5");
    let remaining = db.sql("SELECT * FROM t").unwrap().row_count();
    assert_eq!(remaining, 10, "plain EXPLAIN modified the table");
}

#[test]
fn explain_analyze_of_a_mutation_does_execute() {
    // Documents the deliberate choice to match PostgreSQL/DuckDB: ANALYZE runs
    // the statement, so a mutation under ANALYZE really mutates.
    let (db, _f) = db_with_rows(10);
    let plan = plan_text(&db, "EXPLAIN ANALYZE DELETE FROM t WHERE id > 5");
    assert!(plan.contains("Execution time:"), "{plan}");

    let remaining = db.sql("SELECT * FROM t").unwrap().row_count();
    assert_eq!(remaining, 6, "ANALYZE should have applied the DELETE");
}

#[test]
fn explain_analyze_reports_a_failure_with_its_timing() {
    let (db, _f) = db_with_rows(5);
    let plan = plan_text(&db, "EXPLAIN ANALYZE SELECT * FROM no_such_table");
    assert!(
        plan.contains("Failed after"),
        "a failing statement should still report timing: {plan}"
    );
}
