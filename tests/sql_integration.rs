// Full SQL integration tests — covers correctness across all operations.
// Each test loads data via CSV string and executes SQL through the Database API.

use std::io::Write;
use falcondb::db::Database;
use tempfile::NamedTempFile;

// ─── Test helpers ─────────────────────────────────────────────────────────────

fn db_from_csv(csv: &str, table: &str) -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{csv}").unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), table).unwrap();
    (db, f)
}

fn col_values(db: &Database, sql: &str, col: &str) -> Vec<String> {
    let r = db.sql(sql).unwrap();
    r.to_rows().into_iter()
        .map(|row| {
            let names = r.column_names();
            let idx   = names.iter().position(|n| n == col).unwrap();
            row[idx].clone()
        })
        .collect()
}

fn row_count(db: &Database, sql: &str) -> usize {
    db.sql(sql).unwrap().row_count()
}

fn first_val(db: &Database, sql: &str) -> String {
    let r = db.sql(sql).unwrap();
    r.to_rows()[0][0].clone()
}

// ─── SELECT ───────────────────────────────────────────────────────────────────

#[test]
fn select_star() {
    let (db, _) = db_from_csv("a,b\n1,x\n2,y\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t"), 2);
}

#[test]
fn select_specific_columns() {
    let (db, _) = db_from_csv("id,name,score\n1,alice,9.5\n2,bob,8.0\n", "t");
    let r = db.sql("SELECT id, name FROM t").unwrap();
    assert_eq!(r.column_names(), vec!["id", "name"]);
    assert_eq!(r.row_count(), 2);
}

#[test]
fn select_empty_table() {
    let (db, _) = db_from_csv("id,val\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t"), 0);
}

#[test]
fn select_single_row() {
    let (db, _) = db_from_csv("x\n42\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t"), 1);
    assert_eq!(first_val(&db, "SELECT * FROM t"), "42");
}

// ─── WHERE ────────────────────────────────────────────────────────────────────

#[test]
fn where_gt() {
    let (db, _) = db_from_csv("n\n1\n5\n10\n3\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n > 4"), 2);
}

#[test]
fn where_lt() {
    let (db, _) = db_from_csv("n\n1\n5\n10\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n < 5"), 1);
}

#[test]
fn where_eq_string() {
    let (db, _) = db_from_csv("name,dept\nalice,eng\nbob,hr\ncarol,eng\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE dept = 'eng'"), 2);
}

#[test]
fn where_and() {
    let (db, _) = db_from_csv("a,b\n1,10\n2,10\n1,20\n2,20\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE a = 1 AND b = 10"), 1);
}

#[test]
fn where_or() {
    let (db, _) = db_from_csv("n\n1\n5\n10\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n = 1 OR n = 10"), 2);
}

#[test]
fn where_between() {
    let (db, _) = db_from_csv("n\n1\n5\n10\n15\n20\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n BETWEEN 5 AND 15"), 3);
}

#[test]
fn where_not_between() {
    let (db, _) = db_from_csv("n\n1\n5\n10\n15\n20\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n NOT BETWEEN 5 AND 15"), 2);
}

#[test]
fn where_no_match() {
    let (db, _) = db_from_csv("n\n1\n2\n3\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n > 100"), 0);
}

#[test]
fn where_all_match() {
    let (db, _) = db_from_csv("n\n1\n2\n3\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n > 0"), 3);
}

// ─── NULL handling ────────────────────────────────────────────────────────────

#[test]
fn is_null_filter() {
    let (db, _) = db_from_csv("n\n1\nNULL\n3\nNULL\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n IS NULL"), 2);
}

#[test]
fn is_not_null_filter() {
    let (db, _) = db_from_csv("n\n1\nNULL\n3\nNULL\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n IS NOT NULL"), 2);
}

#[test]
fn null_not_equal_to_value() {
    // NULL > 0 should be false, not NULL matching.
    let (db, _) = db_from_csv("n\n1\nNULL\n3\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n > 0"), 2);
}

#[test]
fn null_in_string_column() {
    let (db, _) = db_from_csv("name\nalice\nNULL\nbob\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE name IS NULL"), 1);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE name IS NOT NULL"), 2);
}

// ─── LIMIT / OFFSET ───────────────────────────────────────────────────────────

#[test]
fn limit_basic() {
    let (db, _) = db_from_csv("n\n1\n2\n3\n4\n5\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t LIMIT 3"), 3);
}

#[test]
fn limit_larger_than_table() {
    let (db, _) = db_from_csv("n\n1\n2\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t LIMIT 100"), 2);
}

#[test]
fn limit_zero() {
    let (db, _) = db_from_csv("n\n1\n2\n3\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t LIMIT 0"), 0);
}

#[test]
fn offset_basic() {
    let (db, _) = db_from_csv("n\n1\n2\n3\n4\n5\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t LIMIT 100 OFFSET 3"), 2);
}

#[test]
fn offset_beyond_table() {
    let (db, _) = db_from_csv("n\n1\n2\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t LIMIT 10 OFFSET 5"), 0);
}

#[test]
fn limit_and_offset_combined() {
    let (db, _) = db_from_csv("n\n10\n20\n30\n40\n50\n", "t");
    let r = db.sql("SELECT * FROM t LIMIT 2 OFFSET 1").unwrap();
    assert_eq!(r.row_count(), 2);
    assert_eq!(r.to_rows()[0][0], "20");
    assert_eq!(r.to_rows()[1][0], "30");
}

// ─── ORDER BY ─────────────────────────────────────────────────────────────────

#[test]
fn order_by_asc() {
    let (db, _) = db_from_csv("n\n3\n1\n2\n", "t");
    let r = db.sql("SELECT * FROM t ORDER BY n ASC").unwrap();
    let vals: Vec<_> = r.to_rows().iter().map(|row| row[0].clone()).collect();
    assert_eq!(vals, vec!["1", "2", "3"]);
}

#[test]
fn order_by_desc() {
    let (db, _) = db_from_csv("n\n3\n1\n2\n", "t");
    let r = db.sql("SELECT * FROM t ORDER BY n DESC").unwrap();
    let vals: Vec<_> = r.to_rows().iter().map(|row| row[0].clone()).collect();
    assert_eq!(vals, vec!["3", "2", "1"]);
}

#[test]
fn order_by_string() {
    let (db, _) = db_from_csv("name\ncharlie\nalice\nbob\n", "t");
    let r = db.sql("SELECT * FROM t ORDER BY name ASC").unwrap();
    let vals: Vec<_> = r.to_rows().iter().map(|row| row[0].clone()).collect();
    assert_eq!(vals, vec!["alice", "bob", "charlie"]);
}

#[test]
fn order_by_multi_key() {
    let (db, _) = db_from_csv("dept,score\neng,8\nhr,9\neng,9\nhr,7\n", "t");
    let r = db.sql("SELECT * FROM t ORDER BY dept ASC, score DESC").unwrap();
    let rows = r.to_rows();
    assert_eq!(rows[0], vec!["eng", "9"]);
    assert_eq!(rows[1], vec!["eng", "8"]);
    assert_eq!(rows[2], vec!["hr", "9"]);
}

#[test]
fn order_by_with_limit() {
    let (db, _) = db_from_csv("n\n3\n1\n4\n1\n5\n9\n", "t");
    let r = db.sql("SELECT * FROM t ORDER BY n DESC LIMIT 3").unwrap();
    assert_eq!(r.row_count(), 3);
    assert_eq!(r.to_rows()[0][0], "9");
}

// ─── GROUP BY + aggregates ────────────────────────────────────────────────────

#[test]
fn group_by_count() {
    let (db, _) = db_from_csv("dept\neng\nhr\neng\nhr\neng\n", "t");
    let r = db.sql("SELECT dept, COUNT(*) as n FROM t GROUP BY dept").unwrap();
    assert_eq!(r.row_count(), 2);
    let mut rows = r.to_rows();
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    assert_eq!(rows[0], vec!["eng", "3"]);
    assert_eq!(rows[1], vec!["hr", "2"]);
}

#[test]
fn group_by_sum() {
    let (db, _) = db_from_csv("dept,salary\neng,100\nhr,80\neng,120\nhr,90\n", "t");
    let r = db.sql("SELECT dept, SUM(salary) as total FROM t GROUP BY dept").unwrap();
    let mut rows = r.to_rows();
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    assert_eq!(rows[0][1], "220");
    assert_eq!(rows[1][1], "170");
}

#[test]
fn group_by_avg() {
    let (db, _) = db_from_csv("dept,score\neng,10\neng,20\nhr,30\n", "t");
    let r = db.sql("SELECT dept, AVG(score) as avg FROM t GROUP BY dept").unwrap();
    let mut rows = r.to_rows();
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    assert_eq!(rows[0][1], "15"); // eng avg = (10+20)/2
}

#[test]
fn group_by_min_max() {
    let (db, _) = db_from_csv("dept,score\neng,5\neng,10\neng,7\n", "t");
    let r = db.sql("SELECT dept, MIN(score) as mn, MAX(score) as mx FROM t GROUP BY dept").unwrap();
    assert_eq!(r.row_count(), 1);
    let row = &r.to_rows()[0];
    assert_eq!(row[1], "5");
    assert_eq!(row[2], "10");
}

#[test]
fn aggregate_without_group_by() {
    let (db, _) = db_from_csv("n\n1\n2\n3\n4\n5\n", "t");
    let r = db.sql("SELECT COUNT(*) as cnt FROM t").unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.to_rows()[0][0], "5");
}

#[test]
fn count_on_empty_table() {
    let (db, _) = db_from_csv("n\n", "t");
    let r = db.sql("SELECT COUNT(*) as cnt FROM t").unwrap();
    assert_eq!(r.to_rows()[0][0], "0");
}

#[test]
fn group_by_single_group() {
    let (db, _) = db_from_csv("dept,n\neng,1\neng,2\neng,3\n", "t");
    let r = db.sql("SELECT dept, SUM(n) as total FROM t GROUP BY dept").unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.to_rows()[0][1], "6");
}

#[test]
fn group_by_all_unique() {
    let (db, _) = db_from_csv("dept,n\neng,1\nhr,2\nfinance,3\n", "t");
    let r = db.sql("SELECT dept, COUNT(*) as cnt FROM t GROUP BY dept").unwrap();
    assert_eq!(r.row_count(), 3);
}

// ─── Type correctness ─────────────────────────────────────────────────────────

#[test]
fn integer_arithmetic_in_filter() {
    let (db, _) = db_from_csv("a,b\n5,3\n10,2\n1,8\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE a > 4"), 2);
}

#[test]
fn float_comparison() {
    let (db, _) = db_from_csv("x\n1.5\n2.5\n3.5\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE x > 2.0"), 2);
}

#[test]
fn boolean_column() {
    let (db, _) = db_from_csv("flag\ntrue\nfalse\ntrue\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t"), 3);
}

// ─── Multiple tables ──────────────────────────────────────────────────────────

#[test]
fn two_tables_independent() {
    let mut f1 = NamedTempFile::new().unwrap();
    write!(f1, "id\n1\n2\n").unwrap();
    let mut f2 = NamedTempFile::new().unwrap();
    write!(f2, "id\n10\n20\n30\n").unwrap();

    let db = Database::in_memory();
    db.load_csv(f1.path(), "a").unwrap();
    db.load_csv(f2.path(), "b").unwrap();

    assert_eq!(row_count(&db, "SELECT * FROM a"), 2);
    assert_eq!(row_count(&db, "SELECT * FROM b"), 3);
}

// ─── Save / load .fdb ─────────────────────────────────────────────────────────

#[test]
fn fdb_roundtrip_preserves_query_results() {
    let dir  = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("data.fdb");

    let (db, _) = db_from_csv("dept,score\neng,9.5\nhr,7.0\neng,8.5\n", "t");
    db.save_fdb("t", &path).unwrap();

    let db2 = Database::in_memory();
    db2.load_fdb(&path, "t2").unwrap();

    let r = db2.sql("SELECT dept, COUNT(*) as n FROM t2 GROUP BY dept").unwrap();
    assert_eq!(r.row_count(), 2);
}

// ─── Error handling ───────────────────────────────────────────────────────────

#[test]
fn unknown_table_returns_error() {
    let db = Database::in_memory();
    assert!(db.sql("SELECT * FROM no_such_table").is_err());
}

#[test]
fn invalid_sql_returns_error() {
    let db = Database::in_memory();
    assert!(db.sql("THIS IS NOT SQL").is_err());
}

#[test]
fn unsupported_statement_returns_error() {
    let db = Database::in_memory();
    // Duplicate CREATE TABLE without IF NOT EXISTS is an error.
    db.sql("CREATE TABLE u (a INT)").unwrap();
    assert!(db.sql("CREATE TABLE u (b TEXT)").is_err());
    // Reference to a non-existent table is always an error.
    assert!(db.sql("SELECT * FROM nonexistent").is_err());
}

// ─── DDL: CREATE TABLE / DROP TABLE / ALTER TABLE ─────────────────────────────

#[test]
fn create_table_and_insert() {
    let db = Database::in_memory();
    db.sql("CREATE TABLE employees (id INT, name VARCHAR, salary FLOAT)").unwrap();
    db.sql("INSERT INTO employees VALUES (1, 'Alice', 90000)").unwrap();
    db.sql("INSERT INTO employees VALUES (2, 'Bob', 80000)").unwrap();
    let r = db.sql("SELECT * FROM employees").unwrap();
    assert_eq!(r.row_count(), 2);
}

#[test]
fn create_table_if_not_exists() {
    let db = Database::in_memory();
    db.sql("CREATE TABLE t (x INT)").unwrap();
    // Second time is silently ignored.
    db.sql("CREATE TABLE IF NOT EXISTS t (y TEXT)").unwrap();
    // Schema unchanged — still one column named x.
    assert_eq!(db.table_info("t").unwrap().col_count, 1);
}

#[test]
fn drop_table() {
    let db = Database::in_memory();
    db.sql("CREATE TABLE t (x INT)").unwrap();
    assert!(db.table_info("t").is_some());
    db.sql("DROP TABLE t").unwrap();
    assert!(db.table_info("t").is_none());
}

#[test]
fn drop_table_if_exists() {
    let db = Database::in_memory();
    // Should not error even if table doesn't exist.
    db.sql("DROP TABLE IF EXISTS nonexistent").unwrap();
}

#[test]
fn drop_table_nonexistent_errors() {
    let db = Database::in_memory();
    assert!(db.sql("DROP TABLE nonexistent").is_err());
}

#[test]
fn alter_table_add_column() {
    let (db, _) = db_from_csv("id,name\n1,alice\n2,bob\n", "t");
    db.sql("ALTER TABLE t ADD COLUMN score FLOAT").unwrap();
    let info = db.table_info("t").unwrap();
    assert_eq!(info.col_count, 3);
    // New column should be NULL for existing rows.
    let r = db.sql("SELECT score FROM t WHERE score IS NULL").unwrap();
    assert_eq!(r.row_count(), 2);
}

#[test]
fn alter_table_drop_column() {
    let (db, _) = db_from_csv("id,name,score\n1,alice,9.5\n2,bob,8.0\n", "t");
    db.sql("ALTER TABLE t DROP COLUMN score").unwrap();
    let info = db.table_info("t").unwrap();
    assert_eq!(info.col_count, 2);
    assert!(db.sql("SELECT score FROM t").is_err());
}

#[test]
fn explain_select() {
    let (db, _) = db_from_csv("dept,salary\neng,100\nhr,80\n", "t");
    let r = db.sql(
        "EXPLAIN SELECT dept, COUNT(*) as n FROM t WHERE salary > 50 GROUP BY dept ORDER BY n DESC LIMIT 5"
    ).unwrap();
    // Result should have at least a few plan lines.
    assert!(r.row_count() >= 3);
    let text: Vec<_> = r.to_rows().into_iter().map(|r| r[0].clone()).collect();
    let joined = text.join("\n");
    assert!(joined.contains("Scan"), "expected Scan in plan:\n{joined}");
    assert!(joined.contains("Filter"), "expected Filter in plan:\n{joined}");
    assert!(joined.contains("GroupBy"), "expected GroupBy in plan:\n{joined}");
    assert!(joined.contains("Sort"), "expected Sort in plan:\n{joined}");
    assert!(joined.contains("Limit"), "expected Limit in plan:\n{joined}");
}

#[test]
fn explain_create_table() {
    let db = Database::in_memory();
    let r = db.sql("EXPLAIN CREATE TABLE employees (id INT NOT NULL, name VARCHAR)").unwrap();
    let text = r.to_rows().into_iter().map(|r| r[0].clone()).collect::<Vec<_>>().join("\n");
    assert!(text.contains("CreateTable"), "expected CreateTable in:\n{text}");
    assert!(text.contains("employees"), "expected table name in:\n{text}");
}

// ─── DML: INSERT / UPDATE / DELETE ────────────────────────────────────────────

#[test]
fn insert_adds_row() {
    let (db, _) = db_from_csv("id,name\n1,alice\n", "t");
    let res = db.sql("INSERT INTO t VALUES (2, 'bob')").unwrap();
    assert_eq!(res.to_rows()[0][0], "1", "one row affected");
    assert_eq!(db.sql("SELECT * FROM t").unwrap().row_count(), 2);
}

#[test]
fn insert_multiple_rows() {
    let (db, _) = db_from_csv("id,name\n1,alice\n", "t");
    db.sql("INSERT INTO t VALUES (2, 'bob'), (3, 'carol')").unwrap();
    assert_eq!(db.sql("SELECT * FROM t").unwrap().row_count(), 3);
}

#[test]
fn insert_named_columns() {
    let (db, _) = db_from_csv("id,name\n1,alice\n", "t");
    db.sql("INSERT INTO t (id) VALUES (7)").unwrap();
    let rows = db.sql("SELECT id FROM t WHERE id = 7").unwrap();
    assert_eq!(rows.row_count(), 1);
}

#[test]
fn insert_is_visible_to_aggregates() {
    let (db, _) = db_from_csv("dept,salary\neng,100\n", "t");
    db.sql("INSERT INTO t VALUES ('eng', 200)").unwrap();
    let rows = db.sql("SELECT SUM(salary) as total FROM t").unwrap().to_rows();
    assert_eq!(rows[0][0], "300");
}

#[test]
fn delete_with_where() {
    let (db, _) = db_from_csv("id,score\n1,9.5\n2,7.0\n3,8.5\n", "t");
    let res = db.sql("DELETE FROM t WHERE score > 8.0").unwrap();
    assert_eq!(res.to_rows()[0][0], "2");
    assert_eq!(db.sql("SELECT * FROM t").unwrap().row_count(), 1);
}

#[test]
fn delete_all_rows() {
    let (db, _) = db_from_csv("id\n1\n2\n3\n", "t");
    db.sql("DELETE FROM t").unwrap();
    assert_eq!(db.sql("SELECT * FROM t").unwrap().row_count(), 0);
}

#[test]
fn update_with_where() {
    let (db, _) = db_from_csv("id,score\n1,9.5\n2,7.0\n", "t");
    let res = db.sql("UPDATE t SET score = 0 WHERE id = 1").unwrap();
    assert_eq!(res.to_rows()[0][0], "1");

    let rows = db.sql("SELECT id, score FROM t ORDER BY id").unwrap().to_rows();
    assert_eq!(rows[0][1], "0");   // changed
    assert_eq!(rows[1][1], "7");   // untouched
}

#[test]
fn update_expression_referencing_column() {
    let (db, _) = db_from_csv("id,score\n1,10\n2,20\n", "t");
    db.sql("UPDATE t SET score = score * 2").unwrap();
    let rows = db.sql("SELECT score FROM t ORDER BY id").unwrap().to_rows();
    assert_eq!(rows[0][0], "20");
    assert_eq!(rows[1][0], "40");
}

// ─── WAL durability ───────────────────────────────────────────────────────────

/// Load `csv` into a WAL-backed database rooted at `dir`.
fn wal_db(dir: &std::path::Path, csv: &str) -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{csv}").unwrap();
    let db = Database::with_wal(dir.join("wal.log")).unwrap();
    db.load_csv(f.path(), "t").unwrap();
    (db, f)
}

#[test]
fn wal_replays_insert_after_restart() {
    let dir = tempfile::TempDir::new().unwrap();
    let csv = "id,name\n1,alice\n";

    {
        let (db, _f) = wal_db(dir.path(), csv);
        db.sql("INSERT INTO t VALUES (2, 'bob')").unwrap();
        assert_eq!(db.sql("SELECT * FROM t").unwrap().row_count(), 2);
    } // "crash" — nothing was persisted to a data file

    // Reopen from the same base CSV and replay the log.
    let (db2, _f2) = wal_db(dir.path(), csv);
    assert_eq!(db2.sql("SELECT * FROM t").unwrap().row_count(), 1, "before recovery");

    let replayed = db2.recover().unwrap();
    assert_eq!(replayed, 1);
    assert_eq!(db2.sql("SELECT * FROM t").unwrap().row_count(), 2, "after recovery");
}

#[test]
fn wal_replays_mixed_dml_in_order() {
    let dir = tempfile::TempDir::new().unwrap();
    let csv = "id,score\n1,10\n2,20\n";

    {
        let (db, _f) = wal_db(dir.path(), csv);
        db.sql("INSERT INTO t VALUES (3, 30)").unwrap();
        db.sql("UPDATE t SET score = 99 WHERE id = 1").unwrap();
        db.sql("DELETE FROM t WHERE id = 2").unwrap();
    }

    let (db2, _f2) = wal_db(dir.path(), csv);
    assert_eq!(db2.recover().unwrap(), 3);

    let rows = db2.sql("SELECT id, score FROM t ORDER BY id").unwrap().to_rows();
    assert_eq!(rows.len(), 2, "id 2 was deleted");
    assert_eq!(rows[0], vec!["1", "99"], "update applied");
    assert_eq!(rows[1], vec!["3", "30"], "insert applied");
}

#[test]
fn recover_is_idempotent_within_one_run() {
    // Replaying must not append to the log it is reading, otherwise each recovery
    // would double the number of entries.
    let dir = tempfile::TempDir::new().unwrap();
    let csv = "id\n1\n";

    {
        let (db, _f) = wal_db(dir.path(), csv);
        db.sql("INSERT INTO t VALUES (2)").unwrap();
    }

    let (db2, _f2) = wal_db(dir.path(), csv);
    assert_eq!(db2.recover().unwrap(), 1);
    assert_eq!(db2.sql("SELECT * FROM t").unwrap().row_count(), 2);

    // A third open must still see exactly one logged statement.
    let (db3, _f3) = wal_db(dir.path(), csv);
    assert_eq!(db3.recover().unwrap(), 1, "log must not have grown during replay");
}

#[test]
fn in_memory_database_ignores_recover() {
    let (db, _) = db_from_csv("id\n1\n", "t");
    assert_eq!(db.recover().unwrap(), 0);
}

#[test]
fn failed_dml_is_not_journalled() {
    let dir = tempfile::TempDir::new().unwrap();
    let csv = "id\n1\n";

    {
        let (db, _f) = wal_db(dir.path(), csv);
        assert!(db.sql("UPDATE t SET nope = 1").is_err());
    }

    let (db2, _f2) = wal_db(dir.path(), csv);
    assert_eq!(db2.recover().unwrap(), 0, "rejected statement must not reach the log");
}

// ─── CTE (WITH) ───────────────────────────────────────────────────────────────

#[test]
fn cte_basic() {
    let (db, _) = db_from_csv("dept,salary\neng,100\nhr,80\neng,120\n", "t");
    let rows = db
        .sql("WITH high AS (SELECT dept, salary FROM t WHERE salary > 90) \
              SELECT dept, salary FROM high ORDER BY salary")
        .unwrap()
        .to_rows();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][1], "100");
    assert_eq!(rows[1][1], "120");
}

#[test]
fn cte_feeds_aggregate() {
    let (db, _) = db_from_csv("dept,salary\neng,100\nhr,80\neng,120\n", "t");
    let rows = db
        .sql("WITH e AS (SELECT dept, salary FROM t WHERE dept = 'eng') \
              SELECT SUM(salary) as total FROM e")
        .unwrap()
        .to_rows();
    assert_eq!(rows[0][0], "220");
}

#[test]
fn cte_can_reference_earlier_cte() {
    let (db, _) = db_from_csv("dept,salary\neng,100\nhr,80\neng,120\n", "t");
    let rows = db
        .sql("WITH a AS (SELECT dept, salary FROM t WHERE salary > 90), \
                   b AS (SELECT dept, salary FROM a WHERE dept = 'eng') \
              SELECT COUNT(*) as n FROM b")
        .unwrap()
        .to_rows();
    assert_eq!(rows[0][0], "2");
}

#[test]
fn cte_does_not_leak_into_later_queries() {
    let (db, _) = db_from_csv("dept,salary\neng,100\n", "t");
    db.sql("WITH tmp AS (SELECT dept FROM t) SELECT dept FROM tmp").unwrap();
    // `tmp` was scoped to that statement only.
    assert!(db.sql("SELECT dept FROM tmp").is_err());
}

/// `WITH RECURSIVE` used to be rejected, and a test asserted that. It has been
/// implemented since (see the recursive branch in `Database::execute_plan`), so
/// the old test was asserting a limitation that no longer existed and had been
/// failing ever since. A red test is worse than no test: every run after it
/// landed had to be read past, and it stopped being evidence of anything.
///
/// These assert what the engine actually does now.
#[test]
fn recursive_cte_marked_but_not_recursive_is_accepted() {
    let (db, _) = db_from_csv("n\n1\n", "t");
    let rows = db
        .sql("WITH RECURSIVE r AS (SELECT n FROM t) SELECT n FROM r")
        .expect("RECURSIVE with no UNION is an ordinary CTE")
        .to_rows();
    assert_eq!(rows.len(), 1);
}

/// Currently fails: the query returns only the anchor row, `["1"]`.
///
/// Ignored rather than deleted, and ignored rather than left red. The engine does
/// not reject a recursive CTE and does not evaluate one either — it silently
/// returns a partial answer, which is worse than the rejection the old test
/// expected. Two observations that do not yet add up, recorded so whoever picks
/// this up does not start from scratch:
///
///   - Without `AS n` on the recursive term, the query fails with
///     `ColumnNotFound("n")`. That error can only come from a second iteration,
///     so the loop does run: the recursive term emits a column named `n + 1` and
///     the next pass cannot find `n`. Standard SQL takes the CTE's column names
///     from the anchor term and matches the recursive term positionally.
///   - With `AS n`, no error — but the result is the anchor alone, so the loop
///     stopped after one pass without adding rows.
///
/// Diagnosing that is a SQL-semantics fix, not a performance one, so it is not
/// being done inside the optimisation work that found it.
#[test]
fn recursive_cte_walks_to_a_fixed_point() {
    let (db, _) = db_from_csv("n\n1\n", "t");
    // Anchor yields 1; each step adds one until the guard stops it at 5.
    let rows = db
        .sql(
            "WITH RECURSIVE r AS ( \
               SELECT n FROM t \
               UNION ALL \
               SELECT n + 1 AS n FROM r WHERE n < 5 \
             ) SELECT n FROM r ORDER BY n",
        )
        .expect("recursive CTE should evaluate")
        .to_rows();
    let got: Vec<String> = rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(got, vec!["1", "2", "3", "4", "5"], "expected the fixed point 1..5");
}

/// The recursive term is matched to the anchor by position, not by name, so it
/// does not need to alias its output back to the anchor's column name.
#[test]
fn recursive_cte_matches_the_anchor_by_position_not_name() {
    let (db, _) = db_from_csv("n
1
", "t");
    let rows = db
        .sql(
            "WITH RECURSIVE r AS (                SELECT n FROM t                UNION ALL                SELECT n + 1 FROM r WHERE n < 5              ) SELECT n FROM r ORDER BY n",
        )
        .expect("an unaliased recursive term should still bind to the anchor's column")
        .to_rows();
    let got: Vec<String> = rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(got, vec!["1", "2", "3", "4", "5"]);
}

/// A recursive term with no stopping predicate has no fixed point. It must be
/// rejected by the iteration cap rather than run until the process dies.
#[test]
fn non_terminating_recursive_cte_hits_the_iteration_guard() {
    let (db, _) = db_from_csv("n
1
", "t");
    let res = db.sql(
        "WITH RECURSIVE r AS (            SELECT n FROM t            UNION ALL            SELECT n + 1 AS n FROM r          ) SELECT n FROM r",
    );
    let Err(err) = res else {
        panic!("a CTE with no fixed point should be rejected, not hang");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("fixed point") && msg.contains(&falcondb::db::MAX_RECURSIVE_ITERS.to_string()),
        "expected the guard's message, got: {msg}"
    );
}

// ─── Scalar subqueries ────────────────────────────────────────────────────────

#[test]
fn scalar_subquery_in_where() {
    let (db, _) = db_from_csv("name,score\na,10\nb,20\nc,30\n", "t");
    // AVG = 20 → only c (30) is strictly greater.
    let rows = db
        .sql("SELECT name FROM t WHERE score > (SELECT AVG(score) FROM t)")
        .unwrap()
        .to_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], "c");
}

#[test]
fn scalar_subquery_over_another_table() {
    let (db, _f1) = db_from_csv("name,score\na,10\nb,50\n", "t");

    let mut f2 = NamedTempFile::new().unwrap();
    write!(f2, "limit_val\n20\n").unwrap();
    db.load_csv(f2.path(), "cfg").unwrap();

    let rows = db
        .sql("SELECT name FROM t WHERE score > (SELECT MAX(limit_val) FROM cfg)")
        .unwrap()
        .to_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], "b");
}

#[test]
fn scalar_subquery_returning_many_rows_errors() {
    let (db, _) = db_from_csv("name,score\na,10\nb,20\n", "t");
    assert!(db
        .sql("SELECT name FROM t WHERE score > (SELECT score FROM t)")
        .is_err());
}

#[test]
fn scalar_subquery_empty_result_is_null() {
    let (db, _) = db_from_csv("name,score\na,10\n", "t");
    // Comparing against NULL yields no rows rather than an error.
    let rows = db
        .sql("SELECT name FROM t WHERE score > (SELECT score FROM t WHERE score > 999)")
        .unwrap();
    assert_eq!(rows.row_count(), 0);
}

#[test]
fn subquery_combined_with_cte() {
    let (db, _) = db_from_csv("dept,salary\neng,100\nhr,80\neng,120\n", "t");
    let rows = db
        .sql("WITH e AS (SELECT dept, salary FROM t WHERE dept = 'eng') \
              SELECT salary FROM e WHERE salary > (SELECT AVG(salary) FROM t)")
        .unwrap()
        .to_rows();
    // AVG over all = 100 → eng salaries strictly above 100 → 120
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], "120");
}

// ─── IN (SELECT …) and EXISTS ─────────────────────────────────────────────────

/// Two tables: `t` (people) and `allow` (a list of departments).
fn two_table_db() -> (Database, NamedTempFile, NamedTempFile) {
    let mut f1 = NamedTempFile::new().unwrap();
    write!(f1, "name,dept\nalice,eng\nbob,hr\ncarol,legal\n").unwrap();
    let mut f2 = NamedTempFile::new().unwrap();
    write!(f2, "dept\neng\nlegal\n").unwrap();

    let db = Database::in_memory();
    db.load_csv(f1.path(), "t").unwrap();
    db.load_csv(f2.path(), "allow").unwrap();
    (db, f1, f2)
}

#[test]
fn in_subquery_filters_by_other_table() {
    let (db, _a, _b) = two_table_db();
    let mut names = db
        .sql("SELECT name FROM t WHERE dept IN (SELECT dept FROM allow)")
        .unwrap()
        .to_rows()
        .into_iter()
        .map(|r| r[0].clone())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["alice", "carol"]);
}

#[test]
fn not_in_subquery_inverts() {
    let (db, _a, _b) = two_table_db();
    let names = db
        .sql("SELECT name FROM t WHERE dept NOT IN (SELECT dept FROM allow)")
        .unwrap()
        .to_rows();
    assert_eq!(names.len(), 1);
    assert_eq!(names[0][0], "bob");
}

#[test]
fn in_subquery_with_empty_result_matches_nothing() {
    let (db, _a, _b) = two_table_db();
    let rows = db
        .sql("SELECT name FROM t WHERE dept IN (SELECT dept FROM allow WHERE dept = 'nope')")
        .unwrap();
    assert_eq!(rows.row_count(), 0);
}

#[test]
fn in_subquery_requires_single_column() {
    let (db, _a, _b) = two_table_db();
    assert!(db
        .sql("SELECT name FROM t WHERE dept IN (SELECT name, dept FROM t)")
        .is_err());
}

#[test]
fn exists_true_returns_all_rows() {
    let (db, _a, _b) = two_table_db();
    let rows = db
        .sql("SELECT name FROM t WHERE EXISTS (SELECT dept FROM allow)")
        .unwrap();
    assert_eq!(rows.row_count(), 3);
}

#[test]
fn exists_false_returns_nothing() {
    let (db, _a, _b) = two_table_db();
    let rows = db
        .sql("SELECT name FROM t WHERE EXISTS (SELECT dept FROM allow WHERE dept = 'nope')")
        .unwrap();
    assert_eq!(rows.row_count(), 0);
}

#[test]
fn not_exists_inverts() {
    let (db, _a, _b) = two_table_db();
    let rows = db
        .sql("SELECT name FROM t WHERE NOT EXISTS (SELECT dept FROM allow WHERE dept = 'nope')")
        .unwrap();
    assert_eq!(rows.row_count(), 3);
}

#[test]
fn in_subquery_works_against_a_cte() {
    let (db, _a, _b) = two_table_db();
    let rows = db
        .sql("WITH ok AS (SELECT dept FROM allow WHERE dept = 'eng') \
              SELECT name FROM t WHERE dept IN (SELECT dept FROM ok)")
        .unwrap()
        .to_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], "alice");
}

// ─── Correlated subqueries ────────────────────────────────────────────────────

/// `emp` (dept, salary) plus `avg_by_dept` style lookups against the same table.
fn emp_db() -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(
        f,
        "name,dept,salary\nalice,eng,100\nbob,eng,200\ncarol,hr,80\ndave,hr,60\n"
    )
    .unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "emp").unwrap();
    (db, f)
}

#[test]
fn correlated_subquery_compares_against_own_group() {
    let (db, _f) = emp_db();
    // Everyone earning more than their own department's average.
    // eng avg = 150 → bob(200). hr avg = 70 → carol(80).
    let mut names = db
        .sql(
            "SELECT name FROM emp e \
             WHERE salary > (SELECT AVG(salary) FROM emp WHERE dept = e.dept)",
        )
        .unwrap()
        .to_rows()
        .into_iter()
        .map(|r| r[0].clone())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["bob", "carol"]);
}

#[test]
fn correlated_exists() {
    let (db, _f) = emp_db();
    let mut f2 = NamedTempFile::new().unwrap();
    write!(f2, "dept\neng\n").unwrap();
    db.load_csv(f2.path(), "allow").unwrap();

    // Only employees whose dept appears in `allow`.
    let mut names = db
        .sql("SELECT name FROM emp e WHERE EXISTS (SELECT dept FROM allow WHERE dept = e.dept)")
        .unwrap()
        .to_rows()
        .into_iter()
        .map(|r| r[0].clone())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["alice", "bob"]);
}

#[test]
fn correlated_not_exists() {
    let (db, _f) = emp_db();
    let mut f2 = NamedTempFile::new().unwrap();
    write!(f2, "dept\neng\n").unwrap();
    db.load_csv(f2.path(), "allow").unwrap();

    let mut names = db
        .sql("SELECT name FROM emp e WHERE NOT EXISTS (SELECT dept FROM allow WHERE dept = e.dept)")
        .unwrap()
        .to_rows()
        .into_iter()
        .map(|r| r[0].clone())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["carol", "dave"]);
}

#[test]
fn shared_column_name_is_not_clobbered_by_outer_row() {
    // Both tables have `dept`. Inside the subquery, an unqualified `dept` must
    // resolve to the *inner* table, not be replaced with the outer row's value —
    // otherwise this degenerates into `WHERE 'eng' = 'eng'` and matches everything.
    let (db, _f) = emp_db();
    let rows = db
        .sql("SELECT name FROM emp e WHERE salary > (SELECT MIN(salary) FROM emp WHERE dept = 'hr')")
        .unwrap()
        .to_rows();
    // MIN over hr = 60 → alice(100), bob(200), carol(80) qualify; dave(60) does not.
    assert_eq!(rows.len(), 3);
}

#[test]
fn uncorrelated_query_still_uses_the_fast_path() {
    // Sanity: adding correlation support must not change uncorrelated results.
    let (db, _f) = emp_db();
    let rows = db
        .sql("SELECT name FROM emp WHERE salary > (SELECT AVG(salary) FROM emp)")
        .unwrap()
        .to_rows();
    // overall avg = 110 → only bob(200)
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], "bob");
}

// ─── Regressions ──────────────────────────────────────────────────────────────

#[test]
fn integer_column_of_zeros_and_ones_is_not_boolean() {
    // Regression: type inference started at Boolean and `parse_bool` accepts "1"/"0",
    // so any integer column containing only 0 and 1 was silently typed Boolean —
    // which then made INSERT of any other integer write NULL.
    let (db, _) = db_from_csv("flag\n1\n0\n1\n", "t");
    db.sql("INSERT INTO t VALUES (7)").unwrap();
    let rows = db.sql("SELECT flag FROM t WHERE flag = 7").unwrap();
    assert_eq!(rows.row_count(), 1, "7 must be storable in an integer column");

    let total = db.sql("SELECT SUM(flag) as s FROM t").unwrap().to_rows();
    assert_eq!(total[0][0], "9"); // 1 + 0 + 1 + 7
}

#[test]
fn explicit_boolean_words_still_infer_boolean() {
    let (db, _) = db_from_csv("flag\ntrue\nfalse\n1\n", "t");
    let rows = db.sql("SELECT flag FROM t").unwrap().to_rows();
    assert_eq!(rows[0][0], "true");
    assert_eq!(rows[1][0], "false");
    assert_eq!(rows[2][0], "true", "1 reads as true once the column is boolean");
}

#[test]
fn update_does_not_null_out_on_type_widening() {
    // Regression: `score * 2` evaluates to Float64; writing it back into an Int64
    // column used to produce NULLs instead of coercing.
    let (db, _) = db_from_csv("id,score\n1,10\n2,20\n", "t");
    db.sql("UPDATE t SET score = score * 2").unwrap();
    let rows = db.sql("SELECT score FROM t ORDER BY id").unwrap().to_rows();
    assert_eq!(rows[0][0], "20");
    assert_eq!(rows[1][0], "40");
}

#[test]
fn insert_incompatible_type_errors_instead_of_writing_null() {
    let (db, _) = db_from_csv("flag\ntrue\nfalse\n", "t");
    // A boolean column must reject a string outright rather than store NULL.
    assert!(db.sql("INSERT INTO t VALUES ('banana')").is_err());
}

#[test]
fn dml_on_missing_table_errors() {
    let db = Database::in_memory();
    assert!(db.sql("INSERT INTO nope VALUES (1)").is_err());
    assert!(db.sql("UPDATE nope SET a = 1").is_err());
    assert!(db.sql("DELETE FROM nope").is_err());
}

#[test]
fn update_unknown_column_errors() {
    let (db, _) = db_from_csv("id\n1\n", "t");
    assert!(db.sql("UPDATE t SET missing = 1").is_err());
}

// ─── Edge cases ───────────────────────────────────────────────────────────────

#[test]
fn large_dataset_correctness() {
    let mut csv = "id,score\n".to_string();
    for i in 0..10_000 {
        csv.push_str(&format!("{},{}\n", i, i % 100));
    }
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{csv}").unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();

    let r = db.sql("SELECT COUNT(*) as n FROM t WHERE score >= 50").unwrap();
    assert_eq!(r.to_rows()[0][0], "5000");
}

#[test]
fn group_by_with_nulls() {
    let (db, _) = db_from_csv("dept,score\neng,10\nNULL,20\neng,30\nNULL,40\n", "t");
    let r = db.sql("SELECT dept, COUNT(*) as n FROM t GROUP BY dept").unwrap();
    assert_eq!(r.row_count(), 2);
}

#[test]
fn order_preserves_values() {
    let (db, _) = db_from_csv("name,score\nalice,9.5\nbob,7.0\ncarol,8.5\n", "t");
    let r = db.sql("SELECT name, score FROM t ORDER BY score DESC").unwrap();
    let rows = r.to_rows();
    assert_eq!(rows[0][0], "alice");
    assert_eq!(rows[0][1], "9.5");
    assert_eq!(rows[1][0], "carol");
    assert_eq!(rows[2][0], "bob");
}

#[test]
fn filter_then_sort_then_limit() {
    let (db, _) = db_from_csv(
        "name,score\nalice,9.5\nbob,4.0\ncarol,8.5\ndave,3.0\neve,7.0\n",
        "t",
    );
    let r = db.sql("SELECT name FROM t WHERE score > 5 ORDER BY score DESC LIMIT 2").unwrap();
    assert_eq!(r.row_count(), 2);
    assert_eq!(r.to_rows()[0][0], "alice");
    assert_eq!(r.to_rows()[1][0], "carol");
}

// ─── LIKE / ILIKE ─────────────────────────────────────────────────────────────

#[test]
fn like_prefix() {
    let (db, _) = db_from_csv("name\nalice\nbob\nalbert\ncarol\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE name LIKE 'al%'"), 2);
}

#[test]
fn like_suffix() {
    let (db, _) = db_from_csv("name\nalice\nbob\ncarolice\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE name LIKE '%ice'"), 2);
}

#[test]
fn like_underscore() {
    let (db, _) = db_from_csv("code\nabc\nxbc\nadc\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE code LIKE '_bc'"), 2);
}

#[test]
fn not_like() {
    let (db, _) = db_from_csv("name\nalice\nbob\ncarol\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE name NOT LIKE 'b%'"), 2);
}

#[test]
fn ilike_case_insensitive() {
    let (db, _) = db_from_csv("name\nAlice\nBOB\ncarol\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE name ILIKE 'a%'"), 1);
}

// ─── IN (...) ─────────────────────────────────────────────────────────────────

#[test]
fn in_list_strings() {
    let (db, _) = db_from_csv("dept\neng\nhr\nfin\neng\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE dept IN ('eng', 'hr')"), 3);
}

#[test]
fn in_list_numbers() {
    let (db, _) = db_from_csv("n\n1\n2\n3\n4\n5\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n IN (1, 3, 5)"), 3);
}

#[test]
fn not_in_list() {
    let (db, _) = db_from_csv("dept\neng\nhr\nfin\n", "t");
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE dept NOT IN ('hr', 'fin')"), 1);
}

// ─── HAVING ───────────────────────────────────────────────────────────────────

#[test]
fn having_count() {
    let (db, _) = db_from_csv("dept\neng\nhr\neng\nhr\neng\n", "t");
    let r = db.sql("SELECT dept, COUNT(*) as n FROM t GROUP BY dept HAVING n > 2").unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.to_rows()[0][0], "eng");
}

#[test]
fn having_sum() {
    let (db, _) = db_from_csv("dept,salary\neng,100\nhr,50\neng,120\nhr,30\n", "t");
    let r = db.sql(
        "SELECT dept, SUM(salary) as total FROM t GROUP BY dept HAVING total > 100"
    ).unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.to_rows()[0][0], "eng");
}

// ─── CASE WHEN ────────────────────────────────────────────────────────────────

#[test]
fn case_when_basic() {
    let (db, _) = db_from_csv("score\n90\n60\n40\n", "t");
    let r = db.sql(
        "SELECT CASE WHEN score >= 70 THEN 'pass' ELSE 'fail' END as result FROM t"
    ).unwrap();
    let rows = r.to_rows();
    assert_eq!(rows[0][0], "pass");
    assert_eq!(rows[1][0], "fail");
    assert_eq!(rows[2][0], "fail");
}

#[test]
fn case_when_multiple_branches() {
    let (db, _) = db_from_csv("n\n1\n2\n3\n", "t");
    let r = db.sql(
        "SELECT CASE WHEN n = 1 THEN 'one' WHEN n = 2 THEN 'two' ELSE 'other' END as w FROM t"
    ).unwrap();
    let rows = r.to_rows();
    assert_eq!(rows[0][0], "one");
    assert_eq!(rows[1][0], "two");
    assert_eq!(rows[2][0], "other");
}

// ─── Scalar functions ─────────────────────────────────────────────────────────

#[test]
fn func_upper_lower() {
    let (db, _) = db_from_csv("name\nalice\nBOB\n", "t");
    let r = db.sql("SELECT UPPER(name) as u, LOWER(name) as l FROM t").unwrap();
    let rows = r.to_rows();
    assert_eq!(rows[0][0], "ALICE");
    assert_eq!(rows[0][1], "alice");
    assert_eq!(rows[1][0], "BOB");
}

#[test]
fn func_length() {
    let (db, _) = db_from_csv("name\nalice\nbo\n", "t");
    assert_eq!(first_val(&db, "SELECT LENGTH(name) FROM t LIMIT 1"), "5");
}

#[test]
fn func_round() {
    let (db, _) = db_from_csv("n\n3.14159\n2.71828\n", "t");
    assert_eq!(first_val(&db, "SELECT ROUND(n, 2) FROM t LIMIT 1"), "3.14");
}

#[test]
fn func_abs() {
    let (db, _) = db_from_csv("n\n-5\n3\n-1\n", "t");
    let vals = col_values(&db, "SELECT ABS(n) as a FROM t", "a");
    assert_eq!(vals, vec!["5", "3", "1"]);
}

#[test]
fn func_coalesce() {
    let (db, _) = db_from_csv("a,b\nNULL,x\ny,z\n", "t");
    assert_eq!(first_val(&db, "SELECT COALESCE(a, b) FROM t LIMIT 1"), "x");
}

// ─── DISTINCT ─────────────────────────────────────────────────────────────────

#[test]
fn distinct_basic() {
    let (db, _) = db_from_csv("dept\neng\nhr\neng\nhr\neng\n", "t");
    assert_eq!(row_count(&db, "SELECT DISTINCT dept FROM t"), 2);
}

#[test]
fn distinct_with_order() {
    let (db, _) = db_from_csv("n\n3\n1\n2\n1\n3\n", "t");
    let r = db.sql("SELECT DISTINCT n FROM t ORDER BY n ASC").unwrap();
    let rows = r.to_rows();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][0], "1");
}

// ─── JOIN ─────────────────────────────────────────────────────────────────────

#[test]
fn inner_join_basic() {
    let mut f1 = tempfile::NamedTempFile::new().unwrap();
    let mut f2 = tempfile::NamedTempFile::new().unwrap();
    use std::io::Write;
    write!(f1, "id,name\n1,alice\n2,bob\n3,carol\n").unwrap();
    write!(f2, "id,score\n1,9.5\n2,7.0\n").unwrap();

    let db = falcondb::db::Database::in_memory();
    db.load_csv(f1.path(), "emp").unwrap();
    db.load_csv(f2.path(), "scores").unwrap();

    let r = db.sql("SELECT name, score FROM emp JOIN scores ON emp.id = scores.id").unwrap();
    assert_eq!(r.row_count(), 2);
    let mut rows = r.to_rows();
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    assert_eq!(rows[0][0], "alice");
    assert_eq!(rows[1][0], "bob");
}

#[test]
fn left_join_keeps_unmatched() {
    let mut f1 = tempfile::NamedTempFile::new().unwrap();
    let mut f2 = tempfile::NamedTempFile::new().unwrap();
    use std::io::Write;
    write!(f1, "id,name\n1,alice\n2,bob\n3,carol\n").unwrap();
    write!(f2, "id,score\n1,9.5\n2,7.0\n").unwrap();

    let db = falcondb::db::Database::in_memory();
    db.load_csv(f1.path(), "emp").unwrap();
    db.load_csv(f2.path(), "scores").unwrap();

    let r = db.sql("SELECT name FROM emp LEFT JOIN scores ON emp.id = scores.id").unwrap();
    // carol has no match → still appears (with NULLs on right side)
    assert_eq!(r.row_count(), 3);
}

// ─── CAST ─────────────────────────────────────────────────────────────────────

#[test]
fn cast_float_to_int() {
    let (db, _) = db_from_csv("n\n3.7\n1.2\n", "t");
    assert_eq!(first_val(&db, "SELECT CAST(n AS INTEGER) FROM t LIMIT 1"), "3");
}

#[test]
fn cast_int_to_text() {
    let (db, _) = db_from_csv("n\n42\n", "t");
    assert_eq!(first_val(&db, "SELECT CAST(n AS TEXT) FROM t"), "42");
}

// ─── COUNT DISTINCT ───────────────────────────────────────────────────────────

#[test]
fn count_distinct_basic() {
    let (db, _) = db_from_csv("dept\neng\nhr\neng\nhr\neng\n", "t");
    assert_eq!(first_val(&db, "SELECT COUNT(DISTINCT dept) as n FROM t"), "2");
}

#[test]
fn count_distinct_with_group() {
    let (db, _) = db_from_csv("dept,grade\neng,A\neng,B\neng,A\nhr,A\n", "t");
    let r = db.sql(
        "SELECT dept, COUNT(DISTINCT grade) as n FROM t GROUP BY dept"
    ).unwrap();
    let mut rows = r.to_rows();
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    assert_eq!(rows[0][1], "2"); // eng: A,B
    assert_eq!(rows[1][1], "1"); // hr: A
}

// ─── UNION ALL ────────────────────────────────────────────────────────────────

#[test]
fn union_all_basic() {
    let mut f1 = tempfile::NamedTempFile::new().unwrap();
    let mut f2 = tempfile::NamedTempFile::new().unwrap();
    use std::io::Write;
    write!(f1, "n\n1\n2\n").unwrap();
    write!(f2, "n\n3\n4\n").unwrap();

    let db = falcondb::db::Database::in_memory();
    db.load_csv(f1.path(), "a").unwrap();
    db.load_csv(f2.path(), "b").unwrap();

    assert_eq!(row_count(&db, "SELECT n FROM a UNION ALL SELECT n FROM b"), 4);
}

#[test]
fn union_all_preserves_duplicates() {
    let mut f1 = tempfile::NamedTempFile::new().unwrap();
    let mut f2 = tempfile::NamedTempFile::new().unwrap();
    use std::io::Write;
    write!(f1, "n\n1\n1\n").unwrap();
    write!(f2, "n\n1\n").unwrap();

    let db = falcondb::db::Database::in_memory();
    db.load_csv(f1.path(), "a").unwrap();
    db.load_csv(f2.path(), "b").unwrap();

    // UNION ALL keeps all rows including duplicates
    assert_eq!(row_count(&db, "SELECT n FROM a UNION ALL SELECT n FROM b"), 3);
}
