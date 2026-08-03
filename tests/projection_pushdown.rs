// Projection pushdown.
//
// The scan rebuilds only the columns some later stage reads, instead of
// materialising every column and discarding most of them at the projection step.
//
// Pushdown is an optimisation, so the tests here are about it *not* being
// observable: same rows, same values, same errors as a full scan. The one case
// worth calling out is a reference to a column that does not exist — pruning the
// scan to nothing would turn that error into a silent empty result.

use std::io::Write;

use falcondb::db::Database;
use tempfile::NamedTempFile;

/// Wide table: id, plus a..e, so pruning has something to prune.
fn wide_db() -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "id,a,b,c,d,e\n").unwrap();
    for i in 1..=20 {
        writeln!(f, "{i},{},{},{},{},{}", i, i * 2, i * 3, i * 4, i * 5).unwrap();
    }
    f.flush().unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();
    (db, f)
}

fn rows(db: &Database, sql: &str) -> Vec<Vec<String>> {
    db.sql(sql).unwrap().to_rows()
}

#[test]
fn selecting_one_column_returns_only_it() {
    let (db, _f) = wide_db();
    let r = db.sql("SELECT a FROM t ORDER BY a").unwrap();
    assert_eq!(r.column_names(), vec!["a"]);
    assert_eq!(r.row_count(), 20);
}

#[test]
fn filter_column_need_not_be_selected() {
    // `b` is read by WHERE but not returned; pushdown must still scan it.
    let (db, _f) = wide_db();
    let got = rows(&db, "SELECT a FROM t WHERE b > 20 ORDER BY a");
    assert_eq!(got.len(), 10);
    assert_eq!(got[0][0], "11");
}

#[test]
fn order_by_column_need_not_be_selected() {
    let (db, _f) = wide_db();
    let got = rows(&db, "SELECT a FROM t ORDER BY c DESC");
    assert_eq!(got.len(), 20);
    assert_eq!(got[0][0], "20", "ORDER BY key was pruned away");
}

#[test]
fn group_by_reads_its_key_and_aggregate_inputs() {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "g,v,unused\na,1,99\na,2,99\nb,10,99\n").unwrap();
    f.flush().unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();

    let mut got = rows(&db, "SELECT g, SUM(v) AS s FROM t GROUP BY g");
    got.sort();
    assert_eq!(got[0][1].parse::<f64>().unwrap(), 3.0);
    assert_eq!(got[1][1].parse::<f64>().unwrap(), 10.0);
}

#[test]
fn having_column_survives() {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "g,v,unused\na,1,0\na,2,0\nb,10,0\n").unwrap();
    f.flush().unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();

    // HAVING is matched against the aggregate's alias, so `s` must survive the
    // scan even though `unused` is pruned.
    let got = rows(&db, "SELECT g, SUM(v) AS s FROM t GROUP BY g HAVING s > 5");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0][0], "b");
}

#[test]
fn having_on_an_unselected_aggregate_errors_rather_than_returning_nothing() {
    // FalconDB resolves HAVING against the aggregated batch, so an aggregate that
    // is not in the SELECT list has no column to match. That used to be swallowed
    // and reported as zero rows; it must be an error instead.
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "g,v\na,1\nb,10\n").unwrap();
    f.flush().unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();

    assert!(
        db.sql("SELECT g FROM t GROUP BY g HAVING SUM(v) > 5").is_err(),
        "an unresolvable HAVING must not be reported as an empty result"
    );
}

#[test]
fn window_clause_columns_survive() {
    let (db, _f) = wide_db();
    // PARTITION BY and ORDER BY read columns the SELECT list does not return.
    let r = db
        .sql("SELECT a, ROW_NUMBER() OVER (ORDER BY d DESC) AS rn FROM t")
        .unwrap();
    assert_eq!(r.row_count(), 20);
}

#[test]
fn select_star_is_unaffected() {
    let (db, _f) = wide_db();
    let r = db.sql("SELECT * FROM t").unwrap();
    assert_eq!(r.column_names().len(), 6, "SELECT * must keep every column");
}

#[test]
fn expression_over_two_columns_reads_both() {
    let (db, _f) = wide_db();
    let got = rows(&db, "SELECT a + b AS s FROM t WHERE id = 3");
    // a=3, b=6
    assert_eq!(got[0][0].parse::<f64>().unwrap(), 9.0);
}

#[test]
fn missing_column_still_errors() {
    // The regression that pushdown introduced: pruning the scan to an empty batch
    // made this return zero rows instead of failing.
    let (db, _f) = wide_db();
    assert!(
        db.sql("SELECT no_such_column FROM t").is_err(),
        "a reference to a missing column must be an error, not an empty result"
    );
}

#[test]
fn missing_column_in_where_still_errors() {
    let (db, _f) = wide_db();
    assert!(db.sql("SELECT a FROM t WHERE no_such_column > 1").is_err());
}

#[test]
fn dropped_column_is_no_longer_selectable() {
    let (db, _f) = wide_db();
    db.sql("ALTER TABLE t DROP COLUMN e").unwrap();
    assert!(db.sql("SELECT e FROM t").is_err());
    // The surviving columns still work.
    assert_eq!(db.sql("SELECT a FROM t").unwrap().row_count(), 20);
}

#[test]
fn results_match_a_select_star_baseline() {
    let (db, _f) = wide_db();

    let narrow = rows(&db, "SELECT a, c FROM t WHERE b > 10 ORDER BY a");
    let wide: Vec<Vec<String>> = db
        .sql("SELECT * FROM t WHERE b > 10 ORDER BY a")
        .unwrap()
        .to_rows()
        .into_iter()
        // columns are id,a,b,c,d,e — pick a and c
        .map(|r| vec![r[1].clone(), r[3].clone()])
        .collect();

    assert_eq!(narrow, wide, "pushdown changed the result");
}

#[test]
fn distinct_still_dedups_after_pushdown() {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "g,v\na,1\na,2\nb,3\n").unwrap();
    f.flush().unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();

    let got = rows(&db, "SELECT DISTINCT g FROM t");
    assert_eq!(got.len(), 2);
}

#[test]
fn limit_and_offset_still_apply() {
    let (db, _f) = wide_db();
    let got = rows(&db, "SELECT a FROM t ORDER BY a LIMIT 3 OFFSET 2");
    assert_eq!(got.len(), 3);
    assert_eq!(got[0][0], "3");
}
