// Projection pruning of a join's output columns.
//
// A join now builds only the columns some later stage reads. The failure mode is
// not a crash: dropping a column the query does read gives a wrong answer or a
// "column not found" where the data was in fact present. So these tests take the
// query shapes that consume join output — projection, WHERE, GROUP BY, ORDER BY,
// aggregates over each side — and check the answers, not just that nothing panics.

use std::io::Write;

use falcondb::db::Database;
use tempfile::NamedTempFile;

/// `left` has 40 rows keyed 0..40; `right` has every fourth key, so a join matches
/// 10 rows. Values differ per side so a mixed-up column shows as a wrong number.
fn join_db() -> (Database, NamedTempFile, NamedTempFile) {
    let mut fl = NamedTempFile::new().unwrap();
    writeln!(fl, "k,lv,tag").unwrap();
    for i in 0..40 {
        writeln!(fl, "{i},{},L{}", i * 10, i % 3).unwrap();
    }
    fl.flush().unwrap();

    let mut fr = NamedTempFile::new().unwrap();
    writeln!(fr, "rk,rv").unwrap();
    for i in (0..40).step_by(4) {
        writeln!(fr, "{i},{}", i * 100).unwrap();
    }
    fr.flush().unwrap();

    let db = Database::in_memory();
    db.load_csv(fl.path(), "l").unwrap();
    db.load_csv(fr.path(), "r").unwrap();
    (db, fl, fr)
}

fn one_i64(db: &Database, sql: &str) -> i64 {
    let res = db.sql(sql).unwrap();
    res.batches[0].column(0).get(0).unwrap().as_f64().unwrap() as i64
}

#[test]
fn count_star_over_a_join_is_not_broken_by_pruning() {
    let (db, _a, _b) = join_db();
    // Reads no columns at all — the case pruning strips furthest.
    assert_eq!(
        one_i64(&db, "SELECT COUNT(*) AS n FROM l JOIN r ON l.k = r.rk"),
        10
    );
}

#[test]
fn aggregates_still_see_the_column_they_name() {
    let (db, _a, _b) = join_db();
    // Matching keys are 0,4,...,36 — lv = k*10, rv = k*100.
    let expect_lv: i64 = (0..40).step_by(4).map(|k| k as i64 * 10).sum();
    let expect_rv: i64 = (0..40).step_by(4).map(|k| k as i64 * 100).sum();

    assert_eq!(
        one_i64(&db, "SELECT SUM(lv) AS n FROM l JOIN r ON l.k = r.rk"),
        expect_lv,
        "left column pruned away"
    );
    assert_eq!(
        one_i64(&db, "SELECT SUM(rv) AS n FROM l JOIN r ON l.k = r.rk"),
        expect_rv,
        "right column pruned away"
    );
}

#[test]
fn a_column_used_only_in_where_survives() {
    let (db, _a, _b) = join_db();
    // `lv` appears nowhere in the projection, so a naive "keep what is selected"
    // would drop it and the filter would have nothing to read.
    let expected = (0..40)
        .step_by(4)
        .filter(|k| *k as i64 * 10 > 100)
        .count() as i64;
    assert_eq!(
        one_i64(
            &db,
            "SELECT COUNT(*) AS n FROM l JOIN r ON l.k = r.rk WHERE lv > 100"
        ),
        expected
    );
}

/// A WHERE over a *right-hand* column after a join is not supported.
///
/// Pre-existing, and unrelated to projection pruning — it fails the same way with
/// pruning disabled. The filter is applied to the left table during the scan, so a
/// right-side column is not in scope yet. Pinned here so the limitation is visible
/// and so that whoever fixes it finds a test to update rather than a silent gap.
#[test]
fn where_on_a_right_side_column_is_a_known_gap() {
    let (db, _a, _b) = join_db();
    let msg = match db.sql("SELECT COUNT(*) AS n FROM l JOIN r ON l.k = r.rk WHERE rv > 1000") {
        Ok(_) => panic!("right-side WHERE now works — update this test and the comment"),
        Err(e) => format!("{e:?}"),
    };
    assert!(msg.contains("rv"), "expected a column-not-found for 'rv', got {msg}");
}

#[test]
fn a_column_used_only_in_group_by_survives() {
    let (db, _a, _b) = join_db();
    let res = db
        .sql("SELECT tag, COUNT(*) AS n FROM l JOIN r ON l.k = r.rk GROUP BY tag")
        .unwrap();
    // Matching keys 0,4,8,...,36 give tags L0,L1,L2 in some order.
    assert_eq!(res.row_count(), 3, "grouping column pruned away");
    let total: i64 = (0..res.row_count())
        .map(|i| res.batches[0].column(1).get(i).unwrap().as_f64().unwrap() as i64)
        .sum();
    assert_eq!(total, 10);
}

#[test]
fn a_column_used_only_in_order_by_survives() {
    let (db, _a, _b) = join_db();
    let res = db
        .sql("SELECT lv FROM l JOIN r ON l.k = r.rk ORDER BY rv DESC")
        .unwrap();
    assert_eq!(res.row_count(), 10);
    // rv descends with k, so lv must too.
    let first = res.batches[0].column(0).get(0).unwrap().as_f64().unwrap() as i64;
    assert_eq!(first, 360, "ordering column pruned away, rows came back unsorted");
}

#[test]
fn select_star_over_a_join_keeps_every_column() {
    let (db, _a, _b) = join_db();
    let res = db.sql("SELECT * FROM l JOIN r ON l.k = r.rk").unwrap();
    assert_eq!(res.row_count(), 10);
    // SELECT * cannot be pruned: k, lv, tag, rk, rv.
    assert_eq!(res.batches[0].column_count(), 5, "SELECT * lost columns");
}

/// Trimming the join inputs must not change which columns come back, or their names.
///
/// A right column keeps its own name unless the left side has one to match, in which
/// case it becomes `right.x`. That decision reads the left schema, so dropping a
/// left column before the join could silently rename a result column. Both sides
/// here have a `k`, which is exactly the case that must not be trimmed.
#[test]
fn shared_column_names_survive_input_trimming() {
    let mut fl = NamedTempFile::new().unwrap();
    writeln!(fl, "k,v").unwrap();
    for i in 0..40 {
        writeln!(fl, "{i},{}", i * 10).unwrap();
    }
    fl.flush().unwrap();

    let mut fr = NamedTempFile::new().unwrap();
    writeln!(fr, "k,v").unwrap();
    for i in (0..40).step_by(4) {
        writeln!(fr, "{i},{}", i * 100).unwrap();
    }
    fr.flush().unwrap();

    let db = Database::in_memory();
    db.load_csv(fl.path(), "l").unwrap();
    db.load_csv(fr.path(), "r").unwrap();

    // `v` exists on both sides: the left one is meant here, and it must not be
    // trimmed away or replaced by the right one.
    let expect: i64 = (0..40).step_by(4).map(|k| k as i64 * 10).sum();
    let res = db
        .sql("SELECT SUM(v) AS n FROM l JOIN r ON l.k = r.k")
        .unwrap();
    let got = res.batches[0].column(0).get(0).unwrap().as_f64().unwrap() as i64;
    assert_eq!(got, expect, "trimming picked the wrong side's column");

    // And a plain projection still sees both, under their join-time names.
    let star = db.sql("SELECT * FROM l JOIN r ON l.k = r.k").unwrap();
    let names = star.batches[0].schema.column_names();
    assert!(names.contains(&"v"), "left v missing: {names:?}");
    assert!(names.contains(&"right.v"), "right v was not renamed: {names:?}");
}
