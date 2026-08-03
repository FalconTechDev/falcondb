// WAL durability: do mutations survive a process restart?
//
// The recovery contract is: open a database on the WAL path, load the same base
// tables, then call `recover()` to replay the journal. Each test simulates a
// crash by dropping the first Database without a clean shutdown and opening a
// second one over the same files.

use std::io::Write;

use falcondb::db::Database;
use tempfile::{NamedTempFile, TempDir};

/// A base CSV plus a WAL path, both living as long as the returned dir.
struct Fixture {
    _dir: TempDir,
    csv:  std::path::PathBuf,
    wal:  std::path::PathBuf,
}

fn fixture(csv_body: &str) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("base.csv");
    let mut f = std::fs::File::create(&csv).unwrap();
    write!(f, "{csv_body}").unwrap();
    f.flush().unwrap();
    Fixture { csv, wal: dir.path().join("t.wal"), _dir: dir }
}

/// Open a database on the fixture, load the base table, and replay the WAL.
fn reopen(fx: &Fixture) -> Database {
    let db = Database::with_wal(&fx.wal).unwrap();
    db.load_csv(&fx.csv, "t").unwrap();
    db.recover().unwrap();
    db
}

fn ids(db: &Database) -> Vec<String> {
    let mut v: Vec<String> = db
        .sql("SELECT id FROM t")
        .unwrap()
        .to_rows()
        .into_iter()
        .map(|r| r[0].clone())
        .collect();
    v.sort();
    v
}

#[test]
fn insert_survives_a_restart() {
    let fx = fixture("id,v\n1,10\n2,20\n");

    {
        let db = reopen(&fx);
        db.sql("INSERT INTO t VALUES (3, 30)").unwrap();
        assert_eq!(db.sql("SELECT id FROM t").unwrap().row_count(), 3);
    } // crash: no clean shutdown

    let db = reopen(&fx);
    assert_eq!(db.sql("SELECT id FROM t").unwrap().row_count(), 3);
    assert!(ids(&db).contains(&"3".to_string()));
}

#[test]
fn delete_survives_a_restart() {
    let fx = fixture("id,v\n1,10\n2,20\n3,30\n");

    {
        let db = reopen(&fx);
        db.sql("DELETE FROM t WHERE id = 2").unwrap();
    }

    let db = reopen(&fx);
    assert_eq!(ids(&db), vec!["1", "3"]);
}

#[test]
fn update_survives_a_restart() {
    let fx = fixture("id,v\n1,10\n2,20\n");

    {
        let db = reopen(&fx);
        db.sql("UPDATE t SET v = 99 WHERE id = 1").unwrap();
    }

    let db = reopen(&fx);
    let r = db.sql("SELECT v FROM t WHERE id = 1").unwrap();
    assert_eq!(r.to_rows()[0][0].parse::<f64>().unwrap(), 99.0);
}

#[test]
fn several_mutations_replay_in_order() {
    let fx = fixture("id,v\n1,1\n");

    {
        let db = reopen(&fx);
        db.sql("INSERT INTO t VALUES (2, 2)").unwrap();
        db.sql("INSERT INTO t VALUES (3, 3)").unwrap();
        db.sql("UPDATE t SET v = 100 WHERE id = 2").unwrap();
        db.sql("DELETE FROM t WHERE id = 3").unwrap();
    }

    let db = reopen(&fx);
    assert_eq!(ids(&db), vec!["1", "2"]);
    let v = db.sql("SELECT v FROM t WHERE id = 2").unwrap();
    assert_eq!(v.to_rows()[0][0].parse::<f64>().unwrap(), 100.0);
}

#[test]
fn recovery_is_idempotent_across_two_restarts() {
    let fx = fixture("id,v\n1,10\n");

    {
        let db = reopen(&fx);
        db.sql("INSERT INTO t VALUES (2, 20)").unwrap();
    }

    let first = ids(&reopen(&fx));
    let second = ids(&reopen(&fx));
    assert_eq!(
        first, second,
        "replaying the same WAL twice produced different data"
    );
    assert_eq!(second, vec!["1", "2"]);
}

#[test]
fn recover_reports_how_many_entries_it_replayed() {
    let fx = fixture("id,v\n1,10\n");

    {
        let db = reopen(&fx);
        db.sql("INSERT INTO t VALUES (2, 20)").unwrap();
        db.sql("INSERT INTO t VALUES (3, 30)").unwrap();
    }

    let db = Database::with_wal(&fx.wal).unwrap();
    db.load_csv(&fx.csv, "t").unwrap();
    assert_eq!(db.recover().unwrap(), 2);
}

#[test]
fn reads_are_not_journalled() {
    let fx = fixture("id,v\n1,10\n");

    {
        let db = reopen(&fx);
        db.sql("SELECT * FROM t").unwrap();
        db.sql("SELECT SUM(v) AS s FROM t").unwrap();
    }

    let db = Database::with_wal(&fx.wal).unwrap();
    db.load_csv(&fx.csv, "t").unwrap();
    assert_eq!(db.recover().unwrap(), 0, "a SELECT was written to the WAL");
}

#[test]
fn recovery_without_a_wal_is_a_no_op() {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "id\n1\n").unwrap();
    f.flush().unwrap();

    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();
    assert_eq!(db.recover().unwrap(), 0);
}

#[test]
fn queries_work_normally_after_recovery() {
    let fx = fixture("id,v\n1,10\n2,20\n");

    {
        let db = reopen(&fx);
        db.sql("INSERT INTO t VALUES (3, 30)").unwrap();
    }

    let db = reopen(&fx);
    let total = db.sql("SELECT SUM(v) AS s FROM t").unwrap().to_rows()[0][0].clone();
    assert_eq!(total.parse::<f64>().unwrap(), 60.0);
    assert_eq!(db.sql("SELECT id FROM t WHERE v > 15").unwrap().row_count(), 2);
}

#[test]
fn rolled_back_transaction_is_not_replayed() {
    let fx = fixture("id,v\n1,10\n");

    {
        let db = reopen(&fx);
        db.sql("BEGIN").unwrap();
        db.sql("INSERT INTO t VALUES (2, 20)").unwrap();
        db.sql("ROLLBACK").unwrap();
        assert_eq!(db.sql("SELECT id FROM t").unwrap().row_count(), 1);
    }

    let db = reopen(&fx);
    assert_eq!(
        ids(&db),
        vec!["1"],
        "a rolled-back insert came back after recovery"
    );
}

#[test]
fn committed_transaction_survives_a_restart() {
    let fx = fixture("id,v\n1,10\n");

    {
        let db = reopen(&fx);
        db.sql("BEGIN").unwrap();
        db.sql("INSERT INTO t VALUES (2, 20)").unwrap();
        db.sql("COMMIT").unwrap();
    }

    let db = reopen(&fx);
    assert_eq!(ids(&db), vec!["1", "2"]);
}
