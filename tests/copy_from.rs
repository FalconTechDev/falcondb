// COPY <table> FROM '<path>' — bulk-loading a file through SQL.

use std::io::Write;

use falcondb::db::Database;
use tempfile::NamedTempFile;

/// A CSV file kept alive for the duration of a test.
fn csv_file(body: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{body}").unwrap();
    f.flush().unwrap();
    f
}

fn path_of(f: &NamedTempFile) -> String {
    // Backslashes would be read as escapes inside the quoted SQL literal.
    f.path().to_str().unwrap().replace('\\', "/")
}

#[test]
fn copy_loads_a_csv() {
    let f = csv_file("id,v\n1,10\n2,20\n3,30\n");
    let db = Database::in_memory();

    let r = db.sql(&format!("COPY t FROM '{}'", path_of(&f))).unwrap();
    assert_eq!(r.to_rows()[0][0], "3", "COPY should report rows loaded");

    assert_eq!(db.sql("SELECT id FROM t").unwrap().row_count(), 3);
}

#[test]
fn copied_data_is_queryable() {
    let f = csv_file("id,v\n1,10\n2,20\n3,30\n");
    let db = Database::in_memory();
    db.sql(&format!("COPY t FROM '{}'", path_of(&f))).unwrap();

    let total = db.sql("SELECT SUM(v) AS s FROM t").unwrap().to_rows()[0][0].clone();
    assert_eq!(total.parse::<f64>().unwrap() as i64, 60);
}

#[test]
fn copy_replaces_existing_contents() {
    let first = csv_file("id,v\n1,10\n2,20\n");
    let second = csv_file("id,v\n9,90\n");
    let db = Database::in_memory();

    db.sql(&format!("COPY t FROM '{}'", path_of(&first))).unwrap();
    db.sql(&format!("COPY t FROM '{}'", path_of(&second))).unwrap();

    let rows = db.sql("SELECT id FROM t").unwrap().to_rows();
    assert_eq!(rows.len(), 1, "COPY should replace, not append");
    assert_eq!(rows[0][0], "9");
}

#[test]
fn explicit_csv_format_is_accepted() {
    let f = csv_file("id\n1\n2\n");
    let db = Database::in_memory();
    db.sql(&format!("COPY t FROM '{}' (FORMAT CSV)", path_of(&f))).unwrap();
    assert_eq!(db.sql("SELECT id FROM t").unwrap().row_count(), 2);
}

#[test]
fn copy_round_trips_through_parquet() {
    let src = csv_file("id,v\n1,10\n2,20\n3,30\n");
    let db = Database::in_memory();
    db.load_csv(src.path(), "src").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let pq = dir.path().join("out.parquet");
    let table = db.get_table("src").unwrap();
    falcondb::storage::parquet_io::write_parquet(&table.batches, &pq).unwrap();

    let pq_path = pq.to_str().unwrap().replace('\\', "/");
    db.sql(&format!("COPY loaded FROM '{pq_path}'")).unwrap();

    assert_eq!(db.sql("SELECT id FROM loaded").unwrap().row_count(), 3);
    let total = db.sql("SELECT SUM(v) AS s FROM loaded").unwrap().to_rows()[0][0].clone();
    assert_eq!(total.parse::<f64>().unwrap() as i64, 60);
}

#[test]
fn missing_file_is_an_error() {
    let db = Database::in_memory();
    assert!(db.sql("COPY t FROM '/no/such/file.csv'").is_err());
}

#[test]
fn copy_to_is_rejected_rather_than_read_as_from() {
    let db = Database::in_memory();
    let err = match db.sql("COPY t TO '/tmp/out.csv'") {
        Err(e) => e.to_string(),
        Ok(_) => panic!("COPY ... TO should not be accepted"),
    };
    assert!(err.contains("TO"), "unhelpful error: {err}");
}

#[test]
fn copy_without_from_is_an_error() {
    let db = Database::in_memory();
    assert!(db.sql("COPY t").is_err());
}

#[test]
fn unquoted_path_is_an_error() {
    let db = Database::in_memory();
    assert!(db.sql("COPY t FROM /tmp/x.csv").is_err());
}
