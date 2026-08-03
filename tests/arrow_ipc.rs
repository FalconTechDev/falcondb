// Arrow IPC export / import.
//
// IPC is the handoff format for the Arrow ecosystem, so the property that
// matters is a faithful round trip: schema, values, NULLs and batch boundaries
// all have to survive.

use std::io::Write;

use falcondb::{
    db::Database,
    storage::arrow_ipc::{
        read_ipc_file, read_ipc_stream, write_ipc_file, write_ipc_stream_to_vec,
    },
};
use tempfile::NamedTempFile;

fn db_from_csv(body: &str) -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{body}").unwrap();
    f.flush().unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();
    (db, f)
}

fn rows_of(batches: &[falcondb::storage::table::RecordBatch]) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.row_count() {
            out.push(
                (0..b.column_count())
                    .map(|c| {
                        b.column(c)
                            .get(r)
                            .map(|v| v.to_string())
                            .unwrap_or_else(|_| "NULL".into())
                    })
                    .collect(),
            );
        }
    }
    out
}

#[test]
fn file_format_round_trips() {
    let (db, _f) = db_from_csv("id,name,score\n1,alice,9.5\n2,bob,7.25\n3,carol,8.0\n");
    let table = db.get_table("t").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.arrow");
    write_ipc_file(&table.batches, &path).unwrap();

    let back = read_ipc_file(&path, "t").unwrap();
    assert_eq!(rows_of(&back), rows_of(&table.batches));
}

#[test]
fn stream_format_round_trips() {
    let (db, _f) = db_from_csv("id,name\n1,alice\n2,bob\n");
    let table = db.get_table("t").unwrap();

    let buf = write_ipc_stream_to_vec(&table.batches).unwrap();
    let back = read_ipc_stream(buf.as_slice(), "t").unwrap();
    assert_eq!(rows_of(&back), rows_of(&table.batches));
}

#[test]
fn schema_survives_the_round_trip() {
    let (db, _f) = db_from_csv("id,name,score\n1,alice,9.5\n");
    let table = db.get_table("t").unwrap();

    let buf = write_ipc_stream_to_vec(&table.batches).unwrap();
    let back = read_ipc_stream(buf.as_slice(), "t").unwrap();

    let before: Vec<_> = table.batches[0].schema.fields.iter().map(|f| f.name.clone()).collect();
    let after: Vec<_> = back[0].schema.fields.iter().map(|f| f.name.clone()).collect();
    assert_eq!(before, after);
}

#[test]
fn nulls_survive_the_round_trip() {
    let (db, _f) = db_from_csv("id,name\n1,alice\n2,\n3,carol\n");
    let table = db.get_table("t").unwrap();

    let buf = write_ipc_stream_to_vec(&table.batches).unwrap();
    let back = read_ipc_stream(buf.as_slice(), "t").unwrap();
    assert_eq!(rows_of(&back), rows_of(&table.batches));
}

#[test]
fn exported_data_is_queryable_after_reimport() {
    let (db, _f) = db_from_csv("id,v\n1,10\n2,20\n3,30\n");
    let table = db.get_table("t").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.arrow");
    write_ipc_file(&table.batches, &path).unwrap();

    let back = read_ipc_file(&path, "reloaded").unwrap();
    let db2 = Database::in_memory();
    db2.create_table_from_batches("reloaded", back).unwrap();

    let total = db2.sql("SELECT SUM(v) AS s FROM reloaded").unwrap().to_rows()[0][0].clone();
    assert_eq!(total.parse::<f64>().unwrap() as i64, 60);
}

#[test]
fn multiple_batches_survive() {
    let (db, _f) = db_from_csv("id,v\n1,1\n");
    for i in 2..12 {
        db.sql(&format!("INSERT INTO t VALUES ({i}, {i})")).unwrap();
    }
    let table = db.get_table("t").unwrap();
    assert!(table.batch_count() > 1, "test needs more than one batch");

    let buf = write_ipc_stream_to_vec(&table.batches).unwrap();
    let back = read_ipc_stream(buf.as_slice(), "t").unwrap();
    assert_eq!(rows_of(&back), rows_of(&table.batches));
}

#[test]
fn writing_no_batches_is_an_error() {
    assert!(write_ipc_stream_to_vec(&[]).is_err());
}

#[test]
fn a_stream_is_not_readable_as_a_file() {
    // The two framings differ; reading one as the other must fail loudly rather
    // than return partial data.
    let (db, _f) = db_from_csv("id\n1\n");
    let table = db.get_table("t").unwrap();

    let buf = write_ipc_stream_to_vec(&table.batches).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream.arrow");
    std::fs::write(&path, &buf).unwrap();

    assert!(read_ipc_file(&path, "t").is_err());
}
