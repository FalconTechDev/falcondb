// Does a Parquet round trip preserve every row?
//
// The benchmark filtered 500 rows where it expected 500_000, so somewhere
// between writing a table and loading it back, 999 rows in 1000 went missing.
// This narrows down which side drops them.

use falcondb::{
    db::Database,
    storage::{
        column::ColumnBuilder,
        parquet_io::{read_parquet, write_parquet},
        schema::{Field, Schema},
        table::{RecordBatch, Table},
        types::DataType,
    },
};

const DEPTS: [&str; 5] = ["eng", "hr", "finance", "ops", "legal"];

/// The exact table the benchmark builds.
fn build_table(rows: usize) -> Table {
    let schema = Schema::new(vec![
        Field::required("id", DataType::Int64),
        Field::required("dept", DataType::Utf8),
        Field::required("score", DataType::Float64),
    ]);
    let batch = RecordBatch::new(
        schema.clone(),
        vec![
            ColumnBuilder::int64((0..rows).map(|i| Some(i as i64)).collect()),
            ColumnBuilder::utf8_from_str((0..rows).map(|i| Some(DEPTS[i % 5])).collect()),
            ColumnBuilder::float64((0..rows).map(|i| Some((i % 100) as f64 + 0.5)).collect()),
        ],
    )
    .unwrap();
    let mut table = Table::new("bench", schema);
    table.append_batch(batch).unwrap();
    table
}

#[test]
fn parquet_round_trip_keeps_every_row() {
    for rows in [1_000usize, 10_000, 100_000, 1_000_000] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.parquet");

        let table = build_table(rows);
        assert_eq!(table.row_count(), rows, "source table is wrong");

        write_parquet(&table.batches, &path).unwrap();

        let back = read_parquet(&path, "bench").unwrap();
        let got: usize = back.iter().map(|b| b.row_count()).sum();
        println!("rows={rows} -> read_parquet gave {got} rows in {} batches", back.len());

        assert_eq!(got, rows, "read_parquet lost rows for size {rows}");
    }
}

#[test]
fn load_parquet_keeps_every_row() {
    for rows in [1_000usize, 10_000, 100_000, 1_000_000] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.parquet");
        write_parquet(&build_table(rows).batches, &path).unwrap();

        let db = Database::in_memory();
        let reported = db.load_parquet(&path, "bench").unwrap();

        let tbl = db.get_table("bench").unwrap();
        println!(
            "rows={rows} -> load_parquet reported {reported}, table holds {} in {} batches",
            tbl.row_count(),
            tbl.batch_count()
        );

        assert_eq!(reported, rows, "load_parquet reported the wrong count");
        assert_eq!(tbl.row_count(), rows, "table lost rows for size {rows}");
    }
}

#[test]
fn sql_sees_every_loaded_row() {
    let rows = 1_000_000usize;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.parquet");
    write_parquet(&build_table(rows).batches, &path).unwrap();

    let db = Database::in_memory();
    db.load_parquet(&path, "bench").unwrap();

    let n = db.sql("SELECT id FROM bench").unwrap().row_count();
    assert_eq!(n, rows, "SELECT saw fewer rows than were loaded");
}
