//! What shape does a Parquet load actually produce?
//!
//! The aggregate kernels decide serial-vs-parallel per batch (`dense.len() >=
//! PAR_SUM_THRESHOLD`), so the batch split — not the row count — is what picks the
//! path. `full_scan/100k` costs 0.63 ns/row against 0.17 ns/row at 1M, which only
//! makes sense if 100k is landing on the serial side.

use falcondb::{
    db::Database,
    storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::{RecordBatch, Table},
        types::DataType,
    },
};

const DEPTS: [&str; 5] = ["eng", "sales", "hr", "ops", "legal"];

fn write_and_load(rows: usize, dir: &tempfile::TempDir) -> Database {
    let path = dir.path().join("shape.parquet");
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
    let mut table = Table::new("shape", schema);
    table.append_batch(batch).unwrap();
    falcondb::storage::parquet_io::write_parquet(&table.batches, &path).unwrap();

    let db = Database::in_memory();
    db.load_parquet(&path, "shape").unwrap();
    db
}

#[test]
fn report_batch_shape_after_parquet_load() {
    for rows in [10_000usize, 100_000, 1_000_000] {
        let dir = tempfile::tempdir().unwrap();
        let db = write_and_load(rows, &dir);
        let table = db.get_table("shape").unwrap();
        let sizes: Vec<usize> = table.batches.iter().map(|b| b.row_count()).collect();
        let total: usize = sizes.iter().sum();
        let below = sizes.iter().filter(|&&s| s < 65_536).count();
        println!(
            "rows={rows:>9}  batches={:>3}  sizes={:?}  below_par_threshold={below}",
            sizes.len(),
            &sizes[..sizes.len().min(8)],
        );
        assert_eq!(total, rows, "load lost rows");
    }
}
