use falcondb::db::Database;
use falcondb::storage::column::ColumnBuilder;
use falcondb::storage::schema::{Field, Schema};
use falcondb::storage::table::RecordBatch;
use falcondb::storage::types::DataType;

fn main() {
    let big = 9_007_199_254_740_993i64; // 2^53 + 1, not representable in f64
    let batch = RecordBatch::new(
        Schema::new(vec![Field::required("n", DataType::Int64)]),
        vec![ColumnBuilder::int64(vec![Some(big), Some(1), Some(-5)])],
    ).unwrap();
    let db = Database::in_memory();
    db.create_table_from_batches("t", vec![batch]).unwrap();

    for sql in [
        "SELECT n FROM t",
        "SELECT n * 1 FROM t",
        "SELECT n + 0 FROM t",
        "SELECT n - 0 FROM t",
        "SELECT SUM(n) FROM t",
        "SELECT MIN(n) FROM t",
        "SELECT MAX(n) FROM t",
        "SELECT COUNT(n) FROM t",
        "SELECT AVG(n) FROM t",
        "SELECT n / 2 FROM t",
    ] {
        match db.sql(sql) {
            Ok(r) => {
                let c = r.batches[0].column(0);
                let vals: Vec<String> = (0..r.batches[0].row_count().min(3))
                    .map(|i| format!("{:?}", c.get(i).unwrap()))
                    .collect();
                println!("{sql:<28} -> {:?}  {}", c.data_type(), vals.join(", "));
            }
            Err(e) => println!("{sql:<28} -> ERROR {e}"),
        }
    }
}
