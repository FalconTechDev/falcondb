// Diagnostic: is the FalconDB filter benchmark measuring real work?
//
// Builds the same dataset the benchmark uses, runs the same filter, and prints
// the result row count plus a wall-clock timing. If ~500k rows come back in a
// few microseconds, the work is being elided; if the time scales with N, it is real.

use std::time::Instant;

use falcondb::{
    query::{
        expr::{BinaryOp, Expr},
        select::{execute_select, SelectQuery},
    },
    storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::{RecordBatch, Table},
        types::{DataType, ScalarValue},
    },
};

const DEPTS: [&str; 5] = ["eng", "hr", "finance", "ops", "legal"];

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
fn filter_does_real_work() {
    let query = SelectQuery::scan_all().with_filter(Expr::BinaryOp {
        left: Box::new(Expr::Column("score".into())),
        op: BinaryOp::Gt,
        right: Box::new(Expr::Literal(ScalarValue::Float64(Some(50.0)))),
    });

    for rows in [10_000usize, 100_000, 1_000_000] {
        let table = build_table(rows);
        println!("rows={rows} batches={}", table.batches.len());

        // Warm up, then time a single pass.
        let _ = execute_select(&table, &query).unwrap();
        let t0 = Instant::now();
        let out = execute_select(&table, &query).unwrap();
        let elapsed = t0.elapsed();

        let produced: usize = out.iter().map(|b| b.row_count()).sum();
        println!(
            "  -> {produced} rows out, {:?} ({:.1} ns/row)",
            elapsed,
            elapsed.as_nanos() as f64 / rows as f64
        );

        // score = (i % 100) + 0.5, so exactly half the rows exceed 50.
        assert_eq!(produced, rows / 2, "filter returned the wrong row count");

        // Sanity floor. Emitting `rows/2` output rows across three columns is
        // 12 bytes per row of pure stores; at 100 GB/s — far above what any
        // single machine sustains — that still cannot happen faster than this.
        // A measurement below the floor means the work was optimised away, not
        // that it got fast.
        let floor_ns = (produced as f64 * 12.0) / 100.0;
        assert!(
            elapsed.as_nanos() as f64 > floor_ns,
            "filter over {rows} rows reported {elapsed:?}, below the {floor_ns:.0}ns \
             memory-bandwidth floor — the work is being elided"
        );
    }
}
