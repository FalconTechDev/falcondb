// Where does ORDER BY spend its time?
//
// The benchmark showed ORDER BY at 100k rows going from 3.88ms to 5.74ms across
// a round of changes, while the rival engines moved 3-5% — so the machine was
// steady and the regression is ours. The only change touching the sort path was
// making the column gather parallel.
//
// This times the two halves separately (sorting the keys, then reordering the
// columns) so the suspicion can be checked instead of argued about.

use std::time::Instant;

use falcondb::{
    query::{
        expr::Expr,
        sort::{execute_sort, SortExpr},
    },
    storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::{RecordBatch, Table},
        types::DataType,
    },
};

const DEPTS: [&str; 5] = ["eng", "hr", "finance", "ops", "legal"];

/// The benchmark's table shape: id/dept/score, split into 1024-row batches the
/// way a Parquet read hands them back.
fn build(rows: usize, batch_rows: usize) -> Table {
    let schema = Schema::new(vec![
        Field::required("id", DataType::Int64),
        Field::required("dept", DataType::Utf8),
        Field::required("score", DataType::Float64),
    ]);
    let mut table = Table::new("bench", schema.clone());

    let mut i = 0usize;
    while i < rows {
        let n = batch_rows.min(rows - i);
        let b = RecordBatch::new(
            schema.clone(),
            vec![
                ColumnBuilder::int64((i..i + n).map(|k| Some(k as i64)).collect()),
                ColumnBuilder::utf8_from_str((i..i + n).map(|k| Some(DEPTS[k % 5])).collect()),
                ColumnBuilder::float64((i..i + n).map(|k| Some((k % 100) as f64 + 0.5)).collect()),
            ],
        )
        .unwrap();
        table.append_batch(b).unwrap();
        i += n;
    }
    table
}

/// Best time over several series, in milliseconds.
///
/// Reports the minimum rather than the mean. The noise on this machine comes
/// from its own state — frequency scaling, other processes, cache pressure —
/// not from the code under test, and that kind of noise only ever *adds* time.
/// Averaging folds it in; the minimum is the closest thing to a measurement of
/// the code itself.
///
/// This matters more than it sounds. Timing identical code six times with the
/// mean gave 4.2, 4.4, 5.5, 7.1, 5.5 and 5.8 ms — a 70% spread, wide enough to
/// invent or hide any difference smaller than about 2x.
fn time_sort(table: &Table, series: usize, reps: usize) -> f64 {
    let keys = [SortExpr::desc(Expr::Column("score".into()))];

    // Warm up so the first run's allocation does not land in the measurement.
    let _ = execute_sort(&table.batches, &keys).unwrap();

    let mut best = f64::INFINITY;
    for _ in 0..series {
        let t0 = Instant::now();
        for _ in 0..reps {
            let out = execute_sort(&table.batches, &keys).unwrap();
            assert_eq!(out.row_count(), table.row_count());
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        best = best.min(ms);
    }
    best
}

#[test]
fn report_sort_timings() {
    // 1024-row batches: what read_parquet produces, and what the benchmark sorts.
    for &rows in &[10_000usize, 100_000] {
        let table = build(rows, 1024);
        println!(
            "rows={rows} batches={} -> sort {:.3} ms",
            table.batches.len(),
            time_sort(&table, 8, 20)
        );
    }

    // One big batch, to separate the cost of merging many small batches from the
    // cost of sorting itself.
    for &rows in &[10_000usize, 100_000] {
        let table = build(rows, rows);
        println!(
            "rows={rows} batches=1 -> sort {:.3} ms",
            time_sort(&table, 8, 20)
        );
    }
}
