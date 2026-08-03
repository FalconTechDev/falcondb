// FalconDB benchmark suite — fair 4-way comparison
//
// All engines read from the **same Parquet file** written to a temp directory
// before each benchmark group. Nobody has a pre-cached in-memory advantage:
// data loading is excluded from the timed region; only query execution is
// measured. For full_scan every engine physically iterates every value.
//
// ## FULL SCAN runs the same query on every engine
//
// It did not always. It used to ask FalconDB for `SELECT *` and walk every cell
// through a boxed ScalarValue, Polars for a one-column sum, DuckDB for whole
// materialised rows, and DataFusion for a bare aggregate — four different
// workloads reported side by side as though they were one. It is now
// `SELECT SUM(score)` everywhere.
//
// ## Why every FalconDB routine asserts its row count
//
// This benchmark reported the same ~4µs filter time at 10k, 100k and 1M rows for
// a long time. The times were real; the data was not. `Database::load_parquet`
// kept only the first batch of a Parquet file — about 1024 rows — while returning
// the file's full row count, so the table looked loaded and every FalconDB
// measurement ran against 1024 rows while the rivals read the whole dataset.
//
// Nothing in the harness could catch that, because a wrong-sized input produces a
// perfectly self-consistent measurement. What caught it was asserting the result:
// the filter must return exactly `rows / 2`, and it returned 500.
//
// Two things follow, and both are load-bearing:
//
//   * Assertions use `assert_eq!`, never `debug_assert_eq!`. Debug assertions are
//     compiled out of the bench profile, which is why the earlier `debug_assert`
//     version of this check never fired.
//   * `iter_batched` builds a fresh input in its untimed setup closure, so no
//     loop-invariant expression is left for LLVM to hoist out of the timing loop.
//     This matters more for FalconDB than for the rivals: FalconDB is ordinary
//     Rust the optimiser can inline into the loop, while Polars and DataFusion sit
//     behind a lazy or async boundary and DuckDB behind FFI into C++. An
//     optimisation that erased FalconDB's work would leave theirs untouched and
//     tilt the comparison our way.
//
// tests/diag_filter.rs times a single un-looped pass with no harness at all and
// asserts a memory-bandwidth floor. If it and this benchmark ever disagree by more
// than a small factor, believe the test.

use std::path::Path;

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use duckdb::Connection;
use polars::prelude::{col, lit, LazyFrame, ScanArgsParquet, SortMultipleOptions};
use datafusion::prelude::SessionContext;
use tokio::runtime::Runtime;

use falcondb::{
    db::Database,
    query::{
        aggregate::{AggExpr, GroupByExec},
        expr::{BinaryOp, Expr},
        select::{execute_select, SelectQuery},
        sort::{execute_sort, SortExpr},
    },
    storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        stream::ChunkedScanner,
        table::{RecordBatch, Table},
        types::{DataType, ScalarValue},
    },
};

// ─── Dataset helpers ──────────────────────────────────────────────────────────

const DEPTS: [&str; 5] = ["eng", "hr", "finance", "ops", "legal"];

/// Write a Parquet file and return its path (inside a tempdir that must outlive benchmarks).
fn write_parquet_file(rows: usize, dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("bench.parquet");
    let schema = Schema::new(vec![
        Field::required("id",    DataType::Int64),
        Field::required("dept",  DataType::Utf8),
        Field::required("score", DataType::Float64),
    ]);
    let batch = RecordBatch::new(schema.clone(), vec![
        ColumnBuilder::int64((0..rows).map(|i| Some(i as i64)).collect()),
        ColumnBuilder::utf8_from_str((0..rows).map(|i| Some(DEPTS[i % 5])).collect()),
        ColumnBuilder::float64((0..rows).map(|i| Some((i % 100) as f64 + 0.5)).collect()),
    ]).unwrap();
    let mut table = Table::new("bench", schema);
    table.append_batch(batch).unwrap();
    falcondb::storage::parquet_io::write_parquet(&table.batches, &path).unwrap();
    path
}

// ─── Per-engine loaders (called OUTSIDE the timed region) ────────────────────

fn load_falcon(path: &Path) -> Database {
    let db = Database::in_memory();
    db.load_parquet(path, "bench").unwrap();
    db
}

fn load_duckdb(path: &Path) -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(&format!(
        "CREATE TABLE bench AS SELECT * FROM read_parquet('{}')",
        path.display()
    )).unwrap();
    conn
}

fn load_polars(path: &Path) -> LazyFrame {
    polars::prelude::LazyFrame::scan_parquet(path, ScanArgsParquet::default()).unwrap()
}

fn load_datafusion(path: &Path, rt: &Runtime) -> SessionContext {
    let ctx  = SessionContext::new();
    let path = path.to_str().unwrap().to_string();
    rt.block_on(async {
        ctx.register_parquet("bench", &path, Default::default()).await.unwrap();
    });
    ctx
}

// ─── Result helpers ───────────────────────────────────────────────────────────

/// Total rows across a batch list. Returning a count from the timed closure makes
/// the work observable, so the result cannot be discarded.
fn count_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.row_count()).sum()
}

fn duck_count(conn: &Connection, sql: &str) -> usize {
    let mut stmt = conn.prepare(sql).unwrap();
    let mut iter = stmt.query([]).unwrap();
    let mut n = 0usize;
    while iter.next().unwrap().is_some() { n += 1; }
    n
}

// ─── Full scan ────────────────────────────────────────────────────────────────

fn bench_full_scan(c: &mut Criterion) {
    let mut g = c.benchmark_group("full_scan");
    let rt    = Runtime::new().unwrap();

    for &rows in &[10_000usize, 100_000, 1_000_000] {
        g.throughput(Throughput::Elements(rows as u64));
        let dir  = tempfile::tempdir().unwrap();
        let path = write_parquet_file(rows, &dir);

        // Pre-load (outside timed region).
        let db   = load_falcon(&path);
        let conn = load_duckdb(&path);
        let lf   = load_polars(&path);
        let ctx  = load_datafusion(&path, &rt);

        // Every engine runs the same query — `SELECT SUM(score)` — and returns the
        // sum. An earlier version asked each engine something different: FalconDB
        // walked every cell through a boxed ScalarValue, Polars summed one column,
        // DuckDB materialised whole rows, and DataFusion computed only the
        // aggregate. Those numbers measured four different workloads and were
        // reported as a comparison.
        //
        // A full aggregate is the honest shared shape here: it forces every engine
        // to read all `rows` values of one column and produce a single answer, with
        // no room to differ in how much of the result gets materialised.
        const SCAN_SQL: &str = "SELECT SUM(score) AS s FROM bench";

        // score = (i % 100) + 0.5, so each block of 100 rows sums to 5000.
        let expected_sum = (rows / 100) as f64 * 5000.0;

        g.bench_with_input(BenchmarkId::new("falcondb", rows), &db, |b, db| {
            b.iter_batched(
                || String::from(SCAN_SQL),
                |sql| {
                    let r = db.sql(&sql).unwrap();
                    let got = r.batches[0].column(0).get(0).unwrap().as_f64().unwrap();
                    assert!(
                        (got - expected_sum).abs() < 1.0,
                        "full scan summed {got}, expected {expected_sum}"
                    );
                    got
                },
                BatchSize::SmallInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("polars", rows), &lf, |b, lf| {
            b.iter(|| {
                let df = black_box(lf).clone()
                    .select([col("score").sum().alias("s")])
                    .collect().unwrap();
                df.column("s").unwrap().f64().unwrap().get(0).unwrap()
            });
        });

        g.bench_with_input(BenchmarkId::new("duckdb", rows), &conn, |b, c| {
            b.iter(|| {
                let mut stmt = black_box(c).prepare(SCAN_SQL).unwrap();
                let v: f64 = stmt.query_row([], |r| r.get(0)).unwrap();
                v
            });
        });

        g.bench_with_input(BenchmarkId::new("datafusion", rows), &ctx, |b, ctx| {
            b.iter(|| rt.block_on(async {
                let df = black_box(ctx).sql(black_box(SCAN_SQL)).await.unwrap();
                df.collect().await.unwrap().len()
            }));
        });
    }
    g.finish();
}

// ─── WHERE filter ─────────────────────────────────────────────────────────────

fn bench_filter(c: &mut Criterion) {
    let mut g = c.benchmark_group("filter_where");
    let rt    = Runtime::new().unwrap();

    for &rows in &[10_000usize, 100_000, 1_000_000] {
        g.throughput(Throughput::Elements(rows as u64));
        let dir  = tempfile::tempdir().unwrap();
        let path = write_parquet_file(rows, &dir);

        let db   = load_falcon(&path);
        let conn = load_duckdb(&path);
        let lf   = load_polars(&path);
        let ctx  = load_datafusion(&path, &rt);

        // score = (i % 100) + 0.5, so exactly half the rows pass the filter.
        let expected = rows / 2;

        // FalconDB: direct table scan with SIMD filter.
        //
        // `iter_batched` builds a fresh query per iteration in the untimed setup
        // closure, so there is no loop-invariant call for LLVM to hoist. The row
        // count is asserted with a real `assert_eq!`, not `debug_assert_eq!` —
        // debug assertions are compiled out of the bench profile, which is why
        // the earlier impossible numbers went unnoticed here.
        g.bench_with_input(BenchmarkId::new("falcondb", rows), &db, |b, db| {
            let tbl = db.get_table("bench").unwrap();
            b.iter_batched(
                || {
                    SelectQuery::scan_all().with_filter(Expr::BinaryOp {
                        left:  Box::new(Expr::Column("score".into())),
                        op:    BinaryOp::Gt,
                        right: Box::new(Expr::Literal(ScalarValue::Float64(Some(50.0)))),
                    })
                },
                |q| {
                    let out = execute_select(&*tbl, &q).unwrap();
                    let n = count_rows(&out);
                    assert_eq!(n, expected, "filter produced the wrong row count");
                    n
                },
                BatchSize::SmallInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("polars", rows), &lf, |b, lf| {
            b.iter(|| {
                black_box(lf).clone()
                    .filter(col("score").gt(lit(50.0f64)))
                    .collect().unwrap()
                    .height()
            });
        });

        g.bench_with_input(BenchmarkId::new("duckdb", rows), &conn, |b, c| {
            b.iter(|| duck_count(black_box(c), black_box("SELECT * FROM bench WHERE score > 50")));
        });

        g.bench_with_input(BenchmarkId::new("datafusion", rows), &ctx, |b, ctx| {
            b.iter(|| rt.block_on(async {
                let df = black_box(ctx)
                    .sql(black_box("SELECT * FROM bench WHERE score > 50")).await.unwrap();
                let out = df.collect().await.unwrap();
                out.iter().map(|b| b.num_rows()).sum::<usize>()
            }));
        });
    }
    g.finish();
}

// ─── GROUP BY + AGG ───────────────────────────────────────────────────────────

fn bench_group_by(c: &mut Criterion) {
    let mut g = c.benchmark_group("group_by_agg");
    let rt    = Runtime::new().unwrap();

    for &rows in &[10_000usize, 100_000, 1_000_000] {
        g.throughput(Throughput::Elements(rows as u64));
        let dir  = tempfile::tempdir().unwrap();
        let path = write_parquet_file(rows, &dir);

        let db   = load_falcon(&path);
        let conn = load_duckdb(&path);
        let lf   = load_polars(&path);
        let ctx  = load_datafusion(&path, &rt);

        let tbl  = db.get_table("bench").unwrap();
        let exec = GroupByExec::new(
            vec![Expr::Column("dept".into())],
            vec![
                AggExpr::count(Expr::Column("id".into()),    "cnt"),
                AggExpr::sum(Expr::Column("score".into()),   "total"),
                AggExpr::avg(Expr::Column("score".into()),   "avg"),
            ],
        );

        g.bench_with_input(BenchmarkId::new("falcondb", rows), &tbl, |b, t| {
            b.iter_batched(
                || exec.clone(),
                |e| {
                    let out = e.execute(&t.batches).unwrap();
                    // Five departments, so the grouped result has five rows.
                    assert_eq!(out.row_count(), DEPTS.len(), "group-by lost groups");
                    out.row_count()
                },
                BatchSize::SmallInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("polars", rows), &lf, |b, lf| {
            b.iter(|| {
                black_box(lf).clone()
                    .group_by([col("dept")])
                    .agg([
                        col("id").count().alias("cnt"),
                        col("score").sum().alias("total"),
                        col("score").mean().alias("avg"),
                    ])
                    .collect().unwrap()
                    .height()
            });
        });

        g.bench_with_input(BenchmarkId::new("duckdb", rows), &conn, |b, c| {
            b.iter(|| duck_count(black_box(c), black_box(
                "SELECT dept, COUNT(*) cnt, SUM(score) total, AVG(score) avg FROM bench GROUP BY dept"
            )));
        });

        g.bench_with_input(BenchmarkId::new("datafusion", rows), &ctx, |b, ctx| {
            b.iter(|| rt.block_on(async {
                let df = black_box(ctx).sql(black_box(
                    "SELECT dept, COUNT(*) cnt, SUM(score) total, AVG(score) avg FROM bench GROUP BY dept"
                )).await.unwrap();
                df.collect().await.unwrap().len()
            }));
        });
    }
    g.finish();
}

// ─── ORDER BY ─────────────────────────────────────────────────────────────────

fn bench_sort(c: &mut Criterion) {
    let mut g = c.benchmark_group("order_by");
    let rt    = Runtime::new().unwrap();

    for &rows in &[10_000usize, 100_000] {
        g.throughput(Throughput::Elements(rows as u64));
        let dir  = tempfile::tempdir().unwrap();
        let path = write_parquet_file(rows, &dir);

        let db   = load_falcon(&path);
        let conn = load_duckdb(&path);
        let lf   = load_polars(&path);
        let ctx  = load_datafusion(&path, &rt);

        let tbl = db.get_table("bench").unwrap();

        g.bench_with_input(BenchmarkId::new("falcondb", rows), &tbl, |b, t| {
            b.iter_batched(
                || vec![SortExpr::desc(Expr::Column("score".into()))],
                |keys| {
                    let out = execute_sort(&t.batches, &keys).unwrap();
                    assert_eq!(out.row_count(), rows, "sort lost rows");
                    out.row_count()
                },
                BatchSize::SmallInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("polars", rows), &lf, |b, lf| {
            b.iter(|| {
                black_box(lf).clone()
                    .sort(["score"], SortMultipleOptions::default().with_order_descending(true))
                    .collect().unwrap()
                    .height()
            });
        });

        g.bench_with_input(BenchmarkId::new("duckdb", rows), &conn, |b, c| {
            b.iter(|| duck_count(black_box(c), black_box("SELECT * FROM bench ORDER BY score DESC")));
        });

        g.bench_with_input(BenchmarkId::new("datafusion", rows), &ctx, |b, ctx| {
            b.iter(|| rt.block_on(async {
                let df = black_box(ctx)
                    .sql(black_box("SELECT * FROM bench ORDER BY score DESC")).await.unwrap();
                let out = df.collect().await.unwrap();
                out.iter().map(|b| b.num_rows()).sum::<usize>()
            }));
        });
    }
    g.finish();
}

// ─── End-to-end SQL pipeline ─────────────────────────────────────────────────

fn bench_sql_pipeline(c: &mut Criterion) {
    let mut g = c.benchmark_group("sql_pipeline");
    const SQL: &str =
        "SELECT dept, COUNT(*) as cnt, AVG(score) as avg \
         FROM bench GROUP BY dept ORDER BY avg DESC";
    let rt = Runtime::new().unwrap();

    for &rows in &[10_000usize, 100_000] {
        g.throughput(Throughput::Elements(rows as u64));
        let dir  = tempfile::tempdir().unwrap();
        let path = write_parquet_file(rows, &dir);

        let db   = load_falcon(&path);
        let conn = load_duckdb(&path);
        let lf   = load_polars(&path);
        let ctx  = load_datafusion(&path, &rt);

        g.bench_with_input(BenchmarkId::new("falcondb", rows), &db, |b, db| {
            b.iter_batched(
                || String::from(SQL),
                |sql| {
                    let n = db.sql(&sql).unwrap().row_count();
                    assert_eq!(n, DEPTS.len(), "pipeline lost groups");
                    n
                },
                BatchSize::SmallInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("polars", rows), &lf, |b, lf| {
            b.iter(|| {
                black_box(lf).clone()
                    .group_by([col("dept")])
                    .agg([
                        col("id").count().alias("cnt"),
                        col("score").mean().alias("avg"),
                    ])
                    .sort(["avg"], SortMultipleOptions::default().with_order_descending(true))
                    .collect().unwrap()
                    .height()
            });
        });

        g.bench_with_input(BenchmarkId::new("duckdb", rows), &conn, |b, c| {
            b.iter(|| duck_count(black_box(c), black_box(SQL)));
        });

        g.bench_with_input(BenchmarkId::new("datafusion", rows), &ctx, |b, ctx| {
            b.iter(|| rt.block_on(async {
                let df = black_box(ctx).sql(black_box(SQL)).await.unwrap();
                df.collect().await.unwrap().len()
            }));
        });
    }
    g.finish();
}

// ─── Chunked streaming (billion-row pattern) ──────────────────────────────────

fn bench_streaming(c: &mut Criterion) {
    let mut g = c.benchmark_group("streaming_filter");

    for &rows in &[100_000usize, 1_000_000] {
        g.throughput(Throughput::Elements(rows as u64));
        let dir  = tempfile::tempdir().unwrap();
        let path = write_parquet_file(rows, &dir);

        let db      = load_falcon(&path);
        let tbl     = db.get_table("bench").unwrap();
        let filter  = Expr::BinaryOp {
            left:  Box::new(Expr::Column("score".into())),
            op:    BinaryOp::Gt,
            right: Box::new(Expr::Literal(ScalarValue::Float64(Some(50.0)))),
        };
        let expected = rows / 2;

        g.bench_with_input(BenchmarkId::new("falcondb_chunked_64k", rows), &tbl, |b, t| {
            b.iter_batched(
                || (ChunkedScanner::new(65_536), filter.clone()),
                |(scanner, f)| {
                    let n = count_rows(&scanner.filter(&**t, &f).unwrap());
                    assert_eq!(n, expected, "chunked filter produced the wrong row count");
                    n
                },
                BatchSize::SmallInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("falcondb_direct", rows), &tbl, |b, t| {
            b.iter_batched(
                || SelectQuery::scan_all().with_filter(filter.clone()),
                |q| {
                    let n = count_rows(&execute_select(&**t, &q).unwrap());
                    assert_eq!(n, expected, "direct filter produced the wrong row count");
                    n
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

criterion_group!(benches,
    bench_full_scan,
    bench_filter,
    bench_group_by,
    bench_sort,
    bench_sql_pipeline,
    bench_streaming,
);
criterion_main!(benches);
