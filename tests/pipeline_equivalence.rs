// Differential test: the push-based pipeline against the existing executor.
//
// The pipeline is being built to replace the executor's materialise-between-every
// step model. Replacing it is only safe if it computes the same answers, so these
// tests run the same logical query through both and compare row for row.
//
// The comparison is the point. A pipeline test that only checked the pipeline
// would pass just as happily on a shared misunderstanding of what the query
// means; running the old path beside it is what makes the check load-bearing.

use std::io::Write;

use falcondb::{
    db::Database,
    query::{
        expr::{BinaryOp, Expr},
        operators::{DistinctOp, FilterOp, LimitOp, ProjectOp, SortOp},
        pipeline::{Operator, Pipeline},
        select::SelectExpr,
        sort::SortExpr,
    },
    storage::{table::RecordBatch, types::ScalarValue},
};
use tempfile::NamedTempFile;

/// A table of `rows` rows: id ascending, grp cycling 0..10, score = id % 100.
fn db_with(rows: usize) -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "id,grp,score").unwrap();
    for i in 0..rows {
        writeln!(f, "{i},{},{}", i % 10, i % 100).unwrap();
    }
    f.flush().unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();
    (db, f)
}

/// Flatten batches into comparable rows of strings.
fn rows_of(batches: &[RecordBatch]) -> Vec<Vec<String>> {
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

fn col_gt(name: &str, n: i64) -> Expr {
    Expr::BinaryOp {
        left: Box::new(Expr::Column(name.into())),
        op: BinaryOp::Gt,
        right: Box::new(Expr::Literal(ScalarValue::Int64(Some(n)))),
    }
}

/// Run a pipeline over the whole table.
fn run_pipeline(db: &Database, ops: Vec<Box<dyn Operator>>) -> Vec<Vec<String>> {
    let table = db.get_table("t").unwrap();
    let mut p = Pipeline::new(ops);
    rows_of(&p.run(&table.batches).unwrap())
}

fn run_sql(db: &Database, sql: &str) -> Vec<Vec<String>> {
    db.sql(sql).unwrap().to_rows()
}

// ─── Filter ───────────────────────────────────────────────────────────────────

#[test]
fn filter_matches_the_executor() {
    let (db, _f) = db_with(20_000);
    let via_pipeline = run_pipeline(
        &db,
        vec![
            Box::new(FilterOp::new(col_gt("id", 19_990))),
            Box::new(ProjectOp::new(vec![SelectExpr::column("id")])),
        ],
    );
    assert_eq!(via_pipeline, run_sql(&db, "SELECT id FROM t WHERE id > 19990"));
}

#[test]
fn filter_that_matches_nothing_agrees() {
    let (db, _f) = db_with(5_000);
    let via_pipeline = run_pipeline(
        &db,
        vec![
            Box::new(FilterOp::new(col_gt("id", 1_000_000))),
            Box::new(ProjectOp::new(vec![SelectExpr::column("id")])),
        ],
    );
    assert!(via_pipeline.is_empty());
    assert_eq!(via_pipeline, run_sql(&db, "SELECT id FROM t WHERE id > 1000000"));
}

#[test]
fn filter_that_matches_everything_agrees() {
    let (db, _f) = db_with(5_000);
    let via_pipeline = run_pipeline(
        &db,
        vec![
            Box::new(FilterOp::new(col_gt("id", -1))),
            Box::new(ProjectOp::new(vec![SelectExpr::column("id")])),
        ],
    );
    assert_eq!(via_pipeline.len(), 5_000);
    assert_eq!(via_pipeline, run_sql(&db, "SELECT id FROM t WHERE id > -1"));
}

// ─── Projection ───────────────────────────────────────────────────────────────

#[test]
fn multi_column_projection_matches() {
    let (db, _f) = db_with(9_000);
    let via_pipeline = run_pipeline(
        &db,
        vec![Box::new(ProjectOp::new(vec![
            SelectExpr::column("grp"),
            SelectExpr::column("id"),
        ]))],
    );
    assert_eq!(via_pipeline, run_sql(&db, "SELECT grp, id FROM t"));
}

#[test]
fn computed_projection_matches() {
    let (db, _f) = db_with(7_000);
    let doubled = Expr::BinaryOp {
        left: Box::new(Expr::Column("id".into())),
        op: BinaryOp::Mul,
        right: Box::new(Expr::Literal(ScalarValue::Int64(Some(2)))),
    };
    let via_pipeline = run_pipeline(
        &db,
        vec![Box::new(ProjectOp::new(vec![SelectExpr::named(doubled, "d")]))],
    );
    assert_eq!(via_pipeline, run_sql(&db, "SELECT id * 2 AS d FROM t"));
}

// ─── Sort ─────────────────────────────────────────────────────────────────────

#[test]
fn sort_matches_the_executor() {
    let (db, _f) = db_with(20_000);
    let via_pipeline = run_pipeline(
        &db,
        vec![
            Box::new(SortOp::new(
                vec![SortExpr::desc(Expr::Column("id".into()))],
                None,
            )),
            Box::new(ProjectOp::new(vec![SelectExpr::column("id")])),
        ],
    );
    assert_eq!(via_pipeline, run_sql(&db, "SELECT id FROM t ORDER BY id DESC"));
}

#[test]
fn sort_on_a_column_with_many_ties_matches() {
    let (db, _f) = db_with(20_000);
    // `grp` has only ten distinct values, so most comparisons are ties.
    let via_pipeline = run_pipeline(
        &db,
        vec![
            Box::new(SortOp::new(
                vec![SortExpr::asc(Expr::Column("grp".into()))],
                None,
            )),
            Box::new(ProjectOp::new(vec![SelectExpr::column("grp")])),
        ],
    );
    assert_eq!(via_pipeline, run_sql(&db, "SELECT grp FROM t ORDER BY grp"));
}

// ─── Limit ────────────────────────────────────────────────────────────────────

#[test]
fn sort_with_limit_matches() {
    let (db, _f) = db_with(20_000);
    let via_pipeline = run_pipeline(
        &db,
        vec![
            Box::new(SortOp::new(
                vec![SortExpr::asc(Expr::Column("id".into()))],
                Some(10),
            )),
            Box::new(ProjectOp::new(vec![SelectExpr::column("id")])),
        ],
    );
    assert_eq!(
        via_pipeline,
        run_sql(&db, "SELECT id FROM t ORDER BY id LIMIT 10")
    );
}

#[test]
fn limit_with_offset_matches() {
    let (db, _f) = db_with(20_000);
    let via_pipeline = run_pipeline(
        &db,
        vec![
            Box::new(SortOp::new(
                vec![SortExpr::asc(Expr::Column("id".into()))],
                Some(3_005),
            )),
            Box::new(LimitOp::new(3_000, Some(5))),
            Box::new(ProjectOp::new(vec![SelectExpr::column("id")])),
        ],
    );
    assert_eq!(
        via_pipeline,
        run_sql(&db, "SELECT id FROM t ORDER BY id LIMIT 5 OFFSET 3000")
    );
}

// ─── Distinct ─────────────────────────────────────────────────────────────────

#[test]
fn distinct_matches_the_executor() {
    let (db, _f) = db_with(20_000);
    let mut via_pipeline = run_pipeline(
        &db,
        vec![
            Box::new(ProjectOp::new(vec![SelectExpr::column("grp")])),
            Box::new(DistinctOp::new()),
        ],
    );
    let mut via_sql = run_sql(&db, "SELECT DISTINCT grp FROM t");
    via_pipeline.sort();
    via_sql.sort();
    assert_eq!(via_pipeline, via_sql);
}

// ─── Composition ──────────────────────────────────────────────────────────────

#[test]
fn filter_sort_limit_project_together_match() {
    let (db, _f) = db_with(20_000);
    let via_pipeline = run_pipeline(
        &db,
        vec![
            Box::new(FilterOp::new(col_gt("score", 90))),
            Box::new(SortOp::new(
                vec![SortExpr::desc(Expr::Column("id".into()))],
                Some(12),
            )),
            Box::new(LimitOp::new(0, Some(12))),
            Box::new(ProjectOp::new(vec![
                SelectExpr::column("id"),
                SelectExpr::column("score"),
            ])),
        ],
    );
    assert_eq!(
        via_pipeline,
        run_sql(&db, "SELECT id, score FROM t WHERE score > 90 ORDER BY id DESC LIMIT 12")
    );
}

#[test]
fn results_survive_a_table_smaller_than_one_chunk() {
    let (db, _f) = db_with(100);
    let via_pipeline = run_pipeline(
        &db,
        vec![
            Box::new(FilterOp::new(col_gt("id", 50))),
            Box::new(ProjectOp::new(vec![SelectExpr::column("id")])),
        ],
    );
    assert_eq!(via_pipeline, run_sql(&db, "SELECT id FROM t WHERE id > 50"));
}

#[test]
fn results_survive_a_table_of_exactly_one_chunk() {
    // CHUNK_ROWS is 2048; an exact multiple exercises the boundary logic.
    let (db, _f) = db_with(2_048);
    let via_pipeline = run_pipeline(
        &db,
        vec![Box::new(ProjectOp::new(vec![SelectExpr::column("id")]))],
    );
    assert_eq!(via_pipeline.len(), 2_048);
    assert_eq!(via_pipeline, run_sql(&db, "SELECT id FROM t"));
}
