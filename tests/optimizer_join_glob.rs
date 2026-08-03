// Predicate deduplication, sort-merge join, and glob file reads.

use std::io::Write;

use falcondb::{
    db::Database,
    query::{
        expr::{BinaryOp, Expr},
        join::{execute_hash_join, execute_sort_merge_join, should_sort_merge, JoinType},
        optimizer::optimize_filter,
    },
    storage::{
        parquet_io::{expand_glob, read_parquet_glob, write_parquet},
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::{RecordBatch, Table},
        types::{DataType, ScalarValue},
    },
};
use tempfile::NamedTempFile;

// ─── Predicate deduplication ──────────────────────────────────────────────────

fn col_gt(name: &str, n: i64) -> Expr {
    Expr::BinaryOp {
        left: Box::new(Expr::Column(name.into())),
        op: BinaryOp::Gt,
        right: Box::new(Expr::Literal(ScalarValue::Int64(Some(n)))),
    }
}

fn and(a: Expr, b: Expr) -> Expr {
    Expr::BinaryOp { left: Box::new(a), op: BinaryOp::And, right: Box::new(b) }
}

/// Count the conjuncts in an AND chain.
fn conjuncts(e: &Expr) -> usize {
    match e {
        Expr::BinaryOp { left, op: BinaryOp::And, right } => conjuncts(left) + conjuncts(right),
        _ => 1,
    }
}

#[test]
fn duplicate_conjuncts_are_removed() {
    let e = and(and(col_gt("a", 5), col_gt("b", 2)), col_gt("a", 5));
    assert_eq!(conjuncts(&optimize_filter(e)), 2);
}

#[test]
fn distinct_conjuncts_are_kept() {
    let e = and(col_gt("a", 5), col_gt("a", 6));
    assert_eq!(conjuncts(&optimize_filter(e)), 2);
}

#[test]
fn a_single_predicate_is_untouched() {
    assert_eq!(conjuncts(&optimize_filter(col_gt("a", 5))), 1);
}

#[test]
fn deduplication_does_not_change_results() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "a,b").unwrap();
    for i in 0..500 {
        writeln!(f, "{i},{}", i % 7).unwrap();
    }
    f.flush().unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();

    let once = db.sql("SELECT a FROM t WHERE a > 100").unwrap().row_count();
    let twice = db
        .sql("SELECT a FROM t WHERE a > 100 AND a > 100")
        .unwrap()
        .row_count();
    assert_eq!(once, twice);
}

// ─── Sort-merge join ──────────────────────────────────────────────────────────

/// One batch of `k` (join key) and `v`.
fn keyed_batch(name: &str, keys: Vec<i64>) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::required("k", DataType::Int64),
        Field::required(name, DataType::Int64),
    ]);
    let vals: Vec<Option<i64>> = keys.iter().map(|&k| Some(k * 10)).collect();
    RecordBatch::new(
        schema,
        vec![
            ColumnBuilder::int64(keys.into_iter().map(Some).collect()),
            ColumnBuilder::int64(vals),
        ],
    )
    .unwrap()
}

/// Sorted (k, lv, rv) triples, so the two joins can be compared irrespective of
/// the order each happens to emit.
fn triples(b: &RecordBatch) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = (0..b.row_count())
        .map(|r| {
            (0..b.column_count())
                .map(|c| b.column(c).get(r).map(|v| v.to_string()).unwrap_or_default())
                .collect()
        })
        .collect();
    out.sort();
    out
}

#[test]
fn sort_merge_matches_hash_join() {
    let left = vec![keyed_batch("lv", (0..500).collect())];
    let right = vec![keyed_batch("rv", (250..750).collect())];

    let via_hash = execute_hash_join(&left, &right, &["k"], &["k"], JoinType::Inner).unwrap();
    let via_merge = execute_sort_merge_join(&left, &right, &["k"], &["k"]).unwrap();

    assert_eq!(via_merge.row_count(), 250);
    assert_eq!(triples(&via_merge), triples(&via_hash));
}

#[test]
fn sort_merge_handles_duplicate_keys() {
    // Three left rows and two right rows share key 1, so the run pairing has to
    // emit the full 3x2 cross product.
    let left = vec![keyed_batch("lv", vec![1, 1, 1, 2])];
    let right = vec![keyed_batch("rv", vec![1, 1, 3])];

    let via_hash = execute_hash_join(&left, &right, &["k"], &["k"], JoinType::Inner).unwrap();
    let via_merge = execute_sort_merge_join(&left, &right, &["k"], &["k"]).unwrap();

    assert_eq!(via_merge.row_count(), 6);
    assert_eq!(triples(&via_merge), triples(&via_hash));
}

#[test]
fn sort_merge_with_no_overlap_is_empty() {
    let left = vec![keyed_batch("lv", (0..100).collect())];
    let right = vec![keyed_batch("rv", (1000..1100).collect())];
    let out = execute_sort_merge_join(&left, &right, &["k"], &["k"]).unwrap();
    assert_eq!(out.row_count(), 0);
}

#[test]
fn sort_merge_handles_unsorted_input() {
    let left = vec![keyed_batch("lv", vec![5, 1, 4, 2, 3])];
    let right = vec![keyed_batch("rv", vec![3, 1, 5])];
    let out = execute_sort_merge_join(&left, &right, &["k"], &["k"]).unwrap();
    assert_eq!(out.row_count(), 3);
}

#[test]
fn sort_merge_across_several_batches() {
    let left = vec![
        keyed_batch("lv", (0..100).collect()),
        keyed_batch("lv", (100..200).collect()),
    ];
    let right = vec![keyed_batch("rv", (50..150).collect())];
    let out = execute_sort_merge_join(&left, &right, &["k"], &["k"]).unwrap();
    assert_eq!(out.row_count(), 100);
}

#[test]
fn hash_join_stays_the_default_at_ordinary_sizes() {
    // The threshold is deliberately high; small joins must not switch strategy.
    assert!(!should_sort_merge(1_000, 1_000));
    assert!(!should_sort_merge(500_000, 500_000));
    assert!(should_sort_merge(2_000_000, 2_000_000));
}

// ─── Glob reads ───────────────────────────────────────────────────────────────

fn write_part(dir: &std::path::Path, name: &str, ids: std::ops::Range<i64>) {
    let schema = Schema::new(vec![Field::required("id", DataType::Int64)]);
    let batch = RecordBatch::new(
        schema.clone(),
        vec![ColumnBuilder::int64(ids.map(Some).collect())],
    )
    .unwrap();
    let mut t = Table::new("t", schema);
    t.append_batch(batch).unwrap();
    write_parquet(&t.batches, &dir.join(name)).unwrap();
}

#[test]
fn glob_expands_to_the_matching_files() {
    let dir = tempfile::tempdir().unwrap();
    write_part(dir.path(), "part-000.parquet", 0..10);
    write_part(dir.path(), "part-001.parquet", 10..20);
    write_part(dir.path(), "other.txt", 0..1);

    let pattern = format!("{}/part-*.parquet", dir.path().to_str().unwrap().replace('\\', "/"));
    let found = expand_glob(&pattern).unwrap();
    assert_eq!(found.len(), 2, "got {found:?}");
}

#[test]
fn glob_reads_files_in_name_order() {
    let dir = tempfile::tempdir().unwrap();
    write_part(dir.path(), "part-000.parquet", 0..10);
    write_part(dir.path(), "part-001.parquet", 10..20);
    write_part(dir.path(), "part-002.parquet", 20..30);

    let pattern = format!("{}/part-*.parquet", dir.path().to_str().unwrap().replace('\\', "/"));
    let batches = read_parquet_glob(&pattern, "t").unwrap();

    let mut ids = Vec::new();
    for b in &batches {
        let c = b.column_by_name("id").unwrap();
        for r in 0..b.row_count() {
            ids.push(c.get(r).unwrap().as_i64().unwrap());
        }
    }
    assert_eq!(ids, (0..30).collect::<Vec<_>>(), "files read out of order");
}

#[test]
fn a_plain_path_passes_through_unchanged() {
    let got = expand_glob("data/file.parquet").unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0], std::path::PathBuf::from("data/file.parquet"));
}

#[test]
fn a_pattern_matching_nothing_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let pattern = format!("{}/nope-*.parquet", dir.path().to_str().unwrap().replace('\\', "/"));
    assert!(expand_glob(&pattern).is_err());
}

#[test]
fn wildcards_in_directories_are_rejected() {
    assert!(expand_glob("data/*/file.parquet").is_err());
}

#[test]
fn copy_accepts_a_glob() {
    let dir = tempfile::tempdir().unwrap();
    write_part(dir.path(), "a-000.parquet", 0..5);
    write_part(dir.path(), "a-001.parquet", 5..10);

    let pattern = format!("{}/a-*.parquet", dir.path().to_str().unwrap().replace('\\', "/"));
    let db = Database::in_memory();
    db.sql(&format!("COPY t FROM '{pattern}'")).unwrap();

    assert_eq!(db.sql("SELECT id FROM t").unwrap().row_count(), 10);
}

#[test]
fn question_mark_matches_exactly_one_character() {
    let dir = tempfile::tempdir().unwrap();
    write_part(dir.path(), "p1.parquet", 0..3);
    write_part(dir.path(), "p2.parquet", 3..6);
    write_part(dir.path(), "p10.parquet", 6..9);

    let pattern = format!("{}/p?.parquet", dir.path().to_str().unwrap().replace('\\', "/"));
    let found = expand_glob(&pattern).unwrap();
    assert_eq!(found.len(), 2, "p10 should not match p?, got {found:?}");
}
