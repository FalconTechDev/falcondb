// The join Bloom filter and the parallel Parquet reader.
//
// The Bloom filter is only allowed to be wrong in one direction: saying "maybe"
// when the answer is no costs a wasted lookup, but saying "no" when the answer is
// yes would silently drop join matches. Most of the tests below exist to catch
// that second case, which no amount of passing row counts elsewhere would reveal.

use std::io::Write;

use falcondb::{
    db::Database,
    storage::{
        column::ColumnBuilder,
        parquet_io::{read_parquet, read_parquet_projected, write_parquet},
        schema::{Field, Schema},
        table::{RecordBatch, Table},
        types::DataType,
    },
};
use tempfile::NamedTempFile;

// ─── Joins ────────────────────────────────────────────────────────────────────

/// Two tables joined on `k`, sized past the Bloom filter's 1024-row threshold.
fn join_db(left_rows: usize, right_rows: usize) -> (Database, NamedTempFile, NamedTempFile) {
    let mut fl = NamedTempFile::new().unwrap();
    writeln!(fl, "k,lv").unwrap();
    for i in 0..left_rows {
        writeln!(fl, "{i},{}", i * 2).unwrap();
    }
    fl.flush().unwrap();

    let mut fr = NamedTempFile::new().unwrap();
    writeln!(fr, "k,rv").unwrap();
    for i in 0..right_rows {
        writeln!(fr, "{i},{}", i * 3).unwrap();
    }
    fr.flush().unwrap();

    let db = Database::in_memory();
    db.load_csv(fl.path(), "l").unwrap();
    db.load_csv(fr.path(), "r").unwrap();
    (db, fl, fr)
}

#[test]
fn inner_join_keeps_every_match_above_the_bloom_threshold() {
    // 4000 left rows, 2000 right: exactly 2000 should match. A false negative in
    // the filter would show up here as a short result.
    let (db, _l, _r) = join_db(4_000, 2_000);
    let n = db
        .sql("SELECT lv FROM l JOIN r ON l.k = r.k")
        .unwrap()
        .row_count();
    assert_eq!(n, 2_000, "join lost matches");
}

#[test]
fn inner_join_below_the_threshold_is_unaffected() {
    let (db, _l, _r) = join_db(500, 300);
    let n = db.sql("SELECT lv FROM l JOIN r ON l.k = r.k").unwrap().row_count();
    assert_eq!(n, 300);
}

#[test]
fn left_join_keeps_unmatched_rows() {
    // Rows the filter rejects must still appear, with NULLs on the right.
    let (db, _l, _r) = join_db(4_000, 2_000);
    let n = db
        .sql("SELECT lv FROM l LEFT JOIN r ON l.k = r.k")
        .unwrap()
        .row_count();
    assert_eq!(n, 4_000, "LEFT JOIN dropped unmatched rows");
}

#[test]
fn join_with_no_overlap_returns_nothing() {
    let mut fl = NamedTempFile::new().unwrap();
    writeln!(fl, "k,lv").unwrap();
    for i in 0..2_000 {
        writeln!(fl, "{i},{i}").unwrap();
    }
    fl.flush().unwrap();

    let mut fr = NamedTempFile::new().unwrap();
    writeln!(fr, "k,rv").unwrap();
    for i in 100_000..102_000 {
        writeln!(fr, "{i},{i}").unwrap();
    }
    fr.flush().unwrap();

    let db = Database::in_memory();
    db.load_csv(fl.path(), "l").unwrap();
    db.load_csv(fr.path(), "r").unwrap();

    assert_eq!(
        db.sql("SELECT lv FROM l JOIN r ON l.k = r.k").unwrap().row_count(),
        0
    );
}

#[test]
fn join_result_matches_a_small_reference() {
    // Cross-check the filtered path against hand-computed answers.
    let (db, _l, _r) = join_db(2_000, 2_000);
    let rows = db
        .sql("SELECT k, lv, rv FROM l JOIN r ON l.k = r.k WHERE k < 3")
        .unwrap()
        .to_rows();
    assert_eq!(rows.len(), 3);
    let mut got: Vec<_> = rows.iter().map(|r| r[0].clone()).collect();
    got.sort();
    assert_eq!(got, ["0", "1", "2"]);
}

#[test]
fn string_keyed_join_survives_the_filter() {
    let mut fl = NamedTempFile::new().unwrap();
    writeln!(fl, "k,lv").unwrap();
    for i in 0..2_000 {
        writeln!(fl, "key{i},{i}").unwrap();
    }
    fl.flush().unwrap();

    let mut fr = NamedTempFile::new().unwrap();
    writeln!(fr, "k,w").unwrap();
    for i in 0..2_000 {
        writeln!(fr, "key{i},{i}").unwrap();
    }
    fr.flush().unwrap();

    let db = Database::in_memory();
    db.load_csv(fl.path(), "l").unwrap();
    db.load_csv(fr.path(), "r").unwrap();

    assert_eq!(
        db.sql("SELECT lv FROM l JOIN r ON l.k = r.k").unwrap().row_count(),
        2_000
    );
}

// ─── Parquet reader ───────────────────────────────────────────────────────────

fn wide_table(rows: usize) -> Table {
    let schema = Schema::new(vec![
        Field::required("id", DataType::Int64),
        Field::required("name", DataType::Utf8),
        Field::required("score", DataType::Float64),
    ]);
    let batch = RecordBatch::new(
        schema.clone(),
        vec![
            ColumnBuilder::int64((0..rows).map(|i| Some(i as i64)).collect()),
            ColumnBuilder::utf8_from_str((0..rows).map(|_| Some("x")).collect()),
            ColumnBuilder::float64((0..rows).map(|i| Some(i as f64)).collect()),
        ],
    )
    .unwrap();
    let mut t = Table::new("t", schema);
    t.append_batch(batch).unwrap();
    t
}

#[test]
fn parallel_row_group_read_preserves_row_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.parquet");
    write_parquet(&wide_table(200_000).batches, &path).unwrap();

    let back = read_parquet(&path, "t").unwrap();
    let total: usize = back.iter().map(|b| b.row_count()).sum();
    assert_eq!(total, 200_000);

    // id ascends from 0; decoding groups in parallel must not reorder them.
    let mut expected = 0i64;
    for b in &back {
        let col = b.column_by_name("id").unwrap();
        for r in 0..b.row_count() {
            let got = col.get(r).unwrap().as_i64().unwrap();
            assert_eq!(got, expected, "row order changed during parallel decode");
            expected += 1;
        }
    }
}

#[test]
fn projected_read_returns_only_the_named_columns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.parquet");
    write_parquet(&wide_table(5_000).batches, &path).unwrap();

    let want: std::collections::HashSet<String> =
        ["id".to_string(), "score".to_string()].into_iter().collect();
    let back = read_parquet_projected(&path, "t", Some(&want)).unwrap();

    assert_eq!(back[0].column_count(), 2, "projection did not drop a column");
    assert!(back[0].column_by_name("id").is_ok());
    assert!(back[0].column_by_name("score").is_ok());
    assert!(back[0].column_by_name("name").is_err());

    let total: usize = back.iter().map(|b| b.row_count()).sum();
    assert_eq!(total, 5_000, "projection changed the row count");
}

#[test]
fn projection_naming_unknown_columns_falls_back_to_all() {
    // The caller's column set is a union across clauses and may name columns from
    // another table; that must not produce an empty batch.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.parquet");
    write_parquet(&wide_table(1_000).batches, &path).unwrap();

    let want: std::collections::HashSet<String> = ["nope".to_string()].into_iter().collect();
    let back = read_parquet_projected(&path, "t", Some(&want)).unwrap();
    assert_eq!(back[0].column_count(), 3);
}

#[test]
fn projected_values_match_a_full_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.parquet");
    write_parquet(&wide_table(3_000).batches, &path).unwrap();

    let full = read_parquet(&path, "t").unwrap();
    let want: std::collections::HashSet<String> = ["score".to_string()].into_iter().collect();
    let proj = read_parquet_projected(&path, "t", Some(&want)).unwrap();

    let flat = |bs: &[RecordBatch], col: &str| -> Vec<String> {
        let mut out = Vec::new();
        for b in bs {
            let c = b.column_by_name(col).unwrap();
            for r in 0..b.row_count() {
                out.push(c.get(r).unwrap().to_string());
            }
        }
        out
    };
    assert_eq!(flat(&proj, "score"), flat(&full, "score"));
}
