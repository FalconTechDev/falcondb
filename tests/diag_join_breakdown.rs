//! Where does a join actually spend its time?
//!
//! Parallelising the probe over twelve cores took 1M-row INNER JOIN from 52.3 ms
//! to 48.9 ms — a 6.5% saving, so the probe is a small share of the total and the
//! remaining ~45 ms is somewhere else. Rather than guess, time the join through
//! the SQL layer against the join operator called directly: if they differ, the
//! cost is not in the join at all.
//!
//! Run with: cargo test --release --test diag_join_breakdown -- --nocapture

use std::time::Instant;

use falcondb::{
    db::Database,
    query::join::{execute_hash_join, JoinType},
    storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::{RecordBatch, Table},
        types::DataType,
    },
};

const DEPTS: [&str; 5] = ["eng", "sales", "hr", "ops", "legal"];

fn write_side(
    dir: &tempfile::TempDir,
    name: &str,
    rows: usize,
    key_stride: i64,
) -> std::path::PathBuf {
    let path = dir.path().join(format!("{name}.parquet"));
    let schema = Schema::new(vec![
        Field::required("k", DataType::Int64),
        Field::required("label", DataType::Utf8),
    ]);
    let batch = RecordBatch::new(
        schema.clone(),
        vec![
            ColumnBuilder::int64((0..rows).map(|i| Some(i as i64 * key_stride)).collect()),
            ColumnBuilder::utf8_from_str((0..rows).map(|i| Some(DEPTS[i % 5])).collect()),
        ],
    )
    .unwrap();
    let mut t = Table::new(name, schema);
    t.append_batch(batch).unwrap();
    falcondb::storage::parquet_io::write_parquet(&t.batches, &path).unwrap();
    path
}

fn best<F: FnMut() -> std::time::Duration>(mut f: F, runs: usize) -> std::time::Duration {
    (0..runs).map(|_| f()).min().unwrap()
}

#[test]
fn join_operator_against_the_sql_path() {
    const ROWS: usize = 1_000_000;
    let dir = tempfile::tempdir().unwrap();
    let left_path = write_side(&dir, "big", ROWS, 1);
    let right_path = write_side(&dir, "small", ROWS / 10, 10);

    let db = Database::in_memory();
    db.load_parquet(&left_path, "big").unwrap();
    db.load_parquet(&right_path, "small").unwrap();

    let big = db.get_table("big").unwrap();
    let small = db.get_table("small").unwrap();
    println!("left batches={}  right batches={}", big.batches.len(), small.batches.len());

    // Warm up both paths before timing either.
    for _ in 0..2 {
        db.sql("SELECT COUNT(*) AS n FROM big JOIN small ON big.k = small.k").unwrap();
        execute_hash_join(&big.batches, &small.batches, &["k"], &["k"], JoinType::Inner).unwrap();
    }

    let sql = best(
        || {
            let t = Instant::now();
            let r = db
                .sql("SELECT COUNT(*) AS n FROM big JOIN small ON big.k = small.k")
                .unwrap();
            let e = t.elapsed();
            assert_eq!(r.batches[0].row_count(), 1);
            e
        },
        5,
    );

    let direct = best(
        || {
            let t = Instant::now();
            let out =
                execute_hash_join(&big.batches, &small.batches, &["k"], &["k"], JoinType::Inner)
                    .unwrap();
            let e = t.elapsed();
            assert_eq!(out.row_count(), ROWS / 10);
            e
        },
        5,
    );

    // The join merges each side into one batch before building or probing.
    // `concat_batches` is exactly that step, so timing it splits the operator into
    // "flatten the inputs" and "everything else".
    let concat_left = best(
        || {
            let t = Instant::now();
            let b = RecordBatch::concat(&big.batches).unwrap();
            let e = t.elapsed();
            assert_eq!(b.row_count(), ROWS);
            e
        },
        5,
    );
    let concat_right = best(
        || {
            let t = Instant::now();
            let b = RecordBatch::concat(&small.batches).unwrap();
            let e = t.elapsed();
            assert_eq!(b.row_count(), ROWS / 10);
            e
        },
        5,
    );

    println!("SQL   SELECT COUNT(*) ... JOIN = {sql:?}");
    println!("direct execute_hash_join      = {direct:?}");
    println!(
        "overhead outside the join operator = {:.1}%",
        (sql.as_secs_f64() - direct.as_secs_f64()) / sql.as_secs_f64() * 100.0
    );
    // COUNT(*) reads no column, so projection pruning builds none of them. SUM over
    // a left column forces one to be gathered. The gap is what materialising a
    // column of the join output actually costs — as opposed to the bookkeeping of
    // recording 100k matches, which both queries pay.
    for _ in 0..2 {
        db.sql("SELECT SUM(k) AS s FROM big JOIN small ON big.k = small.k").unwrap();
    }
    let sql_sum = best(
        || {
            let t = Instant::now();
            db.sql("SELECT SUM(k) AS s FROM big JOIN small ON big.k = small.k").unwrap();
            t.elapsed()
        },
        5,
    );
    println!("  SQL COUNT(*) (no column built)         = {sql:?}");
    println!("  SQL SUM(k)   (one column gathered)     = {sql_sum:?}");

    // Same probe side, a build side 100x smaller. The probe still walks 1M rows,
    // so whatever falls away is the cost of building the hash table.
    let tiny_path = write_side(&dir, "tiny", 1_000, 1_000);
    db.load_parquet(&tiny_path, "tiny").unwrap();
    let tiny = db.get_table("tiny").unwrap();
    for _ in 0..2 {
        execute_hash_join(&big.batches, &tiny.batches, &["k"], &["k"], JoinType::Inner).unwrap();
    }
    let small_build = best(
        || {
            let t = Instant::now();
            execute_hash_join(&big.batches, &tiny.batches, &["k"], &["k"], JoinType::Inner)
                .unwrap();
            t.elapsed()
        },
        5,
    );
    // Same build side (100k keys) and same 1M probe, but the key ranges do not
    // overlap, so nothing matches and no output rows are assembled. The gap to the
    // matching join is what building the result costs.
    let miss_path = write_side(&dir, "miss", ROWS / 10, 10);
    db.load_parquet(&miss_path, "miss").unwrap();
    let miss = db.get_table("miss").unwrap();
    // Shift the probe side out of range by joining on a key column that cannot hit.
    let shifted = RecordBatch::new(
        Schema::new(vec![Field::required("k", DataType::Int64)]),
        vec![ColumnBuilder::int64(
            (0..ROWS).map(|i| Some(i as i64 + 100_000_000)).collect(),
        )],
    )
    .unwrap();
    let shifted = vec![shifted];
    for _ in 0..2 {
        execute_hash_join(&shifted, &miss.batches, &["k"], &["k"], JoinType::Inner).unwrap();
    }
    let no_output = best(
        || {
            let t = Instant::now();
            let out =
                execute_hash_join(&shifted, &miss.batches, &["k"], &["k"], JoinType::Inner)
                    .unwrap();
            let e = t.elapsed();
            assert_eq!(out.row_count(), 0);
            e
        },
        5,
    );
    println!("  100k build + 1M probe, zero matches   = {no_output:?}");
    println!("  same 1M probe, 1k-row build side      = {small_build:?}");
    println!("  concat left  (977 batches, 1M rows)   = {concat_left:?}");
    println!("  concat right ( 98 batches, 100k rows) = {concat_right:?}");
    let merges = concat_left + concat_right;
    println!(
        "  merges are {:.1}% of the join; build+probe+output = {:?}",
        merges.as_secs_f64() / direct.as_secs_f64() * 100.0,
        direct.saturating_sub(merges)
    );
}

/// The same join in the shape the *benchmark* actually runs.
///
/// The test above measures a two-column table, but `SELECT COUNT(*) FROM bench
/// JOIN lookup ON bench.id = lookup.lid` needs only the two key columns, and
/// `project_for_join` trims to them before merging. Its merge is therefore one
/// 8 MB Int64 column, not three columns of mixed types — so the "merges are 27%
/// of the join" figure above describes a shape the benchmark never reaches.
/// Measure the shape we are actually losing on.
///
/// Phases are separated by changing the *input*, not by instrumenting the
/// operator: a pre-merged left side removes the merge, and keys that cannot match
/// remove the output build while leaving build+probe intact.
#[test]
fn breakdown_in_the_benchmark_shape() {
    const LEFT_ROWS:  usize = 1_000_000;
    const RIGHT_ROWS: usize = 100_000;
    const BATCH:      usize = 65_536; // what the Parquet reader produces

    let key_only = Schema::new(vec![Field::required("k", DataType::Int64)]);
    let make = |rows: usize, batch_rows: usize, base: i64, stride: i64| -> Vec<RecordBatch> {
        let mut out = Vec::new();
        let mut start = 0;
        while start < rows {
            let n = batch_rows.min(rows - start);
            out.push(
                RecordBatch::new(
                    key_only.clone(),
                    vec![ColumnBuilder::int64(
                        (0..n).map(|i| Some(base + (start + i) as i64 * stride)).collect(),
                    )],
                )
                .unwrap(),
            );
            start += n;
        }
        out
    };

    // Left keys 0..1M; right keys 0,10,20,... so every tenth left row matches once.
    let left_batched = make(LEFT_ROWS, BATCH, 0, 1);
    let left_single  = make(LEFT_ROWS, LEFT_ROWS, 0, 1);
    let right        = make(RIGHT_ROWS, BATCH, 0, 10);
    // Same size and same batching, but disjoint from the left key range.
    let right_miss   = make(RIGHT_ROWS, BATCH, 1_000_000_000, 10);

    let run = |l: &[RecordBatch], r: &[RecordBatch]| -> (std::time::Duration, usize) {
        let t = Instant::now();
        let out = execute_hash_join(l, r, &["k"], &["k"], JoinType::Inner).unwrap();
        (t.elapsed(), out.row_count())
    };
    for _ in 0..3 {
        run(&left_batched, &right);
        run(&left_single, &right);
        run(&left_batched, &right_miss);
    }

    let (batched, n_match) = {
        let mut best_d = None;
        let mut rows = 0;
        for _ in 0..7 {
            let (d, n) = run(&left_batched, &right);
            rows = n;
            best_d = Some(best_d.map_or(d, |b: std::time::Duration| b.min(d)));
        }
        (best_d.unwrap(), rows)
    };
    let single = best(|| run(&left_single, &right).0, 7);
    let no_out = best(|| run(&left_batched, &right_miss).0, 7);

    println!("benchmark shape: 1M left / 100k right, one Int64 key per side");
    println!("  output rows                       = {n_match}");
    println!("  full join, {:>2} left batches        = {batched:?}", left_batched.len());
    println!("  full join,  1 left batch          = {single:?}");
    println!("  left merge (difference)           = {:?}", batched.saturating_sub(single));
    println!("  build + probe only (no matches)   = {no_out:?}");
    println!(
        "  output build (difference)         = {:?}",
        single.saturating_sub(no_out),
    );
}

/// Is the left-hand merge bound by copying, or by allocating?
///
/// `PrimitiveColumn::concat_all` reserves once and then memcpys dense slices, so
/// 8 MB should cost roughly 8 MB / bandwidth — about 150 us here. It measures
/// 3.2 ms, twenty times that. The suspicion is that the time is not in the copy
/// at all but in first-touch page faults on a freshly allocated 8 MB buffer,
/// which Windows commits lazily.
///
/// If so, the fix is structural — stop materialising the side — rather than
/// making the copy faster, because the copy was never the cost.
#[test]
fn is_the_merge_copy_bound_or_allocation_bound() {
    const ROWS: usize = 1_000_000;
    const BATCH: usize = 65_536;

    let parts: Vec<Vec<i64>> = (0..ROWS)
        .step_by(BATCH)
        .map(|s| ((s as i64)..((s + BATCH).min(ROWS)) as i64).collect())
        .collect();

    // Fresh allocation every time: what concat_all does today.
    let fresh = best(
        || {
            let t = Instant::now();
            let mut v: Vec<i64> = Vec::with_capacity(ROWS);
            for p in &parts {
                v.extend_from_slice(p);
            }
            let e = t.elapsed();
            std::hint::black_box(&v);
            e
        },
        9,
    );

    // Same copy, same total bytes, into a buffer whose pages are already faulted in.
    let mut reused: Vec<i64> = Vec::with_capacity(ROWS);
    reused.extend(std::iter::repeat(0).take(ROWS));
    let warm = best(
        || {
            reused.clear();
            let t = Instant::now();
            for p in &parts {
                reused.extend_from_slice(p);
            }
            let e = t.elapsed();
            std::hint::black_box(&reused);
            e
        },
        9,
    );

    println!("copy 8 MB from {} slices:", parts.len());
    println!("  into a fresh allocation  = {fresh:?}");
    println!("  into a warm buffer       = {warm:?}");
    println!(
        "  allocation share         = {:.1}%",
        (fresh.as_secs_f64() - warm.as_secs_f64()) / fresh.as_secs_f64() * 100.0,
    );
}
