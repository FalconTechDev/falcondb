// Window frames: ROWS BETWEEN n PRECEDING AND m FOLLOWING.
//
// Without an explicit frame a SQL aggregate window is cumulative, which is what
// FalconDB always did. These tests cover the bounded (rolling) frames, and pin
// the cumulative default so adding frames did not change existing behaviour.

use std::io::Write;

use falcondb::db::Database;
use tempfile::NamedTempFile;

/// Table `t(i, v)` with the given values, `i` ascending from 1.
fn db_with_values(vals: &[i64]) -> (Database, NamedTempFile) {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "i,v\n").unwrap();
    for (idx, v) in vals.iter().enumerate() {
        writeln!(f, "{},{v}", idx + 1).unwrap();
    }
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();
    (db, f)
}

/// Window column, ordered by `i` so the result is positional.
fn window_col(db: &Database, expr: &str) -> Vec<String> {
    let sql = format!("SELECT i, {expr} AS w FROM t ORDER BY i");
    let r = db.sql(&sql).unwrap();
    let names = r.column_names();
    let idx = names.iter().position(|n| n == "w").unwrap();
    r.to_rows().into_iter().map(|row| row[idx].clone()).collect()
}

fn as_f64(vals: Vec<String>) -> Vec<f64> {
    vals.into_iter().map(|v| v.parse::<f64>().unwrap()).collect()
}

// ─── Cumulative default (unchanged behaviour) ─────────────────────────────────

#[test]
fn sum_without_a_frame_is_cumulative() {
    let (db, _f) = db_with_values(&[1, 2, 3, 4]);
    let got = as_f64(window_col(&db, "SUM(v) OVER (ORDER BY i)"));
    assert_eq!(got, vec![1.0, 3.0, 6.0, 10.0]);
}

// ─── Rolling SUM ──────────────────────────────────────────────────────────────

#[test]
fn rolling_sum_over_three_rows() {
    let (db, _f) = db_with_values(&[1, 2, 3, 4, 5]);
    let got = as_f64(window_col(
        &db,
        "SUM(v) OVER (ORDER BY i ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)",
    ));
    // Frame fills up over the first two rows, then slides.
    assert_eq!(got, vec![1.0, 3.0, 6.0, 9.0, 12.0]);
}

#[test]
fn rolling_sum_can_look_forward() {
    let (db, _f) = db_with_values(&[1, 2, 3, 4]);
    let got = as_f64(window_col(
        &db,
        "SUM(v) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)",
    ));
    assert_eq!(got, vec![3.0, 6.0, 9.0, 7.0]);
}

#[test]
fn rolling_sum_matches_a_recomputed_frame() {
    // Cross-check the O(n) sliding sum against the naive definition.
    let vals: Vec<i64> = (1..=60).map(|i| (i * 7) % 13).collect();
    let (db, _f) = db_with_values(&vals);
    let got = as_f64(window_col(
        &db,
        "SUM(v) OVER (ORDER BY i ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)",
    ));

    let expected: Vec<f64> = (0..vals.len())
        .map(|p| {
            let lo = p.saturating_sub(4);
            vals[lo..=p].iter().sum::<i64>() as f64
        })
        .collect();
    assert_eq!(got, expected);
}

// ─── Rolling AVG ──────────────────────────────────────────────────────────────

#[test]
fn rolling_average_over_two_rows() {
    let (db, _f) = db_with_values(&[10, 20, 30, 40]);
    let got = as_f64(window_col(
        &db,
        "AVG(v) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)",
    ));
    assert_eq!(got, vec![10.0, 15.0, 25.0, 35.0]);
}

// ─── Rolling MIN / MAX ────────────────────────────────────────────────────────

#[test]
fn rolling_min_tracks_the_frame() {
    let (db, _f) = db_with_values(&[5, 1, 4, 2, 8]);
    let got = as_f64(window_col(
        &db,
        "MIN(v) OVER (ORDER BY i ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)",
    ));
    assert_eq!(got, vec![5.0, 1.0, 1.0, 1.0, 2.0]);
}

#[test]
fn rolling_max_recovers_after_the_peak_leaves() {
    // The 9 must stop being the answer once it falls out of the frame.
    let (db, _f) = db_with_values(&[9, 1, 2, 3, 4]);
    let got = as_f64(window_col(
        &db,
        "MAX(v) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)",
    ));
    assert_eq!(got, vec![9.0, 9.0, 2.0, 3.0, 4.0]);
}

#[test]
fn rolling_min_matches_a_recomputed_frame() {
    let vals: Vec<i64> = (1..=80).map(|i| (i * 31) % 17).collect();
    let (db, _f) = db_with_values(&vals);
    let got = as_f64(window_col(
        &db,
        "MIN(v) OVER (ORDER BY i ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)",
    ));

    let expected: Vec<f64> = (0..vals.len())
        .map(|p| {
            let lo = p.saturating_sub(5);
            *vals[lo..=p].iter().min().unwrap() as f64
        })
        .collect();
    assert_eq!(got, expected);
}

// ─── COUNT ────────────────────────────────────────────────────────────────────

#[test]
fn rolling_count_saturates_at_frame_width() {
    let (db, _f) = db_with_values(&[1, 1, 1, 1, 1]);
    let got = as_f64(window_col(
        &db,
        "COUNT(*) OVER (ORDER BY i ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)",
    ));
    assert_eq!(got, vec![1.0, 2.0, 3.0, 3.0, 3.0]);
}

// ─── Partitioning ─────────────────────────────────────────────────────────────

#[test]
fn frame_does_not_reach_across_partitions() {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "g,i,v\na,1,1\na,2,2\nb,3,10\nb,4,20\n").unwrap();
    let db = Database::in_memory();
    db.load_csv(f.path(), "t").unwrap();

    let r = db
        .sql(
            "SELECT i, SUM(v) OVER (PARTITION BY g ORDER BY i \
             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS w FROM t ORDER BY i",
        )
        .unwrap();
    let names = r.column_names();
    let idx = names.iter().position(|n| n == "w").unwrap();
    let got: Vec<f64> = r
        .to_rows()
        .into_iter()
        .map(|row| row[idx].parse::<f64>().unwrap())
        .collect();

    // Group b restarts: 10 does not pick up group a's rows.
    assert_eq!(got, vec![1.0, 3.0, 10.0, 30.0]);
}

// ─── Rejected frames ──────────────────────────────────────────────────────────

#[test]
fn range_frames_are_rejected_not_silently_treated_as_rows() {
    let (db, _f) = db_with_values(&[1, 2, 3]);
    let msg = match db
        .sql("SELECT SUM(v) OVER (ORDER BY i RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t")
    {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a RANGE frame was accepted; it would silently behave as ROWS"),
    };
    assert!(msg.contains("ROWS"), "unhelpful error: {msg}");
}
