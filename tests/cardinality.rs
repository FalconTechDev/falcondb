// Sampled distinct-value estimation.
//
// Row count cannot tell a join on five distinct keys apart from a join on a
// million, and those two cost wildly different amounts. This estimator samples
// rather than counting, so it stays cheap enough to run while planning.
//
// It is an estimate: the tests assert it lands in the right neighbourhood, not
// on an exact figure. Asserting exactness would make them fail on any change to
// the sampling stride while telling us nothing about whether the answer is
// useful.

use falcondb::{
    query::optimizer::estimate_distinct,
    storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        table::RecordBatch,
        types::DataType,
    },
};

/// One batch where `k` takes `distinct` different values across `rows` rows.
fn batch(rows: usize, distinct: usize) -> RecordBatch {
    let schema = Schema::new(vec![Field::required("k", DataType::Int64)]);
    RecordBatch::new(
        schema,
        vec![ColumnBuilder::int64(
            (0..rows).map(|i| Some((i % distinct) as i64)).collect(),
        )],
    )
    .unwrap()
}

/// Within a factor of `tol` either way.
fn close(got: usize, want: usize, tol: f64) -> bool {
    let g = got as f64;
    let w = want as f64;
    g <= w * tol && g >= w / tol
}

#[test]
fn few_distinct_values_are_estimated_low() {
    let got = estimate_distinct(&[batch(100_000, 5)], "k").unwrap();
    assert!(close(got, 5, 3.0), "estimated {got} for 5 distinct values");
}

#[test]
fn a_unique_column_is_estimated_near_the_row_count() {
    let rows = 50_000;
    let got = estimate_distinct(&[batch(rows, rows)], "k").unwrap();
    assert!(
        close(got, rows, 2.0),
        "estimated {got} for {rows} distinct values"
    );
}

#[test]
fn a_middling_column_lands_between_the_extremes() {
    let low = estimate_distinct(&[batch(100_000, 10)], "k").unwrap();
    let mid = estimate_distinct(&[batch(100_000, 1_000)], "k").unwrap();
    let high = estimate_distinct(&[batch(100_000, 100_000)], "k").unwrap();

    assert!(low < mid, "{low} should be below {mid}");
    assert!(mid < high, "{mid} should be below {high}");
}

#[test]
fn the_estimate_never_exceeds_the_row_count() {
    let got = estimate_distinct(&[batch(1_000, 1_000)], "k").unwrap();
    assert!(got <= 1_000, "estimated {got} distinct values in 1000 rows");
}

#[test]
fn the_estimate_is_at_least_one() {
    let got = estimate_distinct(&[batch(1_000, 1)], "k").unwrap();
    assert!(got >= 1);
}

#[test]
fn several_batches_are_sampled_together() {
    let batches = vec![batch(20_000, 50), batch(20_000, 50), batch(20_000, 50)];
    let got = estimate_distinct(&batches, "k").unwrap();
    assert!(close(got, 50, 3.0), "estimated {got} across three batches");
}

#[test]
fn a_half_unique_column_is_not_wildly_overestimated() {
    // The awkward middle: neither saturated nor a key. The estimate scales
    // sub-linearly here, so it overshoots — but it has to stay in the right
    // order of magnitude to be useful for ordering joins.
    let got = estimate_distinct(&[batch(100_000, 1_000)], "k").unwrap();
    assert!(
        got >= 1_000 && got <= 50_000,
        "estimated {got} for 1000 distinct values in 100k rows"
    );
}

#[test]
fn a_small_table_is_counted_essentially_exactly() {
    // Below the sample target every row is read, so this is exact.
    let got = estimate_distinct(&[batch(100, 7)], "k").unwrap();
    assert_eq!(got, 7);
}

#[test]
fn an_empty_input_has_no_estimate() {
    assert!(estimate_distinct(&[], "k").is_none());
}

#[test]
fn a_zero_row_batch_has_no_estimate() {
    assert!(estimate_distinct(&[batch(0, 1)], "k").is_none());
}

#[test]
fn an_unknown_column_has_no_estimate() {
    assert!(estimate_distinct(&[batch(100, 5)], "nope").is_none());
}

#[test]
fn estimation_is_cheap_on_a_large_table() {
    // Sampling, not counting: a million rows must not take a full pass.
    let t0 = std::time::Instant::now();
    let got = estimate_distinct(&[batch(1_000_000, 1_000)], "k").unwrap();
    let elapsed = t0.elapsed();

    assert!(got > 0);
    assert!(
        elapsed.as_millis() < 200,
        "estimating over 1M rows took {elapsed:?} - is it still sampling?"
    );
}
