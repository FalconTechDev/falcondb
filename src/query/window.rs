use std::collections::hash_map::Entry;

use super::hash::FastMap;

use crate::storage::{
    column::{Column, PrimitiveColumn, Utf8Column},
    schema::Field,
    table::RecordBatch,
    types::{DataType, FalconError, FalconResult, ScalarValue},
};

use super::{expr::Expr, sort::SortOrder};

// ─── Window function kinds ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum WindowFunc {
    RowNumber,
    Rank,
    DenseRank,
    Lag(usize),
    Lead(usize),
    Sum,
    Avg,
    Min,
    Max,
    Count,
}

// ─── Window frame ─────────────────────────────────────────────────────────────

/// Which rows of a partition an aggregate window function sees at each row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrame {
    /// `UNBOUNDED PRECEDING AND CURRENT ROW` — the SQL default. The frame grows
    /// as the scan advances, giving a running total.
    Cumulative,
    /// `ROWS BETWEEN <preceding> PRECEDING AND <following> FOLLOWING` — a frame of
    /// bounded width that slides along the partition. This is what makes a rolling
    /// average rolling rather than cumulative.
    Rows { preceding: usize, following: usize },
}

impl WindowFrame {
    /// Frame bounds for the row at `pos` in a partition of `len` rows, as an
    /// inclusive `[lo, hi]` range of positions.
    fn bounds(&self, pos: usize, len: usize) -> (usize, usize) {
        match self {
            WindowFrame::Cumulative => (0, pos),
            WindowFrame::Rows { preceding, following } => (
                pos.saturating_sub(*preceding),
                (pos + following).min(len - 1),
            ),
        }
    }
}

impl std::fmt::Display for WindowFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowFrame::Cumulative => {
                f.write_str("ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW")
            }
            WindowFrame::Rows { preceding, following } => write!(
                f,
                "ROWS BETWEEN {preceding} PRECEDING AND {following} FOLLOWING"
            ),
        }
    }
}

// ─── Window expression ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WindowExpr {
    pub func:         WindowFunc,
    /// Input column expression (None for ROW_NUMBER / RANK / DENSE_RANK / COUNT(*)).
    pub input:        Option<Expr>,
    pub partition_by: Vec<Expr>,
    pub order_by:     Vec<super::sort::SortExpr>,
    /// Frame for the aggregate functions (SUM/AVG/MIN/MAX/COUNT). Ranking and
    /// offset functions ignore it — their meaning is fixed by the ordering.
    pub frame:        WindowFrame,
    pub alias:        String,
}

impl WindowExpr {
    /// Compute the window column for `batch` (all rows must be in one batch).
    pub fn execute(&self, batch: &RecordBatch) -> FalconResult<Column> {
        let n = batch.row_count();
        if n == 0 {
            return Ok(Column::Int64(PrimitiveColumn::new(vec![])));
        }

        // 1. Evaluate partition keys (one Vec<ScalarValue> per PARTITION BY expression).
        let part_vals: Vec<Vec<ScalarValue>> = self.partition_by.iter()
            .map(|e| e.eval_batch(batch))
            .collect::<FalconResult<_>>()?;

        // 2. Evaluate ORDER BY columns inside the window.
        let order_vals: Vec<Vec<ScalarValue>> = self.order_by.iter()
            .map(|s| s.expr.eval_batch(batch))
            .collect::<FalconResult<_>>()?;

        // 3. Evaluate input column (for SUM/AVG/LAG/LEAD etc.).
        let input_vals: Option<Vec<ScalarValue>> = self.input.as_ref()
            .map(|e| e.eval_batch(batch))
            .transpose()?;

        // 4. Group rows into partitions (preserve insertion order for determinism).
        let mut part_map: FastMap<Vec<String>, Vec<usize>> = FastMap::default();
        let mut part_order: Vec<Vec<String>> = Vec::new();

        for row in 0..n {
            let key: Vec<String> = if part_vals.is_empty() {
                vec![]
            } else {
                part_vals.iter().map(|col| col[row].to_string()).collect()
            };
            match part_map.entry(key.clone()) {
                Entry::Vacant(e) => { part_order.push(key); e.insert(vec![row]); }
                Entry::Occupied(mut e) => { e.get_mut().push(row); }
            }
        }

        // 5. Apply window function per partition.
        let mut result: Vec<ScalarValue> = vec![ScalarValue::Null; n];

        for key in &part_order {
            let indices = &part_map[key];

            // Sort within partition by ORDER BY keys.
            let mut sorted = indices.clone();
            if !order_vals.is_empty() {
                sorted.sort_unstable_by(|&a, &b| {
                    for (col, expr) in order_vals.iter().zip(self.order_by.iter()) {
                        let ord = cmp_scalar(&col[a], &col[b]);
                        let ord = match expr.order { SortOrder::Asc => ord, SortOrder::Desc => ord.reverse() };
                        if ord != std::cmp::Ordering::Equal { return ord; }
                    }
                    std::cmp::Ordering::Equal
                });
            }

            apply_window_func(
                &self.func,
                &sorted,
                &order_vals,
                input_vals.as_deref(),
                self.frame,
                &mut result,
            )?;
        }

        Ok(scalars_to_col(&result))
    }
}

// ─── Per-partition function application ──────────────────────────────────────

fn apply_window_func(
    func:       &WindowFunc,
    sorted:     &[usize],
    order_vals: &[Vec<ScalarValue>],
    input_vals: Option<&[ScalarValue]>,
    frame:      WindowFrame,
    result:     &mut [ScalarValue],
) -> FalconResult<()> {
    match func {
        WindowFunc::RowNumber => {
            for (rank, &row) in sorted.iter().enumerate() {
                result[row] = ScalarValue::Int64(Some(rank as i64 + 1));
            }
        }

        WindowFunc::Rank => {
            let mut rank = 1i64;
            for (pos, &row) in sorted.iter().enumerate() {
                if pos > 0 {
                    let prev = sorted[pos - 1];
                    let tied = order_vals.iter().all(|col| col[row] == col[prev]);
                    if !tied { rank = pos as i64 + 1; }
                }
                result[row] = ScalarValue::Int64(Some(rank));
            }
        }

        WindowFunc::DenseRank => {
            let mut rank = 1i64;
            for (pos, &row) in sorted.iter().enumerate() {
                if pos > 0 {
                    let prev = sorted[pos - 1];
                    let tied = order_vals.iter().all(|col| col[row] == col[prev]);
                    if !tied { rank += 1; }
                }
                result[row] = ScalarValue::Int64(Some(rank));
            }
        }

        WindowFunc::Lag(offset) => {
            let vals = need_input(input_vals, "LAG")?;
            for (pos, &row) in sorted.iter().enumerate() {
                result[row] = if pos >= *offset {
                    vals[sorted[pos - *offset]].clone()
                } else {
                    ScalarValue::Null
                };
            }
        }

        WindowFunc::Lead(offset) => {
            let vals = need_input(input_vals, "LEAD")?;
            let len = sorted.len();
            for (pos, &row) in sorted.iter().enumerate() {
                result[row] = if pos + *offset < len {
                    vals[sorted[pos + *offset]].clone()
                } else {
                    ScalarValue::Null
                };
            }
        }

        WindowFunc::Sum => {
            let vals = need_input(input_vals, "SUM")?;
            frame_sum_avg(sorted, vals, frame, result, false);
        }

        WindowFunc::Avg => {
            let vals = need_input(input_vals, "AVG")?;
            frame_sum_avg(sorted, vals, frame, result, true);
        }

        WindowFunc::Min => {
            let vals = need_input(input_vals, "MIN")?;
            frame_min_max(sorted, vals, frame, result, true);
        }

        WindowFunc::Max => {
            let vals = need_input(input_vals, "MAX")?;
            frame_min_max(sorted, vals, frame, result, false);
        }

        WindowFunc::Count => {
            // COUNT(*) counts frame rows; COUNT(expr) counts non-NULL ones.
            let len = sorted.len();
            for (pos, &row) in sorted.iter().enumerate() {
                let (lo, hi) = frame.bounds(pos, len);
                let n = match input_vals {
                    None => (hi - lo + 1) as i64,
                    Some(vals) => sorted[lo..=hi]
                        .iter()
                        .filter(|&&r| !vals[r].is_null())
                        .count() as i64,
                };
                result[row] = ScalarValue::Int64(Some(n));
            }
        }
    }
    Ok(())
}

// ─── Frame-aware aggregates ───────────────────────────────────────────────────

/// Rolling SUM (or AVG when `want_avg`) over the frame, in O(n).
///
/// Both frame shapes advance monotonically — `lo` and `hi` never move backwards as
/// `pos` increases — so the running total is maintained by adding entering rows and
/// subtracting leaving ones rather than re-summing the frame at every row.
///
/// Sums are kept in f64 and are therefore subject to floating-point drift; adding
/// and subtracting is not exactly associative, so a long rolling sum can differ in
/// the last bits from summing each frame independently.
fn frame_sum_avg(
    sorted:   &[usize],
    vals:     &[ScalarValue],
    frame:    WindowFrame,
    result:   &mut [ScalarValue],
    want_avg: bool,
) {
    let len = sorted.len();
    let mut total = 0.0f64;
    let mut count = 0i64;
    // Half-open [cur_lo, cur_hi) tracking which positions are currently summed.
    let mut cur_lo = 0usize;
    let mut cur_hi = 0usize;

    for (pos, &row) in sorted.iter().enumerate() {
        let (lo, hi) = frame.bounds(pos, len);
        let hi_ex = hi + 1;

        while cur_hi < hi_ex {
            if let Some(v) = vals[sorted[cur_hi]].as_f64() { total += v; count += 1; }
            cur_hi += 1;
        }
        while cur_lo < lo {
            if let Some(v) = vals[sorted[cur_lo]].as_f64() { total -= v; count -= 1; }
            cur_lo += 1;
        }

        result[row] = if count > 0 {
            let out = if want_avg { total / count as f64 } else { total };
            ScalarValue::Float64(Some(out))
        } else {
            ScalarValue::Null
        };
    }
}

/// Rolling MIN (or MAX when `!want_min`) over the frame, in O(n) amortised.
///
/// Uses a monotonic deque of positions: the front always holds the extreme of the
/// current frame. A value that enters while worse than the incumbent can never
/// become the answer before that incumbent leaves, so it is dropped on arrival —
/// which is what keeps the total work linear rather than O(n × frame width).
fn frame_min_max(
    sorted:   &[usize],
    vals:     &[ScalarValue],
    frame:    WindowFrame,
    result:   &mut [ScalarValue],
    want_min: bool,
) {
    use std::collections::VecDeque;

    let len = sorted.len();
    let mut deque: VecDeque<usize> = VecDeque::new(); // positions, values monotonic
    let mut cur_hi = 0usize;

    for (pos, &row) in sorted.iter().enumerate() {
        let (lo, hi) = frame.bounds(pos, len);
        let hi_ex = hi + 1;

        while cur_hi < hi_ex {
            if let Some(v) = vals[sorted[cur_hi]].as_f64() {
                while let Some(&back) = deque.back() {
                    let bv = vals[sorted[back]].as_f64().unwrap_or(f64::NAN);
                    let dominated = if want_min { bv >= v } else { bv <= v };
                    if dominated { deque.pop_back(); } else { break; }
                }
                deque.push_back(cur_hi);
            }
            cur_hi += 1;
        }
        // Drop entries that have fallen out of the left edge of the frame.
        while let Some(&front) = deque.front() {
            if front < lo { deque.pop_front(); } else { break; }
        }

        result[row] = match deque.front() {
            Some(&p) => ScalarValue::Float64(vals[sorted[p]].as_f64()),
            None     => ScalarValue::Null,
        };
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn need_input<'a>(vals: Option<&'a [ScalarValue]>, func: &str) -> FalconResult<&'a [ScalarValue]> {
    vals.ok_or_else(|| FalconError::InvalidQuery(format!("{func} requires an input expression")))
}

fn cmp_scalar(a: &ScalarValue, b: &ScalarValue) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.is_null(), b.is_null()) {
        (true, true)   => Ordering::Equal,
        (true, false)  => Ordering::Greater,
        (false, true)  => Ordering::Less,
        (false, false) => {
            // Temporal types first: `as_f64` does not cover them, so without these
            // arms a window ORDER BY on a date would rank every row as a tie.
            match (a, b) {
                (ScalarValue::Date32(Some(da)), ScalarValue::Date32(Some(db))) => da.cmp(db),
                (ScalarValue::Timestamp(Some(ta)), ScalarValue::Timestamp(Some(tb))) => ta.cmp(tb),
                _ => {
                    if let (Some(fa), Some(fb)) = (a.as_f64(), b.as_f64()) {
                        fa.partial_cmp(&fb).unwrap_or(Ordering::Equal)
                    } else if let (ScalarValue::Utf8(Some(sa)), ScalarValue::Utf8(Some(sb))) = (a, b) {
                        sa.cmp(sb)
                    } else {
                        Ordering::Equal
                    }
                }
            }
        }
    }
}

fn scalars_to_col(vals: &[ScalarValue]) -> Column {
    let dt = vals.iter().find(|v| !v.is_null())
        .map(|v| v.data_type())
        .unwrap_or(DataType::Int64);
    match &dt {
        DataType::Int64 | DataType::Int32 =>
            Column::Int64(PrimitiveColumn::new(vals.iter().map(|v| v.as_i64()).collect())),
        DataType::Float64 | DataType::Float32 =>
            Column::Float64(PrimitiveColumn::new(vals.iter().map(|v| v.as_f64()).collect())),
        DataType::Utf8 =>
            Column::Utf8(Utf8Column::new(vals.iter().map(|v| match v {
                ScalarValue::Utf8(s) => s.clone(), _ => None,
            }).collect())),
        DataType::Boolean =>
            Column::Boolean(crate::storage::column::BooleanColumn::new(
                vals.iter().map(|v| v.as_bool()).collect(),
            )),
        // Temporal values carry their payload in a variant `as_i64` does not read,
        // so without these arms LAG/LEAD/FIRST_VALUE over a date yield all NULLs.
        DataType::Date32 =>
            Column::Date32(PrimitiveColumn::new(vals.iter().map(|v| match v {
                ScalarValue::Date32(d) => *d, _ => None,
            }).collect())),
        DataType::Timestamp =>
            Column::Timestamp(PrimitiveColumn::new(vals.iter().map(|v| match v {
                ScalarValue::Timestamp(t) => *t, _ => None,
            }).collect())),
        DataType::Decimal128(p, s) =>
            Column::Decimal128(crate::storage::column::Decimal128Column::new(
                vals.iter().map(|v| match v {
                    ScalarValue::Decimal128(d) => d.value, _ => None,
                }).collect(),
                *p,
                *s,
            )),
        _ =>
            Column::Int64(PrimitiveColumn::new(vals.iter().map(|v| v.as_i64()).collect())),
    }
}

/// Add extra columns (window results) to a RecordBatch.
pub fn add_window_columns(
    batch:  RecordBatch,
    fields: Vec<Field>,
    cols:   Vec<Column>,
) -> FalconResult<RecordBatch> {
    let mut all_fields  = batch.schema.fields.clone();
    let mut all_columns = batch.columns;
    all_fields.extend(fields);
    all_columns.extend(cols.into_iter().map(std::sync::Arc::new));
    RecordBatch::from_shared(crate::storage::schema::Schema::new(all_fields), all_columns)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        query::{expr::Expr, sort::SortExpr},
        storage::{column::ColumnBuilder, schema::{Field, Schema}, table::RecordBatch, types::DataType},
    };

    fn make_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::required("dept",   DataType::Utf8),
            Field::required("salary", DataType::Float64),
        ]);
        RecordBatch::new(schema, vec![
            ColumnBuilder::utf8_from_str(vec![
                Some("eng"), Some("eng"), Some("hr"), Some("hr"), Some("eng"),
            ]),
            ColumnBuilder::float64(vec![
                Some(100.0), Some(120.0), Some(80.0), Some(90.0), Some(110.0),
            ]),
        ]).unwrap()
    }

    #[test]
    fn row_number_partition_by_dept() {
        let batch = make_batch();
        let expr = WindowExpr {
            func:         WindowFunc::RowNumber,
            input:        None,
            partition_by: vec![Expr::Column("dept".into())],
            order_by:     vec![SortExpr::desc(Expr::Column("salary".into()))],
            frame:        WindowFrame::Cumulative,
            alias:        "rn".into(),
        };
        let col = expr.execute(&batch).unwrap();
        // eng rows: salaries 100, 120, 110 → sorted desc → 120(rn=1), 110(rn=2), 100(rn=3)
        // eng is at batch rows 0, 1, 4 → values 100, 120, 110
        // after sort desc: row1(120)→rn=1, row4(110)→rn=2, row0(100)→rn=3
        assert_eq!(col.get(0).unwrap(), ScalarValue::Int64(Some(3))); // 100 → rank 3
        assert_eq!(col.get(1).unwrap(), ScalarValue::Int64(Some(1))); // 120 → rank 1
        assert_eq!(col.get(4).unwrap(), ScalarValue::Int64(Some(2))); // 110 → rank 2
    }

    #[test]
    fn rank_with_ties() {
        let schema = Schema::new(vec![
            Field::required("score", DataType::Float64),
        ]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::float64(vec![Some(10.0), Some(10.0), Some(8.0)]),
        ]).unwrap();
        let expr = WindowExpr {
            func:         WindowFunc::Rank,
            input:        None,
            partition_by: vec![],
            order_by:     vec![SortExpr::desc(Expr::Column("score".into()))],
            frame:        WindowFrame::Cumulative,
            alias:        "r".into(),
        };
        let col = expr.execute(&batch).unwrap();
        // 10, 10 → rank 1, 1; then 8 → rank 3
        assert_eq!(col.get(0).unwrap(), ScalarValue::Int64(Some(1)));
        assert_eq!(col.get(1).unwrap(), ScalarValue::Int64(Some(1)));
        assert_eq!(col.get(2).unwrap(), ScalarValue::Int64(Some(3)));
    }

    #[test]
    fn dense_rank_with_ties() {
        let schema = Schema::new(vec![Field::required("score", DataType::Float64)]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::float64(vec![Some(10.0), Some(10.0), Some(8.0)]),
        ]).unwrap();
        let expr = WindowExpr {
            func:         WindowFunc::DenseRank,
            input:        None,
            partition_by: vec![],
            order_by:     vec![SortExpr::desc(Expr::Column("score".into()))],
            frame:        WindowFrame::Cumulative,
            alias:        "dr".into(),
        };
        let col = expr.execute(&batch).unwrap();
        assert_eq!(col.get(0).unwrap(), ScalarValue::Int64(Some(1)));
        assert_eq!(col.get(1).unwrap(), ScalarValue::Int64(Some(1)));
        assert_eq!(col.get(2).unwrap(), ScalarValue::Int64(Some(2)));
    }

    #[test]
    fn lag_basic() {
        let schema = Schema::new(vec![Field::required("v", DataType::Float64)]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::float64(vec![Some(1.0), Some(2.0), Some(3.0)]),
        ]).unwrap();
        let expr = WindowExpr {
            func:         WindowFunc::Lag(1),
            input:        Some(Expr::Column("v".into())),
            partition_by: vec![],
            order_by:     vec![SortExpr::asc(Expr::Column("v".into()))],
            frame:        WindowFrame::Cumulative,
            alias:        "lag_v".into(),
        };
        let col = expr.execute(&batch).unwrap();
        assert!(col.get(0).unwrap().is_null()); // first row has no previous
        assert_eq!(col.get(1).unwrap(), ScalarValue::Float64(Some(1.0)));
        assert_eq!(col.get(2).unwrap(), ScalarValue::Float64(Some(2.0)));
    }

    #[test]
    fn running_sum() {
        let schema = Schema::new(vec![Field::required("v", DataType::Float64)]);
        let batch = RecordBatch::new(schema, vec![
            ColumnBuilder::float64(vec![Some(1.0), Some(2.0), Some(3.0)]),
        ]).unwrap();
        let expr = WindowExpr {
            func:         WindowFunc::Sum,
            input:        Some(Expr::Column("v".into())),
            partition_by: vec![],
            order_by:     vec![SortExpr::asc(Expr::Column("v".into()))],
            frame:        WindowFrame::Cumulative,
            alias:        "running_sum".into(),
        };
        let col = expr.execute(&batch).unwrap();
        assert_eq!(col.get(0).unwrap(), ScalarValue::Float64(Some(1.0)));
        assert_eq!(col.get(1).unwrap(), ScalarValue::Float64(Some(3.0)));
        assert_eq!(col.get(2).unwrap(), ScalarValue::Float64(Some(6.0)));
    }
}
