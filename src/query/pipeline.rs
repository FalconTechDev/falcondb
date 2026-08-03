// Push-based, vectorised execution.
//
// The existing executor materialises a full `Vec<RecordBatch>` between every
// step: filter the table and allocate the result, group it and allocate again,
// sort it and allocate again. Each hand-off is a complete pass through main
// memory, and for a pipeline of N steps the data is written and re-read N times.
//
// This module models execution the other way round. Data moves in fixed-size
// chunks, and each chunk is pushed through the whole operator chain before the
// next one is read. A chunk of 2048 rows is ~16 KB per column, so an operator
// chain runs against data that is still in L1/L2 from the step before.
//
// ## Operator kinds
//
// **Streaming** operators (filter, project) turn one chunk into at most one
// chunk and hold no state. They are pure wins here: no intermediate allocation
// at all.
//
// **Blocking** operators (sort, aggregate) cannot answer until they have seen
// every row. They accumulate during `process` and produce during `flush`. They
// do not save memory, but they still sit inside the pipeline so everything
// upstream and downstream of them streams.
//
// ## Why this is built alongside the existing executor, not in place of it
//
// A half-ported engine is a broken engine. Operators are added here one at a
// time and checked against the executor they will eventually replace; the
// planner keeps using the old path until the whole chain exists. The 502
// behavioural tests stay valid throughout, because they exercise SQL rather
// than internals.

use crate::storage::{
    table::RecordBatch,
    types::FalconResult,
};

/// Rows per chunk.
///
/// 2048 is DuckDB's vector size and the reasoning is the same: one chunk of a
/// single 8-byte column is 16 KB, so a handful of columns still fit in L1 and an
/// operator chain runs without spilling to memory between steps. Larger chunks
/// amortise per-chunk overhead better but start missing cache; smaller ones pay
/// the per-call cost too often.
pub const CHUNK_ROWS: usize = 2048;

/// What an operator did with a chunk.
#[derive(Debug)]
pub enum OpResult {
    /// Pass this chunk downstream.
    Emit(RecordBatch),
    /// Nothing to pass on — every row was filtered out, or the operator is
    /// still accumulating.
    Empty,
    /// This operator will not accept more input. Upstream can stop reading,
    /// which is what makes `LIMIT` cheap: the scan ends early.
    Done,
}

/// One stage of a pipeline.
pub trait Operator: Send {
    /// Handle one chunk of input.
    fn process(&mut self, chunk: RecordBatch) -> FalconResult<OpResult>;

    /// Called once, after the last chunk. Blocking operators emit their result
    /// here; streaming operators have nothing to add.
    fn flush(&mut self) -> FalconResult<Vec<RecordBatch>> {
        Ok(Vec::new())
    }

    /// Name for EXPLAIN output and error messages.
    fn name(&self) -> &'static str;
}

/// Split a batch into chunks of at most `CHUNK_ROWS` rows.
///
/// Slicing shares the underlying columns, so this does not copy data.
pub fn into_chunks(batch: &RecordBatch) -> FalconResult<Vec<RecordBatch>> {
    let n = batch.row_count();
    if n <= CHUNK_ROWS {
        return Ok(vec![batch.clone()]);
    }
    let mut out = Vec::with_capacity(n.div_ceil(CHUNK_ROWS));
    let mut start = 0;
    while start < n {
        let take = CHUNK_ROWS.min(n - start);
        out.push(batch.slice(start, take)?);
        start += take;
    }
    Ok(out)
}

/// A chain of operators, driven by pushing chunks through it.
pub struct Pipeline {
    ops: Vec<Box<dyn Operator>>,
}

impl Pipeline {
    pub fn new(ops: Vec<Box<dyn Operator>>) -> Self {
        Self { ops }
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Names of the stages, outermost first. Used by EXPLAIN.
    pub fn stage_names(&self) -> Vec<&'static str> {
        self.ops.iter().map(|o| o.name()).collect()
    }

    /// Run every chunk of `input` through the chain and collect the result.
    ///
    /// Input batches are re-chunked to `CHUNK_ROWS` first, so the pipeline sees a
    /// uniform vector size regardless of how the source happened to batch its
    /// data — Parquet hands back 1024-row batches, CSV hands back one huge one.
    pub fn run(&mut self, input: &[RecordBatch]) -> FalconResult<Vec<RecordBatch>> {
        let mut out = Vec::new();

        'source: for batch in input {
            for chunk in into_chunks(batch)? {
                match self.push_from(0, chunk, &mut out)? {
                    Flow::Continue => {}
                    // A downstream operator is satisfied (LIMIT), so stop reading.
                    Flow::Stop => break 'source,
                }
            }
        }

        self.flush_from(0, &mut out)?;
        Ok(out)
    }

    /// Push one chunk into stage `idx` and onward.
    fn push_from(
        &mut self,
        idx:   usize,
        chunk: RecordBatch,
        out:   &mut Vec<RecordBatch>,
    ) -> FalconResult<Flow> {
        // Past the last stage: this chunk is a result.
        if idx >= self.ops.len() {
            if chunk.row_count() > 0 {
                out.push(chunk);
            }
            return Ok(Flow::Continue);
        }

        match self.ops[idx].process(chunk)? {
            OpResult::Emit(next) => self.push_from(idx + 1, next, out),
            OpResult::Empty      => Ok(Flow::Continue),
            OpResult::Done       => Ok(Flow::Stop),
        }
    }

    /// Flush stage `idx`, then feed whatever it produced through the stages
    /// after it, then flush those in turn.
    ///
    /// The order matters: a blocking operator's output has not been seen by
    /// anything downstream yet, so it has to travel the rest of the chain before
    /// those stages are themselves flushed.
    fn flush_from(&mut self, idx: usize, out: &mut Vec<RecordBatch>) -> FalconResult<()> {
        if idx >= self.ops.len() {
            return Ok(());
        }

        for batch in self.ops[idx].flush()? {
            for chunk in into_chunks(&batch)? {
                // A `Done` here means a downstream LIMIT is satisfied; the
                // remaining flushed rows are simply not needed.
                if let Flow::Stop = self.push_from(idx + 1, chunk, out)? {
                    break;
                }
            }
        }

        self.flush_from(idx + 1, out)
    }
}

/// Whether the source should keep feeding the pipeline.
enum Flow {
    Continue,
    Stop,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        column::ColumnBuilder,
        schema::{Field, Schema},
        types::DataType,
    };

    fn batch(vals: Vec<i64>) -> RecordBatch {
        let schema = Schema::new(vec![Field::required("v", DataType::Int64)]);
        RecordBatch::new(schema, vec![ColumnBuilder::int64(vals.into_iter().map(Some).collect())])
            .unwrap()
    }

    fn values(batches: &[RecordBatch]) -> Vec<i64> {
        let mut out = Vec::new();
        for b in batches {
            let c = b.column(0);
            for r in 0..b.row_count() {
                out.push(c.get(r).unwrap().as_i64().unwrap());
            }
        }
        out
    }

    /// Passes chunks through untouched — checks the driver itself.
    struct Identity;
    impl Operator for Identity {
        fn process(&mut self, chunk: RecordBatch) -> FalconResult<OpResult> {
            Ok(OpResult::Emit(chunk))
        }
        fn name(&self) -> &'static str { "Identity" }
    }

    /// Blocking: swallows every chunk, emits everything at flush.
    struct Collect { held: Vec<RecordBatch> }
    impl Operator for Collect {
        fn process(&mut self, chunk: RecordBatch) -> FalconResult<OpResult> {
            self.held.push(chunk);
            Ok(OpResult::Empty)
        }
        fn flush(&mut self) -> FalconResult<Vec<RecordBatch>> {
            Ok(std::mem::take(&mut self.held))
        }
        fn name(&self) -> &'static str { "Collect" }
    }

    /// Stops the pipeline after `n` rows.
    struct TakeN { left: usize }
    impl Operator for TakeN {
        fn process(&mut self, chunk: RecordBatch) -> FalconResult<OpResult> {
            if self.left == 0 {
                return Ok(OpResult::Done);
            }
            let take = self.left.min(chunk.row_count());
            self.left -= take;
            Ok(OpResult::Emit(chunk.slice(0, take)?))
        }
        fn name(&self) -> &'static str { "TakeN" }
    }

    #[test]
    fn chunking_preserves_values_and_order() {
        let b = batch((0..5_000).collect());
        let chunks = into_chunks(&b).unwrap();
        assert!(chunks.len() > 1, "input should have been split");
        assert!(chunks.iter().all(|c| c.row_count() <= CHUNK_ROWS));
        assert_eq!(values(&chunks), (0..5_000).collect::<Vec<_>>());
    }

    #[test]
    fn small_batch_is_one_chunk() {
        let b = batch((0..10).collect());
        assert_eq!(into_chunks(&b).unwrap().len(), 1);
    }

    #[test]
    fn empty_pipeline_returns_its_input() {
        let mut p = Pipeline::new(vec![]);
        let got = p.run(&[batch((0..3_000).collect())]).unwrap();
        assert_eq!(values(&got), (0..3_000).collect::<Vec<_>>());
    }

    #[test]
    fn streaming_stage_preserves_everything() {
        let mut p = Pipeline::new(vec![Box::new(Identity), Box::new(Identity)]);
        let got = p.run(&[batch((0..5_000).collect())]).unwrap();
        assert_eq!(values(&got), (0..5_000).collect::<Vec<_>>());
    }

    #[test]
    fn blocking_stage_emits_at_flush() {
        let mut p = Pipeline::new(vec![Box::new(Collect { held: Vec::new() })]);
        let got = p.run(&[batch((0..3_000).collect())]).unwrap();
        assert_eq!(values(&got), (0..3_000).collect::<Vec<_>>());
    }

    #[test]
    fn stage_after_a_blocking_stage_still_sees_the_rows() {
        // The regression this guards: flushing a blocking operator without
        // routing its output through the rest of the chain loses those rows.
        let mut p = Pipeline::new(vec![
            Box::new(Collect { held: Vec::new() }),
            Box::new(TakeN { left: 10 }),
        ]);
        let got = p.run(&[batch((0..3_000).collect())]).unwrap();
        assert_eq!(values(&got), (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn two_blocking_stages_chain() {
        let mut p = Pipeline::new(vec![
            Box::new(Collect { held: Vec::new() }),
            Box::new(Collect { held: Vec::new() }),
        ]);
        let got = p.run(&[batch((0..2_500).collect())]).unwrap();
        assert_eq!(values(&got), (0..2_500).collect::<Vec<_>>());
    }

    #[test]
    fn done_stops_the_source_early() {
        let mut p = Pipeline::new(vec![Box::new(TakeN { left: 5 })]);
        let got = p.run(&[batch((0..100_000).collect())]).unwrap();
        assert_eq!(values(&got), (0..5).collect::<Vec<_>>());
    }

    #[test]
    fn multiple_input_batches_are_concatenated_in_order() {
        let mut p = Pipeline::new(vec![Box::new(Identity)]);
        let got = p.run(&[batch((0..100).collect()), batch((100..200).collect())]).unwrap();
        assert_eq!(values(&got), (0..200).collect::<Vec<_>>());
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let mut p = Pipeline::new(vec![Box::new(Identity)]);
        assert!(p.run(&[]).unwrap().is_empty());
    }

    #[test]
    fn stage_names_are_reported_in_order() {
        let p = Pipeline::new(vec![
            Box::new(Identity),
            Box::new(Collect { held: Vec::new() }),
        ]);
        assert_eq!(p.stage_names(), vec!["Identity", "Collect"]);
    }
}
