# FalconDB

A high-performance columnar analytical database engine written in Rust.

[![CI](https://github.com/yourusername/falcondb/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/falcondb/actions)
[![Crates.io](https://img.shields.io/crates/v/falcondb.svg)](https://crates.io/crates/falcondb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Why FalconDB?

FalconDB is built from scratch to answer one question: *what does a columnar database look like when you control every layer?*

- **Columnar storage** — numeric columns hold dense values plus a validity bitmap (Arrow-style), so values sit contiguously and the optimiser can vectorise scans. String columns are dictionary-encoded: each distinct string once, a `u32` per row.
- **Parallel execution** — CSV loading, batch filtering, and projection all use [Rayon](https://github.com/rayon-rs/rayon).
- **SIMD-friendly layout** — arithmetic runs over contiguous slices, no boxing, no heap per value.
- **Three compression codecs** — LZ4 for numeric columns, bit-packing for booleans, dictionary encoding for strings.
- **Full SQL** — `SELECT`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `JOIN` (INNER / LEFT / RIGHT), `LIMIT`, `OFFSET`, `UNION ALL`, `DISTINCT`, `CASE WHEN`, `CAST`, `LIKE`/`ILIKE`, `IN (…)`, `BETWEEN`, `IS NULL`, scalar functions, **window functions** (`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`, `SUM/AVG/MIN/MAX OVER`).
- **DDL** — `CREATE TABLE`, `DROP TABLE` (with `IF [NOT] EXISTS`), `ALTER TABLE ADD/DROP COLUMN`. Schemas can now be defined entirely in SQL — no CSV file required.
- **EXPLAIN** — `EXPLAIN <any-statement>` returns the query plan as a result set: Scan, Filter, GroupBy, Window, Sort, Limit, and Join nodes. No side effects.
- **DML** — `INSERT`, `UPDATE`, `DELETE`. Mutations are copy-on-write: because columns are `Arc`-shared, batches a statement does not touch are carried over as pointer copies and only affected batches are rebuilt.
- **CTEs and scalar subqueries** — `WITH a AS (…), b AS (SELECT … FROM a)`, `WITH RECURSIVE`, and `WHERE x > (SELECT AVG(x) FROM t)`.
- **B-tree and hash indices** — range queries and point lookups without touching the storage layer.
- **Write-ahead log** — crash recovery with CRC32-verified records. DML is journalled statement-by-statement; load your base data, then call `db.recover()` to replay. Sound because FalconDB's DML is deterministic (`INSERT` takes literals only, `UPDATE`/`DELETE` expressions are pure functions of stored data) — adding something like `NOW()` would require switching to row-image logging.
- **Native binary format** — `.fdb` files: header + compressed columns, ~2× smaller than CSV.
- **REST API** — load data and run queries over HTTP.
- **C FFI** — use FalconDB from any language that speaks C ABI.

---

## Benchmarks

Measured on **Intel i7-13700H (6 P-cores + 8 E-cores), Windows 11**, Criterion,
median of 100 samples. Against **DuckDB 1.x**, **Polars** and **DataFusion**, all
reading the same Parquet file. Run them with `scripts/bench.ps1` — not a bare
`cargo bench`, for reasons given under [Methodology](#methodology).

### Scans and aggregates

| | rows | FalconDB | Polars | DuckDB | DataFusion |
|---|---|---|---|---|---|
| `SUM(score)` | 10 k | **1.41 µs** | 8.47 µs | 235 µs | 1.16 ms |
| | 100 k | **8.66 µs** | 19.2 µs | 413 µs | 1.50 ms |
| | 1 M | **61.1 µs** | 158 µs | 500 µs | 2.63 ms |
| `SUM(id)` (integer) | 10 k | **2.65 µs** | 7.39 µs | 212 µs | 1.61 ms |
| | 100 k | 21.4 µs | **14.7 µs** | 359 µs | 2.52 ms |
| | 1 M | **76.4 µs** | 176 µs | 473 µs | 14.3 ms |
| `WHERE score > 50` | 10 k | **16.6 µs** | 50.9 µs | 599 µs | 1.19 ms |
| | 100 k | **127 µs** | 173 µs | 3.98 ms | 2.77 ms |
| | 1 M | **671 µs** | 2.24 ms | 24.0 ms | 17.6 ms |

### Grouping, sorting, pipelines

| | rows | FalconDB | Polars | DuckDB | DataFusion |
|---|---|---|---|---|---|
| `GROUP BY` + 3 aggregates | 10 k | **35.9 µs** | 258 µs | 1.85 ms | 1.62 ms |
| | 100 k | **97.0 µs** | 1.08 ms | 3.60 ms | 2.22 ms |
| | 1 M | **581 µs** | 10.2 ms | 6.51 ms | 7.12 ms |
| `ORDER BY score DESC` | 10 k | **242 µs** | 275 µs | 1.86 ms | 938 µs |
| | 100 k | 1.58 ms | **1.38 ms** | 15.7 ms | 6.46 ms |
| `DISTINCT` | 100 k | 702 µs † | 600 µs | 2.02 ms | 1.31 ms |
| | 1 M | **2.42 ms** | 5.21 ms | 3.55 ms | 3.14 ms |
| `WHERE`+`GROUP BY`+`ORDER BY` | 10 k | **22.3 µs** | 280 µs | 1.64 ms | 1.56 ms |
| | 100 k | **91.6 µs** | 1.12 ms | 3.53 ms | 2.08 ms |

† `DISTINCT` at 100 k is the one point this table will not call. FalconDB's
reading in this run spans 616–837 µs where every other point in the suite spans
about 1%, and a re-run of that group alone put FalconDB at 589 µs against Polars'
617 µs — ahead, where the full run had it behind. Two runs disagreeing on the
*sign* is what "too close to measure" looks like; the median above is the full
run's, reported as measured rather than replaced with the reading that flatters
us. At 1 M rows, where the measurement is stable, FalconDB leads.

### Joins, window functions, DML

Polars is absent from the rows below: it has no `EXISTS`, no in-place
`UPDATE`/`DELETE`, and expresses `ROW_NUMBER` through a different shape entirely.
Timing a different formulation against ours is precisely the mistake this file
used to make — see [Methodology](#methodology).

| | rows | FalconDB | Polars | DuckDB | DataFusion |
|---|---|---|---|---|---|
| `INNER JOIN` | 100 k | **307 µs** | 425 µs | 1.77 ms | 3.90 ms |
| | 1 M | 3.41 ms | **2.60 ms** | 7.81 ms | 16.7 ms |
| `LEFT JOIN` | 100 k | **687 µs** | — | 2.44 ms | 4.13 ms |
| `SEMI JOIN` (`EXISTS`) | 100 k | **592 µs** | — | 2.46 ms | 4.04 ms |
| `ANTI JOIN` (`NOT EXISTS`) | 100 k | **856 µs** | — | 2.47 ms | 4.18 ms |
| `ROW_NUMBER() OVER` | 100 k | **4.31 ms** ‡ | — | 11.6 ms | 5.80 ms |
| `UPDATE` (half the rows) | 100 k | **1.69 ms** | — | 3.63 ms | — |
| `DELETE` (half the rows) | 100 k | **914 µs** | — | 2.59 ms | — |

‡ This one moved the wrong way and is not yet explained. The previous run had
3.67 ms; DuckDB measured the same in both and DataFusion improved 7%, so the
rivals held while this slid 17%. Nothing in the window path was touched between
the runs. Recorded here rather than smoothed over, and it is the next thing to
look at.

### Where FalconDB is behind

Ahead of every engine in 22 of 26 measured points, level in one, and ahead of
DuckDB and DataFusion in all 26. The exceptions are all against Polars:

| | FalconDB | Polars | |
|---|---|---|---|
| `SUM(id)` 100 k | 21.4 µs | 14.7 µs | 1.5× |
| `INNER JOIN` 1 M | 3.41 ms | 2.60 ms | 1.3× |
| `ORDER BY` 100 k | 1.58 ms | 1.38 ms | 1.1× |
| `DISTINCT` 100 k | 702 µs | 600 µs | too close to call — see † |

`ORDER BY` narrowed from 1.3× to 1.1×, from sorting a text column by dictionary
rank rather than by string. The other two are within noise of where they were.

The comparison to the previous run is by *ratio*, not by absolute time: the
rivals drift between runs too, so a FalconDB number that improves while a rival's
worsens says less than it appears to. The re-measurement of `DISTINCT` above is
the clearest demonstration in this file — every engine in that run, ours and all
three rivals, read 15–35% faster than in the full suite purely because fewer
groups ran before it.

Most of it came from one shape of defect, found twice in an hour: a code path
that handles NULLs, running over columns that have none, with no cheap check to
skip the handling. `Column::gather_u32` tested validity and wrote a bitmap bit
per output row -- the path every LEFT, SEMI and ANTI join builds its output
through. Extracting sort keys asked the same question of every row. Both now
check once. `PrimitiveColumn::filter` had always done so, which is how they
became visible: the comparison was against a sibling that already did it right.

What is left of the join gap is the build and the probe, not the plumbing. Of a
1 M-row join, building the hash table is about 1.0 ms on one thread while the
probe uses every core. Three ways of parallelising or shrinking it were measured
and rejected -- see `IntHashTable::build_parallel`, `build_radix` and
`bucket_density` -- which is how it became clear the build is bound by memory it
has to touch once, not by work that can be spread.

`SUM(id)` at 100 k runs on one core because the measured aggregate threshold is
178k rows and a serial sum really is faster below it. Polars reaches 54 GB/s on
800 KB, more than one core gets from L2 here, so it is not doing the same thing.

### How much room is actually left

"Fast" against a rival says nothing about how much the hardware could do, so
`tests/diag_bandwidth.rs` measures the floor directly: summing `f64` at working
sets that sit in L2, in L3 and past L3.

| working set | one thread | all threads |
|---|---|---|
| 0.1 MB (L2) | 50–101 GB/s | 8.6–30.5 GB/s — splitting costs more than it saves |
| 8 MB (L3) | 37–39 GB/s | 104–145 GB/s |
| 256 MB (DRAM) | 8.8–16.4 GB/s | **26–57 GB/s** |

Those are ranges because on this laptop they are not constants. Four
back-to-back runs of the same diagnostic read 40.8, 40.6, 25.9 and 31.3 GB/s
from DRAM, and the L3 figure fell monotonically across them — 130.5, 125.6,
107.7, 103.9 — while a single-threaded, cache-resident kernel measured at its
best the whole time. Sustained multi-core traffic appears to spend a power
budget that the memory side gives up before the cores do; the machine is cool to
the touch throughout.

So the "ceiling" moves by up to 2× depending on what the machine has been doing
for the last few minutes. Two things follow, and the second is the one that
matters:

- A 1 M-row `f64` column is 8 MB and fits this CPU's 24 MB L3, so that scan
  never touches DRAM. Its floor is 55–77 µs depending on which reading you take.
  FalconDB measures 61 µs, which lands inside that band — near the floor on a
  good reading, at it on a bad one.
- Any statement of the form "we are N× off the hardware limit" is worth less
  here than it looks, including the ones this file used to make. The limit was
  quoted as 144.9 GB/s from a single reading; it is a range, and the range is
  wide enough to change the conclusion.

This only applies to sequential reads. `GROUP BY`, joins and sorts are bound by
random-access latency instead, and that ceiling has not been measured — quoting
the bandwidth figure for them would flatter every engine, including this one.

### Methodology

The numbers above are worth what the method is worth, and two of the corrections
below changed the published figures by an order of magnitude.

**Pinned to the P-cores.** On a hybrid CPU, rayon's threads land on a different
P/E mix each run, and unpinned run-to-run noise on *unchanged code* measured
25–46% — larger than most of the effects being chased. `scripts/bench.ps1` pins
the process and raises its priority; the same comparison then lands within ~1%.
Discard the first run: a cold one is systematically slower.

**Every engine is asked the same question.** `load_polars` used to hand the timed
closure a lazy `scan_parquet`, so every Polars iteration re-read and re-decoded
the file while the others queried memory. Every "N× faster than Polars" figure
published before that fix was measuring an in-memory query against a file read.
All four engines now load once, outside the timed region.

**One allocator, chosen explicitly.** All four engines run in one process, so
they share an allocator and its state carries from one group to the next. On the
Windows system heap that was load-bearing rather than incidental: FalconDB's join
probe builds its output in per-unit slices, allocated on a worker thread and
freed on another, and Polars — measured *after* FalconDB in the same process —
went from 3.61 ms to 8.24 ms because of it. Reverting the probe brought Polars
back, which is how the mechanism was established instead of assumed. The bench
now sets mimalloc. Polars returns to 2.89 ms, better than it ever measured on the
system heap; DataFusion gained 29%, DuckDB nothing.

This is not a FalconDB optimisation and no figure above should be read as one.
It does not fix the allocation pattern that exposed it either — building a join's
output in twenty 40 KB pieces instead of one 800 KB piece is a real cost imposed
on whatever else shares the process, and a library does not get to choose the
application's allocator. Every number in this file was re-measured after the
change; figures published before it are not comparable.

**Every number above comes from one run.** Engines in a row are therefore
comparable with each other, which is the comparison this file makes. Numbers are
not comparable *across* runs, and on this laptop that is not a small effect: the
same unchanged function measured 12.26 µs in one run and 5.60 µs in another,
minutes apart, because the machine throttles continuously rather than being
either fine or not.

A ratio taken inside one run is safer but not automatically safe. Comparing two
shapes of the same summation kernel read 0.58× on a throttled machine and 0.93×
on a clean one — two things bound by different resources separate when the clock
moves, and a figure of 42% published from the first reading was wrong. Anything
small enough to be interesting is measured by calling both versions inside one
process, alternately, with an unchanged function alongside them as a reference
for the machine's own speed. That is what the `#[ignore]`d measurement tests in
`src/query/join.rs`, `src/query/sort.rs` and `src/query/aggregate.rs` do.

**Results are asserted, not assumed.** For a long time this suite reported the
same ~4 µs filter at 10 k, 100 k and 1 M rows. The times were real; the data was
not — `load_parquet` kept only the first 1024-row batch while reporting the full
row count. A wrong-sized input produces a perfectly self-consistent measurement,
so every FalconDB closure now asserts its row count with `assert_eq!` — never
`debug_assert!`, which the bench profile compiles out.

---

## Quick start

```rust
use falcondb::db::Database;

fn main() {
    let db = Database::in_memory();

    db.load_csv("data/employees.csv", "employees").unwrap();

    let result = db.sql("
        SELECT dept, COUNT(*) as headcount, AVG(salary) as avg_salary
        FROM employees
        WHERE salary > 50000
        GROUP BY dept
        ORDER BY avg_salary DESC
        LIMIT 10
    ").unwrap();

    print!("{}", result.display());
}
```

```
+----------+-----------+------------+
| dept     | headcount | avg_salary |
+----------+-----------+------------+
| eng      | 142       | 112400     |
| finance  | 58        | 98700      |
| ops      | 91        | 76200      |
+----------+-----------+------------+
(3 rows)
```

---

## REST API

Start the server:

```bash
cargo run --bin falcondb-server
# or with custom bind address:
FALCONDB_BIND=0.0.0.0:9000 cargo run --bin falcondb-server
```

### Endpoints

```
GET  /health                     # liveness check
GET  /tables                     # list all loaded tables
GET  /tables/{name}              # schema + row count
DELETE /tables/{name}            # drop table

POST /load/csv   { "path": "...", "table": "..." }
POST /load/fdb   { "path": "...", "table": "..." }
POST /tables/{name}/save  { "path": "..." }

POST /sql        { "sql": "SELECT ..." }
```

### Example

```bash
# Load data
curl -s -X POST localhost:8080/load/csv \
  -H 'Content-Type: application/json' \
  -d '{"path": "data/employees.csv", "table": "emp"}'

# Query
curl -s -X POST localhost:8080/sql \
  -H 'Content-Type: application/json' \
  -d '{"sql": "SELECT dept, COUNT(*) as n FROM emp GROUP BY dept"}' | jq .
```

```json
{
  "columns": ["dept", "n"],
  "rows": [["eng", "142"], ["hr", "38"]],
  "row_count": 2,
  "elapsed_ms": 3
}
```

---

## C FFI

```c
#include "falcondb.h"

falcondb_db* db = falcondb_open();
falcondb_load_csv(db, "employees.csv", "emp");

falcondb_result* r = falcondb_sql(db, "SELECT * FROM emp LIMIT 5");
printf("%s\n", falcondb_result_display(r));

falcondb_result_free(r);
falcondb_close(db);
```

Build with:
```bash
cargo build --release
# links against target/release/libfalcondb.so (Linux) or .dylib (macOS)
```

---

## Architecture

```
SQL string
    │
    ▼
plan_cache (RwLock<HashMap<String, Arc<ParsedQuery>>>)
    │ hit: Arc::clone — no parse, no alloc
    │ miss: parse_sql() → sqlparser-rs AST → Expr tree → cache insert
    ▼
SelectQuery + GroupByExec + SortExpr + WindowExpr
    │
    ▼
execute_select()          — Rayon parallel filter + projection
    │                        fast path: try_fast_mask() on &[Option<T>] — no boxing
    │                        WHERE mask: Vec<u64>, 1 bit/row — 8× smaller than Vec<bool>
    ▼
GroupByExec::execute()    — fast path: Rayon par_iter() over batches
    │                        each batch: HashMap<String, u32> + Vec<f64> accumulators
    │                        merge phase: sequential HashMap merge of partial results
    │                        COUNT(*) shortcut: sum batch.row_count(), no column read
    ▼
WindowExpr::execute()     — ROW_NUMBER / RANK / DENSE_RANK / LAG / LEAD / SUM / AVG / MIN / MAX
    │                        PARTITION BY + ORDER BY per window clause
    ▼
execute_sort()            — par_sort_unstable_by (Rayon) at ≥ 65 536 rows, serial below
    │
    ▼
QueryResult               — Vec<RecordBatch>
```

### Storage stack

| Layer        | What it does                                         |
|--------------|------------------------------------------------------|
| `Column`     | Typed `Vec<Option<T>>` — the atomic unit             |
| `RecordBatch`| Schema-aligned slice of columns, fixed row count     |
| `Table`      | Vec of batches — split for parallel processing       |
| `compression`| LZ4 / bit-pack / dictionary per column              |
| `native`     | `.fdb` binary file format (header + compressed cols) |
| `index`      | B-tree (range) + hash (point lookup) per column      |
| `wal`        | Append-only WAL with CRC32 + checkpoint support      |

---

## Project structure

```
falcondb/
├── src/
│   ├── lib.rs
│   ├── db.rs               ← top-level Database API
│   ├── storage/
│   │   ├── types.rs        ← DataType, ScalarValue, FalconError
│   │   ├── column.rs       ← Column, PrimitiveColumn, Utf8Column
│   │   ├── schema.rs       ← Field, Schema
│   │   ├── table.rs        ← RecordBatch, Table
│   │   ├── csv.rs          ← parallel CSV loader with type inference
│   │   ├── compression.rs  ← LZ4 / bit-pack / dictionary encoding
│   │   ├── index.rs        ← BTreeIndex, HashIndex, IndexRegistry
│   │   ├── native.rs       ← .fdb file format (read / write)
│   │   └── wal.rs          ← write-ahead log with CRC32
│   ├── query/
│   │   ├── expr.rs         ← expression tree + eval_batch / eval_mask / try_fast_mask
│   │   ├── select.rs       ← SELECT executor (filter, project, limit)
│   │   ├── aggregate.rs    ← GROUP BY + COUNT/SUM/AVG/MIN/MAX (parallel Rayon)
│   │   ├── sort.rs         ← ORDER BY (multi-key, null-aware, parallel Rayon)
│   │   ├── join.rs         ← hash join (INNER / LEFT / RIGHT)
│   │   ├── window.rs       ← window functions (ROW_NUMBER/RANK/LAG/LEAD/SUM/AVG…)
│   │   ├── tuning.rs       ← parallel thresholds derived from the machine, not hard-coded
│   │   └── sql.rs          ← SQL string → query objects (plan cache aware)
│   ├── api/
│   │   └── c_ffi.rs        ← C-compatible extern "C" API
│   └── bin/
│       └── server.rs       ← Actix-web REST server
├── benches/
│   └── query_bench.rs      ← Criterion benchmarks, four engines on one Parquet file
├── tests/
│   ├── diag_bandwidth.rs   ← the machine's read-bandwidth floor (L2 / L3 / DRAM)
│   ├── diag_join_breakdown.rs  ← where a join's time actually goes
│   ├── diag_scan_shape.rs      ← batch shape a Parquet load produces
│   ├── diag_threshold_reach.rs ← which per-batch thresholds can ever fire
│   └── …                       ← correctness suites
├── scripts/
│   ├── bench.ps1           ← pinned benchmark runner — use this, not `cargo bench`
│   ├── bench_notify.ps1    ← posts results to Telegram
│   └── bench_report.py     ← formats a run into a table
├── bench-results/          ← captured output from past runs
└── .github/workflows/
    └── ci.yml              ← test / clippy / fmt / coverage
```

---

## Building

```bash
# Debug
cargo build

# Release (LTO + codegen-units=1)
cargo build --release

# Tests
cargo test

# Benchmarks
cargo bench

# REST server
cargo run --bin falcondb-server
```

---

## Roadmap

### Where the remaining time goes

FalconDB is ahead of DuckDB and DataFusion on every measured workload, so the
open items are structural rather than a slow loop somewhere:

| Limit | Why it matters | Next step |
|---|---|---|
| Scans run above the measured floor | 1 M-row `SUM` takes 61 µs against a 55 µs L3 bandwidth floor | Per-batch overhead. Within 15% of the floor, and the remainder is now smaller than this machine's run-to-run drift |
| A join's hash table is built on one thread | ~1.0 ms of a 4.4 ms 1 M-row join, while the probe uses every core | Not a parallelism problem. Partitioning by bucket and a single-pass radix partition were both tried and both measured slower; the build spends its time allocating and filling ~2.2 MB per join. The next thing to try is allocating less — one bucket per row instead of two, and dropping the `keys` array, which duplicates the key column |
| Thresholds are calibrated, not derived | Parallel cut-offs come from measurements on *this* laptop | The coarse ones now scale with `rayon::current_num_threads()`; the model is untested on a many-core machine |
| Batch-count thresholds remain | `PAR_SUM_THRESHOLD` and the group-by still reason in batches, so 65 536-row batches break them | Convert every threshold to rows, then raise the Parquet batch size |
| No NUMA awareness | A thread may work on memory owned by another socket | Nothing measured yet; needs a server to measure on |
| `INNER JOIN` trails Polars | 1.3× at 1 M rows | Was 2.7×, closed by probing in place and by the hash table borrowing the key column instead of copying it. What remains is not yet identified; merging, index representation and probe have all been addressed |
| Everything lives in RAM | A table larger than memory cannot be loaded | Spill-to-disk and lazy batch reads |
| No cost-based optimizer | Join order is whatever the query text says | Cardinality estimates from zone maps |
| `WHERE` on a right-hand join column | Not supported: the filter is applied during the scan, before the join | Pinned by a test in `tests/join_projection.rs` |

### Features not yet implemented

| Feature              | Status | Notes                                              |
|----------------------|--------|----------------------------------------------------|
| Window functions     | ✅     | `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `LAG/LEAD`, `SUM/AVG/MIN/MAX OVER (PARTITION BY … ORDER BY …)` |
| INSERT / UPDATE / DELETE | ✅ | Copy-on-write over `Arc`-shared columns; returns `rows_affected` |
| CTE (`WITH`)         | ✅     | Materialised in declaration order, so a CTE can reference an earlier one |
| `WITH RECURSIVE`     | ✅     | Anchor `UNION [ALL]` recursive term, iterated to a fixed point. The recursive term is matched to the anchor by position, so it need not alias its columns. Capped at 1,000 iterations — a recursion with no fixed point is rejected with an error rather than run unbounded. Only a single anchor/recursive pair per CTE; more terms, or `INTERSECT`/`EXCEPT` between them, are rejected rather than silently dropped |
| Scalar subqueries    | ✅     | `WHERE x > (SELECT AVG(x) FROM t)` — folded to a literal once, before evaluation. Empty result → NULL; >1 row → error |
| WAL integration for DML | ✅  | Statement-based logging + `recover()` replay          |
| `IN (SELECT …)` / `EXISTS` | ✅ | Uncorrelated: folded to an IN-list / boolean literal before evaluation |
| Correlated subqueries | ✅    | `WHERE salary > (SELECT AVG(salary) FROM emp WHERE dept = e.dept)`. Evaluated per outer row — correct but O(rows × inner cost); a join rewrite is future work |
| Subquery → join rewrite | ❌  | Would turn the per-row path into a single grouped join    |
| DDL (`CREATE`/`DROP`/`ALTER`) | ✅ | `CREATE TABLE`, `DROP TABLE [IF EXISTS]`, `ALTER TABLE ADD/DROP COLUMN` |
| Zone maps (batch statistics) | ✅ | Per-batch min/max/null bounds computed once at load; a selective `WHERE` skips whole batches without reading a value. Pruning is conservative — it never skips a batch that could hold a match |
| EXPLAIN              | ✅   | `EXPLAIN <stmt>` returns a plan table: Scan / Filter / GroupBy / Sort / Limit |
| Cost-based optimizer | ❌   | Join reordering, predicate pushdown across joins       |
| Integer arithmetic   | ⚠️     | `+ - * /` evaluate in `f64` and yield a `Float64` column, even for integer operands. Values print the same up to 2^53, but a column's type can widen across a `UNION`, and very large integers lose precision |
| Vectorised SIMD      | ❌     | Explicit AVX2/AVX-512 for arithmetic and comparisons |
| Python bindings      | ❌     | PyO3 wrapper — `import falcondb`                   |

---

## License

MIT
