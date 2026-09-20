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
- **CTEs and scalar subqueries** — `WITH a AS (…), b AS (SELECT … FROM a)` and `WHERE x > (SELECT AVG(x) FROM t)`.
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
| `SUM(score)` | 10 k | **2.4 µs** | 10.4 µs | 339 µs | 1.59 ms |
| | 100 k | **11.0 µs** | 22.7 µs | 567 µs | 1.91 ms |
| | 1 M | **78.3 µs** | 186 µs | 781 µs | 4.13 ms |
| `SUM(id)` (integer) | 10 k | **3.6 µs** | 9.0 µs | 367 µs | 1.81 ms |
| | 100 k | 28.0 µs | **18.0 µs** | 532 µs | 3.67 ms |
| | 1 M | **100 µs** | 202 µs | 606 µs | 15.0 ms |
| `WHERE score > 50` | 10 k | **19.8 µs** | 67.0 µs | 802 µs | 2.23 ms |
| | 100 k | **191 µs** | 207 µs | 5.00 ms | 4.94 ms |
| | 1 M | **697 µs** | 3.48 ms | 38.6 ms | 26.2 ms |

### Grouping, sorting, pipelines

| | rows | FalconDB | Polars | DuckDB | DataFusion |
|---|---|---|---|---|---|
| `GROUP BY` + 3 aggregates | 10 k | **53.0 µs** | 322 µs | 2.37 ms | 2.17 ms |
| | 100 k | **124 µs** | 1.24 ms | 6.25 ms | 3.74 ms |
| | 1 M | **712 µs** | 9.79 ms | 7.13 ms | 8.54 ms |
| `ORDER BY score DESC` | 10 k | **186 µs** | 349 µs | 2.59 ms | 1.34 ms |
| | 100 k | 2.29 ms | **1.62 ms** | 20.2 ms | 8.73 ms |
| `DISTINCT` | 100 k | **642 µs** | 768 µs | 3.26 ms | 1.65 ms |
| | 1 M | **2.93 ms** | 6.21 ms | 4.73 ms | 4.22 ms |
| `WHERE`+`GROUP BY`+`ORDER BY` | 10 k | **32.0 µs** | 359 µs | 2.86 ms | 2.41 ms |
| | 100 k | **121 µs** | 1.28 ms | 6.93 ms | 2.88 ms |

### Joins, window functions, DML

Polars is absent from the rows below: it has no `EXISTS`, no in-place
`UPDATE`/`DELETE`, and expresses `ROW_NUMBER` through a different shape entirely.
Timing a different formulation against ours is precisely the mistake this file
used to make — see [Methodology](#methodology).

| | rows | FalconDB | Polars | DuckDB | DataFusion |
|---|---|---|---|---|---|
| `INNER JOIN` | 100 k | 545 µs | **527 µs** | 2.16 ms | 8.46 ms |
| | 1 M | 5.44 ms | **2.92 ms** | 15.2 ms | 31.6 ms |
| `LEFT JOIN` | 100 k | **737 µs** | — | 3.47 ms | 7.00 ms |
| `SEMI JOIN` (`EXISTS`) | 100 k | **802 µs** | — | 3.76 ms | 8.80 ms |
| `ANTI JOIN` (`NOT EXISTS`) | 100 k | **1.30 ms** | — | 3.75 ms | 9.17 ms |
| `ROW_NUMBER() OVER` | 100 k | **4.70 ms** | — | 23.6 ms | 8.05 ms |
| `UPDATE` (half the rows) | 100 k | **2.25 ms** | — | 5.07 ms | — |
| `DELETE` (half the rows) | 100 k | **1.11 ms** | — | 3.52 ms | — |


### Where FalconDB is behind

Ahead of every engine in 22 of 26 measured points, and ahead of DuckDB and
DataFusion in all of them. The exceptions are all against Polars:

| | FalconDB | Polars | |
|---|---|---|---|
| `INNER JOIN` 1 M | 5.44 ms | 2.92 ms | 1.9× |
| `SUM(id)` 100 k | 28.0 µs | 18.0 µs | 1.6× |
| `ORDER BY` 100 k | 2.29 ms | 1.62 ms | 1.4× |
| `INNER JOIN` 100 k | 545 µs | 527 µs | 1.03× — parity |

`INNER JOIN` at 1 M was 2.4× behind until the probe stopped concatenating the
left side. What remains of that gap is the probe itself, not the plumbing
around it.

`SUM(id)` and `ORDER BY` at 100 k both *lost* ground here, and not because the
code changed — neither was touched. Polars gained more than FalconDB did from
the allocator change described under [Methodology](#methodology). At these sizes
the work is tens of microseconds and allocator behaviour is a real share of it.
Cross-check them against the 10 k and 1 M rows of the same query before reading
much into either: at 1 M, `SUM(id)` is 2.0× ahead.

### How much room is actually left

"Fast" against a rival says nothing about how much the hardware could do, so
`tests/diag_bandwidth.rs` measures the floor directly: summing `f64` at working
sets that sit in L2, in L3 and past L3.

| working set | throughput |
|---|---|
| 0.1 MB (L2) | 30.5 GB/s — *one* thread does 50.4; splitting costs more than it saves |
| 8 MB (L3) | **144.9 GB/s** |
| 256 MB (DRAM) | 57.2 GB/s |

A 1 M-row `f64` column is 8 MB, which fits this CPU's 24 MB L3 — so that scan
never touches DRAM, and the floor for it is **55 µs**. FalconDB takes 109 µs, so
roughly **half the time is not the data movement**. Polars, at 165 µs, is three
times off the same floor: everyone is far from the ceiling here, FalconDB least so.

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
| Scans run at half the measured floor | 1 M-row `SUM` takes 109 µs against a 55 µs L3 bandwidth floor | Per-batch overhead: ~55 ns × 977 batches. Larger batches do recover ~10%, but every other threshold assumes 1024-row batches — see below |
| Thresholds are calibrated, not derived | Parallel cut-offs come from measurements on *this* laptop | The coarse ones now scale with `rayon::current_num_threads()`; the model is untested on a many-core machine |
| Batch-count thresholds remain | `PAR_SUM_THRESHOLD` and the group-by still reason in batches, so 65 536-row batches break them | Convert every threshold to rows, then raise the Parquet batch size |
| No NUMA awareness | A thread may work on memory owned by another socket | Nothing measured yet; needs a server to measure on |
| `INNER JOIN` trails Polars | 2.7× at 1 M rows, the largest remaining gap | Cause not yet identified — merging, index representation and probe have all been addressed |
| Everything lives in RAM | A table larger than memory cannot be loaded | Spill-to-disk and lazy batch reads |
| No cost-based optimizer | Join order is whatever the query text says | Cardinality estimates from zone maps |
| `WHERE` on a right-hand join column | Not supported: the filter is applied during the scan, before the join | Pinned by a test in `tests/join_projection.rs` |

### Features not yet implemented

| Feature              | Status | Notes                                              |
|----------------------|--------|----------------------------------------------------|
| Window functions     | ✅     | `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `LAG/LEAD`, `SUM/AVG/MIN/MAX OVER (PARTITION BY … ORDER BY …)` |
| INSERT / UPDATE / DELETE | ✅ | Copy-on-write over `Arc`-shared columns; returns `rows_affected` |
| CTE (`WITH`)         | ✅     | Non-recursive; materialised in declaration order, so a CTE can reference an earlier one. `WITH RECURSIVE` is rejected |
| Scalar subqueries    | ✅     | `WHERE x > (SELECT AVG(x) FROM t)` — folded to a literal once, before evaluation. Empty result → NULL; >1 row → error |
| WAL integration for DML | ✅  | Statement-based logging + `recover()` replay          |
| `IN (SELECT …)` / `EXISTS` | ✅ | Uncorrelated: folded to an IN-list / boolean literal before evaluation |
| Correlated subqueries | ✅    | `WHERE salary > (SELECT AVG(salary) FROM emp WHERE dept = e.dept)`. Evaluated per outer row — correct but O(rows × inner cost); a join rewrite is future work |
| Subquery → join rewrite | ❌  | Would turn the per-row path into a single grouped join    |
| DDL (`CREATE`/`DROP`/`ALTER`) | ✅ | `CREATE TABLE`, `DROP TABLE [IF EXISTS]`, `ALTER TABLE ADD/DROP COLUMN` |
| Zone maps (batch statistics) | ✅ | Per-batch min/max/null bounds computed once at load; a selective `WHERE` skips whole batches without reading a value. Pruning is conservative — it never skips a batch that could hold a match |
| EXPLAIN              | ✅   | `EXPLAIN <stmt>` returns a plan table: Scan / Filter / GroupBy / Sort / Limit |
| Cost-based optimizer | ❌   | Join reordering, predicate pushdown across joins       |
| Vectorised SIMD      | ❌     | Explicit AVX2/AVX-512 for arithmetic and comparisons |
| Python bindings      | ❌     | PyO3 wrapper — `import falcondb`                   |

---

## License

MIT
