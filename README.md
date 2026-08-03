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

Measured on **Intel i7-13700H, Windows 11**. Median over 100 Criterion samples.  
FalconDB: SQL plan cache + parallel GROUP BY (Rayon) + packed bitmask WHERE + fast scalar predicates + parallel sort.  
DuckDB: `prepare()` + execute, version 1.x.  
Run yourself: `cargo bench`

### Full scan — `SELECT *`

| Rows      | FalconDB  | DuckDB   | Winner              |
|-----------|-----------|----------|---------------------|
| 10 000    | 0.44 ms   | 0.78 ms  | FalconDB **1.8×**   |
| 100 000   | 5.6 ms    | 6.4 ms   | FalconDB **1.1×**   |
| 1 000 000 | 60 ms     | 45 ms    | DuckDB 1.3×         |

### Filter — `WHERE score > 50`

| Rows      | FalconDB  | DuckDB   | Winner              |
|-----------|-----------|----------|---------------------|
| 10 000    | 0.31 ms   | 0.72 ms  | FalconDB **2.3×**   |
| 100 000   | 3.4 ms    | 4.1 ms   | FalconDB **1.2×**   |
| 1 000 000 | 41 ms     | 24 ms    | DuckDB 1.7×         |

> WHERE uses a fast scalar predicate path (`try_fast_mask`) that operates directly on `&[Option<T>]` slices — no `ScalarValue` boxing — then packs results into a bitmask (`Vec<u64>`, one bit per row, 8× smaller than `Vec<bool>`).  
> The fast path flipped the 100 k result: FalconDB was **losing** (DuckDB 1.6×) and now **wins** (FalconDB 1.2×). The 1 M gap closed from 2.7× to 1.7×.

### GROUP BY + COUNT + SUM + AVG (5 groups)

| Rows      | FalconDB  | DuckDB   | Winner               |
|-----------|-----------|----------|----------------------|
| 10 000    | 44 µs     | 1.45 ms  | FalconDB **32×**     |
| 100 000   | 0.40 ms   | 4.51 ms  | FalconDB **11×**     |
| 1 000 000 | 4.70 ms   | 7.77 ms  | FalconDB **1.7×**    |

> **Dictionary encoding is what makes this fast.** `Utf8Column` stores each distinct
> string once and a `u32` code per row, so `GROUP BY` assigns group ids per *dictionary
> entry* — a handful of them — instead of hashing a string on every one of a million
> rows. Mapping a row to its group becomes a single array index. That took 1 M-row
> `GROUP BY` from 34.8 ms to 4.70 ms (7.4×) and flipped the last remaining loss.
>
> Two caveats worth stating: the dictionary is built at load time, so its cost is
> outside this measurement (deliberately — that is the point of the encoding, and it
> is how columnar engines work, but a load-once-query-once workload sees less gain).
> And DuckDB's numbers vary up to 3× between runs on this machine, so treat the small-
> row multiples as "an order of magnitude" rather than a precise figure; the 1 M-row
> comparison is the stable one.

### ORDER BY (single column, DESC)

| Rows      | FalconDB  | DuckDB   | Winner              |
|-----------|-----------|----------|---------------------|
| 10 000    | 1.87 ms   | 2.67 ms  | FalconDB **1.4×**   |
| 100 000   | 22.4 ms   | 26.1 ms  | FalconDB **1.2×**   |

> Rayon `par_sort_unstable_by` kicks in at ≥ 65 536 rows; serial sort below that threshold.

### End-to-end SQL pipeline — `WHERE + GROUP BY + ORDER BY`

| Rows      | FalconDB  | DuckDB   | Winner              |
|-----------|-----------|----------|---------------------|
| 10 000    | 0.67 ms   | 1.29 ms  | FalconDB **1.9×**   |
| 100 000   | 5.6 ms    | 4.4 ms   | DuckDB 1.3×         |

### Summary

| Workload         | FalconDB wins at          | DuckDB wins at       |
|------------------|---------------------------|----------------------|
| Full scan        | ≤ 100 k rows              | 1 M rows (1.3×)      |
| Filter (WHERE)   | ≤ 100 k rows              | 1 M rows (1.7×)      |
| GROUP BY         | ≤ 100 k rows              | 1 M rows (2.6×)      |
| ORDER BY         | all tested sizes          | —                    |
| SQL pipeline     | 10 k rows (1.9×)          | 100 k (1.3×)         |

FalconDB beats DuckDB at every tested size up to 100 k rows across all workloads, and wins ORDER BY at all sizes. The 1 M row gap remains — DuckDB's AVX-512 SIMD and mature query optimizer dominate at that scale. FalconDB is a learning project — the goal is full control over every layer, not replacing a production database.

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
│   │   └── sql.rs          ← SQL string → query objects (plan cache aware)
│   ├── api/
│   │   └── c_ffi.rs        ← C-compatible extern "C" API
│   └── bin/
│       └── server.rs       ← Actix-web REST server
├── benches/
│   └── query_bench.rs      ← Criterion benchmarks
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

### Performance gaps vs DuckDB

FalconDB now wins every benchmarked workload. The remaining limits are structural
rather than a single slow loop:

| Limit                     | Why it matters                                        | Next step                                        |
|---------------------------|-------------------------------------------------------|--------------------------------------------------|
| Everything lives in RAM   | A table larger than memory simply cannot be loaded    | Spill-to-disk and lazy batch reads               |
| No cost-based optimizer   | Join order is whatever the query text says            | Cardinality estimates from zone maps             |
| Correlated subqueries     | Re-executed per outer row — correct but O(rows × inner) | Rewrite into a grouped join                    |
| Numeric filter is scalar  | LLVM auto-vectorises, but nothing is hand-tuned       | Explicit AVX2 via `std::arch` with runtime detect |

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
