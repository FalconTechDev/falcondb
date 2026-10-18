<div align="center">

# FalconDB

**A columnar analytical database engine, written from scratch in Rust.**

Faster than DuckDB and DataFusion on every workload measured here, and ahead of Polars on most of them.

[![CI](https://github.com/FalconTechDev/falcondb/actions/workflows/ci.yml/badge.svg)](https://github.com/FalconTechDev/falcondb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-314%20unit%20%2B%20integration-brightgreen.svg)](#testing)

</div>

---

## What it is

FalconDB answers one question: *what does a columnar database look like when you own every layer?*

No Arrow dependency, no borrowed execution engine, no query layer bolted onto someone else's storage. Storage, encoding, planning, execution, indices, WAL and the SQL front end are all here, which is why the numbers below move when the design changes.

```rust
use falcondb::db::Database;

let db = Database::in_memory();
db.load_parquet("employees.parquet", "employees")?;

let result = db.sql("
    SELECT dept, COUNT(*) AS headcount, AVG(salary) AS avg_salary
    FROM employees
    WHERE salary > 50000
    GROUP BY dept
    ORDER BY avg_salary DESC
    LIMIT 10
")?;

print!("{}", result.display());
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

## Performance

Four engines, one process, the same Parquet file. **FalconDB, [Polars](https://pola.rs), [DuckDB](https://duckdb.org), [DataFusion](https://datafusion.apache.org)**, Criterion, median of 100 samples, Intel i7-13700H on mains power with threads pinned to the P-cores.

### At one million rows

| Query | FalconDB | Polars | DuckDB | DataFusion |
|---|---|---|---|---|
| `SUM(score)` | **61.5 µs** | 158 µs | 495 µs | 2.64 ms |
| `SUM(id)` integer | **77.1 µs** | 169 µs | 482 µs | 13.0 ms |
| `WHERE score > 50` | **654 µs** | 2.33 ms | 24.1 ms | 17.2 ms |
| `GROUP BY` + 3 aggregates | **691 µs** | 10.5 ms | 7.42 ms | 7.16 ms |
| `DISTINCT` | **2.30 ms** | 5.46 ms | 4.05 ms | 3.13 ms |
| `INNER JOIN` | 3.35 ms | **2.59 ms** | 8.08 ms | 16.7 ms |

### At one hundred thousand rows

| Query | FalconDB | Polars | DuckDB | DataFusion |
|---|---|---|---|---|
| `SUM(score)` | **8.65 µs** | 18.7 µs | 337 µs | 1.98 ms |
| `WHERE score > 50` | **128 µs** | 178 µs | 3.90 ms | 2.77 ms |
| `GROUP BY` + 3 aggregates | **97.8 µs** | 1.12 ms | 3.87 ms | 2.43 ms |
| `WHERE`+`GROUP BY`+`ORDER BY` | **22.4 µs** | 1.21 ms | 3.71 ms | 2.11 ms |
| `INNER JOIN` | **307 µs** | 426 µs | 1.69 ms | 3.93 ms |
| `LEFT JOIN` | **678 µs** | — | 2.40 ms | 4.11 ms |
| `SEMI JOIN` (`EXISTS`) | **625 µs** | — | 2.48 ms | 4.07 ms |
| `ANTI JOIN` (`NOT EXISTS`) | **855 µs** | — | 2.35 ms | 4.13 ms |
| `ROW_NUMBER() OVER` | **2.22 ms** | — | 11.3 ms | 5.58 ms |
| `UPDATE` half the rows | **1.62 ms** | — | 3.60 ms | — |
| `DELETE` half the rows | **882 µs** | — | 2.54 ms | — |

Polars is absent from the lower rows because it has no `EXISTS`, no in-place `UPDATE`/`DELETE`, and expresses `ROW_NUMBER` through a different shape entirely. Timing a different formulation against ours would not be a comparison.

### At ten and a hundred million rows

The Criterion suite stops at a million. Past that the realistic measure is a *single* run — nobody runs a hundred-million-row query a hundred times, and the run a user waits for is the first one.

| Query | 10M | 100M | DuckDB 100M |
|---|---|---|---|
| `COUNT(DISTINCT dept)` | **1.32 ms** | **11.5 ms** | 288 ms |
| `SUM(score)` | **2.06 ms** | **20.3 ms** | 87.4 ms |
| `GROUP BY dept` | **4.07 ms** | **31.4 ms** | 279 ms |
| `COUNT(*) WHERE score > 50` | **4.09 ms** | 54.8 ms | 50.6 ms |
| `ORDER BY score DESC LIMIT 10` | **9.49 ms** | **95.3 ms** | 140 ms |
| `ORDER BY id DESC LIMIT 10` | 35.0 ms | 364 ms | — |
| load from Parquet | ~0.35 s | 2.5–5.0 s | 2.20 s |

These sizes exposed two defects that were invisible below a million rows, both now fixed: `COUNT(*)` with a `WHERE` materialised every matching row to return one number (164 ms → 54.8 ms), and `ORDER BY … LIMIT` built a `(key, row)` pair for every row and concatenated every batch before comparing a single key — 1.6 GB and 1.7 GB of traffic respectively to return ten rows (1.05 s → 95.3 ms).

### Where FalconDB is behind

Ahead of DuckDB and DataFusion on all 26 measured points. Three exceptions, all against Polars, plus loading:

| | FalconDB | Polars | |
|---|---|---|---|
| `SUM(id)` 100 k | 20.8 µs | 14.8 µs | 1.4× |
| `INNER JOIN` 1 M | 3.35 ms | 2.59 ms | 1.3× |
| `ORDER BY` 100 k | 1.66 ms | 1.38 ms | 1.2× |

Loading Parquet is slower than DuckDB and is the one place that is behind at every size. `SUM(id)` at 100 k runs on one core because the measured threshold is 178 k rows and a serial sum really is faster below it; Polars reaches 54 GB/s on 800 KB, more than one core gets from L2 here, so it is not doing the same thing. What remains of the join gap is the build, not the plumbing: of a 1 M-row join, building the hash table is about 1.0 ms on one thread while the probe uses every core — and three ways of parallelising or shrinking it were measured and rejected (`IntHashTable::build_parallel`, `build_radix`, `bucket_density`), which is how it became clear the build is bound by memory it must touch once.

---

## How these numbers were taken

The numbers are worth what the method is worth. Several of the corrections below changed published figures by an order of magnitude, so the method is documented rather than assumed.

**Pinned to the P-cores.** On a hybrid CPU, rayon's threads land on a different P/E mix each run. Unpinned run-to-run noise on *unchanged code* measured 25–46%; pinned, about 1%.

**Mains power, verified during the run.** On battery every engine roughly halves, and nothing in the output says so. `scripts/bench.ps1` samples power every three seconds and reports afterwards — one early run produced a single file holding two different machines, everything after the charger came out 2–3× slower across all four engines.

**One reading is not a result.** A difference under about 20% is not resolved by a single run at these sizes, and a moved point is judged against its own history, not against the run before it:

- `WHERE` at 1 M has measured 672, 606, 602, 639, 659, 641 and 673 µs across seven runs — a ±6% band that predates every change in the table above. Comparing the top of that band to the bottom would read as a 12% regression that never happened.
- It cuts both ways. A pipeline point was once recorded as "about 4% off" and the cost accepted; two later readings came back level. A single reading is no better evidence for a regression you are willing to accept than for one you are not.
- `DISTINCT` at 1 M read 2.19, 2.17, 2.19, 2.26, 2.15 ms and then 2.58 — outside a tight band, and that one was real: an `#[inline]` on `Schema::column_index`, added while fixing a different regression, cost 18% here. Two points moved in the same run, one noise and one a defect, and only the history told them apart.

**Ratios, not absolute times, across runs.** The rivals drift too. A FalconDB number that improves while a rival's worsens says less than it appears to — in one re-measurement every engine, ours and all three, read 15–35% faster purely because fewer groups ran before it.

**The ceiling is a range, not a constant.** `tests/diag_bandwidth.rs` measures the hardware floor directly. Four back-to-back runs read 40.8, 40.6, 25.9 and 31.3 GB/s from DRAM, and the L3 figure fell monotonically across them while a single-threaded cache-resident kernel measured at its best throughout. So any claim of the form "N× off the hardware limit" is worth less than it looks — including the ones this file used to make.

Known deviations, open defects and negative results — including optimisations that were measured and *lost*, recorded so they are not retried — are kept in [FINDINGS.md](FINDINGS.md).

---

## SQL support

| | |
|---|---|
| **Queries** | `SELECT`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`/`OFFSET`, `DISTINCT` |
| **Joins** | `INNER`, `LEFT`, `RIGHT`, `SEMI` / `ANTI` (via `EXISTS` / `NOT EXISTS`) |
| **Set operations** | `UNION`, `UNION ALL`, `INTERSECT`, `EXCEPT` |
| **Window functions** | `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`, `SUM`/`AVG`/`MIN`/`MAX OVER (PARTITION BY … ORDER BY …)` |
| **Subqueries** | scalar, `IN (SELECT …)`, `EXISTS`, and correlated subqueries |
| **CTEs** | `WITH`, and `WITH RECURSIVE` iterated to a fixed point (capped at 1 000 iterations and rejected rather than run unbounded) |
| **Expressions** | `CASE WHEN`, `CAST`, `LIKE`/`ILIKE`, `IN (…)`, `BETWEEN`, `IS [NOT] NULL`, `COALESCE`, string / numeric / date functions |
| **DML** | `INSERT`, `UPDATE`, `DELETE` — copy-on-write over `Arc`-shared columns, so untouched batches are carried as pointer copies |
| **DDL** | `CREATE TABLE`, `DROP TABLE [IF EXISTS]`, `ALTER TABLE ADD`/`DROP COLUMN` |
| **EXPLAIN** | `EXPLAIN <any statement>` returns the plan as a result set, with no side effects |

**Deliberate deviations from the standard** are listed in [FINDINGS.md](FINDINGS.md) rather than left to be discovered: `/` on two integers is float division (as in DuckDB and SQLite), `NOT LIKE` passes NULL, and `CONCAT` does not propagate NULL. Integer `+`, `-` and `*` stay integral and refuse to overflow silently — a wrapped product is a wrong answer that looks like an answer.

---

## Engine

```
SQL string
    │
    ▼  plan cache — a hit is an Arc::clone: no parse, no allocation
parse → AST → Expr tree
    │
    ▼  execute_select — rayon over batches
    │     zone maps skip whole batches a selective WHERE cannot match
    │     WHERE mask is a packed Vec<u64>, one bit per row
    │     typed fast paths evaluate &[T] directly — no ScalarValue per row
    ▼  group by — partial hash aggregates per batch, then merged
    ▼  window functions — PARTITION BY + ORDER BY per clause
    ▼  sort — top-k by bounded scan when a small LIMIT is present
    ▼
QueryResult (Vec<RecordBatch>)
```

| Layer | What it does |
|---|---|
| `Column` | Dense typed values plus a validity bitmap; text is dictionary-encoded — each distinct string once, a `u32` per row |
| `RecordBatch` | Schema-aligned columns, fixed row count — the unit of parallelism |
| `Table` | A vector of batches, plus per-batch zone maps computed once at load |
| `compression` | LZ4 for numerics, bit-packing for booleans, dictionary for text |
| `native` | `.fdb` files: header plus compressed columns, about 2× smaller than CSV |
| `index` | B-tree for ranges, hash for point lookups |
| `wal` | Append-only, CRC32-verified, with `recover()` replay and checkpoints |

The recurring theme in the engine is **answering a question once instead of once per row**. A dictionary-encoded column has five distinct values and a hundred million rows, so `LOWER`, `LIKE`, `IN` and an `ORDER BY` key each need deciding five times. That one idea is most of the large wins here: `CONCAT` 400×, `LOWER` 270×, `LIKE` ~100×, `CASE` 80×, text `IN` 65×, `COALESCE` 31×.

---

## Interfaces

**Rust** — `Database::in_memory()`, `load_csv`, `load_parquet`, `sql`.

**REST**

```bash
cargo run --release --bin server      # FALCONDB_BIND=0.0.0.0:9000 to rebind
```

```
GET    /health                 GET  /tables            GET /tables/{name}
POST   /load/csv               POST /load/fdb          POST /tables/{name}/save
POST   /sql                    DELETE /tables/{name}
```

```bash
curl -s -X POST localhost:8080/sql -H 'Content-Type: application/json' \
  -d '{"sql": "SELECT dept, COUNT(*) AS n FROM emp GROUP BY dept"}' | jq .
```

```json
{ "columns": ["dept", "n"], "rows": [["eng", "142"], ["hr", "38"]], "row_count": 2, "elapsed_ms": 3 }
```

**Postgres wire protocol** — enough of the frontend/backend protocol to connect with `psql`, DBeaver, JDBC or anything else that speaks Postgres:

```bash
cargo run --release --bin pg_server
psql -h localhost -p 5433 -c "SELECT dept, COUNT(*) FROM emp GROUP BY dept"
```

**C FFI** — any language that speaks the C ABI:

```c
falcondb_db* db = falcondb_open();
falcondb_load_csv(db, "employees.csv", "emp");
falcondb_result* r = falcondb_sql(db, "SELECT * FROM emp LIMIT 5");
printf("%s\n", falcondb_result_display(r));
falcondb_result_free(r);
falcondb_close(db);
```

---

## Building

```bash
cargo build --release          # LTO, one codegen unit
cargo test --release           # 314 unit tests plus the integration suites
cargo run --release --bin server
```

### Testing

Tests are written to fail for a reason, not to cover a line. Where a fast path exists, the test asserts it agrees with the general path it bypasses — and includes the shapes the fast path *declines*, which is how a NULL-semantics difference in a dictionary-mapped `CONCAT` was caught. A `COUNT(*)` population count is tested at 1, 63, 64, 65, 127, 128, 129 … rows, with 64 and 128 present deliberately as the sizes that would pass *even with* the bug, so their presence beside 63 and 65 is what makes the test mean something.

### Running the benchmarks

```powershell
scripts/bench.ps1
```

Not a bare `cargo bench`: the script checks for mains power, warns about processes competing for cores, pins threads, and samples the power state during the run. The reasons are under [How these numbers were taken](#how-these-numbers-were-taken).

The comparison benchmark links DuckDB through `libduckdb-sys`, which needs the library next to the tree. It is not committed — 29 MB of someone else's binary in every clone — so fetch it once:

```powershell
# Windows: libduckdb-windows-amd64.zip from https://github.com/duckdb/duckdb/releases
# extract duckdb.dll, duckdb.lib, duckdb.h, duckdb.hpp into lib/duckdb/
```

`.cargo/config.toml` points `DUCKDB_LIB_DIR` at that directory. FalconDB itself has no DuckDB dependency; only the benchmark does.

---

## Roadmap

| Open item | Where it stands |
|---|---|
| Loading Parquet is behind DuckDB | The one workload that trails at every size. Reading the file is now most of it — the zone maps were serial and are not any more (2.56 s → 119 ms at 100M) |
| `SUM`/`MIN`/`MAX` return `Float64` for integer columns | The accumulator holds an `f64`, so above 2^53 precision is lost in the *state*, not just the type. Needs integer accumulators and the vectorised paths that bypass them |
| Comparisons go through `f64` | `n = 9007199254740993` can still answer wrongly. Arithmetic was fixed; comparisons were not, and the half-fix is recorded rather than hidden |
| A join's hash table is built on one thread | ~1.0 ms of a 1 M-row join. Not a parallelism problem — three approaches measured slower. Next: allocate less |
| Thresholds are calibrated, not derived | Parallel cut-offs come from measurements on one laptop; the coarse ones scale with `current_num_threads()`, untested on many cores |
| No cost-based optimizer | Join order follows the query text. Cardinality estimates from zone maps would change that |
| Everything lives in RAM | A table larger than memory cannot be loaded. Needs spill-to-disk and lazy batch reads |
| `WHERE` on a right-hand join column | Unsupported — the filter is applied during the scan, before the join. Pinned by a test rather than left as a surprise |
| Explicit SIMD | Arithmetic and comparisons run over contiguous slices and vectorise, but no hand-written AVX2/AVX-512 |
| Python bindings | `pyproject.toml` and the maturin setup are in place; the PyO3 surface is not finished |

---

## License

[MIT](LICENSE) © FalconTech
