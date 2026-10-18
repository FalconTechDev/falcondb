# Open findings

Things found while working on something else and deliberately not changed, plus
questions a measurement could settle. Written down because the reason for leaving
each one alone is as perishable as the finding itself.

Each entry says what was observed, how it was observed, and why it is still here.
An entry with no evidence behind it does not belong in this file.

---

## Correctness: NULL semantics that differ from SQL

These were all found while making the surrounding code faster, and all left alone
on the same reasoning: a performance change is not the place to alter an answer.
Each is pinned by a test, so changing it is a decision rather than an accident.

### `NOT LIKE` lets NULL rows through

```sql
SELECT s FROM t WHERE s NOT LIKE 'x'   -- a NULL s is returned
```

A NULL never matches, and the general path negates that non-match to `true`. SQL
says a comparison with NULL is unknown and the row should not appear.

Pinned by `typed_like_matches_the_general_path`. The same shape very likely
affects `NOT IN`; `typed_in_matches_the_general_path` records the current answer
there without judging it.

### `CONCAT` does not propagate NULL

```sql
SELECT CONCAT(s, '!') FROM t   -- a NULL s gives "!", not NULL
```

`eval_function`'s CONCAT skips NULL arguments and always returns a string. That
matches Postgres's `concat()` and contradicts the `||` operator in the same
standard, so it is a defensible choice -- but it is inconsistent with `UPPER`,
`LOWER` and the `TRIM`s in the same match, which all return NULL for a NULL input.

Worth deciding deliberately rather than per function. Pinned by
`dict_mapped_functions_match_the_general_path`.

### `ROUND(x, NULL)` errors instead of answering NULL

```sql
SELECT ROUND(f, n) FROM t   -- "function argument 1 is NULL" when n has a NULL
```

A NULL argument to a scalar function should make the result NULL, not fail the
query. Pinned by `dense_numeric_functions_match_the_general_path`.

---

## Correctness: confirmed wrong, found by `semantics_probe`

Three bugs, all demonstrable in one query each. Severity order, worst first.

### ~~A NULL join key matches another NULL key~~ — FIXED

```sql
SELECT COUNT(*) FROM n JOIN n AS m ON n.a = m.a
```

Over a column holding `1, NULL, 3, NULL` this answers **6**. The correct answer is
2: `1` matches `1`, `3` matches `3`, and a NULL key matches nothing -- not even
another NULL. The extra 4 is the two NULLs matching each other both ways.

This returned rows that should not exist, which is the worst kind of wrong: no
error, no warning, just a bigger result. Every `JOIN` on a nullable column was
affected.

Fixed: `row_key_or_null` returns `None` for a NULL key, such keys are never
inserted into the hash table, and a probe with one finds nothing. The row itself
is still processed, so LEFT and ANTI continue to emit it -- for ANTI it belongs in
the output precisely because it matched nothing. Pinned by
`a_null_join_key_matches_nothing`, which checks all four join types.

The typed integer path was never affected: `int_keys_usable` refuses a column with
any NULL in it. Only the general string path was wrong, and only for nullable
keys.

### Integer arithmetic widens to `f64`, so it loses precision and changes type

```sql
SELECT n, n + 1, n * 1, n - 1 FROM big   -- n = 9007199254740993, which is 2^53+1
```

| expression | answer | should be |
|---|---|---|
| `n` | `Int64(9007199254740993)` | correct |
| `n + 1` | `Float64(9007199254740992.0)` | `Int64(9007199254740994)` |
| `n * 1` | `Float64(9007199254740992.0)` | `Int64(9007199254740993)` |
| `n - 1` | `Float64(9007199254740991.0)` | `Int64(9007199254740992)` |

Two faults at once. The result type is `Float64` for integer inputs, and above
2^53 the value is wrong as well -- `n * 1` does not equal `n`, which is as plain
as a counterexample gets. `n + 1` is off by two.

`SUM` was fixed for exactly this reason (`checked_sum_i64` accumulates in `i64`),
so the engine already holds the position that integers should stay exact. Ordinary
arithmetic never got the same treatment.

### `SUM`, `MIN` and `MAX` over an `Int64` column answer in `Float64`

```sql
SELECT SUM(a) FROM n     -- Float64(4.0) for an Int64 column
SELECT MAX(n) FROM big   -- Float64(9.223372036854776e18) for i64::MAX
```

`MAX` of `i64::MAX` comes back as a float one larger than the value it was given.
And the type is inconsistent with the engine's own behaviour on an empty table,
where `MIN(a)` answers `Int64(None)` -- so the same expression is typed `Int64`
when there are no rows and `Float64` when there are.

The kernel is not the problem: `checked_sum_i64` already accumulates in `i64`. The
result is boxed into `ScalarValue::Float64` on the way out.

---

## Correctness: checked and correct

Recorded so the same ground is not covered twice.

- `GROUP BY` puts all NULLs in one group, and `DISTINCT` keeps one NULL. Both
  correct -- NULL forms its own group even though it is not equal to itself.
- Empty table: `COUNT(*)` is 0, `SUM`/`AVG`/`MIN` are NULL, `GROUP BY` yields no
  rows. All correct.
- `LIMIT`/`OFFSET`, including an offset past the end, which yields nothing.
- `1/0` and `1.0/0.0` both error. Postgres errors too, so this is defensible.

---

## Correctness: not yet investigated

### The `%` operator is not supported

```sql
SELECT COUNT(*) FROM t WHERE n % 2 = 0   -- "unsupported operator: %"
```

Modulo is ordinary SQL and the engine rejects it. Found by writing it into a test
without thinking, which is roughly how a user would find it. Not investigated
beyond confirming the message; `MOD(a, b)` was not tried.



### `__having_` prefix can collide

`translate_having` appends aggregates under `__having_N` and `drop_having_scratch`
removes columns by that prefix. A user column actually named `__having_0` would be
deleted from their result. Unlikely enough to live with, cheap enough to fix
properly if anyone cares: the aliases could be recorded in the plan instead of
recognised by name.

---

## Speed: two shapes do not scale past 1M rows

The benchmark suite stops at 1M. One single run of each query at 10M and 100M
against DuckDB -- `tests/one_shot_large.rs`, one wall-clock reading each, no
median -- found two shapes where FalconDB is ahead at 1M and behind at 100M.

Measured on battery, so the absolute numbers are roughly twice what mains power
would give. Both engines are slowed equally, so the ratios hold; that is the part
being claimed.

| query | 10M Falcon | 10M Duck | 100M Falcon | 100M Duck |
|---|---|---|---|---|
| `COUNT(DISTINCT dept)` | 1.83 ms | 34.4 ms | 13.1 ms | 288 ms |
| `GROUP BY dept` | 5.92 ms | 45.7 ms | 30.2 ms | 279 ms |
| `SUM(score)` | 3.15 ms | 6.85 ms | 18.9 ms | 87.4 ms |
| `WHERE score > 50` | 12.6 ms | 11.8 ms | **164 ms** | 50.6 ms |
| `ORDER BY score LIMIT 10` | 108 ms | 24.2 ms | **1.20 s** | 140 ms |
| load from Parquet | 448 ms | 306 ms | 3.67 s | 2.20 s |

### `COUNT(*)` with a `WHERE` crosses over between 10M and 100M

Ahead at 100k (1.4x), level at 10M (0.9x), **3.3x behind at 100M**.

Scaling tells the story. Ten times the rows cost us 13x; they cost DuckDB 4.3x.
Ours is superlinear where theirs is sublinear -- DuckDB gets *more* efficient with
scale, which is what good parallelism looks like, and we get less.

Hypothesis, not yet verified: the query is `SELECT COUNT(*) ... WHERE score > 50`,
and the filter produces a bitmask which is then used to build a filtered batch
that gets counted. At 100M rows with half of them matching, that materialises
50M rows in order to return one number. A popcount over the mask would answer it
without building anything. Worth checking before anything else here.

### `ORDER BY ... LIMIT` degrades with scale

4.5x behind at 10M, **8.6x behind at 100M**. Ten times the rows cost us 11x and
DuckDB 5.8x.

This one is already understood. Top-N selection landed recently and made the
*sort* nearly free for a small limit -- 1.70 ms to 583 us at 100k. What remains is
the pair materialisation: a `(key, row)` pair is built for every row whatever the
limit says, so at 100M that is 1.2 GB of allocation to return ten rows. The commit
that added top-N said removing it was "a different change"; these numbers are the
argument for making it. The fix is to find the top k by scanning the column
directly, never building the pairs.

### Loading is slower than DuckDB at both sizes

448 ms against 306 ms at 10M, 3.67 s against 2.20 s at 100M -- about 1.5x, stable
with scale, so no scaling defect, just a gap. Never measured before because the
suite loads outside the timed region on purpose.

---

## Speed: measured, not acted on

Per-row costs, 100k rows, five distinct strings where text is involved. The
reference for the projection ones is reading the column unchanged, which measures
the machinery rather than the work, so ratios against it overstate what a fix
would win; the per-row figure is the honest number.

| shape | cost | note |
|---|---|---|
| `STRING_AGG(dept)` | 168 ns/row | the most expensive shape left |
| `YEAR(d)` | 143 ns/row | see below -- `EXTRACT` does the same work for 43 |
| `UNION` | 44 ns/row | against 3 ns/row for `UNION ALL`; the gap is the dedup |
| `EXTRACT(YEAR FROM d)` | 43 ns/row | |
| `CAST(n AS DOUBLE)` | 37 ns/row | `Expr::Cast` is its own variant, so it needs its own path |
| `COUNT(DISTINCT n)` | 5.36 ms total | 100k distinct values; 41x a plain COUNT at 5 values |

### `YEAR(d)` costs 3.3x `EXTRACT(YEAR FROM d)`

The same question, two paths: `EXTRACT` has its own evaluator and `YEAR` goes
through `eval_function` per row. Either `YEAR` should translate to `Extract`, or
both should share one path. This is the cheapest item here to fix and the most
clearly wrong.

---

## Open questions

- **`bench-results/` is tracked in git.** Asked twice, never answered. It is
  Criterion's own output directory; if the baselines are not meant to be shared it
  should be ignored.

- **Is the `ORDER BY` 100k gap against Polars real or structural?** It has read
  1.15x, 1.40x, 1.14x and 1.20x across runs. The 1.40x was an outlier; the rest
  cluster near 1.15x. Worth one investigation, not four.

---

## Traps in the code itself

### `execute_hash_join` accepts `JoinType::Semi` and does not do a semi join

SEMI and ANTI have their own entry point, `execute_hash_semi_anti_join`, which
takes `anti: bool`. Passing `JoinType::Semi` or `JoinType::Anti` to the general
`execute_hash_join` is accepted and runs the ordinary join, returning joined rows
rather than filtered left rows.

Found by walking into it: the first version of
`a_null_join_key_matches_nothing` tested SEMI and ANTI through the general
function, and **read as passing** because on that fixture the row counts happened
to coincide. It only showed up because the test checked all four join types, and
one of the four disagreed.

Either the general function should reject those variants, or the two entry points
should be one. As it stands the signature invites a silent wrong answer.

---

## Method notes worth keeping

Not findings, but they cost real time to learn and the next person will otherwise
pay again.

- **Judge a moved benchmark against its own history, not the run before it.**
  `filter_where/1M` has spanned 602-673 us across runs with no code change; a
  comparison picking the low reading against the high one reads as a 12%
  regression. `DISTINCT/1M` sat in 2.15-2.26 ms and then read 2.58 -- outside its
  band, and that one was real.

- **A guard written for one function will swallow another.** Twice: an argument
  match that admitted a second literal only when it was text (written for
  `CONCAT`) silently excluded `ROUND(f, 2)`, and a digits guard for `ROUND`
  excluded `COALESCE`. Both times the code read as working and only a per-function
  measurement showed one shape had not moved.

- **An optimisation's cost includes what it displaces.** Adding qualified-name
  handling to `Schema::column_index` made the function too large to inline, which
  cost 15% on a benchmark that never executes the new code. Forcing `#[inline]` to
  fix that cost 18% on `DISTINCT`. Neither was visible in reasoning.

## Both scaling defects at 10M/100M are fixed (2026-10-18)

Measured one run each, FalconDB only, mains power, 2400MHz.

| query | before | after | DuckDB |
|---|---|---|---|
| `COUNT(*) WHERE score > 50` 10M | 12.60ms | 4.95ms | 11.8ms |
| `COUNT(*) WHERE score > 50` 100M | 164ms | 43.5ms | 50.6ms |
| `ORDER BY score DESC LIMIT 10` 10M | 96.6ms | 12.5ms | 24.2ms |
| `ORDER BY score DESC LIMIT 10` 100M | 1.05s | 146ms | 140ms |

`COUNT(*)` now counts the filter mask (`db.rs`); growth is 8.8x for 10x rows
instead of 13x. `ORDER BY ... LIMIT` lost two separate costs: the `(key, row)`
pair per row, and the concatenation of every batch before the sort -- the second
was the larger of the two (400ms -> 146ms at 100M).

Left deliberately: the integer sort arms do not normalise their key to an
unsigned ordering key, so `ORDER BY id LIMIT 10` still builds every pair. Same
fix, a separate change.

### Integer `ORDER BY ... LIMIT`, and a trap worth keeping (2026-10-18)

`ORDER BY id DESC LIMIT 10` on an ascending column, one run each: 1.74s ->
364ms at 100M, 167ms -> 35.0ms at 10M. Still 3.8x the same query on a repeating
column (95.3ms at 100M), and that remainder is real: on ascending data every row
belongs in the kept set when it arrives, so the appends cannot be avoided.

**The trap.** A bounded max-heap is the textbook top-k and was 14x slower than
the buffer on exactly this shape, because sorted input makes every row beat the
kept set and the heap sifts on all of them. It looked fine when measured on
`score`, whose hundred repeating values make the threshold reject almost
everything. One column shape hid it completely. Ordered columns are ordinary in
analytical data.

Second part of the same trap: the buffer was `2k` at first, so `LIMIT 10` reduced
every ten rows -- 925ms instead of 364ms. The floor is 1024 now.

### Parquet load: what is measurable at 100M and what is not (2026-10-18)

Phase split of a 100M load: `read_parquet` about 1.9s, zone maps 2.56s serial
against 119ms in parallel (21x, measured as the two paths side by side in one
process). The zone maps are now parallel.

Per column within one process: text 851ms vs float 268ms before the borrowed-str
dictionary, 550ms vs 342ms after -- 3.2x the numeric column down to 1.6x.

**Whole-load time at 100M has a band of 2.49s to 4.97s** across isolated runs.
Nothing smaller than that is provable from it, so load numbers quoted to two
decimal places in earlier notes -- including "3.35-3.44s before, 2.81s after" --
should be read as within that band. Phase and per-column comparisons taken side
by side in one process remain usable; subtractions between whole-load totals do
not.

Two contaminations worth naming, both of which produced plausible wrong answers:
running the 10M and 100M harnesses in one process (100M load 2.81s -> 3.69s), and
running three multi-GB profiling tests in one process (`id`, untouched, 484ms ->
1.69s, which read as "the change made everything worse").
