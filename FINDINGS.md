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

### A NULL join key matches another NULL key

```sql
SELECT COUNT(*) FROM n JOIN n AS m ON n.a = m.a
```

Over a column holding `1, NULL, 3, NULL` this answers **6**. The correct answer is
2: `1` matches `1`, `3` matches `3`, and a NULL key matches nothing -- not even
another NULL. The extra 4 is the two NULLs matching each other both ways.

This returns rows that should not exist, which is the worst kind of wrong: no
error, no warning, just a bigger result. Every `JOIN` on a nullable column is
affected.

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

### `__having_` prefix can collide

`translate_having` appends aggregates under `__having_N` and `drop_having_scratch`
removes columns by that prefix. A user column actually named `__having_0` would be
deleted from their result. Unlikely enough to live with, cheap enough to fix
properly if anyone cares: the aliases could be recorded in the plan instead of
recognised by name.

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
