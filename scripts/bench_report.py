"""Turn Criterion output into a comparison table, remembering rival timings.

Only FalconDB changes between runs. Polars, DuckDB and DataFusion are fixed
dependencies, so re-measuring them on every iteration costs ~20 of the ~25
minutes a full benchmark takes and produces the same answer with fresh noise on
top.

So rival timings are accumulated in a history file. Every full run appends its
observations; reports use the mean of everything seen. A FalconDB-only run then
takes about five minutes and still produces a complete table.

Usage
-----
  # Full run: measure everyone, record the rivals, print the table.
  python scripts/bench_report.py bench_out.txt

  # FalconDB-only run: fresh FalconDB numbers, remembered rivals.
  python scripts/bench_report.py bench_out.txt --rivals-from-history

  # Also send the table to Telegram.
  python scripts/bench_report.py bench_out.txt --send

The corresponding FalconDB-only benchmark invocation is:
  <query_bench binary> --bench falcondb
which Criterion filters by benchmark id.
"""
import argparse
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "rival_timings.json")

BAR = "=" * 34

# ─── Parsing ─────────────────────────────────────────────────────────────────

# Criterion pads the benchmark id to a fixed column and puts "time:" on the same
# line when it fits, wrapping to the next line when it does not. Both forms show
# up in a single run, so the id is matched as a prefix rather than a whole line.
ID_RE = re.compile(r"^([a-z_]+)/([a-z0-9_]+)/(\d+)\b")
# A number, whitespace, then a short token ending in 's', then space or ']'.
TIME_RE = re.compile(r"([\d.]+)\s+(\S{1,6}?s)(?=[\s\]])")


def unit_ns(u):
    """Nanoseconds per time unit.

    Criterion writes microseconds with U+00B5, which a mismatched console
    codepage can deliver as mojibake. Rather than enumerate every mangling,
    anything that is not ns/ms/s is taken to be microseconds — those three are
    pure ASCII and always survive intact.
    """
    u = u.strip()
    return {"ns": 1.0, "ms": 1e6, "s": 1e9}.get(u, 1e3)


def parse(path):
    """-> {group: {engine: {rows: median_ns}}}"""
    # utf-8-sig, not utf-8: PowerShell's `Out-File -Encoding utf8` writes a BOM,
    # which otherwise glues itself to the first benchmark id.
    text = io.open(path, encoding="utf-8-sig", errors="replace").read()
    out, pending = {}, None

    def record(key, line):
        nums = TIME_RE.findall(line)
        if len(nums) < 3:
            return False
        val, unit = nums[1]            # middle figure is the median estimate
        g, e, r = key
        out.setdefault(g, {}).setdefault(e, {})[r] = float(val) * unit_ns(unit)
        return True

    for line in text.splitlines():
        m = ID_RE.match(line.strip())
        if m:
            key = (m.group(1), m.group(2), int(m.group(3)))
            if "time:" in line and record(key, line):
                pending = None
            else:
                pending = key
            continue
        # Only the first "time:" after an id is the measurement; the "change:"
        # block that follows contains one too, so `pending` clears on use.
        if pending and "time:" in line:
            record(pending, line)
            pending = None
    return out


# ─── Rival history ───────────────────────────────────────────────────────────

RIVALS = ("polars", "duckdb", "datafusion")


def load_history():
    if not os.path.exists(HISTORY):
        return {}
    with io.open(HISTORY, encoding="utf-8") as f:
        return json.load(f)


def save_history(h):
    with io.open(HISTORY, "w", encoding="utf-8") as f:
        json.dump(h, f, indent=2, sort_keys=True)


def record_rivals(results, history):
    """Append this run's rival timings. Returns how many observations were added."""
    added = 0
    for group, engines in results.items():
        for engine, sizes in engines.items():
            if engine not in RIVALS:
                continue
            for rows, ns in sizes.items():
                key = "%s|%s|%d" % (group, engine, rows)
                history.setdefault(key, []).append(ns)
                added += 1
    return added


def rivals_from_history(history):
    """-> {group: {engine: {rows: mean_ns}}} plus a per-cell observation count."""
    out, counts = {}, {}
    for key, samples in history.items():
        group, engine, rows = key.split("|")
        rows = int(rows)
        out.setdefault(group, {}).setdefault(engine, {})[rows] = sum(samples) / len(samples)
        counts[key] = len(samples)
    return out, counts


def merge(fresh, remembered):
    """Fresh measurements win; remembered rivals fill the gaps."""
    out = {g: {e: dict(s) for e, s in engines.items()} for g, engines in remembered.items()}
    for g, engines in fresh.items():
        for e, sizes in engines.items():
            out.setdefault(g, {}).setdefault(e, {}).update(sizes)
    return out


# ─── Rendering ───────────────────────────────────────────────────────────────

GROUP_TITLES = [
    ("full_scan",        "FULL SCAN (SUM over one column)"),
    ("filter_where",     "FILTER  score > 50"),
    ("group_by_agg",     "GROUP BY + COUNT/SUM/AVG"),
    ("order_by",         "ORDER BY score DESC"),
    ("sql_pipeline",     "SQL PIPELINE (GROUP BY + ORDER BY)"),
    ("streaming_filter", "STREAMING FILTER (chunked vs direct)"),
]

LABEL = {
    "falcondb": "FalconDB",
    "polars": "Polars",
    "duckdb": "DuckDB",
    "datafusion": "DataFusion",
    "falcondb_chunked_64k": "chunked64k",
    "falcondb_direct": "direct",
}
ORDER = ["falcondb", "polars", "duckdb", "datafusion",
         "falcondb_chunked_64k", "falcondb_direct"]


def human(ns):
    if ns < 1e3:
        return "%.0fns" % ns
    if ns < 1e6:
        return "%.1fus" % (ns / 1e3)
    if ns < 1e9:
        return "%.2fms" % (ns / 1e6)
    return "%.2fs" % (ns / 1e9)


def human_rows(n):
    if n >= 1000000:
        return "%dM" % (n // 1000000)
    if n >= 1000:
        return "%dk" % (n // 1000)
    return str(n)


def render_group(title, data):
    sizes = sorted({r for e in data.values() for r in e})
    engines = [e for e in ORDER if e in data]

    out = [BAR, title, BAR, "%-11s" % "" + "".join("%9s" % human_rows(s) for s in sizes)]

    winner = {}
    for s in sizes:
        vals = [(data[e][s], e) for e in engines if s in data[e]]
        if vals:
            winner[s] = min(vals)[1]

    for e in engines:
        row = "%-11s" % LABEL.get(e, e)
        for s in sizes:
            row += "%9s" % (human(data[e][s]) if s in data[e] else "-")
        if any(winner.get(s) == e for s in sizes):
            row += "  <-"
        out.append(row)

    if "falcondb" in data and sizes:
        big = sizes[-1]
        rival = [data[e][big] for e in RIVALS if e in data and big in data[e]]
        if rival and big in data["falcondb"]:
            ratio = min(rival) / data["falcondb"][big]
            verb = "faster" if ratio >= 1 else "slower"
            out.append("FalconDB %.1fx %s than best rival @ %s"
                       % (max(ratio, 1 / ratio), verb, human_rows(big)))
    return out


def render(results, note_lines):
    head = ["FalconDB vs Polars / DuckDB / DataFusion", ""] + note_lines + [""]
    groups = [render_group(t, results[k]) for k, t in GROUP_TITLES if k in results]
    return head, groups


def pack(head, groups, limit=3400):
    out, cur = [], list(head)
    for g in groups:
        block = "\n".join(g)
        if len("\n".join(cur)) + len(block) > limit and len(cur) > len(head):
            out.append("\n".join(cur))
            cur = []
        cur.extend(g)
        cur.append("")
    if cur:
        out.append("\n".join(cur))
    return out


def send(parts, bot, chat):
    for i, part in enumerate(parts):
        data = urllib.parse.urlencode({
            "chat_id": chat,
            "text": "```\n%s\n```" % part,
            "parse_mode": "Markdown",
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.telegram.org/bot%s/sendMessage" % bot, data=data, method="POST")
        with urllib.request.urlopen(req) as resp:
            r = json.load(resp)
            print("part %d/%d: %s" % (i + 1, len(parts), "ok" if r.get("ok") else r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("criterion_output")
    ap.add_argument("--rivals-from-history", action="store_true",
                    help="the run measured FalconDB only; fill rivals from the history file")
    ap.add_argument("--no-record", action="store_true",
                    help="do not append this run's rival timings to the history")
    ap.add_argument("--send", action="store_true", help="post the table to Telegram")
    ap.add_argument("--bot", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    ap.add_argument("--chat", default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    args = ap.parse_args()

    fresh = parse(args.criterion_output)
    if not fresh:
        print("no benchmark results parsed from %s" % args.criterion_output)
        return 1

    history = load_history()
    notes = []

    if args.rivals_from_history:
        remembered, counts = rivals_from_history(history)
        if not remembered:
            print("no rival history yet — run a full benchmark once first")
            return 1
        results = merge(fresh, remembered)
        runs = max(counts.values())
        notes = [
            "FalconDB measured now. Rival timings are the mean",
            "of %d earlier full runs (they do not change between" % runs,
            "iterations, so re-measuring them only adds noise).",
        ]
    else:
        results = fresh
        if not args.no_record:
            n = record_rivals(fresh, history)
            save_history(history)
            print("recorded %d rival observations to %s" % (n, os.path.basename(HISTORY)))
        notes = ["All engines measured in this run.",
                 "All read the SAME Parquet file; loading is",
                 "outside the timed region. Criterion median."]

    head, groups = render(results, notes)
    parts = pack(head, groups)
    for p in parts:
        print(p)
        print("-" * 50)

    if args.send:
        if not args.bot or not args.chat:
            print("--send needs --bot/--chat or TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID")
            return 1
        send(parts, args.bot, args.chat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
