# Runs cargo bench and sends results to Telegram when done.
# Usage: .\bench_notify.ps1

$dir    = "."
$token  = "REDACTED_TELEGRAM_TOKEN"
$chatId = "1241608367"

function Send-Tg([string]$text) {
    $payload = @{ chat_id = $chatId; text = $text } | ConvertTo-Json
    $bytes   = [System.Text.Encoding]::UTF8.GetBytes($payload)
    try {
        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
            -Method Post -Body $bytes -ContentType "application/json; charset=utf-8" | Out-Null
    } catch {
        Write-Host "TG send failed: $_"
    }
}

function Fmt([double]$us) {
    if ($us -lt 1000)    { return "$([math]::Round($us, 0)) us" }
    if ($us -lt 1000000) { return "$([math]::Round($us / 1000, 2)) ms" }
    return "$([math]::Round($us / 1000000, 3)) s"
}

# ── start ──────────────────────────────────────────────────────────────────────
Send-Tg "Bench started - go touch grass, I got this"

Set-Location $dir

$captured = @()
cargo bench 2>&1 | ForEach-Object {
    Write-Host $_
    $captured += "$_"
}

# ── parse ──────────────────────────────────────────────────────────────────────
# Format: standalone "workload/engine/rows" line, then "time: [low MEDIAN high unit]"
$times   = @{}   # key -> median microseconds
$curKey  = $null

foreach ($line in $captured) {
    $l = $line.Trim()

    # Combined line: "bench/name/rows  time:  [low MEDIAN high unit]"
    if ($l -match '^([\w_]+/[\w_]+/\d+)\s+time:\s+\[[\d\.]+ \S+ ([\d\.]+) (\S+)') {
        $med  = [double]$Matches[2]
        $unit = $Matches[3]
        $us   = switch ($unit) {
            'ms' { $med * 1000 }
            's'  { $med * 1000000 }
            default { $med }
        }
        $times[$Matches[1]] = $us
        $curKey = $null
        continue
    }

    # Standalone name line
    if ($l -match '^([\w_]+/[\w_]+/\d+)\s*$') {
        $curKey = $Matches[1]
        continue
    }

    # Time line following a standalone name
    if ($l -match 'time:\s+\[[\d\.]+ \S+ ([\d\.]+) (\S+)' -and $curKey) {
        $med  = [double]$Matches[1]
        $unit = $Matches[2]
        $us   = switch ($unit) {
            'ms' { $med * 1000 }
            's'  { $med * 1000000 }
            default { $med }
        }
        $times[$curKey] = $us
        $curKey = $null
    }
}

# ── format ────────────────────────────────────────────────────────────────────
$workloads = @(
    @{ label = "GROUP BY";     key = "group_by_agg" },
    @{ label = "Full Scan";    key = "full_scan"    },
    @{ label = "Filter WHERE"; key = "filter_where" },
    @{ label = "ORDER BY";     key = "order_by"     },
    @{ label = "SQL Pipeline"; key = "sql_pipeline" }
)
$szLabel = @{ 10000 = "10k"; 100000 = "100k"; 1000000 = "1M" }

$msg = "FalconDB bench done!`n`n"

foreach ($wl in $workloads) {
    $rows = @()
    foreach ($sz in @(10000, 100000, 1000000)) {
        $fk = "$($wl.key)/falcondb/$sz"
        $dk = "$($wl.key)/duckdb/$sz"
        if (-not ($times.ContainsKey($fk) -and $times.ContainsKey($dk))) { continue }

        $fu    = $times[$fk]; $du = $times[$dk]
        $ratio = $du / $fu
        $who   = if ($ratio -ge 1) {
            "Falcon x$([math]::Round($ratio,1))"
        } else {
            "DuckDB x$([math]::Round(1/$ratio,1))"
        }
        $rows += "  $($szLabel[$sz]): $(Fmt $fu) vs $(Fmt $du) -> $who"
    }
    if ($rows.Count -gt 0) {
        $msg += "$($wl.label):`n" + ($rows -join "`n") + "`n`n"
    }
}

if ($msg -eq "FalconDB bench done!`n`n") {
    $msg += "(no benchmark results parsed - check console output)"
}

Write-Host "`n$msg"
Send-Tg $msg
