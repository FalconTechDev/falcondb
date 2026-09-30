# Benchmark runner. Use this instead of a bare `cargo bench`.
#
# Why: on a hybrid CPU (13700H = 6 P-cores + 8 E-cores) rayon's threads land on a
# different P/E mix each run, and unpinned run-to-run noise on *unchanged code*
# measured 25-46%. That is larger than any optimization worth making, and it
# produced three wrong calls in a row: two phantom regressions and one phantom win.
#
# Pinned to the P-cores, the same comparison lands within ~1%.
#
# Two rules that come with it:
#   - Discard the first run. A cold run is systematically slower (up to 14% on the
#     smallest cases); the residual drift is one-directional, not random.
#   - Only compare within a run, or against a baseline recorded by this script.
#     Numbers from a full run and a `-- falcondb` run are not comparable: the
#     absolute times differ by 2x because of cache and thermal state, not code.
#
# Usage:
#   scripts/bench.ps1                      # full suite, compare to last baseline
#   scripts/bench.ps1 -Filter falcondb     # one engine
#   scripts/bench.ps1 -Save mychange       # record a baseline to compare against
#   scripts/bench.ps1 -Baseline mychange   # compare against that baseline
#
# A baseline only covers the groups that were run when it was saved. Passing
# -Baseline together with a -Filter that reaches any other group makes criterion
# panic ("Baseline 'x' must exist"). Either keep the filter identical to the run
# that saved it, or save a fresh baseline and compare engines within the run.
#
# Refuses to run on battery. This is a laptop CPU: on battery it holds its 2.4 GHz
# base clock instead of turbo, and every engine measures about twice as slow. That
# is invisible in the output -- the numbers look like ordinary numbers -- and it
# cost three full runs and two wrong conclusions before anyone checked the power
# state. Relative comparisons within one run survive it; absolute numbers, and any
# comparison against a baseline saved on AC, do not. Override with -AllowBattery
# if you only need engines compared against each other inside this one run.

param(
    [string]$Filter   = "",
    [string]$Save     = "",
    [string]$Baseline = "",
    [string]$Out      = "$env:TEMP\falcon_bench.txt",
    [switch]$AllowBattery,
    [switch]$AllowBusy
)

$ErrorActionPreference = "Stop"

# BatteryStatus 1 means discharging; 2 means running on AC.
$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($battery -and $battery.BatteryStatus -eq 1) {
    $msg = "On battery ($($battery.EstimatedChargeRemaining)% charge). " +
           "This CPU drops to its 2.4 GHz base clock, so every engine measures " +
           "roughly twice as slow and nothing here is comparable with a run made " +
           "on AC power."
    if (-not $AllowBattery) {
        Write-Error "$msg`nPlug in, or pass -AllowBattery to compare engines within this run only."
        exit 1
    }
    Write-Warning "$msg`nContinuing because -AllowBattery was passed: compare engines against each other, not against saved numbers."
}
# Refuses to run while another process is competing for the machine.
#
# Same failure as the battery one and just as invisible: a run made while
# something else was busy looked like ordinary output, with every engine about
# twice as slow -- FalconDB 20.8us -> 42.2us on int_sum/100k, but also Polars
# 14.9 -> 27.6, DuckDB 408 -> 593 and DataFusion 2.39ms -> 5.08ms. When the
# rivals move with us the machine moved, not the code, and the other tell-tale is
# the confidence intervals widening from about 1% to 7%.
#
# That run cost a full suite before anyone looked at the process list, where a
# javaw holding 1.5 GB had been taking cores throughout. A two-second load check
# is cheap against half an hour of benchmarking.
#
# Measures idle time rather than the load counter, which lags, and names the
# heaviest processes so there is something to act on.
#
# Samples repeatedly instead of deciding on one look. The first version refused on
# a single two-second sample and tripped straight after a finished run, when
# Windows was still winding processes down and the only names it could offer were
# svchost, System and dwm -- nothing anyone can close. Transient busy-ness is
# exactly what a benchmark should wait out rather than refuse over.
$busy = 100
for ($attempt = 1; $attempt -le 6; $attempt++) {
    $idle = (Get-Counter '\Processor(_Total)\% Idle Time' -SampleInterval 1 -MaxSamples 2 -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty CounterSamples |
        Measure-Object -Property CookedValue -Average).Average
    if ($null -eq $idle) { $busy = 0; break }
    $busy = [math]::Round(100 - $idle, 1)
    if ($busy -le 10) { break }
    if ($attempt -lt 6) {
        Write-Host "Machine is $busy% busy; waiting for it to settle ($attempt/6)..."
        Start-Sleep -Seconds 5
    }
}
if ($busy -gt 10) {
    # Ranks by CPU used *during a sample*, not by the process's lifetime total.
    #
    # The first version sorted on Process.CPU, which is cumulative since the
    # process started, and so it blamed whatever had worked hardest at some point
    # today. On a refused run it named svchost, System and dwm -- 9089s, 6564s and
    # 6240s of lifetime CPU between them and essentially idle at that moment --
    # while the process actually taking 21.8% of the machine was Telegram, which
    # the list did not mention. Naming the wrong process is worse than naming
    # none, because it sends you to close something harmless and leaves the real
    # one running.
    $n = [Environment]::ProcessorCount
    $before = @{}
    foreach ($proc in Get-Process -ErrorAction SilentlyContinue) {
        if ($proc.CPU) { $before[$proc.Id] = $proc.CPU }
    }
    Start-Sleep -Seconds 3
    $live = foreach ($proc in Get-Process -ErrorAction SilentlyContinue) {
        $prev = $before[$proc.Id]
        if ($null -ne $prev -and $proc.CPU) {
            $used = $proc.CPU - $prev
            if ($used -gt 0.05) {
                [PSCustomObject]@{
                    Name = $proc.ProcessName
                    Pct  = [math]::Round($used / 3 / $n * 100, 1)
                    MB   = [math]::Round($proc.WorkingSet / 1MB)
                }
            }
        }
    }
    $live = $live | Sort-Object Pct -Descending

    # System processes cannot be closed, so listing them under "close this" is
    # useless advice; they are reported separately.
    $system = @('System','Idle','svchost','dwm','csrss','wininit','services','lsass',
                'Registry','MemCompression','audiodg','taskhostw','explorer')
    $closable = $live | Where-Object { $_.Name -notin $system } | Select-Object -First 3 |
        ForEach-Object { "$($_.Name) ($($_.Pct)% now, $($_.MB)MB)" }
    $heaviest = $live | Select-Object -First 3 |
        ForEach-Object { "$($_.Name) ($($_.Pct)% now)" }

    $msg = "The machine is still $busy% busy after waiting 30s. " +
           "Another process competing for cores or memory bandwidth slows every " +
           "engine at once, which looks like ordinary output and is comparable " +
           "with nothing.`nUsing CPU right now: " + ($heaviest -join ", ")
    if ($closable) {
        $msg += "`nWorth closing: " + ($closable -join ", ")
    } else {
        $msg += "`nNothing closable is busy -- this is system activity. Wait and retry."
    }
    if (-not $AllowBusy) {
        Write-Error "$msg`nOr pass -AllowBusy to measure anyway."
        exit 1
    }
    Write-Warning "$msg`nContinuing because -AllowBusy was passed: treat the absolute numbers as unusable."
}

$env:CARGO_TARGET_DIR = "E:\rust-target"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# The bench links DuckDB dynamically; the DLL has to sit next to the test binary.
Copy-Item "$repo\lib\duckdb\duckdb.dll" "$env:CARGO_TARGET_DIR\release\deps\duckdb.dll" -Force

# FFF = logical CPUs 0-11 = the six P-cores. Child processes inherit the mask, so
# the binary cargo spawns is pinned too. /high stops background work from
# interleaving mid-sample.
$affinity = "FFF"

$criterionArgs = @()
if ($Save)     { $criterionArgs += "--save-baseline", $Save }
if ($Baseline) { $criterionArgs += "--baseline", $Baseline }
# Quoted: the filter is a regex and an unquoted "a|b" is a pipe to cmd.
if ($Filter)   { $criterionArgs += ('"' + $Filter + '"') }

$cmd = "cargo bench --bench query_bench -- " + ($criterionArgs -join " ")

# Redirect inside cmd, not in PowerShell. `native 2>&1` in PS 5.1 wraps every
# stderr line in an ErrorRecord, and with ErrorActionPreference=Stop cargo's
# ordinary "Compiling ..." progress then kills the run.
cmd /c "(start /affinity $affinity /high /b /wait $cmd) > `"$Out`" 2>&1"
if ($LASTEXITCODE -ne 0) { Write-Host "bench exited $LASTEXITCODE - see $Out" }

Write-Host "Results: $Out"
