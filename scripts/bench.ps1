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

param(
    [string]$Filter   = "",
    [string]$Save     = "",
    [string]$Baseline = "",
    [string]$Out      = "$env:TEMP\falcon_bench.txt"
)

$ErrorActionPreference = "Stop"

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
