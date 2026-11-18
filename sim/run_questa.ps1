<#
  run_questa.ps1 - QuestaSim driver for the sim/ directory (Windows host).

  Similar in spirit to run.sh (Verilator), but uses vlib/vlog/vsim.
  Defaults are tuned for filelist_snax_gemm_prefill.f and Wave=none.

  Examples
    .\run_questa.bat
    .\run_questa.bat -Filelist filelist_snax_gemm_prefill.f -Top tb_snax_gemm_prefill
    .\run_questa.bat -Wave vcd -Scope /tb_snax_gemm_prefill/dut
    .\run_questa.bat -Wave wlf -Open
    .\run_questa.bat -Clean -NoSv
    .\run_questa.bat -Plusargs +TEST=seq_rand -VsimArgs -sv_seed -VsimArgs 42

  Multi-dimensional / unpacked arrays:
    .\run_questa.bat -Wave vcd -ArrayScope /tb_snax_attention/mem_k
    .\run_questa.bat -Wave vcd -ArrayScope /tb_snax_attention/mem_k,/tb_snax_attention/dut/score_array,/tb_snax_attention/dut/wt_in_i
    .\run_questa.bat -Wave wlf -ArrayScope /tb_snax_attention/mem_k -Open

  -ArrayScope accepts signal paths or wildcards.  When omitted, common arrays
  under /<top>/mem_*, /<top>/dut/*_array, wt_in_i, output lane arrays,
  p_fifo.mem and latch_array.mem are added automatically.  Use
  -NoArrayDefault to disable that list.  VCD array dumping uses
  "vcd add -r -internal -in -out"; WLF logs the array with "log -r".
#>
[CmdletBinding()]
param(
    [string]   $Filelist = "filelist_snax_gemm_prefill.f",
    [string]   $Top      = "",
    [ValidateSet("none", "vcd", "wlf")]
    [string]   $Wave     = "none",
    [string]   $Scope    = "",
    [string[]] $ArrayScope = @(),
    [switch]   $NoArrayDefault,
    [string[]] $Plusargs = @(),
    [string[]] $VsimArgs = @(),
    [string[]] $VlogArgs = @(),
    [string[]] $Incdir   = @(),
    [string[]] $Define   = @(),
    [string]   $QuestaBin = "",
    [string]   $License   = "C:\MentorGraphics\Install\License\license.dat",
    [string]   $Surfer    = "C:\Users\nguyen\Desktop\surfer.exe",
    [switch]   $Open,
    [switch]   $NoSv,
    [switch]   $Clean
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Write-Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor DarkGray }
function Fail($msg) {
    Write-Host ""
    Write-Host "!! $msg" -ForegroundColor Red
    exit 1
}

function Invoke-Logged {
    param([string]$Exe, [string[]]$Arguments, [string]$LogFile)
    if (Test-Path $LogFile) { Remove-Item $LogFile -Force }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LogFile | Out-Host
        $rc = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
    $lines = @()
    if (Test-Path $LogFile) { $lines = @(Get-Content -LiteralPath $LogFile) }
    return [pscustomobject]@{ ExitCode = $rc; Lines = $lines }
}

function Find-QuestaBin {
    if ($Questabin -and (Test-Path (Join-Path $QuestaBin "vsim.exe"))) { return $QuestaBin }
    $cmd = Get-Command vsim.exe -ErrorAction SilentlyContinue
    if ($cmd) { return (Split-Path $cmd.Source) }
    if ($env:QUESTA_HOME) {
        foreach ($sub in @("win64", "win32", "bin")) {
            $p = Join-Path $env:QUESTA_HOME $sub
            if (Test-Path (Join-Path $p "vsim.exe")) { return $p }
        }
    }
    foreach ($p in @("C:\questasim64_2024.1\win64", "C:\questasim64_2023.4\win64")) {
        if (Test-Path (Join-Path $p "vsim.exe")) { return $p }
    }
    return $null
}

# vlog resolves filelist entries relative to the current directory (sim/).
function Get-FilelistEntries($path) {
    $out = @()
    foreach ($raw in Get-Content -LiteralPath $path) {
        $line = $raw.Trim()
        if ($line -eq "") { continue }
        if ($line.StartsWith("#") -or $line.StartsWith("//")) { continue }
        $line = ($line -split "\s+//")[0].Trim()
        if ($line -eq "") { continue }
        if ($line -match '^[+-]') { continue }
        if ([System.IO.Path]::IsPathRooted($line)) { $full = $line }
        else { $full = Join-Path $Root $line }

        if ($full -match '[*?]') {
            $hits = @(Resolve-Path -Path $full -ErrorAction SilentlyContinue)
            foreach ($h in $hits) { $out += $h.Path }
        } else {
            $out += $full
        }
    }
    return @($out)
}

function Resolve-Top {
    param([string]$Requested, [string]$FilelistPath)

    $mods = New-Object System.Collections.Generic.List[string]
    foreach ($f in (Get-FilelistEntries $FilelistPath)) {
        if (-not (Test-Path $f)) {
            $f = Join-Path $Root $f
            if (-not (Test-Path $f)) { Fail "Source from filelist does not exist: $f" }
        }
        foreach ($line in Get-Content -LiteralPath $f) {
            $t = $line.Trim()
            if ($t.StartsWith("//") -or $t.StartsWith("*")) { continue }
            if ($t -match '^(?:module|program)\s+([A-Za-z_][A-Za-z0-9_$]*)') {
                $mods.Add($Matches[1])
            }
        }
    }

    if ($Requested) {
        if (-not ($mods -contains $Requested)) {
            Fail "Top '$Requested' is not defined in the filelist. Found: $($mods -join ', ')"
        }
        return $Requested
    }

    $cands = @($mods | Where-Object { $_ -match '^(?i)(tb|testbench|test|top)' } | Select-Object -Unique)
    if ($cands.Count -eq 1) { return $cands[0] }
    if ($cands.Count -eq 0) {
        Fail "Could not auto-detect the testbench. Pass -Top <module>. Modules found: $($mods -join ', ')"
    }
    Fail "Multiple testbench candidates: $($cands -join ', '). Pick one with -Top <module>."
}

$QBin = Find-QuestaBin
if (-not $QBin) {
    Fail "Questa not found. Put vsim.exe on PATH, set QUESTA_HOME, or pass -QuestaBin <dir>."
}
$Vlib   = Join-Path $QBin "vlib.exe"
$Vlog   = Join-Path $QBin "vlog.exe"
$Vsim   = Join-Path $QBin "vsim.exe"
$Wlf2Vcd = Join-Path $QBin "wlf2vcd.exe"

if (-not (Test-Path $Filelist)) {
    $candidate = Join-Path $Root $Filelist
    if (Test-Path $candidate) { $Filelist = $candidate }
    else { Fail "Filelist not found: $Filelist" }
}

# ------------------------------------------------------------------ license --
if ($License) {
    $env:MGLS_LICENSE_FILE = $License
    $env:LM_LICENSE_FILE   = $License
    Write-Ok "license: $License"
}
if ($env:MGLS_LICENSE_FILE) {
    $first = @($env:MGLS_LICENSE_FILE -split ';')[0]
    if ($first -and ($first -notmatch '@') -and (-not (Test-Path $first))) {
        Write-Host "    warning: MGLS_LICENSE_FILE points to a missing file: $first" -ForegroundColor Yellow
    }
}

# ------------------------------------------------------------------- layout --
$WorkDir = Join-Path $Root "work"
$WaveDir = Join-Path $Root "waves"
$LogDir  = Join-Path $Root "logs"

if ($Clean) {
    Write-Step "Cleaning generated directories"
    foreach ($d in @($WorkDir, $WaveDir, $LogDir)) {
        if (Test-Path $d) { Remove-Item $d -Recurse -Force }
    }
    Write-Ok "work/ waves/ logs/ removed"
}
foreach ($d in @($WorkDir, $WaveDir, $LogDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

$TopName = Resolve-Top -Requested $Top -FilelistPath $Filelist
Write-Ok "filelist: $Filelist"
Write-Ok "top: $TopName"

# The TB uses $readmemh("sim_gemm_prefill/hex/...") relative to this directory.
$dataDir = Join-Path $Root "sim_gemm_prefill"
if (-not (Test-Path (Join-Path $dataDir "hex\weight_packets.hex"))) {
    Write-Host "    warning: sim_gemm_prefill/hex not found under sim/." -ForegroundColor Yellow
    Write-Host "             Copy from ..\tb\sim_gemm_prefill before running this TB." -ForegroundColor Yellow
}

# ------------------------------------------------------------------- compile --
$CompileLog = Join-Path $LogDir "compile.log"
Write-Step "Compiling (vlog) -> $CompileLog"

if (-not (Test-Path (Join-Path $WorkDir "_info"))) {
    $vlibRun = Invoke-Logged -Exe $Vlib -Arguments @("work") -LogFile (Join-Path $LogDir "vlib.log")
    if ($vlibRun.ExitCode -ne 0) { Fail "Could not create the 'work' library (exit $($vlibRun.ExitCode))." }
    Write-Ok "created library work"
}

$argsVlog = @("-work", "work")
if (-not $NoSv) { $argsVlog += "-sv" }
foreach ($d in $Incdir) { $argsVlog += "+incdir+$d" }
foreach ($d in $Define) { $argsVlog += "+define+$d" }
$argsVlog += $VlogArgs
$argsVlog += @("-f", $Filelist)

Write-Ok "vlog $($argsVlog -join ' ')"
$compile = Invoke-Logged -Exe $Vlog -Arguments $argsVlog -LogFile $CompileLog
$errs = @($compile.Lines | Where-Object { "$_" -match '\*\* (Error|Fatal)' })
if ($compile.ExitCode -ne 0 -or $errs.Count -gt 0) {
    $errs | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    Fail "Compilation failed (exit $($compile.ExitCode)). See $CompileLog"
}
Write-Ok "compile clean"

# ---------------------------------------------------------------- simulation --
Write-Step "Simulating top '$TopName' (wave=$Wave)"

$SimLog = Join-Path $LogDir "$TopName.log"
try { if (Test-Path $SimLog) { Remove-Item $SimLog -Force -ErrorAction SilentlyContinue } } catch {}

$argsVsim = @("-c", "-work", "work")
if ($Wave -ne "none") { $argsVsim += "-voptargs=+acc" }

$vcdPath = Join-Path $WaveDir "$TopName.vcd"
$wlfPath = Join-Path $WaveDir "$TopName.wlf"
$wavePath = $null

if ($Wave -ne "none") {
    foreach ($p in @($vcdPath, $wlfPath)) { if (Test-Path $p) { Remove-Item $p -Force } }
}

$dumpScope = "/$TopName/*"
if ($Scope) {
    $dumpScope = $Scope
    if ($dumpScope -notmatch '\*$') {
        if (-not $dumpScope.EndsWith("/")) { $dumpScope += "/" }
        $dumpScope += "*"
    }
}

# Unpacked arrays (and arrays of packed vectors) are often missed by a plain
# scalar scope in some VCD flows.  Add them explicitly.  Caller-provided
# -ArrayScope paths override the default list.
$arrayPatterns = @()
if ($ArrayScope.Count -gt 0) {
    # The .bat launcher may pass "-ArrayScope a,b,c" as one string.  Split on
    # commas so each path becomes its own vcd/log command.
    foreach ($entry in $ArrayScope) {
        foreach ($part in ($entry -split ',')) {
            $p = $part.Trim()
            if ($p) { $arrayPatterns += $p }
        }
    }
} elseif (-not $NoArrayDefault -and $Wave -ne "none") {
    $arrayPatterns = @(
        "/$TopName/mem_*",
        "/$TopName/dut/*_array",
        "/$TopName/dut/wt_in_i",
        "/$TopName/dut/out_vec_fp16_lanes_o",
        "/$TopName/dut/u_latch_array/mem",
        "/$TopName/dut/u_p_fifo/mem",
        "/$TopName/dut/u_softmax_requantizer/s1_score",
        "/$TopName/dut/u_softmax_requantizer/score_msk",
        "/$TopName/dut/u_softmax_requantizer/score_dist",
        "/$TopName/dut/u_softmax_requantizer/mag",
        "/$TopName/dut/u_softmax_requantizer/code_d",
        "/$TopName/dut/u_matmul_engine/aligned_prod_q1"
    )
}

if ($Wave -eq "vcd" -and $arrayPatterns.Count -gt 0) {
    Write-Host "    note: this Questa VCD backend may flatten/skip unpacked arrays; -Wave wlf keeps them as arrays." -ForegroundColor Yellow
}

$doScript = ""
switch ($Wave) {
    "vcd" {
        $doScript += "vcd file {$($vcdPath.Replace('\', '/'))}; vcd add -r -internal -in -out $dumpScope; "
        $wavePath = $vcdPath
    }
    "wlf" {
        $argsVsim += @("-wlf", ($wlfPath.Replace('\', '/')))
        $doScript += "log -r $dumpScope; "
        $wavePath = $vcdPath
    }
    "none" { }
}

# Explicitly add arrays/memories.  This is what makes unpacked arrays such as
# score_array[0:15], wt_in_i[0:15], mem_k[0:31], p_fifo.mem[0:63] and
# latch_array.mem[0:15] visible as arrays instead of being skipped.
foreach ($arr in $arrayPatterns) {
    switch ($Wave) {
        "vcd" {
            $vcdAdd = "vcd add -r -internal -in -out"
            $doScript += "$vcdAdd {$arr}; "
        }
        "wlf" {
            $doScript += "log -r {$arr}; "
        }
    }
}

$doScript += "run -all; quit -f"

$argsVsim += $Plusargs
$argsVsim += $VsimArgs
$argsVsim += "work.$TopName"
$argsVsim += @("-do", $doScript)

Write-Ok "vsim $($argsVsim -join ' ')"
$sim = Invoke-Logged -Exe $Vsim -Arguments $argsVsim -LogFile $SimLog
$simErrs = @($sim.Lines | Where-Object { "$_" -match '\*\* (Error|Fatal)' })
if ($sim.ExitCode -ne 0 -or $simErrs.Count -gt 0) {
    $simErrs | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    $lic = @($sim.Lines | Where-Object { "$_" -match 'Unable to checkout a license|Invalid license' })
    if ($lic.Count -gt 0) {
        Write-Host ""
        Write-Host "Questa license checkout failed." -ForegroundColor Yellow
        Write-Host "  MGLS_LICENSE_FILE = $env:MGLS_LICENSE_FILE" -ForegroundColor Yellow
    }
    Fail "Simulation failed (exit $($sim.ExitCode)). See $SimLog"
}

$cpu = $sim.Lines | Where-Object { "$_" -match '^\s*#\s*Errors:.*Warnings:' } | Select-Object -Last 1
if ($cpu) { Write-Ok $cpu.Trim() }
Write-Ok "simulation complete"

# ---------------------------------------------------------------- wave export --
if ($Wave -eq "wlf") {
    Write-Step "Converting WLF -> VCD"
    if (-not (Test-Path $wlfPath)) { Fail "WLF dump not found: $wlfPath" }
    & $Wlf2Vcd -quiet -o ($vcdPath.Replace('\', '/')) ($wlfPath.Replace('\', '/')) | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "wlf2vcd failed (exit $LASTEXITCODE)" }
}

if ($Wave -ne "none") {
    if (-not (Test-Path $wavePath)) { Fail "Waveform was not produced: $wavePath" }
    $waveBytes = (Get-Item $wavePath).Length
    $sizeMb = [math]::Round($waveBytes / 1MB, 2)
    Write-Ok "waveform: $wavePath ($sizeMb MB)"
    if ($waveBytes -lt 256) {
        Write-Host "    warning: dump is header-only - check -Scope." -ForegroundColor Yellow
    }

    if ($Open) {
        if ($Wave -eq "wlf" -and (Test-Path $wlfPath)) {
            Write-Step "Opening Questa GUI with WLF (array-aware)"
            Start-Process -FilePath $Vsim -ArgumentList @("-view", "`"$wlfPath`"")
            Write-Ok "launched $Vsim -view $wlfPath"
        } elseif (-not (Test-Path $Surfer)) {
            Write-Host "    Surfer not found at $Surfer - waveform left on disk." -ForegroundColor Yellow
        } else {
            Write-Step "Opening Surfer"
            Start-Process -FilePath $Surfer -ArgumentList ("`"$wavePath`"")
            Write-Ok "launched $Surfer"
        }
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
