param(
    [string]$TestName = "1",
    [switch]$NoVcd
)

$ErrorActionPreference = "Stop"
$GlsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $GlsDir

$Vlog = "C:\questasim64_2024.1\win64\vlog.exe"
$Vopt = "C:\questasim64_2024.1\win64\vopt.exe"
$Vsim = "C:\questasim64_2024.1\win64\vsim.exe"

if (Test-Path "work") { Remove-Item -Recurse -Force "work" }
& "C:\questasim64_2024.1\win64\vlib.exe" work

$VlogArgs = @(
    "-sv", "-64",
    "-timescale", "1ns/1ps",
    "+nospecify", "+notimingchecks",
    "-suppress", "2583,2685,2244,2181",
    "-f", "filelist_gls.f"
)
if (-not $NoVcd) {
    $VlogArgs += "+define+DUMP_VCD"
}

Write-Host "[1/3] Compiling GLS netlist + libraries with vlog..."
& $Vlog @VlogArgs | Tee-Object -FilePath "vlog_gls.log"
if ($LASTEXITCODE -ne 0) { throw "vlog failed!" }

# Speed-optimized vopt:
#   -O5 enables maximum compiler optimization
#   +acc=n+/tb_gemm_os_gls/dut enables net visibility ONLY inside dut for VCD
$VoptArgs = @("-64", "-O5", "-suppress", "7061,2718,2685,2583", "tb_gemm_os_gls", "-o", "opt_gls")
if (-not $NoVcd) {
    $VoptArgs = @("-64", "-O5", "-suppress", "7061,2718,2685,2583", "+acc=n", "tb_gemm_os_gls", "-o", "opt_gls")
}
Write-Host "[2/3] Optimizing with vopt (-O5)..."
& $Vopt @VoptArgs | Tee-Object -FilePath "vopt_gls.log"
if ($LASTEXITCODE -ne 0) { throw "vopt failed!" }

Write-Host "[3/3] Running vsim (+TEST=$TestName)..."
& $Vsim -c -64 opt_gls "+TEST=$TestName" "+NO_WAVE" -do "run -all; quit -f" | Tee-Object -FilePath "vsim_gls_test${TestName}.log"
if ($LASTEXITCODE -ne 0) { throw "vsim failed!" }
