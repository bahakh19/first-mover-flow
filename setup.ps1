# One-command setup for Windows (PowerShell 5.1 or 7; also runs under PowerShell 7 on macOS and Linux).
# Python does not need to be installed.
#
#   .\setup.ps1              standard: package + notebooks + tests + generative models + Excel/Parquet input
#   .\setup.ps1 -Full        also the research-notebook and data-gathering extras
#   .\setup.ps1 -Minimal     the package only (strategy, CLI, statistics)
#   .\setup.ps1 -NoKernel    do not register the "Python (fmflow)" Jupyter kernel
#
# Double-clicking setup.cmd does the same. If scripts are blocked, run:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# How: uses uv (https://docs.astral.sh/uv/), installing it if needed; uv downloads a suitable Python by itself and
# creates .venv\ from the lock file. Without internet access to astral.sh, it falls back to py -3 -m venv + pip.
param([switch]$Full, [switch]$Minimal, [switch]$NoKernel)
# "Continue": in Windows PowerShell 5.1, "Stop" turns any redirected stderr line of a native program into an error.
# Native exit codes are checked explicitly instead.
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

function Say($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }

$onWindows = $env:OS -eq "Windows_NT"          # $IsWindows does not exist in Windows PowerShell 5.1

$extras = @("--extra", "plot", "--extra", "notebooks", "--extra", "dev", "--extra", "gen", "--extra", "io")
$pipExtras = "plot,notebooks,dev,gen,io"
if ($Full) { $extras = @("--all-extras"); $pipExtras = "all" }
if ($Minimal) { $extras = @(); $pipExtras = "" }

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = if ($onWindows) {
        @("$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe", "$env:LOCALAPPDATA\uv\uv.exe")
    } else { @("$HOME/.local/bin/uv", "$HOME/.cargo/bin/uv") }
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

$uv = Find-Uv
if (-not $uv) {
    Say "Installing uv (a fast Python package and version manager)"
    try {
        if ($onWindows) {
            powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
            $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
        } else {
            sh -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
            $env:PATH = "$HOME/.local/bin:$env:PATH"
        }
    } catch { Write-Host "uv installation failed: $($_.Exception.Message)" }
    $uv = Find-Uv
}

$venvPy = if ($onWindows) { Join-Path $PSScriptRoot ".venv\Scripts\python.exe" } else { Join-Path $PSScriptRoot ".venv/bin/python" }

function Install-WithPip {
    $py = $null
    foreach ($c in @("py", "python", "python3")) {
        if (Get-Command $c -ErrorAction SilentlyContinue) {
            $args3 = @(); if ($c -eq "py") { $args3 = @("-3") }
            & $c @args3 -c "import sys; sys.exit(sys.version_info < (3, 10))" 2>$null
            if ($LASTEXITCODE -eq 0) { $py = $c; break }
        }
    }
    if (-not $py) {
        Write-Host "No Python >= 3.10 found and uv could not be used."
        Write-Host "Install Python from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') or: winget install Python.Python.3.12"
        exit 1
    }
    Say "Creating .venv with $py and pip"
    if ($py -eq "py") { & py -3 -m venv --clear .venv } else { & $py -m venv --clear .venv }
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed (exit code $LASTEXITCODE)" }
    & $venvPy -m pip install --upgrade pip
    if ($pipExtras) { & $venvPy -m pip install -e ".[$pipExtras]" } else { & $venvPy -m pip install -e . }
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit code $LASTEXITCODE)" }
}

if ($uv) {
    Say "Creating .venv with uv ($(& $uv --version))"
    & $uv sync @extras
    if ($LASTEXITCODE -ne 0) {
        Say "uv sync failed (network or proxy?): falling back to python -m venv + pip"
        $uv = $null
        Install-WithPip
    }
} else {
    Say "uv is unavailable: falling back to python -m venv + pip"
    Install-WithPip
}

if (-not $NoKernel) {
    & $venvPy -c "import ipykernel" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Say "Registering the Jupyter kernel 'Python (fmflow)'"
        & $venvPy -m ipykernel install --user --name fmflow --display-name "Python (fmflow)" | Out-Null
        Write-Host "done"
    }
}

Say "Checking the environment"
& $venvPy -m fmflow.cli doctor

Write-Host ""
Write-Host "Setup complete. Activate the environment once per terminal, from this folder:"
Write-Host ""
if ($onWindows) {
    Write-Host "  .venv\Scripts\Activate.ps1          # PowerShell (cmd.exe: .venv\Scripts\activate.bat)"
    Write-Host "  # if PowerShell refuses to run it: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
} else {
    Write-Host "  . .venv/bin/Activate.ps1             # PowerShell (bash / zsh: source .venv/bin/activate)"
}
Write-Host ""
Write-Host "then:"
Write-Host ""
Write-Host "  python tasks.py demo               # the pre-registered out-of-sample test (about a minute)"
Write-Host "  python tasks.py lab                # JupyterLab on the notebooks"
Write-Host "  python tasks.py test               # fast tests"
Write-Host "  fmflow --help                      # the command-line tools"
if ($uv) {
    # uv may have been installed by this script into a folder the current session does not have on its PATH yet
    $uvCmd = if (Get-Command uv -ErrorAction SilentlyContinue) { "uv" } else { "& `"$uv`"" }
    Write-Host ""
    Write-Host "Without activating:  $uvCmd run python tasks.py demo"
}
