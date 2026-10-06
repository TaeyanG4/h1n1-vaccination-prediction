$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

function Show-RunAlert {
    param(
        [string]$Message,
        [string]$Title,
        [bool]$Success
    )
    Add-Type -AssemblyName System.Windows.Forms
    if ($Success) {
        [System.Media.SystemSounds]::Asterisk.Play()
        $Icon = [System.Windows.Forms.MessageBoxIcon]::Information
    }
    else {
        [System.Media.SystemSounds]::Hand.Play()
        $Icon = [System.Windows.Forms.MessageBoxIcon]::Error
    }
    [System.Windows.Forms.MessageBox]::Show(
        $Message,
        $Title,
        [System.Windows.Forms.MessageBoxButtons]::OK,
        $Icon
    ) | Out-Null
}

try {
    $Venv = Join-Path $Root ".venv-tabicl"
    $Py = Join-Path $Venv "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Py)) {
        Write-Host "Creating isolated TabICLv2 Python 3.11 environment..." -ForegroundColor Cyan
        & py -3.11 -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "Python 3.11 venv creation failed" }
    }

    $Ready = Join-Path $Venv ".tabicl-2.2.0-cu126-ready"
    if (-not (Test-Path -LiteralPath $Ready)) {
        Write-Host "Installing CUDA PyTorch 2.7.1 and TabICL 2.2.0..." -ForegroundColor Cyan
        & $Py -m pip install --upgrade pip setuptools wheel
        if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }
        & $Py -m pip install "torch==2.7.1" --index-url "https://download.pytorch.org/whl/cu126"
        if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }
        & $Py -m pip install "tabicl==2.2.0" pandas
        if ($LASTEXITCODE -ne 0) { throw "TabICL installation failed" }
        & $Py -c "import torch, tabicl; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('tabicl', __import__('importlib.metadata').metadata.version('tabicl'))"
        if ($LASTEXITCODE -ne 0) { throw "TabICL import/GPU check failed" }
        & $Py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 2)"
        if ($LASTEXITCODE -ne 0) { throw "CUDA is not available inside .venv-tabicl" }
        Set-Content -LiteralPath $Ready -Value "TabICL 2.2.0 with torch 2.7.1 cu126 ready"
    }

    Write-Host "Checking frozen folds, protected inputs, and CUDA..." -ForegroundColor Cyan
    & $Py src\v6_tabicl.py check --config configs\v6_tabicl.json
    if ($LASTEXITCODE -ne 0) { throw "v6 TabICL input/GPU check failed" }

    if (-not (Test-Path -LiteralPath "reports\v6_tabicl_smoke.json")) {
        Write-Host "Running TabICLv2 GPU smoke test (first run also downloads the checkpoint)..." -ForegroundColor Cyan
        & $Py src\v6_tabicl.py smoke --config configs\v6_tabicl.json
        if ($LASTEXITCODE -ne 0) { throw "v6 TabICL smoke test failed" }
    }

    $StdoutLog = Join-Path $Root "artifacts\v6_tabicl_stdout.log"
    $StderrLog = Join-Path $Root "artifacts\v6_tabicl_stderr.log"
    $ArgsList = @("src\v6_tabicl.py", "run", "--config", "configs\v6_tabicl.json", "--resume")
    Write-Host "Starting resumable TabICLv2 five-fold run..." -ForegroundColor Cyan
    Write-Host "stdout: $StdoutLog"
    Write-Host "stderr: $StderrLog"
    $Process = Start-Process -FilePath $Py -ArgumentList $ArgsList -WorkingDirectory $Root -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
    if ($Process.ExitCode -ne 0) { throw "v6 TabICL run failed with exit code $($Process.ExitCode)" }

    if (-not (Test-Path -LiteralPath "artifacts\v6_ensemble\results.json")) {
        Write-Host "Running 8-family equal-weight nested ensemble diagnostic..." -ForegroundColor Cyan
        & $Py scripts\v6_ensemble_scan.py
        if ($LASTEXITCODE -ne 0) { throw "v6 ensemble diagnostic failed" }
    }

    Show-RunAlert -Message "H1N1 v6 TabICLv2 + nested ensemble completed. Check artifacts\v6_tabicl\results.json and artifacts\v6_ensemble\results.json." -Title "Kaggle H1N1 - v6 completed" -Success $true
    exit 0
}
catch {
    $Message = $_.Exception.Message
    Write-Error $Message
    Show-RunAlert -Message "H1N1 v6 failed: $Message" -Title "Kaggle H1N1 - v6 failed" -Success $false
    exit 1
}
