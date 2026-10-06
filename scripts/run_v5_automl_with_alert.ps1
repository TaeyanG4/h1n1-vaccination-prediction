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
    $Venv = Join-Path $Root ".venv-autogluon"
    $Py = Join-Path $Venv "Scripts\python.exe"

    if (-not (Test-Path -LiteralPath $Py)) {
        Write-Host "Creating isolated AutoGluon environment..." -ForegroundColor Cyan
        & python -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed" }
    }

    $Ready = Join-Path $Venv ".autogluon-1.6.3-ready"
    if (-not (Test-Path -LiteralPath $Ready)) {
        Write-Host "Installing AutoGluon 1.6.3..." -ForegroundColor Cyan
        & $Py -m pip install --upgrade pip setuptools wheel
        if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }

        & $Py -m pip install "autogluon.tabular[all]==1.6.3" --extra-index-url "https://download.pytorch.org/whl/cpu"
        if ($LASTEXITCODE -ne 0) { throw "AutoGluon installation failed" }

        & $Py -c "import autogluon.tabular, sys; print('AutoGluon', autogluon.tabular.__version__); print(sys.version)"
        if ($LASTEXITCODE -ne 0) { throw "AutoGluon import check failed" }
        Set-Content -LiteralPath $Ready -Value "AutoGluon 1.6.3 ready"
    }

    Write-Host "Checking frozen folds and inputs..." -ForegroundColor Cyan
    & $Py src\v5_automl.py check --config configs\v5_automl.json
    if ($LASTEXITCODE -ne 0) { throw "v5 AutoML input check failed" }

    $ArgsList = @("src\v5_automl.py", "run", "--config", "configs\v5_automl.json")
    if ((Test-Path -LiteralPath "artifacts\v5_automl\predictor\predictor.pkl") -and (-not (Test-Path -LiteralPath "artifacts\v5_automl\results.json"))) {
        $ArgsList += "--resume"
    }

    Write-Host "Starting AutoGluon v5 AutoML run..." -ForegroundColor Cyan
    $StdoutLog = Join-Path $Root "artifacts\v5_automl_stdout.log"
    $StderrLog = Join-Path $Root "artifacts\v5_automl_stderr.log"
    Write-Host "stdout: $StdoutLog"
    Write-Host "stderr: $StderrLog"
    Write-Host "AutoGluon writes normal status messages to stderr; only the Python process exit code is treated as failure."

    $Process = Start-Process -FilePath $Py -ArgumentList $ArgsList -WorkingDirectory $Root -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
    $Code = $Process.ExitCode
    if ($Code -ne 0) { throw "v5 AutoML run failed with exit code $Code" }

    Show-RunAlert -Message "H1N1 v5 AutoML completed. Check artifacts\v5_automl\model_scores.csv and results.json." -Title "Kaggle H1N1 - AutoML completed" -Success $true
    exit 0
}
catch {
    $Message = $_.Exception.Message
    Write-Error $Message
    Show-RunAlert -Message "H1N1 v5 AutoML failed: $Message" -Title "Kaggle H1N1 - AutoML failed" -Success $false
    exit 1
}
