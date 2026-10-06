# Native Python warnings are written to stderr. PowerShell 5.1 can promote
# native stderr records to terminating errors when ErrorActionPreference=Stop,
# even when Python exits with code 0. We therefore check $LASTEXITCODE explicitly.
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$BasePy = (Get-Command python).Source
$Venv = Join-Path $Root ".venv-xgb-optuna"
$Py = Join-Path $Venv "Scripts\python.exe"
$Log = Join-Path $Root "artifacts\v17_xgb_optuna_runner.log"
$Stdout = Join-Path $Root "artifacts\v17_xgb_optuna_stdout.log"
$Stderr = Join-Path $Root "artifacts\v17_xgb_optuna_stderr.log"

function Run-PythonStep {
    param([string[]]$StepArgs)
    if (Test-Path -LiteralPath $Stdout) { Remove-Item -LiteralPath $Stdout -Force }
    if (Test-Path -LiteralPath $Stderr) { Remove-Item -LiteralPath $Stderr -Force }
    $p = Start-Process -FilePath $Py -ArgumentList $StepArgs -WorkingDirectory $Root -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    if (Test-Path -LiteralPath $Stdout) { Get-Content -LiteralPath $Stdout | Add-Content -LiteralPath $Log }
    if (Test-Path -LiteralPath $Stderr) { Get-Content -LiteralPath $Stderr | Add-Content -LiteralPath $Log }
    return $p.ExitCode
}

try {
    if (-not (Test-Path -LiteralPath $Py)) {
        Write-Host "Creating isolated Optuna environment with system site packages..." -ForegroundColor Cyan
        & $BasePy -m venv $Venv --system-site-packages
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    }
    & $Py -c "import optuna" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing Optuna into .venv-xgb-optuna..." -ForegroundColor Cyan
        & $Py -m pip install "optuna>=4,<5" --disable-pip-version-check
        if ($LASTEXITCODE -ne 0) { throw "Optuna installation failed" }
    }
    New-Item -ItemType Directory -Path (Join-Path $Root "artifacts") -Force | Out-Null
    "[$(Get-Date -Format o)] v17 START" | Tee-Object -FilePath $Log -Append
    $code = Run-PythonStep @("src\v17_xgb_optuna.py","check","--config","configs\v17_xgb_optuna.json")
    if ($code -ne 0) { throw "v17 preflight failed (exit $code)" }
    Write-Host "v17 Optuna running. Progress: .\scripts\show_v17_xgb_optuna_progress.ps1" -ForegroundColor Cyan
    $code = Run-PythonStep @("src\v17_xgb_optuna.py","run","--config","configs\v17_xgb_optuna.json")
    if ($code -ne 0) { throw "v17 run failed (exit $code)" }
    "[$(Get-Date -Format o)] v17 COMPLETE" | Tee-Object -FilePath $Log -Append

    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Asterisk.Play()
    [System.Windows.Forms.MessageBox]::Show("v17 XGBoost Optuna + full refit completed. Check artifacts\v17_xgb_optuna\results.json.", "Kaggle H1N1 v17 complete", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
}
catch {
    ($_ | Out-String) | Tee-Object -FilePath $Log -Append
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Hand.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Hand.Play()
    [System.Windows.Forms.MessageBox]::Show("v17 failed or stopped. The Optuna study is resumable. Re-run the same command after checking artifacts\v17_xgb_optuna_runner.log.", "Kaggle H1N1 v17 failed", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    exit 1
}
