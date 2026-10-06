$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
$Log = Join-Path $Root "artifacts\v18_xgb_finalize_ensemble_runner.log"
$Stdout = Join-Path $Root "artifacts\v18_xgb_finalize_ensemble_stdout.log"
$Stderr = Join-Path $Root "artifacts\v18_xgb_finalize_ensemble_stderr.log"

function Run-Step {
    param([string[]]$StepArgs)
    Remove-Item $Stdout,$Stderr -Force -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $Py -ArgumentList $StepArgs -WorkingDirectory $Root -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    if (Test-Path $Stdout) { Get-Content $Stdout | Add-Content $Log }
    if (Test-Path $Stderr) { Get-Content $Stderr | Add-Content $Log }
    return $p.ExitCode
}

try {
    if (-not (Test-Path $Py)) { throw "Missing .venv-xgb-optuna Python" }
    "[$(Get-Date -Format o)] v18 START" | Add-Content $Log
    $code = Run-Step @("src\v18_xgb_finalize_ensemble.py","check","--config","configs\v18_xgb_finalize_ensemble.json")
    if ($code -ne 0) { throw "v18 preflight failed (exit $code)" }
    Write-Host "v18 running: top-5 fixed-round check -> XGB full refit -> v12+XGB ensemble" -ForegroundColor Cyan
    Write-Host "Progress: powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_v18_progress.ps1" -ForegroundColor Cyan
    $code = Run-Step @("src\v18_xgb_finalize_ensemble.py","run","--config","configs\v18_xgb_finalize_ensemble.json")
    if ($code -ne 0) { throw "v18 run failed (exit $code)" }
    "[$(Get-Date -Format o)] v18 COMPLETE" | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Asterisk.Play()
    [System.Windows.Forms.MessageBox]::Show("v18 completed. Top XGB full-refit and v12+XGB ensemble candidates are ready. No Kaggle submission was made.", "Kaggle H1N1 v18 complete", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
}
catch {
    ($_ | Out-String) | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Hand.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Hand.Play()
    [System.Windows.Forms.MessageBox]::Show("v18 failed or stopped. Re-run the same command after checking artifacts\v18_xgb_finalize_ensemble_runner.log.", "Kaggle H1N1 v18 failed", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    exit 1
}
