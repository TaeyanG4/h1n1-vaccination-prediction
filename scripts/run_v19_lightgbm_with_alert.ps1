$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
$Log = Join-Path $Root "artifacts\v19_lightgbm_runner.log"
$Stdout = Join-Path $Root "artifacts\v19_lightgbm_stdout.log"
$Stderr = Join-Path $Root "artifacts\v19_lightgbm_stderr.log"

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
    if (-not (Test-Path $Py)) { throw "Missing Python environment" }
    "[$(Get-Date -Format o)] v19 START" | Add-Content $Log
    $code = Run-Step @("src\v19_lightgbm.py","check","--config","configs\v19_lightgbm.json")
    if ($code -ne 0) { throw "v19 preflight failed (exit $code)" }
    Write-Host "v19 LightGBM running: preprocessing blocks -> light Optuna -> confirm/fixed -> full refit -> ensembles" -ForegroundColor Cyan
    Write-Host "Progress: powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_v19_lightgbm_progress.ps1" -ForegroundColor Cyan
    $code = Run-Step @("src\v19_lightgbm.py","run","--config","configs\v19_lightgbm.json")
    if ($code -ne 0) { throw "v19 run failed (exit $code)" }
    "[$(Get-Date -Format o)] v19 COMPLETE" | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Asterisk.Play()
    [System.Windows.Forms.MessageBox]::Show("v19 LightGBM completed. Full-refit and ensemble candidates are ready. No Kaggle submission was made.", "Kaggle H1N1 v19 complete", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
}
catch {
    ($_ | Out-String) | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Hand.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Hand.Play()
    [System.Windows.Forms.MessageBox]::Show("v19 failed or stopped. Re-run the same command after checking artifacts\v19_lightgbm_runner.log.", "Kaggle H1N1 v19 failed", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    exit 1
}
