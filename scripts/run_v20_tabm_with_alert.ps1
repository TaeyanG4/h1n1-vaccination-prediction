$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
$Log = Join-Path $Root "artifacts\v20_tabm_runner.log"
$Stdout = Join-Path $Root "artifacts\v20_tabm_stdout.log"
$Stderr = Join-Path $Root "artifacts\v20_tabm_stderr.log"

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
    "[$(Get-Date -Format o)] v20 START" | Add-Content $Log
    $code = Run-Step @("src\v20_tabm.py","check","--config","configs\v20_tabm.json")
    if ($code -ne 0) { throw "v20 preflight failed (exit $code)" }
    Write-Host "v20 TabM running: feature/recipe screen -> grouped5 confirm -> full refit -> ensemble scan" -ForegroundColor Cyan
    Write-Host "Progress: powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_v20_tabm_progress.ps1" -ForegroundColor Cyan
    $code = Run-Step @("src\v20_tabm.py","run","--config","configs\v20_tabm.json")
    if ($code -ne 0) { throw "v20 run failed (exit $code)" }
    "[$(Get-Date -Format o)] v20 COMPLETE" | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Asterisk.Play()
    [System.Windows.Forms.MessageBox]::Show("v20 TabM completed. Full-refit and ensemble candidates are ready. No Kaggle submission was made.", "Kaggle H1N1 v20 complete", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
}
catch {
    ($_ | Out-String) | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Hand.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Hand.Play()
    [System.Windows.Forms.MessageBox]::Show("v20 failed or stopped. Re-run the same command after checking artifacts\v20_tabm_runner.log.", "Kaggle H1N1 v20 failed", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    exit 1
}
