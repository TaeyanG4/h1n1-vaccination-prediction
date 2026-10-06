$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
$Log = Join-Path $Root "artifacts\v21_remaining_suite_runner.log"
$Stdout = Join-Path $Root "artifacts\v21_remaining_suite_stdout.log"
$Stderr = Join-Path $Root "artifacts\v21_remaining_suite_stderr.log"

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
    "[$(Get-Date -Format o)] remaining-suite START" | Add-Content $Log
    $code = Run-Step @("src\v21_remaining_suite.py","check","--config","configs\v21_remaining_suite.json")
    if ($code -ne 0) { throw "preflight failed (exit $code)" }
    $code = Run-Step @("src\v21_remaining_suite.py","smoke","--config","configs\v21_remaining_suite.json")
    if ($code -ne 0) { throw "smoke failed (exit $code)" }
    Write-Host "Running remaining high-value suite: RealMLP, ExtraTrees, EBM, TabR, xRFM, TabPFN + OOF ensemble synthesis" -ForegroundColor Cyan
    Write-Host "The suite is resumable and family failures are logged without killing other families." -ForegroundColor Cyan
    Write-Host "Progress: powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_remaining_research_progress.ps1" -ForegroundColor Cyan
    $code = Run-Step @("src\v21_remaining_suite.py","run","--config","configs\v21_remaining_suite.json")
    if ($code -ne 0) { throw "remaining suite failed at top level (exit $code)" }
    "[$(Get-Date -Format o)] remaining-suite COMPLETE" | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Asterisk.Play()
    [System.Windows.Forms.MessageBox]::Show("Remaining tabular research suite completed. Review v21 results and candidates. No Kaggle submission was made.", "Kaggle H1N1 research complete", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
}
catch {
    ($_ | Out-String) | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Hand.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Hand.Play()
    [System.Windows.Forms.MessageBox]::Show("Remaining research suite stopped. Check artifacts\v21_remaining_suite_runner.log. Completed family caches are preserved; rerun the same command.", "Kaggle H1N1 suite stopped", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    exit 1
}
