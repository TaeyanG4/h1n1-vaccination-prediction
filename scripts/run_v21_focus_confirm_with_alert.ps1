$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
$Log = Join-Path $Root "artifacts\v21_focus_confirm_runner.log"
$Stdout = Join-Path $Root "artifacts\v21_focus_confirm_stdout.log"
$Stderr = Join-Path $Root "artifacts\v21_focus_confirm_stderr.log"

try {
    "[$(Get-Date -Format o)] v21 focus START" | Add-Content $Log
    Remove-Item $Stdout,$Stderr -Force -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $Py `
        -ArgumentList @("src\v21_focus_confirm.py","run","--config","configs\v21_remaining_suite.json") `
        -WorkingDirectory $Root -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    if (Test-Path $Stdout) { Get-Content $Stdout | Add-Content $Log }
    if (Test-Path $Stderr) { Get-Content $Stderr | Add-Content $Log }
    if ($p.ExitCode -ne 0) { throw "v21 focus confirm failed (exit $($p.ExitCode))" }
    "[$(Get-Date -Format o)] v21 focus COMPLETE" | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Asterisk.Play()
    [System.Windows.Forms.MessageBox]::Show("EBM + RealMLP-TD grouped 5-fold confirmation completed. No full refit or Kaggle submission was made.", "Kaggle H1N1 v21 focus complete", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
}
catch {
    ($_ | Out-String) | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Hand.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Hand.Play()
    [System.Windows.Forms.MessageBox]::Show("v21 focus confirmation failed. Check artifacts\v21_focus_confirm_runner.log.", "Kaggle H1N1 v21 focus failed", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    exit 1
}
