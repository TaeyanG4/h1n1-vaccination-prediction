$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
$Log = Join-Path $Root "artifacts\v22_final_stack_gating_runner.log"
$Stdout = Join-Path $Root "artifacts\v22_final_stack_gating_stdout.log"
$Stderr = Join-Path $Root "artifacts\v22_final_stack_gating_stderr.log"
try {
    "[$(Get-Date -Format o)] v22 final stack START" | Add-Content $Log
    Remove-Item $Stdout,$Stderr -Force -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $Py `
        -ArgumentList @("src\v22_final_stack_gating.py") `
        -WorkingDirectory $Root -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    if (Test-Path $Stdout) { Get-Content $Stdout | Add-Content $Log }
    if (Test-Path $Stderr) { Get-Content $Stderr | Add-Content $Log }
    if ($p.ExitCode -ne 0) { throw "v22 exited $($p.ExitCode)" }
    "[$(Get-Date -Format o)] v22 final stack COMPLETE" | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Asterisk.Play()
    [System.Windows.Forms.MessageBox]::Show("v22 final logistic stack + slice-aware gating completed. No Kaggle submission was made.","Kaggle H1N1 v22 complete",[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Information)|Out-Null
}
catch {
    ($_|Out-String)|Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Hand.Play()
    [System.Windows.Forms.MessageBox]::Show("v22 failed. Check artifacts\v22_final_stack_gating_runner.log.","Kaggle H1N1 v22 failed",[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Error)|Out-Null
    exit 1
}
