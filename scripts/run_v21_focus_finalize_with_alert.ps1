$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
$Log = Join-Path $Root "artifacts\v21_focus_finalize_runner.log"
$Stdout = Join-Path $Root "artifacts\v21_focus_finalize_stdout.log"
$Stderr = Join-Path $Root "artifacts\v21_focus_finalize_stderr.log"
try {
    "[$(Get-Date -Format o)] v21 focus finalize START" | Add-Content $Log
    Remove-Item $Stdout,$Stderr -Force -ErrorAction SilentlyContinue
    $p=Start-Process -FilePath $Py -ArgumentList @("src\v21_focus_finalize.py") -WorkingDirectory $Root -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    if(Test-Path $Stdout){Get-Content $Stdout|Add-Content $Log}; if(Test-Path $Stderr){Get-Content $Stderr|Add-Content $Log}
    if($p.ExitCode -ne 0){throw "v21 focus finalize failed (exit $($p.ExitCode))"}
    "[$(Get-Date -Format o)] v21 focus finalize COMPLETE" | Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Asterisk.Play()
    [System.Windows.Forms.MessageBox]::Show("v21 EBM/RealMLP full-refit candidates and dry-runs are ready. No Kaggle submission was made.","Kaggle H1N1 v21 candidates ready",[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Information)|Out-Null
}
catch {
    ($_|Out-String)|Add-Content $Log
    Add-Type -AssemblyName System.Windows.Forms
    [System.Media.SystemSounds]::Hand.Play()
    [System.Windows.Forms.MessageBox]::Show("v21 focus finalization failed. Check artifacts\v21_focus_finalize_runner.log.","Kaggle H1N1 v21 finalize failed",[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Error)|Out-Null
    exit 1
}
