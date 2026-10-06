$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

function Show-RunAlert {
    param([string]$Message, [string]$Title, [bool]$Success)
    Add-Type -AssemblyName System.Windows.Forms
    if ($Success) {
        [System.Media.SystemSounds]::Asterisk.Play()
        $Icon = [System.Windows.Forms.MessageBoxIcon]::Information
    } else {
        [System.Media.SystemSounds]::Hand.Play()
        $Icon = [System.Windows.Forms.MessageBoxIcon]::Error
    }
    [System.Windows.Forms.MessageBox]::Show($Message, $Title, [System.Windows.Forms.MessageBoxButtons]::OK, $Icon) | Out-Null
}

try {
    $Py = Join-Path $Root ".venv-autogluon\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Py)) { throw ".venv-autogluon is missing" }

    Write-Host "Checking v7 parent stack, folds, and protected inputs..." -ForegroundColor Cyan
    & $Py src\v7_catboost_hpo.py check --config configs\v7_catboost_hpo.json
    if ($LASTEXITCODE -ne 0) { throw "v7 preflight failed" }

    $StdoutLog = Join-Path $Root "artifacts\v7_catboost_hpo_stdout.log"
    $StderrLog = Join-Path $Root "artifacts\v7_catboost_hpo_stderr.log"
    $ArgsList = @("src\v7_catboost_hpo.py", "run", "--config", "configs\v7_catboost_hpo.json", "--resume")
    Write-Host "Starting v7 controlled CatBoost L2 HPO..." -ForegroundColor Cyan
    Write-Host "stdout: $StdoutLog"
    Write-Host "stderr: $StderrLog"
    $Process = Start-Process -FilePath $Py -ArgumentList $ArgsList -WorkingDirectory $Root -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
    if ($Process.ExitCode -ne 0) { throw "v7 HPO failed with exit code $($Process.ExitCode)" }

    Show-RunAlert -Message "H1N1 v7 CatBoost L2 HPO completed. Check artifacts\v7_catboost_l2_hpo\results.json and submissions\v7_catboost_l2_hpo_full_label.csv." -Title "Kaggle H1N1 - v7 completed" -Success $true
    exit 0
}
catch {
    $Message = $_.Exception.Message
    Write-Error $Message
    Show-RunAlert -Message "H1N1 v7 failed: $Message" -Title "Kaggle H1N1 - v7 failed" -Success $false
    exit 1
}
