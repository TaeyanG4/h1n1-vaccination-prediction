$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

function Show-RunAlert {
    param([string]$Message, [string]$Title, [bool]$Success)
    Add-Type -AssemblyName System.Windows.Forms
    if ($Success) {
        [System.Media.SystemSounds]::Asterisk.Play()
        Start-Sleep -Milliseconds 250
        [System.Media.SystemSounds]::Asterisk.Play()
        $Icon = [System.Windows.Forms.MessageBoxIcon]::Information
    }
    else {
        [System.Media.SystemSounds]::Hand.Play()
        Start-Sleep -Milliseconds 250
        [System.Media.SystemSounds]::Hand.Play()
        $Icon = [System.Windows.Forms.MessageBoxIcon]::Error
    }
    [System.Windows.Forms.MessageBox]::Show($Message, $Title, [System.Windows.Forms.MessageBoxButtons]::OK, $Icon) | Out-Null
}

try {
    $Py = Join-Path $Root ".venv-autogluon\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Py)) {
        throw ".venv-autogluon is missing. Run the existing v5 AutoGluon setup first."
    }

    Write-Host "Checking v11 full-stack contract and folds..." -ForegroundColor Cyan
    & $Py src\v11_v5_fullstack.py check --config configs\v11_v5_fullstack.json
    if ($LASTEXITCODE -ne 0) { throw "v11 preflight failed" }

    $ArgsList = @("src\v11_v5_fullstack.py", "run", "--config", "configs\v11_v5_fullstack.json")
    if ((Test-Path -LiteralPath "artifacts\v11_v5_fullstack\predictor\predictor.pkl") -and (-not (Test-Path -LiteralPath "artifacts\v11_v5_fullstack\results.json"))) {
        $ArgsList += "--resume"
    }

    $StdoutLog = Join-Path $Root "artifacts\v11_v5_fullstack_stdout.log"
    $StderrLog = Join-Path $Root "artifacts\v11_v5_fullstack_stderr.log"
    Write-Host "Starting v11 full-data AutoGluon stack..." -ForegroundColor Cyan
    Write-Host "This rebuilds L1 bagging + L2 CatBoost using all 42,154 labels."
    Write-Host "stdout: $StdoutLog"
    Write-Host "stderr: $StderrLog"
    Write-Host "AutoGluon normally writes status messages to stderr; exit code determines success."

    $Process = Start-Process -FilePath $Py -ArgumentList $ArgsList -WorkingDirectory $Root -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
    if ($Process.ExitCode -ne 0) { throw "v11 full-stack run failed with exit code $($Process.ExitCode)" }

    Show-RunAlert -Message "H1N1 v11 full-stack completed. Check artifacts\v11_v5_fullstack\results.json and the generated submission candidate." -Title "Kaggle H1N1 - v11 completed" -Success $true
    exit 0
}
catch {
    $Message = $_.Exception.Message
    Write-Error $Message
    Show-RunAlert -Message "H1N1 v11 full-stack failed: $Message" -Title "Kaggle H1N1 - v11 failed" -Success $false
    exit 1
}
