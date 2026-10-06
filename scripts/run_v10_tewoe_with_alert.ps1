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
    $Py = "python"
    Write-Host "Checking v10 compact TE+WOE contract..." -ForegroundColor Cyan
    & $Py src\v10_tewoe.py check --config configs\v10_tewoe.json
    if ($LASTEXITCODE -ne 0) { throw "v10 preflight failed" }

    if (-not (Test-Path -LiteralPath "reports\v10_tewoe_smoke.json")) {
        Write-Host "Running short v10 XGB/LGB smoke..." -ForegroundColor Cyan
        & $Py src\v10_tewoe.py smoke --config configs\v10_tewoe.json
        if ($LASTEXITCODE -ne 0) { throw "v10 smoke failed" }
    }

    $StdoutLog = Join-Path $Root "artifacts\v10_tewoe_stdout.log"
    $StderrLog = Join-Path $Root "artifacts\v10_tewoe_stderr.log"
    $ArgsList = @("src\v10_tewoe.py", "run", "--config", "configs\v10_tewoe.json", "--resume")
    Write-Host "Starting v10 compact TE+WOE XGB/LGBM experiment..." -ForegroundColor Cyan
    Write-Host "stdout: $StdoutLog"
    Write-Host "stderr: $StderrLog"
    $Process = Start-Process -FilePath $Py -ArgumentList $ArgsList -WorkingDirectory $Root -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
    if ($Process.ExitCode -ne 0) { throw "v10 run failed with exit code $($Process.ExitCode)" }

    Show-RunAlert -Message "H1N1 v10 TE+WOE experiment completed. Check artifacts\v10_tewoe\comparison.csv and results.json." -Title "Kaggle H1N1 - v10 completed" -Success $true
    exit 0
}
catch {
    $Message = $_.Exception.Message
    Write-Error $Message
    Show-RunAlert -Message "H1N1 v10 failed: $Message" -Title "Kaggle H1N1 - v10 failed" -Success $false
    exit 1
}
