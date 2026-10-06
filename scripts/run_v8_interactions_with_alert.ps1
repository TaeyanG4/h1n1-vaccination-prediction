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
    Write-Host "Checking v8 feature schemas and frozen split..." -ForegroundColor Cyan
    & $Py src\v8_interactions.py check --config configs\v8_interactions.json
    if ($LASTEXITCODE -ne 0) { throw "v8 preflight failed" }

    if (-not (Test-Path -LiteralPath "reports\v8_interactions_smoke.json")) {
        Write-Host "Running short v8 smoke test..." -ForegroundColor Cyan
        & $Py src\v8_interactions.py smoke --config configs\v8_interactions.json
        if ($LASTEXITCODE -ne 0) { throw "v8 smoke failed" }
    }

    $StdoutLog = Join-Path $Root "artifacts\v8_interactions_stdout.log"
    $StderrLog = Join-Path $Root "artifacts\v8_interactions_stderr.log"
    $ArgsList = @("src\v8_interactions.py", "run", "--config", "configs\v8_interactions.json", "--resume")
    Write-Host "Starting v8 interaction experiment..." -ForegroundColor Cyan
    Write-Host "stdout: $StdoutLog"
    Write-Host "stderr: $StderrLog"
    $Process = Start-Process -FilePath $Py -ArgumentList $ArgsList -WorkingDirectory $Root -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
    if ($Process.ExitCode -ne 0) { throw "v8 run failed with exit code $($Process.ExitCode)" }

    Show-RunAlert -Message "H1N1 v8 interaction experiment completed. Check artifacts\v8_interactions\results.json." -Title "Kaggle H1N1 - v8 completed" -Success $true
    exit 0
}
catch {
    $Message = $_.Exception.Message
    Write-Error $Message
    Show-RunAlert -Message "H1N1 v8 failed: $Message" -Title "Kaggle H1N1 - v8 failed" -Success $false
    exit 1
}
