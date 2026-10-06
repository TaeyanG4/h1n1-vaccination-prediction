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
    Write-Host "Checking v9 locked selection / confirmation contract..." -ForegroundColor Cyan
    & $Py src\v9_context_confirm.py check --config configs\v9_context_confirm.json
    if ($LASTEXITCODE -ne 0) { throw "v9 preflight failed" }

    $StdoutLog = Join-Path $Root "artifacts\v9_context_confirm_stdout.log"
    $StderrLog = Join-Path $Root "artifacts\v9_context_confirm_stderr.log"
    $ArgsList = @("src\v9_context_confirm.py", "run", "--config", "configs\v9_context_confirm.json", "--resume")
    Write-Host "Starting v9 context confirmation on untouched folds 2/3/4..." -ForegroundColor Cyan
    Write-Host "stdout: $StdoutLog"
    Write-Host "stderr: $StderrLog"
    $Process = Start-Process -FilePath $Py -ArgumentList $ArgsList -WorkingDirectory $Root -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
    if ($Process.ExitCode -ne 0) { throw "v9 run failed with exit code $($Process.ExitCode)" }

    $ResultPath = Join-Path $Root "artifacts\v9_context_confirm\results.json"
    $Passed = $false
    if (Test-Path -LiteralPath $ResultPath) {
        $Result = Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json
        $Passed = [bool]$Result.confirmation_surface.passed
    }
    if ($Passed) {
        Show-RunAlert -Message "H1N1 v9 context confirmation PASSED. A local submission candidate was created; check artifacts\v9_context_confirm\results.json." -Title "Kaggle H1N1 - v9 passed" -Success $true
    } else {
        Show-RunAlert -Message "H1N1 v9 completed, but the context confirmation gate did not pass. Check artifacts\v9_context_confirm\results.json." -Title "Kaggle H1N1 - v9 completed" -Success $true
    }
    exit 0
}
catch {
    $Message = $_.Exception.Message
    Write-Error $Message
    Show-RunAlert -Message "H1N1 v9 failed: $Message" -Title "Kaggle H1N1 - v9 failed" -Success $false
    exit 1
}
