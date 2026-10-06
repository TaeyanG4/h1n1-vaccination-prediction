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
    $StdoutLog = Join-Path $Root "artifacts\v9_context_full_stdout.log"
    $StderrLog = Join-Path $Root "artifacts\v9_context_full_stderr.log"
    Write-Host "Training frozen v9 context recipe on all 42,154 labels..." -ForegroundColor Cyan
    $Process = Start-Process -FilePath "python" -ArgumentList @("src\v9_context_finalize.py") -WorkingDirectory $Root -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
    if ($Process.ExitCode -ne 0) { throw "v9 full-data refit failed with exit code $($Process.ExitCode)" }
    Show-RunAlert -Message "H1N1 v9 full-data refit completed. Check artifacts\v9_context_full\results.json." -Title "Kaggle H1N1 - v9 full complete" -Success $true
    exit 0
}
catch {
    $Message = $_.Exception.Message
    Write-Error $Message
    Show-RunAlert -Message "H1N1 v9 full-data refit failed: $Message" -Title "Kaggle H1N1 - v9 full failed" -Success $false
    exit 1
}
