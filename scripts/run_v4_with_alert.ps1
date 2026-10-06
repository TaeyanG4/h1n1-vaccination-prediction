$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

$ArgsList = @("src\v4_pipeline.py", "develop")
if (Test-Path -LiteralPath "artifacts\v4\provenance.json") {
    $ArgsList += "--resume"
}

Write-Host "Starting H1N1 v4 XGBoost one-hot experiment..." -ForegroundColor Cyan
& python @ArgsList
$Code = $LASTEXITCODE

Add-Type -AssemblyName System.Windows.Forms
if ($Code -eq 0) {
    [System.Media.SystemSounds]::Asterisk.Play()
    [System.Windows.Forms.MessageBox]::Show(
        "H1N1 v4 experiment completed. Check artifacts\v4\comparison.csv and results.json.",
        "Kaggle H1N1 - Completed",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Information
    ) | Out-Null
}
else {
    [System.Media.SystemSounds]::Hand.Play()
    [System.Windows.Forms.MessageBox]::Show(
        "H1N1 v4 experiment failed with exit code $Code. Check the PowerShell output.",
        "Kaggle H1N1 - Failed",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
    ) | Out-Null
}

exit $Code
