$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
& $Py src\v21_remaining_suite.py status --config configs\v21_remaining_suite.json
Write-Host "`nRecent log:" -ForegroundColor Cyan
Get-Content .\artifacts\v21_remaining_suite_runner.log -Tail 45 -ErrorAction SilentlyContinue
