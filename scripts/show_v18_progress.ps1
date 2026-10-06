$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
& $Py src\v18_xgb_finalize_ensemble.py status --config configs\v18_xgb_finalize_ensemble.json
Write-Host "`nRecent log:" -ForegroundColor Cyan
Get-Content .\artifacts\v18_xgb_finalize_ensemble_runner.log -Tail 25 -ErrorAction SilentlyContinue
