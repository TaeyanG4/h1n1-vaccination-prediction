$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
& $Py src\v19_lightgbm.py status --config configs\v19_lightgbm.json
Write-Host "`nRecent log:" -ForegroundColor Cyan
Get-Content .\artifacts\v19_lightgbm_runner.log -Tail 35 -ErrorAction SilentlyContinue
