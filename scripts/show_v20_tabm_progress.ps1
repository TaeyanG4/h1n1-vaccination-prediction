$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
& $Py src\v20_tabm.py status --config configs\v20_tabm.json
Write-Host "`nRecent log:" -ForegroundColor Cyan
Get-Content .\artifacts\v20_tabm_runner.log -Tail 35 -ErrorAction SilentlyContinue
