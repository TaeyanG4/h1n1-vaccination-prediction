$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py)) {
    Write-Host "v17 environment has not been created yet." -ForegroundColor Yellow
    exit 1
}
& $Py src\v17_xgb_optuna.py status --config configs\v17_xgb_optuna.json
Write-Host "`nRecent runner log:" -ForegroundColor Cyan
Get-Content .\artifacts\v17_xgb_optuna_runner.log -Tail 20 -ErrorAction SilentlyContinue
