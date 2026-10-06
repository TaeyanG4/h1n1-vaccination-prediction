$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv-xgb-optuna\Scripts\python.exe"
& $Py src\v21_focus_confirm.py status --config configs\v21_remaining_suite.json
Write-Host "`nLive stdout:" -ForegroundColor Cyan
Get-Content .\artifacts\v21_focus_confirm_stdout.log -Tail 40 -ErrorAction SilentlyContinue
