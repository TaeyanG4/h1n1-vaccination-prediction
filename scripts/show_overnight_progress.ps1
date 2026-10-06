$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

$Current = Join-Path $Root "reports\overnight_current.json"
$Status = Join-Path $Root "reports\overnight_run_status.json"
$LogDir = Join-Path $Root "artifacts\overnight_logs"

Write-Host "=== Overnight campaign progress ===" -ForegroundColor Cyan
if (Test-Path -LiteralPath $Current) {
    Write-Host "Current/latest step:" -ForegroundColor Yellow
    Get-Content -LiteralPath $Current
}
else {
    Write-Host "No current-step record yet."
}

if (Test-Path -LiteralPath $Status) {
    Write-Host "`nCompleted steps:" -ForegroundColor Yellow
    $s = Get-Content -LiteralPath $Status -Raw | ConvertFrom-Json
    @($s) | Select-Object name,status,exit_code,seconds | Format-Table -AutoSize
}

$Processes = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'v12_exact_v5_fullstack|v13_xgb_full|v14_catboost_feature_hpo|v15_shift_audit|v16_ensemble_scan|run_overnight_campaign'
}
Write-Host "Active matching processes: $(@($Processes).Count)" -ForegroundColor Yellow
$Processes | Select-Object ProcessId,Name,CommandLine | Format-Table -Wrap

if (Test-Path -LiteralPath $Current) {
    try {
        $c = Get-Content -LiteralPath $Current -Raw | ConvertFrom-Json
        if ($c.status -eq "RUNNING" -and (Test-Path -LiteralPath $c.stderr)) {
            Write-Host "`nLatest stderr lines:" -ForegroundColor Yellow
            Get-Content -LiteralPath $c.stderr -Tail 20
        }
        elseif ($c.status -eq "RUNNING" -and (Test-Path -LiteralPath $c.stdout)) {
            Write-Host "`nLatest stdout lines:" -ForegroundColor Yellow
            Get-Content -LiteralPath $c.stdout -Tail 20
        }
    }
    catch {
        Write-Host "Could not parse current-step record: $($_.Exception.Message)" -ForegroundColor Red
    }
}
