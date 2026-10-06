param(
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$AgPy = Join-Path $Root ".venv-autogluon\Scripts\python.exe"
$Py = (Get-Command python).Source
$LogDir = Join-Path $Root "artifacts\overnight_logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Status = @()
$StatusPath = Join-Path $Root "reports\overnight_run_status.json"
$CurrentPath = Join-Path $Root "reports\overnight_current.json"

function Save-Status {
    $script:Status | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Run-Step {
    param([string]$Name, [string]$Exe, [string[]]$StepArgs)
    $Out = Join-Path $LogDir ($Name + "_stdout.log")
    $Err = Join-Path $LogDir ($Name + "_stderr.log")
    Write-Host "[$(Get-Date -Format s)] START $Name" -ForegroundColor Cyan
    $Start = Get-Date
    [pscustomobject]@{
        name = $Name
        status = "RUNNING"
        started = $Start.ToString("o")
        stdout = $Out
        stderr = $Err
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $CurrentPath -Encoding UTF8
    try {
        # The outer PowerShell is already detached. Invoke Python directly here
        # to avoid nested Start-Process argument quoting issues on paths with spaces.
        & $Exe @StepArgs 1> $Out 2> $Err
        $Code = $LASTEXITCODE
        if ($null -eq $Code) {
            $Code = if ($?) { 0 } else { 1 }
        }
    }
    catch {
        $Code = 999
        ($_ | Out-String) | Set-Content -LiteralPath $Err -Encoding UTF8
    }
    $Sec = [int]((Get-Date) - $Start).TotalSeconds
    $State = if ($Code -eq 0) { "OK" } else { "FAILED" }
    $script:Status += [pscustomobject]@{name=$Name; status=$State; exit_code=$Code; seconds=$Sec; stdout=$Out; stderr=$Err}
    Save-Status
    [pscustomobject]@{
        name = $Name
        status = $State
        exit_code = $Code
        seconds = $Sec
        completed = (Get-Date).ToString("o")
        stdout = $Out
        stderr = $Err
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $CurrentPath -Encoding UTF8
    Write-Host "[$(Get-Date -Format s)] END $Name status=$State seconds=$Sec" -ForegroundColor $(if ($Code -eq 0) { "Green" } else { "Yellow" })
}

if (-not (Test-Path -LiteralPath $AgPy)) {
    throw "Missing .venv-autogluon Python: $AgPy"
}

# 1) Exact v5 parent reproduction: exact 44 L1 bases + only CatBoost_BAG_L2, full 42,154 labels.
Run-Step "v12_check" $AgPy @("src\v12_exact_v5_fullstack.py","check","--config","configs\v12_exact_v5_fullstack.json")

if ($PreflightOnly) {
    Run-Step "v14_check" $Py @("src\v14_catboost_feature_hpo.py","check","--config","configs\v14_catboost_feature_hpo.json")
    $Failures = @($Status | Where-Object { $_.status -ne "OK" }).Count
    Write-Host "Preflight-only run finished. failures=$Failures" -ForegroundColor $(if ($Failures -eq 0) { "Green" } else { "Red" })
    exit $(if ($Failures -eq 0) { 0 } else { 1 })
}

Run-Step "v12_exact44_fullstack" $AgPy @("src\v12_exact_v5_fullstack.py","run","--config","configs\v12_exact_v5_fullstack.json")

# 2) Frozen historical-diverse XGB full-data finalization, plus 5-seed stability candidate.
Run-Step "v13_xgb_full" $Py @("src\v13_xgb_full.py","--config","configs\v13_xgb_full.json")

# 3) Compact feature engineering, missing-value ablation, and structural CatBoost HPO.
Run-Step "v14_check" $Py @("src\v14_catboost_feature_hpo.py","check","--config","configs\v14_catboost_feature_hpo.json")
Run-Step "v14_catboost_feature_hpo" $Py @("src\v14_catboost_feature_hpo.py","run","--config","configs\v14_catboost_feature_hpo.json")

# 4) Train/test shift diagnostics.
Run-Step "v15_shift_audit" $Py @("src\v15_shift_audit.py")

# 5) Simple ensembles only after the individual models exist. No optimized blending.
Run-Step "v16_ensemble_scan" $Py @("src\v16_ensemble_scan.py")

# Consolidated morning summary.
Run-Step "overnight_summary" $Py @("src\overnight_summary.py")
Save-Status

Add-Type -AssemblyName System.Windows.Forms
$Failures = @($Status | Where-Object { $_.status -ne "OK" }).Count
if ($Failures -eq 0) {
    [System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Asterisk.Play()
    [System.Windows.Forms.MessageBox]::Show("Overnight H1N1 campaign completed successfully. Check reports\overnight_summary.json.", "Kaggle H1N1 overnight complete", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
}
else {
    [System.Media.SystemSounds]::Exclamation.Play(); Start-Sleep -Milliseconds 300; [System.Media.SystemSounds]::Exclamation.Play()
    [System.Windows.Forms.MessageBox]::Show("Overnight campaign finished with $Failures failed step(s). Other steps continued. Check reports\overnight_run_status.json and artifacts\overnight_logs.", "Kaggle H1N1 overnight finished", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Warning) | Out-Null
}
