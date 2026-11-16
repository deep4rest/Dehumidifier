[CmdletBinding()]
param([ValidateSet('forward','search')][string]$Mode='forward')
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$ErrorActionPreference='Stop'
try {
    $exe=Join-Path $PSScriptRoot 'verify/Dehumidifier.V2Verify.exe'
    $models=Join-Path $PSScriptRoot 'models'
    $config=Join-Path $PSScriptRoot 'examples/qy1070_case.json'
    $results=Join-Path $PSScriptRoot 'results'
    New-Item -ItemType Directory -Force -Path $results | Out-Null
    $summary=Join-Path $results ($Mode+'_latest.json')
    $arguments=@('--mode',$Mode,'--models',$models,'--config',$config,'--results-dir',$results,'--summary-output',$summary)
    if($Mode -eq 'search') {$arguments+=@('--dew-point-target','-45','--supply-temp-target','-35')}
    & $exe @arguments | Out-Null
    if($LASTEXITCODE -ne 0) {throw "计算未成功，退出码 $LASTEXITCODE。请查看上方错误和 results。"}
    $record=Get-Content -Raw -Encoding UTF8 -LiteralPath $summary | ConvertFrom-Json
    if($Mode -eq 'search') {
        if(!$record.feasible -or !$record.best_independently_verified) {throw '没有通过独立复算的方案'}
        $forwardSummary=Join-Path $results 'replay_latest.json'
        & $exe --mode forward --models $models --config $record.best_config_path --results-dir $results --summary-output $forwardSummary | Out-Null
        if($LASTEXITCODE -ne 0) {throw "前向复算失败，退出码 $LASTEXITCODE"}
        $prediction=(Get-Content -Raw -Encoding UTF8 -LiteralPath $forwardSummary | ConvertFrom-Json).prediction
        if(!$prediction.converged -or
            [math]::Abs($prediction.supply_Tdp_degC-$record.best_assessment.SupplyDewPointC) -gt 1e-9 -or
            [math]::Abs($prediction.supply_T_degC-$record.best_assessment.SupplyTemperatureC) -gt 1e-9 -or
            [math]::Abs($prediction.supply_W_g_per_kg-$record.best_assessment.SupplyHumidityGPerKg) -gt 1e-9 -or
            [math]::Abs($prediction.thermal_load_sum_kW-$record.best_assessment.ThermalLoadSumKW) -gt 1e-9) {throw '搜索与前向复算不一致'}
        Write-Host ('搜索露点: {0:F6} ℃；前向复算露点: {1:F6} ℃；一致。' -f $record.best_assessment.SupplyDewPointC,$prediction.supply_Tdp_degC)
        Write-Host ('最佳完整配置: '+$record.best_config_path)
    } else {$prediction=$record.prediction}
    Write-Host ('送风温度: {0:F6} ℃；露点: {1} ℃；冷热总负荷: {2:F6} kW' -f $prediction.supply_T_degC,$prediction.supply_Tdp_degC,$prediction.thermal_load_sum_kW)
    Write-Host ('完整结果: '+$prediction.result_json)
    Write-Host ('状态摘要: '+$summary)
    exit 0
} catch {Write-Host ('运行失败: '+$_.Exception.Message);exit 1}
