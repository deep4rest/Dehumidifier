[CmdletBinding()]
param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$DotnetPath = "dotnet",
    [string]$PackageCache = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path $Root).Path
$Output = [IO.Path]::GetFullPath($Output)
if (Test-Path $Output) { throw "拒绝覆盖已有v2构建: $Output" }
$cache = if ($PackageCache) { [IO.Path]::GetFullPath($PackageCache) } else { Join-Path $Output "nuget-cache" }
$env:NUGET_PACKAGES = $cache
$env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"

# Retain the documented core build/test, verifier, WinForms and NuGet workflow.
& (Join-Path $PSScriptRoot "build.ps1") -Root $Root -Output $Output -DotnetPath $DotnetPath -PackageCache $cache

function Invoke-V2Dotnet([string[]]$Arguments) {
    Write-Host ("dotnet " + ($Arguments -join " "))
    & $DotnetPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "v2 dotnet失败: $LASTEXITCODE" }
}

Push-Location $Root
try {
    $prediction = Join-Path $PSScriptRoot "src/Dehumidifier.Prediction/Dehumidifier.Prediction.csproj"
    $optimization = Join-Path $PSScriptRoot "src/Dehumidifier.Optimization/Dehumidifier.Optimization.csproj"
    $verifier = Join-Path $PSScriptRoot "tools/Dehumidifier.V2Verify/Dehumidifier.V2Verify.csproj"
    $testRoot = Join-Path $Output "test-results"
    foreach ($test in @("Dehumidifier.Prediction.Tests", "Dehumidifier.Optimization.Tests")) {
        $project = Join-Path $PSScriptRoot ("tests/"+$test+"/"+$test+".csproj")
        Invoke-V2Dotnet @("test", $project, "-c", "Release", "--logger", "trx;LogFileName=$test.trx", "--results-directory", $testRoot)
    }
    foreach ($project in @($prediction,$optimization)) {
        Invoke-V2Dotnet @("pack", $project, "-c", "Release", "-o", (Join-Path $Output "nuget"))
    }
    Invoke-V2Dotnet @("publish", $verifier, "-c", "Release", "-r", "win-x64", "--self-contained", "true", "-o", (Join-Path $Output "v2verify"))
    Write-Host "v2 DLL和独立验证程序已构建: $Output"
}
finally { Pop-Location }
