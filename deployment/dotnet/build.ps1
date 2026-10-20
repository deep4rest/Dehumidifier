[CmdletBinding()]
param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$Output = "",
    [string]$DotnetPath = "dotnet",
    [string]$PackageCache = "",
    [switch]$SkipTests,
    [switch]$NoRestore
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path $Root).Path
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $Root ("results\deployment\" + (Get-Date -Format "yyyyMMdd_HHmmss") + "_windows_" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
}
$Output = [IO.Path]::GetFullPath($Output)
$cache = if ([string]::IsNullOrWhiteSpace($PackageCache)) { Join-Path $Output "nuget-cache" } else { [IO.Path]::GetFullPath($PackageCache) }
$verify = Join-Path $Output "verify"
$winforms = Join-Path $Output "winforms"
$nuget = Join-Path $Output "nuget"

if (Test-Path $Output) { throw "拒绝覆盖已有构建目录: $Output" }
New-Item -ItemType Directory -Force -Path $Output, $cache, $verify, $winforms, $nuget | Out-Null
$env:NUGET_PACKAGES = $cache
$env:NUGET_HTTP_CACHE_PATH = Join-Path $Output "http-cache"

function Invoke-Dotnet([string[]]$Arguments) {
    Write-Host ("dotnet " + ($Arguments -join " "))
    # 显式进程调用兼容从 WSL 启动的 Windows PowerShell，并检查实际退出码。
    $buildStdoutPath = Join-Path $Output "dotnet-last.stdout.log"
    $buildStderrPath = Join-Path $Output "dotnet-last.stderr.log"
    $buildQuotedArguments = $Arguments | ForEach-Object { '"' + $_ + '"' }
    $buildProcess = Start-Process -FilePath $DotnetPath -ArgumentList $buildQuotedArguments -WorkingDirectory $Root -PassThru -NoNewWindow -RedirectStandardOutput $buildStdoutPath -RedirectStandardError $buildStderrPath
    $buildProcess.Handle | Out-Null
    $buildProcess.WaitForExit()
    if (Test-Path $buildStdoutPath) { Get-Content -Encoding UTF8 $buildStdoutPath | ForEach-Object { Write-Host $_ } }
    if (Test-Path $buildStderrPath) { Get-Content -Encoding UTF8 $buildStderrPath | ForEach-Object { Write-Host $_ } }
    if ($buildProcess.ExitCode -ne 0) { throw "dotnet command failed with exit code $($buildProcess.ExitCode)" }
}

Push-Location $Root
try {
    $simulation = Join-Path $Root "deployment\dotnet\src\Dehumidifier.Simulation\Dehumidifier.Simulation.csproj"
    $verifier = Join-Path $Root "deployment\dotnet\tools\Dehumidifier.Verify\Dehumidifier.Verify.csproj"
    $forms = Join-Path $Root "deployment\dotnet\samples\WinFormsDemo\WinFormsDemo.csproj"
    $tests = Join-Path $Root "deployment\dotnet\tests\Dehumidifier.Tests\Dehumidifier.Tests.csproj"
    if (-not $NoRestore) {
        Invoke-Dotnet @("restore", $simulation, "--packages", $cache)
        Invoke-Dotnet @("restore", $verifier, "-r", "win-x64", "--packages", $cache)
        Invoke-Dotnet @("restore", $forms, "-r", "win-x64", "--packages", $cache, "-p:SelfContained=true")
        if (-not $SkipTests) { Invoke-Dotnet @("restore", $tests, "--packages", $cache) }
    }
    Invoke-Dotnet @("build", $simulation, "-c", "Release", "--no-restore")
    if (-not $SkipTests) {
        Invoke-Dotnet @("test", $tests, "-c", "Release", "--no-restore")
    }
    Invoke-Dotnet @("publish", $verifier, "-c", "Release", "-r", "win-x64", "--self-contained", "true", "--no-restore", "-o", $verify)
    Invoke-Dotnet @("publish", $forms, "-c", "Release", "-r", "win-x64", "--self-contained", "true", "--no-restore", "-o", $winforms)
    Invoke-Dotnet @("pack", $simulation, "-c", "Release", "--no-build", "--no-restore", "-o", $nuget)
    Write-Host "Windows build completed: $Output"
}
finally { Pop-Location }
