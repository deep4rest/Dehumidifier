# 在解压后的 SDK 根目录运行；Windows x64、Python 3.14 x64。
param([string]$Python = "py")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "命令失败（退出码 $LASTEXITCODE）：$Exe" }
}
if (Test-Path "windows-dist") { throw "windows-dist 已存在，请先保存旧构建并使用新的解压目录。" }
if ($Python -eq "py") {
    Invoke-Checked $Python @("-3.11", "-m", "venv", ".build-venv")
} else {
    Invoke-Checked $Python @("-m", "venv", ".build-venv")
}
$BuildPython = Join-Path $PSScriptRoot ".build-venv\Scripts\python.exe"
Invoke-Checked $BuildPython @("-c", "import struct,sys; assert struct.calcsize('P') == 8; assert sys.version_info >= (3,11)")
Invoke-Checked $BuildPython @("-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.13.0")
Invoke-Checked $BuildPython @("-m", "pip", "install", "-r", "requirements-runtime.txt", "pyinstaller>=6.16,<7")
Invoke-Checked $BuildPython @("verify_sdk.py")
Invoke-Checked $BuildPython @("-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--console", "--name", "DehumidifierEngine", "--distpath", "windows-dist", "--paths", "simulation", "--collect-submodules", "dual_rotor_sim", "--collect-all", "sklearn", "--collect-all", "matplotlib", "--collect-all", "torch", "engine.py")
Copy-Item "artifacts" "windows-dist\DehumidifierEngine\artifacts" -Recurse
Invoke-Checked $BuildPython @("verify_sdk.py", "--engine", "windows-dist\DehumidifierEngine\DehumidifierEngine.exe")
& $BuildPython -m pip freeze | Set-Content "windows-dist\dependencies-resolved.txt" -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw "依赖留档失败" }
Write-Host "构建与数值校验完成。交付 windows-dist\DehumidifierEngine 整个目录，不能只复制 EXE。"
