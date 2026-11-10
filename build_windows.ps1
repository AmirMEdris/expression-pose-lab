$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& .venv/Scripts/python.exe -m PyInstaller --noconfirm ExpressionLab.spec
if ($LASTEXITCODE -ne 0) { throw 'Executable build failed' }
Write-Host 'Built dist/Expression Lab/Expression Lab.exe. Distribute the whole folder.'
