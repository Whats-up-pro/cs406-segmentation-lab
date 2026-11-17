$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (Test-Path -LiteralPath $projectPython) {
    & $projectPython -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
} else {
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
}
