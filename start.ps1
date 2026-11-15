$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = $null
foreach ($candidate in @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "C:\Python312\python.exe"
)) {
    if (Test-Path $candidate) { $python = $candidate; break }
}
if (-not $python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notmatch "WindowsApps") { $python = $cmd.Source }
}
if (-not $python) {
    Write-Host "Python is not installed. Install Python 3.12+ from https://www.python.org/downloads/ and rerun."
    exit 1
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    & $python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path ".\models\genre_model.joblib")) {
    & .\.venv\Scripts\python.exe src\train.py
}

& .\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true
