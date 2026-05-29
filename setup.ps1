# 初回セットアップ: venv 作成 + 依存パッケージインストール
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$Candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)

$Py = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Py) {
    Write-Host "Python が見つかりません。winget install Python.Python.3.12 を実行してください。" -ForegroundColor Red
    exit 1
}

Write-Host "使用する Python: $Py"
& $Py -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
Write-Host "セットアップ完了。run.ps1 または run.bat で起動できます。" -ForegroundColor Green
