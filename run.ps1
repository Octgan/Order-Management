# プロジェクト内 .venv を使って Streamlit を起動します
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "仮想環境が見つかりません。setup.ps1 を先に実行してください。" -ForegroundColor Red
    exit 1
}

Set-Location $Root
& $Python -m streamlit run app.py --server.headless true
