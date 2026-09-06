$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$logDirectory = Join-Path $projectRoot "artifacts\\logs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

foreach ($seed in 0..4) {
    $checkpoint = "artifacts\\checkpoints\\rank_max_20ep_seed_$seed.pt"
    $logFile = Join-Path $logDirectory "rank_max_20ep_seed_$seed.log"
    & .\.venv\Scripts\python.exe scripts\run_experiment.py `
        --pooling rank_max `
        --seed $seed `
        --epochs 20 `
        --data-root data_rankmax `
        --checkpoint $checkpoint *>&1 | Tee-Object -FilePath $logFile

    if ($LASTEXITCODE -ne 0) {
        throw "RankMax experiment for seed $seed failed. See $logFile"
    }
}
