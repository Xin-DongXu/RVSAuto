# Run once after: gh auth login
# Creates https://github.com/Xin-DongXu/RVSAuto and pushes main + tag v11.0.0

$ErrorActionPreference = "Stop"
$env:Path = "C:\Program Files\Git\bin;C:\Program Files\GitHub CLI;" + $env:Path
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

gh auth status

$repoExists = $false
gh repo view Xin-DongXu/RVSAuto *> $null
if ($LASTEXITCODE -eq 0) {
    $repoExists = $true
}

if (-not $repoExists) {
    Write-Host "Creating public repo Xin-DongXu/RVSAuto ..."
    gh repo create Xin-DongXu/RVSAuto --public `
        --description "Batch UniDock virtual screening and self-redocking (RVSAuto)"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (-not (git remote get-url origin 2>$null)) {
        git remote add origin https://github.com/Xin-DongXu/RVSAuto.git
    }
    git push -u origin main
    git push origin v11.0.0
} else {
    Write-Host "Repo exists; pushing main and tag ..."
    git push -u origin main
    git push origin v11.0.0
}

Write-Host ""
Write-Host "Repository: https://github.com/Xin-DongXu/RVSAuto"
Write-Host "Create Release: https://github.com/Xin-DongXu/RVSAuto/releases/new?tag=v11.0.0"
