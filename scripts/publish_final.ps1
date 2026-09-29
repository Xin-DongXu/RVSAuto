# Run once after: gh auth login
# Creates https://github.com/Xin-DongXu/RVSAuto and pushes main + tag v11.0.0

$ErrorActionPreference = "Stop"
$env:Path = "C:\Program Files\Git\bin;C:\Program Files\GitHub CLI;" + $env:Path
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

gh auth status
if (-not (gh repo view Xin-DongXu/RVSAuto 2>$null)) {
    gh repo create Xin-DongXu/RVSAuto --public --description "Batch UniDock virtual screening and self-redocking (RVSAuto)" --source=. --remote=origin --push
} else {
    git push -u origin main
    git push origin v11.0.0
}

Write-Host ""
Write-Host "Create GitHub Release: https://github.com/Xin-DongXu/RVSAuto/releases/new?tag=v11.0.0"
Write-Host "Paste release notes from CHANGELOG.md section [11.0.0]"
