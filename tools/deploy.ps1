# Push cosyvoice-eval to GitHub and open Streamlit Community Cloud deploy.
# Run from the project root after: gh auth login

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is not installed. Install with: winget install GitHub.cli"
}

gh auth status | Out-Null

$repoName = "cosyvoice-eval"
$status = git status --porcelain
if ($status) {
    Write-Host "Uncommitted changes detected. Commit or stash them first."
    git status
    exit 1
}

if (-not (git remote get-url origin 2>$null)) {
    Write-Host "Creating GitHub repository: $repoName"
    gh repo create $repoName --private --source=. --remote=origin --push
} else {
    Write-Host "Pushing to origin/main"
    git push -u origin main
}

$remote = git remote get-url origin
if ($remote -match "github.com[:/](?<owner>[^/]+)/(?<repo>[^/.]+)") {
    $owner = $Matches.owner
    $repo = $Matches.repo
    $branch = (git branch --show-current)
    $deployUrl = "https://share.streamlit.io/deploy?repository=$owner/$repo&branch=$branch&mainModule=app.py"
    Write-Host ""
    Write-Host "Repository pushed."
    Write-Host "Open Streamlit deploy: $deployUrl"
    Start-Process $deployUrl
} else {
    Write-Host "Pushed. Create the app at https://share.streamlit.io"
}

Write-Host ""
Write-Host "Paste these Streamlit secrets (App settings -> Secrets):"
Get-Content ".streamlit/secrets.toml.example"
