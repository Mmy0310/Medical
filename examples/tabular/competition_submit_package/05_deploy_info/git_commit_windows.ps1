param(
    [string]$RepoRoot = ".",
    [string]$CommitMessage = "add deploy-ready package and scripts",
    [switch]$IncludeOptional,
    [switch]$Push,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Set-Location $RepoRoot
$repoTop = git rev-parse --show-toplevel 2>$null
if (-not $repoTop) {
    throw "Current directory is not inside a git repository."
}
Set-Location $repoTop

$requiredPaths = @(
    "examples/tabular/competition_submit_package/02_code_folder",
    "examples/tabular/competition_submit_package/05_deploy_info"
)

$optionalPaths = @(
    "examples/tabular/competition_submit_package/03_docs_folder",
    "examples/tabular/competition_submit_package/04_demo_folder"
)

$pathsToAdd = @()
$pathsToAdd += $requiredPaths
if ($IncludeOptional) {
    $pathsToAdd += $optionalPaths
}

Write-Host "Repository root: $repoTop"
Write-Host "Staging paths:"
$pathsToAdd | ForEach-Object { Write-Host "  - $_" }

if ($DryRun) {
    Write-Host "Dry run mode. No git add/commit executed."
    git status --short
    exit 0
}

foreach ($path in $pathsToAdd) {
    if (Test-Path $path) {
        git add -- $path
    } else {
        Write-Warning "Skip missing path: $path"
    }
}

$staged = git diff --cached --name-only -- $pathsToAdd
if ([string]::IsNullOrWhiteSpace(($staged -join ""))) {
    Write-Host "No staged changes found in selected paths. Nothing to commit."
    exit 0
}

Write-Host "Staged files:"
$staged | ForEach-Object { Write-Host "  - $_" }

git commit -m $CommitMessage

if ($Push) {
    $branch = (git rev-parse --abbrev-ref HEAD).Trim()
    if ([string]::IsNullOrWhiteSpace($branch)) {
        throw "Unable to detect current branch."
    }
    git push origin $branch
}

Write-Host "Done."
