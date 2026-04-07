#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-.}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-add deploy-ready package and scripts}"
INCLUDE_OPTIONAL=0
PUSH_AFTER_COMMIT=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./git_commit_linux.sh [--repo-root PATH] [--message TEXT] [--include-optional] [--push] [--dry-run]

Options:
  --repo-root PATH       Repository path (default: .)
  --message TEXT         Commit message
  --include-optional     Also stage docs/demo folders
  --push                 Push to origin/<current-branch> after commit
  --dry-run              Preview only, no git add/commit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --message)
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --include-optional)
      INCLUDE_OPTIONAL=1
      shift
      ;;
    --push)
      PUSH_AFTER_COMMIT=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

cd "$REPO_ROOT"
REPO_TOP="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_TOP" ]]; then
  echo "Current directory is not inside a git repository."
  exit 1
fi
cd "$REPO_TOP"

required_paths=(
  "examples/tabular/competition_submit_package/02_code_folder"
  "examples/tabular/competition_submit_package/05_deploy_info"
)

optional_paths=(
  "examples/tabular/competition_submit_package/03_docs_folder"
  "examples/tabular/competition_submit_package/04_demo_folder"
)

paths_to_add=("${required_paths[@]}")
if [[ "$INCLUDE_OPTIONAL" -eq 1 ]]; then
  paths_to_add+=("${optional_paths[@]}")
fi

echo "Repository root: $REPO_TOP"
echo "Staging paths:"
for p in "${paths_to_add[@]}"; do
  echo "  - $p"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run mode. No git add/commit executed."
  git status --short
  exit 0
fi

for p in "${paths_to_add[@]}"; do
  if [[ -e "$p" ]]; then
    git add -- "$p"
  else
    echo "Skip missing path: $p"
  fi
done

staged="$(git diff --cached --name-only -- "${paths_to_add[@]}")"
if [[ -z "$staged" ]]; then
  echo "No staged changes found in selected paths. Nothing to commit."
  exit 0
fi

echo "Staged files:"
echo "$staged"

git commit -m "$COMMIT_MESSAGE"

if [[ "$PUSH_AFTER_COMMIT" -eq 1 ]]; then
  branch="$(git rev-parse --abbrev-ref HEAD)"
  git push origin "$branch"
fi

echo "Done."
