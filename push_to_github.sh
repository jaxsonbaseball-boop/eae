#!/usr/bin/env bash
# One-command helper to push Project Apex v3.5 to GitHub (jaxsonbaseball-boop)
set -euo pipefail

REPO_URL="${1:-https://github.com/jaxsonbaseball-boop/project-apex-stuff-plus.git}"
echo "Configuring remote origin -> ${REPO_URL}"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "${REPO_URL}"
else
  git remote add origin "${REPO_URL}"
fi

git branch -M main
git push -u origin main
echo "Successfully pushed Project Apex v3.5 to ${REPO_URL}"
