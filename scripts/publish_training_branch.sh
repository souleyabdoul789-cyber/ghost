#!/usr/bin/env bash
# Publie ghost 0.6.2 vers le nouveau dépôt dédié, sans toucher à main.
set -euo pipefail

REMOTE="${GHOST_GITHUB_REMOTE:-https://github.com/souleyabdoul789-cyber/ghost.git}"
BRANCH="pretraining/0.6.2"

command -v git >/dev/null 2>&1 || { echo "Git est requis." >&2; exit 1; }
NORMALIZED_REMOTE="${REMOTE%.git}"
REPO_NAME="${NORMALIZED_REMOTE##*/}"
if [[ "$REPO_NAME" != "ghost" ]]; then
  echo "Refus : le remote doit pointer vers un dépôt nommé ghost, pas vers $REPO_NAME." >&2
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b "$BRANCH"
fi

if git remote get-url origin >/dev/null 2>&1; then
  CURRENT_REMOTE="$(git remote get-url origin)"
  if [[ "$CURRENT_REMOTE" != "$REMOTE" ]]; then
    echo "Le remote origin actuel ne correspond pas à $REMOTE. Aucun changement effectué." >&2
    exit 3
  fi
else
  git remote add origin "$REMOTE"
fi

git switch -C "$BRANCH"
git add -A
if ! git diff --cached --quiet; then
  git -c user.name="ghost / G-SOCIETY DEV" \
      -c user.email="noreply@users.noreply.github.com" \
      commit -m "ghost 0.6.2: French pretraining pipeline"
else
  echo "Aucun changement local à committer."
fi

# Push normal seulement : aucun --force, aucune écriture sur main.
git push -u origin "HEAD:refs/heads/$BRANCH"
echo "Branche publiée : $BRANCH"
echo "Le workflow ghost-first-pretrain démarre sur le push de cette branche."
