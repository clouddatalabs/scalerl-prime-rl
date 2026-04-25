#!/usr/bin/env bash
#
# Remove logs / checkpoints / weights / rollouts / wandb / evals from the
# repo. Refuses to run from $HOME or `/` so an accidental `cd` doesn't wipe
# the user's whole tree. Anchors to the repo root regardless of cwd.

set -euo pipefail
shopt -s globstar nullglob

# Anchor to the repo root (the directory containing this script's parent).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Hard refuse to run if the resolved root is `$HOME` or `/`. The patterns
# below are recursive globs; running them at home would silently destroy
# every `logs/` / `wandb/` / `rollouts/` under any project the user owns.
if [ "$REPO_ROOT" = "$HOME" ] || [ "$REPO_ROOT" = "/" ]; then
    echo "Error: refusing to clean — repo root resolved to $REPO_ROOT." >&2
    echo "       Move the script back into a subdirectory of the repo." >&2
    exit 1
fi

GREEN='\033[0;32m'
NC='\033[0m'
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

confirm_cleanup() {
    echo "Repo root: $REPO_ROOT"
    echo "This will remove the following paths recursively under that root:"
    echo "  - **/logs    **/checkpoints   **/weights"
    echo "  - **/rollouts **/wandb         **/evals"
    echo "  - *.pydantic_config (top-level)"
    while true; do
        read -r -p "Proceed? [y/N]: " response
        case "$response" in
            [yY]|[yY][eE][sS]) break ;;
            [nN]|[nN][oO]|"") echo "Aborted."; exit 1 ;;
            *) echo "Please answer y or n." ;;
        esac
    done
}

confirm_cleanup
cd "$REPO_ROOT"
# `nullglob` makes the globs expand to nothing (instead of the literal `**/...`)
# when no matches exist; combined with `globstar`, the recursive globs work.
# `**/torchrun` removed — would also nuke `.venv/bin/torchrun` and any
# user file named `torchrun`, with no actual benefit (torchrun output
# lives under `logs/` already).
rm -rf -- **/logs **/checkpoints **/weights **/rollouts **/wandb **/evals
rm -f -- *.pydantic_config
log_info "Cleaned up under $REPO_ROOT"
