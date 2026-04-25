#!/usr/bin/env bash
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

# `clouddatalabs/scalerl-prime-rl` is the public ScaleRL fork; the seven
# recipe ingredients (CISPO, prompt-avg, batch-norm, FP32 LM-head, ZA-filter,
# NPR, async pipeline-RL) live ONLY on this fork. Cloning plain upstream
# `PrimeIntellect-ai/prime-rl` from this script would silently strip every
# ingredient and ship a vanilla install — which is the exact failure mode
# the README banner warns against.
REPO_OWNER="clouddatalabs"
REPO_ID="scalerl-prime-rl"

# Flag defaults (can be overridden via env)
SKIP_CLONE=${SKIP_CLONE:-0}

has_ssh_access() {
    # Probe SSH auth to GitHub without prompting; treat any nonzero as "no ssh"
    # We try a quick ls-remote to avoid cloning on failure.
    # Disable -e for the probe so the script doesn't exit on a failed test.
    set +e
    timeout 5s git ls-remote --heads "git@github.com:${REPO_OWNER}/${REPO_ID}.git" >/dev/null 2>&1
    rc=$?
    set -e
    return $rc
}

ensure_known_hosts() {
    # Make sure ~/.ssh exists with the right perms, then add GitHub host key
    # iff it isn't already there. Prior implementation appended on every run,
    # accumulating duplicate entries.
    mkdir -p "${HOME}/.ssh"
    chmod 700 "${HOME}/.ssh"
    if command -v ssh-keyscan >/dev/null 2>&1; then
        if ! ssh-keygen -F github.com >/dev/null 2>&1; then
            ssh-keyscan -H github.com 2>/dev/null >> "${HOME}/.ssh/known_hosts"
        fi
        chmod 600 "${HOME}/.ssh/known_hosts"
    fi
}

main() {
    # Base-package installation is opt-in: a public bootstrap script that
    # silently mutates the host's apt set is a footgun on managed clusters.
    # Set `INSTALL_BASE_PACKAGES=1` to opt in (matches the legacy behavior).
    if [ "${INSTALL_BASE_PACKAGES:-0}" = "1" ]; then
        if ! command -v sudo &>/dev/null; then
            apt update
            apt install -y sudo
        fi
        log_info "Installing base packages..."
        sudo apt update && sudo apt install -y build-essential openssh-client curl git tmux htop nvtop
    else
        log_info "Skipping base-package install (set INSTALL_BASE_PACKAGES=1 to opt in)."
    fi

    log_info "Configuring SSH known_hosts for GitHub..."
    ensure_known_hosts

    if [ "$SKIP_CLONE" -eq 1 ]; then
        log_info "Skipping clone; assuming we are already inside the repo."
    else
        log_info "Determining best way to clone (SSH vs HTTPS)..."
        if has_ssh_access; then
            log_info "SSH access to GitHub works. Cloning via SSH."
            git clone "git@github.com:${REPO_OWNER}/${REPO_ID}.git"
        else
            log_warn "SSH auth to GitHub not available. Cloning via HTTPS."
            git clone "https://github.com/${REPO_OWNER}/${REPO_ID}.git"
        fi

        log_info "Entering project directory..."
        cd "${REPO_ID}"
    fi

    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    log_info "Sourcing uv environment..."
    if ! command -v uv &> /dev/null; then
        source $HOME/.local/bin/env
    fi

    log_info "Installing prime..."
    uv tool install prime

    log_info "Syncing virtual environment..."
    # `--extra all` (the aggregate extra defined in pyproject.toml) excludes
    # `flash-attn-3` because its wheels ship Hopper sm_90 kernels only and
    # crash on B200 with "no kernel image available." `uv sync --all-extras`
    # would enumerate every named extra and pull FA3 in.
    uv sync --extra all

    log_info "Installing pre-commit hooks..."
    uv run pre-commit install

    log_info "Installation completed!"
}

main