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

    # Auto-detect "already inside the right clone" — operators frequently
    # run `git clone … && cd scalerl-prime-rl && bash scripts/install.sh`
    # to install apt build deps, hitting this script from inside the repo.
    # Without auto-detect, the default `SKIP_CLONE=0` either fails loudly
    # ("destination path 'scalerl-prime-rl' already exists") or silently
    # creates a nested `./scalerl-prime-rl/scalerl-prime-rl/` whose venv
    # the outer tree's sbatch can't find. Treat being inside a git repo
    # whose origin matches our REPO_OWNER/REPO_ID as implicit SKIP_CLONE.
    if [ "$SKIP_CLONE" -eq 0 ] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        ORIGIN_URL=$(git config --get remote.origin.url 2>/dev/null || echo "")
        if echo "$ORIGIN_URL" | grep -q "${REPO_OWNER}/${REPO_ID}"; then
            log_info "Detected existing clone of ${REPO_OWNER}/${REPO_ID} (origin: $ORIGIN_URL); skipping clone."
            SKIP_CLONE=1
        fi
    fi

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
    # Pin uv to the validated minor. `pyproject.toml` has `[tool.uv] preview = true`
    # which opts into preview-stage resolver behavior; resolver semantics
    # (extra-build-variables, override-dependencies, no-build-isolation-package)
    # have changed across uv minors. The lock file was generated against the
    # version pinned below — letting `uv` self-update can break `--locked`
    # (refusal) or, worse, silently re-resolve transitive deps. Override via
    # `UV_VERSION=latest` if you've validated a newer release.
    UV_VERSION="${UV_VERSION:-0.11.6}"
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh

    log_info "Sourcing uv environment..."
    if ! command -v uv &> /dev/null; then
        source $HOME/.local/bin/env
    fi

    # Pre-flight: CUTLASS / FA4 build from source on first sync; fail loud and
    # actionable if the toolchain is missing rather than crash mid-sync 5
    # minutes in with an opaque traceback.
    log_info "Checking build toolchain..."
    missing=()
    for tool in nvcc gcc git curl; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing+=("$tool")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        log_warn "Missing required build tools: ${missing[*]}"
        log_warn "Install on Debian/Ubuntu via 'INSTALL_BASE_PACKAGES=1 bash $0' (apt)"
        log_warn "or your cluster's package manager. CUDA toolkit (nvcc) must"
        log_warn "match the torch CUDA version pinned in pyproject.toml (12.8)."
        exit 1
    fi

    log_info "Syncing virtual environment..."
    # `--extra all` (the aggregate extra defined in pyproject.toml) excludes
    # `flash-attn-3` because its wheels ship Hopper sm_90 kernels only and
    # crash on B200 with "no kernel image available." `uv sync --all-extras`
    # would enumerate every named extra and pull FA3 in.
    # `--locked` matches what Dockerfile.cuda does — without it the host install
    # can drift from uv.lock if any of the git-pinned deps moved upstream.
    uv sync --extra all --locked

    # FA4 lives in the `flash_attn.cute.*` namespace, which the FA2 stub
    # shadow-installs by default; without this repair `import flash_attn`
    # *succeeds* but FA4 kernels silently fall back to FA2 — a wrong-kernel
    # hazard that does NOT surface during `uv run python -c "import flash_attn"`
    # (the README's only smoke check). Run the same repair the Dockerfile
    # and sbatch run, immediately after `uv sync`, so the standalone-install
    # path matches the container build.
    log_info "Repairing flash_attn.cute namespace (FA4)..."
    bash "$(dirname "${BASH_SOURCE[0]}")/fix-flash-attn-cute.sh"

    # pre-commit hooks are dev-only; skip cleanly when the install path
    # is a tarball / vendored drop with no `.git` directory (Snowflake-
    # style consumption). Without this guard, `pre-commit install` aborts
    # the script with a confusing "not a git repository" error after
    # every other step has succeeded. Set INSTALL_PRECOMMIT=0 to
    # explicitly skip even inside a git checkout.
    if [ "${INSTALL_PRECOMMIT:-1}" = "1" ] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        log_info "Installing pre-commit hooks..."
        uv run pre-commit install
    else
        log_info "Skipping pre-commit hooks (not a git checkout, or INSTALL_PRECOMMIT=0)."
    fi

    log_info "Installation completed!"
}

main