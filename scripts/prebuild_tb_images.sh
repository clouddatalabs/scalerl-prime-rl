#!/bin/bash
# Build every Terminal-Bench task image and push to the cluster docker
# registry. Run once after cloning, and again whenever any task's
# environment/ contents change.
#
# Usage:
#   TB_REGISTRY_HOST=rlgpu0:5000 bash scripts/prebuild_tb_images.sh
#
# Prerequisites:
#   - docker daemon reachable on the host running this script
#   - that daemon's /etc/docker/daemon.json includes
#     "insecure-registries": ["rlgpu0:5000"] (followed by `systemctl
#     restart docker`); HTTP-only is acceptable for an internal cluster
#   - registry:2 service running on rlgpu0:5000 with NVMe-backed
#     storage (see your cluster admin docs)
#
# Why DOCKER_BUILDKIT=0:
#   Live builds in our prior smoke runs hit BuildKit content-store
#   poisoning (failing to extract layers despite the build itself
#   succeeding). The legacy builder (DOCKER_BUILDKIT=0) commits
#   intermediate containers via `docker commit` and writes layers
#   directly to the daemon's image storage, bypassing BuildKit's
#   content store entirely. Our task Dockerfiles are simple
#   FROM/RUN/COPY/WORKDIR — no BuildKit-only features needed.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
REGISTRY="${TB_REGISTRY_HOST:-rlgpu0:5000}"
TASKS_DIR="$REPO_ROOT/environments/terminal_bench/tasks"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not on PATH; install docker on this host first." >&2
    exit 1
fi
if [ ! -d "$TASKS_DIR" ]; then
    echo "ERROR: $TASKS_DIR does not exist." >&2
    exit 1
fi

count=0
for task_dir in "$TASKS_DIR"/*/; do
    # Bash `*/` glob filters dangling symlinks (dir-test fails) and
    # plain files (no trailing /). The Dockerfile check below catches
    # the rare task that is a real dir with no environment/ subtree.
    task=$(basename "$task_dir")
    dockerfile="$task_dir/environment/Dockerfile"
    if [ ! -f "$dockerfile" ]; then
        continue
    fi
    tag="$REGISTRY/tb-local/$task:latest"
    echo "[build+push] $tag"
    DOCKER_BUILDKIT=0 docker build \
        -t "$tag" \
        -f "$dockerfile" \
        "$task_dir/environment"
    docker push "$tag"
    count=$((count + 1))
done
echo "[done] built+pushed $count images to $REGISTRY"
