#!/bin/bash
set -euo pipefail

# Constants
REPO_URL="https://github.com/WildMeOrg/wildbook-ia.git"
BRANCH_NAME="build-fix-jul25"
BUILD_DIR="wbia_build_src"

# Helper for logging
log() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

# Optional flags:
#   NO_CACHE=1 / --no-cache  : disable cache
#   PRUNE=1 / --prune        : aggressively prune Docker images/containers/build cache first
NO_CACHE_FLAG=""
PRUNE_FLAG=0
for arg in "$@"; do
    case "$arg" in
        --no-cache) NO_CACHE_FLAG="--no-cache"; shift || true ;;
        --prune) PRUNE_FLAG=1; shift || true ;;
    esac
done
[ "${NO_CACHE:-0}" = "1" ] && NO_CACHE_FLAG="--no-cache"
[ "${PRUNE:-0}" = "1" ] && PRUNE_FLAG=1

if [ "$PRUNE_FLAG" = "1" ]; then
    log "Pruning Docker system to reclaim space..."
    docker system df || true
    docker ps -aq | xargs -r docker rm -f || true
    docker images -q | xargs -r docker rmi -f || true
    docker builder prune -af || true
    docker system prune -af || true
    docker volume prune -f || true
    log "Post-prune disk usage:"; docker system df || true
fi

# Step 1 / 2: Optionally clone fresh source (skip if SKIP_CLONE=1 or --skip-clone)
SKIP_CLONE_FLAG=${SKIP_CLONE:-0}
for arg in "$@"; do
    if [ "$arg" = "--skip-clone" ]; then
        SKIP_CLONE_FLAG=1; shift || true
    fi
done
if [ "$SKIP_CLONE_FLAG" = "0" ]; then
    log "Cleaning previous build directory..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    log "Cloning Wildbook-IA from branch: $BRANCH_NAME..."
    git clone --depth 1 --branch "$BRANCH_NAME" "$REPO_URL" "$BUILD_DIR"
else
    log "Skipping clone; using existing working copy"
fi

# Determine root containing devops directory
if [ -d devops ]; then
    SRC_ROOT="."
elif [ -d "$BUILD_DIR/devops" ]; then
    SRC_ROOT="$BUILD_DIR"
else
    log "devops directory not found in current or cloned paths"; exit 1
fi

DEVOPS_DIR="$SRC_ROOT/devops"
BUILD_CONTEXT_ROOT="$SRC_ROOT"

# Step 3: Build the base image (adds OpenCV dev libs)
IMAGE_NAMESPACE=${IMAGE_NAMESPACE:-wildme}
log "Building base image (wbia-base) from $DEVOPS_DIR..."
docker build $NO_CACHE_FLAG \
    -t wbia-base \
    -f "$DEVOPS_DIR/Dockerfile.base" \
    "$BUILD_CONTEXT_ROOT"
# Tag with namespace (required by publish script expecting ${IMAGE_NAMESPACE}/wbia-base:latest)
docker tag wbia-base ${IMAGE_NAMESPACE}/wbia-base:latest || true

# Step 3b: Build dependencies stage (optional intermediate layer)
if [ -f "$DEVOPS_DIR/Dockerfile.dependencies" ]; then
    log "Building dependencies image (wbia-dependencies)..."
    docker build $NO_CACHE_FLAG \
        -t wbia-dependencies \
        -f "$DEVOPS_DIR/Dockerfile.dependencies" \
        "$BUILD_CONTEXT_ROOT"
    docker tag wbia-dependencies ${IMAGE_NAMESPACE}/wbia-dependencies:latest || true
else
    log "Skipping dependencies image (Dockerfile.dependencies not found)"
fi

# Step 4: Build provision layer (install Python deps and build extensions)
log "Building provision image (wbia-provision)..."
LIGHT_MODE_ARG=""
if [ "${LIGHT_MODE:-0}" = "1" ]; then
    log "LIGHT_MODE=1 (torch pruned, CUDA libs stripped)"
    LIGHT_MODE_ARG="--build-arg LIGHT_MODE=1"
fi
docker build $NO_CACHE_FLAG $LIGHT_MODE_ARG \
    -t wbia-provision \
    -f "$DEVOPS_DIR/Dockerfile.provision" \
    "$BUILD_CONTEXT_ROOT"
# Namespace tag for provision image
docker tag wbia-provision ${IMAGE_NAMESPACE}/wbia-provision:latest || true

# Step 5: Build the final image (includes full repo and entrypoint)
FINAL_DOCKERFILE="$DEVOPS_DIR/Dockerfile.main"
if [ ! -f "$FINAL_DOCKERFILE" ]; then
    # Fallback to primary Dockerfile if Dockerfile.main absent
    FINAL_DOCKERFILE="$DEVOPS_DIR/Dockerfile"
fi
log "Building final WBIA image (${IMAGE_NAMESPACE}/wbia:latest) using $(basename "$FINAL_DOCKERFILE")..."
docker build $NO_CACHE_FLAG \
    -t ${IMAGE_NAMESPACE}/wbia:latest \
    $LIGHT_MODE_ARG \
    --build-arg WBIA_PROVISION_IMAGE=wbia-provision \
    --build-arg BUILD_CONTEXT="$BUILD_DIR" \
    -f "$FINAL_DOCKERFILE" \
    "$BUILD_CONTEXT_ROOT"

# Additional convenience / legacy tags expected by publish script
docker tag ${IMAGE_NAMESPACE}/wbia:latest wbia:latest || true
docker tag ${IMAGE_NAMESPACE}/wbia:latest ${IMAGE_NAMESPACE}/wildbook-ia:latest || true

log "Build complete. You can now run the image using:"
echo "  docker run -it wildme/wbia:latest"
