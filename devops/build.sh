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

# Optional: pass NO_CACHE=1 or --no-cache to disable docker layer cache
NO_CACHE_FLAG=""
for arg in "$@"; do
    if [ "$arg" = "--no-cache" ]; then
        NO_CACHE_FLAG="--no-cache"; shift || true
    fi
done
if [ "${NO_CACHE:-0}" = "1" ]; then
    NO_CACHE_FLAG="--no-cache"
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

# Step 3: Build the base image (adds OpenCV dev libs)
log "Building base image (wbia-base) from $DEVOPS_DIR..."
docker build $NO_CACHE_FLAG \
    -t wbia-base \
    -f "$DEVOPS_DIR/Dockerfile.base" \
    "$DEVOPS_DIR"

# Step 4: Build provision layer (install Python deps and build extensions)
log "Building provision image (wbia-provision)..."
docker build $NO_CACHE_FLAG \
    -t wbia-provision \
    -f "$DEVOPS_DIR/Dockerfile.provision" \
    "$DEVOPS_DIR"

# Step 5: Build the final image (includes full repo and entrypoint)
FINAL_DOCKERFILE="$DEVOPS_DIR/Dockerfile.main"
if [ ! -f "$FINAL_DOCKERFILE" ]; then
    # Fallback to primary Dockerfile if Dockerfile.main absent
    FINAL_DOCKERFILE="$DEVOPS_DIR/Dockerfile"
fi
log "Building final WBIA image (wildme/wbia:latest) using $(basename "$FINAL_DOCKERFILE")..."
docker build $NO_CACHE_FLAG \
        -t wildme/wbia:latest \
        --build-arg BUILD_CONTEXT="$BUILD_DIR" \
        -f "$FINAL_DOCKERFILE" \
        "$DEVOPS_DIR"

log "Build complete. You can now run the image using:"
echo "  docker run -it wildme/wbia:latest"
