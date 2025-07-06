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

# Step 1: Clean and prepare build directory
log "Cleaning previous build directory..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Step 2: Clone the repo and checkout the correct branch
log "Cloning Wildbook-IA from branch: $BRANCH_NAME..."
git clone --depth 1 --branch "$BRANCH_NAME" "$REPO_URL" "$BUILD_DIR"

# Step 3: Build the base image (adds OpenCV dev libs)
log "Building base image (wbia-base)..."
docker build \
    -t wbia-base \
    -f devops/Dockerfile.base \
    devops/

# Step 4: Build provision layer (install Python deps and build extensions)
log "Building provision image (wbia-provision)..."
docker build \
    -t wbia-provision \
    -f devops/Dockerfile.provision \
    devops/

# Step 5: Build the final image (includes full repo and entrypoint)
log "Building final WBIA image (wildme/wbia:latest)..."
docker build \
    -t wildme/wbia:latest \
    --build-arg BUILD_CONTEXT="$BUILD_DIR" \
    -f devops/Dockerfile.main \
    devops/

log "Build complete. You can now run the image using:"
echo "  docker run -it wildme/wbia:latest"
