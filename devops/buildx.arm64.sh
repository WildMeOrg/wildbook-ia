#!/usr/bin/env bash

set -ex

# See https://stackoverflow.com/a/246128/176882
export ROOT_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export DOCKER_BUILDKIT=1

export DOCKER_CLI_EXPERIMENTAL=enabled

# Change to the script's root directory location
cd ${ROOT_LOC}

# docker buildx create --name multi-arch-builder --use

# Build the images in dependence order
while [ $# -ge 1 ]; do
    if [ "$1" == "wbia-base" ]; then
        docker buildx build \
            -t wildme/wbia-base:arm64 \
            --cache-to=type=local,dest=.buildx.cache,mode=max \
            --cache-from=type=local,src=.buildx.cache,mode=max \
            --compress \
            --platform linux/arm64 \
            --push \
            base
        #docker push wildme/wbia-base:arm64
    elif [ "$1" == "wbia-provision" ]; then
        docker buildx build \
            -t wildme/wbia-provision:arm64 \
            --cache-to=type=local,dest=.buildx.cache,mode=max \
            --cache-from=type=local,src=.buildx.cache,mode=max \
            --compress \
            --build-arg WBIA_BASE_IMAGE="wildme/wbia-base:arm64" \
            --platform linux/arm64 \
            --push \
            provision
        #docker push wildme/wbia-provision:arm64
    elif [ "$1" == "wbia" ] || [ "$1" == "wildbook-ia" ]; then
        docker buildx build \
            -t wildme/wbia:arm64 \
            -t wildme/wildbook-ia:arm64 \
            --cache-to=type=local,dest=.buildx.cache,mode=max \
            --cache-from=type=local,src=.buildx.cache,mode=max \
            --compress \
            --build-arg WBIA_BASE_IMAGE="wildme/wbia-base:arm64" \
            --build-arg WBIA_PROVISION_IMAGE="wildme/wbia-provision:arm64" \
            --platform linux/arm64 \
            --push \
        #docker push wildme/wbia:arm64
        #docker push wildme/wildbook-ia:arm64
    elif [ "$1" == "wbia-develop" ]; then
        cd ../
        docker buildx build \
            -t wildme/wbia:develop \
            --cache-to=type=local,dest=.buildx.cache,mode=max \
            --cache-from=type=local,src=.buildx.cache,mode=max \
            --platform linux/arm64 \
            --push \
            devops/develop
        cd devops/
    else
        echo "Image $1 not found"
        exit 1
    fi
    shift
done
