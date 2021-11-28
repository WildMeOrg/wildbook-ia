#!/usr/bin/env bash

set -ex

# See https://stackoverflow.com/a/246128/176882
export ROOT_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export DOCKER_BUILDKIT=1

export DOCKER_CLI_EXPERIMENTAL=enabled

# Change to the script's root directory location
cd ${ROOT_LOC}

# Build the images in dependence order
while [ $# -ge 1 ]; do
    if [ "$1" == "wbia-base" ]; then
        docker build -t wildme/wbia-base:latest base
    elif [ "$1" == "wbia-provision" ]; then
        docker build -t wildme/wbia-provision:latest provision
    elif [ "$1" == "wbia" ]; then
        docker build --no-cache -t wildme/wbia:latest .
    elif [ "$1" == "wbia-develop" ]; then
        cd ../
        docker build -t wildme/wbia:develop devops/local
        cd devops/
    else
        echo "Image $1 not found"
        exit 1
    fi
    shift
done
