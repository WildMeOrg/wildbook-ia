#!/usr/bin/env bash

set -e

usage () {
    echo "Usage: $0 [-t <tag>] [-r <registry-url>] [<image> ...]";
}

# Parse commandline options
while getopts ":t:r:" option; do
    case ${option} in
        t ) TAG=${OPTARG};;
        r ) REGISTRY=${OPTARG};;
        \? ) usage; exit 1;;
    esac
done
shift $((OPTIND - 1))

# Assign variables
TAG=${TAG:-latest}
REGISTRY=${REGISTRY:-wildme}
IMAGES=${@:-wbia-base wbia-dependencies wbia-provision wbia wildbook-ia}

IMG_TAG="${REGISTRY}/${IMG}:${TAG}"

# Tag built images from `build.sh`, which tags as `latest`
for IMG in $IMAGES; do
    echo "Tagging wildme/${IMG}:latest --> ${IMG_TAG}"
    docker tag wildme/${IMG}:latest ${IMG_TAG}
    echo "Pushing ${IMG_TAG}"
    docker push ${IMG_TAG}
done
