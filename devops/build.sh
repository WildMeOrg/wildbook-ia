#!/usr/bin/env bash

set -ex

# See https://stackoverflow.com/a/246128/176882
export ROOT_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Change to the script's root directory location
cd ${ROOT_LOC}

# Build the images in dependence order
docker build -t wildme/wbia-base:latest base
docker build -t wildme/wbia-dependencies:latest dependencies
docker build -t wildme/wbia-provision:latest provision
docker build -t wildme/wbia:latest .

cd ../
# Build the runtime container
docker build -t wildbook/wildbook-ia:latest .
