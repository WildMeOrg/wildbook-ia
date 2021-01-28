# Containerized Wildbook-IA Installation

This directory contains the containerized build instructions (docker build files) for creating images for the Wildbook IA application. The build is divided into four distinct stages: base, dependencies, provision and wbia. These each correspond to the final image name within the container registry (i.e. Docker Hub) as: [wildme/wbia-base](https://hub.docker.com/r/wildme/wbia-base), [wildme/wbia-dependencies](https://hub.docker.com/r/wildme/wbia-dependencies), [wildme/wbia-provision](https://hub.docker.com/r/wildme/wbia-provision) and [wildme/wbia](https://hub.docker.com/r/wildme/wbia).

## Build instructions

To build the images run:
    ./build.sh

This will build the four stage images.

It is not recommended that you upload these images to the container registry. Please leave that to the continuous integration (CI) service.

## Publish instructions

To publish the results use:
    ./publish.sh

This will publish the four build images to [Docker Hub](https://hub.docker.com) by default. You can change the container registry to publish to using the `-r` option. See `./publish.sh -h` for usage details.

Do not run this locally, leave that to the continuous integration (CI) service. See the Nightly workflow in `.github/workflows/nightly.yml` for more information.

## Deployment with Postgres

For example instruction on how to deploy the application with postgres see: [Deploying the application with Postgres](deploy-with-prostgres.md)
