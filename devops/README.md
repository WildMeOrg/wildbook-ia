# Containerized IBEIS Installation

This directory contains the containerized build instructions (docker build files) for creating images for the IBEIS application. The build is divided into four distinct stages: base, dependencies, provision and wbia. These each correspond to the final image name within the container registry (i.e. Docker Hub) as: wildme/wbia-base, wildme/wbia-dependencies, wildme/wbia-provision and wildme/wbia.

## Build instructions

To build the images run:
    ./build.sh

This will build the four stage images.

It is not recommended that you upload these images to the container registry. Please leave that to the continuous integration (CI) service.
