FROM wildme/wbia-base:latest as org.wildme.wbia.app

MAINTAINER Wild Me <dev@wildme.org>

# Fix for ubuntu 18.04 container https://stackoverflow.com/a/58173981/176882
ENV LANG C.UTF-8

ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

USER root

COPY . /tmp/wbia

# XXX Move to base
RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

RUN set -x \
    && cd /tmp/wbia \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install --verbose --no-deps . \
    # Install pytorch early because it's a large package
    && python3 -m pip install torch \
    && python3 -m pip install -r /tmp/wbia/requirements.txt \
    && python3 -m pip install -r /tmp/wbia/requirements.txt -r /tmp/wbia/requirements/plugins.txt

USER wbia

    
