#!/bin/bash

set -ex

setup

exec gosu ${HOST_USER}:${HOST_USER} $@
