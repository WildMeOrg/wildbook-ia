#!/bin/bash

set -ex

source /bin/setup

exec gosu ${HOST_USER}:${HOST_USER} /bin/bash
