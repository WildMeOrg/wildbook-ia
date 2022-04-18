#!/bin/bash

set -ex

source /bin/setup

exec gosu ${HOST_USER}:docker /bin/bash
