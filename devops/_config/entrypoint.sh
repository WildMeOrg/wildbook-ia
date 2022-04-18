#!/bin/bash

set -ex

source /bin/setup

# Supply EXEC_PRIVILEGED=1 to run your given command as the privileged user.
if [ $EXEC_PRIVILEGED ]; then
    exec $@
else
    exec gosu ${HOST_USER}:docker $@
fi
