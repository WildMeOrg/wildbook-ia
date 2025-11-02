#!/bin/bash

set -ex

source /bin/setup

# Supply EXEC_PRIVILEGED=1 to run your given command as the privileged user.
if [ $EXEC_PRIVILEGED ]; then
    exec $@
else
    # Use docker group if it exists, otherwise use the user's primary group
    if getent group docker >/dev/null 2>&1; then
        exec gosu ${HOST_USER}:docker $@
    else
        exec gosu ${HOST_USER} $@
    fi
fi
