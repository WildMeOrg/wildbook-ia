#!/bin/bash

set -e

# Use the user mounted wildbook-ia code to if mounted at /code
if [ -d /code ]; then
    echo "*** $0 --- Uninstalling wildbook-ia"
    pip uninstall -y wildbook-ia
    echo "*** $0 --- Uninstalling sentry_sdk (in development)"
    pip uninstall -y sentry_sdk
    echo "*** $0 --- Installing development version of wildbook-ia at /code"
    pushd /code && pip install -e ".[tests,postgres]" && popd
fi


echo "*** $0 --- progressing to main execution"

# Supply EXEC_PRIVILEGED=1 to run your given command as the privileged user.
if [ $EXEC_PRIVILEGED ]; then
    exec "$@"
else
    # Catch all for CMD or 'command' in docker-compose
    # Executes as `wbia` with PID 1
    exec gosu wbia:wbia "$@"
fi
