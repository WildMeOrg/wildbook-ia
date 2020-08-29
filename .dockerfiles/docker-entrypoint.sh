#!/bin/bash

set -e

# Use the user mounted wildbook-ia code to if mounted at /code
if [ -d /code ]; then
    echo "*** $0 --- Uninstalling wildbook-ia"
    pip uninstall -y wildbook-ia
    echo "*** $0 --- Installing development version of wildbook-ia at /code"
    pushd /code && pip install -e ".[tests]" && popd
fi


echo "*** $0 --- progressing to main execution"

# Catch all for CMD or 'command' in docker-compose
# Executes as `wbia` with PID 1
exec gosu wbia:wbia $@
