#!/bin/bash

set -e

# Catch all for CMD or 'command' in docker-compose
# Executes as `wbia` with PID 1
exec gosu wbia:wbia $@
