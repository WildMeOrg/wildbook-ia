#!/bin/bash

set -ex

# Executes as `wbia` with PID 1
exec gosu wbia:wbia $@
