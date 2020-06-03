#!/bin/bash
# Win32 path hacks
export CWD=$(pwd)

# FIXME: Weird directory dependency
#export PATHSEP=$(python -c "import os; print(os.pathsep)")
#if [[ "$OSTYPE" == "win32" ]]; then
export SYSNAME="$(expr substr $(uname -s) 1 10)"
if [ "$SYSNAME" = "MINGW32_NT" ]; then
    export PY=python
else
    export PY=python2.7
fi

#cd ~jonathan/code/wbia
#$PY main.py $FLAGS
$PY -m wbia $FLAGS
