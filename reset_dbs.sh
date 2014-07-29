#!/bin/bash
cd $(dirname $0)

# <CORRECT_PYTHON>
# GET CORRECT PYTHON ON ALL PLATFORMS
export SYSNAME="$(expr substr $(uname -s) 1 10)"
if [ "$SYSNAME" = "MINGW32_NT" ]; then
    export PY=python
else
    export PY=python2.7
fi
# </CORRECT_PYTHON>

$PY ibeis/tests/reset_testdbs.py $@
#python ibeis/tests/test_gui_import_images.py --set-dbdir
#python ibeis/tests/test_gui_add_roi.py
