#!/bin/bash
cd $(dirname $0)

# <CORRECT_PYTHON>
# GET CORRECT PYTHON ON ALL PLATFORMS
export PYMAJOR="$(python -c "import sys; print(sys.version_info[0])")"
export PYMINOR="$(python -c "import sys; print(sys.version_info[1])")"
export SYSNAME="$(expr substr $(uname -s) 1 10)"
if [ "$SYSNAME" = "MINGW32_NT" ]; then
    export PYEXE=python
else
    if [ "$PYMAJOR" = "3" ]; then
        # virtual env?
        export PYEXE=python2.7
    elif [ "$PYMAJOR" = "2" ] && [ "$PYMINOR" != "7" ] ; then
        # virtual env?
        export PYEXE=python2.7
    else
        export PYEXE=python
    fi
fi
# </CORRECT_PYTHON>

$PYEXE ibeis/tests/reset_testdbs.py $@
#$PYEXE dev.py -t mtest
#$PYEXE dev.py -t nauts

echo "PYEXE = $PYEXE"
#python ibeis/tests/test_gui_import_images.py --set-dbdir
#python ibeis/tests/test_gui_add_roi.py


#profiler.sh ibeis/tests/reset_testdbs.py

reset_pz_mtest()
{
    # Delete PZ_MTEST
    #python -c "import ibeis, os, utool; utool.delete(os.path.join(ibeis.sysres.get_workdir(), 'PZ_MTEST'))"
    python -c "import ibeis, os, utool; utool.vd(os.path.join(ibeis.sysres.get_workdir(), 'PZ_MTEST'))"
    python dev.py -t mtest

}
