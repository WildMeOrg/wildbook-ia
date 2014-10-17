#!/bin/bash
# Runs all tests
# Win32 path hacks
export CWD=$(pwd)
export PYMAJOR="$(python -c "import sys; print(sys.version_info[0])")"

# <CORRECT_PYTHON>
# GET CORRECT PYTHON ON ALL PLATFORMS
export SYSNAME="$(expr substr $(uname -s) 1 10)"
if [ "$SYSNAME" = "MINGW32_NT" ]; then
    export PYEXE=python
else
    if [ "$PYMAJOR" = "3" ]; then
        # virtual env?
        export PYEXE=python
    else
        export PYEXE=python2.7
    fi
fi
# </CORRECT_PYTHON>

PRINT_DELIMETER()
{
    printf "\n#\n#\n#>>>>>>>>>>> next_test\n\n"
}

export TEST_ARGV="--quiet --noshow $@"

export VTOOL_DIR=$($PYEXE -c "import os, vtool;print(str(os.path.dirname(os.path.dirname(vtool.__file__))))")
export HESAFF_DIR=$($PYEXE -c "import os, pyhesaff;print(str(os.path.dirname(os.path.dirname(pyhesaff.__file__))))")

# Default tests to run
set_test_flags()
{
    export DEFAULT=$1
    export VTOOL_TEST=$DEFAULT
    export GUI_TEST=$DEFAULT
    export IBEIS_TEST=$DEFAULT
    export SQL_TEST=$DEFAULT
    export VIEW_TEST=$DEFAULT
    export MISC_TEST=$DEFAULT
    export OTHER_TEST=$DEFAULT
    export HESAFF_TEST=$DEFAULT
}
set_test_flags OFF
export IBEIS_TEST=ON

# Parse for bash commandline args
for i in "$@"
do
case $i in --testall)
    set_test_flags ON
    ;;
esac
case $i in --notestvtool)
    export VTOOL_TEST=OFF
    ;;
esac
case $i in --testvtool)
    export VTOOL_TEST=ON
    ;;
esac
case $i in --notestgui)
    export GUI_TEST=OFF
    ;;
esac
case $i in --testgui)
    export GUI_TEST=ON
    ;;
esac
case $i in --notestibeis)
    export IBEIS_TEST=OFF
    ;;
esac
case $i in --testibeis)
    export IBEIS_TEST=ON
    ;;
esac
case $i in --notestsql)
    export SQL_TEST=OFF
    ;;
esac
case $i in --testsql)
    export SQL_TEST=ON
    ;;
esac
case $i in --notestview)
    export VIEW_TEST=OFF
    ;;
esac
case $i in --testview)
    export VIEW_TEST=ON
    ;;
esac
case $i in --notestmisc)
    export MISC_TEST=OFF
    ;;
esac
case $i in --testmisc)
    export MISC_TEST=ON
    ;;
esac
case $i in --notestother)
    export OTHER_TEST=OFF
    ;;
esac
case $i in --testother)
    export OTHER_TEST=ON
    ;;
esac
case $i in --notesthesaff)
    export HESAFF_TEST=OFF
    ;;
esac
case $i in --testhesaff)
    export HESAFF_TEST=ON
    ;;
esac
done

BEGIN_TESTS()
{
cat <<EOF
  ______ _     _ __   _      _______ _______ _______ _______ _______
 |_____/ |     | | \  |         |    |______ |______    |    |______
 |    \_ |_____| |  \_|         |    |______ ______|    |    ______|
                                                                    

EOF
    echo "BEGIN: TEST_ARGV=$TEST_ARGV"
    PRINT_DELIMETER
    num_passed=0
    num_ran=0
    export FAILED_TESTS=''
}

RUN_TEST()
{
    echo "RUN_TEST: $@"
    export TEST="$PYEXE $@ $TEST_ARGV"
    $TEST
    export RETURN_CODE=$?
    PRINT_DELIMETER
    num_passed=$(($num_passed + (1 - $RETURN_CODE)))
    num_ran=$(($num_ran + 1))
    if [ "$RETURN_CODE" != "0" ] ; then
        export FAILED_TESTS="$FAILED_TESTS\n$TEST"
    fi
}

END_TESTS()
{
    echo "RUN_TESTS: DONE"
    if [ "$FAILED_TESTS" != "" ] ; then
        echo "-----"
        printf "Failed Tests:"
        printf "$FAILED_TESTS\n"
        printf "$FAILED_TESTS\n" >> failed.txt
        echo "-----"
    fi
    echo "$num_passed / $num_ran tests passed"
}

#---------------------------------------------
# START TESTS
BEGIN_TESTS

# Quick Tests (always run)
RUN_TEST ibeis/tests/assert_modules.py

#---------------------------------------------
#VTOOL TESTS
if [ "$VTOOL_TEST" = "ON" ] ; then
cat <<EOF
    _  _ ___ ____ ____ _       ___ ____ ____ ___ ____ 
    |  |  |  |  | |  | |        |  |___ [__   |  [__  
     \/   |  |__| |__| |___     |  |___ ___]  |  ___]
EOF
    RUN_TEST $VTOOL_DIR/vtool/tests/test_draw_keypoint.py 
    RUN_TEST $VTOOL_DIR/vtool/tests/test_spatial_verification.py 
    RUN_TEST $VTOOL_DIR/vtool/tests/test_exhaustive_ori_extract.py 
    RUN_TEST $VTOOL_DIR/vtool/tests/test_vtool.py 
    RUN_TEST $VTOOL_DIR/vtool/tests/test_akmeans.py 
    RUN_TEST $VTOOL_DIR/vtool/tests/test_pyflann.py 
fi
#---------------------------------------------
#GUI TESTS
if [ "$GUI_TEST" = "ON" ] ; then
cat <<EOF
    ____ _  _ _    ___ ____ ____ ___ ____ 
    | __ |  | |     |  |___ [__   |  [__  
    |__] |__| |     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/test_gui_selection.py 
    RUN_TEST ibeis/tests/test_gui_open_database.py 
    RUN_TEST ibeis/tests/test_gui_add_annotation.py 
    RUN_TEST ibeis/tests/test_gui_all.py 
    RUN_TEST ibeis/tests/test_gui_import_images.py 
fi
#---------------------------------------------
#IBEIS TESTS
if [ "$IBEIS_TEST" = "ON" ] ; then
cat <<EOF
    _ ___  ____ _ ____    ___ ____ ____ ___ ____ 
    | |__] |___ | [__      |  |___ [__   |  [__  
    | |__] |___ | ___]     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/test_ibs_detect.py 
    RUN_TEST ibeis/tests/test_ibs_query_components.py 
    RUN_TEST ibeis/tests/test_ibs.py 
    RUN_TEST ibeis/tests/test_ibs_query.py 
    RUN_TEST ibeis/tests/test_ibs_encounters.py 
    RUN_TEST ibeis/tests/test_ibs_control.py 
    RUN_TEST ibeis/tests/test_ibs_feat_compute.py 
    RUN_TEST ibeis/tests/test_ibs_uuid.py 
    RUN_TEST ibeis/tests/test_ibs_localize_images.py 
    RUN_TEST ibeis/tests/test_ibs_add_images.py 
    RUN_TEST ibeis/tests/test_ibs_chip_compute.py 
    RUN_TEST ibeis/tests/test_ibs_info.py 
    RUN_TEST ibeis/tests/test_ibs_add_name.py 
    RUN_TEST ibeis/tests/test_ibs_getters.py 
    RUN_TEST ibeis/tests/test_ibs_convert_bbox_poly.py 
    RUN_TEST ibeis/tests/test_ibs_export.py 
    RUN_TEST ibeis/tests/test_ibs_detectimg_compute.py 
    RUN_TEST ibeis/tests/test_ibs_annotation_uuid.py 
    RUN_TEST ibeis/tests/test_delete_names.py 
    RUN_TEST ibeis/tests/test_delete_features.py 
    RUN_TEST ibeis/tests/test_delete_enc.py 
    RUN_TEST ibeis/tests/test_delete_image.py 
    RUN_TEST ibeis/tests/test_delete_annotation.py 
    RUN_TEST ibeis/tests/test_delete_image_thumbtups.py 
    RUN_TEST ibeis/tests/test_delete_annotation_chips.py 
    RUN_TEST ibeis/tests/test_delete_annotation_all.py 
    RUN_TEST ibeis/tests/test_delete_chips.py 
fi
#---------------------------------------------
#SQL TESTS
if [ "$SQL_TEST" = "ON" ] ; then
cat <<EOF
    ____ ____ _       ___ ____ ____ ___ ____ 
    [__  |  | |        |  |___ [__   |  [__  
    ___] |_\| |___     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/test_sql_numpy.py 
    RUN_TEST ibeis/tests/test_sql_modify.py 
    RUN_TEST ibeis/tests/test_sql_revert.py 
    RUN_TEST ibeis/tests/test_sql_names.py 
    RUN_TEST ibeis/tests/test_sql_ids.py 
    RUN_TEST ibeis/tests/test_sql_control.py 
fi
#---------------------------------------------
#VIEW TESTS
if [ "$VIEW_TEST" = "ON" ] ; then
cat <<EOF
    _  _ _ ____ _ _ _    ___ ____ ____ ___ ____ 
    |  | | |___ | | |     |  |___ [__   |  [__  
     \/  | |___ |_|_|     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/test_view_viz.py 
    RUN_TEST ibeis/tests/test_view_interact.py 
fi
#---------------------------------------------
#MISC TESTS
if [ "$MISC_TEST" = "ON" ] ; then
cat <<EOF
    _  _ _ ____ ____    ___ ____ ____ ___ ____ 
    |\/| | [__  |        |  |___ [__   |  [__  
    |  | | ___] |___     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/test_utool_parallel.py 
    RUN_TEST ibeis/tests/test_pil_hash.py 
fi
#---------------------------------------------
#OTHER TESTS
if [ "$OTHER_TEST" = "ON" ] ; then
cat <<EOF
    ____ ___ _  _ ____ ____    ___ ____ ____ ___ ____ 
    |  |  |  |__| |___ |__/     |  |___ [__   |  [__  
    |__|  |  |  | |___ |  \     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/assert_modules.py 
    RUN_TEST ibeis/tests/make_big_database.py 
    RUN_TEST ibeis/tests/test_qt_lazy.py 
    RUN_TEST ibeis/tests/test_qt_thumb.py 
    RUN_TEST ibeis/tests/sample_data.py 
    RUN_TEST ibeis/tests/reset_testdbs.py 
    RUN_TEST ibeis/tests/test_qt_tree.py 
    RUN_TEST ibeis/tests/test_hots_split_trees.py 
    RUN_TEST ibeis/tests/test_new_schema.py 
    RUN_TEST ibeis/tests/demo.py 
    RUN_TEST ibeis/tests/test_uuid_consistency.py 
fi
#---------------------------------------------
#HESAFF TESTS
if [ "$HESAFF_TEST" = "ON" ] ; then
cat <<EOF
    _  _ ____ ____ ____ ____ ____    ___ ____ ____ ___ ____ 
    |__| |___ [__  |__| |___ |___     |  |___ [__   |  [__  
    |  | |___ ___] |  | |    |        |  |___ ___]  |  ___]
EOF
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_draw_keypoint.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_pyhesaff_simple_parallel.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_cpp_rotation_invariance.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_adaptive_scale.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_ellipse.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_exhaustive_ori_extract.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_pyhesaff_simple_iterative.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_patch_orientation.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_pyhesaff.py 
fi

#---------------------------------------------
# END TESTING
END_TESTS
