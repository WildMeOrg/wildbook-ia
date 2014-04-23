"""
Runs all tests

Bubble text from:
http://patorjk.com/software/taag/#p=display&f=Cybermedium&t=VTOOL%20TESTS
"""
# TODO: MAKE SURE IBS DATABASE CAN HANDLE WHEN IMAGE PATH IS NOT WHERE IT EXPECTED
# TODO: ADD CACHE / LOCALIZE IMAGES IN IBEIS CONTROL

# Win32 path hacks
export CWD=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CWD

export ARGV="--quiet --noshow $@"

export DEFAULT=ON

export GUI_TESTS=$DEFAULT
export IBS_TESTS=$DEFAULT
export SQL_TESTS=$DEFAULT
export MISC_TESTS=$DEFAULT
export VIEW_TESTS=$DEFAULT
export VTOOL_TESTS=$DEFAULT



PRINT_DELIMETER(){
    printf "\n#\n#\n#>>>>>>>>>>> next_test\n\n"
}

cat <<EOF
.______       __    __  .__   __.    .___________. _______     _______.___________.    _______.
|   _  \     |  |  |  | |  \ |  |    |           ||   ____|   /       |           |   /       |
|  |_)  |    |  |  |  | |   \|  |    '---|  |----'|  |__     |   (----'---|  |----'  |   (----'
|      /     |  |  |  | |  . '  |        |  |     |   __|     \   \       |  |        \   \    
|  |\  \----.|  '--'  | |  |\   |        |  |     |  |____.----)   |      |  |    .----)   |   
| _| '._____| \______/  |__| \__|        |__|     |_______|_______/       |__|    |_______/    
EOF

echo "BEGIN: ARGV=$ARGV"
PRINT_DELIMETER

num_passed=0
num_ran=0

export FAILED_TESTS=''

RUN_TEST()
{
    echo "RUN_TEST: $@"
    export TEST="python $@ $ARGV"
    $TEST
    export RETURN_CODE=$?
    PRINT_DELIMETER
    num_passed=$(($num_passed + (1 - $RETURN_CODE)))
    num_ran=$(($num_ran + 1))

    if [ "$RETURN_CODE" != "0" ] ; then
        export FAILED_TESTS="$FAILED_TESTS\n$TEST"
    fi

}


RUN_TEST ibeis/tests/assert_modules.py 


#---------------------------------------------
# VTOOL TESTS
if [ "$VTOOL_TESTS" = "ON" ] ; then 
cat <<EOF
    _  _ ___ ____ ____ _       ___ ____ ____ ___ ____ 
    |  |  |  |  | |  | |        |  |___ [__   |  [__  
     \/   |  |__| |__| |___     |  |___ ___]  |  ___] 
EOF

    RUN_TEST vtool/tests/test_draw_keypoint.py --noshow 

    RUN_TEST vtool/tests/test_spatial_verification.py --noshow 

    RUN_TEST vtool/tests/test_exhaustive_ori_extract.py --noshow 

    RUN_TEST vtool/tests/test_vtool.py 

fi

#---------------------------------------------
# GUI_TESTS
if [ "$GUI_TESTS" = "ON" ] ; then 
cat <<EOF
    ____ _  _ _    ___ ____ ____ ___ ____ 
    | __ |  | |     |  |___ [__   |  [__  
    |__] |__| |     |  |___ ___]  |  ___] 
EOF

    RUN_TEST ibeis/tests/test_gui_import_images.py 

    RUN_TEST ibeis/tests/test_gui_add_roi.py 

    RUN_TEST ibeis/tests/test_gui_selection.py 

    RUN_TEST ibeis/tests/test_gui_open_database.py

    RUN_TEST ibeis/tests/test_gui_all.py
fi


#---------------------------------------------
# IBEIS TESTS
if [ "$IBS_TESTS" = "ON" ] ; then 
cat <<EOF
    _ ___  ____ _ ____    ___ ____ ____ ___ ____ 
    | |__] |___ | [__      |  |___ [__   |  [__  
    | |__] |___ | ___]     |  |___ ___]  |  ___] 
EOF
    
    RUN_TEST ibeis/tests/test_ibs.py

    RUN_TEST ibeis/tests/test_ibs_add_images.py

    RUN_TEST ibeis/tests/test_ibs_chip_compute.py

    RUN_TEST ibeis/tests/test_ibs_query.py

    RUN_TEST ibeis/tests/test_ibs_query_components.py

    RUN_TEST ibeis/tests/test_ibs_getters.py
fi


#---------------------------------------------
# VIEW TESTS
if [ "$VIEW_TESTS" = "ON" ] ; then 
cat <<EOF
    _  _ _ ____ _ _ _    ___ ____ ____ ___ ____ 
    |  | | |___ | | |     |  |___ [__   |  [__  
     \/  | |___ |_|_|     |  |___ ___]  |  ___] 
EOF

    RUN_TEST ibeis/tests/test_view_viz.py

    RUN_TEST ibeis/tests/test_view_interact.py
fi

#---------------------------------------------
# MISC TESTS
if [ "$MISC_TESTS" = "ON" ] ; then 
cat <<EOF
    _  _ _ ____ ____    ___ ____ ____ ___ ____ 
    |\/| | [__  |        |  |___ [__   |  [__  
    |  | | ___] |___     |  |___ ___]  |  ___] 
EOF
    
    RUN_TEST ibeis/tests/test_utool_parallel.py
fi


#---------------------------------------------
# SQL TESTS
if [ "$SQL_TESTS" = "ON" ] ; then 
cat <<EOF
    ____ ____ _       ___ ____ ____ ___ ____ 
    [__  |  | |        |  |___ [__   |  [__  
    ___] |_\| |___     |  |___ ___]  |  ___] 
EOF

    RUN_TEST ibeis/tests/test_sql_numpy.py 

    RUN_TEST ibeis/tests/test_sql_names.py 

fi


#---------------------------------------------
echo "RUN_TESTS: DONE"

if [ "$FAILED_TESTS" != "" ] ; then
    echo "-----"
    printf "Failed Tests:" 
    printf "$FAILED_TESTS\n"
    echo "-----"
fi

echo "$num_passed / $num_ran tests passed"

