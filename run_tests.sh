export ARGV="--quiet$@"

export GUI_TESTS=ON
export IBS_TESTS=ON
export SQL_TESTS=ON

echo "<<<<<<<<<<<<<<<<<<<<<<<<<< BEGIN: ARGV=$ARGV"
echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#---------------------------------------------
# GUI_TESTS
if [ "$GUI_TESTS" = "ON" ] ; then 
    echo
    python tests/test_gui_add_roi.py $ARGV
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    python tests/test_gui_selection.py $ARGV
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    python tests/test_gui_import_images.py $ARGV
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

fi
#---------------------------------------------
# IBEIS TESTS
if [ "$IBS_TESTS"="ON" ] ; then 
    echo
    python tests/test_ibs.py $ARGV
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    python tests/test_add_images.py $ARGV
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    python tests/test_parallel.py $ARGV
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
fi
#---------------------------------------------
# SQL TESTS
if [ "$SQL_TESTS" = "ON" ] ; then 
    echo
    python tests/test_sqldatabase_control.py $ARGV
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
fi
#---------------------------------------------
echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<DONE"
