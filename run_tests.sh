export ARGV="--quiet $@"

export GUI_TESTS=ON
export IBS_TESTS=ON
export SQL_TESTS=ON

#echo <<EOF

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

RUN_TEST()
{
    echo "RUN_TEST: $1"
    python $1 $ARGV
    export RETURN_CODE=$?
    PRINT_DELIMETER
    num_passed=$(($num_passed + (1 - $RETURN_CODE)))
    num_ran=$(($num_ran + 1))
}


#---------------------------------------------
# GUI_TESTS
if [ "$GUI_TESTS" = "ON" ] ; then 
cat <<EOF
   ___ _   _ ___   _____ ___ ___ _____ ___ 
  / __| | | |_ _| |_   _| __/ __|_   _/ __|
 | (_ | |_| || |    | | | _|\\__ \\ | | \\__ \\
  \\___|\\___/|___|   |_| |___|___/ |_| |___/
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
  ___ ___ ___ ___ ___   _____ ___ ___ _____ ___ 
 |_ _| _ ) __|_ _/ __| |_   _| __/ __|_   _/ __|
  | || _ \\ _| | |\\__ \\   | | | _|\\__ \\ | | \\__ \\
 |___|___/___|___|___/   |_| |___|___/ |_| |___/
EOF
    
    RUN_TEST ibeis/tests/test_ibs.py

    RUN_TEST ibeis/tests/test_add_images.py

    RUN_TEST ibeis/tests/test_parallel.py

    RUN_TEST ibeis/tests/test_chip_compute.py

    RUN_TEST ibeis/tests/test_query.py

    RUN_TEST ibeis/tests/test_ibs_getters.py

    RUN_TEST ibeis/tests/test_viz.py

    RUN_TEST ibeis/tests/test_interact.py
fi


#---------------------------------------------
# SQL TESTS
if [ "$SQL_TESTS" = "ON" ] ; then 
cat <<EOF
  ___  ___  _      _____ ___ ___ _____ ___ 
 / __|/ _ \\| |    |_   _| __/ __|_   _/ __|
 \\__ \\ (_) | |__    | | | _|\\__ \\ | | \\__ \\
 |___/\\__\\_\\____|   |_| |___|___/ |_| |___/
EOF

    RUN_TEST ibeis/tests/test_sql_numpy.py 

    RUN_TEST ibeis/tests/test_sql_names.py 

fi


#---------------------------------------------
echo "RUN_TESTS: DONE"

echo "$num_passed / $num_ran tests passed"
