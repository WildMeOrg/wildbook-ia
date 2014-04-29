export TESTS=""
#export TESTS="-t gv_test"
export AUG=""
export AUG="--num-proc 1"
export AUG="$AUG --allgt"
#export AUG="$AUG --cmd"
#export AUG="--print-rankmat --print-rowlbl --print-rowscore --print-hardcase --echo-hardcase"
export DEVPY="python dev.py --db testdb1 --setdb $AUG $TESTS"
$DEVPY $@

RESET(){
    sh reset_dbs.sh
}


UNUSED() {
    #python dev.py --db PZ_MOTHERS -t gv_test
    python dev.py --db PZ_MOTHERS -t tables


    python dev.py --db PZ_MOTHERS -t gv_test --print-rowscore --print-collbl --print-rowlbl
}
