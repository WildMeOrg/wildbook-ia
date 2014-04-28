export AUG=""
#export AUG="--cmd"$AUG
export AUG="--batch --allgt --print-rankmat --print-rowlbl --print-rowscore --print-hardcase --echo-hardcase"
export TESTS="-t gv_test"
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
