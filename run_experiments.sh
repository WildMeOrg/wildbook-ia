#
#-----------------
# DEFAULT DATABASE 
#-----------------
export DEFDB="--db testdb1 --setdb"

#
#-----------------
# DEFINE TESTS 
#-----------------
export TESTS=""
#export TESTS="-t gv_test"

#
#-----------------
# AUGMENT
#-----------------
export AUG=""
export AUG="$AUG --allgt"
export AUG="$AUG --devmode"
#export AUG="$AUG --num-proc 1"
#export AUG="$AUG --cmd"
#export AUG="$AUG --print-rankmat --print-rowlbl --print-rowscore --print-hardcase --echo-hardcase"

#
#-----------------
# BUILD DEVPY CMD
#-----------------
export DEVPY="python dev.py $DEFDB $AUG $TESTS"

#
#-----------------
# RUN DEVPY CMD
#-----------------
$DEVPY $@

#_______________________
# ARCHIVED FUNCS / NOTES
RESET(){
    sh reset_dbs.sh
}


UNUSED() {
    #python dev.py --db PZ_MOTHERS -t gv_test
    python dev.py --db PZ_MOTHERS -t tables
    python dev.py --db PZ_MOTHERS -t gv_test --print-rowscore --print-collbl --print-rowlbl
}

GZ_HARD()
{
    --qrid 69 112 140 178 183 184 197 247 253 255 276 289 306 316 317 326 339 340 369 389 423 430 441 443 444 445 446 450 451 453 454 456 460 463 465 466 494 501 509 534 542 546 550 553 556 619 631 666 681 684 730 786 999 1014 1045
}

SNAILS()
{
    ./dev.py --db snails_drop1 -t best --allgt --echo-hardcase
}
