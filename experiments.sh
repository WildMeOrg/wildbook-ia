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
    python dev.py --db GZ -t best --allgt --echo-hardcase
    export HARD_GZS="--qrid 69 112 140 178 183 184 197 247 253 255 276 289 306 316 317 326 339 340 369 389 423 430 441 443 444 445 446 450 451 453 454 456 460 463 465 466 494 501 509 534 542 546 550 553 556 619 631 666 681 684 730 786 999 1014 1045"
    python dev.py --db GZ -t best --echo-hardcase --qrid 27 95 112 140 183 184 253 255 289 306 316 339 340 430 441 443 444 445 446 450 451 453 454 456 460 463 465 534 550 619 802 803 838 941 981 1014 1040 1047
}

SNAILS()
{
    # Run the best config
    python dev.py --db snails_drop1 -t best --allgt --echo-hardcase
    # Find the hard cases
    export HARD_SNAILS="--qrid 1 14 15 16 17 31 32 34 35 36 37 38 40 45 52 53 57 59 62 63 68 72 73 75 77"
    # Visualize hard cases
    python dev.py --db snails_drop1 -t best $HARD_SNAILS --echo-hardcase
}
