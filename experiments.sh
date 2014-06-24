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
    python dev.py --db GZ -t best --echo-hardcase --qrid 95 112 140 183 184 253 255 289 306 316 339 340 430 441 443 444 445 446 450 451 453 454 456 460 463 465 534 550 802 803 941 1047 --save-figures -c 0
}

SNAILS()
{
    # Run the best config
    python dev.py --db snails_drop1 -t best --allgt --echo-hardcase
    # Find the hard cases
    export HARD_SNAILS="--qrid 1 14 15 16 17 31 32 34 35 36 37 38 40 45 52 53 57 59 62 63 68 72 73 75 77"
    # Visualize hard cases
    python dev.py --db snails_drop1 -t best $HARD_SNAILS --echo-hardcase --save-figures -c 0
    python dev.py --db snails_drop1 -t best --qrid 1 14 --echo-hardcase --save-figures -c 0
}

MOTHERS()
{
    python dev.py --db PZ_MOTHERS -t best --allgt --echo-hardcase
    python dev.py --db PZ_MOTHERS -t gv_test --qrid 49 --save-figures -c 0 1
    python dev.py --db PZ_MOTHERS -t gv_test --allgt
    python dev.py --db PZ_MOTHERS -t gv_test --qrid 27 28 45 71 90 109 --print-all
    python dev.py --db PZ_MOTHERS -t gv_test --qrid 49 --save-figures -c 0 1 2 
}


TECHINCAL_DEVELOPER_DEMO()
{
    # Make a code directory
    export CODE_DIR=~/code
    mkdir $CODE_DIR
    cd $CODE_DIR

    # Download the modules
    git clone https://github.com/Erotemic/utool.git
    git clone https://github.com/Erotemic/vtool.git
    git clone https://github.com/Erotemic/plottool.git
    git clone https://github.com/Erotemic/guitool.git
    git clone https://github.com/Erotemic/hesaff.git
    git clone https://github.com/Erotemic/ibeis.git
    git clone https://github.com/bluemellophone/detecttools.git
    git clone https://github.com/Erotemic/opencv.git
    git clone https://github.com/Erotemic/flann.git
    git clone https://github.com/bluemellophone/pyrf.git
    
    # Install the modules
    sudo $CODE_DIR/opencv/unix_opencv_build.sh
    sudo $CODE_DIR/flann/unix_opencv_build.sh   # OpenCV 2.4.8
    sudo $CODE_DIR/hesaff/unix_hesaff_build.sh
    sudo $CODE_DIR/pyrf/unix_pyrf_build.sh
    sudo python $CODE_DIR/utool/setup.py develop
    sudo python $CODE_DIR/vtool/setup.py develop
    sudo python $CODE_DIR/hesaff/setup.py develop
    sudo python $CODE_DIR/plottool/setup.py develop
    sudo python $CODE_DIR/guitool/setup.py develop
    sudo python $CODE_DIR/ibeis/setup.py develop

    # Get ready to develop
    cd $CODE_DIR/ibeis

    # -----
    # Now that you have IBEIS
    # Download test databases
    sh reset_dbs.sh

    # Set a workdir
    python dev.py --set-workdir F:/data
    # View workdir
    python dev.py --vwd
    # Set a database as your default
    python dev.py --db testdb1 --setdb

    # List the databases available to you
    python dev.py -t list_dbs

    #
    #
    # -----
    # Now that you have a DATA
    # Set your database
    python dev.py --db PZ_MOTHERS --setdb
    python dev.py --db GZ_ALL --setdb

    # Database info
    python dev.py -t dbinfo info --allgt

    # Find hard cases
    python dev.py -t best --allgt --echo-hardcase -w
    # Show roi-match scores
    python dev.py -t scores --allgt -w
    # Show desc-match scores
    python dev.py -t dists --allgt -w
    # Find 
    python dev.py -t scores --allgt -w

    # Info
    python dev.py -t sver --qrids 1 -w
    python dev.py -w -qrid 1 -t gvcomp
    python dev.py -w -h 1 -t sver

    #
    # -----
    # Now that you have your KNOWLEDGE
    # Develop

    python dev.py --help

}
