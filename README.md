# IBEIS
## I.B.E.I.S. = Image Based Ecological Information System

=====

IBEIS is python module and standalone program for the storage and management of
images and derived data for use in compute vision algorithms. It aims to compute
who an animal is, what species an animal is, and where an animal is with the
ultimate goal being to ask important why biological questions. 

Currently the system is build around and SQLite database, a PyQt4 GUI, and
matplotlib visualizations. Algorithms employed are: random forest species
detection and localization, hessian-affine keypoint detection, SIFT keypoint
description, LNBNN identification using approximate nearest neighbors.
Algorithms in development are SMK (selective match kernel) for identifiaction
and deep neural networks for detection and localization. 

Documentation: http://erotemic.github.io/ibeis

Erotemic's IBEIS module dependencies 

* https://github.com/Erotemic/utool
  docs: http://erotemic.github.io/utool
* https://github.com/Erotemic/plottool
  docs: http://erotemic.github.io/plottool
* https://github.com/Erotemic/vtool
  docs: http://erotemic.github.io/vtool
* https://github.com/Erotemic/hesaff
  docs: http://erotemic.github.io/hesaff
* https://github.com/Erotemic/guitool
  docs: http://erotemic.github.io/guitool


bluemellophone's IBEIS modules

* https://github.com/bluemellophone/detecttools
* https://github.com/bluemellophone/pyrf
  docs: http://bluemellophone.github.io/pyrf


hjweide's IBEIS modules

* https://github.com/hjweide/pygist


The IBEIS module itself: 

* https://github.com/Erotemic/ibeis

# IBEIS Development Environment Setup 

```bash
# The following install script install ibeis and all dependencies. 
# If it doesnt you can look at the older instructions which follow
# and try to figure it out. After running this you should have a code
# directory with all of the above repos. 

# NOTE: IBEIS DEPENDS ON PYTHON 2.7. Unfortunately we are having problems moving to 3.

# Navigate to your code directory
export CODE_DIR=~/code
mkdir $CODE_DIR
cd $CODE_DIR

# Clone IBEIS
git clone https://github.com/Erotemic/ibeis.git
cd ibeis

# Generate the prereq install script (does not install anything)
# 
./_scripts/bootstrap.py

# Run the prereq install script (installs prereq libraries)
./_scripts/__install_prereqs__.sh

# Python repositories come with a standard setup.py script to help you install them
# Because IBEIS has several python repos, we created a super_setup script to help 
# you run the same command accross all IBIES repositories.

# Use super_setup.py to pull the latest and greatest from all the respos. 
# This will clone any dependency repos that do not exist.
./super_setup.py --pull

# Switch to current development branch
# (NOTICE: WE WILL SOON CHANGE THE DEV BRANCH TO NEXT)
./super_setup.py --checkout pyqt5 

# Run super_setup to build and install ibeis modules in development mode
# (the build flag builds any c++ files, and the develop flag installs a 
#  python module as a symbolic link to python's site-packages)
./super_setup.py --build --develop

# Usually this needs to be run twice because super_setup often needs to
# configure itself on the first run. (Either running it twice wont hurt)
./super_setup.py --build --develop

# Optional: download a test dataset
./dev.py -t mtest 

```

# Code Sytle Guidelines

For Python try to conform to pep8. 
You should set up your prefered editor to use flake8 as linter.
If using vim I recomend syntastic.

DISABLE THESE ERRORS 
* 'E127', # continuation line over-indented for visual indent
* 'E201', # whitespace after '('
* 'E202', # whitespace before ']'
* 'E203', # whitespace before ', '
* 'E221', # multiple spaces before operator
* 'E222', # multiple spaces after operator
* 'E241', # multiple spaces after ,
* 'E265', # block comment should start with "# "
* 'E271', # multiple spaces after keyword 
* 'E272', # multiple spaces before keyword
* 'E301', # expected 1 blank line, found 0
* 'E501', # > 79

flake8 --ignore=E127,E201,E202,E203,E221,E222,E241,E265,E271,E272,E301,E501 ~/code/ibeis

( Dev comment: my laptop seemst to report these flake8 errors while my desktops
  don't. I'm going to list errors that might need to be explicitly enabled here:

* 'F821',  # undefined name
* 'F403',  # import * used, unable to detect names

)

For C++ code use astyle to format your code:
atyle --style=ansi --indent=spaces --attach-inlines --indent-classes --indent-modifiers --indent-switches --indent-preproc-cond --indent-col1-comments --pad-oper --unpad-paren --delete-empty-lines --add-brackets 


# Updating Documentation
```bash
# utool script to run sphinx-apidoc
autogen_sphinx_docs.py
cp -r _doc/_build/html/* _page
git add _page/.nojekyll
git add _page/*
git add _page
git commit -m "updated docs"
git subtree push --prefix _page origin gh-pages
```

###


#### OLD Environment Setup:
```bash
# Navigate to your code directory
export CODE_DIR=~/code
cd $CODE_DIR

# Clone the IBEIS repositories 
git clone https://github.com/Erotemic/utool.git
git clone https://github.com/Erotemic/vtool.git
git clone https://github.com/Erotemic/plottool.git
git clone https://github.com/Erotemic/guitool.git
git clone https://github.com/Erotemic/hesaff.git
git clone https://github.com/Erotemic/ibeis.git
#
# Set the previous repos up for development by running
#
# > sudo python setup.py develop
#
# in each directory


# e.g.
sudo python $CODE_DIR/utool/setup.py develop
sudo python $CODE_DIR/vtool/setup.py develop
sudo python $CODE_DIR/hesaff/setup.py develop
sudo python $CODE_DIR/plottool/setup.py develop
sudo python $CODE_DIR/guitool/setup.py develop
sudo python $CODE_DIR/ibeis/setup.py develop


# Then clone these repos (these do not have setup.py files)
git clone https://github.com/bluemellophone/detecttools.git
git clone https://github.com/Erotemic/opencv.git
git clone https://github.com/Erotemic/flann.git
git clone https://github.com/bluemellophone/pyrf.git
# For repos with C++ code use the unix/mingw build script in the repo:
# e.g.
sudo ~/code/opencv/unix_build.sh
sudo ~/code/flann/unix_build.sh
sudo ~/code/pyrf/unix_build.sh

# If you want to train random forests with pyrf clone
# https://github.com/bluemellophone/IBEIS2014.git
# otherwise you dont need this
```


# Example usage

(Note: This list is far from complete)

```bash
#--------------------
# Main Commands
#--------------------
python main.py <optional-arguments> [--help]
python dev.py <optional-arguments> [--help]
# main is the standard entry point to the program
# dev is a more advanced developer entry point

# Useful flags.
# Read code comments in dev.py for more info.
# Careful some commands don't work. Most do.
# --cmd          # shows ipython prompt with useful variables populated
# -w, --wait     # waits (useful for showing plots)
# --gui          # starts the gui as well (dev.py does not show gui by default, main does)
# -t [test]


#--------------------
# PSA: Workdirs:
#--------------------
# IBEIS uses the idea of a work directory for databases.
# Use --set-workdir <path> to set your own, or a gui will popup and ask you about it
./main.py --set-workdir /raid/work --preload-exit
./main.py --set-logdir /raid/logs/ibeis --preload-exit

# use --db to specify a database in your WorkDir
# --setdb makes that directory your default directory
python dev.py --db <dbname> --setdb

# Or just use the absolute path
python dev.py --dbdir <full-dbpath>


#--------------------
# Examples:
# Here are are some example commands
#--------------------
# Run the queries for each roi with groundtruth in the PZ_Mothers database
# using the best known configuration of parameters
python dev.py --db PZ_Mothers --allgt -t best
python dev.py --db PZ_Mothers --allgt -t score


# View work dir
python dev.py --vwd --prequit

# List known databases
python dev.py -t list_dbs


# Dump/Print contents of params.args as a dict
python dev.py --prequit --dump-argv

# Dump Current SQL Schema to stdout 
python dev.py --dump-schema --postquit

#------------------
# Convert a hotspotter database to IBEIS
#------------------
# Set this as your workdir
python dev.py --db PZ_Mothers --setdb
# If its in the same location as a hotspotter db, convert it
python dev.py --convert --force-delete
python dev.py --convert --force-delete --db Database_MasterGrevy_Elleni
# Then interact with your new IBEIS database
python dev.py --cmd --gui 
> rid_list = ibs.get_valid_rids()

# Convinience: Convert ALL hotspotter databases
python dev.py -t convert_hsdbs --force-delete


python dev.py --convert --force-delete --db Frogs
python dev.py --convert --force-delete --db GIR_Tanya
python dev.py --convert --force-delete --db GZ_All
python dev.py --convert --force-delete --db Rhinos_Stewart
python dev.py --convert --force-delete --db WD_Siva
python dev.py --convert --force-delete --db WY_Toads
python dev.py --convert --force-delete --db WS_hard
python dev.py --convert --force-delete --db Wildebeast
python dev.py --convert --force-delete --db PZ_FlankHack
python dev.py --convert --force-delete --db PZ_Mothers


#--------------
# Run Result Inspection
#--------------
python dev.py --convert --force-delete --db Mothers --setdb
python dev.py --db Mothers --setdb
python dev.py --cmd --allgt -t inspect


#---------
# Ingest examples
#---------
# Ingest raw images
python ibeis/ingest/ingest_database.py --db JAG_Kieryn

# Ingest an hsdb
python ibeis/ingest/ingest_hsdb.py --db JAG_Kelly --force-delete

# Merge all jaguar databases into single big database
python main.py --merge-species JAG_


#---------
# Run Tests
#---------
./testsuit/run_tests.sh


#---------------
# Technical Demo
#---------------
python dev.py --db PZ_Mothers --setdb

# See a list of tests
python dev.py -t help

# See a plot of scores
python dev.py --allgt -t scores -w

# Run the best configuration and print out the hard cases
python dev.py --allgt -t best --echo-hardcase

# Look at inspection widget
python dev.py --allgt -t inspect -w


#----------------
# Profiling Code
#----------------

profiler.sh dev.py -t best --db testdb1 --allgt --nocache-query --prof-mod "spatial;linalg;keypoint"
profiler.sh dev.py -t best --db PZ_Mothers --all --nocache-query --prof-mod "spatial;linalg;keypoint"


#----------------
# Test Commands
#----------------
# Set a default DB First
./dev.py --setdb --dbdir /path/to/your/DBDIR
./dev.py --setdb --db YOURDB
./dev.py --setdb --db PZ_Mothers
./dev.py --setdb --db PZ_FlankHack

# List all available tests
./dev.py -t help
# Minimal Database Statistics
./dev.py --allgt -t info
# Richer Database statistics
./dev.py --allgt -t dbinfo
# Print algorithm configurations
./dev.py -t printcfg
# Print database tables
./dev.py -t tables
# Print only the image table
./dev.py -t imgtbl
# View data directory in explorer/finder/nautilus
./dev.py -t vdd


# List all IBEIS databases
./dev.py -t listdbs
# List unconverted hotspotter databases in your workdir
./dev.py -t listhsdbs
# Delete cache
./dev.py -t delete_cache

# Plot of chip-to-chip scores
./dev.py --allgt -t scores -w

# Plot of keypoint-to-keypoint distances
./dev.py --allgt -t dists -w

# See how scores degrade as database size increases
./dev.py --allgt -t upsize -w



# Show Annotation Chips 1, 3, 5, and 11
./dev.py -t show --qaid 1 3 5 11 -w
# Query Annotation Chips 1, 3, 5, and 11
./dev.py -t query --qaid 1 3 5 11 -w
# Inspect spatial verification of annotations 1, 3, 5, and 11
./dev.py -t sver --qaid 1 3 5 11 -w
# Compare matching toggling the gravity vector
./dev.py -t gv --qaid 1 11 -w


# Database Stats for all our important datasets:
./dev.py --allgt -t dbinfo --db PZ_RoseMary | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db PZ_Mothers | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db PZ_FlankHack | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db OP_Trip14_Encounter-80_nImg=555 | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db OP_Trip14_Encounter-224_nImg=222 | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db OP_Trip14 | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db GZ_ALL | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db GZ_Siva | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db MISC_Jan12 | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db GIR_Tanya | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db LF_Bajo_bonito | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db LF_WEST_POINT_OPTIMIZADAS | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db LF_OPTIMIZADAS_NI_V_E | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db Rhinos_Stewart | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db Elephants_Stewart | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db WY_Toads | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db Frogs | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db Wildebeest | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db Seals | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db JAG_Kelly | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db JAG_Kieryn | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db polar_bears | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db snails_drop1 | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db WD_Siva | grep -F "[dbinfo]"

python dev.py --dbdir /raid/work2/MBB_Grevys/GZ_Archetype_Appends_4GB -t dbinfo | grep -F "[dbinfo]"
python dev.py --dbdir /raid/work2/MBB_Grevys/GZ_Elleni_16GB -t dbinfo | grep -F "[dbinfo]"
python dev.py --dbdir /raid/work2/MBB_Grevys/GZ_Archetype_Laikipia_5GB -t dbinfo | grep -F "[dbinfo]"
python dev.py --dbdir /raid/work2/MBB_Grevys/GZ_3D_Encounters_group1 -t dbinfo | grep -F "[dbinfo]"
python dev.py --dbdir /raid/work2/MBB_Grevys/GZ_3D_Encounters_group2 -t dbinfo | grep -F "[dbinfo]"
for i in $(/bin/ls /raid/work2/MBB_Grevys/GZ_EncounterGroups); do
    python dev.py --dbdir /raid/work2/MBB_Grevys/GZ_EncounterGroups/$i -t dbinfo | grep -F "[dbinfo]"
done

python dev.py --dbdir /raid/work2/DanPrinctonDrive/HSDB_pztest2 -t dbinfo | grep -F "[dbinfo]"
python dev.py --dbdir /raid/work2/DanPrinctonDrive/elephants-dan-princton-drive-march-2014 -t dbinfo | grep -F "[dbinfo]"

# Some mass editing of metadata
./dev.py --db PZ_FlankHack --edit-notes
./dev.py --db GZ_Siva --edit-notes
./dev.py --db GIR_Tanya --edit-notes
./dev.py --allgt -t dbinfo --db Elephants_Stewart --set-all-species elephant_savanna
./dev.py --allgt -t dbinfo --db polar_bears --set-all-species bear_polar
./dev.py --allgt -t dbinfo --db GZ_ALL --set-all-species zebra_grevys
./dev.py --allgt -t dbinfo --db PZ_FlankHack --set-all-species zebra_plains
./dev.py --allgt -t dbinfo --db GIR_Tanya --set-all-species giraffe
./dev.py --allgt -t dbinfo --db LF_Bajo_bonito --set-all-species lionfish
./dev.py --allgt -t dbinfo --db LF_WEST_POINT_OPTIMIZADAS --set-all-species lionfish
./dev.py --allgt -t dbinfo --db LF_OPTIMIZADAS_NI_V_E --set-all-species lionfish
./dev.py --allgt -t dbinfo --db JAG_Kelly --set-all-species jaguar
./dev.py --allgt -t dbinfo --db JAG_Kieryn --set-all-species jaguar
./dev.py --allgt -t dbinfo --db Wildebeest --set-all-species wildebeest

# Current Experiments:

profiler.sh dev.py -t upsize --allgt --quiet --noshow

python dev.py -t upsize --db PZ_Mothers --qaid 1:30:3 -w


dev.py -t upsize --allgt --quiet --noshow
profiler.sh dev.py -t upsize --quiet --db PZ_Mothers --qaid 1:30
python dev.py -t upsize --quiet --db PZ_Mothers --qaid 1:30
python dev.py -t upsize --quiet --db PZ_Mothers --allgt -w

%run dev.py -t upsize --quiet --db PZ_Mothers --allgt -w
python dev.py -t upsize --quiet --db PZ_Mothers --allgt -w
python dev.py -t upsize --quiet --db PZ_Mothers --qaid 1:10:3 -w


profiler.sh dev.py -t best --allgt --db PZ_Mothers --nocache-big --nocache-query
./dev.py -t best --qaid 1:10 --db PZ_Mothers --nocache-big --nocache-query

./main.py --db PZ_RoseMary --cmd


# Cyth issue debug
python dev.py --db testdb1 --delete-cache
python dev.py --db testdb1 --query 1 --nocache-query --cyth --gui
python dev.py --db testdb1 --query 1 --nocache-query --nocyth --gui

# Rosemary Tests
python dev.py --db PZ_RoseMary -t best --gt 1:20 --screen --cyth 
python dev.py --db PZ_RoseMary -t best --allgt --screen --cyth 
python dev.py --db PZ_RoseMary -t upsize --allgt --screen --cyth

# Ensures the mtest dataset is in your workdir (by downloading it from dropbox)
./dev.py -t mtest 

# Cyth timings (screen disables the backspaces in progress)
./dev.py --db PZ_MTEST -t best --qaid 3:60:3 --nocache-query --screen --cyth --quiet
./dev.py --db PZ_MTEST -t best --qaid 3:60:3 --nocache-query --screen --nocyth --quiet

./dev.py --db PZ_MTEST -t best --qaid 3:60:3 --nocache-query --cyth --quiet
./dev.py --db PZ_MTEST -t best --qaid 3:60:3 --nocache-query --nocyth --quiet

./dev.py --db PZ_MTEST -t best --qaid 3:60:3 --nocache-query --cyth 
./dev.py --db PZ_MTEST -t best --qaid 3:60:3 --nocache-query --nocyth

./dev.py --db PZ_MTEST -t best --allgt --nocyth
 
# Correct output 
python dev.py --db PZ_Mothers -t best --qaid 1:20
python dev.py --db PZ_Mothers -t upsize --allgt --screen --cyth
python dev.py --db PZ_RoseMary -t upsize --allgt --screen --cyth

# EXPERIMENT DATABASES
python dev.py --db testdb1 --setdb 
python dev.py --db PZ_Mothers --setdb 
python dev.py --db PZ_RoseMary --setdb 
# EXPERIMENT CLEANUP
python dev.py --delete-cache --postload-exit
# EXPERIMENT PARTIAL COMMANDS
python dev.py -t best --qaids 27:110
python dev.py -t best --allgt --index 20:100
python dev.py -t best --allgt --echo-hardcase
python dev.py -t best --allgt --index 24 39 44 45 --view-hard  --sf
python dev.py -t best --allgt --view-hard --vdd
# EXPERIMENT FULL COMMANDS
python dev.py -t best --allgt --view-hard 
python dev.py -t upsize --allgt 


profiler.sh dev.py --prof-mod smk_,pandas_helpers,hstypes -t asmk --allgt --index 0:20 --db PZ_Mothers --nocache-big --nocache-query --nocache-save
profiler.sh dev.py --prof-mod smk_,pandas_helpers,hstypes -t smk --allgt --index 0:20 --db PZ_Mothers --nocache-big --nocache-query --nocache-save
./dev.py -t smk --allgt --db PZ_Mothers --nocache-big --nocache-query --index 0:20
./dev.py -t asmk --allgt --db PZ_Mothers --nocache-big --nocache-query --index 0:20

dev.py -t smk_test --allgt --db PZ_Mothers --nocache-big --nocache-query --index 0:20


./dev.py -t smk2 --allgt --db PZ_MTEST --nocache-big --nocache-query


./dev.py -t smk1 --allgt --index 0:2 --db Oxford

# SMK TESTS
python dev.py -t smk2 --allgt --db PZ_Mothers --nocache-big --nocache-query --index 0:20
python dev.py -t smk2 --allgt --db GZ_ALL --nocache-big --nocache-query --index 0:20

python dev.py -t smk2 --allgt --db PZ_Mothers --index 20:30 --va
python dev.py -t smk2 --allgt --db PZ_Master0

python -m memory_profiler dev.py -t smk2 --allgt --db PZ_Mothers --index 0

python -m memory_profiler dev.py -t smk --allgt --db PZ_Master0 --index 0 --nocache-query --nogui 

python dev.py -t smk_64k --allgt --db PZ_Master0
python dev.py -t smk_128k --allgt --db PZ_Master0

python dev.py -t oxford --allgt --db Oxford --index 0:55
 

# Feature Tuning
python dev.py -t test_feats -w --show --db PZ_Mothers --allgt --index 1:2

python dev.py -t featparams -w --show --db PZ_Mothers --allgt
python dev.py -t featparams_big -w --show --db PZ_Mothers --allgt
python dev.py -t featparams_big -w --show --db GZ_ALL --allgt
 --allgt --index 1:2

# SETTING HARD CASES
python dev.py -t best --db PZ_MTEST --allgt --echo-hardcase
python dev.py -t best --db PZ_MTEST --allgt --echo-hardcase --set-aids-as-hard [copypaste from output]  # sorry about this step...
# EG: dev.py -t best --db PZ_MTEST --allgt --echo-hardcase --set-aids-as-hard 27 28 44 49 50 51 53 54 66 72 89 97 110  # Hard as of 2014-11-4
# EG: dev.py -t best --db PZ_MTEST --allgt --echo-hardcase --set-aids-as-hard 27 44 49 50 51 53 54 66 69 89 97 110  # Hard as of 2014-11-6
# EG: dev.py -t best --db PZ_MTEST --allgt --echo-hardcase --set-aids-as-hard 27 43 45 49 51 53 54 66 97  # FGWEIGHT Hard as of 2014-11-6



python dev.py -t best --db PZ_MTEST --allhard

# View all hard cases
python dev.py -t best --db PZ_MTEST --allhard --vh --va

# 72 is a great testcase
python dev.py -t best --db PZ_MTEST --qaid 72 --sel-rows 0 --sel-cols 0 --show -w --dump-extra --vf --va

python dev.py -t best --db PZ_MTEST --qaid 72 --sel-rows 0 --sel-cols 0 --show -w --dump-extra --vf --va

# VSONE TESTS
python dev.py -t vsone_best --db PZ_Mothers --allgt --index 0:2 --print-all --va

# DOCTESTS
TODO: make these tests work
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.preproc.preproc_chip))" --quiet
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.preproc.preproc_featweight))" --quiet

```

#---------------
# Caveats / Things we are not currently doing

* No orientation invariance, gravity vector is always assumed
* We do not add or remove points from kdtrees. They are always rebuilt
* Feature weights are never recomputed unless the database cache is deleted

