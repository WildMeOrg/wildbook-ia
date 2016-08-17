![Logo](_installers/WildBook_logo_72dpi-03.png)

NOTICE: The main github repo has moved to https://github.com/WildbookOrg/ibeis

# IBEIS - Image Analysis

## I.B.E.I.S. = Image Based Ecological Information System

=====

![alt text](http://i.imgur.com/TNCiEBe.png "(Note: the rhino and wildebeest mathces may be dubious. Other species do work well though")

# Program Description

IBEIS program for the storage and management of images and derived data for
use in computer vision algorithms. It aims to compute who an animal is, what
species an animal is, and where an animal is with the ultimate goal being to
ask important why biological questions.  This This repo Image Analysis image
analysis module of IBEIS. It is both a python module and standalone program. 

Currently the system is build around and SQLite database, a PyQt4 GUI, and
matplotlib visualizations. Algorithms employed are: random forest species
detection and localization, hessian-affine keypoint detection, SIFT keypoint
description, LNBNN identification using approximate nearest neighbors.
Algorithms in development are SMK (selective match kernel) for identification
and deep neural networks for detection and localization. 

The core of IBEIS is the IBEISController class. It provides an API into IBEIS
data management and algorithms. The IBEIS API Documentation can be found here:
 http://erotemic.github.io/ibeis

The IBEIS GUI (graphical user interface) is built on top of the API. 
We are also experimenting with a new web frontend that bypasses the older GUI code.

## Self Installing Executables:

Unfortunately we have not released self-installing-executables for IBEIS yet. 
We plan to release these "soon". 

However there are old HotSpotter (the software which IBEIS is based on)
binaries available. These can be downloaded from: http://cs.rpi.edu/hotspotter/

# Visual Demo
![alt text](http://i.imgur.com/QWrzf9O.png "Feature Extraction")
![alt text](http://i.imgur.com/iMHKEDZ.png "Nearest Neighbors")

### Match Scoring 
![alt text](http://imgur.com/Hj43Xxy.png "Match Inspection")

### Spatial Verification
![alt text](http://i.imgur.com/VCz0j9C.jpg "sver")
```bash
python -m vtool.spatial_verification --test-spatially_verify_kpts
```

### Name Scoring
![alt text](http://i.imgur.com/IDUnxu2.jpg "namematch")
```bash
python -m ibeis.algo.hots.chip_match --exec-show_single_namematch --qaid 1 --show
```

### Identification Ranking 
![alt text](http://i.imgur.com/BlajchI.jpg "rankedmatches")
```bash
python -m ibeis.algo.hots.chip_match --exec-show_ranked_matches --show --qaid 86
```

### Inference
![alt text](http://i.imgur.com/RYeeENl.jpg "encgraph")
```bash
python -m ibeis.algo.preproc.preproc_encounter --exec-compute_encounter_groups --show
```

# Internal Modules

In the interest of modular code we are actively developing several different modules. 


Erotemic's IBEIS Image Analysis module dependencies 

* https://github.com/WildbookOrg/utool
  docs: http://erotemic.github.io/utool
* https://github.com/WildbookOrg/plottool
  docs: http://erotemic.github.io/plottool
* https://github.com/WildbookOrg/vtool
  docs: http://erotemic.github.io/vtool
* https://github.com/WildbookOrg/hesaff
  docs: http://erotemic.github.io/hesaff
* https://github.com/WildbookOrg/guitool
  docs: http://erotemic.github.io/guitool


bluemellophone's IBEIS Image Analysis modules

* https://github.com/WildbookOrg/detecttools
* https://github.com/WildbookOrg/pyrf
  docs: http://bluemellophone.github.io/pyrf


The IBEIS module itself: 

* https://github.com/WildbookOrg/ibeis

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
git clone https://github.com/WildbookOrg/ibeis.git
cd ibeis

# Generate the prereq install script (does not install anything)
# 
./_scripts/bootstrap.py
or 
./super_setup.py --bootstrap

# Ensure all python dependencies have been installed
pip install -r requirements.txt
pip install -r optional-requirements.txt

# Run the prereq install script (installs prereq libraries)
./_scripts/__install_prereqs__.sh

# Python repositories come with a standard setup.py script to help you install them
# Because IBEIS has several python repos, we created a super_setup script to help 
# you run the same command accross all IBIES repositories.

# Use super_setup.py to pull the latest and greatest from all the respos. 
# This will clone any dependency repos that do not exist.
./super_setup.py pull

# Ensure you are using WildMe repos
./super_setup.py move-wildme

# Switch to current development branch
./super_setup.py checkout next 

# Run super_setup to build and install ibeis modules in development mode
# (the build flag builds any c++ files, and the develop flag installs a 
#  python module as a symbolic link to python's site-packages)
./super_setup.py build develop

# Usually this needs to be run twice because super_setup often needs to
# configure itself on the first run. (Either running it twice wont hurt)
./super_setup.py build develop

# Optional: set a workdir and download a test dataset
./dev.py --set-workdir ~/data/work --preload-exit
./dev.py -t mtest 
./dev.py -t nauts 
./reset_dbs.py


# make sure everyhing is set up correctly
./assert_modules.sh
```


# Running Tests

NOTE: Make sure whatever editor you are using can perform syntax highlighting on
doctests. The majority of the tests are written in a doctest-like format.  

There are two testing scripts:

run_tests.py

run_tests.py performs only a subset of doctests which are not labeled as SLOW. 

this allows for me to have a high confidence that I'm not breaking things
while also allowing for a high throughput. Because run_tests.py is a python
script only one instance of python is started and all tests are run from that. 
This means that the controller is not reloaded for each individual test.

run_test.py --testall will test all enabled doctests including slow ones. This
adds about a minute onto the runtime of the tests.

A text file records any test that fails as well as all test times.

Tests can easily be run individually using documentation found in each module
with doctests.

The following examples runs the 1st doctest belonging to the function (or class)
baseline_neighbor_filter in the module ibeis.algo.hots.pipeline:

    python -m ibeis.algo.hots.pipeline --test-baseline_neighbor_filter:0
    

# Code Sytle Guidelines

For Python try to conform to pep8. 
You should set up your preferred editor to use flake8 as linter.
If using vim I recommend syntastic.

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
* 'N802', # function name should be lowercase
* 'N803', # argument name should be lowercase
* 'N805', # first argument of a method should be named 'self'
* 'N806', # variable in function should be lowercase

flake8 --ignore=E127,E201,E202,E203,E221,E222,E241,E265,E271,E272,E301,E501,N802,N803,N805,N806 ~/code/ibeis

( Dev comment: my laptop seems to report these flake8 errors while my desktops
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
mkdir _page
cp -r _doc/_build/html/* _page
touch _page/.nojekyll
git add _page/.nojekyll
git add _page/*
git add _page
git commit -m "updated docs"
git subtree push --prefix _page origin gh-pages
```

###


#### OLD Environment Setup:
```bash

Use super_setup.py instead

# Navigate to your code directory
export CODE_DIR=~/code
cd $CODE_DIR

# Clone the IBEIS repositories 
git clone https://github.com/WildbookOrg/utool.git
git clone https://github.com/WildbookOrg/vtool.git
git clone https://github.com/WildbookOrg/plottool.git
git clone https://github.com/WildbookOrg/guitool.git
git clone https://github.com/WildbookOrg/hesaff.git
git clone https://github.com/WildbookOrg/ibeis.git
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
git clone https://github.com/WildbookOrg/flann.git
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

# ** NEW 7-23-2015 **: the following commands are now equivalent and do not
# have to be specified from the ibeis source dir if ibeis is installed
python -m ibeis <optional-arguments> [--help]
python -m ibeis.dev <optional-arguments> [--help]

# Useful flags.
# Read code comments in dev.py for more info.
# Careful some commands don't work. Most do.
# --cmd          # shows ipython prompt with useful variables populated
# -w, --wait     # waits (useful for showing plots)
# --gui          # starts the gui as well (dev.py does not show gui by default, main does)
# --web          # runs the program as a web server
# --quiet        # turns off most prints
# --verbose      # turns on verbosity
# --very-verbose # turns on extra verbosity
# --debug2       # runs extra checks
# --debug-print  # shows where print statments occur
# -t [test]


#--------------------
# PSA: Workdirs:
#--------------------
# IBEIS uses the idea of a work directory for databases.
# Use --set-workdir <path> to set your own, or a gui will popup and ask you about it
./main.py --set-workdir /raid/work --preload-exit
./main.py --set-logdir /raid/logs/ibeis --preload-exit

./dev.py --set-workdir ~/data/work --preload-exit

# use --db to specify a database in your WorkDir
# --setdb makes that directory your default directory
python dev.py --db <dbname> --setdb

# Or just use the absolute path
python dev.py --dbdir <full-dbpath>


#--------------------
# Examples:
# Here are are some example commands
#--------------------
# Run the queries for each roi with groundtruth in the PZ_MTEST database
# using the best known configuration of parameters
python dev.py --db PZ_MTEST --allgt -t best
python dev.py --db PZ_MTEST --allgt -t score


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
python dev.py --db PZ_MTEST --setdb
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
python dev.py --convert --force-delete --db PZ_MTEST


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

# Opening a hotspotter database will automatically convert it
python -m ibeis --db JAG_KELLY
# The explicit hotspotter conversion script can be run via
python -m ibeis.dbio.ingest_hsdb --test-convert_hsdb_to_ibeis:0 --db JAG_KELLY


#---------
# Run Tests
#---------
./testsuit/run_tests.sh


#----------------
# Profiling Code
#----------------

utprof.py dev.py -t best --db testdb1 --allgt --nocache-query --prof-mod "spatial;linalg;keypoint"
utprof.py dev.py -t best --db PZ_MTEST --all --nocache-query --prof-mod "spatial;linalg;keypoint"
utprof.py dev.py -t best --db PZ_MTEST --all --nocache-query --prof-mod "spatial;linalg;keypoint"
utprof.py dev.py -t custom --db PZ_MTEST --allgt --noqcache
utprof.py dev.py -t custom:sv_on=False --db PZ_MTEST --allgt --noqcache


#----------------
# Test Commands
#----------------
# Set a default DB First
./dev.py --setdb --dbdir /path/to/your/DBDIR
./dev.py --setdb --db YOURDB
./dev.py --setdb --db PZ_MTEST
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
python -m ibeis list_dbs
# Delete cache
python -m ibeis delete_cache --db testdb1


# Show a single annotations
python -m ibeis.viz.viz_chip show_chip --db PZ_MTEST --aid 1 --show

# Show annotations 1, 3, 5, and 11
python -m ibeis.viz.viz_chip show_many_chips --db PZ_MTEST --aids=1,3,5,11 --show

# Query annotation 2
python -m ibeis.viz.viz_qres show_qres --db PZ_MTEST --qaids=2 --show


# Database Stats for all our important datasets:
./dev.py --allgt -t dbinfo --db PZ_RoseMary | grep -F "[dbinfo]"
./dev.py --allgt -t dbinfo --db PZ_MTEST | grep -F "[dbinfo]"
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

python -m ibeis --tf draw_annot_scoresep --db PZ_MTEST -a default -t best --show

python -m ibeis.dev -e draw_rank_cdf --db PZ_MTEST --show -a timectrl

# Show disagreement cases
ibeis --tf draw_match_cases --db PZ_MTEST -a default:size=20 \
    -t default:K=[1,4] \
    --filt :disagree=True,index=0:4 --show

# SMK TESTS
python dev.py -t smk2 --allgt --db PZ_MTEST --nocache-big --nocache-query --qindex 0:20
python dev.py -t smk2 --allgt --db GZ_ALL --nocache-big --nocache-query --qindex 0:20

python dev.py -t smk2 --allgt --db PZ_MTEST --qindex 20:30 --va
python dev.py -t smk2 --allgt --db PZ_Master0

# Feature Tuning
python dev.py -t test_feats -w --show --db PZ_MTEST --allgt --qindex 1:2

python dev.py -t featparams -w --show --db PZ_MTEST --allgt
python dev.py -t featparams_big -w --show --db PZ_MTEST --allgt
python dev.py -t featparams_big -w --show --db GZ_ALL --allgt
 --allgt --qindex 1:2


# NEW DATABASE TEST
python dev.py -t best --db seals2 --allgt
python dev.py -t best --db seals2 --allgt --vh --vf
python dev.py -t best --db seals2 --allgt

# Testing Distinctivness Parameters
python -m ibeis.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --db GZ_ALL --aid 2
python -m ibeis.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --db PZ_MTEST --aid 10
python -m ibeis.algo.hots.distinctiveness_normalizer --test-test_single_annot_distinctiveness_params --show --db GZ_ALL --aid 2

python -m ibeis.algo.hots.distinctiveness_normalizer --test-test_single_annot_distinctiveness_params --show --db PZ_MTEST --aid 5
python -m ibeis.algo.hots.distinctiveness_normalizer --test-test_single_annot_distinctiveness_params --show --db PZ_MTEST --aid 1


# 2D Gaussian Curves
python -m vtool.patch --test-test_show_gaussian_patches2 --show

# Test Keypoint Coverage
python -m vtool.coverage_kpts --test-gridsearch_kpts_coverage_mask --show
python -m vtool.coverage_kpts --test-make_kpts_coverage_mask --show

# Test Grid Coverage
python -m vtool.coverage_grid --test-gridsearch_coverage_grid_mask --show
python -m vtool.coverage_grid --test-sparse_grid_coverage --show
python -m vtool.coverage_grid --test-gridsearch_coverage_grid --show

# Test Spatially Constrained Scoring
python -m ibeis.algo.hots.vsone_pipeline --test-compute_query_constrained_matches --show
python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_constrained_matches --show
python -m ibeis.algo.hots.vsone_pipeline --test-gridsearch_constrained_matches --show --testindex 2

# Test VsMany ReRanking
python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show
python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show --homog
python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show --homog --db GZ_ALL
python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show --db GZ_ALL

# Problem cases with the back spot
python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show --homog --db GZ_ALL --qaid 425
python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show --homog --db GZ_ALL --qaid 662
python dev.py -t custom:score_method=csum,prescore_method=csum --db GZ_ALL --show --va -w --qaid 425 --noqcache
# Shows vsone results with some of the competing cases
python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show --homog --db GZ_ALL --qaid 662 --daid_list=425,342,678,233


# More rerank vsone tests
python -c "import utool as ut; ut.write_modscript_alias('Tbig.sh', 'dev.py', '--allgt  --db PZ_Master0')"
sh Tbig.sh -t custom:rrvsone_on=True custom 
sh Tbig.sh -t custom:rrvsone_on=True custom --noqcache


# TODO: 
# static lnbnn, normonly, and count test
# combinme vsone and vsmany matches in vsone rr 

# Sanity Check 
# Make sure vsmany and onevsone are exactly the same
python dev.py --setdb --db PZ_Master0
python dev.py --setdb --db PZ_MTEST

# These yeild the same results for vsmany and vsone reanking
# notice that name scoring and feature scoring are turned off. 
# also everything is reranked
#----
python dev.py --allgt -t \
    custom:fg_on=False \
    custom:rrvsone_on=True,prior_coeff=1,unconstrained_coeff=0.0,fs_lnbnn_min=0,fs_lnbnn_max=1,nAnnotPerNameVsOne=200,nNameShortlistVsone=200,fg_on=False \
    --print-confusion-stats --print-gtscore --noqcache
#----

#----
# Turning back on name scoring and feature scoring and restricting to rerank a subset
# This gives results that are closer to what we should actually expect
python dev.py --allgt -t custom \
    custom:rrvsone_on=True,prior_coeff=1.0,unconstrained_coeff=0.0,fs_lnbnn_min=0,fs_lnbnn_max=1 \
    custom:rrvsone_on=True,prior_coeff=0.5,unconstrained_coeff=0.5,fs_lnbnn_min=0,fs_lnbnn_max=1 \
    custom:rrvsone_on=True,prior_coeff=0.1,unconstrained_coeff=0.9,fs_lnbnn_min=0,fs_lnbnn_max=1 \
    --print-bestcfg
#----

#----
# VsOneRerank Tuning: Tune linar combination
python dev.py --allgt -t \
    custom:fg_weight=0.0 \
\
    custom:rrvsone_on=True,prior_coeff=1.0,unconstrained_coeff=0.0,fs_lnbnn_min=0.0,fs_lnbnn_max=1.0,nAnnotPerNameVsOne=200,nNameShortlistVsone=200 \
\
    custom:rrvsone_on=True,prior_coeff=.5,unconstrained_coeff=0.5,fs_lnbnn_min=0.0,fs_lnbnn_max=1.0,nAnnotPerNameVsOne=200,nNameShortlistVsone=200 \
\
  --db PZ_MTEST

#--print-confusion-stats --print-gtscore
#----

#----
python dev.py --allgt -t \
    custom \
    custom:rrvsone_on=True,prior_coeff=1.0,unconstrained_coeff=0.0,fs_lnbnn_min=0.0,fs_lnbnn_max=1.0,nAnnotPerNameVsOne=200,nNameShortlistVsone=200 \
    custom:rrvsone_on=True,prior_coeff=.5,unconstrained_coeff=0.5,fs_lnbnn_min=0.0,fs_lnbnn_max=1.0,nAnnotPerNameVsOne=2,nNameShortlistVsone=20 \
    custom:rrvsone_on=True,prior_coeff=.0,unconstrained_coeff=1.0,fs_lnbnn_min=0.0,fs_lnbnn_max=1.0,nAnnotPerNameVsOne=2,nNameShortlistVsone=20 \
   --db PZ_Master0 
#----

python dev.py --allgt -t \
    custom:rrvsone_on=True,prior_coeff=1.0,unconstrained_coeff=0.0\
    custom:rrvsone_on=True,prior_coeff=.0,unconstrained_coeff=1.0 \
    custom:rrvsone_on=True,prior_coeff=.5,unconstrained_coeff=0.5 \
   --db PZ_Master0

python dev.py --allgt -t custom --db PZ_Master0 --va --show



--noqcache

python dev.py --allgt -t custom custom:rrvsone_on=True


# Testing no affine invaraiance and rotation invariance
dev.py -t custom:affine_invariance=True,rotation_invariance=True custom:affine_invariance=False,rotation_invariance=True custom:affine_invariance=True,rotation_invariance=False custom:affine_invariance=False,rotation_invariance=False --db PZ_MTEST --va --show

dev.py -t custom:affine_invariance=True,rotation_invariance=True custom:affine_invariance=False,rotation_invariance=True custom:affine_invariance=True,rotation_invariance=False custom:affine_invariance=False,rotation_invariance=False --db PZ_MTEST --allgt

dev.py -t custom:affine_invariance=True,rotation_invariance=True custom:affine_invariance=False,rotation_invariance=True custom:affine_invariance=True,rotation_invariance=False custom:affine_invariance=False,rotation_invariance=False --db GZ_ALL --allgt


python dev.py -t custom:affine_invariance=True,rotation_invariance=True custom:affine_invariance=False,rotation_invariance=True custom:affine_invariance=True,rotation_invariance=False custom:affine_invariance=False,rotation_invariance=False --db PZ_Master0 --allgt --index 0:10 --va --show

```

#---------------
# Caveats / Things we are not currently doing

* We do not add or remove points from kdtrees. They are always rebuilt
