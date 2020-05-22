![Logo](_installers/WildBook_logo_72dpi-03.png)

NOTICE: The main github repo is now: https://github.com/WildbookOrg/ibeis

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
python -m vtool.spatial_verification --test-spatially_verify_kpts --show
```

### Name Scoring
![alt text](http://i.imgur.com/IDUnxu2.jpg "namematch")
```bash
python -m ibeis.algo.hots.chip_match show_single_namematch --qaid 1 --show
```

### Identification Ranking 
![alt text](http://i.imgur.com/BlajchI.jpg "rankedmatches")
```bash
python -m ibeis.algo.hots.chip_match show_ranked_matches --show --qaid 86
```

### Inference
![alt text](http://i.imgur.com/RYeeENl.jpg "encgraph")
```bash
# broken
# python -m ibeis.algo.preproc.preproc_encounter compute_encounter_groups --show
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
    

# Code Style and Development Guidelines

## Contributing

It's recommended that you use `pre-commit` to ensure linting procedures are run
on any commit you make. (See also (pre-commit.com)[https://pre-commit.com/])

Reference (pre-commit's installation instructions)[https://pre-commit.com/#install] for software installation on your OS/platform. After you have the software installed, run `pre-commit install` on the commandline. Now everytime you commit to this project's codebase the linter procedures will automatically run over the changed files.

## Python

Try to conform to pep8. 
You should set up your preferred editor to use flake8 as linter.
If using vim I recommend syntastic.

To run flake8 from the commandline use:
    flake8

This will use the flake8 configuration within `setup.cfg`,
which ignores several errors and stylistic considerations.
See the `setup.cfg` file for a full and accurate listing of stylistic codes to ignore.


( Dev comment: my laptop seems to report these flake8 errors while my desktops
  don't. I'm going to list errors that might need to be explicitly enabled here:

* 'F821',  # undefined name
* 'F403',  # import * used, unable to detect names

)

## C++ (Cplusplus)

For C++ code use `astyle` to format your code:
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

# NEW: You can simply open a hotspotter database and it will be converted to IBEIS
python -m ibeis convert_hsdb_to_ibeis --dbdir <path_to_hsdb>

# This script will exlicitly conver the hsdb
python -m ibeis convert_hsdb_to_ibeis --hsdir <path_to_hsdb> --dbdir <path_to_newdb>


#---------
# Ingest examples
#---------
# Ingest raw images
python -m ibeis.dbio.ingest_database --db JAG_Kieryn


#---------
# Run Tests
#---------
./run_tests.py


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


# Database Stats for all our important datasets:
./dev.py --allgt -t dbinfo --db PZ_MTEST | grep -F "[dbinfo]"

# Some mass editing of metadata
./dev.py --db PZ_FlankHack --edit-notes
./dev.py --db GZ_Siva --edit-notes
./dev.py --db GIR_Tanya --edit-notes
./dev.py --allgt -t dbinfo --db GZ_ALL --set-all-species zebra_grevys

# Current Experiments:

# Main experiments
python -m ibeis --tf draw_annot_scoresep --db PZ_MTEST -a default -t best --show
python -m ibeis.dev -e draw_rank_cdf --db PZ_MTEST --show -a timectrl
# Show disagreement cases
ibeis --tf draw_match_cases --db PZ_MTEST -a default:size=20 \
    -t default:K=[1,4] \
    --filt :disagree=True,index=0:4 --show

# SMK TESTS
python dev.py -t smk2 --allgt --db PZ_MTEST --nocache-big --nocache-query --qindex 0:20
python dev.py -t smk2 --allgt --db PZ_MTEST --qindex 20:30 --va

# Feature Tuning
python dev.py -t test_feats -w --show --db PZ_MTEST --allgt --qindex 1:2

python dev.py -t featparams -w --show --db PZ_MTEST --allgt
python dev.py -t featparams_big -w --show --db PZ_MTEST --allgt

# NEW DATABASE TEST
python dev.py -t best --db seals2 --allgt

# Testing Distinctivness Parameters
python -m ibeis.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --db GZ_ALL --aid 2
python -m ibeis.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --db PZ_MTEST --aid 10
python -m ibeis.algo.hots.distinctiveness_normalizer --test-test_single_annot_distinctiveness_params --show --db GZ_ALL --aid 2


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

# Test VsMany ReRanking
python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show
python -m ibeis.algo.hots.vsone_pipeline --test-vsone_reranking --show --homog

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


# Testing no affine invaraiance and rotation invariance
dev.py -t custom:AI=True,RI=True custom:AI=False,RI=True custom:AI=True,RI=False custom:AI=False,RI=False --db PZ_MTEST --show
```

#---------------
# Caveats / Things we are not currently doing

* We do not add or remove points from kdtrees. They are always rebuilt
