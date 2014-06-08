# IBEIS
## I.B.E.I.S. = Image Based Ecological Information System
=====


Erotemic's IBEIS module dependencies 

* https://github.com/Erotemic/utool
* https://github.com/Erotemic/plottool
* https://github.com/Erotemic/vtool
* https://github.com/Erotemic/hesaff
* https://github.com/Erotemic/guitool


bluemellophone's IBEIS modules

* https://github.com/bluemellophone/detecttools
* https://github.com/bluemellophone/pyrf


hjweide's IBEIS modules

* https://github.com/hjweide/pygist


The IBEIS module itself: 

* https://github.com/Erotemic/ibeis

# Environment Setup:

```bash
# The following install script instal ibeis and all dependencies. 
# If it doesnt you can look at the older instructions which follow
# and try to figure it out

# Navigate to your code directory
export CODE_DIR=~/code
mkdir $CODE_DIR
cd $CODE_DIR

# Clone IBEIS
git clone https://github.com/Erotemic/ibeis.git
cd ibeis

# Generate the prereq install script
./_scripts/bootstrap.py

# Run the prereq install script
./_scripts/__install_prereqs__.sh

# Run super_setup to build and install ibeis modules in development mode
# Usually this needs to be run twice. Either way it wont hurt
./super_setup.py --build --develop
./super_setup.py --build --develop
```


###


#### OLD Environment Setup:
```bash
# Navigate to your code directory
export CODE_DIR=~/code
cd $CODE_DIR

# First clone the IBEIS repos
git clone https://github.com/Erotemic/utool.git
git clone https://github.com/Erotemic/vtool.git
git clone https://github.com/Erotemic/plottool.git
git clone https://github.com/Erotemic/guitool.git
git clone https://github.com/Erotemic/hesaff.git
git clone https://github.com/Erotemic/ibeis.git
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

# use --db to specify a database in your WorkDir
# --setdb makes that directory your default directory
python dev.py --db <dbname> --setdb

# Or just use the absolute path
python dev.py --dbdir <full-dbpath>


#--------------------
# Examples:
# Here are are some example commands
#--------------------
# Run the queries for each roi with groundtruth in the PZ_MOTHERS database
# using the best known configuration of parameters
python dev.py --db PZ_MOTHERS --allgt -t best
python dev.py --db PZ_MOTHERS --allgt -t score


# View work dir
python dev.py --vwd --prequit

# List known databases
python dev.py -t list_dbs


# Dump contents of params.args
python dev.py --prequit --dump-argv


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
python dev.py --db PZ_MOTHERS --setdb

# See a list of tests
python dev.py -t help

# See a plot of scores
python dev.py --allgt -t scores -w

# Run the best configuration and print out the hard cases
python dev.py --allgt -t best --echo-hardcase

# Look at inspection widget
python dev.py --allgt -t inspect -w
```
