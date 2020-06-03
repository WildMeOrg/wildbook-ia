|ReadTheDocs| |Pypi| |Downloads| |Codecov| |CircleCI| |Travis| |Appveyor| 

.. image:: https://i.imgur.com/L0k84xQ.png

This project is a component of the WildMe / WildBook project: See https://github.com/WildbookOrg/


IBEIS - Image Analysis 
----------------------

I.B.E.I.S. = Image Based Ecological Information System
------------------------------------------------------

.. image:: http://i.imgur.com/TNCiEBe.png
    :alt: "(Note: the rhino and wildebeest mathces may be dubious. Other species do work well though")


Installation Instructions (updated 2020-May-03)
-----------------------------------------------

The IBEIS software is now available on `pypi
<https://pypi.org/project/wbia/>`_ for Linux systems. This means if you have
`Python installed
<https://xdoctest.readthedocs.io/en/latest/installing_python.html>`_. You can
simply run:


.. code:: bash

    pip install wbia

to install the software. Then the command to run the GUI is:


.. code:: bash

    wbia

On Windows / OSX I recommend using a Linux virtual machine. However, if you are
computer savvy it is possible to build all of the requirements on from source.
The only tricky components are installing the packages with binary
dependencies: ``pyhesaff`` and ``vtool_ibeis``. If you have these built then
the rest of the dependencies can be installed from pypi even on OSX / Windows.


Running the ``wbia`` command will open the GUI:


If you have already made a database, it will automatically open the most recently used database.

.. image:: https://i.imgur.com/xXF7w8P.png

If this is the first time you've run the program it will not have a database opened:

.. image:: https://i.imgur.com/Ey9Urcv.png

Select new database, (which will first ask you to select a work directory where all of your databases will live).
Then you will be asked to create a database name. Select one and then create the database in your work directory.


You can drag and drop images into the GUI to add them to the database.  Double
clicking an image lets you add "annotations":


.. image:: https://i.imgur.com/t0LQZot.png

You can also right click one or more images and click "Add annotations from
entire images" if your images are already localized to a single individual.

It important than when you add an annotation, you set its species. You can
right click multiple annotations and click "set annotation species". Change
this to anything other than "____".

Once you have annotations with species, you can click one and press "q" to
query for matches in the database of other annotations:


.. image:: https://i.imgur.com/B0ilafa.png

Right clicking and marking each match as "True" or "False" (or alternatively
selecting a row and pressing "T" or "F") will mark images as the same or
different individuals. Groups marked as the same individual will appear in the
"Tree of Names".

Note there are also batch identification methods in the "ID Encounters" "ID
Exemplars" and "Advanced ID Interface" (my personal recommendation). Play
around with different right-click menus (although note that some of these are
buggy and will crash the program), but the main simple identification
procedures are robust and should not crash.


Program Description
-------------------

IBEIS program for the storage and management of images and derived data for
use in computer vision algorithms. It aims to compute who an animal is, what
species an animal is, and where an animal is with the ultimate goal being to
ask important why biological questions.  This This repo Image Analysis image
analysis module of IBEIS. It is both a python module and standalone program. 

Currently the system is build around and SQLite database, a PyQt4 / PyQt5 GUI,
and matplotlib visualizations. Algorithms employed are: random forest species
detection and localization, hessian-affine keypoint detection, SIFT keypoint
description, LNBNN identification using approximate nearest neighbors.
Algorithms in development are SMK (selective match kernel) for identification
and deep neural networks for detection and localization. 

The core of IBEIS is the IBEISController class. It provides an API into IBEIS
data management and algorithms. The IBEIS API Documentation can be found here:
`http://erotemic.github.io/wbia`

The IBEIS GUI (graphical user interface) is built on top of the API. 
We are also experimenting with a new web frontend that bypasses the older GUI code.

Self Installing Executables
---------------------------

Unfortunately we have not released self-installing-executables for IBEIS yet. 
We ~plan~ hope to release these "soon". 

However there are old HotSpotter (the software which IBEIS is based on)
binaries available. These can be downloaded from: `http://cs.rpi.edu/hotspotter/`

Visual Demo
-----------


.. image:: http://i.imgur.com/QWrzf9O.png
   :width: 600
   :alt: Feature Extraction

.. image:: http://i.imgur.com/iMHKEDZ.png
   :width: 600
   :alt: Nearest Neighbors


Match Scoring 
-------------

.. image:: http://imgur.com/Hj43Xxy.png
   :width: 600
   :alt: Match Inspection

Spatial Verification
--------------------

.. image:: http://i.imgur.com/VCz0j9C.jpg
   :width: 600
   :alt: sver


.. code:: bash

    python -m vtool.spatial_verification --test-spatially_verify_kpts --show

Name Scoring
------------

.. image:: http://i.imgur.com/IDUnxu2.jpg
   :width: 600
   :alt: namematch


.. code:: bash

    python -m wbia.algo.hots.chip_match show_single_namematch --qaid 1 --show

Identification Ranking 
----------------------

.. image:: http://i.imgur.com/BlajchI.jpg
   :width: 600
   :alt: rankedmatches


.. code:: bash

    python -m wbia.algo.hots.chip_match show_ranked_matches --show --qaid 86

Inference
---------

.. image:: http://i.imgur.com/RYeeENl.jpg
   :width: 600
   :alt: encgraph


.. code:: bash

    # broken
    # python -m wbia.algo.preproc.preproc_encounter compute_encounter_groups --show

Internal Modules
----------------

In the interest of modular code we are actively developing several different modules. 


Erotemic's IBEIS Image Analysis module dependencies 

* https://github.com/Erotemic/utool

* https://github.com/Erotemic/plottool_ibeis
* https://github.com/Erotemic/vtool_ibeis
* https://github.com/Erotemic/guitool_ibeis
* https://github.com/Erotemic/pyflann_ibeis

* https://github.com/Erotemic/hesaff
* https://github.com/Erotemic/futures_actors


bluemellophone's IBEIS Image Analysis modules

* https://github.com/WildbookOrg/pyrf
  docs: http://bluemellophone.github.io/pyrf


The IBEIS module itself: 

* https://github.com/WildbookOrg/wbia

IBEIS Development Environment Setup 
------------------------------------

.. code:: bash

    # The following install script install wbia and all dependencies. 
    # If it doesnt you can look at the older instructions which follow
    # and try to figure it out. After running this you should have a code
    # directory with all of the above repos. 

    # Navigate to your code directory
    export CODE_DIR=~/code
    mkdir $CODE_DIR
    cd $CODE_DIR

    # Clone IBEIS
    git clone https://github.com/Erotemic/wbia.git
    cd wbia

    # Install the requirements for super_setup
    pip install -r requirements/super_setup.txt

    # Install the development requirements (note-these are now all on pypi, so
    # this is not strictly necessary)
    python super_setup.py ensure

    # NOTE: you can use super_setup to do several things
    python super_setup.py --help
    python super_setup.py versions
    python super_setup.py status
    python super_setup.py check
    python super_setup.py pull

    # Run the run_developer_setup.sh file in each development repo
    python super_setup.py develop

    # Or you can also just do to use pypi versions of dev repos:
    python setup.py develop

    # Optional: set a workdir and download a test dataset
    .python -m wbia.dev 
    .python -m wbia.dev -t mtest 
    python -m wbia.dev -t nauts 
    ./reset_dbs.py

    python -m wbia --set-workdir ~/data/work --preload-exit
    python -m wbia -e ensure_mtest

    # make sure everyhing is set up correctly
    python -m wbia --db PZ_MTEST


Running Tests
-------------

The new way of running tests is with xdoctest, or using the "run_doctests.sh" script.

Code Style and Development Guidelines
-------------------------------------

Contributing
~~~~~~~~~~~~

It's recommended that you use ``pre-commit`` to ensure linting procedures are run
on any commit you make. (See also `pre-commit.com <https://pre-commit.com/>`_)

Reference `pre-commit's installation instructions <https://pre-commit.com/#install>`_ for software installation on your OS/platform. After you have the software installed, run ``pre-commit instal`` on the commandline. Now everytime you commit to this project's codebase the linter procedures will automatically run over the changed files.

Python
~~~~~~

Try to conform to pep8. 
You should set up your preferred editor to use flake8 as linter.
If using vim I recommend syntastic.

To run flake8 from the commandline use::

  flake8

This will use the flake8 configuration within ``setup.cfg``,
which ignores several errors and stylistic considerations.
See the ``setup.cfg`` file for a full and accurate listing of stylistic codes to ignore.


.. Dev comment: my laptop seems to report these flake8 errors while my desktops
   don't. I'm going to list errors that might need to be explicitly enabled here:

     * 'F821',  # undefined name
     * 'F403',  # import * used, unable to detect names

C++ (Cplusplus)
~~~~~~~~~~~~~~~

For C++ code use ``astyle`` to format your code::

  atyle --style=ansi --indent=spaces --attach-inlines --indent-classes --indent-modifiers --indent-switches --indent-preproc-cond --indent-col1-comments --pad-oper --unpad-paren --delete-empty-lines --add-brackets 


Example usage
--------------

(Note: This list is far from complete, and some commands may be outdated)

.. code:: bash

    #--------------------
    # Main Commands
    #--------------------
    python -m wbia.main <optional-arguments> [--help]
    python -m wbia.dev <optional-arguments> [--help]
    # main is the standard entry point to the program
    # dev is a more advanced developer entry point

    # ** NEW 7-23-2015 **: the following commands are now equivalent and do not
    # have to be specified from the wbia source dir if wbia is installed
    python -m wbia <optional-arguments> [--help]
    python -m wbia.dev <optional-arguments> [--help]

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
    ./main.py --set-logdir /raid/logs/wbia --preload-exit

    python -m wbia.dev --set-workdir ~/data/work --preload-exit

    # use --db to specify a database in your WorkDir
    # --setdb makes that directory your default directory
    python -m wbia.dev --db <dbname> --setdb

    # Or just use the absolute path
    python -m wbia.dev --dbdir <full-dbpath>


    #--------------------
    # Examples:
    # Here are are some example commands
    #--------------------
    # Run the queries for each roi with groundtruth in the PZ_MTEST database
    # using the best known configuration of parameters
    python -m wbia.dev --db PZ_MTEST --allgt -t best
    python -m wbia.dev --db PZ_MTEST --allgt -t score


    # View work dir
    python -m wbia.dev --vwd --prequit

    # List known databases
    python -m wbia.dev -t list_dbs


    # Dump/Print contents of params.args as a dict
    python -m wbia.dev --prequit --dump-argv

    # Dump Current SQL Schema to stdout 
    python -m wbia.dev --dump-schema --postquit


    #------------------
    # Convert a hotspotter database to IBEIS
    #------------------

    # NEW: You can simply open a hotspotter database and it will be converted to IBEIS
    python -m wbia convert_hsdb_to_wbia --dbdir <path_to_hsdb>

    # This script will exlicitly conver the hsdb
    python -m wbia convert_hsdb_to_wbia --hsdir <path_to_hsdb> --dbdir <path_to_newdb>

    #---------
    # Ingest examples
    #---------
    # Ingest raw images
    python -m wbia.dbio.ingest_database --db JAG_Kieryn

    #---------
    # Run Tests
    #---------
    ./run_tests.py

    #----------------
    # Test Commands
    #----------------
    # Set a default DB First
    python -m wbia.dev --setdb --dbdir /path/to/your/DBDIR
    python -m wbia.dev --setdb --db YOURDB
    python -m wbia.dev --setdb --db PZ_MTEST
    python -m wbia.dev --setdb --db PZ_FlankHack

    # List all available tests
    python -m wbia.dev -t help
    # Minimal Database Statistics
    python -m wbia.dev --allgt -t info
    # Richer Database statistics
    python -m wbia.dev --allgt -t dbinfo
    # Print algorithm configurations
    python -m wbia.dev -t printcfg
    # Print database tables
    python -m wbia.dev -t tables
    # Print only the image table
    python -m wbia.dev -t imgtbl
    # View data directory in explorer/finder/nautilus
    python -m wbia.dev -t vdd

    # List all IBEIS databases
    python -m wbia list_dbs
    # Delete cache
    python -m wbia delete_cache --db testdb1


    # Show a single annotations
    python -m wbia.viz.viz_chip show_chip --db PZ_MTEST --aid 1 --show
    # Show annotations 1, 3, 5, and 11
    python -m wbia.viz.viz_chip show_many_chips --db PZ_MTEST --aids=1,3,5,11 --show


    # Database Stats for all our important datasets:
    python -m wbia.dev --allgt -t dbinfo --db PZ_MTEST | grep -F "[dbinfo]"

    # Some mass editing of metadata
    python -m wbia.dev --db PZ_FlankHack --edit-notes
    python -m wbia.dev --db GZ_Siva --edit-notes
    python -m wbia.dev --db GIR_Tanya --edit-notes
    python -m wbia.dev --allgt -t dbinfo --db GZ_ALL --set-all-species zebra_grevys

    # Current Experiments:

    # Main experiments
    python -m wbia --tf draw_annot_scoresep --db PZ_MTEST -a default -t best --show
    python -m wbia.dev -e draw_rank_cdf --db PZ_MTEST --show -a timectrl
    # Show disagreement cases
    wbia --tf draw_match_cases --db PZ_MTEST -a default:size=20 \
        -t default:K=[1,4] \
        --filt :disagree=True,index=0:4 --show

    # SMK TESTS
    python -m wbia.dev -t smk2 --allgt --db PZ_MTEST --nocache-big --nocache-query --qindex 0:20
    python -m wbia.dev -t smk2 --allgt --db PZ_MTEST --qindex 20:30 --va

    # Feature Tuning
    python -m wbia.dev -t test_feats -w --show --db PZ_MTEST --allgt --qindex 1:2

    python -m wbia.dev -t featparams -w --show --db PZ_MTEST --allgt
    python -m wbia.dev -t featparams_big -w --show --db PZ_MTEST --allgt

    # NEW DATABASE TEST
    python -m wbia.dev -t best --db seals2 --allgt

    # Testing Distinctivness Parameters
    python -m wbia.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --db GZ_ALL --aid 2
    python -m wbia.algo.hots.distinctiveness_normalizer --test-get_distinctiveness --show --db PZ_MTEST --aid 10
    python -m wbia.algo.hots.distinctiveness_normalizer --test-test_single_annot_distinctiveness_params --show --db GZ_ALL --aid 2

    # 2D Gaussian Curves
    python -m vtool_ibeis.patch --test-test_show_gaussian_patches2 --show

    # Test Keypoint Coverage
    python -m vtool_ibeis.coverage_kpts --test-gridsearch_kpts_coverage_mask --show
    python -m vtool_ibeis.coverage_kpts --test-make_kpts_coverage_mask --show

    # Test Grid Coverage
    python -m vtool_ibeis.coverage_grid --test-gridsearch_coverage_grid_mask --show
    python -m vtool_ibeis.coverage_grid --test-sparse_grid_coverage --show
    python -m vtool_ibeis.coverage_grid --test-gridsearch_coverage_grid --show

    # Test Spatially Constrained Scoring
    python -m wbia.algo.hots.vsone_pipeline --test-compute_query_constrained_matches --show
    python -m wbia.algo.hots.vsone_pipeline --test-gridsearch_constrained_matches --show

    # Test VsMany ReRanking
    python -m wbia.algo.hots.vsone_pipeline --test-vsone_reranking --show
    python -m wbia.algo.hots.vsone_pipeline --test-vsone_reranking --show --homog

    # Problem cases with the back spot
    python -m wbia.algo.hots.vsone_pipeline --test-vsone_reranking --show --homog --db GZ_ALL --qaid 425
    python -m wbia.algo.hots.vsone_pipeline --test-vsone_reranking --show --homog --db GZ_ALL --qaid 662
    python -m wbia.dev -t custom:score_method=csum,prescore_method=csum --db GZ_ALL --show --va -w --qaid 425 --noqcache
    # Shows vsone results with some of the competing cases
    python -m wbia.algo.hots.vsone_pipeline --test-vsone_reranking --show --homog --db GZ_ALL --qaid 662 --daid_list=425,342,678,233

    # More rerank vsone tests
    python -c "import utool as ut; ut.write_modscript_alias('Tbig.sh', 'dev.py', '--allgt  --db PZ_Master0')"
    sh Tbig.sh -t custom:rrvsone_on=True custom 
    sh Tbig.sh -t custom:rrvsone_on=True custom --noqcache

    #----
    # Turning back on name scoring and feature scoring and restricting to rerank a subset
    # This gives results that are closer to what we should actually expect
    python -m wbia.dev --allgt -t custom \
        custom:rrvsone_on=True,prior_coeff=1.0,unconstrained_coeff=0.0,fs_lnbnn_min=0,fs_lnbnn_max=1 \
        custom:rrvsone_on=True,prior_coeff=0.5,unconstrained_coeff=0.5,fs_lnbnn_min=0,fs_lnbnn_max=1 \
        custom:rrvsone_on=True,prior_coeff=0.1,unconstrained_coeff=0.9,fs_lnbnn_min=0,fs_lnbnn_max=1 \
        --print-bestcfg
    #----

    #----
    # VsOneRerank Tuning: Tune linar combination
    python -m wbia.dev --allgt -t \
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
    python -m wbia.dev -t custom:AI=True,RI=True custom:AI=False,RI=True custom:AI=True,RI=False custom:AI=False,RI=False --db PZ_MTEST --show

Caveats / Things we are not currently doing
-------------------------------------------

* We do not add or remove points from kdtrees. They are always rebuilt

.. |CircleCI| image:: https://circleci.com/gh/Erotemic/wbia.svg?style=svg
    :target: https://circleci.com/gh/Erotemic/wbia
.. |Travis| image:: https://img.shields.io/travis/Erotemic/wbia/master.svg?label=Travis%20CI
   :target: https://travis-ci.org/Erotemic/wbia?branch=master
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/github/Erotemic/wbia?branch=master&svg=True
   :target: https://ci.appveyor.com/project/Erotemic/wbia/branch/master
.. |Codecov| image:: https://codecov.io/github/Erotemic/wbia/badge.svg?branch=master&service=github
   :target: https://codecov.io/github/Erotemic/wbia?branch=master
.. |Pypi| image:: https://img.shields.io/pypi/v/wbia.svg
   :target: https://pypi.python.org/pypi/wbia
.. |Downloads| image:: https://img.shields.io/pypi/dm/wbia.svg
   :target: https://pypistats.org/packages/wbia
.. |ReadTheDocs| image:: https://readthedocs.org/projects/wbia/badge/?version=latest
    :target: http://wbia.readthedocs.io/en/latest/
