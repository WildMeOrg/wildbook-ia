==================
WBIA - WildBook IA
==================

|Build| |Pypi| |ReadTheDocs| |Downloads|

.. image:: http://i.imgur.com/TNCiEBe.png
    :alt: "(Note: the rhino and wildebeest matches may be dubious. Other species do work well though")

WBIA program for the storage and management of images and derived data for
use in computer vision algorithms. It aims to compute who an animal is, what
species an animal is, and where an animal is with the ultimate goal being to
ask important why biological questions.

This project is the Machine Learning (ML) / computer vision component of the WildBook project: See https://github.com/WildbookOrg/.  This project is an actively maintained fork of the popular IBEIS (Image Based Ecological Information System) software suite for wildlife conservation.  The original IBEIS project is maintained by Jon Crall (@Erotemic) at https://github.com/Erotemic/ibeis.  The IBEIS toolkit originally was a wrapper around HotSpotter, which original binaries can be downloaded from: http://cs.rpi.edu/hotspotter/

Currently the system is build around and SQLite database, a web GUI,
and matplotlib visualizations. Algorithms employed are: convolutional neural network
detection and localization and classification, hessian-affine keypoint detection, SIFT keypoint
description, LNBNN identification using approximate nearest neighbors.

Requirements
------------

* Python 3.5+
* OpenCV 3.4.10
* Python dependencies listed in requirements.txt

Installation Instructions
-------------------------

PyPI
~~~~

The WBIA software is now available on `pypi
<https://pypi.org/project/wbia/>`_ for Linux systems. This means if you have
`Python installed
<https://xdoctest.readthedocs.io/en/latest/installing_python.html>`_. You can
simply run:

.. code:: bash

    pip install wbia

to install the software. Then the command to run the GUI is:

.. code:: bash

    wbia

We highly recommend using a Python virtual environment: https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv

Docker
~~~~~~

The WBIA software is built and deployed as a Docker image `wildme/wbia`.  You can download and run the pre-configured instance from the command line using:

.. code:: bash

    # Install Docker - https://docs.docker.com/engine/install/
    docker pull wildme/wbia:latest
    docker container run -p <external port>:5000 --name wildbook-ia -v /path/to/local/database/:/data/docker/ wildme/wbia:latest

This image is built using the multi-stage Dockerfiles in `devops/`.

Source
~~~~~~

To be updated soon.

This project depends on an array of other repositories for functionality.

First Party Toolkits (Required)

* https://github.com/WildbookOrg/wbia-utool

* https://github.com/WildbookOrg/wbia-vtool

First Party Dependencies for Third Party Libraries (Required)

* https://github.com/WildbookOrg/wbia-tpl-pyhesaff

* https://github.com/WildbookOrg/wbia-tpl-pyflann

* https://github.com/WildbookOrg/wbia-tpl-pydarknet

* https://github.com/WildbookOrg/wbia-tpl-pyrf

First Party Plug-ins (Optional)

* https://github.com/WildbookOrg/wbia-plugin-cnn

* https://github.com/WildbookOrg/wbia-plugin-flukematch

* https://github.com/WildbookOrg/wbia-plugin-deepsense

* https://github.com/WildbookOrg/wbia-plugin-finfindr

* https://github.com/WildbookOrg/wbia-plugin-curvrank

    + https://github.com/WildbookOrg/wbia-tpl-curvrank

* https://github.com/WildbookOrg/wbia-plugin-kaggle7

    + https://github.com/WildbookOrg/wbia-tpl-kaggle7

* https://github.com/WildbookOrg/wbia-plugin-2d-orientation

    + https://github.com/WildbookOrg/wbia-tpl-2d-orientation

* https://github.com/WildbookOrg/wbia-plugin-lca

    + https://github.com/WildbookOrg/wbia-tpl-lca

Deprecated Toolkits (Deprecated)
* https://github.com/WildbookOrg/wbia-deprecate-ubelt

* https://github.com/WildbookOrg/wbia-deprecate-dtool

* https://github.com/WildbookOrg/wbia-deprecate-guitool

* https://github.com/WildbookOrg/wbia-deprecate-plottool

* https://github.com/WildbookOrg/wbia-deprecate-detecttools

* https://github.com/WildbookOrg/wbia-deprecate-plugin-humpbacktl

* https://github.com/WildbookOrg/wbia-deprecate-tpl-lightnet

* https://github.com/WildbookOrg/wbia-deprecate-tpl-brambox

Plug-in Templates (Reference)

* https://github.com/WildbookOrg/wbia-plugin-template

* https://github.com/WildbookOrg/wbia-plugin-id-example

Miscellaneous (Reference)

* https://github.com/WildbookOrg/wbia-pypkg-build

* https://github.com/WildbookOrg/wbia-project-website

* https://github.com/WildbookOrg/wbia-aws-codedeploy

Citation
--------

If you use this code or its models in your research, please cite:

.. code:: text

    @inproceedings{crall2013hotspotter,
        title={Hotspotter — patterned species instance recognition},
        author={Crall, Jonathan P and Stewart, Charles V and Berger-Wolf, Tanya Y and Rubenstein, Daniel I and Sundaresan, Siva R},
        booktitle={2013 IEEE workshop on applications of computer vision (WACV)},
        pages={230--237},
        year={2013},
        organization={IEEE}
    }

    @inproceedings{parham2018animal,
        title={An animal detection pipeline for identification},
        author={Parham, Jason and Stewart, Charles and Crall, Jonathan and Rubenstein, Daniel and Holmberg, Jason and Berger-Wolf, Tanya},
        booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
        pages={1075--1083},
        year={2018},
        organization={IEEE}
    }

    @inproceedings{berger2015ibeis,
        title={IBEIS: Image-based ecological information system: From pixels to science and conservation},
        author={Berger-Wolf, TY and Rubenstein, DI and Stewart, CV and Holmberg, J and Parham, J and Crall, J},
        booktitle={Bloomberg Data for Good Exchange Conference, New York, NY, USA},
        volume={2},
        year={2015}
    }

    @article{berger2017wildbook,
        title={Wildbook: Crowdsourcing, computer vision, and data science for conservation},
        author={Berger-Wolf, Tanya Y and Rubenstein, Daniel I and Stewart, Charles V and Holmberg, Jason A and Parham, Jason and Menon, Sreejith and Crall, Jonathan and Van Oast, Jon and Kiciman, Emre and Joppa, Lucas},
        journal={arXiv preprint arXiv:1710.08880},
        year={2017}
    }

Documentation
-------------------------

The WBIA API Documentation can be found here: https://wildbook-ia.readthedocs.io/en/latest/

Code Style and Development Guidelines
-------------------------------------

Contributing
~~~~~~~~~~~~

It's recommended that you use ``pre-commit`` to ensure linting procedures are run
on any commit you make. (See also `pre-commit.com <https://pre-commit.com/>`_)

Reference `pre-commit's installation instructions <https://pre-commit.com/#install>`_ for software installation on your OS/platform. After you have the software installed, run ``pre-commit install`` on the command line. Now every time you commit to this project's code base the linter procedures will automatically run over the changed files.  To run pre-commit on files preemtively from the command line use:

.. code:: bash

    git add .
    pre-commit run

    # or

    pre-commit run --all-files

Brunette
~~~~~~~~

Our code base has been formatted by Brunette, which is a fork and more configurable version of Black (https://black.readthedocs.io/en/stable/).

Flake8
~~~~~~

Try to conform to PEP8.  You should set up your preferred editor to use flake8 as its Python linter, but pre-commit will ensure compliance before a git commit is completed.

To run flake8 from the command line use:

.. code:: bash

    flake8


This will use the flake8 configuration within ``setup.cfg``,
which ignores several errors and stylistic considerations.
See the ``setup.cfg`` file for a full and accurate listing of stylistic codes to ignore.

PyTest
~~~~~~

Our code uses Google-style documentation tests (doctests) that uses pytest and xdoctest to enable full support.  To run the tests from the command line use:

.. code:: bash

    pytest


.. |Build| image:: https://img.shields.io/github/workflow/status/WildbookOrg/wildbook-ia/Build%20and%20upload%20to%20PyPI/master
    :target: https://github.com/WildbookOrg/wildbook-ia/actions?query=branch%3Amaster+workflow%3A%22Build+and+upload+to+PyPI%22
    :alt: Build and upload to PyPI (master)

.. |Pypi| image:: https://img.shields.io/pypi/v/wbia.svg
   :target: https://pypi.python.org/pypi/wbia
   :alt: Latest PyPI version

.. |ReadTheDocs| image:: https://readthedocs.org/projects/wildbook-ia/badge/?version=latest
    :target: http://wildbook-ia.readthedocs.io/en/latest/
    :alt: Documentation on ReadTheDocs

.. |Downloads| image:: https://img.shields.io/pypi/dm/wbia.svg
   :target: https://pypistats.org/packages/wbia
