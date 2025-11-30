==================
WBIA - WildBook IA
==================

|Build| |Documentation| |Downloads|

.. image:: http://i.imgur.com/TNCiEBe.png
    :alt: "(Note: the rhino and wildebeest matches may be dubious. Other species do work well though")

WBIA program for the storage and management of images and derived data for use in computer vision algorithms. It aims to compute who an animal is, what species an animal is, and where an animal is with the ultimate goal being to ask important why biological questions.

This project is the Machine Learning (ML) / computer vision component of the WildBook project: See https://github.com/WildMeOrg/.  This project is an actively maintained fork of the popular IBEIS (Image Based Ecological Information System) software suite for wildlife conservation.  The original IBEIS project is maintained by Jon Crall (@Erotemic) at https://github.com/Erotemic/ibeis.  The IBEIS toolkit originally was a wrapper around HotSpotter, which original binaries can be downloaded from: http://cs.rpi.edu/hotspotter/

Currently the system is build around a SQLite database, a web GUI, and matplotlib visualizations. Algorithms employed are: convolutional neural network detection and localization and classification, hessian-affine keypoint detection, SIFT keypoint description, LNBNN identification using approximate nearest neighbors.

Support
--------
While WBIA can be set up independently, the Wild Me team is not able to assist with independent model training. Support is limited to system setup in Docker at this time.

Requirements
------------

* Python 3.7+
* OpenCV 3.4.10
* Python dependencies listed in requirements.txt

Installation Instructions
-------------------------

.. code:: bash

    git clone https://github.com/WildMeOrg/wildbook-ia.git
    cd wildbook-ia/devops
    sudo ./build.sh wbia-base wbia-provision wbia

If your build times out, run the command again. The previous run is cached and the build will advance. Note that you cannot tag this build.

Docker
~~~~~~

The WBIA software is built and deployed as a Docker image `wildme/wbia`.  You can download and run the pre-configured instance from the command line using:

.. code:: bash

    # Install Docker - https://docs.docker.com/engine/install/
    docker pull wildme/wbia:latest
    docker run -p 84:5000 wildme/wbia:latest bash

This image is built using the multi-stage Dockerfiles in `devops/`. Once the container is up and running, go to `localhost:84` to see the WBIA landing page.

Configuration
~~~~~~~~~~~~~

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

The following environment variables can be configured in your Docker deployment:

**Model Storage Configuration**

* ``MODEL_DIR`` - Base directory for model storage (default: ``/models``). Supports local volumes or named volumes for persistent model caching.

* ``MODEL_SAS_QUERY`` - Azure Blob Storage SAS token for downloading private models. Optional; only required if models are stored in private Azure blobs.

* ``MODEL_VERIFY_CHECKSUM`` - Enable MD5 checksum verification for downloaded models (default: ``0``). Set to ``1`` to enable integrity checks.

* ``WBIA_MODELS_DIR`` - Legacy alias for ``MODEL_DIR``. Use ``MODEL_DIR`` in new deployments.

**Storage Options**

*Option 1: Local/NAS Storage (Recommended for Production)*

Mount a local or NAS folder to ``/models`` in your container. Models persist across restarts and eliminate Azure dependencies after initial download.

.. code:: yaml

    volumes:
      - /nas/models:/models

*Option 2: Azure Blob with Local Caching*

Set ``MODEL_SAS_QUERY`` for private blob access. Models are downloaded on first run and cached in a named volume for subsequent restarts.

.. code:: yaml

    environment:
      MODEL_SAS_QUERY: "?sp=r&st=2024-01-01T00:00:00Z&..."
    volumes:
      - model-cache:/models

**Expected Directory Structure**

.. code:: text

    /models/
    ├── detectmodels/
    │   └── rf/              # Detection models
    └── lightnet/            # Lightnet models (cached by utool)

**Docker Compose Example**

.. code:: yaml

    version: "3"
    services:
      wbia:
        image: wildme/wbia:nightly
        environment:
          MODEL_DIR: /models
          MODEL_SAS_QUERY: ${MODEL_SAS_QUERY:-}
          MODEL_VERIFY_CHECKSUM: 0
          HOST_UID: "${HOST_UID}"
          HOST_USER: "${HOST_USER}"
        volumes:
          - /nas/models:/models
          - /data/db:/data/db
        ports:
          - "5000:5000"
        restart: unless-stopped

Dependencies
~~~~~~

This project depends on an array of other repositories for functionality.

First Party Toolkits (Required)

* https://github.com/WildMeOrg/wbia-utool

* https://github.com/WildMeOrg/wbia-vtool

First Party Dependencies for Third Party Libraries (Required)

* https://github.com/WildMeOrg/wbia-tpl-pyhesaff

* https://github.com/WildMeOrg/wbia-tpl-pyflann

* https://github.com/WildMeOrg/wbia-tpl-pydarknet

* https://github.com/WildMeOrg/wbia-tpl-pyrf

First Party Plug-ins (Optional)

* https://github.com/WildMeOrg/wbia-plugin-cnn

* https://github.com/WildMeOrg/wbia-plugin-flukematch

* https://github.com/WildMeOrg/wbia-plugin-deepsense

* https://github.com/WildMeOrg/wbia-plugin-finfindr

* https://github.com/WildMeOrg/wbia-plugin-curvrank

    + https://github.com/WildMeOrg/wbia-tpl-curvrank

* https://github.com/WildMeOrg/wbia-plugin-kaggle7

    + https://github.com/WildMeOrg/wbia-tpl-kaggle7

* https://github.com/WildMeOrg/wbia-plugin-2d-orientation

    + https://github.com/WildMeOrg/wbia-tpl-2d-orientation

* https://github.com/WildMeOrg/wbia-plugin-lca

    + https://github.com/WildMeOrg/wbia-tpl-lca

Deprecated Toolkits (Deprecated)
* https://github.com/WildMeOrg/wbia-deprecate-ubelt

* https://github.com/WildMeOrg/wbia-deprecate-dtool

* https://github.com/WildMeOrg/wbia-deprecate-guitool

* https://github.com/WildMeOrg/wbia-deprecate-plottool

* https://github.com/WildMeOrg/wbia-deprecate-detecttools

* https://github.com/WildMeOrg/wbia-deprecate-plugin-humpbacktl

* https://github.com/WildMeOrg/wbia-deprecate-tpl-lightnet

* https://github.com/WildMeOrg/wbia-deprecate-tpl-brambox

Plug-in Templates (Reference)

* https://github.com/WildMeOrg/wbia-plugin-template

* https://github.com/WildMeOrg/wbia-plugin-id-example

Miscellaneous (Reference)

* https://github.com/WildMeOrg/wbia-pypkg-build

* https://github.com/WildMeOrg/wbia-project-website

* https://github.com/WildMeOrg/wbia-aws-codedeploy

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

The documentation is built and available online at `wildmeorg.github.io/wildbook-ia/ <http://wildmeorg.github.io/wildbook-ia/>`_. However, if you need to build a local copy of the source, the following instructions can be used.

.. code:: bash

   # checkout the source code
   # install the project in development mode
   pip install -e .
   # build the docs
   scripts/build-docs.sh

Then open the html file at ``docs/build/html/index.html``.


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

To run doctests with `+REQUIRES(--web-tests)` do:

.. code:: bash

    pytest --web-tests

.. |Build| image:: https://img.shields.io/github/workflow/status/WildMeOrg/wildbook-ia/Build%20and%20upload%20to%20PyPI/main
    :target: https://github.com/WildMeOrg/wildbook-ia/actions?query=branch%3Amain+workflow%3A%22Build+and+upload+to+PyPI%22
    :alt: Build and upload to PyPI

.. |Pypi| image:: https://img.shields.io/pypi/v/wildbook-ia.svg
   :target: https://pypi.python.org/pypi/wildbook-ia
   :alt: Latest PyPI version

.. |Documentation| image:: https://img.shields.io/github/workflow/status/WildMeOrg/wildbook-ia/Build%20documentation?label=documentation
    :target: https://wildmeorg.github.io/wildbook-ia/
    :alt: Documentation on GitHub Pages

.. |Downloads| image:: https://img.shields.io/pypi/dm/wildbook-ia.svg
   :target: https://pypistats.org/packages/wildbook-ia
