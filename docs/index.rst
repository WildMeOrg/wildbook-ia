.. wbia documentation master file, created by
   sphinx-quickstart on Fri Jun  5 14:27:03 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Wildbook's Image Analysis (IA)
==============================

For details about the Wildbook project see the `Wild Me <https://wildme.org>`_ website.

Wildbook's Image Analysis is colloquially known as Wildbook-IA and by developers as wbia (wib-ee-A). Any references to WBIA in this documentation should be assumed to therefore mean Wildbook-IA.

The Wildbook-IA application is used for the storage, management and analysis of images and derived data used by computer vision algorithms. It aims to compute who an animal is, what species an animal is, and where an animal is with the ultimate goal being to ask important why biological questions.

This project is the Machine Learning (ML) / computer vision component of the `WildBook project <https://github.com/WildMeOrg/>`_. This project is an actively maintained fork of the popular IBEIS (Image Based Ecological Information System) software suite for wildlife conservation. The original IBEIS project is maintained by Jon Crall (@Erotemic) at https://github.com/Erotemic/ibeis. The IBEIS toolkit originally was a wrapper around HotSpotter, which original binaries can be downloaded from: http://cs.rpi.edu/hotspotter/

Currently the system is build around and SQLite database, a web UI, and matplotlib visualizations. Algorithms employed are: convolutional neural network detection and localization and classification, hessian-affine keypoint detection, SIFT keypoint description, LNBNN identification using approximate nearest neighbors.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
