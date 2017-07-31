# -*- coding: utf-8 -*-
# flake8: noqa
"""
TODO DEPRICATE
DEPRICATE THIS ENTIRE FILE

This file contains most every module I've ever used.
Serves as a good check to make sure either everything I
want to use is imported / exists.
"""
from __future__ import absolute_import, division, print_function
# Python
from collections import OrderedDict, defaultdict, namedtuple
from os.path import (dirname, realpath, join, exists, normpath, splitext,
                     expanduser, relpath, isabs, commonprefix, basename)
from itertools import chain, cycle
import six
from six.moves import range, zip, map, zip_longest, builtins, cPickle
from itertools import product as iprod
import argparse
import atexit
import copy
import colorsys
import datetime
import decimal
import fnmatch
import functools
import hashlib
import imp
import inspect
import itertools
import logging
import multiprocessing
import operator
import os
import platform
import re
import shelve
import shlex
import shutil
import signal
import site
import subprocess
import sys
import textwrap
import time
import types
import uuid
import urllib
import warnings
import zipfile
if not sys.platform.startswith('win32'):
    import resource
# PIPI
if six.PY2:
    import functools32
import psutil
# Qt
import sip
from guitool import __PYQT__
from guitool.__PYQT__ import QtCore, QtGui
from guitool.__PYQT__.QtCore import Qt
# Matplotlib
from plottool import __MPL_INIT__
__MPL_INIT__.init_matplotlib()
#mpl.use('Qt4Agg')  # pyinstaller hack
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
# Scientific
import numpy as np
import numpy.linalg as npl
from numpy import (array, rollaxis, sqrt, zeros, ones, diag)
from numpy.core.umath_tests import matrix_multiply
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import MeanShift, estimate_bandwidth
#import statsmodels
#import pandas as pd
#import networkx as netx
#try:
#    import graph_tool
#except ImportError as ex:
#    #print('Warning: %r' % ex)
#    pass

# WEB
import tornado
import flask
import simplejson

# Tools
import utool
import utool as ut
import detecttools
import vtool
import vtool as vt
import plottool
import guitool

# VTool
import vtool
from vtool import chip as ctool
from vtool import image as gtool
from vtool import histogram as htool
from vtool import patch as ptool
from vtool import keypoint as ktool
from vtool import linalg as ltool
from vtool import linalg
from vtool import geometry
from vtool import segmentation
from vtool import spatial_verification as sverif
from vtool.tests import grabdata

# PlotTool
import plottool
import plottool as pt
from plottool import plot_helpers as ph
from plottool import draw_func2 as df2
from plottool import interact_helpers as ih
from plottool import viz_keypoints
from plottool import viz_image2
from plottool import fig_presenter

(print, rrr, profile) = utool.inject2(__name__)

def find_unregisterd():
    import sys
    from ibeis import all_imports

    print('\n'.join(sorted(sys.modules.keys())))

    sys_module_strs = list(map(lambda x: str(x[1]), six.iteritems(sys.modules)))
    all_module_strs = []

    for attrname in dir(all_imports):
        attr = getattr(all_imports, attrname)
        if str(type(attr)).startswith('<type \'module\'>'):
            #print(attr)
            sys_module_strs.append(str(attr))
    set(sys_module_strs) - set(all_module_strs)

def reload_all():
    # reload self
    rrr()
    # This should reload modules roughtly in the order they were imported
    # reload utool first so class functions will be reregistered
    utool.reload_subs()
    guiback.rrr()
    vtool.reload_subs()
    guitool.reload_subs()
    plottool.reload_subs()
    # reload ibeis last
    ibeis.reload_subs()

def class_reload():
    ibs.change_class(IBEISControl.IBEISController)
    IBEISControl.__ALL_CONTROLLERS__

    mod_id_0   = id(IBEISControl)
    class_id_0 = id(IBEISControl.IBEISController)
    utool.printvar2('mod_id_0')
    utool.printvar2('class_id_0')

    reload_all()

    mod_id_1   = id(IBEISControl)
    class_id_1 = id(IBEISControl.IBEISController)
    utool.printvar2('mod_id_1')
    utool.printvar2('class_id_1')
    utool.printvar2('mod_id_0')
    utool.printvar2('class_id_0')

def embed(back):
    """ Allows for embedding in an environment with all imports """
    ibs = back.ibs
    front = back.front
    ibswgt = front
    #import IPython
    #IPython.embed()
    utool.embed()
