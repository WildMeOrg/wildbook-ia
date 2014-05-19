# flake8: noqa
from __future__ import absolute_import, division, print_function
# Python
import __builtin__
from collections import OrderedDict, defaultdict
from os.path import (dirname, realpath, join, exists, normpath, splitext,
                     expanduser, relpath, isabs)
from itertools import izip, chain, imap, cycle
from itertools import product as iprod
import imp
import itertools
import logging
import multiprocessing
import os
import re
import shutil
import site
import sys
import textwrap
import operator
# Matplotlib
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
# Scientific
import numpy as np
import numpy.linalg as npl
from numpy import (array, rollaxis, sqrt, zeros, ones, diag)
from numpy.core.umath_tests import matrix_multiply
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
from scipy.cluster.hierarchy import fclusterdata
#import networkx as netx
#try:
#    import graph_tool
#except ImportError as ex:
#    #print('Warning: %r' % ex)
#    pass
# Qt
import PyQt4
from PyQt4 import QtCore, QtGui
from PyQt4.Qt import (QAbstractItemModel, QModelIndex, QVariant, QWidget,
                      Qt, QObject, pyqtSlot, QKeyEvent)
# UTool
import utool
# A bit of a hack right now
utool.util_sysreq.ensure_in_pythonpath('hesaff')
utool.util_sysreq.ensure_in_pythonpath('pyrf')

# VTool
import vtool
from vtool import chip as ctool
from vtool import image as gtool
from vtool import histogram as htool
from vtool import patch as ptool
from vtool import keypoint as ktool
from vtool import linalg as ltool
from vtool import segmentation
from vtool import spatial_verification as sverif

# DrawTool
import plottool
from plottool import draw_func2 as df2
from plottool import interact_helpers as ih
from plottool import viz_keypoints

# GUITool
import guitool

# IBEIS DEV
from ibeis.dev import main_commands
from ibeis.dev import params
from ibeis.dev import ibsfuncs
from ibeis.dev import results_organizer
from ibeis.dev import results_analyzer
from ibeis.dev import results_all
from ibeis.dev import experiment_configs
from ibeis.dev import experiment_harness
from ibeis.dev import experiment_printres
from ibeis.dev import experiment_helpers as eh
# IBEIS EXPORT
from ibeis.export import export_hsdb
# IBEIS INJEST
from ibeis.injest import injest_hsdb
from ibeis.injest import injest_database
# IBEIS MODEL
from ibeis.model import Config
from ibeis.model import preproc
from ibeis.model import hots
# IBEIS MODEL PREPROCESSING
from ibeis.model.preproc import preproc_image
from ibeis.model.preproc import preproc_chip
from ibeis.model.preproc import preproc_feat
from ibeis.model.preproc import preproc_detectimg
# IBEIS MODEL HOTSPOTTER
from ibeis.model.hots import matching_functions as mf
from ibeis.model.hots import match_chips3 as mc3
from ibeis.model.hots import match_chips3 as nn_filters
from ibeis.model.hots import NNIndex
from ibeis.model.hots import QueryResult
from ibeis.model.hots import QueryRequest
from ibeis.model.hots import voting_rules2 as vr2
from ibeis.model.hots import coverage_image
from ibeis.model.hots import query_helpers
# IBEIS VIEW GUI
from ibeis.gui import guifront
from ibeis.gui import guiback
from ibeis.gui import uidtables
# IBEIS VIEW VIZ
from ibeis.viz import viz_helpers as vh
from ibeis.viz import viz_image
from ibeis.viz import viz_chip
from ibeis.viz import viz_matches
from ibeis.viz import viz_sver
# IBEIS VIEW INTERACT
from ibeis.viz.interact import ishow_image
from ibeis.viz.interact import ishow_chip
from ibeis.viz.interact import ishow_name
from ibeis.viz.interact import ishow_qres
from ibeis.viz.interact import ishow_sver
from ibeis.viz.interact import ishow_matches
from ibeis.viz.interact import iselect_bbox
# IBEIS CONTROl
from ibeis.control import SQLDatabaseControl
from ibeis.control import __SQLITE3__ as lite
from ibeis.control import DB_SCHEMA
from ibeis.control import IBEISControl
from ibeis.control import accessor_decors
# IBEIS
import ibeis
from ibeis import model
from ibeis import control
from ibeis import gui
from ibeis import viz
from ibeis import main_module
from ibeis.viz import interact
from ibeis.model import hots
from ibeis.model import preproc


(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[all_imports]', DEBUG=False)


def reload_all():
    rrr()
    guiback.rrr()
    ibeis.reload_subs()
    utool.reload_subs()
    #vtool.reload_subs()
    #guitool.reload_subs()
    plottool.reload_subs()

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
    ui = front.ui
    utool.embed()
