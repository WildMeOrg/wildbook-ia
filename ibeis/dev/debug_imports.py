# flake8: noqa
from __future__ import division, print_function
# Python
from collections import OrderedDict, defaultdict
from os.path import (dirname, realpath, join, exists, normpath, splitext,
                     expanduser, relpath)
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
# Matplotlib
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
# Scientific
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
from scipy.cluster.hierarchy import fclusterdata
import networkx as netx
try:
    import graph_tool
except ImportError as ex:
    #print('Warning: %r' % ex)
    pass
# Qt
import PyQt4
from PyQt4 import QtCore, QtGui
from PyQt4.Qt import (QAbstractItemModel, QModelIndex, QVariant, QWidget,
                      Qt, QObject, pyqtSlot, QKeyEvent)
# UTool
import utool
from utool import *
# VTool
import vtool

from vtool import chip as ctool
from vtool import image as gtool
from vtool import histogram as htool
from vtool import patch as ptool
from vtool import keypoint as ktool
from vtool import linalg as ltool

from vtool import *
# DrawTool
import drawtool
from drawtool import draw_func2 as df2
# GUITool
import guitool
import drawtool
import ibeis
# IBEIS DEV
import ibeis.dev
from ibeis.dev import main_api
# IBEIS MODEL
import ibeis.model
from ibeis.model import Config
from ibeis.model.preproc import preproc_image
from ibeis.model.preproc import preproc_chip
from ibeis.model.preproc import preproc_feat
from ibeis.view import viz
from ibeis.view import interact
# IBEIS CONTROl
import ibeis.control


def get_ibeis_modules():
    ibeis_modules = [
        utool,
        vtool,
        guitool,
        drawtool
    ]
    return ibeis_modules
