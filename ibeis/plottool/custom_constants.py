from __future__ import absolute_import, division, print_function
from six.moves import map
import utool
import sys
import numpy as np
import matplotlib as mpl
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[df2]', DEBUG=False)
# GENERAL FONTS

SMALLEST = 6
SMALLER  = 8
SMALL    = 10
MED      = 12
LARGE    = 14
#fpargs = dict(family=None, style=None, variant=None, stretch=None, fname=None)


def FontProp(*args, **kwargs):
    """ overwrite fontproperties with custom settings """
    kwargs['family'] = 'monospace'
    return mpl.font_manager.FontProperties(*args, **kwargs)

FONTS = utool.DynStruct()
FONTS.smallest  = FontProp(weight='light', size=SMALLEST)
FONTS.small     = FontProp(weight='light', size=SMALL)
FONTS.smaller   = FontProp(weight='light', size=SMALLER)
FONTS.med       = FontProp(weight='light', size=MED)
FONTS.large     = FontProp(weight='light', size=LARGE)
FONTS.medbold   = FontProp(weight='bold', size=MED)
FONTS.largebold = FontProp(weight='bold', size=LARGE)

# SPECIFIC FONTS

FONTS.legend   = FONTS.small
FONTS.figtitle = FONTS.med
FONTS.axtitle  = FONTS.small
FONTS.subtitle = FONTS.med
#FONTS.xlabel   = FONTS.smaller
FONTS.xlabel   = FONTS.small
FONTS.ylabel   = FONTS.small
FONTS.relative = FONTS.smallest

# COLORS

ORANGE       = np.array((255, 127,   0, 255)) / 255.0
RED          = np.array((255,   0,   0, 255)) / 255.0
GREEN        = np.array((  0, 255,   0, 255)) / 255.0
BLUE         = np.array((  0,   0, 255, 255)) / 255.0
YELLOW       = np.array((255, 255,   0, 255)) / 255.0
BLACK        = np.array((  0,   0,   0, 255)) / 255.0
WHITE        = np.array((255, 255, 255, 255)) / 255.0
GRAY         = np.array((127, 127, 127, 255)) / 255.0
DEEP_PINK    = np.array((255,  20, 147, 255)) / 255.0
PINK         = np.array((255, 100, 100, 255)) / 255.0
FALSE_RED    = np.array((255,  51,   0, 255)) / 255.0
TRUE_GREEN   = np.array((  0, 255,   0, 255)) / 255.0
DARK_GREEN   = np.array((  0, 127,   0, 255)) / 255.0
DARK_BLUE    = np.array((  0,   0, 127, 255)) / 255.0
DARK_RED     = np.array((127,  0,    0, 255)) / 255.0
DARK_ORANGE  = np.array((127,  63,   0, 255)) / 255.0
DARK_YELLOW  = np.array((127, 127,   0, 255)) / 255.0
PURPLE       = np.array((102,   0, 153, 255)) / 255.0
LIGHT_BLUE   = np.array((102, 100, 255, 255)) / 255.0
UNKNOWN_PURP = PURPLE


# GOLDEN RATIOS
PHI_numer = 1 + np.sqrt(5)
PHI_denom = 2.0
PHI = PHI_numer / PHI_denom

DARKEN = .3 if '--darken' in sys.argv else None


def golden_wh2(sz):
    return (PHI * sz, sz)


def golden_wh(x):
    'returns a width / height with a golden aspect ratio'
    return list(map(int, list(map(round, (x * .618, x * .312)))))


# FIGURE GEOMETRY
#DPI = 80
DPI = 60
#DPI = 160
#DPI = 360
#FIGSIZE = (24) # default windows fullscreen
FIGSIZE_MED = (12, 6)
FIGSIZE_SQUARE = (12, 12)
FIGSIZE_GOLD = golden_wh2(8)
FIGSIZE_BIGGER = (24, 12)
FIGSIZE_HUGE = (32, 16)

FIGSIZE = FIGSIZE_MED
# Quality drawings
#FIGSIZE = FIGSIZE_SQUARE
#DPI = 120

base_fnum = 9001
