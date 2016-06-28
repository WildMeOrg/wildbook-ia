# -*- coding: utf-8 -*-
"""
It is better to use constant variables instead of hoping you spell the same
string correctly every time you use it. (Also it makes it much easier if a
string name changes)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
# import utool
import six
import numpy as np
from collections import OrderedDict
import math
from os.path import join
import utool as ut
ut.noinject('[const]')


PI  = math.pi
TAU = 2.0 * PI

# Mapping of semantic viewpoints to yaw angles
VIEWTEXT_TO_YAW_RADIANS = OrderedDict([
    ('right'      , 0.000 * TAU,),
    ('frontright' , 0.125 * TAU,),
    ('front'      , 0.250 * TAU,),
    ('frontleft'  , 0.375 * TAU,),
    ('left'       , 0.500 * TAU,),
    ('backleft'   , 0.625 * TAU,),
    ('back'       , 0.750 * TAU,),
    ('backright'  , 0.875 * TAU,),
])

#VIEWTEXT_TO_QT_VIEWTEXT = {
#    'right'      : 'right',
#    'frontright' : 'frontright',
#    'front'      : 'front',
#    'frontleft'  : 'frontleft',
#    'left'       : 'left',
#    'backleft'   : 'backleft',
#    'back'       : 'back',
#    'backright'  : 'backright',
#}

YAWALIAS = {'frontleft': 'FL', 'frontright': 'FR', 'backleft': 'BL', 'backright': 'BR',
            'front': 'F', 'left': 'L', 'back': 'B', 'right': 'R', }

QUAL_EXCELLENT = 'excellent'
QUAL_GOOD      = 'good'
QUAL_OK        = 'ok'
QUAL_POOR      = 'poor'
QUAL_JUNK      = 'junk'
QUAL_UNKNOWN   = 'UNKNOWN'

QUALITY_INT_TO_TEXT = OrderedDict([
    (5,  QUAL_EXCELLENT,),
    (4,  QUAL_GOOD,),
    (3,  QUAL_OK,),
    (2,  QUAL_POOR,),
    # oops forgot 1. will be mapped to poor
    (0,  QUAL_JUNK,),
    (-1, QUAL_UNKNOWN,),
])

QUALITY_TEXT_TO_INT       = ut.invert_dict(QUALITY_INT_TO_TEXT)
QUALITY_INT_TO_TEXT[1]    = QUAL_JUNK
#QUALITY_TEXT_TO_INTS      = ut.invert_dict(QUALITY_INT_TO_TEXT)
QUALITY_TEXT_TO_INTS = ut.group_items(
    list(QUALITY_INT_TO_TEXT.keys()),
    list(QUALITY_INT_TO_TEXT.values()))
QUALITY_TEXT_TO_INTS[QUAL_UNKNOWN] = -1
QUALITY_INT_TO_TEXT[None] = QUALITY_INT_TO_TEXT[-1]


SEX_INT_TO_TEXT = {
    None: 'UNKNOWN NAME',
    -1  : 'UNKNOWN SEX',
    0   : 'Female',
    1   : 'Male',
}
SEX_TEXT_TO_INT = ut.invert_dict(SEX_INT_TO_TEXT)


class PATH_NAMES(object):
    """ Path names for internal IBEIS database """
    sqldb      = '_ibeis_database.sqlite3'
    _ibsdb     = '_ibsdb'
    cache      = '_ibeis_cache'
    backups    = '_ibeis_backups'
    chips      = 'chips'
    figures    = 'figures'
    flann      = 'flann'
    images     = 'images'
    trees      = 'trees'
    nets       = 'nets'
    uploads    = 'uploads'
    detectimg  = 'detectimg'
    thumbs     = 'thumbs'
    trashdir   = 'trashed_images'
    distinctdir = 'distinctiveness_model'
    scorenormdir = 'scorenorm'
    smartpatrol = 'smart_patrol'
    # Query Results (chipmatch dirs)
    qres       = 'qres_new'
    bigcache   = 'qres_bigcache_new'


class REL_PATHS(object):
    """ all paths are relative to ibs.dbdir """
    _ibsdb   = PATH_NAMES._ibsdb
    trashdir = PATH_NAMES.trashdir
    figures  = join(_ibsdb, PATH_NAMES.figures)
    cache    = join(_ibsdb, PATH_NAMES.cache)
    backups  = join(_ibsdb, PATH_NAMES.backups)
    #chips    = join(_ibsdb, PATH_NAMES.chips)
    images   = join(_ibsdb, PATH_NAMES.images)
    trees    = join(_ibsdb, PATH_NAMES.trees)
    nets     = join(_ibsdb, PATH_NAMES.nets)
    uploads  = join(_ibsdb, PATH_NAMES.uploads)
    # All computed dirs live in <dbdir>/_ibsdb/_ibeis_cache
    chips    = join(cache, PATH_NAMES.chips)
    thumbs   = join(cache, PATH_NAMES.thumbs)
    flann    = join(cache, PATH_NAMES.flann)
    qres     = join(cache, PATH_NAMES.qres)
    bigcache = join(cache, PATH_NAMES.bigcache)
    distinctdir = join(cache, PATH_NAMES.distinctdir)


# Directories that should be excluded from copy operations
EXCLUDE_COPY_REL_DIRS = [
    REL_PATHS.chips,
    REL_PATHS.cache,
    REL_PATHS.backups,
    REL_PATHS.figures,
    REL_PATHS.nets,
    join(PATH_NAMES._ibsdb, '_ibeis_cache*'),
    #'_ibsdb/_ibeis_cache',
    '_ibsdb/chips',  # old path for caches
    './images',  # the hotspotter images dir
]


# TODO: Remove anything under this block completely


UNKNOWN_LBLANNOT_ROWID = 0
UNKNOWN_NAME_ROWID = 0
UNKNOWN_SPECIES_ROWID = 0
# Names normalized to the standard UNKNOWN_NAME
ACCEPTED_UNKNOWN_NAMES = set(['Unassigned'])

INDIVIDUAL_KEY = 'INDIVIDUAL_KEY'
SPECIES_KEY    = 'SPECIES_KEY'
EMPTY_KEY      = ''
UNKNOWN        = '____'
KEY_DEFAULTS   = {
    INDIVIDUAL_KEY : UNKNOWN,
    SPECIES_KEY    : UNKNOWN,
}

# <UNFINISHED METADATA>
# We are letting wildbook do this metadata instead
# Define the special metadata for annotation

ROSEMARY_ANNOT_METADATA = [
    ('local_name'    , 'Local name:',    str),
    ('sun'           , 'Sun:',           ['FS', 'PS', 'NS']),
    ('wind'          , 'Wind:',          ['NW', 'LW', 'MW', 'SW']),
    ('rain'          , 'Rain:',          ['NR', 'LR', 'MR', 'HR']),
    ('cover'         , 'Cover:',         float),
    ('grass'         , 'Grass:',         ['less hf', 'less hk', 'less belly']),
    ('grass_color'   , 'Grass Colour:',  ['B', 'BG', 'GB', 'G']),
    ('grass_species' , 'Grass Species:', str),
    ('bush_type'     , 'Bush type:',     ['OG', 'LB', 'MB', 'TB']),
    ('bit'           , 'Bit:',           int),
    ('other_speceis' , 'Other Species:', str),
]

#ROSEMARY_KEYS = utool.get_list_column(ROSEMARY_ANNOT_METADATA, 0)
#KEY_DEFAULTS.update(**{key: UNKNOWN for key in ROSEMARY_KEYS})
# </UNFINISHED METADATA>

BASE_DATABASE_VERSION = '0.0.0'

#################################################################
# DO NOT DELETE FROM THE TABLE LIST, THE DATABASE UPDATER WILL BREAK!!!
# THIS GOES FOR OLD AND DEPRICATED TABLENAMES AS WELL!!!
# TODO:
# What should happen is when they are depricated they should go into a
# depricated tablename structure with the relevant versions suffixed
#################################################################
AL_RELATION_TABLE    = 'annotation_lblannot_relationship'
GA_RELATION_TABLE    = 'annotgroup_annotation_relationship'
ANNOTGROUP_TABLE     = 'annotgroups'
ANNOTATION_TABLE     = 'annotations'
CHIP_TABLE           = 'chips'
CONFIG_TABLE         = 'configs'
CONTRIBUTOR_TABLE    = 'contributors'
GSG_RELATION_TABLE   = 'imageset_image_relationship'
IMAGESET_TABLE       = 'imagesets'
FEATURE_TABLE        = 'features'
FEATURE_WEIGHT_TABLE = 'feature_weights'
GL_RELATION_TABLE    = 'image_lblimage_relationship'
IMAGE_TABLE          = 'images'
LBLANNOT_TABLE       = 'lblannot'
LBLIMAGE_TABLE       = 'lblimage'
LBLTYPE_TABLE        = 'keys'
METADATA_TABLE       = 'metadata'
# Ugly move from name to names, need better way of versioning old table names
NAME_TABLE_v121      = 'name'
NAME_TABLE_v130      = 'names'
NAME_TABLE           = NAME_TABLE_v130
ANNOTMATCH_TABLE     = 'annotmatch'
SPECIES_TABLE        = 'species'
RESIDUAL_TABLE       = 'residuals'
VERSIONS_TABLE       = 'versions'
#
PARTY_CONTRIB_RELATION_TABLE = 'party_contrib_relation'
PARTY_TABLE                  = 'party'
#################################################################


# DEPCACHE TABLENAMES
#CHIPTHUMB_TABLE = 'chipthumb'


UNKNOWN_PURPLE_RGBA255 = np.array((102,   0, 153, 255))
NAME_BLUE_RGBA255      = np.array((20, 20, 235, 255))
NAME_RED_RGBA255       = np.array((235, 20, 20, 255))
NEW_YELLOW_RGBA255     = np.array((235, 235, 20, 255))

UNKNOWN_PURPLE_RGBA01 = UNKNOWN_PURPLE_RGBA255 / 255.0
NAME_BLUE_RGBA01      = NAME_BLUE_RGBA255 / 255.0
NAME_RED_RGBA01       = NAME_RED_RGBA255 / 255.0
NEW_YELLOW_RGBA01     = NEW_YELLOW_RGBA255 / 255.0

EXEMPLAR_IMAGESETTEXT         = '*Exemplars'
ALL_IMAGE_IMAGESETTEXT        = '*All Images'
UNREVIEWED_IMAGE_IMAGESETTEXT = '*Undetected Images'
REVIEWED_IMAGE_IMAGESETTEXT   = '*Reviewed Detections'
UNGROUPED_IMAGES_IMAGESETTEXT = '*Ungrouped Images'
SPECIAL_IMAGESET_LABELS = [
    EXEMPLAR_IMAGESETTEXT,
    ALL_IMAGE_IMAGESETTEXT,
    UNREVIEWED_IMAGE_IMAGESETTEXT,
    REVIEWED_IMAGE_IMAGESETTEXT,
    UNGROUPED_IMAGES_IMAGESETTEXT
]
NEW_IMAGESET_IMAGESETTEXT = 'NEW IMAGESET'

#IMAGE_THUMB_SUFFIX = '_thumb.png'
#CHIP_THUMB_SUFFIX  = '_chip_thumb.png'
IMAGE_THUMB_SUFFIX = '_thumb.jpg'
IMAGE_BARE_THUMB_SUFFIX = '_thumb_bare.jpg'
CHIP_THUMB_SUFFIX  = '_chip_thumb.jpg'


VS_EXEMPLARS_KEY = 'vs_exemplars'
INTRA_OCCUR_KEY = 'intra_occurrence'

HARD_NOTE_TAG = '<HARDCASE>'

# HACK
if ut.get_computer_name() == 'ibeis.cs.uic.edu':
    #_DEFAULT_WILDBOOK_TARGET = 'prod'
    _DEFAULT_WILDBOOK_TARGET = 'lewa2'
else:
    _DEFAULT_WILDBOOK_TARGET = 'ibeis'
WILDBOOK_TARGET = ut.get_argval('--wildbook-target', type_=str, default=_DEFAULT_WILDBOOK_TARGET,
                                help_='specify the Wildbook target deployment')


class ZIPPED_URLS(object):
    PZ_MTEST       = 'https://lev.cs.rpi.edu/public/databases/PZ_MTEST.zip'
    NAUTS          = 'https://lev.cs.rpi.edu/public/databases/NAUT_test.zip'
    WDS            = 'https://lev.cs.rpi.edu/public/databases/wd_peter2.zip'
    PZ_DISTINCTIVE = 'https://lev.cs.rpi.edu/public/models/distinctivness_zebra_plains.zip'
    GZ_DISTINCTIVE = 'https://lev.cs.rpi.edu/public/models/distinctivness_zebra_grevys.zip'

if six.PY2:
    __STR__ = unicode  # change to str if needed
else:
    __STR__ = str


# TODO: rename to same / different
# add add match, nonmatch, notcomp
TRUTH_UNKNOWN = 2
TRUTH_MATCH = 1
TRUTH_NOT_MATCH = 0


TRUTH_INT_TO_TEXT = {
    TRUTH_UNKNOWN   : 'Unknown',
    TRUTH_NOT_MATCH : 'Not Matched',
    TRUTH_MATCH     : 'Matched',
}


# Turn off features at Lewa :(
SIMPLIFY_INTERFACE = (ut.get_computer_name() == 'ibeis.cs.uic.edu') or ut.get_argflag('--simplify')


# For candidacy document
DBNAME_ALIAS = {
    #'NNP_MasterGIRM_core': 'NNP_GIRM'
    #'NNP_MasterGIRM_core': 'GIRM',
    'NNP_MasterGIRM_core': 'GIRM',
    'PZ_Master1': 'PZ',
    'GZ_Master1': 'GZ',
    'GIRM_Master1': 'GIRM',
    'GZ_ALL': 'GZ',
}


class TEST_SPECIES(object):
    BEAR_POLAR      = 'bear_polar'
    BUILDING        = 'building'
    GIR_RETICULATED = 'giraffe_reticulated'
    GIR_MASAI       = 'giraffe_masai'
    WHALE_FLUKE     = 'whale_fluke',
    WHALE_HUMPBACK  = 'whale_humpback',
    ZEB_GREVY       = 'zebra_grevys'
    ZEB_HYBRID      = 'zebra_hybrid'
    ZEB_PLAIN       = 'zebra_plains'
    UNKNOWN         = UNKNOWN


SPECIES_WITH_DETECTORS = (
    TEST_SPECIES.ZEB_GREVY,
    TEST_SPECIES.ZEB_PLAIN,
    TEST_SPECIES.WHALE_FLUKE,
    TEST_SPECIES.WHALE_HUMPBACK,
)
