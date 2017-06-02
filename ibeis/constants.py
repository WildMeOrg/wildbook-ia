# -*- coding: utf-8 -*-
"""
It is better to use constant variables instead of hoping you spell the same
string correctly every time you use it. (Also it makes it much easier if a
string name changes)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
# import utool
# import six
import numpy as np
from collections import OrderedDict
import math
from os.path import join
import utool as ut
ut.noinject('[const]')


PI  = math.pi
TAU = 2.0 * PI


class VIEW(object):
    """ simplified viewpoint """
    UNKNOWN = None
    R  = 1
    FR = 2
    F  = 3
    FL = 4
    L  = 5
    BL = 6
    B  = 7
    BR = 8

    INT_TO_CODE = ut.odict([
        (UNKNOWN, 'unknown'),
        (R,  'right'),
        (FR, 'frontright'),
        (F,  'front'),
        (FL, 'frontleft'),
        (L,  'left'),
        (BL, 'backleft'),
        (B,  'back'),
        (BR, 'backright'),
    ])

    INT_TO_NICE = ut.odict([
        (UNKNOWN, 'Unknown'),
        (R,  'Right'),
        (FR, 'Front-Right'),
        (F,  'Front'),
        (FL, 'Front-Left'),
        (L,  'Left'),
        (BL, 'Back-Left'),
        (B,  'Back'),
        (BR, 'Back-Right'),
    ])

    CODE_TO_NICE = ut.map_keys(INT_TO_CODE, INT_TO_NICE)
    CODE_TO_INT = ut.invert_dict(INT_TO_CODE)
    NICE_TO_CODE = ut.invert_dict(CODE_TO_NICE)
    NICE_TO_INT  = ut.invert_dict(INT_TO_NICE)

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
VIEWTEXT_TO_VIEWPOINT_RADIANS = VIEWTEXT_TO_YAW_RADIANS

YAWALIAS = {
    'up':             'U',
    'down':           'D',
    'front':          'F',
    'left':           'L',
    'back':           'B',
    'right':          'R',
    'upfront':        'UF',
    'upback':         'UB',
    'upleft':         'UL',
    'upright':        'UR',
    'downfront':      'DF',
    'downback':       'DB',
    'downleft':       'DL',
    'downright':      'DR',
    'frontleft':      'FL',
    'frontright':     'FR',
    'backleft':       'BL',
    'backright':      'BR',
    'upfrontleft':    'UFL',
    'upfrontright':   'UFR',
    'upbackleft':     'UBL',
    'upbackright':    'UBR',
    'downfrontleft':  'DFL',
    'downfrontright': 'DFR',
    'downbackleft':   'DBL',
    'downbackright':  'DBR',
}
VIEWPOINTALIAS = YAWALIAS

YAWALIAS_NICE = {
    'up':             'Up',
    'down':           'Down',
    'front':          'Front',
    'left':           'Left',
    'back':           'Back',
    'right':          'Right',
    'upfront':        'Up-Front',
    'upback':         'Up-Back',
    'upleft':         'Up-Left',
    'upright':        'Up-Right',
    'downfront':      'Down-Front',
    'downback':       'Down-Back',
    'downleft':       'Down-Left',
    'downright':      'Down-Right',
    'frontleft':      'Front-Left',
    'frontright':     'Front-Right',
    'backleft':       'Back-Left',
    'backright':      'Back-Right',
    'upfrontleft':    'Up-Front-Left',
    'upfrontright':   'Up-Front-Right',
    'upbackleft':     'Up-Back-Left',
    'upbackright':    'Up-Back-Right',
    'downfrontleft':  'Down-Front-Left',
    'downfrontright': 'Down-Front-Right',
    'downbackleft':   'Down-Back-Left',
    'downbackright':  'Down-Back-Right',
}
VIEWPOINTALIAS = YAWALIAS_NICE

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
    sqldb       = '_ibeis_database.sqlite3'
    sqlstaging  = '_ibeis_staging.sqlite3'
    _ibsdb      = '_ibsdb'
    cache       = '_ibeis_cache'
    backups     = '_ibeis_backups'
    logs        = '_ibeis_logs'
    chips       = 'chips'
    figures     = 'figures'
    flann       = 'flann'
    images      = 'images'
    trees       = 'trees'
    nets        = 'nets'
    uploads     = 'uploads'
    detectimg   = 'detectimg'
    thumbs      = 'thumbs'
    trashdir    = 'trashed_images'
    distinctdir = 'distinctiveness_model'
    scorenormdir = 'scorenorm'
    smartpatrol = 'smart_patrol'
    # Query Results (chipmatch dirs)
    qres       = 'qres_new'
    bigcache   = 'qres_bigcache_new'


class REL_PATHS(object):
    """ all paths are relative to ibs.dbdir """
    _ibsdb      = PATH_NAMES._ibsdb
    trashdir    = PATH_NAMES.trashdir
    figures     = join(_ibsdb, PATH_NAMES.figures)
    cache       = join(_ibsdb, PATH_NAMES.cache)
    backups     = join(_ibsdb, PATH_NAMES.backups)
    logs        = join(_ibsdb, PATH_NAMES.logs)
    #chips       = join(_ibsdb, PATH_NAMES.chips)
    images      = join(_ibsdb, PATH_NAMES.images)
    trees       = join(_ibsdb, PATH_NAMES.trees)
    nets        = join(_ibsdb, PATH_NAMES.nets)
    uploads     = join(_ibsdb, PATH_NAMES.uploads)
    # All computed dirs live in <dbdir>/_ibsdb/_ibeis_cache
    chips       = join(cache, PATH_NAMES.chips)
    thumbs      = join(cache, PATH_NAMES.thumbs)
    flann       = join(cache, PATH_NAMES.flann)
    qres        = join(cache, PATH_NAMES.qres)
    bigcache    = join(cache, PATH_NAMES.bigcache)
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
# CORE DATABASE
AL_RELATION_TABLE    = 'annotation_lblannot_relationship'
GA_RELATION_TABLE    = 'annotgroup_annotation_relationship'
ANNOTGROUP_TABLE     = 'annotgroups'
ANNOTATION_TABLE     = 'annotations'
PART_TABLE           = 'parts'
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
# STAGING DATABASE
REVIEW_TABLE         = 'reviews'
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
    # _DEFAULT_WILDBOOK_TARGET = 'lewa2'
    _DEFAULT_WILDBOOK_TARGET = 'lewa3'
elif ut.get_computer_name() == 'Leviathan':
    # _DEFAULT_WILDBOOK_TARGET = 'wildbook'
    _DEFAULT_WILDBOOK_TARGET = 'lewa3'
else:
    _DEFAULT_WILDBOOK_TARGET = 'ibeis'
WILDBOOK_TARGET = ut.get_argval('--wildbook-target', type_=str, default=_DEFAULT_WILDBOOK_TARGET,
                                help_='specify the Wildbook target deployment')


class ZIPPED_URLS(object):
    PZ_MTEST       = 'https://lev.cs.rpi.edu/public/databases/PZ_MTEST.zip'
    NAUTS          = 'https://lev.cs.rpi.edu/public/databases/NAUT_test.zip'
    WDS            = 'https://lev.cs.rpi.edu/public/databases/wd_peter2.zip'
    PZ_DISTINCTIVE = 'https://lev.cs.rpi.edu/public/models/distinctivness_zebra_plains.zip'  # DEPRICATE
    GZ_DISTINCTIVE = 'https://lev.cs.rpi.edu/public/models/distinctivness_zebra_grevys.zip'


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
    LYNX            = 'lynx'
    CHEETAH         = 'cheetah'
    SHARK_SANDTIGER = 'shark_sandtiger'
    UNKNOWN         = UNKNOWN


SPECIES_WITH_DETECTORS = (
    TEST_SPECIES.ZEB_GREVY,
    TEST_SPECIES.ZEB_PLAIN,
    TEST_SPECIES.WHALE_FLUKE,
    TEST_SPECIES.WHALE_HUMPBACK,
    TEST_SPECIES.LYNX,
    TEST_SPECIES.CHEETAH,
)

SPECIES_MAPPING = {
    'bear_polar'          :          ('PB', 'Polar Bear'),
    'bird_generic'        :        ('BIRD', 'Bird (Generic)'),
    'building'            :    ('BUILDING', 'Building'),
    'cheetah'             :        ('CHTH', 'Cheetah'),
    'dog_wild'            :          ('WD', 'Wild Dog'),
    'elephant_savanna'    :        ('ELEP', 'Elephant (Savanna)'),
    'elephant_savannah'   :        ('ELEP', 'Elephant (Savanna)'),
    'frog'                :        ('FROG', 'Frog'),
    'giraffe_masai'       :        ('GIRM', 'Giraffe (Masai)'),
    'giraffe_reticulated' :         ('GIR', 'Giraffe (Reticulated)'),
    'hyena'               :       ('HYENA', 'Hyena'),
    'jaguar'              :         ('JAG', 'Jaguar'),
    'leopard'             :        ('LOEP', 'Leopard'),
    'lion'                :        ('LION', 'Lion'),
    'lionfish'            :          ('LF', 'Lionfish'),
    'lynx'                :        ('LYNX', 'Lynx'),
    'nautilus'            :        ('NAUT', 'Nautilus'),
    'other'               :       ('OTHER', 'Other'),
    'rhino_black'         :      ('BRHINO', 'Rhino (Black)'),
    'rhino_white'         :      ('WRHINO', 'Rhino (White)'),
    'seal_saimma_ringed'  :       ('SEAL2', 'Seal (Siamaa Ringed)'),
    'seal_spotted'        :       ('SEAL1', 'Seal (Spotted)'),
    'shark_sandtiger'     :         ('STS', 'Sand Tiger Shark'),
    'snail'               :       ('SNAIL', 'Snail'),
    'snow_leopard'        :       ('SLEOP', 'Snow Leopard'),
    'tiger'               :       ('TIGER', 'Tiger'),
    'turtle_sea_generic'  :          ('ST', 'Sea Turtle (Generic)'),
    'turtle_hawksbill'    :        ('STHB', 'Sea Turtle (Hawksbill)'),
    'turtle_green'        :         ('STG', 'Sea Turtle (Green)'),
    'toads_wyoming'       :      ('WYTOAD', 'Toad (Wyoming)'),
    'unspecified'         : ('UNSPECIFIED', 'Unspecified'),
    'water_buffalo'       :        ('BUFF', 'Water Buffalo'),
    'wildebeest'          :          ('WB', 'Wildebeest'),
    'whale_fluke'         :          ('WF', 'Whale Fluke'),
    'whale_humpback'      :          ('HW', 'Humpback Whale'),
    'whale_shark'         :          ('WS', 'Whale Shark'),
    'zebra_grevys'        :          ('GZ', 'Zebra (Grevy\'s)'),
    'zebra_hybrid'        :          ('HZ', 'Zebra (Hybrid)'),
    'zebra_plains'        :          ('PZ', 'Zebra (Plains)'),
    UNKNOWN               :     ('UNKNOWN', 'Unknown'),
}


class REVIEW(object):
    """
    Enumerated types of review codes and texts

    Notes:
        Unreviewed: Not comparared yet.
        nomatch: Visually comparable and the different
        match: Visually comparable and the same
        notcomp: Not comparable means it is actually impossible to determine.
        unknown: means that it was reviewed, but we just can't figure it out.
    """
    UNREVIEWED   = None
    NEGATIVE     = 0
    POSITIVE     = 1
    INCOMPARABLE = 2
    UNKNOWN      = 3

    INT_TO_CODE = ut.odict([
        (POSITIVE       , 'match'),
        (NEGATIVE       , 'nomatch'),
        (INCOMPARABLE   , 'notcomp'),
        (UNKNOWN        , 'unknown'),
        (UNREVIEWED     , 'unreviewed'),
    ])

    INT_TO_NICE = ut.odict([
        (POSITIVE       , 'Positive'),
        (NEGATIVE       , 'Negative'),
        (INCOMPARABLE   , 'Incomparable'),
        (UNKNOWN        , 'Unknown'),
        (UNREVIEWED     , 'Unreviewed'),
    ])

    CODE_TO_NICE = ut.map_keys(INT_TO_CODE, INT_TO_NICE)
    CODE_TO_INT = ut.invert_dict(INT_TO_CODE)
    NICE_TO_CODE = ut.invert_dict(CODE_TO_NICE)
    NICE_TO_INT  = ut.invert_dict(INT_TO_NICE)

    MATCH_CODE = CODE_TO_INT


class CONFIDENCE(object):
    UNKNOWN         = None
    GUESSING        = 1
    NOT_SURE        = 2
    PRETTY_SURE     = 3
    ABSOLUTELY_SURE = 4

    INT_TO_CODE = ut.odict([
        (ABSOLUTELY_SURE, 'absolutely_sure'),
        (PRETTY_SURE, 'pretty_sure'),
        (NOT_SURE, 'not_sure'),
        (GUESSING, 'guessing'),
        (UNKNOWN, 'unspecified'),
    ])

    INT_TO_NICE = ut.odict([
        (ABSOLUTELY_SURE, 'Doubtless'),
        (PRETTY_SURE, 'Sure'),
        (NOT_SURE, 'Unsure'),
        (GUESSING, 'Guessing'),
        (UNKNOWN, 'Unspecified'),
    ])

    CODE_TO_NICE = ut.map_keys(INT_TO_CODE, INT_TO_NICE)
    CODE_TO_INT = ut.invert_dict(INT_TO_CODE)
    NICE_TO_CODE = ut.invert_dict(CODE_TO_NICE)
    NICE_TO_INT  = ut.invert_dict(INT_TO_NICE)
