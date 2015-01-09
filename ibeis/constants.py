"""
It is better to use constant variables instead of hoping you spell the same
string correctly every time you use it. (Also it makes it much easier if a
string name changes)
"""
from __future__ import absolute_import, division, print_function
# import utool
import six
import numpy as np
from os.path import join


class PATH_NAMES(object):
    """ Path names for internal IBEIS database """
    sqldb      = '_ibeis_database.sqlite3'
    sqldbcache = '_ibeis_database_cache.sqlite3'
    _ibsdb     = '_ibsdb'
    cache      = '_ibeis_cache'
    backups    = '_ibeis_backups'
    chips      = 'chips'
    flann      = 'flann'
    images     = 'images'
    trees      = 'trees'
    qres       = 'qres'
    bigcache   = 'qres_bigcache'
    detectimg  = 'detectimg'
    thumbs     = 'thumbs'
    trashdir   = 'trashed_images'
    distinctdir = 'distinctiveness_model'
    scorenormdir = 'scorenorm'


class REL_PATHS(object):
    """ all paths are relative to ibs.dbdir """
    _ibsdb   = PATH_NAMES._ibsdb
    trashdir = PATH_NAMES.trashdir
    cache    = join(_ibsdb, PATH_NAMES.cache)
    backups  = join(_ibsdb, PATH_NAMES.backups)
    chips    = join(_ibsdb, PATH_NAMES.chips)
    images   = join(_ibsdb, PATH_NAMES.images)
    trees    = join(_ibsdb, PATH_NAMES.trees)
    # All computed dirs live in <dbdir>/_ibsdb/_ibeis_cache
    thumbs   = join(cache, PATH_NAMES.thumbs)
    flann    = join(cache, PATH_NAMES.flann)
    qres     = join(cache, PATH_NAMES.qres)
    bigcache = join(cache, PATH_NAMES.bigcache)
    distinctdir = join(cache, PATH_NAMES.distinctdir)


# TODO: Remove anything under this block completely


UNKNOWN_LBLANNOT_ROWID = 0
UNKNOWN_NAME_ROWID = 0
UNKNOWN_SPECIES_ROWID = 0
# Names normalized to the standard UNKNOWN_NAME
ACCEPTED_UNKNOWN_NAMES = set(['Unassigned'])

# Name used to denote that idkwtfthisis
ENCTEXT_PREFIX = 'enc_'

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
ANNOTATION_TABLE     = 'annotations'
CHIP_TABLE           = 'chips'
CONFIG_TABLE         = 'configs'
CONTRIBUTOR_TABLE    = 'contributors'
EG_RELATION_TABLE    = 'encounter_image_relationship'
ENCOUNTER_TABLE      = 'encounters'
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
SPECIES_TABLE        = 'species'
RESIDUAL_TABLE       = 'residuals'
VERSIONS_TABLE       = 'versions'
#################################################################


UNKNOWN_PURPLE_RGBA255 = np.array((102,   0, 153, 255))
NAME_BLUE_RGBA255      = np.array((20, 20, 235, 255))
NAME_RED_RGBA255       = np.array((235, 20, 20, 255))
NEW_YELLOW_RGBA255     = np.array((235, 235, 20, 255))

UNKNOWN_PURPLE_RGBA01 = UNKNOWN_PURPLE_RGBA255 / 255.0
NAME_BLUE_RGBA01      = NAME_BLUE_RGBA255 / 255.0
NAME_RED_RGBA01       = NAME_RED_RGBA255 / 255.0
NEW_YELLOW_RGBA01     = NEW_YELLOW_RGBA255 / 255.0

EXEMPLAR_ENCTEXT         = 'Exemplars'
ALL_IMAGE_ENCTEXT        = 'All Images'
UNREVIEWED_IMAGE_ENCTEXT = 'Unreviewed Images'
REVIEWED_IMAGE_ENCTEXT   = 'Reviewed Images'
UNGROUPED_IMAGES_ENCTEXT = 'Ungrouped Images'
SPECIAL_ENCOUNTER_LABELS = [EXEMPLAR_ENCTEXT,
                            ALL_IMAGE_ENCTEXT,
                            UNREVIEWED_IMAGE_ENCTEXT,
                            REVIEWED_IMAGE_ENCTEXT,
                            UNGROUPED_IMAGES_ENCTEXT]
NEW_ENCOUNTER_ENCTEXT = 'NEW ENCOUNTER'

#IMAGE_THUMB_SUFFIX = '_thumb.png'
#CHIP_THUMB_SUFFIX  = '_chip_thumb.png'
IMAGE_THUMB_SUFFIX = '_thumb.jpg'
CHIP_THUMB_SUFFIX  = '_chip_thumb.jpg'

# FIXME UNKNOWN should not be a valid species


class Species(object):
    ZEB_PLAIN     = 'zebra_plains'
    ZEB_GREVY     = 'zebra_grevys'
    GIRAFFE       = 'giraffe'
    ELEPHANT_SAV  = 'elephant_savanna'
    JAG           = 'jaguar'
    LEOPARD       = 'leopard'
    LION          = 'lion'
    WILDDOG       = 'wild_dog'
    WHALESHARK    = 'whale_shark'
    SNAILS        = 'snails'
    SEALS_SPOTTED = 'seals_spotted'
    SEALS_RINGED  = 'seals_saimma_ringed'
    POLAR_BEAR    = 'bear_polar'
    FROGS         = 'frogs'
    LIONFISH      = 'lionfish'
    WYTOADS       = 'toads_wyoming'
    RHINO_BLACK   = 'rhino_black'
    RHINO_WHITE   = 'rhino_white'
    WILDEBEEST    = 'wildebeest'
    WATER_BUFFALO = 'water_buffalo'
    CHEETAH       = 'cheetah'
    UNKNOWN       = UNKNOWN

SPECIES_TUPS = [
    (Species.ZEB_PLAIN,     'Zebra (Plains)'),
    (Species.ZEB_GREVY,     'Zebra (Grevy\'s)'),
    (Species.GIRAFFE,       'Giraffes'),
    (Species.ELEPHANT_SAV,  'Elephant (savanna)'),
]

SPECIES_EXTRA = [
    (Species.JAG,           'Jaguar'),
    (Species.LEOPARD,       'Leopard'),
    (Species.LION,          'Lion'),
    (Species.WILDDOG,       'Wild Dog'),
    (Species.LIONFISH,      'Lionfish'),
    (Species.WHALESHARK,    'Whale Shark'),
    (Species.POLAR_BEAR,    'Polar Bear'),
    (Species.WILDEBEEST,    'Wildebeest'),
    (Species.SEALS_SPOTTED, 'Seal (spotted)'),
    (Species.SEALS_RINGED,  'Seal (Siamaa Ringed)'),
    (Species.CHEETAH,       'Cheetah'),
    (Species.UNKNOWN,       'Unknown'),
]

SPECIES_WITH_DETECTORS = (
    Species.ZEB_GREVY,
    Species.ZEB_PLAIN,
    Species.GIRAFFE,
)

VALID_SPECIES = [tup[0] for tup in SPECIES_TUPS]
SPECIES_NICE = [tup[1] for tup in SPECIES_TUPS]


VS_EXEMPLARS_KEY = 'vs_exemplars'
INTRA_ENC_KEY = 'intra_encounter'

HARD_NOTE_TAG = '<HARDCASE>'


if six.PY2:
    __STR__ = unicode  # change to str if needed
else:
    __STR__ = str


# clean namespace
#del utool
#del np
