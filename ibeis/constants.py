from __future__ import absolute_import, division, print_function
import utool
import six
import numpy as np


class PATH_NAMES(object):
    """ Path names for internal IBEIS database """
    sqldb      = '_ibeis_database.sqlite3'
    sqldbcache = '_ibeis_database_cache.sqlite3'
    _ibsdb     = '_ibsdb'
    cache      = '_ibeis_cache'
    chips      = 'chips'
    flann      = 'flann'
    images     = 'images'
    qres       = 'qres'
    bigcache   = 'bigcache'
    detectimg  = 'detectimg'
    thumbs     = 'thumbs'
    trashdir   = 'trashed_images'

UNKNOWN_LBLANNOT_ROWID = 0
# Names normalized to the standard UNKNOWN_NAME
ACCEPTED_UNKNOWN_NAMES = set(['Unassigned'])

# Name used to denote that idkwtfthisis
ENCTEXT_PREFIX = 'enc_'

INDIVIDUAL_KEY = 'INDIVIDUAL_KEY'
SPECIES_KEY    = 'SPECIES_KEY'
EMPTY_KEY      = ''
UNKNOWN        = '____'
KEY_DEFAULTS   = {
    INDIVIDUAL_KEY : '____',
    SPECIES_KEY    : '____',
}

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

BASE_DATABASE_VERSION = '0.0.0'

# DO NOT DELETE FROM THE TABLE LIST, THE DATABASE UPDATER WILL BREAK!!!
#################################################################
AL_RELATION_TABLE = 'annotation_lblannot_relationship'
ANNOTATION_TABLE  = 'annotations'
CHIP_TABLE        = 'chips'
CONFIG_TABLE      = 'configs'
CONTRIBUTOR_TABLE = 'contributors'
EG_RELATION_TABLE = 'encounter_image_relationship'
ENCOUNTER_TABLE   = 'encounters'
FEATURE_TABLE     = 'features'
GL_RELATION_TABLE = 'image_lblimage_relationship'
IMAGE_TABLE       = 'images'
LBLANNOT_TABLE    = 'lblannot'
LBLIMAGE_TABLE    = 'lblimage'
LBLTYPE_TABLE     = 'keys'
METADATA_TABLE    = 'metadata'
RESIDUAL_TABLE    = 'residuals'
VERSIONS_TABLE    = 'versions'
#################################################################


UNKNOWN_PURPLE_RGBA255 = np.array((102,   0, 153, 255))
NAME_BLUE_RGBA255      = np.array((20, 20, 235, 255))
NAME_RED_RGBA255       = np.array((235, 20, 20, 255))
NEW_YELLOW_RGBA255     = np.array((235, 235, 20, 255))

UNKNOWN_PURPLE_RGBA01 = UNKNOWN_PURPLE_RGBA255 / 255.0
NAME_BLUE_RGBA01      = NAME_BLUE_RGBA255 / 255.0
NAME_RED_RGBA01       = NAME_RED_RGBA255 / 255.0
NEW_YELLOW_RGBA01     = NEW_YELLOW_RGBA255 / 255.0

EXEMPLAR_ENCTEXT = 'Exemplars'
ALL_IMAGE_ENCTEXT = 'All Images'
UNREVIEWED_IMAGE_ENCTEXT = 'Unreviewed Images'
REVIEWED_IMAGE_ENCTEXT = 'Reviewed Images'

#IMAGE_THUMB_SUFFIX = '_thumb.png'
#CHIP_THUMB_SUFFIX  = '_chip_thumb.png'
IMAGE_THUMB_SUFFIX = '_thumb.jpg'
CHIP_THUMB_SUFFIX  = '_chip_thumb.jpg'

# FIXME UNKNOWN should not be a valid species


class Species(object):
    ZEB_PLAIN    = 'zebra_plains'
    ZEB_GREVY    = 'zebra_grevys'
    GIRAFFE      = 'giraffe'
    ELEPHANT_SAV = 'elephant_savanna'
    JAG          = 'jaguar'
    LEOPARD      = 'leopard'
    LION         = 'lion'
    WILDDOG      = 'wild_dog'
    WHALESHARK   = 'whale_shark'
    SNAILS       = 'snails'
    SEALS        = 'seals_spotted'
    POLAR_BEAR   = 'bear_polar'
    FROGS        = 'frogs'
    LIONFISH     = 'lionfish'
    WYTOADS      = 'toads_wyoming'
    RHINO_BLACK  = 'rhino_black'
    RHINO_WHITE  = 'rhino_white'
    WILDEBEEST   = 'wildebeest'
    WATER_BUFFALO = 'water_buffalo'
    UNKNOWN      = UNKNOWN

SPECIES_TUPS = [
    (Species.ZEB_PLAIN,    'Zebra (Plains)'),
    (Species.ZEB_GREVY,    'Zebra (Grevy\'s)'),
    (Species.GIRAFFE,      'Giraffes'),
    (Species.ELEPHANT_SAV, 'Elephant (savanna)'),
    (Species.JAG,          'Jaguar'),
    (Species.LEOPARD,      'Leopard'),
    (Species.LION,         'Lion'),
    (Species.WILDDOG,      'Wild Dog'),
    (Species.LIONFISH,     'Lionfish'),
    (Species.WHALESHARK,   'Whale Shark'),
    (Species.POLAR_BEAR,   'Polar Bear'),
    (Species.WILDEBEEST,   'Wildebeest'),
    (Species.UNKNOWN,      'Unknown'),
]

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
del utool
del np
