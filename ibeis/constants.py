"""
It is better to use constant variables instead of hoping you spell the same
string correctly every time you use it. (Also it makes it much easier if a
string name changes)
"""
from __future__ import absolute_import, division, print_function
# import utool
import six
import numpy as np
from collections import namedtuple, OrderedDict
import math
from os.path import join
import utool as ut
ut.noinject('[const]')


TAU = 2 * math.pi


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
    (0,  QUAL_JUNK,),
    (-1, QUAL_UNKNOWN,),
])

QUALITY_TEXT_TO_INT = ut.invert_dict(QUALITY_INT_TO_TEXT)
QUALITY_INT_TO_TEXT[None] = QUALITY_INT_TO_TEXT[-1]


class PATH_NAMES(object):
    """ Path names for internal IBEIS database """
    sqldb      = '_ibeis_database.sqlite3'
    sqldbcache = '_ibeis_database_cache.sqlite3'
    _ibsdb     = '_ibsdb'
    cache      = '_ibeis_cache'
    backups    = '_ibeis_backups'
    chips      = 'chips'
    figures    = 'figures'
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
    smartpatrol = 'smart_patrol'


class REL_PATHS(object):
    """ all paths are relative to ibs.dbdir """
    _ibsdb   = PATH_NAMES._ibsdb
    trashdir = PATH_NAMES.trashdir
    figures  = join(_ibsdb, PATH_NAMES.figures)
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
ANNOTMATCH_TABLE     = 'annotmatch'
SPECIES_TABLE        = 'species'
RESIDUAL_TABLE       = 'residuals'
VERSIONS_TABLE       = 'versions'
#
PARTY_CONTRIB_RELATION_TABLE = 'party_contrib_relation'
PARTY_TABLE                  = 'party'
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
    # TODO change to giraffe_reticulated
    GIRAFFE       = 'giraffe'
    GIRAFFE_MASAI = 'giraffe_masai'
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
    TIGER         = 'tiger'
    HYENA         = 'hyena'
    NAUT          = 'nautilus'
    UNKNOWN       = UNKNOWN

# TODO: allow for custom species
SpeciesTupleProperties = ('species_text', 'species_code', 'species_nice')
SpeciesTuple = namedtuple('SpeciesTuple', SpeciesTupleProperties)

SPECIES_TUPS = [
    SpeciesTuple(Species.ZEB_PLAIN,          'PZ', 'Zebra (Plains)'),
    SpeciesTuple(Species.ZEB_GREVY,          'GZ', 'Zebra (Grevy\'s)'),
    SpeciesTuple(Species.GIRAFFE,           'GIR', 'Giraffes (Reticulated)'),
    SpeciesTuple(Species.GIRAFFE_MASAI,    'GIRM', 'Giraffes (Masai)'),
    SpeciesTuple(Species.ELEPHANT_SAV,     'ELEP', 'Elephant (savanna)'),
    SpeciesTuple(Species.POLAR_BEAR,         'PB', 'Polar Bear'),
    SpeciesTuple(Species.WILDDOG,            'WD', 'Wild Dog'),
    SpeciesTuple(Species.LIONFISH,           'LF', 'Lionfish'),
    SpeciesTuple(Species.WHALESHARK,         'WS', 'Whale Shark'),
    SpeciesTuple(Species.WILDEBEEST,         'WB', 'Wildebeest'),
    SpeciesTuple(Species.JAG,               'JAG', 'Jaguar'),
    SpeciesTuple(Species.LEOPARD,          'LOEP', 'Leopard'),
    SpeciesTuple(Species.TIGER,           'TIGER', 'Tiger'),
    SpeciesTuple(Species.HYENA,           'HYENA', 'Hyena'),
    SpeciesTuple(Species.LION,             'LION', 'Lion'),
    SpeciesTuple(Species.CHEETAH,          'CHTH', 'Cheetah'),
    SpeciesTuple(Species.SEALS_SPOTTED,   'SEAL1', 'Seal (spotted)'),
    SpeciesTuple(Species.SEALS_RINGED,    'SEAL2', 'Seal (Siamaa Ringed)'),
    SpeciesTuple(Species.NAUT,             'NAUT', 'Nautilus'),
    SpeciesTuple(Species.UNKNOWN,       'UNKNOWN', 'Unknown'),
]

# FIXME: infer this
SPECIES_WITH_DETECTORS = (
    Species.ZEB_GREVY,
    Species.ZEB_PLAIN,
    Species.GIRAFFE,
    #Species.GIRAFFE_MASAI,
    Species.ELEPHANT_SAV,
)

SPECIES_CODE_TO_TEXT = {
    tup.species_code: tup.species_text for tup in SPECIES_TUPS
}
SPECIES_TEXT_TO_CODE = {
    tup.species_text: tup.species_code for tup in SPECIES_TUPS
}

VALID_SPECIES = [tup.species_text for tup in SPECIES_TUPS]
SPECIES_NICE  = [tup.species_nice for tup in SPECIES_TUPS]


VS_EXEMPLARS_KEY = 'vs_exemplars'
INTRA_ENC_KEY = 'intra_encounter'

HARD_NOTE_TAG = '<HARDCASE>'
WILDBOOK_TARGET = 'prod'


def get_species_code(species_text_):
    " functions should not be in const """
    species_text = species_text_.lower()
    if species_text == 'none':
        species_text = Species.UNKNOWN
    species_code = SPECIES_TEXT_TO_CODE.get(species_text, species_text)
    return species_code


class ZIPPED_URLS(object):
    PZ_MTEST       = 'https://www.dropbox.com/s/xdae2yvsp57l4t2/PZ_MTEST.zip'
    NAUTS          = 'https://www.dropbox.com/s/8gt3eaiw8rb31rh/NAUT_test.zip'
    PZ_DISTINCTIVE = 'https://www.dropbox.com/s/gbp24qks9z3fzm6/distinctivness_zebra_plains.zip'
    GZ_DISTINCTIVE = 'https://www.dropbox.com/s/nb5gv7wibwo3ib4/distinctivness_zebra_grevys.zip'

if six.PY2:
    __STR__ = unicode  # change to str if needed
else:
    __STR__ = str


def get_working_species_set():
    """ hack to make only species with detectors show up """
    # TODO: FUNCTIONS SHOULD NOT BE IN CONSTANTS
    # TODO: allow for custom user-define species
    #RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS = not ut.get_argflag('--allspecies')
    RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS = ut.get_argflag('--no-allspecies')
    if RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS:
        working_species_tups = [
            (species_tup.species_nice, species_tup.species_text)
            for species_tup in SPECIES_TUPS
            if species_tup.species_text in SPECIES_WITH_DETECTORS
        ]
    else:
        working_species_tups = [
            (species_tup.species_nice, species_tup.species_text)
            for species_tup in SPECIES_TUPS
        ]
    return working_species_tups


TRUTH_UNKNOWN = 2
TRUTH_MATCH = 1
TRUTH_NOT_MATCH = 0


TRUTH_INT_TO_TEXT = {
    TRUTH_UNKNOWN   : 'Not Matched',  # 'Unknown',
    TRUTH_NOT_MATCH : 'Not Matched',
    TRUTH_MATCH     : 'Matched',
}

# clean namespace
#del utool
#del np
