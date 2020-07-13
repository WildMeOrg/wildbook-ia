# -*- coding: utf-8 -*-
"""
It is better to use constant variables instead of hoping you spell the same
string correctly every time you use it. (Also it makes it much easier if a
string name changes)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import math
import utool as ut
from collections import OrderedDict
from os.path import join

(print, rrr, profile) = ut.inject2(__name__)


CONTAINERIZED = ut.get_argflag('--containerized')
PRODUCTION = ut.get_argflag('--production')
HTTPS = ut.get_argflag('--https')


CONTAINER_NAME = ut.get_argval(
    '--container-name', type_=str, default=ut.get_computer_name()
)
ENGINE_SLOT = ut.get_argval('--engine-slot', type_=str, default='default')


PI = math.pi
TAU = 2.0 * PI

VIEWTEXT_TO_YAW_RADIANS = OrderedDict(
    [
        ('right', 0.000 * TAU,),
        ('frontright', 0.125 * TAU,),
        ('front', 0.250 * TAU,),
        ('frontleft', 0.375 * TAU,),
        ('left', 0.500 * TAU,),
        ('backleft', 0.625 * TAU,),
        ('back', 0.750 * TAU,),
        ('backright', 0.875 * TAU,),
    ]
)

# Mapping of viewpoints codes to yaw angles
VIEWTEXT_TO_YAW_RADIANS = OrderedDict(
    [
        ('right', 0.000 * TAU,),
        ('frontright', 0.125 * TAU,),
        ('front', 0.250 * TAU,),
        ('frontleft', 0.375 * TAU,),
        ('left', 0.500 * TAU,),
        ('backleft', 0.625 * TAU,),
        ('back', 0.750 * TAU,),
        ('backright', 0.875 * TAU,),
    ]
)
VIEWTEXT_TO_VIEWPOINT_RADIANS = VIEWTEXT_TO_YAW_RADIANS


# TODO: DEPRICATE ALL THE YAW ALIAS STUFF IN FAVOR OF const.VIEW
YAWALIAS = {
    # maybe this can be CODE_TO_SHORT
    'up': 'U',
    'down': 'D',
    'front': 'F',
    'left': 'L',
    'back': 'B',
    'right': 'R',
    'upfront': 'UF',
    'upback': 'UB',
    'upleft': 'UL',
    'upright': 'UR',
    'downfront': 'DF',
    'downback': 'DB',
    'downleft': 'DL',
    'downright': 'DR',
    'frontleft': 'FL',
    'frontright': 'FR',
    'backleft': 'BL',
    'backright': 'BR',
    'upfrontleft': 'UFL',
    'upfrontright': 'UFR',
    'upbackleft': 'UBL',
    'upbackright': 'UBR',
    'downfrontleft': 'DFL',
    'downfrontright': 'DFR',
    'downbackleft': 'DBL',
    'downbackright': 'DBR',
}

QUAL_EXCELLENT = 'excellent'
QUAL_GOOD = 'good'
QUAL_OK = 'ok'
QUAL_POOR = 'poor'
QUAL_JUNK = 'junk'
QUAL_UNKNOWN = 'UNKNOWN'

QUALITY_INT_TO_TEXT = OrderedDict(
    [
        (5, QUAL_EXCELLENT,),
        (4, QUAL_GOOD,),
        (3, QUAL_OK,),
        (2, QUAL_POOR,),
        # oops forgot 1. will be mapped to poor
        (0, QUAL_JUNK,),
        (-1, QUAL_UNKNOWN,),
    ]
)


QUALITY_TEXT_TO_INT = ut.invert_dict(QUALITY_INT_TO_TEXT)
QUALITY_INT_TO_TEXT[1] = QUAL_JUNK
# QUALITY_TEXT_TO_INTS      = ut.invert_dict(QUALITY_INT_TO_TEXT)
QUALITY_TEXT_TO_INTS = ut.group_items(
    list(QUALITY_INT_TO_TEXT.keys()), list(QUALITY_INT_TO_TEXT.values())
)
QUALITY_TEXT_TO_INTS[QUAL_UNKNOWN] = -1
QUALITY_INT_TO_TEXT[None] = QUALITY_INT_TO_TEXT[-1]


SEX_INT_TO_TEXT = {
    None: 'UNKNOWN NAME',
    -1: 'UNKNOWN SEX',
    0: 'Female',
    1: 'Male',
    2: 'INDETERMINATE SEX',
}
SEX_TEXT_TO_INT = ut.invert_dict(SEX_INT_TO_TEXT)


class PATH_NAMES(object):  # NOQA
    """ Path names for internal IBEIS database """

    sqldb = '_ibeis_database.sqlite3'
    sqlstaging = '_ibeis_staging.sqlite3'
    _ibsdb = '_ibsdb'
    cache = '_ibeis_cache'
    backups = '_ibeis_backups'
    logs = '_ibeis_logs'
    chips = 'chips'
    figures = 'figures'
    flann = 'flann'
    images = 'images'
    trees = 'trees'
    nets = 'nets'
    uploads = 'uploads'
    detectimg = 'detectimg'
    thumbs = 'thumbs'
    trashdir = 'trashed_images'
    distinctdir = 'distinctiveness_model'
    scorenormdir = 'scorenorm'
    smartpatrol = 'smart_patrol'
    # Query Results (chipmatch dirs)
    qres = 'qres_new'
    bigcache = 'qres_bigcache_new'


class REL_PATHS(object):  # NOQA
    """ all paths are relative to ibs.dbdir """

    _ibsdb = PATH_NAMES._ibsdb
    trashdir = PATH_NAMES.trashdir
    figures = join(_ibsdb, PATH_NAMES.figures)
    cache = join(_ibsdb, PATH_NAMES.cache)
    backups = join(_ibsdb, PATH_NAMES.backups)
    logs = join(_ibsdb, PATH_NAMES.logs)
    # chips       = join(_ibsdb, PATH_NAMES.chips)
    images = join(_ibsdb, PATH_NAMES.images)
    trees = join(_ibsdb, PATH_NAMES.trees)
    nets = join(_ibsdb, PATH_NAMES.nets)
    uploads = join(_ibsdb, PATH_NAMES.uploads)
    # All computed dirs live in <dbdir>/_ibsdb/_ibeis_cache
    chips = join(cache, PATH_NAMES.chips)
    thumbs = join(cache, PATH_NAMES.thumbs)
    flann = join(cache, PATH_NAMES.flann)
    qres = join(cache, PATH_NAMES.qres)
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
    # '_ibsdb/_ibeis_cache',
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
SPECIES_KEY = 'SPECIES_KEY'
EMPTY_KEY = ''
UNKNOWN = '____'
KEY_DEFAULTS = {
    INDIVIDUAL_KEY: UNKNOWN,
    SPECIES_KEY: UNKNOWN,
}

# <UNFINISHED METADATA>
# We are letting wildbook do this metadata instead
# Define the special metadata for annotation

ROSEMARY_ANNOT_METADATA = [
    ('local_name', 'Local name:', str),
    ('sun', 'Sun:', ['FS', 'PS', 'NS']),
    ('wind', 'Wind:', ['NW', 'LW', 'MW', 'SW']),
    ('rain', 'Rain:', ['NR', 'LR', 'MR', 'HR']),
    ('cover', 'Cover:', float),
    ('grass', 'Grass:', ['less hf', 'less hk', 'less belly']),
    ('grass_color', 'Grass Colour:', ['B', 'BG', 'GB', 'G']),
    ('grass_species', 'Grass Species:', str),
    ('bush_type', 'Bush type:', ['OG', 'LB', 'MB', 'TB']),
    ('bit', 'Bit:', int),
    ('other_speceis', 'Other Species:', str),
]

# ROSEMARY_KEYS = utool.get_list_column(ROSEMARY_ANNOT_METADATA, 0)
# KEY_DEFAULTS.update(**{key: UNKNOWN for key in ROSEMARY_KEYS})
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
AL_RELATION_TABLE = 'annotation_lblannot_relationship'
GA_RELATION_TABLE = 'annotgroup_annotation_relationship'
ANNOTGROUP_TABLE = 'annotgroups'
ANNOTATION_TABLE = 'annotations'
PART_TABLE = 'parts'
CHIP_TABLE = 'chips'
CONFIG_TABLE = 'configs'
CONTRIBUTOR_TABLE = 'contributors'
GSG_RELATION_TABLE = 'imageset_image_relationship'
IMAGESET_TABLE = 'imagesets'
FEATURE_TABLE = 'features'
FEATURE_WEIGHT_TABLE = 'feature_weights'
GL_RELATION_TABLE = 'image_lblimage_relationship'
IMAGE_TABLE = 'images'
LBLANNOT_TABLE = 'lblannot'
LBLIMAGE_TABLE = 'lblimage'
LBLTYPE_TABLE = 'keys'
METADATA_TABLE = 'metadata'
# Ugly move from name to names, need better way of versioning old table names
NAME_TABLE_v121 = 'name'
NAME_TABLE_v130 = 'names'
NAME_TABLE = NAME_TABLE_v130
ANNOTMATCH_TABLE = 'annotmatch'
SPECIES_TABLE = 'species'
RESIDUAL_TABLE = 'residuals'
VERSIONS_TABLE = 'versions'
#
PARTY_CONTRIB_RELATION_TABLE = 'party_contrib_relation'
PARTY_TABLE = 'party'
#################################################################
# STAGING DATABASE
REVIEW_TABLE = 'reviews'
TEST_TABLE = 'tests'
#################################################################


# DEPCACHE TABLENAMES
# CHIPTHUMB_TABLE = 'chipthumb'


UNKNOWN_PURPLE_RGBA255 = np.array((102, 0, 153, 255))
NAME_BLUE_RGBA255 = np.array((20, 20, 235, 255))
NAME_RED_RGBA255 = np.array((235, 20, 20, 255))
NEW_YELLOW_RGBA255 = np.array((235, 235, 20, 255))

UNKNOWN_PURPLE_RGBA01 = UNKNOWN_PURPLE_RGBA255 / 255.0
NAME_BLUE_RGBA01 = NAME_BLUE_RGBA255 / 255.0
NAME_RED_RGBA01 = NAME_RED_RGBA255 / 255.0
NEW_YELLOW_RGBA01 = NEW_YELLOW_RGBA255 / 255.0

EXEMPLAR_IMAGESETTEXT = '*Exemplars'
ALL_IMAGE_IMAGESETTEXT = '*All Images'
UNREVIEWED_IMAGE_IMAGESETTEXT = '*Undetected Images'
REVIEWED_IMAGE_IMAGESETTEXT = '*Reviewed Detections'
UNGROUPED_IMAGES_IMAGESETTEXT = '*Ungrouped Images'
SPECIAL_IMAGESET_LABELS = [
    EXEMPLAR_IMAGESETTEXT,
    ALL_IMAGE_IMAGESETTEXT,
    UNREVIEWED_IMAGE_IMAGESETTEXT,
    REVIEWED_IMAGE_IMAGESETTEXT,
    UNGROUPED_IMAGES_IMAGESETTEXT,
]
NEW_IMAGESET_IMAGESETTEXT = 'NEW IMAGESET'

# IMAGE_THUMB_SUFFIX = '_thumb.png'
# CHIP_THUMB_SUFFIX  = '_chip_thumb.png'
IMAGE_THUMB_SUFFIX = '_thumb.jpg'
IMAGE_BARE_THUMB_SUFFIX = '_thumb_bare.jpg'
CHIP_THUMB_SUFFIX = '_chip_thumb.jpg'


VS_EXEMPLARS_KEY = 'vs_exemplars'
INTRA_OCCUR_KEY = 'intra_occurrence'

HARD_NOTE_TAG = '<HARDCASE>'

# HACK
COMPUTER_NAME = ut.get_computer_name()
if COMPUTER_NAME in ['wbia.cs.uic.edu']:
    _DEFAULT_WILDBOOK_TARGET = 'lewa3'
elif COMPUTER_NAME in ['Leviathan']:
    _DEFAULT_WILDBOOK_TARGET = 'lewa3'
elif COMPUTER_NAME in ['maasai', 'quagga', 'xadmin-Nitro-AN515-51']:
    _DEFAULT_WILDBOOK_TARGET = 'quagga.princeton.edu'
else:
    _DEFAULT_WILDBOOK_TARGET = 'wbia'
WILDBOOK_TARGET = ut.get_argval(
    '--wildbook-target',
    type_=str,
    default=_DEFAULT_WILDBOOK_TARGET,
    help_='specify the Wildbook target deployment',
)


class ZIPPED_URLS(object):  # NOQA
    PZ_DISTINCTIVE = 'https://wildbookiarepository.azureedge.net/models/distinctivness_zebra_plains.zip'  # DEPRICATE
    GZ_DISTINCTIVE = 'https://wildbookiarepository.azureedge.net/models/distinctivness_zebra_grevys.zip'  # DEPRICATE

    PZ_MTEST = 'https://wildbookiarepository.azureedge.net/databases/PZ_MTEST.zip'
    NAUTS = 'https://wildbookiarepository.azureedge.net/databases/NAUT_test.zip'
    WDS = 'https://wildbookiarepository.azureedge.net/databases/wd_peter2.zip'
    DF_CURVRANK = (
        'https://wildbookiarepository.azureedge.net/databases/testdb_curvrank.zip'
    )
    ID_EXAMPLE = (
        'https://wildbookiarepository.azureedge.net/databases/testdb_identification.zip'
    )
    ORIENTATION = (
        'https://wildbookiarepository.azureedge.net/databases/testdb_orientation.zip'
    )
    K7_EXAMPLE = 'https://wildbookiarepository.azureedge.net/databases/testdb_kaggle7.zip'


# Turn off features at Lewa :(
SIMPLIFY_INTERFACE = (ut.get_computer_name() == 'wbia.cs.uic.edu') or ut.get_argflag(
    '--simplify'
)


# For candidacy document
DBNAME_ALIAS = {
    # 'NNP_MasterGIRM_core': 'NNP_GIRM'
    # 'NNP_MasterGIRM_core': 'GIRM',
    'NNP_MasterGIRM_core': 'GIRM',
    'PZ_Master1': 'PZ',
    'GZ_Master1': 'GZ',
    'GIRM_Master1': 'GIRM',
    'GZ_ALL': 'GZ',
}


class TEST_SPECIES(object):  # NOQA
    ZEB_PLAIN = 'zebra_plains'
    ZEB_GREVY = 'zebra_grevys'
    BEAR_POLAR = 'bear_polar'
    GIR_MASAI = 'giraffe_masai'


SPECIES_WITH_DETECTORS = (
    'cheetah',
    'dolphin_spotted',
    'dolphin_spotted+dorsal',
    'dolphin_spotted+fin_dorsal',
    'giraffe_masai',
    'giraffe_reticulated',
    'jaguar',
    'lynx',
    'manta_ray',
    'manta_ray_giant',
    'right_whale_head',
    'seadragon_leafy',
    'seadragon_leafy+head',
    'seadragon_weedy',
    'seadragon_weedy+head',
    'skunk_spotted',
    'turtle_green',
    'turtle_green+head',
    'turtle_hawksbill',
    'turtle_hawksbill+head',
    'turtle_oliveridley',
    'turtle_oliveridley+head',
    'turtle_sea',
    'turtle_sea+head',
    'whale_fluke',
    'whale_humpback',
    'whale_humpback+fin_dorsal',
    'zebra_grevys',
    'zebra_plains',
    'zebra_mountain',
    'whale_orca',
    'whale_orca+fin_dorsal',
    'leopard',
    'wild_dog',
    'wild_dog_dark',
    'wild_dog_light',
    'wild_dog_puppy',
    'wild_dog_standard',
    'wild_dog_tan',
    # Latin (Flukebook)
    'chelonioidea',
    'chelonioidea+head',
    'chelonia_mydas',
    'chelonia_mydas+head',
    'eretmochelys_imbricata',
    'eretmochelys_imbricata+head',
    'lepidochelys_olivacea',
    'lepidochelys_olivacea+head',
    'eubalaena_australis',
    'eubalaena_glacialis',
    'equus_quagga',
    'equus_grevyi',
    'equus_zebra',
    'giraffa_tippelskirchi',
    'giraffa_camelopardalis_reticulata',
    'manta_birostris',
    'mobula_alfredi',
    'lynx_pardinus',
    'phycodurus_eques',
    'phycodurus_eques+head',
    'phyllopteryx_taeniolatus',
    'phyllopteryx_taeniolatus+head',
    'megaptera_novaeangliae',
    'physeter_macrocephalus',
    'physeter_macrocephalus+fin_dorsal',
    'spilogale_gracilis',
    'stenella_frontalis',
    'stenella_frontalis+dorsal',
    'stenella_frontalis+fin_dorsal',
    'orcinus_orca',
    'orcinus_orca+fin_dorsal',
    'panthera_pardus',
    'panthera_onca',
    'acinonyx_jubatus',
    'lycaon_pictus',
)


SPECIES_MAPPING = {
    'antelope': ('ANTEL', 'Antelope'),
    'airplane': ('PLANE', 'Airplane'),
    'bear_polar': ('PB', 'Polar Bear'),
    'bird': ('BIRD', 'Bird (Generic)'),
    'bird_generic': (None, 'bird'),
    'bicycle': ('BIKE', 'Bicycle'),
    'boat': ('BOAT', 'Boat'),
    'building': ('BUILDING', 'Building'),
    'bus': ('BUS', 'Bus'),
    'car': ('CAR', 'Car'),
    'camel': ('CAMEL', 'Camel'),
    'crane': ('CRANE', 'Crane'),
    'cheetah': ('CHTH', 'Cheetah'),
    'dog_wild': ('WD', 'Wild Dog'),
    'dolphin_fin': ('DF', 'Dolphin Fin'),
    'dolphin_bottlenose_fin': ('BDF', 'Bottlenose Dolphin Fin'),
    'dolphin_spotted': ('SDOLPH', 'Dolphin (Spotted)'),
    'domesticated_cat': ('DOMC', 'Cat (Domesticated)'),
    'domesticated_cow': ('DOMW', 'Cow (Domesticated)'),
    'domesticated_dog': ('DOMD', 'Dog (Domesticated)'),
    'domesticated_horse': ('DOMH', 'Horse (Domesticated)'),
    'domesticated_sheep': ('DOMS', 'Sheep (Domesticated)'),
    'cat_domestic': (None, 'domesticated_cat'),
    'cow_domestic': (None, 'domesticated_cow'),
    'dog_domestic': (None, 'domesticated_dog'),
    'horse_domestic': (None, 'domesticated_horse'),
    'sheep_domestic': (None, 'domesticated_sheep'),
    'elephant_savanna': ('ELEP', 'Elephant (Savanna)'),
    'elephant_savannah': (None, 'elephant_savanna'),
    'frog': ('FROG', 'Frog'),
    'gazelle': ('GAZ', 'Gazelle (Generic)'),
    'gazelle_generic': (None, 'gazelle'),
    'giraffe_masai': ('GIRM', 'Giraffe (Masai)'),
    'giraffe_massai': (None, 'giraffe_masai'),
    'giraffe_maasai': (None, 'giraffe_masai'),
    'giraffe_reticulated': ('GIR', 'Giraffe (Reticulated)'),
    'goat': ('GOAT', 'Goat (Generic)'),
    'goat_generic': (None, 'goat'),
    'grouper_nassau': ('NG', 'Nassau Grouper'),
    'hippopotamus': ('HIPPO', 'Hippopotamus'),
    'hyena': ('HYENA', 'Hyena'),
    'ignore': ('IGNORE', 'IGNORE'),
    'indistinct': ('INDISTINCT', 'INDISTINCT'),
    'jaguar': ('JAG', 'Jaguar'),
    'leopard': ('LOEP', 'Leopard'),
    'lion': ('LION', 'Lion'),
    'lionfish': ('LF', 'Lionfish'),
    'lynx': ('LYNX', 'Lynx'),
    'manta_ray_giant': ('MR', 'Manta Ray (Giant)'),
    'manta_ray': (None, 'manta_ray_giant'),
    'morotcycle': ('MBIKE', 'Motorcycle'),
    'nautilus': ('NAUT', 'Nautilus'),
    'ostrich': ('OST', 'Ostrich'),
    'other': ('OTHER', 'Other'),
    'person': ('PERSON', 'Person'),
    'rhino_black': ('BRHINO', 'Rhino (Black)'),
    'rhino_white': ('WRHINO', 'Rhino (White)'),
    'seal_saimma_ringed': ('SEAL2', 'Seal (Siamaa Ringed)'),
    'seal_spotted': ('SEAL1', 'Seal (Spotted)'),
    'seadragon_leafy': ('SDL', 'Sea Dragon (Leafy)'),
    'seadragon_weedy': ('SDW', 'Sea Dragon (Weedy)'),
    'shark_sandtiger': ('STS', 'Sand Tiger Shark'),
    'skunk_spotted': ('SSKUNK', 'Skunk (Spotted)'),
    'snail': ('SNAIL', 'Snail'),
    'snow_leopard': ('SLEOP', 'Snow Leopard'),
    'tiger': ('TIGER', 'Tiger'),
    'turtle_sea': ('ST', 'Sea Turtle (Generic)'),
    'turtle_sea_generic': (None, 'turtle_sea'),
    'turtle_hawksbill': ('STHB', 'Sea Turtle (Hawksbill)'),
    'turtle_green': ('STG', 'Sea Turtle (Green)'),
    'train': ('TRAIN', 'Train'),
    'truck': ('TRUCK', 'Truck'),
    'toads_wyoming': ('WYTOAD', 'Toad (Wyoming)'),
    'unspecified': ('UNSPECIFIED', 'UNSPECIFIED'),
    'unspecified_animal': (None, 'unspecified'),
    'warthog': ('WART', 'Warthog'),
    'water_buffalo': ('BUFF', 'Water Buffalo'),
    'wildebeest': ('WB', 'Wildebeest'),
    'wild_dog_general': ('WDG', 'Wild Dog General'),
    'wild_dog_dark': ('WDD', 'Wild Dog Dark'),
    'wild_dog_tan': ('WDT', 'Wild Dog Tan'),
    'wild_dog_puppy': ('WDP', 'Wild Dog Puppy'),
    'wild_dog_standard': ('WDS', 'Wild Dog Standard'),
    'wild_dog+tail_general': ('WD+GEN', 'Wild Dog Tail General'),
    'wild_dog+tail_multi_black': ('WD+MB', 'Wild Dog Tail Multi-Black'),
    'wild_dog+tail_long_black': ('WD+LB', 'Wild Dog Tail Long Black'),
    'wild_dog+tail_long_white': ('WD+LW', 'Wild Dog Tail Long White'),
    'wild_dog+tail_long': ('WD+LONG', 'Wild Dog Tail Long'),
    'wild_dog+tail_double_black_brown': ('WD+DBB', 'Wild Dog Tail Double Black Brown'),
    'wild_dog+tail_double_black_white': ('WD+DBW', 'Wild Dog Tail Double Black White'),
    'wild_dog+tail_triple_black': ('WD+TB', 'Wild Dog Tail Triple Black'),
    'wild_dog+tail_short_black': ('WD+SB', 'Wild Dog Tail Short Black'),
    'wild_dog+tail_standard': ('WD+STD', 'Wild Dog Tail Standard'),
    'wild_dog': ('WD', 'Wild Dog'),
    'whale_fluke': ('WF', 'Whale Fluke'),
    'whale_humpback': ('HW', 'Humpback Whale'),
    'whale_shark': ('WS', 'Whale Shark'),
    'zebra_grevys': ('GZ', "Zebra (Grevy's)"),
    'zebra_hybrid': ('HZ', 'Zebra (Hybrid)'),
    'zebra_plains': ('PZ', 'Zebra (Plains)'),
    'zebra_mountain': ('MZ', 'Zebra (Mountain)'),
    UNKNOWN: ('UNKNOWN', 'UNSPECIFIED'),
}

PARTS_MAPPING = {
    'head': 'Head',
    'standard': 'Standard Tail',
    'short_black': 'Short Black Tail',
    'long_black': 'Long Black Tail',
    'double_black_brown': 'Double Black Brown Tail',
    'double_black_white': 'Double Black White Tail',
    'triple_black': 'Triple Black Tail',
    'long_white': 'Long White Tail',
    UNKNOWN: 'UNSPECIFIED,',
}


class _ConstHelper(type):
    """
    Adds code and nice constants to an integer version of a class
    """

    def __new__(cls, name, parents, dct):
        """
        cls = META_DECISION
        code_cls = META_DECISION_CODE
        """

        class CODE(object):
            pass

        class NICE(object):
            pass

        for key in dct.keys():
            if key.isupper():
                value = dct[key]
                if value is None or isinstance(value, int):
                    code = dct['INT_TO_CODE'][value]
                    nice = dct['INT_TO_NICE'][value]
                    setattr(CODE, key, code)
                    setattr(NICE, key, nice)

        dct['CODE'] = CODE
        dct['NICE'] = NICE
        # we need to call type.__new__ to complete the initialization
        return super(_ConstHelper, cls).__new__(cls, name, parents, dct)


@six.add_metaclass(_ConstHelper)
class EVIDENCE_DECISION(object):  # NOQA
    """
    TODO: change to EVIDENCE_DECISION / VISUAL_DECISION
    Enumerated types of review codes and texts

    Notes:
        Unreviewed: Not comparared yet.
        nomatch: Visually comparable and the different
        match: Visually comparable and the same
        notcomp: Not comparable means it is actually impossible to determine.
        unknown: means that it was reviewed, but we just can't figure it out.
    """

    UNREVIEWED = None
    NEGATIVE = 0
    POSITIVE = 1
    INCOMPARABLE = 2
    UNKNOWN = 3

    INT_TO_CODE = ut.odict(
        [
            (POSITIVE, 'match'),
            (NEGATIVE, 'nomatch'),
            (INCOMPARABLE, 'notcomp'),
            (UNKNOWN, 'unknown'),
            (UNREVIEWED, 'unreviewed'),
        ]
    )

    INT_TO_NICE = ut.odict(
        [
            (POSITIVE, 'Positive'),
            (NEGATIVE, 'Negative'),
            (INCOMPARABLE, 'Incomparable'),
            (UNKNOWN, 'Unknown'),
            (UNREVIEWED, 'Unreviewed'),
        ]
    )

    CODE_TO_NICE = ut.map_keys(INT_TO_CODE, INT_TO_NICE)
    CODE_TO_INT = ut.invert_dict(INT_TO_CODE)
    NICE_TO_CODE = ut.invert_dict(CODE_TO_NICE)
    NICE_TO_INT = ut.invert_dict(INT_TO_NICE)

    MATCH_CODE = CODE_TO_INT


@six.add_metaclass(_ConstHelper)
class META_DECISION(object):  # NOQA
    """
    Enumerated types of review codes and texts

    Notes:
        unreviewed: we dont have a meta decision
        same: we know this is the same animal through non-visual means
        diff: we know this is the different animal through non-visual means

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.constants import *  # NOQA
        >>> assert hasattr(META_DECISION, 'CODE')
        >>> assert hasattr(META_DECISION, 'NICE')
        >>> code1 = META_DECISION.INT_TO_CODE[META_DECISION.NULL]
        >>> code2 = META_DECISION.CODE.NULL
        >>> assert code1 == code2
        >>> nice1 = META_DECISION.INT_TO_NICE[META_DECISION.NULL]
        >>> nice2 = META_DECISION.NICE.NULL
        >>> assert nice1 == nice2
    """

    NULL = None
    DIFF = 0
    SAME = 1
    INT_TO_CODE = ut.odict([(NULL, 'null'), (DIFF, 'diff'), (SAME, 'same')])
    INT_TO_NICE = ut.odict([(NULL, 'NULL'), (DIFF, 'Different'), (SAME, 'Same')])
    CODE_TO_NICE = ut.map_keys(INT_TO_CODE, INT_TO_NICE)
    CODE_TO_INT = ut.invert_dict(INT_TO_CODE)
    NICE_TO_CODE = ut.invert_dict(CODE_TO_NICE)
    NICE_TO_INT = ut.invert_dict(INT_TO_NICE)


@six.add_metaclass(_ConstHelper)
class CONFIDENCE(object):
    UNKNOWN = None
    GUESSING = 1
    NOT_SURE = 2
    PRETTY_SURE = 3
    ABSOLUTELY_SURE = 4

    INT_TO_CODE = ut.odict(
        [
            (ABSOLUTELY_SURE, 'absolutely_sure'),
            (PRETTY_SURE, 'pretty_sure'),
            (NOT_SURE, 'not_sure'),
            (GUESSING, 'guessing'),
            (UNKNOWN, 'unspecified'),
        ]
    )

    INT_TO_NICE = ut.odict(
        [
            (ABSOLUTELY_SURE, 'Doubtless'),
            (PRETTY_SURE, 'Sure'),
            (NOT_SURE, 'Unsure'),
            (GUESSING, 'Guessing'),
            (UNKNOWN, 'Unspecified'),
        ]
    )

    CODE_TO_NICE = ut.map_keys(INT_TO_CODE, INT_TO_NICE)
    CODE_TO_INT = ut.invert_dict(INT_TO_CODE)
    NICE_TO_CODE = ut.invert_dict(CODE_TO_NICE)
    NICE_TO_INT = ut.invert_dict(INT_TO_NICE)


@six.add_metaclass(_ConstHelper)
class QUAL(object):
    EXCELLENT = 5
    GOOD = 4
    OK = 3
    POOR = 2
    JUNK = 1
    UNKNOWN = None

    INT_TO_CODE = ut.odict(
        [
            (EXCELLENT, 'excellent'),
            (GOOD, 'good'),
            (OK, 'ok'),
            (POOR, 'poor'),
            (JUNK, 'junk'),
            (UNKNOWN, 'unspecified'),
        ]
    )

    INT_TO_NICE = ut.odict(
        [
            (EXCELLENT, 'Excellent'),
            (GOOD, 'Good'),
            (OK, 'OK'),
            (POOR, 'Poor'),
            (JUNK, 'Junk'),
            (UNKNOWN, 'Unspecified'),
        ]
    )

    CODE_TO_NICE = ut.map_keys(INT_TO_CODE, INT_TO_NICE)
    CODE_TO_INT = ut.invert_dict(INT_TO_CODE)
    NICE_TO_CODE = ut.invert_dict(CODE_TO_NICE)
    NICE_TO_INT = ut.invert_dict(INT_TO_NICE)


@six.add_metaclass(_ConstHelper)
class VIEW(object):
    """
    categorical viewpoint using the faces of a Rhombicuboctahedron

    References:
        https://en.wikipedia.org/wiki/Rhombicuboctahedron
    """

    UNKNOWN = None
    R = 1
    FR = 2
    F = 3
    FL = 4
    L = 5
    BL = 6
    B = 7
    BR = 8

    U = 9
    UF = 10
    UB = 11
    UL = 12
    UR = 13
    UFL = 14
    UFR = 15
    UBL = 16
    UBR = 17

    D = 18
    DF = 19
    DB = 20
    DL = 21
    DR = 22
    DFL = 23
    DFR = 24
    DBL = 25
    DBR = 26

    INT_TO_CODE = ut.odict(
        [
            (UNKNOWN, 'unknown'),
            (R, 'right'),
            (FR, 'frontright'),
            (F, 'front'),
            (FL, 'frontleft'),
            (L, 'left'),
            (BL, 'backleft'),
            (B, 'back'),
            (BR, 'backright'),
            (U, 'up'),
            (UF, 'upfront'),
            (UB, 'upback'),
            (UL, 'upleft'),
            (UR, 'upright'),
            (UFL, 'upfrontleft'),
            (UFR, 'upfrontright'),
            (UBL, 'upbackleft'),
            (UBR, 'upbackright'),
            (D, 'down'),
            (DF, 'downfront'),
            (DB, 'downback'),
            (DL, 'downleft'),
            (DR, 'downright'),
            (DFL, 'downfrontleft'),
            (DFR, 'downfrontright'),
            (DBL, 'downbackleft'),
            (DBR, 'downbackright'),
        ]
    )

    INT_TO_NICE = ut.odict(
        [
            (UNKNOWN, 'Unknown'),
            (R, 'Right'),
            (FR, 'Front-Right'),
            (F, 'Front'),
            (FL, 'Front-Left'),
            (L, 'Left'),
            (BL, 'Back-Left'),
            (B, 'Back'),
            (BR, 'Back-Right'),
            (U, 'Up'),
            (UF, 'Up-Front'),
            (UB, 'Up-Back'),
            (UL, 'Up-Left'),
            (UR, 'Up-Right'),
            (UFL, 'Up-Front-Left'),
            (UFR, 'Up-Front-Right'),
            (UBL, 'Up-Back-Left'),
            (UBR, 'Up-Back-Right'),
            (D, 'Down'),
            (DF, 'Down-Front'),
            (DB, 'Down-Back'),
            (DL, 'Down-Left'),
            (DR, 'Down-Right'),
            (DFL, 'Down-Front-Left'),
            (DFR, 'Down-Front-Right'),
            (DBL, 'Down-Back-Left'),
            (DBR, 'Down-Back-Right'),
        ]
    )

    CODE_TO_NICE = ut.map_keys(INT_TO_CODE, INT_TO_NICE)
    CODE_TO_INT = ut.invert_dict(INT_TO_CODE)
    NICE_TO_CODE = ut.invert_dict(CODE_TO_NICE)
    NICE_TO_INT = ut.invert_dict(INT_TO_NICE)

    DIST = {
        # DIST 0 PAIRS
        (B, B): 0,
        (BL, BL): 0,
        (BR, BR): 0,
        (D, D): 0,
        (DB, DB): 0,
        (DBL, DBL): 0,
        (DBR, DBR): 0,
        (DF, DF): 0,
        (DFL, DFL): 0,
        (DFR, DFR): 0,
        (DL, DL): 0,
        (DR, DR): 0,
        (F, F): 0,
        (FL, FL): 0,
        (FR, FR): 0,
        (L, L): 0,
        (R, R): 0,
        (U, U): 0,
        (UB, UB): 0,
        (UBL, UBL): 0,
        (UBR, UBR): 0,
        (UF, UF): 0,
        (UFL, UFL): 0,
        (UFR, UFR): 0,
        (UL, UL): 0,
        (UR, UR): 0,
        # DIST 1 PAIRS
        (B, BL): 1,
        (B, BR): 1,
        (B, DB): 1,
        (B, DBL): 1,
        (B, DBR): 1,
        (B, UB): 1,
        (B, UBL): 1,
        (B, UBR): 1,
        (BL, DBL): 1,
        (BL, L): 1,
        (BL, UBL): 1,
        (BR, DBR): 1,
        (BR, R): 1,
        (BR, UBR): 1,
        (D, DB): 1,
        (D, DBL): 1,
        (D, DBR): 1,
        (D, DF): 1,
        (D, DFL): 1,
        (D, DFR): 1,
        (D, DL): 1,
        (D, DR): 1,
        (DB, DBL): 1,
        (DB, DBR): 1,
        (DBL, DL): 1,
        (DBL, L): 1,
        (DBR, DR): 1,
        (DBR, R): 1,
        (DF, DFL): 1,
        (DF, DFR): 1,
        (DF, F): 1,
        (DFL, DL): 1,
        (DFL, F): 1,
        (DFL, FL): 1,
        (DFL, L): 1,
        (DFR, DR): 1,
        (DFR, F): 1,
        (DFR, FR): 1,
        (DFR, R): 1,
        (DL, L): 1,
        (DR, R): 1,
        (F, FL): 1,
        (F, FR): 1,
        (F, UF): 1,
        (F, UFL): 1,
        (F, UFR): 1,
        (FL, L): 1,
        (FL, UFL): 1,
        (FR, R): 1,
        (FR, UFR): 1,
        (L, UBL): 1,
        (L, UFL): 1,
        (L, UL): 1,
        (R, UBR): 1,
        (R, UFR): 1,
        (R, UR): 1,
        (U, UB): 1,
        (U, UBL): 1,
        (U, UBR): 1,
        (U, UF): 1,
        (U, UFL): 1,
        (U, UFR): 1,
        (U, UL): 1,
        (U, UR): 1,
        (UB, UBL): 1,
        (UB, UBR): 1,
        (UBL, UL): 1,
        (UBR, UR): 1,
        (UF, UFL): 1,
        (UF, UFR): 1,
        (UFL, UL): 1,
        (UFR, UR): 1,
        # DIST 2 PAIRS
        (B, D): 2,
        (B, DL): 2,
        (B, DR): 2,
        (B, L): 2,
        (B, R): 2,
        (B, U): 2,
        (B, UL): 2,
        (B, UR): 2,
        (BL, BR): 2,
        (BL, D): 2,
        (BL, DB): 2,
        (BL, DBR): 2,
        (BL, DFL): 2,
        (BL, DL): 2,
        (BL, FL): 2,
        (BL, U): 2,
        (BL, UB): 2,
        (BL, UBR): 2,
        (BL, UFL): 2,
        (BL, UL): 2,
        (BR, D): 2,
        (BR, DB): 2,
        (BR, DBL): 2,
        (BR, DFR): 2,
        (BR, DR): 2,
        (BR, FR): 2,
        (BR, U): 2,
        (BR, UB): 2,
        (BR, UBL): 2,
        (BR, UFR): 2,
        (BR, UR): 2,
        (D, F): 2,
        (D, FL): 2,
        (D, FR): 2,
        (D, L): 2,
        (D, R): 2,
        (DB, DF): 2,
        (DB, DFL): 2,
        (DB, DFR): 2,
        (DB, DL): 2,
        (DB, DR): 2,
        (DB, L): 2,
        (DB, R): 2,
        (DB, UB): 2,
        (DB, UBL): 2,
        (DB, UBR): 2,
        (DBL, DBR): 2,
        (DBL, DF): 2,
        (DBL, DFL): 2,
        (DBL, DFR): 2,
        (DBL, DR): 2,
        (DBL, FL): 2,
        (DBL, UB): 2,
        (DBL, UBL): 2,
        (DBL, UBR): 2,
        (DBL, UFL): 2,
        (DBL, UL): 2,
        (DBR, DF): 2,
        (DBR, DFL): 2,
        (DBR, DFR): 2,
        (DBR, DL): 2,
        (DBR, FR): 2,
        (DBR, UB): 2,
        (DBR, UBL): 2,
        (DBR, UBR): 2,
        (DBR, UFR): 2,
        (DBR, UR): 2,
        (DF, DL): 2,
        (DF, DR): 2,
        (DF, FL): 2,
        (DF, FR): 2,
        (DF, L): 2,
        (DF, R): 2,
        (DF, UF): 2,
        (DF, UFL): 2,
        (DF, UFR): 2,
        (DFL, DFR): 2,
        (DFL, DR): 2,
        (DFL, FR): 2,
        (DFL, UBL): 2,
        (DFL, UF): 2,
        (DFL, UFL): 2,
        (DFL, UFR): 2,
        (DFL, UL): 2,
        (DFR, DL): 2,
        (DFR, FL): 2,
        (DFR, UBR): 2,
        (DFR, UF): 2,
        (DFR, UFL): 2,
        (DFR, UFR): 2,
        (DFR, UR): 2,
        (DL, DR): 2,
        (DL, F): 2,
        (DL, FL): 2,
        (DL, UBL): 2,
        (DL, UFL): 2,
        (DL, UL): 2,
        (DR, F): 2,
        (DR, FR): 2,
        (DR, UBR): 2,
        (DR, UFR): 2,
        (DR, UR): 2,
        (F, L): 2,
        (F, R): 2,
        (F, U): 2,
        (F, UL): 2,
        (F, UR): 2,
        (FL, FR): 2,
        (FL, U): 2,
        (FL, UBL): 2,
        (FL, UF): 2,
        (FL, UFR): 2,
        (FL, UL): 2,
        (FR, U): 2,
        (FR, UBR): 2,
        (FR, UF): 2,
        (FR, UFL): 2,
        (FR, UR): 2,
        (L, U): 2,
        (L, UB): 2,
        (L, UF): 2,
        (R, U): 2,
        (R, UB): 2,
        (R, UF): 2,
        (UB, UF): 2,
        (UB, UFL): 2,
        (UB, UFR): 2,
        (UB, UL): 2,
        (UB, UR): 2,
        (UBL, UBR): 2,
        (UBL, UF): 2,
        (UBL, UFL): 2,
        (UBL, UFR): 2,
        (UBL, UR): 2,
        (UBR, UF): 2,
        (UBR, UFL): 2,
        (UBR, UFR): 2,
        (UBR, UL): 2,
        (UF, UL): 2,
        (UF, UR): 2,
        (UFL, UFR): 2,
        (UFL, UR): 2,
        (UFR, UL): 2,
        (UL, UR): 2,
        # DIST 3 PAIRS
        (B, DF): 3,
        (B, DFL): 3,
        (B, DFR): 3,
        (B, FL): 3,
        (B, FR): 3,
        (B, UF): 3,
        (B, UFL): 3,
        (B, UFR): 3,
        (BL, DF): 3,
        (BL, DFR): 3,
        (BL, DR): 3,
        (BL, F): 3,
        (BL, R): 3,
        (BL, UF): 3,
        (BL, UFR): 3,
        (BL, UR): 3,
        (BR, DF): 3,
        (BR, DFL): 3,
        (BR, DL): 3,
        (BR, F): 3,
        (BR, L): 3,
        (BR, UF): 3,
        (BR, UFL): 3,
        (BR, UL): 3,
        (D, UB): 3,
        (D, UBL): 3,
        (D, UBR): 3,
        (D, UF): 3,
        (D, UFL): 3,
        (D, UFR): 3,
        (D, UL): 3,
        (D, UR): 3,
        (DB, F): 3,
        (DB, FL): 3,
        (DB, FR): 3,
        (DB, U): 3,
        (DB, UFL): 3,
        (DB, UFR): 3,
        (DB, UL): 3,
        (DB, UR): 3,
        (DBL, F): 3,
        (DBL, FR): 3,
        (DBL, R): 3,
        (DBL, U): 3,
        (DBL, UF): 3,
        (DBL, UR): 3,
        (DBR, F): 3,
        (DBR, FL): 3,
        (DBR, L): 3,
        (DBR, U): 3,
        (DBR, UF): 3,
        (DBR, UL): 3,
        (DF, U): 3,
        (DF, UBL): 3,
        (DF, UBR): 3,
        (DF, UL): 3,
        (DF, UR): 3,
        (DFL, R): 3,
        (DFL, U): 3,
        (DFL, UB): 3,
        (DFL, UR): 3,
        (DFR, L): 3,
        (DFR, U): 3,
        (DFR, UB): 3,
        (DFR, UL): 3,
        (DL, FR): 3,
        (DL, R): 3,
        (DL, U): 3,
        (DL, UB): 3,
        (DL, UBR): 3,
        (DL, UF): 3,
        (DL, UFR): 3,
        (DR, FL): 3,
        (DR, L): 3,
        (DR, U): 3,
        (DR, UB): 3,
        (DR, UBL): 3,
        (DR, UF): 3,
        (DR, UFL): 3,
        (F, UB): 3,
        (F, UBL): 3,
        (F, UBR): 3,
        (FL, R): 3,
        (FL, UB): 3,
        (FL, UBR): 3,
        (FL, UR): 3,
        (FR, L): 3,
        (FR, UB): 3,
        (FR, UBL): 3,
        (FR, UL): 3,
        (L, UBR): 3,
        (L, UFR): 3,
        (L, UR): 3,
        (R, UBL): 3,
        (R, UFL): 3,
        (R, UL): 3,
        # DIST 4 PAIRS
        (B, F): 4,
        (BL, FR): 4,
        (BR, FL): 4,
        (D, U): 4,
        (DB, UF): 4,
        (DBL, UFR): 4,
        (DBR, UFL): 4,
        (DF, UB): 4,
        (DFL, UBR): 4,
        (DFR, UBL): 4,
        (DL, UR): 4,
        (DR, UL): 4,
        (L, R): 4,
        # UNDEFINED DIST PAIRS
        (B, UNKNOWN): None,
        (BL, UNKNOWN): None,
        (BR, UNKNOWN): None,
        (D, UNKNOWN): None,
        (DB, UNKNOWN): None,
        (DBL, UNKNOWN): None,
        (DBR, UNKNOWN): None,
        (DF, UNKNOWN): None,
        (DFL, UNKNOWN): None,
        (DFR, UNKNOWN): None,
        (DL, UNKNOWN): None,
        (DR, UNKNOWN): None,
        (F, UNKNOWN): None,
        (FL, UNKNOWN): None,
        (FR, UNKNOWN): None,
        (L, UNKNOWN): None,
        (R, UNKNOWN): None,
        (U, UNKNOWN): None,
        (UB, UNKNOWN): None,
        (UBL, UNKNOWN): None,
        (UBR, UNKNOWN): None,
        (UF, UNKNOWN): None,
        (UFL, UNKNOWN): None,
        (UFR, UNKNOWN): None,
        (UL, UNKNOWN): None,
        (UNKNOWN, B): None,
        (UNKNOWN, BL): None,
        (UNKNOWN, BR): None,
        (UNKNOWN, D): None,
        (UNKNOWN, DB): None,
        (UNKNOWN, DBL): None,
        (UNKNOWN, DBR): None,
        (UNKNOWN, DF): None,
        (UNKNOWN, DFL): None,
        (UNKNOWN, DFR): None,
        (UNKNOWN, DL): None,
        (UNKNOWN, DR): None,
        (UNKNOWN, F): None,
        (UNKNOWN, FL): None,
        (UNKNOWN, FR): None,
        (UNKNOWN, L): None,
        (UNKNOWN, R): None,
        (UNKNOWN, U): None,
        (UNKNOWN, UB): None,
        (UNKNOWN, UBL): None,
        (UNKNOWN, UBR): None,
        (UNKNOWN, UF): None,
        (UNKNOWN, UFL): None,
        (UNKNOWN, UFR): None,
        (UNKNOWN, UL): None,
        (UNKNOWN, UR): None,
        (UR, UNKNOWN): None,
        (UNKNOWN, UNKNOWN): None,
    }
    # make distance symmetric
    for (f1, f2), d in list(DIST.items()):
        DIST[(f2, f1)] = d


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.constants
        python -m wbia.constants --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
