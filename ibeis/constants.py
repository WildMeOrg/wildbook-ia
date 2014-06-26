class PATH_NAMES(object):
    """ Path names for internal IBEIS database """
    sqldb  = '_ibeis_database.sqlite3'
    _ibsdb = '_ibsdb'
    cache  = '_ibeis_cache'
    chips  = 'chips'
    flann  = 'flann'
    images = 'images'
    qres   = 'qres'
    bigcache = 'bigcache'
    detectimg = 'detectimg'
    thumbs = 'thumbs'

# Names normalized to the standard UNKNOWN_NAME
ACCEPTED_UNKNOWN_NAMES = set(['Unassigned'])

# Name used to denote that idkwtfthisis
UNKNOWN_NAME = '____'
ENCTEXT_PREFIX = 'enc_'

IMAGE_TABLE       = 'images'
ANNOTATION_TABLE       = 'annotations'
LABEL_TABLE       = 'labels'
ENCOUNTER_TABLE   = 'encounters'
EG_RELATION_TABLE = 'encounter_image_relationship'
AL_RELATION_TABLE = 'annotation_label_relationship'
CHIP_TABLE        = 'chips'
FEATURE_TABLE     = 'features'
CONFIG_TABLE      = 'configs'
KEY_TABLE         = 'keys'


import numpy as np
UNKNOWN_PURPLE_RGBA255 = np.array((102,   0, 153, 255))
NAME_BLUE_RGBA255 = np.array((20, 20, 235, 255))
NAME_RED_RGBA255 = np.array((235, 20, 20, 255))
NEW_YELLOW_RGBA255 = np.array((235, 235, 20, 255))

UNKNOWN_PURPLE_RGBA01 = UNKNOWN_PURPLE_RGBA255 / 255.0
NAME_BLUE_RGBA01      = NAME_BLUE_RGBA255 / 255.0
NAME_RED_RGBA01      = NAME_RED_RGBA255 / 255.0
NEW_YELLOW_RGBA01    = NEW_YELLOW_RGBA255 / 255.0

EXEMPLAR_ENCTEXT = 'Exemplars'
ALLIMAGE_ENCTEXT = 'All_Images'

IMAGE_THUMB_SUFFIX = '_thumb.png'
CHIP_THUMB_SUFFIX = '_chip_thumb.png'
