"""
Module Licence and docstring
"""
from __future__ import division, print_function

IMAGE_UID_TYPE = 'UUID'
ROI_UID_TYPE = 'UUID'
#NAME_UID_TYPE = 'UUID'
NAME_UID_TYPE = 'INTEGER'


def define_IBEIS_schema(ibs):
    # TODO, Add algoritm config column
    ibs.db.schema('images', (
        ('image_uid',                    'UUID PRIMARY KEY'),
        ('image_uri',                    'TEXT NOT NULL'),
        ('image_width',                  'INTEGER'),
        ('image_height',                 'INTEGER'),
        ('image_exif_time_posix',        'INTEGER'),
        ('image_exif_gps_lat',           'REAL'),
        ('image_exif_gps_lon',           'REAL'),
        ('image_confidence',             'REAL',),  # Move to algocfg table?
        ('image_toggle_enabled',         'INTEGER DEFAULT 0'),
        ('image_toggle_aif',             'INTEGER DEFAULT 0'),
    ))
    # Used to store the detected ROIs
    ibs.db.schema('rois', (
        ('roi_uid',                      'UUID PRIMARY KEY'),
        ('image_uid',                    'UUID NOT NULL'),
        ('name_uid',                     '%s NOT NULL' % NAME_UID_TYPE),
        ('roi_xtl',                      'INTEGER NOT NULL'),
        ('roi_ytl',                      'INTEGER NOT NULL'),
        ('roi_width',                    'INTEGER NOT NULL'),
        ('roi_height',                   'INTEGER NOT NULL'),
        ('roi_theta',                    'REAL DEFAULT 0.0'),
        ('roi_viewpoint',                'TEXT'),
    ))
    # Used to store *processed* ROIs as segmentations
    ibs.db.schema('masks', (
        ('mask_uid',                     'INTEGER PRIMARY KEY'),
        ('roi_uid',                      'UUID NOT NULL'),
        ('mask_uri',                     'TEXT NOT NULL'),
    ))
    # Used to store *processed* ROIs as chips
    ibs.db.schema('chips', (
        ('chip_uid',                     'INTEGER PRIMARY KEY'),
        ('roi_uid',                      'UUID NOT NULL'),
        ('chip_width',                   'INTEGER NOT NULL'),
        ('chip_height',                  'INTEGER NOT NULL'),
        ('chip_toggle_hard',             'INTEGER DEFAULT 0'),  # TODO: Remove?
    ), ['CONSTRAINT superkey UNIQUE (roi_uid)']
    )
    # Used to store individual chip features (ellipses)
    ibs.db.schema('features', (
        ('feature_uid',                  'INTEGER PRIMARY KEY'),
        ('chip_uid',                     'INTEGER NOT NULL'),
        ('feature_keypoints',            'NUMPY'),
        ('feature_sifts',                'NUMPY'),
    ), ['CONSTRAINT superkey UNIQUE (chip_uid)']
    )
    # Used to store individual chip identieis (Fred, Sue, ...)
    ibs.db.schema('names', (
        ('name_uid',                     '%s PRIMARY KEY' % NAME_UID_TYPE),
        ('name_text',                    'TEXT NOT NULL'),
    ), ['CONSTRAINT superkey UNIQUE (name_text)']
    )
    # Detection and identification algorithm configurations, populated
    # with caching information
    ibs.db.schema('configs', (
        ('config_uid',                   'INTEGER PRIMARY KEY'),
        ('config_suffix',                'TEXT NOT NULL'),
    ))
    # This table defines the pairing between an encounter and an
    # image. Hence, egpairs stands for encounter-image-pairs.  This table
    # exists for the sole purpose of defining multiple encounters to
    # a single image without the need to duplicate an image's record
    # in the images table.
    ibs.db.schema('encounters', (
        ('encounter_uid',               'INTEGER PRIMARY KEY'),
        ('image_uid',                   'UUID NOT NULL'),
        ('encounter_text',              'TEXT NOT NULL'),
    ))
