"""
Module Licence and docstring
"""
from __future__ import absolute_import, division, print_function

IMAGE_UID_TYPE = 'INTEGER'
ROI_UID_TYPE   = 'INTEGER'
NAME_UID_TYPE  = 'INTEGER'


def define_IBEIS_schema(ibs):
    # TODO, Add algoritm config column
    ibs.db.schema('images', (
        ('image_uid',                    '%s PRIMARY KEY' % IMAGE_UID_TYPE),
        ('image_uuid',                   'UUID NOT NULL'),
        ('image_uri',                    'TEXT NOT NULL'),
        ('image_ext',                    'TEXT NOT NULL'),
        ('image_original_name',          'TEXT NOT NULL'),  # We could parse this out
        #('image_original_path',          'TEXT NOT NULL'),
        ('image_width',                  'INTEGER'),
        ('image_height',                 'INTEGER'),
        ('image_exif_time_posix',        'INTEGER'),
        ('image_exif_gps_lat',           'REAL'),
        ('image_exif_gps_lon',           'REAL'),
        ('image_confidence',             'REAL DEFAULT -1.0',),  # Move to an algocfg table?
        ('image_toggle_enabled',         'INTEGER DEFAULT 0'),
        ('image_toggle_aif',             'INTEGER DEFAULT 0'),
        ('image_notes',                  'TEXT',),
    ), ['CONSTRAINT superkey UNIQUE (image_uuid)']
    )
    # Used to store individual chip identities (Fred, Sue, ...)
    ibs.db.schema('names', (
        ('name_uid',                     '%s PRIMARY KEY' % NAME_UID_TYPE),
        ('name_text',                    'TEXT NOT NULL'),
        ('name_notes',                   'TEXT',),
    ), ['CONSTRAINT superkey UNIQUE (name_text)']
    )
    # Used to store the detected ROIs / bboxed annotations
    ibs.db.schema('rois', (
        ('roi_uid',                      '%s PRIMARY KEY' % ROI_UID_TYPE),
        ('roi_uuid',                     'UUID NOT NULL'),
        ('image_uid',                    '%s NOT NULL' % IMAGE_UID_TYPE),
        ('name_uid',                     '%s NOT NULL' % NAME_UID_TYPE),
        ('roi_xtl',                      'INTEGER NOT NULL'),
        ('roi_ytl',                      'INTEGER NOT NULL'),
        ('roi_width',                    'INTEGER NOT NULL'),
        ('roi_height',                   'INTEGER NOT NULL'),
        ('roi_theta',                    'REAL DEFAULT 0.0'),
        ('roi_viewpoint',                'INTEGER DEFAULT 0'),
        ('roi_species_text',             'TEXT'),
        ('roi_detect_confidence',        'REAL DEFAULT -1.0'),
        ('roi_recognition_db_flag',      'INTEGER DEFAULT 0'),
        ('roi_notes',                    'TEXT'),
    ), ['CONSTRAINT superkey UNIQUE (roi_uuid)']
    )
    # Used to store *processed* ROIs as segmentations
    ibs.db.schema('masks', (
        ('mask_uid',                     'INTEGER PRIMARY KEY'),
        ('config_uid',                   'INTEGER DEFAULT 0'),
        ('roi_uid',                      '%s NOT NULL' % ROI_UID_TYPE),
        ('mask_uri',                     'TEXT NOT NULL'),
    ))
    # Used to store *processed* ROIs as chips
    ibs.db.schema('chips', (
        ('chip_uid',                     'INTEGER PRIMARY KEY'),
        ('roi_uid',                      '%s NOT NULL' % ROI_UID_TYPE),
        ('config_uid',                   'INTEGER DEFAULT 0'),
        ('chip_uri',                     'TEXT'),
        ('chip_width',                   'INTEGER NOT NULL'),
        ('chip_height',                  'INTEGER NOT NULL'),
    ), ['CONSTRAINT superkey UNIQUE (roi_uid, config_uid)']  # TODO: constraint needs modify
    )
    # Used to store individual chip features (ellipses)
    ibs.db.schema('features', (
        ('feature_uid',                  'INTEGER PRIMARY KEY'),
        ('chip_uid',                     'INTEGER NOT NULL'),
        ('config_uid',                   'INTEGER DEFAULT 0'),
        ('feature_num_feats',            'INTEGER NOT NULL'),
        ('feature_keypoints',            'NUMPY'),
        ('feature_sifts',                'NUMPY'),
    ), ['CONSTRAINT superkey UNIQUE (chip_uid, config_uid)']
    )
    # List of all encounters
    ibs.db.schema('encounters', (
        ('encounter_uid',               'INTEGER PRIMARY KEY'),
        ('encounter_text',              'TEXT NOT NULL'),
        ('encounter_notes',             'TEXT NOT NULL'),
    ),  ['CONSTRAINT superkey UNIQUE (encounter_text)']
    )
    # List of recognition directed edges (roi_1) --score--> (roi_2)
    ibs.db.schema('recognitions', (
        ('recognition_uid',             'INTEGER PRIMARY KEY'),
        ('roi_uid1',                    'INTEGER NOT NULL'),
        ('roi_uid2',                    'INTEGER NOT NULL'),
        ('recognition_score',           'REAL NOT NULL'),
        ('recognition_roirank',         'INTEGER NOT NULL'),
        ('recognition_namerank',        'INTEGER NOT NULL'),
        ('recognition_notes',           'TEXT'),
    ),  ['CONSTRAINT superkey UNIQUE (roi_uid1, roi_uid2)']
    )
    # Relationship between encounters and images (many to many mapping)
    # egpairs stands for encounter-image-pairs.
    ibs.db.schema('egpairs', (
        ('egpair_uid',                  'INTEGER PRIMARY KEY'),
        ('image_uid',                   '%s NOT NULL' % IMAGE_UID_TYPE),
        ('encounter_uid',               'INTEGER'),
    ),  ['CONSTRAINT superkey UNIQUE (image_uid, encounter_uid)']
    )
    # Detection and identification algorithm configurations, populated
    # with caching information
    ibs.db.schema('configs', (
        ('config_uid',                   'INTEGER PRIMARY KEY'),
        ('config_suffix',                'TEXT NOT NULL'),
    ),  ['CONSTRAINT superkey UNIQUE (config_suffix)']
    )
