"""
Module Licence and docstring
"""
from __future__ import absolute_import, division, print_function

#IMAGE_UID_TYPE = 'INTEGER'
#ROI_UID_TYPE   = 'INTEGER'
#NAME_UID_TYPE  = 'INTEGER'


def define_IBEIS_schema(ibs):
    # TODO, Add algoritm config column
    ibs.db.schema('images', (
        ('image_rowid',                    'INTEGER PRIMARY KEY'),
        ('image_uuid',                   'UUID NOT NULL'),
        ('image_uri',                    'TEXT NOT NULL'),
        ('image_ext',                    'TEXT NOT NULL'),
        ('image_original_name',          'TEXT NOT NULL'),  # We could parse this out
        #('image_original_path',          'TEXT NOT NULL'),
        ('image_width',                  'INTEGER'),
        ('image_height',                 'INTEGER'),
        ('image_exif_time_posix',        'INTEGER'),
        ('image_exif_gps_lat',           'REAL'),   # there doesn't seem to exist a GPSPoint in SQLite
        ('image_exif_gps_lon',           'REAL'),
        ('image_confidence',             'REAL DEFAULT -1.0',),  # Move to an algocfg table?
        ('image_toggle_enabled',         'INTEGER DEFAULT 0'),
        ('image_toggle_aif',             'INTEGER DEFAULT 0'),
        ('image_notes',                  'TEXT',),
    ), ['CONSTRAINT superkey UNIQUE (image_uuid)']
    )
    # Used to store individual chip identities (Fred, Sue, ...)
    ibs.db.schema('names', (
        ('name_rowid',                     'INTEGER PRIMARY KEY'),
        ('name_text',                    'TEXT NOT NULL'),
        ('name_notes',                   'TEXT',),
    ), ['CONSTRAINT superkey UNIQUE (name_text)']
    )
    # Used to store the detected ROIs / bboxed annotations
    ibs.db.schema('rois', (
        ('roi_rowid',                      'INTEGER PRIMARY KEY'),
        ('roi_uuid',                     'UUID NOT NULL'),
        ('image_rowid',                    'INTEGER NOT NULL'),
        ('name_rowid',                     'INTEGER NOT NULL'),
        ('roi_xtl',                      'INTEGER NOT NULL'),
        ('roi_ytl',                      'INTEGER NOT NULL'),
        ('roi_width',                    'INTEGER NOT NULL'),
        ('roi_height',                   'INTEGER NOT NULL'),
        ('roi_theta',                    'REAL DEFAULT 0.0'),
        ('roi_num_verts',                'INTEGER NOT NULL'),
        ('roi_verts',                    'TEXT'),
        ('roi_viewpoint',                'INTEGER DEFAULT 0'),
        ('roi_species_text',             'TEXT'),
        ('roi_detect_confidence',        'REAL DEFAULT -1.0'),
        ('roi_exemplar_flag',            'INTEGER DEFAULT 0'),
        ('roi_notes',                    'TEXT'),
    ), ['CONSTRAINT superkey UNIQUE (roi_uuid)']
    )
    # Used to store the relationship between Annotation (ROIs) and Labels
    ibs.db.schema('roi_label_relationship', (
        ('rlr_id',                         'INTEGER PRIMARY KEY'),
        ('roi_rowid',                      'INTEGER'),
        ('label_rowid',                    'INTEGER'),
        ('config_rowid',                   'INTEGER'),
        ('label_confidence',               'REAL DEFAULT 0.0'),
    ))
    """
     Used to store the results of Annotations
     the label key must be in 
     {
     "INDIVIDUAL_KEY": 0, 
     "SPECIES_KEY": 1,
     }
    """ 
    ibs.db.schema('labels', (
        ('label_rowid',                   'INTEGER PRIMARY KEY'),
        ('label_key',                     'INTEGER'), # this is "category" in the proposal
        ('label_value',                   'TEXT'),
        ('label_note',                    'TEXT'),
    ), ['CONSTRAINT superkey UNIQUE (label_key, label_value)']
    )
    # Used to store *processed* ROIs as segmentations
    ibs.db.schema('masks', (
        ('mask_rowid',                     'INTEGER PRIMARY KEY'),
        ('config_rowid',                   'INTEGER DEFAULT 0'),
        ('roi_rowid',                      'INTEGER NOT NULL'),
        ('mask_uri',                     'TEXT NOT NULL'),
    ))
    # Used to store *processed* ROIs as chips
    ibs.db.schema('chips', (
        ('chip_rowid',                     'INTEGER PRIMARY KEY'),
        ('roi_rowid',                      'INTEGER NOT NULL'),
        ('config_rowid',                   'INTEGER DEFAULT 0'),
        ('chip_uri',                     'TEXT'),
        ('chip_width',                   'INTEGER NOT NULL'),
        ('chip_height',                  'INTEGER NOT NULL'),
    ), ['CONSTRAINT superkey UNIQUE (roi_rowid, config_rowid)']  # TODO: constraint needs modify
    )
    # Used to store individual chip features (ellipses)
    ibs.db.schema('features', (
        ('feature_rowid',                  'INTEGER PRIMARY KEY'),
        ('chip_rowid',                     'INTEGER NOT NULL'),
        ('config_rowid',                   'INTEGER DEFAULT 0'),
        ('feature_num_feats',            'INTEGER NOT NULL'),
        ('feature_keypoints',            'NUMPY'),
        ('feature_sifts',                'NUMPY'),
    ), ['CONSTRAINT superkey UNIQUE (chip_rowid, config_rowid)']
    )
    # List of all encounters
    ibs.db.schema('encounters', (
        ('encounter_rowid',               'INTEGER PRIMARY KEY'),
        ('encounter_uuid',                'UUID NOT NULL'),
        ('encounter_text',              'TEXT NOT NULL'),
        ('encounter_notes',             'TEXT NOT NULL'),
    ),  ['CONSTRAINT superkey UNIQUE (encounter_text)']
    )
    # List of recognition directed edges (roi_1) --score--> (roi_2)
    ibs.db.schema('recognitions', (
        ('recognition_rowid',             'INTEGER PRIMARY KEY'),
        ('roi_rowid1',                    'INTEGER NOT NULL'),
        ('roi_rowid2',                    'INTEGER NOT NULL'),
        ('recognition_score',           'REAL NOT NULL'),
        ('recognition_roirank',         'INTEGER NOT NULL'),
        ('recognition_namerank',        'INTEGER NOT NULL'),
        ('recognition_notes',           'TEXT'),
    ),  ['CONSTRAINT superkey UNIQUE (roi_rowid1, roi_rowid2)']
    )
    # Relationship between encounters and images (many to many mapping)
    # encounter_image_relationship stands for encounter-image-pairs.
    ibs.db.schema('encounter_image_relationship', (
        ('egpair_rowid',                  'INTEGER PRIMARY KEY'),
        ('image_rowid',                   'INTEGER NOT NULL'),
        ('encounter_rowid',               'INTEGER'),
    ),  ['CONSTRAINT superkey UNIQUE (image_rowid, encounter_rowid)']
    )
    # Detection and identification algorithm configurations, populated
    # with caching information
    ibs.db.schema('configs', (
        ('config_rowid',                   'INTEGER PRIMARY KEY'),
        ('config_suffix',                'TEXT NOT NULL'),
    ),  ['CONSTRAINT superkey UNIQUE (config_suffix)']
    )
