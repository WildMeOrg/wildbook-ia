"""
Module Licence and docstring
"""
from __future__ import absolute_import, division, print_function
from ibeis import constants


def define_IBEIS_schema(ibs):
    ibs.db.schema(constants.IMAGE_TABLE, (
        ('image_rowid',                  'INTEGER PRIMARY KEY'),
        ('image_uuid',                   'UUID NOT NULL'),
        ('image_uri',                    'TEXT NOT NULL'),
        ('image_ext',                    'TEXT NOT NULL'),
        ('image_original_name',          'TEXT NOT NULL'),  # We could parse this out
        #('image_original_path',          'TEXT NOT NULL'),
        ('image_width',                  'INTEGER DEFAULT -1'),
        ('image_height',                 'INTEGER DEFAULT -1'),
        ('image_time_posix',             'INTEGER DEFAULT -1'),  # this should probably be UCT
        ('image_gps_lat',                'REAL DEFAULT -1.0'),   # there doesn't seem to exist a GPSPoint in SQLite (TODO: make one in the __SQLITE3__ custom types
        ('image_gps_lon',                'REAL DEFAULT -1.0'),
        ('image_toggle_enabled',         'INTEGER DEFAULT 0'),
        ('image_toggle_reviewed',        'INTEGER DEFAULT 0'),
        ('image_note',                   'TEXT',),
    ), ['CONSTRAINT superkey UNIQUE (image_uuid)'],
        docstr='''
        First class table used to store image locations and meta-data''')

    ibs.db.schema(constants.ANNOTATION_TABLE, (
        ('annot_rowid',                  'INTEGER PRIMARY KEY'),
        ('annot_uuid',                   'UUID NOT NULL'),
        ('image_rowid',                  'INTEGER NOT NULL'),
        ('annot_xtl',                    'INTEGER NOT NULL'),
        ('annot_ytl',                    'INTEGER NOT NULL'),
        ('annot_width',                  'INTEGER NOT NULL'),
        ('annot_height',                 'INTEGER NOT NULL'),
        ('annot_theta',                  'REAL DEFAULT 0.0'),
        ('annot_num_verts',              'INTEGER NOT NULL'),
        ('annot_verts',                  'TEXT'),
        ('annot_detect_confidence',      'REAL DEFAULT -1.0'),
        ('annot_exemplar_flag',          'INTEGER DEFAULT 0'),
        ('annot_note',                   'TEXT'),
    ), ['CONSTRAINT superkey UNIQUE (annot_uuid)'],
        docstr='''
        Mainly used to store the geometry of the annotation within its parent
        image The one-to-many relationship between images and annotations is
        encoded here Attributes are stored in the Annotation Label Relationship
        Table''')

    ibs.db.schema(constants.AL_RELATION_TABLE, (
        ('alr_rowid',                      'INTEGER PRIMARY KEY'),
        ('annot_rowid',                    'INTEGER NOT NULL'),
        ('lblannot_rowid',                 'INTEGER NOT NULL'),
        ('config_rowid',                   'INTEGER DEFAULT 0'),
        ('alr_confidence',                 'REAL DEFAULT 0.0'),
    ), ['CONSTRAINT superkey UNIQUE (annot_rowid, lblannot_rowid, config_rowid)'],
        docstr='''
        Used to store one-to-many the relationship between annotations (annots)
        and labels''')

    ibs.db.schema(constants.GL_RELATION_TABLE, (
        ('glr_rowid',                    'INTEGER PRIMARY KEY'),
        ('image_rowid',                  'INTEGER NOT NULL'),
        ('lblimage_rowid',               'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('glr_confidence',               'REAL DEFAULT 0.0'),
    ), ['CONSTRAINT superkey UNIQUE (image_rowid, lblimage_rowid, config_rowid)'],
        docstr='''
        Used to store one-to-many the relationship between images
        and labels''')

    ibs.db.schema(constants.LBLIMAGE_TABLE, (
        ('lblimage_rowid',             'INTEGER PRIMARY KEY'),
        ('lblimage_uuid',              'UUID NOT NULL'),
        ('lbltype_rowid',              'INTEGER NOT NULL'),  # this is "category" in the proposal
        ('lblimage_value',             'TEXT NOT NULL'),
        ('lblimage_note',              'TEXT'),
    ), ['CONSTRAINT superkey UNIQUE (lbltype_rowid, lblimage_value)'],
        docstr='''
        Used to store the labels (attributes) of images''')

    ibs.db.schema(constants.LBLANNOT_TABLE, (
        ('lblannot_rowid',                   'INTEGER PRIMARY KEY'),
        ('lblannot_uuid',                    'UUID NOT NULL'),
        ('lbltype_rowid',                    'INTEGER NOT NULL'),  # this is "category" in the proposal
        ('lblannot_value',                   'TEXT NOT NULL'),
        ('lblannot_note',                    'TEXT'),
    ), ['CONSTRAINT superkey UNIQUE (lbltype_rowid, lblannot_value)'],
        docstr='''
        Used to store the labels / attributes of annotations.
        E.G name, species ''')

    ibs.db.schema(constants.LBLTYPE_TABLE, (
        ('lbltype_rowid',                'INTEGER PRIMARY KEY'),
        ('lbltype_text',                 'TEXT NOT NULL'),
        ('lbltype_default',              'TEXT NOT NULL'),
    ), ['CONSTRAINT superkey UNIQUE (lbltype_text)'],
        docstr='''
        List of keys used to define the categories of annotation lables, text
        is for human-readability. The lbltype_default specifies the
        lblannot_value of annotations with a relationship of some
        lbltype_rowid''')

    ibs.db.schema(constants.ENCOUNTER_TABLE, (
        ('encounter_rowid',              'INTEGER PRIMARY KEY'),
        ('encounter_uuid',               'UUID NOT NULL'),
        ('encounter_text',               'TEXT NOT NULL'),
        ('encounter_note',               'TEXT NOT NULL'),
    ),  ['CONSTRAINT superkey UNIQUE (encounter_text)'],
        docstr='''
        List of all encounters''')

    ibs.db.schema(constants.EG_RELATION_TABLE, (
        ('egr_rowid',                    'INTEGER PRIMARY KEY'),
        ('image_rowid',                  'INTEGER NOT NULL'),
        ('encounter_rowid',              'INTEGER'),
    ),  ['CONSTRAINT superkey UNIQUE (image_rowid, encounter_rowid)'],
        docstr='''
        Relationship between encounters and images (many to many mapping) the
        many-to-many relationship between images and encounters is encoded here
        encounter_image_relationship stands for encounter-image-pairs.''')

    ibs.db.schema(constants.CONFIG_TABLE, (
        ('config_rowid',                 'INTEGER PRIMARY KEY'),
        ('config_suffix',                'TEXT NOT NULL'),
    ),  ['CONSTRAINT superkey UNIQUE (config_suffix)'],
        docstr='''
        Used to store the ids of algorithm configurations that generate
        annotation lblannots.  Each user will have a config id for manual
        contributions ''')

    # TODO: constraint needs modify
    ibs.db.schema(constants.CHIP_TABLE, (
        ('chip_rowid',                   'INTEGER PRIMARY KEY'),
        ('annot_rowid',                  'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('chip_uri',                     'TEXT'),
        ('chip_width',                   'INTEGER NOT NULL'),
        ('chip_height',                  'INTEGER NOT NULL'),
    ), ['CONSTRAINT superkey UNIQUE (annot_rowid, config_rowid)'],
        docstr='''
        Used to store *processed* annots as chips''')

    ibs.db.schema(constants.FEATURE_TABLE, (
        ('feature_rowid',                'INTEGER PRIMARY KEY'),
        ('chip_rowid',                   'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('feature_num_feats',            'INTEGER NOT NULL'),
        ('feature_keypoints',            'NUMPY'),
        ('feature_sifts',                'NUMPY'),
    ), ['CONSTRAINT superkey UNIQUE (chip_rowid, config_rowid)'],
        docstr='''
        Used to store individual chip features (ellipses)''')

    ibs.db.schema('IBEIS_VERSION_TABLE', (
        ('version_rowid',          'INTEGER PRIMARY KEY'),
        ('version_text',           'TEXT'),
    ),
        docstr='''
        holds the schema version info''')

    #
    # UNUSED / DEPRECATED
    #
    #IMAGE_UID_TYPE = 'INTEGER'
    #annot_UID_TYPE = 'INTEGER'
    #NAME_UID_TYPE  = 'INTEGER'

    # List of recognition directed edges (annot_1) --score--> (annot_2)
    # ibs.db.schema('recognitions', (
    #     ('recognition_rowid',           'INTEGER PRIMARY KEY'),
    #     ('annot_rowid1',                'INTEGER NOT NULL'),
    #     ('annot_rowid2',                'INTEGER NOT NULL'),
    #     ('recognition_score',           'REAL NOT NULL'),
    #     ('recognition_annotrank',       'INTEGER NOT NULL'),
    #     ('recognition_namerank',        'INTEGER NOT NULL'),
    #     ('recognition_note',            'TEXT'),
    # ),  ['CONSTRAINT superkey UNIQUE (annot_rowid1, annot_rowid2)'])

    # Used to store *processed* annots as segmentations
    # ibs.db.schema('masks', (
    #     ('mask_rowid',                  'INTEGER PRIMARY KEY'),
    #     ('config_rowid',                'INTEGER DEFAULT 0'),
    #     ('annot_rowid',                 'INTEGER NOT NULL'),
    #     ('mask_uri',                    'TEXT NOT NULL'),
    # ))

    # Used to store individual chip identities (Fred, Sue, ...)
    #ibs.db.schema('names', (
    #    ('name_rowid',                   'INTEGER PRIMARY KEY'),
    #    ('name_text',                    'TEXT NOT NULL'),
    #    ('name_note',                    'TEXT',),
    #), ['CONSTRAINT superkey UNIQUE (name_text)'])
