"""
Module Licence and docstring
"""
from __future__ import absolute_import, division, print_function
from ibeis import constants


# =======================
# Schema Version Current
# =======================


VERSION_CURRENT = '1.1.1'

def update_current(db, ibs=None):
    db.add_table(constants.AL_RELATION_TABLE, (
        ('alr_rowid',                    'INTEGER PRIMARY KEY'),
        ('annot_rowid',                  'INTEGER NOT NULL'),
        ('lblannot_rowid',               'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('alr_confidence',               'REAL DEFAULT 0.0'),
    ),
        superkey_colnames=['annot_rowid', 'lblannot_rowid', 'config_rowid'],
        docstr='''
        Used to store one-to-many the relationship between annotations (annots)
        and labels''')

    db.add_table(constants.ANNOTATION_TABLE, (
        ('annot_rowid',                  'INTEGER PRIMARY KEY'),
        ('annot_parent_rowid',           'INTEGER'),
        ('annot_uuid',                   'UUID NOT NULL'),
        ('image_rowid',                  'INTEGER NOT NULL'),
        ('annot_xtl',                    'INTEGER NOT NULL'),
        ('annot_ytl',                    'INTEGER NOT NULL'),
        ('annot_width',                  'INTEGER NOT NULL'),
        ('annot_height',                 'INTEGER NOT NULL'),
        ('annot_theta',                  'REAL DEFAULT 0.0'),
        ('annot_num_verts',              'INTEGER NOT NULL'),
        ('annot_verts',                  'TEXT'),
        ('annot_viewpoint',              'REAL DEFAULT 0.0'),
        ('annot_detect_confidence',      'REAL DEFAULT -1.0'),
        ('annot_exemplar_flag',          'INTEGER DEFAULT 0'),
        ('annot_note',                   'TEXT'),
    ),
        superkey_colnames=['annot_uuid'],
        docstr='''
        Mainly used to store the geometry of the annotation within its parent
        image The one-to-many relationship between images and annotations is
        encoded here Attributes are stored in the Annotation Label Relationship
        Table''')

    db.add_table(constants.CONFIG_TABLE, (
        ('config_rowid',                 'INTEGER PRIMARY KEY'),
        ('contributor_rowid',            'INTEGER'),
        ('config_suffix',                'TEXT NOT NULL'),
    ),
        superkey_colnames=['contributor_rowid', 'config_suffix'],
        docstr='''
        Used to store the ids of algorithm configurations that generate
        annotation lblannots.  Each user will have a config id for manual
        contributions ''')

    db.add_table(constants.CONTRIBUTOR_TABLE, (
        ('contributor_rowid',            'INTEGER PRIMARY KEY'),
        ('contributor_uuid',             'UUID NOT NULL'),
        ('contributor_tag',              'TEXT'),
        ('contributor_name_first',       'TEXT'),
        ('contributor_name_last',        'TEXT'),
        ('contributor_location_city',    'TEXT'),
        ('contributor_location_state',   'TEXT'),
        ('contributor_location_country', 'TEXT'),
        ('contributor_location_zip',     'TEXT'),
        ('contributor_note',             'TEXT'),
    ),
        superkey_colnames=['contributor_tag'],
        docstr='''
        Used to store the contributors to the project
        ''')

    db.add_table(constants.EG_RELATION_TABLE, (
        ('egr_rowid',                    'INTEGER PRIMARY KEY'),
        ('image_rowid',                  'INTEGER NOT NULL'),
        ('encounter_rowid',              'INTEGER'),
    ),
        superkey_colnames=['image_rowid, encounter_rowid'],
        docstr='''
        Relationship between encounters and images (many to many mapping) the
        many-to-many relationship between images and encounters is encoded here
        encounter_image_relationship stands for encounter-image-pairs.''')

    db.add_table(constants.ENCOUNTER_TABLE, (
        ('encounter_rowid',              'INTEGER PRIMARY KEY'),
        ('encounter_uuid',               'UUID NOT NULL'),
        ('config_rowid',                 'INTEGER'),
        ('encounter_text',               'TEXT NOT NULL'),
        ('encounter_note',               'TEXT NOT NULL'),
    ),
        superkey_colnames=['encounter_uuid', 'encounter_text'],
        docstr='''
        List of all encounters''')

    db.add_table(constants.GL_RELATION_TABLE, (
        ('glr_rowid',                    'INTEGER PRIMARY KEY'),
        ('image_rowid',                  'INTEGER NOT NULL'),
        ('lblimage_rowid',               'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('glr_confidence',               'REAL DEFAULT 0.0'),
    ),
        superkey_colnames=['image_rowid', 'lblimage_rowid', 'config_rowid'],
        docstr='''
        Used to store one-to-many the relationship between images
        and labels''')

    db.add_table(constants.IMAGE_TABLE, (
        ('image_rowid',                  'INTEGER PRIMARY KEY'),
        ('contributor_rowid',            'INTEGER'),
        ('image_uuid',                   'UUID NOT NULL'),
        ('image_uri',                    'TEXT NOT NULL'),
        ('image_ext',                    'TEXT NOT NULL'),
        ('image_original_name',          'TEXT NOT NULL'),
        ('image_width',                  'INTEGER DEFAULT -1'),
        ('image_height',                 'INTEGER DEFAULT -1'),
        ('image_time_posix',             'INTEGER DEFAULT -1'),
        ('image_gps_lat',                'REAL DEFAULT -1.0'),
        ('image_gps_lon',                'REAL DEFAULT -1.0'),
        ('image_toggle_enabled',         'INTEGER DEFAULT 0'),
        ('image_toggle_reviewed',        'INTEGER DEFAULT 0'),
        ('image_note',                   'TEXT'),
    ),
        superkey_colnames=['image_uuid'],
        docstr='''
        First class table used to store image locations and meta-data''')

    db.add_table(constants.LBLTYPE_TABLE, (
        ('lbltype_rowid',                'INTEGER PRIMARY KEY'),
        ('lbltype_text',                 'TEXT NOT NULL'),
        ('lbltype_default',              'TEXT NOT NULL'),
    ),
        superkey_colnames=['lbltype_text'],
        docstr='''
        List of keys used to define the categories of annotation lables, text
        is for human-readability. The lbltype_default specifies the
        lblannot_value of annotations with a relationship of some
        lbltype_rowid''')

    db.add_table(constants.LBLANNOT_TABLE, (
        ('lblannot_rowid',               'INTEGER PRIMARY KEY'),
        ('lblannot_uuid',                'UUID NOT NULL'),
        ('lbltype_rowid',                'INTEGER NOT NULL'),
        ('lblannot_value',               'TEXT NOT NULL'),
        ('lblannot_note',                'TEXT'),
    ),
        superkey_colnames=['lbltype_rowid', 'lblannot_value'],
        docstr='''
        Used to store the labels / attributes of annotations.
        E.G name, species ''')

    db.add_table(constants.LBLIMAGE_TABLE, (
        ('lblimage_rowid',               'INTEGER PRIMARY KEY'),
        ('lblimage_uuid',                'UUID NOT NULL'),
        ('lbltype_rowid',                'INTEGER NOT NULL'),
        ('lblimage_value',               'TEXT NOT NULL'),
        ('lblimage_note',                'TEXT'),
    ),
        superkey_colnames=['lbltype_rowid', 'lblimage_value'],
        docstr='''
        Used to store the labels (attributes) of images''')

    db.add_table(constants.METADATA_TABLE, (
        ('metadata_rowid',               'INTEGER PRIMARY KEY'),
        ('metadata_key',                 'TEXT'),
        ('metadata_value',               'TEXT'),
    ),
        superkey_colnames=['metadata_key'],
        docstr='''
        The table that stores permanently all of the metadata about the
        database (tables, etc)''')
