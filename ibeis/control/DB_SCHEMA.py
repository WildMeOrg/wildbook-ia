"""
Module Licence and docstring
"""
from __future__ import absolute_import, division, print_function
from ibeis import constants


# =======================
# Schema Version 1.0.0
# =======================


def update_1_0_0(ibs):
    ibs.db.add_table(constants.IMAGE_TABLE, (
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
    ),
        superkey_colnames=['image_uuid'],
        docstr='''
        First class table used to store image locations and meta-data''')

    ibs.db.add_table(constants.ENCOUNTER_TABLE, (
        ('encounter_rowid',              'INTEGER PRIMARY KEY'),
        ('encounter_uuid',               'UUID NOT NULL'),
        ('encounter_text',               'TEXT NOT NULL'),
        ('encounter_note',               'TEXT NOT NULL'),
    ),
        superkey_colnames=['encounter_text'],
        docstr='''
        List of all encounters''')

    ibs.db.add_table(constants.LBLTYPE_TABLE, (
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

    ibs.db.add_table(constants.CONFIG_TABLE, (
        ('config_rowid',                 'INTEGER PRIMARY KEY'),
        ('config_suffix',                'TEXT NOT NULL'),
    ),
        superkey_colnames=['config_suffix'],
        docstr='''
        Used to store the ids of algorithm configurations that generate
        annotation lblannots.  Each user will have a config id for manual
        contributions ''')

    ##########################
    # FIRST ORDER            #
    ##########################
    ibs.db.add_table(constants.ANNOTATION_TABLE, (
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
    ),
        superkey_colnames=['annot_uuid'],
        docstr='''
        Mainly used to store the geometry of the annotation within its parent
        image The one-to-many relationship between images and annotations is
        encoded here Attributes are stored in the Annotation Label Relationship
        Table''')

    ibs.db.add_table(constants.LBLIMAGE_TABLE, (
        ('lblimage_rowid',               'INTEGER PRIMARY KEY'),
        ('lblimage_uuid',                'UUID NOT NULL'),
        ('lbltype_rowid',                'INTEGER NOT NULL'),  # this is "category" in the proposal
        ('lblimage_value',               'TEXT NOT NULL'),
        ('lblimage_note',                'TEXT'),
    ),
        superkey_colnames=['lbltype_rowid', 'lblimage_value'],
        docstr='''
        Used to store the labels (attributes) of images''')

    ibs.db.add_table(constants.LBLANNOT_TABLE, (
        ('lblannot_rowid',               'INTEGER PRIMARY KEY'),
        ('lblannot_uuid',                'UUID NOT NULL'),
        ('lbltype_rowid',                'INTEGER NOT NULL'),  # this is "category" in the proposal
        ('lblannot_value',               'TEXT NOT NULL'),
        ('lblannot_note',                'TEXT'),
    ),
        superkey_colnames=['lbltype_rowid', 'lblannot_value'],
        docstr='''
        Used to store the labels / attributes of annotations.
        E.G name, species ''')

    ##########################
    # SECOND ORDER           #
    ##########################
    # TODO: constraint needs modification
    ibs.db.add_table(constants.CHIP_TABLE, (
        ('chip_rowid',                   'INTEGER PRIMARY KEY'),
        ('annot_rowid',                  'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('chip_uri',                     'TEXT'),
        ('chip_width',                   'INTEGER NOT NULL'),
        ('chip_height',                  'INTEGER NOT NULL'),
    ),
        superkey_colnames=['annot_rowid', 'config_rowid'],
        docstr='''
        Used to store *processed* annots as chips''')

    ibs.db.add_table(constants.FEATURE_TABLE, (
        ('feature_rowid',                'INTEGER PRIMARY KEY'),
        ('chip_rowid',                   'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('feature_num_feats',            'INTEGER NOT NULL'),
        ('feature_keypoints',            'NUMPY'),
        ('feature_sifts',                'NUMPY'),
    ),
        superkey_colnames=['chip_rowid, config_rowid'],
        docstr='''
        Used to store individual chip features (ellipses)''')

    ibs.db.add_table(constants.EG_RELATION_TABLE, (
        ('egr_rowid',                    'INTEGER PRIMARY KEY'),
        ('image_rowid',                  'INTEGER NOT NULL'),
        ('encounter_rowid',              'INTEGER'),
    ),
        superkey_colnames=['image_rowid, encounter_rowid'],
        docstr='''
        Relationship between encounters and images (many to many mapping) the
        many-to-many relationship between images and encounters is encoded here
        encounter_image_relationship stands for encounter-image-pairs.''')

    ##########################
    # THIRD ORDER            #
    ##########################
    ibs.db.add_table(constants.GL_RELATION_TABLE, (
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

    ibs.db.add_table(constants.AL_RELATION_TABLE, (
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


def post_1_0_0(ibs):
    # We are dropping the versions table and rather using the metadata table
    ibs.db.drop_table(constants.VERSIONS_TABLE)


# =======================
# Schema Version 1.0.1
# =======================


def update_1_0_1(ibs):
    ibs.db.add_table(constants.CONTRIBUTOR_TABLE, (
        ('contributor_rowid',            'INTEGER PRIMARY KEY'),
        ('contributor_tag',              'TEXT'),
        ('contributor_name_first',       'TEXT'),
        ('contributor_name_last',        'TEXT'),
        ('contributor_location_city',    'TEXT'),
        ('contributor_location_state',   'TEXT'),
        ('contributor_location_country', 'TEXT'),
        ('contributor_location_zip',     'INTEGER'),
        ('contributor_note',             'INTEGER'),
    ),
        superkey_colnames=['contributor_rowid'],
        docstr='''
        Used to store the contributors to the project
        ''')

    ibs.db.modify_table(constants.IMAGE_TABLE, (
        # add column at index 1
        (1, 'contributor_rowid', 'INTEGER', None),
    ))

    ibs.db.modify_table(constants.ANNOTATION_TABLE, (
        # add column at index 1
        (1, 'annot_parent_rowid', 'INTEGER', None),
    ))

    ibs.db.modify_table(constants.FEATURE_TABLE, (
        # append column because None
        (None, 'feature_weight', 'REAL DEFAULT 1.0', None),
    ))


# =======================
# Schema Version 1.0.2
# =======================


def update_1_0_2(ibs):
    ibs.db.modify_table(constants.CONTRIBUTOR_TABLE, (
        (1, 'contributor_uuid', 'UUID NOT NULL', None),
    ),
        table_constraints=[],
        superkey_colnames=['contributor_tag']
    )

# =======================
# Schema Version 1.0.3
# =======================


def update_1_0_3(ibs):
    ibs.db.drop_table(constants.CONFIG_TABLE)
    ibs.db.drop_table(constants.CHIP_TABLE)
    ibs.db.drop_table(constants.FEATURE_TABLE)


# ========================
# Valid Versions & Mapping
# ========================


base = constants.BASE_DATABASE_VERSION
VALID_VERSIONS = {
    #version:   (Pre-Update Function,  Update Function,    Post-Update Function)
    base   :    (None,                 None,               None                ),
    '1.0.0':    (None,                 update_1_0_0,       post_1_0_0          ),
    '1.0.1':    (None,                 update_1_0_1,       None                ),
    '1.0.2':    (None,                 update_1_0_2,       None                ),
    '1.0.3':    (None,                 update_1_0_3,       None                ),
}
