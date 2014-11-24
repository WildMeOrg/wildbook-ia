"""
Module Licence and docstring
"""
from __future__ import absolute_import, division, print_function
from ibeis import constants
try:
    from ibeis.control import DB_SCHEMA_CURRENT
    UPDATE_CURRENT  = DB_SCHEMA_CURRENT.update_current
    VERSION_CURRENT = DB_SCHEMA_CURRENT.VERSION_CURRENT
except:
    UPDATE_CURRENT  = None
    VERSION_CURRENT = None
    print("[dbcache] NO DB_SCHEMA_CURRENT AUTO-GENERATED!")
import utool
profile = utool.profile


# =======================
# Schema Version 1.0.0
# =======================


@profile
def update_1_0_0(db, ibs=None):
    db.add_table(constants.IMAGE_TABLE, (
        ('image_rowid',                  'INTEGER PRIMARY KEY'),
        ('image_uuid',                   'UUID NOT NULL'),
        ('image_uri',                    'TEXT NOT NULL'),
        ('image_ext',                    'TEXT NOT NULL'),
        ('image_original_name',          'TEXT NOT NULL'),  # We could parse this out of original_path
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

    db.add_table(constants.ENCOUNTER_TABLE, (
        ('encounter_rowid',              'INTEGER PRIMARY KEY'),
        ('encounter_uuid',               'UUID NOT NULL'),
        ('encounter_text',               'TEXT NOT NULL'),
        ('encounter_note',               'TEXT NOT NULL'),
    ),
        superkey_colnames=['encounter_text'],
        docstr='''
        List of all encounters''')

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

    db.add_table(constants.CONFIG_TABLE, (
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
    db.add_table(constants.ANNOTATION_TABLE, (
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

    db.add_table(constants.LBLIMAGE_TABLE, (
        ('lblimage_rowid',               'INTEGER PRIMARY KEY'),
        ('lblimage_uuid',                'UUID NOT NULL'),
        ('lbltype_rowid',                'INTEGER NOT NULL'),  # this is "category" in the proposal
        ('lblimage_value',               'TEXT NOT NULL'),
        ('lblimage_note',                'TEXT'),
    ),
        superkey_colnames=['lbltype_rowid', 'lblimage_value'],
        docstr='''
        Used to store the labels (attributes) of images''')

    db.add_table(constants.LBLANNOT_TABLE, (
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
    db.add_table(constants.CHIP_TABLE, (
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

    db.add_table(constants.FEATURE_TABLE, (
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

    ##########################
    # THIRD ORDER            #
    ##########################
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


@profile
def post_1_0_0(db, ibs=None):
    # We are dropping the versions table and rather using the metadata table
    db.drop_table(constants.VERSIONS_TABLE)


# =======================
# Schema Version 1.0.1
# =======================


@profile
def update_1_0_1(db, ibs=None):
    # Add a contributor's table
    db.add_table(constants.CONTRIBUTOR_TABLE, (
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

    db.modify_table(constants.IMAGE_TABLE, (
        # add column to v1.0.0 at index 1
        (1, 'contributor_rowid', 'INTEGER', None),
    ))

    db.modify_table(constants.ANNOTATION_TABLE, (
        # add column to v1.0.0 at index 1
        (1, 'annot_parent_rowid', 'INTEGER', None),
    ))

    db.modify_table(constants.FEATURE_TABLE, (
        # append column to v1.0.0 because None
        (None, 'feature_weight', 'REAL DEFAULT 1.0', None),
    ))


# =======================
# Schema Version 1.0.2
# =======================


@profile
def update_1_0_2(db, ibs=None):
    # Fix the contibutor table's constraint
    db.modify_table(constants.CONTRIBUTOR_TABLE, (
        # add column to v1.0.1 at index 1
        (1, 'contributor_uuid', 'UUID NOT NULL', None),
    ),
        table_constraints=[],
        superkey_colnames=['contributor_tag']
    )

# =======================
# Schema Version 1.1.0
# =======================


@profile
def update_1_1_0(db, ibs=None):
    # Moving chips and features to their own cache database
    db.drop_table(constants.CHIP_TABLE)
    db.drop_table(constants.FEATURE_TABLE)

    # Add viewpoint (radians) to annotations
    db.modify_table(constants.ANNOTATION_TABLE, (
        # add column to v1.0.2 at index 11
        (11, 'annot_viewpoint', 'REAL DEFAULT 0.0', None),
    ))

    # Add contributor to configs
    db.modify_table(constants.CONFIG_TABLE, (
        # add column to v1.0.2 at index 1
        (1, 'contributor_uuid', 'UUID', None),
    ),
        table_constraints=[],
        # FIXME: This change may have broken things
        superkey_colnames=['contributor_uuid', 'config_suffix']
    )

    # Add config to encounters
    db.modify_table(constants.ENCOUNTER_TABLE, (
        # add column to v1.0.2 at index 2
        (2, 'config_rowid', 'INTEGER', None),
    ),
        table_constraints=[],
        superkey_colnames=['encounter_uuid', 'encounter_text']
    )

    # Error in the drop table script, re-drop again from post_1_0_0 to kill table's metadata
    db.drop_table(constants.VERSIONS_TABLE)


# =======================
# Schema Version 1.1.1
# =======================


@profile
def update_1_1_1(db, ibs=None):
    # Change name of column
    db.modify_table(constants.CONFIG_TABLE, (
        # rename column and change it's type
        ('contributor_uuid', 'contributor_rowid', '', None),
    ),
        table_constraints=[],
        superkey_colnames=['contributor_rowid', 'config_suffix']
    )

    # Change type of column
    db.modify_table(constants.CONFIG_TABLE, (
        # rename column and change it's type
        ('contributor_rowid', '', 'INTEGER', None),
    ))

    # Change type of columns
    db.modify_table(constants.CONTRIBUTOR_TABLE, (
        # Update column's types
        ('contributor_location_zip', '', 'TEXT', None),
        ('contributor_note', '', 'TEXT', None),
    ))


@profile
def update_1_2_0(db, ibs=None):
    # Add columns to annotaiton table
    db.modify_table(constants.ANNOTATION_TABLE, (
        # the visual uuid will be unique w.r.t. the appearence of the annotation
        (None, 'annot_visual_uuid', 'UUID', None),
        # the visual uuid will be unique w.r.t. the appearence, name, and species of the annotation
        (None, 'annot_semantic_uuid', 'UUID', None),
        (None, 'name_rowid',    'INTEGER DEFAULT 0', None),
        (None, 'species_rowid', 'INTEGER DEFAULT 0', None),

    ),
    )


# ========================
# Valid Versions & Mapping
# ========================


base = constants.BASE_DATABASE_VERSION
VALID_VERSIONS = utool.odict([
    #version:   (Pre-Update Function,  Update Function,    Post-Update Function)
    (base   ,    (None,                 None,               None                )),
    ('1.0.0',    (None,                 update_1_0_0,       post_1_0_0          )),
    ('1.0.1',    (None,                 update_1_0_1,       None                )),
    ('1.0.2',    (None,                 update_1_0_2,       None                )),
    ('1.1.0',    (None,                 update_1_1_0,       None                )),
    ('1.1.1',    (None,                 update_1_1_1,       None                )),
    ('1.2.0',    (None,                 update_1_2_0,       None                )),
])


def test_dbschema():
    """
    test_dbschema

    CommandLine:
        python ibeis/control/DB_SCHEMA.py
        python ibeis/control/DB_SCHEMA.py -n=0
        python ibeis/control/DB_SCHEMA.py -n=1
        python ibeis/control/DB_SCHEMA.py -n=-1
        python ibeis/control/DB_SCHEMA.py --force-incremental-db-update

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.DB_SCHEMA import *  # NOQA
        >>> test_dbschema()
    """
    from ibeis.control import DB_SCHEMA
    from ibeis.control import _sql_helpers
    from ibeis import params
    autogenerate = params.args.dump_autogen_schema
    n = utool.get_argval('-n', int, default=-1)
    db = _sql_helpers.get_nth_test_schema_version(DB_SCHEMA, n=n, autogenerate=autogenerate)
    autogen_str = db.get_schema_current_autogeneration_str()
    print(autogen_str)
    print(' Run with --dump-autogen-schema to autogenerate latest schema version')


if __name__ == '__main__':
    """
    python ibeis/model/preproc/preproc_chip.py
    python ibeis/control/DB_SCHEMA.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut
    ut.doctest_funcs()
