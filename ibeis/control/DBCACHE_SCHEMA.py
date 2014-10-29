"""
Module Licence and docstring
"""
from __future__ import absolute_import, division, print_function
from ibeis import constants


# =======================
# Schema Version 1.0.0
# =======================


def update_1_0_0(ibs):
    ##########################
    # SECOND ORDER           #
    ##########################
    # TODO: constraint needs modification
    ibs.dbcache.add_table(constants.CHIP_TABLE, (
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

    ibs.dbcache.add_table(constants.FEATURE_TABLE, (
        ('feature_rowid',                'INTEGER PRIMARY KEY'),
        ('chip_rowid',                   'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('feature_num_feats',            'INTEGER NOT NULL'),
        ('feature_keypoints',            'NUMPY'),
        # Maybe change name to feature_vecs
        ('feature_sifts',                'NUMPY'),
    ),
        superkey_colnames=['chip_rowid, config_rowid'],
        docstr='''
        Used to store individual chip features (ellipses)''')


# =======================
# Schema Version 1.0.1
# =======================


def update_1_0_1(ibs):
    # When you're ready to make this schema update go live, simply
    # bump ibs.dbcache_version_expected in the controller to '1.0.1'
    ibs.dbcache.add_table(constants.RESIDUAL_TABLE, (
        ('residual_rowid',               'INTEGER PRIMARY KEY'),
        ('feature_rowid',                'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('residual_vector',              'NUMPY'),
    ),
        superkey_colnames=['residual_rowid', 'feature_rowid', 'config_rowid'],
        docstr='''
        Used to store individual SMK/ASMK residual vectors for features''')
    pass


def update_1_0_2(ibs):
    # Change name of feature_sifts to feature_vecs and
    # add new column for feature_weights
    ibs.dbcache.modify_table(constants.FEATURE_TABLE, (
        ('feature_sifts',   'feature_vecs',                 '', None),
        (           None, 'feature_weight', 'REAL DEFAULT 1.0', None),
    ))


# ========================
# Valid Versions & Mapping
# ========================


base = constants.BASE_DATABASE_VERSION
VALID_VERSIONS = {
    #version:   (Pre-Update Function,  Update Function,    Post-Update Function)
    base   :    (None,                 None,               None                ),
    '1.0.0':    (None,                 update_1_0_0,       None                ),
    '1.0.1':    (None,                 update_1_0_1,       None                ),
    '1.0.2':    (None,                 update_1_0_2,       None                ),
}
