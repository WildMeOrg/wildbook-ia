"""
Module Licence and docstring
"""
from __future__ import absolute_import, division, print_function
from ibeis import constants


# =======================
# Schema Version 1.0.0
# =======================


def update_1_0_0(ibs):
    ibs.dbcache.add_table(constants.CONFIG_TABLE, (
        ('config_rowid',                 'INTEGER PRIMARY KEY'),
        ('config_suffix',                'TEXT NOT NULL'),
    ),
        superkey_colnames=['config_suffix'],
        docstr='''
        Used to store the ids of algorithm configurations that generate
        annotation lblannots.  Each user will have a config id for manual
        contributions ''')

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
        ('feature_sifts',                'NUMPY'),
    ),
        superkey_colnames=['chip_rowid, config_rowid'],
        docstr='''
        Used to store individual chip features (ellipses)''')


# ========================
# Valid Versions & Mapping
# ========================


base = constants.BASE_DATABASE_VERSION
VALID_VERSIONS = {
    #version:   (Pre-Update Function,  Update Function,    Post-Update Function)
    base   :    (None,                 None,               None                ),
    '1.0.0':    (None,                 update_1_0_0,       None          ),
}
