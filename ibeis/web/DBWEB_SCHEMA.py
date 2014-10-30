"""
Module Licence and docstring
"""
from __future__ import absolute_import, division, print_function
from ibeis import constants


VIEWPOINT_TABLE = 'viewpoints'


# =======================
# Schema Version 1.0.0
# =======================


def update_1_0_0(db, ibs=None):
    db.add_table(VIEWPOINT_TABLE, (
        ('viewpoint_rowid',              'INTEGER PRIMARY KEY'),
        ('viewpoint_aid',                'INTEGER'),
        ('viewpoint_cpath',              'TEXT'),
        ('viewpoint_value1',             'REAL'),
        ('viewpoint_value2',             'REAL'),
        ('viewpoint_value3',             'REAL'),
    ),
        superkey_colnames=['viewpoint_cpath'],
        docstr='''
        SQLite table to store the web state''')


# ========================
# Valid Versions & Mapping
# ========================


base = constants.BASE_DATABASE_VERSION
VALID_VERSIONS = {
    #version:   (Pre-Update Function,  Update Function,    Post-Update Function)
    base   :    (None,                 None,               None                ),
    '1.0.0':    (None,                 update_1_0_0,       None                ),
}
