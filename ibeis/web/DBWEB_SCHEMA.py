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
        ('annot_rowid',                  'INTEGER'),
        ('viewpoint_value_1',            'INTEGER DEFAULT -1'),
        ('viewpoint_value_2',            'INTEGER DEFAULT -1'),
        ('viewpoint_value_avg',          'INTEGER DEFAULT -1'),
    ),
        superkey_colnames=['annot_rowid'],
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
