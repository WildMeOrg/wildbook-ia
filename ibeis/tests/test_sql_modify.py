#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import utool
from ibeis.control import SQLDatabaseControl
from os.path import join
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_SQL_CONTROL]')
import six

if six.PY2:
    __STR__ = unicode
else:
    __STR__ = str


def add_table(db):
    db.add_table('names', (
        ('name_rowid', 'INTEGER PRIMARY KEY'),
        ('name_text',  'TEXT NOT NULL'),),
        ['CONSTRAINT superkey UNIQUE (name_text)']
    )


def add_column(db):
    db.add_column('names', 'name_integer', 'INTEGER')


def modify_table(db):
    def converter(val):
        return str(val) + '_str'
    ibs.db.modify_table(constants.CONTRIBUTOR_TABLE, (
        ('name_integer', '', 'TEXT', converter),
    ))


def duplicate_table(db):
    db.duplicate_table('names', 'names2')


def duplicate_column(db):
    db.duplicate_column('names', 'name_text', 'name_text2')


def rename_table(db):
    db.rename_table('names2', 'names3')


def rename_column(db):
    db.rename_column('names', 'name_text2', 'name_text3')


def drop_table(db):
    db.drop_table('names3')


def drop_column(db):
    db.drop_column('names', 'name_text3')


def _make_empty_controller():
    print('make_empty_controller')
    sqldb_fname = 'temp_test_sql_control.sqlite3'
    sqldb_dpath = utool.util_cplat.get_app_resource_dir('ibeis', 'testfiles')
    utool.ensuredir(sqldb_dpath)
    utool.remove_file(join(sqldb_dpath, sqldb_fname), dryrun=False)
    db = SQLDatabaseControl.SQLDatabaseController(sqldb_dpath=sqldb_dpath,
                                                  sqldb_fname=sqldb_fname)
    return db


def TEST_SQL_MODIFY():
    def get_rowid_from_text(text_list):
        param_iter = ((text,) for text in text_list)
        return db.get_rowid_from_superkey('names', param_iter, superkey_colnames=('name_text',))

    def add_text(text_list):
        param_iter = ((text,) for text in text_list)
        return db.add_cleanly('names', ('name_text',), param_iter, get_rowid_from_text)

    def get_text(rowid_list):
        return db.get('names', ('name_text',), rowid_list)

    db = _make_empty_controller()
    
    #####################
    # Add table
    #####################
    add_table(db)

    # Verify table's schema,    

    #####################
    # Add data to table
    #####################
    print('[TEST] --- INSERT TEXT --- ')
    text_list = [ utool.random_nonce(8) for _ in range(10) ]
    print(text_list)
    rowid_list = add_text(text_list)
    print(rowid_list)
    text_list_ = get_text(rowid_list)
    print(text_list_)
    print(db.dump_to_string())
    

    return locals()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For windows
    test_locals = utool.run_test(TEST_SQL_MODIFY)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
