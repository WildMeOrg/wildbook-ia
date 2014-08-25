#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import utool
from ibeis.control import SQLDatabaseControl
from os.path import join
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_SQL_NAMES]')


def __define_schema(db):
    db.add_table('names', (
        ('name_rowid', 'INTEGER PRIMARY KEY'),
        ('name_text',  'TEXT NOT NULL'),
        ('CONSTRAINT superkey UNIQUE (name_text)', '')
    ))


def __insert_names(db, name_list):
    ret = db.executemany(
        operation='''
        INSERT OR IGNORE
        INTO names
        (
            name_rowid,
            name_text
        )
        VALUES (NULL, ?)
        ''',
        params_iter=((name,) for name in name_list))
    print(ret)
    # assert ret == [None] * len(name_list)
    #print('INSERT RETURNED: %r' % ret)


def TEST_SQL_NAMES():
    # -------- INIT DATABASE ------------
    #
    # Create new temp database
    sqldb_fname = 'temp_test_sql_names.sqlite3'
    sqldb_dpath = utool.util_cplat.get_app_resource_dir('ibeis', 'testfiles')
    utool.ensuredir(sqldb_dpath)
    print('Remove Old Temp Database')
    utool.util_path.remove_file(join(sqldb_dpath, sqldb_fname), dryrun=False)
    print('New Temp Database')
    db = SQLDatabaseControl.SQLDatabaseController(sqldb_dpath=sqldb_dpath,
                                                  sqldb_fname=sqldb_fname)
    #
    # Define the schema
    __define_schema(db)
    #
    # -------- RUN INSERTS --------------
    print('[TEST] --- INSERT NAMES --- ')
    test_names = [
        'fred',
        'sue',
        'Robert\');DROP TABLE names;--',
        'joe',
        'rob',
    ]
    __insert_names(db, test_names)
    __insert_names(db, test_names[2:3])
    #
    # -------- RUN SELECT NAMES --------------
    print('[TEST] --- SELECT NAMES ---')
    name_text_results = db.executeone('SELECT name_text FROM names', [])
    print(' * name_text_results=%r' % name_text_results)
    #assert name_text_results == test_names, 'unexpected results from select names'

    return locals()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For windows
    test_locals = utool.run_test(TEST_SQL_NAMES)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
