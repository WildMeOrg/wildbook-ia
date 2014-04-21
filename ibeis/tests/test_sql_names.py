#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
#------
TEST_NAME = 'TEST_SQL_NAMES'
#------
import __testing__  # Should be imported before any ibeis stuff
import utool
from ibeis.control import SQLDatabaseControl
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)

printTEST = __testing__.printTEST


def __define_schema(db):
    NAME_UID_TYPE = 'INTEGER'
    db.schema('names', (
        ('name_uid',   '%s PRIMARY KEY' % NAME_UID_TYPE),
        ('name_text',  'TEXT NOT NULL'),
        ('CONSTRAINT superkey UNIQUE (name_text)', '')
    ))


def __insert_names(db, name_list):
    ret = db.executemany(
        operation='''
        INSERT OR IGNORE
        INTO names
        (
            name_uid,
            name_text
        )
        VALUES (NULL, ?)
        ''',
        params_iter=((name,) for name in name_list))
    assert ret == [None] * len(name_list)
    #print('INSERT RETURNED: %r' % ret)


@__testing__.testcontext2(TEST_NAME)
def TEST_SQL_NAMES():

    # -------- INIT DATABASE ------------
    printTEST('Remove Old Temp Database')
    utool.util_path.remove_file('temp.sqlite3', dryrun=False)
    #
    # Create new temp database
    printTEST('New Temp Database')
    db = SQLDatabaseControl.SQLDatabaseControl(database_path='.', database_file='temp.sqlite3')
    #
    # Define the schema
    __define_schema(db)
    #
    # -------- RUN INSERTS --------------
    printTEST('[TEST] --- INSERT NAMES --- ')
    test_names = [
        'fred',
        'sue',
        'Robert\');DROP TABLE Students;--',
        'joe',
        'rob',
    ]
    __insert_names(db, test_names)
    __insert_names(db, test_names[2:3])
    #
    # -------- RUN SELECT NAMES --------------
    printTEST('[TEST] --- SELECT NAMES ---')
    name_text_results = db.executeone('SELECT name_text FROM names', [])
    print(' * name_text_results=%r' % name_text_results)
    #assert name_text_results == test_names, 'unexpected results from select names'
    #
    # -------- RUN SELECT NIDS --------------
    printTEST('[TEST] --- SELECT NIDS ---')
    query_names = test_names[::2] + ['missingno']
    nid_list = db.executemany(
        operation='''
               SELECT name_uid
               FROM names
               WHERE name_text=?
               ''',
        params_iter=((name,) for name in query_names))

    # Get the parameter indexes that failed
    failx_list = [count for count, nid in enumerate(nid_list) if nid is None]
    assert failx_list == [3]
    failed_names = [query_names[failx] for failx in failx_list]  # NOQA
    utool.printvar2('failed_names')

    # We selected a name not in the table.
    # Its return index is an empty list
    print('[TEST] nid_list=%r' % nid_list)
    print('[TEST] query_names=%r' % query_names)
    print('[TEST] test_names=%r' % test_names)
    # SQL INTEGERS START AT 1 APPARENTLY
    #expected_names = [test_names[nid - 1] for nid in nid_list]
    #assert expected_names == query_names, 'unexpected results from select names'
    return locals()


if __name__ == '__main__':
    test_locals = TEST_SQL_NAMES()
    exec(test_locals['execstr'])
