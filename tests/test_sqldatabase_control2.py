#!usr/bin/env python
from __future__ import division, print_function
#------
TEST_NAME = 'TEST_SQL_CONTROL2'
#------
import __testing__  # Should be imported before any ibeis stuff
import sys
import utool
from ibeis.control import SQLDatabaseControl
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)

printTEST = __testing__.printTEST


@__testing__.testcontext
def TEST_SQL_CONTROL2():

    utool.util_path.remove_file('temp.sqlite3', dryrun=False)

    db = SQLDatabaseControl.SQLDatabaseControl(database_path='.', database_file='temp.sqlite3')

    NAME_UID_TYPE = 'INTEGER'
    db.schema('names', (
        ('name_uid',                     '%s PRIMARY KEY' % NAME_UID_TYPE),
        ('name_text',                    'TEXT NOT NULL'),
    ))

    test_names = [
        'fred',
        'sue',
        'joe',
        'rob',
        'Robert\');DROP TABLE Students;--',
    ]
    # -------- RUN INSERTS --------------
    db.executemany(
        operation='''
        INSERT
        INTO names
        (
            name_text
        )
        VALUES (?)
        ''',
        parameters_iter=((name,) for name in test_names))

    printTEST('[TEST] save sql database')
    db.commit()


    # -------- RUN SELECTIONS --------------

    printTEST('[TEST] --- READ NAMES ---')
    db.execute('SELECT name_text FROM names', [])
    name_text_results = db.result_list()
    print('[TEST] name_text_results=%r' % name_text_results)
    assert name_text_results == test_names, 'unexpected results from select names'

    printTEST('[TEST] --- READ NIDS ---')
    query_names = test_names[::2] + ['missingno']
    nid_list = db.executemany(
        operation='''
               SELECT name_uid
               FROM names
               WHERE name_text=?
               ''',
        parameters_iter=((name,) for name in query_names))

    # Get the parameter indexes that failed
    failx_list = [count for count, nid in enumerate(nid_list) if nid == []]
    assert failx_list == [3]
    failed_names = [query_names[failx] for failx in failx_list]
    utool.printvar2('failed_names')

    # We selected a name not in the table.
    # Its return index is an empty list
    print('[TEST] name_text_results=%r' % nid_list)
    print('[TEST] name_text_results=%r' % query_names)
    print('[TEST] name_text_results=%r' % test_names)
    # SQL INTEGERS START AT 1 APPARENTLY
    expected_names = [test_names[nid - 1] for nid in nid_list]
    assert expected_names == query_names, 'unexpected results from select names'

    printTEST('[TEST] dump sql database')
    db.dump('temp.dump.txt')
    db.dump(sys.stdout)
TEST_SQL_CONTROL2.func_name = TEST_NAME


if __name__ == '__main__':
    TEST_SQL_CONTROL2()
