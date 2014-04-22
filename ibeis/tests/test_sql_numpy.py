#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import __testing__  # Should be imported before any ibeis stuff
import numpy as np
import utool
from ibeis.control import SQLDatabaseControl
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_SQL_NUMPY] ')

printTEST = __testing__.printTEST


def TEST_SQL_NUMPY():
    utool.util_path.remove_file('temp.sqlite3', dryrun=False)

    db = SQLDatabaseControl.SQLDatabaseControl(database_path='.', database_file='temp.sqlite3')

    db.schema('temp',    [
        ('temp_id',      'INTEGER PRIMARY KEY'),
        ('temp_hash',    'NUMPY'),
    ])

    tt = utool.tic()
    feats_list = __testing__.get_test_numpy_data(shape=(3e3, 128), dtype=np.uint8)
    print(' * numpy.new time=%r sec' % utool.toc(tt))

    printTEST('[TEST] insert numpy arrays')
    tt = utool.tic()
    feats_iter = ((feats, ) for feats in feats_list)
    db.executemany(operation='''
        INSERT
        INTO temp
        (
            temp_hash
        )
        VALUES (?)
        ''', params_iter=feats_iter)
    print(' * execute insert time=%r sec' % utool.toc(tt))

    printTEST('[TEST] save sql database')
    tt = utool.tic()
    db.commit()
    print(' * commit time=%r sec' % utool.toc(tt))

    printTEST('[TEST] read from sql database')

    tt = utool.tic()
    db.execute('SELECT temp_hash FROM temp', [])
    print(' * execute select time=%r sec' % utool.toc(tt))

    tt = utool.tic()
    result_list = [result for result in db.result_iter()]
    print(' * iter results time=%r sec' % utool.toc(tt))
    print(' * memory(result_list) = %s' % utool.byte_str2(utool.get_object_size(result_list)))
    del result_list
    #print('[TEST] result_list=%r' % result_list)

    printTEST('[TEST] dump sql database')
    tt = utool.tic()
    db.dump('temp.dump.txt')
    print(' * dump time=%r sec' % utool.toc(tt))
    #with open('temp.dump.txt') as file_:
        #print(file_.read())
    return locals()


if __name__ == '__main__':
    test_locals = __testing__.run_test(TEST_SQL_NUMPY)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
