#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import numpy as np
import utool
from ibeis.control import SQLDatabaseControl
from os.path import join
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_SQL_NUMPY] ')


# list of 10,000 chips with 3,000 features apeice.
def grab_numpy_testdata(shape=(3e3, 128), dtype=np.uint8):
    ndata = utool.get_arg('--ndata', type_=int, default=2)
    print('[TEST] build ndata=%d numpy arrays with shape=%r' % (ndata, shape))
    print(' * expected_memory(table_list) = %s' % utool.byte_str2(ndata * np.product(shape)))
    table_list = [np.empty(shape, dtype=dtype) for i in xrange(ndata)]
    print(' * memory+overhead(table_list) = %s' % utool.byte_str2(utool.get_object_size(table_list)))
    return table_list


def TEST_SQL_NUMPY():
    sqldb_fname = 'temp_test_sql_numpy.sqlite3'
    sqldb_dpath = utool.util_cplat.get_project_resource_dir('ibeis', 'testfiles')
    utool.ensuredir(sqldb_dpath)
    utool.util_path.remove_file(join(sqldb_dpath, sqldb_fname), dryrun=False)
    db = SQLDatabaseControl.SQLDatabaseController(sqldb_dpath=sqldb_dpath,
                                                  sqldb_fname=sqldb_fname)

    db.schema('temp',    [
        ('temp_id',      'INTEGER PRIMARY KEY'),
        ('temp_hash',    'NUMPY'),
    ])

    tt = utool.tic()
    feats_list = grab_numpy_testdata(shape=(3e3, 128), dtype=np.uint8)
    print(' * numpy.new time=%r sec' % utool.toc(tt))

    print('[TEST] insert numpy arrays')
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

    print('[TEST] save sql database')
    tt = utool.tic()
    db.commit()
    print(' * commit time=%r sec' % utool.toc(tt))

    print('[TEST] read from sql database')

    tt = utool.tic()
    db.execute('SELECT temp_hash FROM temp', [])
    print(' * execute select time=%r sec' % utool.toc(tt))

    tt = utool.tic()
    result_list = [result for result in db.result_iter()]
    print(' * iter results time=%r sec' % utool.toc(tt))
    print(' * memory(result_list) = %s' % utool.byte_str2(utool.get_object_size(result_list)))
    del result_list
    #print('[TEST] result_list=%r' % result_list)

    print('[TEST] dump sql database')
    tt = utool.tic()
    db.dump('temp.dump.txt')
    print(' * dump time=%r sec' % utool.toc(tt))
    #with open('temp.dump.txt') as file_:
    #    print(file_.read())
    return locals()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For windows
    from ibeis.tests import __testing__
    test_locals = __testing__.run_test(TEST_SQL_NUMPY)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
