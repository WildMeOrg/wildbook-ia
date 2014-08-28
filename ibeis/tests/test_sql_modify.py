#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import utool
from ibeis.control import SQLDatabaseControl
from os.path import join
from functools import partial
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_SQL_CONTROL]')
import six
import random

if six.PY2:
    __STR__ = unicode
else:
    __STR__ = str

###########################


def converter(val):
    return str(val) + '_str'


def get_rowid_from_text(db, text_list):
    param_iter = ((text,) for text in text_list)
    return db.get_rowid_from_superkey('test', param_iter, superkey_colnames=('test_text',))


def add_text(db, text_list):
    param_iter = ((text,) for text in text_list)
    func = partial(get_rowid_from_text, db)
    return db.add_cleanly('test', ('test_text',), param_iter, func)


def get_text(db, tablename, rowid_list):
    return db.get(tablename, ('test_text',), rowid_list)


def set_integers(db, tablename, integer_list, rowid_list):
    param_iter = ((integer,) for integer in integer_list)
    return db.set(tablename, ('test_integer',), param_iter, rowid_list)


def get_integers(db, tablename, rowid_list):
    return db.get(tablename, ('test_integer',), rowid_list)


def get_integers2(db, tablename, rowid_list):
    return db.get(tablename, ('test_integer2',), rowid_list)


###########################


def add_table(db):
    db.add_table('test', (
        ('test_rowid', 'INTEGER PRIMARY KEY'),
        ('test_text',  'TEXT NOT NULL'),),
        ['CONSTRAINT superkey UNIQUE (test_text)']
    )


def add_column(db):
    db.add_column('test', 'test_integer', 'INTEGER')


def modify_table(db):
    db.modify_table('test', (
        ('test_integer', '', 'TEXT', converter),
    ))

def reorder_columns(db, order_list):
    db.reorder_columns('test', order_list)


def duplicate_table(db):
    db.duplicate_table('test', 'test2')


def duplicate_column(db):
    db.duplicate_column('test2', 'test_integer', 'test_integer2')


def rename_table(db):
    db.rename_table('test2', 'test3')


def rename_column(db):
    db.rename_column('test3', 'test_integer2', 'test_integer3')


def drop_table(db):
    db.drop_table('test3')


def drop_column(db):
    db.drop_column('test3', 'test_integer3')


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
    db = _make_empty_controller()
    
    #####################
    # Add table
    #####################
    add_table(db)

    # Verify table's schema
    colname_list = db.get_column_names('test')
    coltype_list = db.get_column_types('test')   
    assert colname_list == ['test_rowid', 'test_text'], 'Actual values: %r ' % colname_list
    assert coltype_list == ['INTEGER PRIMARY KEY', 'TEXT NOT NULL'], 'Actual values: %r ' % coltype_list

    #####################
    # Add data to table
    #####################
    text_list = [ utool.random_nonce(8) for _ in range(10) ]
    rowid_list = add_text(db, text_list)
    text_list_ = get_text(db, 'test', rowid_list)
    assert text_list == text_list_, 'Actual values: %r ' % text_list_

    #####################
    # Add column
    #####################
    add_column(db)

    # Verify table's schema
    colname_list = db.get_column_names('test')
    coltype_list = db.get_column_types('test')   
    assert colname_list == ['test_rowid', 'test_text', 'test_integer'], 'Actual values: %r ' % colname_list
    assert coltype_list == [u'INTEGER PRIMARY KEY', u'TEXT NOT NULL', u'INTEGER'], 'Actual values: %r ' % coltype_list

    integer_list = [ random.randint(0, 100) for _ in range(10) ]
    set_integers(db, 'test', integer_list, rowid_list)
    integer_list_ = get_integers(db, 'test', rowid_list)
    assert integer_list == integer_list_, 'Actual values: %r ' % integer_list_

    #####################
    # Modify table
    #####################
    modify_table(db)

    # Verify table's schema
    colname_list = db.get_column_names('test')
    coltype_list = db.get_column_types('test')   
    assert colname_list == ['test_rowid', 'test_text', 'test_integer'], 'Actual values: %r ' % colname_list
    assert coltype_list == [u'INTEGER PRIMARY KEY', u'TEXT NOT NULL', u'TEXT'], 'Actual values: %r ' % coltype_list

    integer_list = get_integers(db, 'test', rowid_list) 
    assert integer_list == [ converter(integer) for integer in integer_list_ ], 'Actual values: %r ' % integer_list

    #####################
    # Duplicate table
    #####################
    duplicate_table(db)

    # Verify table's schema
    colname_list = db.get_column_names('test2')
    coltype_list = db.get_column_types('test2')   
    assert colname_list == ['test_rowid', 'test_text', 'test_integer'], 'Actual values: %r ' % colname_list
    assert coltype_list == ['INTEGER PRIMARY KEY', 'TEXT NOT NULL', 'TEXT'], 'Actual values: %r ' % coltype_list
    
    text_list_ = get_text(db, 'test2', rowid_list)
    assert text_list == text_list_, 'Actual values: %r ' % text_list_
    integer_list_ = get_integers(db, 'test2', rowid_list)
    assert integer_list == integer_list_, 'Actual values: %r ' % integer_list_

    #####################
    # Duplicate column
    #####################
    duplicate_column(db)

    # Verify table's schema
    colname_list = db.get_column_names('test2')
    coltype_list = db.get_column_types('test2')   
    assert colname_list == ['test_rowid', 'test_text', 'test_integer', 'test_integer2'], 'Actual values: %r ' % colname_list
    assert coltype_list == [u'INTEGER PRIMARY KEY', u'TEXT NOT NULL', u'TEXT', u'TEXT'], 'Actual values: %r ' % coltype_list

    integer_list = get_integers(db, 'test2', rowid_list)
    integer2_list = get_integers2(db, 'test2', rowid_list)
    assert integer_list == integer2_list

    #####################
    # Rename table
    #####################
    rename_table(db)

    tablename_list = db.get_table_names()
    assert tablename_list == ['metadata', 'test', 'test3'], 'Actual values: %r ' % tablename_list

    #####################
    # Rename column
    #####################
    rename_column(db)

    # Verify table's schema
    colname_list = db.get_column_names('test3')
    coltype_list = db.get_column_types('test3')   
    assert colname_list == ['test_rowid', 'test_text', 'test_integer', 'test_integer3'], 'Actual values: %r ' % colname_list
    assert coltype_list == [u'INTEGER PRIMARY KEY', u'TEXT NOT NULL', u'TEXT', u'TEXT'], 'Actual values: %r ' % coltype_list

    #########################
    # Reorder table's columns
    #########################
    colname_original_list = db.get_column_names('test')
    coltype_original_list = db.get_column_types('test')   

    order_list = range(len(colname_original_list))
    while order_list == sorted(order_list):
        random.shuffle(order_list)
    reorder_columns(db, order_list)

    # Verify table's schema
    colname_list_ = db.get_column_names('test')
    coltype_list_ = db.get_column_types('test')   

    # Find correct new order
    combined = sorted(list(zip(order_list, colname_original_list, coltype_original_list)))
    colname_list__ = [ name for i, name, type_ in combined ]
    coltype_list__ = [ type_ for i, name, type_ in combined ]

    assert colname_list_ == colname_list__, 'Actual values: %r ' % colname_list_
    assert coltype_list_ == coltype_list__, 'Actual values: %r ' % coltype_list_
    
    #####################
    # Drop column
    #####################
    drop_column(db)

    # Verify table's schema
    colname_list = db.get_column_names('test3')
    coltype_list = db.get_column_types('test3')   
    assert colname_list == ['test_rowid', 'test_text', 'test_integer'], 'Actual values: %r ' % colname_list
    assert coltype_list == [u'INTEGER PRIMARY KEY', u'TEXT NOT NULL', u'TEXT'], 'Actual values: %r ' % coltype_list

    #####################
    # Drop table
    #####################
    drop_table(db)

    tablename_list = db.get_table_names()
    assert tablename_list == ['metadata', 'test'], 'Actual values: %r ' % tablename_list

    return locals()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For windows
    test_locals = utool.run_test(TEST_SQL_MODIFY)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
