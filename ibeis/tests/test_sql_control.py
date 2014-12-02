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


def _define_test_schema(db):
    db.add_table('names', (
        ('name_rowid', 'INTEGER PRIMARY KEY'),
        ('name_text',  'TEXT NOT NULL'),),
        ['CONSTRAINT superkey UNIQUE (name_text)']
    )


def _make_empty_controller():
    print('make_empty_controller')
    sqldb_fname = 'temp_test_sql_control.sqlite3'
    sqldb_dpath = utool.util_cplat.get_app_resource_dir('ibeis', 'testfiles')
    utool.ensuredir(sqldb_dpath)
    utool.remove_file(join(sqldb_dpath, sqldb_fname), dryrun=False)
    db = SQLDatabaseControl.SQLDatabaseController(sqldb_dpath=sqldb_dpath,
                                                  sqldb_fname=sqldb_fname)
    return db


def TEST_SQL_CONTROL():
    db = _make_empty_controller()
    _define_test_schema(db)
    #
    # -------- RUN INSERTS --------------
    print('[TEST] --- INSERT NAMES --- ')
    target_name_texts_list = list(map(__STR__, [
        'fred',
        'sue',
        'Robert\');DROP TABLE names;--',
        'joe',
        'rob',
    ]))
    target_name_texts_sublist = target_name_texts_list[2:]

    def get_nameid_from_text(text_list):
        param_iter = ((text,) for text in text_list)
        return db.get_rowid_from_superkey('names', param_iter, superkey_colnames=('name_text',))

    def add_names(names_list):
        param_iter = ((text,) for text in names_list)
        return db.add_cleanly('names', ('name_text',), param_iter, get_nameid_from_text)

    def get_name_texts(rowid_list):
        return db.get('names', ('name_text',), rowid_list)

    def set_names(rowid_list, name_list):
        return db.set('names', ('name_text',), name_list, rowid_list)

    rowid_list = add_names(target_name_texts_list)
    rowid_sublist = add_names(target_name_texts_sublist)
    rowid_sublist1 = add_names(target_name_texts_sublist)
    rowid_list1 = add_names(target_name_texts_list)
    assert rowid_sublist1 == rowid_sublist
    assert rowid_list1 == rowid_list
    #
    # -------- RUN SELECT NAMES --------------
    print('[TEST] --- SELECT NAMES ---')
    test_names_list = get_name_texts(rowid_list)
    test_names_sublist = get_name_texts(rowid_sublist)
    assert test_names_list == target_name_texts_list
    assert test_names_sublist == target_name_texts_sublist

    #
    # --- TEST SETTER ---

    name_sublist_new = ['newset_' + name for name in target_name_texts_sublist]

    set_names(rowid_sublist, name_sublist_new)
    name_sublist_new_test  = get_name_texts(rowid_sublist)
    assert name_sublist_new_test == name_sublist_new
    name_sublist_old_fails = get_nameid_from_text(target_name_texts_sublist)
    assert all([rowid is None for rowid in name_sublist_old_fails])
    assert name_sublist_new_test == name_sublist_new

    new_rowid_sublist = add_names(name_sublist_new)
    assert new_rowid_sublist == rowid_sublist
    # Because the old are no longer in the database these should be newids
    old_rowid_sublist = add_names(target_name_texts_sublist)
    assert len(set(old_rowid_sublist) & set(new_rowid_sublist)) == 0

    csv_target = u'''
        1,                                 fred
        2,                                  sue
        3,  newset_Robert');DROP TABLE names;--
        4,                           newset_joe
        5,                           newset_rob
        6,         Robert');DROP TABLE names;--
        7,                                  joe
        8,                                  rob
    '''.strip()

    csv_test = db.get_table_csv('names')
    print('test=')
    print(csv_test)

    try:
        assert csv_test.endswith(csv_target)
    except AssertionError as ex:
        print('target=')
        print(csv_target)
        utool.printex(ex)
        raise

    return locals()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For windows
    test_locals = utool.run_test(TEST_SQL_CONTROL)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
