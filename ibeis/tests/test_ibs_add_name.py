#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_ADD_NAMES]')


def TEST_ADD_NAMES(ibs):
    print('[TEST] GET_TEST_IMAGE_PATHS')
    orig_nids = ibs.get_valid_nids()
    assert len(orig_nids) == 0, 'the database should be empty'
    # The test api returns a list of interesting chip indexes
    name_list = ['____', '06_410', '07_061', '02_044', '07_091', '04_110',
                 '07_233', '07_267', '07_272', '07_300', '04_035', '08_013',
                 '08_016', '02_1110', '08_019', '08_020', '08_038', '01_305',
                 '08_045', '08_051', '01_340', '08_070', '02_168', '08_089',
                 '01_507', '08_106', '09_011', '01_106', '09_013', '09_020',
                 '02_1074', '09_054', '09_148', '09_185', '09_184', '09_196',
                 '01_217', '09_212', '01_461', '09_216', '09_215', '09_341']

    # add a duplicate
    name_list.append(name_list[0])
    try:
        nid_list = ibs.add_names(name_list)
        name_list_test = ibs.get_name_texts(nid_list)
        assert name_list_test == name_list, 'sanity check'
        assert len(name_list) == len(nid_list), 'bad name adder'
        assert nid_list[0] == nid_list[-1], 'first and last names should be the same'
        assert len(list(set(nid_list))) == len(list(set(name_list))), 'num unique ids / names should be the same'
    except AssertionError as ex:
        print('\n\nTEST ERROR')
        utool.printex(ex, 'error in test_ibs_add_name', key_list=locals().keys())
        raise
    return locals()

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For win32
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb_empty', gui=False,
                             allow_newdir=True, delete_ibsdir=True)
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = utool.run_test(TEST_ADD_NAMES, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    execstr += '\n' + utool.ipython_execstr()
    exec(execstr)
