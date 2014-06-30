#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
import ibeis
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_GUI_SELECTION]')


def TEST_GUI_SELECTION(ibs, back):
    print('''
              get_valid_gids
              ''')
    valid_gids = ibs.get_valid_gids()
    print('''
              get_valid_aids
              ''')
    valid_aids = ibs.get_valid_aids()

    print('''
    * len(valid_aids) = %r
    * len(valid_gids) = %r
    ''' % (len(valid_aids), len(valid_gids)))
    assert len(valid_gids) > 0, 'database images cannot be empty for test'

    gid = valid_gids[0]
    aid_list = ibs.get_image_aids(gid)
    aid = aid_list[-1]
    back.select_gid(gid, aids=[aid])

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = ibeis.main(defaultdb='testdb0', gui=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_GUI_SELECTION, ibs, back)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
