#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
from os.path import join, dirname, realpath
sys.path.append(realpath(join(dirname(__file__), '../..')))
from ibeis.tests import __testing__
from ibeis.dev import ibsfuncs
# Python
import multiprocessing
import numpy as np
# Tools
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_ENCOUNTERS]')


def TEST_ENCOUNTERS(ibs):
    ibs.compute_encounters()

    eid_list = ibs.get_valid_eids()
    gids_list = ibs.get_encounter_gids(eid_list)
    rids_list = ibs.get_encounter_rids(eid_list)
    nids_list = ibs.get_encounter_nids(eid_list)

    if ibs.get_dbname() == 'testdb1':
        print('testing assertions')
        try:
            assert eid_list  == [1, 2]
            assert gids_list == [[8, 9, 10, 11, 12, 13, 14], [1, 2, 3, 4, 5, 6, 7]]
            assert map(set, rids_list) == map(set, [[10, 11, 3, 13, 12, 14, 4], [1, 5, 2, 7, 8, 9, 6]])
            assert map(set, nids_list) == map(set, [(8, 9, 2, 3, 7), (2, 4, 5, 6)])

            ut1, ut2 = ibsfuncs.unflat_lookup(ibs.get_image_unixtime, gids_list)
            ut1 = np.array(ut1)
            ut2 = np.array(ut2)
            assert(np.all(ut1 > 9000))
            assert(np.all(ut2 < 9000))
        except AssertionError as ex:
            utool.printex(ex, key_list=['eid_list', 'gids_list', 'rids_list', 'nids_list'])
            raise
        print('assertions passed')

    ibs.delete_encounters(eid_list)
    ibs.compute_encounters()
    #ibs.print_egpairs_table()
    #ibs.print_encounter_table()

    gids_list2 = ibsfuncs.unflat_lookup(ibs.get_roi_gids, rids_list)
    assert gids_list2 == map(tuple, gids_list)

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    main_locals = __testing__.main(defaultdb='testdb1')
    ibs = main_locals['ibs']
    test_locals = __testing__.run_test(TEST_ENCOUNTERS, ibs)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
