#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
# Python
import multiprocessing
# Tools
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_HOTS_SPLITTREE]')


def TEST_HOTS_SPLITTREE(ibs):
    from ibeis.model.hots.hots_nn_index import HOTSSplitIndex
    num_forests = 8
    daid_list = ibs.get_valid_aids()
    num_neighbors = 3
    qfx2_desc = ibs.get_annot_desc(daid_list[2])

    split_index = HOTSSplitIndex(ibs, daid_list, num_forests=num_forests)

    #nid_list  = ibs.get_annot_nids(aid_list)
    ##flag_list = ibs.get_annot_exemplar_flag(aid_list)
    #nid2_aids = utool.group_items(aid_list, nid_list)
    #key_list = nid2_aids.keys()
    #val_list = nid2_aids.values()
    #isunknown_list = ibs.is_nid_unknown(key_list)

    #num_forests = 8
    ## Put one name per forest
    #forest_aids, overflow_aids = utool.sample_zip(val_list, num_forests, allow_overflow=True)

    #forest_indexes = []
    #for tx, aids in enumerate(forest_aids):
    #    print('[nn] building forest %d/%d' % (tx + 1, num_forests))
    #    nn_index = HOTSIndex(ibs, aids)
    #    forest_indexes.append(nn_index)

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For win32
    import ibeis
    main_locals = ibeis.main(defaultdb='GZ_ALL', gui=False)
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = utool.run_test(TEST_HOTS_SPLITTREE, ibs)
    execstr = utool.execstr_dict(test_locals, 'testdb1')
    exec(execstr)
