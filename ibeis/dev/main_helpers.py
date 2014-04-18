from __future__ import absolute_import, division, print_function
import utool
import numpy as np
from ibeis.dev import params
from itertools import izip
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[main_helpers]', DEBUG=False)


def register_utool_aliases():
    #print('REGISTER UTOOL ALIASES')
    import utool
    import matplotlib as mpl
    from ibeis.control.IBEISControl import IBEISControl
    from ibeis.view.guiback import MainWindowBackend
    from ibeis.view.guifront import MainWindowFrontend
    utool.extend_global_aliases([
        (IBEISControl, 'ibs'),
        (MainWindowBackend, 'back'),
        (MainWindowFrontend, 'front'),
        (mpl.figure.Figure, 'fig')
    ])


@utool.indent_decor('[rid_filt]')
def rid_filter(ibs, rid_list, with_hard=True, with_gt=True, with_nogt=True):
    qrid_list = []
    if with_hard:
        notes_list = ibs.get_roi_notes(rid_list)
        qrid_list.extend([rid for (notes, rid) in izip(notes_list, rid_list)
                          if 'hard' in notes.lower().split()])
    if with_gt and not with_nogt:
        gts_list = ibs.get_roi_groundtruth(rid_list)
        qrid_list.extend([rid for (gts, rid) in izip(gts_list, rid_list)
                          if len(gts) > 0])
    if with_gt and with_nogt:
        qrid_list = rid_list
    return qrid_list


@utool.indent_decor('[get_test_rids]')
def get_test_qrids(ibs):
    """ Gets test roi_uids based on command line arguments """
    print('[main_helpers]')
    valid_rids = ibs.get_valid_rids()
    print(utool.dict_str(vars(params.args)))

    # Sample a large pool of query indexes
    histids = None if params.args.histid is None else np.array(params.args.histid)
    if params.args.all_cases:
        print('all cases')
        qrids_all = rid_filter(ibs, valid_rids, with_gt=True, with_nogt=True)
    elif params.args.all_gt_cases:
        print('all gt cases')
        qrids_all = rid_filter(ibs, valid_rids, with_hard=True, with_gt=True, with_nogt=False)
    elif params.args.qrid is None:
        print('did not select cases')
        qrids_all = rid_filter(ibs, valid_rids, with_hard=True, with_gt=False, with_nogt=False)
    else:
        print('Chosen qrid=%r' % params.args.qrid)
        qrids_all = params.args.qrid

    # Filter only the ones you want from the large pool
    if histids is None:
        qrid_list = qrids_all
    else:
        histids = utool.ensure_iterable(histids)
        print('Chosen histids=%r' % histids)
        qrid_list = [qrids_all[id_] for id_ in histids]

    if len(qrid_list) == 0:
        msg = 'no qrid_list history'
        print(msg)
        print('valid_rids = %r' % (valid_rids,))
        qrid_list = valid_rids[0:1]
    print('len(qrid_list) = %d' % len(qrid_list))
    qrid_list = utool.unique_keep_order(qrid_list)
    print('qrid_list = %r' % qrid_list)
    return qrid_list
