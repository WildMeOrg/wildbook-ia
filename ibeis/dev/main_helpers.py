from __future__ import absolute_import, division, print_function
import utool
from ibeis.dev import params
from ibeis.dev import ibsfuncs
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[main_helpers]', DEBUG=False)


def register_utool_aliases():
    #print('REGISTER UTOOL ALIASES')
    import utool
    import matplotlib as mpl
    from ibeis.control import IBEISControl
    from ibeis.gui import guiback
    from ibeis.gui import guifront
    utool.extend_global_aliases([
        (IBEISControl.IBEISController, 'ibs'),
        (guiback.MainWindowBackend, 'back'),
        (guifront.MainWindowFrontend, 'front'),
        (mpl.figure.Figure, 'fig')
    ])


@utool.indent_func
@profile
def get_test_qrids(ibs):
    """ Gets test roi_uids based on command line arguments """
    #print('[main_helpers]')
    test_qrids = []
    valid_rids = ibs.get_valid_rids()
    printDBG('1. valid_rids = %r' % valid_rids[0:5])
    #print(utool.dict_str(vars(params.args)))

    if params.args.qrid is not None:
        printDBG('Testing qrid=%r' % params.args.qrid)
        test_qrids.extend(params.args.qrid)

    if params.args.all_cases:
        printDBG('Testing all %d cases' % (len(valid_rids),))
        printDBG('1. test_qrids = %r' % test_qrids[0:5])
        test_qrids.extend(valid_rids)
        printDBG('2. test_qrids = %r' % test_qrids[0:5])
    else:
        is_hard_list = ibsfuncs.get_roi_is_hard(ibs, valid_rids)
        hard_rids = utool.filter_items(valid_rids, is_hard_list)
        printDBG('Testing %d known hard cases' % len(hard_rids))
        test_qrids.extend(hard_rids)

    if params.args.all_gt_cases:
        has_gt_list = ibs.get_roi_has_groundtruth(valid_rids)
        hasgt_rids = utool.filter_items(valid_rids, has_gt_list)
        print('Testing all %d ground-truthed cases' % len(hasgt_rids))
        test_qrids.extend(hasgt_rids)

    # Sample a large pool of query indexes
    # Filter only the ones you want from the large pool
    if params.args.index is not None:
        indexes = utool.ensure_iterable(params.args.index)
        #printDBG('Chosen indexes=%r' % (indexes,))
        #printDBG('test_qrids = %r' % test_qrids[0:5])
        _test_qrids = [test_qrids[xx] for xx in indexes]
        test_qrids = _test_qrids
        #printDBG('test_qrids = %r' % test_qrids)
    elif len(test_qrids) == 0 and len(valid_rids) > 0:
        #printDBG('no hard or gt rids. Defaulting to the first ROI')
        test_qrids = valid_rids[0:1]

    #print('test_qrids = %r' % test_qrids)
    test_qrids = utool.unique_keep_order2(test_qrids)
    return test_qrids
