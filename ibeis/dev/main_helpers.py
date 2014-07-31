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
def get_test_qaids(ibs):
    """ Gets test annotation_rowids based on command line arguments """
    #print('[main_helpers]')
    test_qaids = []
    valid_aids = ibs.get_valid_aids()
    printDBG('1. valid_aids = %r' % valid_aids[0:5])
    #print(utool.dict_str(vars(params.args)))

    if params.args.qaid is not None:
        printDBG('Testing qaid=%r' % params.args.qaid)
        test_qaids.extend(params.args.qaid)

    if params.args.all_cases:
        printDBG('Testing all %d cases' % (len(valid_aids),))
        printDBG('1. test_qaids = %r' % test_qaids[0:5])
        test_qaids.extend(valid_aids)
        printDBG('2. test_qaids = %r' % test_qaids[0:5])
    else:
        is_hard_list = ibsfuncs.get_annot_is_hard(ibs, valid_aids)
        hard_aids = utool.filter_items(valid_aids, is_hard_list)
        printDBG('Testing %d known hard cases' % len(hard_aids))
        test_qaids.extend(hard_aids)

    if params.args.all_gt_cases:
        has_gt_list = ibs.get_annot_has_groundtruth(valid_aids)
        hasgt_aids = utool.filter_items(valid_aids, has_gt_list)
        print('Testing all %d ground-truthed cases' % len(hasgt_aids))
        test_qaids.extend(hasgt_aids)

    # Sample a large pool of query indexes
    # Filter only the ones you want from the large pool
    if params.args.index is not None:
        indexes = utool.ensure_iterable(params.args.index)
        #printDBG('Chosen indexes=%r' % (indexes,))
        #printDBG('test_qaids = %r' % test_qaids[0:5])
        _test_qaids = [test_qaids[xx] for xx in indexes]
        test_qaids = _test_qaids
        #printDBG('test_qaids = %r' % test_qaids)
    elif len(test_qaids) == 0 and len(valid_aids) > 0:
        #printDBG('no hard or gt aids. Defaulting to the first ANNOTATION')
        test_qaids = valid_aids[0:1]

    #print('test_qaids = %r' % test_qaids)
    test_qaids = utool.unique_keep_order2(test_qaids)
    return test_qaids
