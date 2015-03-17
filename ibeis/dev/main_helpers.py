"""
TODO: Rename to ibeis/init/commands.py
"""
from __future__ import absolute_import, division, print_function
import utool as ut
from ibeis import params
# Inject utool functions
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[main_helpers]')


def register_utool_aliases():
    """
    registers commmon class names with utool so they are printed nicely
    """
    #print('REGISTER UTOOL ALIASES')
    import utool as ut
    import matplotlib as mpl
    from ibeis.control import IBEISControl, SQLDatabaseControl
    from ibeis.gui import guiback
    #from ibeis.gui import guifront
    ut.extend_global_aliases([
        (SQLDatabaseControl.SQLDatabaseController, 'sqldb'),
        (IBEISControl.IBEISController, 'ibs'),
        (guiback.MainWindowBackend, 'back'),
        #(guifront.MainWindowFrontend, 'front'),
        (mpl.figure.Figure, 'fig')
    ])


def ensure_flatiterable(input_):
    if isinstance(input_, int) or not ut.isiterable(input_):
        return [input_]
    elif isinstance(input_, list):
        #print(input_)
        if len(input_) > 0 and ut.isiterable(input_[0]):
            return ut.flatten(input_)
        return input_
    else:
        raise TypeError('cannot ensure %r input_=%r is iterable', (type(input_), input_))


def ensure_flatlistlike(input_):
    #if isinstance(input_, slice):
    #    pass
    iter_ = ensure_flatiterable(input_)
    return list(iter_)


#@ut.indent_func
@profile
def get_test_qaids(ibs, default_qaids=None):
    """ Gets test annot_rowids based on command line arguments """
    #def get_allowed_qaids(ibs):
    available_qaids = []

    # Currently the only avaialable query annots are ones specified on the
    # command line
    if params.args.qaid is not None:
        try:
            args_qaid = ensure_flatlistlike(params.args.qaid)
        except Exception:
            args_qaid = params.args.qaid
        #if __debug__:
        #    printDBG('Testing qaid=%r' % params.args.qaid)
        available_qaids.extend(args_qaid)

    valid_aids = ibs.get_valid_aids(nojunk=ut.get_argflag('--junk'))
    if valid_aids == 0:
        print('[get_test_qaids] no annotations available')

    # ALL CASES
    if params.args.all_cases:
        available_qaids.extend(valid_aids)

    # HARD CASES
    if params.args.all_hard_cases or ut.get_argflag('--hard'):
        is_hard_list = ibs.get_annot_is_hard(valid_aids)
        hard_aids = ut.filter_items(valid_aids, is_hard_list)
        available_qaids.extend(hard_aids)

    # GROUNDTRUTH CASES
    if params.args.all_gt_cases:
        has_gt_list = ibs.get_annot_has_groundtruth(valid_aids)
        hasgt_aids = ut.filter_items(valid_aids, has_gt_list)
        print('[get_test_qaids] Adding all %d/%d ground-truthed test cases' % (len(hasgt_aids), len(valid_aids)))
        available_qaids.extend(hasgt_aids)

    # INDEX SUBSET
    # Sample a large pool of chosen query qindexes
    # Filter only the ones you want from the large pool
    if params.args.qindex is not None:
        # FIXME: should use a slice of the list or a sublist
        qindexes = ensure_flatlistlike(params.args.qindex)
        #printDBG('Chosen qindexes=%r' % (qindexes,))
        #printDBG('available_qaids = %r' % available_qaids[0:5])
        _test_qaids = [available_qaids[qx] for qx in qindexes if qx < len(available_qaids)]
        print('[get_test_qaids] Chose subset of size %d/%d' % (len(_test_qaids), len(available_qaids)))
        available_qaids = _test_qaids
        #printDBG('available_qaids = %r' % available_qaids)
    # DEFAULT [0]
    elif len(available_qaids) == 0 and len(valid_aids) > 0:
        print('[get_test_qaids] no hard or gt aids. Defaulting to the first ANNOTATION')
        #available_qaids = valid_aids[0:1]
        available_qaids = default_qaids
    if available_qaids is None:
        available_qaids = [1]

    if not ut.get_argflag('--junk'):
        available_qaids = ibs.filter_junk_annotations(available_qaids)

    #print('available_qaids = %r' % available_qaids)
    available_qaids = ut.unique_keep_order2(available_qaids)
    return available_qaids


@ut.indent_func
@profile
def get_test_daids(ibs, qaid_list=None):
    """ Gets database annot_rowids based on command line arguments


    CommandLine:
        python dev.py --db PZ_MTEST -t best --exclude-query --qaid 72 -r 0 -c 0 --show --va --vf --dump-extra

    """
    available_daids = ibs.get_valid_aids(nojunk=ut.get_argflag('--junk'))

    if ut.get_argflag('--exclude-singleton'):
        raise NotImplementedError('')
        # singleton_aids =
        #available_daids = list(set(available_daids) - set(singleton_aids))

    if ut.get_argflag('--exclude-query') and qaid_list is not None:
        available_daids = list(set(available_daids) - set(qaid_list))

    if params.args.daid_exclude is not None:
        available_daids = list(set(available_daids) - set(params.args.daid_exclude))

    if params.args.dindex is not None:
        dindexes = ensure_flatlistlike(params.args.dindex)
        #printDBG('Chosen dindexes=%r' % (dindexes,))
        #printDBG('available_daids = %r' % available_daids[0:5])
        _test_daids = [available_daids[dx] for dx in dindexes if dx < len(available_daids)]
        available_daids = _test_daids
        #printDBG('available_daids = %r' % available_daids)

    return available_daids
