"""
TODO: Rename to ibeis/init/commands.py
"""
from __future__ import absolute_import, division, print_function
import utool
from ibeis import params
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[main_helpers]')


def register_utool_aliases():
    """
    registers commmon class names with utool so they are printed nicely
    """
    #print('REGISTER UTOOL ALIASES')
    import utool
    import matplotlib as mpl
    from ibeis.control import IBEISControl, SQLDatabaseControl
    from ibeis.gui import guiback
    #from ibeis.gui import guifront
    utool.extend_global_aliases([
        (SQLDatabaseControl.SQLDatabaseController, 'sqldb'),
        (IBEISControl.IBEISController, 'ibs'),
        (guiback.MainWindowBackend, 'back'),
        #(guifront.MainWindowFrontend, 'front'),
        (mpl.figure.Figure, 'fig')
    ])


def ensure_flatiterable(input_):
    if isinstance(input_, int) or not utool.isiterable(input_):
        return [input_]
    elif isinstance(input_, list):
        print(input_)
        if len(input_) > 0 and utool.isiterable(input_[0]):
            return utool.flatten(input_)
        return input_
    else:
        raise TypeError('cannot ensure %r input_=%r is iterable', (type(input_), input_))


def ensure_flatlistlike(input_):
    iter_ = ensure_flatiterable(input_)
    return list(iter_)


@utool.indent_func
@profile
def get_test_qaids(ibs):
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
        if __debug__:
            printDBG('Testing qaid=%r' % params.args.qaid)
        available_qaids.extend(args_qaid)

    valid_aids_ = utool.lazyfunc(ibs.get_valid_aids)
    if valid_aids_() == 0:
        print('no annotations available')

    if params.args.all_cases:
        available_qaids.extend(valid_aids_())
    if not params.args.all_cases or utool.get_argflag('--hard'):
        is_hard_list = ibs.get_annot_is_hard(valid_aids_())
        hard_aids = utool.filter_items(valid_aids_(), is_hard_list)
        available_qaids.extend(hard_aids)

    if params.args.all_gt_cases:
        has_gt_list = ibs.get_annot_has_groundtruth(valid_aids_())
        hasgt_aids = utool.filter_items(valid_aids_(), has_gt_list)
        print('Testing all %d/%d ground-truthed cases' % (len(hasgt_aids),
                                                          len(valid_aids_())))
        available_qaids.extend(hasgt_aids)

    # Sample a large pool of chosen query indexes
    # Filter only the ones you want from the large pool
    if params.args.index is not None:
        indexes = ensure_flatlistlike(params.args.index)
        printDBG('Chosen indexes=%r' % (indexes,))
        printDBG('available_qaids = %r' % available_qaids[0:5])
        _test_qaids = [available_qaids[xx] for xx in indexes]
        available_qaids = _test_qaids
        printDBG('available_qaids = %r' % available_qaids)
    elif len(available_qaids) == 0 and len(valid_aids_()) > 0:
        printDBG('no hard or gt aids. Defaulting to the first ANNOTATION')
        available_qaids = valid_aids_()[0:1]

    #print('available_qaids = %r' % available_qaids)
    available_qaids = utool.unique_keep_order2(available_qaids)
    return available_qaids


@utool.indent_func
@profile
def get_test_daids(ibs):
    """ Gets database annot_rowids based on command line arguments """
    available_daids = ibs.get_valid_aids()
    if params.args.daid_exclude is not None:
        available_daids = list(set(available_daids) - set(params.args.daid_exclude))
    return available_daids
