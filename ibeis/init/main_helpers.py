"""
TODO: Rename to ibeis/init/commands.py
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import six
from ibeis import params
# Inject utool functions
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[main_helpers]')


VERB_TESTDATA = ut.get_argflag('--verbose-testdata')
VERB_MAIN_HELPERS = ut.get_argflag(('--verbose-main-helpers', '--verbmhelp')) or ut.VERBOSE or VERB_TESTDATA


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


def get_test_qaids(ibs, default_qaids=None):
    """
    Gets test annot_rowids based on command line arguments

    Args:
        ibs (IBEISController):  ibeis controller object
        default_qaids (None): if list then used only if no other aids are available (default = [1])
           as a string it mimics the command line

    Returns:
        list: available_qaids

    CommandLine:
        python -m ibeis.init.main_helpers --test-get_test_qaids
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_Master0
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_Master0 --qaid 1
        python -m ibeis.init.main_helpers --test-get_test_qaids --allgt --db PZ_MTEST
        python -m ibeis.init.main_helpers --test-get_test_qaids --qaid 4 5 8  --verbmhelp
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_MTEST
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_MTEST --qaid 2 --verbmhelp
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_MTEST --qaid 2
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_Master0 --qindex 0:10 --verbmhelp
        python -m ibeis.init.main_helpers --exec-get_test_qaids --controlled --db PZ_Master0 --exec-mode
        python -m ibeis.init.main_helpers --exec-get_test_qaids --db testdb1 --allgt --qindex 0:256

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.init.main_helpers import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> default_qaids = None
        >>> available_qaids = get_test_qaids(ibs, default_qaids)
        >>> ibeis.other.dbinfo.get_dbinfo(ibs, aid_list=available_qaids, with_contrib=False, short=True)
        >>> result = 'available_qaids = ' + ut.obj_str(available_qaids, truncate=True, nl=False)
        >>> print('len(available_qaids) = %d' % len(available_qaids))
        >>> print(result)
        available_qaids = [1]
    """

    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] + --- GET_TEST_QAIDS ---')
        print('[get_test_qaids] * default_qaids = %s' % (ut.obj_str(default_qaids, truncate=True, nl=False)))

    valid_aids = ibs.get_valid_aids()

    if valid_aids == 0:
        print('[get_test_qaids] WARNING no annotations available')

    # ---- INCLUDING STEP
    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] * include step')

    available_qaids = []

    ALL_CASES = params.args.all_cases or default_qaids == 'all'
    GT_CASES = params.args.all_gt_cases or default_qaids == 'gt'
    HARD_CASES = params.args.all_hard_cases or ut.get_argflag(('--all-hard-cases', '--allhard', '--hard'))
    NO_JUNK = not ut.get_argflag('--junk')

    if params.args.qaid is not None:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Including cmdline specified qaids')
        try:
            args_qaid = ensure_flatlistlike(params.args.qaid)
        except Exception:
            args_qaid = params.args.qaid
        available_qaids.extend(args_qaid)

    if ALL_CASES:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Including all qaids')
        available_qaids.extend(valid_aids)

    if HARD_CASES:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Including hard qaids')
        is_hard_list = ibs.get_annot_is_hard(valid_aids)
        hard_aids = ut.filter_items(valid_aids, is_hard_list)
        available_qaids.extend(hard_aids)

    if GT_CASES:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Including groundtruth qaids')
        has_gt_list = ibs.get_annot_has_groundtruth(valid_aids)
        hasgt_aids = ut.filter_items(valid_aids, has_gt_list)
        print('[get_test_qaids] Adding all %d/%d ground-truthed test cases' % (len(hasgt_aids), len(valid_aids)))
        available_qaids.extend(hasgt_aids)

    CONTROLLED_CASES = ut.get_argflag('--controlled') or ut.get_argflag('--controlled_qaids')
    if CONTROLLED_CASES:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Including controlled qaids')
        from ibeis import ibsfuncs
        # Override all other gts with controlled
        controlled_qaids = ibsfuncs.get_two_annots_per_name_and_singletons(ibs, onlygt=True)
        available_qaids.extend(controlled_qaids)

    # ---- CHECK_DEFAULTS QUERY
    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] * len(available_qaids) = %r' % (len(available_qaids)))

    if len(available_qaids) == 0:
        print('[get_test_qaids] * ... defaulting, no available qaids on command line.')
        if default_qaids is None:
            default_qaids = valid_aids[0:1]
        elif isinstance(default_qaids, six.string_types):
            if default_qaids == 'gt':
                default_qaids = ibs.get_valid_aids(hasgt=True)
        available_qaids = default_qaids
    else:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * ... not defaulting')

    available_qaids = ut.unique_keep_order2(available_qaids)

    # ---- EXCLUSION STEP
    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] * len(available_qaids) = %r' % (len(available_qaids)))
        print('[get_test_qaids] * exclude step')

    if NO_JUNK:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Filtering junk')
        available_qaids = ibs.filter_junk_annotations(available_qaids)

    if ut.get_argflag('--unreviewed'):
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Filtering unreviewed')
        isreviewed_list = ibs.get_annot_has_reviewed_matching_aids(available_qaids)
        available_qaids = ut.filterfalse_items(available_qaids, isreviewed_list)

    species = ut.get_argval('--species')

    if species is not None:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Filtering to species=%r' % (species,))
        import numpy as np
        isvalid_list = np.array(ibs.get_annot_species(available_qaids)) == species
        available_qaids = ut.filter_items(available_qaids, isvalid_list)

    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] * len(available_qaids) = %r' % (len(available_qaids)))
        print('[get_test_qaids] * subindex step')

    # ---- INDEX SUBSET

    #ut.get_argval('--qshuffle')
    if ut.get_argval('--qshuffle'):
        # Determenistic shuffling
        available_qaids = ut.list_take(available_qaids, ut.random_indexes(len(available_qaids), seed=42))

    # Sample a large pool of chosen query qindexes
    if params.args.qindex is not None:
        # FIXME: should use a slice of the list or a sublist
        qindexes = ensure_flatlistlike(params.args.qindex)
        _test_qaids = [available_qaids[qx] for qx in qindexes if qx < len(available_qaids)]
        print('[get_test_qaids] Chose subset of size %d/%d' % (len(_test_qaids), len(available_qaids)))
        available_qaids = _test_qaids

    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] * len(available_qaids) = %r' % (len(available_qaids)))
        print('[get_test_qaids] L ___ GET_TEST_QAIDS ___')

    return available_qaids


def get_test_daids(ibs, default_daids='all', qaid_list=None):
    """ Gets database annot_rowids based on command line arguments

    CommandLine:
        python dev.py --db PZ_MTEST -t best --exclude-query --qaid 72 -r 0 -c 0 --show --va --vf --dump-extra

    Args:
        ibs (IBEISController):  ibeis controller object
        default_daids (str): (default = 'all')
        qaid_list (list): list of chosen qaids that may affect daids (default = None)

    Returns:
        list: available_daids

    CommandLine:
        python -m ibeis.init.main_helpers --test-get_test_daids
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_MTEST  --verbmhelp
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_MTEST --exclude-query
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_MTEST --daid-exclude 2 3 4
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_MTEST --species=zebra_grevys
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_Master0 --species=zebra_grevys
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_Master0 --controlled --verbmhelp
        python -m ibeis.init.main_helpers --exec-get_test_daids --controlled --db PZ_Master0 --exec-mode

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.init.main_helpers import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> default_daids = 'all'
        >>> qaid_list = [1]
        >>> available_daids = get_test_daids(ibs, default_daids, qaid_list)
        >>> ibeis.other.dbinfo.get_dbinfo(ibs, aid_list=available_daids, with_contrib=False, short=True)
        >>> result = 'available_daids = ' + ut.obj_str(available_daids, truncate=True, nl=False)
        >>> print('len(available_daids) %d' % len(available_daids))
        >>> print(result)
        available_daids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    """

    if VERB_MAIN_HELPERS:
        print('[get_test_daids] + --- GET_TEST_DAIDS ---')
        print('[get_test_daids] * default_daids = %s' % (ut.obj_str(default_daids, truncate=True, nl=False)))
        print('[get_test_daids] * qaid_list = %s' % (ut.obj_str(qaid_list, truncate=True, nl=False)))

    # ---- INCLUDING STEP
    if VERB_MAIN_HELPERS:
        print('[get_test_daids] * include step')

    available_daids = []

    CONTROLLED_CASES = ut.get_argflag('--controlled') or ut.get_argflag('--controlled_daids')

    if CONTROLLED_CASES:
        print('[get_test_daids] * Including controlled daids')
        from ibeis import ibsfuncs
        controlled_daids = ibsfuncs.get_two_annots_per_name_and_singletons(ibs, onlygt=False)
        available_daids.extend(controlled_daids)

    # ---- CHECK_DEFAULTS DATA
    if VERB_MAIN_HELPERS:
        print('[get_test_daids] * len(available_daids) = %r' % (len(available_daids)))

    if len(available_daids) == 0:
        print('[get_test_daids] * ... defaulting, no available daids on command line.')
        if isinstance(default_daids, six.string_types):
            if default_daids == 'all':
                default_daids = ibs.get_valid_aids()
            elif default_daids == 'gt':
                default_daids = ut.flatten(ibs.get_annot_groundtruth(qaid_list))
        #available_qaids = valid_aids[0:1]
        assert not isinstance(available_daids, six.string_types)
        available_daids = default_daids
    else:
        if VERB_MAIN_HELPERS:
            print('[get_test_daids] * ... not defaulting')

    available_daids = ut.unique_keep_order2(available_daids)

    # ---- EXCLUSION STEP
    if VERB_MAIN_HELPERS:
        print('[get_test_daids] * len(available_daids) = %r' % (len(available_daids)))
        print('[get_test_daids] * exclude step')

    NO_JUNK = not ut.get_argflag('--junk')
    EXCLUDE_QUERY = ut.get_argflag('--exclude-query')
    species = ut.get_argval('--species', type_=str, default=None)
    daids_exclude = params.args.daid_exclude

    if NO_JUNK:
        if VERB_MAIN_HELPERS:
            print('[get_test_daids] * Filtering junk')
        available_daids = ibs.filter_junk_annotations(available_daids)

    if EXCLUDE_QUERY:
        if VERB_MAIN_HELPERS:
            print('[get_test_daids] * Excluding query qaids')
        assert qaid_list is not None, 'must specify qaids to exclude'
        available_daids = ut.setdiff_ordered(available_daids, qaid_list)

    if daids_exclude is not None:
        if VERB_MAIN_HELPERS:
            print('[get_test_daids] * Excluding specified daids')
        available_daids = ut.setdiff_ordered(available_daids, daids_exclude)

    if species is not None:
        if VERB_MAIN_HELPERS:
            print('[get_test_daids] * Filtering to species=%r' % (species,))
        import numpy as np
        isvalid_list = np.array(ibs.get_annot_species(available_daids)) == species
        available_daids = ut.filter_items(available_daids, isvalid_list)

    # ---- SUBINDEXING STEP
    if VERB_MAIN_HELPERS:
        print('[get_test_daids] * len(available_daids) = %r' % (len(available_daids)))
        print('[get_test_daids] * subindex step')

    if params.args.dindex is not None:
        dindexes = ensure_flatlistlike(params.args.dindex)
        _test_daids = [available_daids[dx] for dx in dindexes if dx < len(available_daids)]
        print('[get_test_daids] Chose subset of size %d/%d' % (len(_test_daids), len(available_daids)))
        available_daids = _test_daids

    if VERB_MAIN_HELPERS:
        print('[get_test_daids] * len(available_daids) = %r' % (len(available_daids)))
        print('[get_test_daids] L ___ GET_TEST_DAIDS ___')

    return available_daids


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.init.main_helpers
        python -m ibeis.init.main_helpers --allexamples
        python -m ibeis.init.main_helpers --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
