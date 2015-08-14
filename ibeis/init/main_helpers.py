# -*- coding: utf-8 -*-
"""
TODO: Rename to ibeis/init/commands.py

The AID configuration selection is getting a mjor update right now
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np  # NOQA
import six
from ibeis.init import old_main_helpers
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[main_helpers]')


# DEPRICATE
get_test_daids = old_main_helpers.get_test_daids
get_test_qaids = old_main_helpers.get_test_qaids


VERB_TESTDATA = ut.get_argflag(('--verbose-testdata', '--verbtd'))
VERB_MAIN_HELPERS = ut.get_argflag(('--verbose-main-helpers', '--verbmhelp')) or ut.VERBOSE or VERB_TESTDATA


def testdata_ibeis(default_qaids=[1], default_daids='all', defaultdb='testdb1', ibs=None, verbose=False, return_annot_info=False):
    r"""
    Args:
        default_qaids (list): (default = [1])
        default_daids (str): (default = 'all')
        defaultdb (str): (default = 'testdb1')
        ibs (IBEISController):  ibeis controller object(default = None)
        verbose (bool):  verbosity flag(default = False)
        return_annot_info (bool): (default = False)

    Returns:
        ibs, qaid_list, daid_list, annot_info:

    CommandLine:
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db NNP_Master3
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --acfg default:aids=gt,shuffle,index=0:25 --verbose-testdata
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --acfg default:aids=gt,index=0:25 --verbose-testdata
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db NNP_Master3 --verbose-testdata -a controlled
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db NNP_Master3 --verbose-testdata --aidcfg controlled
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db NNP_Master3 --verbose-testdata --aidcfg default:species=None

        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db NNP_Master3 --acfg controlled --verbose-testdata
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_Master0 --acfg controlled --verbose-testdata

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.init.main_helpers import *  # NOQA
        >>> import ibeis
        >>> default_qaids = [1]
        >>> default_daids = 'all'
        >>> defaultdb = 'testdb1'
        >>> ibs = None
        >>> verbose = False
        >>> return_annot_info = True
        >>> ibs, qaid_list, daid_list, aidcfg = testdata_ibeis(default_qaids, default_daids, defaultdb, ibs, verbose, return_annot_info)
        >>> print('Printing annot config')
        >>> print(ut.dict_str(aidcfg))
        >>> print('Printing annotconfig stats')
        >>> #print('qaid_list = %r' % (np.array(qaid_list),))
        >>> ibs.get_annotconfig_stats(qaid_list, daid_list)
    """
    print('[testdata_ibeis] Getting test annot configs')
    import ibeis
    if ibs is None:
        ibs = ibeis.opendb(defaultdb=defaultdb)
    # TODO: rectify command line with function arguments
    from ibeis.experiments import experiment_helpers
    aidcfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=['default'])
    aidcfg_list = experiment_helpers.get_annotcfg_list(aidcfg_name_list)
    #aidcfg = old_main_helpers.get_commandline_aidcfg()
    aidcfg = aidcfg_list[0]

    qaid_list, daid_list = get_config_qaids_and_daids(ibs, aidcfg)

    ibs.get_annotconfig_stats(qaid_list, daid_list)

    if ut.VERYVERBOSE:
        ibeis.other.dbinfo.print_qd_info(ibs, qaid_list, daid_list, verbose=True)
    if return_annot_info:
        return ibs, qaid_list, daid_list, aidcfg
    else:
        return ibs, qaid_list, daid_list


def get_config_qaids_and_daids(ibs, aidcfg):
    """
    Expands an annot config dict into qaids and daids
    """
    qcfg = aidcfg['qcfg']
    dcfg = aidcfg['dcfg']

    # extract qaid list
    if VERB_MAIN_HELPERS:
        print('\n[expand_aidcfg] + --- GET_TEST_QAIDS ---')
    qaid_list = get_config_aids(ibs, aidcfg=qcfg)
    if VERB_MAIN_HELPERS:
        print('[expand_aidcfg] L ___ GET_TEST_QAIDS')

    # extract daid list
    if VERB_MAIN_HELPERS:
        print('[expand_aidcfg] + --- GET_TEST_DAIDS ---')
    daid_list = get_config_aids(ibs, aidcfg=dcfg, reference_aids=qaid_list)
    if VERB_MAIN_HELPERS:
        print('[expand_aidcfg] L ___ GET_TEST_DAIDS \n')
    return qaid_list, daid_list


def get_config_aids(ibs, aidcfg=None, reference_aids=None):
    """
    New version of this function based on a configuration dictionary built from
    command line argumetns

    Args:
        ibs (IBEISController):  ibeis controller object
        aidcfg (None): (default = None)

    Returns:
        tuple: (available_qaids, qaid_request_info)

    CommandLine:
        python -m ibeis.init.main_helpers --exec-get_config_aids --verbose-testdata
        python -m ibeis.init.main_helpers --exec-get_config_aids --acfg controlled --verbose-testdata

        python -m ibeis.init.main_helpers --exec-testdata_ibeis --verbtd --db NNP_Master3 --acfg controlled
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --verbtd --db PZ_Master0 --acfg controlled

        python -m ibeis.init.main_helpers --exec-testdata_ibeis --verbtd --db NNP_Master3 --acfg controlled:require_viewpoint=True,require_quality=True

        python -m ibeis.dev -a candidacy:qsize=10,dsize=10 -t default --db PZ_MTEST --verbtd --quiet
        python -m ibeis.dev -a candidacy:qsize=10,dsize=10,dper_name=2 -t default --db PZ_MTEST --draw-rank-cdf --show
        python -m ibeis.dev -a candidacy:qsize=10,dsize=20,dper_name=2 -t default --db PZ_MTEST --draw-rank-cdf --show
        python -m ibeis.dev -a candidacy:qsize=10,dsize=30,dper_name=2 -t default --db PZ_MTEST --draw-rank-cdf --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.init.main_helpers import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aidcfg = get_commandline_aidcfg()['qcfg']
        >>> available_qaids = get_config_aids(ibs, aidcfg)
        >>> result = ('(available_qaids) = %s' % (str((available_qaids)),))
        >>> print(result)
    """
    from ibeis import ibsfuncs
    default_aids = aidcfg['default_aids']

    # ---- INCLUDING STEP
    if VERB_MAIN_HELPERS:
        print(' * AIDCFG OVERRIDE')
        print(' * PARSING aidcfg = ' + ut.dict_str(aidcfg, align=True))
        print(' * INCLUDE STEP')
        print(' * default_aids = %s' % (ut.obj_str(default_aids, truncate=True, nl=False)))

    if isinstance(default_aids, six.string_types):
        if VERB_MAIN_HELPERS:
            print(' ... interpreting default aids.')
        # Abstract default aids
        if default_aids in ['all']:
            default_aids = ibs.get_valid_aids()
        elif default_aids in ['allgt', 'gt']:
            default_aids = ibs.get_valid_aids(hasgt=True)
        #elif default_aids in ['reference_gt']:
        #    pass
        else:
            raise NotImplementedError('Unknown default string = %r' % (default_aids,))
    else:
        if VERB_MAIN_HELPERS:
            print(' ... default aids specified.')

    #if aidcfg['include_aids'] is not None:
    #    raise NotImplementedError('Implement include_aids')

    available_aids = default_aids

    if len(available_aids) == 0:
        print(' WARNING no annotations available')

    # ---- FILTERING STEP
    if VERB_MAIN_HELPERS:
        print(' * len(available_aids) = %r' % (len(available_aids)))
        print(' * FILTERING STEP')

    if aidcfg['require_timestamp'] is True:
        if VERB_MAIN_HELPERS:
            print(' * Removing annots without timestamp')
        available_aids = ibs.filter_aids_without_timestamps(available_aids)

    species = None
    if aidcfg['species'] is not None:
        if aidcfg['species'] == 'primary':
            if VERB_MAIN_HELPERS:
                print(' * Finiding primary species')
            species = species = ibs.get_primary_database_species()
        else:
            species = aidcfg['species']
        if VERB_MAIN_HELPERS:
            print(' * Filtering to species=%r' % (species,))
        available_aids = ibs.filter_aids_to_species(available_aids, species)

    if aidcfg['minqual'] is not None or aidcfg['require_quality']:
        # Resolve quality
        if aidcfg['minqual'] is None:
            minqual = 'junk'
        else:
            minqual = aidcfg['minqual']
        if VERB_MAIN_HELPERS:
            print(' * Filtering quality. minqual=%r, require_quality=%r'
                  % (minqual, aidcfg['require_quality']))
        # Filter quality
        available_aids = ibs.filter_aids_to_quality(available_aids, minqual, unknown_ok=not aidcfg['require_quality'])

    if aidcfg['base_viewpoint'] is not None or aidcfg['require_viewpoint']:
        # Resolve base viewpoint
        if aidcfg['base_viewpoint'] == 'primary':
            base_viewpoint = ibsfuncs.get_primary_species_viewpoint(species)
        else:
            base_viewpoint = aidcfg['base_viewpoint']
        valid_yaws = ibsfuncs.get_extended_viewpoints(base_viewpoint, num1=aidcfg['viewpoint_range'], num2=0)
        if VERB_MAIN_HELPERS:
            print(' * Filtering viewpoint. valid_yaws=%r, require_viewpoint=%r'
                  % (valid_yaws, aidcfg['require_viewpoint']))
        # Filter viewpoint
        available_aids = ibs.filter_aids_to_viewpoint(available_aids, valid_yaws, unknown_ok=not aidcfg['require_viewpoint'])

    #if aidcfg['exclude_aids'] is not None:
    #    if VERB_MAIN_HELPERS:
    #        print(' * Excluding %d custom aids' % (len(aidcfg['exclude_aids'])))
    #    available_aids = ut.setdiff_ordered(available_aids, aidcfg['exclude_aids'])

    if aidcfg['exclude_reference'] is not None:
        assert reference_aids is not None, 'reference_aids=%r' % (reference_aids,)
        if VERB_MAIN_HELPERS:
            print(' * Excluding %d reference aids' % (len(reference_aids)))
        available_aids = ut.setdiff_ordered(available_aids, reference_aids)

    if aidcfg['min_per_name'] is not None:
        if VERB_MAIN_HELPERS:
            print(' * Filtering min_per_name=%d' % (aidcfg['min_per_name']))
        grouped_aids_, unique_nids = ibs.group_annots_by_name(available_aids, distinguish_unknowns=True)
        min_gt = aidcfg['min_per_name']
        available_aids = list(ut.iflatten(filter(lambda x: len(x) >= min_gt, grouped_aids_)))

    # ---- FILTERING SELECTION
    if VERB_MAIN_HELPERS:
        print(' * len(available_aids) = %r' % (len(available_aids)))
        print(' * FILTERED SELECTION STEP')

    if aidcfg['sample_per_name'] is not None:
        if VERB_MAIN_HELPERS:
            print(' * Filtering number of annots per name to %r using rule %r' % (aidcfg['sample_per_name'], aidcfg['sample_rule'] ))

        if aidcfg['sample_rule'] == 'ref_max_timedelta':
            # Maximize time delta between query and corresponding database annotations
            assert reference_aids is not None
            # sample wrt the reference set
            # available aids that are groundtruth to the reference

            # TODO: verify
            #ref_multi, avl_multi, ref_single, avl_single = ibs.partition_annots_into_corresponding_groups(reference_aids, available_aids)
            #assert len(ref_single) == 0, 'should not have uncorresponding refs'
            #grouped_reference_aids = ref_multi
            #grouped_available_gt_aids = avl_multi
            #available_gf_aids = avl_single

            # Group reference (q)aids by name
            grouped_reference_aids = ibs.group_annots_by_name(reference_aids)[0]
            # Get the group of available aids that a reference aid could match
            grouped_available_gt_aids = ibs.get_annot_groundtruth(ut.get_list_column(grouped_reference_aids, 0), daid_list=available_aids)
            # The available aids that will should not match a reference aid
            available_gf_aids = ut.setdiff_ordered(available_aids, ut.flatten(grouped_available_gt_aids))
            cmp_func = ut.absdiff
            aggfn = np.mean
            prop_getter = ibs.get_annot_image_unixtimes_asfloat

            def order_by_agg_metric(grouped_reference_aids, grouped_available_gt_aids, prop_getter, cmp_func, aggfn):
                """
                # TODO: generalize this part
                #avl_prop = grouped_available_gt_props[0]
                #ref_prop = grouped_reference_props[0]
                #ref_prop = np.array([0, 10])
                #avl_prop = np.array([5, 7, 30, 10])
                #aggfn = np.product #aggfn = np.max #aggfn = np.mean
                """

                grouped_reference_unixtimes = ibs.unflat_map(prop_getter, grouped_reference_aids)
                grouped_available_gt_unixtimes = ibs.unflat_map(prop_getter, grouped_available_gt_aids)

                grouped_reference_props = grouped_reference_unixtimes
                grouped_available_gt_props = grouped_available_gt_unixtimes

                # Order the available aids by some aggregation over some metric
                preference_scores = [aggfn(cmp_func(ref_prop, avl_prop[:, None]), axis=1)
                                     for ref_prop, avl_prop in zip(grouped_reference_props, grouped_available_gt_props)]

                # Order by increasing timedelta (metric)
                reverse = True
                if reverse:
                    preference_orders = [scores.argsort()[::-1] for scores in preference_scores]
                else:
                    preference_orders = [scores.argsort() for scores in preference_scores]

                pref_ordered_available_gt_aids = ut.list_ziptake(grouped_available_gt_aids, preference_orders)
                return pref_ordered_available_gt_aids

            pref_ordered_available_gt_aids = order_by_agg_metric(grouped_reference_aids, grouped_available_gt_aids, prop_getter, cmp_func, aggfn)
            offset = aidcfg['sample_offset']
            # Potentially choose a different number for reference (groundtruth casees)
            sample_per_ref_name = aidcfg['sample_per_ref_name']
            if sample_per_ref_name is None:
                sample_per_ref_name = aidcfg['sample_per_name']
            #sample_available_gt_aids = ut.get_list_column_slice(pref_ordered_available_gt_aids, offset, offset + aidcfg['sample_per_name'])
            sample_available_gt_aids = ut.get_list_column_slice(pref_ordered_available_gt_aids, offset, offset + aidcfg['sample_per_ref_name'])

            # set the sample to the maximized ref, with all groundfalse
            print('Before special rule filter len(available_aids)=%r' % (len(available_aids)))
            sample_available_gf_aids = ibs.get_annot_rowid_sample(
                available_gf_aids, per_name=aidcfg['sample_per_name'], min_gt=None,
                method='random', offset=aidcfg['sample_offset'], seed=0)
            available_aids = ut.flatten(sample_available_gt_aids) + sample_available_gf_aids
            print('After special rule filter len(available_aids)=%r' % (len(available_aids)))
        else:
            # For the query we just choose a single annot per name
            # For the database we have to do something different
            available_aids = ibs.get_annot_rowid_sample(
                available_aids, per_name=aidcfg['sample_per_name'], min_gt=None,
                method=aidcfg['sample_rule'], offset=aidcfg['sample_offset'], seed=0)

    if aidcfg['sample_size'] is not None:
        """
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --acfg controlled:qoffset=2,drule=ref_max_timedelta,dsize=200
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --acfg controlled:qoffset=2,drule=ref_max_timedelta,dsize=10
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --acfg controlled:qoffset=2,drule=ref_max_timedelta,dsize=41,dper_name=2
        """
        # TODO:
        # Allow removal of multitons if reference_aids is not given
        # Randomly sample which annots are removed
        if reference_aids is not None:
            # Enesure that the sampleing does not conflict with reference aid properties
            if VERB_MAIN_HELPERS:
                print(' * Filtering to sample size %r' % (aidcfg['sample_size'],))
            assert reference_aids is not None and len(reference_aids) > 0
            ref_multi, avl_multi, ref_single, avl_single = ibs.partition_annots_into_corresponding_groups(reference_aids, available_aids)
            assert len(ref_single) == 0, 'should not have uncorresponding refs'
            #singletons, multitons = ibs.partition_annots_into_singleton_multiton(available_aids)
            multitons = ut.flatten(avl_multi)
            singletons = avl_single
            num_single = len(singletons)
            num_multi = len(multitons)
            assert num_single + num_multi == len(available_aids), 'does not sum'
            num_keep_single = aidcfg['sample_size'] - num_multi
            num_remove_single = num_single - num_keep_single
            if num_keep_single < 0 or num_remove_single < 0:
                print('Warning cannot sample to requested sample size completely num_keep_single=%r num_remove_single=%r' % (num_keep_single, num_remove_single))
                #num_keep_single = max(0, min(num_keep_single, num_single))
            singletons = ut.random_sample(singletons, num_keep_single, seed=42)
            available_aids = multitons + singletons
        else:
            # No reference aids. Can remove freely.
            if aidcfg['sample_size'] > available_aids:
                print('Warning sample size too large')
            available_aids = ut.random_sample(available_aids, aidcfg['sample_size'], seed=42)

    # ---- SUBINDEXING STEP
    if VERB_MAIN_HELPERS:
        print(' * len(available_aids) = %r' % (len(available_aids)))
        print(' * SUBINDEX STEP')

    #ut.get_argval('--qshuffle')

    if aidcfg['shuffle']:
        if VERB_MAIN_HELPERS:
            print(' * Shuffling with seed=42')
        # Determenistic shuffling
        available_aids = ut.list_take(available_aids, ut.random_indexes(len(available_aids), seed=42))

    if aidcfg['index'] is not None:
        if VERB_MAIN_HELPERS:
            print(' * Indexing')
        indicies = ensure_flatlistlike(aidcfg['index'])
        _indexed_aids = [available_aids[ix] for ix in indicies if ix < len(available_aids)]
        print('[get_test_daids] Chose subset of size %d/%d' % (len(_indexed_aids), len(available_aids)))
        available_aids = _indexed_aids

    if VERB_MAIN_HELPERS:
        print(' * len(available_aids) = %r' % (len(available_aids)))
    return available_aids


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
    if isinstance(input_, six.string_types):
        input_ = ut.fuzzy_int(input_)
    if isinstance(input_, int) or not ut.isiterable(input_):
        return [input_]
    elif isinstance(input_, (list, tuple)):
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
