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

VERB_TESTDATA, VERYVERB_TESTDATA = ut.get_verbflag('testdata', 'td')
VERYVERB_MAIN_HELPERS = VERYVERB_TESTDATA
VERB_MAIN_HELPERS = VERB_TESTDATA

#VERB_TESTDATA = ut.get_argflag(('--verbose-testdata', '--verbtd')) or VERYVERB_TESTDATA
#VERB_MAIN_HELPERS = ut.get_argflag(('--verbose-main-helpers', '--verbmhelp')) or ut.VERBOSE or VERB_TESTDATA


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
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db GZ_ALL --acfg controlled --verbose-testdata

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
    acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, aidcfg_name_list)

    #aidcfg = old_main_helpers.get_commandline_aidcfg()
    assert len(acfg_list) == 1, 'multiple acfgs specified, but this function is built to return only 1. len(acfg_list)=%r' % (len(acfg_list),)
    aidcfg = acfg_list[0]

    qaid_list, daid_list = expanded_aids_list[0]

    #ibs.get_annotconfig_stats(qaid_list, daid_list)

    if ut.VERYVERBOSE:
        ibeis.other.dbinfo.print_qd_info(ibs, qaid_list, daid_list, verbose=True)
    if return_annot_info:
        return ibs, qaid_list, daid_list, aidcfg
    else:
        return ibs, qaid_list, daid_list


@profile
def expand_acfgs(ibs, aidcfg):
    """
    Expands an annot config dict into qaids and daids
    New version of this function based on a configuration dictionary built from
    command line argumetns

    CommandLine:
        python -m ibeis.init.main_helpers --exec-expand_aidcfg_dict --verbose-testdata
        python -m ibeis.init.main_helpers --exec-expand_aidcfg_dict --acfg controlled --verbose-testdata

        python -m ibeis.init.main_helpers --exec-testdata_ibeis --verbtd --db NNP_Master3 --acfg controlled
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --verbtd --db PZ_Master0 --acfg controlled

        python -m ibeis.init.main_helpers --exec-testdata_ibeis --verbtd --db NNP_Master3 --acfg controlled:require_viewpoint=True,require_quality=True

        python -m ibeis.dev -a candidacy:qsize=10,dsize=10 -t default --db PZ_MTEST --verbtd --quiet
        python -m ibeis.dev -a candidacy:qsize=10,dsize=10,dper_name=2 -t default --db PZ_MTEST --draw-rank-cdf --show
        python -m ibeis.dev -a candidacy:qsize=10,dsize=20,dper_name=2 -t default --db PZ_MTEST --draw-rank-cdf --show
        python -m ibeis.dev -a candidacy:qsize=10,dsize=30,dper_name=2 -t default --db PZ_MTEST --draw-rank-cdf --show

        python -m ibeis.experiments.experiment_printres --exec-print_latexsum -t candidacy_invariance -a viewpoint_compare  --db NNP_Master3 --acfginfo
        utprof.py -m ibeis.experiments.experiment_printres --exec-print_latexsum -t candidacy_k -a varysize  --db PZ_Master0 --acfginfo
        utprof.py -m ibeis.experiments.experiment_printres --exec-print_latexsum -t candidacy_k -a controlled  --db PZ_Master0 --acfginfo
    """
    from ibeis.experiments import annotation_configs
    qcfg = aidcfg['qcfg']
    dcfg = aidcfg['dcfg']

    USE_ACFG_CACHE = not ut.get_argflag(('--nocache-aid', '--nocache'))
    if USE_ACFG_CACHE:
        # Make loading aids a big faster for experiments
        acfg_cachedir = './ACFG_CACHE'
        acfg_cachename = 'ACFG_CACHE'
        aid_cachestr = ibs.get_dbname() + '_' + ut.hashstr27(ut.to_json(aidcfg))
        try:
            (qaid_list, daid_list) = ut.load_cache(acfg_cachedir, acfg_cachename, aid_cachestr)
        except IOError:
            pass
        else:
            return qaid_list, daid_list

    VERB_MAIN_HELPERS2 = VERB_MAIN_HELPERS

    # ---- INCLUDING STEP
    if VERB_MAIN_HELPERS or VERB_MAIN_HELPERS2:
        print('+=== EXPAND_ACFGS ===')
        print(' * acfg = %s' % (ut.dict_str(annotation_configs.compress_aidcfg(aidcfg), align=True),))
        print('+---------------------')

    available_qaids = expand_to_default_aids(ibs, qcfg, prefix='q')
    available_daids = expand_to_default_aids(ibs, dcfg, prefix='d')

    available_qaids = filter_independent_properties(ibs, available_qaids, qcfg, prefix='q')
    available_daids = filter_independent_properties(ibs, available_daids, dcfg, prefix='d')

    available_qaids = filter_reference_properties(ibs, available_qaids, qcfg, reference_aids=available_daids, prefix='q')
    available_daids = filter_reference_properties(ibs, available_daids, dcfg, reference_aids=available_qaids, prefix='d')

    available_qaids = sample_available_aids(ibs, available_qaids, qcfg, reference_aids=None, prefix='q')  # No reference sampling for query
    available_daids = sample_available_aids(ibs, available_daids, dcfg, reference_aids=available_qaids, prefix='d')

    available_qaids = subindex_avaiable_aids(ibs, available_qaids, qcfg, prefix='q')
    available_daids = subindex_avaiable_aids(ibs, available_daids, dcfg, prefix='d')

    qaid_list = available_qaids
    daid_list = available_daids

    if VERB_MAIN_HELPERS or VERB_MAIN_HELPERS2:
        print('L___ EXPAND_ACFGS ___')
        ibs.get_annotconfig_stats(qaid_list, daid_list, verbose=True)

    """
    aidcfg = dcfg
    available_aids = available_daids
    reference_aids = available_qaids

    aidcfg = qcfg
    available_aids = available_qaids
    reference_aids = available_daids

    ut.print_dict(ibs.get_annot_stats_dict(available_qaids, 'q'))
    ut.print_dict(ibs.get_annot_stats_dict(available_daids, 'd'))

    _ = ibs.get_annotconfig_stats(available_qaids, available_daids)

    """
    if USE_ACFG_CACHE:
        ut.ensuredir(acfg_cachedir)
        ut.save_cache(acfg_cachedir, acfg_cachename, aid_cachestr, (qaid_list, daid_list))
    #available_qaids qcfg['ref_has_viewpoint']
    return qaid_list, daid_list


@profile
def expand_to_default_aids(ibs, aidcfg, prefix=''):
    default_aids = aidcfg['default_aids']

    if VERB_MAIN_HELPERS:
        print(' * [INCLUDE %sAIDS]' % (prefix.upper()))
        #print(' * PARSING %saidcfg = %s' % (prefix, ut.dict_str(aidcfg, align=True),))
        print(' * default_%saids = %s' % (prefix, ut.obj_str(default_aids, truncate=True, nl=False)))

    if isinstance(default_aids, six.string_types):
        #if VERB_MAIN_HELPERS:
        #    print(' * interpreting default %saids.' % (prefix,))
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
            print(' ... default %saids specified.' % (prefix,))

    #if aidcfg['include_aids'] is not None:
    #    raise NotImplementedError('Implement include_aids')

    available_aids = default_aids

    if len(available_aids) == 0:
        print(' WARNING no %s annotations available' % (prefix,))

    #if aidcfg['exclude_aids'] is not None:
    #    if VERB_MAIN_HELPERS:
    #        print(' * Excluding %d custom aids' % (len(aidcfg['exclude_aids'])))
    #    available_aids = ut.setdiff_ordered(available_aids, aidcfg['exclude_aids'])

    if VERB_MAIN_HELPERS:
        print(' * DEFAULT: len(available_%saids)=%r\n' % (prefix, len(available_aids)))
    return available_aids


@profile
def filter_independent_properties(ibs, available_aids, aidcfg, prefix=''):
    """ Filtering that doesn't have to do with a reference set of aids """
    from ibeis import ibsfuncs
    if VERB_MAIN_HELPERS:
        print(' * [FILTER INDEPENDENT %sAIDS]' % (prefix.upper()))

    if aidcfg['is_known'] is True:
        if VERB_MAIN_HELPERS:
            print(' * Removing annots without names')
        available_aids = ibs.filter_aids_without_name(available_aids)

    if aidcfg['require_timestamp'] is True:
        if VERB_MAIN_HELPERS:
            print(' * Removing annots without timestamp')
        available_aids = ibs.filter_aids_without_timestamps(available_aids)

    species = None
    if aidcfg['species'] is not None:
        if aidcfg['species'] == 'primary':
            #if VERB_MAIN_HELPERS:
            #    print(' * Finiding primary species')
            species = ibs.get_primary_database_species()
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

    if aidcfg['viewpoint_base'] is not None or aidcfg['require_viewpoint']:
        # Resolve base viewpoint
        if aidcfg['viewpoint_base'] == 'primary':
            viewpoint_base = ibsfuncs.get_primary_species_viewpoint(species)
        elif aidcfg['viewpoint_base'] == 'primary+1':
            viewpoint_base = ibsfuncs.get_primary_species_viewpoint(species, 1)
        else:
            viewpoint_base = aidcfg['viewpoint_base']
        valid_yaws = ibsfuncs.get_extended_viewpoints(viewpoint_base, num1=aidcfg['viewpoint_range'], num2=0)
        if VERB_MAIN_HELPERS:
            print(' * Filtering viewpoint. valid_yaws=%r, require_viewpoint=%r'
                  % (valid_yaws, aidcfg['require_viewpoint']))
        # Filter viewpoint
        available_aids = ibs.filter_aids_to_viewpoint(available_aids, valid_yaws, unknown_ok=not aidcfg['require_viewpoint'])

    # Each aid must have at least this number of other groundtruth aids
    if aidcfg['gt_min_per_name'] is not None:
        if VERB_MAIN_HELPERS:
            print(' * Filtering gt_min_per_name=%d' % (aidcfg['gt_min_per_name']))
        grouped_aids_, unique_nids = ibs.group_annots_by_name(available_aids, distinguish_unknowns=True)
        min_gt = aidcfg['gt_min_per_name']
        available_aids = ut.flatten([x for x in grouped_aids_ if len(x) >= min_gt])

    if VERB_MAIN_HELPERS:
        print(' * I-FILTERED: len(available_%saids)=%r\n' % (prefix, len(available_aids)))

    return available_aids


@profile
def filter_reference_properties(ibs, available_aids, aidcfg, reference_aids, prefix=''):
    from ibeis import ibsfuncs
    import functools
    if VERB_MAIN_HELPERS:
        print(' * [FILTER REFERENCE %sAIDS]' % (prefix.upper()))

    if aidcfg['ref_has_viewpoint'] is not None:
        print(' * Filtering such that %saids has refs with viewpoint=%r' % (prefix, aidcfg['ref_has_viewpoint']))
        species = ibs.get_primary_database_species(available_aids)
        if aidcfg['ref_has_viewpoint']  == 'primary':
            valid_yaws = [ibsfuncs.get_primary_species_viewpoint(species)]
        elif aidcfg['ref_has_viewpoint']  == 'primary+1':
            valid_yaws = [ibsfuncs.get_primary_species_viewpoint(species, 1)]
        ref_multi, avl_multi, ref_single, avl_single = ibs.partition_annots_into_corresponding_groups(reference_aids, available_aids)
        # Filter to only available aids that have a reference with specified viewpoint
        is_valid_yaw = functools.partial(ibs.get_viewpoint_filterflags, valid_yaws=valid_yaws)
        multi_flags = list(map(any, ibs.unflat_map(is_valid_yaw, ref_multi)))
        available_aids = ut.flatten(ut.list_compress(avl_multi, multi_flags))

    if VERB_MAIN_HELPERS:
        print(' * R-FILTERED: len(available_%saids)=%r\n' % (prefix, len(available_aids)))

    return available_aids


@profile
def reference_sample_per_name(ibs, available_aids, aidcfg, reference_aids):
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
            # Bigger is better, replace nan with -inf
            # Then randomize order between all equal values
            for scores in preference_scores:
                scores[np.isnan(scores)] = np.inf
            #preference_orders = [scores.argsort()[::-1] for scores in preference_scores]
            rng = np.random.RandomState(42)
            preference_orders = [np.lexsort((scores, rng.rand(len(scores))))[::-1] for scores in preference_scores]
        else:
            # Smaller is better, replace nan with inf
            for scores in preference_scores:
                scores[np.isnan(scores)] = np.inf
            #preference_orders = [scores.argsort() for scores in preference_scores]
            rng = np.random.RandomState(42)
            preference_orders = [np.lexsort((scores, rng.rand(len(scores)))) for scores in preference_scores]

        pref_ordered_available_gt_aids = ut.list_ziptake(grouped_available_gt_aids, preference_orders)
        return pref_ordered_available_gt_aids

    pref_ordered_available_gt_aids = order_by_agg_metric(grouped_reference_aids, grouped_available_gt_aids, prop_getter, cmp_func, aggfn)
    offset = aidcfg['sample_offset']
    # Potentially choose a different number for reference (groundtruth casees)
    sample_per_ref_name = aidcfg['sample_per_ref_name']
    if sample_per_ref_name is None:
        sample_per_ref_name = aidcfg['sample_per_name']
    #sample_available_gt_aids = ut.get_list_column_slice(pref_ordered_available_gt_aids, offset, offset + aidcfg['sample_per_name'])
    sample_available_gt_aids = ut.get_list_column_slice(pref_ordered_available_gt_aids, offset, offset + sample_per_ref_name)

    # set the sample to the maximized ref, with all groundfalse
    if VERB_MAIN_HELPERS:
        print('Before special rule filter len(available_aids)=%r' % (len(available_aids)))
    sample_available_gf_aids = ibs.get_annot_rowid_sample(
        available_gf_aids, per_name=aidcfg['sample_per_name'], min_gt=None,
        method='random', offset=aidcfg['sample_offset'], seed=0)
    available_aids = ut.flatten(sample_available_gt_aids) + sample_available_gf_aids
    if VERB_MAIN_HELPERS:
        print('After special rule filter len(available_aids)=%r' % (len(available_aids)))
    return available_aids


@profile
def sample_available_aids(ibs, available_aids, aidcfg, reference_aids=None, prefix=''):
    """
    python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --a controlled:qoffset=2,drule=ref_max_timedelta,dsize=200
    python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --a controlled:qoffset=2,drule=ref_max_timedelta,dsize=10
    python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --a controlled:qoffset=2,drule=ref_max_timedelta,dsize=41,dper_name=2
    """
    if VERB_MAIN_HELPERS:
        print(' * [SAMPLE %sAIDS]' % (prefix.upper(),))

    if aidcfg['exclude_reference'] is not None:
        assert reference_aids is not None, 'reference_aids=%r' % (reference_aids,)
        if VERB_MAIN_HELPERS:
            print(' * Excluding %d reference aids' % (len(reference_aids)))
        available_aids = ut.setdiff_ordered(available_aids, reference_aids)

    if aidcfg['sample_per_name'] is not None:
        if VERB_MAIN_HELPERS:
            print(' * Filtering sample_per_name=%r using rule %r' % (aidcfg['sample_per_name'], aidcfg['sample_rule'] ))

        if aidcfg['sample_rule'] == 'ref_max_timedelta':
            available_aids = reference_sample_per_name(ibs, available_aids, aidcfg, reference_aids)
        else:
            # For the query we just choose a single annot per name
            # For the database we have to do something different
            available_aids = ibs.get_annot_rowid_sample(
                available_aids, per_name=aidcfg['sample_per_name'], min_gt=None,
                method=aidcfg['sample_rule'], offset=aidcfg['sample_offset'], seed=0)

    if aidcfg['sample_size'] is not None:
        # TODO:
        # Allow removal of multitons if reference_aids is not given
        # Randomly sample which annots are removed
        if reference_aids is not None:
            # Enesure that the sampleing does not conflict with reference aid properties
            if VERB_MAIN_HELPERS:
                print(' * Filtering to sample size %r' % (aidcfg['sample_size'],))
            assert reference_aids is not None and len(reference_aids) > 0
            ref_multi, avl_multi, ref_single, avl_single = ibs.partition_annots_into_corresponding_groups(reference_aids, available_aids)
            #with ut.embed_on_exception_context:
            assert len(ref_single) == 0, 'should not have uncorresponding refs'
            #set(ibs.get_annot_name_rowids(available_aids)).intersection( set(ibs.get_annot_name_rowids(ref_single)) )
            #singletons, multitons = ibs.partition_annots_into_singleton_multiton(available_aids)

            # We must keep all multitons because they corresopnd with the reference set
            multitons = ut.flatten(avl_multi)
            # We have the option of keeping singletons
            singletons = avl_single
            num_single = len(singletons)
            num_multi = len(multitons)
            assert num_single + num_multi == len(available_aids), 'does not sum'
            num_keep_single = aidcfg['sample_size'] - num_multi
            num_remove_single = num_single - num_keep_single
            if num_remove_single < 0:
                # Too few singletons
                print('Warning: Cannot meet sample_size=%r. available_%saids will be undersized by at least %d' % (aidcfg['sample_size'], prefix, -num_remove_single,))
            if num_keep_single < 0:
                # Too many multitons; Can never remove a multiton
                print('Warning: Cannot meet sample_size=%r. available_%saids will be oversized by at least %d' % (aidcfg['sample_size'], prefix, -num_keep_single,))
            singletons = ut.random_sample(singletons, num_keep_single, seed=42)
            available_aids = multitons + singletons
        else:
            # No reference aids. Can remove freely.
            if aidcfg['sample_size'] > available_aids:
                print('Warning sample size too large')
            available_aids = ut.random_sample(available_aids, aidcfg['sample_size'], seed=42)

    if VERYVERB_MAIN_HELPERS:
        with ut.Indenter('   '):
            ut.print_dict(ibs.get_annot_stats_dict(available_aids, prefix=prefix), dict_name=prefix + 'aid_stats')

    # ---- SUBINDEXING STEP
    if VERB_MAIN_HELPERS:
        print(' * SAMPLE: len(available_%saids) = %r\n' % (prefix, len(available_aids)))

    return available_aids


@profile
def subindex_avaiable_aids(ibs, available_aids, aidcfg, reference_aids=None, prefix=''):
    if VERB_MAIN_HELPERS:
        print(' * [SUBINDEX %sAIDS]' % (prefix.upper(),))
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
        print(' * Chose subset of size %d/%d' % (len(_indexed_aids), len(available_aids)))
        available_aids = _indexed_aids

    if VERB_MAIN_HELPERS:
        print(' * SUBINDEX: len(available_%saids) = %r\n' % (prefix, len(available_aids)))
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
