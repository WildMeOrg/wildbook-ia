# -*- coding: utf-8 -*-
"""
TODO: sort annotations at the end of every step
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import six
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[main_helpers]')


VERB_TESTDATA, VERYVERB_TESTDATA = ut.get_verbflag('testdata', 'td')


# TODO: Make these configurable
SEED1 = 0
SEED2 = 42

USE_ACFG_CACHE = not ut.get_argflag(('--nocache-annot', '--nocache-aid', '--nocache')) and ut.USE_CACHE


@profile
def testdata_single_acfg(ibs, default_options=''):
    r"""
    CommandLine:
        python -m ibeis.init.filter_annots --exec-testdata_single_acfg --verbtd --db PZ_ViewPoints
        python -m ibeis.init.filter_annots --exec-testdata_single_acfg --verbtd --db NNP_Master3 -a is_known=True,view_pername='#primary>0&#primary1>=1'

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.init.filter_annots import *  # NOQA
        >>> from ibeis.experiments import annotation_configs
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_ViewPoints')
        >>> default_options = ''
        >>> aidcfg, aids = testdata_single_acfg(ibs, default_options)
        >>> print('\n RESULT:')
        >>> annotation_configs.print_acfg(aidcfg, aids, ibs, per_name_vpedge=None)
    """
    from ibeis.experiments import annotation_configs
    from ibeis.experiments import experiment_helpers
    cfgstr_options = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=str, default=default_options)
    base_cfg = annotation_configs.single_default
    aidcfg_combo = experiment_helpers.customize_base_cfg('default', cfgstr_options, base_cfg, 'aids', alias_keys=annotation_configs.ALIAS_KEYS)
    aidcfg = aidcfg_combo[0]
    if len(aidcfg_combo) > 1:
        raise AssertionError('Error: combinations not handled for single cfg setting')
    aids = expand_single_acfg(ibs, aidcfg)
    return aidcfg, aids


def expand_single_acfg(ibs, aidcfg, verbose=VERB_TESTDATA):
    from ibeis.experiments import annotation_configs
    if verbose:
        print('+=== EXPAND_SINGLE_ACFG ===')
        print(' * acfg = %s' % (ut.dict_str(annotation_configs.compress_aidcfg(aidcfg), align=True),))
        print('+---------------------')
    available_aids = expand_to_default_aids(ibs, aidcfg)
    available_aids = filter_independent_properties(ibs, available_aids, aidcfg)
    available_aids = sample_available_aids(ibs, available_aids, aidcfg)
    available_aids = subindex_avaiable_aids(ibs, available_aids, aidcfg)
    aids = available_aids
    if verbose:
        print('L___ EXPAND_SINGLE_ACFG ___')
    return aids


def expand_acfgs_consistently(ibs, acfg_combo):
    """
    python -m ibeis.experiments.experiment_helpers --exec-parse_acfg_combo_list  -a varysize
    python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_Master1 -a varysize
    python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_Master1 -a varysize:qsize=None
    python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_Master0 --nofilter-dups  -a varysize
    python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_MTEST -a varysize --nofilter-dups
    python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_Master0 --verbtd --nofilter-dups  -a varysize
    python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list -a viewpoint_compare --db PZ_Master1 --verbtd --nofilter-dups

    """
    # Edit configs so the sample sizes are consistent
    # FIXME: requiers that smallest configs are specified first

    def tmpmin(a, b):
        if a is None:
            return b
        elif b is None:
            return a
        else:
            return min(a, b)
    expanded_aids_list = []

    # Keep track of seen samples
    min_qsize = None
    min_dsize = None

    for acfg in acfg_combo:
        qcfg = acfg['qcfg']
        dcfg = acfg['dcfg']

        # In some cases we may want to clamp these, but others we do not
        if qcfg['force_const_size']:
            qcfg['_orig_sample_size'] = qcfg['sample_size']
            qcfg['sample_size'] = tmpmin(qcfg['sample_size'] , min_qsize)
        if dcfg['force_const_size']:
            dcfg['_orig_sample_size'] = dcfg['sample_size']
            dcfg['sample_size'] = tmpmin(dcfg['sample_size'] , min_dsize)

        # Expand modified acfgdict
        expanded_aids = expand_acfgs(ibs, acfg)

        qsize = len(expanded_aids[0])
        dsize = len(expanded_aids[1])

        if min_qsize is None:
            qcfg['sample_size'] = qsize
            dcfg['sample_size'] = dsize

        if qcfg['sample_size'] != qsize:
            qcfg['_true_sample_size'] = qsize
        if dcfg['sample_size'] != dsize:
            dcfg['_true_sample_size'] = dsize

        if qcfg['force_const_size']:
            min_qsize = tmpmin(min_qsize, qsize)
            min_dsize = tmpmin(min_dsize, dsize)

        #ibs.print_annotconfig_stats(*expanded_aids)
        expanded_aids_list.append(expanded_aids)
    return list(zip(acfg_combo, expanded_aids_list))


@profile
def expand_acfgs(ibs, aidcfg, verbose=VERB_TESTDATA, use_cache=USE_ACFG_CACHE):
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

        python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list:0 --db NNP_Master3 -a viewpoint_compare --nocache-aid --verbtd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.init.filter_annots import *  # NOQA
    """
    from ibeis.experiments import annotation_configs
    from ibeis.experiments import cfghelpers
    qcfg = aidcfg['qcfg']
    dcfg = aidcfg['dcfg']

    if use_cache:
        # Make loading aids a big faster for experiments
        acfg_cachedir = './ACFG_CACHE'
        acfg_cachename = 'ACFG_CACHE'

        RESPECT_INTERNAL_CFGS = False
        if RESPECT_INTERNAL_CFGS:
            aid_cachestr = ibs.get_dbname() + '_' + ut.hashstr27(ut.to_json(aidcfg))
        else:
            import copy
            relevant_aidcfg = copy.deepcopy(aidcfg)
            ut.delete_dict_keys(relevant_aidcfg['qcfg'], cfghelpers.INTERNAL_CFGKEYS)
            ut.delete_dict_keys(relevant_aidcfg['dcfg'], cfghelpers.INTERNAL_CFGKEYS)
            aid_cachestr = ibs.get_dbname() + '_' + ut.hashstr27(ut.to_json(relevant_aidcfg))
        try:
            (qaid_list, daid_list) = ut.load_cache(acfg_cachedir, acfg_cachename, aid_cachestr)
        except IOError:
            pass
        else:
            return qaid_list, daid_list

    # ---- INCLUDING STEP
    if verbose:
        print('+=== EXPAND_ACFGS ===')
        print(' * acfg = %s' % (ut.dict_str(annotation_configs.compress_aidcfg(aidcfg), align=True),))
        print('+---------------------')

    try:
        # Can probably move these commands around
        with ut.Indenter('[Q] '):
            available_qaids = expand_to_default_aids(ibs, qcfg, prefix='q', verbose=verbose)
            available_qaids = filter_independent_properties(ibs, available_qaids, qcfg, prefix='q', verbose=verbose)
            available_qaids = sample_available_aids(ibs, available_qaids, qcfg, prefix='q', verbose=verbose)  # No reference sampling for query

        with ut.Indenter('[D] '):
            available_daids = expand_to_default_aids(ibs, dcfg, prefix='d', verbose=verbose)
            available_daids = filter_independent_properties(ibs, available_daids, dcfg, prefix='d', verbose=verbose)
            available_daids = reference_sample_available_aids(ibs, available_daids, dcfg, reference_aids=available_qaids, prefix='d', verbose=verbose)

        #available_qaids = filter_reference_properties(ibs, available_qaids, qcfg, reference_aids=available_daids, prefix='q')
        #available_daids = filter_reference_properties(ibs, available_daids, dcfg, reference_aids=available_qaids, prefix='d')

        with ut.Indenter('[Q] '):
            available_qaids = subindex_avaiable_aids(ibs, available_qaids, qcfg, prefix='q', verbose=verbose)
        with ut.Indenter('[D] '):
            available_daids = subindex_avaiable_aids(ibs, available_daids, dcfg, prefix='d', verbose=verbose)
    except Exception as ex:
        print('PRINTING ERROR INFO')
        print(' * acfg = %s' % (ut.dict_str(annotation_configs.compress_aidcfg(aidcfg), align=True),))
        ut.printex(ex, 'Error expanding acfgs')
        raise

    qaid_list = available_qaids
    daid_list = available_daids

    if verbose:
        print('L___ EXPAND_ACFGS ___')
        ibs.print_annotconfig_stats(qaid_list, daid_list)

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
    if use_cache:
        ut.ensuredir(acfg_cachedir)
        ut.save_cache(acfg_cachedir, acfg_cachename, aid_cachestr, (qaid_list, daid_list))
    #available_qaids qcfg['ref_has_viewpoint']
    return qaid_list, daid_list


@profile
def expand_to_default_aids(ibs, aidcfg, prefix='', verbose=VERB_TESTDATA):
    default_aids = aidcfg['default_aids']

    if verbose:
        print(' * [INCLUDE %sAIDS]' % (prefix.upper()))
        #print(' * PARSING %saidcfg = %s' % (prefix, ut.dict_str(aidcfg, align=True),))
        print(' * default_%saids = %s' % (prefix, ut.obj_str(default_aids, truncate=True, nl=False)))

    if isinstance(default_aids, six.string_types):
        #if verbose:
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
        if verbose:
            print(' ... default %saids specified.' % (prefix,))

    #if aidcfg['include_aids'] is not None:
    #    raise NotImplementedError('Implement include_aids')

    available_aids = default_aids

    if len(available_aids) == 0:
        print(' WARNING no %s annotations available' % (prefix,))

    #if aidcfg['exclude_aids'] is not None:
    #    if verbose:
    #        print(' * Excluding %d custom aids' % (len(aidcfg['exclude_aids'])))
    #    available_aids = ut.setdiff_ordered(available_aids, aidcfg['exclude_aids'])

    if verbose:
        print(' * HAHID: ' + ibs.get_annot_hashid_semantic_uuid(available_aids, prefix=prefix.upper()))
        print(' * DEFAULT: len(available_%saids)=%r\n' % (prefix, len(available_aids)))
    return available_aids


@profile
def filter_independent_properties(ibs, available_aids, aidcfg, prefix='', verbose=VERB_TESTDATA):
    """ Filtering that doesn't have to do with a reference set of aids """
    from ibeis import ibsfuncs
    if verbose:
        print(' * [FILTER INDEPENDENT %sAIDS]' % (prefix.upper()))
        if VERYVERB_TESTDATA:
            with ut.Indenter('  '):
                ibs.print_annot_stats(available_aids, prefix, per_name_vpedge=None)

    if aidcfg['is_known'] is True:
        if verbose:
            print(' * Removing annots without names')
        available_aids = ibs.filter_aids_without_name(available_aids)

    if aidcfg['require_timestamp'] is True:
        if verbose:
            print(' * Removing annots without timestamp')
        available_aids = ibs.filter_aids_without_timestamps(available_aids)

    species = None
    if aidcfg['species'] is not None:
        if aidcfg['species'] == 'primary':
            #if verbose:
            #    print(' * Finiding primary species')
            species = ibs.get_primary_database_species()
        else:
            species = aidcfg['species']
        if verbose:
            print(' * Filtering to species=%r' % (species,))
        available_aids = ibs.filter_aids_to_species(available_aids, species)

    if aidcfg['minqual'] is not None or aidcfg['require_quality']:
        # Resolve quality
        if aidcfg['minqual'] is None:
            minqual = 'junk'
        else:
            minqual = aidcfg['minqual']
        if verbose:
            print(' * Filtering quality. minqual=%r, require_quality=%r'
                  % (minqual, aidcfg['require_quality']))
        # Filter quality
        available_aids = ibs.filter_aids_to_quality(available_aids, minqual, unknown_ok=not aidcfg['require_quality'])

    #aidcfg['viewpoint_edge_counts'] = None
    #if aidcfg['viewpoint_edge_counts'] is not None:
    #    getter_list = [ibs.get_annot_name_rowids, ibs.get_annot_yaw_texts]
    #    nid2_vp2_aids = ibs.group_annots_by_multi_prop(available_aids, getter_list)
    #    #assert len(available_aids) == len(list(ut.iflatten_dict_values(nid2_vp2_aids)))
    #    nid2_vp2_aids = ut.hierarchical_map_vals(ut.identity, nid2_vp2_aids)  # remove defaultdict structure

    #    nid2_num_vp = ut.hierarchical_map_vals(len, nid2_vp2_aids, max_depth=0)

    #    min_num_vp_pername = 2
    #    def has_required_vps(vp2_aids):
    #        min_examples_per_vp = {
    #            'frontleft': 1,
    #            'left': 2,
    #        }
    #        return all([key in vp2_aids and vp2_aids[key] for key in min_examples_per_vp])

    #    nids_with_multiple_vp = [key for  key, val in nid2_num_vp.items() if val >= min_num_vp_pername]
    #    subset1_nid2_vp2_aids = ut.dict_subset(nid2_vp2_aids, nids_with_multiple_vp)
    #    subset2_flags = ut.hierarchical_map_vals(has_required_vps, nid2_vp2_aids, max_depth=0)
    #    subset2_nids = ut.list_compress(subset2_flags.keys(), subset2_flags.values())
    #    subset2_nid2_vp2_aids = ut.dict_subset(subset1_nid2_vp2_aids, subset2_nids)
    #    available_aids = list(ut.iflatten_dict_values(subset2_nid2_vp2_aids))

    if aidcfg['view_pername'] is not None:
        # the avaiable aids must be from names with certain viewpoint frequency properties
        prop2_nid2_aids = ibs.group_annots_by_prop_and_name(available_aids, ibs.get_annot_yaw_texts)
        countstr = aidcfg['view_pername']
        primary_viewpoint = ibsfuncs.get_primary_species_viewpoint(species)

        if verbose:
            print(' * [FILTER %sAIDS VIEWPOINT COUNTS WITH countstr=%s]' % (prefix.upper(), countstr))

        lhs_dict = {
            'primary': primary_viewpoint,
            'primary1': ibsfuncs.get_extended_viewpoints(primary_viewpoint, num1=1, num2=0, include_base=False)[0]
        }
        self = CountstrParser(lhs_dict, prop2_nid2_aids)
        nid2_flag = self.parse_countstr_expr(countstr)
        nid2_aids = ibs.group_annots_by_name_dict(available_aids)
        valid_nids = [nid for nid, flag in nid2_flag.items() if flag]
        available_aids = ut.flatten(ut.dict_take(nid2_aids, valid_nids))

    if aidcfg['view'] is not None or aidcfg['require_viewpoint']:
        # Resolve base viewpoint
        if aidcfg['view'] == 'primary':
            view = ibsfuncs.get_primary_species_viewpoint(species)
        elif aidcfg['view'] == 'primary1':
            view = ibsfuncs.get_primary_species_viewpoint(species, 1)
        else:
            view = aidcfg['view']
        view_ext1 = aidcfg['view_ext'] if aidcfg['view_ext1'] is None else aidcfg['view_ext1']
        view_ext2 = aidcfg['view_ext'] if aidcfg['view_ext2'] is None else aidcfg['view_ext2']
        valid_yaws = ibsfuncs.get_extended_viewpoints(view, num1=view_ext1, num2=view_ext2)
        if verbose:
            print(' * Filtering viewpoint. valid_yaws=%r, require_viewpoint=%r'
                  % (valid_yaws, aidcfg['require_viewpoint']))
        # Filter viewpoint
        available_aids = ibs.filter_aids_to_viewpoint(available_aids, valid_yaws, unknown_ok=not aidcfg['require_viewpoint'])

    # Each aid must have at least this number of other groundtruth aids
    gt_min_per_name = aidcfg['gt_min_per_name']
    if gt_min_per_name is not None:
        if verbose:
            print(' * Filtering gt_min_per_name=%d' % (gt_min_per_name))
        grouped_aids_, unique_nids = ibs.group_annots_by_name(available_aids, distinguish_unknowns=True)
        available_aids = ut.flatten([aids for aids in grouped_aids_ if len(aids) >= gt_min_per_name])

    if verbose:
        print(' * HAHID: ' + ibs.get_annot_hashid_semantic_uuid(available_aids, prefix=prefix.upper()))
        print(' * I-FILTERED: len(available_%saids)=%r\n' % (prefix, len(available_aids)))

    return available_aids


@profile
def filter_reference_properties(ibs, available_aids, aidcfg, reference_aids, prefix='', verbose=VERB_TESTDATA):
    """
    DEPRICATE
    """
    from ibeis import ibsfuncs
    import functools
    if verbose:
        print(' * [FILTER REFERENCE %sAIDS]' % (prefix.upper()))
        if VERYVERB_TESTDATA:
            with ut.Indenter('  '):
                ibs.print_annot_stats(available_aids, prefix, per_name_vpedge=None)

    if aidcfg['ref_has_viewpoint'] is not None:
        print(' * Filtering such that %saids has refs with viewpoint=%r' % (prefix, aidcfg['ref_has_viewpoint']))
        species = ibs.get_primary_database_species(available_aids)
        if aidcfg['ref_has_viewpoint']  == 'primary':
            valid_yaws = [ibsfuncs.get_primary_species_viewpoint(species)]
        elif aidcfg['ref_has_viewpoint']  == 'primary1':
            valid_yaws = [ibsfuncs.get_primary_species_viewpoint(species, 1, 1)]
        gt_ref_grouped_aids, gt_avl_grouped_aids, gf_ref_grouped_aids, gf_avl_grouped_aids = ibs.partition_annots_into_corresponding_groups(reference_aids, available_aids)
        # Filter to only available aids that have a reference with specified viewpoint
        is_valid_yaw = functools.partial(ibs.get_viewpoint_filterflags, valid_yaws=valid_yaws)
        multi_flags = list(map(any, ibs.unflat_map(is_valid_yaw, gt_ref_grouped_aids)))
        available_aids = ut.flatten(ut.list_compress(gt_avl_grouped_aids, multi_flags))

    if verbose:
        print(' * HAHID: ' + ibs.get_annot_hashid_semantic_uuid(available_aids, prefix=prefix.upper()))
        print(' * R-FILTERED: len(available_%saids)=%r\n' % (prefix, len(available_aids)))

    return available_aids


def get_reference_preference_order(ibs, gt_ref_grouped_aids, gt_avl_grouped_aids, prop_getter, cmp_func, aggfn, rng, verbose=VERB_TESTDATA):
    """
    Orders preference for sampling based on some metric

    # TODO: generalize this part
    #avl_prop = grouped_available_gt_props[0]
    #ref_prop = grouped_reference_props[0]
    #ref_prop = np.array([0, 10])
    #avl_prop = np.array([5, 7, 30, 10])
    #aggfn = np.product #aggfn = np.max #aggfn = np.mean
    """

    grouped_reference_unixtimes = ibs.unflat_map(prop_getter, gt_ref_grouped_aids)
    grouped_available_gt_unixtimes = ibs.unflat_map(prop_getter, gt_avl_grouped_aids)

    grouped_reference_props = grouped_reference_unixtimes
    grouped_available_gt_props = grouped_available_gt_unixtimes

    # Order the available aids by some aggregation over some metric
    preference_scores = [aggfn(cmp_func(ref_prop, avl_prop[:, None]), axis=1)
                         for ref_prop, avl_prop in zip(grouped_reference_props, grouped_available_gt_props)]

    # Order by increasing timedelta (metric)
    reverse = True
    # replace nan with -inf, or inf randomize order between equal values
    if reverse:
        for scores in preference_scores:
            scores[np.isnan(scores)] = -np.inf
        gt_preference_idx_list = [np.lexsort((scores, rng.rand(len(scores))))[::-1] for scores in preference_scores]
    else:
        # Smaller is better, replace nan with inf
        for scores in preference_scores:
            scores[np.isnan(scores)] = np.inf
        gt_preference_idx_list = [np.lexsort((scores, rng.rand(len(scores)))) for scores in preference_scores]
    return gt_preference_idx_list


@profile
def reference_sample_available_aids(ibs, available_aids, aidcfg, reference_aids, prefix='', verbose=VERB_TESTDATA):
    if verbose:
        print(' * [SAMPLE %sAIDS (REF)]' % (prefix.upper(),))
        if VERYVERB_TESTDATA:
            if reference_aids is not None:
                with ut.Indenter('  '):
                    ibs.print_annotconfig_stats(reference_aids, available_aids)

    #if len(available_aids) == 0:
    #    return available_aids

    sample_per_name     = aidcfg['sample_per_name']
    sample_per_ref_name = aidcfg['sample_per_ref_name']
    exclude_reference   = aidcfg['exclude_reference']
    sample_size         = aidcfg['sample_size']
    offset              = aidcfg['sample_offset']
    sample_rule_ref     = aidcfg['sample_rule_ref']
    sample_rule         = aidcfg['sample_rule']
    gt_avl_aids         = aidcfg['gt_avl_aids']

    if sample_per_ref_name is None:
        sample_per_ref_name = sample_per_name

    #assert reference_aids is not None and len(reference_aids) > 0

    if exclude_reference is not None:
        assert reference_aids is not None, 'reference_aids=%r' % (reference_aids,)
        if verbose:
            print(' * Excluding %d reference aids' % (len(reference_aids)))
            if VERYVERB_TESTDATA:
                with ut.Indenter('  '):
                    ibs.print_annot_stats(available_aids, prefix, per_name_vpedge=None)
                with ut.Indenter('  '):
                    ibs.print_annotconfig_stats(reference_aids, available_aids)
        available_aids = ut.setdiff_ordered(available_aids, reference_aids)

    if gt_avl_aids is not None:
        if verbose:
            print(' * Excluding gt_avl_aids custom specified by name')
        # Pick out the annotations that do not belong to the same name as the given gt_avl_aids
        complement = np.setdiff1d(available_aids, gt_avl_aids)
        partitioned_sets = ibs.partition_annots_into_corresponding_groups(gt_avl_aids, complement)
        assert len(set(ut.flatten((ibs.unflat_map(ibs.get_annot_name_rowids, partitioned_sets[1])))).intersection(set(ut.flatten((ibs.unflat_map(ibs.get_annot_name_rowids, partitioned_sets[3])))))) == 0
        gf_avl_aids = (ut.flatten(partitioned_sets[3]))
        assert len(set(ibs.get_annot_name_rowids(reference_aids)).intersection(set(ibs.get_annot_name_rowids(gf_avl_aids) ))) == 0
        #ibs.get_annot_groundtruth(reference_aids, daid_list=gf_avl_aids)
        #ibs.get_annot_groundtruth(reference_aids, daid_list=gt_avl_aids)
        #ibs.get_annot_groundtruth(reference_aids, daid_list=available_aids)
        available_aids = np.hstack([gt_avl_aids, gf_avl_aids])
        available_aids.sort()
        available_aids = available_aids.tolist()
        #sorted(gt_avl_aids) == sorted(ut.flatten(partitioned_sets[0]))

    if not (sample_per_ref_name is not None or sample_size is not None):
        return available_aids

    # This function first partitions aids into a one set that corresonds with
    # the reference set and another that does not correspond with the reference
    # set. The rest of the filters operate on these sets independently
    partitioned_sets = ibs.partition_annots_into_corresponding_groups(reference_aids, available_aids)
    # gt_ref_grouped_aids, and gt_avl_grouped_aids are corresponding lists of anotation groups
    # gf_ref_grouped_aids, and gf_avl_grouped_aids are uncorresonding annotations groups
    (gt_ref_grouped_aids, gt_avl_grouped_aids,
     gf_ref_grouped_aids, gf_avl_grouped_aids) = partitioned_sets

    if sample_per_ref_name is not None:
        if verbose:
            print(' * Filtering gt-ref sample_per_ref_name=%r, sample_rule_ref=%r with reference'
                  % (sample_per_ref_name, sample_rule_ref))
        rng = np.random.RandomState(SEED2)
        if sample_rule_ref == 'max_timedelta':
            # Maximize time delta between query and corresponding database annotations
            cmp_func, aggfn, prop_getter = ut.absdiff, np.mean, ibs.get_annot_image_unixtimes_asfloat
            gt_preference_idx_list = get_reference_preference_order(
                ibs, gt_ref_grouped_aids, gt_avl_grouped_aids, prop_getter,
                cmp_func, aggfn, rng)
        elif sample_rule_ref == 'random':
            gt_preference_idx_list = [ut.random_indexes(len(aids), rng=rng) for aids in gt_avl_grouped_aids]
        else:
            raise ValueError('Unknown sample_rule_ref = %r' % (sample_rule_ref,))
        gt_sample_idxs_list = ut.get_list_column_slice(gt_preference_idx_list, offset, offset + sample_per_ref_name)
        gt_sample_aids = ut.list_ziptake(gt_avl_grouped_aids, gt_sample_idxs_list)
        gt_avl_grouped_aids = gt_sample_aids

    if sample_per_name is not None:
        if verbose:
            print(' * Filtering gf-ref %saids sample_per_name=%r'
                  % (prefix, sample_per_name))
        # sample rule is always random for gf right now
        rng = np.random.RandomState(SEED2)
        if sample_rule == 'random':
            gf_preference_idx_list = [ut.random_indexes(len(aids), rng=rng) for aids in gf_avl_grouped_aids]
        else:
            raise ValueError('Unknown sample_rule=%r' % (sample_rule,))
        gf_sample_idxs_list = ut.get_list_column_slice(gf_preference_idx_list, offset, offset + sample_per_name)
        gf_sample_aids = ut.list_ziptake(gf_avl_grouped_aids, gf_sample_idxs_list)
        gf_avl_grouped_aids = gf_sample_aids

    gt_avl_aids = ut.flatten(gt_avl_grouped_aids)
    gf_avl_aids = ut.flatten(gf_avl_grouped_aids)

    if sample_size is not None:
        if verbose:
            print(' * Filtering %saids to sample_size=%r' % (prefix, sample_size,))
        # Keep all correct matches to the reference set
        # We have the option of keeping ground false
        num_gt = len(gt_avl_aids)
        num_gf = len(gf_avl_aids)
        num_keep_gf = sample_size - num_gt
        num_remove_gf = num_gf - num_keep_gf
        if num_remove_gf < 0:
            # Too few ground false
            print('Warning: Cannot meet sample_size=%r. available_%saids will be undersized by at least %d' % (sample_size, prefix, -num_remove_gf,))
        if num_keep_gf < 0:
            # Too many multitons; Can never remove a multiton
            print('Warning: Cannot meet sample_size=%r. available_%saids will be oversized by at least %d' % (sample_size, prefix, -num_keep_gf,))
        rng = np.random.RandomState(SEED2)
        gf_avl_aids = ut.random_sample(gf_avl_aids, num_keep_gf, rng=rng)

    # random ordering makes for bad hashes
    available_aids = sorted(gt_avl_aids + gf_avl_aids)
    # ---- SUBINDEXING STEP
    if verbose:
        print(' * HAHID: ' + ibs.get_annot_hashid_semantic_uuid(available_aids, prefix=prefix.upper()))
        print(' * REF-SAMPLE: len(available_%saids) = %r\n' % (prefix, len(available_aids)))

    return available_aids


@profile
def sample_available_aids(ibs, available_aids, aidcfg, prefix='', verbose=VERB_TESTDATA):
    """
    python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list:0 --db NNP_Master3 -a viewpoint_compare --nocache-aid --verbtd

    python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --a controlled:qoffset=2,dsample_rule_ref=max_timedelta,dsize=200
    python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --a controlled:qoffset=2,dsample_rule_ref=max_timedelta,dsize=10
    python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --a controlled:qoffset=2,dsample_rule_ref=max_timedelta,dsize=41,dper_name=2

    CommandLine:
        python -m ibeis.init.filter_annots --exec-sample_available_aids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.init.filter_annots import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_ViewPoints')
        >>> available_aids = ibs.get_valid_aids()
        >>> aidcfg = '?'
        >>> reference_aids = None
        >>> prefix = ''
        >>> available_aids = sample_available_aids(ibs, available_aids, aidcfg, reference_aids, prefix)
        >>> result = ('available_aids = %s' % (str(available_aids),))
        >>> print(result)
    """
    if verbose:
        print(' * [SAMPLE %sAIDS (NOREF)]' % (prefix.upper(),))
        if VERYVERB_TESTDATA:
            with ut.Indenter('   '):
                ut.print_dict(ibs.get_annot_stats_dict(available_aids, prefix=prefix), dict_name=prefix + 'aid_presample_stats')

    sample_rule     = aidcfg['sample_rule']
    sample_per_name = aidcfg['sample_per_name']
    sample_size     = aidcfg['sample_size']
    offset          = aidcfg['sample_offset']

    if sample_per_name is not None:
        # For the query we just choose a single annot per name
        # For the database we have to do something different
        grouped_aids = ibs.group_annots_by_name(available_aids)[0]
        # Order based on some preference (like random)
        rng = np.random.RandomState(SEED1)
        if sample_rule == 'random':
            preference_idxs_list = [ut.random_indexes(len(aids), rng=rng) for aids in grouped_aids]
        else:
            raise ValueError('Unknown sample_rule=%r' % (sample_rule,))
        sample_idxs_list = ut.get_list_column_slice(preference_idxs_list, offset, offset + sample_per_name)
        sample_aids = ut.list_ziptake(grouped_aids, sample_idxs_list)
        available_aids = ut.flatten(sample_aids)

    if sample_size is not None:
        if verbose:
            print(' * Filtering to sample size %r' % (sample_size,))
        if sample_size > available_aids:
            print('Warning sample size too large')
        rng = np.random.RandomState(SEED2)
        available_aids = ut.random_sample(available_aids, sample_size, rng=rng)

    if VERYVERB_TESTDATA:
        with ut.Indenter('   '):
            ut.print_dict(ibs.get_annot_stats_dict(available_aids, prefix=prefix), dict_name=prefix + 'aid_postsample_stats')

    # ---- SUBINDEXING STEP
    if verbose:
        print(' * HAHID: ' + ibs.get_annot_hashid_semantic_uuid(available_aids, prefix=prefix.upper()))
        print(' * SAMPLE: len(available_%saids) = %r\n' % (prefix, len(available_aids)))

    return available_aids


@profile
def subindex_avaiable_aids(ibs, available_aids, aidcfg, reference_aids=None, prefix='', verbose=VERB_TESTDATA):
    if verbose:
        print(' * [SUBINDEX %sAIDS]' % (prefix.upper(),))
    #ut.get_argval('--qshuffle')

    if aidcfg['shuffle']:
        if verbose:
            print(' * Shuffling with seed=%r' % (SEED2))
        # Determenistic shuffling
        available_aids = ut.list_take(available_aids, ut.random_indexes(len(available_aids), seed=SEED2))

    if aidcfg['index'] is not None:
        if verbose:
            print(' * Indexing')
        indicies = ensure_flatlistlike(aidcfg['index'])
        _indexed_aids = [available_aids[ix] for ix in indicies if ix < len(available_aids)]
        print(' * Chose subset of size %d/%d' % (len(_indexed_aids), len(available_aids)))
        available_aids = _indexed_aids

    # Always sort aids to preserve hashes? (Maybe sort the vuuids instead)
    available_aids = sorted(available_aids)

    if verbose:
        print(' * HAHID: ' + ibs.get_annot_hashid_semantic_uuid(available_aids, prefix=prefix.upper()))
        print(' * SUBINDEX: len(available_%saids) = %r\n' % (prefix, len(available_aids)))
    return available_aids


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


import operator


@six.add_metaclass(ut.ReloadingMetaclass)
class CountstrParser(object):
    numop = '#'
    compare_op_map = {
        '<'  : operator.lt,
        '<=' : operator.le,
        '>'  : operator.gt,
        '>=' : operator.ge,
        '='  : operator.eq,
        '!=' : operator.ne,
    }

    def __init__(self, lhs_dict, prop2_nid2_aids):
        self.lhs_dict = lhs_dict
        self.prop2_nid2_aids = prop2_nid2_aids
        pass

    def parse_countstr_binop(self, part):
        import utool as ut
        import re
        # Parse binary comparison operation
        left, op, right = re.split(ut.regex_or(('[<>]=?', '=')), part)
        # Parse length operation. Get prop_left_nids, prop_left_values
        if left.startswith(self.numop):
            varname = left[len(self.numop):]
            # Parse varname
            prop = self.lhs_dict.get(varname, varname)
            # Apply length operator to each name with the prop
            prop_left_nids = self.prop2_nid2_aids.get(prop, {}).keys()
            prop_left_values = np.array(list(map(len, self.prop2_nid2_aids.get(prop, {}).values())))
        # Pares number
        if right:
            prop_right_value = int(right)
        # Execute comparison
        prop_binary_result = self.compare_op_map[op](prop_left_values, prop_right_value)
        prop_nid2_result = dict(zip(prop_left_nids, prop_binary_result))
        return prop_nid2_result

    def parse_countstr_expr(self, countstr):
        # Split over ands for now
        and_parts = countstr.split('&')
        prop_nid2_result_list = []
        for part in and_parts:
            prop_nid2_result = self.parse_countstr_binop(part)
            prop_nid2_result_list.append(prop_nid2_result)
        # change to dict_union when parsing ors
        import functools
        expr_nid2_result = reduce(functools.partial(ut.dict_intersection, combine=True, combine_op=operator.and_), prop_nid2_result_list)
        return expr_nid2_result
        #reduce(functools.partial(ut.dict_union3, combine_op=operator.or_), prop_nid2_result_list)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.init.filter_annots
        python -m ibeis.init.filter_annots --allexamples
        python -m ibeis.init.filter_annots --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
