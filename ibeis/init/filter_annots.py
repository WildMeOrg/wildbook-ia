# -*- coding: utf-8 -*-
"""
TODO: sort annotations at the end of every step
"""
from __future__ import absolute_import, division, print_function
import operator
import utool as ut
import numpy as np
import functools
import six
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[main_helpers]')


VERB_TESTDATA, VERYVERB_TESTDATA = ut.get_verbflag('testdata', 'td')


# TODO: Make these configurable
SEED1 = 0
SEED2 = 42

if ut.is_developer():
    USE_ACFG_CACHE = not ut.get_argflag(('--nocache-annot', '--nocache-aid', '--nocache')) and ut.USE_CACHE
else:
    USE_ACFG_CACHE = False


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
    from ibeis.experiments import cfghelpers
    cfgstr_options = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=str, default=default_options)
    base_cfg = annotation_configs.single_default
    aidcfg_combo = cfghelpers.customize_base_cfg('default', cfgstr_options, base_cfg, 'aids', alias_keys=annotation_configs.ALIAS_KEYS)
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
    #avail_aids = expand_to_default_aids(ibs, aidcfg)
    avail_aids = ut._get_all_aids()
    avail_aids = filter_annots_independent(ibs, avail_aids, aidcfg)
    avail_aids = sample_annots(ibs, avail_aids, aidcfg)
    avail_aids = subindex_annots(ibs, avail_aids, aidcfg)
    aids = avail_aids
    if verbose:
        print('L___ EXPAND_SINGLE_ACFG ___')
    return aids


def expand_acfgs_consistently(ibs, acfg_combo):
    """
    CommandLine:
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

    # HACK: Find out the params being varied and disallow those from being
    # prefiltered due to the lack of heirarchical filters
    from ibeis.experiments import annotation_configs
    nonvaried_dict, varied_acfg_list = annotation_configs.partition_acfg_list(acfg_combo)
    hack_exclude_keys = list(set(ut.flatten([list(ut.merge_dicts(*acfg.values()).keys()) for acfg in varied_acfg_list])))

    for combox, acfg in enumerate(acfg_combo):
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
        expanded_aids = expand_acfgs(ibs, acfg, hack_exclude_keys=hack_exclude_keys)

        if dcfg.get('hack_extra', None):
            # SUCH HACK to get a larger database
            _aidcfg = annotation_configs.default['dcfg']
            _aidcfg['sample_per_name'] = 1
            _aidcfg['sample_size'] = 500
            _aidcfg['min_pername'] = 1
            _aidcfg['require_viewpoint'] = True
            _aidcfg['exclude_reference'] = True
            _aidcfg['view'] = 'right'
            prefix = 'hack'
            qaids = expanded_aids[0]
            daids = expanded_aids[1]

            _extra_aids =  ibs.get_valid_aids()
            _extra_aids = ibs.remove_groundtrue_aids(_extra_aids, (qaids + daids))
            _extra_aids = filter_annots_independent(ibs, _extra_aids, _aidcfg, prefix)
            _extra_aids = sample_annots(ibs, _extra_aids, _aidcfg, prefix)
            daids = sorted(daids + _extra_aids)
            expanded_aids = (qaids, daids)

        qsize = len(expanded_aids[0])
        dsize = len(expanded_aids[1])

        if min_qsize is None:
            qcfg['sample_size'] = qsize
        if min_dsize is None:  # UNSURE
            dcfg['sample_size'] = dsize

        if qcfg['sample_size'] != qsize:
            qcfg['_true_sample_size'] = qsize
        if dcfg['sample_size'] != dsize:
            dcfg['_true_sample_size'] = dsize

        if qcfg['force_const_size']:
            min_qsize = tmpmin(min_qsize, qsize)
        if dcfg['force_const_size']:  # UNSURE
            min_dsize = tmpmin(min_dsize, dsize)

        #ibs.print_annotconfig_stats(*expanded_aids)
        expanded_aids_list.append(expanded_aids)

    # Sample afterwords

    return list(zip(acfg_combo, expanded_aids_list))


def get_acfg_cacheinfo(ibs, aidcfg):
    from ibeis.experiments import cfghelpers
    # Make loading aids a big faster for experiments
    if ut.is_developer():
        import ibeis
        from os.path import dirname, join
        repodir = dirname(ut.get_module_dir(ibeis))
        acfg_cachedir = join(repodir, 'ACFG_CACHE')
    else:
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
    acfg_cacheinfo = acfg_cachedir, acfg_cachename, aid_cachestr
    return acfg_cacheinfo


@profile
def expand_acfgs(ibs, aidcfg, verbose=VERB_TESTDATA, use_cache=None, hack_exclude_keys=None):
    """
    Expands an annot config dict into qaids and daids
    New version of this function based on a configuration dictionary built from
    command line argumetns

    FIXME:
        The database should be created first in most circumstances, then
        the queries should be filtered to meet the database restrictions?
        I'm not sure Sometimes you need to set the query aids constant, but
        sometimes you need to set the data aids constant. Seems to depend.

    OkNewIdea:
        3 filters:
            * Common sampling - takes care of things like min time delta, species, quality viewpoint etc.
            * query sampling
            * database sampling
        Basic idea is
            * Sample large pool
            * Partition pool into query and database
        Requires:
            * base sampling params
            * partition1 params
            * partition2 params
            * inter partition params?

    CommandLine:
        python -m ibeis.dev -e print_acfg  -a timectrl:qsize=10,dsize=10  --db PZ_MTEST --veryverbtd --nocache-aid
        python -m ibeis.dev -e print_acfg  -a timectrl:qminqual=good,qsize=10,dsize=10  --db PZ_MTEST --veryverbtd --nocache-aid

        python -m ibeis.dev -e print_acfg  -a timectrl --db PZ_MTEST --verbtd --nocache-aid
        python -m ibeis.dev -e print_acfg  -a timectrl --db PZ_Master1 --verbtd --nocache-aid
        python -m ibeis.dev -e print_acfg  -a timequalctrl --db PZ_Master1 --verbtd --nocache-aid

        python -m ibeis.dev -e rank_cdf   -a controlled:qsize=10,dsize=10,dper_name=2 -t default --db PZ_MTEST
        python -m ibeis.dev -e rank_cdf   -a controlled:qsize=10,dsize=20,dper_name=2 -t default --db PZ_MTEST
        python -m ibeis.dev -e rank_cdf   -a controlled:qsize=10,dsize=30,dper_name=2 -t default --db PZ_MTEST
        python -m ibeis.dev -e print      -a controlled:qsize=10,dsize=10             -t default --db PZ_MTEST --verbtd --nocache-aid

        python -m ibeis.dev -e latexsum -t candinvar -a viewpoint_compare  --db NNP_Master3 --acfginfo
        utprof.py -m ibeis.dev -e print -t candk -a varysize  --db PZ_MTEST --acfginfo
        utprof.py -m ibeis.dev -e latexsum -t candk -a controlled  --db PZ_Master0 --acfginfo

        python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list:0 --db NNP_Master3 -a viewpoint_compare --nocache-aid --verbtd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.init.filter_annots import *  # NOQA
    """

    from ibeis.experiments import annotation_configs
    import copy
    aidcfg = copy.deepcopy(aidcfg)

    # Check if this filter has been cached
    # TODO: keep a database state config that augments the cachestr

    if use_cache is None:
        use_cache = USE_ACFG_CACHE

    save_cache = True
    if use_cache or save_cache:
        acfg_cacheinfo = get_acfg_cacheinfo(ibs, aidcfg)
        acfg_cachedir, acfg_cachename, aid_cachestr = acfg_cacheinfo
    if use_cache:
        try:
            (qaid_list, daid_list) = ut.load_cache(
                acfg_cachedir, acfg_cachename, aid_cachestr)
        except IOError:
            pass
        else:
            return qaid_list, daid_list

    comp_acfg = annotation_configs.compress_aidcfg(aidcfg)

    if verbose:
        print('+=== EXPAND_ACFGS ===')
        print(' * acfg = %s' % (ut.dict_str(comp_acfg, align=True),))
        print('+---------------------')

    # Breakup into common, query, and database configs
    qcfg = aidcfg['qcfg']
    dcfg = aidcfg['dcfg']
    common_cfg = comp_acfg['common']

    # Extract the common independent filtering params
    idenfilt_cfg_default = annotation_configs.INDEPENDENT_DEFAULTS
    idenfilt_cfg_empty = {key: None for key in idenfilt_cfg_default.keys()}
    idenfilt_cfg_common = ut.update_existing(idenfilt_cfg_empty,
                                             common_cfg, copy=True)

    if hack_exclude_keys:
        for key in hack_exclude_keys:
            if key in idenfilt_cfg_common:
                idenfilt_cfg_common[key] = None

    # Find the q/d specific filtering flags that were already taken care of in
    # common filtering. Set them all to None, so we dont rerun that filter
    qpredone_iden_keys = ut.dict_isect(qcfg, idenfilt_cfg_common).keys()
    for key in qpredone_iden_keys:
        qcfg[key] = None

    dpredone_iden_keys = ut.dict_isect(dcfg, idenfilt_cfg_common).keys()
    for key in dpredone_iden_keys:
        dcfg[key] = None

    try:
        # Hack: Make hierarchical filters to supersede this
        initial_aids = ibs._get_all_aids()

        verbflags  = dict(verbose=verbose)
        qfiltflags = dict(prefix='q', **verbflags)
        dfiltflags = dict(prefix='d', **verbflags)

        # Prefilter an initial pool of aids
        default_aids = filter_annots_independent(
            ibs, initial_aids, idenfilt_cfg_common, prefix='',
            withpre=True, **verbflags)
        avail_daids = avail_qaids = default_aids

        # Sample set of query annotations
        avail_qaids = filter_annots_independent(
            ibs, avail_qaids, qcfg, **qfiltflags)
        avail_qaids = sample_annots(
            ibs, avail_qaids, qcfg, **qfiltflags)

        # Sample set of database annotations w.r.t query annots
        avail_daids = filter_annots_independent(
            ibs, avail_daids, dcfg, **dfiltflags)
        avail_daids = sample_annots_wrt_ref(
            ibs, avail_daids, dcfg, reference_aids=avail_qaids,
            **dfiltflags)

        # Subindex if requested (typically not done)
        avail_qaids = subindex_annots(
            ibs, avail_qaids, qcfg, **qfiltflags)
        avail_daids = subindex_annots(
            ibs, avail_daids, dcfg, **dfiltflags)

    except Exception as ex:
        print('PRINTING ERROR INFO')
        print(' * acfg = %s' % (ut.dict_str(comp_acfg, align=True),))
        ut.printex(ex, 'Error expanding acfgs')
        raise

    qaid_list = sorted(avail_qaids)
    daid_list = sorted(avail_daids)

    if verbose:
        print('L___ EXPAND_ACFGS ___')
        ibs.print_annotconfig_stats(qaid_list, daid_list)

    # Save filter to cache
    if save_cache:
        ut.ensuredir(acfg_cachedir)
        ut.save_cache(acfg_cachedir, acfg_cachename, aid_cachestr,
                      (qaid_list, daid_list))

    return qaid_list, daid_list


@profile
def filter_annots_independent(ibs, avail_aids, aidcfg, prefix='',
                              verbose=VERB_TESTDATA, withpre=False):
    r""" Filtering that doesn't have to do with a reference set of aids

    Args:
        ibs (IBEISController):  ibeis controller object
        avail_aids (list):
        aidcfg (dict):
        prefix (str): (default = '')
        verbose (bool):  verbosity flag(default = False)

    Returns:
        list: avail_aids

    CommandLine:
        python -m ibeis.init.filter_annots --exec-filter_annots_independent \
                --verbtd --veryverbtd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.init.filter_annots import *  # NOQA
        >>> import ibeis
        >>> from ibeis.experiments import annotation_configs
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> avail_aids = input_aids = ibs.get_valid_aids()
        >>> aidcfg = annotation_configs.default['dcfg']
        >>> aidcfg['require_timestamp'] = True
        >>> aidcfg['require_quality'] = False
        >>> aidcfg['min_timedelta'] = 60 * 60 * 24
        >>> aidcfg['min_pername'] = 3
        >>> aidcfg['is_known'] = True
        >>> prefix = ''
        >>> verbose = True
        >>> avail_aids = filter_annots_independent(ibs, avail_aids, aidcfg, prefix, verbose)
        >>> result = ('avail_aids = %s' % (str(avail_aids),))
        >>> print(result)
    """
    from ibeis import ibsfuncs

    if aidcfg is None:
        if verbose:
            print('No annot filter returning')
        return avail_aids

    VerbosityContext = verbose_context_factory(
        'FILTER_INDEPENDENT', aidcfg, verbose)
    VerbosityContext.startfilter(withpre=withpre)

    if aidcfg['is_known'] is True:
        with VerbosityContext('is_known'):
            avail_aids = ibs.filter_aids_without_name(avail_aids, invert=not aidcfg['is_known'])
        avail_aids = sorted(avail_aids)

    if aidcfg['require_timestamp'] is True:
        with VerbosityContext('require_timestamp'):
            avail_aids = ibs.filter_aids_without_timestamps(avail_aids)
        avail_aids = sorted(avail_aids)

    species = None
    if aidcfg['species'] is not None:
        if aidcfg['species'] == 'primary':
            species = ibs.get_primary_database_species()
        else:
            species = aidcfg['species']
        with VerbosityContext('species', species=species):
            avail_aids = ibs.filter_aids_to_species(avail_aids, species)
            avail_aids = sorted(avail_aids)

    if aidcfg['minqual'] is not None or aidcfg['require_quality']:
        # Resolve quality
        if aidcfg['minqual'] is None:
            minqual = 'junk'
        else:
            minqual = aidcfg['minqual']
        with VerbosityContext('minqual', 'require_quality'):
            # Filter quality
            avail_aids = ibs.filter_aids_to_quality(
                avail_aids, minqual, unknown_ok=not aidcfg['require_quality'])
        avail_aids = sorted(avail_aids)

    # FIXME: This is NOT an independent filter because it depends on pairwise interactions
    if aidcfg['view_pername'] is not None:
        if species is None:
            # hack
            species = ibs.get_dominant_species(avail_aids)
        # This filter removes entire names.
        # The avaiable aids must be from names with certain viewpoint frequency
        # properties
        prop2_nid2_aids = ibs.group_annots_by_prop_and_name(
            avail_aids, ibs.get_annot_yaw_texts)
        #ut.embed()
        countstr = aidcfg['view_pername']
        primary_viewpoint = ibsfuncs.get_primary_species_viewpoint(species)
        lhs_dict = {
            'primary': primary_viewpoint,
            'primary1': ibsfuncs.get_extended_viewpoints(
                primary_viewpoint, num1=1, num2=0, include_base=False)[0]
        }
        self = CountstrParser(lhs_dict, prop2_nid2_aids)
        nid2_flag = self.parse_countstr_expr(countstr)
        nid2_aids = ibs.group_annots_by_name_dict(avail_aids)
        valid_nids = [nid for nid, flag in nid2_flag.items() if flag]
        with VerbosityContext('view_pername', countstr=countstr):
            avail_aids = ut.flatten(ut.dict_take(nid2_aids, valid_nids))
        avail_aids = sorted(avail_aids)

    if aidcfg['view'] is not None or aidcfg['require_viewpoint']:
        # Resolve base viewpoint
        if aidcfg['view'] == 'primary':
            view = ibsfuncs.get_primary_species_viewpoint(species)
        elif aidcfg['view'] == 'primary1':
            view = ibsfuncs.get_primary_species_viewpoint(species, 1)
        else:
            view = aidcfg['view']
        view_ext1 = (aidcfg['view_ext']
                     if aidcfg['view_ext1'] is None else
                     aidcfg['view_ext1'])
        view_ext2 = (aidcfg['view_ext']
                     if aidcfg['view_ext2'] is None else
                     aidcfg['view_ext2'])
        valid_yaws = ibsfuncs.get_extended_viewpoints(view, num1=view_ext1, num2=view_ext2)
        unknown_ok = not aidcfg['require_viewpoint']
        # Filter viewpoint
        with VerbosityContext('view', 'require_viewpoint', 'view_ext',
                              'view_ext1', 'view_ext2', valid_yaws=valid_yaws):
            avail_aids = ibs.filter_aids_to_viewpoint(
                avail_aids, valid_yaws, unknown_ok=unknown_ok)
        avail_aids = sorted(avail_aids)

    #if True:
    #    # Filter viewpoint
    #    with VerbosityContext('view', hack=True):
    #        avail_aids = ibs.remove_aids_of_viewpoint(
    #            avail_aids, ['front', 'backleft'])

    # FIXME: This is NOT an independent filter because it depends on pairwise interactions
    if aidcfg['min_timedelta'] is not None:
        min_timedelta = ut.ensure_timedelta(aidcfg['min_timedelta'])
        # Filter viewpoint
        with VerbosityContext('min_timedelta', min_timedelta=min_timedelta):
            avail_aids = ibs.filter_annots_using_minimum_timedelta(
                avail_aids, min_timedelta)
        avail_aids = sorted(avail_aids)

    # Each aid must have at least this number of other groundtruth aids
    min_pername = aidcfg['min_pername']
    if min_pername is not None:
        grouped_aids_ = ibs.group_annots_by_name(avail_aids,
                                                 distinguish_unknowns=True)[0]
        with VerbosityContext('min_pername'):
            avail_aids = ut.flatten([
                aids for aids in grouped_aids_ if len(aids) >= min_pername])
        avail_aids = sorted(avail_aids)

    avail_aids = sorted(avail_aids)

    VerbosityContext.endfilter()
    return avail_aids


def get_reference_preference_order(ibs, gt_ref_grouped_aids,
                                   gt_avl_grouped_aids, prop_getter, cmp_func,
                                   aggfn, rng, verbose=VERB_TESTDATA):
    r"""
    Orders preference for sampling based on some metric
    """
    import vtool as vt
    grouped_reference_unixtimes = ibs.unflat_map(prop_getter, gt_ref_grouped_aids)
    grouped_available_gt_unixtimes = ibs.unflat_map(prop_getter, gt_avl_grouped_aids)

    grouped_reference_props = grouped_reference_unixtimes
    grouped_available_gt_props = grouped_available_gt_unixtimes

    # Order the available aids by some aggregation over some metric
    preference_scores = [
        aggfn(cmp_func(ref_prop, avl_prop[:, None]), axis=1)
        for ref_prop, avl_prop in
        zip(grouped_reference_props, grouped_available_gt_props)
    ]
    # Order by increasing timedelta (metric)
    gt_preference_idx_list = vt.argsort_groups(preference_scores, reverse=True, rng=rng)
    return gt_preference_idx_list


@profile
def sample_annots_wrt_ref(ibs, avail_aids, aidcfg, reference_aids, prefix='',
                          verbose=VERB_TESTDATA):
    """
    Sampling when a reference set is given
    """
    sample_per_name     = aidcfg['sample_per_name']
    sample_per_ref_name = aidcfg['sample_per_ref_name']
    exclude_reference   = aidcfg['exclude_reference']
    sample_size         = aidcfg['sample_size']
    offset              = aidcfg['sample_offset']
    sample_rule_ref     = aidcfg['sample_rule_ref']
    sample_rule         = aidcfg['sample_rule']

    avail_aids = sorted(avail_aids)
    reference_aids = sorted(reference_aids)

    VerbosityContext = verbose_context_factory(
        'SAMPLE (REF)', aidcfg, verbose)
    VerbosityContext.startfilter()

    if sample_per_ref_name is None:
        sample_per_ref_name = sample_per_name

    if offset is None:
        offset = 0

    if exclude_reference:
        assert reference_aids is not None, 'reference_aids=%r' % (reference_aids,)
        # VerbosityContext.report_annot_stats(ibs, avail_aids, prefix, '')
        # VerbosityContext.report_annot_stats(ibs, reference_aids, prefix, '')
        with VerbosityContext('exclude_reference', num_ref_aids=len(reference_aids)):
            avail_aids = ut.setdiff_ordered(avail_aids, reference_aids)
            avail_aids = sorted(avail_aids)

    if not (sample_per_ref_name is not None or sample_size is not None):
        VerbosityContext.endfilter()
        return avail_aids

    if isinstance(sample_size, float):
        # A float sample size is a interpolations between full data and small data
        sample_size = int(round((len(avail_aids) * sample_size + (1 - sample_size) * len(reference_aids))))

    # This function first partitions aids into a one set that corresonds with
    # the reference set and another that does not correspond with the reference
    # set. The rest of the filters operate on these sets independently
    partitioned_sets = ibs.partition_annots_into_corresponding_groups(
        reference_aids, avail_aids)
    # gt_ref_grouped_aids, and gt_avl_grouped_aids are corresponding lists of annot groups
    # gf_ref_grouped_aids, and gf_avl_grouped_aids are non-corresonding annot groups
    (gt_ref_grouped_aids, gt_avl_grouped_aids,
     gf_ref_grouped_aids, gf_avl_grouped_aids) = partitioned_sets

    if sample_per_ref_name is not None:
        rng = np.random.RandomState(SEED2)
        if sample_rule_ref == 'maxtimedelta':
            # Maximize time delta between query and corresponding database annotations
            cmp_func = ut.absdiff
            aggfn = np.mean
            prop_getter = ibs.get_annot_image_unixtimes_asfloat
            gt_preference_idx_list = get_reference_preference_order(
                ibs, gt_ref_grouped_aids, gt_avl_grouped_aids, prop_getter,
                cmp_func, aggfn, rng)
        elif sample_rule_ref == 'random':
            gt_preference_idx_list = [ut.random_indexes(len(aids), rng=rng)
                                      for aids in gt_avl_grouped_aids]
        else:
            raise ValueError('Unknown sample_rule_ref = %r' % (sample_rule_ref,))
        gt_sample_idxs_list = ut.get_list_column_slice(
            gt_preference_idx_list, offset, offset + sample_per_ref_name)
        gt_sample_aids = ut.list_ziptake(gt_avl_grouped_aids, gt_sample_idxs_list)
        gt_avl_grouped_aids = gt_sample_aids

        with VerbosityContext('sample_per_ref_name', 'sample_rule_ref', 'sample_offset', sample_per_ref_name=sample_per_ref_name):
            avail_aids = ut.flatten(gt_avl_grouped_aids) + ut.flatten(gf_avl_grouped_aids)

    if sample_per_name is not None:
        # sample rule is always random for gf right now
        rng = np.random.RandomState(SEED2)
        if sample_rule == 'random':
            gf_preference_idx_list = [ut.random_indexes(len(aids), rng=rng)
                                      for aids in gf_avl_grouped_aids]
        else:
            raise ValueError('Unknown sample_rule=%r' % (sample_rule,))
        gf_sample_idxs_list = ut.get_list_column_slice(
            gf_preference_idx_list, offset, offset + sample_per_name)
        gf_sample_aids = ut.list_ziptake(gf_avl_grouped_aids, gf_sample_idxs_list)
        gf_avl_grouped_aids = gf_sample_aids

        with VerbosityContext('sample_per_name', 'sample_rule', 'sample_offset'):
            avail_aids = ut.flatten(gt_avl_grouped_aids) + ut.flatten(gf_avl_grouped_aids)

    gt_avl_aids = ut.flatten(gt_avl_grouped_aids)
    gf_avl_aids = ut.flatten(gf_avl_grouped_aids)

    if sample_size is not None:
        # Keep all correct matches to the reference set
        # We have the option of keeping ground false
        num_gt = len(gt_avl_aids)
        num_gf = len(gf_avl_aids)
        num_keep_gf = sample_size - num_gt
        num_remove_gf = num_gf - num_keep_gf
        if num_remove_gf < 0:
            # Too few ground false
            print(('Warning: Cannot meet sample_size=%r. available_%saids '
                   'will be undersized by at least %d')
                  % (sample_size, prefix, -num_remove_gf,))
        if num_keep_gf < 0:
            # Too many multitons; Can never remove a multiton
            print('Warning: Cannot meet sample_size=%r. available_%saids '
                  'will be oversized by at least %d'
                  % (sample_size, prefix, -num_keep_gf,))
        rng = np.random.RandomState(SEED2)
        gf_avl_aids = ut.random_sample(gf_avl_aids, num_keep_gf, rng=rng)

        # random ordering makes for bad hashes
        with VerbosityContext('sample_size', sample_size=sample_size, num_remove_gf=num_remove_gf, num_keep_gf=num_keep_gf):
            avail_aids = gt_avl_aids + gf_avl_aids

    avail_aids = sorted(gt_avl_aids + gf_avl_aids)

    VerbosityContext.endfilter()
    return avail_aids


@profile
def sample_annots(ibs, avail_aids, aidcfg, prefix='', verbose=VERB_TESTDATA):
    """
    Sampling preserves input sample structure and thust does not always return exact values

    CommandLine:
        python -m ibeis.init.filter_annots --exec-sample_annots --veryverbtd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.init.filter_annots import *  # NOQA
        >>> import ibeis
        >>> from ibeis.experiments import annotation_configs
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> avail_aids = input_aids = ibs.get_valid_aids()
        >>> aidcfg = annotation_configs.default['dcfg']
        >>> aidcfg['sample_per_name'] = 3
        >>> aidcfg['sample_size'] = 10
        >>> aidcfg['min_pername'] = 2
        >>> prefix = ''
        >>> verbose = True
        >>> avail_aids = filter_annots_independent(ibs, avail_aids, aidcfg, prefix, verbose)
        >>> avail_aids = sample_annots(ibs, avail_aids, aidcfg, prefix, avail_aids)
        >>> result = ('avail_aids = %s' % (str(avail_aids),))
        >>> print(result)
    """
    import vtool as vt

    VerbosityContext = verbose_context_factory(
        'SAMPLE (NOREF)', aidcfg, verbose)
    VerbosityContext.startfilter()

    sample_rule     = aidcfg['sample_rule']
    sample_per_name = aidcfg['sample_per_name']
    sample_size     = aidcfg['sample_size']
    offset          = aidcfg['sample_offset']

    unflat_get_annot_unixtimes = functools.partial(
        ibs.unflat_map, ibs.get_annot_image_unixtimes_asfloat)

    if offset is None:
        offset = 0

    if sample_per_name is not None:
        # For the query we just choose a single annot per name
        # For the database we have to do something different
        grouped_aids = ibs.group_annots_by_name(avail_aids)[0]
        # Order based on some preference (like random)
        rng = np.random.RandomState(SEED1)
        # + --- Get nested sample indicies ---
        if sample_rule == 'random':
            preference_idxs_list = [
                ut.random_indexes(len(aids), rng=rng) for aids in grouped_aids]
        elif sample_rule == 'mintime':
            unixtime_list = unflat_get_annot_unixtimes(grouped_aids)
            preference_idxs_list = vt.argsort_groups(unixtime_list, reverse=False, rng=rng)
        elif sample_rule == 'maxtime':
            unixtime_list = unflat_get_annot_unixtimes(grouped_aids)
            preference_idxs_list = vt.argsort_groups(unixtime_list, reverse=True, rng=rng)
        else:
            raise ValueError('Unknown sample_rule=%r' % (sample_rule,))
        # L ___
        sample_idxs_list = ut.get_list_column_slice(
            preference_idxs_list, offset, offset + sample_per_name)
        sample_aids = ut.list_ziptake(grouped_aids, sample_idxs_list)

        with VerbosityContext('sample_per_name', 'sample_rule', 'sample_offset'):
            avail_aids = ut.flatten(sample_aids)
        avail_aids = sorted(avail_aids)

    if sample_size is not None:
        # BUG: Should sample annots while preserving name size
        if sample_size > avail_aids:
            print('Warning sample size too large')
        rng = np.random.RandomState(SEED2)

        # Randomly sample names rather than annotations this makes sampling a
        # knapsack problem. Use a random greedy solution
        grouped_aids = ibs.group_annots_by_name(avail_aids)[0]
        # knapsack items values and weights are are num annots per name
        knapsack_items = [(len(aids), len(aids), count)
                          for count, aids  in enumerate(grouped_aids)]
        ut.deterministic_shuffle(knapsack_items, rng=rng)
        total_value, items_subset = ut.knapsack_greedy(knapsack_items, sample_size)
        group_idx_sample = ut.get_list_column(items_subset, 2)
        subgroup_aids = ut.list_take(grouped_aids, group_idx_sample)

        with VerbosityContext('sample_size'):
            avail_aids = ut.flatten(subgroup_aids)
            #avail_aids = ut.random_sample(avail_aids, sample_size, rng=rng)
        if total_value != sample_size:
            print('Sampling could not get exactly right sample size')
        avail_aids = sorted(avail_aids)

    VerbosityContext.endfilter()
    return avail_aids


@profile
def subindex_annots(ibs, avail_aids, aidcfg, reference_aids=None,
                           prefix='', verbose=VERB_TESTDATA):
    """
    Returns exact subindex of annotations
    """
    VerbosityContext = verbose_context_factory(
        'SUBINDEX', aidcfg, verbose)
    VerbosityContext.startfilter(withpre=False)

    if aidcfg['shuffle']:
        rand_idx = ut.random_indexes(len(avail_aids), seed=SEED2)
        with VerbosityContext('shuffle', SEED2=SEED2):
            avail_aids = ut.list_take(avail_aids, rand_idx)

    if aidcfg['index'] is not None:
        indicies = ensure_flatlistlike(aidcfg['index'])
        _indexed_aids = [avail_aids[ix] for ix in indicies if ix < len(avail_aids)]
        with VerbosityContext('index', subset_size=len(_indexed_aids)):
            avail_aids = _indexed_aids

    # Always sort aids to preserve hashes? (Maybe sort the vuuids instead)
    avail_aids = sorted(avail_aids)

    VerbosityContext.endfilter(withpost=False)
    return avail_aids


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
            valiter = self.prop2_nid2_aids.get(prop, {}).values()
            prop_left_values = np.array(list(map(len, valiter)))
        # Pares number
        if right:
            prop_right_value = int(right)
        # Execute comparison
        prop_binary_result = self.compare_op_map[op](
            prop_left_values, prop_right_value)
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
        andcombine = functools.partial(
            ut.dict_intersection, combine=True, combine_op=operator.and_)
        expr_nid2_result = reduce(andcombine, prop_nid2_result_list)
        return expr_nid2_result
        #reduce(functools.partial(ut.dict_union3, combine_op=operator.or_), prop_nid2_result_list)


def verbose_context_factory(filtertype, aidcfg, verbose):
    """ closure helper """
    class VerbosityContext():
        """
        very hacky way of printing info so we dont pollute the actual function
        too much
        """

        @staticmethod
        def report_annot_stats(ibs, aids, prefix, name_suffix, statskw={}):
            if verbose > 1:
                with ut.Indenter('[%s]  ' % (prefix.upper(),)):
                    # TODO: helpx on statskw
                    #statskw = dict(per_name_vpedge=None, per_name=None)
                    dict_name = prefix + 'aid_stats' + name_suffix
                    #hashid, per_name, per_qual, per_vp, per_name_vpedge, per_image, min_name_hourdist
                    ibs.print_annot_stats(aids, prefix=prefix, label=dict_name, **statskw)

        #def report_annotconfig_stats(ref_aids, aids):
        #    with ut.Indenter('  '):
        #        ibs.print_annotconfig_stats(reference_aids, avail_aids)

        @staticmethod
        def startfilter(withpre=True):
            if verbose:
                prefix = ut.get_var_from_stack('prefix', verbose=False)
                print('[%s] * [%s] %sAIDS' % (prefix.upper(), filtertype, prefix))
                if verbose > 1 and withpre:
                    ibs    = ut.get_var_from_stack('ibs', verbose=False)
                    aids   = ut.get_var_from_stack('avail_aids', verbose=False)
                    VerbosityContext.report_annot_stats(ibs, aids, prefix, '_pre')

        @staticmethod
        def endfilter(withpost=True):
            if verbose:
                ibs    = ut.get_var_from_stack('ibs', verbose=False)
                aids   = ut.get_var_from_stack('avail_aids', verbose=False)
                prefix = ut.get_var_from_stack('prefix', verbose=False)
                hashid = ibs.get_annot_hashid_semantic_uuid(
                    aids, prefix=prefix.upper())
                if withpost:
                    if verbose > 1:
                        VerbosityContext.report_annot_stats(ibs, aids, prefix, '_post')
                print('[%s] * HAHID: %s' % (prefix.upper(), hashid))
                print('[%s] * [%s]: len(avail_%saids) = %r\n' % (
                    prefix.upper(), filtertype, prefix, len(aids)))

        def __init__(self, *keys, **filterextra):
            self.prefix = ut.get_var_from_stack('prefix', verbose=False)
            if verbose:
                dictkw = dict(nl=False, explicit=True, nobraces=True)
                infostr = ''
                if len(keys) > 0:
                    subdict = ut.dict_subset(aidcfg, keys)
                    infostr += '' + ut.dict_str(subdict, **dictkw)
                if len(filterextra) > 0:
                    infostr += ' : ' + ut.dict_str(filterextra, **dictkw)
                print('[%s] * Filtering by %s' % (self.prefix.upper(), infostr.strip()))

        def __enter__(self):
            aids = ut.get_var_from_stack('avail_aids', verbose=False)
            self.num_before = len(aids)

        def __exit__(self, exc_type, exc_value, exc_traceback):
            if verbose:
                aids = ut.get_var_from_stack('avail_aids', verbose=False)
                num_after = len(aids)
                num_removed = self.num_before - num_after
                if num_removed > 0 or verbose > 1:
                    print('[%s]   ... removing %d annots. %d remaning' % (self.prefix.upper(), num_removed, num_after))
    return VerbosityContext


#avail_qaids = filter_reference_properties(ibs, avail_qaids, qcfg, reference_aids=avail_daids, prefix='q')
#avail_daids = filter_reference_properties(ibs, avail_daids, dcfg, reference_aids=avail_qaids, prefix='d')
#@profile
#def filter_reference_properties(ibs, avail_aids, aidcfg, reference_aids, prefix='', verbose=VERB_TESTDATA):
#    """
#    DEPRICATE
#    """
#    from ibeis import ibsfuncs
#    import functools
#    avail_aids = sorted(avail_aids)
#    reference_aids = sorted(reference_aids)
#    if verbose:
#        print(' * [FILTER REFERENCE %sAIDS]' % (prefix.upper()))
#        if VERYVERB_TESTDATA:
#            with ut.Indenter('  '):
#                ibs.print_annot_stats(avail_aids, prefix, per_name_vpedge=None)
#    if aidcfg['ref_has_viewpoint'] is not None:
#        print(' * Filtering such that %saids has refs with viewpoint=%r' % (prefix, aidcfg['ref_has_viewpoint']))
#        species = ibs.get_primary_database_species(avail_aids)
#        if aidcfg['ref_has_viewpoint']  == 'primary':
#            valid_yaws = [ibsfuncs.get_primary_species_viewpoint(species)]
#        elif aidcfg['ref_has_viewpoint']  == 'primary1':
#            valid_yaws = [ibsfuncs.get_primary_species_viewpoint(species, 1, 1)]
#        gt_ref_grouped_aids, gt_avl_grouped_aids, gf_ref_grouped_aids, gf_avl_grouped_aids = ibs.partition_annots_into_corresponding_groups(reference_aids, avail_aids)
#        # Filter to only available aids that have a reference with specified viewpoint
#        is_valid_yaw = functools.partial(ibs.get_viewpoint_filterflags, valid_yaws=valid_yaws)
#        multi_flags = list(map(any, ibs.unflat_map(is_valid_yaw, gt_ref_grouped_aids)))
#        avail_aids = ut.flatten(ut.list_compress(gt_avl_grouped_aids, multi_flags))
#        avail_aids = sorted(avail_aids)
#    if verbose:
#        print(' * HAHID: ' + ibs.get_annot_hashid_semantic_uuid(avail_aids, prefix=prefix.upper()))
#        print(' * R-FILTERED: len(available_%saids)=%r\n' % (prefix, len(avail_aids)))
#    return avail_aids


#@profile
#def expand_to_default_aids(ibs, aidcfg, prefix='', verbose=VERB_TESTDATA):
#    default_aids = aidcfg['default_aids']

#    if verbose:
#        print(' * [INCLUDE %sAIDS]' % (prefix.upper()))
#        #print(' * PARSING %saidcfg = %s' % (prefix, ut.dict_str(aidcfg, align=True),))
#        print(' * default_%saids = %s' % (prefix, ut.obj_str(default_aids,
#                                                             truncate=True,
#                                                             nl=False)))

#    if isinstance(default_aids, six.string_types):
#        #if verbose:
#        #    print(' * interpreting default %saids.' % (prefix,))
#        # Abstract default aids
#        if default_aids in ['all']:
#            default_aids = ibs.get_valid_aids()
#        elif default_aids in ['allgt', 'gt']:
#            default_aids = ibs.get_valid_aids(hasgt=True)
#        elif default_aids in ['largetime24']:
#            # HACK for large timedelta base sample pool
#            default_aids = ibs.get_valid_aids(
#                is_known=True,
#                has_timestamp=True,
#                min_timedelta=24 * 60 * 60,
#            )
#        elif default_aids in ['largetime12']:
#            # HACK for large timedelta base sample pool
#            default_aids = ibs.get_valid_aids(
#                is_known=True,
#                has_timestamp=True,
#                min_timedelta=12 * 60 * 60,
#            )
#        elif default_aids in ['other']:
#            # Hack, should actually become the standard.
#            # Use this function to build the default aids
#            default_aids = ibs.get_valid_aids(
#                is_known=aidcfg['is_known'],
#                min_timedelta=aidcfg['min_timedelta'],
#                has_timestamp=aidcfg['require_timestamp']
#            )
#        #elif default_aids in ['reference_gt']:
#        #    pass
#        else:
#            raise NotImplementedError('Unknown default string = %r' % (default_aids,))
#    else:
#        if verbose:
#            print(' ... default %saids specified.' % (prefix,))

#    #if aidcfg['include_aids'] is not None:
#    #    raise NotImplementedError('Implement include_aids')

#    avail_aids = default_aids

#    if len(avail_aids) == 0:
#        print(' WARNING no %s annotations available' % (prefix,))

#    #if aidcfg['exclude_aids'] is not None:
#    #    if verbose:
#    #        print(' * Excluding %d custom aids' % (len(aidcfg['exclude_aids'])))
#    #    avail_aids = ut.setdiff_ordered(avail_aids, aidcfg['exclude_aids'])
#    avail_aids = sorted(avail_aids)

#    if verbose:
#        print(' * HAHID: ' + ibs.get_annot_hashid_semantic_uuid(
#            avail_aids, prefix=prefix.upper()))
#        print(' * DEFAULT: len(available_%saids)=%r\n' % (prefix, len(avail_aids)))
#    return avail_aids


#    #if gt_avl_aids is not None:
#    #    if verbose:
#    #        print(' * Excluding gt_avl_aids custom specified by name')
#    #    # Pick out the annotations that do not belong to the same name as the given gt_avl_aids
#    #    complement = np.setdiff1d(avail_aids, gt_avl_aids)
#    #    partitioned_sets = ibs.partition_annots_into_corresponding_groups(gt_avl_aids, complement)
#    #    assert len(set(ut.flatten((ibs.unflat_map(ibs.get_annot_name_rowids, partitioned_sets[1])))).intersection(set(ut.flatten((ibs.unflat_map(ibs.get_annot_name_rowids, partitioned_sets[3])))))) == 0
#    #    gf_avl_aids = (ut.flatten(partitioned_sets[3]))
#    #    assert len(set(ibs.get_annot_name_rowids(reference_aids)).intersection(set(ibs.get_annot_name_rowids(gf_avl_aids) ))) == 0
#    #    avail_aids = np.hstack([gt_avl_aids, gf_avl_aids])
#    #    avail_aids.sort()
#    #    avail_aids = avail_aids.tolist()
#    #    avail_aids = sorted(avail_aids)


##
#    #aidcfg['viewpoint_edge_counts'] = None
#    #if aidcfg['viewpoint_edge_counts'] is not None:
#    #    getter_list = [ibs.get_annot_name_rowids, ibs.get_annot_yaw_texts]
#    #    nid2_vp2_aids = ibs.group_annots_by_multi_prop(avail_aids, getter_list)
#    #    #assert len(avail_aids) == len(list(ut.iflatten_dict_values(nid2_vp2_aids)))
#    #    nid2_vp2_aids = ut.hierarchical_map_vals(ut.identity, nid2_vp2_aids)  # remove defaultdict structure
#    #    nid2_num_vp = ut.hierarchical_map_vals(len, nid2_vp2_aids, max_depth=0)
#    #    min_num_vp_pername = 2
#    #    def has_required_vps(vp2_aids):
#    #        min_examples_per_vp = {
#    #            'frontleft': 1,
#    #            'left': 2,
#    #        }
#    #        return all([key in vp2_aids and vp2_aids[key] for key in min_examples_per_vp])
#    #    nids_with_multiple_vp = [key for  key, val in nid2_num_vp.items() if val >= min_num_vp_pername]
#    #    subset1_nid2_vp2_aids = ut.dict_subset(nid2_vp2_aids, nids_with_multiple_vp)
#    #    subset2_flags = ut.hierarchical_map_vals(has_required_vps, nid2_vp2_aids, max_depth=0)
#    #    subset2_nids = ut.list_compress(subset2_flags.keys(), subset2_flags.values())
#    #    subset2_nid2_vp2_aids = ut.dict_subset(subset1_nid2_vp2_aids, subset2_nids)
#    #    avail_aids = list(ut.iflatten_dict_values(subset2_nid2_vp2_aids))


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
