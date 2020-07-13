# -*- coding: utf-8 -*-
"""
TODO:
    * cross validation
    * encounter vs database (time filtering)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import copy
import utool as ut
import numpy as np
import six
from wbia.control import controller_inject

(print, rrr, profile) = ut.inject2(__name__)

VERB_TESTDATA = ut.get_verbflag('testdata', 'td', 'acfg')[0]

SEED1 = 0
SEED2 = 42

if False and ut.is_developer():
    USE_ACFG_CACHE = (
        not ut.get_argflag(('--nocache-annot', '--nocache-aid', '--nocache'))
        and ut.USE_CACHE
    )
    USE_ACFG_CACHE = False
else:
    USE_ACFG_CACHE = False

_tup = controller_inject.make_ibs_register_decorator(__name__)
CLASS_INJECT_KEY, register_ibs_method = _tup


@profile
def time_filter_annots():
    r"""
    python -m wbia.init.filter_annots time_filter_annots \
            --db PZ_Master1 -a ctrl:qmingt=2 --profile

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.init.filter_annots import *  # NOQA
        >>> result = time_filter_annots()
    """
    import wbia

    wbia.testdata_expanded_aids()


@register_ibs_method
def filter_annots_general(ibs, aid_list=None, filter_kw={}, verbose=False, **kwargs):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (list):  list of annotation rowids
        filter_kw (?):

    KWargs::
        has_none_annotmatch, any_match_annotmatch, has_all, is_known,
        any_match_annot, logic_annot, none_match_annotmatch,
        max_num_annotmatch, any_startswith_annot, has_any, require_quality,
        species, any_match, view_ext, has_any_annotmatch, view_pername,
        max_num_annot, min_timedelta, any_startswith, max_numfeat,
        any_startswith_annotmatch, been_adjusted, any_endswith_annot,
        require_viewpoint, logic, has_any_annot, min_num_annotmatch, min_num,
        min_num_annot, has_all_annot, has_none, min_pername,
        any_endswith_annotmatch, any_endswith, require_timestamp, none_match,
        contributor_contains, has_all_annotmatch, logic_annotmatch, min_numfeat,
        none_match_annot, view_ext1, view_ext2, max_num, has_none_annot,
        minqual, view

    CommandLine:
        python -m wbia --tf filter_annots_general
        python -m wbia --tf filter_annots_general --db PZ_Master1 \
                --has_any=[needswork,correctable,mildviewpoint] \
                --has_none=[viewpoint,photobomb,error:viewpoint,quality] --show

        python -m wbia --tf filter_annots_general --db=GZ_Master1  \
                --max-numfeat=300 --show --minqual=junk --species=None
        python -m wbia --tf filter_annots_general --db=lynx \
                --been_adjusted=True

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.init.filter_annots import *  # NOQA
        >>> import wbia
        >>> filter_kw = ut.argparse_dict(get_default_annot_filter_form(),
        >>>                              type_hint=ut.ddict(list, has_any=list,
        >>>                                                 has_none=list,
        >>>                                                 logic=str))
        >>> print('filter_kw = %s' % (ut.repr2(filter_kw),))
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> #filter_kw = dict(is_known=True, min_num=1, has_any='viewpoint')
        >>> #filter_kw = dict(is_known=True, min_num=1, any_match='.*error.*')
        >>> aid_list_ = filter_annots_general(ibs, aid_list, filter_kw)
        >>> print('len(aid_list_) = %r' % (len(aid_list_),))
        >>> all_tags = ut.flatten(ibs.get_annot_all_tags(aid_list_))
        >>> filtered_tag_hist = ut.dict_hist(all_tags)
        >>> ut.print_dict(filtered_tag_hist, key_order_metric='val')
        >>> ut.print_dict(ibs.get_annot_stats_dict(aid_list_), 'annot_stats')
        >>> ut.quit_if_noshow()
        >>> import wbia.viz.interact
        >>> wbia.viz.interact.interact_chip.interact_multichips(ibs, aid_list_)
        >>> ut.show_if_requested()
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    filter_kw_ = get_default_annot_filter_form()
    ut.update_existing(filter_kw_, filter_kw, iswarning=True, assert_exists=True)
    ut.update_existing(filter_kw_, kwargs, iswarning=True, assert_exists=True)
    aid_list_ = aid_list
    # filter_kw = ut.merge_dicts(get_default_annot_filter_form(), filter_kw)
    # TODO MERGE FILTERFLAGS BY TAGS AND FILTERFLAGS INDEPENDANT
    # aid_list_ = ibs.filterannots_by_tags(aid_list_, filter_kw)
    aid_list_ = ibs.filter_annots_independent(aid_list_, filter_kw_, verbose=verbose)
    aid_list_ = filter_annots_intragroup(ibs, aid_list_, filter_kw_, verbose=verbose)
    return aid_list_


@register_ibs_method
def sample_annots_general(ibs, aid_list=None, filter_kw={}, verbose=False, **kwargs):
    """ filter + sampling """
    # hack
    from wbia.expt import annotation_configs

    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    filter_kw_ = annotation_configs.INDEPENDENT_DEFAULTS.copy()
    filter_kw_.update(annotation_configs.SUBINDEX_DEFAULTS.copy())
    filter_kw_.update(annotation_configs.SAMPLE_DEFAULTS.copy())

    ut.update_existing(filter_kw_, filter_kw, iswarning=True, assert_exists=True)
    ut.update_existing(filter_kw_, kwargs, iswarning=True, assert_exists=True)
    aid_list_ = aid_list
    # filter_kw = ut.merge_dicts(get_default_annot_filter_form(), filter_kw)
    # TODO MERGE FILTERFLAGS BY TAGS AND FILTERFLAGS INDEPENDANT
    # aid_list_ = ibs.filterannots_by_tags(aid_list_, filter_kw)
    aid_list_ = ibs.filter_annots_independent(aid_list_, filter_kw_, verbose=verbose)
    aid_list_ = filter_annots_intragroup(ibs, aid_list_, filter_kw_, verbose=verbose)

    aid_list_ = sample_annots(ibs, aid_list_, filter_kw_, verbose=verbose)
    aid_list_ = subindex_annots(ibs, aid_list_, filter_kw_, verbose=verbose)
    return aid_list_


@profile
def get_default_annot_filter_form():
    r"""
    Returns dictionary containing defaults for all valid filter parameters

    CommandLine:
        python -m wbia --tf get_default_annot_filter_form

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.filter_annots import *  # NOQA
        >>> filter_kw = get_default_annot_filter_form()
        >>> print(ut.repr2(filter_kw, align=True))
        >>> print(', '.join(filter_kw.keys()))
    """
    from wbia.expt import annotation_configs

    iden_defaults = annotation_configs.INDEPENDENT_DEFAULTS.copy()
    filter_kw = iden_defaults
    # tag_defaults = get_annot_tag_filterflags(
    #    None, None, {}, request_defaultkw=True)
    # filter_kw = ut.dict_union3(iden_defaults, tag_defaults, combine_op=None)
    return filter_kw


@register_ibs_method
def get_annot_tag_filterflags(ibs, aid_list, filter_kw, request_defaultkw=False):
    r"""
    Filters annotations by tags including those that is belongs to in a pair
    """
    from wbia import tag_funcs

    # Build Filters
    filter_keys = ut.get_func_kwargs(tag_funcs.filterflags_general_tags)

    annotmatch_filterkw = {}
    annot_filterkw = {}
    both_filterkw = {}

    kwreg = ut.KWReg(enabled=request_defaultkw)

    for key in filter_keys:
        annotmatch_filterkw[key] = filter_kw.get(*kwreg(key + '_annotmatch', None))
        annot_filterkw[key] = filter_kw.get(*kwreg(key + '_annot', None))
        both_filterkw[key] = filter_kw.get(*kwreg(key, None))

    if request_defaultkw:
        return kwreg.defaultkw

    # Grab Data
    need_annot_tags = any([var is not None for var in annot_filterkw.values()])
    need_annotmatch_tags = any([var is not None for var in annotmatch_filterkw.values()])
    need_both_tags = any([var is not None for var in both_filterkw.values()])

    if need_annot_tags or need_both_tags:
        annot_tags_list = ibs.get_annot_case_tags(aid_list)

    if need_annotmatch_tags or need_both_tags:
        annotmatch_tags_list = ibs.get_annot_annotmatch_tags(aid_list)

    if need_both_tags:
        both_tags_list = list(
            map(
                ut.unique_ordered,
                map(ut.flatten, zip(annot_tags_list, annotmatch_tags_list)),
            )
        )

    # Filter Data
    flags = np.ones(len(aid_list), dtype=np.bool)
    if need_annot_tags:
        flags_ = tag_funcs.filterflags_general_tags(annot_tags_list, **annot_filterkw)
        np.logical_and(flags_, flags, out=flags)

    if need_annotmatch_tags:
        flags_ = tag_funcs.filterflags_general_tags(
            annotmatch_tags_list, **annotmatch_filterkw
        )
        np.logical_and(flags_, flags, out=flags)

    if need_both_tags:
        flags_ = tag_funcs.filterflags_general_tags(both_tags_list, **both_filterkw)
        np.logical_and(flags_, flags, out=flags)
    return flags


@register_ibs_method
def filterannots_by_tags(ibs, aid_list, filter_kw):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (list):  list of annotation rowids

    CommandLine:
        python -m wbia --tf filterannots_by_tags
        utprof.py -m wbia --tf filterannots_by_tags

    SeeAlso:
        filter_annotmatch_by_tags

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.init.filter_annots import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_Master1')
        >>> aid_list = ibs.get_valid_aids()
        >>> has_any = ut.get_argval('--tags', type_=list,
        >>>                         default=['SceneryMatch', 'Photobomb'])
        >>> min_num = ut.get_argval('--min_num', type_=int, default=1)
        >>> filter_kw = dict(has_any=has_any, min_num=1)
        >>> aid_list_ = filterannots_by_tags(ibs, aid_list, filter_kw)
        >>> print('aid_list_ = %r' % (aid_list_,))
        >>> ut.quit_if_noshow()
        >>> pass
        >>> # TODO: show special annot group in GUI
    """
    flags = get_annot_tag_filterflags(ibs, aid_list, filter_kw)
    aid_list_ = ut.compress(aid_list, flags)
    return aid_list_


def get_acfg_cacheinfo(ibs, aidcfg):
    """
    Returns location and name of the ~~annot~~ data cache
    """
    from os.path import dirname, join

    # Make loading aids a big faster for experiments
    if ut.is_developer():
        import wbia

        repodir = dirname(ut.get_module_dir(wbia))
        acfg_cachedir = join(repodir, 'ACFG_CACHE')
    else:
        # acfg_cachedir = './localdata/ACFG_CACHE'
        acfg_cachedir = join(ibs.get_cachedir(), 'ACFG_CACHE')
        ut.ensuredir(acfg_cachedir)
    acfg_cachename = 'ACFG_CACHE'

    RESPECT_INTERNAL_CFGS = False
    if RESPECT_INTERNAL_CFGS:
        aid_cachestr = ibs.get_dbname() + '_' + ut.hashstr27(ut.to_json(aidcfg))
    else:
        relevant_aidcfg = copy.deepcopy(aidcfg)
        ut.delete_dict_keys(relevant_aidcfg['qcfg'], ut.INTERNAL_CFGKEYS)
        ut.delete_dict_keys(relevant_aidcfg['dcfg'], ut.INTERNAL_CFGKEYS)
        aid_cachestr = ibs.get_dbname() + '_' + ut.hashstr27(ut.to_json(relevant_aidcfg))
    acfg_cacheinfo = (acfg_cachedir, acfg_cachename, aid_cachestr)
    return acfg_cacheinfo


@profile
def expand_single_acfg(ibs, aidcfg, verbose=None):
    """
    for main_helpers """
    from wbia.expt import annotation_configs

    if verbose is None:
        verbose = VERB_TESTDATA
    if verbose:
        print('+=== EXPAND_SINGLE_ACFG ===')
        print(
            ' * acfg = %s'
            % (ut.repr2(annotation_configs.compress_aidcfg(aidcfg), align=True),)
        )
        print('+---------------------')
    avail_aids = ibs._get_all_aids()
    avail_aids = ibs.filter_annotation_set(avail_aids, is_staged=False)
    avail_aids = filter_annots_independent(ibs, avail_aids, aidcfg, verbose=verbose)
    avail_aids = filter_annots_intragroup(ibs, avail_aids, aidcfg, verbose=verbose)
    avail_aids = sample_annots(ibs, avail_aids, aidcfg, verbose=verbose)
    avail_aids = subindex_annots(ibs, avail_aids, aidcfg, verbose=verbose)
    aids = avail_aids
    if verbose:
        print('L___ EXPAND_SINGLE_ACFG ___')
    return aids


@profile
def hack_remove_label_errors(ibs, expanded_aids, verbose=None):
    qaids_, daids_ = expanded_aids

    partitioned_sets = ibs.partition_annots_into_corresponding_groups(qaids_, daids_)
    tup = partitioned_sets
    query_group, data_group, unknown_group, distract_group = tup

    unknown_flags = ibs.unflat_map(
        ibs.get_annot_tag_filterflags,
        unknown_group,
        filter_kw=dict(none_match=['.*error.*']),
    )
    # data_flags  = ibs.unflat_map(
    #    ibs.get_annot_tag_filterflags, data_group,
    #    filter_kw=dict(none_match=['.*error.*']))
    query_flags = ibs.unflat_map(
        ibs.get_annot_tag_filterflags,
        query_group,
        filter_kw=dict(none_match=['.*error.*']),
    )

    query_noterror_flags = list(
        map(
            all,
            ut.list_zipflatten(
                query_flags,
                # data_flags,
            ),
        )
    )
    unknown_noterror_flags = list(map(all, unknown_flags))

    filtered_queries = ut.flatten(ut.compress(query_group, query_noterror_flags))
    filtered_unknown = ut.flatten(ut.compress(unknown_group, unknown_noterror_flags))

    filtered_qaids_ = sorted(filtered_queries + filtered_unknown)

    expanded_aids = (filtered_qaids_, daids_)

    if verbose:
        ut.colorprint('+---------------------', 'red')
        ibs.print_annotconfig_stats(filtered_qaids_, daids_)
        ut.colorprint('L___ HACKED_EXPAND_ACFGS ___', 'red')
    return expanded_aids


@profile
def hack_extra(ibs, expanded_aids):
    # SUCH HACK to get a larger database
    from wbia.expt import annotation_configs

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

    _extra_aids = ibs.get_valid_aids()
    _extra_aids = ibs.remove_groundtrue_aids(_extra_aids, (qaids + daids))
    _extra_aids = filter_annots_independent(ibs, _extra_aids, _aidcfg, prefix)
    _extra_aids = sample_annots(ibs, _extra_aids, _aidcfg, prefix)
    daids = sorted(daids + _extra_aids)
    expanded_aids = (qaids, daids)
    return expanded_aids


def expand_acfgs_consistently(
    ibs, acfg_combo, initial_aids=None, use_cache=None, verbose=None, base=0
):
    r"""
    Expands a set of configurations such that they are comparable

    CommandLine:
        python -m wbia --tf parse_acfg_combo_list  \
                -a varysize
        wbia --tf get_annotcfg_list --db PZ_Master1 -a varysize
        #wbia --tf get_annotcfg_list --db lynx -a default:hack_imageset=True
        wbia --tf get_annotcfg_list --db PZ_Master1 -a varysize:qsize=None
        wbia --tf get_annotcfg_list --db PZ_Master0 --nofilter-dups  -a varysize
        wbia --tf get_annotcfg_list --db PZ_MTEST -a varysize --nofilter-dups
        wbia --tf get_annotcfg_list --db PZ_Master0 --verbtd \
                --nofilter-dups -a varysize
        wbia --tf get_annotcfg_list --db PZ_Master1 -a viewpoint_compare \
                --verbtd --nofilter-dups
        wbia --tf get_annotcfg_list -a timectrl --db GZ_Master1 --verbtd \
                --nofilter-dups

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.filter_annots import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> from wbia.expt import annotation_configs
        >>> from wbia.expt.experiment_helpers import parse_acfg_combo_list
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> #acfg_name_list = ['timectrl:dpername=[1,2]']
        >>> acfg_name_list = ['default:crossval_enc=True,require_timestamp=True']
        >>> aids = ibs.get_valid_aids()
        >>> main_helpers.monkeypatch_encounters(ibs, aids, days=50)
        >>> acfg_combo_list = parse_acfg_combo_list(acfg_name_list)
        >>> acfg_combo = acfg_combo_list[0]
        >>> initial_aids = None
        >>> use_cache = False
        >>> verbose = False
        >>> expanded_aids_combo_list = expand_acfgs_consistently(
        >>>     ibs, acfg_combo, initial_aids=initial_aids, use_cache=use_cache,
        >>>     verbose=verbose)
        >>> # Restore state
        >>> main_helpers.unmonkeypatch_encounters(ibs)
        >>> ut.assert_eq(len(expanded_aids_combo_list), 5)
    """
    from wbia.expt import annotation_configs
    import copy

    if verbose is None:
        verbose = VERB_TESTDATA
    # Edit configs so the sample sizes are consistent
    # FIXME: requiers that smallest configs are specified first

    def tmpmin(a, b):
        if a is None:
            return b
        elif b is None:
            return a
        return min(a, b)

    expanded_aids_list = []

    # Keep track of seen samples
    min_qsize = None
    min_dsize = None

    # HACK: Find out the params being varied and disallow those from being
    # prefiltered due to the lack of heirarchical filters
    nonvaried_dict, varied_acfg_list = annotation_configs.partition_acfg_list(acfg_combo)
    hack_exclude_keys = list(
        set(
            ut.flatten(
                [list(ut.merge_dicts(*acfg.values()).keys()) for acfg in varied_acfg_list]
            )
        )
    )

    # HACK: determine unconstrained min / max nannots
    acfg_combo_in = copy.deepcopy(acfg_combo)

    if False:
        acfg_combo2 = copy.deepcopy(acfg_combo_in)

        unconstrained_expansions = []
        for combox, acfg in enumerate(acfg_combo2):
            qcfg = acfg['qcfg']
            dcfg = acfg['dcfg']
            with ut.Indenter('[PRE %d] ' % (combox,)):
                expanded_aids = expand_acfgs(
                    ibs,
                    acfg,
                    initial_aids=initial_aids,
                    use_cache=use_cache,
                    hack_exclude_keys=hack_exclude_keys,
                    verbose=verbose,
                )
                unconstrained_expansions.append(expanded_aids)

        if any(ut.take_column(ut.take_column(acfg_combo_in, 'dcfg'), 'force_const_size')):
            unconstrained_lens = np.array(
                [(len(q), len(d)) for q, d in unconstrained_expansions]
            )
            # max_dlen = unconstrained_lens.T[1].max()
            min_dlen = unconstrained_lens.T[1].min()

            for acfg in acfg_combo_in:
                dcfg = acfg['dcfg']
                # TODO: make sample size annot_sample_size
                # sample size is #annots
                if dcfg['sample_size'] is None:
                    dcfg['_orig_sample_size'] = dcfg['sample_size']
                    dcfg['sample_size'] = min_dlen

    acfg_combo_out = []

    for combox, acfg in enumerate(acfg_combo_in):
        qcfg = acfg['qcfg']
        dcfg = acfg['dcfg']

        # In some cases we may want to clamp these, but others we do not
        if qcfg['force_const_size']:
            qcfg['_orig_sample_size'] = qcfg['sample_size']
            qcfg['sample_size'] = tmpmin(qcfg['sample_size'], min_qsize)

        if dcfg['force_const_size']:
            dcfg['_orig_sample_size'] = dcfg['sample_size']
            dcfg['sample_size'] = tmpmin(dcfg['sample_size'], min_dsize)

        # Expand modified acfgdict
        with ut.Indenter('[%d] ' % (combox,)):
            expanded_aids = expand_acfgs(
                ibs,
                acfg,
                initial_aids=initial_aids,
                use_cache=use_cache,
                hack_exclude_keys=hack_exclude_keys,
                verbose=verbose,
            )

            # if dcfg.get('hack_extra', None):
            #    assert False
            #    expanded_aids = hack_extra(ibs, expanded_aids)

            qsize = len(expanded_aids[0])
            dsize = len(expanded_aids[1])

            # <hack for float that should not interfere with other hacks
            if qcfg['sample_size'] != qsize:
                qcfg['_orig_sample_size'] = qcfg['sample_size']
            if dcfg['sample_size'] != dsize:
                dcfg['_orig_sample_size'] = dcfg['sample_size']
            # /-->

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

            # so hacky
            # this has to be after sample_size assignment, otherwise the filtering
            # is unstable Remove queries that have labeling errors in them.
            # TODO: fix errors AND remove labels
            # remove_label_errors = ut.is_developer() or ut.get_argflag('--noerrors')
            # ut.is_developer() or ut.get_argflag('--noerrors')
            remove_label_errors = qcfg.get('hackerrors', False)
            if remove_label_errors:
                expanded_aids = hack_remove_label_errors(ibs, expanded_aids, verbose)

        if qcfg['crossval_enc'] or dcfg['crossval_enc']:
            # Hack, just use qaids for cross validated sampleing
            aids = expanded_aids[0]
            qenc_per_name = qcfg['crossval_enc']
            denc_per_name = dcfg['crossval_enc']
            if qenc_per_name is None:
                qenc_per_name = 1
            if denc_per_name is None:
                denc_per_name = 1
            crossval_expansion = encounter_crossval(
                ibs, aids, qenc_per_name=qenc_per_name, denc_per_name=denc_per_name
            )

            import uuid

            unique_joinme = uuid.uuid4()

            for count, aid_pairs in enumerate(crossval_expansion):
                acfg_out = copy.deepcopy(acfg)
                acfg_out['qcfg']['crossval_idx'] = count
                acfg_out['dcfg']['crossval_idx'] = count
                acfg_out['qcfg']['sample_size'] = len(aid_pairs[0])
                acfg_out['dcfg']['sample_size'] = len(aid_pairs[1])
                # FIMXE: needs to be different for all acfgs
                # out of this sample.
                if acfg_out['qcfg'].get('joinme', None) is None:
                    acfg_out['qcfg']['joinme'] = unique_joinme
                    acfg_out['dcfg']['joinme'] = unique_joinme
                # need further hacks to assign sample size correctly
                # after the crossval hack
                acfg_combo_out.append(acfg_out)
            expanded_aids_list.extend(crossval_expansion)

        else:
            acfg_combo_out.append(acfg)
            # ibs.print_annotconfig_stats(*expanded_aids)
            expanded_aids_list.append(expanded_aids)

    # Sample afterwords
    return list(zip(acfg_combo_out, expanded_aids_list))


def crossval_helper(
    nid_to_sample_pool,
    perquery,
    perdatab,
    n_need,
    n_splits=None,
    rng=None,
    rebalance=True,
):
    """
    does sampling based on some grouping (or no grouping) of annots

    perquery = 2
    perdatab = 2

    nid_to_sample_pool = {
        1: [1, 2, 3, 4],
        2: [6, 7, 8, 9],
    }
    """
    if len(nid_to_sample_pool) == 0:
        raise ValueError(
            'Names do not have enough data for %d/%d split' % (perquery, perdatab)
        )
    rng = ut.ensure_rng(rng, impl='python')

    def split_combos(pool, perquery, perdatab, rng):
        import scipy.special

        poolsize = len(pool)
        # Number of ways we can select queries
        n_qmax = int(scipy.special.comb(poolsize, perquery))
        # Number of ways we can select targets from remaining items
        n_dmax = int(scipy.special.comb(poolsize - perquery, perdatab))
        # Total number of query / data combinations
        n_combos = n_qmax * n_dmax

        # Yield random combinations until we get something we havent seen
        poolset = set(pool)
        splits = set()
        while len(splits) < n_combos:
            # combo = tuple(sorted(rng.choice(items, size, replace=False)))
            qcombo = tuple(sorted(rng.sample(pool, perquery)))
            remain = poolset - set(qcombo)
            dcombo = tuple(sorted(rng.sample(remain, perdatab)))
            # TODO: try not to use queries / databases that we've used before
            # until we've exhauseted those possibilities.
            split = (qcombo, dcombo)
            if split not in splits:
                splits.add(split)
                yield split

    if n_splits is None:
        # What is the maximum number of items in a name?
        maxsize_name = max(map(len, nid_to_sample_pool.values()))
        # This is only a heuristic
        n_splits = maxsize_name

    # Create a mapping from each name to a list of query/target splits
    nid_to_splits = ut.ddict(list)
    # Create several splits for each name
    for nid, pool in nid_to_sample_pool.items():
        # Randomly select up to `n_splits` combinations of size `n_need`.
        combo_iter = split_combos(pool, perquery, perdatab, rng)
        for count, fold_split in enumerate(combo_iter, start=1):
            # Earlier samples will be biased towards names with more annots
            nid_to_splits[nid].append(fold_split)
            if count >= n_splits:
                break

    # print(ut.repr2(list(nid_to_splits.values()), strvals=True, nl=2))

    # Some names may have more splits than others
    # nid_to_nsplits = ut.map_vals(len, nid_to_splits)
    # Find the name with the most splits
    # max_nid = ut.argmax(nid_to_nsplits)
    # max_size = nid_to_nsplits[max_nid]

    new_splits = [[] for _ in range(n_splits)]
    if rebalance:
        # Rebalance by adding combos from each name in a cycle.
        # The difference between the largest and smallest split is at most one.
        for count, split in enumerate(ut.iflatten(nid_to_splits.values())):
            new_splits[count % len(new_splits)].append(split)
    else:
        # No rebalancing. The first split contains everything from the dataset
        # and subsequent splits contain less and less.
        for nid, combos in nid_to_splits.items():
            for count, split in enumerate(combos):
                new_splits[count].append(split)

    # Reshape into an expanded aids list
    # List of query / database objects per split, grouped by name.
    reshaped_splits = [ut.listT(splits) for splits in new_splits if len(splits) > 0]
    return reshaped_splits


def encounter_crossval(
    ibs,
    aids,
    qenc_per_name=1,
    denc_per_name=1,
    enc_labels=None,
    confusors=True,
    rng=None,
    annots_per_enc=None,
    rebalance=True,
    n_splits=None,
    early=False,
):
    r"""
    Constructs a list of [ (qaids, daids) ] where there are `qenc_per_name` and
    `denc_per_name` for each individual in the datasets respectively.
    `enc_labels` specifies custom encounter labels.

    CommandLine:
        python -m wbia.init.filter_annots encounter_crossval

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.filter_annots import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> import wbia
        >>> #ibs, aids = wbia.testdata_aids(
        >>> #    defaultdb='WWF_Lynx_Copy',
        >>> #    a='default:minqual=good,require_timestamp=True,view=left')
        >>> ibs, aids = wbia.testdata_aids(defaultdb='PZ_MTEST',
        >>>                                 a='default:require_timestamp=True')
        >>> main_helpers.monkeypatch_encounters(ibs, aids, days=50)
        >>> qenc_per_name = 2
        >>> denc_per_name = 2
        >>> confusors = False
        >>> print('denc_per_name = %r' % (denc_per_name,))
        >>> print('qenc_per_name = %r' % (qenc_per_name,))
        >>> rng = 0
        >>> n_splits = 5
        >>> expanded_aids = encounter_crossval(ibs, aids, n_splits=n_splits,
        >>>                                    qenc_per_name=qenc_per_name,
        >>>                                    denc_per_name=denc_per_name,
        >>>                                    confusors=confusors, rng=rng)
        >>> # ensure stats agree
        >>> cfgargs = dict(per_vp=False, per_multiple=False, combo_dists=False,
        >>>                per_name=False, per_enc=True, use_hist=False)
        >>> for qaids, daids in expanded_aids:
        >>>     stats = ibs.get_annotconfig_stats(qaids, daids, **cfgargs)
        >>>     del stats['confusor_daid_stats']
        >>>     print(ut.repr2(stats, strvals=True, strkeys=True, nl=2))
        >>>     denc_stats = stats['matchable_daid_stats']['denc_per_name']
        >>>     qenc_stats = stats['qaid_stats']['qenc_per_name']
        >>>     assert denc_stats['min'] == denc_stats['max']
        >>>     assert denc_stats['min'] == denc_per_name
        >>>     assert qenc_stats['min'] == qenc_stats['max']
        >>>     assert qenc_stats['min'] == qenc_per_name
        >>> # Restore state
        >>> main_helpers.unmonkeypatch_encounters(ibs)
        >>> #qaids, daids = expanded_aids[0]
        >>> #stats = ibs.get_annotconfig_stats(qaids, daids, use_hist=True)
        >>> #print(ut.repr2(stats, strvals=True, strkeys=True, nl=2))

    """
    qenc_per_name = int(qenc_per_name)
    denc_per_name = int(denc_per_name)

    # We can only select individuals with enough encounters
    # Any name without enought data becomes a confusor
    n_need = qenc_per_name + denc_per_name
    perquery = qenc_per_name
    perdatab = denc_per_name

    annots = ibs.annots(aids)
    if enc_labels is None:
        enc_labels = annots.encounter_text

    # Group annotations by encounter
    encounters = ibs._annot_groups(annots.group(enc_labels)[1])
    enc_nids = ut.take_column(encounters.nids, 0)
    rng = ut.ensure_rng(rng, impl='python')
    if annots_per_enc is not None:
        encounters = [rng.sample(list(a), annots_per_enc) for a in encounters]
    # Group encounters by name
    nid_to_encs = ut.group_items(encounters, enc_nids)

    nid_to_confusors = {
        nid: ut.lmap(tuple, encs)
        for nid, encs in nid_to_encs.items()
        if len(encs) < n_need
    }
    nid_to_sample_pool = {
        nid: ut.lmap(tuple, encs)
        for nid, encs in nid_to_encs.items()
        if len(encs) >= n_need
    }

    reshaped_splits = crossval_helper(
        nid_to_sample_pool,
        perquery,
        perdatab,
        n_splits=n_splits,
        n_need=n_need,
        rng=rng,
        rebalance=rebalance,
    )
    if early:
        return reshaped_splits, nid_to_confusors

    # print(ut.repr4(reshaped_splits, nl=3))
    expanded_aids_list = [
        [ut.flatten(qpart), ut.flatten(dpart)] for qpart, dpart in reshaped_splits
    ]

    expanded_aids_list = [
        [sorted(ut.flatten(ut.flatten(qpart))), sorted(ut.flatten(ut.flatten(dpart)))]
        for qpart, dpart in reshaped_splits
    ]

    if confusors:
        # Add confusors the the dataset
        confusor_aids = ut.flatten(ut.flatten(nid_to_confusors.values()))
        expanded_aids_list = [
            (qaids, sorted(daids + confusor_aids)) for qaids, daids in expanded_aids_list
        ]
    return expanded_aids_list


def annot_crossval(
    ibs,
    aid_list,
    n_qaids_per_name=1,
    n_daids_per_name=1,
    rng=None,
    debug=True,
    n_splits=None,
    confusors=True,
):
    """
    Stratified sampling per name size

    Args:
        n_splits (int): number of query/database splits to create.
            note, some names may not be big enough to split this many times.

    CommandLine:
        python -m wbia.init.filter_annots annot_crossval

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.filter_annots import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> n_qaids_per_name = 2
        >>> n_daids_per_name = 3
        >>> rng = 0
        >>> debug = True
        >>> n_splits = None
        >>> expanded_aids_list = annot_crossval(
        >>>     ibs, aid_list, n_qaids_per_name, n_daids_per_name, rng, debug,
        >>>     n_splits, confusors=False)
        >>> result = ('expanded_aids_list = %s' % (ut.repr2(expanded_aids_list, nl=2),))
        >>> print(result)
    """
    # Parameters
    n_need = n_qaids_per_name + n_daids_per_name
    rebalance = True

    # Group annotations by name
    annots = ibs.annots(aids=aid_list)
    nid_to_aids = ut.group_items(annots.aids, annots.nids)

    # Any name without enough data becomes a confusor
    # Otherwise we can use it in the sampling pool
    nid_to_confusors = {
        nid: aids for nid, aids in nid_to_aids.items() if len(aids) < n_need
    }
    nid_to_sample_pool = {
        nid: aids for nid, aids in nid_to_aids.items() if len(aids) >= n_need
    }

    perquery = n_qaids_per_name
    perdatab = n_daids_per_name

    reshaped_splits = crossval_helper(
        nid_to_sample_pool,
        perquery,
        perdatab,
        n_splits=n_splits,
        n_need=n_need,
        rng=rng,
        rebalance=rebalance,
    )

    expanded_aids_list = [
        [sorted(ut.flatten(qpart)), sorted(ut.flatten(dpart))]
        for qpart, dpart in reshaped_splits
    ]

    if confusors:
        # Add confusors the the dataset
        confusor_aids = ut.flatten(nid_to_confusors.values())
        expanded_aids_list = [
            (qaids, sorted(daids + confusor_aids)) for qaids, daids in expanded_aids_list
        ]

    # if debug:
    #     debug_expanded_aids(expanded_aids_list)
    return expanded_aids_list

    # # Group annotations by encounter
    # unique_encounters, groupxs = annots.group_indicies(enc_labels)
    # encounter_nids = annots.take(ut.take_column(groupxs, 0)).nid
    # encounter_aids = ut.apply_grouping(annots.aids, groupxs)
    # # Group encounters by name
    # nid_to_encs = ut.group_items(encounter_aids, encounter_nids)

    # nid_to_confusors = {nid: enc for nid, enc in nid_to_encs.items()
    #                     if len(enc) <  n_need}
    # nid_to_sample_pool = {nid: enc for nid, enc in nid_to_encs.items()
    #                       if len(enc) >= n_need}

    # # Randomly shuffle encounters for each individual
    # valid_enc_rand_idxs = [(nid, ut.random_indexes(len(enc), rng=rng))
    #                        for nid, enc in nid_to_sample_pool.items()]

    # # For each individual choose a set of query and database encounters
    # crossval_idx_samples = []
    # max_num_encounters = max(map(len, (t[1] for t in valid_enc_rand_idxs)))
    # # TODO: iterate over a sliding window to choose multiple queries OR
    # # iterate over all combinations of queries... (harder)
    # for i in range(max_num_encounters):
    #     encx_split = {}
    #     for nid, idxs in valid_enc_rand_idxs:
    #         if i < len(idxs):
    #             # Choose the database encounters from anything not chosen as a
    #             # query
    #             d_choices = ut.where(ut.not_list(
    #                 ut.index_to_boolmask([i], len(idxs))))
    #             js = rng.choice(d_choices, size=denc_per_name, replace=False)
    #             encx_split[nid] = (idxs[i:i + 1], idxs[js])
    #     crossval_idx_samples.append(encx_split)

    # # convert to aids
    # crossval_aid_samples = []
    # for encx_split in crossval_idx_samples:
    #     aid_split = {}
    #     for nid, (qxs, dxs) in encx_split.items():
    #         qaids = ut.flatten(ut.take(nid_to_sample_pool[nid], qxs))
    #         daids = ut.flatten(ut.take(nid_to_sample_pool[nid], dxs))
    #         assert len(dxs) == denc_per_name
    #         assert len(qxs) == qenc_per_name
    #         aid_split[nid] = (qaids, daids)
    #     crossval_aid_samples.append(aid_split)

    # tups = [(nid, aids_) for aid_split_ in crossval_aid_samples
    #         for nid, aids_ in aid_split_.items()]
    # groups = ut.take_column(tups, 0)
    # aidpairs = ut.take_column(tups, 1)
    # # crossval_samples[0]

    # # rebalance the queries
    # # Rewrite using rebalance code from crossval_annots
    # # Very inefficient but does what I want
    # group_to_idxs = ut.dzip(*ut.group_indices(groups))
    # freq = ut.dict_hist(groups)
    # g = list(freq.keys())[ut.argmax(list(freq.values()))]
    # size = freq[g]
    # new_splits = [[] for _ in range(size)]
    # while True:
    #     try:
    #         g = list(freq.keys())[ut.argmax(list(freq.values()))]
    #         if freq[g] == 0:
    #             raise StopIteration()
    #         group_idxs = group_to_idxs[g]
    #         group_to_idxs[g] = []
    #         freq[g] = 0
    #         priorityx = ut.argsort(list(map(len, new_splits)))
    #         for nextidx, splitx in zip(group_idxs, priorityx):
    #             new_splits[splitx].append(nextidx)
    #     except StopIteration:
    #         break
    # # name_splits = ut.unflat_take(groups, new_splits)
    # aid_splits = ut.unflat_take(aidpairs, new_splits)

    # # Add annots that could not meet these requirements are distractors
    # confusors = ut.flatten(ut.flatten(list(nid_to_confusors.values())))

    # expanded_aids_list = []
    # for aidsplit in aid_splits:
    #     qaids = sorted(ut.flatten(ut.take_column(aidsplit, 0)))
    #     daids = sorted(ut.flatten(ut.take_column(aidsplit, 1)) + confusors)
    #     expanded_aids_list.append((qaids, daids))
    # return expanded_aids_list


@profile
def expand_acfgs(
    ibs,
    aidcfg,
    verbose=None,
    use_cache=None,
    hack_exclude_keys=None,
    initial_aids=None,
    save_cache=True,
):
    r"""
    Main multi-expansion function. Expands an annot config dict into qaids and
    daids.  New version of this function based on a configuration dictionary
    built from command line argumetns

    Args:
        ibs (IBEISController):  wbia controller object
        aidcfg (dict): configuration of the annotation filter
        verbose (bool):  verbosity flag(default = False)
        use_cache (bool):  turns on disk based caching(default = None)
        hack_exclude_keys (None): (default = None)
        initial_aids (None): (default = None)

    Returns:
        tuple: expanded_aids=(qaid_list, daid_list) - expanded list of aids
            that meet the criteria of the aidcfg filter

    TODO:
        The database should be created first in most circumstances, then
        the queries should be filtered to meet the database restrictions?
        I'm not sure Sometimes you need to set the query aids constant, but
        sometimes you need to set the data aids constant. Seems to depend.

        This function very much needs the idea of filter chains

        OkNewIdea:
            3 filters:
                * Common sampling - takes care of things like min time delta,
                * species, quality viewpoint etc.
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
        python -m wbia.dev -e print_acfg  -a timectrl:qsize=10,dsize=10  --db PZ_MTEST --veryverbtd --nocache-aid
        python -m wbia.dev -e print_acfg  -a timectrl:qminqual=good,qsize=10,dsize=10  --db PZ_MTEST --veryverbtd --nocache-aid

        python -m wbia.dev -e print_acfg  -a timectrl --db PZ_MTEST --verbtd --nocache-aid
        python -m wbia.dev -e print_acfg  -a timectrl --db PZ_Master1 --verbtd --nocache-aid
        python -m wbia.dev -e print_acfg  -a timequalctrl --db PZ_Master1 --verbtd --nocache-aid

        python -m wbia.dev -e rank_cmc   -a controlled:qsize=10,dsize=10,dper_name=2 -t default --db PZ_MTEST
        python -m wbia.dev -e rank_cmc   -a controlled:qsize=10,dsize=20,dper_name=2 -t default --db PZ_MTEST
        python -m wbia.dev -e print      -a controlled:qsize=10,dsize=10             -t default --db PZ_MTEST --verbtd --nocache-aid

        python -m wbia.dev -e latexsum -t candinvar -a viewpoint_compare  --db NNP_Master3 --acfginfo
        utprof.py -m wbia.dev -e print -t candk -a varysize  --db PZ_MTEST --acfginfo
        utprof.py -m wbia.dev -e latexsum -t candk -a controlled  --db PZ_Master0 --acfginfo

        python -m wbia --tf get_annotcfg_list:0 --db NNP_Master3 -a viewpoint_compare --nocache-aid --verbtd

        python -m wbia --tf get_annotcfg_list  --db PZ_Master1 \
            -a timectrl:qhas_any=\(needswork,correctable,mildviewpoint\),qhas_none=\(viewpoint,photobomb,error:viewpoint,quality\) \
            --acfginfo --veryverbtd  --veryverbtd
        python -m wbia --tf draw_rank_cmc --db PZ_Master1 --show -t best \
            -a timectrl:qhas_any=\(needswork,correctable,mildviewpoint\),qhas_none=\(viewpoint,photobomb,error:viewpoint,quality\) \
            --acfginfo --veryverbtd

        python -m wbia --tf get_annotcfg_list  --db Oxford -a default:qhas_any=\(query,\),dpername=2,exclude_reference=True --acfginfo --verbtd  --veryverbtd --nocache-aid

    CommandLine:
        python -m wbia.init.filter_annots --exec-expand_acfgs --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.filter_annots import *  # NOQA
        >>> import wbia
        >>> from wbia.expt import annotation_configs
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aidcfg = copy.deepcopy(annotation_configs.default)
        >>> aidcfg['qcfg']['species'] = 'primary'
        >>> initial_aids = None
        >>> expanded_aids = expand_acfgs(ibs, aidcfg, initial_aids=initial_aids)
        >>> result = ut.repr3(expanded_aids, nl=1, nobr=True)
        >>> print(result)
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    """
    from wbia.expt import annotation_configs

    if verbose is None:
        verbose = VERB_TESTDATA

    assert isinstance(aidcfg, dict), 'type(aidcfg)=%r' % (type(aidcfg),)
    aidcfg = copy.deepcopy(aidcfg)

    # Check if this filter has been cached
    # TODO: keep a database state config that augments the cachestr?
    if use_cache is None:
        use_cache = USE_ACFG_CACHE

    # save_cache = True
    if use_cache and save_cache:
        acfg_cacheinfo = get_acfg_cacheinfo(ibs, aidcfg)
        acfg_cachedir, acfg_cachename, aid_cachestr = acfg_cacheinfo
    if use_cache:
        aids_tup = ut.tryload_cache(acfg_cachedir, acfg_cachename, aid_cachestr)
        if aids_tup is not None:
            (qaid_list, daid_list) = aids_tup
            return qaid_list, daid_list

    comp_acfg = annotation_configs.compress_aidcfg(aidcfg)

    if verbose:
        ut.colorprint('+=== EXPAND_ACFGS ===', 'yellow')
        print(' * acfg = %s' % (ut.repr2(comp_acfg, align=True),))
        ut.colorprint('+---------------------', 'yellow')

    # Breakup into common, query, and database configs
    qcfg = aidcfg['qcfg']
    dcfg = aidcfg['dcfg']
    common_cfg = comp_acfg['common']

    # Extract the common independent filtering params
    idenfilt_cfg_default = annotation_configs.INDEPENDENT_DEFAULTS
    idenfilt_cfg_empty = {key: None for key in idenfilt_cfg_default.keys()}
    idenfilt_cfg_common = ut.update_existing(idenfilt_cfg_empty, common_cfg, copy=True)

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

    # if aidcfg['qcfg']['hack_imageset'] is True:
    #    return ibs.get_imageset_expanded_aids()
    # Hack: Make hierarchical filters to supersede this
    if initial_aids is None:
        initial_aids = ibs._get_all_aids()

    initial_aids = ibs.filter_annotation_set(initial_aids, is_staged=False)

    verbflags = dict(verbose=verbose)
    qfiltflags = dict(prefix='q', **verbflags)
    dfiltflags = dict(prefix='d', **verbflags)

    default_aids = initial_aids

    # A chain of filters on all of the aids
    global_filter_chain = [
        (filter_annots_independent, idenfilt_cfg_common),
        (filter_annots_intragroup, idenfilt_cfg_common),
    ]

    # Chains of filters individually for each partition
    partition_chains = [
        [
            # Query partition chain
            (filter_annots_independent, qcfg),
            (filter_annots_intragroup, qcfg),
            (sample_annots, qcfg),
        ],
        [
            # Database partition chain
            (filter_annots_independent, dcfg),
            (filter_annots_intragroup, dcfg),
            (sample_annots_wrt_ref, dcfg, 0),
        ],
    ]
    try:
        # GLOBAL FILTER CHAIN
        # applies filtering to all available aids
        for filtfn, filtcfg in global_filter_chain:
            default_aids = filtfn(
                ibs, default_aids, filtcfg, prefix='', withpre=True, **verbflags
            )

        # PARTITION FILTER CHAIN
        # chain of filters for query / database annots
        default_qaids = default_daids = default_aids
        partition_avail_aids = [default_qaids, default_daids]
        partion_kwargs = [qfiltflags, dfiltflags]
        for index in range(len(partition_chains)):
            filter_chain = partition_chains[index]
            avail_aids = partition_avail_aids[index]
            _partkw = partion_kwargs[index].copy()
            for filter_tup in filter_chain:
                filtfn, filtcfg = filter_tup[0:2]
                if len(filter_tup) == 3:
                    # handle filters that take reference sets
                    refindex = filter_tup[2]
                    ref_aids = partition_avail_aids[refindex]
                    _partkw['ref_aids'] = ref_aids
                # Execute filtering
                avail_aids = filtfn(ibs, avail_aids, filtcfg, **_partkw)
            partition_avail_aids[index] = avail_aids

        # SUBINDEX EACH PARTITIONED CHAIN
        subindex_cfgs = [qcfg, dcfg]
        for index in range(len(partition_avail_aids)):
            avail_aids = partition_avail_aids[index]
            _partkw = partion_kwargs[index]
            filtcfg = subindex_cfgs[index]
            avail_aids = subindex_annots(ibs, avail_aids, filtcfg, **_partkw)
            partition_avail_aids[index] = avail_aids

        # UNPACK FILTER RESULTS
        avail_qaids, avail_daids = partition_avail_aids

    except Exception as ex:
        print('PRINTING ERROR INFO')
        print(' * acfg = %s' % (ut.repr2(comp_acfg, align=True),))
        ut.printex(ex, 'Error executing filter chains')
        raise

    qaid_list = sorted(avail_qaids)
    daid_list = sorted(avail_daids)

    if verbose:
        ut.colorprint('+---------------------', 'yellow')
        ibs.print_annotconfig_stats(qaid_list, daid_list)
        ut.colorprint('L___ EXPAND_ACFGS ___', 'yellow')

    # Save filter to cache
    if use_cache and save_cache:
        ut.ensuredir(acfg_cachedir)
        try:
            ut.save_cache(
                acfg_cachedir, acfg_cachename, aid_cachestr, (qaid_list, daid_list)
            )
        except IOError:
            pass
    return qaid_list, daid_list


def expand_species(ibs, species, avail_aids=None):
    if species == 'primary':
        species = ibs.get_primary_database_species()
    if species is None and avail_aids is not None:
        species = ibs.get_dominant_species(avail_aids)
    return species


@profile
@register_ibs_method
def filter_annots_independent(
    ibs, avail_aids, aidcfg, prefix='', verbose=VERB_TESTDATA, withpre=False
):
    r"""
    Filtering that doesn't have to do with a reference set of aids

    TODO make filterflags version

    Args:
        ibs (IBEISController):  wbia controller object
        avail_aids (list):
        aidcfg (dict):
        prefix (str): (default = '')
        verbose (bool):  verbosity flag(default = False)

    Returns:
        list: avail_aids

    CommandLine:
        python -m wbia --tf filter_annots_independent --veryverbtd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.init.filter_annots import *  # NOQA
        >>> import wbia
        >>> from wbia.expt import annotation_configs
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> avail_aids = input_aids = ibs.get_valid_aids()
        >>> aidcfg = annotation_configs.default['dcfg']
        >>> aidcfg['require_timestamp'] = True
        >>> aidcfg['require_quality'] = False
        >>> aidcfg['is_known'] = True
        >>> prefix = ''
        >>> verbose = True
        >>> avail_aids = filter_annots_independent(ibs, avail_aids, aidcfg,
        >>>                                        prefix, verbose)
        >>> result = ('avail_aids = %s' % (str(avail_aids),))
        >>> print(result)

    Ignore:
        # Testing tag features
        python -m wbia --tf draw_rank_cmc --db PZ_Master1 --show -t best \
            -a timectrl:qhas_any=\(needswork,correctable,mildviewpoint\),qhas_none=\(viewpoint,photobomb,error:viewpoint,quality\) \
            ---acfginfo --veryverbtd
    """
    from wbia.other import ibsfuncs

    if aidcfg is None:
        if verbose:
            print('No annot filter returning')
        return avail_aids

    VerbosityContext = verb_context('FILTER_INDEPENDENT', aidcfg, verbose)
    VerbosityContext.startfilter(withpre=withpre)

    if aidcfg.get('is_known') is True:
        with VerbosityContext('is_known'):
            avail_aids = ibs.filter_aids_without_name(
                avail_aids, invert=not aidcfg['is_known']
            )
        # avail_aids = sorted(avail_aids)

    if aidcfg.get('is_exemplar') is not None:
        flags = ibs.get_annot_exemplar_flags(avail_aids)
        is_valid = [flag == aidcfg['is_exemplar'] for flag in flags]
        with VerbosityContext('is_exemplar'):
            avail_aids = ut.compress(avail_aids, is_valid)
        # avail_aids = sorted(avail_aids)

    if aidcfg.get('reviewed') is not None:
        flags = ibs.get_annot_reviewed(avail_aids)
        is_valid = [flag == aidcfg['reviewed'] for flag in flags]
        with VerbosityContext('reviewed'):
            avail_aids = ut.compress(avail_aids, is_valid)
        # avail_aids = sorted(avail_aids)

    if aidcfg.get('multiple') is not None:
        flags = ibs.get_annot_multiple(avail_aids)
        is_valid = [flag == aidcfg['multiple'] for flag in flags]
        with VerbosityContext('multiple'):
            avail_aids = ut.compress(avail_aids, is_valid)
        # avail_aids = sorted(avail_aids)

    if aidcfg.get('require_timestamp') is True:
        with VerbosityContext('require_timestamp'):
            avail_aids = ibs.filter_aids_without_timestamps(avail_aids)

    if aidcfg.get('require_gps') is True:
        with VerbosityContext('require_gps'):
            annots = ibs.annots(avail_aids)
            annots = annots.compress(~np.isnan(np.array(annots.gps)).any(axis=1))
            avail_aids = annots.aids
        # avail_aids = sorted(avail_aids)

    if aidcfg.get('max_timestamp') is not None:
        with VerbosityContext('max_timestamp'):
            max_dt = aidcfg.get('max_timestamp')
            import datetime

            if isinstance(max_dt, (int, float)):
                max_unixtime = max_dt
            elif isinstance(max_dt, six.string_types):
                if max_dt == 'now':
                    max_dt = datetime.datetime.utcnow()
                else:
                    max_dt = max_dt.replace('/', '-')
                    y, m, d = max_dt.split('-')
                    max_dt = ut.date_to_datetime(datetime.date(y, m, d))
            max_unixtime = ut.datetime_to_posixtime(max_dt)
            unixtimes = np.array(ibs.annots(avail_aids).image_unixtimes_asfloat)
            flag_list = np.logical_or(np.isnan(unixtimes), unixtimes <= max_unixtime)
            ut.compress(avail_aids, flag_list)
        # avail_aids = sorted(avail_aids)

    cfg_species = aidcfg.get('species')
    if isinstance(cfg_species, six.string_types) and cfg_species.lower() == 'none':
        cfg_species = None

    metadata = ut.LazyDict(species=lambda: expand_species(ibs, cfg_species, None))

    if cfg_species is not None:
        species = metadata['species']
        with VerbosityContext('species', species=species):
            avail_aids = ibs.filter_aids_to_species(avail_aids, species)
            # avail_aids = sorted(avail_aids)

    if aidcfg.get('been_adjusted', None):
        # HACK to see if the annotation has been adjusted from the default
        # value set by dbio.ingest_database
        flag_list = ibs.get_annot_been_adjusted(avail_aids)
        with VerbosityContext('been_adjusted'):
            avail_aids = ut.compress(avail_aids, flag_list)

    if aidcfg.get('contributor_contains', None):
        contributor_contains = aidcfg['contributor_contains']
        gid_list = ibs.get_annot_gids(avail_aids)
        tag_list = ibs.get_image_contributor_tag(gid_list)
        flag_list = [contributor_contains in tag for tag in tag_list]
        with VerbosityContext('contributor_contains'):
            avail_aids = ut.compress(avail_aids, flag_list)

    if aidcfg.get('minqual') is not None or aidcfg.get('require_quality'):
        minqual = 'junk' if aidcfg['minqual'] is None else aidcfg['minqual']
        with VerbosityContext('minqual', 'require_quality'):
            # Filter quality
            avail_aids = ibs.filter_aids_to_quality(
                avail_aids, minqual, unknown_ok=not aidcfg['require_quality']
            )
        # avail_aids = sorted(avail_aids)

    if aidcfg.get('max_unixtime', None) is not None:
        max_unixtime = aidcfg.get('max_unixtime', None)
        unixtimes = np.array(ibs.get_annot_image_unixtimes_asfloat(avail_aids))
        flags = unixtimes <= max_unixtime
        with VerbosityContext('max_unixtime'):
            avail_aids = ut.compress(avail_aids, flags)
        # avail_aids = sorted(avail_aids)

    if aidcfg.get('min_unixtime', None) is not None:
        min_unixtime = aidcfg.get('min_unixtime', None)
        unixtimes = np.array(ibs.get_annot_image_unixtimes_asfloat(avail_aids))
        flags = unixtimes >= min_unixtime
        with VerbosityContext('min_unixtime'):
            avail_aids = ut.compress(avail_aids, flags)
        # avail_aids = sorted(avail_aids)

    if aidcfg.get('max_numfeat') is not None or aidcfg.get('min_numfeat') is not None:
        max_numfeat = aidcfg['max_numfeat']
        min_numfeat = aidcfg['min_numfeat']
        if max_numfeat is None:
            max_numfeat = np.inf
        if min_numfeat is None:
            min_numfeat = 0
        numfeat_list = np.array(ibs.get_annot_num_feats(avail_aids))
        flags_list = np.logical_and(
            numfeat_list >= min_numfeat, numfeat_list <= max_numfeat
        )
        with VerbosityContext('max_numfeat', 'min_numfeat'):
            avail_aids = ut.compress(avail_aids, flags_list)

    if aidcfg.get('view') is not None or aidcfg.get('require_viewpoint'):
        # Resolve base viewpoint
        if aidcfg['view'] == 'primary':
            view = ibsfuncs.get_primary_species_viewpoint(metadata['species'])
        elif aidcfg['view'] == 'primary1':
            view = ibsfuncs.get_primary_species_viewpoint(metadata['species'], 1)
        else:
            view = aidcfg['view']
        if isinstance(view, six.string_types) and view.lower() == 'none':
            view = None
        OLD = False
        if OLD:
            view_ext1 = (
                aidcfg['view_ext'] if aidcfg['view_ext1'] is None else aidcfg['view_ext1']
            )
            view_ext2 = (
                aidcfg['view_ext'] if aidcfg['view_ext2'] is None else aidcfg['view_ext2']
            )
            valid_yaws = ibsfuncs.get_extended_viewpoints(
                view, num1=view_ext1, num2=view_ext2
            )
            unknown_ok = not aidcfg['require_viewpoint']
            with VerbosityContext(
                'view',
                'require_viewpoint',
                'view_ext',
                'view_ext1',
                'view_ext2',
                valid_yaws=valid_yaws,
            ):
                avail_aids = ibs.filter_aids_to_viewpoint(
                    avail_aids, valid_yaws, unknown_ok=unknown_ok
                )
            avail_aids = sorted(avail_aids)
        else:

            def rectify_view(vstr):
                # FIXME: I stopped implementing the += stuff
                vstr_num = vstr.lower()
                num = 0
                if not vstr_num.endswith('1'):
                    vstr = vstr_num
                else:
                    if '+' in vstr:
                        vstr, numstr = vstr_num.split('+')
                        num = int(numstr)
                    if '-' in vstr:
                        vstr, numstr = vstr_num.split('+')
                        num = -int(numstr)
                assert num == 0, 'cant do += yet'
                if vstr == 'primary':
                    return ibsfuncs.get_primary_species_viewpoint(metadata['species'])
                for yawtxt, other_yawtxt in ibs.const.YAWALIAS.items():
                    other_yawtxt = ut.ensure_iterable(other_yawtxt)
                    if vstr == yawtxt.lower():
                        return yawtxt
                    for x in other_yawtxt:
                        if vstr == x.lower():
                            return yawtxt
                raise ValueError('unknown viewpoint vstr=%r' % (vstr,))

            if view is None:
                valid_yaw_txts = None
            else:
                valid_yaw_txts = [
                    rectify_view(vstr) for vstr in ut.smart_cast(view, list)
                ]
            unknown_ok = not aidcfg['require_viewpoint']
            yaw_flags = ibs.get_viewpoint_filterflags(
                avail_aids, valid_yaw_txts, unknown_ok=unknown_ok, assume_unique=True
            )
            yaw_flags = list(yaw_flags)
            with VerbosityContext(
                'view',
                'require_viewpoint',
                'view_ext',
                'view_ext1',
                'view_ext2',
                valid_yaws=valid_yaw_txts,
            ):
                avail_aids = ut.compress(avail_aids, yaw_flags)

    # if aidcfg.get('exclude_view') is not None:
    #    raise NotImplementedError('view tag resolution of exclude_view')
    #    # Filter viewpoint
    #    # TODO need to resolve viewpoints
    #    exclude_view = aidcfg.get('exclude_view')
    #    with VerbosityContext('exclude_view', hack=True):
    #        avail_aids = ibs.remove_aids_of_viewpoint(
    #            avail_aids, exclude_view)

    if aidcfg.get('min_pername_global') is not None:
        # Keep annots with at least this many groundtruths in the database
        min_pername_global = aidcfg.get('min_pername_global')
        num_gt_global_list = ibs.get_annot_num_groundtruth(avail_aids, noself=False)
        flag_list = np.array(num_gt_global_list) >= min_pername_global
        with VerbosityContext('exclude_view'):
            avail_aids = ut.compress(avail_aids, flag_list)
        # avail_aids = sorted(avail_aids)

    if aidcfg.get('max_pername_global') is not None:
        max_pername_global = aidcfg.get('max_pername_global')
        num_gt_global_list = ibs.get_annot_num_groundtruth(avail_aids, noself=False)
        flag_list = np.array(num_gt_global_list) <= max_pername_global
        with VerbosityContext('exclude_view'):
            avail_aids = ut.compress(avail_aids, flag_list)
        # avail_aids = sorted(avail_aids)

    # FILTER HACK integrating some notion of tag functions
    # TODO: further integrate
    if aidcfg.get('has_any', None) or aidcfg.get('has_none', None):
        filterkw = ut.dict_subset(aidcfg, ['has_any', 'has_none'], None)
        flags = get_annot_tag_filterflags(ibs, avail_aids, filterkw)
        with VerbosityContext('has_any', 'has_none'):
            avail_aids = ut.compress(avail_aids, flags)
        # avail_aids = sorted(avail_aids)

    avail_aids = sorted(avail_aids)

    VerbosityContext.endfilter()
    return avail_aids


@profile
def filter_annots_intragroup(
    ibs, avail_aids, aidcfg, prefix='', verbose=VERB_TESTDATA, withpre=False
):
    r"""
    This filters annots using information about the relationships
    between the annotations in the ``avail_aids`` group. This function is not
    independent and a second consecutive call may yield new results.
    Thus, the order in which this filter is applied matters.

    CommandLine:
        wbia --tf get_annotcfg_list \
                -a default:qsame_imageset=True,been_adjusted=True,excluderef=True \
                --db lynx --veryverbtd --nocache-aid

    Ignore:
        >>> aidcfg['min_timedelta'] = 60 * 60 * 24
        >>> aidcfg['min_pername'] = 3
    """
    from wbia.other import ibsfuncs

    if aidcfg is None:
        if verbose:
            print('No annot filter returning')
        return avail_aids

    VerbosityContext = verb_context('FILTER_INTRAGROUP', aidcfg, verbose)
    VerbosityContext.startfilter(withpre=withpre)

    metadata = ut.LazyDict(
        species=lambda: expand_species(ibs, aidcfg['species'], avail_aids)
    )

    if aidcfg['same_imageset'] is not None:
        same_imageset = aidcfg['same_imageset']
        assert same_imageset is True
        imgsetid_list = ibs.get_annot_primary_imageset(avail_aids)
        nid_list = ibs.get_annot_nids(avail_aids)
        multiprop2_aids = ut.hierarchical_group_items(
            avail_aids, [nid_list, imgsetid_list]
        )
        qaid_list = []
        # TODO: sampling using different enouncters
        for imgsetid, nid2_aids in multiprop2_aids.items():
            if len(nid2_aids) == 1:
                pass
            else:
                aids_list = list(nid2_aids.values())
                idx = ut.list_argmax(list(map(len, aids_list)))
                qaids = aids_list[idx]
                qaid_list.extend(qaids)
        with VerbosityContext('same_imageset'):
            avail_aids = qaid_list
        avail_aids = sorted(avail_aids)

    # TODO:
    # Filter via GPS distance
    # try:
    #    if aidcfg['min_spacedelta'] is not None:
    #        pass
    #    if aidcfg['min_spacetimedelta'] is not None:
    #        pass
    # except KeyError:
    #    pass

    # FIXME: This is NOT an independent filter because it depends on pairwise
    # interactions
    if aidcfg['view_pername'] is not None:
        species = metadata['species']
        # This filter removes entire names.  The avaiable aids must be from
        # names with certain viewpoint frequency properties
        prop2_nid2_aids = ibs.group_annots_by_prop_and_name(
            avail_aids, ibs.get_annot_viewpoint_code
        )

        countstr = aidcfg['view_pername']
        primary_viewpoint = ibsfuncs.get_primary_species_viewpoint(species)
        lhs_dict = {
            'primary': primary_viewpoint,
            'primary1': ibsfuncs.get_extended_viewpoints(
                primary_viewpoint, num1=1, num2=0, include_base=False
            )[0],
        }
        self = ut.CountstrParser(lhs_dict, prop2_nid2_aids)
        nid2_flag = self.parse_countstr_expr(countstr)
        nid2_aids = ibs.group_annots_by_name_dict(avail_aids)
        valid_nids = [nid for nid, flag in nid2_flag.items() if flag]
        with VerbosityContext('view_pername', countstr=countstr):
            avail_aids = ut.flatten(ut.dict_take(nid2_aids, valid_nids))
        # avail_aids = sorted(avail_aids)

    if aidcfg['min_timedelta'] is not None:
        min_timedelta = ut.ensure_timedelta(aidcfg['min_timedelta'])
        with VerbosityContext('min_timedelta', min_timedelta=min_timedelta):
            avail_aids = ibs.filter_annots_using_minimum_timedelta(
                avail_aids, min_timedelta
            )
        # avail_aids = sorted(avail_aids)

    # Each aid must have at least this number of other groundtruth aids
    min_pername = aidcfg['min_pername']
    if min_pername is not None:
        grouped_aids_ = ibs.group_annots_by_name(
            avail_aids, distinguish_unknowns=True, assume_unique=True
        )[0]
        with VerbosityContext('min_pername'):
            flags = np.array(ut.lmap(len, grouped_aids_)) >= min_pername
            avail_aids = ut.flatten(ut.compress(grouped_aids_, flags))
            # avail_aids = ut.flatten([
            #    aids for aids in grouped_aids_ if len(aids) >= min_pername])
        # avail_aids = sorted(avail_aids)

    max_pername = aidcfg['max_pername']
    if max_pername is not None:
        grouped_aids_ = ibs.group_annots_by_name(
            avail_aids, distinguish_unknowns=True, assume_unique=True
        )[0]
        with VerbosityContext('max_pername'):
            avail_aids = ut.flatten(
                [aids for aids in grouped_aids_ if len(aids) <= max_pername]
            )
        # avail_aids = sorted(avail_aids)

    avail_aids = sorted(avail_aids)

    VerbosityContext.endfilter()
    return avail_aids


@profile
def get_reference_preference_order(
    ibs,
    gt_ref_grouped_aids,
    gt_avl_grouped_aids,
    prop_getter,
    cmp_func,
    aggfn,
    rng,
    verbose=VERB_TESTDATA,
):
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
        for ref_prop, avl_prop in zip(grouped_reference_props, grouped_available_gt_props)
    ]
    # Order by increasing timedelta (metric)
    gt_preference_idx_list = vt.argsort_groups(preference_scores, reverse=True, rng=rng)
    return gt_preference_idx_list


@profile
def sample_annots_wrt_ref(
    ibs, avail_aids, aidcfg, ref_aids, prefix='', verbose=VERB_TESTDATA
):
    """
    Sampling when a reference set is given
    """
    sample_per_name = aidcfg.get('sample_per_name')
    sample_per_ref_name = aidcfg.get('sample_per_ref_name')
    exclude_reference = aidcfg.get('exclude_reference')
    sample_size = aidcfg.get('sample_size')
    offset = aidcfg.get('sample_offset')
    sample_rule_ref = aidcfg.get('sample_rule_ref')
    sample_rule = aidcfg.get('sample_rule')
    sample_occur = aidcfg.get('sample_occur')
    exclude_ref_contact = aidcfg.get('exclude_ref_contact')

    avail_aids = sorted(avail_aids)
    ref_aids = sorted(ref_aids)

    VerbosityContext = verb_context('SAMPLE (REF)', aidcfg, verbose)
    VerbosityContext.startfilter()

    if sample_per_ref_name is None:
        sample_per_ref_name = sample_per_name

    if offset is None:
        offset = 0

    if exclude_reference:
        assert ref_aids is not None, 'ref_aids=%r' % (ref_aids,)
        # VerbosityContext.report_annot_stats(ibs, avail_aids, prefix, '')
        # VerbosityContext.report_annot_stats(ibs, ref_aids, prefix, '')
        with VerbosityContext('exclude_reference', num_ref_aids=len(ref_aids)):
            avail_aids = ut.setdiff_ordered(avail_aids, ref_aids)
            avail_aids = sorted(avail_aids)

    if exclude_ref_contact:
        with VerbosityContext('exclude_ref_contact', num_ref_aids=len(ref_aids)):
            # also_exclude_overlaps = ibs.get_dbname() == 'Oxford'
            contact_aids_list = ibs.get_annot_contact_aids(
                ref_aids, daid_list=avail_aids, assume_unique=True
            )
            # Disallow the same name in the same image
            x = ibs.unflat_map(ibs.get_annot_nids, contact_aids_list)
            y = ibs.get_annot_nids(ref_aids)
            sameimg_samename_aids = ut.flatten(
                [
                    ut.compress(aids, np.array(x0) == y0)
                    for aids, x0, y0 in zip(contact_aids_list, x, y)
                ]
            )
            # contact_aids = ut.flatten(contact_aids_list)
            avail_aids = ut.setdiff_ordered(avail_aids, sameimg_samename_aids)

    if sample_occur is True:
        with VerbosityContext('sample_occur', num_ref_aids=len(ref_aids)):
            # Get other aids from the references' encounters
            ref_enc_texts = ibs.get_annot_encounter_text(ref_aids)
            avail_enc_texts = ibs.get_annot_encounter_text(avail_aids)
            flags = ut.setdiff_flags(avail_enc_texts, ref_enc_texts)
            avail_aids = ut.compress(avail_aids, flags)

    if not (sample_per_ref_name is not None or sample_size is not None):
        VerbosityContext.endfilter()
        return avail_aids

    if ut.is_float(sample_size):
        # A float sample size is a interpolations between full data and small
        # data
        sample_size = int(
            round((len(avail_aids) * sample_size + (1 - sample_size) * len(ref_aids)))
        )
        if verbose:
            print('Expanding sample size to: %r' % (sample_size,))

    # This function first partitions aids into a one set that corresonds with
    # the reference set and another that does not correspond with the reference
    # set. The rest of the filters operate on these sets independently
    partitioned_sets = ibs.partition_annots_into_corresponding_groups(
        ref_aids, avail_aids
    )
    # items
    # [0], and [1] are corresponding lists of annot groups
    # [2], and [3] are non-corresonding annot groups
    (
        gt_ref_grouped_aids,
        gt_avl_grouped_aids,
        gf_ref_grouped_aids,
        gf_avl_grouped_aids,
    ) = partitioned_sets

    if sample_per_ref_name is not None:
        rng = np.random.RandomState(SEED2)
        if sample_rule_ref == 'maxtimedelta':
            # Maximize time delta between query and corresponding database
            # annotations
            cmp_func = ut.absdiff
            aggfn = np.mean
            prop_getter = ibs.get_annot_image_unixtimes_asfloat
            gt_preference_idx_list = get_reference_preference_order(
                ibs,
                gt_ref_grouped_aids,
                gt_avl_grouped_aids,
                prop_getter,
                cmp_func,
                aggfn,
                rng,
            )
        elif sample_rule_ref == 'random':
            gt_preference_idx_list = [
                ut.random_indexes(len(aids), rng=rng) for aids in gt_avl_grouped_aids
            ]
        else:
            raise ValueError('Unknown sample_rule_ref = %r' % (sample_rule_ref,))
        gt_sample_idxs_list = ut.get_list_column_slice(
            gt_preference_idx_list, offset, offset + sample_per_ref_name
        )
        gt_sample_aids = ut.list_ziptake(gt_avl_grouped_aids, gt_sample_idxs_list)
        gt_avl_grouped_aids = gt_sample_aids

        with VerbosityContext(
            'sample_per_ref_name',
            'sample_rule_ref',
            'sample_offset',
            sample_per_ref_name=sample_per_ref_name,
        ):
            avail_aids = ut.flatten(gt_avl_grouped_aids) + ut.flatten(gf_avl_grouped_aids)

    if sample_per_name is not None:
        # sample rule is always random for gf right now
        rng = np.random.RandomState(SEED2)
        if sample_rule == 'random':
            gf_preference_idx_list = [
                ut.random_indexes(len(aids), rng=rng) for aids in gf_avl_grouped_aids
            ]
        else:
            raise ValueError('Unknown sample_rule=%r' % (sample_rule,))
        gf_sample_idxs_list = ut.get_list_column_slice(
            gf_preference_idx_list, offset, offset + sample_per_name
        )
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
            print(
                (
                    'Warning: Cannot meet sample_size=%r. available_%saids '
                    'will be undersized by at least %d'
                )
                % (sample_size, prefix, -num_remove_gf,)
            )
        if num_keep_gf < 0:
            # Too many multitons; Can never remove a multiton
            print(
                'Warning: Cannot meet sample_size=%r. available_%saids '
                'will be oversized by at least %d' % (sample_size, prefix, -num_keep_gf,)
            )
        rng = np.random.RandomState(SEED2)
        gf_avl_aids = ut.random_sample(gf_avl_aids, num_keep_gf, rng=rng)

        # random ordering makes for bad hashes
        with VerbosityContext(
            'sample_size',
            sample_size=sample_size,
            num_remove_gf=num_remove_gf,
            num_keep_gf=num_keep_gf,
        ):
            avail_aids = gt_avl_aids + gf_avl_aids

    avail_aids = sorted(gt_avl_aids + gf_avl_aids)

    VerbosityContext.endfilter()
    return avail_aids


@profile
def multi_sampled_seaturtle_queries():
    import wbia
    from wbia.expt import annotation_configs
    from wbia.expt import experiment_helpers
    from wbia.init.filter_annots import expand_acfgs
    import copy

    aidcfg = copy.deepcopy(annotation_configs.default)
    db = 'seaturtles'  # 'testdb1'
    ibs = wbia.opendb(defaultdb=db)
    a = [
        'default:sample_occur=True,occur_offset=0,exclude_reference=True,qhas_any=(left,right),num_names=1'
    ]
    acfg_combo_list = experiment_helpers.parse_acfg_combo_list(a)
    aidcfg = acfg_combo_list[0][0]

    if False:
        # Do each name individually. A bit slower, but more correct
        qaids_list = []
        daids_list = []
        aidcfg['qcfg']['name_offset'] = 0
        aidcfg['qcfg']['occur_offset'] = 0
        prev = -1
        while True:
            aidcfg['qcfg']['occur_offset'] = 0
            while True:
                qaids, daids = expand_acfgs(
                    ibs, aidcfg, use_cache=False, save_cache=False
                )
                aidcfg['qcfg']['occur_offset'] += 1
                if len(qaids) == 0:
                    break
                qaids_list.append(qaids)
                daids_list.append(daids)
                print(qaids)
            if len(qaids_list) == prev:
                break
            prev = len(qaids_list)
            aidcfg['qcfg']['name_offset'] += 1

        for qaids, daids in zip(qaids_list, daids_list):
            ibs.print_annotconfig_stats(qaids, daids, enc_per_name=True, per_enc=True)
    else:
        # A bit faster because we can do multiple names at the same time
        qaids_list = []
        daids_list = []
        aidcfg['qcfg']['num_names'] = None
        aidcfg['dcfg']['num_names'] = None
        aidcfg['qcfg']['name_offset'] = 0
        aidcfg['qcfg']['occur_offset'] = 0
        while True:
            qaids, daids = expand_acfgs(ibs, aidcfg, use_cache=False, save_cache=False)
            aidcfg['qcfg']['occur_offset'] += 1
            if len(qaids) == 0:
                break
            qaids_list.append(qaids)
            daids_list.append(daids)
            print(qaids)

        for qaids, daids in zip(qaids_list, daids_list):
            ibs.print_annotconfig_stats(qaids, daids, enc_per_name=True, per_enc=True)


@profile
def sample_annots(ibs, avail_aids, aidcfg, prefix='', verbose=VERB_TESTDATA):
    r"""
    Sampling preserves input sample structure and thust does not always return
    exact values

    CommandLine:
        python -m wbia --tf sample_annots --veryverbtd

        python -m wbia --tf get_annotcfg_list --db seaturtles \
            -a default:qhas_any=\(left,right\),sample_occur=True,exclude_reference=True,sample_offset=0,num_names=1 --acfginfo

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.init.filter_annots import *  # NOQA
        >>> import wbia
        >>> from wbia.expt import annotation_configs
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> avail_aids = input_aids = ibs.get_valid_aids()
        >>> aidcfg = copy.deepcopy(annotation_configs.default['dcfg'])
        >>> aidcfg['sample_per_name'] = 3
        >>> aidcfg['sample_size'] = 10
        >>> aidcfg['min_pername'] = 2
        >>> prefix = ''
        >>> verbose = True
        >>> avail_aids = filter_annots_independent(ibs, avail_aids, aidcfg,
        >>>                                        prefix, verbose)
        >>> avail_aids = sample_annots(ibs, avail_aids, aidcfg,
        >>>                            prefix, avail_aids)
        >>> result = ('avail_aids = %s' % (str(avail_aids),))
        >>> print(result)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.init.filter_annots import *  # NOQA
        >>> import wbia
        >>> from wbia.expt import annotation_configs
        >>> db = 'seaturtles'  # 'testdb1'
        >>> ibs = wbia.opendb(defaultdb=db)
        >>> aidcfg = copy.deepcopy(annotation_configs.default)['qcfg']
        >>> aidcfg['sample_occur'] = True
        >>> initial_aids = ibs.get_valid_aids()
        >>> withpre, verbose, prefix = True, 2, ''
        >>> avail_aids = filter_annots_independent(
        >>>     ibs, initial_aids, {'has_any': ['left', 'right']}, prefix, verbose)
        >>> qaids = sample_annots(ibs, avail_aids, aidcfg, prefix, verbose)
        >>> avail_aids = initial_aids
        >>> ref_aids = qaids
        >>> dcfg = dict(exclude_reference=True, sample_occur=True)
        >>> daids = sample_annots_wrt_ref(ibs, initial_aids, dcfg, qaids, prefix, verbose)
        >>> ibs.print_annotconfig_stats(qaids, daids, enc_per_name=True, per_enc=True)
    """
    import vtool as vt
    from wbia.expt import annotation_configs

    def get_cfg(key):
        default_dict = annotation_configs.SAMPLE_DEFAULTS
        return aidcfg.get(key, default_dict[key])

    VerbosityContext = verb_context('SAMPLE (NOREF)', aidcfg, verbose)
    VerbosityContext.startfilter()

    sample_rule = get_cfg('sample_rule')
    sample_per_name = get_cfg('sample_per_name')
    sample_size = get_cfg('sample_size')
    offset = get_cfg('sample_offset')
    occur_offset = get_cfg('occur_offset')
    name_offset = get_cfg('name_offset')
    num_names = get_cfg('num_names')
    sample_occur = get_cfg('sample_occur')

    unflat_get_annot_unixtimes = functools.partial(
        ibs.unflat_map, ibs.get_annot_image_unixtimes_asfloat
    )

    if offset is None:
        offset = 0
    if occur_offset is None:
        occur_offset = 0
    if name_offset is None:
        name_offset = 0

    if num_names is not None:
        grouped_aids = ibs.group_annots_by_name(avail_aids, assume_unique=True)[0]
        with VerbosityContext('num_names'):
            name_slice = slice(name_offset, name_offset + num_names)
            avail_aids = ut.flatten(grouped_aids[name_slice])

    if sample_occur is True:
        # Occurrence / Encounter sampling
        occur_texts = ibs.get_annot_occurrence_text(avail_aids)
        names = ibs.get_annot_names(avail_aids)
        grouped_ = ut.hierarchical_group_items(avail_aids, [names, occur_texts])
        # ensure dictionary ordering for offset consistency
        sgrouped_ = ut.sort_dict(ut.hmap_vals(ut.sort_dict, grouped_, max_depth=0))
        occur_slice = slice(occur_offset, occur_offset + 1)
        chosen = [
            ut.flatten(list(sub.values())[occur_slice]) for sub in sgrouped_.values()
        ]

        with VerbosityContext('sample_offset'):
            # TODO: num ocurrences to sample
            # TODO: num annots per encounter to sample
            avail_aids = ut.flatten(chosen)
        # now find which groups of annotations share those tags

    if sample_per_name is not None:
        # For the query we just choose a single annot per name
        # For the database we have to do something different
        grouped_aids = ibs.group_annots_by_name(avail_aids, assume_unique=True)[0]
        # Order based on some preference (like random)
        sample_seed = get_cfg('sample_seed')
        rng = np.random.RandomState(sample_seed)
        # + --- Get nested sample indicies ---
        if sample_rule == 'random':
            preference_idxs_list = [
                ut.random_indexes(len(aids), rng=rng) for aids in grouped_aids
            ]
        elif sample_rule == 'mintime':
            unixtime_list = unflat_get_annot_unixtimes(grouped_aids)
            preference_idxs_list = vt.argsort_groups(
                unixtime_list, reverse=False, rng=rng
            )
        elif sample_rule == 'maxtime':
            unixtime_list = unflat_get_annot_unixtimes(grouped_aids)
            preference_idxs_list = vt.argsort_groups(unixtime_list, reverse=True, rng=rng)
        elif sample_rule == 'qual_and_view':
            if sample_rule != 'qual_and_view':
                # Hacked in
                with VerbosityContext('sample_per_name', 'sample_rule', 'sample_offset'):
                    flags = ibs.get_annot_quality_viewpoint_subset(
                        avail_aids, annots_per_view=sample_per_name
                    )
                    avail_aids = ut.compress(avail_aids, flags)
        else:
            raise ValueError('Unknown sample_rule=%r' % (sample_rule,))
        # L ___
        if sample_rule != 'qual_and_view':
            sample_idxs_list = list(
                ut.iget_list_column_slice(
                    preference_idxs_list, offset, offset + sample_per_name
                )
            )
            sample_aids = ut.list_ziptake(grouped_aids, sample_idxs_list)

            with VerbosityContext('sample_per_name', 'sample_rule', 'sample_offset'):
                avail_aids = ut.flatten(sample_aids)
        avail_aids = sorted(avail_aids)

    if sample_size is not None:
        # BUG: Should sample annots while preserving name size
        if sample_size > len(avail_aids):
            print('Warning sample size too large')
        rng = np.random.RandomState(SEED2)
        # Randomly sample names rather than annotations this makes sampling a
        # knapsack problem. Use a random greedy solution
        grouped_aids = ibs.group_annots_by_name(avail_aids, assume_unique=True)[0]
        # knapsack items values and weights are are num annots per name
        knapsack_items = [
            (len(aids), len(aids), count) for count, aids in enumerate(grouped_aids)
        ]
        ut.deterministic_shuffle(knapsack_items, rng=rng)
        total_value, items_subset = ut.knapsack_greedy(knapsack_items, sample_size)
        group_idx_sample = ut.get_list_column(items_subset, 2)
        subgroup_aids = ut.take(grouped_aids, group_idx_sample)
        with VerbosityContext('sample_size'):
            avail_aids = ut.flatten(subgroup_aids)
            # avail_aids = ut.random_sample(avail_aids, sample_size, rng=rng)
        if total_value != sample_size:
            print('Sampling could not get exactly right sample size')
        avail_aids = sorted(avail_aids)

    VerbosityContext.endfilter()
    return avail_aids


@profile
def subindex_annots(
    ibs, avail_aids, aidcfg, ref_aids=None, prefix='', verbose=VERB_TESTDATA
):
    """
    Returns exact subindex of annotations
    """
    VerbosityContext = verb_context('SUBINDEX', aidcfg, verbose)
    VerbosityContext.startfilter(withpre=False)

    if aidcfg['shuffle']:
        rand_idx = ut.random_indexes(len(avail_aids), seed=SEED2)
        with VerbosityContext('shuffle', SEED2=SEED2):
            avail_aids = ut.take(avail_aids, rand_idx)

    if aidcfg['index'] is not None:
        indicies = ensure_flatlistlike(aidcfg['index'])
        _indexed_aids = [avail_aids[ix] for ix in indicies if ix < len(avail_aids)]
        with VerbosityContext('index', subset_size=len(_indexed_aids)):
            avail_aids = _indexed_aids

    # Always sort aids to preserve hashes? (Maybe sort the vuuids instead)
    avail_aids = sorted(avail_aids)

    VerbosityContext.endfilter(withpost=False)
    return avail_aids


@profile
def ensure_flatiterable(input_):
    if isinstance(input_, six.string_types):
        input_ = ut.fuzzy_int(input_)
    if isinstance(input_, int) or not ut.isiterable(input_):
        return [input_]
    elif isinstance(input_, (list, tuple)):
        # print(input_)
        if len(input_) > 0 and ut.isiterable(input_[0]):
            return ut.flatten(input_)
        return input_
    else:
        raise TypeError('cannot ensure %r input_=%r is iterable', (type(input_), input_))


def ensure_flatlistlike(input_):
    # if isinstance(input_, slice):
    #    pass
    iter_ = ensure_flatiterable(input_)
    return list(iter_)


def verb_context(filtertype, aidcfg, verbose):
    """ closure helper """

    class VerbosityContext(object):
        """
        Printing filter info in a way that avoids polluting the function
        namespace. This is a hack.

        This is a with_statement context class that expect a variable avail_aids
        to be modified inside the context. It prints the state of the variable
        before and after filtering. Several static methods can be used
        at the start and end of larger filtering functions.
        """

        def __init__(self, *keys, **filterextra):
            self.prefix = ut.get_var_from_stack('prefix', verbose=False)
            if verbose:
                dictkw = dict(nl=False, explicit=True, nobraces=True)
                infostr = ''
                if len(keys) > 0:
                    subdict = ut.dict_subset(aidcfg, keys, None)
                    infostr += '' + ut.repr2(subdict, **dictkw)
                print('[%s] * Filter by %s' % (self.prefix.upper(), infostr.strip()))
                if verbose > 1 and len(filterextra) > 0:
                    infostr2 = ut.repr2(filterextra, nl=False, explicit=False)
                    print('[%s]      %s' % (self.prefix.upper(), infostr2))

        def __enter__(self):
            aids = ut.get_var_from_stack('avail_aids', verbose=False)
            self.num_before = len(aids)

        def __exit__(self, exc_type, exc_value, exc_traceback):
            if verbose:
                aids = ut.get_var_from_stack('avail_aids', verbose=False)
                num_after = len(aids)
                num_removed = self.num_before - num_after
                if num_removed > 0 or verbose > 1:
                    print(
                        '[%s]   ... removed %d annots. %d remain'
                        % (self.prefix.upper(), num_removed, num_after)
                    )

        @staticmethod
        def report_annot_stats(ibs, aids, prefix, name_suffix, statskw={}):
            if verbose > 1:
                with ut.Indenter('[%s]  ' % (prefix.upper(),)):
                    # TODO: helpx on statskw
                    # statskw = dict(per_name_vpedge=None, per_name=None)
                    dict_name = prefix + 'aid_stats' + name_suffix
                    # hashid, per_name, per_qual, per_vp, per_name_vpedge,
                    # per_image, min_name_hourdist
                    ibs.print_annot_stats(aids, prefix=prefix, label=dict_name, **statskw)

        # def report_annotconfig_stats(ref_aids, aids):
        #    with ut.Indenter('  '):
        #        ibs.print_annotconfig_stats(ref_aids, avail_aids)

        @staticmethod
        def startfilter(withpre=True):
            """
            Args:
                withpre (bool): if True reports stats before filtering
            """
            if verbose:
                prefix = ut.get_var_from_stack('prefix', verbose=False)
                print('[%s] * [%s] %sAIDS' % (prefix.upper(), filtertype, prefix))
                if verbose > 1 and withpre:
                    ibs = ut.get_var_from_stack('ibs', verbose=False)
                    aids = ut.get_var_from_stack('avail_aids', verbose=False)
                    VerbosityContext.report_annot_stats(ibs, aids, prefix, '_pre')

        @staticmethod
        def endfilter(withpost=True):
            if verbose:
                ibs = ut.get_var_from_stack('ibs', verbose=False)
                aids = ut.get_var_from_stack('avail_aids', verbose=False)
                prefix = ut.get_var_from_stack('prefix', verbose=False)
                hashid = ibs.get_annot_hashid_semantic_uuid(aids, prefix=prefix.upper())
                if withpost:
                    if verbose > 1:
                        VerbosityContext.report_annot_stats(ibs, aids, prefix, '_post')
                print('[%s] * HAHID: %s' % (prefix.upper(), hashid))
                print(
                    '[%s] * [%s]: len(avail_%saids) = %r\n'
                    % (prefix.upper(), filtertype, prefix, len(aids))
                )

    return VerbosityContext


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.init.filter_annots
        python -m wbia.init.filter_annots --allexamples
        python -m wbia.init.filter_annots --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
