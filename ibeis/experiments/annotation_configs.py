# -*- coding: utf-8 -*-
"""
Definitions for common aid configurations

Rename to annot_cfgdef
"""
from __future__ import absolute_import, division, print_function
import utool as ut
from ibeis.experiments import cfghelpers
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[aidcfg]')


# easier to type names to alias some of these options
ALIAS_KEYS = {
    #'aids'     : 'default_aids',
    'pername'  : 'sample_per_name',
    'offset'   : 'sample_offset',
    'refrule'  : 'sample_rule_ref',
    'rule'     : 'sample_rule',
    'size'     : 'sample_size',
    'mingt'    : 'min_pername',
    'excluderef': 'exclude_reference',
}

OTHER_DEFAULTS = {
    'force_const_size'    : None,  # forces a consistnet sample size across combinations
    'hack_extra' : None,  # hack param to make bigger db sizes
}

# Defaults for the independent filter
INDEPENDENT_DEFAULTS = {
    'species'             : 'primary',  # specify the species
    # Timedelta Params
    'require_timestamp'   : None,
    'min_timedelta'       : None,
    # Quality Params
    'require_quality'     : None,  # if True unknown qualities are removed
    'minqual'             : 'poor',
    # Viewpoint params
    'require_viewpoint'   : None,
    'view'                : None,
    'view_ext'            : 0,      # num viewpoints to extend in dir1 and dir2
    'view_ext1'           : None,   # num viewpoints to extend in dir1
    'view_ext2'           : None,   # num viewpoints to extend in dir2
    'view_pername'        : None,   # formatted string filtering the viewpoints
    'is_known'            : None,
    'min_pername'         : None,  # minimum number of aids for each name in sample
}

SUBINDEX_DEFAULTS = {
    # Final indexing
    'shuffle'             : False,  # randomize order before indexing
    'index'               : None,   # choose only a subset
}

SAMPLE_DEFAULTS = {
    'sample_size'         : None,  # Gets as close to sample size without removing other props
    # Per Name / Exemplar Params
    'sample_per_name'     : None,  # Choos num_annots to sample from each name.
    'sample_rule'         : 'random',
    'sample_offset'       : None,  # UNUSED
}

SAMPLE_REF_DEFAULTS = {
    'exclude_reference'   : None,  # excludes any aids specified in a reference set (ie qaids)
    'sample_rule_ref'     : 'random',
    'sample_per_ref_name' : None,  # when sampling daids, choose this many correct matches per query
}


# Base common settings, but some default settings will be different
# for query and database annotations
DEFAULT_AIDCFG = ut.merge_dicts(OTHER_DEFAULTS, INDEPENDENT_DEFAULTS, SAMPLE_DEFAULTS, SAMPLE_REF_DEFAULTS, SUBINDEX_DEFAULTS)
__default_aidcfg = DEFAULT_AIDCFG
#'default_aids'        : 'all',  # initial set to choose from
# Databse size
#'exclude_aids'       : None,   # removes specified aids from selection
#'include_aids'       : None,   # force inclusion?
#'gt_avl_aids'         : None,   # The only aids available as reference groundtruth
#'ref_has_viewpoint'   : None,  # All aids must have a gt with this viewpoint
#'ref_has_qual'        : None,  # All aids must have a gt with this viewpoint
#'name_choose_rule'   : 'timestamp',  # Choose #annots for each name


# Maps from a top level setting to its depenants
#__acfg_dependants_map = {
#    'view': ('view is None', ['view_ext', 'view_ext1', 'view_ext1']),
#}


#def remove_disabled_acfg_dependants(acfg):
#    if acfg['view'] is None:
#        pass


def compress_aidcfg(acfg, filter_nones=False, filter_empty=False, force_noncommon=[]):
    r"""
    Args:
        acfg (dict):

    Returns:
        dict: acfg

    CommandLine:
        python -m ibeis.experiments.annotation_configs --exec-compress_aidcfg

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.annotation_configs import *  # NOQA
        >>> acfg = default
        >>> acfg = compress_aidcfg(acfg)
        >>> result = ('acfg = %s' % (ut.dict_str(acfg),))
        >>> print(default)
        >>> print(result)
    """
    import copy
    if 'qcfg' not in acfg or 'dcfg' not in acfg:
        return acfg
    acfg = copy.deepcopy(acfg)
    common_cfg = ut.dict_intersection(acfg['qcfg'], acfg['dcfg'])
    ut.delete_keys(common_cfg, force_noncommon)
    ut.delete_keys(acfg['qcfg'], common_cfg.keys())
    ut.delete_keys(acfg['dcfg'], common_cfg.keys())
    acfg['common'] = common_cfg
    if filter_nones:
        acfg['common'] = ut.dict_filter_nones(acfg['common'])
        acfg['qcfg'] = ut.dict_filter_nones(acfg['qcfg'])
        acfg['dcfg'] = ut.dict_filter_nones(acfg['dcfg'])
    if filter_empty:
        if len(acfg['common']) == 0:
            del acfg['common']
        if len(acfg['qcfg']) == 0:
            del acfg['qcfg']
        if len(acfg['dcfg']) == 0:
            del acfg['dcfg']
    return acfg


def partition_acfg_list(acfg_list):
    for acfg in acfg_list:
        assert acfg['qcfg']['_cfgname'] == acfg['dcfg']['_cfgname'], (
            'should be the same for now')

    # Hack to make common params between q and d appear the same
    _acfg_list = [compress_aidcfg(acfg) for acfg in acfg_list]

    flat_acfg_list = flatten_acfg_list(_acfg_list)
    flat_nonvaried_dict, flat_varied_acfg_list = cfghelpers.partition_varied_cfg_list(flat_acfg_list)
    nonvaried_dict = unflatten_acfgdict(flat_nonvaried_dict)
    varied_acfg_list = [unflatten_acfgdict(acfg) for acfg in flat_varied_acfg_list]
    return nonvaried_dict, varied_acfg_list


def get_varied_acfg_labels(acfg_list, mainkey='_cfgname'):
    """
        >>> from ibeis.experiments.annotation_configs import *  # NOQA

    """
    #print(ut.list_str(varied_acfg_list, nl=2))
    for acfg in acfg_list:
        assert acfg['qcfg']['_cfgname'] == acfg['dcfg']['_cfgname'], (
            'should be the same for now')
    cfgname_list = [acfg['qcfg']['_cfgname'] for acfg in acfg_list]

    # Hack to make common params between q and d appear the same
    _acfg_list = [compress_aidcfg(acfg) for acfg in acfg_list]

    flat_acfg_list = flatten_acfg_list(_acfg_list)
    nonvaried_dict, varied_acfg_list = cfghelpers.partition_varied_cfg_list(flat_acfg_list)

    SUPER_HACK = True
    if SUPER_HACK:
        # SUPER HACK, recompress remake the varied list after knownig what is varied
        _varied_keys = list(set(ut.flatten(
            [list(ut.flatten(
                [list(x.keys())
                 for x in unflatten_acfgdict(cfg).values()]
            )) for cfg in varied_acfg_list]
        )))
        _acfg_list = [
            compress_aidcfg(acfg, force_noncommon=_varied_keys)
            for acfg in acfg_list]
        flat_acfg_list = flatten_acfg_list(_acfg_list)
        nonvaried_dict, varied_acfg_list = cfghelpers.partition_varied_cfg_list(
            flat_acfg_list)

    shortened_cfg_list = [
        #{shorten_to_alias_labels(key): val for key, val in _dict.items()}
        ut.map_dict_keys(shorten_to_alias_labels, _dict)
        for _dict in varied_acfg_list]
    nonlbl_keys = cfghelpers.INTERNAL_CFGKEYS
    nonlbl_keys = [prefix +  key for key in nonlbl_keys for prefix in ['', 'q', 'd']]
    # hack for sorting by q/d stuff first

    def get_key_order(cfg):
        keys = [k for k in cfg.keys() if k not in nonlbl_keys]
        sortorder = [2 * k.startswith('q') + 1 * k.startswith('d') for k in keys]
        return ut.sortedby(keys, sortorder)[::-1]

    shortened_lbl_list = [
        cfghelpers.get_cfg_lbl(cfg, name, nonlbl_keys, key_order=get_key_order(cfg))
        for cfg, name in zip(shortened_cfg_list, cfgname_list)]
    return shortened_lbl_list


def shorten_to_alias_labels(key):
    search_list = list(ALIAS_KEYS.values()) + ['qcfg_', 'dcfg_', 'common_']
    repl_list = list(ALIAS_KEYS.keys()) + ['q', 'd', '']
    return ut.multi_replace(key, search_list, repl_list)


def flatten_acfg_list(acfg_list):
    flat_acfg_list = []
    for acfg in acfg_list:
        flat_dict = {
            prefix + '_' + key: val
            for prefix, subdict in acfg.items()
            for key, val in subdict.items()
        }
        flat_acfg_list.append(flat_dict)
    return flat_acfg_list


def compress_acfg_list_for_printing(acfg_list):
    r"""
    CommandLine:
        python -m ibeis.experiments.annotation_configs --exec-compress_acfg_list_for_printing

    Example:
        >>> from ibeis.experiments.annotation_configs import *  # NOQA
        >>> qcfg_list = [{'f': 1, 'b': 1}, {'f': 2, 'b': 1}, {'f': 3, 'b': 1, 'z': 4}]
        >>> acfg_list = [{'qcfg': qcfg} for qcfg in qcfg_list]
        >>> nonvaried_dict, varied_dicts = compress_acfg_list_for_printing(acfg_list)
        >>> result = ('varied_dicts = %s\n' % (ut.list_str(varied_dicts),))
        >>> result += ('nonvaried_dict = %s' % (ut.dict_str(nonvaried_dict),))
        >>> print(result)
    """
    flat_acfg_list = flatten_acfg_list(acfg_list)
    nonvaried_dict, varied_acfg_list = cfghelpers.partition_varied_cfg_list(flat_acfg_list)
    nonvaried_compressed_dict = compress_aidcfg(unflatten_acfgdict(nonvaried_dict), filter_nones=True)
    varied_compressed_dict_list = [
        compress_aidcfg(unflatten_acfgdict(cfg), filter_empty=True)
        for cfg in varied_acfg_list]
    return nonvaried_compressed_dict, varied_compressed_dict_list


def print_acfg_list(acfg_list, expanded_aids_list=None, ibs=None,
                    combined=False, **kwargs):

    _tup = compress_acfg_list_for_printing(acfg_list)
    nonvaried_compressed_dict, varied_compressed_dict_list = _tup

    ut.colorprint('+=== <Info acfg_list> ===', 'white')
    #print('Printing acfg_list info. len(acfg_list) = %r' % (len(acfg_list),))
    print('non-varied aidcfg = ' + ut.dict_str(nonvaried_compressed_dict))
    seen_ = ut.ddict(list)

    # get default kwkeys for annot info
    if ibs is not None:
        annotstats_kw = kwargs.copy()
        kwkeys = ut.parse_func_kwarg_keys(ibs.get_annot_stats_dict)
        annotstats_kw.update(ut.argparse_dict(
            dict(zip(kwkeys, [None] * len(kwkeys))), only_specified=True))

    for acfgx in range(len(acfg_list)):
        acfg = acfg_list[acfgx]
        title = ('q_cfgname=' + acfg['qcfg']['_cfgname'] +
                 ' d_cfgname=' + acfg['dcfg']['_cfgname'])

        ut.colorprint('+--- acfg %d / %d -- %s ---- ' %
                      (acfgx + 1, len(acfg_list), title), 'lightgray')
        print('acfg = ' + ut.dict_str(varied_compressed_dict_list[acfgx], strvals=True))

        if expanded_aids_list is not None:
            qaids, daids = expanded_aids_list[acfgx]
            key = (ut.hashstr_arr27(qaids, 'qaids'), ut.hashstr_arr27(daids, 'daids'))
            if key not in seen_:
                if ibs is not None:
                    seen_[key].append(acfgx)
                    annotconfig_stats_strs, _ = ibs.get_annotconfig_stats(
                        qaids, daids, verbose=True, combined=combined, **annotstats_kw)
            else:
                dupindex = seen_[key]
                print('DUPLICATE of index %r' % (dupindex,))
                dupdict = varied_compressed_dict_list[dupindex[0]]
                print('DUP OF acfg = ' + ut.dict_str(dupdict, strvals=True))
    ut.colorprint('L___ </Info acfg_list> ___', 'white')


def print_acfg(acfg, expanded_aids=None, ibs=None, **kwargs):
    print('acfg = ' + ut.dict_str(compress_aidcfg(acfg)))
    if expanded_aids is not None:
        ibs.print_annot_stats(expanded_aids, label='expanded_aids = ', **kwargs)


def unflatten_acfgdict(flat_dict, prefix_list=['dcfg', 'qcfg']):
    acfg = {prefix: {} for prefix in prefix_list}
    for prefix in prefix_list:
        for key, val in flat_dict.items():
            if key.startswith(prefix + '_'):
                acfg[prefix][key[len(prefix) + 1:]] = val
    return acfg


def apply_timecontrol(acfg, min_timedelta='6h', require_timestamp=True):
    return {
        'qcfg': ut.augdict(
            acfg['qcfg'], {
                'require_timestamp': require_timestamp,
                'min_timedelta': min_timedelta,
            }),

        'dcfg': ut.augdict(
            acfg['dcfg'], {
                'require_timestamp': require_timestamp,
                'min_timedelta': min_timedelta,
            }),
    }


def apply_qualcontrol(acfg):
    return {
        'qcfg': ut.augdict(
            acfg['qcfg'], {
                'require_quality': True,
            }),

        'dcfg': ut.augdict(
            acfg['dcfg'], {
                'require_quality': True,
            }),
    }


__baseline_aidcfg = ut.augdict(__default_aidcfg, {
    'is_known': True,
    'minqual': 'ok',
    'view': 'primary',
    'view_ext': 1,
})


__controlled_aidcfg = ut.augdict(__baseline_aidcfg, {
    #'require_timestamp': True,
    'view_ext': 0,
    'minqual': 'ok',
    'is_known': True,
})

single_default = __default_aidcfg


exclude_vars = list(vars().keys())   # this line is before tests
exclude_vars.append('exclude_vars')


default = {
    'qcfg': ut.augdict(
        __default_aidcfg, {
            #'default_aids': (1,)
        }),

    'dcfg': ut.augdict(
        __default_aidcfg, {
        }),
}


"""
ibeis -e print_acfg --db PZ_Master1 -a unctrl
"""
unctrl = uncontrolled = {
    'qcfg': ut.augdict(
        __baseline_aidcfg, {
            #'default_aids': 'allgt',
            'min_pername': 2,
        }),

    'dcfg': ut.augdict(
        __baseline_aidcfg, {
        }),
}


# Uncontrolled but comparable to controlled
unctrl_comp =  {
    'qcfg': ut.augdict(
        __baseline_aidcfg, {
            #'default_aids': 'allgt',
            'sample_per_name': 1,
            'min_pername': 2,
            'view_ext': 0,
        }),

    'dcfg': ut.augdict(
        __baseline_aidcfg, {
        }),
}

"""
ibeis -e print_acfg --db PZ_Master1 -a ctrl
ibeis -e print_acfg --db PZ_Master1 -a unctrl ctrl::unctrl:qpername=1,qview_ext=0
ibeis -e print_acfg --db PZ_Master1 -a unctrl ctrl::unctrl_comp
"""
ctrl = controlled = {
    'qcfg': ut.augdict(
        __controlled_aidcfg, {
            #'default_aids': 'allgt',
            'sample_per_name': 1,
            'min_pername': 2,
            #'sample_size': 128,  # keep this small for now until we can run full results
        }),

    'dcfg': ut.augdict(
        __controlled_aidcfg, {
            #'default_aids': 'all',
            'sample_per_name': 1,
            'exclude_reference': True,
            #'sample_size': 300,  # keep this small for now until we can run full results
            'min_pername': 1,  # allows for singletons to be in the database
        }),
}


"""
ibeis -e print_acfg --db PZ_Master1 -a timectrl
"""
timectrl = timecontrolled = apply_timecontrol(ctrl)
timectrl1h = timecontrolled = apply_timecontrol(ctrl, '1h')

timectrlL = timecontrolled = apply_timecontrol(ctrl, require_timestamp=False)

"""
ibeis -e print_acfg --db PZ_Master1 -a timequalctrl
"""
timequalctrl = timequalcontrolled = apply_qualcontrol(timectrl)


# Just vary the samples per name without messing with the number of annots in the database
varypername = {
    'qcfg': ut.augdict(
        ctrl['qcfg'], {
            'min_pername': 4,  # ensures each query will have a correct example for the groundtruth
            'force_const_size': True,
        }),

    'dcfg': ut.augdict(
        ctrl['qcfg'], {
            'sample_per_name': [1, 2, 3],
            #'sample_per_name': [1, 3],
            #'sample_per_ref_name': [1, 2, 3],
            #'sample_per_ref_name': [1, 3],
            'force_const_size': True,
        }),
}


varypername2 = {
    'qcfg': ut.augdict(
        ctrl['qcfg'], {
            'min_pername': 3,  # ensures each query will have a correct example for the groundtruth
            'force_const_size': True,
        }),

    'dcfg': ut.augdict(
        ctrl['dcfg'], {
            #'sample_per_name': [1, 2, 3],
            'sample_per_name': [1, 2],
            #'sample_per_ref_name': [1, 2, 3],
            #'sample_per_ref_name': [1, 2],
            'force_const_size': True,
        }),
}
varypername2_td = apply_timecontrol(varypername2)


"""
ibeis -e print_acfg --db PZ_Master1 -a ctrl2
ibeis -e print_acfg --db PZ_Master1 -a timectrl2
ibeis -e rank_cdf --db PZ_Master1 -a timectrl2 -t invarbest
"""
ctrl2 = {
    'qcfg': ut.augdict(
        ctrl['qcfg'], {
            'min_pername': 3,  # ensures each query will have a correct example for the groundtruth
            #'force_const_size': True,
        }),

    'dcfg': ut.augdict(
        ctrl['dcfg'], {
            #'sample_per_name': [1, 2, 3],
            'sample_per_name': 2,
            #[1, 2],
            #'sample_per_ref_name': [1, 2, 3],
            #'sample_per_ref_name': [1, 2],
            'force_const_size': True,
        }),
}

timectrl2 = apply_timecontrol(ctrl2)


varypername_td = apply_timecontrol(varypername)
varypername_td1h = apply_timecontrol(varypername, '1h')
"""
ibeis -e print_acfg --db PZ_Master1 -a varypername_tdqual
"""
varypername_tdqual = apply_qualcontrol(varypername_td)


#varypername_pzm = {
#    'qcfg': ut.augdict(
#        varypername['qcfg'], {
#            'sample_size': 500,
#        }),

#    'dcfg': ut.augdict(
#        varypername['dcfg'], {
#        }),
#}

#varypername_gz = {
#    'qcfg': ut.augdict(
#        varypername['qcfg'], {
#            'sample_size': 200,
#        }),

#    'dcfg': ut.augdict(
#        varypername['dcfg'], {
#        }),
#}

#varypername_girm = {
#    'qcfg': ut.augdict(
#        varypername['qcfg'], {
#            'sample_size': 50,
#        }),

#    'dcfg': ut.augdict(
#        varypername['dcfg'], {
#        }),
#}


"""
python -m ibeis.ibsfuncs --exec-get_num_annots_per_name --db PZ_Master1
python -m ibeis.dev -e get_annotcfg_list --db PZ_Master1 -a varysize_master1
python -m ibeis.experiments.experiment_helpers --exec-parse_acfg_combo_list  -a varysize_master1
python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_Master1 -a varysize_master1
python -m ibeis.experiments.experiment_drawing --exec-draw_rank_surface --no3dsurf -t candidacy_k -a varysize_master1 --db PZ_Master1
python -m ibeis.experiments.experiment_drawing --exec-draw_rank_surface --no3dsurf -t candidacy_k -a varysize_master1 --db PZ_Master1
python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_Master1 -a varysize_master1 --combo-slice=1:12:6
python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_Master1 -a varysize_master1.dsize=1000,dper_name=[1,2]
python -m ibeis.experiments.experiment_drawing --exec-draw_rank_surface --db PZ_Master1 -a varysize_master1.dsize=1000,dper_name=[1,2] --show -t default
python -m ibeis.experiments.experiment_printres --exec-print_results --db PZ_Master1 -a varysize_pzm -t candidacy_k
./dev.py -e print_test_results --db PZ_Master1 -a varysize_pzm:dper_name=[1,2],dsize=1500 -t candidacy_k --acfginfo
./dev.py -e draw_rank_surface  --db PZ_Master1 -a varysize_pzm:dper_name=[1,2],dsize=1500 -t candidacy_k --show
./dev.py -e draw_rank_cdf      --db PZ_Master1 -a varysize_pzm:dper_name=[1,2],dsize=1500 -t candidacy_k --show
./dev.py -e draw_rank_cdf      --db PZ_Master1 -a varysize_pzm:dper_name=[1,2],dsize=1500 -t candidacy_k:K=1 --show
./dev.py -e print_test_results --db PZ_Master1 -a varysize_pzm:dper_name=[1,2],dsize=1500 -t candidacy_k:K=1 --echo-hardcase
./dev.py -e print_test_results --db PZ_Master1 -a varysize_pzm:dper_name=[1,2],dsize=1500 -t candidacy_k:K=1
./dev.py -e print_test_results --db PZ_Master1 -a varysize_pzm -t candidacy_k --acfginfo
./dev.py -e print_test_results --db PZ_Master1 -a varysize_pzm -t candidacy_k --acfginfo

./dev.py -e print_test_results --db PZ_Master1 -a varysize_pzm:dper_name=[1,2],dsize=1500 -t candidacy_k:K=1 --echo-hardcase
./dev.py -e print_test_results --db PZ_Master1 -a varysize_pzm:dper_name=1,dsize=1500 -t candidacy_k:K=1 --echo-hardcase
./dev.py -e print_test_results --db PZ_Master1 -a varysize_pzm:dper_name=2,dsize=1500 -t candidacy_k:K=1 --echo-hardcase
"""
#varysize = {
#    'qcfg': ut.augdict(
#        __controlled_aidcfg, {
#            #'default_aids': 'allgt',
#            'sample_size': 50,
#            'sample_per_name': 1,
#            'min_pername': 4,  # ensures each query will have a correct example for the groundtruth
#        }),

#    'dcfg': ut.augdict(
#        __controlled_aidcfg, {
#            #'default_aids': 'all',
#            'sample_per_name': [1, 2, 3],
#            'exclude_reference': True,
#            'sample_size': [50, 200, 500],
#            'min_pername': 1,
#        }),
#}


"""
ibeis -e print_acfg -a varysize2 --db PZ_Master1 --verbtd --nocache
ibeis -e print_acfg -a varysize2 --db NNP_MasterGIRM_core --verbtd --nocache
"""

varysize = {
    'qcfg': ut.augdict(
        __controlled_aidcfg, {
            #'default_aids': 'allgt',
            'sample_size': None,
            'sample_per_name': 1,
            #'force_const_size': True,
            'min_pername': 4,  # ensures each query will have a correct example for the groundtruth
        }),

    'dcfg': ut.augdict(
        __controlled_aidcfg, {
            #'default_aids': 'all',
            'sample_per_name': [1, 2, 3],
            #'sample_per_name': [1, 3],
            'exclude_reference': True,
            'sample_size': [0.25, 0.5, 0.75],  # .95], 1.0],
            'min_pername': 1,
        }),
}

"""
ibeis -e print_acfg -a varysize2_td --db PZ_Master1 --verbtd --nocache
"""
varysize_td = apply_timecontrol(varysize)
varysize_td1h = apply_timecontrol(varysize, '1h')
varysize_tdqual = apply_qualcontrol(varysize_td)


#varysize_pzm = {
#    'qcfg': ut.augdict(
#        varysize['qcfg'], {
#            'sample_size': 500,
#        }),

#    'dcfg': ut.augdict(
#        varysize['dcfg'], {
#            #'sample_size': [1500, 2000, 2500, 3000],
#            'sample_size': [1500, 2500, 3500, 4500],
#            #'sample_size': [1500, 17500, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500],
#        }),
#}


#varysize_gz = {
#    'qcfg': ut.augdict(
#        varysize['qcfg'], {
#            'sample_size': 60,
#        }),

#    'dcfg': ut.augdict(
#        varysize['dcfg'], {
#            'sample_size': [200, 300, 400, 500],
#        }),
#}


#varysize_girm = {
#    'qcfg': ut.augdict(
#        varysize['qcfg'], {
#            'sample_size': 30,
#        }),

#    'dcfg': ut.augdict(
#        varysize['dcfg'], {
#            'sample_size': [60, 90, 120, 150],
#        }),
#}


# Compare query of frontleft animals when database has only left sides
"""
ibeis -e print_acfg -a viewpoint_compare --db PZ_Master1 --verbtd --nocache

python -m ibeis.experiments.experiment_helpers --exec-parse_acfg_combo_list -a viewpoint_compare
python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_Master1 -a viewpoint_compare
python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_Master1 -a viewpoint_compare --verbtd
# Check composition of names per viewpoint
python -m ibeis.ibsfuncs --exec-group_annots_by_multi_prop --db PZ_Master1 --props=yaw_texts,name_rowids --keys1 frontleft
python -m ibeis.ibsfuncs --exec-get_annot_stats_dict --db PZ_Master1 --per_name_vpedge=True


TODO: Need to explicitly setup the common config I think?

ibeis -e print_acfg -a viewdiff:min_timedelta=1h --db PZ_Master1 --verbtd --nocache-aid
"""
viewpoint_compare = {
    'qcfg': ut.augdict(
        controlled['qcfg'], {
            #'view_pername': 'len(primary) > 2 and len(primary1) > 2',
            'sample_size': None,
            'view_pername': '#primary>0&#primary1>1',  # To be a query you must have at least two primary1 views and at least one primary view
            'force_const_size': True,
            'view': 'primary1',
            'sample_per_name': 1,
            #'min_pername': 2,
        }),

    'dcfg': ut.augdict(
        controlled['dcfg'], {
            'view': ['primary1', 'primary'],
            'force_const_size': True,
            'view_pername': '#primary>0&#primary1>1',  # To be a query you must have at least two primary1 views and at least one primary view
            #'view': ['primary1', 'primary1'],  # daids are not the same here. there is a nondetermenism (ordering problem)
            #'view': ['primary'],
            #'sample_per_name': 1,
            #'sample_rule_ref': 'maxtimedelta',
            'sample_per_ref_name': 1,
            'sample_per_name': 1,  # None this seems to produce odd results where the per_ref is still more then 1
            'sample_size': None,  # TODO: need to make this consistent accross both experiment modes
        }),
}


viewdiff = vp = viewpoint_compare = {
    'qcfg': ut.augdict(
        controlled['qcfg'], {
            #'view_pername': 'len(primary) > 2 and len(primary1) > 2',
            'sample_size': None,
            'view_pername': '#primary>0&#primary1>0',  # To be a query you must have at least two primary1 views and at least one primary view
            'force_const_size': True,
            'view': 'primary1',
            'sample_per_name': 1,
            #'min_pername': 2,
        }),

    'dcfg': ut.augdict(
        controlled['dcfg'], {
            'view': ['primary'],
            'force_const_size': True,
            #'view_pername': '#primary>0&#primary1>0',  # To be a query you must have at least two primary1 views and at least one primary view
            #'view': ['primary1', 'primary1'],  # daids are not the same here. there is a nondetermenism (ordering problem)
            #'view': ['primary'],
            #'sample_per_name': 1,
            #'sample_rule_ref': 'maxtimedelta',
            'sample_per_ref_name': 1,
            'sample_per_name': 1,  # None this seems to produce odd results where the per_ref is still more then 1
            'sample_size': None,  # TODO: need to make this consistent accross both experiment modes
        }),
}

"""
ibeis -e print_acfg -a viewdiff --db PZ_Master1 --verbtd --nocache --per_vp=True
ibeis -e print_acfg -a viewdiff_td --db PZ_Master1 --verbtd --nocache --per_vp=True
"""
viewdiff_td = apply_timecontrol(viewdiff)
viewdiff_td1h = apply_timecontrol(viewdiff, '1h')

# THIS IS A GOOD START
# NEED TO DO THIS CONFIG AND THEN SWITCH DCFG TO USE primary1

include_vars = list(vars().keys())  # this line is after tests

# List of all valid tests
TEST_NAMES = set(include_vars) - set(exclude_vars)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.experiments.annotation_configs
        python -m ibeis.experiments.annotation_configs --allexamples
        python -m ibeis.experiments.annotation_configs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
