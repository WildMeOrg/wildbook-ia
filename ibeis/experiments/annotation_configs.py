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
    'aids'     : 'default_aids',
    'pername'  : 'sample_per_name',
    'offset'   : 'sample_offset',
    'refrule'  : 'sample_rule_ref',
    'rule'     : 'sample_rule',
    'size'     : 'sample_size',
    'mingt'    : 'gt_min_per_name',
}


# Base common settings, but some default settings will be different
# for query and database annotations
__default_aidcfg = {
    'default_aids'        : 'all',  # initial set to choose from
    #'include_aids'       : None,   # force inclusion?
    'gt_avl_aids'         : None,   # The only aids available as reference groundtruth
    # Default filtering
    'species'             : 'primary',  # specify the species
    'minqual'             : 'poor',
    'is_known'            : None,
    'view'                : None,
    'view_ext'            : 0,      # num viewpoints to extend in dir1 and dir2
    'view_ext1'           : None,   # num viewpoints to extend in dir1
    'view_ext2'           : None,   # num viewpoints to extend in dir2
    'view_pername'        : None,   # formatted string filtering the viewpoints
    'require_quality'     : False,  # if True unknown qualities are removed
    'require_viewpoint'   : False,
    'require_timestamp'   : False,
    'force_const_size'    : False,  # forces a consistnet sample size across combinations
    #'exclude_aids'       : None,   # removes specified aids from selection
    # Filtered selection
    'exclude_reference'   : None,  # excludes any aids specified in a reference set (ie qaids)
    'ref_has_viewpoint'   : None,  # All aids must have a gt with this viewpoint
    'ref_has_qual'        : None,  # All aids must have a gt with this viewpoint
    'gt_min_per_name'     : None,  # minimum numer of aids for each name in sample
    'sample_per_name'     : None,  # Choos num_annots to sample from each name.
    'sample_per_ref_name' : None,  # when sampling daids, choose this many correct matches per query
    'sample_rule_ref'     : 'random',
    'sample_rule'         : 'random',
    'sample_offset'       : 0,
    'sample_size'         : None,  # Tries to get as close to sample size without removing othe properties
    #'name_choose_rule'   : 'timestamp',  # Choose #annots for each name
    # Final indexing
    'shuffle'             : False,  # randomize order before indexing
    'index'               : None,   # choose only a subset
    'min_timedelta'       : None,
}


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


def get_varied_labels(acfg_list):
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
        _varied_keys = list(set(ut.flatten([list(ut.flatten([list(x.keys()) for x in unflatten_acfgdict(cfg).values()])) for cfg in varied_acfg_list])))
        _acfg_list = [compress_aidcfg(acfg, force_noncommon=_varied_keys) for acfg in acfg_list]
        flat_acfg_list = flatten_acfg_list(_acfg_list)
        nonvaried_dict, varied_acfg_list = cfghelpers.partition_varied_cfg_list(flat_acfg_list)

    shortened_cfg_list = [
        #{shorten_to_alias_labels(key): val for key, val in _dict.items()}
        ut.map_dict_keys(shorten_to_alias_labels, _dict)
        for _dict in varied_acfg_list]
    nonlbl_keys = cfghelpers.INTERNAL_CFGKEYS
    nonlbl_keys = [prefix +  key for key in nonlbl_keys for prefix in ['', 'q', 'd']]
    shortened_lbl_list = [
        cfghelpers.get_cfg_lbl(cfg, name, nonlbl_keys)
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
    """
    Example:
        >>> from ibeis.experiments.annotation_configs import *  # NOQA
        >>> qcfg_list = [{'f': 1, 'b': 1}, {'f': 2, 'b': 1}, {'f': 3, 'b': 1, 'z': 4}]
        >>> acfg_list = [{'qcfg': qcfg} for qcfg in qcfg_list]
        >>> nonvaried_comp_dict, varied_comp_dict_list = compress_acfg_list_for_printing(acfg_list)
    """
    flat_acfg_list = flatten_acfg_list(acfg_list)
    nonvaried_dict, varied_acfg_list = cfghelpers.partition_varied_cfg_list(flat_acfg_list)
    nonvaried_compressed_dict = compress_aidcfg(unflatten_acfgdict(nonvaried_dict))
    varied_compressed_dict_list = [
        compress_aidcfg(unflatten_acfgdict(cfg), filter_empty=True)
        for cfg in varied_acfg_list]
    return nonvaried_compressed_dict, varied_compressed_dict_list


def print_acfg_list(acfg_list, expanded_aids_list=None, ibs=None, combined=False, **kwargs):
    kwargs = kwargs.copy()
    nonvaried_compressed_dict, varied_compressed_dict_list = compress_acfg_list_for_printing(acfg_list)
    ut.colorprint('+=== <Info acfg_list> ===', 'white')
    #print('Printing acfg_list info. len(acfg_list) = %r' % (len(acfg_list),))
    print('non-varied aidcfg = ' + ut.dict_str(nonvaried_compressed_dict))
    seen_ = ut.ddict(list)
    for acfgx in range(len(acfg_list)):
        acfg = acfg_list[acfgx]
        title = 'q_cfgname=' + acfg['qcfg']['_cfgname'] + ' d_cfgname=' + acfg['dcfg']['_cfgname']

        ut.colorprint('+--- acfg %d / %d -- %s ---- ' % (acfgx + 1, len(acfg_list), title), 'white')
        print('acfg = ' + ut.dict_str(varied_compressed_dict_list[acfgx], strvals=True))
        if expanded_aids_list is not None:
            qaids, daids = expanded_aids_list[acfgx]
            key = (ut.hashstr_arr27(qaids, 'qaids'), ut.hashstr_arr27(daids, 'daids'))
            if key not in seen_:
                seen_[key].append(acfgx)
                annotconfig_stats_strs, _ = ibs.get_annotconfig_stats(qaids, daids, verbose=True, combined=combined, **kwargs)
            else:
                print('DUPLICATE of index %r' % (seen_[key],))
                print('DUP OF acfg = ' + ut.dict_str(varied_compressed_dict_list[seen_[key][0]], strvals=True))
            #if combined:
            #    ibs.print_annot_stats(list(qaids) + list(daids), label='combined = ', **kwargs)
        #annotconfig_stats_strs, _ = ibs.get_annotconfig_stats(qaids, daids, verbose=False)
        #print(ut.dict_str(ut.dict_subset(annotconfig_stats_strs, ['num_qaids', 'num_daids', 'num_annot_intersect', 'aids_per_correct_name', 'aids_per_imposter_name', 'num_unmatchable_queries', 'num_matchable_queries'])))
        #_ = ibs.get_annotconfig_stats(qaids, daids)
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
            'default_aids': (1,)
        }),

    'dcfg': ut.augdict(
        __default_aidcfg, {
        }),
}


"""
python -m ibeis.dev -e get_annotcfg_list --db PZ_Master1 -a baseline
"""
uncontrolled = {
    'qcfg': ut.augdict(
        __baseline_aidcfg, {
            'default_aids': 'allgt',
            'gt_min_per_name': 2,
        }),

    'dcfg': ut.augdict(
        __baseline_aidcfg, {
        }),
}

"""
python -m ibeis.dev -e get_annotcfg_list --db PZ_Master1 -a controlled
"""
controlled = {
    'qcfg': ut.augdict(
        __controlled_aidcfg, {
            'default_aids': 'allgt',
            'sample_per_name': 1,
            'gt_min_per_name': 2,
            #'sample_size': 128,  # keep this small for now until we can run full results
        }),

    'dcfg': ut.augdict(
        __controlled_aidcfg, {
            'default_aids': 'all',
            'sample_per_name': 1,
            'exclude_reference': True,
            #'sample_size': 300,  # keep this small for now until we can run full results
            'gt_min_per_name': 1,  # allows for singletons to be in the database
        }),
}


#cont_uncon_compare = {
#    'qcfg': [
#        ut.augdict(
#            controlled['qcfg'], {
#                'force_const_size': True,
#            }),
#        ut.augdict(
#            uncontrolled['qcfg'], {
#                'force_const_size': True,
#            }),
#    ],

#    'dcfg': [
#        ut.augdict(
#            controlled['dcfg'], {
#                'force_const_size': True,
#            }),
#        ut.augdict(
#            uncontrolled['dcfg'], {
#                'force_const_size': True,
#            }),
#    ]
#}

largetimedelta = {
    'qcfg': ut.augdict(
        controlled['qcfg'], {
            'sample_rule': 'mintime',
            'require_timestamp': True,
            'min_timedelta': 60 * 60 * 24,
        }),

    'dcfg': ut.augdict(
        controlled['dcfg'], {
            'sample_rule_ref': 'maxtimedelta',
            'sample_per_name': None,
            'require_timestamp': True,
            'min_timedelta': 60 * 60 * 24,
        }),
}


controlled2 = {
    'qcfg': ut.augdict(
        controlled['qcfg'], {
            'sample_size': None,  # keep this small for now until we can run full results
        }),

    'dcfg': ut.augdict(
        controlled['dcfg'], {
            'sample_size': None,  # keep this small for now until we can run full results
        }),
}


#controlled = {
#    'qcfg': ut.augdict(
#        __controlled_aidcfg, {
#            'default_aids': 'allgt',
#            'sample_per_name': 1,
#            'gt_min_per_name': 2,  # ensures each query will have a correct example for the groundtruth
#        }),

#    'dcfg': ut.augdict(
#        __controlled_aidcfg, {
#            'default_aids': 'all',
#            'sample_per_name': 1,
#            'exclude_reference': True,
#            'sample_rule_ref': 'maxtimedelta',
#            'gt_min_per_name': 1,
#        }),
#}


# Just vary the samples per name without messing with the number of annots in the database
varypername = {
    'qcfg': ut.augdict(
        __controlled_aidcfg, {
            'default_aids': 'allgt',
            'sample_size': 50,
            'sample_per_name': 1,
            'gt_min_per_name': 4,  # ensures each query will have a correct example for the groundtruth
        }),

    'dcfg': ut.augdict(
        __controlled_aidcfg, {
            'default_aids': 'all',
            #'sample_per_name': [1, 2, 3],
            'sample_per_name': [1, 3],
            #'sample_per_ref_name': [1, 2, 3],
            'sample_per_ref_name': [1, 3],
            'exclude_reference': True,
            'gt_min_per_name': 1,
            'force_const_size': True,
        }),
}

varypername_pzm = {
    'qcfg': ut.augdict(
        varypername['qcfg'], {
            'sample_size': 500,
        }),

    'dcfg': ut.augdict(
        varypername['dcfg'], {
        }),
}

varypername_gz = {
    'qcfg': ut.augdict(
        varypername['qcfg'], {
            'sample_size': 200,
        }),

    'dcfg': ut.augdict(
        varypername['dcfg'], {
        }),
}

varypername_girm = {
    'qcfg': ut.augdict(
        varypername['qcfg'], {
            'sample_size': 50,
        }),

    'dcfg': ut.augdict(
        varypername['dcfg'], {
        }),
}


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
varysize = {
    'qcfg': ut.augdict(
        __controlled_aidcfg, {
            'default_aids': 'allgt',
            'sample_size': 50,
            'sample_per_name': 1,
            'gt_min_per_name': 4,  # ensures each query will have a correct example for the groundtruth
        }),

    'dcfg': ut.augdict(
        __controlled_aidcfg, {
            'default_aids': 'all',
            'sample_per_name': [1, 2, 3],
            'exclude_reference': True,
            'sample_size': [50, 200, 500],
            'gt_min_per_name': 1,
        }),
}


varysize_pzm = {
    'qcfg': ut.augdict(
        varysize['qcfg'], {
            'sample_size': 500,
        }),

    'dcfg': ut.augdict(
        varysize['dcfg'], {
            #'sample_size': [1500, 2000, 2500, 3000],
            'sample_size': [1500, 2500, 3500, 4500],
            #'sample_size': [1500, 17500, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500],
        }),
}


varysize_gz = {
    'qcfg': ut.augdict(
        varysize['qcfg'], {
            'sample_size': 60,
        }),

    'dcfg': ut.augdict(
        varysize['dcfg'], {
            'sample_size': [200, 300, 400, 500],
        }),
}


varysize_girm = {
    'qcfg': ut.augdict(
        varysize['qcfg'], {
            'sample_size': 30,
        }),

    'dcfg': ut.augdict(
        varysize['dcfg'], {
            'sample_size': [60, 90, 120, 150],
        }),
}


# Compare query of frontleft animals when database has only left sides
"""
python -m ibeis.experiments.experiment_helpers --exec-parse_acfg_combo_list -a viewpoint_compare
python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_Master1 -a viewpoint_compare
python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list --db PZ_Master1 -a viewpoint_compare --verbtd
# Check composition of names per viewpoint
python -m ibeis.ibsfuncs --exec-group_annots_by_multi_prop --db PZ_Master1 --props=yaw_texts,name_rowids --keys1 frontleft
python -m ibeis.ibsfuncs --exec-get_annot_stats_dict --db PZ_Master1 --per_name_vpedge=True

"""
viewpoint_compare = {
    'qcfg': ut.augdict(
        controlled['qcfg'], {
            #'view_pername': 'len(primary) > 2 and len(primary1) > 2',
            'sample_size': None,
            'view_pername': '#primary>0&#primary1>1',  # To be a query you must have at least two primary1 views and at least one primary view
            'force_const_size': True,
            'view': 'primary1',
            #'gt_min_per_name': 3,
        }),

    'dcfg': ut.augdict(
        controlled['dcfg'], {
            'view': ['primary1', 'primary'],
            'force_const_size': True,
            #'view': ['primary1', 'primary1'],  # daids are not the same here. there is a nondetermenism (ordering problem)
            #'view': ['primary'],
            #'sample_per_name': 1,
            #'sample_rule_ref': 'maxtimedelta',
            'sample_per_ref_name': 1,
            'sample_per_name': None,  # this seems to produce odd results where the per_ref is still more then 1
            'sample_size': None,  # TODO: need to make this consistent accross both experiment modes
        }),
}

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
