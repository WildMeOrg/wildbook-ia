# -*- coding: utf-8 -*-
"""
Definitions for common aid configurations
"""
from __future__ import absolute_import, division, print_function
import utool as ut
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[aidcfg]')


# easier to type names to alias some of these options
ALIAS_KEYS = {
    'aids'     : 'default_aids',
    'per_name' : 'sample_per_name',
    'offset'   : 'sample_offset',
    'rule'     : 'sample_rule',
    'size'     : 'sample_size',
    'min_gt'   : 'gt_min_per_name',
}


# Base common settings, but some default settings will be different
# for query and database annotations
__default_aidcfg = {
    'default_aids'      : 'all',  # initial set to choose from
    #'include_aids'      : None,   # force inclusion?
    # Default filtering
    'species'           : 'primary',  # specify the species
    'minqual'           : 'poor',
    'is_known'          : None,
    'viewpoint_base'    : None,
    'viewpoint_counts'  : None,
    'viewpoint_range'   : 0,
    'require_quality'   : False,  # if True unknown qualities are removed
    'require_viewpoint' : False,
    'require_timestamp' : False,
    #'exclude_aids'      : None,   # removes specified aids from selection
    # Filtered selection
    'exclude_reference' : None,  # excludes any aids specified in a reference set (ie qaids)
    'ref_has_viewpoint' : None,  # All aids must have a gt with this viewpoint
    'ref_has_qual'      : None,  # All aids must have a gt with this viewpoint
    'gt_min_per_name'   : None,  # minimum numer of aids for each name in sample
    'sample_per_name'   : None,  # Choos num_annots to sample from each name.
    'sample_per_ref_name': None,  # when sampling daids, choose this many correct matches per query
    'sample_rule'       : 'random',
    'sample_offset'     : 0,
    'sample_size'       : None,  # Tries to get as close to sample size without removing othe properties
    #'name_choose_rule' : 'timestamp',  # Choose #annots for each name
    # Final indexing
    'shuffle'           : False,  # randomize order before indexing
    'index'             : None,   # choose only a subset
}


__controlled_aidcfg = ut.augdict(__default_aidcfg, {
    #'require_timestamp': True,
    'viewpoint_base': 'primary',
    'viewpoint_range': 0,
    'minqual': 'ok',
    'is_known': True,
})


def compress_aidcfg(acfg, filter_nones=False, filter_empty=False):
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
    acfg = copy.deepcopy(acfg)
    common_cfg = ut.dict_intersection(acfg['qcfg'], acfg['dcfg'])
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
    #print(ut.list_str(varied_acfg_list, nl=2))
    cfgname_list = []
    for aid in acfg_list:
        assert aid['qcfg']['_cfgname'] == aid['dcfg']['_cfgname'], 'should be the same for now'
        cfgname_list.append(aid['qcfg']['_cfgname'])

    flat_dict_list, nonvaried_dict, varied_acfg_list = partition_varied_acfg_list(acfg_list)

    shortened_cfg_list = [{shorten_to_alias_labels(key): val for key, val in _dict.items()} for _dict in varied_acfg_list]
    #shortened_lbl_list = [ut.dict_str(_dict, explicit=True, nl=False) for _dict in shortened_cfg_list]
    nonlbl_keys = ['_cfgstr', '_cfgname', '_cfgtype', 'q_cfgstr', 'q_cfgname', 'q_cfgtype', 'd_cfgstr', 'd_cfgname', 'd_cfgtype']
    shortened_lbl_list = [ut.dict_str(ut.delete_keys(_dict.copy(), nonlbl_keys), explicit=True, nl=False) for _dict in shortened_cfg_list]

    shortened_lbl_list = [ut.multi_replace(lbl, ['dict(', ')', ' '], ['', '', '']).rstrip(',') for lbl in  shortened_lbl_list]
    shortened_lbl_list = [cfgname + ':' + lbl for cfgname, lbl in zip(cfgname_list, shortened_lbl_list)]
    #print('\n'.join(shortened_lbl_list))
    return shortened_lbl_list


def shorten_to_alias_labels(key):
    search_list = list(ALIAS_KEYS.values()) + ['qcfg_', 'dcfg_']
    repl_list = list(ALIAS_KEYS.keys()) + ['q', 'd']
    return ut.multi_replace(key, search_list, repl_list)


def partition_varied_acfg_list(acfg_list):
    flat_dict_list = []
    for acfg in acfg_list:
        #compressed_acfg = annotation_configs.compress_aidcfg(test_result.acfg)
        flat_dict = {prefix + '_' + key: val for prefix, subdict in acfg.items() for key, val in subdict.items()}
        #compressed_acfg_list.append(compressed_acfg)
        flat_dict_list.append(flat_dict)
        #print(ut.dict_str(compressed_acfg))
    nonvaried_dict = reduce(ut.dict_intersection, flat_dict_list)
    varied_acfg_list = [ut.delete_dict_keys(_dict.copy(), list(nonvaried_dict.keys())) for _dict in flat_dict_list]
    return flat_dict_list, nonvaried_dict, varied_acfg_list


def compress_acfg_list_for_printing(acfg_list):
    flat_dict_list, nonvaried_dict, varied_acfg_list = partition_varied_acfg_list(acfg_list)
    nonvaried_compressed_dict = compress_aidcfg(unflatten_acfgdict(nonvaried_dict))
    varied_compressed_dict_list = [compress_aidcfg(unflatten_acfgdict(cfg), filter_empty=True) for cfg in varied_acfg_list]
    return nonvaried_compressed_dict, varied_compressed_dict_list


def print_acfg_list(acfg_list, expanded_aids_list=None, ibs=None, **kwargs):
    nonvaried_compressed_dict, varied_compressed_dict_list = compress_acfg_list_for_printing(acfg_list)
    ut.colorprint('+=== <Info acfg_list> ===', 'white')
    #print('Printing acfg_list info. len(acfg_list) = %r' % (len(acfg_list),))
    print('non-varied aidcfg = ' + ut.dict_str(nonvaried_compressed_dict))
    for acfgx in range(len(acfg_list)):
        ut.colorprint('+--- acfg %d / %d ---- ' % (acfgx + 1, len(acfg_list)), 'white')
        print('acfg = ' + ut.dict_str(varied_compressed_dict_list[acfgx]))
        if expanded_aids_list is not None:
            qaids, daids = expanded_aids_list[acfgx]
            annotconfig_stats_strs, _ = ibs.get_annotconfig_stats(qaids, daids, verbose=True, **kwargs)
        #annotconfig_stats_strs, _ = ibs.get_annotconfig_stats(qaids, daids, verbose=False)
        #print(ut.dict_str(ut.dict_subset(annotconfig_stats_strs, ['num_qaids', 'num_daids', 'num_annot_intersect', 'aids_per_correct_name', 'aids_per_imposter_name', 'num_unmatchable_queries', 'num_matchable_queries'])))
        #_ = ibs.get_annotconfig_stats(qaids, daids)
    ut.colorprint('L___ </Info acfg_list> ___', 'white')


def print_acfg(acfg):
    print(ut.dict_str(compress_aidcfg(acfg)))


def unflatten_acfgdict(flat_dict, prefix_list=['dcfg', 'qcfg']):
    acfg = {prefix: {} for prefix in prefix_list}
    for prefix in prefix_list:
        for key, val in flat_dict.items():
            if key.startswith(prefix + '_'):
                acfg[prefix][key[len(prefix) + 1:]] = val
    return acfg


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

controlled = {
    'qcfg': ut.augdict(
        __controlled_aidcfg, {
            'default_aids': 'allgt',
            'sample_per_name': 1,
            'gt_min_per_name': 2,
            'sample_size': 128,  # keep this small for now until we can run full results
        }),

    'dcfg': ut.augdict(
        __controlled_aidcfg, {
            'default_aids': 'all',
            'sample_per_name': 1,
            'exclude_reference': True,
            'sample_size': 300,  # keep this small for now until we can run full results
            'gt_min_per_name': 1,  # allows for singletons to be in the database
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
#            'sample_rule': 'ref_max_timedelta',
#            'gt_min_per_name': 1,
#        }),
#}


# Compare query of frontleft animals when database has only left sides
"""
python -m ibeis.init.main_helpers --exec-testdata_ibeis --db NNP_Master3 -a viewpoint_compare

"""
viewpoint_compare = {
    'qcfg': ut.augdict(
        controlled['qcfg'], {
            #'viewpoint_counts': 'len(primary) > 2 and len(primary1) > 2',
            'viewpoint_counts': '#primary>=1&#primary1>=2',
            'viewpoint_base': 'primary1',
            #'gt_min_per_name': 3,
        }),

    'dcfg': ut.augdict(
        controlled['dcfg'], {
            'viewpoint_base': ['primary', 'primary1'],
            'sample_per_name': 1,
            #'sample_per_name': None, # this seems to produce odd results where the per_ref is still more then 1
            'sample_per_ref_name': 1,
            'sample_size': None,
        }),
}

# THIS IS A GOOD START
# NEED TO DO THIS CONFIG AND THEN SWITCH DCFG TO USE primary1


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
            'sample_per_name': [1, 2, 3],
            'exclude_reference': True,
            'gt_min_per_name': 1,
        }),
}


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
            'sample_size': [50, 100, 200, 300, 500],
            'gt_min_per_name': 1,
        }),
}


varysize2 = {
    'qcfg': ut.augdict(
        varysize['qcfg'], {
        }),

    'dcfg': ut.augdict(
        varysize['dcfg'], {
            'sample_size': [50, 100, 200, 300, 500, 1000, 2000, 3000],
        }),
}

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
