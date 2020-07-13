# -*- coding: utf-8 -*-
"""
Definitions for common aid configurations

Rename to annot_cfgdef
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np  # NOQA

print, rrr, profile = ut.inject2(__name__)


# easier to type names to alias some of these options
ALIAS_KEYS = {
    # 'aids'     : 'default_aids',
    'pername': 'sample_per_name',
    'offset': 'sample_offset',
    'refrule': 'sample_rule_ref',
    'rule': 'sample_rule',
    'size': 'sample_size',
    'mingt': 'min_pername',
    'excluderef': 'exclude_reference',
}


INDEPENDENT_DEFAULTS_PARAM_INFO = [
    ut.ParamInfo('reviewed', None, valid_values=[True, False, None]),
    ut.ParamInfo(
        'minqual', None, valid_values=[None, 'junk', 'poor', 'ok', 'good', 'excellent']
    ),
    ut.ParamInfo('multiple', None, valid_values=[True, False, None]),
    ut.ParamInfo('species', None),
    ut.ParamInfo('view', None),  # TODO: allow for lists
    ut.ParamInfo('require_quality', None, valid_values=[True, False, None]),
    ut.ParamInfo('require_viewpoint', None, valid_values=[True, False, None]),
    ut.ParamInfo('is_exemplar', None, valid_values=[True, False, None]),
    ut.ParamInfo(
        'min_pername_global',
        None,
        type_=int,
        min_=0,
        help_='Keep annot if it has at least this many global names',
    ),
    ut.ParamInfo(
        'max_pername_global',
        None,
        type_=int,
        min_=0,
        help_='Keep annot if it has at most this many global names',
    ),
    ut.ParamInfo(
        'min_unixtime',
        None,
        type_=float,
        min_=0,
        help_='Remove anything before this timestamp',
    ),
    ut.ParamInfo(
        'max_unixtime',
        None,
        type_=float,
        min_=0,
        help_='Remove anything after this timestamp',
    ),
    # ut.ParamInfo('view', None),
]


INTRAGROUP_DEFAULTS_PARAM_INFO = [
    ut.ParamInfo(
        'min_pername',
        None,
        type_=int,
        min_=0,
        help_='Keeps names with at least this number of aids within the group',
    ),
    ut.ParamInfo(
        'max_pername',
        None,
        type_=int,
        min_=0,
        help_='Keeps names with at most this number of aids within the group',
    ),
]

SAMPLE_DEFAULTS_PARAM_INFO = [
    ut.ParamInfo(
        'sample_per_name',
        None,
        type_=int,
        min_=0,
        help_='Take this many annots per name',
    ),
    ut.ParamInfo(
        'sample_rule',
        'random',
        valid_values=['random', 'mintime', 'maxtime', 'qual_and_view'],
        help_='Method of samping from names',
    ),
    ut.ParamInfo(
        'sample_seed',
        0,
        type_=int,
        none_ok=True,
        help_='Random seed for sampling from names',
    ),
]

SUBINDEX_DEFAULTS_PARAM_INFO = [
    ut.ParamInfo('index', None),
]

OTHER_DEFAULTS = {
    # forces a consistnet sample size across combinations
    'force_const_size': None,
    'crossval_enc': None,
    # 'hack_extra' : None,  # hack param to make bigger db sizes
    # 'hack_imageset': None,
    #  Hack out errors in test data
    # 'hackerrors'    : True,
    'hackerrors': False,
    'joinme': None,
}

# Defaults for the independent filter
# THese filters are orderless
INDEPENDENT_DEFAULTS = {
    # 'species'             : 'primary',  # specify the species
    # 'species'             : None,
    # Timedelta Params
    'require_timestamp': None,
    'require_gps': None,
    'max_timestamp': None,
    'contributor_contains': None,
    # Quality Params
    # 'require_quality'     : None,  # if True unknown qualities are removed
    # 'minqual'             : 'poor',
    'minqual': None,
    'been_adjusted': None,  # HACK PARAM
    # Viewpoint params
    # 'require_viewpoint'   : None,
    # 'view'                : None,
    'view_ext': 0,  # num viewpoints to extend in dir1 and dir2
    'view_ext1': None,  # num viewpoints to extend in dir1
    'view_ext2': None,  # num viewpoints to extend in dir2
    'is_known': None,
    # maximum number of features detected by default config
    'min_numfeat': None,
    # minimum number of features detected by default config
    'max_numfeat': None,
    'reviewed': None,
    'multiple': None,
}

# HACK
from wbia import tag_funcs  # NOQA  #

# Build Filters
filter_keys = ut.get_func_kwargs(tag_funcs.filterflags_general_tags)
for key in filter_keys:
    INDEPENDENT_DEFAULTS[key] = None

for pi in INDEPENDENT_DEFAULTS_PARAM_INFO:
    INDEPENDENT_DEFAULTS[pi.varname] = pi.default


INTRAGROUP_DEFAULTS = {
    # if True all annots must belong to the same imageset
    'same_imageset': None,
    'view_pername': None,  # formatted string filtering the viewpoints
    'min_timedelta': None,
    # minimum number of aids for each name in sample
    # 'min_pername'         : None,
    # 'max_pername'         : None,
    'min_spacedelta': None,
    'min_spacetimedelta': None,
}
for pi in INTRAGROUP_DEFAULTS_PARAM_INFO:
    INTRAGROUP_DEFAULTS[pi.varname] = pi.default

# HACK
INDEPENDENT_DEFAULTS.update(INTRAGROUP_DEFAULTS)  # hack

SUBINDEX_DEFAULTS = {
    # Final indexing
    'shuffle': False,  # randomize order before indexing
    # 'index'               : None,   # choose only a subset
}
for pi in SUBINDEX_DEFAULTS_PARAM_INFO:
    SUBINDEX_DEFAULTS[pi.varname] = pi.default

SAMPLE_DEFAULTS = {
    'sample_size': None,
    'num_names': None,
    # Gets as close to sample size without removing other props
    # Per Name / Exemplar Params
    # 'sample_per_name'     : None,  # Choos num_annots to sample from each name.
    # 'sample_rule'         : 'random',
    'sample_offset': None,  # UNUSED
    'occur_offset': None,  # UNUSED
    'name_offset': None,  # UNUSED
    'sample_occur': None,
}
for pi in SAMPLE_DEFAULTS_PARAM_INFO:
    SAMPLE_DEFAULTS[pi.varname] = pi.default

SAMPLE_REF_DEFAULTS = {
    # excludes any aids specified in a reference set (ie qaids)
    'exclude_reference': None,
    'exclude_ref_contact': None,
    'sample_rule_ref': 'random',
    # when sampling daids, choose this many correct matches per query
    'sample_per_ref_name': None,
}


# Base common settings, but some default settings will be different
# for query and database annotations
DEFAULT_AIDCFG = ut.merge_dicts(
    OTHER_DEFAULTS,
    INDEPENDENT_DEFAULTS,
    SAMPLE_DEFAULTS,
    SAMPLE_REF_DEFAULTS,
    SUBINDEX_DEFAULTS,
)
__default_aidcfg = DEFAULT_AIDCFG


def compress_aidcfg(acfg, filter_nones=False, filter_empty=False, force_noncommon=[]):
    r"""
    Idea is to add a third subconfig named `common` that is the intersection of
    `qcfg` and `dcfg`.

    Args:
        acfg (dict):

    Returns:
        dict: acfg

    CommandLine:
        #python -m wbia --tf compress_aidcfg
        python -m wbia.expt.annotation_configs --exec-compress_aidcfg --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.annotation_configs import *  # NOQA
        >>> acfg = default
        >>> acfg = compress_aidcfg(acfg)
        >>> result = ('acfg = %s' % (ut.repr2(acfg),))
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
        assert (
            acfg['qcfg']['_cfgname'] == acfg['dcfg']['_cfgname']
        ), 'should be the same for now'

    # Hack to make common params between q and d appear the same
    _acfg_list = [compress_aidcfg(acfg) for acfg in acfg_list]

    flat_acfg_list = flatten_acfg_list(_acfg_list)
    tup = ut.partition_varied_cfg_list(flat_acfg_list)
    flat_nonvaried_dict, flat_varied_acfg_list = tup
    nonvaried_dict = unflatten_acfgdict(flat_nonvaried_dict)
    varied_acfg_list = [unflatten_acfgdict(acfg) for acfg in flat_varied_acfg_list]
    return nonvaried_dict, varied_acfg_list


def get_varied_acfg_labels(acfg_list, mainkey='_cfgname', checkname=False):
    """
    >>> from wbia.expt.annotation_configs import *  # NOQA
    """
    # print(ut.repr2(varied_acfg_list, nl=2))
    for acfg in acfg_list:
        assert acfg['qcfg'].get(mainkey, '') == acfg['dcfg'].get(
            mainkey, ''
        ), 'should be the same for now'
    cfgname_list = [acfg['qcfg'].get(mainkey, '') for acfg in acfg_list]
    if checkname and ut.allsame(cfgname_list):
        cfgname_list = [None] * len(cfgname_list)

    # Hack to make common params between q and d appear the same
    _acfg_list = [compress_aidcfg(acfg) for acfg in acfg_list]

    flat_acfg_list = flatten_acfg_list(_acfg_list)
    nonvaried_dict, varied_acfg_list = ut.partition_varied_cfg_list(flat_acfg_list)

    SUPER_HACK = True
    if SUPER_HACK:
        # SUPER HACK, recompress remake the varied list after knowing what is varied
        _varied_keys = list(
            set(
                ut.flatten(
                    [
                        list(
                            ut.flatten(
                                [list(x.keys()) for x in unflatten_acfgdict(cfg).values()]
                            )
                        )
                        for cfg in varied_acfg_list
                    ]
                )
            )
        )
        _acfg_list = [
            compress_aidcfg(acfg, force_noncommon=_varied_keys) for acfg in acfg_list
        ]
        flat_acfg_list = flatten_acfg_list(_acfg_list)
        nonvaried_dict, varied_acfg_list = ut.partition_varied_cfg_list(flat_acfg_list)

    shortened_cfg_list = [
        # {shorten_to_alias_labels(key): val for key, val in _dict.items()}
        ut.map_dict_keys(shorten_to_alias_labels, _dict)
        for _dict in varied_acfg_list
    ]
    nonlbl_keys = ut.INTERNAL_CFGKEYS
    nonlbl_keys = [prefix + key for key in nonlbl_keys for prefix in ['', 'q', 'd']]
    # hack for sorting by q/d stuff first

    def get_key_order(cfg):
        keys = [k for k in cfg.keys() if k not in nonlbl_keys]
        sortorder = [2 * k.startswith('q') + 1 * k.startswith('d') for k in keys]
        return ut.sortedby(keys, sortorder)[::-1]

    cfglbl_list = [
        ut.get_cfg_lbl(cfg, name, nonlbl_keys, key_order=get_key_order(cfg))
        for cfg, name in zip(shortened_cfg_list, cfgname_list)
    ]

    if checkname:
        cfglbl_list = [x.lstrip(':') for x in cfglbl_list]
    return cfglbl_list


def shorten_to_alias_labels(key):
    if key is None:
        return None
    search_list = list(ALIAS_KEYS.values()) + ['qcfg_', 'dcfg_', 'common_']
    repl_list = list(ALIAS_KEYS.keys()) + ['q', 'd', '']
    return ut.multi_replace(key, search_list, repl_list)


def flatten_acfg_list(acfg_list):
    """
    Returns a new config where subconfig params are prefixed by subconfig keys
    """
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
        python -m wbia --tf compress_acfg_list_for_printing

    Ignore:
        >>> from wbia.expt.annotation_configs import *  # NOQA
        >>> qcfg_list = [{'f': 1, 'b': 1}, {'f': 2, 'b': 1}, {'f': 3, 'b': 1, 'z': 4}]
        >>> acfg_list = [{'qcfg': qcfg} for qcfg in qcfg_list]
        >>> nonvaried_dict, varied_dicts = compress_acfg_list_for_printing(acfg_list)
        >>> result = ('varied_dicts = %s\n' % (ut.repr2(varied_dicts),))
        >>> result += ('nonvaried_dict = %s' % (ut.repr2(nonvaried_dict),))
        >>> print(result)
    """
    flat_acfg_list = flatten_acfg_list(acfg_list)
    tup = ut.partition_varied_cfg_list(flat_acfg_list)
    nonvaried_dict, varied_acfg_list = tup
    nonvaried_compressed_dict = compress_aidcfg(
        unflatten_acfgdict(nonvaried_dict), filter_nones=True
    )
    varied_compressed_dict_list = [
        compress_aidcfg(unflatten_acfgdict(cfg), filter_empty=True)
        for cfg in varied_acfg_list
    ]
    return nonvaried_compressed_dict, varied_compressed_dict_list


def print_acfg_list(
    acfg_list,
    expanded_aids_list=None,
    ibs=None,
    combined=False,
    only_summary=False,
    **kwargs,
):
    r"""
    Args:
        acfg_list (list):
        expanded_aids_list (list): (default = None)
        ibs (IBEISController):  wbia controller object(default = None)
        combined (bool): (default = False)

    CommandLine:
        python -m wbia.expt.annotation_configs --exec-print_acfg_list

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.annotation_configs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> a = ['default']
        >>> acfg_list, expanded_aids_list = wbia.expt.experiment_helpers.get_annotcfg_list(
        >>>     ibs, acfg_name_list=a, verbose=0)
        >>> combined = False
        >>> result = print_acfg_list(acfg_list, expanded_aids_list, ibs, combined)
        >>> print(result)
    """
    _tup = compress_acfg_list_for_printing(acfg_list)
    nonvaried_compressed_dict, varied_compressed_dict_list = _tup

    ut.colorprint('+=== <Info acfg_list> ===', 'white')
    # print('Printing acfg_list info. len(acfg_list) = %r' % (len(acfg_list),))
    print('non-varied aidcfg = ' + ut.repr2(nonvaried_compressed_dict))
    seen_ = ut.ddict(list)

    # get default kwkeys for annot info
    if ibs is not None:
        annotstats_kw = kwargs.copy()
        kwkeys = ut.parse_func_kwarg_keys(ibs.get_annot_stats_dict)
        annotstats_kw.update(
            ut.argparse_dict(dict(zip(kwkeys, [None] * len(kwkeys))), only_specified=True)
        )

    hashid_list = []
    for acfgx in range(len(acfg_list)):
        acfg = acfg_list[acfgx]
        title = (
            'q_cfgname='
            + acfg['qcfg']['_cfgname']
            + ' d_cfgname='
            + acfg['dcfg']['_cfgname']
        )

        if not only_summary:
            ut.colorprint(
                '+--- acfg %d / %d -- %s ---- ' % (acfgx + 1, len(acfg_list), title),
                'lightgray',
            )
            print('acfg = ' + ut.repr2(varied_compressed_dict_list[acfgx], si=True))

        if expanded_aids_list is not None:
            qaids, daids = expanded_aids_list[acfgx]
            key = (ut.hashstr_arr27(qaids, 'qaids'), ut.hashstr_arr27(daids, 'daids'))
            if key not in seen_:
                if ibs is not None:
                    seen_[key].append(acfgx)
                    stats_ = ibs.get_annotconfig_stats(
                        qaids, daids, verbose=False, combined=combined, **annotstats_kw
                    )
                    hashids = (
                        stats_['qaid_stats']['qhashid'],
                        stats_['daid_stats']['dhashid'],
                    )
                    hashid_list.append(hashids)
                    stats_str2 = ut.repr2(
                        stats_, si=True, nl=True, explicit=False, nobraces=False
                    )
                    if not only_summary:
                        print('annot_config_stats = ' + stats_str2)
            else:
                dupindex = seen_[key]
                dupdict = varied_compressed_dict_list[dupindex[0]]
                if not only_summary:
                    print('DUPLICATE of index %r' % (dupindex,))
                    print('DUP OF acfg = ' + ut.repr2(dupdict, si=True))
    print('hashid summary = ' + ut.repr2(hashid_list, nl=1))
    ut.colorprint('L___ </Info acfg_list> ___', 'white')


def print_acfg(acfg, expanded_aids=None, ibs=None, **kwargs):
    print('acfg = ' + ut.repr2(compress_aidcfg(acfg)))
    if expanded_aids is not None:
        ibs.print_annot_stats(expanded_aids, label='expanded_aids = ', **kwargs)


def unflatten_acfgdict(flat_dict, prefix_list=['dcfg', 'qcfg']):
    acfg = {prefix: {} for prefix in prefix_list}
    for prefix in prefix_list:
        for key, val in flat_dict.items():
            if key.startswith(prefix + '_'):
                acfg[prefix][key[len(prefix) + 1 :]] = val
    return acfg


def apply_timecontrol(acfg, min_timedelta='6h', require_timestamp=True):
    return {
        'qcfg': ut.augdict(
            acfg['qcfg'],
            {'require_timestamp': require_timestamp, 'min_timedelta': min_timedelta},
        ),
        'dcfg': ut.augdict(
            acfg['dcfg'],
            {'require_timestamp': require_timestamp, 'min_timedelta': min_timedelta},
        ),
    }


def apply_qualcontrol(acfg):
    return {
        'qcfg': ut.augdict(acfg['qcfg'], {'require_quality': True}),
        'dcfg': ut.augdict(acfg['dcfg'], {'require_quality': True}),
    }


__baseline_aidcfg = ut.augdict(
    __default_aidcfg,
    {'is_known': True, 'minqual': 'ok', 'view': 'primary', 'view_ext': 1},
)


__controlled_aidcfg = ut.augdict(
    __baseline_aidcfg,
    {
        # 'require_timestamp': True,
        'view_ext': 0,
        'minqual': 'ok',
        'species': 'primary',
        'is_known': True,
    },
)

single_default = __default_aidcfg

exclude_vars = list(locals().keys())  # this line is before tests
exclude_vars.append('exclude_vars')

default = {
    'qcfg': ut.augdict(
        single_default,
        {
            # 'default_aids': (1,)
        },
    ),
    'dcfg': ut.augdict(single_default, {}),
}


default2 = {
    'qcfg': ut.augdict(default['qcfg'], {'exclude_reference': True, 'is_known': True}),
    'dcfg': ut.augdict(default['dcfg'], {'exclude_reference': True, 'is_known': True}),
}


"""
wbia -e print_acfg --db PZ_Master1 -a unctrl
"""
unctrl = uncontrolled = {
    'qcfg': ut.augdict(
        __baseline_aidcfg,
        {
            # 'default_aids': 'allgt',
            'min_pername': 2,
            'species': 'primary',
        },
    ),
    'dcfg': ut.augdict(__baseline_aidcfg, {'species': 'primary'}),
}


# Uncontrolled but comparable to controlled
unctrl_comp = {
    'qcfg': ut.augdict(
        __baseline_aidcfg,
        {
            # 'default_aids': 'allgt',
            'species': 'primary',
            'sample_per_name': 1,
            'min_pername': 2,
            'view_ext': 0,
        },
    ),
    'dcfg': ut.augdict(__baseline_aidcfg, {'species': 'primary'}),
}

"""
wbia -e print_acfg --db PZ_Master1 -a ctrl
wbia -e print_acfg --db PZ_Master1 -a unctrl ctrl::unctrl:qpername=1,qview_ext=0
wbia -e print_acfg --db PZ_Master1 -a unctrl ctrl::unctrl_comp
"""
ctrl = controlled = {
    'qcfg': ut.augdict(
        __controlled_aidcfg,
        {
            # 'default_aids': 'allgt',
            'sample_per_name': 1,
            'min_pername': 2,
        },
    ),
    'dcfg': ut.augdict(
        __controlled_aidcfg,
        {
            # 'default_aids': 'all',
            'sample_per_name': 1,
            'exclude_reference': True,
            'min_pername': 1,  # allows for singletons to be in the database
        },
    ),
}


"""
wbia -e print_acfg --db PZ_Master1 -a timectrl
"""
timectrl = timecontrolled = apply_timecontrol(ctrl)
timectrl1h = timecontrolled = apply_timecontrol(ctrl, '1h')

timectrlL = timecontrolled = apply_timecontrol(ctrl, require_timestamp=False)

"""
wbia -e print_acfg --db PZ_Master1 -a timequalctrl
"""
timequalctrl = timequalcontrolled = apply_qualcontrol(timectrl)


# Just vary the samples per name without messing with the number of annots in
# the database
varypername = {
    'qcfg': ut.augdict(
        ctrl['qcfg'],
        {
            # ensures each query will have a correct example for the groundtruth
            'min_pername': 4,
            'force_const_size': True,
        },
    ),
    'dcfg': ut.augdict(
        ctrl['qcfg'],
        {
            'sample_per_name': [1, 2, 3],
            # 'sample_per_name': [1, 3],
            # 'sample_per_ref_name': [1, 2, 3],
            # 'sample_per_ref_name': [1, 3],
            'force_const_size': True,
        },
    ),
}


varypername2 = {
    'qcfg': ut.augdict(ctrl['qcfg'], {'min_pername': 3, 'force_const_size': True}),
    'dcfg': ut.augdict(
        ctrl['dcfg'], {'sample_per_name': [1, 2], 'force_const_size': True}
    ),
}
varypername2_td = apply_timecontrol(varypername2)


"""
wbia -e print_acfg --db PZ_Master1 -a ctrl2
wbia -e print_acfg --db PZ_Master1 -a timectrl2
wbia -e rank_cmc --db PZ_Master1 -a timectrl2 -t invarbest
"""
ctrl2 = {
    'qcfg': ut.augdict(
        ctrl['qcfg'],
        {
            'min_pername': 3,
            # 'force_const_size': True,
        },
    ),
    'dcfg': ut.augdict(ctrl['dcfg'], {'sample_per_name': 2, 'force_const_size': True}),
}

timectrl2 = apply_timecontrol(ctrl2)


varypername_td = apply_timecontrol(varypername)
varypername_td1h = apply_timecontrol(varypername, '1h')
"""
wbia -e print_acfg --db PZ_Master1 -a varypername_tdqual
"""
varypername_tdqual = apply_qualcontrol(varypername_td)


"""
python -m wbia --tf get_num_annots_per_name --db PZ_Master1
wbia -e print_acfg -a varysize2 --db PZ_Master1 --verbtd --nocache
wbia -e print_acfg -a varysize2 --db NNP_MasterGIRM_core --verbtd --nocache
"""

varysize = {
    'qcfg': ut.augdict(
        __controlled_aidcfg,
        {
            # 'default_aids': 'allgt',
            'sample_size': None,
            'sample_per_name': 1,
            # 'force_const_size': True,
            'min_pername': 4,
        },
    ),
    'dcfg': ut.augdict(
        __controlled_aidcfg,
        {
            # 'default_aids': 'all',
            'sample_per_name': [1, 2, 3],
            # 'sample_per_name': [1, 3],
            'exclude_reference': True,
            'sample_size': [0.25, 0.5, 0.75],  # .95], 1.0],
            'min_pername': 1,
        },
    ),
}

"""
wbia -e print_acfg -a varysize2_td --db PZ_Master1 --verbtd --nocache
"""
varysize_td = apply_timecontrol(varysize)
varysize_td1h = apply_timecontrol(varysize, '1h')
varysize_tdqual = apply_qualcontrol(varysize_td)


varynannots = {
    'qcfg': ut.augdict(
        __controlled_aidcfg,
        {
            # 'default_aids': 'allgt',
            'sample_size': None,
            'sample_per_name': 1,
            # 'force_const_size': True,
            # 'min_pername': 4,
            'min_pername': 2,
        },
    ),
    'dcfg': ut.augdict(
        __controlled_aidcfg,
        {
            # 'default_aids': 'all',
            'sample_per_name': [1],
            # 'sample_per_name': [1, 3],
            'exclude_reference': True,
            # 'sample_size': [.01, .125, 0.25, .375, 0.5, .625, 0.75],  # , .875],  # .95], 1.0],
            # 'sample_size': [.01, .05, .125, 0.25, .375, 0.5, 0.75],  # , .875],  # .95], 1.0],
            'sample_size': [
                0.0,
                0.01,
                0.05,
                0.125,
                0.25,
                0.375,
                0.5,
                0.75,
                0.875,
                0.95,
                1.0,
            ],
            # 'sample_size': [.01, .025, .05, .125, 0.25, .375, 0.5, 0.75, .875, .95, 1.0],
            # 'sample_size': ((10 * np.logspace(0, np.log(100), num=11, base=np.e)).astype(np.int) / 1000).tolist(),
            # (10 * np.logspace(0, np.log2(100), num=11, base=2)).astype(np.int) / 1000,
            'min_pername': 1,
        },
    ),
}
varynannots_td = apply_timecontrol(varynannots)
varynannots_td1h = apply_timecontrol(varynannots, '1h')
# varysize_tdqual = apply_qualcontrol(varysize_td)


# Compare query of frontleft animals when database has only left sides
"""
wbia -e print_acfg -a viewpoint_compare --db PZ_Master1 --verbtd --nocache
python -m wbia --tf parse_acfg_combo_list -a viewpoint_compare
python -m wbia --tf get_annotcfg_list --db PZ_Master1 -a viewpoint_compare \
        --verbtd
# Check composition of names per viewpoint
python -m wbia --tf group_annots_by_multi_prop --db PZ_Master1 \
        --props=yaw_texts,name_rowids --keys1 frontleft
python -m wbia --tf get_annot_stats_dict --db PZ_Master1 \
        --per_name_vpedge=True


TODO: Need to explicitly setup the common config I think?
wbia -e print_acfg -a viewdiff:min_timedelta=1h --db PZ_Master1 --verbtd --nocache-aid
wbia --tf get_annotcfg_list -a viewdiff:min_timedelta=1h --db PZ_Master1 \
        --verbtd --nocache-aid
"""
viewpoint_compare = {
    'qcfg': ut.augdict(
        ctrl['qcfg'],
        {
            'sample_size': None,
            # To be a query you must have at least two primary1 views and at
            # least one primary view
            'view_pername': '#primary>0&#primary1>1',
            'force_const_size': True,
            'view': 'primary1',
            'sample_per_name': 1,
            # 'min_pername': 2,
        },
    ),
    'dcfg': ut.augdict(
        ctrl['dcfg'],
        {
            'view': ['primary1', 'primary'],
            'force_const_size': True,
            # To be a query you must have at least two primary1 views and at
            # least one primary view
            'view_pername': '#primary>0&#primary1>1',
            'sample_per_ref_name': 1,
            'sample_per_name': 1,
            # TODO: need to make this consistent accross both experiment modes
            'sample_size': None,
        },
    ),
}


viewdiff = vp = viewpoint_compare = {
    'qcfg': ut.augdict(
        ctrl['qcfg'],
        ut.odict(
            [
                ('sample_size', None),
                # To be a query you must have at least two primary1 views and at
                # least one primary view
                ('view_pername', '#primary>0&#primary1>0'),
                ('force_const_size', True),
                ('view', 'primary1'),
                ('sample_per_name', 1),
            ]
        ),
    ),
    'dcfg': ut.augdict(
        ctrl['dcfg'],
        {
            'view': ['primary'],
            'force_const_size': True,
            'sample_per_ref_name': 1,
            'sample_per_name': 1,  # None this seems to produce odd results
            # where the per_ref is still more then 1
            'sample_size': None,
            'view_pername': '#primary>0&#primary1>0',
        },
    ),
}


# Use tags to find a small set of difficult cases
timectrlhard = viewpoint_compare = {
    'qcfg': ut.augdict(
        timectrl['qcfg'],
        ut.odict(
            [
                ('has_any', ('needswork', 'correctable', 'mildviewpoint')),
                ('has_none', ('viewpoint', 'photobomb', 'error:viewpoint', 'quality')),
            ]
        ),
    ),
    'dcfg': ut.augdict(timectrl['dcfg'], {}),
}
"""
wbia -e print_acfg -a viewdiff --db PZ_Master1 --verbtd --nocache --per_vp=True
wbia -e print_acfg -a viewdiff_td --db PZ_Master1 --verbtd --nocache --per_vp=True
"""
viewdiff_td = apply_timecontrol(viewdiff)
viewdiff_td1h = apply_timecontrol(viewdiff, '1h')

r"""
wbia get_annotcfg_list --db Oxford -a default:qhas_any=\(query,\),dpername=2,exclude_reference=True --acfginfo --verbtd  --veryverbtd
wbia get_annotcfg_list --db Oxford -a oxford --acfginfo
('_QSUUIDS((55)qxlgljvomqpdvlny)', '_DSUUIDS((4240)vhtqsdkrwetbftis)'),

wbia draw_rank_cmc --db Oxford --save oxfordccm.png -p :proot=smk,num_words=[64000],nAssign=[1],sv_on=[False] -a oxford

"""
oxford = {
    'qcfg': ut.augdict(
        default['qcfg'],
        {
            'has_any': ('query',),
            'exclude_reference': True,
            'minqual': 'poor',
            'require_quality': False,
        },
    ),
    'dcfg': ut.augdict(
        default['dcfg'],
        {
            # 'sample_per_name': 2,
            'exclude_reference': True,
            'minqual': 'poor',
            'require_quality': False,
        },
    ),
}

# THIS IS A GOOD START
# NEED TO DO THIS CONFIG AND THEN SWITCH DCFG TO USE primary1

include_vars = list(locals().keys())  # this line is after tests

# List of all valid tests
TEST_NAMES = set(include_vars) - set(exclude_vars)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.expt.annotation_configs
        python -m wbia.expt.annotation_configs --allexamples
        python -m wbia.expt.annotation_configs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
