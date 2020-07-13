# -*- coding: utf-8 -*-
# TODO: Need test harness to do (something smarter than) gridsearch of these guys
"""
In this file dicts specify all possible combinations of the varied parameters
and lists specify the union of parameters

Rename to pipe_cfgdef
"""
from __future__ import absolute_import, division, print_function
import utool as ut

print, rrr, profile = ut.inject2(__name__)


ALIAS_KEYS = {
    'proot': 'pipeline_root',
    'RI': 'rotation_invariance',
    'AI': 'affine_invariance',
    # 'AQH': 'query_rotation_heuristic',
    'QRH': 'query_rotation_heuristic',
    'SV': 'sv_on',
    'nWords': 'num_words',
    # 'SVxy': 'xy_thresh',
    # 'SVxy': 'xy_thresh',
}


def augbase(basedict, updatedict):
    newdict = basedict.copy()
    newdict.update(updatedict)
    return newdict


def apply_param(cfg, **kwargs):
    import copy

    cfg = copy.deepcopy(cfg)
    for _ in cfg:
        _.update(**kwargs)
    return cfg


def apply_k(cfg):
    return apply_param(cfg, K=[1, 2, 3, 4, 5, 7, 10])


def apply_knorm(cfg):
    return apply_param(cfg, K=[1, 2, 3, 4, 10], Knorm=[1, 2, 3])


def apply_CircQRH(cfg):
    return apply_param(cfg, query_rotation_heuristic=True, affine_invariance=False)


def apply_Ell(cfg):
    return apply_param(cfg, query_rotation_heuristic=False, affine_invariance=True)


def apply_EllQRH(cfg):
    return apply_param(cfg, query_rotation_heuristic=True, affine_invariance=True)


exclude_vars = list(locals().keys())  # this line is before tests

default = [{}]

baseline = [{'resize_dim': 'area', 'dim_size': 450}]
# baseline = [{}]

smk = [{'proot': 'smk'}]

ScoreMech = candidacy_namescore = [
    {'score_method': ['nsum'], 'prescore_method': ['nsum']},
    # {
    #    'score_method':      ['nsum'],
    #    'prescore_method':   ['csum'],
    # },
    {'score_method': ['csum'], 'prescore_method': ['csum']},
]


CircQRH = apply_CircQRH(default)
Ell = apply_Ell(default)
EllQRH = apply_EllQRH(default)

CircQRH_K = apply_k(CircQRH)
CircQRH_Knrom = apply_knorm(CircQRH)
Ell_K = apply_k(Ell)
EllQRH_K = apply_k(EllQRH)

Ell_ScoreMech = apply_Ell(ScoreMech)
CircQRH_ScoreMech = apply_CircQRH(ScoreMech)


def best(metadata):
    """
    Infer the best pipeline config based on the metadata
    """
    if metadata is not None:
        ibs = metadata.get('ibs', None)
        if ibs is not None:
            dbname = ibs.get_dbname()
            if dbname == 'PZ_Master1':
                return apply_param(CircQRH, K=3)
            if dbname in ['GZ_Master1', 'GZ_ALL']:
                return apply_param(Ell, K=1)
            if dbname in ['NNP_MasterGIRM_core', 'GIRM_Master1']:
                return apply_param(Ell, K=2)
            if dbname in ['WS_Hard']:
                return apply_param(default)
    return default


featscoremetch = [
    {'lnbnn_on': True, 'fg_on': [True, False]},
    {'lnbnn_on': False, 'ratio_thresh': 0, 'fg_on': [True, False]},
    {'lnbnn_on': False, 'dist_on': True, 'fg_on': [True, False]},
    {'lnbnn_on': False, 'const_on': True, 'fg_on': [True, False]},
    # {
    #    'lnbnn_on': False,
    #    'lograt_on': True,
    # },
]


def get_candidacy_dbnames():
    return [
        'PZ_MTEST',
        # 'NNP_MasterGIRM_core',
        'PZ_Master0',
        'NNP_Master3',
        'GZ_ALL',
        'PZ_FlankHack',
        # 'JAG_Kelly',
        # 'JAG_Kieryn',
        # 'LF_Bajo_bonito',
        # 'LF_OPTIMIZADAS_NI_V_E',
        # 'LF_WEST_POINT_OPTIMIZADAS',
        # 'GZ_Master0',
        # 'GIR_Tanya',
    ]


# Test all combinations of invariance
invar = candinvar = candidacy_invariance = [
    {
        'affine_invariance': [True],
        'rotation_invariance': [False],
        'query_rotation_heuristic': [False],
    },
    {
        'affine_invariance': [True],
        'rotation_invariance': [True],
        'query_rotation_heuristic': [False],
    },
    {
        'affine_invariance': [False],
        'rotation_invariance': [True],
        'query_rotation_heuristic': [False],
    },
    {
        'affine_invariance': [False],
        'rotation_invariance': [False],
        'query_rotation_heuristic': [False],
    },
    {
        'affine_invariance': [True],
        'rotation_invariance': [False],
        'query_rotation_heuristic': [True],
    },
    {
        'affine_invariance': [False],
        'rotation_invariance': [False],
        'query_rotation_heuristic': [True],
    },
]

# Special value used to specify the current IBEIS configuration
custom = 'custom'


include_vars = list(locals().keys())  # this line is after tests

# List of all valid tests
TEST_NAMES = set(include_vars) - set(exclude_vars)
