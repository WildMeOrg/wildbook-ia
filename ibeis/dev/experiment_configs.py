# TODO: Need test harness to do a gridsearch of these guys
"""
In this file dicts specify all possible combinations of the varied parameters
and lists specify the union of parameters
"""

from __future__ import absolute_import, division, print_function
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[cfgbank]')
# Python


def augbase(basedict, updatedict):
    newdict = basedict.copy()
    newdict.update(updatedict)
    return newdict

exclude_vars = vars().keys()   # this line is before tests


small_best = {
    'pipeline_root':   ['vsmany'],
    'checks':          [1024],  # , 8192],
    'K':               [4],  # 5, 10],
    #'xy_thresh':      [.01],  # [.002],
    'xy_thresh':       [.005],  # # [.002],
    'nShortlist':      [50],
    #'use_chip_extent': [False, True],
    'use_chip_extent': [True],
    'sv_on':           [True],  # True, False],
    'score_method':    ['csum'],  # 'bordaw', 'topk', 'topkw'],  # , 'nsum', 'borda', 'topk', 'nunique']
    'prescore_method': ['csum'],  # 'bordaw', 'topk', 'topkw'],  # , 'nsum', 'borda', 'topk', 'nunique']
    'chip_sqrt_area':  [450],
    'fg_on'   : [True],
}


nsum = augbase(small_best, {
    'score_method':      ['nsum'],
    'prescore_method':   ['nsum'],
})


rrvsone_best = augbase(nsum, {
    'rrvsone_on': [True],
})

rrvsone_grid = augbase(rrvsone_best, {
    'grid_scale_factor': [.05, .1, .15, .2, .25, .3, .5][::1],
    'grid_steps': [1, 3, 7],
    'grid_sigma': [1.2, 1.6, 2.0],
})

nsum_nosv = augbase(nsum, {
    'sv_on':    [False],
    'score_method':      ['nsum'],
    'prescore_method':   ['nsum'],
})

vsmany = augbase(small_best, {
    'pipeline_root':   ['vsmany'],
    'K':               [4],  # 5, 10],
    'Knorm':           [1],  # 2, 3],
    'lnbnn_weight':    [1],  # 1,]
    'prescore_method':  ['nsum'],
    'score_method':  ['nsum'],
})

vsone = augbase(small_best, {
    'pipeline_root':  ['vsone'],
    'checks':        [256],
    'K':             [2],
    'Knorm':         [1],
    'lnbnn_weight':  [0],
    'ratio_weight':  [1.0],
    'ratio_thresh':  [.625],
    'prescore_method':  ['csum'],
    'score_method':  ['csum'],
})

pzmastertest = augbase(small_best, {
    'K': [4, 8, 10, 16, 29],
    'Knorm': [1, 4, 7],
})

fgweight = augbase(small_best, {
    'fg_on':  [True]  # , 0.0],
})


vary_sver = augbase(small_best, {
    'sv_on'          : [False, True],
    'use_chip_extent' : [False, True],
    'xy_thresh'       : [.1, .01, .001],
    'fg_on'   : [True],
    'algorithm'       : ['linear'],
})

sver_new = augbase(small_best, {
    'sv_on'           : [True],
    'use_chip_extent' : [True],
    'xy_thresh'       : [.001],
    'fg_on'   : [True],
    #'algorithm'       : ['linear'],
})

nov6 = augbase(small_best, {
    'K': [4, 5, 6, 7, 8, 9, 10, 20],
    'score_method':   ['csum'],  # 'bordaw', 'topk', 'topkw'],  # , 'nsum', 'borda', 'topk', 'nunique']
})


K = augbase(small_best, {
    'K': [4, 10],
})

# Feature parameters
featparams = {
    #'threshold': [16.0 / 3.0, 32.0 / 3.0],  # 8.0  / 3.0
    'numberOfScales': [1, 2, 3],
    #'maxIterations': [16, 32],
    #'convergenceThreshold': [.05, .1],
    #'initialSigma': [6.0, 3.0, 2.0, 1.6, 1.2, 1.1],
    'initialSigma': [3.2, 1.6, 0.8],
    #'edgeEigenValueRatio': [10, 5, 3],
}

featparams_big = {
    'threshold': [16.0 / 3.0, 32.0 / 3.0],  # 8.0  / 3.0
    'numberOfScales': [1, 2, 3],
    'maxIterations': [16, 32],
    'convergenceThreshold': [.05, .1],
    #'initialSigma': [6.0, 3.0, 2.0, 1.6, 1.2, 1.1],
    'initialSigma': [3.2, 2.4, 1.6, 1.2, 0.8],
    'edgeEigenValueRatio': [10, 5, 3],
}

smk00 = {
    'pipeline_root': ['smk', 'asmk', 'vsmany'],
    'sv_on':         [False, True],  # True, False],
    'nWords':        [64000],  # True, False],
}

# HACKED IN
featparams_big2 = augbase(
    featparams_big,
    {
        'K': [4, 7, 10, 20],
        'fg_on': [True],
    })
# low threshold = more keypoints
# low initialSigma = more keypoints


vsone_best = augbase(vsone, { })


smk_test2 = {
    'pipeline_root': ['smk', 'asmk'],
    'sv_on':         [True],  # True, False],
}

asmk = {
    'pipeline_root': ['asmk'],
    'sv_on':         [False],  # True, False],
}

smk = {
    'pipeline_root': ['smk'],
    'sv_on':         [False],  # True, False],
}

smk0 = {
    'pipeline_root': ['asmk'],
    'sv_on':         [False],  # True, False],
    'nWords':        [64000],  # True, False],
}

smk1 = {
    'pipeline_root': ['smk'],
    'sv_on':         [False],  # True, False],
    'nWords':        [64000],  # True, False],
}

smk2 = {
    'pipeline_root': ['smk', 'asmk', 'vsmany'],
    'sv_on':         [False, True],  # True, False],
    'nWords':        [64000],  # True, False],
}


# Test to make sure things are working for oxford
oxford = {
    'pipeline_root':    ['smk', 'asmk', 'vsmany'],
    'sv_on':            [False, True],  # True, False],
    'nWords':           [64000, 128000],
    'xy_thresh':        [.1, .01, .001],
    'use_chip_extent':  [True, False],
}

smkd = {
    'pipeline_root': ['smk'],
    'sv_on':         [False, True],  # True, False],
    'nWords':        [8000],  # True, False],
}

smk3 = {
    'pipeline_root': ['smk', 'asmk'],
    #'smk_thresh':    [0.0, 0.001],  # True, False],
    'nWords':        [64000],  # True, False],
}

smk5 = {
    'pipeline_root': ['smk', 'asmk', 'vsmany'],
    'sv_on':         [False, True],
    'smk_thresh':    [0.0, 0.001],
    'smk_alpha':     [3],
    'nWords':        [128000, 64000, 8000],
}


smk6 = {
    'pipeline_root': ['smk', 'asmk', 'vsmany'],
    #'pipeline_root': ['smk'],
    #'sv_on':         [False, True],
    'sv_on':         [True],
    'nAssign':       [2, 4, 10],
    'massign_equal_weights': [True, False],
    #'smk_thresh':    [0.0, 0.001],
    #'smk_alpha':     [3],
    #'nWords':        [128000, 64000, 8000],
    #'nWords':        [128000],
    'nWords':        [64000, 128000],
}


# when vocab is clipped only self matches will be allowed
# so set nAssign = 2 at least for this case (PZ_MTEST)
smk6d = {
    #'pipeline_root': ['smk', 'asmk', 'vsmany'],
    'pipeline_root': ['smk'],
    #'sv_on':         [False, True],
    'sv_on':         [True, False],
    'nAssign':       [10],
    'massign_equal_weights': [True],
    #'smk_thresh':    [0.0, 0.001],
    #'smk_alpha':     [3],
    #'nWords':        [128000, 64000, 8000],
    #'nWords':        [128000],
    #'nWords':        [64000, 128000],
    'allow_self_match': [False, True],
}


smk6_overnight = {
    'pipeline_root': ['smk', 'asmk', 'vsmany'],
    #'pipeline_root': ['smk', 'vsmany'],
    #'pipeline_root': ['smk'],
    'sv_on':          [True],
    'nAssign':        [4, 10],
    #'nAssign':       [2, 4, 10],
    'massign_equal_weights': [True],  # , False],
    #'smk_thresh':    [0.0, 0.0001],
    #'smk_alpha':     [3],
    'nWords':         [128000, 64000, 8000],
    #'nWords':        [128000],
}


smk7_overnight = {
    'pipeline_root': ['smk', 'asmk', 'vsmany'],
    #'pipeline_root': ['smk', 'vsmany'],
    #'pipeline_root': ['smk'],
    'loglnbnn_weight': [0.0, 1.0],  #
    'lnbnn_weight':   [0.0, 1.0],  #
    'crowded_weight': [0.0, 1.0],  #
    'sv_on':          [True],
    'nAssign':        [4, 10],
    #'nAssign':       [2, 4, 10],
    'massign_equal_weights': [True],  # , False],
    #'smk_thresh':    [0.0, 0.0001],
    #'smk_alpha':     [3],
    'nWords':         [128000, 64000, 8000],
    #'nWords':        [128000],
}


lnbnn2_base = {
    'pipeline_root': ['vsmany'],
    'sv_on':          [True],
    'lnbnn_weight': [0.0],
    'loglnbnn_weight': [0.0],
    'normonly_weight': [0.0],
}

lnbnn2_w1 = augbase(
    lnbnn2_base, {
        'loglnbnn_weight': [0.0, 1.0],
    })

lnbnn2_w2 = augbase(
    lnbnn2_base, {
        'normonly_weight': [0.0, 1.0],
    })

lnbnn2_w3 = augbase(
    lnbnn2_base, {
        'lnbnn_weight': [0.0, 1.0],
    })

'''
CommandLine:

python dev.py --allgt -t lnbnn2 --db PZ_Mothers --noqcache
python dev.py --allgt -t lnbnn2 --db GZ_ALL --noqcache
'''

lnbnn2 = [
    lnbnn2_w1,
    lnbnn2_w2,
    lnbnn2_w3
]


'''
SINGLE QUERY COMMANDS
python dev.py -t smk6d --db PZ_Mothers --allgt --index 0:1 --noqcache

'''


'''
TESTING COMMANDS:
python dev.py -t smk6d --db PZ_Mothers --allgt --index 20:30 --noqcache

python dev.py -t smk6 --db PZ_Mothers --allgt
python dev.py -t smk6 --db GZ_ALL --allgt

python dev.py --db GZ_ALL --vdd
python dev.py --db PZ_Mothers --vdd

# GZ_HARDCASE
python dev.py -t smk6 --db GZ_ALL --allgt --index 2 12 35 36 37
python dev.py -t smk6 --db GZ_ALL --allgt --index 2 12 35 36 37 --va

python dev.py -t smk6d --db GZ_ALL --allgt --index 2 12 --verbose --noqcache --debug2
python dev.py -t smk6d --db GZ_ALL --allgt --index 2 12 --va

python dev.py -t smk6d --db GZ_ALL --allgt --index 2 --verbose --noqcache --debug2
'''

'''
COMPUTE RESULTS COMMANDS:

python dev.py -t smk6 --db PZ_Mothers --allgt
python dev.py -t smk6 --db GZ_ALL --allgt
'''

'''
VIEW RESULTS COMMANDS:

python dev.py -t smk6 --db PZ_Mothers --allgt --va --vf
python dev.py -t smk6 --db GZ_ALL --allgt --va --vf
'''

'''
HARD CASES COMMANDS:

python dev.py -t smk6 --db PZ_Mothers --allgt --va --vf --index 2 3 4 5 6 7 8 10 13 14 15 16 17 18 19 20 21 24 25 28 31 32 34 36 37 38 39 40 43 44 45 46 47 48 49 50 51 52 53 55 56 58 59 60 63 64 65 66 69 70 73 74 76 77 78 80 81 82 83 86 88 90 91 93 94 95 96 97 98 99 101 103 104
python dev.py -t smk6 --db GZ_ALL --allgt --va --vf
'''

# Things to try:
# * negentropy
# * lower nAssign
# * more nWords
# * no multi assign weights
# * no cliphack
# * float32 rvecs
# Things to fix:
# * batched queries (possibly intermintent reporting)

smk_test = {
    'pipeline_root': ['vsmany', 'smk', 'asmk'],
    'sv_on':         [False],  # True, False],
    'smk_thresh':    [.001],
    'smk_alpha':     [3],
    'massign_alpha': [1.2],
    'massign_sigma': [80.0],
    'nAssign':       [4],
    'nWords':        [64000],  # True, False],
    'massign_equal_weights': [True],
    'vocab_weighting': ['idf', 'negentropy'],  # 'idf'
}

smk_best = {
    'pipeline_root': ['smk'],
    'sv_on':         [True],
    'smk_thresh':    [0.0],
    'smk_alpha':     [3],
    'vocab_weighting': ['idf'],
    'nAssign': [10],
}

smk_8k = smk_best.copy()
smk_8k.update({
    'nWords':        [8000],
})

smk_64k = smk_best.copy()
smk_64k.update({
    'nWords':        [64000],
})

smk_128k = smk_best.copy()
smk_128k.update({
    'nWords':        [128000],
})


# ---
# Older SMK Tests
# ---


smk_8k_compare = smk_8k.copy()
smk_8k_compare.update({
    'pipeline_root': ['smk', 'asmk',  'vsmany'],
})

# --- -------------------------------
# Vsmany - vsone tests
# ---


vsmany_2 = {
    'pipeline_root':   ['vsmany'],
    'checks':          [1024],  # , 8192],
    'K':               [5],  # 5, 10],
    'Knorm':           [1],  # 2, 3],
    'Krecip':          [0],  # , 5, 10],
    'bboxdist_weight': [0],  # 1,]
    'recip_weight':    [0],  # 1,]
    'bursty_weight':   [0],  # 1,]
    'ratio_weight':    [0, 1],  # 1,]
    'lnbnn_weight':    [0, 1],  # 1,]
    'lograt_weight':    [0, 1],  # 1,]
    'bboxdist_thresh': [None],  # .5,]
    'recip_thresh':    [0],  # 0
    'bursty_thresh':   [None],  #
    'ratio_thresh':    [None],  # 1.2, 1.6
    'lnbnn_thresh':    [None],  #
    'lograt_thresh':    [None],  #
    'nShortlist':      [50],
    'sv_on':           [True],  # True, False],
    'score_method':    ['csum'],
    'max_alts':        [1000],
}

vsone_1 = {
    'pipeline_root':   ['vsone'],
    'checks':          [256],  # , 8192],
    'K':               [1],  # 5, 10],
    'Knorm':           [1],  # 2, 3],
    'Krecip':          [0],  # , 5, 10],
    'bboxdist_weight': [0],  # 1,]
    'recip_weight':    [0],  # 1,]
    'bursty_weight':   [0],  # 1,]
    'ratio_weight':    [1],  # 1,]
    'lnbnn_weight':    [0],  # 1,]
    'lograt_weight':    [0],  # 1,]
    'bboxdist_thresh': [None],  # .5,]
    'recip_thresh':    [0],  # 0
    'bursty_thresh':   [None],  #
    'ratio_thresh':    [.6666],  # 1.2, 1.6
    'lnbnn_thresh':    [None],  #
    'lograt_thresh':    [None],  #
    'nShortlist':      [50],
    'sv_on':           [True],  # True, False],
    'score_method':    ['csum'],  # , 'pl'],  #, 'nsum', 'borda', 'topk', 'nunique']
    'max_alts':        [500],
}

vsone_std = {
    'pipeline_root':  'vsone',
    'checks':        256,
    'K':             1,
    'Knorm':         1,
    'Krecip':        0,
    'ratio_weight':  1,
    'lnbnn_weight':  0,
    'ratio_thresh':  .666,
}

vsmany_scoremethod = {
    'pipeline_root':   ['vsmany'],
    'checks':          [1024],  # , 8192],
    'K':               [5],  # 5, 10],
    'Knorm':           [1],  # 2, 3],
    'Krecip':          [0],  # , 5, 10],
    'bboxdist_weight': [0],  # 1,]
    'recip_weight':    [0],  # 1,]
    'bursty_weight':   [0],  # 1,]
    'ratio_weight':    [0],  # 1,]
    'lnbnn_weight':    [1],  # 1,]
    'lograt_weight':    [0],  # 1,]
    'bboxdist_thresh': [None],  # .5,]
    'recip_thresh':    [0],  # 0
    'bursty_thresh':   [None],  #
    'ratio_thresh':    [None],  # 1.2, 1.6
    'lnbnn_thresh':    [None],  #
    'lograt_thresh':    [None],  #
    'nShortlist':      [50],
    'sv_on':           [True],  # True, False],
    'score_method':    ['csum', 'pl', 'plw', 'borda'],  # 'bordaw', 'topk', 'topkw'],  # , 'nsum', 'borda', 'topk', 'nunique']
    'max_alts':        [1000],
}

vsmany_best = {
    'pipeline_root':   ['vsmany'],
    'checks':          [1024],  # , 8192],
    'K':               [4],  # 5, 10],
    'Knorm':           [1],  # 2, 3],
    'Krecip':          [0],  # , 5, 10],
    'bboxdist_weight': [0],  # 1,]
    'recip_weight':    [0],  # 1,]
    'bursty_weight':   [0],  # 1,]
    'ratio_weight':    [0],  # 1,]
    'lnbnn_weight':    [1],  # 1,]
    'lograt_weight':    [0],  # 1,]
    'bboxdist_thresh': [None],  # .5,]
    'recip_thresh':    [0],  # 0
    'bursty_thresh':   [None],  #
    'ratio_thresh':    [None],  # 1.2, 1.6
    'lnbnn_thresh':    [None],  #
    'lograt_thresh':    [None],  #
    'xy_thresh':       [.01],  # [.002],
    'nShortlist':      [50],
    'sv_on':           [True],  # True, False],
    'prescore_method': ['csum'],  # 'bordaw', 'topk', 'topkw'],  # , 'nsum', 'borda', 'topk', 'nunique']
    'score_method': ['csum'],  # 'bordaw', 'topk', 'topkw'],  # , 'nsum', 'borda', 'topk', 'nunique']
    'max_alts':        [1000],
    'chip_sqrt_area':  [450],
}
best = vsmany_best

gv_test = vsmany_best.copy()
gv_test.update({
    'nogravity_hack': [False, True],
    'gravity_weighting': [False, True],
    #'use_adaptive_scale': [True, False],
})


# 450 works the best on GZ
chipsize_test = vsmany_best.copy()
chipsize_test.update({
    'chip_sqrt_area':  [400, 450, 500, 600, 750],
})


shortlist_test = vsmany_best.copy()
shortlist_test.update({
    'nShortlist':  [50, 100, 500, 1000],
})

coverage = vsmany_best.copy()
coverage.update({
    'score_method': ['csum', 'coverage', 'coverage1', 'coverage2', 'coverage3'],
})

adaptive_test = vsmany_best.copy()
adaptive_test.update({
    'K': [2, 5],
    'use_adaptive_scale': [True, False],
})

overnight = vsmany_best.copy()
overnight.update({
    #'K':                   [5, 7, 10, 20],
    #'Knorm':               [1, 3, 5, 10],
    #'normalizer_rule':     ['name', 'last'],
    #'use_adaptive_scale':  [True, False],
    #'score_method':        ['pl', 'csum'],  # , 'pl'],  #, 'nsum', 'borda', 'topk', 'nunique']
})

overnight_huge = vsmany_best.copy()
overnight_huge.update({
    'K':                   [2, 5, 7, 10],
    'Knorm':               [1, 3, 5],
    'normalizer_rule':     ['name', 'last'],
    'use_adaptive_scale':  [False, True],
    #'scale_min': [0, 10, 20],
    #'score_method':        ['pl', 'csum'],  # , 'pl'],  #, 'nsum', 'borda', 'topk', 'nunique']
})


overnight_k = vsmany_best.copy()
overnight_k.update({
    'K': [1, 3, 5, 7, 10, 20],
    'Knorm': [1, 3, 5, 7, 10],
})

k_small = vsmany_best.copy()
k_small.update({
    'K': [2, 5],
    'Knorm': [1],
})

k_big = vsmany_best.copy()
k_big.update({
    'K': [3, 4, 5, 7, 10, 15],
    'Knorm': [1, 2, 3],
})

normrule = vsmany_best.copy()
normrule.update({
    'K': [5, 10, 3],
    'Knorm': [1, 3, 5, 10, 20],
    'normalizer_rule': ['name', 'last'],
})

normrule2 = vsmany_best.copy()
normrule2.update({
    'K': [5, 3],
    'Knorm': [1, 3],
    'normalizer_rule': ['name', 'last'],
})

small_scale_test = small_scale = vsmany_best.copy()
small_scale_test.update({
    'scale_min': [0, 20],
    'scale_max': [30, 9001],
})


scale_test = vsmany_best.copy()
scale_test.update({
    'scale_min': [0, 10, 20],
    'scale_max': [30, 100, 9001],
    #'use_adaptive_scale': [True, False],
})


vsmany_score = vsmany_best.copy()
vsmany_score.update({
    'score_method': ['csum', 'pl', 'plw', 'borda', 'bordaw'],
})

vsmany_nosv = vsmany_best.copy()
vsmany_nosv.update({
    'sv_on': [False]
})

vsmany_sv = vsmany_best.copy()
vsmany_sv.update({
    'xy_thresh': [None, .1, .01, .001, .002]
})

vsmany_k = vsmany_best.copy()
vsmany_k.update({
    'K': [2, 5, 10, 30]
})


vsmany_big_social = vsmany_best.copy()
vsmany_big_social.update({
    'K':                [5, 10, 20],
    'Knorm':            [1, 3],
    'Krecip':           [0],
    'lnbnn_weight':     [1],
    'score_method':     ['pl', 'plw', 'csum', 'borda', 'bordaw'],
})


vsmany_1 = {
    'pipeline_root':      ['vsmany'],
    'checks':          [1024],  # , 8192],
    'K':               [5],  # 5, 10],
    'Knorm':           [1],  # 2, 3],
    'Krecip':          [0],  # , 5, 10],
    'bboxdist_weight':  [0],  # 1,]
    'recip_weight':    [0],  # 1,]
    'bursty_weight':   [0],  # 1,]
    'ratio_weight':    [0],  # 1,]
    'lnbnn_weight':    [1],  # 1,]
    'lograt_weight':    [0],  # 1,]
    'bboxdist_thresh':  [None],  # .5,]
    'recip_thresh':    [0],  # 0
    'bursty_thresh':   [None],  #
    'ratio_thresh':    [None],  # 1.2, 1.6
    'lnbnn_thresh':    [None],  #
    'lograt_thresh':   [None],  #
    'nShortlist':      [50],
    'sv_on':           [True],  # True, False],
    'score_method':    ['csum'],  # , 'nsum', 'borda', 'topk', 'nunique']
    'max_alts':        [1000],
}

# Special value used to specify the current IBEIS configuration
custom = 'custom'


include_vars = vars().keys()  # this line is after tests

# List of all valid tests
TEST_NAMES = set(include_vars) - set(exclude_vars)
