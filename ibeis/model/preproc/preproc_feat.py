from __future__ import absolute_import, division, print_function
# Python
from itertools import izip
# Science
import pyhesaff
# UTool
import utool
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[preproc_feat]', DEBUG=False)


USE_OPENMP = not utool.WIN32


def gen_feat_worker(tup):
    """
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async
    """
    cid, cpath, dict_args, feat_config_uid = tup
    kpts, desc = pyhesaff.detect_kpts(cpath, **dict_args)
    return cid, len(kpts), kpts, desc, feat_config_uid


def gen_feat_openmp(cid_list, cfpath_list, dict_args, feat_config_uid):
    """ Compute features in parallel on the C++ side, return generator here """
    print('Detecting %r features in parallel: ' % len(cid_list))
    kpts_list, desc_list = pyhesaff.detect_kpts_list(cfpath_list, **dict_args)
    for cid, kpts, desc in izip(cid_list, kpts_list, desc_list):
        yield cid, len(kpts), kpts, desc, feat_config_uid


def add_feat_params_gen(ibs, cid_list, nFeat=None):
    """ Computes features and yields results asynchronously """
    if nFeat is None:
        nFeat = len(cid_list)
    feat_cfg  = ibs.cfg.feat_cfg
    dict_args = feat_cfg.get_dict_args()
    feat_config_uid = ibs.get_feat_config_uid()
    cfpath_list = ibs.get_chip_paths(cid_list)
    if USE_OPENMP:
        # Use Avi's openmp parallelization
        return gen_feat_openmp(cid_list, cfpath_list, dict_args, feat_config_uid)
    else:
        # Multiprocessing parallelization
        featcfg_iter = (feat_config_uid for _ in xrange(nFeat))
        dictargs_iter = (dict_args for _ in xrange(nFeat))
        arg_iter = izip(cid_list, cfpath_list, dictargs_iter, featcfg_iter)
        arg_list = list(arg_iter)
        return utool.util_parallel.generate(gen_feat_worker, arg_list)
