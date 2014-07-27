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
USE_OPENMP = False  # do not use openmp until we have the gravity vector


def gen_feat_worker(tup):
    """
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async
    """
    cid, cpath, dict_args = tup
    kpts, desc = pyhesaff.detect_kpts(cpath, **dict_args)
    return cid, len(kpts), kpts, desc


def gen_feat_openmp(cid_list, cfpath_list, dict_args):
    """ Compute features in parallel on the C++ side, return generator here """
    print('Detecting %r features in parallel: ' % len(cid_list))
    kpts_list, desc_list = pyhesaff.detect_kpts_list(cfpath_list, **dict_args)
    for cid, kpts, desc in izip(cid_list, kpts_list, desc_list):
        yield cid, len(kpts), kpts, desc


def add_feat_params_gen(ibs, cid_list, nFeat=None, **kwargs):
    """ Computes features and yields results asynchronously:
        TODO: Remove IBEIS from this equation. Move the firewall towards the
        controller """
    if nFeat is None:
        nFeat = len(cid_list)
    # Get config from IBEIS controller
    feat_cfg          = ibs.cfg.feat_cfg
    dict_args         = feat_cfg.get_dict_args()
    feat_config_rowid = ibs.get_feat_config_rowid()
    cfpath_list       = ibs.get_chip_paths(cid_list)
    if USE_OPENMP:
        # Use Avi's openmp parallelization
        return gen_feat_openmp(cid_list, cfpath_list, dict_args, feat_config_rowid)
    else:
        # Multiprocessing parallelization
        featgen = generate_feats(cfpath_list, dict_args=dict_args,
                                 cid_list=cid_list, nFeat=nFeat, **kwargs)
        return ((cid, nKpts, kpts, desc, feat_config_rowid)
                for cid, nKpts, kpts, desc in featgen)


def generate_feats(cfpath_list, dict_args={}, cid_list=None, nFeat=None, **kwargs):
    # chip-ids are an artifact of the IBEIS Controller. Make dummyones if needbe.
    if cid_list is None:
        cid_list = range(len(cfpath_list))
    if nFeat is None:
        nFeat = len(cfpath_list)
    dictargs_iter = (dict_args for _ in xrange(nFeat))
    arg_iter = izip(cid_list, cfpath_list, dictargs_iter)
    arg_list = list(arg_iter)
    featgen = utool.util_parallel.generate(gen_feat_worker, arg_list, **kwargs)
    return featgen
