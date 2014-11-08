from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip, range
# Science
import pyhesaff
# UTool
import utool


# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_feat]')


USE_OPENMP = not utool.WIN32
USE_OPENMP = False  # do not use openmp until we have the gravity vector


def gen_feat_worker(tup):
    """
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Cyth::
        cdef:
            long cid
            str cpath
            dict dict_args
            np.ndarray[kpts_t, ndims=2] kpts
            np.ndarray[desc_t, ndims=2] desc
    """
    cid, cpath, dict_args = tup
    kpts, desc = pyhesaff.detect_kpts(cpath, **dict_args)
    return (cid, len(kpts), kpts, desc)


def gen_feat_openmp(cid_list, cfpath_list, dict_args):
    """ Compute features in parallel on the C++ side, return generator here """
    print('Detecting %r features in parallel: ' % len(cid_list))
    kpts_list, desc_list = pyhesaff.detect_kpts_list(cfpath_list, **dict_args)
    for cid, kpts, desc in zip(cid_list, kpts_list, desc_list):
        yield cid, len(kpts), kpts, desc


def add_feat_params_gen(ibs, cid_list, qreq_=None, nInput=None):
    """
    Computes features and yields results asynchronously: TODO: Remove IBEIS from
    this equation. Move the firewall towards the controller

    Args:
        ibs (IBEISController):
        cid_list (list):
        nInput (None):

    Returns:
        generator : generates param tups

    Example:
        >>> from ibeis.model.preproc.preproc_feat import *  # NOQA
    """
    if nInput is None:
        nInput = len(cid_list)
    # Get config from IBEIS controller
    feat_cfg          = ibs.cfg.feat_cfg
    dict_args         = feat_cfg.get_dict_args()
    feat_config_rowid = ibs.get_feat_config_rowid()
    cfpath_list       = ibs.get_chip_paths(cid_list)
    print('[preproc_feat] cfgstr = %s' % feat_cfg.get_cfgstr())
    if USE_OPENMP:
        # Use Avi's openmp parallelization
        return gen_feat_openmp(cid_list, cfpath_list, dict_args, feat_config_rowid)
    else:
        # Multiprocessing parallelization
        featgen = generate_feats(cfpath_list, dict_args=dict_args,
                                 cid_list=cid_list, nInput=nInput)
        return ((cid, nKpts, kpts, desc, feat_config_rowid) for cid, nKpts, kpts, desc in featgen)


def generate_feats(cfpath_list, dict_args={}, cid_list=None, nInput=None, **kwargs):
    # chip-ids are an artifact of the IBEIS Controller. Make dummyones if needbe.
    """ Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        cfpath_list (list):
        dict_args (dict):
        cid_list (list):
        nInput (None):

    Kwargs:
        passed to utool.generate

    Returns:
        featgen

    Example:
        >>> from ibeis.model.preproc.preproc_feat import *  # NOQA

    Cyth:
        cdef:
            list cfpath_list
            long nInput
            object cid_list
            dict dict_args
            dict kwargs
    """
    if cid_list is None:
        cid_list = list(range(len(cfpath_list)))
    if nInput is None:
        nInput = len(cfpath_list)
    dictargs_iter = (dict_args for _ in range(nInput))
    arg_iter = zip(cid_list, cfpath_list, dictargs_iter)
    # eager evaluation.
    # TODO: see if we can just pass in the iterator or if there is benefit in
    # doing so
    arg_list = list(arg_iter)
    featgen = utool.util_parallel.generate(gen_feat_worker, arg_list, nTasks=nInput, **kwargs)
    return featgen


def on_delete(ibs, gid_list, qreq_=None):
    print('Warning: Not Implemented')
