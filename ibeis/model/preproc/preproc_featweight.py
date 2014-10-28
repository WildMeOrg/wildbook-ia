"""
# DOCTEST ENABLED
DoctestCMD:
    python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.preproc.preproc_featweight))" --quiet
"""
from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip, range, map  # NOQA
# UTool
import utool
import utool as ut
import vtool.patch as ptool
import vtool.image as gtool  # NOQA
#import vtool.image as gtool
import numpy as np
from ibeis.model.preproc import preproc_chip
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_featweight]')


def gen_featweight_worker(tup):
    """
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        tup (aid, tuple(kpts(ndarray), probchip_fpath )): keypoints and probability chip file path

    Example:
        >>> # DOCTEST ENABLE
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:1]
        >>> chip_list = ibs.get_annot_chips(aid_list)
        >>> kpts_list = ibs.get_annot_kpts(aid_list)
        >>> probchip_fpath_list = preproc_chip.compute_and_write_probchip(ibs, aid_list)
        >>> probchip_list = [gtool.imread(fpath, grayscale=False) for fpath in probchip_fpath_list]
        >>> kpts     = kpts_list[0]
        >>> aids     = aid_list[0]
        >>> probchip = probchip_list[0]
        >>> tup = (aid, kpts, probchip)
        >>> (aid, weights) = gen_featweight_worker(tup)
        >>> print(weights.sum())
        275.025

    """
    (aid, kpts, probchip) = tup
    #ptool.get_warped_patches()
    patch_list = [ptool.get_warped_patch(probchip, kp)[0].astype(np.float32) / 255.0 for kp in kpts]
    weight_list = [patch.sum() / (patch.size) for patch in patch_list]
    weights = np.array(weight_list, dtype=np.float32)
    return (aid, weights)


def compute_featweights(ibs, aid_list):
    """

    Example:
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> featweight_list = compute_featweights(ibs, aid_list)
    """

    probchip_fpath_list = preproc_chip.compute_and_write_probchip(ibs, aid_list)
    if ut.DEBUG2:
        from PIL import Image
        probchip_size_list = [Image.open(fpath).size for fpath in probchip_fpath_list]
        chipsize_list = ibs.get_annot_chipsizes(aid_list)
        assert chipsize_list == probchip_size_list, 'probably need to clear chip or probchip cache'

    kpts_list = ibs.get_annot_kpts(aid_list)
    probchip_list = [gtool.imread(fpath) for fpath in probchip_fpath_list]

    arg_iter = zip(aid_list, kpts_list, probchip_list)
    featweight_gen = utool.util_parallel.generate(gen_featweight_worker, arg_iter, nTasks=len(aid_list))
    featweight_param_list = list(featweight_gen)
    #arg_iter = zip(aid_list, kpts_list, probchip_list)
    #featweight_param_list1 = [gen_featweight_worker((aid, kpts, probchip)) for aid, kpts, probchip in arg_iter]
    #featweight_aids = ut.get_list_column(featweight_param_list, 0)
    featweight_list = ut.get_list_column(featweight_param_list, 1)
    return featweight_list


#def get_annot_probchip_fname_iter(ibs, aid_list):
#    """ Returns probability chip path iterator

#    Args:
#        ibs (IBEISController):
#        aid_list (list):

#    Returns:
#        probchip_fname_iter

#    Example:
#        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
#        >>> import ibeis
#        >>> ibs = ibeis.opendb('testdb1')
#        >>> aid_list = ibs.get_valid_aids()
#        >>> probchip_fname_iter = get_annot_probchip_fname_iter(ibs, aid_list)
#        >>> probchip_fname_list = list(probchip_fname_iter)
#    """
#    cfpath_list = ibs.get_annot_cpaths(aid_list)
#    cfname_list = [splitext(basename(cfpath))[0] for cfpath in cfpath_list]
#    suffix = ibs.cfg.detect_cfg.get_cfgstr()
#    ext = '.png'
#    probchip_fname_iter = (''.join([cfname, suffix, ext]) for cfname in cfname_list)
#    return probchip_fname_iter


#def get_annot_probchip_fpath_list(ibs, aid_list):
#    cachedir = get_probchip_cachedir(ibs)
#    probchip_fname_list = get_annot_probchip_fname_iter(ibs, aid_list)
#    probchip_fpath_list = [join(cachedir, fname) for fname in probchip_fname_list]
#    return probchip_fpath_list


#class FeatWeightConfig(object):
#    # TODO: Put this in a config
#    def __init__(fw_cfg):
#        fw_cfg.sqrt_area   = 800
