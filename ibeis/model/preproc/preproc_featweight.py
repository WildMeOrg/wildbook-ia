from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip, range, map  # NOQA
# UTool
from os.path import join, basename, splitext
import utool
import utool as ut
import vtool.patch as ptool
import vtool.image as gtool  # NOQA
#import vtool.image as gtool
import numpy as np
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_featweight]')


def gen_featweight_worker(tup):
    """
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        tup (aid, tuple(kpts(ndarray), probchip_fpath )): keypoints and probability chip file path

    Example:
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aids_list = ibs.get_valid_aids()[0:1]
        >>> chip_list = ibs.get_annot_chips(aids_list)
        >>> kpts_list = ibs.get_annot_kpts(aids_list)
        >>> probchip_fpath = 'something'
        >>> kpts = kpts_list[0]
        >>> aids = aids_list[0]
        >>> chip = chip_list[0]
        >>> probchip = chip
        >>> #probchip = gtool.imread(probchip_fpath)

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
    """

    probchip_fpath_list = compute_and_write_probchip(ibs, aid_list)
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


def get_probchip_cachedir(ibs):
    return join(ibs.get_cachedir(), 'probchip')


def get_annot_probchip_fname_iter(ibs, aid_list):
    """ Returns probability chip path iterator

    Args:
        ibs (IBEISController):
        aid_list (list):

    Returns:
        probchip_fname_iter

    Example:
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> probchip_fname_iter = get_annot_probchip_fname_iter(ibs, aid_list)
        >>> probchip_fname_list = list(probchip_fname_iter)
    """
    cfpath_list = ibs.get_annot_cpaths(aid_list)
    cfname_list = [splitext(basename(cfpath))[0] for cfpath in cfpath_list]
    suffix = ibs.cfg.detect_cfg.get_cfgstr()
    ext = '.png'
    probchip_fname_iter = (''.join([cfname, suffix, ext]) for cfname in cfname_list)
    return probchip_fname_iter


def get_annot_probchip_fpath_list(ibs, aid_list):
    cachedir = get_probchip_cachedir(ibs)
    probchip_fname_list = get_annot_probchip_fname_iter(ibs, aid_list)
    probchip_fpath_list = [join(cachedir, fname) for fname in probchip_fname_list]
    return probchip_fpath_list


#class FeatWeightConfig(object):
#    # TODO: Put this in a config
#    def __init__(fw_cfg):
#        fw_cfg.sqrt_area   = 800


def compute_and_write_probchip(ibs, aid_list):
    """

    Example:
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
    """
    # Get probchip dest information (output path)
    from ibeis.model.detect import randomforest
    species = ibs.cfg.detect_cfg.species
    use_chunks = ibs.cfg.other_cfg.detect_use_chunks
    cachedir = get_probchip_cachedir(ibs)
    utool.ensuredir(cachedir)
    probchip_fpath_list = get_annot_probchip_fpath_list(ibs, aid_list)
    # Get img configuration information
    # Get img source information (image, annotation_bbox, theta)
    cfpath_list  = ibs.get_annot_cpaths(aid_list)
    # Define "Asynchronous" generator
    randomforest.compute_hough_images(cfpath_list, probchip_fpath_list, species, use_chunks=use_chunks)
    # Fix stupid bug in pyrf
    probchip_fpath_list_ = [fpath + '.png' for fpath in probchip_fpath_list]
    return probchip_fpath_list_
    print('[preproc_probchip] Done computing probability images')
