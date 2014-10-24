from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip, range, map  # NOQA
# UTool
import utool
import vtool.patch as ptool
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
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aids_list = ibs.get_valid_aids()[0:1]
        >>> chip_list = ibs.get_annot_chips(aids_list)
        >>> kpts_list = ibs.get_annot_kpts(aids_list)
        >>> probchip_fpath = 'something'
        >>> kpts = kpts_list[0]
        >>> aids = aids_list[0]
        >>> chip = chip_list[0]
        >>> probchip_img = chip
        >>> #probchip_img = gtool.imread(probchip_fpath)

    """
    (aid, kpts, probchip_img) = tup
    #ptool.get_warped_patches()
    patch_list = [ptool.get_warped_patch(probchip_img, kp)[0] for kp in kpts]
    weight_list = [patch.sum() / patch.size for patch in patch_list]
    weights = np.array(weight_list, dtype=np.float32)
    return (aid, weights)
