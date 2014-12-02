from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip, range, map  # NOQA
# UTool
import utool
import utool as ut
import vtool.patch as vtpatch
import vtool.image as vtimage  # NOQA
#import vtool.image as vtimage
import numpy as np
from ibeis.model.preproc import preproc_probchip
from os.path import exists
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_featweight]')


def gen_featweight_worker(tup):
    """
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        tup (aid, tuple(kpts(ndarray), probchip_fpath )): keypoints and probability chip file path

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> ax = 2
        >>> aid_list = ibs.get_valid_aids()[ax:ax + 1]
        >>> chip_list = ibs.get_annot_chips(aid_list)
        >>> kpts_list = ibs.get_annot_kpts(aid_list)
        >>> probchip_fpath_list = preproc_probchip.compute_and_write_probchip(ibs, aid_list)
        >>> probchip_list = [vtimage.imread(fpath, grayscale=False) if exists(fpath) else None for fpath in probchip_fpath_list]
        >>> kpts  = kpts_list[0]
        >>> aid   = aid_list[0]
        >>> probchip = probchip_list[0]
        >>> tup = (aid, kpts, probchip)
        >>> (aid, weights) = gen_featweight_worker(tup)
        >>> result = str((aid, np.array_str(weights[0:3], precision=3)))
        >>> print(result)
        (3, '[ 0.098  0.155  0.422]')

    """
    (aid, kpts, probchip) = tup
    if probchip is None:
        # hack for undetected chips. SETS ALL FEATWEIGHTS TO .25 = 1/4
        weights = np.full(len(kpts), .25, dtype=np.float32)
    else:
        #vtpatch.get_warped_patches()
        patch_list = [vtpatch.get_warped_patch(probchip, kp)[0].astype(np.float32) / 255.0 for kp in kpts]
        weight_list = [patch.sum() / (patch.size) for patch in patch_list]
        weights = np.array(weight_list, dtype=np.float32)
    return (aid, weights)


def compute_fgweights(ibs, aid_list, qreq_=None):
    """

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[1:2]
        >>> qreq_ = None
        >>> featweight_list = compute_fgweights(ibs, aid_list)
        >>> result = np.array_str(featweight_list[0][0:3], precision=3)
        >>> print(result)
        [ 0.125  0.061  0.053]

    """
    print('[preproc_featweight] Preparing to compute fgweights')
    probchip_fpath_list = preproc_probchip.compute_and_write_probchip(ibs, aid_list, qreq_=qreq_)
    if ut.DEBUG2:
        from PIL import Image
        probchip_size_list = [Image.open(fpath).size for fpath in probchip_fpath_list]
        chipsize_list = ibs.get_annot_chipsizes(aid_list)
        assert chipsize_list == probchip_size_list, 'probably need to clear chip or probchip cache'

    kpts_list = ibs.get_annot_kpts(aid_list)
    probchip_list = [vtimage.imread(fpath) if exists(fpath) else None for fpath in probchip_fpath_list]

    print('[preproc_featweight] Computing fgweights')
    arg_iter = zip(aid_list, kpts_list, probchip_list)
    featweight_gen = utool.util_parallel.generate(gen_featweight_worker, arg_iter, nTasks=len(aid_list))
    featweight_param_list = list(featweight_gen)
    #arg_iter = zip(aid_list, kpts_list, probchip_list)
    #featweight_param_list1 = [gen_featweight_worker((aid, kpts, probchip)) for aid, kpts, probchip in arg_iter]
    #featweight_aids = ut.get_list_column(featweight_param_list, 0)
    featweight_list = ut.get_list_column(featweight_param_list, 1)
    print('[preproc_featweight] Done computing fgweights')
    return featweight_list


def add_featweight_params_gen(ibs, fid_list, qreq_=None):
    """
    add_featweight_params_gen

    DEPRICATE

    Args:
        ibs (IBEISController):
        fid_list (list):

    Returns:
        featweight_list

    Example:
        >>> # DEPRICATE
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> fid_list = ibs.get_valid_fids()
        >>> result = add_featweight_params_gen(ibs, fid_list)
        >>> print(result)
    """
    # HACK: TODO AUTOGENERATE THIS
    from ibeis import constants
    cid_list = ibs.dbcache.get(constants.FEATURE_TABLE, ('chip_rowid',), fid_list)
    aid_list = ibs.dbcache.get(constants.CHIP_TABLE, ('annot_rowid',), cid_list)
    featweight_list = compute_fgweights(ibs, aid_list, qreq_=qreq_)
    return featweight_list


def generate_featweight_properties(ibs, fid_list, qreq_=None):
    """
    Args:
        ibs (IBEISController):
        fid_list (list):

    Returns:
        featweight_list

    CommandLine:
        python -m ibeis.model.preproc.preproc_featweight --test-generate_featweight_properties

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[1:2]
        >>> fid_list = ibs.get_annot_feat_rowids(aid_list, ensure=True)
        >>> #fid_list = ibs.get_valid_fids()[1:2]
        >>> featweighttup_gen = generate_featweight_properties(ibs, fid_list)
        >>> featweighttup_list = list(featweighttup_gen)
        >>> featweight_list = featweighttup_list[0][0]
        >>> result = np.array_str(featweight_list[0:3], precision=3)
        >>> print(result)
        [ 0.125  0.061  0.053]
    """
    # HACK: TODO AUTOGENERATE THIS
    from ibeis import constants
    #cid_list = ibs.get_feat_cids(fid_list)
    #aid_list = ibs.get_chip_aids(cid_list)
    cid_list = ibs.dbcache.get(constants.FEATURE_TABLE, ('chip_rowid',), fid_list)
    aid_list = ibs.dbcache.get(constants.CHIP_TABLE, ('annot_rowid',), cid_list)
    featweight_list = compute_fgweights(ibs, aid_list, qreq_=qreq_)
    return zip(featweight_list)


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
#    cfpath_list = ibs.get_annot_chip_fpaths(aid_list)
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


def on_delete(ibs, featweight_rowid_list):
    print('Warning: Not Implemented')
    print('TODO: Delete probability chips, or should that be its own preproc?')

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.preproc.preproc_featweight
        python -m ibeis.model.preproc.preproc_featweight --allexamples
        python -m ibeis.model.preproc.preproc_featweight --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
