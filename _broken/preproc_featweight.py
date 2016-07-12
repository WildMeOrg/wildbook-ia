# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
# Python
from six.moves import zip, range, map  # NOQA
# UTool
import utool as ut
import vtool as vt
#import vtool.image as vtimage
import numpy as np
from ibeis.algo.preproc import preproc_probchip
from os.path import exists
# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[preproc_featweight]')


def test_featweight_worker():
    """
    test function

    python -m ibeis.algo.preproc.preproc_featweight --test-gen_featweight_worker --show --cnn
    """
    import ibeis
    qreq_ = ibeis.main_helpers.testdata_qreq_(defaultdb='PZ_MTEST', p=['default:fw_detector=cnn'], qaid_override=[1])
    ibs = qreq_.ibs
    config2_ = qreq_.qparams
    lazy = True
    aid_list            = qreq_.qaids
    #aid_list = ibs.get_valid_aids()[0:30]
    kpts_list           = ibs.get_annot_kpts(aid_list)
    chipsize_list       = ibs.get_annot_chip_sizes(aid_list, config2_=config2_)
    probchip_fpath_list = preproc_probchip.compute_and_write_probchip(ibs,
                                                                      aid_list,
                                                                      lazy=lazy,
                                                                      config2_=config2_)
    print('probchip_fpath_list = %r' % (probchip_fpath_list,))
    probchip_list       = [vt.imread(fpath, grayscale=True) if exists(fpath) else None
                           for fpath in probchip_fpath_list]

    _iter = list(zip(aid_list, kpts_list, probchip_list, chipsize_list))
    _iter = ut.InteractiveIter(_iter, enabled=ut.get_argflag('--show'))
    for aid, kpts, probchip, chipsize in _iter:
        #kpts     = kpts_list[0]
        #aid      = aid_list[0]
        #probchip = probchip_list[0]
        #chipsize = chipsize_list[0]
        tup = (aid, kpts, probchip, chipsize)
        (aid, weights) = gen_featweight_worker(tup)
        if aid == 3 and ibs.get_dbname() == 'testdb1':
            # Run Asserts if not interactive
            weights_03_test = weights[0:3]
            print('weights[0:3] = %r' % (weights_03_test,))
            #weights_03_target = [ 0.098, 0.155,  0.422]
            #weights_03_target = [ 0.324, 0.407,  0.688]
            #weights_thresh    = [ 0.09, 0.09,  0.09]
            #ut.assert_almost_eq(weights_03_test, weights_03_target, weights_thresh)
            ut.assert_inbounds(weights_03_test, 0, 1)
            if not ut.show_was_requested():
                break
        if ut.show_was_requested():
            import plottool as pt
            #sfx, sfy = (probchip.shape[1] / chipsize[0], probchip.shape[0] / chipsize[1])
            #kpts_ = vt.offset_kpts(kpts, (0, 0), (sfx, sfy))
            pnum_ = pt.make_pnum_nextgen(1, 3)  # *pt.get_square_row_cols(4))
            fnum = 1
            pt.figure(fnum=fnum, doclf=True)
            ###
            pt.imshow(ibs.get_annot_chips(aid, config2_=config2_), pnum=pnum_(0), fnum=fnum)
            if ut.get_argflag('--numlbl'):
                pt.gca().set_xlabel('(1)')
            ###
            pt.imshow(probchip, pnum=pnum_(2), fnum=fnum)
            if ut.get_argflag('--numlbl'):
                pt.gca().set_xlabel('(2)')
            #pt.draw_kpts2(kpts_, ell_alpha=.4, color_list=pt.ORANGE)
            ###
            #pt.imshow(probchip, pnum=pnum_(3), fnum=fnum)
            #color_list = pt.draw_kpts2(kpts_, weights=weights, ell_alpha=.7, cmap_='jet')
            #cb = pt.colorbar(weights, color_list)
            #cb.set_label('featweights')
            ###
            pt.imshow(ibs.get_annot_chips(aid, config2_=qreq_.qparams), pnum=pnum_(1), fnum=fnum)
            #color_list = pt.draw_kpts2(kpts, weights=weights, ell_alpha=.3, cmap_='jet')
            color_list = pt.draw_kpts2(kpts, weights=weights, ell_alpha=.3)
            cb = pt.colorbar(weights, color_list)
            cb.set_label('featweights')
            if ut.get_argflag('--numlbl'):
                pt.gca().set_xlabel('(3)')
            #pt.draw_kpts2(kpts, ell_alpha=.4)
            pt.draw()
            pt.show_if_requested()


def gen_featweight_worker(tup):
    """
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        tup (aid, tuple(kpts(ndarray), probchip_fpath )): keypoints and probability chip file path
           aid, kpts, probchip_fpath

    CommandLine:
        python -m ibeis.algo.preproc.preproc_featweight --test-gen_featweight_worker --show
        python -m ibeis.algo.preproc.preproc_featweight --test-gen_featweight_worker --show --dpath figures --save ~/latex/crall-candidacy-2015/figures/gen_featweight.jpg
        python -m ibeis.algo.preproc.preproc_featweight --test-gen_featweight_worker --show --db PZ_MTEST --qaid_list=1,2,3,4,5,6,7,8,9

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_featweight import *  # NOQA
        >>> test_featweight_worker()

    Ignore::
        import plottool as pt
        pt.imshow(probchip_list[0])
        patch_list = [vt.patch.get_warped_patch(probchip, kp)[0].astype(np.float32) / 255.0 for kp in kpts[0:1]]
        patch_ = patch_list[0].copy()
        patch = patch_
        patch = patch_[-20:, :20, 0]

        import vtool as vt
        gaussian_patch = vt.gaussian_patch(patch.shape[1], patch.shape[0], shape=patch.shape[0:2], norm_01=False)

        import cv2
        sigma = 1/10
        xkernel = (cv2.getGaussianKernel(patch.shape[1], sigma))
        ykernel = (cv2.getGaussianKernel(patch.shape[0], sigma))

        #ykernel = ykernel / ykernel.max()
        #xkernel = ykernel / xkernel.max()

        gaussian_kern2 = ykernel.dot(xkernel.T)
        print(gaussian_kern2.sum())

        patch2 = patch.copy()
        patch2 = np.multiply(patch2,   ykernel)
        patch2 = np.multiply(patch2.T, xkernel).T

        if len(patch3.shape) == 2:
            patch3 = patch.copy() * gaussian_patch[:,:]
        else:
            patch3 = patch.copy() * gaussian_patch[:,:, None]

        sum2 = patch2.sum() / (patch2.size)
        sum3 = patch3.sum() / (patch3.size)

        print(sum2)
        print(sum3)

        fig = pt.figure(fnum=1, pnum=(1, 3, 1), doclf=True, docla=True)
        pt.imshow(patch * 255)
        fig = pt.figure(fnum=1, pnum=(1, 3, 2))
        pt.imshow(gaussian_kern2 * 255.0)
        fig = pt.figure(fnum=1, pnum=(1, 3, 3))
        pt.imshow(patch2 * 255.0)
        pt.update()
    """
    (aid, kpts, probchip, chipsize) = tup
    if probchip is None:
        # hack for undetected chips. SETS ALL FEATWEIGHTS TO .25 = 1/4
        weights = np.full(len(kpts), .25, dtype=np.float32)
    else:
        sfx, sfy = (probchip.shape[1] / chipsize[0], probchip.shape[0] / chipsize[1])
        kpts_ = vt.offset_kpts(kpts, (0, 0), (sfx, sfy))
        #vt.patch.get_warped_patches()
        patch_list  = [vt.patch.get_warped_patch(probchip, kp)[0].astype(np.float32) / 255.0
                       for kp in kpts_]
        weight_list = [vt.patch.gaussian_average_patch(patch) for patch in patch_list]
        #weight_list = [patch.sum() / (patch.size) for patch in patch_list]
        weights = np.array(weight_list, dtype=np.float32)
    return (aid, weights)


def compute_fgweights(ibs, aid_list, config2_=None):
    """

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.algo.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[1:2]
        >>> config2_ = None
        >>> featweight_list = compute_fgweights(ibs, aid_list)
        >>> result = np.array_str(featweight_list[0][0:3], precision=3)
        >>> print(result)
        [ 0.125  0.061  0.053]

    """
    nTasks = len(aid_list)
    print('[preproc_featweight.compute_fgweights] Preparing to compute %d fgweights' % (nTasks,))
    probchip_fpath_list = preproc_probchip.compute_and_write_probchip(ibs,
                                                                      aid_list,
                                                                      config2_=config2_)
    chipsize_list = ibs.get_annot_chip_sizes(aid_list, config2_=config2_)

    #if ut.DEBUG2:
    #    from PIL import Image
    #    probchip_size_list = [Image.open(fpath).size for fpath in probchip_fpath_list]  # NOQA
    #    #with ut.embed_on_exception_context:
    #    # does not need to happen anymore
    #    assert chipsize_list == probchip_size_list, 'probably need to clear chip or probchip cache'

    kpts_list = ibs.get_annot_kpts(aid_list, config2_=config2_)
    # Force grayscale reading of chips
    probchip_list = [vt.imread(fpath, grayscale=True) if exists(fpath) else None
                     for fpath in probchip_fpath_list]

    print('[preproc_featweight.compute_fgweights] Computing %d fgweights' % (nTasks,))
    arg_iter = zip(aid_list, kpts_list, probchip_list, chipsize_list)
    featweight_gen = ut.generate(gen_featweight_worker, arg_iter,
                                    nTasks=nTasks, ordered=True, freq=10)
    featweight_param_list = list(featweight_gen)
    #arg_iter = zip(aid_list, kpts_list, probchip_list)
    #featweight_param_list1 = [gen_featweight_worker((aid, kpts, probchip)) for
    #aid, kpts, probchip in arg_iter]
    #featweight_aids = ut.get_list_column(featweight_param_list, 0)
    featweight_list = ut.get_list_column(featweight_param_list, 1)
    print('[preproc_featweight.compute_fgweights] Done computing %d fgweights' % (nTasks,))
    return featweight_list


def generate_featweight_properties(ibs, feat_rowid_list, config2_=None):
    """
    Args:
        ibs (IBEISController):
        fid_list (list):

    Returns:
        featweight_list

    CommandLine:
        python -m ibeis.algo.preproc.preproc_featweight --test-generate_featweight_properties

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> config2_ = ibs.new_query_params(dict(fg_on=True, fw_detector='rf'))
        >>> aid_list = ibs.get_valid_aids()[1:2]
        >>> fid_list = ibs.get_annot_feat_rowids(aid_list, ensure=True)
        >>> #fid_list = ibs.get_valid_fids()[1:2]
        >>> featweighttup_gen = generate_featweight_properties(ibs, fid_list, config2_=config2_)
        >>> featweighttup_list = list(featweighttup_gen)
        >>> featweight_list = featweighttup_list[0][0]
        >>> featweight_test = featweight_list[0:3]
        >>> featweight_target = [ 0.349, 0.218, 0.242]
        >>> ut.assert_almost_eq(featweight_test, featweight_target, .3)
    """
    # HACK: TODO AUTOGENERATE THIS
    #cid_list = ibs.get_feat_cids(feat_rowid_list)
    #aid_list = ibs.get_chip_aids(cid_list)
    chip_rowid_list = ibs.dbcache.get(ibs.const.FEATURE_TABLE, ('chip_rowid',), feat_rowid_list)
    aid_list = ibs.dbcache.get(ibs.const.CHIP_TABLE, ('annot_rowid',), chip_rowid_list)
    featweight_list = compute_fgweights(ibs, aid_list, config2_=config2_)
    return zip(featweight_list)


#def get_annot_probchip_fname_iter(ibs, aid_list):
#    """ Returns probability chip path iterator

#    Args:
#        ibs (IBEISController):
#        aid_list (list):

#    Returns:
#        probchip_fname_iter

#    Example:
#        >>> from ibeis.algo.preproc.preproc_featweight import *  # NOQA
#        >>> import ibeis
#        >>> ibs = ibeis.opendb('testdb1')
#        >>> aid_list = ibs.get_valid_aids()
#        >>> probchip_fname_iter = get_annot_probchip_fname_iter(ibs, aid_list)
#        >>> probchip_fname_list = list(probchip_fname_iter)
#    """
#    cfpath_list = ibs.get_annot_chip_fpath(aid_list, config2_=config2_)
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


def on_delete(ibs, featweight_rowid_list, config2_=None):
    # no external data to remove
    return 0

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.algo.preproc.preproc_featweight
        python -m ibeis.algo.preproc.preproc_featweight --allexamples
        python -m ibeis.algo.preproc.preproc_featweight --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
