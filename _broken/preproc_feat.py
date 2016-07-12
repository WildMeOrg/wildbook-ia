# -*- coding: utf-8 -*-
r"""
Computes patch based features based on Hesaff, SIFT, or convnets.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, range
import pyhesaff
import utool
import utool as ut
#ut.noinject('[preproc_feat]')
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_feat]')


#USE_OPENMP = not utool.WIN32
USE_OPENMP = False  # do not use openmp until we have the gravity vector


def generate_feat_properties(ibs, cid_list, config2_=None, nInput=None):
    r"""
    Computes features and yields results asynchronously: TODO: Remove IBEIS from
    this equation. Move the firewall towards the controller

    Args:
        ibs (IBEISController):
        cid_list (list):
        nInput (None):

    Returns:
        generator : generates param tups

    SeeAlso:
        ~/code/ibeis_cnn/ibeis_cnn/_plugin.py

    CommandLine:
        python -m ibeis.algo.preproc.preproc_feat --test-generate_feat_properties:0 --show
        python -m ibeis.algo.preproc.preproc_feat --test-generate_feat_properties:1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_feat import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> config2_ = ibs.new_query_params({})
        >>> nInput = None
        >>> aid_list = ibs.get_valid_aids()[::2]
        >>> ut.assert_all_not_None(aid_list, 'aid_list')
        >>> cid_list = ibs.get_annot_chip_rowids(aid_list, config2_=config2_)
        >>> ut.assert_all_not_None(cid_list, 'cid_list')
        >>> featgen = generate_feat_properties(ibs, cid_list, config2_, nInput)
        >>> feat_list = list(featgen)
        >>> assert len(feat_list) == len(aid_list)
        >>> (nFeat, kpts, vecs) = feat_list[0]
        >>> assert nFeat == len(kpts) and nFeat == len(vecs)
        >>> assert kpts.shape[1] == 6
        >>> assert vecs.shape[1] == 128
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> chip_fpath = ibs.get_annot_chip_fpath(aid_list[0], config2_=config2_)
        >>> pt.interact_keypoints.ishow_keypoints(chip_fpath, kpts, vecs)
        >>> ut.show_if_requested()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_feat import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> cfgdict = {}
        >>> cfgdict['feat_type'] = 'hesaff+siam128'
        >>> qreq_ = ibs.new_query_request([1], [1, 2, 3], cfgdict)
        >>> query_config2 = qreq_.extern_query_config2
        >>> data_config2 = qreq_.extern_data_config2
        >>> cid_list = ibs.get_annot_chip_rowids(ibs.get_valid_aids())
        >>> config2_ = query_config2
        >>> nInput = None
        >>> featgen = generate_feat_properties(ibs, cid_list, config2_, nInput)
        >>> result = list(featgen)
        >>> print(result)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_feat import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> config2_ = ibs.new_query_params({'affine_invariance': False, 'bgmethod': 'cnn'})
        >>> nInput = None
        >>> aid_list = ibs.get_valid_aids()[0:4]
        >>> ut.assert_all_not_None(aid_list, 'aid_list')
        >>> cid_list = ibs.get_annot_chip_rowids(aid_list, config2_=config2_)
        >>> ut.assert_all_not_None(cid_list, 'cid_list')
        >>> featgen = generate_feat_properties(ibs, cid_list, config2_, nInput)
        >>> feat_list = list(featgen)
        >>> assert len(feat_list) == len(aid_list)
        >>> (nFeat, kpts, vecs) = feat_list[0]
        >>> assert nFeat == len(kpts) and nFeat == len(vecs)
        >>> assert kpts.shape[1] == 6
        >>> assert vecs.shape[1] == 128
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> chip_fpath = ibs.get_annot_chip_fpath(aid_list[0], config2_=config2_)
        >>> pt.interact_keypoints.ishow_keypoints(chip_fpath, kpts, vecs)
        >>> ut.show_if_requested()

    Ignore:
        # STARTBLOCK
        import plottool as pt
        chip_fpath_list = ibs.get_chip_fpath(cid_list)
        fpath_list = list(ut.interleave((probchip_fpath_list, chip_fpath_list)))
        iteract_obj = pt.interact_multi_image.MultiImageInteraction(fpath_list, nPerPage=4)
        ut.show_if_requested()
        # ENDBLOCK
    """

    if nInput is None:
        nInput = len(cid_list)
    if config2_ is not None:
        # Get config from config2_ object
        #print('id(config2_) = ' + str(id(config2_)))
        feat_cfgstr     = config2_.get('feat_cfgstr')
        hesaff_params   = config2_.get('hesaff_params')
        feat_type       = config2_.get('feat_type')
        bgmethod        = config2_.get('bgmethod')
        assert feat_cfgstr is not None
        assert hesaff_params is not None
    else:
        # TODO: assert False here
        # Get config from IBEIS controller
        bgmethod        = ibs.cfg.feat_cfg.bgmethod
        feat_type       = ibs.cfg.feat_cfg.feat_type
        feat_cfgstr     = ibs.cfg.feat_cfg.get_cfgstr()
        hesaff_params   = ibs.cfg.feat_cfg.get_hesaff_params()

    ut.assert_all_not_None(cid_list, 'cid_list')
    chip_fpath_list = ibs.get_chip_fpath(cid_list, check_external_storage=True)

    if bgmethod is not None:
        aid_list = ibs.get_chip_aids(cid_list)
        probchip_fpath_list = ibs.get_annot_probchip_fpath(aid_list)
    else:
        probchip_fpath_list = (None for _ in range(nInput))

    if ut.NOT_QUIET:
        print('[preproc_feat] feat_cfgstr = %s' % feat_cfgstr)
        if ut.VERYVERBOSE:
            print('hesaff_params = ' + ut.dict_str(hesaff_params))

    if feat_type == 'hesaff+sift':
        if USE_OPENMP:
            # Use Avi's openmp parallelization
            assert bgmethod is None, 'not implemented'
            featgen_mp = gen_feat_openmp(cid_list, chip_fpath_list, hesaff_params)
            featgen = ut.ProgressIter(featgen_mp, lbl='openmp feat')
        else:
            # Multiprocessing parallelization
            featgen = extract_hesaff_sift_feats(
                chip_fpath_list, probchip_fpath_list,
                hesaff_params=hesaff_params, nInput=nInput, ordered=True)
    elif feat_type == 'hesaff+siam128':
        from ibeis_cnn import _plugin
        assert bgmethod is None, 'not implemented'
        featgen = _plugin.generate_siam_l2_128_feats(ibs, cid_list, config2_=config2_)
    else:
        raise AssertionError('unknown feat_type=%r' % (feat_type,))

    for nFeat, kpts, vecs in featgen:
        yield (nFeat, kpts, vecs,)


def extract_hesaff_sift_feats(chip_fpath_list, probchip_fpath_list, hesaff_params={}, nInput=None, **kwargs):
    """ Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        chip_fpath_list (list):
        hesaff_params (dict):
        nInput (None):

    Kwargs:
        passed to utool.generate

    Returns:
        featgen
    """
    if nInput is None:
        nInput = len(chip_fpath_list)
    dictargs_iter = (hesaff_params for _ in range(nInput))
    arg_iter = zip(chip_fpath_list, probchip_fpath_list, dictargs_iter)
    # eager evaluation.
    # TODO: Check if there is any benefit to just passing in the iterator.
    arg_list = list(arg_iter)
    featgen = utool.util_parallel.generate(gen_feat_worker, arg_list, nTasks=nInput,
                                           freq=10, **kwargs)
    return featgen


#def gen_feat_worker(tup):
#    r"""
#    Function to be parallelized by multiprocessing / joblib / whatever.
#    Must take in one argument to be used by multiprocessing.map_async
#    """
#    cid, cpath, hesaff_params = tup
#    kpts, vecs = pyhesaff.detect_feats(cpath, **hesaff_params)
#    return (cid, len(kpts), kpts, vecs)


def gen_feat_worker(tup):
    r"""
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        tup (tuple):

    Returns:
        tuple: (None, kpts, vecs)

    CommandLine:
        python -m ibeis.algo.preproc.preproc_feat --exec-gen_feat_worker --show
        python -m ibeis.algo.preproc.preproc_feat --exec-gen_feat_worker --show --aid 2
        python -m ibeis.algo.preproc.preproc_feat --exec-gen_feat_worker --show --aid 1988 --db GZ_Master1
        python -m ibeis.algo.preproc.preproc_feat --exec-gen_feat_worker --show --aid 1988 --db GZ_Master1 --affine-invariance=False --scale_max=30
        python -m ibeis.algo.preproc.preproc_feat --exec-gen_feat_worker --show --aid 1988 --db GZ_Master1 --affine-invariance=False --scale_max=30
        python -m ibeis.algo.preproc.preproc_feat --exec-gen_feat_worker --show --aid 1988 --db GZ_Master1 --affine-invariance=False --bgmethod=None  --scale_max=30

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.preproc.preproc_feat import *  # NOQA
        >>> import utool as ut
        >>> import ibeis
        >>> import vtool as vt
        >>> ibs, aid_list = ibeis.testdata_aids('PZ_MTEST')
        >>> aid = aid_list[0]
        >>> chip_fpath = ibs.get_annot_chip_fpath(aid)
        >>> bgmethod = ut.get_argval('--bgmethod', type_=str, default='cnn')
        >>> probchip_fpath = ibs.get_annot_probchip_fpath(aid) if 'cnn' == bgmethod else None
        >>> hesaff_params = {}  # {'affine_invariance': False}
        >>> hesaff_params = ut.argparse_dict(pyhesaff.get_hesaff_default_params())
        >>> tup = (chip_fpath, probchip_fpath, hesaff_params)
        >>> (num_kpts, kpts, vecs) = gen_feat_worker(tup)
        >>> result = ('(num_kpts, kpts, vecs) = %s' % (ut.repr2((num_kpts, kpts, vecs)),))
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> masked_chip, = ut.exec_func_src(gen_feat_worker, key_list=['masked_chip'], sentinal='kpts, vecs = pyhesaff')
        >>> pt.interact_keypoints.ishow_keypoints(masked_chip, kpts, vecs)
        >>> #pt.plot_score_histograms([vt.get_scales(kpts)])
        >>> ut.show_if_requested()
        >>> print(result)
    """
    import numpy as np
    import vtool as vt
    chip_fpath, probchip_fpath, hesaff_params = tup
    chip = vt.imread(chip_fpath)
    if probchip_fpath is not None:
        probchip = vt.imread(probchip_fpath, grayscale=True)
        masked_chip = (chip * (probchip[:, :, None].astype(np.float32) / 255)).astype(np.uint8)
    else:
        masked_chip = chip
    kpts, vecs = pyhesaff.detect_feats_in_image(masked_chip, **hesaff_params)
    num_kpts = kpts.shape[0]
    return (num_kpts, kpts, vecs)


def gen_feat_worker_old(tup):
    r"""
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async
    """
    #import vtool as vt
    chip_fpath, probchip_fpath, hesaff_params = tup
    kpts, vecs = pyhesaff.detect_feats(chip_fpath, **hesaff_params)
    return (kpts.shape[0], kpts, vecs)


def gen_feat_openmp(cid_list, chip_fpath_list, hesaff_params):
    r"""
    Compute features in parallel on the C++ side, return generator here
    """
    print('Detecting %r features in parallel: ' % len(cid_list))
    kpts_list, desc_list = pyhesaff.detect_feats_list(chip_fpath_list, **hesaff_params)
    for cid, kpts, vecs in zip(cid_list, kpts_list, desc_list):
        nFeat = len(kpts)
        yield cid, nFeat, kpts, vecs


def on_delete(ibs, fid_list):
    # remove dependants of these rowids
    # No external data to remove
    return 0


if __name__ == '__main__':
    """
    python -m ibeis.algo.preproc.preproc_feat
    python -m ibeis.algo.preproc.preproc_feat --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
