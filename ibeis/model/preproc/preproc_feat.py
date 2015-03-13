from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip, range
# Science
import pyhesaff
# UTool
import utool
import utool as ut


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
            dict hesaff_params
            np.ndarray[kpts_t, ndims=2] kpts
            np.ndarray[desc_t, ndims=2] desc
    """
    cid, cpath, hesaff_params = tup
    kpts, desc = pyhesaff.detect_kpts(cpath, **hesaff_params)
    return (cid, len(kpts), kpts, desc)


def gen_feat_openmp(cid_list, cfpath_list, hesaff_params):
    """ Compute features in parallel on the C++ side, return generator here """
    print('Detecting %r features in parallel: ' % len(cid_list))
    kpts_list, desc_list = pyhesaff.detect_kpts_list(cfpath_list, **hesaff_params)
    for cid, kpts, desc in zip(cid_list, kpts_list, desc_list):
        nFeat = len(kpts)
        yield cid, nFeat, kpts, desc


def generate_feat_properties(ibs, cid_list, config2_=None, nInput=None):
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
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_feat import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> config2_ = None
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
    """
    if nInput is None:
        nInput = len(cid_list)
    if config2_ is not None:
        # Get config from config2_ object
        #print('id(config2_) = ' + str(id(config2_)))
        feat_cfgstr     = config2_.get('feat_cfgstr')
        hesaff_params   = config2_.get('hesaff_params')
        assert feat_cfgstr is not None
        assert hesaff_params is not None
    else:
        # Get config from IBEIS controller
        feat_cfgstr     = ibs.cfg.feat_cfg.get_cfgstr()
        hesaff_params   = ibs.cfg.feat_cfg.get_hesaff_params()

    ut.assert_all_not_None(cid_list, 'cid_list')
    cfpath_list       = ibs.get_chip_uris(cid_list)
    if ut.VERBOSE:
        print('[preproc_feat] cfgstr = %s' % feat_cfgstr)
        #print('hesaff_params = ' + ut.dict_str(hesaff_params))
    if USE_OPENMP:
        # Use Avi's openmp parallelization
        featgen_mp = gen_feat_openmp(cid_list, cfpath_list, hesaff_params)
        for (cid, nFeat, kpts, vecs) in ut.ProgressIter(featgen_mp, lbl='openmp feat'):
            yield (nFeat, kpts, vecs,)
    else:
        # Multiprocessing parallelization
        featgen = generate_feats(cfpath_list, hesaff_params=hesaff_params,
                                 cid_list=cid_list, nInput=nInput, ordered=True)
        for cid, nFeat, kpts, vecs in featgen:
            yield (nFeat, kpts, vecs,)
    pass


def generate_feats(cfpath_list, hesaff_params={}, cid_list=None, nInput=None, **kwargs):
    # chip-ids are an artifact of the IBEIS Controller. Make dummyones if needbe.
    """ Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        cfpath_list (list):
        hesaff_params (dict):
        cid_list (list):
        nInput (None):

    Kwargs:
        passed to utool.generate

    Returns:
        featgen

    Cyth:
        cdef:
            list cfpath_list
            long nInput
            object cid_list
            dict hesaff_params
            dict kwargs
    """
    if cid_list is None:
        cid_list = list(range(len(cfpath_list)))
    if nInput is None:
        nInput = len(cfpath_list)
    dictargs_iter = (hesaff_params for _ in range(nInput))
    arg_iter = zip(cid_list, cfpath_list, dictargs_iter)
    # eager evaluation.
    # TODO: see if we can just pass in the iterator or if there is benefit in
    # doing so
    arg_list = list(arg_iter)
    #if ut.VERBOSE:
    #    print('arg_list = ' + ut.list_str(arg_list))
    featgen = utool.util_parallel.generate(gen_feat_worker, arg_list, nTasks=nInput, **kwargs)
    return featgen


def on_delete(ibs, fid_list):
    # remove dependants of these rowids
    # No external data to remove
    return 0


if __name__ == '__main__':
    """
    python -m ibeis.model.preproc.preproc_feat
    python -m ibeis.model.preproc.preproc_feat --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
