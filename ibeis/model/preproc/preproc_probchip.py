"""
Preprocess Probability Chips

Uses random forests code to detect the probability that pixel belongs to the
forground.

TODO:
    * Create a probchip controller table.
    * Integrate into the the controller using autogen functions.
        - get_probchip_fpaths, get_annot_probchip_fpaths, add_annot_probchip

    * User should be able to manually paint on a chip to denote the foreground
      when the randomforest algorithm messes up.
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip
from ibeis.model.preproc import preproc_chip
from os.path import join, splitext
import utool  # NOQA
import utool as ut
import vtool
import numpy as np
# VTool
#import vtool.chip as ctool
#import vtool.image as gtool
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[preproc_probchip]', DEBUG=False)


def postprocess_dev():
    """
    References:
        http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    """
    from plottool import df2 as df2
    import cv2
    import numpy as np  # NOQA

    fpath = '/media/raid/work/GZ_ALL/_ibsdb/figures/nsum_hard/qaid=420_res_5ujbs8h&%vw1olnx_quuid=31cfdc3e/probchip_aid=478_auuid=5c327c5d-4bcc-22e4-764e-535e5874f1c7_CHIP(sz450)_FEATWEIGHT(ON,uselabel,rf)_CHIP()_zebra_grevys.png.png'
    img = cv2.imread(fpath)
    df2.imshow(img, fnum=1)
    kernel = np.ones((5, 5), np.uint8)
    blur = cv2.GaussianBlur(img, (5, 5), 1.6)
    dilation = cv2.dilate(img, kernel, iterations=10)
    df2.imshow(blur, fnum=2)
    df2.imshow(dilation, fnum=3)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
    df2.imshow(closing, fnum=4)
    df2.present()
    pass


def group_aids_by_featweight_species(ibs, aid_list, qreq_=None):
    """ helper

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> qreq_ = None
        >>> aid_list = ibs.get_valid_aids()
        >>> grouped_aids, unique_species = group_aids_by_featweight_species(ibs, aid_list, qreq_)
    """
    if qreq_ is None:
        featweight_species = ibs.cfg.featweight_cfg.featweight_species
    else:
        featweight_species = qreq_.qparams.featweight_species
    if featweight_species == 'uselabel':
        # Use the labeled species for the detector
        species_list = ibs.get_annot_species_texts(aid_list)
    else:
        species_list = [featweight_species]
    aid_list = np.array(aid_list)
    species_list = np.array(species_list)
    species_rowid = np.array(ibs.get_species_rowids_from_text(species_list))
    unique_species_rowids, groupxs = vtool.group_indicies(species_rowid)
    grouped_aids    = vtool.apply_grouping(aid_list, groupxs)
    grouped_species = vtool.apply_grouping(species_list, groupxs)
    unique_species = ut.get_list_column(grouped_species, 0)
    return grouped_aids, unique_species


# TODO: Move to controller:
def get_probchip_cachedir(ibs):
    return join(ibs.get_cachedir(), 'probchip')


def get_probchip_fname_fmt(ibs, qreq_=None, species=None):
    """ Returns format of probability chip file names

    Args:
        ibs (IBEISController):
        suffix (None):

    Returns:
        probchip_fname_fmt

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> qreq_ = None
        >>> probchip_fname_fmt = get_probchip_fname_fmt(ibs)
        >>> #want = 'probchip_aid=%d_bbox=%s_CHIP(sz450)_FEATWEIGHT(ON,uselabel,rf)_CHIP().png'
        >>> #assert probchip_fname_fmt == want, probchip_fname_fmt
        >>> result = probchip_fname_fmt
        >>> print(result)
        probchip_aid=%d_bbox=%s_theta=%s_gid=%d_CHIP(sz450)_FEATWEIGHT(ON,uselabel,rf)_CHIP().png

    """
    cfname_fmt = preproc_chip.get_chip_fname_fmt(ibs)

    if qreq_ is None:
        # FIXME FIXME FIXME: ugly, bad code that wont generalize at all.
        # you can compute probchips correctly only once, if you change anything
        # you have to delete your cache.
        probchip_cfgstr = ibs.cfg.featweight_cfg.get_cfgstr(use_feat=False, use_chip=False)
    else:
        raise NotImplementedError('qreq_ is not None')

    suffix = probchip_cfgstr
    if species is not None:
        # HACK, we sortof know the species here already from the
        # config string, but this helps in case we mess the config up
        suffix += '_' + species
    #probchip_cfgstr = ibs.cfg.detect_cfg.get_cfgstr()   # algo settings cfgstr
    fname_noext, ext = splitext(cfname_fmt)
    probchip_fname_fmt = ''.join(['prob', fname_noext, suffix, ext])
    return probchip_fname_fmt


def get_annot_probchip_fpath_list(ibs, aid_list, qreq_=None, species=None):
    """ Build probability chip file paths based on the current IBEIS configuration

    Args:
        ibs (IBEISController):
        aid_list (list):
        suffix (None):

    Returns:
        probchip_fpath_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_probchip import *  # NOQA
        >>> from os.path import basename
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> qreq_ = None
        >>> probchip_fpath_list = get_annot_probchip_fpath_list(ibs, aid_list)
        >>> result = basename(probchip_fpath_list[1])
        >>> print(result)
        probchip_aid=5_bbox=(0,0,1072,804)_theta=0.0tau_gid=5_CHIP(sz450)_FEATWEIGHT(ON,uselabel,rf)_CHIP().png
    """
    ibs.probchipdir = get_probchip_cachedir(ibs)
    cachedir = get_probchip_cachedir(ibs)
    ut.ensuredir(cachedir)
    #grouped_aids, unique_species = group_aids_by_featweight_species(ibs, aid_list, qreq_)
    #annot_uuid_list = ibs.get_annot_visual_uuids(aid_list)
    #for aids, species in zip(grouped_aids, unique_species):
    #probchip_fname_iter = (None if auuid is None else probchip_fname_fmt % auuid
    #                       for auuid in annot_uuid_list)
    ##probchip_fname_iter = (None if auuid is None else probchip_fname_fmt % (aid, auuid) for (aid, auuid) in
    #                       #zip(aid_list, annot_uuid_list))
    ##probchip_fpath_list = [None if fname is None else join(cachedir, fname)
    #                       #for fname in probchip_fname_iter]
    probchip_fname_fmt = get_probchip_fname_fmt(ibs, qreq_=qreq_, species=species)
    probchip_fpath_list = preproc_chip.format_aid_bbox_theta_gid_fnames(ibs, aid_list, probchip_fname_fmt, cachedir)
    return probchip_fpath_list


def compute_and_write_probchip(ibs, aid_list, qreq_=None):
    """ Computes probability chips using pyrf

    Example:
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> qreq_ = None
        >>> aid_list = ibs.get_valid_aids()
        >>> compute_and_write_probchip(ibs, aid_list, qreq_)

    Dev::
        #ibs.delete_annot_chips(aid_list)
        #probchip_fpath_list = get_annot_probchip_fpath_list(ibs, aid_list)
    """
    # Get probchip dest information (output path)
    from ibeis.model.detect import randomforest
    grouped_aids, unique_species = group_aids_by_featweight_species(ibs, aid_list, qreq_)
    cachedir   = get_probchip_cachedir(ibs)
    ut.ensuredir(cachedir)

    probchip_fpath_list_ = []
    if ut.VERBOSE:
        print('[preproc_probchip] +--------------------')
    for aids, species in zip(grouped_aids, unique_species):
        if ut.VERBOSE:
            print('[preproc_probchip] Computing probchips for species=%r' % species)
            print('[preproc_probchip] |--------------------')
        if len(aids) == 0:
            continue
        probchip_fpath_list = get_annot_probchip_fpath_list(ibs, aids, species=species)
        cfpath_list  = ibs.get_annot_chip_fpaths(aids)
        # Ensure that all chips are computed
        preproc_chip.compute_and_write_chips_lazy(ibs, aids, qreq_=qreq_)
        config = {
            # 'scale_list': [1.0],
            'output_gpath_list': probchip_fpath_list,
            'mode': 1,
        }
        results_list = list(randomforest.detect_gpath_list_with_species(ibs, cfpath_list, species, **config))  # NOQA
        probchip_fpath_list_.extend(probchip_fpath_list)
    if ut.VERBOSE:
        print('[preproc_probchip] Done computing probability images')
        print('[preproc_probchip] L_______________________')
    return probchip_fpath_list_

if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, ibeis.model.preproc.preproc_probchip; utool.doctest_funcs(ibeis.model.preproc.preproc_probchip, allexamples=True)"
        python -c "import utool, ibeis.model.preproc.preproc_probchip; utool.doctest_funcs(ibeis.model.preproc.preproc_probchip)"
        python -m ibeis.model.preproc.preproc_probchip
        python -m ibeis.model.preproc.preproc_probchip --allexamples
        python -m ibeis.model.preproc.preproc_probchip --allexamples --serial --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut  # NOQA
    ut.doctest_funcs()
