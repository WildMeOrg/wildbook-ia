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


def group_aids_by_featweight_species(ibs, aid_list, qreq_=None):
    """ helper

    Example:
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
        species_list = ibs.get_annot_species(aid_list)
    else:
        species_list = [featweight_species]
    aid_list = np.array(aid_list)
    species_list = np.array(species_list)
    species_rowid = np.array(ibs.get_species_lblannot_rowid(species_list))
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
        >>> # DOCTEST ENABLED
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> ibs, aid_list = preproc_chip.test_setup_preproc_chip()
        >>> qreq_ = None
        >>> probchip_fname_fmt = get_probchip_fname_fmt(ibs)
        >>> assert probchip_fname_fmt == 'probchip_auuid_%s_CHIP(sz450)_DETECT(rf,zebra_grevys).png', probchip_fname_fmt
        >>> print(probchip_fname_fmt)
        probchip_auuid_%s_CHIP(sz450)_DETECT(rf,zebra_grevys).png

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
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> ibs, aid_list = test_setup_preproc_chip()
        >>> qreq_ = None
        >>> probchip_fpath_list = get_annot_probchip_fpath_list(ibs, aid_list)
        >>> print(probchip_fpath_list[1])
    """
    ibs.probchipdir = get_probchip_cachedir(ibs)
    cachedir = get_probchip_cachedir(ibs)
    ut.ensuredir(cachedir)

    #grouped_aids, unique_species = group_aids_by_featweight_species(ibs, aid_list, qreq_)

    probchip_fname_fmt = get_probchip_fname_fmt(ibs, qreq_=qreq_, species=species)
    annot_uuid_list = ibs.get_annot_uuids(aid_list)

    #for aids, species in zip(grouped_aids, unique_species):
    probchip_fname_iter = (None if auuid is None else probchip_fname_fmt % auuid
                           for auuid in annot_uuid_list)
    probchip_fpath_list = [None if fname is None else join(cachedir, fname)
                           for fname in probchip_fname_iter]
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
    if qreq_ is None:
        use_chunks = ibs.cfg.other_cfg.detect_use_chunks
    else:
        use_chunks = qreq_.qparams.detect_use_chunks

    grouped_aids, unique_species = group_aids_by_featweight_species(ibs, aid_list, qreq_)
    cachedir   = get_probchip_cachedir(ibs)
    ut.ensuredir(cachedir)

    gropued_probchip_fpath_lists = []
    print('[preproc_probchip] +--------------------')
    for aids, species in zip(grouped_aids, unique_species):
        if not ut.QUIET:
            print('[preproc_probchip] |--------------------')
            print('[preproc_probchip] Computing probchips for species=%r' % species)
        if len(aids) == 0:
            continue
        probchip_fpath_list = get_annot_probchip_fpath_list(ibs, aids, species=species)
        cfpath_list  = ibs.get_annot_cpaths(aids)
        preproc_chip.compute_and_write_chips_lazy(ibs, aids, qreq_=qreq_)
        # Ensure that all chips are computed
        # LAZY-CODE IS DONE HERE randomforest only computes probchips that it needs to
        randomforest.compute_probability_images(cfpath_list, probchip_fpath_list, species, use_chunks=use_chunks)
        # Fix stupid bug in pyrf
        fixed_probchip_fpath_list = [fpath + '.png' for fpath in probchip_fpath_list]
        gropued_probchip_fpath_lists.append(fixed_probchip_fpath_list)
    probchip_fpath_list_ = ut.flatten(gropued_probchip_fpath_lists)
    if not ut.QUIET:
        print('[preproc_probchip] Done computing probability images')
    print('[preproc_probchip] L_______________________')
    return probchip_fpath_list_

if __name__ == '__main__':
    """
    python ibeis/model/preproc/preproc_probchip.py
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import ut as ut  # NOQA
    testable_list = [
    ]
    ut.doctest_funcs(testable_list)
