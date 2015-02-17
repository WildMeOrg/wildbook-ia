"""
Preprocess Chips

Extracts annotation chips from imaages and applies optional image
normalizations.

TODO:
    * Have controller delete cached chip_fpath if there is a cache miss.
    * Implemented funcs based on custom qparams in non None `qreq_` objects
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, range, filter  # NOQA
from os.path import exists, join
#import os
import utool as ut  # NOQA
import vtool.chip as ctool
import vtool.image as gtool
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[preproc_chip]', DEBUG=False)


#-------------
# Chip reading
#-------------

def compute_or_read_annotation_chips(ibs, aid_list, ensure=True):
    r"""
    Ignore::
        ----------------------
        Found 1 line(s) in 'code\\ibeis\\ibeis\\control\\manual_dependant_funcs.py':
        manual_dependant_funcs.py : 261 |    chip_list = preproc_chip.compute_or_read_annotation_chips(ibs, aid_list, ensure=ensure)
        ----------------------
        Found 1 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_chip.py':
        preproc_chip.py :  25 |def compute_or_read_annotation_chips(ibs, aid_list, ensure=True):
        ====================
        ====================
        Found 1 line(s) in 'code\\ibeis\\ibeis\\control\\manual_dependant_funcs.py':
        Found 1 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_chip.py':
    """
    if ensure:
        try:
            ut.assert_all_not_None(aid_list, 'aid_list')
        except AssertionError as ex:
            ut.printex(ex, key_list=['aid_list'])
            raise
    cfpath_list = get_annot_cfpath_list(ibs, aid_list)
    try:
        if ensure:
            chip_list = [gtool.imread(cfpath) for cfpath in cfpath_list]
        else:
            chip_list = [None if cfpath is None else gtool.imread(cfpath) for cfpath in cfpath_list]
    except IOError as ex:
        if not ut.QUIET:
            ut.printex(ex, '[preproc_chip] Handing Exception: ', iswarning=True)
        ibs.add_annot_chips(aid_list)
        try:
            chip_list = [gtool.imread(cfpath) for cfpath in cfpath_list]
        except IOError:
            print('[preproc_chip] cache must have been deleted from disk')
            # TODO: WE CAN SEARCH FOR NON EXISTANT PATHS HERE AND CALL
            # ibs.delete_annot_chips
            compute_and_write_chips_lazy(ibs, aid_list)
            # Try just one more time
            chip_list = [gtool.imread(cfpath) for cfpath in cfpath_list]

    return chip_list


@ut.indent_func
def add_annot_chips_params_gen(ibs, aid_list, qreq_=None):
    """Computes parameters for SQLController

    Igore::
        ----------------------
        Found 1 line(s) in 'code\\ibeis\\ibeis\\control\\manual_dependant_funcs.py':
        manual_dependant_funcs.py :  74 |            params_iter = preproc_chip.add_annot_chips_params_gen(ibs, dirty_aids)
        ----------------------
        Found 2 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_chip.py':
        preproc_chip.py :  69 |def add_annot_chips_params_gen(ibs, aid_list, qreq_=None):
        preproc_chip.py :  86 |        >>> params_iter = add_annot_chips_params_gen(ibs, aid_list)

    computes chips if they do not exist.
    generates values for add_annot_chips sqlcommands

    Args:
        ibs (IBEISController):
        aid_list (list):
        qreq_ (QueryRequest):

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> from os.path import basename
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> params_iter = add_annot_chips_params_gen(ibs, aid_list)
        >>> params_list = list(params_iter)
        >>> (aid, chip_config_rowid, cfpath, width, height,) = params_list[0]
        >>> fname = basename(cfpath)
        >>> result = (fname, width, height)
        >>> print(result)
        ('chip_aid=1_bbox=(0,0,1047,715)_theta=0.0_gid=1_CHIP(sz450).png', 545, 372)
    """
    try:
        # THIS DOESNT ACTUALLY COMPUTE ANYTHING!!!
        cfpath_list = get_annot_cfpath_list(ibs, aid_list)
        chip_config_rowid = ibs.get_chip_config_rowid(qreq_=qreq_)
        for cfpath, aid in zip(cfpath_list, aid_list):
            pil_chip = gtool.open_pil_image(cfpath)
            width, height = pil_chip.size
            if ut.DEBUG2:
                print('Yeild Chip Param: aid=%r, cpath=%r' % (aid, cfpath))
            yield (aid, chip_config_rowid, cfpath, width, height,)
    except IOError as ex:
        ut.printex(ex, 'ERROR IN PREPROC CHIPS')


@ut.indent_func
def delete_chips(ibs, cid_list, verbose=ut.VERBOSE):
    r"""
    Ignore::
        ----------------------
        Found 4 line(s) in 'code\\ibeis\\ibeis\\ibsfuncs.py':
        ibsfuncs.py :  426 |        ibs.delete_chips(invalid_cids, verbose=True)
        ibsfuncs.py :  681 |def delete_cache(ibs, delete_chips=False, delete_encounters=False):
        ibsfuncs.py :  689 |    if delete_chips:
        ibsfuncs.py :  743 |    ibs.delete_chips(all_cids)
        ----------------------
        Found 3 line(s) in 'code\\ibeis\\ibeis\\control\\manual_dependant_funcs.py':
        manual_dependant_funcs.py : 119 |    ibs.delete_chips(cid_list)
        manual_dependant_funcs.py : 131 |def delete_chips(ibs, cid_list, verbose=ut.VERBOSE, qreq_=None):
        manual_dependant_funcs.py : 137 |    #preproc_chip.delete_chips(ibs, cid_list, verbose=verbose)
        ----------------------
        Found 2 line(s) in 'code\\ibeis\\ibeis\\dev\\duct_tape.py':
        duct_tape.py :  52 |                         delete_chips_for_missing_annotations=False,
        duct_tape.py :  93 |    if delete_chips_for_missing_annotations:
        ----------------------
        Found 1 line(s) in 'code\\ibeis\\ibeis\\dev\\main_commands.py':
        main_commands.py :  89 |        ibs.delete_cache(delete_chips=True, delete_encounters=True)
        ----------------------
        Found 4 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_chip.py':
        preproc_chip.py : 122 |def delete_chips(ibs, cid_list, verbose=ut.VERBOSE):
        preproc_chip.py : 142 |        >>> delete_chips(ibs, cid_list, verbose=True)
        preproc_chip.py : 143 |        >>> ibs.delete_chips(cid_list)
        preproc_chip.py : 288 |        >>> ibs.delete_chips(cid_list2)
        ----------------------
        Found 1 line(s) in 'code\\ibeis\\ibeis\\tests\\test_delete_chips.py':
        test_delete_chips.py : 21 |    ibs.delete_chips(cid)

    DEPRICATE

    Removes chips from disk (does not remove from SQLController)
    this action must be performed by you.

    Args:
        ibs (IBEISController):
        cid_list (list):
        verbose (bool):

    Example:
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> compute_and_write_chips_lazy(ibs, aid_list)
        >>> cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=True)
        >>> print(set(cid_list))
        >>> delete_chips(ibs, cid_list, verbose=True)
        >>> ibs.delete_chips(cid_list)
        >>> cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=False)
    """
    on_delete(ibs, cid_list, qreq_=None, verbose=verbose, strict=False)


#  ^^ OLD FUNCS ^^

def compute_or_read_chip_images(ibs, cid_list, ensure=True, qreq_=None):
    """Reads chips and tries to compute them if they do not exist

    Ignore:
        ----------------------
        Found 4 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_chip.py':
        preproc_chip.py : 187 |def compute_or_read_chip_images(ibs, cid_list, ensure=True, qreq_=None):
        preproc_chip.py : 205 |        >>> chip_list = preproc_chip.compute_or_read_chip_images(ibs, cid_list)
        preproc_chip.py : 219 |        >>> # Now compute_or_read_chip_images should catch the bad thing
        preproc_chip.py : 221 |        >>> chip_list = preproc_chip.compute_or_read_chip_images(ibs, cid_list)

    Args:
        ibs (IBEISController):
        cid_list (list):
        ensure (bool):

    Returns:
        chip_list

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> import numpy as np
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=True)
        >>> chip_list = preproc_chip.compute_or_read_chip_images(ibs, cid_list)
        >>> result = np.array(list(map(np.shape, chip_list))).sum(0).tolist()
        >>> print(result)
        [1434, 2274, 12]

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> import numpy as np
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=True)
        >>> # Do a bad thing. Remove from disk without removing from sql
        >>> preproc_chip.on_delete(ibs, cid_list)
        >>> # Now compute_or_read_chip_images should catch the bad thing
        >>> # we did and correct for it.
        >>> chip_list = preproc_chip.compute_or_read_chip_images(ibs, cid_list)
        >>> result = np.array(list(map(np.shape, chip_list))).sum(0).tolist()
        >>> print(result)
        [1434, 2274, 12]
    """
    cfpath_list = ibs.get_chip_uris(cid_list)
    try:
        if ensure:
            try:
                ut.assert_all_not_None(cid_list, 'cid_list')
            except AssertionError as ex:
                ut.printex(ex, key_list=['cid_list'])
                raise
            chip_list = [gtool.imread(cfpath) for cfpath in cfpath_list]
        else:
            chip_list = [None if cfpath is None else gtool.imread(cfpath) for cfpath in cfpath_list]
    except IOError as ex:
        if not ut.QUIET:
            ut.printex(ex, '[preproc_chip] Handing Exception: ', iswarning=True)
        # Remove bad annotations from the sql database
        aid_list = ibs.get_chip_aids(cid_list)
        valid_list    = [cid is not None for cid in cid_list]
        valid_aids    = ut.filter_items(aid_list, valid_list)
        valid_cfpaths = ut.filter_items(cfpath_list, valid_list)
        bad_aids      = ut.filterfalse_items(valid_aids, map(exists, valid_cfpaths))
        ibs.delete_annot_chips(bad_aids)
        # Try readding things
        new_cid_list = ibs.add_annot_chips(aid_list)
        cfpath_list = ibs.get_chip_uris(new_cid_list)
        chip_list = [gtool.imread(cfpath) for cfpath in cfpath_list]
    return chip_list


def generate_chip_properties(ibs, aid_list, qreq_=None):
    r"""Computes parameters for SQLController

    computes chips if they do not exist.
    generates values for add_annot_chips sqlcommands

    Ignore:
        ----------------------
        Found 2 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_chip.py':
        preproc_chip.py : 262 |def generate_chip_properties(ibs, aid_list, qreq_=None):
        preproc_chip.py : 279 |        >>> params_iter = generate_chip_properties(ibs, aid_list)

    Args:
        ibs (IBEISController):
        aid_list (list):
        qreq_ (QueryRequest):

    Example:
        >>> # ENABLE DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> from os.path import basename
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> params_iter = generate_chip_properties(ibs, aid_list)
        >>> params_list = list(params_iter)
        >>> (cfpath, width, height,) = params_list[0]
        >>> fname = basename(cfpath)
        >>> fname_ = ut.regex_replace('auuid=.*_CHIP', 'auuid={uuid}_CHIP', fname)
        >>> result = (fname_, width, height)
        >>> print(result)
        ('chip_aid=1_auuid={uuid}_CHIP(sz450).png', 545, 372)
    """
    try:
        # the old function didn't even call this
        compute_and_write_chips(ibs, aid_list)
        cfpath_list = get_annot_cfpath_list(ibs, aid_list)
        #chip_config_rowid = ibs.get_chip_config_rowid()
        for cfpath, aid in zip(cfpath_list, aid_list):
            pil_chip = gtool.open_pil_image(cfpath)
            width, height = pil_chip.size
            if ut.DEBUG2:
                print('Yeild Chip Param: aid=%r, cpath=%r' % (aid, cfpath))
            yield (cfpath, width, height,)
    except IOError as ex:
        ut.printex(ex, 'ERROR IN PREPROC CHIPS')


#--------------
# Chip deleters
#--------------

def on_delete(ibs, cid_list, qreq_=None, verbose=True, strict=False):
    r"""
    Cleans up chips on disk.  Called on delete from sql controller.

    Ignore::
        Found 1 line(s) in 'code\\ibeis\\ibeis\\control\\manual_dependant_funcs.py':
        manual_dependant_funcs.py : 138 |    preproc_chip.on_delete(ibs, cid_list, verbose=verbose)
        ----------------------
        Found 1 line(s) in 'code\\ibeis\\ibeis\\control\\template_definitions.py':
        template_definitions.py : 312 |        preproc_{tbl}.on_delete({self}, {tbl}_rowid_list)
        ----------------------

    CommandLine:
        python -m ibeis.model.preproc.preproc_chip --test-on_delete

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> compute_and_write_chips_lazy(ibs, aid_list)
        >>> cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=True)
        >>> assert len(ut.filter_Nones(cid_list)) == len(cid_list)
        >>> # Run test function
        >>> qreq_ = None
        >>> verbose = True
        >>> strict = True
        >>> nRemoved1 = preproc_chip.on_delete(ibs, cid_list, qreq_=qreq_, verbose=verbose, strict=strict)
        >>> assert nRemoved1 == len(cid_list), 'nRemoved=%r, target=%r' % (nRemoved1,  len(cid_list))
        >>> nRemoved2 = preproc_chip.on_delete(ibs, cid_list, qreq_=qreq_, verbose=verbose, strict=strict)
        >>> assert nRemoved2 == 0, 'nRemoved=%r' % (nRemoved2,)
        >>> # We have done a bad thing at this point. SQL still thinks chips exist
        >>> cid_list2 = ibs.get_annot_chip_rowids(aid_list, ensure=False)
        >>> ibs.delete_chips(cid_list2)
    """
    chip_fpath_list = ibs.get_chip_uris(cid_list)
    aid_list = ibs.get_chip_aids(cid_list)
    gid_list = ibs.get_annot_gids(aid_list)
    ibs.delete_image_thumbs(gid_list)
    ibs.delete_annot_chip_thumbs(aid_list)
    nRemoved = ut.remove_existing_fpaths(chip_fpath_list, lbl='chips')
    #cid_list_ = ut.filter_Nones(cid_list)
    #chip_fpath_list = ibs.get_chip_uris(cid_list_)
    #exists_list = list(map(exists, chip_fpath_list))
    #if verbose:
    #    nTotal = len(cid_list)
    #    nValid = len(cid_list_)
    #    nExist = sum(exists_list)
    #    print('[preproc_chip.on_delete] requesting delete of %d chips' % (nTotal,))
    #    if nValid != nTotal:
    #        print('[preproc_chip.on_delete] trying to delete %d/%d non None chips ' % (nValid, nTotal))
    #    print('[preproc_chip.on_delete] %d/%d exist and need to be deleted' % (nExist, nValid))
    #nRemoved = 0
    #existing_cfpath_iter = ut.filter_items(chip_fpath_list, exists_list)
    #ut.remove_fpaths(existing_cfpath_iter)
    #for cfpath in existing_cfpath_iter:
    #    try:
    #        os.remove(cfpath)
    #        nRemoved += 1
    #    except OSError:
    #        print('[preproc_chip.on_delete] !!! cannot remove: %r ' % cfpath)
    #        if strict:
    #            raise
    #if verbose:
    #    print('[preproc_chip] sucesfully deleted %d/%d chips' % (nRemoved, nExist))
    return nRemoved


#---------------
# Chip filenames
#---------------


def get_annot_cfpath_list(ibs, aid_list):
    r"""
    Build chip file paths based on the current IBEIS configuration

    A chip depends on the chip config and the parent annotation's bounding box.
    (and technically the parent image (should we put that in the file path?)

    Ignore::
        ----------------------
        Found 1 line(s) in 'code\\ibeis\\ibeis\\control\\manual_annot_funcs.py':
        manual_annot_funcs.py : 1677 |    #cfpath_list = preproc_chip.get_annot_cfpath_list(ibs, aid_list)
        ----------------------
        Found 7 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_chip.py':
        preproc_chip.py :  45 |    cfpath_list = get_annot_cfpath_list(ibs, aid_list)
        preproc_chip.py : 109 |        cfpath_list = get_annot_cfpath_list(ibs, aid_list)
        preproc_chip.py : 297 |        cfpath_list = get_annot_cfpath_list(ibs, aid_list)
        preproc_chip.py : 426 |def get_annot_cfpath_list(ibs, aid_list):
        preproc_chip.py : 447 |        >>> cfpath_list = get_annot_cfpath_list(ibs, aid_list)
        preproc_chip.py : 599 |    cfpath_list = get_annot_cfpath_list(ibs, aid_list)
        preproc_chip.py : 650 |    cfpath_list = get_annot_cfpath_list(ibs, aid_list)

    Args:
        ibs (IBEISController):
        aid_list (list):
        suffix (None):

    Returns:
        cfpath_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from os.path import basename
        >>> ibs, aid_list = testdata_preproc_chip()
        >>> aid_list = aid_list[0:1]
        >>> cfpath_list = get_annot_cfpath_list(ibs, aid_list)
        >>> fname = '\n'.join(map(basename, cfpath_list))
        >>> result = fname
        >>> print(result)
        chip_aid=1_bbox=(0,0,1047,715)_theta=0.0tau_gid=1_CHIP(sz450).png

    """
    cfname_fmt = get_chip_fname_fmt(ibs)
    cfpath_list = format_aid_bbox_theta_gid_fnames(ibs, aid_list, cfname_fmt, ibs.chipdir)
    return cfpath_list


def get_chip_fname_fmt(ibs):
    r"""Returns format of chip file names

    Ignore::
        ----------------------
        Found 4 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_chip.py':
        preproc_chip.py : 468 |    cfname_fmt = get_chip_fname_fmt(ibs)
        preproc_chip.py : 473 |def get_chip_fname_fmt(ibs):
        preproc_chip.py : 487 |        >>> cfname_fmt = get_chip_fname_fmt(ibs)
        preproc_chip.py : 525 |        >>> fname_fmt = get_chip_fname_fmt(ibs)
        ----------------------
        Found 1 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_probchip.py':
        preproc_probchip.py : 112 |    cfname_fmt = preproc_chip.get_chip_fname_fmt(ibs)
        ====================

    Args:
        ibs (IBEISController):

    Returns:
        cfname_fmt

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> cfname_fmt = get_chip_fname_fmt(ibs)
        >>> result = cfname_fmt
        >>> print(result)
        chip_aid=%d_bbox=%s_theta=%s_gid=%d_CHIP(sz450).png
    """
    chip_cfgstr = ibs.cfg.chip_cfg.get_cfgstr()   # algo settings cfgstr
    chip_ext = ibs.cfg.chip_cfg['chipfmt']  # png / jpeg (BUGS WILL BE INTRODUCED IF THIS CHANGES)
    suffix = chip_cfgstr + chip_ext
    # Chip filenames are a function of annotation_rowid and cfgstr
    # TODO: Use annot uuids, use verts info as well
    #cfname_fmt = ('aid_%d' + suffix)
    #cfname_fmt = ''.join(['chip_auuid_%s' , suffix])
    #cfname_fmt = ''.join(['chip_aid=%d_auuid=%s' , suffix])
    # TODO: can use visual_uuids instead
    cfname_fmt = ''.join(['chip_aid=%d_bbox=%s_theta=%s_gid=%d' , suffix])
    return cfname_fmt


def format_aid_bbox_theta_gid_fnames(ibs, aid_list, fname_fmt, dpath):
    r"""
    format_aid_bbox_theta_gid_fnames

    Ignore::
        Found 4 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_chip.py':
        preproc_chip.py : 469 |    cfpath_list = format_aid_bbox_theta_gid_fnames(ibs, aid_list, cfname_fmt, ibs.chipdir)
        preproc_chip.py : 517 |def format_aid_bbox_theta_gid_fnames(ibs, aid_list, fname_fmt, dpath):
        preproc_chip.py : 519 |    format_aid_bbox_theta_gid_fnames
        preproc_chip.py : 539 |        >>> fpath_list = format_aid_bbox_theta_gid_fnames(ibs, aid_list, fname_fmt, dpath)
        ----------------------
        Found 1 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_probchip.py':
        preproc_probchip.py : 168 |    probchip_fpath_list = preproc_chip.format_aid_bbox_theta_gid_fnames(ibs, aid_list, probchip_fname_fmt, cachedir)

    Args:
        ibs (IBEISController):
        aid_list (list):
        fname_fmt (str):
        dpath (str):

    Returns:
        list: fpath_list

    Example:
        >>> from ibeis.model.preproc.preproc_chip import *   # NOQA
        >>> import ibeis
        >>> from os.path import basename
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> fname_fmt = get_chip_fname_fmt(ibs)
        >>> dpath = ibs.chipdir
        >>> fpath_list = format_aid_bbox_theta_gid_fnames(ibs, aid_list, fname_fmt, dpath)
        >>> result = str(basename(fpath_list[0]))
        >>> print(result)
        chip_aid=1_bbox=(0,0,1047,715)_theta=0.0tau_gid=1_CHIP(sz450).png
    """

    #ut.assert_all_not_None(aid_list, 'aid_list')
    #annot_uuid_list = ibs.get_annot_uuids(aid_list)
    #cfname_iter = (None if aid is None else cfname_fmt % aid for aid in iter(aid_list))
    #cfname_iter = (None if auuid is None else cfname_fmt % auuid for auuid in annot_uuid_list)
    #cfname_iter = (None if auuid is None else cfname_fmt % (aid, auuid)
    #               for (aid, auuid) in zip(aid_list, annot_uuid_list))
    # TODO: can use visual_uuids instead
    annot_bbox_list  = ibs.get_annot_bboxes(aid_list)
    annot_theta_list = ibs.get_annot_thetas(aid_list)
    annot_gid_list   = ibs.get_annot_gids(aid_list)
    try:
        annot_bboxstr_list = list([ut.bbox_str(bbox, pad=0, sep=',') for bbox in annot_bbox_list])
        annot_thetastr_list = list([ut.theta_str(theta, taustr='tau') for theta in annot_theta_list])
    except Exception as ex:
        ut.printex(ex, 'problem in chip_fname', keys=[
            'aid_list',
            'annot_bbox_list',
            'annot_theta_list',
            'annot_gid_list',
            'annot_bboxstr_list',
            'annot_thetastr_list',
        ]
        )
        raise
    tup_iter = zip(aid_list, annot_bboxstr_list, annot_thetastr_list, annot_gid_list)
    fname_iter = (None if tup[0] is None else fname_fmt % tup
                   for tup in tup_iter)
    fpath_list = [None if fname is None else join(dpath, fname)
                   for fname in fname_iter]
    return fpath_list


#---------------
# Chip computing
#---------------


# Parallelizable Worker
def gen_chip(tup):
    r"""
    Parallel worker. Crops chip out of an image, applies filters, etc

    Args:
        tup (tuple): (cfpath, gfpath, bbox, theta, new_size, filter_list)

    Returns:
        cfpath

    Example:
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
    """
    cfpath, gfpath, bbox, theta, new_size, filter_list = tup
    chipBGR = ctool.compute_chip(gfpath, bbox, theta, new_size, filter_list)
    #if DEBUG:
    #printDBG('write chip: %r' % cfpath)
    gtool.imwrite(cfpath, chipBGR)
    return cfpath


#@ut.indent_func
def compute_and_write_chips(ibs, aid_list):
    r"""Spawns compute compute chip processess.

    Ignore::
        Found 2 line(s) in 'code\\ibeis\\ibeis\\control\\manual_dependant_funcs.py':
        manual_dependant_funcs.py :  72 |            preproc_chip.compute_and_write_chips(ibs, aid_list)
        manual_dependant_funcs.py :  73 |            #preproc_chip.compute_and_write_chips_lazy(ibs, aid_list)
        ----------------------
        Found 1 line(s) in 'code\\ibeis\\ibeis\\tests\\test_ibs_chip_compute.py':
        test_ibs_chip_compute.py : 16 |    preproc_chip.compute_and_write_chips(ibs, aid_list)

    Args:
        ibs (IBEISController):
        aid_list (list):

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> ibs, aid_list = testdata_preproc_chip()
        >>> ibs.delete_annot_chips(aid_list)
        >>> cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=False)
        >>> compute_and_write_chips(ibs, aid_list)
    """
    ut.ensuredir(ibs.chipdir)
    # Get chip configuration information
    chip_sqrt_area = ibs.cfg.chip_cfg.chip_sqrt_area
    filter_list = ctool.get_filter_list(ibs.cfg.chip_cfg.to_dict())
    # Get chip dest information (output path)
    cfpath_list = get_annot_cfpath_list(ibs, aid_list)
    # Get chip source information (image, annotation_bbox, theta)
    gfpath_list = ibs.get_annot_image_paths(aid_list)
    bbox_list   = ibs.get_annot_bboxes(aid_list)
    theta_list  = ibs.get_annot_thetas(aid_list)
    # Get how big to resize each chip
    target_area = chip_sqrt_area ** 2
    bad_aids = [aid for aid, (x, y, w, h) in zip(aid_list, bbox_list) if w == 0 or h == 0]
    if len(bad_aids) > 0:
        msg = ("REMOVE INVALID (BAD WIDTH AND/OR HEIGHT) AIDS TO COMPUTE AND WRITE CHIPS")
        msg += ("INVALID AIDS: %r" % (bad_aids, ))
        print(msg)
        raise Exception(msg)
    bbox_size_iter = ((w, h) for (x, y, w, h) in bbox_list)
    newsize_list = ctool.get_scaled_sizes_with_area(target_area, bbox_size_iter)
    # Define "Asynchronous" generator
    # Compute and write chips in asychronous process
    nChips = len(aid_list)
    filtlist_iter = [filter_list for _ in range(nChips)]
    arg_prepend_iter = zip(cfpath_list, gfpath_list, bbox_list, theta_list,
                            newsize_list, filtlist_iter)
    arg_list = list(arg_prepend_iter)
    chip_async_iter = ut.util_parallel.generate(gen_chip, arg_list)
    if ut.VERBOSE:
        print('Computing %d chips asynchronously' % (len(cfpath_list)))
    for cfpath in chip_async_iter:
        #print('Wrote chip: %r' % cfpath)
        pass
    if not ut.VERBOSE:
        print('Done computing chips')
    #ut.print_traceback()


#@ut.indent_func
def compute_and_write_chips_lazy(ibs, aid_list, qreq_=None):
    r"""Spanws compute chip procesess if a chip does not exist on disk

    This is regardless of if it exists in the SQL database

    Args:
        ibs (IBEISController):
        aid_list (list):

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> ibs, aid_list = testdata_preproc_chip()
    """
    if ut.VERBOSE:
        print('[preproc_chip] compute_and_write_chips_lazy')
    # Mark which aid's need their chips computed
    cfpath_list = get_annot_cfpath_list(ibs, aid_list)
    exists_flags = [exists(cfpath) for cfpath in cfpath_list]
    invalid_aids = ut.get_dirty_items(aid_list, exists_flags)
    if ut.VERBOSE:
        print('[preproc_chip] %d / %d chips need to be computed' %
              (len(invalid_aids), len(aid_list)))
    compute_and_write_chips(ibs, invalid_aids)
    return cfpath_list


# TODO
#def read_chip_fpath(ibs, cid_list):
#    """ T_ExternFileGetter """
#    try:
#        return readfunc(fpath)
#    except IOError:
#        if not exists(fpath):
#            on_delete
#    else:
#        pass


#-------------
# Testing
#-------------

def testdata_preproc_chip():
    r"""testdata function

    Ignore:
        ----------------------
        Found 11 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_chip.py':
        preproc_chip.py :  98 |        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        preproc_chip.py : 174 |        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        preproc_chip.py : 211 |        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        preproc_chip.py : 223 |        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        preproc_chip.py : 284 |        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        preproc_chip.py : 372 |        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        preproc_chip.py : 459 |        >>> ibs, aid_list = testdata_preproc_chip()
        preproc_chip.py : 498 |        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        preproc_chip.py : 651 |        >>> ibs, aid_list = testdata_preproc_chip()
        preproc_chip.py : 729 |        >>> ibs, aid_list = testdata_preproc_chip()
        preproc_chip.py : 760 |def testdata_preproc_chip():
        ----------------------
        Found 2 line(s) in 'code\\ibeis\\ibeis\\model\\preproc\\preproc_probchip.py':
        preproc_probchip.py : 102 |        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        preproc_probchip.py : 148 |        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
    """
    import ibeis
    ibs = ibeis.opendb('testdb1')
    aid_list = ibs.get_valid_aids()[0::4]
    return ibs, aid_list


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, ibeis.model.preproc.preproc_chip; utool.doctest_funcs(ibeis.model.preproc.preproc_chip, allexamples=True)"
        python -c "import utool, ibeis.model.preproc.preproc_chip; utool.doctest_funcs(ibeis.model.preproc.preproc_chip)"
        python -m ibeis.model.preproc.preproc_chip
        python -m ibeis.model.preproc.preproc_chip --allexamples --serial --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut  # NOQA
    ut.doctest_funcs()
