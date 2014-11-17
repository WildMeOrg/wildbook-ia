"""
Preprocess Chips

Extracts annotation chips from imaages and applies optional image
normalizations.

TODO:
    * Have controller delete cached chip_fpath if there is a cache miss.
    * Implemented funcs based on custom qparams in non None qreq_ objects
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, range, filter  # NOQA
from os.path import exists, join
import os
import utool as ut  # NOQA
import vtool.chip as ctool
import vtool.image as gtool
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[preproc_chip]', DEBUG=False)


#-------------
# Chip reading
#-------------

def compute_or_read_annotation_chips(ibs, aid_list, ensure=True):
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
            ut.printex(ex, '[preproc_chip] Handing Exception: ')
        ibs.add_chips(aid_list)
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
def add_chips_params_gen(ibs, aid_list, qreq_=None):
    """Computes parameters for SQLController

    computes chips if they do not exist.
    generates values for add_chips sqlcommands

    Args:
        ibs (IBEISController):
        aid_list (list):

    Example:
        >>> # ENABLE DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> from os.path import basename
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> params_iter = add_chips_params_gen(ibs, aid_list)
        >>> params_list = list(params_iter)
        >>> (aid, chip_config_rowid, cfpath, width, height,) = params_list[0]
        >>> fname = basename(cfpath)
        >>> fname_ = ut.regex_replace('auuid=.*_CHIP', 'auuid={uuid}_CHIP', fname)
        >>> result = (fname_, width, height)
        >>> print(result)
        ('chip_aid=1_auuid={uuid}_CHIP(sz450).png', 545, 372)
    """
    try:
        # THIS DOESNT ACTUALLY COMPUTE ANYTHING!!!
        cfpath_list = get_annot_cfpath_list(ibs, aid_list)
        chip_config_rowid = ibs.get_chip_config_rowid()
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
    """

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
        >>> cid_list = ibs.get_annot_cids(aid_list, ensure=True)
        >>> print(set(cid_list))
        >>> delete_chips(ibs, cid_list, verbose=True)
        >>> ibs.delete_chips(cid_list)
        >>> cid_list = ibs.get_annot_cids(aid_list, ensure=False)
    """
    on_delete(ibs, cid_list, qreq_=None, verbose=verbose, strict=False)


#  ^^ OLD FUNCS ^^

def compute_or_read_chip_images(ibs, cid_list, ensure=True, qreq_=None):
    """Reads chips and tries to compute them if they do not exist

    Args:
        ibs (IBEISController):
        cid_list (list):
        ensure (bool):

    Returns:
        chip_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> import numpy as np
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> cid_list = ibs.get_annot_cids(aid_list, ensure=True)
        >>> chip_list = preproc_chip.compute_or_read_chip_images(ibs, cid_list)
        >>> result = np.array(list(map(np.shape, chip_list))).sum(0).tolist()
        >>> print(result)
        [1434, 2274, 12]

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> import numpy as np
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> cid_list = ibs.get_annot_cids(aid_list, ensure=True)
        >>> # Do a bad thing. Remove from disk without removing from sql
        >>> preproc_chip.on_delete(ibs, cid_list)
        >>> # Now compute_or_read_chip_images should catch the bad thing
        >>> # we did and correct for it.
        >>> chip_list = preproc_chip.compute_or_read_chip_images(ibs, cid_list)
        >>> result = np.array(list(map(np.shape, chip_list))).sum(0).tolist()
        >>> print(result)
        [1434, 2274, 12]
    """
    cfpath_list = ibs.get_chip_paths(cid_list)
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
            ut.printex(ex, '[preproc_chip] Handing Exception: ')
        # Remove bad annotations from the sql database
        aid_list = ibs.get_chip_aids(cid_list)
        valid_list    = [cid is not None for cid in cid_list]
        valid_aids    = ut.filter_items(aid_list, valid_list)
        valid_cfpaths = ut.filter_items(cfpath_list, valid_list)
        bad_aids      = ut.filterfalse_items(valid_aids, map(exists, valid_cfpaths))
        ibs.delete_annot_chips(bad_aids)
        # Try readding things
        new_cid_list = ibs.add_chips(aid_list)
        cfpath_list = ibs.get_chip_paths(new_cid_list)
        chip_list = [gtool.imread(cfpath) for cfpath in cfpath_list]
    return chip_list


def generate_chip_properties(ibs, aid_list, qreq_=None):
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
    """
    Cleans up chips on disk.  Called on delete from sql controller.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> ibs, aid_list = preproc_chip.testdata_preproc_chip()
        >>> compute_and_write_chips_lazy(ibs, aid_list)
        >>> cid_list = ibs.get_annot_cids(aid_list, ensure=True)
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
        >>> cid_list2 = ibs.get_annot_cids(aid_list, ensure=False)
        >>> ibs.delete_chips(cid_list2)
    """
    cid_list_ = ut.filter_Nones(cid_list)
    chip_fpath_list = ibs.get_chip_paths(cid_list_)
    exists_list = list(map(exists, chip_fpath_list))
    if verbose:
        nTotal = len(cid_list)
        nValid = len(cid_list_)
        nExist = sum(exists_list)
        print('[preproc_chip.on_delete] requesting delete of %d chips' % (nTotal,))
        if nValid != nTotal:
            print('[preproc_chip.on_delete] trying to delete %d/%d non None chips ' % (nValid, nTotal))
        print('[preproc_chip.on_delete] %d/%d exist and need to be deleted' % (nExist, nValid))
    nRemoved = 0
    existing_cfpath_iter = ut.ifilter_items(chip_fpath_list, exists_list)
    for cfpath in existing_cfpath_iter:
        try:
            os.remove(cfpath)
            nRemoved += 1
        except OSError:
            print('[preproc_chip.on_delete] !!! cannot remove: %r ' % cfpath)
            if strict:
                raise
    if verbose:
        print('[preproc_chip] sucesfully deleted %d/%d chips' % (nRemoved, nExist))
    return nRemoved


#---------------
# Chip filenames
#---------------


def get_chip_fname_fmt(ibs):
    r"""Returns format of chip file names

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
        >>> assert cfname_fmt == 'chip_aid=%d_auuid=%s_CHIP(sz450).png', cfname_fmt
        >>> result = cfname_fmt
        >>> print(result)
        chip_aid=%d_auuid=%s_CHIP(sz450).png
    """
    chip_cfgstr = ibs.cfg.chip_cfg.get_cfgstr()   # algo settings cfgstr
    chip_ext = ibs.cfg.chip_cfg['chipfmt']  # png / jpeg (BUGS WILL BE INTRODUCED IF THIS CHANGES)
    suffix = chip_cfgstr + chip_ext
    # Chip filenames are a function of annotation_rowid and cfgstr
    # TODO: Use annot uuids, use verts info as well
    #cfname_fmt = ('aid_%d' + suffix)
    #cfname_fmt = ''.join(['chip_auuid_%s' , suffix])
    cfname_fmt = ''.join(['chip_aid=%d_auuid=%s' , suffix])
    return cfname_fmt


def get_annot_cfpath_list(ibs, aid_list):
    r"""
    Build chip file paths based on the current IBEIS configuration
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
        >>> result = ut.regex_replace('auuid=.*_CHIP', 'auuid={uuid}_CHIP', fname)
        >>> print(result)
        chip_aid=1_auuid={uuid}_CHIP(sz450).png

    chip_aid=1_auuid=2d021761-819d-4c40-af5a-7d8d3fc5b36f_CHIP(sz450).png
    """
    # TODO: Use annot uuids, use verts info as well
    #ut.assert_all_not_None(aid_list, 'aid_list')
    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    cfname_fmt = get_chip_fname_fmt(ibs)
    #cfname_iter = (None if aid is None else cfname_fmt % aid for aid in iter(aid_list))
    #cfname_iter = (None if auuid is None else cfname_fmt % auuid for auuid in annot_uuid_list)
    cfname_iter = (None if auuid is None else cfname_fmt % (aid, auuid)
                   for (aid, auuid) in zip(aid_list, annot_uuid_list))
    cfpath_list = [None if cfname is None else join(ibs.chipdir, cfname)
                   for cfname in cfname_iter]
    return cfpath_list


#---------------
# Chip computing
#---------------


# Parallelizable Worker
def gen_chip(tup):
    """Parallel worker. Crops chip out of an image, applies filters, etc

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
    """Spawns compute compute chip processess.

    Args:
        ibs (IBEISController):
        aid_list (list):

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> ibs, aid_list = testdata_preproc_chip()
        >>> ibs.delete_annot_chips(aid_list)
        >>> cid_list = ibs.get_annot_cids(aid_list, ensure=False)
        >>> compute_and_write_chips(ibs, aid_list)
    """
    ut.ensuredir(ibs.chipdir)
    # Get chip configuration information
    chip_sqrt_area = ibs.cfg.chip_cfg['chip_sqrt_area']
    filter_list = ctool.get_filter_list(ibs.cfg.chip_cfg.to_dict())
    # Get chip dest information (output path)
    cfpath_list = get_annot_cfpath_list(ibs, aid_list)
    # Get chip source information (image, annotation_bbox, theta)
    gfpath_list = ibs.get_annot_gpaths(aid_list)
    bbox_list   = ibs.get_annot_bboxes(aid_list)
    theta_list  = ibs.get_annot_thetas(aid_list)
    # Get how big to resize each chip
    target_area = chip_sqrt_area ** 2
    bad_aids = []
    for aid, (x, y, w, h) in zip(aid_list, bbox_list):
        if w == 0 or h == 0:
            bad_aids.append(aid)
    if len(bad_aids) > 0:
        print("REMOVE INVALID (BAD WIDTH AND/OR HEIGHT) AIDS TO COMPUTE AND WRITE CHIPS")
        print("INVALID AIDS: %r" % (bad_aids, ))
        raise
    bbox_size_iter = ((w, h) for (x, y, w, h) in bbox_list)
    newsize_list = ctool.get_scaled_sizes_with_area(target_area, bbox_size_iter)
    # Define "Asynchronous" generator
    # Compute and write chips in asychronous process
    nChips = len(aid_list)
    filtlist_iter = (filter_list for _ in range(nChips))
    arg_prepend_iter = zip(cfpath_list, gfpath_list, bbox_list, theta_list,
                            newsize_list, filtlist_iter)
    arg_list = list(arg_prepend_iter)
    chip_async_iter = ut.util_parallel.generate(gen_chip, arg_list)
    if not ut.QUIET:
        print('Computing %d chips asynchronously' % (len(cfpath_list)))
    for cfpath in chip_async_iter:
        #print('Wrote chip: %r' % cfpath)
        pass
    if not ut.QUIET:
        print('Done computing chips')
    #ut.print_traceback()


#@ut.indent_func
def compute_and_write_chips_lazy(ibs, aid_list, qreq_=None):
    """Spanws compute chip procesess if a chip does not exist on disk

    This is regardless of if it exists in the SQL database

    Args:
        ibs (IBEISController):
        aid_list (list):

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> ibs, aid_list = testdata_preproc_chip()
    """
    print('[preproc_chip] compute_and_write_chips_lazy')
    # Mark which aid's need their chips computed
    cfpath_list = get_annot_cfpath_list(ibs, aid_list)
    exists_flags = [exists(cfpath) for cfpath in cfpath_list]
    invalid_aids = ut.get_dirty_items(aid_list, exists_flags)
    print('[preproc_chip] %d / %d chips need to be computed' %
          (len(invalid_aids), len(aid_list)))
    compute_and_write_chips(ibs, invalid_aids)
    return cfpath_list


#-------------
# Testing
#-------------

def testdata_preproc_chip():
    """testdata function """
    import ibeis
    ibs = ibeis.opendb('testdb1')
    aid_list = ibs.get_valid_aids()[0::4]
    return ibs, aid_list


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, ibeis.model.preproc.preproc_chip; utool.doctest_funcs(ibeis.model.preproc.preproc_chip, allexamples=True)"
        python -c "import utool, ibeis.model.preproc.preproc_chip; utool.doctest_funcs(ibeis.model.preproc.preproc_chip)"
        python ibeis/model/preproc/preproc_chip.py
        python ibeis/model/preproc/preproc_chip.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut  # NOQA
    ut.doctest_funcs()
