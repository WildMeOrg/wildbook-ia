"""
# DOCTEST ENABLED
DoctestCMD:
    python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.preproc.preproc_chip))" --quiet
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, range
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


#@functools32.lru_cache(max_size=16)  # TODO: LRU cache needs to handle cfgstrs first
@ut.indent_func
def compute_or_read_annotation_chips(ibs, aid_list, ensure=True):
    """Reads chips and tries to compute them if they do not exist

    Args:
        ibs (IBEISController):
        aid_list (list):
        ensure (bool):

    Returns:
        chip_list

    Example:
        >>> # DISABLE DOCTEST
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> ibs, aid_list = preproc_chip.test_setup_preproc_chip()
    """
    #print('[preproc_chip] compute_or_read_chips')
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
        >>> ibs, aid_list = preproc_chip.test_setup_preproc_chip()
        >>> params_iter = add_chips_params_gen(ibs, aid_list)
        >>> params_list = list(params_iter)
    """
    try:
        cfpath_list = get_annot_cfpath_list(ibs, aid_list)
        chip_config_rowid = ibs.get_chip_config_rowid()
        for cfpath, aid in zip(cfpath_list, aid_list):
            pil_chip = gtool.open_pil_image(cfpath)
            width, height = pil_chip.size
            if ut.DEBUG2:
                print('Yeild Chip Param: aid=%r, cpath=%r' % (aid, cfpath))
            yield (aid, cfpath, width, height, chip_config_rowid)
    except IOError as ex:
        ut.printex(ex, 'ERROR IN PREPROC CHIPS')


#--------------
# Chip deleters
#--------------


@ut.indent_func
def delete_chips(ibs, cid_list, verbose=ut.VERBOSE):
    """Removes chips from disk (does not remove from SQLController)
    this action must be performed by you.

    Args:
        ibs (IBEISController):
        cid_list (list):
        verbose (bool):

    Example:
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> from ibeis.model.preproc import preproc_chip
        >>> ibs, aid_list = preproc_chip.test_setup_preproc_chip()
        >>> compute_and_write_chips_lazy(ibs, aid_list)
        >>> cid_list = ibs.get_annot_cids(aid_list, ensure=True)
        >>> print(set(cid_list))
        >>> delete_chips(ibs, cid_list, verbose=True)
        >>> ibs.delete_chips(cid_list)
        >>> cid_list = ibs.get_annot_cids(aid_list, ensure=False)
    """
    # TODO: Fixme, depends on current algo config
    chip_fpath_list = ibs.get_chip_paths(cid_list)
    if verbose:
        print('[preproc_chip] deleting %d chips' % len(cid_list))
    needs_delete = list(map(exists, chip_fpath_list))
    if verbose:
        print('[preproc_chip] %d exist and need to be deleted' % sum(needs_delete))
    count = 0
    for cfpath in ut.ifilter_items(chip_fpath_list, needs_delete):
        try:
            os.remove(cfpath)
            count += 1
        except OSError:
            print('[preproc_chip] cannot remove: %r ' % cfpath)
    print('[preproc_chip] deleted %d chips' % count)


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
        >>> ibs, aid_list = preproc_chip.test_setup_preproc_chip()
        >>> cfname_fmt = get_chip_fname_fmt(ibs)
        >>> assert cfname_fmt == 'chip_auuid_%s_CHIP(sz450).png', cfname_fmt
        >>> print(cfname_fmt)
        chip_auuid_%s_CHIP(sz450).png
    """
    chip_cfgstr = ibs.cfg.chip_cfg.get_cfgstr()   # algo settings cfgstr
    chip_ext = ibs.cfg.chip_cfg['chipfmt']  # png / jpeg (BUGS WILL BE INTRODUCED IF THIS CHANGES)
    suffix = chip_cfgstr + chip_ext
    # Chip filenames are a function of annotation_rowid and cfgstr
    # TODO: Use annot uuids, use verts info as well
    #cfname_fmt = ('aid_%d' + suffix)
    cfname_fmt = ''.join(['chip_auuid_%s' , suffix])
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
        >>> ibs, aid_list = test_setup_preproc_chip()
        >>> cfpath_list = get_annot_cfpath_list(ibs, aid_list)
        >>> print('cfpath_list = \n' + '\n'.join(cfpath_list))
        >>> #assert 'chip_auuid_%s_CHIP(sz450).png' == cfpath_list
    """
    # TODO: Use annot uuids, use verts info as well
    #ut.assert_all_not_None(aid_list, 'aid_list')
    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    cfname_fmt = get_chip_fname_fmt(ibs)
    #cfname_iter = (None if aid is None else cfname_fmt % aid for aid in iter(aid_list))
    cfname_iter = (None if auuid is None else cfname_fmt % auuid for auuid in annot_uuid_list)
    cfpath_list = [None if cfname is None else join(ibs.chipdir, cfname) for cfname in cfname_iter]
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
        >>> ibs, aid_list = test_setup_preproc_chip()
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
        >>> from ibeis.model.preproc.preproc_chip import *  # NOQA
        >>> ibs, aid_list = test_setup_preproc_chip()
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

def test_setup_preproc_chip():
    """testdata function """
    import ibeis
    ibs = ibeis.opendb('testdb1')
    aid_list = ibs.get_valid_aids()
    return ibs, aid_list


def on_delete(ibs, cid_list, qreq_=None):
    print('Warning: Not Implemented')


if __name__ == '__main__':
    """
    python ibeis/model/preproc/preproc_chip.py
    python ibeis/model/preproc/preproc_chip.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    ut.doctest_funcs()
