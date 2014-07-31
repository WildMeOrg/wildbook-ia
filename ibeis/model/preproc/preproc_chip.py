from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip, range
from os.path import exists, join
import os
# UTool
import utool
# VTool
import vtool.chip as ctool
import vtool.image as gtool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[preproc_chip]', DEBUG=False)


#-------------
# Chip reading
#-------------


#@functools32.lru_cache(max_size=16)  # TODO: LRU cache needs to handle cfgstrs first
@utool.indent_func
def compute_or_read_annotation_chips(ibs, aid_list, ensure=True):
    """ Reads chips and tries to compute them if they do not exist """
    #print('[preproc_chip] compute_or_read_chips')
    if ensure:
        try:
            utool.assert_all_not_None(aid_list, 'aid_list')
        except AssertionError as ex:
            utool.printex(ex, key_list=['aid_list'])
            raise
    cfpath_list = get_annot_cfpath_list(ibs, aid_list)
    try:
        if ensure:
            chip_list = [gtool.imread(cfpath) for cfpath in cfpath_list]
        else:
            chip_list = [None if cfpath is None else gtool.imread(cfpath) for cfpath in cfpath_list]
    except IOError as ex:
        if not utool.QUIET:
            utool.printex(ex, '[preproc_chip] Handing Exception: ')
        ibs.add_chips(aid_list)
        try:
            chip_list = [gtool.imread(cfpath) for cfpath in cfpath_list]
        except IOError:
            print('[preproc_chip] cache must have been deleted from disk')
            compute_and_write_chips_lazy(ibs, aid_list)
            # Try just one more time
            chip_list = [gtool.imread(cfpath) for cfpath in cfpath_list]

    return chip_list


@utool.indent_func
def add_chips_params_gen(ibs, aid_list):
    """ computes chips if they do not exist.
    generates values for add_chips sqlcommands """
    cfpath_list = get_annot_cfpath_list(ibs, aid_list)
    chip_config_rowid = ibs.get_chip_config_rowid()
    for cfpath, aid in zip(cfpath_list, aid_list):
        pil_chip = gtool.open_pil_image(cfpath)
        width, height = pil_chip.size
        #print('Yeild Chip Param: aid=%r, cpath=%r' % (aid, cfpath))
        yield (aid, cfpath, width, height, chip_config_rowid)


#--------------
# Chip deleters
#--------------

@utool.indent_func
def delete_chips(ibs, cid_list, verbose=utool.VERBOSE):
    """ Removes chips from disk (not SQL)"""
    # TODO: Fixme, depends on current algo config
    chip_fpath_list = ibs.get_chip_paths(cid_list)
    if verbose:
        print('[preproc_chip] deleting %d chips' % len(cid_list))
    for cfpath in chip_fpath_list:
        if cfpath is None:
            continue
        try:
            os.remove(cfpath)
        except OSError:
            if exists(cfpath):
                print('[preproc_chip] cannot remove: %r ' % cfpath)


#---------------
# Chip filenames
#---------------


@utool.indent_func
def get_chip_fname_fmt(ibs=None, suffix=None):
    """ Returns format of chip file names """
    if suffix is None:
        chip_cfg = ibs.cfg.chip_cfg
        chip_cfgstr = chip_cfg.get_cfgstr()   # algo settings cfgstr
        chip_cfgfmt = chip_cfg['chipfmt']  # png / jpeg (BUGS WILL BE INTRODUCED IF THIS CHANGES)
        suffix = chip_cfgstr + chip_cfgfmt
    # Chip filenames are a function of annotation_rowid and cfgstr
    _cfname_fmt = ('aid_%d' + suffix)
    return _cfname_fmt


@utool.indent_func
def get_annot_cfpath_list(ibs, aid_list, suffix=None):
    """ Returns chip path list """
    #utool.assert_all_not_None(aid_list, 'aid_list')
    _cfname_fmt = get_chip_fname_fmt(ibs=ibs, suffix=suffix)
    cfname_iter = (None if aid is None else _cfname_fmt % aid for aid in iter(aid_list))
    cfpath_list = [None if cfname is None else join(ibs.chipdir, cfname) for cfname in cfname_iter]
    return cfpath_list


#---------------
# Chip computing
#---------------

def gen_chip(cfpath, gfpath, bbox, theta, new_size, filter_list=[]):
    """ worker function for parallel process """
    chipBGR = ctool.compute_chip(gfpath, bbox, theta, new_size, filter_list)
    return chipBGR, cfpath


def gen_chip2_no_write(tup):
    """ worker function for parallel generator """
    cfpath, gfpath, bbox, theta, new_size, filter_list = tup
    chipBGR = ctool.compute_chip(gfpath, bbox, theta, new_size, filter_list)
    return chipBGR, cfpath


def gen_chip2_and_write(tup):
    """ worker function for parallel generator """
    cfpath, gfpath, bbox, theta, new_size, filter_list = tup
    chipBGR = ctool.compute_chip(gfpath, bbox, theta, new_size, filter_list)
    #printDBG('write chip: %r' % cfpath)
    gtool.imwrite(cfpath, chipBGR)
    return cfpath

gen_chip2 = gen_chip2_and_write


@utool.indent_func
def gen_chips_async(cfpath_list, gfpath_list, bbox_list, theta_list,
                    newsize_list, filter_list=[], nChips=None):
    """ Computes chips and yeilds results asynchronously for writing  """
    # Compute and write chips in asychronous process
    if nChips is None:
        nChips = len(cfpath_list)
    filtlist_iter = (filter_list for _ in range(nChips))
    arg_prepend_iter = zip(cfpath_list, gfpath_list, bbox_list, theta_list,
                            newsize_list, filtlist_iter)
    arg_list = list(arg_prepend_iter)
    return utool.util_parallel.generate(gen_chip2, arg_list)


"""
#def gen_chips_async_OLD(cfpath_list, gfpath_list, bbox_list, theta_list,
                    #newsize_list, filter_list=[]):
    # TODO: Actually make this compute in parallel
    #chipinfo_iter = zip(cfpath_list, gfpath_list, bbox_list,
                         #theta_list, newsize_list)
    #num_chips = len(cfpath_list)
    #mark_prog, end_prog = utool.progress_func(num_chips, lbl='chips: ',
                                              #mark_start=True, flush_after=4)
    #for count, chipinfo in enumerate(chipinfo_iter):
        #(cfpath, gfpath, bbox, theta, new_size) = chipinfo
        #chipBGR = ctool.compute_chip(gfpath, bbox, theta, new_size, filter_list)
        #mark_prog(count)
        #yield chipBGR, cfpath
    #end_prog()

    #arg_list = list(chipinfo_iter)
    #args_dict = {'filter_list': filter_list}
    #result_list = utool.util_parallel.process(gen_chip, arg_list, args_dict)
    #for result in result_list:
        #yield result
"""


@utool.indent_func
def compute_and_write_chips(ibs, aid_list):
    utool.ensuredir(ibs.chipdir)
    # Get chip configuration information
    sqrt_area   = ibs.cfg.chip_cfg['chip_sqrt_area']
    filter_list = ctool.get_filter_list(ibs.cfg.chip_cfg.to_dict())
    # Get chip dest information (output path)
    cfpath_list = get_annot_cfpath_list(ibs, aid_list)
    # Get chip source information (image, annotation_bbox, theta)
    gfpath_list = ibs.get_annot_gpaths(aid_list)
    bbox_list   = ibs.get_annot_bboxes(aid_list)
    theta_list  = ibs.get_annot_thetas(aid_list)
    # Get how big to resize each chip
    target_area = sqrt_area ** 2
    bbox_size_iter = ((w, h) for (x, y, w, h) in bbox_list)
    newsize_list = ctool.get_scaled_sizes_with_area(target_area, bbox_size_iter)
    # Define "Asynchronous" generator
    chip_async_iter = gen_chips_async(cfpath_list, gfpath_list, bbox_list, theta_list,
                                      newsize_list, filter_list)
    print('Computing %d chips asynchronously' % (len(cfpath_list)))
    for cfpath in chip_async_iter:
        #print('Wrote chip: %r' % cfpath)
        pass
    print('Done computing chips')
    """
        #yield cfpath
    # Write results to disk as they come back from parallel processess
    #for chipBGR, cfpath in chip_async_iter:
        #printDBG('write chip: %r' % cfpath)
       #gtool.imwrite(cfpath, chipBGR)
    """


@utool.indent_func
def compute_and_write_chips_lazy(ibs, aid_list):
    """
    Will write a chip if it does not exist on disk, regardless of if it exists
    in the SQL database
    """
    print('[preproc_chip] compute_and_write_chips_lazy')
    # Mark which aid's need their chips computed
    cfpath_list = get_annot_cfpath_list(ibs, aid_list)
    exists_flags = [exists(cfpath) for cfpath in cfpath_list]
    invalid_aids = utool.get_dirty_items(aid_list, exists_flags)
    print('[preproc_chip] %d / %d chips need to be computed' %
          (len(invalid_aids), len(aid_list)))
    compute_and_write_chips(ibs, invalid_aids)
