from __future__ import absolute_import, division, print_function
# Python
from itertools import izip
from os.path import join
from os.path import exists
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


#@utool.lru_cache(16)  # TODO: LRU cache needs to handle cfg_uids first
def compute_or_read_roi_chips(ibs, rid_list):
    """ Reads chips and tries to compute them if they do not exist """
    printDBG('[preproc_chip] compute_or_read_chips')
    try:
        utool.assert_all_not_None(rid_list, 'rid_list')
    except AssertionError as ex:
        utool.printex(ex, key_list=['rid_list'])
        raise
    cfpath_list = ibs.get_roi_cpaths(rid_list)
    try:
        chip_list = [gtool.imread(cfpath) for cfpath in cfpath_list]
    except IOError as ex:
        if not utool.QUIET:
            utool.printex(ex, '[preproc_chip] Handing Exception: ')
        ibs.add_chips(rid_list)
        chip_list = [gtool.imread(cfpath) for cfpath in cfpath_list]
    return chip_list


def add_chips_parameters_gen(ibs, rid_list):
    """ computes chips if they do not exist.
    generates values for add_chips sqlcommands """
    cfpath_list = ibs.get_roi_cpaths(rid_list)
    chip_config_uid = ibs.get_chip_config_uid()
    for cfpath, rid in izip(cfpath_list, rid_list):
        pil_chip = gtool.open_pil_image(cfpath)
        width, height = pil_chip.size
        yield (rid, width, height, chip_config_uid)


#---------------
# Chip filenames
#---------------


def get_chip_fname_fmt(ibs):
    """ Returns format of chip file names """
    chip_cfg = ibs.cfg.chip_cfg
    chipcfg_uid = chip_cfg.get_uid()   # algo settings uid
    chipcfg_fmt = chip_cfg['chipfmt']  # png / jpeg
    # Chip filenames are a function of roi_uid and cfg_uid
    _cfname_fmt = ('rid_%s' + chipcfg_uid + chipcfg_fmt)
    return _cfname_fmt


def get_roi_cfpath_list(ibs, rid_list):
    """ Returns chip path list """
    utool.assert_all_not_None(rid_list, 'rid_list')
    _cfname_fmt = get_chip_fname_fmt(ibs)
    cfname_iter = (_cfname_fmt  % rid for rid in iter(rid_list))
    cfpath_list = [join(ibs.chipdir, cfname) for cfname in cfname_iter]
    return cfpath_list


#---------------
# Chip computing
#---------------


def gen_chips_async(cfpath_list, gfpath_list, bbox_list, theta_list,
                    newsize_list, filter_list=[]):
    """ Computes chips and yeilds results asynchronously for writing  """
    # TODO: Actually make this compute in parallel
    chipinfo_iter = izip(cfpath_list, gfpath_list, bbox_list,
                         theta_list, newsize_list)
    num_chips = len(cfpath_list)
    mark_prog, end_prog = utool.progress_func(num_chips, lbl='chips: ')
    for count, chipinfo in enumerate(chipinfo_iter):
        mark_prog(count)
        (cfpath, gfpath, bbox, theta, new_size) = chipinfo
        chipBGR = ctool.compute_chip(gfpath, bbox, theta,
                                     new_size, filter_list)
        yield chipBGR, cfpath
    end_prog()


def compute_and_write_chips(ibs, rid_list):
    utool.ensuredir(ibs.chipdir)
    # Get chip configuration information
    sqrt_area   = ibs.cfg.chip_cfg['chip_sqrt_area']
    filter_list = ctool.get_filter_list(ibs.cfg.chip_cfg.to_dict())
    # Get chip dest information (output path)
    cfpath_list = get_roi_cfpath_list(ibs, rid_list)
    # Get chip source information (image, roi_bbox, theta)
    gfpath_list = ibs.get_roi_gpaths(rid_list)
    bbox_list   = ibs.get_roi_bboxes(rid_list)
    theta_list  = ibs.get_roi_thetas(rid_list)
    # Get how big to resize each chip
    target_area = sqrt_area ** 2
    bbox_size_iter = ((w, h) for (x, y, w, h) in bbox_list)
    newsize_list = ctool.get_scaled_sizes_with_area(target_area, bbox_size_iter)
    # Define "Asynchronous" generator
    chip_async_iter = gen_chips_async(cfpath_list, gfpath_list,
                                      bbox_list, theta_list,
                                      newsize_list, filter_list)
    # Write results to disk as they come back from parallel processess
    for chipBGR, chip_fpath in chip_async_iter:
        printDBG('write chip: %r' % chip_fpath)
        gtool.imwrite(chip_fpath, chipBGR)


def compute_and_write_chips_lazy(ibs, rid_list):
    printDBG('[preproc_chip] compute_and_write_chips_lazy')
    # Mark which rid's need their chips computed
    cfpath_list = get_roi_cfpath_list(ibs, rid_list)
    dirty_flags = [not exists(cfpath) for cfpath in cfpath_list]
    invalid_rids = [rid for (rid, flag) in izip(rid_list, dirty_flags) if flag]
    printDBG('[preproc_chip] %d / %d chips need to be computed' %
             (len(invalid_rids), len(rid_list)))
    compute_and_write_chips(ibs, invalid_rids)
