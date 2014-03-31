from __future__ import division, print_function
# Python
from itertools import izip
from os.path import join
# Science
from PIL import Image
# Hotspotter
import utool
import vtool.chip as ctool
(print, print_,
 printDBG, rrr, profile) = utool.inject(__name__, '[chip_compute]', DEBUG=False)


def batch_extract_chips(gfpath_list, cfpath_list, bbox_list, theta_list,
                        uniform_size=None, uniform_sqrt_area=None,
                        filter_list=[]):
    '''
    cfpath_fmt - a string with a %d embedded where the cid will go.
    '''


# Main Script
@utool.indent_func
def compute_chips(ibs, rid_list, chip_cfg):
    print('=============================')
    print('[cc2] Precomputing chips and loading chip paths: %r' % ibs.get_db_name())
    chip_uid = chip_cfg.get_uid()

    gpath_list = ibs.get_roi_gpaths(rid_list)
    theta_list = ibs.get_roi_thetas(rid_list)
    bbox_list = ibs.get_roi_bboxes(rid_list)

    target_area = chip_cfg['chip_sqrt_area'] ** 2
    newsize_list = ctool.get_scaled_sizes_with_area(target_area, ((w, h) for (x, y, w, h) in bbox_list))

    args_list = [args for args in izip(gpath_list, bbox_list, theta_list, newsize_list)]
    filter_list = ctool.get_filter_list(chip_cfg)
    chip_kwargs = {
        'filter_list': filter_list,
    }
    chip_list = utool.util_parallel.process(ctool.compute_chip, args_list, chip_kwargs, force_serial=True)

    _cfname_fmt = 'rid%s' + chip_uid + chip_cfg['chipfmt']
    _cfpath_fmt = join(ibs.chipdir, _cfname_fmt)
    cfpath_list = [_cfpath_fmt  % cid for cid in iter(rid_list)]
    # Normalized Chip Sizes: ensure chips have about sqrt_area squared pixels

    try:
        # Hackish way to read images sizes a little faster.
        # change the directory so the os doesnt have to do as much work
        import os
        cwd = os.getcwd()
        os.chdir(ibs.chipdir)
        cfname_list = [_cfname_fmt  % cid for cid in iter(rid_list)]
        rsize_list = [(None, None) if path is None else Image.open(path).size
                      for path in iter(cfname_list)]
        os.chdir(cwd)
        return rsize_list
    except IOError as ex:
        print('[cc2] ex=%r' % ex)
        print('path=%r' % path)
        if utool.checkpath(path, verbose=True):
            import time
            time.sleep(1)  # delays for 1 seconds
            print('[cc2] file exists but cause IOError?')
            print('[cc2] probably corrupted. Removing it')
            try:
                utool.remove_file(path)
            except OSError:
                print('Something bad happened')
                raise
        raise
