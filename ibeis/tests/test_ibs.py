#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
try:
    import __testing__  # Should be imported before any ibeis stuff
except ImportError:
    pass
import multiprocessing
import utool
from ibeis import viz
printTEST = __testing__.printTEST
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_IBS]')


def TEST_IBS(ibs):
    gpath_list = __testing__.get_test_image_paths(ibs, ndata=1)

    printTEST('[TEST] 1.ibs.add_images(gpath_list=%r)' % gpath_list)
    gid_list = ibs.add_images(gpath_list)
    print(' * gid_list=%r' % gid_list)

    printTEST('[TEST] 2. ibs.get_image_paths(gid_list)')
    gpaths_list = ibs.get_image_paths(gid_list)
    print(' * gpaths_list=%r' % gpaths_list)

    printTEST('[TEST] 3. get_image_properties')
    uri_list   = ibs.get_image_uris(gid_list)
    path_list  = ibs.get_image_paths(gid_list)
    gsize_list = ibs.get_image_size(gid_list)
    time_list  = ibs.get_image_unixtime(gid_list)
    gps_list   = ibs.get_image_gps(gid_list)
    print(' * uri_list=%r' % uri_list)
    print(' * path_list=%r' % path_list)
    print(' * gsize_list=%r' % gsize_list)
    print(' * time_list=%r' % time_list)
    print(' * gps_list=%r' % gps_list)

    printTEST('[TEST] 4. get_image_properties')
    mult_cols_list = ibs.get_table_properties('images', ('image_uri', 'image_width', 'image_height'), gid_list)
    print(' * gps_list=%r' % mult_cols_list)

    printTEST('[TEST] 5. add_rois')
    gid = gid_list[0]
    gid_list = [gid, gid]
    bbox_list = [(50, 50, 100, 100), (75, 75, 102, 101)]
    theta_list = [0, 1.1]
    rid_list = ibs.add_rois(gid_list, bbox_list, theta_list)
    print(' * rid_list=%r' % rid_list)

    printTEST('[TEST] 6. get_roi_properties')
    gid_list    = ibs.get_roi_gids(rid_list)
    bbox_list   = ibs.get_roi_bboxes(rid_list)
    theta_list  = ibs.get_roi_thetas(rid_list)
    rids_list   = ibs.get_image_rids(gid)
    print(' * gid_list=%r' % gid_list)
    print(' * bbox_list=%r' % bbox_list)
    print(' * theta_list=%r' % theta_list)
    print(' * rids_list=%r' % rids_list)

    viz.show_image(ibs, gid)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    # Initialize database
    main_locals = __testing__.main(defaultdb='testdb')
    ibs = main_locals['ibs']
    test_locals = __testing__.run_test(TEST_IBS, ibs)
    execstr     = __testing__.main_loop(test_locals)
    exec(execstr)
