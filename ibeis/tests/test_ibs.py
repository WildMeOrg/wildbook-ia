#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import utool
from ibeis import viz
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_IBS]')


def TEST_IBS(ibs):
    gpath_list = grabdata.get_test_gpaths(ndata=2)

    print('[TEST] 1.ibs.add_images(gpath_list=%r)' % gpath_list)
    gid_list = ibs.add_images(gpath_list)
    print(' * gid_list=%r' % gid_list)

    print('[TEST] 2. ibs.get_image_paths(gid_list)')
    gpaths_list = ibs.get_image_paths(gid_list)
    print(' * gpaths_list=%r' % gpaths_list)

    print('[TEST] 3. get_image_props')
    uri_list   = ibs.get_image_uris(gid_list)
    path_list  = ibs.get_image_paths(gid_list)
    gsize_list = ibs.get_image_sizes(gid_list)
    time_list  = ibs.get_image_unixtime(gid_list)
    gps_list   = ibs.get_image_gps(gid_list)
    print(' * uri_list=%r' % uri_list)
    print(' * path_list=%r' % path_list)
    print(' * gsize_list=%r' % gsize_list)
    print(' * time_list=%r' % time_list)
    print(' * gps_list=%r' % gps_list)

    print('[TEST] 4. get_image_props')
    mult_cols_list = ibs.db.get('images', ('image_uri', 'image_width', 'image_height'), gid_list)
    print(' * gps_list=%r' % mult_cols_list)

    print('[TEST] 5. add_annots')
    gid = gid_list[0]
    gid_list = [gid, gid]
    bbox_list = [(50, 50, 100, 100), (75, 75, 102, 101)]
    theta_list = [0, 1.1]
    nid_list = None
    name_list = None
    notes_list = None
    aid_list = ibs.add_annots(gid_list, bbox_list, theta_list=theta_list,
                            nid_list=nid_list, name_list=name_list,
                            notes_list=notes_list)
    print(' * aid_list=%r' % aid_list)

    print('[TEST] 6. get_annot_props')
    gid_list    = ibs.get_annot_gids(aid_list)
    bbox_list   = ibs.get_annot_bboxes(aid_list)
    theta_list  = ibs.get_annot_thetas(aid_list)
    aids_list   = ibs.get_image_aids(gid)
    print(' * gid_list=%r' % gid_list)
    print(' * bbox_list=%r' % bbox_list)
    print(' * theta_list=%r' % theta_list)
    print(' * aids_list=%r' % aids_list)
    assert len(bbox_list) == len(theta_list), "len(bbox_list) != len(theta_list)"
    assert len(theta_list) == len(aid_list), "len(theta_list) != len(aid_list)"

    viz.show_image(ibs, gid)
    return locals()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For win32
    import ibeis
    # Initialize database
    try:
        main_locals = ibeis.main(defaultdb='testdb0', gui=False)
        ibs = main_locals['ibs']
        test_locals = utool.run_test(TEST_IBS, ibs)
    except RuntimeError as ex:
        utool.printex(ex)
        import matplotlib as mpl
        backend = mpl.get_backend()
        print('[test] current backend is: %r' % backend)
        raise
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
