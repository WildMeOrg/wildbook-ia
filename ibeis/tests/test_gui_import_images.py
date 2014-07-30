#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from six.moves import map
import multiprocessing
import utool
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_GUI_IMPORT_IMAGES]')


def TEST_GUI_IMPORT_IMAGES(ibs, back):
    print('[TEST] GET_TEST_IMAGE_PATHS')
    # The test api returns a list of interesting chip indexes
    mode = 'FILE'
    if mode == 'FILE':
        gpath_list = list(map(utool.unixpath, grabdata.get_test_gpaths()))

        # else:
        #    dir_ = utool.truepath(join(sysres.get_workdir(), 'PZ_MOTHERS/images'))
        #    gpath_list = utool.list_images(dir_, fullpath=True, recursive=True)[::4]
        print('[TEST] IMPORT IMAGES FROM FILE\n * gpath_list=%r' % gpath_list)
        gid_list = back.import_images(gpath_list=gpath_list)
        thumbtup_list = ibs.get_image_thumbtup(gid_list)
        imgpath_list = [tup[1] for tup in thumbtup_list]
        gpath_list2 = ibs.get_image_paths(gid_list)
        for path in gpath_list2:
            assert path in imgpath_list, "Imported Image not in db, path=%r" % path
    elif mode == 'DIR':
        dir_ = grabdata.get_testdata_dir()
        print('[TEST] IMPORT IMAGES FROM DIR\n * dir_=%r' % dir_)
        gid_list = back.import_images(dir_=dir_)
    else:
        raise AssertionError('unknown mode=%r' % mode)

    print('[TEST] * len(gid_list)=%r' % len(gid_list))
    return locals()

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb0', gui=True, allow_newdir=True)
    ibs  = main_locals['ibs']   # IBEIS Control
    back = main_locals['back']  # IBEIS GUI backend
    test_locals = utool.run_test(TEST_GUI_IMPORT_IMAGES, ibs, back)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
