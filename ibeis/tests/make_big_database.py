#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from os.path import join, exists
import ibeis
from ibeis import ibsfuncs
from ibeis.dev import sysres
import multiprocessing
import utool
from vtool.tests import grabdata
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[MAKE_BIG_DB]')


__test__ = False  # This is not a test


def get_big_imgdir(workdir):
    """
    Get a place where a lot of images are.
    this probably will only work on jon's machines
    """
    if utool.get_computer_name() == 'BakerStreet':
        imgdir = r'D:\data\raw\Animals\Grevys\gz_mpala_cropped\images'
    elif  utool.get_computer_name() == 'Hyrule':
        imgdir = join(workdir, 'GZ_Cropped/images')
    elif  utool.get_computer_name() == 'Ooo':
        imgdir = join(workdir, 'FROG_tufts/images')
    else:
        raise AssertionError('this test only works on Jons computers')
    return imgdir


def MAKE_BIG_DB():
    workdir = sysres.get_workdir()
    dbname = 'testdb_big'
    dbdir  = join(workdir, dbname)
    utool.delete(dbdir)

    main_locals = ibeis.main(dbdir=dbdir, gui=False)
    ibs = main_locals['ibs']    # IBEIS Control
    gpath_list = grabdata.get_test_gpaths(ndata=1)

    imgdir = get_big_imgdir(workdir)
    gname_list = utool.list_images(imgdir, recursive=True)
    gpath_list = [join(imgdir, gname) for gname in gname_list]
    gpath_list = gpath_list

    assert all(map(exists, gpath_list)), 'some images dont exist'

    #nImages = len(gpath_list)
    #with utool.Timer('Add %d Images' % nImages):
    gid_list = ibs.add_images(gpath_list)

    #with utool.Timer('Convert %d Images to annotations' % nImages):
    aid_list = ibsfuncs.use_images_as_annotations(ibs, gid_list)

    #with utool.Timer('Compute %d chips' % nImages):
    cid_list = ibs.add_chips(aid_list)

    #with utool.Timer('Compute %d features' % nImages):
    fid_list = ibs.add_feats(cid_list)

    #with utool.Timer('Getting %d nFeats' % nImages):
    nFeats_list = ibs.get_num_feats(fid_list)

    print('Total number of features in the database: %r' % sum(nFeats_list))
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = utool.run_test(MAKE_BIG_DB)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec('execstr')
