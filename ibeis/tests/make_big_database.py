#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
from os.path import join, dirname, realpath
sys.path.append(realpath(join(dirname(__file__), '../..')))
from ibeis.tests import __testing__
from os.path import join, exists
from ibeis.dev import params
import ibeis
from ibeis.dev import ibsfuncs
import multiprocessing
import utool
from __testing__ import printTEST
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[MAKE_BIG_DB]')

sys.argv.append('--nogui')


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
    workdir = params.get_workdir()
    dbname = 'testdb_big'
    dbdir  = join(workdir, dbname)
    utool.delete(dbdir)

    main_locals = ibeis.main(dbdir=dbdir, gui=False)
    ibs = main_locals['ibs']    # IBEIS Control
    gpath_list = __testing__.get_test_gpaths(ndata=1)

    imgdir = get_big_imgdir(workdir)
    gname_list = utool.list_images(imgdir)
    gpath_list = [join(imgdir, gname) for gname in gname_list]
    gpath_list = gpath_list

    assert all(map(exists, gpath_list)), 'some images dont exist'

    #nImages = len(gpath_list)
    #with utool.Timer('Add %d Images' % nImages):
    gid_list = ibs.add_images(gpath_list)

    #with utool.Timer('Convert %d Images to rois' % nImages):
    rid_list = ibsfuncs.use_images_as_rois(ibs, gid_list)

    #with utool.Timer('Compute %d chips' % nImages):
    cid_list = ibs.add_chips(rid_list)

    #with utool.Timer('Compute %d features' % nImages):
    fid_list = ibs.add_feats(cid_list)

    #with utool.Timer('Getting %d nFeats' % nImages):
    nFeats_list = ibs.get_num_feats(fid_list)

    print('Total number of features in the database: %r' % sum(nFeats_list))
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = __testing__.run_test(MAKE_BIG_DB)
    execstr     = __testing__.main_loop(test_locals)
    exec('execstr')
