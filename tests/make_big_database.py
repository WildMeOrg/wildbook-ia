#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
#------
TEST_NAME = 'BIGDB'
#------
import __testing__
import sys
from os.path import join, exists
from ibeis.dev import params
import multiprocessing
import utool
printTEST = __testing__.printTEST
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)

sys.argv.append('--nogui')


@__testing__.testcontext
def BIGDB():
    workdir = params.get_workdir()
    dbname = 'test_big_ibeis'
    utool.delete(join(workdir, dbname))

    main_locals = __testing__.main(defaultdb=dbname, allow_newdir=True, nogui=True)
    ibs = main_locals['ibs']    # IBEIS Control
    gpath_list = __testing__.get_test_image_paths(ibs, ndata=1)
    # this will probably will only work on jon's machines
    if utool.get_computer_name() == 'BakerStreet':
        imgdir = r'D:\data\raw\Animals\Grevys\gz_mpala_cropped\images'
    elif  utool.get_computer_name() == 'Hyrule':
        imgdir = join(workdir, 'GZ_Cropped/images')
    elif  utool.get_computer_name() == 'Ooo':
        imgdir = join(workdir, 'FROG_tufts/images')
    else:
        raise AssertionError('this test only works on Jons computers')
    gname_list = utool.list_images(imgdir)
    gpath_list = [join(imgdir, gname) for gname in gname_list]
    gpath_list = gpath_list

    assert all(map(exists, gpath_list)), 'some images dont exist'

    #nImages = len(gpath_list)

    #with utool.Timer('Add %d Images' % nImages):
    gid_list = ibs.add_images(gpath_list)

    #with utool.Timer('Convert %d Images to rois' % nImages):
    rid_list = ibs.use_images_as_rois(gid_list)

    #with utool.Timer('Compute %d chips' % nImages):
    cid_list = ibs.add_chips(rid_list)

    #with utool.Timer('Compute %d features' % nImages):
    fid_list = ibs.add_feats(cid_list)

    #with utool.Timer('Getting %d nFeats' % nImages):
    nFeats_list = ibs.get_num_feats(fid_list)

    print('Total number of features in the database: %r' % sum(nFeats_list))

    # Run Qt Loop to use the GUI
    printTEST('[TEST] MAIN_LOOP')
    __testing__.main_loop(main_locals, rungui=False)
BIGDB.func_name = TEST_NAME


if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    BIGDB()
    #from drawtool import draw_func2 as df2
    #exec(df2.present())
