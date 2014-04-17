#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#-----
TEST_NAME = 'SAMPLE_DATA'
#-----
import sys
sys.argv.append('--nogui')
import __testing__
import multiprocessing
import utool
from ibeis.view import viz
from plottool import draw_func2 as df2
from ibeis.dev.all_imports import *  # NOQA
import vtool.tests.dummy as dummy
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)
printTEST = __testing__.printTEST


def get_test_kpts(ibs, n=11):
    rid = ibs.get_valid_rids()[0]
    kpts = ibs.get_roi_kpts(rid)
    kpts_samp = utool.util_list.spaced_items(kpts, n=n)
    return kpts_samp


def draw_data(img1, img2, kpts1, kpts2):
    # Draw keypoints
    kpkw = dict(ddd=True, rect=True, ori=True, eig=True, pts=True)
    pnum_ = df2.get_pnum_func(1, 2)
    viz.show_kpts(img1, kpts1, fnum=0, pnum=pnum_(0), **kpkw)
    viz.show_kpts(img2, kpts2, fnum=0, pnum=pnum_(1), **kpkw)


@__testing__.testcontext2(TEST_NAME)
def SAMPLE_DATA():
    main_locals = __testing__.main()
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA
    ##
    # Sample keypoints
    kpts = get_test_kpts(ibs, 11)
    img = dummy.get_kpts_dummy_img(kpts, sf=1.2)
    # Matching keypoints
    kpts1 = dummy.perterb_kpts(kpts, ori_std=np.tau / 6, damping=4)
    kpts2 = dummy.perterb_kpts(kpts, ori_std=np.tau / 6, damping=4)
    # Keypoint images
    img1  = dummy.get_kpts_dummy_img(kpts1, sf=1.2)
    img2  = dummy.get_kpts_dummy_img(kpts2, sf=1.2)
    # Matching indexes
    fx1_m = np.arange(len(kpts1))
    fx2_m = np.arange(len(kpts1))
    fm = np.vstack((fx1_m, fx2_m)).T

    # WRITE SPATIAL VERIFICATION DATA
    #sv_reprs = utool.get_reprs('kpts1', 'kpts2', 'fm')
    #sys.stdout.write(sv_reprs)
    #sys.stdout.write('\n')

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = SAMPLE_DATA()
    exec(test_locals['execstr'])
