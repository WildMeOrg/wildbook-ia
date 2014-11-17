#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from plottool import draw_func2 as df2
from plottool.viz_keypoints import show_keypoints
import numpy as np
import vtool.tests.dummy as dummy
print, print_, printDBG, rrr, profile = utool.inject(__name__,
                                                     '[%s]' % 'SAMPLE_DATA')

__test__ = False  # This is not a test


def get_test_kpts(ibs, n=11):
    aid = ibs.get_valid_aids()[0]
    kpts = ibs.get_annot_kpts(aid)
    kpts_samp = utool.spaced_items(kpts, n=n)
    return kpts_samp


def draw_data(img1, img2, kpts1, kpts2):
    # Draw keypoints
    kpkw = dict(ddd=True, rect=True, ori=True, eig=True, pts=True)
    pnum_ = df2.get_pnum_func(1, 2)
    show_keypoints(img1, kpts1, fnum=0, pnum=pnum_(0), **kpkw)
    show_keypoints(img2, kpts2, fnum=0, pnum=pnum_(1), **kpkw)


def SAMPLE_DATA(ibs):
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
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']
    test_locals = utool.run_test(SAMPLE_DATA, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
