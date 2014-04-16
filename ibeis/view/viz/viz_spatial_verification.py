from __future__ import absolute_import, division, print_function
import utool
import drawtool.draw_sv as draw_sv
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_misc]', DEBUG=False)


WRITE_SV_DEBUG = utool.get_flag('--write-sv-debug')


def _get_sv_vartup_for_drawtool(ibs, rid1, rid2, chipmatch_FILT, rid2_svtup):
    """ Compiles IBEIS information into info suitable for drawtool """
    chip1, chip2 = ibs.get_roi_chips([rid1, rid2])
    kpts1, kpts2 = ibs.get_roi_kpts([rid1, rid2])
    rid2_fm, rid2_fs, rid2_fk = chipmatch_FILT
    fm = rid2_fm[rid2]
    (homog_inliers, H, aff_inliers, Aff) = rid2_svtup[rid2]
    homog_tup = (homog_inliers, H)
    aff_tup = (aff_inliers, Aff)
    sv_vartup = chip1, chip2, kpts1, kpts2, fm, homog_tup, aff_tup
    return sv_vartup


@utool.indent_func
def show_sv(ibs, rid1, rid2, chipmatch_FILT, rid2_svtup, **kwargs):
    """ Compiles IBEIS information and sends it to drawtool """
    print('\n[viz] ======================')
    sv_vartup = _get_sv_vartup_for_drawtool(ibs, rid1, rid2, chipmatch_FILT, rid2_svtup)
    (chip1, chip2, kpts1, kpts2, fm, homog_tup, aff_tup) = sv_vartup
    if WRITE_SV_DEBUG:
        keys = ('chip1', 'chip2', 'kpts1', 'kpts2', 'fm', 'homog_tup', 'aff_tup')
        utool.save_testdata(*keys)
        print('[vizsv] write test info')
        utool.qflag()
    draw_sv.show_sv(chip1, chip2, kpts1, kpts2, fm, homog_tup=homog_tup, aff_tup=aff_tup, **kwargs)
