from __future__ import absolute_import, division, print_function
import utool
import plottool.draw_sv as draw_sv
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_misc]', DEBUG=False)


WRITE_SV_DEBUG = utool.get_flag('--write-sv-debug')


def _get_sv_vartup_for_plottool(ibs, rid1, rid2, chipmatch_FILT, rid2_svtup):
    """ Compiles IBEIS information into info suitable for plottool """
    chip1, chip2 = ibs.get_roi_chips([rid1, rid2])
    kpts1, kpts2 = ibs.get_roi_kpts([rid1, rid2])
    rid2_fm, rid2_fs, rid2_fk = chipmatch_FILT
    fm = rid2_fm[rid2]
    (homog_inliers, H, aff_inliers, Aff) = rid2_svtup[rid2]
    homog_tup = (homog_inliers, H)
    aff_tup = (aff_inliers, Aff)
    sv_vartup = chip1, chip2, kpts1, kpts2, fm, homog_tup, aff_tup
    return sv_vartup


def _compute_svvars(ibs, rid1):
    """ If spatial-verfication dbginfo is not in we need to compute it """
    from ibeis.model.hots import query_helpers
    qrids = [rid1]
    qcomp = query_helpers.get_query_components(ibs, qrids)
    qrid2_chipmatch_FILT = qcomp['qrid2_chipmatch_FILT']
    qrid2_svtups         = qcomp['qrid2_svtups']
    chipmatch_FILT = qrid2_chipmatch_FILT[rid1]
    rid2_svtup     = qrid2_svtups[rid1]
    return chipmatch_FILT, rid2_svtup


@utool.indent_func
def show_sver(ibs, rid1, rid2, chipmatch_FILT=None, rid2_svtup=None, **kwargs):
    """ Compiles IBEIS information and sends it to plottool """
    print('\n[show_sver] ====================== [show_sver]')
    #print(utool.func_str(show_sv, kwargs=locals()))
    if chipmatch_FILT is None or rid2_svtup is None:
        chipmatch_FILT, rid2_svtup = _compute_svvars(ibs, rid1)
    sv_vartup = _get_sv_vartup_for_plottool(ibs, rid1, rid2, chipmatch_FILT, rid2_svtup)
    (chip1, chip2, kpts1, kpts2, fm, homog_tup, aff_tup) = sv_vartup
    if WRITE_SV_DEBUG:
        keys = ('chip1', 'chip2', 'kpts1', 'kpts2', 'fm', 'homog_tup', 'aff_tup')
        utool.save_testdata(*keys)
        print('[vizsv] write test info')
        utool.qflag()
    draw_sv.show_sv(chip1, chip2, kpts1, kpts2, fm, homog_tup=homog_tup, aff_tup=aff_tup, **kwargs)
