# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import wbia.plottool.draw_sv as draw_sv

(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[viz_sver]', DEBUG=False)


WRITE_SV_DEBUG = ut.get_argflag('--write-sv-debug')


def _get_sv_vartup_for_plottool(
    ibs, aid1, aid2, chipmatch_FILT, aid2_svtup, config2_=None
):
    """ Compiles IBEIS information into info suitable for plottool """
    chip1, chip2 = ibs.get_annot_chips([aid1, aid2], config2_=config2_)
    kpts1, kpts2 = ibs.get_annot_kpts([aid1, aid2], config2_=config2_)
    aid2_fm = chipmatch_FILT.aid2_fm
    fm = aid2_fm[aid2]
    (homog_inliers, homog_err, H, aff_inliers, aff_err, Aff) = aid2_svtup[aid2]
    homog_tup = (homog_inliers, H)
    aff_tup = (aff_inliers, Aff)
    sv_vartup = chip1, chip2, kpts1, kpts2, fm, homog_tup, aff_tup
    return sv_vartup


def _compute_svvars(ibs, aid1):
    """
    DEPRICATE
    If spatial-verfication dbginfo is not in we need to compute it
    """
    from wbia.algo.hots import _pipeline_helpers as plh

    daids = ibs.get_valid_aids()
    qaids = [aid1]
    cfgdict = dict()
    qreq_ = ibs.new_query_request(qaids, daids, cfgdict)
    assert len(daids) > 0, '!!! nothing to search'
    assert len(qaids) > 0, '!!! nothing to query'
    qreq_.lazy_load()
    pipeline_locals_ = plh.testrun_pipeline_upto(qreq_, None)
    qaid2_chipmatch_FILT = pipeline_locals_['qaid2_chipmatch_FILT']
    qaid2_svtups = qreq_.metadata['qaid2_svtups']
    chipmatch_FILT = qaid2_chipmatch_FILT[aid1]
    aid2_svtup = qaid2_svtups[aid1]
    return chipmatch_FILT, aid2_svtup


@ut.indent_func
def show_sver(
    ibs, aid1, aid2, chipmatch_FILT=None, aid2_svtup=None, config2_=None, **kwargs
):
    """
    Compiles IBEIS information and sends it to plottool

    CommandLine:
        python -m wbia.viz.viz_sver --test-show_sver --show

    Example:
        >>> # SLOW_DOCTEST
        >>> from wbia.viz.viz_sver import *   # NOQA
        >>> import wbia
        >>> import utool as ut
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> aid1, aid2 = aid_list[0:2]
        >>> chipmatch_FILT = None
        >>> aid2_svtup = None
        >>> kwargs = {}
        >>> show_sver(ibs, aid1, aid2)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> exec(pt.present())
    """
    print('\n[show_sver] ====================== [show_sver]')
    # print(ut.func_str(show_sv, kwargs=locals()))
    if chipmatch_FILT is None or aid2_svtup is None:
        chipmatch_FILT, aid2_svtup = _compute_svvars(ibs, aid1)
    sv_vartup = _get_sv_vartup_for_plottool(
        ibs, aid1, aid2, chipmatch_FILT, aid2_svtup, config2_=config2_
    )
    (chip1, chip2, kpts1, kpts2, fm, homog_tup, aff_tup) = sv_vartup
    if WRITE_SV_DEBUG:
        keys = ('chip1', 'chip2', 'kpts1', 'kpts2', 'fm', 'homog_tup', 'aff_tup')
        ut.save_testdata(*keys)
        print('[vizsv] write test info')
        ut.qflag()
    draw_sv.show_sv(
        chip1, chip2, kpts1, kpts2, fm, homog_tup=homog_tup, aff_tup=aff_tup, **kwargs
    )


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.viz_sver --allexamples
        python -m wbia.viz.viz_sver --allexamples --show
    """
    ut.doctest_funcs()
