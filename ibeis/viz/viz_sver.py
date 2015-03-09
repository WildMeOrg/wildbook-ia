from __future__ import absolute_import, division, print_function
import utool
import utool as ut
import plottool.draw_sv as draw_sv
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_sver]', DEBUG=False)


WRITE_SV_DEBUG = utool.get_argflag('--write-sv-debug')


def _get_sv_vartup_for_plottool(ibs, aid1, aid2, chipmatch_FILT, aid2_svtup, qreq_=None):
    """ Compiles IBEIS information into info suitable for plottool """
    chip1, chip2 = ibs.get_annot_chips([aid1, aid2], qreq_=qreq_)
    kpts1, kpts2 = ibs.get_annot_kpts([aid1, aid2], qreq_=qreq_)
    aid2_fm = chipmatch_FILT.aid2_fm
    #aid2_fs = chipmatch_FILT.aid2_fs
    #aid2_fk = chipmatch_FILT.aid2_fk
    fm = aid2_fm[aid2]
    (homog_inliers, homog_err, H, aff_inliers, aff_err, Aff) = aid2_svtup[aid2]
    homog_tup = (homog_inliers, H)
    aff_tup = (aff_inliers, Aff)
    sv_vartup = chip1, chip2, kpts1, kpts2, fm, homog_tup, aff_tup
    return sv_vartup


def _compute_svvars(ibs, aid1):
    """ If spatial-verfication dbginfo is not in we need to compute it """
    from ibeis.model.hots import pipeline
    from ibeis.model.hots import query_request
    daids = ibs.get_valid_aids()
    qaids = [aid1]
    cfgdict = dict(with_metadata=True)
    qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids, cfgdict)
    assert len(daids) > 0, '!!! nothing to search'
    assert len(qaids) > 0, '!!! nothing to query'
    qreq_.lazy_load()
    pipeline_locals_ = pipeline.testrun_pipeline_upto(qreq_, None)
    qaid2_chipmatch_FILT = pipeline_locals_['qaid2_chipmatch_FILT']
    qaid2_svtups         = qreq_.metadata['qaid2_svtups']
    chipmatch_FILT = qaid2_chipmatch_FILT[aid1]
    aid2_svtup     = qaid2_svtups[aid1]
    return chipmatch_FILT, aid2_svtup


@utool.indent_func
def show_sver(ibs, aid1, aid2, chipmatch_FILT=None, aid2_svtup=None, qreq_=None, **kwargs):
    """
    Compiles IBEIS information and sends it to plottool

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.viz.viz_sver import *   # NOQA
        >>> import ibeis
        >>> import utool as ut
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> aid1, aid2 = aid_list[0:2]
        >>> chipmatch_FILT = None
        >>> aid2_svtup = None
        >>> kwargs = {}
        >>> show_sver(ibs, aid1, aid2)
        >>> if ut.get_argflag('--show') or ut.inIPython():
        ...     import plottool as pt
        ...     exec(pt.present())
    """
    print('\n[show_sver] ====================== [show_sver]')
    #print(utool.func_str(show_sv, kwargs=locals()))
    if chipmatch_FILT is None or aid2_svtup is None:
        chipmatch_FILT, aid2_svtup = _compute_svvars(ibs, aid1)
    sv_vartup = _get_sv_vartup_for_plottool(ibs, aid1, aid2, chipmatch_FILT, aid2_svtup, qreq_=qreq_)
    (chip1, chip2, kpts1, kpts2, fm, homog_tup, aff_tup) = sv_vartup
    if WRITE_SV_DEBUG:
        keys = ('chip1', 'chip2', 'kpts1', 'kpts2', 'fm', 'homog_tup', 'aff_tup')
        utool.save_testdata(*keys)
        print('[vizsv] write test info')
        utool.qflag()
    draw_sv.show_sv(chip1, chip2, kpts1, kpts2, fm, homog_tup=homog_tup, aff_tup=aff_tup, **kwargs)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.viz_sver --allexamples
        python -m ibeis.viz.viz_sver --allexamples --show
    """
    ut.doctest_funcs()
