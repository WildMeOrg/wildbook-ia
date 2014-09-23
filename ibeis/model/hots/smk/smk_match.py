from __future__ import absolute_import, division, print_function
#import six
import utool
#import numpy as np
from ibeis.model.hots import pipeline
from ibeis.model.hots.smk import smk_index
from ibeis.model.hots.smk import smk_core
#from six.moves import zip
#from ibeis.model.hots.smk.hstypes import INTEGER_TYPE
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_match]')


@profile
def query_inverted_index(annots_df, qaid, invindex, withinfo=True,
                         aggregate=False, alpha=3, thresh=0):
    """
    >>> from ibeis.model.hots.smk.smk_match import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_index
    >>> from ibeis.model.hots.smk import smk_debug
    >>> ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_internals()
    >>> qaid = qaids[0]
    >>> aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    >>> alpha     = ibs.cfg.query_cfg.smk_cfg.alpha
    >>> thresh    = ibs.cfg.query_cfg.smk_cfg.thresh
    >>> withinfo = True
    >>> daid2_totalscore, daid2_chipmatch = query_inverted_index(annots_df, qaid, invindex, withinfo, aggregate, alpha, thresh)
    """
    #from ibeis.model.hots.smk import smk_index
    # Get query words / residuals
    query_repr = smk_index.compute_query_repr(annots_df, qaid, invindex,
                                              aggregate, alpha, thresh)
    (wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma) = query_repr
    if False and __debug__:
        from ibeis.model.hots.smk import smk_debug
        qfx2_vec = annots_df['vecs'][qaid]
        #smk_debug.invindex_dbgstr(invindex)
        assert smk_debug.check_wx2_rvecs2(invindex, wx2_qrvecs, wx2_qfxs, qfx2_vec), 'bad query_repr in query_inverted_index'
        assert smk_debug.check_wx2_rvecs2(invindex), 'bad invindex in query_inverted_index'
    # Compute match kernel for all database aids
    kernel_args = (wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma, invindex,
                   withinfo, alpha, thresh)
    daid2_totalscore, daid2_chipmatch = smk_core.match_kernel(*kernel_args)
    # Prevent self matches
    can_match_self = True
    if (not can_match_self) and qaid in daid2_totalscore:
        import numpy as np
        daid2_totalscore[qaid] = 0
        daid2_chipmatch[0][qaid] = np.empty((0, 2), dtype=np.int32)
        daid2_chipmatch[1][qaid] = np.empty((0), dtype=np.float32)
        daid2_chipmatch[2][qaid] = np.empty((0), dtype=np.int32)
    # Build chipmatches if daid2_wx2_scoremat is not None
    #if __debug__:
    #    from ibeis.model.hots.smk import smk_debug
    #    smk_debug.check_daid2_chipmatch(daid2_chipmatch)
    return daid2_totalscore, daid2_chipmatch


@profile
def query_smk(annots_df, invindex, qreq_):
    """
    ibeis interface
    >>> from ibeis.model.hots.smk.smk_match import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_match
    >>> from ibeis.model.hots.smk import smk_debug
    >>> from ibeis.model.hots import query_request
    >>> ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_internals()
    >>> qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
    >>> qaid2_qres_ = smk_match.query_smk(annots_df, invindex, qreq_)

    qres = qaid2_qres_[qaids[0]]
    fig = qres.show_top(ibs)

    """
    qaids = qreq_.get_external_qaids()
    qaid2_chipmatch = {}
    qaid2_scores    = {}
    aggregate = qreq_.qparams.aggregate
    alpha     = qreq_.qparams.alpha
    thresh    = qreq_.qparams.thresh
    lbl = 'asmk query: ' if aggregate else 'smk query: '
    mark, end_ = utool.log_progress(lbl, len(qaids), flushfreq=1,
                                    writefreq=1, with_totaltime=True,
                                    backspace=False)
    withinfo = True
    for count, qaid in enumerate(qaids):
        mark(count)
        daid2_score, daid2_chipmatch = query_inverted_index(
            annots_df, qaid, invindex, withinfo, aggregate)
        qaid2_scores[qaid]    = daid2_score
        qaid2_chipmatch[qaid] = daid2_chipmatch
    end_()
    try:
        filt2_meta = {}
        qaid2_qres_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch, filt2_meta, qreq_)
    except Exception as ex:
        utool.printex(ex)
        utool.qflag()
        raise
    #,
    #qaid2_scores=qaid2_scores)
    return qaid2_qres_


if __name__ == '__main__':
    def main():
        from ibeis.model.hots.smk import smk_debug
        from ibeis.model.hots.smk import smk_match
        from ibeis.model.hots import query_request
        ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_internals()
        qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
        qaid2_qres_ = smk_match.query_smk(annots_df, invindex, qreq_)
        qres = qaid2_qres_[qaids[0]]
        fig = qres.show_top(ibs)
        fig.show()
    main()
    from plottool import draw_func2 as df2
    exec(df2.present())
