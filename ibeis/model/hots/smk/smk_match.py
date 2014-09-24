from __future__ import absolute_import, division, print_function
#import six
import utool
import numpy as np
from ibeis.model.hots.smk import smk_index
from ibeis.model.hots.smk import smk_core
#from six.moves import zip
#from ibeis.model.hots.smk.hstypes import INTEGER_TYPE
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_match]')


@profile
def query_inverted_index(annots_df, qaid, invindex, withinfo=True,
                         aggregate=False, alpha=3, thresh=0):
    """
    Total time: 10.5774 s

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
                                              aggregate, alpha, thresh)  # 45 %
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
    daid2_totalscore, daid2_chipmatch = smk_core.match_kernel(*kernel_args)  # 54 %
    # Prevent self matches
    can_match_self = False
    if (not can_match_self) and qaid in daid2_totalscore:
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
def selective_match_kernel(qreq_):
    """
    Total time: 21.3564 s

    ibeis query interface
    >>> from ibeis.model.hots.smk.smk_match import *  # NOQA
    >>> from ibeis.model.hots.smk import smk_match
    >>> from ibeis.model.hots.smk import smk_debug
    >>> from ibeis.model.hots import query_request
    >>> ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_internals()
    >>> qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
    >>> qaid2_qres_ = smk_match.selective_match_kernel(qreq_)

    qres = qaid2_qres_[qaids[0]]
    fig = qres.show_top(ibs)
    """
    daids = qreq_.get_external_daids()
    qaids = qreq_.get_external_qaids()
    ibs   = qreq_.ibs
    taids = ibs.get_valid_aids()  # exemplar
    # Params
    nWords    = qreq_.qparams.nWords
    aggregate = qreq_.qparams.aggregate
    alpha     = qreq_.qparams.alpha
    thresh    = qreq_.qparams.thresh
    # Build Pandas dataframe (or maybe not)
    annots_df = smk_index.make_annot_df(ibs)  # .3%
    # Load vocabulary
    words = smk_index.learn_visual_words(annots_df, taids, nWords)
    # Index database annotations
    invindex = smk_index.index_data_annots(annots_df, daids, words,
                                           aggregate=aggregate)  # 18.5%
    withinfo = True
    # Progress
    lbl = 'asmk query: ' if aggregate else 'smk query: '
    logkw = dict(flushfreq=1, writefreq=1, with_totaltime=True, backspace=False)
    mark, end_ = utool.log_progress(lbl, len(qaids), **logkw)
    # Output
    qaid2_chipmatch = {}
    qaid2_scores    = {}
    # Foreach query annotation
    for count, qaid in enumerate(qaids):
        mark(count)
        daid2_score, daid2_chipmatch = query_inverted_index(
            annots_df, qaid, invindex, withinfo, aggregate, alpha, thresh)  # 81.2%
        qaid2_scores[qaid]    = daid2_score
        qaid2_chipmatch[qaid] = daid2_chipmatch
    end_()
    return qaid2_scores, qaid2_chipmatch


if __name__ == '__main__':
    def main():
        from ibeis.model.hots.smk import smk_debug
        from ibeis.model.hots.smk import smk_match
        from ibeis.model.hots import query_request
        from ibeis.model.hots import pipeline
        ibs, taids, daids, qaids = smk_debug.testdata_ibeis2()
        qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
        qreq_.ibs = ibs
        qaid2_scores, qaid2_chipmatch = smk_match.selective_match_kernel(qreq_)
        filt2_meta = {}
        qaid2_qres_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch, filt2_meta, qreq_)
        qres = qaid2_qres_[qaids[0]]
        fig = qres.show_top(ibs)
        fig.show()
    main()
    from plottool import draw_func2 as df2
    exec(df2.present())
