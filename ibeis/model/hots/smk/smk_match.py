from __future__ import absolute_import, division, print_function
#import six
from six.moves import zip, range, map  # NOQA
import utool
import numpy as np
from ibeis.model.hots.smk import smk_index
from ibeis.model.hots.smk import smk_core
#from six.moves import zip
#from ibeis.model.hots.smk.hstypes import INTEGER_TYPE
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_match]')


@profile
def query_inverted_index(annots_df, qaid, invindex, withinfo=True,
                         aggregate=False, alpha=3, thresh=0, nAssign=1,
                         can_match_self=False):
    """
    Example:
        >>> from ibeis.model.hots.smk.smk_match import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_internals_full()
        >>> qaid = qaids[0]
        >>> aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
        >>> alpha     = ibs.cfg.query_cfg.smk_cfg.alpha
        >>> thresh    = ibs.cfg.query_cfg.smk_cfg.thresh
        >>> withinfo = True
        >>> daid2_totalscore, daid2_chipmatch = query_inverted_index(annots_df, qaid, invindex, withinfo, aggregate, alpha, thresh)
    """
    #from ibeis.model.hots.smk import smk_index
    # Get query words / residuals
    qindex = smk_index.new_qindex(annots_df, qaid, invindex, aggregate, alpha,
                                  thresh, nAssign)
    (wx2_qrvecs, wx2_maws, wx2_qaids, wx2_qfxs, query_gamma) = qindex
    if utool.DEBUG2:
        from ibeis.model.hots.smk import smk_debug
        qfx2_vec = annots_df['vecs'][qaid]
        smk_debug.invindex_dbgstr(invindex)
        assert smk_debug.check_wx2_rvecs2(
            invindex, wx2_qrvecs, wx2_qfxs, qfx2_vec), 'bad qindex'
        assert smk_debug.check_wx2_rvecs2(invindex), 'bad invindex'
    # Compute match kernel for all database aids
    kernel_args = (wx2_qrvecs, wx2_maws, wx2_qaids, wx2_qfxs, query_gamma,
                   invindex, withinfo, alpha, thresh)
    daid2_totalscore, daid2_chipmatch = smk_core.match_kernel(*kernel_args)  # 54 %
    # Prevent self matches
    #can_match_self = not utool.get_argflag('--noself')
    can_match_self = utool.get_argflag('--self-match')
    if (not can_match_self) and qaid in daid2_totalscore:
        # If we cannot do self-matches
        daid2_totalscore[qaid] = 0
        daid2_chipmatch[0][qaid] = np.empty((0, 2), dtype=np.int32)
        daid2_chipmatch[1][qaid] = np.empty((0), dtype=np.float32)
        daid2_chipmatch[2][qaid] = np.empty((0), dtype=np.int32)
    # Build chipmatches if daid2_wx2_scoremat is not None
    if utool.DEBUG2:
        from ibeis.model.hots.smk import smk_debug
        smk_debug.check_daid2_chipmatch(daid2_chipmatch)
    return daid2_totalscore, daid2_chipmatch


@utool.indent_func('[smk_query]')
#@utool.memprof
def selective_match_kernel(qreq_):
    """
    ibeis query interface

    Example:
        >>> from ibeis.model.hots.smk.smk_match import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_match
        >>> from ibeis.model.hots.smk import smk_debug
        >>> from ibeis.model.hots import query_request
        >>> ibs, annots_df, daids, qaids, invindex = smk_debug.testdata_internals_full()
        >>> qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
        >>> qaid2_qres_ = smk_match.selective_match_kernel(qreq_)

    Dev::
        qres = qaid2_qres_[qaids[0]]
        fig = qres.show_top(ibs)
    """
    memtrack = utool.MemoryTracker('selective_match_kernel')
    daids = qreq_.get_external_daids()
    qaids = qreq_.get_external_qaids()
    ibs   = qreq_.ibs
    # Params
    nWords    = qreq_.qparams.nWords
    aggregate = qreq_.qparams.aggregate
    alpha     = qreq_.qparams.alpha
    thresh    = qreq_.qparams.thresh
    nAssign   = qreq_.qparams.nAssign
    # Build ~~Pandas~~ dataframe (or maybe not)
    memtrack.report()
    annots_df = smk_index.make_annot_df(ibs)  # .3%
    if hasattr(qreq_, 'words'):
        # Hack
        words = qreq_.words
        invindex = qreq_.invindex
    else:
        print('\n\n+--- QREQ NEEDS TO LOAD VOCAB --- ')
        # Load vocabulary
        taids = ibs.get_valid_aids()  # exemplar
        words = smk_index.learn_visual_words(annots_df, taids, nWords)
        memtrack.report('learned visual words')
        #utool.embed()
        # Index database annotations
        invindex = smk_index.index_data_annots(annots_df, daids, words, aggregate=aggregate)
        memtrack.report('indexed database annotations')
        print('L___ FINISHED LOADING VOCAB ___\n')
    withinfo = True
    # Progress
    lbl = 'asmk query: ' if aggregate else 'smk query: '
    logkw = dict(flushfreq=1, writefreq=1, with_totaltime=True, backspace=False)
    mark, end_ = utool.log_progress(lbl, len(qaids), **logkw)
    # Output
    qaid2_chipmatch = {}
    qaid2_scores    = {}
    #return {}, {}
    memtrack.report('SMK Init')

    # Foreach query annotation
    for count, qaid in enumerate(qaids):
        mark(count)
        daid2_score, daid2_chipmatch = query_inverted_index(annots_df, qaid,
                                                            invindex, withinfo,
                                                            aggregate, alpha,
                                                            thresh, nAssign)
        qaid2_scores[qaid]    = daid2_score
        qaid2_chipmatch[qaid] = daid2_chipmatch
        memtrack.report('Query')
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
