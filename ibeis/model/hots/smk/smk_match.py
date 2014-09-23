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
    >>> ibs, annots_df, qaid, invindex = smk_debug.testdata_query_repr()
    >>> aggregate = ibs.cfg.query_cfg.smk_cfg.aggregate
    >>> alpha = ibs.cfg.query_cfg.smk_cfg.alpha
    >>> thresh = ibs.cfg.query_cfg.smk_cfg.thresh
    >>> tup = smk_index.compute_query_repr(annots_df, qaid, invindex, aggregate, alpha, thresh)
    >>> wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma = tup
    >>> assert smk_debug.check_wx2_rvecs(wx2_qrvecs), 'has nan'
    >>> query_inverted_index(annots_df, qaid, invindex)
    >>> withinfo = False
    >>> daid2_totalscore = query_inverted_index(annots_df, qaid, invindex, withinfo=withinfo)
    """
    #from ibeis.model.hots.smk import smk_index
    # Get query words / residuals
    tup = smk_index.compute_query_repr(annots_df, qaid, invindex, aggregate,
                                       alpha, thresh)
    wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma = tup
    # Compute match kernel for all database aids
    #match_kernel = utool.cached_func('match_kernel', appname='smk',
    #                                 key_argx=None)(smk_core.match_kernel)
    match_kernel = smk_core.match_kernel
    _args = (wx2_qrvecs, wx2_qaids, wx2_qfxs, query_gamma, invindex, withinfo,
             alpha, thresh)
    daid2_totalscore, daid2_chipmatch = match_kernel(*_args)
    # Build chipmatches if daid2_wx2_scoremat is not None
    if withinfo:
        assert daid2_chipmatch is not None
        return daid2_totalscore, daid2_chipmatch
    else:
        return daid2_totalscore


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
    """
    qaids = qreq_.get_external_qaids()
    qaid2_chipmatch = {}
    qaid2_scores = {}
    aggregate = qreq_.qparams.aggregate
    lbl = 'asmk query: ' if aggregate else 'smk query: '
    mark, end_ = utool.log_progress(lbl, len(qaids), flushfreq=1,
                                    writefreq=1, with_totaltime=True,
                                    backspace=False)
    withinfo = True
    for count, qaid in enumerate(qaids):
        mark(count)
        daid2_score, daid2_chipmatch = query_inverted_index(
            annots_df, qaid, invindex, withinfo, aggregate)
        qaid2_scores[qaid] = daid2_score
        qaid2_chipmatch[qaid] = daid2_chipmatch
    end_()
    qaid2_qres_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch, {}, qreq_)
    #,
    #qaid2_scores=qaid2_scores)
    return qaid2_qres_


if __name__ == '__main__':

    def main():
        from ibeis.model.hots.smk import smk_debug
        from ibeis.model.hots.smk import smk_match
        from ibeis.model.hots import query_request
        ibs, annots_df, daids, qaids, invindex  = smk_debug.testdata_internals()
        qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
        qaid2_qres_ = smk_match.query_smk(annots_df, invindex, qreq_)
        return qaid2_qres_
    main()
