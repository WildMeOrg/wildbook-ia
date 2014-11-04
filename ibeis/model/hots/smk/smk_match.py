from __future__ import absolute_import, division, print_function
#import six
from six.moves import zip, range, map  # NOQA
import utool
import numpy as np
from ibeis.model.hots.smk import smk_index
from ibeis.model.hots.smk import smk_repr
from ibeis.model.hots.smk import smk_core
#from six.moves import zip
#from ibeis.model.hots.hstypes import INTEGER_TYPE
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk_match]')


DEBUG_SMK = utool.DEBUG2 or utool.get_argflag('--debug-smk')


@utool.indent_func('[smk_query]')
#@utool.memprof
@profile
def execute_smk_L5(qreq_):
    """
    ibeis query interface

    Example:
        >>> from ibeis.model.hots.smk.smk_match import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_match
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, qreq_ = smk_debug.testdata_internals_full()
        >>> qaid2_scores, qaid2_chipmatch = smk_match.execute_smk_L5(qreq_)

    Dev::
        from ibeis.model.hots import pipeline
        filt2_meta = {}
        # Get both spatial verified and not
        qaid2_chipmatch_FILT_ = qaid2_chipmatch
        qaid2_chipmatch_SVER_ = pipeline.spatial_verification(qaid2_chipmatch_FILT_, qreq_)
        qaid2_qres_FILT_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch_FILT_, filt2_meta, qreq_)
        qaid2_qres_SVER_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch_SVER_, filt2_meta, qreq_)
        qres_FILT = qaid2_qres_FILT_[qaids[0]]
        qres_SVER = qaid2_qres_SVER_[qaids[0]]
        fig1 = qres_FILT.show_top(ibs, fnum=1, figtitle='filt')
        fig2 = qres_SVER.show_top(ibs, fnum=2, figtitle='sver')
        fig1.show()
        fig2.show()

    CommandLine::
        python -m memory_profiler dev.py --db PZ_Mothers -t smk2 --allgt --index 0
        python dev.py -t smk2 --allgt --db GZ_ALL
        python dev.py -t smk2 --allgt --db GZ_ALL
        python dev.py -t smk2 --allgt --db GZ_ALL --index 2:10 --vf --va
        python dev.py -t smk2 --allgt --db GZ_ALL --index 2:10 --vf --va --print-cfgstr
        python dev.py -t smk2 --allgt --db GZ_ALL --index 2:20 --vf --va
        python dev.py -t smk2 --allgt --db GZ_ALL --noqcache --index 2:20 --va --vf
        python dev.py -t smk2 --allgt --db PZ_Master0 && python dev.py -t smk3 --allgt --db PZ_Master0
        python dev.py -t smk2 --allgt --db PZ_Master0 --index 2:10 --va
        python dev.py -t smk2 --allgt --db PZ_Mothers --index 20:30
        python dev.py -t smk2 --allgt --db PZ_Mothers --noqcache --index 18:20 --super-strict --va
        python dev.py -t smk2 --db PZ_Master0 --qaid 7199 --va --quality --vf --noqcache
        python dev.py -t smk3 --allgt --db GZ_ALL --index 2:10 --vf --va
        python dev.py -t smk5 --allgt --db PZ_Master0 --noqcache ; python dev.py -t smk5 --allgt --db GZ_ALL --noqcache
        python dev.py -t smkd --allgt --db PZ_Mothers --index 1:3 --va --quality --vf --noqcache

        python dev.py -t smk_8k --allgt --db PZ_Mothers --index 20:30 --va --vf
        python dev.py -t smk_8k --allgt --db PZ_Mothers --index 20:30 --echo-hardcase
        python dev.py -t smk_8k --allgt --db PZ_Mothers --index 20:30 --vh
        python dev.py -t smk_8k_compare --allgt --db PZ_Mothers --index 20:30 --view-hard
    """
    memtrack = utool.MemoryTracker('[SMK ENTRY]')
    qaids = qreq_.get_external_qaids()
    ibs   = qreq_.ibs
    # Params
    qparams = qreq_.qparams
    memtrack.report('[SMK PREINIT]')
    # Build ~~Pandas~~ dataframe (or maybe not)
    annots_df = smk_repr.make_annot_df(ibs)
    words, invindex = prepare_qreq(qreq_, annots_df, memtrack)
    withinfo = True

    # Execute smk for each query
    memtrack.report('[SMK QREQ INITIALIZED]')
    print('[SMK_MEM] invindex is using ' + utool.get_object_size_str(invindex))
    print('[SMK_MEM] qreq_ is using ' + utool.get_object_size_str(qreq_))

    if utool.DEBUG2:
        from ibeis.model.hots.smk import smk_debug
        smk_debug.invindex_dbgstr(invindex)

    qaid2_scores, qaid2_chipmatch = execute_smk_L4(annots_df, qaids, invindex, qparams, withinfo)
    memtrack.report('[SMK QREQ FINISHED]')
    return qaid2_scores, qaid2_chipmatch


#@utool.memprof
def prepare_qreq(qreq_, annots_df, memtrack):
    """ Called if pipeline did not setup qreq correctly """
    print('\n\n+--- QREQ NEEDS TO LOAD VOCAB --- ')
    if hasattr(qreq_, 'words'):
        # Hack
        raise NotImplementedError('pipeline still isnt fully ready for smk')
        words = qreq_.words
        invindex = qreq_.invindex
    else:
        # Load vocabulary
        qparams = qreq_.qparams
        daids = qreq_.get_external_daids()
        words = smk_index.learn_visual_words(annots_df, qreq_, memtrack=memtrack)
        memtrack.report('[SMK LEARN VWORDS]')
        # Index database annotations
        with_internals = True
        invindex = smk_repr.index_data_annots(annots_df, daids, words, qparams,
                                               with_internals, memtrack)
        memtrack.report('[SMK INDEX ANNOTS]')
        print('L___ FINISHED LOADING VOCAB ___\n')
    return words, invindex


@profile
def execute_smk_L4(annots_df, qaids, invindex, qparams, withinfo):
    """
    Loop over execute_smk_L3

    CommandLine:
        python dev.py -t smk --allgt --db PZ_Mothers --index 1:3 --noqcache --va --vf
    """
    # Progress
    lbl = 'ASMK query: ' if qparams.aggregate else 'SMK query: '
    logkw = dict(flushfreq=1, writefreq=1, with_totaltime=True, backspace=False)
    mark, end_ = utool.log_progress(lbl, len(qaids), **logkw)
    # Output
    qaid2_chipmatch = {}
    qaid2_scores    = {}
    # Foreach query annotation
    for count, qaid in enumerate(qaids):
        mark(count)
        tup = execute_smk_L3(annots_df, qaid, invindex, qparams, withinfo)
        daid2_score, daid2_chipmatch = tup
        qaid2_scores[qaid]    = daid2_score
        qaid2_chipmatch[qaid] = daid2_chipmatch
        #memtrack.report('[SMK SINGLE QUERY]')
    end_()
    if DEBUG_SMK:
        from ibeis.model.hots.smk import smk_debug
        smk_debug.check_qaid2_chipmatch(qaid2_chipmatch, qaids)
    return qaid2_scores, qaid2_chipmatch


@profile
def execute_smk_L3(annots_df, qaid, invindex, qparams, withinfo=True):
    """
    Executes a single smk query

    Example:
        >>> from ibeis.model.hots.smk.smk_match import *  # NOQA
        >>> from ibeis.model.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, qreq_ = smk_debug.testdata_internals_full()
        >>> qaid = qaids[0]
        >>> qparams = qreq_.qparams
        >>> withinfo = True
        >>> daid2_totalscore, daid2_chipmatch = execute_smk_L3(annots_df, qaid, invindex, qparams, withinfo)
    """
    #from ibeis.model.hots.smk import smk_index
    # Get query words / residuals
    qindex = smk_repr.new_qindex(annots_df, qaid, invindex, qparams)
    # Compute match kernel for all database aids
    daid2_totalscore, daid2_chipmatch = smk_core.match_kernel_L2(qindex, invindex, qparams, withinfo)  # 54 %
    # Prevent self matches
    allow_self_match = qparams.allow_self_match
    #utool.get_argflag('--self-match')
    if (not allow_self_match) and qaid in daid2_totalscore:
        # If we cannot do self-matches
        daid2_totalscore[qaid] = 0
        daid2_chipmatch[0][qaid] = np.empty((0, 2), dtype=np.int32)
        daid2_chipmatch[1][qaid] = np.empty((0), dtype=np.float32)
        daid2_chipmatch[2][qaid] = np.empty((0), dtype=np.int32)
    # Build chipmatches if daid2_wx2_scoremat is not None
    if DEBUG_SMK:
        from ibeis.model.hots.smk import smk_debug
        smk_debug.check_daid2_chipmatch(daid2_chipmatch)
    return daid2_totalscore, daid2_chipmatch


if __name__ == '__main__':
    def main():
        from ibeis.model.hots.smk import smk_debug
        from ibeis.model.hots.smk import smk_match
        from ibeis.model.hots import pipeline
        ibs, taids, daids, qaids, qreq_ = smk_debug.testdata_ibeis2()
        qaid2_scores, qaid2_chipmatch = smk_match.execute_smk_L5(qreq_)
        filt2_meta = {}
        qaid2_qres_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch, filt2_meta, qreq_)
        qres = qaid2_qres_[qaids[0]]
        fig = qres.show_top(ibs)
        fig.show()
    main()
    from plottool import draw_func2 as df2
    exec(df2.present())
