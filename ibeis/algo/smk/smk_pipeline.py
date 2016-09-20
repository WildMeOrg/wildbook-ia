# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import dtool
import six
import utool as ut
import numpy as np
from six.moves import zip
from ibeis.algo.smk import match_chips5 as mc5
from ibeis.algo.smk import vocab_indexer
from ibeis import core_annots
from ibeis.algo import Config as old_config
(print, rrr, profile) = ut.inject2(__name__)


class MatchHeuristicsConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('can_match_self', False),
        ut.ParamInfo('can_match_samename', True),
        ut.ParamInfo('can_match_sameimg', False),
    ]


class SMKRequestConfig(dtool.Config):
    """ Figure out how to do this """
    _param_info_list = [
        ut.ParamInfo('proot', 'smk'),
        ut.ParamInfo('smk_alpha', 3.0),
        ut.ParamInfo('smk_thresh', 0.0),
        #ut.ParamInfo('smk_thresh', -1.0),
        ut.ParamInfo('agg', True),
    ]
    _sub_config_list = [
        core_annots.ChipConfig,
        core_annots.FeatConfig,
        old_config.SpatialVerifyConfig,
        vocab_indexer.VocabConfig,
        vocab_indexer.InvertedIndexConfig,
        MatchHeuristicsConfig,
    ]


@ut.reloadable_class
class InvertedAnnots2(object):
    def __init__(inva, aids, qreq_):
        print('Loading up inverted assigments')
        colnames = ('wx_list', 'fxs_list', 'maws_list', 'agg_rvecs', 'agg_flags')
        tablename = 'inverted_agg_assign'
        input_tuple = (aids, [qreq_.daids])
        table = qreq_.ibs.depc[tablename]
        tbl_rowids = qreq_.ibs.depc.get_rowids(tablename, input_tuple, config=qreq_.config)
        print('Reading data')
        #if ut.is_developer():
        #    cacher = ut.Cacher(
        #        fname=qreq_.ibs.get_dbname(),
        #        cfgstr=ut.hashstr_arr27(tbl_rowids, 'group'),
        #        cache_dir=ut.truepath('~/Desktop')
        #    )
        #    groups = cacher.tryload()
        #    if groups is None:
        #        print('Developer cache miss')
        #        groups = table.get_row_data(tbl_rowids, colnames)
        #        cacher.save(groups)
        #else:
        groups = table.get_row_data(tbl_rowids, colnames, showprog=True)
        with ut.Timer('Formating inverted assigments'):
            inva.aids = aids
            # 431.61 vs 143.87 MB here
            inva.wx_lists = ut.itake_column(groups, 0)
            inva.wx_lists = [np.array(wx_list, dtype=np.int32) for wx_list in inva.wx_lists]
            # Is this better to use?
            # As nested lists: 471.35 MB
            # As nested ndarrays: 157.12 MB
            inva.fxs_lists = ut.itake_column(groups, 1)
            inva.fxs_lists = [[np.array(fxs, dtype=np.uint16) for fxs in fxs_list] for fxs_list in inva.fxs_lists]
            # [ut.lmap(np.array, fx_list) for fx_list in x.fxs_lists]
            inva.maws_lists = ut.itake_column(groups, 2)
            inva.maws_lists = [[np.array(m, dtype=np.float32)
                                for m in maws] for maws in inva.maws_lists]
            inva.agg_rvecs = ut.take_column(groups, 3)
            inva.agg_flags = ut.take_column(groups, 4)
            # less memory hogs
            inva.aid_to_idx = ut.make_index_lookup(inva.aids)
            inva.int_rvec = qreq_.qparams.int_rvec
            inva.gamma_list = None
            # Inverted list
            inva.wx_to_idf = None
            inva.wx_to_aids = None

    def __getstate__(inva):
        state = inva.__dict__
        return state

    def __setstate__(inva, state):
        inva.__dict__.update(**state)

    def get_size_info(inva):
        import sys
        def get_homog_list_nbytes_scalar(list_scalar):
            if list_scalar is None:
                return 0
            if len(list_scalar) == 0:
                return 0
            else:
                val = list_scalar[0]
                item_nbytes = ut.get_object_nbytes(val)
                return item_nbytes * len(list_scalar)

        def get_homog_list_nbytes_nested(list_nested):
            if list_nested is None:
                return 0
            if len(list_nested) == 0:
                return 0
            else:
                val = list_nested[0]
                if isinstance(val, np.ndarray):
                    nbytes = sum(sys.getsizeof(v) for v in list_nested)
                    #item_nbytes = sum(v.nbytes for v in list_nested)
                else:
                    nest_nbytes = sys.getsizeof(val) * len(list_nested)
                    totals = sum(ut.lmap(len, list_nested))
                    item_nbytes = sys.getsizeof(val[0]) * totals
                    nbytes = nest_nbytes + item_nbytes
                return nbytes

        def get_homog_dict_nbytes_nested(dict_nested):
            if dict_nested is None:
                return 0
            wxkeybytes = get_homog_list_nbytes_scalar(list(dict_nested.keys()))
            wxvalbytes = get_homog_list_nbytes_nested(list(dict_nested.values()))
            wxbytes = wxkeybytes + wxvalbytes + sys.getsizeof(dict_nested)
            return wxbytes

        def get_homog_dict_nbytes_scalar(dict_scalar):
            if dict_scalar is None:
                return 0
            wxkeybytes = get_homog_list_nbytes_scalar(list(dict_scalar.keys()))
            wxvalbytes = get_homog_list_nbytes_scalar(list(dict_scalar.values()))
            wxbytes = wxkeybytes + wxvalbytes + sys.getsizeof(dict_scalar)
            return wxbytes

        sizes = {
            'aids'       : get_homog_list_nbytes_scalar(inva.aids),
            'wx_lists'   : get_homog_list_nbytes_nested(inva.wx_lists),
            'fxs_lists'  : get_homog_list_nbytes_nested(inva.fxs_lists),
            'maws_lists' : get_homog_list_nbytes_nested(inva.maws_lists),
            'agg_rvecs'  : get_homog_list_nbytes_nested(inva.agg_rvecs),
            'agg_flags'  : get_homog_list_nbytes_nested(inva.agg_flags),
            'aid_to_idx' : get_homog_dict_nbytes_scalar(inva.aid_to_idx),
            'gamma_list' : get_homog_list_nbytes_scalar(inva.gamma_list),
            'wx_to_aids' : get_homog_dict_nbytes_nested(inva.wx_to_aids),
            'wx_to_idf'  : get_homog_dict_nbytes_scalar(inva.wx_to_idf),
        }
        return sizes

    def print_size_info(inva):
        sizes = inva.get_size_info()
        sizes = ut.sort_dict(sizes, value_key=ut.identity)
        total_nbytes =  sum(sizes.values())
        print(ut.align(ut.repr3(ut.map_dict_vals(ut.byte_str2, sizes), strvals=True), ':'))
        print('total_nbytes = %r' % (ut.byte_str2(total_nbytes),))

    def get_nbytes(inva):
        sizes = inva.get_size_info()
        total_nbytes =  sum(sizes.values())
        return total_nbytes

    @profile
    def compute_gammas(inva, wx_to_idf, alpha, thresh):
        gamma_list = []
        _iter = zip(inva.wx_lists, inva.agg_rvecs, inva.agg_flags)
        _prog = ut.ProgPartial(nTotal=len(inva.wx_lists), bs=True, lbl='gamma', adjust=True)
        for wx_list, phiX_list, flagsX_list in _prog(_iter):
            if inva.int_rvec:
                phiX_list = vocab_indexer.uncast_residual_integer(phiX_list)
            gammaX = gamma(wx_list, phiX_list, flagsX_list, wx_to_idf, alpha,
                           thresh)
            gamma_list.append(gammaX)
        return gamma_list

    @profile
    def compute_idf(inva):
        with ut.Timer('Computing idf'):
            num_docs_total = len(inva.aids)
            # idf denominator (the num of docs containing a word for each word)
            # The max(maws) to denote the probab that this word indexes an annot
            #wx_to_ndocs = ut.DefaultValueDict(0)
            #wx_to_ndocs = ut.ddict(lambda: 0)

            wx_list = sorted(inva.wx_to_aids.keys())
            wx_to_ndocs = {wx: 0.0 for wx in wx_list}

            if True:
                # Unweighted documents
                wx_to_ndocs = {wx: len(set(aids))
                               for wx, aids in inva.wx_to_aids.items()}
            else:
                # Weighted documents
                for wx, maws in zip(ut.iflatten(inva.wx_lists), ut.iflatten(inva.maws_lists)):
                    # Determine how many documents use each word
                    wx_to_ndocs[wx] += min(1.0, sum(maws))

            #wx_list = sorted(wx_to_ndocs.keys())
            ndocs_arr = np.array(ut.take(wx_to_ndocs, wx_list), dtype=np.float)
            # Typically for IDF, 1 is added to the denom to prevent divide by 0
            # We add epsilon to numer and denom to ensure recep is a probability
            out = ndocs_arr
            out = np.add(ndocs_arr, 1, out=out)
            out = np.divide(num_docs_total + 1, ndocs_arr, out=out)
            idf_list = np.log(ndocs_arr, out=out)

            wx_to_idf = dict(zip(wx_list, idf_list))
            wx_to_idf = ut.DefaultValueDict(0, wx_to_idf)
            return wx_to_idf

    @profile
    @ut.memoize
    def get_annot(inva, aid):
        idx = inva.aid_to_idx[aid]
        X = SingleAnnot2(inva, idx)
        return X

    def compute_inverted_list(inva):
        with ut.Timer('Building inverted list'):
            wx_to_aids = ut.ddict(list)
            for aid, wxs in zip(inva.aids, inva.wx_lists):
                for wx in wxs:
                    wx_to_aids[wx].append(aid)
            return wx_to_aids


@ut.reloadable_class
class SingleAnnot2(object):
    def __init__(X, inva, idx):
        X.aid = inva.aids[idx]
        X.wx_list = inva.wx_lists[idx]
        X.fxs_list = inva.fxs_lists[idx]
        X.maws_list = inva.maws_lists[idx]
        X.agg_rvecs = inva.agg_rvecs[idx]
        X.agg_flags = inva.agg_flags[idx]
        X.gamma = inva.gamma_list[idx]
        X.wx_to_idx = ut.make_index_lookup(X.wx_list)
        X.int_rvec = inva.int_rvec

        X.wx_set = set(X.wx_list)

    @property
    def words(X):
        return X.wx_set

    @profile
    def fxs(X, c):
        idx = X.wx_to_idx[c]
        fxs = X.fxs_list[idx]
        return fxs

    @profile
    def maws(X, c):
        idx = X.wx_to_idx[c]
        maws = X.maws_list[idx]
        return maws

    @profile
    def Phi_flags(X, c):
        idx = X.wx_to_idx[c]
        PhiX = X.agg_rvecs[idx]
        if X.int_rvec:
            PhiX = vocab_indexer.uncast_residual_integer(PhiX)
        flags = X.agg_flags[idx]
        return PhiX, flags


@ut.reloadable_class
class SMKRequest(mc5.EstimatorRequest):
    r"""
    qreq_-like object. Trying to work on becoming more scikit-ish

    CommandLine:
        python -m ibeis.algo.smk.smk_pipeline SMKRequest --profile
        python -m ibeis.algo.smk.smk_pipeline SMKRequest --show

        python -m ibeis draw_rank_cdf --db PZ_MTEST --show \
            -p :proot=smk,num_words=[64000,8000,4000],nAssign=[1,2,4],sv_on=[True,False] \
                default:proot=vsmany,sv_on=[True,False] \
            -a default:qmingt=2

        python -m ibeis draw_rank_cdf --db PZ_MTEST --show \
            -p :proot=smk,num_words=[64000],nAssign=[1],sv_on=[True] \
                default:proot=vsmany,sv_on=[True] \
            -a default:qmingt=2

        python -m ibeis draw_rank_cdf --db PZ_Master1 --show \
            -p :proot=smk,num_words=[64000],nAssign=[1],sv_on=[False] \
            -a ctrl:qmingt=2

        python -m ibeis draw_rank_cdf --db PZ_Master1 \
            -p :proot=smk,num_words=[64000],nAssign=[1],sv_on=[True] \
            -a ctrl:qmingt=2,qindex=60:80 --profile

        python -m ibeis draw_rank_cdf --db GZ_ALL \
            -p :proot=smk,num_words=[64000],nAssign=[1],sv_on=[True] \
            -a ctrl:qmingt=2,qindex=40:60 --profile

    Example:
        >>> from ibeis.algo.smk.smk_pipeline import *  # NOQA
        >>> import ibeis
        >>> ibs, aid_list = ibeis.testdata_aids(defaultdb='PZ_MTEST')
        >>> qaids = aid_list[0:2]
        >>> daids = aid_list[:]
        >>> config = {'nAssign': 2, 'num_words': 64000, 'sv_on': True}
        >>> qreq_ = SMKRequest(ibs, qaids, daids, config)
        >>> qreq_.ensure_data()
        >>> cm_list = qreq_.execute()
        >>> #cm_list = qreq_.execute_pipeline()
        >>> ut.quit_if_noshow()
        >>> ut.qt4ensure()
        >>> cm_list[0].ishow_analysis(qreq_, fnum=1, viz_name_score=False)
        >>> cm_list[1].ishow_analysis(qreq_, fnum=2, viz_name_score=False)
        >>> ut.show_if_requested()

    """
    def __init__(qreq_, ibs=None, qaids=None, daids=None, config=None):
        super(SMKRequest, qreq_).__init__()
        if config is None:
            config = {}

        qreq_.ibs = ibs
        qreq_.qaids = qaids
        qreq_.daids = daids

        qreq_.config = config

        #qreq_.vocab = None
        #qreq_.dinva = None

        qreq_.qinva = None
        qreq_.dinva = None
        qreq_.smk = SMK()

        # Hack to work with existing hs code
        qreq_.stack_config = SMKRequestConfig(**config)
        # Flat config
        qreq_.qparams = dtool.base.StackedConfig([
            dict(qreq_.stack_config.parse_items())
        ])
        #    # TODO: add vocab, inva, features
        qreq_.cachedir = ut.ensuredir((ibs.cachedir, 'smk'))

    def ensure_data(qreq_):
        print('Ensure data for %s' % (qreq_,))

        memtrack = ut.MemoryTracker()
        #qreq_.cachedir = ut.ensuredir((ibs.cachedir, 'smk'))
        qreq_.ensure_nids()

        def make_cacher(name, cfgstr=None):
            if cfgstr is None:
                cfgstr = ut.hashstr27(qreq_.get_cfgstr())
            if False and ut.is_developer():
                return ut.Cacher(
                    fname=name + '_' + qreq_.ibs.get_dbname(),
                    cfgstr=cfgstr,
                    cache_dir=ut.ensuredir(ut.truepath('~/Desktop/smkcache'))
                )
            else:
                wrp = ut.DynStruct()
                def ensure(func):
                    return func()
                wrp.ensure = ensure
                return wrp

        depc = qreq_.ibs.depc
        inva_pcfgstr = depc.stacked_config(
            None, 'inverted_agg_assign', config=qreq_.config).get_cfgstr()
        dannot_vuuid = qreq_.ibs.get_annot_hashid_visual_uuid(qreq_.daids).strip('_')
        qannot_vuuid = qreq_.ibs.get_annot_hashid_visual_uuid(qreq_.qaids).strip('_')
        tannot_vuuid = dannot_vuuid
        dannot_suuid = qreq_.ibs.get_annot_hashid_semantic_uuid(qreq_.daids).strip('_')
        qannot_suuid = qreq_.ibs.get_annot_hashid_semantic_uuid(qreq_.qaids).strip('_')

        inva_phashid = ut.hashstr27(inva_pcfgstr + tannot_vuuid)
        dinva_cfgstr = '_'.join([dannot_vuuid, inva_phashid])
        qinva_cfgstr = '_'.join([qannot_vuuid, inva_phashid])

        #vocab = vocab_indexer.new_load_vocab(ibs, qreq_.daids, config)
        dinva_cacher = make_cacher('inva', dinva_cfgstr)
        qinva_cacher = make_cacher('inva', qinva_cfgstr)
        didf_cacher  = make_cacher('didf', dinva_cfgstr)

        gamma_phashid = ut.hashstr27(qreq_.get_pipe_cfgstr() + tannot_vuuid)
        dgamma_cfgstr = '_'.join([dannot_suuid, gamma_phashid])
        qgamma_cfgstr = '_'.join([qannot_suuid, gamma_phashid])
        dgamma_cacher = make_cacher('dgamma', cfgstr=dgamma_cfgstr)
        qgamma_cacher = make_cacher('qgamma', cfgstr=qgamma_cfgstr)

        memtrack.report()

        dinva = dinva_cacher.ensure(
            lambda: InvertedAnnots2(qreq_.daids, qreq_))

        memtrack.report()

        qinva = qinva_cacher.ensure(
            lambda: InvertedAnnots2(qreq_.qaids, qreq_))

        memtrack.report()

        dinva.wx_to_aids = dinva.compute_inverted_list()
        memtrack.report()

        wx_to_idf = didf_cacher.ensure(
            lambda: dinva.compute_idf())
        dinva.wx_to_idf = wx_to_idf
        memtrack.report()

        thresh = qreq_.qparams['smk_thresh']
        alpha = qreq_.qparams['smk_alpha']

        dinva.gamma_list = dgamma_cacher.ensure(
            lambda: dinva.compute_gammas(wx_to_idf, alpha, thresh))
        memtrack.report()

        qinva.gamma_list = qgamma_cacher.ensure(
            lambda: qinva.compute_gammas(wx_to_idf, alpha, thresh))
        memtrack.report()

        qreq_.qinva = qinva
        qreq_.dinva = dinva

        print('loading keypoints')
        if qreq_.qparams.sv_on:
            qreq_.data_kpts = qreq_.ibs.get_annot_kpts(
                qreq_.daids, config2_=qreq_.extern_data_config2)
        memtrack.report()

        print('building aid index')
        qreq_.daid_to_didx = ut.make_index_lookup(qreq_.daids)
        memtrack.report()

    def execute_pipeline(qreq_):
        """
        >>> from ibeis.algo.smk.smk_pipeline import *  # NOQA
        >>> ibs, smk, qreq_ = testdata_smk()
        >>> cm_list = qreq_.execute()
        """
        smk = qreq_.smk
        cm_list = smk.predict_matches(qreq_)
        return cm_list

    def get_qreq_qannot_kpts(qreq_, qaids):
        return qreq_.ibs.get_annot_kpts(
            qaids, config2_=qreq_.extern_query_config2)

    def get_qreq_dannot_kpts(qreq_, daids):
        didx_list = ut.take(qreq_.daid_to_didx, daids)
        return ut.take(qreq_.data_kpts, didx_list)
        #return qreq_.ibs.get_annot_kpts(
        #    daids, config2_=qreq_.extern_data_config2)


@ut.reloadable_class
class SMK(ut.NiceRepr):
    """
    Harness class that controls the execution of the SMK algorithm

    K(X, Y) = gamma(X) * gamma(Y) * sum([Mc(Xc, Yc) for c in words])
    """

    def predict_matches(smk, qreq_, verbose=True):
        """
        >>> from ibeis.algo.smk.smk_pipeline import *  # NOQA
        >>> ibs, smk, qreq_ = testdata_smk()
        >>> verbose = True
        """
        print('Predicting matches')
        #assert qreq_.qinva.vocab is qreq_.dinva.vocab
        #X_list = qreq_.qinva.inverted_annots(qreq_.qaids)
        #Y_list = qreq_.dinva.inverted_annots(qreq_.daids)
        #verbose = 2
        _prog = ut.ProgPartial(lbl='smk query', bs=verbose <= 1, enabled=verbose)
        daids = np.array(qreq_.daids)
        cm_list = [smk.match_single(qaid, daids, qreq_, verbose=verbose > 1)
                   for qaid in _prog(qreq_.qaids)]
        return cm_list

    @profile
    def match_single(smk, qaid, daids, qreq_, verbose=True):
        """
        CommandLine:
            python -m ibeis.algo.smk.smk_pipeline SMK.match_single --profile
            python -m ibeis.algo.smk.smk_pipeline SMK.match_single --show

            python -m ibeis SMK.match_single -a ctrl:qmingt=2 --profile --db PZ_Master1
            python -m ibeis SMK.match_single -a ctrl --profile --db GZ_ALL

        Example:
            >>> # FUTURE_ENABLE
            >>> from ibeis.algo.smk.smk_pipeline import *  # NOQA
            >>> import ibeis
            >>> qreq_ = ibeis.testdata_qreq_(defaultdb='PZ_MTEST')
            >>> ibs = qreq_.ibs
            >>> daids = qreq_.daids
            >>> #ibs, daids = ibeis.testdata_aids(defaultdb='PZ_MTEST', default_set='dcfg')
            >>> qreq_ = SMKRequest(ibs, daids[0:1], daids, {'agg': True,
            >>>                                             'num_words': 64000,
            >>>                                             'sv_on': True})
            >>> qreq_.ensure_data()
            >>> qaid = qreq_.qaids[0]
            >>> daids = qreq_.daids
            >>> daid = daids[1]
            >>> verbose = True
            >>> cm = qreq_.smk.match_single(qaid, daids, qreq_)
            >>> ut.quit_if_noshow()
            >>> ut.qt4ensure()
            >>> cm.ishow_analysis(qreq_)
            >>> ut.show_if_requested()
        """
        from ibeis.algo.hots import chip_match
        from ibeis.algo.hots import pipeline

        alpha  = qreq_.qparams['smk_alpha']
        thresh = qreq_.qparams['smk_thresh']
        agg    = qreq_.qparams['agg']
        nNameShortList  = qreq_.qparams.nNameShortlistSVER
        #nAnnotPerName   = qreq_.qparams.nAnnotPerNameSVER

        sv_on   = qreq_.qparams.sv_on
        if sv_on:
            shortsize = nNameShortList
        else:
            shortsize = None

        X = qreq_.qinva.get_annot(qaid)

        # Determine which database annotations need to be checked
        #with ut.Timer('searching qaid=%r' % (qaid,), verbose=verbose):
        hit_inva_wxs = ut.take(qreq_.dinva.wx_to_aids, X.wx_list)
        hit_daids = np.array(list(set(ut.iflatten(hit_inva_wxs))))

        # Mark impossible daids
        #with ut.Timer('checking impossible daids=%r' % (qaid,), verbose=verbose):
        valid_flags = check_can_match(qaid, hit_daids, qreq_)
        valid_daids = hit_daids.compress(valid_flags)

        shortlist = ut.Shortlist(shortsize)
        #gammaX = smk.gamma(X, wx_to_idf, agg, alpha, thresh)
        _prog = ut.ProgPartial(lbl='smk scoring qaid=%r' % (qaid,),
                               enabled=verbose, bs=True, adjust=True)

        gammaX = X.gamma
        wx_to_idf = qreq_.dinva.wx_to_idf

        #with ut.Timer('scoring', verbose=verbose):
        for daid in _prog(valid_daids):
            Y = qreq_.dinva.get_annot(daid)
            gammaY = Y.gamma
            gammaXY = gammaX * gammaY

            # Words in common define matches
            common_words = sorted(X.words.intersection(Y.words))
            X_idx = ut.take(X.wx_to_idx, common_words)
            Y_idx = ut.take(Y.wx_to_idx, common_words)
            idf_list = ut.take(wx_to_idf, common_words)

            if agg:
                score_list = agg_match_scores(X, Y, X_idx, Y_idx, alpha, thresh)
            else:
                score_list = sep_match_scores(X, Y, X_idx, Y_idx, alpha, thresh)

            score_list *= idf_list
            score_list *= gammaXY
            score = score_list.sum()
            item = (score, score_list, X, Y, X_idx, Y_idx)
            shortlist.insert(item)

        # Build chipmatches for the shortlist results

        #with ut.Timer('build cms', verbose=verbose):
        cm = chip_match.ChipMatch(qaid=qaid, fsv_col_lbls=['smk'])
        cm.daid_list = []
        cm.fm_list = []
        cm.fsv_list = []
        _prog = ut.ProgPartial(lbl='smk build cm qaid=%r' % (qaid,),
                               enabled=verbose, bs=True, adjust=True)
        for item in _prog(shortlist):
            (score, score_list, X, Y, X_idx, Y_idx) = item
            # Only build matches for those that sver will use
            if agg:
                fm, fs = agg_build_matches(X, Y, X_idx, Y_idx, score_list)
            else:
                fm, fs = sep_build_matches(X, Y, X_idx, Y_idx, score_list)
            if len(fm) > 0:
                #assert not np.any(np.isnan(fs))
                daid = Y.aid
                fsv = fs[:, None]
                cm.daid_list.append(daid)
                cm.fm_list.append(fm)
                cm.fsv_list.append(fsv)
        cm._update_daid_index()
        cm.arraycast_self()
        cm.score_maxcsum(qreq_)

        #if False:
        #    cm.assert_self(qreq_=qreq_, verbose=True)

        if sv_on:
            #top_aids = cm.get_name_shortlist_aids(nNameShortList, nAnnotPerName)
            #cm = cm.shortlist_subset(top_aids)
            #with ut.Timer('sver', verbose=verbose):
            # Spatially verify chip matches and rescore
            cm = pipeline.sver_single_chipmatch(qreq_, cm, verbose=verbose)
            cm.score_maxcsum(qreq_)

        return cm


@profile
def agg_match_scores(X, Y, X_idx, Y_idx, alpha, thresh):
    # Agg speedup
    PhiX = X.agg_rvecs.take(X_idx, axis=0)
    PhiY = X.agg_rvecs.take(X_idx, axis=0)
    flagsX = X.agg_flags.take(X_idx, axis=0)
    flagsY = Y.agg_flags.take(Y_idx, axis=0)
    if X.int_rvec:
        PhiX = vocab_indexer.uncast_residual_integer(PhiX)
        PhiY = vocab_indexer.uncast_residual_integer(PhiY)
    u = (PhiX * PhiY).sum(axis=1)
    flags = np.logical_or(flagsX.T[0], flagsY.T[0])
    u[flags] = 1
    score_list = selectivity(u, alpha, thresh, out=u)
    return score_list


@profile
def agg_build_matches(X, Y, X_idx, Y_idx, score_list):
    """
    profile = make_profiler()
    _ = profile(agg_build_matches)(X, Y, X_idx, Y_idx, score_list)
    print(get_profile_text(profile)[0])

    %timeit agg_build_matches(X, Y, X_idx, Y_idx, score_list)

    """
    # Build feature matches
    X_fxs = ut.take(X.fxs_list, X_idx)
    Y_fxs = ut.take(Y.fxs_list, Y_idx)

    X_maws = ut.take(X.maws_list, X_idx)
    Y_maws = ut.take(Y.maws_list, Y_idx)

    # Spread word score according to contriubtion (maw) weight
    unflat_fs = [maws1[:, None].dot(maws2[:, None].T).ravel()
                 for maws1, maws2 in zip(X_maws, Y_maws)]
    factor_list = np.array([contrib.sum() for contrib in unflat_fs],
                           dtype=np.float32)
    factor_list = np.multiply(factor_list, score_list, out=factor_list)
    for contrib, factor in zip(unflat_fs, factor_list):
        np.multiply(contrib, factor, out=contrib)

    # itertools.product seems fastest for small arrays
    unflat_fm = (ut.product(fxs1, fxs2)
                 for fxs1, fxs2 in zip(X_fxs, Y_fxs))

    fm = np.array(ut.flatten(unflat_fm), dtype=np.int32)
    fs = np.array(ut.flatten(unflat_fs), dtype=np.float32)
    isvalid = np.greater(fs, 0)
    fm = fm.compress(isvalid, axis=0)
    fs = fs.compress(isvalid, axis=0)
    return fm, fs


@profile
def sep_match_scores(X, Y, X_idx, Y_idx, alpha, thresh):
    raise NotImplementedError('sep version not finished')
    # Agg speedup
    phisX = X.rvecs.take(X_idx, axis=0)
    phisY = X.rvecs.take(X_idx, axis=0)
    flagsX = X.flags.take(X_idx, axis=0)
    flagsY = Y.flags.take(Y_idx, axis=0)
    scores_list = []
    for phiX, phiY in zip(phisX, phisY):
        if X.int_rvec:
            phisX = vocab_indexer.uncast_residual_integer(phisX)
            phisY = vocab_indexer.uncast_residual_integer(phisY)
        u = (phiX * phiY).sum(axis=1)
        flags = np.logical_or(flagsX.T[0], flagsY.T[0])
        u[flags] = 1
        scores = selectivity(u, alpha, thresh, out=u)
        scores_list.append(scores)
    return scores_list


@profile
def sep_build_matches(X, Y, X_idx, Y_idx, score_list):
    raise NotImplementedError('sep version not finished')
    # Build feature matches
    X_fxs = ut.take(X.fxs_list, X_idx)
    Y_fxs = ut.take(Y.fxs_list, Y_idx)

    X_maws = ut.take(X.maws_list, X_idx)
    Y_maws = ut.take(Y.maws_list, Y_idx)

    # Spread word score according to contriubtion (maw) weight
    unflat_weight = [maws1[:, None].dot(maws2[:, None].T).ravel()
                     for maws1, maws2 in zip(X_maws, Y_maws)]
    flat_weight = np.array(ut.flatten(unflat_weight), dtype=np.float32)
    fs = np.array(ut.flatten(score_list), dtype=np.float32)
    np.multiply(fs, flat_weight, out=fs)

    # itertools.product seems fastest for small arrays
    unflat_fm = (ut.product(fxs1, fxs2)
                 for fxs1, fxs2 in zip(X_fxs, Y_fxs))

    fm = np.array(ut.flatten(unflat_fm), dtype=np.int32)
    isvalid = np.greater(fs, 0)
    fm = fm.compress(isvalid, axis=0)
    fs = fs.compress(isvalid, axis=0)
    return fm, fs


def check_can_match(qaid, hit_daids, qreq_):
    can_match_samename = qreq_.qparams.can_match_samename
    can_match_sameimg = qreq_.qparams.can_match_sameimg
    can_match_self = False
    valid_flags = np.ones(len(hit_daids), dtype=np.bool)
    # Check that the two annots meet the conditions
    if not can_match_self:
        valid_flags[hit_daids == qaid] = False
    if not can_match_samename:
        qnid = qreq_.get_qreq_annot_nids([qaid])[0]
        hit_dnids = qreq_.get_qreq_annot_nids(hit_daids)
        valid_flags[hit_dnids == qnid] = False
    if not can_match_sameimg:
        qgid = qreq_.get_qreq_annot_gids([qaid])[0]
        hit_dgids = qreq_.get_qreq_annot_gids(hit_daids)
        valid_flags[hit_dgids == qgid] = False
    return valid_flags


@profile
def gamma(wx_list, phisX_list, flagsX_list, wx_to_idf, alpha, thresh):
    r"""
    Computes gamma (self consistency criterion)
    It is a scalar which ensures K(X, X) = 1

    Returns:
        float: sccw self-consistency-criterion weight

    Math:
        gamma(X) = (sum_{c in C} w_c M(X_c, X_c))^{-.5}

        >>> from ibeis.algo.smk.smk_pipeline import *  # NOQA
        >>> ibs, smk, qreq_= testdata_smk()
        >>> X = qreq_.qinva.grouped_annots[0]
        >>> wx_to_idf = qreq_.wx_to_idf
        >>> print('X.gamma = %r' % (gamma(X),))
    """
    if isinstance(phisX_list, np.ndarray):
        # Agg speedup
        phisX = phisX_list
        u_list = (phisX ** 2).sum(axis=1)
        u_list[flagsX_list.T[0]] = 1
        scores = selectivity(u_list, alpha, thresh, out=u_list)
    else:
        #u_list = [phisX.dot(phisX.T) for phisX in phisX_list]
        scores = np.array([
            selective_match_score(phisX, phisX, flagsX, flagsX, alpha,
                                   thresh).sum()
            for phisX, flagsX in zip(phisX_list, flagsX_list)
        ])
    idf_list = np.array(ut.take(wx_to_idf, wx_list))
    scores *= idf_list
    score = scores.sum()
    sccw = np.reciprocal(np.sqrt(score))
    return sccw


@profile
def selective_match_score(phisX, phisY, flagsX, flagsY, alpha, thresh):
    """
    computes the score of each feature match
    """
    u = phisX.dot(phisY.T)
    # Give error flags full scores. These are typically distinctive and
    # important cases without enough info to get residual data.
    flags = np.logical_or(flagsX[:, None], flagsY)
    u[flags] = 1
    score = selectivity(u, alpha, thresh, out=u)
    return score


#def match_score_sep(X, Y, c):
#    """ matching score between separated residual vectors

#    flagsX = np.array([1, 0, 0], dtype=np.bool)
#    flagsY = np.array([1, 1, 0], dtype=np.bool)
#    phisX = vt.tests.dummy.testdata_dummy_sift(3, asint=False)
#    phisY = vt.tests.dummy.testdata_dummy_sift(3, asint=False)
#    """
#    phisX, flagsX = X.phis_flags(c)
#    phisY, flagsY = Y.phis_flags(c)
#    # Give error flags full scores. These are typically distinctive and
#    # important cases without enough info to get residual data.
#    u = phisX.dot(phisY.T)
#    flags = np.logical_or(flagsX[:, None], flagsY)
#    u[flags] = 1
#    return u


@profile
def selectivity(u, alpha=3.0, thresh=-1, out=None):
    r"""
    Rescales and thresholds scores. This is sigma from the SMK paper

    Notes:
        # Exact definition from paper
        sigma_alpha(u) = bincase{
            sign(u) * (u**alpha) if u > thresh,
            0 otherwise,
        }

    CommandLine:
        python -m plottool plot_func --show --range=-1,1 --setup="import ibeis" \
                --func ibeis.algo.smk.smk_pipeline.selectivity \
                "lambda u: sign(u) * abs(u)**3.0 * greater_equal(u, 0)" \
    """
    score = u
    flags = np.less(score, thresh)
    isign = np.sign(score)
    score = np.abs(score, out=out)
    score = np.power(score, alpha, out=out)
    score = np.multiply(isign, score, out=out)
    score[flags] = 0
    #
    #score = np.sign(u) * np.power(np.abs(u), alpha)
    #score *= flags
    return score


def testdata_smk(*args, **kwargs):
    """
    >>> from ibeis.algo.smk.smk_pipeline import *  # NOQA
    >>> kwargs = {}
    """
    import ibeis
    import sklearn
    import sklearn.cross_validation
    # import sklearn.model_selection
    ibs, aid_list = ibeis.testdata_aids(defaultdb='PZ_MTEST')
    nid_list = np.array(ibs.annots(aid_list).nids)
    rng = ut.ensure_rng(0)
    xvalkw = dict(n_folds=4, shuffle=False, random_state=rng)

    skf = sklearn.cross_validation.StratifiedKFold(nid_list, **xvalkw)
    train_idx, test_idx = six.next(iter(skf))
    daids = ut.take(aid_list, train_idx)
    qaids = ut.take(aid_list, test_idx)

    config = {
        'num_words': 1000,
    }
    config.update(**kwargs)
    qreq_ = SMKRequest(ibs, qaids, daids, config)
    smk = qreq_.smk
    #qreq_ = ibs.new_query_request(qaids, daids, cfgdict={'pipeline_root': 'smk', 'proot': 'smk'})
    #qreq_ = ibs.new_query_request(qaids, daids, cfgdict={})
    return ibs, smk, qreq_


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/ibeis/ibeis/algo/smk
        python ~/code/ibeis/ibeis/algo/smk/smk_pipeline.py
        python ~/code/ibeis/ibeis/algo/smk/smk_pipeline.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
