# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import dtool
import six
import utool as ut
import numpy as np
from ibeis.algo.smk import match_chips5 as mc5
from ibeis.algo.smk import vocab_indexer
from ibeis import core_annots
from ibeis.algo import Config as old_config
(print, rrr, profile) = ut.inject2(__name__)


class SMKConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('smk_alpha', 3.0),
        ut.ParamInfo('smk_thresh', 0.25),
        ut.ParamInfo('agg', True),
    ]


class MatchHeuristicsConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('can_match_self', False),
        ut.ParamInfo('can_match_samename', True),
        ut.ParamInfo('can_match_sameimg', False),
    ]


class SMKRequestConfig(dtool.Config):
    """ Figure out how to do this """
    _param_info_list = [
        ut.ParamInfo('proot', 'smk')
    ]
    _sub_config_list = [
        core_annots.ChipConfig,
        core_annots.FeatConfig,
        old_config.SpatialVerifyConfig,
        vocab_indexer.VocabConfig,
        vocab_indexer.InvertedIndexConfig,
        MatchHeuristicsConfig,
        SMKConfig,
    ]


@ut.reloadable_class
class SMKRequest(mc5.EstimatorRequest):
    """
    qreq_-like object. Trying to work on becoming more scikit-ish

    CommandLine:
        python -m ibeis.algo.smk.smk_pipeline SMKRequest
        python -m ibeis.algo.smk.smk_pipeline SMKRequest --show

        python -m ibeis draw_rank_cdf --db PZ_MTEST --show -p :proot=smk,num_words=64000 -a default
        python -m ibeis draw_rank_cdf --db PZ_MTEST --show -p :proot=smk,num_words=64000 default:proot=vsmany -a default
        python -m ibeis draw_rank_cdf --db PZ_MTEST --show -p :proot=smk,num_words=64000,nAssign=[2,4] default:proot=vsmany -a default
        python -m ibeis draw_rank_cdf --db PZ_MTEST --show -p :proot=smk,num_words=64000,nAssign=[2,4] default:proot=vsmany -a default:qmingt=2
        python -m ibeis draw_rank_cdf --db PZ_MTEST --show -p :proot=smk,num_words=64000,nAssign=[1,2,4] default:proot=vsmany -a default:qmingt=2
        python -m ibeis draw_rank_cdf --db PZ_MTEST --show -p :proot=smk,num_words=64000 -a default --pcfginfo

        python -m ibeis draw_rank_cdf --db PZ_MTEST --show -p :proot=smk,num_words=[32000,64000,8000,4000],nAssign=[1,2,4] default:proot=vsmany -a default:qmingt=2
        python -m ibeis draw_rank_cdf --db PZ_MTEST --show -p :proot=smk,num_words=[64000,32000],nAssign=[1,2,4] default:proot=vsmany -a default:qmingt=2
        python -m ibeis draw_rank_cdf --db PZ_MTEST --show -p :proot=smk,num_words=[64000,1000,400,50],nAssign=[1] default:proot=vsmany -a default:qmingt=2

        python -m ibeis draw_rank_cdf --db PZ_MTEST --show \
            -p :proot=smk,num_words=[64000,32000,8000,4000],nAssign=[1,2,4],sv_on=[True,False] \
                default:proot=vsmany,sv_on=[True,False] \
            -a default:qmingt=2

        python -m ibeis draw_rank_cdf --db PZ_MTEST --show \
            -p :proot=smk,num_words=[64000,8000,4000],nAssign=[1,2,4],sv_on=[True,False] \
                default:proot=vsmany,sv_on=[True,False] \
            -a default:qmingt=2

    Example:
        >>> from ibeis.algo.smk.smk_pipeline import *  # NOQA
        >>> import ibeis
        >>> ibs, aid_list = ibeis.testdata_aids(defaultdb='PZ_MTEST')
        >>> qaids = aid_list[0:2]
        >>> daids = aid_list[:]
        >>> config = {'nAssign': 2, 'num_words': 64000, 'sv_on': False}
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
        ibs = qreq_.ibs
        config = qreq_.config

        vocab = ibs.depc.get('vocab', [qreq_.daids], 'words',
                             config=qreq_.config)[0]
        # Hack in config to vocab class
        # (maybe this becomes part of the depcache)
        vocab.config = ibs.depc.configclass_dict['vocab'](**qreq_.config)

        qfstack = vocab_indexer.ForwardIndex(ibs, qreq_.qaids, config, name='query')
        dfstack = vocab_indexer.ForwardIndex(ibs, qreq_.daids, config, name='data')
        qinva = vocab_indexer.InvertedIndex(qfstack, vocab, config=config)
        dinva = vocab_indexer.InvertedIndex(dfstack, vocab, config=config)

        #qreq_.cachedir = ut.ensuredir((ibs.cachedir, 'smk'))
        qreq_.ensure_nids()

        qfstack.ensure(qreq_.cachedir)
        dfstack.ensure(qreq_.cachedir)
        #qinva.build(groups=True)
        #dinva.build(groups=True, idf=True)

        qinva.ensure(qreq_.cachedir)
        dinva.ensure(qreq_.cachedir)

        _cacher = ut.cached_func('idf_' +  dinva.get_hashid(), cache_dir=qreq_.cachedir)
        _ensureidf = _cacher(dinva.compute_idf)
        dinva.wx_to_idf = _ensureidf()

        thresh = qreq_.qparams['smk_thresh']
        alpha = qreq_.qparams['smk_alpha']
        agg = qreq_.qparams['agg']

        smk = qreq_.smk
        wx_to_idf = dinva.wx_to_idf
        for inva in [qinva, dinva]:
            _cacher = ut.cached_func('gamma_sccw_' +  inva.get_hashid(), cache_dir=qreq_.cachedir,
                                     key_argx=[1, 2, 3, 4])
            X_list = inva.grouped_annots
            _ensuregamma = _cacher(smk.precompute_gammas)
            gamma_list = _ensuregamma(X_list, wx_to_idf, agg, alpha, thresh)
            for X, gamma in zip(X_list, gamma_list):
                X.gamma = gamma

        qreq_.qinva = qinva
        qreq_.dinva = dinva

    def execute_pipeline(qreq_):
        """
        >>> from ibeis.algo.smk.smk_pipeline import *  # NOQA
        >>> ibs, smk, qreq_ = testdata_smk()
        >>> cm_list = qreq_.execute()
        """
        smk = qreq_.smk
        cm_list = smk.predict_matches(qreq_)
        return cm_list


@ut.reloadable_class
class SMK(ut.NiceRepr):
    """
    Harness class that controls the execution of the SMK algorithm

    K(X, Y) = gamma(X) * gamma(Y) * sum([Mc(Xc, Yc) for c in words])
    """
    def __init__(smk):
        pass

    #def predict(smk, qreq_):
    #    # TODO
    #    pass
    #def fit(smk, dinva):
    #    qreq_.dinva = dinva

    def predict_matches(smk, qreq_, verbose=True):
        """
        >>> from ibeis.algo.smk.smk_pipeline import *  # NOQA
        >>> ibs, smk, qreq_ = testdata_smk()
        >>> verbose = True
        """
        print('Predicting matches')
        assert qreq_.qinva.vocab is qreq_.dinva.vocab
        X_list = qreq_.qinva.inverted_annots(qreq_.qaids)
        Y_list = qreq_.dinva.inverted_annots(qreq_.daids)
        _prog = ut.ProgPartial(lbl='smk query', bs=verbose <= 1, enabled=verbose)
        cm_list = [smk.match_single(X, Y_list, qreq_, verbose=verbose > 1)
                   for X in _prog(X_list)]
        return cm_list

    def check_can_match(smk, X, Y, qreq_):
        can_match_samename = qreq_.qparams.can_match_samename
        can_match_sameimg = qreq_.qparams.can_match_sameimg
        can_match_self = False
        flag = True
        # Check that the two annots meet the conditions
        if not can_match_self:
            aid1, aid2 = X.aid, Y.aid
            if aid1 == aid2:
                flag = False
        if not can_match_samename:
            nid1, nid2 = qreq_.get_qreq_annot_nids([X.aid, Y.aid])
            if nid1 == nid2:
                flag = False
        if not can_match_sameimg:
            gid1, gid2 = qreq_.get_qreq_annot_gids([X.aid, Y.aid])
            if gid1 == gid2:
                flag = False
        return flag

    @profile
    def match_single(smk, X, Y_list, qreq_, verbose=True):
        """
        CommandLine:
            python -m ibeis.algo.smk.smk_pipeline SMK.match_single --profile
            python -m ibeis.algo.smk.smk_pipeline SMK.match_single --show

        Example:
            >>> # FUTURE_ENABLE
            >>> from ibeis.algo.smk.smk_pipeline import *  # NOQA
            >>> import ibeis
            >>> ibs, daids = ibeis.testdata_aids(defaultdb='PZ_MTEST')
            >>> qreq_ = SMKRequest(ibs, daids[0:1], daids, {'agg': True, 'num_words': 64000, 'sv_on': True})
            >>> qreq_.ensure_data()
            >>> #ibs, smk, qreq_ = testdata_smk()
            >>> X = qreq_.qinva.grouped_annots[0]
            >>> Y_list = qreq_.dinva.grouped_annots
            >>> Y = Y_list[1]
            >>> c = ut.isect(X.words, Y.words)[0]
            >>> cm = qreq_.smk.match_single(X, Y_list, qreq_)
            >>> ut.quit_if_noshow()
            >>> ut.qt4ensure()
            >>> cm.ishow_analysis(qreq_)
            >>> ut.show_if_requested()
        """
        from ibeis.algo.hots import chip_match
        from ibeis.algo.hots import pipeline

        qaid = X.aid
        daid_list = []
        fm_list = []
        fsv_list = []
        fsv_col_lbls = ['smk']

        wx_to_idf = qreq_.dinva.wx_to_idf

        alpha  = qreq_.qparams['smk_alpha']
        thresh = qreq_.qparams['smk_thresh']
        agg    = qreq_.qparams['agg']
        nNameShortList  = qreq_.qparams.nNameShortlistSVER
        nAnnotPerName   = qreq_.qparams.nAnnotPerNameSVER
        sv_on   = qreq_.qparams.sv_on

        #gammaX = smk.gamma(X, wx_to_idf, agg, alpha, thresh)
        gammaX = X.gamma

        for Y in ut.ProgIter(Y_list, lbl='smk match qaid=%r' % (qaid,), enabled=verbose):
            if smk.check_can_match(X, Y, qreq_):
                #gammaY = smk.gamma(Y, wx_to_idf, agg, alpha, thresh)
                gammaY = Y.gamma
                gammaXY = gammaX * gammaY

                fm = []
                fs = []
                # Words in common define matches
                common_words = ut.isect(X.words, Y.words)
                for c in common_words:
                    #Explicitly computes the feature matches that will be scored
                    score = smk.selective_match_score(X, Y, c, agg, alpha, thresh)
                    score *= wx_to_idf[c]
                    word_fm, word_fs = smk.build_matches(X, Y, c, score)
                    assert not np.any(np.isnan(word_fs))
                    word_fs *= gammaXY
                    fm.extend(word_fm)
                    fs.extend(word_fs)

                #if len(fm) > 0:
                daid = Y.aid
                fsv = np.array(fs)[:, None]
                daid_list.append(daid)
                fm = np.array(fm)
                fm_list.append(fm)
                fsv_list.append(fsv)

        # Build initial matches
        cm = chip_match.ChipMatch(qaid=qaid, daid_list=daid_list,
                                  fm_list=fm_list, fsv_list=fsv_list,
                                  fsv_col_lbls=fsv_col_lbls)
        cm.arraycast_self()
        # Score matches and take a shortlist
        cm.score_maxcsum(qreq_)

        if sv_on:
            top_aids = cm.get_name_shortlist_aids(nNameShortList, nAnnotPerName)
            cm = cm.shortlist_subset(top_aids)
            # Spatially verify chip matches
            cm = pipeline.sver_single_chipmatch(qreq_, cm, verbose=verbose)
            # Rescore
            cm.score_maxcsum(qreq_)

        return cm

    def build_matches(smk, X, Y, c, score):
        # Build matching index list as well
        word_fm = list(ut.product(X.fxs(c), Y.fxs(c)))
        if score.size != len(word_fm):
            # Spread word score according to contriubtion (maw) weight
            contribution = X.maws(c)[:, None].dot(Y.maws(c)[:, None].T)
            contrib_weight = (contribution / contribution.sum())
            word_fs = (contrib_weight * score).ravel()
        else:
            # Scores were computed separately, so dont spread
            word_fs = score.ravel()
        isvalid = word_fs > 0
        word_fs = word_fs.compress(isvalid)
        word_fm = ut.compress(word_fm, isvalid)
        return word_fm, word_fs

    def precompute_gammas(smk, X_list, wx_to_idf, agg, alpha, thresh):
        gamma_list = []
        for X in ut.ProgIter(X_list, lbl='precompute gamma'):
            gamma = smk.gamma(X, wx_to_idf, agg, alpha, thresh)
            gamma_list.append(gamma)
        return gamma_list

    @profile
    def gamma(smk, X, wx_to_idf, agg, alpha, thresh):
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
            >>> print('X.gamma = %r' % (smk.gamma(X),))
        """
        scores = np.array([
            smk.selective_match_score(X, X, c, agg, alpha, thresh).sum()
            for c in X.words
        ])
        idf_list = np.array(ut.take(wx_to_idf, X.words))
        scores *= idf_list
        score = scores.sum()
        sccw = np.reciprocal(np.sqrt(score))
        return sccw

    @profile
    def selective_match_score(smk, X, Y, c, agg, alpha, thresh):
        """
        Just computes the total score of all feature matches
        """
        if agg:
            u = smk.match_score_agg(X, Y, c)
        else:
            u = smk.match_score_sep(X, Y, c)
        score = smk.selectivity(u, alpha, thresh, out=u)
        return score

    @profile
    def match_score_agg(smk, X, Y, c):
        """ matching score between aggregated residual vectors
        """
        PhiX, flagsX = X.Phi_flags(c)
        PhiY, flagsY = Y.Phi_flags(c)
        # Give error flags full scores. These are typically distinctive and
        # important cases without enough info to get residual data.
        u = PhiX.dot(PhiY.T)
        flags = np.logical_or(flagsX[:, None], flagsY)
        u[flags] = 1.0
        return u

    @profile
    def match_score_sep(smk, X, Y, c):
        """ matching score between separated residual vectors

        flagsX = np.array([1, 0, 0], dtype=np.bool)
        flagsY = np.array([1, 1, 0], dtype=np.bool)
        phisX = vt.tests.dummy.testdata_dummy_sift(3, asint=False)
        phisY = vt.tests.dummy.testdata_dummy_sift(3, asint=False)
        """
        phisX, flagsX = X.phis_flags(c)
        phisY, flagsY = Y.phis_flags(c)
        # Give error flags full scores. These are typically distinctive and
        # important cases without enough info to get residual data.
        u = phisX.dot(phisY.T)
        flags = np.logical_or(flagsX[:, None], flagsY)
        u[flags] = 1
        #u = X.phis(c).dot(Y.phis(c).T)
        return u

    @profile
    def selectivity(smk, u, alpha, thresh, out=None):
        r"""
        Rescales and thresholds scores. This is sigma from the SMK paper

        In the paper they use:
            sigma_alpha(u) = bincase{
                sign(u) * (u**alpha) if u > thresh,
                0 otherwise,
            }
        but this make little sense because a negative threshold of -1
        will let -1 descriptors (completely oppositely aligned) have
        the same score as perfectly aligned descriptors. Instead I suggest
        (to achieve the same effect), a thresh = .5 and  a preprocessing step of
            u'  = (u + 1) / 2
        """
        score = u
        score = np.add(score, 1.0, out=out)
        score = np.divide(score, 2.0, out=out)
        flags = np.less_equal(score, thresh)
        score = np.power(score, alpha, out=out)
        score[flags] = 0
        #u = (u + 1.0) / 2.0
        #flags = np.greater(u, thresh)
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
    qreq_.ensure_data()
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
