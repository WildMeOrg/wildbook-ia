# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool as vt
from os.path import join
from operator import xor
from vtool import matching
import six
from ibeis.model.hots import hstypes
from ibeis.model.hots import old_chip_match
from ibeis.model.hots import scoring
from ibeis.model.hots import name_scoring
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
print, rrr, profile = ut.inject2(__name__, '[chip_match]', DEBUG=False)


DEBUG_CHIPMATCH = False

#import six

MAX_FNAME_LEN = 64 if ut.WIN32 else 200
TRUNCATE_UUIDS = ut.get_argflag(('--truncate-uuids', '--trunc-uuids'))
#or ( ut.is_developer() and not ut.get_argflag(('--notrunc-uuids',)))


@profile
def get_chipmatch_fname(qaid, qreq_, qauuid=None, cfgstr=None, TRUNCATE_UUIDS=TRUNCATE_UUIDS, MAX_FNAME_LEN=MAX_FNAME_LEN):
    """
    CommandLine:
        python -m ibeis.model.hots.chip_match --test-get_chipmatch_fname

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.chip_match import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list=[18])
        >>> cm = cm_list[0]
        >>> fname = get_chipmatch_fname(cm.qaid, qreq_, qauuid=None, TRUNCATE_UUIDS=False, MAX_FNAME_LEN=200)
        >>> result = ('fname = %s' % (ut.repr2(fname),))
        >>> print(result)
        fname = 'qaid=18_cm_qjjzmjiwwwdhyzrw_quuid=a126d459-b730-573e-7a21-92894b016565.cPkl'

        fname = 'qaid=18_cm_mnzkiegiilcsbwxy_quuid=a126d459-b730-573e-7a21-92894b016565.cPkl'
    """
    if qauuid is None:
        print('[chipmatch] Warning qasuuid should be passed into get_chipmatch_fname')
        qauuid = qreq_.ibs.get_annot_semantic_uuids(qaid)
    if cfgstr is None:
        print('[chipmatch] Warning cfgstr should be passed into get_chipmatch_fname')
        cfgstr = qreq_.get_cfgstr(with_query=False, with_data=True, with_pipe=True)
    #print('cfgstr = %r' % (cfgstr,))
    fname_fmt = 'qaid={qaid}_cm_{cfgstr}_quuid={qauuid}{ext}'
    text_type = six.text_type
    #text_type = str
    qauuid_str = text_type(qauuid)[0:8] if TRUNCATE_UUIDS else text_type(qauuid)
    fmt_dict = dict(cfgstr=cfgstr, qaid=qaid, qauuid=qauuid_str, ext='.cPkl')
    fname = ut.long_fname_format(fname_fmt, fmt_dict, ['cfgstr'],
                                 max_len=MAX_FNAME_LEN, hack27=True)
    return fname


@six.add_metaclass(ut.ReloadingMetaclass)
class ChipMatch2(old_chip_match._OldStyleChipMatchSimulator):
    """
    behaves as as the ChipMatchOldTup named tuple until we
    completely replace the old structure
    """

    # Standard Contstructor

    def __init__(cm, *args, **kwargs):
        """
        qaid and daid_list are not optional. fm_list and fsv_list are strongly
        encouraged and will probalby break things if they are not there.
        """
        cm.qaid         = None
        cm.daid_list    = None
        cm.fm_list      = None
        cm.fsv_list     = None
        cm.fk_list      = None
        cm.score_list   = None
        cm.H_list       = None
        cm.fsv_col_lbls = None
        cm.fs_list = None
        # This is aligned with daid list, need to avoid confusion with
        # unique_nids
        cm.dnid_list = None
        # standard groupings
        # TODO: rename unique_nids to indicate it is aligned with name_groupxs
        cm.unique_nids = None  # belongs to name_groupxs
        cm.qnid = None
        # Name probabilities
        cm.prob_list = None
        # Annot scores
        cm.annot_score_list = None
        # Name scores
        cm.name_score_list = None
        # TODO: have subclass or dict for special scores
        # Special annot scores
        cm.special_annot_scores = [
            'csum',
            'acov',
        ]
        for score_method in cm.special_annot_scores:
            setattr(cm, score_method + '_score_list', None)
        #cm.csum_score_list = None
        #cm.acov_score_list = None
        # Special name scores
        cm.special_name_scores = [
            'nsum',
            'maxcsum',
            'ncov',
        ]
        for score_method in cm.special_name_scores:
            setattr(cm, score_method + '_score_list', None)
        #cm.nsum_score_list = None
        #cm.maxcsum_score_list = None
        #cm.ncov_score_list = None
        # Re-evaluatables (for convinience only)
        cm.daid2_idx = None  # maps onto cm.daid_list
        cm.nid2_nidx = None  # maps onto cm.unique_nids
        cm.name_groupxs = None
        if len(args) + len(kwargs) > 0:
            cm.initialize(*args, **kwargs)

    def initialize(cm, qaid=None, daid_list=None, fm_list=None, fsv_list=None,
                   fk_list=None, score_list=None, H_list=None,
                   fsv_col_lbls=None, dnid_list=None, qnid=None,
                   unique_nids=None, name_score_list=None,
                   annot_score_list=None, autoinit=True):
        """
        qaid and daid_list are not optional. fm_list and fsv_list are strongly
        encouraged and will probalby break things if they are not there.
        """
        if DEBUG_CHIPMATCH:
            assert daid_list is not None, 'must give daids'
            assert fm_list is None or len(fm_list) == len(daid_list), 'incompatable data'
            assert fsv_list is None or len(fsv_list) == len(daid_list), 'incompatable data'
            assert fk_list is None or len(fk_list) == len(daid_list), 'incompatable data'
            assert H_list is None or len(H_list) == len(daid_list), 'incompatable data'
            assert score_list is None or len(score_list) == len(daid_list), 'incompatable data'
            assert dnid_list is None or len(dnid_list) == len(daid_list), 'incompatable data'
        cm.qaid         = qaid
        cm.daid_list    = np.array(daid_list, dtype=hstypes.INDEX_TYPE)
        cm.fm_list      = fm_list
        cm.fsv_list     = fsv_list
        cm.fk_list      = (fk_list if fk_list is not None else
                           [np.zeros(fm.shape[0]) for fm in cm.fm_list]
                           if cm.fm_list is not None else None)
        cm.score_list   = score_list
        cm.H_list       = H_list
        cm.fsv_col_lbls = fsv_col_lbls
        #cm.daid2_idx    = None
        #cm.fs_list = None
        # TODO
        cm.dnid_list = None if dnid_list is None else np.array(dnid_list, dtype=hstypes.INDEX_TYPE)
        cm.qnid = qnid
        cm.unique_nids = None if unique_nids is None else np.array(unique_nids, dtype=hstypes.INDEX_TYPE)
        cm.name_score_list = name_score_list
        cm.annot_score_list = annot_score_list
        # standard groupings
        #cm.unique_nids = None  # belongs to name_groupxs
        #cm.nid2_nidx = None
        #cm.name_groupxs = None
        ## Name probabilities
        #cm.prob_list = None
        ## Annot scores
        #cm.annot_score_list = None
        ## (Aggregated) Name scores
        #cm.name_score_list = None
        #cm.maxcsum_score_list = None
        ## TODO: have subclass or dict for special scores
        if autoinit:
            cm._update_daid_index()
            if cm.dnid_list is not None:
                cm._update_unique_nid_index()

    def _empty_hack(cm):
        if cm.daid_list is None:
            cm.daid_list = np.empty(0, dtype=np.int)
        assert len(cm.daid_list) == 0
        cm.fsv_col_lbls = []
        cm.fm_list = []
        cm.fsv_list = []
        cm.fk_list = []
        cm.H_list = []
        cm.daid2_idx = {}
        cm.fs_list = []
        cm.dnid_list = np.empty(0, dtype=hstypes.INDEX_TYPE)
        cm.unique_nids = np.empty(0, dtype=hstypes.INDEX_TYPE)
        cm.score_list = np.empty(0)
        cm.name_score_list = np.empty(0)
        cm.annot_score_list = np.empty(0)

    #------------------
    # Modification / Evaluation Functions
    #------------------

    def _cast_scores(cm, dtype=np.float):
        cm.fsv_list = [fsv.astype(dtype) for fsv in cm.fsv_list]

    def extend_results(cm, qreq_, other_aids=None):
        """
        returns a new cmtup_old that contains empty data for an extended set of
        aids
        """
        if other_aids is None:
            other_aids = qreq_.get_external_daids()
        ibs = qreq_.ibs
        other_aids_ = other_aids
        other_aids_ = np.setdiff1d(other_aids_, cm.daid_list)
        other_aids_ = np.setdiff1d(other_aids_, [cm.qaid])
        other_nids_ = ibs.get_annot_nids(other_aids_)
        other_unique_nids = np.setdiff1d(np.unique(other_nids_),
                                         cm.unique_nids)
        num = len(other_aids_)
        num2 = len(other_unique_nids)
        #print('PRINT EXTENDING BY num = %r' % (num,))
        #print('num2 = %r' % (num2,))

        def extend_scores(num, vals):
            if vals is None:
                return None
            return np.append(vals, np.full(num, -np.inf))

        def extend_nplists(num, x_list, shape, dtype):
            if x_list is None:
                return None
            return (x_list + [np.empty(shape, dtype=dtype)] * num)

        def extend_pylist(num, x_list, val):
            if x_list is None:
                return None
            return (x_list + [None] * num)

        daid_list = np.append(cm.daid_list, other_aids_)
        dnid_list = np.append(cm.dnid_list, other_nids_)

        qaid         = cm.qaid
        qnid         = cm.qnid
        fsv_col_lbls = cm.fsv_col_lbls

        num_vs = 0 if fsv_col_lbls is None else len(fsv_col_lbls)

        fm_list  = extend_nplists(num, cm.fm_list, (0, 2), hstypes.FM_DTYPE)
        fk_list  = extend_nplists(num, cm.fk_list, (0), hstypes.FK_DTYPE)
        fs_list  = extend_nplists(num, cm.fs_list, (0), hstypes.FS_DTYPE)
        fsv_list = extend_nplists(num, cm.fsv_list, (0, num_vs), hstypes.FS_DTYPE)
        H_list   = extend_pylist(num, cm.H_list, None)

        score_list = extend_scores(num, cm.score_list)
        annot_score_list = extend_scores(num, cm.annot_score_list)

        unique_nids  = np.append(cm.unique_nids, other_unique_nids)
        name_score_list = extend_scores(num2, cm.name_score_list)

        cm2 = ChipMatch2(qaid, daid_list, fm_list, fsv_list, fk_list,
                         score_list, H_list, fsv_col_lbls, dnid_list,
                         qnid, unique_nids, name_score_list,
                         annot_score_list, autoinit=False)
        cm2.fs_list = fs_list
        for score_method in cm2.special_annot_scores:
            attr = score_method + '_score_list'
            setattr(cm2, attr, extend_scores(num, getattr(cm, attr, None)))
        for score_method in cm2.special_name_scores:
            attr = score_method + '_score_list'
            setattr(cm2, attr, extend_scores(num2, getattr(cm, attr, None)))
        cm2._update_daid_index()
        cm2._update_unique_nid_index()
        return cm2

    def _update_daid_index(cm):
        cm.daid2_idx = (None if cm.daid_list is None else
                        ut.make_index_lookup(cm.daid_list))
        #{daid: idx for idx, daid in enumerate(cm.daid_list)})

    def _update_unique_nid_index(cm):
        #assert cm.unique_nids is not None
        unique_nids_, name_groupxs_ = vt.group_indices(cm.dnid_list)
        #assert unique_nids_.dtype == hstypes.INTEGER_TYPE
        if cm.unique_nids is None:
            assert cm.name_score_list is None
            cm.unique_nids = unique_nids_
        cm.nid2_nidx = ut.make_index_lookup(cm.unique_nids)
        nidx_list = np.array(ut.dict_take(cm.nid2_nidx, unique_nids_))
        inverse_idx_list = nidx_list.argsort()
        cm.name_groupxs = ut.list_take(name_groupxs_, inverse_idx_list)
        #cm.unique_nids  = unique_nids
        #cm.name_groupxs = name_groupxs
        #cm.nid2_nidx    = ut.make_index_lookup(cm.unique_nids)

    def evaluate_dnids(cm, ibs):
        cm.qnid = ibs.get_annot_name_rowids(cm.qaid)
        dnid_list = ibs.get_annot_name_rowids(cm.daid_list)
        cm.dnid_list = np.array(dnid_list, dtype=hstypes.INDEX_TYPE)
        cm._update_unique_nid_index()
        # evaluate name groupings as well
        #unique_nids, name_groupxs = vt.group_indices(cm.dnid_list)
        #cm.unique_nids  = unique_nids
        #cm.name_groupxs = name_groupxs
        #cm.nid2_nidx    = ut.make_index_lookup(cm.unique_nids)

    def sortself(cm):
        """ reorders the internal data using cm.score_list """
        sortx = cm.argsort()
        cm.daid_list = vt.trytake(cm.daid_list, sortx)
        cm.dnid_list = vt.trytake(cm.dnid_list, sortx)
        cm.fm_list = vt.trytake(cm.fm_list, sortx)
        cm.fsv_list = vt.trytake(cm.fsv_list, sortx)
        cm.fs_list = vt.trytake(cm.fs_list, sortx)
        cm.fk_list = vt.trytake(cm.fk_list, sortx)
        cm.score_list = vt.trytake(cm.score_list, sortx)
        cm.csum_score_list = vt.trytake(cm.csum_score_list, sortx)
        cm.H_list = vt.trytake(cm.H_list, sortx)
        cm._update_daid_index()

    def shortlist_subset(cm, top_aids):
        """ returns a new cmtup_old with only the requested daids """
        qaid = cm.qaid
        qnid = cm.qnid
        idx_list = ut.dict_take(cm.daid2_idx, top_aids)
        daid_list = vt.list_take_(cm.daid_list, idx_list)
        fm_list = vt.list_take_(cm.fm_list, idx_list)
        fsv_list = vt.list_take_(cm.fsv_list, idx_list)
        fk_list = vt.trytake(cm.fk_list, idx_list)
        #score_list   = vt.trytake(cm.score_list, idx_list)
        score_list = None  # don't transfer scores
        H_list = vt.trytake(cm.H_list, idx_list)
        dnid_list = vt.trytake(cm.dnid_list, idx_list)
        fsv_col_lbls = cm.fsv_col_lbls
        cm_subset = ChipMatch2(qaid, daid_list, fm_list, fsv_list, fk_list,
                               score_list, H_list, fsv_col_lbls, dnid_list,
                               qnid)
        return cm_subset

    # Alternative Cosntructors / Convertors

    @classmethod
    @profile
    def from_qres(cls, qres):
        r"""
        """
        aid2_fm_    = qres.aid2_fm
        aid2_fsv_   = qres.aid2_fsv
        aid2_fk_    = qres.aid2_fk
        aid2_score_ = qres.aid2_score
        aid2_H_     = qres.aid2_H
        qaid        = qres.qaid
        cmtup_old = (aid2_fm_, aid2_fsv_, aid2_fk_, aid2_score_, aid2_H_)
        fsv_col_lbls = qres.filtkey_list
        cm = cls.from_cmtup_old(cmtup_old, qaid, fsv_col_lbls, daid_list=qres.daids)
        #with ut.embed_on_exception_context:
        #if 'lnbnn' in fsv_col_lbls:
        #    assert 'lnbnn' in fsv_col_lbls, 'cm.fsv_col_lbls=%r' % (cm.fsv_col_lbls,)
        #    fs_list = [fsv.T[cm.fsv_col_lbls.index('lnbnn')] for fsv in cm.fsv_list]
        #else:
        if True:
            fs_list = ut.dict_take(qres.aid2_fs, cm.daid_list,
                                   np.empty((0,), dtype=hstypes.FS_DTYPE))
        cm.fs_list = fs_list
        return cm

    @classmethod
    @profile
    def from_unscored(cls, prior_cm, fm_list, fs_list, H_list=None, fsv_col_lbls=None):
        qaid = prior_cm.qaid
        daid_list = prior_cm.daid_list
        fsv_list = matching.ensure_fsv_list(fs_list)
        if fsv_col_lbls is None:
            fsv_col_lbls = ['unknown']
            #fsv_col_lbls = [str(count) for count in range(num_cols)]
            #fsv_col_lbls
        #score_list = [fsv.prod(axis=1).sum() for fsv in fsv_list]
        score_list = [-1 for fsv in fsv_list]
        #fsv.prod(axis=1).sum() for fsv in fsv_list]
        cm = cls(qaid, daid_list, fm_list, fsv_list, None, score_list, H_list, fsv_col_lbls)
        cm.fs_list = fs_list
        return cm

    @classmethod
    @profile
    def from_vsmany_match_tup(cls, valid_match_tup, qaid=None, fsv_col_lbls=None):
        r"""
        Args:
            valid_match_tup (tuple):
            qaid (int):  query annotation id
            fsv_col_lbls (None):

        Returns:
            ChipMatch2: cm
        """
        # CONTIGUOUS ARRAYS MAKE A HUGE DIFFERENCE
        # Vsmany - create new cmtup_old
        (valid_daid, valid_qfx, valid_dfx, valid_scorevec, valid_rank) = valid_match_tup
        #valid_fm = np.vstack((valid_qfx, valid_dfx)).T
        valid_fm = np.ascontiguousarray(np.hstack((valid_qfx[:, None], valid_dfx[:, None])))
        daid_list, daid_groupxs = vt.group_indices(valid_daid)
        fm_list  = vt.apply_grouping(valid_fm, daid_groupxs)
        #fsv_list = vt.apply_grouping(valid_scorevec, daid_groupxs)
        fsv_list = vt.apply_grouping(np.ascontiguousarray(valid_scorevec), daid_groupxs)
        fk_list  = vt.apply_grouping(valid_rank, daid_groupxs)
        cm = cls(qaid, daid_list, fm_list, fsv_list, fk_list, fsv_col_lbls=fsv_col_lbls)
        return cm

    @classmethod
    @profile
    def from_vsone_match_tup(cls, valid_match_tup_list, daid_list=None,
                             qaid=None, fsv_col_lbls=None):
        assert all(list(map(ut.list_allsame, ut.get_list_column(valid_match_tup_list, 0)))),\
            'internal daids should not have different daids for vsone'
        qfx_list = ut.get_list_column(valid_match_tup_list, 1)
        dfx_list = ut.get_list_column(valid_match_tup_list, 2)
        fm_list  = [np.vstack(dfx_qfx).T for dfx_qfx in zip(dfx_list, qfx_list)]
        fsv_list = ut.get_list_column(valid_match_tup_list, 3)
        fk_list  = ut.get_list_column(valid_match_tup_list, 4)
        cm = cls(qaid, daid_list, fm_list, fsv_list, fk_list, fsv_col_lbls=fsv_col_lbls)
        return cm

    #def as_qres2(cm, qreq_):
    #    qres = qreq_.make_empty_query_result(cm.qaid)
    #    #ut.assert_eq(qaid, cm.qaid)
    #    qres.filtkey_list = cm.fsv_col_lbls
    #    qres.aid2_fm    = dict(zip(cm.daid_list, cm.fm_list))
    #    qres.aid2_fsv   = dict(zip(cm.daid_list, cm.fsv_list))
    #    qres.aid2_fs    = dict(zip(cm.daid_list, [fsv.prod(axis=1) for fsv in cm.fsv_list]))
    #    qres.aid2_fk    = dict(zip(cm.daid_list, cm.fk_list))
    #    qres.aid2_score = dict(zip(cm.daid_list, cm.score_list))
    #    qres.aid2_H     = None if cm.H_list is None else dict(zip(cm.daid_list, cm.H_list))
    #    qres.aid2_prob  = None if cm.prob_list is None else dict(zip(cm.daid_list, cm.prob_list))
    #    return qres

    #def as_qres(cm, qreq_):
    #    from ibeis.model.hots import scoring
    #    assert qreq_ is not None
    #    # Perform final scoring
    #    # TODO: only score if already unscored
    #    score_method = qreq_.qparams.score_method
    #    # TODO: move scoring part to pipeline
    #    scoring.score_chipmatch_list(qreq_, [cm], score_method)
    #    # Normalize scores if requested
    #    if qreq_.qparams.score_normalization:
    #        normalizer = qreq_.normalizer
    #        cm.prob_list = normalizer.normalize_score_list(cm.score_list)
    #    qres = cm.as_qres2(qreq_)
    #    return qres

    @classmethod
    def from_json(cls, json_str):
        r"""
        Convert json string back to ChipMatch object

        CommandLine:
            # FIXME; util_test is broken with classmethods
            python -m ibeis.model.hots.chip_match --test-from_json --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> import ibeis
            >>> cls = ChipMatch2
            >>> cm1, qreq_ = ibeis.testdata_cm()
            >>> json_str = cm1.to_json()
            >>> cm = ChipMatch2.from_json(json_str)
            >>> ut.quit_if_noshow()
            >>> cm.score_nsum(qreq_)
            >>> cm.show_single_namematch(qreq_, 1)
            >>> ut.show_if_requested()
        """

        def convert_numpy_lists(arr_list, dtype):
            return [np.array(arr, dtype=dtype) for arr in arr_list]

        def convert_numpy(arr, dtype):
            return np.array(ut.replace_nones(arr, np.nan), dtype=dtype)

        class_dict = ut.from_json(json_str)
        key_list = ut.get_kwargs(cls.initialize)[0]  # HACKY
        key_list.remove('autoinit')
        if ut.VERBOSE:
            other_keys = list(set(class_dict.keys()) - set(key_list))
            if len(other_keys) > 0:
                print('Not unserializing extra attributes: %s' % (ut.list_str(other_keys)))
        dict_subset = ut.dict_subset(class_dict, key_list)
        dict_subset['fm_list'] = convert_numpy_lists(dict_subset['fm_list'],
                                                     hstypes.FM_DTYPE)
        dict_subset['fsv_list'] = convert_numpy_lists(dict_subset['fsv_list'],
                                                      hstypes.FS_DTYPE)
        dict_subset['score_list'] = convert_numpy(dict_subset['score_list'],
                                                  hstypes.FS_DTYPE)
        cm = cls(**dict_subset)
        return cm

    def to_json(cm):
        r"""
        Serialize ChipMatch object as JSON string

        CommandLine:
            python -m ibeis.model.hots.chip_match --test-ChipMatch2.to_json:0
            python -m ibeis.model.hots.chip_match --test-ChipMatch2.to_json
            python -m ibeis.model.hots.chip_match --test-ChipMatch2.to_json:1 --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> # Simple doctest demonstrating the json format
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb(defaultdb='testdb1')
            >>> cm, qreq_ = ibs.query_chips(1, [2, 3, 4, 5], return_cm=True, return_request=True)
            >>> cm.compress_feature_matches(num=4, rng=np.random.RandomState(0))
            >>> # Serialize
            >>> print('\n\nRaw ChipMatch2 JSON:\n')
            >>> json_str = cm.to_json()
            >>> print(json_str)
            >>> print('\n\nPretty ChipMatch2 JSON:\n')
            >>> # Pretty String Formatting
            >>> dictrep = ut.from_json(json_str)
            >>> dictrep = ut.delete_dict_keys(dictrep, [key for key, val in dictrep.items() if val is None])
            >>> result  = ut.dict_str(dictrep, nl=2, precision=2, hack_liststr=True, key_order_metric='strlen')
            >>> result = result.replace('u\'', '"').replace('\'', '"')
            >>> print(result)

        Example:
            >>> # ENABLE_DOCTEST
            >>> # test to convert back and forth from json
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> import ibeis
            >>> cm, qreq_ = ibeis.testdata_cm()
            >>> cm1 = cm
            >>> # Serialize
            >>> json_str = cm.to_json()
            >>> print(repr(json_str))
            >>> # Unserialize
            >>> cm = ChipMatch2.from_json(json_str)
            >>> # Show if it works
            >>> ut.quit_if_noshow()
            >>> cm.score_nsum(qreq_)
            >>> cm.show_single_namematch(qreq_, 1)
            >>> ut.show_if_requested()
            >>> # result = ('json_str = \n%s' % (str(json_str),))
            >>> # print(result)

        """
        json_str = ut.to_json(cm.__dict__)
        return json_str

    def as_dict(cm):
        return cm.__getstate__()

    def as_simple_dict(cm, keys=[]):
        state_dict = cm.__getstate__()
        keys = ['qaid', 'daid_list', 'score_list'] + keys
        simple_dict = ut.dict_subset(state_dict, keys)
        return simple_dict

    def __getstate__(cm):
        state_dict = cm.__dict__
        return state_dict

    def __setstate__(cm, state_dict):
        cm.__dict__.update(state_dict)

    # --- IO

    def get_fpath(cm, qreq_):
        dpath = qreq_.get_qresdir()
        fname = get_chipmatch_fname(cm.qaid, qreq_)
        fpath = join(dpath, fname)
        return fpath

    def save(cm, qreq_, verbose=None):
        fpath = cm.get_fpath(qreq_)
        cm.save_to_fpath(fpath, verbose=verbose)

    def save_to_fpath(cm, fpath, verbose=None):
        """
        CommandLine:
            python -m ibeis.model.hots.chip_match --exec-ChipMatch2.save_to_fpath --verbtest --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> qaid = 18
            >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list=[qaid])
            >>> cm = cm_list[0]
            >>> dpath = ut.get_app_resource_dir('ibeis')
            >>> fpath = join(dpath, 'tmp_chipmatch.cPkl')
            >>> ut.delete(fpath)
            >>> cm.save_to_fpath(fpath)
            >>> cm2 = ChipMatch2.load_from_fpath(fpath)
            >>> assert cm == cm2
            >>> ut.quit_if_noshow()
            >>> cm.ishow_analysis(qreq_)
            >>> ut.show_if_requested()
        """
        #ut.save_data(fpath, cm.__getstate__(), verbose=verbose)
        ut.save_cPkl(fpath, cm.__getstate__(), verbose=verbose)

    @classmethod
    def load(cls, qreq_, qaid, dpath=None, verbose=None):
        fname = get_chipmatch_fname(qaid, qreq_)
        if dpath is None:
            dpath = qreq_.get_qresdir()
        fpath = join(dpath, fname)
        cm = cls.load_from_fpath(fpath, verbose=verbose)
        return cm

    @classmethod
    def load_from_fpath(cls, fpath, verbose=None):
        #state_dict = ut.load_data(fpath, verbose=verbose)
        state_dict = ut.load_cPkl(fpath, verbose=verbose)
        cm = cls()
        cm.__setstate__(state_dict)
        return cm

    # ---

    def compress_feature_matches(cm, num=10, rng=np.random, use_random=True):
        """
        Removes all but the best feature matches for testing purposes
        rng = np.random.RandomState(0)
        """
        #num = 10
        fs_list = cm.get_fsv_prod_list()
        score_sortx = [fs.argsort()[::-1] for fs in fs_list]
        if use_random:
            # keep jagedness
            score_sortx_filt = [
                sortx[0:min(rng.randint(num // 2, num), len(sortx))]
                for sortx in score_sortx]
        else:
            score_sortx_filt = [sortx[0:min(num, len(sortx))]
                                for sortx in score_sortx]
        cm.fsv_list = vt.ziptake(cm.fsv_list, score_sortx_filt, axis=0)
        cm.fm_list = vt.ziptake(cm.fm_list, score_sortx_filt, axis=0)
        cm.fk_list = vt.ziptake(cm.fk_list, score_sortx_filt, axis=0)
        if cm.fs_list is not None:
            cm.fs_list = vt.ziptake(cm.fs_list, score_sortx_filt, axis=0)
        cm.H_list = None
        cm.fs_list = None

    # Override eequality

    def __eq__(cm, other):
        if isinstance(other, cm.__class__):
            flag = True
            flag &= len(cm.fm_list) == len(other.fm_list)
            def check_arrs_eq(arr1, arr2):
                if arr1 is None and arr2 is None:
                    return True
                elif len(arr1) != len(arr2):
                    return False
                elif any(len(x) != len(y) for x, y in zip(arr1, arr2)):
                    return False
                elif all(np.all(x == y) for x, y in zip(arr1, arr2)):
                    return True
                else:
                    return False
            flag &= cm.qaid == other.qaid
            flag &= cm.qnid == other.qnid
            flag &= check_arrs_eq(cm.fm_list, other.fm_list)
            flag &= check_arrs_eq(cm.fs_list, other.fs_list)
            flag &= check_arrs_eq(cm.fk_list, other.fk_list)
            return flag
            #return cm.__dict__ == other.__dict__
        else:
            return False

    #------------------
    # Getter Functions
    #------------------

    def get_num_feat_score_cols(cm):
        return len(cm.fsv_col_lbls)

    def get_fs(cm, idx=None, colx=None, daid=None, col=None):
        assert xor(idx is None, daid is None)
        assert xor(colx is None or col is None)
        if daid is not None:
            idx = cm.daid2_idx[daid]
        if col is not None:
            colx = cm.fsv_col_lbls.index(col)
        fs = cm.fsv_list[idx][colx]
        return fs

    def get_fsv_prod_list(cm):
        return [fsv.prod(axis=1) for fsv in cm.fsv_list]

    def get_annot_fm(cm, daid):
        idx = ut.dict_take(cm.daid2_idx, daid)
        fm  = ut.list_take(cm.fm_list, idx)
        return fm

    def get_fs_list(cm, colx=None, col=None):
        assert xor(colx is None, col is None)
        if col is not None:
            colx = cm.fsv_col_lbls.index(col)
        fs_list = [fsv.T[colx].T for fsv in cm.fsv_list]
        return fs_list

    def get_groundtruth_flags(cm):
        assert cm.dnid_list is not None, 'run cm.evaluate_dnids'
        gt_flags = cm.dnid_list == cm.qnid
        return gt_flags

    def get_groundtruth_daids(cm):
        gt_flags = cm.get_groundtruth_flags()
        gt_daids = vt.list_compress_(cm.daid_list, gt_flags)
        return gt_daids

    def get_nid_scores(cm, nid_list):
        nidx_list = ut.dict_take(cm.nid2_nidx, nid_list)
        name_scores = vt.list_take_(cm.name_score_list, nidx_list)
        return name_scores

    def get_ranked_nids(cm):
        sortx = cm.name_score_list.argsort()[::-1]
        sorted_name_scores = cm.name_score_list.take(sortx, axis=0)
        sorted_nids = cm.unique_nids.take(sortx, axis=0)
        return sorted_nids, sorted_name_scores

    def get_ranked_nids_and_aids(cm):
        """ Hacky func """
        sortx = cm.name_score_list.argsort()[::-1]
        sorted_name_scores = cm.name_score_list.take(sortx, axis=0)
        sorted_nids = cm.unique_nids.take(sortx, axis=0)
        sorted_groupxs = ut.list_take(cm.name_groupxs, sortx)
        sorted_daids = vt.apply_grouping(cm.daid_list,  sorted_groupxs)
        sorted_annot_scores = vt.apply_grouping(cm.annot_score_list,  sorted_groupxs)
        # do subsorting
        subsortx_list = [scores.argsort()[::-1] for scores in sorted_annot_scores]
        subsorted_daids = vt.ziptake(sorted_daids, subsortx_list)
        subsorted_annot_scores = vt.ziptake(sorted_annot_scores, subsortx_list)
        nscoretup = name_scoring.NameScoreTup(sorted_nids, sorted_name_scores,
                                              subsorted_daids,
                                              subsorted_annot_scores)
        return nscoretup

    def get_num_matches_list(cm):
        num_matches_list = list(map(len, cm.fm_list))
        return num_matches_list

    def get_name_shortlist_aids(cm, nNameShortList, nAnnotPerName):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list=[18])
            >>> cm = cm_list[0]
            >>> cm.score_nsum(qreq_)
            >>> top_daids = cm.get_name_shortlist_aids(5, 2)
            >>> assert cm.qnid in ibs.get_annot_name_rowids(top_daids)
        """
        top_daids = scoring.get_name_shortlist_aids(
            cm.daid_list, cm.dnid_list, cm.annot_score_list,
            cm.name_score_list, cm.nid2_nidx, nNameShortList, nAnnotPerName)
        return top_daids

    def get_chip_shortlist_aids(cm, num_shortlist):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list=[18])
            >>> cm = cm_list[0]
            >>> cm.score_nsum(qreq_)
            >>> top_daids = cm.get_chip_shortlist_aids(5 * 2)
            >>> assert cm.qnid in ibs.get_annot_name_rowids(top_daids)
        """
        sortx = np.array(cm.annot_score_list).argsort()[::-1]
        topx = sortx[:min(num_shortlist, len(sortx))]
        top_daids = cm.daid_list[topx]
        return top_daids

    def argsort(cm):
        #if cm.score_list is None:
        #    num_matches_list = cm.get_num_matches_list()
        #    sortx = ut.list_argsort(num_matches_list, reverse=True)
        #else:
        #sortx = ut.list_argsort(cm.score_list, reverse=True)
        sortx = np.argsort(cm.score_list)[::-1]
        return sortx
        #return np.array(sortx)

    def name_argsort(cm):
        #return np.array(ut.list_argsort(cm.name_score_list, reverse=True))
        return np.argsort(cm.name_score_list)[::-1]

    @property
    def ranks(cm):
        sortx = cm.argsort()
        return sortx.argsort()

    @property
    def unique_name_ranks(cm):
        sortx = cm.name_argsort()
        return sortx.argsort()

    #+=================
    # Score Aggregation Functions
    #------------------

    # Cannonical Setters

    @profile
    def set_cannonical_annot_score(cm, annot_score_list):
        cm.annot_score_list = annot_score_list
        #cm.name_score_list  = None
        cm.score_list       = annot_score_list

    @profile
    def set_cannonical_name_score(cm, annot_score_list, name_score_list):
        cm.annot_score_list = annot_score_list
        cm.name_score_list  = name_score_list
        # align with score_list
        cm.score_list = name_scoring.align_name_scores_with_annots(
            cm.annot_score_list, cm.daid_list, cm.daid2_idx, cm.name_groupxs,
            cm.name_score_list)

    # --- ChipSum Score

    @profile
    def evaluate_csum_score(cm, qreq_):
        csum_score_list = scoring.compute_csum_score(cm)
        cm.csum_score_list = csum_score_list

    @profile
    def score_csum(cm, qreq_):
        """
        CommandLine:
            python -m ibeis.model.hots.chip_match --test-score_csum --show
            python -m ibeis.model.hots.chip_match --test-score_csum --show --qaid 18

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver()
            >>> cm = cm_list[0]
            >>> cm.score_csum(qreq_)
            >>> ut.quit_if_noshow()
            >>> cm.show_ranked_matches(qreq_, figtitle='score_csum')
            >>> ut.show_if_requested()
        """
        cm.evaluate_csum_score(qreq_)
        cm.set_cannonical_annot_score(cm.csum_score_list)

    # --- MaxChipSum Score

    @profile
    def score_maxcsum(cm, qreq_):
        cm.evaluate_dnids(qreq_.ibs)
        cm.score_csum(qreq_)
        cm.maxcsum_score_list = np.array([
            scores.max()
            for scores in vt.apply_grouping(cm.csum_score_list,
                                            cm.name_groupxs)
        ])
        cm.set_cannonical_name_score(cm.csum_score_list, cm.maxcsum_score_list)

    # --- NameSum Score

    @profile
    def evaluate_nsum_score(cm, qreq_):
        cm.evaluate_dnids(qreq_.ibs)
        nsum_nid_list, nsum_score_list = name_scoring.compute_nsum_score(cm, qreq_=qreq_)
        assert np.all(cm.unique_nids == nsum_nid_list), 'name score not in alignment'
        cm.nsum_score_list = nsum_score_list

    @profile
    def score_nsum(cm, qreq_):
        """
        CommandLine:
            python -m ibeis.model.hots.chip_match --test-score_nsum --show --qaid 1
            python -m ibeis.model.hots.chip_match --test-score_nsum --show --qaid 18

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18])
            >>> cm = cm_list[0]
            >>> cm.score_nsum(qreq_)
            >>> gt_score = cm.score_list.compress(cm.get_groundtruth_flags()).max()
            >>> cm.print_csv()
            >>> assert cm.get_top_nids()[0] == cm.unique_nids[cm.name_score_list.argmax()], 'bug in alignment'
            >>> ut.quit_if_noshow()
            >>> cm.show_ranked_matches(qreq_, figtitle='score_nsum')
            >>> ut.show_if_requested()
            >>> assert cm.get_top_nids()[0] == cm.qnid, 'is this case truely hard?'
        """
        cm.evaluate_csum_score(qreq_)
        cm.evaluate_nsum_score(qreq_)
        cm.set_cannonical_name_score(cm.csum_score_list, cm.nsum_score_list)

    # --- ChipCoverage Score

    @profile
    def evaluate_acov_score(cm, qreq_):
        daid_list, acov_score_list = scoring.compute_annot_coverage_score(
            qreq_, cm, qreq_.qparams)
        assert np.all(daid_list == np.array(cm.daid_list)), 'daids out of alignment'
        cm.acov_score_list = acov_score_list

    @profile
    def score_annot_coverage(cm, qreq_):
        """
        CommandLine:
            python -m ibeis.model.hots.chip_match --test-score_annot_coverage --show
            python -m ibeis.model.hots.chip_match --test-score_annot_coverage --show --qaid 18

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver()
            >>> cm = cm_list[0]
            >>> cm.fs_list = cm.get_fs_list(col='lnbnn')
            >>> cm.score_annot_coverage(qreq_)
            >>> ut.quit_if_noshow()
            >>> cm.show_ranked_matches(qreq_, figtitle='score_annot_coverage')
            >>> ut.show_if_requested()
        """
        cm.evaluate_acov_score(qreq_)
        cm.set_cannonical_annot_score(cm.acov_score_list)

    # --- NameCoverage Score

    @profile
    def evaluate_ncov_score(cm, qreq_):
        cm.evaluate_dnids(qreq_.ibs)
        ncov_nid_list, ncov_score_list = scoring.compute_name_coverage_score(
            qreq_, cm, qreq_.qparams)
        assert np.all(cm.unique_nids == ncov_nid_list)
        cm.ncov_score_list = ncov_score_list

    @profile
    def score_name_coverage(cm, qreq_):
        """
        CommandLine:
            python -m ibeis.model.hots.chip_match --test-score_name_coverage --show
            python -m ibeis.model.hots.chip_match --test-score_name_coverage --show --qaid 18

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver()
            >>> cm = cm_list[0]
            >>> cm.fs_list = cm.get_fs_list(col='lnbnn')
            >>> cm.score_name_coverage(qreq_)
            >>> ut.quit_if_noshow()
            >>> cm.show_ranked_matches(qreq_, figtitle='score_name_coverage')
            >>> ut.show_if_requested()
        """
        if cm.csum_score_list is None:
            cm.evaluate_csum_score(qreq_)
        cm.evaluate_ncov_score(qreq_)
        cm.set_cannonical_name_score(cm.csum_score_list, cm.ncov_score_list)

    #------------------
    # Result Functions
    #------------------

    def get_top_scores(cm, ntop=None):
        sortx = cm.score_list.argsort()[::-1]
        _top_scores = vt.list_take_(cm.score_list, sortx)
        top_scores = ut.listclip(_top_scores, ntop)
        return top_scores

    def get_top_nids(cm, ntop=None):
        sortx = cm.score_list.argsort()[::-1]
        _top_nids = vt.list_take_(cm.dnid_list, sortx)
        top_nids = ut.listclip(_top_nids, ntop)
        return top_nids

    def get_top_aids(cm, ntop=None):
        sortx = cm.score_list.argsort()[::-1]
        _top_aids = vt.list_take_(cm.daid_list, sortx)
        top_aids = ut.listclip(_top_aids, ntop)
        return top_aids

    def get_top_truth_aids(cm, ibs, truth, ntop=None):
        """
        """
        sortx = cm.score_list.argsort()[::-1]
        _top_aids = vt.list_take_(cm.daid_list, sortx)
        truth_list = ibs.get_aidpair_truths([cm.qaid] * len(_top_aids), _top_aids)
        flag_list = truth_list == truth
        _top_aids = _top_aids.compress(flag_list, axis=0)
        top_truth_aids = ut.listclip(_top_aids, ntop)
        return top_truth_aids

    def get_top_gf_aids(cm, ibs, ntop=None):
        import ibeis
        return cm.get_top_truth_aids(ibs, ibeis.const.TRUTH_NOT_MATCH, ntop)

    def get_top_gt_aids(cm, ibs, ntop=None):
        import ibeis
        return cm.get_top_truth_aids(ibs, ibeis.const.TRUTH_MATCH, ntop)

    def get_annot_scores(cm, daids, score_method=None):
        #idx_list = [cm.daid2_idx.get(daid, None) for daid in daids]
        score_list = cm.score_list
        idx_list = ut.dict_take(cm.daid2_idx, daids, None)
        score_list = [None if idx is None else score_list[idx]
                      for idx in idx_list]
        return score_list

    def get_annot_ranks(cm, daids):  # score_method=None):
        score_ranks = cm.score_list.argsort()[::-1].argsort()
        idx_list = ut.dict_take(cm.daid2_idx, daids, None)
        rank_list = [None if idx is None else score_ranks[idx]
                      for idx in idx_list]
        return rank_list

    def get_name_ranks(cm, dnids):  # score_method=None):
        score_ranks = cm.name_score_list.argsort()[::-1].argsort()
        idx_list = ut.dict_take(cm.nid2_nidx, dnids, None)
        rank_list = [None if idx is None else score_ranks[idx]
                      for idx in idx_list]
        return rank_list

    #------------------
    # String Functions
    #------------------

    def print_inspect_str(cm, qreq_):
        print(cm.get_inspect_str(qreq_))

    def get_inspect_str(cm, qreq_):
        r"""
        Args:
            qreq_ (QueryRequest):  query request object with hyper-parameters

        Returns:
            str: varinfo

        CommandLine:
            python -m ibeis.model.hots.chip_match --exec-get_inspect_str

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> import ibeis
            >>> cm, qreq_ = ibeis.testdata_cm()
            >>> varinfo = cm.get_inspect_str(qreq_)
            >>> result = ('varinfo = %s' % (str(varinfo),))
            >>> print(result)
        """
        cm.assert_self(qreq_)
        #ut.embed()

        top_lbls = [' top aids', ' scores', ' ranks']

        ibs = qreq_.ibs

        top_aids   = np.array(cm.get_top_aids(6), dtype=np.int32)
        top_scores = np.array(cm.get_annot_scores(top_aids), dtype=np.float64)
        #top_rawscores = np.array(cm.get_aid_scores(top_aids, rawscore=True), dtype=np.float64)
        top_ranks  = np.arange(len(top_aids))
        top_list   = [top_aids, top_scores, top_ranks]

        top_lbls += [' isgt']
        istrue = ibs.get_aidpair_truths([cm.qaid] * len(top_aids), top_aids)
        top_list.append(np.array(istrue, dtype=np.int32))

        top_lbls = ['top nid'] + top_lbls
        top_list = [ibs.get_annot_name_rowids(top_aids)] + top_list

        top_stack = np.vstack(top_list)
        #top_stack = np.array(top_stack, dtype=object)
        top_stack = np.array(top_stack, dtype=np.float)
        #np.int32)
        top_str = np.array_str(top_stack, precision=3, suppress_small=True, max_line_width=200)

        top_lbl = '\n'.join(top_lbls)
        inspect_list = ['QueryResult', qreq_.get_cfgstr(), ]
        if ibs is not None:
            gt_aids = cm.get_top_gt_aids(qreq_.ibs)
            gt_ranks  = cm.get_annot_ranks(gt_aids)
            gt_scores = cm.get_annot_scores(gt_aids)
            inspect_list.append('len(cm.daid_list) = %r' % len(cm.daid_list))
            inspect_list.append('len(cm.unique_nids) = %r' % len(cm.unique_nids))
            inspect_list.append('gt_ranks = %r' % gt_ranks)
            inspect_list.append('gt_aids = %r' % gt_aids)
            inspect_list.append('gt_scores = %r' % gt_scores)

        inspect_list.extend([
            'qaid=%r ' % cm.qaid,
            'qnid=%r ' % cm.qnid,
            ut.hz_str(top_lbl, ' ', top_str),
            #'num feat matches per annotation stats:',
            #ut.indent(ut.dict_str(nFeatMatch_stats)),
            #ut.indent(nFeatMatch_stats_str),
        ])

        inspect_str = '\n'.join(inspect_list)

        #inspect_str = ut.indent(inspect_str, '[INSPECT] ')
        return inspect_str

    def print_rawinfostr(cm):
        print(cm.get_rawinfostr())

    def print_csv(cm, *args, **kwargs):
        print(cm.get_cvs_str(*args, **kwargs))

    def get_rawinfostr(cm):
        def varinfo(varname, onlyrepr=False, canshowrepr=True, cm=cm):
            import utool as ut
            varval = getattr(cm, varname.replace('cm.', ''))
            show_if_smaller_than = 7
            if canshowrepr:
                if hasattr(varval, 'size'):
                    show_repr = ut.isiterable(varval) and getattr(varval, 'size', 100) < show_if_smaller_than
                else:
                    show_repr = ut.isiterable(varval) and len(varval) < show_if_smaller_than
            else:
                show_repr = False
            varinfo_list = []
            print_summary = not onlyrepr and ut.isiterable(varval)
            show_repr = show_repr or (onlyrepr or not print_summary)
            symbol = '*'
            if show_repr:
                varinfo_list += ['    * %s = %r' % (varname, varval)]
                symbol = '+'
            if print_summary:
                varinfo_list += [
                    '    %s varinfo(%s):' % (symbol, varname,),
                    '        depth = %r' % (ut.depth_profile(varval),),
                    '        types = %s' % (ut.list_type_profile(varval),),
                ]
                #varinfo = '\n'.join(ut.align_lines(varinfo_list, '='))
            varinfo = '\n'.join(ut.align_lines(varinfo_list, '='))
            return varinfo
        str_list = []
        append = str_list.append
        append('ChipMatch2:')
        append('    * cm.qaid = %r' % (cm.qaid,))
        append('    * cm.qnid = %r' % (cm.qnid,))
        #append('    * len(cm.daid2_idx) = %r' % (len(cm.daid2_idx),))
        append(varinfo('cm.daid2_idx'))
        append(varinfo('cm.fsv_col_lbls', onlyrepr=True))
        append(varinfo('cm.daid_list'))
        append(varinfo('cm.dnid_list'))
        append(varinfo('cm.fs_list'))
        append(varinfo('cm.fm_list'))
        append(varinfo('cm.fk_list'))
        append(varinfo('cm.fsv_list'))
        append(varinfo('cm.H_list', canshowrepr=False))
        append(varinfo('cm.score_list'))
        append(varinfo('cm.annot_score_list'))
        #
        append(varinfo('cm.csum_score_list'))
        append(varinfo('cm.acov_score_list'))
        #
        append(varinfo('cm.unique_nids'))
        append(varinfo('cm.nid2_nidx'))
        append(varinfo('cm.name_score_list'))
        append(varinfo('cm.nsum_score_list'))
        append(varinfo('cm.ncov_score_list'))
        #append(varinfo('cm.annot_score_dict[\'csum\']'))
        #append(varinfo('cm.annot_score_dict[\'acov\']'))
        #append(varinfo('cm.name_score_dict[\'nsum\']'))
        #append(varinfo('cm.name_score_dict[\'ncov\']'))
        #infostr = '\n'.join(ut.align_lines(str_list, '='))
        infostr = '\n'.join(str_list)
        return infostr

    def get_cvs_str(cm,  numtop=6, ibs=None, sort=True):
        r"""
        Args:
            numtop (int): (default = 6)
            ibs (IBEISController):  ibeis controller object(default = None)
            sort (bool): (default = True)

        Returns:
            str: csv_str

        Notes:
            Very weird that it got a score
            qaid 6 vs 41 has
                [72, 79, 0, 17, 6, 60, 15, 36, 63]
                [72, 79, 0, 17, 6, 60, 15, 36, 63]
                [72, 79, 0, 17, 6, 60, 15, 36, 63]
                [0.060, 0.053, 0.0497, 0.040, 0.016, 0, 0, 0, 0]
                [7, 40, 41, 86, 103, 88, 8, 101, 35]
            makes very little sense

        CommandLine:
            python -m ibeis.model.hots.chip_match --test-get_cvs_str --force-serial

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver()
            >>> cm = cm_list[0]
            >>> numtop = 6
            >>> ibs = None
            >>> sort = True
            >>> csv_str = cm.get_cvs_str(numtop, ibs, sort)
            >>> result = ('csv_str = \n%s' % (str(csv_str),))
            >>> print(result)
        """
        if not sort or cm.score_list is None:
            if sort:
                print('Warning: cm.score_list is None and sort is True')
            sortx = list(range(len(cm.daid_list)))
        else:
            sortx = ut.list_argsort(cm.score_list, reverse=True)
        if ibs is not None:
            qnid = ibs.get_annot_nids(cm.qaid)
            dnid_list = ibs.get_annot_nids(cm.daid_list)
        else:
            qnid = cm.qnid
            dnid_list = cm.dnid_list
        # Build columns for the csv, filtering out unavailable information
        column_lbls_ = ['daid', 'dnid', 'score', 'num_matches', 'annot_scores',
                        'fm_depth', 'fsv_depth']
        column_list_ = [
            vt.list_take_(cm.daid_list,  sortx),
            None if dnid_list is None else vt.list_take_(dnid_list, sortx),
            None if cm.score_list is None else vt.list_take_(cm.score_list, sortx),
            vt.list_take_(cm.get_num_matches_list(), sortx),
            None if cm.annot_score_list is None else vt.list_take_(cm.annot_score_list, sortx),
            #None if cm.name_score_list is None else vt.list_take_(cm.name_score_list, sortx),
            ut.lmap(str, ut.depth_profile(vt.list_take_(cm.fm_list,  sortx))),
            ut.lmap(str, ut.depth_profile(vt.list_take_(cm.fsv_list, sortx))),
        ]
        isnone_list = ut.flag_None_items(column_list_)
        column_lbls = ut.filterfalse_items(column_lbls_, isnone_list)
        column_list = ut.filterfalse_items(column_list_, isnone_list)
        # Clip to the top results
        if numtop is not None:
            column_list = [ut.listclip(col, numtop) for col in column_list]
        # hard case for python text parsing
        # better know about quoted hash symbols
        header = ut.codeblock(
            '''
            # qaid = {qaid}
            # qnid = {qnid}
            # fsv_col_lbls = {fsv_col_lbls}
            '''
        ).format(qaid=cm.qaid, qnid=qnid, fsv_col_lbls=cm.fsv_col_lbls)

        csv_str = ut.make_csv_table(column_list, column_lbls, header, comma_repl=';')
        return csv_str

    #------------------
    # Testing Functions
    #------------------

    def assert_self(cm, qreq_=None, strict=False, verbose=ut.NOT_QUIET):
        assert cm.qaid is not None, 'must have qaid'
        assert cm.daid_list is not None, 'must give daids'
        assert cm.fm_list is None or len(cm.fm_list) == len(cm.daid_list), 'incompatable data'
        assert cm.fsv_list is None or len(cm.fsv_list) == len(cm.daid_list), 'incompatable data'
        assert cm.fk_list is None or len(cm.fk_list) == len(cm.daid_list), 'incompatable data'
        assert cm.H_list is None or len(cm.H_list) == len(cm.daid_list), 'incompatable data'
        assert cm.score_list is None or len(cm.score_list) == len(cm.daid_list), 'incompatable data'
        assert cm.dnid_list is None or len(cm.dnid_list) == len(cm.daid_list), 'incompatable data'

        class TestLogger(object):
            def __init__(testlog):
                testlog.test_out = ut.ddict(list)
                testlog.current_test = None
                testlog.failed_list = []

            def start_test(testlog, name):
                testlog.current_test = name

            def log_skipped(testlog, msg):
                if verbose:
                    print('[cm] skip: ' + msg)

            def log_passed(testlog, msg):
                if verbose:
                    print('[cm] pass: ' + msg)

            def skip_test(testlog):
                testlog.log_skipped(testlog.current_test)
                testlog.current_test = None

            def log_failed(testlog, msg):
                testlog.test_out[testlog.current_test].append(msg)
                testlog.failed_list.append(msg)
                print('[cm] FAILED!: ' + msg)

            def end_test(testlog):
                if len(testlog.test_out[testlog.current_test]) == 0:
                    testlog.log_passed(testlog.current_test)
                else:
                    testlog.log_failed(testlog.current_test)
                testlog.current_test = None

            def context(testlog, name):
                testlog.start_test(name)
                return testlog

            def __enter__(testlog):
                return testlog

            def __exit__(testlog, a, b, c):
                if testlog.current_test is not None:
                    testlog.end_test()

        testlog = TestLogger()

        with testlog.context('lookup score by daid'):
            if cm.score_list is None:
                testlog.skip_test()
            else:
                daids = cm.get_top_aids()
                scores = cm.get_top_scores()
                scores_ = cm.get_annot_scores(daids)
                if not np.all(scores == scores_):
                    testlog.log_failed('score mappings are NOT ok')

        with testlog.context('dnid_list = name(daid_list'):
            if strict or qreq_ is not None and cm.dnid_list is not None:
                if not np.all(cm.dnid_list == qreq_.ibs.get_annot_name_rowids(cm.daid_list)):
                    testlog.log_failed('annot aligned nids are NOT ok')
            else:
                testlog.skip_test()

        if strict or cm.unique_nids is not None:
            with testlog.context('unique nid mapping'):
                nidx_list = ut.dict_take(cm.nid2_nidx, cm.unique_nids)
                assert nidx_list == list(range(len(nidx_list)))
                assert np.all(cm.unique_nids[nidx_list] == cm.unique_nids)

            with testlog.context('allsame(grouped(dnid_list))'):
                grouped_nids = vt.apply_grouping(cm.dnid_list, cm.name_groupxs)
                for nids in grouped_nids:
                    if not ut.list_allsame(nids):
                        testlog.log_failed('internal dnid name grouping is NOT consistent')

            with testlog.context('allsame(name(grouped(daid_list)))'):
                if qreq_ is None:
                    testlog.skip_test()
                else:
                    # this might fail if this result is old and the names have changed
                    grouped_aids = vt.apply_grouping(cm.daid_list, cm.name_groupxs)
                    grouped_mapped_nids = qreq_.ibs.unflat_map(qreq_.ibs.get_annot_name_rowids, grouped_aids)
                    for nids in grouped_mapped_nids:
                        if not ut.list_allsame(nids):
                            testlog.log_failed('internal daid name grouping is NOT consistent')

            with testlog.context('dnid_list - unique_nid alignment'):
                grouped_nids = vt.apply_grouping(cm.dnid_list, cm.name_groupxs)
                for nids, nid in zip(grouped_nids, cm.unique_nids):
                    if not np.all(nids == nid):
                        testlog.log_failed(
                            'cm.unique_nids is NOT aligned with '
                            'vt.apply_grouping(cm.dnid_list, cm.name_groupxs). '
                            ' nids=%r, nid=%r' % (nids, nid)
                        )
                        break

            if qreq_ is not None:
                testlog.start_test('daid_list - unique_nid alignment')
                for nids, nid in zip(grouped_mapped_nids, cm.unique_nids):
                    if not np.all(nids == nid):
                        testlog.log_failed(
                            'cm.unique_nids is NOT aligned with '
                            'vt.apply_grouping(name(cm.daid_list), cm.name_groupxs). '
                            ' name(aids)=%r, nid=%r' % (nids, nid)
                        )
                        break
                testlog.end_test()

        assert len(testlog.failed_list) == 0, '\n'.join(testlog.failed_list)
        testlog.log_passed('lengths are ok')

        try:
            assert ut.list_all_eq_to([fsv.shape[1] for fsv in cm.fsv_list], len(cm.fsv_col_lbls))
        except Exception as ex:
            cm.print_rawinfostr()
            raise
        assert ut.list_all_eq_to([fm.shape[1] for fm in cm.fm_list], 2), 'bad fm'
        testlog.log_passed('shapes are ok')

        if strict or qreq_ is not None:
            external_qaids = qreq_.get_external_qaids().tolist()
            external_daids = qreq_.get_external_daids().tolist()
            if qreq_.qparams.pipeline_root == 'vsone':
                assert len(external_qaids) == 1, 'only one external qaid for vsone'
                if strict or qreq_.indexer is not None:
                    nExternalQVecs = qreq_.ibs.get_annot_vecs(
                        external_qaids[0],
                        config2_=qreq_.get_external_query_config2()).shape[0]
                    assert qreq_.indexer.idx2_vec.shape[0] == nExternalQVecs, (
                        'did not index query descriptors properly')
                testlog.log_passed('vsone daids are ok are ok')

            nFeats1 = qreq_.ibs.get_annot_num_feats(
                cm.qaid, config2_=qreq_.get_external_query_config2())
            nFeats2_list = np.array(
                qreq_.ibs.get_annot_num_feats(
                    cm.daid_list, config2_=qreq_.get_external_data_config2()))
            try:
                assert ut.list_issubset(cm.daid_list, external_daids), (
                    'cmtup_old must be subset of daids')
            except AssertionError as ex:
                ut.printex(ex, keys=['daid_list', 'external_daids'])
                raise
            try:
                fm_list = cm.fm_list
                fx2s_list = [fm_.T[1] for fm_ in fm_list]
                fx1s_list = [fm_.T[0] for fm_ in fm_list]
                max_fx1_list = np.array([
                    -1 if len(fx1s) == 0 else fx1s.max()
                    for fx1s in fx1s_list])
                max_fx2_list = np.array([
                    -1 if len(fx2s) == 0 else fx2s.max()
                    for fx2s in fx2s_list])
                ut.assert_lessthan(max_fx2_list, nFeats2_list,
                                   'max feat index must be less than num feats')
                ut.assert_lessthan(max_fx1_list, nFeats1,
                                   'max feat index must be less than num feats')
            except AssertionError as ex:
                ut.printex(ex, keys=['qaid', 'daid_list', 'nFeats1',
                                     'nFeats2_list', 'max_fx1_list',
                                     'max_fx2_list', ])
                raise
            testlog.log_passed('nFeats are ok in fm')
        else:
            testlog.log_skipped('nFeat check')

        if qreq_ is not None:
            pass

    def show_single_namematch(cm, qreq_, dnid, fnum=None, pnum=None,
                              homog=ut.get_argflag('--homog'), **kwargs):
        """

        CommandLine:
            python -m ibeis --tf ChipMatch2.show_single_namematch --show
            python -m ibeis --tf ChipMatch2.show_single_namematch --show --qaid 1
            python -m ibeis --tf ChipMatch2.show_single_namematch --show --qaid 1 --dpath figures --save ~/latex/crall-candidacy-2015/figures/namematch.jpg

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> import ibeis
            >>> cm, qreq_ = ibeis.testdata_cm('PZ_MTEST', default_qaids=[18])
            >>> homog = False
            >>> dnid = cm.qnid
            >>> cm.show_single_namematch(qreq_, dnid)
            >>> ut.quit_if_noshow()
            >>> ut.show_if_requested()
        """
        from ibeis.viz import viz_matches
        qaid = cm.qaid
        if cm.nid2_nidx is None:
            raise AssertionError('cm.nid2_nidx has not been evaluated yet')
            #cm.score_nsum(qreq_)
        # <GET NAME GROUPXS>
        try:
            nidx = cm.nid2_nidx[dnid]
            #if nidx == 144:
            #    raise
        except KeyError:
            #def extend():
                #pass
            #cm.daid_list
            #cm.print_inspect_str(qreq_)
            #cm_orig = cm  # NOQA
            #cm_orig.assert_self(qreq_)
            #other_aids = qreq_.get_external_daids()
            # Hack to get rid of key error
            cm.assert_self(verbose=False)
            cm2 = cm.extend_results(qreq_)
            cm2.assert_self(verbose=False)
            cm = cm2
            #cm2.assert_self(qreq_)
            #ut.embed()
            nidx = cm.nid2_nidx[dnid]
            #raise
        groupxs = cm.name_groupxs[nidx]
        daids = np.take(cm.daid_list, groupxs)
        dnids = np.take(cm.dnid_list, groupxs)
        assert np.all(dnid == dnids), (
            'inconsistent naming, dnid=%r, dnids=%r' % (dnid, dnids,))
        groupxs = groupxs.compress(daids != cm.qaid)
        # </GET NAME GROUPXS>
        # sort annots in this name by the chip score
        # HACK USE cm.annot_score_list
        group_sortx = cm.csum_score_list.take(groupxs).argsort()[::-1]
        sorted_groupxs = groupxs.take(group_sortx)
        # get the info for this name
        name_fm_list  = ut.list_take(cm.fm_list, sorted_groupxs)
        REMOVE_EMPTY_MATCHES = len(sorted_groupxs) > 3
        REMOVE_EMPTY_MATCHES = True
        if REMOVE_EMPTY_MATCHES:
            isvalid_list = np.array([len(fm) > 0 for fm in name_fm_list])
            MAX_MATCHES = 3
            isvalid_list = ut.make_at_least_n_items_valid(isvalid_list, MAX_MATCHES)
            name_fm_list = ut.list_compress(name_fm_list, isvalid_list)
            sorted_groupxs = sorted_groupxs.compress(isvalid_list)

        name_H1_list   = (None if not homog or cm.H_list is None else
                          ut.list_take(cm.H_list, sorted_groupxs))
        name_fsv_list  = (None if cm.fsv_list is None else
                          ut.list_take(cm.fsv_list, sorted_groupxs))
        name_fs_list   = (None if name_fsv_list is None else
                          [fsv.prod(axis=1) for fsv in name_fsv_list])
        name_daid_list = ut.list_take(cm.daid_list, sorted_groupxs)
        # find features marked as invalid by name scoring
        featflag_list  = name_scoring.get_chipmatch_namescore_nonvoting_feature_flags(
            cm, qreq_=qreq_)
        name_featflag_list = ut.list_take(featflag_list, sorted_groupxs)
        # Get the scores for names and chips
        name_score = cm.name_score_list[nidx]
        name_rank = ut.listfind(cm.name_score_list.argsort()[::-1].tolist(), nidx)
        name_annot_scores = cm.csum_score_list.take(sorted_groupxs)

        _ = viz_matches.show_name_matches(
            qreq_.ibs, qaid, name_daid_list, name_fm_list, name_fs_list,
            name_H1_list, name_featflag_list, name_score=name_score, name_rank=name_rank,
            name_annot_scores=name_annot_scores, qreq_=qreq_, fnum=fnum,
            pnum=pnum, **kwargs)
        return _

    def show_single_annotmatch(cm, qreq_, daid=None, fnum=None, pnum=None,
                               homog=ut.get_argflag('--homog'), aid2=None, **kwargs):
        """
        TODO: rename daid to aid2

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18])
            >>> cm = cm_list[0]
            >>> cm.score_nsum(qreq_)
            >>> ut.quit_if_noshow()
            >>> daid = cm.get_groundtruth_daids()[0]
            >>> cm.show_single_annotmatch(qreq_, daid)
            >>> ut.show_if_requested()
        """
        from ibeis.viz import viz_matches
        if aid2 is not None:
            assert daid is None, 'use aid2 instead of daid kwarg'
            daid = aid2

        if daid is None:
            idx = cm.argsort()[0]
            daid = cm.daid_list[idx]
        else:
            idx = cm.daid2_idx[daid]
        fm   = cm.fm_list[idx]
        H1   = None if not homog or cm.H_list is None else cm.H_list[idx]
        fsv  = None if cm.fsv_list is None else cm.fsv_list[idx]
        fs   = None if fsv is None else fsv.prod(axis=1)
        showkw = dict(fm=fm, fs=fs, H1=H1, fnum=fnum, pnum=pnum, **kwargs)
        score = None if cm.score_list is None else cm.score_list[idx]
        viz_matches.show_matches2(qreq_.ibs, cm.qaid, daid, qreq_=qreq_,
                                  score=score, **showkw)

    def show_ranked_matches(cm, qreq_, clip_top=6, *args, **kwargs):
        r"""
        Plots the ranked-list of name/annot matches using matplotlib

        Args:
            qreq_ (QueryRequest): query request object with hyper-parameters
            clip_top (int): (default = 6)

        Kwargs:
            fnum, figtitle, plottype, ...more

        SeeAlso:
            ibeis.viz.viz_matches.show_matches2
            ibeis.viz.viz_matches.show_name_matches

        CommandLine:
            python -m ibeis --tf ChipMatch2.show_ranked_matches --show --qaid 1
            python -m ibeis --tf ChipMatch2.show_ranked_matches --qaid 86 --colorbar_=False --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> from ibeis.viz import viz_matches
            >>> defaultkw = dict(ut.recursive_parse_kwargs(viz_matches.show_name_matches))
            >>> kwargs = ut.argparse_dict(defaultkw, only_specified=True)
            >>> del kwargs['qaid']
            >>> kwargs['plottype'] = kwargs.get('plottype', 'namematch')
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[1])
            >>> cm = cm_list[0]
            >>> cm.score_nsum(qreq_)
            >>> clip_top = ut.get_argval('--clip-top', default=3)
            >>> print('kwargs = %s' % (ut.repr2(kwargs, nl=True),))
            >>> cm.show_ranked_matches(qreq_, clip_top, **kwargs)
            >>> ut.show_if_requested()
        """
        idx_list  = ut.listclip(cm.argsort(), clip_top)
        cm.show_index_matches(qreq_, idx_list, *args, **kwargs)

    def show_daids_matches(cm, qreq_, daids, *args, **kwargs):
        idx_list = ut.dict_take(cm.daid2_idx, daids)
        cm.show_index_matches(qreq_, idx_list, *args, **kwargs)

    def show_index_matches(cm, qreq_, idx_list, fnum=None, figtitle=None,
                           plottype='annotmatch', **kwargs):
        import plottool as pt
        if fnum is None:
            fnum = pt.next_fnum()
        nRows, nCols  = pt.get_square_row_cols(len(idx_list), fix=False)
        if ut.get_argflag('--vert'):
            # HACK
            nRows, nCols = nCols, nRows
        next_pnum     = pt.make_pnum_nextgen(nRows, nCols)
        for idx in idx_list:
            daid  = cm.daid_list[idx]
            pnum = next_pnum()
            if plottype == 'namematch':
                dnid = qreq_.ibs.get_annot_nids(daid)
                cm.show_single_namematch(qreq_, dnid, pnum=pnum, fnum=fnum, **kwargs)
            elif plottype == 'annotmatch':
                cm.show_single_annotmatch(qreq_, daid, fnum=fnum, pnum=pnum, **kwargs)
                # FIXME:
                score = vt.trytake(cm.score_list, idx)
                annot_score = vt.trytake(cm.annot_score_list, idx)
                score_str = ('score = %.3f' % (score,)
                             if score is not None else
                             'score = None')
                annot_score_str = ('annot_score = %.3f' % (annot_score,)
                                   if annot_score is not None else
                                   'annot_score = None')
                title = score_str + '\n' + annot_score_str
                pt.set_title(title)
            else:
                raise NotImplementedError('Unknown plottype=%r' % (plottype,))
        if figtitle is not None:
            pt.set_figtitle(figtitle)

    show_matches = show_single_annotmatch  # HACK

    def ishow_single_annotmatch(cm, qreq_, aid2=None, **kwargs):
        r"""
        Iteract with a match to an individual annotation (or maybe name?)

        Args:
            qreq_ (QueryRequest):  query request object with hyper-parameters
            aid2 (int):  annotation id(default = None)

        CommandLine:
            python -m ibeis.model.hots.chip_match --exec-ishow_single_annotmatch --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[1])
            >>> cm = cm_list[0]
            >>> cm.score_nsum(qreq_)
            >>> aid2 = None
            >>> result = cm.ishow_single_annotmatch(qreq_, aid2)
            >>> print(result)
            >>> ut.show_if_requested()
        """
        from ibeis.viz.interact import interact_matches  # NOQA
        #if aid == 'top':
        #    aid = qres.get_top_aids(ibs)
        kwshow = {
            'mode': 1,
        }
        if aid2 is None:
            aid2 = cm.get_top_aids(ntop=1)[0]
        kwshow.update(**kwargs)
        try:
            match_interaction = interact_matches.MatchInteraction(qreq_.ibs,
                                                                  cm, aid2,
                                                                  qreq_=qreq_,
                                                                  **kwshow)
            return match_interaction
        except Exception as ex:
            ut.printex(ex, 'failed in qres.show_matches', keys=['aid', 'qreq_'])
            raise
        if not kwargs.get('noupdate', False):
            import plottool as pt
            pt.update()

    ishow_match = ishow_single_annotmatch
    ishow_matches = ishow_single_annotmatch

    def ishow_analysis(cm, qreq_, **kwargs):
        """
        CommandLine:
            python -m ibeis.model.hots.chip_match --exec-ChipMatch2.ishow_analysis --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> qaid = 18
            >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list=[qaid])
            >>> cm = cm_list[0]
            >>> cm.score_nsum(qreq_)
            >>> ut.quit_if_noshow()
            >>> cm.ishow_analysis(qreq_)
            >>> ut.show_if_requested()
        """
        from ibeis.viz.interact import interact_qres
        kwshow = {
            'show_query': False,
            'show_timedelta': True,
        }
        kwshow.update(kwargs)
        return interact_qres.ishow_analysis(qreq_.ibs, cm, qreq_=qreq_, **kwshow)

    def show_analysis(cm, qreq_, **kwargs):
        from ibeis.viz import viz_qres
        kwshow = {
            'show_query': False,
            'show_timedelta': True,
        }
        kwshow.update(kwargs)
        return viz_qres.show_qres_analysis(qreq_.ibs, cm, qreq_=qreq_, **kwshow)

    def imwrite_single_annotmatch(cm, qreq_, aid, **kwargs):
        """
        CommandLine:
            python -m ibeis.model.hots.chip_match --exec-ChipMatch2.imwrite_single_annotmatch --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> import ibeis
            >>> kwargs = {}
            >>> kwargs['dpi'] = ut.get_argval('--dpi', int, None)
            >>> kwargs['figsize'] = ut.get_argval('--figsize', list, None)
            >>> kwargs['fpath'] = ut.get_argval('--fpath', str, None)
            >>> kwargs['draw_fmatches'] = not ut.get_argflag('--no-fmatches')
            >>> kwargs['vert'] = ut.get_argflag('--vert')
            >>> kwargs['draw_border'] = ut.get_argflag('--draw_border')
            >>> kwargs['saveax'] = ut.get_argflag('--saveax')
            >>> kwargs['in_image'] = ut.get_argflag('--in-image')
            >>> kwargs['draw_lbl'] = ut.get_argflag('--no-draw-lbl')
            >>> print('kwargs = %s' % (ut.dict_str(kwargs),))
            >>> cm, qreq_ = ibeis.testdata_cm()
            >>> aid = cm.get_top_aids()[0]
            >>> img_fpath = cm.imwrite_single_annotmatch(qreq_, aid, **kwargs)
            >>> ut.quit_if_noshow()
            >>> # show the image dumped to disk
            >>> ut.startfile(img_fpath, quote=True)
            >>> ut.show_if_requested()
        """
        import plottool as pt
        import matplotlib as mpl
        # Pop save kwargs from kwargs
        save_keys = ['dpi', 'figsize', 'saveax', 'fpath', 'fpath_strict', 'verbose']
        save_vals = ut.dict_take_pop(kwargs, save_keys, None)
        savekw = dict(zip(save_keys, save_vals))
        fpath = savekw.pop('fpath')
        if fpath is None and 'fpath_strict' not in savekw:
            savekw['usetitle'] = True
        was_interactive = mpl.is_interactive()
        if was_interactive:
            mpl.interactive(False)
        # Make new figure
        fnum = pt.ensure_fnum(kwargs.pop('fnum', None))
        #fig = pt.figure(fnum=fnum, doclf=True, docla=True)
        fig = pt.plt.figure(fnum)
        fig.clf()
        # Draw Matches
        cm.show_single_annotmatch(qreq_, aid, colorbar_=False, fnum=fnum, **kwargs)
        #if not kwargs.get('notitle', False):
        #    pt.set_figtitle(cm.make_smaller_title())
        # Save Figure
        # Setting fig=fig might make the dpi and figsize code not work
        img_fpath = pt.save_figure(fpath=fpath, fig=fig, **savekw)
        pt.plt.close(fig)  # Ensure that this figure will not pop up
        if was_interactive:
            mpl.interactive(was_interactive)
        #if False:
        #    ut.startfile(img_fpath)
        return img_fpath

    def qt_inspect_gui(cm, ibs, ranks_lt=6, qreq_=None, name_scoring=False):
        r"""
        Args:
            ibs (IBEISController):  ibeis controller object
            ranks_lt (int): (default = 6)
            qreq_ (QueryRequest):  query request object with hyper-parameters(default = None)
            name_scoring (bool): (default = False)

        Returns:
            QueryResult: qres_wgt -  object of feature correspondences and scores

        CommandLine:
            python -m ibeis.model.hots.chip_match --exec-qt_inspect_gui --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[1])
            >>> cm = cm_list[0]
            >>> cm.score_nsum(qreq_)
            >>> ranks_lt = 6
            >>> name_scoring = False
            >>> qres_wgt = cm.qt_inspect_gui(ibs, ranks_lt, qreq_, name_scoring)
            >>> ut.quit_if_noshow()
            >>> import guitool
            >>> guitool.qtapp_loop(qwin=qres_wgt)
        """
        print('[qres] qt_inspect_gui')
        from ibeis.gui import inspect_gui
        import guitool
        guitool.ensure_qapp()
        cm_list = [cm]
        print('[inspect_matches] make_qres_widget')
        qres_wgt = inspect_gui.QueryResultsWidget(ibs, cm_list,
                                                  ranks_lt=ranks_lt,
                                                  name_scoring=name_scoring,
                                                  qreq_=qreq_)
        print('[inspect_matches] show')
        qres_wgt.show()
        print('[inspect_matches] raise')
        qres_wgt.raise_()
        return qres_wgt

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.chip_match
        python -m ibeis.model.hots.chip_match --allexamples
        python -m ibeis.model.hots.chip_match --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
