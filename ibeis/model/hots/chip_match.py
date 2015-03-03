from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
import vtool as vt
from operator import xor
import six
from ibeis.model.hots import hstypes
from collections import namedtuple, defaultdict
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[chip_match]', DEBUG=False)


ChipMatchOldTup = namedtuple('ChipMatchOldTup', ('aid2_fm', 'aid2_fsv', 'aid2_fk', 'aid2_score', 'aid2_H'))


def fix_cmtup_old(cmtup_old_):
    r"""
    removes matches without enough support
    enforces type and shape of arrays

    CommandLine:
        python -m ibeis.model.hots.chip_match --test-fix_cmtup_old

    Note:
        difference between windows and linux:
        windows in on python32 and linux is python64
        therefore we get dtype=np.int32 printing on linux but not on windows

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.chip_match import *  # NOQA
        >>> # build test data
        >>> cmtup_old_ = (
        ...    {1: [(0, 0), (1, 1)], 2: [(0, 0), (1, 1), (2, 2)]},
        ...    {1: [    .5,     .7], 2: [    .2,     .4,     .6]},
        ...    {1: [     1,      1], 2: [     1,      1,      1]},
        ...    None,
        ...    None,
        ...    )
        >>> # execute function
        >>> cmtup_old = fix_cmtup_old(cmtup_old_)
        >>> # verify results
        >>> result = ut.dict_str(cmtup_old._asdict(), precision=2)
        >>> print(result)
        {
            'aid2_fm': {
                2: np.array([[0, 0],
                             [1, 1],
                             [2, 2]], dtype=np.int32),
            },
            'aid2_fsv': {
                2: np.array([ 0.2,  0.4,  0.6], dtype=np.float64),
            },
            'aid2_fk': {
                2: np.array([1, 1, 1], dtype=np.int16),
            },
            'aid2_score': {},
            'aid2_H': None,
        }


    """
    (aid2_fm_, aid2_fsv_, aid2_fk_, aid2_score_, aid2_H_) = cmtup_old_
    minMatches = 2  # TODO: paramaterize
    # FIXME: This is slow
    fm_dtype  = hstypes.FM_DTYPE
    fsv_dtype = hstypes.FS_DTYPE
    fk_dtype  = hstypes.FK_DTYPE
    # Mark valid chipmatches
    aid_list_     = list(six.iterkeys(aid2_fm_))
    fm_list_      = list(six.itervalues(aid2_fm_))
    isvalid_list_ = [len(fm) > minMatches for fm in fm_list_]
    # Filter invalid chipmatches
    aid_list   = ut.filter_items(aid_list_, isvalid_list_)
    fm_list    = ut.filter_items(fm_list_, isvalid_list_)
    fsv_list   = ut.dict_take(aid2_fsv_, aid_list)
    fk_list    = ut.dict_take(aid2_fk_, aid_list)
    score_list = None if aid2_score_ is None or len(aid2_score_) == 0 else ut.dict_take(aid2_score_, aid_list)
    H_list     = None if aid2_H_ is None else ut.dict_take(aid2_H_, aid_list)
    # Convert to numpy an dictionary format
    aid2_fm    = {aid: np.array(fm, fm_dtype) for aid, fm in zip(aid_list, fm_list)}
    aid2_fsv   = {aid: np.array(fsv, fsv_dtype) for aid, fsv in zip(aid_list, fsv_list)}
    aid2_fk    = {aid: np.array(fk, fk_dtype) for aid, fk in zip(aid_list, fk_list)}
    aid2_score = {} if score_list is None else {aid: score for aid, score in zip(aid_list, score_list)}
    aid2_H     = None if H_list is None else {aid: H for aid, H in zip(aid_list, H_list)}
    # Ensure shape
    #for aid, fm in six.iteritems(aid2_fm_):
    #    fm.shape = (fm.size // 2, 2)
    cmtup_old = ChipMatchOldTup(aid2_fm, aid2_fsv, aid2_fk, aid2_score, aid2_H)
    return cmtup_old


def new_cmtup_old(with_homog=False, with_score=True):
    """ returns new cmtup_old for a single qaid """
    aid2_fm = defaultdict(list)
    aid2_fsv = defaultdict(list)
    aid2_fk = defaultdict(list)
    aid2_score = dict() if with_score else None
    aid2_H = dict() if with_homog else None
    cmtup_old = ChipMatchOldTup(aid2_fm, aid2_fsv, aid2_fk, aid2_score, aid2_H)
    return cmtup_old


class _DefaultDictProxy(object):
    """
    simulates a dict when using parallel lists the point of this class is that
    when there are many instances of this class, then key2_idx can be shared between
    them. Ideally this class wont be used and will disappear when the parallel
    lists are being used properly.
    """
    def __init__(self, key2_idx, key_list, val_list):
        self.key_list = key_list
        self.val_list = val_list
        self.key2_idx = key2_idx

    def __repr__(self):
        return repr(dict(self.items()))

    def __str__(self):
        return str(dict(self.items()))

    def __len__(self):
        return len(self.key_list)

    #def __del__(self, key):
    #    raise NotImplementedError()

    def copy(self):
        return dict(self.items())

    def __eq__(self, key):
        raise NotImplementedError()

    def pop(self, key):
        raise NotImplementedError()

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
        #raise NotImplementedError()

    def __contains__(self, key):
        return key in self.key2_idx

    def __getitem__(self, key):
        try:
            return self.val_list[self.key2_idx[key]]
        except (KeyError, IndexError):
            # behave like a default dict here
            self[key] = []
            return self[key]
        #return ut.list_take(self.val_list, ut.dict_take(self.key2_idx, key))

    def __setitem__(self, key, val):
        try:
            idx = self.key2_idx[key]
        except KeyError:
            idx = len(self.key_list)
            self.key_list.append(key)
            self.key2_idx[key] = idx
        try:
            self.val_list[idx] = val
        except IndexError:
            if idx == len(self.val_list):
                self.val_list.append(val)
            else:
                raise
            #else:
            #    offset = idx - len(self.val_list)
            #    self.val_list.extend(([None] * offset) + [val])

    def iteritems(self):
        for key, val in zip(self.key_list, self.val_list):
            yield key, val

    def iterkeys(self):
        return iter(self.key_list)

    def itervalues(self):
        return iter(self.val_list)

    def values(self):
        return list(self.itervalues())

    def keys(self):
        return list(self.iterkeys())

    def items(self):
        return list(self.iteritems())


class _OldStyleChipMatchSimulator(object):
    # SIMULATE OLD CHIPMATCHES UNTIL TRANSFER IS COMPLETE
    # TRY NOT TO USE THESE AS THEY WILL BE MUCH SLOWER THAN
    # NORMAL.
    _oldfields = ('aid2_fm', 'aid2_fsv', 'aid2_fk', 'aid2_score', 'aid2_H')

    @classmethod
    def from_cmtup_old(cls, cmtup_old, qaid=None, fsv_col_lbls=None):
        (aid2_fm_, aid2_fsv_, aid2_fk_, aid2_score_, aid2_H_) = cmtup_old
        assert len(aid2_fsv_) == len(aid2_fm_), 'bad old cmtup_old'
        assert len(aid2_fk_) == len(aid2_fm_), 'bad old cmtup_old'
        assert aid2_score_ is None or len(aid2_score_) == 0 or len(aid2_score_) == len(aid2_fm_), 'bad old cmtup_old'
        assert aid2_H_ is None or len(aid2_H_) == len(aid2_fm_), 'bad old cmtup_old'
        aid_list = list(six.iterkeys(aid2_fm_))
        daid_list    = aid_list
        fm_list      = ut.dict_take(aid2_fm_, aid_list)
        fsv_list     = ut.dict_take(aid2_fsv_, aid_list)
        fk_list      = ut.dict_take(aid2_fk_, aid_list)
        score_list   = (None if aid2_score_ is None or (len(aid2_score_) == 0 and len(daid_list) > 0)
                           else ut.dict_take(aid2_score_, aid_list))
        H_list       = (None if aid2_H_ is None else
                        ut.dict_take(aid2_H_, aid_list))
        fsv_col_lbls = fsv_col_lbls
        cm = ChipMatch2(qaid, daid_list, fm_list, fsv_list, fk_list, score_list, H_list, fsv_col_lbls)
        return cm

    def to_cmtup_old(cm):
        aid2_fm    = dict(zip(cm.daid_list, cm.fm_list))
        aid2_fsv   = dict(zip(cm.daid_list, cm.fsv_list))
        aid2_fk    = dict(zip(cm.daid_list, cm.fk_list))
        aid2_score = {} if cm.score_list is None else dict(zip(cm.daid_list, cm.score_list))
        aid2_H     = None if cm.H_list is None else dict(zip(cm.daid_list, cm.H_list))
        cmtup_old  = ChipMatchOldTup(aid2_fm, aid2_fsv, aid2_fk, aid2_score, aid2_H)
        return cmtup_old

    def __iter__(cm):
        for field in cm._oldfields:
            yield getattr(cm, field)

    def __getitem__(cm, index):
        if isinstance(index, six.string_types):
            return cm.__dict__[index]
        else:
            return getattr(cm, cm._oldfields[index])

    def _asdict(cm):
        return ut.odict(
            [(field, None if  getattr(cm, field) is None else getattr(cm, field).copy())
                for field in cm._oldfields])

    @property
    def aid2_fm(cm):
        return _DefaultDictProxy(cm.daid2_idx, cm.daid_list, cm.fm_list)

    @property
    def aid2_fsv(cm):
        return _DefaultDictProxy(cm.daid2_idx, cm.daid_list, cm.fsv_list)

    @property
    def aid2_fk(cm):
        return _DefaultDictProxy(cm.daid2_idx, cm.daid_list, cm.fk_list)

    @property
    def aid2_H(cm):
        return None if cm.H_list is None else _DefaultDictProxy(cm.daid2_idx, cm.daid_list, cm.H_list)

    @property
    def aid2_score(cm):
        return {} if cm.score_list is None else _DefaultDictProxy(cm.daid2_idx, cm.daid_list, cm.score_list)

    # qres compatibility

    @property
    def filtkey_list(cm):
        """ for compatibility with qres """
        return cm.fsv_col_lbls

    @property
    def aid2_fs(cm):
        return _DefaultDictProxy(cm.daid2_idx, cm.daid_list, cm.fsv_list)

#import six


def test_from_qres(qres):
    """
    CommandLine:
        python -m ibeis.model.hots.chip_match --test-test_from_qres

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.chip_match import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> qres = ibs._query_chips4([1], [2, 3, 4, 5], cfgdict=dict(), verbose=True)[1]
        >>> cm = ChipMatch2.from_qres(qres)
        >>> cm.print_csv(ibs=ibs)
    """
    pass


@six.add_metaclass(ut.ReloadingMetaclass)
class ChipMatch2(_OldStyleChipMatchSimulator):
    """
    behaves as as the ChipMatchOldTup named tuple until we
    completely replace the old structure
    """

    # Alternative  Cosntructors

    @classmethod
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
        cm = ChipMatch2.from_cmtup_old(cmtup_old, qaid, fsv_col_lbls)
        fs_list = [fsv.T[cm.fsv_col_lbls.index('lnbnn')] for fsv in cm.fsv_list]
        cm.fs_list = fs_list
        return cm

    @classmethod
    def from_unscored(cls, prior_cm, fm_list, fs_list, H_list=None, fsv_col_lbls=None):
        from vtool import matching
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
        cm = ChipMatch2(qaid, daid_list, fm_list, fsv_list, None, score_list, H_list, fsv_col_lbls)
        cm.fs_list = fs_list
        return cm

    @classmethod
    def from_vsmany_match_tup(cls, valid_match_tup, qaid=None, fsv_col_lbls=None):
        # Vsmany - create new cmtup_old
        (valid_daid, valid_qfx, valid_dfx, valid_scorevec, valid_rank) = valid_match_tup
        valid_fm = np.vstack((valid_qfx, valid_dfx)).T
        daid_list, groupxs = vt.group_indices(valid_daid)
        fm_list  = vt.apply_grouping(valid_fm, groupxs)
        fsv_list = vt.apply_grouping(valid_scorevec, groupxs)
        fk_list  = vt.apply_grouping(valid_rank, groupxs)
        cm = ChipMatch2(qaid, daid_list, fm_list, fsv_list, fk_list, fsv_col_lbls=fsv_col_lbls)
        return cm

    @classmethod
    def from_vsone_match_tup(cls, valid_match_tup_list, daid_list=None, qaid=None, fsv_col_lbls=None):
        assert all(list(map(ut.list_allsame, ut.get_list_column(valid_match_tup_list, 0)))),\
            'internal daids should not have different daids for vsone'
        qfx_list = ut.get_list_column(valid_match_tup_list, 1)
        dfx_list = ut.get_list_column(valid_match_tup_list, 2)
        fm_list  = [np.vstack(dfx_qfx).T for dfx_qfx in zip(dfx_list, qfx_list)]
        fsv_list = ut.get_list_column(valid_match_tup_list, 3)
        fk_list  = ut.get_list_column(valid_match_tup_list, 4)
        cm = ChipMatch2(qaid, daid_list, fm_list, fsv_list, fk_list, fsv_col_lbls=fsv_col_lbls)
        return cm

    # Standard Contstructor

    def __init__(cm, qaid=None, daid_list=None, fm_list=None, fsv_list=None, fk_list=None,
                 score_list=None, H_list=None, fsv_col_lbls=None, dnid_list=None, qnid=None):
        """
        qaid and daid_list are not optional. fm_list and fsv_list are strongly
        encouraged and will probalby break things if they are not there.
        """
        assert daid_list is not None, 'must give daids'
        assert fm_list is None or len(fm_list) == len(daid_list), 'incompatable data'
        assert fsv_list is None or len(fsv_list) == len(daid_list), 'incompatable data'
        assert fk_list is None or len(fk_list) == len(daid_list), 'incompatable data'
        assert H_list is None or len(H_list) == len(daid_list), 'incompatable data'
        assert score_list is None or len(score_list) == len(daid_list), 'incompatable data'
        assert dnid_list is None or len(dnid_list) == len(daid_list), 'incompatable data'
        cm.qaid         = qaid
        cm.daid_list    = daid_list
        cm.fm_list      = fm_list
        cm.fsv_list     = fsv_list
        cm.fk_list      = (fk_list if fk_list is not None else
                           [np.zeros(fm.shape[0]) for fm in cm.fm_list])
        cm.score_list   = score_list
        cm.H_list       = H_list
        cm.fsv_col_lbls = fsv_col_lbls
        cm.daid2_idx    = None
        cm.fs_list = None
        # TODO
        cm.prob_list = None
        cm.dnid_list = dnid_list
        cm.qnid = qnid
        cm.annot_score_list = None
        cm.unique_nids = None
        cm.name_score_list = None
        cm._update_daid_index()

    def get_fs(cm, idx=None, colx=None, daid=None, col=None):
        assert xor(idx is None, daid is None)
        assert xor(colx is None or col is None)
        if daid is not None:
            idx = cm.daid2_idx[daid]
        if col is not None:
            colx = cm.fsv_col_lbls.index(col)
        fs = cm.fsv_list[idx][colx]
        return fs

    def get_fs_list(cm, colx=None, col=None):
        assert xor(colx is None or col is None)
        if col is not None:
            colx = cm.fsv_col_lbls.index(col)
        fs_list = [fsv.T[colx].T for fsv in cm.fsv_list]
        return fs_list

    def evaluate_dnids(cm, ibs):
        cm.qnid = ibs.get_annot_name_rowids(cm.qaid)
        cm.dnid_list = np.array(ibs.get_annot_name_rowids(cm.daid_list))

    def assign_name_scores(cm, unique_nids, name_score_list):
        cm.unique_nids     = unique_nids
        cm.name_score_list = name_score_list
        cm.nid2_idx = ut.make_index_lookup(cm.unique_nids)

    def get_nid_scores(cm, nid_list):
        idx_list = ut.dict_take(cm.nid2_idx, nid_list)
        return vt.list_take_(cm.name_score_list, idx_list)

    def _update_daid_index(cm):
        cm.daid2_idx = (None if cm.daid_list is None else
                        {daid: idx for idx, daid in enumerate(cm.daid_list)})

    def get_num_matches_list(cm):
        num_matches_list = list(map(len, cm.fm_list))
        return num_matches_list

    def argsort(cm):
        if cm.score_list is None:
            num_matches_list = cm.get_num_matches_list()
            sortx = ut.list_argsort(num_matches_list, reverse=True)
        else:
            sortx = ut.list_argsort(cm.score_list, reverse=True)
        return sortx

    def sortself(cm):
        """ reorders the internal data using cm.score_list """
        def trytake(list_, sortx):
            if list_ is None:
                return None
            return vt.list_take_(list_, sortx)
        sortx               = cm.argsort()
        cm.daid_list        = trytake(cm.daid_list, sortx)
        cm.dnid_list        = trytake(cm.dnid_list, sortx)
        cm.fm_list          = trytake(cm.fm_list, sortx)
        cm.fsv_list         = trytake(cm.fsv_list, sortx)
        cm.fs_list          = trytake(cm.fs_list, sortx)
        cm.fk_list          = trytake(cm.fk_list, sortx)
        cm.score_list       = trytake(cm.score_list, sortx)
        cm.annot_score_list = trytake(cm.annot_score_list, sortx)
        cm.H_list           = trytake(cm.H_list, sortx)
        cm._update_daid_index()

    def get_num_feat_score_cols(cm):
        return len(cm.fsv_col_lbls)

    def shortlist_subset(cm, top_aids):
        """ returns a new cmtup_old with only the requested daids """
        def trytake(list_, index_list):
            return None if list_ is None else vt.list_take_(list_, index_list)
        qaid         = cm.qaid
        qnid         = cm.qnid
        idx_list     = ut.dict_take(cm.daid2_idx, top_aids)
        daid_list    = vt.list_take_(cm.daid_list, idx_list)
        fm_list      = vt.list_take_(cm.fm_list, idx_list)
        fsv_list     = vt.list_take_(cm.fsv_list, idx_list)
        fk_list      = trytake(cm.fk_list, idx_list)
        #score_list   = trytake(cm.score_list, idx_list)
        score_list   = None  # don't transfer scores
        H_list       = trytake(cm.H_list, idx_list)
        dnid_list    = trytake(cm.dnid_list, idx_list)
        fsv_col_lbls = cm.fsv_col_lbls
        cm_subset = ChipMatch2(qaid, daid_list, fm_list, fsv_list, fk_list,
                               score_list, H_list, fsv_col_lbls, dnid_list, qnid)
        return cm_subset

    def get_rawinfostr(cm):
        def varinfo(varname, forcerepr=False):
            import utool as ut
            varval = getattr(cm, varname)
            if not forcerepr and ut.isiterable(varval):
                varinfo_list = [
                    '    * varinfo(cm.%s):' % (varname,),
                    '        depth = %r' % (ut.depth_profile(varval),),
                    '        types = %s' % (ut.list_type_profile(varval),),
                ]
                #varinfo = '\n'.join(ut.align_lines(varinfo_list, '='))
                varinfo = '\n'.join(ut.align_lines(varinfo_list, '='))
            else:
                varinfo = '    * cm.%s = %r' % (varname, varval)
            return varinfo
        str_list = []
        append = str_list.append
        append('ChipMatch2:')
        append('    * cm.qaid = %r' % (cm.qaid,))
        append('    * cm.qnid = %r' % (cm.qnid,))
        append('    * len(cm.daid2_idx) = %r' % (len(cm.daid2_idx),))
        append(varinfo('fsv_col_lbls', forcerepr=True))
        append(varinfo('daid_list'))
        append(varinfo('dnid_list'))
        append(varinfo('fs_list'))
        append(varinfo('fm_list'))
        append(varinfo('fk_list'))
        append(varinfo('fsv_list'))
        append(varinfo('H_list'))
        append(varinfo('annot_score_list'))
        append(varinfo('name_score_list'))
        append(varinfo('unique_nids'))
        append(varinfo('score_list'))
        #infostr = '\n'.join(ut.align_lines(str_list, '='))
        infostr = '\n'.join(str_list)
        return infostr

    def get_cvs_str(cm,  numtop=6, ibs=None, sort=True):
        """
        Notes:
        Very weird that it got a score

        qaid 6 vs 41 has
            [72, 79, 0, 17, 6, 60, 15, 36, 63]
            [72, 79, 0, 17, 6, 60, 15, 36, 63]
            [72, 79, 0, 17, 6, 60, 15, 36, 63]
            [0.060, 0.053, 0.0497, 0.040, 0.016, 0, 0, 0, 0]
            [7, 40, 41, 86, 103, 88, 8, 101, 35]

        makes very little sense
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
        column_lbls_ = ['daid', 'dnid', 'score', 'num_matches', 'fm_depth', 'fsv_depth']
        column_list_ = [
            vt.list_take_(cm.daid_list,  sortx),
            None if dnid_list is None else vt.list_take_(dnid_list, sortx),
            None if cm.score_list is None else vt.list_take_(cm.score_list, sortx),
            vt.list_take_(cm.get_num_matches_list(), sortx),
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

    def print_rawinfostr(cm):
        print(cm.get_rawinfostr())

    def print_csv(cm, *args, **kwargs):
        print(cm.get_cvs_str(*args, **kwargs))
        #daid_sorted  = ut.list_take(cm.daid_list, sortx)
        #fm_sorted    = ut.list_take(cm.fm_list, sortx)
        #fsv_sorted   = ut.list_take(cm.fsv_list, sortx)
        #fk_sorted    = ut.list_take(cm.fk_list, sortx)
        #score_sorted = ut.list_take(cm.score_list, sortx)
        #H_sorted     = ut.list_take(cm.H_list, sortx)

        #print(list(map(len, fm_sorted)))
        #print(ut.numpy_str(np.array(list(map(len, fsv_sorted))), precision=3))
        #print(ut.numpy_str(np.array(score_sorted), precision=3))
        ##print(list(map(len, fk_sorted)))
        ##print(score_sorted)
        #print(daid_sorted)

    def tokwargs(cm):
        """
        Can be unpacked and passed as kwargs
        **cm.tokwargs()
        """
        return ut.KwargsWrapper(cm)

    def get_property_string():
        pass

    def assert_self(cm, qreq_=None, strict=False, verbose=ut.NOT_QUIET):
        assert cm.qaid is not None, 'must have qaid'
        assert cm.daid_list is not None, 'must give daids'
        assert cm.fm_list is None or len(cm.fm_list) == len(cm.daid_list), 'incompatable data'
        assert cm.fsv_list is None or len(cm.fsv_list) == len(cm.daid_list), 'incompatable data'
        assert cm.fk_list is None or len(cm.fk_list) == len(cm.daid_list), 'incompatable data'
        assert cm.H_list is None or len(cm.H_list) == len(cm.daid_list), 'incompatable data'
        assert cm.score_list is None or len(cm.score_list) == len(cm.daid_list), 'incompatable data'
        assert cm.dnid_list is None or len(cm.dnid_list) == len(cm.daid_list), 'incompatable data'
        if verbose:
            print('[cm] lengths are ok')
        try:
            assert ut.list_all_eq_to([fsv.shape[1] for fsv in cm.fsv_list], len(cm.fsv_col_lbls))
        except Exception as ex:
            cm.print_rawinfostr()
            raise
        assert ut.list_all_eq_to([fm.shape[1] for fm in cm.fm_list], 2), 'bad fm'
        if verbose:
            print('[cm] shapes are ok')
        if strict or qreq_ is not None:
            external_qaids = qreq_.get_external_qaids().tolist()
            external_daids = qreq_.get_external_daids().tolist()
            if qreq_.qparams.pipeline_root == 'vsone':
                assert len(external_qaids) == 1, 'only one external qaid for vsone'
                if strict or qreq_.indexer is not None:
                    nExternalQVecs = qreq_.ibs.get_annot_vecs(external_qaids[0], qreq_=qreq_).shape[0]
                    assert qreq_.indexer.idx2_vec.shape[0] == nExternalQVecs, 'did not index query descriptors properly'
                if verbose:
                    print('[cm] vsone daids are ok are ok')

            nFeats1 = qreq_.ibs.get_annot_num_feats(cm.qaid, qreq_=qreq_)
            nFeats2_list = np.array(qreq_.ibs.get_annot_num_feats(cm.daid_list, qreq_=qreq_))
            try:
                assert ut.list_issubset(cm.daid_list, external_daids), 'cmtup_old must be subset of daids'
            except AssertionError as ex:
                ut.printex(ex, keys=['daid_list', 'external_daids'])
                raise
            try:
                fm_list = cm.fm_list
                fx2s_list = [fm_.T[1] for fm_ in fm_list]
                fx1s_list = [fm_.T[0] for fm_ in fm_list]
                max_fx1_list = np.array([-1 if len(fx1s) == 0 else fx1s.max() for fx1s in fx1s_list])
                max_fx2_list = np.array([-1 if len(fx2s) == 0 else fx2s.max() for fx2s in fx2s_list])
                ut.assert_lessthan(max_fx2_list, nFeats2_list, 'max feat index must be less than num feats')
                ut.assert_lessthan(max_fx1_list, nFeats1, 'max feat index must be less than num feats')
            except AssertionError as ex:
                ut.printex(ex, keys=['qaid', 'daid_list', 'nFeats1',
                                     'nFeats2_list', 'max_fx1_list',
                                     'max_fx2_list', ])
                raise
            if verbose:
                print('[cm] nFeats are ok in fm')

    def testshow(cm, qreq_, daid=None, **kwargs):
        if ut.show_was_requested():
            import plottool as pt
            from ibeis.viz import viz_matches
            if daid is None:
                idx = cm.argsort()[0]
                daid = cm.daid_list[idx]
            else:
                idx = cm.daid2_idx[daid]
            fm   = cm.fm_list[idx]
            fsv  = None if cm.fsv_list is None else cm.fsv_list[idx]
            fs   = None if fsv is None else fsv.prod(axis=1)
            viz_matches.show_matches2(qreq_.ibs, cm.qaid, daid, fm, fs, **kwargs)
            pt.show_if_requested()


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
