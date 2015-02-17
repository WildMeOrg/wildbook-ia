from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
import six
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[chip_match]', DEBUG=False)


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
    def from_chipmatch_old(cls, chipmatch_old, qaid=None, fsv_col_lbls=None):
        (aid2_fm_, aid2_fsv_, aid2_fk_, aid2_score_, aid2_H_) = chipmatch_old
        assert len(aid2_fsv_) == len(aid2_fm_), 'bad old chipmatch'
        assert len(aid2_fk_) == len(aid2_fm_), 'bad old chipmatch'
        assert aid2_score_ is None or len(aid2_score_) == 0 or len(aid2_score_) == len(aid2_fm_), 'bad old chipmatch'
        assert aid2_H_ is None or len(aid2_H_) == len(aid2_fm_), 'bad old chipmatch'
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

    def to_oldstyle_chipmatch(cm):
        from ibeis.model.hots import hstypes
        aid2_fm    = dict(zip(cm.daid_list, cm.fm_list))
        aid2_fsv   = dict(zip(cm.daid_list, cm.fsv_list))
        aid2_fk    = dict(zip(cm.daid_list, cm.fk_list))
        aid2_score = {} if cm.score_list is None else dict(zip(cm.daid_list, cm.score_list))
        aid2_H     = None if cm.H_list is None else dict(zip(cm.daid_list, cm.H_list))
        chipmatch  = hstypes.ChipMatch(aid2_fm, aid2_fsv, aid2_fk, aid2_score, aid2_H)
        return chipmatch

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
    behaves as as the ChipMatch named tuple until we
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
        chipmatch_old = (aid2_fm_, aid2_fsv_, aid2_fk_, aid2_score_, aid2_H_)
        fsv_col_lbls = qres.filtkey_list
        cm = ChipMatch2.from_chipmatch_old(chipmatch_old, qaid, fsv_col_lbls)
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
        score_list = [fsv.prod(axis=1).sum() for fsv in fsv_list]
        cm = ChipMatch2(qaid, daid_list, fm_list, fsv_list, None, score_list, H_list, fsv_col_lbls)
        cm.fs_list = fs_list
        return cm

    # Standard Contstructor

    def __init__(cm, qaid=None, daid_list=None, fm_list=None, fsv_list=None, fk_list=None,
                 score_list=None, H_list=None, fsv_col_lbls=None):
        assert daid_list is not None, 'must give daids'
        assert fm_list is None or len(fm_list) == len(daid_list), 'incompatable data'
        assert fsv_list is None or len(fsv_list) == len(daid_list), 'incompatable data'
        assert fk_list is None or len(fk_list) == len(daid_list), 'incompatable data'
        assert H_list is None or len(H_list) == len(daid_list), 'incompatable data'
        assert score_list is None or len(score_list) == len(daid_list), 'incompatable data'
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
        cm._update_daid_index()
        # Metadata for backwards compatability
        # Current hack, but this should remain.
        # fsv should be persistant
        # this is the combined scores
        # there should also be coefficients
        # and there should also be different types of scores
        cm.fs_list = None

    def _update_daid_index(cm):
        cm.daid2_idx = (None if cm.daid_list is None else
                        {daid: idx for idx, daid in enumerate(cm.daid_list)})

    def sortself(cm):
        """ reorders the internal data using cm.score_list """
        def trytake(list_, sortx):
            return None if list_ is None else ut.list_take(list_, sortx)
        sortx         = ut.list_argsort(cm.score_list, reverse=True)
        cm.daid_list  = trytake(cm.daid_list, sortx)
        cm.fm_list    = trytake(cm.fm_list, sortx)
        cm.fsv_list   = trytake(cm.fsv_list, sortx)
        cm.fs_list    = trytake(cm.fs_list, sortx)
        cm.fk_list    = trytake(cm.fk_list, sortx)
        cm.score_list = trytake(cm.score_list, sortx)
        cm.H_list     = trytake(cm.H_list, sortx)
        cm._update_daid_index()

    def get_num_feat_score_cols(cm):
        return len(cm.fsv_col_lbls)

    def shortlist_subset(cm, top_aids):
        """ returns a new chipmatch with only the requested daids """
        qaid         = cm.qaid
        idx_list     = ut.dict_take(cm.daid2_idx, top_aids)
        daid_list    = ut.list_take(cm.daid_list, idx_list)
        fm_list      = ut.list_take(cm.fm_list, idx_list)
        fsv_list     = ut.list_take(cm.fsv_list, idx_list)
        fk_list      = ut.list_take(cm.fk_list, idx_list)
        score_list   = ut.list_take(cm.score_list, idx_list)
        H_list       = ut.list_take(cm.H_list, idx_list)
        fsv_col_lbls = cm.fsv_col_lbls
        cm_subset = ChipMatch2(qaid, daid_list, fm_list, fsv_list, fk_list,
                               score_list, H_list, fsv_col_lbls)
        return cm_subset

    def get_property_string():
        pass

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
        if not sort:
            sortx = list(range(len(cm.score_list)))
        else:
            sortx = ut.list_argsort(cm.score_list, reverse=True)
        column_list = [
            ut.list_take(cm.daid_list,  sortx),
            ut.list_take(cm.score_list, sortx),
            ut.lmap(str, ut.depth_profile(ut.list_take(cm.fm_list,  sortx))),
            ut.lmap(str, ut.depth_profile(ut.list_take(cm.fsv_list, sortx))),
        ]
        column_lbls = ['daid', 'score', 'fm_depth', 'fsv_depth']
        if ibs is not None:
            column_list.insert(1, ibs.get_annot_nids(ut.list_take(cm.daid_list,  sortx)))
            column_lbls.insert(1, 'dnid')
            qnid = ibs.get_annot_nids(cm.qaid)
        else:
            qnid = '?'
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

    def get_rawinfostr(cm):
        def varinfo(var):
            import utool as ut
            if var is None:
                return None
            return ut.depth_profile(var)
        str_list = []
        append = str_list.append
        append('cm.qaid = %r' % (cm.qaid,))
        append('cm.fsv_col_lbls = %r' % (varinfo(cm.fsv_col_lbls),))
        append('len(cm.daid2_idx) = %r' % (len(cm.daid2_idx),))
        append('depth(cm.daid_list) = %r' % (varinfo(cm.daid_list),))
        append('depth(cm.score_list) = %r' % (varinfo(cm.score_list),))
        append('depth(cm.fs_list) = %r' % (varinfo(cm.fs_list),))
        append('depth(cm.fm_list) = %r' % (varinfo(cm.fm_list),))
        append('depth(cm.fsv_list) = %r' % (varinfo(cm.fsv_list),))
        append('depth(cm.H_list) = %r' % (varinfo(cm.H_list),))
        infostr = '\n'.join(ut.align_lines(str_list, '='))
        return infostr

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
