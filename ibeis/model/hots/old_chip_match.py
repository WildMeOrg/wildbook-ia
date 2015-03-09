from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
import six
from ibeis.model.hots import hstypes
from collections import namedtuple, defaultdict
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[old_chip_match]', DEBUG=False)

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
        cm = cls(qaid, daid_list, fm_list, fsv_list, fk_list, score_list, H_list, fsv_col_lbls)
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

    def tokwargs(cm):
        """
        Can be unpacked and passed as kwargs
        **cm.tokwargs()
        """
        return ut.KwargsWrapper(cm)
