"""
hstypes
Todo:
* SIFT: Root_SIFT -> L2 normalized -> Centering.
# http://hal.archives-ouvertes.fr/docs/00/84/07/21/PDF/RR-8325.pdf
The devil is in the deatails
http://www.robots.ox.ac.uk/~vilem/bmvc2011.pdf
This says dont clip, do rootsift instead
# http://hal.archives-ouvertes.fr/docs/00/68/81/69/PDF/hal_v1.pdf
* Quantization of residual vectors
* Burstiness normalization for N-SMK
* Implemented A-SMK
* Incorporate Spatial Verification
* Implement correct cfgstrs based on algorithm input
for cached computations.
* Color by word
* Profile on hyrule
* Train vocab on paris
* Remove self matches.
* New SIFT parameters for pyhesaff (root, powerlaw, meanwhatever, output_dtype)

Issues:
* 10GB are in use when performing query on Oxford 5K
* errors when there is a word without any database vectors.
currently a weight of zero is hacked in
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
import six
from collections import namedtuple, defaultdict
ut.noinject('[hstypes]')


#INTEGER_TYPE = np.int32
INDEX_TYPE = np.int32

#INTEGER_TYPE = np.int64
INTEGER_TYPE = np.int32

#FLOAT_TYPE = np.float64
FLOAT_TYPE = np.float32

VEC_DIM = 128

VEC_TYPE = np.uint8
VEC_IINFO = np.iinfo(VEC_TYPE)
VEC_MAX = VEC_IINFO.max
VEC_MIN = VEC_IINFO.min
# Psuedo max values come from SIFT descriptors implementation
VEC_PSEUDO_MAX = 512
# unit sphere points can only be twice the maximum descriptor magnitude away
# from each other. The pseudo max is 512, so 1024 is the upper bound
# FURTHERMORE SIFT Descriptors are constrained to be in the upper right quadrent
# which means any two vectors with one full component and zeros elsewhere are
# maximally distant. VEC_PSEUDO_MAX_DISTANCE = np.sqrt(2) * VEC_PSEUDO_MAX
if VEC_MIN == 0:
    # Can be on only on one quadrent of unit sphere
    VEC_PSEUDO_MAX_DISTANCE = VEC_PSEUDO_MAX * np.sqrt(2.0)
    VEC_PSEUDO_MAX_DISTANCE_SQRD = 2.0 * (512.0 ** 2.0)
elif VEC_MIN < 0:
    # Can be on whole unit sphere
    VEC_PSEUDO_MAX_DISTANCE = VEC_PSEUDO_MAX * 2
else:
    raise AssertionError('impossible state')

PSEUDO_UINT8_MAX_SQRD = float(VEC_PSEUDO_MAX) ** 2


RVEC_TYPE = np.int8
#RVEC_TYPE = np.float16
if RVEC_TYPE == np.int8:
    # Unfortunatley int8 cannot represent NaN, maybe used a masked array
    RVEC_INFO = np.iinfo(RVEC_TYPE)
    RVEC_MAX = 128
    RVEC_MIN = -128
    # Psuedo max values is used for a quantization trick where you pack more data
    # into a smaller space than would normally be allowed. We are able to do this
    # because values will hardly ever be close to the true max.
    RVEC_PSEUDO_MAX = RVEC_MAX * 2
    RVEC_PSEUDO_MAX_SQRD = float(RVEC_PSEUDO_MAX ** 2)
elif RVEC_TYPE == np.float16:
    RVEC_INFO = np.finfo(RVEC_TYPE)
    RVEC_MAX = 1.0
    RVEC_MIN = -1.0
    RVEC_PSEUDO_MAX = RVEC_MAX
    RVEC_PSEUDO_MAX_SQRD = float(RVEC_PSEUDO_MAX ** 2)
else:
    raise AssertionError('impossible RVEC_TYPE')


# Feature Match datatype
FM_DTYPE  = INTEGER_TYPE
# Feature Score datatype
FS_DTYPE  = FLOAT_TYPE
# Feature Rank datatype
FK_DTYPE  = np.int16


class FiltKeys(object):
    DISTINCTIVENESS = 'distinctiveness'
    FG = 'fg'
    RATIO = 'ratio'
    DIST = 'dist'
    LNBNN = 'lnbnn'
    DUPVOTE = 'dupvote'
    HOMOGERR = 'homogerr'

# Denote which scores should be  used as weights
# the others are used as scores
WEIGHT_FILTERS = [FiltKeys.FG, FiltKeys.DISTINCTIVENESS, FiltKeys.HOMOGERR]


ChipMatch = namedtuple('ChipMatch', ('aid2_fm', 'aid2_fsv', 'aid2_fk', 'aid2_score', 'aid2_H'))


class DefaultDictProxy(object):
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
        raise NotImplementedError()

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

#import six


@six.add_metaclass(ut.ReloadingMetaclass)
class ChipMatch2(object):
    """
    Rename to QueryToDatabaseMatch?

    behaves as as the ChipMatch named tuple until we
    completely replace the old structure
    """
    #
    _fields = ('aid2_fm', 'aid2_fsv', 'aid2_fk', 'aid2_score', 'aid2_H')

    # Alternative Cosntructors

    @classmethod
    def from_prior(cls, prior_cm, fm_list, fs_list, H_list=None, fsv_col_lbls=None):
        from vtool import matching
        qaid = prior_cm.qaid
        daid_list = prior_cm.daid_list
        fsv_list = matching.ensure_fsv_list(fs_list)
        score_list = [fsv.prod(axis=1).sum() for fsv in fsv_list]
        cm = ChipMatch2(qaid, daid_list, fm_list, fsv_list, None, score_list, H_list, fsv_col_lbls)
        return cm

    # Standard Contstructor

    def __init__(cm, qaid=None, daid_list=None, fm_list=None, fsv_list=None, fk_list=None,
                 score_list=None, H_list=None, fsv_col_lbls=None):
        cm.qaid         = qaid
        cm.daid_list    = daid_list
        cm.fm_list      = fm_list
        cm.fsv_list     = fsv_list
        if fk_list is None:
            cm.fk_list = [np.zeros(fm.shape[0]) for fm in cm.fm_list]
        else:
            cm.fk_list      = fk_list

        cm.score_list   = score_list
        cm.H_list       = H_list
        cm.fsv_col_lbls = fsv_col_lbls
        # Metadata for backwards compatability
        cm.daid2_idx    = (None if cm.daid_list is None else
                           {daid: idx for idx, daid in enumerate(cm.daid_list)})

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

    def tokwargs(cm):
        """
        Can be unpacked and passed as kwargs
        **cm.tokwargs()
        """
        import collections
        class KwargsWrapper(collections.Mapping):
            """
            Allows an arbitrary object attributes to be passed as a **kwargs
            argument
            """
            def __init__(self, obj):
                self.obj = obj

            def __getitem__(self, key):
                return self.obj.__dict__[key]

            def __iter__(self):
                return iter(self.obj.__dict__)

            def __len__(self):
                return len(self.obj.__dict__)
        return KwargsWrapper(cm)

    def get_cvs_str(cm, numtop=6, ibs=None):
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
        sortx = ut.list_argsort(cm.score_list)[::-1]
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

    # SIMULATE OLD CHIPMATCHES UNTIL TRANSFER IS COMPLETE
    # TRY NOT TO USE THESE AS THEY WILL BE MUCH SLOWER THAN
    # NORMAL.

    @classmethod
    def from_chipmatch_old(cls, chipmatch_old, qaid=None, fsv_col_lbls=None):
        (aid2_fm_, aid2_fsv_, aid2_fk_, aid2_score_, aid2_H_) = chipmatch_old
        aid_list = list(six.iterkeys(aid2_fm_))
        daid_list    = aid_list
        fm_list      = ut.dict_take(aid2_fm_, aid_list)
        fsv_list     = ut.dict_take(aid2_fsv_, aid_list)
        fk_list      = ut.dict_take(aid2_fk_, aid_list)
        score_list   = (None if aid2_score_ is None or len(aid2_score_) == 0
                           else ut.dict_take(aid2_score_, aid_list))
        H_list       = (None if aid2_H_ is None else
                        ut.dict_take(aid2_H_, aid_list))
        fsv_col_lbls = fsv_col_lbls
        cm = ChipMatch2(qaid, daid_list, fm_list, fsv_list, fk_list, score_list, H_list, fsv_col_lbls)
        return cm

    def to_oldstyle_chipmatch(cm):
        aid2_fm    = dict(zip(cm.daid_list, cm.fm_list))
        aid2_fsv   = dict(zip(cm.daid_list, cm.fsv_list))
        aid2_fk    = dict(zip(cm.daid_list, cm.fk_list))
        aid2_score = dict(zip(cm.daid_list, cm.score_list))
        aid2_H     = dict(zip(cm.daid_list, cm.H_list))
        chipmatch = ChipMatch(aid2_fm, aid2_fsv, aid2_fk, aid2_score, aid2_H)
        return chipmatch

    def __iter__(cm):
        for field in cm._fields:
            yield getattr(cm, field)

    def __getitem__(cm, index):
        if isinstance(index, six.string_types):
            return cm.__dict__[index]
        else:
            return getattr(cm, cm._fields[index])

    def _asdict(cm):
        return ut.odict(
            [(field, None if  getattr(cm, field) is None else getattr(cm, field).copy())
                for field in cm._fields])

    @property
    def aid2_fm(cm):
        return DefaultDictProxy(cm.daid2_idx, cm.daid_list, cm.fm_list)

    @property
    def aid2_fsv(cm):
        return DefaultDictProxy(cm.daid2_idx, cm.daid_list, cm.fsv_list)

    @property
    def aid2_fk(cm):
        return DefaultDictProxy(cm.daid2_idx, cm.daid_list, cm.fk_list)

    @property
    def aid2_H(cm):
        return None if cm.H_list is None else DefaultDictProxy(cm.daid2_idx, cm.daid_list, cm.H_list)

    @property
    def aid2_score(cm):
        return {} if cm.score_list is None else DefaultDictProxy(cm.daid2_idx, cm.daid_list, cm.score_list)


# Replace old chipmatch with ducktype
# Keep this turned off for now until we actually start using it
#ChipMatch = ChipMatch2


def fix_chipmatch(chipmatch_):
    r"""
    removes matches without enough support
    enforces type and shape of arrays

    CommandLine:
        python -m ibeis.model.hots.hstypes --test-fix_chipmatch

    Note:
        difference between windows and linux:
        windows in on python32 and linux is python64
        therefore we get dtype=np.int32 printing on linux but not on windows

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.hstypes import *  # NOQA
        >>> # build test data
        >>> chipmatch_ = (
        ...    {1: [(0, 0), (1, 1)], 2: [(0, 0), (1, 1), (2, 2)]},
        ...    {1: [    .5,     .7], 2: [    .2,     .4,     .6]},
        ...    {1: [     1,      1], 2: [     1,      1,      1]},
        ...    None,
        ...    None,
        ...    )
        >>> # execute function
        >>> chipmatch = fix_chipmatch(chipmatch_)
        >>> # verify results
        >>> result_full = ut.dict_str(chipmatch._asdict(), precision=2)
        >>> print(result_full)
        >>> result = ut.hashstr(result_full)
        >>> print(result)
        ukih5wutvz@2fxj+

    """
    (aid2_fm_, aid2_fsv_, aid2_fk_, aid2_score_, aid2_H_) = chipmatch_
    minMatches = 2  # TODO: paramaterize
    # FIXME: This is slow
    fm_dtype  = FM_DTYPE
    fsv_dtype = FS_DTYPE
    fk_dtype  = FK_DTYPE
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
    chipmatch = ChipMatch(aid2_fm, aid2_fsv, aid2_fk, aid2_score, aid2_H)
    return chipmatch


def new_chipmatch(with_homog=False, with_score=True):
    """ returns new chipmatch for a single qaid """
    aid2_fm = defaultdict(list)
    aid2_fsv = defaultdict(list)
    aid2_fk = defaultdict(list)
    aid2_score = dict() if with_score else None
    aid2_H = dict() if with_homog else None
    chipmatch = ChipMatch(aid2_fm, aid2_fsv, aid2_fk, aid2_score, aid2_H)
    return chipmatch


def chipmatch_subset(chipmatch, aids):
    aid2_fm    = ut.dict_subset(chipmatch.aid2_fm, aids)
    aid2_fk    = ut.dict_subset(chipmatch.aid2_fk, aids)
    aid2_fsv   = ut.dict_subset(chipmatch.aid2_fsv, aids)
    aid2_H     = ut.dict_subset(chipmatch.aid2_H, aids) if chipmatch.aid2_H is not None else None
    aid2_score = ut.dict_subset(chipmatch.aid2_score, aids) if len(chipmatch.aid2_score) != 0 else {}
    chipmatch_ = ChipMatch(aid2_fm, aid2_fsv, aid2_fk, aid2_score, aid2_H)
    return chipmatch_


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.hstypes
        python -m ibeis.model.hots.hstypes --allexamples
        python -m ibeis.model.hots.hstypes --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
