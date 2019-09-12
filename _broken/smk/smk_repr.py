# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
#import six
import utool as ut
#import weakref
import numpy as np
import six
from six.moves import zip, map  # NOQA
from vtool import nearest_neighbors as nntool
from ibeis.algo.hots import hstypes
from ibeis.algo.hots.smk import smk_scoring
from ibeis.algo.hots.smk import smk_index
from collections import namedtuple
(print, print_, printDBG, rrr, profile) = ut.inject2(__name__, '[smk_repr]')


DEBUG_SMK = ut.DEBUG2 or ut.get_argflag('--debug-smk')


INVERTED_INDEX_INJECT_KEY = ('InvertedIndex', __name__)


@six.add_metaclass(ut.ReloadingMetaclass)
class InvertedIndex(object):
    r"""
    Stores inverted index state information
    (mapping from words to database aids and fxs_list)

    Attributes:
        idx2_dvec    (ndarray[S x DIM]): stacked index -> descriptor vector (currently sift)
        idx2_daid    (ndarray[S x 1]): stacked index -> annot id
        idx2_dfx     (ndarray[S x 1]): stacked index -> feature index (wrt daid)
        idx2_fweight (ndarray[S x 1]): stacked index -> feature weight

        idx2_wxs     (list): stacked index -> word indexes (jagged)

        words        (ndarray[C x DIM]): visual word centroids
        wordflann    (FLANN): FLANN search structure

        wx2_idxs     (dict of lists of ndarrays): word index -> stacked indexes
        wx2_fxs      (dict of lists of ndarrays): word index -> aggregate feature indexes
        wx2_aids     (dict of ndarrays[N_c x 1]): word index -> aggregate aids

        wx2_drvecs   (dict of ndarrays[N_c x DIM]): word index -> residual vectors
        wx2_dflags   (dict of ndarrays[N_c x 1]): word index -> residual flags
        wx2_idf      (dict of ndarrays[N_c x 1]): word index -> idf (wx normalizer)
        wx2_maws     (dict of ndarrays[N_c x 1]): word index -> multi-assign weights

        daids        (ndarray): indexed annotation ids
        daid2_sccw   (dict of floats): daid -> sccw (daid self-consistency weight)
        daid2_label  (dict of tuples): daid -> label (name, view)

    """

    def __init__(invindex, words, wordflann, idx2_vec, idx2_aid, idx2_fx,
                 daids, daid2_label):
        invindex.words        = words
        invindex.wordflann    = wordflann
        invindex.idx2_dvec    = idx2_vec
        invindex.idx2_daid    = idx2_aid
        invindex.idx2_dfx     = idx2_fx
        invindex.daids        = daids
        invindex.daid2_label  = daid2_label
        invindex.wx2_idxs     = None
        invindex.wx2_aids     = None
        invindex.wx2_fxs      = None
        invindex.wx2_maws     = None
        invindex.wx2_drvecs   = None
        invindex.wx2_dflags   = None
        invindex.wx2_idf      = None
        invindex.daid2_sccw   = None
        invindex.idx2_fweight = None
        invindex.idx2_wxs     = None   # stacked index -> word indexes

        # Inject debug function
        from ibeis.algo.hots.smk import smk_debug
        ut.make_class_method_decorator(INVERTED_INDEX_INJECT_KEY)(smk_debug.invindex_dbgstr)
        ut.inject_instance(invindex, classkey=INVERTED_INDEX_INJECT_KEY)


@ut.make_class_method_decorator(INVERTED_INDEX_INJECT_KEY)
def report_memory(obj, objname='obj'):
    """
    obj = invindex
    objname = 'invindex'

    """
    print('Object Memory Usage for %s' % objname)
    maxlen = max(map(len, six.iterkeys(obj.__dict__)))
    for key, val in six.iteritems(obj.__dict__):
        fmtstr = 'memusage({0}.{1}){2} = '
        lbl = fmtstr.format(objname, key, ' ' * (maxlen - len(key)))
        sizestr = ut.get_object_size_str(val, lbl=lbl, unit='MB')
        print(sizestr)


report_memsize = ut.make_class_method_decorator(INVERTED_INDEX_INJECT_KEY)(ut.report_memsize)


QueryIndex = namedtuple(
    'QueryIndex', (
        'wx2_qrvecs',
        'wx2_qflags',
        'wx2_maws',
        'wx2_qaids',
        'wx2_qfxs',
        'query_sccw',
    ))


class LazyGetter(object):
    """
    DEPRICATE
    """

    def __init__(self, getter_func):
        self.getter_func = getter_func

    def __getitem__(self, index):
        return self.getter_func(index)

    def __call__(self, index):
        return self.getter_func(index)


class DataFrameProxy(object):
    """
    DEPRICATE

    pandas is actually really slow. This class emulates it so
    I don't have to change my function calls, but without all the slowness.
    """

    def __init__(annots_df, ibs):
        annots_df.ibs = ibs

    def __getitem__(annots_df, key):
        if key == 'kpts':
            return LazyGetter(annots_df.ibs.get_annot_kpts)
        elif key == 'vecs':
            return LazyGetter(annots_df.ibs.get_annot_vecs)
        elif key == 'labels':
            return LazyGetter(annots_df.ibs.get_annot_class_labels)


@profile
def make_annot_df(ibs):
    """
    Creates a pandas like DataFrame interface to an IBEISController

    DEPRICATE

    Args:
        ibs ():

    Returns:
        annots_df

    Example:
        >>> from ibeis.algo.hots.smk.smk_repr import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> ibs = smk_debug.testdata_ibeis()
        >>> annots_df = make_annot_df(ibs)
        >>> print(ut.hashstr(repr(annots_df.values)))
        j12n+x93m4c!4un3

    #>>> from ibeis.algo.hots.smk import smk_debug
    #>>> smk_debug.rrr()
    #>>> smk_debug.check_dtype(annots_df)

    Auto:
        from ibeis.algo.hots.smk import smk_repr
        import utool as ut
        argdoc = ut.make_default_docstr(smk_repr.make_annot_df)
        print(argdoc)
    """
    annots_df = DataFrameProxy(ibs)
    return annots_df


@profile
def new_qindex(annots_df, qaid, invindex, qparams):
    r"""
    Gets query read for computations

    Args:
        annots_df (DataFrameProxy): pandas-like data interface
        qaid (int): query annotation id
        invindex (InvertedIndex): inverted index object
        qparams (QueryParams): query parameters object

    Returns:
        qindex: named tuple containing query information

    CommandLine:
        python -m ibeis.algo.hots.smk.smk_repr --test-new_qindex

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.smk.smk_repr import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> ibs, annots_df, qaid, invindex, qparams = smk_debug.testdata_query_repr(db='PZ_Mothers', nWords=128000)
        >>> qindex = new_qindex(annots_df, qaid, invindex, qparams)
        >>> assert smk_debug.check_wx2_rvecs(qindex.wx2_qrvecs), 'has nan'
        >>> smk_debug.invindex_dbgstr(invindex)

    Ignore::
        idx2_vec = qfx2_vec
        idx2_aid = qfx2_aid
        idx2_fx  = qfx2_qfx
        wx2_idxs = _wx2_qfxs
        wx2_maws = _wx2_maws
        from ibeis.algo.hots.smk import smk_repr
        import utool as ut
        ut.rrrr()
        print(ut.make_default_docstr(smk_repr.new_qindex))
    """
    # TODO: Precompute and lookup residuals and assignments
    if not ut.QUIET:
        print('[smk_repr] Query Repr qaid=%r' % (qaid,))
    #
    nAssign               = qparams.nAssign
    massign_alpha         = qparams.massign_alpha
    massign_sigma         = qparams.massign_sigma
    massign_equal_weights = qparams.massign_equal_weights
    #
    aggregate             = qparams.aggregate
    smk_alpha             = qparams.smk_alpha
    smk_thresh            = qparams.smk_thresh
    #
    wx2_idf   = invindex.wx2_idf
    words     = invindex.words
    wordflann = invindex.wordflann
    #qfx2_vec  = annots_df['vecs'][qaid]
    # TODO: remove all mention of annot_df and ensure that qparams is passed corectly to config2_
    qfx2_vec  = annots_df.ibs.get_annot_vecs(qaid, config2_=qparams)
    #-------------------
    # Assign query to (multiple) words
    #-------------------
    _wx2_qfxs, _wx2_maws, qfx2_wxs = smk_index.assign_to_words_(
        wordflann, words, qfx2_vec, nAssign, massign_alpha,
        massign_sigma, massign_equal_weights)
    # Hack to make implementing asmk easier, very redundant
    qfx2_aid = np.array([qaid] * len(qfx2_wxs), dtype=hstypes.INTEGER_TYPE)
    qfx2_qfx = np.arange(len(qfx2_vec))
    #-------------------
    # Compute query residuals
    #-------------------
    wx2_qrvecs, wx2_qaids, wx2_qfxs, wx2_maws, wx2_qflags = smk_index.compute_residuals_(
        words, _wx2_qfxs, _wx2_maws, qfx2_vec, qfx2_aid, qfx2_qfx, aggregate)
    # each value in wx2_ dicts is a list with len equal to the number of rvecs
    if ut.VERBOSE:
        print('[smk_repr] Query SCCW smk_alpha=%r, smk_thresh=%r' % (smk_alpha, smk_thresh))
    #-------------------
    # Compute query sccw
    #-------------------
    wx_sublist  = np.array(wx2_qrvecs.keys(), dtype=hstypes.INDEX_TYPE)
    idf_list    = [wx2_idf[wx]    for wx in wx_sublist]
    rvecs_list  = [wx2_qrvecs[wx] for wx in wx_sublist]
    maws_list   = [wx2_maws[wx]   for wx in wx_sublist]
    flags_list  = [wx2_qflags[wx] for wx in wx_sublist]
    query_sccw = smk_scoring.sccw_summation(rvecs_list, flags_list, idf_list, maws_list, smk_alpha, smk_thresh)
    try:
        assert query_sccw > 0, 'query_sccw=%r is not positive!' % (query_sccw,)
    except Exception as ex:
        ut.printex(ex)
        raise
    #-------------------
    # Build query representationm class/tuple
    #-------------------
    if DEBUG_SMK:
        from ibeis.algo.hots.smk import smk_debug
        qfx2_vec = annots_df['vecs'][qaid]
        assert smk_debug.check_wx2_rvecs2(
            invindex, wx2_qrvecs, wx2_qfxs, qfx2_vec), 'bad qindex'

    qindex = QueryIndex(wx2_qrvecs, wx2_qflags, wx2_maws, wx2_qaids, wx2_qfxs, query_sccw)
    return qindex


#@profile
def index_data_annots(annots_df, daids, words, qparams, with_internals=True,
                      memtrack=None, delete_rawvecs=False):
    """
    Builds the initial inverted index from a dataframe, daids, and words.
    Optionally builds the internals of the inverted structure

    Args:
        annots_df ():
        daids ():
        words ():
        qparams ():
        with_internals ():
        memtrack (): memory debugging object

    Returns:
        invindex

    Example:
        >>> from ibeis.algo.hots.smk.smk_repr import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, qreq_, words = smk_debug.testdata_words()
        >>> qparams = qreq_.qparams
        >>> with_internals = False
        >>> invindex = index_data_annots(annots_df, daids, words, qparams, with_internals)

    Ignore:
        #>>> print(ut.hashstr(repr(list(invindex.__dict__.values()))))
        #v8+i5i8+55j0swio

    Auto:
        from ibeis.algo.hots.smk import smk_repr
        import utool as ut
        ut.rrrr()
        print(ut.make_default_docstr(smk_repr.index_data_annots))
    """
    if not ut.QUIET:
        print('[smk_repr] index_data_annots')
    flann_params = {}
    # Compute fast lookup index for the words
    wordflann = nntool.flann_cache(words, flann_params=flann_params, appname='smk')
    _vecs_list = annots_df['vecs'][daids]
    _label_list = annots_df['labels'][daids]
    idx2_dvec, idx2_daid, idx2_dfx = nntool.invertible_stack(_vecs_list, daids)

    # TODO:
    # Need to individually cache residual vectors.
    # rvecs_list = annots_df['rvecs'][daids]
    #
    # Residual vectors depend on
    # * nearest word (word assignment)
    # * original vectors
    # * multiassignment

    daid2_label = dict(zip(daids, _label_list))

    invindex = InvertedIndex(words, wordflann, idx2_dvec, idx2_daid, idx2_dfx,
                             daids, daid2_label)
    # Decrement reference count so memory can be cleared in the next function
    del words, idx2_dvec, idx2_daid, idx2_dfx, daids, daid2_label
    del _vecs_list, _label_list
    if with_internals:
        compute_data_internals_(invindex, qparams, memtrack=memtrack,
                                delete_rawvecs=delete_rawvecs)  # 99%
    return invindex


@profile
def compute_data_internals_(invindex, qparams, memtrack=None,
                            delete_rawvecs=True):
    """
    Builds each of the inverted index internals.

        invindex (InvertedIndex): object for fast vocab lookup
        qparams (QueryParams): hyper-parameters
        memtrack (None):
        delete_rawvecs (bool):

    Returns:
        None

    Example:
        >>> from ibeis.algo.hots.smk.smk_repr import *  # NOQA
        >>> from ibeis.algo.hots.smk import smk_debug
        >>> ibs, annots_df, daids, qaids, invindex, qreq_ = smk_debug.testdata_raw_internals0()
        >>> compute_data_internals_(invindex, qreq_.qparams)

    Ignore:
        idx2_vec = idx2_dvec
        wx2_maws = _wx2_maws  # NOQA
    """
    # Get information
    #if memtrack is None:
    #    memtrack = ut.MemoryTracker('[DATA INTERNALS ENTRY]')

    #memtrack.report('[DATA INTERNALS1]')

    #
    aggregate             = qparams.aggregate
    smk_alpha             = qparams.smk_alpha
    smk_thresh            = qparams.smk_thresh
    #
    massign_alpha         = qparams.massign_alpha
    massign_sigma         = qparams.massign_sigma
    massign_equal_weights = qparams.massign_equal_weights
    #
    vocab_weighting       = qparams.vocab_weighting
    #
    nAssign = 1  # single assignment for database side

    idx2_vec  = invindex.idx2_dvec
    idx2_dfx  = invindex.idx2_dfx
    idx2_daid = invindex.idx2_daid
    daids     = invindex.daids
    wordflann = invindex.wordflann
    words     = invindex.words
    daid2_label = invindex.daid2_label
    wx_series = np.arange(len(words))
    #memtrack.track_obj(idx2_vec, 'idx2_vec')
    if not ut.QUIET:
        print('[smk_repr] compute_data_internals_')
    if ut.VERBOSE:
        print('[smk_repr] * len(daids) = %r' % (len(daids),))
        print('[smk_repr] * len(words) = %r' % (len(words),))
        print('[smk_repr] * len(idx2_vec) = %r' % (len(idx2_vec),))
        print('[smk_repr] * aggregate = %r' % (aggregate,))
        print('[smk_repr] * smk_alpha = %r' % (smk_alpha,))
        print('[smk_repr] * smk_thresh = %r' % (smk_thresh,))

    # Try to use the cache
    #cfgstr = ut.hashstr_arr(words, 'words') + qparams.feat_cfgstr
    #cachekw = dict(
        #cfgstr=cfgstr,
        #appname='smk_test'
    #)
    #invindex_cache = ut.Cacher('inverted_index', **cachekw)
    #try:
    #    raise IOError('cache is off')
    #    #cachetup = invindex_cache.load()
    #    #(idx2_wxs, wx2_idxs, wx2_idf, wx2_drvecs, wx2_aids, wx2_fxs, wx2_maws, daid2_sccw) = cachetup
    #    invindex.idx2_dvec = None
    #except IOError as ex:
    # Database word assignments (perform single assignment on database side)
    wx2_idxs, _wx2_maws, idx2_wxs = smk_index.assign_to_words_(
        wordflann, words, idx2_vec, nAssign, massign_alpha, massign_sigma,
        massign_equal_weights)
    if ut.DEBUG2:
        assert len(idx2_wxs) == len(idx2_vec)
        assert len(wx2_idxs.keys()) == len(_wx2_maws.keys())
        assert len(wx2_idxs.keys()) <= len(words)
        try:
            assert len(wx2_idxs.keys()) == len(words)
        except AssertionError as ex:
            ut.printex(ex, iswarning=True)
    # Database word inverse-document-frequency (idf weights)
    wx2_idf = smk_index.compute_word_idf_(
        wx_series, wx2_idxs, idx2_daid, daids, daid2_label, vocab_weighting,
        verbose=True)
    if ut.DEBUG2:
        assert len(wx2_idf) == len(wx2_idf.keys())
    # Compute (normalized) residual vectors and inverse mappings
    wx2_drvecs, wx2_aids, wx2_fxs, wx2_dmaws, wx2_dflags = smk_index.compute_residuals_(
        words, wx2_idxs, _wx2_maws, idx2_vec, idx2_daid, idx2_dfx,
        aggregate, verbose=True)
    if not ut.QUIET:
        print('[smk_repr] unloading idx2_vec')
    if delete_rawvecs:
        # Try to save some memory
        del _wx2_maws
        invindex.idx2_dvec = None
        del idx2_vec
    # Compute annotation normalization factor
    daid2_sccw = smk_index.compute_data_sccw_(
        idx2_daid, wx2_drvecs, wx2_dflags, wx2_aids, wx2_idf, wx2_dmaws, smk_alpha,
        smk_thresh, verbose=True)
    # Cache save
    #cachetup = (idx2_wxs, wx2_idxs, wx2_idf, wx2_drvecs, wx2_aids, wx2_fxs, wx2_dmaws, daid2_sccw)
    #invindex_cache.save(cachetup)

    # Store information
    invindex.idx2_wxs    = idx2_wxs   # stacked index -> word indexes (might not be needed)
    invindex.wx2_idxs    = wx2_idxs
    invindex.wx2_idf     = wx2_idf
    invindex.wx2_drvecs  = wx2_drvecs
    invindex.wx2_dflags  = wx2_dflags  # flag nan rvecs
    invindex.wx2_aids    = wx2_aids    # needed for asmk
    invindex.wx2_fxs     = wx2_fxs     # needed for asmk
    invindex.wx2_dmaws   = wx2_dmaws   # needed for awx2_mawssmk
    invindex.daid2_sccw  = daid2_sccw
    #memtrack.report('[DATA INTERNALS3]')

    if ut.DEBUG2:
        from ibeis.algo.hots.smk import smk_debug
        smk_debug.check_invindex_wx2(invindex)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.algo.hots.smk.smk_repr
        python -m ibeis.algo.hots.smk.smk_repr --allexamples
        python -m ibeis.algo.hots.smk.smk_repr --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
