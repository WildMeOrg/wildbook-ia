# -*- coding: utf-8 -*-
"""
TODO: semantic_uuids should be replaced with PCC-like hashes pertaining to
annotation clusters if any form of name scoring is used.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import exists, join
from wbia.algo.hots import chip_match
import utool as ut
import numpy as np

(print, rrr, profile) = ut.inject2(__name__, '[mc5]')


class EstimatorRequest(ut.NiceRepr):
    def __init__(qreq_):
        qreq_.use_single_cache = True
        qreq_.use_bulk_cache = True
        qreq_.min_bulk_size = 64
        qreq_.chunksize = 256
        qreq_.prog_hook = None

    def __len__(qreq_):
        return len(qreq_.qaids)

    def execute(qreq_, qaids=None, prog_hook=None, use_cache=True):
        # assert qaids is None
        if qaids is not None:
            qreq_ = qreq_.shallowcopy(qaids)
        if use_cache:
            cm_list = execute_bulk(qreq_)
        else:
            cm_list = qreq_.execute_pipeline()
        # cm_list = qreq_.execute_pipeline()
        return cm_list

    def shallowcopy(qreq_, qaids=None):
        """
        Creates a copy of qreq with the same qparams object and a subset of
        the qx and dx objects.  used to generate chunks of vsone and vsmany
        queries

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.smk.match_chips5 import *  # NOQA
            >>> import wbia
            >>> wbia, smk, qreq_ = testdata_smk()
            >>> qreq2_ = qreq_.shallowcopy(qaids=1)
            >>> assert qreq_.daids is qreq2_.daids, 'should be the same'
            >>> assert len(qreq_.qaids) != len(qreq2_.qaids), 'should be diff'
            >>> #assert qreq_.metadata is not qreq2_.metadata
        """
        # qreq2_ = qreq_.__class__()
        cls = qreq_.__class__
        qreq2_ = cls.__new__(cls)
        qreq2_.__dict__.update(qreq_.__dict__)
        qaids = ut.ensure_iterable(qaids)
        assert ut.issubset(qaids, qreq_.qaids), 'not a subset'
        qreq2_.qaids = qaids
        return qreq2_

    def get_pipe_hashid(qreq_):
        return ut.hashstr27(str(qreq_.stack_config))

    def get_pipe_cfgstr(qreq_):
        pipe_cfgstr = qreq_.stack_config.get_cfgstr()
        return pipe_cfgstr

    def get_data_hashid(qreq_):
        data_hashid = qreq_.ibs.get_annot_hashid_semantic_uuid(qreq_.daids, prefix='D')
        return data_hashid

    def get_query_hashid(qreq_):
        # TODO: SYSTEM : semantic should only be used if name scoring is on
        query_hashid = qreq_.ibs.get_annot_hashid_semantic_uuid(qreq_.qaids, prefix='Q')
        return query_hashid

    def get_cfgstr(
        qreq_, with_input=False, with_data=True, with_pipe=True, hash_pipe=False
    ):
        cfgstr_list = []
        if with_input:
            cfgstr_list.append(qreq_.get_query_hashid())
        if with_data:
            cfgstr_list.append(qreq_.get_data_hashid())
        if with_pipe:
            pipe_cfgstr = qreq_.get_pipe_cfgstr()
            if hash_pipe:
                pipe_cfgstr = ut.hashstr27(pipe_cfgstr)
            cfgstr_list.append(pipe_cfgstr)
        cfgstr = ''.join(cfgstr_list)
        return cfgstr

    def __nice__(qreq_):
        return ' '.join(qreq_.get_nice_parts())

    def get_chipmatch_fpaths(qreq_, qaid_list):
        r"""
        Efficient function to get a list of chipmatch paths
        """
        cfgstr = qreq_.get_cfgstr(with_input=False, with_data=True, with_pipe=True)
        qauuid_list = qreq_.ibs.get_annot_semantic_uuids(qaid_list)
        fname_list = [
            chip_match.get_chipmatch_fname(qaid, qreq_, qauuid=qauuid, cfgstr=cfgstr)
            for qaid, qauuid in zip(qaid_list, qauuid_list)
        ]
        dpath = ut.ensuredir((qreq_.cachedir, 'mc5_cms'))
        fpath_list = [join(dpath, fname) for fname in fname_list]
        return fpath_list

    def get_nice_parts(qreq_):
        parts = []
        parts.append(qreq_.ibs.get_dbname())
        parts.append('nQ=%d' % len(qreq_.qaids))
        parts.append('nD=%d' % len(qreq_.daids))
        parts.append(qreq_.get_pipe_hashid())
        return parts

    # Hacked in functions

    def ensure_nids(qreq_):
        # Hacked over from hotspotter, seriously hacky
        ibs = qreq_.ibs
        qreq_.unique_aids = np.union1d(qreq_.qaids, qreq_.daids)
        qreq_.unique_nids = ibs.get_annot_nids(qreq_.unique_aids)
        qreq_.aid_to_idx = ut.make_index_lookup(qreq_.unique_aids)

    @ut.accepts_numpy
    def get_qreq_annot_nids(qreq_, aids):
        # uses own internal state to grab name rowids instead of using wbia.
        if not hasattr(qreq_, 'aid_to_idx'):
            qreq_.ensure_nids()
        idxs = ut.take(qreq_.aid_to_idx, aids)
        nids = ut.take(qreq_.unique_nids, idxs)
        return nids

    @ut.accepts_numpy
    def get_qreq_annot_gids(qreq_, aids):
        # Hack uses own internal state to grab name rowids
        # instead of using wbia.
        return qreq_.ibs.get_annot_gids(aids)

    @property
    def dnids(qreq_):
        """ TODO: save dnids in qreq_ state """
        # return qreq_.dannots.nids
        return qreq_.get_qreq_annot_nids(qreq_.daids)

    @property
    def qnids(qreq_):
        """ TODO: save qnids in qreq_ state """
        # return qreq_.qannots.nids
        return qreq_.get_qreq_annot_nids(qreq_.qaids)

    @property
    def extern_query_config2(qreq_):
        return qreq_.qparams

    @property
    def extern_data_config2(qreq_):
        return qreq_.qparams


def execute_bulk(qreq_):
    # Do not use bulk single queries
    bulk_on = qreq_.use_bulk_cache and len(qreq_.qaids) > qreq_.min_bulk_size
    if bulk_on:
        # Try and load directly from a big cache
        bc_dpath = ut.ensuredir((qreq_.cachedir, 'bulk_mc5'))
        bc_fname = 'bulk_mc5_' + '_'.join(qreq_.get_nice_parts())
        bc_cfgstr = qreq_.get_cfgstr(with_input=True)
        try:
            cm_list = ut.load_cache(bc_dpath, bc_fname, bc_cfgstr)
            print('... bulk cache hit %r/%r' % (len(qreq_), len(qreq_)))
        except (IOError, AttributeError):
            # Fallback to smallcache
            cm_list = execute_singles(qreq_)
            ut.save_cache(bc_dpath, bc_fname, bc_cfgstr, cm_list)
    else:
        # Fallback to smallcache
        cm_list = execute_singles(qreq_)
    return cm_list


def _load_singles(qreq_):
    # Find existing cached chip matches
    # Try loading as many as possible
    fpath_list = qreq_.get_chipmatch_fpaths(qreq_.qaids)
    exists_flags = [exists(fpath) for fpath in fpath_list]
    qaids_hit = ut.compress(qreq_.qaids, exists_flags)
    fpaths_hit = ut.compress(fpath_list, exists_flags)
    # First, try a fast reload assuming no errors
    fpath_iter = ut.ProgIter(
        fpaths_hit,
        length=len(fpaths_hit),
        enabled=len(fpaths_hit) > 1,
        label='loading cache hits',
        adjust=True,
        freq=1,
    )
    try:
        qaid_to_hit = {
            qaid: chip_match.ChipMatch.load_from_fpath(fpath, verbose=False)
            for qaid, fpath in zip(qaids_hit, fpath_iter)
        }
    except chip_match.NeedRecomputeError as ex:
        # Fallback to a slow reload
        ut.printex(ex, 'Some cached results need to recompute', iswarning=True)
        qaid_to_hit = _load_singles_fallback(fpaths_hit)
    return qaid_to_hit


def _load_singles_fallback(fpaths_hit):
    fpath_iter = ut.ProgIter(
        fpaths_hit,
        enabled=len(fpaths_hit) > 1,
        label='checking chipmatch cache',
        adjust=True,
        freq=1,
    )
    # Recompute those that fail loading
    qaid_to_hit = {}
    for fpath in fpath_iter:
        try:
            cm = chip_match.ChipMatch.load_from_fpath(fpath, verbose=False)
        except chip_match.NeedRecomputeError:
            pass
        else:
            qaid_to_hit[cm.qaid] = cm
    print(
        '%d / %d cached matches need to be recomputed'
        % (len(fpaths_hit) - len(qaid_to_hit), len(fpaths_hit))
    )
    return qaid_to_hit


def execute_singles(qreq_):
    if qreq_.use_single_cache:
        qaid_to_hit = _load_singles(qreq_)
    else:
        qaid_to_hit = {}
    hit_all = len(qaid_to_hit) == len(qreq_.qaids)
    hit_any = len(qaid_to_hit) > 0

    if hit_all:
        qaid_to_cm = qaid_to_hit
    else:
        if hit_any:
            print('... partial cm cache hit %d/%d' % (len(qaid_to_hit), len(qreq_)))
            hit_aids = list(qaid_to_hit.keys())
            miss_aids = ut.setdiff(qreq_.qaids, hit_aids)
            qreq_miss = qreq_.shallowcopy(miss_aids)
        else:
            qreq_miss = qreq_
        # Compute misses
        qreq_miss.ensure_data()
        qaid_to_cm = execute_and_save(qreq_miss)
        # Merge misses with hits
        if hit_any:
            qaid_to_cm.update(qaid_to_hit)
    cm_list = ut.take(qaid_to_cm, qreq_.qaids)
    return cm_list


def execute_and_save(qreq_miss):
    # Iterate over vsone queries in chunks.
    total_chunks = ut.get_num_chunks(len(qreq_miss.qaids), qreq_miss.chunksize)
    qaid_chunk_iter = ut.ichunks(qreq_miss.qaids, qreq_miss.chunksize)
    _prog = ut.ProgPartial(
        length=total_chunks,
        freq=1,
        label='[mc5] query chunk: ',
        prog_hook=qreq_miss.prog_hook,
        bs=False,
    )
    qaid_chunk_iter = iter(_prog(qaid_chunk_iter))

    qaid_to_cm = {}
    for qaids in qaid_chunk_iter:
        sub_qreq = qreq_miss.shallowcopy(qaids=qaids)
        cm_batch = sub_qreq.execute_pipeline()
        assert len(cm_batch) == len(qaids), 'bad alignment'
        assert all([qaid == cm.qaid for qaid, cm in zip(qaids, cm_batch)])

        # TODO: we already computed the fpaths
        # should be able to pass them in
        fpath_list = sub_qreq.get_chipmatch_fpaths(qaids)
        _prog = ut.ProgPartial(
            length=len(cm_batch),
            adjust=True,
            freq=1,
            label='saving chip matches',
            bs=True,
        )
        for cm, fpath in _prog(zip(cm_batch, fpath_list)):
            cm.save_to_fpath(fpath, verbose=False)
        qaid_to_cm.update({cm.qaid: cm for cm in cm_batch})

    return qaid_to_cm
