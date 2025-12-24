# -*- coding: utf-8 -*-
"""
FAISS-based Neighbor Index for HotSpotter

This module provides a drop-in replacement for the FLANN-based NeighborIndex
using Facebook AI's FAISS library for approximate nearest neighbor search.

FAISS IVF (Inverted File Index) provides significant speedups for large databases
by clustering the feature space and only searching relevant clusters at query time.

Key parameters:
    nlist: Number of clusters (Voronoi cells) to partition the feature space
    nprobe: Number of clusters to search at query time (accuracy/speed trade-off)

For GPU acceleration, set use_gpu=True in faiss_params.

References:
    https://github.com/facebookresearch/faiss
    https://faiss.ai/
"""
import logging
from os.path import basename, join

import numpy as np
import utool as ut

from wbia.algo.hots import hstypes
from wbia.algo.hots.neighbor_index import invert_index

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning('[faiss] FAISS not available. Install with: pip install faiss-gpu (or faiss-cpu)')

# Default FAISS parameters
DEFAULT_FAISS_PARAMS = {
    'nlist': 100,       # Number of clusters (increase for larger databases)
    'nprobe': 10,       # Number of clusters to search (accuracy/speed trade-off)
    'use_gpu': True,    # Use GPU acceleration if available
    'gpu_id': 0,        # GPU device ID
    'train_ratio': 1.0, # Ratio of vectors to use for training (1.0 = all)
}

# Scaling recommendations for nlist based on database size
# Rule of thumb: nlist ~= sqrt(n_vectors) for optimal trade-off
NLIST_RECOMMENDATIONS = {
    10000: 100,      # 10K vectors -> 100 clusters
    100000: 316,     # 100K vectors -> ~316 clusters
    1000000: 1000,   # 1M vectors -> 1000 clusters
    10000000: 3162,  # 10M vectors -> ~3162 clusters
}


def get_recommended_nlist(num_vectors):
    """
    Get recommended nlist based on database size.
    Uses sqrt(n) heuristic with bounds.
    """
    nlist = int(np.sqrt(num_vectors))
    # Clamp to reasonable range
    nlist = max(16, min(nlist, 65536))
    return nlist


def get_recommended_nprobe(nlist, accuracy='balanced'):
    """
    Get recommended nprobe based on nlist and desired accuracy level.

    Args:
        nlist: Number of clusters
        accuracy: 'fast' (1%), 'balanced' (10%), 'accurate' (25%)
    """
    ratios = {
        'fast': 0.01,
        'balanced': 0.10,
        'accurate': 0.25,
    }
    ratio = ratios.get(accuracy, 0.10)
    nprobe = max(1, int(nlist * ratio))
    return nprobe


@ut.reloadable_class
class FaissNeighborIndex(object):
    """
    FAISS-based neighbor index with IVF clustering for scalable search.

    This is a drop-in replacement for NeighborIndex that uses FAISS IVF
    instead of FLANN kd-trees. The interface is compatible with the
    existing HotSpotter pipeline.

    Key advantages over FLANN:
        - Better scaling to large databases (millions of vectors)
        - GPU acceleration support
        - Tunable accuracy/speed trade-off via nprobe
        - Active maintenance by Meta

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index_faiss import *
        >>> nnindexer, qreq_, ibs = testdata_faiss_nnindexer()
    """

    ext = '.faiss'
    prefix1 = 'faiss'

    def __init__(nnindexer, faiss_params, cfgstr):
        """
        Initialize an empty FAISS neighbor indexer.

        Args:
            faiss_params (dict): FAISS configuration parameters
                - nlist: Number of IVF clusters
                - nprobe: Number of clusters to search
                - use_gpu: Whether to use GPU acceleration
                - gpu_id: GPU device ID
            cfgstr (str): Configuration string for caching
        """
        if not FAISS_AVAILABLE:
            raise ImportError('FAISS is not installed. Install with: pip install faiss-gpu')

        nnindexer.index = None           # FAISS index structure
        nnindexer.ax2_aid = None         # (A x 1) Mapping to original annot ids
        nnindexer.idx2_vec = None        # (M x D) Descriptors to index
        nnindexer.idx2_fgw = None        # (M x 1) Descriptor foreground weight
        nnindexer.idx2_ax = None         # (M x 1) Index into the aid_list
        nnindexer.idx2_fx = None         # (M x 1) Index into the annot's features
        nnindexer.cfgstr = cfgstr        # Configuration id

        # Merge with defaults
        nnindexer.faiss_params = DEFAULT_FAISS_PARAMS.copy()
        if faiss_params is not None:
            nnindexer.faiss_params.update(faiss_params)

        nnindexer.nlist = nnindexer.faiss_params.get('nlist', 100)
        nnindexer.nprobe = nnindexer.faiss_params.get('nprobe', 10)
        nnindexer.use_gpu = nnindexer.faiss_params.get('use_gpu', True)
        nnindexer.gpu_id = nnindexer.faiss_params.get('gpu_id', 0)

        nnindexer.num_indexed = None
        nnindexer.faiss_fpath = None
        nnindexer.max_distance_sqrd = None
        nnindexer.is_trained = False
        nnindexer.gpu_resources = None

    def init_support(indexer, aid_list, vecs_list, fgws_list, fxs_list, verbose=True):
        """
        Prepares inverted indices and FAISS index structure.

        Flattens vecs_list and builds a reverse index from the flattened indices
        (idx) to the original aids and fxs.
        """
        assert indexer.index is None, 'already initialized'

        logger.info('[faiss] Preparing data for indexing / loading index')
        # Check input
        assert len(aid_list) == len(vecs_list), 'invalid input. bad len'
        assert len(aid_list) > 0, 'len(aid_list) == 0. Cannot invert index without features!'

        # Create indexes into the input aids
        ax_list = np.arange(len(aid_list))

        # Invert indices (reuse existing function)
        tup = invert_index(vecs_list, fgws_list, ax_list, fxs_list, verbose=verbose)
        idx2_vec, idx2_fgw, idx2_ax, idx2_fx = tup

        ax2_aid = np.array(aid_list)

        indexer.ax2_aid = ax2_aid
        indexer.idx2_vec = idx2_vec
        indexer.idx2_fgw = idx2_fgw
        indexer.idx2_ax = idx2_ax
        indexer.idx2_fx = idx2_fx
        indexer.aid2_ax = ut.make_index_lookup(indexer.ax2_aid)
        indexer.num_indexed = indexer.idx2_vec.shape[0]

        if indexer.idx2_vec.dtype == hstypes.VEC_TYPE:
            indexer.max_distance_sqrd = hstypes.VEC_PSEUDO_MAX_DISTANCE_SQRD
        else:
            indexer.max_distance_sqrd = None

        # Auto-tune nlist based on database size
        if indexer.num_indexed > 0:
            recommended_nlist = get_recommended_nlist(indexer.num_indexed)
            if indexer.nlist != recommended_nlist:
                logger.info(
                    '[faiss] Auto-adjusting nlist from %d to %d for %d vectors'
                    % (indexer.nlist, recommended_nlist, indexer.num_indexed)
                )
                indexer.nlist = recommended_nlist
                indexer.nprobe = get_recommended_nprobe(indexer.nlist)

    def _build_index(indexer, verbose=True):
        """
        Build the FAISS IVF index structure.
        """
        num_vecs = indexer.num_indexed
        dim = indexer.idx2_vec.shape[1] if num_vecs > 0 else hstypes.VEC_DIM

        if num_vecs == 0:
            logger.warning('[faiss] Cannot build index over 0 vectors')
            return

        # Ensure nlist doesn't exceed number of vectors
        effective_nlist = min(indexer.nlist, num_vecs)
        if effective_nlist < indexer.nlist:
            logger.info('[faiss] Reducing nlist from %d to %d (num_vecs=%d)'
                       % (indexer.nlist, effective_nlist, num_vecs))
            indexer.nlist = effective_nlist

        if verbose:
            logger.info('[faiss] Building IVF index: %d vectors, dim=%d, nlist=%d'
                       % (num_vecs, dim, indexer.nlist))

        # Convert uint8 SIFT descriptors to float32 for FAISS
        vecs_float = indexer.idx2_vec.astype(np.float32)

        # Create the IVF index
        # Using IndexFlatL2 as the quantizer for exact L2 distance within clusters
        quantizer = faiss.IndexFlatL2(dim)

        # IVFFlat: IVF with exact search within clusters (no compression)
        index_cpu = faiss.IndexIVFFlat(quantizer, dim, indexer.nlist, faiss.METRIC_L2)

        # Train the index (learn cluster centroids)
        if verbose:
            logger.info('[faiss] Training IVF clusters...')

        # Use subset for training if dataset is very large
        train_ratio = indexer.faiss_params.get('train_ratio', 1.0)
        if train_ratio < 1.0 and num_vecs > 100000:
            n_train = int(num_vecs * train_ratio)
            train_indices = np.random.choice(num_vecs, n_train, replace=False)
            train_vecs = vecs_float[train_indices]
            logger.info('[faiss] Training on %d/%d vectors (%.1f%%)'
                       % (n_train, num_vecs, train_ratio * 100))
        else:
            train_vecs = vecs_float

        index_cpu.train(train_vecs)
        indexer.is_trained = True

        # Add vectors to the index
        if verbose:
            logger.info('[faiss] Adding %d vectors to index...' % num_vecs)
        index_cpu.add(vecs_float)

        # Move to GPU if requested and available
        try:
            num_gpus = faiss.get_num_gpus() if indexer.use_gpu else 0
        except Exception as ex:
            logger.warning('[faiss] GPU detection failed: %s. Using CPU.' % ex)
            num_gpus = 0

        if indexer.use_gpu and num_gpus > 0:
            if verbose:
                logger.info('[faiss] Moving index to GPU %d' % indexer.gpu_id)
            indexer.gpu_resources = faiss.StandardGpuResources()
            indexer.index = faiss.index_cpu_to_gpu(
                indexer.gpu_resources, indexer.gpu_id, index_cpu
            )
        else:
            if indexer.use_gpu and num_gpus == 0:
                logger.warning('[faiss] GPU requested but no GPU available, using CPU')
            indexer.index = index_cpu

        # Set search parameters
        indexer.index.nprobe = indexer.nprobe

        if verbose:
            logger.info('[faiss] Index ready. nprobe=%d (searching %.1f%% of clusters)'
                       % (indexer.nprobe, 100.0 * indexer.nprobe / indexer.nlist))

    def ensure_indexer(nnindexer, cachedir, verbose=True, force_rebuild=False,
                       memtrack=None, prog_hook=None):
        """
        Ensures that you get a neighbor indexer. Either loads a cached
        indexer or rebuilds a new one.
        """
        if force_rebuild:
            logger.info('[faiss] Force rebuild requested')
            load_success = False
        else:
            try:
                load_success = nnindexer.load(cachedir, verbose=verbose)
            except Exception as ex:
                logger.warning('[faiss] Failed to load cached index: %s' % ex)
                load_success = False

        if load_success:
            if not ut.QUIET:
                logger.info('[faiss] Cache hit: %d vectors, %d annots'
                           % (nnindexer.num_indexed_vecs(), nnindexer.num_indexed_annots()))
        else:
            if not ut.QUIET:
                logger.info('[faiss] Cache miss: building new index')
            if prog_hook is not None:
                prog_hook.set_progress(1, 2, 'Building FAISS index')
            nnindexer.build_and_save(cachedir, verbose=verbose, memtrack=memtrack)

        if prog_hook is not None:
            prog_hook.set_progress(2, 2, 'Finished loading FAISS indexer')

    def build_and_save(nnindexer, cachedir, verbose=True, memtrack=None):
        nnindexer._build_index(verbose=verbose)
        nnindexer.save(cachedir, verbose=verbose)

    def reindex(nnindexer, verbose=True, memtrack=None):
        """Rebuild the FAISS index."""
        nnindexer._build_index(verbose=verbose)

    # ---- Caching interface ----

    def save(nnindexer, cachedir=None, fpath=None, verbose=True):
        """Save FAISS index to disk."""
        if fpath is None:
            faiss_fpath = nnindexer.get_fpath(cachedir)
        else:
            faiss_fpath = fpath

        nnindexer.faiss_fpath = faiss_fpath

        if nnindexer.index is None:
            logger.warning('[faiss] No index to save')
            return False

        if verbose:
            logger.info('[faiss] Saving index to %s' % ut.path_ndir_split(faiss_fpath, n=3))

        # If index is on GPU, move to CPU for saving
        if hasattr(nnindexer.index, 'index'):
            # GPU index - extract CPU version
            index_cpu = faiss.index_gpu_to_cpu(nnindexer.index)
        else:
            index_cpu = nnindexer.index

        faiss.write_index(index_cpu, faiss_fpath)
        return True

    def load(nnindexer, cachedir=None, fpath=None, verbose=True):
        """Load FAISS index from disk."""
        if fpath is None:
            faiss_fpath = nnindexer.get_fpath(cachedir)
        else:
            faiss_fpath = fpath

        nnindexer.faiss_fpath = faiss_fpath

        if not ut.checkpath(faiss_fpath, verbose=verbose):
            return False

        try:
            if verbose:
                logger.info('[faiss] Loading index from %s' % ut.path_ndir_split(faiss_fpath, n=3))

            index_cpu = faiss.read_index(faiss_fpath)

            # Move to GPU if requested
            try:
                num_gpus = faiss.get_num_gpus() if nnindexer.use_gpu else 0
            except Exception as ex:
                logger.warning('[faiss] GPU detection failed: %s. Using CPU.' % ex)
                num_gpus = 0

            if nnindexer.use_gpu and num_gpus > 0:
                if verbose:
                    logger.info('[faiss] Moving loaded index to GPU %d' % nnindexer.gpu_id)
                nnindexer.gpu_resources = faiss.StandardGpuResources()
                nnindexer.index = faiss.index_cpu_to_gpu(
                    nnindexer.gpu_resources, nnindexer.gpu_id, index_cpu
                )
            else:
                nnindexer.index = index_cpu

            # Set search parameters
            nnindexer.index.nprobe = nnindexer.nprobe
            nnindexer.is_trained = True

            return True
        except Exception as ex:
            logger.warning('[faiss] Failed to load index: %s' % ex)
            return False

    def get_prefix(nnindexer):
        return nnindexer.prefix1

    def get_cfgstr(nnindexer, noquery=False):
        """Returns string which uniquely identifies configuration and support data."""
        cfgstr_parts = []

        # FAISS params
        faiss_cfg = 'FAISS(nlist=%d,nprobe=%d)' % (nnindexer.nlist, nnindexer.nprobe)
        cfgstr_parts.append(faiss_cfg)

        # Data hash
        if nnindexer.idx2_vec is not None:
            vecs_hashstr = ut.hashstr_arr(nnindexer.idx2_vec, '_VECS')
            cfgstr_parts.append(vecs_hashstr)

        return ''.join(cfgstr_parts)

    def get_fname(nnindexer):
        return basename(nnindexer.get_fpath(''))

    def get_fpath(nnindexer, cachedir, cfgstr=None):
        _args2_fpath = ut.util_cache._args2_fpath
        prefix = nnindexer.get_prefix()
        cfgstr = nnindexer.get_cfgstr(noquery=True)
        ext = nnindexer.ext
        fpath = _args2_fpath(cachedir, prefix, cfgstr, ext)
        return fpath

    # ---- Query interface ----

    def get_dtype(nnindexer):
        return nnindexer.idx2_vec.dtype

    @profile
    def knn(indexer, qfx2_vec, K):
        """
        Returns the indices and squared distance to the nearest K neighbors.

        Args:
            qfx2_vec: (N x D) array of N, D-dimensional query vectors
            K: number of approximate nearest neighbors to find

        Returns:
            tuple: (qfx2_idx, qfx2_dist)
                qfx2_idx: (N x K) indices of nearest neighbors
                qfx2_dist: (N x K) normalized squared distances
        """
        if K == 0:
            return indexer.empty_neighbors(len(qfx2_vec), 0)

        if K > indexer.num_indexed:
            K = indexer.num_indexed
            if K == 0:
                return indexer.empty_neighbors(len(qfx2_vec), 0)

        if len(qfx2_vec) == 0:
            return indexer.empty_neighbors(0, K)

        # Convert query vectors to float32 for FAISS
        qfx2_vec_float = qfx2_vec.astype(np.float32)

        # Perform search
        qfx2_raw_dist, qfx2_idx = indexer.index.search(qfx2_vec_float, K)

        # FAISS returns int64 indices, convert to int32 for compatibility
        qfx2_idx = qfx2_idx.astype(np.int32)

        # Handle potential -1 indices (not enough neighbors)
        # This shouldn't happen with IVFFlat but be safe
        invalid_mask = qfx2_idx < 0
        if np.any(invalid_mask):
            logger.warning('[faiss] Got %d invalid indices' % invalid_mask.sum())
            qfx2_idx[invalid_mask] = 0
            qfx2_raw_dist[invalid_mask] = np.finfo(np.float32).max

        # Normalize distances (same as FLANN version)
        if indexer.max_distance_sqrd is not None:
            qfx2_dist = np.divide(qfx2_raw_dist, indexer.max_distance_sqrd)
        else:
            qfx2_dist = qfx2_raw_dist

        return (qfx2_idx, qfx2_dist)

    def batch_knn(indexer, vecs, K, chunksize=4096, label='batch knn'):
        """
        KNN with batching and progress reporting.
        """
        idxs = np.empty((vecs.shape[0], K), dtype=np.int32)
        dists = np.empty((vecs.shape[0], K), dtype=np.float32)

        num_chunks = ut.get_num_chunks(vecs.shape[0], chunksize)
        iter_ = ut.ichunk_slices(vecs.shape[0], chunksize)
        prog = ut.ProgIter(iter_, length=num_chunks, label=label)

        for sl_ in prog:
            idxs[sl_], dists[sl_] = indexer.knn(vecs[sl_], K=K)

        return idxs, dists

    def empty_neighbors(nnindexer, nQfx, K):
        qfx2_idx = np.empty((nQfx, K), dtype=np.int32)
        qfx2_dist = np.empty((nQfx, K), dtype=np.float64)
        return (qfx2_idx, qfx2_dist)

    # ---- Index metadata ----

    def num_indexed_vecs(nnindexer):
        if nnindexer.idx2_vec is None:
            return 0
        return nnindexer.idx2_vec.shape[0]

    def num_indexed_annots(nnindexer):
        if nnindexer.ax2_aid is None:
            return 0
        return (nnindexer.ax2_aid != -1).sum()

    def get_indexed_aids(nnindexer):
        return nnindexer.ax2_aid[nnindexer.ax2_aid != -1]

    # ---- Lookup methods (same as NeighborIndex) ----

    def get_nn_vecs(nnindexer, qfx2_nnidx):
        """Gets matching vectors."""
        return nnindexer.idx2_vec.take(qfx2_nnidx, axis=0)

    def get_nn_axs(nnindexer, qfx2_nnidx):
        """Gets matching internal annotation indices."""
        return nnindexer.idx2_ax.take(qfx2_nnidx)

    def get_nn_aids(nnindexer, qfx2_nnidx):
        """
        Gets annotation IDs for nearest neighbor indices.

        Args:
            qfx2_nnidx: (N x K) indices from knn search

        Returns:
            qfx2_aid: (N x K) annotation IDs
        """
        qfx2_ax = nnindexer.idx2_ax.take(qfx2_nnidx)
        qfx2_aid = nnindexer.ax2_aid.take(qfx2_ax)
        return qfx2_aid

    def get_nn_featxs(nnindexer, qfx2_nnidx):
        """Gets feature indices within annotations."""
        qfx2_fx = nnindexer.idx2_fx.take(qfx2_nnidx)
        return qfx2_fx

    def get_nn_fgws(nnindexer, qfx2_nnidx):
        """Gets foreground weights of neighbors."""
        if nnindexer.idx2_fgw is None:
            return np.ones(qfx2_nnidx.shape)
        return nnindexer.idx2_fgw.take(qfx2_nnidx)


def testdata_faiss_nnindexer(dbname='testdb1', use_memcache=True):
    """
    Test data for FAISS indexer development.
    """
    import wbia
    qreq_ = wbia.testdata_qreq_(defaultdb=dbname, p='default:fg_on=False')
    ibs = qreq_.ibs
    daid_list = qreq_.daids

    # Get support data
    from wbia.algo.hots.neighbor_index import get_support_data
    vecs_list, fgws_list, fxs_list = get_support_data(qreq_, daid_list)

    # Build FAISS index
    faiss_params = {'nlist': 10, 'nprobe': 3, 'use_gpu': False}
    cfgstr = 'test_faiss'
    nnindexer = FaissNeighborIndex(faiss_params, cfgstr)
    nnindexer.init_support(daid_list, vecs_list, fgws_list, fxs_list, verbose=True)
    nnindexer._build_index(verbose=True)

    return nnindexer, qreq_, ibs
