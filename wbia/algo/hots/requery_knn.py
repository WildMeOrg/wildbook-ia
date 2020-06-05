# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool as vt
import itertools as it
from six.moves import range, zip, map  # NOQA

(print, rrr, profile) = ut.inject2(__name__)


# DEBUG_REQUERY = True
DEBUG_REQUERY = False


class FinalResults(ut.NiceRepr):
    def __init__(final, shape):
        final.qfx2_idx = np.full(shape, -1, dtype=np.int32)
        final.qfx2_dist = np.full(shape, np.nan, dtype=np.float64)
        final.qfx2_truek = np.full(shape, -1, dtype=np.int32)

    def assign(final, index, idxs, dists, trueks):
        final.qfx2_idx[index, :] = idxs
        final.qfx2_dist[index, :] = dists
        final.qfx2_truek[index, :] = trueks

    def __nice__(final):
        return str(final.qfx2_idx)


class TempQuery(ut.NiceRepr):
    """ queries that are incomplete """

    def __init__(query, vecs, invalid_axs, get_neighbors, get_axs):
        # Static attributes
        query.invalid_axs = invalid_axs
        query.get_neighbors = get_neighbors
        query.get_axs = get_axs
        # Dynamic attributes
        query.index = np.arange(len(vecs))
        query.vecs = vecs

    def __nice__(query):
        return str(query.index)

    def neighbors(query, temp_K):
        _idxs, _dists = query.get_neighbors(query.vecs, temp_K)
        idxs = vt.atleast_nd(_idxs, 2)
        dists = vt.atleast_nd(_dists, 2)
        # Flag any neighbors that are invalid
        validflags = ~in1d_shape(query.get_axs(idxs), query.invalid_axs)
        # Store results in an object
        cand = TempResults(query.index, idxs, dists, validflags)
        return cand

    def compress_inplace(query, flags):
        query.index = query.index.compress(flags, axis=0)
        query.vecs = query.vecs.compress(flags, axis=0)


class TempResults(ut.NiceRepr):
    def __init__(cand, index, idxs, dists, validflags):
        cand.index = index
        cand.idxs = idxs
        cand.dists = dists
        cand.validflags = validflags

    def __nice__(cand):
        return str(cand.index)

    def compress(cand, flags):
        qfx = cand.index.compress(flags, axis=0)
        idx_ = cand.idxs.compress(flags, axis=0)
        dist_ = cand.dists.compress(flags, axis=0)
        valid_ = cand.validflags.compress(flags, axis=0)
        return TempResults(qfx, idx_, dist_, valid_)

    def done_flags(cand, num_neighbs):
        return cand.validflags.sum(axis=1) >= num_neighbs

    def done_part(cand, num_neighbs):
        # Find the first `num_neighbs` complete columns in each row
        rowxs, colxs = np.where(cand.validflags)
        unique_rows, groupxs = vt.group_indices(rowxs, assume_sorted=True)
        first_k_groupxs = [groupx[0:num_neighbs] for groupx in groupxs]
        if DEBUG_REQUERY:
            assert all(ut.issorted(groupx) for groupx in groupxs)
            assert all([len(group) == num_neighbs for group in first_k_groupxs])
        chosen_xs = np.array(ut.flatten(first_k_groupxs), dtype=np.int)
        # chosen_xs = np.hstack(first_k_groupxs)
        # then convert these to multi-indices
        done_rows = rowxs.take(chosen_xs)
        done_cols = colxs.take(chosen_xs)
        multi_index = (done_rows, done_cols)
        # done_shape = (cand.validflags.shape[0], num_neighbs)
        # flat_xs = np.ravel_multi_index(multi_index, done_shape)
        flat_xs = np.ravel_multi_index(multi_index, cand.idxs.shape)
        _shape = (-1, num_neighbs)
        idxs = cand.idxs.take(flat_xs).reshape(_shape)
        dists = cand.dists.take(flat_xs).reshape(_shape)

        trueks = colxs.take(chosen_xs).reshape(_shape)
        if DEBUG_REQUERY:
            # dists2 = dists.copy()
            for count, (row, cols) in enumerate(zip(unique_rows, groupxs)):
                pass
            assert np.all(np.diff(dists, axis=1) >= 0)
            valid = cand.validflags.take(flat_xs).reshape(_shape)
            assert np.all(valid)
        return idxs, dists, trueks


def in1d_shape(arr1, arr2):
    return np.in1d(arr1, arr2).reshape(arr1.shape)


def requery_knn(
    get_neighbors,
    get_axs,
    qfx2_vec,
    num_neighbs,
    invalid_axs=[],
    pad=2,
    limit=4,
    recover=True,
):
    """
    Searches for `num_neighbs`, while ignoring certain matches.  K is
    increassed until enough valid neighbors are found or a limit is reached.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.neighbor_index import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.testdata_qreq_(defaultdb='testdb1', a='default')
        >>> qreq_.load_indexer()
        >>> indexer = qreq_.indexer
        >>> qannot = qreq_.internal_qannots[1]
        >>> qfx2_vec = qannot.vecs
        >>> ibs = qreq_.ibs
        >>> qaid = qannot.aid
        >>> impossible_aids = ibs.get_annot_groundtruth(qaid, noself=False)
        >>> invalid_axs = np.array(ut.take(indexer.aid2_ax, impossible_aids))
        >>> pad = 0
        >>> limit = 1
        >>> num_neighbs = 3
        >>> def get_neighbors(vecs, temp_K):
        >>>     return indexer.flann.nn_index(vecs, temp_K, checks=indexer.checks,
        >>>                                   cores=indexer.cores)
        >>> get_axs = indexer.get_nn_axs
        >>> res = requery_knn(
        >>>     get_neighbors, get_axs, qfx2_vec, num_neighbs, invalid_axs, pad,
        >>>     limit, recover=True)
        >>> qfx2_idx, qfx2_dist = res
        >>> assert np.all(np.diff(qfx2_dist, axis=1) >= 0)

    Ignore:
        >>> from wbia.algo.hots.neighbor_index import *  # NOQA
        >>> from wbia.algo.hots.requery_knn import *  # NOQA
        >>> max_k = 9
        >>> n_pts = 5
        >>> num_neighbs = 3
        >>> temp_K = num_neighbs * 2
        >>> #
        >>> # Create dummy data
        >>> rng = np.random.RandomState(0)
        >>> tx2_idx_full = rng.randint(0, 10, size=(n_pts, max_k))
        >>> tx2_idx_full[:, 0] = 0
        >>> tx2_dist_full = np.meshgrid(np.arange(max_k), np.arange(n_pts))[0] / 10
        >>> tx2_dist_full += (rng.rand(n_pts, max_k) * 10).astype(np.int) / 100
        >>> qfx2_vec = np.arange(n_pts)[:, None]
        >>> vecs = qfx2_vec
        >>> #
        >>> pad = 0
        >>> limit = 1
        >>> recover = True
        >>> #
        >>> invalid_axs = np.array([0, 1, 2, 5, 7, 9])
        >>> get_axs = ut.identity
        >>> #
        >>> def get_neighbors(vecs, temp_K):
        >>>     # simulates finding k nearest neighbors
        >>>     idxs = tx2_idx_full[vecs.ravel(), 0:temp_K]
        >>>     dists = tx2_dist_full[vecs.ravel(), 0:temp_K]
        >>>     return idxs, dists
        >>> #
        >>> res = requery_knn(
        >>>     get_neighbors, get_axs, qfx2_vec, num_neighbs, invalid_axs, pad,
        >>>     limit, recover=True)
        >>> qfx2_idx, qfx2_dist = res
    """

    # Alloc space for final results
    shape = (len(qfx2_vec), num_neighbs)
    final = FinalResults(shape)  # NOQA
    query = TempQuery(qfx2_vec, invalid_axs, get_neighbors, get_axs)

    temp_K = num_neighbs + pad
    assert limit > 0, 'must have at least one iteration'
    at_limit = False

    for count in it.count():
        # print('count = %r' % (count,))
        cand = query.neighbors(temp_K)
        # Find which query features have found enough neighbors
        done_flags = cand.done_flags(num_neighbs)
        if DEBUG_REQUERY:
            print('count = %r' % (count,))
            assert np.all(np.diff(cand.dists, axis=1) >= 0)
            print('done_flags = %r' % (done_flags,))
        # Move any done queries into results and compress the query
        if np.any(done_flags):
            # Get the valid part of the results
            done = cand.compress(done_flags)
            idxs, dists, trueks = done.done_part(num_neighbs)
            final.assign(done.index, idxs, dists, trueks)
            if DEBUG_REQUERY:
                assert np.all(np.diff(dists, axis=1) >= 0)
                blocks = final.qfx2_dist
                nanelem_flags = np.isnan(blocks)
                nanrow_flags = np.any(nanelem_flags, axis=1)
                assert np.all(nanelem_flags.sum(axis=1)[nanrow_flags] == num_neighbs)
                assert np.all(np.diff(blocks[~nanrow_flags], axis=1) >= 0)
                print('final.qfx2_dist')
                print(final.qfx2_dist)
            if np.all(done_flags):
                # If everything was found then we are done
                break
            else:
                # Continue query with remaining invalid results
                query.compress_inplace(~done_flags)

        # double the search space
        temp_K *= 2

        at_limit = limit is not None and count >= limit
        if at_limit:
            if len(done_flags) == 0:
                import utool

                utool.embed()
            print(
                '[knn] Hit limit=%r and found %d/%d'
                % (limit, sum(done_flags), len(done_flags))
            )
            break

    if at_limit and recover:
        # If over the limit, then we need to do the best with what we have
        # otherwise we would just return nan
        best = cand.compress(~done_flags)
        print('[knn] Recover for %d features' % (len(best.index)))
        # Simply override the last indices to be valid and use those
        best.validflags[:, -num_neighbs:] = True
        # Now we can find a valid part
        idxs, dists, trueks = best.done_part(num_neighbs)
        final.assign(best.index, idxs, dists, trueks)
        if DEBUG_REQUERY:
            print('final.qfx2_dist')
            print(final.qfx2_dist)
    return final.qfx2_idx, final.qfx2_dist
