from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import exists
from ibeis.algo.hots import chip_match
# from ibeis.algo.hots import pipeline
import utool as ut


class RequestCacher(object):

    def __init__(self):
        self.use_cache = True
        self.use_bigcache = True
        self.min_bigcache_size = 64
        self.use_cache_save = True
        self.save_bigcache = True

    def request_cached1(self, qreq_):
        # Do not use bigcache single queries
        bigcache_on = (self.use_bigcache and self.use_cache and
                       len(self.qreq_.qaids) > self.min_bigcache_size)
        if bigcache_on:
            # Try and load directly from a big cache
            bc_dpath = ut.ensuredir((qreq_.cachedir, 'bc5'))
            bc_fname = 'BC5_' + '_'.join(qreq_.get_nice_parts())
            bc_cfgstr = qreq_.get_cfgstr(with_input=True)
            try:
                cm_list = ut.load_cache(bc_dpath, bc_fname, bc_cfgstr)
            except (IOError, AttributeError):
                # Fallback to smallcache
                cm_list = self.request_cached2()
                ut.save_cache(bc_dpath, bc_fname, bc_cfgstr, cm_list)
        else:
            # Fallback to smallcache
            cm_list = self.request_cached2()
        return cm_list

    def _load_singles(self, qreq_):
        # Find existing cached chip matches
        # Try loading as many as possible
        fpath_list = qreq_.get_chipmatch_fpaths(qreq_.qaids)
        exists_flags = [exists(fpath) for fpath in fpath_list]
        qaids_hit = ut.compress(qreq_.qaids, exists_flags)
        fpaths_hit = ut.compress(fpath_list, exists_flags)
        # First, try a fast reload assuming no errors
        fpath_iter = ut.ProgIter(
            fpaths_hit, nTotal=len(fpaths_hit), enabled=len(fpaths_hit) > 1,
            lbl='loading cache hits', adjust=True, freq=1)
        try:
            qaid2_cm_hit = {
                qaid: chip_match.ChipMatch.load_from_fpath(fpath, verbose=False)
                for qaid, fpath in zip(qaids_hit, fpath_iter)
            }
        except chip_match.NeedRecomputeError as ex:
            # Fallback to a slow reload
            ut.printex(ex, 'Some cached chips need to recompute',
                       iswarning=True)
            qaid2_cm_hit = self._load_singles_fallback(fpaths_hit)
        return qaid2_cm_hit

    def _load_singles_fallback(self, fpaths_hit):
        fpath_iter = ut.ProgIter(
            fpaths_hit, enabled=len(fpaths_hit) > 1,
            lbl='checking chipmatch cache', adjust=True, freq=1)
        # Recompute those that fail loading
        qaid2_cm_hit = {}
        for fpath in fpath_iter:
            try:
                cm = chip_match.ChipMatch.load_from_fpath(fpath, verbose=False)
            except chip_match.NeedRecomputeError:
                pass
            else:
                qaid2_cm_hit[cm.qaid] = cm
        print('%d / %d cached matches need to be recomputed' % (
            len(fpaths_hit) - len(qaid2_cm_hit), len(fpaths_hit)))
        return qaid2_cm_hit

    def request_cached2(self, qreq_):
        if self.use_cache:
            qaid2_cm_hit = self._load_singles(qreq_)
        else:
            qaid2_cm_hit = {}
        hit_all = len(qaid2_cm_hit) == len(qreq_.qaids)
        hit_any = len(qaid2_cm_hit) > 0

        if hit_all:
            qaid_to_cm = qaid2_cm_hit
        else:
            if hit_any:
                print('... partial cm cache hit %d/%d' % (
                    len(qaid2_cm_hit), len(qreq_.qaids)))
                qreq_miss = qreq_.shallowcopy(list(qaid2_cm_hit.keys()))
            else:
                qreq_miss = qreq_

            qaid_to_cm = self.execute_and_save(qreq_miss)

            # Merge cache hits with computed misses
            if hit_any:
                qaid_to_cm.update(qaid2_cm_hit)
        cm_list = ut.take(qaid_to_cm, qreq_.qaids)
        return cm_list

    def execute_and_save(self, qreq_miss):
        # Iterate over vsone queries in chunks.
        total_chunks = ut.get_nTotalChunks(len(qreq_miss.qaids), self.chunksize)
        qaid_chunk_iter = ut.ichunks(qreq_miss.qaids, self.chunksize)
        qaid_chunk_iter = ut.ProgIter(qaid_chunk_iter, nTotal=total_chunks,
                                      freq=1, lbl='[mc5] query chunk: ',
                                      prog_hook=qreq_miss.prog_hook)

        qaid_to_cm = {}
        for qaids in qaid_chunk_iter:
            sub_qreq = qreq_miss.shallowcopy(qaids=qaids)

            # FIXME;
            cm_batch = sub_qreq.execute_pipeline()
            assert all([qaid == cm.qaid for qaid, cm in zip(qaids, cm_batch)])

            # TODO: we already computed the fpaths
            # should be able to pass them in
            fpath_list = sub_qreq.get_chipmatch_fpaths(qaids)
            _iter = ut.ProgIter(zip(cm_batch, fpath_list), nTotal=len(cm_batch),
                                lbl='saving chip matches', adjust=True, freq=1)
            for cm, fpath in _iter:
                cm.save_to_fpath(fpath, verbose=False)
            qaid_to_cm.update({cm.qaid: cm for cm in cm_batch})

        return qaid_to_cm
