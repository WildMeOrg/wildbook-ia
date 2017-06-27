# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import vtool as vt
import numpy as np
import ubelt as ub
import pandas as pd
import dtool as dt
from os.path import join
from ibeis.algo.graph import nx_utils as nxu
print, rrr, profile = ut.inject2(__name__)


class PairFeatureConfig(dt.Config):
    """
    Config for building pairwise feature dimensions

    I.E. Config to distil unordered feature correspondences into a fixed length
    vector.
    """
    _param_info_list = [
        # ut.ParamInfo('indices', slice(0, 5)),
        ut.ParamInfo('indices', []),
        ut.ParamInfo('summary_ops', {
            # 'invsum',
            'sum', 'std', 'mean', 'len', 'med'}),
        ut.ParamInfo('local_keys', None),
        ut.ParamInfo('sorters', [
            # 'ratio', 'norm_dist', 'match_dist'
            # 'lnbnn', 'lnbnn_norm_dist',
        ]),
        # ut.ParamInfo('bin_key', None, valid_values=[None, 'ratio']),
        ut.ParamInfo('bin_key', 'ratio', valid_values=[None, 'ratio']),
        # ut.ParamInfo('bins', [.5, .6, .7, .8])
        # ut.ParamInfo('bins', (.625, .8), type_=eval),
        ut.ParamInfo('bins', (.625,), type_=eval),
        # ut.ParamInfo('bins', None, type_=eval),
        # ut.ParamInfo('need_lnbnn', False),
        # ut.ParamInfo('med', True),
    ]


class VsOneMatchConfig(dt.Config):
    _param_info_list = vt.matching.VSONE_DEFAULT_CONFIG


class VsOneFeatConfig(dt.Config):
    """ keypoint params """
    _param_info_list = vt.matching.VSONE_FEAT_CONFIG


class MatchConfig(dt.Config):
    _param_info_list = (vt.matching.VSONE_DEFAULT_CONFIG +
                        vt.matching.VSONE_FEAT_CONFIG)


class PairwiseFeatureExtractor(object):
    r"""
    Args:
        ibs (ibeis.IBEISController): image analysis api
        match_config (dict): config for building feature correspondences
        pairfeat_cfg (dict): config for making the pairwise feat vec
        global_keys (list): global keys to use
        need_lnbnn (bool): use LNBNN for enrichment
        feat_dims (list): subset of feature dimensions (from pruning)
                          if None, then all dimensions are used
        use_cache (bool):  turns on disk based caching (default = True)
        verbose (int):  verbosity flag (default = 1)

    CommandLine:
        python -m ibeis.algo.verif.pairfeat PairwiseFeatureExtractor

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.algo.verif.pairfeat import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> extr = PairwiseFeatureExtractor(ibs)
        >>> edges = [(1, 2), (2, 3)]
        >>> X = extr.transform(edges)
        >>> featinfo = vt.AnnotPairFeatInfo(X.columns)
        >>> print(featinfo.get_infostr())
    """

    def __init__(extr, ibs=None, match_config={}, pairfeat_cfg={},
                 global_keys=[], need_lnbnn=False, feat_dims=None,
                 use_cache=True, verbose=1):

        extr.ibs = ibs
        extr.match_config = MatchConfig(**match_config)
        extr.pairfeat_cfg = PairFeatureConfig(**pairfeat_cfg)
        extr.global_keys = global_keys
        extr.need_lnbnn = need_lnbnn
        extr.feat_dims = feat_dims
        extr.verbose = verbose
        extr.use_cache = use_cache

    def transform(extr, edges):
        if extr.use_cache:
            feats = extr._cached_pairwise_features(edges)
        else:
            feats = extr._make_pairwise_features(edges)
            feats = extr._postprocess_feats(feats)
        return feats

    def _exec_pairwise_match(extr, edges, prog_hook=None):
        """
        Performs one-vs-one matching between pairs of annotations.
        This establishes the feature correspondences.

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.verif.pairfeat import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb('testdb1')
            >>> extr = PairwiseFeatureExtractor(ibs)
            >>> edges = [(1, 2), (2, 3)]
            >>> prog_hook = None
            >>> match_list = extr._exec_pairwise_match(edges)
            >>> match1, match2 = match_list
            >>> assert match1.annot2 is match2.annot1
            >>> assert match1.annot1 is not match2.annot2
        """
        ibs = extr.ibs
        match_config = extr.match_config
        edges = ut.lmap(tuple, ut.aslist(edges))
        qaids = ut.take_column(edges, 0)
        daids = ut.take_column(edges, 1)
        # TODO: ensure feat/chip configs are resepected
        match_list = ibs.depc.get('pairwise_match', (qaids, daids), 'match',
                                  config=match_config)

        # Hack: Postprocess matches to re-add annotation info in lazy-dict
        # format
        from ibeis import core_annots
        config = ut.hashdict(match_config)
        qannot_cfg = dannot_cfg = config
        preload = True
        configured_lazy_annots = core_annots.make_configured_annots(
            ibs, qaids, daids, qannot_cfg, dannot_cfg, preload=preload)
        for qaid, daid, match in zip(qaids, daids, match_list):
            match.annot1 = configured_lazy_annots[config][qaid]
            match.annot2 = configured_lazy_annots[config][daid]
            match.config = config
        return match_list

    def _enrich_matches_lnbnn(extr, matches, other_aids, other_nids,
                              inplace=False):
        """
        Given a set of one-vs-one matches, searches for LNBNN normalizers in a
        larger database to enrich the matches with database-level
        distinctiveness.
        """
        from ibeis.algo.hots import nn_weights
        raise NotImplementedError('havent tested since the re-work. '
                                  'Need to ensure that things work correctly.')
        ibs = extr.ibs
        cfgdict = {
            'can_match_samename': False,
            'can_match_sameimg': True,
            'K': 3,
            'Knorm': 3,
            'prescore_method': 'csum',
            'score_method': 'csum'
        }
        custom_nid_lookup = ut.dzip(other_aids, other_nids)
        aids = [m.annot2['aid'] for m in matches]
        qreq_ = ibs.new_query_request(aids, other_aids, cfgdict=cfgdict,
                                      custom_nid_lookup=custom_nid_lookup,
                                      verbose=extr.verbose >= 2)

        qreq_.load_indexer()
        indexer = qreq_.indexer
        if not inplace:
            matches_ = [match.copy() for match in matches]
        else:
            matches_ = matches
        K = qreq_.qparams.K
        Knorm = qreq_.qparams.Knorm
        normalizer_rule  = qreq_.qparams.normalizer_rule

        extr.print('Stacking vecs for batch lnbnn matching')
        offset_list = np.cumsum([0] + [match_.fm.shape[0] for match_ in matches_])
        stacked_vecs = np.vstack([
            match_.matched_vecs2()
            for match_ in ut.ProgIter(matches_, label='stack matched vecs')
        ])

        vecs = stacked_vecs
        num = (K + Knorm)
        idxs, dists = indexer.batch_knn(vecs, num, chunksize=8192,
                                        label='lnbnn scoring')

        idx_list = [idxs[l:r] for l, r in ut.itertwo(offset_list)]
        dist_list = [dists[l:r] for l, r in ut.itertwo(offset_list)]
        iter_ = zip(matches_, idx_list, dist_list)
        prog = ut.ProgIter(iter_, nTotal=len(matches_), label='lnbnn scoring')
        for match_, neighb_idx, neighb_dist in prog:
            qaid = match_.annot2['aid']
            norm_k = nn_weights.get_normk(qreq_, qaid, neighb_idx, Knorm,
                                          normalizer_rule)
            ndist = vt.take_col_per_row(neighb_dist, norm_k)
            vdist = match_.local_measures['match_dist']
            lnbnn_dist = nn_weights.lnbnn_fn(vdist, ndist)
            lnbnn_clip_dist = np.clip(lnbnn_dist, 0, np.inf)
            match_.local_measures['lnbnn_norm_dist'] = ndist
            match_.local_measures['lnbnn'] = lnbnn_dist
            match_.local_measures['lnbnn_clip'] = lnbnn_clip_dist
            match_.fs = lnbnn_dist
        return matches_

    def _enriched_pairwise_matches(extr, edges, prog_hook=None):
        """
        Adds extra domain specific local and global properties that the match
        object (feature corresopndences) doesnt directly provide.

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.verif.pairfeat import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb('testdb1')
            >>> match_config = {
            >>>     'K': 1, 'Knorm': 3, 'affine_invariance': True,
            >>>     'augment_orientation': True, 'checks': 20, 'ratio_thresh': 0.8,
            >>>     'refine_method': 'homog', 'sv_on': True, 'sver_xy_thresh': 0.01,
            >>>     'symmetric': True, 'weight': 'fgweights'
            >>> }
            >>> global_keys = ['gps', 'qual', 'time', 'yaw']
            >>> extr = PairwiseFeatureExtractor(ibs, match_config=match_config,
            >>>                                 global_keys=global_keys)
            >>> edges = [(1, 2), (2, 3)]
            >>> prog_hook = None
            >>> match_list = extr._enriched_pairwise_matches(edges)
            >>> match1, match2 = match_list
            >>> assert match1.annot2 is match2.annot1
            >>> assert match1.annot1 is not match2.annot2
            >>> assert len(match1.global_measures) == 4
        """
        if extr.global_keys is None:
            raise ValueError('specify global keys')
            # global_keys = ['yaw', 'qual', 'gps', 'time']
            # global_keys = ['view', 'qual', 'gps', 'time']
        matches = extr._exec_pairwise_match(edges, prog_hook=prog_hook)
        print('enriching matches')
        if extr.need_lnbnn:
            extr._enrich_matches_lnbnn(matches, inplace=True)
        # Ensure matches know about relavent metadata
        for match in matches:
            vt.matching.ensure_metadata_normxy(match.annot1)
            vt.matching.ensure_metadata_normxy(match.annot2)
        for match in ut.ProgIter(matches, label='setup globals'):
            match.add_global_measures(extr.global_keys)
        for match in ut.ProgIter(matches, label='setup locals'):
            match.add_local_measures()
        return matches

    def _make_pairwise_features(extr, edges):
        """
        Construct matches and their pairwise features

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.algo.verif.pairfeat import *
            >>> from ibeis.algo.graph import demo
            >>> infr = demo.demodata_mtest_infr()
            >>> extr = PairwiseFeatureExtractor(ibs=infr.ibs)
            >>> config = {'K': 1, 'Knorm': 3, 'affine_invariance': True,
            >>>           'augment_orientation': True, 'checks': 20,
            >>>           'ratio_thresh': 0.8, 'refine_method': 'homog',
            >>>           'sv_on': True, 'sver_xy_thresh': 0.01,
            >>>           'symmetric': True, 'weight': 'fgweights'}
            >>> local_keys =  [
            >>>     'fgweights', 'match_dist', 'norm_dist', 'norm_x1', 'norm_x2',
            >>>     'norm_y1', 'norm_y2', 'ratio_score', 'scale1', 'scale2',
            >>>     'sver_err_ori', 'sver_err_scale', 'sver_err_xy',
            >>>     'weighted_norm_dist', 'weighted_ratio_score']
            >>> pairfeat_cfg = {
            >>>     'bin_key': 'ratio',
            >>>     'bins': [0.6, 0.7, 0.8],
            >>>     'indices': [],
            >>>     'local_keys': local_keys,
            >>>     'sorters': [],
            >>>     'summary_ops': ['len', 'mean', 'sum']
            >>> }
            >>> global_keys = ['gps', 'qual', 'time', 'view']
            >>> extr = PairwiseFeatureExtractor(ibs, match_config=match_config,
            >>>                                 pairfeat_cfg=pairfeat_cfg,
            >>>                                 global_keys=global_keys)
            >>> multi_index = True
            >>> edges = [(1, 2), (2, 3)]
            >>> matches, X = extr._make_pairwise_features(edges)
            >>> featinfo = vt.AnnotPairFeatInfo(X.columns)
            >>> print(featinfo.get_infostr())
            >>> match = matches[0]
            >>> glob_X = match._make_global_feature_vector(global_keys)
            >>> assert len(glob_X) == 19
        """
        edges = ut.lmap(tuple, ut.aslist(edges))
        if len(edges) == 0:
            return [], []

        matches = extr._enriched_pairwise_matches(edges)
        # ---------------
        # Try different feature constructions
        print('building pairwise features')
        pairfeat_cfg = extr.pairfeat_cfg
        pairfeat_cfg['summary_ops'] = set(pairfeat_cfg['summary_ops'])
        X = pd.DataFrame([
            m.make_feature_vector(**pairfeat_cfg)
            for m in ut.ProgIter(matches, label='making pairwise feats')
        ])
        multi_index = True
        if multi_index:
            # Index features by edges
            uv_index = nxu.ensure_multi_index(edges, ('aid1', 'aid2'))
            X.index = uv_index
        X[pd.isnull(X)] = np.nan
        X[np.isinf(X)] = np.nan
        # Re-order column names to ensure dimensions are consistent
        X = X.reindex_axis(sorted(X.columns), axis=1)

        # hack to fix feature validity
        if 'global(speed)' in X.columns and np.any(np.isinf(X['global(speed)'])):
            flags = np.isinf(X['global(speed)'])
            numer = X.loc[flags, 'global(gps_delta)']
            denom = X.loc[flags, 'global(time_delta)']
            newvals = np.full(len(numer), np.nan)
            newvals[(numer == 0) & (denom == 0)] = 0
            X.loc[flags, 'global(speed)'] = newvals

        aid_pairs_ = [(m.annot1['aid'], m.annot2['aid']) for m in matches]
        assert aid_pairs_ == edges, 'edge ordering changed'

        return matches, X

    def _make_cfgstr(extr, edges):
        ibs = extr.ibs
        edge_uuids = ibs.unflat_map(ibs.get_annot_visual_uuids, edges)
        edge_hashid = ut.hashid_arr(edge_uuids, 'edges')

        _cfg_lbl = ut.partial(ut.repr2, si=True, itemsep='', kvsep=':')
        match_configclass = ibs.depc_annot.configclass_dict['pairwise_match']

        cfgstr = '_'.join([
            edge_hashid,
            _cfg_lbl(extr.match_config),
            _cfg_lbl(extr.pairfeat_cfg),
            'global(' + _cfg_lbl(extr.global_keys) + ')',
            'pairwise_match_version=%r' % (match_configclass().version,)
        ])
        return cfgstr

    def _postprocess_feats(extr, feats):
        # Take the filtered subset of columns
        if extr.feat_dims is not None:
            missing = set(extr.feat_dims).difference(feats.columns)
            if any(missing):
                # print('We have: ' + ut.repr4(feats.columns))
                alt = feats.columns.difference(extr.feat_dims)
                mis_msg = ('Missing feature dims: ' + ut.repr4(missing))
                alt_msg = ('Did you mean? ' + ut.repr4(alt))
                print(mis_msg)
                print(alt_msg)
                raise KeyError(mis_msg)
            feats = feats[extr.feat_dims]
        return feats

    def _cached_pairwise_features(extr, edges):
        """
        Create pairwise features for annotations in a test inference object
        based on the features used to learn here

        TODO: need a more systematic way of specifying which feature dimensions
        need to be computed

        Notes:
            Given a edge (u, v), we need to:
            * Check which classifiers we have
            * Check which feat-cols the classifier needs,
               and construct a configuration that can acheive that.
                * Construct the chip/feat config
                * Construct the vsone config
                * Additional LNBNN enriching config
                * Pairwise feature construction config
            * Then we can apply the feature to the classifier

        edges = [(1, 2)]
        """
        print('Requesting %d cached pairwise features' % len(edges))
        edges = list(edges)

        # TODO: use object properties
        if len(edges) == 0:
            assert extr.feat_dims is not None, 'no edges and unset feat dims'
            index = nxu.ensure_multi_index([], ('aid1', 'aid2'))
            feats = pd.DataFrame(columns=extr.feat_dims, index=index)
            return feats

        use_cache = not extr.need_lnbnn and len(edges) > 2

        cache_dir = join(extr.ibs.get_cachedir(), 'infr_bulk_cache')
        feat_cfgstr = extr._make_cfgstr(edges)
        feat_cacher = ub.Cacher('bulk_pairfeats_v3',
                                feat_cfgstr, enabled=use_cache,
                                dpath=cache_dir, verbose=extr.verbose > 3)
        if feat_cacher.enabled:
            ut.ensuredir(cache_dir)
        if feat_cacher.exists() and extr.verbose > 3:
            fpath = feat_cacher.get_fpath()
            print('Load match cache size: {}'.format(ut.get_file_nBytes_str(fpath)))
        data = feat_cacher.tryload()
        if data is None:
            data = extr._make_pairwise_features(edges)
            feat_cacher.save(data)
            if feat_cacher.enabled and extr.verbose > 3:
                fpath = feat_cacher.get_fpath()
                print('Save match cache size: {}'.format(ut.get_file_nBytes_str(fpath)))
        matches, feats = data
        feats = extr._postprocess_feats(feats)
        return feats


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.verif.pairfeat
        python -m ibeis.algo.verif.pairfeat --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
