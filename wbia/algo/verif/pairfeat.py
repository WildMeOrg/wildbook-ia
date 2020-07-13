# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import vtool as vt
import numpy as np
import ubelt as ub
import pandas as pd
from wbia import dtool as dt
from os.path import join
from wbia.algo.graph import nx_utils as nxu
from wbia.core_annots import ChipConfig

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
        ut.ParamInfo(
            'summary_ops',
            {
                # 'invsum',
                'sum',
                'std',
                'mean',
                'len',
                'med',
            },
        ),
        ut.ParamInfo('local_keys', None),
        ut.ParamInfo(
            'sorters',
            [
                # 'ratio', 'norm_dist', 'match_dist'
                # 'lnbnn', 'lnbnn_norm_dist',
            ],
        ),
        # ut.ParamInfo('bin_key', None, valid_values=[None, 'ratio']),
        ut.ParamInfo('bin_key', 'ratio', valid_values=[None, 'ratio']),
        # ut.ParamInfo('bins', [.5, .6, .7, .8])
        # ut.ParamInfo('bins', None, type_=eval),
        ut.ParamInfo('bins', (0.625,), type_=eval),
        # ut.ParamInfo('need_lnbnn', False),
        ut.ParamInfo(
            'use_na', False
        ),  # change to True if sklearn has RFs with nan support
    ]


class VsOneMatchConfig(dt.Config):
    _param_info_list = vt.matching.VSONE_DEFAULT_CONFIG


class VsOneFeatConfig(dt.Config):
    """ keypoint params """

    _param_info_list = vt.matching.VSONE_FEAT_CONFIG


class MatchConfig(dt.Config):
    _param_info_list = (
        vt.matching.VSONE_DEFAULT_CONFIG
        + vt.matching.VSONE_FEAT_CONFIG
        + ChipConfig._param_info_list
    )


class PairwiseFeatureExtractor(object):
    r"""
    Args:
        ibs (wbia.IBEISController): image analysis api
        match_config (dict): config for building feature correspondences
        pairfeat_cfg (dict): config for making the pairwise feat vec
        global_keys (list): global keys to use
        need_lnbnn (bool): use LNBNN for enrichment
        feat_dims (list): subset of feature dimensions (from pruning)
                          if None, then all dimensions are used
        use_cache (bool):  turns on disk based caching (default = True)
        verbose (int):  verbosity flag (default = 1)

    CommandLine:
        python -m wbia.algo.verif.pairfeat PairwiseFeatureExtractor

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.verif.pairfeat import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> extr = PairwiseFeatureExtractor(ibs)
        >>> edges = [(1, 2), (2, 3)]
        >>> X = extr.transform(edges)
        >>> featinfo = vt.AnnotPairFeatInfo(X.columns)
        >>> print(featinfo.get_infostr())
    """

    def __init__(
        extr,
        ibs=None,
        config={},
        use_cache=True,
        verbose=1,
        # Nested config props
        match_config=None,
        pairfeat_cfg=None,
        global_keys=None,
        need_lnbnn=None,
        feat_dims=None,
    ):

        extr.verbose = verbose
        extr.use_cache = use_cache
        extr.ibs = ibs

        # Configs for this are a bit foobar. Allow config to be a catch-all It
        # can either store params in nested or flat form
        config = config.copy()
        vars_ = vars()

        def _popconfig(key, default):
            """ ensures param is either specified in func args xor config """
            if key in config:
                if vars_.get(key, None) is not None:
                    raise ValueError('{} specified twice'.format(key))
                value = config.pop(key)
            else:
                # See if the local namespace has it
                value = vars_.get(key, None)
                if value is None:
                    value = default
            return value

        # These also sort-of belong to pair-feat config
        extr.global_keys = _popconfig('global_keys', [])
        extr.need_lnbnn = _popconfig('need_lnbnn', False)
        extr.feat_dims = _popconfig('feat_dims', None)

        extr.match_config = MatchConfig(**_popconfig('match_config', {}))
        extr.pairfeat_cfg = PairFeatureConfig(**_popconfig('pairfeat_cfg', {}))

        # Allow config to store flat versions of these params
        extr.match_config.pop_update(config)
        extr.pairfeat_cfg.pop_update(config)

        if len(config) > 0:
            raise ValueError('Unused config items: ' + ut.repr4(config))

    def transform(extr, edges):
        """
        Converts an annotation edge into their corresponding feature.
        By default this is a caching operation.
        """
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

        CommandLine:
            python -m wbia.algo.verif.pairfeat _exec_pairwise_match --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.verif.pairfeat import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb('testdb1')
            >>> match_config = dict(histeq=True)
            >>> extr = PairwiseFeatureExtractor(ibs, match_config=match_config)
            >>> edges = [(1, 2), (2, 3)]
            >>> prog_hook = None
            >>> match_list = extr._exec_pairwise_match(edges)
            >>> match1, match2 = match_list
            >>> assert match1.annot2 is match2.annot1
            >>> assert match1.annot1 is not match2.annot2
            >>> ut.quit_if_noshow()
            >>> match2.show()
            >>> ut.show_if_requested()
        """
        if extr.verbose:
            print('[extr] executing pairwise one-vs-one matching')
        ibs = extr.ibs
        match_config = extr.match_config
        edges = ut.lmap(tuple, ut.aslist(edges))
        qaids = ut.take_column(edges, 0)
        daids = ut.take_column(edges, 1)

        # The depcache does the pairwise matching procedure
        match_list = ibs.depc.get(
            'pairwise_match', (qaids, daids), 'match', config=match_config
        )

        # Hack: Postprocess matches to re-add wbia annotation info
        # in lazy-dict format
        from wbia import core_annots

        config = ut.hashdict(match_config)
        qannot_cfg = dannot_cfg = config
        preload = True
        configured_lazy_annots = core_annots.make_configured_annots(
            ibs, qaids, daids, qannot_cfg, dannot_cfg, preload=preload
        )
        for qaid, daid, match in zip(qaids, daids, match_list):
            match.annot1 = configured_lazy_annots[config][qaid]
            match.annot2 = configured_lazy_annots[config][daid]
            match.config = config
        return match_list

    def _enrich_matches_lnbnn(extr, matches, other_aids, other_nids, inplace=False):
        """
        Given a set of one-vs-one matches, searches for LNBNN normalizers in a
        larger database to enrich the matches with database-level
        distinctiveness.
        """
        from wbia.algo.hots import nn_weights

        raise NotImplementedError(
            'havent tested since the re-work. '
            'Need to ensure that things work correctly.'
        )
        ibs = extr.ibs
        cfgdict = {
            'can_match_samename': False,
            'can_match_sameimg': True,
            'K': 3,
            'Knorm': 3,
            'prescore_method': 'csum',
            'score_method': 'csum',
        }
        custom_nid_lookup = ut.dzip(other_aids, other_nids)
        aids = [m.annot2['aid'] for m in matches]
        qreq_ = ibs.new_query_request(
            aids,
            other_aids,
            cfgdict=cfgdict,
            custom_nid_lookup=custom_nid_lookup,
            verbose=extr.verbose >= 2,
        )

        qreq_.load_indexer()
        indexer = qreq_.indexer
        if not inplace:
            matches_ = [match.copy() for match in matches]
        else:
            matches_ = matches
        K = qreq_.qparams.K
        Knorm = qreq_.qparams.Knorm
        normalizer_rule = qreq_.qparams.normalizer_rule

        extr.print('Stacking vecs for batch lnbnn matching')
        offset_list = np.cumsum([0] + [match_.fm.shape[0] for match_ in matches_])
        stacked_vecs = np.vstack(
            [
                match_.matched_vecs2()
                for match_ in ut.ProgIter(matches_, label='stack matched vecs')
            ]
        )

        vecs = stacked_vecs
        num = K + Knorm
        idxs, dists = indexer.batch_knn(vecs, num, chunksize=8192, label='lnbnn scoring')

        idx_list = [idxs[left:right] for left, right in ut.itertwo(offset_list)]
        dist_list = [dists[left:right] for left, right in ut.itertwo(offset_list)]
        iter_ = zip(matches_, idx_list, dist_list)
        prog = ut.ProgIter(iter_, length=len(matches_), label='lnbnn scoring')
        for match_, neighb_idx, neighb_dist in prog:
            qaid = match_.annot2['aid']
            norm_k = nn_weights.get_normk(qreq_, qaid, neighb_idx, Knorm, normalizer_rule)
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
            >>> from wbia.algo.verif.pairfeat import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb('testdb1')
            >>> match_config = {
            >>>     'K': 1, 'Knorm': 3, 'affine_invariance': True,
            >>>     'augment_orientation': True, 'checks': 20, 'ratio_thresh': 0.8,
            >>>     'refine_method': 'homog', 'sv_on': True, 'sver_xy_thresh': 0.01,
            >>>     'symmetric': True, 'weight': 'fgweights'
            >>> }
            >>> global_keys = ['gps', 'qual', 'time']
            >>> extr = PairwiseFeatureExtractor(ibs, match_config=match_config,
            >>>                                 global_keys=global_keys)
            >>> assert extr.global_keys == global_keys
            >>> edges = [(1, 2), (2, 3)]
            >>> prog_hook = None
            >>> match_list = extr._enriched_pairwise_matches(edges)
            >>> match1, match2 = match_list
            >>> assert match1.annot2 is match2.annot1
            >>> assert match1.annot1 is not match2.annot2
            >>> print('match1.global_measures = {!r}'.format(match1.global_measures))
            >>> assert len(match1.global_measures) == 3, 'global measures'
        """
        # print('extr.global_keys = {!r}'.format(extr.global_keys))
        if extr.global_keys is None:
            raise ValueError('specify global keys')
            # global_keys = ['view_int', 'qual', 'gps', 'time']
            # global_keys = ['view', 'qual', 'gps', 'time']
        matches = extr._exec_pairwise_match(edges, prog_hook=prog_hook)
        if extr.need_lnbnn:
            extr._enrich_matches_lnbnn(matches, inplace=True)
        if extr.verbose:
            print('[extr] enriching match attributes')
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

        CommandLine:
            python -m wbia.algo.verif.pairfeat _make_pairwise_features

        Doctest:
            >>> from wbia.algo.verif.pairfeat import *
            >>> from wbia.algo.graph import demo
            >>> infr = demo.demodata_mtest_infr()
            >>> extr = PairwiseFeatureExtractor(ibs=infr.ibs)
            >>> match_config = {'K': 1, 'Knorm': 3, 'affine_invariance': True,
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
            >>>     'summary_ops': {'len', 'mean', 'sum'}
            >>> }
            >>> global_keys = ['gps', 'qual', 'time', 'view']
            >>> ibs = infr.ibs
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
        print('[extr] building pairwise features')
        pairfeat_cfg = extr.pairfeat_cfg.copy()
        use_na = pairfeat_cfg.pop('use_na')
        pairfeat_cfg['summary_ops'] = set(pairfeat_cfg['summary_ops'])
        X = pd.DataFrame(
            [
                m.make_feature_vector(**pairfeat_cfg)
                for m in ut.ProgIter(matches, label='making pairwise feats')
            ]
        )
        multi_index = True
        if multi_index:
            # Index features by edges
            uv_index = nxu.ensure_multi_index(edges, ('aid1', 'aid2'))
            X.index = uv_index
        X[pd.isnull(X)] = np.nan
        X[np.isinf(X)] = np.nan
        # Re-order column names to ensure dimensions are consistent
        X = X.reindex(sorted(X.columns), axis=1)

        # hack to fix feature validity
        if 'global(speed)' in X.columns:
            if np.any(np.isinf(X['global(speed)'])):
                flags = np.isinf(X['global(speed)'])
                numer = X.loc[flags, 'global(gps_delta)']
                denom = X.loc[flags, 'global(time_delta)']
                newvals = np.full(len(numer), np.nan)
                newvals[(numer == 0) & (denom == 0)] = 0
                X.loc[flags, 'global(speed)'] = newvals

        aid_pairs_ = [(m.annot1['aid'], m.annot2['aid']) for m in matches]
        assert aid_pairs_ == edges, 'edge ordering changed'

        if not use_na:
            # Fill nan values with very large values to workaround lack of nan
            # support in sklearn master.
            X[pd.isnull(X)] = (2 ** 30) - 1
        return matches, X

    def _make_cfgstr(extr, edges):
        ibs = extr.ibs
        edge_uuids = ibs.unflat_map(ibs.get_annot_visual_uuids, edges)
        edge_hashid = ut.hashid_arr(edge_uuids, 'edges')

        _cfg_lbl = ut.partial(ut.repr2, si=True, itemsep='', kvsep=':')
        match_configclass = ibs.depc_annot.configclass_dict['pairwise_match']

        cfgstr = '_'.join(
            [
                edge_hashid,
                _cfg_lbl(extr.match_config),
                _cfg_lbl(extr.pairfeat_cfg),
                'global(' + _cfg_lbl(extr.global_keys) + ')',
                'pairwise_match_version=%r' % (match_configclass().version,),
            ]
        )
        return cfgstr

    def _postprocess_feats(extr, feats):
        # Take the filtered subset of columns
        if extr.feat_dims is not None:
            missing = set(extr.feat_dims).difference(feats.columns)
            if any(missing):
                # print('We have: ' + ut.repr4(feats.columns))
                alt = feats.columns.difference(extr.feat_dims)
                mis_msg = 'Missing feature dims: ' + ut.repr4(missing)
                alt_msg = 'Did you mean? ' + ut.repr4(alt)
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
        edges = list(edges)
        if extr.verbose:
            print('[pairfeat] Requesting {} cached pairwise features'.format(len(edges)))

        # TODO: use object properties
        if len(edges) == 0:
            assert extr.feat_dims is not None, 'no edges and unset feat dims'
            index = nxu.ensure_multi_index([], ('aid1', 'aid2'))
            feats = pd.DataFrame(columns=extr.feat_dims, index=index)
            return feats
        else:
            use_cache = not extr.need_lnbnn and len(edges) > 2
            cache_dir = join(extr.ibs.get_cachedir(), 'infr_bulk_cache')
            feat_cfgstr = extr._make_cfgstr(edges)
            cacher = ub.Cacher(
                'bulk_pairfeats_v3',
                feat_cfgstr,
                enabled=use_cache,
                dpath=cache_dir,
                verbose=extr.verbose - 3,
            )

            # if cacher.exists() and extr.verbose > 3:
            #     fpath = cacher.get_fpath()
            #     print('Load match cache size: {}'.format(
            #         ut.get_file_nBytes_str(fpath)))

            data = cacher.tryload()
            if data is None:
                data = extr._make_pairwise_features(edges)
                cacher.save(data)

                # if cacher.enabled and extr.verbose > 3:
                #     fpath = cacher.get_fpath()
                #     print('Save match cache size: {}'.format(
                #         ut.get_file_nBytes_str(fpath)))

            matches, feats = data
            feats = extr._postprocess_feats(feats)
        return feats


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.verif.pairfeat
        python -m wbia.algo.verif.pairfeat --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
