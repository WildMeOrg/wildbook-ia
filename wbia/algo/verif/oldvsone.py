# -*- coding: utf-8 -*-
import utool as ut
from wbia.algo.graph.state import POSTV, NEGTV, INCMP
import numpy as np


# @profile
# def edge_hashids(samples):
#     qvuuids = samples.annots1.visual_uuids
#     dvuuids = samples.annots2.visual_uuids
#     # edge_uuids = [ut.combine_uuids(uuids)
#     #                for uuids in zip(qvuuids, dvuuids)]
#     edge_hashids = [make_edge_hashid(uuid1, uuid2) for uuid1, uuid2 in zip(qvuuids, dvuuids)]
#     # edge_uuids = [combine_2uuids(uuid1, uuid2)
#     #                for uuid1, uuid2 in zip(qvuuids, dvuuids)]
#     return edge_hashids

# @profile
# def edge_hashid(samples):
#     edge_hashids = samples.edge_hashids()
#     edge_hashid = ut.hashstr_arr27(edge_hashids, 'edges', hashlen=32,
#                                    pathsafe=True)
#     return edge_hashid

# @profile
# def make_edge_hashid(uuid1, uuid2):
#     """
#     Slightly faster than using ut.combine_uuids, because we condense and don't
#     bother casting back to UUIDS
#     """
#     sep_str = '-'
#     sep_byte = six.b(sep_str)
#     pref = six.b('{}2'.format(sep_str))
#     combined_bytes = pref + sep_byte.join([uuid1.bytes, uuid2.bytes])
#     bytes_sha1 = hashlib.sha1(combined_bytes)
#     # Digest them into a hash
#     hashbytes_20 = bytes_sha1.digest()
#     hashbytes_16 = hashbytes_20[0:16]
#     # uuid_ = uuid.UUID(bytes=hashbytes_16)
#     return hashbytes_16


def demo_single_pairwise_feature_vector():
    r"""
    CommandLine:
        python -m wbia.algo.verif.vsone demo_single_pairwise_feature_vector

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.verif.vsone import *  # NOQA
        >>> match = demo_single_pairwise_feature_vector()
        >>> print(match)
    """
    import vtool as vt
    import wbia

    ibs = wbia.opendb('testdb1')
    qaid, daid = 1, 2
    annot1 = ibs.annots([qaid])[0]._make_lazy_dict()
    annot2 = ibs.annots([daid])[0]._make_lazy_dict()

    vt.matching.ensure_metadata_normxy(annot1)
    vt.matching.ensure_metadata_normxy(annot2)

    match = vt.PairwiseMatch(annot1, annot2)
    cfgdict = {'checks': 200, 'symmetric': False}
    match.assign(cfgdict=cfgdict)
    match.apply_ratio_test({'ratio_thresh': 0.638}, inplace=True)
    match.apply_sver(inplace=True)

    # match.add_global_measures(['yaw', 'qual', 'gps', 'time'])
    match.add_global_measures(['view', 'qual', 'gps', 'time'])
    match.add_local_measures()

    # sorters = ['ratio', 'norm_dist', 'match_dist']
    match.make_feature_vector()
    return match

    def demo_classes(pblm):
        r"""
        CommandLine:
            python -m wbia.algo.verif.vsone demo_classes --saveparts --save=classes.png --clipwhite

            python -m wbia.algo.verif.vsone demo_classes --saveparts --save=figures/classes.png --clipwhite --dpath=~/latex/crall-iccv-2017

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.verif.vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty(defaultdb='PZ_PB_RF_TRAIN')
            >>> pblm.load_features()
            >>> pblm.load_samples()
            >>> pblm.build_feature_subsets()
            >>> pblm.demo_classes()
            >>> ut.show_if_requested()
        """
        task_key = 'match_state'
        labels = pblm.samples.subtasks[task_key]
        pb_labels = pblm.samples.subtasks['photobomb_state']
        classname_offset = {
            POSTV: 0,
            NEGTV: 0,
            INCMP: 0,
        }
        class_name = POSTV
        class_name = NEGTV
        class_name = INCMP

        feats = pblm.samples.X_dict['learn(sum,glob)']

        offset = 0
        class_to_edge = {}
        for class_name in labels.class_names:
            print('Find example of %r' % (class_name,))
            # Find an example of each class (that is not a photobomb)
            pbflags = pb_labels.indicator_df['notpb']
            flags = labels.indicator_df[class_name]
            assert np.all(pbflags.index == flags.index)
            flags = flags & pbflags
            ratio = feats['sum(ratio)']
            if class_name == INCMP:
                # flags &= feats['global(delta_yaw)'] > 3
                flags &= feats['global(delta_view)'] > 2
                # flags &= feats['sum(ratio)'] > 0
            if class_name == NEGTV:
                low = ratio[flags].max()
                flags &= feats['sum(ratio)'] >= low
            if class_name == POSTV:
                low = ratio[flags].median() / 2
                high = ratio[flags].median()
                flags &= feats['sum(ratio)'] < high
                flags &= feats['sum(ratio)'] > low
            # flags &= pblm.samples.simple_scores[flags]['score_lnbnn_1vM'] > 0
            idxs = np.where(flags)[0]
            print('Found %d candidates' % (len(idxs)))
            offset = classname_offset[class_name]
            idx = idxs[offset]
            series = labels.indicator_df.iloc[idx]
            assert series[class_name]
            edge = series.name
            class_to_edge[class_name] = edge

        import wbia.plottool as pt
        import wbia.guitool as gt

        gt.ensure_qapp()
        pt.qtensure()

        fnum = 1
        pt.figure(fnum=fnum, pnum=(1, 3, 1))
        pnum_ = pt.make_pnum_nextgen(1, 3)

        # classname_alias = {
        #     POSTV: 'positive',
        #     NEGTV: 'negative',
        #     INCMP: 'incomparable',
        # }

        ibs = pblm.infr.ibs
        for class_name in class_to_edge.keys():
            edge = class_to_edge[class_name]
            aid1, aid2 = edge
            # alias = classname_alias[class_name]
            print('class_name = %r' % (class_name,))
            annot1 = ibs.annots([aid1])[0]._make_lazy_dict()
            annot2 = ibs.annots([aid2])[0]._make_lazy_dict()
            vt.matching.ensure_metadata_normxy(annot1)
            vt.matching.ensure_metadata_normxy(annot2)
            match = vt.PairwiseMatch(annot1, annot2)
            cfgdict = pblm.hyper_params.vsone_match.asdict()
            match.apply_all(cfgdict)
            pt.figure(fnum=fnum, pnum=pnum_())
            match.show(show_ell=False, show_ori=False)
            # pt.set_title(alias)

    def find_opt_ratio(pblm):
        """
        script to help find the correct value for the ratio threshold

            >>> from wbia.algo.verif.vsone import *  # NOQA
            >>> pblm = OneVsOneProblem.from_empty('PZ_PB_RF_TRAIN')
            >>> pblm = OneVsOneProblem.from_empty('GZ_Master1')
        """
        # Find best ratio threshold
        pblm.load_samples()
        infr = pblm.infr
        edges = ut.emap(tuple, pblm.samples.aid_pairs.tolist())
        task = pblm.samples['match_state']
        pos_idx = task.class_names.tolist().index(POSTV)

        config = {'ratio_thresh': 1.0, 'sv_on': False}
        matches = infr._exec_pairwise_match(edges, config)

        import wbia.plottool as pt
        import sklearn.metrics

        pt.qtensure()
        thresholds = np.linspace(0, 1.0, 100)
        pos_truth = task.y_bin.T[pos_idx]
        ratio_fs = [m.local_measures['ratio'] for m in matches]

        aucs = []
        # Given the current correspondences: Find the optimal
        # correspondence threshold.
        for thresh in ut.ProgIter(thresholds, 'computing thresh'):
            scores = np.array([fs[fs < thresh].sum() for fs in ratio_fs])
            roc = sklearn.metrics.roc_auc_score(pos_truth, scores)
            aucs.append(roc)
        aucs = np.array(aucs)
        opt_auc = aucs.max()
        opt_thresh = thresholds[aucs.argmax()]

        if True:
            pt.plt.plot(thresholds, aucs, 'r-', label='')
            pt.plt.plot(opt_thresh, opt_auc, 'ro', label='L opt=%r' % (opt_thresh,))
            pt.set_ylabel('auc')
            pt.set_xlabel('ratio threshold')
            pt.legend()

        # colors = {
        #     1: 'r',
        #     2: 'b',
        #     3: 'g',
        # }
        # def predict_truth(ratio_fs, opt_thresh, pos_truth):
        #     # Filter correspondence using thresh then sum their scores
        #     new_ratio_fs = [fs < opt_thresh for fs in ratio_fs]
        #     scores = np.array([fs.sum() for fs in new_ratio_fs])
        #     # Find the point (summed score threshold) that maximizes MCC
        #     fpr, tpr, points = sklearn.metrics.roc_curve(pos_truth, scores)
        #     mccs = np.array([sklearn.metrics.matthews_corrcoef(
        #         pos_truth, scores > point) for point in points])
        #     opt_point = points[mccs.argmax()]
        #     pos_pred = scores > opt_point
        #     return pos_pred
        # thresholds = np.linspace(0, 1.0, 100)
        # pos_truth = task.y_bin.T[pos_idx]
        # ratio_fs = [m.local_measures['ratio'] for m in matches]
        # thresh_levels = []
        # for level in range(1, 3 + 1):
        #     if ut.allsame(pos_truth):
        #         print('breaking')
        #         break
        #     print('level = %r' % (level,))
        #     aucs = []
        #     # Given the current correspondences: Find the optimal
        #     # correspondence threshold.
        #     for thresh in ut.ProgIter(thresholds, 'computing thresh'):
        #         scores = np.array([fs[fs < thresh].sum() for fs in ratio_fs])
        #         roc = sklearn.metrics.roc_auc_score(pos_truth, scores)
        #         aucs.append(roc)
        #     aucs = np.array(aucs)
        #     opt_auc = aucs.max()
        #     opt_thresh = thresholds[aucs.argmax()]
        #     thresh_levels.append(opt_thresh)

        #     if True:
        #         color = colors[level]
        #         pt.plt.plot(thresholds, aucs, color + '-', label='L%d' % level)
        #         pt.plt.plot(opt_thresh, opt_auc, color + 'o',
        #                     label='L%d opt=%r' % (level, opt_thresh,))

        #     # Remove the positive samples that this threshold fails on
        #     pred = predict_truth(ratio_fs, opt_thresh, pos_truth)
        #     flags = pred != pos_truth | ~pos_truth

        #     ratio_fs = ut.compress(ratio_fs, flags)
        #     pos_truth = pos_truth.compress(flags)

        # submax_thresh, submax_roc = vt.argsubmax(aucs, thresholds)

        # Now find all pairs that would be correctly classified using this
        # threshold

        # ratio_fs = thresh_ratio_fs
        # rocs = []
        # for thresh in ut.ProgIter(thresholds, 'computing thresh'):
        #     scores = np.array([fs[fs < thresh].sum() for fs in ratio_fs])
        #     roc = sklearn.metrics.roc_auc_score(pos_truth, scores)
        #     rocs.append(roc)
        # submax_thresh, submax_roc = vt.argsubmax(rocs, thresholds)
        # pt.plt.plot(thresholds, rocs, 'b-', label='L2')
        # pt.plt.plot(submax_thresh, submax_roc, 'bo', label='L2 opt=%r' % (submax_thresh,))

    # def simple_confusion(pblm, score_key=None, task_key=None,
    #                      target_class=None):
    #     if score_key is None:
    #         score_key = 'score_lnbnn_1vM'
    #     if task_key is None:
    #         task_key = pblm.primary_task_key
    #     task = pblm.samples[task_key]
    #     if target_class is None:
    #         target_class = task.default_class_name

    #     target_class_idx = task.lookup_class_idx(target_class)
    #     scores = pblm.samples.simple_scores[score_key]
    #     y = task.y_bin.T[target_class_idx]
    #     conf = vt.ConfusionMetrics().fit(scores, y)
    #     conf.label = score_key
    #     return conf
