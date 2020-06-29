# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
from six.moves import range
from wbia.plottool import draw_func2 as df2
from wbia.plottool.viz_featrow import draw_feat_row
from wbia.viz import viz_helpers as vh
import wbia.plottool as pt  # NOQA
import six  # NOQA

(print, rrr, profile) = ut.inject2(__name__)


def get_annotfeat_nn_index(ibs, qaid, qfx, qreq_=None):
    # raise NotImplementedError('this doesnt work anymore. Need to submit mc4 query with metadata on and then reextract the required params')
    # from . import match_chips3 as mc3
    # ibs._init_query_requestor()
    if qreq_ is None:
        daid_list = ibs.get_valid_aids()
        qreq_ = ibs.new_query_request([qaid], daid_list)
    qreq_.load_indexer()  # TODO: ensure lazy

    # if isinstance(qfx, six.string_types):
    special = qfx == 'special'
    if special:
        qfx2_vecs = ibs.get_annot_vecs(qaid)
    else:
        qfx = int(qfx)
        qfx2_vecs = ibs.get_annot_vecs(qaid)[qfx : (qfx + 1)]
    K = qreq_.qparams.K
    Knorm = qreq_.qparams.Knorm
    if ut.VERBOSE:
        print('Knorm = %r' % (Knorm,))
    qfx2_idx, qfx2_dist = qreq_.indexer.knn(qfx2_vecs, 10)

    if special:
        import numpy as np

        # Find a query feature with "good" results
        qfx2_daid = qreq_.indexer.get_nn_aids(qfx2_idx)
        qfx2_dnid = ibs.get_annot_nids(qfx2_daid)
        nid = ibs.get_annot_nids(qaid)

        # slice_ = slice(None)
        slice_ = slice(0, K + Knorm)
        flags = qfx2_dnid.T[slice_].T == nid
        flags = np.logical_and(flags, qfx2_daid[:, slice_] != qaid)
        flags_first = flags[:, 0:K]
        flags_last = flags[:, K:]
        num_gt_matches = flags_first.sum(axis=1) - flags_last.sum(axis=1)
        print('num_gt_matches = %r' % (num_gt_matches,))
        print(num_gt_matches.max())
        has_good_num = num_gt_matches >= num_gt_matches.max() - 1
        candidate_qfxs = np.where(has_good_num)[0]

        cand_nids = qfx2_dnid[candidate_qfxs].T[slice_].T
        cand_flags = cand_nids == nid
        cand_dist = qfx2_dist[candidate_qfxs].T[slice_].T
        cand_dist_gt = cand_dist * cand_flags
        cand_dist_gf = cand_dist * ~cand_flags
        cand_score = cand_dist_gt.sum(axis=1) - cand_dist_gf.sum(axis=1)
        top_candxs = cand_score.argsort()
        print('cand_nids = %r' % (cand_nids,))
        print('top_candxs = %r' % (top_candxs,))

        cand_idx = top_candxs[1]
        # cand_idx = ut.take_percentile(top_candxs, .1)[-1]
        qfx = candidate_qfxs[cand_idx]
        print('qfx = %r' % (qfx,))
        qfx2_dist = qfx2_dist[qfx : (qfx + 1)]
        qfx2_idx = qfx2_idx[qfx : (qfx + 1)]

    qfx2_daid = qreq_.indexer.get_nn_aids(qfx2_idx)
    qfx2_dfx = qreq_.indexer.get_nn_featxs(qfx2_idx)
    return qfx, qfx2_daid, qfx2_dfx, qfx2_dist, K, Knorm


def show_top_featmatches(qreq_, cm_list):
    """
    Args:
        qreq_ (wbia.QueryRequest):  query request object with hyper-parameters
        cm_list (list):

    SeeAlso:
        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST -t best:lnbnn_on=True,lnbnn_normalizer=normlnbnn-test -a default --sephack

        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_Master1 -t best:lnbnn_on=True -a timectrl --sephack
        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST -t best:lnbnn_on=True -a default:size=30 --sephack
        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST -t best:K=1,Knorm=5,lnbnn_on=True -a default:size=30 --sephack
        python -m wbia --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST -t best:K=1,Knorm=3,lnbnn_on=True -a default --sephack


    CommandLine:
        python -m wbia.viz.viz_nearest_descriptors --exec-show_top_featmatches --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_nearest_descriptors import *  # NOQA
        >>> import wbia
        >>> cm_list, qreq_ = wbia.testdata_cmlist(defaultdb='PZ_MTEST',
        >>>                                        a=['default:has_none=mother,size=30'])
        >>> show_top_featmatches(qreq_, cm_list)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    # for cm in cm_list:
    #     cm.score_annot_csum(qreq_)
    import numpy as np
    import vtool as vt
    from functools import partial

    # Stack chipmatches
    ibs = qreq_.ibs
    infos = [cm.get_flat_fm_info() for cm in cm_list]
    flat_metadata = dict(
        [(k, np.concatenate(v)) for k, v in ut.dict_stack2(infos).items()]
    )
    fsv_flat = flat_metadata['fsv']
    flat_metadata['fs'] = fsv_flat.prod(axis=1)
    aids1 = flat_metadata['aid1'][:, None]
    aids2 = flat_metadata['aid2'][:, None]
    flat_metadata['aid_pairs'] = np.concatenate([aids1, aids2], axis=1)

    # Take sample of metadata
    sortx = flat_metadata['fs'].argsort()[::-1]
    num = len(cm_list) * 3
    # num = 10
    taker = partial(np.take, indices=sortx[:num], axis=0)
    flat_metadata_top = ut.map_dict_vals(taker, flat_metadata)
    aid1s, aid2s, fms = ut.dict_take(flat_metadata_top, ['aid1', 'aid2', 'fm'])

    annots = {}
    aids = np.unique(np.hstack((aid1s, aid2s)))
    annots = {aid: ibs.get_annot_lazy_dict(aid, config2_=qreq_.qparams) for aid in aids}

    label_lists = (
        ibs.get_match_truths(aid1s, aid2s) == ibs.const.EVIDENCE_DECISION.POSITIVE
    )
    patch_size = 64

    def extract_patches(annots, aid, fxs):
        """ custom_func(lazydict, key, subkeys) for multigroup_lookup """
        annot = annots[aid]
        kpts = annot['kpts']
        rchip = annot['rchip']
        kpts_m = kpts.take(fxs, axis=0)
        warped_patches, warped_subkpts = vt.get_warped_patches(
            rchip, kpts_m, patch_size=patch_size
        )
        return warped_patches

    data_lists = vt.multigroup_lookup(annots, [aid1s, aid2s], fms.T, extract_patches)

    import wbia.plottool as pt  # NOQA

    pt.ensureqt()
    import wbia_cnn

    inter = wbia_cnn.draw_results.interact_patches(
        label_lists,
        data_lists,
        flat_metadata_top,
        chunck_sizes=(2, 4),
        ibs=ibs,
        hack_one_per_aid=False,
        sortby='fs',
        qreq_=qreq_,
    )
    inter.show()


# @ut.indent_func('[show_neardesc]')
def show_nearest_descriptors(ibs, qaid, qfx, fnum=None, stride=5, qreq_=None, **kwargs):
    r"""
    Args:
        ibs (wbia.IBEISController): image analysis api
        qaid (int):  query annotation id
        qfx (int): query feature index
        fnum (int):  figure number
        stride (int):
        consecutive_distance_compare (bool):

    CommandLine:
        # Find a good match to inspect
        python -m wbia.viz.interact.interact_matches --test-testdata_match_interact --show --db PZ_MTEST --qaid 3

        # Now inspect it
        python -m wbia.viz.viz_nearest_descriptors --test-show_nearest_descriptors --show --db PZ_MTEST --qaid 3 --qfx 879
        python -m wbia.viz.viz_nearest_descriptors --test-show_nearest_descriptors --show
        python -m wbia.viz.viz_nearest_descriptors --test-show_nearest_descriptors --db PZ_MTEST --qaid 3 --qfx 879 --diskshow --save foo.png --dpi=256

    SeeAlso:
        plottool.viz_featrow
        ~/code/plottool/plottool/viz_featrow.py

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_nearest_descriptors import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> if True:
        >>>     import matplotlib as mpl
        >>>     from wbia.scripts.thesis import TMP_RC
        >>>     mpl.rcParams.update(TMP_RC)
        >>> qreq_ = wbia.testdata_qreq_()
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> qaid = qreq_.qaids[0]
        >>> qfx = ut.get_argval('--qfx', type_=None, default=879)
        >>> fnum = None
        >>> stride = 5
        >>> # execute function
        >>> skip = False
        >>> result = show_nearest_descriptors(ibs, qaid, qfx, fnum, stride,
        >>>                                   draw_chip=True,
        >>>                                   draw_warped=True,
        >>>                                   draw_unwarped=False,
        >>>                                   draw_desc=False, qreq_=qreq_)
        >>> # verify results
        >>> print(result)
        >>> pt.show_if_requested()
    """
    import wbia.plottool as pt  # NOQA

    consecutive_distance_compare = True
    draw_chip = kwargs.get('draw_chip', False)
    draw_desc = kwargs.get('draw_desc', True)
    draw_warped = kwargs.get('draw_warped', True)
    draw_unwarped = kwargs.get('draw_unwarped', True)
    # skip = kwargs.get('skip', True)
    # Plots the nearest neighbors of a given feature (qaid, qfx)
    if fnum is None:
        fnum = df2.next_fnum()
    try:
        # Flann NN query
        (qfx, qfx2_daid, qfx2_dfx, qfx2_dist, K, Knorm) = get_annotfeat_nn_index(
            ibs, qaid, qfx, qreq_=qreq_
        )

        # Adds metadata to a feature match
        def get_extract_tuple(aid, fx, k=-1):
            rchip = ibs.get_annot_chips(aid)
            kp = ibs.get_annot_kpts(aid)[fx]
            sift = ibs.get_annot_vecs(aid)[fx]
            if not ut.get_argflag('--texknormplot'):
                aidstr = vh.get_aidstrs(aid)
                nidstr = vh.get_nidstrs(ibs.get_annot_nids(aid))
                id_str = ' ' + aidstr + ' ' + nidstr + ' fx=%r' % (fx,)
            else:
                id_str = nidstr = aidstr = ''
            info = ''
            if k == -1:
                if pt.is_texmode():
                    info = '\\vspace{1cm}'
                    info += 'Query $\\mathbf{d}_i$'
                    info += '\n\\_'
                    info += '\n\\_'
                else:
                    if len(id_str) > '':
                        info = 'Query: %s' % (id_str,)
                    else:
                        info = 'Query'
                type_ = 'Query'
            elif k < K:
                type_ = 'Match'
                if ut.get_argflag('--texknormplot') and pt.is_texmode():
                    # info = 'Match:\n$k=%r$, $\\frac{||\\mathbf{d}_i - \\mathbf{d}_j||}{Z}=%.3f$' % (k, qfx2_dist[0, k])
                    info = '\\vspace{1cm}'
                    info += 'Match: $\\mathbf{d}_{j_%r}$\n$\\textrm{dist}=%.3f$' % (
                        k,
                        qfx2_dist[0, k],
                    )
                    # info += '\n$s_{\\tt{LNBNN}}=%.3f$' % (qfx2_dist[0, K + Knorm - 1] - qfx2_dist[0, k])
                    info += '\n$s=%.3f$' % (qfx2_dist[0, K + Knorm - 1] - qfx2_dist[0, k])
                else:
                    info = 'Match:%s\nk=%r, dist=%.3f' % (id_str, k, qfx2_dist[0, k])
                    info += '\nLNBNN=%.3f' % (
                        qfx2_dist[0, K + Knorm - 1] - qfx2_dist[0, k]
                    )
            elif k < Knorm + K:
                type_ = 'Norm'
                if ut.get_argflag('--texknormplot') and pt.is_texmode():
                    # info = 'Norm: $j_%r$\ndist=%.3f' % (id_str, k, qfx2_dist[0, k])
                    info = '\\vspace{1cm}'
                    info += 'Norm: $j_%r$\n$\\textrm{dist}=%.3f$' % (k, qfx2_dist[0, k])
                    info += '\n\\_'
                else:
                    info = 'Norm: %s\n$k=%r$, dist=$%.3f$' % (id_str, k, qfx2_dist[0, k],)
            else:
                raise Exception('[viz] problem k=%r')
            return (rchip, kp, sift, fx, aid, info, type_)

        extracted_list = []
        # Remember the query sift feature
        extracted_list.append(get_extract_tuple(qaid, qfx, -1))
        origsift = extracted_list[0][2]
        skipped = 0
        for k in range(K + Knorm):
            # if qfx2_daid[0, k] == qaid and qfx2_dfx[0, k] == qfx:
            if qfx2_daid[0, k] == qaid:
                skipped += 1
                continue
            tup = get_extract_tuple(qfx2_daid[0, k], qfx2_dfx[0, k], k)
            extracted_list.append(tup)
        # Draw the _select_ith_match plot
        nRows = len(extracted_list)
        if stride is None:
            stride = nRows
        # Draw selected feature matches
        prevsift = None
        px = 0  # plot offset
        px_shift = 0  # plot stride shift
        nExtracted = len(extracted_list)
        featrow_kw = dict(
            draw_chip=draw_chip,
            draw_desc=draw_desc,
            draw_warped=draw_warped,
            draw_unwarped=draw_unwarped,
        )
        if ut.get_argflag('--texknormplot'):
            featrow_kw['ell_color'] = pt.ORANGE
            featrow_kw['ell_linewidth'] = 1
            featrow_kw['arm1_lw'] = 0.5
            featrow_kw['stroke'] = 0
            pass
        for listx, tup in enumerate(extracted_list):
            (rchip, kp, sift, fx, aid, info, type_) = tup
            if listx % stride == 0:
                # Create a temporary nRows and fnum in case we are splitting
                # up nearest neighbors into separate figures with stride
                _fnum = fnum + listx
                _nRows = min(nExtracted - listx, stride)
                px_shift = px
                df2.figure(fnum=_fnum, docla=True, doclf=True)
            px_ = px - px_shift
            px = draw_feat_row(
                rchip,
                fx,
                kp,
                sift,
                _fnum,
                _nRows,
                px=px_,
                prevsift=prevsift,
                origsift=origsift,
                aid=aid,
                info=info,
                type_=type_,
                **featrow_kw,
            )

            px += px_shift
            if prevsift is None or consecutive_distance_compare:
                prevsift = sift

        # df2.adjust_subplots(hspace=.85, wspace=0, top=.95, bottom=.087, left=.05, right=.95)

    except Exception as ex:
        print('[viz] Error in show nearest descriptors')
        print(ex)
        raise


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.viz_nearest_descriptors
        python -m wbia.viz.viz_nearest_descriptors --allexamples
        python -m wbia.viz.viz_nearest_descriptors --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
