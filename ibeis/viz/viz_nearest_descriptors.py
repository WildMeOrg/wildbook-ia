# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool
import utool as ut
from six.moves import range
import plottool as pt  # NOQA
from plottool import draw_func2 as df2
from plottool.viz_featrow import draw_feat_row
from ibeis.viz import viz_helpers as vh
(print, rrr, profile) = utool.inject2(__name__, '[viz_nndesc]', DEBUG=False)


def get_annotfeat_nn_index(ibs, qaid, qfx, qreq_=None):
    #raise NotImplementedError('this doesnt work anymore. Need to submit mc4 query with metadata on and then reextract the required params')
    #from . import match_chips3 as mc3
    #ibs._init_query_requestor()
    if qreq_ is None:
        daid_list = ibs.get_valid_aids()
        qreq_ = ibs.new_query_request([qaid], daid_list)
    qreq_.load_indexer()  # TODO: ensure lazy
    qfx2_vecs = ibs.get_annot_vecs(qaid)[qfx:(qfx + 1)]
    K = qreq_.qparams.K
    Knorm = qreq_.qparams.Knorm
    if ut.VERBOSE:
        print('Knorm = %r' % (Knorm,))
    qfx2_idx, qfx2_dist = qreq_.indexer.knn(qfx2_vecs, 10)
    qfx2_daid = qreq_.indexer.get_nn_aids(qfx2_idx)
    qfx2_dfx = qreq_.indexer.get_nn_featxs(qfx2_idx)
    return qfx2_daid, qfx2_dfx, qfx2_dist, K, Knorm


#@utool.indent_func('[show_neardesc]')
def show_nearest_descriptors(ibs, qaid, qfx, fnum=None, stride=5,
                             qreq_=None, **kwargs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        qaid (int):  query annotation id
        qfx (query feature index):
        fnum (int):  figure number
        stride (int):
        consecutive_distance_compare (bool):

    CommandLine:
        # Find a good match to inspect
        python -m ibeis.viz.interact.interact_matches --test-testdata_match_interact --show --db PZ_MTEST --qaid 3

        # Now inspect it
        python -m ibeis.viz.viz_nearest_descriptors --test-show_nearest_descriptors --show --db PZ_MTEST --qaid 3 --qfx 879
        python -m ibeis.viz.viz_nearest_descriptors --test-show_nearest_descriptors --show
        python -m ibeis.viz.viz_nearest_descriptors --test-show_nearest_descriptors --db PZ_MTEST --qaid 3 --qfx 879 --diskshow --save foo.png --dpi=256

    SeeAlso:
        plottool.viz_featrow
        ~/code/plottool/plottool/viz_featrow.py

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_nearest_descriptors import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> qreq_ = ibeis.testdata_qreq_()
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid = qreq_.qaids[0]
        >>> qfx = ut.get_argval('--qfx', type_=int, default=879)
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
    consecutive_distance_compare = True
    draw_chip     = kwargs.get('draw_chip', False)
    draw_desc     = kwargs.get('draw_desc', True)
    draw_warped   = kwargs.get('draw_warped', True)
    draw_unwarped = kwargs.get('draw_unwarped', True)
    #skip = kwargs.get('skip', True)
    # Plots the nearest neighbors of a given feature (qaid, qfx)
    if fnum is None:
        fnum = df2.next_fnum()
    try:
        # Flann NN query
        (qfx2_daid, qfx2_dfx, qfx2_dist, K, Knorm) = get_annotfeat_nn_index(ibs, qaid, qfx, qreq_=qreq_)

        # Adds metadata to a feature match
        def get_extract_tuple(aid, fx, k=-1):
            rchip = ibs.get_annot_chips(aid)
            kp    = ibs.get_annot_kpts(aid)[fx]
            sift  = ibs.get_annot_vecs(aid)[fx]
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
                if ut.get_argflag('--texknormplot') and  pt.is_texmode():
                    #info = 'Match:\n$k=%r$, $\\frac{||\\mathbf{d}_i - \\mathbf{d}_j||}{Z}=%.3f$' % (k, qfx2_dist[0, k])
                    info = '\\vspace{1cm}'
                    info += 'Match: $\\mathbf{d}_{j_%r}$\n$\\textrm{dist}=%.3f$' % (k, qfx2_dist[0, k])
                    info += '\n$s_{\\tt{LNBNN}}=%.3f$' % (qfx2_dist[0, K + Knorm - 1] - qfx2_dist[0, k])
                else:
                    info = 'Match:%s\nk=%r, dist=%.3f' % (id_str, k, qfx2_dist[0, k])
                    info += '\nLNBNN=%.3f' % (qfx2_dist[0, K + Knorm - 1] - qfx2_dist[0, k])
            elif k < Knorm + K:
                type_ = 'Norm'
                if ut.get_argflag('--texknormplot') and  pt.is_texmode():
                    #info = 'Norm: $j_%r$\ndist=%.3f' % (id_str, k, qfx2_dist[0, k])
                    info = '\\vspace{1cm}'
                    info += 'Norm: $j_%r$\n$\\textrm{dist}=%.3f$' % (k, qfx2_dist[0, k])
                    info += '\n\\_'
                else:
                    info = 'Norm: %s\n$k=%r$, dist=$%.3f$' % (id_str, k, qfx2_dist[0, k])
            else:
                raise Exception('[viz] problem k=%r')
            return (rchip, kp, sift, fx, aid, info, type_)

        extracted_list = []
        # Remember the query sift feature
        extracted_list.append(get_extract_tuple(qaid, qfx, -1))
        origsift = extracted_list[0][2]
        skipped = 0
        for k in range(K + Knorm):
            #if qfx2_daid[0, k] == qaid and qfx2_dfx[0, k] == qfx:
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
            draw_chip=draw_chip, draw_desc=draw_desc, draw_warped=draw_warped,
            draw_unwarped=draw_unwarped,
        )
        if ut.get_argflag('--texknormplot'):
            featrow_kw['ell_color'] = pt.ORANGE
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
            px = draw_feat_row(rchip, fx, kp, sift, _fnum, _nRows, px=px_,
                               prevsift=prevsift, origsift=origsift, aid=aid,
                               info=info, type_=type_, **featrow_kw)

            px += px_shift
            if prevsift is None or consecutive_distance_compare:
                prevsift = sift

        df2.adjust_subplots_safe(hspace=.85, wspace=0, top=.95, bottom=.087, left=.05, right=.95)

    except Exception as ex:
        print('[viz] Error in show nearest descriptors')
        print(ex)
        raise


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.viz_nearest_descriptors
        python -m ibeis.viz.viz_nearest_descriptors --allexamples
        python -m ibeis.viz.viz_nearest_descriptors --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
