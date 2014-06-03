from __future__ import absolute_import, division, print_function
import utool
from plottool import draw_func2 as df2
from plottool.viz_featrow import draw_feat_row
from . import viz_helpers as vh
from ibeis.model.hots import query_helpers
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_nndesc]', DEBUG=False)


@utool.indent_func('[show_neardesc]')
def show_nearest_descriptors(ibs, qrid, qfx, fnum=None, stride=5,
                             consecutive_distance_compare=False):
    # Plots the nearest neighbors of a given feature (qrid, qfx)
    if fnum is None:
        fnum = df2.next_fnum()
    try:
        # Flann NN query
        (qfx2_rid, qfx2_fx, qfx2_dist, K, Knorm) = query_helpers.get_roifeat_nn_index(ibs, qrid, qfx)

        # Adds metadata to a feature match
        def get_extract_tuple(rid, fx, k=-1):
            rchip = ibs.get_roi_chips(rid)
            kp    = ibs.get_roi_kpts(rid)[fx]
            sift  = ibs.get_roi_desc(rid)[fx]
            if k == -1:
                info = '\nquery %s, fx=%r' % (vh.get_ridstrs(rid), fx)
                type_ = 'query'
            elif k < K:
                type_ = 'match'
                info = '\nmatch %s, fx=%r k=%r, dist=%r' % (vh.get_ridstrs(rid), fx, k, qfx2_dist[0, k])
            elif k < Knorm + K:
                type_ = 'norm'
                info = '\nnorm  %s, fx=%r k=%r, dist=%r' % (vh.get_ridstrs(rid), fx, k, qfx2_dist[0, k])
            else:
                raise Exception('[viz] problem k=%r')
            return (rchip, kp, sift, fx, rid, info, type_)

        extracted_list = []
        extracted_list.append(get_extract_tuple(qrid, qfx, -1))
        skipped = 0
        for k in xrange(K + Knorm):
            if qfx2_rid[0, k] == qrid and qfx2_fx[0, k] == qfx:
                skipped += 1
                continue
            tup = get_extract_tuple(qfx2_rid[0, k], qfx2_fx[0, k], k)
            extracted_list.append(tup)
        # Draw the _select_ith_match plot
        nRows, nCols = len(extracted_list), 3
        if stride is None:
            stride = nRows
        # Draw selected feature matches
        prevsift = None
        px = 0  # plot offset
        px_shift = 0  # plot stride shift
        nExtracted = len(extracted_list)
        for listx, tup in enumerate(extracted_list):
            (rchip, kp, sift, fx, rid, info, type_) = tup
            if listx % stride == 0:
                # Create a temporary nRows and fnum in case we are splitting
                # up nearest neighbors into separate figures with stride
                _fnum = fnum + listx
                _nRows = min(nExtracted - listx, stride)
                px_shift = px
                df2.figure(fnum=_fnum, docla=True, doclf=True)
            printDBG('[viz] ' + info.replace('\n', ''))
            px_ = px - px_shift
            px = draw_feat_row(rchip, fx, kp, sift, _fnum, _nRows, nCols, px_,
                               prevsift=prevsift, rid=rid, info=info, type_=type_) + px_shift
            if prevsift is None or consecutive_distance_compare:
                prevsift = sift

        df2.adjust_subplots_safe(hspace=1)

    except Exception as ex:
        print('[viz] Error in show nearest descriptors')
        print(ex)
        raise
