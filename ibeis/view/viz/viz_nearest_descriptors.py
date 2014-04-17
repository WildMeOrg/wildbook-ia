from __future__ import absolute_import, division, print_function
import drawtool.draw_func2 as df2
import utool
from ibeis.model.hots import match_chips3 as mc3
from .viz_featrow import draw_feat_row
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_nndesc]', DEBUG=False)


@utool.indent_decor('[viz.show_near_desc]')
def show_nearest_descriptors(ibs, qrid, qfx, fnum=None, stride=5,
                             consecutive_distance_compare=False):
    # Plots the nearest neighbors of a given feature (qrid, qfx)
    if fnum is None:
        fnum = df2.next_fnum()
    # Find the nearest neighbors of a descriptor using mc3 and flann
    ibs._init_query_requestor()
    qreq = mc3.quickly_ensure_qreq(ibs)
    data_index = qreq._data_index
    if data_index is None:
        pass
    dx2_rid = data_index.ax2_rid
    dx2_fx  = data_index.ax2_fx
    K       = ibs.qreq.cfg.nn_cfg.K
    Knorm   = ibs.qreq.cfg.nn_cfg.Knorm
    qfx2_desc = ibs.get_desc(qrid)[qfx:(qfx + 1)]

    try:
        # Flann NN query
        (qfx2_dx, qfx2_dist) = data_index.nn_index(ibs, qfx2_desc, num=(K + Knorm))
        qfx2_rid = dx2_rid[qfx2_dx]
        qfx2_fx = dx2_fx[qfx2_dx]

        # Adds metadata to a feature match
        def get_extract_tuple(rid, fx, k=-1):
            rchip = ibs.get_chips(rid)
            kp    = ibs.get_kpts(rid)[fx]
            sift  = ibs.get_desc(rid)[fx]
            if k == -1:
                info = '\nquery %s, fx=%r' % (ibs.ridstr(rid), fx)
                type_ = 'query'
            elif k < K:
                type_ = 'match'
                info = '\nmatch %s, fx=%r k=%r, dist=%r' % (ibs.ridstr(rid), fx, k, qfx2_dist[0, k])
            elif k < Knorm + K:
                type_ = 'norm'
                info = '\nnorm  %s, fx=%r k=%r, dist=%r' % (ibs.ridstr(rid), fx, k, qfx2_dist[0, k])
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
