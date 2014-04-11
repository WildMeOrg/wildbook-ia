from __future__ import absolute_import, division, print_function
import drawtool.draw_func2 as df2
import utool
from ibeis.model.hots import match_chips3 as mc3
from .viz_featrow import draw_feat_row
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_nndesc]', DEBUG=False)


@utool.indent_decor('[viz.show_near_desc]')
def show_nearest_descriptors(ibs, qcid, qfx, fnum=None, stride=5,
                             consecutive_distance_compare=False):
    # Plots the nearest neighbors of a given feature (qcid, qfx)
    if fnum is None:
        fnum = df2.next_fnum()
    # Find the nearest neighbors of a descriptor using mc3 and flann
    qreq = mc3.quickly_ensure_qreq(ibs)
    data_index = qreq._data_index
    if data_index is None:
        pass
    dx2_cid = data_index.ax2_cid
    dx2_fx = data_index.ax2_fx
    flann  = data_index.flann
    K      = ibs.qreq.cfg.nn_cfg.K
    Knorm  = ibs.qreq.cfg.nn_cfg.Knorm
    checks = ibs.qreq.cfg.nn_cfg.checks
    qfx2_desc = ibs.get_desc(qcid)[qfx:qfx + 1]

    try:
        # Flann NN query
        (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_desc, K + Knorm, checks=checks)
        qfx2_cid = dx2_cid[qfx2_dx]
        qfx2_fx = dx2_fx[qfx2_dx]

        # Adds metadata to a feature match
        def get_extract_tuple(cid, fx, k=-1):
            rchip = ibs.get_chips(cid)
            kp    = ibs.get_kpts(cid)[fx]
            sift  = ibs.get_desc(cid)[fx]
            if k == -1:
                info = '\nquery %s, fx=%r' % (ibs.cidstr(cid), fx)
                type_ = 'query'
            elif k < K:
                type_ = 'match'
                info = '\nmatch %s, fx=%r k=%r, dist=%r' % (ibs.cidstr(cid), fx, k, qfx2_dist[0, k])
            elif k < Knorm + K:
                type_ = 'norm'
                info = '\nnorm  %s, fx=%r k=%r, dist=%r' % (ibs.cidstr(cid), fx, k, qfx2_dist[0, k])
            else:
                raise Exception('[viz] problem k=%r')
            return (rchip, kp, sift, fx, cid, info, type_)

        extracted_list = []
        extracted_list.append(get_extract_tuple(qcid, qfx, -1))
        skipped = 0
        for k in xrange(K + Knorm):
            if qfx2_cid[0, k] == qcid and qfx2_fx[0, k] == qfx:
                skipped += 1
                continue
            tup = get_extract_tuple(qfx2_cid[0, k], qfx2_fx[0, k], k)
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
            (rchip, kp, sift, fx, cid, info, type_) = tup
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
                               prevsift=prevsift, cid=cid, info=info, type_=type_) + px_shift
            if prevsift is None or consecutive_distance_compare:
                prevsift = sift

        df2.adjust_subplots_safe(hspace=1)

    except Exception as ex:
        print('[viz] Error in show nearest descriptors')
        print(ex)
        raise
