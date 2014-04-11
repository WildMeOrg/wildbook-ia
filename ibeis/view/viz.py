from __future__ import absolute_import, division, print_function
import warnings
# Scientific
import numpy as np
# UTool
import utool
# VTool
import vtool.image as gtool
from vtool import keypoint as ktool
# Drawtool
import drawtool.draw_func2 as df2
# IBEIS
from ibeis.dev import params
from ibeis.model.jon_recognition import match_chips3 as mc3
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz]', DEBUG=False)

#from interaction import interact_keypoints, interact_chipres, interact_chip # NOQA
from . import viz_helpers as vh
from .viz_helpers import draw  # NOQA
from .viz_image import show_image  # NOQA
from .viz_chip import show_chip, show_keypoints  # NOQA
from .viz_matches import show_qres, show_chipres, annotate_chipres  # NOQA

FNUMS = dict(image=1, chip=2, res=3, inspect=4, special=5, name=6)

IN_IMAGE_OVERRIDE = utool.get_arg('--in-image-override', type_=bool, default=None)
SHOW_QUERY_OVERRIDE = utool.get_arg('--show-query-override', type_=bool, default=None)
NO_LABEL_OVERRIDE = utool.get_arg('--no-label-override', type_=bool, default=None)

SIFT_OR_VECFIELD = utool.get_arg('--vecfield', type_=bool)


def register_FNUMS(FNUMS_):
    global FNUMS
    FNUMS = FNUMS_


#=============
# Splash Viz
#=============


def show_splash(fnum=1, **kwargs):
    #printDBG('[viz] show_splash()')
    splash_fpath = 'splash.png'
    img = gtool.imread(splash_fpath)
    df2.imshow(img, fnum=fnum, **kwargs)

#=============
# Name Viz
#=============


def show_name_of(ibs, cid, **kwargs):
    nid = ibs.get_chip_names(cid)
    show_name(ibs, nid, sel_cids=[cid], **kwargs)


def show_name(ibs, nid, nid2_cids=None, fnum=0, sel_cids=[], subtitle='',
              annote=False, **kwargs):
    print('[viz] show_name nid=%r' % nid)
    nid2_name = ibs.tables.nid2_name
    cid2_nid   = ibs.tables.cid2_nid
    name = nid2_name[nid]
    if not nid2_cids is None:
        cids = nid2_cids[nid]
    else:
        cids = np.where(cid2_nid == nid)[0]
    print('[viz] show_name %r' % ibs.cidstr(cids))
    nRows, nCols = vh.get_square_row_cols(len(cids))
    print('[viz*] r=%r, c=%r' % (nRows, nCols))
    #gs2 = gridspec.GridSpec(nRows, nCols)
    pnum = lambda px: (nRows, nCols, px + 1)
    fig = df2.figure(fnum=fnum, pnum=pnum(0), **kwargs)
    fig.clf()
    # Trigger computation of all chips in parallel
    ibs.refresh_features(cids)
    for px, cid in enumerate(cids):
        show_chip(ibs, cid=cid, pnum=pnum(px), draw_ell=annote, kpts_alpha=.2)
        if cid in sel_cids:
            ax = df2.gca()
            df2.draw_border(ax, df2.GREEN, 4)
        #plot_cid3(ibs, cid)
    if isinstance(nid, np.ndarray):
        nid = nid[0]
    if isinstance(name, np.ndarray):
        name = name[0]

    figtitle = 'Name View nid=%r name=%r' % (nid, name)
    df2.set_figtitle(figtitle)
    #if not annote:
        #title += ' noannote'
    #gs2.tight_layout(fig)
    #gs2.update(top=df2.TOP_SUBPLOT_ADJUST)
    #df2.set_figtitle(title, subtitle)


#==========================
# Image Viz
#==========================

#==========================
# Chip Viz
#==========================

#==========================
# ChipRes Viz
#==========================

#==========================
# Result Viz
#==========================


@utool.indent_func
@profile
def show_top(res, ibs, *args, **kwargs):
    topN_cids = res.topN_cids(ibs)
    N = len(topN_cids)
    cidstr = ibs.cidstr(res.qcid)
    figtitle = kwargs.pop('figtitle', 'q%s -- TOP %r' % (cidstr, N))
    return _show_res(ibs, res, topN_cids=topN_cids, figtitle=figtitle,
                     draw_kpts=False, draw_ell=False,
                     all_kpts=False, **kwargs)


@utool.indent_func
@profile
def res_show_analysis(res, ibs, **kwargs):
        print('[viz] res.show_analysis()')
        # Parse arguments
        noshow_gt  = kwargs.pop('noshow_gt', params.args.noshow_gt)
        show_query = kwargs.pop('show_query', params.args.noshow_query)
        cid_list    = kwargs.pop('cid_list', None)
        figtitle   = kwargs.pop('figtitle', None)

        # Debug printing
        #print('[viz.analysis] noshow_gt  = %r' % noshow_gt)
        #print('[viz.analysis] show_query = %r' % show_query)
        #print('[viz.analysis] cid_list    = %r' % cid_list)

        # Compare to cid_list instead of using top ranks
        if cid_list is None:
            print('[viz.analysis] showing topN cids')
            topN_cids = res.topN_cids(ibs)
            if figtitle is None:
                if len(topN_cids) == 0:
                    warnings.warn('len(topN_cids) == 0')
                    figtitle = 'WARNING: no top scores!' + ibs.cidstr(res.qcid)
                else:
                    topscore = res.get_cid2_score()[topN_cids][0]
                    figtitle = ('q%s -- topscore=%r' % (ibs.cidstr(res.qcid), topscore))
        else:
            print('[viz.analysis] showing a given list of cids')
            topN_cids = cid_list
            if figtitle is None:
                figtitle = 'comparing to ' + ibs.cidstr(topN_cids) + figtitle

        # Do we show the ground truth?
        def missed_cids():
            showgt_cids = ibs.get_other_indexed_cids(res.qcid)
            return np.setdiff1d(showgt_cids, topN_cids)
        showgt_cids = [] if noshow_gt else missed_cids()

        return _show_res(ibs, res, gt_cids=showgt_cids, topN_cids=topN_cids,
                         figtitle=figtitle, show_query=show_query, **kwargs)


@utool.indent_func
@profile
def _show_res(ibs, res, **kwargs):
    ''' Displays query chip, groundtruth matches, and top 5 matches'''
    #printDBG('[viz._show_res()] %s ' % utool.printableVal(locals()))
    #printDBG = print
    in_image = ibs.prefs.display_cfg.show_results_in_image
    annote     = kwargs.pop('annote', 2)  # this is toggled
    fnum       = kwargs.get('fnum', 3)
    figtitle   = kwargs.get('figtitle', '')
    aug        = kwargs.get('aug', '')
    topN_cids  = kwargs.get('topN_cids', [])
    gt_cids    = kwargs.get('gt_cids',   [])
    all_kpts   = kwargs.get('all_kpts', False)
    interact   = kwargs.get('interact', True)
    show_query = kwargs.get('show_query', False)
    dosquare   = kwargs.get('dosquare', False)
    if SHOW_QUERY_OVERRIDE is not None:
        show_query = SHOW_QUERY_OVERRIDE

    max_nCols = 5
    if len(topN_cids) in [6, 7]:
        max_nCols = 3
    if len(topN_cids) in [8]:
        max_nCols = 4

    printDBG('[viz]========================')
    printDBG('[viz._show_res()]----------------')
    all_gts = ibs.get_other_indexed_cids(res.qcid)
    _tup = tuple(map(len, (topN_cids, gt_cids, all_gts)))
    printDBG('[viz._show_res()] #topN=%r #missed_gts=%r/%r' % _tup)
    printDBG('[viz._show_res()] * fnum=%r' % (fnum,))
    printDBG('[viz._show_res()] * figtitle=%r' % (figtitle,))
    printDBG('[viz._show_res()] * max_nCols=%r' % (max_nCols,))
    printDBG('[viz._show_res()] * show_query=%r' % (show_query,))
    ranked_cids = res.topN_cids(ibs, N='all')
    # Build a subplot grid
    nQuerySubplts = 1 if show_query else 0
    nGtSubplts    = nQuerySubplts + (0 if gt_cids is None else len(gt_cids))
    nTopNSubplts  = 0 if topN_cids is None else len(topN_cids)
    nTopNCols     = min(max_nCols, nTopNSubplts)
    nGTCols       = min(max_nCols, nGtSubplts)
    nGTCols       = max(nGTCols, nTopNCols)
    nTopNCols     = nGTCols
    nGtRows       = 0 if nGTCols == 0 else int(np.ceil(nGtSubplts / nGTCols))
    nTopNRows     = 0 if nTopNCols == 0 else int(np.ceil(nTopNSubplts / nTopNCols))
    nGtCells      = nGtRows * nGTCols
    nRows         = nTopNRows + nGtRows

    # HACK:
    _color_list = df2.distinct_colors(len(topN_cids))
    cid2_color = {cid: _color_list[ox] for ox, cid in enumerate(topN_cids)}

    if IN_IMAGE_OVERRIDE is not None:
        in_image = IN_IMAGE_OVERRIDE

    # Helpers
    def _show_query_fn(plotx_shift, rowcols):
        'helper for viz._show_res'
        plotx = plotx_shift + 1
        pnum = (rowcols[0], rowcols[1], plotx)
        #print('[viz] Plotting Query: pnum=%r' % (pnum,))
        _kwshow = dict(draw_kpts=annote)
        _kwshow.update(kwargs)
        _kwshow['prefix'] = 'q'
        _kwshow['res'] = res
        _kwshow['pnum'] = pnum
        _kwshow['cid2_color'] = cid2_color
        _kwshow['draw_ell'] = annote >= 1
        #_kwshow['in_image'] = in_image
        show_chip(ibs, **_kwshow)
        #if in_image:
            #roi1 = ibs.cid2_roi(res.qcid)
            #df2.draw_roi(roi1, bbox_color=df2.ORANGE, label='q' + ibs.cidstr(res.qcid))

    def _plot_matches_cids(cid_list, plotx_shift, rowcols):

        def _show_matches_fn(cid, orank, pnum):
            'Helper function for drawing matches to one cid'
            aug = 'rank=%r\n' % orank
            #printDBG('[viz._show_res()] plotting: %r'  % (pnum,))
            _kwshow  = dict(draw_ell=annote, draw_pts=False, draw_lines=annote,
                            ell_alpha=.5, all_kpts=all_kpts)
            _kwshow.update(kwargs)
            _kwshow['fnum'] = fnum
            _kwshow['pnum'] = pnum
            _kwshow['title_aug'] = aug
            # If we already are showing the query dont show it here
            if not show_query:
                _kwshow['draw_ell'] = annote == 1
                _kwshow['draw_lines'] = annote >= 1
                show_chipres(res, ibs, cid, in_image=in_image, **_kwshow)
            else:
                _kwshow['draw_ell'] = annote >= 1
                if annote == 2:
                    # TODO Find a better name
                    _kwshow['color'] = cid2_color[cid]
                    _kwshow['sel_fx2'] = res.cid2_fm[cid][:, 1]
                show_chip(ibs, cid, in_image=in_image, **_kwshow)
                annotate_chipres(ibs, res, cid, in_image=in_image, show_query=not show_query)

        'helper for viz._show_res to draw many cids'
        #printDBG('[viz._show_res()] Plotting Chips %s:' % ibs.cidstr(cid_list))
        if cid_list is None:
            return
        # Do lazy load before show_chipres
        ibs.get_chips(cid_list)
        ibs.get_kpts(cid_list)
        for ox, cid in enumerate(cid_list):
            plotx = ox + plotx_shift + 1
            pnum = (rowcols[0], rowcols[1], plotx)
            oranks = np.where(ranked_cids == cid)[0]
            if len(oranks) == 0:
                orank = -1
                continue
            orank = oranks[0] + 1
            _show_matches_fn(cid, orank, pnum)

    if dosquare:
        # HACK
        nSubplots = nGtSubplts + nTopNSubplts
        nRows, nCols = vh.get_square_row_cols(nSubplots, 3)
        nTopNCols = nGTCols = nCols
        shift_topN = 1
        printDBG('nRows, nCols = (%r, %r)' % (nRows, nCols))
    else:
        shift_topN = nGtCells

    if nGtSubplts == 1:
        nGTCols = 1

    fig = df2.figure(fnum=fnum, pnum=(nRows, nGTCols, 1), docla=True, doclf=True)
    df2.disconnect_callback(fig, 'button_press_event')
    df2.plt.subplot(nRows, nGTCols, 1)
    # Plot Query
    if show_query:
        _show_query_fn(0, (nRows, nGTCols))
    # Plot Ground Truth
    _plot_matches_cids(gt_cids, nQuerySubplts, (nRows, nGTCols))
    _plot_matches_cids(topN_cids, shift_topN, (nRows, nTopNCols))
    #figtitle += ' q%s name=%s' % (ibs.cidstr(res.qcid), ibs.cid2_name(res.qcid))
    figtitle += aug
    df2.set_figtitle(figtitle, incanvas=not NO_LABEL_OVERRIDE)

    # Result Interaction
    if interact:
        printDBG('[viz._show_res()] starting interaction')

        def _ctrlclicked_cid(cid):
            printDBG('ctrl+clicked cid=%r' % cid)
            fnum = FNUMS['special']
            fig = df2.figure(fnum=fnum, docla=True, doclf=True)
            df2.disconnect_callback(fig, 'button_press_event')
            viz_spatial_verification(ibs, res.qcid, cid2=cid, fnum=fnum)
            fig.canvas.draw()
            df2.bring_to_front(fig)

        def _clicked_cid(cid):
            printDBG('clicked cid=%r' % cid)
            fnum = FNUMS['inspect']
            res.interact_chipres(ibs, cid, fnum=fnum)
            fig = df2.gcf()
            fig.canvas.draw()
            df2.bring_to_front(fig)

        def _clicked_none():
            # Toggle if the click is not in any axis
            printDBG('clicked none')
            #print(kwargs)
            _show_res(ibs, res, annote=(annote + 1) % 3, **kwargs)
            fig.canvas.draw()

        def _on_res_click(event):
            'result interaction mpl event callback slot'
            print('[viz] clicked result')
            if event.xdata is None or event.inaxes is None:
                #print('clicked outside axes')
                return _clicked_none()
            ax = event.inaxes
            hs_viewtype = ax.__dict__.get('_hs_viewtype', '')
            printDBG(event.__dict__)
            printDBG('hs_viewtype=%r' % hs_viewtype)
            # Clicked a specific chipres
            if hs_viewtype.find('chipres') == 0:
                cid = ax.__dict__.get('_hs_cid')
                # Ctrl-Click
                key = '' if event.key is None else event.key
                print('key = %r' % key)
                if key.find('control') == 0:
                    print('[viz] result control clicked')
                    return _ctrlclicked_cid(cid)
                # Left-Click
                else:
                    print('[viz] result clicked')
                    return _clicked_cid(cid)

        df2.connect_callback(fig, 'button_press_event', _on_res_click)
    df2.adjust_subplots_safe()
    printDBG('[viz._show_res()] Finished')
    return fig

#==========================#
#  --- TESTING FUNCS ---   #
#==========================#


def show_keypoint_gradient_orientations(ibs, cid, fx, fnum=None, pnum=None):
    # Draw the gradient vectors of a patch overlaying the keypoint
    if fnum is None:
        fnum = df2.next_fnum()
    rchip = ibs.get_chips(cid)
    kp = ibs.get_kpts(cid)[fx]
    sift = ibs.get_desc(cid)[fx]
    df2.draw_keypoint_gradient_orientations(rchip, kp, sift=sift,
                                            mode='vec', fnum=fnum, pnum=pnum)
    df2.set_title('Gradient orientation\n %s, fx=%d' % (ibs.cidstr(cid), fx))


def kp_info(kp):
    kpts = np.array([kp])
    xy_str    = ktool.get_xy_strs(kpts)[0]
    shape_str = ktool.get_shape_strs(kpts)[0]
    ori_ = ktool.get_oris(kpts)[0]
    ori_str = 'ori=%.2f' % ori_
    scale = ktool.get_scales(kpts)[0]
    return xy_str, shape_str, scale, ori_str


@utool.indent_decor('[viz.draw_feat_row]')
def draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, prevsift=None,
                  cid=None, info='', type_=None):
    pnum_ = lambda px: (nRows, nCols, px)

    def _draw_patch(**kwargs):
        return df2.draw_keypoint_patch(rchip, kp, sift, ori_color=df2.DEEP_PINK, **kwargs)

    # Feature strings
    xy_str, shape_str, scale, ori_str = kp_info(kp)

    # Draw the unwarped selected feature
    ax = _draw_patch(fnum=fnum, pnum=pnum_(px + 1))
    ax._hs_viewtype = 'unwarped'
    ax._hs_cid = cid
    ax._hs_fx = fx
    unwarped_lbl = 'affine feature invV =\n' + shape_str + '\n' + ori_str
    df2.set_xlabel(unwarped_lbl, ax)

    # Draw the warped selected feature
    ax = _draw_patch(fnum=fnum, pnum=pnum_(px + 2), warped=True)
    ax._hs_viewtype = 'warped'
    ax._hs_cid = cid
    ax._hs_fx = fx
    warped_lbl = ('warped feature\n' +
                  'fx=%r scale=%.1f\n' +
                  '%s' + info) % (fx, scale, xy_str)
    df2.set_xlabel(warped_lbl, ax)

    border_color = {None: None,
                    'query': None,
                    'match': df2.BLUE,
                    'norm': df2.ORANGE}[type_]
    if border_color is not None:
        df2.draw_border(ax, color=border_color)

    # Draw the SIFT representation
    pnum = pnum_(px + 3)
    if SIFT_OR_VECFIELD:
        df2.figure(fnum=fnum, pnum=pnum)
        df2.draw_keypoint_gradient_orientations(rchip, kp, sift=sift)
    else:
        sigtitle = '' if px != 3 else 'sift histogram'
        ax = df2.plot_sift_signature(sift, sigtitle, fnum=fnum, pnum=pnum)
        ax._hs_viewtype = 'histogram'
        if prevsift is not None:
            #dist_list = ['L1', 'L2', 'hist_isect', 'emd']
            #dist_list = ['L2', 'hist_isect']
            dist_list = ['L2']
            distmap = utool.compute_distances(sift, prevsift, dist_list)
            dist_str = ', '.join(['(%s, %.2E)' % (key, val) for key, val in distmap.iteritems()])
            df2.set_xlabel(dist_str)
    return px + nCols

#----


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

from .viz_misc import viz_spatial_verification
