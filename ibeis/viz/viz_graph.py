# -*- coding: utf-8 -*-
"""
Displays the matching graph of individuals

WindowsDepends:
    pip install networkx
    wget http://www.graphviz.org/pub/graphviz/stable/windows/graphviz-2.38.msi
    graphviz-2.38.msi
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import plottool as pt
import vtool as vt
#import sys
#from os.path import join
try:
    import networkx as netx
    #if ut.WIN32:
    #    # Make sure graphviz is in the path
    #    win32_graphviz_bin_paths = [
    #        r'C:\Program Files (x86)\Graphviz2.38\bin',
    #        r'C:\Program Files\Graphviz2.38\bin',
    #    ]
    #    found_dot_exe = False
    #    for path in win32_graphviz_bin_paths:
    #        if ut.checkpath(path):
    #            if ut.checkpath(join(path, 'dot.exe')):
    #                sys.path.append(path)
    #                found_dot_exe = True
    #                break
    #    assert found_dot_exe, 'could not find graphviz'
except ImportError as ex:
    ut.printex(ex, 'Cannot import networkx', iswarning=True)
#import itertools


def show_chipmatch_graph(ibs, cm_list, qreq_, fnum=None, pnum=None, **kwargs):
    """

    CommandLine:
        python -m ibeis.viz.viz_graph --test-show_chipmatch_graph:0 --show
        python -m ibeis.viz.viz_graph --test-show_chipmatch_graph:1 --show
        python -m ibeis.viz.viz_graph --test-show_chipmatch_graph:1 --show --zoom=.15
        python -m ibeis.viz.viz_graph --test-show_chipmatch_graph:1 --zoom=.25 --save foo.jpg --diskshow --figsize=12,6 --dpath=. --dpi=280


    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> from ibeis.model.hots.chip_match import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[1, 2, 3, 4, 5, 6, 7, 18])
        >>> [cm.score_nsum(qreq_) for cm in cm_list]
        >>> ut.quit_if_noshow()
        >>> daid = cm.get_groundtruth_daids()[0]
        >>> zoom = ut.get_argval('--zoom', float, .4)
        >>> show_chipmatch_graph(ibs, cm_list, qreq_, zoom=zoom)
        >>> #cm.show_single_annotmatch(qreq_, daid)
        >>> ut.show_if_requested()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> from ibeis.model.hots.chip_match import *  # NOQA
        >>> #ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list='all')
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list='all')
        >>> [cm.score_nsum(qreq_) for cm in cm_list]
        >>> ut.quit_if_noshow()
        >>> daid = cm.get_groundtruth_daids()[0]
        >>> zoom = ut.get_argval('--zoom', float, .4)
        >>> show_chipmatch_graph(ibs, cm_list, qreq_, zoom=zoom)
        >>> #cm.show_single_annotmatch(qreq_, daid)
        >>> ut.show_if_requested()

    """
    pass
    # breaking
    num = 3
    qaid_list   = [cm.qaid for cm in cm_list]
    daids_list  = [cm.get_top_aids(num).tolist() for cm in cm_list]
    scores_list = [cm.get_annot_scores(daids) for cm, daids in zip(cm_list, daids_list)]
    #graph = netx.Graph()
    #node = graph.add_node(img)
    # FDSFDS!!!
    netx_graph = make_ibeis_matching_graph(ibs, qaid_list, daids_list, scores_list)
    fnum = None
    zoom = kwargs.get('zoom', .4)
    viz_netx_chipgraph(ibs, netx_graph, fnum=fnum, with_images=True, zoom=zoom)


def make_ibeis_matching_graph(ibs, qaid_list, daids_list, scores_list):
    print('make_ibeis_matching_graph')
    aid1_list = ut.flatten([[qaid] * len(daids) for qaid, daids in zip(qaid_list, daids_list)])
    aid2_list = ut.flatten(daids_list)
    unique_aids = list(set(aid2_list + qaid_list))
    score_list = ut.flatten(scores_list)

    # Make a graph between the chips
    nodes = list(zip(unique_aids))
    edges = list(zip(aid1_list, aid2_list, score_list))
    node_lbls = [('aid', 'int')]
    edge_lbls = [('weight', 'float')]
    netx_graph = make_netx_graph(nodes, edges, node_lbls, edge_lbls)
    return netx_graph


def make_netx_graph(nodes, edges, node_lbls=[], edge_lbls=[]):
    print('make_netx_graph')
    # Make a graph between the chips
    netx_nodes = [(ntup[0], {key[0]: val for (key, val) in zip(node_lbls, ntup[1:])})
                  for ntup in iter(nodes)]
    netx_edges = [(etup[0], etup[1], {key[0]: val for (key, val) in zip(edge_lbls, etup[2:])})
                  for etup in iter(edges)]
    netx_graph = netx.DiGraph()
    netx_graph.add_nodes_from(netx_nodes)
    netx_graph.add_edges_from(netx_edges)
    return netx_graph


def viz_netx_chipgraph(ibs, netx_graph, fnum=None, with_images=False, zoom=.4):
    if fnum is None:
        fnum = pt.next_fnum()
    print('[encounter] drawing chip graph')
    pt.figure(fnum=fnum, pnum=(1, 1, 1))
    ax = pt.gca()
    #pos = netx.spring_layout(graph)
    pos = netx.graphviz_layout(netx_graph)
    #pos = netx.fruchterman_reingold_layout(netx_graph)
    #pos = netx.spring_layout(netx_graph)
    netx.draw(netx_graph, pos=pos, ax=ax)
    if with_images:
        import cv2
        aid_list = netx_graph.nodes()
        pos_list = [pos[aid] for aid in aid_list]
        img_list = ibs.get_annot_chips(aid_list)
        img_list = [vt.resize_thumb(img, (220, 220)) for img in img_list]
        img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list]
        netx_draw_images_at_positions(img_list, pos_list, zoom=zoom)


def netx_draw_images_at_positions(img_list, pos_list, zoom=.4):
    """
    References:
        https://gist.github.com/shobhit/3236373
        http://matplotlib.org/examples/pylab_examples/demo_annotation_box.html
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    print('[encounter] drawing %d images' % len(img_list))
    # Thumb stack
    ax  = pt.gca()
    fig = pt.gcf()
    trans_data_to_axes = ax.transData.transform
    trans_figure_to_axes = fig.transFigure.inverted().transform
    def trans_data_to_figure(pt_data):
        pt_ax = trans_data_to_axes(pt_data)
        pt_fig = trans_figure_to_axes(pt_ax)
        return pt_fig
    #for pos, img in ut.ProgressIter(zip(pos_list, img_list),  lbl='drawing img'):
    artist_list = []
    offset_img_list = []
    for pos, img in zip(pos_list, img_list):
        FIG_COORDS = False
        if FIG_COORDS:
            pass
            #x, y = trans_data_to_figure(pos)
            #height, width = trans_data_to_figure(img.shape[0:2])
            ## Hack to get the sizes correct
            #height *= .17
            #width *= .17
            #tlx = x - (width / 2.0)
            #tly = y - (height / 2.0)
            #img_bbox = [tlx, tly, width, height]
            ## Make new axis for the image
            #img_ax = pt.plt.axes(img_bbox)
            #img_ax.imshow(img)
            #pt.draw_border(img_ax)
            #img_ax.set_aspect('equal')
            #img_ax.axis('off')
        else:
            x, y = pos
            height, width = img.shape[0:2]
            #offset_img = OffsetImage(img, zoom=.4)
            offset_img = OffsetImage(img, zoom=zoom)
            artist = AnnotationBbox(
                offset_img,
                (x, y),
                xybox=(-0., 0.),
                xycoords='data',
                boxcoords="offset points",
                pad=0.1)  # ,arrowprops=dict(arrowstyle="->"))
            offset_img_list.append(offset_img)
            artist_list.append(artist)

    for artist in artist_list:
        ax.add_artist(artist)

    def _onresize(event):
        print('foo' + ut.get_timestamp())

    def zoom_factory(ax, base_scale=1.1):
        """
        References:
            https://gist.github.com/tacaswell/3144287
        """
        def zoom_fun(event):
            print('zooming')
            # get the current x and y limits
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            # set the range
            #cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
            #cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
            xdata = event.xdata  # get event x location
            ydata = event.ydata  # get event y location
            if xdata is None or ydata is None:
                return
            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = base_scale
            else:
                raise NotImplementedError('event.button=%r' % (event.button,))
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)
            # set new limits
            #ax.set_xlim([xdata - cur_xrange * scale_factor,
            #             xdata + cur_xrange * scale_factor])
            #ax.set_ylim([ydata - cur_yrange * scale_factor,
            #             ydata + cur_yrange * scale_factor])
            # ----
            for offset_img in offset_img_list:
                zoom = offset_img.get_zoom()
                offset_img.set_zoom(zoom / (scale_factor ** (1.2)))
            # Get distance from the cursor to the edge of the figure frame
            x_left = xdata - cur_xlim[0]
            x_right = cur_xlim[1] - xdata
            y_top = ydata - cur_ylim[0]
            y_bottom = cur_ylim[1] - ydata
            ax.set_xlim([xdata - x_left * scale_factor, xdata + x_right * scale_factor])
            ax.set_ylim([ydata - y_top * scale_factor, ydata + y_bottom * scale_factor])

            # ----
            ax.figure.canvas.draw()  # force re-draw

        fig = ax.get_figure()  # get the figure of interest
        # attach the call back
        fig.canvas.mpl_connect('scroll_event', zoom_fun)

        #return the function
        return zoom_fun
    #pt.interact_helpers.connect_callback(fig, 'resize_event', _onresize)
    zoom_factory(ax)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.viz_graph
        python -m ibeis.viz.viz_graph --allexamples
        python -m ibeis.viz.viz_graph --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
