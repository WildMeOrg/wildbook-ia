# -*- coding: utf-8 -*-
"""
Displays the matching graph of individuals

WindowsDepends:
    pip install networkx
    wget http://www.graphviz.org/pub/graphviz/stable/windows/graphviz-2.38.msi
    graphviz-2.38.msi
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import plottool as pt
import vtool as vt
import numpy as np  # NOQA
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


ZOOM = ut.get_argval('--zoom', type_=float, default=.4)


def show_chipmatch_graph(ibs, cm_list, qreq_, fnum=None, pnum=None, **kwargs):
    r"""

    CommandLine:
        python -m ibeis.viz.viz_graph --test-show_chipmatch_graph:0 --show
        python -m ibeis.viz.viz_graph --test-show_chipmatch_graph:1 --show
        python -m ibeis.viz.viz_graph --test-show_chipmatch_graph:1 --show --zoom=.15
        python -m ibeis.viz.viz_graph --test-show_chipmatch_graph:1 --zoom=.25 --save foo.jpg --diskshow --figsize=12,6 --dpath=. --dpi=280


    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> from ibeis.algo.hots.chip_match import *  # NOQA
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
        >>> from ibeis.algo.hots.chip_match import *  # NOQA
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


def get_name_rowid_edges_from_nids(ibs, nids):
    aids_list = ibs.get_name_aids(nids)
    import itertools
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    #if full:
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)
    return aids1, aids2


def get_name_rowid_edges_from_aids2(ibs, aids_list):
    # grouped version
    import itertools
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    #if full:
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)
    return aids1, aids2


def get_name_rowid_edges_from_aids(ibs, aid_list):
    aids_list, nids = ibs.group_annots_by_name(aid_list)
    #aids_list = ibs.get_name_aids(nids)
    import itertools
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    #if full:
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)
    return aids1, aids2


#def make_netx_graph_from_nids(ibs, nids, full=False):
#    aids_list = ibs.get_name_aids(nids)
#    unique_aids = list(ut.flatten(aids_list))

#    aids1, aids2 = get_name_rowid_edges_from_aids2(ibs, aids_list)

#    if not full:
#        annotmatch_rowids = ibs.get_annotmatch_rowid_from_superkey(aids1, aids2)
#        annotmatch_rowids = ut.filter_Nones(annotmatch_rowids)
#        aids1 = ibs.get_annotmatch_aid1(annotmatch_rowids)
#        aids2 = ibs.get_annotmatch_aid2(annotmatch_rowids)

#    return make_netx_graph_from_aidpairs(ibs, aids1, aids2, unique_aids=unique_aids)


def make_netx_graph_from_aids(ibs, aids_list, full=False):
    #aids_list, nid_list = ibs.group_annots_by_name(aid_list)
    unique_aids = list(ut.flatten(aids_list))

    aids1, aids2 = get_name_rowid_edges_from_aids2(ibs, aids_list)

    if not full:
        annotmatch_rowids = ibs.get_annotmatch_rowid_from_superkey(aids1, aids2)
        annotmatch_rowids = ut.filter_Nones(annotmatch_rowids)
        aids1 = ibs.get_annotmatch_aid1(annotmatch_rowids)
        aids2 = ibs.get_annotmatch_aid2(annotmatch_rowids)

    return make_netx_graph_from_aidpairs(ibs, aids1, aids2, unique_aids=unique_aids)


def make_netx_graph_from_aidpairs(ibs, aids1, aids2, unique_aids=None):
    # Enumerate annotmatch properties
    import numpy as np  # NOQA
    rng = np.random.RandomState(0)
    edge_props = {
        'weight': rng.rand(len(aids1)),
        #'reviewer_confidence': rng.rand(len(aids1)),
        #'algo_confidence': rng.rand(len(aids1)),
    }

    edge_keys = list(edge_props.keys())
    edge_vals = ut.dict_take(edge_props, edge_keys)

    if unique_aids is None:
        unique_aids = list(set(aids1 + aids2))
    # Make a graph between the chips
    nodes = list(zip(unique_aids))
    edges = list(zip(aids1, aids2, *edge_vals))
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


def netx_draw_images_at_positions(img_list, pos_list, zoom=.4):
    """
    References:
        https://gist.github.com/shobhit/3236373
        http://matplotlib.org/examples/pylab_examples/demo_annotation_box.html

        http://matplotlib.org/api/text_api.html
        http://matplotlib.org/api/offsetbox_api.html

    TODO: look into DraggableAnnotation
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    print('[viz_graph] drawing %d images' % len(img_list))
    # Thumb stackartist
    ax  = pt.gca()
    artist_list = []
    offset_img_list = []
    for pos, img in zip(pos_list, img_list):
        x, y = pos
        height, width = img.shape[0:2]
        offset_img = OffsetImage(img, zoom=zoom)
        artist = AnnotationBbox(
            offset_img,
            (x, y),
            xybox=(-0., 0.),
            xycoords='data',
            boxcoords="offset points",
            #pad=0.1,
            pad=0.25,
            #frameon=False,
            frameon=True,
            #bboxprops=dict(fc="cyan"),
        )  # ,arrowprops=dict(arrowstyle="->"))
        offset_img_list.append(offset_img)
        artist_list.append(artist)

    for artist in artist_list:
        ax.add_artist(artist)

    # TODO: move this to the interaction

    def _onresize(event):
        print('foo' + ut.get_timestamp())

    def zoom_factory(ax, base_scale=1.1):
        """
        References:
            https://gist.github.com/tacaswell/3144287
        """
        def zoom_fun(event):
            #print('zooming')
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
    return artist_list


def viz_netx_chipgraph(ibs, netx_graph, fnum=None, with_images=False, zoom=ZOOM):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        netx_graph (?):
        fnum (int):  figure number(default = None)
        with_images (bool): (default = False)
        zoom (float): (default = 0.4)

    Returns:
        ?: pos

    CommandLine:
        python -m ibeis.viz.viz_graph --exec-viz_netx_chipgraph --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> nid_list = ibs.get_valid_nids()[0:5]
        >>> fnum = None
        >>> with_images = True
        >>> zoom = 0.4
        >>> #pos = viz_netx_chipgraph(ibs, netx_graph, fnum, with_images, zoom)
        >>> make_name_graph_interaction(ibs, None, ibs.get_valid_aids()[0:1])
        >>> #make_name_graph_interaction(ibs, nid_list)
        >>> ut.show_if_requested()
    """
    if fnum is None:
        fnum = pt.next_fnum()

    #zoom = .8
    print('[viz_graph] drawing chip graph')
    pt.figure(fnum=fnum, pnum=(1, 1, 1))
    ax = pt.gca()
    #pos = netx.spring_layout(graph)

    aid_list = netx_graph.nodes()

    IMPLICIT_LAYOUT = len(set(ibs.get_annot_nids(aid_list))) != 1
    # FIXME
    print('zoom = %r' % (zoom,))

    if IMPLICIT_LAYOUT:
        # HACK:
        # Use name edge to make pos (very bad)
        aids1, aids2 = get_name_rowid_edges_from_aids(ibs, aid_list)
        netx_graph_hack = make_netx_graph_from_aidpairs(ibs, aids1, aids2, unique_aids=aid_list)
        pos = netx.nx_agraph.graphviz_layout(netx_graph_hack)
    else:
        pos = netx.nx_agraph.graphviz_layout(netx_graph)

    #pos = netx.fruchterman_reingold_layout(netx_graph)
    #pos = netx.spring_layout(netx_graph)
    netx.draw(netx_graph, pos=pos, ax=ax)

    with_nid_edges = True
    if with_nid_edges:
        import matplotlib as mpl
        import scipy.sparse as spsparse

        aids1, aids2 = get_name_rowid_edges_from_aids(ibs, aid_list)
        edge_pts1 = np.array(ut.dict_take(pos, aids1), dtype=np.int32)
        edge_pts2 = np.array(ut.dict_take(pos, aids2), dtype=np.int32)

        if len(edge_pts1) == 0:
            edge_pts1 = edge_pts1[:, None]

        if len(edge_pts2) == 0:
            edge_pts2 = edge_pts2[:, None]

        I = np.array(aids1)
        J = np.array(aids2)
        if len(aid_list) > 0:
            N = max(aid_list) + 1
        else:
            N = 1
        forced_edge_idxs = ut.dict_take(dict(zip(zip(I, J), range(len(I)))), netx_graph.edges())
        data = vt.L2(edge_pts1, edge_pts2)
        if len(forced_edge_idxs) > 0:
            data[forced_edge_idxs] = 0.00001

        graph = spsparse.coo_matrix((data, (I, J)), shape=(N, N))

        def extract_connected_compoments(graph):
            import scipy.sparse as spsparse
            import utool as ut
            # I think this is how extraction is done?
            # only returns edge info
            # so singletons are not represented
            shape = graph.shape
            csr_graph = graph.tocsr()
            num_components, labels = spsparse.csgraph.connected_components(csr_graph)
            unique_labels = np.unique(labels)
            group_flags_list = [labels == groupid for groupid in unique_labels]
            subgraph_list = []
            for label, group_flags in zip(unique_labels, group_flags_list):
                num_members = group_flags.sum()
                ixs = list(range(num_members))
                if num_members == 0:
                    continue
                group_rowix, group_cols = csr_graph[group_flags, :].nonzero()
                if len(group_cols) == 0:
                    continue
                ix2_row = dict(zip(ixs, np.nonzero(group_flags)[0]))
                group_rows = ut.dict_take(ix2_row, group_rowix)
                component = (group_rows, group_cols.tolist())
                data = csr_graph[component].tolist()[0]
                subgraph = spsparse.coo_matrix((data, component), shape=shape)
                subgraph_list.append(subgraph)
            #assert len(compoment_list) == num_components, 'bad impl'
            return subgraph_list
        subgraph_list = extract_connected_compoments(graph)

        spantree_aids1_ = []
        spantree_aids2_ = []

        for subgraph in subgraph_list:
            subgraph_spantree = spsparse.csgraph.minimum_spanning_tree(subgraph)
            min_aids1_, min_aids2_ = subgraph_spantree.nonzero()
            spantree_aids1_.extend(min_aids1_)
            spantree_aids2_.extend(min_aids2_)

        edge_pts1_ = np.array(ut.dict_take(pos, spantree_aids1_))
        edge_pts2_ = np.array(ut.dict_take(pos, spantree_aids2_))

        segments = list(zip(edge_pts1_, edge_pts2_))
        #pt.distinct_colors
        color_list = pt.DARK_ORANGE
        #color_list = pt.BLACK
        line_group = mpl.collections.LineCollection(segments, color=color_list, alpha=.3, lw=4)
        ax.add_collection(line_group)

    if with_images:
        import cv2
        pos_list = ut.dict_take(pos, aid_list)
        img_list = ibs.get_annot_chips(aid_list)
        img_list = [vt.resize_thumb(img, (220, 220)) for img in img_list]
        img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list]
        artist_list = netx_draw_images_at_positions(img_list, pos_list, zoom=zoom)
        for artist, aid in zip(artist_list, aid_list):
            pt.set_plotdat(artist, 'aid', aid)
    return pos


def make_name_graph_interaction(ibs, nids=None, aids=None, selected_aids=[], zoom=ZOOM, with_all=True):
    """
    CommandLine:
        python -m ibeis.viz.viz_graph --exec-make_name_graph_interaction --db PZ_Master1 --aids 2068 1003 --show
        python -m ibeis.viz.viz_graph --exec-make_name_graph_interaction --db testdb1 --show

        python -m ibeis.viz.viz_graph --exec-make_name_graph_interaction --db PZ_Master1 --aids 2068 1003 1342 758 --show
        python -m ibeis.viz.viz_graph --exec-make_name_graph_interaction --db PZ_Master1 --aids 758 1342 --show

        python -m ibeis.viz.viz_graph --exec-make_name_graph_interaction --db PZ_Master1 --aids 2068 1003 --show
        python -m ibeis --tf make_name_graph_interaction --aids 3 --show

        python -m ibeis.viz.viz_graph --exec-make_name_graph_interaction --aids 193 194 195 196 --show --db WD_Siva

        python -m ibeis.viz.viz_graph --exec-make_name_graph_interaction --aids 792 --show --db GZ_ALL --no-with-all

        python -m ibeis.viz.viz_graph --exec-make_name_graph_interaction --aids=788,789,790,791 --show --db GZ_ALL --no-with-all
        python -m ibeis.viz.viz_graph --exec-make_name_graph_interaction --aids=782,783,792 --show --db GZ_ALL --no-with-all



    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph import *  # NOQA
        >>> import ibeis
        >>> aids = ut.get_argval('--aids', type_=list, default=None)
        >>> defaultdb='testdb1'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> nids = None if aids is not None else ibs.get_valid_nids()[0:5]
        >>> with_all = not ut.get_argflag('--no-with-all')
        >>> make_name_graph_interaction(ibs, nids, aids, with_all=with_all)
        >>> defaultdb='testdb1'
        >>> ut.show_if_requested()

    """
    import plottool as pt
    from plottool.abstract_interaction import AbstractInteraction
    print('aids = %r' % (aids,))

    class NameGraphInteraction(AbstractInteraction):
        def __init__(self, ibs, nids=None, aids=None, selected_aids=[]):
            super(NameGraphInteraction, self).__init__()
            self.ibs = ibs
            self.selected_aids = selected_aids
            self._nids = nids if nids is not None else []
            self._aids = aids if aids is not None else []
            self.with_images = True
            self._aids2 = None

        def update_netx_graph(self):
            nids_list = []

            if self._aids2 is not None:
                nids2 = ibs.get_annot_nids(self._aids2)
                nids_list += [nids2]

            if with_all:
                nids_list += [ibs.get_annot_nids(self._aids)]
                nids_list += [self._nids]
                nids = list(set(ut.flatten(nids_list)))
                aids_list = ibs.get_name_aids(nids)
            else:
                aids_list = ibs.group_annots_by_name(self._aids)[0]
            self.netx_graph = make_netx_graph_from_aids(ibs, aids_list)
            # TODO: allow for a subset of grouped aids to be shown
            #self.netx_graph = make_netx_graph_from_nids(ibs, nids)
            self._aids2 = self.netx_graph.nodes()
            self.aid2_index = {key: val for val, key in enumerate(self._aids2)}

        def plot(self, fnum, pnum):
            from ibeis.viz.viz_graph import viz_netx_chipgraph
            self.update_netx_graph()
            self.pos = viz_netx_chipgraph(self.ibs, self.netx_graph,
                                          fnum=self.fnum, with_images=self.with_images,
                                          zoom=zoom)
            self.ax = pt.gca()

            for aid in self.selected_aids:
                self.highlight_aid(aid)
                pass

            #self.static_plot(fnum, pnum)

        def highlight_aid(self, aid, color=pt.ORANGE):
            ax = self.ax
            index = self.aid2_index[aid]
            artist = ax.artists[index]
            artist.patch.set_facecolor(color)

        def toggle_images(self):
            self.with_images = not self.with_images
            self.show_page()

        def toggle_selected_aid(self, aid):
            if aid in self.selected_aids:
                self.selected_aids.remove(aid)
                self.highlight_aid(aid, pt.WHITE)
            else:
                self.selected_aids.append(aid)
                self.highlight_aid(aid, pt.ORANGE)
            self.draw()

        def on_key_press(self, event):
            print(event)

            if event.key == 'r':
                self.show_page()
                self.draw()

            if event.key == 'i':
                ut.embed()

            if len(self.selected_aids) == 2:
                ibs = self.ibs
                aid1, aid2 = self.selected_aids
                _rowid = ibs.get_annotmatch_rowid_from_superkey([aid1], [aid2])
                if _rowid is None:
                    _rowid = ibs.get_annotmatch_rowid_from_superkey([aid2], [aid1])
                rowid = _rowid  # NOQA

        def mark_pair_truth(self, truth):
            if len(len(self.selected_aids)) != 2:
                print('This funciton only work if exactly 2 are selected')
                return
            aid1, aid2 = self.selected_aids
            print('aid2 = %r' % (aid2,))
            print('aid1 = %r' % (aid1,))

        def on_click_inside(self, event, ax):
            self.ax = ax
            self.event = event
            event = self.event
            #print(ax)
            #print(event.x)
            #print(event.y)
            aids = list(self.pos.keys())
            pos_list = ut.dict_take(self.pos, aids)
            #x = 10
            #y = 10
            import numpy as np  # NOQA
            x, y = event.xdata, event.ydata
            point = np.array([x, y])
            pos_list = np.array(pos_list)
            index, dist = vt.closest_point(point, pos_list, distfunc=vt.L2)
            print('dist = %r' % (dist,))

            aid = aids[index]
            context_shown = False

            CHECK_PAIR = True
            if CHECK_PAIR:
                if self.event.button == 3 and not context_shown:
                    if len(self.selected_aids) != 2:
                        print('This funciton only work if exactly 2 are selected')
                    else:
                        from ibeis.gui import inspect_gui
                        context_shown = True
                        aid1, aid2 = (self.selected_aids)
                        qres = None
                        qreq_ = None
                        options = inspect_gui.get_aidpair_context_menu_options(ibs, aid1, aid2, qres, qreq_=qreq_)
                        self.show_popup_menu(options, event)

            SELECT_ANNOT = dist < 35
            if SELECT_ANNOT:
                print(ut.obj_str(ibs.get_annot_info(aid, default=True, name=False, gname=False)))

                if self.event.button == 1:
                    self.toggle_selected_aid(aid)

                if self.event.button == 3 and not context_shown:
                    # right click
                    from ibeis.viz.interact import interact_chip
                    context_shown = True
                    #refresh_func = functools.partial(viz.show_name, ibs, nid, fnum=fnum, sel_aids=sel_aids)
                    refresh_func = None
                    config2_ = None
                    options = interact_chip.build_annot_context_options(
                        ibs, aid, refresh_func=refresh_func,
                        with_interact_name=False,
                        config2_=config2_)
                    self.show_popup_menu(options, event)
            else:
                if self.event.button == 3:
                    options = [
                        ('Toggle images', self.toggle_images),
                    ]
                    self.show_popup_menu(options, event)

    self = NameGraphInteraction(ibs, nids, aids, selected_aids=selected_aids)
    self.show_page()
    self.show()

    #ax = self.fig.axes[0]
    #for index in range(len(ax.artists)):
    #    artist = ax.artists[index]
    #    bbox = artist.patch
    #    bbox.set_facecolor(pt.ORANGE)
    #    #offset_img = artist.offsetbox
    #    #img = offset_img.get_data()
    #    #offset_img.set_data(vt.draw_border(img, thickness=5))

    #bbox = artist.patch

    #ax.figure.canvas.draw()  # force re-draw

    return self


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
