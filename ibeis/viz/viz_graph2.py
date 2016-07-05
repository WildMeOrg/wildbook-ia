# -*- coding: utf-8 -*-
"""
CommandLine:
    python -m ibeis.viz.viz_graph2 make_qt_graph_interface --show
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import vtool as vt
import numpy as np  # NOQA
import plottool as pt
import networkx as nx
import itertools
from ibeis.algo.hots import graph_iden
import guitool as gt
from guitool.__PYQT__.QtCore import Qt
from guitool.__PYQT__ import QtCore, QtWidgets
from plottool import interact_helpers as ih
from matplotlib.backend_bases import MouseEvent


class MatplotlibWidget(gt.GuitoolWidget):
    click_inside_signal = QtCore.pyqtSignal(MouseEvent, object)

    def initialize(self):
        from guitool import __PYQT__
        import plottool as pt
        from plottool.interactions import zoom_factory, pan_factory
        from plottool.abstract_interaction import AbstractInteraction
        if __PYQT__._internal.GUITOOL_PYQT_VERSION == 4:
            import matplotlib.backends.backend_qt4agg as backend_qt
        else:
            import matplotlib.backends.backend_qt5agg as backend_qt
        FigureCanvas = backend_qt.FigureCanvasQTAgg
        self.fig = pt.plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        #self.ax.plot([1, 2, 3], [1, 2, 3])
        self.addWidget(self.canvas)

        self.pan_events = pan_factory(self.ax)
        self.zoon_events = zoom_factory(self.ax)
        ih.connect_callback(self.fig, 'button_press_event', self.on_click)
        ih.connect_callback(self.fig, 'draw_event', self.draw_callback)

        self.MOUSE_BUTTONS = AbstractInteraction.MOUSE_BUTTONS
        self.setMinimumHeight(20)
        self.setMinimumWidth(20)

    def on_click(self, event):
        print('[mplwidget] on_click')
        if ih.clicked_inside_axis(event):
            ax = event.inaxes
            self.click_inside_signal.emit(event, ax)

    def draw_callback(self, event):
        pass
        # self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        # self.draw_artists()


class AnnotGraphWidget(gt.GuitoolWidget):

    def initialize(self, infr=None, use_image=False):
        self.infr = infr
        self.selected_aids = []
        self.node2_aid = nx.get_node_attributes(self.infr.graph, 'aid')
        self.aid2_node = ut.invert_dict(self.node2_aid)
        node2_label = {
            #node: '%d:aid=%r' % (node, aid)
            node: 'aid=%r' % (aid)
            for node, aid in self.node2_aid.items()
        }
        #self.show_cuts = False
        self.use_image = use_image
        self.show_cuts = False
        # self.config = InferenceConfig()
        nx.set_node_attributes(self.infr.graph, 'label', node2_label)

        self.splitter = self.addNewSplitter(orientation=Qt.Horizontal)
        splitter = self.splitter
        self.ctrls = splitter.addNewWidget(orientation=Qt.Vertical,
                                           vertical_stretch=1, margin=1,
                                           spacing=1)

        graph_tables_widget = splitter.addNewTabWidget(verticalStretch=1)
        edge_tab = graph_tables_widget.addNewTab('Edges')
        node_tab = graph_tables_widget.addNewTab('Nodes')
        #self.edge_table = edge_tab.addNewTableWidget()
        #self.node_table = node_tab.addNewTableWidget()
        self.edge_api_widget = gt.APIItemWidget()
        self.node_api_widget = gt.APIItemWidget()
        node_tab.addWidget(self.node_api_widget)
        edge_tab.addWidget(self.edge_api_widget)

        self.mpl_wgt = MatplotlibWidget(parent=self)
        splitter.addWidget(self.mpl_wgt)

        ctrls = self.ctrls
        bbar1 = ctrls.addNewWidget(ori='vert', margin=1, spacing=1)
        bbar2 = ctrls.addNewWidget(ori='vert', margin=1, spacing=1)
        bbar3 = ctrls.addNewWidget(ori='vert', margin=1, spacing=1)
        def _simple_button3(func, text=None, refresh=True):
            def _simple_onevent():
                func()
                if refresh:
                    self.draw_graph()
            wrapped_func = ut.preserve_sig(_simple_onevent, func)
            bbar3.addNewButton(text, pressed=wrapped_func)

        def _simple_button2(func, refresh=True):
            def _simple_onevent():
                func()
                if refresh:
                    self.draw_graph()
            wrapped_func = ut.preserve_sig(_simple_onevent, func)
            bbar3.addNewButton(pressed=wrapped_func)

        bbar1.addNewButton('Mark: Match', pressed=self.mark_match)
        bbar1.addNewButton('Mark: Non-Match', pressed=self.mark_nonmatch)
        bbar1.addNewButton('Mark: Not-Comp', pressed=self.mark_notcomp)

        thresh_wgt = bbar1.addNewWidget(orientation=Qt.Horizontal, margin=1)
        thresh_wgt.addNewLabel('Thresh:')
        self.thresh_lbl = thresh_wgt.addNewLineEdit('.5', editable=False)

        bbar2.addNewButton('Deselect', pressed=self.deselect)
        bbar2.addNewButton('Show Annots', pressed=self.show_selected)
        bbar2.addNewButton('Infer Cut', pressed=self.cut)

        bbar2.addNewCheckBox('Show Cuts', changed=self.toggle_cuts,
                             checked=self.show_cuts)
        bbar2.addNewCheckBox('Show Img', changed=self.toggle_imgs,
                             checked=self.use_image)
        self.toggle_pin = bbar2.addNewCheckBox(changed=self.toggle_pin,
                                               checked=False)

        bbar2.addNewButton('Accept', pressed=self.confirm)

        #Debug row
        _simple_button3(self.reset)
        _simple_button3(self.infr.apply_mst)
        _simple_button3(self.infr.apply_scores)
        _simple_button3(self.infr.apply_feedback)
        _simple_button3(self.infr.apply_weights)
        _simple_button3(self.infr.apply_cuts)
        _simple_button3(self.infr.apply_all)
        _simple_button3(self.embed)

        self.cb = None
        self.mpl_wgt.click_inside_signal.connect(self.on_click_inside)
        self.populate_node_model()

    def reset(self):
        self.infr.initialize_graph()
        self.toggle_pin.setChecked(False)

    def populate_node_model(self):
        aids = sorted(list(self.infr.graph.nodes()))
        col_name_list = [
            'aid',
            'data',
            'thumb',
        ]
        def get_node_data(aid):
            data = self.infr.graph.node[aid].copy()
            ut.delete_dict_keys(data,
                                ['color', 'framewidth', 'image', 'label',
                                 'pos', 'shape', 'size', 'height', 'width'])
            return ut.repr2(data, precision=2)
        col_getter_dict = {
            'aid': np.array(aids),
            'data': get_node_data,
            'thumb': self.infr.ibs.get_annot_chip_thumbtup,
        }
        col_ider_dict = {
            'thumb': 'aid',
            'data': 'aid',
        }
        col_types_dict = {
            'thumb': 'PIXMAP',
        }
        node_api = gt.CustomAPI(col_name_list,
                                col_ider_dict=col_ider_dict,
                                col_types_dict=col_types_dict,
                                col_getter_dict=col_getter_dict,
                                sortby='aid', sort_reverse=False)
        headers = node_api.make_headers(tblnice='Nodes')
        self.node_api_widget.change_headers(headers)
        return node_api

    def populate_edge_model(self):
        graph = self.infr.graph
        col_name_list = [
            'aid1',
            'aid2',
            'data',
            #'thumb',
        ]
        def get_edge_data(edge):
            aid1, aid2 = edge
            attrs = graph.get_edge_data(aid1, aid2).copy()
            ut.delete_dict_keys(attrs, ['color', 'implicit', 'stroke', 'lw',
                                        'end_pt', 'head_lp', 'alpha', 'style',
                                        'ctrl_pts', 'pos'])
            return ut.repr2(attrs, precision=2)
        uv_list = list(graph.edges())
        aids1 = ut.take_column(uv_list, 0)
        aids2 = ut.take_column(uv_list, 1)
        col_getter_dict = {
            'aid1': np.array(aids1),
            'aid2': np.array(aids2),
            'data': get_edge_data,
            #'thumb': self.infr.ibs.get_annot_chip_thumbtup,
        }
        col_ider_dict = {
            #'thumb': 'aid',
            'data': ('aid1', 'aid2'),
        }
        col_types_dict = {
            #'thumb': 'PIXMAP',
        }
        edge_api = gt.CustomAPI(col_name_list,
                                col_ider_dict=col_ider_dict,
                                col_types_dict=col_types_dict,
                                col_getter_dict=col_getter_dict,
                                sortby='aid1', sort_reverse=False)
        headers = edge_api.make_headers(tblnice='Edges')
        self.edge_api_widget.change_headers(headers)

    def populate_edge_table(self):
        print('Updating saved query table')
        from guitool.__PYQT__.QtCore import Qt
        graph = self.infr.graph
        horHeaders = ['aid1', 'aid2', 'score', 'weight', 'reviewed_weight', 'is_cut']
        self.edge_table.setColumnCount(len(horHeaders))
        print('Populating table')
        uvd_list = list(graph.edges(data=True))
        self.edge_table.setRowCount(len(uvd_list))
        for row, uvd in enumerate(uvd_list):
            aid1, aid2 = uvd[0:2]
            attrs = uvd[2].copy()
            attrs['aid1'] = aid1
            attrs['aid2'] = aid2
            # col_data = [aid1, aid2, attrs]
            for col, colname in enumerate(horHeaders):
                data = attrs.get(colname, None)
                newitem = QtWidgets.QTableWidgetItem(str(data))
                self.edge_table.setItem(row, col, newitem)
                newitem.setFlags(newitem.flags() ^ Qt.ItemIsEditable)
            ut.delete_dict_keys(attrs, horHeaders)
            ut.delete_dict_keys(attrs, ['color', 'implicit', 'stroke', 'lw',
                                        'style', 'ctrl_pts', 'pos'])
            attrs = {key: val for key, val in attrs.items() if val is not None}

            newitem = QtWidgets.QTableWidgetItem(str(attrs))
            self.edge_table.setItem(row, col, newitem)
            newitem.setFlags(newitem.flags() ^ Qt.ItemIsEditable)

        self.edge_table.setHorizontalHeaderLabels(horHeaders)
        self.edge_table.resizeColumnsToContents()
        self.edge_table.resizeRowsToContents()
        print('Finished populating table')

    def on_click_inside(self, event, ax):
        pos = self.plotinfo['node']['pos']
        nodes = list(pos.keys())
        pos_list = ut.dict_take(pos, nodes)

        # TODO: FIXME
        #x = 10
        #y = 10
        import numpy as np  # NOQA
        x, y = event.xdata, event.ydata
        point = np.array([x, y])
        pos_list = np.array(pos_list)
        index, dist = vt.closest_point(point, pos_list, distfunc=vt.L2)
        node = nodes[index]
        aid = self.node2_aid[node]
        context_shown = False

        CHECK_PAIR = True
        if CHECK_PAIR:
            if event.button == 3 and not context_shown:
                if len(self.selected_aids) != 2:
                    print('This funciton only work if exactly 2 are selected')
                else:
                    from ibeis.gui import inspect_gui
                    context_shown = True
                    aid1, aid2 = (self.selected_aids)
                    qres = None
                    qreq_ = None
                    options = inspect_gui.get_aidpair_context_menu_options(
                        self.infr.ibs, aid1, aid2, qres, qreq_=qreq_)
                    self.show_popup_menu(options, event)

        bbox = vt.bbox_from_center_wh(self.plotinfo['node']['pos'][node],
                                      self.plotinfo['node']['size'][node])
        SELECT_ANNOT = vt.point_inside_bbox(point, bbox)
        #SELECT_ANNOT = dist < 35

        if SELECT_ANNOT:
            #print(ut.obj_str(ibs.get_annot_info(aid, default=True,
            #                                    name=False, gname=False)))

            if event.button == 1:
                self.toggle_selected_aid(aid)

            if event.button == 3 and not context_shown:
                # right click
                from ibeis.viz.interact import interact_chip
                context_shown = True
                #refresh_func = functools.partial(viz.show_name, ibs, nid,
                #fnum=fnum, sel_aids=sel_aids)
                refresh_func = None
                config2_ = None
                options = interact_chip.build_annot_context_options(
                    self.infr.ibs, aid, refresh_func=refresh_func,
                    with_interact_name=False,
                    config2_=config2_)
                self.show_popup_menu(options, event)
        # else:
        #     if event.button == 3:
        #         options = [
        #             ('Toggle images', self.toggle_imgs),
        #         ]
        #         self.show_popup_menu(options, event)

    def show_popup_menu(self, options, event):
        """
        context menu
        """
        height = self.mpl_wgt.fig.canvas.geometry().height()
        qpoint = gt.newQPoint(event.x, height - event.y)
        qwin = self.mpl_wgt.fig.canvas
        gt.popup_menu(qwin, qpoint, options)

    def toggle_imgs(self, flag=False):
        print('toggle toggle_imgs flag = %r' % (flag,))
        self.use_image = flag
        self.draw_graph()

    def toggle_cuts(self, flag=False):
        print('toggle show_cuts flag = %r' % (flag,))
        self.show_cuts = flag
        self.draw_graph()

    def toggle_pin(self, flag):
        if flag:
            nx.set_node_attributes(self.infr.graph, 'pin', 'true')
        else:
            ut.nx_delete_node_attr(self.infr.graph, 'pin')

    def cut(self):
        # keys = ['min_labels', 'max_labels']
        # infrkw = ut.dict_subset(self.config, keys)
        infrkw = {}
        self.infr.infer_cut(**infrkw)
        self.draw_graph()

    def mark_nonmatch(self):
        print('BREAK LINK self.selected_aids = %r' % (self.selected_aids,))
        for aid1, aid2 in itertools.combinations(self.selected_aids, 2):
            self.infr.add_feedback(aid1, aid2, 'nonmatch')
        self.infr.apply_feedback()
        self.draw_graph()

    def mark_match(self):
        print('MAKE LINK self.selected_aids = %r' % (self.selected_aids,))
        for aid1, aid2 in itertools.combinations(self.selected_aids, 2):
            self.infr.add_feedback(aid1, aid2, 'match')
        self.infr.apply_feedback()
        self.draw_graph()

    def mark_notcomp(self):
        print('MAKE LINK self.selected_aids = %r' % (self.selected_aids,))
        for aid1, aid2 in itertools.combinations(self.selected_aids, 2):
            self.infr.add_feedback(aid1, aid2, 'notcomp')
        self.infr.apply_feedback()
        self.draw_graph()

    def deselect(self):
        print('self.selected_aids = %r' % (self.selected_aids,))
        for aid in self.selected_aids[:]:
            self.toggle_selected_aid(aid)

    def confirm(self):
        print('Not done yet')
        print(self.infr.current_name_labels)

    def show_selected(self):
        print('show_selected')
        from ibeis.viz import viz_chip
        fnum = pt.ensure_fnum(10)
        print('fnum = %r' % (fnum,))
        fig = pt.figure(fnum=fnum)
        viz_chip.show_many_chips(self.infr.ibs, self.selected_aids)
        fig.show()
        fig.canvas.draw()
        #fig.canvas.update()

    def highlight_aid(self, aid, color=None):
        node = self.aid2_node[aid]
        frame = self.plotinfo['patch_frame_dict'][node]
        framewidth = self.infr.graph.node[node]['framewidth']
        if color is True:
            color = pt.ORANGE
        if color is None or color is False:
            color = pt.DARK_BLUE
            color = self.infr.graph.node[node]['color']
            color = pt.fix_hex_color(color)
            frame.set_linewidth(framewidth)
        else:
            frame.set_linewidth(framewidth * 2)
        frame.set_facecolor(color)
        frame.set_edgecolor(color)

    def toggle_selected_aid(self, aid):
        if aid in self.selected_aids:
            self.selected_aids.remove(aid)
            #self.highlight_aid(aid, pt.WHITE)
            self.highlight_aid(aid, color=None)
        else:
            self.selected_aids.append(aid)
            self.highlight_aid(aid, True)
        print('self.selected_aids = %r' % (self.selected_aids,))
        self.mpl_wgt.fig.canvas.draw()

    def update_graph_layout(self):
        graph = self.infr.graph
        self.infr.update_graph_visual_attributes(self.show_cuts)
        layoutkw = dict(prog='neato', splines='spline', sep=10 / 72)
        _, layout_info = pt.nx_agraph_layout(graph, inplace=True, **layoutkw)
        pass

    def draw_graph(self):
        print('Start draw page')
        self.mpl_wgt.ax.cla()
        self.update_graph_layout()

        # Update Qt things
        self.populate_edge_model()
        self.thresh_lbl.setText('%.2f' % (self.infr.thresh))

        # Update MPL things
        layoutkw = dict(prog='neato', splines='spline', sep=10 / 72)

        #draw_implicit=self.show_cuts)
        self.plotinfo = pt.show_nx(self.infr.graph,
                                   layout='custom',
                                   as_directed=False,
                                   ax=self.mpl_wgt.ax,
                                   layoutkw=layoutkw,
                                   #node_labels=True,
                                   modify_ax=False,
                                   use_image=self.use_image, verbose=0)
        # self.mpl_wgt.ax.set_aspect('auto')
        self.mpl_wgt.ax.set_aspect('equal')
        # self.mpl_wgt.ax.set_aspect('scaled')

        for aid in self.selected_aids:
            self.highlight_aid(aid, True)

        ut.util_graph.graph_info(self.infr.graph, verbose=True)
        self.show_colorbar()

        print('End draw page')
        self.mpl_wgt.canvas.draw()
        # fig.canvas.blit(ax.bbox)

        self.mpl_wgt.fig.subplots_adjust(left=.02, top=.98, bottom=.02, right=.85)
        print('Finished Plot')

    def show_colorbar(self):
        xy = (1, self.infr.thresh)
        xytext = (2.5, .3 if self.infr.thresh < .5 else .7)
        print('xy = %r' % (xy,))
        print('xytext = %r' % (xytext,))
        #_, edge_weights, edge_colors = self.infr.get_colored_edge_weights()
        #pt.colorbar(edge_weights, edge_colors, lbl='weights')
        # self.cb = None

        if self.cb is not None:
            ax = self.cb.ax
            from plottool import plot_helpers as ph
            ph.del_plotdat(self.mpl_wgt.ax, pt.DF2_DIVIDER_KEY)
            ph.del_plotdat(self.mpl_wgt.ax, 'df2_div_axes')
            ax.cla()
            ax.xaxis.set_ticks_position('none')
            ax.set_xticks([])
            ax.axes.get_xaxis().set_visible(False)

        _normal_ticks = np.linspace(0, 1, num=11)
        _normal_scores = np.linspace(0, 1, num=500)
        _normal_colors = self.infr.get_colored_weights(_normal_scores)
        cb = pt.colorbar(_normal_scores, _normal_colors, lbl='weights',
                         ticklabels=_normal_ticks)

        self.thresh_annot = cb.ax.annotate(
            'threshold',
            xy=xy,
            xytext=xytext,
            arrowprops=dict(
                alpha=.5,
                fc="0.6",
                connectionstyle="angle3,angleA=90,angleB=0"),)
        # self.thresh_annot.set_animated(True)
        self.cb = cb
        # else:
        #     self.thresh_annot.set_x(xy[0])
        #     self.thresh_annot.set_y(xy[1])
        #     self.thresh_annot.set_position(xytext)
        #     self.thresh_annot.set_position(xy)
        #     print('not redoing cb yet')

        #ax = pt.gca()
        #self.enable_pan_and_zoom(ax)
        #ax.autoscale()
        #for aid in self.selected_aids:
        #    self.highlight_aid(aid, pt.ORANGE)
        #self.static_plot(fnum, pnum)
        #self.make_hud()
        #print(ut.repr2(self.infr.graph.edge, nl=2))

    def embed(self):
        fig = self.mpl_wgt.fig  # NOQA
        ax = self.mpl_wgt.ax  # NOQA
        infr = self.infr  # NOQA
        ibs = infr.ibs  # NOQA
        import utool
        utool.embed()


def make_qt_graph_interface(ibs, aids):
    r"""
    CommandLine:
        python -m ibeis.viz.viz_graph2 make_qt_graph_interface --show --aids="[1, 2, 3, 4, 5, 6, 7, 8, 9]"
        python -m ibeis.viz.viz_graph2 make_qt_graph_interface --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph2 import *  # NOQA
        >>> import guitool as gt
        >>> import ibeis
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> aids = ut.get_argval('--aids', type_=list, default=None)
        >>> gt.ensure_qtapp()
        >>> win = make_qt_graph_interface(ibs, aids)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=win, freq=10)
    """
    if aids is None:
        aids = ibs.get_valid_aids()[0:40]
    print('make_qt_graph_interface aids = %r' % (aids,))
    temp_nids = None
    nids = ibs.get_annot_name_rowids(aids)
    infr = graph_iden.AnnotInference2(ibs, aids, nids, temp_nids)
    infr.initialize_graph()
    #infr.apply_scores()
    # infr.apply_feedback()
    # infr.apply_mst()
    #infr.apply_weights()
    #infr.apply_cuts()
    if ut.get_argflag('--cut'):
        infr.apply_all()
    gt.ensure_qtapp()
    print('infr = %r' % (infr,))
    win = AnnotGraphWidget(infr=infr, use_image=False)
    win.resize(900, 600)
    win.draw_graph()
    return win


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.viz.viz_graph2
        python -m ibeis.viz.viz_graph2 --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
