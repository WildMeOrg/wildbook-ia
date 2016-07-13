# -*- coding: utf-8 -*-
"""
CommandLine:
    #python -m ibeis.viz.viz_graph2 make_qt_graph_interface --show
    python -m ibeis.viz.viz_graph2 make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9
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
from guitool.__PYQT__ import QtCore, QtWidgets  # NOQA
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
        ih.connect_callback(self.fig, 'pick_event', self.on_pick)

        self.MOUSE_BUTTONS = AbstractInteraction.MOUSE_BUTTONS
        self.setMinimumHeight(20)
        self.setMinimumWidth(20)

    def on_click(self, event):
        #print('[mplwidget] on_click')
        if ih.clicked_inside_axis(event):
            ax = event.inaxes
            self.click_inside_signal.emit(event, ax)

    def on_pick(self, event):
        print('PICK: event.artist = %r' % (event.artist,))
        edge = pt.get_plotdat(event.artist, 'edge')
        node = pt.get_plotdat(event.artist, 'node')
        edge_data = pt.get_plotdat(event.artist, 'edge_data')
        node_data = pt.get_plotdat(event.artist, 'node_data')
        if edge_data is not None:
            print('edge = %r' % (edge,))
            print('edge_data = %s' % (ut.repr3(edge_data),))
            pass
        if node_data is not None:
            print('node = %r' % (node,))
            pass

    def draw_callback(self, event):
        pass
        # self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        # self.draw_artists()


class AnnotGraphWidget(gt.GuitoolWidget):
    signal_graph_update = QtCore.pyqtSignal()
    signal_state_update = QtCore.pyqtSignal(bool)

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
        # self.config = InferenceConfig()
        nx.set_node_attributes(self.infr.graph, 'label', node2_label)

        self.splitter = self.newSplitter(orientation=Qt.Horizontal)
        splitter = self.splitter
        self.ctrls = splitter.addNewWidget(orientation=Qt.Vertical,
                                           vertical_stretch=1, margin=1,
                                           spacing=1)

        graph_tables_widget = self.addNewTabWidget(verticalStretch=1)

        self.status_bar = self.addNewWidget(orientation=Qt.Horizontal,
                                            vertical_stretch=1, margin=1,
                                            spacing=1)

        #graph_tables_widget = splitter.addNewTabWidget(verticalStretch=1)
        self.edge_tab = graph_tables_widget.addNewTab('Edges')
        self.node_tab = graph_tables_widget.addNewTab('Nodes')
        self.graph_tab = graph_tables_widget.addNewTab('Graph')

        #self.addWidget(splitter)
        self.graph_tab.addWidget(splitter)
        #self.edge_table = edge_tab.addNewTableWidget()
        #self.node_table = node_tab.addNewTableWidget()

        self.edge_api_widget = gt.APIItemWidget()
        self.node_api_widget = gt.APIItemWidget()

        self.num_names_lbl = self.status_bar.addNewLabel('NUM_NAMES_LBL')
        self.state_lbl = self.status_bar.addNewLabel('STATE_LBL')
        self.status_bar.addNewButton('Filter')
        self.status_bar.addNewButton('Accept', pressed=self.accept)

        self.node_tab.addWidget(self.node_api_widget)
        self.edge_tab.addWidget(self.edge_api_widget)

        self.mpl_wgt = MatplotlibWidget(parent=self)
        self.mpl_wgt.installEventFilter(self)

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
        bbar2.addNewButton('Infer Cut', pressed=self.infer_cut)

        def refresh_via_cb(flag):
            self.draw_graph()

        self.show_cuts_cb = bbar2.addNewCheckBox(
            'Show Cuts', changed=refresh_via_cb, checked=False)
        self.show_review_cuts_cb =  bbar2.addNewCheckBox(
            'Show Reviewed Cuts', changed=refresh_via_cb, checked=False)
        self.use_image_cb = bbar2.addNewCheckBox(
            'Show Img', changed=refresh_via_cb, checked=use_image)
        self.toggle_pin_cb = bbar2.addNewCheckBox(
            'Pin Positions', changed=self.set_pin_state, checked=False)

        #Debug row
        _simple_button3(self.reset_all)
        _simple_button3(self.infr.reset_name_labels)
        _simple_button3(self.infr.apply_mst)
        _simple_button3(self.infr.apply_scores)
        _simple_button3(self.infr.apply_feedback)
        _simple_button3(self.infr.apply_weights)
        _simple_button3(self.infr.apply_cuts)
        _simple_button3(self.infr.apply_all)
        #bbar3.addNewButton('embed', pressed=self.embed)
        self.status_bar.addNewButton('embed', pressed=self.embed)

        self.mpl_needs_update = True
        self.cb = None
        self.mpl_wgt.click_inside_signal.connect(self.on_click_inside)

        self.signal_graph_update.connect(self.draw_graph, type=Qt.UniqueConnection)
        self.signal_state_update.connect(self.on_state_update)
        self.edge_api_widget.view.doubleClicked.connect(self.edge_doubleclick)
        self.edge_api_widget.view.contextMenuClicked.connect(self.edge_context)
        self.edge_api_widget.view.connect_keypress_to_slot(self.edge_keypress)

        #self.signal_state_update.emit(True)
        self.update_state(structure_changed=True)
        #self.edge_api_widget.resize_headers()

    def update_state(self, structure_changed=False):
        self.infr.apply_feedback()
        self.infr.apply_weights()
        num_names, num_inconsistent = self.infr.connected_compoment_relabel()
        self.num_names_lbl.setText('Names: %d' % (num_names,))
        if num_inconsistent:
            self.state_lbl.setText('Inconsistent Names: %d' % (num_inconsistent,))
            self.state_lbl.setColor('black', self.infr.truth_colors['nonmatch'][0:3] * 255)
        else:
            self.state_lbl.setText('Consistent')
            self.state_lbl.setColor('black', self.infr.truth_colors['match'][0:3] * 255)
        self.infr.apply_cuts()

        self.signal_state_update.emit(structure_changed)

    @QtCore.pyqtSlot(bool)
    def on_state_update(self, structure_changed=False):
        self.mpl_needs_update = True
        #import networkx as nx
        #ccs = list(nx.connected_components(self.infr.graph))
        if structure_changed:
            self.populate_node_model()
            self.populate_edge_model()

    def reset_all(self):
        self.infr.initialize_graph()
        self.infr.reset_feedback()
        #self.populate_node_model()
        #self.populate_edge_model()
        self.toggle_pin_cb.setChecked(False)

    def populate_node_model(self):
        aids = sorted(list(self.infr.graph.nodes()))
        col_name_list = [
            'aid',
            'data',
            'thumb',
            'name_label'
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
            'name_label': lambda node: self.infr.graph.node[node].get('name_label', None)
        }
        col_ider_dict = {
            'thumb': 'aid',
            'data': 'aid',
            'name_label': 'aid',
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

        self.node_tab.setTabText(str('Nodes (%r)' % (self.node_api_widget.model.rowCount())))
        return node_api

    def populate_edge_model(self):
        graph = self.infr.graph
        ibs = self.infr.ibs
        col_name_list = [
            'index',
            'aid1', 'aid2',
            'score',
            'rank',
            'matched', 'reviewed',
            'thumb1', 'thumb2',
            'timedelta',
            'tags',
            'data',
        ]
        def get_edge_data(edge):
            aid1, aid2 = edge
            attrs = graph.get_edge_data(aid1, aid2).copy()
            ut.delete_dict_keys(attrs, [
                'implicit', 'style', 'tail_lp', 'taillabel', 'label', 'lp',
                'headlabel', 'linestyle', 'color', 'stroke', 'lw', 'end_pt',
                'head_lp', 'alpha', 'ctrl_pts', 'pos'])
            attrs = {k: v for k, v in attrs.items() if v is not None}
            return ut.repr2(attrs, precision=2)

        def get_reviewed_status(ibs, aid_pair):
            """ Data role for status column """
            aid1, aid2 = aid_pair
            assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
            assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
            state = ibs.get_annot_pair_is_reviewed([aid1], [aid2])[0]
            state_to_text = {
                None: 'Unreviewed',
                2: 'Auto-reviewed',
                1: 'User-reviewed',
            }
            default = '??? unknown mode %r' % (state,)
            text = state_to_text.get(state, default)
            return text

        def get_match_text(edge):
            aid1, aid2 = edge
            nid1 = graph.node[aid1]['name_label']
            nid2 = graph.node[aid2]['name_label']
            if nid1 == nid2:
                return 'matched'
            else:
                return 'not matched'

        def edge_attr_getter(attr, default=None):
            def get_edge_attr(edge):
                data = graph.get_edge_data(*edge)
                return data.get(attr, default)
            return get_edge_attr

        def get_pair_tags(edge):
            aid1, aid2 = edge
            assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
            assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
            am_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey(
                [aid1], [aid2])
            tag_text = ibs.get_annotmatch_tag_text(am_rowids)[0]
            if tag_text is None:
                tag_text = ''
            return str(tag_text)

        uv_list = list(graph.edges())
        aids1 = ut.take_column(uv_list, 0)
        aids2 = ut.take_column(uv_list, 1)
        col_getter_dict = {
            'index': np.arange(len(aids1)),
            'aid1': np.array(aids1),
            'aid2': np.array(aids2),
            'data': get_edge_data,
            #'thumb': self.infr.ibs.get_annot_chip_thumbtup,
            'thumb1': ibs.get_annot_chip_thumbtup,
            'thumb2': ibs.get_annot_chip_thumbtup,
            'timedelta': lambda edge: ibs.get_unflat_annots_timedelta_list([edge])[0][0],
            'matched':  lambda edge: get_match_text(edge),
            #'reviewed':  lambda edge: get_reviewed_status(ibs, edge),
            'score':  edge_attr_getter('score'),
            'rank':  edge_attr_getter('rank', -1),
            'reviewed':  edge_attr_getter('reviewed_state', 'unreviewed'),
            'tags': lambda edge: get_pair_tags(edge),
        }

        col_ider_dict = {
            'thumb1': 'aid1',
            'thumb2': 'aid2',
            'data': ('aid1', 'aid2'),
            'score': ('aid1', 'aid2'),
            'rank': ('aid1', 'aid2'),
            'timedelta': ('aid1', 'aid2'),
            'matched'     : ('aid1', 'aid2'),
            'reviewed'    : ('aid1', 'aid2'),
            'tags': ('aid1', 'aid2'),
        }
        col_types_dict = {
            'rank': int,
            'score': float,
            'timedelta': float,
            'thumb1': 'PIXMAP',
            'thumb2': 'PIXMAP',
        }

        col_display_role_func_dict = {
            'timedelta': ut.partial(ut.get_posix_timedelta_str, year=True, approx=2),
        }

        def get_match_status_bgrole(ibs, aid_pair):
            """ Background role for status column """
            aid1, aid2 = aid_pair
            #truth = ibs.get_match_truth(aid1, aid2)
            #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
            #truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=0.35)
            nid1 = graph.node[aid1]['name_label']
            nid2 = graph.node[aid2]['name_label']

            color = self.infr.truth_colors['match' if nid1 == nid2 else 'nonmatch']
            #graph.get_edge_data(*aid_pair).get('reviewed_state', 'unreviewed')]
            lighten_amount = .35
            if lighten_amount is not None:
                color = pt.lighten_rgb(color, lighten_amount)
            color = pt.to_base255(color)
            return color

        def get_reviewed_status_bgrole(ibs, aid_pair):
            """ Background role for status column """
            #aid1, aid2 = aid_pair
            #truth = ibs.get_match_truth(aid1, aid2)
            #annotmach_reviewed = ibs.get_annot_pair_is_reviewed([aid1], [aid2])[0]
            color = self.infr.truth_colors[graph.get_edge_data(*aid_pair).get('reviewed_state', 'unreviewed')]
            lighten_amount = .35
            if lighten_amount is not None:
                color = pt.lighten_rgb(color, lighten_amount)
            color = pt.to_base255(color)
            #print('truth_color = %r' % (truth_color,))
            #truth = ibs.get_annot_pair_truth([aid1], [aid2])[0]
            #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
            #if annotmach_reviewed == 0 or annotmach_reviewed is None:
            #    lighten_amount = .9
            #elif annotmach_reviewed == 2:
            #    lighten_amount = .7
            #else:
            #    lighten_amount = .35
            #truth_color = vh.get_truth_color(truth, base255=True,
            #                                 lighten_amount=lighten_amount)
            #truth = ibs.get_match_truth(aid1, aid2)
            #print('get status bgrole: %r truth=%r' % (aid_pair, truth))
            #truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=0.35)
            return color

        col_bgrole_dict = {
            'matched' : ut.partial(get_match_status_bgrole, ibs),
            'reviewed': ut.partial(get_reviewed_status_bgrole, ibs),
        }

        col_width_dict = {
            'index': 42,
            'aid1': 42,
            'aid2': 42,
        }

        edge_api = gt.CustomAPI(
            col_name_list,
            col_ider_dict=col_ider_dict,
            col_types_dict=col_types_dict,
            col_getter_dict=col_getter_dict,
            col_bgrole_dict=col_bgrole_dict,
            col_display_role_func_dict=col_display_role_func_dict,
            col_width_dict=col_width_dict,
            get_thumb_size=lambda: 221,
            sortby='score',
            #sortby='aid1',
            sort_reverse=True
        )
        #api = edge_api

        headers = edge_api.make_headers(tblnice='Edges')
        self.edge_api_widget.change_headers(headers)
        self.edge_api_widget.resize_headers(edge_api)
        self.edge_tab.setTabText(str('Edges (%r)' % (self.edge_api_widget.model.rowCount())))

    @gt.slot_(QtCore.QModelIndex, QtCore.QPoint)
    def edge_context(qres_wgt, qtindex, qpoint):
        print('context')
        pass

    def edge_doubleclick(self, qtindex):
        print('[qres_wgt] _on_doubleclick: ')
        print('[qres_wgt] DoubleClicked: ' + str(gt.qtype.qindexinfo(qtindex)))
        #col = qtindex.column()
        #if qres_wgt.review_api.col_edit_list[col]:
        #    print('do nothing special for editable columns')
        #    return
        #return self.show_match_at_qtindex(qtindex)

    def edge_keypress(self, view, event):
        selected_qtindex_list = view.selectedRows()
        print('selected_qtindex_list = %r' % (selected_qtindex_list,))
        event_key = event.key()

        def aid_pair_gen():
            for qtindex in selected_qtindex_list:
                model = qtindex.model()
                aid1  = model.get_header_data('aid1', qtindex)
                aid2  = model.get_header_data('aid2', qtindex)
                yield aid1, aid2
        if event_key == QtCore.Qt.Key_R:
            print('R')
        elif event_key == QtCore.Qt.Key_T:
            print('T')
            for aid1, aid2 in aid_pair_gen():
                self.infr.add_feedback(aid1, aid2, 'match')
        elif event_key == QtCore.Qt.Key_F:
            print('F')
            for aid1, aid2 in aid_pair_gen():
                self.infr.add_feedback(aid1, aid2, 'nonmatch')
        elif event_key == QtCore.Qt.Key_N:
            print('N')
            for aid1, aid2 in aid_pair_gen():
                self.infr.add_feedback(aid1, aid2, 'notcomp')
        else:
            print('Key  not handled %r' % (event_key,))
            return

        self.update_state()

        for qtindex in selected_qtindex_list:
            # This should work by itself
            self.edge_api_widget.model.dataChanged.emit(qtindex, qtindex)
            # but it doesnt seem to be, but this seems to solve the issue
            self.edge_api_widget.model.layoutChanged.emit()

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

    def set_pin_state(self, flag):
        if flag:
            nx.set_node_attributes(self.infr.graph, 'pin', 'true')
        else:
            ut.nx_delete_node_attr(self.infr.graph, 'pin')

    def infer_cut(self):
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

    def accept(self):
        print('Not done yet')
        infr = self.infr
        graph = infr.graph
        num_names, num_inconsistent = self.infr.connected_compoment_relabel()
        msg = ut.codeblock(
            '''
            Are you sure this is correct?
            #orig_names=%r
            #new_names=%r
            #inconsistent=%r
            ''') % (len(ut.unique(infr.orig_name_labels)), num_names, num_inconsistent)
        if not gt.are_you_sure(msg=msg):
            raise Exception('Cancel')
        node_to_label = nx.get_node_attrs(graph, 'name_label')
        unique_labels = set(node_to_label.values())
        new_names = self.infr.ibs.make_next_name(num_names)
        to_newname = dict(zip(unique_labels, new_names))
        node_to_newname = {node: to_newname[label] for node, label in node_to_label.items()}
        aid_list = list(node_to_newname.keys())
        name_list = list(node_to_newname.values())

        ibs = self.infr.ibs
        if False:
            ibs.set_annot_names(aid_list, name_list)
        else:
            print('DRY RUN. NOT DOING ANYTHING')

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
            color = pt.ensure_nonhex_color(color)
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
        self.infr.update_graph_visual_attributes(
            show_cuts=self.show_cuts_cb.isChecked(),
            show_reviewed_cuts=self.show_review_cuts_cb.isChecked(),
        )
        layoutkw = dict(
            prog='neato',
            #defaultdist=100,
            splines='spline',
            sep=10 / 72,
            #esep=10 / 72
        )
        pt.nx_agraph_layout(graph, inplace=True, **layoutkw)

    def eventFilter(self, source, event):
        #print("___EVENT___")
        #print('event = %r' % (event,))
        #print('source = %r' % (source,))
        if event.type() == QtCore.QEvent.Show:
            if self.mpl_needs_update:
                self.signal_graph_update.emit()
            #print('event = %r' % (event,))
            #print('source = %r' % (source,))
        return super(AnnotGraphWidget, self).eventFilter(source, event)

    @QtCore.pyqtSlot()
    def draw_graph(self):
        self.mpl_needs_update = False
        print('Start draw page')
        self.mpl_wgt.ax.cla()
        self.update_graph_layout()

        # Update Qt things
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
                                   use_image=self.use_image_cb.isChecked(),
                                   verbose=0)
        # self.mpl_wgt.ax.set_aspect('auto')
        self.mpl_wgt.ax.set_aspect('equal')
        # self.mpl_wgt.ax.set_aspect('scaled')

        for aid in self.selected_aids:
            self.highlight_aid(aid, True)

        #ut.util_graph.graph_info(self.infr.graph, verbose=True)
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
        print('self.infr.thresh = %r' % (self.infr.thresh,))

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
        self.cb = cb

    def embed(self):
        fig = self.mpl_wgt.fig  # NOQA
        ax = self.mpl_wgt.ax  # NOQA
        infr = self.infr  # NOQA
        ibs = infr.ibs  # NOQA
        import utool
        utool.embed()


def make_qt_graph_interface(ibs, aids=None, nids=None):
    r"""
    CommandLine:
        python -m ibeis.viz.viz_graph2 make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9
        python -m ibeis.viz.viz_graph2 make_qt_graph_interface --show

        python -m ibeis.viz.viz_graph2 make_qt_graph_interface --db LEWA_splits --nids=1 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph2 import *  # NOQA
        >>> import guitool as gt
        >>> import ibeis
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> aids = ut.get_argval('--aids', type_=list, default=None)
        >>> nids = ut.get_argval('--nids', type_=list, default=None)
        >>> gt.ensure_qtapp()
        >>> win = make_qt_graph_interface(ibs, aids, nids)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=win, freq=10)
    """
    if nids is not None and aids is None:
        aids = ut.flatten(ibs.get_name_aids(nids))
    if aids is None:
        aids = ibs.get_valid_aids()[0:40]
    print('make_qt_graph_interface aids = %r' % (aids,))
    #temp_nids = None
    nids = ibs.get_annot_name_rowids(aids)
    infr = graph_iden.AnnotInference2(ibs, aids, nids)
    infr.initialize_graph()

    infr.remove_name_labels()
    infr.apply_scores()
    #infr.apply_feedback()
    #infr.apply_weights()
    #infr.connected_compoment_relabel()
    #infr.apply_cuts()

    # infr.apply_mst()
    #infr.apply_weights()
    #infr.apply_cuts()
    if ut.get_argflag('--cut'):
        infr.apply_all()
    gt.ensure_qtapp()
    print('infr = %r' % (infr,))
    win = AnnotGraphWidget(infr=infr, use_image=False)
    win.resize(900, 600)
    #win.draw_graph()
    win.show()
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
