# -*- coding: utf-8 -*-
"""
CommandLine:
    ibeis make_qt_graph_interface --show
    ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9
    ibeis make_qt_graph_interface --show --aids=1,4,5,6,8,9
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import vtool as vt
import numpy as np
import networkx as nx
import itertools
from ibeis.algo.hots import graph_iden
import guitool as gt
import plottool as pt
from plottool import abstract_interaction
from guitool.__PYQT__.QtCore import Qt
from guitool.__PYQT__ import QtCore, QtWidgets  # NOQA
from plottool import interact_helpers as ih
from matplotlib.backend_bases import MouseEvent

from guitool import __PYQT__
if __PYQT__._internal.GUITOOL_PYQT_VERSION == 4:
    import matplotlib.backends.backend_qt4agg as backend_qt
else:
    import matplotlib.backends.backend_qt5agg as backend_qt
FigureCanvas = backend_qt.FigureCanvasQTAgg


GRAPH_REVIEW_CFG_DEFAULTS = {
    'ranks_top': 3,
    'ranks_bot': 2,

    'filter_reviewed': True,
    'filter_photobombs': False,

    'filter_true_matches': True,
    'filter_false_matches': False,

    'filter_nonmatch_between_ccs': True,
    'filter_dup_namepairs': True,

    'show_match_thumb': True,
}


class MatplotlibWidget(gt.GuitoolWidget):
    click_inside_signal = QtCore.pyqtSignal(MouseEvent, object)

    def initialize(self):
        from plottool.interactions import zoom_factory, pan_factory
        self.fig = pt.plt.figure()
        self.fig._no_raise_plottool = True
        self.canvas = FigureCanvas(self.fig)

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.addWidget(self.canvas)

        self.pan_events = pan_factory(self.ax)
        self.zoon_events = zoom_factory(self.ax)
        ih.connect_callback(self.fig, 'button_press_event', self.on_click)

        self.MOUSE_BUTTONS = abstract_interaction.AbstractInteraction.MOUSE_BUTTONS
        self.setMinimumHeight(20)
        self.setMinimumWidth(20)

    def on_click(self, event):
        if ih.clicked_inside_axis(event):
            ax = event.inaxes
            self.click_inside_signal.emit(event, ax)


class DevGraphWidget(gt.GuitoolWidget):

    def initialize(graph_widget, use_image, self_parent):
        graph_widget.self_parent = parent = self_parent
        #parent = graph_widget.parent()
        infr = parent.infr
        graph_widget.plotinfo = None
        graph_widget.infr = infr

        graph_widget.mpl_needs_update = True
        graph_widget.cb = None

        graph_widget.selected_aids = []
        graph_widget.splitter = graph_widget.addNewSplitter(
            orientation=Qt.Horizontal)
        graph_widget.ctrls_ = graph_widget.splitter.addNewWidget(
            orientation=Qt.Vertical, vertical_stretch=1, margin=1, spacing=1)
        graph_widget.ctrls = graph_widget.ctrls_.addNewScrollArea()

        graph_widget.mpl_wgt = MatplotlibWidget(parent=graph_widget)
        graph_widget.mpl_wgt.installEventFilter(graph_widget)
        graph_widget.splitter.addWidget(graph_widget.mpl_wgt)

        ctrls = graph_widget.ctrls
        bbar1 = ctrls.addNewWidget(ori='vert', margin=1, spacing=1)
        bbar2 = ctrls.addNewWidget(ori='vert', margin=1, spacing=1)

        bbar1.addNewButton('Mark: Match', pressed=graph_widget.mark_match)
        bbar1.addNewButton('Mark: Non-Match', pressed=graph_widget.mark_nonmatch)
        bbar1.addNewButton('Mark: Not-Comp', pressed=graph_widget.mark_notcomp)

        bbar2.addNewButton('Deselect', pressed=graph_widget.deselect)
        bbar2.addNewButton('Show Annots', pressed=graph_widget.show_selected)

        def refresh_via_cb(flag):
            graph_widget.on_state_update()

        graph_widget.show_cuts_cb = bbar2.addNewCheckBox(
            'Show Cuts', changed=refresh_via_cb, checked=False)
        graph_widget.show_review_cuts_cb =  bbar2.addNewCheckBox(
            'Show Reviewed Cuts', changed=refresh_via_cb, checked=True)
        graph_widget.only_reviwed_cb =  bbar2.addNewCheckBox(
            'Only Reviewed', changed=refresh_via_cb, checked=True)
        graph_widget.use_image_cb = bbar2.addNewCheckBox(
            'Show Img', changed=refresh_via_cb, checked=use_image)
        graph_widget.toggle_pin_cb = bbar2.addNewCheckBox(
            'Pin Positions', changed=graph_widget.set_pin_state, checked=False)

        # Connect signals and slots
        graph_widget.mpl_wgt.click_inside_signal.connect(graph_widget.on_click_inside)

    def on_state_update(graph_widget):
        if graph_widget.mpl_wgt is None or graph_widget.mpl_wgt.visibleRegion().isEmpty():
            # Flag that graph should draw next time it is visible
            graph_widget.mpl_needs_update = True
        else:
            # Draw the graph because it is visible
            graph_widget.draw_graph()

    def draw_graph(graph_widget):
        graph_widget.mpl_needs_update = False
        print('[graph] Start draw page')
        graph_widget.mpl_wgt.ax.cla()

        graph_widget.infr.update_visual_attrs(
            show_cuts=graph_widget.show_cuts_cb.isChecked(),
            show_reviewed_cuts=graph_widget.show_review_cuts_cb.isChecked(),
            only_reviewed=graph_widget.only_reviwed_cb.isChecked(),
        )

        try:
            graph_widget.plotinfo = pt.show_nx(
                graph_widget.infr.graph, layout='custom', as_directed=False,
                ax=graph_widget.mpl_wgt.ax,
                use_image=graph_widget.use_image_cb.isChecked(), verbose=0)
        except IOError:
            graph_widget.infr.initialize_visual_node_attrs()
            graph_widget.plotinfo = pt.show_nx(
                graph_widget.infr.graph, layout='custom', as_directed=False,
                ax=graph_widget.mpl_wgt.ax,
                use_image=graph_widget.use_image_cb.isChecked(), verbose=0)
        graph_widget.mpl_wgt.ax.set_aspect('equal')

        for aid in graph_widget.selected_aids:
            graph_widget.highlight_aid(aid, True)

        graph_widget.mpl_wgt.canvas.draw()
        # fig.canvas.blit(ax.bbox)
        graph_widget.mpl_wgt.fig.subplots_adjust(left=.02, top=.98, bottom=.02,
                                                 right=.85)
        print('[graph] End draw page')

    def on_click_inside(graph_widget, event, ax):
        pos = graph_widget.plotinfo['node']['pos']
        nodes = list(pos.keys())
        pos_list = ut.dict_take(pos, nodes)

        # TODO: FIXME
        #x = 10
        #y = 10
        x, y = event.xdata, event.ydata
        point = np.array([x, y])
        pos_list = np.array(pos_list)
        index, dist = vt.closest_point(point, pos_list, distfunc=vt.L2)
        node = nodes[index]
        aid = graph_widget.infr.node_to_aid[node]
        context_shown = False

        CHECK_PAIR = True
        if CHECK_PAIR:
            if event.button == 3 and not context_shown:
                if len(graph_widget.selected_aids) != 2:
                    print('This funciton only work if exactly 2 are selected')
                else:
                    from ibeis.gui import inspect_gui
                    context_shown = True
                    aid1, aid2 = (graph_widget.selected_aids)
                    qres = None
                    qreq_ = None
                    options = inspect_gui.get_aidpair_context_menu_options(
                        graph_widget.infr.ibs, aid1, aid2, qres, qreq_=qreq_)
                    graph_widget.show_popup_menu(options, event)

        bbox = vt.bbox_from_center_wh(graph_widget.plotinfo['node']['pos'][node],
                                      graph_widget.plotinfo['node']['size'][node])
        SELECT_ANNOT = vt.point_inside_bbox(point, bbox)

        if SELECT_ANNOT:
            #print(ut.obj_str(ibs.get_annot_info(aid, default=True,
            #                                    name=False, gname=False)))
            if event.button == 1:
                graph_widget.toggle_selected_aid(aid)

            if event.button == 3 and not context_shown:
                # right click
                from ibeis.viz.interact import interact_chip
                context_shown = True
                #refresh_func = functools.partial(viz.show_name, ibs, nid,
                #fnum=fnum, sel_aids=sel_aids)
                refresh_func = None
                config2_ = None
                options = interact_chip.build_annot_context_options(
                    graph_widget.infr.ibs, aid, refresh_func=refresh_func,
                    with_interact_name=False,
                    config2_=config2_)
                graph_widget.show_popup_menu(options, event)

    def show_popup_menu(graph_widget, options, event):
        """
        context menu
        """
        height = graph_widget.mpl_wgt.fig.canvas.geometry().height()
        qpoint = gt.newQPoint(event.x, height - event.y)
        qwin = graph_widget.mpl_wgt.fig.canvas
        gt.popup_menu(qwin, qpoint, options)

    def set_pin_state(graph_widget, flag):
        if flag:
            nx.set_node_attributes(graph_widget.infr.graph, 'pin', 'true')
        else:
            ut.nx_delete_node_attr(graph_widget.infr.graph, 'pin')

    def mark_nonmatch(graph_widget):
        print('BREAK LINK graph_widget.selected_aids = %r' % (graph_widget.selected_aids,))
        for aid1, aid2 in itertools.combinations(graph_widget.selected_aids, 2):
            graph_widget.infr.add_feedback(aid1, aid2, 'nomatch')
        # TODO: just use signal / slot
        #graph_widget.infr.apply_feedback_edges()
        graph_widget.self_parent.update_state()
        #graph_widget.on_state_update()

    def mark_match(graph_widget):
        print('MAKE LINK graph_widget.selected_aids = %r' % (graph_widget.selected_aids,))
        for aid1, aid2 in itertools.combinations(graph_widget.selected_aids, 2):
            graph_widget.infr.add_feedback(aid1, aid2, 'match')
        # TODO: just use signal / slot
        graph_widget.self_parent.update_state()
        #graph_widget.infr.apply_feedback_edges()
        #graph_widget.on_state_update()

    def mark_notcomp(graph_widget):
        print('MAKE LINK graph_widget.selected_aids = %r' % (graph_widget.selected_aids,))
        for aid1, aid2 in itertools.combinations(graph_widget.selected_aids, 2):
            graph_widget.infr.add_feedback(aid1, aid2, 'notcomp')
        # TODO: just use signal / slot
        graph_widget.self_parent.update_state()
        #graph_widget.infr.apply_feedback_edges()
        #graph_widget.on_state_update()

    def show_selected(graph_widget):
        # TODO: move to mpl widget
        print('[graph] show_selected')
        from ibeis.viz import viz_chip
        fnum = pt.ensure_fnum(10)
        print('fnum = %r' % (fnum,))
        fig = pt.figure(fnum=fnum)
        viz_chip.show_many_chips(graph_widget.infr.ibs, graph_widget.selected_aids, fnum=fnum)
        #fig.canvas.update()
        fig.show()
        fig.canvas.draw()
        graph_widget._figscope = fig

    def infer_cut(graph_widget):
        infrkw = {}
        # keys = ['min_labels', 'max_labels']
        # infrkw = ut.dict_subset(graph_widget.config, keys)
        graph_widget.infr.infer_cut(**infrkw)
        graph_widget.on_state_update()

    def highlight_aid(graph_widget, aid, color=None):
        # TODO: move to mpl widget
        if graph_widget.plotinfo is None:
            return
        node = graph_widget.infr.aid_to_node[aid]
        frame = graph_widget.plotinfo['patch_frame_dict'][node]
        framewidth = graph_widget.infr.graph.node[node]['framewidth']
        if color is True:
            color = pt.ORANGE
        if color is None or color is False:
            color = pt.DARK_BLUE
            color = graph_widget.infr.graph.node[node]['color']
            color = pt.ensure_nonhex_color(color)
            frame.set_linewidth(framewidth)
        else:
            frame.set_linewidth(framewidth * 2)
        frame.set_facecolor(color)
        frame.set_edgecolor(color)

    def deselect(graph_widget):
        print('graph_widget.selected_aids = %r' % (graph_widget.selected_aids,))
        graph_widget.toggle_selected_aid(graph_widget.selected_aids[:])

    def toggle_selected_aid(graph_widget, aids):
        for aid in ut.ensure_iterable(aids):
            # TODO: move to mpl widget
            if aid in graph_widget.selected_aids:
                graph_widget.selected_aids.remove(aid)
                #graph_widget.highlight_aid(aid, pt.WHITE)
                graph_widget.highlight_aid(aid, color=None)
            else:
                graph_widget.selected_aids.append(aid)
                graph_widget.highlight_aid(aid, True)
        print('graph_widget.selected_aids = %r' % (graph_widget.selected_aids,))
        if graph_widget.mpl_wgt is not None:
            graph_widget.mpl_wgt.fig.canvas.draw()
        #graph_widget.on_state_update()

    def eventFilter(graph_widget, source, event):
        if event.type() == QtCore.QEvent.Show:
            if graph_widget.mpl_needs_update:
                graph_widget.on_state_update()
        return super(DevGraphWidget, graph_widget).eventFilter(source, event)


class AnnotGraphWidget(gt.GuitoolWidget):
    #signal_state_update = QtCore.pyqtSignal(bool)

    def initialize(self, infr=None, use_image=False, init_mode='rereview',
                   review_cfg=None):
        print('[graph] initialize')

        self.pcfg = {
            'can_match_samename': True,
            'K': 3,
            'Knorm': 3,
            'prescore_method': 'csum',
            'score_method': 'csum'
        }

        self.init_mode = init_mode
        print('self.init_mode = %r' % (self.init_mode,))

        if review_cfg is None:
            if self.init_mode is None:
                self.preset_unfiltered_config()
            elif self.init_mode == 'split':
                self.preset_filtered_config()
            elif self.init_mode == 'rereview':
                self.preset_unfiltered_config()
            elif self.init_mode == 'review':
                self.preset_unfiltered_config()
            else:
                raise ValueError('Unknown mode = %r' % (self.init_mode,))

        self.infr = infr
        # self.config = InferenceConfig()

        self.menubar = gt.newMenubar(self)
        self.menuFile = self.menubar.newMenu('Dev')
        self.menuFile.newAction(triggered=self.print_info)
        self.menuFile.newAction(triggered=self.embed)

        graph_tables_widget = self.addNewTabWidget(verticalStretch=1)

        self.status_bar = self.addNewWidget(
            orientation=Qt.Horizontal, vertical_stretch=1, margin=1, spacing=1)
        self.prog_bar = self.addNewProgressBar(visible=False)

        self.edge_tab = graph_tables_widget.addNewTab('Edges')
        self.node_tab = graph_tables_widget.addNewTab('Nodes')

        self.node_api_widget = gt.APIItemWidget()
        self.edge_api_widget = gt.APIItemWidget()

        self.edge_api_widget.view.doubleClicked.connect(self.edge_doubleclick)
        self.edge_api_widget.view.contextMenuClicked.connect(self.edge_context)
        self.edge_api_widget.view.connect_keypress_to_slot(self.edge_keypress)
        self.edge_api_widget.view.connect_single_key_to_slot(gt.ALT_KEY, self.on_alt_pressed)

        self.status_bar.addNewButton('Reset Rereview', pressed=self.reset_rereview)
        self.status_bar.addNewButton('Reset Empty', pressed=self.reset_empty)

        self.num_names_lbl = self.status_bar.addNewLabel('NUM_NAMES_LBL')
        self.state_lbl = self.status_bar.addNewLabel('STATE_LBL')

        self.status_bar.addNewButton('Score Edges', pressed=self.score_edges)
        self.status_bar.addNewButton('Edit Filters', pressed=self.edit_filters)
        self.status_bar.addNewButton('Repopulate', pressed=self.repopulate)
        self.status_bar.addNewButton('Accept', pressed=self.accept)

        self.node_tab.addWidget(self.node_api_widget)
        self.edge_tab.addWidget(self.edge_api_widget)

        if self.init_mode in ['split', 'rereview', 'review']:
            # TODO: separate graph view into its own class
            self.graph_tab = graph_tables_widget.addNewTab('Graph')
            # TODO: make this its own proper widget
            self.graph_widget = DevGraphWidget(parent=self, self_parent=self,
                                               use_image=use_image)
            self.graph_tab.addWidget(self.graph_widget)
        else:
            self.graph_widget = None
            self.graph_tab = None

        #self.signal_state_update.connect(self.on_state_update)

    def preset_unfiltered_config(self):
        self.review_cfg = GRAPH_REVIEW_CFG_DEFAULTS.copy()
        for key in self.review_cfg.keys():
            if key.startswith('filter_'):
                self.review_cfg[key] = False

    def preset_filtered_config(self):
        self.review_cfg = GRAPH_REVIEW_CFG_DEFAULTS.copy()

    def showEvent(self, event):
        super(AnnotGraphWidget, self).showEvent(event)
        print('[graph] showEvent')
        # Fire initialize event after we show the GUI
        QtCore.QTimer.singleShot(50, self.init_inference)

    def init_inference(self):
        print('[graph] init_inference')
        if self.init_mode is None:
            pass
        elif self.init_mode == 'split':
            self.preset_filtered_config()
            self.reset_split()
        elif self.init_mode == 'rereview':
            self.preset_unfiltered_config()
            self.reset_rereview()
        elif self.init_mode == 'review':
            self.reset_review()
        else:
            raise ValueError('Unknown init_mode=%r' % (self.init_mode,))
        #self.apply_scores()
        self.update_state(structure_changed=True)

    def update_state(self, structure_changed=False):
        print('[graph] update_state mode=%s' % (self.init_mode,))
        #if self.init_mode in ['split', 'rereview']:
        if self.init_mode == 'split':
            self.infr.apply_feedback_edges()
            if structure_changed:
                # FIXME: when should score be reapplied?
                # This should happen in split mode, but not None mode
                self.apply_scores()
            self.infr.apply_weights()
            self.infr.connected_component_reviewed_relabel()
            self.infr.apply_cuts()
        elif self.init_mode == 'rereview':
            self.infr.apply_feedback_edges()
            self.infr.apply_match_scores()
            self.infr.apply_weights()
            self.infr.connected_component_reviewed_relabel()
            self.infr.apply_cuts()
        elif self.init_mode == 'review':
            self.infr.apply_match_edges()
            self.infr.apply_feedback_edges()
            self.infr.apply_match_scores()
            self.infr.apply_weights()
            # self.infr.connected_component_reviewed_relabel()
            # self.infr.apply_cuts()

        # Set gui status indicators
        status = self.infr.connected_component_status()
        if status['num_inconsistent']:
            self.state_lbl.setText('Inconsistent Names: %d' % (status['num_inconsistent'],))
            self.state_lbl.setColor('black', self.infr.truth_colors['nomatch'][0:3] * 255)
        else:
            self.state_lbl.setText('Consistent')
            self.state_lbl.setColor('black', self.infr.truth_colors['match'][0:3] * 255)

        self.num_names_lbl.setText('Names: max=%r, ~min=%r' % (
            status['num_names_max'],
            status['num_names_min'],))

        #self.signal_state_update.emit(structure_changed)
        self.on_state_update(structure_changed)
        self.edge_api_widget.model.layoutChanged.emit()
        self.node_api_widget.model.layoutChanged.emit()

    @QtCore.pyqtSlot(bool)
    def on_state_update(self, structure_changed=False):
        print('[graph] on_update_state mode=%s' % (self.init_mode,))
        if structure_changed:
            self.populate_node_model()
            self.populate_edge_model()
        if self.graph_widget is not None:
            self.graph_widget.on_state_update()

    def apply_scores(self):
        with gt.GuiProgContext('Computing Matches', self.prog_bar) as ctx:
            self.infr.exec_matching(prog_hook=ctx.prog_hook)
            self.infr.apply_match_edges(self.review_cfg)
            self.infr.apply_match_scores()

    def score_edges(self):
        with gt.GuiProgContext('Scoring Edges', self.prog_bar) as ctx:
            self.infr.exec_matching(prog_hook=ctx.prog_hook)
            self.infr.apply_match_scores()
        self.repopulate()

    def reset_review(self):
        print('[graph] reset_review')
        infr = self.infr
        with gt.GuiProgContext('Reset Review', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            infr.initialize_graph()
            infr.apply_feedback_edges()
            infr.apply_match_edges()
            infr.apply_match_scores()
            ctx.set_progress(2, 3)
            infr.initialize_visual_node_attrs()
            self.repopulate()
            ctx.set_progress(3, 3)

    def reset_rereview(self):
        """
        Goal:
            All names are removed.
            Reset edges so only reviewed edges are shown.
            You can change the state of those edges.
            They are not filtered.
        """
        print('[graph] reset_rereview')
        infr = self.infr
        self.init_mode = 'rereview'
        with gt.GuiProgContext('Reset Review', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            infr.initialize_graph()
            #ctx.set_progress(1, 3)
            infr.reset_name_labels()
            infr.reset_feedback()
            infr.apply_feedback_edges()
            #infr.mst_review()
            infr.remove_name_labels()
            infr.apply_match_scores()
            ctx.set_progress(2, 3)
            infr.initialize_visual_node_attrs()
            self.repopulate()
            ctx.set_progress(3, 3)

    def reset_split(self):
        infr = self.infr
        self.init_mode = 'split'
        with gt.GuiProgContext('Initializing', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            infr.initialize_graph()
            ctx.set_progress(1, 3)
            infr.remove_feedback()
            ctx.set_progress(2, 3)
            infr.remove_name_labels()
            # if ut.get_argflag('--cut'):
            #     infr.apply_all()
            infr.initialize_visual_node_attrs()
            ctx.set_progress(3, 3)
        pass

    def reset_empty(self):
        print('[graph] reset_empty')
        infr = self.infr
        self.init_mode = 'split'
        with gt.GuiProgContext('Reset Empty', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            infr.remove_feedback()
            infr.remove_name_labels()
            infr.apply_cuts()
            self.repopulate()
            ctx.set_progress(3, 3)

    def edit_filters(self):
        import dtool
        # TODO: split up review configs / show thumbs etc...
        config = dtool.Config.from_dict(self.review_cfg)
        dlg = gt.ConfigConfirmWidget.as_dialog(self, title='Edit Filters',
                                               msg='Edit Filters',
                                               with_spoiler=False,
                                               config=config)
        dlg.resize(700, 500)
        dlg.exec_()
        print('config = %r' % (config,))
        updated_config = dlg.widget.config  # NOQA
        print('updated_config = %r' % (updated_config,))
        self.review_cfg = updated_config.asdict()
        self.repopulate()

    def repopulate(self):
        self.update_state(structure_changed=True)

    def populate_node_model(self):
        print('[graph] populate_node_model')
        node_api = make_node_api(self.infr)
        headers = node_api.make_headers(tblnice='Nodes')
        self.node_api_widget.change_headers(headers)
        self.node_api_widget.view.verticalHeader().setVisible(True)
        try:
            self.node_api_widget.view.verticalHeader().setMovable(True)
        except AttributeError:
            self.node_api_widget.view.verticalHeader().setSectionsMovable(True)

        self.node_tab.setTabText('Nodes (%r)' % (self.node_api_widget.model.num_rows_total))
        return node_api

    def populate_edge_model(self):
        print('[graph] populate_edge_model')
        if self.init_mode is None:
            self.review_cfg['show_match_thumb'] = False

        edge_api = make_edge_api(self.infr, review_cfg=self.review_cfg)
        headers = edge_api.make_headers(tblnice='Edges')
        self.edge_api_widget.change_headers(headers)
        self.edge_api_widget.resize_headers(edge_api)
        self.edge_api_widget.view.verticalHeader().setVisible(True)
        self.edge_tab.setTabText('Edges (%r)' % (self.edge_api_widget.model.num_rows_total))
        self.edge_api_widget.view.verticalHeader().setDefaultSectionSize(221)

    def edge_doubleclick(self, qtindex):
        """
        qtindex = qtindex = self.edge_api_widget.view.get_row_and_qtindex_from_id(1)[0]
        """
        print('[graph] _on_doubleclick: ')
        print('[graph] DoubleClicked: ' + str(gt.qtype.qindexinfo(qtindex)))
        model = qtindex.model()
        aid1  = model.get_header_data('aid1', qtindex)
        aid2  = model.get_header_data('aid2', qtindex)
        cm, aid1, aid2 = self.infr.lookup_cm(aid1, aid2)
        if cm is not None:
            cm.ishow_single_annotmatch(self.infr.qreq_, aid2, mode=0)
        else:
            # Hack
            self.graph_widget.deselect()
            self.graph_widget.toggle_selected_aid([aid1, aid2])
            #self.graph_widget.selected_aids = [aid1, aid2]
            self.graph_widget.show_selected()

    def get_edge_options(self):
        view = self.edge_api_widget.view
        selected_qtindex_list = view.selectedRows()

        def _mark_pairs(state):
            #for aid1, aid2 in aid_pair_gen():
            for qtindex in selected_qtindex_list:
                model = qtindex.model()
                aid1  = model.get_header_data('aid1', qtindex)
                aid2  = model.get_header_data('aid2', qtindex)
                self.infr.add_feedback(aid1, aid2, state)
            self.update_state()

        options = [
            ('Mark &True', lambda: _mark_pairs('match')),
            ('Mark &False', lambda: _mark_pairs('nomatch')),
            ('Mark &Non-Comparable', lambda: _mark_pairs('notcomp')),
            ('&Unreview', lambda: _mark_pairs('unreviewed')),
        ]

        if len(selected_qtindex_list) == 1:
            from ibeis.gui import inspect_gui
            ibs = self.infr.ibs
            qtindex = selected_qtindex_list[0]
            model = qtindex.model()
            aid1  = model.get_header_data('aid1', qtindex)
            aid2  = model.get_header_data('aid2', qtindex)
            qreq_ = self.infr.qreq_
            pair_tag_options = inspect_gui.make_aidpair_tag_context_options(ibs, aid1, aid2)
            chip_context_options = inspect_gui.make_annotpair_context_options(
                ibs, aid1, aid2, qreq_=qreq_)
            options += [
                ('Match Ta&gs', pair_tag_options)
            ]
            options += chip_context_options

            if True or self.init_mode != 'split':
                from ibeis.viz import viz_graph2
                nids = ut.unique(ibs.get_annot_nids([aid1, aid2]))
                options += [
                    ('New Split Case Interaction',
                     ut.partial(viz_graph2.make_qt_graph_interface, ibs,
                                nids=nids)),
                ]

            options += [
                ('VsOne', inspect_gui.make_vsone_context_options(ibs, aid1, aid2, qreq_))
            ]

        return options

    @gt.slot_(QtCore.QModelIndex, QtCore.QPoint)
    def edge_context(self, qtindex, qpoint):
        print('context')
        #print('option_dict = %s' % (ut.repr3(option_dict, nl=2),))
        options = self.get_edge_options()
        gt.popup_menu(self, qpoint, options)

    def on_alt_pressed(self, view, event):
        selected_qtindex_list = view.selectedRows()
        if len(selected_qtindex_list) > 0:
            # popup context menu on alt
            qtindex = selected_qtindex_list[-1]
            qrect = view.visualRect(qtindex)
            pos = qrect.center()
            self.edge_context(qtindex, pos)

    def edge_keypress(self, view, event):
        """
        view = self.edge_api_widget.view
        """
        event_key = event.key()
        options = self.get_edge_options()
        option_dict = gt.make_option_dict(options, shortcuts=True)
        handled = False
        for key, func in option_dict.items():
            if event_key == getattr(QtCore.Qt, 'Key_' + key.upper()):
                func()
                handled = True
                break
        if not handled:
            print('Key  not handled %r' % (event_key,))
            return

    def accept(self):
        print('[graph] Not done yet')
        infr = self.infr
        graph = infr.graph
        num_names, num_inconsistent = self.infr.connected_component_reviewed_relabel()
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
        node_to_newname = {node: to_newname[name_label]
                           for node, name_label in node_to_label.items()}
        aid_list = list(node_to_newname.keys())
        name_list = list(node_to_newname.values())
        #print('aid_list = %r' % (aid_list,))
        #print('name_list = %r' % (name_list,))

        # LOG ACTIVITY
        import logging
        # ut.vd(review_log_dir)
        # create logger with 'spam_application'
        logger = logging.getLogger('query_review')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # create file handler which logs even debug messages
        dbdir = self.infr.ibs.get_dbdir()
        expt_dir = ut.ensuredir(ut.unixjoin(dbdir, 'SPECIAL_GGR_EXPT_LOGS'))
        review_log_dir = ut.ensuredir(ut.unixjoin(expt_dir, 'review_logs'))
        log_fpath = ut.unixjoin(review_log_dir,
                                'split_log_%s.json' % (self.infr.ibs.dbname))
        fh = logging.FileHandler(log_fpath)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        self.logger = logger
        logger.info('=================')
        logger.info(msg)
        logger.info('ACCEPT SPLIT CASE')
        logger.info('aid_list = %r' % (aid_list,))
        logger.info('name_list = %r' % (name_list,))
        # logger.info('_initial_feedback = ' + ut.repr2(self.infr._initial_feedback, nl=1))
        logger.info('user_feedback = ' + ut.repr2(self.infr.user_feedback, nl=1))

        # keep track of residual data
        new_df, old_df = infr.match_state_delta()
        num_added = len(old_df) - len(new_df)
        num_changed = len(old_df)

        pdkw = dict(max_rows=len(new_df) + 1)
        #print_ = print
        print_ = logger.info
        print_('There were %d added annot match rows' % (num_added,))
        print_('There were %d changed annot match rows' % (num_changed,))
        print_('---DATAFRAME\nold_df =\n' + old_df.to_string(**pdkw))
        print_('---DATAFRAME\nnew_df =\n' + new_df.to_string(**pdkw))

        ibs = self.infr.ibs
        dryrun = False
        if not dryrun:
            # Set names
            ibs.set_annot_names(aid_list, name_list)

            # Add am rowids for nonexisting rows
            if len(new_df) > 0:
                is_add = np.isnan(new_df['am_rowid'].values)
                add_df = new_df.loc[is_add]
                add_ams = ibs.add_annotmatch_undirected(add_df['aid1'].values,
                                                        add_df['aid2'].values)
                new_df.loc[is_add, 'am_rowid'] = add_ams
                new_df.set_index('am_rowid', drop=False, inplace=True)

                # Set residual matching data
                truth_options = [ibs.const.TRUTH_MATCH,
                                 ibs.const.TRUTH_NOT_MATCH,
                                 ibs.const.TRUTH_UNKNOWN]
                truth_keys = ['p_match', 'p_nomatch', 'p_notcomp']
                truth_idxs = new_df[truth_keys].values.argmax(axis=1)
                new_truth = ut.take(truth_options, truth_idxs)
                am_rowids = new_df['am_rowid'].values

                ibs.set_annotmatch_truth(am_rowids, new_truth)
        else:
            print('DRY RUN. NOT DOING ANYTHING')
        gt.user_info(self, 'Name Change Complete')

    def sizeHint(self):
        return QtCore.QSize(1100, 500)

    def closeEvent(self, event):
        #event.accept()
        abstract_interaction.unregister_interaction(self)
        super(AnnotGraphWidget, self).closeEvent(event)

    def print_info(self):
        print('[graph] print_info')
        #print('_initial_feedback = ' + ut.repr2(self.infr._initial_feedback, nl=1))
        print('user_feedback = ' + ut.repr2(self.infr.user_feedback, nl=1))
        infr = self.infr
        print('infr = %r' % (infr,))
        if infr is not None and infr.graph is not None:
            print(ut.repr3(ut.graph_info(infr.graph)))

    def embed(self):
        infr = self.infr  # NOQA
        ibs = infr.ibs  # NOQA
        graph = infr.graph  # NOQA
        import utool
        utool.embed()


def make_node_api(infr):
    aids = sorted(list(infr.graph.nodes()))
    col_name_list = [
        'aid',
        #'data',
        'thumb',
        'name_label'
    ]
    #if not DEVELOPER_MODE:
    #    col_name_list.remove('data')

    def get_node_data(aid):
        data = infr.graph.node[aid].copy()
        ut.delete_dict_keys(data, infr.visual_node_attrs)
        return ut.repr2(data, precision=2)
    col_getter_dict = {
        'aid': np.array(aids),
        'data': get_node_data,
        'thumb': infr.ibs.get_annot_chip_thumbtup,
        'name_label': lambda node: infr.graph.node[node].get('name_label', None)
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
    return node_api


def make_edge_api(infr, review_cfg={}):
    """
    TODO:
        mark an edge that would cause an inconsistency
        to be marked as a deeper red.
        These are edges we currently believe to be false
        By default edges should not be red. they should be a light unknown yellow.
        Dark unknown yellow is for noncomparable annotations

    """
    graph = infr.graph
    ibs = infr.ibs

    def get_edge_data(edge):
        aid1, aid2 = edge
        attrs = graph.get_edge_data(aid1, aid2).copy()
        remove_attrs = infr.visual_edge_attrs + ['rank', 'reviewed_state', 'score']
        try:
            remove_attrs.remove('style')
        except ValueError:
            pass
        ut.delete_dict_keys(attrs, remove_attrs)
        attrs = {k: v for k, v in attrs.items() if v is not None}
        #attrs['name_label1'] = graph.node[aid1]['name_label']
        #attrs['name_label2'] = graph.node[aid2]['name_label']
        return ut.repr2(attrs, precision=2, explicit=True, nobr=True)

    #def get_reviewed_status(ibs, edge):
    #    """ Data role for status column """
    #    aid1, aid2 = edge
    #    assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
    #    assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
    #    state = ibs.get_annot_pair_is_reviewed([aid1], [aid2])[0]
    #    state_to_text = {
    #        None: 'Unreviewed',
    #        2: 'Auto-reviewed',
    #        1: 'User-reviewed',
    #    }
    #    default = '??? unknown mode %r' % (state,)
    #    text = state_to_text.get(state, default)
    #    return text

    def get_match_text(edge):
        aid1, aid2 = edge
        assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
        assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
        nid1 = graph.node[aid1]['name_label']
        nid2 = graph.node[aid2]['name_label']
        if nid1 == nid2:
            return 'matched nid=%d' % (nid1,)
        else:
            return 'not matched nids=(%d,%d)' % (nid1, nid2)

    def get_match_status_bgrole(ibs, edge):
        """ Background role for status column """
        aid1, aid2 = edge
        nid1 = graph.node[aid1]['name_label']
        nid2 = graph.node[aid2]['name_label']

        lighten_amount = .35
        state = edge_attr_getter('reviewed_state', 'unreviewed')(edge)
        if state == 'unreviewed':
            lighten_amount = .7

        color = infr.truth_colors['match' if nid1 == nid2 else 'nomatch']
        #graph.get_edge_data(*edge).get('reviewed_state', 'unreviewed')]
        if lighten_amount is not None:
            color = pt.lighten_rgb(color, lighten_amount)
        color = pt.to_base255(color)
        return color

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

    def edge_assert(edge):
        aid1, aid2 = edge
        assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
        assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)

    def get_reviewed_status_bgrole(ibs, edge):
        """ Background role for status column """
        data = graph.get_edge_data(*edge)
        state = data.get('reviewed_state', 'unreviewed')
        color = infr.truth_colors[state]
        lighten_amount = .35
        if state == 'unreviewed':
            lighten_amount = .7
        if lighten_amount is not None:
            color = pt.lighten_rgb(color, lighten_amount)
        color = pt.to_base255(color)
        return color

    def get_match_thumbtup(edge, thumbsize=None):
        #sibs, qaid2_cm, qaids, daids, index, qreq_=None,
        #                   thumbsize=(128, 128), match_thumbtup_cache={}):
        aid1, aid2 = edge
        cm, aid1, aid2 = infr.lookup_cm(aid1, aid2)
        if cm is None:
            return None
        #assert cm.qaid == aid1, 'aids do not aggree'
        # Hacky new way of drawing
        from ibeis.gui import id_review_api
        fpath, func, func2 = id_review_api.make_ensure_match_img_nosql_func(
            infr.qreq_, cm, aid2)
        thumbdat = {
            'fpath': fpath,
            'thread_func': func,
            'main_func': func2,
        }
        return thumbdat

    def get_num_other(aid):
        node = infr.aid_to_node[aid]
        node_to_name_label = nx.get_node_attributes(graph, 'name_label')
        name_label = node_to_name_label[node]
        labels = list(node_to_name_label.values())
        return labels.count(name_label)

    aids1, aids2 = infr.get_filtered_edges(review_cfg)

    col_name_list = [
        #'index',
        'thumb1', 'thumb2',
        'match_thumb',
        'matched', 'reviewed',
        'score', 'rank',
        'tags',
        'timedelta',
        'kmdist',
        'speed',
        'cc_size1',
        'cc_size2',
        'aid1', 'aid2',
        #'data',
    ]

    #if not DEVELOPER_MODE:
    #    col_name_list.remove('data')

    if not review_cfg['show_match_thumb']:
        # FIXME; do one-vs-one scoring instead
        col_name_list.remove('match_thumb')

    col_getter_dict = {
        'index': np.arange(len(aids1)),
        'aid1': aids1,
        'aid2': aids2,
        'data': get_edge_data,
        'timedelta': lambda edge: (edge_assert(edge),
                                   ibs.get_unflat_annots_timedelta_list([edge])[0][0])[1],
        'speed': lambda edge: (edge_assert(edge),
                               ibs.get_unflat_annots_speeds_list2([edge])[0][0])[1],
        'kmdist': lambda edge: (edge_assert(edge),
                                ibs.get_unflat_annots_kmdists_list([edge])[0][0])[1],
        'matched':  get_match_text,
        'reviewed':  edge_attr_getter('reviewed_state', 'unreviewed'),
        'score':  edge_attr_getter('score'),
        'rank':  edge_attr_getter('rank', -1),
        'tags': get_pair_tags,
        'cc_size1': lambda edge: get_num_other(edge[0]),
        'cc_size2': lambda edge: get_num_other(edge[1]),
        'thumb1': ibs.get_annot_chip_thumbtup,
        'thumb2': ibs.get_annot_chip_thumbtup,
        'match_thumb': get_match_thumbtup,
    }

    col_ider_dict = {
        'thumb1': 'aid1',
        'thumb2': 'aid2',
        'match_thumb': ('aid1', 'aid2'),
        'data': ('aid1', 'aid2'),
        'score': ('aid1', 'aid2'),
        'rank': ('aid1', 'aid2'),
        'timedelta': ('aid1', 'aid2'),
        'speed': ('aid1', 'aid2'),
        'kmdist': ('aid1', 'aid2'),
        'matched': ('aid1', 'aid2'),
        'reviewed': ('aid1', 'aid2'),
        'tags': ('aid1', 'aid2'),
        'cc_size1': ('aid1', 'aid2'),
        'cc_size2': ('aid1', 'aid2'),
    }
    col_types_dict = {
        'rank': int,
        'score': float,
        'timedelta': float,
        'speed': float,
        'thumb1': 'PIXMAP',
        'thumb2': 'PIXMAP',
        'match_thumb': 'PIXMAP',
        'cc_size1': int,
        'cc_size2': int,
    }

    col_display_role_func_dict = {
        'timedelta': ut.partial(ut.get_posix_timedelta_str, year=True, approx=2),
        'speed': lambda speed: '%.2f km/h' % (speed,),
        'kmdist': lambda speed: '%.2f km' % (speed,),
    }

    col_bgrole_dict = {
        'matched' : ut.partial(get_match_status_bgrole, ibs),
        'reviewed': ut.partial(get_reviewed_status_bgrole, ibs),
    }

    col_width_dict = {
        'index': 42,
        'aid1': 50,
        'aid2': 50,
        'cc_size1': 80,
        'cc_size2': 80,
        'score': 65,
        'rank': 42,
        #'timedelta': 65,
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
    # api = edge_api
    return edge_api


def make_qt_graph_review(qreq_, cm_list):
    r"""
    CommandLine:
        ibeis make_qt_graph_review --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph2 import *  # NOQA
        >>> import guitool as gt
        >>> import ibeis
        >>> defaultdb = 'PZ_MTEST'
        >>> qreq_ = ibeis.testdata_qreq_(defaultdb=defaultdb)
        >>> cm_list = qreq_.execute()
        >>> gt.ensure_qtapp()
        >>> win = make_qt_graph_review(qreq_, cm_list)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=win, freq=10)
    """
    gt.ensure_qtapp()
    infr = graph_iden.AnnotInference.from_qreq_(qreq_, cm_list)
    gt.ensure_qtapp()
    print('infr = %r' % (infr,))
    win = AnnotGraphWidget(infr=infr, use_image=False, init_mode='review')
    abstract_interaction.register_interaction(win)
    win.show()
    return win


def make_qt_graph_interface(ibs, aids=None, nids=None, init_mode='rereview'):
    r"""
    CommandLine:
        ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9
        ibeis make_qt_graph_interface --show

        ibeis make_qt_graph_interface --db LEWA_splits --nids=1 --show --init-mode=rereview
        ibeis make_qt_graph_interface --db PZ_MTEST --nids=1 --show --init-mode=rereview

        ibeis make_qt_graph_interface --dbdir=~/lev/media/danger/GGR/GGR-IBEIS --nids=2300 --show
        ibeis make_qt_graph_interface --dbdir=~/lev/media/danger/GGR/GGR-IBEIS --nids=4617 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph2 import *  # NOQA
        >>> import guitool as gt
        >>> import ibeis
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> aids = ut.get_argval('--aids', type_=list, default=None)
        >>> nids = ut.get_argval('--nids', type_=list, default=None)
        >>> init_mode = ut.get_argval('--init_mode', default='rereview')
        >>> gt.ensure_qtapp()
        >>> win = make_qt_graph_interface(ibs, aids, nids, init_mode)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=win, freq=10)
    """
    #print('make_qt_graph_interface init()')
    #print('nids = %r' % (nids,))
    #print('aids = %r' % (aids,))
    if nids is not None and aids is None:
        aids = ut.flatten(ibs.get_name_aids(nids))
    if aids is None:
        aids = ibs.get_valid_aids()[0:40]
    print('make_qt_graph_interface aids = %r' % (aids,))
    #temp_nids = None
    nids = ibs.get_annot_name_rowids(aids)
    infr = graph_iden.AnnotInference(ibs, aids, nids, verbose=ut.VERBOSE)
    gt.ensure_qtapp()
    print('infr = %r' % (infr,))
    win = AnnotGraphWidget(infr=infr, use_image=False, init_mode=init_mode)
    abstract_interaction.register_interaction(win)
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
