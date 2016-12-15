# -*- coding: utf-8 -*-
"""
CommandLine:
    ibeis make_qt_graph_interface --show
    ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import vtool as vt
import numpy as np
import dtool
import networkx as nx
import itertools as it
from ibeis.algo.hots import graph_iden
import guitool as gt
import plottool as pt
from plottool import abstract_interaction
from guitool.__PYQT__ import QtCore
from guitool.__PYQT__.QtCore import Qt
from guitool import mpl_widget
from guitool import PrefWidget2

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


class DevGraphWidget(gt.GuitoolWidget):
    signal_graph_update = QtCore.pyqtSignal()

    def emit_graph_update(graph_widget):
        graph_widget.on_graph_update()

    def init_signals_and_slots(self):
        # https://doc.qt.io/archives/qtjambi-4.5.2_01/com/trolltech/qt/core/Qt.ConnectionType.html
        # connection_type = QtCore.Qt.AutoConnection
        # connection_type = QtCore.Qt.BlockingQueuedConnection
        # connection_type = QtCore.Qt.DirectConnection
        connection_type = QtCore.Qt.QueuedConnection
        self.signal_graph_update.connect(self.on_graph_update, type=connection_type)

    def initialize(graph_widget, use_image, self_parent):
        graph_widget.self_parent = self_parent
        graph_widget.plotinfo = None

        graph_widget.mpl_needs_update = True
        graph_widget.cb = None

        graph_widget.selected_aids = []
        graph_widget.splitter = graph_widget.addNewSplitter(
            orientation=Qt.Horizontal)
        graph_widget.ctrls_ = graph_widget.splitter.addNewWidget(
            orientation=Qt.Vertical, verticalStretch=1, margin=1, spacing=1)
        graph_widget.ctrls = graph_widget.ctrls_.addNewSplitter(
            orientation='vert')

        graph_widget.mpl_wgt = mpl_widget.MatplotlibWidget(
            parent=graph_widget, horizontalStretch=1)
        graph_widget.splitter.addWidget(graph_widget.mpl_wgt)

        ctrls = graph_widget.ctrls
        bbar1 = ctrls.addNewWidget(ori='vert', margin=1, spacing=1)
        bbar2 = ctrls.addNewWidget(ori='vert', margin=1, spacing=1)

        graph_widget.mark_state_funcs = self_parent.make_mark_state_funcs(
            graph_widget.selected_graph_pairs)
        for key, func in graph_widget.mark_state_funcs:
            bbar1.addNewButton(key.replace(' &', ': ').replace('&', ''), pressed=func)

        bbar1.addNewButton('Deselect', pressed=graph_widget.deselect)
        bbar1.addNewButton('Show Annots', pressed=graph_widget.show_selected)

        small_graph = len(self_parent.infr.aids) < 20

        class GraphVizConfig(dtool.Config):
            _param_info_list = [
                # Appearance
                ut.ParamInfo('show_image', default=use_image),
                ut.ParamInfo('in_image', default=use_image, hideif=lambda cfg: not cfg['show_image']),
                ut.ParamInfo('pin_positions', default=use_image),

                # Visibility
                ut.ParamInfo('show_reviewed_edges', small_graph),
                ut.ParamInfo('show_unreviewed_edges', small_graph),
                ut.ParamInfo('show_reviewed_cuts', small_graph),
                ut.ParamInfo('show_inferred_same', small_graph),
                ut.ParamInfo('show_inferred_diff', small_graph),
                ut.ParamInfo('highlight_reviews', True),
                ut.ParamInfo('show_recent_review', False),
                ut.ParamInfo('show_labels', small_graph),
                ut.ParamInfo('splines', 'spline' if small_graph else 'line', valid_values=['line', 'spline', 'ortho']),
                ut.ParamInfo('groupby', 'name_label', valid_values=['name_label', None]),
            ]

        def on_graphviz_config_changed(key=None):
            if key == 'pin_positions':
                graph_widget.set_pin_state(graph_widget.graphviz_config[key])
            else:
                graph_widget.emit_graph_update()

        graph_widget.graphviz_config = GraphVizConfig()
        graph_widget.graphviz_config_widget = PrefWidget2.EditConfigWidget(
            config=graph_widget.graphviz_config, with_buttons=False,
            changed=on_graphviz_config_changed)
        # remove headers
        graph_widget.graphviz_config_widget.tree_view.header().hide()
        bbar2.addWidget(graph_widget.graphviz_config_widget)

        # Connect signals and slots
        graph_widget.mpl_wgt.click_inside_signal.connect(graph_widget.on_click_inside)
        graph_widget.mpl_wgt.key_press_signal.connect(graph_widget.on_key_press)
        graph_widget.mpl_wgt.pick_event_signal.connect(graph_widget.on_pick)
        graph_widget.splitter.setSizes([30, 70])
        graph_widget.init_signals_and_slots()

    @property
    def infr(graph_widget):
        return graph_widget.self_parent.infr

    def on_graph_update(graph_widget):
        # ut.cprint('[graph] on_graph_update', 'green')
        if graph_widget.mpl_wgt is None or graph_widget.mpl_wgt.visibleRegion().isEmpty():
            # Flag that graph should draw next time it is visible
            graph_widget.mpl_needs_update = True
        else:
            # Draw the graph because it is visible
            graph_widget.draw_graph()

    def draw_graph(graph_widget):
        # Is it possible to make things more responsive with threads?
        # http://stackoverflow.com/questions/20324804/how-to-use-qthread-correctly-in-pyqt-with-movetothread
        # self.my_thread = QtCore.QThread()
        # self.my_thread.start()
        # self.graph_widget.moveToThread(self.my_thread)
        print('[graph] draw_graph', 'green')
        graph_widget.mpl_needs_update = False
        graph_widget.mpl_wgt.ax.cla()

        visibility_kw = graph_widget.graphviz_config.asdict()

        visibility_kw.pop('pin_positions')
        in_image = visibility_kw.pop('in_image')
        use_image = visibility_kw.get('show_image')
        if use_image:
            graph_widget.infr.update_node_image_config(in_image=in_image)
        graph_widget.infr.update_visual_attrs(
            **visibility_kw
        )
        try:
            graph_widget.plotinfo = pt.show_nx(
                graph_widget.infr.graph, layout='custom', as_directed=False,
                ax=graph_widget.mpl_wgt.ax,
                use_image=use_image, verbose=0)
        except IOError:
            graph_widget.infr.initialize_visual_node_attrs()
            graph_widget.plotinfo = pt.show_nx(
                graph_widget.infr.graph, layout='custom', as_directed=False,
                ax=graph_widget.mpl_wgt.ax,
                use_image=use_image, verbose=0)
        graph_widget.mpl_wgt.ax.set_aspect('equal')

        for aid in graph_widget.selected_aids:
            graph_widget.highlight_aid(aid, True)

        graph_widget.mpl_wgt.canvas.draw()
        graph_widget.mpl_wgt.fig.subplots_adjust(left=.02, top=.98, bottom=.02,
                                                 right=.85)

    def set_pin_state(graph_widget, flag):
        if flag:
            nx.set_node_attributes(graph_widget.infr.graph, 'pin', 'true')
        else:
            ut.nx_delete_node_attr(graph_widget.infr.graph, 'pin')

    def selected_graph_pairs(graph_widget):
        return it.combinations(graph_widget.selected_aids, 2)

    def show_selected(graph_widget):
        print('[graph_widget] show_selected')
        from ibeis.viz import viz_chip
        fnum = pt.ensure_fnum(10)
        print('fnum = %r' % (fnum,))
        fig = pt.figure(fnum=fnum)
        viz_chip.show_many_chips(graph_widget.infr.ibs, graph_widget.selected_aids, fnum=fnum)
        #fig.canvas.update()
        fig.show()
        fig.canvas.draw()
        graph_widget._figscope = fig

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
        print('[graph_widget] deselect')
        # print('graph_widget.selected_aids = %r' % (graph_widget.selected_aids,))
        graph_widget.toggle_selected_aid(graph_widget.selected_aids[:])

    def toggle_selected_aid(graph_widget, aids):
        print('[graph_widget] toggle_selected_aid')
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

    def show_popup_menu(graph_widget, options, event):
        """
        context menu
        """
        height = graph_widget.mpl_wgt.fig.canvas.geometry().height()
        qpoint = gt.newQPoint(event.x, height - event.y)
        qwin = graph_widget.mpl_wgt.fig.canvas
        gt.popup_menu(qwin, qpoint, options)

    def on_key_press(graph_widget, event):
        # called by matplotlib events
        key = event.key.upper()
        option_dict = gt.make_option_dict(graph_widget.mark_state_funcs,
                                          shortcuts=True)
        assert 'D' not in option_dict
        option_dict['D'] = graph_widget.deselect
        if key in option_dict:
            option_dict[key]()

    def on_pick(self, event):
        artist = event.artist
        plotdat = pt.get_plotdat_dict(artist)
        infr = self.self_parent.infr
        if plotdat:
            if 'node' in plotdat:
                if False:
                    all_node_data = ut.sort_dict(plotdat['node_data'].copy())
                    visual_node_data = ut.dict_subset(all_node_data, infr.visual_node_attrs, None)
                    node_data = ut.delete_dict_keys(all_node_data, infr.visual_node_attrs)
                    print('visual_node_data: ' + ut.repr2(visual_node_data, nl=1))
                    print('node_data: ' + ut.repr2(node_data, nl=1))
                    print('node: ' + ut.repr2(plotdat['node']))
            elif 'edge' in plotdat:
                all_edge_data = ut.sort_dict(plotdat['edge_data'].copy())
                visual_edge_data = ut.dict_subset(all_edge_data, infr.visual_edge_attrs, None)
                edge_data = ut.delete_dict_keys(all_edge_data, infr.visual_edge_attrs)
                print('visual_edge_data: ' + ut.repr2(visual_edge_data, nl=1))
                print('edge_data: ' + ut.repr2(edge_data, nl=1))
                print('edge: ' + ut.repr2(plotdat['edge']))
            else:
                print('unknown artist ' + ut.repr2(plotdat))
                print('artist = %r' % (artist,))
                print('event = %r' % (event,))

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

        if event.button == 3 and not context_shown:
            if len(graph_widget.selected_aids) != 2:
                print('This funciton only work if exactly 2 are selected')
            else:
                from ibeis.gui import inspect_gui
                context_shown = True
                aid1, aid2 = (graph_widget.selected_aids)
                qres = None
                qreq_ = None
                # TODO: rectify with get_edge_options so that can be used here
                # Really, just rectify the selection model
                options = inspect_gui.get_aidpair_context_menu_options(
                    graph_widget.infr.ibs, aid1, aid2, qres, qreq_=qreq_)
                graph_widget.show_popup_menu(options, event)

        bbox = vt.bbox_from_center_wh(graph_widget.plotinfo['node']['pos'][node],
                                      graph_widget.plotinfo['node']['size'][node])
        annot_selected = vt.point_inside_bbox(point, bbox)

        if annot_selected:
            ibs = graph_widget.infr.ibs
            print(ut.obj_str(ibs.get_annot_info(aid, default=True,
                                                name=True, gname=True)))
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

    def eventFilter(graph_widget, source, event):
        if event.type() == QtCore.QEvent.Show:
            if graph_widget.mpl_needs_update:
                graph_widget.emit_graph_update()
        return super(DevGraphWidget, graph_widget).eventFilter(source, event)


class AnnotGraphWidget(gt.GuitoolWidget):
    signal_state_update = QtCore.pyqtSignal(bool, bool)

    def initialize(self, infr=None, use_image=False, init_mode='rereview',
                   review_cfg=None):
        print('[viz_graph] initialize')

        self.init_mode = init_mode
        print('self.init_mode = %r' % (self.init_mode,))

        if review_cfg is None:
            mode = 'filtered' if self.init_mode == 'split' else 'unfiltered'
            self.preset_config(mode)

        self.infr = infr
        self.initialize_menus()

        self.graph_tab_widget = self.addNewTabWidget(verticalStretch=1)

        self.statbar1 = self.addNewWidget(
            orientation='horiz', verticalStretch=1, margin=1, spacing=1)
        self.statbar2 = self.addNewWidget(
            orientation='horiz', verticalStretch=1, margin=1, spacing=1)

        self.prog_bar = self.addNewProgressBar(visible=False)

        self.initialize_api_tabs()

        self.statbar1.addNewButton('Match and Score', min_width=1,
                                   pressed=self.match_and_score_edges)
        self.statbar1.addNewButton('ScoreVsOne', min_width=1,
                                   pressed=self.score_edges_vsone)
        self.statbar1.addNewButton('Edit Filters', min_width=1,
                                   pressed=self.edit_filters)
        self.statbar1.addNewButton('Repopulate', min_width=1,
                                   pressed=self.repopulate)

        self.statbar2.addNewButton('Reset DBState', min_width=1,
                                   pressed=self.reset_review)
        self.statbar2.addNewButton('Reset Rereview', min_width=1,
                                   pressed=self.reset_rereview)

        self.num_names_lbl = self.statbar2.addNewLabel('NUM_NAMES_LBL')
        self.state_lbl = self.statbar2.addNewLabel('STATE_LBL')

        self.statbar2.addNewButton('Accept', pressed=self.accept)

        # _show_graph = self.init_mode in ['split', 'rereview', 'review']
        _show_graph = True
        if _show_graph:
            # TODO: separate graph view into its own class
            self.graph_tab = self.graph_tab_widget.addNewTab('Graph')
            # TODO: make this its own proper widget
            self.graph_widget = DevGraphWidget(parent=self, self_parent=self,
                                               use_image=use_image)
            self.graph_tab.addWidget(self.graph_widget)
            # self.graph_widget.connect_kepress_to_slot
        else:
            self.graph_widget = None
            self.graph_tab = None
        self.init_signals_and_slots()

    def init_signals_and_slots(self):
        # https://doc.qt.io/archives/qtjambi-4.5.2_01/com/trolltech/qt/core/Qt.ConnectionType.html
        # connection_type = QtCore.Qt.AutoConnection
        # connection_type = QtCore.Qt.BlockingQueuedConnection
        # connection_type = QtCore.Qt.DirectConnection
        connection_type = QtCore.Qt.QueuedConnection
        self.signal_state_update.connect(self.update_state,
                                         type=connection_type)

    def initialize_api_tabs(self):
        self.api_tabs = {}
        self.api_widgets = {}

        def _add_item_widget_tab(key, view_class='table'):
            title = key.title().replace('_', ' ')
            self.api_tabs[key] = self.graph_tab_widget.addNewTab(title)
            self.api_widgets[key] = gt.APIItemWidget(view_class=view_class)
            self.api_tabs[key].addWidget(self.api_widgets[key])

        _add_item_widget_tab('edges')
        _add_item_widget_tab('nodes')
        _add_item_widget_tab('name_nodes', view_class='tree')
        _add_item_widget_tab('name_edges', view_class='tree')

        edge_view = self.api_widgets['edges'].view
        edge_view.doubleClicked.connect(self.edge_doubleclick)
        edge_view.contextMenuClicked.connect(self.edge_context)
        edge_view.connect_keypress_to_slot(self.edge_keypress)
        edge_view.connect_single_key_to_slot(gt.ALT_KEY, self.on_alt_pressed)

        name_edge_view = self.api_widgets['name_edges'].view
        name_edge_view.doubleClicked.connect(self.edge_doubleclick)
        name_edge_view.contextMenuClicked.connect(self.edge_context)
        name_edge_view.connect_keypress_to_slot(self.edge_keypress)
        name_edge_view.connect_single_key_to_slot(gt.ALT_KEY, self.on_alt_pressed)

    def populate_edge_model(self):
        print('[viz_graph] populate_edge_model')
        # if self.init_mode is None:
        #     self.review_cfg['show_match_thumb'] = False
        key = 'edges'
        tab = self.api_tabs[key]
        widget = self.api_widgets[key]
        api = make_edge_api(self.infr, review_cfg=self.review_cfg)
        title = key.title().replace('_', ' ')
        headers = api.make_headers(tblnice=title, tblname=key)
        widget.change_headers(headers)
        widget.resize_headers(api)
        widget.view.verticalHeader().setVisible(True)
        tab.setTabText('%s (%r)' % (title, widget.model.num_rows_total))
        widget.view.verticalHeader().setDefaultSectionSize(221)

    def populate_node_model(self):
        print('[viz_graph] populate_node_model')
        api = make_node_api(self.infr)
        key = 'nodes'
        tab = self.api_tabs[key]
        widget = self.api_widgets[key]
        title = key.title().replace('_', ' ')
        headers = api.make_headers(tblnice=title, tblname=key)
        widget.change_headers(headers)
        widget.view.verticalHeader().setVisible(True)
        try:
            widget.view.verticalHeader().setMovable(True)
        except AttributeError:
            widget.view.verticalHeader().setSectionsMovable(True)
        tab.setTabText('%s (%r)' % (title, widget.model.num_rows_total))

    def populate_name_node_model(self):
        api = make_name_node_api(self.infr, review_cfg=self.review_cfg)
        key = 'name_nodes'
        tab = self.api_tabs[key]
        widget = self.api_widgets[key]
        title = key.title().replace('_', ' ')
        headers = api.make_headers(tblnice=title, tblname=key)
        widget.change_headers(headers)
        tab.setTabText('%s (%r)' % (title, widget.model.num_rows_total))

    def populate_name_edge_model(self):
        api = make_name_edge_api(self.infr, review_cfg=self.review_cfg)
        key = 'name_edges'
        tab = self.api_tabs[key]
        widget = self.api_widgets[key]
        title = key.title().replace('_', ' ')
        headers = api.make_headers(tblnice=title, tblname=key)
        widget.change_headers(headers)
        tab.setTabText('%s (%r)' % (title, widget.model.num_rows_total))

    def initialize_menus(self):
        self.menubar = gt.newMenubar(self)
        self.menus = {}

        key = 'Dev'
        menu = self.menus[key] = self.menubar.newMenu(key)
        menu.newAction(triggered=self.print_info)
        menu.newAction(triggered=self.embed, shortcut='ctrl+shift+I')
        menu.newAction(triggered=self.expand_image_and_names)
        menu.newAction(triggered=self.emit_state_update)
        menu.newAction(triggered=self.print_staging_table)
        menu.newAction(triggered=self.print_annotmatch_table)
        menu.newAction(triggered=self.print_deltas)

        key = 'Actions'
        menu = self.menus[key] = self.menubar.newMenu(key)
        menu.newAction(triggered=self.commit_to_staging)
        menu.newAction(triggered=self.commit_to_database)

        key = 'Debug'
        menu = self.menus[key] = self.menubar.newMenu(key)
        menu.newAction(triggered=self.name_rebase)
        menu.newAction(triggered=self.ensure_full)
        menu.newAction(triggered=self.ensure_cliques)
        menu.newAction(triggered=self.vsone_subset)

    def preset_config(self, mode='filtered'):
        print('[graph] preset_config mode=%r' % (mode,))
        if mode == 'filtered':
            self.review_cfg = GRAPH_REVIEW_CFG_DEFAULTS.copy()
        elif mode == 'unfiltered':
            self.review_cfg = GRAPH_REVIEW_CFG_DEFAULTS.copy()
            for key in self.review_cfg.keys():
                if key.startswith('filter_'):
                    self.review_cfg[key] = False

    def showEvent(self, event):
        super(AnnotGraphWidget, self).showEvent(event)
        ut.cprint('[viz_graph] showEvent', 'green')
        # Fire initialize event after we show the GUI
        QtCore.QTimer.singleShot(50, self.init_inference)

    def init_inference(self):
        print('[viz_graph] init_inference mode=%r' % (self.init_mode))
        if self.init_mode is None:
            pass
        elif self.init_mode == 'split':
            self.preset_config('filtered')
            self.reset_split()
        elif self.init_mode == 'rereview':
            self.preset_config('unfiltered')
            self.reset_rereview()
        elif self.init_mode == 'review':
            self.reset_review()
        else:
            raise ValueError('Unknown init_mode=%r' % (self.init_mode,))
        self.repopulate()

        if ut.get_argflag('--graph'):
            index = self.graph_tab_widget.indexOf(self.graph_tab)
            self.graph_tab_widget.setCurrentIndex(index)
            # self.graph_tab_widget.setCurrentIndex(2)

    def repopulate(self):
        # self.update_state(structure_changed=True)
        self.emit_state_update(structure_changed=True)

    def emit_state_update(self, structure_changed=False, disable_global_update=False):
        self.signal_state_update.emit(structure_changed, disable_global_update)

    def update_state(self, structure_changed=False, disable_global_update=False):
        print('[viz_graph] update_state mode=%s' % (self.init_mode,))
        #if self.init_mode in ['split', 'rereview']:
        if not disable_global_update:
            if self.init_mode == 'split':
                self.infr.apply_feedback_edges()
                self.infr.apply_weights()
                self.infr.relabel_using_reviews()
                # self.infr.apply_cuts()
            elif self.init_mode == 'rereview':
                self.infr.apply_feedback_edges()
                self.infr.apply_match_scores()
                self.infr.apply_weights()
                self.infr.relabel_using_reviews()
                # self.infr.apply_cuts()
            elif self.init_mode == 'review':
                self.infr.apply_match_edges()
                self.infr.review_dummy_edges()
                self.infr.apply_feedback_edges()
                self.infr.apply_match_scores()
                self.infr.apply_weights()
                self.infr.relabel_using_reviews()
                self.infr.apply_review_inference()
                # probably don't need apply_cuts
                self.infr.apply_cuts()

        # Set gui status indicators
        status = self.infr.connected_component_status()
        truth_colors = self.infr._get_truth_colors()
        if status['num_inconsistent']:
            self.state_lbl.setText('Inconsistent Names: %d' % (status['num_inconsistent'],))
            self.state_lbl.setColor('black', truth_colors['nomatch'][0:3] * 255)
        else:
            self.state_lbl.setText('Consistent')
            self.state_lbl.setColor('black', truth_colors['match'][0:3] * 255)

        self.num_names_lbl.setText('Names: max=%r, ~min=%r' % (
            status['num_names_max'],
            status['num_names_min'],))

        # print('[viz_graph] on_update_state mode=%s' % (self.init_mode,))
        if structure_changed:
            # TODO: only make this API if the tab is clicked
            self.populate_node_model()
            self.populate_edge_model()
            self.populate_name_node_model()
            self.populate_name_edge_model()
        if self.graph_widget is not None:
            self.graph_widget.emit_graph_update()

        for widget in self.api_widgets.values():
            model = widget.view.model()
            # FIXME: should only clear cache in viewport
            model.clear_cache()
            model.layoutChanged.emit()

    def ensure_cliques(self):
        self.infr.ensure_cliques()
        self.infr.relabel_using_reviews()
        self.infr.apply_review_inference()
        self.repopulate()

    def ensure_full(self):
        self.infr.ensure_full()
        self.infr.relabel_using_reviews()
        self.infr.apply_review_inference()
        self.repopulate()

    def match_and_score_edges(self):
        with gt.GuiProgContext('Scoring Edges', self.prog_bar) as ctx:
            self.infr.exec_matching(prog_hook=ctx.prog_hook)
            self.infr.apply_match_edges(self.review_cfg)
            self.infr.apply_match_scores()
            self.infr.apply_weights()
        self.repopulate()

    def score_edges_vsone(self):
        with gt.GuiProgContext('Scoring Edges', self.prog_bar) as ctx:
            self.infr.exec_vsone(prog_hook=ctx.prog_hook)
            self.infr.apply_match_scores()
            self.infr.apply_weights()
        self.repopulate()

    def reset_review(self):
        print('[viz_graph] reset_review')
        infr = self.infr
        with gt.GuiProgContext('Reset Review', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            with ut.Timer('reset_feedback'):
                infr.reset_feedback('staging')
            with ut.Timer('reinit_name_labels'):
                infr.reset_labels_to_ibeis()
            if self.graph_widget is not None:
                self.graph_widget.set_pin_state(True)
            with ut.Timer('apply_feedback_edges'):
                infr.apply_feedback_edges()
            with ut.Timer('review_dummy_edges'):
                infr.review_dummy_edges()
            with ut.Timer('apply_match_edges'):
                infr.apply_match_edges()
            infr.apply_match_scores()
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
        print('[viz_graph] reset_rereview')
        infr = self.infr
        self.init_mode = 'rereview'
        with gt.GuiProgContext('Reset Review', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            infr.initialize_graph()
            if self.graph_widget is not None:
                self.graph_widget.set_pin_state(True)
            ctx.set_progress(msg='reset name labels')
            infr.reset_name_labels()
            ctx.set_progress(msg='reset feedback')
            infr.reset_feedback('staging')
            ctx.set_progress(msg='reset feedback edges')
            infr.apply_feedback_edges()
            ctx.set_progress(msg='remove name labels')
            infr.remove_name_labels()
            ctx.set_progress(msg='apply match scores')
            infr.apply_match_scores()
            ctx.set_progress(msg='repopulate')
            self.repopulate()
            ctx.set_progress(8, 8)

    def reset_split(self):
        infr = self.infr
        self.init_mode = 'split'
        with gt.GuiProgContext('Initializing', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            infr.initialize_graph()
            if self.graph_widget is not None:
                self.graph_widget.set_pin_state(True)
            ctx.set_progress(1, 3)
            infr.remove_feedback()
            ctx.set_progress(2, 3)
            infr.remove_name_labels()
            ctx.set_progress(3, 3)

    def edit_filters(self):
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

    def edge_doubleclick(self, qtindex):
        """
        qtindex = qtindex = self.api_widgets['edges'].view.get_row_and_qtindex_from_id(1)[0]
        """
        print('[viz_graph] DoubleClicked: ' + str(gt.qtype.qindexinfo(qtindex)))
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

    def mark_pair_state(self, pairs, state):
        valid_states = ['match', 'nomatch', 'notcomp', 'unreviewed']
        statetags = state.split('+')
        state = statetags[0]
        tags = statetags[1].split(';') if len(statetags) > 1 else []
        assert state in valid_states
        for aid1, aid2 in pairs:
            self.infr.add_feedback(aid1, aid2, state, tags=tags, apply=True)
        self.emit_state_update(disable_global_update=True)

    def make_mark_state_funcs(self, selection_func):
        def _mark_selected_pair_state(state):
            self.mark_pair_state(selection_func(), state)
        options = [
            ('Mark &True', ut.partial(_mark_selected_pair_state, 'match')),
            ('Mark &False', ut.partial(_mark_selected_pair_state, 'nomatch')),
            ('Mark &Not-Comparable', ut.partial(_mark_selected_pair_state, 'notcomp')),
            ('Mark &Photobomb', ut.partial(_mark_selected_pair_state, 'nomatch+photobomb')),
            ('Mark &SceneryMatch', ut.partial(_mark_selected_pair_state, 'nomatch+scenerymatch')),
            ('&Unreview', ut.partial(_mark_selected_pair_state, 'unreviewed')),
            # unreview will only remove internal feedback, anything commited will not change
        ]
        return options

    def get_edge_options(self, view):
        selected_qtindex_list_ = view.selectedRows()

        selected_qtindex_edges = []
        selected_qtindex_names = []
        for qtindex in selected_qtindex_list_:
            model = qtindex.model()
            print('model.name = %r' % (model.name,))
            if model.name == 'name_edges':
                level = qtindex.internalPointer().level
                if level == 0:
                    selected_qtindex_names.append(qtindex)
                elif level == 1:
                    selected_qtindex_edges.append(qtindex)
            elif model.name == 'edges':
                selected_qtindex_edges.append(qtindex)

        def _pairs():
            for qtindex in selected_qtindex_edges:
                model = qtindex.model()
                aid1  = model.get_header_data('aid1', qtindex)
                aid2  = model.get_header_data('aid2', qtindex)
                yield aid1, aid2
        def _mark_selected_pair_state(state):
            self.mark_pair_state(_pairs(), state)

        options = []
        if len(selected_qtindex_edges) > 0:
            options += self.make_mark_state_funcs(_pairs)

        if len(selected_qtindex_edges) == 1:
            from ibeis.gui import inspect_gui
            ibs = self.infr.ibs
            qtindex = selected_qtindex_edges[0]
            model = qtindex.model()
            aid1  = model.get_header_data('aid1', qtindex)
            aid2  = model.get_header_data('aid2', qtindex)
            qreq_ = self.infr.qreq_
            pair_tag_options = inspect_gui.make_aidpair_tag_context_options(ibs, aid1, aid2)
            chip_context_options = inspect_gui.make_annotpair_context_options(
                ibs, aid1, aid2, qreq_=qreq_)
            external_options = []
            external_options += [
                ('Match Ta&gs', pair_tag_options)
            ]
            external_options += chip_context_options
            options += [
                ('External Options', external_options),
            ]

            if True or self.init_mode != 'split':
                from ibeis.viz import viz_graph2
                nids = ut.unique(ibs.get_annot_nids([aid1, aid2]))
                options += [
                    ('New Split Case Interaction',
                     ut.partial(viz_graph2.make_qt_graph_interface, ibs,
                                nids=nids)),
                ]

            options += [
                ('Tune Vsone(vt)', inspect_gui.make_vsone_context_options(ibs, aid1, aid2, qreq_)[0][1])
            ]

        # if len(selected_qtindex_names) > 0:
        #     import utool
        #     utool.embed()
        #     options += [
        #         # TODO
        #         # ('Restrict Graph To Names', lambda: self.restrict_graph_to_names),
        #         ('Vsone subset', lambda: self.vsone_subset()),
        #     ]
        # else:
        if len(selected_qtindex_edges) > 0:
            options += [
                ('Vsone subset', lambda: self.vsone_subset(_pairs())),
            ]

        return options

    def vsone_subset(self, edges=None):
        print('[graph] vsone_subset')
        if edges is None:
            edges = self.infr.graph.edges()
        self.infr.exec_vsone_subset(edges)

    def restrict_graph_to_names(self):
        pass

    @gt.slot_(QtCore.QModelIndex, QtCore.QPoint)
    def edge_context(self, qtindex, qpoint):
        print('context')
        #print('option_dict = %s' % (ut.repr3(option_dict, nl=2),))
        view = qtindex.model().view
        options = self.get_edge_options(view)
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
        view = self.api_widgets['edges'].view
        """
        event_key = event.key()
        options = self.get_edge_options(view)
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
        return handled

    def print_staging_table(self):
        db = self.infr.ibs.staging
        print(db.get_table_csv('reviews'))

    def print_annotmatch_table(self):
        db = self.infr.ibs.db
        print(db.get_table_csv('annotmatch'))

    def commit_to_staging(self):
        print('[graph] commit to staging')
        self.infr.commit_to_staging()

    def commit_to_database(self):
        print('[graph] commit to database')
        self.infr.write_ibeis_staging_feedback()
        self.infr.write_ibeis_annotmatch_feedback()

    def print_deltas(self):
        pairs = [('external', 'internal'),
                 ('annotmatch', 'all'),
                 ('staging', 'all'),
                 ('annotmatch', 'staging')]
        for old, new in pairs:
            print('old = %r' % (old,))
            print('new = %r' % (new,))
            print(self.infr.match_state_delta(old, new))

    def name_rebase(self):
        num_names, num_inconsistent = self.infr.relabel_using_reviews()
        aid_to_newname = self.infr.get_ibeis_name_assignment()
        nx.set_node_attributes(self.infr.graph, 'name_label', aid_to_newname)

    def accept(self):
        import pandas as pd
        print('[viz_graph] accept')
        infr = self.infr
        num_new_names, num_inconsistent = self.infr.relabel_using_reviews()

        aid_to_newname = self.infr.get_ibeis_name_assignment()
        aid_list = list(aid_to_newname.keys())
        new_name_list = list(aid_to_newname.values())
        old_name_list = infr.ibs.get_annot_name_texts(aid_list)

        num_names_changed = sum([n1 != n2 for n1, n2 in
                                 zip(new_name_list, old_name_list)])
        num_old_names = len(set(old_name_list))

        # keep track of residual data
        changed_df = infr.match_state_delta()
        num_added = pd.isnull(changed_df['am_rowid']).values.sum()
        num_edges_modified = len(changed_df) - num_added

        msg = ut.codeblock(
            '''
            Are you sure this is correct?
            #orig_names=%r
            #new_names=%r
            #names_changed=%r
            #edges_modified=%r
            #inconsistent=%r
            ''') % (num_old_names, num_new_names, num_names_changed,
                    num_edges_modified, num_inconsistent)

        lines = []
        print_ = lines.append

        print_('=================')
        print_(msg)
        print_('ACCEPT GRAPH REVIEW')
        print_('aid_list = %r' % (aid_list,))
        print_('old_name_list = %r' % (old_name_list,))
        print_('new_name_list = %r' % (new_name_list,))
        # logger.info('_initial_feedback = ' + ut.repr2(self.infr._initial_feedback, nl=1))
        # print_('external_feedback = ' + ut.repr2(self.infr.external_feedback, nl=1))
        print_('internal_feedback = ' + ut.repr2(self.infr.internal_feedback, nl=1))

        pdkw = dict(max_rows=len(changed_df) + 1)
        #print_ = print
        print_('There were %d added annot match rows' % (num_added,))
        print_('There were %d modified annot match rows' % (num_edges_modified,))
        print_('---DATAFRAME\nchanged_info =\n' + changed_df.to_string(**pdkw))

        print('\n'.join(lines))

        if not gt.are_you_sure(msg=msg):
            raise Exception('Cancel')
        print('\n'.join(lines))

        infr.write_ibeis_name_assignment(aid_to_newname)
        infr.commit_to_database()
        gt.user_info(self, 'Name Change Complete')

    def sizeHint(self):
        return QtCore.QSize(1100, 500)

    def closeEvent(self, event):
        abstract_interaction.unregister_interaction(self)
        super(AnnotGraphWidget, self).closeEvent(event)

    def print_info(self):
        print('[graph] print_info')
        print('external_feedback = ' + ut.repr2(self.infr.external_feedback, nl=1))
        print('internal_feedback = ' + ut.repr2(self.infr.internal_feedback, nl=1))
        infr = self.infr
        print('infr = %r' % (infr,))
        if infr is not None and infr.graph is not None:
            print(ut.repr3(ut.graph_info(infr.simplify_graph())))

    def embed(self):
        infr = self.infr  # NOQA
        ibs = infr.ibs  # NOQA
        graph = infr.graph  # NOQA
        import utool
        utool.embed()

    def expand_image_and_names(self):
        # call get_name_image_closure()
        ibs = self.infr.ibs
        aids = self.infr.aids
        annots = ibs.annots(aids)
        aids = annots.get_name_image_closure()
        nids = ibs.get_annot_nids(aids)
        new_infr = graph_iden.AnnotInference(ibs, aids, nids,
                                             verbose=self.infr.verbose)
        new_infr.initialize_graph()
        self.infr = new_infr
        self.init_inference()


class EdgeAPIHelper(object):
    def __init__(self, infr):
        self.infr = infr
        self.graph = infr.graph
        self.ibs = infr.ibs

    def make_partial_edge_headers(self):
        """
        These are partial api headers meant to augment edge headers
        """

        custom_edge_props = [
            # TODO: allow user to specify things like hardness / failed / passed or
            # whatever
            'priority',
            'maybe_error',
            'failed',
            'hardness',
            'normscore',
        ]

        edge_col_name_list = [
            #'index',
            'thumb1', 'thumb2',
            'match_thumb',
            'inference', 'review',
            'score', 'rank',
            'tags',
            'timedelta',
            'kmdist',
            'speed',
        ]
        edge_col_name_list.extend(custom_edge_props)
        edge_col_name_list += [
            'cc_size1',
            'cc_size2',
            'aid1', 'aid2',
            #'data',
        ]

        col_getter_dict = {
            'data': self.get_edge_data,
            'timedelta': self.get_edge_timedelta,
            'speed': self.get_edge_speed,
            'kmdist': self.get_edge_kmdist,
            'inference':  self.get_inference_text,
            'review': self.get_review_text,
            'score':  self.edge_attr_getter('score'),
            'rank':  self.edge_attr_getter('rank', -1),
            'tags': self.get_pair_tags,
            'cc_size1': lambda edge: self.get_num_other(edge[0]),
            'cc_size2': lambda edge: self.get_num_other(edge[1]),

            'thumb1': self.ibs.get_annot_chip_thumbtup,
            'thumb2': self.ibs.get_annot_chip_thumbtup,
            'match_thumb': self.get_match_thumbtup,
        }
        for name in custom_edge_props:
            col_getter_dict[name] = self.edge_attr_getter(name)

        col_ider_dict = {name: ('aid1', 'aid2') for name in col_getter_dict.keys()}
        # col_ider_dict = ({
        col_ider_dict.update({
            'thumb1': 'aid1',
            'thumb2': 'aid2',
        })

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
            'inference' : self.get_inference_bgrole,
            'review': self.get_review_bgrole,
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

        partial_headers = {
            'edge_col_name_list': edge_col_name_list,
            'col_getter_dict': col_getter_dict,
            'col_ider_dict': col_ider_dict,
            'col_ider_dict': col_ider_dict,
            'col_types_dict': col_types_dict,
            'col_display_role_func_dict': col_display_role_func_dict,
            'col_bgrole_dict': col_bgrole_dict,
            'col_width_dict': col_width_dict,
        }
        return partial_headers

    def get_edge_timedelta(self, edge):
        self.edge_assert(edge)
        return self.ibs.get_unflat_annots_timedelta_list([edge])[0][0]

    def get_edge_speed(self, edge):
        self.edge_assert(edge)
        return self.ibs.get_unflat_annots_speeds_list2([edge])[0][0]

    def get_edge_kmdist(self, edge):
        self.edge_assert(edge)
        return self.ibs.get_unflat_annots_kmdists_list([edge])[0][0]

    def get_edge_data(self, edge):
        aid1, aid2 = edge
        attrs = self.graph.get_edge_data(aid1, aid2).copy()
        remove_attrs = self.infr.visual_edge_attrs + ['rank', 'reviewed_state', 'score']
        try:
            remove_attrs.remove('style')
        except ValueError:
            pass
        ut.delete_dict_keys(attrs, remove_attrs)
        attrs = {k: v for k, v in attrs.items() if v is not None}
        return ut.repr2(attrs, precision=2, explicit=True, nobr=True)

    def edge_attr_getter(self, attr, default=None):
        def get_edge_attr(edge):
            data = self.graph.get_edge_data(*edge)
            return data.get(attr, default)
        return get_edge_attr

    def get_pair_tags(self, edge):
        aid1, aid2 = edge
        self.edge_assert(edge)
        # ibs = self.infr.ibs
        # # FIXME: use graph properties instead
        # am_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey(
        #     [aid1], [aid2])
        # tag_text = ibs.get_annotmatch_tag_text(am_rowids)[0]
        tags = self.infr.graph.get_edge_data(*edge).get('reviewed_tags', [])
        if tags is None:
            tag_text = ''
        else:
            tag_text = ';'.join(tags)
        return str(tag_text)

    def edge_assert(self, edge):
        aid1, aid2 = edge
        assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
        assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)

    def _get_inference_info(self, edge):
        aid1, aid2 = edge
        nid1 = self.graph.node[aid1]['name_label']
        nid2 = self.graph.node[aid2]['name_label']
        data = self.infr.graph.get_edge_data(*edge)
        inferred_state = data['inferred_state']
        maybe_error = data.get('maybe_error', False)

        if inferred_state is None:
            state = 'unknown'
        elif inferred_state.startswith('inconsistent'):
            state = inferred_state
        else:
            inferred_truth = {'same': True, 'diff': False}[inferred_state]
            name_truth = (nid1 == nid2)
            if name_truth != inferred_truth:
                state = 'disagree'
            else:
                state = 'same' if nid1 == nid2 else 'diff'

        if state == 'inconsistent_outgoing':
            text = 'inconsistent'
        else:
            text = state

        if nid1 == nid2:
            text += ' nid=%r' % (nid1,)
        else:
            text += ' nids=%r,%r' % (nid1, nid2)

        if maybe_error:
            text += ' (FIXME?)'

        if state == 'inconsistent_outgoing':
            text += ' (outgoing)'

        info = (state, text, maybe_error)
        return info

    def get_inference_text(self, edge):
        info = self._get_inference_info(edge)
        state, text, maybe_error = info
        return text

    def get_review_text(self, edge):
        graph = self.infr.graph
        text = graph.get_edge_data(*edge).get('reviewed_state', 'unreviewed')
        return text

    def get_inference_bgrole(self, edge):
        """ Background role for status column """
        state, text, maybe_error = self._get_inference_info(edge)
        if state == 'disagree':
            color = pt.WHITE
        elif state.startswith('inconsistent'):
            color = pt.ORANGE
            if state == 'inconsistent_outgoing':
                lighten_amount = .55
                color = pt.lighten_rgb(color, lighten_amount)
            elif not maybe_error:
                lighten_amount = .35
                color = pt.lighten_rgb(color, lighten_amount)
        else:
            lighten_amount = .35
            truth_colors = self.infr._get_truth_colors()
            if state == 'unknown':
                lighten_amount = .7
                color = truth_colors['unreviewed']
            else:
                color = truth_colors['match'] if state == 'same' else truth_colors['nomatch']
            #self.graph.get_edge_data(*edge).get('reviewed_state', 'unreviewed')]
            if lighten_amount is not None:
                color = pt.lighten_rgb(color, lighten_amount)
        color = pt.to_base255(color)
        return color

    def get_review_bgrole(self, edge):
        """ Background role for status column """
        data = self.graph.get_edge_data(*edge)
        state = data.get('reviewed_state', 'unreviewed')
        truth_colors = self.infr._get_truth_colors()
        if state == 'unreviewed':
            inference_state, text, maybe_error = self._get_inference_info(edge)
            if inference_state == 'same':
                color = truth_colors['match']
            elif inference_state == 'diff':
                color = truth_colors['nomatch']
            else:
                color = truth_colors[state]
        else:
            color = truth_colors[state]
        lighten_amount = .35
        if state == 'unreviewed':
            lighten_amount = .7
        if lighten_amount is not None:
            color = pt.lighten_rgb(color, lighten_amount)
        color = pt.to_base255(color)
        return color

    def get_match_thumbtup(self, edge, thumbsize=None):
        #sibs, qaid2_cm, qaids, daids, index, qreq_=None,
        #                   thumbsize=(128, 128), match_thumbtup_cache={}):
        aid1, aid2 = edge
        cm, aid1, aid2 = self.infr.lookup_cm(aid1, aid2)
        from ibeis.gui import id_review_api
        if cm is None:
            # HACK: check if a PairwiseMatch exists
            match = self.infr.vsone_matches.get((aid1, aid2))
            if match is not None:
                fpath, func, func2 = id_review_api.make_ensure_match_img_nosql_func(
                    self.infr.ibs, match, None)
                thumbdat = {
                    'fpath': fpath,
                    'thread_func': func,
                    'main_func': func2,
                }
                return thumbdat
            else:
                return None
        #assert cm.qaid == aid1, 'aids do not aggree'
        # Hacky new way of drawing
        fpath, func, func2 = id_review_api.make_ensure_match_img_nosql_func(
            self.infr.qreq_, cm, aid2)
        thumbdat = {
            'fpath': fpath,
            'thread_func': func,
            'main_func': func2,
        }
        return thumbdat

    def get_num_other(self, aid):
        node = self.infr.aid_to_node[aid]
        node_to_name_label = nx.get_node_attributes(self.graph, 'name_label')
        name_label = node_to_name_label[node]
        labels = list(node_to_name_label.values())
        return labels.count(name_label)


def make_name_edge_api(infr, review_cfg={}):
    node_to_name = infr.get_node_attrs('name_label')
    name_to_nodes = ut.group_items(node_to_name.keys(), node_to_name.values())

    name_to_edges = {name: list(infr.graph.subgraph(nodes).edges())
                     for name, nodes in name_to_nodes.items()}

    names = list(name_to_edges.keys())
    flat_edges, grouped_edge_idxs = ut.invertible_flatten1(name_to_edges.values())
    grouped_edges = ut.apply_grouping(flat_edges, grouped_edge_idxs)
    n_annots_list = [len(set(ut.flatten(edges))) for edges in grouped_edges]

    nid_col_name_list = ['name_label', 'n_annots', 'n_edges']

    self = EdgeAPIHelper(infr)
    partial_headers = self.make_partial_edge_headers()
    edge_col_name_list = partial_headers['edge_col_name_list']

    col_name_list = nid_col_name_list + edge_col_name_list
    col_level_dict = {}
    for col in nid_col_name_list:
        col_level_dict[col] = 0
    for col in edge_col_name_list:
        col_level_dict[col] = 1
    iders = [
        list(range(len(names))),
        grouped_edge_idxs,
    ]
    col_getter_dict = {
        'name_label': names,
        'n_edges': list(map(len, grouped_edge_idxs)),
        'n_annots': n_annots_list,
        'aid1': ut.take_column(flat_edges, 0),
        'aid2': ut.take_column(flat_edges, 1),
    }

    col_getter_dict.update(partial_headers['col_getter_dict'])

    col_display_role_func_dict = partial_headers['col_display_role_func_dict']
    col_bgrole_dict = partial_headers['col_bgrole_dict']

    col_ider_dict = partial_headers['col_ider_dict']

    def name_bg_role(idx):
        name = names[idx]
        edges = name_to_edges[name]
        color = pt.WHITE
        for edge in edges:
            data = self.infr.graph.get_edge_data(*edge)
            inferred_state = data['inferred_state']
            if inferred_state.startswith('inconsistent'):
                color = pt.ORANGE
                break
        color = pt.to_base255(color)
        return color

    col_bgrole_dict['name_label'] = name_bg_role
    col_bgrole_dict['n_edges'] = name_bg_role
    col_bgrole_dict['n_annots'] = name_bg_role

    name_api = gt.CustomAPI(
        col_name_list,
        col_ider_dict=col_ider_dict,
        col_types_dict=partial_headers['col_types_dict'],
        col_getter_dict=col_getter_dict,
        col_bgrole_dict=col_bgrole_dict,
        col_display_role_func_dict=col_display_role_func_dict,
        col_width_dict=partial_headers['col_width_dict'],
        get_thumb_size=lambda: 221,
        col_level_dict=col_level_dict,
        iders=iders,
        sortby='n_edges',
        sort_reverse=True
        #sortby='aid1',
    )
    return name_api


def make_edge_api(infr, review_cfg={}):
    """
    TODO:
        mark an edge that would cause an inconsistency
        to be marked as a deeper red.
        These are edges we currently believe to be false
        By default edges should not be red. they should be a light unknown yellow.
        Dark unknown yellow is for noncomparable annotations

    """
    aids1, aids2 = infr.get_filtered_edges(review_cfg)

    self = EdgeAPIHelper(infr)
    partial_headers = self.make_partial_edge_headers()
    # from six import next
    # data = next(infr.graph.edges(data=True))[-1]

    #if not DEVELOPER_MODE:
    #    col_name_list.remove('data')
    col_name_list = partial_headers['edge_col_name_list']

    if not review_cfg['show_match_thumb']:
        # FIXME: do one-vs-one scoring instead
        col_name_list.remove('match_thumb')

    col_getter_dict = {
        'index': np.arange(len(aids1)),
        'aid1': aids1,
        'aid2': aids2,
    }
    col_getter_dict.update(partial_headers['col_getter_dict'])

    edge_api = gt.CustomAPI(
        col_name_list,
        col_ider_dict=partial_headers['col_ider_dict'],
        col_types_dict=partial_headers['col_types_dict'],
        col_getter_dict=col_getter_dict,
        col_bgrole_dict=partial_headers['col_bgrole_dict'],
        col_display_role_func_dict=partial_headers['col_display_role_func_dict'],
        col_width_dict=partial_headers['col_width_dict'],
        get_thumb_size=lambda: 221,
        sortby='score',
        #sortby='aid1',
        sort_reverse=True
    )
    # api = edge_api
    return edge_api


def make_node_api(infr):
    aids = sorted(list(infr.graph.nodes()))
    col_name_list = ['aid', 'thumb', 'name_label']

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


def make_name_node_api(infr, review_cfg={}):
    # TODO: only make this API if the tab is clicked
    node_to_name = infr.get_node_attrs('name_label')
    name_to_nodes = ut.group_items(node_to_name.keys(), node_to_name.values())

    self = EdgeAPIHelper(infr)

    # utool.embed()
    names = list(name_to_nodes.keys())
    flat_aids, grouped_aid_idxs = ut.invertible_flatten1(name_to_nodes.values())

    col_name_list = [
        'name_label',
        'n_annots',
        'aid',
        'thumb'
    ]
    col_level_dict = {
        'name_label': 0,
        'n_annots': 0,
        'aid': 1,
        'thumb': 1,
    }
    iders = [
        list(range(len(names))),
        grouped_aid_idxs,
    ]
    col_getter_dict = {
        'thumb': infr.ibs.get_annot_chip_thumbtup,
        'name_label': names,
        'n_annots': list(map(len, grouped_aid_idxs)),
        'aid': flat_aids
    }
    col_ider_dict = {
        'thumb': 'aid',
    }
    col_types_dict = {
        'thumb': 'PIXMAP',
    }

    col_bgrole_dict = {
        'inference' : self.get_inference_bgrole,
        'review': self.get_review_bgrole,
    }

    name_api = gt.CustomAPI(
        col_name_list,
        iders=iders,
        # col_types_dict=col_types_dict,
        col_ider_dict=col_ider_dict,
        col_getter_dict=col_getter_dict,
        col_types_dict=col_types_dict,
        col_bgrole_dict=col_bgrole_dict,
        # col_display_role_func_dict=col_display_role_func_dict,
        # col_width_dict=col_width_dict,
        # get_thumb_size=lambda: 221,
        col_level_dict=col_level_dict,
        sortby='n_annots',
        sort_reverse=True
    )
    return name_api


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


def make_qt_graph_interface(ibs, aids=None, nids=None, gids=None,
                            init_mode='review', graph_tab=False):
    r"""
    CommandLine:
        ibeis make_qt_graph_interface --dbdir ~/lev/media/hdd/work/WWF_Lynx/ --show --nids=281 --graph-tab
        ibeis make_qt_graph_interface --dbdir ~/lev/media/hdd/work/WWF_Lynx/ --show --gids=2289 --graph-tab

        ibeis make_qt_graph_interface --dbdir ~/lev/media/hdd/work/WWF_Lynx/ --show --graph-tab --aids=2587,2398
        ibeis make_qt_graph_interface --show
        ibeis make_qt_graph_interface --show --db PZ_PB_RF_TRAIN

        ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9 --graph-tab
        ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9
        ibeis make_qt_graph_interface --show

        ibeis make_qt_graph_interface --show --db RotanTurtles --aids=610,716

        ibeis make_qt_graph_interface --db LEWA_splits --nids=1 --show --sample

        ibeis make_qt_graph_interface --db PZ_MTEST --nids=1 --show --init-mode=rereview

        ibeis make_qt_graph_interface --dbdir=~/lev/media/danger/GGR/GGR-IBEIS --nids=2300 --show
        ibeis make_qt_graph_interface --dbdir=~/lev/media/danger/GGR/GGR-IBEIS --nids=4617 --show

        # unmount
        fusermount -u ~/lev
        # mount
        sshfs -o idmap=user lev:/ ~/lev
        ibeis make_qt_graph_interface --dbdir=/home/joncrall/lev/media/hdd/work/EWT_Cheetahs --show -a default:view=right;frontright;backright

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.viz.viz_graph2 import *  # NOQA
        >>> import guitool as gt
        >>> import ibeis
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> aids = ut.get_argval('--aids', type_=list, default=None)
        >>> nids = ut.get_argval('--nids', type_=list, default=None)
        >>> gids = ut.get_argval('--gids', type_=list, default=None)
        >>> init_mode = ut.get_argval('--init_mode', default='review')
        >>> graph_tab = ut.get_argflag('--graph-tab')
        >>> gt.ensure_qtapp()
        >>> win = make_qt_graph_interface(ibs, aids, nids, gids, init_mode, graph_tab)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=win, freq=10)
    """
    print('[qt_graph] make_qt_graph_interface init()')
    print('[qt_graph] nids = %s' % (ut.trunc_repr(nids),))
    print('[qt_graph] aids = %s' % (ut.trunc_repr(aids),))
    print('[qt_graph] gids = %s' % (ut.trunc_repr(gids),))
    if gids is not None:
        nids = ut.unique(ut.flatten(ibs.get_image_nids(gids)))
    if nids is not None and aids is None:
        aids = ut.flatten(ibs.get_name_aids(nids))
    if aids is None:
        # import ibeis
        # aids = ibeis.testdata_aids(ibs=ibs)
        # ['right', 'frontright', 'backright']
        # aids = ibs.get_valid_aids()
        aids = ibs.filter_annots_general(view=['right', 'frontright', 'backright'])
        # [0:20]
    if ut.get_argflag('--sample'):
        rng = np.random.RandomState(42)
        aids = rng.choice(aids, 30, replace=False)

    print('make_qt_graph_interface aids = %r' % (aids,))
    nids = ibs.get_annot_name_rowids(aids)
    # infr = graph_iden.AnnotInference(ibs, aids, nids, verbose=ut.VERBOSE)
    infr = graph_iden.AnnotInference(ibs, aids, nids, verbose=1)
    infr.initialize_graph()
    gt.ensure_qtapp()
    print('infr = %r' % (infr,))
    win = AnnotGraphWidget(infr=infr, use_image=False, init_mode=init_mode)
    abstract_interaction.register_interaction(win)
    win.show()

    if graph_tab:
        index = win.graph_tab_widget.indexOf(win.graph_tab)
        win.graph_tab_widget.setCurrentIndex(index)
        print('win.graph_widget.use_image_cb.setChecked = %r' % (win.graph_widget.use_image_cb.setChecked,))
        win.graph_widget.use_image_cb.setChecked(True)

    if False:
        win.expand_image_and_names()

    return win


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.viz.viz_graph2
        python -m ibeis.viz.viz_graph2 --allexamples
        ibeis make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9 --graph
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
