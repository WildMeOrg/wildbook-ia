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


class MatplotlibWidget(gt.GuitoolWidget):
    click_inside_signal = QtCore.pyqtSignal(MouseEvent, object)

    def initialize(self):
        from guitool import __PYQT__
        from plottool.interactions import zoom_factory, pan_factory
        if __PYQT__._internal.GUITOOL_PYQT_VERSION == 4:
            import matplotlib.backends.backend_qt4agg as backend_qt
        else:
            import matplotlib.backends.backend_qt5agg as backend_qt
        FigureCanvas = backend_qt.FigureCanvasQTAgg
        self.fig = pt.plt.figure()
        self.fig._no_raise_plottool = True
        self.canvas = FigureCanvas(self.fig)

        #self.canvas.manager = ut.DynStruct()
        #self.canvas.manager.window = self

        self.ax = self.fig.add_subplot(1, 1, 1)
        #self.ax.plot([1, 2, 3], [1, 2, 3])
        self.addWidget(self.canvas)

        self.pan_events = pan_factory(self.ax)
        self.zoon_events = zoom_factory(self.ax)
        ih.connect_callback(self.fig, 'button_press_event', self.on_click)
        ih.connect_callback(self.fig, 'draw_event', self.draw_callback)
        ih.connect_callback(self.fig, 'pick_event', self.on_pick)

        self.MOUSE_BUTTONS = abstract_interaction.AbstractInteraction.MOUSE_BUTTONS
        self.setMinimumHeight(20)
        self.setMinimumWidth(20)

    def on_click(self, event):
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
        # self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        # self.draw_artists()
        return


GRAPH_REVIEW_CFG_DEFAULTS = {
    'ranks_top': 3,
    'ranks_bot': 2,

    #'remove_nmatch_in_cc': True,
    #'remove_match_between_cc': True,

    #'ranks_top': 5,
    #'directed': False,
    #'name_scoring': True,
    'filter_reviewed': True,
    'filter_photobombs': False,

    'filter_true_matches': True,
    'filter_false_matches': False,

    'filter_nonmatch_between_ccs': True,

    'filter_dup_namepairs': True,

    #'show_chips': True,
    #'filter_duplicate_true_matches': False,
}


class AnnotGraphWidget(gt.GuitoolWidget):
    signal_graph_update = QtCore.pyqtSignal()
    signal_state_update = QtCore.pyqtSignal(bool)

    def initialize(self, infr=None, use_image=False):
        print('[graph] initialize')

        self.pcfg = {
            'can_match_samename': True,
            'K': 3,
            'Knorm': 3,
            'prescore_method': 'csum',
            'score_method': 'csum'
        }

        self.review_cfg = GRAPH_REVIEW_CFG_DEFAULTS.copy()
        self.infr = infr
        self.selected_aids = []
        # self.config = InferenceConfig()

        self.splitter = self.newSplitter(orientation=Qt.Horizontal)
        splitter = self.splitter
        self.ctrls_ = splitter.addNewWidget(orientation=Qt.Vertical,
                                            vertical_stretch=1, margin=1,
                                            spacing=1)
        self.ctrls = self.ctrls_.addNewScrollArea()

        graph_tables_widget = self.addNewTabWidget(verticalStretch=1)

        self.status_bar = self.addNewWidget(
            orientation=Qt.Horizontal, vertical_stretch=1, margin=1, spacing=1)
        self.dev_bar = self.addNewWidget(
            orientation=Qt.Horizontal, vertical_stretch=1, margin=1, spacing=1)
        self.prog_bar = self.addNewProgressBar(visible=False)

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
        self.status_bar.addNewButton('Reset Original', pressed=self.reset_original)
        self.status_bar.addNewButton('Reset Empty', pressed=self.reset_empty)
        self.status_bar.addNewButton('Edit Filters', pressed=self.edit_filters)
        self.status_bar.addNewButton('Repopulate', pressed=self.repopulate)
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
            bbar2.addNewButton(pressed=wrapped_func)

        bbar1.addNewButton('Mark: Match', pressed=self.mark_match)
        bbar1.addNewButton('Mark: Non-Match', pressed=self.mark_nonmatch)
        bbar1.addNewButton('Mark: Not-Comp', pressed=self.mark_notcomp)

        thresh_wgt = bbar1.addNewWidget(orientation=Qt.Horizontal, margin=1)
        thresh_wgt.addNewLabel('Thresh:')
        self.thresh_lbl = thresh_wgt.addNewLineEdit('.5', editable=False)

        bbar2.addNewButton('Deselect', pressed=self.deselect)
        bbar2.addNewButton('Show Annots', pressed=self.show_selected)
        bbar2.addNewButton('Infer Cut', pressed=self.infer_cut)
        _simple_button2(self.infr.apply_all)

        def refresh_via_cb(flag):
            self.draw_graph()

        self.show_cuts_cb = bbar2.addNewCheckBox(
            'Show Cuts', changed=refresh_via_cb, checked=False)
        self.show_review_cuts_cb =  bbar2.addNewCheckBox(
            'Show Reviewed Cuts', changed=refresh_via_cb, checked=True)
        self.only_reviwed_cb =  bbar2.addNewCheckBox(
            'Only Reviewed', changed=refresh_via_cb, checked=True)
        self.use_image_cb = bbar2.addNewCheckBox(
            'Show Img', changed=refresh_via_cb, checked=use_image)
        self.toggle_pin_cb = bbar2.addNewCheckBox(
            'Pin Positions', changed=self.set_pin_state, checked=False)

        #Debug row
        _simple_button3(self.reset_all)

        _simple_button3(self.infr.reset_feedback)
        _simple_button3(self.infr.reset_name_labels)

        _simple_button3(self.infr.remove_feedback)
        _simple_button3(self.infr.remove_name_labels)

        bbar3.layout().addSpacing(10)
        _simple_button3(self.infr.mst_review)
        _simple_button3(self.infr.connected_compoment_relabel)
        bbar3.layout().addSpacing(10)

        _simple_button3(self.infr.apply_mst)
        _simple_button3(self.apply_scores)
        _simple_button3(self.infr.apply_feedback)
        _simple_button3(self.infr.apply_weights)
        _simple_button3(self.infr.apply_cuts)
        _simple_button3(self.infr.remove_cuts)

        bbar3.layout().addSpacing(10)
        _simple_button3(self.update_state)
        _simple_button3(self.draw_graph, refresh=False)

        self.dev_bar.addNewButton(pressed=self.print_info)
        self.dev_bar.addNewButton(pressed=self.embed)

        self.mpl_needs_update = True
        self.cb = None
        self.mpl_wgt.click_inside_signal.connect(self.on_click_inside)

        self.signal_graph_update.connect(self.draw_graph, type=Qt.UniqueConnection)
        self.signal_state_update.connect(self.on_state_update)
        self.edge_api_widget.view.doubleClicked.connect(self.edge_doubleclick)
        self.edge_api_widget.view.contextMenuClicked.connect(self.edge_context)
        self.edge_api_widget.view.connect_keypress_to_slot(self.edge_keypress)
        self.edge_api_widget.view.connect_single_key_to_slot(gt.ALT_KEY, self.on_alt_pressed)

    def showEvent(self, event):
        super(AnnotGraphWidget, self).showEvent(event)
        print('[graph] showEvent')
        # Fire initialize event after we show the GUI
        QtCore.QTimer.singleShot(50, self.init_inference)

    def init_inference(self):
        print('[graph] init_inference')
        with gt.GuiProgContext('Initializing', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            infr = self.infr
            infr.initialize_graph()
            ctx.set_progress(1, 3)
            infr.initialize_visual_node_attrs()
            infr.remove_feedback()
            ctx.set_progress(2, 3)
            infr.remove_name_labels()
            if ut.get_argflag('--cut'):
                infr.apply_all()
            ctx.set_progress(3, 3)
        self.node2_aid = nx.get_node_attributes(self.infr.graph, 'aid')
        self.aid2_node = ut.invert_dict(self.node2_aid)
        #self.apply_scores()
        self.update_state(structure_changed=True)

    def apply_scores(self):
        with gt.GuiProgContext('Computing Scores', self.prog_bar) as ctx:
            self.infr.apply_scores(self.review_cfg, prog_hook=ctx.prog_hook)

    def reset_original(self):
        print('[graph] reset_original')
        infr = self.infr
        infr.reset_feedback()
        infr.reset_name_labels()
        infr.apply_cuts()
        infr.mst_review()
        self.update_state(structure_changed=True)

    def reset_empty(self):
        print('[graph] reset_empty')
        infr = self.infr
        infr.remove_feedback()
        infr.remove_name_labels()
        infr.apply_cuts()
        self.update_state(structure_changed=True)

    def edit_filters(self):
        import dtool
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

    def update_state(self, structure_changed=False):
        print('[graph] update_state')
        self.infr.apply_feedback()
        if structure_changed:
            self.apply_scores()
        self.infr.apply_weights()
        num_names, num_inconsistent = self.infr.connected_compoment_relabel()
        self.infr.apply_cuts()
        # Set gui status indicators
        self.num_names_lbl.setText('Names: %d' % (num_names,))
        if num_inconsistent:
            self.state_lbl.setText('Inconsistent Names: %d' % (num_inconsistent,))
            self.state_lbl.setColor('black', self.infr.truth_colors['nonmatch'][0:3] * 255)
        else:
            self.state_lbl.setText('Consistent')
            self.state_lbl.setColor('black', self.infr.truth_colors['match'][0:3] * 255)
        self.signal_state_update.emit(structure_changed)
        self.edge_api_widget.model.layoutChanged.emit()
        self.node_api_widget.model.layoutChanged.emit()

    @QtCore.pyqtSlot(bool)
    def on_state_update(self, structure_changed=False):
        if structure_changed:
            self.populate_node_model()
            self.populate_edge_model()
        if self.mpl_wgt.visibleRegion().isEmpty():
            # Flag that graph should draw next time it is visible
            self.mpl_needs_update = True
        else:
            # Draw the graph because it is visible
            self.signal_graph_update.emit()

    def reset_all(self):
        self.infr.initialize_graph()
        self.infr.initialize_visual_node_attrs()
        self.infr.reset_feedback()
        self.toggle_pin_cb.setChecked(False)

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
        edge_api = make_edge_api(self.infr, review_cfg=self.review_cfg)
        headers = edge_api.make_headers(tblnice='Edges')
        self.edge_api_widget.change_headers(headers)
        self.edge_api_widget.resize_headers(edge_api)
        self.edge_api_widget.view.verticalHeader().setVisible(True)
        self.edge_tab.setTabText('Edges (%r)' % (self.edge_api_widget.model.num_rows_total))
        self.edge_api_widget.view.verticalHeader().setDefaultSectionSize(221)

    def sizeHint(self):
        return QtCore.QSize(1100, 500)

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
        cm.ishow_single_annotmatch(self.infr.qreq_, aid2, mode=0)

    def get_edge_options(self):
        view = self.edge_api_widget.view
        selected_qtindex_list = view.selectedRows()

        def aid_pair_gen():
            for qtindex in selected_qtindex_list:
                model = qtindex.model()
                aid1  = model.get_header_data('aid1', qtindex)
                aid2  = model.get_header_data('aid2', qtindex)
                yield aid1, aid2

        def mark_pairs(state):
            for aid1, aid2 in aid_pair_gen():
                self.infr.add_feedback(aid1, aid2, state)
            self.update_state()
            #for qtindex in selected_qtindex_list:
            #    # This should work by itself
            #    self.edge_api_widget.model.dataChanged.emit(qtindex, qtindex)
            #    # but it doesnt seem to be, but this seems to solve the issue
            self.edge_api_widget.model.layoutChanged.emit()
            self.node_api_widget.model.layoutChanged.emit()
        options = [
            ('Mark &True', lambda: mark_pairs('match')),
            ('Mark &False', lambda: mark_pairs('nonmatch')),
            ('Mark &Non-Comparable', lambda: mark_pairs('notcomp')),
            ('&Unreview', lambda: mark_pairs('unreviewed')),
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
            chip_context_options = inspect_gui.make_annotpair_context_options(ibs, aid1, aid2, qreq_=qreq_)
            options += [
                ('Match Ta&gs', pair_tag_options)
            ]
            options += chip_context_options

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

        def make_option_dict(options):
            option_dict = {key[key.find('&') + 1]: val for key, val in options
                           if '&' in key}
            return option_dict

        options = self.get_edge_options()
        option_dict = make_option_dict(options)
        handled = False
        for key, func in option_dict.items():
            if event_key == getattr(QtCore.Qt, 'Key_' + key.upper()):
                func()
                handled = True
                break
        if not handled:
            print('Key  not handled %r' % (event_key,))
            return

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
        infrkw = {}
        # keys = ['min_labels', 'max_labels']
        # infrkw = ut.dict_subset(self.config, keys)
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
        print('[graph] Not done yet')
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
        dbdir = self.infr.qreq_.ibs.get_dbdir()
        expt_dir = ut.ensuredir(ut.unixjoin(dbdir, 'SPECIAL_GGR_EXPT_LOGS'))
        review_log_dir = ut.ensuredir(ut.unixjoin(expt_dir, 'review_logs'))
        log_fpath = ut.unixjoin(review_log_dir,
                                'split_log_%s.json' % (self.infr.qreq_.ibs.dbname))
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
        logger.info('_initial_feedback = ' + ut.repr2(self.infr._initial_feedback, nl=1))
        logger.info('user_feedback = ' + ut.repr2(self.infr.user_feedback, nl=1))

        ibs = self.infr.ibs
        dryrun = False
        if not dryrun:
            ibs.set_annot_names(aid_list, name_list)
        else:
            print('DRY RUN. NOT DOING ANYTHING')
        gt.user_info(self, 'Name Change Complete')

    def show_selected(self):
        print('[graph] show_selected')
        from ibeis.viz import viz_chip
        fnum = pt.ensure_fnum(10)
        print('fnum = %r' % (fnum,))
        fig = pt.figure(fnum=fnum)
        viz_chip.show_many_chips(self.infr.ibs, self.selected_aids)
        #fig.canvas.update()
        fig.show()
        fig.canvas.draw()

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

    def closeEvent(self, event):
        event.accept()
        abstract_interaction.unregister_interaction(self)

    def eventFilter(self, source, event):
        """ handle key press """
        # print("___EVENT___")
        # print('event = %r' % (event,))
        # print('source = %r' % (source,))
        if event.type() == QtCore.QEvent.Show:
            if self.mpl_needs_update:
                self.signal_graph_update.emit()
            #print('event = %r' % (event,))
            #print('source = %r' % (source,))
        return super(AnnotGraphWidget, self).eventFilter(source, event)

    @QtCore.pyqtSlot()
    def draw_graph(self):
        self.mpl_needs_update = False
        print('[graph] Start draw page')
        self.mpl_wgt.ax.cla()

        self.infr.update_visual_attrs(
            show_cuts=self.show_cuts_cb.isChecked(),
            show_reviewed_cuts=self.show_review_cuts_cb.isChecked(),
            only_reviewed=self.only_reviwed_cb.isChecked(),
        )

        # Update Qt things
        self.thresh_lbl.setText('%.2f' % (self.infr.thresh))

        # Update MPL things
        #layoutkw = dict(prog='neato', splines='spline', sep=10 / 72)

        #draw_implicit=self.show_cuts)
        self.plotinfo = pt.show_nx(self.infr.graph,
                                   layout='custom',
                                   as_directed=False,
                                   ax=self.mpl_wgt.ax,
                                   #layoutkw=layoutkw,
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
        self.draw_colorbar()
        self.mpl_wgt.canvas.draw()
        # fig.canvas.blit(ax.bbox)
        self.mpl_wgt.fig.subplots_adjust(left=.02, top=.98, bottom=.02, right=.85)
        print('[graph] End draw page')

    def draw_colorbar(self):
        xy = (1, self.infr.thresh)
        xytext = (2.5, .3 if self.infr.thresh < .5 else .7)
        #print('xy = %r' % (xy,))
        #print('xytext = %r' % (xytext,))
        #print('self.infr.thresh = %r' % (self.infr.thresh,))

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

    def print_info(self):
        print('[graph] print_info')
        print('_initial_feedback = ' + ut.repr2(self.infr._initial_feedback, nl=1))
        print('user_feedback = ' + ut.repr2(self.infr.user_feedback, nl=1))

    def embed(self):
        fig = self.mpl_wgt.fig  # NOQA
        ax = self.mpl_wgt.ax  # NOQA
        infr = self.infr  # NOQA
        ibs = infr.ibs  # NOQA
        graph = infr.graph  # NOQA
        import utool
        utool.embed()


def make_node_api(infr):
    aids = sorted(list(infr.graph.nodes()))
    col_name_list = [
        'aid',
        'data',
        'thumb',
        'name_label'
    ]
    def get_node_data(aid):
        data = infr.graph.node[aid].copy()
        ut.delete_dict_keys(data,
                            ['color', 'framewidth', 'image', 'label',
                             'pos', 'shape', 'size', 'height', 'width'])
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


def get_filtered_edges(ibs, graph, review_cfg):
    nodes = list(graph.nodes())
    uv_list = list(graph.edges())

    node_to_aids = nx.get_node_attributes(graph, 'aid')
    node_to_nids = nx.get_node_attributes(graph, 'name_label')
    aids = ut.take(node_to_aids, nodes)
    nids = ut.take(node_to_nids, nodes)
    aid_to_nid = dict(zip(aids, nids))
    nid2_aids = ut.group_items(aids, nids)

    # Initial set of edges
    aids1 = ut.take_column(uv_list, 0)
    aids2 = ut.take_column(uv_list, 1)

    num_filtered = 0

    def filter_between_ccs_neg(aids1, aids2, isneg_flags):
        """
        If two cc's have at least X=1 negative reviews then remove all other
        reviews between those cc's
        """
        neg_aids1 = ut.compress(aids1, isneg_flags)
        neg_aids2 = ut.compress(aids2, isneg_flags)
        neg_nids1 = ut.take(aid_to_nid, neg_aids1)
        neg_nids2 = ut.take(aid_to_nid, neg_aids2)

        # Ignore inconsistent names
        # Determine which CCs photobomb each other
        invalid_nid_map = ut.ddict(set)
        for nid1, nid2 in zip(neg_nids1, neg_nids2):
            if nid1 != nid2:
                invalid_nid_map[nid1].add(nid2)
                invalid_nid_map[nid2].add(nid1)

        impossible_aid_map = ut.ddict(set)
        for nid1, other_nids in invalid_nid_map.items():
            for aid1 in nid2_aids[nid1]:
                for nid2 in other_nids:
                    for aid2 in nid2_aids[nid2]:
                        impossible_aid_map[aid1].add(aid2)
                        impossible_aid_map[aid2].add(aid1)

        valid_flags = [aid2 not in impossible_aid_map[aid1]
                       for aid1, aid2 in zip(aids1, aids2)]
        return valid_flags

    if review_cfg['filter_nonmatch_between_ccs']:
        review_states = [
            graph.get_edge_data(*edge).get('reviewed_state', 'unreviewed')
            for edge in zip(aids1, aids2)]
        is_nonmatched = [state == 'nonmatch' for state in review_states]
        #isneg_flags = is_nonmatched
        valid_flags = filter_between_ccs_neg(aids1, aids2, is_nonmatched)
        num_filtered += len(valid_flags) - sum(valid_flags)
        aids1 = ut.compress(aids1, valid_flags)
        aids2 = ut.compress(aids2, valid_flags)

    if review_cfg['filter_photobombs']:
        am_list = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
        ispb_flags = ibs.get_annotmatch_prop('Photobomb', am_list)
        #isneg_flags = ispb_flags
        valid_flags = filter_between_ccs_neg(aids1, aids2, ispb_flags)
        num_filtered += len(valid_flags) - sum(valid_flags)
        aids1 = ut.compress(aids1, valid_flags)
        aids2 = ut.compress(aids2, valid_flags)

    if review_cfg['filter_true_matches']:
        nids1 = ut.take(aid_to_nid, aids1)
        nids2 = ut.take(aid_to_nid, aids2)
        valid_flags = [nid1 != nid2 for nid1, nid2 in zip(nids1, nids2)]
        num_filtered += len(valid_flags) - sum(valid_flags)
        aids1 = ut.compress(aids1, valid_flags)
        aids2 = ut.compress(aids2, valid_flags)

    if review_cfg['filter_false_matches']:
        nids1 = ut.take(aid_to_nid, aids1)
        nids2 = ut.take(aid_to_nid, aids2)
        valid_flags = [nid1 == nid2 for nid1, nid2 in zip(nids1, nids2)]
        num_filtered += len(valid_flags) - sum(valid_flags)
        aids1 = ut.compress(aids1, valid_flags)
        aids2 = ut.compress(aids2, valid_flags)

    if review_cfg['filter_reviewed']:
        valid_flags = [
            graph.get_edge_data(*edge).get('reviewed_state', 'unreviewed') == 'unreviewed'
            for edge in zip(aids1, aids2)]
        num_filtered += len(valid_flags) - sum(valid_flags)
        aids1 = ut.compress(aids1, valid_flags)
        aids2 = ut.compress(aids2, valid_flags)

    if review_cfg['filter_dup_namepairs']:
        # Only look at a maximum of one review between the current set of
        # connected compoments
        nids1 = ut.take(aid_to_nid, aids1)
        nids2 = ut.take(aid_to_nid, aids2)
        scores = np.array([
            # hack
            max(graph.get_edge_data(*edge).get('score', -1), -1)
            for edge in zip(aids1, aids2)])
        review_states = [
            graph.get_edge_data(*edge).get('reviewed_state', 'unreviewed')
            for edge in zip(aids1, aids2)]
        is_notcomp = np.array([state == 'notcomp' for state in review_states], dtype=np.bool)
        # Notcomps should not be considered in this filtering
        scores[is_notcomp] = -2
        #
        namepair_id_list = np.array(vt.compute_unique_data_ids_(
            list(zip(nids1, nids2))), dtype=np.int)
        unique_namepair_ids, namepair_groupxs = vt.group_indices(namepair_id_list)
        score_namepair_groups = vt.apply_grouping(scores, namepair_groupxs)
        unique_rowx2 = sorted([
            groupx[score_group.argmax()]
            for groupx, score_group in zip(namepair_groupxs, score_namepair_groups)
        ])
        aids1 = ut.take(aids1, unique_rowx2)
        aids2 = ut.take(aids2, unique_rowx2)

    print('[graph] num_filtered = %r' % (num_filtered,))
    return aids1, aids2


def make_edge_api(infr, review_cfg={}):
    graph = infr.graph
    ibs = infr.ibs

    def get_edge_data(edge):
        aid1, aid2 = edge
        attrs = graph.get_edge_data(aid1, aid2).copy()
        ut.delete_dict_keys(attrs, infr.visual_edge_attrs + [
            'rank', 'reviewed_state', 'score'])
        attrs = {k: v for k, v in attrs.items() if v is not None}
        attrs['name_label1'] = graph.node[aid1]['name_label']
        attrs['name_label2'] = graph.node[aid2]['name_label']
        return ut.repr2(attrs, precision=2)

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

        color = infr.truth_colors['match' if nid1 == nid2 else 'nonmatch']
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

    aids1, aids2 = get_filtered_edges(ibs, graph, review_cfg)

    col_name_list = [
        #'index',
        'aid1', 'aid2', 'score', 'rank', 'matched', 'reviewed', 'thumb1',
        'thumb2', 'match_thumb', 'timedelta', 'tags', 'data',
    ]

    col_getter_dict = {
        'index': np.arange(len(aids1)),
        'aid1': aids1,
        'aid2': aids2,
        'data': get_edge_data,
        'timedelta': lambda edge: ibs.get_unflat_annots_timedelta_list([edge])[0][0],
        'matched':  lambda edge: get_match_text(edge),
        'reviewed':  edge_attr_getter('reviewed_state', 'unreviewed'),
        'score':  edge_attr_getter('score'),
        'rank':  edge_attr_getter('rank', -1),
        'tags': lambda edge: get_pair_tags(edge),
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
        'match_thumb': 'PIXMAP',
    }

    col_display_role_func_dict = {
        'timedelta': ut.partial(ut.get_posix_timedelta_str, year=True, approx=2),
    }

    col_bgrole_dict = {
        'matched' : ut.partial(get_match_status_bgrole, ibs),
        'reviewed': ut.partial(get_reviewed_status_bgrole, ibs),
    }

    col_width_dict = {
        'index': 42,
        'aid1': 42,
        'aid2': 42,
        'score': 65,
        'rank': 42,
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
    gt.ensure_qtapp()
    print('infr = %r' % (infr,))
    win = AnnotGraphWidget(infr=infr, use_image=False)
    abstract_interaction.register_interaction(win)
    #win.resize(900, 600)
    #win.draw_graph()
    win.show()
    #win.init_inference()

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
