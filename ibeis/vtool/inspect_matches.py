# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
(print, rrr, profile) = ut.inject2(__name__)
try:
    import guitool as gt
    INSPECT_BASE = gt.GuitoolWidget
except ImportError:
    INSPECT_BASE = object


def lazy_test_annot(key):
    rchip_fpath = ut.grab_test_imgpath(key)
    annot = ut.LazyDict({
        'aid': key.split('.')[0],
        'nid': key[0:4],
        'rchip_fpath': rchip_fpath
    })
    return annot


def match_inspect_graph():
    """

    CommandLine:
        python -m vtool.inspect_matches match_inspect_graph --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.inspect_matches import *  # NOQA
        >>> import vtool as vt
        >>> gt.ensure_qapp()
        >>> ut.qtensure()
        >>> self = match_inspect_graph()
        >>> self.show()
        >>> ut.quit_if_noshow()
        >>> self.update()
        >>> gt.qtapp_loop(qwin=self, freq=10)
    """
    import vtool as vt
    annots = [lazy_test_annot('easy1.png'),
              lazy_test_annot('easy2.png'),
              lazy_test_annot('easy3.png'),
              lazy_test_annot('zebra.png'),
              lazy_test_annot('hard3.png')]
    matches = [vt.PairwiseMatch(a1, a2) for a1, a2 in ut.combinations(annots, 2)]
    self = MultiMatchInspector(matches=matches)
    return self


class MultiMatchInspector(INSPECT_BASE):

    def initialize(self, matches):
        self.matches = matches

        self.splitter = self.addNewSplitter(orientation='horiz')
        # tab_widget = self.addNewTabWidget(verticalStretch=1)
        # self.edge_tab = tab_widget.addNewTab('Edges')
        # self.match_tab = tab_widget.addNewTab('Matches')

        self.edge_api_widget = gt.APIItemWidget(
            doubleClicked=self.edge_doubleclick)
        self.match_inspector = MatchInspector(match=None)

        self.splitter.addWidget(self.edge_api_widget)
        self.splitter.addWidget(self.match_inspector)

        self.populate_edge_model()

    def edge_doubleclick(self, qtindex):
        row = qtindex.row()
        match = self.matches[row]
        self.match_inspector.set_match(match)

    def populate_edge_model(self):
        edge_api = gt.CustomAPI(
            col_name_list=['index', 'aid1', 'aid2'],
            col_getter_dict={
                'index': list(range(len(self.matches))),
                'aid1': [m.annot1['aid'] for m in self.matches],
                'aid2': [m.annot2['aid'] for m in self.matches],
            }, sort_reverse=False)
        headers = edge_api.make_headers(tblnice='Edges')
        self.edge_api_widget.change_headers(headers)
        self.edge_api_widget.resize_headers(edge_api)
        self.edge_api_widget.view.verticalHeader().setVisible(True)
        # self.edge_api_widget.view.verticalHeader().setDefaultSectionSize(24)
        # self.edge_api_widget.view.verticalHeader().setDefaultSectionSize(221)
        # self.edge_tab.setTabText('Matches (%r)' % (self.edge_api_widget.model.num_rows_total))


class MatchInspector(INSPECT_BASE):
    """
    A widget that contains
        (1) a viewport that displays an annotation pair with matches overlayed.
        (2) a control panel for tuning matching parameters
        (3) a text area displaying information about the match vector

    CommandLine:
        python -m vtool.inspect_matches MatchInspector:0 --show
        python -m vtool.inspect_matches MatchInspector:1 --show

    Example:
        >>> # SCRIPT
        >>> from vtool.inspect_matches import *  # NOQA
        >>> import vtool as vt
        >>> gt.ensure_qapp()
        >>> ut.qtensure()
        >>> annot1 = lazy_test_annot('easy1.png')
        >>> annot2 = lazy_test_annot('easy2.png')
        >>> match = vt.PairwiseMatch(annot1, annot2)
        >>> self = MatchInspector(match=match)
        >>> self.show()
        >>> ut.quit_if_noshow()
        >>> #self.update()
        >>> gt.qtapp_loop(qwin=self, freq=10)

    Example:
        >>> # SCRIPT
        >>> from vtool.inspect_matches import *  # NOQA
        >>> import vtool as vt
        >>> import ibeis
        >>> gt.ensure_qapp()
        >>> ut.qtensure()
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> annots = ibs.annots([1, 2])
        >>> annot1 = annots[0]._make_lazy_dict()
        >>> annot2 = annots[1]._make_lazy_dict()
        >>> match = vt.PairwiseMatch(annot1, annot2)
        >>> self = MatchInspector(match=match)
        >>> self.show()
        >>> ut.quit_if_noshow()
        >>> #self.update()
        >>> gt.qtapp_loop(qwin=self, freq=10)
    """

    def showEvent(self, event):
        super(MatchInspector, self).showEvent(event)
        # ut.cprint('[viz_graph] showEvent', 'green')
        # Fire initialize event after we show the GUI
        # QtCore.QTimer.singleShot(50, self.init_inference)
        self.first_show()

    def first_show(self):
        # Show the match if updating is on, otherwise just draw the annot pair
        if self.autoupdate_cb.checkState():
            self.update()
        else:
            self.draw_pair()

    def set_match(self, match=None, on_context=None, info_text=None):
        self.match = match
        self.info_text = info_text
        self.on_context = on_context
        if self.isVisible():
            self.first_show()

    def initialize(self, match=None, on_context=None, autoupdate=True,
                   info_text=None):
        from plottool import abstract_interaction
        from guitool.__PYQT__ import QtCore
        self.set_match(match, on_context, info_text)
        self._setup_configs()
        self._setup_layout(autoupdate=autoupdate)
        abstract_interaction.register_interaction(self)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.execContextMenu)

    def execContextMenu(self, qpoint):
        if self.on_context:
            options = self.on_context()
        else:
            options = [('No context set', None)]
        gt.popup_menu(self, qpoint, options)

    def embed(self):
        match = self.match  # NOQA
        import utool
        utool.embed()

    def _new_config_widget(self, cfg, changed=None):
        from guitool import PrefWidget2
        user_mode = 0
        cfg_widget = PrefWidget2.EditConfigWidget(
            config=cfg, user_mode=user_mode, parent=self, changed=changed)
        return cfg_widget

    def closeEvent(self, event):
        from plottool import abstract_interaction
        abstract_interaction.unregister_interaction(self)
        super(MatchInspector, self).closeEvent(event)

    def _setup_configs(self):
        from vtool import matching
        import dtool

        import vtool as vt

        default_dict = vt.get_extract_features_default_params()
        TmpFeatConfig = dtool.from_param_info_list([
            ut.ParamInfo(key, val) for key, val in default_dict.items()
            # ut.ParamInfo('affine_invariance', True),
            # ut.ParamInfo('rotation_invariance', False),
        ])

        self.featconfig = TmpFeatConfig()
        self.featconfig_widget = self._new_config_widget(
            self.featconfig, changed=self.on_feat_cfg_changed)

        TmpVsOneConfig = dtool.from_param_info_list(
            matching.VSONE_DEFAULT_CONFIG)
        self.config = TmpVsOneConfig()
        self.config_widget = self._new_config_widget(
            self.config, changed=self.on_cfg_changed)

        TmpDisplayConfig = dtool.from_param_info_list([
            ut.ParamInfo('overlay', True),
            ut.ParamInfo('show_all_kpts', False),
            ut.ParamInfo('mask_blend', 0.0, min_=0, max_=1),

            ut.ParamInfo('show_homog', False, hideif=':not overlay'),
            ut.ParamInfo('show_ori', False, hideif=':not overlay'),
            ut.ParamInfo('show_ell', True, hideif=':not overlay'),
            ut.ParamInfo('show_pts', False, hideif=':not overlay'),
            ut.ParamInfo('show_lines', True, hideif=lambda cfg: not cfg['overlay']),
            ut.ParamInfo('show_rect', False, hideif=':not overlay'),
            ut.ParamInfo('show_eig', False, hideif=':not overlay'),
        ])
        self.disp_config = TmpDisplayConfig()
        self.disp_config_widget = self._new_config_widget(
            self.disp_config, changed=self.on_cfg_changed)

    def _setup_layout(self, autoupdate=True):
        from guitool.__PYQT__ import QtWidgets
        self.menubar = gt.newMenubar(self)
        self.menuFile = self.menubar.newMenu('Dev')
        self.menuFile.newAction(triggered=self.embed)
        splitter1 = self.addNewSplitter(orientation='horiz')
        config_vframe = splitter1.newWidget()
        splitter2     = splitter1.addNewSplitter(orientation='vert')
        config_vframe.addWidget(QtWidgets.QLabel('Feat Config'))
        config_vframe.addWidget(self.featconfig_widget)
        config_vframe.addWidget(QtWidgets.QLabel('Query Config'))
        config_vframe.addWidget(self.config_widget)
        config_vframe.addWidget(QtWidgets.QLabel('Display Config'))
        config_vframe.addWidget(self.disp_config_widget)
        # update_hframe = config_vframe.addNewWidget(orientation='horiz')
        # update_hframe.addNewButton('Update', pressed=self.update)
        self.autoupdate_cb = config_vframe.addNewCheckBox(
            'auto-update', checked=autoupdate, changed=self.update)

        self.mpl_widget = MatplotlibWidget()
        splitter2.addWidget(self.mpl_widget)

        self.infobox = splitter2.addNewTextEdit()

    def execute_vsone(self):
        from vtool import matching
        print('[inspect_match] Execute vsone')
        cfgdict = self.config.asdict()

        feat_cfgdict = self.featconfig.asdict()

        match = self.match
        match._inplace_default = True
        matching.ensure_metadata_vsone(match.annot1, match.annot2,
                                       cfgdict=feat_cfgdict)

        match.apply_all(cfgdict)

    def draw_pair(self):
        if self.match is None:
            return
        self.mpl_widget.clf()
        ax = self.mpl_widget.ax
        info_html = ''
        if self.info_text is not None:
            info_html = '<pre>' + self.info_text + '</pre>'
        self.infobox.setText(info_html)
        self.match.show(ax=ax, overlay=False)
        self.mpl_widget.fig.canvas.draw()

    def draw_vsone(self):
        match = self.match
        summary = match._make_local_summary_feature_vector(summary_ops={'sum'})
        info_html = ''
        if self.info_text is not None:
            info_html = '<pre>' + self.info_text + '</pre>'
        feat_html = '<pre>' + ut.align(ut.repr4(summary), ':') + '</pre>'
        self.infobox.setText(info_html + feat_html)

        self.mpl_widget.clf()
        ax = self.mpl_widget.ax
        match.show(ax=ax, **self.disp_config)
        #fig.show()
        self.mpl_widget.fig.canvas.draw()

    def update(self, state=None):
        if self.autoupdate_cb.checkState() and self.match is not None:
            self.execute_vsone()
            self.draw_vsone()

    def on_cfg_changed(self, *args):
        self.update()
        self.cfg_needs_update = True

    def on_feat_cfg_changed(self, *args):
        print('Update feats')
        feat_keys = ['vecs', 'kpts', '_feats', 'flann']
        self.match.annot1._mutable = True
        self.match.annot2._mutable = True
        for key in feat_keys:
            if key in self.match.annot1:
                del self.match.annot1[key]
            if key in self.match.annot2:
                del self.match.annot2[key]
        self.update()
        self.cfg_needs_update = True


def get_qt_backend():
    from guitool import __PYQT__
    if __PYQT__._internal.GUITOOL_PYQT_VERSION == 4:
        import matplotlib.backends.backend_qt4agg as backend_qt
    else:
        import matplotlib.backends.backend_qt5agg as backend_qt
    return backend_qt


class MatplotlibWidget(INSPECT_BASE):
    """
    A qt widget that contains a matplotlib figure

    References:
        http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html
    """
    #click_inside_signal = QtCore.pyqtSignal(MouseEvent, object)

    def initialize(self):
        import matplotlib as mpl
        from plottool import abstract_interaction
        backend_qt = get_qt_backend()
        FigureCanvas = backend_qt.FigureCanvasQTAgg

        self.fig = mpl.figure.Figure()
        self.fig._no_raise_plottool = True
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.addWidget(self.canvas)
        self.reset_ax()

        # import plottool as pt
        #from plottool import interact_helpers as ih
        # self.fig = pt.plt.figure()
        #self.canvas.manager = ut.DynStruct()
        #self.canvas.manager.window = self
        #ih.connect_callback(self.fig, 'button_press_event', self.on_click)
        #ih.connect_callback(self.fig, 'draw_event', self.draw_callback)
        #ih.connect_callback(self.fig, 'pick_event', self.on_pick)
        self.MOUSE_BUTTONS = abstract_interaction.AbstractInteraction.MOUSE_BUTTONS
        self.setMinimumHeight(20)
        self.setMinimumWidth(20)

    def clf(self):
        self.fig.clf()
        self.reset_ax()

    def reset_ax(self):
        import plottool as pt
        # from plottool.interactions import zoom_factory, pan_factory
        self.ax = self.fig.add_subplot(1, 1, 1)
        pt.adjust_subplots(left=0, right=1, top=1, bottom=0, fig=self.fig)
        # self.pan_events = pan_factory(self.ax)
        # self.zoon_events = zoom_factory(self.ax)
        return self.ax

    #def on_click(self, event):
    #    from plottool import interact_helpers as ih
    #    if ih.clicked_inside_axis(event):
    #        ax = event.inaxes
    #        self.click_inside_signal.emit(event, ax)

    #def on_pick(self, event):
    #    print('PICK: event.artist = %r' % (event.artist,))
    #    edge = pt.get_plotdat(event.artist, 'edge')
    #    node = pt.get_plotdat(event.artist, 'node')
    #    edge_data = pt.get_plotdat(event.artist, 'edge_data')
    #    node_data = pt.get_plotdat(event.artist, 'node_data')
    #    if edge_data is not None:
    #        print('edge = %r' % (edge,))
    #        print('edge_data = %s' % (ut.repr3(edge_data),))
    #        pass
    #    if node_data is not None:
    #        print('node = %r' % (node,))
    #        pass

    def draw_callback(self, event):
        # self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        # self.draw_artists()
        return


def make_match_interaction(matches, metadata, type_='RAT+SV', **kwargs):
    import plottool.interact_matches
    #import plottool as pt
    fm, fs = matches[type_][0:2]
    try:
        H1 = metadata['H_' + type_.split('+')[0]]
    except Exception:
        H1 = None
    #fm, fs = matches['RAT'][0:2]
    annot1 = metadata['annot1']
    annot2 = metadata['annot2']
    rchip1, kpts1, vecs1 = ut.dict_take(annot1, ['rchip', 'kpts', 'vecs'])
    rchip2, kpts2, vecs2 = ut.dict_take(annot2, ['rchip', 'kpts', 'vecs'])
    #pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm, fs=fs)
    fsv = fs[:, None]
    interact = plottool.interact_matches.MatchInteraction2(
        rchip1, rchip2, kpts1, kpts2, fm, fs, fsv, vecs1, vecs2, H1=H1,
        **kwargs)
    return interact


def show_matching_dict(matches, metadata, *args, **kwargs):
    interact = make_match_interaction(matches, metadata, *args, **kwargs)
    interact.show_page()
    return interact
    #MatchInteraction2


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m vtool.inspect_matches
        python -m vtool.inspect_matches --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
