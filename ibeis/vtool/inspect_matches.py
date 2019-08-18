# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import ubelt as ub
try:
    import guitool as gt
    from guitool import mpl_widget
    INSPECT_BASE = gt.GuitoolWidget
    MatplotlibWidget = mpl_widget.MatplotlibWidget
except ImportError:
    MatplotlibWidget = object
    INSPECT_BASE = object


def lazy_test_annot(key):
    import numpy as np
    rchip_fpath = ut.grab_test_imgpath(key)
    annot = ut.LazyDict({
        'aid': key.split('.')[0],
        'nid': key[0:4],
        'rchip_fpath': rchip_fpath,
        'gps': (np.nan, np.nan),
        'yaw': np.nan,
        'view': np.nan,
        'qual': np.nan,
        'time': np.nan,
    })
    return annot

try:
    import dtool as dt

    MatchDisplayConfig = dt.from_param_info_list([
        ut.ParamInfo('overlay', True),
        ut.ParamInfo('show_all_kpts', False),
        ut.ParamInfo('mask_blend', 0.0, min_=0, max_=1),

        ut.ParamInfo('heatmask', True, hideif=':not overlay'),
        ut.ParamInfo('show_homog', False, hideif=':not overlay'),
        ut.ParamInfo('show_ori', False, hideif=':not overlay'),
        ut.ParamInfo('show_ell', False, hideif=':not overlay'),
        ut.ParamInfo('show_pts', False, hideif=':not overlay'),
        ut.ParamInfo('show_lines', False, hideif=lambda cfg: not cfg['overlay']),
        ut.ParamInfo('show_rect', False, hideif=':not overlay'),
        ut.ParamInfo('show_eig', False, hideif=':not overlay'),
        ut.ParamInfo('ell_alpha', 0.6, min_=0, max_=1, hideif=':not overlay'),
        ut.ParamInfo('line_alpha', 0.35, min_=0, max_=1, hideif=':not overlay'),
    ])
except ImportError:
    pass


class MatchInspector(INSPECT_BASE):
    """
    A widget that contains
        (1) a viewport that displays an annotation pair with matches overlayed.
        (2) a control panel for tuning matching parameters
        (3) a text area displaying information about the match vector

    CommandLine:
        python -m vtool.inspect_matches MatchInspector:0 --show
        python -m vtool.inspect_matches MatchInspector:1 --show

        python -m vtool.inspect_matches MatchInspector:1 --db GZ_Master1 --aids=1041,1045 --show

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
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aids = ub.argval('--aids', default=[1, 2])
        >>> print('aids = %r' % (aids,))
        >>> annots = ibs.annots(aids)
        >>> annot1 = annots[0]._make_lazy_dict()
        >>> annot2 = annots[1]._make_lazy_dict()
        >>> cfgdict = MatchDisplayConfig().asdict()
        >>> cfgdict = ut.argparse_dict(cfgdict)
        >>> match = vt.PairwiseMatch(annot1, annot2)
        >>> self = MatchInspector(match=match, cfgdict=cfgdict)
        >>> self.show()
        >>> ut.quit_if_noshow()
        >>> #self.update()
        >>> gt.qtapp_loop(qwin=self, freq=10)
    """

    def showEvent(self, event):
        super(MatchInspector, self).showEvent(event)
        # Fire initialize event after we show the GUI
        # QtCore.QTimer.singleShot(50, self.init_inference)
        self.first_show()

    def first_show(self, state=None):
        if self.match is not None:
            # Show the match if updating is on, otherwise just draw the annot
            # pair
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
                   info_text=None, cfgdict=None):
        from plottool import abstract_interaction
        from guitool.__PYQT__ import QtCore
        self.set_match(match, on_context, info_text)
        self._setup_configs(cfgdict=cfgdict)
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

    def screenshot(self):
        import plottool as pt
        with pt.RenderingContext() as render:
            self.match.show(**self.disp_config)
        fpaths = gt.newFileDialog('.', mode='save', exec_=True)
        if fpaths is not None and len(fpaths) > 0:
            fpath = fpaths[0]
            if not fpath.endswith('.jpg'):
                fpath += '.jpg'
            import vtool as vt
            vt.imwrite(fpath, render.image)

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

    def _setup_configs(self, cfgdict=None):
        from vtool import matching
        import dtool
        # import pyhesaff

        # default_dict = pyhesaff.get_hesaff_default_params()
        # default_dict = vt.get_extract_features_default_params()
        TmpFeatConfig = dtool.from_param_info_list(matching.VSONE_FEAT_CONFIG)

        TmpNChipConfig = dtool.from_param_info_list(matching.NORM_CHIP_CONFIG)
        # [
        #     ut.ParamInfo(key, val) for key, val in default_dict.items()
        #     # ut.ParamInfo('affine_invariance', True),
        #     # ut.ParamInfo('rotation_invariance', False),
        # ])

        self.featconfig = TmpFeatConfig()
        self.chipconfig = TmpNChipConfig()

        TmpVsOneConfig = dtool.from_param_info_list(
            matching.VSONE_DEFAULT_CONFIG)
        self.config = TmpVsOneConfig()
        self.disp_config = MatchDisplayConfig()

        if cfgdict is not None:
            print('[inspect_match] default cfgdict = %r' % (cfgdict,))
            self.config.update(**cfgdict)
            self.featconfig.update(**cfgdict)
            self.chipconfig.update(**cfgdict)
            self.disp_config.update(**cfgdict)

        # Make config widgets after setting defaults
        self.chipconfig_widget = self._new_config_widget(
            self.chipconfig, changed=self.on_chip_cfg_changed)
        self.featconfig_widget = self._new_config_widget(
            self.featconfig, changed=self.on_feat_cfg_changed)
        self.config_widget = self._new_config_widget(
            self.config, changed=self.on_cfg_changed)
        self.disp_config_widget = self._new_config_widget(
            self.disp_config, changed=self.on_cfg_changed)

    def _setup_layout(self, autoupdate=True):
        from guitool.__PYQT__ import QtWidgets
        self.menubar = gt.newMenubar(self)
        self.menuFile = self.menubar.newMenu('Dev')
        self.menuFile.newAction(triggered=self.embed)
        self.menuFile.newAction(triggered=self.screenshot)
        splitter1 = self.addNewSplitter(orientation='horiz')
        config_vframe = splitter1.newWidget()
        splitter2     = splitter1.addNewSplitter(orientation='vert')
        config_vframe.addWidget(QtWidgets.QLabel('Chip Config'))
        config_vframe.addWidget(self.chipconfig_widget)
        config_vframe.addWidget(QtWidgets.QLabel('Feat Config'))
        config_vframe.addWidget(self.featconfig_widget)
        config_vframe.addWidget(QtWidgets.QLabel('Query Config'))
        config_vframe.addWidget(self.config_widget)
        config_vframe.addWidget(QtWidgets.QLabel('Display Config'))
        config_vframe.addWidget(self.disp_config_widget)
        # update_hframe = config_vframe.addNewWidget(orientation='horiz')
        # update_hframe.addNewButton('Update', pressed=self.update)
        self.autoupdate_cb = config_vframe.addNewCheckBox(
            'auto-update', checked=autoupdate, changed=self.first_show)

        self.mpl_widget = MatplotlibWidget(parent=self)
        splitter2.addWidget(self.mpl_widget)

        self.infobox = splitter2.addNewTextEdit()

    def execute_vsone(self):
        from vtool import matching
        print('[inspect_match] Execute vsone')

        cfgdict = {}
        cfgdict.update(self.featconfig.asdict())
        cfgdict.update(self.chipconfig.asdict())

        match = self.match
        match.verbose = True
        match._inplace_default = True
        matching.ensure_metadata_vsone(match.annot1, match.annot2,
                                       cfgdict=cfgdict)

        match_config = self.config.asdict()
        match.apply_all(match_config)

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
        feat_html = '<pre>' + ut.align(ub.repr2(summary), ':') + '</pre>'
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

    def on_chip_cfg_changed(self, *args):
        print('Update feats')
        feat_keys = ['nchip', 'vecs', 'kpts', '_feats', 'flann']
        self.match.annot1._mutable = True
        self.match.annot2._mutable = True
        for key in feat_keys:
            if key in self.match.annot1:
                del self.match.annot1[key]
            if key in self.match.annot2:
                del self.match.annot2[key]
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
    rchip1, kpts1, vecs1 = ub.dict_take(annot1, ['nchip', 'kpts', 'vecs'])
    rchip2, kpts2, vecs2 = ub.dict_take(annot2, ['nchip', 'kpts', 'vecs'])
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


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool.inspect_matches
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
