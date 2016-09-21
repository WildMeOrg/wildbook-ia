# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
#import numpy as np
(print, rrr, profile) = ut.inject2(__name__)
try:
    import guitool as gt
    INSPECT_BASE = gt.GuitoolWidget
except ImportError:
    INSPECT_BASE = object


class MatchInspector(INSPECT_BASE):
    """
    CommandLine:
        python -m vtool.inspect_matches MatchInspector --show

    Example:
        >>> from vtool.inspect_matches import *  # NOQA
        >>> gt.ensure_qapp()
        >>> ut.qt4ensure()
        >>> rchip_fpath1 = ut.grab_test_imgpath('easy1.png')
        >>> rchip_fpath2 = ut.grab_test_imgpath('easy2.png')
        >>> metadata = ut.LazyDict()
        >>> metadata['annot1'] = ut.LazyDict()
        >>> metadata['annot2'] = ut.LazyDict()
        >>> metadata['annot1']['rchip_fpath'] = rchip_fpath1
        >>> metadata['annot2']['rchip_fpath'] = rchip_fpath2
        >>> self = MatchInspector(metadata=metadata)
        >>> self.show()
        >>> ut.quit_if_noshow()
        >>> self.update()
        >>> gt.qtapp_loop(qwin=self, freq=10)
    """

    def initialize(self, metadata):
        self.metadata = metadata
        self._setup_configs()
        self._setup_layout()

    def _setup_configs(self):
        from guitool import PrefWidget2
        from vtool import matching
        import dtool

        class TmpVsoneConfig(dtool.Config):
            _param_info_list = (
                matching.VSONE_DEFAULT_CONFIG
            )
        self.config = TmpVsoneConfig()

        class TmpDisplayConfig(dtool.Config):
            _param_info_list = [
                ut.ParamInfo('show_homog', False)
            ]
        self.disp_config = TmpDisplayConfig()

        def new_confg_widget(cfg, changed=None):
            user_mode = 0
            cfg_widget = PrefWidget2.EditConfigWidget(
                config=cfg, user_mode=user_mode, parent=self, changed=changed)
            return cfg_widget

        self.config_widget = new_confg_widget(self.config,
                                              changed=self.on_cfg_changed)
        self.disp_config_widget = new_confg_widget(self.disp_config,
                                                   changed=self.on_cfg_changed)

    def _setup_layout(self):
        self.mpl_widget = MatplotlibWidget()
        from guitool.__PYQT__ import QtWidgets
        splitter = self.addNewSplitter(orientation='horiz')
        config_vframe = splitter.newWidget()
        config_vframe.addWidget(QtWidgets.QLabel('Query Config'))
        config_vframe.addWidget(self.config_widget)
        config_vframe.addWidget(QtWidgets.QLabel('Display Config'))
        config_vframe.addWidget(self.disp_config_widget)
        config_vframe.addNewButton('Update', pressed=self.update)
        splitter.addWidget(self.mpl_widget)

    def execute_vsone(self):
        from vtool import matching
        print('Execute vsone')
        cfgdict = self.config.asdict()
        print('cfgdict = %s' % (ut.repr2(cfgdict),))
        metadata = self.metadata
        match = matching.vsone_matching(metadata, cfgdict)

        import plottool as pt
        annot1 = self.metadata['annot1']
        annot2 = self.metadata['annot2']
        rchip1, kpts1, vecs1 = ut.dict_take(annot1, ['rchip', 'kpts', 'vecs'])
        rchip2, kpts2, vecs2 = ut.dict_take(annot2, ['rchip', 'kpts', 'vecs'])
        type_ = 'RAT+SV'
        fm, fs = match.matches[type_][0:2]

        if self.disp_config['show_homog']:
            H1 = metadata['H_' + type_.split('+')[0]]
        else:
            H1 = None

        #fsv = fs[:, None]
        fig = self.mpl_widget.fig
        ax = self.mpl_widget.ax
        ax.cla()
        #fig.clf()
        print('fm = %r' % (len(fm),))
        ax, xywh1, xywh2 = pt.show_chipmatch2(
            rchip1, rchip2, kpts1, kpts2, fm, fs, H1=H1,
            ax=ax, colorbar_=False,
        )
        #fig.show()
        fig.canvas.draw()

    def update(self):
        print('update')
        import vtool as vt
        vt.rrrr()
        self.execute_vsone()

    def on_cfg_changed(self):
        self.update()
        self.cfg_needs_update = True


class MatplotlibWidget(INSPECT_BASE):
    #click_inside_signal = QtCore.pyqtSignal(MouseEvent, object)

    def initialize(self):
        import plottool as pt
        from plottool.interactions import zoom_factory, pan_factory
        #from plottool import interact_helpers as ih
        from plottool import abstract_interaction
        from guitool import __PYQT__
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
        #ih.connect_callback(self.fig, 'button_press_event', self.on_click)
        #ih.connect_callback(self.fig, 'draw_event', self.draw_callback)
        #ih.connect_callback(self.fig, 'pick_event', self.on_pick)

        self.MOUSE_BUTTONS = abstract_interaction.AbstractInteraction.MOUSE_BUTTONS
        self.setMinimumHeight(20)
        self.setMinimumWidth(20)

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
