# -*- coding: utf-8 -*-
"""
This module controls the GUI backend.  It is the layer between the GUI frontend
(newgui.py) and the IBEIS controller.  All the functionality of the nonvisual
gui components is written or called from here

TODO:
    open_database should not allow you to open subfolders

    python -m utool.util_inspect check_module_usage --pat="guiback.py"


Notes:
    LAYOUT TERMS;
        Margins / Content Margins;
           - space around the widgets in the layout
        Spacing
           - space between widges in the layout
        Stretch
           - relative size ratio vector (1 component for each widget)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import sys
import functools
import traceback  # NOQA
import utool as ut
import ubelt as ub
import wbia.guitool as gt
from wbia.guitool import slot_, signal_, cast_from_qt
from wbia.guitool.__PYQT__ import QtCore, QtGui, QtWidgets
from wbia import constants as const
from wbia.other import ibsfuncs
from wbia import sysres
from wbia import viz
from wbia.control import IBEISControl
from wbia.gui import clock_offset_gui
from wbia.gui import guiexcept
from wbia.gui import guiheaders as gh
from wbia.gui import newgui
from wbia.viz import interact
from os.path import exists, join, dirname, normpath
from wbia.plottool import fig_presenter
from six.moves import zip

(print, rrr, profile) = ut.inject2(__name__, '[back]')


VERBOSE = ut.VERBOSE


def backreport(func):
    """
    reports errors on backend functions
    should be around every function by default
    """

    def backreport_wrapper(back, *args, **kwargs):
        try:
            result = func(back, *args, **kwargs)
        except guiexcept.UserCancel:
            print('handling user cancel')
            return None
        except Exception as ex:
            # error_msg = "Error caught while performing function. \n %r" % ex
            error_msg = 'Error: %s' % (ex,)
            import traceback  # NOQA

            detailed_msg = traceback.format_exc()
            gt.msgbox(title='Error Catch!', msg=error_msg, detailed_msg=detailed_msg)
            raise
        return result

    backreport_wrapper = ut.preserve_sig(backreport_wrapper, func)
    return backreport_wrapper


def backblock(func):
    """ BLOCKING DECORATOR
    TODO: This decorator has to be specific to either front or back. Is there a
    way to make it more general?
    """

    @functools.wraps(func)
    # @gt.checks_qt_error
    @backreport
    def bacblock_wrapper(back, *args, **kwargs):
        _wasBlocked_ = back.front.blockSignals(True)
        try:
            result = func(back, *args, **kwargs)
        except Exception:
            # error_msg = "Error caught while performing function. \n %r" % ex
            # gt.msgbox(title="Error Catch!", msg=error_msg)
            raise
        finally:
            back.front.blockSignals(_wasBlocked_)
        return result

    bacblock_wrapper = ut.preserve_sig(bacblock_wrapper, func)
    return bacblock_wrapper


def blocking_slot(*types_):
    """
    A blocking slot accepts the types which are passed to QtCore.pyqtSlot.
    In addition it also causes the gui frontend to block signals while
    the decorated function is processing.
    """

    def wrap_bslot(func):
        # @slot_(*types_)
        @QtCore.pyqtSlot(*types_)
        @backblock
        @functools.wraps(func)
        def wrapped_bslot(*args, **kwargs):
            result = func(*args, **kwargs)
            sys.stdout.flush()
            return result

        wrapped_bslot = ut.preserve_sig(wrapped_bslot, func)
        return wrapped_bslot

    return wrap_bslot


class CustomAnnotCfgSelector(gt.GuitoolWidget):
    """
    CommandLine:
        python -m wbia.gui.guiback CustomAnnotCfgSelector --show
        python -m wbia.gui.guiback CustomAnnotCfgSelector --show --db PZ_MTEST
        python -m wbia.gui.guiback CustomAnnotCfgSelector --show --debugwidget
        python -m wbia.gui.guiback show_advanced_id_interface --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.gui.guiback import *  # NOQA
        >>> import wbia
        >>> gt.ensure_qtapp()
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> self = CustomAnnotCfgSelector(ibs)
        >>> rect = gt.QtWidgets.QDesktopWidget().availableGeometry(screen=0)
        >>> self.move(rect.x(), rect.y())
        >>> self.show()
        >>> self.apply_new_config()
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=self, freq=10)
    """

    def __init__(self, ibs):
        from wbia.expt import annotation_configs
        from wbia import dtool
        from wbia.guitool import PrefWidget2
        from wbia.guitool.__PYQT__.QtCore import Qt

        super(CustomAnnotCfgSelector, self).__init__()
        self.ibs = ibs

        self.qaids = None
        self.accept_flag = False

        class TmpAnnotConfig(dtool.Config):
            _param_info_list = (
                annotation_configs.INDEPENDENT_DEFAULTS_PARAM_INFO
                + annotation_configs.INTRAGROUP_DEFAULTS_PARAM_INFO
                + annotation_configs.SAMPLE_DEFAULTS_PARAM_INFO
                + annotation_configs.SUBINDEX_DEFAULTS_PARAM_INFO
            )

        class TmpPipelineConfig(dtool.Config):
            _param_info_list = [
                ut.ParamInfo('K', ibs.cfg.query_cfg.nn_cfg.K, min_=1, none_ok=False),
                ut.ParamInfo(
                    'Knorm', ibs.cfg.query_cfg.nn_cfg.Knorm, min_=1, none_ok=False
                ),
                # ibs.cfg.query_cfg.nn_cfg.lookup_paraminfo('Knorm'),
                ibs.cfg.query_cfg.nnweight_cfg.lookup_paraminfo('normalizer_rule'),
                ut.ParamInfo(
                    'fgw_thresh',
                    ibs.cfg.query_cfg.flann_cfg.fgw_thresh,
                    type_=float,
                    min_=0,
                    max_=1,
                ),
                ut.ParamInfo(
                    'query_rotation_heuristic', ibs.cfg.query_cfg.query_rotation_heuristic
                ),
                ut.ParamInfo(
                    'minscale_thresh',
                    ibs.cfg.query_cfg.flann_cfg.minscale_thresh,
                    type_=float,
                ),
                ut.ParamInfo(
                    'maxscale_thresh',
                    ibs.cfg.query_cfg.flann_cfg.maxscale_thresh,
                    type_=float,
                ),
                ibs.cfg.query_cfg.nnweight_cfg.lookup_paraminfo('can_match_samename'),
                # ut.ParamInfo('normalizer_rule', ibs.cfg.query_cfg.nnweight_cfg.normalizer_rule),
                # ut.ParamInfo('AI', True),
            ]

        self.qcfg = TmpAnnotConfig()
        self.dcfg = TmpAnnotConfig()

        self.pcfg = TmpPipelineConfig()
        self.review_cfg = dtool.Config.from_dict(
            {'filter_reviewed': True, 'ranks_top': 1, 'filter_true_matches': True}
        )
        self.info_cfg = dtool.Config.from_dict(
            {key: False for key in ibs.parse_annot_config_stats_filter_kws()}
        )
        self.exemplar_cfg = dtool.Config.from_dict(
            {
                # 'imgsetid': None,
                'exemplars_per_view': ibs.cfg.other_cfg.exemplars_per_view,
            }
        )

        self.info_cfg['species_hist'] = True
        self.info_cfg['per_vp'] = True
        self.info_cfg['per_qual'] = True
        self.info_cfg['hashid'] = True
        self.info_cfg['per_name'] = True
        self.info_cfg['hashid_visual'] = True
        self.info_cfg['hashid_uuid'] = True
        self.info_cfg['per_multiple'] = True

        for cfg in [self.qcfg, self.dcfg]:
            cfg['minqual'] = 'good'
            cfg['reviewed'] = True
            cfg['multiple'] = False
            # cfg['min_pername'] = 0
            # from wbia.other import ibsfuncs
            cfg['species'] = self.ibs.get_primary_database_species()
            cfg['require_viewpoint'] = True
            cfg['view'] = ibsfuncs.get_primary_species_viewpoint(cfg['species'])
            # 'right,frontright,backright'

        self.setWindowTitle('Custom Annot Selector')

        # cfg_size_policy = (QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        def new_confg_widget(cfg, changed=None):
            user_mode = 0
            cfg_widget = PrefWidget2.EditConfigWidget(
                config=cfg, user_mode=user_mode, parent=self, changed=changed
            )
            # cfg_widget.setSizePolicy(*cfg_size_policy)
            return cfg_widget

        self.editQueryConfig = new_confg_widget(self.qcfg, changed=self.on_cfg_changed)
        self.editDataConfig = new_confg_widget(self.dcfg, changed=self.on_cfg_changed)
        self.editPipeConfig = new_confg_widget(self.pcfg, changed=self.on_cfg_changed)
        self.editReviewConfig = new_confg_widget(self.review_cfg)
        self.editInfoConfig = new_confg_widget(self.info_cfg)
        # self.editExemplarConfig = new_confg_widget(self.exemplar_cfg, changed=self.on_cfg_changed)

        tabwgt = self.addNewTabWidget(verticalStretch=1)
        tab1 = tabwgt.addNewTab('Custom Query')
        tab2 = tabwgt.addNewTab('Saved Queries')

        table = self.saved_queries = QtWidgets.QTableWidget()
        table.doubleClicked.connect(self.on_table_doubleclick)
        tab2.addWidget(self.saved_queries)

        splitter = tab1.addNewSplitter(orientation=Qt.Vertical)

        acfg_hframe = splitter.newWidget(orientation=Qt.Horizontal)

        query_vframe = acfg_hframe.addNewVWidget()
        query_vframe.addWidget(QtWidgets.QLabel('Query Config'))
        query_vframe.addWidget(self.editQueryConfig)

        data_vframe = acfg_hframe.addNewVWidget()
        data_vframe.addWidget(QtWidgets.QLabel('Data Config'))
        data_vframe.addWidget(self.editDataConfig)
        # data_vframe.setVisible(False)

        info_vframe = acfg_hframe.addNewVWidget()
        info_vframe.addNewLabel('Exemplar Config')
        self.editExemplarConfig = info_vframe.addNewEditConfigWidget(
            config=self.exemplar_cfg, changed=self.on_cfg_changed
        )
        info_vframe.addNewButton('Set Exemplars', pressed=self.set_exemplars)

        pcfg_hframe = splitter.newWidget(orientation=Qt.Horizontal)
        pipe_vframe = pcfg_hframe.addNewVWidget()
        pipe_vframe.addNewLabel('Pipeline Config')
        pipe_vframe.addWidget(self.editPipeConfig)

        review_vframe = pcfg_hframe.addNewVWidget()
        review_vframe.addNewLabel('Review Config')
        review_vframe.addWidget(self.editReviewConfig)

        info_vframe = pcfg_hframe.addNewVWidget()
        info_vframe.addNewLabel('Info Config')
        info_vframe.addWidget(self.editInfoConfig)

        # stats_vwidget = splitter.newWidget(orientation=Qt.Vertical)
        stats_vwidget = splitter.newWidget(orientation=Qt.Vertical, verticalStretch=1)
        stats_vwidget.addNewLabel('Expanded Annot Info (Info Config Changes Display)')
        self.qstats = QtWidgets.QTextEdit()
        self.qstats.setReadOnly(True)
        self.qstats.setWordWrapMode(QtGui.QTextOption.WrapAnywhere)
        stats_vwidget.addWidget(self.qstats)
        # Hack a copy for tab2
        self.qstats2 = QtWidgets.QTextEdit()
        self.qstats2.setReadOnly(True)
        self.qstats2.setWordWrapMode(QtGui.QTextOption.WrapAnywhere)
        tab2.addWidget(self.qstats2)
        # self.layout().addWidget(self.qstats)

        button_bar = self.addNewHWidget()

        self.update_button = button_bar.addNewButton(
            'Apply New Config', pressed=self.apply_new_config
        )
        self.execute_button = button_bar.addNewButton(
            'Execute New Query', pressed=self.execute_query
        )
        self.load_bc_button = button_bar.addNewButton(
            'Load Cached Query', pressed=self.load_bc
        )

        # button_bar.addNewButton('TestLog', pressed=self.log_query)
        button_bar.addNewButton('Embed', pressed=self.embed)
        # testlog = QtWidgets.QPushButton('TestLog')
        # testlog.pressed.connect(self.log_query)
        # button_bar.addWidget(testlog)

        self.prog_bar = self.addNewProgressBar(visible=False)
        gt.fix_child_attr_heirarchy(self, 'setSpacing', 0)
        gt.fix_child_attr_heirarchy(self, 'setMargin', 2)
        self.cfg_needs_update = True
        self.bc_info = None
        self.load_bc_button.setEnabled(self.bc_info is not None)
        self.load_bc_button.setDisabled(self.bc_info is None)
        # layout.addWidget(self.update_button, 3, 2, 1, 1)
        # layout.addWidget(self.editQueryConfig)
        # layout.addWidget(self.editDataConfig)
        self.populate_table()

    def set_exemplars(self):
        print('set exemplars')
        ibs = self.ibs
        print('self.exemplar_cfg = %r' % (self.exemplar_cfg,))
        with gt.GuiProgContext('Querying', self.prog_bar) as ctx:  # NOQA
            ibs.set_exemplars_from_quality_and_viewpoint(
                prog_hook=ctx.prog_hook, **self.exemplar_cfg
            )

    def onstart(self):
        if self.saved_queries.rowCount() > 0:
            self.load_previous_query(self.saved_queries.rowCount() - 1)

    def sizeHint(self):
        return QtCore.QSize(900, 960)

    def embed(self):
        import utool

        utool.embed()

    def make_commandline_str(self):
        from wbia.expt import experiment_helpers
        from wbia.expt import annotation_configs

        cfgdict_list, pipecfg_list = experiment_helpers.get_pipecfg_list(
            ['default:'], ibs=self.ibs
        )
        default_pcfg = dict(pipecfg_list[0].parse_items())
        # Hackish ways to get cmdline string
        p = ut.get_cfg_lbl(self.pcfg.asdict(), name='default', default_cfg=default_pcfg)
        a = (
            'default'
            + annotation_configs.get_varied_acfg_labels(
                [self.acfg, annotation_configs.default]
            )[0]
        )
        dbdir = self.ibs.get_dbdir()
        wbia_part = ['python', '-m', 'wbia']
        data_part = ['--dbdir', dbdir, '-a', a, '-t', p]
        cmd_parts = wbia_part + ['draw_rank_cmc'] + data_part + ['--show']
        cmdstr = ' '.join(cmd_parts)
        return cmdstr

    def populate_table(self):
        # data = {'col1': ['1','2','3'], 'col2':['4','5','6'], 'col3':['7','8','9']}
        print('Updating saved query table')
        from wbia.guitool.__PYQT__.QtCore import Qt

        self.table_data = self.get_saved_queries()
        horHeaders = ['fname', 'num_qaids', 'num_daids', 'has_bc']
        data = self.table_data
        table = self.saved_queries
        self.saved_queries.setColumnCount(len(horHeaders))
        print('Populating table')
        for n, key in ut.ProgIter(enumerate(horHeaders), lbl='pop table'):
            if n == 0:
                self.saved_queries.setRowCount(len(data[key]))
            for m, item in enumerate(data[key]):
                newitem = QtWidgets.QTableWidgetItem(str(item))
                table.setItem(m, n, newitem)
                newitem.setFlags(newitem.flags() ^ Qt.ItemIsEditable)
        table.setHorizontalHeaderLabels(horHeaders)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        print('Finished populating table')

    def on_cfg_changed(self):
        self.cfg_needs_update = True
        # print('detected change')

    @property
    def acfg(self):
        acfg = {'qcfg': self.qcfg.asdict(), 'dcfg': self.dcfg.asdict()}
        return acfg

    def expt_query_dir(self):
        # dbdir = self.ibs.get_dbdir()
        dbdir = self.ibs.get_dbdir()
        expt_dir = ut.ensuredir(ut.unixjoin(dbdir, 'SPECIAL_GGR_EXPT_LOGS'))
        expt_query_dir = ut.ensuredir(ut.unixjoin(expt_dir, 'saved_queries'))
        return expt_query_dir

    def apply_new_config(self):
        print('apply_new_config')
        with gt.GuiProgContext('Updating', self.prog_bar) as ctx:  # NOQA
            ctx.set_total(5)
            ibs = self.ibs
            # Discard the loaded query info
            self.query_info = None
            ctx.set_progress(1)
            self.qaids = ibs.sample_annots_general(**self.qcfg)
            ctx.set_progress(2)
            self.daids = ibs.sample_annots_general(**self.dcfg)
            ctx.set_progress(3)
            ctx.set_progress(4)
            self.update_config_info()
            self.execute_button.setText('Execute New Query')
            ctx.set_progress(5)
        print('apply_new_config is done')
        self.cfg_needs_update = False
        self.bc_info = None
        self.load_bc_button.setEnabled(self.bc_info is not None)
        self.load_bc_button.setDisabled(self.bc_info is None)

    def update_config_info(self, extra=None):
        ibs = self.ibs
        from wbia.algo.Config import QueryConfig

        # use defaults instead of back's ibs.cfg
        query_cfg = QueryConfig()
        self.qreq_ = self.ibs.new_query_request(
            self.qaids, self.daids, cfgdict=self.pcfg, query_cfg=query_cfg
        )
        qreq_ = self.qreq_
        stats_dict = ibs.get_annotconfig_stats(qreq_.qaids, qreq_.daids, **self.info_cfg)
        cmdstr = self.make_commandline_str()
        stat_parts = ut.filter_Nones(
            [
                cmdstr,
                extra,
                'pipe_cfgstr=%s' % (self.qreq_.get_cfgstr(with_data=False)),
                ut.repr2(stats_dict, strvals=True),
            ]
        )
        stats_str = '\n'.join(stat_parts)
        print(stats_str)
        self.qstats.setPlainText(stats_str)
        self.qstats2.setPlainText(stats_str)
        pass

    @property
    def expanded_aids(self):
        return (self.qaids, self.daids)

    def get_saved_queries(self):
        print('Reading saved queries')
        expt_query_dir = self.expt_query_dir()
        prev_queries = ut.glob(expt_query_dir, 'long_*.json')
        data = ut.ddict(list)
        from os.path import basename

        for long_fpath in sorted(prev_queries):
            short_fpath = long_fpath.replace('long', 'short')
            data['long_path'].append(long_fpath)
            data['short_path'].append(short_fpath)
            data['fname'].append(basename(long_fpath))
            short_info = ut.load_json(short_fpath)
            # Fix old formats
            # if 'expanded_uuids' in short_info:
            #     quuids, duuids = short_info['expanded_uuids']
            #     short_info['num_qaids'] = len(quuids)
            #     short_info['num_daids'] = len(duuids)
            #     del short_info['expanded_uuids']
            #     if 'expanded_aids' in short_info:
            #         del short_info['expanded_aids']
            #     ut.save_json(short_fpath, short_info, pretty=True)
            # if 'num_daids' not in short_info:
            #     long_info = ut.load_json(long_fpath)
            #     quuids, duuids = long_info['expanded_uuids']
            #     short_info['num_qaids'] = len(quuids)
            #     short_info['num_daids'] = len(duuids)
            #     ut.save_json(short_fpath, short_info, pretty=True)

            data['num_qaids'].append(short_info.get('num_qaids', '?'))
            data['num_daids'].append(short_info.get('num_daids', '?'))
            data['has_bc'].append('bc_info' in short_info)
        print('Finished reading saved queries')
        return data

    def load_previous_query(self, row):
        print('loading previous query')
        print('apply_new_config')
        with gt.GuiProgContext('Loading', self.prog_bar) as ctx:  # NOQA
            ctx.set_total(5)
            ctx.set_progress(0)
            long_fpath = self.table_data['long_path'][row]
            short_fpath = self.table_data['short_path'][row]

            query_info = ut.load_json(long_fpath)
            query_info_short_text = ut.load_text(short_fpath)
            ctx.set_progress(1)

            self.bc_info = query_info.get('bc_info')
            self.load_bc_button.setEnabled(self.bc_info is not None)
            self.load_bc_button.setDisabled(self.bc_info is None)

            self.query_info = query_info

            quuids, duuids = query_info['expanded_uuids']
            ibs = self.ibs
            self.qaids = ibs.get_annot_aids_from_uuid(quuids)
            ctx.set_progress(2)
            self.daids = ibs.get_annot_aids_from_uuid(duuids)
            ctx.set_progress(3)

            self.editPipeConfig.set_to_external(self.query_info['pcfg'])
            self.editQueryConfig.set_to_external(self.query_info['acfg']['qcfg'])
            self.editDataConfig.set_to_external(self.query_info['acfg']['dcfg'])
            # self.pcfg.update(**)
            # TODO: Update acfg as well.

            self.update_config_info('SAVED MANIFEST INFO:' + query_info_short_text)
            print('...loaded previous query')
            self.execute_button.setText('Re-Execute Saved Query')
            # need to do this or block signals from editPipeConfig
            self.cfg_needs_update = False
            ctx.set_progress(5)

    def log_query(self, qreq_=None, test=True):
        """ DEPRICATE """
        expt_query_dir = self.expt_query_dir()
        # ut.vd(expt_query_dir)
        ibs = self.ibs

        # TODO: Save the BIGCACHE file to the log, this allows
        # us to re-load that query even if its slightly invalid

        if qreq_ is None and test:
            from wbia.algo.Config import QueryConfig

            query_cfg = QueryConfig()
            qreq_ = self.ibs.new_query_request(
                self.qaids, self.daids, cfgdict=self.pcfg, query_cfg=query_cfg
            )

        ts = ut.get_timestamp(isutc=True, timezone=True)

        expt_long_fpath = ut.unixjoin(
            expt_query_dir, 'long_expt_%s_%s.json' % (self.ibs.dbname, ts)
        )
        expt_short_fpath = ut.unixjoin(
            expt_query_dir, 'short_expt_%s_%s.json' % (self.ibs.dbname, ts)
        )

        bc_info = qreq_.get_bigcache_info()

        query_info = ut.odict(
            [
                ('computer', ut.get_computer_name()),
                ('timestamp', ts),
                ('num_qaids', len(self.qaids)),
                ('num_daids', len(self.daids)),
                ('pcfg', self.pcfg.asdict()),
                ('acfg', self.acfg),
                ('review_cfg', self.review_cfg.asdict()),
                ('expanded_aids', (self.qaids, self.daids)),
                (
                    'expanded_uuids',
                    (ibs.get_annot_uuids(self.qaids), ibs.get_annot_uuids(self.daids)),
                ),
                ('qreq_cfgstr', qreq_.get_cfgstr(with_input=True)),
                ('qparams', qreq_.qparams),
                (
                    'qvuuid_hash',
                    qreq_.ibs.get_annot_hashid_visual_uuid(self.qaids, prefix='Q'),
                ),
                (
                    'dvuuid_hash',
                    qreq_.ibs.get_annot_hashid_visual_uuid(self.daids, prefix='D'),
                ),
                ('bc_info', bc_info),
            ]
        )

        if test:
            del query_info['expanded_aids']
            del query_info['expanded_uuids']
            del query_info['qparams']
            print('expt_long_fpath = %r' % (expt_long_fpath,))
            print('expt_short_fpath = %r' % (expt_short_fpath,))
            print('query_info = %s' % (ut.to_json(query_info, pretty=1),))
        else:
            ut.save_json(expt_long_fpath, query_info)
            del query_info['expanded_uuids']
            del query_info['expanded_aids']
            del query_info['qparams']
            ut.save_json(expt_short_fpath, query_info, pretty=1)

    @backreport
    @slot_()
    def execute_query(self):
        print('accept')
        # assert not self.cfg_needs_update, 'NEED TO APPLY ACFG/PCFG BEFORE EXECUTING'
        if self.cfg_needs_update:
            options = ['Apply now and continue', 'Apply now and wait']
            reply = gt.user_option(
                msg=ut.codeblock(
                    """
                    Information display is out of date. You should apply the
                    modified configuration before you continue.
                    """
                ),
                options=options,
            )

            if reply == options[0]:
                self.apply_new_config()
            elif reply == options[1]:
                self.apply_new_config()
                raise guiexcept.UserCancel()
            else:
                raise guiexcept.UserCancel()
        self.accept_flag = True
        from wbia.gui import inspect_gui

        review_cfg = self.review_cfg.asdict().copy()

        ibs = self.ibs
        qreq_ = self.qreq_

        if self.query_info is None:
            # Dont log on a re-executed query
            self.log_query(qreq_, test=False)

        with gt.GuiProgContext('Querying', self.prog_bar) as ctx:  # NOQA
            cm_list = qreq_.execute(prog_hook=ctx.prog_hook)

        qres_wgt = inspect_gui.QueryResultsWidget(
            ibs, cm_list, qreq_=qreq_, review_cfg=review_cfg
        )
        self.qres_wgt = qres_wgt
        qres_wgt.show()
        qres_wgt.raise_()
        print('Showing query results')
        if self.query_info is None:
            self.populate_table()

    def load_bc(self):
        from wbia.gui import inspect_gui

        review_cfg = self.review_cfg.asdict().copy()
        bc_dpath, bc_fname, bc_cfgstr = self.bc_info
        qaid2_cm = ut.load_cache(bc_dpath, bc_fname, bc_cfgstr)
        qreq_ = self.qreq_
        ibs = self.ibs
        cm_list = [qaid2_cm[qaid] for qaid in qreq_.qaids]
        qres_wgt = inspect_gui.QueryResultsWidget(
            ibs, cm_list, qreq_=qreq_, review_cfg=review_cfg
        )
        self.qres_wgt = qres_wgt
        qres_wgt.show()
        qres_wgt.raise_()

    def on_table_doubleclick(self, index):
        # print('index = %r' % (index,))
        row = index.row()
        self.load_previous_query(row)


class NewDatabaseWidget(gt.GuitoolWidget):
    r"""
    Args:
        parent (None): (default = None)

    CommandLine:
        python -m wbia.gui.guiback NewDatabaseWidget --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.gui.guiback import *  # NOQA
        >>> gt.ensure_qtapp()
        >>> self = NewDatabaseWidget(back=None)
        >>> self.resize(400, 200)
        >>> self.show()
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=self, freq=10)
    """

    def initialize(self, back=None, mode='new', on_chosen=None):
        # Save arguments
        if back is not None:
            self.back = back
            if on_chosen is None:
                on_chosen = back.open_database
            self.on_chosen = on_chosen
            self.workdir = back.get_work_directory()
        else:
            self.back = None
            self.workdir = ut.truepath('.')
            self.on_chosen = None

        title_mode = {
            'new': 'Create a new IBEIS Database',
            'copy': 'Create an IBEIS Database copy',
        }
        instruction_mode = {
            'new': 'Choose a name for the new database',
            'copy': 'Choose a name for the database copy',
        }

        self.dbname = 'MyNewIBEISDatabase'
        if mode == 'copy':
            self.dbname = back.ibs.get_dbname() + '_Copy'

        # Build layout
        self.setWindowTitle(title_mode[mode])
        self.instructions = self.addNewLabel(instruction_mode[mode], align='center')
        # ---
        self.dbname_row = self.addNewHWidget()
        self.dbname_row.edit = self.dbname_row.addNewLineEdit(self.dbname, align='center')
        self.dbname_row.edit.textChanged.connect(self.update_state)
        # ---
        self.workdir_row = self.addNewHWidget()
        self.workdir_row.lbl = self.workdir_row.addNewLabel('Current Workdir:')
        self.workdir_row.edit = self.workdir_row.addNewLineEdit(
            self.workdir, align='right'
        )
        self.workdir_row.button = self.workdir_row.addNewButton(
            '...', shrink_to_text=True, pressed=self.change_workdir
        )
        self.workdir_row.viewbut = self.workdir_row.addNewButton(
            '➤', shrink_to_text=True, pressed=self.view_workdir
        )
        self.workdir_row.edit.textChanged.connect(self.update_state)
        # ---
        self.current_row = self.addNewHWidget()
        self.create_but = self.newButton(
            'Create in workdir', pressed=self.create_in_workdir
        )
        self.current_row.lbl = self.current_row.addNewLabel(
            'Current choice:', align='left'
        )
        self.current_row.edit = self.current_row.addNewLabel(
            '{current_dbdir}', align='right'
        )

        self.button_row = self.addNewHWidget()
        self.button_row.addNewButton('Cancel', pressed=self.cancel)
        self.button_row.addNewButton(
            'Create in a different directory', pressed=self.create_in_customdir
        )
        self.button_row.addWidget(self.create_but)

        self.update_state()

    def update_state(self):
        workdir = ut.truepath(self.workdir_row.edit.text())
        dbname = self.dbname_row.edit.text()
        current_choice = normpath(join(workdir, dbname))
        workdir_exists = ut.checkpath(workdir, verbose=False)
        print('workdir_exists = %r' % (workdir_exists,))
        if workdir_exists:
            if ut.checkpath(current_choice, verbose=False):
                self.current_row.edit.setColorFG((0, 0, 255))
                self.create_but.setText('Open existing database')
            else:
                self.current_row.edit.setColorFG(None)
                self.create_but.setText('Create in workdir')
            self.create_but.setEnabled(True)
        else:
            self.current_row.edit.setColorFG((255, 0, 0))
            self.create_but.setText('Create in workdir')
            self.create_but.setEnabled(False)
        self.current_row.edit.setText(current_choice)

    def view_workdir(self):
        ut.view_directory(ut.truepath(self.workdir_row.edit.text()))

    def change_workdir(self):
        print('change workdir')
        ut.colorprint('change workdir', 'yellow')
        new_workdir = gt.select_directory(
            'Select new work directory',
            other_sidebar_dpaths=[self.workdir_row.edit.text()],
        )
        if new_workdir is not None:
            print('new_workdir = %r' % (new_workdir,))
            self.workdir_row.edit.setText(new_workdir)
            self.update_state()
            if self.back is not None:
                import wbia

                wbia.sysres.set_workdir(work_dir=new_workdir, allow_gui=False)

    def create_in_workdir(self):
        print('Create in Workdir')
        ut.colorprint('Create in Workdir', 'yellow')
        self.choose_new_dbdir(self.current_row.edit.text())

    def create_in_customdir(self):
        print('Create in Custom')
        new_dbdir = gt.select_directory(
            'Select directory for %s' % (self.dbname),
            other_sidebar_dpaths=[self.workdir_row.edit.text()],
        )
        current_choice = normpath(join(new_dbdir, self.dbname))
        self.choose_new_dbdir(current_choice)

    def choose_new_dbdir(self, dbdir):
        print('Chose dbdir = %r' % (dbdir,))
        ut.colorprint('Chose dbdir = %r' % (dbdir,), 'yellow')
        if self.on_chosen is not None:
            self.on_chosen(dbdir)
        self.close()

    def cancel(self):
        print('Cancel')
        ut.colorprint('Cancel', 'yellow')
        self.close()


# ------------------------
# Backend MainWindow Class
# ------------------------
# QtReloadingMetaClass = ut.reloading_meta_metaclass_factory(gt.QtCore.pyqtWrapperType)

GUIBACK_BASE = QtCore.QObject


# @six.add_metaclass(QtReloadingMetaClass)  # cant do this quit yet
class MainWindowBackend(GUIBACK_BASE):
    """
    Sends and recieves signals to and from the frontend

    Args:
        ibs (wbia.IBEISController):  image analysis api(default = None)

    CommandLine:
        python -m wbia.gui.guiback MainWindowBackend --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.gui.guiback import *  # NOQA
        >>> import wbia
        >>> back = testdata_guiback(defaultdb=None)
        >>> ut.quit_if_noshow()
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=back.front, freq=10)
    """

    # Backend Signals
    updateWindowTitleSignal = signal_(str)
    # changeSpeciesSignal = signal_(str)
    # incQuerySignal = signal_(int)

    # ------------------------
    # Constructor
    # ------------------------
    def __init__(back, ibs=None):
        """ Creates GUIBackend object """
        # GUIBACK_BASE.__init__(back)
        super(MainWindowBackend, back).__init__()
        if ut.VERBOSE:
            print('[back] MainWindowBackend.__init__(ibs=%r)' % (ibs,))
        back.ibs = None
        back.cfg = None
        back.edit_prefs_wgt = None
        # State variables
        back.sel_aids = []
        back.sel_nids = []
        back.sel_gids = []
        back.sel_cm = []
        # if ut.is_developer():
        back.daids_mode = None
        # else:
        # back.daids_mode = const.VS_EXEMPLARS_KEY
        # back.imageset_query_results = ut.ddict(dict)
        # used to store partials defined in the frontend
        back.special_query_funcs = {}
        # Create GUIFrontend object
        back.mainwin = newgui.IBEISMainWindow(back=back, ibs=ibs)
        back.front = back.mainwin.ibswgt
        back.web_ibs = None
        back.wb_server_running = None
        back.ibswgt = back.front  # Alias
        # connect signals and other objects
        fig_presenter.register_qt4_win(back.mainwin)
        # register self with the wbia controller
        back.register_self()
        # back.changeSpeciesSignal.connect(back.ibswgt.species_combo.setItemText)

        # back.incQuerySignal.connect(back.incremental_query_slot)

    # def __del__(back):
    #    back.cleanup()

    def set_daids_mode(back, new_mode):
        if new_mode == 'toggle':
            if back.daids_mode == const.VS_EXEMPLARS_KEY:
                back.daids_mode = const.INTRA_OCCUR_KEY
            else:
                back.daids_mode = const.VS_EXEMPLARS_KEY
        else:
            back.daids_mode = new_mode
        try:
            back.mainwin.actionToggleQueryMode.setText(
                'Toggle Query Mode currently: %s' % back.daids_mode
            )
        except Exception as ex:
            ut.printex(ex)
        # back.front.menuActions.

    def cleanup(back):
        back.kill_web_server_parallel()
        if back.wb_server_running:
            back.shutdown_wildbook()
        try:
            if back.ibs is not None:
                back.ibs.remove_observer(back)
        except Exception as ex:
            ut.printex(ex, '[back] observer fail')

    # @ut.indent_func
    def notify(back):
        """ Observer's notify function. """
        back.refresh_state()

    # @ut.indent_func
    def notify_controller_killed(back):
        """ Observer's notify function that the wbia controller has been killed. """
        back.ibs = None

    def register_self(back):
        if back.ibs is not None:
            back.ibs.register_observer(back)

    # ------------------------
    # Draw Functions
    # ------------------------

    def show(back):
        back.mainwin.show()

    def get_background_web_domain(back, **kwargs):
        web_url = '127.0.0.1'
        web_port = back.ibs.get_web_port_via_scan()
        if web_port is None:
            raise ValueError('IA web server is not running on any expected port')
        web_domain = '%s:%s' % (web_url, web_port,)
        return web_domain

    def show_imgsetid_list_in_web(back, imgsetid_list, **kwargs):
        import webbrowser

        back.start_web_server_parallel(browser=False)
        imgsetid_list = ut.ensure_iterable(imgsetid_list)
        imgsetid_str = ','.join(map(str, imgsetid_list))
        web_domain = back.get_background_web_domain()
        url = 'http://%s/view/images/?imgsetid=%s' % (web_domain, imgsetid_str,)
        webbrowser.open(url)

    def show_imgsetid_detection_turk_in_web(back, imgsetid_list, **kwargs):
        import webbrowser

        back.start_web_server_parallel(browser=False)
        imgsetid_list = ut.ensure_iterable(imgsetid_list)
        imgsetid_str = ','.join(map(str, imgsetid_list))
        web_domain = back.get_background_web_domain()
        url = 'http://%s/turk/detection/?imgsetid=%s' % (web_domain, imgsetid_str,)
        webbrowser.open(url)

    def show_imgsetid_annotation_turk_in_web(back, imgsetid_list, **kwargs):
        import webbrowser

        back.start_web_server_parallel(browser=False)
        imgsetid_list = ut.ensure_iterable(imgsetid_list)
        imgsetid_str = ','.join(map(str, imgsetid_list))
        web_domain = back.get_background_web_domain()
        url = 'http://%s/turk/annotation/?imgsetid=%s' % (web_domain, imgsetid_str,)
        webbrowser.open(url)

    def show_image(back, gid, sel_aids=[], web=False, **kwargs):
        if web:
            back.show_images_in_web(gid)
        else:
            kwargs.update({'sel_aids': sel_aids, 'select_callback': back.select_gid})
            interact.ishow_image(back.ibs, gid, **kwargs)

    def show_images_in_web(back, gid_list, **kwargs):
        import webbrowser

        back.start_web_server_parallel(browser=False)
        gid_list = ut.ensure_iterable(gid_list)
        gid_text = ','.join(map(str, gid_list))
        web_domain = back.get_background_web_domain()
        if len(gid_list) == 1:
            url = 'http://%s/view/detection?gid=%s' % (web_domain, gid_text,)
        else:
            url = 'http://%s/view/images?gid=%s' % (web_domain, gid_text,)
        webbrowser.open(url)

    def show_annotation(back, aid, show_image=False, web=False, **kwargs):
        if web:
            import webbrowser

            back.start_web_server_parallel(browser=False)
            web_domain = back.get_background_web_domain()
            url = 'http://%s/view/annotations?aid=%s' % (web_domain, aid,)
            webbrowser.open(url)
        else:
            interact.ishow_chip(back.ibs, aid, **kwargs)

        if show_image:
            gid = back.ibs.get_annot_gids(aid)
            # interact.ishow_image(back.ibs, gid, sel_aids=[aid])
            back.show_image(gid, sel_aids=[aid], web=web, **kwargs)

    def show_aid_list_in_web(back, aid_list, **kwargs):
        import webbrowser

        back.start_web_server_parallel(browser=False)

        if not isinstance(aid_list, (tuple, list)):
            aid_list = [aid_list]
        if len(aid_list) > 0:
            aid_list = ','.join(map(str, aid_list))
        else:
            aid_list = ''

        web_domain = back.get_background_web_domain()
        url = 'http://%s/view/annotations?aid=%s' % (web_domain, aid_list,)
        webbrowser.open(url)

    def show_name(back, nid, sel_aids=[], **kwargs):
        kwargs.update({'sel_aids': sel_aids, 'select_aid_callback': back.select_aid})
        # nid = back.ibs.get_name_rowids_from_text(name)
        interact.ishow_name(back.ibs, nid, **kwargs)
        pass

    def show_nid_list_in_web(back, nid_list, **kwargs):
        import webbrowser

        back.start_web_server_parallel(browser=False)

        if not isinstance(nid_list, (tuple, list)):
            nid_list = [nid_list]

        aids_list = back.ibs.get_name_aids(nid_list)
        aid_list = []
        for aids in aids_list:
            if len(aids) > 0:
                aid_list.append(aids[0])

        if len(aid_list) > 0:
            aid_str = ','.join(map(str, aid_list))
        else:
            aid_str = ''

        web_domain = back.get_background_web_domain()
        url = 'http://%s/view/names?aid=%s' % (web_domain, aid_str,)
        webbrowser.open(url)

    def show_hough_image_(back, gid, **kwargs):
        species = back.get_selected_species()
        viz.show_hough_image(back.ibs, gid, species=species, **kwargs)
        viz.draw()

    def run_detection_on_imageset(back, imgsetid_list, refresh=True, **kwargs):
        gid_list = ut.flatten(back.ibs.get_imageset_gids(imgsetid_list))
        back.run_detection_on_images(gid_list, refresh=refresh, **kwargs)

    def run_detection_on_images(back, gid_list, refresh=True, **kwargs):
        ibs = back.ibs
        detector = back.ibs.cfg.detect_cfg.detector
        if detector in ['cnn_yolo', 'yolo', 'cnn']:
            ibs.detect_cnn_yolo(gid_list)
        elif detector in ['random_forest', 'rf']:
            species = back.ibs.cfg.detect_cfg.species_text
            ibs.detect_random_forest(gid_list, species)
        else:
            raise ValueError('Detector not recognized')
        if refresh:
            back.front.update_tables([gh.IMAGE_TABLE])

    def show_probability_chip(back, aid, **kwargs):
        viz.show_probability_chip(back.ibs, aid, **kwargs)
        viz.draw()

    @blocking_slot()
    def review_queries(back, cm_list, qreq_=None, review_cfg={}, query_title=''):
        # Qt QueryResults Interaction
        from wbia.gui import inspect_gui

        ibs = back.ibs

        def finished_review_callback():
            try:
                # TODO: only call this if connected to wildbook
                # TODO: probably need to remove verboseity as well
                if back.wb_server_running:
                    back.ibs.wildbook_signal_annot_name_changes()
            except Exception as ex:
                ut.printex(ex, 'Wildbook call did not work. Maybe not connected?')
            back.front.update_tables()

        # Overwrite
        review_cfg = review_cfg.copy()
        filter_reviewed = review_cfg.pop('filter_reviewed', None)
        if filter_reviewed is None:
            filter_reviewed = ibs.cfg.other_cfg.ensure_attr('filter_reviewed', True)
            if filter_reviewed is None:
                # only filter big queries if not specified
                filter_reviewed = len(cm_list) > 6
        print('EVIDENCE_DECISION QUERIES')
        print('review_cfg = %s' % (ut.repr3(review_cfg),))
        print('filter_reviewed = %s' % (filter_reviewed,))
        review_cfg['filter_reviewed'] = filter_reviewed
        review_cfg['ranks_top'] = review_cfg.get('ranks_top', ibs.cfg.other_cfg.ranks_top)
        back.qres_wgt = inspect_gui.QueryResultsWidget(
            ibs,
            cm_list,
            callback=finished_review_callback,
            qreq_=qreq_,
            query_title=query_title,
            review_cfg=review_cfg,
        )
        back.qres_wgt.show()
        back.qres_wgt.raise_()

    # ----------------------
    # State Management Functions
    # ----------------------

    def refresh_state(back):
        """ Blanket refresh function. Try not to call this """
        back.front.update_tables()
        back.ibswgt.update_species_available(reselect=True)

    def connect_wbia_control(back, ibs):
        if ut.VERBOSE:
            print('[back] connect_wbia(ibs=%r)' % (ibs,))
        if ibs is None:
            return None
        back.ibs = ibs
        # register self with the wbia controller
        back.register_self()
        # deselect
        back._set_selection(sel_gids=[], sel_aids=[], sel_nids=[], sel_imgsetids=[None])
        back.front.connect_wbia_control(ibs)
        exemplar_gsid = ibs.get_imageset_imgsetids_from_text(const.EXEMPLAR_IMAGESETTEXT)
        num_exemplars = len(ibsfuncs._get_gids_in_imgsetid(ibs, exemplar_gsid))
        if num_exemplars == 0:
            back.daids_mode = const.INTRA_OCCUR_KEY
        else:
            back.daids_mode = const.VS_EXEMPLARS_KEY
        back.set_daids_mode(back.daids_mode)

    @blocking_slot()
    def default_config(back):
        """ Button Click -> Preferences Defaults """
        print('[back] default preferences')
        back.ibs._default_config()
        back.edit_prefs_wgt.refresh_layout()
        back.edit_prefs_wgt.pref_model.rootPref.save()
        # due to weirdness of Preferences structs
        # we have to close the widget otherwise we will
        # be looking at an outated object
        back.edit_prefs_wgt.close()

    @ut.indent_func
    def get_selected_gid(back):
        """ selected image id """
        if len(back.sel_gids) == 0:
            if len(back.sel_aids) == 0:
                sel_gids = back.ibs.get_annot_gids(back.sel_aids)
                if len(sel_gids) == 0:
                    raise guiexcept.InvalidRequest('There are no selected images')
                gid = sel_gids[0]
                return gid
            raise guiexcept.InvalidRequest('There are no selected images')
        gid = back.sel_gids[0]
        return gid

    @ut.indent_func
    def get_selected_aids(back):
        """ selected annotation id """
        if len(back.sel_aids) == 0:
            raise guiexcept.InvalidRequest('There are no selected ANNOTATIONs')
        # aid = back.sel_aids[0]
        return back.sel_aids

    @ut.indent_func
    def get_selected_imgsetid(back):
        """ selected imageset id """
        if len(back.sel_imgsetids) == 0:
            raise guiexcept.InvalidRequest('There are no selected ImageSets')
        imgsetid = back.sel_imgsetids[0]
        return imgsetid

    # --------------------------------------------------------------------------
    # Selection Functions
    # --------------------------------------------------------------------------

    def _set_selection2(back, tablename, id_list, mode='set'):
        # here tablename is a backend const tablename

        def set_collections(old, aug):
            return ut.ensure_iterable(aug)

        def add_collections(old, aug):
            return list(set(old) | set(ut.ensure_iterable(aug)))

        def diff_collections(old, aug):
            return list(set(old) - set(ut.ensure_iterable(aug)))

        modify_collections = {
            'set': set_collections,
            'add': add_collections,
            'diff': diff_collections,
        }[mode]

        attr_map = {
            const.ANNOTATION_TABLE: 'sel_aids',
            const.IMAGE_TABLE: 'sel_gids',
            const.NAME_TABLE: 'sel_nids',
        }
        attr = attr_map[tablename]
        new_id_list = modify_collections(getattr(back, attr), id_list)
        setattr(back, attr, new_id_list)

    def _set_selection3(back, tablename, id_list, mode='set'):
        """
        text = '51e10019-968b-5f2e-2287-8432464d7547 '
        """

        def ensure_uuids_are_ids(id_list, uuid_to_id_fn):
            import uuid

            if len(id_list) > 0 and isinstance(id_list[0], uuid.UUID):
                id_list = uuid_to_id_fn(id_list)
            return id_list

        def ensure_texts_are_ids(id_list, text_to_id_fn):
            if len(id_list) > 0 and isinstance(id_list[0], six.string_types):
                id_list = text_to_id_fn(id_list)
            return id_list

        if tablename == const.ANNOTATION_TABLE:
            id_list = ensure_uuids_are_ids(
                id_list, back.ibs.get_annot_aids_from_visual_uuid
            )
            aid_list = ut.ensure_iterable(id_list)
            nid_list = back.ibs.get_annot_nids(aid_list)
            gid_list = back.ibs.get_annot_gids(aid_list)
            flag_list = ut.flag_None_items(gid_list)
            nid_list = ut.filterfalse_items(nid_list, flag_list)
            gid_list = ut.filterfalse_items(gid_list, flag_list)
            aid_list = ut.filterfalse_items(aid_list, flag_list)
        elif tablename == const.IMAGE_TABLE:
            id_list = ensure_uuids_are_ids(id_list, back.ibs.get_image_gids_from_uuid)
            gid_list = ut.ensure_iterable(id_list)
            aid_list = ut.flatten(back.ibs.get_image_aids(gid_list))
            nid_list = back.ibs.get_annot_nids(aid_list)
            flag_list = ut.flag_None_items(nid_list)
            aid_list = ut.filterfalse_items(aid_list, flag_list)
            aid_list = ut.filterfalse_items(aid_list, flag_list)
        elif tablename == const.NAME_TABLE:
            id_list = ensure_texts_are_ids(id_list, back.ibs.get_name_rowids_from_text_)
            nid_list = ut.ensure_iterable(id_list)
            aid_list = ut.flatten(back.ibs.get_name_aids(nid_list))
            gid_list = back.ibs.get_annot_gids(aid_list)
            flag_list = ut.flag_None_items(gid_list)
            aid_list = ut.filterfalse_items(aid_list, flag_list)
            gid_list = ut.filterfalse_items(gid_list, flag_list)
        back._set_selection2(const.ANNOTATION_TABLE, aid_list, mode)
        back._set_selection2(const.NAME_TABLE, nid_list, mode)
        back._set_selection2(const.IMAGE_TABLE, gid_list, mode)
        return id_list

    def _clear_selection(back):
        print('[back] _clear_selection')
        back.sel_aids = []
        back.sel_gids = []
        back.sel_nids = []

    def update_selection_texts(back):
        if back.ibs is None:
            return
        sel_imagesettexts = back.ibs.get_imageset_text(back.sel_imgsetids)
        if sel_imagesettexts == [None]:
            sel_imagesettexts = []
        else:
            sel_imagesettexts = map(str, sel_imagesettexts)
        back.ibswgt.set_selection_status(gh.IMAGESET_TABLE, sel_imagesettexts)
        back.ibswgt.set_selection_status(gh.IMAGE_TABLE, back.sel_gids)
        back.ibswgt.set_selection_status(gh.ANNOTATION_TABLE, back.sel_aids)
        back.ibswgt.set_selection_status(gh.NAMES_TREE, back.sel_nids)

    def _set_selection(
        back,
        sel_gids=None,
        sel_aids=None,
        sel_nids=None,
        sel_cm=None,
        sel_imgsetids=None,
        mode='set',
        **kwargs
    ):
        def modify_collection_attr(self, attr, aug, mode):
            aug = ut.ensure_iterable(aug)
            old = getattr(self, attr)
            if mode == 'set':
                new = aug
            elif mode == 'add':
                new = list(set(old) + set(aug))
            elif mode == 'remove':
                new = list(set(old) - set(aug))
            else:
                raise AssertionError('uknown mode=%r' % (mode,))
            setattr(self, attr, new)

        if sel_imgsetids is not None:
            sel_imgsetids = ut.ensure_iterable(sel_imgsetids)
            back.sel_imgsetids = sel_imgsetids
            sel_imagesettexts = back.ibs.get_imageset_text(back.sel_imgsetids)
            if sel_imagesettexts == [None]:
                sel_imagesettexts = []
            else:
                sel_imagesettexts = map(str, sel_imagesettexts)
            back.ibswgt.set_selection_status(gh.IMAGESET_TABLE, sel_imagesettexts)
        if sel_gids is not None:
            modify_collection_attr(back, 'sel_gids', sel_gids, mode)
            back.ibswgt.set_selection_status(gh.IMAGE_TABLE, back.sel_gids)
        if sel_aids is not None:
            sel_aids = ut.ensure_iterable(sel_aids)
            back.sel_aids = sel_aids
            back.ibswgt.set_selection_status(gh.ANNOTATION_TABLE, back.sel_aids)
        if sel_nids is not None:
            sel_nids = ut.ensure_iterable(sel_nids)
            back.sel_nids = sel_nids
            back.ibswgt.set_selection_status(gh.NAMES_TREE, back.sel_nids)
        if sel_cm is not None:
            raise NotImplementedError('no select cm implemented')
            back.sel_sel_qres = sel_cm

    def select_imgsetid(back, imgsetid=None, **kwargs):
        """ Table Click -> Result Table """
        imgsetid = cast_from_qt(imgsetid)
        if False:
            prefix = ut.get_caller_name(range(1, 8))
        else:
            prefix = ''
        print(prefix + '[back] select imageset imgsetid=%r' % (imgsetid))
        back._set_selection(sel_imgsetids=imgsetid, **kwargs)

    def select_gid(
        back, gid, imgsetid=None, show=True, sel_aids=None, fnum=None, web=False, **kwargs
    ):
        r"""
        Table Click -> Image Table

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> print('''
            >>>           get_valid_gids
            >>>           ''')
            >>> valid_gids = ibs.get_valid_gids()
            >>> print('''
            >>>           get_valid_aids
            >>>           ''')
            >>> valid_aids = ibs.get_valid_aids()
            >>> #
            >>> print('''
            >>> * len(valid_aids) = %r
            >>> * len(valid_gids) = %r
            >>> ''' % (len(valid_aids), len(valid_gids)))
            >>> assert len(valid_gids) > 0, 'database images cannot be empty for test'
            >>> #
            >>> gid = valid_gids[0]
            >>> aid_list = ibs.get_image_aids(gid)
            >>> aid = aid_list[-1]
            >>> back.select_gid(gid, aids=[aid])
        """
        # Select the first ANNOTATION in the image if unspecified
        if sel_aids is None:
            sel_aids = back.ibs.get_image_aids(gid)
            if len(sel_aids) > 0:
                sel_aids = sel_aids[0:1]
            else:
                sel_aids = []
        print(
            '[back] select_gid(gid=%r, imgsetid=%r, sel_aids=%r)'
            % (gid, imgsetid, sel_aids)
        )
        back._set_selection(
            sel_gids=gid, sel_aids=sel_aids, sel_imgsetids=imgsetid, **kwargs
        )
        if show:
            back.show_image(gid, sel_aids=sel_aids, fnum=fnum, web=web)

    def copy_species_to_imageset(back, aid, imgsetid=None, refresh=True):
        species_rowid = back.ibs.get_annot_species_rowids(aid)
        aid_list = back.ibs.get_imageset_aids(imgsetid)
        species_rowid_list = [species_rowid] * len(aid_list)
        back.ibs.set_annot_species_rowids(aid_list, species_rowid_list)
        if refresh:
            back.front.update_tables([gh.IMAGE_TABLE, gh.ANNOTATION_TABLE])

    def select_gid_from_aid(back, aid, imgsetid=None, show=True, web=False):
        gid = back.ibs.get_annot_gids(aid)
        back.select_gid(gid, imgsetid=imgsetid, show=show, web=web, sel_aids=[aid])

    def select_aid(
        back, aid, imgsetid=None, show=True, show_annotation=True, web=False, **kwargs
    ):
        """ Table Click -> Chip Table """
        print('[back] select aid=%r, imgsetid=%r' % (aid, imgsetid))
        gid = back.ibs.get_annot_gids(aid)
        nid = back.ibs.get_annot_name_rowids(aid)
        back._set_selection(
            sel_aids=aid, sel_gids=gid, sel_nids=nid, sel_imgsetids=imgsetid, **kwargs
        )
        if show and show_annotation:
            back.show_annotation(aid, web=web, **kwargs)

    @backblock
    def select_nid(back, nid, imgsetid=None, show=True, show_name=True, **kwargs):
        """ Table Click -> Name Table """
        nid = cast_from_qt(nid)
        print('[back] select nid=%r, imgsetid=%r' % (nid, imgsetid))
        back._set_selection(sel_nids=nid, sel_imgsetids=imgsetid, **kwargs)
        if show and show_name:
            back.show_name(nid, **kwargs)

    # --------------------------------------------------------------------------
    # Action menu slots
    # --------------------------------------------------------------------------

    @blocking_slot()
    def add_annotation_from_image(back, gid_list, refresh=True):
        """ Context -> Add Annotation from Image"""
        print('[back] add_annotation_from_image')
        assert isinstance(gid_list, list), 'must pass in list here'
        size_list = back.ibs.get_image_sizes(gid_list)
        bbox_list = [(0, 0, w, h) for (w, h) in size_list]
        theta_list = [0.0] * len(gid_list)
        aid_list = back.ibs.add_annots(gid_list, bbox_list, theta_list)
        if refresh:
            back.front.update_tables([gh.IMAGE_TABLE, gh.ANNOTATION_TABLE])
        return aid_list

    @blocking_slot()
    def delete_image_annotations(back, gid_list):
        aid_list = ut.flatten(back.ibs.get_image_aids(gid_list))
        back.delete_annot(aid_list)

    @blocking_slot()
    def delete_annot(back, aid_list=None):
        """ Action -> Delete Annotation

        CommandLine:
            python -m wbia.gui.guiback --test-delete_annot --show
            python -m wbia.gui.guiback --test-delete_annot --show --no-api-cache
            python -m wbia.gui.guiback --test-delete_annot --show --assert-api-cache
            python -m wbia.gui.guiback --test-delete_annot --show --debug-api-cache --yes

        SeeAlso:
            manual_annot_funcs.delete_annots

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.guiback import *  # NOQA
            >>> back = testdata_guiback()
            >>> ibs = back.ibs
            >>> imgsetid_list = back.ibs.get_valid_imgsetids()
            >>> imgsetid = ut.take(imgsetid_list, ut.list_argmax(list(map(len, back.ibs.get_imageset_gids(imgsetid_list)))))
            >>> back.front.select_imageset_tab(imgsetid)
            >>> gid = back.ibs.get_imageset_gids(imgsetid)[0]
            >>> # add a test annotation to delete
            >>> aid_list = back.add_annotation_from_image([gid])
            >>> # delte annotations
            >>> aids1 = back.ibs.get_image_aids(gid)
            >>> back.delete_annot(aid_list)
            >>> aids2 = back.ibs.get_image_aids(gid)
            >>> #assert len(aids2) == len(aids1) - 1
            >>> ut.quit_if_noshow()
            >>> gt.qtapp_loop(back.mainwin, frequency=100)
        """
        print('[back] delete_annot, aid_list = %r' % (aid_list,))
        if aid_list is None:
            aid_list = back.get_selected_aids()
        if not back.are_you_sure(use_msg='Delete %d annotations?' % (len(aid_list))):
            return
        back._set_selection3(const.ANNOTATION_TABLE, aid_list, mode='diff')
        # get the image-id of the annotation we are deleting
        # gid_list = back.ibs.get_annot_gids(aid_list)
        # delete the annotation
        back.ibs.delete_annots(aid_list)
        # Select only one image
        # try:
        #    if len(gid_list) > 0:
        #        gid = gid_list[0]
        # except AttributeError:
        #    gid = gid_list
        # back.select_gid(gid, show=False)
        # update display, to show image without the deleted annotation
        back.front.update_tables()

    @blocking_slot()
    def unset_names(back, aid_list):
        msg = '[back] unsetting %d names' % (len(aid_list))
        print(msg)
        if not back.are_you_sure(msg):
            return
        back.ibs.set_annot_names(aid_list, [const.UNKNOWN] * len(aid_list))
        back.front.update_tables()

    @blocking_slot()
    def toggle_thumbnails(back):
        ibswgt = back.front
        tabwgt = ibswgt._tables_tab_widget
        index = tabwgt.currentIndex()
        tblname = ibswgt.tblname_list[index]
        view = ibswgt.views[tblname]
        col_name_list = view.col_name_list
        if 'thumb' in col_name_list:
            idx = col_name_list.index('thumb')
            view.col_hidden_list[idx] = not view.col_hidden_list[idx]
            view.hide_cols()
            # view.resizeRowsToContents() Too slow to use
        back.front.update_tables()

    # @blocking_slot(int)
    # @QtCore.pyqtSlot(int)
    @QtCore.pyqtSlot()
    @QtCore.pyqtSlot(int)
    def delete_image(back, gid_list=None):
        """ Action -> Delete Images"""
        print('[back] delete_image, gid_list = %r' % (gid_list,))
        if gid_list is None or gid_list is False:
            gid_list = [back.get_selected_gid()]
        gid_list = ut.ensure_iterable(gid_list)
        if not back.are_you_sure(action='delete %d images!' % (len(gid_list))):
            return
        # FIXME: The api cache seems to break here
        back.ibs.delete_images(gid_list)
        back.ibs.reset_table_cache()
        back.front.update_tables()

    @blocking_slot()
    def delete_all_imagesets(back):
        print('\n\n[back] delete all imagesets')
        if not back.are_you_sure(action='delete ALL imagesets'):
            return
        back.ibs.delete_all_imagesets()
        back.update_special_imagesets_()
        back.front.update_tables()

    @blocking_slot()
    def update_special_imagesets_(back):
        use_more_special_imagesets = back.ibs.cfg.other_cfg.ensure_attr(
            'use_more_special_imagesets', False
        )
        back.ibs.update_special_imagesets(use_more_special_imagesets)
        back.front.update_tables([gh.IMAGESET_TABLE])

    @blocking_slot(int)
    def delete_imageset_and_images(back, imgsetid_list):
        print('\n\n[back] delete_imageset_and_images')
        if back.contains_special_imagesets(imgsetid_list):
            back.display_special_imagesets_error()
            return
        if not back.are_you_sure(action='delete this imageset AND ITS IMAGES!'):
            return
        gid_list = ut.flatten(back.ibs.get_imageset_gids(imgsetid_list))
        back.ibs.delete_images(gid_list)
        back.ibs.delete_imagesets(imgsetid_list)
        back.update_special_imagesets_()
        back.front.update_tables()

    @blocking_slot(int)
    def mark_imageset_as_shipped(back, imgsetid_list):
        print('\n\n[back] mark_imageset_as_shipped')
        if back.contains_special_imagesets(imgsetid_list):
            back.display_special_imagesets_error()
            return
        if not back.are_you_sure(action='mark this imageset and shipped to Wildbook'):
            return

        back.ibs.set_imageset_shipped_flags(imgsetid_list, [1] * len(imgsetid_list))
        back.ibs.update_special_imagesets()
        back.front.update_tables()

    @blocking_slot(int)
    def delete_imageset(back, imgsetid_list):
        print('\n\n[back] delete_imageset')
        if back.contains_special_imagesets(imgsetid_list):
            back.display_special_imagesets_error()
            return
        if not back.are_you_sure(action='delete %d imagesets' % (len(imgsetid_list))):
            return
        back.ibs.delete_imagesets(imgsetid_list)
        back.update_special_imagesets_()
        back.front.update_tables()

    @blocking_slot(int)
    def export_imagesets(back, imgsetid_list):
        print('\n\n[back] export imageset')

        # new_dbname = back.user_input(
        #    msg='What do you want to name the new database?',
        #    title='Export to New Database')
        # if new_dbname is None or len(new_dbname) == 0:
        #    print('Abort export to new database. new_dbname=%r' % new_dbname)
        #    return
        back.ibs.export_imagesets(imgsetid_list, new_dbdir=None)

    @blocking_slot()
    def train_rf_with_imageset(back, **kwargs):
        from wbia.algo.detect import randomforest

        imgsetid = back._eidfromkw(kwargs)
        if imgsetid < 0:
            gid_list = back.ibs.get_valid_gids()
        else:
            gid_list = back.ibs.get_valid_gids(imgsetid=imgsetid)
        species = back.ibs.cfg.detect_cfg.species_text
        if species == 'none':
            species = None
        print(
            '[train_rf_with_imageset] Training Random Forest trees with imgsetid=%r and species=%r'
            % (imgsetid, species,)
        )
        randomforest.train_gid_list(back.ibs, gid_list, teardown=False, species=species)

    @blocking_slot(int)
    def merge_imagesets(back, imgsetid_list, destination_imgsetid):
        assert len(imgsetid_list) > 1, 'Cannot merge fewer than two imagesets'
        print('[back] merge_imagesets: %r, %r' % (destination_imgsetid, imgsetid_list))
        if back.contains_special_imagesets(imgsetid_list):
            back.display_special_imagesets_error()
            return
        ibs = back.ibs
        try:
            destination_index = imgsetid_list.index(destination_imgsetid)
        except Exception:
            # Default to the first value selected if the imgsetid doesn't exist in imgsetid_list
            print(
                '[back] merge_imagesets cannot find index for %r'
                % (destination_imgsetid,)
            )
            destination_index = 0
            destination_imgsetid = imgsetid_list[destination_index]
        deprecated_imgsetids = list(imgsetid_list)
        deprecated_imgsetids.pop(destination_index)
        gid_list = ut.flatten(
            [ibs.get_valid_gids(imgsetid=imgsetid) for imgsetid in imgsetid_list]
        )
        imgsetid_list = [destination_imgsetid] * len(gid_list)
        ibs.set_image_imgsetids(gid_list, imgsetid_list)
        ibs.delete_imagesets(deprecated_imgsetids)
        for imgsetid in deprecated_imgsetids:
            back.front.imageset_tabwgt._close_tab_with_imgsetid(imgsetid)
        back.front.update_tables([gh.IMAGESET_TABLE], clear_view_selection=True)

    @blocking_slot(int)
    def copy_imageset(back, imgsetid_list):
        print('[back] copy_imageset: %r' % (imgsetid_list,))
        if back.contains_special_imagesets(imgsetid_list):
            back.display_special_imagesets_error()
            return
        ibs = back.ibs
        new_imgsetid_list = ibs.copy_imagesets(imgsetid_list)
        print('[back] new_imgsetid_list: %r' % (new_imgsetid_list,))
        back.front.update_tables([gh.IMAGESET_TABLE], clear_view_selection=True)

    @blocking_slot(list)
    def remove_from_imageset(back, gid_list):
        imgsetid = back.get_selected_imgsetid()
        back.ibs.unrelate_images_and_imagesets(gid_list, [imgsetid] * len(gid_list))
        back.update_special_imagesets_()
        back.front.update_tables(
            [gh.IMAGE_TABLE, gh.IMAGESET_TABLE], clear_view_selection=True
        )

    @blocking_slot(list)
    def send_to_new_imageset(back, gid_list, mode='move'):
        assert len(gid_list) > 0, 'Cannot create a new imageset with no images'
        print('\n\n[back] send_to_new_imageset')
        ibs = back.ibs
        # imagesettext = const.NEW_IMAGESET_IMAGESETTEXT
        # imagesettext_list = [imagesettext] * len(gid_list)
        # ibs.set_image_imagesettext(gid_list, imagesettext_list)
        new_imgsetid = ibs.create_new_imageset_from_images(gid_list)  # NOQA
        if mode == 'move':
            imgsetid = back.get_selected_imgsetid()
            imgsetid_list = [imgsetid] * len(gid_list)
            ibs.unrelate_images_and_imagesets(gid_list, imgsetid_list)
        elif mode == 'copy':
            pass
        else:
            raise AssertionError('invalid mode=%r' % (mode,))
        back.update_special_imagesets_()
        back.front.update_tables(
            [gh.IMAGE_TABLE, gh.IMAGESET_TABLE], clear_view_selection=True
        )

    # --------------------------------------------------------------------------
    # Batch menu slots
    # --------------------------------------------------------------------------

    @blocking_slot()
    def imageset_set_species(back, refresh=True):
        """
        HACK: sets the species columns of all annotations in the imageset
        to be whatever is currently in the detect config
        """
        print('[back] imageset_set_species')
        ibs = back.ibs
        imgsetid = back.get_selected_imgsetid()
        aid_list = back.ibs.get_valid_aids(imgsetid=imgsetid)
        species_list = [ibs.cfg.detect_cfg.species_text] * len(aid_list)
        ibs.set_annot_species(aid_list, species_list)
        if refresh:
            back.front.update_tables([gh.ANNOTATION_TABLE])

    @blocking_slot()
    def change_detection_species(back, index, species_text):
        """ callback for combo box """
        print('[back] change_detection_species(%r, %r)' % (index, species_text))
        ibs = back.ibs
        # Load full blown configs for each species
        if back.edit_prefs_wgt:
            back.edit_prefs_wgt.close()
        if species_text == 'none':
            cfgname = const.UNKNOWN  # 'cfg'
        else:
            cfgname = species_text
        #
        current_species = None if species_text == 'none' else species_text
        #####
        # <GENERAL CONFIG SAVE>
        config_fpath = ut.unixjoin(ibs.get_dbdir(), 'general_config.cPkl')
        try:
            general_config = ut.load_cPkl(config_fpath)
        except IOError:
            general_config = {}
        general_config['current_species'] = current_species
        ut.save_cPkl(ut.unixjoin(ibs.get_dbdir(), 'general_config.cPkl'), general_config)
        # </GENERAL CONFIG SAVE>
        #####
        ibs._load_named_config(cfgname)
        ibs.cfg.detect_cfg.species_text = species_text
        ibs.cfg.save()

        # TODO: incorporate this as a signal in guiback which connects to a
        # slot in guifront
        # back.front.detect_button.setEnabled(
        #     ibs.has_species_detector(species_text)
        # )

    def get_selected_species(back):
        species_text = back.ibs.cfg.detect_cfg.species_text
        if species_text == 'none':
            species_text = None
        print('species_text = %r' % (species_text,))

        if species_text is None or species_text == const.UNKNOWN:
            # hack to set species for user
            pass
            # species_text = back.ibs.get_primary_database_species()
            # print('\'species_text = %r' % (species_text,))
            # sig = signal_(str)
            # sig.connect(back.ibswgt.species_combo.setItemText)
            # back.ibswgt.species_combo.setItemText(species_text)
            # back.changeSpeciesSignal.emit(str(species_text))
            # sig.emit(species_text)
            # back.ibs.cfg.detect_cfg.species_text = species_text
        return species_text

    @blocking_slot()
    def change_daids_mode(back, index, value):
        print('[back] change_daids_mode(%r, %r)' % (index, value))
        back.daids_mode = value
        # ibs = back.ibs
        # ibs.cfg.detect_cfg.species_text = value
        # ibs.cfg.save()

    @blocking_slot()
    def do_group_occurrence_step(back, refresh=True):
        """
        Group Step for computing occurrneces

        CommandLine:
            python -m wbia.gui.guiback --test-MainWindowBackend.do_group_occurrence_step --show --no-cnn

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.guiback import *  # NOQA
            >>> import wbia
            >>> main_locals = wbia.main(defaultdb='testdb1')
            >>> ibs, back = ut.dict_take(main_locals, ['ibs', 'back'])
            >>> ut.exec_funckw(back.do_group_occurrence_step, globals())
            >>> back.do_group_occurrence_step()
            >>> ut.quit_if_noshow()
        """
        print('[back] do_group_occurrence_step')
        from wbia import dtool

        # class TmpConfig(dtool.Config):
        #    _param_info_list = back.ibs.cfg.occur_cfg.get_param_info_list()
        ibs = back.ibs

        ungrouped_gid_list = ibs.get_ungrouped_gids()
        # ungrouped_images = ibs.get_image_instancelist(ungrouped_gid_list)
        # image_list['unixtime']

        existing_imgset_id_list = ibs.get_valid_imgsetids(
            is_occurrence=True, shipped=False, min_num_gids=1
        )

        class TmpConfig(dtool.Config):
            _param_info_list = [
                ut.ParamInfo('seconds_thresh', 1600, 'sec'),
                ut.ParamInfo('use_gps', True, ''),
            ]

        config = TmpConfig(**back.ibs.cfg.occur_cfg.to_dict())

        options = [
            'Create new occurrences',
            'Add to existing',
            'Regroup all',
        ]
        reply, new_config = back.user_option(
            title='Occurrence Grouping',
            msg=ut.codeblock(
                """
                Choose how we should group the %d ungrouped images into occurrences.
                We can either:
                    (1) append new occurrences
                    (2) add to the %d existing occurrences
                    (3) redo everything
                """
            )
            % (len(ungrouped_gid_list), len(existing_imgset_id_list)),
            config=config,
            options=options,
            default=options[0],
        )
        print('reply = %r' % (reply,))

        if reply not in options:
            raise guiexcept.UserCancel

        if reply != options[2] and len(ungrouped_gid_list) == 0:
            back.user_warning(msg='There are no ungrouped images.')
            raise guiexcept.UserCancel

        if reply == options[0]:
            back.ibs.compute_occurrences(config=new_config)
        elif reply == options[1]:
            # Add to existing imaesets
            imagesettext_list = ibs.get_imageset_text(existing_imgset_id_list)
            import numpy as np

            # Get unixtimes of the new images
            unixtime_list = ibs.get_image_unixtime(ungrouped_gid_list)
            # Get unixtimes of the occurrences
            imgset_gids_list = ibs.get_imageset_gids(existing_imgset_id_list)
            imgset_unixtimes_list = ibs.unflat_map(
                ibs.get_image_unixtime, imgset_gids_list
            )
            # imageset_start_time_posix_list = np.array(ut.lmap(np.min, unixtimes_list))
            imageset_mean_time_posix_list = np.array(
                ut.lmap(np.mean, imgset_unixtimes_list)
            )
            # imageset_end_time_posix_list = np.array(ut.lmap(np.max, unixtimes_list))

            assigned_idx = [
                np.abs(x - imageset_mean_time_posix_list).argmin() for x in unixtime_list
            ]
            assigned_imgset = ut.take(imagesettext_list, assigned_idx)
            ibs.set_image_imagesettext(ungrouped_gid_list, assigned_imgset)
            # HACK TO UPDATE IMAGESET POSIX TIMES
            # CAREFUL THIS BLOWS AWAY SMART DATA
            ibs.update_imageset_info(ibs.get_valid_imgsetids())
        elif reply == options[2]:
            if back.are_you_sure(use_msg='Regrouping will destroy all existing groups'):
                back.ibs.delete_all_imagesets()
                back.ibs.compute_occurrences(config=new_config)
            else:
                raise guiexcept.UserCancel

        back.update_special_imagesets_()
        print('[back] about to finish computing imagesets')
        back.front.imageset_tabwgt._close_all_tabs()
        if refresh:
            back.front.update_tables()
        print('[back] finished computing imagesets')

    @blocking_slot()
    def run_detection_step(back, refresh=True, **kwargs):
        r"""
        Args:
            refresh (bool): (default = True)

        CommandLine:
            python -m wbia.gui.guiback run_detection_step --show

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.guiback import *  # NOQA
            >>> import wbia
            >>> main_locals = wbia.main(defaultdb='testdb1')
            >>> ibs, back = ut.dict_take(main_locals, ['ibs', 'back'])
            >>> ut.exec_funckw(back.run_detection_step, globals())
            >>> back.run_detection_step()
            >>> back.cleanup()
            >>> ut.quit_if_noshow()
        """
        print('\n\n')
        imgsetid = back._eidfromkw(kwargs)
        ibs = back.ibs

        # Get images without any annotations
        set_gids = ibs.get_valid_gids(imgsetid=imgsetid)
        is_empty = [n == 0 for n in ibs.get_image_num_annotations(set_gids)]
        gid_list = ut.compress(set_gids, is_empty)

        print('[back] run_detection_step(imgsetid=%r)' % (imgsetid))
        print('[back] detected %d/%d empty gids' % (len(gid_list), len(set_gids)))

        imgset_text = back.ibs.get_imageset_text(imgsetid)

        detector = back.ibs.cfg.detect_cfg.detector

        review_in_web = True
        comp_name = ut.get_computer_name()
        db_name = ibs.dbname
        is_lewa = comp_name in ['wbia.cs.uic.edu'] or db_name in ['LEWA', 'lewa_grevys']
        if is_lewa:
            review_in_web_mode = 'annotations'
        else:
            review_in_web_mode = 'detections'
        # review_in_web_mode = 'annotations'
        assert review_in_web_mode in ['detections', 'annotations']
        if True:
            # TODO better confirm dialog
            from wbia import dtool

            species_text = ibs.get_all_species_texts()
            species_nice = ibs.get_all_species_nice()

            class TmpDetectConfig(dtool.Config):
                _param_info_list = [
                    ut.ParamInfo('review_in_web', review_in_web),
                    ut.ParamInfo(
                        'detector',
                        ibs.cfg.detect_cfg.detector,
                        valid_values=['cnn', 'rf'],
                    ),
                    ut.ParamInfo(
                        'species',
                        ibs.cfg.detect_cfg.species_text,
                        valid_values=species_nice,
                        hideif=lambda cfg: cfg['detector'] != 'rf',
                    ),
                ]

            config = TmpDetectConfig(**back.ibs.cfg.detect_cfg.to_dict())
            options = [
                'Start Detection',
            ]
            reply, new_config = back.user_option(
                title='Run Detection Confirmation',
                msg=ut.codeblock(
                    """
                    Preparing to run detection on %d images in \'%s\'.
                    """
                )
                % (len(gid_list), imgset_text),
                config=config,
                options=options,
                default=options[0],
            )
            print('reply = %r' % (reply,))

            if reply not in options:
                raise guiexcept.UserCancel

            review_in_web = new_config['review_in_web']
            detector = new_config['detector']
            species = new_config['species']
            nice2_text = dict(zip(species_nice, species_text))
            species = nice2_text.get(species, species)

            if detector in ['cnn', 'cnn_yolo', 'yolo']:
                ibs.detect_cnn_yolo(gid_list)
                back.front.update_tables([gh.IMAGE_TABLE, gh.ANNOTATION_TABLE])
            elif detector in ['random_forest', 'rf']:
                ibs.detect_random_forest(gid_list, species)
            else:
                raise ValueError('Detector=%r not recognized' % (detector,))
            print('[back] about to finish detection')
            if refresh:
                back.front.update_tables([gh.IMAGE_TABLE, gh.ANNOTATION_TABLE])

        else:
            # OLD CODE
            if detector in ['cnn', 'cnn_yolo', 'yolo']:
                # Construct message
                msg_fmtstr_list = ['You are about to run detection using CNN YOLO...']
                fmtdict = dict()
                # Append detection configuration information
                msg_fmtstr_list += ['    Images:   {num_gids}']  # Add more spaces
                # msg_fmtstr_list += ['* # database annotations={num_daids}.']
                # msg_fmtstr_list += ['* database species={d_species_phrase}.']
                fmtdict['num_gids'] = len(gid_list)
                # Finish building confirmation message
                msg_fmtstr_list += ['']
                msg_fmtstr_list += ["Press 'Yes' to continue"]
                msg_fmtstr = '\n'.join(msg_fmtstr_list)
                msg_str = msg_fmtstr.format(**fmtdict)
                if back.are_you_sure(use_msg=msg_str):
                    print('[back] run_detection_step(imgsetid=%r)' % (imgsetid))
                    ibs.detect_cnn_yolo(gid_list)
                    print('[back] about to finish detection')
                    if refresh:
                        back.front.update_tables([gh.IMAGE_TABLE, gh.ANNOTATION_TABLE])
                    print('[back] finished detection')
            elif detector in ['random_forest', 'rf']:
                species = ibs.cfg.detect_cfg.species_text
                # Construct message
                msg_fmtstr_list = [
                    'You are about to run detection using Random Forests...'
                ]
                fmtdict = dict()
                # Append detection configuration information
                msg_fmtstr_list += ['    Images:   {num_gids}']  # Add more spaces
                msg_fmtstr_list += ['    Species: {species_phrase}']
                # msg_fmtstr_list += ['* # database annotations={num_daids}.']
                # msg_fmtstr_list += ['* database species={d_species_phrase}.']
                fmtdict['num_gids'] = len(gid_list)
                fmtdict['species_phrase'] = species
                # Finish building confirmation message
                msg_fmtstr_list += ['']
                msg_fmtstr_list += ["Press 'Yes' to continue"]
                msg_fmtstr = '\n'.join(msg_fmtstr_list)
                msg_str = msg_fmtstr.format(**fmtdict)
                if back.are_you_sure(use_msg=msg_str):
                    print(
                        '[back] run_detection_step(species=%r, imgsetid=%r)'
                        % (species, imgsetid)
                    )
                    ibs.detect_random_forest(gid_list, species)
                    print('[back] about to finish detection')
                    if refresh:
                        back.front.update_tables([gh.IMAGE_TABLE, gh.ANNOTATION_TABLE])
                    print('[back] finished detection')
            else:
                raise ValueError('Detector not recognized')

        if review_in_web:
            # back.user_info(msg='Detection has finished. Launching web review')
            web_domain = back.get_background_web_domain()
            if review_in_web_mode == 'annotations':
                url = 'http://%s/turk/annotation/?imgsetid=%s' % (web_domain, imgsetid,)
            elif review_in_web_mode == 'detections':
                url = 'http://%s/turk/detection/?imgsetid=%s' % (web_domain, imgsetid,)
            else:
                raise ValueError('invalid value for review_in_web_mode')
            print('[guiback] Opening... %r' % (url,))
            import webbrowser

            back.start_web_server_parallel(browser=False)
            webbrowser.open(url)
        else:
            back.user_info(msg='Detection has finished.')

    @blocking_slot()
    def show_advanced_id_interface(back):
        """
        CommandLine:
            python -m wbia.gui.guiback show_advanced_id_interface --show

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.guiback import *  # NOQA
            >>> import wbia
            >>> main_locals = wbia.main(defaultdb='testdb1')
            >>> ibs, back = ut.dict_take(main_locals, ['ibs', 'back'])
            >>> ut.exec_funckw(back.show_advanced_id_interface, globals())
            >>> back.show_advanced_id_interface()
            >>> back.cleanup()
            >>> ut.quit_if_noshow()
            >>> #gt.ensure_qapp()  # must be ensured before any embeding
            >>> import wbia.plottool as pt
            >>> gt.qtapp_loop(qwin=back)
        """
        # from wbia.init import filter_annots
        # filter_kw = filter_annots.get_default_annot_filter_form()

        back.custom_query_widget = CustomAnnotCfgSelector(back.ibs)
        back.custom_query_widget.show()
        app = gt.get_qtapp()
        app.processEvents()
        app.processEvents()
        back.custom_query_widget.onstart()
        # back.custom_query_widget.apply_new_config()
        # dlg = wgt.as_dialog(back)
        # dlg.show()
        # dlg.exec_()
        # if not wgt.accept_flag:
        #    #reply not in options:
        #    raise guiexcept.UserCancel
        # back.compute_queries(qaid_list=wgt.qaids, daid_list=back.daids)

    def confirm_query_dialog2(
        back,
        species2_expanded_aids=None,
        cfgdict=None,
        query_msg=None,
        query_title=None,
        review_cfg={},
    ):
        """
        Asks the user to confirm starting the identification query
        """
        msg_str, detailed_msg = back.make_confirm_query_msg2(
            species2_expanded_aids, cfgdict=cfgdict, query_title=query_title
        )
        if query_title is None:
            query_title = 'custom'
        confirm_kw = dict(
            use_msg=msg_str,
            title='Begin %s ID' % (query_title,),
            default='Yes',
            detailed_msg=detailed_msg,
        )

        # TODO better confirm dialog
        if True:
            from wbia import dtool

            ibs = back.ibs  # NOQA
            # class TmpIDConfig(dtool.Config):
            #    _param_info_list = [
            #        #ut.ParamInfo('K', ibs.cfg.query_cfg.nn_cfg.K),
            #        #ut.ParamInfo('Knorm', ibs.cfg.query_cfg.nn_cfg.Knorm),
            #        #ut.ParamInfo('chip_size', ibs.cfg.chip_cfg.dim_size),
            #    ]
            # **back.ibs.cfg.to_dict())
            # config = TmpIDConfig()

            review_config = {
                'filter_reviewed': review_cfg.get(
                    'filter_reviewed',
                    ibs.cfg.other_cfg.ensure_attr('filter_reviewed', True),
                ),
                'ranks_top': review_cfg.get(
                    'ranks_top', ibs.cfg.other_cfg.ensure_attr('ranks_top', 2)
                ),
                'filter_true_matches': review_cfg.get('filter_true_matches', False),
            }

            if cfgdict is None:
                cfgdict = {}
            tmpdict = cfgdict.copy()
            tmpdict.update(review_config)

            config = dtool.Config.from_dict(tmpdict)

            # print('config = %r' % (config,))
            options = [
                'Start ID',
            ]

            if ut.get_argflag(('--yes', '-y')):
                reply = options[0]
                new_config = config
            else:
                reply, new_config = back.user_option(
                    title='Begin %s ID' % (query_title,),
                    msg=msg_str,
                    config=config,
                    options=options,
                    default=options[0],
                    detailed_msg=detailed_msg,
                )
                print('reply = %r' % (reply,))
            updated_config = new_config.asdict()
            updated_review_cfg = ut.dict_subset(updated_config, review_config.keys())
            ut.delete_dict_keys(updated_config, review_config.keys())

            if reply not in options:
                raise guiexcept.UserCancel
            return updated_config, updated_review_cfg
        else:
            if not back.are_you_sure(**confirm_kw):
                raise guiexcept.UserCancel

    @blocking_slot()
    def compute_queries(
        back,
        refresh=True,
        daids_mode=None,
        query_is_known=None,
        qaid_list=None,
        use_prioritized_name_subset=False,
        use_visual_selection=False,
        cfgdict={},
        query_msg=None,
        custom_qaid_list_title=None,
        daid_list=None,
        partition_queries_by_species=False,
        **kwargs
    ):
        """
        MAIN QUERY FUNCTION

        execute_query

        Batch -> Compute OldStyle Queries
        and Actions -> Query

        Computes query results for all annotations in an imageset.
        Results are either vs-exemplar or intra-imageset

        CommandLine:
            ./reset_dbs.py && ./main.py --query 1 -y
            ./reset_dbs.py --reset-mtest && ./main.py --query 1 -y --db PZ_MTEST --progtext
            ./main.py --query 1 -y
            python -m wbia --query 1 -y
            python -m wbia --query 1:119 --db PZ_MTEST --nocache-query --nocache-nnmid -y
            python -m wbia --query 1:119 --db PZ_MTEST --nocache-query --nocache-nnmid -y --force-all-progress
            python -m wbia --query 1:119 --db PZ_MTEST --nocache-query --nocache-nnmid --hots-batch-size=3 -y
            python -m wbia --query 1:119 --db PZ_MTEST --nocache-query --nocache-nnmid --hots-batch-size=3 -y
            python -m wbia --query 1:119 --db PZ_MTEST --nocache-query --nocache-nnmid --hots-batch-size=32 -y

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.gui.guiback import *  # NOQA
            >>> import wbia
            >>> main_locals = wbia.main(db='testdb2')
            >>> back = main_locals['back']
            >>> ibs = back.ibs
            >>> query_is_known = None
            >>> refresh = True
            >>> daids_mode = None
            >>> imgsetid = None
            >>> kwargs = {}
            >>> print(result)
        """
        imgsetid = back._eidfromkw(kwargs)
        print('\n')
        print('------')
        print(
            '[back] compute_queries: imgsetid=%r, mode=%r' % (imgsetid, back.daids_mode)
        )
        print('[back] use_prioritized_name_subset = %r' % (use_prioritized_name_subset,))
        print('[back] use_visual_selection        = %r' % (use_visual_selection,))
        print('[back] daids_mode                  = %r' % (daids_mode,))
        print('[back] cfgdict                     = %r' % (cfgdict,))
        print('[back] query_is_known              = %r' % (query_is_known,))

        if qaid_list is not None:
            if custom_qaid_list_title is None:
                custom_qaid_list_title = 'Custom'
            qaid_title = custom_qaid_list_title
        elif use_visual_selection:
            qaid_title = 'Selected'
        else:
            # if not visual selection, then qaids are selected by imageset
            qaid_title = back.ibs.get_imageset_text(imgsetid)
        if use_prioritized_name_subset:
            qaid_title += '(priority_subset)'

        # Set title variables
        daid_title = None
        if daids_mode == const.VS_EXEMPLARS_KEY:
            daid_title = 'Exemplars'
        elif daids_mode == const.INTRA_OCCUR_KEY:
            daid_title = 'Intra ImageSet'
        elif daids_mode == 'All':
            daid_title = 'Everything'
        else:
            print('Unknown daids_mode=%r' % (daids_mode,))

        query_title = '%s-vs-%s' % (qaid_title, daid_title)

        print('query_title = %r' % (query_title,))
        if daids_mode == const.VS_EXEMPLARS_KEY:
            # Automatic setting of exemplars
            back.set_exemplars_from_quality_and_viewpoint_()

        species2_expanded_aids = back._get_expanded_aids_groups(
            imgsetid,
            daids_mode=daids_mode,
            use_prioritized_name_subset=use_prioritized_name_subset,
            use_visual_selection=use_visual_selection,
            qaid_list=qaid_list,
            daid_list=daid_list,
            query_is_known=query_is_known,
            partition_queries_by_species=partition_queries_by_species,
            remove_unknown_species=False,
        )

        if not partition_queries_by_species:
            # hack: overwrite species2_expanded_aids to undo species partition
            all_qaids = set()
            all_daids = set()
            for _qaids, _daids in species2_expanded_aids.values():
                all_qaids.update(_qaids)
                all_daids.update(_daids)
            species2_expanded_aids = {
                'all_species': (sorted(all_qaids), sorted(all_daids))
            }

        if len(species2_expanded_aids) == 0:
            raise guiexcept.InvalidRequest(
                'Is the database empty? Are species labels assigned correctly? '
                'There are no pairs of query and database annotations with the same species'
            )

        review_cfg = kwargs.copy()
        review_cfg['filter_true_matches'] = daids_mode == const.VS_EXEMPLARS_KEY

        # Check if there is nothing to do
        flags = []
        for species, expanded_aids in species2_expanded_aids.items():
            qaids, daids = expanded_aids
            flag = len(qaids) == 1 and len(daids) == 1 and daids[0] == qaids[0]
            flags.append(flag)
        if len(flags) > 0 and all(flags):
            raise AssertionError('No need to compute a query against itself')

        cfgdict, review_cfg = back.confirm_query_dialog2(
            species2_expanded_aids,
            query_msg=query_msg,
            query_title=query_title,
            cfgdict=cfgdict,
            review_cfg=review_cfg,
        )
        print('cfgdict = %r' % (cfgdict,))

        prog_bar = back.front.prog_bar
        prog_bar.setVisible(True)
        prog_bar.setWindowTitle('Initialize query')
        prog_hook = prog_bar.utool_prog_hook
        # prog_bar = guitool.newProgressBar(None)  # back.front)
        # Doesn't seem to work correctly
        # prog_hook.show_indefinite_progress()
        prog_hook.force_event_update()
        # prog_hook.set_progress(0)
        prog_bar.setWindowTitle('Start query')
        # import utool
        # utool.embed()

        query_results = {}
        for key, (qaids, daids) in ut.ProgressIter(
            species2_expanded_aids.items(), prog_hook=prog_hook
        ):
            prog_bar.setWindowTitle('Initialize %r query' % (key,))
            qreq_ = back.ibs.new_query_request(qaids, daids, cfgdict=cfgdict)
            prog_hook.initialize_subhooks(1)
            subhook = prog_hook.next_subhook()
            cm_list = qreq_.execute(prog_hook=subhook)
            query_results[key] = (cm_list, qreq_)

            # HACK IN IMAGESET INFO
            if daids_mode == const.INTRA_OCCUR_KEY:
                for cm in cm_list:
                    # if cm is not None:
                    cm.imgsetid = imgsetid
        back.front.prog_bar.setVisible(False)

        print('[back] About to finish compute_queries: imgsetid=%r' % (imgsetid,))
        for key in query_results.keys():
            (cm_list, qreq_) = query_results[key]
            # Filter duplicate names if running vsexemplar
            back.review_queries(
                cm_list,
                qreq_=qreq_,
                query_title=query_title + ' ' + str(key),
                review_cfg=review_cfg,
            )
        if refresh:
            back.front.update_tables()
        print('[back] FINISHED compute_queries: imgsetid=%r' % (imgsetid,))

    def get_selected_daids(
        back, imgsetid=None, daids_mode=None, qaid_list=None, species=None
    ):
        daids_mode = back.daids_mode if daids_mode is None else daids_mode
        daids_mode_valid_kw_dict = {
            const.VS_EXEMPLARS_KEY: {'is_exemplar': True},
            const.INTRA_OCCUR_KEY: {'imgsetid': imgsetid},
            'all': {},
        }
        if qaid_list is not None and species is None:
            ibs = back.ibs
            hist_ = ut.dict_hist(ibs.get_annot_species_texts(qaid_list))
            print('[back] len(qaid_list)=%r' % (len(qaid_list)))
            print('[back] hist_ = %r' % (hist_,))
            if len(hist_) == 1:
                # select the query species if there is only one
                species = hist_.keys()[0]

        if species is None:
            species = back.get_selected_species()

        valid_kw = {
            'minqual': 'ok',
        }
        if species != const.UNKNOWN:
            # Query everything if you don't know the species
            valid_kw['species'] = species
        mode_str = {
            const.VS_EXEMPLARS_KEY: 'vs_exemplar',
            const.INTRA_OCCUR_KEY: 'intra_occurrence',
            'all': 'all',
        }[daids_mode]
        valid_kw.update(daids_mode_valid_kw_dict[daids_mode])
        print('[back] get_selected_daids: ' + mode_str)
        print('[back] ... valid_kw = ' + ut.repr2(valid_kw))
        daid_list = back.ibs.get_valid_aids(**valid_kw)
        return daid_list

    def make_confirm_query_msg2(
        back, species2_expanded_aids, cfgdict=None, query_msg=None, query_title=None
    ):
        r"""
        CommandLine:
            python -m wbia.gui.guiback --test-MainWindowBackend.make_confirm_query_msg2 --show

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.guiback import *  # NOQA
            >>> import wbia
            >>> main_locals = wbia.main(defaultdb='testdb1')
            >>> ibs, back = ut.dict_take(main_locals, ['ibs', 'back'])
            >>> ut.exec_funckw(back.make_confirm_query_msg2, globals())
            >>> imgsetid = ibs.get_imageset_imgsetids_from_text('*All Images')
            >>> species2_expanded_aids = back._get_expanded_aids_groups(imgsetid)
            >>> short_msg, detailed_msg = back.make_confirm_query_msg2(species2_expanded_aids)
            >>> print(short_msg)
            >>> print(detailed_msg)
            >>> ut.quit_if_noshow()
            >>> back.confirm_query_dialog2(species2_expanded_aids)
        """
        ibs = back.ibs
        species_text = ibs.get_all_species_texts()
        species_nice = ibs.get_all_species_nice()
        species_dict = dict(zip(species_text, species_nice))

        def get_unique_species_phrase(aid_list):
            def boldspecies(species):
                species_bold_nice = "'%s'" % (species_dict.get(species, species).upper(),)
                return species_bold_nice

            species_list = list(set(ibs.get_annot_species_texts(aid_list)))
            species_nice_list = list(map(boldspecies, species_list))
            species_phrase = ut.conj_phrase(species_nice_list, 'and')
            return species_phrase

        # Build confirmation message
        fmtdict = dict()
        msg_fmtstr_list = []

        if query_msg is not None:
            msg_fmtstr_list += [query_msg]
        if query_title is None:
            query_title = 'custom'

        ngroups = len(species2_expanded_aids)
        if ngroups > 1:
            msg_fmtstr_list += [
                (
                    'You are about to run {query_title} '
                    'identification with {ngroups} groups...'
                ).format(
                    query_title=query_title, ngroups=ngroups,
                )
            ]
        else:
            msg_fmtstr_list += [
                ('You are about to run {query_title} ' 'identification...').format(
                    query_title=query_title,
                )
            ]

        species_list = list(species2_expanded_aids.keys())

        detailed_msg_list = []
        annotstats_kw = {}
        for count, species in enumerate(species_list):
            qaids, daids = species2_expanded_aids[species]
            species_nice = species_dict.get(species, species)
            # species_phrase = get_unique_species_phrase(qaids + daids)
            msg_fmtstr_list += ['']
            fmtdict = {}
            qaid_stats = ibs.get_annot_stats_dict(
                qaids, prefix='q', per_name=True, old=False
            )
            daid_stats = ibs.get_annot_stats_dict(
                daids, prefix='d', per_name=True, old=False
            )
            stats_ = ibs.get_annotconfig_stats(
                qaids, daids, combined=False, species_hist=True, **annotstats_kw
            )
            fmtdict.update(**qaid_stats)
            fmtdict.update(**daid_stats)
            fmtdict['qannots'] = ut.pluralize('annotation', len(qaids))
            fmtdict['dannots'] = ut.pluralize('annotation', len(daids))
            fmtdict['species_nice'] = species_nice
            fmtdict['count'] = count
            # Add simple info
            if ngroups > 1:
                part1 = 'Group {count} '
            else:
                part1 = 'This '
            part2 = (
                part1
                + 'will identify {num_qaids} query {qannots} against {num_daids} {species_nice} database {dannots}.'
            ).format(**fmtdict)
            msg_fmtstr_list += [part2]
            # Add detailed info
            stats_str2 = ut.repr2(
                stats_, strvals=True, newlines=2, explicit=False, nobraces=False
            )
            detailed_msg_list.append('--- Group %d ---' % (count,))
            detailed_msg_list.append(stats_str2)

        # Finish building confirmation message
        msg_fmtstr_list += ['']
        msg_fmtstr_list += ["Press 'Yes' to continue"]
        msg_fmtstr = '\n'.join(msg_fmtstr_list)
        msg_str = msg_fmtstr.format(**fmtdict)

        if cfgdict is not None and len(cfgdict) > 0:
            detailed_msg_list = [
                'Special Settings: {}'.format(ut.repr2(cfgdict))
            ] + detailed_msg_list

        detailed_msg = '\n'.join(detailed_msg_list)

        return msg_str, detailed_msg

    def run_annot_splits(back, aid_list):
        """
        Checks for mismatches within a group of annotations

        Args:
            aid_list (int):  list of annotation ids

        CommandLine:
            python -m wbia.gui.guiback --test-MainWindowBackend.run_annot_splits --show

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.guiback import *  # NOQA
            >>> back = testdata_guiback()
            >>> ibs = back.ibs
            >>> aids_list, nids = back.ibs.group_annots_by_name(back.ibs.get_valid_aids())
            >>> aid_list = aids_list[ut.list_argmax(list(map(len, aids_list)))]
            >>> back.run_annot_splits(aid_list)
            >>> ut.quit_if_noshow()
            >>> guitool.qtapp_loop(back.mainwin, frequency=100)
        """
        cfgdict = {
            'can_match_samename': True,
            'K': 3,
            'Knorm': 3,
            'prescore_method': 'csum',
            'score_method': 'csum',
        }
        ranks_top = min(len(aid_list), 10)
        review_cfg = {
            'filter_reviewed': False,
            'ranks_top': ranks_top,
            'name_scoring': False,
        }
        ibs = back.ibs
        cfgdict, review_cfg = back.confirm_query_dialog2(
            {'split': (aid_list, aid_list)},
            cfgdict=cfgdict,
            query_msg='Checking for SPLIT cases (matching each annotation within a name)',
            review_cfg=review_cfg,
        )
        qreq_ = ibs.new_query_request(aid_list, aid_list, cfgdict=cfgdict)
        cm_list = qreq_.execute()
        back.review_queries(
            cm_list, qreq_=qreq_, query_title='Annot Splits', review_cfg=review_cfg
        )

        if False:
            from wbia.viz import viz_graph2
            import imp

            imp.reload(viz_graph2)
            win = viz_graph2.make_qt_graph_review(qreq_, cm_list, review_cfg=review_cfg)
            win.show()

    def run_merge_checks(back):
        r"""
        Checks for missed matches within a group of annotations

        CommandLine:
            python -m wbia.gui.guiback --test-run_merge_checks --show

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.guiback import *  # NOQA
            >>> back = testdata_guiback()
            >>> result = back.run_merge_checks()
            >>> print(result)
            >>> ut.quit_if_noshow()
            >>> guitool.qtapp_loop(back.mainwin, frequency=100)
        """
        pass
        qaid_list = back.ibs.get_valid_aids(is_exemplar=True)
        cfgdict = {
            'can_match_samename': False,
            # 'K': 3,
            # 'Knorm': 3,
            # 'prescore_method': 'csum',
            # 'score_method': 'csum'
        }
        query_msg = 'Checking for MERGE cases (this is an exemplars-vs-exemplars query)'
        back.compute_queries(
            qaid_list=qaid_list,
            daids_mode=const.VS_EXEMPLARS_KEY,
            query_msg=query_msg,
            cfgdict=cfgdict,
            custom_qaid_list_title='Merge Candidates',
        )

    def run_merge_checks_multitons(back):
        r"""
        Checks for missed matches within a group of annotations.
        Only uses annotations with more 2 annots per id.
        """
        pass
        ibs = back.ibs
        # qaid_list = back.ibs.get_valid_aids(is_exemplar=True)
        from wbia import dtool

        config = dtool.Config.from_dict(
            {
                'K': 1,
                'Knorm': 5,
                'min_pername': 1,
                'max_pername': 1,
                'exemplars_per_name': 1,
                'method': 'randomize',
                'seed': 42,
            }
        )
        # ibswgt = None
        dlg = gt.ConfigConfirmWidget.as_dialog(
            title='Confirm Merge Query', msg='Confirm', config=config
        )
        self = dlg.widget
        dlg.resize(700, 500)
        dlg.exec_()
        print('config = %r' % (config,))
        updated_config = self.config  # NOQA
        print('updated_config = %r' % (updated_config,))

        min_pername = updated_config['min_pername']
        max_pername = updated_config['max_pername']
        aid_list = ibs.filter_annots_general(
            min_pername=min_pername, max_pername=max_pername, minqual='ok'
        )
        if updated_config['method'] == 'randomize':
            import numpy as np

            rng = np.random.RandomState(int(updated_config['seed']))
            grouped_aids = ibs.group_annots_by_name(aid_list)[0]
            grouped_aids2 = [
                ut.random_sample(aids, updated_config['exemplars_per_name'], rng=rng)
                for aids in grouped_aids
            ]
            aid_list = ut.flatten(grouped_aids2)
        else:
            new_flag_list = ibs.get_annot_quality_viewpoint_subset(
                aid_list, updated_config['exemplars_per_name'], allow_unknown=True
            )
            aid_list = ut.compress(aid_list, new_flag_list)

        ibs.print_annot_stats(aid_list)

        daid_list = qaid_list = aid_list
        # len(aids)
        cfgdict = {
            'can_match_samename': False,
            'K': updated_config['K'],
            'Knorm': updated_config['Knorm'],
            # 'prescore_method': 'csum',
            # 'score_method': 'csum'
        }
        query_msg = 'Checking for MERGE cases (this is an special query)'
        back.compute_queries(
            qaid_list=qaid_list,
            daid_list=daid_list,
            query_msg=query_msg,
            cfgdict=cfgdict,
            custom_qaid_list_title='Merge2 Candidates',
        )

    def _get_expanded_aids_groups(
        back,
        imgsetid,
        daids_mode=None,
        use_prioritized_name_subset=False,
        use_visual_selection=False,
        qaid_list=None,
        daid_list=None,
        query_is_known=None,
        partition_queries_by_species=True,
        remove_unknown_species=None,
    ):
        """
        Get the query annotation ids to search and
        the database annotation ids to be searched
        The query is either a specific selection or everything from this
        image set that matches the appropriate filters.

        Example:
            >>> # DISABLE_DOCTEST
            >>> ut.exec_funckw(back._get_expanded_aids_groups, globals())
            >>> imgsetid = ibs.get_imageset_imgsetids_from_text('*All Images')
            >>> species2_expanded_aids = back._get_expanded_aids_groups(imgsetid)
        """
        ibs = back.ibs
        daids_mode = back.daids_mode if daids_mode is None else daids_mode
        if imgsetid is None:
            raise Exception('[back] invalid imgsetid')

        if ibs.cfg.other_cfg.enable_custom_filter:
            back.user_warning(
                msg=ut.codeblock(
                    """
                other_cfg.enable_custom_filter=True is not longer supported.
                Please turn off in Preferences
                """
                )
            )

        if remove_unknown_species is None:
            # Default behavior is don't remove unknown species if qaid_list is specified
            remove_unknown_species = qaid_list is not None or use_visual_selection

        # Query aids are either: given, taken from gui selection, or by imageset
        if qaid_list is not None:
            qaid_list = qaid_list
        elif use_visual_selection:
            qaid_list = back.get_selected_aids()
        else:
            qaid_list = ibs.get_valid_aids(
                imgsetid=imgsetid, is_known=query_is_known, minqual='ok'
            )

        print('[back] Initially loaded len(qaid_list) = %r' % (len(qaid_list),))
        if use_prioritized_name_subset:
            # Pick only a few queries per name to execute
            annots_per_view = 2  # FIXME: use a configuration
            new_flag_list = back.ibs.get_annot_quality_viewpoint_subset(
                aid_list=qaid_list,
                annots_per_view=annots_per_view,
                allow_unknown=True,
                verbose=True,
            )
            qaid_list = ut.compress(qaid_list, new_flag_list)
            print(
                '[back] Filtered query by quality and viewpoint: len(qaid_list) = %r'
                % (len(qaid_list),)
            )

        print('[back] Found len(qaid_list) = %r' % (len(qaid_list),))
        # Group annotations by species
        species2_qaids = ibs.group_annots_by_prop(qaid_list, ibs.get_annot_species)
        # ID unknown species against each database species
        nospecies_qaids = species2_qaids.pop(ibs.const.UNKNOWN, [])
        print('[back] num Queries without species = %r' % (len(nospecies_qaids),))
        print('species2_qaids = %r' % (species2_qaids,))

        species2_expanded_aids = {}
        species_list = ut.unique(
            list(ibs.get_all_species_texts()) + (list(species2_qaids.keys()))
        )
        for species in species_list:
            print('[back] Finding daids for species = %r' % (species,))
            qaids = species2_qaids[species]
            if daid_list is not None:
                daids = daid_list
            else:
                if not partition_queries_by_species:
                    species = const.UNKNOWN
                daids = back.get_selected_daids(
                    imgsetid=imgsetid,
                    daids_mode=daids_mode,
                    qaid_list=qaids,
                    species=species,
                )
            print('[back] * Found len(daids) = %r' % (len(daids),))
            qaids_ = ut.unique(qaids + nospecies_qaids)
            if len(qaids_) > 0 and len(daids) > 0:
                species2_expanded_aids[species] = (qaids_, daids)
            else:
                print('[back] ! len(nospecies_qaids) = %r' % (len(nospecies_qaids),))
                print('[back] ! len(qaids) = %r' % (len(qaids),))
                print('[back] ! len(qaids_) = %r' % (len(qaids_),))
                print('[back] ! len(daids) = %r' % (len(daids),))
                print('WARNING: species = %r is an invalid query' % (species,))

        # Dont query unknown species
        if remove_unknown_species:
            if ibs.const.UNKNOWN in species2_expanded_aids:
                del species2_expanded_aids[ibs.const.UNKNOWN]

        return species2_expanded_aids

    @blocking_slot()
    def filter_imageset_as_camera_trap(back, refresh=True, score_thresh=0.30):
        ibs = back.ibs
        imgsetid = back.get_selected_imgsetid()
        gid_list = ibs.get_imageset_gids(imgsetid)
        aid_list = ut.flatten(ibs.get_image_aids(gid_list))

        imgset_name = ibs.get_imageset_text(imgsetid)
        imgset_dest_name = '%s (FILTERED)' % (imgset_name,)

        confirm_kw = dict(
            use_msg=ut.codeblock(
                """
            This action will process the current ImageSet and search for the images (%d total) which contain zebras.

            Note: this process is meant for images from sources like camera traps and aerial drones, where the vast majority of the images have no zebras in them.  If you expect the majority of your images to contain zebras, run the standard Detect for better detection accuracy (but will be slower).

            This operation will find Plains and Grevy's zebras.

            Warning: performing this operation will delete all current annotations (%d total) on all images in the current ImageSet. This operation will also de-associate any metadata (e.g. names) from the annotations.  This operation will not delete or modify any images, annotations, or names outside this ImageSet.

            Selecting YES will remove this ImageSet's annotations, perform a filtering classification for zebras, run the detection pipeline, and create (or will empty if currently exists) an ImageSet named:

            \t %r

            This new ImageSet will contain the images where zebras were detected.
            """
            )
            % (len(gid_list), len(aid_list), imgset_dest_name,),
            title='Process ImageSet?',
            default='Yes',
        )
        if not back.are_you_sure(**confirm_kw):
            raise guiexcept.UserCancel

        depc = ibs.depc_image

        ibs.delete_annots(aid_list)
        # imgset_rowid_list = ibs.get_imageset_imgsetids_from_text(['POSITIVE_SET', 'NEGATIVE_SET'])
        imgset_rowid_list = ibs.get_imageset_imgsetids_from_text([imgset_dest_name])
        for imgset_rowid in imgset_rowid_list:
            imgset_rowid_list_ = [imgset_rowid] * len(gid_list)
            ibs.unrelate_images_and_imagesets(gid_list, imgset_rowid_list_)

        config = {'classifier_weight_filepath': 'coco_zebra'}
        class_list = depc.get_property('classifier', gid_list, 'class', config=config)
        score_list = depc.get_property('classifier', gid_list, 'score', config=config)
        score_list_ = [
            score_ if class_ == 'positive' else 1.0 - score_
            for class_, score_ in zip(class_list, score_list)
        ]
        flag_list = [score_ >= score_thresh for score_ in score_list_]
        print('%d / %d' % (flag_list.count(True), len(flag_list),))

        pos_gid_list = ut.compress(gid_list, flag_list)
        # neg_gid_list = ut.compress(gid_list, ut.not_list(flag_list))
        # ibs.set_image_imagesettext(pos_gid_list, ['POSITIVE_SET'] * len(pos_gid_list))
        # ibs.set_image_imagesettext(neg_gid_list, ['NEGATIVE_SET'] * len(neg_gid_list))
        ibs.set_image_imagesettext(pos_gid_list, [imgset_dest_name] * len(pos_gid_list))

        # Run detection
        aids_list = ibs.detect_cnn_yolo(pos_gid_list)
        # aids_list = ibs.get_image_aids(pos_gid_list)
        species_list_list = map(ibs.get_annot_species_texts, aids_list)
        species_set_list = map(set, species_list_list)
        wanted_set = set(['zebra_grevys', 'zebra_plains'])
        flag_list = [
            len(wanted_set & species_set) == 0 for species_set in species_set_list
        ]
        print('%d / %d' % (flag_list.count(True), len(flag_list),))
        nothing_gid_list = ut.compress(pos_gid_list, flag_list)
        imgset_rowid_list_ = [imgset_rowid_list[0]] * len(nothing_gid_list)
        ibs.unrelate_images_and_imagesets(nothing_gid_list, imgset_rowid_list_)
        # ibs.set_image_imagesettext(nothing_gid_list, ['NEGATIVE_SET'] * len(nothing_gid_list))

    @blocking_slot()
    def commit_to_wb_step(back, refresh=True, dry=False):
        """
        Step 6) Commit

        Sets all imagesets as reviwed and ships them to wildbook

        commit step

        Args:
            refresh (bool): (default = True)

        CommandLine:
            python -m wbia.gui.guiback commit_to_wb_step --show

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.guiback import *  # NOQA
            >>> import wbia
            >>> main_locals = wbia.main(defaultdb='testdb1')
            >>> ibs, back = ut.dict_take(main_locals, ['ibs', 'back'])
            >>> ut.exec_funckw(back.do_group_occurrence_step, globals())
            >>> dry = True
            >>> back.do_group_occurrence_step(dry=dry)

            >>> from wbia.gui.guiback import *  # NOQA
            >>> refresh = True
            >>> result = back.commit_to_wb_step(refresh)
            >>> print(result)
            >>> ut.quit_if_noshow()
            >>> import wbia.plottool as pt
            >>> ut.show_if_requested()
        """
        imgsetid = back.get_selected_imgsetid()
        if back.contains_special_imagesets([imgsetid]) or imgsetid is None:
            back.user_warning(
                msg=ut.codeblock(
                    """
                This operation is only allowed for OCCURRENCES.
                Tried to send a special ImageSet to Wildbook as an occurrence.
                Special ImageSets are living entities and are never truely complete.
                """
                )
            )
        else:
            # Check to make sure imagesets are ok:
            # First, check if imageset can be pushed
            ibs = back.ibs

            # imgsets = ibs.imagesets(imgsetid)
            # aid_list = imgsets.aids[0]
            aid_list = ibs.get_imageset_aids(imgsetid)

            assert (
                len(aid_list) > 0
            ), 'ImageSet imgsetid=%r cannot be shipped with0 annots' % (imgsetid,)

            unknown_flags = ibs.is_aid_unknown(aid_list)
            nUnnamed = sum(unknown_flags)

            unnamed_aid_list = ut.compress(aid_list, unknown_flags)
            named_aid_list = ut.compress(aid_list, ut.not_list(unknown_flags))
            nid2_aids = ut.group_items(named_aid_list, ibs.get_annot_nids(named_aid_list))
            nMultiEncounters = sum([len(x) > 1 for x in nid2_aids.values()])
            nSingleEncounters = sum([len(x) == 1 for x in nid2_aids.values()])

            unnamed_ok_aid_list = ibs.filter_annots_general(
                unnamed_aid_list, minqual='ok',
            )
            nUnnamedOk = sum(unnamed_ok_aid_list)

            other_aids = ibs.get_annot_groundtruth(
                ut.take_column(nid2_aids.values(), 0), is_exemplar=True
            )
            other_aids = [set(aids) - set(aid_list) for aids in other_aids]
            nMatchedExemplars = sum([len(x) >= 1 for x in other_aids])

            msg_list = [
                '%d Encounter%s matched an exemplar'
                % ((nMatchedExemplars), '' if nMatchedExemplars == 1 else 's'),
                '%d Encounter%s only one annotation'
                % ((nSingleEncounters), ' has' if nSingleEncounters == 1 else 's have'),
                '%d Encounter%s more than one annotation'
                % ((nMultiEncounters), ' has' if nMultiEncounters == 1 else 's have'),
                '%d annotation%s have not had a name assigned%s'
                % (nUnnamed, '' if nUnnamed == 1 else 's', '!' if nUnnamed > 0 else '.'),
                '%d annotation%s unnamed with an identifiable quality%s'
                % (
                    nUnnamedOk,
                    '' if nUnnamedOk == 1 else 's',
                    '!' if nUnnamedOk > 0 else '.',
                ),
            ]

            # Set all images to be reviewed
            gid_list = back.ibs.get_valid_gids(imgsetid=imgsetid)

            confirm_kw = dict(
                use_msg=ut.codeblock(
                    """
                Have you finished reviewing ALL detections,
                Intra-Occurence IDs, and Vs-Exemplar IDs?

                %s

                Selecting YES will remove this occurrence, mark all images are
                reviewed and processed, and then send it to wildbook.
                """
                )
                % '\n'.join(msg_list),
                title='Complete Occurrence?',
                default='Yes',
            )
            if not back.are_you_sure(**confirm_kw):
                raise guiexcept.UserCancel

            assert nUnnamedOk == 0, (
                'ImageSet imgsetid=%r1 cannot be shipped becuase '
                'annotation(s) %r with an identifiable quality have '
                'not been named'
            ) % (imgsetid, unnamed_ok_aid_list,)

            if not dry:
                # back.start_web_server_parallel(browser=False)
                # gid_list = ibs.get_imageset_gids(imgsetid)
                back.ibs.set_image_reviewed(gid_list, [1] * len(gid_list))
                # Set imageset to be processed
                back.ibs.set_imageset_processed_flags([imgsetid], [1])
                back.ibs.wildbook_signal_imgsetid_list([imgsetid])
                back.front.imageset_tabwgt._close_tab_with_imgsetid(imgsetid)
            if refresh:
                back.front.update_tables([gh.IMAGESET_TABLE])

    def send_unshipped_processed_imagesets(back, refresh=True):
        back.start_web_server_parallel(browser=False)
        processed_set = set(back.ibs.get_valid_imgsetids(processed=True))
        shipped_set = set(back.ibs.get_valid_imgsetids(shipped=True))
        imgsetid_list = list(processed_set - shipped_set)
        back.ibs.wildbook_signal_imgsetid_list(imgsetid_list)

    # --------------------------------------------------------------------------
    # Option menu slots
    # --------------------------------------------------------------------------

    @blocking_slot()
    def layout_figures(back):
        """ Options -> Layout Figures"""
        print('[back] layout_figures')
        fig_presenter.all_figures_tile()
        pass

    @slot_()
    @backreport
    def edit_preferences(back):
        """ Options -> Edit Preferences"""
        print('[back] edit_preferences')
        assert back.ibs is not None, 'No database is loaded. Open a database to continue'
        epw = back.ibs.cfg.createQWidget()
        fig_presenter.register_qt4_win(epw)
        epw.ui.defaultPrefsBUT.clicked.connect(back.default_config)
        epw.show()
        back.edit_prefs_wgt = epw
        # query_cfgstr = ''.join(back.ibs.cfg.query_cfg.get_cfgstr())
        # print('[back] query_cfgstr = %s' % query_cfgstr)
        # print('')

    # --------------------------------------------------------------------------
    # Help menu slots
    # --------------------------------------------------------------------------

    @slot_()
    @backreport
    def view_docs(back):
        """ Help -> View Documentation"""
        print('[back] view_docs')
        raise NotImplementedError()
        pass

    @slot_()
    @backreport
    def view_database_dir(back):
        """ Help -> View Directory Slots"""
        print('[back] view_database_dir')
        ut.view_directory(back.ibs.get_dbdir())
        pass

    @slot_()
    @backreport
    def view_app_files_dir(back):
        print('[back] view_app_files_dir')
        ut.view_directory(ut.get_app_resource_dir('wbia'))
        pass

    @slot_()
    @backreport
    def view_log_dir_local(back):
        print('[back] view_log_dir_local')
        ut.view_directory(back.ibs.get_logdir_local())

    @slot_()
    @backreport
    def view_log_dir_global(back):
        print('[back] view_log_dir_global')
        ut.view_directory(back.ibs.get_logdir_global())

    @slot_()
    @backreport
    def view_logs_global(back):
        print('[back] view_logs_global')
        log_fpath = ut.get_current_log_fpath()
        log_text = back.ibs.get_current_log_text()
        gt.msgbox(
            'Click show details to view logs from log_fpath=%r' % (log_fpath,),
            detailed_msg=log_text,
        )
        # ut.startfile(back.ibs.get_logdir_global())

    @slot_()
    @backreport
    def redownload_detection_models(back):
        print('[back] redownload_detection_models')
        if not back.are_you_sure('[back] redownload_detection_models'):
            return
        ibsfuncs.redownload_detection_models(back.ibs)

    @slot_()
    @backreport
    def delete_cache(back):
        """ Help -> Delete Directory Slots"""
        print('[back] delete_cache')
        if not back.are_you_sure('[back] delete_cache'):
            return
        back.ibs.delete_cache()
        print('[back] finished delete_cache')

    @slot_()
    @backreport
    def delete_thumbnails(back):
        """ Help -> Delete Thumbnails """
        msg = '[back] delete_thumbnails'
        print(msg)
        if not back.are_you_sure(msg):
            return
        back.ibs.delete_thumbnails()
        print('[back] finished delete_thumbnails')

    @slot_()
    @backreport
    def delete_global_prefs(back):
        msg = '[back] delete_global_prefs'
        if not back.are_you_sure(msg):
            return
        ut.delete(ut.get_app_resource_dir('wbia', 'global_cache'))

    @slot_()
    @backreport
    def delete_queryresults_dir(back):
        msg = '[back] delete_queryresults_dir'
        print(msg)
        if not back.are_you_sure(
            use_msg=('Are you sure you want to delete the ' 'cached query results?')
        ):
            return
        ut.delete(back.ibs.qresdir)
        ut.delete(back.ibs.bigcachedir)
        ut.ensuredir(back.ibs.qresdir)
        ut.ensuredir(back.ibs.bigcachedir)

    @blocking_slot()
    def dev_reload(back):
        """ Help -> Developer Reload"""
        print('[back] dev_reload')
        back.ibs.rrr()
        # back.rrr()
        # reload_all()

    @blocking_slot()
    def dev_mode(back):
        """ Help -> Developer Mode"""
        print('[back] dev_mode')
        ibs = back.ibs  # NOQA
        front = back.front  # NOQA
        # import IPython
        # IPython.embed()
        ut.embed()

    @blocking_slot()
    def dev_cls(back):
        """ Help -> Developer Mode"""
        print('[back] dev_cls')
        print('\n'.join([''] * 100))
        if back.ibs is not None:
            back.ibs.reset_table_cache()
        back.refresh_state()
        from wbia.plottool import draw_func2 as df2

        df2.update()

    @slot_()
    @backreport
    def dev_export_annotations(back):
        ibs = back.ibs
        ibs.export_to_xml()

    def start_web_server_parallel(back, browser=True):
        import wbia

        ibs = back.ibs
        if back.web_ibs is None:
            print('[guiback] Starting web service')
            # back.web_ibs = wbia.opendb_in_background(dbdir=ibs.get_dbdir(), web=True, browser=browser)
            back.web_ibs = wbia.opendb_bg_web(
                dbdir=ibs.get_dbdir(), web=True, browser=browser, start_job_queue=False
            )
            print('[guiback] Web service started')
        else:
            print('[guiback] CANNOT START WEB SERVER: WEB INSTANCE ALREADY RUNNING')

    def kill_web_server_parallel(back):
        if back.web_ibs is not None:
            print('[guiback] Stopping web service')
            # back.web_ibs.terminate()
            back.web_ibs.terminate2()
            back.web_ibs = None
        else:
            print('[guiback] CANNOT TERMINATE WEB SERVER: WEB INSTANCE NOT RUNNING')

    @blocking_slot()
    def fix_and_clean_database(back):
        """ Help -> Fix/Clean Database """
        print('[back] Fix/Clean Database')
        back.ibs.fix_and_clean_database()
        back.front.update_tables()

    @blocking_slot()
    def run_integrity_checks(back):
        back.ibs.run_integrity_checks()

    # --------------------------------------------------------------------------
    # File Slots
    # --------------------------------------------------------------------------

    @blocking_slot()
    def new_database(back, new_dbdir=None):
        """ File -> New Database

        Args:
            new_dbdir (None): (default = None)

        CommandLine:
            python -m wbia.gui.guiback new_database --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.gui.guiback import *  # NOQA
            >>> import wbia
            >>> #back = testdata_guiback(defaultdb='testdb1')
            >>> back = testdata_guiback(defaultdb=None)
            >>> dbdir = None
            >>> result = back.new_database(dbdir)
            >>> ut.quit_if_noshow()
            >>> gt.qtapp_loop(qwin=back.front, freq=10)
        """
        if new_dbdir is None:
            old = False
            if old:
                new_dbname = back.user_input(
                    msg='What do you want to name the new database?', title='New Database'
                )
                if new_dbname is None or len(new_dbname) == 0:
                    print('Abort new database. new_dbname=%r' % new_dbname)
                    return
                    new_dbdir_options = ['Choose Directory', 'My Work Dir']
                reply = back.user_option(
                    msg='Where should I put the new database?',
                    title='Import Images',
                    options=new_dbdir_options,
                    default=new_dbdir_options[1],
                    use_cache=False,
                )
                if reply == 'Choose Directory':
                    print('[back] new_database(): SELECT A DIRECTORY')
                    putdir = gt.select_directory(
                        'Select new database directory',
                        other_sidebar_dpaths=[back.get_work_directory()],
                    )
                elif reply == 'My Work Dir':
                    putdir = back.get_work_directory()
                else:
                    print('Abort new database')
                    return
                new_dbdir = join(putdir, new_dbname)

                if not exists(putdir):
                    raise ValueError('Directory %r does not exist.' % putdir)
                if exists(new_dbdir):
                    raise ValueError('New DB %r already exists.' % new_dbdir)

                ut.ensuredir(new_dbdir)
                print('[back] new_database(new_dbdir=%r)' % new_dbdir)
                back.open_database(dbdir=new_dbdir)
            else:
                from wbia.guitool.__PYQT__.QtCore import Qt  # NOQA
                from wbia.guitool.__PYQT__ import QtGui  # NOQA

                dlg = NewDatabaseWidget.as_dialog(
                    back.front, back=back, on_chosen=back.open_database, mode='new'
                )
                dlg.exec_()

    @blocking_slot()
    def open_database(back, dbdir=None):
        """
        File -> Open Database

        Args:
            dbdir (None): (default = None)

        CommandLine:
            python -m wbia.gui.guiback --test-open_database

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.guiback import *  # NOQA
            >>> back = testdata_guiback(defaultdb='testdb1')
            >>> testdb0 = sysres.db_to_dbdir('testdb0')
            >>> testdb1 = sysres.db_to_dbdir('testdb1')
            >>> print('[TEST] TEST_OPEN_DATABASE testdb1=%r' % testdb1)
            >>> back.open_database(testdb1)
            >>> print('[TEST] TEST_OPEN_DATABASE testdb0=%r' % testdb0)
            >>> back.open_database(testdb0)
            >>> import wbia
            >>> #dbdir = join(wbia.sysres.get_workdir(), 'PZ_MTEST', '_ibsdb')
            >>> dbdir = None
            >>> result = back.open_database(dbdir)
            >>> print(result)
        """
        if dbdir is None:
            print('[back] new_database(): SELECT A DIRECTORY')
            # director
            dbdir = gt.select_directory(
                'Open a database directory',
                other_sidebar_dpaths=[back.get_work_directory()],
            )
            if dbdir is None:
                return
        print('[back] open_database(dbdir=%r)' % dbdir)
        with ut.Indenter(lbl='    [opendb]'):
            try:
                # should this use wbia.opendb? probably. at least it should be
                # be request IBEISControl
                # ibs = IBEISControl.IBEISController(dbdir=dbdir)
                ibs = IBEISControl.request_IBEISController(dbdir=dbdir)
                back.connect_wbia_control(ibs)
            except Exception as ex:
                ut.printex(ex, 'caught Exception while opening database')
                raise
            else:
                sysres.set_default_dbdir(dbdir)

    @blocking_slot()
    def export_database_as_csv(back):
        """ File -> Export Database """
        print('[back] export_database_as_csv')
        dump_dir = join(back.ibs.get_dbdir(), 'CSV_DUMP')
        ut.ensuredir(dump_dir)
        ut.view_directory(dump_dir)
        back.ibs.dump_database_csv()

    @blocking_slot()
    def backup_database(back):
        """ File -> Backup Database"""
        print('[back] backup_database')
        back.ibs.backup_database()

    @blocking_slot()
    def make_database_duplicate(back):
        """ File -> Copy Database"""
        print('[back] make_database_duplicate')

        def on_chosen(new_dbdir):
            back.ibs.copy_database(new_dbdir)

        dlg = NewDatabaseWidget.as_dialog(
            back.front, back=back, on_chosen=on_chosen, mode='copy'
        )
        dlg.exec_()

    @blocking_slot()
    def import_images_from_file(
        back, gpath_list=None, refresh=True, as_annots=False, clock_offset=False
    ):
        r"""
        File -> Import Images From File

        Example
            >>> # xdoctest: +REQUIRES(--gui)
            >>> print('[TEST] GET_TEST_IMAGE_PATHS')
            >>> # The test api returns a list of interesting chip indexes
            >>> mode = 'FILE'
            >>> if mode == 'FILE':
            >>>     gpath_list = list(map(utool.unixpath, grabdata.get_test_gpaths()))
            >>> #
            >>>     # else:
            >>>     #    dir_ = utool.truepath(join(sysres.get_workdir(), 'PZ_MOTHERS/images'))
            >>>     #    gpath_list = utool.list_images(dir_, fullpath=True, recursive=True)[::4]
            >>>     print('[TEST] IMPORT IMAGES FROM FILE\n * gpath_list=%r' % gpath_list)
            >>>     gid_list = back.import_images(gpath_list=gpath_list)
            >>>     thumbtup_list = ibs.get_image_thumbtup(gid_list)
            >>>     imgpath_list = [tup[1] for tup in thumbtup_list]
            >>>     gpath_list2 = ibs.get_image_paths(gid_list)
            >>>     for path in gpath_list2:
            >>>         assert path in imgpath_list, "Imported Image not in db, path=%r" % path
            >>> elif mode == 'DIR':
            >>>     dir_ = grabdata.get_testdata_dir()
            >>>     print('[TEST] IMPORT IMAGES FROM DIR\n * dir_=%r' % dir_)
            >>>     gid_list = back.import_images(dir_=dir_)
            >>> else:
            >>>     raise AssertionError('unknown mode=%r' % mode)
            >>> print('[TEST] * len(gid_list)=%r' % len(gid_list))
        """
        print('[back] import_images_from_file')
        if back.ibs is None:
            raise ValueError('back.ibs is None! must open IBEIS database first')
        if gpath_list is None:
            gpath_list = gt.select_images('Select image files to import')

        ibs = back.ibs
        gid_list = back.ibs.add_images(
            gpath_list,
            as_annots=as_annots,
            location_for_names=ibs.cfg.other_cfg.location_for_names,
        )
        back._process_new_images(refresh, gid_list, clock_offset=clock_offset)
        return gid_list

    @blocking_slot()
    def import_button_click(back):
        msg = 'How do you want to import images?'
        ans = back.user_option(
            msg=msg,
            title='Import Images',
            options=[
                'Directory',
                'Files',
                'Smart XML',
                'Encounters (1)',
                'Encounters (2)',
            ],
            use_cache=False,
            default='Directory',
        )
        if ans == 'Directory':
            back.import_images_from_dir()
        elif ans == 'Files':
            back.import_images_from_file()
        elif ans == 'Smart XML':
            back.import_images_from_dir_with_smart()
        elif ans == 'Encounters (1)':
            back.import_images_from_encounters_1()
        elif ans == 'Encounters (2)':
            back.import_images_from_encounters_2()
        elif ans is None:
            pass
        else:
            raise Exception('Unknown anser=%r' % (ans,))

    @blocking_slot()
    def import_images_from_dir(
        back,
        dir_=None,
        size_filter=None,
        refresh=True,
        clock_offset=False,
        return_dir=False,
        defaultdir=None,
    ):
        """ File -> Import Images From Directory"""
        print('[back] import_images_from_dir')
        if dir_ is None:
            dir_ = gt.select_directory(
                'Select directory with images in it', directory=defaultdir
            )
        if dir_ is None:
            return
        gpath_list = ut.list_images(dir_, fullpath=True, recursive=True)
        if size_filter is not None:
            raise NotImplementedError('Can someone implement the size filter?')
        ibs = back.ibs
        gid_list = back.ibs.add_images(
            gpath_list, location_for_names=ibs.cfg.other_cfg.location_for_names
        )
        back._process_new_images(refresh, gid_list, clock_offset=clock_offset)
        if return_dir:
            return gid_list, dir_
        else:
            return gid_list

    @blocking_slot()
    def import_images_from_dir_with_smart(
        back,
        dir_=None,
        size_filter=None,
        refresh=True,
        smart_xml_fpath=None,
        defaultdir=None,
    ):
        """ File -> Import Images From Directory with smart

        Args:
            dir_ (None): (default = None)
            size_filter (None): (default = None)
            refresh (bool): (default = True)

        Returns:
            list: gid_list

        CommandLine:
            python -m wbia.gui.guiback --test-import_images_from_dir_with_smart --show
            python -m wbia.gui.guiback --test-import_images_from_dir_with_smart --show --auto

        Example:
            >>> # DEV_GUI_DOCTEST
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.guiback import *  # NOQA
            >>> back = testdata_guiback(defaultdb='freshsmart_test', delete_ibsdir=True, allow_newdir=True)
            >>> ibs = back.ibs
            >>> defaultdir = ut.truepath('~/lewa-desktop/Desktop/GZ_Foal_Patrol_22_06_2015')
            >>> dir_ = None if not ut.get_argflag('--auto') else join(defaultdir, 'Photos')
            >>> smart_xml_fpath = None if not ut.get_argflag('--auto') else join(defaultdir, 'Patrols', 'LWC_000526LEWA_GZ_FOAL_PATROL.xml')
            >>> size_filter = None
            >>> refresh = True
            >>> gid_list = back.import_images_from_dir_with_smart(dir_, size_filter, refresh, defaultdir=defaultdir, smart_xml_fpath=smart_xml_fpath)
            >>> result = ('gid_list = %s' % (str(gid_list),))
            >>> print(result)
            >>> ut.quit_if_noshow()
            >>> gt.qtapp_loop(back.mainwin, frequency=100)
        """
        print('[back] import_images_from_dir_with_smart')
        gid_list, add_dir_ = back.import_images_from_dir(
            dir_=dir_,
            size_filter=size_filter,
            refresh=False,
            clock_offset=False,
            return_dir=True,
            defaultdir=defaultdir,
        )
        back._group_images_with_smartxml(
            gid_list,
            refresh=refresh,
            smart_xml_fpath=smart_xml_fpath,
            defaultdir=dirname(add_dir_),
        )

    def _group_images_with_smartxml(
        back, gid_list, refresh=True, smart_xml_fpath=None, defaultdir=None
    ):
        """
        Clusters the newly imported images with smart xml file
        """
        if gid_list is not None and len(gid_list) > 0:
            if smart_xml_fpath is None:
                name_filter = 'XML Files (*.xml)'
                xml_path_list = gt.select_files(
                    caption='Select Patrol XML File:',
                    directory=defaultdir,
                    name_filter=name_filter,
                    single_file=True,
                )
                try:
                    assert len(xml_path_list) == 1, 'Must specity one Patrol XML file'
                    smart_xml_fpath = xml_path_list[0]
                    assert (
                        len(smart_xml_fpath) > 0
                    ), 'Must specity a valid Patrol XML file'
                except AssertionError as e:
                    back.ibs.delete_images(gid_list)
                    print(
                        (
                            '[back] ERROR: Parsing Patrol XML file failed, '
                            'rolling back by deleting %d images...'
                        )
                        % (len(gid_list,))
                    )
                    raise e

            back.ibs.compute_occurrences_smart(gid_list, smart_xml_fpath)
        if refresh:
            back.update_special_imagesets_()
            # back.front.update_tables([gh.IMAGESET_TABLE])
            back.front.update_tables()

    def _process_new_images(back, refresh, gid_list, clock_offset=False):
        if refresh:
            back.update_special_imagesets_()
            back.front.update_tables([gh.IMAGE_TABLE, gh.IMAGESET_TABLE])
        if clock_offset:
            co_wgt = clock_offset_gui.ClockOffsetWidget(back.ibs, gid_list)
            co_wgt.show()
        return gid_list

    @blocking_slot()
    def import_images_from_encounters(
        back,
        level=1,
        dir_list=None,
        size_filter=None,
        refresh=True,
        clock_offset=False,
        return_dir=False,
        defaultdir=None,
    ):
        import os

        """ File -> Import Images From Encounters"""
        print('[back] import_images_from_encounters')
        assert level in [1, 2]
        if dir_list is None:
            if level == 1:
                prompt = 'Select folder(s) of encounter(s) (1 level - folders with only images)'
            if level == 2:
                prompt = 'Select folder(s) of encounter(s) (2 levels - folders of folders with only images)'
            dir_list = gt.select_directories(prompt, directory=defaultdir)
        if dir_list is None or len(dir_list) == 0:
            return

        # We need to check that the first directory is not a subdirectory of the others
        if len(dir_list) >= 2:
            subdir1 = dir_list[0]
            subdir2, _ = os.path.split(dir_list[1])
            if subdir1 == subdir2:
                dir_list = dir_list[1:]

        # Check the folders for invalid values
        invalid_list = []
        warning_set = set([])
        for index, dir_ in enumerate(dir_list):
            for root, subdirs, files in os.walk(dir_):
                images = ut.list_images(root, recursive=False)
                try:
                    # Assert structure
                    if level == 1:
                        assert len(subdirs) == 0
                        assert len(images) > 0
                    if level == 2:
                        assert len(images) == 0
                        assert len(subdirs) > 0

                        # Check subdirectories for level 1 structure
                        for subdir in subdirs:
                            for root_, subdirs_, files_ in os.walk(subdir):
                                images_ = ut.list_images(root, recursive=False)
                                try:
                                    assert len(subdirs_) == 0
                                    assert len(images_) > 0
                                except AssertionError:
                                    invalid_list.append(join(root, root_))
                                # Combined a warning set of non-image files
                                for file_ in set(files_) - set(images_):
                                    if len(file_.strip('.')) == 0:
                                        warning_set.add(join(root, file_))
                                # Only look at the root path
                                break
                except AssertionError:
                    invalid_list.append(root)
                # Combined a warning set of non-image files
                for file_ in set(files) - set(images):
                    if len(file_.strip('.')) == 0:
                        warning_set.add(file_)
                # Only look at the root path
                break

        # If invalid, give user input information
        invalid = len(invalid_list) > 0
        if invalid:
            if level == 1:
                raise IOError(
                    """
                    [guiback] The following encounter folder structures (1 level) are not valid: %r
                    [guiback]     * The selected folders must contain images
                    [guiback]     * The selected folders must NOT contain any sub-folders
                    [guiback]     * The selected folders must NOT be empty
                """
                    % (invalid_list,)
                )
            if level == 2:
                raise IOError(
                    """
                    [guiback] The following encounter folder structures (2 levels) are not valid: %r
                    [guiback]     * The selected folders must NOT contain images
                    [guiback]     * The selected folders must contain sub-folders
                    [guiback]     * The selected folders must NOT be empty
                    [guiback]     * The sub-folders in the selected folders must contain images
                    [guiback]     * The sub-folders in the selected folders must NOT contain any sub-folders
                    [guiback]     * The sub-folders in the selected folders must NOT be empty
                """
                    % (invalid_list,)
                )
        print('[guiback] Encounters are valid, continue with import')

        # print any warning files
        if len(warning_set) > 0:
            warning_list = list(sorted(warning_set))
            args = (warning_list,)
            print(
                '[guiback] WARNING: Some files in the encounters will not be imported: %r'
                % args
            )

        # Compile the list of images now that the encounter's structure have been verified
        gpath_list = []
        for dir_ in dir_list:
            gpath_list_ = ut.list_images(dir_, fullpath=True, recursive=True)
            gpath_list += gpath_list_

        # Check that the encounters behave as expected
        if size_filter is not None:
            raise NotImplementedError('Can someone implement the size filter?')

        # Add images to ibs
        ibs = back.ibs
        gid_list = back.ibs.add_images(
            gpath_list, location_for_names=ibs.cfg.other_cfg.location_for_names
        )

        # Add imagesets for newly added images
        imageset_text_list = []
        for gpath in gpath_list:
            base, gname = os.path.split(gpath)
            base, level1 = os.path.split(base)
            if level == 1:
                imageset_text = level1
            if level == 2:
                base, level2 = os.path.split(base)
                imageset_text = '%s  (+)  %s' % (level2, level1,)
            imageset_text_list.append(imageset_text)
        ibs.set_image_imagesettext(gid_list, imageset_text_list)

        # Refresh GUI and return
        back._process_new_images(refresh, gid_list, clock_offset=clock_offset)
        if return_dir:
            return gid_list, dir_list
        else:
            return gid_list

    @blocking_slot()
    def import_images_from_encounters_1(
        back, dir_list=None, size_filter=None, refresh=True, defaultdir=None
    ):
        """ File -> Import Images From Encounters (1 level)

        Args:
            dir_ (None): (default = None)
            size_filter (None): (default = None)
            refresh (bool): (default = True)

        Returns:
            list: gid_list
        """
        print('[back] import_images_from_encounters_1')
        gid_list, add_dir_ = back.import_images_from_encounters(
            level=1,
            dir_list=dir_list,
            size_filter=size_filter,
            refresh=False,
            clock_offset=False,
            return_dir=True,
            defaultdir=defaultdir,
        )

    @blocking_slot()
    def import_images_from_encounters_2(
        back, dir_list=None, size_filter=None, refresh=True, defaultdir=None
    ):
        """ File -> Import Images From Encounters (2 levels)

        Args:
            dir_ (None): (default = None)
            size_filter (None): (default = None)
            refresh (bool): (default = True)

        Returns:
            list: gid_list
        """
        print('[back] import_images_from_encounters_2')
        gid_list, add_dir_ = back.import_images_from_encounters(
            level=2,
            dir_list=dir_list,
            size_filter=size_filter,
            refresh=False,
            clock_offset=False,
            return_dir=True,
            defaultdir=defaultdir,
        )

    @blocking_slot()
    def import_images_as_annots_from_file(back, gpath_list=None, refresh=True):
        return back.import_images_from_file(gpath_list=None, refresh=True, as_annots=True)

    @slot_()
    @backreport
    def localize_images(back):
        """ File -> Localize Images """
        print('[back] localize_images')
        back.ibs.localize_images()

    @slot_()
    def quit(back):
        """ File -> Quit"""
        print('[back] ')
        # back.cleanup()
        gt.exit_application()

    # --------------------------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------------------------

    def user_info(back, **kwargs):
        return gt.user_info(parent=back.front, **kwargs)

    def user_warning(back, title='Warning', **kwargs):
        return gt.user_info(parent=back.front, title=title, **kwargs)

    def user_input(back, msg='user input', **kwargs):
        return gt.user_input(parent=back.front, msg=msg, **kwargs)

    def user_option(back, **kwargs):
        if kwargs.get('config', None) is not None:
            # options, config, msg, title
            dlg = gt.ConfigConfirmWidget.as_dialog(**kwargs)
            # dlg.resize(700, 500)
            confirm_widget = dlg.widget
            dlg.exec_()
            return confirm_widget.confirm_option, confirm_widget.config
            # return gt.user_option(parent=back.front, **kwargs)
        else:
            return gt.user_option(parent=back.front, **kwargs)

    def are_you_sure(
        back,
        use_msg=None,
        title='Confirmation',
        default=None,
        action=None,
        detailed_msg=None,
    ):
        """ Prompt user for conformation before changing something """
        if action is None:
            default_msg = 'Are you sure?'
        else:
            default_msg = 'Are you sure you want to %s?' % (action,)
        msg = default_msg if use_msg is None else use_msg
        print('[back] Asking User if sure')
        print('[back] title = %s' % (title,))
        print('[back] msg =\n%s' % (msg,))
        print('[back] detailed_msg =\n%s' % (detailed_msg,))
        if ut.get_argflag('-y') or ut.get_argflag('--yes'):
            # DONT ASK WHEN SPECIFIED
            return True
        ans = back.user_option(
            msg=msg,
            title=title,
            options=['Yes', 'No'],
            use_cache=False,
            default=default,
            detailed_msg=detailed_msg,
        )
        print('[back] User answered: %r' % (ans,))
        return ans == 'Yes'

    def get_work_directory(back):
        return sysres.get_workdir()

    def _eidfromkw(back, kwargs):
        if 'imgsetid' not in kwargs:
            imgsetid = back.get_selected_imgsetid()
        else:
            imgsetid = kwargs['imgsetid']
        return imgsetid

    def contains_special_imagesets(back, imgsetid_list):
        isspecial_list = back.ibs.is_special_imageset(imgsetid_list)
        return any(isspecial_list)

    def display_special_imagesets_error(back):
        back.user_warning(msg='Contains special imagesets')

    @slot_()
    def override_all_annotation_species(back, aids=None, gids=None):
        """
        Give the user a dialog box asking to input a species
        """
        if aids is None:
            if gids is not None:
                aid_list = list(ub.flatten(back.ibs.images(gids=gids).aids))
            else:
                aid_list = back.ibs.get_valid_aids()
        else:
            aid_list = aids
        resp = gt.user_input(
            title='edit species',
            msg='Override species for {} annots'.format(len(aid_list)),
            text='',
        )
        if resp is not None:
            print('override_all_annotation_species. resp = %r' % (resp,))
            species_rowid = back.ibs.add_species(resp)
            use_msg = 'Are you sure you want to change %d annotations species to %r?' % (
                len(aid_list),
                resp,
            )
            if back.are_you_sure(use_msg=use_msg):
                print('performing override')
                back.ibs.set_annot_species_rowids(
                    aid_list, [species_rowid] * len(aid_list)
                )
                # FIXME: api-cache is broken here too
                back.ibs.reset_table_cache()

    @blocking_slot()
    def update_species_nice_name(back):
        from wbia.control.manual_species_funcs import _convert_species_nice_to_code

        ibs = back.ibs
        species_text = back.get_selected_species()
        if species_text in [const.UNKNOWN, '']:
            back.user_warning(msg='Cannot rename this species...')
            raise guiexcept.UserCancel
        species_rowid = ibs.get_species_rowids_from_text(species_text)
        species_nice = ibs.get_species_nice(species_rowid)
        new_species_nice = back.user_input(
            msg='Rename species\n    Name: %r \n    Tag:  %r'
            % (species_nice, species_text),
            title='Rename Species',
        )
        if new_species_nice is not None:
            species_rowid = [species_rowid]
            new_species_nice = [new_species_nice]
            species_code = _convert_species_nice_to_code(new_species_nice)
            ibs._set_species_nice(species_rowid, new_species_nice)
            ibs._set_species_code(species_rowid, species_code)
            back.ibswgt.update_species_available(
                reselect=True, reselect_new_name=new_species_nice[0]
            )

    @blocking_slot()
    def delete_selected_species(back):
        ibs = back.ibs
        species_text = back.get_selected_species()
        if species_text in [const.UNKNOWN, '']:
            back.user_warning(msg='Cannot delete this species...')
            raise guiexcept.UserCancel
        species_rowid = ibs.get_species_rowids_from_text(species_text)
        species_nice = ibs.get_species_nice(species_rowid)

        msg_str = (
            'You are about to delete species\n    Name: %r \n    '
            + 'Tag:  %r\n\nDo you wish to continue?\nAll annotations '
            + 'with this species will be set to unknown.'
        )
        msg_str = msg_str % (species_nice, species_text,)
        confirm_kw = dict(use_msg=msg_str, title='Delete Selected Species?', default='No')
        if not back.are_you_sure(**confirm_kw):
            raise guiexcept.UserCancel
        ibs.delete_species([species_rowid])
        back.ibswgt.update_species_available(deleting=True)

    @slot_()
    def set_exemplars_from_quality_and_viewpoint_(back):
        exemplars_per_view = back.ibs.cfg.other_cfg.exemplars_per_view
        imgsetid = back.get_selected_imgsetid()
        print('set_exemplars_from_quality_and_viewpoint, imgsetid=%r' % (imgsetid,))
        HACK = back.ibs.cfg.other_cfg.enable_custom_filter
        assert not HACK, 'enable_custom_filter is no longer supported'

        back.ibs.set_exemplars_from_quality_and_viewpoint(
            imgsetid=imgsetid, exemplars_per_view=exemplars_per_view
        )

    @slot_()
    def batch_rename_consecutive_via_species_(back):
        # imgsetid = back.get_selected_imgsetid()
        # back.ibs.batch_rename_consecutive_via_species(imgsetid=imgsetid)
        imgsetid = None
        print('batch_rename_consecutive_via_species, imgsetid=%r' % (imgsetid,))
        location_text = back.ibs.cfg.other_cfg.location_for_names
        back.ibs.batch_rename_consecutive_via_species(
            imgsetid=imgsetid, location_text=location_text
        )

    @slot_()
    def run_tests(back):
        from wbia.tests import run_tests

        run_tests.run_tests()

    @slot_()
    def run_utool_tests(back):
        import utool.tests.run_tests

        utool.tests.run_tests.run_tests()

    @slot_()
    def run_vtool_tests(back):
        import vtool.tests.run_tests

        vtool.tests.run_tests.run_tests()

    @slot_()
    def assert_modules(back):
        from wbia.tests import assert_modules

        detailed_msg = assert_modules.assert_modules()
        gt.msgbox(msg='Running checks', title='Module Checks', detailed_msg=detailed_msg)

    @slot_()
    def display_dbinfo(back):
        r"""
        CommandLine:
            python -m wbia.gui.guiback --test-display_dbinfo

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.gui.guiback import *  # NOQA
            >>> back = testdata_guiback()
            >>> result = back.display_dbinfo()
            >>> print(result)
        """
        dbinfo = back.ibs.get_dbinfo_str()
        print(dbinfo)
        gt.msgbox(msg=back.ibs.get_infostr(), title='DBInfo', detailed_msg=dbinfo)

    @slot_()
    def show_about_message(back):
        import wbia

        version = wbia.__version__
        about_msg = (
            'IBEIS version %s\nImage Based Ecological Information System\nhttp://wbia.org/'
            % (version,)
        )
        gt.msgbox(msg=about_msg, title='About')

    @slot_()
    def take_screenshot(back):
        """ dev command only """
        print('[back] TAKING SCREENSHOT')
        from wbia.guitool.__PYQT__.QtGui import QPixmap

        # screengrab_fpath = ut.truepath('~/latex/wbia_userguide/figures/filemenu.jpg')

        # Find the focused window
        app = gt.get_qtapp()
        widget = app.focusWidget()
        if widget is None or widget == 0:
            widget = back.mainwin
        window = widget.window()
        win_title = window.windowTitle()
        window_id = window.winId()

        # Resolve screengrab path
        screengrab_dpath = ut.get_argval('--screengrab_dpath', type_=str, default=None)
        screengrab_fname = ut.get_argval('--screengrab_fname', type_=str, default=None)
        if screengrab_fname is None:
            screengrab_fname = win_title
        if screengrab_dpath is None:
            screengrab_dpath = './screenshots'
            ut.ensuredir(screengrab_dpath)
        screengrab_dpath = ut.truepath(screengrab_dpath)
        screengrab_fname = ut.sanitize_filename(screengrab_fname)
        fpath_base = join(screengrab_dpath, screengrab_fname)
        fpath_fmt = fpath_base + '_%d.jpg'
        screengrab_fpath = ut.get_nonconflicting_path(fpath_fmt)

        # Grab image in window
        screenimg = QPixmap.grabWindow(window_id)
        # Save image to disk
        screenimg.save(screengrab_fpath, 'jpg')
        print('saved screengrab to %r' % (screengrab_fpath,))
        if ut.get_argflag('--diskshow'):
            ut.startfile(screengrab_fpath)

    @slot_()
    def make_qt_graph_interface(back):
        from wbia.viz import viz_graph2

        imgsetid = back.get_selected_imgsetid()
        aids = back.ibs.get_valid_aids(imgsetid=imgsetid)
        if len(aids) == 0:
            raise AssertionError('Choose an imageset with annotations')
        back.graph_iden_win = viz_graph2.make_qt_graph_interface(aids=aids, ibs=back.ibs)

    @slot_()
    def reconnect_controller(back):
        back.connect_wbia_control(back.ibs)

    @slot_()
    def browse_wildbook(back):
        wb_base_url = back.ibs.get_wildbook_base_url()
        ut.get_prefered_browser().open(wb_base_url)

    @slot_()
    def install_wildbook(back):
        import wbia.control.wildbook_manager as wb_man

        wb_man.install_wildbook()

    @slot_()
    def startup_wildbook(back):
        import wbia.control.wildbook_manager as wb_man

        back.wb_server_running = True
        wb_man.startup_wildbook_server()

    @slot_()
    def shutdown_wildbook(back):
        import wbia.control.wildbook_manager as wb_man

        wb_man.shutdown_wildbook_server()
        back.wb_server_running = False

    @slot_()
    def force_wildbook_namechange(back):
        back.ibs.wildbook_signal_annot_name_changes()

    @slot_()
    def set_workdir(back):
        import wbia

        wbia.sysres.set_workdir(work_dir=None, allow_gui=True)

    @slot_()
    def ensure_demodata(back):
        from wbia import demodata

        demodata.ensure_demodata()

    @slot_()
    def launch_ipy_notebook(back):
        from wbia.templates import generate_notebook

        generate_notebook.autogen_ipynb(back.ibs, launch=True)

    @slot_()
    def update_source_install(back):
        import wbia
        from os.path import dirname

        repo_path = dirname(ut.truepath(ut.get_modpath(wbia, prefer_pkg=True)))
        with ut.ChdirContext(repo_path):
            command = ut.python_executable() + ' super_setup.py pull'
            ut.cmd(command)
        print('Done updating source install')

    @slot_()
    def toggle_output_widget(back):
        current = back.front.outputLog.isVisible()
        back.front.outputLog.setVisible(not current)


def testdata_guiback(defaultdb='testdb2', **kwargs):
    import wbia

    print('testdata guiback')
    if defaultdb is None:
        back = wbia.main_module._init_gui()
        # back = MainWindowBackend()
    else:
        main_locals = wbia.main(defaultdb=defaultdb, **kwargs)
        back = main_locals['back']
    return back


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m wbia.gui.guiback
    """
    import xdoctest

    xdoctest.doctest_module(__file__)
