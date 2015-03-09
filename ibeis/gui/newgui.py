#!/usr/bin/env python2.7
"""
This should probably be renamed guifront.py This defines all of the visual
components to the GUI It is invoked from guiback, which handles the nonvisual
logic.


BUGS:
    * On opening new database the tables are refreshed for the closing database.
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, map, filter  # NOQA
from os.path import isdir
import sys
from ibeis import constants as const
import functools
from guitool.__PYQT__ import QtGui, QtCore
from guitool.__PYQT__.QtCore import Qt
from guitool.__PYQT__.QtGui import QSizePolicy
from guitool import signal_, slot_, checks_qt_error, ChangeLayoutContext  # NOQA
from ibeis import ibsfuncs
from ibeis.gui import guiheaders as gh
from ibeis.gui import guimenus
import six
from ibeis.viz.interact import interact_annotations2
from ibeis.gui.guiheaders import (IMAGE_TABLE, IMAGE_GRID, ANNOTATION_TABLE,
                                  NAME_TABLE, NAMES_TREE, ENCOUNTER_TABLE)
from ibeis.gui.models_and_views import (IBEISStripeModel, IBEISTableView,
                                        IBEISItemModel, IBEISTreeView,
                                        EncTableModel, EncTableView,
                                        IBEISTableWidget, IBEISTreeWidget,
                                        EncTableWidget)
import guitool
from plottool import color_funcs
import utool as ut
import plottool as pt
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[newgui]')


IBEIS_WIDGET_BASE = QtGui.QWidget

VERBOSE_GUI = ut.VERBOSE or ut.get_argflag('--verbose-gui')
WITH_GUILOG = ut.get_argflag('--guilog')
#WITH_GUILOG = not ut.get_argflag('--noguilog')

"""
from ibeis.gui.guiheaders import (IMAGE_TABLE, IMAGE_GRID, ANNOTATION_TABLE,
                                  NAME_TABLE, NAMES_TREE, ENCOUNTER_TABLE)
ibsgwt = back.front
view   = ibsgwt.views[IMAGE_TABLE]
model  = ibsgwt.models[IMAGE_TABLE]
row = model.get_row_from_id(3)
view.selectRow(row)
"""

#############################
###### Tab Widgets #######
#############################


class APITabWidget(QtGui.QTabWidget):
    def __init__(tabwgt, parent=None, horizontalStretch=1):
        QtGui.QTabWidget.__init__(tabwgt, parent)
        tabwgt.ibswgt = parent
        tabwgt._sizePolicy = guitool.newSizePolicy(tabwgt, horizontalStretch=horizontalStretch)
        tabwgt.setSizePolicy(tabwgt._sizePolicy)
        #tabwgt.currentChanged.connect(tabwgt.setCurrentIndex)

    #def setCurrentIndex(tabwgt, index):
    #    tblname = tabwgt.ibswgt.tblname_list[index]
    #    print('Set %r current Index: %r ' % (tblname, index))
    #    #model = tabwgt.ibswgt.models[tblname]
    #    #with ChangeLayoutContext([model]):
    #    #    QtGui.QTabWidget.setCurrentIndex(tabwgt, index)


class EncoutnerTabWidget(QtGui.QTabWidget):
    def __init__(enc_tabwgt, parent=None, horizontalStretch=1):
        QtGui.QTabWidget.__init__(enc_tabwgt, parent)
        enc_tabwgt.ibswgt = parent
        enc_tabwgt.setTabsClosable(True)
        enc_tabwgt.setMaximumSize(9999, guitool.get_cplat_tab_height())
        enc_tabwgt.tabbar = enc_tabwgt.tabBar()
        enc_tabwgt.tabbar.setMovable(False)
        enc_tabwgt.setStyleSheet('border: none;')
        enc_tabwgt.tabbar.setStyleSheet('border: none;')
        sizePolicy = guitool.newSizePolicy(enc_tabwgt, horizontalStretch=horizontalStretch)
        enc_tabwgt.setSizePolicy(sizePolicy)

        enc_tabwgt.tabCloseRequested.connect(enc_tabwgt._close_tab)
        enc_tabwgt.currentChanged.connect(enc_tabwgt._on_change)

        enc_tabwgt.eid_list = []
        # TURNING ON / OFF ALL IMAGES
        # enc_tabwgt._add_enc_tab(-1, const.ALL_IMAGE_ENCTEXT)

    @slot_(int)
    def _on_change(enc_tabwgt, index):
        """ Switch to the current encounter tab """
        print('[encounter_tab_widget] _onchange(index=%r)' % (index,))
        if 0 <= index and index < len(enc_tabwgt.eid_list):
            eid = enc_tabwgt.eid_list[index]
            #if ut.VERBOSE:
            print('[ENCTAB.ONCHANGE] eid = %r' % (eid,))
            enc_tabwgt.ibswgt._change_enc(eid)
        else:
            enc_tabwgt.ibswgt._change_enc(-1)

    @slot_(int)
    def _close_tab(enc_tabwgt, index):
        print('[encounter_tab_widget] _close_tab(index=%r)' % (index,))
        if enc_tabwgt.eid_list[index] is not None:
            enc_tabwgt.eid_list.pop(index)
            enc_tabwgt.removeTab(index)

    @slot_()
    def _close_all_tabs(enc_tabwgt):
        print('[encounter_tab_widget] _close_all_tabs()')
        while len(enc_tabwgt.eid_list) > 0:
            index = 0
            enc_tabwgt.eid_list.pop(index)
            enc_tabwgt.removeTab(index)

    @slot_(int)
    def _close_tab_with_eid(enc_tabwgt, eid):
        print('[encounter_tab_widget] _close_tab_with_eid(eid=%r)' % (eid))
        try:
            index = enc_tabwgt.eid_list.index(eid)
            enc_tabwgt._close_tab(index)
        except:
            pass

    def _add_enc_tab(enc_tabwgt, eid, enctext):
        # <HACK>
        # if enctext == const.ALL_IMAGE_ENCTEXT:
        #     eid = None
        # </HACK>
        if eid not in enc_tabwgt.eid_list:
            # tab_name = str(eid) + ' - ' + str(enctext)
            tab_name = str(enctext)
            enc_tabwgt.addTab(QtGui.QWidget(), tab_name)

            enc_tabwgt.eid_list.append(eid)
            index = len(enc_tabwgt.eid_list) - 1
        else:
            index = enc_tabwgt.eid_list.index(eid)

        enc_tabwgt.setCurrentIndex(index)
        enc_tabwgt._on_change(index)

    def _update_enc_tab_name(enc_tabwgt, eid, enctext):
        for index, _id in enumerate(enc_tabwgt.eid_list):
            if eid == _id:
                enc_tabwgt.setTabText(index, enctext)


#############################
###### Window Widgets #######
#############################


class IBEISMainWindow(QtGui.QMainWindow):
    quitSignal = signal_()
    dropSignal = signal_(list)
    def __init__(mainwin, back=None, ibs=None, parent=None):
        QtGui.QMainWindow.__init__(mainwin, parent)
        # Menus
        mainwin.setUnifiedTitleAndToolBarOnMac(False)
        guimenus.setup_menus(mainwin, back)
        # Central Widget
        mainwin.ibswgt = IBEISGuiWidget(back=back, ibs=ibs, parent=mainwin)
        mainwin.setCentralWidget(mainwin.ibswgt)
        mainwin.setAcceptDrops(True)
        if back is not None:
            mainwin.quitSignal.connect(back.quit)
        else:
            raise AssertionError('need backend')
        mainwin.dropSignal.connect(mainwin.ibswgt.imagesDropped)
        #
        mainwin.resize(900, 600)

    @slot_()
    def closeEvent(mainwin, event):
        event.accept()
        mainwin.quitSignal.emit()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            self.dropSignal.emit(links)
        else:
            event.ignore()

    @slot_()
    def expand_names_tree(mainwin):
        view = mainwin.ibswgt.views[gh.NAMES_TREE]
        view.expandAll()


class IBEISGuiWidget(IBEIS_WIDGET_BASE):
    #@checks_qt_error
    def __init__(ibswgt, back=None, ibs=None, parent=None):
        IBEIS_WIDGET_BASE.__init__(ibswgt, parent)
        ibswgt.ibs = ibs
        ibswgt.back = back
        # Sturcutres that will hold models and views
        ibswgt.models       = {}
        ibswgt.views        = {}
        ibswgt.redirects    = {}
        #ibswgt.widgets      = {}

        # FIXME: Duplicate models
        # Create models and views
        # Define the abstract item models and views for the tables
        ibswgt.tblname_list   = []
        ibswgt.modelview_defs = []

        # OLD STATIC WAY OF USING TABS AND API VIEWS
        #ibswgt.tblname_list = [
        #    IMAGE_TABLE,
        #    IMAGE_GRID,
        #    ANNOTATION_TABLE,
        #    #
        #    #NAME_TABLE,
        #    #
        #    NAMES_TREE,
        #]
        #ibswgt.modelview_defs = [
        #    (IMAGE_GRID,       IBEISTableWidget, IBEISStripeModel, IBEISTableView),
        #    (ANNOTATION_TABLE, IBEISTableWidget, IBEISItemModel, IBEISTableView),
        #    #
        #    #(NAME_TABLE,       IBEISTableWidget, IBEISItemModel, IBEISTableView),
        #    #
        #    (NAMES_TREE,       IBEISTreeWidget,  IBEISItemModel,  IBEISTreeView),
        #    (ENCOUNTER_TABLE,  EncTableWidget,   EncTableModel,   EncTableView),
        #]

        # NEW DYNAMIC WAY OF USING TABS AND API VIEWS
        if True:
            # ADD IMAGE TABLE
            ibswgt.tblname_list.append(IMAGE_TABLE)
            ibswgt.modelview_defs.append((IMAGE_TABLE,      IBEISTableWidget, IBEISItemModel, IBEISTableView))
        if not ut.get_argflag('--onlyimgtbl'):
            # ADD IMAGE GRID
            ibswgt.tblname_list.append(IMAGE_GRID)
            ibswgt.modelview_defs.append((IMAGE_GRID,       IBEISTableWidget, IBEISStripeModel, IBEISTableView))
        if not (ut.get_argflag('--noannottbl') or ut.get_argflag('--onlyimgtbl')):
            # ADD ANNOT GRID
            ibswgt.tblname_list.append(ANNOTATION_TABLE)
            ibswgt.modelview_defs.append((ANNOTATION_TABLE, IBEISTableWidget, IBEISItemModel, IBEISTableView))
        if not (ut.get_argflag('--nonametree') or ut.get_argflag('--onlyimgtbl')):
            # ADD NAME TREE
            ibswgt.tblname_list.append(NAMES_TREE)
            ibswgt.modelview_defs.append((NAMES_TREE,       IBEISTreeWidget,  IBEISItemModel,  IBEISTreeView))
        # ADD ENCOUNTER TABLE
        ibswgt.super_tblname_list = ibswgt.tblname_list + [ENCOUNTER_TABLE]
        ibswgt.modelview_defs.append((ENCOUNTER_TABLE,  EncTableWidget,   EncTableModel,   EncTableView))

        # DO INITALIZATION
        # Create and layout components
        ibswgt._init_components()
        ibswgt._init_layout()
        # Connect signals and slots
        ibswgt._connect_signals_and_slots()
        # Connect the IBEIS control
        ibswgt.connect_ibeis_control(ibswgt.ibs)

    #@checks_qt_error
    def _init_components(ibswgt):
        """ Defines gui components """
        # Layout
        ibswgt.vlayout = QtGui.QVBoxLayout(ibswgt)
        ibswgt.hsplitter = guitool.newSplitter(ibswgt, Qt.Horizontal, verticalStretch=18)
        ibswgt.vsplitter = guitool.newSplitter(ibswgt, Qt.Vertical)
        #ibswgt.hsplitter = guitool.newWidget(ibswgt, Qt.Horizontal, verticalStretch=18)
        #ibswgt.vsplitter = guitool.newWidget(ibswgt)
        #
        # Tables Tab
        ibswgt._tab_table_wgt = APITabWidget(ibswgt, horizontalStretch=81)
        #guitool.newTabWidget(ibswgt, horizontalStretch=81)
        for tblname, WidgetClass, ModelClass, ViewClass in ibswgt.modelview_defs:
            #widget = WidgetClass(parent=ibswgt)
            #ibswgt.widgets[tblname] = widget
            #ibswgt.models[tblname]  = widget.model
            #ibswgt.views[tblname]   = widget.view
            ibswgt.views[tblname]  = ViewClass(parent=ibswgt)  # Make view first to pass as parent
            # FIXME: It is very bad to give the model a view. Only the view should have a model
            ibswgt.models[tblname] = ModelClass(parent=ibswgt.views[tblname])
        # Connect models and views
        for tblname in ibswgt.super_tblname_list:
            ibswgt.views[tblname].setModel(ibswgt.models[tblname])
        # Add Image, ANNOTATION, and Names as tabs
        for tblname in ibswgt.tblname_list:
            #ibswgt._tab_table_wgt.addTab(ibswgt.widgets[tblname], tblname)
            ibswgt._tab_table_wgt.addTab(ibswgt.views[tblname], tblname)
        # Custom Encounter Tab Wiget
        ibswgt.enc_tabwgt = EncoutnerTabWidget(parent=ibswgt, horizontalStretch=19)
        # Other components
        ibswgt.outputLog   = guitool.newOutputLog(ibswgt, pointSize=8,
                                                  visible=WITH_GUILOG, verticalStretch=6)
        ibswgt.progressBar = guitool.newProgressBar(ibswgt, visible=False, verticalStretch=1)
        # New widget has black magic (for implicit layouts) in it
        ibswgt.status_wgt  = guitool.newWidget(ibswgt, Qt.Vertical,
                                               verticalStretch=6,
                                               horizontalSizePolicy=QSizePolicy.Maximum)

        _NEWLBL = functools.partial(guitool.newLabel, ibswgt)
        _NEWBUT = functools.partial(guitool.newButton, ibswgt)
        _COMBO  = functools.partial(guitool.newComboBox, ibswgt)
        _NEWTEXT = functools.partial(guitool.newLineEdit, ibswgt, enabled=False)

        primary_fontkw = dict(bold=True, pointSize=11)
        secondary_fontkw = dict(bold=False, pointSize=9)
        advanced_fontkw = dict(bold=False, pointSize=8, italic=True)

        ibswgt.status_widget_list = [
            _NEWLBL('Selected Encounter: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(),
            _NEWLBL('Selected Image: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(),
            _NEWLBL('Selected Annotation: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(),
            _NEWLBL('Selected Name: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(),
        ]

        back = ibswgt.back

        # TODO: update these options depending on ibs.get_species_with_detectors
        # when a controller is attached to the gui
        detection_combo_box_options = [
            # Text              # Value
            ('Select Species',  'none'),
        ] + const.get_working_species_set()

        ibswgt.species_combo = _COMBO(detection_combo_box_options,
                                      ibswgt.back.change_detection_species,
                                      fontkw=primary_fontkw)

        ibswgt.reviewed_button = _NEWBUT('5) Complete',
                                         ibswgt.back.encounter_reviewed_all_images,
                                         bgcolor=color_funcs.adjust_hsv_of_rgb255((0, 232, 211), 0., -.9, 0.),
                                         fontkw=primary_fontkw,
                                         enabled=False)

        ibswgt.import_button = _NEWBUT('1) Import',
                                       back.import_images_from_file,
                                       bgcolor=(235, 200, 200), fontkw=primary_fontkw)

        ibswgt.encounter_button = _NEWBUT('2) Group',
                                          ibswgt.back.compute_encounters,
                                          bgcolor=(255, 255, 150), fontkw=primary_fontkw)

        ibswgt.detect_button = _NEWBUT('3) Detect',
                                       ibswgt.back.run_detection,
                                       bgcolor=(150, 255, 150), fontkw=primary_fontkw)

        identify_color = (255, 150, 0)
        ibswgt.inc_query_button = _NEWBUT('4) Identify',
                                          ibswgt.back.incremental_query,
                                          bgcolor=identify_color,
                                          fgcolor=(0, 0, 0), fontkw=primary_fontkw)

        color_funcs.adjust_hsv_of_rgb255(identify_color, 0.01, -0.2, 0.0)

        ibswgt.batch_unknown_intra_encounter_query_button = _NEWBUT(
            'Intra Encounter',
            functools.partial(
                back.compute_queries,
                query_is_known=False, query_mode=const.INTRA_ENC_KEY),
            bgcolor=color_funcs.adjust_hsv_of_rgb255(identify_color, -0.01, -0.7, 0.0),
            fgcolor=(0, 0, 0), fontkw=advanced_fontkw)

        ibswgt.batch_unknown_vsexemplar_query_button = _NEWBUT(
            'Vs Exemplar',
            functools.partial(
                back.compute_queries,
                query_is_known=False, query_mode=const.VS_EXEMPLARS_KEY),
            bgcolor=color_funcs.adjust_hsv_of_rgb255(identify_color, 0.01, -0.7, 0.0),
            fgcolor=(0, 0, 0), fontkw=advanced_fontkw)

        ibswgt.control_widget_lists = [
            [
                ibswgt.import_button,

                ibswgt.encounter_button,

                _NEWLBL('Encounter: ', align='right', fontkw=primary_fontkw),

                ibswgt.detect_button,

                ibswgt.inc_query_button,

                ibswgt.reviewed_button,

            ],
            [
                _NEWLBL('Species Selector: ', align='right', fontkw=primary_fontkw),

                ibswgt.species_combo,

                _NEWLBL(''),

                _NEWLBL('*Advanced Batch Identification: ', align='right', fontkw=advanced_fontkw),
                ibswgt.batch_unknown_intra_encounter_query_button,
                ibswgt.batch_unknown_vsexemplar_query_button,

                _NEWLBL(''),

            ],
        ]

        #detection_combo_box_options = [
        #    # Text              # Value
        #    #('4) Intra Encounter', const.INTRA_ENC_KEY),
        #    ('5) Vs Exemplars',    const.VS_EXEMPLARS_KEY),
        #]

        #ibswgt.species_button = _NEWBUT('Update Encounter Species',
        #                                ibswgt.back.encounter_set_species,
        #                                bgcolor=(100, 255, 150))
        #ibswgt.querydb_combo = _COMBO(detection_combo_box_options,
        #                              ibswgt.back.change_query_mode)

    def _init_layout(ibswgt):
        """ Lays out the defined components """
        # Add elements to the layout
        ibswgt.vlayout.addWidget(ibswgt.enc_tabwgt)
        ibswgt.vlayout.addWidget(ibswgt.vsplitter)
        ibswgt.vsplitter.addWidget(ibswgt.hsplitter)
        ibswgt.vsplitter.addWidget(ibswgt.status_wgt)
        # Horizontal Upper
        ibswgt.hsplitter.addWidget(ibswgt.views[ENCOUNTER_TABLE])
        ibswgt.hsplitter.addWidget(ibswgt._tab_table_wgt)
        # Horizontal Lower
        ibswgt.status_wgt.addWidget(ibswgt.outputLog)
        ibswgt.status_wgt.addWidget(ibswgt.progressBar)
        # Add control widgets (import, group, species selector, etc...)
        ibswgt.control_layout_list = []
        for control_widgets in ibswgt.control_widget_lists:
            ibswgt.control_layout_list.append(QtGui.QHBoxLayout(ibswgt))
            ibswgt.status_wgt.addLayout(ibswgt.control_layout_list[-1])
            for widget in control_widgets:
                ibswgt.control_layout_list[-1].addWidget(widget)
        # Add selected ids status widget
        ibswgt.selectionStatusLayout = QtGui.QHBoxLayout(ibswgt)
        ibswgt.status_wgt.addLayout(ibswgt.selectionStatusLayout)
        for widget in ibswgt.status_widget_list:
            ibswgt.selectionStatusLayout.addWidget(widget)

    def _init_redirects(ibswgt):
        """
        redirects allows user to go from a row of a table to corresponding rows
        of other tables
        """
        redirects = gh.get_redirects(ibswgt.ibs)
        for src_table in redirects.keys():
            for src_table_name in redirects[src_table].keys():
                dst_table, mapping_func = redirects[src_table][src_table_name]
                src_table_col = gh.TABLE_COLNAMES[src_table].index(src_table_name)
                ibswgt.register_redirect(src_table, src_table_col, dst_table, mapping_func)

    def set_status_text(ibswgt, key, text):
        #printDBG('set_status_text[%r] = %r' % (index, text))
        key_to_index = {
            ENCOUNTER_TABLE: 1,
            IMAGE_TABLE: 3,
            IMAGE_GRID: 3,
            ANNOTATION_TABLE: 5,
            NAMES_TREE: 7,
            NAME_TABLE: 7,
        }
        index = key_to_index[key]
        ibswgt.status_widget_list[index].setText(text)

    def changing_models_gen(ibswgt, tblnames=None):
        """
        Loops over tablenames emitting layoutChanged at the end for each
        """
        if tblnames is None:
            tblnames = ibswgt.super_tblname_list
        print('[newgui] changing_models_gen(tblnames=%r)' % (tblnames,))
        model_list = [ibswgt.models[tblname] for tblname in tblnames]
        #model_list = [ibswgt.models[tblname] for tblname in tblnames if ibswgt.views[tblname].isVisible()]
        with ChangeLayoutContext(model_list):
            for tblname in tblnames:
                yield tblname

    def update_tables(ibswgt, tblnames=None, clear_view_selection=True):
        """ forces changing models """
        print('[newgui] update_tables(%r)' % (tblnames,))
        hack_selections = []
        #print('[new_gui.UPDATE_TABLES]')
        for tblname in ibswgt.changing_models_gen(tblnames=tblnames):
            #print('[new_gui.update_tables] tblname=%r' % (tblname, ))
            model = ibswgt.models[tblname]
            view  = ibswgt.views[tblname]
            #if clear_view_selection:
            selection_model = view.selectionModel()
            selection_model.clearSelection()
            hack_selections.append(selection_model)
            model._update()
        # Hack: Call this outside changing models gen
        for selection_model in hack_selections:
            selection_model.clearSelection()

    @profile
    def connect_ibeis_control(ibswgt, ibs):
        """ Connects a new ibscontroler to the models """
        print('[newgui] connecting ibs control. ibs=%r' % (ibs,))
        if ibs is None:
            print('[newgui] invalid ibs')
            title = 'No Database Opened'
            ibswgt.setWindowTitle(title)
        else:
            print('[newgui] Connecting valid ibs=%r' % ibs.get_dbname())
            # Give the frontend the new control
            ibswgt.ibs = ibs
            with ut.Timer('update special'):
                ibs.update_special_encounters()
            # Update the api models to use the new control
            with ut.Timer('make headers'):
                header_dict = gh.make_ibeis_headers_dict(ibswgt.ibs)
            # Enable the redirections between tables
            ibswgt._init_redirects()
            title = ibsfuncs.get_title(ibswgt.ibs)
            ibswgt.setWindowTitle(title)
            if ut.VERBOSE:
                print('[newgui] Calling model _update_headers')
            block_wgt_flag = ibswgt._tab_table_wgt.blockSignals(True)
            with ut.Timer('[newgui] update models'):
                for tblname in ibswgt.changing_models_gen(ibswgt.super_tblname_list):
                    model = ibswgt.models[tblname]
                    view = ibswgt.views[tblname]
                    #if not view.isVisible():
                    #    print(view)
                    #ut.embed()
                    header = header_dict[tblname]
                    #widget = ibswgt.widgets[tblname]
                    #widget.change_headers(header)
                    #
                    # NOT SURE IF THESE BLOCKERS SHOULD BE COMMENTED
                    block_model_flag = model.blockSignals(True)
                    model._update_headers(**header)
                    view._update_headers(**header)  # should use model headers
                    model.blockSignals(block_model_flag)
                    #
                    #view.infer_delegates_from_model()
                for tblname in ibswgt.super_tblname_list:
                    view = ibswgt.views[tblname]
                    #if not view.isVisible():
                    #    print(view)
                    #    continue
                    view.hide_cols()
            ibswgt._tab_table_wgt.blockSignals(block_wgt_flag)
            ibswgt._change_enc(-1)

            LOAD_ENCOUNTER_ON_START = True
            if LOAD_ENCOUNTER_ON_START:
                eid_list = ibs.get_valid_eids(shipped=False)
                if len(eid_list) > 0:
                    DEFAULT_LARGEST_ENCOUNTER = False
                    if DEFAULT_LARGEST_ENCOUNTER:
                        numImg_list = ibs.get_encounter_num_gids(eid_list)
                        argx = ut.list_argsort(numImg_list)[-1]
                        eid = eid_list[argx]
                    else:  # Grab "first" encounter
                        eid = eid_list[0]
                    ibswgt.select_encounter_tab(eid)
                    #ibswgt._change_enc(eid)

    def setWindowTitle(ibswgt, title):
        parent_ = ibswgt.parent()
        if parent_ is not None:
            parent_.setWindowTitle(title)
        else:
            IBEIS_WIDGET_BASE.setWindowTitle(ibswgt, title)

    @profile
    def _change_enc(ibswgt, eid):
        print('[newgui] _change_enc(%r)' % eid)
        for tblname in ibswgt.changing_models_gen(tblnames=ibswgt.tblname_list):
            ibswgt.views[tblname]._change_enc(eid)
            #ibswgt.models[tblname]._change_enc(eid)  # the view should take care of this call
        try:
            #if eid is None:
            #    # HACK
            #    enctext = const.ALL_IMAGE_ENCTEXT
            #else:
            #    enctext = ibswgt.ibs.get_encounter_enctext(eid)
            ibswgt.back.select_eid(eid)
            ibswgt.species_combo.setDefault(ibswgt.ibs.cfg.detect_cfg.species_text)
            #text_list = [
            #    'Identify Mode: Within-Encounter (%s vs. %s)' % (enctext, enctext),
            #    'Identify Mode: Exemplars (%s vs. %s)' % (enctext, const.EXEMPLAR_ENCTEXT)]
            #text_list = [
            #    'Identify Mode: Within-Encounter' ,
            #    'Identify Mode: Exemplars']
            #query_text =
            #ibswgt.query_button
            #ibswgt.querydb_combo.setOptionText(text_list)
            #ibswgt.query_
            #ibswgt.control_widget_lists[1][0].setText('Identify (intra-encounter)\nQUERY(%r vs. %r)' % (enctext, enctext))
            #ibswgt.control_widget_lists[1][1].setText('Identify (vs exemplar database)\nQUERY(%r vs. %r)' % (enctext, const.EXEMPLAR_ENCTEXT))
        except Exception as ex:
            ut.printex(ex, iswarning=True)
        ibswgt.set_table_tab(IMAGE_TABLE)

    def _update_enc_tab_name(ibswgt, eid, enctext):
        ibswgt.enc_tabwgt._update_enc_tab_name(eid, enctext)

    #@checks_qt_error
    def _connect_signals_and_slots(ibswgt):
        #ibswgt._tab_table_wgt.
        print('[newgui] _connect_signals_and_slots')
        for tblname in ibswgt.super_tblname_list:
            tblview = ibswgt.views[tblname]
            tblview.doubleClicked.connect(ibswgt.on_doubleclick)
            tblview.clicked.connect(ibswgt.on_click)
            tblview.contextMenuClicked.connect(ibswgt.on_contextMenuClicked)
            # CONNECT HOOK TO GET NUM ROWS
            tblview.rows_updated.connect(ibswgt.on_rows_updated)
            #front.printSignal.connect(back.backend_print)
            #front.raiseExceptionSignal.connect(back.backend_exception)

    #------------
    # SLOTS
    #------------

    def get_table_tab_index(ibswgt, tblname):
        view = ibswgt.views[tblname]
        index = ibswgt._tab_table_wgt.indexOf(view)
        return index

    def set_table_tab(ibswgt, tblname):
        print('[newgui] set_table_tab: %r ' % (tblname,))
        index = ibswgt.get_table_tab_index(tblname)
        ibswgt._tab_table_wgt.setCurrentIndex(index)

    def select_encounter_tab(ibswgt, eid):
        if False:
            prefix = ut.get_caller_name(range(0, 10))
            prefix = prefix.replace('[wrp_noexectb]', 'w')
            prefix = prefix.replace('[slot_wrapper]', 's')
            prefix = prefix.replace('[X]', 'x')
        else:
            prefix = ''
        print(prefix + '[newgui] select_encounter_tab eid=%r' % (eid,))
        enctext = ibswgt.ibs.get_encounter_enctext(eid)
        ibswgt.enc_tabwgt._add_enc_tab(eid, enctext)
        #ibswgt.back.select_eid(eid)

    @slot_(str, int)
    def on_rows_updated(ibswgt, tblname, nRows):
        """
        When the rows are updated change the tab names
        """
        if VERBOSE_GUI:
            print('[newgui] on_rows_updated: tblname=%12r nRows=%r ' % (tblname, nRows))
        #printDBG('Rows updated in tblname=%r, nRows=%r' % (str(tblname), nRows))
        if tblname == ENCOUNTER_TABLE:  # Hack
            print('... tblname == ENCOUNTER_TABLE, ...hack return')
            return
        tblname = str(tblname)
        tblnice = gh.TABLE_NICE[tblname]
        index = ibswgt.get_table_tab_index(tblname)
        text = tblnice + ' ' + str(nRows)
        #printDBG('Rows updated in index=%r, text=%r' % (index, text))
        # CHANGE TAB NAME TO SHOW NUMBER OF ROWS
        ibswgt._tab_table_wgt.setTabText(index, text)

    @slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuClicked(ibswgt, qtindex, pos):
        """
        Right click anywhere in the GUI
        Context menus on right click of a table
        """
        def _goto_image_image(gid):
            ibswgt.set_table_tab(IMAGE_TABLE)
            imgtbl = ibswgt.views[IMAGE_TABLE]
            imgtbl.select_row_from_id(gid, scroll=True)

        def _goto_annot_image(aid):
            ibswgt.set_table_tab(IMAGE_TABLE)
            imgtbl = ibswgt.views[IMAGE_TABLE]
            gid = ibswgt.back.ibs.get_annot_gids(aid)
            imgtbl.select_row_from_id(gid, scroll=True)

        def _goto_annot_annot(aid):
            ibswgt.set_table_tab(ANNOTATION_TABLE)
            imgtbl = ibswgt.views[ANNOTATION_TABLE]
            imgtbl.select_row_from_id(aid, scroll=True)

        def _goto_annot_name(aid):
            ibswgt.set_table_tab(NAMES_TREE)
            nametree = ibswgt.views[NAMES_TREE]
            nid = ibswgt.back.ibs.get_annot_nids(aid)
            nametree.select_row_from_id(nid, scroll=True)

        #printDBG('[newgui] contextmenu')
        model = qtindex.model()
        tblview = ibswgt.views[model.name]
        context_options = []
        id_list = sorted(list(set(
            [model._get_row_id(_qtindex) for _qtindex in tblview.selectedIndexes()]
        )))
        # ---- ENCOUNTER CONTEXT ----
        if model.name == ENCOUNTER_TABLE:
            merge_destination_id = model._get_row_id(qtindex)  # This is for the benefit of merge encounters
            enctext = ibswgt.back.ibs.get_encounter_enctext(merge_destination_id)
            # Conditional context menu
            if len(id_list) == 1:
                context_options += [
                    ('Run detection on encounter (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_encounter(id_list)),
                    ('Merge %d encounter into %s' %  (len(id_list), (enctext)),
                        lambda: ibswgt.back.merge_encounters(id_list, merge_destination_id)),
                    ('----', lambda: None),
                    ('Delete encounter', lambda: ibswgt.back.delete_encounter(id_list)),
                    ('Delete encounter (and images)', lambda: ibswgt.back.delete_encounter_and_images(id_list)),
                    ('Export encounter', lambda: ibswgt.back.export_encounters(id_list)),
                ]
            else:
                context_options += [
                    ('Run detection on encounters (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_encounter(id_list)),
                    ('Merge %d encounters into %s' %  (len(id_list), (enctext)),
                        lambda: ibswgt.back.merge_encounters(id_list, merge_destination_id)),
                    ('----', lambda: None),
                    ('Delete encounters', lambda: ibswgt.back.delete_encounter(id_list)),
                    ('Delete encounters (and images)', lambda: ibswgt.back.delete_encounter_and_images(id_list)),
                    # ('export encounters', lambda: ibswgt.back.export_encounters(id_list)),
                ]
        # ---- IMAGE CONTEXT ----
        elif model.name == IMAGE_TABLE:
            current_enctext = ibswgt.back.ibs.get_encounter_enctext(ibswgt.back.get_selected_eid())
            # Conditional context menu
            if len(id_list) == 1:
                gid = id_list[0]
                eid = model.eid
                context_options += [
                    ('View image',
                        lambda: ibswgt.back.select_gid(gid, eid, show=True)),
                    ('View detection image (Hough) [dev]',
                        lambda: ibswgt.back.show_hough_image(gid)),
                    ('Add annotation from entire image',
                        lambda: ibswgt.back.add_annotation_from_image([gid])),
                    ('Run detection on image (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_images([gid])),
                ]
            else:
                context_options += [
                    ('Add annotation from entire images',
                        lambda: ibswgt.back.add_annotation_from_image(id_list)),
                    ('Run detection on images (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_images(id_list)),
                ]
            # Special condition for encounters
            if current_enctext != const.NEW_ENCOUNTER_ENCTEXT:
                context_options += [
                    ('----', lambda: None),
                    ('Move to new encounter',
                        lambda: ibswgt.back.send_to_new_encounter(id_list, mode='move')),
                    ('Copy to new encounter',
                        lambda: ibswgt.back.send_to_new_encounter(id_list, mode='copy')),
                ]
            if current_enctext != const.UNGROUPED_IMAGES_ENCTEXT:
                context_options += [
                    ('----', lambda: None),
                    ('Remove from encounter',
                        lambda: ibswgt.back.remove_from_encounter(id_list)),
                ]
            # Continue the conditional context menu
            if len(id_list) == 1:
                # We get gid from above
                context_options += [
                    ('----', lambda: None),
                    ('Delete image\'s annotations',
                        lambda: ibswgt.back.delete_image_annotations([gid])),
                    ('Delete image',
                        lambda: ibswgt.back.delete_image(gid)),
                ]
            else:
                context_options += [
                    ('----', lambda: None),
                    ('Delete images\' annotations',
                        lambda: ibswgt.back.delete_image_annotations(id_list)),
                    ('Delete images',
                        lambda: ibswgt.back.delete_image(id_list)),
                ]
        # ---- IMAGE GRID CONTEXT ----
        elif model.name == IMAGE_GRID:
            current_enctext = ibswgt.back.ibs.get_encounter_enctext(ibswgt.back.get_selected_eid())
            # Conditional context menu
            if len(id_list) == 1:
                gid = id_list[0]
                eid = model.eid
                context_options += [
                    ('View image',
                        lambda: ibswgt.back.select_gid(gid, eid, show=True)),
                    ('View detection image (Hough) [dev]',
                        lambda: ibswgt.back.show_hough_image(gid)),
                    ('Add annotation from entire image',
                        lambda: ibswgt.back.add_annotation_from_image([gid])),
                    ('Run detection on image (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_images([gid])),
                    ('----', lambda: None),
                    ('Go to image in Images Table',
                        lambda: _goto_image_image(gid)),
                ]
            # Special condition for encounters
            if current_enctext != const.NEW_ENCOUNTER_ENCTEXT:
                context_options += [
                    ('----', lambda: None),
                    ('Move to new encounter',
                        lambda: ibswgt.back.send_to_new_encounter(id_list, mode='move')),
                    ('Copy to new encounter',
                        lambda: ibswgt.back.send_to_new_encounter(id_list, mode='copy')),
                ]
            if current_enctext != const.UNGROUPED_IMAGES_ENCTEXT:
                context_options += [
                    ('----', lambda: None),
                    ('Remove from encounter',
                        lambda: ibswgt.back.remove_from_encounter(id_list)),
                ]
            # Continue the conditional context menu
            if len(id_list) == 1:
                # We get gid from above
                context_options += [
                    ('----', lambda: None),
                    ('Delete image\'s annotations',
                        lambda: ibswgt.back.delete_image_annotations([gid])),
                    ('Delete image',
                        lambda: ibswgt.back.delete_image(gid)),
                ]
        # ---- ANNOTATION CONTEXT ----
        elif model.name == ANNOTATION_TABLE:
            # Conditional context menu
            if len(id_list) == 1:
                aid = id_list[0]
                eid = model.eid
                context_options += [
                    ('Edit Annotation in Image',
                        lambda: ibswgt.spawn_edit_image_annotation_interaction_from_aid(aid, eid)),
                    ('----', lambda: None),
                    ('View annotation',
                        lambda: ibswgt.back.select_aid(aid, eid, show=True)),
                    ('View image',
                        lambda: ibswgt.back.select_gid_from_aid(aid, eid, show=True)),
                    ('View detection chip (probability) [dev]',
                        lambda: ibswgt.back.show_probability_chip(aid)),
                    ('----', lambda: None),
                    ('Go to image',
                        lambda: _goto_annot_image(aid)),
                    ('Go to name',
                        lambda: _goto_annot_name(aid)),
                    ('----', lambda: None),
                    ('Unset annotation\'s name',
                        lambda: ibswgt.back.unset_names([aid])),
                    ('Delete annotation',
                        lambda: ibswgt.back.delete_annot(id_list)),
                ]
            else:
                context_options += [
                    ('Unset annotations\' names', lambda: ibswgt.back.unset_names(id_list)),
                    ('Delete annotations', lambda: ibswgt.back.delete_annot(id_list)),
                ]
        # ---- NAMES TREE CONTEXT ----
        elif model.name == NAMES_TREE:
            if len(id_list) == 1:
                aid = id_list[0]
                eid = model.eid
                context_options += [
                    ('View annotation', lambda: ibswgt.back.select_aid(aid, eid, show=True)),
                    ('View image', lambda: ibswgt.back.select_gid_from_aid(aid, eid, show=True)),
                    ('----', lambda: None),
                    ('Go to image', lambda: _goto_annot_image(aid)),
                    ('Go to annotation', lambda: _goto_annot_annot(aid)),
                ]
        # Show the context menu
        if len(context_options) > 0:
            guitool.popup_menu(tblview, pos, context_options)

    def register_redirect(ibswgt, src_table, src_table_col, dst_table, mapping_func):
        if src_table not in ibswgt.redirects.keys():
            ibswgt.redirects[src_table] = {}
        ibswgt.redirects[src_table][src_table_col] = (dst_table, mapping_func)

    @slot_(QtCore.QModelIndex)
    def on_click(ibswgt, qtindex):
        """
        Clicking anywhere in the GUI
        """
        #printDBG('on_click')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex)
        #model_name = model.name
        #print('clicked: %s' + ut.dict_str(locals()))
        try:
            dst_table, mapping_func = ibswgt.redirects[model.name][qtindex.column()]
            dst_id = mapping_func(id_)
            print("[on_click] Redirecting to: %r" % (dst_table, ))
            print("[on_click]     Mapping %r -> %r" % (id_, dst_id, ))
            ibswgt.set_table_tab(dst_table)
            ibswgt.views[dst_table].select_row_from_id(id_, scroll=True)
            return None
        except Exception as ex:
            if ut.VERYVERBOSE:
                ut.printex(ex, 'no redirect listed for this table', iswarning=True)
            # No redirect listed for this table
            pass

        # If no link, process normally
        if model.name == ENCOUNTER_TABLE:
            pass
            #printDBG('clicked encounter')
        else:
            table_key = model.name
            # FIXME: stripe model needs to forward get_level
            if not hasattr(model, '_get_level'):
                level = 0
            else:
                level = model._get_level(qtindex)
            eid = model.eid
            ibswgt.select_table_id(table_key, level, id_, eid)

    def select_table_id(ibswgt, table_key, level, id_, eid):
        select_func_dict = {
            (IMAGE_TABLE, 0)      : ibswgt.back.select_gid,
            (IMAGE_GRID, 0)       : ibswgt.back.select_gid,
            (ANNOTATION_TABLE, 0) : ibswgt.back.select_aid,
            (NAME_TABLE, 0)       : ibswgt.back.select_nid,
            (NAMES_TREE, 0)       : ibswgt.back.select_nid,
            (NAME_TABLE, 1)       : ibswgt.back.select_aid,
            (NAMES_TREE, 1)       : ibswgt.back.select_aid,
        }
        select_func = select_func_dict[(table_key, level)]
        select_func(id_, eid, show=False)

    @slot_(QtCore.QModelIndex)
    def on_doubleclick(ibswgt, qtindex):
        """
        Double clicking anywhere in the GUI
        """
        print('\n+--- DOUBLE CLICK ---')
        #printDBG('on_doubleclick')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex)
        if model.name == ENCOUNTER_TABLE:
            eid = id_
            ibswgt.select_encounter_tab(eid)
        else:
            eid = model.eid
            if (model.name == IMAGE_TABLE) or (model.name == IMAGE_GRID):
                gid = id_
                ibswgt.spawn_edit_image_annotation_interaction(model, qtindex, gid, eid)
            elif model.name == ANNOTATION_TABLE:
                aid = id_
                ibswgt.back.select_aid(aid, eid)
            elif model.name == NAME_TABLE:
                nid = id_
                ibswgt.back.select_nid(nid, eid)
            elif model.name == NAMES_TREE:
                level = model._get_level(qtindex)
                if level == 0:
                    nid = id_
                    ibswgt.back.select_nid(nid, eid, show=True)
                elif level == 1:
                    aid = id_
                    ibswgt.back.select_aid(aid, eid, show=True)

    def spawn_edit_image_annotation_interaction_from_aid(ibswgt, aid, eid):
        """
        hack for letting annots spawn image editing

        CommandLine:
            python -m ibeis.gui.newgui --test-spawn_edit_image_annotation_interaction_from_aid --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.gui.newgui import *  # NOQA
            >>> import ibeis
            >>> main_locals = ibeis.main(defaultdb='testdb1')
            >>> ibs, back = ut.dict_take(main_locals, ['ibs', 'back'])
            >>> ibswgt = back.ibswgt  # NOQA
            >>> aid = 4
            >>> eid = 1
            >>> ibswgt.spawn_edit_image_annotation_interaction_from_aid(aid, eid)
            >>> if ut.show_was_requested():
            >>>    guitool.qtapp_loop(qwin=ibswgt)
        """
        gid = ibswgt.back.ibs.get_annot_gids(aid)
        view = ibswgt.views[IMAGE_TABLE]
        model = view.model()
        qtindex, row = view.get_row_and_qtindex_from_id(gid)
        ibswgt.spawn_edit_image_annotation_interaction(model, qtindex, gid, eid)

    def spawn_edit_image_annotation_interaction(ibswgt, model, qtindex, gid, eid):
        """
        TODO: needs reimplement using more standard interaction methods

        """
        print('[newgui] Creating new annotation interaction: gid=%r' % (gid,))
        ibs = ibswgt.ibs
        # Select gid
        ibswgt.back.select_gid(gid, eid, show=False)
        # Interact with gid
        nextcb, prevcb, current_gid = ibswgt._interactannot2_callbacks(model, qtindex)
        iannot2_kw = {
            'rows_updated_callback': ibswgt.update_tables,
            'next_callback': nextcb,
            'prev_callback': prevcb,
        }
        assert current_gid == gid, 'problem in next/prev updater'
        ibswgt.annot_interact = interact_annotations2.ANNOTATION_Interaction2(ibs, gid, **iannot2_kw)
        # hacky GID_PROG: TODO: FIX WITH OTHER HACKS OF THIS TYPE
        _, row = model.view.get_row_and_qtindex_from_id(gid)
        pt.set_figtitle('%d/%d' % (row + 1, model.rowCount()))

    def make_adjacent_qtindex_callbacks(ibswgt, model, qtindex):
        r"""
        Returns:
            tuple: (current_rowid, next_callback, prev_callback)

        CommandLine:
            python -m ibeis.gui.newgui --test-make_adjacent_qtindex_callbacks

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.gui.newgui import *  # NOQA
            >>> # build test data
            >>> ibs, back, ibswgt, testdata_main_loop = testdata_guifront()
            >>> gid = ibs.get_valid_gids()[0]
            >>> model = ibswgt.models[gh.IMAGE_TABLE]
            >>> qtindex, row = model.get_row_and_qtindex_from_id(gid)
            >>> # execute function
            >>> (current_rowid, next_callback, prev_callback) = ibswgt.make_adjacent_qtindex_callbacks(model, qtindex)
            >>> assert prev_callback is None, 'should not be a previous image id'
            >>> current_rowid1, next_callback1, prev_callback1 = next_callback()
            >>> assert next_callback() is None, 'race condition not prevented'
            >>> current_rowid2, next_callback2, prev_callback2 = next_callback1()
            >>> # testdata main loop func
            >>> testdata_main_loop(globals(), locals())
        """
        current_rowid = model._get_row_id(qtindex)
        next_qtindex = model._get_adjacent_qtindex(qtindex, 1)
        prev_qtindex = model._get_adjacent_qtindex(qtindex, -1)
        next_callback = None
        prev_callback = None
        numclicks = [0]  # semephore, invalidates both functions after one call
        if next_qtindex is not None and next_qtindex.isValid():
            def next_callback():
                if numclicks[0] != 0:
                    print('race condition in next_callback %d ' % numclicks[0])
                    return
                numclicks[0] += 1
                return ibswgt.make_adjacent_qtindex_callbacks(model, next_qtindex)
        if prev_qtindex is not None and prev_qtindex.isValid():
            def prev_callback():
                if numclicks[0] != 0:
                    print('race condition in next_callback %d ' % numclicks[0])
                    return
                numclicks[0] += 1
                return ibswgt.make_adjacent_qtindex_callbacks(model, next_qtindex)
        return current_rowid, next_callback, prev_callback

    def _interactannot2_callbacks(ibswgt, model, qtindex):
        """
        callbacks for the edit image annotation (from image table) interaction

        TODO: needs reimplement
        """
        #if not qtindex.isValid():
        #    raise AssertionError('Bug: qtindex got invalidated')
        #    # BUG: somewhere qtindex gets invalidated
        #    #return None, None, -1
        # HACK FOR NEXT AND PREVIOUS CLICK CALLBACKS
        cur_gid = model._get_row_id(qtindex)
        next_qtindex = model._get_adjacent_qtindex(qtindex, 1)
        prev_qtindex = model._get_adjacent_qtindex(qtindex, -1)
        next_callback = None
        prev_callback = None
        numclicks = [0]  # semephore
        if next_qtindex is not None and next_qtindex.isValid():
            def next_callback():
                if numclicks[0] != 0:
                    print('race condition in next_callback %d ' % numclicks[0])
                    return
                numclicks[0] += 1
                # call this function again with next index
                nextcb, prevcb, new_gid1 = ibswgt._interactannot2_callbacks(model, next_qtindex)
                print('[newgui] next_callback: new_gid1=%r' % (new_gid1))
                ibswgt.annot_interact.update_image_and_callbacks(new_gid1, nextcb, prevcb, do_save=True)
                # hacky GID_PROG: TODO: FIX WITH OTHER HACKS OF THIS TYPE
                _, row = model.view.get_row_and_qtindex_from_id(new_gid1)
                pt.set_figtitle('%d/%d' % (row + 1, model.rowCount()))
        if prev_qtindex is not None and prev_qtindex.isValid():
            def prev_callback():
                if numclicks[0] != 0:
                    print('race condition in prev_callback %d ' % numclicks[0])
                    return
                numclicks[0] += 1
                # call this function again with previous index
                nextcb, prevcb, new_gid2 = ibswgt._interactannot2_callbacks(model, prev_qtindex)
                print('[newgui] prev_callback: new_gid2=%r' % (new_gid2))
                ibswgt.annot_interact.update_image_and_callbacks(new_gid2, nextcb, prevcb, do_save=True)
                # hacky GID_PROG: TODO: FIX WITH OTHER HACKS OF THIS TYPE
                _, row = model.view.get_row_and_qtindex_from_id(new_gid2)
                pt.set_figtitle('%d/%d' % (row + 1, model.rowCount()))
        return next_callback, prev_callback, cur_gid

    @slot_(list)
    def imagesDropped(ibswgt, url_list):
        """
        image drag and drop event
        """
        print('[drop_event] url_list=%r' % (url_list,))
        gpath_list = list(filter(ut.matches_image, url_list))
        dir_list   = list(filter(isdir, url_list))
        if len(dir_list) > 0:
            options = ['No', 'Yes']
            title   = 'Non-Images dropped'
            msg     = 'Recursively import from directories?'
            ans = guitool.user_option(ibswgt, msg=msg, title=title,
                                      options=options)
            if ans == 'Yes':
                unflat_gpaths = [ut.list_images(dir_, fullpath=True, recursive=True)
                                 for dir_ in dir_list]
                flat_gpaths = ut.flatten(unflat_gpaths)
                flat_unix_gpaths = list(map(ut.unixpath, flat_gpaths))
                gpath_list.extend(flat_unix_gpaths)
            else:
                return
        print('[drop_event] gpath_list=%r' % (gpath_list,))
        if len(gpath_list) > 0:
            ibswgt.back.import_images_from_file(gpath_list=gpath_list)


def testdata_guifront():
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1')
    ibs, back = ut.dict_take(main_locals, ['ibs', 'back'])
    ibswgt = back.ibswgt  # NOQA
    globals__ = globals()
    locals__  = locals()
    def testdata_main_loop(globals_=globals__, locals_=locals__):
        locals_  = locals_.copy()
        globals_ = globals_.copy()
        locals_.update(locals__)
        globals_.update(globals__)
        if '--cmd' in sys.argv:
            guitool.qtapp_loop(qwin=ibswgt, ipy=True)
            six.exec_(ut.ipython_execstr(), globals_, locals_)
        elif ut.show_was_requested():
            guitool.qtapp_loop(qwin=ibswgt)
    return ibs, back, ibswgt, testdata_main_loop


def testfunc():
    r"""
    CommandLine:
        python -m ibeis.gui.newgui --test-testfunc --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.newgui import *  # NOQA
        >>> result = testfunc()
        >>> # verify results
        >>> print(result)
    """
    ibs, back, ibswgt, testdata_main_loop = testdata_guifront()
    testdata_main_loop(globals(), locals())


#if __name__ == '__main__':
#    """
#    CommandLine:
#        python -m ibeis.gui.newgui
#        python -m ibeis.gui.newgui --allexamples
#        python -m ibeis.gui.newgui --allexamples --noface --nosrc
#    """
#    testfunc()

#    #import ibeis
#    #main_locals = ibeis.main(defaultdb='testdb1')
#    #ibs, back = ut.dict_take(main_locals, ['ibs', 'back'])
#    #ibswgt = back.ibswgt

#    ##ibswgt = IBEISGuiWidget(back=back, ibs=ibs)

#    #if '--cmd' in sys.argv:
#    #    guitool.qtapp_loop(qwin=ibswgt, ipy=True)
#    #    exec(ut.ipython_execstr())
#    #else:
#    #    guitool.qtapp_loop(qwin=ibswgt)
if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.gui.newgui
        python -m ibeis.gui.newgui --allexamples
        python -m ibeis.gui.newgui --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
