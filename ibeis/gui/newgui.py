#!/usr/bin/env python2.7
"""
This should probably be renamed guifront.py This defines all of the visual
components to the GUI It is invoked from guiback, which handles the nonvisual
logic.


BUGS:
    * Copying the ungrouped encounter raises an error. Should have the option
    to copy or move it. Other special encounter should not have this option.

    Should gray out an option if it is not available.


"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, map, filter  # NOQA
from os.path import isdir
import sys
from ibeis import constants as const
from ibeis import species
import functools
from guitool.__PYQT__ import QtGui, QtCore
from guitool.__PYQT__.QtCore import Qt
from guitool.__PYQT__.QtGui import QSizePolicy
from guitool import signal_, slot_, checks_qt_error, ChangeLayoutContext, BlockContext  # NOQA
from ibeis import ibsfuncs
from ibeis.gui import guiheaders as gh
from ibeis.gui import guimenus
import six
from ibeis.viz.interact import interact_annotations2
from ibeis.gui.guiheaders import (IMAGE_TABLE, IMAGE_GRID, ANNOTATION_TABLE, NAME_TABLE, NAMES_TREE, ENCOUNTER_TABLE)  # NOQA
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

VERBOSE_GUI = ut.VERBOSE or ut.get_argflag(('--verbose-gui', '--verbgui'))
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
    """
    Holds the table-tabs

    use setCurrentIndex to change the selection
    """
    def __init__(tabwgt, parent=None, horizontalStretch=1):
        QtGui.QTabWidget.__init__(tabwgt, parent)
        tabwgt.ibswgt = parent
        tabwgt._sizePolicy = guitool.newSizePolicy(tabwgt, horizontalStretch=horizontalStretch)
        tabwgt.setSizePolicy(tabwgt._sizePolicy)
        #tabwgt.currentChanged.connect(tabwgt.setCurrentIndex)
        tabwgt.currentChanged.connect(tabwgt._on_tabletab_change)
        tabwgt.current_tblname = None

    @slot_(int)
    def _on_tabletab_change(tabwgt, index):
        """ Switch to the current encounter tab """
        print('[apitab] _onchange(index=%r)' % (index,))
        tblname = tabwgt.ibswgt.tblname_list[index]
        tabwgt.current_tblname = tblname
        print('[apitab] _onchange(tblname=%r)' % (tblname,))
        tabwgt.ibswgt.back._clear_selection()
        view = tabwgt.ibswgt.views[tblname]
        selected = view.selectionModel().selection()
        deselected = QtGui.QItemSelection()
        tabwgt.ibswgt.update_selection(selected, deselected)
        #tabwgt.ibswgt.back.update_selection_texts()

    #def setCurrentIndex(tabwgt, index):
    #    tblname = tabwgt.ibswgt.tblname_list[index]
    #    print('Set %r current Index: %r ' % (tblname, index))
    #    #model = tabwgt.ibswgt.models[tblname]
    #    #with ChangeLayoutContext([model]):
    #    #    QtGui.QTabWidget.setCurrentIndex(tabwgt, index)


class EncoutnerTabWidget(QtGui.QTabWidget):
    """
    Handles the super-tabs for the encounters that hold the table-tabs
    """
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
        enc_tabwgt.currentChanged.connect(enc_tabwgt._on_enctab_change)

        enc_tabwgt.eid_list = []
        # TURNING ON / OFF ALL IMAGES
        # enc_tabwgt._add_enc_tab(-1, const.ALL_IMAGE_ENCTEXT)

    @slot_(int)
    def _on_enctab_change(enc_tabwgt, index):
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
        #with ut.Indenter('[_ADD_ENC_TAB]'):
        print('[_add_enc_tab] eid=%r, enctext=%r' % (eid, enctext))
        if eid not in enc_tabwgt.eid_list:
            # tab_name = str(eid) + ' - ' + str(enctext)
            tab_name = str(enctext)
            enc_tabwgt.addTab(QtGui.QWidget(), tab_name)

            enc_tabwgt.eid_list.append(eid)
            index = len(enc_tabwgt.eid_list) - 1
        else:
            index = enc_tabwgt.eid_list.index(eid)

        #with BlockContext(enc_tabwgt):
        #print('SET CURRENT INDEX')
        enc_tabwgt.setCurrentIndex(index)
        #print('DONE SETTING CURRENT INDEX')
        enc_tabwgt._on_enctab_change(index)

    def _update_enc_tab_name(enc_tabwgt, eid, enctext):
        for index, _id in enumerate(enc_tabwgt.eid_list):
            if eid == _id:
                enc_tabwgt.setTabText(index, enctext)


#############################
######## Main Widget ########
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


#############################
##### IBEIS GUI Widget ######
#############################


class IBEISGuiWidget(IBEIS_WIDGET_BASE):
    """
    CommandLine:
        # Testing
        python -m ibies --db NNP_Master3 --onlyimgtbl

    """
    def __init__(ibswgt, back=None, ibs=None, parent=None):
        IBEIS_WIDGET_BASE.__init__(ibswgt, parent)
        ibswgt.ibs = ibs
        ibswgt.back = back
        # Structures that will hold models and views
        ibswgt.models       = {}
        ibswgt.views        = {}
        ibswgt.redirects    = {}

        # FIXME: Duplicate models
        # Create models and views
        # Define the abstract item models and views for the tables
        ibswgt.tblname_list   = []
        ibswgt.modelview_defs = []

        # NEW DYNAMIC WAY OF USING TABS AND API VIEWS
        # ADD IMAGE TABLE
        if True:
            ibswgt.tblname_list.append(IMAGE_TABLE)
            ibswgt.modelview_defs.append((IMAGE_TABLE, IBEISTableWidget, IBEISItemModel, IBEISTableView))
        # ADD IMAGE GRID
        if not ut.get_argflag('--onlyimgtbl'):
            ibswgt.tblname_list.append(IMAGE_GRID)
            ibswgt.modelview_defs.append((IMAGE_GRID, IBEISTableWidget, IBEISStripeModel, IBEISTableView))
        # ADD ANNOT GRID
        if not (ut.get_argflag('--noannottbl') or ut.get_argflag('--onlyimgtbl')):
            ibswgt.tblname_list.append(gh.ANNOTATION_TABLE)
            ibswgt.modelview_defs.append((gh.ANNOTATION_TABLE, IBEISTableWidget, IBEISItemModel, IBEISTableView))
        # ADD NAME TREE
        if not (ut.get_argflag('--nonametree') or ut.get_argflag('--onlyimgtbl')):
            ibswgt.tblname_list.append(NAMES_TREE)
            ibswgt.modelview_defs.append((NAMES_TREE, IBEISTreeWidget, IBEISItemModel, IBEISTreeView))
        # ADD ENCOUNTER TABLE
        ibswgt.super_tblname_list = ibswgt.tblname_list + [ENCOUNTER_TABLE]
        ibswgt.modelview_defs.append((ENCOUNTER_TABLE,  EncTableWidget,  EncTableModel, EncTableView))

        # DO INITALIZATION
        # Create and layout components
        ibswgt._init_components()
        ibswgt._init_layout()
        # Connect signals and slots
        ibswgt._connect_signals_and_slots()
        # Connect the IBEIS control
        ibswgt.connect_ibeis_control(ibswgt.ibs)

    def _connect_signals_and_slots(ibswgt):
        print('[newgui] _connect_signals_and_slots')
        for tblname in ibswgt.super_tblname_list:
            tblview = ibswgt.views[tblname]
            tblview.doubleClicked.connect(ibswgt.on_doubleclick)
            #tblview.clicked.connect(ibswgt.on_click)
            tblview.contextMenuClicked.connect(ibswgt.on_contextMenuClicked)
            if tblname != gh.ENCOUNTER_TABLE:
                tblview.selectionModel().selectionChanged.connect(ibswgt.update_selection)
            #front.printSignal.connect(back.backend_print)
            #front.raiseExceptionSignal.connect(back.backend_exception)
            # CONNECT HOOK TO GET NUM ROWS
            tblview.rows_updated.connect(ibswgt.on_rows_updated)

    @slot_(QtGui.QItemSelection, QtGui.QItemSelection)
    def update_selection(ibswgt, selected, deselected):
        """
        Quirky behavior: if you select two columns in a row and then unselect
        only one, the whole row is unselected, because this function only deals
        with deltas.

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.gui.newgui import *  # NOQA
            >>> ibs, back, ibswgt, testdata_main_loop = testdata_guifront()
            >>> ibswgt.set_table_tab(gh.NAMES_TREE)
            >>> view = ibswgt.views[gh.NAMES_TREE]
            >>> view.expandAll()
            >>> AUTOSELECT = False
            >>> if AUTOSELECT:
            ...     view.selectAll()
            >>> selmodel = view.selectionModel()
            >>> selected = selmodel.selection()
            >>> deselected = QtGui.QItemSelection()
            >>> # verify results
            >>> print(result)

        """
        #print('selected = ' + str(selected.indexes()))
        #print('deselected = ' + str(deselected.indexes()))
        deselected_model_index_list_ = deselected.indexes()
        selected_model_index_list_   = selected.indexes()

        def get_selection_info(model_index_list_):
            model_index_list = [qtindex for qtindex in model_index_list_ if qtindex.isValid()]
            model_list       = [qtindex.model() for qtindex in model_index_list]
            tablename_list   = [model.name for model in model_list]
            level_list       = [model._get_level(qtindex) for model, qtindex in zip(model_list, model_index_list)]
            rowid_list       = [model._get_row_id(qtindex) for model, qtindex in zip(model_list, model_index_list)]
            table_key_list = list(zip(tablename_list, level_list))
            return table_key_list, rowid_list

        select_table_key_list, select_rowid_list = get_selection_info(selected_model_index_list_)
        deselect_table_key_list, deselect_rowid_list = get_selection_info(deselected_model_index_list_)

        table_key2_selected_rowids   = dict(ut.group_items(select_rowid_list, select_table_key_list))
        table_key2_deselected_rowids = dict(ut.group_items(deselect_rowid_list, deselect_table_key_list))

        table_key2_selected_rowids   = {key: list(set(val)) for key, val in six.iteritems(table_key2_selected_rowids)}
        table_key2_deselected_rowids = {key: list(set(val)) for key, val in six.iteritems(table_key2_deselected_rowids)}
        if ut.VERBOSE:
            print('table_key2_selected_rowids = ' + ut.dict_str(table_key2_selected_rowids))
            print('table_key2_deselected_rowids = ' + ut.dict_str(table_key2_deselected_rowids))

        gh_const_tablename_map = {
            (IMAGE_TABLE, 0)         : const.IMAGE_TABLE,
            (IMAGE_GRID, 0)          : const.IMAGE_TABLE,
            (gh.ANNOTATION_TABLE, 0) : const.ANNOTATION_TABLE,
            (NAME_TABLE, 0)          : const.NAME_TABLE,
            (NAMES_TREE, 0)          : const.NAME_TABLE,
            (NAMES_TREE, 1)          : const.ANNOTATION_TABLE,
        }
        # here tablename is a backend const tablename
        for table_key, id_list in six.iteritems(table_key2_deselected_rowids):
            tablename = gh_const_tablename_map[table_key]
            ibswgt.back._set_selection3(tablename, id_list, mode='diff')
        for table_key, id_list in six.iteritems(table_key2_selected_rowids):
            tablename = gh_const_tablename_map[table_key]
            ibswgt.back._set_selection3(tablename, id_list, mode='add')
        ibswgt.back.update_selection_texts()

        #tblview.selectionModel().selectedIndexes()

    def _init_components(ibswgt):
        """ Defines gui components """
        # Layout
        ibswgt.vlayout = QtGui.QVBoxLayout(ibswgt)
        #ibswgt.hsplitter = guitool.newSplitter(ibswgt, Qt.Horizontal, verticalStretch=18)
        ibswgt.hsplitter = guitool.newSplitter(ibswgt, Qt.Horizontal, verticalStretch=18)
        ibswgt.vsplitter = guitool.newSplitter(ibswgt, Qt.Vertical)
        #ibswgt.hsplitter = guitool.newWidget(ibswgt, Qt.Horizontal, verticalStretch=18)
        #ibswgt.vsplitter = guitool.newWidget(ibswgt)
        #
        # Tables Tab
        ibswgt._table_tab_wgt = APITabWidget(ibswgt, horizontalStretch=81)
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
            #ibswgt._table_tab_wgt.addTab(ibswgt.widgets[tblname], tblname)
            ibswgt._table_tab_wgt.addTab(ibswgt.views[tblname], tblname)
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
        _NEWTEXT = functools.partial(guitool.newLineEdit, ibswgt)

        primary_fontkw = dict(bold=True, pointSize=11)
        secondary_fontkw = dict(bold=False, pointSize=9)
        advanced_fontkw = dict(bold=False, pointSize=8, italic=True)
        identify_color = (255, 150, 0)

        ibswgt.tablename_to_status_widget_index = {
            ENCOUNTER_TABLE: 1,
            IMAGE_TABLE: 3,
            IMAGE_GRID: 3,
            gh.ANNOTATION_TABLE: 5,
            NAMES_TREE: 7,
            NAME_TABLE: 7,
        }
        ibswgt.status_widget_list = [
            _NEWLBL('Selected Encounter: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(enabled=True, readOnly=True),
            ##
            _NEWLBL('Selected Image: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(enabled=True, readOnly=False,
                     editingFinishedSlot=ibswgt.select_image_text_editing_finished),
            ##
            _NEWLBL('Selected Annotation: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(enabled=True, readOnly=False,
                     editingFinishedSlot=ibswgt.select_annot_text_editing_finished),
            ##
            _NEWLBL('Selected Name: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(enabled=True, readOnly=False,
                     editingFinishedSlot=ibswgt.select_name_text_editing_finished),
        ]

        back = ibswgt.back

        # TODO: update these options depending on ibs.get_species_with_detectors
        # when a controller is attached to the gui
        detection_combo_box_options = [
            # Text              # Value
            #('Select Species',  'none'),
            ('Select Species',  const.Species.UNKNOWN),
            #'none'),
        ] + species.get_working_species_set()

        ibswgt.species_combo = _COMBO(detection_combo_box_options,
                                      ibswgt.back.change_detection_species,
                                      fontkw=primary_fontkw)

        ibswgt.batch_intra_encounter_query_button = _NEWBUT(
            'Intra Encounter',
            functools.partial(
                back.compute_queries, query_is_known=None,
                daids_mode=const.INTRA_ENC_KEY,
                use_prioritized_name_subset=False,
                cfgdict={'can_match_samename': False, 'use_k_padding': False}),
            bgcolor=color_funcs.adjust_hsv_of_rgb255(identify_color, -0.01, -0.7, 0.0),
            fgcolor=(0, 0, 0), fontkw=advanced_fontkw)

        ibswgt.batch_vsexemplar_query_button = _NEWBUT(
            'Vs Exemplar',
            functools.partial(
                back.compute_queries,
                use_prioritized_name_subset=True,
                query_is_known=None, daids_mode=const.VS_EXEMPLARS_KEY,
                cfgdict={'can_match_samename': False, 'use_k_padding': False},
            ),
            bgcolor=color_funcs.adjust_hsv_of_rgb255(identify_color, 0.01, -0.7, 0.0),
            fgcolor=(0, 0, 0), fontkw=advanced_fontkw)

        ibswgt.import_button = _NEWBUT(
            '1) Import',
            back.import_images_from_dir,
            bgcolor=(235, 200, 200), fontkw=primary_fontkw)

        ibswgt.encounter_button = _NEWBUT(
            '2) Group',
            ibswgt.back.compute_encounters,
            bgcolor=(255, 255, 150), fontkw=primary_fontkw)

        ibswgt.detect_button = _NEWBUT(
            '3) Detect',
            ibswgt.back.run_detection,
            bgcolor=(150, 255, 150), fontkw=primary_fontkw)

        ibswgt.inc_query_button = _NEWBUT(
            '4) Identify',
            ibswgt.back.incremental_query,
            bgcolor=identify_color,
            fgcolor=(0, 0, 0), fontkw=primary_fontkw)

        hack_enabled_machines = [
            'ibeis.cs.uic.edu',
            'pachy.cs.uic.edu',
            'hyrule',
        ]
        enable_complete = ut.get_computer_name() in hack_enabled_machines

        ibswgt.reviewed_button = _NEWBUT(
            '5) Complete',
            ibswgt.back.encounter_reviewed_all_images,
            bgcolor=color_funcs.adjust_hsv_of_rgb255((0, 232, 211), 0., -.9, 0.),
            fontkw=primary_fontkw,
            enabled=enable_complete)

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
                ibswgt.batch_intra_encounter_query_button,
                ibswgt.batch_vsexemplar_query_button,
                _NEWLBL(''),
            ],
        ]

    def _init_layout(ibswgt):
        """ Lays out the defined components """
        # Add elements to the layout
        ibswgt.vlayout.addWidget(ibswgt.enc_tabwgt)
        ibswgt.vlayout.addWidget(ibswgt.vsplitter)
        ibswgt.vsplitter.addWidget(ibswgt.hsplitter)
        ibswgt.vsplitter.addWidget(ibswgt.status_wgt)
        # Horizontal Upper
        ibswgt.hsplitter.addWidget(ibswgt.views[ENCOUNTER_TABLE])
        ibswgt.hsplitter.addWidget(ibswgt._table_tab_wgt)
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

    def changing_models_gen(ibswgt, tblnames=None):
        """
        Loops over tablenames emitting layoutChanged at the end for each
        """
        tblnames = ibswgt.super_tblname_list if tblnames is None else tblnames
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
            if clear_view_selection:
                hack_selections.append(view.clearSelection)
            model._update()
        # Hack: Call this outside changing models gen
        for clearSelection in hack_selections:
            clearSelection()

    def connect_ibeis_control(ibswgt, ibs):
        """ Connects a new ibscontroler to the models """
        print('[newgui] connecting ibs control. ibs=%r' % (ibs,))
        ibswgt.enc_tabwgt._close_all_tabs()
        if ibs is None:
            print('[newgui] invalid ibs')
            title = 'No Database Opened'
            ibswgt.setWindowTitle(title)
        else:
            print('[newgui] Connecting valid ibs=%r' % ibs.get_dbname())
            #with ut.Indenter('[CONNECTING]'):
            # Give the frontend the new control
            ibswgt.ibs = ibs
            with ut.Timer('update special'):
                ibs.update_special_encounters()
            # Update the api models to use the new control
            with ut.Timer('make headers'):
                header_dict, declare_tup = gh.make_ibeis_headers_dict(ibswgt.ibs)
            ibswgt.declare_tup = declare_tup
            # Enable the redirections between tables
            #ibswgt._init_redirects()
            title = ibsfuncs.get_title(ibswgt.ibs)
            ibswgt.setWindowTitle(title)
            if ut.VERBOSE:
                print('[newgui] Calling model _update_headers')
            #block_wgt_flag = ibswgt._table_tab_wgt.blockSignals(True)

            with ut.Timer('[newgui] update models'):
                #for tblname in ibswgt.changing_models_gen(ibswgt.super_tblname_list):
                for tblname in ibswgt.super_tblname_list:
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
                    #block_model_flag = model.blockSignals(True)
                    model._update_headers(**header)
                    view._update_headers(**header)  # should use model headers
                    #model.blockSignals(block_model_flag)
                    #
                    #view.infer_delegates_from_model()
                for tblname in ibswgt.super_tblname_list:
                    view = ibswgt.views[tblname]
                    #if not view.isVisible():
                    #    print(view)
                    #    continue
                    view.hide_cols()
            #ibswgt._table_tab_wgt.blockSignals(block_wgt_flag)

            # FIXME: bad code
            # TODO: load previously loaded encounter or nothing
            LOAD_ENCOUNTER_ON_START = True
            if LOAD_ENCOUNTER_ON_START:
                #with ut.Indenter('[LOAD_ENC]'):
                eid_list = ibs.get_valid_eids(shipped=False)
                if len(eid_list) > 0:
                    DEFAULT_LARGEST_ENCOUNTER = False
                    if DEFAULT_LARGEST_ENCOUNTER:
                        numImg_list = ibs.get_encounter_num_gids(eid_list)
                        argx = ut.list_argsort(numImg_list)[-1]
                        eid = eid_list[argx]
                    else:  # Grab "first" encounter
                        eid = eid_list[0]
                    #ibswgt._change_enc(eid)
                    ibswgt.select_encounter_tab(eid)
                else:
                    #with ut.Indenter('[SET NEG1 ENC]'):
                    ibswgt._change_enc(-1)

    def setWindowTitle(ibswgt, title):
        parent_ = ibswgt.parent()
        if parent_ is not None:
            parent_.setWindowTitle(title)
        else:
            IBEIS_WIDGET_BASE.setWindowTitle(ibswgt, title)

    def _change_enc(ibswgt, eid):
        print('[newgui] _change_enc(eid=%r, uuid=%r)' % (eid, ibswgt.back.ibs.get_encounter_uuid(eid)))
        for tblname in ibswgt.tblname_list:
            view = ibswgt.views[tblname]
            view.clearSelection()
        for tblname in ibswgt.changing_models_gen(tblnames=ibswgt.tblname_list):
            view = ibswgt.views[tblname]
            view._change_enc(eid)
            #ibswgt.models[tblname]._change_enc(eid)  # the view should take care of this call
        try:
            #if eid is None:
            #    # HACK
            #    enctext = const.ALL_IMAGE_ENCTEXT
            #else:
            #    enctext = ibswgt.ibs.get_encounter_text(eid)
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

    #------------
    # SLOT HELPERS
    #------------

    def get_table_tab_index(ibswgt, tblname):
        view = ibswgt.views[tblname]
        index = ibswgt._table_tab_wgt.indexOf(view)
        return index

    def set_status_text(ibswgt, key, text):
        #printDBG('set_status_text[%r] = %r' % (index, text))
        index = ibswgt.tablename_to_status_widget_index[key]
        ibswgt.status_widget_list[index].setText(text)

    def set_table_tab(ibswgt, tblname):
        """
        Programmatically change to the table-tab to either:
        Image, ImageGrid, Annotation, or Names table/tree

        Example:
            >>> # GUI_DOCTEST
            >>> from ibeis.gui.newgui import *  # NOQA
            >>> # build test data
            >>> ibs, back, ibswgt, testdata_main_loop = testdata_guifront()
            >>> ibswgt.set_table_tab(gh.ANNOTATION_TABLE)
        """
        print('[newgui] set_table_tab: %r ' % (tblname,))
        index = ibswgt.get_table_tab_index(tblname)
        ibswgt._table_tab_wgt.setCurrentIndex(index)

    def select_encounter_tab(ibswgt, eid):
        if False:
            prefix = ut.get_caller_name(range(0, 10))
            prefix = prefix.replace('[wrp_noexectb]', 'w')
            prefix = prefix.replace('[slot_wrapper]', 's')
            prefix = prefix.replace('[X]', 'x')
        else:
            prefix = ''
        print(prefix + '[newgui] select_encounter_tab eid=%r' % (eid,))
        enctext = ibswgt.ibs.get_encounter_text(eid)
        #ibswgt.back.select_eid(eid)
        ibswgt.enc_tabwgt._add_enc_tab(eid, enctext)

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

    #------------
    # SLOTS
    #------------

    @slot_(str)
    def select_annot_text_editing_finished(ibswgt):
        tablename = gh.ANNOTATION_TABLE
        index = ibswgt.tablename_to_status_widget_index[tablename]
        text = ibswgt.status_widget_list[index].text()
        ibswgt.select_table_indicies_from_text(tablename, text)

    @slot_(str)
    def select_name_text_editing_finished(ibswgt):
        tablename = gh.NAMES_TREE
        index = ibswgt.tablename_to_status_widget_index[tablename]
        text = ibswgt.status_widget_list[index].text()
        ibswgt.select_table_indicies_from_text(tablename, text)

    @slot_(str)
    def select_image_text_editing_finished(ibswgt):
        tablename = gh.IMAGE_TABLE
        index = ibswgt.tablename_to_status_widget_index[tablename]
        text = ibswgt.status_widget_list[index].text()
        ibswgt.select_table_indicies_from_text(tablename, text)

    def select_table_indicies_from_text(ibswgt, tblname, text):
        """
        Args:
            tblname - tablename of the id to parse from text

        Ignore:
            text = '[1, 2,  3,]'
               text = '51e10019-968b-5f2e-2287-8432464d7547 '

        Example:
            >>> # GUI_DOCTEST
            >>> from ibeis.gui.newgui import *  # NOQA
            >>> # build test data
            >>> ibs, back, ibswgt, testdata_main_loop = testdata_guifront()
            >>> ibswgt.set_table_tab(gh.ANNOTATION_TABLE)
            >>> tblname = gh.NAMES_TREE
            >>> text = 'lena'
            >>> ibswgt.select_table_indicies_from_text(tblname, text)
        """
        if not ut.QUIET:
            print('[newgui] select_table_indicies_from_text')
            print('[newgui]  * gh.tblname = %r' % (tblname,))
            print('[newgui]  * text = %r' % (text,))
        to_backend_tablename = {
            gh.ANNOTATION_TABLE : const.ANNOTATION_TABLE,
            gh.NAMES_TREE       : const.NAME_TABLE,
            gh.IMAGE_TABLE      : const.IMAGE_TABLE,
        }
        backend_tablename = to_backend_tablename[tblname]
        if not ut.QUIET:
            print('[newgui]  * backend_tablename = %r' % (backend_tablename,))
        if text == '':
            text = '[]'
        try:
            #MODE1 = True
            #if MODE1:
            id_list_ = text.lstrip('[').rstrip(']').split(',')
            id_list = [id_.strip() for id_ in id_list_]
            id_list = [id_ for id_ in id_list if len(id_) > 0]
            try:
                id_list = list(map(int, id_list))
            except ValueError:
                import uuid
                try:
                    # First check to see if the text is a UUID
                    id_list = list(map(uuid.UUID, id_list))
                except ValueError:
                    if tblname != gh.NAMES_TREE:
                        raise
                    else:
                        # then maybe it was a name that was selected
                        id_list = list(map(str, id_list))
            #else:
            #    id_list_ = eval(text, globals(), locals())
            #    id_list = ut.ensure_iterable(id_list_)  # NOQA
        except Exception as ex:
            ut.printex(ex, iswarning=True, keys=['text'])
        else:
            if not ut.QUIET:
                print('[newgui]  * id_list = %r' % (id_list,))
            #print(id_list)
            id_list = ibswgt.back._set_selection3(backend_tablename, id_list, mode='set')

        # Select the index if we are in the right table tab
        if len(id_list) == 1 and ibswgt._table_tab_wgt.current_tblname == tblname:
            if not ut.QUIET:
                print('[newgui]  * attempting to select from rowid')
            #view = ibswgt.views[tblname]
            #view.select_row_from_id(id_list[0])
            ibswgt.goto_table_id(tblname, id_list[0])

            pass
        else:
            # TODO: convert the id into the ids corresponding with this tablename and move
            # to the first one
            pass

        #if goto_table_id:
        #    pass
        ibswgt.back.update_selection_texts()
        #pass

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
        TABLE_NICE = ibswgt.declare_tup[1]  # hack
        tblnice = TABLE_NICE[tblname]
        index = ibswgt.get_table_tab_index(tblname)
        text = tblnice + ' ' + str(nRows)
        #printDBG('Rows updated in index=%r, text=%r' % (index, text))
        # CHANGE TAB NAME TO SHOW NUMBER OF ROWS
        ibswgt._table_tab_wgt.setTabText(index, text)

    def goto_table_id(ibswgt, tablename, _id):
        print('[newgui] goto_table_id(tablenamd=%r, _id=%r)' % (tablename, _id))
        ibswgt.set_table_tab(tablename)
        view = ibswgt.views[tablename]
        view.select_row_from_id(_id, scroll=True)

    @slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuClicked(ibswgt, qtindex, pos):
        """
        Right click anywhere in the GUI
        Context menus on right click of a table
        """
        if not qtindex.isValid():
            return

        def _goto_image(gid):
            ibswgt.goto_table_id(IMAGE_TABLE, gid)

        def _goto_annot_image(aid):
            _goto_image(ibswgt.back.ibs.get_annot_gids(aid))

        def _goto_annot(aid):
            ibswgt.goto_table_id(gh.ANNOTATION_TABLE, aid)

        def _goto_annot_name(aid):
            ibswgt.goto_table_id(NAMES_TREE, ibswgt.back.ibs.get_annot_nids(aid))

        #printDBG('[newgui] contextmenu')
        model = qtindex.model()
        tblview = ibswgt.views[model.name]
        context_options = []
        qtindex_list = tblview.selectedIndexes()
        id_list      = [model._get_row_id(_qtindex) for _qtindex in qtindex_list]
        level_list   = [model._get_level(_qtindex) for _qtindex in qtindex_list]
        level2_ids_ = ut.group_items(id_list, level_list)
        level2_ids = {level: ut.unique_keep_order2(ids) for level, ids in six.iteritems(level2_ids_)}

        ibs = ibswgt.back.ibs
        back = ibswgt.back

        # ---- ENCOUNTER CONTEXT ----
        if model.name == ENCOUNTER_TABLE:
            merge_destination_id = model._get_row_id(qtindex)  # This is for the benefit of merge encounters
            enctext = ibswgt.back.ibs.get_encounter_text(merge_destination_id)
            eid_list = level2_ids[0]
            # Conditional context menu
            # TODO: remove duplicate code
            if len(eid_list) == 1:
                context_options += [
                    ('View encounter in Web', lambda: ibswgt.back.show_eid_list_in_web(eid_list)),
                    ('----', lambda: None),
                    ('Run detection on encounter (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_encounter(eid_list)),
                    ('Merge %d encounter into %s' %  (len(eid_list), (enctext)),
                        lambda: ibswgt.back.merge_encounters(eid_list, merge_destination_id)),
                    ('Copy encounter', lambda: ibswgt.back.copy_encounter(eid_list)),
                    ('Export encounter', lambda: ibswgt.back.export_encounters(eid_list)),
                    ('----', lambda: None),
                    ('Delete encounter', lambda: ibswgt.back.delete_encounter(eid_list)),
                    ('----', lambda: None),
                    ('Delete encounter AND images', lambda: ibswgt.back.delete_encounter_and_images(eid_list)),
                ]
            else:
                context_options += [
                    ('Run detection on encounters (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_encounter(eid_list)),
                    ('Copy encounter', lambda: ibswgt.back.copy_encounter(eid_list)),
                    ('Merge %d encounters into %s' %  (len(eid_list), (enctext)),
                        lambda: ibswgt.back.merge_encounters(eid_list, merge_destination_id)),
                    ('----', lambda: None),
                    ('Delete encounters', lambda: ibswgt.back.delete_encounter(eid_list)),
                    ('----', lambda: None),
                    ('Delete encounters AND images', lambda: ibswgt.back.delete_encounter_and_images(eid_list)),
                    # ('export encounters', lambda: ibswgt.back.export_encounters(eid_list)),
                ]
        # ---- IMAGE CONTEXT ----
        elif model.name == IMAGE_TABLE:
            current_enctext = ibswgt.back.ibs.get_encounter_text(ibswgt.back.get_selected_eid())
            gid_list = level2_ids[0]
            # Conditional context menu
            if len(gid_list) == 1:
                gid = gid_list[0]
                eid = model.eid
                view_aid_options1 = [
                    ('View aid=%r in Matplotlib' % (aid,), lambda: ibswgt.back.show_annotation(aid, web=False))
                    for aid in ibs.get_image_aids(gid)
                ]
                view_aid_options2 = [
                    ('View aid=%r in Web' % (aid,), lambda: ibswgt.back.show_annotation(aid, web=True))
                    for aid in ibs.get_image_aids(gid)
                ]
                context_options += [
                    ('View image in Matplotlib',
                        lambda: ibswgt.back.select_gid(gid, eid, show=True, web=False)),
                    ('View image in Web',
                        lambda: ibswgt.back.select_gid(gid, eid, show=True, web=True)),
                    ('View detection image (Hough) [dev]',
                        lambda: ibswgt.back.show_hough_image(gid)),
                    ('View annotation in Matplotlib:',
                       view_aid_options1),
                    ('View annotation in Web:',
                       view_aid_options2),
                    ('Add annotation from entire image',
                        lambda: ibswgt.back.add_annotation_from_image([gid])),
                    ('Run detection on image (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_images([gid])),
                ]
            else:
                context_options += [
                    ('View images in Web',
                        lambda: ibswgt.back.show_gid_list_in_web(gid_list)),
                    ('----', lambda: None),
                    ('Add annotation from entire images',
                        lambda: ibswgt.back.add_annotation_from_image(gid_list)),
                    ('Run detection on images (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_images(gid_list)),
                ]
            # Special condition for encounters
            if current_enctext != const.NEW_ENCOUNTER_ENCTEXT:
                context_options += [
                    ('----', lambda: None),
                    ('Move to new encounter',
                        lambda: ibswgt.back.send_to_new_encounter(gid_list, mode='move')),
                    ('Copy to new encounter',
                        lambda: ibswgt.back.send_to_new_encounter(gid_list, mode='copy')),
                ]
            if current_enctext != const.UNGROUPED_IMAGES_ENCTEXT:
                context_options += [
                    ('----', lambda: None),
                    ('Remove from encounter',
                        lambda: ibswgt.back.remove_from_encounter(gid_list)),
                ]
            # Continue the conditional context menu
            if len(gid_list) == 1:
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
                        lambda: ibswgt.back.delete_image_annotations(gid_list)),
                    ('Delete images',
                        lambda: ibswgt.back.delete_image(gid_list)),
                ]
        # ---- IMAGE GRID CONTEXT ----
        elif model.name == IMAGE_GRID:
            current_enctext = ibswgt.back.ibs.get_encounter_text(ibswgt.back.get_selected_eid())
            # Conditional context menu
            gid_list = level2_ids[0]
            if len(gid_list) == 1:
                gid = gid_list[0]
                eid = model.eid
                context_options += [
                    ('Go to image in Images Table',
                        lambda: ibswgt.goto_table_id(IMAGE_TABLE, gid)),
                    ('----', lambda: None),
                    ('View image in Matplotlib',
                        lambda: ibswgt.back.select_gid(gid, eid, show=True, web=False)),
                    ('View image in Web',
                        lambda: ibswgt.back.select_gid(gid, eid, show=True, web=True)),
                    ('View detection image (Hough) [dev]',
                        lambda: ibswgt.back.show_hough_image(gid)),
                    ('Add annotation from entire image',
                        lambda: ibswgt.back.add_annotation_from_image([gid])),
                    ('Run detection on image (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_images([gid])),
                ]
            else:
                context_options += [
                    ('View images in Web',
                        lambda: ibswgt.back.show_gid_list_in_web(gid_list)),
                    ('----', lambda: None),
                    ('Add annotation from entire images',
                        lambda: ibswgt.back.add_annotation_from_image(gid_list)),
                    ('Run detection on images (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_images(gid_list)),
                ]

            # Special condition for encounters
            if current_enctext != const.NEW_ENCOUNTER_ENCTEXT:
                context_options += [
                    ('----', lambda: None),
                    ('Move to new encounter',
                        lambda: ibswgt.back.send_to_new_encounter(gid_list, mode='move')),
                    ('Copy to new encounter',
                        lambda: ibswgt.back.send_to_new_encounter(gid_list, mode='copy')),
                ]
            if current_enctext != const.UNGROUPED_IMAGES_ENCTEXT:
                context_options += [
                    ('----', lambda: None),
                    ('Remove from encounter',
                        lambda: ibswgt.back.remove_from_encounter(gid_list)),
                ]
            # Continue the conditional context menu
            if len(gid_list) == 1:
                # We get gid from above
                context_options += [
                    ('----', lambda: None),
                    ('Delete image\'s annotations',
                        lambda: ibswgt.back.delete_image_annotations([gid])),
                    ('Delete image',
                        lambda: ibswgt.back.delete_image(gid)),
                ]
        # ---- ANNOTATION CONTEXT ----
        elif model.name == gh.ANNOTATION_TABLE:
            aid_list = level2_ids[0]
            # Conditional context menu
            # TODO: UNIFY COMMMON CONTEXT MENUS
            if len(aid_list) == 1:
                aid = aid_list[0]
                eid = model.eid
                context_options += [
                    ('----', lambda: None),
                    ('Go to image',
                        lambda: _goto_annot_image(aid)),
                    ('Go to name',
                        lambda: _goto_annot_name(aid)),
                    ('----', lambda: None),
                    ('Edit Annotation in Image',
                        lambda: ibswgt.spawn_edit_image_annotation_interaction_from_aid(aid, eid)),
                    ('----', lambda: None),
                    ('View annotation in Matplotlib',
                        #lambda: ibswgt.back.select_aid(aid, eid, show=True)),
                        lambda: ibswgt.back.show_annotation(aid, web=False)),
                    ('View annotation in Web',
                        #lambda: ibswgt.back.select_aid(aid, eid, show=True)),
                        lambda: ibswgt.back.show_annotation(aid, web=True)),
                    ('View image in Matplotlib',
                        lambda: ibswgt.back.select_gid_from_aid(aid, eid, show=True, web=False)),
                    ('View image in Web',
                        lambda: ibswgt.back.select_gid_from_aid(aid, eid, show=True, web=True)),
                    #back.show_image(gid, sel_aids=sel_aids)
                    ('View detection chip (probability) [dev]',
                        lambda: ibswgt.back.show_probability_chip(aid)),
                    ('----', lambda: None),
                    ('Unset annotation\'s name',
                        lambda: ibswgt.back.unset_names([aid])),
                    ('Delete annotation',
                        lambda: ibswgt.back.delete_annot(aid_list)),
                ]
            else:
                context_options += [
                    ('View annotations in Web',
                        lambda: ibswgt.back.show_aid_list_in_web(aid_list)),
                    ('Unset annotations\' names', lambda: ibswgt.back.unset_names(aid_list)),
                    ('Delete annotations', lambda: ibswgt.back.delete_annot(aid_list)),
                ]
        # ---- NAMES TREE CONTEXT ----
        elif model.name == NAMES_TREE:
            # TODO: map level list to tablename more reliably
            ut.print_dict(level2_ids)
            nid_list = level2_ids.get(0, [])
            aid_list = level2_ids.get(1, [])
            if len(aid_list) > 0 and len(nid_list) > 0:
                # two types of indices are selected, just return
                # fixme to do something useful
                print('multiple types of indicies selected')
                return
            else:
                if len(aid_list) == 1:
                    aid = aid_list[0]
                    eid = model.eid
                    context_options += [
                        ('Go to image', lambda: _goto_annot_image(aid)),
                        ('Go to annotation', lambda: _goto_annot(aid)),
                        ('----', lambda: None),
                        ('View annotation in Matplotlib', lambda: ibswgt.back.select_aid(aid, eid, show=False)),
                        ('View annotation in Web', lambda: ibswgt.back.select_aid(aid, eid, show=True)),
                        ('View image in Matplotlib', lambda: ibswgt.back.select_gid_from_aid(aid, eid, show=True, web=False)),
                        ('View image in Web', lambda: ibswgt.back.select_gid_from_aid(aid, eid, show=True, web=True)),
                    ]
                if len(aid_list) > 0:
                    def set_annot_names_to_same_new_name(ibswgt, aid_list):
                        ibswgt.back.ibs.set_annot_names_to_same_new_name(aid_list)
                        ibswgt.update_tables(tblnames=[gh.NAMES_TREE])
                    context_options += [
                        ('Rename annots (%s) to new name' % ut.list_str_summarized(aid_list, 'aid_list'),
                            lambda: set_annot_names_to_same_new_name(ibswgt, aid_list)),
                    ]
                if len(nid_list) > 0:
                    def run_splits(ibs, nid_list):
                        print('Checking for splits')
                        aids_list = ibs.get_name_aids(nid_list)
                        aid_list = sorted(list(set(ut.flatten(aids_list))))
                        back.run_annot_splits(aid_list)

                    def export_nids(ibs, nid_list):
                        from ibeis.dbio import export_subset
                        if not back.are_you_sure('Confirm export of nid_list=%r' % (nid_list,)):
                            return
                        export_subset.export_names(ibs, nid_list)

                    def create_new_encounter_from_names_(ibs, nid_list):
                        ibs.create_new_encounter_from_names(nid_list)
                        ibswgt.update_tables([gh.ENCOUNTER_TABLE], clear_view_selection=False)

                    context_options += [
                        ('View name(s) in Web', lambda: ibswgt.back.show_nid_list_in_web(nid_list)),
                        ('----', lambda: None),
                        ('Check for splits', lambda: run_splits(ibs, nid_list)),
                        ('Export names', lambda: export_nids(ibs, nid_list)),
                        ('Create Encounter From Name(s)', lambda: create_new_encounter_from_names_(ibs, nid_list)),
                    ]
                else:
                    print('nutin')
                    pass
        # Show the context menu
        ut.print_list(context_options)
        if len(context_options) > 0:
            guitool.popup_menu(tblview, pos, context_options)

    @slot_(QtCore.QModelIndex)
    def on_doubleclick(ibswgt, qtindex):
        """
        Double clicking anywhere in the GUI
        """
        print('\n+--- DOUBLE CLICK ---')
        if not qtindex.isValid():
            print('[doubleclick] invalid qtindex')
            return
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
            elif model.name == gh.ANNOTATION_TABLE:
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

    def register_redirect(ibswgt, src_table, src_table_col, dst_table, mapping_func):
        if src_table not in ibswgt.redirects.keys():
            ibswgt.redirects[src_table] = {}
        ibswgt.redirects[src_table][src_table_col] = (dst_table, mapping_func)

    #def _init_redirects(ibswgt):
    #    """
    #    redirects allows user to go from a row of a table to corresponding rows
    #    of other tables
    #    """
    #    redirects = gh.get_redirects(ibswgt.ibs)
    #    for src_table in redirects.keys():
    #        for src_table_name in redirects[src_table].keys():
    #            dst_table, mapping_func = redirects[src_table][src_table_name]
    #            src_table_col = gh.TABLE_COLNAMES[src_table].index(src_table_name)
    #            ibswgt.register_redirect(src_table, src_table_col, dst_table, mapping_func)

    @slot_(QtCore.QModelIndex)
    def on_click(ibswgt, qtindex):
        """
        Clicking anywhere in the GUI

        DOENT DO ANYTHING ANYMORE SELECTION MODEL USED INSTEAD

        DEPRICATE
        """
        return
        #printDBG('on_click')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex)
        #model_name = model.name
        #print('clicked: %s' + ut.dict_str(locals()))
        if False:
            try:
                dst_table, mapping_func = ibswgt.redirects[model.name][qtindex.column()]
                dst_id = mapping_func(id_)
                print("[on_click] Redirecting to: %r" % (dst_table, ))
                print("[on_click]     Mapping %r -> %r" % (id_, dst_id, ))
                ibswgt.set_table_tab(dst_table)
                ibswgt.views[dst_table].select_row_from_id(id_, scroll=True)
                return None
            except Exception as ex:
                print('no redirects')
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
            (IMAGE_TABLE, 0)         : ibswgt.back.select_gid,
            (IMAGE_GRID, 0)          : ibswgt.back.select_gid,
            (gh.ANNOTATION_TABLE, 0) : ibswgt.back.select_aid,
            (NAME_TABLE, 0)          : ibswgt.back.select_nid,
            (NAMES_TREE, 0)          : ibswgt.back.select_nid,
            (NAMES_TREE, 1)          : ibswgt.back.select_aid,
        }
        select_func = select_func_dict[(table_key, level)]
        select_func(id_, eid, show=False)

    def filter_annotation_table(ibswgt):
        r"""
        Args:


        CommandLine:
            python -m ibeis.gui.newgui --test-filter_annotation_table

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.gui.newgui import *  # NOQA
            >>> ibs, back, ibswgt, testdata_main_loop = testdata_guifront('testdb3')
            >>> result = ibswgt.filter_annotation_table(
            >>> print(result)
        """
        model = ibswgt.models[gh.ANNOTATION_TABLE]  # NOQA

        ibs = ibswgt.back.ibs
        annotmatch_rowid_list = ibs._get_all_annotmatch_rowids()
        isscenerymatch_list = ibs.get_annotmatch_is_scenerymatch(annotmatch_rowid_list)
        ut.list_take(isscenerymatch_list, ut.list_where(isscenerymatch_list))

        def get_aids_with_annotmatchprop():
            from ibeis import constants as const
            from ibeis.control import _autogen_annotmatch_funcs
            colnames = (_autogen_annotmatch_funcs.ANNOT_ROWID1, _autogen_annotmatch_funcs.ANNOT_ROWID2)
            tblname = const.ANNOTMATCH_TABLE
            wherecol = _autogen_annotmatch_funcs.ANNOTMATCH_IS_SCENERYMATCH
            whereclause = wherecol + '=?'
            colname_str = ', '.join(colnames)
            operation = ut.codeblock(
                '''
                SELECT {colname_str}
                FROM {tblname}
                WHERE {whereclause}
                ''').format(colname_str=colname_str, tblname=tblname, whereclause=whereclause)

            ibs.db.cur.execute(operation, [True])
            scenery_aids = list(set(ut.flatten(ibs.db.cur.fetchall())))
            return scenery_aids

        #annotmatch_rowid_list = ibs._get_all_annotmatch_rowids()
        #ishard_list         = ibs.get_annotmatch_is_hard(annotmatch_rowid_list)
        #isphotobomb_list    = ibs.get_annotmatch_is_photobomb(annotmatch_rowid_list)
        #isscenerymatch_list = ibs.get_annotmatch_is_scenerymatch(annotmatch_rowid_list)
        #isnondistinct_list  = ibs.get_annotmatch_is_nondistinct(annotmatch_rowid_list)
        #hards        = np.array(ut.replace_nones(ishard_list, False))
        #photobombs   = np.array(ut.replace_nones(isphotobomb_list, False))
        #scenerys     = np.array(ut.replace_nones(isscenerymatch_list, False))
        #nondistincts = np.array(ut.replace_nones(isnondistinct_list, False))
        #flags = vt.and_lists(vt.or_lists(hards, nondistincts), ~photobombs, ~scenerys)
        #annotmatch_rowid_list_ = ut.list_compress(annotmatch_rowid_list, flags)

        #aid1_list = ibs.get_annotmatch_aid1(annotmatch_rowid_list_)
        #aid2_list = ibs.get_annotmatch_aid2(annotmatch_rowid_list_)
        #aid_list = sorted(list(set(aid1_list + aid2_list)))

        #def filter_to_background(aid_list, ibs=ibswgt.back.ibs):
        #    ibswgt.back.ibs
        #model.set_ider_filters()
        #with ChangeLayoutContext([model]):
        #    IBEISSTRIPEMODEL_BASE._update_rows(model)
        #pass


######################
###### Testing #######
######################

def testdata_guifront(defaultdb='testdb1'):
    import ibeis
    main_locals = ibeis.main(defaultdb=defaultdb)
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
        python -m ibeis.gui.newgui --test-testfunc --cmd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.newgui import *  # NOQA
        >>> result = testfunc()
        >>> # verify results
        >>> print(result)
    """
    ibs, back, ibswgt, testdata_main_loop = testdata_guifront()
    view = ibswgt.views[gh.IMAGE_TABLE]
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
