#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This should probably be renamed guifront.py This defines all of the visual
components to the GUI It is invoked from guiback, which handles the nonvisual
logic.


BUGS:
    * Copying the ungrouped imageset raises an error. Should have the option
    to copy or move it. Other special imageset should not have this option.

    Should gray out an option if it is not available.


CommandLine:
    wbia --dbdir ~/lev/media/hdd/work/WWF_Lynx/ --name-tab
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, map, filter
from os.path import isdir
import sys
import functools
import utool as ut
from wbia.guitool.__PYQT__ import QtCore
from wbia.guitool.__PYQT__ import QtWidgets
from wbia.guitool.__PYQT__.QtCore import Qt
from wbia.guitool import slot_, ChangeLayoutContext

# from wbia.guitool import BlockContext
# from wbia.guitool import checks_qt_error
import wbia.guitool as gt
import wbia.plottool as pt
from wbia.plottool import color_funcs
from wbia.gui import guiheaders as gh
from wbia.gui import guimenus
import six
from wbia.viz.interact import interact_annotations2
from wbia.gui.guiheaders import (
    IMAGE_TABLE,
    IMAGE_GRID,
    ANNOTATION_TABLE,
    NAME_TABLE,
    NAMES_TREE,
    IMAGESET_TABLE,
)
from wbia.gui.models_and_views import (
    IBEISStripeModel,
    IBEISTableView,
    IBEISItemModel,
    IBEISTreeView,
    ImagesetTableModel,
    ImagesetTableView,
    IBEISTableWidget,
    IBEISTreeWidget,
    ImagesetTableWidget,
)
from wbia import constants as const

print, rrr, profile = ut.inject2(__name__)


VERBOSE_GUI = ut.VERBOSE or ut.get_argflag(('--verbose-gui', '--verbgui'))
WITH_GUILOG = ut.get_argflag('--guilog')

# ############################
# ##### Tab Widgets #######
# ############################


class APITabWidget(QtWidgets.QTabWidget):
    """
    Holds the table-tabs

    use setCurrentIndex to change the selection
    """

    def __init__(tabwgt, parent=None, horizontalStretch=1):
        # QtWidgets.QTabWidget.__init__(tabwgt, parent)
        super(APITabWidget, tabwgt).__init__(parent)
        tabwgt.ibswgt = parent
        tabwgt._sizePolicy = gt.newSizePolicy(tabwgt, horizontalStretch=horizontalStretch)
        tabwgt.setSizePolicy(tabwgt._sizePolicy)
        # tabwgt.currentChanged.connect(tabwgt.setCurrentIndex)
        tabwgt.currentChanged.connect(tabwgt._on_tabletab_change)
        tabwgt.current_tblname = None

    @slot_(int)
    def _on_tabletab_change(tabwgt, index):
        """ Switch to the current imageset tab """
        if VERBOSE_GUI:
            print('[apitab] _onchange(index=%r)' % (index,))
        tblname = tabwgt.ibswgt.tblname_list[index]
        tabwgt.current_tblname = tblname
        if VERBOSE_GUI:
            print('[apitab] _onchange(tblname=%r)' % (tblname,))
        tabwgt.ibswgt.back._clear_selection()
        view = tabwgt.ibswgt.views[tblname]
        selected = view.selectionModel().selection()
        deselected = QtCore.QItemSelection()
        tabwgt.ibswgt.update_selection(selected, deselected)


class ImageSetTabWidget(QtWidgets.QTabWidget):
    """
    Handles tabs that contain individual image sets.
    """

    def __init__(imageset_tabwgt, parent=None, horizontalStretch=1):
        QtWidgets.QTabWidget.__init__(imageset_tabwgt, parent)
        imageset_tabwgt.ibswgt = parent
        imageset_tabwgt.setTabsClosable(True)
        imageset_tabwgt.setMaximumSize(9999, gt.get_cplat_tab_height())
        imageset_tabwgt.tabbar = imageset_tabwgt.tabBar()
        imageset_tabwgt.tabbar.setMovable(False)
        imageset_tabwgt.setStyleSheet('border: none;')
        imageset_tabwgt.tabbar.setStyleSheet('border: none;')
        sizePolicy = gt.newSizePolicy(
            imageset_tabwgt, horizontalStretch=horizontalStretch
        )
        imageset_tabwgt.setSizePolicy(sizePolicy)

        imageset_tabwgt.tabCloseRequested.connect(imageset_tabwgt._close_tab)
        imageset_tabwgt.currentChanged.connect(imageset_tabwgt._on_imagesettab_change)

        imageset_tabwgt.imgsetid_list = []
        # TURNING ON / OFF ALL IMAGES
        # imageset_tabwgt._add_imageset_tab(-1, const.ALL_IMAGE_IMAGESETTEXT)

    @slot_(int)
    def _on_imagesettab_change(imageset_tabwgt, index):
        """ Switch to the current imageset tab """
        if VERBOSE_GUI:
            print('[imageset_tab_widget] _onchange(index=%r)' % (index,))
        if 0 <= index and index < len(imageset_tabwgt.imgsetid_list):
            imgsetid = imageset_tabwgt.imgsetid_list[index]
            if VERBOSE_GUI:
                print('[IMAGESETTAB.ONCHANGE] imgsetid = %r' % (imgsetid,))
            imageset_tabwgt.ibswgt._change_imageset(imgsetid)
        else:
            imageset_tabwgt.ibswgt._change_imageset(-1)

    @slot_(int)
    def _close_tab(imageset_tabwgt, index):
        print('[imageset_tab_widget] _close_tab(index=%r)' % (index,))
        if imageset_tabwgt.imgsetid_list[index] is not None:
            imageset_tabwgt.imgsetid_list.pop(index)
            imageset_tabwgt.removeTab(index)

    @slot_()
    def _close_all_tabs(imageset_tabwgt):
        if VERBOSE_GUI:
            print('[imageset_tab_widget] _close_all_tabs()')
        while len(imageset_tabwgt.imgsetid_list) > 0:
            index = 0
            imageset_tabwgt.imgsetid_list.pop(index)
            imageset_tabwgt.removeTab(index)

    @slot_(int)
    def _close_tab_with_imgsetid(imageset_tabwgt, imgsetid):
        print('[imageset_tab_widget] _close_tab_with_imgsetid(imgsetid=%r)' % (imgsetid))
        try:
            index = imageset_tabwgt.imgsetid_list.index(imgsetid)
            imageset_tabwgt._close_tab(index)
        except Exception:
            pass

    def _add_imageset_tab(imageset_tabwgt, imgsetid, imagesettext):
        if VERBOSE_GUI:
            print(
                '[_add_imageset_tab] imgsetid=%r, imagesettext=%r'
                % (imgsetid, imagesettext)
            )
        if imgsetid not in imageset_tabwgt.imgsetid_list:
            if VERBOSE_GUI:
                print('[_add_imageset_tab] adding new image tab')
            imageset_tabwgt.imgsetid_list.append(imgsetid)
            index = len(imageset_tabwgt.imgsetid_list) - 1
            tab_name = str(imagesettext)
            # Only has a tab, doesn't actually contain anything
            hack_newtab = QtWidgets.QWidget(parent=imageset_tabwgt)
            imageset_tabwgt.addTab(hack_newtab, tab_name)
        else:
            if VERBOSE_GUI:
                print('[_add_imageset_tab] using existing image tab')
            index = imageset_tabwgt.imgsetid_list.index(imgsetid)

        if VERBOSE_GUI:
            print('[_add_imageset_tab] setCurrentIndex(index=%r)' % (index,))
        imageset_tabwgt.setCurrentIndex(index)
        # Dont call this, it is triggered twice
        # imageset_tabwgt._on_imagesettab_change(index)

    def _update_imageset_tab_name(imageset_tabwgt, imgsetid, imagesettext):
        for index, _id in enumerate(imageset_tabwgt.imgsetid_list):
            if imgsetid == _id:
                imageset_tabwgt.setTabText(index, imagesettext)


# ############################
# ####### Main Widget ########
# ############################


class IBEISMainWindow(QtWidgets.QMainWindow):
    quitSignal = QtCore.pyqtSignal()
    dropSignal = QtCore.pyqtSignal(list)

    def __init__(mainwin, back=None, ibs=None, parent=None):
        super(IBEISMainWindow, mainwin).__init__(parent)
        # Menus
        try:
            mainwin.setUnifiedTitleAndToolBarOnMac(False)
        except AttributeError as ex:
            ut.printex(
                ex, 'setUnifiedTitleAndToolBarOnMac is not working', iswarning=True
            )
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


# ############################
# #### IBEIS GUI Widget ######
# ############################


IBEIS_WIDGET_BASE = QtWidgets.QWidget


class IBEISGuiWidget(IBEIS_WIDGET_BASE):
    """
    CommandLine:
        # Testing
        python -m wbia --db NNP_Master3 --onlyimgtbl
        python -m wbia --db PZ_Master1 --onlyimgtbl
    """

    def __init__(ibswgt, back=None, ibs=None, parent=None):
        super(IBEISGuiWidget, ibswgt).__init__(parent)
        ibswgt.ibs = ibs
        ibswgt.back = back
        # Structures that will hold models and views
        ibswgt.models = {}
        ibswgt.views = {}

        # FIXME: Duplicate models
        # Create models and views
        # Define the abstract item models and views for the tables
        ibswgt.tblname_list = []
        ibswgt.modelview_defs = []

        # NEW DYNAMIC WAY OF USING TABS AND API VIEWS
        # ADD IMAGE TABLE
        if True:
            ibswgt.tblname_list.append(IMAGE_TABLE)
            ibswgt.modelview_defs.append(
                (IMAGE_TABLE, IBEISTableWidget, IBEISItemModel, IBEISTableView)
            )
        # ADD IMAGE GRID
        USE_GRID = False
        if USE_GRID and not ut.get_argflag('--onlyimgtbl'):
            ibswgt.tblname_list.append(IMAGE_GRID)
            ibswgt.modelview_defs.append(
                (IMAGE_GRID, IBEISTableWidget, IBEISStripeModel, IBEISTableView)
            )
        # ADD ANNOT GRID
        if not (ut.get_argflag('--noannottbl') or ut.get_argflag('--onlyimgtbl')):
            ibswgt.tblname_list.append(gh.ANNOTATION_TABLE)
            ibswgt.modelview_defs.append(
                (gh.ANNOTATION_TABLE, IBEISTableWidget, IBEISItemModel, IBEISTableView)
            )
        # ADD NAME TREE
        if not (ut.get_argflag('--nonametree') or ut.get_argflag('--onlyimgtbl')):
            ibswgt.tblname_list.append(NAMES_TREE)
            ibswgt.modelview_defs.append(
                (NAMES_TREE, IBEISTreeWidget, IBEISItemModel, IBEISTreeView)
            )
        # ADD IMAGESET TABLE
        ibswgt.super_tblname_list = ibswgt.tblname_list + [IMAGESET_TABLE]
        ibswgt.modelview_defs.append(
            (IMAGESET_TABLE, ImagesetTableWidget, ImagesetTableModel, ImagesetTableView)
        )

        # DO INITALIZATION
        # Create and layout components
        ibswgt._init_components()
        ibswgt._connect_signals_and_slots()
        ibswgt.connect_wbia_control(ibswgt.ibs)

        # Lazy load data every so often
        ibswgt.data_load_timer = QtCore.QTimer()
        ibswgt.data_load_timer.timeout.connect(ibswgt.data_load_loop)
        ibswgt.data_load_freq = 1000
        # ibswgt.tt = None
        ibswgt.data_load_timer.start(ibswgt.data_load_freq)

    # @QtCore.pyqtSlot
    def data_load_loop(ibswgt):
        # Get the current view
        view = ibswgt._tables_tab_widget.currentWidget()
        model = view.model()
        if model.canFetchMore(None):
            model.fetchMore(None)
        # if ibswgt.tt is None:
        #    ellapsed = ibswgt.data_load_freq
        # else:
        #    ellapsed = ut.toc(ibswgt.tt)
        # print('Load more data for %r' % (view,))
        # # freq = ibswgt.data_load_freq / (1000 * 2)
        # # frac = freq / ellapsed
        # # print('ibswgt.data_load_freq = %r' % (freq,))
        # # print('ellapsed = %r' % (ellapsed,))
        # # print('frac = %r' % (frac,))
        # # if model.batch_size is not None:
        #    #new_batch_size = model.batch_size
        #    #new_batch_size = int(new_batch_size * frac * 2)
        #    #new_batch_size = max(new_batch_size * 2, 2)
        #    #new_batch_size = min(model.batch_size * 2, new_batch_size)
        #    #new_batch_size = min(100, new_batch_size)
        #    #model.batch_size = new_batch_size
        #    #print('model.batch_size = %r' % (model.batch_size,))
        # # ibswgt.tt = ut.tic()
        # else:
        #    ibswgt.tt = None

    def _init_components(ibswgt):
        """ Defines gui components and inits layout """
        # Layout
        ibswgt.vlayout = QtWidgets.QVBoxLayout(ibswgt)
        ibswgt.setLayout(ibswgt.vlayout)

        ibswgt.vsplitter = gt.newSplitter(ibswgt, orientation=Qt.Vertical)
        ibswgt.hsplitter = gt.newSplitter(
            ibswgt, orientation=Qt.Horizontal, verticalStretch=18
        )
        #
        # Tables Tab
        ibswgt._tables_tab_widget = APITabWidget(parent=ibswgt, horizontalStretch=81)
        for tblname, WidgetClass, ModelClass, ViewClass in ibswgt.modelview_defs:
            # Make view first to pass as parent
            ibswgt.views[tblname] = ViewClass(parent=ibswgt)
            # FIXME: It is very bad to give the model a view. Only the view
            # should have a model
            ibswgt.models[tblname] = ModelClass(parent=ibswgt.views[tblname])
        # Connect models and views
        for tblname in ibswgt.super_tblname_list:
            ibswgt.views[tblname].setModel(ibswgt.models[tblname])
        # Add Image, ANNOTATION, and Names as tabs
        for tblname in ibswgt.tblname_list:
            ibswgt._tables_tab_widget.addTab(ibswgt.views[tblname], tblname)

        # Force full loading
        ibswgt.models[IMAGE_TABLE].batch_size = 1000
        ibswgt.models[IMAGESET_TABLE].batch_size = 1000
        ibswgt.models[ANNOTATION_TABLE].batch_size = 1000
        ibswgt.models[NAMES_TREE].batch_size = 2
        ibswgt.models[NAMES_TREE].batch_size = 100

        # Custom ImageSet Tab Wiget
        ibswgt.imageset_tabwgt = ImageSetTabWidget(parent=ibswgt)
        ibswgt.vlayout.addWidget(ibswgt.vsplitter)
        ibswgt.vsplitter.addWidget(ibswgt.hsplitter)

        ibswgt.status_wgt = status_wgt = gt.newWidget(
            ibswgt.vsplitter,
            orientation=Qt.Vertical,
            spacing=3,
            margin=0,
            name='StatusWidget',
        )
        ibswgt.vsplitter.addWidget(status_wgt)

        # On the LEFT add the the table of ImageSets
        imgset_table_view = ibswgt.views[IMAGESET_TABLE]
        imgset_table_view.setSizePolicy(
            gt.newSizePolicy(hSizePolicy='Expanding', hStretch=2)
        )

        # On the RIGHT add the DataTables and tabs
        right_hack_wgt = gt.newWidget()
        right_hack_wgt.addWidget(ibswgt.imageset_tabwgt)
        right_hack_wgt.addWidget(ibswgt._tables_tab_widget)
        right_hack_wgt.setSizePolicy(gt.newSizePolicy(horizontalStretch=5))

        # Hack because the tables aren't actually belonging to the tabs
        # They are just controlled by the change
        ibswgt.hsplitter.addWidget(imgset_table_view)
        ibswgt.hsplitter.addWidget(right_hack_wgt)

        _NEWLBL = functools.partial(gt.newLabel, ibswgt)
        _NEWBUT = functools.partial(gt.newButton, ibswgt)
        # _COMBO  = functools.partial(gt.newComboBox, ibswgt)
        _NEWTEXT = functools.partial(gt.newLineEdit, ibswgt, verticalStretch=1)

        primary_fontkw = dict(bold=True, pointSize=11)
        secondary_fontkw = dict(bold=False, pointSize=9)
        # advanced_fontkw = dict(bold=False, pointSize=8, italic=True)
        identify_color = (255, 150, 0)

        ibswgt.tablename_to_status_widget_index = {
            IMAGESET_TABLE: 1,
            IMAGE_TABLE: 3,
            IMAGE_GRID: 3,
            gh.ANNOTATION_TABLE: 5,
            NAMES_TREE: 7,
            NAME_TABLE: 7,
        }
        ibswgt.key_to_objnice = {
            IMAGESET_TABLE: 'ImageSet',
            IMAGE_TABLE: 'Image',
            ANNOTATION_TABLE: 'Annotation',
            NAMES_TREE: 'Name',
        }
        ibswgt.status_widget_list = [
            _NEWLBL('Selected ImageSet: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(enabled=True, readOnly=True),
            ##
            _NEWLBL('Selected Image: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(
                enabled=True,
                readOnly=False,
                editingFinishedSlot=ibswgt.select_image_text_editing_finished,
            ),
            ##
            _NEWLBL('Selected Annotation: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(
                enabled=True,
                readOnly=False,
                editingFinishedSlot=ibswgt.select_annot_text_editing_finished,
            ),
            ##
            _NEWLBL('Selected Name: ', fontkw=secondary_fontkw, align='right'),
            _NEWTEXT(
                enabled=True,
                readOnly=False,
                editingFinishedSlot=ibswgt.select_name_text_editing_finished,
            ),
        ]

        back = ibswgt.back

        ibswgt.batch_intra_occurrence_query_button = _NEWBUT(
            '4) ID Encounters',
            functools.partial(
                back.compute_queries,
                daids_mode=const.INTRA_OCCUR_KEY,
                query_is_known=None,
                use_prioritized_name_subset=False,
                cfgdict={'can_match_samename': False, 'use_k_padding': False},
            ),
            bgcolor=color_funcs.adjust_hsv_of_rgb255(identify_color, -0.01, -0.7, 0.0),
            fgcolor=(0, 0, 0),
            fontkw=primary_fontkw,
        )

        ibswgt.batch_vsexemplar_query_button = _NEWBUT(
            '5) ID Exemplars',
            functools.partial(
                back.compute_queries,
                daids_mode=const.VS_EXEMPLARS_KEY,
                use_prioritized_name_subset=True,
                query_is_known=None,
                cfgdict={'can_match_samename': False, 'use_k_padding': False},
            ),
            bgcolor=color_funcs.adjust_hsv_of_rgb255(identify_color, -0.02, -0.7, 0.0),
            fgcolor=(0, 0, 0),
            fontkw=primary_fontkw,
        )

        ibswgt.import_button = _NEWBUT(
            '1) Import',
            # back.import_images_from_dir,
            back.import_button_click,
            bgcolor=(235, 200, 200),
            fontkw=primary_fontkw,
        )

        ibswgt.imageset_button = _NEWBUT(
            '2) Group',
            ibswgt.back.do_group_occurrence_step,
            bgcolor=(255, 255, 150),
            fontkw=primary_fontkw,
        )

        ibswgt.detect_button = _NEWBUT(
            '3) Detect',
            ibswgt.back.run_detection_step,
            bgcolor=(150, 255, 150),
            fontkw=primary_fontkw,
        )

        ibswgt.reviewed_button = _NEWBUT(
            '6) Complete',
            ibswgt.back.commit_to_wb_step,
            bgcolor=color_funcs.adjust_hsv_of_rgb255((0, 232, 211), 0.0, -0.9, 0.0),
            fontkw=primary_fontkw,
            enabled=True,
        )

        ibswgt.control_widget_lists = [
            [
                ibswgt.import_button,
                ibswgt.imageset_button,
                _NEWLBL(''),
                ibswgt.detect_button,
                ibswgt.batch_intra_occurrence_query_button,
                ibswgt.batch_vsexemplar_query_button,
                ibswgt.reviewed_button,
            ],
            [
                _NEWBUT(
                    'Advanced ID Interface',
                    back.show_advanced_id_interface,
                    bgcolor=color_funcs.adjust_hsv_of_rgb255(identify_color),
                    fgcolor=(0, 0, 0),
                    fontkw=primary_fontkw,
                )
            ],
        ]

        # Other components
        # New widget has black magic (for implicit layouts) in it
        # Add control widgets (import, group, species selector, etc...)
        for count, control_widgets in enumerate(ibswgt.control_widget_lists):
            _container = status_wgt.addNewWidget(
                orientation=Qt.Horizontal, margin=0, name='ControlContainer%d' % (count,),
            )
            for widget in control_widgets:
                _container.addWidget(widget)

        # Output log (turned off by default)
        ibswgt.outputLog = gt.newOutputLog(
            status_wgt, pointSize=8, visible=WITH_GUILOG, verticalStretch=6
        )
        status_wgt.addWidget(ibswgt.outputLog)

        # Add selected ids status widget
        ibswgt.selectionStatusWidget = status_wgt.addNewWidget(
            orientation=Qt.Horizontal, margin=3, name='SelectionStatus'
        )
        for widget in ibswgt.status_widget_list:
            ibswgt.selectionStatusWidget.addWidget(widget)

        # Progress bar is at the bottom
        ibswgt.prog_bar = status_wgt.addNewProgressBar(
            visible=False, verticalStretch=1, name='prog_bar'
        )

        # ibswgt.vsplitter.print_widget_heirarchy(max_depth=4)

    def _connect_signals_and_slots(ibswgt):
        if VERBOSE_GUI:
            print('[newgui] _connect_signals_and_slots')
        for tblname in ibswgt.super_tblname_list:
            tblview = ibswgt.views[tblname]
            tblview.doubleClicked.connect(ibswgt.on_doubleclick)
            tblview.contextMenuClicked.connect(ibswgt.on_contextMenuClicked)
            if tblname != gh.IMAGESET_TABLE:
                tblview.selectionModel().selectionChanged.connect(ibswgt.update_selection)
            # front.printSignal.connect(back.backend_print)
            # front.raiseExceptionSignal.connect(back.backend_exception)
            # CONNECT HOOK TO GET NUM ROWS
            tblview.rows_updated.connect(ibswgt.on_rows_updated)

    @slot_(QtCore.QItemSelection, QtCore.QItemSelection)
    def update_selection(ibswgt, selected, deselected):
        """
        Quirky behavior: if you select two columns in a row and then unselect
        only one, the whole row is unselected, because this function only deals
        with deltas.

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.gui.newgui import *  # NOQA
            >>> ibs, back, ibswgt, testdata_main_loop = testdata_guifront()
            >>> ibswgt.set_table_tab(gh.NAMES_TREE)
            >>> view = ibswgt.views[gh.NAMES_TREE]
            >>> view.expandAll()
            >>> AUTOSELECT = False
            >>> if AUTOSELECT:
            ...     view.selectAll()
            >>> selmodel = view.selectionModel()
            >>> selected = selmodel.selection()
            >>> deselected = QtCore.QItemSelection()
            >>> # verify results
            >>> print(result)

        """
        # print('[ibswgt] update selection')
        # print('selected = ' + str(selected.indexes()))
        # print('deselected = ' + str(deselected.indexes()))
        deselected_model_index_list_ = deselected.indexes()
        selected_model_index_list_ = selected.indexes()

        def get_selection_info(model_index_list_):
            model_index_list = [
                qtindex for qtindex in model_index_list_ if qtindex.isValid()
            ]
            model_list = [qtindex.model() for qtindex in model_index_list]
            tablename_list = [model.name for model in model_list]
            level_list = [
                model._get_level(qtindex)
                for model, qtindex in zip(model_list, model_index_list)
            ]
            rowid_list = [
                model._get_row_id(qtindex)
                for model, qtindex in zip(model_list, model_index_list)
            ]
            table_key_list = list(zip(tablename_list, level_list))
            return table_key_list, rowid_list

        select_table_key_list, select_rowid_list = get_selection_info(
            selected_model_index_list_
        )
        deselect_table_key_list, deselect_rowid_list = get_selection_info(
            deselected_model_index_list_
        )

        table_key2_selected_rowids = dict(
            ut.group_items(select_rowid_list, select_table_key_list)
        )
        table_key2_deselected_rowids = dict(
            ut.group_items(deselect_rowid_list, deselect_table_key_list)
        )

        table_key2_selected_rowids = {
            key: list(set(val)) for key, val in six.iteritems(table_key2_selected_rowids)
        }
        table_key2_deselected_rowids = {
            key: list(set(val))
            for key, val in six.iteritems(table_key2_deselected_rowids)
        }
        if ut.VERBOSE:
            print('table_key2_selected_rowids = ' + ut.repr2(table_key2_selected_rowids))
            print(
                'table_key2_deselected_rowids = ' + ut.repr2(table_key2_deselected_rowids)
            )

        gh_const_tablename_map = {
            (IMAGE_TABLE, 0): const.IMAGE_TABLE,
            (IMAGE_GRID, 0): const.IMAGE_TABLE,
            (gh.ANNOTATION_TABLE, 0): const.ANNOTATION_TABLE,
            (NAME_TABLE, 0): const.NAME_TABLE,
            (NAMES_TREE, 0): const.NAME_TABLE,
            (NAMES_TREE, 1): const.ANNOTATION_TABLE,
        }
        # here tablename is a backend const tablename
        for table_key, id_list in six.iteritems(table_key2_deselected_rowids):
            tablename = gh_const_tablename_map[table_key]
            ibswgt.back._set_selection3(tablename, id_list, mode='diff')
        for table_key, id_list in six.iteritems(table_key2_selected_rowids):
            tablename = gh_const_tablename_map[table_key]
            ibswgt.back._set_selection3(tablename, id_list, mode='add')
        ibswgt.back.update_selection_texts()

        # tblview.selectionModel().selectedIndexes()

    def changing_models_gen(ibswgt, tblnames=None):
        """
        Loops over tablenames emitting layoutChanged at the end for each
        """
        tblnames = ibswgt.super_tblname_list if tblnames is None else tblnames
        if VERBOSE_GUI:
            print('[newgui] changing_models_gen(tblnames=%r)' % (tblnames,))
        model_list = [ibswgt.models[tblname] for tblname in tblnames]
        # model_list = [ibswgt.models[tblname] for tblname in tblnames if
        # ibswgt.views[tblname].isVisible()]
        with ChangeLayoutContext(model_list):
            for tblname in tblnames:
                yield tblname

    def update_tables(ibswgt, tblnames=None, clear_view_selection=True):
        """ forces changing models """
        print('[newgui] update_tables(%r)' % (tblnames,))
        hack_selections = []
        # print('[new_gui.UPDATE_TABLES]')
        for tblname in ibswgt.changing_models_gen(tblnames=tblnames):
            # print('[new_gui.update_tables] tblname=%r' % (tblname, ))
            model = ibswgt.models[tblname]
            view = ibswgt.views[tblname]
            if clear_view_selection:
                hack_selections.append(view.clearSelection)
            model._update()
        # Hack: Call this outside changing models gen
        for clearSelection in hack_selections:
            clearSelection()

    def connect_wbia_control(ibswgt, ibs):
        """ Connects a new ibscontroler to the models """
        if VERBOSE_GUI:
            print('[newgui] connect_wbia_control(ibs=%r)' % (ibs,))
        ibswgt.imageset_tabwgt._close_all_tabs()
        if ibs is None:
            if VERBOSE_GUI:
                print('[newgui] invalid ibs')
            title = 'No Database Opened'
            ibswgt.setWindowTitle(title)
        else:
            if VERBOSE_GUI:
                print('[newgui] Connecting valid ibs=%r' % ibs.get_dbname())
            # with ut.Indenter('[CONNECTING]'):
            # Give the frontend the new control
            ibswgt.ibs = ibs
            if not ut.get_argflag('--fast'):
                with ut.Timer('update special', verbose=VERBOSE_GUI):
                    if not ibs.readonly:
                        ibs.update_special_imagesets()
            else:
                if VERBOSE_GUI:
                    print('Skipping special imagesets')
            # Update the api models to use the new control
            with ut.Timer('make headers', verbose=VERBOSE_GUI):
                header_dict, declare_tup = gh.make_wbia_headers_dict(ibswgt.ibs)
            ibswgt.declare_tup = declare_tup

            def get_title(ibs):
                """
                DEPRICATE OR MOVE
                """
                if ibs is None:
                    title = 'IBEIS - No Database Directory Open'
                elif ibs.dbdir is None:
                    title = 'IBEIS - !! INVALID DATABASE !!'
                else:
                    dbdir = ibs.get_dbdir()
                    dbname = ibs.get_dbname()
                    title = 'IBEIS - %r - Database Directory = %s' % (dbname, dbdir)
                    wb_target = ibs.const.WILDBOOK_TARGET
                    # params.args.wildbook_target
                    if wb_target is not None:
                        title = '%s - Wildbook Target = %s' % (title, wb_target)
                return title

            title = get_title(ibswgt.ibs)
            ibswgt.setWindowTitle(title)
            if ut.VERBOSE:
                print('[newgui] Calling model _update_headers')
            # block_wgt_flag = ibswgt._tables_tab_widget.blockSignals(True)

            with ut.Timer('[newgui] update models', verbose=VERBOSE_GUI):
                # for tblname in ibswgt.changing_models_gen(ibswgt.super_tblname_list):
                for tblname in ibswgt.super_tblname_list:
                    model = ibswgt.models[tblname]
                    view = ibswgt.views[tblname]
                    # if not view.isVisible():
                    #    print(view)
                    # ut.embed()
                    header = header_dict[tblname]
                    # widget = ibswgt.widgets[tblname]
                    # widget.change_headers(header)
                    # NOT SURE IF THESE BLOCKERS SHOULD BE COMMENTED
                    # block_model_flag = model.blockSignals(True)
                    model._update_headers(**header)
                    view._update_headers(**header)  # should use model headers
                    # model.blockSignals(block_model_flag)
                    #
                    # view.infer_delegates_from_model()
                for tblname in ibswgt.super_tblname_list:
                    view = ibswgt.views[tblname]
                    # if not view.isVisible():
                    #    print(view)
                    #    continue
                    view.hide_cols()
            # ibswgt._tables_tab_widget.blockSignals(block_wgt_flag)

            # Update species with ones enabled in database
            if not ibs.readonly:
                ibswgt.update_species_available()

            # FIXME: bad code
            # TODO: load previously loaded imageset or nothing
            LOAD_IMAGESET_ON_START = not ut.get_argflag('--fast')
            if LOAD_IMAGESET_ON_START:
                imgsetid_list = ibs.get_valid_imgsetids(shipped=False)
                if len(imgsetid_list) > 0:
                    DEFAULT_LARGEST_IMAGESET = False
                    if DEFAULT_LARGEST_IMAGESET:
                        numImg_list = ibs.get_imageset_num_gids(imgsetid_list)
                        argx = ut.list_argsort(numImg_list, reverse=True)[0]
                        imgsetid = imgsetid_list[argx]
                    else:  # Grab "first" imageset
                        imgsetid = imgsetid_list[0]
                    # ibswgt._change_imageset(imgsetid)
                    ibswgt.select_imageset_tab(imgsetid)
                else:
                    ibswgt._change_imageset(-1)

    def update_species_available(
        ibswgt, reselect=False, reselect_new_name=None, deleting=False
    ):
        ibs = ibswgt.ibs
        # TODO: update these options depending on ibs.get_species_with_detectors
        # when a controller is attached to the gui
        detection_combo_box_options = [
            # Text              # Value
            # ('Select Species',  'none'),
            ('Select Species', const.UNKNOWN),
            ('Unknown', const.UNKNOWN),
            # 'none'),
        ] + sorted(list(ibs.get_working_species()))
        species_text = ibswgt.back.get_selected_species()
        reselect_index = None
        if not deleting and reselect_new_name is None and species_text is not None:
            species_rowid = ibs.get_species_rowids_from_text(species_text)
            reselect_new_name = ibs.get_species_nice(species_rowid)
            if VERBOSE_GUI:
                print('[front] Reselecting old selection: %r' % (reselect_new_name,))
        nice_name_list = [str(_[0]) for _ in detection_combo_box_options]
        if reselect_new_name in nice_name_list:
            reselect_index = nice_name_list.index(reselect_new_name)
            if VERBOSE_GUI:
                print('[front] Reselecting renamed selection: %r' % (reselect_new_name,))
        if VERBOSE_GUI:
            print('[front] Reselecting index: %r' % (reselect_index,))
        # ibswgt.species_combo.setOptions(detection_combo_box_options)
        # ibswgt.species_combo.updateOptions(reselect=reselect, reselect_index=reselect_index)

    def setWindowTitle(ibswgt, title):
        parent_ = ibswgt.parent()
        if parent_ is not None:
            parent_.setWindowTitle(title)
        else:
            IBEIS_WIDGET_BASE.setWindowTitle(ibswgt, title)

    def _change_imageset(ibswgt, imgsetid):
        if VERBOSE_GUI:
            print(
                '[newgui] _change_imageset(imgsetid=%r, uuid=%r)'
                % (imgsetid, ibswgt.back.ibs.get_imageset_uuid(imgsetid))
            )
        for tblname in ibswgt.tblname_list:
            view = ibswgt.views[tblname]
            view.clearSelection()
        for tblname in ibswgt.changing_models_gen(tblnames=ibswgt.tblname_list):
            view = ibswgt.views[tblname]
            view._change_imageset(imgsetid)
        try:
            ibswgt.back.select_imgsetid(imgsetid)
        except Exception as ex:
            ut.printex(ex, iswarning=True)
        ibswgt.set_table_tab(IMAGE_TABLE)

    def _update_imageset_tab_name(ibswgt, imgsetid, imagesettext):
        ibswgt.imageset_tabwgt._update_imageset_tab_name(imgsetid, imagesettext)

    # ------------
    # SLOT HELPERS
    # ------------

    def get_table_tab_index(ibswgt, tblname):
        view = ibswgt.views[tblname]
        index = ibswgt._tables_tab_widget.indexOf(view)
        return index

    def set_selection_status(ibswgt, key, ids):
        ids = list(ids)
        text = repr(ids)
        index = ibswgt.tablename_to_status_widget_index[key]

        if len(ids) <= 1:
            text2 = 'Selected %s:' % (ibswgt.key_to_objnice[key],)
        else:
            text2 = 'Selected %s %s:' % (
                len(ids),
                ut.pluralize(ibswgt.key_to_objnice[key], len(ids)),
            )
        ibswgt.status_widget_list[index - 1].setText(text2)
        ibswgt.status_widget_list[index].setText(text)

    def set_table_tab(ibswgt, tblname):
        """
        Programmatically change to the table-tab to either:
        Image, ImageGrid, Annotation, or Names table/tree

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.newgui import *  # NOQA
            >>> ibs, back, ibswgt, testdata_main_loop = testdata_guifront()
            >>> ibswgt.set_table_tab(gh.ANNOTATION_TABLE)
        """
        if VERBOSE_GUI:
            print('[newgui] set_table_tab: %r ' % (tblname,))
        with ut.Timer('set table tab', verbose=VERBOSE_GUI):
            index = ibswgt.get_table_tab_index(tblname)
            ibswgt._tables_tab_widget.setCurrentIndex(index)

    def select_imageset_tab(ibswgt, imgsetid):
        if VERBOSE_GUI:
            print('[newgui] select_imageset_tab imgsetid=%r' % (imgsetid,))
        if isinstance(imgsetid, six.string_types):
            # Hack
            imagesettext = imgsetid
            imgsetid = ibswgt.ibs.get_imageset_imgsetids_from_text(imagesettext)
        else:
            imagesettext = ibswgt.ibs.get_imageset_text(imgsetid)
        # ibswgt.back.select_imgsetid(imgsetid)
        ibswgt.imageset_tabwgt._add_imageset_tab(imgsetid, imagesettext)

    def spawn_edit_image_annotation_interaction_from_aid(
        ibswgt, aid, imgsetid, model=None, qtindex=None
    ):
        """
        hack for letting annots spawn image editing

        CommandLine:
            python -m wbia.gui.newgui spawn_edit_image_annotation_interaction_from_aid --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.gui.newgui import *  # NOQA
            >>> import wbia
            >>> main_locals = wbia.main(defaultdb='testdb1')
            >>> ibs, back = ut.dict_take(main_locals, ['ibs', 'back'])
            >>> ibswgt = back.ibswgt  # NOQA
            >>> aid = 4
            >>> imgsetid = 1
            >>> ibswgt.spawn_edit_image_annotation_interaction_from_aid(aid, imgsetid)
            >>> if ut.show_was_requested():
            >>>    gt.qtapp_loop(qwin=ibswgt)
        """
        gid = ibswgt.back.ibs.get_annot_gids(aid)
        if model is None:
            view = ibswgt.views[IMAGE_TABLE]
            model = view.model()
            qtindex, row = view.get_row_and_qtindex_from_id(gid)
        ibswgt.spawn_edit_image_annotation_interaction(model, qtindex, gid, imgsetid)

    def spawn_edit_image_annotation_interaction(ibswgt, model, qtindex, gid, imgsetid):
        """
        TODO: needs reimplement using more standard interaction methods
        """
        print('[newgui] Creating new annotation interaction: gid=%r' % (gid,))
        ibs = ibswgt.ibs
        # Select gid
        ibswgt.back.select_gid(gid, imgsetid, show=False)
        # Interact with gid
        nextcb, prevcb, current_gid = ibswgt._interactannot2_callbacks(model, qtindex)
        iannot2_kw = {
            'rows_updated_callback': ibswgt.update_tables,
            'next_callback': nextcb,
            'prev_callback': prevcb,
        }
        assert current_gid == gid, 'problem in next/prev updater'
        ibswgt.annot_interact = interact_annotations2.ANNOTATION_Interaction2(
            ibs, gid, **iannot2_kw
        )
        # hacky GID_PROG: TODO: FIX WITH OTHER HACKS OF THIS TYPE
        # FIXME; this should depend on the model.
        # _, row = model.view.get_row_and_qtindex_from_id(gid)
        # pt.set_figtitle('%d/%d' % (row + 1, model.rowCount()))
        level_num_rows = model._get_level_row_count(qtindex)
        level_row = model._get_level_row_index(qtindex)
        pt.set_figtitle('%d/%d' % (level_row + 1, level_num_rows))

    def make_adjacent_qtindex_callbacks(ibswgt, model, qtindex):
        r"""
        Returns:
            tuple: (current_rowid, next_callback, prev_callback)

        CommandLine:
            python -m wbia.gui.newgui --test-make_adjacent_qtindex_callbacks

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.gui.newgui import *  # NOQA
            >>> ibs, back, ibswgt, testdata_main_loop = testdata_guifront()
            >>> gid = ibs.get_valid_gids()[0]
            >>> model = ibswgt.models[gh.IMAGE_TABLE]
            >>> qtindex, row = model.get_row_and_qtindex_from_id(gid)
            >>> tup = ibswgt.make_adjacent_qtindex_callbacks(model, qtindex)
            >>> (current_rowid, next_callback, prev_callback) = tup
            >>> assert prev_callback is None, 'should not be a previous image id'
            >>> current_rowid1, next_callback1, prev_callback1 = next_callback()
            >>> assert next_callback() is None, 'race condition not prevented'
            >>> current_rowid2, next_callback2, prev_callback2 = next_callback1()
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

        python -m wbia --db lynx --imgsetid 2

        TODO: needs reimplement
        """
        # if not qtindex.isValid():
        #    raise AssertionError('Bug: qtindex got invalidated')
        #    # BUG: somewhere qtindex gets invalidated
        #    #return None, None, -1
        # HACK FOR NEXT AND PREVIOUS CLICK CALLBACKS
        # print('model.name = %r' % (model.name,))
        if model.name == gh.IMAGE_TABLE:
            cur_gid = model._get_row_id(qtindex)
        # elif model.name == gh.IMAGE_GRID:
        #     cur_gid = model._get_row_id(qtindex)
        elif model.name == gh.NAMES_TREE:
            cur_level = model._get_level(qtindex)
            if cur_level == 1:
                cur_aid = model._get_row_id(qtindex)
                cur_gid = ibswgt.ibs.get_annot_gids(cur_aid)
            else:
                raise NotImplementedError(
                    'Unknown model.name=%r, cur_level=%r' % (model.name, cur_level)
                )
        else:
            print('gh.IMAGE_TABLE = %r' % (gh.IMAGE_TABLE,))
            raise NotImplementedError('Unknown model.name =%r' % (model.name,))
        next_qtindex = model._get_adjacent_qtindex(qtindex, 1)
        prev_qtindex = model._get_adjacent_qtindex(qtindex, -1)
        numclicks = [0]  # semephore

        def make_qtindex_callback(qtindex_, type_='nextprev'):
            def _qtindex_callback():
                if numclicks[0] != 0:
                    print('race condition in %s_callback %d ' % (type_, numclicks[0]))
                    return
                numclicks[0] += 1
                # call this function again with next index
                nextcb, prevcb, new_gid1 = ibswgt._interactannot2_callbacks(
                    model, qtindex_
                )
                print('[newgui] %s_callback: new_gid1=%r' % (type_, new_gid1))
                ibswgt.annot_interact.update_image_and_callbacks(
                    new_gid1, nextcb, prevcb, do_save=True
                )
                # hacky GID_PROG: TODO: FIX WITH OTHER HACKS OF THIS TYPE
                # _, row = model.view.get_row_and_qtindex_from_id(new_gid1)
                # pt.set_figtitle('%d/%d' % (row + 1, model.rowCount()))
                level_num_rows = model._get_level_row_count(qtindex_)
                level_row = model._get_level_row_index(qtindex_)
                pt.set_figtitle('%d/%d' % (level_row + 1, level_num_rows))

            return _qtindex_callback

        if next_qtindex is not None and next_qtindex.isValid():
            next_callback = make_qtindex_callback(next_qtindex, 'next')
        else:
            next_callback = None

        if prev_qtindex is not None and prev_qtindex.isValid():
            prev_callback = make_qtindex_callback(prev_qtindex, 'prev')
        else:
            prev_callback = None

        return next_callback, prev_callback, cur_gid

    # ------------
    # SLOTS
    # ------------

    @slot_()
    def select_annot_text_editing_finished(ibswgt):
        tablename = gh.ANNOTATION_TABLE
        index = ibswgt.tablename_to_status_widget_index[tablename]
        text = ibswgt.status_widget_list[index].text()
        ibswgt.select_table_indicies_from_text(tablename, text)

    @slot_()
    def select_name_text_editing_finished(ibswgt):
        tablename = gh.NAMES_TREE
        index = ibswgt.tablename_to_status_widget_index[tablename]
        text = ibswgt.status_widget_list[index].text()
        ibswgt.select_table_indicies_from_text(tablename, text)

    @slot_()
    def select_image_text_editing_finished(ibswgt):
        tablename = gh.IMAGE_TABLE
        index = ibswgt.tablename_to_status_widget_index[tablename]
        text = ibswgt.status_widget_list[index].text()
        ibswgt.select_table_indicies_from_text(tablename, text)

    def select_table_indicies_from_text(ibswgt, tblname, text, allow_table_change=False):
        """
        Args:
            tblname - tablename of the id to parse from text

        Ignore:
            text = '[1, 2,  3,]'
               text = '51e10019-968b-5f2e-2287-8432464d7547 '

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.newgui import *  # NOQA
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
            gh.ANNOTATION_TABLE: const.ANNOTATION_TABLE,
            gh.NAMES_TREE: const.NAME_TABLE,
            gh.IMAGE_TABLE: const.IMAGE_TABLE,
        }
        backend_tablename = to_backend_tablename[tblname]
        if not ut.QUIET:
            print('[newgui]  * backend_tablename = %r' % (backend_tablename,))
        if text == '':
            text = '[]'
        try:
            # MODE1 = True
            # if MODE1:
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
            # else:
            #    id_list_ = eval(text, globals(), locals())
            #    id_list = ut.ensure_iterable(id_list_)  # NOQA
        except Exception as ex:
            ut.printex(ex, iswarning=True, keys=['text'])
        else:
            if not ut.QUIET:
                print('[newgui]  * id_list = %r' % (id_list,))
            # print(id_list)
            id_list = ibswgt.back._set_selection3(backend_tablename, id_list, mode='set')

        # Select the index if we are in the right table tab
        if len(id_list) == 1 and (
            allow_table_change or ibswgt._tables_tab_widget.current_tblname == tblname
        ):
            if not ut.QUIET:
                print('[newgui]  * attempting to select from rowid')
            # view = ibswgt.views[tblname]
            # view.select_row_from_id(id_list[0])
            ibswgt.goto_table_id(tblname, id_list[0])
        else:
            # TODO: convert the id into the ids corresponding with this tablename and move
            # to the first one
            pass

        ibswgt.back.update_selection_texts()
        # pass

    @slot_(str, int)
    def on_rows_updated(ibswgt, tblname, nRows):
        """
        When the rows are updated change the tab names
        """
        if VERBOSE_GUI:
            print('[newgui] on_rows_updated: tblname=%12r nRows=%r ' % (tblname, nRows))
        if tblname == IMAGESET_TABLE:  # Hack
            # print('... tblname == IMAGESET_TABLE, ...hack return')
            return
        tblname = str(tblname)
        TABLE_NICE = ibswgt.declare_tup[1]  # hack
        tblnice = TABLE_NICE[tblname]
        index = ibswgt.get_table_tab_index(tblname)
        text = tblnice + ' ' + str(nRows)
        # CHANGE TAB NAME TO SHOW NUMBER OF ROWS
        ibswgt._tables_tab_widget.setTabText(index, text)

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

        CommandLine:
            python -m wbia --db WS_ALL --imgsetid 2 --select-name=A-003
        """
        if not qtindex.isValid():
            return

        model = qtindex.model()
        tblview = ibswgt.views[model.name]
        context_options = []
        qtindex_list = tblview.selectedIndexes()
        id_list = [model._get_row_id(_qtindex) for _qtindex in qtindex_list]
        level_list = [model._get_level(_qtindex) for _qtindex in qtindex_list]
        level2_ids_ = ut.group_items(id_list, level_list)
        level2_ids = {
            level: ut.unique_ordered(ids) for level, ids in six.iteritems(level2_ids_)
        }

        ibs = ibswgt.back.ibs
        back = ibswgt.back

        col = qtindex.column()
        setter_func = model.col_setter_list[col]
        if setter_func is not None:
            # Make a copy of the list
            id_list_ = id_list[:]
            clone_highlight = len(id_list_) > 1
            if not clone_highlight:
                id_list_ = [_.get_id() for _ in model.level_index_list]
            # Getter and current value
            getter_func = model.col_getter_list[col]
            current_id = model._get_row_id(qtindex)
            current_value = getter_func(current_id)
            current_list = [current_value] * len(id_list_)
            # Add to context menus

            def _setter_refresh(id_list_, current_list, tablename):
                args = (
                    current_list[0],
                    len(id_list_),
                )
                if not back.are_you_sure('Confirm cloning value %s to %d rows?' % args):
                    return
                setter_func(id_list_, current_list)
                ibswgt.update_tables(tblnames=[tablename], clear_view_selection=False)

            if clone_highlight:
                context_options += [
                    (
                        'Clone selected value to highlighted rows',
                        lambda: _setter_refresh(id_list_, current_list, model.name),
                    ),
                    ('----', lambda: None),
                ]
            else:
                context_options += [
                    (
                        'Clone selected value to all rows',
                        lambda: _setter_refresh(id_list_, current_list, model.name),
                    ),
                    ('----', lambda: None),
                ]

        def build_annot_context_options(ibswgt, ibs, aid_list, imgsetid, **kwargs):
            context_options = []
            if len(aid_list) == 1:
                aid = aid_list[0]
                if kwargs.get('goto_image', True):
                    context_options += [
                        (
                            'Go to image',
                            lambda: ibswgt.goto_table_id(
                                IMAGE_TABLE, ibswgt.back.ibs.get_annot_gids(aid)
                            ),
                        )
                    ]
                if kwargs.get('goto_annot', True):
                    context_options += [
                        (
                            'Go to annot',
                            lambda: ibswgt.goto_table_id(gh.ANNOTATION_TABLE, aid),
                        )
                    ]
                if kwargs.get('goto_name', True):
                    context_options += [
                        (
                            'Go to name',
                            lambda: ibswgt.goto_table_id(
                                NAMES_TREE, ibswgt.back.ibs.get_annot_nids(aid)
                            ),
                        ),
                    ]
                if kwargs.get('canedit', True):
                    context_options += [
                        (
                            'Edit Annotation in Image',
                            lambda: ibswgt.spawn_edit_image_annotation_interaction_from_aid(
                                aid, imgsetid
                            ),
                        ),
                    ]
                    context_options += [
                        (
                            "Clone annotation's species to ImageSet",
                            lambda: ibswgt.back.copy_species_to_imageset(aid, imgsetid),
                        ),
                    ]

                context_options += [
                    ('----', lambda: None),
                    (
                        'View annotation in Web',
                        # lambda: ibswgt.back.select_aid(aid, imgsetid, show=True)),
                        lambda: ibswgt.back.show_annotation(aid, web=True),
                    ),
                    (
                        'View image in Web',
                        lambda: ibswgt.back.select_gid_from_aid(
                            aid, imgsetid, show=True, web=True
                        ),
                    ),
                    (
                        'Set annotation species',
                        lambda: ibswgt.back.override_all_annotation_species(aid_list),
                    ),
                    ('----', lambda: None),
                    ("Remove annotation's name", lambda: ibswgt.back.unset_names([aid]),),
                    ('Delete annotation', lambda: ibswgt.back.delete_annot(aid_list)),
                    ('----', lambda: None),
                ]
                from wbia.viz.interact import interact_chip
                from wbia import viz

                context_options += interact_chip.build_annot_context_options(
                    ibswgt.back.ibs,
                    aid,
                    refresh_func=viz.draw,
                    with_interact_image=False,
                )
            else:
                context_options += [
                    (
                        'View annotations in Web',
                        lambda: ibswgt.back.show_aid_list_in_web(aid_list),
                    ),
                    (
                        'Set annotation species',
                        lambda: ibswgt.back.override_all_annotation_species(aid_list),
                    ),
                    (
                        "Unset annotations' names",
                        lambda: ibswgt.back.unset_names(aid_list),
                    ),
                    ('Delete annotations', lambda: ibswgt.back.delete_annot(aid_list)),
                ]
            return context_options

        def name_context_options(ibswgt, ibs, nid_list, aid_list, imgsetid, **kwargs):
            context_options = []
            if len(aid_list) == 1:
                aid = aid_list[0]
                context_options += build_annot_context_options(
                    ibswgt, ibs, [aid], imgsetid
                )
            if len(aid_list) > 0:

                def set_annot_names_to_same_new_name(ibswgt, aid_list):
                    ibswgt.back.ibs.set_annot_names_to_same_new_name(aid_list)
                    ibswgt.update_tables(tblnames=[gh.NAMES_TREE])

                context_options += [
                    (
                        'Rename annots (%s) to new name'
                        % ut.list_str_summarized(aid_list, 'aid_list'),
                        lambda: set_annot_names_to_same_new_name(ibswgt, aid_list),
                    ),
                ]
            if len(nid_list) > 0:

                def run_splits(ibs, nid_list):
                    print('Checking for splits')
                    aids_list = ibs.get_name_aids(nid_list)
                    aid_list = sorted(list(set(ut.flatten(aids_list))))
                    back.run_annot_splits(aid_list)

                def export_nids(ibs, nid_list):
                    from wbia.dbio import export_subset

                    if not back.are_you_sure(
                        'Confirm export of nid_list=%r' % (nid_list,)
                    ):
                        return
                    export_subset.export_names(ibs, nid_list)

                def create_new_imageset_from_names_(ibs, nid_list):
                    ibs.create_new_imageset_from_names(nid_list)
                    ibswgt.update_tables([gh.IMAGESET_TABLE], clear_view_selection=False)

                context_options += [
                    (
                        'View name(s) in Web',
                        lambda: ibswgt.back.show_nid_list_in_web(nid_list),
                    ),
                    ('----', lambda: None),
                    ('Check for splits', lambda: run_splits(ibs, nid_list)),
                    ('Export names', lambda: export_nids(ibs, nid_list)),
                    (
                        'Create ImageSet From Name(s)',
                        lambda: create_new_imageset_from_names_(ibs, nid_list),
                    ),
                ]

                from wbia.viz.interact import interact_name

                context_options += interact_name.build_name_context_options(
                    ibswgt.back.ibs, nid_list
                )
            elif len(nid_list) == 0:
                # from wbia.viz.interact import interact_name
                # context_options += interact_name.build_name_context_options(
                #    ibswgt.back.ibs, nid_list)
                # print('nutin')
                pass
            return context_options

        def build_image_context_options(ibswgt, ibs, gid_list, imgsetid, **kwargs):
            current_imagesettext = ibswgt.back.ibs.get_imageset_text(imgsetid)
            context_options = []
            # Conditional context menu

            def view_images_in_directory(gid_list):
                images = ibs.images(gid_list)
                fpaths = images.paths
                ut.view_file_in_directory(fpaths)

            def view_images_in_web(gid_list):
                ibswgt.back.show_images_in_web(gid_list)

            fmt = {
                'images': ut.pluralize('image', len(gid_list)),
                "images's": ut.pluralize("image's", len(gid_list)),
                'annotations': ut.pluralize('annotation', len(gid_list)),
                'times': ut.pluralize('time', len(gid_list)),
            }

            context_options = [
                (
                    "Edit {images's} {times}".format(**fmt),
                    lambda: ibswgt.edit_image_time(gid_list),
                ),
                (
                    'View {images} in Web'.format(**fmt),
                    lambda: view_images_in_web(gid_list),
                ),
                (
                    'View {images} in Directory'.format(**fmt),
                    lambda: view_images_in_directory(gid_list),
                ),
                ('---', lambda: None),
                (
                    'Add annotation from entire {images}'.format(**fmt),
                    lambda: ibswgt.back.add_annotation_from_image(gid_list),
                ),
                (
                    'Run detection on {images} (can cause duplicates)'.format(**fmt),
                    lambda: ibswgt.back.run_detection_on_images(gid_list),
                ),
            ]

            if len(gid_list) == 1:
                gid = gid_list[0]
                imgsetid = model.imgsetid
                aid_list = ibs.get_image_aids(gid)

                annot_options = [
                    (
                        'Options aid=%r' % (aid,),
                        build_annot_context_options(
                            ibswgt, ibs, [aid], imgsetid, goto_image=False
                        ),
                    )
                    for aid in aid_list
                ]
                if len(aid_list) == 1:
                    annot_option_item = (
                        'Annot Options (aid=%r)' % (aid_list[0],),
                        annot_options[0][1],
                    )
                else:
                    annot_option_item = ('Annot Options', annot_options)
                if kwargs.get('goto_image_in_imgtbl', False):
                    context_options += [
                        (
                            'Go to image in Images Table',
                            lambda: ibswgt.goto_table_id(IMAGE_TABLE, gid),
                        ),
                    ]

                context_options += [
                    (
                        'View image in Matplotlib',
                        lambda: ibswgt.back.select_gid(
                            gid, imgsetid, show=True, web=False
                        ),
                    ),
                    (
                        'View detection image (Hough) [dev]',
                        lambda: ibswgt.back.show_hough_image_(gid),
                    ),
                    annot_option_item,
                ]
            # Special condition for imagesets
            if current_imagesettext != const.NEW_IMAGESET_IMAGESETTEXT:
                context_options += [
                    ('----', lambda: None),
                    (
                        'Move to new imageset',
                        lambda: ibswgt.back.send_to_new_imageset(gid_list, mode='move'),
                    ),
                    (
                        'Copy to new imageset',
                        lambda: ibswgt.back.send_to_new_imageset(gid_list, mode='copy'),
                    ),
                ]
            if current_imagesettext != const.UNGROUPED_IMAGES_IMAGESETTEXT:
                context_options += [
                    ('----', lambda: None),
                    (
                        'Remove from imageset',
                        lambda: ibswgt.back.remove_from_imageset(gid_list),
                    ),
                ]

            from wbia.viz import viz_graph2

            context_options += [
                ('----', lambda: None),
                (
                    'Graph interaction (Annots)',
                    lambda: viz_graph2.make_qt_graph_interface(
                        # ibs, nids=ut.unique(ut.flatten(ibs.get_image_nids(gid_list))),
                        ibs,
                        gids=gid_list,
                        graph_tab=True,
                    ),
                ),
            ]

            # Continue the conditional context menu
            if len(gid_list) == 1:
                # We get gid from above
                context_options += [
                    ('----', lambda: None),
                    (
                        "Delete image's annotations",
                        lambda: ibswgt.back.delete_image_annotations([gid]),
                    ),
                    (
                        "Set image's annotations species",
                        lambda: ibswgt.back.override_all_annotation_species(gids=[gid]),
                    ),
                    ('Delete image', lambda: ibswgt.back.delete_image(gid)),
                ]
            else:
                context_options += [
                    ('----', lambda: None),
                    (
                        "Delete images' annotations",
                        lambda: ibswgt.back.delete_image_annotations(gid_list),
                    ),
                    (
                        "Set images' annotations species",
                        lambda: ibswgt.back.override_all_annotation_species(
                            gids=gid_list
                        ),
                    ),
                    ('Delete images', lambda: ibswgt.back.delete_image(gid_list)),
                ]
            return context_options

        # ---- IMAGESET CONTEXT ----
        if model.name == IMAGESET_TABLE:
            # This is for the benefit of merge imagesets
            merge_destination_id = model._get_row_id(qtindex)
            imagesettext = ibswgt.back.ibs.get_imageset_text(merge_destination_id)
            imgsetid_list = level2_ids[0]
            # Conditional context menu
            # TODO: remove duplicate code
            if len(imgsetid_list) == 1:
                context_options += [
                    (
                        'View imageset in Web',
                        lambda: ibswgt.back.show_imgsetid_list_in_web(imgsetid_list),
                    ),
                    (
                        "Turk imageset's detections in Web",
                        lambda: ibswgt.back.show_imgsetid_detection_turk_in_web(
                            imgsetid_list
                        ),
                    ),
                    (
                        "Turk imageset's annotations in Web",
                        lambda: ibswgt.back.show_imgsetid_annotation_turk_in_web(
                            imgsetid_list
                        ),
                    ),
                    ('----', lambda: None),
                    (
                        'Run detection on imageset (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_imageset(imgsetid_list),
                    ),
                    (
                        'Merge %d imageset into %s'
                        % (len(imgsetid_list), (imagesettext)),
                        lambda: ibswgt.back.merge_imagesets(
                            imgsetid_list, merge_destination_id
                        ),
                    ),
                    ('Copy imageset', lambda: ibswgt.back.copy_imageset(imgsetid_list)),
                    (
                        'Export imageset',
                        lambda: ibswgt.back.export_imagesets(imgsetid_list),
                    ),
                    ('----', lambda: None),
                    (
                        'Delete imageset',
                        lambda: ibswgt.back.delete_imageset(imgsetid_list),
                    ),
                    ('----', lambda: None),
                    (
                        'Delete imageset AND images',
                        lambda: ibswgt.back.delete_imageset_and_images(imgsetid_list),
                    ),
                    ('----', lambda: None),
                    (
                        'Mark imageset as Shipped to WildBook',
                        lambda: ibswgt.back.mark_imageset_as_shipped(imgsetid_list),
                    ),
                ]
            else:
                context_options += [
                    (
                        'Run detection on imagesets (can cause duplicates)',
                        lambda: ibswgt.back.run_detection_on_imageset(imgsetid_list),
                    ),
                    ('Copy imageset', lambda: ibswgt.back.copy_imageset(imgsetid_list)),
                    (
                        'Merge %d imagesets into %s'
                        % (len(imgsetid_list), (imagesettext)),
                        lambda: ibswgt.back.merge_imagesets(
                            imgsetid_list, merge_destination_id
                        ),
                    ),
                    ('----', lambda: None),
                    (
                        'Delete imagesets',
                        lambda: ibswgt.back.delete_imageset(imgsetid_list),
                    ),
                    ('----', lambda: None),
                    (
                        'Delete imagesets AND images',
                        lambda: ibswgt.back.delete_imageset_and_images(imgsetid_list),
                    ),
                    # ('export imagesets', lambda: ibswgt.back.export_imagesets(imgsetid_list)),
                ]
        # ---- IMAGE CONTEXT ----
        elif model.name == IMAGE_TABLE:
            gid_list = level2_ids[0]
            imgsetid = ibswgt.back.get_selected_imgsetid()
            context_options += build_image_context_options(
                ibswgt, ibs, gid_list, imgsetid
            )
        # ---- IMAGE GRID CONTEXT ----
        elif model.name == IMAGE_GRID:
            gid_list = level2_ids[0]
            imgsetid = ibswgt.back.get_selected_imgsetid()
            context_options += build_image_context_options(
                ibswgt, ibs, gid_list, imgsetid, goto_image_in_imgtbl=True
            )
        # ---- ANNOTATION CONTEXT ----
        elif model.name == gh.ANNOTATION_TABLE:
            aid_list = level2_ids[0]
            # Conditional context menu
            # TODO: UNIFY COMMMON CONTEXT MENUS
            context_options += build_annot_context_options(
                ibswgt, ibs, aid_list, model.imgsetid, goto_annot=False
            )
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
                imgsetid = model.imgsetid
                context_options += name_context_options(
                    ibswgt, ibs, nid_list, aid_list, imgsetid
                )
        # Show the context menu
        # ut.print_list(context_options, nl=2)
        if len(context_options) > 0:
            gt.popup_menu(tblview, pos, context_options)

    @slot_(QtCore.QModelIndex)
    def on_doubleclick(ibswgt, qtindex):
        """
        Double clicking anywhere in the GUI

        CommandLine:
            python -m wbia --db lynx --imgsetid 2 --select-name=goku
        """
        print('\n+--- DOUBLE CLICK ---')
        if not qtindex.isValid():
            print('[doubleclick] invalid qtindex')
            return
        model = qtindex.model()
        id_ = model._get_row_id(qtindex)
        if model.name == IMAGESET_TABLE:
            imgsetid = id_
            ibswgt.select_imageset_tab(imgsetid)
        else:
            imgsetid = model.imgsetid
            if (model.name == IMAGE_TABLE) or (model.name == IMAGE_GRID):
                gid = id_
                ibswgt.spawn_edit_image_annotation_interaction(
                    model, qtindex, gid, imgsetid
                )
            elif model.name == gh.ANNOTATION_TABLE:
                aid = id_
                ibswgt.back.select_aid(aid, imgsetid)
            elif model.name == NAME_TABLE:
                nid = id_
                ibswgt.back.select_nid(nid, imgsetid)
            elif model.name == NAMES_TREE:
                level = model._get_level(qtindex)
                if level == 0:
                    nid = id_
                    ibswgt.back.select_nid(nid, imgsetid, show=True)
                elif level == 1:
                    aid = id_
                    ibswgt.spawn_edit_image_annotation_interaction_from_aid(
                        aid, imgsetid, model, qtindex
                    )
                    # ibswgt.back.select_aid(aid, imgsetid, show=True)

    # @slot_(list)
    def imagesDropped(ibswgt, url_list):
        r"""
        image drag and drop event

        CommandLine:
            python -m wbia.gui.newgui imagesDropped --show

        Example:
            >>> # xdoctest: +REQUIRES(--gui)
            >>> from wbia.gui.newgui import *  # NOQA
            >>> ibs, back, ibswgt, testdata_main_loop = testdata_guifront('hstest')
            >>> url_list = [ut.grab_test_imgpath('carl.jpg'), ut.grab_test_imgpath('lena.png')]
            >>> #url_list += [ut.truepath('~/Downloads/Clutter/wd_peter2.zip')]
            >>> url = url_list[0]
            >>> ut.quit_if_noshow()
            >>> ibswgt.imagesDropped(url_list)
            >>> testdata_main_loop(globals(), locals())
        """
        print('[drop_event] url_list=%r' % (url_list,))
        has_zipext = ut.partial(ut.fpath_has_ext, exts=['.zip'])
        gpath_list = list(filter(ut.fpath_has_imgext, url_list))
        dir_list = list(filter(isdir, url_list))
        zipfile_list = list(filter(has_zipext, url_list))
        old = False
        if old:
            if len(dir_list) > 0:
                options = ['No', 'Yes']
                title = 'Non-Images dropped'
                msg = 'Recursively import from directories?'
                ans = gt.user_option(ibswgt, msg=msg, title=title, options=options)
                if ans == 'Yes':
                    unflat_gpaths = [
                        ut.list_images(dir_, fullpath=True, recursive=True)
                        for dir_ in dir_list
                    ]
                    flat_gpaths = ut.flatten(unflat_gpaths)
                    flat_unix_gpaths = list(map(ut.unixpath, flat_gpaths))
                    gpath_list.extend(flat_unix_gpaths)
                else:
                    return
            print('[drop_event] gpath_list=%r' % (gpath_list,))
            if len(gpath_list) > 0:
                ibswgt.back.import_images_from_file(gpath_list=gpath_list)
        else:
            from wbia.dbio import ingest_database

            ibs = ibswgt.back.ibs
            ingestable = ingest_database.Ingestable2(
                ibs.get_dbdir(), gpath_list, dir_list, zipfile_list
            )
            num_gpaths = len(ingestable.imgpath_list)
            num_dpaths = len(ingestable.imgdir_list)
            num_zips = len(ingestable.zipfile_list)
            confirm_list = []
            if num_gpaths > 0:
                confirm_list += [ut.quantstr('image file', num_gpaths)]
            if num_dpaths > 0:
                confirm_list += [
                    'recursively from ' + ut.quantstr('directory', num_dpaths, 's')
                ]
            if num_zips > 0:
                confirm_list += [ut.quantstr('zip file', num_zips, 's')]
            confirm_msg = 'Import from: ' + ut.conj_phrase(confirm_list, 'and') + '.'
            # gt.rrrr()
            config = ingestable.ingest_config
            # cfg = config
            dlg = gt.ConfigConfirmWidget.as_dialog(
                ibswgt, title='Confirm Import Images', msg=confirm_msg, config=config
            )
            dlg.resize(700, 500)
            self = dlg.widget
            dlg.exec_()
            print('config = %r' % (config,))
            updated_config = self.config  # NOQA
            print('updated_config = %r' % (updated_config,))
            gid_list = ingestable.execute(ibs=ibs)
            ibswgt.back._process_new_images(
                refresh=True, gid_list=gid_list, clock_offset=False
            )

    def select_table_id(ibswgt, table_key, level, id_, imgsetid):
        select_func_dict = {
            (IMAGE_TABLE, 0): ibswgt.back.select_gid,
            (IMAGE_GRID, 0): ibswgt.back.select_gid,
            (gh.ANNOTATION_TABLE, 0): ibswgt.back.select_aid,
            (NAME_TABLE, 0): ibswgt.back.select_nid,
            (NAMES_TREE, 0): ibswgt.back.select_nid,
            (NAMES_TREE, 1): ibswgt.back.select_aid,
        }
        select_func = select_func_dict[(table_key, level)]
        select_func(id_, imgsetid, show=False)

    def edit_image_time(ibswgt, gid_list):
        """

        CommandLine:
            python -m wbia.gui.newgui --exec-edit_image_time --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.gui.newgui import *  # NOQA
            >>> ibs, back, ibswgt, testdata_main_loop = testdata_guifront('testdb3')
            >>> #ibs, back, ibswgt, testdata_main_loop = testdata_guifront('lynx')
            >>> ibswgt.edit_image_time([277, 630])
            >>> testdata_main_loop(globals(), locals())
        """
        from wbia.gui import clock_offset_gui

        ibswgt.co_wgt = clock_offset_gui.ClockOffsetWidget(
            ibswgt.ibs, gid_list, hack=True
        )
        ibswgt.co_wgt.show()

    def filter_annotation_table(ibswgt):
        r"""
        TODO:  Finish implementation

        CommandLine:
            python -m wbia.gui.newgui --test-filter_annotation_table --show --db lynx --imgsetid 2

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.gui.newgui import *  # NOQA
            >>> ibs, back, ibswgt, testdata_main_loop = testdata_guifront('testdb3')
            >>> #ibs, back, ibswgt, testdata_main_loop = testdata_guifront('PZ_Master1')
            >>> result = ibswgt.filter_annotation_table()
            >>> print(result)
            >>> testdata_main_loop(globals(), locals())
        """
        from functools import partial

        # ibs.filter_annots_general()

        ibs = ibswgt.back.ibs

        # ibs.filterannots_by_tags(aid_list)
        print('\n------FILTERING ANNOTS\n\n')

        # annotmatch_rowid_list = ibs._get_all_annotmatch_rowids()
        # isscenerymatch_list = ibs.get_annotmatch_is_scenerymatch(annotmatch_rowid_list)
        # ut.take(isscenerymatch_list, ut.list_where(isscenerymatch_list))

        # Applies annotation based filtering to the annotation table
        # filter_kw = dict(any_matches='.*error.*', been_adjusted=True)
        filter_kw = dict(been_adjusted=True)
        # filter_kw = dict(require_timestamp=True)
        filter_fn = partial(ibs.filter_annots_general, filter_kw=filter_kw)

        model = ibswgt.models[gh.ANNOTATION_TABLE]  # NOQA
        model.set_ider_filters([filter_fn])
        with ChangeLayoutContext([model]):
            model._update_rows(rebuild_structure=True)


# #####################
# ##### Testing #######
# #####################


def testdata_guifront(defaultdb='testdb1'):
    import wbia

    main_locals = wbia.main(defaultdb=defaultdb, allow_newdir=True)
    ibs, back = ut.dict_take(main_locals, ['ibs', 'back'])
    ibswgt = back.ibswgt  # NOQA
    globals__ = globals()
    locals__ = locals()

    def testdata_main_loop(globals_=globals__, locals_=locals__):
        locals_ = locals_.copy()
        globals_ = globals_.copy()
        locals_.update(locals__)
        globals_.update(globals__)
        if '--cmd' in sys.argv:
            gt.qtapp_loop(qwin=ibswgt, ipy=True)
            six.exec_(ut.ipython_execstr(), globals_, locals_)
        elif ut.show_was_requested():
            gt.qtapp_loop(qwin=ibswgt)

    return ibs, back, ibswgt, testdata_main_loop


def testfunc():
    r"""
    CommandLine:
        python -m wbia.gui.newgui --test-testfunc --show
        python -m wbia.gui.newgui --test-testfunc --cmd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.gui.newgui import *  # NOQA
        >>> result = testfunc()
        >>> print(result)
    """
    ibs, back, ibswgt, testdata_main_loop = testdata_guifront()
    view = ibswgt.views[gh.IMAGE_TABLE]
    testdata_main_loop(globals(), locals())


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.gui.newgui
        python -m wbia.gui.newgui --allexamples
        python -m wbia.gui.newgui --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
