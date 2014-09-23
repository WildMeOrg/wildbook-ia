#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from six.moves import zip, map
import functools  # NOQA
from guitool.__PYQT__ import QtGui, QtCore
from guitool.__PYQT__.QtCore import Qt
from guitool.__PYQT__.QtGui import QSizePolicy
from guitool import signal_, slot_, checks_qt_error, ChangeLayoutContext  # NOQA
from ibeis.control import IBEISControl
from ibeis import ibsfuncs
from ibeis.gui import guiheaders as gh
from ibeis.gui import guimenus
from ibeis.viz.interact import interact_annotations2
from ibeis import constants
from ibeis.gui.guiheaders import (IMAGE_TABLE, IMAGE_GRID, ANNOTATION_TABLE,
                                  NAME_TABLE, NAMES_TREE, ENCOUNTER_TABLE)
from ibeis.gui.models_and_views import (IBEISStripeModel, IBEISTableView,
                                        IBEISItemModel, IBEISTreeView,
                                        EncTableModel, EncTableView,
                                        IBEISTableWidget, IBEISTreeWidget,
                                        EncTableWidget)
import guitool
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[newgui]')


IBEIS_WIDGET_BASE = QtGui.QWidget


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
        # enc_tabwgt._add_enc_tab(-1, constants.ALL_IMAGE_ENCTEXT)

    @slot_(int)
    def _on_change(enc_tabwgt, index):
        """ Switch to the current encounter tab """
        if 0 <= index and index < len(enc_tabwgt.eid_list):
            eid = enc_tabwgt.eid_list[index]
            #if utool.VERBOSE:
            print('[ENCTAB.ONCHANGE] eid = %r' % (eid,))
            enc_tabwgt.ibswgt._change_enc(eid)
        else:
            enc_tabwgt.ibswgt._change_enc(-1)

    @slot_(int)
    def _close_tab(enc_tabwgt, index):
        if enc_tabwgt.eid_list[index] is not None:
            enc_tabwgt.eid_list.pop(index)
            enc_tabwgt.removeTab(index)

    @slot_()
    def _close_all_tabs(enc_tabwgt):
        while len(enc_tabwgt.eid_list) > 0:
            index = 0
            enc_tabwgt.eid_list.pop(index)
            enc_tabwgt.removeTab(index)

    @slot_(int)
    def _close_tab_with_eid(enc_tabwgt, eid):
        try:
            index = enc_tabwgt.eid_list.index(eid)
            enc_tabwgt._close_tab(index)
        except:
            pass

    def _add_enc_tab(enc_tabwgt, eid, enctext):
        # <HACK>
        # if enctext == constants.ALL_IMAGE_ENCTEXT:
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


class IBEISGuiWidget(IBEIS_WIDGET_BASE):
    #@checks_qt_error
    def __init__(ibswgt, back=None, ibs=None, parent=None):
        IBEIS_WIDGET_BASE.__init__(ibswgt, parent)
        ibswgt.ibs = ibs
        ibswgt.back = back
        # Sturcutres that will hold models and views
        ibswgt.models       = {}
        ibswgt.views        = {}
        #ibswgt.widgets      = {}
        ibswgt.tblname_list = [
            IMAGE_TABLE,
            IMAGE_GRID,
            ANNOTATION_TABLE,
            #NAME_TABLE,
            NAMES_TREE
        ]
        # FIXME: Duplicate models
        # Create models and views
        # Define the abstract item models and views for the tables
        ibswgt.modelview_defs = [
            (IMAGE_TABLE,      IBEISTableWidget, IBEISItemModel, IBEISTableView),
            (IMAGE_GRID,       IBEISTableWidget, IBEISStripeModel, IBEISTableView),
            (ANNOTATION_TABLE, IBEISTableWidget, IBEISItemModel, IBEISTableView),
            #(NAME_TABLE,       IBEISTableWidget, IBEISItemModel, IBEISTableView),
            (NAMES_TREE,       IBEISTreeWidget,  IBEISItemModel,  IBEISTreeView),
            (ENCOUNTER_TABLE,  EncTableWidget,   EncTableModel,   EncTableView),
        ]
        ibswgt.super_tblname_list = ibswgt.tblname_list + [ENCOUNTER_TABLE]
        # Create and layout components
        ibswgt._init_components()
        ibswgt._init_layout()
        # Connect signals and slots
        ibswgt._connect_signals_and_slots()
        # Connect the IBEIS control
        print("WIDGET: %r" % (ibswgt.ibs))
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
                                                  visible=utool.get_flag('--guilog'), verticalStretch=6)
        ibswgt.progressBar = guitool.newProgressBar(ibswgt, visible=False, verticalStretch=1)
        # New widget has black magic (for implicit layouts) in it
        ibswgt.status_wgt  = guitool.newWidget(ibswgt, Qt.Vertical,
                                               verticalStretch=6,
                                               horizontalSizePolicy=QSizePolicy.Maximum)

        ibswgt.statusBar = QtGui.QHBoxLayout(ibswgt)
        _NEWLBL = functools.partial(guitool.newLabel, ibswgt)
        ibswgt.statusLabel_list = [
            _NEWLBL(''),
            _NEWLBL('Status Bar'),
            _NEWLBL(''),
            _NEWLBL(''),
        ]

        ibswgt.buttonBars = []
        _NEWBUT = functools.partial(guitool.newButton, ibswgt)
        _COMBO  = functools.partial(guitool.newComboBox, ibswgt)
        back = ibswgt.back
        #_SEP = lambda: None

        ibswgt.query_button = _NEWBUT('Run Identification',
                                      ibswgt.back.compute_queries,
                                      bgcolor=(150, 150, 255),
                                      fgcolor=(0, 0, 0))

        detection_combo_box_options = [
            # Text              # Value
            ('Select Species',  'none'),
        ] + list(zip(constants.SPECIES_NICE, constants.VALID_SPECIES))
        ibswgt.species_combo = _COMBO(detection_combo_box_options,
                                      ibswgt.back.change_detection_species)

        ibswgt.species_button = _NEWBUT('Update Encounter Species',
                                        ibswgt.back.encounter_set_species,
                                        bgcolor=(150, 255, 150))

        ibswgt.reviewed_button = _NEWBUT('Set Encounter as Reviewed',
                                         ibswgt.back.encounter_reviewed_all_images,
                                         bgcolor=(0, 232, 211))

        ibswgt.import_button = _NEWBUT('Step 1) Import Images\n(via files)',
                                       back.import_images_from_file,
                                       bgcolor=(235, 200, 200),)

        ibswgt.encounter_button = _NEWBUT('Step 2) Group Images into Encounters',
                                          ibswgt.back.compute_encounters,
                                          bgcolor=(255, 255, 150))

        ibswgt.detect_button = _NEWBUT('Step 3) Detect',
                                       ibswgt.back.run_detection_coarse,
                                       bgcolor=(150, 255, 150))

        detection_combo_box_options = [
            # Text              # Value
            ('Step 4) Intra Encounter', constants.INTRA_ENC_KEY),
            ('Step 5) Vs Exemplars',    constants.VS_EXEMPLARS_KEY),
        ]
        ibswgt.querydb_combo = _COMBO(detection_combo_box_options,
                                      ibswgt.back.change_query_mode)

        ibswgt.button_list = [
            [
                ibswgt.import_button,

                #_NEWBUT('Import Images\n(via dir)',
                #        back.import_images_from_dir,
                #        bgcolor=(235, 200, 200)),
                #_NEWBUT('Import Images\n(via dir + size filter)',
                #        bgcolor=(235, 200, 200)),

                #_NEWBUT('Filter Images (GIST)'),

                ibswgt.encounter_button,

                ibswgt.species_combo,

                ibswgt.detect_button,

            ],
            [
                #_NEWBUT('Review Detections',
                #        ibswgt.back.review_detections,
                #        bgcolor=(170, 250, 170)),
                ibswgt.species_button,

                ibswgt.reviewed_button,

                ibswgt.querydb_combo,

                ibswgt.query_button,

                #_NEWBUT('Identify\n(vs exemplar database)',
                #        ibswgt.back.compute_queries_vs_exemplar,
                #        bgcolor=(150, 150, 255),
                #        fgcolor=(0, 0, 0)),

                #_NEWBUT('Review Recognitions',
                #        ibswgt.back.review_queries,
                #        bgcolor=(170, 170, 250),
                #        fgcolor=(0, 0, 0)),

                # _SEP(),

                #_NEWBUT('Delete Encounters', ibswgt.back.delete_all_encounters,
                #        bgcolor=(255, 0, 0),
                #        fgcolor=(0, 0, 0)),
            ]
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
        ibswgt.hsplitter.addWidget(ibswgt._tab_table_wgt)
        # Horizontal Lower
        ibswgt.status_wgt.addWidget(ibswgt.outputLog)
        ibswgt.status_wgt.addWidget(ibswgt.progressBar)
        # Add buttonbar
        for row in ibswgt.button_list:
            ibswgt.buttonBars.append(QtGui.QHBoxLayout(ibswgt))
            ibswgt.status_wgt.addLayout(ibswgt.buttonBars[-1])
            for button in row:
                ibswgt.buttonBars[-1].addWidget(button)
        ibswgt.status_wgt.addLayout(ibswgt.statusBar)
        # Add statusbar
        for widget in ibswgt.statusLabel_list:
            ibswgt.statusBar.addWidget(widget)

    def set_status_text(ibswgt, index, text):
        #printDBG('set_status_text[%r] = %r' % (index, text))
        ibswgt.statusLabel_list[index].setText(text)

    def changing_models_gen(ibswgt, tblnames=None):
        """
        Loops over tablenames emitting layoutChanged at the end for each
        """
        if tblnames is None:
            tblnames = ibswgt.super_tblname_list
        print('[newgui] change models: %r' % (tblnames,))
        model_list = [ibswgt.models[tblname] for tblname in tblnames]
        #model_list = [ibswgt.models[tblname] for tblname in tblnames if ibswgt.views[tblname].isVisible()]
        with ChangeLayoutContext(model_list):
            for tblname in tblnames:
                yield tblname

    def update_tables(ibswgt, tblnames=None, clear_view_selection=True):
        """ forces changing models """
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

    def connect_ibeis_control(ibswgt, ibs):
        """ Connects a new ibscontroler to the models """
        print('[newgui] connecting ibs control')
        if ibs is None:
            print('[newgui] invalid ibs')
            title = 'No Database Opened'
            ibswgt.setWindowTitle(title)
        else:
            print('[newgui] Connecting valid ibs=%r' % ibs.get_dbname())
            # Give the frontend the new control
            ibswgt.ibs = ibs
            with utool.Timer('update special'):
                ibs.update_special_encounters()
            # Update the api models to use the new control
            with utool.Timer('make headers'):
                header_dict = gh.make_ibeis_headers_dict(ibswgt.ibs)
            title = ibsfuncs.get_title(ibswgt.ibs)
            ibswgt.setWindowTitle(title)
            if utool.VERBOSE:
                print('[newgui] Calling model _update_headers')
            block_wgt_flag = ibswgt._tab_table_wgt.blockSignals(True)
            with utool.Timer('update models'):
                for tblname in ibswgt.changing_models_gen(ibswgt.super_tblname_list):
                    model = ibswgt.models[tblname]
                    view = ibswgt.views[tblname]
                    header = header_dict[tblname]
                    #widget = ibswgt.widgets[tblname]
                    #widget.change_headers(header)
                    block_model_flag = model.blockSignals(True)
                    model._update_headers(**header)
                    view._update_headers(**header)  # should use model headers
                    model.blockSignals(block_model_flag)
                    #view.infer_delegates_from_model()
                for tblname in ibswgt.super_tblname_list:
                    view = ibswgt.views[tblname]
                    view.hide_cols()
            ibswgt._tab_table_wgt.blockSignals(block_wgt_flag)
            ibswgt._change_enc(-1)

    def setWindowTitle(ibswgt, title):
        parent_ = ibswgt.parent()
        if parent_ is not None:
            parent_.setWindowTitle(title)
        else:
            IBEIS_WIDGET_BASE.setWindowTitle(ibswgt, title)

    def _change_enc(ibswgt, eid):
        print('_change_enc(%r)' % eid)
        for tblname in ibswgt.changing_models_gen(tblnames=ibswgt.tblname_list):
            ibswgt.views[tblname]._change_enc(eid)
            #ibswgt.models[tblname]._change_enc(eid)  # the view should take care of this call
        try:
            #if eid is None:
            #    # HACK
            #    enctext = constants.ALL_IMAGE_ENCTEXT
            #else:
            #    enctext = ibswgt.ibs.get_encounter_enctext(eid)
            ibswgt.back.select_eid(eid)
            ibswgt.species_combo.setDefault(ibswgt.ibs.cfg.detect_cfg.species)
            #text_list = [
            #    'Identify Mode: Within-Encounter (%s vs. %s)' % (enctext, enctext),
            #    'Identify Mode: Exemplars (%s vs. %s)' % (enctext, constants.EXEMPLAR_ENCTEXT)]
            #text_list = [
            #    'Identify Mode: Within-Encounter' ,
            #    'Identify Mode: Exemplars']
            #query_text =
            #ibswgt.query_button
            #ibswgt.querydb_combo.setOptionText(text_list)
            #ibswgt.query_
            #ibswgt.button_list[1][0].setText('Identify (intra-encounter)\nQUERY(%r vs. %r)' % (enctext, enctext))
            #ibswgt.button_list[1][1].setText('Identify (vs exemplar database)\nQUERY(%r vs. %r)' % (enctext, constants.EXEMPLAR_ENCTEXT))
        except Exception:
            pass

    def _update_enc_tab_name(ibswgt, eid, enctext):
        ibswgt.enc_tabwgt._update_enc_tab_name(eid, enctext)

    #@checks_qt_error
    def _connect_signals_and_slots(ibswgt):
        for tblname in ibswgt.super_tblname_list:
            tblview = ibswgt.views[tblname]
            tblview.doubleClicked.connect(ibswgt.on_doubleclick)
            tblview.clicked.connect(ibswgt.on_click)
            tblview.contextMenuClicked.connect(ibswgt.on_contextMenuClicked)
            tblview.rows_updated.connect(ibswgt.on_rows_updated)
            #front.printSignal.connect(back.backend_print)
            #front.raiseExceptionSignal.connect(back.backend_exception)

    #------------
    # SLOTS
    #------------

    @slot_(str, int)
    def on_rows_updated(ibswgt, tblname, nRows):
        """ When the rows are updated change the tab names """
        #printDBG('Rows updated in tblname=%r, nRows=%r' % (str(tblname), nRows))
        if tblname == ENCOUNTER_TABLE:  # Hack
            return
        tblname = str(tblname)
        tblnice = gh.TABLE_NICE[tblname]
        view = ibswgt.views[tblname]
        index = ibswgt._tab_table_wgt.indexOf(view)
        text = tblnice + ' ' + str(nRows)
        #printDBG('Rows updated in index=%r, text=%r' % (index, text))
        ibswgt._tab_table_wgt.setTabText(index, text)

    @slot_(QtCore.QModelIndex, QtCore.QPoint)
    def on_contextMenuClicked(ibswgt, qtindex, pos):
        #printDBG('[newgui] contextmenu')
        model = qtindex.model()
        tblview = ibswgt.views[model.name]
        id_list = sorted(list(set([model._get_row_id(_qtindex) for _qtindex in
                                   tblview.selectedIndexes()])))
        if model.name == ENCOUNTER_TABLE:
            options = [
                ('delete encounter(s)', lambda: ibswgt.back.delete_encounter(id_list)),
                ('export encounter(s)', lambda: ibswgt.back.export_encounters(id_list)),
            ]
            if len(id_list) > 1:
                merge_destination_id = model._get_row_id(qtindex)  # This is for the benefit of merge encounters
                options += [
                    ('merge encounters', lambda: ibswgt.back.merge_encounters(id_list, merge_destination_id)),
                ]
            guitool.popup_menu(tblview, pos, options)
        elif model.name == IMAGE_TABLE:
            if len(id_list) == 1:
                gid = id_list[0]
                guitool.popup_menu(tblview, pos, [
                    ('view hough image', lambda: ibswgt.back.show_hough_image(gid)),
                    ('delete image', lambda: ibswgt.back.delete_image(gid)),
                ])
            else:
                print(id_list)
                guitool.popup_menu(tblview, pos, [
                    ('move to new encounter', lambda: ibswgt.back.send_to_new_encounter(id_list, mode='move')),
                    ('copy to new encounter', lambda: ibswgt.back.send_to_new_encounter(id_list, mode='copy')),
                    ('delete images', lambda: ibswgt.back.delete_image(id_list)),
                ])
        elif model.name == ANNOTATION_TABLE:
            if len(id_list) == 1:
                cid = id_list[0]
                guitool.popup_menu(tblview, pos, [
                    ('view probability chip', lambda: ibswgt.back.show_probability_chip(cid)),
                    ('delete annotation', lambda: ibswgt.back.delete_annot(id_list)),
                ])
            else:
                guitool.popup_menu(tblview, pos, [
                    ('delete annotations', lambda: ibswgt.back.delete_annot(id_list)),
                ])

    @slot_(QtCore.QModelIndex)
    def on_click(ibswgt, qtindex):
        #printDBG('on_click')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex)
        #model_name = model.name
        #print('clicked: %s' + utool.dict_str(locals()))
        if model.name == ENCOUNTER_TABLE:
            pass
            #printDBG('clicked encounter')
        else:
            eid = model.eid
            if model.name == IMAGE_TABLE:
                gid = id_
                ibswgt.back.select_gid(gid, eid, show=False)
            elif model.name == IMAGE_GRID:
                gid = id_
                ibswgt.back.select_gid(gid, eid, show=False)
            elif model.name == ANNOTATION_TABLE:
                aid = id_
                ibswgt.back.select_aid(aid, eid, show=False)
            elif model.name in (NAME_TABLE, NAMES_TREE,):
                level = model._get_level(qtindex)
                if level == 0:
                    nid = id_
                    ibswgt.back.select_nid(nid, eid, show=False)
                elif level == 1:
                    aid = id_
                    ibswgt.back.select_aid(aid, eid, show=False)

    @slot_(QtCore.QModelIndex)
    def on_doubleclick(ibswgt, qtindex):
        #printDBG('on_doubleclick')
        model = qtindex.model()
        id_ = model._get_row_id(qtindex)
        if model.name == ENCOUNTER_TABLE:
            eid = id_
            enctext = ibswgt.ibs.get_encounter_enctext(eid)
            ibswgt.enc_tabwgt._add_enc_tab(eid, enctext)
            ibswgt.back.select_eid(eid)
        else:
            eid = model.eid
            if (model.name == IMAGE_TABLE) or (model.name == IMAGE_GRID):
                print('[newgui] Creating new annotation interaction')
                ANNOTATION_Interaction2 = interact_annotations2.ANNOTATION_Interaction2
                gid = id_
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
                ibswgt.annot_interact = ANNOTATION_Interaction2(ibs, gid, **iannot2_kw)
                #ibswgt.annot_interact.update_image_and_callbacks(gid, nextcb, prevcb)
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

    def _interactannot2_callbacks(ibswgt, model, qtindex):
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
        return next_callback, prev_callback, cur_gid

    @slot_(list)
    def imagesDropped(ibswgt, url_list):
        from os.path import isdir
        print('[drop_event] url_list=%r' % (url_list,))
        gpath_list = filter(utool.matches_image, url_list)
        dir_list   = filter(isdir, url_list)
        if len(dir_list) > 0:
            ans = guitool.user_option(ibswgt, title='Non-Images dropped',
                                      msg='Recursively import from directories?')
            if ans == 'Yes':
                gpath_list.extend(list(map(utool.unixpath,
                                           utool.flatten([utool.list_images(dir_, fullpath=True, recursive=True)
                                                          for dir_ in dir_list]))))
            else:
                return
        print('[drop_event] gpath_list=%r' % (gpath_list,))
        if len(gpath_list) > 0:
            ibswgt.back.import_images_from_file(gpath_list=gpath_list)

if __name__ == '__main__':
    import ibeis
    import sys
    ibeis._preload(mpl=False, par=False)
    guitool.ensure_qtapp()
    dbdir = ibeis.sysres.get_args_dbdir(defaultdb='cache')
    ibs = IBEISControl.IBEISController(dbdir=dbdir)
    ibswgt = IBEISGuiWidget(ibs=ibs)

    if '--cmd' in sys.argv:
        guitool.qtapp_loop(qwin=ibswgt, ipy=True)
        exec(utool.ipython_execstr())
    else:
        guitool.qtapp_loop(qwin=ibswgt)
