from __future__ import absolute_import, division, print_function
# Python
import sys
from os.path import split, exists, join
import functools
# Qt
from PyQt4 import QtCore
# GUITool
import guitool
from guitool import drawing, slot_, signal_
# IBEIS
from ibeis.dev import params
from ibeis.view import guifront
from ibeis.view import gui_item_tables as item_table
from ibeis.view import interact
# Utool
import utool
from ibeis.control import IBEISControl
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[back]', DEBUG=False)


VERBOSE = utool.get_flag('--verbose')

# Wrapped QT UUID type (usually string or long)
QT_IMAGE_UID_TYPE = item_table.QT_IMAGE_UID_TYPE
QT_ROI_UID_TYPE   = item_table.QT_ROI_UID_TYPE
QT_NAME_UID_TYPE  = item_table.QT_NAME_UID_TYPE

qt_cast = item_table.qt_cast

# For UUIDs the cast is special
qt_roi_uid_cast   = item_table.qt_roi_uid_cast
qt_image_uid_cast = item_table.qt_image_uid_cast
qt_name_uid_cast  = item_table.qt_name_uid_cast


# BLOCKING DECORATOR
# TODO: This decorator has to be specific to either front or back. Is there a
# way to make it more general?
def backblock(func):
    @functools.wraps(func)
    @utool.ignores_exc_tb
    def bacblock_wrapper(back, *args, **kwargs):
        wasBlocked_ = back.front.blockSignals(True)
        try:
            result = func(back, *args, **kwargs)
        except Exception as ex:
            back.front.blockSignals(wasBlocked_)  # unblock signals on exception
            #print(traceback.format_exc())
            msg = ('caught exception in %r' % func.func_name)
            print('\n\n\n')
            utool.printex(ex, msg)
            print('\n\n\n')
            back.user_info(msg=msg, title=str(type(ex)))
            raise
            #raise
        back.front.blockSignals(wasBlocked_)
        return result
    return bacblock_wrapper


def blocking_slot(*types_):
    def wrap1(func):
        @utool.ignores_exc_tb
        def wrap2(*args, **kwargs):
            printDBG('[back*] ' + utool.func_str(func))
            printDBG('[back*] ' + utool.func_str(func, args, kwargs))
            result = func(*args, **kwargs)
            sys.stdout.flush()
            return result
        wrap2 = functools.update_wrapper(wrap2, func)
        wrap3 = slot_(*types_)(backblock(wrap2))
        wrap3 = functools.update_wrapper(wrap3, func)
        printDBG('blocking slot: %r, types=%r' % (wrap3.func_name, types_))
        return wrap3
    return wrap1


#------------------------
# Backend MainWindow Class
#------------------------
class MainWindowBackend(QtCore.QObject):
    '''
    Sends and recieves signals to and from the frontend
    '''
    # Backend Signals
    populateSignal = signal_(str, list, list, list, list)
    setEnabledSignal = signal_(bool)

    #------------------------
    # Constructor
    #------------------------
    def __init__(back, ibs=None):
        print('[back] MainWindowBackend.__init__()')
        super(MainWindowBackend, back).__init__()
        back.ibs  = None
        back.cfg = None
        # State variables
        back.sel_rids = []
        back.sel_nids = []
        back.sel_gids = []
        back.sel_qres = []

        # connect signals and other objects
        back.front = guifront.MainWindowFrontend(back=back)
        back.populateSignal.connect(back.front.populate_tbl)
        back.setEnabledSignal.connect(back.front.setEnabled)

    #------------------------
    # Draw Functions
    #------------------------

    def show(back):
        back.front.show()

    @drawing
    def show_splash(back, fnum, **kwargs):
        pass

    @drawing
    def select_bbox(back, gid, **kwargs):
        bbox = interact.select_bbox(back.ibs, gid)
        return bbox

    @drawing
    def show_image(back, gid, sel_rids=[], **kwargs):
        kwargs.update({
            'sel_rids': sel_rids,
            'select_callback': back.select_gid,
        })
        interact.interact_image(back.ibs, gid, **kwargs)

    @drawing
    def show_roi(back, rid, show_image=False, **kwargs):
        interact.interact_chip(back.ibs, rid, **kwargs)
        if show_image:
            gid = back.ibs.get_roi_gids(rid)
            interact.interact_image(back.ibs, gid, sel_rids=[rid])

    @drawing
    def show_name(back, name, sel_rids=[], **kwargs):
        pass

    @drawing
    def show_query_result(back, res, **kwargs):
        pass

    @drawing
    def show_roimatch(back, res, rid, **kwargs):
        pass

    #----------------------
    # State Management Functions (ewww... state)
    #----------------------

    @utool.indent_func
    def get_selected_gid(back):
        'selected image id'
        if len(back.sel_gids) == 0:
            raise AssertionError('There are no selected images')
        gid = back.sel_gids[0]
        return gid

    @utool.indent_func
    def get_selected_rid(back):
        'selected roi id'
        if len(back.sel_rids) == 0:
            raise AssertionError('There are no selected ROIs')
        rid = back.sel_rids[0]
        return rid

    #@utool.indent_func
    def update_window_title(back):
        if VERBOSE:
            print('[back] update_window_title()')
        if back.ibs is None:
            title = 'IBEIS - No Database Open'
        if back.ibs.dbdir is None:
            title = 'IBEIS - invalid database'
        else:
            dbdir = back.ibs.dbdir
            db_name = split(dbdir)[1]
            title = 'IBEIS - %r - %s' % (db_name, dbdir)
        back.front.setWindowTitle(title)

    #@utool.indent_func
    def refresh_state(back):
        back.update_window_title()
        back.populate_tables()

    @utool.indent_func
    def connect_ibeis_control(back, ibs):
        print('[back] connect_ibeis()')
        back.ibs = ibs
        back.refresh_state()

    #--------------------------------------------------------------------------
    # Populate functions
    #----------------------1----------------------------------------------------

    @utool.indent_func
    def populate_image_table(back, **kwargs):
        item_table.emit_populate_table(back, item_table.IMAGE_TABLE, **kwargs)

    @utool.indent_func
    def populate_name_table(back, **kwargs):
        item_table.emit_populate_table(back, item_table.NAME_TABLE, **kwargs)

    @utool.indent_func
    def populate_roi_table(back, **kwargs):
        item_table.emit_populate_table(back, item_table.ROI_TABLE, **kwargs)

    @utool.indent_func
    def populate_result_table(back, **kwargs):
        #res = back.current_res
        res = None
        return
        if res is None:
            # Clear the table if there are no results
            print('[back] no results available')
            return
        #item_table.emit_populate_table(back, item_table.RES_TABLE, index_list=[])
        top_cxs = res.topN_cxs(back.ibs, N='all')
        qrid = res.qrid
        # The ! mark is used for ascii sorting. TODO: can we work around this?
        prefix_cols = [{'rank': '!Query',
                        'score': '---',
                        'name': back.ibs.get_roi_name(qrid),
                        'rid': qrid, }]
        extra_cols = {
            'score':  lambda cxs:  [res.cx2_score[rid] for rid in iter(cxs)],
        }
        back.emit_populate_table(item_table.RES_TABLE, index_list=top_cxs,
                                 prefix_cols=prefix_cols,
                                 extra_cols=extra_cols,
                                 **kwargs)

    def populate_tables(back, image=True, roi=True, name=True, res=True):
        if image:
            back.populate_image_table()
        if roi:
            back.populate_roi_table()
        if name:
            back.populate_name_table()
        if res:
            back.populate_result_table()

    #--------------------------------------------------------------------------
    # Helper functions
    #--------------------------------------------------------------------------

    def user_info(back, **kwargs):
        return guitool.user_info(parent=back.front, **kwargs)

    def user_input(back, **kwargs):
        return guitool.user_input(parent=back.front, **kwargs)

    def user_option(back, **kwargs):
        return guitool.user_option(parent=back.front, **kwargs)

    def get_work_directory(back):
        return params.get_workdir()

    def user_select_new_dbdir(back):
        pass

    #--------------------------------------------------------------------------
    # Selection Functions
    #--------------------------------------------------------------------------

    def _set_selection(back, gids=None, rids=None, nids=None, qres=None, **kwargs):
        if gids is not None:
            back.sel_gids = gids
        if rids is not None:
            back.sel_rids = rids
        if nids is not None:
            back.sel_nids = nids
        if qres is not None:
            back.sel_qres = qres

    @blocking_slot(QT_IMAGE_UID_TYPE)
    def select_gid(back, gid, sel_rids=[], **kwargs):
        # Table Click -> Image Table
        gid = qt_image_uid_cast(gid)
        print('[back] select_gid(gid=%r, sel_rids=%r)' % (gid, sel_rids))
        back._set_selection(gids=(gid,), rids=sel_rids, **kwargs)
        back.show_image(gid, sel_rids=sel_rids)

    @blocking_slot(QT_ROI_UID_TYPE)
    def select_rid(back, rid, show_roi=True, **kwargs):
        # Table Click -> Chip Table
        rid = qt_roi_uid_cast(rid)
        print('[back] select rid=%r' % rid)
        back._set_selection(rids=[rid], **kwargs)
        if show_roi:
            back.show_roi(rid, **kwargs)

    @slot_(QT_NAME_UID_TYPE)
    def select_nid(back, nid, **kwargs):
        # Table Click -> Name Table
        nid = qt_name_uid_cast(nid)
        print('[back] select nid=%r' % nid)
        back._set_selection(nids=[nid], **kwargs)

    @slot_(QT_ROI_UID_TYPE)
    def select_res_rid(back, rid, **kwargs):
        # Table Click -> Result Table
        print('[back] select result rid=%r' % rid)
        rid = qt_roi_uid_cast(rid)

    #--------------------------------------------------------------------------
    # Misc Slots
    #--------------------------------------------------------------------------

    @slot_(str)
    def backend_print(back, msg):
        'slot so guifront can print'
        print(msg)

    @slot_()
    def clear_selection(back, **kwargs):
        print('[back] clear selection')

    @blocking_slot()
    def default_preferences(back):
        # Button Click -> Preferences Defaults
        print('[back] default preferences')

    @blocking_slot(QT_ROI_UID_TYPE, str, str)
    def change_roi_property(back, rid, key, val):
        # Table Edit -> Change Chip Property
        rid = qt_roi_uid_cast(rid)
        val = qt_cast(val)
        print('[back] change_roi_property(rid=%r, key=%r, val=%r)' % (rid, key, val))
        back.ibs.set_roi_properties((rid,), key, (val,))

    @blocking_slot(QT_NAME_UID_TYPE, str, str)
    def alias_name(back, nid, key, val):
        # Table Edit -> Change name
        nid = qt_name_uid_cast(nid)
        print('[back] alias_name(nid=%r, key=%r, val=%r)' % (nid, key, val))

    @blocking_slot(QT_IMAGE_UID_TYPE, str, bool)
    def change_image_property(back, gid, key, val):
        # Table Edit -> Change Image Property
        gid = qt_image_uid_cast(gid)
        print('[back] alias_name(gid=%r, key=%r, val=%r)' % (gid, key, val))

    #--------------------------------------------------------------------------
    # File Slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def new_database(back, new_dbdir=None):
        """ File -> New Database"""
        if new_dbdir is None:
            new_dbname = back.user_input(
                msg='What do you want to name the new database?',
                title='New Database')
            if new_dbname is None or len(new_dbname) == 0:
                print('Abort new database. new_dbname=%r' % new_dbname)
                return
            reply = back.user_option(
                msg='Where should I put the new database?',
                title='Import Images',
                options=['Choose Directory', 'My Work Dir'],
                use_cache=False)
            if reply == 'Choose Directory':
                print('[back] new_database(): SELECT A DIRECTORY')
                putdir = guitool.select_directory('Select new database directory')
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
        utool.ensuredir(new_dbdir)
        print('[back] new_database(new_dbdir=%r)' % new_dbdir)
        back.open_database(dbdir=new_dbdir)

    @blocking_slot()
    def open_database(back, dbdir=None):
        """ File -> Open Database"""
        if dbdir is None:
            print('[back] new_database(): SELECT A DIRECTORY')
            dbdir = guitool.select_directory('Select new database directory')
            if dbdir is None:
                return
        print('[back] open_database(dbdir=%r)' % dbdir)
        try:
            ibs = IBEISControl.IBEISControl(dbdir=dbdir)
            back.connect_ibeis_control(ibs)
        except Exception as ex:
            print('[guiback] Caught: %s: %s' % (type(ex), ex))
            raise
        else:
            utool.global_cache_write('cached_dbdir', dbdir)

    @blocking_slot()
    def save_database(back):
        """ File -> Save Database"""
        print('[back] ')
        # Depricate
        pass

    @blocking_slot()
    def import_images(back, gpath_list=None, dir_=None):
        """ File -> Import Images (ctrl + i)"""
        print('[back] import_images')
        reply = None
        if gpath_list is None and dir_ is None:
            reply = back.user_option(
                msg='Import specific files or whole directory?',
                title='Import Images',
                options=['Files', 'Directory'],
                use_cache=False)
        if reply == 'Files' or gpath_list is not None:
            gid_list = back.import_images_from_file(gpath_list=gpath_list)
        if reply == 'Directory' or dir_ is not None:
            gid_list = back.import_images_from_dir(dir_=dir_)
        return gid_list

    @blocking_slot()
    def import_images_from_file(back, gpath_list=None):
        print('[back] import_images_from_file')
        """ File -> Import Images From File"""
        if back.ibs is None:
            raise ValueError('back.ibs is None! must open IBEIS database first')
        if gpath_list is None:
            gpath_list = guitool.select_images('Select image files to import')
        gid_list = back.ibs.add_images(gpath_list)
        back.populate_image_table()
        return gid_list

    @blocking_slot()
    def import_images_from_dir(back, dir_=None):
        print('[back] import_images_from_dir')
        """ File -> Import Images From Directory"""
        if dir_ is None:
            dir_ = guitool.select_directory('Select directory with images in it')
        printDBG('[back] dir=%r' % dir_)
        gpath_list = utool.list_images(dir_, fullpath=True)
        gid_list = back.ibs.add_images(gpath_list)
        back.populate_image_table()
        return gid_list
        #print('')

    @slot_()
    def quit(back):
        """ File -> Quit"""
        print('[back] ')
        guitool.exit_application()

    #--------------------------------------------------------------------------
    # Action menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def new_prop(back):
        """ Action -> New Chip Property"""
        # Depricate
        print('[back] new_prop')
        pass

    @blocking_slot()
    def add_roi(back, gid=None, bbox=None, theta=0.0):
        """ Action -> Add ROI"""
        print('[back] add_roi')
        if gid is None:
            gid = back.get_selected_gid()
        if bbox is None:
            bbox = back.select_bbox(gid)
        printDBG('[back.add_roi] * adding bbox=%r' % bbox)
        rid = back.ibs.add_rois([gid], [bbox], [theta])[0]
        printDBG('[back.add_roi] * added rid=%r' % rid)
        back.populate_tables()
        back.show_image(gid)
        #back.select_gid(gid, rids=[rid])
        return rid

    @blocking_slot()
    def reselect_roi(back, rid=None, roi=None, **kwargs):
        """ Action -> Reselect ROI"""
        print('[back] reselect_roi')
        pass

    @blocking_slot()
    def query(back, rid=None, **kwargs):
        """ Action -> Query"""
        print('[back] query')
        pass

    @blocking_slot()
    def reselect_ori(back, rid=None, theta=None, **kwargs):
        """ Action -> Reselect ORI"""
        print('[back] reselect_ori')
        pass

    @blocking_slot()
    def delete_roi(back):
        """ Action -> Delete Chip"""
        print('[back] delete_roi')
        pass

    @blocking_slot(QT_IMAGE_UID_TYPE)
    def delete_image(back, gid=None):
        """ Action -> Delete Images"""
        print('[back] delete_image')
        gid = qt_image_uid_cast(gid)
        pass

    @blocking_slot()
    def select_next(back):
        """ Action -> Next"""
        print('[back] select_next')
        pass

    #--------------------------------------------------------------------------
    # Batch menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def precompute_feats(back):
        """ Batch -> Precompute Feats"""
        print('[back] precompute_feats')
        pass

    @blocking_slot()
    def precompute_queries(back):
        """ Batch -> Precompute Queries"""
        print('[back] precompute_queries')
        pass

    #--------------------------------------------------------------------------
    # Option menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def layout_figures(back):
        """ Options -> Layout Figures"""
        print('[back] layout_figures')
        pass

    @slot_()
    def edit_preferences(back):
        print('[back] edit_preferences')
        pass
        """ Options -> Edit Preferences"""
        #back.edit_prefs = back.cfg.createQWidget()
        #epw = back.edit_prefs
        #epw.ui.defaultPrefsBUT.clicked.connect(back.default_preferences)
        #query_uid = ''.join(back.cfg.query_cfg.get_uid())
        #print('[back] query_uid = %s' % query_uid)
        #print('')

    #--------------------------------------------------------------------------
    # Help menu slots
    #--------------------------------------------------------------------------

    @slot_()
    def view_docs(back):
        """ Help -> View Documentation"""
        print('[back] view_docs')
        pass

    @slot_()
    def view_database_dir(back):
        """ Help -> View Directory Slots"""
        print('[back] view_database_dir')
        utool.view_directory(back.ibs.dbdir)
        pass

    @slot_()
    def view_computed_dir(back):
        print('[back] view_computed_dir')
        pass

    @slot_()
    def view_global_dir(back):
        print('[back] view_global_dir')
        pass

    @slot_()
    def delete_cache(back):
        """ Help -> Delete Directory Slots"""
        print('[back] delete_cache')
        pass

    @slot_()
    def delete_global_prefs(back):
        # RCOS TODO: Add are you sure dialog?
        print('[back] delete_global_prefs')
        pass

    @slot_()
    def delete_queryresults_dir(back):
        # RCOS TODO: Add are you sure dialog?
        print('[back] delete_queryresults_dir')
        pass

    @blocking_slot()
    def dev_reload(back):
        """ Help -> Developer Reload"""
        print('[back] dev_reload')
        #from ibeis.dev import debug_imports
        #ibeis_modules = debug_imports.get_ibeis_modules()
        #for module in ibeis_modules:
            #if not hasattr(module, 'rrr'):
                #utool.inject_reload_function(module=module)
            #if hasattr(module, 'rrr'):
                #module.rrr()

    @blocking_slot()
    def dev_mode(back):
        """ Help -> Developer Mode"""
        print('[back] dev_mode')
        pass
