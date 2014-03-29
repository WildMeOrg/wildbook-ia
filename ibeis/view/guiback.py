from __future__ import division, print_function
# Python
import sys
from os.path import split, exists
import functools
import traceback
import uuid
# Qt
from PyQt4 import QtCore
# GUITool
import guitool
from guitool import drawing, slot_, signal_
# IBEIS
from ibeis.dev import params
from ibeis.view import guifront
from ibeis.view import gui_item_tables
from ibeis.view import interact
# Utool
import utool
from ibeis.control import IBEISControl
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[back]', DEBUG=False)


UUID_type = gui_item_tables.UUID_type


def uuid_cast(qtinput):
    return uuid.UUID(UUID_type(qtinput))


# BLOCKING DECORATOR
# TODO: This decorator has to be specific to either front or back. Is there a
# way to make it more general?
def backblock(func):
    @functools.wraps(func)
    def bacblock_wrapper(back, *args, **kwargs):
        wasBlocked_ = back.front.blockSignals(True)
        try:
            result = func(back, *args, **kwargs)
        except Exception as ex:
            raise
            back.front.blockSignals(wasBlocked_)
            print('!!!!!!!!!!!!!')
            print('[guiback] caught exception in %r' % func.func_name)
            print(traceback.format_exc())
            back.user_info('Error:\nex=%r' % ex)
            raise
        back.front.blockSignals(wasBlocked_)
        return result
    return bacblock_wrapper


def blocking_slot(*types_):
    def wrap1(func):
        def wrap2(*args, **kwargs):
            print('[back*] ' + func.func_name)
            result = func(*args, **kwargs)
            sys.stdout.flush()
            return result
        wrap2 = functools.update_wrapper(wrap2, func)
        wrap3 = slot_(*types_)(backblock(wrap2))
        wrap3 = functools.update_wrapper(wrap3, func)
        print('blocking slot: %r' % wrap3.func_name)
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
        back.sel_cids = []
        back.sel_nids = []
        back.sel_gids = []
        back.qcid2_res = {}

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
    def show_image(back, gid, sel_cids=[], **kwargs):
        interact.interact_image(back.ibs, gid, sel_cids)
        pass

    @drawing
    def show_chip(back, cid, **kwargs):
        pass

    @drawing
    def show_name(back, name, sel_cids=[], **kwargs):
        pass

    @drawing
    def show_query_result(back, res, **kwargs):
        pass

    @drawing
    def show_chipmatch(back, res, cid, **kwargs):
        pass

    #----------------------
    # State Management Functions (ewww... state)
    #----------------------

    def get_selected_gid(back):
        'selected image id'
        pass

    def get_selected_cid(back):
        'selected chip id'
        pass

    def update_window_title(back):
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

    def refresh_state(back):
        back.update_window_title()
        back.populate_tables()

    def connect_ibeis_control(back, ibs):
        print('[back] connect_ibeis()')
        back.ibs = ibs
        back.refresh_state()

    #--------------------------------------------------------------------------
    # Populate functions
    #----------------------1----------------------------------------------------

    def populate_image_table(back, **kwargs):
        gui_item_tables.emit_populate_table(back, 'gids', **kwargs)

    def populate_name_table(back, **kwargs):
        gui_item_tables.emit_populate_table(back, 'nids', **kwargs)

    def populate_chip_table(back, **kwargs):
        gui_item_tables.emit_populate_table(back, 'cids', **kwargs)

    def populate_result_table(back, **kwargs):
        #res = back.current_res
        res = None
        if res is None:
            # Clear the table if there are no results
            print('[back] no results available')
            gui_item_tables.emit_populate_table(back, 'res', index_list=[])
            return
        top_cxs = res.topN_cxs(back.ibs, N='all')
        qcid = res.qcid
        # The ! mark is used for ascii sorting. TODO: can we work around this?
        prefix_cols = [{'rank': '!Query',
                        'score': '---',
                        'name': back.ibs.get_chip_name(qcid),
                        'cid': qcid, }]
        extra_cols = {
            'score':  lambda cxs:  [res.cx2_score[cid] for cid in iter(cxs)],
        }
        back.emit_populate_table('res', index_list=top_cxs,
                                 prefix_cols=prefix_cols,
                                 extra_cols=extra_cols,
                                 **kwargs)

    def populate_tables(back, image=True, chip=True, name=True, res=True):
        if image:
            back.populate_image_table()
        if chip:
            back.populate_chip_table()
        if name:
            back.populate_name_table()
        if res:
            back.populate_result_table()

    #--------------------------------------------------------------------------
    # Helper functions
    #--------------------------------------------------------------------------

    def user_info(back, *args, **kwargs):
        return guitool.user_info(parent=back.front, *args, **kwargs)

    def user_input(back, *args, **kwargs):
        return guitool.user_input(parent=back.front, *args, **kwargs)

    def user_option(back, *args, **kwargs):
        return guitool.user_option(parent=back.front, *args, **kwargs)

    def get_work_directory(back):
        return params.get_workdir()

    def user_select_new_dbdir(back):
        pass

    #--------------------------------------------------------------------------
    # Selection Functions
    #--------------------------------------------------------------------------

    @blocking_slot(UUID_type)
    def select_gid(back, gid, **kwargs):
        # Table Click -> Image Table
        gid = uuid_cast(gid)
        print('[back] select gid=%r' % gid)
        back.show_image(gid)
        pass

    @blocking_slot(UUID_type)
    def select_cid(back, cid, **kwargs):
        # Table Click -> Chip Table
        cid = uuid_cast(cid)
        print('[back] select cid=%r' % cid)
        pass

    @slot_(str)
    def select_name(back, name):
        # Table Click -> Name Table
        name = str(name)
        print('[back] select name=%r' % name)
        pass

    @slot_(UUID_type)
    def select_res_cid(back, cid, **kwargs):
        print('[back] select result cid=%r' % cid)
        cid = uuid_cast(cid)
        # Table Click -> Result Table
        pass

    #--------------------------------------------------------------------------
    # Misc Slots
    #--------------------------------------------------------------------------

    @slot_(str)
    def backend_print(back, msg):
        'slot so guifront can print'
        print(msg)

    @slot_()
    def clear_selection(back, **kwargs):
        pass

    @blocking_slot()
    def default_preferences(back):
        # Button Click -> Preferences Defaults
        pass

    @blocking_slot(UUID_type, str, str)
    def change_chip_property(back, cid, key, val):
        cid = uuid_cast(cid)
        # Table Edit -> Change Chip Property
        pass

    @blocking_slot(UUID_type, str, str)
    def alias_name(back, nid, key, val):
        # Table Edit -> Change name
        nid = uuid_cast(nid)
        pass

    @blocking_slot(UUID_type, str, bool)
    def change_image_property(back, gid, key, val):
        # Table Edit -> Change Image Property
        gid = uuid_cast(gid)
        pass

    #--------------------------------------------------------------------------
    # File Slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def new_database(back, new_dbdir=None):
        # File -> New Database
        if new_dbdir is None:
            print('[back] new_database(): SELECT A DIRECTORY')
            new_dbdir = guitool.select_directory('Select new database directory')
            if new_dbdir is None:
                return
        print('[back] new_database(new_dbdir=%r)' % new_dbdir)
        if not exists(new_dbdir):
            utool.ensuredir(new_dbdir, verbose=True)
        back.open_database(dbdir=new_dbdir)

    @blocking_slot()
    def open_database(back, dbdir=None):
        # File -> Open Database
        if dbdir is None:
            print('[back] new_database(): SELECT A DIRECTORY')
            dbdir = guitool.select_directory('Select new database directory')
            if dbdir is None:
                return
        print('[back] open_database(dbdir=%r)' % dbdir)
        ibs = IBEISControl.IBEISControl(dbdir=dbdir)
        back.connect_ibeis_control(ibs)

    @blocking_slot()
    def save_database(back):
        # File -> Save Database
        print('[back] ')
        pass

    @blocking_slot()
    def import_images(back, gpath_list=None, dir_=None):
        # File -> Import Images (ctrl + i)
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
        # File -> Import Images From File
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
        # File -> Import Images From Directory
        pass
        if dir_ is None:
            dir_ = guitool.select_directory('Select directory with images in it')
        print('[back] dir=%r' % dir_)
        gpath_list = utool.list_images(dir_, fullpath=True)
        gid_list = back.ibs.add_images(gpath_list)
        back.populate_image_table()
        return gid_list
        #print('')

    @slot_()
    def quit(back):
        # File -> Quit
        print('[back] ')
        guitool.exit_application()

    #--------------------------------------------------------------------------
    # Action menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def new_prop(back):
        # Action -> New Chip Property
        # Depricate?
        print('[back] ')
        pass

    @blocking_slot()
    def add_roi(back, gid=None, roi=None, theta=0.0):
        # Action -> Add ROI
        print('[back] ')
        pass

    @blocking_slot()
    def query(back, cid=None, **kwargs):
        # Action -> Query
        print('[back] ')
        pass

    @blocking_slot()
    def reselect_roi(back, cid=None, roi=None, **kwargs):
        # Action -> Reselect ROI
        print('[back] ')
        pass

    @blocking_slot()
    def reselect_ori(back, cid=None, theta=None, **kwargs):
        # Action -> Reselect ORI
        print('[back] ')
        pass

    @blocking_slot()
    def delete_chip(back):
        # Action -> Delete Chip
        print('[back] ')
        pass

    @blocking_slot(UUID_type)
    def delete_image(back, gid=None):
        # Action -> Delete Images
        print('[back] ')
        gid = uuid_cast(gid)
        pass

    @blocking_slot()
    def select_next(back):
        # Action -> Next
        print('[back] ')
        pass

    #--------------------------------------------------------------------------
    # Batch menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def precompute_feats(back):
        # Batch -> Precompute Feats
        print('[back] ')
        pass

    @blocking_slot()
    def precompute_queries(back):
        print('[back] ')
        pass

    #--------------------------------------------------------------------------
    # Option menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def layout_figures(back):
        # Options -> Layout Figures
        print('[back] ')
        pass

    @slot_()
    def edit_preferences(back):
        print('[back] ')
        pass
        # Options -> Edit Preferences
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
        # Help -> View Documentation
        print('[back] ')
        pass

    @slot_()
    def view_database_dir(back):
        # Help -> View Directory Slots
        print('[back] ')
        pass

    @slot_()
    def view_computed_dir(back):
        print('[back] ')
        pass

    @slot_()
    def view_global_dir(back):
        print('[back] view_global_dir')
        pass

    @slot_()
    def delete_cache(back):
        # Help -> Delete Directory Slots
        print('[back] delete_cache')
        pass

    @slot_()
    def delete_global_prefs(back):
        # RCOS TODO: Are you sure?
        print('[back] delete_global_prefs')
        pass

    @slot_()
    def delete_queryresults_dir(back):
        # RCOS TODO: Are you sure?
        print('[back] delete_queryresults_dir')
        pass

    @blocking_slot()
    def dev_reload(back):
        # Help -> Developer Reload
        print('[back] dev_reload')
        pass

    @blocking_slot()
    def dev_mode(back):
        # Help -> Developer Mode
        print('[back] dev_mode')
        pass
