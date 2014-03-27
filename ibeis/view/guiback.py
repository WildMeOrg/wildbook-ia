from __future__ import division, print_function
# Python
from os.path import split
# Qt
from PyQt4 import QtCore
from PyQt4.Qt import pyqtSignal
# IBEIS
from ibeis.dev import params
from ibeis.view import guifront
from ibeis.view import guitool
from ibeis.view import guitool_dialogs
from ibeis.view.guitool import drawing, slot_


# BLOCKING DECORATOR
# TODO: This decorator has to be specific to either front or back. Is there a
# way to make it more general?
def backblock(func):
    def bacblock_wrapper(back, *args, **kwargs):
        wasBlocked_ = back.front.blockSignals(True)
        try:
            result = func(back, *args, **kwargs)
        except Exception as ex:
            raise
            import traceback
            back.front.blockSignals(wasBlocked_)
            print('!!!!!!!!!!!!!')
            print('[guitool] caught exception in %r' % func.func_name)
            print(traceback.format_exc())
            back.user_info('Error:\nex=%r' % ex)
            raise
        back.front.blockSignals(wasBlocked_)
        return result
    bacblock_wrapper.func_name = func.func_name
    return bacblock_wrapper


def blocking_slot(*types_):
    def wrap1(func):
        def wrap2(*args, **kwargs):
            return func(*args, **kwargs)
        wrap2.func_name = func.func_name
        wrap3 = slot_(*types_)(backblock(wrap2))
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
    populateSignal = pyqtSignal(str, list, list, list, list)
    setEnabledSignal = pyqtSignal(bool)

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

        # A map from short internal headers to fancy headers seen by the user
        back.fancy_headers = {
            'gid':        'Image Index',
            'nid':        'Name Index',
            'cid':        'Chip ID',
            'aif':        'All Detected',
            'gname':      'Image Name',
            'nCxs':       '#Chips',
            'name':       'Name',
            'nGt':        '#GT',
            'nKpts':      '#Kpts',
            'theta':      'Theta',
            'roi':        'ROI (x, y, w, h)',
            'rank':       'Rank',
            'score':      'Confidence',
            'match_name': 'Matching Name',
        }
        back.reverse_fancy = {v: k for (k, v) in back.fancy_headers.items()}

        # A list of default internal headers to display
        back.table_headers = {
            'gids':  ['gid', 'gname', 'nCxs', 'aif'],
            'cids':  ['cid', 'name', 'gname', 'nGt', 'nKpts', 'theta'],
            'nids':  ['nid', 'name', 'nCxs'],
            'res':   ['rank', 'score', 'name', 'cid']
        }

        # Lists internal headers whos items are editable
        back.table_editable = {
            'gids':  [],
            'cids':  ['name'],
            'nids':  ['name'],
            'res':   ['name'],
        }

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
    # Work Functions
    #----------------------

    def get_selected_gid(back):
        'selected image id'
        pass

    def get_selected_cid(back):
        'selected chip id'
        pass

    def update_window_title(back):
        if back.ibs is None:
            title = 'IBEIS - NULL database'
        if back.ibs.dbdir is None:
            title = 'IBEIS - invalid database'
        else:
            dbdir = back.ibs.dbdir
            db_name = split(dbdir)[1]
            title = 'IBEIS - %r - %s' % (db_name, dbdir)
        back.front.setWindowTitle(title)

    def connect_ibeis_control(back, ibs):
        print('[back] connect_ibeis()')
        back.ibs = ibs

    #--------------------------------------------------------------------------
    # Populate functions
    #--------------------------------------------------------------------------

    def _populate_table(back, tblname, extra_cols={},
                        index_list=None, prefix_cols=[]):
        print('[back] _populate_table(%r)' % tblname)

        def make_header_lists(tbl_headers, editable_list, prop_keys=[]):
            col_headers = tbl_headers[:] + prop_keys
            col_editable = [False] * len(tbl_headers) + [True] * len(prop_keys)
            for header in editable_list:
                col_editable[col_headers.index(header)] = True
            return col_headers, col_editable

        headers = back.table_headers[tblname]
        editable = back.table_editable[tblname]
        if tblname == 'cxs':  # in ['cxs', 'res']: TODO props in restable
            prop_keys = back.ibs.tables.prop_dict.keys()
        else:
            prop_keys = []
            col_headers, col_editable = make_header_lists(headers, editable, prop_keys)
        if index_list is None:
            index_list = back.ibs.get_valid_ids(tblname)
        # Prefix datatup
        prefix_datatup = [[prefix_col.get(header, 'error')
                           for header in col_headers]
                          for prefix_col in prefix_cols]
        body_datatup = back.ibs.get_datatup_list(tblname, index_list,
                                                 col_headers, extra_cols)
        datatup_list = prefix_datatup + body_datatup
        row_list = range(len(datatup_list))
        # Populate with fancy headers.
        col_fancyheaders = [back.fancy_headers[key]
                            if key in back.fancy_headers else key
                            for key in col_headers]
        back.populateSignal.emit(tblname, col_fancyheaders, col_editable,
                                 row_list, datatup_list)

    def populate_image_table(back, **kwargs):
        back._populate_table('gids', **kwargs)

    def populate_name_table(back, **kwargs):
        back._populate_table('nids', **kwargs)

    def populate_chip_table(back, **kwargs):
        back._populate_table('cids', **kwargs)

    def populate_result_table(back, **kwargs):
        res = back.current_res
        if res is None:
            # Clear the table if there are no results
            print('[back] no results available')
            back._populate_table('res', index_list=[])
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
        back._populate_table('res', index_list=top_cxs,
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
        return guitool_dialogs.user_info(parent=back.front, *args, **kwargs)

    def user_input(back, *args, **kwargs):
        return guitool_dialogs.user_input(parent=back.front, *args, **kwargs)

    def user_option(back, *args, **kwargs):
        return guitool_dialogs.user_option(parent=back.front, *args, **kwargs)

    def get_work_directory(back):
        return params.get_workdir()

    def user_select_new_dbdir(back):
        pass

    #--------------------------------------------------------------------------
    # Selection Functions
    #--------------------------------------------------------------------------

    @blocking_slot(int)
    def select_gid(back, gid, **kwargs):
        # Table Click -> Image Table
        pass

    @blocking_slot(int)
    def select_cid(back, cid, **kwargs):
        # Table Click -> Chip Table
        pass

    @slot_(str)
    def select_name(back, name):
        # Table Click -> Name Table
        pass

    @slot_(int)
    def select_res_cid(back, cid, **kwargs):
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

    @blocking_slot(int, str, str)
    def change_chip_property(back, cid, key, val):
        # Table Edit -> Change Chip Property
        pass

    @blocking_slot(int, str, str)
    def alias_name(back, nid, key, val):
        # Table Edit -> Change name
        pass

    @blocking_slot(int, str, bool)
    def change_image_property(back, gid, key, val):
        # Table Edit -> Change Image Property
        pass

    #--------------------------------------------------------------------------
    # File Slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def new_database(back, new_dbdir=None):
        # File -> New Database
        pass

    @blocking_slot()
    def open_database(back, dbdir=None):
        # File -> Open Database
        pass

    @blocking_slot()
    def save_database(back):
        # File -> Save Database
        pass

    @blocking_slot()
    def import_images(back, gpath_list=None, dir_=None):
        # File -> Import Images (ctrl + i)
        print('[back] import images')
        if not (gpath_list is None and dir_ is None):
            reply = back.user_option(
                msg='Import specific files or whole directory?',
                title='Import Images',
                options=['Files', 'Directory'],
                use_cache=False)
        else:
            reply = None
        if reply == 'Files' or gpath_list is not None:
            back.import_images_from_file()
        if reply == 'Directory' or dir_ is not None:
            back.import_images_from_dir()

    @blocking_slot()
    def import_images_from_file(back, gpath_list=None):
        # File -> Import Images From File
        if back.ibs is None:
            raise ValueError('back.ibs is None! must open IBEIS database first')
        if gpath_list is None:
            gpath_list = guitool_dialogs.select_images('Select image files to import')
        back.ibs.add_images(gpath_list)
        back.populate_image_table()
        print('')

    @blocking_slot()
    def import_images_from_dir(back):
        # File -> Import Images From Directory
        pass
        #msg = 'Select directory with images in it'
        #img_dpath = guitool.select_directory(msg)
        #print('[back] selected %r' % img_dpath)
        #fpath_list = util.list_images(img_dpath, fullpath=True)
        #back.ibs.add_images(fpath_list)
        #back.populate_image_table()
        #print('')

    @slot_()
    def quit(back):
        # File -> Quit
        guitool.exit_application()

    #--------------------------------------------------------------------------
    # Action menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def new_prop(back):
        # Action -> New Chip Property
        # Depricate?
        pass

    @blocking_slot()
    def add_chip(back, gid=None, roi=None, theta=0.0):
        # Action -> Add ROI
        pass

    @blocking_slot()
    def query(back, cid=None, **kwargs):
        # Action -> Query
        pass

    @blocking_slot()
    def reselect_roi(back, cid=None, roi=None, **kwargs):
        # Action -> Reselect ROI
        pass

    @blocking_slot()
    def reselect_ori(back, cid=None, theta=None, **kwargs):
        # Action -> Reselect ORI
        pass

    @blocking_slot()
    def delete_chip(back):
        # Action -> Delete Chip
        pass

    @blocking_slot()
    def delete_image(back, gid=None):
        # Action -> Delete Images
        pass

    @blocking_slot()
    def select_next(back):
        # Action -> Next
        pass

    #--------------------------------------------------------------------------
    # Batch menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def precompute_feats(back):
        # Batch -> Precompute Feats
        pass

    @blocking_slot()
    def precompute_queries(back):
        pass

    #--------------------------------------------------------------------------
    # Option menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def layout_figures(back):
        # Options -> Layout Figures
        pass

    @slot_()
    def edit_preferences(back):
        pass
        # Options -> Edit Preferences
        back.edit_prefs = back.cfg.createQWidget()
        epw = back.edit_prefs
        epw.ui.defaultPrefsBUT.clicked.connect(back.default_preferences)
        query_uid = ''.join(back.cfg.query_cfg.get_uid())
        print('[back] query_uid = %s' % query_uid)
        print('')

    #--------------------------------------------------------------------------
    # Help menu slots
    #--------------------------------------------------------------------------

    @slot_()
    def view_docs(back):
        # Help -> View Documentation
        pass

    @slot_()
    def view_database_dir(back):
        # Help -> View Directory Slots
        pass

    @slot_()
    def view_computed_dir(back):
        pass

    @slot_()
    def view_global_dir(back):
        pass

    @slot_()
    def delete_cache(back):
        # Help -> Delete Directory Slots
        pass

    @slot_()
    def delete_global_prefs(back):
        # RCOS TODO: Are you sure?
        pass

    @slot_()
    def delete_queryresults_dir(back):
        # RCOS TODO: Are you sure?
        pass

    @blocking_slot()
    def dev_reload(back):
        # Help -> Developer Reload
        pass

    @blocking_slot()
    def dev_mode(back):
        # Help -> Developer Reload
        pass
