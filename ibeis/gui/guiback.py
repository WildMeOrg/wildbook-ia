from __future__ import absolute_import, division, print_function
# Python
import sys
from os.path import exists, join
import functools
# Qt
from PyQt4 import QtCore
# GUITool
import guitool
from guitool import slot_, signal_, cast_from_qt
# PlotTool
from plottool import fig_presenter
# IBEIS
from ibeis.dev import ibsfuncs, sysres
from ibeis.gui import newgui
from ibeis.gui import guiheaders as gh
from ibeis import viz
from ibeis.viz import interact
# Utool
import utool
from ibeis.control import IBEISControl
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[back]', DEBUG=False)


VERBOSE = utool.VERBOSE


def default_decorator(func):
    return func
    #return utool.indent_func('[back.' + func.func_name + ']')(func)


def backblock(func):
    """ BLOCKING DECORATOR
    TODO: This decorator has to be specific to either front or back. Is there a
    way to make it more general? """
    func = default_decorator(func)
    @functools.wraps(func)
    #@guitool.checks_qt_error
    def bacblock_wrapper(back, *args, **kwargs):
        _wasBlocked_ = back.front.blockSignals(True)
        try:
            result = func(back, *args, **kwargs)
        except Exception:
            raise
        finally:
            back.front.blockSignals(_wasBlocked_)  # unblock regardless
        return result
    return bacblock_wrapper


def blocking_slot(*types_):
    """
    A blocking slot accepts the types which are passed to QtCore.pyqtSlot.
    In addition it also causes the gui frontend to block signals while
    the decorated function is processing.
    """
    def wrap_bslot(func):
        @slot_(*types_)
        @backblock
        @functools.wraps(func)
        def wrapped_bslot(*args, **kwargs):
            printDBG('[back*] ' + utool.func_str(func))
            printDBG('[back*] ' + utool.func_str(func, args, kwargs))
            result = func(*args, **kwargs)
            sys.stdout.flush()
            return result
        printDBG('blocking slot: %r, types=%r' % (wrapped_bslot.func_name, types_))
        return wrapped_bslot
    return wrap_bslot


#------------------------
# Backend MainWindow Class
#------------------------
class MainWindowBackend(QtCore.QObject):
    """
    Sends and recieves signals to and from the frontend
    """
    # Backend Signals
    updateWindowTitleSignal = signal_(str)

    #------------------------
    # Constructor
    #------------------------
    def __init__(back, ibs=None):
        """ Creates GUIBackend object """
        QtCore.QObject.__init__(back)
        print('[back] MainWindowBackend.__init__()')
        back.ibs  = None
        back.cfg = None
        # State variables
        back.sel_rids = []
        back.sel_nids = []
        back.sel_gids = []
        back.sel_qres = []
        back.active_enc = 0
        back.encounter_query_results = utool.ddict(dict)

        # Create GUIFrontend object
        back.mainwin = newgui.IBEISMainWindow(back=back, ibs=ibs)
        back.front = back.mainwin.ibswgt
        back.ibswgt = back.front  # Alias
        # connect signals and other objects
        fig_presenter.register_qt4_win(back.mainwin)

    #------------------------
    # Draw Functions
    #------------------------

    def show(back):
        back.mainwin.show()

    def select_bbox(back, gid, **kwargs):
        bbox = interact.iselect_bbox(back.ibs, gid)
        return bbox

    def show_image(back, gid, sel_rids=[], **kwargs):
        kwargs.update({
            'sel_rids': sel_rids,
            'select_callback': back.select_gid,
        })
        interact.ishow_image(back.ibs, gid, **kwargs)

    def show_roi(back, rid, show_image=False, **kwargs):
        interact.ishow_chip(back.ibs, rid, **kwargs)
        if show_image:
            gid = back.ibs.get_roi_gids(rid)
            interact.ishow_image(back.ibs, gid, sel_rids=[rid])

    def show_name(back, nid, sel_rids=[], **kwargs):
        #nid = back.ibs.get_name_nids(name)
        kwargs.update({
            'sel_rids': sel_rids,
            'select_rid_callback': back.select_rid,
        })
        interact.ishow_name(back.ibs, nid, **kwargs)
        pass

    def show_qres(back, qres, **kwargs):
        kwargs['annote_mode'] = kwargs.get('annote_mode', 2)
        kwargs['top_rids'] = kwargs.get('top_rids', 6)
        interact.ishow_qres(back.ibs, qres, **kwargs)
        # HACK
        from ibeis.gui import inspect_gui
        qrid2_qres = {qres.qrid: qres}
        backed_callback = back.front.update_tables
        back.qres_wgt1 = inspect_gui.QueryResultsWidget(back.ibs, qrid2_qres, callback=backend_callback, ranks_lt=kwargs['top_rids'], )
        back.qres_wgt1.show()
        back.qres_wgt1.raise_()
        pass

    def show_hough(back, gid, **kwargs):
        viz.show_hough(back.ibs, gid, **kwargs)
        viz.draw()

    #----------------------
    # State Management Functions (ewww... state)
    #----------------------

    #@utool.indent_func
    def update_window_title(back):
        pass

    #@utool.indent_func
    def refresh_state(back):
        """ Blanket refresh function. Try not to call this """
        back.front.update_tables()

    #@utool.indent_func
    def connect_ibeis_control(back, ibs):
        print('[back] connect_ibeis()')
        back.ibs = ibs
        back.front.connect_ibeis_control(ibs)
        back._set_selection(sel_gids=[], sel_rids=[], sel_nids=[],
                            sel_eids=[None])

    @blocking_slot()
    def default_config(back):
        """ Button Click -> Preferences Defaults """
        print('[back] default preferences')
        back.ibs._default_config()
        back.edit_prefs_wgt.refresh_layout()
        back.edit_prefs_wgt.pref_model.rootPref.save()

    @utool.indent_func
    def get_selected_gid(back):
        """ selected image id """
        if len(back.sel_gids) == 0:
            if len(back.sel_rids) == 0:
                gid = back.ibs.get_roi_gids(back.sel_rids)[0]
                return gid
            raise AssertionError('There are no selected images')
        gid = back.sel_gids[0]
        return gid

    @utool.indent_func
    def get_selected_rid(back):
        """ selected roi id """
        if len(back.sel_rids) == 0:
            raise AssertionError('There are no selected ROIs')
        rid = back.sel_rids[0]
        return rid

    @utool.indent_func
    def get_selected_eid(back):
        """ selected encounter id """
        if len(back.sel_eids) == 0:
            raise AssertionError('There are no selected Encounters')
        eid = back.sel_eids[0]
        return eid

    @utool.indent_func
    def get_selected_qres(back):
        """ selected query result """
        if len(back.sel_qres) > 0:
            qres = back.sel_qres[0]
            return qres
        else:
            return None

    #--------------------------------------------------------------------------
    # Selection Functions
    #--------------------------------------------------------------------------

    def _set_selection(back, sel_gids=None, sel_rids=None, sel_nids=None,
                       sel_qres=None, sel_eids=None, **kwargs):
        if sel_eids is not None:
            back.sel_eids = sel_eids
            back.ibswgt.set_status_label(0, 'Selected Encounter: %r' % (sel_eids,))
        if sel_gids is not None:
            back.sel_gids = sel_gids
            back.ibswgt.set_status_label(1, 'Selected Image: %r' % (sel_gids,))
        if sel_rids is not None:
            back.sel_rids = sel_rids
            back.ibswgt.set_status_label(2, 'Selected ROI: %r' % (sel_rids,))
        if sel_nids is not None:
            back.sel_nids = sel_nids
            back.ibswgt.set_status_label(3, 'Selected Name: %r' % (sel_nids,))
        if sel_qres is not None:
            back.sel_sel_qres = sel_qres

    @backblock
    def select_eid(back, eid=None, **kwargs):
        """ Table Click -> Result Table """
        eid = cast_from_qt(eid)
        back._set_selection(sel_eids=(eid,), **kwargs)
        print('[back] select encounter eid=%r' % (eid))

    @backblock
    def select_gid(back, gid, eid=None, show=True, sel_rids=None, **kwargs):
        """ Table Click -> Image Table """
        # Select the first ROI in the image if unspecified
        if sel_rids is None:
            sel_rids = back.ibs.get_image_rids(gid)
            if len(sel_rids) > 0:
                sel_rids = sel_rids[0:1]
            else:
                sel_rids = []
        print('[back] select_gid(gid=%r, eid=%r, sel_rids=%r)' % (gid, eid, sel_rids))
        back._set_selection(sel_gids=(gid,), sel_rids=sel_rids, sel_eids=[eid], **kwargs)
        if show:
            back.show_image(gid, sel_rids=sel_rids)

    @backblock
    def select_rid(back, rid, eid=None, show=True, show_roi=True, **kwargs):
        """ Table Click -> Chip Table """
        print('[back] select rid=%r, eid=%r' % (rid, eid))
        gid = back.ibs.get_roi_gids(rid)
        nid = back.ibs.get_roi_nids(rid)
        back._set_selection(sel_rids=(rid,), sel_gids=[gid], sel_nids=[nid], sel_eids=[eid], **kwargs)
        if show and show_roi:
            back.show_roi(rid, **kwargs)

    @backblock
    def select_nid(back, nid, eid=None, show=True, show_name=True, **kwargs):
        """ Table Click -> Name Table """
        nid = cast_from_qt(nid)
        print('[back] select nid=%r, eid=%r' % (nid, eid))
        back._set_selection(sel_nids=(nid,), sel_eids=[eid], **kwargs)
        if show and show_name:
            back.show_name(nid, **kwargs)

    @backblock
    def select_qres_rid(back, rid, eid=None, show=True, **kwargs):
        """ Table Click -> Result Table """
        eid = cast_from_qt(eid)
        rid = cast_from_qt(rid)
        print('[back] select result rid=%r, eid=%r' % (rid, eid))

    #--------------------------------------------------------------------------
    # Action menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def add_roi(back, gid=None, bbox=None, theta=0.0, refresh=True):
        """ Action -> Add ROI"""
        print('[back] add_roi')
        if gid is None:
            gid = back.get_selected_gid()
        if bbox is None:
            bbox = back.select_bbox(gid)
        printDBG('[back.add_roi] * adding bbox=%r' % (bbox,))
        rid = back.ibs.add_rois([gid], [bbox], [theta])[0]
        printDBG('[back.add_roi] * added rid=%r' % (rid,))
        if refresh:
            back.front.update_tables([gh.IMAGE_TABLE, gh.ROI_TABLE])
            #back.show_image(gid)
            pass
        back.select_gid(gid, sel_rids=[rid])
        return rid

    @blocking_slot()
    def reselect_roi(back, rid=None, bbox=None, refresh=True, **kwargs):
        """ Action -> Reselect ROI"""
        if rid is None:
            rid = back.get_selected_rid()
        gid = back.ibs.get_roi_gids(rid)
        if bbox is None:
            bbox = back.select_bbox(gid)
        print('[back] reselect_roi')
        back.ibs.set_roi_bboxes([rid], [bbox])
        if refresh:
            back.front.update_tables([gh.ROI_TABLE])
            back.show_image(gid)

    @blocking_slot()
    def query(back, rid=None, refresh=True, **kwargs):
        """ Action -> Query"""
        print('\n\n[back] query')
        if rid is None:
            rid = back.get_selected_rid()
        eid = back._eidfromkw(kwargs)
        if eid is None:
            print('[back] query_database(rid=%r)' % (rid,))
            qrid2_qres = back.ibs.query_database([rid])
        else:
            print('[back] query_encounter(rid=%r, eid=%r)' % (rid, eid))
            qrid2_qres = back.ibs.query_encounter([rid], eid)
        qres = qrid2_qres[rid]
        back._set_selection(sel_qres=[qres])
        if refresh:
            #back.populate_tables(qres=True, default=False)
            back.show_qres(qres)

    @blocking_slot()
    def reselect_ori(back, rid=None, theta=None, **kwargs):
        """ Action -> Reselect ORI"""
        print('[back] reselect_ori')
        raise NotImplementedError()
        pass

    @blocking_slot()
    def delete_roi(back, rid=None):
        """ Action -> Delete Chip"""
        print('[back] delete_roi')
        if rid is None:
            rid = back.get_selected_rid()
        # get the image-id of the roi we are deleting
        gid = back.ibs.get_roi_gids(rid)
        # delete the roi
        back.ibs.delete_rois([rid])
        # update display, to show image without the deleted roi
        back.select_gid(gid)
        back.front.update_tables()

    @blocking_slot(int)
    def delete_image(back, gid=None):
        """ Action -> Delete Images"""
        print('[back] delete_image')
        gid = cast_from_qt(gid)
        back.ibs.delete_images([gid])
        back.front.update_tables()

    @blocking_slot()
    def delete_all_encounters(back):
        print('\n\n[back] delete all encounters')
        back.ibs.delete_all_encounters()
        back.front.update_tables()

    @blocking_slot(int)
    def delete_encounter(back, eid):
        print('\n\n[back] delete encounter')
        back.ibs.delete_encounters([eid])
        back.front.update_tables()

    @blocking_slot()
    def select_next(back):
        """ Action -> Next"""
        print('[back] select_next')
        raise NotImplementedError()
        pass

    @blocking_slot()
    def select_prev(back):
        """ Action -> Prev"""
        print('[back] select_prev')
        raise NotImplementedError()
        pass

    #--------------------------------------------------------------------------
    # Batch menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def _run_detection(back, quick=True, refresh=True, **kwargs):
        print('\n\n')
        eid = back._eidfromkw(kwargs)
        ibs = back.ibs
        gid_list = ibsfuncs.get_empty_gids(ibs, eid=eid)
        species = ibs.cfg.detect_cfg.species
        print('[back] _run_detection(quick=%r, species=%r, eid=%r)' % (quick, species, eid))
        ibs.detect_random_forest(gid_list, species, quick=quick)
        print('[back] about to finish detection')
        if refresh:
            back.front.update_tables([gh.IMAGE_TABLE, gh.ROI_TABLE])
        print('[back] finished detection')

    @blocking_slot()
    def run_detection_coarse(back, refresh=True):
        back._run_detection(quick=True)

    @blocking_slot()
    def run_detection_fine(back, refresh=True):
        back._run_detection(quick=False)

    @blocking_slot()
    def compute_feats(back, refresh=True, **kwargs):
        """ Batch -> Precompute Feats"""
        print('[back] compute_feats')
        eid = back._eidfromkw(kwargs)
        ibsfuncs.compute_all_features(back.ibs, eid=eid)
        if refresh:
            back.front.update_tables()

    @blocking_slot()
    def compute_queries(back, refresh=True, **kwargs):
        """ Batch -> Precompute Queries"""
        eid = back._eidfromkw(kwargs)
        print('[back] compute_queries: eid=%r' % (eid,))
        back.compute_feats(refresh=False, **kwargs)
        valid_rids = back.ibs.get_valid_rids(eid=eid)
        if eid is None:
            qrid2_qres = back.ibs.query_database(valid_rids)
        else:
            qrid2_qres = back.ibs.query_encounter(valid_rids, eid)
        back.encounter_query_results[eid].update(qrid2_qres)
        print('[back] About to finish compute_queries: eid=%r' % (eid,))
        back.review_queries(eid=eid)
        if refresh:
            back.front.update_tables()
        print('[back] FINISHED compute_queries: eid=%r' % (eid,))

    @blocking_slot()
    def review_queries(back, **kwargs):
        eid = back.get_selected_eid()
        if eid not in back.encounter_query_results:
            raise AssertionError('Queries have not been computed yet')
        qrid2_qres = back.encounter_query_results[eid]
        # review_kw = {
        #     'on_change_callback': back.front.update_tables,
        #     'nPerPage': 6,
        # }
        ibs = back.ibs
        # Matplotlib QueryResults interaction
        #from ibeis.viz.interact import interact_qres2
        #back.query_review = interact_qres2.Interact_QueryResult(ibs, qrid2_qres, **review_kw)
        #back.query_review.show()
        # Qt QueryResults Interaction
        from ibeis.gui import inspect_gui
        backend_callback = back.front.update_tables
        back.qres_wgt = inspect_gui.QueryResultsWidget(ibs, qrid2_qres, callback=backend_callback)
        back.qres_wgt.show()
        back.qres_wgt.raise_()

    @blocking_slot()
    def review_detections(back, **kwargs):
        from plottool.interact_multi_image import MultiImageInteraction
        eid = back.get_selected_eid()
        ibs = back.ibs
        gid_list = ibs.get_valid_gids(eid=eid)
        gpath_list = ibs.get_image_paths(gid_list)
        bboxes_list = ibs.get_image_roi_bboxes(gid_list)
        thetas_list = ibs.get_image_roi_thetas(gid_list)
        multi_image_interaction = MultiImageInteraction(gpath_list, bboxes_list=bboxes_list, thetas_list=thetas_list)
        back.multi_image_interaction = multi_image_interaction

    @blocking_slot()
    def compute_encounters(back, refresh=True):
        """ Batch -> Compute Encounters """
        print('[back] compute_encounters')
        back.ibs.compute_encounters()
        print('[back] about to finish computing encounters')
        if refresh:
            back.front.update_tables()
        print('[back] finished computing encounters')

    #--------------------------------------------------------------------------
    # Option menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def layout_figures(back):
        """ Options -> Layout Figures"""
        print('[back] layout_figures')
        fig_presenter.all_figures_tile()
        pass

    @slot_()
    def edit_preferences(back):
        """ Options -> Edit Preferences"""
        print('[back] edit_preferences')
        epw = back.ibs.cfg.createQWidget()
        fig_presenter.register_qt4_win(epw)
        epw.ui.defaultPrefsBUT.clicked.connect(back.default_config)
        epw.show()
        back.edit_prefs_wgt = epw
        #query_cfgstr = ''.join(back.ibs.cfg.query_cfg.get_cfgstr())
        #print('[back] query_cfgstr = %s' % query_cfgstr)
        #print('')

    #--------------------------------------------------------------------------
    # Help menu slots
    #--------------------------------------------------------------------------

    @slot_()
    def view_docs(back):
        """ Help -> View Documentation"""
        print('[back] view_docs')
        raise NotImplementedError()
        pass

    @slot_()
    def view_database_dir(back):
        """ Help -> View Directory Slots"""
        print('[back] view_database_dir')
        utool.view_directory(back.ibs.get_dbdir())
        pass

    @slot_()
    def view_app_files_dir(back):
        print('[back] view_model_dir')
        utool.view_directory(utool.get_app_resource_dir('ibeis'))
        pass

    @slot_()
    def delete_detection_models(back):
        print('[back] delete_detection_models')
        utool.delete(utool.get_app_resource_dir('ibeis', 'detectmodels'))
        pass

    @slot_()
    def delete_cache(back):
        """ Help -> Delete Directory Slots"""
        print('[back] delete_cache')
        utool.delete(back.ibs.get_cachedir())
        pass

    @slot_()
    def delete_global_prefs(back):
        print('[back] delete_global_prefs')
        # TODO: Add are you sure dialog?
        utool.delete(utool.get_app_resource_dir('ibeis', 'global_cache'))
        pass

    @slot_()
    def delete_queryresults_dir(back):
        print('[back] delete_queryresults_dir')
        # TODO: Add are you sure dialog?
        utool.delete(back.ibs.qresdir)
        pass

    @blocking_slot()
    def dev_reload(back):
        """ Help -> Developer Reload"""
        print('[back] dev_reload')
        from ibeis.dev.all_imports import reload_all
        reload_all()

    @blocking_slot()
    def dev_mode(back):
        """ Help -> Developer Mode"""
        print('[back] dev_mode')
        from ibeis.dev import all_imports
        all_imports.embed(back)

    @blocking_slot()
    def dev_cls(back):
        """ Help -> Developer Mode"""
        print('[back] dev_cls')
        print('\n'.join([''] * 100))
        back.refresh_state()
        from plottool import draw_func2 as df2
        df2.update()

    @blocking_slot()
    def dev_dumpdb(back):
        """ Help -> Developer Mode"""
        print('[back] dev_dumpdb')
        back.ibs.db.dump()
        utool.view_directory(back.ibs._ibsdb)
        back.ibs.db.dump_tables_to_csv()

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
        with utool.Indenter(lbl='    [opendb]'):
            try:
                ibs = IBEISControl.IBEISController(dbdir=dbdir)
                back.connect_ibeis_control(ibs)
            except Exception as ex:
                utool.printex(ex, 'caught Exception while opening database')
                raise
            else:
                sysres.set_default_dbdir(dbdir)

    @blocking_slot()
    def export_database(back):
        """ File -> Export Database"""
        print('[back] export_database')
        back.ibs.db.dump()
        back.ibs.db.dump_tables_to_csv()

    @blocking_slot()
    def import_images(back, gpath_list=None, dir_=None, refresh=True):
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
            gid_list = back.import_images_from_file(gpath_list=gpath_list,
                                                    refresh=refresh)
        if reply == 'Directory' or dir_ is not None:
            gid_list = back.import_images_from_dir(dir_=dir_, refresh=refresh)
        return gid_list

    @blocking_slot()
    def import_images_from_file(back, gpath_list=None, refresh=True):
        print('[back] import_images_from_file')
        """ File -> Import Images From File"""
        if back.ibs is None:
            raise ValueError('back.ibs is None! must open IBEIS database first')
        if gpath_list is None:
            gpath_list = guitool.select_images('Select image files to import')
        gid_list = back.ibs.add_images(gpath_list)
        if refresh:
            back.front.update_tables([gh.IMAGE_TABLE])
            #back.populate_image_table()
        return gid_list

    @blocking_slot()
    def import_images_from_dir(back, dir_=None, size_filter=None, refresh=True):
        """ File -> Import Images From Directory"""
        print('[back] import_images_from_dir')
        if dir_ is None:
            dir_ = guitool.select_directory('Select directory with images in it')
        printDBG('[back] dir=%r' % dir_)
        if dir_ is None:
            return
        gpath_list = utool.list_images(dir_, fullpath=True)
        if size_filter is not None:
            raise NotImplementedError('Can someone implement the size filter?')
        gid_list = back.ibs.add_images(gpath_list)
        if refresh:
            back.front.update_tables([gh.IMAGE_TABLE])
        return gid_list
        #print('')

    @slot_()
    def localize_images(back):
        """ File -> Localize Images """
        print('[back] localize_images')
        back.ibs.localize_images()

    @slot_()
    def quit(back):
        """ File -> Quit"""
        print('[back] ')
        guitool.exit_application()

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
        return sysres.get_workdir()

    def user_select_new_dbdir(back):
        raise NotImplementedError()
        pass

    def _eidfromkw(back, kwargs):
        if 'eid' not in kwargs:
            eid = back.get_selected_eid()
        else:
            eid = kwargs['eid']
        return eid
