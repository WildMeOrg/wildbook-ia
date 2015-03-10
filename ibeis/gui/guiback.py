"""
This module controls the GUI backend.  It is the layer between the GUI frontend
(newgui.py) and the IBEIS controller.  All the functionality of the nonvisual
gui components is written or called from here
"""
from __future__ import absolute_import, division, print_function
# Python
import six
import sys
from six.moves import zip
from os.path import exists, join
import functools
# GUITool
import guitool
from guitool.__PYQT__ import QtCore
from guitool import slot_, signal_, cast_from_qt
# PlotTool
from plottool import fig_presenter
# IBEIS
from ibeis import ibsfuncs, sysres
from ibeis.gui import newgui
from ibeis.gui import guiheaders as gh
from ibeis import viz
from ibeis.viz import interact
from ibeis import constants as const
from ibeis.control import IBEISControl
from ibeis.gui import clock_offset_gui
from ibeis.gui import guiexcept
# Utool
#import utool
import utool as ut
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[back]', DEBUG=False)

VERBOSE = ut.VERBOSE


def backreport(func):
    """
    reports errors on backend functions
    should be around every function by default
    """
    def backreport_wrapper(back, *args, **kwargs):
        try:
            result = func(back, *args, **kwargs)
        except guiexcept.UserCancel as ex:
            print('handling user cancel')
            return None
        except Exception as ex:
            error_msg = "Error caught while performing function. \n %r" % ex
            guitool.msgbox(title="Error Catch!", msg=error_msg)
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
    #@guitool.checks_qt_error
    @backreport
    def bacblock_wrapper(back, *args, **kwargs):
        _wasBlocked_ = back.front.blockSignals(True)
        try:
            result = func(back, *args, **kwargs)
        except Exception:
            #error_msg = "Error caught while performing function. \n %r" % ex
            #guitool.msgbox(title="Error Catch!", msg=error_msg)
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
        @slot_(*types_)
        @backblock
        @functools.wraps(func)
        def wrapped_bslot(*args, **kwargs):
            #printDBG('[back*] ' + ut.func_str(func))
            #printDBG('[back*] ' + ut.func_str(func, args, kwargs))
            result = func(*args, **kwargs)
            sys.stdout.flush()
            return result
        #printDBG('blocking slot: %r, types=%r' % (wrapped_bslot.__name__, types_))
        wrapped_bslot = ut.preserve_sig(wrapped_bslot, func)
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
    #incQuerySignal = signal_(int)

    #------------------------
    # Constructor
    #------------------------
    def __init__(back, ibs=None):
        """ Creates GUIBackend object """
        QtCore.QObject.__init__(back)
        if ut.VERBOSE:
            print('[back] MainWindowBackend.__init__(ibs=%r)' % (ibs,))
        back.ibs = None
        back.cfg = None
        back.edit_prefs_wgt = None
        # State variables
        back.sel_aids = []
        back.sel_nids = []
        back.sel_gids = []
        back.sel_qres = []
        back.active_enc = 0
        if ut.is_developer():
            back.query_mode = const.INTRA_ENC_KEY
        else:
            back.query_mode = const.VS_EXEMPLARS_KEY
        back.encounter_query_results = ut.ddict(dict)

        # Create GUIFrontend object
        back.mainwin = newgui.IBEISMainWindow(back=back, ibs=ibs)
        back.front = back.mainwin.ibswgt
        back.ibswgt = back.front  # Alias
        # connect signals and other objects
        fig_presenter.register_qt4_win(back.mainwin)
        # register self with the ibeis controller
        back.register_self()
        back.set_query_mode(back.query_mode)
        #back.incQuerySignal.connect(back.incremental_query_slot)

    #def __del__(back):
    #    back.cleanup()

    def set_query_mode(back, new_mode):
        if new_mode == 'toggle':
            if back.query_mode == const.VS_EXEMPLARS_KEY:
                back.query_mode = const.INTRA_ENC_KEY
            else:
                back.query_mode = const.VS_EXEMPLARS_KEY
        else:
            back.query_mode = new_mode
        try:
            back.mainwin.actionToggleQueryMode.setText('Toggle Query Mode currently: %s' % back.query_mode)
        except Exception as ex:
            ut.printex(ex)
        #back.front.menuActions.

    def cleanup(back):
        if back.ibs is not None:
            back.ibs.remove_observer(back)

    #@ut.indent_func
    def notify(back):
        """ Observer's notify function. """
        back.refresh_state()

    #@ut.indent_func
    def notify_controller_killed(back):
        """ Observer's notify function that the ibeis controller has been killed. """
        back.ibs = None

    def register_self(back):
        if back.ibs is not None:
            back.ibs.register_observer(back)

    #------------------------
    # Draw Functions
    #------------------------

    def show(back):
        back.mainwin.show()

    def select_bbox(back, gid, **kwargs):
        bbox = interact.iselect_bbox(back.ibs, gid)
        return bbox

    def show_image(back, gid, sel_aids=[], **kwargs):
        kwargs.update({
            'sel_aids': sel_aids,
            'select_callback': back.select_gid,
        })
        interact.ishow_image(back.ibs, gid, **kwargs)

    def show_annotation(back, aid, show_image=False, **kwargs):
        interact.ishow_chip(back.ibs, aid, **kwargs)
        if show_image:
            gid = back.ibs.get_annot_gids(aid)
            interact.ishow_image(back.ibs, gid, sel_aids=[aid])

    def show_name(back, nid, sel_aids=[], **kwargs):
        kwargs.update({
            'sel_aids': sel_aids,
            'select_aid_callback': back.select_aid,
        })
        #nid = back.ibs.get_name_rowids_from_text(name)
        interact.ishow_name(back.ibs, nid, **kwargs)
        pass

    def show_hough_image(back, gid, **kwargs):
        viz.show_hough_image(back.ibs, gid, **kwargs)
        viz.draw()

    def run_detection_on_encounter(back, eid_list, refresh=True, **kwargs):
        gid_list = ut.flatten(back.ibs.get_encounter_gids(eid_list))
        back.run_detection_on_images(gid_list, refresh=refresh, **kwargs)

    def run_detection_on_images(back, gid_list, refresh=True, **kwargs):
        species = back.ibs.cfg.detect_cfg.species_text
        back.ibs.detect_random_forest(gid_list, species)
        if refresh:
            back.front.update_tables([gh.IMAGE_TABLE])

    def show_probability_chip(back, cid, **kwargs):
        viz.show_probability_chip(back.ibs, cid, **kwargs)
        viz.draw()

    @blocking_slot()
    def review_queries(back, qaid2_qres=None, **kwargs):
        eid = back.get_selected_eid()
        if eid not in back.encounter_query_results:
            raise guiexcept.InvalidRequest('Queries have not been computed yet')
        if qaid2_qres is None:
            qaid2_qres = back.encounter_query_results[eid]
        # review_kw = {
        #     'on_change_callback': back.front.update_tables,
        #     'nPerPage': 6,
        # }
        ibs = back.ibs
        # Matplotlib QueryResults interaction
        #from ibeis.viz.interact import interact_qres2
        #back.query_review = interact_qres2.Interact_QueryResult(ibs, qaid2_qres, **review_kw)
        #back.query_review.show()
        # Qt QueryResults Interaction
        from ibeis.gui import inspect_gui
        backend_callback = back.front.update_tables
        ranks_lt = ibs.cfg.other_cfg.ranks_lt
        back.qres_wgt = inspect_gui.QueryResultsWidget(ibs, qaid2_qres, callback=backend_callback, ranks_lt=ranks_lt)
        back.qres_wgt.show()
        back.qres_wgt.raise_()

    #def show_qres(back, qres, **kwargs):
    #    top_aids = kwargs.get('top_aids', 6)
    #    # SHOW MATPLOTLIB RESULTS (NO DECISION INTERACTIONS)
    #    if ut.get_argflag(('--show-mplres',)):
    #        kwargs['annot_mode'] = kwargs.get('annot_mode', 2)
    #        kwargs['top_aids'] = top_aids
    #        kwargs['sidebyside'] = True
    #        kwargs['show_query'] = False
    #        #kwargs['sidebyside'] = False
    #        #kwargs['show_query'] = True
    #        kwargs['in_image'] = False
    #        qres.ishow_top(back.ibs, **kwargs)

    #    #interact.ishow_matches(back.ibs, qres, **kwargs)
    #    # HACK SHOW QT RESULTS
    #    if not ut.get_argflag(('--noshow-qtres',)):
    #        from ibeis.gui import inspect_gui
    #        qaid2_qres = {qres.qaid: qres}
    #        backend_callback = back.front.update_tables
    #        back.qres_wgt1 = inspect_gui.QueryResultsWidget(back.ibs, qaid2_qres,
    #                                                        callback=backend_callback,
    #                                                        ranks_lt=top_aids,)
    #        back.qres_wgt1.show()
    #        back.qres_wgt1.raise_()
    #    pass

    #----------------------
    # State Management Functions (ewww... state)
    #----------------------

    #@ut.indent_func
    def update_window_title(back):
        pass

    #@ut.indent_func
    def refresh_state(back):
        """ Blanket refresh function. Try not to call this """
        back.front.update_tables()

    #@ut.indent_func
    def connect_ibeis_control(back, ibs):
        if ut.VERBOSE:
            print('[back] connect_ibeis(ibs=%r)' % (ibs,))
        if ibs is None:
            return None
        back.ibs = ibs
        # register self with the ibeis controller
        back.register_self()
        back._set_selection(sel_gids=[], sel_aids=[], sel_nids=[],
                            sel_eids=[None])
        back.front.connect_ibeis_control(ibs)

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
                gid = back.ibs.get_annot_gids(back.sel_aids)[0]
                return gid
            raise guiexcept.InvalidRequest('There are no selected images')
        gid = back.sel_gids[0]
        return gid

    @ut.indent_func
    def get_selected_aid(back):
        """ selected annotation id """
        if len(back.sel_aids) == 0:
            raise guiexcept.InvalidRequest('There are no selected ANNOTATIONs')
        aid = back.sel_aids[0]
        return aid

    @ut.indent_func
    def get_selected_eid(back):
        """ selected encounter id """
        if len(back.sel_eids) == 0:
            raise guiexcept.InvalidRequest('There are no selected Encounters')
        eid = back.sel_eids[0]
        return eid

    @ut.indent_func
    def get_selected_qres(back):
        """
        UNUSED DEPRICATE

        selected query result """
        if len(back.sel_qres) > 0:
            qres = back.sel_qres[0]
            return qres
        else:
            return None

    #--------------------------------------------------------------------------
    # Selection Functions
    #--------------------------------------------------------------------------

    def _set_selection(back, sel_gids=None, sel_aids=None, sel_nids=None,
                       sel_qres=None, sel_eids=None, **kwargs):
        if sel_eids is not None:
            back.sel_eids = sel_eids
            sel_enctexts = back.ibs.get_encounter_enctext(sel_eids)
            if sel_enctexts == [None]:
                sel_enctexts = []
            else:
                sel_enctexts = map(str, sel_enctexts)
            back.ibswgt.set_status_text(gh.ENCOUNTER_TABLE, repr(sel_enctexts,))
        if sel_gids is not None:
            back.sel_gids = sel_gids
            back.ibswgt.set_status_text(gh.IMAGE_TABLE, repr(sel_gids,))
        if sel_aids is not None:
            back.sel_aids = sel_aids
            back.ibswgt.set_status_text(gh.ANNOTATION_TABLE, repr(sel_aids,))
        if sel_nids is not None:
            back.sel_nids = sel_nids
            back.ibswgt.set_status_text(gh.NAMES_TREE, repr(sel_nids,))
        if sel_qres is not None:
            raise NotImplementedError('no select qres implemented')
            back.sel_sel_qres = sel_qres

    #@backblock
    def select_eid(back, eid=None, **kwargs):
        """ Table Click -> Result Table """
        eid = cast_from_qt(eid)
        if False:
            prefix = ut.get_caller_name(range(1, 8))
        else:
            prefix = ''
        print(prefix + '[back] select encounter eid=%r' % (eid))
        back._set_selection(sel_eids=(eid,), **kwargs)

    #@backblock
    def select_gid(back, gid, eid=None, show=True, sel_aids=None, **kwargs):
        """ Table Click -> Image Table """
        # Select the first ANNOTATION in the image if unspecified
        if sel_aids is None:
            sel_aids = back.ibs.get_image_aids(gid)
            if len(sel_aids) > 0:
                sel_aids = sel_aids[0:1]
            else:
                sel_aids = []
        print('[back] select_gid(gid=%r, eid=%r, sel_aids=%r)' % (gid, eid, sel_aids))
        back._set_selection(sel_gids=(gid,), sel_aids=sel_aids, sel_eids=[eid], **kwargs)
        if show:
            back.show_image(gid, sel_aids=sel_aids)

    #@backblock
    def select_gid_from_aid(back, aid, eid=None, show=True):
        gid = back.ibs.get_annot_gids(aid)
        back.select_gid(gid, eid=eid, show=show, sel_aids=[aid])

    #@backblock
    def select_aid(back, aid, eid=None, show=True, show_annotation=True, **kwargs):
        """ Table Click -> Chip Table """
        print('[back] select aid=%r, eid=%r' % (aid, eid))
        gid = back.ibs.get_annot_gids(aid)
        nid = back.ibs.get_annot_name_rowids(aid)
        back._set_selection(sel_aids=(aid,), sel_gids=[gid], sel_nids=[nid], sel_eids=[eid], **kwargs)
        if show and show_annotation:
            back.show_annotation(aid, **kwargs)

    @backblock
    def select_nid(back, nid, eid=None, show=True, show_name=True, **kwargs):
        """ Table Click -> Name Table """
        nid = cast_from_qt(nid)
        print('[back] select nid=%r, eid=%r' % (nid, eid))
        back._set_selection(sel_nids=(nid,), sel_eids=[eid], **kwargs)
        if show and show_name:
            back.show_name(nid, **kwargs)

    @backblock
    def select_qres_aid(back, aid, eid=None, show=True, **kwargs):
        """ Table Click -> Result Table """
        eid = cast_from_qt(eid)
        aid = cast_from_qt(aid)
        print('[back] select result aid=%r, eid=%r' % (aid, eid))

    #--------------------------------------------------------------------------
    # Action menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def add_annot(back, gid=None, bbox=None, theta=0.0, refresh=True):
        """ Action -> Add ANNOTATION"""
        print('[back] add_annot')
        if gid is None:
            gid = back.get_selected_gid()
        if bbox is None:
            bbox = back.select_bbox(gid)
        #printDBG('[back.add_annot] * adding bbox=%r' % (bbox,))
        aid = back.ibs.add_annots([gid], [bbox], [theta])[0]
        #printDBG('[back.add_annot] * added aid=%r' % (aid,))
        if refresh:
            back.front.update_tables([gh.IMAGE_TABLE, gh.ANNOTATION_TABLE])
            #back.show_image(gid)
            pass
        back.select_gid(gid, sel_aids=[aid])
        return aid

    @blocking_slot()
    def add_annotation_from_image(back, gid_list, refresh=True):
        """ Action -> Add Annotation from Image"""
        print('[back] add_annotation_from_image')
        size_list = back.ibs.get_image_sizes(gid_list)
        bbox_list = [ (0, 0, w, h) for (w, h) in size_list ]
        theta_list = [0.0] * len(gid_list)
        back.ibs.add_annots(gid_list, bbox_list, theta_list)
        if refresh:
            back.front.update_tables([gh.IMAGE_TABLE, gh.ANNOTATION_TABLE])

    @blocking_slot()
    def reselect_annotation(back, aid=None, bbox=None, refresh=True, **kwargs):
        """ Action -> Reselect ANNOTATION"""
        if aid is None:
            aid = back.get_selected_aid()
        gid = back.ibs.get_annot_gids(aid)
        if bbox is None:
            bbox = back.select_bbox(gid)
        print('[back] reselect_annotation')
        back.ibs.set_annot_bboxes([aid], [bbox])
        if refresh:
            back.front.update_tables([gh.ANNOTATION_TABLE])
            back.show_image(gid)

    @blocking_slot()
    def reselect_ori(back, aid=None, theta=None, **kwargs):
        """ Action -> Reselect ORI"""
        print('[back] reselect_ori')
        raise NotImplementedError()
        pass

    @blocking_slot()
    def delete_image_annotations(back, gid_list):
        aid_list = ut.flatten(back.ibs.get_image_aids(gid_list))
        back.delete_annot(aid_list)

    @blocking_slot()
    def delete_annot(back, aid_list=None):
        """ Action -> Delete Chip"""
        print('[back] delete_annot, aid_list = %r' % (aid_list, ))
        if aid_list is None:
            aid_list = [back.get_selected_aid()]
        if not back.are_you_sure():
            return
        # get the image-id of the annotation we are deleting
        gid_list = back.ibs.get_annot_gids(aid_list)
        # delete the annotation
        back.ibs.delete_annots(aid_list)
        # Select only one image
        try:
            if len(gid_list) > 0:
                gid = gid_list[0]
        except AttributeError:
            gid = gid_list
        back.select_gid(gid, show=False)
        # update display, to show image without the deleted annotation
        back.front.update_tables()

    @blocking_slot()
    def unset_names(back, aid_list):
        print('[back] unset_names')
        if not back.are_you_sure():
            return
        back.ibs.set_annot_names(aid_list, [const.UNKNOWN] * len(aid_list))
        back.front.update_tables()

    @blocking_slot()
    def toggle_thumbnails(back):
        ibswgt = back.front
        tabwgt = ibswgt._tab_table_wgt
        index = tabwgt.currentIndex()
        tblname = ibswgt.tblname_list[index]
        view = ibswgt.views[tblname]
        col_name_list = view.col_name_list
        if 'thumb' in col_name_list:
            idx = col_name_list.index('thumb')
            view.col_hidden_list[idx] = not view.col_hidden_list[idx]
            view.hide_cols()
            #view.resizeRowsToContents() Too slow to use
        back.front.update_tables()

    @blocking_slot(int)
    def delete_image(back, gid_list=None):
        """ Action -> Delete Images"""
        print('[back] delete_image, gid_list = %r' % (gid_list, ))
        if gid_list is None or gid_list is False:
            gid_list = [back.get_selected_gid()]
        if not back.are_you_sure():
            return
        back.ibs.delete_images(gid_list)
        back.front.update_tables()

    @blocking_slot()
    def delete_all_encounters(back):
        print('\n\n[back] delete all encounters')
        if not back.are_you_sure():
            return
        back.ibs.delete_all_encounters()
        back.ibs.update_special_encounters()
        back.front.update_tables()

    @blocking_slot(int)
    def delete_encounter_and_images(back, eid_list):
        print('\n\n[back] delete_encounter_and_images')
        if back.contains_special_encounters(eid_list):
            back.display_special_encounters_error()
            return
        if not back.are_you_sure():
            return
        gid_list = ut.flatten(back.ibs.get_encounter_gids(eid_list))
        back.ibs.delete_images(gid_list)
        back.ibs.delete_encounters(eid_list)
        back.ibs.update_special_encounters()
        back.front.update_tables()

    @blocking_slot(int)
    def delete_encounter(back, eid_list):
        print('\n\n[back] delete_encounter')
        if back.contains_special_encounters(eid_list):
            back.display_special_encounters_error()
            return
        if not back.are_you_sure():
            return
        back.ibs.delete_encounters(eid_list)
        back.ibs.update_special_encounters()
        back.front.update_tables()

    @blocking_slot(int)
    def export_encounters(back, eid_list):
        print('\n\n[back] export encounter')

        #new_dbname = back.user_input(
        #    msg='What do you want to name the new database?',
        #    title='Export to New Database')
        #if new_dbname is None or len(new_dbname) == 0:
        #    print('Abort export to new database. new_dbname=%r' % new_dbname)
        #    return
        back.ibs.export_encounters(eid_list, new_dbdir=None)

    @blocking_slot()
    def train_rf_with_encounter(back, **kwargs):
        from ibeis.model.detect import randomforest
        eid = back._eidfromkw(kwargs)
        if eid < 0:
            gid_list = back.ibs.get_valid_gids()
        else:
            gid_list = back.ibs.get_valid_gids(eid=eid)
        species = back.ibs.cfg.detect_cfg.species_text
        if species == 'none':
            species = None
        print("[train_rf_with_encounter] Training Random Forest trees with enc=%r and species=%r" % (eid, species, ))
        randomforest.train_gid_list(back.ibs, gid_list, teardown=False, species=species)

    @blocking_slot(int)
    def merge_encounters(back, eid_list, destination_eid):
        assert len(eid_list) > 1, "Cannot merge fewer than two encounters"
        print('[back] merge_encounters: %r, %r' % (destination_eid, eid_list))
        if back.contains_special_encounters(eid_list):
            back.display_special_encounters_error()
            return
        ibs = back.ibs
        try:
            destination_index = eid_list.index(destination_eid)
        except:
            # Default to the first value selected if the eid doesn't exist in eid_list
            print('[back] merge_encounters cannot find index for %r' % (destination_eid,))
            destination_index = 0
            destination_eid = eid_list[destination_index]
        deprecated_eids = list(eid_list)
        deprecated_eids.pop(destination_index)
        gid_list = ut.flatten([ ibs.get_valid_gids(eid=eid) for eid in eid_list] )
        eid_list = [destination_eid] * len(gid_list)
        ibs.set_image_eids(gid_list, eid_list)
        ibs.delete_encounters(deprecated_eids)
        for eid in deprecated_eids:
            back.front.enc_tabwgt._close_tab_with_eid(eid)
        back.front.update_tables([gh.ENCOUNTER_TABLE], clear_view_selection=True)

    @blocking_slot(list)
    def remove_from_encounter(back, gid_list):
        eid = back.get_selected_eid()
        back.ibs.unrelate_images_and_encounters(gid_list, [eid] * len(gid_list))
        back.ibs.update_special_encounters()
        back.front.update_tables([gh.IMAGE_TABLE, gh.ENCOUNTER_TABLE], clear_view_selection=True)

    @blocking_slot(list)
    def send_to_new_encounter(back, gid_list, mode='move'):
        assert len(gid_list) > 0, "Cannot create a new encounter with no images"
        print('\n\n[back] send_to_new_encounter')
        ibs = back.ibs
        enctext = const.NEW_ENCOUNTER_ENCTEXT
        enctext_list = [enctext] * len(gid_list)
        ibs.set_image_enctext(gid_list, enctext_list)
        eid = back.get_selected_eid()
        eid_list = [eid] * len(gid_list)
        if mode == 'move':
            ibs.unrelate_images_and_encounters(gid_list, eid_list)
        elif mode == 'copy':
            pass
        else:
            raise AssertionError('invalid mode=%r' % (mode,))
        back.front.update_tables([gh.IMAGE_TABLE, gh.ENCOUNTER_TABLE], clear_view_selection=True)

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
    def encounter_set_species(back, refresh=True):
        """
        HACK: sets the species columns of all annotations in the encounter
        to be whatever is currently in the detect config
        """
        print('[back] encounter_set_species')
        ibs = back.ibs
        eid = back.get_selected_eid()
        aid_list = back.ibs.get_valid_aids(eid=eid)
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
            cfgname = 'cfg'
        else:
            cfgname = species_text
        ibs._load_named_config(cfgname)
        ibs.cfg.detect_cfg.species_text = species_text
        ibs.cfg.save()

    def get_selected_species(back):
        species_text = back.ibs.cfg.detect_cfg.species_text
        if species_text == 'none':
            species_text = None
        return species_text

    @blocking_slot()
    def change_query_mode(back, index, value):
        print('[back] change_query_mode(%r, %r)' % (index, value))
        back.query_mode = value
        #ibs = back.ibs
        #ibs.cfg.detect_cfg.species_text = value
        #ibs.cfg.save()

    @blocking_slot()
    def run_detection(back, refresh=True, **kwargs):
        print('\n\n')
        eid = back._eidfromkw(kwargs)
        ibs = back.ibs
        gid_list = ibsfuncs.get_empty_gids(ibs, eid=eid)
        species = ibs.cfg.detect_cfg.species_text
        # Construct message
        msg_fmtstr_list = ['You are about to run detection...']
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
        msg_fmtstr_list += ['Press \'Yes\' to continue']
        msg_fmtstr = '\n'.join(msg_fmtstr_list)
        msg_str = msg_fmtstr.format(**fmtdict)
        if back.are_you_sure(use_msg=msg_str):
            print('[back] run_detection(species=%r, eid=%r)' % (species, eid))
            ibs.detect_random_forest(gid_list, species)
            print('[back] about to finish detection')
            if refresh:
                back.front.update_tables([gh.IMAGE_TABLE, gh.ANNOTATION_TABLE])
            print('[back] finished detection')

    @blocking_slot()
    def compute_feats(back, refresh=True, **kwargs):
        """ Batch -> Precompute Feats"""
        print('[back] compute_feats')
        eid = back._eidfromkw(kwargs)
        ibsfuncs.compute_all_features(back.ibs, eid=eid)
        if refresh:
            back.front.update_tables()

    @blocking_slot()
    def compute_thumbs(back, refresh=True, **kwargs):
        """ Batch -> Precompute Thumbs"""
        print('[back] compute_thumbs')
        eid = back._eidfromkw(kwargs)
        ibsfuncs.compute_all_thumbs(back.ibs, eid=eid)
        if refresh:
            back.front.update_tables()

    def get_selected_qaids(back, eid=None, nojunk=True, is_known=None):
        species = back.get_selected_species()
        valid_kw = dict(
            eid=eid,
            nojunk=nojunk,
            is_known=is_known,
            species=species,
        )
        qaid_list = back.ibs.get_valid_aids(**valid_kw)
        return qaid_list

    def get_selected_daids(back, eid=None, query_mode=None):
        query_mode = back.query_mode if query_mode is None else query_mode
        query_mode_valid_kw_dict = {
            const.VS_EXEMPLARS_KEY: {
                'is_exemplar': True,
            },
            const.INTRA_ENC_KEY: {
                'eid': eid,
            },
        }
        valid_kw = {
            'species': back.get_selected_species(),
            'nojunk':  True,
        }
        mode_str = {
            const.VS_EXEMPLARS_KEY: 'vs_exemplar',
            const.INTRA_ENC_KEY: 'intra_encounter',
        }[query_mode]
        valid_kw.update(query_mode_valid_kw_dict[query_mode])
        print('[back] get_selected_daids: ' + mode_str)
        print('[back] ... valid_kw = ' + ut.dict_str(valid_kw))
        daid_list = back.ibs.get_valid_aids(**valid_kw)
        return daid_list

    def make_confirm_query_msg(back, daid_list, qaid_list):
        r"""
        Args:
            daid_list (list):
            qaid_list (list):

        CommandLine:
            python -m ibeis.gui.guiback --test-confirm_query_dialog

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.gui.guiback import *  # NOQA
            >>> import ibeis
            >>> # build test data
            >>> main_locals = ibeis.main(defaultdb='testdb1')
            >>> ibs, back = ut.dict_take(main_locals, ['ibs', 'back'])
            >>> daid_list = [1, 2, 3, 4, 5]
            >>> qaid_list = [4, 5, 6, 7, 8, 9]
            >>> # execute function
            >>> result = back.make_confirm_query_msg(daid_list, qaid_list)
            >>> # verify results
            >>> print(result)
        """
        ibs = back.ibs
        species_dict = dict(zip(const.VALID_SPECIES, const.SPECIES_NICE))

        def pluralize(wordtext, list_):
            return wordtext + 's' if len(list_) > 1 else wordtext

        def get_unique_species_phrase(aid_list):
            def boldspecies(species):
                species_bold_nice = '\'%s\'' % (species_dict.get(species, species).upper(),)
                return species_bold_nice
            species_list = list(set(ibs.get_annot_species_texts(aid_list)))
            species_nice_list = list(map(boldspecies, species_list))
            species_phrase = ut.cond_phrase(species_nice_list, 'and')
            return species_phrase

        # Build confirmation message
        fmtdict = dict()
        msg_fmtstr_list = ['You are about to run identification...']
        # Append database information to query confirmation
        if daid_list is not None:
            msg_fmtstr_list += ['    Database annotations: {num_daids}']
            msg_fmtstr_list += ['    Database species:         {d_species_phrase}']
            fmtdict['d_annotation_s']  = pluralize('annotation', daid_list)
            fmtdict['num_daids'] = len(daid_list)
            fmtdict['d_species_phrase'] = get_unique_species_phrase(daid_list)
        # Append query information to query confirmation
        if qaid_list is not None:
            msg_fmtstr_list += ['    Query annotations: {num_qaids}']
            msg_fmtstr_list += ['    Query species:         {q_species_phrase}']
            fmtdict['q_annotation_s']  = pluralize('annotation', qaid_list)
            fmtdict['num_qaids'] = len(qaid_list)
            fmtdict['q_species_phrase'] = get_unique_species_phrase(qaid_list)

        if qaid_list is not None and daid_list is not None:
            overlap_aids = ut.list_intersection(daid_list, qaid_list)
            num_overlap = len(overlap_aids)
            msg_fmtstr_list += ['    Num Overlap: {num_overlap}']
            fmtdict['num_overlap'] = num_overlap

        # Finish building confirmation message
        msg_fmtstr_list += ['']
        msg_fmtstr_list += ['Press \'Yes\' to continue']
        msg_fmtstr = '\n'.join(msg_fmtstr_list)
        msg_str = msg_fmtstr.format(**fmtdict)
        return msg_str

    def confirm_query_dialog(back, daid_list=None, qaid_list=None):
        msg_str = back.make_confirm_query_msg(daid_list, qaid_list)
        confirm_kw = dict(use_msg=msg_str, title='Begin Identification?', default='Yes')
        if not back.are_you_sure(**confirm_kw):
            raise guiexcept.UserCancel

    #@blocking_slot()
    #def query(back, aid=None, refresh=True, query_mode=None, **kwargs):
    #    """ Action -> Query

    #    queries a single annotation vs the exemplars or intra encounter
    #    """
    #    if aid is None:
    #        aid = back.get_selected_aid()
    #    eid = back._eidfromkw(kwargs)
    #    print('------')
    #    print('\n\n[back] query: eid=%r, mode=%r' % (eid, back.query_mode))
    #    if query_mode is None:
    #        query_mode = back.query_mode
    #    # Get the query annotation ids to search
    #    qaid_list = [aid]
    #    daid_list = back.get_selected_daids(eid=eid, query_mode=query_mode)
    #    back.confirm_query_dialog(daid_list, qaid_list)
    #    # Get the database annotation ids to be searched
    #    # Execute Query
    #    if len(daid_list) == 0:
    #        raise guiexcept.InvalidRequest('No exemplars set for this species')
    #    qaid2_qres = back.ibs._query_chips4(qaid_list, daid_list)
    #    if query_mode == const.INTRA_ENC_KEY:
    #        # HACK IN ENCOUNTER INFO
    #        for qres in six.itervalues(qaid2_qres):
    #            qres.eid = eid
    #    qres = qaid2_qres[aid]
    #    #back._set_selection(sel_qres=[qres])
    #    if refresh:
    #        #back.populate_tables(qres=True, default=False)
    #        back.review_queries(qaid2_qres, eid=eid)

    @blocking_slot()
    def compute_queries(back, refresh=True, query_mode=None, query_is_known=None, qaid_list=None, use_visual_selection=False, **kwargs):
        """
        Batch -> Compute OldStyle Queries
        and Actions -> Query

        Computes query results for all annotations in an encounter.
        Results are either vs-exemplar or intra-encounter

        CommandLine:
            ./main.py --query 1 -y
        """
        eid = back._eidfromkw(kwargs)
        print('------')
        print('\n\n[back] compute_queries: eid=%r, mode=%r' % (eid, back.query_mode))
        if eid is None:
            print('[back] invalid eid')
            return
        #back.compute_feats(refresh=False, **kwargs)
        # Get the query annotation ids to search and
        # the database annotation ids to be searched
        if qaid_list is None:
            if use_visual_selection:
                # old style Actions->Query execution
                qaid_list = [back.get_selected_aid()]
                #qaid_list = back.get_selected_qaids(eid=eid, is_known=query_is_known)
            else:
                # if not visual selection, then qaids are selected by encounter
                qaid_list = back.get_selected_qaids(eid=eid, is_known=query_is_known)
        daid_list = back.get_selected_daids(eid=eid, query_mode=query_mode)
        if len(qaid_list) == 0:
            raise guiexcept.InvalidRequest('No unknown query exemplars')
        back.confirm_query_dialog(daid_list, qaid_list)
        qaid2_qres = back.ibs._query_chips4(qaid_list, daid_list)
        # HACK IN ENCOUNTER INFO
        if query_mode == const.INTRA_ENC_KEY:
            for qres in six.itervalues(qaid2_qres):
                qres.eid = eid
        back.encounter_query_results[eid].update(qaid2_qres)
        print('[back] About to finish compute_queries: eid=%r' % (eid,))
        back.review_queries(qaid2_qres=qaid2_qres, eid=eid)
        if refresh:
            back.front.update_tables()
        print('[back] FINISHED compute_queries: eid=%r' % (eid,))

    #@blocking_slot()
    @slot_()
    @backreport
    def incremental_query(back, refresh=True, **kwargs):
        r"""

        Runs each query against the current database and allows for user
        interaction to add exemplars one at a time.

        CommandLine:
            python -m ibeis.gui.guiback --test-incremental_query

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.gui.guiback import *  # NOQA
            >>> import ibeis
            >>> main_locals = ibeis.main(db='testdb1')
            >>> # build test data
            >>> back = main_locals['back']
            >>> ibs = back.ibs
            >>> # execute function
            >>> refresh = True
            >>> kwargs = {}
            >>> back.incremental_query()
            >>> # verify results
            >>> print(result)
        """
        from ibeis.model.hots import qt_inc_automatch as iautomatch
        from ibeis.gui.guiheaders import NAMES_TREE  # ADD AS NEEDED
        eid = back._eidfromkw(kwargs)
        print('------')
        print('\n\n[back] incremental_query: eid=%r, mode=%r' % (eid, back.query_mode))
        if eid is None:
            print('[back] invalid eid')
            return
        # daid list is computed inside the incremental query so there is
        # no need to specify it here
        qaid_list = back.get_selected_qaids(eid=eid, is_known=False)
        if any(back.ibs.get_annot_exemplar_flags(qaid_list)):
            raise AssertionError('Database is not clean. There are unknown animals with exemplar_flag=True. Run Help->Fix/Clean Database')
        if len(qaid_list) == 0:
            msg = ut.codeblock(
                '''
                There are no annotations (of species=%r) left in this encounter.

                * Has the encounter been completed?
                * Is the species correctly set?
                * Do you need to run detection?
                ''') % (set(back.ibs.get_annot_species(qaid_list)),)
            back.user_info(msg=msg, title='Warning')
            return

        back.confirm_query_dialog(qaid_list=qaid_list)
        #TODO fix names tree thingie
        back.front.set_table_tab(NAMES_TREE)
        iautomatch.exec_interactive_incremental_queries(back.ibs, qaid_list, back=back)

    @blocking_slot()
    def review_detections(back, **kwargs):
        from plottool.interact_multi_image import MultiImageInteraction
        eid = back.get_selected_eid()
        ibs = back.ibs
        gid_list = ibs.get_valid_gids(eid=eid)
        gpath_list = ibs.get_image_paths(gid_list)
        bboxes_list = ibs.get_image_annotation_bboxes(gid_list)
        thetas_list = ibs.get_image_annotation_thetas(gid_list)
        multi_image_interaction = MultiImageInteraction(gpath_list, bboxes_list=bboxes_list, thetas_list=thetas_list)
        back.multi_image_interaction = multi_image_interaction

    @blocking_slot()
    def compute_encounters(back, refresh=True):
        """ Batch -> Compute Encounters """
        print('[back] compute_encounters')
        #back.ibs.delete_all_encounters()
        back.ibs.compute_encounters()
        back.ibs.update_special_encounters()
        print('[back] about to finish computing encounters')
        back.front.enc_tabwgt._close_all_tabs()
        if refresh:
            back.front.update_tables()
        print('[back] finished computing encounters')

    @blocking_slot()
    def encounter_reviewed_all_images(back, refresh=True, all_image_bypass=False):
        eid = back.get_selected_eid()
        if eid is not None or all_image_bypass:
            # Set all images to be reviewed
            gid_list = back.ibs.get_valid_gids(eid=eid)
            flag_list = [1] * len(gid_list)
            back.ibs.set_image_reviewed(gid_list, flag_list)
            # Set encounter to be processed
            back.ibs.set_encounter_processed_flags([eid], [1])
            back.ibs.wildbook_signal_eid_list([eid])
            back.front.enc_tabwgt._close_tab_with_eid(eid)
            if refresh:
                back.front.update_tables([gh.ENCOUNTER_TABLE])

    def send_unshipped_processed_encounters(back, refresh=True):
        processed_set = set(back.ibs.get_valid_eids(processed=True))
        shipped_set = set(back.ibs.get_valid_eids(shipped=True))
        eid_list = list(processed_set - shipped_set)
        back.ibs.wildbook_signal_eid_list(eid_list)

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
    @backreport
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
        print('[back] view_model_dir')
        ut.view_directory(ut.get_app_resource_dir('ibeis'))
        pass

    @slot_()
    @backreport
    def view_log_dir(back):
        print('[back] view_model_dir')
        ut.view_directory(back.ibs.get_logdir())
        pass

    @slot_()
    @backreport
    def redownload_detection_models(back):
        from ibeis import ibsfuncs
        print('[back] redownload_detection_models')
        if not back.are_you_sure():
            return
        ibsfuncs.redownload_detection_models(back.ibs)

    @slot_()
    @backreport
    def delete_cache(back):
        """ Help -> Delete Directory Slots"""
        print('[back] delete_cache')
        if not back.are_you_sure():
            return
        back.ibs.delete_cache()
        print('[back] finished delete_cache')
        pass

    @slot_()
    @backreport
    def delete_thumbnails(back):
        """ Help -> Delete Thumbnails """
        print('[back] delete_thumbnails')
        if not back.are_you_sure():
            return
        back.ibs.delete_thumbnails()
        print('[back] finished delete_thumbnails')
        pass

    @slot_()
    @backreport
    def delete_global_prefs(back):
        print('[back] delete_global_prefs')
        if not back.are_you_sure():
            return
        ut.delete(ut.get_app_resource_dir('ibeis', 'global_cache'))
        pass

    @slot_()
    @backreport
    def delete_queryresults_dir(back):
        print('[back] delete_queryresults_dir')
        if not back.are_you_sure():
            return
        ut.delete(back.ibs.qresdir)
        pass

    @blocking_slot()
    def dev_reload(back):
        """ Help -> Developer Reload"""
        print('[back] dev_reload')
        from ibeis.all_imports import reload_all
        reload_all()

    @blocking_slot()
    def dev_mode(back):
        """ Help -> Developer Mode"""
        print('[back] dev_mode')
        from ibeis import all_imports
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
        ut.view_directory(back.ibs._ibsdb)
        back.ibs.db.dump_tables_to_csv()

    @backreport
    def dev_export_annotations(back):
        ibs = back.ibs
        ibsfuncs.export_to_xml(ibs)

    @blocking_slot()
    def fix_and_clean_database(back):
        """ Help -> Fix/Clean Database """
        print('[back] Fix/Clean Database')
        back.ibs.fix_and_clean_database()
        back.front.update_tables()

    @blocking_slot()
    def run_consistency_checks(back):
        back.ibs.check_consistency()

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
            new_dbdir_options = ['Choose Directory', 'My Work Dir'],
            reply = back.user_option(
                msg='Where should I put the new database?',
                title='Import Images',
                options=new_dbdir_options,
                default=new_dbdir_options[1],
                use_cache=False)
            if reply == 'Choose Directory':
                print('[back] new_database(): SELECT A DIRECTORY')
                putdir = guitool.select_directory('Select new database directory', other_sidebar_dpaths=[back.get_work_directory()])
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

    @blocking_slot()
    def open_database(back, dbdir=None):
        """ File -> Open Database"""
        if dbdir is None:
            print('[back] new_database(): SELECT A DIRECTORY')
            #director
            dbdir = guitool.select_directory('Select new database directory', other_sidebar_dpaths=[back.get_work_directory()])
            if dbdir is None:
                return
        print('[back] open_database(dbdir=%r)' % dbdir)
        with ut.Indenter(lbl='    [opendb]'):
            try:
                ibs = IBEISControl.IBEISController(dbdir=dbdir)
                back.connect_ibeis_control(ibs)
            except Exception as ex:
                ut.printex(ex, 'caught Exception while opening database')
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
    def backup_database(back):
        """ File -> Backup Database"""
        print('[back] backup_database')
        back.ibs.backup_database()

    @blocking_slot()
    def import_images(back, gpath_list=None, dir_=None, refresh=True, clock_offset=True):
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
                                                    refresh=refresh, clock_offset=True)
        if reply == 'Directory' or dir_ is not None:
            gid_list = back.import_images_from_dir(dir_=dir_, refresh=refresh,
                                                   clock_offset=True)
        return gid_list

    @blocking_slot()
    def import_images_from_file(back, gpath_list=None, refresh=True, as_annots=False,
                                clock_offset=True):
        print('[back] import_images_from_file')
        """ File -> Import Images From File"""
        if back.ibs is None:
            raise ValueError('back.ibs is None! must open IBEIS database first')
        if gpath_list is None:
            gpath_list = guitool.select_images('Select image files to import')
        gid_list = back.ibs.add_images(gpath_list, as_annots=as_annots)
        return back._add_images(refresh, gid_list, clock_offset=clock_offset)

    @blocking_slot()
    def import_images_from_dir(back, dir_=None, size_filter=None, refresh=True,
                               clock_offset=True):
        """ File -> Import Images From Directory"""
        print('[back] import_images_from_dir')
        if dir_ is None:
            dir_ = guitool.select_directory('Select directory with images in it')
        #printDBG('[back] dir=%r' % dir_)
        if dir_ is None:
            return
        gpath_list = ut.list_images(dir_, fullpath=True, recursive=True)
        if size_filter is not None:
            raise NotImplementedError('Can someone implement the size filter?')
        gid_list = back.ibs.add_images(gpath_list)
        return back._add_images(refresh, gid_list, clock_offset=clock_offset)
        #print('')

    @blocking_slot()
    def import_images_with_smart(back, gpath_list=None, dir_=None, refresh=True):
        """ File -> Import Images with smart"""
        print('[back] import_images_with_smart')
        gid_list = back.import_images(gpath_list=gpath_list, dir_=dir_, refresh=refresh,
                                      clock_offset=False)
        back._add_images_with_smart_patrol(gid_list, refresh=refresh)

    @blocking_slot()
    def import_images_from_file_with_smart(back, gpath_list=None, refresh=True, as_annots=False):
        """ File -> Import Images From File with smart"""
        print('[back] import_images_from_file_with_smart')
        gid_list = back.import_images_from_file(gpath_list=gpath_list, refresh=refresh,
                                                as_annots=as_annots, clock_offset=False)
        back._add_images_with_smart_patrol(gid_list, refresh=refresh)

    @blocking_slot()
    def import_images_from_dir_with_smart(back, dir_=None, size_filter=None, refresh=True):
        """ File -> Import Images From Directory with smart"""
        print('[back] import_images_from_dir_with_smart')
        gid_list = back.import_images_from_dir(dir_=dir_, size_filter=size_filter,
                                               refresh=refresh, clock_offset=False)
        back._add_images_with_smart_patrol(gid_list, refresh=refresh)

    def _add_images_with_smart_patrol(back, gid_list, refresh=True):
        if gid_list is not None and len(gid_list) > 0:
            name_filter = 'XML Files (*.xml)'
            xml_path_list = guitool.select_files(caption='Select Patrol XML File:',
                                                 directory=None, name_filter=name_filter)
            # xml_path_list = ['/Users/bluemellophone/Desktop/LWC_000261.xml']
            assert len(xml_path_list) < 2, "Cannot specity more than one Patrol XML file"
            if len(xml_path_list) == 1:
                smart_xml_fpath = xml_path_list[0]
                back.ibs.compute_encounters_smart(gid_list, smart_xml_fpath)
        if refresh:
            ibsfuncs.update_ungrouped_special_encounter(back.ibs)
            back.front.update_tables([gh.ENCOUNTER_TABLE])

    def _add_images(back, refresh, gid_list, clock_offset=True):
        if refresh:
            back.ibs.update_special_encounters()
            back.front.update_tables([gh.IMAGE_TABLE, gh.ENCOUNTER_TABLE])
        if clock_offset:
            co_wgt = clock_offset_gui.ClockOffsetWidget(back.ibs, gid_list)
            co_wgt.show()
        return gid_list

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
        guitool.exit_application()

    #--------------------------------------------------------------------------
    # Helper functions
    #--------------------------------------------------------------------------

    def popup_annot_info(back, aid_list, **kwargs):
        if not isinstance(aid_list, list):
            aid_list = [aid_list]
        ibs = back.ibs
        gid_list  = ibs.get_annot_gids(aid_list)
        eids_list = ibs.get_image_eids(gid_list)
        for aid, gid, eids in zip(aid_list, gid_list, eids_list):
            back.user_info(msg='aid=%r, gid=%r, eids=%r' % (aid, gid, eids))

    def user_info(back, **kwargs):
        return guitool.user_info(parent=back.front, **kwargs)

    def user_input(back, **kwargs):
        return guitool.user_input(parent=back.front, **kwargs)

    def user_option(back, **kwargs):
        return guitool.user_option(parent=back.front, **kwargs)

    def are_you_sure(back, use_msg=None, title='Confirmation', default=None):
        """ Prompt user for conformation before changing something """
        msg = 'Are you sure?' if use_msg is None else use_msg
        print('[back] Asking User if sure')
        print('[back] title = %s' % (title,))
        print('[back] msg =\n%s' % (msg,))
        if ut.get_argflag('-y') or ut.get_argflag('--yes'):
            # DONT ASK WHEN SPECIFIED
            return True
        ans = back.user_option(msg=msg, title=title, options=['No', 'Yes'],
                               use_cache=False, default=default)
        return ans == 'Yes'

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

    def contains_special_encounters(back, eid_list):
        enctext_list = back.ibs.get_encounter_enctext(eid_list)
        is_valid = [enctext not in const.SPECIAL_ENCOUNTER_LABELS for enctext in enctext_list]
        #filtered_eid_list = ut.filter_items(eid_list, is_valid)
        return not all(is_valid)

    def display_special_encounters_error(back):
        back.user_info(msg="Contains special encounters")

    @slot_()
    def set_exemplars_from_quality_and_viewpoint(back):
        back.ibs.set_exemplars_from_quality_and_viewpoint()

    @slot_()
    def batch_rename_consecutive_via_species(back):
        back.ibs.batch_rename_consecutive_via_species()

    @slot_()
    def run_tests(back):
        from ibeis.tests import run_tests
        run_tests.run_tests()
