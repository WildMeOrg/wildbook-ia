# -*- coding: utf-8 -*-
"""
This module controls the GUI backend.  It is the layer between the GUI frontend
(newgui.py) and the IBEIS controller.  All the functionality of the nonvisual
gui components is written or called from here

TODO:
    open_database should not allow you to open subfolders
"""
from __future__ import absolute_import, division, print_function
import six  # NOQA
import sys
import functools
import traceback  # NOQA
import guitool
import utool as ut
from guitool import slot_, signal_, cast_from_qt
from guitool.__PYQT__ import QtCore
from ibeis import constants as const
from ibeis import ibsfuncs, sysres
from ibeis import viz
from ibeis.control import IBEISControl
from ibeis.gui import clock_offset_gui
from ibeis.gui import guiexcept
from ibeis.gui import guiheaders as gh
from ibeis.gui import newgui
from ibeis.viz import interact
from os.path import exists, join, dirname
from plottool import fig_presenter
from six.moves import zip
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[back]', DEBUG=False)

VERBOSE = ut.VERBOSE

WEB_URL = '127.0.0.1'
WEB_PORT = 5000
WEB_DOMAIN = '%s:%s' % (WEB_URL, WEB_PORT, )


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
            #error_msg = "Error caught while performing function. \n %r" % ex
            error_msg = 'Error: %s' % (ex,)
            import traceback  # NOQA
            detailed_msg = traceback.format_exc()
            guitool.msgbox(title="Error Catch!", msg=error_msg, detailed_msg=detailed_msg)
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
#QtReloadingMetaClass = ut.reloading_meta_metaclass_factory(guitool.QtCore.pyqtWrapperType)

GUIBACK_BASE = QtCore.QObject


#@six.add_metaclass(QtReloadingMetaClass)  # cant do this quit yet
class MainWindowBackend(GUIBACK_BASE):
    """
    Sends and recieves signals to and from the frontend
    """
    # Backend Signals
    updateWindowTitleSignal = signal_(str)
    #changeSpeciesSignal = signal_(str)
    #incQuerySignal = signal_(int)

    #------------------------
    # Constructor
    #------------------------
    def __init__(back, ibs=None):
        """ Creates GUIBackend object """
        #GUIBACK_BASE.__init__(back)
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
        if ut.is_developer():
            back.daids_mode = const.INTRA_OCCUR_KEY
        else:
            back.daids_mode = const.VS_EXEMPLARS_KEY
        #back.imageset_query_results = ut.ddict(dict)

        # Create GUIFrontend object
        back.mainwin = newgui.IBEISMainWindow(back=back, ibs=ibs)
        back.front = back.mainwin.ibswgt
        back.web_instance = None
        back.wb_server_running = None
        back.ibswgt = back.front  # Alias
        # connect signals and other objects
        fig_presenter.register_qt4_win(back.mainwin)
        # register self with the ibeis controller
        back.register_self()
        back.set_daids_mode(back.daids_mode)
        #back.changeSpeciesSignal.connect(back.ibswgt.species_combo.setItemText)

        #back.incQuerySignal.connect(back.incremental_query_slot)

    #def __del__(back):
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
            back.mainwin.actionToggleQueryMode.setText('Toggle Query Mode currently: %s' % back.daids_mode)
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

    def show_imgsetid_list_in_web(back, imgsetid_list, **kwargs):
        import webbrowser
        back.start_web_server_parallel(browser=False)

        if not isinstance(imgsetid_list, (tuple, list)):
            imgsetid_list = [imgsetid_list]
        if len(imgsetid_list) > 0:
            imgsetid_str = ','.join( map(str, imgsetid_list) )
        else:
            imgsetid_str = ''

        url = 'http://%s/view/images?imgsetid=%s' % (WEB_DOMAIN, imgsetid_str, )
        webbrowser.open(url)

    def show_image(back, gid, sel_aids=[], web=False, **kwargs):
        if web:
            import webbrowser
            back.start_web_server_parallel(browser=False)
            url = 'http://%s/turk/detection?gid=%s&refer=dmlldy9pbWFnZXM=' % (WEB_DOMAIN, gid, )
            webbrowser.open(url)
        else:
            kwargs.update({
                'sel_aids': sel_aids,
                'select_callback': back.select_gid,
            })
            interact.ishow_image(back.ibs, gid, **kwargs)

    def show_gid_list_in_web(back, gid_list, **kwargs):
        import webbrowser
        back.start_web_server_parallel(browser=False)

        if not isinstance(gid_list, (tuple, list)):
            gid_list = [gid_list]
        if len(gid_list) > 0:
            gid_list = ','.join( map(str, gid_list) )
        else:
            gid_list = ''

        url = 'http://%s/view/images?gid=%s' % (WEB_DOMAIN, gid_list, )
        webbrowser.open(url)

    def show_annotation(back, aid, show_image=False, web=False, **kwargs):
        if web:
            import webbrowser
            back.start_web_server_parallel(browser=False)
            url = 'http://%s/view/annotations?aid=%s' % (WEB_DOMAIN, aid, )
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
            aid_list = ','.join( map(str, aid_list) )
        else:
            aid_list = ''

        url = 'http://%s/view/annotations?aid=%s' % (WEB_DOMAIN, aid_list, )
        webbrowser.open(url)

    def show_name(back, nid, sel_aids=[], **kwargs):
        kwargs.update({
            'sel_aids': sel_aids,
            'select_aid_callback': back.select_aid,
        })
        #nid = back.ibs.get_name_rowids_from_text(name)
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
            aid_str = ','.join( map(str, aid_list) )
        else:
            aid_str = ''

        url = 'http://%s/view/names?aid=%s' % (WEB_DOMAIN, aid_str, )
        webbrowser.open(url)

    def show_hough_image(back, gid, **kwargs):
        viz.show_hough_image(back.ibs, gid, **kwargs)
        viz.draw()

    def run_detection_on_imageset(back, imgsetid_list, refresh=True, **kwargs):
        gid_list = ut.flatten(back.ibs.get_imageset_gids(imgsetid_list))
        back.run_detection_on_images(gid_list, refresh=refresh, **kwargs)

    def run_detection_on_images(back, gid_list, refresh=True, **kwargs):
        detector = back.ibs.cfg.detect_cfg.detector
        if detector in ['cnn_yolo', 'yolo', 'cnn']:
            back.ibs.detect_cnn_yolo(gid_list)
        elif detector in ['random_forest', 'rf']:
            species = back.ibs.cfg.detect_cfg.species_text
            back.ibs.detect_random_forest(gid_list, species)
        else:
            raise ValueError('Detector not recognized')
        if refresh:
            back.front.update_tables([gh.IMAGE_TABLE])

    def show_probability_chip(back, aid, **kwargs):
        viz.show_probability_chip(back.ibs, aid, **kwargs)
        viz.draw()

    @blocking_slot()
    def review_queries(back, cm_list, qreq_=None, **kwargs):
        # Qt QueryResults Interaction
        from ibeis.gui import inspect_gui
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

        kwargs['ranks_lt'] = kwargs.get('ranks_lt', ibs.cfg.other_cfg.ranks_lt)
        kwargs['qreq_'] = kwargs.get('qreq_', qreq_)

        #ibs.cfg.other_cfg.ranks_lt = 2
        # Overwrite
        ranks_lt = kwargs.pop('ranks_lt', ibs.cfg.other_cfg.ensure_attr('ranks_lt', 2))
        filter_reviewed = ibs.cfg.other_cfg.ensure_attr('filter_reviewed', True)
        if filter_reviewed is None:
            # only filter big queries if not specified
            filter_reviewed = len(cm_list) > 6
        print('REVIEW QUERIES')
        print('**kwargs = %s' % (ut.repr3(kwargs),))
        print('filter_reviewed = %s' % (filter_reviewed,))
        print('ranks_lt = %s' % (ranks_lt,))
        back.qres_wgt = inspect_gui.QueryResultsWidget(ibs, cm_list,
                                                       callback=finished_review_callback,
                                                       ranks_lt=ranks_lt,
                                                       filter_reviewed=filter_reviewed,
                                                       **kwargs)
        back.qres_wgt.show()
        back.qres_wgt.raise_()

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
        back.ibswgt.update_species_available(reselect=True)

    #@ut.indent_func
    def connect_ibeis_control(back, ibs):
        if ut.VERBOSE:
            print('[back] connect_ibeis(ibs=%r)' % (ibs,))
        if ibs is None:
            return None
        back.ibs = ibs
        # register self with the ibeis controller
        back.register_self()
        # deselect
        back._set_selection(sel_gids=[], sel_aids=[], sel_nids=[],
                            sel_imgsetids=[None])
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
        #aid = back.sel_aids[0]
        return back.sel_aids

    @ut.indent_func
    def get_selected_imgsetid(back):
        """ selected imageset id """
        if len(back.sel_imgsetids) == 0:
            raise guiexcept.InvalidRequest('There are no selected ImageSets')
        imgsetid = back.sel_imgsetids[0]
        return imgsetid

    @ut.indent_func
    def get_selected_qres(back):
        """
        UNUSED DEPRICATE

        selected query result """
        if len(back.sel_cm) > 0:
            cm = back.sel_cm[0]
            return cm
        else:
            return None

    #--------------------------------------------------------------------------
    # Selection Functions
    #--------------------------------------------------------------------------

    def _set_selection2(back, tablename, id_list, mode='set'):
        # here tablename is a backend const tablename

        def set_collections(old, aug):
            return ut.ensure_iterable(aug)

        def add_collections(old, aug):
            return list(set(old) | set(ut.ensure_iterable(aug)))

        def diff_collections(old, aug):
            return list(set(old) - set(ut.ensure_iterable(aug)))

        modify_collections = {'set': set_collections,
                              'add': add_collections,
                              'diff': diff_collections}[mode]

        attr_map = {
            const.ANNOTATION_TABLE : 'sel_aids',
            const.IMAGE_TABLE      : 'sel_gids',
            const.NAME_TABLE       : 'sel_nids',
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
            id_list = ensure_uuids_are_ids(id_list, back.ibs.get_annot_aids_from_visual_uuid)
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
        back.ibswgt.set_status_text(gh.IMAGESET_TABLE, repr(sel_imagesettexts,))
        back.ibswgt.set_status_text(gh.IMAGE_TABLE, repr(back.sel_gids,))
        back.ibswgt.set_status_text(gh.ANNOTATION_TABLE, repr(back.sel_aids,))
        back.ibswgt.set_status_text(gh.NAMES_TREE, repr(back.sel_nids,))

    def _set_selection(back, sel_gids=None, sel_aids=None, sel_nids=None,
                       sel_cm=None, sel_imgsetids=None, mode='set', **kwargs):
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
            back.ibswgt.set_status_text(gh.IMAGESET_TABLE, repr(sel_imagesettexts,))
        if sel_gids is not None:
            modify_collection_attr(back, 'sel_gids', sel_gids, mode)
            back.ibswgt.set_status_text(gh.IMAGE_TABLE, repr(back.sel_gids,))
        if sel_aids is not None:
            sel_aids = ut.ensure_iterable(sel_aids)
            back.sel_aids = sel_aids
            back.ibswgt.set_status_text(gh.ANNOTATION_TABLE, repr(back.sel_aids,))
        if sel_nids is not None:
            sel_nids = ut.ensure_iterable(sel_nids)
            back.sel_nids = sel_nids
            back.ibswgt.set_status_text(gh.NAMES_TREE, repr(back.sel_nids,))
        if sel_cm is not None:
            raise NotImplementedError('no select cm implemented')
            back.sel_sel_qres = sel_cm

    #@backblock
    def select_imgsetid(back, imgsetid=None, **kwargs):
        """ Table Click -> Result Table """
        imgsetid = cast_from_qt(imgsetid)
        if False:
            prefix = ut.get_caller_name(range(1, 8))
        else:
            prefix = ''
        print(prefix + '[back] select imageset imgsetid=%r' % (imgsetid))
        back._set_selection(sel_imgsetids=imgsetid, **kwargs)

    #@backblock
    def select_gid(back, gid, imgsetid=None, show=True, sel_aids=None, fnum=None, web=False, **kwargs):
        r"""
        Table Click -> Image Table

        Example:
            >>> # GUI_DOCTEST
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
        print('[back] select_gid(gid=%r, imgsetid=%r, sel_aids=%r)' % (gid, imgsetid, sel_aids))
        back._set_selection(sel_gids=gid, sel_aids=sel_aids, sel_imgsetids=imgsetid, **kwargs)
        if show:
            back.show_image(gid, sel_aids=sel_aids, fnum=fnum, web=web)

    #@backblock
    def select_gid_from_aid(back, aid, imgsetid=None, show=True, web=False):
        gid = back.ibs.get_annot_gids(aid)
        back.select_gid(gid, imgsetid=imgsetid, show=show, web=web, sel_aids=[aid])

    #@backblock
    def select_aid(back, aid, imgsetid=None, show=True, show_annotation=True, web=False, **kwargs):
        """ Table Click -> Chip Table """
        print('[back] select aid=%r, imgsetid=%r' % (aid, imgsetid))
        gid = back.ibs.get_annot_gids(aid)
        nid = back.ibs.get_annot_name_rowids(aid)
        back._set_selection(sel_aids=aid, sel_gids=gid, sel_nids=nid, sel_imgsetids=imgsetid, **kwargs)
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

    @backblock
    def select_qres_aid(back, aid, imgsetid=None, show=True, **kwargs):
        """ Table Click -> Result Table """
        imgsetid = cast_from_qt(imgsetid)
        aid = cast_from_qt(aid)
        print('[back] select result aid=%r, imgsetid=%r' % (aid, imgsetid))

    #--------------------------------------------------------------------------
    # Action menu slots
    #--------------------------------------------------------------------------

    @blocking_slot()
    def add_annotation_from_image(back, gid_list, refresh=True):
        """ Context -> Add Annotation from Image"""
        print('[back] add_annotation_from_image')
        assert isinstance(gid_list, list), 'must pass in list here'
        size_list = back.ibs.get_image_sizes(gid_list)
        bbox_list = [ (0, 0, w, h) for (w, h) in size_list ]
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
            python -m ibeis.gui.guiback --test-delete_annot --show
            python -m ibeis.gui.guiback --test-delete_annot --show --no-api-cache
            python -m ibeis.gui.guiback --test-delete_annot --show --assert-api-cache
            python -m ibeis.gui.guiback --test-delete_annot --show --debug-api-cache --yes

        SeeAlso:
            manual_annot_funcs.delete_annots

        Example:
            >>> # GUI_DOCTEST
            >>> from ibeis.gui.guiback import *  # NOQA
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
            >>> guitool.qtapp_loop(back.mainwin, frequency=100)
        """
        print('[back] delete_annot, aid_list = %r' % (aid_list, ))
        if aid_list is None:
            aid_list = back.get_selected_aids()
        if not back.are_you_sure(use_msg='Delete %d annotations?' % (len(aid_list))):
            return
        back._set_selection3(const.ANNOTATION_TABLE, aid_list, mode='diff')
        # get the image-id of the annotation we are deleting
        #gid_list = back.ibs.get_annot_gids(aid_list)
        # delete the annotation
        back.ibs.delete_annots(aid_list)
        # Select only one image
        #try:
        #    if len(gid_list) > 0:
        #        gid = gid_list[0]
        #except AttributeError:
        #    gid = gid_list
        #back.select_gid(gid, show=False)
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
        tabwgt = ibswgt._table_tab_wgt
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
        back.ibs.update_special_imagesets()
        back.front.update_tables()

    @blocking_slot()
    def update_special_imagesets(back):
        back.ibs.update_special_imagesets()
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
        back.ibs.update_special_imagesets()
        back.front.update_tables()

    @blocking_slot(int)
    def export_imagesets(back, imgsetid_list):
        print('\n\n[back] export imageset')

        #new_dbname = back.user_input(
        #    msg='What do you want to name the new database?',
        #    title='Export to New Database')
        #if new_dbname is None or len(new_dbname) == 0:
        #    print('Abort export to new database. new_dbname=%r' % new_dbname)
        #    return
        back.ibs.export_imagesets(imgsetid_list, new_dbdir=None)

    @blocking_slot()
    def train_rf_with_imageset(back, **kwargs):
        from ibeis.algo.detect import randomforest
        imgsetid = back._eidfromkw(kwargs)
        if imgsetid < 0:
            gid_list = back.ibs.get_valid_gids()
        else:
            gid_list = back.ibs.get_valid_gids(imgsetid=imgsetid)
        species = back.ibs.cfg.detect_cfg.species_text
        if species == 'none':
            species = None
        print("[train_rf_with_imageset] Training Random Forest trees with imgsetid=%r and species=%r" % (imgsetid, species, ))
        randomforest.train_gid_list(back.ibs, gid_list, teardown=False, species=species)

    @blocking_slot(int)
    def merge_imagesets(back, imgsetid_list, destination_imgsetid):
        assert len(imgsetid_list) > 1, "Cannot merge fewer than two imagesets"
        print('[back] merge_imagesets: %r, %r' % (destination_imgsetid, imgsetid_list))
        if back.contains_special_imagesets(imgsetid_list):
            back.display_special_imagesets_error()
            return
        ibs = back.ibs
        try:
            destination_index = imgsetid_list.index(destination_imgsetid)
        except:
            # Default to the first value selected if the imgsetid doesn't exist in imgsetid_list
            print('[back] merge_imagesets cannot find index for %r' % (destination_imgsetid,))
            destination_index = 0
            destination_imgsetid = imgsetid_list[destination_index]
        deprecated_imgsetids = list(imgsetid_list)
        deprecated_imgsetids.pop(destination_index)
        gid_list = ut.flatten([ ibs.get_valid_gids(imgsetid=imgsetid) for imgsetid in imgsetid_list] )
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
        back.ibs.update_special_imagesets()
        back.front.update_tables([gh.IMAGE_TABLE, gh.IMAGESET_TABLE], clear_view_selection=True)

    @blocking_slot(list)
    def send_to_new_imageset(back, gid_list, mode='move'):
        assert len(gid_list) > 0, "Cannot create a new imageset with no images"
        print('\n\n[back] send_to_new_imageset')
        ibs = back.ibs
        #imagesettext = const.NEW_IMAGESET_IMAGESETTEXT
        #imagesettext_list = [imagesettext] * len(gid_list)
        #ibs.set_image_imagesettext(gid_list, imagesettext_list)
        new_imgsetid = ibs.create_new_imageset_from_images(gid_list)  # NOQA
        if mode == 'move':
            imgsetid = back.get_selected_imgsetid()
            imgsetid_list = [imgsetid] * len(gid_list)
            ibs.unrelate_images_and_imagesets(gid_list, imgsetid_list)
        elif mode == 'copy':
            pass
        else:
            raise AssertionError('invalid mode=%r' % (mode,))
        back.ibs.update_special_imagesets()
        back.front.update_tables([gh.IMAGE_TABLE, gh.IMAGESET_TABLE], clear_view_selection=True)

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

        # TODO: incorporate this as a signal in guiback which connects to a slot in guifront
        back.front.detect_button.setEnabled(ibs.has_species_detector(species_text))

    def get_selected_species(back):
        species_text = back.ibs.cfg.detect_cfg.species_text
        if species_text == 'none':
            species_text = None
        print('species_text = %r' % (species_text,))

        if species_text is None or species_text == const.UNKNOWN:
            # hack to set species for user
            pass
            #species_text = back.ibs.get_primary_database_species()
            #print('\'species_text = %r' % (species_text,))
            #sig = signal_(str)
            #sig.connect(back.ibswgt.species_combo.setItemText)
            #back.ibswgt.species_combo.setItemText(species_text)
            #back.changeSpeciesSignal.emit(str(species_text))
            #sig.emit(species_text)
            #back.ibs.cfg.detect_cfg.species_text = species_text
        return species_text

    @blocking_slot()
    def change_daids_mode(back, index, value):
        print('[back] change_daids_mode(%r, %r)' % (index, value))
        back.daids_mode = value
        #ibs = back.ibs
        #ibs.cfg.detect_cfg.species_text = value
        #ibs.cfg.save()

    @blocking_slot()
    def run_detection(back, refresh=True, **kwargs):
        print('\n\n')
        imgsetid = back._eidfromkw(kwargs)
        ibs = back.ibs
        gid_list = ibsfuncs.get_empty_gids(ibs, imgsetid=imgsetid)

        detector = back.ibs.cfg.detect_cfg.detector
        if detector in ['cnn_yolo', 'yolo', 'cnn']:
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
            msg_fmtstr_list += ['Press \'Yes\' to continue']
            msg_fmtstr = '\n'.join(msg_fmtstr_list)
            msg_str = msg_fmtstr.format(**fmtdict)
            if back.are_you_sure(use_msg=msg_str):
                print('[back] run_detection(imgsetid=%r)' % (imgsetid))
                ibs.detect_cnn_yolo(gid_list)
                print('[back] about to finish detection')
                if refresh:
                    back.front.update_tables([gh.IMAGE_TABLE, gh.ANNOTATION_TABLE])
                print('[back] finished detection')
        elif detector in ['random_forest', 'rf']:
            species = ibs.cfg.detect_cfg.species_text
            # Construct message
            msg_fmtstr_list = ['You are about to run detection using Random Forests...']
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
                print('[back] run_detection(species=%r, imgsetid=%r)' % (species, imgsetid))
                ibs.detect_random_forest(gid_list, species)
                print('[back] about to finish detection')
                if refresh:
                    back.front.update_tables([gh.IMAGE_TABLE, gh.ANNOTATION_TABLE])
                print('[back] finished detection')
        else:
            raise ValueError('Detector not recognized')

    @blocking_slot()
    def compute_feats(back, refresh=True, **kwargs):
        """ Batch -> Precompute Feats"""
        print('[back] compute_feats')
        imgsetid = back._eidfromkw(kwargs)
        ibsfuncs.compute_all_features(back.ibs, imgsetid=imgsetid)
        if refresh:
            back.front.update_tables()

    @blocking_slot()
    def compute_thumbs(back, refresh=True, **kwargs):
        """ Batch -> Precompute Thumbs"""
        print('[back] compute_thumbs')
        imgsetid = back._eidfromkw(kwargs)
        back.ibs.preprocess_image_thumbs(imgsetid=imgsetid)
        if refresh:
            back.front.update_tables()

    def get_selected_qaids(back, imgsetid=None, minqual='poor', is_known=None):
        species = back.get_selected_species()

        valid_kw = dict(
            imgsetid=imgsetid,
            minqual=minqual,
            is_known=is_known,
            species=species,
        )
        qaid_list = back.ibs.get_valid_aids(**valid_kw)
        return qaid_list

    def get_selected_daids(back, imgsetid=None, daids_mode=None, qaid_list=None):
        daids_mode = back.daids_mode if daids_mode is None else daids_mode
        daids_mode_valid_kw_dict = {
            const.VS_EXEMPLARS_KEY: {
                'is_exemplar': True,
            },
            const.INTRA_OCCUR_KEY: {
                'imgsetid': imgsetid,
            },
            'all': {
            }
        }
        species = None
        if qaid_list is None:
            ibs = back.ibs
            hist_ = ut.dict_hist(ibs.get_annot_species_texts(qaid_list))
            if len(hist_) == 1:
                # select the query species if there is only one
                species = back.get_selected_species()

        if species is None:
            species = back.get_selected_species()

        valid_kw = {
            'species': back.get_selected_species(),
            'minqual':  'poor',
        }
        mode_str = {
            const.VS_EXEMPLARS_KEY: 'vs_exemplar',
            const.INTRA_OCCUR_KEY: 'intra_occurrence',
            'all': 'all'
        }[daids_mode]
        valid_kw.update(daids_mode_valid_kw_dict[daids_mode])
        print('[back] get_selected_daids: ' + mode_str)
        print('[back] ... valid_kw = ' + ut.dict_str(valid_kw))
        daid_list = back.ibs.get_valid_aids(**valid_kw)
        return daid_list

    def make_confirm_query_msg(back, daid_list, qaid_list, cfgdict=None, query_msg=None):
        r"""
        Args:
            daid_list (list):
            qaid_list (list):

        CommandLine:
            python -m ibeis.gui.guiback --test-MainWindowBackend.make_confirm_query_msg

        Example:
            >>> # GUI_DOCTEST
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
        species_text = ibs.get_all_species_texts()
        species_nice = ibs.get_all_species_nice()
        species_dict = dict(zip(species_text, species_nice))

        def get_unique_species_phrase(aid_list):
            def boldspecies(species):
                species_bold_nice = '\'%s\'' % (species_dict.get(species, species).upper(),)
                return species_bold_nice
            species_list = list(set(ibs.get_annot_species_texts(aid_list)))
            species_nice_list = list(map(boldspecies, species_list))
            species_phrase = ut.conj_phrase(species_nice_list, 'and')
            return species_phrase

        # Build confirmation message
        fmtdict = dict()
        msg_fmtstr_list = ['You are about to run identification...']
        if query_msg is not None:
            msg_fmtstr_list = [query_msg]
        msg_fmtstr_list += ['    -----']
        # Append database information to query confirmation
        if daid_list is not None:
            msg_fmtstr_list += ['    Database annotations: {num_daids}']
            msg_fmtstr_list += ['    Database species:         {d_species_phrase}']
            fmtdict['d_annotation_s']  = ut.pluralize('annotation', len(daid_list))
            fmtdict['num_daids'] = len(daid_list)
            fmtdict['d_species_phrase'] = get_unique_species_phrase(daid_list)
            if qaid_list is not None:
                msg_fmtstr_list += ['    -----']
        # Append query information to query confirmation
        if qaid_list is not None:
            msg_fmtstr_list += ['    Query annotations: {num_qaids}']
            msg_fmtstr_list += ['    Query species:         {q_species_phrase}']
            fmtdict['q_annotation_s']  = ut.pluralize('annotation', len(qaid_list))
            fmtdict['num_qaids'] = len(qaid_list)
            fmtdict['q_species_phrase'] = get_unique_species_phrase(qaid_list)

        if qaid_list is not None and daid_list is not None:
            overlap_aids = ut.list_intersection(daid_list, qaid_list)
            num_overlap = len(overlap_aids)
            msg_fmtstr_list += ['    -----']
            msg_fmtstr_list += ['    Num Overlap: {num_overlap}']
            fmtdict['num_overlap'] = num_overlap
        if cfgdict is not None and len(cfgdict) > 0:
            fmtdict['special_settings'] = ut.dict_str(cfgdict)
            msg_fmtstr_list += ['Special Settings: {special_settings}']

        # Finish building confirmation message
        msg_fmtstr_list += ['']
        msg_fmtstr_list += ['Press \'Yes\' to continue']
        msg_fmtstr = '\n'.join(msg_fmtstr_list)
        msg_str = msg_fmtstr.format(**fmtdict)
        return msg_str

    def confirm_query_dialog(back, daid_list=None, qaid_list=None,
                             cfgdict=None, query_msg=None):
        """
        Asks the user to confirm starting the identification query
        """
        msg_str = back.make_confirm_query_msg(
            daid_list, qaid_list, cfgdict=cfgdict, query_msg=query_msg)
        confirm_kw = dict(use_msg=msg_str, title='Begin Identification?',
                          default='Yes')
        if not back.are_you_sure(**confirm_kw):
            raise guiexcept.UserCancel

    def run_annot_splits(back, aid_list):
        """
        Checks for mismatches within a group of annotations

        Args:
            aid_list (int):  list of annotation ids

        CommandLine:
            python -m ibeis.gui.guiback --test-MainWindowBackend.run_annot_splits --show

        Example:
            >>> # GUI_DOCTEST
            >>> from ibeis.gui.guiback import *  # NOQA
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
            'score_method': 'csum'
        }
        ranks_lt = min(len(aid_list), 10)
        ibs = back.ibs
        qreq_ = ibs.new_query_request(aid_list, aid_list, cfgdict=cfgdict)
        back.confirm_query_dialog(aid_list, aid_list, cfgdict=cfgdict, query_msg='Checking for SPLIT cases (matching each annotation within a name)')
        cm_list = ibs.query_chips(qreq_=qreq_, return_cm=True)
        back.review_queries(cm_list, qreq_=qreq_,
                            filter_reviewed=False,
                            name_scoring=False,
                            ranks_lt=ranks_lt,
                            query_title='Annot Splits')

    def run_merge_checks(back):
        r"""
        Checks for missed matches within a group of annotations

        CommandLine:
            python -m ibeis.gui.guiback --test-run_merge_checks --show

        Example:
            >>> # GUI_DOCTEST
            >>> from ibeis.gui.guiback import *  # NOQA
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
            #'K': 3,
            #'Knorm': 3,
            #'prescore_method': 'csum',
            #'score_method': 'csum'
        }
        query_msg = 'Checking for MERGE cases (this is an exemplars-vs-exemplars query)'
        back.compute_queries(qaid_list=qaid_list, daids_mode=const.VS_EXEMPLARS_KEY,
                             query_msg=query_msg, cfgdict=cfgdict,
                             custom_qaid_list_title='Merge Candidates')

    @blocking_slot()
    def compute_queries(back, refresh=True, daids_mode=None,
                        query_is_known=None, qaid_list=None,
                        use_prioritized_name_subset=False,
                        use_visual_selection=False, cfgdict={},
                        query_msg=None,
                        custom_qaid_list_title=None,
                        **kwargs):
        """
        MAIN QUERY FUNCTION

        execute_query

        Batch -> Compute OldStyle Queries
        and Actions -> Query

        Computes query results for all annotations in an imageset.
        Results are either vs-exemplar or intra-imageset

        CommandLine:
            ./main.py --query 1 -y
            python -m ibeis --query 1 -y
            python -m ibeis --query 1:119 --db PZ_MTEST --nocache-query --nocache-nnmid -y
            python -m ibeis --query 1:119 --db PZ_MTEST --nocache-query --nocache-nnmid -y --force-all-progress
            python -m ibeis --query 1:119 --db PZ_MTEST --nocache-query --nocache-nnmid --hots-batch-size=3 -y
            python -m ibeis --query 1:119 --db PZ_MTEST --nocache-query --nocache-nnmid --hots-batch-size=3 -y
            python -m ibeis --query 1:119 --db PZ_MTEST --nocache-query --nocache-nnmid --hots-batch-size=32 -y

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.gui.guiback import *  # NOQA
            >>> import ibeis
            >>> main_locals = ibeis.main(db='testdb2')
            >>> # build test data
            >>> back = main_locals['back']
            >>> ibs = back.ibs
            >>> query_is_known = None
            >>> # execute function
            >>> refresh = True
            >>> daids_mode = None
            >>> imgsetid = None
            >>> kwargs = {}
            >>> # verify results
            >>> print(result)
        """
        imgsetid = back._eidfromkw(kwargs)
        daids_mode = back.daids_mode if daids_mode is None else daids_mode
        print('------')
        print('\n\n')
        print('[back] compute_queries: imgsetid=%r, mode=%r' % (imgsetid, back.daids_mode))
        print('[back] use_prioritized_name_subset = %r' % (use_prioritized_name_subset,))
        print('[back] use_visual_selection        = %r' % (use_visual_selection,))
        print('[back] daids_mode                  = %r' % (daids_mode,))
        print('[back] cfgdict                     = %r' % (cfgdict,))
        print('[back] query_is_known              = %r' % (query_is_known,))
        if imgsetid is None:
            print('[back] invalid imgsetid')
            return
        # Get the query annotation ids to search and
        # the database annotation ids to be searched
        query_title = ''

        if qaid_list is None:
            if use_visual_selection:
                # old style Actions->Query execution
                qaid_list = back.get_selected_aids()
                query_title += 'selection'
            else:
                # if not visual selection, then qaids are selected by imageset
                qaid_list = back.get_selected_qaids(imgsetid=imgsetid, is_known=query_is_known)
                query_title += 'imageset=' + back.ibs.get_imageset_text(imgsetid)
        else:
            if custom_qaid_list_title is None:
                custom_qaid_list_title = 'custom'
            query_title += custom_qaid_list_title
        if use_prioritized_name_subset:
            # you do get unknowns back in this list
            HACK = back.ibs.cfg.other_cfg.enable_custom_filter
            #True
            if not HACK:
                new_aid_list, new_flag_list = back.ibs.get_annot_quality_viewpoint_subset(
                    aid_list=qaid_list, annots_per_view=2, verbose=True)
                qaid_list = ut.compress(new_aid_list, new_flag_list)
            else:
                qaid_list = back.ibs.get_prioritized_name_subset(qaid_list, annots_per_name=2)
            query_title += ' priority_subset'

        if daids_mode == const.VS_EXEMPLARS_KEY:
            query_title += ' vs exemplars'
        elif daids_mode == const.INTRA_OCCUR_KEY:
            query_title += ' intra imageset'
        elif daids_mode == 'all':
            query_title += ' all'
        else:
            print('Unknown daids_mode=%r' % (daids_mode,))

        daid_list = back.get_selected_daids(imgsetid=imgsetid, daids_mode=daids_mode, qaid_list=None)
        if len(qaid_list) == 0:
            raise guiexcept.InvalidRequest('No query annotations. Is the species correctly set?')
        if len(daid_list) == 0:
            raise guiexcept.InvalidRequest('No database annotations. Is the species correctly set?')

        FILTER_HACK = True
        if FILTER_HACK:
            if not use_visual_selection:
                qaid_list = back.ibs.filter_aids_custom(qaid_list)
            daid_list = back.ibs.filter_aids_custom(daid_list)
        qreq_ = back.ibs.new_query_request(qaid_list, daid_list,
                                           cfgdict=cfgdict)
        back.confirm_query_dialog(daid_list, qaid_list, cfgdict=cfgdict,
                                  query_msg=query_msg)
        #if not ut.WIN32:
        #    progbar = guitool.newProgressBar(back.mainwin)
        #else:
        progbar = guitool.newProgressBar(None)  # back.front)
        progbar.setWindowTitle('querying')
        progbar.utool_prog_hook.set_progress(0)
        # Doesn't seem to work correctly
        #progbar.utool_prog_hook.show_indefinite_progress()
        progbar.utool_prog_hook.force_event_update()
        cm_list = back.ibs.query_chips(qreq_=qreq_,
                                       prog_hook=progbar.utool_prog_hook)
        progbar.close()
        del progbar
        # HACK IN IMAGESET INFO
        if daids_mode == const.INTRA_OCCUR_KEY:
            for cm in cm_list:
                #if cm is not None:
                cm.imgsetid = imgsetid
        print('[back] About to finish compute_queries: imgsetid=%r' % (imgsetid,))
        # Filter duplicate names if running vsexemplar
        filter_duplicate_namepair_matches = (daids_mode == const.VS_EXEMPLARS_KEY)

        back.review_queries(
            cm_list,
            filter_duplicate_namepair_matches=filter_duplicate_namepair_matches,
            qreq_=qreq_, query_title=query_title, **kwargs)
        if refresh:
            back.front.update_tables()
        print('[back] FINISHED compute_queries: imgsetid=%r' % (imgsetid,))

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
        """
        from ibeis.algo.hots import qt_inc_automatch as iautomatch
        from ibeis.gui.guiheaders import NAMES_TREE  # ADD AS NEEDED
        imgsetid = back._eidfromkw(kwargs)
        print('------')
        print('\n\n[back] incremental_query: imgsetid=%r, mode=%r' % (imgsetid, back.daids_mode))
        if imgsetid is None:
            print('[back] invalid imgsetid')
            return
        # daid list is computed inside the incremental query so there is
        # no need to specify it here
        restrict_incremental_to_unknowns = back.ibs.cfg.other_cfg.ensure_attr('restrict_incremental_to_unknowns', True)
        #restrict_incremental_to_unknowns = back.ibs.cfg.other_cfg.ensure_attr('restrict_incremental_to_unknowns', False)
        #import utool
        #utool.embed()
        print('restrict_incremental_to_unknowns = %r' % (restrict_incremental_to_unknowns,))
        if restrict_incremental_to_unknowns:
            is_known = True
        else:
            is_known = None
        qaid_list = back.get_selected_qaids(imgsetid=imgsetid, is_known=is_known)
        if is_known:
            if any(back.ibs.get_annot_exemplar_flags(qaid_list)):
                raise AssertionError('Database is not clean. There are unknown animals with exemplar_flag=True. Run Help->Fix/Clean Database')
        if len(qaid_list) == 0:
            msg = ut.codeblock(
                '''
                No annotations (of species=%r) remain in this imageset.

                * Has the imageset been completed?
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
        imgsetid = back.get_selected_imgsetid()
        ibs = back.ibs
        gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
        gpath_list = ibs.get_image_paths(gid_list)
        bboxes_list = ibs.get_image_annotation_bboxes(gid_list)
        thetas_list = ibs.get_image_annotation_thetas(gid_list)
        multi_image_interaction = MultiImageInteraction(gpath_list, bboxes_list=bboxes_list, thetas_list=thetas_list)
        back.multi_image_interaction = multi_image_interaction

    @blocking_slot()
    def compute_occurrences(back, refresh=True):
        """ Batch -> Compute ImageSets """
        print('[back] compute_occurrences')
        #back.ibs.delete_all_imagesets()
        back.ibs.compute_occurrences()
        back.ibs.update_special_imagesets()
        print('[back] about to finish computing imagesets')
        back.front.imageset_tabwgt._close_all_tabs()
        if refresh:
            back.front.update_tables()
        print('[back] finished computing imagesets')

    @blocking_slot()
    def imageset_reviewed_all_images(back, refresh=True, all_image_bypass=False):
        """
        Sets all imagesets as reviwed and ships them to wildbook
        """
        imgsetid = back.get_selected_imgsetid()
        if imgsetid is not None or all_image_bypass:
            # Set all images to be reviewed
            gid_list = back.ibs.get_valid_gids(imgsetid=imgsetid)
            #gid_list = ibs.get_imageset_gids(imgsetid)
            back.ibs.set_image_reviewed(gid_list, [1] * len(gid_list))
            # Set imageset to be processed
            back.ibs.set_imageset_processed_flags([imgsetid], [1])
            back.ibs.wildbook_signal_imgsetid_list([imgsetid])
            back.front.imageset_tabwgt._close_tab_with_imgsetid(imgsetid)
            if refresh:
                back.front.update_tables([gh.IMAGESET_TABLE])

    def send_unshipped_processed_imagesets(back, refresh=True):
        processed_set = set(back.ibs.get_valid_imgsetids(processed=True))
        shipped_set = set(back.ibs.get_valid_imgsetids(shipped=True))
        imgsetid_list = list(processed_set - shipped_set)
        back.ibs.wildbook_signal_imgsetid_list(imgsetid_list)

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
        assert back.ibs is not None, 'No database is loaded. Open a database to continue'
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

    @slot_()
    @backreport
    def view_logs(back):
        print('[back] view_model_dir')
        log_fpath = ut.get_current_log_fpath()
        log_text = back.ibs.get_current_log_text()
        guitool.msgbox('Click show details to view logs from log_fpath=%r' % (log_fpath,), detailed_msg=log_text)
        #ut.startfile(back.ibs.get_logdir())

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
        if not back.are_you_sure(use_msg=('Are you sure you want to delete the '
                                          'cached query results?')):
            return
        ut.delete(back.ibs.qresdir)
        pass

    @blocking_slot()
    def dev_reload(back):
        """ Help -> Developer Reload"""
        print('[back] dev_reload')
        #from ibeis.all_imports import reload_all
        back.ibs.rrr()
        #back.rrr()
        #reload_all()

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
        if back.ibs is not None:
            back.ibs.reset_table_cache()
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

    @slot_()
    @backreport
    def dev_export_annotations(back):
        ibs = back.ibs
        ibsfuncs.export_to_xml(ibs)

    def start_web_server_parallel(back, browser=True):
        import ibeis
        ibs = back.ibs
        if back.web_instance is None:
            print('[guiback] Starting web service')
            back.web_instance = ibeis.opendb_in_background(dbdir=ibs.get_dbdir(), web=True, browser=browser)
        else:
            print('[guiback] CANNOT START WEB SERVER: WEB INSTANCE ALREADY RUNNING')

    def kill_web_server_parallel(back):
        if back.web_instance is not None:
            print('[guiback] Stopping web service')
            back.web_instance.terminate()
            back.web_instance = None
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
            new_dbdir_options = ['Choose Directory', 'My Work Dir']
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
        """
        File -> Open Database

        Args:
            dbdir (None): (default = None)

        CommandLine:
            python -m ibeis.gui.guiback --test-open_database

        Example:
            >>> # GUI_DOCTEST
            >>> from ibeis.gui.guiback import *  # NOQA
            >>> back = testdata_guiback(defaultdb='testdb1')
            >>> testdb0 = sysres.db_to_dbdir('testdb0')
            >>> testdb1 = sysres.db_to_dbdir('testdb1')
            >>> print('[TEST] TEST_OPEN_DATABASE testdb1=%r' % testdb1)
            >>> back.open_database(testdb1)
            >>> print('[TEST] TEST_OPEN_DATABASE testdb0=%r' % testdb0)
            >>> back.open_database(testdb0)
            >>> import ibeis
            >>> #dbdir = join(ibeis.sysres.get_workdir(), 'PZ_MTEST', '_ibsdb')
            >>> dbdir = None
            >>> result = back.open_database(dbdir)
            >>> print(result)
        """
        if dbdir is None:
            print('[back] new_database(): SELECT A DIRECTORY')
            #director
            dbdir = guitool.select_directory('Open a database directory', other_sidebar_dpaths=[back.get_work_directory()])
            if dbdir is None:
                return
        print('[back] open_database(dbdir=%r)' % dbdir)
        with ut.Indenter(lbl='    [opendb]'):
            try:
                # should this use ibeis.opendb? probably. at least it should be
                # be request IBEISControl
                #ibs = IBEISControl.IBEISController(dbdir=dbdir)
                ibs = IBEISControl.request_IBEISController(dbdir=dbdir)
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

    #@blocking_slot()
    #def import_images(back, gpath_list=None, dir_=None, refresh=True, clock_offset=True):
    #    """ File -> Import Images (ctrl + i)"""
    #    print('[back] import_images')
    #    if back.ibs is None:
    #        raise ValueError('back.ibs is None! must open IBEIS database first')
    #    reply = None
    #    if gpath_list is None and dir_ is None:
    #        reply = back.user_option(
    #            msg='Import specific files or whole directory?',
    #            title='Import Images',
    #            options=['Files', 'Directory'],
    #            use_cache=False)
    #    if reply == 'Files' or gpath_list is not None:
    #        gid_list = back.import_images_from_file(gpath_list=gpath_list,
    #                                                refresh=refresh, clock_offset=True)
    #    if reply == 'Directory' or dir_ is not None:
    #        gid_list = back.import_images_from_dir(dir_=dir_, refresh=refresh,
    #                                               clock_offset=True)
    #    return gid_list

    @blocking_slot()
    def import_images_from_file(back, gpath_list=None, refresh=True, as_annots=False,
                                clock_offset=True):
        r"""
        File -> Import Images From File

        Example
            # GUI_DOCTEST
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
            >>> #
            >>> print('[TEST] * len(gid_list)=%r' % len(gid_list))
        """
        print('[back] import_images_from_file')
        if back.ibs is None:
            raise ValueError('back.ibs is None! must open IBEIS database first')
        if gpath_list is None:
            gpath_list = guitool.select_images('Select image files to import')
        gid_list = back.ibs.add_images(gpath_list, as_annots=as_annots)
        back._process_new_images(refresh, gid_list, clock_offset=clock_offset)
        return gid_list

    @blocking_slot()
    def import_images_from_dir(back, dir_=None, size_filter=None, refresh=True,
                               clock_offset=True, return_dir=False, defaultdir=None):
        """ File -> Import Images From Directory"""
        print('[back] import_images_from_dir')
        if dir_ is None:
            dir_ = guitool.select_directory('Select directory with images in it', directory=defaultdir)
        #printDBG('[back] dir=%r' % dir_)
        if dir_ is None:
            return
        gpath_list = ut.list_images(dir_, fullpath=True, recursive=True)
        if size_filter is not None:
            raise NotImplementedError('Can someone implement the size filter?')
        gid_list = back.ibs.add_images(gpath_list)
        back._process_new_images(refresh, gid_list, clock_offset=clock_offset)
        if return_dir:
            return gid_list, dir_
        else:
            return gid_list

        #print('')

    #@blocking_slot()
    #def import_images_with_smart(back, gpath_list=None, dir_=None, refresh=True):
    #    """ File -> Import Images with smart"""
    #    print('[back] import_images_with_smart')
    #    gid_list = back.import_images(gpath_list=gpath_list, dir_=dir_, refresh=refresh,
    #                                  clock_offset=False)
    #    back._group_images_with_smartxml(gid_list, refresh=refresh)

    #@blocking_slot()
    #def import_images_from_file_with_smart(back, gpath_list=None, refresh=True, as_annots=False):
    #    """ File -> Import Images From File with smart"""
    #    print('[back] import_images_from_file_with_smart')
    #    gid_list = back.import_images_from_file(gpath_list=gpath_list, refresh=refresh,
    #                                            as_annots=as_annots, clock_offset=False)
    #    back._group_images_with_smartxml(gid_list, refresh=refresh)

    @blocking_slot()
    def import_images_from_dir_with_smart(back, dir_=None, size_filter=None, refresh=True, smart_xml_fpath=None, defaultdir=None):
        """ File -> Import Images From Directory with smart

        Args:
            dir_ (None): (default = None)
            size_filter (None): (default = None)
            refresh (bool): (default = True)

        Returns:
            list: gid_list

        CommandLine:
            python -m ibeis.gui.guiback --test-import_images_from_dir_with_smart --show
            python -m ibeis.gui.guiback --test-import_images_from_dir_with_smart --show --auto

        Example:
            >>> # DEV_GUI_DOCTEST
            >>> from ibeis.gui.guiback import *  # NOQA
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
            >>> guitool.qtapp_loop(back.mainwin, frequency=100)
        """
        print('[back] import_images_from_dir_with_smart')
        gid_list, add_dir_ = back.import_images_from_dir(
            dir_=dir_, size_filter=size_filter, refresh=False,
            clock_offset=False, return_dir=True, defaultdir=defaultdir)
        back._group_images_with_smartxml(gid_list, refresh=refresh, smart_xml_fpath=smart_xml_fpath,
                                         defaultdir=dirname(add_dir_))

    def _group_images_with_smartxml(back, gid_list, refresh=True, smart_xml_fpath=None, defaultdir=None):
        """
        Clusters the newly imported images with smart xml file
        """
        if gid_list is not None and len(gid_list) > 0:
            if smart_xml_fpath is None:
                name_filter = 'XML Files (*.xml)'
                xml_path_list = guitool.select_files(caption='Select Patrol XML File:',
                                                     directory=defaultdir,
                                                     name_filter=name_filter,
                                                     single_file=True)
                # xml_path_list = ['/Users/bluemellophone/Desktop/LWC_000261.xml']
                assert len(xml_path_list) == 1, "Must specity one Patrol XML file"
                smart_xml_fpath = xml_path_list[0]
            back.ibs.compute_occurrences_smart(gid_list, smart_xml_fpath)
        if refresh:
            back.ibs.update_special_imagesets()
            #back.front.update_tables([gh.IMAGESET_TABLE])
            back.front.update_tables()

    def _process_new_images(back, refresh, gid_list, clock_offset=True):
        if refresh:
            back.ibs.update_special_imagesets()
            back.front.update_tables([gh.IMAGE_TABLE, gh.IMAGESET_TABLE])
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
        imgsetids_list = ibs.get_image_imgsetids(gid_list)
        for aid, gid, imgsetids in zip(aid_list, gid_list, imgsetids_list):
            back.user_info(msg='aid=%r, gid=%r, imgsetids=%r' % (aid, gid, imgsetids))

    def user_info(back, **kwargs):
        return guitool.user_info(parent=back.front, **kwargs)

    def user_input(back, **kwargs):
        return guitool.user_input(parent=back.front, **kwargs)

    def user_option(back, **kwargs):
        return guitool.user_option(parent=back.front, **kwargs)

    def are_you_sure(back, use_msg=None, title='Confirmation', default=None, action=None):
        """ Prompt user for conformation before changing something """
        if action is None:
            default_msg = 'Are you sure?'
        else:
            default_msg = 'Are you sure you want to %s?' % (action,)
        msg = default_msg if use_msg is None else use_msg
        print('[back] Asking User if sure')
        print('[back] title = %s' % (title,))
        print('[back] msg =\n%s' % (msg,))
        if ut.get_argflag('-y') or ut.get_argflag('--yes'):
            # DONT ASK WHEN SPECIFIED
            return True
        ans = back.user_option(msg=msg, title=title, options=['No', 'Yes'],
                               use_cache=False, default=default)
        print('[back] User answered: %r' % (ans,))
        return ans == 'Yes'

    def get_work_directory(back):
        return sysres.get_workdir()

    def user_select_new_dbdir(back):
        raise NotImplementedError()
        pass

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
        back.user_info(msg="Contains special imagesets")

    @slot_()
    def override_all_annotation_species(back):
        aid_list = back.ibs.get_valid_aids()
        species_text = back.get_selected_species()
        print('override_all_annotation_species. species_text = %r' % (species_text,))
        species_rowid = back.ibs.get_species_rowids_from_text(species_text)
        use_msg = ('Are you sure you want to change %d annotations species to %r?'
                   % (len(aid_list), species_text))
        if back.are_you_sure(use_msg=use_msg):
            print('performing override')
            back.ibs.set_annot_species_rowids(aid_list, [species_rowid] * len(aid_list))
            # FIXME: api-cache is broken here too
            back.ibs.reset_table_cache()

    @blocking_slot()
    def update_species_nice_name(back):
        from ibeis.control.manual_species_funcs import _convert_species_nice_to_code
        ibs = back.ibs
        species_text = back.get_selected_species()
        if species_text in [const.UNKNOWN, '']:
            back.user_info(msg="Cannot rename this species...")
            return
        species_rowid = ibs.get_species_rowids_from_text(species_text)
        species_nice = ibs.get_species_nice(species_rowid)
        new_species_nice = back.user_input(
            msg='Rename species\n    Name: %r \n    Tag:  %r' % (species_nice, species_text),
            title='Rename Species')
        if new_species_nice is not None:
            species_rowid = [species_rowid]
            new_species_nice = [new_species_nice]
            species_code = _convert_species_nice_to_code(new_species_nice)
            ibs._set_species_nice(species_rowid, new_species_nice)
            ibs._set_species_code(species_rowid, species_code)
            back.ibswgt.update_species_available(reselect=True, reselect_new_name=new_species_nice[0])

    @blocking_slot()
    def delete_selected_species(back):
        ibs = back.ibs
        species_text = back.get_selected_species()
        if species_text in [const.UNKNOWN, '']:
            back.user_info(msg="Cannot delete this species...")
            return
        species_rowid = ibs.get_species_rowids_from_text(species_text)
        species_nice = ibs.get_species_nice(species_rowid)

        msg_str = 'You are about to delete species\n    Name: %r \n    ' + \
                  'Tag:  %r\n\nDo you wish to continue?\nAll annotations ' + \
                  'with this species will be set to unknown.'
        msg_str = msg_str % (species_nice, species_text, )
        confirm_kw = dict(use_msg=msg_str, title='Delete Selected Species?',
                          default='No')
        if not back.are_you_sure(**confirm_kw):
            raise guiexcept.UserCancel
        ibs.delete_species([species_rowid])
        back.ibswgt.update_species_available(deleting=True)

    @slot_()
    def set_exemplars_from_quality_and_viewpoint(back):
        imgsetid = back.get_selected_imgsetid()
        print('set_exemplars_from_quality_and_viewpoint, imgsetid=%r' % (imgsetid,))
        back.ibs.set_exemplars_from_quality_and_viewpoint(imgsetid=imgsetid)

    @slot_()
    def batch_rename_consecutive_via_species(back):
        #imgsetid = back.get_selected_imgsetid()
        #back.ibs.batch_rename_consecutive_via_species(imgsetid=imgsetid)
        imgsetid = None
        print('batch_rename_consecutive_via_species, imgsetid=%r' % (imgsetid,))
        back.ibs.batch_rename_consecutive_via_species(imgsetid=imgsetid)

    @slot_()
    def run_tests(back):
        from ibeis.tests import run_tests
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
        from ibeis.tests import assert_modules
        detailed_msg = assert_modules.assert_modules()
        guitool.msgbox(msg="Running checks", title="Module Checks", detailed_msg=detailed_msg)

    @slot_()
    def display_dbinfo(back):
        r"""
        CommandLine:
            python -m ibeis.gui.guiback --test-display_dbinfo

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.gui.guiback import *  # NOQA
            >>> # build test data
            >>> back = testdata_guiback()
            >>> # execute function
            >>> result = back.display_dbinfo()
            >>> # verify results
            >>> print(result)
        """
        dbinfo = back.ibs.get_dbinfo_str()
        print(dbinfo)
        guitool.msgbox(msg=back.ibs.get_infostr(), title="DBInfo", detailed_msg=dbinfo)

    @slot_()
    def show_about_message(back):
        import ibeis
        version = ibeis.__version__
        about_msg = 'IBEIS version %s\nImage Based Ecological Information System\nhttp://ibeis.org/' % (version,)
        guitool.msgbox(msg=about_msg, title='About')

    @slot_()
    def take_screenshot(back):
        """ dev command only """
        from guitool.__PYQT__.QtGui import QPixmap
        print('TAKING SCREENSHOT')
        #screengrab_fpath = ut.truepath('~/latex/ibeis_userguide/figures/filemenu.jpg')
        screengrab_dpath = ut.truepath(ut.get_argval('--screengrab_dpath', type_=str, default='.'))
        screengrab_fname = ut.get_argval('--screengrab_fname', type_=str, default='screenshot')
        screengrab_fpath = ut.get_nonconflicting_path(join(screengrab_dpath, screengrab_fname + '_%d.jpg'))
        screenimg = QPixmap.grabWindow(back.mainwin.winId())
        screenimg.save(screengrab_fpath, 'jpg')
        if ut.get_argflag('--diskshow'):
            ut.startfile(screengrab_fpath)

    @slot_()
    def reconnect_controller(back):
        back.connect_ibeis_control(back.ibs)

    @slot_()
    def browse_wildbook(back):
        wb_base_url = back.ibs.get_wildbook_base_url()
        ut.get_prefered_browser().open(wb_base_url)

    @slot_()
    def install_wildbook(back):
        import ibeis
        ibeis.control.manual_wildbook_funcs.install_wildbook()

    @slot_()
    def startup_wildbook(back):
        import ibeis
        back.wb_server_running = True
        ibeis.control.manual_wildbook_funcs.startup_wildbook_server()

    @slot_()
    def shutdown_wildbook(back):
        import ibeis
        ibeis.control.manual_wildbook_funcs.shutdown_wildbook_server()
        back.wb_server_running = False

    @slot_()
    def force_wildbook_namechange(back):
        back.ibs.wildbook_signal_annot_name_changes()

    @slot_()
    def set_workdir(back):
        import ibeis
        ibeis.sysres.set_workdir(work_dir=None, allow_gui=True)

    @slot_()
    def launch_ipy_notebook(back):
        from ibeis.templates import generate_notebook
        generate_notebook.autogen_ipynb(back.ibs, launch=True)

    @slot_()
    def update_source_install(back):
        import ibeis
        from os.path import dirname
        repo_path = dirname(ut.truepath(ut.get_modpath_from_modname(ibeis, prefer_pkg=True)))
        with ut.ChdirContext(repo_path):
            command = ut.python_executable() + ' super_setup.py pull'
            ut.cmd(command)
        print('Done updating source install')


def testdata_guiback(defaultdb='testdb2', **kwargs):
    import ibeis
    print('launching ipython notebook')
    main_locals = ibeis.main(defaultdb=defaultdb, **kwargs)
    back = main_locals['back']
    return back


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.gui.guiback
        python -m ibeis.gui.guiback --allexamples
        python -m ibeis.gui.guiback --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
