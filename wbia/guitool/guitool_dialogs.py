# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from six.moves import map
from wbia.guitool.__PYQT__ import QtCore, QtGui  # NOQA
from wbia.guitool.__PYQT__ import QtWidgets  # NOQA
from wbia.guitool.__PYQT__.QtCore import Qt
import os
from os.path import dirname
import platform
from utool import util_cache, util_path
import utool as ut

ut.noinject(__name__, '[guitool.dialogs]', DEBUG=False)


SELDIR_CACHEID = 'guitool_selected_directory'


def _guitool_cache_write(key, val):
    """ Writes to global IBEIS cache """
    util_cache.global_cache_write(
        key, val, appname='wbia'
    )  # HACK, user should specify appname


def _guitool_cache_read(key, **kwargs):
    """ Reads from global IBEIS cache """
    return util_cache.global_cache_read(
        key, appname='wbia', **kwargs
    )  # HACK, user should specify appname


def are_you_sure(parent=None, msg=None, title='Confirmation', default=None):
    """ Prompt user for conformation before changing something """
    msg = 'Are you sure?' if msg is None else msg
    print('[gt] Asking User if sure')
    print('[gt] title = %s' % (title,))
    print('[gt] msg =\n%s' % (msg,))
    if ut.get_argflag('-y') or ut.get_argflag('--yes'):
        # DONT ASK WHEN SPECIFIED
        return True
    ans = user_option(
        parent=parent,
        msg=msg,
        title=title,
        options=['Yes', 'No'],
        use_cache=False,
        default=default,
    )
    return ans == 'Yes'


def user_option(
    parent=None,
    msg='msg',
    title='user_option',
    options=['Yes', 'No'],
    use_cache=False,
    default=None,
    detailed_msg=None,
):
    """
    Prompts user with several options with ability to save decision

    Args:
        parent (None):
        msg (str):
        title (str):
        options (list):
        use_cache (bool):
        default (str): default option

    Returns:
        str: reply

    CommandLine:
        python -m wbia.guitool.guitool_dialogs --test-user_option

    Example:
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.guitool_dialogs import *  # NOQA
        >>> import wbia.guitool as gt
        >>> gt.ensure_qtapp()
        >>> parent = None
        >>> msg = 'msg'
        >>> title = 'user_option'
        >>> options = ['Yes', 'No']
        >>> use_cache = False
        >>> default = 'Yes'
        >>> # execute function
        >>> detailed_msg = 'hi'
        >>> reply = user_option(parent, msg, title, options, use_cache, default, detailed_msg)
        >>> result = str(reply)
        >>> print(result)
        >>> # xdoctest: +REQUIRES(--show)
        >>> ut.quit_if_noshow()
        >>> #gt.guitool_main.qtapp_loop()
    """
    if ut.VERBOSE:
        print('[gt] user_option:\n %r: %s' % (title, msg))
    # Recall decision
    cache_id = title + msg
    if use_cache:
        reply = _guitool_cache_read(cache_id, default=None)
        if reply is not None:
            return reply
    # Create message box
    msgbox = _newMsgBox(msg, title, parent, resizable=detailed_msg is not None)
    #     _addOptions(msgbox, options)
    # def _addOptions(msgbox, options):
    # msgbox.addButton(QtWidgets.QMessageBox.Close)
    options = list(options)[::-1]
    for opt in options:
        role = QtWidgets.QMessageBox.ApplyRole
        msgbox.addButton(QtWidgets.QPushButton(opt), role)
    # Set default button
    if default is not None:
        assert default in options, 'default=%r is not in options=%r' % (default, options,)
        for qbutton in msgbox.buttons():
            if default == qbutton.text():
                msgbox.setDefaultButton(qbutton)
    if use_cache:
        # Add a remember me option if caching is on
        dontPrompt = _cacheReply(msgbox)

    if detailed_msg is not None:
        msgbox.setDetailedText(detailed_msg)
    # Wait for output
    optx = msgbox.exec_()
    if optx == QtWidgets.QMessageBox.Cancel:
        # User Canceled
        return None
    try:
        # User Selected an option
        reply = options[optx]
    except KeyError as ex:
        # This should be unreachable code.
        print('[gt] USER OPTION EXCEPTION !')
        print('[gt] optx = %r' % optx)
        print('[gt] options = %r' % options)
        print('[gt] ex = %r' % ex)
        raise
    # Remember decision if caching is on
    if use_cache and dontPrompt.isChecked():
        _guitool_cache_write(cache_id, reply)
    # Close the message box
    del msgbox
    return reply


def user_input(parent=None, msg='msg', title='user_input', text=''):
    r"""
    Args:
        parent (None):
        msg (str):
        title (str):

    Returns:
        str:

    CommandLine:
        python -m wbia.guitool.guitool_dialogs --test-user_input --show

    Example:
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.guitool_dialogs import *  # NOQA
        >>> parent = None
        >>> msg = 'msg'
        >>> title = 'user_input'
        >>> text = 'default text'
        >>> import wbia.guitool as gt
        >>> gt.ensure_qtapp()
        >>> dpath = user_input(parent, msg, title, text)
        >>> result = str(dpath)
        >>> print(result)
    """
    reply, ok = QtWidgets.QInputDialog.getText(parent, title, msg, text=text)
    if not ok:
        return None
    return str(reply)


def user_info(parent=None, msg='msg', title='user_info'):
    print('[gt] dlg.user_info title=%r, msg=%r' % (title, msg))
    msgbox = _newMsgBox(msg, title, parent)
    msgbox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msgbox.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msgbox.setModal(False)
    msgbox.open(msgbox.close)
    msgbox.show()


def user_question(msg):
    raise NotImplementedError('user_question')
    msgbox = QtWidgets.QMessageBox.question(None, '', 'lovely day?')
    return msgbox


def newFileDialog(
    directory_, other_sidebar_dpaths=[], use_sidebar_cwd=True, mode='open', exec_=False
):
    r"""
    Args:
        directory_ (?):
        other_sidebar_dpaths (list): (default = [])
        use_sidebar_cwd (bool): (default = True)
        _dialog_class_ (wrappertype): (default = <class 'PyQt5.QtWidgets.QFileDialog'>)

    Returns:
        ?: qdlg

    CommandLine:
        python -m wbia.guitool.guitool_dialogs newFileDialog --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.guitool_dialogs import *  # NOQA
        >>> import wbia.guitool
        >>> guitool.ensure_qtapp()
        >>> directory_ = '.'
        >>> _dialog_class_ = QtWidgets.QFileDialog
        >>> if ut.show_was_requested():
        >>>     files = newFileDialog(directory_, mode='save', exec_=True)
        >>>     print('files = %r' % (files,))
        >>> else:
        >>>     dlg = newFileDialog(directory_, mode='save', exec_=True)
    """
    qdlg = QtWidgets.QFileDialog()
    if mode == 'open':
        qdlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
    elif mode == 'save':
        qdlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    else:
        raise KeyError(mode)

    _set_dialog_sidebar(qdlg, directory_, use_sidebar_cwd, other_sidebar_dpaths)
    if exec_:
        qdlg.exec_()
        return qdlg.selectedFiles()
    else:
        return qdlg


def _set_dialog_sidebar(
    qdlg, directory_=None, use_sidebar_cwd=True, other_sidebar_dpaths=[]
):
    sidebar_urls = qdlg.sidebarUrls()[:]
    if use_sidebar_cwd:
        sidebar_urls.append(QtCore.QUrl.fromLocalFile(os.getcwd()))
    if directory_ is not None:
        sidebar_urls.append(QtCore.QUrl.fromLocalFile(directory_))
    sidebar_urls.extend(list(map(QtCore.QUrl.fromUserInput, other_sidebar_dpaths)))
    sidebar_urls = ut.unique(sidebar_urls)
    # print('sidebar_urls = %r' % (sidebar_urls,))
    qdlg.setSidebarUrls(sidebar_urls)


class QDirectoriesDialog(QtWidgets.QFileDialog):
    def __init__(self, *args):
        super(QDirectoriesDialog, self).__init__(*args)
        print('USING QDirectoriesDialog')
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.DirectoryOnly)

        list_view_list = self.findChildren(QtWidgets.QListView, 'listView')
        tree_view_list = self.findChildren(QtWidgets.QTreeView)
        for view in list_view_list + tree_view_list:
            view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
            # view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)


def newDirectoryDialog(
    caption,
    directory=None,
    other_sidebar_dpaths=[],
    use_sidebar_cwd=True,
    single_directory=True,
):
    # hack to fix the dialog window on ubuntu
    if 'ubuntu' in platform.platform().lower():
        qopt = (
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontUseNativeDialog
        )
    else:
        qopt = QtWidgets.QFileDialog.ShowDirsOnly
    if directory is None:
        directory = '.'
    qtkw = {'caption': caption, 'options': qopt, 'directory': directory}
    _dialog_class_ = QtWidgets.QFileDialog if single_directory else QDirectoriesDialog
    qdlg = _dialog_class_()
    _set_dialog_sidebar(qdlg, directory, use_sidebar_cwd, other_sidebar_dpaths)
    if single_directory:
        dpath_list = [str(qdlg.getExistingDirectory(None, **qtkw))]
    else:
        qdlg.exec_()
        dpath_list = qdlg.selectedFiles()
    return qdlg, dpath_list


def select_directory(
    caption='Select Directory',
    directory=None,
    other_sidebar_dpaths=[],
    use_sidebar_cwd=True,
):
    r"""
    Args:
        caption (str): (default = 'Select Directory')
        directory (None): default directory to start in (default = None)
        other_sidebar_dpaths (list): (default = [])
        use_sidebar_cwd (bool): (default = True)

    Returns:
        str: dpath

    CommandLine:
        python -m wbia.guitool.guitool_dialogs --test-select_directory

    Example:
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.guitool_dialogs import *  # NOQA
        >>> import wbia.guitool
        >>> guitool.ensure_qtapp()
        >>> # build test data
        >>> caption = 'Select Directory'
        >>> directory = None  # os.path.dirname(guitool.__file__)
        >>> # execute function
        >>> other_sidebar_dpaths = [os.path.dirname(ut.__file__)]
        >>> dpath = select_directory(caption, directory, other_sidebar_dpaths)
        >>> # verify results
        >>> result = str(dpath)
        >>> print(result)
    """
    print('[gt] select_directory(caption=%r, directory=%r)' % (caption, directory))
    if directory is None:
        directory_ = _guitool_cache_read(SELDIR_CACHEID, default='.')
    else:
        directory_ = directory
    # hack to fix the dialog window on ubuntu
    qdlg, dpath_list = newDirectoryDialog(
        caption, directory_, other_sidebar_dpaths, use_sidebar_cwd
    )
    assert len(dpath_list) <= 1
    dpath = None if len(dpath_list) == 0 else dpath_list[0]
    print('[gt] dialog returned dpath = %r' % dpath)
    if dpath == '' or dpath is None:
        dpath = None
        print('[gt] Cancel Select')
        return dpath
    else:
        _guitool_cache_write(SELDIR_CACHEID, dirname(dpath))
    print('[gt] Selected Directory: %r' % dpath)
    return dpath


def select_directories(
    caption='Select Folder(s)',
    directory=None,
    other_sidebar_dpaths=[],
    use_sidebar_cwd=True,
):
    r"""
    Args:
        caption (str): (default = 'Select Directory')
        directory (None): default directory to start in (default = None)
        other_sidebar_dpaths (list): (default = [])
        use_sidebar_cwd (bool): (default = True)

    Returns:
        str: dpath

    CommandLine:
        python -m wbia.guitool.guitool_dialogs --test-select_directory

    Example:
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.guitool_dialogs import *  # NOQA
        >>> import wbia.guitool
        >>> guitool.ensure_qtapp()
        >>> # build test data
        >>> caption = 'Select Directory'
        >>> directory = None  # os.path.dirname(guitool.__file__)
        >>> # execute function
        >>> other_sidebar_dpaths = [os.path.dirname(ut.__file__)]
        >>> dpath = select_directory(caption, directory, other_sidebar_dpaths)
        >>> # verify results
        >>> result = str(dpath)
        >>> print(result)
    """
    print('[gt] select_directory(caption=%r, directory=%r)' % (caption, directory))
    if directory is None:
        directory_ = _guitool_cache_read(SELDIR_CACHEID, default='.')
    else:
        directory_ = directory
    qdlg, dpath_list = newDirectoryDialog(
        caption,
        directory_,
        other_sidebar_dpaths,
        use_sidebar_cwd,
        single_directory=False,
    )
    print('[gt] dialog returned dpath_list = %r' % dpath_list)
    if dpath_list is None or len(dpath_list) == 0:
        dpath_list = []
        print('[gt] Cancel Select')
        return dpath_list
    else:
        for dpath in dpath_list:
            _guitool_cache_write(SELDIR_CACHEID, dirname(dpath))
    print('[gt] Selected Directories: %r' % dpath_list)
    return dpath_list


def select_images(caption='Select images:', directory=None):
    name_filter = _getQtImageNameFilter()
    return select_files(caption, directory, name_filter)


def select_files(
    caption='Select Files:',
    directory=None,
    name_filter=None,
    other_sidebar_dpaths=[],
    use_sidebar_cwd=True,
    single_file=False,
):
    """
    Selects one or more files from disk using a qt dialog

    Args:
        caption (str): (default = 'Select Files:')
        directory (None): default directory to start in (default = None)
        name_filter (None): (default = None)
        other_sidebar_dpaths (list): (default = [])
        use_sidebar_cwd (bool): (default = True)

    References:
        http://qt-project.org/doc/qt-4.8/qfiledialog.html

    CommandLine:
        python -m wbia.guitool.guitool_dialogs --test-select_files

    Example:
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.guitool_dialogs import *  # NOQA
        >>> import wbia.guitool
        >>> guitool.ensure_qtapp()
        >>> # build test data
        >>> caption = 'Select Files'
        >>> name_filter = 'Python Files (*.py)'
        >>> directory = os.path.dirname(guitool.__file__)
        >>> # execute function
        >>> other_sidebar_dpaths = [os.path.dirname(ut.__file__)]
        >>> dpath = select_files(caption, directory, name_filter, other_sidebar_dpaths, single_file=True)
        >>> # verify results
        >>> result = str(dpath)
        >>> print(result)
    """
    # print(caption)
    if directory is None:
        directory = _guitool_cache_read(SELDIR_CACHEID, default='.')
    # qdlg = QtWidgets.QFileDialog()
    qdlg = newFileDialog(directory, other_sidebar_dpaths=[], use_sidebar_cwd=True)
    kwargs = {
        'caption': caption,
        'directory': directory,
        'filter': name_filter,
    }
    if single_file:
        response = qdlg.getOpenFileName(**kwargs)
    else:
        response = qdlg.getOpenFileNames(**kwargs)
    qfile_list = response[0]
    file_list = list(map(str, qfile_list))
    print('[gt] Selected %d files' % len(file_list))
    _guitool_cache_write(SELDIR_CACHEID, directory)
    return file_list


# Prevent messageboxes from being garbage collected
__MESSAGE_BOXES__ = []


def _register_msgbox(msgbox):
    """ Dont let the message box lose scope """
    global __MESSAGE_BOXES__
    __MESSAGE_BOXES__.append(msgbox)

    @QtCore.pyqtSlot(QtCore.QObject)
    def _close_msgbox(qobj):
        global __MESSAGE_BOXES__
        __MESSAGE_BOXES__.remove(msgbox)

    msgbox.destroyed.connect(_close_msgbox)


def _newMsgBox(
    msg='', title='', parent=None, options=None, cache_reply=False, resizable=False
):
    if resizable:
        msgbox = ResizableMessageBox(parent)
    else:
        msgbox = QtWidgets.QMessageBox(parent)
    # msgbox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    # std_buts = QtWidgets.QMessageBox.Close
    # std_buts = QtWidgets.QMessageBox.NoButton
    std_buts = QtWidgets.QMessageBox.Cancel
    msgbox.setStandardButtons(std_buts)
    msgbox.setWindowTitle(title)
    msgbox.setText(msg)
    msgbox.setModal(parent is not None)
    return msgbox


class ResizableMessageBox(QtWidgets.QMessageBox):
    """
    References:
        http://stackoverflow.com/questions/2655354/how-to-allow-resizing-of-qmessagebox-in-pyqt4
    """

    def __init__(self, *args):
        QtWidgets.QMessageBox.__init__(self, *args)
        self.setSizeGripEnabled(True)

    def event(self, event):

        # print(event)
        # print(event.type())
        # print(ut.invert_dict(dict(QtCore.QEvent.__dict__))[event.type()])
        # print(event.spontaneous())
        # print(event.isAccepted())
        result = QtWidgets.QMessageBox.event(self, event)
        # print(event.isAccepted())
        # print('----')
        # if event != QtCore.QEvent.DeferredDelete:
        try:
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )

            textEdit = self.findChild(QtWidgets.QTextEdit)
            if textEdit is not None:
                textEdit.setMinimumHeight(0)
                textEdit.setMaximumHeight(16777215)
                textEdit.setMinimumWidth(0)
                textEdit.setMaximumWidth(16777215)
                textEdit.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
                )
        except RuntimeError as ex:
            if ut.VERBOSE:
                msg = 'Closing seems to cause C++ errors. Unsure how to fix properly.'
                ut.printex(ex, msg, iswarning=True, keys=['event', 'event.type()'])

        return result


def msgbox(msg='', title='msgbox', detailed_msg=None):
    """ Make a non modal critical QtWidgets.QMessageBox.

    CommandLine:
        python -m wbia.guitool.guitool_dialogs --test-msgbox
        python -m wbia.guitool.guitool_dialogs --test-msgbox --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia import guitool
        >>> guitool.ensure_qtapp()
        >>> from wbia.guitool.guitool_dialogs import *  # NOQA
        >>> from wbia.guitool.guitool_dialogs import _register_msgbox  # NOQA
        >>> # build test data
        >>> msg = 'Hello World!'
        >>> detailed_msg = 'I have a detailed message for you.'
        >>> title = 'msgbox'
        >>> # execute function
        >>> msgbox = msgbox(msg, title, detailed_msg=detailed_msg)
        >>> # verify results
        >>> result = str(msgbox)
        >>> print(result)
        >>> # xdoctest: +REQUIRES(--show)
        >>> ut.quit_if_noshow()
        >>> msgbox.exec_()
    """
    # msgbox = QtWidgets.QMessageBox(None)
    msgbox = ResizableMessageBox(None)
    msgbox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msgbox.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msgbox.setWindowTitle(title)
    # TODO: custom resizable msgbox
    msgbox.setText(msg)
    if detailed_msg is not None:
        msgbox.setDetailedText(detailed_msg)
    msgbox.setModal(False)
    msgbox.open(msgbox.close)
    msgbox.show()
    _register_msgbox(msgbox)
    return msgbox


def build_nested_qmenu(widget, context_options, name=None):
    """ builds nested menu for context menus but can be used for other menu
    related things.

    References:
        http://pyqt.sourceforge.net/Docs/PyQt4/qkeysequence.html
    """
    if name is None:
        menu = QtWidgets.QMenu(widget)
    else:
        menu = QtWidgets.QMenu(name, widget)
    action_list = []
    for option_tup in context_options:
        if len(option_tup) == 2:
            opt, func = option_tup
            shortcut = QtGui.QKeySequence(0)
        elif len(option_tup) == 3:
            opt, shortcut_str, func = option_tup
            shortcut = QtGui.QKeySequence(shortcut_str)

        if func is None:
            action = menu.addAction(opt, lambda: None, shortcut)
            action_list.append(action)
        elif isinstance(func, list):
            # Recursive case
            sub_menu, sub_action_list = build_nested_qmenu(widget, func, opt)
            menu.addMenu(sub_menu)
            action_list.append((sub_menu, sub_action_list))
        else:
            action = menu.addAction(opt, func, shortcut)
            action_list.append(action)
    return menu, action_list


def popup_menu(widget, pos, context_options):
    r"""
    For (right-click) context menus

    Args:
        widget (QWidget):
        pos (QPoint):
        context_options (list): of tuples. Can also replace func with
            nested context_options list. Tuples can be in the format:
             (name, func)
             (name, shortcut, func) - NOT FULLY SUPPORTED. USE AMPERSAND & INSTEAD

    Returns:
        tuple: (selection, actions)

    CommandLine:
        python -m wbia.guitool.guitool_dialogs --test-popup_menu
        python -m wbia.guitool.guitool_dialogs --test-popup_menu --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> # GUI_DOCTEST
        >>> # xdoctest: +REQUIRES(--gui)
        >>> from wbia.guitool.guitool_dialogs import *  # NOQA
        >>> import wbia.guitool
        >>> import wbia.plottool as pt
        >>> import functools
        >>> from wbia.plottool import interact_helpers as ih
        >>> fig = pt.figure()
        >>> def spam(x=''):
        ...    print('spam' + str(x))
        >>> def eggs():
        ...    print('eggs')
        >>> def bacon():
        ...    print('bacon')
        >>> def nospam():
        ...    print('i dont like spam')
        ...    import webbrowser
        ...    webbrowser.open('https://www.youtube.com/watch?v=anwy2MPT5RE')
        >>> context_options = [
        ...     ('spam',  spam),
        ...     ('&bacon', bacon),
        ...     ('n&est', [
        ...         ('e&ggs', eggs),
        ...         ('&s&pam', functools.partial(spam, 1)),
        ...         ('sp&a&m', functools.partial(spam, 2)),
        ...         ('&spam', functools.partial(spam, 3)),
        ...         ('&spamspamspam', [
        ...              ('&spamspamspam', [
        ...                  ('&spam', nospam),
        ...             ])
        ...         ])
        ...     ])
        ... ]
        >>> widget = fig.canvas
        >>> pos = guitool.newQPoint(10, 10)
        >>> # Hacky way to get a right click to span a context menu
        >>> def figure_clicked(event, fig=fig, context_options=context_options):
        ...     import wbia.guitool
        ...     import wbia.plottool as pt
        ...     from wbia.plottool import interact_helpers as ih
        ...     pos = guitool.newQPoint(event.x, fig.canvas.geometry().height() - event.y)
        ...     widget = fig.canvas
        ...     (selection, actions) = popup_menu(widget, pos, context_options)
        ...     #print(str((selection, actions)))
        >>> if ut.show_was_requested():
        ...     ih.connect_callback(fig, 'button_press_event', figure_clicked)
        ...     pt.show_if_requested()
        >>> else:
        ...    (selection, actions) = popup_menu(widget, pos, context_options)
    """
    # menu = QtWidgets.QMenu(widget)
    # actions = [menu.addAction(opt, ut.tracefunc(func)) for (opt, func) in context_options]
    menu, action_list = build_nested_qmenu(widget, context_options)
    # actions = [menu.addAction(opt, func) for (opt, func) in context_options]
    selection = menu.exec_(widget.mapToGlobal(pos))
    return selection, action_list
    # pos=QtGui.QCursor.pos()


def connect_context_menu(widget, context_options):
    def popup_slot(pos):
        return popup_menu(widget, pos, context_options)

    # Remove other context menus from this widget
    for _slot in _get_scope(widget, '_popup_scope'):
        widget.customContextMenuRequested.disconnect(_slot)
    _clear_scope(widget, '_popup_scope')
    # Enable custom context menus to be requested
    widget.setContextMenuPolicy(Qt.CustomContextMenu)
    # Connect our context slot
    widget.customContextMenuRequested.connect(popup_slot)
    # Make sure popup_slot does not lose scope.
    _enforce_scope(widget, popup_slot, '_popup_scope')
    return popup_slot


def _get_scope(qobj, scope_title='_scope_list'):
    """ Helper for ensuring qt objects are not garbage collected """
    if not hasattr(qobj, scope_title):
        setattr(qobj, scope_title, [])
    return getattr(qobj, scope_title)


def _clear_scope(qobj, scope_title='_scope_list'):
    """ Helper for ensuring qt objects are not garbage collected """
    setattr(qobj, scope_title, [])


def _enforce_scope(qobj, scoped_obj, scope_title='_scope_list'):
    """ Helper for ensuring qt objects are not garbage collected """
    _get_scope(qobj, scope_title).append(scoped_obj)


def _cacheReply(msgbox):
    dontPrompt = QtWidgets.QCheckBox('dont ask me again', parent=msgbox)
    dontPrompt.blockSignals(True)
    msgbox.addButton(dontPrompt, QtWidgets.QMessageBox.ActionRole)
    return dontPrompt


def _getQtImageNameFilter():
    imgNamePat = ' '.join(['*' + ext for ext in util_path.IMG_EXTENSIONS])
    imgNameFilter = 'Images (%s)' % (imgNamePat)
    return imgNameFilter


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.guitool.guitool_dialogs
        python -m wbia.guitool.guitool_dialogs --allexamples
        python -m wbia.guitool.guitool_dialogs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
