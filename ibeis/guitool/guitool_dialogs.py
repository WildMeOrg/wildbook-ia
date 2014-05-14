from __future__ import absolute_import, division, print_function
from PyQt4 import QtCore, QtGui  # NOQA
from PyQt4.QtCore import Qt
from os.path import split
# UTool
from utool import util_cache, util_path


SELDIR_CACHEID = 'guitool_selected_directory'


def _guitool_cache_write(key, val):
    """ Writes to global IBEIS cache """
    util_cache.global_cache_write(key, val, appname='ibeis')  # HACK, user should specify appname


def _guitool_cache_read(key, **kwargs):
    """ Reads from global IBEIS cache """
    return util_cache.global_cache_read(key, appname='ibeis', **kwargs)  # HACK, user should specify appname


def user_option(parent=None, msg='msg', title='user_option',
                options=['No', 'Yes'], use_cache=False):
    'Prompts user with several options with ability to save decision'
    print('[*guitools] user_option:\n %r: %s' + title + ': ' + msg)
    # Recall decision
    print('[*guitools] asking user: %r %r' % (msg, title))
    cache_id = title + msg
    if use_cache:
        reply = _guitool_cache_read(cache_id, default=None)
        if reply is not None:
            return reply
    # Create message box
    msgBox = _newMsgBox(msg, title, parent)
    _addOptions(msgBox, options)
    if use_cache:
        # Add a remember me option if caching is on
        dontPrompt = _cacheReply(msgBox)
    # Wait for output
    optx = msgBox.exec_()
    if optx == QtGui.QMessageBox.Cancel:
        # User Canceled
        return None
    try:
        # User Selected an option
        reply = options[optx]
    except KeyError as ex:
        # This should be unreachable code.
        print('[*guitools] USER OPTION EXCEPTION !')
        print('[*guitools] optx = %r' % optx)
        print('[*guitools] options = %r' % options)
        print('[*guitools] ex = %r' % ex)
        raise
    # Remember decision if caching is on
    if use_cache and dontPrompt.isChecked():
        _guitool_cache_write(cache_id, reply)
    # Close the message box
    del msgBox
    return reply


def user_input(parent=None, msg='msg', title='user_input'):
    reply, ok = QtGui.QInputDialog.getText(parent, title, msg)
    if not ok:
        return None
    return str(reply)


def user_info(parent=None, msg='msg', title='user_info'):
    print('[dlg.user_info] title=%r, msg=%r' % (title, msg))
    msgBox = _newMsgBox(msg, title, parent)
    msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
    msgBox.setModal(False)
    msgBox.open(msgBox.close)
    msgBox.show()


def user_question(msg):
    raise NotImplementedError('user_question')
    msgBox = QtGui.QMessageBox.question(None, '', 'lovely day?')
    return msgBox


def select_directory(caption='Select Directory', directory=None):
    print(caption)
    if directory is None:
        directory_ = _guitool_cache_read(SELDIR_CACHEID, default='.')
    else:
        directory = directory_
    qdlg = QtGui.QFileDialog()
    qopt = QtGui.QFileDialog.ShowDirsOnly
    qtkw = {
        'caption': caption,
        'options': qopt,
        'directory': directory_
    }
    dpath = str(qdlg.getExistingDirectory(**qtkw))
    if dpath == '' or dpath is None:
        dpath = None
        return dpath
    else:
        _guitool_cache_write(SELDIR_CACHEID, split(dpath)[0])
    print('Selected Directory: %r' % dpath)
    return dpath


def select_images(caption='Select images:', directory=None):
    name_filter = _getQtImageNameFilter()
    return select_files(caption, directory, name_filter)


def select_files(caption='Select Files:', directory=None, name_filter=None):
    'Selects one or more files from disk using a qt dialog'
    print(caption)
    if directory is None:
        directory = _guitool_cache_read(SELDIR_CACHEID, default='.')
    qdlg = QtGui.QFileDialog()
    qfile_list = qdlg.getOpenFileNames(caption=caption, directory=directory, filter=name_filter)
    file_list = map(str, qfile_list)
    print('Selected %d files' % len(file_list))
    _guitool_cache_write(SELDIR_CACHEID, directory)
    return file_list


def msgbox(msg, title='msgbox'):
    'Make a non modal critical QtGui.QMessageBox.'
    msgBox = QtGui.QMessageBox(None)
    msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
    msgBox.setWindowTitle(title)
    msgBox.setText(msg)
    msgBox.setModal(False)
    msgBox.open(msgBox.close)
    msgBox.show()
    return msgBox


def popup_menu(widget, opt2_callback, parent=None):
    def popup_slot(pos):
        print(pos)
        menu = QtGui.QMenu()
        actions = [menu.addAction(opt, func) for opt, func in
                   iter(opt2_callback)]
        #pos=QtGui.QCursor.pos()
        selection = menu.exec_(widget.mapToGlobal(pos))
        return selection, actions
    if parent is not None:
        # Make sure popup_slot does not lose scope.
        for _slot in _get_scope(parent, '_popup_scope'):
            parent.customContextMenuRequested.disconnect(_slot)
        _clear_scope(parent, '_popup_scope')
        parent.setContextMenuPolicy(Qt.CustomContextMenu)
        parent.customContextMenuRequested.connect(popup_slot)
        _enfore_scope(parent, popup_slot, '_popup_scope')
    return popup_slot


def _get_scope(qobj, scope_title='_scope_list'):
    if not hasattr(qobj, scope_title):
        setattr(qobj, scope_title, [])
    return getattr(qobj, scope_title)


def _clear_scope(qobj, scope_title='_scope_list'):
    setattr(qobj, scope_title, [])


def _enfore_scope(qobj, scoped_obj, scope_title='_scope_list'):
    _get_scope(qobj, scope_title).append(scoped_obj)


def _addOptions(msgBox, options):
    #msgBox.addButton(QtGui.QMessageBox.Close)
    for opt in options:
        role = QtGui.QMessageBox.ApplyRole
        msgBox.addButton(QtGui.QPushButton(opt), role)


def _cacheReply(msgBox):
    dontPrompt = QtGui.QCheckBox('dont ask me again', parent=msgBox)
    dontPrompt.blockSignals(True)
    msgBox.addButton(dontPrompt, QtGui.QMessageBox.ActionRole)
    return dontPrompt


def _newMsgBox(msg='', title='', parent=None, options=None, cache_reply=False):
    msgBox = QtGui.QMessageBox(parent)
    #msgBox.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    #std_buts = QtGui.QMessageBox.Close
    #std_buts = QtGui.QMessageBox.NoButton
    std_buts = QtGui.QMessageBox.Cancel
    msgBox.setStandardButtons(std_buts)
    msgBox.setWindowTitle(title)
    msgBox.setText(msg)
    msgBox.setModal(parent is not None)
    return msgBox


def _getQtImageNameFilter():
    imgNamePat = ' '.join(['*' + ext for ext in util_path.IMG_EXTENSIONS])
    imgNameFilter = 'Images (%s)' % (imgNamePat)
    return imgNameFilter
