# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import six
from six.moves import map, range
from guitool.__PYQT__ import QtCore, QtGui
from guitool.__PYQT__.QtGui import QSizePolicy
from guitool.__PYQT__.QtCore import Qt
import functools
import utool
import utool as ut  # NOQA
from guitool import guitool_dialogs
import weakref
(print, rrr, profile) = utool.inject2(__name__, '[guitool_components]')


ALIGN_DICT = {
    'center': Qt.AlignCenter,
    'right': Qt.AlignRight | Qt.AlignVCenter,
    'left': Qt.AlignLeft | Qt.AlignVCenter,
    'justify': Qt.AlignJustify,
}


def newSizePolicy(widget,
                  verticalSizePolicy=QSizePolicy.Expanding,
                  horizontalSizePolicy=QSizePolicy.Expanding,
                  horizontalStretch=0,
                  verticalStretch=0):
    """
    input: widget - the central widget
    """
    sizePolicy = QSizePolicy(horizontalSizePolicy, verticalSizePolicy)
    sizePolicy.setHorizontalStretch(horizontalStretch)
    sizePolicy.setVerticalStretch(verticalStretch)
    #sizePolicy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    return sizePolicy


def newSplitter(widget, orientation=Qt.Horizontal, verticalStretch=1):
    """
    input: widget - the central widget
    """
    hsplitter = QtGui.QSplitter(orientation, widget)
    # This line makes the hsplitter resize with the widget
    sizePolicy = newSizePolicy(hsplitter, verticalStretch=verticalStretch)
    hsplitter.setSizePolicy(sizePolicy)
    setattr(hsplitter, '_guitool_sizepolicy', sizePolicy)
    return hsplitter


def newTabWidget(parent, horizontalStretch=1):
    tabwgt = QtGui.QTabWidget(parent)
    sizePolicy = newSizePolicy(tabwgt, horizontalStretch=horizontalStretch)
    tabwgt.setSizePolicy(sizePolicy)
    setattr(tabwgt, '_guitool_sizepolicy', sizePolicy)
    return tabwgt


def newMenubar(widget):
    """ Defines the menubar on top of the main widget """
    menubar = QtGui.QMenuBar(widget)
    menubar.setGeometry(QtCore.QRect(0, 0, 1013, 23))
    menubar.setContextMenuPolicy(Qt.DefaultContextMenu)
    menubar.setDefaultUp(False)
    menubar.setNativeMenuBar(False)
    widget.setMenuBar(menubar)
    return menubar


def newQPoint(x, y):
    return QtCore.QPoint(int(round(x)), int(round(y)))


def newMenu(widget, menubar, name, text):
    """ Defines each menu category in the menubar """
    menu = QtGui.QMenu(menubar)
    menu.setObjectName(name)
    menu.setTitle(text)
    # Define a custom newAction function for the menu
    # The QT function is called addAction
    newAction = functools.partial(newMenuAction, widget, name)
    setattr(menu, 'newAction', newAction)
    # Add the menu to the menubar
    menubar.addAction(menu.menuAction())
    return menu


def newMenuAction(front, menu_name, name=None, text=None, shortcut=None,
                  tooltip=None, slot_fn=None, enabled=True):
    """
    Added as a helper function to menus
    """
    if name is None:
        # it is usually better to specify the name explicitly
        name = ut.convert_text_to_varname('action' + text)
    assert name is not None, 'menuAction name cannot be None'
    # Dynamically add new menu actions programatically
    action_name = name
    action_text = text
    action_shortcut = shortcut
    action_tooltip  = tooltip
    if hasattr(front, action_name):
        raise Exception('menu action already defined')
    # Create new action
    action = QtGui.QAction(front)
    setattr(front, action_name, action)
    action.setEnabled(enabled)
    action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    menu = getattr(front, menu_name)
    menu.addAction(action)
    if action_text is None:
        action_text = action_name
    if action_text is not None:
        action.setText(action_text)
    if action_tooltip is not None:
        action.setToolTip(action_tooltip)
    if action_shortcut is not None:
        action.setShortcut(action_shortcut)
        #print('<%s>.setShortcut(%r)' % (action_name, action_shortcut,))
    if slot_fn is not None:
        action.triggered.connect(slot_fn)
    return action


SHOW_TEXT = ut.get_argflag('--progtext')


class ProgressHooks(QtCore.QObject):
    """
    hooks into utool.ProgressIterator

    TODO:
        use signals and slots to connect to the progress bar
        still doesn't work correctly even with signals and slots, probably
          need to do task function in another thread

    References:
        http://stackoverflow.com/questions/19442443/busy-indication-with-pyqt-progress-bar
    """
    set_progress_signal = QtCore.pyqtSignal(int, int)
    show_indefinite_progress_signal = QtCore.pyqtSignal()

    def __init__(proghook, progressBar, substep_min=0, substep_size=1, level=0):
        super(ProgressHooks, proghook).__init__()
        proghook.progressBarRef = weakref.ref(progressBar)
        proghook.substep_min = substep_min
        proghook.substep_size = substep_size
        proghook.count = 0
        proghook.nTotal = None
        proghook.progiter = None
        proghook.lbl = ''
        proghook.level = level
        proghook.child_hook_gen = None
        proghook.set_progress_signal.connect(proghook.set_progress_slot)
        proghook.show_indefinite_progress_signal.connect(proghook.show_indefinite_progress_slot)

    def initialize_subhooks(proghook, num_child_subhooks):
        proghook.child_hook_gen = iter(proghook.make_substep_hooks(num_child_subhooks))

    def next_subhook(proghook):
        return six.next(proghook.child_hook_gen)

    def register_progiter(proghook, progiter):
        proghook.progiter = weakref.ref(progiter)
        proghook.nTotal = proghook.progiter().nTotal
        proghook.lbl = proghook.progiter().lbl

    def make_substep_hooks(proghook, num_substeps):
        """ make hooks that take up a fraction of this hooks step size.
            substep sizes are all fractional
        """
        step_min = ((proghook.progiter().count - 1) / proghook.nTotal) * proghook.substep_size  + proghook.substep_min
        step_size = (1.0 / proghook.nTotal) * proghook.substep_size

        substep_size = step_size / num_substeps
        substep_min_list = [(step * substep_size) + step_min for step in range(num_substeps)]

        DEBUG = False
        if DEBUG:
            with ut.Indenter(' ' * 4 * proghook.level):
                print('\n')
                print('+____<NEW SUBSTEPS>____')
                print('Making %d substeps for proghook.lbl = %s' % (num_substeps, proghook.lbl,))
                print(' * step_min         = %.2f' % (step_min,))
                print(' * step_size        = %.2f' % (step_size,))
                print(' * substep_size     = %.2f' % (substep_size,))
                print(' * substep_min_list = %r' % (substep_min_list,))
                print(r'L____</NEW SUBSTEPS>____')
                print('\n')

        subhook_list = [ProgressHooks(proghook.progressBarRef(), substep_min, substep_size, proghook.level + 1)
                        for substep_min in substep_min_list]
        return subhook_list

    @QtCore.pyqtSlot()
    def show_indefinite_progress_slot(proghook):
        progbar = proghook.progressBarRef()
        progbar.reset()
        progbar.setMaximum(0)
        progbar.setProperty('value', 0)
        proghook.force_event_update()

    def show_indefinite_progress(proghook):
        proghook.show_indefinite_progress_signal.emit()

    def force_event_update(proghook):
        # major hack
        import guitool
        qtapp = guitool.get_qtapp()
        qtapp.processEvents()

    def set_progress(proghook, count, nTotal=None):
        if nTotal is None:
            nTotal = proghook.nTotal
        else:
            proghook.nTotal = nTotal
        if nTotal is None:
            nTotal = 100
        proghook.set_progress_signal.emit(count, nTotal)

    @QtCore.pyqtSlot(int, int)
    def set_progress_slot(proghook, count, nTotal=None):
        if nTotal is None:
            nTotal = proghook.nTotal
        else:
            proghook.nTotal = nTotal
        proghook.count = count
        local_fraction = (count) / nTotal
        global_fraction = (local_fraction * proghook.substep_size) + proghook.substep_min
        DEBUG = False

        if DEBUG:
            with ut.Indenter(' ' * 4 * proghook.level):
                print('\n')
                print('+-----------')
                print('proghook.substep_min = %.3f' % (proghook.substep_min,))
                print('proghook.lbl = %r' % (proghook.lbl,))
                print('proghook.substep_size = %.3f' % (proghook.substep_size,))
                print('global_fraction = %.3f' % (global_fraction,))
                print('local_fraction = %.3f' % (local_fraction,))
                print('L___________')
        if SHOW_TEXT:
            resolution = 75
            num_full = int(round(global_fraction * resolution))
            num_empty = resolution - num_full
            print('\n')
            print('[' + '#' * num_full + '.' * num_empty + '] %7.3f%%' % (global_fraction * 100))
            print('\n')
        #assert local_fraction <= 1.0
        #assert global_fraction <= 1.0
        progbar = proghook.progressBarRef()
        progbar.setRange(0, 10000)
        progbar.setMinimum(0)
        progbar.setMaximum(10000)
        value = int(round(progbar.maximum() * global_fraction))
        progbar.setFormat(proghook.lbl + ' %p%')
        progbar.setValue(value)
        #progbar.setProperty('value', value)
        # major hack
        proghook.force_event_update()
        #import guitool
        #qtapp = guitool.get_qtapp()
        #qtapp.processEvents()

    def __call__(proghook, count, nTotal=None):
        proghook.set_progress(count, nTotal)
        #proghook.set_progress_slot(count, nTotal)


def newProgressBar(parent, visible=True, verticalStretch=1):
    r"""
    Args:
        parent (?):
        visible (bool):
        verticalStretch (int):

    Returns:
        QProgressBar: progressBar

    CommandLine:
        python -m guitool.guitool_components --test-newProgressBar:0
        python -m guitool.guitool_components --test-newProgressBar:0 --show
        python -m guitool.guitool_components --test-newProgressBar:1 --progtext

    Example:
        >>> # GUI_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> # build test data
        >>> import guitool
        >>> guitool.ensure_qtapp()
        >>> parent = None
        >>> visible = True
        >>> verticalStretch = 1
        >>> # hook into utool progress iter
        >>> progressBar = newProgressBar(parent, visible, verticalStretch)
        >>> progressBar.show()
        >>> progressBar.utool_prog_hook.show_indefinite_progress()
        >>> #progressBar.utool_prog_hook.set_progress(0)
        >>> #import time
        >>> qtapp = guitool.get_qtapp()
        >>> [(qtapp.processEvents(), ut.get_nth_prime_bruteforce(300)) for x in range(100)]
        >>> #time.sleep(5)
        >>> progiter = ut.ProgressIter(range(100), freq=1, autoadjust=False, prog_hook=progressBar.utool_prog_hook)
        >>> results1 = [ut.get_nth_prime_bruteforce(300) for x in progiter]
        >>> # verify results
        >>> ut.quit_if_noshow()
        >>> guitool.qtapp_loop(freq=10)

    Example:
        >>> # GUI_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> # build test data
        >>> import guitool
        >>> guitool.ensure_qtapp()
        >>> parent = None
        >>> visible = True
        >>> verticalStretch = 1
        >>> def complex_tasks(proghook):
        ...     progkw = dict(freq=1, backspace=False, autoadjust=False)
        ...     num = 800
        ...     for x in ut.ProgressIter(range(4), lbl='TASK', prog_hook=proghook, **progkw):
        ...         ut.get_nth_prime_bruteforce(num)
        ...         subhooks = proghook.make_substep_hooks(2)
        ...         for task1 in ut.ProgressIter(range(2), lbl='task1.1', prog_hook=subhooks[0], **progkw):
        ...             ut.get_nth_prime_bruteforce(num)
        ...             subsubhooks = subhooks[0].make_substep_hooks(3)
        ...             for task1 in ut.ProgressIter(range(7), lbl='task1.1.1', prog_hook=subsubhooks[0], **progkw):
        ...                 ut.get_nth_prime_bruteforce(num)
        ...             for task1 in ut.ProgressIter(range(11), lbl='task1.1.2', prog_hook=subsubhooks[1], **progkw):
        ...                 ut.get_nth_prime_bruteforce(num)
        ...             for task1 in ut.ProgressIter(range(3), lbl='task1.1.3', prog_hook=subsubhooks[2], **progkw):
        ...                 ut.get_nth_prime_bruteforce(num)
        ...         for task2 in ut.ProgressIter(range(10), lbl='task1.2', prog_hook=subhooks[1], **progkw):
        ...             ut.get_nth_prime_bruteforce(num)
        >>> # hook into utool progress iter
        >>> progressBar = newProgressBar(parent, visible, verticalStretch)
        >>> complex_tasks(progressBar.utool_prog_hook)
        >>> # verify results
        >>> ut.quit_if_noshow()
        >>> guitool.qtapp_loop(freq=10)


    Ignore:
        from guitool.guitool_components import *  # NOQA
        # build test data
        import guitool
        guitool.ensure_qtapp()

    """
    progressBar = QtGui.QProgressBar(parent)
    sizePolicy = newSizePolicy(progressBar,
                               verticalSizePolicy=QSizePolicy.Maximum,
                               verticalStretch=verticalStretch)
    progressBar.setSizePolicy(sizePolicy)
    progressBar.setMaximum(10000)
    progressBar.setProperty('value', 0)
    #def utool_prog_hook(count, nTotal):
    #    progressBar.setProperty('value', int(100 * count / nTotal))
    #    # major hack
    #    import guitool
    #    qtapp = guitool.get_qtapp()
    #    qtapp.processEvents()
    #    pass
    progressBar.utool_prog_hook = ProgressHooks(progressBar)
    #progressBar.setTextVisible(False)
    progressBar.setTextVisible(True)
    progressBar.setFormat('%p%')
    progressBar.setVisible(visible)
    progressBar.setMinimumWidth(600)
    setattr(progressBar, '_guitool_sizepolicy', sizePolicy)
    if visible:
        # hack to make progres bar show up immediately
        import guitool
        progressBar.show()
        qtapp = guitool.get_qtapp()
        qtapp.processEvents()
    return progressBar


def newOutputLog(parent, pointSize=6, visible=True, verticalStretch=1):
    from guitool.guitool_misc import QLoggedOutput
    outputLog = QLoggedOutput(parent)
    sizePolicy = newSizePolicy(outputLog,
                               #verticalSizePolicy=QSizePolicy.Preferred,
                               verticalStretch=verticalStretch)
    outputLog.setSizePolicy(sizePolicy)
    outputLog.setAcceptRichText(False)
    outputLog.setVisible(visible)
    #outputLog.setFontPointSize(8)
    outputLog.setFont(newFont('Courier New', pointSize))
    setattr(outputLog, '_guitool_sizepolicy', sizePolicy)
    return outputLog


def newTextEdit(parent, label=None, visible=True, label_pos='above'):
    """ This is a text area """
    outputEdit = QtGui.QTextEdit(parent)
    sizePolicy = newSizePolicy(outputEdit, verticalStretch=1)
    outputEdit.setSizePolicy(sizePolicy)
    outputEdit.setAcceptRichText(False)
    outputEdit.setVisible(visible)
    if label is None:
        pass
    setattr(outputEdit, '_guitool_sizepolicy', sizePolicy)
    return outputEdit


def newLineEdit(parent, text=None, enabled=True, align='center',
                textChangedSlot=None, textEditedSlot=None,
                editingFinishedSlot=None, visible=True, readOnly=False,
                verticalStretch=0, fontkw={}):
    """ This is a text line

    Example:
        >>> # DISABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> parent = None
        >>> text = None
        >>> visible = True
        >>> # execute function
        >>> widget = newLineEdit(parent, text, visible)
        >>> # verify results
        >>> result = str(widget)
        >>> print(result)
    """
    widget = QtGui.QLineEdit(parent)
    sizePolicy = newSizePolicy(widget,
                               verticalSizePolicy=QSizePolicy.Fixed,
                               verticalStretch=verticalStretch)
    widget.setSizePolicy(sizePolicy)
    if text is not None:
        widget.setText(text)
    widget.setEnabled(enabled)
    widget.setAlignment(ALIGN_DICT[align])
    widget.setReadOnly(readOnly)
    if textChangedSlot is not None:
        widget.textChanged.connect(textChangedSlot)
    if editingFinishedSlot is not None:
        widget.editingFinished.connect(editingFinishedSlot)
    if textEditedSlot is not None:
        widget.textEdited.connect(textEditedSlot)

    #outputEdit.setAcceptRichText(False)
    #outputEdit.setVisible(visible)
    adjust_font(widget, **fontkw)
    setattr(widget, '_guitool_sizepolicy', sizePolicy)
    return widget


def newWidget(parent=None, orientation=Qt.Vertical,
              verticalSizePolicy=QSizePolicy.Expanding,
              horizontalSizePolicy=QSizePolicy.Expanding,
              verticalStretch=1, special_layout=None):
    r"""
    Args:
        parent (QWidget):
        orientation (Orientation): (default = 2)
        verticalSizePolicy (Policy): (default = 7)
        horizontalSizePolicy (Policy): (default = 7)
        verticalStretch (int): (default = 1)

    Returns:
        QWidget: widget
    """
    #widget = QtGui.QWidget(parent)
    #if special_layout is None:
    widget = GuitoolWidget(parent, orientation, verticalSizePolicy,
                           horizontalSizePolicy, verticalStretch)
    #sizePolicy = newSizePolicy(widget,
    #                           horizontalSizePolicy=horizontalSizePolicy,
    #                           verticalSizePolicy=verticalSizePolicy,
    #                           verticalStretch=verticalStretch)
    #widget.setSizePolicy(sizePolicy)
    #if orientation == Qt.Vertical:
    #    layout = QtGui.QVBoxLayout(widget)
    #elif orientation == Qt.Horizontal:
    #    layout = QtGui.QHBoxLayout(widget)
    #else:
    #    raise NotImplementedError('orientation')
    ## Black magic
    #widget._guitool_layout = layout
    #widget.addWidget = widget._guitool_layout.addWidget
    #widget.addLayout = widget._guitool_layout.addLayout
    #setattr(widget, '_guitool_sizepolicy', sizePolicy)
    #elif special_layout == 'form':
    #    import utool
    #    utool.embed()
    #    layout = QtGui.QFormLayout(widget)
    #    widget.addItem = layout.addItem
    #    widget.addRow = layout.addRow
    #    widget.addWidget = layout.addWidget
    #    widget.addChildWidget = layout.addChildWidget
    return widget


class GuitoolWidget(QtGui.QWidget):
    """
    CommandLine:
        python -m guitool.guitool_components GuitoolWidget --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> import guitool
        >>> import guitool as gt
        >>> guitool.ensure_qtapp()
        >>> ut.exec_funckw(newWidget, globals())
        >>> widget = GuitoolWidget(parent)
        >>> widget.addWidget(gt.newButton(
        >>>     widget, 'Print Hi', lambda: print('hi')))
        >>> widget.addWidget(gt.newButton(
        >>>     widget, 'Popup Hi', lambda: gt.user_info(widget, 'hi')))
        >>> #widget.addRow('input 1', gt.newLineEdit(widget))
        >>> #widget.addRow('input 2', gt.newComboBox(widget, ['one', 'two']))
        >>> widget.show()
        >>> widget.resize(int(ut.PHI * 500), 500)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=widget, freq=10)
    """
    closed = QtCore.pyqtSignal()

    def _inject_new_widget_methods(self):
        import guitool as gt
        # Black magic
        guitype_list = ['Widget', 'Button', 'LineEdit', 'ComboBox', 'Label', 'Spoiler']
        # Creates addNewWidget and newWidget
        for guitype in guitype_list:
            if hasattr(gt, 'new' + guitype):
                newfunc = getattr(gt, 'new' + guitype)
                ut.inject_func_as_method(self, newfunc, 'new' + guitype)
            else:
                newfunc = getattr(gt, guitype)
            addnew_func = self._addnew_factory(newfunc)
            ut.inject_func_as_method(self, addnew_func, 'addNew' + guitype)
        # Above code is the same as saying
        #     self.newButton = ut.partial(newButton, self)
        #     self.newWidget = ut.partial(newWidget, self)
        #     ... etc

    def __init__(self, parent=None, orientation=Qt.Vertical,
                 verticalSizePolicy=QSizePolicy.Expanding,
                 horizontalSizePolicy=QSizePolicy.Expanding,
                 verticalStretch=0, **kwargs):
        super(GuitoolWidget, self).__init__(parent)

        sizePolicy = newSizePolicy(self,
                                   horizontalSizePolicy=horizontalSizePolicy,
                                   verticalSizePolicy=verticalSizePolicy,
                                   verticalStretch=verticalStretch)
        self.setSizePolicy(sizePolicy)
        if orientation == Qt.Vertical:
            layout = QtGui.QVBoxLayout(self)
        elif orientation == Qt.Horizontal:
            layout = QtGui.QHBoxLayout(self)
        else:
            raise NotImplementedError('orientation')
        self._guitool_layout = layout
        #self.addWidget = self._guitool_layout.addWidget
        #self.addLayout = self._guitool_layout.addLayout
        setattr(self, '_guitool_sizepolicy', sizePolicy)
        self._inject_new_widget_methods()
        self.initialize(**kwargs)

        if False:
            # debug code
            self.setStyleSheet("background-color: rgb(255,0,0); margin:5px; border:1px solid rgb(0, 255, 0); ")
            self.setStyleSheet("background-color: border:5px solid rgb(255, 0, 0); ")

    @classmethod
    def as_dialog(cls, parent=None, **kwargs):
        widget = cls(**kwargs)
        dlg = QtGui.QDialog(parent)
        dlg.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        dlg.widget = widget
        dlg.vlayout = QtGui.QVBoxLayout(dlg)
        dlg.vlayout.addWidget(widget)
        widget.closed.connect(dlg.close)
        dlg.setWindowTitle(widget.windowTitle())
        return dlg

    def initialize(self, **kwargs):
        pass

    def addLayout(self, *args, **kwargs):
        return self._guitool_layout.addLayout(*args, **kwargs)

    def addWidget(self, widget, *args, **kwargs):
        self._guitool_layout.addWidget(widget, *args, **kwargs)
        return widget

    def newHWidget(self, **kwargs):
        return self.addNewWidget(orientation=Qt.Horizontal, **kwargs)

    #def addNewWidget(self, *args, **kwargs):
    #    new_widget = self.newWidget(*args, **kwargs)
    #    return self.addWidget(new_widget)

    def _addnew_factory(self, newfunc):
        def _addnew(self, *args, **kwargs):
            new_widget = newfunc(self, *args, **kwargs)
            return self.addWidget(new_widget)
        return _addnew

    def closeEvent(self, event):
        event.accept()
        self.closed.emit()


class ConfigConfirmWidget(GuitoolWidget):
    """

    CommandLine:
        python -m guitool.guitool_components ConfigConfirmWidget --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> import guitool
        >>> import dtool
        >>> guitool.ensure_qapp()  # must be ensured before any embeding
        >>> cls = dtool.Config
        >>> dict_ = {'K': 1, 'Knorm': 5, 'min_pername': 1, 'max_pername': 1,}
        >>> tablename = None
        >>> config = cls.from_dict(dict_, tablename)
        >>> dlg = guitool.ConfigConfirmWidget.as_dialog(
        >>>     title='Confirm Merge Query',
        >>>     msg='Confirm',
        >>>     config=config)
        >>> #dlg.resize(700, 500)
        >>> self = dlg.widget
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> dlg.show()
        >>> guitool.qtapp_loop(qwin=dlg)
        >>> updated_config = self.config  # NOQA
        >>> print('updated_config = %r' % (updated_config,))
    """
    def __init__(self, *args, **kwargs):
        # FIXME: http://doc.qt.io/qt-5/qsizepolicy.html
        kwargs['horizontalSizePolicy'] = QtGui.QSizePolicy.Minimum
        kwargs['verticalSizePolicy'] = QSizePolicy.Expanding
        super(ConfigConfirmWidget, self).__init__(*args, **kwargs)

    def initialize(self, title, msg, config, options=None, default=None):
        import copy
        from guitool import PrefWidget2
        self.msg = msg
        self.orig_config = config
        self.config = copy.deepcopy(config)
        self.confirm_option = None

        self.setWindowTitle(title)
        self.addNewLabel(msg, align='left')

        if config is not None:
            self.editConfig = PrefWidget2.newConfigWidget(self.config, user_mode=True)
            if not ut.get_argflag('--nospoiler'):
                self.spoiler = Spoiler(self, title='advanced configuration')
                self.spoiler.setContentLayout(self.editConfig.layout())
                self.spoiler.toggle_finished.connect(self._size_adjust_slot)
                self.addWidget(self.spoiler)
            else:
                self.addWidget(self.editConfig)

        self.button_row = self.newHWidget()
        if options is None:
            options = ['Confirm']
        def _make_option_clicked(opt):
            def _wrap():
                return self.confirm(opt)
            return _wrap
        for opt in options:
            self.button_row.addNewButton(opt, clicked=_make_option_clicked(opt))
        self.button_row.addNewButton('Cancel', clicked=self.cancel)

        self.layout().setAlignment(Qt.AlignBottom)
        #self.layout().setSizeConstraint(QtGui.QLayout.SetFixedSize)
        #self.resize(668, 530)
        #self.update_state()

    def update_state(self, *args):
        print('*args = %r' % (args,))
        print('Update state')
        if self.param_info_dict is None:
            print('Need dtool config')

        for key, pi in self.param_info_dict.items():
            row = self.row_dict[key]
            if pi.type_ is bool:
                value = row.edit.currentValue()
                print('Changed: key, value = %r, %r' % (key, value))
                self.config[key] = value

        for key, pi in self.param_info_dict.items():
            row = self.row_dict[key]
            flag = not pi.is_hidden(self.config)
            row.edit.setEnabled(flag)

    def confirm(self, confirm_option=None):
        print('[gt] Confirmed config')
        print('confirm_option = %r' % (confirm_option,))
        self.confirm_option = confirm_option
        self.close()

    def _size_adjust_slot(self):
        print('_size_adjust_slot = ')
        self.adjustSize()
        parent = self.parent()
        if parent is not None:
            parent.adjustSize()

    def cancel(self):
        print('[gt] Canceled confirm config')
        self.close()


def newButton(parent=None, text='', clicked=None, qicon=None, visible=True,
              enabled=True, bgcolor=None, fgcolor=None, fontkw={},
              shrink_to_text=False):
    """ wrapper around QtGui.QPushButton

    Args:
        parent (QWidget): parent widget
        text (str):
        clicked (func): callback function
        qicon (None):
        visible (bool):
        enabled (bool):
        bgcolor (None):
        fgcolor (None):
        fontkw (dict): (default = {})

    Kwargs:
        parent, text, clicked, qicon, visible, enabled, bgcolor, fgcolor,
        fontkw

    connectable signals:
        void clicked(bool checked=false)
        void pressed()
        void released()
        void toggled(bool checked)

    Returns:
       QtGui.QPushButton

    CommandLine:
        python -m guitool.guitool_components --exec-newButton
        python -m guitool.guitool_components --test-newButton

    Example:
        >>> # ENABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> import guitool
        >>> guitool.ensure_qtapp()
        >>> parent = None
        >>> text = ''
        >>> clicked = None
        >>> qicon = None
        >>> visible = True
        >>> enabled = True
        >>> bgcolor = None
        >>> fgcolor = None
        >>> fontkw = {}
        >>> button = newButton(parent, text, clicked, qicon, visible, enabled,
        >>>                    bgcolor, fgcolor, fontkw)
        >>> result = ('button = %s' % (str(button),))
        >>> print(result)
    """
    but_args = [text]
    but_kwargs = {
        'parent': parent
    }
    if clicked is not None:
        but_kwargs['clicked'] = clicked
    else:
        enabled = False
    if qicon is not None:
        but_args = [qicon] + but_args
    button = QtGui.QPushButton(*but_args, **but_kwargs)
    style_sheet_str = make_style_sheet(bgcolor=bgcolor, fgcolor=fgcolor)
    if style_sheet_str is not None:
        button.setStyleSheet(style_sheet_str)

    button.setVisible(visible)
    button.setEnabled(enabled)
    adjust_font(button, **fontkw)
    #sizePolicy = newSizePolicy(button,
    #                           #verticalSizePolicy=QSizePolicy.Fixed,
    #                           #horizontalSizePolicy=QSizePolicy.Fixed,
    #                           verticalStretch=0)
    #button.setSizePolicy(sizePolicy)
    if shrink_to_text:
        width = get_widget_text_width(button) + 10
        button.setMaximumWidth(width)
    return button


def get_widget_text_width(widget):
    # http://stackoverflow.com/questions/14418375/shrink-a-button-width
    text = widget.text()
    double = text.count('&&')
    text = text.replace('&', '') + ('&' * double)
    text_width = widget.fontMetrics().boundingRect(text).width()
    return text_width


def newComboBox(parent=None, options=None, changed=None, default=None, visible=True,
                enabled=True, bgcolor=None, fgcolor=None, fontkw={}):
    """ wrapper around QtGui.QComboBox

    Args:
        parent (None):
        options (list): a list of tuples, which are a of the following form:
            [
                (visible text 1, backend value 1),
                (visible text 2, backend value 2),
                (visible text 3, backend value 3),
            ]
        changed (None):
        default (str): backend value of default item
        visible (bool):
        enabled (bool):
        bgcolor (None):
        fgcolor (None):
        bold (bool):

    Returns:
        QtGui.QComboBox: combo

    CommandLine:
        python -m guitool.guitool_components --test-newComboBox --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> import guitool
        >>> guitool.ensure_qtapp()
        >>> exec(ut.execstr_funckw(newComboBox), globals())
        >>> parent = None
        >>> options = ['red', 'blue']
        >>> # execute function
        >>> combo = newComboBox(parent, options)
        >>> # verify results
        >>> result = str(combo)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> combo.show()
        >>> guitool.qtapp_loop(qwin=combo, freq=10)
    """

    # Check for tuple option formating
    flags = [isinstance(opt, tuple) and len(opt) == 2 for opt in options]
    options_ = [opt if flag else (str(opt), opt)
                for flag, opt in zip(flags, options)]

    class CustomComboBox(QtGui.QComboBox):
        def __init__(combo, parent=None, default=None, options_=None, changed=None):
            QtGui.QComboBox.__init__(combo, parent)
            combo.ibswgt = parent
            combo.options_ = options_
            combo.changed = changed
            #combo.allow_add = allow_add  # TODO
            # combo.setEditable(True)
            combo.updateOptions()
            combo.setDefault(default)
            combo.currentIndexChanged['int'].connect(combo.currentIndexChangedCustom)

        def currentValue(combo):
            index = combo.currentIndex()
            opt = combo.options_[index]
            value = opt[1]
            return value

        def setOptions(combo, options):
            flags = [isinstance(opt, tuple) and len(opt) == 2 for opt in options]
            options_ = [opt if flag else (str(opt), opt)
                        for flag, opt in zip(flags, options)]
            combo.options_ = options_

        def updateOptions(combo, reselect=False, reselect_index=None):
            if reselect_index is None:
                reselect_index = combo.currentIndex()
            combo.clear()
            combo.addItems( [ option[0] for option in combo.options_ ] )
            if reselect and reselect_index < len(combo.options_):
                combo.setCurrentIndex(reselect_index)

        def setOptionText(combo, option_text_list):
            for index, text in enumerate(option_text_list):
                combo.setItemText(index, text)
            #combo.removeItem()

        def currentIndexChangedCustom(combo, index):
            if combo.changed is not None:
                combo.changed(index, combo.options_[index][1])

        def setDefault(combo, default=None):
            if default is not None:
                combo.setCurrentValue(default)
            else:
                combo.setCurrentIndex(0)

        def setCurrentValue(combo, value):
            index = combo.findValueIndex(value)
            combo.setCurrentIndex(index)

        def findValueIndex(combo, value):
            """ finds index of backend value and sets the current index """
            for index, (text, val) in enumerate(combo.options_):
                if value == val:
                    return index
            else:
                # Hack, try the text if value doesnt work
                for index, (text, val) in enumerate(combo.options_):
                    if value == text:
                        return index
                else:
                    raise ValueError('No such option value=%r' % (value,))

    combo_kwargs = {
        'parent' : parent,
        'options_': options_,
        'default': default,
        'changed': changed,
    }
    combo = CustomComboBox(**combo_kwargs)
    #if changed is None:
    #    enabled = False
    combo.setVisible(visible)
    combo.setEnabled(enabled)
    adjust_font(combo, **fontkw)
    return combo


def newCheckBox(parent=None, text='', changed=None, checked=False, visible=True,
                enabled=True, bgcolor=None, fgcolor=None):
    """ wrapper around QtGui.QCheckBox
    """
    class CustomCheckBox(QtGui.QCheckBox):
        def __init__(check, text='', parent=None, checked=False, changed=None):
            QtGui.QComboBox.__init__(check, text, parent=parent)
            check.ibswgt = parent
            check.changed = changed
            if checked:
                check.setCheckState(2)  # 2 is equivelant to checked, 1 to partial, 0 to not checked
            check.stateChanged.connect(check.stateChangedCustom)

        def stateChangedCustom(check, state):
            check.changed(state == 2)

    check_kwargs = {
        'text'   : text,
        'checked': checked,
        'parent' : parent,
        'changed': changed,
    }
    check = CustomCheckBox(**check_kwargs)
    if changed is None:
        enabled = False
    check.setVisible(visible)
    check.setEnabled(enabled)
    return check


def newFont(fontname='Courier New', pointSize=-1, weight=-1, italic=False):
    """ wrapper around QtGui.QFont """
    #fontname = 'Courier New'
    #pointSize = 8
    #weight = -1
    #italic = False
    font = QtGui.QFont(fontname, pointSize=pointSize, weight=weight, italic=italic)
    return font


def adjust_font(widget, bold=False, pointSize=None, italic=False):
    if bold or pointSize is not None:
        font = widget.font()
        font.setBold(bold)
        font.setItalic(italic)
        if pointSize is not None:
            font.setPointSize(pointSize)
        widget.setFont(font)


def make_style_sheet(bgcolor=None, fgcolor=None):
    style_list = []
    fmtdict = {}
    if bgcolor is not None:
        style_list.append('background-color: rgb({bgcolor})')
        fmtdict['bgcolor'] = ','.join(map(str, bgcolor))
    if fgcolor is not None:
        style_list.append('color: rgb({fgcolor})')
        fmtdict['fgcolor'] = ','.join(map(str, fgcolor))
    if len(style_list) > 0:
        style_sheet_fmt = ';'.join(style_list)
        style_sheet_str = style_sheet_fmt.format(**fmtdict)
        return style_sheet_str
    else:
        return None

#def make_qstyle():
#    style_factory = QtGui.QStyleFactory()
#    style = style_factory.create('cleanlooks')
#    #app_style = QtGui.Q Application.style()


def newLabel(parent=None, text='', align='center', gpath=None, fontkw={}):
    r"""
    Args:
        parent (None): (default = None)
        text (str):  (default = '')
        align (str): (default = 'center')
        gpath (None): (default = None)
        fontkw (dict): (default = {})

    Kwargs:
        parent, text, align, gpath, fontkw

    Returns:
        ?: label

    CommandLine:
        python -m guitool.guitool_components --exec-newLabel --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> import guitool
        >>> guitool.ensure_qtapp()
        >>> parent = None
        >>> text = ''
        >>> align = 'center'
        >>> gpath = ut.grab_test_imgpath('lena.png')
        >>> fontkw = {}
        >>> label = newLabel(parent, text, align, gpath, fontkw)
        >>> ut.quit_if_noshow()
        >>> label.show()
        >>> guitool.qtapp_loop(qwin=label, freq=10)
    """
    label = QtGui.QLabel(text, parent=parent)
    label.setAlignment(ALIGN_DICT[align])
    adjust_font(label, **fontkw)
    if gpath is not None:
        # http://stackoverflow.com/questions/8211982/qt-resizing-a-qlabel-containing-a-qpixmap-while-keeping-its-aspect-ratio
        # TODO
        label._orig_pixmap = QtGui.QPixmap(gpath)
        label.setPixmap(label._orig_pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setScaledContents(True)

        def _on_resize_slot():
            #print('_on_resize_slot')
            label.setPixmap(label._orig_pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            #label.setPixmap(label._orig_pixmap.scaled(label.size()))
        label._on_resize_slot = _on_resize_slot
        #ut.embed()

    def setColorFG(self, fgcolor):
        #current_sheet = self.styleSheet()
        style_sheet_str = make_style_sheet(bgcolor=None, fgcolor=fgcolor)
        if style_sheet_str is None:
            style_sheet_str = ''
        self.setStyleSheet(style_sheet_str)
    ut.inject_func_as_method(label, setColorFG)
    return label


def getAvailableFonts():
    fontdb = QtGui.QFontDatabase()
    available_fonts = list(map(str, list(fontdb.families())))
    return available_fonts


def layoutSplitter(splitter):
    old_sizes = splitter.sizes()
    print(old_sizes)
    phi = utool.get_phi()
    total = sum(old_sizes)
    ratio = 1 / phi
    sizes = []
    for count, size in enumerate(old_sizes[:-1]):
        new_size = int(round(total * ratio))
        total -= new_size
        sizes.append(new_size)
    sizes.append(total)
    splitter.setSizes(sizes)
    print(sizes)
    print('===')


def msg_event(title, msg):
    """ Returns a message event slot """
    return lambda: guitool_dialogs.msgbox(msg=msg, title=title)


class Spoiler(QtGui.QWidget):
    """
    References:
        # Adapted from c++ version
        http://stackoverflow.com/questions/32476006/how-to-make-an-expandable-collapsable-section-widget-in-qt

    CommandLine:
        python -m guitool.guitool_components Spoiler --show

    Example:
        >>> from guitool.guitool_components import *  # NOQA
        >>> # build test data
        >>> import guitool
        >>> import guitool as gt
        >>> guitool.ensure_qtapp()
        >>> #ut.exec_funckw(newWidget, globals())
        >>> parent = None
        >>> widget1 = GuitoolWidget(parent)
        >>> widget1.addWidget(gt.newButton(
        >>>     widget1, 'Print Hi', lambda: print('hi')))
        >>> widget2 = GuitoolWidget(parent)
        >>> spoiler = widget1.addNewSpoiler(title='spoiler title')
        >>> widget2.addWidget(gt.newButton(
        >>>     widget2, 'Popup Hi', lambda: gt.user_info(widget2, 'hi')))
        >>> spoiler.setContentLayout(widget2.layout())
        >>> self = spoiler
        >>> widget1.show()
        >>> widget1.resize(int(ut.PHI * 500), 500)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=widget1, freq=10)

    #Example:
    #    >>> from guitool.guitool_components import *  # NOQA
    #    >>> # build test data
    #    >>> import guitool
    #    >>> import guitool as gt
    #    >>> guitool.ensure_qtapp()
    #    >>> #ut.exec_funckw(newWidget, globals())
    #    >>> parent = None
    #    >>> widget1 = GuitoolWidget(parent)
    #    >>> widget1.addWidget(gt.newButton(
    #    >>>     widget1, 'Print Hi', lambda: print('hi')))
    #    >>> widget2 = GuitoolWidget(parent)
    #    >>> spoiler = Spoiler(widget1, title='spoiler title')
    #    >>> widget1.addWidget(spoiler)
    #    >>> widget2.addWidget(gt.newButton(
    #    >>>     widget2, 'Popup Hi', lambda: gt.user_info(widget2, 'hi')))
    #    >>> spoiler.setContentLayout(widget2.layout())
    #    >>> self = spoiler
    #    >>> widget1.show()
    #    >>> widget1.resize(int(ut.PHI * 500), 500)
    #    >>> ut.quit_if_noshow()
    #    >>> gt.qtapp_loop(qwin=widget1, freq=10)
    """

    def __init__(self, parent=None, title='', animationDuration=300):
        super(Spoiler, self).__init__(parent=parent)

        self.animationDuration = 150
        self.toggleAnimation = QtCore.QParallelAnimationGroup()
        self.contentArea = QtGui.QScrollArea()
        self.headerLine = QtGui.QFrame()
        self.toggleButton = QtGui.QToolButton()
        self.mainLayout = QtGui.QGridLayout()

        toggleButton = self.toggleButton
        toggleButton.setStyleSheet("QToolButton { border: none; }")
        toggleButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toggleButton.setArrowType(QtCore.Qt.RightArrow)
        toggleButton.setText(str(title))
        toggleButton.setCheckable(True)
        toggleButton.setChecked(False)

        headerLine = self.headerLine
        headerLine.setFrameShape(QtGui.QFrame.HLine)
        headerLine.setFrameShadow(QtGui.QFrame.Sunken)
        headerLine.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)

        #self.contentArea.setStyleSheet("QScrollArea { background-color: white; border: none; }")
        self.contentArea.setStyleSheet("QScrollArea { border: none; }")
        #self.contentArea.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        self.contentArea.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        #self.contentArea.setWidgetResizable(True)
        # start out collapsed
        self.contentArea.setMaximumHeight(0)
        self.contentArea.setMinimumHeight(0)
        # let the entire widget grow and shrink with its content
        self.toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self, "minimumHeight"))
        self.toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self, "maximumHeight"))
        self.toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self.contentArea, "maximumHeight"))
        # don't waste space
        mainLayout = self.mainLayout
        mainLayout.setVerticalSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        row = 0
        mainLayout.addWidget(self.toggleButton, row, 0, 1, 1, QtCore.Qt.AlignLeft)
        row += 1
        mainLayout.addWidget(self.headerLine, row, 2, 1, 1)
        mainLayout.addWidget(self.contentArea, row, 0, 1, 3)
        self.setLayout(self.mainLayout)

        def toggle_spoiler(checked):
            arrow_type = QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
            direction = QtCore.QAbstractAnimation.Forward if checked else QtCore.QAbstractAnimation.Backward
            toggleButton.setArrowType(arrow_type)
            self.toggleAnimation.setDirection(direction)
            self.toggleAnimation.start()
            print('parent = %r' % (parent,))
            #self.toggled.emit()
            #if parent is not None:
            #    parent.adjustSize()

        self.toggle_finished = self.toggleAnimation.finished
        self.toggleButton.clicked.connect(toggle_spoiler)

    def setContentLayout(self, contentLayout):
        # Not sure if this is equivalent to self.contentArea.destroy()
        self.contentArea.destroy()
        self.contentArea.setLayout(contentLayout)

        # Find collapsed height
        collapsedHeight = self.sizeHint().height() - self.contentArea.maximumHeight()
        contentHeight = contentLayout.sizeHint().height()

        for i in range(self.toggleAnimation.animationCount()):
            spoilerAnimation = self.toggleAnimation.animationAt(i)
            spoilerAnimation.setDuration(self.animationDuration)
            spoilerAnimation.setStartValue(collapsedHeight)
            spoilerAnimation.setEndValue(collapsedHeight + contentHeight)
        contentAnimation = self.toggleAnimation.animationAt(self.toggleAnimation.animationCount() - 1)
        contentAnimation.setDuration(self.animationDuration)
        contentAnimation.setStartValue(0)
        contentAnimation.setEndValue(contentHeight)


if __name__ == '__main__':
    """
    CommandLine:
        python -m guitool.guitool_components
        python -m guitool.guitool_components --allexamples
        python -m guitool.guitool_components --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
