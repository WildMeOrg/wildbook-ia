# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import six
from six.moves import map, range  # NOQA
from guitool.__PYQT__ import QtCore, QtGui
from guitool.__PYQT__.QtGui import QSizePolicy
from guitool.__PYQT__.QtCore import Qt
import functools
import utool
import utool as ut  # NOQA
from guitool import guitool_dialogs
import weakref
(print, rrr, profile) = utool.inject2(__name__, '[guitool_components]')

DEBUG_WIDGET = ut.get_argflag('--debugwidget')

if DEBUG_WIDGET:
    WIDGET_BASE = QtGui.QFrame
else:
    WIDGET_BASE = QtGui.QWidget

ALIGN_DICT = {
    'center': Qt.AlignCenter,
    'right': Qt.AlignRight | Qt.AlignVCenter,
    'left': Qt.AlignLeft | Qt.AlignVCenter,
    'justify': Qt.AlignJustify,
}


def newSizePolicy(widget,
                  verticalSizePolicy=QSizePolicy.Expanding,
                  horizontalSizePolicy=QSizePolicy.Expanding,
                  horizontalStretch=None,
                  verticalStretch=None):
    """
    input: widget - the central widget
    """
    if verticalStretch is None:
        verticalStretch = 0
    if horizontalStretch is None:
        horizontalStretch = 0
    sizePolicy = QSizePolicy(horizontalSizePolicy, verticalSizePolicy)
    sizePolicy.setHorizontalStretch(horizontalStretch)
    sizePolicy.setVerticalStretch(verticalStretch)
    #sizePolicy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    return sizePolicy


def newSplitter(widget=None, orientation=Qt.Horizontal, verticalStretch=1):
    """
    input: widget - the central widget
    """
    splitter = QtGui.QSplitter(orientation, widget)
    _inject_new_widget_methods(splitter)
    # This line makes the splitter resize with the widget
    sizePolicy = newSizePolicy(splitter, verticalStretch=verticalStretch)
    splitter.setSizePolicy(sizePolicy)
    setattr(splitter, '_guitool_sizepolicy', sizePolicy)
    return splitter


def newTabWidget(parent, horizontalStretch=1, verticalStretch=1):
    tabwgt = QtGui.QTabWidget(parent)
    sizePolicy = newSizePolicy(tabwgt, horizontalStretch=horizontalStretch,
                               verticalStretch=verticalStretch)
    tabwgt.setSizePolicy(sizePolicy)
    setattr(tabwgt, '_guitool_sizepolicy', sizePolicy)

    def addNewTab(self, name):
        tab = QtGui.QTabWidget()
        self.addTab(tab, name)
        tab.setLayout(QtGui.QVBoxLayout())
        # tab.setSizePolicy(*cfg_size_policy)
        _inject_new_widget_methods(tab)
        return tab
    ut.inject_func_as_method(tabwgt, addNewTab)
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


class ProgressHooks(QtCore.QObject, ut.NiceRepr):
    """
    hooks into utool.ProgressIterator.

    A hook represents a fraction of a progress step.
    Hooks can be divided recursively

    TODO:
        use signals and slots to connect to the progress bar
        still doesn't work correctly even with signals and slots, probably
          need to do task function in another thread

        if False:
            for x in ut.ProgIter(ut.expensive_task_gen(40000), nTotal=40000,
                                 prog_hook=ctx.prog_hook):
                pass

    References:
        http://stackoverflow.com/questions/19442443/busy-indication-with-pyqt-progress-bar

    Args:
        progbar (Qt.QProgressBar):
        substep_min (int): (default = 0)
        substep_size (int): (default = 1)
        level (int): (default = 0)

    CommandLine:
        python -m guitool.guitool_components ProgressHooks --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> import guitool as gt
        >>> app = gt.ensure_qtapp()[0]
        >>> parent = newWidget()
        >>> parent.show()
        >>> parent.resize(600, 40)
        >>> progbar = newProgressBar(parent, visible=True)
        >>> proghook = progbar.utool_prog_hook
        >>> subhooks = proghook.subdivide_hooks(num=4)
        >>> hook = subhooks[0]
        >>> hook(0, 2)
        >>> hook(1, 2)
        >>> substep_hooks = hook.make_substep_hooks(num=4)
        >>> hook(2, 2)
        >>> subhook2 = subhooks[1]
        >>> subsubhooks = subhook2.subdivide_hooks(num=2)
        >>> subsubhooks[0](0, 3)
        >>> subsubhooks[0](1, 3)
        >>> subsubhooks[0](2, 3, 'special part')
        >>> subsubhooks[0](3, 3, 'other part')
        >>> app.processEvents()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    progress_changed_signal = QtCore.pyqtSignal(float, str)
    show_indefinite_progress_signal = QtCore.pyqtSignal()

    def __init__(proghook, progbar, global_min=0, global_max=1, level=0):
        super(ProgressHooks, proghook).__init__()
        proghook.progressBarRef = weakref.ref(progbar)
        proghook.global_min = global_min
        proghook.global_max = global_max
        #proghook.substep_min = substep_min
        #proghook.substep_size = substep_size
        proghook._count = 0
        proghook.nTotal = 1
        proghook.progiter = None
        proghook.lbl = 'prog'
        proghook.level = level
        proghook.child_hook_gen = None
        proghook.progress_changed_signal.connect(proghook.on_progress_changed)
        proghook.show_indefinite_progress_signal.connect(proghook.show_indefinite_progress_slot)

    def __nice__(proghook):
        return '(' + proghook.lbl + ', %r, %r)' % proghook.global_bounds()

    @property
    def count(proghook):
        progiter = None
        if proghook.progiter is not None:
            progiter = proghook.progiter()
        if progiter is  not None:
            count = progiter.count
        else:
            count = proghook._count
        return count

    def global_bounds(proghook):
        min_ = proghook.global_min
        max_ = proghook.global_max
        return (min_, max_)

    def global_extent(proghook):
        min_, max_ = proghook.global_bounds()
        return max_ - min_

    def register_progiter(proghook, progiter):
        proghook.progiter = weakref.ref(progiter)
        proghook.nTotal = proghook.progiter().nTotal
        proghook.lbl = proghook.progiter().lbl

    def initialize_subhooks(proghook, num=None, spacing=None):
        subhooks = proghook.make_substep_hooks(num, spacing)
        proghook.child_hook_gen = iter(subhooks)

    def next_subhook(proghook):
        return six.next(proghook.child_hook_gen)

    def subdivide_hooks(proghook, num=None, spacing=None):
        """
        Branches this hook into several new leafs.
        Only progress leafs are used to indicate global progress.
        """
        if num is None:
            num = len(spacing) - 1
        if spacing is None:
            # Assume uniform sub iterators
            import numpy as np
            spacing = np.linspace(0, 1, num + 1)

        #min_, max_ = proghook.global_bounds()
        extent = proghook.global_extent()
        global_spacing = proghook.global_min + (spacing * extent)
        sub_min_list = global_spacing[:-1]
        sub_max_list = global_spacing[1:]

        progbar = proghook.progressBarRef()
        subhook_list = [ProgressHooks(progbar, min_, max_, proghook.level + 1)
                        for min_, max_ in zip(sub_min_list, sub_max_list)]
        return subhook_list

    def make_substep_hooks(proghook, num=None, spacing=None):
        """
        This takes into account your current position, and gives you only
        enough subhooks to complete a single step.

        Need to know current count, stepsize, and total number of steps in this
        subhook.
        """
        if num is None:
            num = len(spacing) - 1
        if spacing is None:
            # Assume uniform sub iterators
            import numpy as np
            spacing = np.linspace(0, 1, num + 1)
        nTotal = proghook.nTotal

        #min_, max_ = proghook.global_bounds()
        step_extent_local = 1 / nTotal
        #assert proghook.count < nTotal, 'already finished this subhook'
        if proghook.count >= nTotal:
            # HACK
            count = nTotal - 1
        else:
            count = proghook.count - 1
        step_extent_global = step_extent_local * proghook.global_extent()
        step_min = count * step_extent_global + proghook.global_min
        global_spacing = step_min + (spacing * step_extent_global)
        sub_min_list = global_spacing[:-1]
        sub_max_list = global_spacing[1:]

        proghook.nTotal / proghook.global_extent()

        progbar = proghook.progressBarRef()
        subhook_list = [ProgressHooks(progbar, min_, max_, proghook.level + 1)
                        for min_, max_ in zip(sub_min_list, sub_max_list)]
        return subhook_list

        #subhook_list = [ProgressHooks(proghook.progressBarRef(), substep_min, substep_size, proghook.level + 1)
        #                for substep_min in substep_min_list]
        #return subhook_list

        #step_min = ((count - 1) / nTotal) * proghook.substep_size  + proghook.substep_min
        #step_size = (1.0 / nTotal) * proghook.substep_size

        #substep_size = step_size / num_substeps
        #substep_min_list = [(step * substep_size) + step_min for step in range(num_substeps)]

        #DEBUG = False
        #if DEBUG:
        #    with ut.Indenter(' ' * 4 * nTotal):
        #        print('\n')
        #        print('+____<NEW SUBSTEPS>____')
        #        print('Making %d substeps for proghook.lbl = %s' % (num_substeps, proghook.lbl,))
        #        print(' * step_min         = %.2f' % (step_min,))
        #        print(' * step_size        = %.2f' % (step_size,))
        #        print(' * substep_size     = %.2f' % (substep_size,))
        #        print(' * substep_min_list = %r' % (substep_min_list,))
        #        print(r'L____</NEW SUBSTEPS>____')
        #        print('\n')

    def set_progress(proghook, count, nTotal=None, lbl=None):
        if nTotal is None:
            nTotal = proghook.nTotal
            if nTotal is None:
                nTotal = 100
        else:
            proghook.nTotal = nTotal
        proghook._count = count
        if lbl is not None:
            proghook.lbl = lbl
        global_fraction = proghook.global_progress()
        proghook.progress_changed_signal.emit(global_fraction, proghook.lbl)
        #proghook.set_progress_slot(count, nTotal)

    def __call__(proghook, count, nTotal=None, lbl=None):
        proghook.set_progress(count, nTotal, lbl)

    def local_progress(proghook):
        """ percent done of this subhook """
        nTotal = proghook.nTotal
        count = proghook.count
        local_fraction = (count) / nTotal
        return local_fraction

    def global_progress(proghook):
        """ percent done of entire process """
        local_fraction = proghook.local_progress()
        extent = proghook.global_extent()
        global_min = proghook.global_min
        global_fraction = global_min + (local_fraction * extent)
        return global_fraction

    @QtCore.pyqtSlot(float, str)
    def on_progress_changed(proghook, global_fraction, lbl):
        if SHOW_TEXT:
            resolution = 75
            num_full = int(round(global_fraction * resolution))
            num_empty = resolution - num_full
            print('\n')
            print('[' + '#' * num_full + '.' * num_empty + '] %7.3f%%' % (global_fraction * 100))
            print('\n')
        progbar = proghook.progressBarRef()
        progbar.setRange(0, 10000)
        progbar.setMinimum(0)
        progbar.setMaximum(10000)
        value = int(round(progbar.maximum() * global_fraction))
        progbar.setFormat(lbl + ' %p%')
        progbar.setValue(value)
        #progbar.setProperty('value', value)
        # major hack
        proghook.force_event_update()

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
        >>> def complex_tasks(hook):
        >>>     progkw = dict(freq=1, backspace=False, autoadjust=False)
        >>>     num = 800
        >>>     for x in ut.ProgressIter(range(4), lbl='TASK', prog_hook=hook, **progkw):
        >>>         ut.get_nth_prime_bruteforce(num)
        >>>         subhook1, subhook2 = hook.make_substep_hooks(2)
        >>>         for task1 in ut.ProgressIter(range(2), lbl='task1.1', prog_hook=subhook1, **progkw):
        >>>             ut.get_nth_prime_bruteforce(num)
        >>>             subsubhooks = subhook1.make_substep_hooks(3)
        >>>             for task1 in ut.ProgressIter(range(7), lbl='task1.1.1', prog_hook=subsubhooks[0], **progkw):
        >>>                 ut.get_nth_prime_bruteforce(num)
        >>>             for task1 in ut.ProgressIter(range(11), lbl='task1.1.2', prog_hook=subsubhooks[1], **progkw):
        >>>                 ut.get_nth_prime_bruteforce(num)
        >>>             for task1 in ut.ProgressIter(range(3), lbl='task1.1.3', prog_hook=subsubhooks[2], **progkw):
        >>>                 ut.get_nth_prime_bruteforce(num)
        >>>         for task2 in ut.ProgressIter(range(10), lbl='task1.2', prog_hook=subhook2, **progkw):
        >>>             ut.get_nth_prime_bruteforce(num)
        >>> # hook into utool progress iter
        >>> progressBar = newProgressBar(parent, visible, verticalStretch)
        >>> hook = progressBar.utool_prog_hook
        >>> complex_tasks(hook)
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
    #label.setAlignment(ALIGN_DICT[align])
    if isinstance(align, six.string_types):
        align = ALIGN_DICT[align]
    label.setAlignment(align)
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


class ResizableTextEdit(QtGui.QTextEdit):
    """
    http://stackoverflow.com/questions/3050537/resizing-qts-qtextedit-to-match-text-height-maximumviewportsize
    """
    def sizeHint(self):
        text = self.toPlainText()
        font = self.document().defaultFont()    # or another font if you change it
        fontMetrics = QtGui.QFontMetrics(font)      # a QFontMetrics based on our font
        textSize = fontMetrics.size(0, text)

        textWidth = textSize.width() + 30       # constant may need to be tweaked
        textHeight = textSize.height() + 30     # constant may need to be tweaked
        return (textWidth, textHeight)


def newTextEdit(parent=None, label=None, visible=None, label_pos='above',
                align='left', text=None, enabled=True, editable=True, fit_to_text=False):
    """ This is a text area """
    #if fit_to_text:
    #outputEdit = ResizableTextEdit(parent)
    #else:
    outputEdit = QtGui.QTextEdit(parent)
    sizePolicy = newSizePolicy(outputEdit, verticalStretch=1)
    outputEdit.setSizePolicy(sizePolicy)
    outputEdit.setAcceptRichText(False)
    if visible is not None:
        outputEdit.setVisible(visible)
    outputEdit.setEnabled(enabled)
    outputEdit.setReadOnly(not editable)
    if text is not None:
        outputEdit.setText(text)
    if isinstance(align, six.string_types):
        align = ALIGN_DICT[align]
    outputEdit.setAlignment(align)
    if label is None:
        pass

    if fit_to_text:
        font = outputEdit.document().defaultFont()    # or another font if you change it
        fontMetrics = QtGui.QFontMetrics(font)      # a QFontMetrics based on our font
        textSize = fontMetrics.size(0, text)

        textWidth = textSize.width() + 30       # constant may need to be tweaked
        textHeight = textSize.height() + 30     # constant may need to be tweaked
        outputEdit.setMinimumSize(textWidth, textHeight)
    #else:
    #    outputEdit.setMinimumHeight(0)

    setattr(outputEdit, '_guitool_sizepolicy', sizePolicy)
    return outputEdit


def newLineEdit(parent, text=None, enabled=True, align='center',
                textChangedSlot=None, textEditedSlot=None,
                editingFinishedSlot=None, visible=True, readOnly=False,
                editable=None,
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
    if editable is not None:
        readOnly = editable
    widget = QtGui.QLineEdit(parent)
    sizePolicy = newSizePolicy(widget,
                               verticalSizePolicy=QSizePolicy.Fixed,
                               verticalStretch=verticalStretch)
    widget.setSizePolicy(sizePolicy)
    if text is not None:
        widget.setText(text)
    widget.setEnabled(enabled)
    if isinstance(align, six.string_types):
        align = ALIGN_DICT[align]
    widget.setAlignment(align)
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


def newFrame(*args, **kwargs):
    kwargs = kwargs.copy()
    widget = QtGui.QFrame()
    orientation = kwargs.get('orientation', None)
    if orientation is None:
        orientation = Qt.Vertical
    if orientation == Qt.Vertical:
        layout = QtGui.QVBoxLayout(widget)
    elif orientation == Qt.Horizontal:
        layout = QtGui.QHBoxLayout(widget)
    else:
        raise NotImplementedError('orientation')
    widget.setLayout(layout)
    _inject_new_widget_methods(widget)
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


def _make_new_widget_func(widget_cls):
    def new_widget_maker(*args, **kwargs):
        kwargs = kwargs.copy()
        verticalStretch = kwargs.pop('verticalStretch', 1)
        widget = widget_cls(*args, **kwargs)
        _inject_new_widget_methods(widget)
        # This line makes the widget resize with the widget
        sizePolicy = newSizePolicy(widget, verticalStretch=verticalStretch)
        widget.setSizePolicy(sizePolicy)
        setattr(widget, '_guitool_sizepolicy', sizePolicy)
        return widget
    return new_widget_maker


def _inject_new_widget_methods(self):
    """ helper for guitool widgets """
    import guitool as gt
    # Black magic
    guitype_list = [
        'Widget', 'Button', 'LineEdit', 'ComboBox', 'Label', 'Spoiler',
        'Frame', 'Splitter', 'TabWidget']
    # Creates addNewWidget and newWidget
    for guitype in guitype_list:
        if isinstance(guitype, tuple):
            guitype, widget_cls = guitype
            newfunc = _make_new_widget_func(widget_cls)
        else:
            if hasattr(gt, 'new' + guitype):
                newfunc = getattr(gt, 'new' + guitype)
                ut.inject_func_as_method(self, newfunc, 'new' + guitype)
            else:
                newfunc = getattr(gt, guitype)
        addnew_func = _addnew_factory(self, newfunc)
        ut.inject_func_as_method(self, addnew_func, 'addNew' + guitype)

    if not hasattr(self, 'addWidget'):
        def _make_add_new_widgets():
            def addWidget(self, widget, *args, **kwargs):
                self.layout().addWidget(widget, *args, **kwargs)
                return widget

            def newHWidget(self, **kwargs):
                return self.addNewWidget(orientation=Qt.Horizontal, **kwargs)

            def newVWidget(self, **kwargs):
                return self.addNewWidget(orientation=Qt.Vertical, **kwargs)
            return addWidget, newVWidget, newHWidget
        for func  in _make_add_new_widgets():
            ut.inject_func_as_method(self, func, ut.get_funcname(func))

    ut.inject_func_as_method(self, print_widget_heirarchy)
    # Above code is the same as saying
    #     self.newButton = ut.partial(newButton, self)
    #     self.newWidget = ut.partial(newWidget, self)
    #     ... etc


def _addnew_factory(self, newfunc):
    """ helper for guitool widgets """
    def _addnew(self, *args, **kwargs):
        new_widget = newfunc(self, *args, **kwargs)
        self.addWidget(new_widget)
        return new_widget
    return _addnew


#class GuitoolWidget(QtGui.QWidget):
class GuitoolWidget(WIDGET_BASE):
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

    def __init__(self, parent=None, orientation=Qt.Vertical,
                 verticalSizePolicy=QSizePolicy.Expanding,
                 horizontalSizePolicy=QSizePolicy.Expanding,
                 verticalStretch=0, **kwargs):
        super(GuitoolWidget, self).__init__(parent)

        #sizePolicy = newSizePolicy(self,
        #                           horizontalSizePolicy=horizontalSizePolicy,
        #                           verticalSizePolicy=verticalSizePolicy,
        #                           verticalStretch=verticalStretch)
        #self.setSizePolicy(sizePolicy)
        #setattr(self, '_guitool_sizepolicy', sizePolicy)
        if orientation == Qt.Vertical:
            layout = QtGui.QVBoxLayout(self)
        elif orientation == Qt.Horizontal:
            layout = QtGui.QHBoxLayout(self)
        else:
            raise NotImplementedError('orientation')
        #layout.setSpacing(0)
        self.setLayout(layout)
        self._guitool_layout = layout
        #layout.setAlignment(Qt.AlignBottom)
        #self.addWidget = self._guitool_layout.addWidget
        #self.addLayout = self._guitool_layout.addLayout
        _inject_new_widget_methods(self)
        self.initialize(**kwargs)

        if DEBUG_WIDGET:
            # debug code
            self.setStyleSheet("background-color: rgb(255,0,0); margin:5px; border:1px solid rgb(0, 255, 0); ")
            #self.setStyleSheet("background-color: border:5px solid rgb(255, 0, 0); ")

    @classmethod
    def as_dialog(cls, parent=None, **kwargs):
        widget = cls(**kwargs)
        dlg = QtGui.QDialog(parent)
        #dlg.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        #dlg.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        #dlg.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        #widget.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        dlg.widget = widget
        dlg.vlayout = QtGui.QVBoxLayout(dlg)
        #dlg.vlayout.setAlignment(Qt.AlignBottom)
        dlg.vlayout.addWidget(widget)
        widget.closed.connect(dlg.close)
        dlg.setWindowTitle(widget.windowTitle())

        if DEBUG_WIDGET:
            # debug code
            dlg.setStyleSheet("background-color: rgb(255,0,0); margin:0px; border:1px solid rgb(0, 255, 0); ")
        return dlg

    def initialize(self, **kwargs):
        pass

    def addLayout(self, *args, **kwargs):
        return self._guitool_layout.addLayout(*args, **kwargs)

    #def addWidget(self, widget, *args, **kwargs):
    #    #self._guitool_layout.addWidget(widget, *args, **kwargs)
    #    self.layout().addWidget(widget, *args, **kwargs)
    #    return widget

    #def addNewWidget(self, *args, **kwargs):
    #    new_widget = self.newWidget(*args, **kwargs)
    #    return self.addWidget(new_widget)

    def closeEvent(self, event):
        event.accept()
        self.closed.emit()


def prop_text_map(prop, val):
    if prop == 'QtGui.QSizePolicy':
        pol_info = {eval('QtGui.QSizePolicy.' + key): key for key in
                    ['Fixed', 'Minimum', 'Maximum', 'Preferred', 'Expanding',
                     'MinimumExpanding', 'Ignored', ]}
        return pol_info[val]
    else:
        return val


def get_nested_attr(obj, attr):
    """
    attr = 'sizePolicy().verticalPolicy()'
    """
    attr_list = attr.split('.')
    current = obj
    for a in attr_list:
        flag = a.endswith('()')
        a_ = a[:-2] if flag else a
        current = getattr(current, a_, None)
        if current is None:
            raise AttributeError(attr)
        if flag:
            current = current()
    return current


def walk_widget_heirarchy(obj, **kwargs):
    default_attrs = [
        'sizePolicy'
        'widgetResizable'
        'maximumHeight'
        'minimumHeight'
        'alignment'
        'spacing',
    ]
    attrs = kwargs.get('attrs', None)
    max_depth = kwargs.get('max_depth', None)
    skip = kwargs.get('skip', False)
    level = kwargs.get('level', 0)

    if attrs is None:
        attrs = default_attrs
    else:
        attrs = ut.ensure_iterable(attrs)

    children = obj.children()
    lines = []
    info = str(ut.type_str(obj.__class__)).replace('PyQt4', '') + ' - ' + repr(obj.objectName())
    #print(info)
    lines.append(info)
    for attr in attrs:
        if attr == 'sizePolicy' and hasattr(obj, 'sizePolicy'):
            vval = prop_text_map('QtGui.QSizePolicy', obj.sizePolicy().verticalPolicy())
            hval = prop_text_map('QtGui.QSizePolicy', obj.sizePolicy().horizontalPolicy())
            lines.append('  * verticalSizePolicy   = %r' % vval)
            lines.append('  * horizontalSizePolicy = %r' % hval)
        else:
            try:
                val = get_nested_attr(obj, attr + '()')
                lines.append('  * %s = %r' % (attr, prop_text_map(attr, val)))
            except AttributeError:
                pass
    if skip and len(lines) == 1:
        lines = []
    #if hasattr(obj, 'alignment'):
    #    val = obj.alignment()
    #    lines.append('  * widgetResizable = %r' % prop_text_map('widgetResizable', val))
    lines = [ut.indent(line, ' ' * level * 4) for line in lines]
    next_level = level + 1
    kwargs = kwargs.copy()
    kwargs['level'] = level + 1
    if max_depth is None or next_level <= max_depth:
        for child in children:
            child_info = walk_widget_heirarchy(child, **kwargs)
            lines.extend(child_info)
    return lines


def print_widget_heirarchy(obj, *args, **kwargs):
    lines = walk_widget_heirarchy(obj, *args, **kwargs)
    text = '\n'.join(lines)
    print(text)


def fix_child_attr_heirarchy(obj, attr, val):
    if hasattr(obj, attr):
        getattr(obj, attr)(val)
        # obj.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    for child in obj.children():
        fix_child_attr_heirarchy(child, attr, val)


def fix_child_size_heirarchy(obj, pol):
    if hasattr(obj, 'sizePolicy'):
        obj.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    for child in obj.children():
        fix_child_size_heirarchy(child, pol)


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
        >>> tablename = None
        >>> dict_ = {'K': 1, 'Knorm': 5,
        >>>          'choice': ut.ParamInfo(varname='choice', default='one',
        >>>                                 valid_values=['one', 'two'])}
        >>> config = dtool.Config.from_dict(dict_, tablename)
        >>> dlg = guitool.ConfigConfirmWidget.as_dialog(
        >>>     title='Confirm Merge Query',
        >>>     msg='Confirm',
        >>>     detailed_msg=ut.lorium_ipsum()*10,
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
        #kwargs['horizontalSizePolicy'] = QSizePolicy.Minimum
        kwargs['horizontalSizePolicy'] = QSizePolicy.Expanding
        kwargs['verticalSizePolicy'] = QSizePolicy.Expanding
        super(ConfigConfirmWidget, self).__init__(*args, **kwargs)

    def initialize(self, title, msg, config, options=None, default=None, detailed_msg=None):
        #import copy
        from guitool import PrefWidget2
        self.msg = msg
        self.orig_config = config
        self.config = config.deepcopy()
        self.confirm_option = None

        self.setWindowTitle(title)

        layout = self.layout()

        if 1:
            msg_widget = newLabel(self, text=msg, align='left')
            #msg_widget = newTextEdit(self, text=msg, align='left', editable=False, fit_to_text=True)
            msg_widget.setObjectName('msg_widget')
            #msg_widget = self.addNewLabel(msg, align='left')
            #msg_widget.setSizePolicy(newSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred,
            #                                       verticalStretch=1))
            print(msg_widget.sizeHint())
            msg_widget.setSizePolicy(newSizePolicy(QtGui.QSizePolicy.Expanding,
                                                   QtGui.QSizePolicy.Maximum,
                                                   verticalStretch=1))
            #msg_widget.setSizePolicy(newSizePolicy(QtGui.QSizePolicy.Preferred,
            #QtGui.QSizePolicy.Expanding, #verticalStretch=1))
            layout.addWidget(msg_widget)

        if 1 and config is not None:
            self.editConfig = PrefWidget2.newConfigWidget(self.config, user_mode=True)
            #if not ut.get_argflag('--nospoiler'):
            self.spoiler = Spoiler(self, title='Advanced Configuration')
            #self.spoiler.setSizePolicy(newSizePolicy(QtGui.QSizePolicy.Expanding,
            #                                         QtGui.QSizePolicy.Preferred,
            #                                         verticalStretch=0))
            self.spoiler.setObjectName('spoiler')
            self.spoiler.setContentLayout(self.editConfig)
            #self.layout().addStretch(1)
            self.addWidget(self.spoiler)
            #self.addWidget(self.spoiler, alignment=Qt.AlignTop)
            self.spoiler.toggle_finished.connect(self._size_adjust_slot)
            #else:
            #    self.addWidget(self.editConfig)

        if 1 and detailed_msg is not None:
            detailed_msg_widget = newTextEdit(text=detailed_msg, editable=False)
            detailed_msg_widget.setObjectName('detailed_msg_widget')
            self.spoiler2 = Spoiler(self, title='Details')
            #self.spoiler2.setSizePolicy(newSizePolicy(QtGui.QSizePolicy.Expanding,
            #                                          QtGui.QSizePolicy.Preferred,
            #                                          verticalStretch=0))
            self.spoiler2.setObjectName('spoiler2')
            self.spoiler2.setContentLayout(detailed_msg_widget)
            self.addWidget(self.spoiler2)
            self.spoiler2.toggle_finished.connect(self._size_adjust_slot)
            #self.spoiler2.setAlignment(Qt.AlignBottom)

        if 1:
            self.button_row = self.newHWidget(verticalStretch=1000)
            self.button_row.setObjectName('button_row')
            self.button_row.setSizePolicy(newSizePolicy(QtGui.QSizePolicy.Expanding,
                                                        QtGui.QSizePolicy.Maximum))
            self.button_row._guitool_layout.setAlignment(Qt.AlignBottom)
            if options is None:
                options = ['Confirm']
                if default is None:
                    default = options[0]
            def _make_option_clicked(opt):
                def _wrap():
                    return self.confirm(opt)
                return _wrap

            self.default_button = None
            for opt in options:
                button = self.button_row.addNewButton(opt, clicked=_make_option_clicked(opt))
                if opt == default:
                    self.default_button = button

            button = self.button_row.addNewButton('Cancel', clicked=self.cancel)
            if self.default_button is None:
                self.default_button = button
            # button.setDefault(True)
            # button.setAutoDefault(True)
            # button.setFocus(Qt.OtherFocusReason)
            # button.setFocus(Qt.ActiveWindowFocusReason)

            # button.setFocusPolicy(QtCore.Qt.TabFocus)
            # button.setFocus(True)
            # QtCore.Qt.TabFocus)
            # import utool
            # utool.embed()
            # button.setFocus(True)
            # button.
            button.activateWindow()
            # import utool
            # utool.embed()

        self.print_widget_heirarchy()

        #self.layout().setAlignment(Qt.AlignBottom)
        self.layout().setAlignment(Qt.AlignTop)
        #self.layout().setSizeConstraint(QtGui.QLayout.SetFixedSize)
        #self.resize(668, 530)
        #self.update_state()

    @classmethod
    def as_dialog(cls, *args, **kwargs):
        dlg = super(ConfigConfirmWidget, cls).as_dialog(*args, **kwargs)
        # import utool
        # utool.embed()
        # Set focust after creating window
        dlg.widget.default_button.setFocus(True)
        return dlg
        # Set default button

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

    def _size_adjust_slot(self, checked):

        #def adjusted_size(q):
        #    """
        #    gvim ~/code/qt4/src/gui/kernel/qwidget.cpp
        #    """
        #    #Q_Q(const QWidget);
        #    s = q.sizeHint()
        #    layout = q.layout()

        #    if (q.isWindow()):
        #        exp = Qt.Orientations()
        #        if (layout) :
        #            if (layout.hasHeightForWidth()):
        #                s.setHeight(layout.totalHeightForWidth(s.width()))
        #            exp = layout.expandingDirections()
        #        else:
        #            if (q.sizePolicy().hasHeightForWidth()):
        #                s.setHeight(q.heightForWidth(s.width()))
        #            exp = q.sizePolicy().expandingDirections()
        #        if (exp & Qt.Horizontal):
        #            s.setWidth(max(s.width(), 200))
        #        if (exp & Qt.Vertical):
        #            s.setHeight(max(s.height(), 100))

        #        #if defined(Q_WS_X11)
        #        try:
        #            screen = QtGui.QApplication.desktop().screenGeometry(q.pos())
        #        except Exception:
        #            #else // all others
        #            screen = QtGui.QApplication.desktop().screenGeometry(q.x11Info().screen())
        #            #endif

        #        #if defined (Q_WS_WINCE) || defined (Q_OS_SYMBIAN)
        #        try:
        #            s.setWidth(min(s.width(), screen.width()))
        #            s.setHeight(min(s.height(), screen.height()))
        #        except Exception:
        #            #else
        #            s.setWidth(min(s.width(), screen.width() * 2 / 3))
        #            s.setHeight(min(s.height(), screen.height() * 2 / 3))
        #            #endif
        #        #if (QTLWExtra *extra = maybeTopData())
        #        #    extra.sizeAdjusted = true

        #    if (not s.isValid()):
        #        r = q.childrenRect()  # get children rectangle
        #        if (not r.isNull()):
        #            s = r.size() + QtCore.QSize(2 * r.x(), 2 * r.y())
        #    return s

        def _adjust_widget(w):
            print('-----------')
            print('w = %r' % (w,))
            orig_size = w.size()
            hint_size = w.sizeHint()
            #adj_size = adjusted_size(w)
            r = w.childrenRect()  # get children rectangle
            adj_size = r.size()
            #+ QtCore.QSize(2 * r.x(), 2 * r.y())
            #height = min(adj_size.height(), hint_size.height())
            height = hint_size.height()
            newsize = (orig_size.width(), height)
            print('orig_size = %r' % (orig_size,))
            print('hint_size = %r' % (hint_size,))
            print('adj_size = %r' % (adj_size,))
            print('newsize = %r' % (newsize,))
            #w.setMinimumSize(*newsize)
            w.resize(*newsize)
            print('Actual new size = %r' % (w.size()))

        top = self.topLevelWidget()
        #top.ensurePolished()
        if not checked:
            _adjust_widget(top)
        #_adjust_widget(self)

        #parent = self.parent()
        #_adjust_widget(self)
        #if parent is not None:
        #    _adjust_widget(parent)

    def cancel(self):
        print('[gt] Canceled confirm config')
        self.close()


def newButton(parent=None, text='', clicked=None, pressed=None, qicon=None, visible=True,
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
    enabled = False
    if clicked is not None:
        but_kwargs['clicked'] = clicked
        enabled = True
    if pressed is not None:
        but_kwargs['pressed'] = pressed
        enabled = True
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


class Spoiler(WIDGET_BASE):
    r"""
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
        >>>      widget1, 'Print Hi', lambda: print('hi')))
        >>> #widget2 = GuitoolWidget(parent)
        >>> #widget2.addWidget(gt.newButton(
        >>> #    widget2, 'Popup Hi', lambda: gt.user_info(widget2, 'hi')))
        >>> spoiler = Spoiler(title='spoiler title')
        >>> widget1._guitool_layout.addWidget(spoiler)
        >>> #top = widget1.addNewFrame()
        >>> #top._guitool_layout.addWidget(spoiler)
        >>> detailed_msg = 'Foo\nbar'
        >>> child_widget = QtGui.QTextEdit()
        >>> #child_widget.setWordWrap(True)
        >>> #child_widget = QtGui.QPushButton()
        >>> child_widget.setObjectName('child_widget')
        >>> child_widget.setText(ut.lorium_ipsum() * 10)
        >>> #vbox = QtGui.QVBoxLayout()
        >>> #vbox.setContentsMargins(0, 0, 0, 0)
        >>> #vbox.addWidget(child_widget)
        >>> #child_widget.setSizePolicy(newSizePolicy(QtGui.QSizePolicy.Ignored,
        >>> #                                         QtGui.QSizePolicy.Ignored))
        >>> # spoiler = widget1.addNewSpoiler(title='spoiler title')
        >>> #contentLayout = widget2.layout()
        >>> spoiler.setContentLayout(child_widget)
        >>> widget1.print_widget_heirarchy()
        >>> #widget1.setStyleSheet("background-color: rgb(255,0,0); margin:5px; border:1px solid rgb(0, 255, 0); ")
        >>> widget1.layout().setAlignment(Qt.AlignBottom)
        >>> widget1.show()
        >>> #widget1.resize(int(ut.PHI * 500), 500)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=widget1, freq=10)
    """
    toggle_finished = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None, title='', animationDuration=300, checked=False, contentWidget=None):
        super(Spoiler, self).__init__(parent=parent)

        # Maps checked states to arrows and animation directions
        self._arrow_states = {
            False: QtCore.Qt.RightArrow,
            True: QtCore.Qt.DownArrow,
        }
        self._animation_state = {
            False: QtCore.QAbstractAnimation.Backward,
            True: QtCore.QAbstractAnimation.Forward,
        }
        self.change_policy = False
        #:
        self._header_size_policy_states = {
            #False: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed),
            #False: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum),
            False: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum),
            True: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding),
        }
        self._self_size_policy = {
            #False: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed),
            #False: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum),
            False: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum),
            True: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding),
        }
        self._scroll_size_policy_states = {
            #False: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed),
            False: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding),
            #False: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding),
            True: newSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding),
        }
        if not self.change_policy:
            del self._header_size_policy_states[True]
            del self._scroll_size_policy_states[True]
            del self._scroll_size_policy_states[False]
        self.checked = checked

        self.animationDuration = 150
        #150

        self.toggleButton = QtGui.QToolButton()
        toggleButton = self.toggleButton
        toggleButton.setStyleSheet('QToolButton { border: none; }')
        toggleButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toggleButton.setText(str(title))
        toggleButton.setCheckable(True)
        toggleButton.setArrowType(self._arrow_states[self.checked])
        toggleButton.setChecked(self.checked)
        toggleButton.clicked.connect(self.toggle_spoiler)

        self.headerLine = QtGui.QFrame()
        self.headerLine.setFrameShape(QtGui.QFrame.HLine)
        self.headerLine.setFrameShadow(QtGui.QFrame.Sunken)
        if self.change_policy:
            self.headerLine.setSizePolicy(self._header_size_policy_states[self.checked])
        else:
            self.headerLine.setSizePolicy(self._header_size_policy_states[False])

        if False:
            if contentWidget is None:
                self.contentWidget = QtGui.QScrollArea()
                self.contentWidget.setStyleSheet('QScrollArea { background-color: white; border: none; }')
                if self.change_policy:
                    self.contentWidget.setSizePolicy(self._scroll_size_policy_states[self.checked])
                else:
                    self.contentWidget.setSizePolicy(self._scroll_size_policy_states[False])
                self.contentWidget.setStyleSheet('QScrollArea { border: none; }')

                # start out collapsed
                self.contentWidget.setMaximumHeight(0)
                self.contentWidget.setMinimumHeight(0)
                self.contentWidget.setWidgetResizable(True)
            else:
                self.contentWidget = contentWidget
        else:
            self.contentWidget = None

        # let the entire widget grow and shrink with its content
        # The animation forces the minimum and maximum height to be equal
        # By having the minimum and maximum height simultaniously
        self.toggleAnimation = QtCore.QParallelAnimationGroup()
        self.spoiler_animations = [
            QtCore.QPropertyAnimation(self, 'minimumHeight'),
            QtCore.QPropertyAnimation(self, 'maximumHeight'),
        ]
        self.content_animations = [
            #QtCore.QPropertyAnimation(self.contentWidget, 'maximumHeight')
        ]
        for animation in self.spoiler_animations + self.content_animations:
            self.toggleAnimation.addAnimation(animation)
        #self.toggle_finished = self.toggleAnimation.finished

        # don't waste space
        self.mainLayout = QtGui.QGridLayout()
        #self.mainLayout = QtGui.QVBoxLayout()
        mainLayout = self.mainLayout
        mainLayout.setVerticalSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        #mainLayout.addWidget(self.toggleButton, alignment=QtCore.Qt.AlignLeft)
        #mainLayout.addWidget(self.contentWidget)
        mainLayout.addWidget(self.toggleButton, 0, 0, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        #mainLayout.addWidget(self.headerLine, 1, 2, 1, 1)
        #mainLayout.addWidget(self.contentWidget, 1, 0, 1, 3)
        self.setLayout(self.mainLayout)

        self.setMaximumHeight(16777215)
        self.setMinimumHeight(0)

        self.toggleAnimation.finished.connect(self.finalize_animation)
        self.setSizePolicy(self._self_size_policy[self.checked])

        if DEBUG_WIDGET:
            # debug code
            self.setStyleSheet("background-color: rgb(255,0,0); margin:5px; border:1px solid rgb(0, 255, 0); ")

    def finalize_animation(self):
        if self.checked:
            self.contentWidget.setMaximumHeight(16777215)
            self.contentWidget.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            self.setMinimumHeight(0)
        else:
            self.contentWidget.setMaximumHeight(0)
            self.contentWidget.setMinimumHeight(0)
            #self.setMaximumHeight(0)
        self.toggle_finished.emit(self.checked)

    def toggle_spoiler(self, checked):
        self.checked = checked
        self.toggleButton.setArrowType(self._arrow_states[self.checked])
        self.toggleAnimation.setDirection(self._animation_state[self.checked])

        self.setSizePolicy(self._self_size_policy[self.checked])

        if self.change_policy:
            self.headerLine.setSizePolicy(self._header_size_policy_states[self.checked])
            self.contentWidget.setSizePolicy(self._scroll_size_policy_states[self.checked])
        self.toggleAnimation.start()

    def setContentLayout(self, contentLayout):
        # Not sure if this is equivalent to self.contentWidget.destroy()
        #self.contentWidget.destroy()
        try:
            self.contentWidget.setLayout(contentLayout)
        except Exception:
            #import utool
            #utool.embed()
            # HACKY
            contentWidgetNew = contentLayout
            contentWidgetOld = self.contentWidget
            #self.contentWidget.setWidget(contentWidget)

            if contentWidgetOld is not None:
                # Replace existing scrollbar with something else
                self.mainLayout.removeWidget(contentWidgetOld)
                for animation in self.content_animations:
                    self.toggleAnimation.removeAnimation(animation)

            self.contentWidget = contentWidgetNew
            self.content_animations = [
                QtCore.QPropertyAnimation(self.contentWidget, 'maximumHeight')
            ]
            for animation in self.content_animations:
                self.toggleAnimation.addAnimation(animation)

            self.contentWidget.setMaximumHeight(0)
            self.contentWidget.setMinimumHeight(0)

            self.mainLayout.addWidget(self.contentWidget, 1, 0, 1, 3)
            #if False:
            #    if self.change_policy:
            #        self.contentWidget.setSizePolicy(self._scroll_size_policy_states[self.checked])
            #    else:
            #        self.contentWidget.setSizePolicy(self._scroll_size_policy_states[False])

        # Find content height
        collapsedConentHeight = 0
        expandedContentHeight = contentLayout.sizeHint().height()

        # Container height
        collapsedSpoilerHeight = self.sizeHint().height() - self.contentWidget.maximumHeight()
        expandedSpoilerHeight = collapsedSpoilerHeight + expandedContentHeight

        contentStart = collapsedConentHeight
        contentEnd = expandedContentHeight

        spoilerStart = collapsedSpoilerHeight
        spoilerEnd = expandedSpoilerHeight

        if self.checked:
            # Start expanded
            spoilerStart, spoilerEnd = spoilerEnd, spoilerStart
            contentStart, contentEnd = contentEnd, contentStart
            self.contentWidget.setMinimumHeight(contentStart)
        self.spoilerStart = spoilerStart
        self.spoilerEnd = spoilerEnd

        for spoilerAnimation in self.spoiler_animations:
            spoilerAnimation.setDuration(self.animationDuration)
            spoilerAnimation.setStartValue(spoilerStart)
            spoilerAnimation.setEndValue(spoilerEnd)

        for contentAnimation in self.content_animations:
            contentAnimation.setDuration(self.animationDuration)
            contentAnimation.setStartValue(contentStart)
            contentAnimation.setEndValue(contentEnd)


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
