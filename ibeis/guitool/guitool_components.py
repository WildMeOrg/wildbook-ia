from __future__ import absolute_import, division, print_function
from six.moves import map
from guitool.__PYQT__ import QtCore, QtGui
from guitool.__PYQT__.QtGui import QSizePolicy
from guitool.__PYQT__.QtCore import Qt
import functools
import utool
import utool as ut  # NOQA
from guitool import guitool_dialogs
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[guitool_components]')


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
    if slot_fn is not None:
        action.triggered.connect(slot_fn)
    return action


def newProgressBar(parent, visible=True, verticalStretch=1):
    r"""
    Args:
        parent (?):
        visible (bool):
        verticalStretch (int):

    Returns:
        ?: progressBar

    CommandLine:
        python -m guitool.guitool_components --test-newProgressBar

    Example:
        >>> # DISABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> # build test data
        >>> parent = '?'
        >>> visible = True
        >>> verticalStretch = 1
        >>> # execute function
        >>> progressBar = newProgressBar(parent, visible, verticalStretch)
        >>> # verify results
        >>> result = str(progressBar)
        >>> print(result)
    """
    progressBar = QtGui.QProgressBar(parent)
    sizePolicy = newSizePolicy(progressBar,
                               verticalSizePolicy=QSizePolicy.Maximum,
                               verticalStretch=verticalStretch)
    progressBar.setSizePolicy(sizePolicy)
    progressBar.setProperty('value', 42)
    progressBar.setTextVisible(False)
    progressBar.setVisible(visible)
    setattr(progressBar, '_guitool_sizepolicy', sizePolicy)
    return progressBar


def newOutputLog(parent, pointSize=6, visible=True, verticalStretch=1):
    from .guitool_misc import QLoggedOutput
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


def newTextEdit(parent, visible=True):
    """ This is a text area """
    outputEdit = QtGui.QTextEdit(parent)
    sizePolicy = newSizePolicy(outputEdit, verticalStretch=1)
    outputEdit.setSizePolicy(sizePolicy)
    outputEdit.setAcceptRichText(False)
    outputEdit.setVisible(visible)
    setattr(outputEdit, '_guitool_sizepolicy', sizePolicy)
    return outputEdit


def newLineEdit(parent, text=None, enabled=True, align='center', textChangedSlot=None, visible=True, fontkw={}):
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
    sizePolicy = newSizePolicy(widget, verticalStretch=1)
    widget.setSizePolicy(sizePolicy)
    if text is not None:
        widget.setText(text)
    widget.setEnabled(enabled)
    widget.setAlignment(ALIGN_DICT[align])

    if textChangedSlot is not None:
        widget.textChangedSlot.connect(textChangedSlot)

    #outputEdit.setAcceptRichText(False)
    #outputEdit.setVisible(visible)
    adjust_font(widget, **fontkw)
    setattr(widget, '_guitool_sizepolicy', sizePolicy)
    return widget


def newWidget(parent, orientation=Qt.Vertical,
              verticalSizePolicy=QSizePolicy.Expanding,
              horizontalSizePolicy=QSizePolicy.Expanding,
              verticalStretch=1):
    widget = QtGui.QWidget(parent)

    sizePolicy = newSizePolicy(widget,
                               horizontalSizePolicy=horizontalSizePolicy,
                               verticalSizePolicy=verticalSizePolicy,
                               verticalStretch=1)
    widget.setSizePolicy(sizePolicy)
    if orientation == Qt.Vertical:
        layout = QtGui.QVBoxLayout(widget)
    elif orientation == Qt.Horizontal:
        layout = QtGui.QHBoxLayout(widget)
    else:
        raise NotImplementedError('orientation')
    # Black magic
    widget._guitool_layout = layout
    widget.addWidget = widget._guitool_layout.addWidget
    widget.addLayout = widget._guitool_layout.addLayout
    setattr(widget, '_guitool_sizepolicy', sizePolicy)
    return widget


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


def newButton(parent=None, text='', clicked=None, qicon=None, visible=True,
              enabled=True, bgcolor=None, fgcolor=None, fontkw={}):
    """ wrapper around QtGui.QPushButton
    connectable signals:
        void clicked(bool checked=false)
        void pressed()
        void released()
        void toggled(bool checked)

    Args:
        parent (None):
        text (str):
        clicked (None):
        qicon (None):
        visible (bool):
        enabled (bool):
        bgcolor (None):
        fgcolor (None):
        bold (bool):

    Returns:
        ?: button

    CommandLine:
        python -m guitool.guitool_components --test-newButton

    Example:
        >>> # ENABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> # build test data
        >>> parent = None
        >>> text = ''
        >>> clicked = None
        >>> qicon = None
        >>> visible = True
        >>> enabled = True
        >>> bgcolor = None
        >>> fgcolor = None
        >>> bold = False
        >>> # execute function
        >>> button = newButton(parent, text, clicked, qicon, visible, enabled, bgcolor, fgcolor, bold)
        >>> # verify results
        >>> result = str(button)
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
    return button


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
        ?: combo

    CommandLine:
        python -m guitool.guitool_components --test-newComboBox

    Example:
        >>> # DISABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> # build test data
        >>> parent = None
        >>> options = None
        >>> changed = None
        >>> default = None
        >>> visible = True
        >>> enabled = True
        >>> bgcolor = None
        >>> fgcolor = None
        >>> bold = False
        >>> # execute function
        >>> combo = newComboBox(parent, options, changed, default, visible, enabled, bgcolor, fgcolor, bold)
        >>> # verify results
        >>> result = str(combo)
        >>> print(result)
    """
    class CustomComboBox(QtGui.QComboBox):
        def __init__(combo, parent=None, default=None, options=None, changed=None):
            QtGui.QComboBox.__init__(combo, parent)
            combo.ibswgt = parent
            combo.options = options
            combo.changed = changed
            combo.setEditable(True)
            combo.addItems( [ option[0] for option in combo.options ] )
            combo.currentIndexChanged['int'].connect(combo.currentIndexChangedCustom)
            combo.setDefault(default)

        def setOptionText(combo, option_text_list):
            for index, text in enumerate(option_text_list):
                combo.setItemText(index, text)
            #combo.removeItem()

        def currentIndexChangedCustom(combo, index):
            combo.changed(index, combo.options[index][1])

        def setDefault(combo, default=None):
            """ finds index of backend value and sets the current index """
            if default is not None:
                for index, (text, value) in enumerate(options):
                    if value == default:
                        combo.setCurrentIndex(index)
                        break
            else:
                combo.setCurrentIndex(0)

    combo_kwargs = {
        'parent' : parent,
        'options': options,
        'default': default,
        'changed': changed,
    }
    combo = CustomComboBox(**combo_kwargs)
    if changed is None:
        enabled = False
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
#    #app_style = QtGui.QApplication.style()


def newLabel(parent=None, text='', align='center', fontkw={}):
    label = QtGui.QLabel(text, parent=parent)
    label.setAlignment(ALIGN_DICT[align])
    adjust_font(label, **fontkw)
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
