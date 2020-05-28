from guitool_ibeis.__PYQT__.QtCore import Qt
import utool as ut
import collections
(print, rrr, profile) = ut.inject2(__name__)


def define_qt_enum(block):
    keys = (line.split()[0] for line in ut.codeblock(block).split('\n'))
    return collections.OrderedDict((key, getattr(Qt, key)) for key in keys)
    # return ut.sort_dict({key: getattr(Qt, key) for key in keys}, 'vals')


ItemDataRoles = define_qt_enum(
    '''
    DisplayRole                # key data to be rendered in the form of text. (QString)
    DecorationRole             # data to be rendered as an icon. (QColor QIcon or QPixmap)
    EditRole                   # data in a form suitable for editing in an editor. (QString)
    ToolTipRole                # data displayed in the items tooltip. (QString)
    StatusTipRole              # data displayed in the status bar. (QString)
    WhatsThisRole              # data displayed in "Whats This?" mode. (QString)
    SizeHintRole               # size hint for item that will be supplied to views. (QSize)
    FontRole                   # font used for items rendered with default delegate. (QFont)
    TextAlignmentRole          # text alignment of items with default delegate. (Qt::AlignmentFlag)
    BackgroundRole             # background brush for items with default delegate. (QBrush)
    ForegroundRole             # foreground brush for items rendered with default delegate. (QBrush)
    CheckStateRole             # checked state of an item. (Qt::CheckState)
    InitialSortOrderRole       # initial sort order of a header view (Qt::SortOrder).
    AccessibleTextRole         # text used by accessibility extensions and plugins (QString)
    AccessibleDescriptionRole  # accessibe description of the item for (QString)
    UserRole                   # first role that can be used for application-specific purposes.
    BackgroundColorRole        # Obsolete. Use BackgroundRole instead.
    TextColorRole              # Obsolete. Use ForegroundRole instead.
    '''
)


# WindowType / WindowFlags
# http://doc.qt.io/qt-5/qt.html#WindowType-enum
# http://doc.qt.io/qt-5/qt.html#WindowType-enum
WindowTypes = define_qt_enum(
    '''
    Widget
    Window
    Dialog
    Sheet
    Drawer
    Popup
    Tool
    ToolTip
    SplashScreen
    Desktop
    SubWindow
    ForeignWindow
    CoverWindow
    ''')

TopLevelWindowTypes = define_qt_enum(
    '''
    MSWindowsFixedSizeDialogHint
    MSWindowsOwnDC
    BypassWindowManagerHint
    X11BypassWindowManagerHint
    FramelessWindowHint
    NoDropShadowWindowHint
    ''')

WindowFlags = define_qt_enum(
    '''
    CustomizeWindowHint
    WindowTitleHint
    WindowSystemMenuHint
    WindowMinimizeButtonHint
    WindowMaximizeButtonHint
    WindowMinMaxButtonsHint
    WindowCloseButtonHint
    WindowContextHelpButtonHint
    MacWindowToolBarButtonHint
    WindowFullscreenButtonHint
    BypassGraphicsProxyWidget
    WindowShadeButtonHint
    WindowStaysOnTopHint
    WindowStaysOnBottomHint
    WindowTransparentForInput
    WindowOverridesSystemGestures
    WindowDoesNotAcceptFocus
    MaximizeUsingFullscreenGeometryHint
    WindowType_Mask
    '''
)


LayoutDirection = define_qt_enum(
    '''
    LeftToRight
    RightToLeft
    ''')



def parse_window_type_and_flags(self):

    type_ = self.windowType()
    for key, val in WindowTypes.items():
        if bin(val).count('1') == 1:
            pass
        # print('{:<16s}: 0x{:08b}'.format(key, val))
        print('{:<16s}: 0x{:08x}'.format(key, val))

    has = []
    missing = []
    flags = int(self.windowFlags())
    for key, val in WindowFlags.items():
        if flags & val == val:
            has.append(key)
        else:
            missing.append(key)
    print('has = %s' % (ut.repr4(has),))
    print('missing = %s' % (ut.repr4(missing),))
    pass


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m guitool_ibeis.qt_enums
        python -m guitool_ibeis.qt_enums --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
