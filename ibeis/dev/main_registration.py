from __future__ import absolute_import, division, print_function
import utool
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[main_reg]', DEBUG=False)


def register_utool_aliases():
    #print('REGISTER UTOOL ALIASES')
    import utool
    import matplotlib as mpl
    from ibeis.control.IBEISControl import IBEISControl
    from ibeis.view.guiback import MainWindowBackend
    from ibeis.view.guifront import MainWindowFrontend
    utool.extend_global_aliases([
        (IBEISControl, 'ibs'),
        (MainWindowBackend, 'back'),
        (MainWindowFrontend, 'front'),
        (mpl.figure.Figure, 'fig')
    ])
