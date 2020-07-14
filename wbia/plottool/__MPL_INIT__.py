# -*- coding: utf-8 -*-
"""
Notes:
    To use various backends certian packages are required

    PyQt
    ...

    Tk
    pip install
    sudo apt-get install tk
    sudo apt-get install tk-dev

    Wx
    pip install wxPython

    GTK
    pip install PyGTK
    pip install pygobject
    pip install pygobject

    Cairo
    pip install pycairo
    pip install py2cairo
    pip install cairocffi
    sudo apt-get install libcairo2-dev


CommandLine:
    python -m wbia.plottool.draw_func2 --exec-imshow --show --mplbe=GTKAgg
    python -m wbia.plottool.draw_func2 --exec-imshow --show --mplbe=TkAgg
    python -m wbia.plottool.draw_func2 --exec-imshow --show --mplbe=WxAgg
    python -m wbia.plottool.draw_func2 --exec-imshow --show --mplbe=WebAgg
    python -m wbia.plottool.draw_func2 --exec-imshow --show --mplbe=gdk
    python -m wbia.plottool.draw_func2 --exec-imshow --show --mplbe=cairo

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
import utool as ut
from six.moves import builtins

ut.noinject(__name__, '[plottool.__MPL_INIT__]')

try:
    profile = getattr(builtins, 'profile')
except AttributeError:

    def profile(func):
        return func


__IS_INITIALIZED__ = False
__WHO_INITIALIZED__ = None


VERBOSE_MPLINIT = ut.get_argflag(('--verb-mpl', '--verbose'))
TARGET_BACKEND = ut.get_argval(
    ('--mpl-backend', '--mplbe'), type_=str, default=os.environ.get('MPL_BACKEND', None)
)
FALLBACK_BACKEND = ut.get_argval(
    ('--mpl-fallback-backend', '--mplfbbe'), type_=str, default='agg'
)


def print_all_backends():
    import matplotlib.rcsetup as rcsetup

    print(rcsetup.all_backends)
    valid_backends = [
        'GTK',
        'GTKAgg',
        'GTKCairo',
        'MacOSX',
        'Qt4Agg',
        'Qt5Agg',
        'TkAgg',
        'WX',
        'WXAgg',
        'CocoaAgg',
        'GTK3Cairo',
        'GTK3Agg',
        'WebAgg',
        'nbAgg',
        'agg',
        'cairo',
        'emf',
        'gdk',
        'pdf',
        'pgf',
        'ps',
        'svg',
        'template',
    ]
    del valid_backends


def get_pyqt():
    have_guitool = ut.check_module_installed('guitool')
    try:
        if have_guitool:
            from wbia.guitool import __PYQT__ as PyQt  # NOQA

            pyqt_version = PyQt._internal.GUITOOL_PYQT_VERSION
        else:
            try:
                import PyQt5 as PyQt

                pyqt_version = 5
            except ImportError:
                import PyQt4 as PyQt

                pyqt_version = 4
    except ImportError:
        PyQt = None
        pyqt_version = None
    return PyQt, pyqt_version


def get_target_backend():
    if (
        not sys.platform.startswith('win32')
        and not sys.platform.startswith('darwin')
        and os.environ.get('DISPLAY', None) is None
    ):
        # Write to files if we cannot display
        # target_backend = 'PDF'
        target_backend = FALLBACK_BACKEND
    else:
        target_backend = TARGET_BACKEND
        if target_backend is None:
            PyQt, pyqt_version = get_pyqt()
            if pyqt_version is None:
                print('[!plotttool] WARNING backend fallback to %s' % (FALLBACK_BACKEND,))
                target_backend = FALLBACK_BACKEND
            elif pyqt_version == 4:
                target_backend = 'Qt4Agg'
            elif pyqt_version == 5:
                target_backend = 'Qt5Agg'
            else:
                raise ValueError('Unknown pyqt version %r' % (pyqt_version,))
    return target_backend


def _init_mpl_rcparams():
    import matplotlib as mpl
    from matplotlib import style

    # http://matplotlib.org/users/style_sheets.html
    nogg = ut.get_argflag('--nogg')
    if not nogg:
        style.use('ggplot')
    # style.use(['ggplot'])
    # print('style.available = %r' % (style.available,))
    # style.use(['bmh'])
    # style.use(['classic'])
    # import utool
    # utool.embed()
    # style.use(['ggplot', 'dark_background'])
    if ut.get_argflag('--notoolbar'):
        toolbar = 'None'
    else:
        toolbar = 'toolbar2'
    mpl.rcParams['toolbar'] = toolbar
    # mpl.rc('text', usetex=False)

    if ut.get_argflag('--usetex'):
        # mpl.rc('text', usetex=True)
        mpl.rcParams['text.usetex'] = True
        # matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
        mpl.rcParams['text.latex.unicode'] = True
    mpl_keypress_shortcuts = [
        key for key in mpl.rcParams.keys() if key.find('keymap') == 0
    ]
    for key in mpl_keypress_shortcuts:
        mpl.rcParams[key] = ''

    CUSTOM_GGPLOT = 1
    if CUSTOM_GGPLOT and not nogg:
        ggplot_style = style.library['ggplot']  # NOQA
        # print('ggplot_style = %r' % (ggplot_style,))
        custom_gg = {
            'axes.axisbelow': True,
            # 'axes.edgecolor': 'white',
            'axes.facecolor': '#E5E5E5',
            'axes.edgecolor': 'none',
            # 'axes.facecolor': 'white',
            'axes.grid': True,
            'axes.labelcolor': '#555555',
            'axes.labelsize': 'large',
            'axes.linewidth': 1.0,
            'axes.titlesize': 'x-large',
            'figure.edgecolor': '0.50',
            'figure.facecolor': 'white',
            'font.size': 10.0,
            'grid.color': 'white',
            'grid.linestyle': '-',
            'patch.antialiased': True,
            'patch.edgecolor': '#EEEEEE',
            'patch.facecolor': '#348ABD',
            'patch.linewidth': 0.5,
            'xtick.color': '#555555',
            'xtick.direction': 'out',
            'ytick.color': '#555555',
            'ytick.direction': 'out',
            'axes.prop_cycle': mpl.cycler(
                'color',
                [
                    '#E24A33',
                    '#348ABD',
                    '#988ED5',
                    '#777777',
                    '#FBC15E',
                    '#8EBA42',
                    '#FFB5B8',
                ],
            ),
        }
        mpl.rcParams.update(custom_gg)

    NICE_DARK_BG = False
    if NICE_DARK_BG:
        dark_style = {
            'axes.edgecolor': 'white',
            'axes.facecolor': 'black',
            'axes.labelcolor': 'white',
            'figure.edgecolor': 'black',
            'figure.facecolor': 'black',
            'grid.color': 'white',
            'lines.color': 'white',
            'patch.edgecolor': 'white',
            'savefig.edgecolor': 'black',
            'savefig.facecolor': 'black',
            'text.color': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
        }
        mpl.rcParams.update(dark_style)
    mpl.rcParams['figure.subplot.top'] = 0.8
    # mpl.rcParams['text'].usetex = False
    # for key in mpl_keypress_shortcuts:
    #    print('%s = %s' % (key, mpl.rcParams[key]))
    # Disable mpl shortcuts
    #    mpl.rcParams['toolbar'] = 'None'
    #    mpl.rcParams['interactive'] = True

    # import matplotlib.pyplot as plt
    # plt.xkcd()


def _mpl_set_backend(target_backend):
    import matplotlib as mpl

    if ut.get_argflag('--leave-mpl-backend-alone'):
        print('[pt] LEAVE THE BACKEND ALONE !!! was specified')
        print('[pt] not changing mpl backend')
    else:
        # mpl.use(target_backend, force=True)
        mpl.use(target_backend, force=False)
        current_backend = mpl.get_backend()
    if not ut.QUIET and ut.VERBOSE:
        print('[pt] current backend is: %r' % current_backend)


def _init_mpl_mainprocess(verbose=VERBOSE_MPLINIT):
    global __IS_INITIALIZED__
    global __WHO_INITIALIZED__
    import matplotlib as mpl

    # mpl.interactive(True)
    current_backend = mpl.get_backend()
    target_backend = get_target_backend()
    if __IS_INITIALIZED__ is True:
        if verbose:
            print(
                '[!plottool] matplotlib has already been initialized.  backend=%r'
                % current_backend
            )
            print('[!plottool] Initially initialized by %r' % __WHO_INITIALIZED__)
            print(
                '[!plottool] Trying to be init by %r'
                % (ut.get_caller_name(N=range(0, 5)))
            )
        return False
    __IS_INITIALIZED__ = True

    if verbose:
        print('[plottool] matplotlib initialized by %r' % __WHO_INITIALIZED__)
        __WHO_INITIALIZED__ = ut.get_caller_name(N=range(0, 5))
    if verbose:
        print('--- INIT MPL---')
        print('[pt] current backend is: %r' % current_backend)
        print('[pt] mpl.use(%r)' % target_backend)
    if current_backend != target_backend:
        _mpl_set_backend(target_backend)
    _init_mpl_rcparams()


@profile
def init_matplotlib(verbose=VERBOSE_MPLINIT):
    if ut.in_main_process():
        PyQt, pyqt_version = get_pyqt()
        return _init_mpl_mainprocess(verbose=verbose)
