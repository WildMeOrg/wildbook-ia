#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runs IBIES gui
"""
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool as ut
import ubelt as ub
import sys

(print, rrr, profile) = ut.inject2(__name__)

CMD = ub.argflag('--cmd')


def dependencies_for_myprogram():
    """ Let pyintaller find these modules

    References:
        http://stackoverflow.com/questions/18596410/importerror-no-module-named-mpl-toolkits-with-maptlotlib-1-3-0-and-py2exe
    """
    from wbia.guitool.__PYQT__ import QtCore, QtGui  # Pyinstaller hacks  # NOQA

    # from PyQt4 import QtCore, QtGui  # NOQA
    # from PyQt4 import QtCore, QtGui  # NOQA
    from scipy.sparse.csgraph import _validation  # NOQA
    from scipy.special import _ufuncs_cxx  # NOQA
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # NOQA

    # import lru  # NOQA
    # Workaround for mpl_toolkits
    import importlib

    importlib.import_module('mpl_toolkits').__path__


def main():  # nocover
    import wbia

    print('Looks like the imports worked')
    print('wbia = {!r}'.format(wbia))
    print('wbia.__file__ = {!r}'.format(wbia.__file__))
    print('wbia.__version__ = {!r}'.format(wbia.__version__))


def run_wbia():
    r"""
    CommandLine:
        python -m wbia
        python -m wbia find_installed_tomcat
        python -m wbia get_annot_groundtruth:1
    """
    import wbia  # NOQA

    # ut.set_process_title('wbia_main')
    # main_locals = wbia.main()
    # wbia.main_loop(main_locals)
    # ut.set_process_title('wbia_main')
    cmdline_varags = ut.get_cmdline_varargs()
    if len(cmdline_varags) > 0 and cmdline_varags[0] == 'rsync':
        from wbia.scripts import rsync_wbiadb

        rsync_wbiadb.rsync_ibsdb_main()
        sys.exit(0)

    if ub.argflag('--devcmd'):
        # Hack to let devs mess around when using an installer version
        # TODO: add more hacks
        ut.embed()

    if ub.argflag('-e'):
        """
        wbia -e print -a default -t default
        """
        # Run dev script if -e given
        import wbia.dev  # NOQA

        wbia.dev.devmain()
        print('... exiting')
        sys.exit(0)

    # if ub.argflag('-e'):
    #    import wbia
    #    expt_kw = ut.get_arg_dict(ut.get_func_kwargs(wbia.run_experiment),
    #    prefix_list=['--', '-'])
    #    wbia.run_experiment(**expt_kw)
    #    sys.exit(0)

    import wbia

    main_locals = wbia.main()
    execstr = wbia.main_loop(main_locals)
    # <DEBUG CODE>
    if 'back' in main_locals and CMD:
        back = main_locals['back']
        front = getattr(back, 'front', None)  # NOQA
        # front = back.front
        # ui = front.ui
    ibs = main_locals['ibs']  # NOQA
    print('-- EXECSTR --')
    print(ub.codeblock(execstr))
    print('-- /EXECSTR --')
    exec(execstr)
    # </DEBUG CODE>


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    run_wbia()
