#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runs IBIES gui
"""
import logging
import multiprocessing
import sys

import ubelt as ub
import utool as ut

from wbia.dev import devmain
from wbia.entry_points import main, main_loop
from wbia.scripts.rsync_wbiadb import rsync_ibsdb_main

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')

CMD = ub.argflag('--cmd')


# FIXME (27-Jul-12020) This is currently used by CI to verify installation.
#       Either make this the main function or move to location that makes sense.


def smoke_test():  # nocover

    import wbia

    print('Looks like the imports worked')
    print('wbia = {!r}'.format(wbia))
    print('wbia.__file__ = {!r}'.format(wbia.__file__))
    print('wbia.__version__ = {!r}'.format(wbia.__version__))

    from utool.__main__ import main as utool_main
    from vtool.__main__ import main as vtool_main

    utool_main()
    vtool_main()

    from pydarknet.__main__ import main as pydarknet_main
    from pyflann.__main__ import main as pyflann_main
    from pyhesaff.__main__ import main as pyhesaff_main

    # from pyrf.__main__ import main as pyrf_main

    pyhesaff_main()
    pyflann_main()
    pydarknet_main()
    # pyrf_main()

    import brambox

    print('Brambox: {!r}, {!r}'.format(brambox.__version__, brambox.__file__))

    import lightnet

    print('Lightnet: {!r}, {!r}'.format(lightnet.__version__, lightnet.__file__))


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
        rsync_ibsdb_main()
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

        devmain()
        logger.info('... exiting')
        sys.exit(0)

    main_locals = main()
    execstr = main_loop(main_locals)
    # <DEBUG CODE>
    if 'back' in main_locals and CMD:
        back = main_locals['back']
        front = getattr(back, 'front', None)  # NOQA
        # front = back.front
        # ui = front.ui
    ibs = main_locals['ibs']  # NOQA
    logger.info('-- EXECSTR --')
    logger.info(ub.codeblock(execstr))
    logger.info('-- /EXECSTR --')
    exec(execstr)
    # </DEBUG CODE>


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    run_wbia()
