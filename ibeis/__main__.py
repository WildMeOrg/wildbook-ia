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

CMD = ub.argflag('--cmd')


def dependencies_for_myprogram():
    """ Let pyintaller find these modules

    References:
        http://stackoverflow.com/questions/18596410/importerror-no-module-named-mpl-toolkits-with-maptlotlib-1-3-0-and-py2exe
    """
    from guitool_ibeis.__PYQT__ import QtCore, QtGui  # Pyinstaller hacks  # NOQA
    # from PyQt4 import QtCore, QtGui  # NOQA
    #from PyQt4 import QtCore, QtGui  # NOQA
    from scipy.sparse.csgraph import _validation  # NOQA
    from scipy.special import _ufuncs_cxx  # NOQA
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # NOQA
    #import lru  # NOQA
    # Workaround for mpl_toolkits
    import importlib
    importlib.import_module('mpl_toolkits').__path__


def run_ibeis():
    r"""
    CommandLine:
        python -m ibeis
        python -m ibeis find_installed_tomcat
        python -m ibeis get_annot_groundtruth:1
    """
    import ibeis  # NOQA
    #ut.set_process_title('IBEIS_main')
    #main_locals = ibeis.main()
    #ibeis.main_loop(main_locals)
    #ut.set_process_title('IBEIS_main')
    cmdline_varags = ut.get_cmdline_varargs()
    if len(cmdline_varags) > 0 and cmdline_varags[0] == 'rsync':
        from ibeis.scripts import rsync_ibeisdb
        rsync_ibeisdb.rsync_ibsdb_main()
        sys.exit(0)

    if ub.argflag('--devcmd'):
        # Hack to let devs mess around when using an installer version
        # TODO: add more hacks
        #import utool.tests.run_tests
        #utool.tests.run_tests.run_tests()
        ut.embed()
    # Run the tests of other modules
    elif ub.argflag('--run-utool-tests'):
        raise Exception('Deprecated functionality')
    elif ub.argflag('--run-vtool_ibeis-tests'):
        raise Exception('Deprecated functionality')
    elif ub.argflag(('--run-ibeis-tests', '--run-tests')):
        raise Exception('Deprecated functionality')

    if ub.argflag('-e'):
        """
        ibeis -e print -a default -t default
        """
        # Run dev script if -e given
        import ibeis.dev  # NOQA
        ibeis.dev.devmain()
        print('... exiting')
        sys.exit(0)

    # Attempt to run a test using the funciton name alone
    # with the --tf flag
    # if False:
    #     import ibeis.tests.run_tests
    #     import ibeis.tests.reset_testdbs
    #     import ibeis.scripts.thesis
    #     ignore_prefix = [
    #         #'ibeis.tests',
    #         'ibeis.control.__SQLITE3__',
    #         '_autogen_explicit_controller']
    #     ignore_suffix = ['_grave']
    #     func_to_module_dict = {
    #         'demo_bayesnet': 'ibeis.unstable.demobayes',
    #     }
    #     ut.main_function_tester('ibeis', ignore_prefix, ignore_suffix,
    #                             func_to_module_dict=func_to_module_dict)

    #if ub.argflag('-e'):
    #    import ibeis
    #    expt_kw = ut.get_arg_dict(ut.get_func_kwargs(ibeis.run_experiment),
    #    prefix_list=['--', '-'])
    #    ibeis.run_experiment(**expt_kw)
    #    sys.exit(0)

    doctest_modname = ut.get_argval(
        ('--doctest-module', '--tmod', '-tm', '--testmod'),
        type_=str, default=None, help_='specify a module to doctest')
    if doctest_modname is not None:
        """
        Allow any doctest to be run the main ibeis script

        python -m ibeis --tmod utool.util_str --test-align:0
        python -m ibeis --tmod ibeis.algo.hots.pipeline --test-request_ibeis_query_L0:0 --show
        python -m ibeis --tf request_ibeis_query_L0:0 --show
        ./dist/ibeis/IBEISApp --tmod ibeis.algo.hots.pipeline --test-request_ibeis_query_L0:0 --show  # NOQA
        ./dist/ibeis/IBEISApp --tmod utool.util_str --test-align:0
        ./dist/IBEIS.app/Contents/MacOS/IBEISApp --tmod utool.util_str --test-align:0
        ./dist/IBEIS.app/Contents/MacOS/IBEISApp --run-utool-tests
        ./dist/IBEIS.app/Contents/MacOS/IBEISApp --run-vtool_ibeis-tests
        """
        print('[ibeis] Testing module')
        mod_alias_list = {
            'exptdraw': 'ibeis.expt.experiment_drawing'
        }
        doctest_modname = mod_alias_list.get(doctest_modname, doctest_modname)
        module = ut.import_modname(doctest_modname)
        (nPass, nTotal, failed_list, error_report_list) = ut.doctest_funcs(module=module)
        retcode = 1 - (len(failed_list) == 0)
        #print(module)
        sys.exit(retcode)

    import ibeis
    main_locals = ibeis.main()
    execstr = ibeis.main_loop(main_locals)
    # <DEBUG CODE>
    if 'back' in main_locals and CMD:
        back = main_locals['back']
        front = getattr(back, 'front', None)  # NOQA
        #front = back.front
        #ui = front.ui
    ibs = main_locals['ibs']  # NOQA
    print('-- EXECSTR --')
    print(ub.codeblock(execstr))
    print('-- /EXECSTR --')
    exec(execstr)
    # </DEBUG CODE>


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    run_ibeis()
