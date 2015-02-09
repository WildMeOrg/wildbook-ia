#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import utool as ut


def ensure_testing_data():
    print('Making sure test data exists')
    import ibeis
    from os.path import join
    import sys
    ut.change_term_title('ENSURE IBEIS TETSDATA')
    workdir = ibeis.get_workdir()
    RESET_TESTDATA = ut.get_argflag('--reset-testdata')
    if RESET_TESTDATA:
        if ut.are_you_sure('reseting testdata'):
            ut.delete(join(workdir, 'testdb1'))
            ut.delete(join(workdir, 'PZ_MTEST'))
            ut.delete(join(workdir, 'NAUT_test'))
            print('Reset testdata please rerun script without reset flag')
        else:
            print('Not reseting...')
        sys.exit(0)

    if not ut.checkpath(join(workdir, 'testdb1')):
        ut.cmd('sh reset_dbs.sh')
    if not ut.checkpath(join(workdir, 'PZ_MTEST')):
        ibeis.ensure_pz_mtest()
    if not ut.checkpath(join(workdir, 'NAUT_test')):
        ibeis.ensure_nauts()


def run_tests():
    # Build module list and run tests
    import sys
    ensure_testing_data()
    ut.change_term_title('RUN IBEIS TESTS')
    exclude_doctests_fnames = set([
        'template_definitions.py',
        'autogen_test_script.py',
    ])
    exclude_dirs = [
        '_broken',
        'old',
        'tests',
        'timeits',
        '_scripts',
        '_timeits',
        '_doc',
        'notebook',
    ]
    dpath_list = ['ibeis']
    doctest_modname_list = ut.find_doctestable_modnames(
        dpath_list, exclude_doctests_fnames, exclude_dirs)

    for modname in doctest_modname_list:
        exec('import ' + modname, globals(), locals())
    module_list = [sys.modules[name] for name in doctest_modname_list]
    ut.doctest_module_list(module_list)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    run_tests()
