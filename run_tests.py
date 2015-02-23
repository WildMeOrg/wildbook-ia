#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import utool as ut


def ensure_testing_data():
    from ibeis.tests import reset_testdbs
    print('Making sure test data exists')
    import ibeis
    from os.path import join
    ut.change_term_title('ENSURE IBEIS TETSDATA')
    reset_testdbs.reset_testdbs()
    workdir = ibeis.get_workdir()
    if not ut.checkpath(join(workdir, 'PZ_MTEST')):
        ibeis.ensure_pz_mtest()
    if not ut.checkpath(join(workdir, 'NAUT_test')):
        ibeis.ensure_nauts()


def run_tests():
    # starts logging for tests
    import ibeis
    ibeis._preload()
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
