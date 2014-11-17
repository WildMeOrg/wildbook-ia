#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool as ut
import sys
from six.moves import range, map, filter, zip  # NOQA


def autogen_ibeis_runtest():
    """ special case to generate tests script for IBEIS

    Example:
        >>> from autogen_test_script import *  # NOQA
        >>> test_script = autogen_ibeis_runtest()
        >>> print(test_script)

    CommandLine:
        python -c "import utool; utool.autogen_ibeis_runtest()"
        python -c "import utool; print(utool.autogen_ibeis_runtest())"

        python -c "import utool; print(utool.autogen_ibeis_runtest())" > _run_tests2.sh
        chmod +x _run_tests2.sh

    """

    quick_tests = [
        'ibeis/tests/assert_modules.py'
    ]

    #test_repos = [
    #    '~/code/ibeis'
    #    '~/code/vtool'
    #    '~/code/hesaff'
    #    '~/code/guitool'
    #]

    #test_pattern = [
    #    '~/code/ibeis/test_ibs*.py'
    #]

    test_argvs = '--quiet --noshow'

    misc_pats = [
        'test_utool_parallel.py',
        'test_pil_hash.py',
    ]

    repodir = '~/code/ibeis'
    testdir = 'ibeis/tests'

    exclude_list = []

    # Hacky, but not too bad way of getting in doctests
    # Test to see if doctest_funcs appears after main
    # Do not doctest these modules
    exclude_doctests_fnames = set(['template_definitions.py',
                                   'autogen_test_script.py'])
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
    doctest_modname_list = ut.find_doctestable_modnames(dpath_list, exclude_doctests_fnames, exclude_dirs)

    # Verbosity to show which modules at least have some tests
    #untested_modnames = ut.find_untested_modpaths(dpath_list, exclude_doctests_fnames, exclude_dirs)
    #print('\nUNTESTED MODULES:' + ut.indentjoin(untested_modnames))

    #print('\nTESTED MODULES:' + ut.indentjoin(doctest_modname_list))

    #doctest_modname_list = ut.codeblock('''
    #    ibeis.control.DBCACHE_SCHEMA
    #    ibeis.control.DB_SCHEMA
    #    ibeis.control._autogen_ibeiscontrol_funcs
    #    ibeis.dbio.export_subset
    #    ibeis.dev.experiment_harness
    #    ibeis.model.detect.randomforest
    #    ibeis.model.hots.match_chips4
    #    ibeis.model.hots.nn_weights
    #    ibeis.model.hots.pipeline
    #    ibeis.model.hots.voting_rules2
    #    ibeis.model.preproc.preproc_chip
    #    ibeis.model.preproc.preproc_detectimg
    #    ibeis.model.preproc.preproc_encounter
    #    ibeis.model.preproc.preproc_feat
    #    ibeis.model.preproc.preproc_image
    #    ibeis.viz.viz_sver
    #''').splitlines()

    #module_list = [__import__(name, globals(), locals(), fromlist=[], level=0) for name in modname_list]

    for modname in doctest_modname_list:
        exec('import ' + modname, globals(), locals())
    module_list = [sys.modules[name] for name in doctest_modname_list]
    testcmds = ut.get_module_testlines(module_list, remove_pyc=True, verbose=False, pythoncmd='RUN_TEST')
    #print('\n'.join(testcmds))

    test_headers = [
        # title, default, module, testpattern
        ut.def_test('VTOOL',  dpath='vtool/tests', pat=['test*.py'], modname='vtool'),
        ut.def_test('GUI',    dpath=testdir, pat=['test_gui*.py']),
        ut.def_test('IBEIS',  dpath=testdir, pat=['test_ibs*.py', 'test_delete*.py'], default=True),
        ut.def_test('SQL',    dpath=testdir, pat=['test_sql*.py']),
        ut.def_test('VIEW',   dpath=testdir, pat=['test_view*.py']),
        ut.def_test('MISC',   dpath=testdir, pat=misc_pats),
        ut.def_test('OTHER',  dpath=testdir, pat='OTHER'),
        ut.def_test('HESAFF', dpath='pyhesaff/tests', pat=['test_*.py'], modname='pyhesaff'),
        ut.def_test('DOC', testcmds=testcmds, default=True)
    ]

    script_text = ut.make_run_tests_script_text(test_headers, test_argvs, quick_tests, repodir, exclude_list)

    return script_text

if __name__ == '__main__':
    """
    CommandLine:
        python autogen_test_script.py
        python autogen_test_script.py --verbose > _run_tests2.sh
        python autogen_test_script.py -o _run_tests2.sh
        reset_dbs.sh && _run_tests2.sh
        reset_dbs.sh && _run_tests2.sh --testall
    """
    text = autogen_ibeis_runtest()

    runtests_fpath = ut.get_argval(('-o', '--outfile'), type_=str, default=None)
    if runtests_fpath is None and ut.get_argflag('-w'):
        runtests_fpath = '_run_tests2.py'

    if runtests_fpath is not None:
        ut.write_to(runtests_fpath, text)
    elif ut.get_argflag('--verbose'):
        print(text)
