#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool as ut


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

    exclude_list = [
    ]

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
    ]

    script_text = ut.autogen_run_tests(test_headers, test_argvs, quick_tests, repodir, exclude_list)

    # HACK TO APPEND EXTRA STUFF
    # TODO Incorporate this more nicely into autogen script
    #python -c "import utool; utool.doctest_funcs(utool.util_class, allexamples=True)"
    script_text = '\n'.join(script_text.split('\n')[0:-2])
    script_text += '\n' + ut.codeblock(
        '''

        # EXTRA DOCTESTS
        RUN_TEST ibeis/model/hots/nn_weights.py --allexamples
        RUN_TEST ibeis/control/DBCACHE_SCHEMA.py --allexamples
        RUN_TEST ibeis/control/DB_SCHEMA.py --allexamples
        RUN_TEST ibeis/model/preproc/preproc_image.py --allexamples
        RUN_TEST ibeis/model/preproc/preproc_chip.py --allexamples
        RUN_TEST ibeis/model/preproc/preproc_feat.py --allexamples
        RUN_TEST ibeis/model/preproc/preproc_encounter.py --allexamples
        RUN_TEST ibeis/model/preproc/preproc_detectimg.py --allexamples
        RUN_TEST ibeis/model/hots/match_chips4.py --allexamples
        RUN_TEST ibeis/model/hots/voting_rules2.py --allexamples
        RUN_TEST ibeis/model/hots/pipeline.py --allexamples
        RUN_TEST ibeis/dbio/export_subset.py --allexamples

        END_TESTS
        '''
    )
    #print(script_text)
    return script_text


if __name__ == '__main__':
    """
    python autogen_test_script.py
    python autogen_test_script.py > _run_tests2.sh
    reset_dbs.sh && _run_tests2.sh
    reset_dbs.sh && _run_tests2.sh --testall
    """
    print(autogen_ibeis_runtest())
    pass
