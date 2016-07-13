# -*- coding: utf-8 -*-
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
    if not ut.checkpath(join(workdir, 'wd_peter2')):
        ibeis.ensure_wilddogs()


def run_tests():
    """
        >>> from ibeis.tests.run_tests import *  # NOQA

    """

    # DONT USE THESE FLAGS
    #print('--testall and --testslow give you more tests')
    # starts logging for tests
    import ibeis

    ibeis._preload()
    # Build module list and run tests
    import sys
    ensure_testing_data()
    if False:
        ut.change_term_title('RUN IBEIS TESTS')
    exclude_doctests_fnames = set([
        '_autogen_explicit_controller',
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
    if ut.in_pyinstaller_package():
        from os.path import dirname
        dpath_list = [dirname(ibeis.__file__)]
        # Run tests for installer
        doctest_modname_list_ = [
            'ibeis.ibsfuncs',
            'ibeis.viz.interact.interact_matches',
            'ibeis.viz.interact.interact_annotations2',
            'ibeis.viz.interact.interact_name',
            'ibeis.viz.interact.interact_query_decision',
            'ibeis.viz.interact.interact_chip',
            'ibeis.viz.interact.interact_qres',
            'ibeis.algo.Config',
            'ibeis.algo.hots._pipeline_helpers',
            'ibeis.algo.hots.name_scoring',
            'ibeis.algo.hots.devcases',
            'ibeis.algo.hots.neighbor_index',
            'ibeis.algo.hots.automated_helpers',
            'ibeis.algo.hots.hots_query_result',
            'ibeis.algo.hots.automated_oracle',
            'ibeis.algo.hots.nn_weights',
            'ibeis.algo.hots.pipeline',
            'ibeis.algo.hots.automated_params',
            'ibeis.algo.hots.vsone_pipeline',
            'ibeis.algo.hots.automatch_suggestor',
            'ibeis.algo.hots.score_normalization',
            'ibeis.algo.hots.query_request',
            'ibeis.algo.hots.chip_match',
            'ibeis.algo.hots.multi_index',
            'ibeis.algo.hots.qt_inc_automatch',
            'ibeis.algo.hots.query_params',
            'ibeis.algo.hots.precision_recall',
            'ibeis.algo.hots.hstypes',
            'ibeis.algo.hots.match_chips4',
            'ibeis.algo.hots.distinctiveness_normalizer',
            'ibeis.algo.hots.automated_matcher',
            'ibeis.algo.hots.special_query',
            'ibeis.algo.hots.scoring',
            'ibeis.algo.preproc.preproc_annot',
            'ibeis.algo.preproc.preproc_imageset',
            'ibeis.algo.preproc.preproc_image',
            'ibeis.algo.preproc.preproc_residual',
            'ibeis.algo.detect.grabmodels',
            'ibeis.control.manual_annot_funcs',
            'ibeis.control.manual_chip_funcs',
            'ibeis.control.manual_species_funcs',
            'ibeis.control.manual_ibeiscontrol_funcs',
            'ibeis.control._autogen_party_funcs',
            'ibeis.control.manual_garelate_funcs',
            'ibeis.control.manual_name_funcs',
            'ibeis.control._sql_helpers',
            'ibeis.control.manual_wildbook_funcs',
            'ibeis.control.controller_inject',
            'ibeis.control.manual_lblimage_funcs',
            'ibeis.control.IBEISControl',
            'ibeis.control._autogen_featweight_funcs',
            'ibeis.control.manual_imageset_funcs',
            'ibeis.control.manual_feat_funcs',
            'ibeis.control.manual_gsgrelate_funcs',
            'ibeis.control._autogen_annotmatch_funcs',
            'ibeis.control.manual_meta_funcs',
            'ibeis.control.manual_lblannot_funcs',
            'ibeis.control.DB_SCHEMA',
            'ibeis.control.manual_lbltype_funcs',
            'ibeis.control.SQLDatabaseControl',
            'ibeis.control.manual_image_funcs',
            'ibeis.control.manual_annotgroup_funcs',
            'ibeis.control.DBCACHE_SCHEMA',
            'ibeis.init.main_helpers',
            'ibeis.init.sysres',
            'ibeis.gui.clock_offset_gui',
            'ibeis.dbio.export_subset',
            'ibeis.dbio.export_hsdb',
            'ibeis.dbio.ingest_database',
        ]
    else:
        dpath_list = ['ibeis']
        doctest_modname_list_ = ut.find_doctestable_modnames(dpath_list, exclude_doctests_fnames, exclude_dirs)

    exclude_doctest_pattern = ut.get_argval(('--exclude-doctest-patterns', '--x'), type_=list, default=[])
    if exclude_doctest_pattern is not None:
        import re
        is_ok = [all([re.search(pat, name) is None for pat in exclude_doctest_pattern])
                 for name in doctest_modname_list_]
        doctest_modname_list = ut.compress(doctest_modname_list_, is_ok)
    else:
        doctest_modname_list = doctest_modname_list_

    coverage = ut.get_argflag(('--coverage', '--cov',))
    if coverage:
        import coverage
        cov = coverage.Coverage(source=doctest_modname_list)
        cov.start()
        print('Starting coverage')

        exclude_lines = [
            'pragma: no cover',
            'def __repr__',
            'if self.debug:',
            'if settings.DEBUG',
            'raise AssertionError',
            'raise NotImplementedError',
            'if 0:',
            'if ut.VERBOSE',
            'if _debug:',
            'if __name__ == .__main__.:',
            'print(.*)',
        ]
        for line in exclude_lines:
            cov.exclude(line)

    doctest_modname_list2 = []
    for modname in doctest_modname_list:
        try:
            exec('import ' + modname, globals(), locals())
        except ImportError as ex:
            ut.printex(ex, iswarning=True)
            if not ut.in_pyinstaller_package():
                raise
        else:
            doctest_modname_list2.append(modname)

    module_list = [sys.modules[name] for name in doctest_modname_list2]

    nPass, nTotal, failed_cmd_list = ut.doctest_module_list(module_list)

    if coverage:
        print('Stoping coverage')
        cov.stop()
        print('Saving coverage')
        cov.save()
        print('Generating coverage html report')
        cov.html_report()

    if nPass != nTotal:
        return 1
    else:
        return 0

if __name__ == '__main__':
    """
    python -m ibeis --run-tests
    """
    import multiprocessing
    multiprocessing.freeze_support()
    run_tests()
