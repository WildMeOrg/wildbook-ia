# -*- coding: utf-8 -*-
#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


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


def dynamic_doctest_modnames():
    r"""
    CommandLine:
        python -m ibeis.tests.run_tests dynamic_doctest_modnames --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.tests.run_tests import *  # NOQA
        >>> doctest_modname_list_ = dynamic_doctest_modnames()
        >>> result = ('doctest_modname_list_ = %s' % (ut.repr3(doctest_modname_list_),))
        >>> print(result)
    """
    exclude_doctests_fnames = set([
        '_autogen_explicit_controller',
        'template_definitions.py',
        'autogen_test_script.py',
    ])
    exclude_dirs = ['_broken', 'old', 'tests', 'timeits', '_scripts',
                    '_timeits', '_doc', 'notebook', ]
    dpath_list = ['ibeis']
    doctest_modname_list_ = ut.find_doctestable_modnames(dpath_list, exclude_doctests_fnames, exclude_dirs)
    return doctest_modname_list_


def static_doctest_modnames():
    doctest_modname_list_ = [
        'ibeis',
        'ibeis.annots', 'ibeis.core_annots', 'ibeis.main_module',
        'ibeis.new_annots', 'ibeis.tag_funcs', 'ibeis.core_images',
        'ibeis.annotmatch_funcs', 'ibeis.images',
        'ibeis.viz.viz_graph2', 'ibeis.viz.viz_helpers', 'ibeis.viz.viz_hough',
        'ibeis.viz.viz_chip', 'ibeis.viz.viz_image', 'ibeis.viz.viz_name',
        'ibeis.viz.viz_matches', 'ibeis.viz.viz_graph2', 'ibeis.viz.viz_sver',
        'ibeis.viz.viz_other', 'ibeis.viz.viz_nearest_descriptors',
        'ibeis.viz.viz_qres', 'ibeis.viz.interact.interact_matches',
        'ibeis.viz.interact.interact_annotations2',
        'ibeis.viz.interact.interact_name',
        'ibeis.viz.interact.interact_query_decision',
        'ibeis.viz.interact.interact_chip', 'ibeis.viz.interact.interact_qres',
        'ibeis.templates.generate_notebook',
        'ibeis.scripts.classify_shark',
        'ibeis.scripts.specialdraw',
        'ibeis.scripts.gen_cand_expts',
        'ibeis.control.manual_annot_funcs', 'ibeis.control.manual_chip_funcs',
        'ibeis.control.manual_species_funcs',
        'ibeis.control.manual_ibeiscontrol_funcs',
        'ibeis.control._autogen_party_funcs',
        'ibeis.control.manual_garelate_funcs',
        'ibeis.control.manual_name_funcs', 'ibeis.control.accessor_decors',
        'ibeis.control._sql_helpers', 'ibeis.control.manual_wildbook_funcs',
        'ibeis.control.controller_inject',
        'ibeis.control.manual_lblimage_funcs', 'ibeis.control.IBEISControl',
        'ibeis.control.manual_feat_funcs', 'ibeis.control.wildbook_manager',
        'ibeis.control.manual_annotmatch_funcs',
        'ibeis.control.manual_gsgrelate_funcs',
        'ibeis.control.manual_meta_funcs',
        'ibeis.control.manual_lblannot_funcs',
        'ibeis.control.manual_featweight_funcs', 'ibeis.control.DB_SCHEMA',
        'ibeis.control.manual_lbltype_funcs',
        'ibeis.control.manual_image_funcs',
        'ibeis.control.manual_imageset_funcs',
        'ibeis.control.manual_annotgroup_funcs',
        'ibeis.algo.Config',
        'ibeis.unstable.demobayes', 'ibeis.algo.hots._pipeline_helpers',
        'ibeis.algo.hots.name_scoring', 'ibeis.algo.hots.devcases',
        'ibeis.algo.hots.neighbor_index', 'ibeis.algo.hots.pgm_viz',
        'ibeis.algo.hots.pgm_ext', 'ibeis.algo.hots.bayes',
        'ibeis.algo.hots.nn_weights',
        'ibeis.algo.hots.pipeline',
        'ibeis.algo.hots.orig_graph_iden', 'ibeis.algo.hots.query_request',
        'ibeis.algo.hots.chip_match', 'ibeis.algo.hots.multi_index',
        'ibeis.algo.hots.testem', 'ibeis.algo.hots.query_params',
        'ibeis.algo.hots.precision_recall', 'ibeis.algo.hots.hstypes',
        'ibeis.algo.hots.match_chips4', 'ibeis.algo.hots.neighbor_index_cache',
        'ibeis.algo.graph.core',
        'ibeis.algo.hots.scoring',
        'ibeis.algo.preproc.preproc_annot',
        'ibeis.algo.preproc.preproc_occurrence',
        'ibeis.algo.preproc.preproc_image',
        'ibeis.algo.preproc.preproc_residual', 'ibeis.algo.detect.grabmodels',
        'ibeis.other.dbinfo', 'ibeis.other.ibsfuncs',
        'ibeis.other.detectfuncs',
        'ibeis.other.detectcore',
        'ibeis.other.detectgrave',
        'ibeis.other.detecttrain',
        'ibeis.init.main_helpers', 'ibeis.init.filter_annots',
        'ibeis.init.sysres',
        'ibeis.gui.guimenus', 'ibeis.gui.guiback', 'ibeis.gui.inspect_gui',
        'ibeis.gui.newgui', 'ibeis.gui.dbfix_widget',
        'ibeis.gui.clock_offset_gui',
        'ibeis.dbio.export_subset', 'ibeis.dbio.ingest_hsdb',
        'ibeis.dbio.export_hsdb', 'ibeis.dbio.ingest_database',
        'ibeis.expt.harness', 'ibeis.expt.old_storage',
        'ibeis.expt.experiment_helpers', 'ibeis.expt.annotation_configs',
        'ibeis.expt.experiment_drawing', 'ibeis.expt.experiment_printres',
        'ibeis.expt.test_result', 'ibeis.expt.cfghelpers',
        'ibeis.web.routes_ajax', 'ibeis.web.routes', 'ibeis.web.apis_query',
        'ibeis.web.app', 'ibeis.web.job_engine', 'ibeis.web.apis_json',
        'ibeis.web.apis_detect', 'ibeis.web.apis_engine', 'ibeis.web.test_api',
        'ibeis.web.apis', 'ibeis.web.routes_csv', 'ibeis.web.routes_submit',
    ]
    return doctest_modname_list_


def run_tests():
    """
    >>> from ibeis.tests.run_tests import *  # NOQA
    """
    # starts logging for tests
    import ibeis

    ibeis._preload()
    # Build module list and run tests
    import sys
    if True:
        ensure_testing_data()

    if ut.in_pyinstaller_package():
        # Run tests for installer
        doctest_modname_list_ = static_doctest_modnames()
    else:
        doctest_modname_list_ = dynamic_doctest_modnames()

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

    try:
        import guitool  # NOQA
    except ImportError:
        HAVE_GUI = False
    else:
        HAVE_GUI = True

    # Remove gui things if possible
    import re
    if not HAVE_GUI:
        doctest_modname_list = [
            modname for modname in doctest_modname_list_ if
            not re.search('\\bgui\\b', modname) and
            not re.search('\\bviz\\b', modname)
        ]

    for modname in doctest_modname_list:
        try:
            exec('import ' + modname, globals(), locals())
        except ImportError as ex:
            ut.printex(ex, iswarning=True)
            # import parse
            # if not HAVE_GUI:
            #     try:
            #         parsed = parse.parse('No module named {}', str(ex))
            #         if parsed is None:
            #             parsed = parse.parse('cannot import name {}', str(ex))
            #         if parsed is not None:
            #             if parsed[0].endswith('_gui'):
            #                 print('skipping gui module %r' % (parsed[0],))
            #                 continue
            #             if parsed[0].startswith('viz_'):
            #                 print('skipping viz module %r' % (parsed[0],))
            #                 continue
            #             if parsed[0].startswith('interact_'):
            #                 print('skipping interact module %r' % (parsed[0],))
            #                 continue
            #             # if parsed[0] in ['sip']:
            #             #     print('skipping Qt module %r' % (parsed[0],))
            #             #     continue
            #     except Exception:
            #         pass
            if not ut.in_pyinstaller_package():
                raise
        else:
            doctest_modname_list2.append(modname)

    module_list = [sys.modules[name] for name in doctest_modname_list2]

    # Write to py.test / nose format
    if ut.get_argflag('--tonose'):
        convert_tests_from_ibeis_to_nose(module_list)
        return 0

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


def convert_tests_from_ibeis_to_nose(module_list):
    # PARSE OUT TESTABLE DOCTESTTUPS
    #import utool as ut
    testtup_list = []
    seen_ = set()

    topimport_list = []

    for module in module_list:
        mod_doctest_tup = ut.get_module_doctest_tup(module=module,
                                                    verbose=False,
                                                    allexamples=True)
        enabled_testtup_list, frame_fpath, all_testflags, module = mod_doctest_tup
        flags = [tup.src not in seen_ for tup in enabled_testtup_list]
        enabled_testtup_list = ut.compress(enabled_testtup_list, flags)
        testtup_list.extend(enabled_testtup_list)
        if len(enabled_testtup_list) > 0:
            topimport_list.append('from %s import *  # NOQA' % (module.__name__,))

    print('Found %d test tups' % (len(testtup_list)))

    autogen_test_src_funcs = []
    #import redbaron
    for testtup in testtup_list:
        name = testtup.name
        num  = testtup.num
        src  = testtup.src
        want = testtup.want
        import re
        src = re.sub('# ENABLE_DOCTEST\n', '', src)
        src = re.sub('from [^*]* import \* *# NOQA\n', '', src)
        src = re.sub('from [^*]* import \*\n', '', src)

        src = ut.str_between(src, None, 'ut.quit_if_noshow').rstrip('\n')
        src = ut.str_between(src, None, 'ut.show_if_requested').rstrip('\n')
        # import utool
        # utool.embed()
        """
        """
        #flag = testtup.flag
        if want.endswith('\n'):
            want = want[:-1]
        if want:
            #src_node = redbaron.RedBaron(src)
            #if len(src_node.find_all('name', 'result')) > 0:
            #    src_node.append('assert result == %r' % (want,))
            if '\nresult = ' in src:
                src += '\nassert str(result) == %r' % (want,)
        func_src = 'def test_%s_%d():\n' % (name.replace('.', '_'), num,) + ut.indent(src)
        autogen_test_src_funcs.append(func_src)

    autogen_test_src = '\n'.join(topimport_list) + '\n\n\n' + '\n\n\n'.join(autogen_test_src_funcs) + '\n'
    from ibeis import tests
    from os.path import join
    moddir = ut.get_module_dir(tests)
    ut.writeto(join(moddir, 'test_autogen_nose_tests.py'), autogen_test_src)


if __name__ == '__main__':
    """
    Run the unit tests for IBEIS

    Commandline usage: python -m ibeis.tests.run_tests

    """
    import multiprocessing
    multiprocessing.freeze_support()
    run_tests()
