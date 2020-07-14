# -*- coding: utf-8 -*-
#!/usr/bin/env python  # NOQA
from __future__ import absolute_import, division, print_function
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def ensure_testing_data():
    from wbia.tests import reset_testdbs

    print('Making sure test data exists')
    import wbia
    from os.path import join

    ut.change_term_title('ENSURE IBEIS TETSDATA')
    reset_testdbs.reset_testdbs()
    workdir = wbia.get_workdir()
    if not ut.checkpath(join(workdir, 'PZ_MTEST')):
        wbia.ensure_pz_mtest()
    if not ut.checkpath(join(workdir, 'NAUT_test')):
        wbia.ensure_nauts()
    if not ut.checkpath(join(workdir, 'wd_peter2')):
        wbia.ensure_wilddogs()


def dynamic_doctest_modnames():
    r"""
    CommandLine:
        python -m wbia.tests.run_tests dynamic_doctest_modnames --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.tests.run_tests import *  # NOQA
        >>> doctest_modname_list_ = dynamic_doctest_modnames()
        >>> result = ('doctest_modname_list_ = %s' % (ut.repr3(doctest_modname_list_),))
        >>> print(result)
    """
    exclude_doctests_fnames = set(
        [
            '_autogen_explicit_controller',
            'template_definitions.py',
            'autogen_test_script.py',
        ]
    )
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
    dpath_list = ['wbia']
    doctest_modname_list_ = ut.find_doctestable_modnames(
        dpath_list, exclude_doctests_fnames, exclude_dirs
    )
    return doctest_modname_list_


def static_doctest_modnames():
    doctest_modname_list_ = [
        'wbia',
        'wbia.annots',
        'wbia.core_annots',
        'wbia.main_module',
        'wbia.new_annots',
        'wbia.tag_funcs',
        'wbia.core_images',
        'wbia.annotmatch_funcs',
        'wbia.images',
        'wbia.viz.viz_graph2',
        'wbia.viz.viz_helpers',
        'wbia.viz.viz_hough',
        'wbia.viz.viz_chip',
        'wbia.viz.viz_image',
        'wbia.viz.viz_name',
        'wbia.viz.viz_matches',
        'wbia.viz.viz_graph2',
        'wbia.viz.viz_sver',
        'wbia.viz.viz_other',
        'wbia.viz.viz_nearest_descriptors',
        'wbia.viz.viz_qres',
        'wbia.viz.interact.interact_matches',
        'wbia.viz.interact.interact_annotations2',
        'wbia.viz.interact.interact_name',
        'wbia.viz.interact.interact_query_decision',
        'wbia.viz.interact.interact_chip',
        'wbia.viz.interact.interact_qres',
        'wbia.templates.generate_notebook',
        'wbia.scripts.classify_shark',
        'wbia.scripts.specialdraw',
        'wbia.scripts.gen_cand_expts',
        'wbia.control.manual_annot_funcs',
        'wbia.control.manual_chip_funcs',
        'wbia.control.manual_species_funcs',
        'wbia.control.manual_wbiacontrol_funcs',
        'wbia.control._autogen_party_funcs',
        'wbia.control.manual_garelate_funcs',
        'wbia.control.manual_name_funcs',
        'wbia.control.accessor_decors',
        'wbia.control._sql_helpers',
        'wbia.control.manual_wildbook_funcs',
        'wbia.control.controller_inject',
        'wbia.control.manual_lblimage_funcs',
        'wbia.control.IBEISControl',
        'wbia.control.manual_feat_funcs',
        'wbia.control.wildbook_manager',
        'wbia.control.manual_annotmatch_funcs',
        'wbia.control.manual_gsgrelate_funcs',
        'wbia.control.manual_meta_funcs',
        'wbia.control.manual_lblannot_funcs',
        'wbia.control.manual_featweight_funcs',
        'wbia.control.DB_SCHEMA',
        'wbia.control.manual_lbltype_funcs',
        'wbia.control.manual_image_funcs',
        'wbia.control.manual_imageset_funcs',
        'wbia.control.manual_annotgroup_funcs',
        'wbia.algo.Config',
        'wbia.unstable.demobayes',
        'wbia.algo.hots._pipeline_helpers',
        'wbia.algo.hots.name_scoring',
        'wbia.algo.hots.devcases',
        'wbia.algo.hots.neighbor_index',
        'wbia.algo.hots.pgm_viz',
        'wbia.algo.hots.pgm_ext',
        'wbia.algo.hots.bayes',
        'wbia.algo.hots.nn_weights',
        'wbia.algo.hots.pipeline',
        'wbia.algo.hots.orig_graph_iden',
        'wbia.algo.hots.query_request',
        'wbia.algo.hots.chip_match',
        'wbia.algo.hots.multi_index',
        'wbia.algo.hots.testem',
        'wbia.algo.hots.query_params',
        'wbia.algo.hots.precision_recall',
        'wbia.algo.hots.hstypes',
        'wbia.algo.hots.match_chips4',
        'wbia.algo.hots.neighbor_index_cache',
        'wbia.algo.graph.core',
        'wbia.algo.hots.scoring',
        'wbia.algo.preproc.preproc_annot',
        'wbia.algo.preproc.preproc_occurrence',
        'wbia.algo.preproc.preproc_image',
        'wbia.algo.preproc.preproc_residual',
        'wbia.algo.detect.grabmodels',
        'wbia.other.dbinfo',
        'wbia.other.ibsfuncs',
        'wbia.other.detectfuncs',
        'wbia.other.detectcore',
        'wbia.other.detectgrave',
        'wbia.other.detecttrain',
        'wbia.init.main_helpers',
        'wbia.init.filter_annots',
        'wbia.init.sysres',
        'wbia.gui.guimenus',
        'wbia.gui.guiback',
        'wbia.gui.inspect_gui',
        'wbia.gui.newgui',
        'wbia.gui.dbfix_widget',
        'wbia.gui.clock_offset_gui',
        'wbia.dbio.export_subset',
        'wbia.dbio.ingest_hsdb',
        'wbia.dbio.export_hsdb',
        'wbia.dbio.ingest_database',
        'wbia.expt.harness',
        'wbia.expt.old_storage',
        'wbia.expt.experiment_helpers',
        'wbia.expt.annotation_configs',
        'wbia.expt.experiment_drawing',
        'wbia.expt.experiment_printres',
        'wbia.expt.test_result',
        'wbia.expt.cfghelpers',
        'wbia.web.routes_ajax',
        'wbia.web.routes',
        'wbia.web.apis_query',
        'wbia.web.app',
        'wbia.web.job_engine',
        'wbia.web.apis_json',
        'wbia.web.apis_detect',
        'wbia.web.apis_engine',
        'wbia.web.test_api',
        'wbia.web.apis',
        'wbia.web.routes_csv',
        'wbia.web.routes_submit',
    ]
    return doctest_modname_list_


def run_tests():
    """
    >>> from wbia.tests.run_tests import *  # NOQA
    """
    # starts logging for tests
    import wbia

    wbia._preload()
    # Build module list and run tests
    import sys

    if True:
        ensure_testing_data()

    if ut.in_pyinstaller_package():
        # Run tests for installer
        doctest_modname_list_ = static_doctest_modnames()
    else:
        doctest_modname_list_ = dynamic_doctest_modnames()

    exclude_doctest_pattern = ut.get_argval(
        ('--exclude-doctest-patterns', '--x'), type_=list, default=[]
    )
    if exclude_doctest_pattern is not None:
        import re

        is_ok = [
            all([re.search(pat, name) is None for pat in exclude_doctest_pattern])
            for name in doctest_modname_list_
        ]
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
        import wbia.guitool  # NOQA
    except ImportError:
        HAVE_GUI = False
    else:
        HAVE_GUI = True

    # Remove gui things if possible
    import re

    if not HAVE_GUI:
        doctest_modname_list = [
            modname
            for modname in doctest_modname_list_
            if not re.search('\\bgui\\b', modname) and not re.search('\\bviz\\b', modname)
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
        convert_tests_from_wbia_to_nose(module_list)
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


def convert_tests_from_wbia_to_nose(module_list):
    # PARSE OUT TESTABLE DOCTESTTUPS
    # import utool as ut
    testtup_list = []
    seen_ = set()

    topimport_list = []

    for module in module_list:
        mod_doctest_tup = ut.get_module_doctest_tup(
            module=module, verbose=False, allexamples=True
        )
        enabled_testtup_list, frame_fpath, all_testflags, module = mod_doctest_tup
        flags = [tup.src not in seen_ for tup in enabled_testtup_list]
        enabled_testtup_list = ut.compress(enabled_testtup_list, flags)
        testtup_list.extend(enabled_testtup_list)
        if len(enabled_testtup_list) > 0:
            topimport_list.append('from %s import *  # NOQA' % (module.__name__,))

    print('Found %d test tups' % (len(testtup_list)))

    autogen_test_src_funcs = []
    # import redbaron
    for testtup in testtup_list:
        name = testtup.name
        num = testtup.num
        src = testtup.src
        want = testtup.want
        import re

        src = re.sub('# ENABLE_DOCTEST\n', '', src)
        src = re.sub(r'from [^*]* import \* *# NOQA\n', '', src)
        src = re.sub(r'from [^*]* import \*\n', '', src)

        src = ut.str_between(src, None, 'ut.quit_if_noshow').rstrip('\n')
        src = ut.str_between(src, None, 'ut.show_if_requested').rstrip('\n')
        # import utool
        # utool.embed()
        """
        """
        # flag = testtup.flag
        if want.endswith('\n'):
            want = want[:-1]
        if want:
            # src_node = redbaron.RedBaron(src)
            # if len(src_node.find_all('name', 'result')) > 0:
            #    src_node.append('assert result == %r' % (want,))
            if '\nresult = ' in src:
                src += '\nassert str(result) == %r' % (want,)
        func_src = 'def test_%s_%d():\n' % (name.replace('.', '_'), num,) + ut.indent(src)
        autogen_test_src_funcs.append(func_src)

    autogen_test_src = (
        '\n'.join(topimport_list)
        + '\n\n\n'
        + '\n\n\n'.join(autogen_test_src_funcs)
        + '\n'
    )
    from wbia import tests
    from os.path import join

    moddir = ut.get_module_dir(tests)
    ut.writeto(join(moddir, 'test_autogen_nose_tests.py'), autogen_test_src)


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m wbia.tests.run_tests
    """
    import xdoctest

    xdoctest.doctest_module(__file__)
