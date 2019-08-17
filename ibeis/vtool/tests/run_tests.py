#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import utool as ut


def run_tests():
    # Build module list and run tests
    import sys
    exclude_doctests_fnames = set([
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
    import vtool as vt
    from os.path import dirname
    #dpath_list = ['vtool']
    if ut.in_pyinstaller_package():
        # HACK, find_doctestable_modnames does not work in pyinstaller
        """
        import utool as ut
        import vtool as vt
        dpath_list = [dirname(vt.__file__)]
        doctest_modname_list = ut.find_doctestable_modnames(
            dpath_list, exclude_doctests_fnames, exclude_dirs)
        print(ut.indent('doctest_modname_list = ' + ut.repr2(doctest_modname_list), ' ' * 8))

        """
        doctest_modname_list = [
            'vtool.spatial_verification',
            'vtool.constrained_matching',
            'vtool.coverage_kpts',
            'vtool.image',
            'vtool.histogram',
            'vtool.chip',
            'vtool.distance',
            'vtool.coverage_grid',
            'vtool.linalg',
            'vtool.geometry',
            'vtool.other',
            'vtool.util_math',
            'vtool.score_normalization',
            'vtool.test_constrained_matching',
            'vtool.keypoint',
            'vtool.sver_c_wrapper',
            'vtool.quality_classifier',
            'vtool.features',
            'vtool.nearest_neighbors',
            'vtool.segmentation',
            'vtool.exif',
            'vtool.patch',
            'vtool.confusion',
            'vtool.blend',
            'vtool.clustering2',
            'vtool.matching',
        ]
    else:
        dpath_list = [dirname(vt.__file__)]
        doctest_modname_list = ut.find_doctestable_modnames(
            dpath_list, exclude_doctests_fnames, exclude_dirs)

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

    modname_list2 = []
    for modname in doctest_modname_list:
        try:
            exec('import ' + modname, globals(), locals())
        except ImportError as ex:
            ut.printex(ex)
            if not ut.in_pyinstaller_package():
                raise
        else:
            modname_list2.append(modname)

    if coverage:
        print('Stoping coverage')
        cov.stop()
        print('Saving coverage')
        cov.save()
        print('Generating coverage html report')
        cov.html_report()

    module_list = [sys.modules[name] for name in modname_list2]
    nPass, nTotal, failed_cmd_list = ut.doctest_module_list(module_list)
    if nPass != nTotal:
        return 1
    else:
        return 0

if __name__ == '__main__':
    import multiprocessing
    ut.change_term_title('RUN VTOOL TESTS')
    multiprocessing.freeze_support()
    retcode = run_tests()
    sys.exit(retcode)
