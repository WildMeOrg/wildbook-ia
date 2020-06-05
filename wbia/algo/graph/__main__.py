#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def main():
    import ubelt as ub

    ub.doctest_package('wbia.algo.graph', ignore_patterns=['*_grave*'])

    # import utool as ut
    # ignore_prefix = [
    #     #'wbia.tests',
    #     'wbia.control.__SQLITE3__',
    #     '_autogen_explicit_controller']
    # ignore_suffix = ['_grave']
    # func_to_module_dict = {
    #     # 'demo_bayesnet': 'wbia.unstable.demobayes',
    # }
    # ut.main_function_tester('wbia.algo.graph', ignore_prefix, ignore_suffix,
    #                         func_to_module_dict=func_to_module_dict)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.graph
        python -m wbia.algo.graph --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    main()

    # import utool as ut  # NOQA
    # ut.doctest_funcs(),irjk
