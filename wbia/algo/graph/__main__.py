#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')


def main():
    import ubelt as ub

    ub.doctest_package('wbia.algo.graph', ignore_patterns=['*_grave*'])


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
