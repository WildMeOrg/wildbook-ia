#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function

if __name__ == '__main__':
    from ibeis.tests import run_tests
    import multiprocessing
    multiprocessing.freeze_support()
    run_tests.run_tests()
