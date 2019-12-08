#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys

if __name__ == '__main__':
    from ibeis.tests import run_tests
    import multiprocessing
    multiprocessing.freeze_support()
    retcode = run_tests.run_tests()
    sys.exit(retcode)
