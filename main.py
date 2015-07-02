#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Runs IBIES gui

DEPRICATED

use ibeis.__main__.py instead

or more desirably

python -m ibeis

"""
from __future__ import absolute_import, division, print_function
import multiprocessing
from ibeis.__main__ import run_ibeis

if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    run_ibeis()
