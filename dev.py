#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dev.py was moved to ibeis/dev.py
Now runnable via python -m ibeis.dev
"""
from __future__ import absolute_import, division, print_function
from ibeis.dev import *  # NOQA

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    devmain()
