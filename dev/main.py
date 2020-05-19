#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runs IBIES gui

Pyinstaller entry point
When running from non-pyinstaller source use
    python -m ibeis
    which is equivalent to
    python ibeis/__main__.py

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ibeis.__main__ import run_ibeis


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    run_ibeis()
