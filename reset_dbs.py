#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

if __name__ == '__main__':
    """
    SeeAlso:
        ~/code/ibeis/ibeis/tests/reset_testdbs.py
    """
    import ibeis
    from ibeis.tests import reset_testdbs
    ibeis.ENABLE_WILDBOOK_SIGNAL = False
    reset_testdbs.reset_testdbs()
