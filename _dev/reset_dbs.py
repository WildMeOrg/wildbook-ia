#!/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    """
    SeeAlso:
        ~/code/wbia/wbia/tests/reset_testdbs.py
    """
    import wbia
    from wbia.tests import reset_testdbs

    wbia.ENABLE_WILDBOOK_SIGNAL = False
    reset_testdbs.reset_testdbs()
