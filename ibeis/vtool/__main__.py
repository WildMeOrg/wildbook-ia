#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function


def vtool_main():
    ignore_prefix = []
    ignore_suffix = []
    import utool as ut
    try:
        import vtool as vt  # NOQA
    except ImportError:
        raise
    ut.main_function_tester('vtool', ignore_prefix, ignore_suffix)

if __name__ == '__main__':
    """
    python -m vtool --tf stack_images
    """
    print('Checking vtool main')
    vtool_main()
