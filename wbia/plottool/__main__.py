#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function


def plottool_main():
    ignore_prefix = []
    ignore_suffix = []
    import utool as ut

    try:
        import wbia.plottool as pt  # NOQA
    except ImportError:
        raise
    # allows for --tf
    ut.main_function_tester('plottool', ignore_prefix, ignore_suffix)


if __name__ == '__main__':
    """
    python -m wbia.plottool --tf show_chipmatch2
    """
    print('Checking plottool main')
    plottool_main()
