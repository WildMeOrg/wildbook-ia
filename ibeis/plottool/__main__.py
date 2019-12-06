#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function


def plottool_ibeis_main():
    ignore_prefix = []
    ignore_suffix = []
    import utool as ut
    try:
        import plottool_ibeis as pt  # NOQA
    except ImportError:
        raise
    # allows for --tf
    ut.main_function_tester('plottool_ibeis', ignore_prefix, ignore_suffix)

if __name__ == '__main__':
    """
    python -m plottool_ibeis --tf show_chipmatch2
    """
    print('Checking plottool_ibeis main')
    plottool_ibeis_main()
