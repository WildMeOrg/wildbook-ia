# -*- coding: utf-8 -*-
def custom_globals():
    import utool as ut

    ut.util_io.__PRINT_WRITES__ = False
    ut.util_io.__PRINT_READS__ = False
    ut.util_parallel.__FORCE_SERIAL__ = True
    ut.util_cache.VERBOSE_CACHE = False
    ut.NOT_QUIET = False

    import wbia.plottool as pt  # NOQA
    import matplotlib as mpl

    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['figure.titlesize'] = 20


def make_cells_wider():
    # Make notebook cells wider
    from IPython.core.display import HTML

    # This must be the last line in a cell
    return HTML('<style>body .container { width:99% !important; }</style>')
