def custom_globals():
    import utool as ut
    ut.util_io.__PRINT_WRITES__ = False
    ut.util_io.__PRINT_READS__ = False
    ut.util_parallel.__FORCE_SERIAL__ = True
    ut.util_cache.VERBOSE_CACHE = False
    ut.NOT_QUIET = False

    import plottool as pt
    pt.custom_figure.TITLE_SIZE = 20
    pt.custom_figure.LABEL_SIZE = 20
    pt.custom_figure.FIGTITLE_SIZE = 20


def make_cells_wider():
    # Make notebook cells wider
    from IPython.core.display import HTML
    # This must be the last line in a cell
    return HTML('<style>body .container { width:99% !important; }</style>')
