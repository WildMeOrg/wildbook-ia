from __future__ import division, print_function
import drawtool.draw_func2 as df2
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_helpers]', DEBUG=False)


def draw():
    df2.adjust_subplots_safe()
    df2.draw()


def get_ibsdat(ax, key, default=None):
    """ returns internal IBEIS property from a matplotlib axis """
    _ibsdat = ax.__dict__.get('_ibsdat', None)
    if _ibsdat is None:
        return default
    val = _ibsdat.get(key, default)
    return val


def set_ibsdat(ax, key, val):
    """ sets internal IBEIS property to a matplotlib axis """
    if not '_ibsdat' in ax.__dict__:
        ax.__dict__['_ibsdat'] = {}
    _ibsdat = ax.__dict__['_ibsdat']
    _ibsdat[key] = val
