# flake8: noqa
from __future__ import absolute_import, division, print_function
from .dev import params
from . import main_module
from .main_module import main, _preload, main_loop

__LOADED__ = False

def import_subs():
    global __LOADED__
    from . import dev
    from . import viz
    from . import control
    from . import model
    __LOADED__ = True

def ensure_subs():
    if not __LOADED__:
        import_subs()


def reload_subs():
    if not __LOADED__:
        import_subs()
    dev.reload_subs()
    viz.reload_subs()
    #control.reload_subs()
    model.reload_subs()
    #injest.reload_subs()
    #gui.reload_subs()
    #tests.reload_subs()
