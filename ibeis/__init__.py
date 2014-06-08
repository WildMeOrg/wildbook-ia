# flake8: noqa
from __future__ import absolute_import, division, print_function
from . import constants
from . import main_module
from .dev import params
from .dev import sysres
from .main_module import main, _preload, main_loop
from .control.IBEISControl import IBEISController
from .dev.sysres import get_workdir, set_workdir

__LOADED__ = False
__version__ = '0.1.0.dev1'

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
    #ingest.reload_subs()
    #gui.reload_subs()
    #tests.reload_subs()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
