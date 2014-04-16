from __future__ import absolute_import, division, print_function
import shelve
import atexit
from os.path import join, normpath
from .util_inject import inject
from . import util_path
from . import util_cplat
print, print_, printDBG, rrr, profile = inject(__name__, '[cache]')


__PRINT_WRITES__ = False
__PRINT_READS__ = False

__SHELF__ = None  # GLOBAL CACHE


def read_from(fpath):
    verbose = __PRINT_READS__
    if verbose:
        print('[util] * Reading text file: %r ' % util_path.split(fpath)[1])
    try:
        if not util_path.checkpath(fpath, verbose=verbose):
            raise IOError('[util] * FILE DOES NOT EXIST!')
        with open(fpath, 'r') as file_:
            text = file_.read()
    except IOError as ex:
        print('!!!!!!!')
        print('IOError: %s' % (ex,))
        print('[util] * Error reading fpath=%r' % fpath)
        raise
    return text


def write_to(fpath, to_write):
    if __PRINT_WRITES__:
        print('[util] * Writing to text file: %r ' % fpath)
    with open(fpath, 'w') as file_:
        file_.write(to_write)


# --- Global Cache ---

def get_global_cache_dir(appname='ibeis', ensure=False):
    """
        Returns a cache directory for a project in the correct location for each
        operating system: IE: '~/.config, '~/AppData/Roaming', or
        '~/Library/Application Support'
    """
    # TODO: Make a decoupled way to set the application name
    os_resource_dpath = util_cplat.get_resource_dir()
    project_cache_dname = '%s_cache' % appname
    global_cache_dir = normpath(join(os_resource_dpath, project_cache_dname))
    util_path.ensuredir(global_cache_dir)
    return global_cache_dir


def get_global_shelf_fpath(**kwargs):
    global_cache_dir = get_global_cache_dir(ensure=True, **kwargs)
    shelf_fpath = join(global_cache_dir, 'global_cache.shelf')
    return shelf_fpath


def get_global_shelf(**kwargs):
    global __SHELF__
    if __SHELF__ is None:
        try:
            shelf_fpath = get_global_shelf_fpath(**kwargs)
            __SHELF__ = shelve.open(shelf_fpath)
        except Exception as ex:
            print('!!!')
            print('[util_cache] Failed opening: shelf_fpath=%r' % shelf_fpath)
            print('[util_cache] Caught: %s: %s' % (type(ex), ex))
            raise
        #shelf_file = open(shelf_fpath, 'w')
    return __SHELF__


def close_global_shelf(**kwargs):
    global __SHELF__
    if __SHELF__ is not None:
        __SHELF__.close()
    __SHELF__ = None


def global_cache_read(key, **kwargs):
    shelf = get_global_shelf()
    if not 'default' in kwargs:
        return shelf[key]
    else:
        return shelf.get(key, kwargs['default'])


def global_cache_write(key, val):
    """ Writes cache files to a safe place in each operating system """
    shelf = get_global_shelf()
    shelf[key] = val


def delete_global_cache():
    """ Reads cache files to a safe place in each operating system """
    close_global_shelf()
    shelf_fpath = get_global_shelf_fpath()
    util_path.remove_file(shelf_fpath, verbose=True, dryrun=False)


atexit.register(close_global_shelf)  # ensure proper cleanup when exiting python
