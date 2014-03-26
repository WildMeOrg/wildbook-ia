from __future__ import division, print_function
import shelve
from os.path import join, normpath
from .util_inject import inject
from . import util_path
from . import util_cplat
import pylru  # because we dont have functools.lru_cache
print, print_, printDBG, rrr, profile = inject(__name__, '[cache]')


__PRINT_WRITES__ = False
__PRINT_READS__ = False

__SHELF__ = None  # GLOBAL CACHE


class lru_cache(object):
    # because we dont have functools.lru_cache
    def __init__(cache, max_size=100):
        cache.max_size = max_size
        cache.cache_ = pylru.lrucache(max_size)
        cache.func_name = None

    def clear_cache(cache):
        printDBG('[cache.lru] clearing %r lru_cache' % (cache.func_name,))
        cache.cache_.clear()

    def __call__(cache, func):
        def wrapped(self, *args):  # wrap a class
            try:
                value = cache.cache_[args]
                printDBG(func.func_name + '(%r) ...lrucache hit' % (args,))
                return value
            except KeyError:
                printDBG(func.func_name + '(%r) ...lrucache miss' % (args,))

            value = func(self, *args)
            cache.cache_[args] = value
            return value
        cache.func_name = func.func_name
        printDBG('[@tools.lru] wrapping %r with max_size=%r lru_cache' %
                 (cache.func_name, cache.max_size))
        wrapped.func_name = func.func_name
        wrapped.clear_cache = cache.clear_cache
        return wrapped


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

def get_global_cache_dir(projectname='ibeis', ensure=False):
    os_resource_dpath = util_cplat.get_resource_dir()
    project_cache_dname = '%s_cache' % projectname
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


def global_cache_read(key, default=None, **kwargs):
    shelf = get_global_shelf()
    if default is None:
        return shelf[key]
    else:
        return shelf.get(key, default)


def global_cache_write(key, val):
    shelf = get_global_shelf()
    shelf[key] = val


def delete_global_cache():
    close_global_shelf()
    shelf_fpath = get_global_shelf_fpath()
    util_path.remove_file(shelf_fpath, verbose=True, dryrun=False)
