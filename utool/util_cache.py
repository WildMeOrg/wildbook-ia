from __future__ import absolute_import, division, print_function
import shelve
import cPickle
import atexit
from os.path import join, normpath
from . import util_inject
from . import util_hash
from . import util_path
from . import util_str
from . import util_cplat
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[cache]')


__PRINT_IO__ = True
__PRINT_WRITES__ = False or __PRINT_IO__
__PRINT_READS__  = False or __PRINT_IO__
__SHELF__ = None  # GLOBAL CACHE


def read_from(fpath):
    if __PRINT_READS__:
        print('[cache] * Reading text file: %r ' % util_path.tail(fpath))
    try:
        if not util_path.checkpath(fpath, verbose=__PRINT_READS__):
            raise IOError('[cache] * FILE DOES NOT EXIST!')
        with open(fpath, 'r') as file_:
            text = file_.read()
    except IOError as ex:
        print('!!!!!!!')
        print('IOError: %s' % (ex,))
        print('[cache] * Error reading fpath=%r' % util_path.tail(fpath))
        raise
    return text


def write_to(fpath, to_write):
    if __PRINT_WRITES__:
        print('[cache] * Writing to text file: %r ' % util_path.tail(fpath))
    with open(fpath, 'w') as file_:
        file_.write(to_write)


def text_dict_write(fpath, key, val):
    """
    Very naive, but readable way of storing a dictionary on disk
    """
    try:
        dict_text = read_from(fpath)
    except IOError:
        dict_text = '{}'
    dict_ = eval(dict_text)
    dict_[key] = val
    dict_text2 = util_str.dict_str(dict_, strvals=False)
    print(dict_text2)
    write_to(fpath, dict_text2)


def _args2_fpath(dpath, fname, uid, ext, write_hashtbl=False):
    """
    Ensures that the filename is not too long (looking at you windows)
    Windows MAX_PATH=260 characters
    Absolute length is limited to 32,000 characters
    Each filename component is limited to 255 characters

    if write_hashtbl is True, hashed values expaneded and written to a text file
    """
    if len(ext) > 0 and ext[0] != '.':
        raise Exception('Fatal Error: Please be explicit and use a dot in ext')
    fname_uid = fname + uid
    if len(fname_uid) > 128:
        hashed_uid = util_hash.hashstr(uid, 8)
        if write_hashtbl:
            text_dict_write(join(dpath, 'hashtbl.txt'), hashed_uid, uid)
        fname_uid = fname + '_' + hashed_uid
    fpath = join(dpath, fname_uid + ext)
    fpath = normpath(fpath)
    return fpath


def save_cache(dpath, fname, uid, data):
    fpath = _args2_fpath(dpath, fname, uid, '.cPkl', write_hashtbl=True)
    save_cPkl(fpath, data)


def load_cache(dpath, fname, uid):
    fpath = _args2_fpath(dpath, fname, uid, '.cPkl')
    return load_cPkl(fpath)


#------------------------------
# TODO: Split into utool.util_io

def save_cPkl(fpath, data):
    # TODO: Split into utool.util_io
    if __PRINT_WRITES__:
        print('[cache] * save_cPkl(%r, data)' % (util_path.tail(fpath),))
    with open(fpath, 'wb') as file_:
        cPickle.dump(data, file_, cPickle.HIGHEST_PROTOCOL)


def load_cPkl(fpath):
    # TODO: Split into utool.util_io
    if __PRINT_READS__:
        print('[cache] * load_cPkl(%r, data)' % (util_path.tail(fpath),))
    with open(fpath, 'rb') as file_:
        data = cPickle.load(file_)
    return data


# --- Global Cache ---

def get_global_cache_dir(appname='utool', ensure=False):
    """
        Returns a cache directory for an application in a directory oked by the
        operating system: IE:
            '~/.config', '~/AppData/Roaming', or '~/Library/Application Support'
    """
    # TODO: Make a decoupled way to set the application name
    os_resource_dpath = util_cplat.get_resource_dir()
    project_cache_dname = '%s_cache' % appname
    global_cache_dir = normpath(join(os_resource_dpath, project_cache_dname))
    if ensure:
        util_path.ensuredir(global_cache_dir)
    return global_cache_dir


def get_global_shelf_fpath(appname=None, ensure=False):
    """ Returns the filepath to the global shelf """
    global_cache_dir = get_global_cache_dir(appname, ensure=ensure)
    shelf_fpath = join(global_cache_dir, 'global_cache.shelf')
    return shelf_fpath


def get_global_shelf(appname=None):
    """ Returns the global shelf object """
    global __SHELF__
    if __SHELF__ is None:
        try:
            shelf_fpath = get_global_shelf_fpath(appname, ensure=True)
            __SHELF__ = shelve.open(shelf_fpath)
        except Exception as ex:
            from . import util_dbg
            util_dbg.printex(ex, 'Failed opening shelf_fpath',
                             key_list=['shelf_fpath'])
            raise
        #shelf_file = open(shelf_fpath, 'w')
    return __SHELF__


def close_global_shelf(appname=None):
    global __SHELF__
    if __SHELF__ is not None:
        __SHELF__.close()
    __SHELF__ = None


def global_cache_read(key, appname=None, **kwargs):
    shelf = get_global_shelf(appname)
    if 'default' in kwargs:
        return shelf.get(key, kwargs['default'])
    else:
        return shelf[key]


def global_cache_dump(appname=None):
    shelf_fpath = get_global_shelf_fpath(appname)
    shelf = get_global_shelf(appname)
    print('shelf_fpath = %r' % shelf_fpath)
    print(util_str.dict_str(shelf))


def global_cache_write(key, val, appname=None):
    """ Writes cache files to a safe place in each operating system """
    shelf = get_global_shelf(appname)
    shelf[key] = val


def delete_global_cache(appname=None):
    """ Reads cache files to a safe place in each operating system """
    close_global_shelf(appname)
    shelf_fpath = get_global_shelf_fpath(appname)
    util_path.remove_file(shelf_fpath, verbose=True, dryrun=False)


atexit.register(close_global_shelf)  # ensure proper cleanup when exiting python

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
