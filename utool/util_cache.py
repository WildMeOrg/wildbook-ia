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
    verbose = __PRINT_READS__
    if verbose:
        print('[cache] * Reading text file: %r ' % util_path.split(fpath)[1])
    try:
        if not util_path.checkpath(fpath, verbose=verbose):
            raise IOError('[cache] * FILE DOES NOT EXIST!')
        with open(fpath, 'r') as file_:
            text = file_.read()
    except IOError as ex:
        print('!!!!!!!')
        print('IOError: %s' % (ex,))
        print('[cache] * Error reading fpath=%r' % fpath)
        raise
    return text


def write_to(fpath, to_write):
    if __PRINT_WRITES__:
        print('[cache] * Writing to text file: %r ' % fpath)
    with open(fpath, 'w') as file_:
        file_.write(to_write)


def text_dict_write(fpath, key, val):
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


def save_cPkl(fpath, data):
    if __PRINT_WRITES__:
        print('[cache] * save_cPkl(%r, data)' % (fpath,))
    with open(fpath, 'wb') as file_:
        cPickle.dump(data, file_, cPickle.HIGHEST_PROTOCOL)


def load_cPkl(fpath):
    if __PRINT_READS__:
        print('[cache] * load_cPkl(%r, data)' % (fpath,))
    with open(fpath, 'rb') as file_:
        data = cPickle.load(file_)
    return data


# --- Global Cache ---

def get_global_cache_dir(appname='ibeis', ensure=False, **kwargs):
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
            from . import util_dbg
            util_dbg.printex(ex, 'Failed opening shelf_fpath',
                             key_list=['shelf_fpath'])
            raise
        #shelf_file = open(shelf_fpath, 'w')
    return __SHELF__


def close_global_shelf(**kwargs):
    global __SHELF__
    if __SHELF__ is not None:
        __SHELF__.close()
    __SHELF__ = None


def global_cache_read(key, **kwargs):
    shelf = get_global_shelf(**kwargs)
    if not 'default' in kwargs:
        return shelf[key]
    else:
        return shelf.get(key, kwargs['default'])


def global_cache_dump(**kwargs):
    shelf_fpath = get_global_shelf_fpath(**kwargs)
    shelf = get_global_shelf(**kwargs)
    print('shelf_fpath = %r' % shelf_fpath)
    print(util_str.dict_str(shelf))


def global_cache_write(key, val, **kwargs):
    """ Writes cache files to a safe place in each operating system """
    shelf = get_global_shelf(**kwargs)
    shelf[key] = val


def delete_global_cache(**kwargs):
    """ Reads cache files to a safe place in each operating system """
    close_global_shelf(**kwargs)
    shelf_fpath = get_global_shelf_fpath(**kwargs)
    util_path.remove_file(shelf_fpath, verbose=True, dryrun=False)


atexit.register(close_global_shelf)  # ensure proper cleanup when exiting python
