from __future__ import print_function, division
from os.path import join, exists, dirname, normpath
import sys
import os
import ctypes as C

# TODO: rectify with the ctypes interface in pyhesaff

#============================
# General CTypes Interface
#============================

__QUIET__ = '--quiet' in sys.argv
__VERBOSE__ = '--verbose' in sys.argv
__VERYVERBOSE__ = '--veryverbose' in sys.argv or '--very-verbose' in sys.argv


def get_lib_fname_list(libname):
    """
    Args:
        <libname>: library name (e.g. 'hesaff', not 'libhesaff')

    Returns:
        <libnames>: list of plausible library file names
    """
    if sys.platform.startswith('win32'):
        libnames = ['lib' + libname + '.dll', libname + '.dll']
    elif sys.platform.startswith('darwin'):
        libnames = ['lib' + libname + '.dylib']
    elif sys.platform.startswith('linux'):
        libnames = ['lib' + libname + '.so']
    else:
        raise Exception('Unknown operating system: %r' % sys.platform)
    return libnames


def get_lib_dpath_list(root_dir):
    """
    returns possible lib locations

    Args:
        <root_dir>: deepest directory to look for a library (dll, so, dylib)

    Returns:
        returns <libnames>: list of plausible directories to look.
    """
    get_lib_dpath_list = [root_dir,
                          join(root_dir, 'lib'),
                          join(root_dir, 'build'),
                          join(root_dir, 'build', 'lib')]
    return get_lib_dpath_list


def find_lib_fpath(libname, root_dir, recurse_down=True, verbose=False, veryverbose=None):
    """ Search for the library """
    if veryverbose is None:
        veryverbose = __VERYVERBOSE__
    lib_fname_list = get_lib_fname_list(libname)
    tried_fpaths = []
    while root_dir is not None:
        for lib_fname in lib_fname_list:
            for lib_dpath in get_lib_dpath_list(root_dir):
                lib_fpath = normpath(join(lib_dpath, lib_fname))
                if exists(lib_fpath):
                    if verbose:
                        print('\n[C] Checked: '.join(tried_fpaths))
                    if __VERBOSE__ and not __QUIET__:
                        print('using: %r' % lib_fpath)
                    return lib_fpath
                else:
                    # Remember which candiate library fpaths did not exist
                    tried_fpaths.append(lib_fpath)
            _new_root = dirname(root_dir)
            if _new_root == root_dir:
                root_dir = None
                break
            else:
                root_dir = _new_root
        if not recurse_down:
            break

    msg = ('\n[C!] ERROR: load_clib(libname=%r root_dir=%r, recurse_down=%r, verbose=%r)' %
           (libname, root_dir, recurse_down, verbose) +
           '\n[C!] Cannot find dynamic library')
    if veryverbose:
        print(msg)
        print('\n[C!] Checked: '.join(tried_fpaths))
    raise ImportError(msg)


def load_clib(libname, root_dir):
    """
    Does the work.

    Args:
        libname:  library name (e.g. 'hesaff', not 'libhesaff')

        root_dir: the deepest directory searched for the
                  library file (dll, dylib, or so).

    Returns:
        clib: a ctypes object used to interface with the library
    """
    lib_fpath = find_lib_fpath(libname, root_dir)
    try:
        clib = C.cdll[lib_fpath]

        def def_cfunc(return_type, func_name, arg_type_list):
            """ Function to define the types that python needs to talk to c """
            cfunc = getattr(clib, func_name)
            cfunc.restype = return_type
            cfunc.argtypes = arg_type_list
        clib.__LIB_FPATH__ = lib_fpath
        return clib, def_cfunc
    except OSError as ex:
        print('[C!] Caught OSError:\n%s' % ex)
        errsuffix = 'Is there a missing dependency?'
    except Exception as ex:
        print('[C!] Caught Exception:\n%s' % ex)
        errsuffix = 'Was the library correctly compiled?'
    print('[C!] cwd=%r' % os.getcwd())
    print('[C!] load_clib(libname=%r root_dir=%r)' % (libname, root_dir))
    print('[C!] lib_fpath = %r' % lib_fpath)
    errmsg = '[C] Cannot LOAD %r dynamic library. ' % (libname,) + errsuffix
    print(errmsg)
    raise ImportError(errmsg)
