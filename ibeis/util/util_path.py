from __future__ import division, print_function
from os.path import (join, normpath, split, isdir, isfile, exists, islink, ismount)
from itertools import izip
import fnmatch
import os
import shutil
import warnings
from .util_dbg import get_caller_name
from .util_inject import inject
from .util_progress import progress_func
print, print_, printDBG, rrr, profile = inject(__name__, '[path]')


__VERBOSE__ = False


def path_ndir_split(path, n):
    path, ndirs = split(path)
    for i in xrange(n - 1):
        path, name = split(path)
        ndirs = name + os.path.sep + ndirs
    return ndirs


def remove_file(fpath, verbose=True, dryrun=False, **kwargs):
    try:
        if dryrun:
            if verbose:
                print('[util] Dryrem %r' % fpath)
        else:
            if verbose:
                print('[util] Removing %r' % fpath)
            os.remove(fpath)
    except OSError as e:
        warnings.warn('OSError: %s,\n Could not delete %s' % (str(e), fpath))
        return False
    return True


def remove_dirs(dpath, dryrun=False, **kwargs):
    print('[util] Removing directory: %r' % dpath)
    try:
        shutil.rmtree(dpath)
    except OSError as e:
        warnings.warn('OSError: %s,\n Could not delete %s' % (str(e), dpath))
        return False
    return True


def remove_files_in_dir(dpath, fname_pattern='*', recursive=False, verbose=True,
                        dryrun=False, **kwargs):
    print('[util] Removing files:')
    print('  * in dpath = %r ' % dpath)
    print('  * matching pattern = %r' % fname_pattern)
    print('  * recursive = %r' % recursive)
    num_removed, num_matched = (0, 0)
    if not exists(dpath):
        msg = ('!!! dir = %r does not exist!' % dpath)
        print(msg)
        warnings.warn(msg, category=UserWarning)
    for root, dname_list, fname_list in os.walk(dpath):
        for fname in fnmatch.filter(fname_list, fname_pattern):
            num_matched += 1
            num_removed += remove_file(join(root, fname), verbose=verbose,
                                       dryrun=dryrun, **kwargs)
        if not recursive:
            break
    print('[util] ... Removed %d/%d files' % (num_removed, num_matched))
    return True


def delete(path, dryrun=False, recursive=True, verbose=True, **kwargs):
    # Deletes regardless of what the path is
    print('[util] Deleting path=%r' % path)
    rmargs = dict(dryrun=dryrun, recursive=recursive, verbose=verbose, **kwargs)
    if not exists(path):
        msg = ('..does not exist!')
        print(msg)
        return False
    if isdir(path):
        flag = remove_files_in_dir(path, **rmargs)
        flag = flag and remove_dirs(path, **rmargs)
    elif isfile(path):
        flag = remove_file(path, **rmargs)
    return flag


def longest_existing_path(_path):
    while True:
        _path_new = os.path.dirname(_path)
        if exists(_path_new):
            _path = _path_new
            break
        if _path_new == _path:
            print('!!! This is a very illformated path indeed.')
            _path = ''
            break
        _path = _path_new
    return _path


def checkpath(path_, verbose=__VERBOSE__):
    'returns true if path_ exists on the filesystem'
    path_ = normpath(path_)
    if verbose:
        pretty_path = path_ndir_split(path_, 2)
        caller_name = get_caller_name()
        print_('[%s] checkpath(%r)' % (caller_name, pretty_path))
        if exists(path_):
            path_type = ''
            if isfile(path_):
                path_type += 'file'
            if isdir(path_):
                path_type += 'directory'
            if islink(path_):
                path_type += 'link'
            if ismount(path_):
                path_type += 'mount'
            path_type = 'file' if isfile(path_) else 'directory'
            print_('...(%s) exists\n' % (path_type,))
        else:
            print_('... does not exist\n')
            if __VERBOSE__:
                print_('[util] \n  ! Does not exist\n')
                _longest_path = longest_existing_path(path_)
                print_('[util] ... The longest existing path is: %r\n' % _longest_path)
            return False
        return True
    else:
        return exists(path_)


def ensurepath(path_, **kwargs):
    if not checkpath(path_, **kwargs):
        print('[util] mkdir(%r)' % path_)
        os.makedirs(path_)
    return True


def ensuredir(path_, **kwargs):
    return ensurepath(path_, **kwargs)


def assertpath(path_):
    if not checkpath(path_):
        raise AssertionError('Asserted path does not exist: ' + path_)


# ---File Copy---
def copy_task(cp_list, test=False, nooverwrite=False, print_tasks=True):
    '''
    Input list of tuples:
        format = [(src_1, dst_1), ..., (src_N, dst_N)]
    Copies all files src_i to dst_i
    '''
    num_overwrite = 0
    _cp_tasks = []  # Build this list with the actual tasks
    if nooverwrite:
        print('[util] Removed: copy task ')
    else:
        print('[util] Begining copy + overwrite task.')
    for (src, dst) in iter(cp_list):
        if exists(dst):
            num_overwrite += 1
            if print_tasks:
                print('[util] !!! Overwriting ')
            if not nooverwrite:
                _cp_tasks.append((src, dst))
        else:
            if print_tasks:
                print('[util] ... Copying ')
                _cp_tasks.append((src, dst))
        if print_tasks:
            print('[util]    ' + src + ' -> \n    ' + dst)
    print('[util] About to copy %d files' % len(cp_list))
    if nooverwrite:
        print('[util] Skipping %d tasks which would have overwriten files' % num_overwrite)
    else:
        print('[util] There will be %d overwrites' % num_overwrite)
    if not test:
        print('[util]... Copying')
        for (src, dst) in iter(_cp_tasks):
            shutil.copy(src, dst)
        print('[util]... Finished copying')
    else:
        print('[util]... In test mode. Nothing was copied.')


def copy(src, dst):
    if exists(src):
        if exists(dst):
            prefix = 'C+O'
            print('[util] [Copying + Overwrite]:')
        else:
            prefix = 'C'
            print('[util] [Copying]: ')
        print('[%s] | %s' % (prefix, src))
        print('[%s] ->%s' % (prefix, dst))
        shutil.copy(src, dst)
    else:
        prefix = 'Miss'
        print('[util] [Cannot Copy]: ')
        print('[%s] src=%s does not exist!' % (prefix, src))
        print('[%s] dst=%s' % (prefix, dst))


def copy_all(src_dir, dest_dir, glob_str_list, recursive=False):
    ensuredir(dest_dir)
    if not isinstance(glob_str_list, list):
        glob_str_list = [glob_str_list]
    for root, dirs, files in os.walk(src_dir):
        for dname_ in dirs:
            for glob_str in glob_str_list:
                if fnmatch.fnmatch(dname_, glob_str):
                    src = normpath(join(src_dir, dname_))
                    dst = normpath(join(dest_dir, dname_))
                    ensuredir(dst)
        for fname_ in files:
            for glob_str in glob_str_list:
                if fnmatch.fnmatch(fname_, glob_str):
                    src = normpath(join(src_dir, fname_))
                    dst = normpath(join(dest_dir, fname_))
                    copy(src, dst)
        if not recursive:
            break


def copy_list(src_list, dst_list, lbl='Copying'):
    # Feb - 6 - 2014 Copy function
    def domove(src, dst, count):
        try:
            shutil.copy(src, dst)
        except OSError:
            return False
        mark_progress(count)
        return True
    task_iter = izip(src_list, dst_list)
    mark_progress, end_progress = progress_func(len(src_list), lbl=lbl)
    success_list = [domove(src, dst, count) for count, (src, dst) in enumerate(task_iter)]
    end_progress()
    return success_list


def move_list(src_list, dst_list, lbl='Moving'):
    # Feb - 6 - 2014 Move function
    def domove(src, dst, count):
        try:
            shutil.move(src, dst)
        except OSError:
            return False
        mark_progress(count)
        return True
    task_iter = izip(src_list, dst_list)
    mark_progress, end_progress = progress_func(len(src_list), lbl=lbl)
    success_list = [domove(src, dst, count) for count, (src, dst) in enumerate(task_iter)]
    end_progress()
    return success_list


def win_shortcut(source, link_name):
    import ctypes
    csl = ctypes.windll.kernel32.CreateSymbolicLinkW
    csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    csl.restype = ctypes.c_ubyte
    flags = 1 if isdir(source) else 0
    retval = csl(link_name, source, flags)
    if retval == 0:
        #warn_msg = '[util] Unable to create symbolic link on windows.'
        #print(warn_msg)
        #warnings.warn(warn_msg, category=UserWarning)
        if checkpath(link_name):
            return True
        raise ctypes.WinError()


def symlink(source, link_name, noraise=False):
    if os.path.islink(link_name):
        print('[util] symlink %r exists' % (link_name))
        return
    print('[util] Creating symlink: source=%r link_name=%r' % (source, link_name))
    try:
        os_symlink = getattr(os, "symlink", None)
        if callable(os_symlink):
            os_symlink(source, link_name)
        else:
            win_shortcut(source, link_name)
    except Exception:
        checkpath(link_name, True)
        checkpath(source, True)
        if not noraise:
            raise


def file_bytes(fpath):
    return os.stat(fpath).st_size


def byte_str2(nBytes):
    if nBytes < 2.0 ** 10:
        return byte_str(nBytes, 'KB')
    if nBytes < 2.0 ** 20:
        return byte_str(nBytes, 'KB')
    if nBytes < 2.0 ** 30:
        return byte_str(nBytes, 'MB')
    else:
        return byte_str(nBytes, 'GB')


def byte_str(nBytes, unit='bytes'):
    if unit.lower().startswith('b'):
        nUnit = nBytes
    elif unit.lower().startswith('k'):
        nUnit =  nBytes / (2.0 ** 10)
    elif unit.lower().startswith('m'):
        nUnit =  nBytes / (2.0 ** 20)
    elif unit.lower().startswith('g'):
        nUnit = nBytes / (2.0 ** 30)
    else:
        raise NotImplementedError('unknown nBytes=%r unit=%r' % (nBytes, unit))
    return '%.2f %s' % (nUnit, unit)


def file_megabytes(fpath):
    return os.stat(fpath).st_size / (2.0 ** 20)


def file_megabytes_str(fpath):
    return ('%.2f MB' % file_megabytes(fpath))


def glob(dirname, pattern, recursive=False):
    matching_fnames = []
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            if not fnmatch.fnmatch(fname, pattern):
                continue
            matching_fnames.append(join(root, fname))
        if not recursive:
            break
    return matching_fnames
