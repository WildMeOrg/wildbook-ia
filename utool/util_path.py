from __future__ import absolute_import, division, print_function
from os.path import (join, realpath, relpath, normpath, split, isdir, isfile, exists,
                     islink, ismount, expanduser, dirname)
from itertools import izip
import os
import sys
import shutil
import fnmatch
import warnings
import zipfile
import urllib
import time
from .util_dbg import get_caller_name
from .util_inject import inject
from .util_progress import progress_func
print, print_, printDBG, rrr, profile = inject(__name__, '[path]')


__VERBOSE__ = '--verbose' in sys.argv


__IMG_EXTS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.ppm']
__LOWER_EXTS = [ext.lower() for ext in __IMG_EXTS]
__UPPER_EXTS = [ext.upper() for ext in __IMG_EXTS]
IMG_EXTENSIONS =  set(__LOWER_EXTS + __UPPER_EXTS)


def unixpath(path):
    """ Corrects fundamental problems with windows paths.~ """
    return truepath(path).replace('\\', '/')


def truepath(path):
    """ Normalizes and returns absolute path with so specs """
    return normpath(realpath(expanduser(path)))


def path_ndir_split(path, n):
    path, ndirs = split(path)
    for i in xrange(n - 1):
        path, name = split(path)
        ndirs = name + os.path.sep + ndirs
    return ndirs


def remove_file(fpath, verbose=True, dryrun=False, ignore_errors=True, **kwargs):
    if dryrun:
        if verbose:
            print('[util] Dryrem %r' % fpath)
        return
    else:
        try:
            os.remove(fpath)
            if verbose:
                print('[util] Removed %r' % fpath)
        except OSError:
            print('[util] Misrem %r' % fpath)
            #warnings.warn('OSError: %s,\n Could not delete %s' % (str(e), fpath))
            if not ignore_errors:
                raise
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


def remove_files_in_dir(dpath, fname_pattern_list='*', recursive=False, verbose=True,
                        dryrun=False, ignore_errors=False, **kwargs):
    if isinstance(fname_pattern_list, str):
        fname_pattern_list = [fname_pattern_list]
    print('[util] Removing files:')
    print('  * in dpath = %r ' % dpath)
    print('  * matching patterns = %r' % fname_pattern_list)
    print('  * recursive = %r' % recursive)
    num_removed, num_matched = (0, 0)
    kwargs.update({
        'dryrun': dryrun,
        'verbose': verbose,
    })
    if not exists(dpath):
        msg = ('!!! dir = %r does not exist!' % dpath)
        print(msg)
        warnings.warn(msg, category=UserWarning)
    for root, dname_list, fname_list in os.walk(dpath):
        for fname_pattern in fname_pattern_list:
            for fname in fnmatch.filter(fname_list, fname_pattern):
                num_matched += 1
                num_removed += remove_file(join(root, fname), **kwargs)
        if not recursive:
            break
    print('[util] ... Removed %d/%d files' % (num_removed, num_matched))
    return True


def delete(path, dryrun=False, recursive=True, verbose=True, ignore_errors=True, **kwargs):
    # Deletes regardless of what the path is
    print('[util] Deleting path=%r' % path)
    rmargs = dict(dryrun=dryrun, recursive=recursive, verbose=verbose,
                  ignore_errors=ignore_errors, **kwargs)
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
    """ returns true if path_ exists on the filesystem """
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
    """
    Input list of tuples:
        format = [(src_1, dst_1), ..., (src_N, dst_N)]
    Copies all files src_i to dst_i
    """
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
            shutil.copy2(src, dst)
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
        if isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
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


def copy_list(src_list, dst_list, lbl='Copying: ', ):
    # Feb - 6 - 2014 Copy function
    task_iter = izip(src_list, dst_list)
    mark_progress, end_progress = progress_func(len(src_list), lbl=lbl)
    def docopy(src, dst, count):
        try:
            shutil.copy2(src, dst)
        except OSError:
            return False
        except shutil.Error:
            pass
        mark_progress(count)
        return True
    success_list = [docopy(src, dst, count) for count, (src, dst) in enumerate(task_iter)]
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


def file_megabytes(fpath):
    return os.stat(fpath).st_size / (2.0 ** 20)


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


# --- Images ----

def num_images_in_dir(path):
    'returns the number of images in a directory'
    num_imgs = 0
    for root, dirs, files in os.walk(path):
        for fname in files:
            if matches_image(fname):
                num_imgs += 1
    return num_imgs


def matches_image(fname):
    fname_ = fname.lower()
    img_pats = ['*' + ext for ext in IMG_EXTENSIONS]
    return any([fnmatch.fnmatch(fname_, pat) for pat in img_pats])


def dirsplit(path):
    return path.split(os.sep)


def list_images(img_dpath, ignore_list=[], recursive=True, fullpath=False):
    print(ignore_list)
    ignore_set = set(ignore_list)
    gname_list_ = []
    assertpath(img_dpath)
    # Get all the files in a directory recursively
    for root, dlist, flist in os.walk(img_dpath):
        rel_dpath = relpath(root, img_dpath)
        if any([dname in ignore_set for dname in dirsplit(rel_dpath)]):
            continue
        for fname in iter(flist):
            gname = join(rel_dpath, fname).replace('\\', '/').replace('./', '')
            if fullpath:
                gname_list_.append(join(img_dpath, gname))
            else:
                gname_list_.append(gname)
        if not recursive:
            break
    # Filter out non images or ignorables
    gname_list = [gname for gname in iter(gname_list_)
                  if not gname in ignore_set and matches_image(gname)]
    return gname_list


def fpaths_to_fnames(fpath_list):
    """
    Input: Filepath list
    Output: Filename list
    """
    fname_list = [split(fpath)[1] for fpath in fpath_list]
    return fname_list


def fnames_to_fpaths(fname_list, path):
    fpath_list = [join(path, fname) for fname in fname_list]
    return fpath_list


BadZipfile = zipfile.BadZipfile


def unzip_file(zip_fpath):
    zip_file = zipfile.ZipFile(zip_fpath)
    zip_dir  = dirname(zip_fpath)
    zipped_namelist = zip_file.namelist()
    for member in zipped_namelist:
        (dname, fname) = split(member)
        dpath = join(zip_dir, dname)
        ensurepath(dpath)
        print('[utool] Unzip ' + fname + ' in ' + dpath)
        zip_file.extract(member, path=zip_dir)


def download_url(url, filename):
    # From http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/
    start_time_ptr = [0]
    def reporthook(count, block_size, total_size):
        if count == 0:
            start_time_ptr[0] = time.time()
            return
        duration = time.time() - start_time_ptr[0]
        if duration == 0:
            duration = 1E-9
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write('\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
                        (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()
    print('[utool] Downloading url=%r to filename=%r' % (url, filename))
    urllib.urlretrieve(url, filename=filename, reporthook=reporthook)


def get_module_dir(module, *args):
    module_dir = truepath(dirname(module.__file__))
    if len(args) > 0:
        module_dir = join(module_dir, *args)
    return module_dir


def tail(fpath):
    return split(fpath)[1]
