# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from vtool._pyflann_backend import pyflann as pyflann
import utool as ut
import uuid
import six
from os.path import exists, join
import lockfile

(print, rrr, profile) = ut.inject2(__name__)


class Win32CompatTempFile(object):
    """
    mimics tempfile.NamedTemporaryFile but allows the file to be closed without
    being deleted.  This lets a second process (like the FLANN) read/write to
    the file in a win32 system. The file is instead deleted after the
    Win32CompatTempFile object goes out of scope.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.smk.pickle_flann import *  # NOQA
        >>> verbose = True
        >>> temp = Win32CompatTempFile(verbose=verbose)
        >>> data = six.b(str('10010'))
        >>> print('data = %r' % (data,))
        >>> data1 = temp.read()
        >>> print('data1 = %r' % (data1,))
        >>> temp.write(data)
        >>> data2 = temp.read()
        >>> print('data2 = %r' % (data2,))
        >>> temp.close()
        >>> assert data != data1
        >>> assert data == data2
        >>> ut.assert_raises(ValueError, temp.close)
        >>> assert not ut.checkpath(temp.fpath, verbose=verbose)
    """

    def __init__(temp, delete=True, verbose=False):
        temp.delete = delete
        appname = 'wbia'
        temp.dpath = ut.ensure_app_resource_dir(appname, 'tempfiles')
        temp.fpath = None
        temp.fname = None
        temp._isclosed = False
        temp.verbose = verbose
        temp._create_unique_file()

    @property
    def name(temp):
        return temp.fpath

    def read(temp):
        temp._check_open()
        with open(temp.fpath, 'rb') as file_:
            return file_.read()

    def write(temp, data):
        temp._check_open()
        with open(temp.fpath, 'wb') as file_:
            file_.write(data)
            file_.flush()

    def close(temp):
        temp._check_open()
        if temp.delete and exists(temp.fpath):
            ut.delete(temp.fpath, verbose=temp.verbose)
        temp._isclosed = True

    def _create_unique_file(temp):
        temp._check_open()
        with lockfile.LockFile(join(temp.dpath, 'tempfile.lock')):
            flag = True
            while flag or exists(temp.fpath):
                temp.fname = six.text_type(uuid.uuid4()) + '.temp'
                temp.fpath = join(temp.dpath, temp.fname)
                flag = False
            ut.touch(temp.fpath, verbose=temp.verbose)

    def _check_open(temp):
        if temp._isclosed:
            raise ValueError('I/O operation on closed object')

    def __del__(temp):
        if not temp._isclosed:
            temp.close()


if pyflann is not None:

    class PickleFLANN(pyflann.FLANN):
        """
        Adds the ability to pickle a flann class on a unix system.
        (Actually, pickle still wont work because we need the original point data.
        But we can do a custom dumps and a loads)

        CommandLine:
            python -m wbia.algo.smk.pickle_flann PickleFLANN

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.smk.pickle_flann import *  # NOQA
            >>> import numpy as np
            >>> rng = np.random.RandomState(42)
            >>> data = rng.rand(10, 2)
            >>> query = rng.rand(5, 2)
            >>> flann = PickleFLANN()
            >>> flann.build_index(data, random_seed=42)
            >>> index_bytes = flann.dumps()
            >>> flann2 = PickleFLANN()
            >>> flann2.loads(index_bytes, data)
            >>> assert flann2 is not flann
            >>> assert flann2.dumps() == index_bytes
            >>> idx1 = flann.nn_index(query)[0]
            >>> idx2 = flann2.nn_index(query)[0]
            >>> assert np.all(idx1 == idx2)
        """

        def dumps(self):
            """
            # Make a special wordflann pickle
            http://www.linuxscrew.com/2010/03/24/fastest-way-to-create-ramdisk-in-ubuntulinux/
            sudo mkdir /tmp/ramdisk; chmod 777 /tmp/ramdisk
            sudo mount -t tmpfs -o size=256M tmpfs /tmp/ramdisk/
            http://zeblog.co/?p=1588
            """
            # import tempfile
            # assert not ut.WIN32, 'Fix on WIN32. Cannot write to temp file'
            # temp = tempfile.NamedTemporaryFile(delete=True)
            temp = Win32CompatTempFile(delete=True, verbose=False)
            try:
                self.save_index(temp.name)
                index_bytes = temp.read()
            except Exception:
                raise
            finally:
                temp.close()
            return index_bytes

        def loads(self, index_bytes, pts):
            # import tempfile
            # assert not ut.WIN32, 'Fix on WIN32. Cannot write to temp file'
            # temp = tempfile.NamedTemporaryFile(delete=True)
            temp = Win32CompatTempFile(delete=True, verbose=False)
            try:
                temp.write(index_bytes)
                # temp.file.flush()
                self.load_index(temp.name, pts)
            except Exception:
                raise
            finally:
                temp.close()


else:
    PickleFLANN = None


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.smk.pickle_flann
        python -m wbia.algo.smk.pickle_flann --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
