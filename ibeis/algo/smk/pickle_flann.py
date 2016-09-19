# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import pyflann
import utool as ut
(print, rrr, profile) = ut.inject2(__name__)


class PickleFLANN(pyflann.FLANN):
    """
    Adds the ability to pickle a flann class on a unix system.
    (Actually, pickle still wont work because we need the original point data.
     But we can do a custom dumps and a loads)
    """

    def dumps(self):
        """
        # Make a special wordflann pickle
        # THIS WILL NOT WORK ON WINDOWS
        http://www.linuxscrew.com/2010/03/24/fastest-way-to-create-ramdisk-in-ubuntulinux/
        sudo mkdir /tmp/ramdisk; chmod 777 /tmp/ramdisk
        sudo mount -t tmpfs -o size=256M tmpfs /tmp/ramdisk/
        http://zeblog.co/?p=1588
        """
        import tempfile
        assert not ut.WIN32, 'Fix on WIN32. Cannot write to temp file'
        temp = tempfile.NamedTemporaryFile(delete=True)
        try:
            self.save_index(temp.name)
            index_bytes = temp.read()
        except Exception:
            raise
        finally:
            temp.close()
        return index_bytes

    def loads(self, index_bytes, pts):
        import tempfile
        assert not ut.WIN32, 'Fix on WIN32. Cannot write to temp file'
        temp = tempfile.NamedTemporaryFile(delete=True)
        try:
            temp.write(index_bytes)
            temp.file.flush()
            self.load_index(temp.name, pts)
        except Exception:
            raise
        finally:
            temp.close()
