# -*- coding: utf-8 -*-
# LICENCE
from __future__ import absolute_import, division, print_function, unicode_literals
from PIL import Image
try:
    import cv2
except ImportError as ex:
    print('WARNING: import cv2 is failing!')
    cv2 = None
import utool as ut
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[img]', DEBUG=False)


def open_pil_image(image_fpath):
    pil_img = Image.open(image_fpath)
    return pil_img


def print_image_checks(img_fpath):
    hasimg = ut.checkpath(img_fpath, verbose=True)
    if hasimg:
        _tup = (img_fpath, ut.filesize_str(img_fpath))
        print('[io] Image %r (%s) exists. Is it corrupted?' % _tup)
    else:
        print('[io] Image %r does not exists ' (img_fpath,))
    return hasimg


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.image
        python -m vtool.image --allexamples
        python -m vtool.image --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
