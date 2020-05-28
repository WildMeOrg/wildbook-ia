# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from PIL import Image
import cv2
import utool as ut


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
