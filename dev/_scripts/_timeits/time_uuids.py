# -*- coding: utf-8 -*-
"""
Script to help time determenistic uuid creation
"""
from __future__ import absolute_import, division, print_function
from six.moves import range, builtins
import os
import multiprocessing
import time
from PIL import Image
import hashlib
import numpy as np
import uuid
from utool._internal.meta_util_six import get_funcname

# My data getters
from vtool_ibeis.tests import grabdata

elephant = grabdata.get_testimg_path('elephant.jpg')
lena = grabdata.get_testimg_path('lena.jpg')
zebra = grabdata.get_testimg_path('zebra.jpg')
jeff = grabdata.get_testimg_path('jeff.png')
gpath = zebra
if not os.path.exists(gpath):
    gpath = zebra


try:
    getattr(builtins, 'profile')
    __LINE_PROFILE__ = True
except AttributeError:
    __LINE_PROFILE__ = False

    def profile(func):
        return func


@profile
def get_image_uuid(img_bytes_):
    # hash the bytes using sha1
    bytes_sha1 = hashlib.sha1(img_bytes_)
    hashbytes_20 = bytes_sha1.digest()
    # sha1 produces 20 bytes, but UUID requires 16 bytes
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)
    return uuid_


@profile
def make_uuid_PIL_bytes(gpath):
    pil_img = Image.open(gpath, 'r')  # NOQA
    # Read PIL image data
    img_bytes_ = pil_img.tobytes()
    uuid_ = get_image_uuid(img_bytes_)
    return uuid_


@profile
def make_uuid_NUMPY_bytes(gpath):
    pil_img = Image.open(gpath, 'r')  # NOQA
    # Read PIL image data
    np_img = np.asarray(pil_img)
    np_flat = np_img.ravel()
    img_bytes_ = np_flat.tostring()
    uuid_ = get_image_uuid(img_bytes_)
    return uuid_


@profile
def make_uuid_NUMPY_STRIDE_16_bytes(gpath):
    pil_img = Image.open(gpath, 'r')  # NOQA
    # Read PIL image data
    np_img = np.asarray(pil_img)
    np_flat = np_img.ravel()[::16]
    img_bytes_ = np_flat.tostring()
    uuid_ = get_image_uuid(img_bytes_)
    return uuid_


@profile
def make_uuid_NUMPY_STRIDE_64_bytes(gpath):
    pil_img = Image.open(gpath, 'r')  # NOQA
    # Read PIL image data
    img_bytes_ = np.asarray(pil_img).ravel()[::64].tostring()
    uuid_ = get_image_uuid(img_bytes_)
    return uuid_


@profile
def make_uuid_CONTIG_NUMPY_bytes(gpath):
    pil_img = Image.open(gpath, 'r')  # NOQA
    # Read PIL image data
    np_img = np.asarray(pil_img)
    np_flat = np_img.ravel().tostring()
    np_contig = np.ascontiguousarray(np_flat)
    img_bytes_ = np_contig.tostring()
    uuid_ = get_image_uuid(img_bytes_)
    return uuid_


@profile
def make_uuid_CONTIG_NUMPY_STRIDE_16_bytes(gpath):
    pil_img = Image.open(gpath, 'r')  # NOQA
    # Read PIL image data
    np_img = np.asarray(pil_img)
    np_contig = np.ascontiguousarray(np_img.ravel()[::16])
    img_bytes_ = np_contig.tostring()
    uuid_ = get_image_uuid(img_bytes_)
    return uuid_


@profile
def make_uuid_CONTIG_NUMPY_STRIDE_64_bytes(gpath):
    pil_img = Image.open(gpath, 'r')  # NOQA
    # Read PIL image data
    img_bytes_ = np.ascontiguousarray(np.asarray(pil_img).ravel()[::64]).tostring()
    uuid_ = get_image_uuid(img_bytes_)
    return uuid_


if __name__ == '__main__':
    multiprocessing.freeze_support()  # win32
    test_funcs = [
        make_uuid_PIL_bytes,
        make_uuid_NUMPY_bytes,
        make_uuid_NUMPY_STRIDE_16_bytes,
        make_uuid_NUMPY_STRIDE_64_bytes,
        make_uuid_CONTIG_NUMPY_bytes,
        make_uuid_CONTIG_NUMPY_STRIDE_16_bytes,
        make_uuid_CONTIG_NUMPY_STRIDE_64_bytes,
    ]
    func_strs = ', '.join([get_funcname(func) for func in test_funcs])
    # cool trick
    setup = 'from __main__ import (gpath, %s) ' % (func_strs,)

    number = 10

    for func in test_funcs:
        funcname = get_funcname(func)
        print('Running: %s' % funcname)
        if __LINE_PROFILE__:
            start = time.time()
            for _ in range(number):
                func(gpath)
            total_time = time.time() - start
        else:
            import timeit

            stmt = '%s(gpath)' % funcname
            total_time = timeit.timeit(stmt=stmt, setup=setup, number=number)
        print('timed: %r seconds in %s' % (total_time, funcname))
