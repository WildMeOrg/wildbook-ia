#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from PIL import Image
from ibeis.model.preproc import preproc_image
from uuid import UUID
from vtool.tests import grabdata
import hashlib
import multiprocessing
import numpy as np
import utool
import uuid
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_UUID]')

MACHINE_NAME = utool.get_computer_name()

MACHINE_VALS = {}
MACHINE_VALS['lena'] = {
    'GPATH' : {
        'hyrule':         '/home/joncrall/.config/utool/testdata/lena.jpg',
        'hendrik-vostro': '/home/hendrik/.config/utool/testdata/lena.jpg',
    },

    'UUID': {
        'hyrule':         UUID('d3daf98d-3035-65b7-2ff4-ca4076ab0cf1'),
        'hendrik-vostro': UUID('f2c4658f-a722-793b-8578-c392fd550888'),
    },

    'SUM' : {
        'hyrule':         18615625,
        'hendrik-vostro': 18615820,
    },

    'BYTES': {
        'hyrule':         r'\xe2\x98U\xccib\xc8\x7fm\xbfXp\xb1cb\xccbl\xdfTb\xe1;[\xd1ba\xae\x81\xa3\xc1\x84n\xb5df\xcd`k\xd2]m\xe5H^\xd2a^\xd9\x80[\xce\x81Q\xc9e_\xcaNj\xfc]k\xd9N]\xcfb`\xe4}^`\x8f ~~~TRUNCATED~~~ u\x1cJ\xdcq\x86\x8e<<\xde\x1bf\x95^j\xdd\xc7V\xb2',
        'hendrik-vostro': r'\xe2\x99W\xccib\xc8\x7fm\xbfXn\xb1cb\xccbl\xdfTb\xe1;[\xd1aa\xae\x81\xa1\xc1\x84n\xb5df\xcd_m\xd0]m\xe5H^\xd2a^\xd7\x80]\xd1\x81R\xc9e_\xcaNj\xfb]k\xd9N]\xcfb`\xe4}^`\x8f ~~~TRUNCATED~~~ u\x1cJ\xdaq\x86\x8e<<\xde\x1bf\x95^j\xdd\xc8V\xb1',
    },

    'HASH16' : {
        'hyrule':         r'\xd3\xda\xf9\x8d05e\xb7/\xf4\xca@v\xab\x0c\xf1',
        'hendrik-vostro': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88',
    },

    'HASH20' : {
        'hyrule':         r'\xd3\xda\xf9\x8d05e\xb7/\xf4\xca@v\xab\x0c\xf1\x99\x93\xd5\xcb',
        'hendrik-vostro': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88\xac.sk',
    },
}


def test_uuid(name):
    print('-----------------------')
    print('[TEST_UUID] TESTING: %r' % name)
    if name.find('http') != -1:
        gpath = utool.grab_file_url(name)
    else:
        gpath = grabdata.get_test_gpaths(names=name)[0]
    import cv2
    cv2_version = cv2.__version__
    npimg2 = cv2.imread(gpath)
    npshape = npimg2.shape
    npsum = npimg2.sum()

    pil_img = Image.open(gpath)
    uuid1 = preproc_image.get_image_uuid(pil_img)
    npimg = np.asarray(pil_img)
    img_bytes_ = npimg.ravel()[::64].tostring()
    bytes_sha1 = hashlib.sha1(img_bytes_)
    hashbytes_20 = bytes_sha1.digest()
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)

    sum_ = npimg.sum()
    size = pil_img.size

    locals_ = locals()
    def print2(key):
        val = utool.truncate_str(str(locals_[key]))
        print('%r: {%r: %r}' % (key, MACHINE_NAME, val))

    print2('gpath')
    print2('cv2_version')
    print2('npshape')
    print2('npsum')
    print2('uuid1')
    print2('size')
    print2('sum_')
    print2('img_bytes_')
    print2('hashbytes_16')
    print2('hashbytes_20')
    print2('bytes_sha1')
    print2('uuid_')

    locals_ = locals()
    def print2(key):
        val = locals_[key]
        print('%r: %r' % (MACHINE_NAME, val))

    try:
        target_uuid = MACHINE_VALS[name]['UUID'][MACHINE_NAME]
        print2('target_uuid')
        assert uuid1 == target_uuid, 'uuid and target_uuid do not match'
        assert uuid_ == uuid1, 'uuid_ does not match uuid1'
    except Exception as ex:
        utool.printex(ex)
        return False
    return True


def TEST_UUID():
    print('Machine Name: %r' % MACHINE_NAME)
    try:
        print('Image.PILLOW_VERSION: %r' % Image.PILLOW_VERSION)
        assert Image.PILLOW_VERSION == '2.4.0'
        print('Image.PILLOW_VERSION: %r' % Image.PILLOW_VERSION)
        assert Image.PILLOW_VERSION == '2.4.0'
    except Exception as ex:
        utool.printex(ex)
        pass

    right = 'http://i.imgur.com/QqSkNZe.png'

    assert all([test_uuid(name) for name in ['lena', 'jeff', 'easy1', right]])

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = utool.run_test(TEST_UUID)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
