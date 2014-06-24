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
    'MACHINE_GPATH' : {
        'hyrule': '/home/joncrall/.config/utool/testdata/lena.jpg',
    },

    'MACHINE_UUIDS': {
        'hyrule': UUID('d3daf98d-3035-65b7-2ff4-ca4076ab0cf1'),
        'hendrick': UUID('f2c4658f-a722-793b-8578-c392fd550888'),
    },

    'MACHINE_SUMS' : {
        'hyrule': 18615625,
    },

    'MACHINE_BYTES': {
        'hyrule': r'\xe2\x98U\xccib\xc8\x7fm\xbfXp\xb1cb\xccbl\xdfTb\xe1;[\xd1ba\xae\x81\xa3\xc1\x84n\xb5df\xcd`k\xd2]m\xe5H^\xd2a^\xd9\x80[\xce\x81Q\xc9e_\xcaNj\xfc]k\xd9N]\xcfb`\xe4}^`\x8f ~~~TRUNCATED~~~ u\x1cJ\xdcq\x86\x8e<<\xde\x1bf\x95^j\xdd\xc7V\xb2',
    },

    'MACHINE_HASH16' : {
        'hyrule': r'\xd3\xda\xf9\x8d05e\xb7/\xf4\xca@v\xab\x0c\xf1',
    },

    'MACHINE_HASH20' : {
        'hyrule': r'\xd3\xda\xf9\x8d05e\xb7/\xf4\xca@v\xab\x0c\xf1\x99\x93\xd5\xcb',
    },
}


def TEST_UUID():
    print('Machine Name: %r' % MACHINE_NAME)

    print('Image.PILLOW_VERSION: %r' % Image.PILLOW_VERSION)
    assert Image.PILLOW_VERSION == '1.1.7'
    print('Image.PILLOW_VERSION: %r' % Image.PILLOW_VERSION)
    assert Image.PILLOW_VERSION == '1.1.7'

    print('[TEST_UUID]')
    gpath = grabdata.get_test_gpaths(names='lena')[0]

    print('gpath = %r' % gpath)

    pil_img = Image.open(gpath)
    uuid1 = preproc_image.get_image_uuid(pil_img)
    target_uuid = UUID('f2c4658f-a722-793b-8578-c392fd550888')
    print('uuid1 = %r' % uuid1)
    print('target_uuid = %r' % target_uuid)

    pil_img = Image.open(gpath)
    npimg = np.asarray(pil_img)
    print('[ginfo] npimg.sum() = %r' % npimg.sum())
    img_bytes_ = npimg.ravel()[::64].tostring()

    print('img_bytes_ = %r' % (utool.truncate_str(img_bytes_)))
    bytes_sha1 = hashlib.sha1(img_bytes_)
    hashbytes_20 = bytes_sha1.digest()
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)

    print('hashbytes_16 = %r' % (hashbytes_16,))
    print('hashbytes_20 = %r' % (hashbytes_20,))
    print('bytes_sha1 = %r' % (bytes_sha1,))
    print('uuid_ = %r' % (uuid_,))

    assert uuid1 == target_uuid, 'uuid and target_uuid do not match'

    assert uuid_ == uuid1, 'uuid_ does not match uuid1'

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = utool.run_test(TEST_UUID)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
