#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from ibeis.dev import ibsfuncs
#from itertools import izip
# Python
import multiprocessing
#import numpy as np
from uuid import UUID
# Tools
import utool
from ibeis.control.IBEISControl import IBEISController
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_UUID]')
from ibeis.model.preproc import preproc_image
from vtool.tests import grabdata
from PIL import Image
import hashlib
import numpy as np
import uuid

def TEST_ENCOUNTERS(ibs):
    print('[TEST_UUID]')
    gpath = grabdata.get_test_gpaths(names='lena')[0]
    img = Image.open(gpath)
    uuid1 = preproc_image.get_image_uuid(img)
    uuid2 = UUID('f2c4658f-a722-793b-8578-c392fd550888')
    print('uuid1 = %r' % uuid1)
    print('uuid2 = %r' % uuid2)
    assert uuid1 == uuid2, 'uuid and uuid2 do not match'

    pil_img = Image.open(gpath)
    npimg = np.asarray(pil_img)
    print('[ginfo] npimg.sum() = %r' % npimg.sum())
    img_bytes_ = npimg.ravel()[::64].tostring()

    print('img_bytes_ = %r' % (utool.truncate_str(img_bytes_)))
    bytes_sha1 = hashlib.sha1(img_bytes_)
    hashbytes_20 = bytes_sha1.digest()
    # sha1 produces 20 bytes, but UUID requires 16 bytes
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)

    print('hashbytes_16 = %r' % (hashbytes_16,))
    print('hashbytes_20 = %r' % (hashbytes_20,))
    print('bytes_sha1 = %r' % (bytes_sha1,))
    print('uuid_ = %r' % (uuid_,))
    #uuid_ = uuid.uuid4()
    #print('[ginfo] hashbytes_16 = %r' % (hashbytes_16,))
    #print('[ginfo] uuid_ = %r' % (uuid_,))

    assert uuid_ == uuid1, 'uuid_ does not match uuid1'

    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']
    test_locals = utool.run_test(TEST_ENCOUNTERS, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
