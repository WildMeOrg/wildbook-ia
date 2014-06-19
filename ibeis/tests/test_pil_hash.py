#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from vtool.tests import grabdata
from ibeis.model.preproc import preproc_image
from PIL import Image
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_PIL_HASH]')

def TEST_PIL_HASH():
    print('[TEST] GET_TEST_IMAGE_PATHS')
    # The test api returns a list of interesting chip indexes
    gpath_list = grabdata.get_test_gpaths(ndata=None)
    pil_img_list = [Image.open(gpath, 'r') for gpath in gpath_list]
    uuid_list = [preproc_image.get_image_uuid(pil_img) for pil_img in pil_img_list]
    unique_uuid_list = list(set(uuid_list))
    assert len(uuid_list) == len(unique_uuid_list), 'Reinstall PIL, watch for libjpeg'
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For win32
    test_locals = utool.run_test(TEST_PIL_HASH)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
