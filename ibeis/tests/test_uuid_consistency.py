#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from six.moves import map
from ibeis.model.preproc import preproc_image
from PIL import Image


def TEST_UUID_CONSISTENCY(ibs):
    gid_list   = ibs.get_valid_gids()
    uuid_list  = ibs.get_image_uuids(gid_list)
    gpath_list = ibs.get_image_paths(gid_list)

    def _uuid_from_gpath(gpath):
        try:
            pil_img = Image.open(gpath, 'r')      # Open PIL Image
        except IOError:
            return None
        uuid = preproc_image.get_image_uuid(pil_img)
        return uuid

    uuid_list_test = list(map(_uuid_from_gpath, gpath_list))
    assert uuid_list == uuid_list_test, 'uuids are inconsistent!'
