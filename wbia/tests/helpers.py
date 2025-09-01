# -*- coding: utf-8 -*-
import utool as ut
from wbia.tests.config.urls import TEST_IMAGES_URL

__all__ = ('get_testdata_dir',)


def get_testdata_dir(ensure=True, key='testdb1'):
    """
    Gets test img directory and downloads it if it doesn't exist
    """
    testdata_map = {
        # 'testdb1': 'https://cthulhu.dyn.wildme.io/public/data/testdata.zip'}
        'testdb1': TEST_IMAGES_URL,
    }
    zipped_testdata_url = testdata_map[key]
    testdata_dir = ut.grab_zipped_url(zipped_testdata_url, ensure=ensure)
    return testdata_dir
