# -*- coding: utf-8 -*-
import utool as ut


__all__ = ('get_testdata_dir',)


def get_testdata_dir(ensure=True, key='testdb1'):
    """
    Gets test img directory and downloads it if it doesn't exist
    """
    testdata_map = {
        # 'testdb1': 'https://cthulhu.dyn.wildme.io/public/data/testdata.zip'}
        'testdb1': 'https://wildbookiarepository.azureedge.net/data/testdata.zip',
    }
    zipped_testdata_url = testdata_map[key]
    testdata_dir = ut.grab_zipped_url(zipped_testdata_url, ensure=ensure)
    return testdata_dir
