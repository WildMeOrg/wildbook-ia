from __future__ import absolute_import, division, print_function
import utool
from os.path import join, realpath
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[testdata]')


def get_testdata_dir(ensure=True):
    """
    Gets test img directory and downloads it if it doesn't exist
    """
    zipped_testdata_url = 'https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip'
    testdata_dir = utool.grab_downloaded_testdata(zipped_testdata_url, ensure=ensure)
    return testdata_dir


def get_testimg_path(gname):
    """
    Returns path to image in testdata
    """
    testdata_dir = get_testdata_dir(ensure=True)
    gpath = realpath(join(testdata_dir, gname))
    return gpath


def ensure_testdata():
    # DEPRICATE
    get_testdata_dir(ensure=True)


if __name__ == '__main__':
    testdata_dir = get_testdata_dir()
    print('testdata lives in: %r' % testdata_dir)
