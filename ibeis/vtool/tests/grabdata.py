from __future__ import absolute_import, division, print_function
import utool
from os.path import exists, dirname, join, realpath

TESTDATA_DIR = realpath(join(dirname(__file__), 'testdata'))


def assert_testdata():
    assert exists(TESTDATA_DIR), ('TESTDATA_DIR=%r does not exist' % TESTDATA_DIR)


def get_testdata_dir(ensure=True):
    """
    Gets test img directory and downloads it if it doesn't exist
    """
    if ensure:
        ensure_testdata()
    assert_testdata()
    return TESTDATA_DIR


def ensure_testdata():
    if not exists(TESTDATA_DIR):
        download_testdata()
    assert_testdata()


def get_testimg_path(gname):
    """
    Returns path to image in testdata
    """
    testdata_dir = get_testdata_dir()
    gpath = realpath(join(testdata_dir, gname))
    return gpath


def download_testdata():
    zip_fpath = realpath(join(TESTDATA_DIR, '..', 'testdata.zip'))
    # Download and unzip testdata
    print('[grabdata] Downloading TESTDATA_DIR=%s' % TESTDATA_DIR)
    dropbox_link = 'https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip'
    utool.download_url(dropbox_link, zip_fpath)
    utool.unzip_file(zip_fpath)
    # Cleanup
    utool.delete(zip_fpath)

if __name__ == '__main__':
    ensure_testdata()
