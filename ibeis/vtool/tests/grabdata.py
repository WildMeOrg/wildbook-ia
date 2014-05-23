from __future__ import absolute_import, division, print_function
import utool
from os.path import join, realpath
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[testdata]')


def get_testdata_dir(ensure=True):
    """
    Gets test img directory and downloads it if it doesn't exist
    """
    zipped_testdata_url = 'https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip'
    testdata_dir = utool.grab_zipped_url(zipped_testdata_url, ensure=ensure)
    return testdata_dir


def get_test_gpaths(ndata=None, lena=True, zebra=False, jeff=False):
    ndata_arg = utool.get_arg('--ndata', type_=int, default=None, help_='use --ndata to specify bigger data')
    if ndata_arg is not None:
        ndata = ndata_arg
    if ndata is None:
        ndata = 1
    imgdir = get_testdata_dir()
    # Build gpath_list
    gname_list = utool.flatten([
        ['lena.jpg']  * utool.get_flag('--lena',   lena, help_='add lena to test images'),
        ['zebra.jpg'] * utool.get_flag('--zebra', zebra, help_='add zebra to test images'),
        ['jeff.png']  * utool.get_flag('--jeff',   jeff, help_='add jeff to test images'),
    ])
    gname_list = utool.util_list.flatten([gname_list] * ndata)
    gpath_list = utool.fnames_to_fpaths(gname_list, imgdir)
    return gpath_list


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
