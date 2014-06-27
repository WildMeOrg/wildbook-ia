from __future__ import absolute_import, division, print_function
import utool
from os.path import join, realpath
from itertools import cycle
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[testdata]')


def get_testdata_dir(ensure=True, key='testdb1'):
    """
    Gets test img directory and downloads it if it doesn't exist
    """
    testdata_map = {
        'testdb1': 'https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip'
    }
    zipped_testdata_url = testdata_map[key]
    testdata_dir = utool.grab_zipped_url(zipped_testdata_url, ensure=ensure)
    return testdata_dir


def get_test_gpaths(ndata=None, names=None, **kwargs):
    # Read ndata from args or command line
    ndata_arg = utool.get_arg('--ndata', type_=int, default=None, help_='use --ndata to specify bigger data')
    if ndata_arg is not None:
        ndata = ndata_arg
    imgdir = get_testdata_dir(**kwargs)
    gpath_list = sorted(list(utool.list_images(imgdir, full=True, recursive=True)))
    # Get only the gpaths of certain names
    if names is not None:
        gpath_list = [gpath for gpath in gpath_list if
                      utool.basename_noext(gpath) in names]
    # Get a some number of test images
    if ndata is not None:
        gpath_cycle = cycle(gpath_list)
        gpath_list  = [gpath_cycle.next() for _ in xrange(ndata)]
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
