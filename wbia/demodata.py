# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool
import six
from os.path import join, realpath
from itertools import cycle
from six.moves import range

print, rrr, profile = utool.inject2(__name__)


def get_testdata_dir(ensure=True, key='testdb1'):
    """
    Gets test img directory and downloads it if it doesn't exist
    """
    testdata_map = {'testdb1': 'https://cthulhu.dyn.wildme.io/public/data/testdata.zip'}
    zipped_testdata_url = testdata_map[key]
    testdata_dir = utool.grab_zipped_url(zipped_testdata_url, ensure=ensure)
    return testdata_dir


def get_test_gpaths(ndata=None, names=None, **kwargs):
    # Read ndata from args or command line
    ndata_arg = utool.get_argval(
        '--ndata', type_=int, default=None, help_='use --ndata to specify bigger data'
    )
    if ndata_arg is not None:
        ndata = ndata_arg
    imgdir = get_testdata_dir(**kwargs)
    gpath_list = sorted(list(utool.list_images(imgdir, full=True, recursive=True)))
    # Get only the gpaths of certain names
    if names is not None:
        gpath_list = [
            gpath for gpath in gpath_list if utool.basename_noext(gpath) in names
        ]
    # Get a some number of test images
    if ndata is not None:
        gpath_cycle = cycle(gpath_list)
        if six.PY2:
            gpath_list = [gpath_cycle.next() for _ in range(ndata)]
        else:
            gpath_list = [next(gpath_cycle) for _ in range(ndata)]
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


def ensure_demodata():
    """
    Ensures that you have testdb1 and PZ_MTEST demo databases.
    """
    import wbia
    from wbia import demodata

    # inconsistent ways of getting test data
    demodata.get_testdata_dir(key='testdb1')
    wbia.sysres.ensure_pz_mtest()


if __name__ == '__main__':
    testdata_dir = get_testdata_dir()
    print('testdata lives in: %r' % testdata_dir)
