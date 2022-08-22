# -*- coding: utf-8 -*-
import logging
from itertools import cycle
from os.path import join, realpath

import utool

from wbia.tests.helpers import get_testdata_dir

print, rrr, profile = utool.inject2(__name__)
logger = logging.getLogger('wbia')


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
