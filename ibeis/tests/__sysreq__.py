from __future__ import absolute_import, division, print_function
from os.path import realpath, dirname, join, exists
import sys


def ensure_utool_in_pythonpath():
    utool_path = realpath(join(dirname(__file__), '..'))
    while utool_path != '' and not exists(join(utool_path, 'utool')):
        utool_path = join(utool_path, '..')
    assert exists(join(utool_path, 'utool')), ('cannot find utool in: %r' % utool_path)
    sys.path.append(realpath(utool_path))

ensure_utool_in_pythonpath()

import utool
utool.ensure_in_pythonpath('hesaff')
utool.ensure_in_pythonpath('ibeis')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
