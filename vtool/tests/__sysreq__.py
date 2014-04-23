from __future__ import absolute_import, print_function, division
import sys
from os.path import  realpath, dirname, join, exists, expanduser
import matplotlib
matplotlib.use('Qt4Agg', warn=True, force=True)


def ensure_utool_in_pythonpath():
    utool_path = realpath(join(dirname(__file__), '..'))
    while utool_path != '' and not exists(join(utool_path, 'utool')):
        utool_path = join(utool_path, '..')
    try:
        assert exists(join(utool_path, 'utool')), ('cannot find utool in: %r' % utool_path)
    except AssertionError:
        # Last ditch effort
        utool_path = join(expanduser('~'), 'code', 'ibeis')
        if not exists(utool_path):
            raise
    sys.path.append(realpath(utool_path))

ensure_utool_in_pythonpath()


ensure_utool_in_pythonpath()
import utool
utool.inject_colored_exceptions()
utool.ensure_in_pythonpath('hesaff')
