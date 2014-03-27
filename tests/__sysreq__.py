from __future__ import division, print_function
import sys
sys.argv.append('--strict')

from os.path import realpath, dirname, join, exists

utool_path = realpath(join(dirname(__file__), '..'))
print('[test] appending: %r' % utool_path)
try:
    assert exists(join(utool_path, 'utool')), ('cannot find util in: %r' % utool_path)
except AssertionError as ex:
    print('Caught Assertion Error: %s' % (ex))
    utool_path = join('..', utool_path)
    assert exists(join(utool_path, 'utool')), ('cannot find util in: %r' % utool_path)

sys.path.append(realpath(utool_path))
import utool.util_sysreq as sysreq
sysreq.ensure_in_pythonpath('hesaff')
