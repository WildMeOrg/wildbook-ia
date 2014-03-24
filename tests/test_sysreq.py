from __future__ import division, print_function
import sys
from os.path import realpath, dirname, join
utool_path = realpath(join(dirname(__file__), '../..'))
print('[test] appending: %r' % utool_path)
sys.path.append(realpath(utool_path))
import utool.util_sysreq as sysreq
sysreq.ensure_in_pythonpath('hesaff')
