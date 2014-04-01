from __future__ import division, print_function
import __builtin__
import sys
import functools
from os.path import realpath, dirname, join, exists
import numpy as np


sys.argv.append('--strict')  # Tests are always strict
VERBOSE = '--verbose' in sys.argv
QUIET   = '--quiet' in sys.argv


def ensure_util_in_pythonpath():
    utool_path = realpath(join(dirname(__file__), '..'))
    if VERBOSE:
        print('[test] appending to pythonpath: %r' % utool_path)
    try:
        assert exists(join(utool_path, 'utool')), ('cannot find util in: %r' % utool_path)
    except AssertionError as ex:
        print('Caught Assertion Error: %s' % (ex))
        utool_path = join('..', utool_path)
        assert exists(join(utool_path, 'utool')), ('cannot find util in: %r' % utool_path)
    sys.path.append(realpath(utool_path))

ensure_util_in_pythonpath()

import utool
utool.util_sysreq.ensure_in_pythonpath('hesaff')
utool.util_sysreq.ensure_in_pythonpath('ibeis')

INTERACTIVE = utool.get_flag(('--interactive', '-i'))


def testcontext(func):
    @functools.wraps(func)
    def test_wrapper(*args, **kwargs):
        try:
            printTEST('[TEST] %s SUCCESS' % (func.func_name,))
            func(*args, **kwargs)
            printTEST('[TEST] %s SUCCESS' % (func.func_name,))
            print(r'''
                  .-""""""-.
                .'          '.
               /   O      O   \
              :                :
              |                |
              : ',          ,' :
               \  '-......-'  /
                '.          .'
                  '-......-'
                  ''')

        except Exception as ex:
            exc_type, exc_value, tb = sys.exc_info()
            # Get locals in the wrapped function
            locals_ = tb.tb_next.tb_frame.f_locals
            printTEST('[TEST] %s FAILED: %s %s' % (func.func_name, type(ex), ex))
            print(r'''
                  .-""""""-.
                .'          '.
               /   O      O   \
              :           `    :
              |                |
              :    .------.    :
               \  '        '  /
                '.          .'
                  '-......-'
                  ''')
            ibs = locals_.get('ibs', None)
            if ibs is not None:
                ibs.db.dump()
            if '--strict' in sys.argv:
                raise
    return test_wrapper


def get_pyhesaff_test_image_paths(ndata):
    import pyhesaff
    #root = utool.getroot()
    imgdir = dirname(pyhesaff.__file__)
    gname_list = utool.flatten([
        ['lena.png']  * utool.get_flag('--lena',   True, help_='add lena to test images'),
        ['zebra.png'] * utool.get_flag('--zebra', False, help_='add zebra to test images'),
        ['test.png']  * utool.get_flag('--jeff',  False, help_='add jeff to test images'),
    ])
    # Build gpath_list
    if ndata == 0:
        gname_list = ['test.png']
    else:
        gname_list = utool.util_list.flatten([gname_list] * ndata)
    gpath_list = [join(imgdir, path) for path in gname_list]
    return gpath_list


def get_test_image_paths(ibs=None, ndata=None):
    if ndata is None:
        # Increase data size
        ndata = utool.get_arg('--ndata', type_=int, default=1, help_='use --ndata to specify bigger data')
    gpath_list = get_pyhesaff_test_image_paths(ndata)
    if INTERACTIVE:
        gpath_list = None
    return gpath_list


# list of 10,000 chips with 3,000 features apeice.
def get_test_numpy_data(shape=(3e3, 128), dtype=np.uint8):
    ndata = utool.get_arg('--ndata', type_=int, default=2)
    printTEST('[TEST] build ndata=%d numpy arrays with shape=%r' % (ndata, shape))
    print(' * expected_memory(table_list) = %s' % utool.byte_str2(ndata * np.product(shape)))
    table_list = [np.empty(shape, dtype=dtype) for i in xrange(ndata)]
    print(' * memory+overhead(table_list) = %s' % utool.byte_str2(utool.get_object_size(table_list)))
    return table_list


def main(defaultdb='testdb', allow_newdir=False, **kwargs):
    from ibeis.dev import main_api
    from ibeis.dev import params
    printTEST('[TEST] Executing main. defaultdb=%r' % defaultdb)
    if defaultdb == 'testdb':
        allow_newdir = True
        defaultdbdir = join(params.get_workdir(), 'testdb')
        utool.ensuredir(defaultdbdir)
        if utool.get_flag('--clean'):
            utool.util_path.remove_files_in_dir(defaultdbdir, dryrun=False)
    main_locals = main_api.main(defaultdb=defaultdb, allow_newdir=allow_newdir, **kwargs)
    return main_locals


def main_loop(main_locals, **kwargs):
    from ibeis.dev import main_api
    printTEST('[TEST] TEST_LOOP')
    parent_locals = utool.get_parent_locals()
    parent_globals = utool.get_parent_globals()
    main_locals.update(parent_locals)
    main_locals.update(parent_globals)
    main_api.main_loop(main_locals, **kwargs)


def printTEST(msg, wait=False):
    __builtin__.print('\n=============================')
    __builtin__.print(msg)
    if INTERACTIVE and wait:
        raw_input('press enter to continue')
