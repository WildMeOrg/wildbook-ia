from __future__ import absolute_import, division, print_function
import __builtin__
import sys
sys.argv.append('--strict')  # Tests are always strict
VERBOSE = '--verbose' in sys.argv
QUIET   = '--quiet' in sys.argv

import functools
from os.path import realpath, dirname, join, exists
import numpy as np


def ensure_utool_in_pythonpath():
    utool_path = realpath(join(dirname(__file__), '..'))
    if VERBOSE:
        print('[test] appending to pythonpath: %r' % utool_path)
    try:
        assert exists(join(utool_path, 'utool')), ('cannot find utool in: %r' % utool_path)
    except AssertionError as ex:
        print('Caught Assertion Error: %s' % (ex))
        utool_path = join('..', utool_path)
        assert exists(join(utool_path, 'utool')), ('cannot find utool in: %r' % utool_path)
    sys.path.append(realpath(utool_path))

ensure_utool_in_pythonpath()

import utool
utool.util_sysreq.ensure_in_pythonpath('hesaff')
utool.util_sysreq.ensure_in_pythonpath('ibeis')
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[__testing__]')
from ibeis.dev import main_api
from ibeis.dev import params
import pyhesaff


INTERACTIVE = utool.get_flag(('--interactive', '-i'))


class MyException(Exception):
    pass


# Oooh hacky way to make test context take an arg or not
def testcontext2(name):
    def test_wrapper2(func):
        func.func_name = name
        @testcontext
        @functools.wraps(func)
        def test_wrapper3(*args, **kwargs):
            return func(*args, **kwargs)
        return test_wrapper3
    return test_wrapper2


def testcontext(func):
    @functools.wraps(func)
    def test_wrapper(*args, **kwargs):
        with utool.Indenter('[' + func.func_name.lower().replace('test_', '') + ']'):
            try:
                printTEST('[TEST] %s BEGIN' % (func.func_name,))
                test_locals = func(*args, **kwargs)
                printTEST('[TEST] %s FINISH -- SUCCESS' % (func.func_name,))
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
                # Build big execstring that you return in the locals dict
                if not isinstance(test_locals, dict):
                    test_locals = {}
                locals_execstr = utool.execstr_dict(test_locals, 'test_locals')
                embed_execstr  = utool.execstr_embed()
                ifs_execstr = '''
                if utool.get_flag(('--wait', '-w')):
                    print('waiting')
                    in_ = raw_input('press enter')
                if utool.get_flag('--cmd2') or locals().get('in_', '') == 'cmd':
                '''
                execstr = (utool.unindent(ifs_execstr)  + '\n' +
                           utool.indent(locals_execstr) + '\n' +
                           utool.indent(embed_execstr))
                test_locals['execstr'] = execstr
                return test_locals
            except Exception as ex:
                exc_type, exc_value, tb = sys.exc_info()
                # Get locals in the wrapped function
                locals_ = tb.tb_next.tb_frame.f_locals
                printTEST('[TEST] %s FINISH -- FAILED: %s %s' % (func.func_name, type(ex), ex))
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
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    raise exc_type, exc_value, exc_traceback.tb_next
    return test_wrapper


def get_pyhesaff_test_image_paths(ndata, lena=True, zebra=False, jeff=False):
    #root = utool.getroot()
    imgdir = dirname(pyhesaff.__file__)
    gname_list = utool.flatten([
        ['lena.png']  * utool.get_flag('--lena',   lena, help_='add lena to test images'),
        ['zebra.png'] * utool.get_flag('--zebra', zebra, help_='add zebra to test images'),
        ['test.png']  * utool.get_flag('--jeff',   jeff, help_='add jeff to test images'),
    ])
    # Build gpath_list
    if ndata == 0:
        gname_list = ['test.png']
    else:
        if ndata is None:
            ndata = 1
        gname_list = utool.util_list.flatten([gname_list] * ndata)
    gpath_list = [join(imgdir, path) for path in gname_list]
    return gpath_list


def get_pyhesaff_test_gpaths(**kwargs):
    return get_pyhesaff_test_image_paths(**kwargs)  # NOQA


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
    printTEST('[TEST] Executing main. defaultdb=%r' % defaultdb)
    known_testdbs = ['testdb', 'test_big_ibeis']
    if defaultdb in known_testdbs:
        allow_newdir = True
        defaultdbdir = join(params.get_workdir(), defaultdb)
        utool.ensuredir(defaultdbdir)
        if utool.get_flag('--clean'):
            utool.util_path.remove_files_in_dir(defaultdbdir, dryrun=False)
    if utool.get_flag('--clean'):
        sys.exit(0)
    main_locals = main_api.main(defaultdb=defaultdb, allow_newdir=allow_newdir, **kwargs)
    return main_locals


def main_loop(main_locals, **kwargs):
    printTEST('[TEST] TEST_LOOP')
    parent_locals = utool.get_parent_locals()
    parent_globals = utool.get_parent_globals()
    main_locals.update(parent_locals)
    main_locals.update(parent_globals)
    main_api.main_loop(main_locals, **kwargs)


def printTEST(msg, wait=False):
    __builtin__.print('\n=============================')
    __builtin__.print('**' + msg)
    if INTERACTIVE and wait:
        raw_input('press enter to continue')


def execfunc(*args, **kwargs):
    from drawtool import draw_func2 as df2
    return df2.present
