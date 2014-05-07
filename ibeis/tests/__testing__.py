from __future__ import absolute_import, division, print_function
import __builtin__

try:
    from . import __sysreq__
except Exception:
    import __sysreq__  # NOQA

import ibeis
ibeis._preload()
import utool
import sys
import numpy as np
from vtool.tests import grabdata
from plottool import fig_presenter
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[__testing__]')

VERBOSE = '--verbose' in sys.argv
QUIET   = '--quiet' in sys.argv

INTERACTIVE = utool.get_flag(('--interactive', '-i'))


HAPPY_FACE = r'''
               .-""""""-.
             .'          '.
            /   O      O   \
           :                :
           |                |
           : ',          ,' :
            \  '-......-'  /
             '.          .'
               '-......-'
                   '''


SAD_FACE = r'''
               .-""""""-.
             .'          '.
            /   O      O   \
           :           `    :
           |                |
           :    .------.    :
            \  '        '  /
             '.          .'
               '-......-'
                  '''


def run_test(func, *args, **kwargs):
    """
    Runs the test function
    Input:
        Anything that needs to be passed to <func>
    Output:
        executable (python code) which add tests_locals dictionary keys to your
        local namespace
    """
    with utool.Indenter('[' + func.func_name.lower().replace('test_', '') + ']'):
        try:
            printTEST('[TEST] %s BEGIN' % (func.func_name,))
            test_locals = func(*args, **kwargs)
            printTEST('[TEST] %s FINISH -- SUCCESS' % (func.func_name,))
            print(HAPPY_FACE)
            return test_locals
        except Exception as ex:
            # Get locals in the wrapped function
            utool.printex(ex)
            exc_type, exc_value, tb = sys.exc_info()
            printTEST('[TEST] %s FINISH -- FAILED: %s %s' % (func.func_name, type(ex), ex))
            print(SAD_FACE)
            locals_ = tb.tb_next.tb_frame.f_locals
            ibs = locals_.get('ibs', None)
            if ibs is not None:
                ibs.db.dump()
            if '--strict' in sys.argv:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                raise exc_type, exc_value, exc_traceback.tb_next


def get_testdata_dir():
    imgdir = grabdata.get_testdata_dir()
    return imgdir


def get_test_gpaths(ndata=None, lena=True, zebra=False, jeff=False):
    # FIXME: The testdata no longer lives in hesaff
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


def main(defaultdb='testdb', allow_newdir=False, gui=False, **kwargs):
    # TODO remove all of this fluff
    printTEST('[TEST] Executing main. defaultdb=%r' % defaultdb)
    #from ibeis import params
    #known_testdbs = ['testdb', 'testdb_big']
    #if defaultdb in known_testdbs:
        #allow_newdir = True
        #defaultdbdir = join(sysres.get_workdir(), defaultdb)
        #utool.ensuredir(defaultdbdir)
        #if utool.get_flag('--clean'):
            #utool.util_path.remove_files_in_dir(defaultdbdir, dryrun=False)
    #if utool.get_flag('--clean'):
        #sys.exit(0)
    main_locals = ibeis.main(defaultdb=defaultdb, allow_newdir=allow_newdir, gui=gui, **kwargs)
    return main_locals


def main_loop(test_locals, rungui=False, **kwargs):
    """
    Runs ibs main loop (if applicable), does a present, and returns a useful
    execstr
    """
    printTEST('[TEST] TEST_LOOP')
    # Build big execstring that you return in the locals dict
    ipycmd_execstr = ibeis.main_loop(test_locals, rungui=rungui, **kwargs)
    if not '--noshow' in sys.argv:
        fig_presenter.present()
    if not isinstance(test_locals, dict):
        test_locals = {}
    locals_execstr = utool.execstr_dict(test_locals, 'test_locals')
    execstr = locals_execstr + '\n' + ipycmd_execstr
    return execstr


def printTEST(msg, wait=False):
    __builtin__.print('\n=============================')
    __builtin__.print('**' + msg)
    if INTERACTIVE and wait:
        raw_input('press enter to continue')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
