from __future__ import absolute_import, division, print_function
import __builtin__
import sys


__DEBUG_ALL__ = '--debug-all' in sys.argv
__DEBUG_PROF__ = '--debug-prof' in sys.argv or '--debug-profile' in sys.argv


# Read all flags with --debug in them
ARGV_DEBUG_FLAGS = []
for argv in sys.argv:
    if argv.startswith('--debug'):
        ARGV_DEBUG_FLAGS.append(argv.replace('--debug', '').strip('-'))


#print('ARGV_DEBUG_FLAGS: %r' % (ARGV_DEBUG_FLAGS,))


__STDOUT__ = sys.stdout
__PRINT_FUNC__     = __builtin__.print
__PRINT_DBG_FUNC__ = __builtin__.print
__WRITE_FUNC__ = __STDOUT__.write
__FLUSH_FUNC__ = __STDOUT__.flush
__RELOAD_OK__  = '--noreloadable' not in sys.argv


__INJECTED_MODULES__ = set([])

# Do not inject into these modules
__INJECT_BLACKLIST__ = frozenset(['tri', 'gc', 'sys', 'string', 'types', '_dia', 'responce', 'six', __name__])


def _inject_funcs(module, *func_list):
    for func in func_list:
        if (module is not None and
                hasattr(module, '__name__') and
                module.__name__ not in __INJECT_BLACKLIST__ and
                not module.__name__.startswith('six') and
                not module.__name__.startswith('sys')):
            #print('setting: %s.%s = %r' % (module.__name__, func.func_name, func))
            setattr(module, func.func_name, func)


def _add_injected_module(module):
    global __INJECTED_MODULES__
    __INJECTED_MODULES__.add(module)


def get_injected_modules():
    return list(__INJECTED_MODULES__)


def _get_module(module_name=None, module=None):
    if module is None and module_name is not None:
        try:
            module = sys.modules[module_name]
        except KeyError as ex:
            print(ex)
            raise KeyError(('module_name=%r must be loaded before ' +
                            'receiving injections') % module_name)
    elif module is not None and module_name is None:
        pass
    else:
        raise ValueError('module_name or module must be exclusively specified')
    _add_injected_module(module)
    return module


def inject_colored_exceptions():
    import sys
    def myexcepthook(type, value, tb):
        #https://stackoverflow.com/questions/14775916/coloring-exceptions-from-python-on-a-terminal
        import traceback
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import TerminalFormatter
        tbtext = ''.join(traceback.format_exception(type, value, tb))
        lexer = get_lexer_by_name("pytb", stripall=True)
        formatter = TerminalFormatter(bg="dark")
        sys.stderr.write(highlight(tbtext, lexer, formatter))
    if not sys.platform.startswith('win32'):
        sys.excepthook = myexcepthook


def inject_print_functions(module_name=None, module_prefix='[???]', DEBUG=False, module=None):
    module = _get_module(module_name, module)

    def print(msg):
        __PRINT_FUNC__(msg)

    def print_(msg):
        __WRITE_FUNC__(msg)

    # turn on module debugging with command line flags
    dotpos = module.__name__.rfind('.')
    if dotpos == -1:
        module_name = module.__name__
    else:
        module_name = module.__name__[dotpos + 1:]
    def _replchars(str_):
        return str_.replace('_', '-').replace(']', '').replace('[', '')
    flag1 = '--debug-%s' % _replchars(module_name)
    flag2 = '--debug-%s' % _replchars(module_prefix)
    DEBUG_FLAG = any([flag in sys.argv for flag in [flag1, flag2]])
    for curflag in ARGV_DEBUG_FLAGS:
        if curflag in module_prefix:
            DEBUG_FLAG = True
    if __DEBUG_ALL__ or DEBUG or DEBUG_FLAG:
        print('DEBUGGING: %r == %r' % (module_name, module_prefix))
        def printDBG(msg):
            __PRINT_DBG_FUNC__(module_prefix + ' DEBUG ' + msg)
    else:
        def printDBG(msg):
            pass

    _inject_funcs(module, print, print_, printDBG)
    return print, print_, printDBG


def inject_reload_function(module_name=None, module_prefix='[???]', module=None):
    'Injects dynamic module reloading'
    module = _get_module(module_name, module)
    if module_name is None:
        module_name = str(module.__name__)
    def rrr():
        """ Dynamic module reloading """
        if not __RELOAD_OK__:
            raise Exception('Reloading has been forced off')
        try:
            import imp
            __builtin__.print('RELOAD: ' + str(module_prefix) + ' __name__=' + module_name)
            imp.reload(module)
        except Exception as ex:
            print(ex)
            print('%s Failed to reload' % module_prefix)
            raise
    _inject_funcs(module, rrr)
    return rrr


def inject_profile_function(module_name=None, module_prefix='[???]', module=None):
    module = _get_module(module_name, module)
    try:
        profile = getattr(__builtin__, 'profile')
        if __DEBUG_PROF__:
            print('[util_inject] PROFILE ON: %r' % module)
        return profile
    except AttributeError:
        def profile(func):
            return func
        if __DEBUG_PROF__:
            print('[util_inject] PROFILE OFF: %r' % module)
    _inject_funcs(module, profile)
    return profile


def inject(module_name=None, module_prefix='[???]', DEBUG=False, module=None):
    '''
    Usage:
        from __future__ import absolute_import, division, print_function
        from util.util_inject import inject
        print, print_, printDBG, rrr, profile = inject(__name__, '[mod]')
    '''
    module = _get_module(module_name, module)
    rrr         = inject_reload_function(None, module_prefix, module)
    profile_    = inject_profile_function(None, module_prefix, module)
    print_funcs = inject_print_functions(None, module_prefix, DEBUG, module)
    print, print_, printDBG = print_funcs
    return print, print_, printDBG, rrr, profile_


def inject_all(DEBUG=False):
    '''
    Injects the print, print_, printDBG, rrr, and profile functions into all
    loaded modules
    '''
    raise NotImplemented('!!!')
    for key, module in sys.modules.items():
        if module is None or not hasattr(module, '__name__'):
            continue
        try:
            module_prefix = '[%s]' % key
            inject(module_name=key, module_prefix=module_prefix, DEBUG=DEBUG)
        except Exception as ex:
            print('<!!!>')
            print('[util_inject] Cannot Inject: %s: %s' % (type(ex), ex))
            print('[util_inject] key=%r' % key)
            print('[util_inject] module=%r' % module)
            print('</!!!>')
            raise


print, print_, printDBG, rrr, profile = inject(__name__, '[inject]')
