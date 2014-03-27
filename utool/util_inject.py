from __future__ import division, print_function
import __builtin__
import sys


__DEBUG__ = '--debug' in sys.argv

__STDOUT__ = sys.stdout
__PRINT_FUNC__     = __builtin__.print
__PRINT_DBG_FUNC__ = __builtin__.print
__WRITE_FUNC__ = __STDOUT__.write
__FLUSH_FUNC__ = __STDOUT__.flush


__INJECTED_MODULES__ = set([])


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


def _inject_colored_exception_hook():
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

    if __DEBUG__ or DEBUG:
        def printDBG(msg):
            __PRINT_DBG_FUNC__(module_prefix + ' DEBUG ' + msg)
    else:
        def printDBG(msg):
            pass

    module.print  = print
    module.print_ = print_
    module.printDBG = printDBG
    return print, print_, printDBG


def inject_reload_function(module_name=None, module_prefix='[???]', module=None):
    'Injects dynamic module reloading'
    module = _get_module(module_name, module)
    def rrr():
        'Dynamic module reloading'
        import imp
        __builtin__.print(module_prefix + ' reloading ' + module_name)
        imp.reload(module)
    module.rrr = rrr


def inject_profile_function(module_name=None, module_prefix='[???]', module=None):
    module = _get_module(module_name, module)
    try:
        profile = module.profile
    except AttributeError:
        profile = lambda func: func
    module.profile = profile
    return profile


def inject(module_name=None, module_prefix='[???]', DEBUG=False, module=None):
    '''
    Usage:
        from __future__ import print_function, division
        from util.util_inject import inject
        print, print_, printDBG, rrr, profile = inject(__name__, '[mod]')
    '''
    rrr         = inject_reload_function(module_name, module_prefix, module)
    profile     = inject_profile_function(module_name, module_prefix, module)
    print_funcs = inject_print_functions(module_name, module_prefix, DEBUG, module)
    print, print_, printDBG = print_funcs
    return print, print_, printDBG, rrr, profile


def inject_all(DEBUG=False):
    '''
    Injects the print, print_, printDBG, rrr, and profile functions into all
    loaded modules
    '''
    for key, val in sys.modules.items():
        if val is None:
            continue
        try:
            module_prefix = '[%s]' % key
            inject(module_name=key, module_prefix=module_prefix, DEBUG=DEBUG)
        except Exception as ex:
            print(ex)
            print('key=%r' % key)
            print('val=%r' % val)
            raise


print, print_, printDBG, rrr, profile = inject(__name__, '[inject]')
