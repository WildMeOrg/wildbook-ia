from __future__ import absolute_import, division, print_function
import fnmatch
import inspect
import numpy as np
import sys
import shelve
import textwrap
import types
from . import util_inject
from .util_arg import get_flag
from .util_inject import inject
from .util_list import is_listlike, list_eq
from .util_print import Indenter
from .util_str import pack_into, truncate_str, horiz_string, indent
from .util_type import get_type
print, print_, printDBG, rrr, profile = inject(__name__, '[dbg]')

# --- Exec Strings ---
IPYTHON_EMBED_STR = r'''
try:
    import IPython
    print('Presenting in new ipython shell.')
    embedded = True
    IPython.embed()
except Exception as ex:
    warnings.warn(repr(ex)+'\n!!!!!!!!')
    embedded = False
'''


def execstr_embed():
    return IPYTHON_EMBED_STR


def ipython_execstr2():
    return textwrap.dedent(r'''
    import sys
    embedded = False
    try:
        __IPYTHON__
        in_ipython = True
    except NameError:
        in_ipython = False
    try:
        import IPython
        have_ipython = True
    except NameError:
        have_ipython = False
    if in_ipython:
        print('Presenting in current ipython shell.')
    elif '--cmd' in sys.argv:
        print('[utool.dbg] Requested IPython shell with --cmd argument.')
        if have_ipython:
            print('[utool.dbg] Found IPython')
            try:
                import IPython
                print('[utool.dbg] Presenting in new ipython shell.')
                embedded = True
                IPython.embed()
            except Exception as ex:
                print(repr(ex)+'\n!!!!!!!!')
                embedded = False
        else:
            print('[utool.dbg] IPython is not installed')
    ''')


def ipython_execstr():
    return textwrap.dedent(r'''
    import sys
    embedded = False
    if '-w' in sys.argv or '--wait' in sys.argv:
        print('waiting')
        in_ = raw_input('press enter')
    if '--cmd' in sys.argv or locals().get('in_', '') == 'cmd':
        print('[utool.dbg] Requested IPython shell with --cmd argument.')
        try:
            __IPYTHON__
            print('[ipython_execstr] Already in IPython!')
        except NameError:
            try:
                import IPython
                print('[utool.dbg] Presenting in new ipython shell.')
                embedded = True
                IPython.embed()
            except Exception as ex:
                print('[ipython_execstr]: Error: ' + str(type(ex)) + str(ex))
                raise
    ''')

#if 'PyQt4' in sys.modules:
    #from PyQt4 import QtCore
    #from IPython.lib.inputhook import enable_qt4
    #from IPython.lib.guisupport import start_event_loop_qt4
    #qapp = QtCore.QCoreApplication.instance()
    ##qapp.exec_()
    #print('[utool.dbg] Starting ipython qt4 hook')
    #enable_qt4()
    #start_event_loop_qt4(qapp)


def execstr_parent_locals():
    parent_locals = get_parent_locals()
    return execstr_dict(parent_locals, 'parent_locals')


def execstr_attr_list(obj_name, attr_list=None):
    #if attr_list is None:
        #exec(execstr_parent_locals())
        #exec('attr_list = dir('+obj_name+')')
    execstr_list = [obj_name + '.' + attr for attr in attr_list]
    return execstr_list


def execstr_dict(dict_, local_name, exclude_list=None):
    #if local_name is None:
        #local_name = dict_
        #exec(execstr_parent_locals())
        #exec('dict_ = local_name')
    if exclude_list is None:
        execstr = '\n'.join((key + ' = ' + local_name + '[' + repr(key) + ']'
                            for (key, val) in dict_.iteritems()))
    else:
        if not isinstance(exclude_list, list):
            exclude_list = [exclude_list]
        exec_list = []
        for (key, val) in dict_.iteritems():
            if not any((fnmatch.fnmatch(key, pat) for pat in iter(exclude_list))):
                exec_list.append(key + ' = ' + local_name + '[' + repr(key) + ']')
        execstr = '\n'.join(exec_list)
    return execstr


def execstr_func(func):
    print(' ! Getting executable source for: ' + func.func_name)
    _src = inspect.getsource(func)
    execstr = textwrap.dedent(_src[_src.find(':') + 1:])
    # Remove return statments
    while True:
        stmtx = execstr.find('return')  # Find first 'return'
        if stmtx == -1:
            break  # Fail condition
        # The characters which might make a return not have its own line
        stmt_endx = len(execstr) - 1
        for stmt_break in '\n;':
            print(execstr)
            print('')
            print(stmtx)
            stmt_endx_new = execstr[stmtx:].find(stmt_break)
            if -1 < stmt_endx_new < stmt_endx:
                stmt_endx = stmt_endx_new
        # now have variables stmt_x, stmt_endx
        before = execstr[:stmtx]
        after  = execstr[stmt_endx:]
        execstr = before + after
    return execstr


def execstr_src(func):
    return execstr_func(func)


def save_testdata(*args, **kwargs):
    uid = kwargs.get('uid', '')
    shelf_fname = 'test_data_%s.shelf' % uid
    shelf = shelve.open(shelf_fname)
    locals_ = get_parent_locals()
    print('save_testdata(%r)' % (args,))
    for key in args:
        shelf[key] = locals_[key]
    shelf.close()


def load_testdata(*args, **kwargs):
    uid = kwargs.get('uid', '')
    shelf_fname = 'test_data_%s.shelf' % uid
    shelf = shelve.open(shelf_fname)
    ret = [shelf[key] for key in args]
    shelf.close()
    if len(ret) == 1:
        ret = ret[0]
    print('load_testdata(%r)' % (args,))
    return ret


def import_testdata():
    shelf = shelve.open('test_data.shelf')
    print('importing\n * ' + '\n * '.join(shelf.keys()))
    shelf_exec = execstr_dict(shelf, 'shelf')
    exec(shelf_exec)
    shelf.close()
    return import_testdata.func_code.co_code


def embed(parent_locals=None, parent_globals=None, exec_lines=None):
    if parent_locals is None:
        parent_locals = get_parent_locals()
    if parent_globals is None:
        parent_globals = get_parent_globals()
    exec(execstr_dict(parent_globals, 'parent_globals'))
    exec(execstr_dict(parent_locals,  'parent_locals'))
    print('')
    print('[util] embedding')
    import IPython
    try:
        pass
        # make qt not loop forever (I had qflag loop forever with this off)
        from PyQt4.QtCore import pyqtRemoveInputHook
        pyqtRemoveInputHook()
    except ImportError as ex:
        print(ex)
    config_dict = {}
    #if exec_lines is not None:
    #    config_dict['exec_lines'] = exec_lines
    IPython.embed(**config_dict)


def quitflag(num=None, embed_=False, parent_locals=None, parent_globals=None):
    if num is None or get_flag('--quit' + str(num)):
        if parent_locals is None:
            parent_locals = get_parent_locals()
        if parent_globals is None:
            parent_globals = get_parent_globals()
        exec(execstr_dict(parent_locals, 'parent_locals'))
        exec(execstr_dict(parent_globals, 'parent_globals'))
        if embed_:
            print('Triggered --quit' + str(num))
            embed(parent_locals=parent_locals,
                  parent_globals=parent_globals)
        print('Triggered --quit' + str(num))
        sys.exit(1)


def qflag(num=None, embed_=True):
    return quitflag(num, embed_=embed_,
                    parent_locals=get_parent_locals(),
                    parent_globals=get_parent_globals())


def quit(num=None, embed_=False):
    return quitflag(num, embed_=embed_,
                    parent_locals=get_parent_locals(),
                    parent_globals=get_parent_globals())


def inIPython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def haveIPython():
    try:
        import IPython  # NOQA
        return True
    except NameError:
        return False


def print_frame(frame):
    frame = frame if 'frame' in vars() else inspect.currentframe()
    attr_list = ['f_code.co_name', 'f_back', 'f_lineno',
                 'f_code.co_names', 'f_code.co_filename']
    obj_name = 'frame'
    execstr_print_list = ['print("%r=%%r" %% (%s,))' % (_execstr, _execstr)
                          for _execstr in execstr_attr_list(obj_name, attr_list)]
    execstr = '\n'.join(execstr_print_list)
    exec(execstr)
    local_varnames = pack_into('; '.join(frame.f_locals.keys()))
    print(local_varnames)
    print('--- End Frame ---')


def search_stack_for_localvar(varname):
    curr_frame = inspect.currentframe()
    print(' * Searching parent frames for: ' + str(varname))
    frame_no = 0
    while curr_frame.f_back is not None:
        if varname in curr_frame.f_locals.keys():
            print(' * Found in frame: ' + str(frame_no))
            return curr_frame.f_locals[varname]
        frame_no += 1
        curr_frame = curr_frame.f_back
    print('... Found nothing in all ' + str(frame_no) + ' frames.')
    return None


def search_stack_for_var(varname):
    curr_frame = inspect.currentframe()
    print(' * Searching parent frames for: ' + str(varname))
    frame_no = 0
    while curr_frame.f_back is not None:
        if varname in curr_frame.f_locals.keys():
            print(' * Found local in frame: ' + str(frame_no))
            return curr_frame.f_locals[varname]
        if varname in curr_frame.f_globals.keys():
            print(' * Found global in frame: ' + str(frame_no))
            return curr_frame.f_globals[varname]
        frame_no += 1
        curr_frame = curr_frame.f_back
    print('... Found nothing in all ' + str(frame_no) + ' frames.')
    return None


def get_stack_frame(N=0):
    frame_level0 = inspect.currentframe()
    frame_cur = frame_level0
    for _ix in xrange(N + 1):
        frame_next = frame_cur.f_back
        if frame_next is None:
            raise AssertionError('Frame level %r is root' % _ix)
        frame_cur = frame_next
    return frame_cur


def get_parent_frame(N=0):
    parent_frame = get_stack_frame(N=N + 2)
    return parent_frame


def get_parent_locals(N=0):
    parent_frame = get_parent_frame(N=N + 1)
    return parent_frame.f_locals


def get_parent_globals(N=0):
    parent_frame = get_stack_frame(N=N + 1)
    return parent_frame.f_globals


def get_caller_locals(N=0):
    """ returns the locals of the function that called you """
    locals_ = get_parent_locals(N=N + 1)
    return locals_


def get_caller_prefix(N=0, aserror=False):
    prefix_fmt = '[!%s]' if aserror else '[%s]'
    prefix = prefix_fmt % (get_caller_name(N=N + 1),)
    return prefix


def get_caller_name(N=0):
    """ returns the name of the function that called you """
    parent_frame = get_parent_frame(N=N + 1)
    caller_name = parent_frame.f_code.co_name
    return caller_name


def module_functions(module):
    module_members = inspect.getmembers(module)
    function_list = []
    for key, val in module_members:
        if inspect.isfunction(val) and inspect.getmodule(val) == module:
            function_list.append((key, val))
    return function_list


def public_attributes(input):
    public_attr_list = []
    all_attr_list = dir(input)
    for attr in all_attr_list:
        if attr.find('__') == 0:
            continue
        public_attr_list.append(attr)
    return public_attr_list


def explore_stack():
    stack = inspect.stack()
    tup = stack[0]
    for ix, tup in reversed(list(enumerate(stack))):
        frame = tup[0]
        print('--- Frame %2d: ---' % (ix))
        print_frame(frame)
        print('\n')
        #next_frame = curr_frame.f_back


def explore_module(module_, seen=None, maxdepth=2, nonmodules=False):
    def __childiter(module):
        for aname in iter(dir(module)):
            if aname.find('_') == 0:
                continue
            try:
                yield module.__dict__[aname], aname
            except KeyError as ex:
                print(repr(ex))
                pass

    def __explore_module(module, indent, seen, depth, maxdepth, nonmodules):
        valid_children = []
        ret = u''
        modname = str(module.__name__)
        #modname = repr(module)
        for child, aname in __childiter(module):
            try:
                childtype = type(child)
                if not isinstance(childtype, types.ModuleType):
                    if nonmodules:
                        #print_(depth)
                        fullstr = indent + '    ' + str(aname) + ' = ' + repr(child)
                        truncstr = truncate_str(fullstr) + '\n'
                        ret +=  truncstr
                    continue
                childname = str(child.__name__)
                if seen is not None:
                    if childname in seen:
                        continue
                    elif maxdepth is None:
                        seen.add(childname)
                if childname.find('_') == 0:
                    continue
                valid_children.append(child)
            except Exception as ex:
                print(repr(ex))
                pass
        # Print
        # print_(depth)
        ret += indent + modname + '\n'
        # Recurse
        if maxdepth is not None and depth >= maxdepth:
            return ret
        ret += ''.join([__explore_module(child,
                                         indent + '    ',
                                         seen, depth + 1,
                                         maxdepth,
                                         nonmodules)
                       for child in iter(valid_children)])
        return ret
    #ret +=
    #print('#module = ' + str(module_))
    ret = __explore_module(module_, '     ', seen, 0, maxdepth, nonmodules)
    #print(ret)
    sys.stdout.flush()
    return ret


def debug_npstack(stacktup):
    print('Debugging numpy [hv]stack:')
    print('len(stacktup) = %r' % len(stacktup))
    for count, item in enumerate(stacktup):
        if isinstance(item, np.ndarray):
            print(' * item[%d].shape = %r' % (count, item.shape))
        elif isinstance(item, list) or isinstance(item, tuple):
            print(' * len(item[%d]) = %d' % (count, len(item)))
            print(' * DEBUG LIST')
            with Indenter(' * '):
                debug_list(item)
        else:
            print(' *  type(item[%d]) = %r' % (count, type(item)))


def debug_list(list_):
    dbgmessage = []
    append = dbgmessage.append
    append('debug_list')
    dim2 = None
    if all([is_listlike(item) for item in list_]):
        append(' * list items are all listlike')
        all_lens = [len(item) for item in list_]
        if list_eq(all_lens):
            dim2 = all_lens[0]
            append(' * uniform lens=%d' % dim2)
        else:
            append(' * nonuniform lens = %r' % np.unique(all_lens).tolist())
    else:
        all_types = [type(item) for item in list_]
        if list_eq(all_types):
            append(' * uniform types=%r' % all_types[0])
        else:
            append(' * nonuniform types: %r' % np.unique(all_types).tolist())
    print('\n'.join(dbgmessage))
    return dim2


def debug_hstack(stacktup):
    try:
        return np.hstack(stacktup)
    except ValueError as ex:
        print('ValueError in debug_hstack: ' + str(ex))
        debug_npstack(stacktup)
        raise


def debug_vstack(stacktup):
    try:
        return np.vstack(stacktup)
    except ValueError as ex:
        print('ValueError in debug_vstack: ' + str(ex))
        debug_npstack(stacktup)
        raise


def debug_exception(func):
    def ex_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            print('[tools] ERROR: %s(%r, %r)' % (func.func_name, args, kwargs))
            print('[tools] ERROR: %r' % ex)
            raise
    ex_wrapper.func_name = func.func_name
    return ex_wrapper


def get_func_name(func):
    """ Works on must functionlike objects including str, which has no func_name """
    try:
        return func.func_name
    except AttributeError:
        if isinstance(func, type):
            return repr(str).replace('<type \'', '').replace('\'>', '')
        else:
            raise NotImplementedError(('cannot get func_name of func=%r'
                                       'type(func)=%r') % (func, type(func)))


def printex(ex, msg='[!?] Caught exception',
            prefix=None, key_list=[], locals_=None):
    """ Prints an exception with relevant info """
    if prefix is None:
        prefix = get_caller_prefix(aserror=True)
    if locals_ is None:
        locals_ = get_caller_locals()
    exstr = formatex(ex, msg, prefix, key_list, locals_)
    print(exstr)


def formatex(ex, msg='[!?] Caught exception',
             prefix=None, key_list=[], locals_=None):
    """ Formats an exception with relevant info """
    ex_str = []
    append_exstr = ex_str.append
    if prefix is None:
        prefix = get_caller_prefix(aserror=True)
    if locals_ is None:
        locals_ = get_caller_locals()
    append_exstr('<!!! EXCEPTION !!!>')
    append_exstr(prefix + ' ' + msg + '%s: %s' % (type(ex), ex))
    for key in key_list:
        if isinstance(key, tuple):
            func = key[0]
            key = key[1]
            assert key in locals_
            val = locals_[key]
            funcvalstr = str(func(val))
            append_exstr('%s %s(%s) = %s' % (prefix, get_func_name(func), key, funcvalstr))
        elif key in locals_:
            valstr = truncate_str(repr(locals_[key]), maxlen=200)
            append_exstr('%s %s = %s' % (prefix, key, valstr))
        else:
            append_exstr('%s !!! %s not populated!' % (prefix, key))
    append_exstr('</!!! EXCEPTION !!!>')
    return '\n'.join(ex_str)


def get_reprs(*args, **kwargs):
    if 'locals_' in kwargs:
        locals_ = kwargs['locals_']
    else:
        locals_ = locals()
        locals_.update(get_caller_locals())

    msg_list = []
    var_list = list(args) + kwargs.get('var_list', [])
    for key in var_list:
        var = locals_[key]
        msg = horiz_string(str(key) + ' = ', repr(var))
        msg_list.append(msg)

    reprs = '\n' + indent('\n##\n'.join(msg_list)) + '\n'
    return reprs


def printvar2(varstr, attr=''):
    locals_ = get_parent_locals()
    printvar(locals_, varstr, attr)


def printvar(locals_, varname, attr='.shape'):
    npprintopts = np.get_printoptions()
    np.set_printoptions(threshold=5)
    dotpos = varname.find('.')
    # Locate var
    if dotpos == -1:
        var = locals_[varname]
    else:
        varname_ = varname[:dotpos]
        dotname_ = varname[dotpos:]
        var_ = locals_[varname_]  # NOQA
        var = eval('var_' + dotname_)
    # Print in format
    typestr = get_type(var)
    if isinstance(var, np.ndarray):
        varstr = eval('str(var' + attr + ')')
        print('[var] %s %s = %s' % (typestr, varname + attr, varstr))
    elif isinstance(var, list):
        if attr == '.shape':
            func = 'len'
        else:
            func = ''
        varstr = eval('str(' + func + '(var))')
        print('[var] %s len(%s) = %s' % (typestr, varname, varstr))
    else:
        print('[var] %s %s = %r' % (typestr, varname, var))
    np.set_printoptions(**npprintopts)


def dict_dbgstr(dict_name, locals_=None):
    if locals_ is None:
        locals_ = get_parent_locals()
    lenstr = len_dbgstr(dict_name, locals_)
    keystr = keys_dbgstr(dict_name, locals_)
    return keystr + ' ' + lenstr
    #printvar(locals_, dict_name)


def keys_dbgstr(dict_name, locals_=None):
    if locals_ is None:
        locals_ = get_parent_locals()
    dict_ = locals_[dict_name]
    key_str = dict_name + '.keys() = ' + repr(dict_.keys())
    return key_str
    #dict_ = locals_[dict_name]


def print_varlen(name_, locals_=None):
    if locals_ is None:
        locals_ = get_parent_locals()
    prefix = get_caller_prefix()
    print(prefix + ' ' + len_dbgstr(name_, locals_))


def len_dbgstr(lenable_name, locals_=None):
    try:
        if locals_ is None:
            locals_ = get_parent_locals()
        lenable_ = locals_[lenable_name]
    except Exception:
        exec(execstr_dict(locals_, 'locals_'))
        try:
            lenable_ = eval(lenable_name)
        except Exception as ex:
            print('locals.keys = %r' % (locals_.keys(),))
            printex(ex, '[!util_dbg]')
            raise Exception('Cannot lendbg: %r' % lenable_name)
    len_str = 'len(%s) = %d' % (lenable_name, len(lenable_))
    return len_str


def list_dbgstr(list_name, trunc=2):
    locals_ = get_parent_locals()
    list_   = locals_[list_name]
    if trunc is None:
        pos = len(list_)
    else:
        pos     = min(trunc, len(list_) - 1)
    list_str = list_name + ' = ' + repr(list_[0:pos],)
    return list_str


def all_rrr():
    raise NotImplementedError('!!! STOP !!!')
    util_inject.inject_all()
    for mod in util_inject.get_injected_modules():
        try:
            if hasattr(mod, 'rrr'):
                mod.rrr()
        except Exception as ex:
            print(ex)
            print('mod = %r ' % mod)
