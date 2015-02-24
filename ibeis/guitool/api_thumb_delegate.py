"""
CommandLine:
    main.py --db PZ_MUGU_19 --eid 13 --verbthumb

Notes:
    http://stackoverflow.com/questions/8312725/how-to-create-executable-file-for-a-qt-application
    http://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt
    For windows need at least these dlls:
        mingwm10.dll
        libgcc_s_dw2-1.dll
        QtCore4.dll
        QtGui4.dll

Error Notes:
[guitool.qtapp_loop()] starting qt app loop: qwin=<ibeis.gui.newgui.IBEISMainWindow object at 0x7f556abcab00>
[guitool] qapp.setActiveWindow(qwin)
[guitool.qtapp_loop()] qapp.exec_()  # runing main loop
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 108, in myexcepthook
    tbtext = ''.join(traceback.format_exception(type, value, tb))
  File "/usr/lib/python2.7/traceback.py", line 141, in format_exception
    list = list + format_tb(tb, limit)
  File "/usr/lib/python2.7/traceback.py", line 76, in format_tb
    return format_list(extract_tb(tb, limit))
  File "/usr/lib/python2.7/traceback.py", line 101, in extract_tb
    line = linecache.getline(filename, lineno, f.f_globals)
  File "/usr/lib/python2.7/linecache.py", line 14, in getline
    lines = getlines(filename, module_globals)
  File "/usr/lib/python2.7/linecache.py", line 40, in getlines
    return updatecache(filename, module_globals)
  File "/usr/lib/python2.7/linecache.py", line 132, in updatecache
    with open(fullname, 'rU') as fp:
RuntimeError: maximum recursion depth exceeded

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 731, in index
    node = parent_node[row]
  File "/home/joncrall/code/guitool/guitool/api_tree_node.py", line 47, in __getitem__
    return self.get_child(index)
  File "/home/joncrall/code/guitool/guitool/api_tree_node.py", line 72, in get_child
    self.lazy_checks()
  File "/home/joncrall/code/guitool/guitool/api_tree_node.py", line 128, in lazy_checks
    if isinstance(self.child_nodes, GeneratorType):
RuntimeError: maximum recursion depth exceeded in __instancecheck__
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 108, in myexcepthook
    tbtext = ''.join(traceback.format_exception(type, value, tb))
  File "/usr/lib/python2.7/traceback.py", line 141, in format_exception
    list = list + format_tb(tb, limit)
  File "/usr/lib/python2.7/traceback.py", line 76, in format_tb
    return format_list(extract_tb(tb, limit))
  File "/usr/lib/python2.7/traceback.py", line 101, in extract_tb
    line = linecache.getline(filename, lineno, f.f_globals)
  File "/usr/lib/python2.7/linecache.py", line 14, in getline
    lines = getlines(filename, module_globals)
  File "/usr/lib/python2.7/linecache.py", line 40, in getlines
    return updatecache(filename, module_globals)
  File "/usr/lib/python2.7/linecache.py", line 132, in updatecache
    with open(fullname, 'rU') as fp:
RuntimeError: maximum recursion depth exceeded

---...
..
..

RuntimeError: maximum recursion depth exceeded
Error in APIThumbDelegate
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 109, in myexcepthook
    lexer = get_lexer_by_name('pytb', stripall=True)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 87, in get_lexer_by_name
    _load_lexers(module_name)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 41, in _load_lexers
    mod = __import__(module_name, None, None, ['__all__'])
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/python.py", line 14, in <module>
    from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 20, in <module>
    from pygments.filters import get_filter_by_name
  File "/usr/local/lib/python2.7/dist-packages/pygments/filters/__init__.py", line 13, in <module>
    import re
RuntimeError: maximum recursion depth exceeded while calling a Python object

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 324, in sizeHint
    utool.printex(ex, 'Error in APIThumbDelegate', tb=True)
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 877, in printex
    exstr = formatex(ex, msg, prefix, key_list, locals_, iswarning, tb=tb)
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 916, in formatex
    errstr_list.append(traceback.format_exc())
  File "/usr/lib/python2.7/traceback.py", line 242, in format_exc
    return ''.join(format_exception(etype, value, tb, limit))
  File "/usr/lib/python2.7/traceback.py", line 141, in format_exception
    list = list + format_tb(tb, limit)
  File "/usr/lib/python2.7/traceback.py", line 76, in format_tb
    return format_list(extract_tb(tb, limit))
  File "/usr/lib/python2.7/traceback.py", line 101, in extract_tb
    line = linecache.getline(filename, lineno, f.f_globals)
  File "/usr/lib/python2.7/linecache.py", line 14, in getline
    lines = getlines(filename, module_globals)
  File "/usr/lib/python2.7/linecache.py", line 40, in getlines
    return updatecache(filename, module_globals)
RuntimeError: maximum recursion depth exceeded
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 109, in myexcepthook
    lexer = get_lexer_by_name('pytb', stripall=True)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 87, in get_lexer_by_name
    _load_lexers(module_name)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 41, in _load_lexers
    mod = __import__(module_name, None, None, ['__all__'])
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/python.py", line 14, in <module>
    from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 20, in <module>
    from pygments.filters import get_filter_by_name
  File "/usr/local/lib/python2.7/dist-packages/pygments/filters/__init__.py", line 13, in <module>
    import re
RuntimeError: maximum recursion depth exceeded while calling a Python object

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 832, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 621, in _get_data
    utool.printex(ex, 'problem getting in column %r' % (col,))
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 887, in printex
    print_func('\n+------\n')
  File "/home/joncrall/code/utool/utool/util_inject.py", line 144, in print
    util_logging.__UTOOL_PRINT__(*args)
  File "/home/joncrall/code/utool/utool/util_logging.py", line 225, in utool_print
    return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
  File "/usr/lib/python2.7/logging/__init__.py", line 1151, in info
    if self.isEnabledFor(INFO):
  File "/usr/lib/python2.7/logging/__init__.py", line 1351, in isEnabledFor
    return level >= self.getEffectiveLevel()
RuntimeError: maximum recursion depth exceeded
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 109, in myexcepthook
    lexer = get_lexer_by_name('pytb', stripall=True)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 87, in get_lexer_by_name
    _load_lexers(module_name)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 41, in _load_lexers
    mod = __import__(module_name, None, None, ['__all__'])
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/python.py", line 14, in <module>
    from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 20, in <module>
    from pygments.filters import get_filter_by_name
  File "/usr/local/lib/python2.7/dist-packages/pygments/filters/__init__.py", line 13, in <module>
    import re
RuntimeError: maximum recursion depth exceeded while calling a Python object

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 813, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 621, in _get_data
    utool.printex(ex, 'problem getting in column %r' % (col,))
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 887, in printex
    print_func('\n+------\n')
  File "/home/joncrall/code/utool/utool/util_inject.py", line 144, in print
    util_logging.__UTOOL_PRINT__(*args)
  File "/home/joncrall/code/utool/utool/util_logging.py", line 225, in utool_print
    return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
  File "/usr/lib/python2.7/logging/__init__.py", line 1152, in info
    self._log(INFO, msg, args, **kwargs)
  File "/usr/lib/python2.7/logging/__init__.py", line 1270, in _log
    record = self.makeRecord(self.name, level, fn, lno, msg, args, exc_info, func, extra)
  File "/usr/lib/python2.7/logging/__init__.py", line 1244, in makeRecord
    rv = LogRecord(name, level, fn, lno, msg, args, exc_info, func)
  File "/usr/lib/python2.7/logging/__init__.py", line 266, in __init__
    self.levelname = getLevelName(level)
RuntimeError: maximum recursion depth exceeded
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 109, in myexcepthook
    lexer = get_lexer_by_name('pytb', stripall=True)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 87, in get_lexer_by_name
    _load_lexers(module_name)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 41, in _load_lexers
    mod = __import__(module_name, None, None, ['__all__'])
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/python.py", line 14, in <module>
    from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 24, in <module>
    from pygments.regexopt import regex_opt
  File "/usr/local/lib/python2.7/dist-packages/pygments/regexopt.py", line 19, in <module>
    CS_ESCAPE = re.compile(r'[\^\\\-\]]')
  File "/usr/lib/python2.7/re.py", line 190, in compile
    return _compile(pattern, flags)
  File "/usr/lib/python2.7/re.py", line 242, in _compile
    p = sre_compile.compile(pattern, flags)
  File "/usr/lib/python2.7/sre_compile.py", line 498, in compile
    p = sre_parse.parse(p, flags)
  File "/usr/lib/python2.7/sre_parse.py", line 678, in parse
    source = Tokenizer(str)
  File "/usr/lib/python2.7/sre_parse.py", line 181, in __init__
    self.__next()
RuntimeError: maximum recursion depth exceeded
Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 621, in _get_data
    utool.printex(ex, 'problem getting in column %r' % (col,))
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 887, in printex
    print_func('\n+------\n')
  File "/home/joncrall/code/utool/utool/util_inject.py", line 144, in print
    util_logging.__UTOOL_PRINT__(*args)
  File "/home/joncrall/code/utool/utool/util_logging.py", line 225, in utool_print
    return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
  File "/usr/lib/python2.7/logging/__init__.py", line 1152, in info
    self._log(INFO, msg, args, **kwargs)
  File "/usr/lib/python2.7/logging/__init__.py", line 1270, in _log
    record = self.makeRecord(self.name, level, fn, lno, msg, args, exc_info, func, extra)
  File "/usr/lib/python2.7/logging/__init__.py", line 1244, in makeRecord
    rv = LogRecord(name, level, fn, lno, msg, args, exc_info, func)
  File "/usr/lib/python2.7/logging/__init__.py", line 266, in __init__
    self.levelname = getLevelName(level)
RuntimeError: maximum recursion depth exceeded
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 109, in myexcepthook
    lexer = get_lexer_by_name('pytb', stripall=True)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 87, in get_lexer_by_name
    _load_lexers(module_name)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 41, in _load_lexers
    mod = __import__(module_name, None, None, ['__all__'])
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/python.py", line 14, in <module>
    from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 24, in <module>
    from pygments.regexopt import regex_opt
  File "/usr/local/lib/python2.7/dist-packages/pygments/regexopt.py", line 19, in <module>
    CS_ESCAPE = re.compile(r'[\^\\\-\]]')
  File "/usr/lib/python2.7/re.py", line 190, in compile
    return _compile(pattern, flags)
  File "/usr/lib/python2.7/re.py", line 242, in _compile
    p = sre_compile.compile(pattern, flags)
  File "/usr/lib/python2.7/sre_compile.py", line 498, in compile
    p = sre_parse.parse(p, flags)
  File "/usr/lib/python2.7/sre_parse.py", line 678, in parse
    source = Tokenizer(str)
  File "/usr/lib/python2.7/sre_parse.py", line 181, in __init__
    self.__next()
RuntimeError: maximum recursion depth exceeded

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 621, in _get_data
    utool.printex(ex, 'problem getting in column %r' % (col,))
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 887, in printex
    print_func('\n+------\n')
  File "/home/joncrall/code/utool/utool/util_inject.py", line 144, in print
    util_logging.__UTOOL_PRINT__(*args)
  File "/home/joncrall/code/utool/utool/util_logging.py", line 225, in utool_print
    return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
  File "/usr/lib/python2.7/logging/__init__.py", line 1152, in info
    self._log(INFO, msg, args, **kwargs)
  File "/usr/lib/python2.7/logging/__init__.py", line 1270, in _log
    record = self.makeRecord(self.name, level, fn, lno, msg, args, exc_info, func, extra)
  File "/usr/lib/python2.7/logging/__init__.py", line 1244, in makeRecord
    rv = LogRecord(name, level, fn, lno, msg, args, exc_info, func)
  File "/usr/lib/python2.7/logging/__init__.py", line 266, in __init__
    self.levelname = getLevelName(level)
RuntimeError: maximum recursion depth exceeded
Error in APIThumbDelegate
Traceback (most recent call last):
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 109, in myexcepthook
    lexer = get_lexer_by_name('pytb', stripall=True)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 88, in get_lexer_by_name
    return _lexer_cache[name](**options)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex '^(?=  File "[^"]+", line \\d+)' in state 'root' of <class 'pygments.lexers.python.PythonTracebackLexer'>: maximum recursion depth exceeded while calling a Python object

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 324, in sizeHint
    utool.printex(ex, 'Error in APIThumbDelegate', tb=True)
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 887, in printex
    print_func('\n+------\n')
  File "/home/joncrall/code/utool/utool/util_inject.py", line 144, in print
    util_logging.__UTOOL_PRINT__(*args)
  File "/home/joncrall/code/utool/utool/util_logging.py", line 225, in utool_print
    return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
  File "/usr/lib/python2.7/logging/__init__.py", line 1152, in info
    self._log(INFO, msg, args, **kwargs)
  File "/usr/lib/python2.7/logging/__init__.py", line 1271, in _log
    self.handle(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 1281, in handle
    self.callHandlers(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 1321, in callHandlers
    hdlr.handle(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 749, in handle
    self.emit(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 942, in emit
    StreamHandler.emit(self, record)
  File "/usr/lib/python2.7/logging/__init__.py", line 879, in emit
    self.handleError(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 802, in handleError
    None, sys.stderr)
  File "/usr/lib/python2.7/traceback.py", line 125, in print_exception
    print_tb(tb, limit, file)
  File "/usr/lib/python2.7/traceback.py", line 67, in print_tb
    '  File "%s", line %d, in %s' % (filename, lineno, name))
RuntimeError: maximum recursion depth exceeded
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 109, in myexcepthook
    lexer = get_lexer_by_name('pytb', stripall=True)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 88, in get_lexer_by_name
    return _lexer_cache[name](**options)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex '^(?=  File "[^"]+", line \\d+)' in state 'root' of <class 'pygments.lexers.python.PythonTracebackLexer'>: maximum recursion depth exceeded while calling a Python object

riginal exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 813, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 621, in _get_data
    utool.printex(ex, 'problem getting in column %r' % (col,))
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 887, in printex
    print_func('\n+------\n')
  File "/home/joncrall/code/utool/utool/util_inject.py", line 144, in print
    util_logging.__UTOOL_PRINT__(*args)
  File "/home/joncrall/code/utool/utool/util_logging.py", line 225, in utool_print
    return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
  File "/usr/lib/python2.7/logging/__init__.py", line 1152, in info
    self._log(INFO, msg, args, **kwargs)
  File "/usr/lib/python2.7/logging/__init__.py", line 1271, in _log
    self.handle(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 1281, in handle
    self.callHandlers(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 1321, in callHandlers
    hdlr.handle(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 749, in handle
    self.emit(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 942, in emit
    StreamHandler.emit(self, record)
  File "/usr/lib/python2.7/logging/__init__.py", line 879, in emit
    self.handleError(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 802, in handleError
    None, sys.stderr)
  File "/usr/lib/python2.7/traceback.py", line 125, in print_exception
    print_tb(tb, limit, file)
  File "/usr/lib/python2.7/traceback.py", line 68, in print_tb
    linecache.checkcache(filename)
  File "/usr/lib/python2.7/linecache.py", line 64, in checkcache
    if size != stat.st_size or mtime != stat.st_mtime:
RuntimeError: maximum recursion depth exceeded in cmp
Traceback (most recent call last):
  File "/usr/lib/python2.7/logging/__init__.py", line 851, in emit
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 109, in myexcepthook
    lexer = get_lexer_by_name('pytb', stripall=True)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 88, in get_lexer_by_name
    return _lexer_cache[name](**options)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex '^(?=  File "[^"]+", line \\d+)' in state 'root' of <class 'pygments.lexers.python.PythonTracebackLexer'>: maximum recursion depth exceeded while calling a Python object

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 621, in _get_data
    utool.printex(ex, 'problem getting in column %r' % (col,))
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 887, in printex
    print_func('\n+------\n')
  File "/home/joncrall/code/utool/utool/util_inject.py", line 144, in print
    util_logging.__UTOOL_PRINT__(*args)
  File "/home/joncrall/code/utool/utool/util_logging.py", line 225, in utool_print
    return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
  File "/usr/lib/python2.7/logging/__init__.py", line 1152, in info
    self._log(INFO, msg, args, **kwargs)
  File "/usr/lib/python2.7/logging/__init__.py", line 1271, in _log
    self.handle(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 1281, in handle
    self.callHandlers(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 1321, in callHandlers
    hdlr.handle(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 749, in handle
    self.emit(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 942, in emit
    StreamHandler.emit(self, record)
  File "/usr/lib/python2.7/logging/__init__.py", line 879, in emit
    self.handleError(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 802, in handleError
    None, sys.stderr)
  File "/usr/lib/python2.7/traceback.py", line 125, in print_exception
    print_tb(tb, limit, file)
  File "/usr/lib/python2.7/traceback.py", line 68, in print_tb
    linecache.checkcache(filename)
  File "/usr/lib/python2.7/linecache.py", line 64, in checkcache
    if size != stat.st_size or mtime != stat.st_mtime:
RuntimeError: maximum recursion depth exceeded in cmp
Traceback (most recent call last):
  File "/usr/lib/python2.7/logging/__init__.py", line 851, in emit
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 109, in myexcepthook
    lexer = get_lexer_by_name('pytb', stripall=True)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 88, in get_lexer_by_name
    return _lexer_cache[name](**options)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex '^(?=  File "[^"]+", line \\d+)' in state 'root' of <class 'pygments.lexers.python.PythonTracebackLexer'>: maximum recursion depth exceeded while calling a Python object

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 621, in _get_data
    utool.printex(ex, 'problem getting in column %r' % (col,))
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 887, in printex
    print_func('\n+------\n')
  File "/home/joncrall/code/utool/utool/util_inject.py", line 144, in print
    util_logging.__UTOOL_PRINT__(*args)
  File "/home/joncrall/code/utool/utool/util_logging.py", line 225, in utool_print
    return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
  File "/usr/lib/python2.7/logging/__init__.py", line 1152, in info
    self._log(INFO, msg, args, **kwargs)
  File "/usr/lib/python2.7/logging/__init__.py", line 1271, in _log
    self.handle(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 1281, in handle
    self.callHandlers(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 1321, in callHandlers
    hdlr.handle(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 749, in handle
    self.emit(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 942, in emit
    StreamHandler.emit(self, record)
  File "/usr/lib/python2.7/logging/__init__.py", line 879, in emit
    self.handleError(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 802, in handleError
    None, sys.stderr)
  File "/usr/lib/python2.7/traceback.py", line 125, in print_exception
    print_tb(tb, limit, file)
  File "/usr/lib/python2.7/traceback.py", line 68, in print_tb
    linecache.checkcache(filename)
  File "/usr/lib/python2.7/linecache.py", line 64, in checkcache
    if size != stat.st_size or mtime != stat.st_mtime:
RuntimeError: maximum recursion depth exceeded in cmp
Traceback (most recent call last):
Error in APIThumbDelegate

+------

<!!! EXCEPTION !!!>
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 314, in sizeHint
    thumb_path = dgt.get_thumb_path_if_exists(view, offset, qtindex)
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 201, in get_thumb_path_if_exists
    data = dgt.get_model_data(qtindex)
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 172, in get_model_data
    data = model.data(qtindex, QtCore.Qt.DisplayRole, **datakw)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 840, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 621, in _get_data
    utool.printex(ex, 'problem getting in column %r' % (col,))
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 887, in printex
    print_func('\n+------\n')
  File "/home/joncrall/code/utool/utool/util_inject.py", line 144, in print
    util_logging.__UTOOL_PRINT__(*args)
  File "/home/joncrall/code/utool/utool/util_logging.py", line 225, in utool_print
    return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
  File "/usr/lib/python2.7/logging/__init__.py", line 1152, in info
    self._log(INFO, msg, args, **kwargs)
  File "/usr/lib/python2.7/logging/__init__.py", line 1271, in _log
    self.handle(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 1281, in handle
    self.callHandlers(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 1321, in callHandlers
    hdlr.handle(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 749, in handle
    self.emit(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 942, in emit
    StreamHandler.emit(self, record)
  File "/usr/lib/python2.7/logging/__init__.py", line 879, in emit
    self.handleError(record)
  File "/usr/lib/python2.7/logging/__init__.py", line 802, in handleError
    None, sys.stderr)
  File "/usr/lib/python2.7/traceback.py", line 125, in print_exception
    print_tb(tb, limit, file)
  File "/usr/lib/python2.7/traceback.py", line 67, in print_tb
    '  File "%s", line %d, in %s' % (filename, lineno, name))
RuntimeError: maximum recursion depth exceeded

[!sizeHint] Error in APIThumbDelegate
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded
</!!! EXCEPTION !!!>

L______


+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 4
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded
</!!! EXCEPTION !!!>

L______

Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 111, in myexcepthook
    formatted_text = highlight(tbtext, lexer, formatter)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 86, in highlight
    return format(lex(code, lexer), formatter, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 66, in format
    formatter.format(tokens, realoutfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 100, in format
    return Formatter.format(self, tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatter.py", line 95, in format
    return self.format_unencoded(tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 136, in format_unencoded
    for ttype, value in tokensource:
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 192, in streamer
    for i, t, v in self.get_tokens_unprocessed(text):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 635, in get_tokens_unprocessed
    for item in action(self, m):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 324, in callback
    data), ctx):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 380, in callback
    lx = _other(**kwargs)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex "'''" in state 'tsqs' of <class 'pygments.lexers.python.PythonLexer'>: maximum recursion depth exceeded while calling a Python object

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 832, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 579, in get_annot_exemplar_flags
    annot_exemplar_flag_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_exemplar_flag',), aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 420, in get_where
    eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 560, in _executemany_operation_fmt
    auto_commit=True, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 994, in executemany
    with SQLExecutionContext(db, operation, **contextkw) as context:
  File "/home/joncrall/code/ibeis/ibeis/control/_sql_helpers.py", line 315, in __init__
    context.operation_type = get_operation_type(operation)  # Parse the optype
RuntimeError: maximum recursion depth exceeded

+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 4
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded
</!!! EXCEPTION !!!>

L______

Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 111, in myexcepthook
    formatted_text = highlight(tbtext, lexer, formatter)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 86, in highlight
    return format(lex(code, lexer), formatter, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 66, in format
    formatter.format(tokens, realoutfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 100, in format
    return Formatter.format(self, tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatter.py", line 95, in format
    return self.format_unencoded(tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 136, in format_unencoded
    for ttype, value in tokensource:
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 192, in streamer
    for i, t, v in self.get_tokens_unprocessed(text):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 635, in get_tokens_unprocessed
    for item in action(self, m):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 324, in callback
    data), ctx):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 380, in callback
    lx = _other(**kwargs)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex "'''" in state 'tsqs' of <class 'pygments.lexers.python.PythonLexer'>: maximum recursion depth exceeded while calling a Python object

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 579, in get_annot_exemplar_flags
    annot_exemplar_flag_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_exemplar_flag',), aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 420, in get_where
    eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 560, in _executemany_operation_fmt
    auto_commit=True, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 994, in executemany
    with SQLExecutionContext(db, operation, **contextkw) as context:
  File "/home/joncrall/code/ibeis/ibeis/control/_sql_helpers.py", line 315, in __init__
    context.operation_type = get_operation_type(operation)  # Parse the optype
RuntimeError: maximum recursion depth exceeded

+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 4
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded
</!!! EXCEPTION !!!>

L______

Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 111, in myexcepthook
    formatted_text = highlight(tbtext, lexer, formatter)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 86, in highlight
    return format(lex(code, lexer), formatter, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 66, in format
    formatter.format(tokens, realoutfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 100, in format
    return Formatter.format(self, tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatter.py", line 95, in format
    return self.format_unencoded(tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 136, in format_unencoded
    for ttype, value in tokensource:
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 192, in streamer
    for i, t, v in self.get_tokens_unprocessed(text):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 635, in get_tokens_unprocessed
    for item in action(self, m):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 324, in callback
    data), ctx):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 380, in callback
    lx = _other(**kwargs)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex "'''" in state 'tsqs' of <class 'pygments.lexers.python.PythonLexer'>: maximum recursion depth exceeded while calling a Python object

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 813, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 579, in get_annot_exemplar_flags
    annot_exemplar_flag_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_exemplar_flag',), aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 420, in get_where
    eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 560, in _executemany_operation_fmt
    auto_commit=True, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 994, in executemany
    with SQLExecutionContext(db, operation, **contextkw) as context:
  File "/home/joncrall/code/ibeis/ibeis/control/_sql_helpers.py", line 315, in __init__
    context.operation_type = get_operation_type(operation)  # Parse the optype
RuntimeError: maximum recursion depth exceeded

+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 5
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded
</!!! EXCEPTION !!!>

L______

Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 111, in myexcepthook
    formatted_text = highlight(tbtext, lexer, formatter)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 86, in highlight
    return format(lex(code, lexer), formatter, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 66, in format
    formatter.format(tokens, realoutfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 100, in format
    return Formatter.format(self, tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatter.py", line 95, in format
    return self.format_unencoded(tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 136, in format_unencoded
    for ttype, value in tokensource:
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 192, in streamer
    for i, t, v in self.get_tokens_unprocessed(text):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 635, in get_tokens_unprocessed
    for item in action(self, m):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 324, in callback
    data), ctx):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 380, in callback
    lx = _other(**kwargs)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex "'''" in state 'tsqs' of <class 'pygments.lexers.python.PythonLexer'>: maximum recursion depth exceeded while calling a Python object

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1799, in get_annot_yaw_texts
    yaw_list = ibs.get_annot_yaws(aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 993, in get_annot_yaws
    yaw_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_YAW,), aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
RuntimeError: maximum recursion depth exceeded

+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 6
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded
</!!! EXCEPTION !!!>

L______

Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 111, in myexcepthook
    formatted_text = highlight(tbtext, lexer, formatter)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 86, in highlight
    return format(lex(code, lexer), formatter, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 66, in format
    formatter.format(tokens, realoutfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 100, in format
    return Formatter.format(self, tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatter.py", line 95, in format
    return self.format_unencoded(tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 136, in format_unencoded
    for ttype, value in tokensource:
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 192, in streamer
    for i, t, v in self.get_tokens_unprocessed(text):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 635, in get_tokens_unprocessed
    for item in action(self, m):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 324, in callback
    data), ctx):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 380, in callback
    lx = _other(**kwargs)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex "'''" in state 'tsqs' of <class 'pygments.lexers.python.PythonLexer'>: maximum recursion depth exceeded while calling a Python object

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1783, in get_annot_quality_texts
    quality_list = ibs.get_annot_qualities(aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1752, in get_annot_qualities
    const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid', eager=eager)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
RuntimeError: maximum recursion depth exceeded

+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 2
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded while getting the str of an object
</!!! EXCEPTION !!!>

L______

Error in APIThumbDelegate

+------

<!!! EXCEPTION !!!>
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 314, in sizeHint
    thumb_path = dgt.get_thumb_path_if_exists(view, offset, qtindex)
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 201, in get_thumb_path_if_exists
    data = dgt.get_model_data(qtindex)
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 172, in get_model_data
    data = model.data(qtindex, QtCore.Qt.DisplayRole, **datakw)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 840, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_chip_funcs.py", line 260, in get_annot_chip_thumbtup
    thumb_gpaths = ibs.get_annot_chip_thumbpath(aid_list, thumbsize=thumbsize, qreq_=qreq_)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_chip_funcs.py", line 220, in get_annot_chip_thumbpath
    thumb_suffix = '_' + str(thumbsize) + const.CHIP_THUMB_SUFFIX
RuntimeError: maximum recursion depth exceeded while getting the str of an object

[!sizeHint] Error in APIThumbDelegate
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded while getting the str of an object
</!!! EXCEPTION !!!>

L______


+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 5
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded while calling a Python object
</!!! EXCEPTION !!!>

L______

Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 111, in myexcepthook
    formatted_text = highlight(tbtext, lexer, formatter)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 86, in highlight
    return format(lex(code, lexer), formatter, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 66, in format
    formatter.format(tokens, realoutfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 100, in format
    return Formatter.format(self, tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatter.py", line 95, in format
    return self.format_unencoded(tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 136, in format_unencoded
    for ttype, value in tokensource:
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 192, in streamer
    for i, t, v in self.get_tokens_unprocessed(text):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 635, in get_tokens_unprocessed
    for item in action(self, m):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 324, in callback
    data), ctx):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 380, in callback
    lx = _other(**kwargs)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex "'''" in state 'tsqs' of <class 'pygments.lexers.python.PythonLexer'>: maximum recursion depth exceeded in cmp

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1799, in get_annot_yaw_texts
    yaw_list = ibs.get_annot_yaws(aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 993, in get_annot_yaws
    yaw_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_YAW,), aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 420, in get_where
    eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 560, in _executemany_operation_fmt
    auto_commit=True, eager=eager, **kwargs)
RuntimeError: maximum recursion depth exceeded while calling a Python object

+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 6
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded while calling a Python object
</!!! EXCEPTION !!!>

L______

Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 111, in myexcepthook
    formatted_text = highlight(tbtext, lexer, formatter)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 86, in highlight
    return format(lex(code, lexer), formatter, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 66, in format
    formatter.format(tokens, realoutfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 100, in format
    return Formatter.format(self, tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatter.py", line 95, in format
    return self.format_unencoded(tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 136, in format_unencoded
    for ttype, value in tokensource:
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 192, in streamer
    for i, t, v in self.get_tokens_unprocessed(text):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 635, in get_tokens_unprocessed
    for item in action(self, m):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 324, in callback
    data), ctx):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 380, in callback
    lx = _other(**kwargs)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex "'''" in state 'tsqs' of <class 'pygments.lexers.python.PythonLexer'>: maximum recursion depth exceeded in cmp

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1783, in get_annot_quality_texts
    quality_list = ibs.get_annot_qualities(aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1752, in get_annot_qualities
    const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid', eager=eager)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 420, in get_where
    eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 560, in _executemany_operation_fmt
    auto_commit=True, eager=eager, **kwargs)
RuntimeError: maximum recursion depth exceeded while calling a Python object

+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 2
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded
</!!! EXCEPTION !!!>

L______

Error in APIThumbDelegate

+------

<!!! EXCEPTION !!!>
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 314, in sizeHint
    thumb_path = dgt.get_thumb_path_if_exists(view, offset, qtindex)
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 201, in get_thumb_path_if_exists
    data = dgt.get_model_data(qtindex)
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 172, in get_model_data
    data = model.data(qtindex, QtCore.Qt.DisplayRole, **datakw)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 840, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_chip_funcs.py", line 260, in get_annot_chip_thumbtup
    thumb_gpaths = ibs.get_annot_chip_thumbpath(aid_list, thumbsize=thumbsize, qreq_=qreq_)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_chip_funcs.py", line 221, in get_annot_chip_thumbpath
    annot_uuid_list = ibs.get_annot_visual_uuids(aid_list)
  File "/home/joncrall/code/utool/utool/util_decor.py", line 90, in wrp_noexectb
    raise exc_type, exc_value, exc_traceback
RuntimeError: maximum recursion depth exceeded

[!sizeHint] Error in APIThumbDelegate
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded
</!!! EXCEPTION !!!>

L______


+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 5
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded in __instancecheck__
</!!! EXCEPTION !!!>

L______

Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 111, in myexcepthook
    formatted_text = highlight(tbtext, lexer, formatter)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 86, in highlight
    return format(lex(code, lexer), formatter, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 66, in format
    formatter.format(tokens, realoutfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 100, in format
    return Formatter.format(self, tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatter.py", line 95, in format
    return self.format_unencoded(tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 136, in format_unencoded
    for ttype, value in tokensource:
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 192, in streamer
    for i, t, v in self.get_tokens_unprocessed(text):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 635, in get_tokens_unprocessed
    for item in action(self, m):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 324, in callback
    data), ctx):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 380, in callback
    lx = _other(**kwargs)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 484, in _process_state
    str(tdef)))
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex '%(\\(\\w+\\))?[-#0 +]*([0-9]+|[*])?(\\.([0-9]+|[*]))?[hlL]?[diouxXeEfFgGcrs%]' in state 'strings' of <class 'pygments.lexers.python.PythonLexer'>: maximum recursion depth exceeded

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1799, in get_annot_yaw_texts
    yaw_list = ibs.get_annot_yaws(aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 993, in get_annot_yaws
    yaw_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_YAW,), aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 420, in get_where
    eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 560, in _executemany_operation_fmt
    auto_commit=True, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 970, in executemany
    if isinstance(params_iter, (list, tuple)):
RuntimeError: maximum recursion depth exceeded in __instancecheck__

+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 6
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded in __instancecheck__
</!!! EXCEPTION !!!>

L______

Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 111, in myexcepthook
    formatted_text = highlight(tbtext, lexer, formatter)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 86, in highlight
    return format(lex(code, lexer), formatter, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 66, in format
    formatter.format(tokens, realoutfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 100, in format
    return Formatter.format(self, tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatter.py", line 95, in format
    return self.format_unencoded(tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 136, in format_unencoded
    for ttype, value in tokensource:
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 192, in streamer
    for i, t, v in self.get_tokens_unprocessed(text):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 635, in get_tokens_unprocessed
    for item in action(self, m):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 324, in callback
    data), ctx):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 380, in callback
    lx = _other(**kwargs)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 484, in _process_state
    str(tdef)))
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex '%(\\(\\w+\\))?[-#0 +]*([0-9]+|[*])?(\\.([0-9]+|[*]))?[hlL]?[diouxXeEfFgGcrs%]' in state 'strings' of <class 'pygments.lexers.python.PythonLexer'>: maximum recursion depth exceeded

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1783, in get_annot_quality_texts
    quality_list = ibs.get_annot_qualities(aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1752, in get_annot_qualities
    const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid', eager=eager)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 420, in get_where
    eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 560, in _executemany_operation_fmt
    auto_commit=True, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 970, in executemany
    if isinstance(params_iter, (list, tuple)):
RuntimeError: maximum recursion depth exceeded in __instancecheck__

+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 2
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded while calling a Python object
</!!! EXCEPTION !!!>

L______

Error in APIThumbDelegate

+------

<!!! EXCEPTION !!!>
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 314, in sizeHint
    thumb_path = dgt.get_thumb_path_if_exists(view, offset, qtindex)
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 201, in get_thumb_path_if_exists
    data = dgt.get_model_data(qtindex)
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 172, in get_model_data
    data = model.data(qtindex, QtCore.Qt.DisplayRole, **datakw)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 840, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_chip_funcs.py", line 260, in get_annot_chip_thumbtup
    thumb_gpaths = ibs.get_annot_chip_thumbpath(aid_list, thumbsize=thumbsize, qreq_=qreq_)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_chip_funcs.py", line 221, in get_annot_chip_thumbpath
    annot_uuid_list = ibs.get_annot_visual_uuids(aid_list)
  File "/home/joncrall/code/utool/utool/util_decor.py", line 90, in wrp_noexectb
    raise exc_type, exc_value, exc_traceback
RuntimeError: maximum recursion depth exceeded while calling a Python object

[!sizeHint] Error in APIThumbDelegate
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded while calling a Python object
</!!! EXCEPTION !!!>

L______


+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 5
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded
</!!! EXCEPTION !!!>

L______

Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 111, in myexcepthook
    formatted_text = highlight(tbtext, lexer, formatter)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 86, in highlight
    return format(lex(code, lexer), formatter, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 66, in format
    formatter.format(tokens, realoutfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 100, in format
    return Formatter.format(self, tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatter.py", line 95, in format
    return self.format_unencoded(tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 136, in format_unencoded
    for ttype, value in tokensource:
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 192, in streamer
    for i, t, v in self.get_tokens_unprocessed(text):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 635, in get_tokens_unprocessed
    for item in action(self, m):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 324, in callback
    data), ctx):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 380, in callback
    lx = _other(**kwargs)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 484, in _process_state
    str(tdef)))
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex '%(\\(\\w+\\))?[-#0 +]*([0-9]+|[*])?(\\.([0-9]+|[*]))?[hlL]?[diouxXeEfFgGcrs%]' in state 'strings' of <class 'pygments.lexers.python.PythonLexer'>: maximum recursion depth exceeded

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1799, in get_annot_yaw_texts
    yaw_list = ibs.get_annot_yaws(aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 993, in get_annot_yaws
    yaw_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_YAW,), aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 420, in get_where
    eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 560, in _executemany_operation_fmt
    auto_commit=True, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 994, in executemany
    with SQLExecutionContext(db, operation, **contextkw) as context:
  File "/home/joncrall/code/ibeis/ibeis/control/_sql_helpers.py", line 315, in __init__
    context.operation_type = get_operation_type(operation)  # Parse the optype
RuntimeError: maximum recursion depth exceeded

+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 6
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded
</!!! EXCEPTION !!!>

L______

Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 111, in myexcepthook
    formatted_text = highlight(tbtext, lexer, formatter)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 86, in highlight
    return format(lex(code, lexer), formatter, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/__init__.py", line 66, in format
    formatter.format(tokens, realoutfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 100, in format
    return Formatter.format(self, tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatter.py", line 95, in format
    return self.format_unencoded(tokensource, outfile)
  File "/usr/local/lib/python2.7/dist-packages/pygments/formatters/terminal.py", line 136, in format_unencoded
    for ttype, value in tokensource:
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 192, in streamer
    for i, t, v in self.get_tokens_unprocessed(text):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 635, in get_tokens_unprocessed
    for item in action(self, m):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 324, in callback
    data), ctx):
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 380, in callback
    lx = _other(**kwargs)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 581, in __call__
    cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 520, in process_tokendef
    cls._process_state(tokendefs, processed, state)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 484, in _process_state
    str(tdef)))
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 502, in _process_state
    (tdef[0], state, cls, err))
ValueError: uncompilable regex '%(\\(\\w+\\))?[-#0 +]*([0-9]+|[*])?(\\.([0-9]+|[*]))?[hlL]?[diouxXeEfFgGcrs%]' in state 'strings' of <class 'pygments.lexers.python.PythonLexer'>: maximum recursion depth exceeded

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 845, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1783, in get_annot_quality_texts
    quality_list = ibs.get_annot_qualities(aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 1752, in get_annot_qualities
    const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid', eager=eager)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 420, in get_where
    eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 560, in _executemany_operation_fmt
    auto_commit=True, eager=eager, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 994, in executemany
    with SQLExecutionContext(db, operation, **contextkw) as context:
  File "/home/joncrall/code/ibeis/ibeis/control/_sql_helpers.py", line 315, in __init__
    context.operation_type = get_operation_type(operation)  # Parse the optype
RuntimeError: maximum recursion depth exceeded

+------

<!!! EXCEPTION !!!>
[!_get_data] problem getting in column 2
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded while calling a Python object
</!!! EXCEPTION !!!>

L______

Error in APIThumbDelegate

+------

<!!! EXCEPTION !!!>
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 314, in sizeHint
    thumb_path = dgt.get_thumb_path_if_exists(view, offset, qtindex)
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 201, in get_thumb_path_if_exists
    data = dgt.get_model_data(qtindex)
  File "/home/joncrall/code/guitool/guitool/api_thumb_delegate.py", line 172, in get_model_data
    data = model.data(qtindex, QtCore.Qt.DisplayRole, **datakw)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 840, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 619, in _get_data
    data = getter(row_id, **kwargs)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_chip_funcs.py", line 260, in get_annot_chip_thumbtup
    thumb_gpaths = ibs.get_annot_chip_thumbpath(aid_list, thumbsize=thumbsize, qreq_=qreq_)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_chip_funcs.py", line 221, in get_annot_chip_thumbpath
    annot_uuid_list = ibs.get_annot_visual_uuids(aid_list)
  File "/home/joncrall/code/ibeis/ibeis/control/manual_annot_funcs.py", line 943, in get_annot_visual_uuids
    const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid')
  File "/home/joncrall/code/ibeis/ibeis/control/SQLDatabaseControl.py", line 449, in get
    return db.get_where(tblname, colnames, params_iter, where_clause, eager=eager, **kwargs)
RuntimeError: maximum recursion depth exceeded while calling a Python object

[!sizeHint] Error in APIThumbDelegate
<type 'exceptions.RuntimeError'>: maximum recursion depth exceeded while calling a Python object
</!!! EXCEPTION !!!>

L______
L______

Original exception was:
Traceback (most recent call last):
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 813, in data
    data = model._get_data(qtindex, **kwargs)
  File "/home/joncrall/code/guitool/guitool/api_item_model.py", line 621, in _get_data
    utool.printex(ex, 'problem getting in column %r' % (col,))
  File "/home/joncrall/code/utool/utool/util_dbg.py", line 887, in printex
    print_func('\n+------\n')
  File "/home/joncrall/code/utool/utool/util_inject.py", line 144, in print
    util_logging.__UTOOL_PRINT__(*args)
  File "/home/joncrall/code/utool/utool/util_logging.py", line 225, in utool_print
    return  __UTOOL_ROOT_LOGGER__.info(', '.join(map(str, args)))
  File "/usr/lib/python2.7/logging/__init__.py", line 1152, in info
    self._log(INFO, msg, args, **kwargs)
  File "/usr/lib/python2.7/logging/__init__.py", line 1270, in _log
    record = self.makeRecord(self.name, level, fn, lno, msg, args, exc_info, func, extra)
  File "/usr/lib/python2.7/logging/__init__.py", line 1244, in makeRecord
    rv = LogRecord(name, level, fn, lno, msg, args, exc_info, func)
  File "/usr/lib/python2.7/logging/__init__.py", line 266, in __init__
    self.levelname = getLevelName(level)
RuntimeError: maximum recursion depth exceeded
Error in sys.excepthook:
Traceback (most recent call last):
  File "/home/joncrall/code/utool/utool/util_inject.py", line 109, in myexcepthook
    lexer = get_lexer_by_name('pytb', stripall=True)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 87, in get_lexer_by_name
    _load_lexers(module_name)
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/__init__.py", line 41, in _load_lexers
    mod = __import__(module_name, None, None, ['__all__'])
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexers/python.py", line 14, in <module>
    from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
  File "/usr/local/lib/python2.7/dist-packages/pygments/lexer.py", line 24, in <module>
    from pygments.regexopt import regex_opt
  File "/usr/local/lib/python2.7/dist-packages/pygments/regexopt.py", line 19, in <module>
    CS_ESCAPE = re.compile(r'[\^\\\-\]]')
  File "/usr/lib/python2.7/re.py", line 190, in compile
    return _compile(pattern, flags)
  File "/usr/lib/python2.7/re.py", line 242, in _compile
    p = sre_compile.compile(pattern, flags)
  File "/usr/lib/python2.7/sre_compile.py", line 498, in compile
    p = sre_parse.parse(p, flags)
  File "/usr/lib/python2.7/sre_parse.py", line 678, in parse
    source = Tokenizer(str)
  File "/usr/lib/python2.7/sre_parse.py", line 181, in __init__
    self.__next()
RuntimeError: maximum recursion depth exceeded

"""
from __future__ import absolute_import, division, print_function
from guitool.__PYQT__ import QtGui, QtCore
#import cv2  # NOQA
#import numpy as np
import utool
#import time
#from six.moves import zip
from os.path import exists
from vtool import image as gtool
#from vtool import linalg, geometry
from vtool import geometry
#from multiprocessing import Process
#from guitool import guitool_components as comp
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[APIThumbDelegate]', DEBUG=False)
import utool as ut
ut.noinject(__name__, '[APIThumbDelegate]', DEBUG=False)


VERBOSE_QT = ut.get_argflag(('--verbose-qt', '--verbqt'))
VERBOSE_THUMB = utool.VERBOSE or ut.get_argflag(('--verbose-thumb', '--verbthumb')) or VERBOSE_QT


MAX_NUM_THUMB_THREADS = 1


def read_thumb_size(thumb_path):
    if VERBOSE_THUMB:
        print('[ThumbDelegate] Reading thumb size')
    npimg = gtool.imread(thumb_path, delete_if_corrupted=True)
    (height, width) = npimg.shape[0:2]
    del npimg
    return width, height


def test_show_qimg(qimg):
    qpixmap = QtGui.QPixmap(qimg)
    lbl = QtGui.QLabel()
    lbl.setPixmap(qpixmap)
    lbl.show()   # show label with qim image
    return lbl


#@ut.memprof
def read_thumb_as_qimg(thumb_path):
    r"""
    Args:
        thumb_path (?):

    Returns:
        tuple: (qimg, width, height)

    CommandLine:
        python -m guitool.api_thumb_delegate --test-read_thumb_as_qimg --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from guitool.api_thumb_delegate import *  # NOQA
        >>> import guitool
        >>> # build test data
        >>> thumb_path = ut.grab_test_imgpath('carl.jpg')
        >>> # execute function
        >>> guitool.ensure_qtapp()
        >>> (qimg) = ut.memprof(read_thumb_as_qimg)(thumb_path)
        >>> if ut.show_was_requested():
        >>>    lbl = test_show_qimg(qimg)
        >>>    guitool.qtapp_loop()
        >>> # verify results
        >>> print(qimg)

    Timeit::
        %timeit np.dstack((npimg, np.full(npimg.shape[0:2], 255, dtype=np.uint8)))
        %timeit cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)
        npimg1 = np.dstack((npimg, np.full(npimg.shape[0:2], 255, dtype=np.uint8)))
        # seems to be memory leak in cvtColor?
        npimg2 = cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)

    """
    if VERBOSE_THUMB:
        print('[ThumbDelegate] Reading thumb as qimg')
    # Read thumbnail image and convert to 32bit aligned for Qt
    #if False:
    #    data  = np.dstack((npimg, np.full(npimg.shape[0:2], 255, dtype=np.uint8)))
    #if False:
    #    # Reading the npimage and then handing it off to Qt causes a memory
    #    # leak. The numpy array probably is never unallocated because qt doesn't
    #    # own it and it never loses its reference count
    #    #npimg = gtool.imread(thumb_path, delete_if_corrupted=True)
    #    #print('npimg.dtype = %r, %r' % (npimg.shape, npimg.dtype))
    #    #npimg   = cv2.cvtColor(npimg, cv2.COLOR_BGR2BGRA)
    #    #format_ = QtGui.QImage.Format_ARGB32
    #    ##    #data    = npimg.astype(np.uint8)
    #    ##    #npimg   = np.dstack((npimg[:, :, 3], npimg[:, :, 0:2]))
    #    ##    #data    = npimg.astype(np.uint8)
    #    ##else:
    #    ## Memory seems to be no freed by the QImage?
    #    ##data = np.ascontiguousarray(npimg[:, :, ::-1].astype(np.uint8), dtype=np.uint8)
    #    ##data = np.ascontiguousarray(npimg[:, :, :].astype(np.uint8), dtype=np.uint8)
    #    #data = npimg
    #    ##format_ = QtGui.QImage.Format_RGB888
    #    #(height, width) = data.shape[0:2]
    #    #qimg    = QtGui.QImage(data, width, height, format_)
    #    #del npimg
    #    #del data
    #else:
    #format_ = QtGui.QImage.Format_ARGB32
    #qimg    = QtGui.QImage(thumb_path, format_)
    qimg    = QtGui.QImage(thumb_path)
    return qimg


RUNNING_CREATION_THREADS = {}


def register_thread(key, val):
    global RUNNING_CREATION_THREADS
    RUNNING_CREATION_THREADS[key] = val


def unregister_thread(key):
    global RUNNING_CREATION_THREADS
    del RUNNING_CREATION_THREADS[key]


DELEGATE_BASE = QtGui.QItemDelegate


class APIThumbDelegate(DELEGATE_BASE):
    """
    TODO: The delegate can have a reference to the view, and it is allowed
    to resize the rows to fit the images.  It probably should not resize columns
    but it can get the column width and resize the image to that size.

    get_thumb_size is a callback function which should return whatever the
    requested thumbnail size is

    SeeAlso:
         api_item_view.infer_delegates
    """
    def __init__(dgt, parent=None, get_thumb_size=None):
        if VERBOSE_THUMB:
            print('[ThumbDelegate] __init__ parent=%r, get_thumb_size=%r' %
                    (parent, get_thumb_size))
        DELEGATE_BASE.__init__(dgt, parent)
        dgt.pool = None
        # TODO: get from the view
        if get_thumb_size is None:
            dgt.get_thumb_size = lambda: 128  # 256
        else:
            dgt.get_thumb_size = get_thumb_size  # 256
        dgt.last_thumbsize = None

    def get_model_data(dgt, qtindex):
        """
        The model data for a thumb should be a tuple:
        (thumb_path, img_path, imgsize, bboxes, thetas)
        """
        model = qtindex.model()
        datakw = dict(thumbsize=dgt.get_thumb_size())
        data = model.data(qtindex, QtCore.Qt.DisplayRole, **datakw)
        if data is None:
            return None
        # The data should be specified as a thumbtup
        #if isinstance(data, QtCore.QVariant):
        if hasattr(data, 'toPyObject'):
            data = data.toPyObject()
        if data is None:
            return None
        assert isinstance(data, tuple), (
            'data=%r is %r. should be a thumbtup' % (data, type(data)))
        thumbtup = data
        #(thumb_path, img_path, bbox_list) = thumbtup
        return thumbtup

    def get_thumb_path_if_exists(dgt, view, offset, qtindex):
        """
        Checks if the thumbnail is ready to paint

        Returns:
            thumb_path if computed otherwise returns None
        """

        # Check if still in viewport
        if view_would_not_be_visible(view, offset):
            return None

        # Get data from the models display role
        try:
            data = dgt.get_model_data(qtindex)
            if data is None:
                print('[thumb_delegate] no data')
                return
            (thumb_path, img_path, img_size, bbox_list, theta_list) = data
            invalid = (thumb_path is None or img_path is None or bbox_list is None
                       or img_size is None)
            if invalid:
                print('[thumb_delegate] something is wrong')
                return
        except AssertionError as ex:
            utool.printex(ex, 'error getting thumbnail data')
            return

        # Check if still in viewport
        if view_would_not_be_visible(view, offset):
            return None

        if not exists(thumb_path):
            if not exists(img_path):
                if VERBOSE_THUMB:
                    print('[ThumbDelegate] SOURCE IMAGE NOT COMPUTED: %r' % (img_path,))
                return None
            # Start computation of thumb if needed
            #qtindex.model()._update()  # should probably be deleted
            # where you are when you request the run
            if VERBOSE_THUMB:
                print('[ThumbDelegate] Spawning thumbnail creation thread')
            thumbsize = dgt.get_thumb_size()
            thumb_creation_thread = ThumbnailCreationThread(
                thumb_path, img_path, img_size, thumbsize,
                qtindex, view, offset, bbox_list, theta_list
            )
            #register_thread(thumb_path, thumb_creation_thread)
            # Initialize threadcount
            dgt.pool = QtCore.QThreadPool()
            dgt.pool.setMaxThreadCount(MAX_NUM_THUMB_THREADS)
            dgt.pool.start(thumb_creation_thread)
            # print('[ThumbDelegate] Waiting to compute')
            return None
        else:
            # thumb is computed return the path
            return thumb_path

    def adjust_thumb_cell_size(dgt, qtindex, width, height):
        """
        called during paint to ensure that the cell is large enough for the
        image.
        """
        view = dgt.parent()
        if isinstance(view, QtGui.QTableView):
            # dimensions of the table cells
            col_width = view.columnWidth(qtindex.column())
            col_height = view.rowHeight(qtindex.row())
            thumbsize = dgt.get_thumb_size()
            if thumbsize != dgt.last_thumbsize:
                # has thumbsize changed?
                if thumbsize != col_width:
                    view.setColumnWidth(qtindex.column(), thumbsize)
                if height != col_height:
                    view.setRowHeight(qtindex.row(), height)
                dgt.last_thumbsize = thumbsize
            # Let columns shrink
            if thumbsize != col_width:
                view.setColumnWidth(qtindex.column(), thumbsize)
            # Let rows grow
            if height > col_height:
                view.setRowHeight(qtindex.row(), height)
        elif isinstance(view, QtGui.QTreeView):
            col_width = view.columnWidth(qtindex.column())
            col_height = view.rowHeight(qtindex)
            # TODO: finishme

    def paint(dgt, painter, option, qtindex):
        """
        TODO: prevent recursive paint
        """
        view = dgt.parent()
        offset = view.verticalOffset() + option.rect.y()
        # Check if still in viewport
        if view_would_not_be_visible(view, offset):
            return None
        try:
            thumb_path = dgt.get_thumb_path_if_exists(view, offset, qtindex)
            if thumb_path is not None:
                # Check if still in viewport
                if view_would_not_be_visible(view, offset):
                    return None
                # Read the precomputed thumbnail
                qimg = read_thumb_as_qimg(thumb_path)
                width, height = qimg.width(), qimg.height()
                # Adjust the cell size to fit the image
                dgt.adjust_thumb_cell_size(qtindex, width, height)
                # Check if still in viewport
                if view_would_not_be_visible(view, offset):
                    return None
                # Paint image on an item in some view
                painter.save()
                painter.setClipRect(option.rect)
                painter.translate(option.rect.x(), option.rect.y())
                painter.drawImage(QtCore.QRectF(0, 0, width, height), qimg)
                painter.restore()
        except Exception as ex:
            # PSA: Always report errors on Exceptions!
            print('Error in APIThumbDelegate')
            utool.printex(ex, 'Error in APIThumbDelegate')
            painter.save()
            painter.restore()

    def sizeHint(dgt, option, qtindex):
        view = dgt.parent()
        offset = view.verticalOffset() + option.rect.y()
        try:
            thumb_path = dgt.get_thumb_path_if_exists(view, offset, qtindex)
            if thumb_path is not None:
                # Read the precomputed thumbnail
                width, height = read_thumb_size(thumb_path)
                return QtCore.QSize(width, height)
            else:
                #print("[APIThumbDelegate] Name not found")
                return QtCore.QSize()
        except Exception as ex:
            print("Error in APIThumbDelegate")
            utool.printex(ex, 'Error in APIThumbDelegate', tb=True)
            return QtCore.QSize()


def view_would_not_be_visible(view, offset):
    viewport = view.viewport()
    height = viewport.size().height()
    height_offset = view.verticalOffset()
    current_offset = height_offset + height // 2
    # Check if the current scroll position is far beyond the
    # scroll position when this was initially requested.
    return abs(current_offset - offset) >= height


def get_thread_thumb_info(bbox_list, theta_list, thumbsize, img_size):
    r"""
    Args:
        bbox_list (list):
        theta_list (list):
        thumbsize (?):
        img_size (?):

    CommandLine:
        python -m guitool.api_thumb_delegate --test-get_thread_thumb_info

    Example:
        >>> # ENABLE_DOCTEST
        >>> from guitool.api_thumb_delegate import *  # NOQA
        >>> # build test data
        >>> bbox_list = [(100, 50, 400, 200)]
        >>> theta_list = [0]
        >>> thumbsize = 128
        >>> img_size = 600, 300
        >>> # execute function
        >>> result = get_thread_thumb_info(bbox_list, theta_list, thumbsize, img_size)
        >>> # verify results
        >>> print(result)
        ((128, 64), [[[21, 11], [107, 11], [107, 53], [21, 53], [21, 11]]])

    """
    theta_list = [theta_list] if not utool.is_listlike(theta_list) else theta_list
    max_dsize = (thumbsize, thumbsize)
    dsize, sx, sy = gtool.resized_clamped_thumb_dims(img_size, max_dsize)
    # Compute new verts list
    new_verts_list = list(gtool.scale_bbox_to_verts_gen(bbox_list, theta_list, sx, sy))
    return dsize, new_verts_list


def make_thread_thumb(img_path, dsize, new_verts_list):
    orange_bgr = (0, 128, 255)
    img = gtool.imread(img_path)  # Read Image (.0424s) <- Takes most time!
    thumb = gtool.resize(img, dsize)  # Resize to thumb dims (.0015s)
    del img
    # Draw bboxes on thumb (not image)
    for new_verts in new_verts_list:
        geometry.draw_verts(thumb, new_verts, color=orange_bgr, thickness=2, out=thumb)
        #thumb = geometry.draw_verts(thumb, new_verts, color=orange_bgr, thickness=2)
    return thumb


RUNNABLE_BASE = QtCore.QRunnable


class ThumbnailCreationThread(RUNNABLE_BASE):
    """
    Helper to compute thumbnails concurrently
    """

    def __init__(thread, thumb_path, img_path, img_size, thumbsize, qtindex, view, offset, bbox_list, theta_list):
        RUNNABLE_BASE.__init__(thread)
        thread.thumb_path = thumb_path
        thread.img_path = img_path
        thread.img_size = img_size
        thread.qtindex = qtindex
        thread.offset = offset
        thread.thumbsize = thumbsize
        thread.view = view
        thread.bbox_list = bbox_list
        thread.theta_list = theta_list

    def thumb_would_not_be_visible(thread):
        return view_would_not_be_visible(thread.view, thread.offset)

    def _run(thread):
        """ Compute thumbnail in a different thread """
        #time.sleep(.005)  # Wait a in case the user is just scrolling
        if thread.thumb_would_not_be_visible():
            return
        # Precompute info BEFORE reading the image (.0002s)
        dsize, new_verts_list = get_thread_thumb_info(
            thread.bbox_list, thread.theta_list, thread.thumbsize, thread.img_size)
        #time.sleep(.005)  # Wait a in case the user is just scrolling
        if thread.thumb_would_not_be_visible():
            return
        # -----------------
        # This part takes time, hopefully the user actually wants to see this
        # thumbnail.
        thumb = make_thread_thumb(thread.img_path, dsize, new_verts_list)
        if thread.thumb_would_not_be_visible():
            return
        gtool.imwrite(thread.thumb_path, thumb)
        del thumb
        if thread.thumb_would_not_be_visible():
            return
        #print('[ThumbCreationThread] Thumb Written: %s' % thread.thumb_path)
        thread.qtindex.model().dataChanged.emit(thread.qtindex, thread.qtindex)
        #unregister_thread(thread.thumb_path)

    def run(thread):
        try:
            thread._run()
        except Exception as ex:
            utool.printex(ex, 'thread failed', tb=True)
            #raise

    #def __del__(self):
    #    print('About to delete creation thread')


# GRAVE:
#print('[APIItemDelegate] Request Thumb: rc=(%d, %d), nBboxes=%r' %
#      (qtindex.row(), qtindex.column(), len(bbox_list)))
#print('[APIItemDelegate] bbox_list = %r' % (bbox_list,))


if __name__ == '__main__':
    """
    CommandLine:
        python -m guitool.api_thumb_delegate
        python -m guitool.api_thumb_delegate --allexamples
        python -m guitool.api_thumb_delegate --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
