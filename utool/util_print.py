from __future__ import division, print_function
import numpy as np
import functools
from .util_inject import inject, get_injected_modules
print, print_, printDBG, rrr, profile = inject(__name__, '[print]')


def horiz_print(*args):
    toprint = horiz_string(args)
    print(toprint)


def horiz_string(str_list):
    '''
    prints a list of objects ensuring that the next item in the list
    is all the way to the right of any previous items.
    str_list = ['A = ', str(np.array(((1,2),(3,4)))), ' * ', str(np.array(((1,2),(3,4))))]
    '''
    all_lines = []
    hpos = 0
    for sx in xrange(len(str_list)):
        str_ = str(str_list[sx])
        lines = str_.split('\n')
        line_diff = len(lines) - len(all_lines)
        # Vertical padding
        if line_diff > 0:
            all_lines += [' ' * hpos] * line_diff
        # Add strings
        for lx, line in enumerate(lines):
            all_lines[lx] += line
            hpos = max(hpos, len(all_lines[lx]))
        # Horizontal padding
        for lx in xrange(len(all_lines)):
            hpos_diff = hpos - len(all_lines[lx])
            if hpos_diff > 0:
                all_lines[lx] += ' ' * hpos_diff
    ret = '\n'.join(all_lines)
    return ret


def str2(obj):
    if isinstance(obj, dict):
        return str(obj).replace(', ', '\n')[1:-1]
    if isinstance(obj, type):
        return str(obj).replace('<type \'', '').replace('\'>', '')
    else:
        return str(obj)


class Indenter(object):
    # THIS IS MUCH BETTER
    def __init__(self, lbl='    '):
        #self.modules = modules
        self.modules = get_injected_modules()
        self.old_prints = {}
        self.old_prints_ = {}
        self.old_printDBGs = {}
        self.lbl = lbl
        self.INDENT_PRINT_ = False

    def start(self):
        # Chain functions together rather than overwriting stdout
        def indent_msg(msg):
            return self.lbl + str(msg).replace('\n', '\n' + self.lbl)

        def save_module_functions(dict_, func_name):
            for mod in self.modules:
                try:
                    dict_[mod] = getattr(mod, func_name)
                except KeyError as ex:
                    print('KeyError: ' + str(ex))
                    print('WARNING: module=%r was loaded between indent sessions' % mod)
                except AttributeError as ex:
                    print('AttributeError: ' + str(ex))
                    print('WARNING: module=%r is not managed by __common__' % mod)

        save_module_functions(self.old_prints, 'print')
        save_module_functions(self.old_printDBGs, 'printDBG')

        #for mod in self.modules:
            #try:
                #self.old_prints[mod] = mod.print
                #if self.INDENT_PRINT_:
                    #self.old_prints_[mod] = mod.print_
            #except KeyError as ex:
                #print('KeyError: ' + str(ex))
                #print('WARNING: module=%r was loaded between indent sessions' % mod)
            #except AttributeError as ex:
                #print('AttributeError: ' + str(ex))
                #print('WARNING: module=%r is not managed by __common__' % mod)

        for mod in self.old_prints.keys():
            indent_print = lambda msg: self.old_prints[mod](indent_msg(msg))
            mod.print = indent_print

        for mod in self.old_printDBGs.keys():
            indent_printDBG = lambda msg: self.old_printDBGs[mod](indent_msg(msg))
            mod.printDBG = indent_printDBG
            #if self.INDENT_PRINT_:
                #indent_print_ = lambda msg: self.old_prints_[mod](indent_msg(msg))
                #mod.print_ = indent_print_

    def stop(self):
        for mod in self.old_prints.iterkeys():
            mod.print = self.old_prints[mod]
        for mod in self.old_printDBGs.iterkeys():
            mod.printDBG = self.old_printDBGs[mod]
            #if self.INDENT_PRINT_:
                #mod.print_ =  self.old_prints_[mod]

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


def indent_decor(lbl):
    def indent_decor_wrapper1(func):
        @functools.wraps(func)
        def indent_decor_wrapper2(*args, **kwargs):
            with Indenter(lbl):
                ret = func(*args, **kwargs)
                return ret
        return indent_decor_wrapper2
    return indent_decor_wrapper1


def printshape(arr_name, locals_):
    arr = locals_[arr_name]
    if type(arr) is np.ndarray:
        print(arr_name + '.shape = ' + str(arr.shape))
    else:
        print('len(%s) = %r' % (arr_name, len(arr)))


class NpPrintOpts(object):
    def __init__(self, **kwargs):
        self.orig_opts = np.get_printoptions()
        self.new_opts = kwargs
    def __enter__(self):
        np.set_printoptions(**self.new_opts)
    def __exit__(self, type, value, trace):
        np.set_printoptions(**self.orig_opts)
