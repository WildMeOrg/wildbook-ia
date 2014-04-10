from __future__ import absolute_import, division, print_function
import sys
import warnings
from collections import OrderedDict
import numpy as np
from .util_str import byte_str2
from .util_inject import inject
from .Printable import printableVal
from .DynStruct import DynStruct
print, print_, printDBG, rrr, profile = inject(__name__, '[dev]')


def DEPRICATED(func):
    'deprication decorator'
    warn_msg = 'Depricated call to: %s' % func.__name__

    def __DEP_WRAPPER(*args, **kwargs):
        raise Exception('dep')
        warnings.warn(warn_msg, category=DeprecationWarning)
        #warnings.warn(warn_msg, category=DeprecationWarning)
        return func(*args, **kwargs)
    __DEP_WRAPPER.__name__ = func.__name__
    __DEP_WRAPPER.__doc__ = func.__doc__
    __DEP_WRAPPER.__dict__.update(func.__dict__)
    return __DEP_WRAPPER


# --- Info Strings ---

def stats_str(*args, **kwargs):
    # wrapper for printable_mystats
    return printable_mystats(*args, **kwargs)


def printable_mystats(_list, newlines=False):
    stat_dict = mystats(_list)
    stat_strs = ['%r: %s' % (key, val) for key, val in stat_dict.iteritems()]
    if newlines:
        indent = '    '
        head = '{\n' + indent
        sep  = ',\n' + indent
        tail = '\n}'
    else:
        head = '{'
        sep = ', '
        tail = '}'
    ret = head + sep.join(stat_strs) + tail
    return ret
#def mystats2_latex(mystats):
    #statdict_ = eval(mystats)


def mystats(_list):
    if len(_list) == 0:
        return {'empty_list': True}
    nparr = np.array(_list)
    min_val = nparr.min()
    max_val = nparr.max()
    nMin = np.sum(nparr == min_val)  # number of entries with min val
    nMax = np.sum(nparr == max_val)  # number of entries with min val
    return OrderedDict([('max',   np.float32(max_val)),
                        ('min',   np.float32(min_val)),
                        ('mean',  np.float32(nparr.mean())),
                        ('std',   np.float32(nparr.std())),
                        ('nMin',  np.int32(nMin)),
                        ('nMax',  np.int32(nMax)),
                        ('shape', repr(nparr.shape))])


def myprint(input=None, prefix='', indent='', lbl=''):
    if len(lbl) > len(prefix):
        prefix = lbl
    if len(prefix) > 0:
        prefix += ' '
    print_(indent + prefix + str(type(input)) + ' ')
    if isinstance(input, list):
        print(indent + '[')
        for item in iter(input):
            myprint(item, indent=indent + '  ')
        print(indent + ']')
    elif isinstance(input, str):
        print(input)
    elif isinstance(input, dict):
        print(printableVal(input))
    else:
        print(indent + '{')
        attribute_list = dir(input)
        for attr in attribute_list:
            if attr.find('__') == 0:
                continue
            val = str(input.__getattribute__(attr))
            #val = input[attr]
            # Format methods nicer
            #if val.find('built-in method'):
                #val = '<built-in method>'
            print(indent + '  ' + attr + ' : ' + val)
        print(indent + '}')


def info(var, lbl):
    if isinstance(var, np.ndarray):
        return npinfo(var, lbl)
    if isinstance(var, list):
        return listinfo(var, lbl)


def npinfo(ndarr, lbl='ndarr'):
    info = ''
    info += (lbl + ': shape=%r ; dtype=%r' % (ndarr.shape, ndarr.dtype))
    return info


def listinfo(list_, lbl='ndarr'):
    if not isinstance(list_, list):
        raise Exception('!!')
    info = ''
    type_set = set([])
    for _ in iter(list_):
        type_set.add(str(type(_)))
    info += (lbl + ': len=%r ; types=%r' % (len(list_), type_set))
    return info


#expected_type = np.float32
#expected_dims = 5
def numpy_list_num_bits(nparr_list, expected_type, expected_dims):
    num_bits = 0
    num_items = 0
    num_elemt = 0
    bit_per_item = {
        np.float32: 32,
        np.uint8: 8
    }[expected_type]
    for nparr in iter(nparr_list):
        arr_len, arr_dims = nparr.shape
        if not nparr.dtype.type is expected_type:
            msg = 'Expected Type: ' + repr(expected_type)
            msg += 'Got Type: ' + repr(nparr.dtype)
            raise Exception(msg)
        if arr_dims != expected_dims:
            msg = 'Expected Dims: ' + repr(expected_dims)
            msg += 'Got Dims: ' + repr(arr_dims)
            raise Exception(msg)
        num_bits += len(nparr) * expected_dims * bit_per_item
        num_elemt += len(nparr) * expected_dims
        num_items += len(nparr)
    return num_bits,  num_items, num_elemt


def runprofile(cmd, globals_=globals(), locals_=locals()):
    # Meliae # from meliae import loader # om = loader.load('filename.json') # s = om.summarize();
    import cProfile
    import os
    print('[util] Profiling Command: ' + cmd)
    cProfOut_fpath = 'OpenGLContext.profile'
    cProfile.runctx( cmd, globals_, locals_, filename=cProfOut_fpath)
    # RUN SNAKE
    print('[util] Profiled Output: ' + cProfOut_fpath)
    if sys.platform == 'win32':
        rsr_fpath = 'C:/Python27/Scripts/runsnake.exe'
    else:
        rsr_fpath = 'runsnake'
    view_cmd = rsr_fpath + ' "' + cProfOut_fpath + '"'
    os.system(view_cmd)
    return True


def memory_profile(with_gc=False):
    #http://stackoverflow.com/questions/2629680/deciding-between-subprocess-multiprocessing-and-thread-in-python
    import guppy
    if with_gc:
        garbage_collect()
    hp = guppy.hpy()
    print('[hpy] Waiting for heap output...')
    heap_output = hp.heap()
    print(heap_output)
    print('[hpy] total heap size: ' + byte_str2(heap_output.size))
    import resources2
    resources2.memstats()
    # Graphical Browser
    #hp.pb()


def disable_garbage_collection():
    import gc
    gc.disable()


def enable_garbage_collection():
    import gc
    gc.enable()


def garbage_collect():
    import gc
    gc.collect()


def get_object_size(obj):
    seen = set([])
    def _get_object_size(obj):
        if (obj is None or isinstance(obj, (str, int, bool, float))):
            return sys.getsizeof(obj)

        object_id = id(obj)
        if object_id in seen:
            return 0
        seen.add(object_id)

        totalsize = sys.getsizeof(obj)
        if isinstance(obj, np.ndarray):
            totalsize += obj.nbytes
        elif (isinstance(obj, (tuple, list, set, frozenset))):
            for item in obj:
                totalsize += _get_object_size(item)
        elif isinstance(obj, dict):
            try:
                for key, val in obj.iteritems():
                        totalsize += _get_object_size(key)
                        totalsize += _get_object_size(val)
            except RuntimeError:
                print(key)
                raise
        elif isinstance(obj, object) and hasattr(obj, '__dict__'):
            totalsize += _get_object_size(obj.__dict__)
            return totalsize
        return totalsize
    return _get_object_size(obj)


def print_object_size(obj, lbl=''):
    print(lbl + byte_str2(get_object_size(obj)))


def get_object_base():
    if '--dyn' in sys.argv:
        return DynStruct
    else:
        return object

#http://www.huyng.com/posts/python-performance-analysis/
#Once youve gotten your code setup with the @profile decorator, use kernprof.py to run your script.
#kernprof.py -l -v fib.py
