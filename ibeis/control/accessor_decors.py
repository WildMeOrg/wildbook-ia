from __future__ import absolute_import, division, print_function
# import decorator  # NOQA
import utool
from six.moves import builtins
from utool._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[decor]')

DEBUG_ADDERS  = utool.get_argflag('--debug-adders')
DEBUG_SETTERS = utool.get_argflag('--debug-setters')
DEBUG_GETTERS = utool.get_argflag('--debug-getters')

#
#-----------------
# IBEIS DECORATORS
#-----------------


# DECORATORS::OTHERS

def default_decorator(input_):
    if utool.is_funclike(input_):
        func_ = input_
        #return utool.indent_func(profile(func_))
        #return profile(func_)
        return func_
    else:
        #@decorator.decorator
        def closure_default(func):
            #return utool.indent_func(input_)(profile(func))
            return utool.indent_func(input_)(func)
        return closure_default


# DECORATORS::ADDER
#TABLE_CACHE = {}


#class ColumnsCache(object):
#    def __init__(self):
#        self._cache = {}

#    def __setitem__(self, index, value):
#        self._cache[index] = value

#    def __getitem__(self, index):
#        return self._cache[index]

#    def __delitem__(self, index):
#        del self._cache[index]


API_CACHE = utool.get_argflag('--api-cache')
if utool.in_main_process():
    if API_CACHE:
        print('[accessor_decors] API_CACHE IS ENABLED')
    else:
        #print('[accessor_decors] API_CACHE IS DISABLED')
        pass


def init_tablecache():
    #return utool.ddict(ColumnsCache)
    return utool.ddict(lambda: utool.ddict(dict))


def _delete_items(dict_, key_list):
    invalid_keys = iter(set(key_list) - set(dict_.rows()))
    for key in invalid_keys:
        del dict_[key]


def cache_getter(tblname, colname):
    """ Creates a getter cacher """
    #@decorator.decorator
    def closure_getter_cacher(getter_func):
        getter_func = profile(getter_func)  # Autoprofilehack
        if not API_CACHE:
            return getter_func
        def wrp_getter_cacher(self, rowid_list, *args, **kwargs):
            # the class must have a table_cache property
            cache_ = self.table_cache[tblname][colname]
            # Get cached values for each rowid
            vals_list = [cache_.get(rowid, None) for rowid in rowid_list]
            # Compute any cache misses
            miss_list = [val is None for val in vals_list]
            #DEBUG_CACHE_HITS = False
            #if DEBUG_CACHE_HITS:
            #    num_miss  = sum(miss_list)
            #    num_total = len(rowid_list)
            #    num_hit   = num_total - num_miss
            #    print('\n[get] %s.%s %d / %d cache hits' % (tblname, colname, num_hit, num_total))
            if not any(miss_list):
                return vals_list
            miss_rowid_list = utool.filter_items(rowid_list, miss_list)
            miss_vals = getter_func(self, miss_rowid_list, *args, **kwargs)
            # Write the misses to the cache
            miss_iter_ = iter(enumerate(iter(miss_vals)))
            for index, flag in enumerate(miss_list):
                if flag:
                    miss_index, miss_val = miss_iter_.next()
                    rowid = rowid_list[index]
                    vals_list[index] = miss_val  # Output write
                    cache_[rowid] = miss_val  # Cache write
            return vals_list
        wrp_getter_cacher = utool.preserve_sig(wrp_getter_cacher, getter_func)
        return wrp_getter_cacher
    return closure_getter_cacher


def cache_invalidator(tblname, colnames=None):
    """ cacher setter decorator """
    #@decorator.decorator
    def closure_cache_invalidator(setter_func):
        if not API_CACHE:
            return setter_func
        def wrp_cache_invalidator(self, rowid_list, *args, **kwargs):
            # the class must have a table_cache property
            colscache_ = self.table_cache[tblname]
            colnames_ =  colscache_.keys() if colnames is None else colnames
            # Delete the cached values for the rowids in these columns of this table
            for colname in colnames_:
                cache_ = colscache_[colname]
                _delete_items(cache_, rowid_list)
            # Preform set action
            setter_func(self, rowid_list, *args, **kwargs)
        wrp_cache_invalidator = utool.preserve_sig(wrp_cache_invalidator, setter_func)
        return wrp_cache_invalidator
    return closure_cache_invalidator


#@decorator.decorator
def adder(func):
    func_ = default_decorator(func)
    #@utool.on_exception_report_input
    @utool.accepts_scalar_input
    @utool.ignores_exc_tb
    def wrp_adder(*args, **kwargs):
        if DEBUG_ADDERS or (not utool.QUIET and utool.VERYVERBOSE):
            print('+------')
            print('[ADD]: ' + get_funcname(func))
            funccall_str = utool.func_str(func, args, kwargs, packed=True)
            print('\n' + funccall_str + '\n')
            print('L------')
        if not utool.QUIET and utool.VERYVERBOSE:
            print('[ADD]: ' + get_funcname(func))
            builtins.print('\n' + utool.func_str(func, args, kwargs) + '\n')
        return func_(*args, **kwargs)
    wrp_adder = utool.preserve_sig(wrp_adder, func)
    wrp_adder = utool.on_exception_report_input(wrp_adder)
    return wrp_adder


# DECORATORS::DELETER

#@decorator.decorator
def deleter(func):
    func_ = default_decorator(func)
    @utool.accepts_scalar_input
    @utool.ignores_exc_tb
    def wrp_deleter(*args, **kwargs):
        if not utool.QUIET and utool.VERYVERBOSE:
            print('[DELETE]: ' + get_funcname(func))
            builtins.print('\n' + utool.func_str(func, args, kwargs) + '\n')
        return func_(*args, **kwargs)
    wrp_deleter = utool.preserve_sig(wrp_deleter, func)
    return wrp_deleter


# DECORATORS::SETTER

#@decorator.decorator
def setter_general(func):
    func = default_decorator(func)
    return func


#@decorator.decorator
def setter(func):
    func_ = default_decorator(func)
    @utool.accepts_scalar_input2(argx_list=[0, 1])
    #@utool.accepts_scalar_input2(argx_list=range(0, 2))
    #@utool.accepts_scalar_input2(argx_list=range(1, 2))
    #@utool.on_exception_report_input
    @utool.ignores_exc_tb
    def wrp_setter(*args, **kwargs):
        if DEBUG_SETTERS or (not utool.QUIET and utool.VERYVERBOSE):
            print('+------')
            print('[SET]: ' + get_funcname(func))
            funccall_str = utool.func_str(func, args, kwargs, packed=True)
            print('\n' + funccall_str + '\n')
            print('L------')
            #builtins.print('\n' + funccall_str + '\n')
        #print('set: funcname=%r, args=%r, kwargs=%r' % (get_funcname(func), args, kwargs))
        return func_(*args, **kwargs)
    wrp_setter = utool.preserve_sig(wrp_setter, func)
    wrp_setter = utool.on_exception_report_input(wrp_setter)
    return wrp_setter


# DECORATORS::GETTER

def getter(func):
    """
    Getter decorator for functions which takes as the first input a unique id
    list and returns a heterogeous list of values
    """
    #func_ = func
    func_ = default_decorator(func)
    #@utool.on_exception_report_input
    @utool.accepts_scalar_input
    @utool.ignores_exc_tb
    def wrp_getter(*args, **kwargs):
        #if utool.DEBUG:
        #    print('[IN GETTER] args=%r' % (args,))
        #    print('[IN GETTER] kwargs=%r' % (kwargs,))
        if DEBUG_GETTERS or (not utool.QUIET and utool.VERYVERBOSE):
            print('+------')
            print('[GET]: ' + get_funcname(func))
            funccall_str = utool.func_str(func, args, kwargs, packed=True)
            print('\n' + funccall_str + '\n')
            print('L------')
        return func_(*args, **kwargs)
    wrp_getter = utool.preserve_sig(wrp_getter, func)
    wrp_getter = utool.on_exception_report_input(wrp_getter)
    return wrp_getter


#@decorator.decorator
def getter_vector_output(func):
    """
    Getter decorator for functions which takes as the first input a unique id
    list and returns a homogenous list of values
    """
    func_ = default_decorator(func)
    @utool.accepts_scalar_input_vector_output
    @utool.ignores_exc_tb
    def getter_vector_wrp(*args, **kwargs):
        return func_(*args, **kwargs)
    getter_vector_wrp = utool.preserve_sig(getter_vector_wrp, func)
    return getter_vector_wrp

getter_1toM = getter_vector_output
getter_1to1 = getter
getter_1to1 = getter


#@decorator.decorator
def getter_numpy(func):
    """
    Getter decorator for functions which takes as the first input a unique id
    list and returns a heterogeous list of values
    """
    #getter_func = getter(func)
    func_ = default_decorator(func)
    @utool.accepts_numpy
    #@utool.on_exception_report_input
    @utool.accepts_scalar_input
    @utool.ignores_exc_tb
    def getter_numpy_wrp(*args, **kwargs):
        return func_(*args, **kwargs)
    getter_numpy_wrp = utool.preserve_sig(getter_numpy_wrp, func)
    getter_numpy_wrp = utool.on_exception_report_input(getter_numpy_wrp)
    return getter_numpy_wrp


#@decorator.decorator
def getter_numpy_vector_output(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a heterogeous list of values """
    #getter_func = getter_vector_output(func)
    func_ = default_decorator(func)
    @utool.accepts_numpy
    @utool.accepts_scalar_input_vector_output
    @utool.ignores_exc_tb
    def getter_numpy_vector_wrp(*args, **kwargs):
        return func_(*args, **kwargs)
    getter_numpy_vector_wrp = utool.preserve_sig(getter_numpy_vector_wrp, func)
    return getter_numpy_vector_wrp


def ider(func):
    """ This function takes returns ids subject to conditions """
    ider_func = default_decorator(func)
    ider_func = utool.preserve_sig(ider_func, func)
    return ider_func
