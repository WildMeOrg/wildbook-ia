from __future__ import absolute_import, division, print_function
import utool
from six.moves import builtins
from functools import wraps
from utool._internal.meta_util_six import get_funcname
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[decor]')

#
#-----------------
# IBEIS DECORATORS
#-----------------


# DECORATORS::OTHERS

def default_decorator(input_):
    if utool.is_funclike(input_):
        func_ = input_
        #return utool.indent_func(profile(func_))
        return profile(func_)
        #return func_
    else:
        def closure_default(func):
            return utool.indent_func(input_)(profile(func))
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


API_CACHE = utool.get_flag('--api-cache')
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

        return wrp_getter_cacher
    return closure_getter_cacher


def cache_invalidator(tblname, colnames=None):
    """ cacher setter decorator """
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
        return wrp_cache_invalidator
    return closure_cache_invalidator


def adder(func):
    func = default_decorator(func)
    @utool.accepts_scalar_input
    @utool.ignores_exc_tb
    @wraps(func)
    def wrp_adder(*args, **kwargs):
        if not utool.QUIET and utool.VERBOSE:
            print('[ADD]: ' + get_funcname(func))
            builtins.print('\n' + utool.func_str(func, args, kwargs) + '\n')
        return func(*args, **kwargs)
    return wrp_adder


# DECORATORS::DELETER

def deleter(func):
    func = default_decorator(func)
    @utool.accepts_scalar_input
    @utool.ignores_exc_tb
    @wraps(func)
    def wrp_deleter(*args, **kwargs):
        if not utool.QUIET and utool.VERBOSE:
            print('[DELETE]: ' + get_funcname(func))
            builtins.print('\n' + utool.func_str(func, args, kwargs) + '\n')
        return func(*args, **kwargs)
    return wrp_deleter


# DECORATORS::SETTER

def setter_general(func):
    func = default_decorator(func)
    return func


def setter(func):
    func = default_decorator(func)
    @utool.accepts_scalar_input2(argx_list=range(0, 2))
    @utool.ignores_exc_tb
    @wraps(func)
    def wrp_setter(*args, **kwargs):
        if not utool.QUIET and utool.VERBOSE:
            print('[SET]: ' + get_funcname(func))
            builtins.print('\n' + utool.func_str(func, args, kwargs) + '\n')
        #print('set: funcname=%r, args=%r, kwargs=%r' % (get_funcname(func), args, kwargs))
        return func(*args, **kwargs)
    return wrp_setter


# DECORATORS::GETTER

def getter(func):
    """
    Getter decorator for functions which takes as the first input a unique id
    list and returns a heterogeous list of values
    """
    func = default_decorator(func)
    @utool.on_exception_report_input
    @utool.accepts_scalar_input
    @utool.ignores_exc_tb
    @wraps(func)
    def wrp_getter(*args, **kwargs):
        return func(*args, **kwargs)
    return wrp_getter


def getter_vector_output(func):
    """
    Getter decorator for functions which takes as the first input a unique id
    list and returns a homogenous list of values
    """
    func = default_decorator(func)
    @utool.accepts_scalar_input_vector_output
    @utool.ignores_exc_tb
    @wraps(func)
    def getter_vector_wrp(*args, **kwargs):
        return func(*args, **kwargs)
    return getter_vector_wrp

getter_1toM = getter_vector_output
getter_1to1 = getter
getter_1to1 = getter


def getter_numpy(func):
    """
    Getter decorator for functions which takes as the first input a unique id
    list and returns a heterogeous list of values
    """
    getter_func = getter(func)
    @utool.accepts_numpy
    @wraps(func)
    def getter_numpy_wrp(*args, **kwargs):
        return getter_func(*args, **kwargs)
    return getter_numpy_wrp


def getter_numpy_vector_output(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a heterogeous list of values """
    getter_func = getter_vector_output(func)
    @utool.accepts_numpy
    @wraps(func)
    def getter_numpy_vector_wrp(*args, **kwargs):
        return getter_func(*args, **kwargs)
    return getter_numpy_vector_wrp


def ider(func):
    """ This function takes returns ids subject to conditions """
    return default_decorator(func)
