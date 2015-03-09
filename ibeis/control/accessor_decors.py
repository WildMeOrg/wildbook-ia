from __future__ import absolute_import, division, print_function
# import decorator  # NOQA
import six
import utool as ut
from six.moves import builtins
from utool._internal.meta_util_six import get_funcname
#import numpy as np
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[decor]')

DEBUG_ADDERS  = ut.get_argflag('--debug-adders')
DEBUG_SETTERS = ut.get_argflag('--debug-setters')
DEBUG_GETTERS = ut.get_argflag('--debug-getters')
VERB_CONTROL = ut.get_argflag('--verb-control')

#
#-----------------
# IBEIS DECORATORS
#-----------------


# DECORATORS::OTHERS

def default_decorator(input_):
    """
    This should be the first decorator applied to all Controller functions
    """
    func_ = input_
    return profile(func_)
    #return ut.indent_func(profile(func_))
    #return func_


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

DEV_CACHE = ut.get_argflag(('--dev-cache', '--devcache'))

RELEASE_MODE = True
if RELEASE_MODE:
    API_CACHE = not ut.get_argflag('--no-api-cache')
    ASSERT_API_CACHE = ut.get_argflag(('--assert-api-cache', '--naac'))
else:
    API_CACHE = ut.get_argflag('--api-cache')
    ASSERT_API_CACHE = not ut.get_argflag(('--noassert-api-cache', '--naac'))

if ut.VERBOSE:
    if ut.in_main_process():
        if API_CACHE:
            print('[accessor_decors] API_CACHE IS ENABLED')
        else:
            print('[accessor_decors] API_CACHE IS DISABLED')


def init_tablecache():
    r"""
    Returns:
       defaultdict: tablecache

    CommandLine:
        python -m ibeis.control.accessor_decors --test-init_tablecache

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.accessor_decors import *  # NOQA
        >>> result = init_tablecache()
        >>> print(result)
    """
    #return ut.ddict(ColumnsCache)
    # 4 levels of dictionaries
    # tablename, colname, kwargs, and then rowids
    tablecache = ut.ddict(lambda: ut.ddict(lambda: ut.ddict(dict)))
    return tablecache


def cache_getter(tblname, colname, cfgkeys=None, force=False, debug=False, native_rowids=True):
    """
    Creates a getter cacher
    the class must have a table_cache property
    varargs are currently unallowed

    Args:
        tblname (str):
        colname (str):

    Returns:
        function: closure_getter_cacher

    CommandLine:
        python -m ibeis.control.accessor_decors --test-cache_getter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.accessor_decors import *  # NOQA
        >>> import ibeis
        >>> from ibeis import constants as const
        >>> ibs = ibeis.opendb('testdb1')
        >>> #ibs = ibeis.opendb('PZ_MTEST')
        >>> valid_nids = ibs.get_valid_nids()
        >>> tblname = const.NAME_TABLE
        >>> colname = 'annot_rowid'
        >>> rowid_list = valid_nids
        >>> rowid_list1 = rowid_list[::2]
        >>> rowid_list2 = rowid_list[::3]
        >>> rowid_list3 = rowid_list[1::2]
        >>> kwargs = {}
        >>> getter_func = ibs.get_name_aids.im_func
        >>> wrp_getter_cacher = cache_getter(tblname, colname, force=True, debug=False)(getter_func)
        >>> ### Test Getter (caches)
        >>> val_list1 = getter_func(ibs, rowid_list1)
        >>> val_list2 = wrp_getter_cacher(ibs, rowid_list1)
        >>> print(ut.dict_str(ibs.table_cache))
        >>> val_list3 = wrp_getter_cacher(ibs, rowid_list1)
        >>> val_list4 = wrp_getter_cacher(ibs, rowid_list2)
        >>> print(ut.dict_str(ibs.table_cache))
        >>> val_list5 = wrp_getter_cacher(ibs, rowid_list3)
        >>> val_list  = wrp_getter_cacher(ibs, rowid_list)
        >>> ut.assert_eq(val_list1, val_list2, 'run1')
        >>> ut.assert_eq(val_list1, val_list2, 'run2')
        >>> print(ut.dict_str(ibs.table_cache))
        >>> ### Test Setter (invalidates)
        >>> setter_func = ibs.set_name_texts
        >>> wrp_cache_invalidator = cache_invalidator(tblname, force=True)(lambda *a: None)
        >>> wrp_cache_invalidator(ibs, rowid_list1)
        >>> print(ut.dict_str(ibs.table_cache))

    Ignore:
        %timeit getter_func(ibs, rowid_list)
        %timeit wrp_getter_cacher(ibs, rowid_list)
    """
    def closure_getter_cacher(getter_func):
        if not API_CACHE and not force:
            # Turn of API Cache
            return getter_func

        def debug_cache_hits(ismiss_list, rowid_list):
            num_miss  = sum(ismiss_list)
            num_total = len(rowid_list)
            num_hit   = num_total - num_miss
            print('\n[get] %s.%s %d / %d cache hits' %
                  (tblname, colname, num_hit, num_total))

        def assert_cache_hits(ibs, ismiss_list, rowid_list, kwargs_hash, **kwargs):
            cached_rowid_list = ut.filterfalse_items(rowid_list, ismiss_list)
            cache_ = ibs.table_cache[tblname][colname][kwargs_hash]
            # Load cached values for each rowid
            cache_vals_list = ut.dict_take_list(cache_, cached_rowid_list, None)
            db_vals_list = getter_func(ibs, cached_rowid_list, **kwargs)
            # Assert everything is valid
            msg_fmt = ut.codeblock(
                '''
                [assert_cache_hits] tblname = %r
                [assert_cache_hits] colname = %r
                [assert_cache_hits] cfgkeys = %r
                [assert_cache_hits] CACHE INVALID: %r != %r
                '''
            )
            msg = msg_fmt % (tblname, colname, cfgkeys, cache_vals_list, db_vals_list, )
            try:
                list1 = cache_vals_list
                list2 = db_vals_list
                assert ut.lists_eq(list1, list2), msg
                #if isinstance(db_vals_list, list):
                #    assert cache_vals_list == db_vals_list, msg
                #else:
                #    assert np.all(cache_vals_list == db_vals_list), msg
            except AssertionError as ex:
                raise ex
            except Exception as ex2:
                print(type(cache_vals_list))
                print(type(db_vals_list))
                ut.printex(ex2)
                ut.embed()
                raise

        #@profile cannot profile this because it is alrady being profiled by
        def wrp_getter_cacher(ibs, rowid_list, **kwargs):
            """
            Wrapper function that caches rowid values in a dictionary
            """
            # HACK TAKE OUT GETTING DEBUG OUT OF KWARGS
            debug_ = kwargs.pop('debug', False)
            if cfgkeys is not None:
                kwargs_hash = ut.get_dict_hashid(ut.dict_take_list(kwargs, cfgkeys, None))
            else:
                kwargs_hash = None

            cache_ = ibs.table_cache[tblname][colname][kwargs_hash]
            # Load cached values for each rowid
            vals_list = ut.dict_take_list(cache_, rowid_list, None)
            # Mark rowids with cache misses
            ismiss_list = [val is None for val in vals_list]
            if debug or debug_:
                # debug_cache_hits(ismiss_list, rowid_list)
                print('[cache_getter] "debug_cache_hits" turned off')
            # HACK !!! DEBUG THESE GETTERS BY ASSERTING INFORMATION IN CACHE IS CORRECT
            with_assert = ASSERT_API_CACHE
            if with_assert:
                assert_cache_hits(ibs, ismiss_list, rowid_list, kwargs_hash, **kwargs)
            # END HACK
            if any(ismiss_list):
                miss_indices = ut.list_where(ismiss_list)
                miss_rowids  = ut.filter_items(rowid_list, ismiss_list)
                # call wrapped function
                miss_vals = getter_func(ibs, miss_rowids, **kwargs)
                # overwrite missed output
                for index, val in zip(miss_indices, miss_vals):
                    vals_list[index] = val  # Output write
                # cache save
                for rowid, val in zip(miss_rowids, miss_vals):
                    cache_[rowid] = val     # Cache writd(ne
            return vals_list
        wrp_getter_cacher = ut.preserve_sig(wrp_getter_cacher, getter_func)
        return wrp_getter_cacher
    return closure_getter_cacher


def cache_invalidator(tblname, colnames=None, native_rowids=False, force=False):
    """ cacher setter/deleter decorator
    FIXME: this is more general than just setters
    """
    def closure_cache_invalidator(setter_func):
        if not API_CACHE and not force:
            return setter_func
        def wrp_cache_invalidator(self, rowid_list, *args, **kwargs):
            # the class must have a table_cache property
            colscache_ = self.table_cache[tblname]
            colnames_ =  list(six.iterkeys(colscache_)) if colnames is None else colnames
            # Clear the cache of any specified colname
            # when the invalidator is called
            for colname in colnames_:
                kwargs_cache_ = colscache_[colname]
                if not native_rowids:
                    # We dont know the rowsids so clear everything
                    for cache_ in six.itervalues(kwargs_cache_):
                        cache_.clear()
                else:
                    # We know the rowids to delete
                    # iterate over all getter kwargs values
                    for cache_ in six.itervalues(kwargs_cache_):
                        ut.delete_dict_keys(cache_, rowid_list)
            # Preform set/delete action
            return setter_func(self, rowid_list, *args, **kwargs)
        wrp_cache_invalidator = ut.preserve_sig(wrp_cache_invalidator, setter_func)
        return wrp_cache_invalidator
    return closure_cache_invalidator


def dev_cache_getter(tblname, colname, *args, **kwargs):
    """ cache getter for when the database is gaurenteed not to change """
    def closure_dev_getter_cacher(getter_func):
        if not DEV_CACHE:
            return getter_func
        return cache_getter(tblname, colname, *args, **kwargs)(getter_func)
    return closure_dev_getter_cacher


# alias for easy removal of dev functions.
# dev case is same as normal case though
dev_cache_invalidator = cache_invalidator


#@decorator.decorator
def adder(func):
    func_ = default_decorator(func)
    #@ut.on_exception_report_input
    @ut.accepts_scalar_input
    @ut.ignores_exc_tb
    def wrp_adder(*args, **kwargs):
        if DEBUG_ADDERS or VERB_CONTROL:
            print('+------')
            print('[ADD]: ' + get_funcname(func))
            funccall_str = ut.func_str(func, args, kwargs, packed=True)
            print('\n' + funccall_str + '\n')
            print('L------')
        if VERB_CONTROL:
            print('[ADD]: ' + get_funcname(func))
            builtins.print('\n' + ut.func_str(func, args, kwargs) + '\n')
        return func_(*args, **kwargs)
    wrp_adder = ut.preserve_sig(wrp_adder, func)
    wrp_adder = ut.on_exception_report_input(wrp_adder)
    return wrp_adder


# DECORATORS::DELETER

#@decorator.decorator
def deleter(func):
    func_ = default_decorator(func)
    @ut.accepts_scalar_input
    @ut.ignores_exc_tb
    def wrp_deleter(*args, **kwargs):
        if VERB_CONTROL:
            print('[DELETE]: ' + get_funcname(func))
            builtins.print('\n' + ut.func_str(func, args, kwargs) + '\n')
        return func_(*args, **kwargs)
    wrp_deleter = ut.preserve_sig(wrp_deleter, func)
    return wrp_deleter


# DECORATORS::SETTER

#@decorator.decorator
def setter_general(func):
    func = default_decorator(func)
    return func


#@decorator.decorator
def setter(func):
    func_ = default_decorator(func)
    @ut.accepts_scalar_input2(argx_list=[0, 1], outer_wrapper=False)
    #@ut.accepts_scalar_input2(argx_list=range(0, 2))
    #@ut.accepts_scalar_input2(argx_list=range(1, 2))
    #@ut.on_exception_report_input
    @ut.ignores_exc_tb
    def wrp_setter(*args, **kwargs):
        if DEBUG_SETTERS or VERB_CONTROL:
            print('+------')
            print('[SET]: ' + get_funcname(func))
            print('[SET]: called by: ' + ut.get_caller_name(range(1, 7)))
            funccall_str = ut.func_str(func, args, kwargs, packed=True)
            print('\n' + funccall_str + '\n')
            print('L------')
            #builtins.print('\n' + funccall_str + '\n')
        #print('set: funcname=%r, args=%r, kwargs=%r' % (get_funcname(func), args, kwargs))
        return func_(*args, **kwargs)
    wrp_setter = ut.preserve_sig(wrp_setter, func)
    wrp_setter = ut.on_exception_report_input(wrp_setter)
    return wrp_setter


# DECORATORS::GETTER

def getter(func):
    """
    Getter decorator for functions which takes as the first input a unique id
    list and returns a heterogeous list of values
    """
    #func_ = func
    func_ = default_decorator(func)
    #@ut.on_exception_report_input
    @ut.accepts_scalar_input
    @ut.ignores_exc_tb
    def wrp_getter(*args, **kwargs):
        #if ut.DEBUG:
        #    print('[IN GETTER] args=%r' % (args,))
        #    print('[IN GETTER] kwargs=%r' % (kwargs,))
        if DEBUG_GETTERS  or VERB_CONTROL:
            print('+------')
            print('[GET]: ' + get_funcname(func))
            funccall_str = ut.func_str(func, args, kwargs, packed=True)
            print('\n' + funccall_str + '\n')
            print('L------')
        return func_(*args, **kwargs)
    wrp_getter = ut.preserve_sig(wrp_getter, func)
    wrp_getter = ut.on_exception_report_input(wrp_getter)
    return wrp_getter


#@decorator.decorator
def getter_vector_output(func):
    """
    Getter decorator for functions which takes as the first input a unique id
    list and returns a homogenous list of values
    """
    func_ = default_decorator(func)
    @ut.accepts_scalar_input_vector_output
    @ut.ignores_exc_tb
    def getter_vector_wrp(*args, **kwargs):
        return func_(*args, **kwargs)
    getter_vector_wrp = ut.preserve_sig(getter_vector_wrp, func)
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
    @ut.accepts_numpy
    #@ut.on_exception_report_input
    @ut.accepts_scalar_input
    @ut.ignores_exc_tb
    def getter_numpy_wrp(*args, **kwargs):
        return func_(*args, **kwargs)
    getter_numpy_wrp = ut.preserve_sig(getter_numpy_wrp, func)
    getter_numpy_wrp = ut.on_exception_report_input(getter_numpy_wrp)
    return getter_numpy_wrp


#@decorator.decorator
def getter_numpy_vector_output(func):
    """ Getter decorator for functions which takes as the first input a unique
    id list and returns a heterogeous list of values """
    #getter_func = getter_vector_output(func)
    func_ = default_decorator(func)
    @ut.accepts_numpy
    @ut.accepts_scalar_input_vector_output
    @ut.ignores_exc_tb
    def getter_numpy_vector_wrp(*args, **kwargs):
        return func_(*args, **kwargs)
    getter_numpy_vector_wrp = ut.preserve_sig(getter_numpy_vector_wrp, func)
    return getter_numpy_vector_wrp


def ider(func):
    """ This function takes returns ids subject to conditions """
    ider_func = default_decorator(func)
    ider_func = ut.preserve_sig(ider_func, func)
    return ider_func
