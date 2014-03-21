from __future__ import division, print_function
from .util_inject import inject
import pylru  # because we dont have functools.lru_cache
print, print_, printDBG, rrr, profile = inject(__name__, '[cache]')


class lru_cache(object):
    # because we dont have functools.lru_cache
    def __init__(cache, max_size=100):
        cache.max_size = max_size
        cache.cache_ = pylru.lrucache(max_size)
        cache.func_name = None

    def clear_cache(cache):
        printDBG('[cache.lru] clearing %r lru_cache' % (cache.func_name,))
        cache.cache_.clear()

    def __call__(cache, func):
        def wrapped(self, *args):  # wrap a class
            try:
                value = cache.cache_[args]
                printDBG(func.func_name + '(%r) ...lrucache hit' % (args,))
                return value
            except KeyError:
                printDBG(func.func_name + '(%r) ...lrucache miss' % (args,))

            value = func(self, *args)
            cache.cache_[args] = value
            return value
        cache.func_name = func.func_name
        printDBG('[@tools.lru] wrapping %r with max_size=%r lru_cache' %
                 (cache.func_name, cache.max_size))
        wrapped.func_name = func.func_name
        wrapped.clear_cache = cache.clear_cache
        return wrapped
