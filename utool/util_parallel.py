'''
There are a lot of fancier things we can do here.
A good thing to do would be to keep similar function calls
and use multiprocessing.Queues for the backend.
This way we can print out progress.
'''
from __future__ import print_function, division
import multiprocessing


__POOL__ = None


def init_pool(num_procs=None):
    global __POOL__
    if num_procs is None:
        num_procs = multiprocessing.cpu_count()
    print('[parallel] initializing pool with %d processes' % num_procs)
    if num_procs == 1:
        print('[parallel] num_procs=1, Will process in serial')
        __POOL__ = 1
        return
    assert __POOL__ is None, 'pool is a singleton. can only initialize once'
    assert multiprocessing.current_process().name, 'can only initialize from main process'
    __POOL__ = multiprocessing.Pool(processes=num_procs)
    #for key, val in __POOL__.__dict__.iteritems():
        #print('%s = %r' % (key, val))


def close_pool():
    global __POOL__
    print('[parallel] closing pool')
    if __POOL__ is not None:
        if __POOL__ != 1:
            __POOL__.close()
            __POOL__.join()
        __POOL__ = None


def _process_serial(func, args_list, verbose=False):
    return map(func, args_list)


def _process_parallel(func, args_list, verbose=False):
    results = __POOL__.map(func, args_list)
    return results


def _process_parallel2(func, args_list, verbose=False):
    results = __POOL__.map_async(func, args_list)
    return results


def process(func, args_list, verbose=False):
    _ = (len(args_list), func.func_name, __POOL__._processes)
    print('[parallel] executing %d %s tasks using %d processes' % _)
    if __POOL__ == 1:
        return _process_serial(func, args_list, verbose)
    else:
        return _process_parallel(func, args_list, verbose)
