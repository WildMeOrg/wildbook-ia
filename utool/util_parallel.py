'''
There are a lot of fancier things we can do here.
A good thing to do would be to keep similar function calls
and use multiprocessing.Queues for the backend.
This way we can print out progress.
'''
from __future__ import absolute_import, division, print_function
import multiprocessing
import atexit
import sys
import signal
from .util_progress import progress_func
from .util_time import tic, toc
from . import util_arg
from .util_cplat import WIN32
from .util_dbg import printex


QUIET   = util_arg.QUIET
VERBOSE = util_arg.VERBOSE

__POOL__ = None
__TIME__ = '--time' in sys.argv
__SERIAL_FALLBACK__ = '--noserial-fallback' not in sys.argv
__NUM_PROCS__ = util_arg.get_arg('--num-procs', int, default=None)
__FORCE_SERIAL__ = util_arg.get_flag('--utool-force-serial')


def get_default_numprocs():
    if __NUM_PROCS__ is not None:
        return __NUM_PROCS__
    if WIN32:
        num_procs = 3  # default windows to 3 processes for now
    else:
        num_procs = max(multiprocessing.cpu_count() - 2, 1)
    return num_procs


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def init_pool(num_procs=None, maxtasksperchild=None):
    global __POOL__
    if num_procs is None:
        # Get number of cpu cores
        num_procs = get_default_numprocs()
    if not QUIET:
        print('[parallel] initializing pool with %d processes' % num_procs)
    if num_procs == 1:
        print('[parallel] num_procs=1, Will process in serial')
        __POOL__ = 1
        return
    if '--strict' in sys.argv:
        #assert __POOL__ is None, 'pool is a singleton. can only initialize once'
        assert multiprocessing.current_process().name, 'can only initialize from main process'
    if __POOL__ is not None:
        print('close pool before reinitializing')
        return
    # Create the pool of processes
    __POOL__ = multiprocessing.Pool(processes=num_procs, initializer=init_worker,
                                    maxtasksperchild=maxtasksperchild)
    #for key, val in __POOL__.__dict__.iteritems():
    #    print('%s = %r' % (key, val))


def close_pool():
    global __POOL__
    if __POOL__ is not None:
        if not QUIET:
            print('[parallel] closing pool')
        if not isinstance(__POOL__, int):
            # Must join after close to avoid runtime errors
            __POOL__.close()
            __POOL__.join()
        __POOL__ = None


def _process_serial(func, args_list, args_dict={}):
    num_tasks = len(args_list)
    result_list = []
    mark_prog, end_prog = progress_func(max_val=num_tasks,
                                        lbl=func.func_name + ': ')
    mark_prog(0)
    # Execute each task sequentially
    for count, args in enumerate(args_list):
        result = func(*args, **args_dict)
        result_list.append(result)
        mark_prog(count)
    end_prog()
    return result_list


def _process_parallel(func, args_list, args_dict={}):
    # Define progress observers
    num_tasks = len(args_list)
    num_tasks_returned_ptr = [0]
    mark_prog, end_prog = progress_func(max_val=num_tasks,
                                        lbl=func.func_name + ': ')
    def _callback(result):
        mark_prog(num_tasks_returned_ptr[0])
        sys.stdout.flush()
        num_tasks_returned_ptr[0] += 1
    # Send all tasks to be executed asynconously
    apply_results = [__POOL__.apply_async(func, args, args_dict, _callback)
                     for args in args_list]
    # Wait until all tasks have been processed
    while num_tasks_returned_ptr[0] < num_tasks:
        #print('Waiting: ' + str(num_tasks_returned_ptr[0]) + '/' + str(num_tasks))
        pass
    end_prog()
    # Get the results
    result_list = [ap.get() for ap in apply_results]
    return result_list


def _generate_parallel(func, args_list, ordered=False, chunksize=1):
    print('[parallel] executing %d %s tasks using %d processes' %
            (len(args_list), func.func_name, __POOL__._processes))
    mark_prog, end_prog = progress_func(max_val=len(args_list), lbl=func.func_name + ': ')
    if ordered:
        generator = __POOL__.imap_unordered(func, args_list, chunksize)
    else:
        generator = __POOL__.imap(func, args_list, chunksize)
    try:
        for count, result in enumerate(generator):
            mark_prog(count)
            yield result
    except Exception as ex:
        printex(ex, 'Parallel Generation Failed!', '[utool]')
        print('__SERIAL_FALLBACK__ = %r' % __SERIAL_FALLBACK__)
        if __SERIAL_FALLBACK__:
            for result in _generate_serial(func, args_list, mark_prog):
                yield result
        else:
            raise
    end_prog()


def _generate_serial(func, args_list):
    print('[parallel] executing %d %s tasks in serial' %
            (len(args_list), func.func_name))
    mark_prog, end_prog = progress_func(max_val=len(args_list), lbl=func.func_name + ': ')
    for count, args in enumerate(args_list):
        mark_prog(count)
        result = func(args)
        yield result
    end_prog()


def ensure_pool():
    try:
        assert __POOL__ is not None, 'must init_pool() first'
    except AssertionError as ex:
        print('(WARNING) AssertionError: ' + str(ex))
        init_pool()


def generate(func, args_list, ordered=False, force_serial=__FORCE_SERIAL__):
    """ Returns a generator which asynchronously returns results """
    if len(args_list) == 0:
        print('[parallel] submitted 0 tasks')
        return []
    ensure_pool()
    if __TIME__:
        tt = tic(func.func_name)
    if isinstance(__POOL__, int) or force_serial:
        return _generate_serial(func, args_list, ordered=ordered)
    else:
        return _generate_parallel(func, args_list)
    if __TIME__:
        toc(tt)


def process(func, args_list, args_dict={}, force_serial=__FORCE_SERIAL__):
    ensure_pool()
    if __POOL__ == 1 or force_serial:
        if not QUIET:
            print('[parallel] executing %d %s tasks in serial' %
                  (len(args_list), func.func_name))
        result_list = _process_serial(func, args_list, args_dict)
    else:
        print('[parallel] executing %d %s tasks using %d processes' %
              (len(args_list), func.func_name, __POOL__._processes))
        result_list = _process_parallel(func, args_list, args_dict)
    return result_list

atexit.register(close_pool)
