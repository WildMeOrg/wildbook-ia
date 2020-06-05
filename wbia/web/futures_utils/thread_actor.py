# -*- coding: utf-8 -*-
""" Implements ThreadActor """
from __future__ import absolute_import, division, print_function
from concurrent.futures import _base
from concurrent.futures import thread
from wbia.web.futures_utils import _base_actor
import queue
import threading
import weakref
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)

# Most of this code is duplicated from the concurrent.futures.thread and
# concurrent.futures.process modules, writen by Brian Quinlan. The main
# difference is that we expose an `Actor` class which can be inherited from and
# provides the `executor` classmethod. This creates an asynchronously
# maintained instance of this class in a separate thread/process

__author__ = 'Jon Crall (erotemic@gmail.com)'


class _WorkItem(object):
    def __init__(self, future, message):
        self.future = future
        self.message = message


def _thread_actor_eventloop(executor_reference, work_queue, _ActorClass, *args, **kwargs):
    """
    actor event loop run in a separate thread.

    Creates the instance of the actor (passing in the required *args, and
    **kwargs). Then the eventloop starts and feeds the actor messages from the
    _call_queue. Results are placed in the _result_queue, which are then placed
    in Future objects.
    """
    try:
        actor = _ActorClass(*args, **kwargs)
        while True:
            work_item = work_queue.get(block=True)
            if work_item is not None:
                if work_item.future.set_running_or_notify_cancel():
                    # Send the message to the actor
                    try:
                        result = actor.handle(work_item.message)
                    except BaseException as e:
                        work_item.future.set_exception(e)
                        # Delete references to object.
                        del e
                    else:
                        work_item.future.set_result(result)
                # Delete references to object. See issue16284
                del work_item
                continue
            executor = executor_reference()
            # Exit if:
            #   - The interpreter is shutting down OR
            #   - The executor that owns the worker has been collected OR
            #   - The executor that owns the worker has been shutdown.
            if thread._shutdown or executor is None or executor._shutdown:
                # Notice other workers
                work_queue.put(None)
                return
            del executor
    except BaseException:
        _base.LOGGER.critical('Exception in worker', exc_info=True)


class ThreadActorExecutor(_base_actor.ActorExecutor):
    def __init__(self, _ActorClass, *args, **kwargs):
        """Initializes a new ThreadPoolExecutor instance.
        """
        self._ActorClass = _ActorClass
        self._work_queue = queue.Queue()
        self._threads = set()
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._did_initialize = False

        if args or kwargs:
            # If given actor initialization args we must start the Actor
            # immediately. Otherwise just wait until we get a message
            self._initialize_actor(*args, **kwargs)

    def post(self, message):
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')

            f = _base.Future()
            w = _WorkItem(f, message)

            self._work_queue.put(w)
            self._initialize_actor()
            return f

    post.__doc__ = _base_actor.ActorExecutor.post.__doc__

    def _initialize_actor(self, *args, **kwargs):
        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        # We only maintain one thread for an actor
        if len(self._threads) < 1:
            assert self._did_initialize is False, 'only initialize actor once'
            self._did_initialize = True
            t = threading.Thread(
                target=_thread_actor_eventloop,
                args=(weakref.ref(self, weakref_cb), self._work_queue, self._ActorClass)
                + args,
                kwargs=kwargs,
            )
            t.daemon = True
            t.start()
            self._threads.add(t)
            thread._threads_queues[t] = self._work_queue

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            self._shutdown = True
            self._work_queue.put(None)
        if wait:
            for t in self._threads:
                t.join()

    shutdown.__doc__ = _base.Executor.shutdown.__doc__


class ThreadActor(_base_actor.Actor):
    @classmethod
    def executor(cls, *args, **kwargs):
        return ThreadActorExecutor(cls, *args, **kwargs)

    # executor.__doc__ = _base_actor.Actor.executor.__doc___


# ThreadActor.__doc__ = _base_actor.Actor.__doc___
