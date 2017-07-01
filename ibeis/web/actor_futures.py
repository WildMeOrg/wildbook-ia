"""
A simple actor model using multiprocessing Futures
"""
from concurrent import futures
from concurrent.futures import _base
import os
from multiprocessing.connection import wait
import queue
import weakref
import threading
import multiprocessing


class ActorManager(multiprocessing.Process):
    """
    ActorManager manages a single actor.
    """
    def __init__(self, _ActorClass, _task_queue, _result_queue):
        super(ActorManager, self).__init__()
        self._task_queue = _task_queue
        self._result_queue = _result_queue
        self._ActorClass = _ActorClass

    def run(self):
        """ actor event loop """

        actor = self._ActorClass()
        while True:
            call_item = self._task_queue.get(block=True)
            if call_item is None:
                # Wake up queue management thread
                self._result_queue.put(os.getpid())
                return
            try:
                r = actor.handle(call_item.message)
            except BaseException as e:
                exc = futures.process._ExceptionWithTraceback(e, e.__traceback__)
                self._result_queue.put(futures.process._ResultItem(
                    call_item.work_id, exception=exc))
            else:
                self._result_queue.put(futures.process._ResultItem(
                    call_item.work_id, result=r))


class _WorkItem(object):
    def __init__(self, future, message):
        self.future = future
        self.message = message


class _CallItem(object):
    def __init__(self, work_id, message):
        self.work_id = work_id
        self.message = message


def _add_call_item_to_queue(pending_work_items,
                            work_ids,
                            call_queue):
    """Fills call_queue with _WorkItems from pending_work_items.

    This function never blocks.

    Args:
        pending_work_items: A dict mapping work ids to _WorkItems e.g.
            {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        work_ids: A queue.Queue of work ids e.g. Queue([5, 6, ...]). Work ids
            are consumed and the corresponding _WorkItems from
            pending_work_items are transformed into _CallItems and put in
            call_queue.
        call_queue: A multiprocessing.Queue that will be filled with _CallItems
            derived from _WorkItems.
    """
    while True:
        if call_queue.full():
            return
        try:
            work_id = work_ids.get(block=False)
        except queue.Empty:
            return
        else:
            work_item = pending_work_items[work_id]

            if work_item.future.set_running_or_notify_cancel():
                call_queue.put(_CallItem(work_id,
                                         work_item.message),
                               block=True)
            else:
                del pending_work_items[work_id]
                continue


def _queue_management_worker(executor_reference,
                             _manager_proc,
                             pending_work_items,
                             work_ids_queue,
                             _task_queue,
                             _result_queue):
    """Manages the communication between this process and the worker processes."""
    executor = None

    def shutting_down():
        return futures.process._shutdown or executor is None or executor._shutdown_thread

    def shutdown_worker():
        # This is an upper bound
        if _manager_proc.is_alive():
            _task_queue.put_nowait(None)
        # Release the queue's resources as soon as possible.
        _task_queue.close()
        # If .join() is not called on the created processes then
        # some multiprocessing.Queue methods may deadlock on Mac OS X.
        _manager_proc.join()

    reader = _result_queue._reader

    while True:
        _add_call_item_to_queue(pending_work_items,
                                work_ids_queue,
                                _task_queue)

        sentinel = _manager_proc.sentinel
        assert sentinel
        ready = wait([reader, sentinel])
        if reader in ready:
            result_item = reader.recv()
        else:
            # Mark the process pool broken so that submits fail right now.
            executor = executor_reference()
            if executor is not None:
                executor._broken = True
                executor._shutdown_thread = True
                executor = None
            # All futures in flight must be marked failed
            for work_id, work_item in pending_work_items.items():
                work_item.future.set_exception(
                    futures.process.BrokenProcessPool(
                        "A process in the process pool was "
                        "terminated abruptly while the future was "
                        "running or pending."
                    ))
                # Delete references to object. See issue16284
                del work_item
            pending_work_items.clear()
            # Terminate remaining workers forcibly: the queues or their
            # locks may be in a dirty state and block forever.
            _manager_proc.terminate()
            shutdown_worker()
            return
        if isinstance(result_item, int):
            # Clean shutdown of a worker using its PID
            # (avoids marking the executor broken)
            assert shutting_down()
            _manager_proc.join()
            if _manager_proc is None:
                shutdown_worker()
                return
        elif result_item is not None:
            work_item = pending_work_items.pop(result_item.work_id, None)
            # work_item can be None if another process terminated (see above)
            if work_item is not None:
                if result_item.exception:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)
                # Delete references to object. See issue16284
                del work_item
        # Check whether we should start shutting down.
        executor = executor_reference()
        # No more work items can be added if:
        #   - The interpreter is shutting down OR
        #   - The executor that owns this worker has been collected OR
        #   - The executor that owns this worker has been shutdown.
        if shutting_down():
            try:
                # Since no new work items can be added, it is safe to shutdown
                # this thread if there are no pending work items.
                if not pending_work_items:
                    shutdown_worker()
                    return
            except queue.Full:
                # This is not a problem: we will eventually be woken up (in
                # _result_queue.get()) and be able to send a sentinel again.
                pass
        executor = None


class ProcessActorExecutor(_base.Executor):
    """
    Executor to manage exactly one actor
    """

    def __init__(self, _ActorClass):
        futures.process._check_system_limits()

        self._ActorClass = _ActorClass
        # todo: If we want to cancel futures we need to give the task_queue a
        # maximum size
        self._task_queue = multiprocessing.JoinableQueue()
        self._task_queue._ignore_epipe = True
        self._result_queue = multiprocessing.Queue()
        self._work_ids = queue.Queue()
        self._queue_management_thread = None

        # We only maintain one process for our actor
        self._manager_proc = None

        # Shutdown is a two-step process.
        self._shutdown_thread = False
        self._shutdown_lock = threading.Lock()
        self._broken = False
        self._queue_count = 0
        self._pending_work_items = {}

    def _start_queue_management_thread(self):
        # When the executor gets lost, the weakref callback will wake up
        # the queue management thread.

        def weakref_cb(_, q=self._result_queue):
            q.put(None)

        if self._queue_management_thread is None:
            # Start the processes so that their sentinel are known.
            self._manager_proc = ActorManager(self._ActorClass,
                                              self._task_queue,
                                              self._result_queue)
            self._manager_proc.start()

            self._queue_management_thread = threading.Thread(
                    target=_queue_management_worker,
                    args=(weakref.ref(self, weakref_cb),
                          self._manager_proc,
                          self._pending_work_items,
                          self._work_ids,
                          self._task_queue,
                          self._result_queue))
            self._queue_management_thread.daemon = True
            self._queue_management_thread.start()
            # use structures already in futures as much as possible
            futures.process._threads_queues[self._queue_management_thread] = self._result_queue

    def post(self, message):
        """
        analagous to submit.
        sends a message to an actor and returns a future
        """
        with self._shutdown_lock:
            if self._broken:
                raise futures.process.BrokenProcessPool(
                    'A child process terminated '
                    'abruptly, the process pool is not usable anymore')
            if self._shutdown_thread:
                raise RuntimeError('cannot schedule new futures after shutdown')

            f = _base.Future()
            w = _WorkItem(f, message)

            self._pending_work_items[self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            # Wake up queue management thread
            self._result_queue.put(None)

            self._start_queue_management_thread()
            return f

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            self._shutdown_thread = True
        if self._queue_management_thread:
            # Wake up queue management thread
            self._result_queue.put(None)
            if wait:
                self._queue_management_thread.join()
        # To reduce the risk of opening too many files, remove references to
        # objects that use file descriptors.
        self._queue_management_thread = None
        self._task_queue = None
        self._result_queue = None
        self._manager_proc = None


class Actor(object):
    """
    Base actor class.

    Actors receive messages, which are arbitrary objects from their managing
    executor.

    """

    @classmethod
    def executor(cls):
        return ProcessActorExecutor(cls)

    def handle(self, message):
        raise NotImplementedError('must implement message handler')


class TestActor(Actor):
    """
    An actor is given messages from its manager and performs actions in a
    single thread. Its state is private and threadsafe.
    """
    def __init__(actor):
        actor.state = {}

    def handle(actor, message):
        if not isinstance(message, dict):
            raise ValueError('Commands must be passed in a message dict')
        message = message.copy()
        action = message.pop('action', None)
        if action is None:
            raise ValueError('message must have an action item')
        if action == 'hello world':
            content = 'hello world'
            return content
        elif action == 'wait':
            import time
            time.wait(message.get('time', 0))
            return 'finished'
        elif action == 'debug':
            return actor
        elif action == 'start':
            actor.state['a'] = 3
            return 'started'
        elif action == 'add':
            for i in range(10000000):
                actor.state['a'] += 1
            return 'added', actor.state['a']
        else:
            raise ValueError('Unknown action=%r' % (action,))


def test():
    # from actor2 import *
    # from actor2 import _add_call_item_to_queue, _queue_management_worker

    def done_callback(result):
        print('result = %r' % (result,))
        print('DOING DONE CALLBACK')

    print('Starting Test')
    executor = TestActor.executor()
    print('About to send messages')

    f1 = executor.post({'action': 'hello world'})
    print(f1.result())

    f2 = executor.post({'action': 'start'})
    print(f2.result())

    f3 = executor.post({'action': 'add'})
    print(f3.result())

    f4 = executor.post({'action': 'wait', 'time': 0})
    f4.add_done_callback(done_callback)

    print(f4.result())

    print('Test completed')

# if __name__ == '__main__':
#     test()
