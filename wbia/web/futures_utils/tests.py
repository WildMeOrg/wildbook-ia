# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.web import futures_utils
from concurrent import futures
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


class TestActorMixin(object):
    """
    An actor is given messages from its manager and performs actions in a
    single thread. Its state is private and threadsafe.

    The handle method must be implemented by the user.
    """

    def __init__(actor, a=None, factor=1):
        print('init mixin with args')
        print('a = %r' % (a,))
        actor.state = {}
        if a is not None:
            actor.state['a'] = a * factor

    def handle(actor, message):
        print('handling message = {}'.format(message))
        if not isinstance(message, dict):
            raise ValueError('Commands must be passed in a message dict')
        message = message.copy()
        action = message.pop('action', None)
        if action is None:
            raise ValueError('message must have an action item')
        if action == 'hello world':
            content = 'hello world'
            return content
        elif action == 'debug':
            return actor
        elif action == 'wait':
            import time

            num = message.get('time', 0)
            time.sleep(num)
            return num
        elif action == 'prime':
            import ubelt as ub

            a = actor.state['a']
            n = message['n']
            return n, a, ub.find_nth_prime(n + a)
        elif action == 'start':
            actor.state['a'] = 3
            return 'started'
        elif action == 'add':
            for i in range(10000000):
                actor.state['a'] += 1
            return 'added', actor.state['a']
        else:
            raise ValueError('Unknown action=%r' % (action,))


class TestProcessActor(TestActorMixin, futures_utils.ProcessActor):
    pass


class TestThreadActor(TestActorMixin, futures_utils.ThreadActor):
    pass


def test_simple(ActorClass):
    # from actor2 import *
    # from actor2 import _add_call_item_to_queue, _queue_management_worker

    print('-----------------')
    print('Simple test of {}'.format(ActorClass))

    test_state = {'done': False}

    def done_callback(result):
        test_state['done'] = True
        print('result = %r' % (result,))
        print('DOING DONE CALLBACK')

    print('Starting Test')
    executor = ActorClass.executor()
    print('About to send messages')

    f1 = executor.post({'action': 'hello world'})
    print(f1.result())

    f2 = executor.post({'action': 'start'})
    print(f2.result())

    f3 = executor.post({'action': 'add'})
    print(f3.result())

    print('Test completed')
    print('L______________')


def test_callbacks(ActorClass):
    print('-----------------')
    print('Test callbacks for {}'.format(ActorClass))

    test_state = {'num': False}

    def done_callback(f):
        num = f.result()
        test_state['num'] += num
        print('DONE CALLBACK GOT = {}'.format(num))

    executor = ActorClass.executor()
    f1 = executor.post({'action': 'wait', 'time': 1})
    f1.add_done_callback(done_callback)

    f2 = executor.post({'action': 'wait', 'time': 2})
    f2.add_done_callback(done_callback)

    f3 = executor.post({'action': 'wait', 'time': 3})
    f3.add_done_callback(done_callback)

    # Should reach this immediately before any task is done
    assert test_state['num'] == 0, 'should not have finished any task yet'

    # Wait for the second result
    print(f2.result())
    assert test_state['num'] == 3, 'should have finished task 1 and 2'

    # Wait for the third result
    print(f3.result())
    assert test_state['num'] == 6

    print('Test completed')
    print('L______________')


def test_cancel(ActorClass):
    print('-----------------')
    print('Test cancel for {}'.format(ActorClass))

    test_state = {'num': False}

    def done_callback(f):
        try:
            num = f.result()
        except futures.CancelledError:
            num = 'canceled'
            print('Canceled task {}'.format(f))
        else:
            test_state['num'] += num
            print('DONE CALLBACK GOT = {}'.format(num))

    executor = ActorClass.executor()
    f1 = executor.post({'action': 'wait', 'time': 1})
    f1.add_done_callback(done_callback)

    f2 = executor.post({'action': 'wait', 'time': 2})
    f2.add_done_callback(done_callback)

    f3 = executor.post({'action': 'wait', 'time': 3})
    f3.add_done_callback(done_callback)

    f4 = executor.post({'action': 'wait', 'time': 4})
    f4.add_done_callback(done_callback)

    can_cancel = f3.cancel()
    # print('can_cancel = %r' % (can_cancel,))
    assert can_cancel, 'we should be able to cancel in time'

    f4.result()
    assert test_state['num'] == 7, 'f3 was not cancelled'

    print('Test completed')
    print('L______________')


def test_actor_args(ActorClass):
    ex1 = ActorClass.executor(8, factor=8)
    f1 = ex1.post({'action': 'add'})
    assert f1.result()[1] == 10000064


def test_multiple(ActorClass):
    print('-----------------')
    print('Test multiple for {}'.format(ActorClass))
    # Make multiple actors and send them each multiple jobs
    n_actors = 5
    n_jobs = 10
    actors_exs = [ActorClass.executor(a) for a in range(1, n_actors)]
    fs = []
    for jobid in range(n_jobs):
        n = jobid + 500
        fs += [ex.post({'action': 'prime', 'n': n}) for ex in actors_exs]

    for f in futures.as_completed(fs):
        print('n, a, prime = {}'.format(f.result()))

    actors = [ex.post({'action': 'debug'}).result() for ex in actors_exs]
    for a in actors:
        print(a.state)
    print('Test completed')
    print('L______________')


def main():
    """
    from wbia.web.futures_utils.tests import *
    ActorClass = TestProcessActor
    ActorClass = TestThreadActor

    """
    classes = [
        TestProcessActor,
        TestThreadActor,
    ]
    for ActorClass in classes:
        test_multiple(ActorClass)
        test_actor_args(ActorClass)
        test_simple(ActorClass)
        test_callbacks(ActorClass)
        test_cancel(ActorClass)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.web.futures_utils.tests
    """
    main()
