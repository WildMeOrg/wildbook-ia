# -*- coding: utf-8 -*-
"""
Accepts and handles requests for tasks.

Each of the following runs in its own Thread/Process.

Accepter:
    Receives tasks and requests
    Delegates tasks and responds to requests
    Tasks are delgated to an engine

Engine:
    the engine accepts requests.
    the engine immediately responds WHERE it will be ready.
    the engine sends a message to the collector saying that something will be ready.
    the engine then executes a task.
    The engine is given direct access to the data.

Collector:
    The collector accepts requests
    The collector can respond:
        * <ResultContent>
        * Results are ready.
        * Results are not ready.
        * Unknown jobid.
        * Error computing results.
        * Progress percent.

References:
    Simple task farm, with routed replies in pyzmq
    For http://stackoverflow.com/questions/7809200/implementing-task-farm-messaging-pattern-with-zeromq
"""
from __future__ import absolute_import, division, print_function, unicode_literals
#from __future__ import absolute_import, division, print_function
#import os
import six
import utool as ut
import time
import random
import zmq
#import threading  # NOQA


# BASICALLY DO A CLIENT/SERVER TO SPAWN PROCESSES
# AND THEN A PUBLISH SUBSCRIBE TO RETURN DATA


ctx = zmq.Context.instance()
client_iface = 'tcp://127.0.0.1:5555'
engine_iface = 'tcp://127.0.0.1:5556'

collect_pull_iface = 'tcp://127.0.0.1:5557'


def result_collector():
    collector = ctx.socket(zmq.PULL)
    collector.bind(collect_pull_iface)
    collector.setsockopt_string(zmq.IDENTITY, 'Controller.COLLECTOR')
    collecter_data = {}
    COLLECT_MODE = 'timeout'
    if COLLECT_MODE == 'poller':
        poller = zmq.Poller()
        poller.register(collector, zmq.POLLIN)
        while True:
            print('polling')
            evnts = poller.poll(10)
            evnts = dict(poller.poll())
            if collector in evnts:
                #print('Collecting...')
                reply_result = collector.recv_json()
                print('COLLECT Controller.COLLECTOR: Collected: %r' % (reply_result,))
                result_id = reply_result['result_id']
                collecter_data[result_id] = reply_result
            else:
                #print('waiting')
                pass
    elif COLLECT_MODE == 'poller':
        collector.RCVTIMEO = 1000
        while True:
            print('Collecting...')
            try:
                reply_result = collector.recv_json()
                print('COLLECT Controller.COLLECTOR: Collected: %r' % (reply_result,))
                result_id = reply_result['result_id']
                collecter_data[result_id] = reply_result
            except Exception:
                # TODO: add in a responce if a job status request is given
                #print('loop')
                pass

        #result = collector.recv_json()
        #if collecter_data.has_key(result['consumer']):  # NOQA
        #    collecter_data[result['consumer']] = collecter_data[result['consumer']] + 1
        #else:
        #    collecter_data[result['consumer']] = 1
        #if x == 999:
        #    print(collecter_data)


def scheduler():
    r"""
    IBEIS:
        THis will belong to a thread on the webserver main process.

    ROUTER-DEALER queue device, for load-balancing requests from clients
    across engines, and routing replies to the originating client."""
    # ----
    router = ctx.socket(zmq.ROUTER)
    router.bind(client_iface)
    # ----
    dealer = ctx.socket(zmq.DEALER)
    # this is optional, it just makes identities more obvious when they appear
    dealer.setsockopt_string(zmq.IDENTITY, 'Controller.DEALER')
    dealer.bind(engine_iface)
    # the remainder of this function can be entirely replaced with
    if True:
        zmq.device(zmq.QUEUE, router, dealer)
    else:
        # but this shows what is really going on:
        poller = zmq.Poller()
        poller.register(router, zmq.POLLIN)
        poller.register(dealer, zmq.POLLIN)
        while True:
            evts = dict(poller.poll())
            # poll() returns a list of tuples [(socket, evt), (socket, evt)]
            # dict(poll()) turns this into {socket:evt, socket:evt}
            if router in evts:
                msg = router.recv_multipart()
                # ROUTER sockets prepend the identity of the sender, for routing replies
                #client = msg[0]   # NOQA
                print('Controller.ROUTER received %s, relaying via DEALER' % msg)
                #dealer.send_string(msg)
                dealer.send_json({'msg': msg})
            if dealer in evts:
                msg = dealer.recv_multipart()
                #client = msg[0]  # NOQA
                print('Controller.DEALER received %s, relaying via ROUTER' % msg)
                #router.send_string(msg)
                router.send_json({'msg': msg})


def process_request(msg):
    """process the message (reverse letters)"""
    #return ['foobar']
    import time
    time.sleep(.1)
    return [ part[::-1] for part in msg ]


def engine(id_):
    r"""
    IBEIS:
        This will be part of a worker process with its own IBEISController
        instance.

        Needs to send where the results will go and then publish the results there.


    The engine - receives messages, performs some action, and sends a reply,
    preserving the leading two message parts as routing identities
    """
    engine_sock = ctx.socket(zmq.ROUTER)
    engine_sock.connect(engine_iface)

    push_sock = ctx.socket(zmq.PUSH)
    push_sock.connect(collect_pull_iface)
    while True:
        msg = engine_sock.recv_multipart()
        print('engine %s recvd message:' % id_, msg)
        # note that the first two parts will be ['Controller.ROUTER', 'Client.<id_>']
        # these are needed for the reply to propagate up to the right client
        idents, request = msg[:2], msg[2:]
        result_id = 'result_%s' % (id_,)
        future_msg = 'Job started. Check for status at ' + six.text_type(result_id)
        reply_future = idents + [future_msg, result_id]
        print('engine %s sending reply:' % id_, reply_future)
        engine_sock.send_json({'msg': reply_future})
        print('engine %s is working...' % (id_,))
        reply_result = dict(idents=idents, result=process_request(request), result_id=result_id)
        #print('ENGINE %s computed results, but has nowhere to send it!' % (id_,))
        print('ENGINE %s computed results, pushing it to collector' % (id_,))
        push_sock.send_json(reply_result)
        print('...pushed')


class BackgroundJobQueue(object):
    def __init__(self, id_):
        self.id_ = id_
        self.initialize_background_processes()
        self.initialize_main_thread()

    def initialize_background_processes(self):
        print('Initialize Background Processes')
        #spawner = ut.spawn_background_process
        spawner = ut.spawn_background_daemon_thread
        self.st = spawner(scheduler)
        self.ct = spawner(result_collector)
        self.engines = [spawner(engine, i) for i in range(2)]

    def initialize_main_thread(self):
        client_socket = ctx.socket(zmq.DEALER)
        client_socket.setsockopt_string(zmq.IDENTITY, 'Client.%s' % (self.id_,))
        client_socket.connect(client_iface)
        self.client_socket = client_socket

    def queue_job(self):
        r"""
        IBEIS:
            This is just a function that lives in the main thread and ships off
            a job.

        The client - sends messages, and receives replies after they
        have been processed by the

        """
        msg = ['hello', 'world', six.text_type(random.randint(10, 100))]
        print('\n+-------')
        print('client %s sending : %s' % (self.id_, msg))
        print('msg = %r' % (msg,))
        #self.client_socket.send_string(msg)
        #self.client_socket.send_multipart(msg)
        self.client_socket.send_json({'msg': msg})
        msg = self.client_socket.recv_json()
        print('client %s received: %s' % (self.id_, msg))
        print('\nL______')

    def get_job_status(self, job_id):
        pass


def _on_ctrl_c(signal, frame):
    print('[ibeis.main_module] Caught ctrl+c')
    print('[ibeis.main_module] sys.exit(0)')
    import sys
    sys.exit(0)


def _init_signals():
    import signal
    signal.signal(signal.SIGINT, _on_ctrl_c)


def test_zmq_task():
    """
    CommandLine:
        python -m ibeis.web.zmq_task_queue --exec-test_zmq_task

    Example:
        >>> # SCRIPT
        >>> from ibeis.web.zmq_task_queue import *  # NOQA
        >>> test_zmq_task()
    """
    import utool as ut  # NOQA
    _init_signals()
    # now start a few clients, and fire off some requests
    #clients = []
    time.sleep(1)

    print('Initializing Main Thread')
    sender = BackgroundJobQueue(1)
    print('... waiting for jobs')
    if ut.get_argflag('--cmd'):
        ut.embed()
        sender.queue_job()
    else:
        time.sleep(.5)
        sender.queue_job()
        time.sleep(1.5)
        sender.queue_job()
        time.sleep(1.5)
        time.sleep(1.5)

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.zmq_task_queue
        python -m ibeis.web.zmq_task_queue --allexamples
        python -m ibeis.web.zmq_task_queue --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    if True:
        test_zmq_task()
    else:
        import utool as ut  # NOQA
        ut.doctest_funcs()
