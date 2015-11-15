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


#python -m ibeis --tf test_zmq_task
python -m ibeis.web.zmq_task_queue --main

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['UTOOL_NOCNN'] = 'True'
#from __future__ import absolute_import, division, print_function
#import os
import six
import utool as ut
import time  # NOQA
import random  # NOQA
import zmq
import uuid
print, rrr, profile = ut.inject2(__name__, '[zmqstuff]')
#import threading  # NOQA


# BASICALLY DO A CLIENT/SERVER TO SPAWN PROCESSES
# AND THEN A PUBLISH SUBSCRIBE TO RETURN DATA


ctx = zmq.Context.instance()

import itertools
import functools
#portgen = itertools.count(5555)
portgen = itertools.count(5335)
portgen_ = functools.partial(six.next, portgen)

url = 'tcp://127.0.0.1'

client_iface = url + ':' + six.text_type(portgen_())
engine_iface = url + ':' + six.text_type(portgen_())
collect_pull_iface = url + ':' + six.text_type(portgen_())
pair_iface = url + ':' + six.text_type(portgen_())


__SPAWNER__ = ut.spawn_background_process
#__SPAWNER__ = ut.spawn_background_daemon_thread


def dbgwait():
    pass
    #time.sleep(.03)


class JobQueueClient(object):
    def __init__(self, id_):
        self.id_ = id_
        self.num_engines = 1

    def __del__(self):
        print('Cleaning up job client')
        for i in self.engine_proces:
            i.terminate()
        self.scheduler_proc.terminate()
        self.collector_proc.terminate()
        #for i in self.engine_proces:
        #    i.join()
        #self.scheduler_proc.join()
        #self.collector_proc.join()
        print('Killed external procs')

    def init(self):
        self.initialize_background_processes()
        self.initialize_client_thread()

    def initialize_background_processes(self):
        print('Initialize Background Processes')
        self.scheduler_proc = __SPAWNER__(schedule_loop)
        #__SPAWNER__(new_schedule_queue)
        self.collector_proc = __SPAWNER__(collector_loop)
        self.engine_proces = [__SPAWNER__(engine_loop, i)
                              for i in range(self.num_engines)]

    def initialize_client_thread(self):
        self.client_socket = ctx.socket(zmq.DEALER)
        self.client_socket.setsockopt_string(zmq.IDENTITY, 'Client.%s' % (self.id_,))
        self.client_socket.connect(client_iface)

        self.pair_socket = ctx.socket(zmq.PAIR)
        self.pair_socket.connect(pair_iface)
        self.pair_socket.setsockopt_string(zmq.IDENTITY, 'PAIR.%s' % (self.id_,))

    def queue_job(self):
        r"""
        IBEIS:
            This is just a function that lives in the main thread and ships off
            a job.

        The client - sends messages, and receives replies after they
        have been processed by the

        """
        #msg = ['hello', 'world', six.text_type(random.randint(10, 100))]
        msg = {'action': 'helloworld', 'args': [], 'kwargs': {}}
        print('[client] %s sending request : %s' % (self.id_, msg))
        self.client_socket.send_json(msg)
        resp = self.client_socket.recv_json()
        print('[client] %s received: %s' % (self.id_, resp))
        jobid = resp['jobid']
        return jobid

    def get_job_status(self, jobid):
        print('[client] Requesting job status of jobid=%r' % (jobid,))
        pair_msg = dict(action='job_status', jobid=jobid)
        self.pair_socket.send_json(pair_msg)
        print('[client] SENT JOB STATUS REQUEST')
        dbgwait()
        print('[client] ... waiting for pair reply')
        reply = self.pair_socket.recv_json()
        print('[client] got reply = %r' % (reply,))


def new_schedule_queue(input_iface, output_iface):
    with ut.Indenter('[schedule2]'):
        print('Initializing scheduler')
        router = ctx.socket(zmq.ROUTER)
        router.bind(input_iface)
        dealer = ctx.socket(zmq.DEALER)
        dealer.setsockopt_string(zmq.IDENTITY, 'Controller.DEALER')
        dealer.bind(output_iface)
        # main exec loop
        zmq.device(zmq.QUEUE, router, dealer)
        print('Exiting scheduler')


def schedule_loop():
    r"""
    IBEIS:
        THis will belong to a thread on the webserver main process.

    ROUTER-DEALER queue device, for load-balancing requests from clients
    across engine_sockets, and routing replies to the originating client.
    """
    print('[scheduler] Initializing scheduler')
    # setsockopt_string optional, it just makes identities more obvious when they appear
    # Client is router
    router = ctx.socket(zmq.ROUTER)
    router.bind(client_iface)
    # Engine is dealer
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt_string(zmq.IDENTITY, 'Controller.DEALER')
    dealer.bind(engine_iface)
    if True:
        # the remainder of this function can be entirely replaced with
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
                print('[scheduler] ROUTER(client) relayed %s via DEALER(engine)' % msg)
                dealer.send_multipart(msg)
            if dealer in evts:
                msg = dealer.recv_multipart()
                print('[scheduler] DEALER(engine) relayed %s via ROUTER(client)' % msg)
                dealer.send_multipart(msg)
    # ----
    print('[scheduler] Exiting scheduler')


def engine_loop(id_):
    r"""
    IBEIS:
        This will be part of a worker process with its own IBEISController
        instance.

        Needs to send where the results will go and then publish the results there.


    The engine_loop - receives messages, performs some action, and sends a reply,
    preserving the leading two message parts as routing identities
    """
    import ibeis
    with ut.Indenter('[engine %d] ' % (id_)):
        print('Initializing engine')
        ibs = None
        ibs = ut.DynStruct()
        if True:
            dbname = 'testdb1'
            ibs = ibeis.opendb(dbname)

        engine_sock = ctx.socket(zmq.ROUTER)
        engine_sock.connect(engine_iface)

        push_sock = ctx.socket(zmq.PUSH)
        push_sock.connect(collect_pull_iface)
        while True:
            msg = engine_sock.recv_multipart()
            print('recvd message: %s' % (msg,))

            # note that the first two parts will be ['Controller.ROUTER', 'Client.<id_>']
            # these are needed for the reply to propagate up to the right client
            idents, request_json = msg[:2], msg[2]
            request = ut.from_json(request_json)

            action = request['action']
            #jobid = 'result_%s' % (id_,)
            jobid = 'result_%s' % (uuid.uuid4(),)

            reply_json = ut.to_json({
                'jobid': jobid,
                'status': 'ok',
                'text': 'job accepted',
            }).encode('utf-8')
            reply = idents + [reply_json]
            print('sending reply:')
            engine_sock.send_multipart(reply)
            dbgwait()
            print('is working...')
            time.sleep(5)
            args = request['args']
            kwargs = request['kwargs']
            if action == 'helloworld':
                result = 'hola' + ut.repr2((args, kwargs))
                #result = process_request(request),
            else:
                if False:
                    dbname = None
                    ibs = ibeis.opendb(dbname)
                elif True:
                    ibs.compute = ut.identity
                    result = ibs.compute(None)
                    #ibs.compute()
                    print('finished computing, pushing results to collector')
            reply_result = dict(
                idents=idents,
                result=result,
                jobid=jobid
            )
            dbgwait()
            print('...pushing')
            dbgwait()
            push_sock.send_json(reply_result)
        # ----
        print('Exiting scheduler')


def collector_loop():
    """
    Service that stores completed algorithm results
    """
    collector_socket = ctx.socket(zmq.PULL)
    collector_socket.bind(collect_pull_iface)
    collector_socket.setsockopt_string(zmq.IDENTITY, 'Controller.COLLECTOR')

    pair_socket = ctx.socket(zmq.PAIR)
    pair_socket.bind(pair_iface)

    collecter_data = {}

    def handle_collect():
        reply_result = collector_socket.recv_json()
        print('[collect] Collected: %r' % (reply_result,))
        jobid = reply_result['jobid']
        collecter_data[jobid] = reply_result
        print('[collect] stored result')

    def handle_request():
        pair_msg = pair_socket.recv_json()
        reply = {}
        print('[collect] Received Request: %r' % (pair_msg,))
        action = pair_msg['action']
        print('[collect] action = %r' % (action,))
        if action == 'job_status':
            print('[collect] ...building action=%r response' % (action,))
            jobid = pair_msg['jobid']
            if jobid in collecter_data:
                reply['result'] = collecter_data[jobid]['result']
                reply['jobstatus'] = 'completed'
            else:
                reply['jobstatus'] = 'unknown'
            reply['jobid'] = jobid
            print('[collect] Replying action=%r request' % (action,))
            pair_socket.send_json(reply)
        elif action == 'result':
            print('[collect] Building action=%r response' % (action,))
            jobid = pair_msg['jobid']
            print('[collect] Received Request: %r' % (pair_msg,))
            reply_result = collecter_data[jobid]
            result = reply_result['result']
            reply = {'jobid': jobid, 'result': result, 'status': 'ok'}
            print('[collect] Responding to pair_msg')
            pair_socket.send_json(reply)
        else:
            print('[collect] ...error unknown action')
            pair_socket.send_json({'status': 'error', 'text': 'unknown action'})

    #COLLECT_MODE = 'poller'
    COLLECT_MODE = 'timeout'
    if COLLECT_MODE is None:
        assert False
    elif COLLECT_MODE == 'timeout':
        collector_socket.RCVTIMEO = 100
        pair_socket.RCVTIMEO = 100
        # Timeout event loop
        while True:
            #print('Collecting...')
            try:
                # FIXME: This should be accessed from a database
                # (at least an inmemory database)
                # rather than from this instate python dictionary
                handle_collect()
            except zmq.error.Again:
                pass
            try:
                handle_request()
            except zmq.error.Again:
                pass
    elif COLLECT_MODE == 'poller':
        assert False
        poller = zmq.Poller()
        poller.register(collector_socket, zmq.POLLIN)
        poller.register(pair_socket, zmq.POLLIN)
        # Polling event loop
        while True:
            print('polling')
            evnts = poller.poll(10)
            evnts = dict(poller.poll())
            if collector_socket in evnts:
                handle_collect()
            if pair_socket in evnts:
                handle_request()


def process_request(msg):
    """process the message (reverse letters)"""
    #return ['foobar']
    dbgwait()
    return [ part[::-1] for part in msg ]


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
        python -b -m ibeis.web.zmq_task_queue --exec-test_zmq_task

    Example:
        >>> # SCRIPT
        >>> from ibeis.web.zmq_task_queue import *  # NOQA
        >>> test_zmq_task()
    """
    import utool as ut  # NOQA
    _init_signals()
    # now start a few clients, and fire off some requests
    #clients = []
    dbgwait()

    print('Initializing JobQueueClient')
    sender = JobQueueClient(1)
    sender.init()
    dbgwait()
    print('... waiting for jobs')
    if ut.get_argflag('--cmd'):
        ut.embed()
        sender.queue_job()
    else:
        print('[test] ... emit test1')
        jobid = sender.queue_job()
        dbgwait()
        dbgwait()
        dbgwait()
        dbgwait()
        dbgwait()
        sender.get_job_status(jobid)
        dbgwait()
        #sender.queue_job()
        dbgwait()
    print('FINISHED TEST SCRIPT')

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.zmq_task_queue
        python -m ibeis.web.zmq_task_queue --allexamples
        python -m ibeis.web.zmq_task_queue --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    if ut.get_argflag('--main'):
        test_zmq_task()
    else:
        import utool as ut  # NOQA
        ut.doctest_funcs()
