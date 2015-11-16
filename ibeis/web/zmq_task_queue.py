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
    http://stackoverflow.com/questions/7809200/implementing-task-farm-messaging-pattern-with-zeromq
    https://gist.github.com/minrk/1358832


#python -m ibeis --tf test_zmq_task
python -m ibeis.web.zmq_task_queue --main
python -m ibeis.web.zmq_task_queue --main --bg
python -m ibeis.web.zmq_task_queue --main --fg

"""
from __future__ import absolute_import, division, print_function, unicode_literals
if True:
    import os
    os.environ['UTOOL_NOCNN'] = 'True'
if True:
    import six
    import utool as ut
    import time  # NOQA
    import random  # NOQA
    import zmq
    import uuid  # NOQA
    import itertools
    import functools
    from functools import partial
print, rrr, profile = ut.inject2(__name__, '[zmqstuff]')
#import threading  # NOQA


# BASICALLY DO A CLIENT/SERVER TO SPAWN PROCESSES
# AND THEN A PUBLISH SUBSCRIBE TO RETURN DATA

ctx = zmq.Context.instance()

#portgen = itertools.count(5555)
portgen = itertools.count(5335)
portgen_ = functools.partial(six.next, portgen)

url = 'tcp://127.0.0.1'

engine_iface1 = url + ':' + six.text_type(portgen_())
engine_iface2 = url + ':' + six.text_type(portgen_())
collect_iface1 = url + ':' + six.text_type(portgen_())
collect_iface2 = url + ':' + six.text_type(portgen_())
collect_pushpull_iface = url + ':' + six.text_type(portgen_())


__SPAWNER__ = ut.spawn_background_process
#__SPAWNER__ = ut.spawn_background_daemon_thread


def dbgwait(t=.03):
    #time.sleep(t)
    pass


def test_zmq_task():
    """
    CommandLine:
        python -m ibeis.web.zmq_task_queue --exec-test_zmq_task
        python -b -m ibeis.web.zmq_task_queue --exec-test_zmq_task

        python -m ibeis.web.zmq_task_queue --main
        python -m ibeis.web.zmq_task_queue --main --bg
        python -m ibeis.web.zmq_task_queue --main --fg

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

    import numpy as np
    client_id = np.random.randint(1000)
    sender = ClientProc(client_id)
    reciever = BackgroundProcs()
    if ut.get_argflag('--bg'):
        reciever.initialize_background_processes()
        print('parent process is looping forever')
        while True:
            time.sleep(1)
    elif ut.get_argflag('--fg'):
        sender.initialize_client_thread()
    else:
        reciever.initialize_background_processes()
        sender.initialize_client_thread()

    # Foreground test script
    dbgwait()
    print('... waiting for jobs')
    if ut.get_argflag('--cmd'):
        ut.embed()
        sender.queue_job()
    else:
        print('[test] ... emit test1')
        #jobid1 = sender.queue_job('helloworld')
        #dbgwait()
        #sender.get_job_status(jobid1)
        #dbgwait()
        jobid2 = sender.queue_job('helloworld', 10)
        jobid4 = sender.queue_job('helloworld', 10)
        jobid5 = sender.queue_job('helloworld', 10)
        #jobid2 = 'foobar'

        while True:
            reply = sender.get_job_status(jobid2)
            time.sleep(1)
            if reply['jobstatus'] == 'completed':
                reply = sender.get_job_result(jobid2)
                result = reply['result']
                print('Final result = %r' % (result,))
                break
        #sender.queue_job()
        dbgwait()
    print('FINISHED TEST SCRIPT')


class BackgroundProcs(object):
    def __init__(self):
        self.num_engines = 1
        self.engine_queue_proc = None
        self.collect_queue_proc = None
        self.engine_procs = None
        self.collect_proc = None

    def __del__(self):
        print('Cleaning up job client')
        if self.engine_procs is not None:
            for i in self.engine_proces:
                i.terminate()
        if self.engine_queue_proc is not None:
            self.engine_queue_proc.terminate()
        if self.collect_proc is not None:
            self.collect_proc.terminate()
        if self.collect_queue_proc is not None:
            self.collect_queue_proc.terminate()
        print('Killed external procs')

    def initialize_background_processes(self):
        print = partial(ut.colorprint, color='fuchsia')
        print('Initialize Background Processes')
        only_engine = ut.get_argflag('--only-engine')
        spawn_collector = not only_engine
        spawn_engine = not ut.get_argflag('--no-engine')
        spawn_queue = not only_engine

        if spawn_queue:
            self.engine_queue_proc = __SPAWNER__(engine_queue_loop)
            self.collect_queue_proc = __SPAWNER__(collect_queue_loop)
        if spawn_collector:
            self.collect_proc = __SPAWNER__(collector_loop)
        if spawn_engine:
            self.engine_procs = [__SPAWNER__(engine_loop, i)
                                  for i in range(self.num_engines)]


class ClientProc(object):
    def __init__(self, id_):
        self.id_ = id_

    def init(self):
        self.initialize_background_processes()
        self.initialize_client_thread()

    def initialize_client_thread(self):
        print = partial(ut.colorprint, color='blue')
        print('Initializing ClientProc')
        self.engine_deal_sock = ctx.socket(zmq.DEALER)
        self.engine_deal_sock.setsockopt_string(zmq.IDENTITY, 'Client-%s.EngineDeal' % (self.id_,))
        self.engine_deal_sock.connect(engine_iface1)
        print('connect engine_iface1 = %r' % (engine_iface1,))

        self.collect_deal_sock = ctx.socket(zmq.DEALER)
        self.collect_deal_sock.setsockopt_string(zmq.IDENTITY, 'Client-%s.CollectDeal' % (self.id_,))
        self.collect_deal_sock.connect(collect_iface1)
        print('connect collect_iface1 = %r' % (collect_iface1,))

    def queue_job(self, action, *args, **kwargs):
        r"""
        IBEIS:
            This is just a function that lives in the main thread and ships off
            a job.

        The client - sends messages, and receives replies after they
        have been processed by the
        """
        with ut.Indenter('[client %d] ' % (self.id_)):
            print = partial(ut.colorprint, color='blue')
            print('----')
            msg = {'action': action, 'args': args, 'kwargs': kwargs}
            print('Sending job: %s' % (msg))
            self.engine_deal_sock.send_json(msg)
            print('..sent, waiting for response')
            resp = self.engine_deal_sock.recv_json()
            print('Received engine response: %s' % ( resp))
            jobid = resp['jobid']
            return jobid

    def get_job_status(self, jobid):
        with ut.Indenter('[client %d] ' % (self.id_)):
            print = partial(ut.colorprint, color='teal')
            print('----')
            print('Get status of jobid=%r' % (jobid,))
            pair_msg = dict(action='job_status', jobid=jobid)
            self.collect_deal_sock.send_json(pair_msg)
            print('... waiting for collector reply')
            reply = self.collect_deal_sock.recv_json()
            print('got reply = %r' % (reply,))
        return reply

    def get_job_result(self, jobid):
        with ut.Indenter('[client %d] ' % (self.id_)):
            print = partial(ut.colorprint, color='teal')
            print('----')
            print('Get result of jobid=%r' % (jobid,))
            pair_msg = dict(action='job_result', jobid=jobid)
            self.collect_deal_sock.send_json(pair_msg)
            print('... waiting for collector reply')
            reply = self.collect_deal_sock.recv_json()
            print('got reply = %r' % (reply,))
        return reply


def new_queue_loop(iface1, iface2, name=None):
    """
    Args:
        iface1 (str): address for the client that deals
        iface2 (str): address for the server that routes
        name (None): (default = None)
    """

    assert name is not None, 'must name queue'
    queue_name = name + '_queue'
    loop_name = queue_name + '_loop'
    def queue_loop():
        print = partial(ut.colorprint, color='green')
        with ut.Indenter('[%s] ' % (name,)):
            print('Init new_queue_loop: name=%r' % (name,))
            # bind the client dealer to the queue router
            rout_sock = ctx.socket(zmq.ROUTER)
            rout_sock.setsockopt_string(zmq.IDENTITY, name + '.' + 'ROUTER')
            rout_sock.bind(iface1)
            print('bind %s_iface2 = %r' % (name, iface1,))
            # bind the server router to the queue dealer
            deal_sock = ctx.socket(zmq.DEALER)
            deal_sock.setsockopt_string(zmq.IDENTITY, name + '.' + 'DEALER')
            deal_sock.bind(iface2)
            print('bind %s_iface2 = %r' % (name, iface2,))
            #if False:
            if 0:
                # the remainder of this function can be entirely replaced with
                zmq.device(zmq.QUEUE, rout_sock, deal_sock)
            else:
                # but this shows what is really going on:
                poller = zmq.Poller()
                poller.register(rout_sock, zmq.POLLIN)
                poller.register(deal_sock, zmq.POLLIN)
                while True:
                    evts = dict(poller.poll())
                    # poll() returns a list of tuples [(socket, evt), (socket, evt)]
                    # dict(poll()) turns this into {socket:evt, socket:evt}
                    if rout_sock in evts:
                        msg = rout_sock.recv_multipart()
                        # ROUTER sockets prepend the identity of the sender,
                        # for routing replies
                        print('ROUTER relayed %r via DEALER' % (msg,))
                        deal_sock.send_multipart(msg)
                    if deal_sock in evts:
                        msg = deal_sock.recv_multipart()
                        print('DEALER relayed %r via ROUTER' % (msg,))
                        rout_sock.send_multipart(msg)
            print('Exiting %s' % (loop_name,))
    ut.set_funcname(queue_loop, loop_name)
    return queue_loop

engine_queue_loop = new_queue_loop(engine_iface1, engine_iface2, name='engine')
collect_queue_loop = new_queue_loop(collect_iface1, collect_iface2, name='collect')


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
    #base_print = print  # NOQA
    print = partial(ut.colorprint, color='red')
    with ut.Indenter('[engine %d] ' % (id_)):
        print('Initializing engine')
        print('connect engine_iface2 = %r' % (engine_iface2,))
        print('connect collect_pushpull_iface = %r' % (collect_pushpull_iface,))
        ibs = None
        ibs = ut.DynStruct()
        if True:
            dbname = 'testdb1'
            ibs = ibeis.opendb(dbname)

        engine_rout_sock = ctx.socket(zmq.ROUTER)
        engine_rout_sock.connect(engine_iface2)

        collect_push_sock = ctx.socket(zmq.PUSH)
        collect_push_sock.connect(collect_pushpull_iface)
        job_counter = 0
        while True:
            idents, request = rcv_multipart_json(engine_rout_sock)
            action = request['action']

            #jobid = 'result_%s' % (id_,)
            #jobid = 'result_%s' % (uuid.uuid4(),)
            job_counter += 1
            jobid = 'jobid_%s-%04d' % (id_, job_counter,)
            print('Creating jobid %r' % (jobid,))

            # Reply immediately with a new jobid
            reply_notify = {
                'jobid': jobid,
                'status': 'ok',
                'text': 'job accepted',
                'action': 'notification'
            }
            print('...notifying collector about new job')
            collect_push_sock.send_json(reply_notify)
            print('... notifying client that job was accepted')
            send_multipart_json(engine_rout_sock, idents, reply_notify)
            dbgwait()

            # Start working
            # (maybe this should be offloaded to another process?)
            print('is working...')
            time.sleep(.1)
            args = request['args']
            kwargs = request['kwargs']

            # Map actions to IBEISController calls here
            if action == 'helloworld':
                def helloworld(time_=0, *args, **kwargs):
                    time.sleep(time_)
                    retval = ('HELLO time_=%r ' % (time_,)) + ut.repr2((args, kwargs))
                    return retval
                action_func = helloworld
                result = action_func(*args, **kwargs)
            else:
                if False:
                    dbname = None
                    ibs = ibeis.opendb(dbname)
                elif True:
                    ibs.compute = ut.identity
                    result = ibs.compute(None)
                    #ibs.compute()

            # Store results in the collector
            reply_result = dict(
                idents=idents,
                result=result,
                jobid=jobid,
                action='store',
            )
            print('...done working. pushing result to collector')
            dbgwait()
            collect_push_sock.send_json(reply_result)
        # ----
        print('Exiting scheduler')


def send_multipart_json(sock, idents, reply):
    reply_json = ut.to_json(reply).encode('utf-8')
    multi_reply = idents + [reply_json]
    sock.send_multipart(multi_reply)


def rcv_multipart_json(sock):
    # note that the first two parts will be ['Controller.ROUTER', 'Client.<id_>']
    # these are needed for the reply to propagate up to the right client
    multi_msg = sock.recv_multipart()
    print('----')
    print('Handling Request: %r' % (multi_msg,))
    idents = multi_msg[:2]
    request_json = multi_msg[2]
    request = ut.from_json(request_json)
    return idents, request


def collector_loop():
    """
    Service that stores completed algorithm results
    """
    print = partial(ut.colorprint, color='yellow')
    with ut.Indenter('[collect] '):

        collect_rout_sock = ctx.socket(zmq.ROUTER)
        collect_rout_sock.setsockopt_string(zmq.IDENTITY, 'COLLECTOR.collect_rout_sock')
        collect_rout_sock.connect(collect_iface2)
        print('connect collect_iface2  = %r' % (collect_iface2,))

        collect_pull_sock = ctx.socket(zmq.PULL)
        collect_pull_sock.setsockopt_string(zmq.IDENTITY, 'COLLECTOR.collect_pull_sock')
        collect_pull_sock.bind(collect_pushpull_iface)
        print('bind collect_pushpull_iface = %r' % (collect_pushpull_iface,))

        collecter_data = {}
        awaiting_data = {}

        def handle_collect():
            reply_result = collect_pull_sock.recv_json()
            print('HANDLING COLLECT')
            try:
                #reply_result = ut.from_json(reply_result)
                pass
            except TypeError:
                print('ERROR reply_result = %r' % (reply_result,))
                raise

            action = reply_result['action']
            if action == 'notification':
                print('notified about working job')
                jobid = reply_result['jobid']
                awaiting_data[jobid] = reply_result['text']
            elif action == 'store':
                print('Collected: %r' % (reply_result,))
                jobid = reply_result['jobid']
                collecter_data[jobid] = reply_result
                print('stored result')

        def handle_request():
            #pair_msg = collect_rout_sock.recv_json()
            idents, request = rcv_multipart_json(collect_rout_sock)
            reply = {}
            action = request['action']
            print('...building action=%r response' % (action,))
            if action == 'job_status':
                jobid = request['jobid']
                if jobid in collecter_data:
                    reply['jobstatus'] = 'completed'
                elif jobid in awaiting_data:
                    reply['jobstatus'] = 'working'
                else:
                    reply['jobstatus'] = 'unknown'
                reply['status'] = 'ok'
                reply['jobid'] = jobid
            elif action == 'job_result':
                jobid = request['jobid']
                result = collecter_data[jobid]['result']
                reply['jobid'] = jobid
                reply['status'] = 'ok'
                reply['result'] = result
            else:
                print('...error unknown action=%r' % (action,))
                reply['status'] = 'error'
                reply['text'] = 'unknown action'
            send_multipart_json(collect_rout_sock, idents, reply)

        #COLLECT_MODE = 'poller'
        COLLECT_MODE = 'timeout'
        if COLLECT_MODE is None:
            assert False
        elif COLLECT_MODE == 'timeout':
            collect_pull_sock.RCVTIMEO = 100
            collect_rout_sock.RCVTIMEO = 100
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
            poller.register(collect_pull_sock, zmq.POLLIN)
            poller.register(collect_rout_sock, zmq.POLLIN)
            # Polling event loop
            while True:
                print('polling')
                evnts = poller.poll(10)
                evnts = dict(poller.poll())
                if collect_pull_sock in evnts:
                    handle_collect()
                if collect_rout_sock in evnts:
                    handle_request()


def _on_ctrl_c(signal, frame):
    print('[ibeis.main_module] Caught ctrl+c')
    print('[ibeis.main_module] sys.exit(0)')
    import sys
    sys.exit(0)


def _init_signals():
    import signal
    signal.signal(signal.SIGINT, _on_ctrl_c)

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
