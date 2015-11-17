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


NUM_JOBS = 2
NUM_ENGINES = 1


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
        def wait_for_job_result(jobid):
            while True:
                reply = sender.get_job_status(jobid)
                if reply['jobstatus'] == 'completed':
                    reply = sender.get_job_result(jobid)
                    json_result = reply['json_result']
                    result = ut.from_json(json_result)
                    print('Job %r result = %r' % (jobid, result,))
                    return result
                time.sleep(3.0)

        print('[test] ... emit test1')
        jobid1 = sender.queue_job('helloworld', 1)
        wait_for_job_result(jobid1)
        #dbgwait()
        #sender.get_job_status(jobid1)
        #dbgwait()
        #jobid_list = [sender.queue_job('helloworld', 5) for _ in range(NUM_JOBS)]
        #jobid_list += [sender.queue_job('get_valid_aids')]
        jobid_list = []

        #identify_jobid = sender.queue_job('query_chips', [1], [3, 4, 5], cfgdict={'K': 1})
        identify_jobid = sender.queue_job('query_chips_simple_dict', [1], [3, 4, 5], cfgdict={'K': 1})

        #jobid2 = sender.queue_job('helloworld', 5)
        ##time.sleep(.2)
        #jobid4 = sender.queue_job('helloworld', 5)
        ##time.sleep(.2)
        #jobid5 = sender.queue_job('helloworld', 5)
        #time.sleep(.2)
        #jobid2 = 'foobar'

        for jobid in jobid_list:
            wait_for_job_result(jobid)

        wait_for_job_result(identify_jobid)

        #wait_for_job_result(jobid2)
        #wait_for_job_result(jobid4)
        #wait_for_job_result(jobid5)
        #sender.queue_job()
        dbgwait()
    print('FINISHED TEST SCRIPT')


class BackgroundProcs(object):
    def __init__(self):
        #self.num_engines = 3
        self.num_engines = NUM_ENGINES
        self.engine_queue_proc = None
        self.collect_queue_proc = None
        self.engine_procs = None
        self.collect_proc = None

    def __del__(self):
        print('Cleaning up job client')
        if self.engine_procs is not None:
            for i in self.engine_procs:
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
        self.verbose = 2

    def init(self):
        self.initialize_background_processes()
        self.initialize_client_thread()

    def initialize_client_thread(self):
        print = partial(ut.colorprint, color='blue')
        print('Initializing ClientProc')
        self.engine_deal_sock = ctx.socket(zmq.DEALER)
        self.engine_deal_sock.setsockopt_string(zmq.IDENTITY, 'client%s.engine.DEALER' % (self.id_,))
        self.engine_deal_sock.connect(engine_iface1)
        print('connect engine_iface1 = %r' % (engine_iface1,))

        self.collect_deal_sock = ctx.socket(zmq.DEALER)
        self.collect_deal_sock.setsockopt_string(zmq.IDENTITY, 'client%s.collect.DEALER' % (self.id_,))
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
            print('Queue job: %s' % (msg))
            self.engine_deal_sock.send_json(msg)
            if self.verbose > 2:
                print('..sent, waiting for response')
            resp = self.engine_deal_sock.recv_json()
            if self.verbose > 1:
                print('Got reply: %s' % ( resp))
            jobid = resp['jobid']
            return jobid

    def get_job_status(self, jobid):
        with ut.Indenter('[client %d] ' % (self.id_)):
            print = partial(ut.colorprint, color='teal')
            print('----')
            print('Request status of jobid=%r' % (jobid,))
            pair_msg = dict(action='job_status', jobid=jobid)
            self.collect_deal_sock.send_json(pair_msg)
            if self.verbose > 2:
                print('... waiting for collector reply')
            reply = self.collect_deal_sock.recv_json()
            if self.verbose > 1:
                print('got reply = %r' % (reply,))
        return reply

    def get_job_result(self, jobid):
        with ut.Indenter('[client %d] ' % (self.id_)):
            print = partial(ut.colorprint, color='teal')
            print('----')
            print('Request result of jobid=%r' % (jobid,))
            pair_msg = dict(action='job_result', jobid=jobid)
            self.collect_deal_sock.send_json(pair_msg)
            if self.verbose > 2:
                print('... waiting for collector reply')
            reply = self.collect_deal_sock.recv_json()
            if self.verbose > 1:
                print('got reply = %r' % (reply,))
        return reply


def make_queue_loop(iface1, iface2, name=None):
    """
    Standard queue loop

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
        with ut.Indenter('[%s] ' % (queue_name,)):
            print('Init make_queue_loop: name=%r' % (name,))
            # bind the client dealer to the queue router
            rout_sock = ctx.socket(zmq.ROUTER)
            rout_sock.setsockopt_string(zmq.IDENTITY, 'queue.' + name + '.' + 'ROUTER')
            rout_sock.bind(iface1)
            print('bind %s_iface2 = %r' % (name, iface1,))
            # bind the server router to the queue dealer
            deal_sock = ctx.socket(zmq.DEALER)
            deal_sock.setsockopt_string(zmq.IDENTITY, 'queue.' + name + '.' + 'DEALER')
            deal_sock.bind(iface2)
            print('bind %s_iface2 = %r' % (name, iface2,))
            if 1:
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

collect_queue_loop = make_queue_loop(collect_iface1, collect_iface2, name='collect')


def engine_queue_loop():
    """
    Specialized queue loop
    """
    iface1, iface2 = engine_iface1, engine_iface2
    name = 'engine'
    queue_name = name + '_queue'
    loop_name = queue_name + '_loop'
    print = partial(ut.colorprint, color='red')
    with ut.Indenter('[%s] ' % (queue_name,)):
        print('Init specialized make_queue_loop: name=%r' % (name,))
        # bind the client dealer to the queue router
        rout_sock = ctx.socket(zmq.ROUTER)
        rout_sock.setsockopt_string(zmq.IDENTITY, 'special_queue.' + name + '.' + 'ROUTER')
        rout_sock.bind(iface1)
        print('bind %s_iface2 = %r' % (name, iface1,))
        # bind the server router to the queue dealer
        deal_sock = ctx.socket(zmq.DEALER)
        deal_sock.setsockopt_string(zmq.IDENTITY, 'special_queue.' + name + '.' + 'DEALER')
        deal_sock.bind(iface2)
        print('bind %s_iface2 = %r' % (name, iface2,))

        collect_deal_sock = ctx.socket(zmq.DEALER)
        collect_deal_sock.setsockopt_string(zmq.IDENTITY, queue_name + '.collect.DEALER')
        collect_deal_sock.connect(collect_iface1)
        print('connect collect_iface1 = %r' % (collect_iface1,))
        job_counter = 0

        # but this shows what is really going on:
        poller = zmq.Poller()
        poller.register(rout_sock, zmq.POLLIN)
        poller.register(deal_sock, zmq.POLLIN)
        while True:
            evts = dict(poller.poll())
            if rout_sock in evts:
                # HACK GET REQUEST FROM CLIENT
                job_counter += 1
                idents, request = rcv_multipart_json(rout_sock, num=1, print=print)

                #jobid = 'result_%s' % (id_,)
                #jobid = 'result_%s' % (uuid.uuid4(),)
                jobid = 'jobid-%04d' % (job_counter,)
                print('Creating jobid %r' % (jobid,))

                # Reply immediately with a new jobid
                reply_notify = {
                    'jobid': jobid,
                    'status': 'ok',
                    'text': 'job accepted',
                    'action': 'notification',
                }
                request['jobid'] = jobid
                print('...notifying collector about new job')
                collect_deal_sock.send_json(reply_notify)
                print('... notifying client that job was accepted')
                send_multipart_json(rout_sock, idents, reply_notify)
                print('... notifying backend engine to start')
                send_multipart_json(deal_sock, idents, request)
            if deal_sock in evts:
                pass
        print('Exiting %s' % (loop_name,))


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
    print = partial(ut.colorprint, color='darkred')
    with ut.Indenter('[engine %d] ' % (id_)):
        print('Initializing engine')
        print('connect engine_iface2 = %r' % (engine_iface2,))
        ibs = None
        ibs = ut.DynStruct()
        if True:
            dbname = 'testdb1'
            ibs = ibeis.opendb(dbname)

        engine_rout_sock = ctx.socket(zmq.ROUTER)
        engine_rout_sock.connect(engine_iface2)

        collect_deal_sock = ctx.socket(zmq.DEALER)
        collect_deal_sock.setsockopt_string(zmq.IDENTITY, 'engine.collect.DEALER')
        collect_deal_sock.connect(collect_iface1)
        print('connect collect_iface1 = %r' % (collect_iface1,))
        while True:
            idents, request = rcv_multipart_json(engine_rout_sock, print=print)
            action = request['action']
            jobid  = request['jobid']
            args   = request['args']
            kwargs = request['kwargs']

            # Start working
            print('starting job=%r' % (jobid,))
            # Map actions to IBEISController calls here
            if action == 'helloworld':
                def helloworld(time_=0, *args, **kwargs):
                    time.sleep(time_)
                    retval = ('HELLO time_=%r ' % (time_,)) + ut.repr2((args, kwargs))
                    return retval
                action_func = helloworld
            else:
                # check for ibs func
                action_func = getattr(ibs, action)
                print('resolving to ibeis function')

            try:
                result = action_func(*args, **kwargs)
                exec_status = 'ok'
            except Exception as ex:
                result = ut.formatex(ex, keys=['jobid'], tb=True)
                exec_status = 'exception'

            json_result = ut.to_json(result)

            # Store results in the collector
            reply_result = dict(
                idents=idents,
                exec_status=exec_status,
                json_result=json_result,
                jobid=jobid,
                action='store',
            )
            print('...done working. pushing result to collector')
            collect_deal_sock.send_json(reply_result)
        # ----
        print('Exiting scheduler')


def send_multipart_json(sock, idents, reply):
    reply_json = ut.to_json(reply).encode('utf-8')
    multi_reply = idents + [reply_json]
    sock.send_multipart(multi_reply)


def rcv_multipart_json(sock, num=2, print=print):
    # note that the first two parts will be ['Controller.ROUTER', 'Client.<id_>']
    # these are needed for the reply to propagate up to the right client
    multi_msg = sock.recv_multipart()
    print('----')
    print('RCV Json: %r' % (multi_msg,))
    idents = multi_msg[:num]
    request_json = multi_msg[num]
    request = ut.from_json(request_json)
    return idents, request


def collector_loop():
    """
    Service that stores completed algorithm results
    """
    print = partial(ut.colorprint, color='yellow')
    with ut.Indenter('[collect] '):

        collect_rout_sock = ctx.socket(zmq.ROUTER)
        collect_rout_sock.setsockopt_string(zmq.IDENTITY, 'collect.ROUTER')
        collect_rout_sock.connect(collect_iface2)
        print('connect collect_iface2  = %r' % (collect_iface2,))

        collecter_data = {}
        awaiting_data = {}

        while True:
            idents, request = rcv_multipart_json(collect_rout_sock, print=print)
            reply = {}
            action = request['action']
            print('...building action=%r response' % (action,))
            if action == 'notification':
                jobid = request['jobid']
                awaiting_data[jobid] = request['text']
            elif action == 'store':
                jobid = request['jobid']
                collecter_data[jobid] = request
                print('stored result')
            elif action == 'job_status':
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
                json_result = collecter_data[jobid]['json_result']
                reply['jobid'] = jobid
                reply['status'] = 'ok'
                reply['json_result'] = json_result
            else:
                print('...error unknown action=%r' % (action,))
                reply['status'] = 'error'
                reply['text'] = 'unknown action'
            send_multipart_json(collect_rout_sock, idents, reply)


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
        with ut.Timer('full'):
            test_zmq_task()
    else:
        import utool as ut  # NOQA
        ut.doctest_funcs()
