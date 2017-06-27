# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
#if False:
#    import os
#    os.environ['UTOOL_NOCNN'] = 'True'
import utool as ut
import time
import zmq
import uuid  # NOQA
import random
import ibeis
from functools import partial
print, rrr, profile = ut.inject2(__name__)

ctx = zmq.Context.instance()

URL = 'tcp://127.0.0.1'
VERBOSE_JOBS = ut.get_argflag('--bg') or ut.get_argflag('--verbose-jobs')


def _get_random_open_port():
    port = random.randint(1024, 49151)
    while not ut.is_local_port_open(port):
        port = random.randint(1024, 49151)
    assert ut.is_local_port_open(port)
    return port


class GraphClient(object):
    def __init__(client, **kwargs):
        client.id_ = 0
        client.ready = False
        client.verbose = 2 if VERBOSE_JOBS else 1
        client.backend = GraphBackend(**kwargs)
        client.initialize_client_thread()

    def initialize_client_thread(client):
        """
        Creates a ZMQ object in this thread. This talks to background processes.
        """
        print = partial(ut.colorprint, color='blue')
        print('GraphClient ports:')
        print(ut.repr4(client.backend.port_dict))

        if client.verbose:
            print('Initializing GraphClient')
        client.engine_deal_sock = ctx.socket(zmq.DEALER)
        client.engine_deal_sock.setsockopt_string(zmq.IDENTITY,
                                                    'client%s.server.DEALER' %
                                                    (client.id_,))
        client.engine_deal_sock.connect(client.backend.port_dict['engine_url'])
        if client.verbose:
            print('connect engine_url = %r' % (client.backend.port_dict['engine_url'],))

    def post(client, json_dict):
        r"""The client - sends messages, and receives replies."""
        with ut.Indenter('[client %d] ' % (client.id_)):
            print = partial(ut.colorprint, color='blue')
            if client.verbose >= 1:
                print('----')
            client.engine_deal_sock.send_json(json_dict)
            if client.verbose >= 3:
                print('..sent, waiting for response')
            # RETURNED FROM: job_client_return
            reply_dict = client.engine_deal_sock.recv_json()
            if client.verbose >= 2:
                print('Got reply: %s' % ( reply_dict))
            return reply_dict

    def push(client, edge, priority):
        if not client.ready:
            client.on_ready()

    def on_ready(client):
        client.ready = True
        client.callback('ready')

    def on_finish(client):
        client.callback('')


class GraphBackend(object):
    def __init__(backend, dbdir, **kwargs):
        backend.dbdir = dbdir
        backend.engine_proc = None
        backend.port_dict = None
        backend._initialize_job_ports(**kwargs)
        backend._initialize_background_processes(**kwargs)

    def __del__(backend):
        if VERBOSE_JOBS:
            print('Cleaning up job backend')
        if backend.engine_proc is not None:
            backend.engine_proc.terminate()
        if VERBOSE_JOBS:
            print('Killed external procs')

    def _initialize_job_ports(backend, **kwargs):
        key_list = [
            'engine_url',
        ]
        # Get ports
        port_list = []
        while len(port_list) < len(key_list):
            port = _get_random_open_port()
            if port not in port_list:
                port_list.append(port)
        port_list = sorted(port_list)
        # Assign ports
        assert len(key_list) == len(port_list)
        backend.port_dict = {
            key : '%s:%d' % (URL, port)
            for key, port in zip(key_list, port_list)
        }

    def _initialize_background_processes(backend, wait=0, **kwargs):
        print = partial(ut.colorprint, color='fuchsia')
        #if VERBOSE_JOBS:
        print('Initialize Background Processes')
        def _spawner(func, *args, **kwargs):
            if wait != 0:
                print('Waiting for background process (%s) to spin up' % (ut.get_funcname(func,)))
            proc = ut.spawn_background_process(func, *args, **kwargs)
            # time.sleep(wait)
            assert proc.is_alive(), 'proc (%s) died too soon' % (ut.get_funcname(func,))
            return proc

        url = backend.port_dict['engine_url']
        backend.engine_proc = _spawner(GraphServer.main, url, backend.dbdir)
        assert backend.engine_procs.is_alive(), 'server died too soon'


##########################################################################################


class ZMQServer(object):
    @classmethod
    def main(cls, url, *args, **kwargs):
        print = partial(ut.colorprint, color='darkred')

        if VERBOSE_JOBS:
            print('Initializing server %r' % (cls, ))
            print('connect url = %r' % (url, ))

        server = cls(*args, **kwargs)
        engine_rout_sock = ctx.socket(zmq.ROUTER)
        engine_rout_sock.connect(url)

        try:
            while True:
                idents, json_dict = rcv_multipart_json(engine_rout_sock, print=print)
                try:
                    result_dict = server.on_request(json_dict)
                except Exception as ex:
                    result = ut.formatex(ex, tb=True)
                    result = ut.strip_ansi(result)
                    # Send back result
                    result_dict = {
                        'status': 'error',
                        'content': result
                    }
                if result_dict:
                    # TODO: do something if the request returns an object
                    pass

        except KeyboardInterrupt:
            print('Caught ctrl+c in server loop. Gracefully exiting')
        # ----
        if VERBOSE_JOBS:
            print('Exiting server loop')


class GraphServer(ZMQServer):
    def __init__(server, dbdir):
        server.dbdir = dbdir
        assert dbdir is not None
        server.ibs = ibeis.opendb(dbdir=dbdir, use_cache=False, web=False, force_serial=True)
        server.infr = None

    def on_request(server, json_dict):
        """ Run whenever the server recieves a message """
        action = json_dict.pop('action', None)

        # Map actions to IBEISController calls here
        if action == 'helloworld':
            time_ = 3.0
            time.sleep(time_)
            result = 'HELLO time_=%r ' % (time_,)
        elif action == 'start':
            result = server.start(**json_dict)
        else:
            raise KeyError('Unknown action = %r given to GraphServer' % (action, ))

        result_dict = {
            'status': 'success',
            'content': result
        }
        return result_dict

    def start(server, aid_list, query_config_dict, callback_dict):
        assert server.infr is None, 'Cannot start the AnnotInference twice per process'
        server.infr = ibeis.AnnotInference(ibs=server.ibs, aids=aid_list, autoinit=True)
        # Configure callbacks
        server.infr.connect_manual_review_callback(server.on_manual_review)
        # Configure query_annot_infr
        server.infr.set_config(**query_config_dict)
        server.infr.queue_params['pos_redun'] = 2
        server.infr.queue_params['neg_redun'] = 2
        # Initialize
        server.infr.reset_feedback('annotmatch', apply=True)
        server.infr.ensure_mst()
        server.infr.apply_nondynamic_update()
        # Register callbacks
        server.infr.set_callbacks(**callback_dict)
        # Start Main Loop
        server.main_loop()

    def main_loop(server):
        server.infr.lnbnn_priority_loop()

    def on_manual_review_request(server, edge, priority):
        import requests
        data_dict = {
            'graph_uuid': 'TADA',
            'nonce': '0123456789',
            'edge': edge,
            'priority': priority,
        }
        ibeis_flask_url = 'http://127.0.0.1:5000/internal/query/graph/v2/'
        requests.post(ibeis_flask_url, data=data_dict)


def send_multipart_json(sock, idents, reply):
    """ helper """
    reply_json = ut.to_json(reply).encode('utf-8')
    multi_reply = idents + [reply_json]
    sock.send_multipart(multi_reply)


def rcv_multipart_json(sock, num=2, print=print):
    """ helper """
    # note that the first two parts will be ['Controller.ROUTER', 'Client.<id_>']
    # these are needed for the reply to propagate up to the right client
    multi_msg = sock.recv_multipart()
    if VERBOSE_JOBS:
        print('----')
        print('RCV Json: %s' % (ut.repr2(multi_msg, truncate=True),))
    idents = multi_msg[:num]
    request_json = multi_msg[num]
    request = ut.from_json(request_json)
    return idents, request


def _on_ctrl_c(signal, frame):
    print('[ibeis.zmq] Caught ctrl+c')
    print('[ibeis.zmq] sys.exit(0)')
    import sys
    sys.exit(0)


def _init_signals():
    import signal
    signal.signal(signal.SIGINT, _on_ctrl_c)

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.job_engine
        python -m ibeis.web.job_engine --allexamples
        python -m ibeis.web.job_engine --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
