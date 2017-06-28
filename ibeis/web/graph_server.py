# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import time
import zmq
import random
import ibeis
from functools import partial
import multiprocessing
print, rrr, profile = ut.inject2(__name__)


def _get_random_open_port():
    port = random.randint(1024, 49151)
    while not ut.is_local_port_open(port):
        port = random.randint(1024, 49151)
    assert ut.is_local_port_open(port)
    return port


PARALLEL_MODULE = 'thread'
if PARALLEL_MODULE == 'multiprocessing':
    ParUnit = multiprocessing.Process
else:
    import threading
    ParUnit = threading.Thread


class GraphServer2(ParUnit):

    def __init__(self, task_queue, result_queue):
        super(GraphServer2, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        """ main loop """
        terminate = False

        # Create GraphActor in separate process and send messages to it
        handler = GraphActor()

        while not terminate:
            message = self.task_queue.get()
            if True:
                print('self.name, message = {}, {}'.format(self.name, message))
            try:
                if message is StopIteration:
                    content = 'shutdown'
                    terminate = True
                else:
                    content = handler.handle(message)
            except Exception as ex:
                print('Error handling message')
                status = 'error'
                content = ut.formatex(ex, tb=True)
                content = ut.strip_ansi(content)
            else:
                status = 'success'

            # Send back result
            respose = {
                'status': status,
                'content': content
            }
            if True:
                print('Done. self.name, message = {}, {}'.format(self.name, message))
            self.task_queue.task_done()
            self.result_queue.put(respose)


class GraphActor(object):
    def __init__(handler):
        handler.infr = None

    def handle(handler, message):
        if not isinstance(message, dict):
            raise ValueError('Commands must be passed in a message dict')
        action = message.pop('action', None)
        if action is None:
            raise ValueError('Payload must have an action item')
        if action == 'hello world':
            import time
            time.sleep(message.get('wait', 0))
            content = 'hello world'
            print(content)
            return content
        elif action == 'debug':
            return handler
        elif action == 'start':
            return handler.start(**message)
        elif action == 'continue_review':
            return handler.continue_review(**message)
        else:
            raise ValueError('Unknown action=%r' % (action,))

    def start(handler, dbdir, aids='all', config={}):
        import ibeis
        assert dbdir is not None, 'must specify dbdir'
        assert handler.infr is None, ('AnnotInference already running')
        ibs = ibeis.opendb(dbdir=dbdir, use_cache=False, web=False,
                           force_serial=True)
        handler.infr = ibeis.AnnotInference(ibs=ibs, aids=aids, autoinit=True)
        # Configure callbacks
        handler.infr.callbacks['request_review'] = handler.on_request_review
        # Configure query_annot_infr
        handler.infr.set_config(config)
        handler.infr.queue_params['pos_redun'] = 2
        handler.infr.queue_params['neg_redun'] = 2
        # Initialize
        handler.infr.reset_feedback('annotmatch', apply=True)
        handler.infr.ensure_mst()
        handler.infr.apply_nondynamic_update()

        # Start Main Loop
        handler.infr.refresh_candidate_edges()
        return handler.continue_review()

    def continue_review(handler):
        handler.infr.continue_review()
        return 'ready_for_interaction'

    def on_request_review(handler, edge, priority):
        print('handler.on_request_review edge = %r' % (edge,))
        import requests
        data_dict = {
            'graph_uuid': 'TADA',
            'nonce': '0123456789',
            'edge': edge,
            'priority': priority,
        }
        ibeis_flask_url = 'http://127.0.0.1:5000/internal/query/graph/v2/'
        if False:
            requests.post(ibeis_flask_url, data=data_dict)


class GraphClient2(object):
    """
    Example:
        >>> from ibeis.web.graph_server import *
        >>> import ibeis
        >>> dbdir = ibeis.sysres.db_to_dbdir('PZ_MTEST')
        >>> client = GraphClient2(dbdir, autoinit=True)
        >>> #client.post({'action': 'debug'})
        >>> #sofar = list(client.results())
        >>> #pass
        >>> #payload = {'action': 'hello world'}
        >>> #client.post(payload)
        >>> #sofar = list(client.results())
        >>> client.post({'action': 'start', 'dbdir': dbdir})
        >>> sofar = list(client.results())
        >>> print('sofar = %r' % (sofar,))
        >>> print(sofar[0]['content'])
    """
    def __init__(client, dbdir, autoinit=False):
        client.task_queue = None
        client.result_queue = None
        client.server = None
        if autoinit:
            client.initialize()

    def initialize(client):
        client.task_queue = multiprocessing.JoinableQueue()
        client.result_queue = multiprocessing.Queue()
        client.server = GraphServer2(client.task_queue, client.result_queue)
        client.server.start()

    def post(client, payload):
        client.task_queue.put(payload)

    def results(client):
        while not client.result_queue.empty():
            yield client.result_queue.get()

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

ctx = zmq.Context.instance()

URL = 'tcp://127.0.0.1'
VERBOSE_JOBS = ut.get_argflag('--bg') or ut.get_argflag('--verbose-jobs')


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
    def __init__(backend, dbdir, wait=0):
        backend.dbdir = dbdir
        backend.engine_proc = None
        backend.port_dict = None
        backend._initialize_job_ports()
        backend._initialize_background_processes(wait=wait)

    def __del__(backend):
        if VERBOSE_JOBS:
            print('Cleaning up job backend')
        if backend.engine_proc is not None:
            backend.engine_proc.terminate()
        if VERBOSE_JOBS:
            print('Killed external procs')

    def _initialize_job_ports(backend):
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

    def _initialize_background_processes(backend, wait=0):
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
        server.infr.set_config(query_config_dict)
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
