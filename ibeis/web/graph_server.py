# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import random
import time
import uuid
import multiprocessing
print, rrr, profile = ut.inject2(__name__)


def ut_to_json_encode(dict_):
    # Encode correctly for UUIDs and other information
    for key in dict_:
        dict_[key] = ut.to_json(dict_[key])
    return dict_


class GraphServer(ut.KillableProcess):

    def __init__(self, task_queue, result_queue):
        super(GraphServer, self).__init__()
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
            response = {
                'status': status,
                'content': content
            }
            if True:
                print('GraphServer Task Done')
                print('\tself.name = {}'.format(self.name))
                print('\tmessage   = {}'.format(message))
                print('\tstatus    = {}'.format(status))
                print('\tcontent   = {}'.format(content))
            self.task_queue.task_done()
            self.result_queue.put(response)


class GraphActor(object):
    def __init__(handler):
        handler.infr = None
        handler.callbacks = None

    def handle(handler, message):
        if not isinstance(message, dict):
            raise ValueError('Commands must be passed in a message dict')
        action = message.pop('action', None)
        if action is None:
            raise ValueError('Payload must have an action item')
        if action == 'hello world':
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
        elif action == 'add_feedback':
            return handler.add_feedback(**message)
        else:
            raise ValueError('Unknown action=%r' % (action,))

    def start(handler, dbdir, callbacks, aids='all', config={},
              **kwargs):
        import ibeis
        assert dbdir is not None, 'must specify dbdir'
        assert handler.infr is None, ('AnnotInference already running')
        ibs = ibeis.opendb(dbdir=dbdir, use_cache=False, web=False,
                           force_serial=True)
        # Save the callbacks config
        handler.callbacks = callbacks
        for key in ['uuid', 'nonce', 'urls']:
            assert key in handler.callbacks
        for key in ['review', 'ready', 'finished']:
            assert key in handler.callbacks['urls']
        # Create the AnnotInference
        handler.infr = ibeis.AnnotInference(ibs=ibs, aids=aids, autoinit=True)
        # Configure callbacks
        handler.infr.callbacks['request_review'] = handler.on_request_review
        handler.infr.callbacks['review_ready'] = handler.on_review_ready
        handler.infr.callbacks['review_finished'] = handler.on_review_finished
        # Configure query_annot_infr
        handler.infr.params['manual.n_peek'] = 50
        # handler.infr.params['autoreview.enabled'] = False
        handler.infr.params['redun.pos'] = 2
        handler.infr.params['redun.neg'] = 2
        # Initialize
        # TODO: Initialize state from staging reviews after annotmatch timestamps (in case of crash)
        handler.infr.reset_feedback('annotmatch', apply=True)
        handler.infr.ensure_mst()
        handler.infr.apply_nondynamic_update()

        # Start Main Loop
        handler.infr.refresh_candidate_edges()
        return handler.continue_review()

    def continue_review(handler, **kwargs):
        handler.infr.continue_review()
        return 'waiting_for_interaction'

    def add_feedback(handler, edge, decision, tags, user_id, confidence,
                     user_times, **kwargs):
        handler.infr.add_feedback(edge, decision, tags=tags, user_id=user_id,
                                  confidence=confidence, user_times=user_times)
        handler.infr.write_ibeis_staging_feedback()

    def on_request_review(handler, data_list):
        print('handler.on_request_review %d edges' % (len(data_list), ))
        import requests
        callback_url = handler.callbacks['urls']['review']
        if callback_url is not None:
            data_dict = ut_to_json_encode({
                'nonce': handler.callbacks['nonce'],
                'graph_uuid': handler.callbacks['uuid'],
                'data': data_list,
            })
            # Send
            requests.post(callback_url, data=data_dict)

    def on_review_ready(handler):
        print('handler.on_review_ready')
        import requests
        callback_url = handler.callbacks['urls']['ready']
        if callback_url is not None:
            data_dict = ut_to_json_encode({
                'nonce': handler.callbacks['nonce'],
                'graph_uuid': handler.callbacks['uuid'],
            })
            # Send
            requests.post(callback_url, data=data_dict)

    def on_review_finished(handler):
        print('handler.on_review_ready')
        import requests
        callback_url = handler.callbacks['urls']['finished']
        if callback_url is not None:
            data_dict = ut_to_json_encode({
                'nonce': handler.callbacks['nonce'],
                'graph_uuid': handler.callbacks['uuid'],
            })
            # Send
            requests.post(callback_url, data=data_dict)


class GraphClient(object):
    """
    Example:
        >>> from ibeis.web.graph_server import *
        >>> import ibeis
        >>> client = GraphClient(autoinit=True)
        >>> #client.post({'action': 'debug'})
        >>> #sofar = list(client.results())
        >>> #pass
        >>> #payload = {'action': 'hello world'}
        >>> #client.post(payload)
        >>> #sofar = list(client.results())
        >>> payload = {
        >>>     'action'           : 'start',
        >>>     'dbdir'            : ibeis.sysres.db_to_dbdir('PZ_MTEST'),
        >>>     'aids'             : 'all',
        >>>     'callbacks'        : {
        >>>         'nonce'        : None,
        >>>         'uuid'         : None,
        >>>         'urls'         : {
        >>>             'review'   : None,
        >>>             'ready'    : None,
        >>>             'finished' : None,
        >>>         },
        >>>     },
        >>> }
        >>> client.post(payload)
        >>> sofar = list(client.results())
        >>> print('sofar = %r' % (sofar,))
        >>> print(sofar[0]['content'])
    """
    def __init__(client, autoinit=False, callbacks={}):
        client.nonce = uuid.uuid4()

        client.task_queue = None
        client.result_queue = None
        client.server = None

        client.review_dict = {}
        client.review_vip = None

        client.callbacks = callbacks
        if autoinit:
            client.initialize()

    def __del__(client):
        client.task_queue = multiprocessing.JoinableQueue()
        client.result_queue = multiprocessing.Queue()
        client.server = GraphServer(client.task_queue, client.result_queue)

    def initialize(client):
        client.task_queue = multiprocessing.JoinableQueue()
        client.result_queue = multiprocessing.Queue()
        client.server = GraphServer(client.task_queue, client.result_queue)
        client.server.start()

    def post(client, payload):
        client.task_queue.put(payload)

    def update(client, data_list):
        client.review_dict = []
        client.review_vip = None
        for (edge, priority, edge_data_dict) in data_list:
            aid1, aid2 = edge
            if aid2 < aid1:
                aid1, aid2 = aid2, aid1
            edge = (aid1, aid2, )
            if client.review_vip is None:
                client.review_vip = edge
            client.review_dict[edge] = (priority, edge_data_dict, )

    def sample(client):
        edge_list = client.review_dict.keys()
        if len(edge_list) == 0:
            return None
        if client.review_vip is not None and client.review_vip in edge_list:
            edge = client.review_vip
            client.review_vip = None
        else:
            edge = random.choice(edge_list)
        priority, data_dict = client.review_dict[edge]
        return edge, priority, data_dict

    def results(client):
        while not client.result_queue.empty():
            yield client.result_queue.get()


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.job_engine
        python -m ibeis.web.job_engine --allexamples
        python -m ibeis.web.job_engine --allexamples --noface --nosrc
    """
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
