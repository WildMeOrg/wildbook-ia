# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import random
import time
import uuid
import multiprocessing
import collections
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
        actor = GraphActor()

        while not terminate:
            message = self.task_queue.get()
            if True:
                print('self.name, message = {}, {}'.format(self.name, message))
            try:
                if message is StopIteration:
                    content = 'shutdown'
                    terminate = True
                else:
                    # TODO: Maybe we can make a Futures object that will store
                    # the results of this message instead of using the results
                    # queue
                    content = actor.handle(message)
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
    """

    Doctest:
        >>> from ibeis.web.graph_server import *
        >>> client = GraphClient(autoinit=False)
        >>> actor = GraphActor()
        >>> payload = client._test_start_payload()
        >>> locals().update(payload)
        >>> # Start the process
        >>> content = actor.handle(payload)
        >>> user_request = content['user_request']
        >>> # Respond with a user decision
        >>> edge, priority, edge_data = user_request[0]
        >>> user_resp_payload = {
        >>>     'action': 'add_feedback',
        >>>     'edge': edge,
        >>>     'evidence_decision': 'match',
        >>>     'meta_decision': 'null',
        >>>     'tags': [],
        >>>     'user_id': 'user:doctest',
        >>>     'confidence': 'pretty_sure',
        >>>     'timestamp_s1': 1,
        >>>     'timestamp_c1': 2,
        >>>     'timestamp_c2': 3,
        >>>     'timestamp': 4,
        >>> }
        >>> content = actor.handle(user_resp_payload)
        >>> actor.infr.dump_logs()

    """
    def __init__(actor):
        actor.infr = None
        actor.callbacks = None

    def handle(actor, message):
        if not isinstance(message, dict):
            raise ValueError('Commands must be passed in a message dict')
        message = message.copy()
        action = message.pop('action', None)
        if action is None:
            raise ValueError('Payload must have an action item')
        if action == 'hello world':
            time.sleep(message.get('wait', 0))
            content = 'hello world'
            print(content)
            return content
        elif action == 'debug':
            return actor
        elif action == 'logs':
            return actor.infr.logs
        elif action == 'start':
            return actor.start(**message)
        elif action == 'continue_review':
            return actor.continue_review(**message)
        elif action == 'add_feedback':
            return actor.add_feedback(**message)
        else:
            raise ValueError('Unknown action=%r' % (action,))

    def start(actor, dbdir, callbacks, aids='all', config={},
              **kwargs):
        import ibeis
        assert dbdir is not None, 'must specify dbdir'
        assert actor.infr is None, ('AnnotInference already running')
        ibs = ibeis.opendb(dbdir=dbdir, use_cache=False, web=False,
                           force_serial=True)
        # Save the callbacks config
        actor.callbacks = callbacks
        for key in ['uuid', 'nonce', 'urls']:
            assert key in actor.callbacks
        for key in ['review', 'ready', 'finished']:
            assert key in actor.callbacks['urls']
        # Create the AnnotInference
        actor.infr = ibeis.AnnotInference(ibs=ibs, aids=aids, autoinit=True)
        # Configure callbacks
        actor.infr.callbacks['request_review'] = actor.on_request_review
        actor.infr.callbacks['review_ready'] = actor.on_review_ready
        actor.infr.callbacks['review_finished'] = actor.on_review_finished
        # Configure query_annot_infr
        actor.infr.params['manual.n_peek'] = 50
        # actor.infr.params['autoreview.enabled'] = False
        actor.infr.params['redun.pos'] = 2
        actor.infr.params['redun.neg'] = 2
        # Initialize
        # TODO: Initialize state from staging reviews after annotmatch
        # timestamps (in case of crash)
        actor.infr.reset_feedback('annotmatch', apply=True)
        actor.infr.ensure_mst()

        # Load random forests (TODO: should this be config specifiable?)
        actor.infr.load_published()

        actor.infr.apply_nondynamic_update()

        # Start Main Loop
        actor.infr.refresh_candidate_edges()
        actor.infr.print('begin review loop')
        return actor.continue_review()

    def continue_review(actor, **kwargs):
        # This will signal on_request_review with the same data
        user_request = actor.infr.continue_review()
        if user_request:
            msg = 'waiting_for_user'
        else:
            msg = 'queue_is_empty'
        # we return it here as well for local tests
        content = {
            'message': msg,
            'user_request': user_request,
        }
        return content

    def add_feedback(actor, edge, evidence_decision, meta_decision, tags,
                     user_id, confidence, timestamp_s1, timestamp_c1,
                     timestamp_c2, timestamp):
        content = actor.infr.add_feedback(
            edge, evidence_decision=evidence_decision,
            meta_decision=meta_decision, tags=tags, user_id=user_id,
            confidence=confidence, timestamp_s1=timestamp_s1,
            timestamp_c1=timestamp_c1, timestamp_c2=timestamp_c2,
            timestamp=timestamp,
        )
        actor.infr.write_ibeis_staging_feedback()
        return content

    def on_request_review(actor, data_list):
        print('actor.on_request_review %d edges' % (len(data_list), ))
        import requests
        callback_url = actor.callbacks['urls']['review']
        if callback_url is not None:
            data_dict = ut_to_json_encode({
                'nonce': actor.callbacks['nonce'],
                'graph_uuid': actor.callbacks['uuid'],
                'data': data_list,
            })
            # Send
            requests.post(callback_url, data=data_dict)

    def on_review_ready(actor):
        print('actor.on_review_ready')
        import requests
        callback_url = actor.callbacks['urls']['ready']
        if callback_url is not None:
            data_dict = ut_to_json_encode({
                'nonce': actor.callbacks['nonce'],
                'graph_uuid': actor.callbacks['uuid'],
            })
            # Send
            requests.post(callback_url, data=data_dict)

    def on_review_finished(actor):
        print('actor.on_review_ready')
        import requests
        callback_url = actor.callbacks['urls']['finished']
        if callback_url is not None:
            data_dict = ut_to_json_encode({
                'nonce': actor.callbacks['nonce'],
                'graph_uuid': actor.callbacks['uuid'],
            })
            # Send
            requests.post(callback_url, data=data_dict)


@ut.reloadable_class
class GraphClient(object):
    """
    Example:
        >>> from ibeis.web.graph_server import *
        >>> import ibeis
        >>> client = GraphClient(autoinit=True)
        >>> # Start the GraphActor in another proc
        >>> payload = client._test_start_payload()
        >>> result = client.get(payload)
        >>> user_request = result['content']['user_request']
        >>> # Wait for a response and  the GraphActor in another proc
        >>> edge, priority, edge_data = user_request[0]
        >>> user_resp_payload = {
        >>>     'action': 'add_feedback',
        >>>     'edge': edge,
        >>>     'evidence_decision': 'match',
        >>>     'meta_decision': 'null',
        >>>     'tags': [],
        >>>     'user_id': 'user:doctest',
        >>>     'confidence': 'pretty_sure',
        >>>     'timestamp_s1': 1,
        >>>     'timestamp_c1': 2,
        >>>     'timestamp_c2': 3,
        >>>     'timestamp': 4,
        >>> }
        >>> result = client.get(user_resp_payload)
        >>>
        >>> content = actor.handle(user_resp_payload)
        >>> # Debug by getting the actor over a mp.Pipe
        >>> result = client.get({'action': 'debug'})
        >>> actor = result['content']
        >>> actor.infr.dump_logs()
        >>> #print(client.get({'action': 'logs'})['content'])
        >>> #actor = result['content']

    """
    def __init__(client, autoinit=False, callbacks={}):
        client.nonce = uuid.uuid4()

        client.task_queue = None
        client.result_queue = None
        client.server = None

        client.review_dict = {}
        client.review_vip = None

        client.result_history = collections.deque(maxlen=1000)

        client.callbacks = callbacks
        if autoinit:
            client.initialize()

    def _test_start_payload(client):
        import ibeis
        payload = {
            'action'           : 'start',
            'dbdir'            : ibeis.sysres.db_to_dbdir('PZ_MTEST'),
            'aids'             : 'all',
            'callbacks'        : {
                'nonce'        : client.nonce,
                'uuid'         : None,
                'urls'         : {
                    'review'   : None,
                    'ready'    : None,
                    'finished' : None,
                },
            },
        }
        return payload

    def __del__(client):
        if client.server and client.server.is_alive():
            client.get(StopIteration)
    #     client.task_queue = multiprocessing.JoinableQueue()
    #     client.result_queue = multiprocessing.Queue()
    #     client.server = GraphServer(client.task_queue, client.result_queue)

    def initialize(client):
        client.task_queue = multiprocessing.JoinableQueue()
        client.result_queue = multiprocessing.Queue()
        client.server = GraphServer(client.task_queue, client.result_queue)
        client.server.start()

    def post(client, payload):
        client.task_queue.put(payload)

    def get(client, payload):
        # Exhaust any existing results
        list(client.results())
        # Post the command
        client.post(payload)
        # Wait for a response
        result = client.wait_for_result()
        return result

    def wait_for_result(client):
        wait = 0
        while True:
            for result in client.results():
                return result
            time.sleep(wait)
            wait = max(1, wait + .01)

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
            item = client.result_queue.get()
            client.result_history.append(item)
            yield item


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
