# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import random
import time
import uuid
from ibeis.web import futures_utils
print, rrr, profile = ut.inject2(__name__)


def ut_to_json_encode(dict_):
    # Encode correctly for UUIDs and other information
    for key in dict_:
        dict_[key] = ut.to_json(dict_[key])
    return dict_


class GraphActor(futures_utils.ProcessActor):
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
    CommandLine:
        python -m ibeis.web.graph_server GraphClient

    Example:
        >>> from ibeis.web.graph_server import *
        >>> import ibeis
        >>> client = GraphClient(autoinit=True)
        >>> # Start the GraphActor in another proc
        >>> payload = client._test_start_payload()
        >>> f1 = client.post(payload)
        >>> user_request = f1.result()['user_request']
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
        >>> f2 = client.post(user_resp_payload)
        >>> f2.result()
        >>> # Debug by getting the actor over a mp.Pipe
        >>> f3 = client.post({'action': 'debug'})
        >>> actor = f3.result()
        >>> actor.infr.dump_logs()
        >>> #print(client.post({'action': 'logs'}).result())

    """
    def __init__(client, autoinit=False, callbacks={}):
        client.nonce = uuid.uuid4()
        client.review_dict = {}
        client.review_vip = None
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

    def initialize(client):
        client.executor = GraphActor.executor()

    def post(client, payload):
        return client.executor.post(payload)

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
