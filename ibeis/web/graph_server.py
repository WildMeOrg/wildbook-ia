# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from ibeis.control import controller_inject
import utool as ut
import random
import time
# from ibeis.web import futures_utils as futures_actors
import futures_actors
print, rrr, profile = ut.inject2(__name__)


def double_review_test():
    # from ibeis.web.graph_server import *
    import ibeis
    actor = GraphActor()
    config = {
        'manual.n_peek'   : 1,
        'manual.autosave' : False,
        'ranking.enabled' : False,
        'autoreview.enabled' : False,
        'redun.enabled'   : False,
        'redun.enabled'   : False,
        'queue.conf.thresh' : 'absolutely_sure',
        'algo.hardcase' : True,
    }
    # Start the process
    dbdir = ibeis.sysres.db_to_dbdir('PZ_MTEST')
    payload = {'action': 'start', 'dbdir': dbdir, 'aids': 'all',
               'config': config, 'init': 'annotmatch'}
    start_resp = actor.handle(payload)
    print('start_resp = {!r}'.format(start_resp))
    infr = actor.infr

    infr.verbose = 100

    user_resp = infr.continue_review()
    edge, p, d = user_resp[0]
    print('edge = {!r}'.format(edge))

    last = None

    while True:
        infr.add_feedback(edge, infr.edge_decision(edge))
        user_resp = infr.continue_review()
        edge, p, d = user_resp[0]
        print('edge = {!r}'.format(edge))
        assert last != edge
        last = edge

    # Respond with a user decision


def ut_to_json_encode(dict_):
    # Encode correctly for UUIDs and other information
    for key in dict_:
        dict_[key] = ut.to_json(dict_[key])
    return dict_


def testdata_start_payload(aids='all'):
    import ibeis
    payload = {
        'action'       : 'start',
        'dbdir'        : ibeis.sysres.db_to_dbdir('PZ_MTEST'),
        'aids'         : aids,
        'config'       : {
            'manual.n_peek'   : 50,
            'manual.autosave' : False,
        }
    }
    return payload


def testdata_feedback_payload(edge, decision):
    payload = {
        'action': 'add_feedback',
        'edge': edge,
        'evidence_decision': decision,
        'meta_decision': 'null',
        'tags': [],
        'user_id': 'user:doctest',
        'confidence': 'pretty_sure',
        'timestamp_s1': 1,
        'timestamp_c1': 2,
        'timestamp_c2': 3,
        'timestamp': 4,
    }
    return payload


def test_foo(future):
    print('FOO %r' % (future, ))


GRAPH_ACTOR_CLASS = futures_actors.ProcessActor if ut.LINUX or ut.WIN32 else futures_actors.ThreadActor


class GraphActor(GRAPH_ACTOR_CLASS):
    """

    CommandLine:
        python -m ibeis.web.graph_server GraphActor

    Doctest:
        >>> from ibeis.web.graph_server import *
        >>> actor = GraphActor()
        >>> payload = testdata_start_payload()
        >>> # Start the process
        >>> start_resp = actor.handle(payload)
        >>> print('start_resp = {!r}'.format(start_resp))
        >>> # Respond with a user decision
        >>> user_request = actor.handle({'action': 'continue_review'})
        >>> # Wait for a response and  the GraphActor in another proc
        >>> edge, priority, edge_data = user_request[0]
        >>> user_resp_payload = testdata_feedback_payload(edge, 'match')
        >>> content = actor.handle(user_resp_payload)
        >>> actor.infr.dump_logs()


    Doctest:
        >>> from ibeis.web.graph_server import *
        >>> import ibeis
        >>> actor = GraphActor()
        >>> config = {
        >>>     'manual.n_peek'   : 1,
        >>>     'manual.autosave' : False,
        >>>     'ranking.enabled' : False,
        >>>     'autoreview.enabled' : False,
        >>>     'redun.enabled'   : False,
        >>>     'redun.enabled'   : False,
        >>>     'queue.conf.thresh' : 'absolutely_sure',
        >>>     'algo.hardcase' : True,
        >>> }
        >>> # Start the process
        >>> dbdir = ibeis.sysres.db_to_dbdir('PZ_MTEST')
        >>> payload = {'action': 'start', 'dbdir': dbdir, 'aids': 'all',
        >>>            'config': config, 'init': 'annotmatch'}
        >>> start_resp = actor.handle(payload)
        >>> print('start_resp = {!r}'.format(start_resp))
        >>> # Respond with a user decision
        >>> user_request = actor.handle({'action': 'continue_review'})
        >>> print('user_request = {!r}'.format(user_request))
        >>> # Wait for a response and  the GraphActor in another proc
        >>> edge, priority, edge_data = user_request[0]
        >>> user_resp_payload = testdata_feedback_payload(edge, 'match')
        >>> content = actor.handle(user_resp_payload)
        >>> actor.infr.dump_logs()
        >>> actor.infr.status()
    """
    def __init__(actor):
        actor.infr = None

    def handle(actor, message):
        if not isinstance(message, dict):
            raise ValueError('Commands must be passed in a message dict')
        message = message.copy()
        action = message.pop('action', None)
        if action is None:
            raise ValueError('Payload must have an action item')
        if action == 'wait':
            num = message.get('num', 0)
            time.sleep(num)
            return message
        elif action == 'debug':
            return actor
        elif action == 'error':
            raise Exception('FOOBAR')
        elif action == 'logs':
            return actor.infr.logs
        else:
            func = getattr(actor, action, None)
            if func is None:
                raise ValueError('Unknown action=%r' % (action,))
            else:
                return func(**message)

    def start(actor, dbdir, aids='all', config={},
              **kwargs):
        import ibeis
        assert dbdir is not None, 'must specify dbdir'
        assert actor.infr is None, ('AnnotInference already running')
        ibs = ibeis.opendb(dbdir=dbdir, use_cache=False, web=False,
                           force_serial=True)
        # Create the AnnotInference
        actor.infr = ibeis.AnnotInference(ibs=ibs, aids=aids, autoinit=True)
        # Configure query_annot_infr
        for key in config:
            actor.infr.params[key] = config[key]
        # Initialize
        # TODO: Initialize state from staging reviews after annotmatch
        # timestamps (in case of crash)
        table = kwargs.get('init', 'staging')
        actor.infr.reset_feedback(table, apply=True)
        actor.infr.ensure_mst()
        actor.infr.apply_nondynamic_update()

        # Load random forests (TODO: should this be config specifiable?)
        actor.infr.load_published()

        # Start actor.infr Main Loop
        actor.infr.start_id_review()
        return 'initialized'

    def continue_review(actor):
        # This will signal on_request_review with the same data
        user_request = actor.infr.continue_review()
        return user_request

    def add_feedback(actor, **feedback):
        return actor.infr.accept(feedback)

    def add_annots(actor, aids):
        actor.infr.add_annots(aids)
        return 'added'

    def get_feat_extractor(actor):
        match_state_verifier = actor.infr.verifiers.get('match_state', None)
        if match_state_verifier is not None:
            return match_state_verifier.extr


@ut.reloadable_class
class GraphClient(object):
    """
    CommandLine:
        python -m ibeis.web.graph_server GraphClient

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.web.graph_server import *
        >>> import ibeis
        >>> client = GraphClient(autoinit=True)
        >>> # Start the GraphActor in another proc
        >>> payload = testdata_start_payload()
        >>> client.post(payload).result()
        >>> f1 = client.post({'action': 'continue_review'})
        >>> f1.add_done_callback(test_foo)
        >>> user_request = f1.result()
        >>> # Wait for a response and  the GraphActor in another proc
        >>> edge, priority, edge_data = user_request[0]
        >>> user_resp_payload = testdata_feedback_payload(edge, 'match')
        >>> f2 = client.post(user_resp_payload)
        >>> f2.result()
        >>> # Debug by getting the actor over a mp.Pipe
        >>> f3 = client.post({'action': 'debug'})
        >>> actor = f3.result()
        >>> actor.infr.dump_logs()
        >>> #print(client.post({'action': 'logs'}).result())

    # Ignore:
    #     >>> from ibeis.web.graph_server import *
    #     >>> import ibeis
    #     >>> client = GraphClient(autoinit=True)
    #     >>> # Start the GraphActor in another proc
    #     >>> client.post(testdata_start_payload(list(range(1, 10)))).result()
    #     >>> #
    #     >>> f1 = client.post({'action': 'continue_review'})
    #     >>> user_request = f1.result()
    #     >>> # The infr algorithm needs a review
    #     >>> edge, priority, edge_data = user_request[0]
    #     >>> #
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'continue_review'})
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'continue_review'})
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'wait', 'num': float(30)})
    #     >>> client.post({'action': 'continue_review'})
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'continue_review'})
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'continue_review'})
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'continue_review'})
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'continue_review'})
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'continue_review'})
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'continue_review'})
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'continue_review'})
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'continue_review'})
    #     >>> client.post(testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'continue_review'})

    """
    def __init__(client, graph_uuid=None, callbacks={}, autoinit=False):
        client.graph_uuid = graph_uuid
        client.callbacks = callbacks
        client.executor = None
        client.review_dict = {}
        client.review_vip = None
        client.futures = []

        client.aids = None
        client.config = None
        client.extr = None

        if autoinit:
            client.initialize()

    def initialize(client):
        client.executor = GraphActor.executor()

    def __del__(client):
        client.shutdown()

    def shutdown(client):
        if client.executor is not None:
            client.executor.shutdown(wait=True)
            client.executor = None

    def post(client, payload):
        if not isinstance(payload, dict) or 'action' not in payload:
            raise ValueError('payload must be a dict with an action')
        future = client.executor.post(payload)
        client.futures.append((payload['action'], future))
        return future

    def cleanup(client):
        # remove done items from our list
        new_futures = []
        for action, future in client.futures:
            if not future.done():
                if future.running():
                    new_futures.append((action, future))
                elif action == 'continue_review':
                    future.cancel()
                else:
                    new_futures.append((action, future))
        client.futures = new_futures

    def add_annots(client):
        raise NotImplementedError('not done yet')

    def update(client, data_list):
        print('UPDATING GRAPH CLIENT WITH {} ITEM(S):'.format(len(data_list)))
        print('First few are: ' + ut.repr4(data_list[0:3], si=2, precision=4))
        client.review_dict = {}
        client.review_vip = None
        if data_list is None:
            client.review_dict = None
        else:
            for (edge, priority, edge_data_dict) in data_list:
                aid1, aid2 = edge
                if aid2 < aid1:
                    aid1, aid2 = aid2, aid1
                edge = (aid1, aid2, )
                if client.review_vip is None:
                    client.review_vip = edge
                client.review_dict[edge] = (priority, edge_data_dict, )

    def check(client, edge):
        if edge not in client.review_dict:
            return None
        priority, data_dict = client.review_dict[edge]
        return edge, priority, data_dict

    def sample(client):
        if client.review_dict is None:
            raise controller_inject.WebReviewFinishedException(client.graph_uuid)
        print('SAMPLING')
        edge_list = list(client.review_dict.keys())
        if len(edge_list) == 0:
            return None
        if client.review_vip is not None and client.review_vip in edge_list:
            print('SHOWING VIP TO USER!!!')
            edge = client.review_vip
            client.review_vip = None
        else:
            print('VIP ALREADY SHOWN')
            edge = random.choice(edge_list)
        priority, data_dict = client.review_dict[edge]
        print('SAMPLED edge = {!r}'.format(edge))
        return edge, priority, data_dict


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.graph_server
        python -m ibeis.web.graph_server --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
