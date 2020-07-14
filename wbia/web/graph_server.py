# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from wbia.control import controller_inject
import utool as ut
import concurrent
import random
import time

# from wbia.web import futures_utils as futures_actors
import futures_actors

print, rrr, profile = ut.inject2(__name__)


def double_review_test():
    # from wbia.web.graph_server import *
    import wbia

    actor = GraphActor()
    config = {
        'manual.n_peek': 1,
        'manual.autosave': False,
        'ranking.enabled': False,
        'autoreview.enabled': False,
        'redun.enabled': False,
        'queue.conf.thresh': 'absolutely_sure',
        'algo.hardcase': True,
    }
    # Start the process
    dbdir = wbia.sysres.db_to_dbdir('PZ_MTEST')
    payload = {
        'action': 'start',
        'dbdir': dbdir,
        'aids': 'all',
        'config': config,
        'init': 'annotmatch',
    }
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
    import wbia

    payload = {
        'action': 'start',
        'dbdir': wbia.sysres.db_to_dbdir('PZ_MTEST'),
        'aids': aids,
        'config': {'manual.n_peek': 50, 'manual.autosave': False},
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
    print('FOO %r' % (future,))


GRAPH_ACTOR_CLASS = futures_actors.ProcessActor


class GraphActor(GRAPH_ACTOR_CLASS):
    """

    CommandLine:
        python -m wbia.web.graph_server GraphActor

    Doctest:
        >>> from wbia.web.graph_server import *
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
        >>> from wbia.web.graph_server import *
        >>> import wbia
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
        >>> dbdir = wbia.sysres.db_to_dbdir('PZ_MTEST')
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
        elif action == 'latest_logs':
            return actor.infr.latest_logs(colored=True)
        elif action == 'logs':
            return actor.infr.logs
        else:
            func = getattr(actor, action, None)
            if func is None:
                raise ValueError('Unknown action=%r' % (action,))
            else:
                try:
                    return func(**message)
                except Exception as ex:
                    import sys
                    import traceback

                    traceback.print_exc()
                    trace = traceback.format_exc()

                    if actor.infr is not None:
                        actor.infr.print('Actor Server Error: {!r}'.format(ex))
                        actor.infr.print('Actor Server Traceback: {!r}'.format(trace))
                    else:
                        print(ex)
                        print(trace)

                    raise sys.exc_info()[0](trace)

    def start(actor, dbdir, aids='all', config={}, **kwargs):
        import wbia

        assert dbdir is not None, 'must specify dbdir'
        assert actor.infr is None, 'AnnotInference already running'
        ibs = wbia.opendb(dbdir=dbdir, use_cache=False, web=False, force_serial=True)

        # Create the AnnotInference
        print('starting via actor with ibs = %r' % (ibs,))
        actor.infr = wbia.AnnotInference(ibs=ibs, aids=aids, autoinit=True)
        actor.infr.print('started via actor')
        actor.infr.print('config = {}'.format(ut.repr3(config)))
        # Configure query_annot_infr
        for key in config:
            actor.infr.params[key] = config[key]
        # Initialize
        # TODO: Initialize state from staging reviews after annotmatch
        # timestamps (in case of crash)

        actor.infr.print('Initializing infr tables')
        table = kwargs.get('init', 'staging')
        actor.infr.reset_feedback(table, apply=True)
        actor.infr.ensure_mst()
        actor.infr.apply_nondynamic_update()

        actor.infr.print('infr.status() = {}'.format(ut.repr4(actor.infr.status())))

        # Load random forests (TODO: should this be config specifiable?)
        actor.infr.print('loading published models')
        try:
            actor.infr.load_published()
        except Exception:
            pass

        # Start actor.infr Main Loop
        actor.infr.print('start id review')
        actor.infr.start_id_review()
        return 'initialized'

    def continue_review(actor):
        # This will signal on_request_review with the same data
        user_request = actor.infr.continue_review()
        return user_request

    def add_feedback(actor, **feedback):
        response = actor.infr.accept(feedback)
        return response

    def remove_annots(actor, aids, **kwargs):
        print('Removing aids=%r from AnnotInference' % (aids,))
        response = actor.infr.remove_aids(aids)
        print('\t got response = %r' % (response,))
        print('Applying NonDynamic Update to AnnotInference')
        actor.infr.apply_nondynamic_update()
        print('\t ...applied')
        return 'removed'

    def update_task_thresh(actor, task, decision, value, **kwargs):
        print('Updating actor.infr.task_thresh with %r %r %r' % (task, decision, value,))
        actor.infr.task_thresh[task][decision] = value
        print('Updated actor.infr.task_thresh = %r' % (actor.infr.task_thresh,))
        return 'updated'

    def add_annots(actor, aids, **kwargs):
        actor.infr.add_annots(aids)
        return 'added'

    def get_infr_status(actor):
        infr_status = {}
        try:
            infr_status['phase'] = actor.infr.phase
        except Exception:
            pass
        try:
            infr_status['loop_phase'] = actor.infr.loop_phase
        except Exception:
            pass
        try:
            infr_status['is_inconsistent'] = len(actor.infr.nid_to_errors) > 0
        except Exception:
            pass
        try:
            infr_status['is_converged'] = actor.infr.phase == 4
        except Exception:
            pass
        try:
            infr_status['num_meaningful'] = actor.infr.refresh.num_meaningful
        except Exception:
            pass
        try:
            infr_status['num_pccs'] = len(actor.infr.queue)
        except Exception:
            pass
        try:
            infr_status['num_inconsistent_ccs'] = len(actor.infr.nid_to_errors)
        except Exception:
            pass
        try:
            infr_status['cc_status'] = actor.infr.connected_component_status()
        except Exception:
            pass

        return infr_status

    def get_feat_extractor(actor):
        if actor.infr.verifiers is None:
            actor.infr.verifiers = {}
        match_state_verifier = actor.infr.verifiers.get('match_state', None)
        if match_state_verifier is not None:
            return match_state_verifier.extr


@ut.reloadable_class
class GraphClient(object):
    """
    CommandLine:
        python -m wbia.web.graph_server GraphClient

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.web.graph_server import *
        >>> import wbia
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
    #     >>> from wbia.web.graph_server import *
    #     >>> import wbia
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

        # Hack around the double review problem
        client.prev_vip = None

        # Save status of the client (the status of the futures)
        client.status = 'Initialized'
        client.infr_status = None
        client.exception = None

        client.aids = None
        client.imagesets = None
        client.config = None
        client.extr = None

        if autoinit:
            client.initialize()

    def initialize(client):
        client.executor = GraphActor.executor()

    def __del__(client):
        client.shutdown()

    def shutdown(client):
        for action, future in client.futures:
            future.cancel()
        client.futures = []
        client.status = 'Shutdown'
        if client.executor is not None:
            client.executor.shutdown(wait=True)
            client.executor = None

    def post(client, payload):
        if not isinstance(payload, dict) or 'action' not in payload:
            raise ValueError('payload must be a dict with an action')
        future = client.executor.post(payload)
        client.futures.append((payload['action'], future))

        # Update graph_client infr_status for all external calls
        payload_ = {
            'action': 'get_infr_status',
        }
        future_ = client.executor.post(payload_)
        client.futures.append((payload_['action'], future_))

        return future

    def cleanup(client):
        # remove done items from our list
        latest_infr_status = None
        new_futures = []
        for action, future in client.futures:
            exception = None
            if future.done():
                try:
                    if action == 'get_infr_status':
                        latest_infr_status = future.result()
                    exception = future.exception()
                except concurrent.futures.CancelledError:
                    pass
                if exception is not None:
                    new_futures.append((action, future))
            else:
                if future.running():
                    new_futures.append((action, future))
                elif action == 'continue_review':
                    future.cancel()
                elif action == 'latest_logs':
                    future.cancel()
                else:
                    new_futures.append((action, future))
        client.futures = new_futures
        return latest_infr_status

    def refresh_status(client):
        latest_infr_status = client.cleanup()
        if latest_infr_status is not None:
            client.infr_status = latest_infr_status

        num_futures = len(client.futures)
        if client.review_dict is None:
            client.status = 'Finished'
        elif num_futures == 0:
            client.status = 'Waiting (Empty Queue)'
        else:
            action, future = client.futures[0]
            exception = None
            if future.done():
                try:
                    exception = future.exception()
                except concurrent.futures.CancelledError:
                    pass
            if exception is None:
                status = 'Working'
                client.exception = None
            else:
                status = 'Exception'
                client.exception = exception
            client.status = '%s (%d in Futures Queue)' % (status, num_futures,)
        return client.status, client.exception

    def add_annots(client):
        raise NotImplementedError('not done yet')

    def update(client, data_list):
        client.review_vip = None

        if data_list is None:
            print('GRAPH CLIENT GOT NONE UPDATE')
            client.review_dict = None
        else:
            data_list = list(data_list)
            num_samples = 5
            num_items = len(data_list)
            num_samples = min(num_samples, num_items)
            first = list(data_list[:num_samples])

            print('UPDATING GRAPH CLIENT WITH {} ITEM(S):'.format(num_items))
            print('First few are: ' + ut.repr4(first, si=2, precision=4))
            client.review_dict = {}

            for (edge, priority, edge_data_dict) in data_list:
                aid1, aid2 = edge
                if aid2 < aid1:
                    aid1, aid2 = aid2, aid1
                edge = (
                    aid1,
                    aid2,
                )
                if client.review_vip is None:
                    # Hack around the double review problem
                    if edge != client.prev_vip:
                        client.review_vip = edge
                client.review_dict[edge] = (
                    priority,
                    edge_data_dict,
                )

    def check(client, edge):
        if edge not in client.review_dict:
            return None
        priority, data_dict = client.review_dict[edge]
        return edge, priority, data_dict

    def sample(client, previous_edge_list=[], max_previous_edges=10):
        if client.review_dict is None:
            raise controller_inject.WebReviewFinishedException(client.graph_uuid)
        print('SAMPLING')
        edge_list = list(client.review_dict.keys())
        if len(edge_list) == 0:
            return None

        edge = None
        if client.review_vip is not None and client.review_vip in edge_list:
            if len(edge_list) >= max_previous_edges:
                vip_1 = int(client.review_vip[0])
                vip_2 = int(client.review_vip[1])

                found = False
                for edge_1, edge_2 in previous_edge_list:
                    if edge_1 == vip_1 and edge_2 == vip_2:
                        found = True
                        break

                if not found:
                    print('SHOWING VIP TO USER!!!')
                    edge = client.review_vip
                    client.prev_vip = edge
                    client.review_vip = None
                else:
                    print(
                        'VIP ALREADY SHOWN TO THIS USER!!! (PROBABLY A RACE CONDITION, SAMPLE RANDOMLY INSTEAD)'
                    )
            else:
                print('GETTING TOO LOW FOR VIP RACE CONDITION CHECK!!!')

        if edge is None:
            print('VIP ALREADY SHOWN!!!')
            edge = random.choice(edge_list)

        priority, data_dict = client.review_dict[edge]
        print('SAMPLED edge = {!r}'.format(edge))
        return edge, priority, data_dict


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.web.graph_server
        python -m wbia.web.graph_server --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
