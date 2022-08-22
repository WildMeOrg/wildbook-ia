# -*- coding: utf-8 -*-
import concurrent
import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import utool as ut

from wbia.control import controller_inject

logger = logging.getLogger('wbia')


class ProcessActorExecutor(ProcessPoolExecutor):
    def __init__(self, actor_class, *args, **kwargs):
        """Initializes a new ThreadPoolExecutor instance."""
        super(ProcessActorExecutor, self).__init__(*args, **kwargs)
        self.actor_instance = actor_class()

    def post(self, payload):
        return self.submit(self.actor_instance.handle, payload)


class ThreadedActorExecutor(ThreadPoolExecutor):
    def __init__(self, actor_class, *args, **kwargs):
        """Initializes a new ThreadPoolExecutor instance."""
        super(ThreadedActorExecutor, self).__init__(*args, **kwargs)
        self.actor_instance = actor_class()

    def post(self, payload):
        return self.submit(self.actor_instance.handle, payload)


class Actor(object):
    @classmethod
    def executor(cls):
        """
        Creates an asychronous instance of this Actor and returns the executor
        to manage it.
        """
        raise NotImplementedError('use ProcessActor or ThreadActor')

    def handle(self, message):
        """
        This method recieves, handles, and responds to the messages sent from
        the executor. This function can return arbitrary values. These values
        can be accessed from the main thread using the Future object returned
        when the message was posted to this actor by the executor.
        """
        raise NotImplementedError('must implement message handler')


class ProcessActor(Actor):
    @classmethod
    def executor(cls, *args, **kwargs):
        # assert 'mp_context' not in kwargs
        # kwargs['mp_context'] = multiprocessing.get_context('spawn')
        return ProcessActorExecutor(cls, *args, **kwargs)


class ThreadActor(Actor):
    @classmethod
    def executor(cls, *args, **kwargs):
        return ThreadedActorExecutor(cls, *args, **kwargs)


def double_review_test():
    # from wbia.web.graph_server import *
    import wbia

    actor = GraphAlgorithmActor()
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
    logger.info('start_resp = {!r}'.format(start_resp))
    infr = actor.infr

    infr.verbose = 100

    user_resp = infr.resume()
    edge, p, d = user_resp[0]
    logger.info('edge = {!r}'.format(edge))

    last = None

    while True:
        infr.add_feedback(edge, infr.edge_decision(edge))
        user_resp = infr.resume()
        edge, p, d = user_resp[0]
        logger.info('edge = {!r}'.format(edge))
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


def _testdata_feedback_payload(edge, decision):
    payload = {
        'action': 'feedback',
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


def _test_foo(future):
    logger.info('FOO {!r}'.format(future))


# GRAPH_ACTOR_CLASS = ProcessActor if ut.LINUX or ut.WIN32 else ThreadActor
GRAPH_ACTOR_CLASS = ThreadActor


class GraphActor(GRAPH_ACTOR_CLASS):
    def __init__(actor, *args, **kwargs):
        super(GraphActor, actor).__init__(*args, **kwargs)

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
        else:
            func = getattr(actor, action, None)
            if func is None:
                raise ValueError('Unknown action={!r}'.format(action))
            else:
                try:
                    return func(**message)
                except Exception as ex:
                    import traceback

                    traceback.print_exc()
                    trace = traceback.format_exc()

                    if actor.infr is not None:
                        actor.infr.print('Actor Server Error: {!r}'.format(ex))
                        actor.infr.print('Actor Server Traceback: {!r}'.format(trace))
                    else:
                        logger.info(ex)
                        logger.info(trace)
                    raise

    def start(actor, dbdir, aids='all', config={}, **kwargs):
        raise NotImplementedError()

    def resume(actor):
        raise NotImplementedError()

    def feedback(actor, **feedback):
        raise NotImplementedError()

    def add_aids(actor, aids, **kwargs):
        raise NotImplementedError()

    def remove_aids(actor, aids, **kwargs):
        raise NotImplementedError()

    def logs(actor):
        raise NotImplementedError()

    def status(actor):
        raise NotImplementedError()

    def metadata(actor):
        raise NotImplementedError()


@ut.reloadable_class
class GraphClient(object):

    actor_cls = GraphActor

    def __init__(
        client,
        aids,
        actor_config={},
        imagesets=None,
        graph_uuid=None,
        callbacks={},
        autoinit=False,
    ):
        client.aids = aids
        client.imagesets = imagesets
        client.actor_config = actor_config
        client.metadata = {}

        client.graph_uuid = graph_uuid
        client.callbacks = callbacks
        client.executor = None

        client.review_dict = {}
        client.previous_review_vip = None
        client.review_vip = None
        client.futures = []

        # Save status of the client (the status of the futures)
        client.status = 'Initialized'
        client.actor_status = None
        client.exception = None

        if autoinit:
            client.initialize()

    def initialize(client):
        logger.info(
            'GraphClient %r using backend GraphActor = %r'
            % (
                client,
                client.actor_cls,
            )
        )
        client.executor = client.actor_cls.executor(max_workers=1)

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

        # Update graph_client actor status for all external calls
        payload_ = {
            'action': 'status',
        }
        future_ = client.executor.post(payload_)
        client.futures.append((payload_['action'], future_))

        return future

    def cleanup(client):
        logger.info('GraphClient.cleanup')
        # remove done items from our list
        logger.info('Current Futures: {!r}'.format(client.futures))
        latest_actor_status = None
        new_futures = []
        for action, future in client.futures:
            exception = None
            if future.done():
                try:
                    if action == 'status':
                        latest_actor_status = future.result()
                    exception = future.exception()
                except concurrent.futures.CancelledError:
                    pass

                if exception is not None:
                    exception_str = str(exception)
                    # Skip any errors that arise from database integrity errors
                    logger.warning('Found exception future: {}'.format(exception_str))
                    logger.warning('\taction: {!r}'.format(action))
                    logger.warning('\tfuture: {!r}'.format(future))

                    if 'sqlite3.IntegrityError' in exception_str:
                        pass
                    else:
                        new_futures.append((action, future))
            else:
                if future.running():
                    new_futures.append((action, future))
                elif action in ['resume', 'logs']:
                    future.cancel()
                else:
                    new_futures.append((action, future))

        client.futures = new_futures
        logger.info('New Futures: {!r}'.format(client.futures))
        return latest_actor_status

    def refresh_status(client):
        latest_actor_status = client.cleanup()
        if latest_actor_status is not None:
            client.actor_status = latest_actor_status

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
            client.status = '%s (%d in Futures Queue)' % (status, num_futures)
        return client.status, client.exception

    def refresh_metadata(client):
        payload = {
            'action': 'metadata',
        }
        future = client.post(payload)
        client.metadata = future.result()

    def add_aids(client):
        raise NotImplementedError('not done yet')

    def update(client, data_list):
        client.review_vip = None

        if data_list is None:
            logger.info('GRAPH CLIENT GOT NONE UPDATE, EMPTY QUEUE')
            client.review_dict = {}
        elif isinstance(data_list, str):
            logger.info('GRAPH CLIENT GOT FINISHED UPDATE')
            client.review_dict = None
            client.refresh_status()
            assert client.status == 'Finished'
            assert 'finished' in data_list
            client.status = '{} ({})'.format(
                client.status,
                data_list,
            )
            return True
        else:
            data_list = list(data_list)
            num_samples = 5
            num_items = len(data_list)
            num_samples = min(num_samples, num_items)
            first = list(data_list[:num_samples])

            logger.info('UPDATING GRAPH CLIENT WITH {} ITEM(S):'.format(num_items))
            logger.info('First few are: ' + ut.repr4(first, si=2, precision=4))
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
                    if edge != client.previous_review_vip:
                        client.review_vip = edge
                client.review_dict[edge] = (
                    priority,
                    edge_data_dict,
                )

        return False

    def check(client, edge):
        if edge not in client.review_dict:
            return None
        priority, data_dict = client.review_dict[edge]
        return edge, priority, data_dict

    def sample(client, previous_edge_list=[], max_previous_edges=10):
        if client.review_dict is None:
            raise controller_inject.WebReviewFinishedException(client.graph_uuid)
        logger.info('SAMPLING')
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
                    logger.info('SHOWING VIP TO USER!!!')
                    edge = client.review_vip
                    client.previous_review_vip = edge
                    client.review_vip = None
                else:
                    logger.info(
                        'VIP ALREADY SHOWN TO THIS USER!!! (PROBABLY A RACE CONDITION, SAMPLE RANDOMLY INSTEAD)'
                    )
            else:
                logger.info('GETTING TOO LOW FOR VIP RACE CONDITION CHECK!!!')

        if edge is None:
            logger.info('VIP ALREADY SHOWN!!!')
            edge = random.choice(edge_list)

        priority, data_dict = client.review_dict[edge]
        logger.info('SAMPLED edge = {!r}'.format(edge))
        return edge, priority, data_dict

    def sync(self, ibs):
        raise NotImplementedError()


class GraphAlgorithmActor(GraphActor):
    """

    CommandLine:
        python -m wbia.web.graph_server GraphAlgorithmActor

    Doctest:
        >>> from wbia.web.graph_server import *
        >>> actor = GraphAlgorithmActor()
        >>> payload = testdata_start_payload()
        >>> # Start the process
        >>> start_resp = actor.handle(payload)
        >>> print('start_resp = {!r}'.format(start_resp))
        >>> # Respond with a user decision
        >>> user_request = actor.handle({'action': 'resume'})
        >>> # Wait for a response and  the GraphAlgorithmActor in another proc
        >>> edge, priority, edge_data = user_request[0]
        >>> user_resp_payload = _testdata_feedback_payload(edge, 'match')
        >>> content = actor.handle(user_resp_payload)
        >>> actor.infr.dump_logs()


    Doctest:
        >>> # xdoctest: +REQUIRES(module:wbia_cnn, --slow)
        >>> from wbia.web.graph_server import *
        >>> import wbia
        >>> actor = GraphAlgorithmActor()
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
        >>> user_request = actor.handle({'action': 'resume'})
        >>> print('user_request = {!r}'.format(user_request))
        >>> # Wait for a response and  the GraphAlgorithmActor in another proc
        >>> edge, priority, edge_data = user_request[0]
        >>> user_resp_payload = _testdata_feedback_payload(edge, 'match')
        >>> content = actor.handle(user_resp_payload)
        >>> actor.infr.dump_logs()
        >>> actor.infr.status()
    """

    def __init__(actor, *args, **kwargs):
        super(GraphAlgorithmActor, actor).__init__(*args, **kwargs)
        actor.infr = None
        actor.graph_uuid = None

    def start(actor, dbdir, aids='all', config={}, graph_uuid=None, **kwargs):
        import wbia

        assert dbdir is not None, 'must specify dbdir'
        assert actor.infr is None, 'AnnotInference already running'
        ibs = wbia.opendb(dbdir=dbdir, use_cache=False, web=False, force_serial=True)

        # Create the AnnotInference
        logger.info('starting via actor with ibs = {!r}'.format(ibs))
        actor.infr = wbia.AnnotInference(ibs=ibs, aids=aids, autoinit=True)
        actor.graph_uuid = graph_uuid

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
        actor.infr.load_published()

        # Start actor.infr Main Loop
        actor.infr.print('start id review')
        actor.infr.start_id_review()
        return 'initialized'

    def resume(actor):
        # This will signal on_request_review with the same data
        user_request = actor.infr.resume()
        return user_request

    def feedback(actor, **feedback):
        response = actor.infr.accept(feedback)
        return response

    def add_aids(actor, aids, **kwargs):
        actor.infr.add_aids(aids)
        return 'added'

    def remove_aids(actor, aids, **kwargs):
        logger.info('Removing aids={!r} from AnnotInference'.format(aids))
        response = actor.infr.remove_aids(aids)
        logger.info('\t got response = {!r}'.format(response))
        logger.info('Applying NonDynamic Update to AnnotInference')
        actor.infr.apply_nondynamic_update()
        logger.info('\t ...applied')
        return 'removed'

    def logs(actor):
        return actor.infr.latest_logs(colored=True)

    def status(actor):
        actor_status = {}
        try:
            actor_status['phase'] = actor.infr.phase
        except Exception:
            pass
        try:
            actor_status['loop_phase'] = actor.infr.loop_phase
        except Exception:
            pass
        try:
            actor_status['is_inconsistent'] = len(actor.infr.nid_to_errors) > 0
        except Exception:
            pass
        try:
            actor_status['is_converged'] = actor.infr.phase == 4
        except Exception:
            pass
        try:
            actor_status['num_meaningful'] = actor.infr.refresh.num_meaningful
        except Exception:
            pass
        try:
            actor_status['num_pccs'] = len(actor.infr.queue)
        except Exception:
            pass
        try:
            actor_status['num_inconsistent_ccs'] = len(actor.infr.nid_to_errors)
        except Exception:
            pass
        try:
            actor_status['cc_status'] = actor.infr.connected_component_status()
        except Exception:
            pass

        return actor_status

    def metadata(actor):
        if actor.infr.verifiers is None:
            actor.infr.verifiers = {}
        verifier = actor.infr.verifiers.get('match_state', None)
        extr = None if verifier is None else verifier.extr
        metadata = {
            'extr': extr,
        }
        return metadata


class GraphAlgorithmClient(GraphClient):
    """
    CommandLine:
        python -m wbia.web.graph_server GraphAlgorithmClient

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.web.graph_server import *
        >>> import wbia
        >>> client = GraphAlgorithmClient(aids='all', autoinit=True)
        >>> # Start the GraphAlgorithmActor in another proc
        >>> payload = testdata_start_payload()
        >>> client.post(payload).result()
        >>> future = client.post({'action': 'resume'})
        >>> future.add_done_callback(_test_foo)
        >>> user_request = future.result()
        >>> # Wait for a response and  the GraphAlgorithmActor in another proc
        >>> edge, priority, edge_data = user_request[0]
        >>> user_resp_payload = _testdata_feedback_payload(edge, 'match')
        >>> future = client.post(user_resp_payload)
        >>> future.result()
        >>> # Debug by getting the actor over a mp.Pipe
        >>> future = client.post({'action': 'debug'})
        >>> actor = future.result()
        >>> actor.infr.dump_logs()
        >>> #print(client.post({'action': 'logs'}).result())

    # Ignore:
    #     >>> from wbia.web.graph_server import *
    #     >>> import wbia
    #     >>> client = GraphAlgorithmClient(autoinit=True)
    #     >>> # Start the GraphAlgorithmActor in another proc
    #     >>> client.post(testdata_start_payload(list(range(1, 10)))).result()
    #     >>> #
    #     >>> future = client.post({'action': 'resume'})
    #     >>> user_request = future.result()
    #     >>> # The infr algorithm needs a review
    #     >>> edge, priority, edge_data = user_request[0]
    #     >>> #
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'resume'})
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'resume'})
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'wait', 'num': float(30)})
    #     >>> client.post({'action': 'resume'})
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'resume'})
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'resume'})
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'resume'})
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'resume'})
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'resume'})
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'resume'})
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'resume'})
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'resume'})
    #     >>> client.post(_testdata_feedback_payload(edge, 'match'))
    #     >>> client.post({'action': 'resume'})
    """

    actor_cls = GraphAlgorithmActor

    def sync(self, ibs):
        import wbia

        # Create the AnnotInference
        infr = wbia.AnnotInference(ibs=ibs, aids=self.aids, autoinit=True)
        for key in self.actor_config:
            infr.params[key] = self.actor_config[key]
        infr.reset_feedback('staging', apply=True)

        infr.relabel_using_reviews(rectify=True)
        edge_delta_df = infr.match_state_delta(old='annotmatch', new='all')
        name_delta_df = infr.get_wbia_name_delta()

        ############################################################################

        col_list = list(edge_delta_df.columns)
        match_aid_edge_list = list(edge_delta_df.index)
        match_aid1_list = ut.take_column(match_aid_edge_list, 0)
        match_aid2_list = ut.take_column(match_aid_edge_list, 1)
        match_annot_uuid1_list = ibs.get_annot_uuids(match_aid1_list)
        match_annot_uuid2_list = ibs.get_annot_uuids(match_aid2_list)
        match_annot_uuid_edge_list = list(
            zip(match_annot_uuid1_list, match_annot_uuid2_list)
        )

        zipped = list(zip(*(list(edge_delta_df[col]) for col in col_list)))

        match_list = []
        for match_annot_uuid_edge, zipped_ in list(
            zip(match_annot_uuid_edge_list, zipped)
        ):
            match_dict = {
                'edge': match_annot_uuid_edge,
            }
            for index, col in enumerate(col_list):
                match_dict[col] = zipped_[index]
            match_list.append(match_dict)

        ############################################################################

        col_list = list(name_delta_df.columns)
        name_aid_list = list(name_delta_df.index)
        name_annot_uuid_list = ibs.get_annot_uuids(name_aid_list)
        old_name_list = list(name_delta_df['old_name'])
        new_name_list = list(name_delta_df['new_name'])
        zipped = list(zip(name_annot_uuid_list, old_name_list, new_name_list))
        name_dict = {
            str(name_annot_uuid): {'old': old_name, 'new': new_name}
            for name_annot_uuid, old_name, new_name in zipped
        }

        ############################################################################

        ret_dict = {
            'match_list': match_list,
            'name_dict': name_dict,
        }

        infr.write_wbia_staging_feedback()
        infr.write_wbia_annotmatch_feedback(edge_delta_df)
        infr.write_wbia_name_assignment(name_delta_df)
        edge_delta_df.reset_index()

        return ret_dict
