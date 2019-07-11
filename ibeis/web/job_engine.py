# -*- coding: utf-8 -*-
"""
Accepts and handles requests for tasks.

Each of the following runs in its own Thread/Process.

BASICALLY DO A CLIENT/SERVER TO SPAWN PROCESSES
AND THEN A PUBLISH SUBSCRIBE TO RETURN DATA

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

Notes:
    We are essentially goint to be spawning two processes.
    We can test these simultaniously using

        python -m ibeis.web.job_engine job_engine_tester

    We can test these separately by first starting the background server
        python -m ibeis.web.job_engine job_engine_tester --bg

        Alternative:
            python -m ibeis.web.job_engine job_engine_tester --bg --no-engine
            python -m ibeis.web.job_engine job_engine_tester --bg --only-engine --fg-engine

    And then running the forground process
        python -m ibeis.web.job_engine job_engine_tester --fg
"""
from __future__ import absolute_import, division, print_function, unicode_literals
#if False:
#    import os
#    os.environ['UTOOL_NOCNN'] = 'True'
import utool as ut
import time
import zmq
import uuid  # NOQA
import numpy as np
import shelve
import random
from datetime import datetime
import pytz
from os.path import join, exists, abspath, splitext, basename
from functools import partial
from ibeis.control import controller_inject
print, rrr, profile = ut.inject2(__name__)


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_api   = controller_inject.get_ibeis_flask_api(__name__)

ctx = zmq.Context.instance()

# FIXME: needs to use correct number of ports
URL = 'tcp://127.0.0.1'
NUM_ENGINES = 1
VERBOSE_JOBS = ut.get_argflag('--bg') or ut.get_argflag('--fg') or ut.get_argflag('--verbose-jobs')


TIMESTAMP_FMTSTR = '%Y-%m-%d %H:%M:%S %Z'
TIMESTAMP_TIMEZONE = 'US/Pacific'


def update_proctitle(procname, dbname=None):
    try:
        import setproctitle
        print('CHANGING PROCESS TITLE')
        old_title = setproctitle.getproctitle()
        print('old_title = %r' % (old_title,))
        # new_title = 'IBEIS_' + procname + ' ' + old_title
        # new_title = procname + ' ' + old_title
        # new_title = 'ibeis_zmq_loop'
        hostname = ut.get_computer_name()
        new_title = 'IBEIS_%s_%s_%s' % (dbname, hostname, procname, )
        print('new_title = %r' % (new_title,))
        setproctitle.setproctitle(new_title)
    except ImportError:
        print('pip install setproctitle')


def _get_engine_job_paths(ibs):
    shelve_path = ibs.get_shelves_path()
    ut.ensuredir(shelve_path)
    record_filepath_list = list(ut.iglob(join(shelve_path, '*.pkl')))
    return record_filepath_list


@register_ibs_method
def initialize_job_manager(ibs):
    """
    Starts a background zmq job engine. Initializes a zmq object in this thread
    that can talk to the background processes.

    Run from the webserver

    CommandLine:
        python -m ibeis.web.job_engine --exec-initialize_job_manager:0

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.web.job_engine import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> from ibeis.web import apis_engine
        >>> from ibeis.web import job_engine
        >>> ibs.load_plugin_module(job_engine)
        >>> ibs.load_plugin_module(apis_engine)
        >>> ibs.initialize_job_manager()
        >>> print('Initializqation success. Now closing')
        >>> ibs.close_job_manager()
        >>> print('Closing success.')

    Example:
        >>> # WEB_DOCTEST
        >>> from ibeis.web.job_engine import *  # NOQA
        >>> import ibeis
        >>> import requests
        >>> web_instance = ibeis.opendb_bg_web(db='testdb1')
        >>> web_port = ibs.get_web_port_via_scan()
        >>> if web_port is None:
        >>>     raise ValueError('IA web server is not running on any expected port')
        >>> baseurl = 'http://127.0.1.1:%s' % (web_port, )
        >>> _payload = {'image_attrs_list': [], 'annot_attrs_list': []}
        >>> payload = ut.map_dict_vals(ut.to_json, _payload)
        >>> resp1 = requests.post(baseurl + '/api/test/helloworld/?f=b', data=payload)
        >>> #resp2 = requests.post(baseurl + '/api/image/json/', data=payload)
        >>> #print(resp2)
        >>> web_instance.terminate()
        >>> #json_dict = resp2.json()
        >>> #text = json_dict['response']
        >>> #print(text)
    """
    ibs.job_manager = ut.DynStruct()

    use_static_ports = False

    if ut.get_argflag('--web-deterministic-ports'):
        use_static_ports = True

    if ut.get_argflag('--fg'):
        ibs.job_manager.reciever = JobBackend(use_static_ports=True)
    else:
        record_filepath_list = _get_engine_job_paths(ibs)
        job_counter = len(record_filepath_list) + 1
        ibs.job_manager.reciever = JobBackend(use_static_ports=use_static_ports)
        ibs.job_manager.reciever.initialize_background_processes(
            dbdir=ibs.get_dbdir(),
            containerized=ibs.containerized,
            job_counter=job_counter,
        )

    ibs.job_manager.jobiface = JobInterface(0, ibs.job_manager.reciever.port_dict, ibs=ibs)
    ibs.job_manager.jobiface.initialize_client_thread()
    # Wait until the collector becomes live
    while 0 and True:
        result = ibs.get_job_status(-1)
        print('result = %r' % (result,))
        if result['status'] == 'ok':
            break
    ibs.job_manager.jobiface.queue_interrupted_jobs()

    # import ibeis
    # #dbdir = '/media/raid/work/testdb1'
    # from ibeis.web import app
    # web_port = ibs.get_web_port_via_scan()
    # if web_port is None:
    #     raise ValueError('IA web server is not running on any expected port')
    # proc = ut.spawn_background_process(app.start_from_ibeis, ibs, port=web_port)


@register_ibs_method
def close_job_manager(ibs):
    # if hasattr(ibs, 'job_manager') and ibs.job_manager is not None:
    #     pass
    del ibs.job_manager.reciever
    del ibs.job_manager.jobiface
    ibs.job_manager = None


@register_ibs_method
@register_api('/api/engine/job/', methods=['GET', 'POST'], __api_plural_check__=False)
def get_job_id_list(ibs):
    """
    Web call that returns the list of job ids

    CommandLine:
        # Run Everything together
        python -m ibeis.web.job_engine --exec-get_job_status

        # Start job queue in its own process
        python -m ibeis.web.job_engine job_engine_tester --bg
        # Start web server in its own process
        ./main.py --web --fg
        pass
        # Run foreground process
        python -m ibeis.web.job_engine --exec-get_job_status:0 --fg

    Example:
        >>> # WEB_DOCTEST
        >>> from ibeis.web.job_engine import *  # NOQA
        >>> import ibeis
        >>> web_ibs = ibeis.opendb_bg_web('testdb1')  # , domain='http://52.33.105.88')
        >>> # Test get status of a job id that does not exist
        >>> response = web_ibs.send_ibeis_request('/api/engine/job/', jobid='badjob')
        >>> web_ibs.terminate2()

    """
    status = ibs.job_manager.jobiface.get_job_id_list()
    jobid_list = status['jobid_list']
    return jobid_list


@register_ibs_method
@register_api('/api/engine/job/status/', methods=['GET', 'POST'], __api_plural_check__=False)
def get_job_status(ibs, jobid=None):
    """
    Web call that returns the status of a job

    Returns one of:
        received              - job has been received, but not ingested yet
        accepted              - job has been accepted (validated)
        queued                - job has been transferred to the engine queue
        working               - job is being worked on by the engine
        publishing            - job is done on the engine, pushing results to collector
        completed | exception - job is complete or has an error

    CommandLine:
        # Run Everything together
        python -m ibeis.web.job_engine --exec-get_job_status

        # Start job queue in its own process
        python -m ibeis.web.job_engine job_engine_tester --bg
        # Start web server in its own process
        ./main.py --web --fg
        pass
        # Run foreground process
        python -m ibeis.web.job_engine --exec-get_job_status:0 --fg

    Example:
        >>> # WEB_DOCTEST
        >>> from ibeis.web.job_engine import *  # NOQA
        >>> import ibeis
        >>> web_ibs = ibeis.opendb_bg_web('testdb1')  # , domain='http://52.33.105.88')
        >>> # Test get status of a job id that does not exist
        >>> response = web_ibs.send_ibeis_request('/api/engine/job/status/', jobid='badjob')
        >>> web_ibs.terminate2()

    """
    if jobid is None:
        status = ibs.job_manager.jobiface.get_job_status_dict()
    else:
        status = ibs.job_manager.jobiface.get_job_status(jobid)
    return status


@register_ibs_method
@register_api('/api/engine/job/metadata/', methods=['GET', 'POST'], __api_plural_check__=False)
def get_job_metadata(ibs, jobid):
    """
    Web call that returns the metadata of a job

    CommandLine:
        # Run Everything together
        python -m ibeis.web.job_engine --exec-get_job_metadata

        # Start job queue in its own process
        python -m ibeis.web.job_engine job_engine_tester --bg
        # Start web server in its own process
        ./main.py --web --fg
        pass
        # Run foreground process
        python -m ibeis.web.job_engine --exec-get_job_metadata:0 --fg

    Example:
        >>> # WEB_DOCTEST
        >>> from ibeis.web.job_engine import *  # NOQA
        >>> import ibeis
        >>> web_ibs = ibeis.opendb_bg_web('testdb1')  # , domain='http://52.33.105.88')
        >>> # Test get metadata of a job id that does not exist
        >>> response = web_ibs.send_ibeis_request('/api/engine/job/metadata/', jobid='badjob')
        >>> web_ibs.terminate2()

    """
    status = ibs.job_manager.jobiface.get_job_metadata(jobid)
    return status


@register_ibs_method
@register_api('/api/engine/job/result/', methods=['GET', 'POST'])
def get_job_result(ibs, jobid):
    """
    Web call that returns the result of a job
    """
    result = ibs.job_manager.jobiface.get_job_result(jobid)
    return result


@register_ibs_method
@register_api('/api/engine/job/result/wait/', methods=['GET', 'POST'])
def wait_for_job_result(ibs, jobid, timeout=10, freq=.1):
    ibs.job_manager.jobiface.wait_for_job_result(jobid, timeout=timeout, freq=freq)
    result = ibs.job_manager.jobiface.get_unpacked_result(jobid)
    return result


def _get_random_open_port():
    port = random.randint(1024, 49151)
    while not ut.is_local_port_open(port):
        port = random.randint(1024, 49151)
    assert ut.is_local_port_open(port)
    return port


def job_engine_tester():
    """
    CommandLine:
        python -m ibeis.web.job_engine --exec-job_engine_tester
        python -b -m ibeis.web.job_engine --exec-job_engine_tester

        python -m ibeis.web.job_engine job_engine_tester
        python -m ibeis.web.job_engine job_engine_tester --bg
        python -m ibeis.web.job_engine job_engine_tester --fg

    Example:
        >>> # SCRIPT
        >>> from ibeis.web.job_engine import *  # NOQA
        >>> job_engine_tester()
    """
    _init_signals()
    # now start a few clients, and fire off some requests
    client_id = np.random.randint(1000)
    reciever = JobBackend(use_static_ports=True)
    jobiface = JobInterface(client_id, reciever.port_dict)
    from ibeis.init import sysres
    if ut.get_argflag('--bg'):
        dbdir = sysres.get_args_dbdir(defaultdb='cache', allow_newdir=False,
                                      db=None, dbdir=None)
        reciever.initialize_background_processes(dbdir)
        print('[testzmq] parent process is looping forever')
        while True:
            time.sleep(1)
    elif ut.get_argflag('--fg'):
        jobiface.initialize_client_thread()
    else:
        dbdir = sysres.get_args_dbdir(defaultdb='cache', allow_newdir=False,
                                      db=None, dbdir=None)
        reciever.initialize_background_processes(dbdir)
        jobiface.initialize_client_thread()

    # Foreground test script
    print('... waiting for jobs')
    if ut.get_argflag('--cmd'):
        ut.embed()
        #jobiface.queue_job()
    else:
        print('[test] ... emit test1')
        callback_url = None
        callback_method = None
        args = (1,)
        jobid1 = jobiface.queue_job('helloworld', callback_url,
                                    callback_method, *args)
        jobiface.wait_for_job_result(jobid1)
        jobid_list = []

        args = ([1], [3, 4, 5])
        kwargs = dict(cfgdict={'K': 1})
        identify_jobid = jobiface.queue_job('query_chips_simple_dict',
                                            callback_url, callback_method,
                                            *args, **kwargs)
        for jobid in jobid_list:
            jobiface.wait_for_job_result(jobid)

        jobiface.wait_for_job_result(identify_jobid)
    print('FINISHED TEST SCRIPT')


class JobBackend(object):
    def __init__(self, **kwargs):
        #self.num_engines = 3
        self.num_engines = NUM_ENGINES
        self.engine_queue_proc = None
        self.engine_procs = None
        self.collect_queue_proc = None
        self.collect_proc = None
        # --
        only_engine = ut.get_argflag('--only-engine')
        self.spawn_collector = not only_engine
        self.spawn_engine = not ut.get_argflag('--no-engine')
        self.fg_engine = ut.get_argflag('--fg-engine')
        self.spawn_queue = not only_engine
        # Find ports
        self.port_dict = None
        self._initialize_job_ports(**kwargs)
        print('JobBackend ports:')
        ut.print_dict(self.port_dict)

    def __del__(self):
        if VERBOSE_JOBS:
            print('Cleaning up job backend')
        if self.engine_procs is not None:
            for i in self.engine_procs:
                i.terminate()
        if self.engine_queue_proc is not None:
            self.engine_queue_proc.terminate()
        if self.collect_proc is not None:
            self.collect_proc.terminate()
        if self.collect_queue_proc is not None:
            self.collect_queue_proc.terminate()
        if VERBOSE_JOBS:
            print('Killed external procs')

    def _initialize_job_ports(self, use_static_ports=False, static_root=51381):
        # _portgen = functools.partial(six.next, itertools.count(51381))
        key_list = [
            'engine_url1',
            'engine_url2',
            'collect_url1',
            'collect_url2',
            # 'collect_pushpull_url',
        ]
        # Get ports
        if use_static_ports:
            port_list = range(static_root, static_root + len(key_list))
        else:
            port_list = []
            while len(port_list) < len(key_list):
                port = _get_random_open_port()
                if port not in port_list:
                    port_list.append(port)
            port_list = sorted(port_list)
        # Assign ports
        assert len(key_list) == len(port_list)
        self.port_dict = {
            key : '%s:%d' % (URL, port)
            for key, port in list(zip(key_list, port_list))
        }

    def initialize_background_processes(self, dbdir=None, containerized=False,
                                        thread=True, job_counter=0):
        print = partial(ut.colorprint, color='fuchsia')
        #if VERBOSE_JOBS:
        print('Initialize Background Processes')

        def _spawner(func, *args, **kwargs):

            if thread:
                # mp.set_start_method('spawn')
                _spawner_func_ = ut.spawn_background_daemon_thread
            else:
                _spawner_func_ = ut.spawn_background_process

            proc = _spawner_func_(func, *args, **kwargs)
            assert proc.is_alive(), 'proc (%s) died too soon' % (ut.get_funcname(func,))
            return proc

        if self.spawn_queue:
            self.engine_queue_proc = _spawner(engine_queue_loop, self.port_dict, job_counter=job_counter)
            self.collect_queue_proc = _spawner(collect_queue_loop, self.port_dict)

        if self.spawn_collector:
            self.collect_proc = _spawner(collector_loop, self.port_dict, dbdir, containerized)

        if self.spawn_engine:
            if self.fg_engine:
                print('ENGINE IS IN DEBUG FOREGROUND MODE')
                # Spawn engine in foreground process
                assert self.num_engines == 1, 'fg engine only works with one engine'
                engine_loop(0, self.port_dict, dbdir)
                assert False, 'should never see this'
            else:
                # Normal case
                self.engine_procs = [
                    _spawner(engine_loop, i, self.port_dict, dbdir, containerized)
                    for i in range(self.num_engines)
                ]

        # Check if online
        # wait for processes to spin up
        if self.spawn_queue:
            assert self.engine_queue_proc.is_alive(), 'engine died too soon'
            assert self.collect_queue_proc.is_alive(), 'collector queue died too soon'

        if self.spawn_collector:
            assert self.collect_proc.is_alive(), 'collector died too soon'

        if self.spawn_engine:
            for engine in self.engine_procs:
                assert engine.is_alive(), 'engine died too soon'


class JobInterface(object):
    def __init__(jobiface, id_, port_dict, ibs=None):
        jobiface.id_ = id_
        jobiface.ibs = ibs
        jobiface.verbose = 2 if VERBOSE_JOBS else 1
        jobiface.port_dict = port_dict
        print('JobInterface ports:')
        ut.print_dict(jobiface.port_dict)

    # def init(jobiface):
    #     # Starts several new processes
    #     jobiface.initialize_background_processes()
    #     # Does not create a new process, but connects sockets on this process
    #     jobiface.initialize_client_thread()

    def initialize_client_thread(jobiface):
        """
        Creates a ZMQ object in this thread. This talks to background processes.
        """
        print = partial(ut.colorprint, color='blue')
        if jobiface.verbose:
            print('Initializing JobInterface')
        jobiface.engine_deal_sock = ctx.socket(zmq.DEALER)
        jobiface.engine_deal_sock.setsockopt_string(zmq.IDENTITY,
                                                    'client%s.engine.DEALER' %
                                                    (jobiface.id_,))
        jobiface.engine_deal_sock.connect(jobiface.port_dict['engine_url1'])
        if jobiface.verbose:
            print('connect engine_url1 = %r' % (jobiface.port_dict['engine_url1'],))

        jobiface.collect_deal_sock = ctx.socket(zmq.DEALER)
        jobiface.collect_deal_sock.setsockopt_string(zmq.IDENTITY,
                                                     'client%s.collect.DEALER'
                                                     % (jobiface.id_,))
        jobiface.collect_deal_sock.connect(jobiface.port_dict['collect_url1'])
        if jobiface.verbose:
            print('connect collect_url1 = %r' % (jobiface.port_dict['collect_url1'],))

    def queue_interrupted_jobs(jobiface):
        import tqdm

        MAX_ATTEMPTS = 20

        ibs = jobiface.ibs
        if ibs is not None:
            record_filepath_list = _get_engine_job_paths(ibs)
            print('Found %d engine jobs to reload...' % (len(record_filepath_list), ))
            for record_filepath in tqdm.tqdm(record_filepath_list):
                lock_filepath = record_filepath.replace('.pkl', '.lock')
                record = ut.load_cPkl(record_filepath)
                jobid = splitext(basename(record_filepath))[0]

                engine_request = record.get('request',   None)
                attempts       = record.get('attempts',  0)
                completed      = record.get('completed', False)
                suppressed     = attempts >= MAX_ATTEMPTS

                if exists(lock_filepath):
                    ut.delete(lock_filepath)

                assert engine_request is not None
                if completed or suppressed:
                    status = 'completed' if completed else 'suppressed'
                    reply_notify = {
                        'jobid': jobid,
                        'status': status,
                        'action': 'register',
                    }
                    jobiface.collect_deal_sock.send_json(reply_notify)
                    reply = jobiface.collect_deal_sock.recv_json()
                    jobid_ = reply['jobid']
                    assert jobid_ == jobid
                else:
                    with ut.Indenter('[client %d] ' % (jobiface.id_)):
                        color = 'brightblue' if attempts == 0 else 'brightred'
                        print_ = partial(ut.colorprint, color=color)

                        print_('RESTARTING FAILED JOB FROM RESTART (ATTEMPT %d)' % (attempts + 1, ))
                        print_(ut.repr3(record_filepath))
                        print_(ut.repr3(record))

                        engine_request['restart_jobid'] = jobid
                        jobiface.engine_deal_sock.send_json(engine_request)
                        reply = jobiface.engine_deal_sock.recv_json()
                        jobid_ = reply['jobid']
                        assert jobid_ == jobid

                        record['attempts'] = attempts + 1
                        ut.save_cPkl(record_filepath, record)

    def queue_job(jobiface, action, callback_url=None, callback_method=None, *args, **kwargs):
        r"""
        IBEIS:
            This is just a function that lives in the main thread and ships off
            a job.

        FIXME: I do not like having callback_url and callback_method specified
               like this with args and kwargs. If these must be there then
               they should be specified first, or
               THE PREFERED OPTION IS
               args and kwargs should not be specified without the * syntax

        The client - sends messages, and receives replies after they
        have been processed by the
        """
        # NAME: job_client
        with ut.Indenter('[client %d] ' % (jobiface.id_)):
            print = partial(ut.colorprint, color='blue')
            if jobiface.verbose >= 1:
                print('----')

            request = None
            try:
                import flask
                if flask.request:
                    request = {
                        'endpoint': flask.request.path,
                        'function': flask.request.endpoint,
                        'input'   : flask.request.processed,
                    }
            except RuntimeError:
                pass

            engine_request = {
                'action'          : action,
                'args'            : args,
                'kwargs'          : kwargs,
                'callback_url'    : callback_url,
                'callback_method' : callback_method,
                'request'         : request,
                'restart_jobid'   : None,
            }
            if jobiface.verbose >= 2:
                print('Queue job: %s' % (engine_request))

            # Flow of information tags:
            # CALLS: engine_queue
            jobiface.engine_deal_sock.send_json(engine_request)
            if jobiface.verbose >= 3:
                print('..sent, waiting for response')
            # RETURNED FROM: job_client_return
            reply_notify = jobiface.engine_deal_sock.recv_json()
            if jobiface.verbose >= 2:
                print('Got reply: %s' % ( reply_notify))
            jobid = reply_notify['jobid']

            ibs = jobiface.ibs
            if ibs is not None:
                shelve_path = ibs.get_shelves_path()
                ut.ensuredir(shelve_path)
                record_filename = '%s.pkl' % (jobid, )
                record_filepath = join(shelve_path, record_filename)
                record = {
                    'request':   engine_request,
                    'attempts':  0,
                    'completed': False,
                }
                ut.save_cPkl(record_filepath, record)

            # Release memory
            engine_request = None

            return jobid

    def get_job_id_list(jobiface):
        with ut.Indenter('[client %d] ' % (jobiface.id_)):
            print = partial(ut.colorprint, color='teal')
            if jobiface.verbose >= 1:
                print('----')
                print('Request list of job ids')
            pair_msg = dict(action='job_id_list')
            # CALLS: collector_request_status
            jobiface.collect_deal_sock.send_json(pair_msg)
            if jobiface.verbose >= 3:
                print('... waiting for collector reply')
            reply = jobiface.collect_deal_sock.recv_json()
            if jobiface.verbose >= 2:
                print('got reply = %s' % (ut.repr2(reply, truncate=True),))
        return reply

    def get_job_status(jobiface, jobid):
        with ut.Indenter('[client %d] ' % (jobiface.id_)):
            print = partial(ut.colorprint, color='teal')
            if jobiface.verbose >= 1:
                print('----')
                print('Request status of jobid=%r' % (jobid,))
            pair_msg = dict(action='job_status', jobid=jobid)
            # CALLS: collector_request_status
            jobiface.collect_deal_sock.send_json(pair_msg)
            if jobiface.verbose >= 3:
                print('... waiting for collector reply')
            reply = jobiface.collect_deal_sock.recv_json()
            if jobiface.verbose >= 2:
                print('got reply = %s' % (ut.repr2(reply, truncate=True),))
        return reply

    def get_job_status_dict(jobiface):
        with ut.Indenter('[client %d] ' % (jobiface.id_)):
            print = partial(ut.colorprint, color='teal')
            if jobiface.verbose >= 1:
                print('----')
                print('Request list of job ids')
            pair_msg = dict(action='job_status_dict')
            # CALLS: collector_request_status
            jobiface.collect_deal_sock.send_json(pair_msg)
            if jobiface.verbose >= 3:
                print('... waiting for collector reply')
            reply = jobiface.collect_deal_sock.recv_json()
            if jobiface.verbose >= 2:
                print('got reply = %s' % (ut.repr2(reply, truncate=True),))
        return reply

    def get_job_metadata(jobiface, jobid):
        with ut.Indenter('[client %d] ' % (jobiface.id_)):
            print = partial(ut.colorprint, color='teal')
            if jobiface.verbose >= 1:
                print('----')
                print('Request metadata of jobid=%r' % (jobid,))
            pair_msg = dict(action='job_input', jobid=jobid)
            # CALLS: collector_request_metadata
            jobiface.collect_deal_sock.send_json(pair_msg)
            if jobiface.verbose >= 3:
                print('... waiting for collector reply')
            reply = jobiface.collect_deal_sock.recv_json()
            if jobiface.verbose >= 2:
                print('got reply = %s' % (ut.repr2(reply, truncate=True),))
        return reply

    def get_job_result(jobiface, jobid):
        with ut.Indenter('[client %d] ' % (jobiface.id_)):
            if jobiface.verbose >= 1:
                print = partial(ut.colorprint, color='teal')
                print('----')
                print('Request result of jobid=%r' % (jobid,))
            pair_msg = dict(action='job_result', jobid=jobid)
            # CALLER: collector_request_result
            jobiface.collect_deal_sock.send_json(pair_msg)
            if jobiface.verbose >= 3:
                print('... waiting for collector reply')
            reply = jobiface.collect_deal_sock.recv_json()
            if jobiface.verbose >= 2:
                print('got reply = %s' % (ut.repr2(reply, truncate=True),))
        return reply

    def get_unpacked_result(jobiface, jobid):
        reply = jobiface.get_job_result(jobid)
        json_result = reply['json_result']
        try:
            result = ut.from_json(json_result)
        except TypeError as ex:
            ut.printex(ex, keys=['json_result'], iswarning=True)
            result = json_result
        except Exception as ex:
            ut.printex(ex, 'Failed to unpack result', keys=['json_result'])
            result = reply['json_result']
            # raise
            # raise
        #print('Job %r result = %s' % (jobid, ut.repr2(result, truncate=True),))
        # Release raw JSON result
        json_result = None
        return result

    def wait_for_job_result(jobiface, jobid, timeout=10, freq=.1):
        t = ut.Timer(verbose=False)
        t.tic()
        while True:
            reply = jobiface.get_job_status(jobid)
            if reply['jobstatus'] == 'completed':
                return
            elif reply['jobstatus'] == 'exception':
                result = jobiface.get_unpacked_result(jobid)
                #raise Exception(result)
                print('Exception occured in engine')
                return result
            elif reply['jobstatus'] == 'working':
                pass
            elif reply['jobstatus'] == 'unknown':
                pass
            else:
                raise Exception('Unknown jobstatus=%r' % (reply['jobstatus'],))
            reply = None  # Release memory
            time.sleep(freq)
            if timeout is not None and t.toc() > timeout:
                raise Exception('Timeout')


def make_queue_loop(name='collect'):
    """
    Standard queue loop

    Args:
        name (None): (default = None)
    """
    assert name is not None, 'must name queue'
    queue_name = name + '_queue'
    loop_name = queue_name + '_loop'
    def queue_loop(port_dict):
        iface1, iface2 = port_dict['collect_url1'], port_dict['collect_url2']
        print = partial(ut.colorprint, color='green')
        update_proctitle(queue_name)

        with ut.Indenter('[%s] ' % (queue_name,)):
            if VERBOSE_JOBS:
                print('Init make_queue_loop: name=%r' % (name,))
            # bind the client dealer to the queue router
            rout_sock = ctx.socket(zmq.ROUTER)
            rout_sock.setsockopt_string(zmq.IDENTITY, 'queue.' + name + '.' + 'ROUTER')
            rout_sock.bind(iface1)
            if VERBOSE_JOBS:
                print('bind %s_url1 = %r' % (name, iface1,))
            # bind the server router to the queue dealer
            deal_sock = ctx.socket(zmq.DEALER)
            deal_sock.setsockopt_string(zmq.IDENTITY, 'queue.' + name + '.' + 'DEALER')
            deal_sock.bind(iface2)
            if VERBOSE_JOBS:
                print('bind %s_url2 = %r' % (name, iface2,))
            try:
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
                            # ROUTER sockets prepend the identity of the jobiface,
                            # for routing replies
                            if VERBOSE_JOBS:
                                print('ROUTER relayed %r via DEALER' % (msg,))
                            deal_sock.send_multipart(msg)
                            msg = None
                        if deal_sock in evts:
                            msg = deal_sock.recv_multipart()
                            if VERBOSE_JOBS:
                                print('DEALER relayed %r via ROUTER' % (msg,))
                            rout_sock.send_multipart(msg)
                            msg = None
            except KeyboardInterrupt:
                print('Caught ctrl+c in collector loop. Gracefully exiting')
            if VERBOSE_JOBS:
                print('Exiting %s' % (loop_name,))
    ut.set_funcname(queue_loop, loop_name)
    return queue_loop


collect_queue_loop = make_queue_loop(name='collect')


def engine_queue_loop(port_dict, job_counter=0):
    """
    Specialized queue loop
    """
    # Flow of information tags:
    # NAME: engine_queue
    update_proctitle('engine_queue_loop')

    iface1, iface2 = port_dict['engine_url1'], port_dict['engine_url2']
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
        if VERBOSE_JOBS:
            print('bind %s_url2 = %r' % (name, iface1,))

        # bind the server router to the queue dealer
        deal_sock = ctx.socket(zmq.DEALER)
        deal_sock.setsockopt_string(zmq.IDENTITY, 'special_queue.' + name + '.' + 'DEALER')
        deal_sock.bind(iface2)
        if VERBOSE_JOBS:
            print('bind %s_url2 = %r' % (name, iface2,))

        collect_deal_sock = ctx.socket(zmq.DEALER)
        collect_deal_sock.setsockopt_string(zmq.IDENTITY, queue_name + '.collect.DEALER')
        collect_deal_sock.connect(port_dict['collect_url1'])
        if VERBOSE_JOBS:
            print('connect collect_url1 = %r' % (port_dict['collect_url1'],))

        # but this shows what is really going on:
        poller = zmq.Poller()
        poller.register(rout_sock, zmq.POLLIN)
        poller.register(deal_sock, zmq.POLLIN)
        try:
            while True:
                evts = dict(poller.poll())
                if rout_sock in evts:
                    # HACK GET REQUEST FROM CLIENT

                    # jobid = 'jobid-%04d' % (job_counter,)
                    jobid = '%s' % (uuid.uuid4(), )
                    job_counter += 1
                    print('Creating jobid %r (counter %d)' % (jobid, job_counter, ))

                    # CALLER: job_client
                    idents, engine_request = rcv_multipart_json(rout_sock, num=1, print=print)

                    action          = engine_request['action']
                    args            = engine_request['args']
                    kwargs          = engine_request['kwargs']
                    callback_url    = engine_request['callback_url']
                    callback_method = engine_request['callback_method']
                    request         = engine_request['request']
                    restart_jobid   = engine_request.get('restart_jobid', None)

                    if restart_jobid is not None:
                        '[RESTARTING] Replacing jobid=%s with previous restart_jobid=%s' % (jobid, restart_jobid, )
                        jobid = restart_jobid

                    ######################################################################
                    # Status: Received (Notify Collector)
                    # Reply immediately with a new jobid
                    reply_notify = {
                        'jobid': jobid,
                        'status': 'received',
                        'action': 'notification',
                    }

                    if VERBOSE_JOBS:
                        print('...notifying collector about new job')
                    # CALLS: collector_notify
                    collect_deal_sock.send_json(reply_notify)

                    ######################################################################
                    # Status: Received (Notify Client)
                    if VERBOSE_JOBS:
                        print('... notifying client that job was accepted')
                    # RETURNS: job_client_return
                    send_multipart_json(rout_sock, idents, reply_notify)

                    ######################################################################
                    # Status: Metadata

                    # Reply immediately with a new jobid
                    metadata_notify = {
                        'jobid': jobid,
                        'metadata': {
                            'jobcounter'      : job_counter,
                            'action'          : action,
                            'args'            : args,
                            'kwargs'          : kwargs,
                            'callback_url'    : callback_url,
                            'callback_method' : callback_method,
                            'request'         : request,
                            'times'           : {
                                'received'    : _timestamp(),
                                'started'     : None,
                                'runtime'     : None,
                                'updated'     : None,
                                'completed'   : None,
                            }
                        },
                        'action': 'metadata',
                    }

                    if VERBOSE_JOBS:
                        print('...notifying collector about job metadata')
                    # CALLS: collector_notify
                    collect_deal_sock.send_json(metadata_notify)

                    ######################################################################
                    # Status: Accepted (Metadata Processed)
                    # Reply immediately with a new jobid
                    reply_notify = {
                        'jobid': jobid,
                        'status': 'accepted',
                        'action': 'notification',
                    }

                    if VERBOSE_JOBS:
                        print('...notifying collector about new job')
                    # CALLS: collector_notify
                    collect_deal_sock.send_json(reply_notify)

                    ######################################################################
                    # Status: Queueing on the Engine
                    assert 'jobid' not in engine_request
                    engine_request['jobid'] = jobid

                    if VERBOSE_JOBS:
                        print('... notifying backend engine to start')
                    # CALL: engine_
                    send_multipart_json(deal_sock, idents, engine_request)

                    # Release
                    idents = None
                    engine_request = None

                    ######################################################################
                    # Status: Queued
                    queued_notify = {
                        'jobid': jobid,
                        'status': 'queued',
                        'action': 'notification',
                    }

                    if VERBOSE_JOBS:
                        print('...notifying collector that job was queued')
                    # CALLS: collector_notify
                    collect_deal_sock.send_json(queued_notify)
                if deal_sock in evts:
                    pass
        except KeyboardInterrupt:
            print('Caught ctrl+c in %s queue. Gracefully exiting' % (loop_name,))

        if VERBOSE_JOBS:
            print('Exiting %s queue' % (loop_name,))


def engine_loop(id_, port_dict, dbdir, containerized):
    r"""
    IBEIS:
        This will be part of a worker process with its own IBEISController
        instance.

        Needs to send where the results will go and then publish the results there.

    The engine_loop - receives messages, performs some action, and sends a reply,
    preserving the leading two message parts as routing identities
    """
    # NAME: engine_
    # CALLED_FROM: engine_queue
    import ibeis
    #base_print = print  # NOQA
    print = partial(ut.colorprint, color='darkred')
    with ut.Indenter('[engine %d] ' % (id_)):
        if VERBOSE_JOBS:
            print('Initializing engine')
            print('connect engine_url2 = %r' % (port_dict['engine_url2'],))
        assert dbdir is not None

        engine_rout_sock = ctx.socket(zmq.ROUTER)
        engine_rout_sock.connect(port_dict['engine_url2'])

        collect_deal_sock = ctx.socket(zmq.DEALER)
        collect_deal_sock.setsockopt_string(zmq.IDENTITY, 'engine.collect.DEALER')
        collect_deal_sock.connect(port_dict['collect_url1'])

        if VERBOSE_JOBS:
            print('connect collect_url1 = %r' % (port_dict['collect_url1'],))
            print('engine is initialized')

        try:
            while True:
                # ibs = ibeis.opendb(dbdir=dbdir, use_cache=False, web=False, force_serial=True)
                ibs = ibeis.opendb(dbdir=dbdir, use_cache=False, web=False)
                update_proctitle('engine_loop', dbname=ibs.dbname)

                idents, engine_request = rcv_multipart_json(engine_rout_sock, print=print)

                action = engine_request['action']
                jobid  = engine_request['jobid']
                args   = engine_request['args']
                kwargs = engine_request['kwargs']
                callback_url = engine_request['callback_url']
                callback_method = engine_request['callback_method']

                # Notify start working
                reply_notify = {
                    'jobid': jobid,
                    'status': 'working',
                    'action': 'notification',
                }
                collect_deal_sock.send_json(reply_notify)

                engine_result = on_engine_request(ibs, jobid, action, args, kwargs)
                exec_status = engine_result['exec_status']

                # Notify start working
                reply_notify = {
                    'jobid': jobid,
                    'status': 'publishing',
                    'action': 'notification',
                }
                collect_deal_sock.send_json(reply_notify)

                # Store results in the collector
                collect_request = dict(
                    idents=idents,
                    action='store',
                    jobid=jobid,
                    engine_result=engine_result,
                    callback_url=callback_url,
                    callback_method=callback_method,
                )
                # if VERBOSE_JOBS:
                print('...done working. pushing result to collector for jobid %s' % (jobid, ))

                # CALLS: collector_store
                collect_deal_sock.send_json(collect_request)

                # Notify start working
                reply_notify = {
                    'jobid': jobid,
                    'status': exec_status,
                    'action': 'notification',
                }
                collect_deal_sock.send_json(reply_notify)

                # We no longer need the engine result, and can clear it's memory
                engine_request = None
                engine_result = None
                collect_request = None

                # Release the IBEIS controller for each job, hopefully freeing memory
                ibs = None

                # Explicitly try to release GPU memory
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass

                # Explicitly release Python memory
                try:
                    import gc
                    gc.collect()
                except:
                    pass

        except KeyboardInterrupt:
            print('Caught ctrl+c in engine loop. Gracefully exiting')
        # ----
        if VERBOSE_JOBS:
            print('Exiting engine loop')


def on_engine_request(ibs, jobid, action, args, kwargs):
    """ Run whenever the engine recieves a message """
    # Start working
    if VERBOSE_JOBS:
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
        if VERBOSE_JOBS:
            print('resolving action=%r to ibeis function=%r' % (action, action_func))
    try:
        key = '__jobid__'
        assert key not in kwargs
        kwargs[key] = jobid
        result = action_func(*args, **kwargs)
        exec_status = 'completed'
    except Exception as ex:
        result = ut.formatex(ex, keys=['jobid'], tb=True)
        result = ut.strip_ansi(result)
        exec_status = 'exception'
    json_result = ut.to_json(result)
    result = None  # Clear any used memory
    engine_result = dict(
        exec_status=exec_status,
        json_result=json_result,
        jobid=jobid,
    )
    return engine_result


def collector_loop(port_dict, dbdir, containerized):
    """
    Service that stores completed algorithm results
    """
    import ibeis
    print = partial(ut.colorprint, color='yellow')
    with ut.Indenter('[collect] '):
        collect_rout_sock = ctx.socket(zmq.ROUTER)
        collect_rout_sock.setsockopt_string(zmq.IDENTITY, 'collect.ROUTER')
        collect_rout_sock.connect(port_dict['collect_url2'])
        if VERBOSE_JOBS:
            print('connect collect_url2  = %r' % (port_dict['collect_url2'],))

        ibs = ibeis.opendb(dbdir=dbdir, use_cache=False, web=False)
        update_proctitle('collector_loop', dbname=ibs.dbname)

        # shelve_path = join(ut.get_shelves_dir(appname='ibeis'), 'engine')
        shelve_path = ibs.get_shelves_path()
        ut.ensuredir(shelve_path)

        # if exists(shelve_path):
        #     ut.delete(shelve_path)

        collecter_data = {}

        try:
            while True:
                # several callers here
                # CALLER: collector_notify
                # CALLER: collector_store
                # CALLER: collector_request_status
                # CALLER: collector_request_metadata
                # CALLER: collector_request_result
                idents, collect_request = rcv_multipart_json(collect_rout_sock, print=print)
                try:
                    reply = on_collect_request(ibs, collect_request, collecter_data,
                                               shelve_path, containerized=containerized)
                except Exception as ex:
                    import traceback
                    import sys
                    print(ut.repr3(collect_request))
                    ut.printex(ex, 'ERROR in collection')
                    print(traceback.format_exc())
                    # or
                    print(sys.exc_info()[0])
                send_multipart_json(collect_rout_sock, idents, reply)

                idents = None
                collect_request = None

                # Explicitly release Python memory
                try:
                    import gc
                    gc.collect()
                except:
                    pass
        except KeyboardInterrupt:
            print('Caught ctrl+c in collector loop. Gracefully exiting')
        if VERBOSE_JOBS:
            print('Exiting collector')


def _timestamp():
    timezone = pytz.timezone(TIMESTAMP_TIMEZONE)
    now = datetime.now(timezone)
    timestamp = now.strftime(TIMESTAMP_FMTSTR)
    return timestamp


def on_collect_request(ibs, collect_request, collecter_data,
                       shelve_path, containerized=False):
    """ Run whenever the collector recieves a message """
    import requests

    reply = {
        'status': 'ok',
    }

    action = collect_request['action']
    if VERBOSE_JOBS:
        print('...building action=%r response' % (action,))

    if action == 'notification':
        # From the Queue
        jobid = collect_request['jobid']
        status = collect_request['status']

        if jobid not in collecter_data:
            collecter_data[jobid] = {
                'status' : None,
                'input'  : None,
                'output' : None,
            }

        collecter_data[jobid]['status'] = status
        print('Notify %s' % ut.repr3(collecter_data[jobid]))

        lock_filepath = join(shelve_path, '%s.lock' % (jobid, ))
        if status == 'received':
            # Make waiting lock
            ut.touch(lock_filepath)
        elif status == 'completed':
            if exists(lock_filepath):
                ut.delete(lock_filepath)

            # Mark the engine request as finished
            record_filename = '%s.pkl' % (jobid, )
            record_filepath = join(shelve_path, record_filename)
            record = ut.load_cPkl(record_filepath)
            record['completed'] = True
            ut.save_cPkl(record_filepath, record)

        shelve_input_filepath = collecter_data.get(jobid, {}).get('input', None)
        if shelve_input_filepath is not None:
            shelve_input_filepath = collecter_data[jobid]['input']
            shelf = shelve.open(shelve_input_filepath, writeback=True)

            key = str('metadata')
            metadata = shelf[key]
            times = metadata.get('times', {})

            times['updated'] = _timestamp()
            if status == 'working':
                times['started'] = _timestamp()
            if status == 'completed':
                times['completed'] = _timestamp()

            # Calculate runtime
            started   = times.get('started', None)
            completed = times.get('completed', None)
            runtime   = times.get('runtime', None)

            if None not in [started, completed]:
                try:
                    assert runtime is None
                    TIMESTAMP_FMTSTR_ = ' '.join(TIMESTAMP_FMTSTR.split(' ')[:-1])
                    started = ' '.join(started.split(' ')[:-1])
                    completed = ' '.join(completed.split(' ')[:-1])
                    started_date = datetime.strptime(started, TIMESTAMP_FMTSTR_)
                    completed_date = datetime.strptime(completed, TIMESTAMP_FMTSTR_)
                    delta = completed_date - started_date
                    total_seconds = int(delta.total_seconds())
                    total_seconds_ = total_seconds
                    hours = total_seconds // (60 * 60)
                    total_seconds -= hours * 60 * 60
                    minutes = total_seconds // 60
                    total_seconds -= minutes * 60
                    seconds = total_seconds
                    args = (hours, minutes, seconds, total_seconds_, )
                    times['runtime'] = '%d hours %d min. %s sec. (total: %d sec.)' % args
                except:
                    times['runtime'] = 'ERROR'

            # Save result to shelf
            try:
                shelf[key] = metadata
            finally:
                shelf.close()
            metadata = None  # Release memory
    elif action == 'register':
        jobid     = collect_request['jobid']
        status    = collect_request['status']
        completed = status == 'completed'

        shelve_input_filepath  = abspath(join(shelve_path, '%s.input.shelve' % (jobid, )))
        shelve_output_filepath = abspath(join(shelve_path, '%s.output.shelve' % (jobid, )))

        if completed:
            assert status == 'completed'

            # Ensure these shelves are valid
            try:
                shelf = shelve.open(shelve_input_filepath, 'r')
            except:
                shelf = shelve.open(shelve_input_filepath)
            shelf = None

            try:
                shelf = shelve.open(shelve_output_filepath, 'r')
            except:
                shelf = shelve.open(shelve_output_filepath)
            shelf = None
        else:
            assert status == 'suppressed'

            # Ensure these shelves are valid
            try:
                shelf = shelve.open(shelve_input_filepath, 'r')
            except:
                shelf = shelve.open(shelve_input_filepath)
            shelf = None

            try:
                try:
                    shelf = shelve.open(shelve_output_filepath, 'r')
                except:
                    shelf = shelve.open(shelve_output_filepath)
                shelf = None
            except:
                # The shelve appears to be corrupted, ignore it
                shelve_output_filepath = None

        collecter_data[jobid] = {
            'status' : status,
            'input'  : shelve_input_filepath,
            'output' : shelve_output_filepath,
        }
        print('Register %s' % ut.repr3(collecter_data[jobid]))
        reply['jobid'] = jobid
    elif action == 'metadata':
        # From the Engine
        jobid    = collect_request['jobid']
        metadata = collect_request.get('metadata', None)

        if jobid not in collecter_data:
            collecter_data[jobid] = {
                'status' : None,
                'input'  : None,
                'output' : None,
            }

        shelve_input_filepath = abspath(join(shelve_path, '%s.input.shelve' % (jobid, )))
        collecter_data[jobid]['input'] = shelve_input_filepath

        shelf = shelve.open(shelve_input_filepath, writeback=True)
        try:
            key = str('metadata')
            shelf[key] = metadata
        finally:
            shelf.close()

        print('Store Metadata %s' % ut.repr3(collecter_data[jobid]))

        # Release memory
        metadata = None

        if VERBOSE_JOBS:
            print('stored metadata')
    elif action == 'store':
        # From the Engine
        engine_result   = collect_request['engine_result']
        callback_url    = collect_request['callback_url']
        callback_method = collect_request['callback_method']

        jobid = engine_result['jobid']
        assert jobid in collecter_data

        shelve_output_filepath = abspath(join(shelve_path, '%s.output.shelve' % (jobid, )))
        collecter_data[jobid]['output'] = shelve_output_filepath

        shelf = shelve.open(shelve_output_filepath, writeback=True)
        try:
            key = str('result')
            shelf[key] = engine_result
        finally:
            shelf.close()

        # Release memory
        engine_result = None

        print('Store Result %s' % ut.repr3(collecter_data[jobid]))

        if callback_url is not None:

            if containerized:
                callback_url = callback_url.replace('://localhost/', '://wildbook:8080/')

            if callback_method is None:
                callback_method = 'post'
            else:
                callback_method = callback_method.lower()

            # if VERBOSE_JOBS:
            print('calling callback_url using callback_method')

            try:
                args = (callback_url, callback_method)
                print('ATTEMPTING CALLBACK TO %r\n\tMETHOD: %r' % args)

                # requests.get(callback_url)
                data_dict = {'jobid': jobid}
                if callback_method == 'post':
                    response = requests.post(callback_url, data=data_dict)
                elif callback_method == 'get':
                    response = requests.get(callback_url, params=data_dict)
                elif callback_method == 'put':
                    response = requests.put(callback_url, data=data_dict)
                else:
                    raise ValueError('callback_method %r unsupported' %
                                     (callback_method, ))
                try:
                    text = response.text
                    text = unicode(text).encode('utf-8')
                except:
                    text = None

                args = (callback_url, callback_method, data_dict, response, text, )
                print('CALLBACK COMPLETED TO %r\n\tMETHOD: %r\n\tDATA: %r\n\tRESPONSE: %r\n\tTEXT: %r' % args)

            except Exception as ex:
                msg = (('ERROR in collector. '
                        'Tried to call callback_url=%r with callback_method=%r')
                       % (callback_url, callback_method, ))
                print(msg)
                ut.printex(ex, msg)
            #requests.post(callback_url)
        if VERBOSE_JOBS:
            print('stored result')
    elif action == 'job_status':
        # From a Client
        jobid = collect_request['jobid']
        reply['jobid'] = jobid
        reply['jobstatus'] = collecter_data.get(jobid, {}).get('status', 'unknown')
    elif action == 'job_status_dict':
        # From a Client
        # print('Fetch Status %s' % ut.repr3(collecter_data))

        json_result = {}
        for jobid in collecter_data:
            status = collecter_data[jobid]['status']

            shelve_input_filepath = collecter_data[jobid]['input']
            try:
                shelf = shelve.open(shelve_input_filepath, 'r')
            except:
                shelf = shelve.open(shelve_input_filepath)
            try:
                key = str('metadata')
                metadata = shelf[key]
            finally:
                shelf.close()

            times = metadata.get('times', {})
            request = metadata.get('request', {})
            if request is None:
                request = {}

            json_result[jobid] = {
                'status'         : status,
                'jobcounter'     : metadata.get('jobcounter', None),
                'action'         : metadata.get('action', None),
                'endpoint'       : request.get('endpoint', None),
                'function'       : request.get('function', None),
                'time_received'  : times.get('received', None),
                'time_started'   : times.get('started', None),
                'time_runtime'   : times.get('runtime', None),
                'time_updated'   : times.get('updated', None),
                'time_completed' : times.get('completed', None),
            }

        metadata = None  # Release memory
        reply['json_result'] = json_result
    elif action == 'job_id_list':
        reply['jobid_list'] = sorted(list(collecter_data.keys()))
    elif action == 'job_input':
        # From a Client
        jobid = collect_request['jobid']
        reply['jobid'] = jobid

        if jobid not in collecter_data:
            reply['status'] = 'invalid'
            reply['json_result'] = None
        else:
            assert 'input' in collecter_data[jobid]
            assert 'output' in collecter_data[jobid]

            print('Fetch Input %s' % ut.repr3(collecter_data[jobid]))

            shelve_input_filepath = collecter_data[jobid]['input']
            try:
                shelf = shelve.open(shelve_input_filepath, 'r')
            except:
                shelf = shelve.open(shelve_input_filepath)
            try:
                key = str('metadata')
                metadata = shelf[key]
            finally:
                shelf.close()

            reply['json_result'] = metadata

            # Clear loaded memory
            metadata = None
    elif action == 'job_result':
        # From a Client
        jobid = collect_request['jobid']
        reply['jobid'] = jobid

        if jobid not in collecter_data:
            reply['status'] = 'invalid'
            reply['json_result'] = None
        else:
            assert 'status' in collecter_data[jobid]
            assert 'input' in collecter_data[jobid]
            assert 'output' in collecter_data[jobid]

            status = collecter_data[jobid]['status']
            print('Fetch Result %s' % ut.repr3(collecter_data[jobid]))

            shelve_output_filepath = collecter_data[jobid]['output']
            if shelve_output_filepath is None:
                # Job failed to store output
                reply['status'] = 'incomplete' if status != 'suppressed' else 'suppressed'
                reply['json_result'] = None
            else:
                try:
                    shelf = shelve.open(shelve_output_filepath, 'r')
                except:
                    shelf = shelve.open(shelve_output_filepath)
                try:
                    key = str('result')
                    engine_result = shelf[key]
                finally:
                    shelf.close()

                reply['status'] = engine_result['exec_status']
                json_result = engine_result['json_result']
                reply['json_result'] = ut.from_json(json_result)

                # Clear loaded memory
                engine_result = None
    else:
        # Other
        print('...error unknown action=%r' % (action,))
        reply['status'] = 'error'

    return reply


def send_multipart_json(sock, idents, reply):
    """ helper """
    reply_json = ut.to_json(reply).encode('utf-8')
    reply = None
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
    request_json = None
    multi_msg = None
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
