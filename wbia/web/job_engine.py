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

    python -m wbia.web.job_engine job_engine_tester

    We can test these separately by first starting the background server
    python -m wbia.web.job_engine job_engine_tester --bg

    Alternative:
    python -m wbia.web.job_engine job_engine_tester --bg --no-engine
    python -m wbia.web.job_engine job_engine_tester --bg --only-engine --fg-engine

    And then running the forground process
    python -m wbia.web.job_engine job_engine_tester --fg
"""
import multiprocessing
import os
import random
import re
import threading
import time
import uuid  # NOQA
from datetime import datetime, timedelta
from functools import partial
from os.path import basename, exists, join, splitext

import flask
import numpy as np
import pytz

# if False:
#    import os
#    os.environ['UTOOL_NOCNN'] = 'True'
import logging
import utool as ut
import zmq

from wbia.control import controller_inject
from wbia.utils import call_houston

print, rrr, profile = ut.inject2(__name__)  # NOQA
logger = logging.getLogger('wbia')

# ZMQ receive timeout for engine/collector sockets.  Configurable via
# WBIA_ENGINE_TIMEOUT_SEC (seconds).  The old 30s default caused crashes
# on databases with >1000 annotations.  Note: on engine failure, callers
# will block for this long before receiving a timeout error.
try:
    _timeout_sec = int(os.environ.get('WBIA_ENGINE_TIMEOUT_SEC', '300'))
    if _timeout_sec <= 0:
        raise ValueError('must be positive')
except (ValueError, TypeError):
    _timeout_sec = 300
_ENGINE_RCVTIMEO_MS = _timeout_sec * 1000

CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)
register_api = controller_inject.get_wbia_flask_api(__name__)

def _get_global_zmq_ctx():
    """Return the process-global ZMQ context (created on first call).

    Lazy so that importing this module in the Gunicorn master does not
    create a ZMQ context that would be inherited (and become invalid)
    across fork().  Background processes (engine, collector) each call
    this in their own process after being spawned via multiprocessing.
    """
    return zmq.Context.instance()


# FIXME: needs to use correct number of ports
URL = 'tcp://127.0.0.1'
NUM_DEFAULT_ENGINES = ut.get_argval('--engine-lane-workers', int, 2)
NUM_SLOW_ENGINES = ut.get_argval('--engine-slow-lane-workers', int, NUM_DEFAULT_ENGINES)
NUM_FAST_ENGINES = ut.get_argval('--engine-fast-lane-workers', int, NUM_DEFAULT_ENGINES)
NUM_ENGINES = {
    'slow': NUM_SLOW_ENGINES,
    'fast': NUM_FAST_ENGINES,
}
VERBOSE_JOBS = ut.get_argflag('--verbose-jobs')


TIMESTAMP_FMTSTR = '%Y-%m-%d %H:%M:%S %Z'
TIMESTAMP_TIMEZONE = 'US/Pacific'


def update_proctitle(procname, dbname=None):
    try:
        import setproctitle

        print('CHANGING PROCESS TITLE')
        old_title = setproctitle.getproctitle()
        print('old_title = {!r}'.format(old_title))
        hostname = ut.get_computer_name()
        new_title = 'WBIA_{}_{}_{}'.format(dbname, hostname, procname)
        print('new_title = {!r}'.format(new_title))
        setproctitle.setproctitle(new_title)
    except ImportError:
        print('pip install setproctitle')


def _get_engine_job_paths(ibs):
    shelve_path = ibs.get_shelves_path()
    ut.ensuredir(shelve_path)
    record_filepath_list = list(ut.iglob(join(shelve_path, '*.pkl')))
    return record_filepath_list


def _save_engine_record(record_filepath, record):
    """Write a job record .pkl atomically.

    ``ut.save_cPkl`` opens the destination with ``'wb'`` and dumps in place,
    so a process killed mid-write (container stop/restart, OOM) leaves a
    truncated pickle behind and the next startup dies in
    ``queue_interrupted_jobs``.  Write to a uniquely named sibling temp file
    and ``os.replace`` it into position so readers only ever see a complete
    record, and so two writers racing on the same job can never share (and
    half-publish) one temp file.  The temp suffix does not end in ``.pkl`` so
    the startup ``*.pkl`` glob ignores it; the ``{jobid}*`` archive glob still
    sweeps up any stragglers.
    """
    tmp_filepath = '{}.{}.tmp'.format(record_filepath, uuid.uuid4().hex)
    try:
        ut.save_cPkl(tmp_filepath, record, verbose=False)
        os.replace(tmp_filepath, record_filepath)
    except BaseException:
        try:
            os.remove(tmp_filepath)
        except OSError:
            pass
        raise


def _archive_job_files(jobid, shelve_path, shelve_archive_path, job_store=None):
    """Move every on-disk file for ``jobid`` into the archive directory and
    drop its SQLite row (if any)."""
    ut.ensuredir(shelve_archive_path)
    job_scr_filepath_list = list(ut.iglob(join(shelve_path, '{}*'.format(jobid))))
    for job_scr_filepath in job_scr_filepath_list:
        job_dst_filepath = job_scr_filepath.replace(shelve_path, shelve_archive_path)
        # ut.copy allows for overwrite, ut.move does not
        ut.copy(job_scr_filepath, job_dst_filepath, overwrite=True)
        ut.delete(job_scr_filepath)
    if job_store is not None:
        job_store.delete_job(jobid)


@register_ibs_method
def fetch_job(ibs, jobid):
    from os.path import exists

    shelve_path = ibs.get_shelves_path()
    job_record_filename = '{}.pkl'.format(jobid)
    job_record_filepath = join(shelve_path, job_record_filename)
    assert exists(job_record_filepath)

    job_record = ut.load_cPkl(job_record_filepath)

    job_action = job_record['request']['action']
    job_args = job_record['request']['args']
    job_kwargs = job_record['request']['kwargs']

    job_func = getattr(ibs, job_action, None)

    return job_action, job_func, job_args, job_kwargs


@register_ibs_method
def retry_job(ibs, jobid):
    job_action, job_func, job_args, job_kwargs = ibs.fetch_job(jobid)

    if job_func is not None:
        job_result = job_func(*job_args, **job_kwargs)

    return job_action, job_func, job_args, job_kwargs, job_result


@register_ibs_method
def initialize_job_manager(ibs):
    """
    Starts a background zmq job engine. Initializes a zmq object in this thread
    that can talk to the background processes.

    Run from the webserver

    CommandLine:
        python -m wbia.web.job_engine --exec-initialize_job_manager:0

    Example:
        >>> # DISABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(--job-engine-tests)
        >>> from wbia.web.job_engine import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> from wbia.web import apis_engine
        >>> from wbia.web import job_engine
        >>> ibs.load_plugin_module(job_engine)
        >>> ibs.load_plugin_module(apis_engine)
        >>> ibs.initialize_job_manager()
        >>> print('Initializqation success. Now closing')
        >>> ibs.close_job_manager()
        >>> print('Closing success.')

    """
    ibs.job_manager = ut.DynStruct()

    use_static_ports = False

    if ut.get_argflag('--web-deterministic-ports'):
        use_static_ports = True

    if ut.get_argflag('--fg'):
        ibs.job_manager.reciever = JobBackend(use_static_ports=True)
    else:
        ibs.job_manager.reciever = JobBackend(use_static_ports=use_static_ports)
        ibs.job_manager.reciever.initialize_background_processes(
            dbdir=ibs.get_dbdir(), containerized=ibs.containerized
        )

    ibs.job_manager.jobiface = JobInterface(
        0, ibs.job_manager.reciever.port_dict, ibs=ibs
    )

    # Use temporary sockets for startup tasks (queue_interrupted_jobs).
    # These must be opened, used, and CLOSED before Gunicorn forks,
    # because ZMQ sockets cannot survive fork().  The permanent sockets
    # for request handling are created lazily post-fork via _ensure_sockets().
    ibs.job_manager.jobiface.initialize_client_thread()
    ibs.job_manager.jobiface.queue_interrupted_jobs()
    ibs.job_manager.jobiface.close_client_sockets()

    # import wbia
    # #dbdir = '/media/raid/work/testdb1'
    # from wbia.web import app
    # web_port = ibs.get_web_port_via_scan()
    # if web_port is None:
    #     raise ValueError('IA web server is not running on any expected port')
    # proc = ut.spawn_background_process(app.start_from_wbia, ibs, port=web_port)


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
        python -m wbia.web.job_engine --exec-get_job_status

        # Start job queue in its own process
        python -m wbia.web.job_engine job_engine_tester --bg
        # Start web server in its own process
        ./main.py --web --fg
        pass
        # Run foreground process
        python -m wbia.web.job_engine --exec-get_job_status:0 --fg

    Example:
        >>> # xdoctest: +REQUIRES(--web-tests)
        >>> # xdoctest: +REQUIRES(--job-engine-tests)
        >>> from wbia.web.job_engine import *  # NOQA
        >>> import wbia
        >>> with wbia.opendb_bg_web('testdb1', managed=True) as web_ibs:  # , domain='http://52.33.105.88')
        ...     # Test get status of a job id that does not exist
        ...     response = web_ibs.send_wbia_request('/api/engine/job/', jobid='badjob')

    """
    status = ibs.job_manager.jobiface.get_job_id_list()
    jobid_list = status['jobid_list']
    return jobid_list


@register_ibs_method
@register_api(
    '/api/engine/process/status/', methods=['GET', 'POST'], __api_plural_check__=False
)
def get_process_alive_status(ibs):
    status_dict = ibs.job_manager.reciever.get_process_alive_status()
    print('status_dict = {!r}'.format(status_dict))
    return status_dict


@register_ibs_method
@register_api(
    '/api/engine/job/status/', methods=['GET', 'POST'], __api_plural_check__=False
)
def get_job_status(ibs, jobid=None, limit=0, **kwargs):
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
        python -m wbia.web.job_engine --exec-get_job_status

        # Start job queue in its own process
        python -m wbia.web.job_engine job_engine_tester --bg
        # Start web server in its own process
        ./main.py --web --fg
        pass
        # Run foreground process
        python -m wbia.web.job_engine --exec-get_job_status:0 --fg

    Example:
        >>> # xdoctest: +REQUIRES(--web-tests)
        >>> # xdoctest: +REQUIRES(--job-engine-tests)
        >>> from wbia.web.job_engine import *  # NOQA
        >>> import wbia
        >>> with wbia.opendb_bg_web('testdb1', managed=True) as web_ibs:  # , domain='http://52.33.105.88')
        ...     # Test get status of a job id that does not exist
        ...     response = web_ibs.send_wbia_request('/api/engine/job/status/', jobid='badjob')

    """
    if jobid is None:
        status = ibs.job_manager.jobiface.get_job_status_dict(limit=limit)
    else:
        status = ibs.job_manager.jobiface.get_job_status(jobid)
    return status


@register_ibs_method
@register_api(
    '/api/engine/job/metadata/', methods=['GET', 'POST'], __api_plural_check__=False
)
def get_job_metadata(ibs, jobid):
    """
    Web call that returns the metadata of a job

    CommandLine:
        # Run Everything together
        python -m wbia.web.job_engine --exec-get_job_metadata

        # Start job queue in its own process
        python -m wbia.web.job_engine job_engine_tester --bg
        # Start web server in its own process
        ./main.py --web --fg
        pass
        # Run foreground process
        python -m wbia.web.job_engine --exec-get_job_metadata:0 --fg

    Example:
        >>> # xdoctest: +REQUIRES(--web-tests)
        >>> # xdoctest: +REQUIRES(--slow)
        >>> # xdoctest: +REQUIRES(--job-engine-tests)
        >>> # xdoctest: +REQUIRES(--web-tests)
        >>> from wbia.web.job_engine import *  # NOQA
        >>> import wbia
        >>> with wbia.opendb_bg_web('testdb1', managed=True) as web_ibs:  # , domain='http://52.33.105.88')
        ...     # Test get metadata of a job id that does not exist
        ...     response = web_ibs.send_wbia_request('/api/engine/job/metadata/', jobid='badjob')

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


def _get_random_open_port():
    port = random.randint(1024, 49151)
    while not ut.is_local_port_open(port):
        port = random.randint(1024, 49151)
    assert ut.is_local_port_open(port)
    return port


def job_engine_tester():
    """
    CommandLine:
        python -m wbia.web.job_engine --exec-job_engine_tester
        python -b -m wbia.web.job_engine --exec-job_engine_tester

        python -m wbia.web.job_engine job_engine_tester
        python -m wbia.web.job_engine job_engine_tester --bg
        python -m wbia.web.job_engine job_engine_tester --fg

    Example:
        >>> # SCRIPT
        >>> from wbia.web.job_engine import *  # NOQA
        >>> job_engine_tester()
    """
    _init_signals()
    # now start a few clients, and fire off some requests
    client_id = np.random.randint(1000)
    reciever = JobBackend(use_static_ports=True)
    jobiface = JobInterface(client_id, reciever.port_dict)
    from wbia.init import sysres

    if ut.get_argflag('--bg'):
        dbdir = sysres.get_args_dbdir(
            defaultdb='cache', allow_newdir=False, db=None, dbdir=None
        )
        reciever.initialize_background_processes(dbdir)
        print('[testzmq] parent process is looping forever')
        while True:
            time.sleep(1)
    elif ut.get_argflag('--fg'):
        jobiface.initialize_client_thread()
    else:
        dbdir = sysres.get_args_dbdir(
            defaultdb='cache', allow_newdir=False, db=None, dbdir=None
        )
        reciever.initialize_background_processes(dbdir)
        jobiface.initialize_client_thread()

    # Foreground test script
    print('... waiting for jobs')
    if ut.get_argflag('--cmd'):
        ut.embed()
        # jobiface.queue_job()
    else:
        print('[test] ... emit test1')
        callback_url = None
        callback_method = None
        args = (1,)
        jobid1 = jobiface.queue_job(
            action='helloworld',
            callback_url=callback_url,
            callback_method=callback_method,
            args=args,
        )
        jobiface.wait_for_job_result(jobid1)
        jobid_list = []

        args = ([1], [3, 4, 5])
        kwargs = dict(cfgdict={'K': 1})
        identify_jobid = jobiface.queue_job(
            action='query_chips_simple_dict',
            callback_url=callback_url,
            callback_method=callback_method,
            args=args,
            kwargs=kwargs,
        )
        for jobid in jobid_list:
            jobiface.wait_for_job_result(jobid)

        jobiface.wait_for_job_result(identify_jobid)
    print('FINISHED TEST SCRIPT')


def spawn_background_process(func, *args, **kwargs):
    import utool as ut

    func_name = ut.get_funcname(func)
    name = 'job-engine.Progress-' + func_name
    proc_obj = multiprocessing.Process(target=func, name=name, args=args, kwargs=kwargs)
    proc_obj.daemon = True
    proc_obj.start()
    return proc_obj


class JobBackend(object):
    def __init__(self, **kwargs):
        # self.num_engines = 3
        self.engine_queue_proc = None
        self.engine_lanes = ['fast', 'slow']

        self.engine_lanes = [lane.lower() for lane in self.engine_lanes]
        assert 'slow' in self.engine_lanes

        self.num_engines = {lane: NUM_ENGINES[lane] for lane in self.engine_lanes}
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
            for lane in self.engine_procs:
                for engine in self.engine_procs[lane]:
                    try:
                        engine.terminate()
                    except Exception:
                        pass
        if self.engine_queue_proc is not None:
            try:
                self.engine_queue_proc.terminate()
            except Exception:
                pass
        if self.collect_proc is not None:
            try:
                self.collect_proc.terminate()
            except Exception:
                pass
        if self.collect_queue_proc is not None:
            try:
                self.collect_queue_proc.terminate()
            except Exception:
                pass
        if VERBOSE_JOBS:
            print('Killed external procs')

    def _initialize_job_ports(self, use_static_ports=False, static_root=51381):
        # _portgen = functools.partial(next, itertools.count(51381))
        key_list = [
            'collect_pull_url',
            'collect_push_url',
            'engine_pull_url',
            # 'engine_push_url',
            # 'collect_pushpull_url',
        ]
        for lane in self.engine_lanes:
            key_list.append('engine_{}_push_url'.format(lane))

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
            key: '%s:%d' % (URL, port) for key, port in list(zip(key_list, port_list))
        }

    def initialize_background_processes(
        self, dbdir=None, containerized=False, thread=False
    ):
        # if VERBOSE_JOBS:
        print('Initialize Background Processes')

        def _spawner(func, *args, **kwargs):

            # if thread:
            #     _spawner_func_ = ut.spawn_background_daemon_thread
            # else:
            #     _spawner_func_ = ut.spawn_background_process
            _spawner_func_ = spawn_background_process

            proc = _spawner_func_(func, *args, **kwargs)
            assert proc.is_alive(), 'proc (%s) died too soon' % (ut.get_funcname(func))
            return proc

        if self.spawn_queue:
            self.engine_queue_proc = _spawner(
                engine_queue_loop, self.port_dict, self.engine_lanes
            )
            self.collect_queue_proc = _spawner(collect_queue_loop, self.port_dict)

        if self.spawn_collector:
            self.collect_proc = _spawner(
                collector_loop, self.port_dict, dbdir, containerized
            )

        if self.spawn_engine:
            if self.fg_engine:
                print('ENGINE IS IN DEBUG FOREGROUND MODE')
                # Spawn engine in foreground process
                assert self.num_engines == 1, 'fg engine only works with one engine'
                engine_loop(0, self.port_dict, dbdir)
                assert False, 'should never see this'
            else:
                # Normal case
                if self.engine_procs is None:
                    self.engine_procs = {}

                for lane in self.engine_lanes:
                    if lane not in self.engine_procs:
                        self.engine_procs[lane] = []
                    for i in range(self.num_engines[lane]):
                        proc = _spawner(
                            engine_loop, i, self.port_dict, dbdir, containerized, lane
                        )
                        self.engine_procs[lane].append(proc)

        # Check if online
        # wait for processes to spin up
        if self.spawn_queue:
            assert self.engine_queue_proc.is_alive(), 'engine died too soon'
            assert self.collect_queue_proc.is_alive(), 'collector queue died too soon'

        if self.spawn_collector:
            assert self.collect_proc.is_alive(), 'collector died too soon'

        if self.spawn_engine:
            for lane in self.engine_procs:
                for engine in self.engine_procs[lane]:
                    assert engine.is_alive(), 'engine died too soon'

    def get_process_alive_status(self):
        """Check if background processes are alive.

        Uses os.kill(pid, 0) instead of proc.is_alive() because
        is_alive() relies on waitpid(), which only works for direct
        children.  After Gunicorn fork(), the worker process cannot
        waitpid() on the master's children (engine, collector).
        """
        import os

        def _pid_alive(proc):
            try:
                pid = proc.pid
                if pid is None:
                    return False
                os.kill(pid, 0)
                return True
            except (ProcessLookupError, PermissionError):
                return False
            except Exception:
                return False

        status_dict = {}

        if self.spawn_queue:
            status_dict['engine_queue'] = _pid_alive(self.engine_queue_proc)
            status_dict['collect_queue'] = _pid_alive(self.collect_queue_proc)

        if self.spawn_collector:
            status_dict['collector'] = _pid_alive(self.collect_proc)

        if self.spawn_engine:
            for lane in self.engine_procs:
                for id_, engine in enumerate(self.engine_procs[lane]):
                    engine_str = 'engine.{}.{}'.format(lane, id_)
                    status_dict[engine_str] = _pid_alive(engine)

        return status_dict


def initialize_process_record(
    record_filepath,
    shelve_path,
    shelve_archive_path,
    jobiface_id,
    job_store=None,
):
    MAX_ATTEMPTS = 20
    ARCHIVE_DAYS = 3

    timezone = pytz.timezone(TIMESTAMP_TIMEZONE)
    now = datetime.now(timezone)
    now = now.replace(microsecond=0)
    now = now.replace(second=0)
    now = now.replace(minute=0)
    now = now.replace(hour=0)

    archive_delta = timedelta(days=ARCHIVE_DAYS)
    archive_date = now - archive_delta
    archive_timestamp = archive_date.strftime(TIMESTAMP_FMTSTR)

    jobid = splitext(basename(record_filepath))[0]
    jobcounter = None

    # Load the engine record (.pkl — still written by web threads for crash recovery)
    try:
        record = ut.load_cPkl(record_filepath, verbose=False)
        if not isinstance(record, dict):
            raise TypeError('expected dict, got {!r}'.format(type(record)))
    except Exception as ex:
        # A truncated / unreadable record (e.g. the process was killed
        # mid-write) must not abort startup.  Quarantine it and move on.
        print(
            '[job_engine] Unreadable job record %r (%s: %s) — archiving'
            % (record_filepath, type(ex).__name__, ex)
        )
        try:
            _archive_job_files(jobid, shelve_path, shelve_archive_path, job_store)
        except Exception as archive_ex:
            # Quarantine itself failed (permissions, disk full, SQLite lock).
            # It may be partial (e.g. file moved but SQLite row not dropped);
            # whatever is left gets retried next boot.  Skip the record for
            # this boot rather than aborting startup.
            print(
                '[job_engine] Quarantine FAILED for %r (%s: %s) — skipping for this boot'
                % (record_filepath, type(archive_ex).__name__, archive_ex)
            )
        archived, completed, suppressed, corrupted = True, False, False, True
        return jobcounter, jobid, None, archived, completed, suppressed, corrupted

    # Load the record info
    engine_request = record.get('request', None)
    attempts = record.get('attempts', 0)
    completed = record.get('completed', False)

    # Check status
    suppressed = attempts >= MAX_ATTEMPTS
    corrupted = engine_request is None

    # Load metadata from SQLite (or None if not yet stored)
    metadata = job_store.get_metadata(jobid) if job_store is not None else None

    if metadata is None:
        # Pre-upgrade jobs won't have SQLite rows — treat as corrupted so
        # they get archived on next restart.  This is expected on first
        # boot after the shelve→SQLite migration.
        print('No SQLite metadata for %s — archiving (expected after upgrade)' % (jobid,))
        corrupted = True

    archived = False
    if not corrupted:
        jobcounter = metadata.get('jobcounter', None)
        times = metadata.get('times', {})

        if jobcounter is None:
            print('Missing jobcounter...corrupted')
            corrupted = True

        job_age = None
        if not corrupted and completed:
            completed_timestamp = times.get('completed', None)
            if completed_timestamp is not None:
                try:
                    archive_elapsed = calculate_timedelta(
                        completed_timestamp, archive_timestamp
                    )
                    job_age = archive_elapsed[-1]
                    archived = job_age > 0
                except Exception:
                    args = (
                        completed_timestamp,
                        archive_timestamp,
                    )
                    print(
                        '[job_engine] Could not determine archive status!\n\tCompleted: %r\n\tArchive: %r'
                        % args
                    )

            if archived:
                color = 'brightmagenta'
                print_ = partial(ut.colorprint, color=color)
                print_('ARCHIVING JOB (AGE: %d SECONDS)' % (job_age,))
                # Move .pkl (and any leftover shelve files) to archive and
                # remove from SQLite
                _archive_job_files(jobid, shelve_path, shelve_archive_path, job_store)

    if archived:
        # We have archived the job, don't bother registering it
        engine_request = None
    else:
        if completed or suppressed or corrupted:
            # Register the job, pass the jobcounter and jobid only
            engine_request = None
        else:
            # We have a pending job, restart with the original request
            color = 'brightblue' if attempts == 0 else 'brightred'
            print_ = partial(ut.colorprint, color=color)

            print_('RESTARTING FAILED JOB FROM RESTART (ATTEMPT %d)' % (attempts + 1,))
            print_(ut.repr3(record_filepath))

            times = metadata.get('times', {})
            received = times['received']

            engine_request['restart_jobid'] = jobid
            engine_request['restart_jobcounter'] = jobcounter
            engine_request['restart_received'] = received
            record['attempts'] = attempts + 1

            _save_engine_record(record_filepath, record)

    values = jobcounter, jobid, engine_request, archived, completed, suppressed, corrupted
    return values


class JobInterface(object):
    def __init__(jobiface, id_, port_dict, ibs=None):
        jobiface.id_ = id_
        jobiface.ibs = ibs
        jobiface.verbose = 2 if VERBOSE_JOBS else 1
        jobiface.port_dict = port_dict
        # Locks to protect ZMQ sockets from concurrent access by
        # multiple Gunicorn/web-server threads.  ZMQ sockets are NOT
        # thread-safe, so every send/recv pair must be atomic.
        jobiface._engine_lock = threading.Lock()
        jobiface._collect_lock = threading.Lock()
        jobiface._init_lock = threading.Lock()
        jobiface.engine_recieve_socket = None
        jobiface.collect_recieve_socket = None
        # Per-thread SQLite readers for job status/result queries.
        # Bypasses the collector ZMQ bottleneck for read-only operations.
        # Each thread gets its own connection via threading.local() to
        # avoid interleaved execute()/fetchone() on a shared connection.
        jobiface._reader_local = threading.local()
        print('JobInterface ports:')
        ut.print_dict(jobiface.port_dict)

    def _get_reader_store(jobiface):
        """Return a per-thread JobStore for direct SQLite access.

        Each Gunicorn thread gets its own connection, so reads are fully
        concurrent.  SQLite WAL mode allows concurrent reads while the
        collector writes.
        """
        store = getattr(jobiface._reader_local, 'store', None)
        if store is not None:
            return store
        from wbia.web.job_store import JobStore
        shelve_path = jobiface.ibs.get_shelves_path()
        store = JobStore(join(shelve_path, 'jobs.db'))
        jobiface._reader_local.store = store
        return store

    def _ensure_sockets(jobiface):
        """Lazily create ZMQ sockets on first use (post-fork), or recreate
        after a timeout reset destroyed one or both sockets.

        Uses _init_lock to prevent concurrent initialization.
        """
        if jobiface.engine_recieve_socket is not None and jobiface.collect_recieve_socket is not None:
            return
        with jobiface._init_lock:
            if jobiface.engine_recieve_socket is not None and jobiface.collect_recieve_socket is not None:
                return
            jobiface._rebuild_sockets()

    def _engine_request(jobiface, msg):
        """Send a message to the engine and return the reply (thread-safe)."""
        import time as _time

        action = msg.get('action', msg.get('__set_jobcounter__', 'unknown'))
        jobiface._ensure_sockets()
        acquired = jobiface._engine_lock.acquire(timeout=10)
        if not acquired:
            logger.warning('[job_engine] Engine lock contention — could not acquire '
                           'within 10s for action=%r (thread=%s)',
                           action, threading.current_thread().name)
            raise RuntimeError(
                'Could not acquire engine lock within 10s. '
                'Another request is likely waiting on the engine.'
            )
        _t0 = _time.monotonic()
        try:
            jobiface.engine_recieve_socket.send_json(msg)
            reply = jobiface.engine_recieve_socket.recv_json()
            _elapsed = _time.monotonic() - _t0
            if _elapsed > 5.0:
                logger.warning('[job_engine] Slow engine response: %.1fs for action=%r',
                               _elapsed, action)
            return reply
        except zmq.error.Again:
            _elapsed = _time.monotonic() - _t0
            logger.error('[job_engine] Engine TIMEOUT after %.1fs for action=%r '
                         '(thread=%s). Resetting socket.',
                         _elapsed, action, threading.current_thread().name)
            jobiface._reset_engine_socket()
            raise RuntimeError(
                'Job engine did not respond within the timeout period. '
                'The engine process may have crashed or is overloaded.'
            )
        finally:
            jobiface._engine_lock.release()

    def _collect_request(jobiface, msg):
        """Send a message to the collector and return the reply (thread-safe)."""
        import time as _time

        action = msg.get('action', 'unknown')
        jobiface._ensure_sockets()
        # Use a timeout on the lock so threads don't pile up behind a slow
        # request.  If another thread is already waiting on recv_json(),
        # fail fast rather than queueing indefinitely.
        acquired = jobiface._collect_lock.acquire(timeout=10)
        if not acquired:
            logger.warning('[job_engine] Collector lock contention — could not acquire '
                           'within 10s for action=%r (thread=%s)',
                           action, threading.current_thread().name)
            raise RuntimeError(
                'Could not acquire collector lock within 10s. '
                'Another request is likely waiting on the collector.'
            )
        _t0 = _time.monotonic()
        try:
            jobiface.collect_recieve_socket.send_json(msg)
            reply = jobiface.collect_recieve_socket.recv_json()
            _elapsed = _time.monotonic() - _t0
            if _elapsed > 5.0:
                logger.warning('[job_engine] Slow collector response: %.1fs for action=%r',
                               _elapsed, action)
            return reply
        except zmq.error.Again:
            _elapsed = _time.monotonic() - _t0
            logger.error('[job_engine] Collector TIMEOUT after %.1fs for action=%r '
                         '(thread=%s). Resetting socket.',
                         _elapsed, action, threading.current_thread().name)
            jobiface._reset_collect_socket()
            raise RuntimeError(
                'Job collector did not respond within the timeout period. '
                'The collector process may have crashed or is overloaded.'
            )
        finally:
            jobiface._collect_lock.release()

    def _reset_engine_socket(jobiface):
        """Close and nullify the engine socket so _ensure_sockets recreates it."""
        try:
            if jobiface.engine_recieve_socket is not None:
                jobiface.engine_recieve_socket.close(linger=0)
        except Exception:
            pass
        jobiface.engine_recieve_socket = None

    def _reset_collect_socket(jobiface):
        """Close and nullify the collect socket so _ensure_sockets recreates it."""
        try:
            if jobiface.collect_recieve_socket is not None:
                jobiface.collect_recieve_socket.close(linger=0)
        except Exception:
            pass
        jobiface.collect_recieve_socket = None

    def __del__(jobiface):
        if VERBOSE_JOBS:
            print('Cleaning up job frontend')
        # Best-effort cleanup — skip if lock not acquired to avoid both
        # hanging at shutdown and racing with active request threads.
        for attr, lock_attr, url_key in [
            ('engine_recieve_socket', '_engine_lock', 'engine_pull_url'),
            ('collect_recieve_socket', '_collect_lock', 'collect_pull_url'),
        ]:
            try:
                lock = getattr(jobiface, lock_attr, None)
                sock = getattr(jobiface, attr, None)
                if sock is None:
                    continue
                acquired = lock.acquire(blocking=False) if lock else False
                if not acquired:
                    continue  # another thread owns the socket — let it clean up
                try:
                    sock.disconnect(jobiface.port_dict[url_key])
                    sock.close(linger=0)
                finally:
                    lock.release()
            except Exception:
                pass

    # def init(jobiface):
    #     # Starts several new processes
    #     jobiface.initialize_background_processes()
    #     # Does not create a new process, but connects sockets on this process
    #     jobiface.initialize_client_thread()

    def close_client_sockets(jobiface):
        """Close ZMQ client sockets and context (call before fork)."""
        if jobiface.engine_recieve_socket is not None:
            jobiface.engine_recieve_socket.close()
            jobiface.engine_recieve_socket = None
        if jobiface.collect_recieve_socket is not None:
            jobiface.collect_recieve_socket.close()
            jobiface.collect_recieve_socket = None
        zmq_ctx = getattr(jobiface, '_zmq_ctx', None)
        if zmq_ctx is not None:
            zmq_ctx.term()
            jobiface._zmq_ctx = None

    def _rebuild_sockets(jobiface):
        """Recreate whichever ZMQ sockets are missing (None).

        Called by _ensure_sockets after a timeout-triggered reset destroyed
        one socket, or on first use when both are None.  Only touches the
        socket(s) that need rebuilding; leaves the other untouched.
        """
        # Ensure we have a ZMQ context
        if getattr(jobiface, '_zmq_ctx', None) is None:
            jobiface._zmq_ctx = zmq.Context()

        if jobiface.engine_recieve_socket is None:
            jobiface.engine_recieve_socket = jobiface._zmq_ctx.socket(zmq.DEALER)
            jobiface.engine_recieve_socket.setsockopt_string(
                zmq.IDENTITY, 'client{}.engine.DEALER'.format(jobiface.id_)
            )
            jobiface.engine_recieve_socket.setsockopt(zmq.RCVTIMEO, _ENGINE_RCVTIMEO_MS)
            jobiface.engine_recieve_socket.setsockopt(zmq.LINGER, 0)
            jobiface.engine_recieve_socket.connect(jobiface.port_dict['engine_pull_url'])

        if jobiface.collect_recieve_socket is None:
            jobiface.collect_recieve_socket = jobiface._zmq_ctx.socket(zmq.DEALER)
            jobiface.collect_recieve_socket.setsockopt_string(
                zmq.IDENTITY, 'client{}.collect.DEALER'.format(jobiface.id_)
            )
            jobiface.collect_recieve_socket.setsockopt(zmq.RCVTIMEO, _ENGINE_RCVTIMEO_MS)
            jobiface.collect_recieve_socket.setsockopt(zmq.LINGER, 0)
            jobiface.collect_recieve_socket.connect(jobiface.port_dict['collect_pull_url'])

    def initialize_client_thread(jobiface):
        """
        Creates ZMQ sockets for talking to background engine/collector processes.
        Used during startup (pre-fork) for one-time initialization tasks.
        After fork(), _rebuild_sockets() is used instead.
        """
        if jobiface.verbose:
            print('Initializing JobInterface')
        # Create a fresh ZMQ context for this process (critical after fork)
        jobiface._zmq_ctx = zmq.Context()
        jobiface.engine_recieve_socket = jobiface._zmq_ctx.socket(zmq.DEALER)
        jobiface.engine_recieve_socket.setsockopt_string(
            zmq.IDENTITY, 'client{}.engine.DEALER'.format(jobiface.id_)
        )
        jobiface.engine_recieve_socket.setsockopt(zmq.RCVTIMEO, _ENGINE_RCVTIMEO_MS)
        jobiface.engine_recieve_socket.setsockopt(zmq.LINGER, 0)
        jobiface.engine_recieve_socket.connect(jobiface.port_dict['engine_pull_url'])
        if jobiface.verbose:
            print(
                'connect engine_pull_url = {!r}'.format(
                    jobiface.port_dict['engine_pull_url']
                )
            )

        jobiface.collect_recieve_socket = jobiface._zmq_ctx.socket(zmq.DEALER)
        jobiface.collect_recieve_socket.setsockopt_string(
            zmq.IDENTITY, 'client{}.collect.DEALER'.format(jobiface.id_)
        )
        jobiface.collect_recieve_socket.setsockopt(zmq.RCVTIMEO, _ENGINE_RCVTIMEO_MS)
        jobiface.collect_recieve_socket.setsockopt(zmq.LINGER, 0)
        jobiface.collect_recieve_socket.connect(jobiface.port_dict['collect_pull_url'])
        if jobiface.verbose:
            print(
                'connect collect_pull_url = %r'
                % (jobiface.port_dict['collect_pull_url'],)
            )

    def queue_interrupted_jobs(jobiface):
        import tqdm

        from wbia.web.job_store import JobStore

        ibs = jobiface.ibs

        if ibs is not None:
            shelve_path = ibs.get_shelves_path()
            shelve_path = shelve_path.rstrip('/')
            shelve_archive_path = '{}_ARCHIVE'.format(shelve_path)
            ut.ensuredir(shelve_archive_path)

            # Open a JobStore for startup recovery (read + batch register).
            # The collector will open its own instance once it finishes starting.
            job_store = JobStore(join(shelve_path, 'jobs.db'))
            try:
                record_filepath_list = _get_engine_job_paths(ibs)

                num_records = len(record_filepath_list)
                print('Reloading %d engine jobs...' % (num_records,))

                arg_iter = list(
                    zip(
                        record_filepath_list,
                        [shelve_path] * num_records,
                        [shelve_archive_path] * num_records,
                        [jobiface.id_] * num_records,
                        [job_store] * num_records,
                    )
                )
                # Run sequentially — JobStore is not fork-safe
                values_list = []
                for args in arg_iter:
                    values_list.append(initialize_process_record(*args))

                print('Processed %d records' % (len(values_list),))

                restart_jobcounter_list = []
                restart_jobid_list = []
                restart_request_list = []

                global_jobcounter = 0
                num_registered, num_restarted = 0, 0
                num_completed, num_archived, num_suppressed, num_corrupted = 0, 0, 0, 0

                # Build batch of jobs to register
                batch_register = []  # list of (jobid, status, jobcounter)

                for values in tqdm.tqdm(values_list):
                    (
                        jobcounter,
                        jobid,
                        engine_request,
                        archived,
                        completed,
                        suppressed,
                        corrupted,
                    ) = values

                    if archived:
                        assert engine_request is None
                        num_archived += 1
                        continue

                    if jobcounter is not None:
                        global_jobcounter = max(global_jobcounter, jobcounter)

                    if engine_request is None:
                        assert not archived
                        if completed:
                            status = 'completed'
                            num_completed += 1
                        elif suppressed:
                            status = 'suppressed'
                            num_suppressed += 1
                        else:
                            status = 'corrupted'
                            num_corrupted += 1

                        batch_register.append((jobid, status, jobcounter))
                    else:
                        num_restarted += 1
                        restart_jobcounter_list.append(jobcounter)
                        restart_jobid_list.append(jobid)
                        restart_request_list.append(engine_request)

                    num_registered += 1

                assert num_restarted == len(restart_jobcounter_list)

                # Write directly to the local JobStore instead of routing
                # through ZMQ.  The collector subprocess may still be starting
                # (loading the database), so a ZMQ send here can easily time out
                # with thousands of legacy jobs.  SQLite WAL mode allows this
                # concurrent writer safely.
                if batch_register:
                    print('Batch registering %d jobs directly to SQLite...' % (len(batch_register),))
                    job_store.register_batch([
                        {'jobid': jid, 'status': st, 'jobcounter': jc}
                        for jid, st, jc in batch_register
                    ])
                    print('Batch register complete: %d jobs registered' % (len(batch_register),))

                print('Registered %d jobs...' % (num_registered,))
                print('\t %d completed jobs' % (num_completed,))
                print('\t %d restarted jobs' % (num_restarted,))
                print('\t %d suppressed jobs' % (num_suppressed,))
                print('\t %d corrupted jobs' % (num_corrupted,))
                print('Archived %d jobs...' % (num_archived,))

                # Reclaim space from archived jobs
                if num_archived > 0:
                    try:
                        job_store.vacuum()
                    except Exception:
                        pass
            finally:
                job_store.close()

            # Update the jobcounter to be up to date
            update_notify = {
                '__set_jobcounter__': global_jobcounter,
            }
            print('Updating completed job counter: {!r}'.format(update_notify))
            reply = jobiface._engine_request(update_notify)
            jobcounter_ = reply['jobcounter']
            assert jobcounter_ == global_jobcounter

            print('Re-sending %d engine jobs...' % (len(restart_jobcounter_list),))

            index_list = np.argsort(restart_jobcounter_list)
            zipped = list(
                zip(restart_jobcounter_list, restart_jobid_list, restart_request_list)
            )
            zipped = ut.take(zipped, index_list)

            for jobcounter, jobid, engine_request in tqdm.tqdm(zipped):
                reply = jobiface._engine_request(engine_request)
                jobcounter_ = reply['jobcounter']
                jobid_ = reply['jobid']
                assert jobcounter_ == jobcounter
                assert jobid_ == jobid

    def queue_job(
        jobiface,
        action,
        callback_url=None,
        callback_method=None,
        callback_detailed=False,
        lane='slow',
        jobid=None,
        args=None,
        kwargs=None,
    ):
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
        if jobiface.verbose >= 1:
            print('----')
        request = {}
        try:
            if flask.request:
                request = {
                    'endpoint': flask.request.path,
                    'function': flask.request.endpoint,
                    'input': flask.request.processed,
                }
        except RuntimeError:
            pass

        if args is None:
            args = tuple([])

        if kwargs is None:
            kwargs = {}

        if jobid is not None:
            assert isinstance(jobid, str)
            try:
                jobid_ = uuid.UUID(jobid)
                assert jobid_ is not None and isinstance(jobid_, uuid.UUID)
            except Exception:
                assert len(jobid) < 32, 'Job IDs cannot be more than 32 characters'
                pattern = r'^[a-zA-Z0-9-_]+$'
                matched = bool(re.match(pattern, jobid))
                assert matched, 'Job IDs must be alpha-numeric'
            jobid = str(jobid_)

        engine_request = {
            'action': action,
            'args': args,
            'kwargs': kwargs,
            'callback_url': callback_url,
            'callback_method': callback_method,
            'callback_detailed': callback_detailed,
            'request': request,
            'restart_jobid': jobid,
            'restart_jobcounter': None,
            'restart_received': None,
            'lane': lane,
        }
        if True or jobiface.verbose >= 2:
            print('Queue job: {}'.format(ut.repr2(engine_request, truncate=True)))

        # Send request to job engine
        reply_notify = jobiface._engine_request(engine_request)
        print('reply_notify = {!r}'.format(reply_notify))
        jobid_ = reply_notify['jobid']

        if jobid is not None:
            assert jobid == jobid_
        jobid = jobid_

        ibs = jobiface.ibs
        if ibs is not None:
            shelve_path = ibs.get_shelves_path()
            ut.ensuredir(shelve_path)

            record_filename = '{}.pkl'.format(jobid)
            record_filepath = join(shelve_path, record_filename)
            record = {
                'request': engine_request,
                'attempts': 0,
                'completed': False,
            }
            _save_engine_record(record_filepath, record)

        # Release memory
        action = None
        args = None
        kwargs = None
        callback_url = None
        callback_method = None
        callback_detailed = None
        request = None
        engine_request = None

        return jobid

    def get_job_id_list(jobiface):
        """Read job ID list directly from SQLite (no ZMQ round-trip)."""
        store = jobiface._get_reader_store()
        return {'status': 'ok', 'jobid_list': store.get_job_ids()}

    def get_job_status(jobiface, jobid):
        """Read job status directly from SQLite (no ZMQ round-trip)."""
        store = jobiface._get_reader_store()
        status = store.get_status(jobid) or 'unknown'
        return {'status': 'ok', 'jobid': jobid, 'jobstatus': status}

    def get_job_status_dict(jobiface, limit=0):
        """Read all job statuses directly from SQLite (no ZMQ round-trip)."""
        store = jobiface._get_reader_store()
        return {
            'status': 'ok',
            'json_result': store.get_job_status_dict(limit=limit),
        }

    def get_job_metadata(jobiface, jobid):
        """Read job metadata directly from SQLite (no ZMQ round-trip)."""
        store = jobiface._get_reader_store()
        if not store.job_exists(jobid):
            return {'status': 'invalid', 'jobid': jobid, 'json_result': None}
        metadata = store.get_metadata(jobid)
        reply = {'status': 'ok', 'jobid': jobid, 'json_result': metadata}
        if metadata is None:
            # Skeleton row from startup batch-register — metadata was never
            # migrated.  Report as 'working' so callers retry rather than
            # treating it as corrupted.
            status = store.get_status(jobid)
            reply['status'] = 'corrupted' if status == 'corrupted' else 'working'
        return reply

    def get_job_result(jobiface, jobid):
        """Read job result directly from SQLite (no ZMQ round-trip)."""
        store = jobiface._get_reader_store()
        if not store.job_exists(jobid):
            return {'status': 'invalid', 'jobid': jobid, 'json_result': None}
        status = store.get_status(jobid)
        result_data = store.get_result(jobid)
        if result_data is None:
            if status == 'completed':
                # The job row has status='completed' but no json_result.
                # This can happen when: (a) the job was recovered at startup
                # from a .pkl file and batch-registered with status only
                # (result data is not migrated to SQLite), or (b) a stale
                # WAL snapshot caused get_status() and get_result() to see
                # different points in time.  Report as 'working' so callers
                # retry instead of treating the job as corrupted.
                reply_status = 'working'
            elif status == 'corrupted':
                reply_status = 'corrupted'
            elif status == 'suppressed':
                reply_status = 'suppressed'
            else:
                reply_status = status
            return {'status': reply_status, 'jobid': jobid, 'json_result': None}
        return {
            'status': result_data['exec_status'],
            'jobid': jobid,
            'json_result': ut.from_json(result_data['json_result']),
        }

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
        # Release raw JSON result
        json_result = None
        return result

    def wait_for_job_result(jobiface, jobid, timeout=10, freq=0.1):
        t = ut.Timer(verbose=False)
        t.tic()
        while True:
            reply = jobiface.get_job_status(jobid)
            if reply['jobstatus'] == 'completed':
                return
            elif reply['jobstatus'] == 'exception':
                result = jobiface.get_unpacked_result(jobid)
                # raise Exception(result)
                print('Exception occured in engine')
                return result
            elif reply['jobstatus'] == 'working':
                pass
            elif reply['jobstatus'] == 'unknown':
                pass
            else:
                raise Exception('Unknown jobstatus={!r}'.format(reply['jobstatus']))
            reply = None  # Release memory
            time.sleep(freq)
            if timeout is not None and t.toc() > timeout:
                raise Exception('Timeout')


def collect_queue_loop(port_dict):
    name = 'collect'
    assert name is not None, 'must name queue'
    queue_name = name + '_queue'
    loop_name = queue_name + '_loop'

    update_proctitle(queue_name)

    interface_pull = port_dict['{}_pull_url'.format(name)]
    interface_push = port_dict['{}_push_url'.format(name)]

    if VERBOSE_JOBS:
        print('Init make_queue_loop: name={!r}'.format(name))
    # bind the client dealer to the queue router
    recieve_socket = _get_global_zmq_ctx().socket(zmq.ROUTER)  # CHECKED - ROUTER
    recieve_socket.setsockopt_string(zmq.IDENTITY, 'queue.' + name + '.' + 'ROUTER')
    recieve_socket.bind(interface_pull)
    if VERBOSE_JOBS:
        print('bind {}_url1 = {!r}'.format(name, interface_pull))

    # bind the server router to the queue dealer
    send_socket = _get_global_zmq_ctx().socket(zmq.DEALER)  # CHECKED - DEALER
    send_socket.setsockopt_string(zmq.IDENTITY, 'queue.' + name + '.' + 'DEALER')
    send_socket.bind(interface_push)
    if VERBOSE_JOBS:
        print('bind {}_url2 = {!r}'.format(name, interface_push))

    try:
        zmq.device(zmq.QUEUE, recieve_socket, send_socket)  # CHECKED - QUEUE
    except KeyboardInterrupt:
        print('Caught ctrl+c in queue loop. Gracefully exiting')

    recieve_socket.unbind(interface_pull)
    recieve_socket.close()
    send_socket.unbind(interface_push)
    send_socket.close()

    if VERBOSE_JOBS:
        print('Exiting {}'.format(loop_name))


def engine_queue_loop(port_dict, engine_lanes):
    """
    Specialized queue loop
    """
    # Flow of information tags:
    # NAME: engine_queue
    name = 'engine'
    queue_name = name + '_queue'
    loop_name = queue_name + '_loop'
    update_proctitle(queue_name)

    print = partial(ut.colorprint, color='red')

    interface_engine_pull = port_dict['engine_pull_url']
    interface_engine_push_dict = {
        lane: port_dict['engine_{}_push_url'.format(lane)] for lane in engine_lanes
    }
    interface_collect_pull = port_dict['collect_pull_url']

    print('Init specialized make_queue_loop: name={!r}'.format(name))

    # bind the client dealer to the queue router
    engine_receive_socket = _get_global_zmq_ctx().socket(zmq.ROUTER)  # CHECK2 - REP
    engine_receive_socket.setsockopt_string(
        zmq.IDENTITY, 'special_queue.' + name + '.' + 'ROUTER'
    )
    engine_receive_socket.bind(interface_engine_pull)
    if VERBOSE_JOBS:
        print('bind {}_url2 = {!r}'.format(name, interface_engine_pull))

    # bind the server router to the queue dealer
    engine_send_socket_dict = {}
    for lane in interface_engine_push_dict:
        engine_send_socket = _get_global_zmq_ctx().socket(zmq.DEALER)  # CHECKED - DEALER
        engine_send_socket.setsockopt_string(
            zmq.IDENTITY, 'special_queue.' + lane + '.' + name + '.' + 'DEALER'
        )
        engine_send_socket.bind(interface_engine_push_dict[lane])
        if VERBOSE_JOBS:
            print(
                'bind {} {}_url2 = {!r}'.format(
                    name, lane, interface_engine_push_dict[lane]
                )
            )
        engine_send_socket_dict[lane] = engine_send_socket

    collect_recieve_socket = _get_global_zmq_ctx().socket(zmq.DEALER)  # CHECKED - DEALER
    collect_recieve_socket.setsockopt_string(zmq.IDENTITY, queue_name + '.collect.DEALER')
    collect_recieve_socket.connect(interface_collect_pull)
    if VERBOSE_JOBS:
        print('connect collect_pull_url = %r' % (interface_collect_pull))

    # but this shows what is really going on:
    poller = zmq.Poller()
    poller.register(engine_receive_socket, zmq.POLLIN)
    for lane in engine_send_socket_dict:
        engine_send_socket = engine_send_socket_dict[lane]
        poller.register(engine_send_socket, zmq.POLLIN)

    # always start at 0
    global_jobcounter = 0

    try:
        while True:
            evts = dict(poller.poll())
            if engine_receive_socket in evts:
                # CALLER: job_client
                idents, engine_request = rcv_multipart_json(
                    engine_receive_socket, num=1, print=print
                )

                set_jobcounter = engine_request.get('__set_jobcounter__', None)
                if set_jobcounter is not None:
                    global_jobcounter = set_jobcounter
                    reply_notify = {
                        'jobcounter': global_jobcounter,
                    }
                    print(
                        '... notifying client that jobcounter was updated to %d'
                        % (global_jobcounter,)
                    )
                    # RETURNS: job_client_return
                    send_multipart_json(engine_receive_socket, idents, reply_notify)
                    continue

                # jobid = 'jobid-%04d' % (jobcounter,)
                jobid = '{}'.format(uuid.uuid4())
                jobcounter = global_jobcounter + 1
                received = _timestamp()

                action = engine_request['action']
                args = engine_request['args']
                kwargs = engine_request['kwargs']
                callback_url = engine_request['callback_url']
                callback_method = engine_request['callback_method']
                callback_detailed = engine_request.get('callback_detailed', False)
                request = engine_request['request']
                restart_jobid = engine_request.get('restart_jobid', None)
                restart_jobcounter = engine_request.get('restart_jobcounter', None)
                restart_received = engine_request.get('restart_received', None)
                lane = engine_request.get('lane', 'slow')

                if lane not in engine_lanes:
                    print(
                        'WARNING: did not recognize desired lane %r from %r'
                        % (lane, engine_lanes)
                    )
                    print('WARNING: Defaulting to slow lane')
                    lane = 'slow'

                engine_request['lane'] = lane

                if restart_jobid is not None:
                    '[RESTARTING] Replacing jobid={} with previous restart_jobid={}'.format(
                        jobid,
                        restart_jobid,
                    )
                    jobid = restart_jobid

                if restart_jobcounter is not None:
                    '[RESTARTING] Replacing jobcounter={} with previous restart_jobcounter={}'.format(
                        jobcounter,
                        restart_jobcounter,
                    )
                    jobcounter = restart_jobcounter

                print('Creating jobid %r (counter %d)' % (jobid, jobcounter))

                if restart_received is not None:
                    received = restart_received

                ######################################################################
                # Status: Received (Notify Collector)
                # Reply immediately with a new jobid
                reply_notify = {
                    'jobid': jobid,
                    'jobcounter': jobcounter,
                    'status': 'received',
                    'action': 'notification',
                }

                if VERBOSE_JOBS:
                    print('...notifying collector about new job')
                # CALLS: collector_notify
                collect_recieve_socket.send_json(reply_notify)

                ######################################################################
                # Status: Received (Notify Client)
                if VERBOSE_JOBS:
                    print('... notifying client that job was accepted')
                    print('{!r}'.format(idents))
                    print('{!r}'.format(reply_notify))
                # RETURNS: job_client_return
                send_multipart_json(engine_receive_socket, idents, reply_notify)

                ######################################################################
                # Status: Metadata

                # Reply immediately with a new jobid
                metadata_notify = {
                    'jobid': jobid,
                    'metadata': {
                        'jobcounter': jobcounter,
                        'action': action,
                        'args': args,
                        'kwargs': kwargs,
                        'callback_url': callback_url,
                        'callback_method': callback_method,
                        'callback_detailed': callback_detailed,
                        'request': request,
                        'times': {
                            'received': received,
                            'started': None,
                            'updated': None,
                            'completed': None,
                            'runtime': None,
                            'turnaround': None,
                            'runtime_sec': None,
                            'turnaround_sec': None,
                        },
                        'lane': lane,
                    },
                    'action': 'metadata',
                }

                if VERBOSE_JOBS:
                    print('...notifying collector about job metadata')
                # CALLS: collector_notify
                collect_recieve_socket.send_json(metadata_notify)

                ######################################################################
                # Status: Accepted (Metadata Processed)

                # We have been accepted, let's update the global_jobcounter
                global_jobcounter = jobcounter

                # Reply immediately with a new jobid
                reply_notify = {
                    'jobid': jobid,
                    'status': 'accepted',
                    'action': 'notification',
                }

                if VERBOSE_JOBS:
                    print('...notifying collector about new job')
                # CALLS: collector_notify
                collect_recieve_socket.send_json(reply_notify)

                ######################################################################
                # Status: Queueing on the Engine
                assert 'jobid' not in engine_request
                engine_request['jobid'] = jobid

                if VERBOSE_JOBS:
                    print('... notifying backend engine to start')
                # CALL: engine_
                engine_send_socket = engine_send_socket_dict[lane]
                send_multipart_json(engine_send_socket, idents, engine_request)

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
                collect_recieve_socket.send_json(queued_notify)
    except KeyboardInterrupt:
        print('Caught ctrl+c in {} queue. Gracefully exiting'.format(loop_name))

    poller.unregister(engine_receive_socket)
    for lane in engine_send_socket_dict:
        engine_send_socket = engine_send_socket_dict[lane]
        poller.unregister(engine_send_socket)

    engine_receive_socket.unbind(interface_engine_pull)
    engine_receive_socket.close()

    for lane in interface_engine_push_dict:
        engine_send_socket = engine_send_socket_dict[lane]
        engine_send_socket.unbind(interface_engine_push_dict[lane])
        engine_send_socket.close()

    collect_recieve_socket.disconnect(interface_collect_pull)
    collect_recieve_socket.close()

    if VERBOSE_JOBS:
        print('Exiting {} queue'.format(loop_name))


def engine_loop(id_, port_dict, dbdir, containerized, lane):
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
    import wbia

    # try:
    #     import tensorflow as tf  # NOQA
    #     from keras import backend as K  #ß NOQA

    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     sess = tf.Session(config=config)
    #     K.set_session(sess)
    # except (ImportError, RuntimeError):
    #     pass

    # base_print = print  # NOQA
    print = partial(ut.colorprint, color='brightgreen')
    interface_engine_push = port_dict['engine_{}_push_url'.format(lane)]
    interface_collect_pull = port_dict['collect_pull_url']

    if VERBOSE_JOBS:
        print('Initializing {} engine {}'.format(lane, id_))
        print('connect engine_{}_push_url = {!r}'.format(lane, interface_engine_push))

    assert dbdir is not None

    engine_send_sock = _get_global_zmq_ctx().socket(zmq.ROUTER)  # CHECKED - ROUTER
    engine_send_sock.setsockopt_string(
        zmq.IDENTITY,
        'engine.{}.{}'.format(lane, id_),
    )
    engine_send_sock.connect(interface_engine_push)

    collect_recieve_socket = _get_global_zmq_ctx().socket(zmq.DEALER)
    collect_recieve_socket.setsockopt_string(
        zmq.IDENTITY,
        'engine.{}.{}.collect.DEALER'.format(lane, id_),
    )
    collect_recieve_socket.connect(interface_collect_pull)

    if VERBOSE_JOBS:
        print('connect collect_pull_url = {!r}'.format(interface_collect_pull))
        print('engine is initialized')

    ibs = wbia.opendb(dbdir=dbdir, use_cache=False, web=False, daily_backup=False)
    update_proctitle('engine_loop.{}.{}'.format(lane, id_), dbname=ibs.dbname)

    try:
        while True:
            try:
                idents, engine_request = rcv_multipart_json(engine_send_sock, print=print)

                action = engine_request['action']
                jobid = engine_request['jobid']
                args = engine_request['args']
                kwargs = engine_request['kwargs']
                callback_url = engine_request['callback_url']
                callback_method = engine_request['callback_method']
                callback_detailed = engine_request.get('callback_detailed', False)
                lane_ = engine_request['lane']

                if VERBOSE_JOBS:
                    print('\tjobid = {!r}'.format(jobid))
                    print('\taction = {!r}'.format(action))
                    print('\targs = {!r}'.format(args))
                    print('\tkwargs = {!r}'.format(kwargs))
                    print('\tlane = {!r}'.format(lane))
                    print('\tlane_ = {!r}'.format(lane_))

                # Notify start working
                reply_notify = {
                    # 'idents': idents,
                    'jobid': jobid,
                    'status': 'working',
                    'action': 'notification',
                }
                collect_recieve_socket.send_json(reply_notify)

                engine_result = on_engine_request(ibs, jobid, action, args, kwargs)
                exec_status = engine_result['exec_status']

                # Notify start working
                reply_notify = {
                    # 'idents': idents,
                    'jobid': jobid,
                    'status': 'publishing',
                    'action': 'notification',
                }
                collect_recieve_socket.send_json(reply_notify)

                # Store results in the collector
                collect_request = {
                    # 'idents': idents,
                    'action': 'store',
                    'jobid': jobid,
                    'engine_result': engine_result,
                    'callback_url': callback_url,
                    'callback_method': callback_method,
                    'callback_detailed': callback_detailed,
                }
                # if VERBOSE_JOBS:
                print(
                    '...done working. pushing result to collector for jobid {}'.format(
                        jobid
                    )
                )

                # CALLS: collector_store
                collect_recieve_socket.send_json(collect_request)

                # Notify start working
                reply_notify = {
                    # 'idents': idents,
                    'jobid': jobid,
                    'status': exec_status,
                    'action': 'notification',
                }
                collect_recieve_socket.send_json(reply_notify)

                # We no longer need the engine result, and can clear it's memory
                engine_request = None
                engine_result = None
                collect_request = None
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                result = ut.formatex(ex, keys=['jobid'], tb=True)
                result = ut.strip_ansi(result)
                print_ = partial(ut.colorprint, color='brightred')
                print_(result)
                raise
    except KeyboardInterrupt:
        print('Caught ctrl+c in engine loop. Gracefully exiting')

    engine_send_sock.disconnect(interface_engine_push)
    engine_send_sock.close()
    collect_recieve_socket.disconnect(interface_collect_pull)
    collect_recieve_socket.close()

    # Release the IBEIS controller for each job, hopefully freeing memory
    ibs = None

    # Explicitly try to release GPU memory
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    # Explicitly release Python memory
    try:
        import gc

        gc.collect()
    except Exception:
        pass

    # ----
    if VERBOSE_JOBS:
        print('Exiting engine loop')


def on_engine_request(
    ibs, jobid, action, args, kwargs, attempts=3, retry_delay_min=1, retry_delay_max=60
):
    """Run whenever the engine recieves a message"""
    assert attempts > 0
    attempts = int(attempts)

    assert 0 <= retry_delay_min and retry_delay_min <= 60 * 60
    retry_delay_min = int(retry_delay_min)
    assert 0 <= retry_delay_max and retry_delay_max <= 60 * 60
    retry_delay_max = int(retry_delay_max)
    assert retry_delay_min < retry_delay_max

    # Start working
    if VERBOSE_JOBS:
        print('starting job={!r}'.format(jobid))
    # Map actions to IBEISController calls here
    if action == 'helloworld':

        def helloworld(time_=0, *args, **kwargs):
            time.sleep(time_)
            retval = ('HELLO time_={!r} '.format(time_)) + ut.repr2((args, kwargs))
            return retval

        action_func = helloworld
    else:
        # check for ibs func
        action_func = getattr(ibs, action)
        if VERBOSE_JOBS:
            print(
                'resolving action={!r} to wbia function={!r}'.format(action, action_func)
            )

    key = '__jobid__'
    kwargs[key] = jobid
    exec_status = None
    while exec_status is None:
        try:
            attempt = 0
            while attempt < 10:  # Global max attempts of 10
                attempt += 1
                try:
                    result = action_func(*args, **kwargs)
                    break  # success, no exception, break out of the loop
                except Exception:
                    if attempt < attempts:
                        print(
                            'JOB %r FAILED (attempt %d of %d)!'
                            % (jobid, attempt, attempts)
                        )
                        retry_delay = random.uniform(retry_delay_min, retry_delay_max)
                        print(
                            '\t WAITING {:0.02f} SECONDS THEN RETRYING'.format(
                                retry_delay
                            )
                        )
                        time.sleep(retry_delay)
                    else:
                        raise
            exec_status = 'completed'
        except Exception as ex:
            # Remove __jobid__ from kwargs if it's not accepted by the action_func
            if key in kwargs:
                kwargs.pop(key, None)
                continue
            result = ut.formatex(ex, keys=['jobid'], tb=True)
            result = ut.strip_ansi(result)
            exec_status = 'exception'
    json_result = ut.to_json(result)
    result = None  # Clear any used memory
    engine_result = {
        'exec_status': exec_status,
        'json_result': json_result,
        'jobid': jobid,
    }
    return engine_result


def collector_loop(port_dict, dbdir, containerized):
    """
    Service that stores completed algorithm results.

    Uses SQLite WAL-mode (via JobStore) instead of per-job shelve files.
    """
    import wbia

    from wbia.web.job_store import JobStore

    print = partial(ut.colorprint, color='yellow')
    collect_rout_sock = _get_global_zmq_ctx().socket(zmq.ROUTER)  # CHECK2 - PULL
    collect_rout_sock.setsockopt_string(zmq.IDENTITY, 'collect.ROUTER')
    collect_rout_sock.connect(port_dict['collect_push_url'])
    if VERBOSE_JOBS:
        print('connect collect_push_url  = {!r}'.format(port_dict['collect_push_url']))

    ibs = wbia.opendb(dbdir=dbdir, use_cache=False, web=False, daily_backup=False)
    update_proctitle('collector_loop', dbname=ibs.dbname)

    shelve_path = ibs.get_shelves_path()
    ut.ensuredir(shelve_path)

    job_store = JobStore(join(shelve_path, 'jobs.db'))

    try:
        while True:
            idents, collect_request = rcv_multipart_json(collect_rout_sock, print=print)
            _action = collect_request.get('action', 'unknown') if collect_request else 'unknown'
            _t0 = time.monotonic()
            try:
                reply = on_collect_request(
                    ibs,
                    collect_request,
                    job_store,
                    shelve_path,
                    containerized=containerized,
                )
            except Exception as ex:
                import traceback

                print(ut.repr3(collect_request))
                ut.printex(ex, 'ERROR in collection')
                print(traceback.format_exc())
                reply = {}

            _elapsed = time.monotonic() - _t0
            if _elapsed > 2.0:
                print('SLOW collector action=%r took %.1fs' % (_action, _elapsed))

            send_multipart_json(collect_rout_sock, idents, reply)

            idents = None
            collect_request = None
    except KeyboardInterrupt:
        print('Caught ctrl+c in collector loop. Gracefully exiting')

    job_store.close()
    collect_rout_sock.disconnect(port_dict['collect_push_url'])
    collect_rout_sock.close()

    if VERBOSE_JOBS:
        print('Exiting collector')


def _fire_callback(callback_url, callback_method, data_dict, print=print):
    """Send a job-completion callback in a daemon thread.

    Runs the HTTP request off the collector's main loop so that slow or
    unreachable callback targets never block ZMQ message processing.
    """
    import requests

    _CB_TIMEOUT = (10, 30)  # (connect, read) seconds

    jobid = data_dict.get('jobid', '?')

    def _do_callback():
        try:
            print(
                'Attempting callback for jobid=%s to %r via %s'
                % (jobid, callback_url, callback_method)
            )

            if callback_method == 'POST':
                if callback_url.startswith('houston+'):
                    response = call_houston(
                        callback_url,
                        method='POST',
                        data=ut.to_json(data_dict),
                        headers={'Content-Type': 'application/json'},
                        timeout=_CB_TIMEOUT,
                    )
                else:
                    response = requests.post(callback_url, data=data_dict, timeout=_CB_TIMEOUT)
            elif callback_method == 'GET':
                if callback_url.startswith('houston+'):
                    response = call_houston(
                        callback_url, method='GET', params=data_dict,
                        timeout=_CB_TIMEOUT,
                    )
                else:
                    response = requests.get(callback_url, params=data_dict, timeout=_CB_TIMEOUT)
            elif callback_method == 'PUT':
                if callback_url.startswith('houston+'):
                    response = call_houston(
                        callback_url,
                        method='PUT',
                        data=ut.to_json(data_dict),
                        headers={'Content-Type': 'application/json'},
                        timeout=_CB_TIMEOUT,
                    )
                else:
                    response = requests.put(callback_url, data=data_dict, timeout=_CB_TIMEOUT)
            else:
                raise RuntimeError('Unsupported callback method: {!r}'.format(callback_method))

            status_code = getattr(response, 'status_code', None)
            print(
                'Callback completed for jobid=%s — HTTP %s' % (jobid, status_code)
            )
            if status_code is not None and status_code >= 400:
                try:
                    body = response.text[:500]
                except Exception:
                    body = '(unreadable)'
                print(
                    'Callback WARNING for jobid=%s: HTTP %s — %s'
                    % (jobid, status_code, body)
                )
        except Exception as ex:
            import traceback
            print(
                'Callback FAILED for jobid=%s to %r: %s\n%s'
                % (jobid, callback_url, ex, traceback.format_exc())
            )

    t = threading.Thread(target=_do_callback, daemon=True)
    t.start()


def _timestamp():
    timezone = pytz.timezone(TIMESTAMP_TIMEZONE)
    now = datetime.now(timezone)
    timestamp = now.strftime(TIMESTAMP_FMTSTR)
    return timestamp


def convert_to_date(timestamp):
    TIMESTAMP_FMTSTR_ = ' '.join(TIMESTAMP_FMTSTR.split(' ')[:-1])
    timestamp_ = ' '.join(timestamp.split(' ')[:-1])
    timestamp_date = datetime.strptime(timestamp_, TIMESTAMP_FMTSTR_)
    return timestamp_date


def calculate_timedelta(start, end):
    start_date = convert_to_date(start)
    end_date = convert_to_date(end)
    delta = end_date - start_date

    total_seconds = int(delta.total_seconds())
    total_seconds_ = total_seconds

    hours = total_seconds_ // (60 * 60)
    total_seconds_ -= hours * 60 * 60
    minutes = total_seconds_ // 60
    total_seconds_ -= minutes * 60
    seconds = total_seconds_

    return hours, minutes, seconds, total_seconds


def on_collect_request(
    ibs, collect_request, job_store, shelve_path, containerized=False
):
    """Run whenever the collector receives a message.

    *job_store* is a :class:`wbia.web.job_store.JobStore` (SQLite WAL).
    """
    action = collect_request.get('action', None)
    jobid = collect_request.get('jobid', None)
    status = collect_request.get('status', None)

    reply = {
        'status': 'ok',
        'jobid': jobid,
    }

    # Validate jobid
    if jobid is not None:
        try:
            assert isinstance(jobid, str)
            try:
                uuid.UUID(jobid)
            except Exception:
                assert len(jobid) < 32, 'Job IDs cannot be more than 32 characters'
                pattern = r'^[a-zA-Z0-9-_]+$'
                matched = bool(re.match(pattern, jobid))
                assert matched, 'Job IDs must be alpha-numeric'
        except AssertionError:
            print('Invalid Job ID')
            reply['status'] = 'error'
            return reply

    if jobid is not None:
        print(
            'on_collect_request action = %r, jobid = %r, status = %r'
            % (action, jobid, status)
        )

    if action == 'notification':
        assert jobid is not None

        # Ensure row exists
        job_store.ensure_job(jobid, status)

        current_status = job_store.get_status(jobid)
        print(
            'Updating jobid = {!r} status {!r} -> {!r}'.format(
                jobid, current_status, status
            )
        )
        job_store.update_status(jobid, status)

        if status == 'completed':
            # Mark the engine request .pkl as finished — do it in a
            # background thread so disk I/O doesn't block the collector.
            record_filename = '{}.pkl'.format(jobid)
            record_filepath = join(shelve_path, record_filename)

            def _mark_pkl_completed(_path=record_filepath):
                try:
                    if exists(_path):
                        _rec = ut.load_cPkl(_path, verbose=False)
                        _rec['completed'] = True
                        _save_engine_record(_path, _rec)
                except Exception:
                    pass

            threading.Thread(target=_mark_pkl_completed, daemon=True).start()

        # Update times
        times = job_store.get_times(jobid)
        times['updated'] = _timestamp()

        if status == 'working':
            times['started'] = _timestamp()

        if status == 'completed':
            times['completed'] = _timestamp()

        # Calculate runtime
        received = times.get('received', None)
        started = times.get('started', None)
        completed = times.get('completed', None)
        runtime = times.get('runtime', None)
        turnaround = times.get('turnaround', None)

        if None not in [started, completed] and runtime is None:
            hours, minutes, seconds, total_seconds = calculate_timedelta(
                started, completed
            )
            times['runtime'] = '%d hours %d min. %s sec. (total: %d sec.)' % (
                hours, minutes, seconds, total_seconds,
            )
            times['runtime_sec'] = total_seconds

        if None not in [received, completed] and turnaround is None:
            hours, minutes, seconds, total_seconds = calculate_timedelta(
                received, completed
            )
            times['turnaround'] = '%d hours %d min. %s sec. (total: %d sec.)' % (
                hours, minutes, seconds, total_seconds,
            )
            times['turnaround_sec'] = total_seconds

        job_store.update_times(jobid, times)

    elif action == 'register':
        assert jobid is not None

        metadata = job_store.get_metadata(jobid)
        result = job_store.get_result(jobid)

        if status == 'completed':
            if None in [metadata, result]:
                status = 'corrupted'

        _jc = metadata.get('jobcounter', -1) if metadata else -1
        job_store.register_job(jobid, status, _jc)
        print('Register jobid=%s status=%s jobcounter=%s' % (jobid, status, _jc))

    elif action == 'metadata':
        metadata = collect_request.get('metadata', None)
        job_store.store_metadata(jobid, metadata)
        print('Stored Metadata for jobid=%s' % (jobid,))
        metadata = None

    elif action == 'store':
        engine_result = collect_request.get('engine_result', None)
        callback_url = collect_request.get('callback_url', None)
        callback_method = collect_request.get('callback_method', None)
        callback_detailed = collect_request.get('callback_detailed', False)

        jobid = engine_result.get('jobid', jobid)

        job_store.store_result(jobid, engine_result)
        print('Stored Result for jobid=%s' % (jobid,))

        engine_result = None  # Release memory

        if callback_url is not None:
            if callback_method is None:
                callback_method = 'POST'

            callback_method = callback_method.upper()
            message = 'callback_method {!r} unsupported'.format(callback_method)
            assert callback_method in ['POST', 'GET', 'PUT'], message

            # Build the callback payload on the collector thread (needs
            # job_store access), then fire the HTTP request in a daemon
            # thread so the collector loop is never blocked by network I/O.
            data_dict = {'jobid': jobid}

            if callback_detailed:
                result_data = job_store.get_result(jobid)
                if result_data is not None:
                    data_dict['status'] = result_data['exec_status']
                    data_dict['json_result'] = ut.from_json(result_data['json_result'])
                result_data = None

            _fire_callback(callback_url, callback_method, data_dict, print)

    elif action == 'job_status':
        reply['jobstatus'] = job_store.get_status(jobid) or 'unknown'

    elif action == 'job_status_dict':
        request_limit = collect_request.get('limit', 0)
        reply['json_result'] = job_store.get_job_status_dict(limit=request_limit)

    elif action == 'job_id_list':
        reply['jobid_list'] = job_store.get_job_ids()

    elif action == 'job_input':
        if not job_store.job_exists(jobid):
            reply['status'] = 'invalid'
            reply['json_result'] = None
        else:
            metadata = job_store.get_metadata(jobid)
            if metadata is None:
                reply['status'] = 'corrupted'
            reply['json_result'] = metadata

    elif action == 'job_result':
        if not job_store.job_exists(jobid):
            reply['status'] = 'invalid'
            reply['json_result'] = None
        else:
            status = job_store.get_status(jobid)
            result_data = job_store.get_result(jobid)

            if result_data is None:
                if status in ['corrupted']:
                    reply['status'] = 'corrupted'
                elif status in ['suppressed']:
                    reply['status'] = 'suppressed'
                elif status in ['completed']:
                    reply['status'] = 'corrupted'
                else:
                    reply['status'] = status
                reply['json_result'] = None
            else:
                reply['status'] = result_data['exec_status']
                reply['json_result'] = ut.from_json(result_data['json_result'])

            result_data = None
    else:
        print('...error unknown action={!r}'.format(action))
        reply['status'] = 'error'

    return reply


def send_multipart_json(sock, idents, reply):
    """helper"""
    reply_json = ut.to_json(reply).encode('utf-8')
    reply = None
    multi_reply = idents + [reply_json]
    sock.send_multipart(multi_reply)


def rcv_multipart_json(sock, num=2, print=print):
    """helper"""
    # note that the first two parts will be ['Controller.ROUTER', 'Client.<id_>']
    # these are needed for the reply to propagate up to the right client
    multi_msg = sock.recv_multipart()
    if VERBOSE_JOBS:
        print('----')
        print('RCV Json: {}'.format(ut.repr2(multi_msg, truncate=True)))
    idents = multi_msg[:num]
    request_json = multi_msg[num]
    request = ut.from_json(request_json)
    request_json = None
    multi_msg = None
    return idents, request


def _on_ctrl_c(signal, frame):
    print('[wbia.zmq] Caught ctrl+c')
    print('[wbia.zmq] sys.exit(0)')
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
