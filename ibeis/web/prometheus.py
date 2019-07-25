from prometheus_client import Info, Gauge, Enum, Histogram  # NOQA
from ibeis.control import controller_inject
import utool as ut


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_api   = controller_inject.get_ibeis_flask_api(__name__)


PROMETHEUS_COUNTER = 0
PROMETHEUS_LIMIT = 1


PROMETHEUS_DATA = {
    'info'       : Info(
        'ibeis_db',
        'Description of IBEIS database',
    ),
    'engine'     : Gauge(
        'ibeis_engine_jobs',
        'Job engine status',
        ['status'],
    ),
    'elapsed'    : Gauge(
        'ibeis_elapsed_seconds',
        'Number of elapsed seconds for the current working job',
    ),
    'elapsed'    : Gauge(
        'ibeis_elapsed_seconds',
        'Number of elapsed seconds for the current working job',
    ),
    'runtime'    : Gauge(
        'ibeis_runtime_seconds',
        'Number of runtime seconds for the current working job',
    ),
    'turnaround' : Gauge(
        'ibeis_turnaround_seconds',
        'Number of turnaround seconds for the current working job',
    ),
}


PROMETHUS_JOB_CACHE_DICT = {}


@register_ibs_method
@register_api('/api/test/prometheus/', methods=['GET', 'POST', 'DELETE', 'PUT'], __api_plural_check__=False)
def prometheus_update(ibs, *args, **kwargs):

    global PROMETHEUS_COUNTER

    PROMETHEUS_COUNTER = PROMETHEUS_COUNTER + 1  # NOQA
    # print('PROMETHEUS LIMIT %d / %d' % (PROMETHEUS_COUNTER, PROMETHEUS_LIMIT, ))

    if PROMETHEUS_COUNTER >= PROMETHEUS_LIMIT:
        PROMETHEUS_COUNTER = 0

        PROMETHEUS_DATA['info'].info({
            'uuid': str(ibs.get_db_init_uuid()),
            'dbname': ibs.dbname,
            'hostname': ut.get_computer_name(),
            'version': ibs.db.get_db_version(),
            'containerized': str(int(ibs.containerized)),
            'production': str(int(ibs.production)),
        })

        job_status_dict = ibs.get_job_status()['json_result']

        job_uuid_list = list(job_status_dict.keys())
        status_dict = {
            'received'   : 0,
            'accepted'   : 0,
            'queued'     : 0,
            'working'    : 0,
            'publishing' : 0,
            'completed'  : 0,
            'exception'  : 0,
            'suppressed' : 0,
        }

        is_working = False
        for job_uuid in job_uuid_list:
            job_status = job_status_dict[job_uuid]
            status = job_status['status']

            if status in ['working']:
                from ibeis.web.job_engine import TIMESTAMP_FMTSTR, TIMESTAMP_TIMEZONE
                from datetime import datetime
                import pytz

                started = job_status['time_started']
                assert started is not None
                TIMESTAMP_FMTSTR_ = ' '.join(TIMESTAMP_FMTSTR.split(' ')[:-1])
                started_ = ' '.join(started.split(' ')[:-1])
                timezone = pytz.timezone(TIMESTAMP_TIMEZONE)
                started_date = datetime.strptime(started_, TIMESTAMP_FMTSTR_)
                current_date = datetime.now(timezone)
                started_date = started_date.replace(tzinfo=current_date.tzinfo)
                delta = current_date - started_date
                total_seconds = int(delta.total_seconds())
                print('ELAPSED (%s): %d seconds...' % (job_uuid, total_seconds, ))
                PROMETHEUS_DATA['elapsed'].set(total_seconds)
                is_working = True

            if status not in status_dict:
                print('UNRECOGNIZED STATUS %r' % (status, ))
            status_dict[status] += 1

            if job_uuid not in PROMETHUS_JOB_CACHE_DICT:
                PROMETHUS_JOB_CACHE_DICT[job_uuid] = {}

            runtime_sec = job_status.get('time_runtime_sec', None)
            if runtime_sec is not None and 'runtime' not in PROMETHUS_JOB_CACHE_DICT[job_uuid]:
                PROMETHUS_JOB_CACHE_DICT[job_uuid]['runtime'] = runtime_sec
                PROMETHEUS_DATA['runtime'].observe(runtime_sec)

            turnaround_sec = job_status.get('time_turnaround_sec', None)
            if turnaround_sec is not None and 'turnaround' not in PROMETHUS_JOB_CACHE_DICT[job_uuid]:
                PROMETHUS_JOB_CACHE_DICT[job_uuid]['turnaround'] = turnaround_sec
                PROMETHEUS_DATA['turnaround'].observe(turnaround_sec)

        if not is_working:
            PROMETHEUS_DATA['elapsed'].set(0.0)

        # print(ut.repr3(status_dict))
        for status in status_dict:
            number = status_dict[status]
            PROMETHEUS_DATA['engine'].labels(status=status).set(number)
