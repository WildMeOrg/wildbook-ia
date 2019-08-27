from prometheus_client import Info, Gauge, Enum, Histogram  # NOQA
from ibeis.control import controller_inject
import ibeis.constants as const
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
    'imagesets'     : Gauge(
        'ibeis_assets_imagesets',
        'Number of imagesets in IBEIS database',
        ['name'],
    ),
    'images'     : Gauge(
        'ibeis_assets_images',
        'Number of images in IBEIS database',
        ['name'],
    ),
    'annotations': Gauge(
        'ibeis_assets_annotations',
        'Number of annotations in IBEIS database',
        ['name'],
    ),
    'parts'      : Gauge(
        'ibeis_assets_parts',
        'Number of parts in IBEIS database',
        ['name'],
    ),
    'names'      : Gauge(
        'ibeis_assets_names',
        'Number of names in IBEIS database',
        ['name'],
    ),
    'species'    : Gauge(
        'ibeis_assets_species',
        'Number of species in IBEIS database',
        ['name'],
    ),
    'engine'     : Gauge(
        'ibeis_engine_jobs',
        'Job engine status',
        ['status', 'name', 'endpoint'],
    ),
    'elapsed'    : Gauge(
        'ibeis_elapsed_seconds',
        'Number of elapsed seconds for the current working job',
        ['name', 'endpoint'],
    ),
    'runtime'    : Gauge(
        'ibeis_runtime_seconds',
        'Number of runtime seconds for the current working job',
        ['name', 'endpoint'],
    ),
    'turnaround' : Gauge(
        'ibeis_turnaround_seconds',
        'Number of turnaround seconds for the current working job',
        ['name', 'endpoint'],
    ),
}


PROMETHUS_JOB_CACHE_DICT = {}


@register_ibs_method
@register_api('/api/test/prometheus/', methods=['GET', 'POST', 'DELETE', 'PUT'], __api_plural_check__=False)
def prometheus_update(ibs, *args, **kwargs):
    try:
        CONTAINER_NAME = const.CONTAINER_NAME

        global PROMETHEUS_COUNTER

        PROMETHEUS_COUNTER = PROMETHEUS_COUNTER + 1  # NOQA
        # print('PROMETHEUS LIMIT %d / %d' % (PROMETHEUS_COUNTER, PROMETHEUS_LIMIT, ))

        if PROMETHEUS_COUNTER >= PROMETHEUS_LIMIT:
            PROMETHEUS_COUNTER = 0

            try:
                PROMETHEUS_DATA['info'].info({
                    'uuid': str(ibs.get_db_init_uuid()),
                    'dbname': ibs.dbname,
                    'hostname': ut.get_computer_name(),
                    'container': CONTAINER_NAME,
                    'version': ibs.db.get_db_version(),
                    'containerized': str(int(ibs.containerized)),
                    'production': str(int(ibs.production)),
                })
            except:
                pass

            try:
                num_imageset_rowids = len(ibs._get_all_imageset_rowids())
                num_gids = len(ibs._get_all_gids())
                num_aids = len(ibs._get_all_aids())
                num_pids = len(ibs._get_all_part_rowids())
                num_nids = len(ibs._get_all_name_rowids())
                num_species = len(ibs._get_all_species_rowids())
                PROMETHEUS_DATA['imagesets'].labels(name=CONTAINER_NAME).set(num_imageset_rowids)
                PROMETHEUS_DATA['images'].labels(name=CONTAINER_NAME).set(num_gids)
                PROMETHEUS_DATA['annotations'].labels(name=CONTAINER_NAME).set(num_aids)
                PROMETHEUS_DATA['parts'].labels(name=CONTAINER_NAME).set(num_pids)
                PROMETHEUS_DATA['names'].labels(name=CONTAINER_NAME).set(num_nids)
                PROMETHEUS_DATA['species'].labels(name=CONTAINER_NAME).set(num_species)
            except:
                pass

            try:
                job_status_dict = ibs.get_job_status()['json_result']
            except:
                pass

            try:
                job_uuid_list = list(job_status_dict.keys())
                status_dict_template = {
                    'received'   : 0,
                    'accepted'   : 0,
                    'queued'     : 0,
                    'working'    : 0,
                    'publishing' : 0,
                    'completed'  : 0,
                    'exception'  : 0,
                    'suppressed' : 0,
                    'corrupted'  : 0,
                    '_error'     : 0,
                }
                status_dict = {
                    '*': status_dict_template.copy()
                }

                endpoints = set([])
                working_endpoint = None
            except:
                pass

            for job_uuid in job_uuid_list:
                try:
                    job_status = job_status_dict[job_uuid]

                    status = job_status['status']
                    endpoint = job_status['endpoint']

                    status = '%s' % (status, )
                    endpoint = '%s' % (endpoint, )

                    if status not in status_dict_template.keys():
                        status = '_error'

                    if endpoint not in status_dict:
                        status_dict[endpoint] = status_dict_template.copy()

                    endpoints.add(endpoint)
                except:
                    pass

                try:
                    if status in ['working']:
                        from ibeis.web.job_engine import calculate_timedelta, _timestamp
                        started = job_status['time_started']
                        now = _timestamp()
                        hours, minutes, seconds, total_seconds = calculate_timedelta(started, now)
                        print('ELAPSED (%s): %d seconds...' % (job_uuid, total_seconds, ))
                        PROMETHEUS_DATA['elapsed'].labels(name=CONTAINER_NAME, endpoint=endpoint).set(total_seconds)
                        PROMETHEUS_DATA['elapsed'].labels(name=CONTAINER_NAME, endpoint='*').set(total_seconds)
                        working_endpoint = endpoint
                except:
                    pass

                try:
                    if status not in status_dict_template:
                        print('UNRECOGNIZED STATUS %r' % (status, ))
                    status_dict[endpoint][status] += 1
                    status_dict['*'][status] += 1

                    if job_uuid not in PROMETHUS_JOB_CACHE_DICT:
                        PROMETHUS_JOB_CACHE_DICT[job_uuid] = {}
                except:
                    pass

                try:
                    runtime_sec = job_status.get('time_runtime_sec', None)
                    if runtime_sec is not None and 'runtime' not in PROMETHUS_JOB_CACHE_DICT[job_uuid]:
                        PROMETHUS_JOB_CACHE_DICT[job_uuid]['runtime'] = runtime_sec
                        PROMETHEUS_DATA['runtime'].labels(name=CONTAINER_NAME, endpoint=endpoint).set(runtime_sec)
                        PROMETHEUS_DATA['runtime'].labels(name=CONTAINER_NAME, endpoint='*').set(runtime_sec)
                except:
                    pass

                try:
                    turnaround_sec = job_status.get('time_turnaround_sec', None)
                    if turnaround_sec is not None and 'turnaround' not in PROMETHUS_JOB_CACHE_DICT[job_uuid]:
                        PROMETHUS_JOB_CACHE_DICT[job_uuid]['turnaround'] = turnaround_sec
                        PROMETHEUS_DATA['turnaround'].labels(name=CONTAINER_NAME, endpoint=endpoint).set(turnaround_sec)
                        PROMETHEUS_DATA['turnaround'].labels(name=CONTAINER_NAME, endpoint='*').set(turnaround_sec)
                except:
                    pass

            try:
                if working_endpoint is None:
                    PROMETHEUS_DATA['elapsed'].labels(name=CONTAINER_NAME, endpoint='*').set(0.0)

                for endpoint in endpoints:
                    if endpoint == working_endpoint:
                        continue
                    PROMETHEUS_DATA['elapsed'].labels(name=CONTAINER_NAME, endpoint=endpoint).set(0.0)
            except:
                pass

            try:
                # print(ut.repr3(status_dict))
                for endpoint in status_dict:
                    for status in status_dict[endpoint]:
                        number = status_dict[endpoint][status]
                        PROMETHEUS_DATA['engine'].labels(status=status, name=CONTAINER_NAME, endpoint=endpoint).set(number)
            except:
                pass
    except:
        pass
