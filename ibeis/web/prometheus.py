from prometheus_client import Info, Gauge, Enum
# import ibeis

PROMETHEUS_COUNTER = 0
PROMETHEUS_LIMIT = 60

PROMETHEUS_DATA = {
    'info'    : Info('ibeis_db', 'Description of IBEIS database'),
    'engine'  : Gauge('ibeis_engine_jobs', 'Job engine status', ['status']),
    # 'images'  : Gauge('ibeis_db_images', 'Number of Images'),
    # 'status'  : Enum('ibeis_engine_status', 'The current status of the job engine', states=['waiting', 'queued', 'working', 'stopped', 'error'])
}
