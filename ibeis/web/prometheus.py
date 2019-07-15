from prometheus_client import Info, Gauge
# import ibeis

prom_data = {
    'info'    : Info('ibeis_db', 'Description of IBEIS database'),
    'images'  : Gauge('ibeis_db_images', 'Number of Images'),
}

prom_data['info'].info({'version': '2.0.0'})
prom_data['images'].set(10)
