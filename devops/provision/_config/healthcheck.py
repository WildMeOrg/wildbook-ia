# -*- coding: utf-8 -*-
import requests
import sys

config = {
    'protocol': 'http',
    'server': 'localhost',
    'port': 5000,
    'endpoint': '/api/test/heartbeat/',
}

args = (
    config['protocol'],
    config['server'],
    config['port'],
    config['endpoint'],
)
url = '%s://%s:%s%s' % args

try:
    raw = requests.get(url, timeout=120)
    response = raw.json()
    healthy = response.get('response')

    if not healthy:
        raise RuntimeError

    sys.exit(0)
except Exception:
    sys.exit(1)
