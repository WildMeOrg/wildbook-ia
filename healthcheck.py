#!/usr/bin/env python
import requests
import time
import utool as ut

config = {
    'protocol' : 'http',
    'server'   : 'localhost',
    'port'     :  5005,
    'endpoint' : '/api/test/heartbeat/',
}

args = (config['protocol'], config['server'], config['port'], config['endpoint'], )
url = '%s://%s:%s%s' % args

bootstrapping = True
history = []
while True:
    history = history[-15:]
    failures = sum(history)

    print('bootstrapping = %s\toffline (%d): %r' % (bootstrapping, failures, history[::-1], ))

    if failures >= 10:
        print('Running restart...')
        # subprocess.call('./healthcheck.sh')
        ut.shell('tmux send-keys -t flukebook.0 "C-c" ENTER')
        ut.shell('tmux send-keys -t flukebook.0 "up" ENTER')
        bootstrapping = True
        history = []

    if len(history) > 0:
        time.sleep(60)

    try:
        raw = requests.get(url, timeout=120)
        response = raw.json()
        healthy = response.get('response')
    except Exception as ex:
        healthy = False

    if bootstrapping:
        if healthy:
            bootstrapping = False
        else:
            continue

    history.append(not healthy)
