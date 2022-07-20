# -*- coding: utf-8 -*-
import wbia
from wbia._version import __version__


def test_version():
    with wbia.opendb_with_web('testdb2') as (ibs, client):
        resp = client.get('/api/version/')
        assert resp.status_code == 200
        assert resp.content_type == 'application/json; charset=utf-8'
        assert resp.json['response'] == {'version': __version__}
