# -*- coding: utf-8 -*-
import wbia


def test_review_identification_no_more_to_review():
    with wbia.opendb_with_web('testdb2') as (ibs, client):
        resp = client.get('/review/identification/lnbnn/')
        assert resp.status_code == 200
        assert b'Traceback' not in resp.data
        assert b'<h1>Nothing more to review!</h1>' in resp.data
