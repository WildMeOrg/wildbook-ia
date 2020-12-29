# -*- coding: utf-8 -*-
import wbia


def test_turk_identification_no_more_to_review():
    with wbia.opendb_bg_web('testdb2', managed=True) as web_ibs:
        resp = web_ibs.get('/turk/identification/lnbnn/')
        assert resp.status_code == 200
        assert b'Traceback' not in resp.content, resp.content
        assert b'<h1>No more to review!</h1>' in resp.content, resp.content
