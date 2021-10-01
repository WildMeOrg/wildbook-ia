# -*- coding: utf-8 -*-
import os
import urllib.parse

from oauthlib.oauth2 import BackendApplicationClient, TokenExpiredError
from requests_oauthlib import OAuth2Session


# Allow non-ssl communication
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
HOUSTON_TOKEN_API = '/api/v1/auth/tokens'


def call_houston(uri, cached_session=[], method='GET', **kwargs):
    HOUSTON_CLIENT_ID = os.getenv('HOUSTON_CLIENT_ID')
    HOUSTON_CLIENT_SECRET = os.getenv('HOUSTON_CLIENT_SECRET')
    uri = uri.replace('houston+', '')

    def update_token():
        token_url = urllib.parse.urljoin(uri, HOUSTON_TOKEN_API)
        session.fetch_token(
            token_url=token_url,
            client_id=HOUSTON_CLIENT_ID,
            client_secret=HOUSTON_CLIENT_SECRET,
        )

    if cached_session:
        session = cached_session[0]
    else:
        client = BackendApplicationClient(client_id=HOUSTON_CLIENT_ID)
        session = OAuth2Session(client=client)
        cached_session.append(session)
        update_token()

    def get_response():
        return session.request(method, uri, **kwargs)

    try:
        resp = get_response()
        if resp.status_code == 401:
            update_token()
            resp = get_response()
        return resp
    except TokenExpiredError:
        update_token()
        return get_response()
