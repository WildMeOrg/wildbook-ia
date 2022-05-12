# -*- coding: utf-8 -*-
import os

from oauthlib.oauth2 import BackendApplicationClient, TokenExpiredError
from requests_oauthlib import OAuth2Session
from urllib.parse import urlparse


# Allow non-ssl communication
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

HOUSTON_TOKEN_API = '%s/api/v1/auth/tokens'

HOUSTON_CLIENT_ID = os.getenv('HOUSTON_CLIENT_ID')
HOUSTON_CLIENT_SECRET = os.getenv('HOUSTON_CLIENT_SECRET')

HOUSTON_SESSION = None


def init_houston_session():
    global HOUSTON_SESSION

    if HOUSTON_SESSION is None:
        client = BackendApplicationClient(client_id=HOUSTON_CLIENT_ID)
        HOUSTON_SESSION = OAuth2Session(client=client)


def refresh_houston_session_token(hostname):
    global HOUSTON_SESSION

    init_houston_session()

    HOUSTON_SESSION.fetch_token(
        token_url=HOUSTON_TOKEN_API % (hostname,),
        client_id=HOUSTON_CLIENT_ID,
        client_secret=HOUSTON_CLIENT_SECRET,
    )


def call_houston(uri, method='GET', retry=True, **kwargs):
    clean_uri = uri.replace('houston+', '')

    if HOUSTON_SESSION is None:
        parse_uri = urlparse(clean_uri)
        hostname = '%s://%s' % (
            parse_uri.scheme,
            parse_uri.netloc,
        )

        refresh_houston_session_token(hostname)

    try:
        HOUSTON_SESSION.request(method, clean_uri, **kwargs)
    except (Exception, TokenExpiredError):
        if retry:
            return call_houston(uri, method=method, retry=False, **kwargs)
        else:
            raise
