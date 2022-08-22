# -*- coding: utf-8 -*-
import os
from urllib.parse import urlparse

from oauthlib.oauth2 import BackendApplicationClient, TokenExpiredError
from requests_oauthlib import OAuth2Session

# Allow non-ssl communication
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

HOUSTON_TOKEN_API = '%s/api/v1/auth/tokens'

HOUSTON_CLIENT_ID = os.getenv('HOUSTON_CLIENT_ID')
HOUSTON_CLIENT_SECRET = os.getenv('HOUSTON_CLIENT_SECRET')

HOUSTON_SESSIONS = {}


def init_houston_session(hostname):
    global HOUSTON_SESSIONS

    if HOUSTON_SESSIONS.get(hostname, None) is None:
        client = BackendApplicationClient(client_id=HOUSTON_CLIENT_ID)
        HOUSTON_SESSIONS[hostname] = OAuth2Session(client=client)

    return HOUSTON_SESSIONS[hostname]


def forget_houston_session(hostname):
    global HOUSTON_SESSIONS

    return HOUSTON_SESSIONS.pop(hostname, None)


def refresh_houston_session_token(hostname):
    session = init_houston_session(hostname)

    session.fetch_token(
        token_url=HOUSTON_TOKEN_API % (hostname,),
        client_id=HOUSTON_CLIENT_ID,
        client_secret=HOUSTON_CLIENT_SECRET,
    )

    return session


def get_houston_session(hostname):
    session = HOUSTON_SESSIONS.get(hostname, None)

    if session is None:
        session = refresh_houston_session_token(hostname)

    return session


def call_houston(uri, method='GET', retry=True, **kwargs):
    clean_uri = uri.replace('houston+', '')

    parse_uri = urlparse(clean_uri)
    hostname = '{}://{}'.format(
        parse_uri.scheme,
        parse_uri.netloc,
    )

    session = get_houston_session(hostname)

    try:
        return session.request(method, clean_uri, **kwargs)
    except (Exception, TokenExpiredError):
        if retry:
            forget_houston_session(hostname)
            return call_houston(uri, method=method, retry=False, **kwargs)
        else:
            raise
