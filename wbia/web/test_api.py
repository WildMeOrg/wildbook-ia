#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This is a proof of concept for connecting to an authenticated Qubica Server"""
from __future__ import print_function, division, absolute_import
from hashlib import sha1
import utool as ut
import hmac
import requests


(print, rrr, profile) = ut.inject2(__name__)

# System variables
APPLICATION_PROTOCOL = 'http'
APPLICATION_DOMAIN = '127.0.0.1'
APPLICATION_PORT = None
APPLICATION_NAME = 'IBEIS'
APPLICATION_SECRET_KEY = 'CB73808F-A6F6-094B-5FCD-385EBAFF8FC0'


def _raise(exception, message):
    raise exception('[%s] ERROR: %s' % (__file__, message))


def get_signature(key, message):
    return str(hmac.new(key, message, sha1).digest().encode('base64').rstrip('\n'))


def get_authorization_header(uri, user_email=None, user_enc_pass=None):
    # Get signature
    secret_key_signature = get_signature(APPLICATION_SECRET_KEY, uri)
    application_authentication = '%s:%s' % (APPLICATION_NAME, secret_key_signature,)
    if user_email is None or user_enc_pass is None:
        return '%s' % (application_authentication,)
    return '%s:%s:%s' % (application_authentication, user_email, user_enc_pass)


def _api_result(uri, method, user_email=None, user_enc_pass=None, **kwargs):
    """Make a general (method) API request to the server"""
    # Make GET request to server
    method = method.upper()
    url = '%s://%s:%s%s' % (
        APPLICATION_PROTOCOL,
        APPLICATION_DOMAIN,
        APPLICATION_PORT,
        uri,
    )
    header = get_authorization_header(url, user_email, user_enc_pass)
    headers = {'Authorization': header}
    args = (
        method,
        url,
        headers,
        kwargs,
    )
    print('Server request (%r): %r\n\tHeaders: %r\n\tArgs: %r' % args)
    try:
        if method == 'GET':
            req = requests.get(url, headers=headers, params=kwargs, verify=False)
        elif method == 'POST':
            req = requests.post(url, headers=headers, payload=kwargs, verify=False)
        else:
            _raise(KeyError, '_api_result got unsupported method=%r' % (method,))
    except requests.exceptions.ConnectionError as ex:
        _raise(IOError, '_api_result could not connect to server %s' % (ex,))
    return req.status_code, req.text, req.json


def get_api_result(uri, user_email=None, user_enc_pass=None, **kwargs):
    """Make a GET API request to the server"""
    return _api_result(
        uri, 'get', user_email=user_email, user_enc_pass=user_enc_pass, **kwargs
    )


def post_api_result(uri, user_email=None, user_enc_pass=None, **kwargs):
    """Make a GET API request to the server"""
    return _api_result(
        uri, 'post', user_email=user_email, user_enc_pass=user_enc_pass, **kwargs
    )


def run_test_api():
    r"""
    CommandLine:
        python -m wbia.web.test_api --test-run_test_api

    Example:
        >>> # xdoctest: +REQUIRES(--web)
        >>> from wbia.web.test_api import *  # NOQA
        >>> response = run_test_api()
        >>> print('Server response: %r' % (response, ))
        >>> result = response
        (200, u'{"status": {"cache": -1, "message": "", "code": 200, "success": true}, "response": "testdb1"}', <bound method Response.json of <Response [200]>>)
    """
    import wbia
    import time

    global APPLICATION_PORT

    web_instance = wbia.opendb_in_background(db='testdb1', web=True, precache=False)

    # Get the application port from the background process
    if APPLICATION_PORT is None:
        web_port = web_instance.get_web_port_via_scan()
        if web_port is None:
            raise ValueError('IA web server is not running on any expected port')
        APPLICATION_PORT = '%s' % (web_port,)
    assert APPLICATION_PORT is not None

    # let the webapi startup in the background
    time.sleep(0.1)
    uri = '/api/core/dbname/'
    # Make GET request to the server as a test
    response = get_api_result(uri)
    status_code, text, json = response
    web_instance.terminate()
    return response


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.web.test_api
        python -m wbia.web.test_api --allexamples
        python -m wbia.web.test_api --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
