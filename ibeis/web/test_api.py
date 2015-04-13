#!/usr/bin/env python
"""
    This is a proof of concept for connecting to an authenticated Qubica Server
"""
from __future__ import print_function, division, absolute_import
from hashlib import sha1
import hmac
import requests

# System variables
APPLICATION_PROTOCOL   = 'http'
APPLICATION_DOMAIN     = '127.0.0.1'
APPLICATION_PORT       = '5000'
APPLICATION_NAME       = 'IBEIS'
APPLICATION_SECRET_KEY = 'CB73808F-A6F6-094B-5FCD-385EBAFF8FC0'


def _raise(exception, message):
    raise exception('[%s] ERROR: %s' % (__file__, message))


def get_signature(key, message):
    return str(hmac.new(key, message, sha1).digest().encode("base64").rstrip('\n'))


def get_authorization_header(uri, user_email=None, user_enc_pass=None):
    # Get signature
    secret_key_signature = get_signature(APPLICATION_SECRET_KEY, uri)
    application_authentication = '%s:%s' % (APPLICATION_NAME, secret_key_signature, )
    if user_email is None or user_enc_pass is None:
        return '%s' % (application_authentication, )
    return '%s:%s:%s' % (application_authentication, user_email, user_enc_pass)


def _api_result(uri, method, user_email=None, user_enc_pass=None, **kwargs):
    """
        Make a general (method) API request to the server
    """
    # Make GET request to server
    method = method.upper()
    url = '%s://%s:%s%s' % (APPLICATION_PROTOCOL, APPLICATION_DOMAIN, APPLICATION_PORT, uri)
    header = get_authorization_header(url, user_email, user_enc_pass)
    headers = {'Authorization': header}
    args = (method, url, headers, kwargs, )
    print('Server request (%r): %r\n\tHeaders: %r\n\tArgs: %r' % args)
    try:
        if method == 'GET':
            req = requests.get(url, headers=headers, params=kwargs, verify=False)
        elif method == 'POST':
            req = requests.post(url, headers=headers, payload=kwargs, verify=False)
        else:
            _raise(KeyError, '_api_result got unsupported method=%r' % (method, ))
    except requests.exceptions.ConnectionError as ex:
        _raise(IOError, '_api_result could not connect to server %s' % (ex, ))
    return req.status_code, req.text, req.json


def get_api_result(uri, user_email=None, user_enc_pass=None, **kwargs):
    """
        Make a GET API request to the server
    """
    return _api_result(uri, 'get', user_email=user_email,
                       user_enc_pass=user_enc_pass, **kwargs)


def post_api_result(uri, user_email=None, user_enc_pass=None, **kwargs):
    """
        Make a GET API request to the server
    """
    return _api_result(uri, 'post', user_email=user_email,
                       user_enc_pass=user_enc_pass, **kwargs)


if __name__ == '__main__':
    uri = '/api/core/dbname'
    # Make GET request to the server as a test
    response = get_api_result(uri)
    status_code, text, json = response
    print('Server response: %r' % (response, ))
