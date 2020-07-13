# -*- coding: utf-8 -*-
"""
TODO:
    Move flask registering into another file.
    Should also make the actual flask registration lazy.
    It should only be executed if a web instance is being started.


python -c "import wbia"
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import six
import sys
from wbia import dtool
from datetime import timedelta
from functools import update_wrapper
import warnings
from functools import wraps
from os.path import abspath, join, dirname

# import simplejson as json
# import json
# import pickle
# from six.moves import cPickle as pickle
import traceback
from hashlib import sha1
import os

# import numpy as np
import hmac
from wbia import constants as const
import string
import random
import base64

# <flask>
# TODO: allow optional flask import
try:
    import flask
    from flask import session, request

    HAS_FLASK = True
except Exception:
    HAS_FLASK = False
    msg = 'Missing flask and/or Flask-session.\n' 'pip install Flask'
    warnings.warn(msg)
    if ut.STRICT:
        raise

try:
    # from flask.ext.cors import CORS
    from flask_cors import CORS

    HAS_FLASK_CORS = True
except Exception:
    HAS_FLASK_CORS = False
    warnings.warn('Missing flask.ext.cors')
    if ut.SUPER_STRICT:
        raise

try:
    from flask_cas import CAS  # NOQA
    from flask_cas import login_required as login_required_cas

    # from flask.ext.cas import CAS
    # from flask.ext.cas import login_required
    # HAS_FLASK_CAS = True
    HAS_FLASK_CAS = False
except Exception:
    HAS_FLASK_CAS = False
    login_required_cas = ut.identity
    msg = (
        'Missing flask.ext.cas.\n'
        'To install try pip install git+https://github.com/cameronbwhite/Flask-CAS.git'
    )
    warnings.warn(msg)
    # sudo
    print('')
    if ut.SUPER_STRICT:
        raise


# </flask>
print, rrr, profile = ut.inject2(__name__)


# INJECTED_MODULES = []
UTOOL_AUTOGEN_SPHINX_RUNNING = not (
    os.environ.get('UTOOL_AUTOGEN_SPHINX_RUNNING', 'OFF') == 'OFF'
)

GLOBAL_APP_ENABLED = (
    not UTOOL_AUTOGEN_SPHINX_RUNNING and not ut.get_argflag('--no-flask') and HAS_FLASK
)
GLOBAL_APP_NAME = 'IBEIS'
GLOBAL_APP_SECRET = os.urandom(64)

GLOBAL_APP = None
GLOBAL_CORS = None
GLOBAL_CAS = None

REMOTE_PROXY_URL = None
REMOTE_PROXY_PORT = 5001

WEB_DEBUG_INCLUDE_TRACE = True

CONTROLLER_CLASSNAME = 'IBEISController'

MICROSOFT_API_ENABLED = ut.get_argflag('--web') and ut.get_argflag(
    '--microsoft'
)  # True == Microsoft Deployment (i.e., only allow MICROSOFT_API_PREFIX prefix below)
MICROSOFT_API_PREFIX = '/v0.1/wildbook/'
MICROSOFT_API_DEBUG = True

if MICROSOFT_API_ENABLED:
    WEB_DEBUG_INCLUDE_TRACE = MICROSOFT_API_DEBUG


STRICT_VERSION_API = (
    False  # True == Microsoft Deployment (i.e., only allow /wildme/v0.1/ prefixes)
)


def get_flask_app(templates_auto_reload=True):
    # TODO this should be initialized explicity in main_module.py only if needed
    global GLOBAL_APP
    global GLOBAL_CORS
    global GLOBAL_CAS
    global HAS_FLASK
    if not HAS_FLASK:
        print('flask is not installed')
        return None
    if GLOBAL_APP is None:
        if hasattr(sys, '_MEIPASS'):
            # hack for pyinstaller directory
            root_dpath = sys._MEIPASS
        else:
            root_dpath = abspath(dirname(dirname(__file__)))
        tempalte_dpath = join(root_dpath, 'web', 'templates')
        static_dpath = join(root_dpath, 'web', 'static')
        if ut.VERBOSE:
            print('[get_flask_app] root_dpath = %r' % (root_dpath,))
            print('[get_flask_app] tempalte_dpath = %r' % (tempalte_dpath,))
            print('[get_flask_app] static_dpath = %r' % (static_dpath,))
            print('[get_flask_app] GLOBAL_APP_NAME = %r' % (GLOBAL_APP_NAME,))
        GLOBAL_APP = flask.Flask(
            GLOBAL_APP_NAME, template_folder=tempalte_dpath, static_folder=static_dpath
        )

        if ut.VERBOSE:
            print('[get_flask_app] USING FLASK SECRET KEY: %r' % (GLOBAL_APP_SECRET,))
        GLOBAL_APP.secret_key = GLOBAL_APP_SECRET

        if templates_auto_reload:
            GLOBAL_APP.config['TEMPLATES_AUTO_RELOAD'] = True
        GLOBAL_APP.QUERY_OBJECT = None
        GLOBAL_APP.QUERY_OBJECT_JOBID = None
        GLOBAL_APP.QUERY_OBJECT_FEEDBACK_BUFFER = []
        GLOBAL_APP.GRAPH_CLIENT_DICT = {}

        if HAS_FLASK_CORS:
            GLOBAL_CORS = CORS(
                GLOBAL_APP, resources={r'/api/*': {'origins': '*'}}
            )  # NOQA

        # if HAS_FLASK_CAS:
        #     GLOBAL_CAS = CAS(GLOBAL_APP, '/cas')
        #     GLOBAL_APP.config['SESSION_TYPE']    = 'memcached'
        #     GLOBAL_APP.config['SECRET_KEY']      = GLOBAL_APP_SECRET
        #     GLOBAL_APP.config['CAS_SERVER']      = 'https://cas-auth.rpi.edu'
        #     GLOBAL_APP.config['CAS_AFTER_LOGIN'] = 'root'
    return GLOBAL_APP


# try and load flask
try:
    if GLOBAL_APP_ENABLED:
        get_flask_app()
except AttributeError:
    if six.PY3:
        print('Warning flask is broken in python-3.4.0')
        GLOBAL_APP_ENABLED = False
        HAS_FLASK = False
    else:
        raise


class WebException(ut.NiceRepr, Exception):
    def __init__(self, message, rawreturn=None, code=400):
        self.code = code
        self.message = message
        self.rawreturn = rawreturn

        from wbia.web.app import PROMETHEUS

        if PROMETHEUS:
            ibs = flask.current_app.ibs
            tag = '%s' % (self.code,)
            ibs.prometheus_increment_exception(tag)

    def get_rawreturn(self, debug_stack_trace=False):
        if self.rawreturn is None:
            if debug_stack_trace:
                return str(traceback.format_exc())
            else:
                return None
        else:
            return self.rawreturn

    def __nice__(self):
        return '(%r: %r)' % (self.code, self.message,)


class WebMissingUUIDException(WebException):
    def __init__(self, missing_image_uuid_list=[], missing_annot_uuid_list=[]):
        args = (
            len(missing_image_uuid_list),
            len(missing_annot_uuid_list),
        )
        message = 'Missing image and/or annotation UUIDs (%d, %d)' % args
        rawreturn = {
            'missing_image_uuid_list': missing_image_uuid_list,
            'missing_annot_uuid_list': missing_annot_uuid_list,
        }
        code = 600
        super(WebMissingUUIDException, self).__init__(message, rawreturn, code)


class WebDuplicateUUIDException(WebException):
    def __init__(self, qdup_pos_map={}, ddup_pos_map={}):
        message = (
            'Some UUIDs are specified more than once at positions:\n'
            'duplicate_database_uuids=%s\n'
            'duplicate_query_uuids=%s\n'
        ) % (ut.repr3(qdup_pos_map, nl=1), ut.repr3(ddup_pos_map, nl=1))
        qdup_pos_map_ = {str(k): v for k, v in qdup_pos_map.items()}
        ddup_pos_map_ = {str(k): v for k, v in ddup_pos_map.items()}
        rawreturn = {
            'qdup_pos_map': qdup_pos_map_,
            'ddup_pos_map': ddup_pos_map_,
        }
        code = 601
        super(WebDuplicateUUIDException, self).__init__(message, rawreturn, code)


class WebUnknownUUIDException(WebException):
    def __init__(self, unknown_uuid_type_list, unknown_uuid_list):
        uuid_type_str = ', '.join(sorted(set(unknown_uuid_type_list)))
        args = (
            uuid_type_str,
            len(unknown_uuid_list),
        )
        message = 'Unknown %s UUIDs (%d)' % args
        rawreturn = {
            'unknown_uuid_type_list': unknown_uuid_type_list,
            'unknown_uuid_list': unknown_uuid_list,
        }
        code = 602
        super(WebUnknownUUIDException, self).__init__(message, rawreturn, code)


class WebReviewNotReadyException(WebException):
    def __init__(self, query_uuid):
        args = (query_uuid,)
        message = 'The query_uuid %r is not yet ready for review' % args
        rawreturn = {
            'query_uuid': query_uuid,
        }
        code = 603
        super(WebReviewNotReadyException, self).__init__(message, rawreturn, code)


class WebUnavailableUUIDException(WebException):
    def __init__(self, unavailable_annot_uuid_list, query_uuid):
        self.query_uuid = query_uuid
        args = (query_uuid,)
        message = (
            'A running query %s is using (at least one of) the requested annotations.  Filter out these annotations from the new query or stop the previous query.'
            % args
        )
        rawreturn = {
            'unavailable_annot_uuid_list': unavailable_annot_uuid_list,
            'query_uuid': query_uuid,
        }
        code = 604
        super(WebUnavailableUUIDException, self).__init__(message, rawreturn, code)


class WebReviewFinishedException(WebException):
    def __init__(self, query_uuid):
        args = (query_uuid,)
        message = 'The query_uuid %r has nothing more to review' % args
        rawreturn = {
            'query_uuid': query_uuid,
        }
        code = 605
        super(WebReviewFinishedException, self).__init__(message, rawreturn, code)


class WebMultipleNamedDuplicateException(WebException):
    def __init__(self, bad_dict):
        message = (
            'Duplcate UUIDs are specified with more than one name:\n'
            'bad_database_uuids=%s\n'
        ) % (ut.repr3(bad_dict, nl=1),)
        bad_dict = {str(k): v for k, v in bad_dict.items()}
        rawreturn = {
            'bad_dict': bad_dict,
        }
        code = 606
        super(WebMultipleNamedDuplicateException, self).__init__(message, rawreturn, code)


class WebMatchThumbException(WebException):
    def __init__(self, reference, qannot_uuid, dannot_uuid, version, message):
        rawreturn = {
            'reference': reference,
            'qannot_uuid': qannot_uuid,
            'dannot_uuid': dannot_uuid,
            'version': version,
        }
        code = 607
        super(WebMatchThumbException, self).__init__(message, rawreturn, code)


class WebInvalidUUIDException(WebException):
    def __init__(self, invalid_image_uuid_list=[], invalid_annot_uuid_list=[]):
        args = (
            len(invalid_image_uuid_list),
            len(invalid_annot_uuid_list),
        )
        message = 'Invalid image and/or annotation UUIDs (%d, %d)' % args
        rawreturn = {
            'invalid_image_uuid_list': invalid_image_uuid_list,
            'invalid_annot_uuid_list': invalid_annot_uuid_list,
        }
        code = 608
        super(WebInvalidUUIDException, self).__init__(message, rawreturn, code)


class WebInvalidMatchException(WebException):
    def __init__(self, qaid_list, daid_list):
        message = 'The ID request is invalid because the daid_list is empty (after filtering out the qaid_list)'
        rawreturn = {
            'qaid_list': qaid_list,
            'daid_list': daid_list,
        }
        code = 609
        super(WebInvalidMatchException, self).__init__(message, rawreturn, code)


class WebMissingInput(WebException):
    def __init__(self, message, key=None):
        rawreturn = {}
        if key is not None:
            rawreturn['parameter'] = key
        if message is not None:
            rawreturn['message'] = message
        code = 400
        super(WebMissingInput, self).__init__(message, rawreturn, code)


class WebInvalidInput(WebException):
    def __init__(self, message, key=None, value=None, image=False):
        rawreturn = {}
        if key is not None:
            rawreturn['parameter'] = key
        if value is not None:
            rawreturn['value'] = value
        if message is not None:
            rawreturn['message'] = message
        code = 415 if image else 400
        super(WebInvalidInput, self).__init__(message, rawreturn, code)


class WebRuntimeException(WebException):
    def __init__(self, message):
        rawreturn = {'message': message}
        code = 500
        super(WebRuntimeException, self).__init__(message, rawreturn, code)


def translate_wbia_webreturn(
    rawreturn,
    success=True,
    code=None,
    message=None,
    jQuery_callback=None,
    cache=None,
    __skip_microsoft_validation__=False,
):
    if MICROSOFT_API_ENABLED and not __skip_microsoft_validation__:
        if rawreturn is not None:
            assert isinstance(
                rawreturn, dict
            ), 'Microsoft APIs must return a Python dictionary'
        template = rawreturn
    else:
        if code is None:
            code = ''
        if message is None:
            message = ''
        if cache is None:
            cache = -1
        template = {
            'status': {
                'success': success,
                'code': code,
                'message': message,
                'cache': cache,
                # 'debug': {}  # TODO
            },
            'response': rawreturn,
        }
    response = ut.to_json(template)

    if jQuery_callback is not None and isinstance(jQuery_callback, six.string_types):
        print('[web] Including jQuery callback function: %r' % (jQuery_callback,))
        response = '%s(%s)' % (jQuery_callback, response)
    return response


def _process_input(multidict=None):
    if multidict is None:
        return {}
    if isinstance(multidict, dict):
        from werkzeug.datastructures import ImmutableMultiDict

        multidict = ImmutableMultiDict([item for item in multidict.items()])
    kwargs2 = {}
    for (arg, value) in multidict.lists():
        if len(value) > 1:
            raise WebException('Cannot specify a parameter more than once: %r' % (arg,))
        # value = str(value[0])
        value = value[0]
        if (
            ',' in value
            and '[' not in value
            and ']' not in value
            and '{' not in value
            and '}' not in value
        ):
            value = '[%s]' % (value,)
        if value in ['True', 'False']:
            value = value.lower()
        try:
            converted = ut.from_json(value)
        except Exception:
            # try making string and try again...
            try:
                value_ = '"%s"' % (value,)
                converted = ut.from_json(value_)
            except Exception as ex:
                print('FAILED TO JSON CONVERT: %s' % (ex,))
                print(ut.repr3(value))
                converted = value
        if arg.endswith('_list') and not isinstance(converted, (list, tuple)):
            if isinstance(converted, str) and ',' in converted:
                converted = converted.strip().split(',')
            else:
                converted = [converted]
        # Allow JSON formatted strings to be placed into note fields
        if (arg.endswith('note_list') or arg.endswith('notes_list')) and isinstance(
            converted, (list, tuple)
        ):
            type_ = type(converted)
            temp_list = []
            for _ in converted:
                if isinstance(_, dict):
                    temp_list.append('%s' % (_,))
                else:
                    temp_list.append(_)
            converted = type_(temp_list)
        kwargs2[arg] = converted
    return kwargs2


def translate_wbia_webcall(func, *args, **kwargs):
    r"""
    Called from flask request context

    Args:
        func (function):  live python function

    Returns:
        tuple: (output, True, 200, None, jQuery_callback)

    CommandLine:
        python -m wbia.control.controller_inject --exec-translate_wbia_webcall
        python -m wbia.control.controller_inject --exec-translate_wbia_webcall --domain http://52.33.105.88

    Example:
        >>> # xdoctest: +REQUIRES(--web)
        >>> from wbia.control.controller_inject import *  # NOQA
        >>> import wbia
        >>> import time
        >>> import wbia.web
        >>> web_ibs = wbia.opendb_bg_web('testdb1', wait=1, start_job_queue=False)
        >>> aids = web_ibs.send_wbia_request('/api/annot/', 'get')
        >>> uuid_list = web_ibs.send_wbia_request('/api/annot/uuids/', aid_list=aids)
        >>> failrsp = web_ibs.send_wbia_request('/api/annot/uuids/')
        >>> failrsp2 = web_ibs.send_wbia_request('/api/query/chips/simple_dict//', 'get', qaid_list=[0], daid_list=[0])
        >>> log_text = web_ibs.send_wbia_request('/api/query/chips/simple_dict/', 'get', qaid_list=[0], daid_list=[0])
        >>> time.sleep(.1)
        >>> print('\n---\nuuid_list = %r' % (uuid_list,))
        >>> print('\n---\nfailrsp =\n%s' % (failrsp,))
        >>> print('\n---\nfailrsp2 =\n%s' % (failrsp2,))
        >>> print('Finished test')
        >>> web_ibs.terminate2()

    Ignore:
        app = get_flask_app()
        with app.app_context():
            #ibs = wbia.opendb('testdb1')
            func = ibs.get_annot_uuids
            args = tuple()
            kwargs = dict()
    """
    assert len(args) == 0, 'There should not be any args=%r' % (args,)

    # print('Calling: %r with args: %r and kwargs: %r' % (func, args, kwargs, ))
    ibs = flask.current_app.ibs
    funcstr = ut.func_str(func, (ibs,) + args, kwargs=kwargs, truncate=True)
    if 'heartbeat' in funcstr:
        pass
    elif 'metrics' in funcstr:
        pass
    else:
        print('[TRANSLATE] Calling: %s' % (funcstr,))

    try:
        key_list = sorted(list(kwargs.keys()))
        type_list = []
        message_list = []

        for key in key_list:
            try:
                values = kwargs[key]
                type_ = type(values).__name__
                if type_ == 'list':
                    if len(values) == 0:
                        type_ = 'empty list'
                        message_ = '[]'
                    else:
                        value = values[0]
                        type_ += ' of ' + type(value).__name__
                        length1 = len(values)
                        try:
                            length2 = len(set(values))
                        except TypeError:
                            length2 = len(set(map(str, values)))
                        length3 = min(length1, 3)
                        mod = '...' if length1 != length3 else ''
                        message_ = 'length %d with unique %d of %s%s' % (
                            length1,
                            length2,
                            values[:length3],
                            mod,
                        )
                else:
                    message_ = '%s' % (values,)
            except Exception:
                type_ = 'UNKNOWN'
                message_ = 'ERROR IN PARSING'

            type_list.append(type_)
            message_list.append(message_)

        zipped = list(zip(key_list, type_list, message_list))

        if len(zipped) > 0:
            length1 = max(list(map(len, key_list)))
            length2 = max(list(map(len, type_list)))

            for key_, type_, message_ in zipped:
                key_ = key_.rjust(length1)
                type_ = type_.ljust(length2)
                try:
                    print('[TRANSLATE] \t %s (%s) : %s' % (key_, type_, message_,))
                except UnicodeEncodeError:
                    print('[TRANSLATE] \t %s (%s) : UNICODE ERROR')
    except Exception:
        print('[TRANSLATE] ERROR IN KWARGS PARSING')

    try:
        # TODO, have better way to differentiate ibs funcs from other funcs
        output = func(**kwargs)
    except TypeError:
        try:
            output = func(ibs=ibs, **kwargs)
        except WebException:
            raise
        except Exception as ex2:  # NOQA
            if MICROSOFT_API_ENABLED:
                if isinstance(ex2, TypeError) and 'required positional' in str(ex2):
                    parameter = str(ex2).split(':')[1].strip().strip("'")
                    raise WebMissingInput('Missing required parameter', parameter)
                elif isinstance(ex2, WebException):
                    raise
                else:
                    raise WebRuntimeException(
                        'An unknown error has occurred, please contact the API administrator at dev@wildme.org.'
                    )
            else:
                msg_list = []
                # msg_list.append('Error in translate_wbia_webcall')
                msg_list.append('Expected Function Definition: ' + ut.func_defsig(func))
                msg_list.append('Received Function Definition: %s' % (funcstr,))
                msg_list.append('Received Function Parameters:')
                for key in kwargs:
                    value = kwargs[key]
                    value_str = '%r' % (value,)
                    value_str = ut.truncate_str(value_str, maxlen=256)
                    msg_list.append('\t%r: %s' % (key, value_str,))
                # msg_list.append('\targs = %r' % (args,))
                # msg_list.append('flask.request.args = %r' % (flask.request.args,))
                # msg_list.append('flask.request.form = %r' % (flask.request.form,))
                msg_list.append('%s: %s' % (type(ex2).__name__, ex2,))
                if WEB_DEBUG_INCLUDE_TRACE:
                    trace = str(traceback.format_exc())
                    msg_list.append(trace)
                msg = '\n'.join(msg_list)
            print(msg)
            # error_msg = ut.formatex(ex2, msg, tb=True)
            # print(error_msg)
            # error_msg = ut.strip_ansi(error_msg)
            # raise Exception(error_msg)
            raise Exception(msg)
            # raise
    resp_tup = (output, True, 200, None)
    return resp_tup


def authentication_challenge():
    """Sends a 401 response that enables basic auth."""
    rawreturn = ''
    success = False
    code = 401
    message = 'Could not verify your authentication, login with proper credentials.'
    jQuery_callback = None
    webreturn = translate_wbia_webreturn(
        rawreturn, success, code, message, jQuery_callback
    )
    response = flask.make_response(webreturn, code)
    response.headers['WWW-Authenticate'] = 'Basic realm="Login Required"'
    return response


def authentication_user_validate():
    """
    This function is called to check if a username /
    password combination is valid.
    """
    auth = flask.request.authorization
    if auth is None:
        return False
    username = auth.username
    password = auth.password
    return username == 'wbia' and password == 'wbia'


def authentication_user_only(func):
    @wraps(func)
    def wrp_authenticate_user(*args, **kwargs):
        if not authentication_user_validate():
            return authentication_challenge()
        return func(*args, **kwargs)

    # wrp_authenticate_user = ut.preserve_sig(wrp_authenticate_user, func)
    return wrp_authenticate_user


def create_key():
    hyphen_list = [8, 13, 18, 23]
    key_list = [
        '-' if _ in hyphen_list else random.choice(string.hexdigits) for _ in range(36)
    ]
    return ''.join(key_list).upper()


def get_signature(key, message):
    def encode(x):
        if not isinstance(x, bytes):
            x = bytes(x, 'utf-8')
        return x

    def decode(x):
        return x.decode('utf-8')

    if six.PY3:
        key = encode(key)
        message = encode(message)

    signature = hmac.new(key, message, sha1)
    signature = signature.digest()
    signature = base64.b64encode(signature)
    signature = decode(signature)
    signature = str(signature)
    signature = signature.strip()
    return signature


def get_url_authorization(url):
    hash_ = get_signature(GLOBAL_APP_SECRET, url)
    hash_challenge = '%s:%s' % (GLOBAL_APP_NAME, hash_,)
    return hash_challenge


def authentication_hash_validate():
    """
    This function is called to check if a username /
    password combination is valid.
    """

    def last_occurence_delete(string, character):
        index = string.rfind(character)
        if index is None or index < 0:
            return string
        return string[:index] + string[index + 1 :]

    hash_response = str(flask.request.headers.get('Authorization', ''))
    if len(hash_response) == 0:
        return False
    hash_challenge_list = []
    # Check normal url
    url = str(flask.request.url)
    hash_challenge = get_url_authorization(url)
    hash_challenge_list.append(hash_challenge)
    # If hash at the end of the url, try alternate hash as well
    url = last_occurence_delete(url, '/')
    hash_challenge = get_url_authorization(url)
    hash_challenge_list.append(hash_challenge)
    if '?' in url:
        url.replace('?', '/?')
        hash_challenge = get_url_authorization(url)
        hash_challenge_list.append(hash_challenge)
    return hash_response in hash_challenge_list


def authentication_hash_only(func):
    @wraps(func)
    def wrp_authentication_hash(*args, **kwargs):
        if not authentication_hash_validate():
            return authentication_challenge()
        return func(*args, **kwargs)

    return wrp_authentication_hash


def authentication_either(func):
    """ authenticated by either hash or user """

    @wraps(func)
    def wrp_authentication_either(*args, **kwargs):
        if not (authentication_hash_validate() or authentication_user_validate()):
            return authentication_challenge()
        return func(*args, **kwargs)

    return wrp_authentication_either


def crossdomain(
    origin=None,
    methods=None,
    headers=None,
    max_age=21600,
    attach_to_all=True,
    automatic_options=True,
):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = flask.current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            print(origin)
            print(flask.request.method)

            if automatic_options and flask.request.method == 'OPTIONS':
                resp = flask.current_app.make_default_options_response()
            else:
                resp = flask.make_response(f(*args, **kwargs))
            if not attach_to_all and flask.request.method != 'OPTIONS':
                return resp

            h = resp.headers

            print(origin)
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Origin'] = '*'
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)

    return decorator


def remote_api_wrapper(func):
    def remote_api_call(ibs, *args, **kwargs):
        if REMOTE_PROXY_URL is None:
            return func(ibs, *args, **kwargs)
        else:
            co_varnames = func.func_code.co_varnames
            if co_varnames[0] == 'ibs':
                co_varnames = tuple(co_varnames[1:])
            kwargs_ = dict(zip(co_varnames, args))
            kwargs.update(kwargs_)
            kwargs.pop('ibs', None)
            return api_remote_wbia(REMOTE_PROXY_URL, func, REMOTE_PROXY_PORT, **kwargs)

    remote_api_call = ut.preserve_sig(remote_api_call, func)
    return remote_api_call


API_SEEN_SET = set([])


def get_wbia_flask_api(__name__, DEBUG_PYTHON_STACK_TRACE_JSON_RESPONSE=False):
    """For function calls that resolve to api calls and return json."""
    if __name__ == '__main__':
        return ut.dummy_args_decor
    if GLOBAL_APP_ENABLED:

        def register_api(
            rule, __api_plural_check__=True, __api_microsoft_check__=True, **options
        ):
            global API_SEEN_SET
            assert rule.endswith('/'), 'An API should always end in a forward-slash'
            assert (
                'methods' in options
            ), 'An api should always have a specified methods list'
            rule_ = rule + ':'.join(options['methods'])
            if rule_ in API_SEEN_SET:
                msg = 'An API rule (%s) has been duplicated' % (rule_,)
                warnings.warn(msg + '. Ignoring duplicate (may break web)')
                return ut.identity
                # raise AssertionError(msg)
            API_SEEN_SET.add(rule_)

            if MICROSOFT_API_ENABLED and __api_microsoft_check__:
                if not rule.startswith(MICROSOFT_API_PREFIX):
                    # msg = 'API rule=%r is does not adhere to the Microsoft format, ignoring.' % (rule_, )
                    # warnings.warn(msg)
                    return ut.identity
                else:
                    print('Registering API rule=%r' % (rule_,))

            try:
                if not MICROSOFT_API_ENABLED:
                    assert (
                        'annotation' not in rule
                    ), 'An API rule should use "annot" instead of annotation(s)"'
                assert (
                    'imgset' not in rule
                ), 'An API should use "imageset" instead of imgset(s)"'
                assert '_' not in rule, 'An API should never contain an underscore'
                assert '-' not in rule, 'An API should never contain a hyphen'
                if __api_plural_check__:
                    assert 's/' not in rule, 'Use singular (non-plural) URL routes'
                check_list = [
                    'annotgroup',
                    'autogen',
                    'chip',
                    'config',
                    # 'contributor',
                    'gar',
                    'metadata',
                ]
                for check in check_list:
                    assert '/api/%s/' % (check,) not in rule, 'failed check=%r' % (check,)
            except Exception:
                iswarning = not ut.SUPER_STRICT
                ut.printex(
                    'CONSIDER RENAMING API RULE: %r' % (rule,),
                    iswarning=iswarning,
                    tb=True,
                )
                if not iswarning:
                    raise

            # accpet args to flask.route
            def regsiter_closure(func):
                # make translation function in closure scope
                # and register it with flask.
                app = get_flask_app()

                @app.route(rule, **options)
                # @crossdomain(origin='*')
                # @authentication_either
                @wraps(func)
                # def translated_call(*args, **kwargs):
                def translated_call(**kwargs):
                    def html_newlines(text):
                        r = '<br />\n'
                        text = text.replace(' ', '&nbsp;')
                        text = (
                            text.replace('\r\n', r)
                            .replace('\n\r', r)
                            .replace('\r', r)
                            .replace('\n', r)
                        )
                        return text

                    __format__ = False  # Default __format__ value
                    ignore_cookie_set = False
                    try:
                        # print('Processing: %r with args: %r and kwargs: %r' % (func, args, kwargs, ))
                        # Pipe web input into Python web call
                        kwargs2 = _process_input(flask.request.args)
                        kwargs3 = _process_input(flask.request.form)
                        try:
                            # kwargs4 = _process_input(flask.request.get_json())
                            kwargs4 = ut.from_json(flask.request.data)
                        except Exception:
                            kwargs4 = {}
                        kwargs.update(kwargs2)
                        kwargs.update(kwargs3)
                        kwargs.update(kwargs4)

                        # Update the request object to include the final rectified inputs for possible future reference
                        flask.request.processed = ut.to_json(kwargs)

                        jQuery_callback = None
                        if 'callback' in kwargs and 'jQuery' in kwargs['callback']:
                            jQuery_callback = str(kwargs.pop('callback', None))
                            kwargs.pop('_', None)

                        # print('KWARGS:  %s' % (kwargs, ))
                        # print('COOKIES: %s' % (request.cookies, ))
                        __format__ = request.cookies.get('__format__', None)
                        __format__ = kwargs.pop('__format__', __format__)
                        if __format__ is not None:
                            __format__ = str(__format__).lower()
                            ignore_cookie_set = __format__ in ['onetime', 'true']
                            __format__ = __format__ in ['true', 'enabled', 'enable']

                        from wbia.web.app import PROMETHEUS

                        if PROMETHEUS:
                            exclude_tag_list = [
                                '/api/test/heartbeat/',
                                '/v0.1/wildbook/status/',
                                '/v0.1/vulcan/status/',
                            ]
                            tag = request.url_rule.rule
                            if tag not in exclude_tag_list:
                                ibs = flask.current_app.ibs
                                ibs.prometheus_increment_api(tag)

                        resp_tup = translate_wbia_webcall(func, **kwargs)
                        rawreturn, success, code, message = resp_tup
                    except WebException as webex:
                        # ut.printex(webex)
                        print('CAUGHT2: %r' % (webex,))
                        rawreturn = webex.get_rawreturn(
                            DEBUG_PYTHON_STACK_TRACE_JSON_RESPONSE
                        )
                        success = False
                        code = webex.code
                        message = webex.message
                        jQuery_callback = None
                    except Exception as ex:
                        print('CAUGHT2: %r' % (ex,))
                        # ut.printex(ex)
                        rawreturn = None
                        if DEBUG_PYTHON_STACK_TRACE_JSON_RESPONSE:
                            rawreturn = str(traceback.format_exc())
                        success = False
                        code = 500
                        message = str(ex)
                        # errmsg = str(ex)
                        # message = 'API error, Python Exception thrown: %s' % (errmsg)
                        if "'int' object is not iterable" in message:
                            rawreturn = """
                            HINT: the input for this call is most likely expected to be a list.
                            Try adding a comma at the end of the input (to cast the conversion into a list) or encapsulate the input with [].
                            """
                        jQuery_callback = None

                    # print('RECEIVED FORMAT: %r' % (__format__, ))

                    if __format__:
                        # Hack for readable error messages
                        webreturn = translate_wbia_webreturn(
                            rawreturn, success, code, message, jQuery_callback
                        )
                        webreturn = ut.repr3(ut.from_json(webreturn), strvals=True)

                        try:
                            from ansi2html import Ansi2HTMLConverter

                            conv = Ansi2HTMLConverter()
                            webreturn = conv.convert(webreturn)
                        except ImportError as ex:
                            ut.printex(ex, 'pip install ansi2html', iswarning=True)
                            webreturn = ut.strip_ansi(webreturn)
                            webreturn = (
                                '<p><samp>\n' + html_newlines(webreturn) + '\n</samp></p>'
                            )
                            webreturn = (
                                '<meta http-equiv="Content-Type" content="text/html;charset=ISO-8859-8">\n'
                                + webreturn
                            )

                        def get_func_href(funcname):
                            url = (
                                'http://'
                                + request.environ['HTTP_HOST']
                                + flask.url_for(funcname)
                                + '?__format__=True'
                            )
                            return '<a href="{url}">{url}</a>'.format(url=url)

                        if not success:
                            webreturn += (
                                '<pre>See logs for details: %s</pre>'
                                % get_func_href('get_current_log_text')
                            )
                            webreturn += (
                                '<pre>Might also look into db_info: %s</pre>'
                                % get_func_href('get_dbinfo')
                            )
                    else:
                        webreturn = translate_wbia_webreturn(
                            rawreturn, success, code, message, jQuery_callback
                        )
                        webreturn = ut.strip_ansi(webreturn)

                    resp = flask.make_response(webreturn, code)
                    resp.status_code = code

                    if not __format__:
                        resp.headers['Content-Type'] = 'application/json; charset=utf-8'
                        resp.headers['mimetype'] = 'application/json'

                    if not ignore_cookie_set:
                        if __format__:
                            resp.set_cookie('__format__', 'enabled')
                        else:
                            resp.set_cookie('__format__', '', expires=0)

                    return resp

                # return the original unmodified function
                if REMOTE_PROXY_URL is None:
                    return func
                else:
                    return remote_api_wrapper(func)

            return regsiter_closure

        return register_api
    else:
        return ut.dummy_args_decor


def authenticated():
    return get_user(username=None) is not None


def authenticate(username, **kwargs):
    get_user(username=username, **kwargs)


def deauthenticate():
    get_user(username=False)


def get_user(username=None, name=None, organization=None):
    USER_KEY = '_USER_'

    if USER_KEY not in session:
        session[USER_KEY] = None

    if username is not None:
        if username in [False]:
            # De-authenticate
            session[USER_KEY] = None
        else:
            # Authenticate
            assert isinstance(username, six.string_types), 'user must be a string'
            username = username.lower()
            session[USER_KEY] = {
                'username': username,
                'name': name,
                'organization': organization,
            }

    return session[USER_KEY]


def login_required_session(function):
    @wraps(function)
    def wrap(*args, **kwargs):
        if not authenticated():
            from wbia.web import appfuncs as appf

            refer = flask.request.url.replace(flask.request.url_root, '')
            refer = appf.encode_refer_url(refer)
            return flask.redirect(flask.url_for('login', refer=refer))
        else:
            return function(*args, **kwargs)

    return wrap


def get_wbia_flask_route(__name__):
    """For function calls that resolve to webpages and return html."""
    if __name__ == '__main__':
        return ut.dummy_args_decor
    if GLOBAL_APP_ENABLED:

        def register_route(
            rule,
            __route_prefix_check__=True,
            __route_postfix_check__=True,
            __route_authenticate__=True,
            __route_microsoft_check__=True,
            **options,
        ):

            # GLOBALLY DISABLE LOGINS
            __route_authenticate__ = False

            if MICROSOFT_API_ENABLED and __route_microsoft_check__:
                __route_authenticate__ = False
                if not rule.startswith(MICROSOFT_API_PREFIX):
                    # msg = 'Route rule=%r not allowed with the Microsoft format, ignoring.' % (rule, )
                    # warnings.warn(msg)
                    return ut.identity
                else:
                    print('Registering Route rule=%r' % (rule,))

            if __route_prefix_check__:
                assert not rule.startswith(
                    '/api/'
                ), 'Cannot start a route rule (%r) with the prefix "/api/"' % (rule,)
            else:
                __route_authenticate__ = False
            if __route_postfix_check__:
                assert rule.endswith('/'), 'A route should always end in a forward-slash'
            assert (
                'methods' in options
            ), 'A route should always have a specified methods list'

            # if '_' in rule:
            #     print('CONSIDER RENAMING RULE: %r' % (rule, ))
            # accpet args to flask.route
            def regsiter_closure(func):
                # make translation function in closure scope
                # and register it with flask.
                app = get_flask_app()

                # login_required = login_required_cas if HAS_FLASK_CAS else login_required_session
                login_required = login_required_session

                if not __route_authenticate__:
                    login_required = ut.identity

                @app.route(rule, **options)
                # @crossdomain(origin='*')
                # @authentication_user_only
                @login_required
                @wraps(func)
                def translated_call(**kwargs):
                    # debug = {'kwargs': kwargs}
                    try:
                        # Pipe web input into Python web call
                        kwargs2 = _process_input(flask.request.args)
                        kwargs3 = _process_input(flask.request.form)
                        try:
                            # kwargs4 = _process_input(flask.request.get_json())
                            kwargs4 = ut.from_json(flask.request.data)
                        except Exception:
                            kwargs4 = {}
                        kwargs.update(kwargs2)
                        kwargs.update(kwargs3)
                        kwargs.update(kwargs4)
                        jQuery_callback = None
                        if 'callback' in kwargs and 'jQuery' in kwargs['callback']:
                            jQuery_callback = str(kwargs.pop('callback', None))
                            kwargs.pop('_', None)

                        args = ()
                        print(
                            'Processing: %r with args: %r and kwargs: %r'
                            % (func, args, kwargs,)
                        )

                        from wbia.web.app import PROMETHEUS

                        if PROMETHEUS:
                            ibs = flask.current_app.ibs
                            tag = request.url_rule.rule
                            ibs.prometheus_increment_route(tag)

                        result = func(**kwargs)
                    except Exception as ex:
                        rawreturn = str(traceback.format_exc())
                        success = False
                        code = 400
                        message = 'Route error, Python Exception thrown: %r' % (str(ex),)
                        jQuery_callback = None
                        result = translate_wbia_webreturn(
                            rawreturn,
                            success,
                            code,
                            message,
                            jQuery_callback,
                            __skip_microsoft_validation__=True,
                        )
                    return result

                # wrp_getter_cacher = ut.preserve_sig(wrp_getter_cacher, getter_func)
                # return the original unmodified function
                return func

            return regsiter_closure

        return register_route
    else:
        return ut.dummy_args_decor


def api_remote_wbia(remote_wbia_url, remote_api_func, remote_wbia_port=5001, **kwargs):
    import requests

    if GLOBAL_APP_ENABLED and GLOBAL_APP is None:
        raise ValueError('Flask has not been initialized')
    api_name = remote_api_func.__name__
    route_list = list(GLOBAL_APP.url_map.iter_rules(api_name))
    assert len(route_list) == 1, 'More than one route resolved'
    route = route_list[0]
    api_route = route.rule
    assert api_route.startswith('/api/'), 'Must be an API route'
    method_list = sorted(list(route.methods - set(['HEAD', 'OPTIONS'])))
    remote_api_method = method_list[0].upper()

    assert api_route is not None, 'Route could not be found'

    args = (remote_wbia_url, remote_wbia_port, api_route)
    remote_api_url = 'http://%s:%s%s' % args
    headers = {'Authorization': get_url_authorization(remote_api_url)}

    for key in kwargs.keys():
        value = kwargs[key]
        if isinstance(value, (tuple, list, set)):
            value = str(list(value))
        kwargs[key] = value

    print('[REMOTE] %s' % ('-' * 80,))
    print('[REMOTE] Calling remote IBEIS API: %r' % (remote_api_url,))
    print('[REMOTE] \tMethod:  %r' % (remote_api_method,))
    if ut.DEBUG2 or ut.VERBOSE:
        print('[REMOTE] \tHeaders: %s' % (ut.repr2(headers),))
        print('[REMOTE] \tKWArgs:  %s' % (ut.repr2(kwargs),))

    # Make request to server
    try:
        if remote_api_method == 'GET':
            req = requests.get(remote_api_url, headers=headers, data=kwargs, verify=False)
        elif remote_api_method == 'POST':
            req = requests.post(
                remote_api_url, headers=headers, data=kwargs, verify=False
            )
        elif remote_api_method == 'PUT':
            req = requests.put(remote_api_url, headers=headers, data=kwargs, verify=False)
        elif remote_api_method == 'DELETE':
            req = requests.delete(
                remote_api_url, headers=headers, data=kwargs, verify=False
            )
        else:
            message = '_api_result got unsupported method=%r' % (remote_api_method,)
            raise KeyError(message)
    except requests.exceptions.ConnectionError as ex:
        message = '_api_result could not connect to server %s' % (ex,)
        raise IOError(message)
    response = req.text
    converted = ut.from_json(value)
    response = converted.get('response', None)
    print('[REMOTE] got response')
    if ut.DEBUG2:
        print('response = %s' % (response,))
    return response


##########################################################################################


def dev_autogen_explicit_imports():
    r"""
    CommandLine:
        python -m wbia --tf dev_autogen_explicit_imports

    Example:
        >>> # SCRIPT
        >>> from wbia.control.controller_inject import *  # NOQA
        >>> dev_autogen_explicit_imports()
    """
    import wbia  # NOQA

    classname = CONTROLLER_CLASSNAME
    print(ut.autogen_import_list(classname))


def dev_autogen_explicit_injects():
    r"""
    CommandLine:
        python -m wbia --tf dev_autogen_explicit_injects

    Example:
        >>> # SCRIPT
        >>> from wbia.control.controller_inject import *  # NOQA
        >>> dev_autogen_explicit_injects()
    """
    import wbia  # NOQA
    import wbia.control.IBEISControl

    classname = CONTROLLER_CLASSNAME
    regen_command = 'python -m wbia dev_autogen_explicit_injects'
    conditional_imports = [
        modname
        for modname in wbia.control.IBEISControl.AUTOLOAD_PLUGIN_MODNAMES
        if isinstance(modname, tuple)
    ]
    source_block = ut.autogen_explicit_injectable_metaclass(
        classname, regen_command, conditional_imports
    )
    dpath = ut.get_module_dir(wbia.control.IBEISControl)
    fpath = ut.unixjoin(dpath, '_autogen_explicit_controller.py')
    ut.writeto(fpath, source_block, verbose=2)


def make_ibs_register_decorator(modname):
    """builds variables and functions that controller injectable modules need."""
    if __name__ == '__main__':
        print('WARNING: cannot register controller functions as main')
    CLASS_INJECT_KEY = (CONTROLLER_CLASSNAME, modname)
    # Create dectorator to inject these functions into the IBEISController
    register_ibs_unaliased_method = ut.make_class_method_decorator(
        CLASS_INJECT_KEY, modname
    )

    # TODO Replace IBEISContoller INEJECTED MODULES with this one
    # INJECTED_MODULES.append(sys.modules[modname])

    def register_ibs_method(func):
        """ registers autogenerated functions with the utool class method injector """
        # func  = profile(func)
        register_ibs_unaliased_method(func)
        # aliastup = (func, '_injected_' + ut.get_funcname(func))
        # register_ibs_aliased_method(aliastup)
        return func

    return CLASS_INJECT_KEY, register_ibs_method


_decors_image = dtool.make_depcache_decors(const.IMAGE_TABLE)
_decors_annot = dtool.make_depcache_decors(const.ANNOTATION_TABLE)
_decors_part = dtool.make_depcache_decors(const.PART_TABLE)

register_preprocs = {
    'image': _decors_image['preproc'],
    'annot': _decors_annot['preproc'],
    'part': _decors_part['preproc'],
}
register_subprops = {
    'image': _decors_image['subprop'],
    'annot': _decors_annot['subprop'],
    'part': _decors_part['subprop'],
}


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.control.controller_inject
        python -m wbia.control.controller_inject --allexamples
        python -m wbia.control.controller_inject --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
