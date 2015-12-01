# -*- coding: utf-8 -*-
"""
TODO:
    Move flask registering into another file.
    Should also make the actual flask registration lazy.
    It should only be executed if a web instance is being started.
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import six
import sys
from datetime import timedelta
from functools import update_wrapper
from functools import wraps
from os.path import abspath, join, dirname
# import simplejson as json
#import json
#import pickle
#from six.moves import cPickle as pickle
import traceback
from hashlib import sha1
import os
#import numpy as np
import hmac
import string
import random
import requests
# <flask>
# TODO: allow optional flask import
try:
    import flask
    HAS_FLASK = True
except Exception as ex:
    HAS_FLASK = False
    ut.printex(ex, 'Missing flask', iswarning=True)
    if ut.STRICT:
        raise

try:
    from flask.ext.cors import CORS
    HAS_FLASK_CORS = True
except Exception as ex:
    HAS_FLASK_CORS = False
    ut.printex(ex, 'Missing flask.ext.cors', iswarning=True)
    if ut.SUPER_STRICT:
        raise
# </flask>
print, rrr, profile = ut.inject2(__name__, '[controller_inject]')


#INJECTED_MODULES = []
UTOOL_AUTOGEN_SPHINX_RUNNING = not (
    os.environ.get('UTOOL_AUTOGEN_SPHINX_RUNNING', 'OFF') == 'OFF')

GLOBAL_APP_ENABLED = (not UTOOL_AUTOGEN_SPHINX_RUNNING and
                      not ut.get_argflag('--no-flask') and HAS_FLASK)
GLOBAL_APP_NAME = 'IBEIS'
GLOBAL_APP_SECRET = 'CB73808F-A6F6-094B-5FCD-385EBAFF8FC0'

GLOBAL_APP = None
GLOBAL_CORS = None
#JSON_PYTHON_OBJECT_TAG = '__PYTHON_OBJECT__'

# REMOTE_PROXY_URL = 'dozer.cs.rpi.edu'
REMOTE_PROXY_URL = None
REMOTE_PROXY_PORT = 5001


CONTROLLER_CLASSNAME = 'IBEISController'


def get_flask_app():
    # TODO this should be initialized explicity in main_module.py only if needed
    global GLOBAL_APP
    global GLOBAL_CORS
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
        GLOBAL_APP = flask.Flask(GLOBAL_APP_NAME,
                                 template_folder=tempalte_dpath,
                                 static_folder=static_dpath)
        if HAS_FLASK_CORS:
            GLOBAL_CORS = CORS(GLOBAL_APP, resources={r"/api/*": {"origins": "*"}})  # NOQA
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


class WebException(Exception):
    def __init__(self, message, code=400):
        self.code = code
        self.message = message

    def __str__(self):
        return repr('%r: %r' % (self.code, self.message, ))


def translate_ibeis_webreturn(rawreturn, success=True, code=None, message=None,
                              jQuery_callback=None, cache=None):
    if code is None:
        code = ''
    if message is None:
        message = ''
    if cache is None:
        cache = -1
    template = {
        'status': {
            'success': success,
            'code':    code,
            'message': message,
            'cache':   cache,
        },
        'response' : rawreturn
    }
    #response = json.dumps(template, cls=JSONPythonObjectEncoder)
    response = ut.to_json(template)
    if jQuery_callback is not None and isinstance(jQuery_callback, str):
        print('[web] Including jQuery callback function: %r' % (jQuery_callback, ))
        response = '%s(%s)' % (jQuery_callback, response)
    return response


def translate_ibeis_webcall(func, *args, **kwargs):
    #print('Processing: %r with args: %r and kwargs: %r' % (func, args, kwargs, ))
    def _process_input(multidict):
        for (arg, value) in multidict.iterlists():
            if len(value) > 1:
                raise WebException('Cannot specify a parameter more than once: %r' % (arg, ))
            value = str(value[0])
            if ',' in value and '[' not in value and ']' not in value:
                value = '[%s]' % (value, )
            if value in ['True', 'False']:
                value = value.lower()
            try:
                #converted = json.loads(value, object_hook=_as_python_object)
                converted = ut.from_json(value)
            except Exception:
                # try making string and try again...
                value = '"%s"' % (value, )
                #converted = json.loads(value, object_hook=_as_python_object)
                converted = ut.from_json(value)
            if arg.endswith('_list') and not isinstance(converted, (list, tuple)):
                if isinstance(converted, str) and ',' in converted:
                    converted = converted.strip().split(',')
                else:
                    converted = [converted]
            # Allow JSON formatted strings to be placed into note fields
            if ((arg.endswith('note_list') or arg.endswith('notes_list')) and
               isinstance(converted, (list, tuple))):
                type_ = type(converted)
                temp_list = []
                for _ in converted:
                    if isinstance(_, dict):
                        temp_list.append('%s' % (_, ))
                    else:
                        temp_list.append(_)
                converted = type_(temp_list)
            kwargs[arg] = converted
    # Pipe web input into Python web call
    _process_input(flask.request.args)
    _process_input(flask.request.form)
    jQuery_callback = None
    if 'callback' in kwargs and 'jQuery' in kwargs['callback']:
        jQuery_callback = str(kwargs.pop('callback', None))
        kwargs.pop('_', None)
    print('Calling: %r with args: %r and kwargs: %r' % (func, args, kwargs, ))
    ibs = flask.current_app.ibs
    assert len(args) == 0, ''
    try:
        #output = func(*args, **kwargs)
        output = func(**kwargs)
    except TypeError:
        #output = func(ibs=ibs, *args, **kwargs)
        try:
            output = func(ibs=ibs, **kwargs)
        except Exception:
            print('Error in translate ibeis web call')
            print('kwargs = %r' % (kwargs,))
            print('args = %r' % (args,))
            print('flask.request.args = %r' % (flask.request.args,))
            print('flask.request.form = %r' % (flask.request.form,))
            raise
    return (output, True, 200, None, jQuery_callback)


def authentication_challenge():
    """
    Sends a 401 response that enables basic auth
    """
    rawreturn = ''
    success = False
    code = 401
    message = 'Could not verify your authentication, login with proper credentials.'
    jQuery_callback = None
    webreturn = translate_ibeis_webreturn(rawreturn, success, code, message, jQuery_callback)
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
    return username == 'ibeis' and password == 'ibeis'


def authentication_user_only(func):
    @wraps(func)
    def wrp_authenticate_user(*args, **kwargs):
        if not authentication_user_validate():
            return authentication_challenge()
        return func(*args, **kwargs)
    #wrp_authenticate_user = ut.preserve_sig(wrp_authenticate_user, func)
    return wrp_authenticate_user


def create_key():
    hyphen_list = [8, 13, 18, 23]
    key_list = ['-' if _ in hyphen_list else random.choice(string.hexdigits)
                for _ in xrange(36) ]
    return ''.join(key_list).upper()


def get_signature(key, message):
    return str(hmac.new(key, message, sha1).digest().encode("base64").rstrip('\n'))


def get_url_authorization(url):
    hash_ = get_signature(GLOBAL_APP_SECRET, url)
    hash_challenge = '%s:%s' % (GLOBAL_APP_NAME, hash_, )
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
        return string[:index] + string[index + 1:]

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


def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
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
            return api_remote_ibeis(REMOTE_PROXY_URL, func, REMOTE_PROXY_PORT, **kwargs)
    remote_api_call = ut.preserve_sig(remote_api_call, func)
    return remote_api_call


def get_ibeis_flask_api(__name__, DEBUG_PYTHON_STACK_TRACE_JSON_RESPONSE=True):
    if __name__ == '__main__':
        return ut.dummy_args_decor
    if GLOBAL_APP_ENABLED:
        def register_api(rule, **options):
            # accpet args to flask.route
            def regsiter_closure(func):
                # make translation function in closure scope
                # and register it with flask.
                app = get_flask_app()
                @app.route(rule, **options)
                # @crossdomain(origin='*')
                # @authentication_either
                @wraps(func)
                #def translated_call(*args, **kwargs):
                def translated_call(**kwargs):
                    #from flask import make_response
                    try:
                        #values = translate_ibeis_webcall(func, *args, **kwargs)
                        values = translate_ibeis_webcall(func, **kwargs)
                        rawreturn, success, code, message, jQuery_callback = values
                    except WebException as webex:
                        rawreturn = ''
                        print(traceback.format_exc())
                        if DEBUG_PYTHON_STACK_TRACE_JSON_RESPONSE:
                            rawreturn = str(traceback.format_exc())
                        success = False
                        code = webex.code
                        message = webex.message
                        jQuery_callback = None
                    except Exception as ex:
                        rawreturn = ''
                        ut.printex(ex)
                        #print(traceback.format_exc())
                        if DEBUG_PYTHON_STACK_TRACE_JSON_RESPONSE:
                            rawreturn = str(traceback.format_exc())
                        success = False
                        code = 500
                        message = 'API error, Python Exception thrown: %r' % (str(ex))
                        if "'int' object is not iterable" in message:
                            rawreturn = (
                                'HINT: the input for this call is most likely '
                                'expected to be a list.  Try adding a comma at '
                                'the end of the input (to cast the conversion '
                                'into a list) or encapsualte the input with '
                                '[].')
                        jQuery_callback = None
                    webreturn = translate_ibeis_webreturn(rawreturn, success,
                                                          code, message,
                                                          jQuery_callback)
                    return flask.make_response(webreturn, code)
                # return the original unmodified function
                if REMOTE_PROXY_URL is None:
                    return func
                else:
                    return remote_api_wrapper(func)
            return regsiter_closure
        return register_api
    else:
        return ut.dummy_args_decor


def get_ibeis_flask_route(__name__):
    if __name__ == '__main__':
        return ut.dummy_args_decor
    if GLOBAL_APP_ENABLED:
        def register_route(rule, **options):
            # accpet args to flask.route
            def regsiter_closure(func):
                # make translation function in closure scope
                # and register it with flask.
                app = get_flask_app()
                @app.route(rule, **options)
                # @crossdomain(origin='*')
                # @authentication_user_only
                @wraps(func)
                #def translated_call(*args, **kwargs):
                def translated_call(**kwargs):
                    try:
                        #result = func(*args, **kwargs)
                        result = func(**kwargs)
                    except Exception as ex:
                        rawreturn = str(traceback.format_exc())
                        success = False
                        code = 400
                        message = (
                            'Route error, Python Exception thrown: %r' %
                            (str(ex), ))
                        jQuery_callback = None
                        result = translate_ibeis_webreturn(rawreturn, success,
                                                           code, message,
                                                           jQuery_callback)
                    return result
                #wrp_getter_cacher = ut.preserve_sig(wrp_getter_cacher, getter_func)
                # return the original unmodified function
                return func
            return regsiter_closure
        return register_route
    else:
        return ut.dummy_args_decor


def api_remote_ibeis(remote_ibeis_url, remote_api_func, remote_ibeis_port=5001,
                     **kwargs):
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

    args = (remote_ibeis_url, remote_ibeis_port, api_route)
    remote_api_url = 'http://%s:%s%s' % args
    headers = {
        'Authorization': get_url_authorization(remote_api_url)
    }

    for key in kwargs.keys():
        value = kwargs[key]
        if isinstance(value, (tuple, list, set)):
            value = str(list(value))
        kwargs[key] = value

    print('[REMOTE] %s' % ('-' * 80, ))
    print('[REMOTE] Calling remote IBEIS API: %r' % (remote_api_url, ))
    print('[REMOTE] \tMethod:  %r' % (remote_api_method, ))
    print('[REMOTE] \tHeaders: %s' % (ut.dict_str(headers), ))
    print('[REMOTE] \tKWArgs:  %s' % (ut.dict_str(kwargs), ))

    # Make request to server
    try:
        if remote_api_method == 'GET':
            req = requests.get(remote_api_url, headers=headers, data=kwargs,
                               verify=False)
        elif remote_api_method == 'POST':
            req = requests.post(remote_api_url, headers=headers, data=kwargs,
                                verify=False)
        elif remote_api_method == 'PUT':
            req = requests.put(remote_api_url, headers=headers, data=kwargs,
                               verify=False)
        elif remote_api_method == 'DELETE':
            req = requests.delete(remote_api_url, headers=headers, data=kwargs,
                                  verify=False)
        else:
            message = '_api_result got unsupported method=%r' % (remote_api_method, )
            raise KeyError(message)
    except requests.exceptions.ConnectionError as ex:
        message = '_api_result could not connect to server %s' % (ex, )
        raise IOError(message)
    response = req.text
    #converted = json.loads(response, object_hook=_as_python_object)
    converted = ut.from_json(value)
    response = converted.get('response', None)
    print('response = %s' % (response,))
    return response


##########################################################################################

def dev_autogen_explicit_imports():
    r"""
    CommandLine:
        python -m ibeis --tf dev_autogen_explicit_imports

    Example:
        >>> # SCRIPT
        >>> from ibeis.control.controller_inject import *  # NOQA
        >>> dev_autogen_explicit_imports()
    """
    import ibeis  # NOQA
    classname = CONTROLLER_CLASSNAME
    print(ut.autogen_import_list(classname))


def dev_autogen_explicit_injects():
    r"""
    CommandLine:
        python -m ibeis --tf dev_autogen_explicit_injects

    Example:
        >>> # SCRIPT
        >>> from ibeis.control.controller_inject import *  # NOQA
        >>> dev_autogen_explicit_injects()
    """
    import ibeis  # NOQA
    classname = CONTROLLER_CLASSNAME
    regen_command = (
        'python -m ibeis.control.controller_inject '
        '--exec-dev_autogen_explicit_injects')
    import ibeis.control.IBEISControl
    conditional_imports = [
        modname for modname in ibeis.control.IBEISControl.inject_modnames
        if isinstance(modname, tuple)
    ]
    source_block = ut.autogen_explicit_injectable_metaclass(
        classname, regen_command, conditional_imports)
    dpath = ut.get_module_dir(ibeis.control.IBEISControl)
    fpath = ut.unixjoin(dpath, '_autogen_explicit_controller.py')
    ut.writeto(fpath, source_block)


def make_ibs_register_decorator(modname):
    """
    builds variables and functions that controller injectable modules need
    """
    #global INJECTED_MODULES
    if __name__ == '__main__':
        print('WARNING: cannot register controller functions as main')
    #else:
    CLASS_INJECT_KEY = (CONTROLLER_CLASSNAME, modname)
    # Create dectorator to inject these functions into the IBEISController
    #register_ibs_aliased_method   = ut.make_class_method_decorator(CLASS_INJECT_KEY)
    register_ibs_unaliased_method = ut.make_class_method_decorator(
        CLASS_INJECT_KEY, modname)

    # TODO Replace IBEISContoller INEJECTED MODULES with this one
    #INJECTED_MODULES.append(sys.modules[modname])

    def register_ibs_method(func):
        """ registers autogenerated functions with the utool class method injector """
        #func  = profile(func)
        register_ibs_unaliased_method(func)
        #aliastup = (func, '_injected_' + ut.get_funcname(func))
        #register_ibs_aliased_method(aliastup)
        return func
    return CLASS_INJECT_KEY, register_ibs_method


class ExternalStorageException(Exception):
    """ TODO move to a common place for templated SQL functions """
    def __init__(self, *args, **kwargs):
        super(ExternalStorageException, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.controller_inject
        python -m ibeis.control.controller_inject --allexamples
        python -m ibeis.control.controller_inject --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
