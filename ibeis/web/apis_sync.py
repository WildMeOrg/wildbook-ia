# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado

SeeAlso:
    routes.turk_identification
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ibeis.control import controller_inject
from flask import url_for, request, current_app  # NOQA
import numpy as np   # NOQA
import utool as ut
import uuid
import requests
ut.noinject('[apis_sync]')


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_api   = controller_inject.get_ibeis_flask_api(__name__)


REMOTE = 'http://52.41.169.106:5555'  # AMI IN AWS CALLED "IBEIS IA DETECT"
REMOTE_UUID = uuid.UUID('e468d14b-3a39-4165-8f62-16f9e3deea39')


def _api_url(func_name):
    func_api_url = url_for(func_name)
    print(func_api_url)
    api_url = '%s%s' % (REMOTE, func_api_url, )
    return api_url


def _verify_response(response):
    response_dict = ut.from_json(response.text)
    print(response_dict)
    status = response_dict.get('status', {})
    assert status.get('success', False)
    response = response_dict.get('response', None)
    return response


def _get(**kwargs):
    api_url = _api_url(**kwargs)
    response = requests.get(api_url)
    return _verify_response(response)


def assert_remote_online(ibs):
    uuid = _get(func_name='get_db_init_uuid')
    version = _get(func_name='get_database_version')
    assert uuid == REMOTE_UUID
    assert version == ibs.get_database_version()


@register_api('/api/sync/', methods=['GET'])
def _detect_remote_sync_images(ibs, gid_list=None):
    assert_remote_online(ibs)
    return 'online'


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.app
        python -m ibeis.web.app --allexamples
        python -m ibeis.web.app --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
