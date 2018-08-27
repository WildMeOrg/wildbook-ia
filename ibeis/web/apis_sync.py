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
import six
ut.noinject('[apis_sync]')


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_api   = controller_inject.get_ibeis_flask_api(__name__)


REMOTE_TESTING = False


if REMOTE_TESTING:
    REMOTE_DOMAIN = '127.0.0.1'
    REMOTE_PORT = '5555'
    REMOTE_UUID = None
else:
    REMOTE_DOMAIN = '35.161.135.191'
    REMOTE_PORT = '5555'
    REMOTE_UUID = 'e468d14b-3a39-4165-8f62-16f9e3deea39'

    remote_args = ut.get_arg_dict()
    REMOTE_DOMAIN = remote_args.get('sync-domain', REMOTE_DOMAIN)
    REMOTE_PORT = remote_args.get('sync-port', REMOTE_PORT)
    REMOTE_UUID = remote_args.get('sync-uuid', REMOTE_UUID)

    if REMOTE_UUID in [True, '', 'none', 'None']:
        REMOTE_UUID = None


REMOTE_URL = 'http://%s:%s' % (REMOTE_DOMAIN, REMOTE_PORT, )
REMOTE_UUID = None if REMOTE_UUID is None else uuid.UUID(REMOTE_UUID)


def _construct_route_url(route_rule):
    if not route_rule.startswith('/'):
        route_rule = '/' + route_rule
    if not route_rule.endswith('/'):
        route_rule = route_rule + '/'
    route_url = '%s%s' % (REMOTE_URL, route_rule, )
    return route_url


def _verify_response(response):
    try:
        response_dict = ut.from_json(response.text)
    except ValueError:
        raise AssertionError('Could not get valid JSON response from server')
    status = response_dict.get('status', {})
    assert status.get('success', False)
    response = response_dict.get('response', None)
    return response


def _get(route_rule, **kwargs):
    route_url = _construct_route_url(route_rule)
    response = requests.get(route_url, **kwargs)
    return _verify_response(response)


def _assert_remote_online(ibs):
    try:
        version = _get('/api/core/db/version/')
        uuid = _get('/api/core/db/uuid/init/')
        assert version == ibs.get_database_version()
        if REMOTE_UUID is not None:
            assert uuid == REMOTE_UUID
    except:
        raise IOError('Remote IBEIS DETECT database offline at %s' % (REMOTE_URL, ))


@register_ibs_method
def _detect_remote_push_images(ibs, gid_list):
    route_url = _construct_route_url('/api/upload/image/')

    num_images = len(gid_list)
    image_path_list = ibs.get_image_paths(gid_list)
    for index, image_path in enumerate(image_path_list):
        print('\tSending %d / %d: %r' % (index, num_images, image_path, ))
        file_dict = {
            'image': open(image_path, 'rb'),
        }
        response = requests.post(route_url, files=file_dict)
        _verify_response(response)
        print('\t...sent')


@register_ibs_method
def _detect_remote_push_imageset(ibs, image_uuid_list):
    route_url = _construct_route_url('/api/image/imageset/text/json/')

    db_name = ibs.get_dbname()
    db_uuid = ibs.get_db_init_uuid()
    time_str = ut.get_timestamp()
    imageset_text = 'Sync from %s (%s) at %s' % (db_name, db_uuid, time_str)
    imageset_text_list = [imageset_text] * len(image_uuid_list)

    data_dict = {
        'image_uuid_list': image_uuid_list,
        'imageset_text_list': imageset_text_list,
    }
    for key in data_dict:
        data_dict[key] = ut.to_json(data_dict[key])
    response = requests.put(route_url, data=data_dict)
    _verify_response(response)


@register_ibs_method
def _detect_remote_push_annots(ibs, aid_list):
    route_url = _construct_route_url('/api/annot/json/')

    print('\tSending...')
    data_dict = {
        'image_uuid_list': ibs.get_annot_image_uuids(aid_list),
        'annot_uuid_list': ibs.get_annot_uuids(aid_list),
        'annot_bbox_list': ibs.get_annot_bboxes(aid_list),
    }
    for key in data_dict:
        data_dict[key] = ut.to_json(data_dict[key])
    response = requests.post(route_url, data=data_dict)
    _verify_response(response)
    print('\t...sent')


@register_ibs_method
def _detect_remote_push_metadata(ibs, route_rule, uuid_str, value_str,
                                 uuid_list, value_list):
    route_url = _construct_route_url(route_rule)

    print('\tSetting %s metadata for %s' % (route_rule, uuid_str, ))
    data_dict = {
        uuid_str: uuid_list,
        value_str: value_list,
    }
    for key in data_dict:
        data_dict[key] = ut.to_json(data_dict[key])
    response = requests.put(route_url, data=data_dict)
    _verify_response(response)
    print('\t...set')


@register_ibs_method
def _detect_remote_push_annot_metadata(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    ibs._detect_remote_push_metadata('/api/annot/bbox/json/',
                                     'annot_uuid_list',
                                     'bbox_list',
                                     annot_uuid_list,
                                     ibs.get_annot_bboxes(aid_list))
    ibs._detect_remote_push_metadata('/api/annot/theta/json/',
                                     'annot_uuid_list',
                                     'theta_list',
                                     annot_uuid_list,
                                     ibs.get_annot_thetas(aid_list))
    ibs._detect_remote_push_metadata('/api/annot/viewpoint/json/',
                                     'annot_uuid_list',
                                     'viewpoint_list',
                                     annot_uuid_list,
                                     ibs.get_annot_viewpoints(aid_list))
    ibs._detect_remote_push_metadata('/api/annot/quality/text/json/',
                                     'annot_uuid_list',
                                     'quality_text_list',
                                     annot_uuid_list,
                                     ibs.get_annot_quality_texts(aid_list))
    ibs._detect_remote_push_metadata('/api/annot/species/json/',
                                     'annot_uuid_list',
                                     'species_text_list',
                                     annot_uuid_list,
                                     ibs.get_annot_species_texts(aid_list))
    ibs._detect_remote_push_metadata('/api/annot/multiple/json/',
                                     'annot_uuid_list',
                                     'flag_list',
                                     annot_uuid_list,
                                     ibs.get_annot_multiple(aid_list))
    ibs._detect_remote_push_metadata('/api/annot/interest/json/',
                                     'annot_uuid_list',
                                     'flag_list',
                                     annot_uuid_list,
                                     ibs.get_annot_interest(aid_list))
    ibs._detect_remote_push_metadata('/api/annot/tags/json/',
                                     'annot_uuid_list',
                                     'annot_tags_list',
                                     annot_uuid_list,
                                     ibs.get_annot_tag_text(aid_list))
    ibs._detect_remote_push_metadata('/api/annot/name/text/json/',
                                     'annot_uuid_list',
                                     'name_text_list',
                                     annot_uuid_list,
                                     ibs.get_annot_name_texts(aid_list))


@register_ibs_method
def _detect_remote_push_parts(ibs, part_rowid_list):
    route_url = _construct_route_url('/api/part/json/')

    print('\tSending...')
    data_dict = {
        'annot_uuid_list': ibs.get_part_annot_uuids(part_rowid_list),
        'part_uuid_list': ibs.get_part_uuids(part_rowid_list),
        'part_bbox_list': ibs.get_part_bboxes(part_rowid_list),
    }
    for key in data_dict:
        data_dict[key] = ut.to_json(data_dict[key])
    response = requests.post(route_url, data=data_dict)
    _verify_response(response)
    print('\t...sent')


@register_ibs_method
def _detect_remote_push_part_metadata(ibs, part_uuid_list):
    part_rowid_list = ibs.get_part_rowids_from_uuid(part_uuid_list)
    ibs._detect_remote_push_metadata('/api/part/bbox/json/',
                                     'part_uuid_list',
                                     'bbox_list',
                                     part_uuid_list,
                                     ibs.get_part_bboxes(part_rowid_list))
    ibs._detect_remote_push_metadata('/api/part/theta/json/',
                                     'part_uuid_list',
                                     'theta_list',
                                     part_uuid_list,
                                     ibs.get_part_thetas(part_rowid_list))
    ibs._detect_remote_push_metadata('/api/part/viewpoint/json/',
                                     'part_uuid_list',
                                     'viewpoint_list',
                                     part_uuid_list,
                                     ibs.get_part_viewpoints(part_rowid_list))
    ibs._detect_remote_push_metadata('/api/part/quality/text/json/',
                                     'part_uuid_list',
                                     'quality_text_list',
                                     part_uuid_list,
                                     ibs.get_part_quality_texts(part_rowid_list))  # NOQA
    ibs._detect_remote_push_metadata('/api/part/type/json/',
                                     'part_uuid_list',
                                     'type_text_list',
                                     part_uuid_list,
                                     ibs.get_part_types(part_rowid_list))
    ibs._detect_remote_push_metadata('/api/part/tags/json/',
                                     'part_uuid_list',
                                     'part_tags_list',
                                     part_uuid_list,
                                     ibs.get_part_tag_text(part_rowid_list))


@register_ibs_method
@register_api('/api/sync/', methods=['GET'])
def detect_remote_sync_images(ibs, gid_list=None,
                              only_sync_missing_images=True):
    _assert_remote_online(ibs)

    if gid_list is None:
        gid_list = ibs.get_valid_gids()

    confirm_list = [
        ut.random_nonce()[:5]
        for _ in range(3)
    ]
    confirm_str = '-'.join(confirm_list)
    print('You are about to submit %d images to a remote DETECT database at %r with UUID=%r.' % (len(gid_list), REMOTE_URL, REMOTE_UUID, ))
    print('Only do this action if you are confident in the detection accuracy of the images, annotations, annotation metadata, parts and part metadata.')
    print('In order to continue, please type exactly the confirmation string %r' % (confirm_str, ))

    if six.PY2:
        input_func = raw_input
    else:
        input_func = input
    response_str = input_func('Confirmation string [Empty to abort]: ')
    response_str = response_str.lower()
    assert confirm_str == response_str, 'Confirmation string mismatch, aborting...'

    ############################################################################

    # Sync images
    image_uuid_list = ibs.get_image_uuids(gid_list)
    image_uuid_list_ = _get('/api/image/json/')

    missing_gid_list = [
        gid
        for gid, image_uuid in list(zip(gid_list, image_uuid_list))
        if image_uuid not in image_uuid_list_
    ]

    num_missing = len(missing_gid_list)
    if num_missing > 0:
        print('Need to push %d images...' % (num_missing, ))
        ibs._detect_remote_push_images(missing_gid_list)
        print('...pushed')

    # Filter only missing
    gid_list_ = missing_gid_list if only_sync_missing_images else gid_list
    image_uuid_list_ = ibs.get_image_uuids(gid_list_)

    ############################################################################

    # Sync imageset
    print('Setting imageset...')
    ibs._detect_remote_push_imageset(image_uuid_list_)
    print('...set')

    ############################################################################

    # Sync annots
    aid_list = ut.flatten(ibs.get_image_aids(gid_list_))
    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    annot_uuid_list_ = _get('/api/annot/json/')

    missing_aid_list = [
        aid
        for aid, annot_uuid in list(zip(aid_list, annot_uuid_list))
        if annot_uuid not in annot_uuid_list_
    ]

    num_missing = len(missing_aid_list)
    if num_missing > 0:
        print('Need to push %d annots...' % (num_missing, ))
        ibs._detect_remote_push_annots(missing_aid_list)
        print('...pushed')

    ############################################################################

    # Sync annotation metadata
    print('Synching annotation metadata...')
    if len(annot_uuid_list) > 0:
        ibs._detect_remote_push_annot_metadata(annot_uuid_list)
    print('...synched')

    ############################################################################

    # Sync parts
    part_rowid_list = ut.flatten(ibs.get_annot_part_rowids(aid_list))
    part_uuid_list = ibs.get_part_uuids(part_rowid_list)
    part_uuid_list_ = _get('/api/part/json/')

    missing_part_rowid_list = [
        part_rowid
        for part_rowid, part_uuid in list(zip(part_rowid_list, part_uuid_list))
        if part_uuid not in part_uuid_list_
    ]

    num_missing = len(missing_part_rowid_list)
    if num_missing > 0:
        print('Need to push %d parts...' % (num_missing, ))
        ibs._detect_remote_push_parts(missing_part_rowid_list)
        print('...pushed')

    ############################################################################

    # Sync part metadata
    print('Synching part metadata...')
    if len(part_uuid_list) > 0:
        ibs._detect_remote_push_part_metadata(part_uuid_list)
    print('...synched')


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
