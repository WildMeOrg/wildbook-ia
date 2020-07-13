# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado

SeeAlso:
    routes.turk_identification
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from wbia.control import controller_inject
from flask import url_for, request, current_app  # NOQA
import numpy as np  # NOQA
import utool as ut
import uuid
import requests
import six
import json

(print, rrr, profile) = ut.inject2(__name__)

CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)
register_api = controller_inject.get_wbia_flask_api(__name__)


REMOTE_TESTING = True


if not REMOTE_TESTING:
    REMOTE_DOMAIN = '127.0.0.1'
    REMOTE_PORT = '5555'
    REMOTE_UUID = None
else:
    # info for flukebook background server
    REMOTE_DOMAIN = 'kaiju.dyn.wildme.io'
    REMOTE_PORT = '5015'
    # https://kaiju.dyn.wildme.io:5015/api/core/db/uuid/init/
    REMOTE_UUID = '511ab1d0-5af4-4808-b0a7-c8115240ab8e'

    remote_args = ut.get_arg_dict()
    REMOTE_DOMAIN = remote_args.get('sync-domain', REMOTE_DOMAIN)
    REMOTE_PORT = remote_args.get('sync-port', REMOTE_PORT)
    REMOTE_UUID = remote_args.get('sync-uuid', REMOTE_UUID)

    if REMOTE_UUID in [True, '', 'none', 'None']:
        REMOTE_UUID = None


REMOTE_URL = 'http://%s:%s' % (REMOTE_DOMAIN, REMOTE_PORT,)
REMOTE_UUID = None if REMOTE_UUID is None else uuid.UUID(REMOTE_UUID)


def _construct_route_url(route_rule):
    if not route_rule.startswith('/'):
        route_rule = '/' + route_rule
    if not route_rule.endswith('/'):
        route_rule = route_rule + '/'
    route_url = '%s%s' % (REMOTE_URL, route_rule,)
    return route_url


@register_ibs_method
def _construct_route_url_ibs(ibs, route_rule):
    if not route_rule.startswith('/'):
        route_rule = '/' + route_rule
    if not route_rule.endswith('/'):
        route_rule = route_rule + '/'
    route_url = '%s%s' % (REMOTE_URL, route_rule,)
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


@register_ibs_method
def _verify_response_ibs(ibs, response):
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


@register_ibs_method
def _get_ibs(ibs, route_rule, **kwargs):
    route_url = ibs._construct_route_url_ibs(route_rule)
    response = requests.get(route_url, **kwargs)
    return ibs._verify_response_ibs(response)


def _assert_remote_online(ibs):
    try:
        version = _get('/api/core/db/version/')
        uuid = _get('/api/core/db/uuid/init/')
        assert version == ibs.get_database_version()
        if REMOTE_UUID is not None:
            assert uuid == REMOTE_UUID
    except Exception:
        raise IOError('Remote IBEIS DETECT database offline at %s' % (REMOTE_URL,))


@register_ibs_method
def _detect_remote_push_images(ibs, gid_list):
    route_url = _construct_route_url('/api/upload/image/')

    num_images = len(gid_list)
    image_path_list = ibs.get_image_paths(gid_list)
    for index, image_path in enumerate(image_path_list):
        print('\tSending %d / %d: %r' % (index, num_images, image_path,))
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
def _detect_remote_push_metadata(
    ibs, route_rule, uuid_str, value_str, uuid_list, value_list
):
    route_url = _construct_route_url(route_rule)

    print('\tSetting %s metadata for %s' % (route_rule, uuid_str,))
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
    ibs._detect_remote_push_metadata(
        '/api/annot/bbox/json/',
        'annot_uuid_list',
        'bbox_list',
        annot_uuid_list,
        ibs.get_annot_bboxes(aid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/annot/theta/json/',
        'annot_uuid_list',
        'theta_list',
        annot_uuid_list,
        ibs.get_annot_thetas(aid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/annot/viewpoint/json/',
        'annot_uuid_list',
        'viewpoint_list',
        annot_uuid_list,
        ibs.get_annot_viewpoints(aid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/annot/quality/text/json/',
        'annot_uuid_list',
        'quality_text_list',
        annot_uuid_list,
        ibs.get_annot_quality_texts(aid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/annot/species/json/',
        'annot_uuid_list',
        'species_text_list',
        annot_uuid_list,
        ibs.get_annot_species_texts(aid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/annot/multiple/json/',
        'annot_uuid_list',
        'flag_list',
        annot_uuid_list,
        ibs.get_annot_multiple(aid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/annot/interest/json/',
        'annot_uuid_list',
        'flag_list',
        annot_uuid_list,
        ibs.get_annot_interest(aid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/annot/tags/json/',
        'annot_uuid_list',
        'annot_tags_list',
        annot_uuid_list,
        ibs.get_annot_tag_text(aid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/annot/name/text/json/',
        'annot_uuid_list',
        'name_text_list',
        annot_uuid_list,
        ibs.get_annot_name_texts(aid_list),
    )


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
    ibs._detect_remote_push_metadata(
        '/api/part/bbox/json/',
        'part_uuid_list',
        'bbox_list',
        part_uuid_list,
        ibs.get_part_bboxes(part_rowid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/part/theta/json/',
        'part_uuid_list',
        'theta_list',
        part_uuid_list,
        ibs.get_part_thetas(part_rowid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/part/viewpoint/json/',
        'part_uuid_list',
        'viewpoint_list',
        part_uuid_list,
        ibs.get_part_viewpoints(part_rowid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/part/quality/text/json/',
        'part_uuid_list',
        'quality_text_list',
        part_uuid_list,
        ibs.get_part_quality_texts(part_rowid_list),
    )  # NOQA
    ibs._detect_remote_push_metadata(
        '/api/part/type/json/',
        'part_uuid_list',
        'type_text_list',
        part_uuid_list,
        ibs.get_part_types(part_rowid_list),
    )
    ibs._detect_remote_push_metadata(
        '/api/part/tags/json/',
        'part_uuid_list',
        'part_tags_list',
        part_uuid_list,
        ibs.get_part_tag_text(part_rowid_list),
    )


@register_ibs_method
def _sync_get_remote_info(ibs):
    route_url = _construct_route_url('/api/core/db/info/')
    response = requests.get(route_url)
    _verify_response(response)
    print(response.text)
    return response


@register_ibs_method
def _sync_get_species_aids(ibs, species_name):
    route_url = _construct_route_url('/api/annot/')
    data_dict = {
        'species': species_name,
    }
    response = requests.get(route_url, data=data_dict)
    _verify_response(response)
    return response.json()['response']


# eubalaena_australis, eubalaena_glacialis
@register_ibs_method
def _sync_get_image(ibs, remote_image_rowid):
    endpoint = '/api/image/%s/' % remote_image_rowid
    return _get(endpoint)


@register_ibs_method
def sync_get_training_data(ibs, species_name, force_update=False, **kwargs):

    aid_list = ibs._sync_get_training_aids(species_name, **kwargs)
    ann_uuids = ibs._sync_get_annot_endpoint('/api/annot/uuid/', aid_list)
    local_aids = []

    # avoid re-downloading annots based on UUID
    if not force_update:
        local_ann_uuids = set(ibs.get_valid_annot_uuids())
        remote_ann_uuids = set(ann_uuids)
        dupe_uuids = local_ann_uuids & remote_ann_uuids  # & is set-union
        new_uuids = remote_ann_uuids - dupe_uuids
        local_aids = ibs.get_annot_aids_from_uuid(dupe_uuids)
        aid_list = ibs._sync_get_aids_for_uuids(new_uuids)
        ann_uuids = ibs._sync_get_annot_endpoint('/api/annot/uuid/', aid_list)

    # get needed info
    viewpoints = ibs._sync_get_annot_endpoint('/api/annot/viewpoint/', aid_list)
    bboxes = ibs._sync_get_annot_endpoint('/api/annot/bbox/', aid_list)
    thetas = ibs._sync_get_annot_endpoint('/api/annot/theta/', aid_list)
    name_texts = ibs._sync_get_annot_endpoint('/api/annot/name/text/', aid_list)
    name_uuids = ibs._sync_get_annot_endpoint('/api/annot/name/uuid/', aid_list)
    images = ibs._sync_get_annot_endpoint('/api/annot/image/rowid/', aid_list)
    gpaths = [ibs._construct_route_url_ibs('/api/image/src/%s/' % gid) for gid in images]
    specieses = [species_name] * len(aid_list)

    gid_list = ibs.add_images(gpaths)
    nid_list = ibs.add_names(name_texts, name_uuids)

    local_aids += ibs.add_annots(
        gid_list,
        bbox_list=bboxes,
        theta_list=thetas,
        species_list=specieses,
        nid_list=nid_list,
        annot_uuid_list=ann_uuids,
        viewpoint_list=viewpoints,
    )

    return local_aids


@register_ibs_method
def _sync_get_names(ibs, aid_list):
    return ibs._sync_get_annot_endpoint('/api/annot/name/rowid/', aid_list)


@register_ibs_method
def _sync_get_annot_endpoint(ibs, endpoint, aid_list):
    route_url = _construct_route_url(endpoint)
    print('\tGetting info on %d aids from %s' % (len(aid_list), route_url,))
    data_dict = {
        'aid_list': json.dumps(aid_list),
    }
    return _get(endpoint, data=data_dict)


@register_ibs_method
def _sync_get_aids_for_uuids(ibs, uuids):
    data_dict = {
        'uuid_list': ut.to_json(list(uuids)),
    }
    return ibs._get_ibs('/api/annot/rowid/uuid/', data=data_dict)


@register_ibs_method
def _sync_filter_only_multiple_sightings(ibs, aid_list):
    r"""
    Returns:
        filtered_aids (list): the subset of aid_list such that every annot
        has a name and each name appears at least 2x.
    """
    name_list = ibs._sync_get_names(aid_list)
    name_hist = ut.dict_hist(name_list)
    aid_names = zip(aid_list, name_list)
    filtered_aids = [aid for (aid, name) in aid_names if name_hist[name] > 1]
    filtered_aid_names = [name for (aid, name) in aid_names if name_hist[name] > 1]
    return filtered_aids, filtered_aid_names


@register_ibs_method
def _sync_get_training_aids(ibs, species_name, limit=1000):
    aid_list = ibs._sync_get_species_aids(species_name)
    aid_list, name_list = ibs._sync_filter_only_multiple_sightings(aid_list)
    # if limit is not None:
    #     aid_list = ibs._sync_get_training_subset(aid_list, name_list, limit)
    return aid_list[:limit]


# @register_ibs_method
# def _sync_get_training_subset(ibs, aid_list, name_list, limit):
#     min_aid_per_name = 2
#     max_aid_per_name = 5
#     name_to_aids = {}
#     pass


@register_ibs_method
@register_api('/api/sync/', methods=['GET'])
def detect_remote_sync_images(ibs, gid_list=None, only_sync_missing_images=True):
    _assert_remote_online(ibs)

    if gid_list is None:
        gid_list = ibs.get_valid_gids()

    confirm_list = [ut.random_nonce()[:5] for _ in range(3)]
    confirm_str = '-'.join(confirm_list)
    print(
        'You are about to submit %d images to a remote DETECT database at %r with UUID=%r.'
        % (len(gid_list), REMOTE_URL, REMOTE_UUID,)
    )
    print(
        'Only do this action if you are confident in the detection accuracy of the images, annotations, annotation metadata, parts and part metadata.'
    )
    print(
        'In order to continue, please type exactly the confirmation string %r'
        % (confirm_str,)
    )

    if six.PY2:
        input_func = raw_input  # NOQA
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
        print('Need to push %d images...' % (num_missing,))
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
        print('Need to push %d annots...' % (num_missing,))
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
        print('Need to push %d parts...' % (num_missing,))
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
        python -m wbia.web.app
        python -m wbia.web.app --allexamples
        python -m wbia.web.app --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
