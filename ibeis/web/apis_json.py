# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from os.path import splitext, basename
import uuid
import six
from ibeis.web.routes_ajax import image_src
from ibeis.control import controller_inject
import utool as ut


register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_api('/api/imageset/json/', methods=['POST'])
def add_imagesets_json(ibs, imageset_text_list, imageset_uuid_list=None, config_rowid_list=None,
                       imageset_notes_list=None):
    r"""
    Adds a list of imagesets.

    Args:
        imagesettext_list (list):
        imageset_uuid_list (list):
        config_rowid_list (list):
        notes_list (list):

    Returns:
        imageset_uuid_list (list): added imageset uuids

    RESTful:
        Method: POST
        URL:    /api/imageset/json/
    """
    imageset_rowid_list = ibs.add_imagesets_json(imageset_text_list,
                                                 imageset_uuid_list=imageset_uuid_list,
                                                 config_rowid_list=config_rowid_list,
                                                 notes_list=imageset_notes_list)
    imageset_uuid_list = ibs.get_imageset_uuid(imageset_rowid_list)
    return imageset_uuid_list


@register_api('/api/image/json/', methods=['POST'])
def add_images_json(ibs, image_uri_list, image_uuid_list, image_width_list,
                    image_height_list, image_orig_name_list=None, image_ext_list=None,
                    image_time_posix_list=None, image_gps_lat_list=None,
                    image_gps_lon_list=None, image_orientation_list=None,
                    image_notes_list=None, **kwargs):
    """
    REST:
        Method: POST
        URL: /api/image/json/

    Ignore:
        sudo pip install boto

    Args:
        image_uri_list (list) : list of string image uris, most likely HTTP(S) or S3
            encoded URLs.  Alternatively, this can be a list of dictionaries (JSON
            objects) that specify AWS S3 stored assets.  An example below:

                image_uri_list = [
                    'http://domain.com/example/asset1.png',
                    '/home/example/Desktop/example/asset2.jpg',
                    's3://s3.amazon.com/example-bucket-2/asset1-in-bucket-2.tif',
                    {
                        'bucket'          : 'example-bucket-1',
                        'key'             : 'example/asset1.png',
                        'auth_domain'     : None,  # Uses 127.0.0.1
                        'auth_access_id'  : None,  # Uses system default
                        'auth_secret_key' : None,  # Uses system default
                    },
                    {
                        'bucket' : 'example-bucket-1',
                        'key'    : 'example/asset2.jpg',
                        # if unspecified, auth uses 127.0.0.1 and system defaults
                    },
                    {
                        'bucket'          : 'example-bucket-2',
                        'key'             : 'example/asset1-in-bucket-2.tif',
                        'auth_domain'     : 's3.amazon.com',
                        'auth_access_id'  : '____________________',
                        'auth_secret_key' : '________________________________________',
                    },
                ]

            Note that you cannot specify AWS authentication access ids or secret keys
            using string uri's.  For specific authentication methods, please use the
            latter list of dictionaries.

        image_uuid_list (list of str) : list of image UUIDs to be used in IBEIS IA
        image_width_list (list of int) : list of image widths
        image_height_list (list of int) : list of image heights
        image_orig_name_list (list of str): list of original image names
        image_ext_list (list of str): list of original image names
        image_time_posix_list (list of int): list of image's POSIX timestamps
        image_gps_lat_list (list of float): list of image's GPS latitude values
        image_gps_lon_list (list of float): list of image's GPS longitude values
        image_orientation_list (list of int): list of image's orientation flags
        image_notes_list (list of str) : optional list of any related notes with
            the images
        **kwargs : key-value pairs passed to the ibs.add_images() function.

    CommandLine:
        python -m ibeis.web.app --test-add_images_json

    Example:
        >>> # WEB_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> web_instance = ibeis.opendb(db='testdb1')
        >>> _payload = {
        >>>     'image_uri_list': [
        >>>         'https://upload.wikimedia.org/wikipedia/commons/4/49/Zebra_running_Ngorongoro.jpg',
        >>>         {
        >>>             'bucket'          : 'test-asset-store',
        >>>             'key'             : 'caribwhale/20130903-JAC-0002.JPG',
        >>>         },
        >>>     ],
        >>>     'image_uuid_list': [
        >>>         uuid.UUID('7fea8101-7dec-44e3-bf5d-b8287fd231e2'),
        >>>         uuid.UUID('c081119a-e08e-4863-a710-3210171d27d6'),
        >>>     ],
        >>>     'image_width_list': [
        >>>         1992,
        >>>         1194,
        >>>     ],
        >>>     'image_height_list': [
        >>>         1328,
        >>>         401,
        >>>     ],
        >>> }
        >>> gid_list = ibeis.web.app.add_images_json(web_instance, **_payload)
        >>> print(gid_list)
        >>> print(web_instance.get_image_uuids(gid_list))
        >>> print(web_instance.get_image_uris(gid_list))
        >>> print(web_instance.get_image_paths(gid_list))
        >>> print(web_instance.get_image_uris_original(gid_list))
    """
    def _get_standard_ext(gpath):
        ext = splitext(gpath)[1].lower()
        return '.jpg' if ext == '.jpeg' else ext

    def _parse_imageinfo(index):
        def _resolve_uri():
            list_ = image_uri_list
            if list_ is None or index >= len(list_) or list_[index] is None:
                raise ValueError('Must specify all required fields')
            value = list_[index]
            if isinstance(value, dict):
                value = ut.s3_dict_encode_to_str(value)
            return value

        def _resolve(list_, default='', assert_=False):
            if list_ is None or index >= len(list_) or list_[index] is None:
                if assert_:
                    raise ValueError('Must specify all required fields')
                return default
            return list_[index]

        uri = _resolve_uri()
        orig_gname = basename(uri)
        ext = _get_standard_ext(uri)

        uuid_ = _resolve(image_uuid_list, assert_=True)
        if isinstance(uuid_, six.string_types):
            uuid_ = uuid.UUID(uuid_)

        param_tup = (
            uuid_,
            uri,
            uri,
            _resolve(image_orig_name_list, default=orig_gname),
            _resolve(image_ext_list, default=ext),
            int(_resolve(image_width_list, assert_=True)),
            int(_resolve(image_height_list, assert_=True)),
            int(_resolve(image_time_posix_list, default=-1)),
            float(_resolve(image_gps_lat_list, default=-1.0)),
            float(_resolve(image_gps_lon_list, default=-1.0)),
            int(_resolve(image_orientation_list, default=0)),
            _resolve(image_notes_list),
        )
        return param_tup

    # TODO: FIX ME SO THAT WE DON'T HAVE TO LOCALIZE EVERYTHING
    kwargs['auto_localize'] = kwargs.get('auto_localize', True)
    kwargs['sanitize'] = kwargs.get('sanitize', False)

    index_list = range(len(image_uri_list))
    params_gen = ut.generate(_parse_imageinfo, index_list, adjust=True,
                             force_serial=True, **kwargs)
    params_gen = list(params_gen)
    gpath_list = [ _[0] for _ in params_gen ]
    gid_list = ibs.add_images(gpath_list, params_list=params_gen, **kwargs)  # NOQA
    # return gid_list
    image_uuid_list = ibs.get_image_uuids(gid_list)
    return image_uuid_list


@register_api('/api/annot/json/', methods=['POST'])
def add_annots_json(ibs, image_uuid_list, annot_uuid_list, annot_bbox_list,
                    annot_theta_list=None, annot_species_list=None,
                    annot_name_list=None, annot_notes_list=None, **kwargs):
    """
    REST:
        Method: POST
        URL: /api/annot/json/

    Ignore:
        sudo pip install boto

    Args:
        image_uuid_list (list of str) : list of image UUIDs to be used in IBEIS IA
        annot_uuid_list (list of str) : list of annotations UUIDs to be used in IBEIS IA
        annot_bbox_list (list of 4-tuple) : list of bounding box coordinates encoded as
            a 4-tuple of the values (xtl, ytl, width, height) where xtl is the
            'top left corner, x value' and ytl is the 'top left corner, y value'.
        annot_theta_list (list of float) : list of radian rotation around center.
            Defaults to 0.0 (no rotation).
        annot_species_list (list of str) : list of species for the annotation, if known.
            If the list is partially known, use None (null in JSON) for unknown entries.
        annot_name_list (list of str) : list of names for the annotation, if known.
            If the list is partially known, use None (null in JSON) for unknown entries.
        annot_notes_list (list of str) : list of notes to be added to the annotation.
        **kwargs : key-value pairs passed to the ibs.add_annots() function.

    CommandLine:
        python -m ibeis.web.app --test-add_annots_json

    Example:
        >>> import ibeis
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> web_instance = ibeis.opendb(db='testdb1')
        >>> _payload = {
        >>>     'image_uuid_list': [
        >>>         uuid.UUID('7fea8101-7dec-44e3-bf5d-b8287fd231e2'),
        >>>         uuid.UUID('c081119a-e08e-4863-a710-3210171d27d6'),
        >>>     ],
        >>>     'annot_uuid_list': [
        >>>         uuid.UUID('fe1547c5-1425-4757-9b8f-b2b4a47f552d'),
        >>>         uuid.UUID('86d3959f-7167-4822-b99f-42d453a50745'),
        >>>     ],
        >>>     'annot_bbox_list': [
        >>>         [0, 0, 1992, 1328],
        >>>         [0, 0, 1194, 401],
        >>>     ],
        >>> }
        >>> aid_list = ibeis.web.app.add_annots_json(web_instance, **_payload)
        >>> print(aid_list)
        >>> print(web_instance.get_annot_image_uuids(aid_list))
        >>> print(web_instance.get_annot_uuids(aid_list))
        >>> print(web_instance.get_annot_bboxes(aid_list))
    """

    image_uuid_list = [
        uuid.UUID(uuid_) if isinstance(uuid_, six.string_types) else uuid_
        for uuid_ in image_uuid_list
    ]
    annot_uuid_list = [
        uuid.UUID(uuid_) if isinstance(uuid_, six.string_types) else uuid_
        for uuid_ in annot_uuid_list
    ]
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    aid_list = ibs.add_annots(gid_list, annot_uuid_list=annot_uuid_list,  # NOQA
                              bbox_list=annot_bbox_list, theta_list=annot_theta_list,
                              species_list=annot_species_list, name_list=annot_name_list,
                              notes_list=annot_notes_list, **kwargs)
    # return aid_list
    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    return annot_uuid_list


@register_api('/api/name/json/', methods=['POST'])
def add_names_json(ibs, name_text_list, name_uuid_list=None, name_note_list=None):
    nid_list = ibs.add_names_json(name_text_list,
                                  name_uuid_list=name_uuid_list,
                                  name_note_list=name_note_list)
    return ibs.get_name_uuids(nid_list)


@register_api('/api/species/json/', methods=['POST'], __api_plural_check__=False)
def add_species_json(ibs, species_nice_list, species_text_list=None,
                     species_code_list=None, species_uuid_list=None,
                     species_note_list=None, skip_cleaning=False):
    species_rowid_list = ibs.add_species(species_nice_list,
                                         species_text_list=species_text_list,
                                         species_code_list=species_code_list,
                                         species_uuid_list=species_uuid_list,
                                         species_note_list=species_note_list,
                                         skip_cleaning=skip_cleaning)
    return ibs.get_species_uuids(species_rowid_list)


@register_api('/api/imageset/json/', methods=['GET'])
def get_valid_imageset_uuids_json(ibs, **kwargs):
    imgsetid_list = ibs.get_valid_imgsetids(**kwargs)
    return ibs.get_imageset_uuid(imgsetid_list)


@register_api('/api/imageset/annot/uuid/json/', methods=['GET'])
def get_imageset_annot_uuids_json(ibs, imageset_uuid_list):
    imgsetid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    aids_list = ibs.get_imageset_aids(imgsetid_list)
    annot_uuids_list = [
        ibs.get_annot_uuids(aid_list)
        for aid_list in aids_list
    ]
    return annot_uuids_list


@register_api('/api/imageset/occurrence/json/', methods=['GET'])
def get_imageset_isoccurrence_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_isoccurrence(imageset_rowid_list)


@register_api('/api/imageset/num/annot/reviewed/json/', methods=['GET'])
def get_imageset_num_annots_reviewed_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_num_annots_reviewed(imageset_rowid_list)


@register_api('/api/imageset/num/image/reviewed/json/', methods=['GET'])
def get_imageset_num_imgs_reviewed_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_num_imgs_reviewed(imageset_rowid_list)


@register_api('/api/imageset/num/name/exemplar/json/', methods=['GET'])
def get_imageset_num_names_with_exemplar_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_num_names_with_exemplar(imageset_rowid_list)


@register_api('/api/imageset/num/image/json/', methods=['GET'])
def get_imageset_num_gids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_num_gids(imageset_rowid_list)


@register_api('/api/imageset/num/annot/json/', methods=['GET'])
def get_imageset_num_aids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_num_aids(imageset_rowid_list)


@register_api('/api/imageset/annot/rowid/json/', methods=['GET'])
def get_imageset_aids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_aids(imageset_rowid_list)


@register_api('/api/imageset/image/rowid/json/', methods=['GET'])
def get_imageset_gids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_gids(imageset_rowid_list)


@register_api('/api/imageset/image/uuid/json/', methods=['GET'])
def get_imageset_image_uuids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_image_uuids(imageset_rowid_list)


@register_api('/api/imageset/name/rowid/json/', methods=['GET'])
def get_imageset_nids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_nids(imageset_rowid_list)


@register_api('/api/imageset/name/uuid/json/', methods=['GET'])
def get_imageset_name_uuids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_name_uuids(imageset_rowid_list)


@register_api('/api/imageset/text/json/', methods=['GET'])
def get_imageset_text_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_text(imageset_rowid_list)


@register_api('/api/imageset/rowid/uuid/json/', methods=['GET'])
def get_imageset_imgsetids_from_uuid_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_imgsetids_from_uuid(imageset_rowid_list)


@register_api('/api/imageset/rowid/text/json/', methods=['GET'])
def get_imageset_imgsetids_from_text_json(ibs, imageset_uuid_list, **kwargs):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_imgsetids_from_text(imageset_rowid_list, **kwargs)


@register_api('/api/imageset/note/json/', methods=['GET'])
def get_imageset_note_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_note(imageset_rowid_list)


@register_api('/api/imageset/time/posix/end/json/', methods=['GET'])
def get_imageset_end_time_posix_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_end_time_posix(imageset_rowid_list)


@register_api('/api/imageset/gps/lat/json/', methods=['GET'], __api_plural_check__=False)
def get_imageset_gps_lats_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_gps_lats(imageset_rowid_list)


@register_api('/api/imageset/gps/lon/json/', methods=['GET'], __api_plural_check__=False)
def get_imageset_gps_lons_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_gps_lons(imageset_rowid_list)


@register_api('/api/imageset/processed/json/', methods=['GET'])
def get_imageset_processed_flags_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_processed_flags(imageset_rowid_list)


@register_api('/api/imageset/shipped/json/', methods=['GET'])
def get_imageset_shipped_flags_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_shipped_flags(imageset_rowid_list)


@register_api('/api/imageset/time/posix/start/json/', methods=['GET'])
def get_imageset_start_time_posix_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_start_time_posix(imageset_rowid_list)


@register_api('/api/imageset/duration/json/', methods=['GET'])
def get_imageset_duration_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_duration(imageset_rowid_list)


@register_api('/api/imageset/smart/waypoint/json/', methods=['GET'])
def get_imageset_smart_waypoint_ids_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_smart_waypoint_ids(imageset_rowid_list)


@register_api('/api/imageset/smart/xml/file/name/json/', methods=['GET'])
def get_imageset_smart_xml_fnames_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_smart_xml_fnames(imageset_rowid_list)


@register_api('/api/imageset/smart/xml/file/content/json/', methods=['GET'])
def get_imageset_smart_xml_contents_json(ibs, imageset_uuid_list):
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    return ibs.get_imageset_smart_xml_contents(imageset_rowid_list)


@register_api('/api/image/json/<uuid>/', methods=['GET'])
def image_base64_api_json(ibs, uuid=None, thumbnail=False, fresh=False, **kwargs):
    rowid = ibs.get_image_rowid_from_uuid(uuid)
    return image_src(rowid, thumbnail=thumbnail, fresh=fresh, **kwargs)


@register_api('/api/image/json/', methods=['GET'])
def get_valid_image_uuids_json(ibs, **kwargs):
    gid_list = ibs.get_valid_gids(**kwargs)
    return ibs.get_image_uuids(gid_list)


@register_api('/api/image/dict/json/', methods=['GET'])
def get_image_uuids_with_annot_uuids(ibs, gid_list=None):
    if gid_list is None:
        gid_list = sorted(ibs.get_valid_gids())
    aids_list = ibs.get_image_aids(gid_list)
    zipped = zip(gid_list, aids_list)
    combined_dict = {
        str(ibs.get_image_uiids(gid)) : ibs.get_annot_uuids(aid_list)
        for gid, aid_list in zipped
    }
    return combined_dict


@register_api('/api/image/rowid/uuid/json/', methods=['GET'])
def get_image_gids_from_uuid_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_gids_from_uuid(gid_list)


@register_api('/api/image/uri/json/', methods=['GET'])
def get_image_uris_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_uris(gid_list)


@register_api('/api/image/uri/original/json/', methods=['GET'])
def get_image_uris_original_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_uris_original(gid_list)


@register_api('/api/image/file/path/json/', methods=['GET'])
def get_image_paths_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_paths(gid_list)


@register_api('/api/image/file/name/json/', methods=['GET'])
def get_image_gnames_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_gnames(gid_list)


@register_api('/api/image/size/json/', methods=['GET'])
def get_image_sizes_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_sizes(gid_list)


@register_api('/api/image/width/json/', methods=['GET'])
def get_image_widths_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_widths(gid_list)


@register_api('/api/image/height/json/', methods=['GET'])
def get_image_heights_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_heights(gid_list)


@register_api('/api/image/gps/json/', methods=['GET'], __api_plural_check__=False)
def get_image_gps_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_gps(gid_list)


@register_api('/api/image/lat/json/', methods=['GET'])
def get_image_lat_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_lat(gid_list)


@register_api('/api/image/lon/json/', methods=['GET'])
def get_image_lon_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_lon(gid_list)


@register_api('/api/image/orientation/json/', methods=['GET'])
def get_image_orientation_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_orientation(gid_list)


@register_api('/api/image/orientation/str/json/', methods=['GET'])
def get_image_orientation_str_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_orientation_str(gid_list)


@register_api('/api/image/reviewed/json/', methods=['GET'])
def get_image_reviewed_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_reviewed(gid_list)


@register_api('/api/image/detect/confidence/json/', methods=['GET'])
def get_image_detect_confidence_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_detect_confidence(gid_list)


@register_api('/api/image/note/json/', methods=['GET'])
def get_image_notes_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_notes(gid_list)


@register_api('/api/image/name/rowid/json/', methods=['GET'])
def get_image_nids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_nids(gid_list)


@register_api('/api/image/name/uuid/json/', methods=['GET'])
def get_image_name_uuids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_name_uuids(gid_list)


@register_api('/api/image/species/rowid/json/', methods=['GET'], __api_plural_check__=False)
def get_image_species_rowids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_species_rowids(gid_list)


@register_api('/api/image/species/uuid/json/', methods=['GET'], __api_plural_check__=False)
def get_image_species_uuids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_species_uuids(gid_list)


@register_api('/api/image/imageset/rowid/json/', methods=['GET'])
def get_image_imgsetids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_imgsetids(gid_list)


@register_api('/api/image/imageset/uuid/json/', methods=['GET'])
def get_image_imgset_uuids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_imgset_uuids(gid_list)


@register_api('/api/image/imageset/text/json/', methods=['GET'])
def get_image_imagesettext_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_imagesettext(gid_list)


@register_api('/api/image/annot/rowid/json/', methods=['GET'])
def get_image_aids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_aids(gid_list)


@register_api('/api/image/annot/uuid/json/', methods=['GET'])
def get_image_annot_uuids_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_annot_uuids(gid_list)


@register_api('/api/image/annot/rowid/species/json/', methods=['GET'], __api_plural_check__=False)
def get_image_aids_of_species_json(ibs, image_uuid_list, **kwargs):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_aids_of_species(gid_list, **kwargs)


@register_api('/api/image/annot/uuid/species/json/', methods=['GET'], __api_plural_check__=False)
def get_image_annot_uuids_of_species_json(ibs, image_uuid_list, **kwargs):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_annot_uuids_of_species(gid_list, **kwargs)


@register_api('/api/image/num/annot/json/', methods=['GET'])
def get_image_num_annotations_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_num_annotations(gid_list)


@register_api('/api/image/unixtime/json/', methods=['GET'])
def get_image_unixtimes_json(ibs, image_uuid_list):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_unixtime(gid_list)


@register_api('/api/image/timedelta/posix/json/', methods=['GET'])
def get_image_timedelta_posix_json(ibs, image_uuid_list, **kwargs):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_timedelta_posix(gid_list, **kwargs)


@register_api('/api/image/location/code/json/', methods=['GET'])
def get_image_location_codes_json(ibs, image_uuid_list, **kwargs):
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    return ibs.get_image_location_codes(gid_list, **kwargs)


@register_api('/api/annot/json/', methods=['GET'])
def get_valid_annot_uuids_json(ibs, **kwargs):
    aid_list = ibs.get_valid_aids(**kwargs)
    return ibs.get_annot_uuids(aid_list)


@register_api('/api/annot/json/<uuid>/', methods=['GET'])
def annotation_src_api_json(ibs, uuid=None):
    aid = ibs.get_annot_aids_from_uuid(uuid)
    return ibs.annotation_src_api(aid)


@register_api('/api/annot/rowid/uuid/json/', methods=['GET'])
def get_annot_aids_from_uuid_json(ibs, annot_uuid_list):
    return ibs.get_annot_aids_from_uuid(annot_uuid_list)


@register_api('/api/annot/image/rowid/json/', methods=['GET'])
def get_annot_gids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_gids(aid_list)


@register_api('/api/annot/uuid/hashid/json/', methods=['GET'])
def get_annot_hashid_uuid_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_hashid_uuid(aid_list, **kwargs)


@register_api('/api/annot/exemplar/json/', methods=['POST'])
def set_exemplars_from_quality_and_viewpoint_json(ibs, annot_uuid_list,
                                                  annot_name_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    if annot_name_list is not None:
        # Set names for query annotations
        nid_list = ibs.add_names(annot_name_list)
        ibs.set_annot_name_rowids(aid_list, nid_list)
    new_flag_list = ibs.set_exemplars_from_quality_and_viewpoint(aid_list, **kwargs)
    new_annot_uuid_list = ibs.get_annot_uuids(aid_list)
    return new_annot_uuid_list, new_flag_list


@register_api('/api/annot/bbox/json/', methods=['GET'])
def get_annot_bboxes_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_bboxes(aid_list)


@register_api('/api/annot/detect/confidence/json/', methods=['GET'])
def get_annot_detect_confidence_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_detect_confidence(aid_list)


@register_api('/api/annot/exemplar/json/', methods=['GET'])
def get_annot_exemplar_flags_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_exemplar_flags(aid_list)


@register_api('/api/annot/theta/json/', methods=['GET'])
def get_annot_thetas_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_thetas(aid_list)


@register_api('/api/annot/vert/json/', methods=['GET'])
def get_annot_verts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_verts(aid_list)


@register_api('/api/annot/vert/rotated/json/', methods=['GET'])
def get_annot_rotated_verts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_rotated_verts(aid_list)


@register_api('/api/annot/yaw/json/', methods=['GET'])
def get_annot_yaws_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_yaws(aid_list)


@register_api('/api/annot/num/vert/json/', methods=['GET'])
def get_annot_num_verts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_num_verts(aid_list)


@register_api('/api/annot/name/rowid/json/', methods=['GET'])
def get_annot_nids_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_nids(aid_list, **kwargs)


@register_api('/api/annot/name/uuid/json/', methods=['GET'])
def get_annot_name_rowids_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    nid_list = ibs.get_annot_name_rowids(aid_list, **kwargs)
    return ibs.get_name_uuids(nid_list)


@register_api('/api/annot/name/text/json/', methods=['GET'])
def get_annot_name_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_name_texts(aid_list)


@register_api('/api/annot/note/json/', methods=['GET'])
def get_annot_notes_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_notes(aid_list)


@register_api('/api/annot/species/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_species_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_species(aid_list)


@register_api('/api/annot/species/rowid/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_species_rowids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_species_rowids(aid_list)


@register_api('/api/annot/species/uuid/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_species_uuids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_species_uuids(aid_list)


@register_api('/api/annot/species/text/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_species_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_species_texts(aid_list)


@register_api('/api/annot/imageset/rowid/json/', methods=['GET'])
def get_annot_imgsetids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_imgsetids(aid_list)


@register_api('/api/annot/imageset/uuid/json/', methods=['GET'])
def get_annot_imgset_uuids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_imgset_uuids(aid_list)


@register_api('/api/annot/imageset/text/json/', methods=['GET'])
def get_annot_image_set_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_image_set_texts(aid_list)


@register_api('/api/annot/image/name/json/', methods=['GET'])
def get_annot_image_names_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_image_names(aid_list)


@register_api('/api/annot/image/unixtime/json/', methods=['GET'])
def get_annot_image_unixtimes_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_image_unixtimes(aid_list)


@register_api('/api/annot/image/gps/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_image_gps_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_image_gps(aid_list)


@register_api('/api/annot/image/file/path/json/', methods=['GET'])
def get_annot_image_paths_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_image_paths(aid_list)


@register_api('/api/annot/image/uuid/json/', methods=['GET'])
def get_annot_image_uuids_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    image_uuid_list = [
        None if aid is None else ibs.get_annot_image_uuids(aid)
        for aid in aid_list
    ]
    return image_uuid_list


@register_api('/api/annot/quality/json/', methods=['GET'])
def get_annot_qualities_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_qualities(aid_list, **kwargs)


@register_api('/api/annot/quality/text/json/', methods=['GET'])
def get_annot_quality_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_quality_texts(aid_list)


@register_api('/api/annot/yaw/text/json/', methods=['GET'])
def get_annot_yaw_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_yaw_texts(aid_list)


@register_api('/api/annot/sex/json/', methods=['GET'])
def get_annot_sex_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_sex(aid_list, **kwargs)


@register_api('/api/annot/sex/text/json/', methods=['GET'])
def get_annot_sex_texts_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_sex_texts(aid_list)


@register_api('/api/annot/reviewed/json/', methods=['GET'])
def get_annot_reviewed_json(ibs, annot_uuid_list):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_reviewed(aid_list)


@register_api('/api/annot/multiple/json/', methods=['GET'])
def get_annot_multiple_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_multiple(aid_list, **kwargs)


# @register_api('/api/annot/image/contributor/tag/json/', methods=['GET'])
# def get_annot_image_contributor_tag_json(ibs, annot_uuid_list):
#     aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
#     return ibs.get_annot_image_contributor_tag(aid_list)


@register_api('/api/annot/age/months/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est(aid_list, **kwargs)


@register_api('/api/annot/age/months/text/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_texts_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est_texts(aid_list, **kwargs)


@register_api('/api/annot/age/months/min/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_min_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est_min(aid_list, **kwargs)


@register_api('/api/annot/age/months/max/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_max_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est_max(aid_list, **kwargs)


@register_api('/api/annot/age/months/min/text/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_min_texts_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est_min_texts(aid_list, **kwargs)


@register_api('/api/annot/age/months/max/text/json/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_max_texts_json(ibs, annot_uuid_list, **kwargs):
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    return ibs.get_annot_age_months_est_max_texts(aid_list, **kwargs)


@register_api('/api/name/json/', methods=['GET'])
def get_valid_name_uuids_json(ibs, **kwargs):
    nid_list = ibs.get_valid_nids(**kwargs)
    return ibs.get_name_uuids(nid_list)


@register_api('/api/name/dict/json/', methods=['GET'])
def get_name_nids_with_gids_json(ibs, nid_list=None):
    if nid_list is None:
        nid_list = sorted(ibs.get_valid_nids())
    name_list = ibs.get_name_texts(nid_list)
    gids_list = ibs.get_name_gids(nid_list)

    zipped = zip(nid_list, name_list, gids_list)
    combined_dict = {
        name : (ibs.get_name_uuids(nid), ibs.get_image_uuids(gid_list))
        for nid, name, gid_list in zipped
    }
    return combined_dict


@register_api('/api/name/annot/rowid/json/', methods=['GET'])
def get_name_aids_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_aids(nid_list, **kwargs)


@register_api('/api/name/annot/uuid/json/', methods=['GET'])
def get_name_annot_uuids_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_annot_uuids(nid_list, **kwargs)


@register_api('/api/name/annot/rowid/exemplar/json/', methods=['GET'])
def get_name_exemplar_aids_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_exemplar_aids(ibs, nid_list)


@register_api('/api/name/annot/uuid/exemplar/json/', methods=['GET'])
def get_name_exemplar_name_uuids_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_exemplar_name_uuids(nid_list, **kwargs)


@register_api('/api/name/image/rowid/json/', methods=['GET'])
def get_name_gids_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_gids(ibs, nid_list)


@register_api('/api/name/image/uuid/json/', methods=['GET'])
def get_name_image_uuids_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_image_uuids(ibs, nid_list)


@register_api('/api/name/note/json/', methods=['GET'])
def get_name_notes_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_notes(ibs, nid_list)


@register_api('/api/name/num/annot/json/', methods=['GET'])
def get_name_num_annotations_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_num_annotations(ibs, nid_list)


@register_api('/api/name/num/annot/exemplar/json/', methods=['GET'])
def get_name_num_exemplar_annotations_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_num_exemplar_annotations(ibs, nid_list)


@register_api('/api/name/temp/json/', methods=['GET'])
def get_name_temp_flag_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_temp_flag(nid_list, **kwargs)


@register_api('/api/name/alias/text/json/', methods=['GET'], __api_plural_check__=False)
def get_name_alias_texts_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_alias_texts(ibs, nid_list)


@register_api('/api/name/text/json/', methods=['GET'])
def get_name_texts_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_texts(nid_list, **kwargs)


@register_api('/api/name/rowid/text/json/', methods=['GET'])
def get_name_rowids_from_text_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_rowids_from_text(nid_list, **kwargs)


@register_api('/api/name/rowid/uuid/json/', methods=['GET'])
def get_name_rowids_from_uuid_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_rowids_from_uuid(nid_list, **kwargs)


@register_api('/api/name/sex/json/', methods=['GET'])
def get_name_sex_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_sex(nid_list, **kwargs)


@register_api('/api/name/sex/text/json/', methods=['GET'])
def get_name_sex_text_json(ibs, name_uuid_list, **kwargs):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_sex_text(nid_list, **kwargs)


@register_api('/api/name/age/months/min/json/', methods=['GET'], __api_plural_check__=False)
def get_name_age_months_est_min_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_age_months_est_min(ibs, nid_list)


@register_api('/api/name/age/months/max/json/', methods=['GET'], __api_plural_check__=False)
def get_name_age_months_est_max_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_age_months_est_max(ibs, nid_list)


@register_api('/api/name/imageset/rowid/json/', methods=['GET'])
def get_name_imgsetids_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_imgsetids(ibs, nid_list)


@register_api('/api/name/imageset/uuid/json/', methods=['GET'])
def get_name_imgset_uuids_json(ibs, name_uuid_list):
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    return ibs.get_name_imgset_uuids(ibs, nid_list)


@register_api('/api/species/json/', methods=['GET'], __api_plural_check__=False)
def _get_all_species_rowids_json(ibs, **kwargs):
    species_rowid_list = ibs._get_all_species_rowids(**kwargs)
    return ibs.get_species_uuids(species_rowid_list, **kwargs)


@register_api('/api/species/rowid/text/json/', methods=['GET'], __api_plural_check__=False)
def get_species_rowids_from_text_json(ibs, species_uuid_list, **kwargs):
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    return ibs.get_species_rowids_from_text(species_rowid_list, **kwargs)


@register_api('/api/species/rowid/uuid/json/', methods=['GET'], __api_plural_check__=False)
def get_species_rowids_from_uuids_json(ibs, species_uuid_list):
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    return ibs.get_species_rowids_from_uuids(species_rowid_list)


@register_api('/api/species/text/json/', methods=['GET'], __api_plural_check__=False)
def get_species_texts_json(ibs, species_uuid_list):
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    return ibs.get_species_texts(species_rowid_list)


@register_api('/api/species/nice/json/', methods=['GET'], __api_plural_check__=False)
def get_species_nice_json(ibs, species_uuid_list):
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    return ibs.get_species_nice(species_rowid_list)


@register_api('/api/species/code/json/', methods=['GET'], __api_plural_check__=False)
def get_species_codes_json(ibs, species_uuid_list):
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    return ibs.get_species_codes(species_rowid_list)


@register_api('/api/species/note/json/', methods=['GET'], __api_plural_check__=False)
def get_species_notes_json(ibs, species_uuid_list):
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    return ibs.get_species_notes(species_rowid_list)


@register_api('/chaos/imageset/', methods=['GET', 'POST'], __api_plural_check__=False)
def chaos_imageset(ibs):
    """
    REST:
        Method: POST
        URL: /api/image/json/

    Args:
        image_uuid_list (list of str) : list of image UUIDs to be delete from IBEIS
    """
    from random import shuffle, randint
    gid_list = ibs.get_valid_gids()
    shuffle(gid_list)
    sample = min(len(gid_list) // 2, 50)
    assert sample > 0, 'Cannot create a chaos imageset using an empty database'
    gid_list_ = gid_list[:sample]
    imagetset_name = 'RANDOM_CHAOS_TEST_IMAGESET_%08d' % (randint(0, 99999999))
    imagetset_rowid = ibs.add_imagesets(imagetset_name)
    imagetset_uuid = ibs.get_imageset_uuid(imagetset_rowid)
    ibs.add_image_relationship(gid_list_, [imagetset_rowid] * len(gid_list_))
    return imagetset_name, imagetset_uuid


@register_api('/api/imageset/json/', methods=['DELETE'])
def delete_imageset_json(ibs, imageset_uuid_list):
    """
    REST:
        Method: DELETE
        URL: /api/imageset/json/

    Args:
        imageset_uuid_list (list of str) : list of imageset UUIDs to be delete from IBEIS
    """
    imgsetid_list = ibs.get_imageset_imgsetids_from_uuid(imageset_uuid_list)
    ibs.delete_imagesets(imgsetid_list)
    return True


@register_api('/api/image/json/', methods=['DELETE'])
def delete_images_json(ibs, image_uuid_list):
    """
    REST:
        Method: DELETE
        URL: /api/image/json/

    Args:
        image_uuid_list (list of str) : list of image UUIDs to be delete from IBEIS
    """
    gid_list = ibs.get_image_gids_from_uuid(image_uuid_list)
    ibs.delete_images(gid_list)
    return True


@register_api('/api/annot/json/', methods=['DELETE'])
def delete_annots_json(ibs, annot_uuid_list):
    """
    REST:
        Method: DELETE
        URL: /api/annot/json/

    Args:
        annot_uuid_list (list of str) : list of annot UUIDs to be delete from IBEIS
    """
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    ibs.delete_annots(aid_list)
    return True


@register_api('/api/species/json/', methods=['DELETE'], __api_plural_check__=False)
def delete_species_json(ibs, species_uuid_list):
    """
    REST:
        Method: DELETE
        URL: /api/species/json/

    Args:
        species_uuid_list (list of str) : list of species UUIDs to be delete from IBEIS
    """
    species_rowid_list = ibs.get_species_rowids_from_uuids(species_uuid_list)
    ibs.delete_species(species_rowid_list)
    return True


@register_api('/api/name/json/', methods=['DELETE'])
def delete_name_json(ibs, name_uuid_list):
    """
    REST:
        Method: DELETE
        URL: /api/name/json/

    Args:
        name_uuid_list (list of str) : list of name UUIDs to be delete from IBEIS
    """
    nid_list = ibs.get_name_rowids_from_uuid(name_uuid_list)
    ibs.delete_names(nid_list)
    return True


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
