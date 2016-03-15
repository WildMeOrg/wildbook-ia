# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from os.path import splitext, basename
import uuid
import six
from ibeis.control import controller_inject
import utool as ut


register_api   = controller_inject.get_ibeis_flask_api(__name__)


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
                        'auth_domain'     : None,  # Uses localhost
                        'auth_access_id'  : None,  # Uses system default
                        'auth_secret_key' : None,  # Uses system default
                    },
                    {
                        'bucket' : 'example-bucket-1',
                        'key'    : 'example/asset2.jpg',
                        # if unspecified, auth uses localhost and system defaults
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


@register_api('/api/image/json/', methods=['DELETE'])
def delete_images_json(ibs, image_uuid_list):
    """
    REST:
        Method: POST
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
        Method: POST
        URL: /api/annot/json/

    Args:
        annot_uuid_list (list of str) : list of annot UUIDs to be delete from IBEIS
    """
    aid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    ibs.delete_annots(aid_list)
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
