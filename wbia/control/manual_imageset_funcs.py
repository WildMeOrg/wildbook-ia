# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import six
from wbia import constants as const
from wbia.control import accessor_decors, controller_inject
from wbia.control.controller_inject import make_ibs_register_decorator
import functools
import utool as ut
import uuid

print, rrr, profile = ut.inject2(__name__)


IMAGESET_OCCURRENCE_FLAG = 'imageset_occurrence_flag'
IMAGESET_END_TIME_POSIX = 'imageset_end_time_posix'
IMAGESET_GPS_LAT = 'imageset_gps_lat'
IMAGESET_GPS_LON = 'imageset_gps_lon'
IMAGESET_NOTE = 'imageset_note'
IMAGESET_PROCESSED_FLAG = 'imageset_processed_flag'
IMAGESET_ROWID = 'imageset_rowid'
IMAGESET_SHIPPED_FLAG = 'imageset_shipped_flag'
IMAGESET_START_TIME_POSIX = 'imageset_start_time_posix'
IMAGESET_SMART_WAYPOINT_ID = 'imageset_smart_waypoint_id'
IMAGESET_SMART_XML_FNAME = 'imageset_smart_xml_fname'


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api = controller_inject.get_wbia_flask_api(__name__)


@register_ibs_method
@accessor_decors.ider
def _get_all_imageset_rowids(ibs):
    r"""
    Returns:
        list_ (list):  all unfiltered imgsetids (imageset rowids)
    """
    all_imgsetids = ibs.db.get_all_rowids(const.IMAGESET_TABLE)
    return all_imgsetids


@register_ibs_method
@accessor_decors.ider
def _get_all_imgsetids(ibs):
    r"""
    alias
    """
    return _get_all_imageset_rowids(ibs)


@register_ibs_method
@accessor_decors.ider
@register_api('/api/imageset/', methods=['GET'])
def get_valid_imgsetids(
    ibs,
    min_num_gids=0,
    processed=None,
    shipped=None,
    is_occurrence=None,
    is_special=None,
):
    r"""
    FIX NAME imgagesetids

    Returns:
        list_ (list):  list of all imageset ids

    RESTful:
        Method: GET
        URL:    /api/imageset/
    """
    imgsetid_list = ibs._get_all_imgsetids()
    if min_num_gids > 0:
        num_gids_list = ibs.get_imageset_num_gids(imgsetid_list)
        flag_list = [num_gids >= min_num_gids for num_gids in num_gids_list]
        imgsetid_list = ut.compress(imgsetid_list, flag_list)
    if processed is not None:
        flag_list = ibs.get_imageset_processed_flags(imgsetid_list)
        isvalid_list = [flag == 1 if processed else flag == 0 for flag in flag_list]
        imgsetid_list = ut.compress(imgsetid_list, isvalid_list)
    if shipped is not None:
        flag_list = ibs.get_imageset_shipped_flags(imgsetid_list)
        isvalid_list = [flag == 1 if shipped else flag == 0 for flag in flag_list]
        imgsetid_list = ut.compress(imgsetid_list, isvalid_list)
    if is_occurrence is not None:
        flag_list = ibs.get_imageset_occurrence_flags(imgsetid_list)
        isvalid_list = [flag == is_occurrence for flag in flag_list]
        imgsetid_list = ut.compress(imgsetid_list, isvalid_list)
    if is_special is not None:
        flag_list = ibs.is_special_imageset(imgsetid_list)
        isvalid_list = [flag == is_special for flag in flag_list]
        imgsetid_list = ut.compress(imgsetid_list, isvalid_list)
    return imgsetid_list


@register_ibs_method
def is_special_imageset(ibs, imgsetid_list):
    imagesettext_list = ibs.get_imageset_text(imgsetid_list)
    isspecial_list = [
        str(imagesettext) in set(const.SPECIAL_IMAGESET_LABELS)
        for imagesettext in imagesettext_list
    ]
    return isspecial_list


@register_ibs_method
@accessor_decors.adder
@register_api('/api/imageset/', methods=['POST'])
def add_imagesets(
    ibs,
    imagesettext_list,
    imageset_uuid_list=None,
    notes_list=None,
    occurence_flag_list=None,
):
    r"""
    Adds a list of imagesets.

    Args:
        imagesettext_list (list):
        imageset_uuid_list (list):
        notes_list (list):

    Returns:
        imgsetid_list (list): added imageset rowids

    RESTful:
        Method: POST
        URL:    /api/imageset/
    """
    if ut.VERBOSE:
        print('[ibs] adding %d imagesets' % len(imagesettext_list))
    # Add imageset text names to database
    if notes_list is None:
        notes_list = [''] * len(imagesettext_list)
    if imageset_uuid_list is None:
        imageset_uuid_list = [uuid.uuid4() for _ in range(len(imagesettext_list))]
    if occurence_flag_list is None:
        occurence_flag_list = [0] * len(imagesettext_list)
    colnames = [
        'imageset_text',
        'imageset_uuid',
        'imageset_occurrence_flag',
        'imageset_note',
    ]
    params_iter = zip(
        imagesettext_list, imageset_uuid_list, occurence_flag_list, notes_list
    )
    get_rowid_from_superkey = functools.partial(
        ibs.get_imageset_imgsetids_from_text, ensure=False
    )
    imgsetid_list = ibs.db.add_cleanly(
        const.IMAGESET_TABLE, colnames, params_iter, get_rowid_from_superkey
    )
    return imgsetid_list


# SETTERS::IMAGESET


@register_ibs_method
@accessor_decors.setter
@register_api('/api/imageset/text/', methods=['PUT'])
def set_imageset_text(ibs, imgsetid_list, imageset_text_list):
    r"""
    Sets names of imagesets (groups of animals)

    RESTful:
        Method: PUT
        URL:    /api/imageset/text/
    """
    # Special set checks
    if any(ibs.is_special_imageset(imgsetid_list)):
        raise ValueError('cannot rename special imagesets')
    id_iter = ((imgsetid,) for imgsetid in imgsetid_list)
    val_list = ((imageset_text,) for imageset_text in imageset_text_list)
    ibs.db.set(const.IMAGESET_TABLE, ('imageset_text',), val_list, id_iter)


#
# GETTERS::IMAGESET


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/num/image/reviewed/', methods=['GET'])
def get_imageset_num_imgs_reviewed(ibs, imgsetid_list):
    r"""
    RESTful:
        Method: GET
        URL:    /api/imageset/num/image/reviewed/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> # Reset and compute imagesets
        >>> ibs.delete_all_imagesets()
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> imgsetid_list = ibs.get_valid_imgsetids()
        >>> num_reviwed_list = ibs.get_imageset_num_imgs_reviewed(imgsetid_list)
        >>> result = num_reviwed_list
        >>> print(result)
        [0, 0]
    """
    gids_list = ibs.get_imageset_gids(imgsetid_list)
    flags_list = ibs.unflat_map(ibs.get_image_reviewed, gids_list)
    num_reviwed_list = [sum(flags) for flags in flags_list]
    return num_reviwed_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/num/annot/reviewed/', methods=['GET'])
def get_imageset_num_annots_reviewed(ibs, imgsetid_list):
    r"""
    RESTful:
        Method: GET
        URL:    /api/imageset/num/annot/reviewed/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> # Reset and compute imagesets
        >>> ibs.delete_all_imagesets()
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> imgsetid_list = ibs.get_valid_imgsetids()
        >>> num_reviwed_list = ibs.get_imageset_num_imgs_reviewed(imgsetid_list)
        >>> result = num_reviwed_list
        >>> print(result)
        [0, 0]
    """
    aids_list = ibs.get_imageset_aids(imgsetid_list)
    flags_list = ibs.unflat_map(ibs.get_annot_reviewed, aids_list)
    num_reviwed_list = [sum(flags) for flags in flags_list]
    return num_reviwed_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/imageset/num/annotmatch/reviewed/', methods=['GET'])
def get_imageset_num_annotmatch_reviewed(ibs, imgsetid_list):
    r"""
    RESTful:
        Method: GET
        URL:    /api/imageset/num/annotmatch/reviewed/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> imgsetid_list = ibs._get_all_imageset_rowids()
        >>> num_annots_reviewed_list = ibs.get_imageset_num_annotmatch_reviewed(imgsetid_list)
    """
    aids_list = ibs.get_imageset_custom_filtered_aids(imgsetid_list)
    has_revieweds_list = ibs.unflat_map(
        ibs.get_annot_has_reviewed_matching_aids, aids_list
    )
    num_annots_reviewed_list = list(map(sum, has_revieweds_list))
    return num_annots_reviewed_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/num/name/exemplar/', methods=['GET'])
def get_imageset_num_names_with_exemplar(ibs, imgsetid_list):
    r"""
    RESTful:
        Method: GET
        URL:    /api/imageset/num/name/exemplar/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> imgsetid_list = ibs._get_all_imageset_rowids()
        >>> num_annots_reviewed_list = ibs.get_imageset_num_annotmatch_reviewed(imgsetid_list)
    """
    aids_list = ibs.get_imageset_custom_filtered_aids(imgsetid_list)
    exflags_list = ibs.unflat_map(ibs.get_annot_exemplar_flags, aids_list)
    nids_list = ibs.unflat_map(ibs.get_annot_name_rowids, aids_list)
    groups_list = [
        ut.group_items(exflags, nids) for exflags, nids in zip(exflags_list, nids_list)
    ]
    # num_names_list = [len(groups) for groups in groups_list]
    num_exemplared_names_list = [
        sum([any(exflags) for exflags in six.itervalues(groups)])
        for groups in groups_list
    ]
    return num_exemplared_names_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_imageset_fraction_names_with_exemplar(ibs, imgsetid_list):
    r"""
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb2')
        >>> imgsetid_list = ibs._get_all_imageset_rowids()
        >>> fraction_exemplared_names_list = ibs.get_imageset_fraction_names_with_exemplar(imgsetid_list)
    """
    aids_list = ibs.get_imageset_custom_filtered_aids(imgsetid_list)
    # exflags_list = ibs.unflat_map(ibs.get_annot_exemplar_flags, aids_list)
    nids_list = list(
        map(list, map(set, ibs.unflat_map(ibs.get_annot_name_rowids, aids_list)))
    )
    aids_list_list = ibs.unflat_map(ibs.get_name_aids, nids_list)
    flags_list_list = list(
        map(lambda x: ibs.unflat_map(ibs.get_annot_exemplar_flags, x), aids_list_list)
    )
    # groups_list = [ut.group_items(exflags, nids)
    #               for exflags, nids in zip(exflags_list, nids_list)]
    num_names_list = list(map(len, nids_list))
    num_exemplared_names_list = [
        sum([any(exflags) for exflags in flags_list]) for flags_list in flags_list_list
    ]
    fraction_exemplared_names_list = [
        None if num_names == 0 else num_exemplared_names / num_names
        for num_exemplared_names, num_names in zip(
            num_exemplared_names_list, num_names_list
        )
    ]
    return fraction_exemplared_names_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_imageset_fraction_annotmatch_reviewed(ibs, imgsetid_list):
    aids_list = ibs.get_imageset_custom_filtered_aids(imgsetid_list)
    flags_list = ibs.unflat_map(ibs.get_annot_has_reviewed_matching_aids, aids_list)
    fraction_annotmatch_reviewed_list = [
        None if len(flags) == 0 else sum(flags) / len(flags) for flags in flags_list
    ]
    return fraction_annotmatch_reviewed_list


@register_ibs_method
# @register_api('/api/imageset/aids/filtered/custom/', methods=['GET'])
def get_imageset_custom_filtered_aids(ibs, imgsetid_list):
    r"""
    hacks to filter aids to only certain views and qualities
    """
    aids_list_ = ibs.get_imageset_aids(imgsetid_list)
    # HACK: Get percentage for the annots we currently care about
    aids_list = [(aids) for aids in aids_list_]
    return aids_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_imageset_fraction_imgs_reviewed(ibs, imgsetid_list):
    gids_list = ibs.get_imageset_gids(imgsetid_list)
    flags_list = ibs.unflat_map(ibs.get_image_reviewed, gids_list)
    fraction_imgs_reviewed_list = [
        None if len(flags) == 0 else sum(flags) / len(flags) for flags in flags_list
    ]
    return fraction_imgs_reviewed_list


def _percent_str(pcnt):
    return 'undef' if pcnt is None else '%06.2f %%' % (pcnt * 100,)


@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(
    const.IMAGESET_TABLE, 'percent_names_with_exemplar_str', debug=False
)  # HACK
def get_imageset_percent_names_with_exemplar_str(ibs, imgsetid_list):
    fraction_exemplared_names_list = ibs.get_imageset_fraction_names_with_exemplar(
        imgsetid_list
    )
    percent_exemplared_names_list_str = list(
        map(_percent_str, fraction_exemplared_names_list)
    )
    return percent_exemplared_names_list_str


@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(
    const.IMAGESET_TABLE, 'percent_imgs_reviewed_str', debug=False
)  # HACK
def get_imageset_percent_imgs_reviewed_str(ibs, imgsetid_list):
    fraction_imgs_reviewed_list = ibs.get_imageset_fraction_imgs_reviewed(imgsetid_list)
    percent_imgs_reviewed_str_list = list(map(_percent_str, fraction_imgs_reviewed_list))
    return percent_imgs_reviewed_str_list


@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(
    const.IMAGESET_TABLE, 'percent_annotmatch_reviewed_str', debug=False
)  # HACK
def get_imageset_percent_annotmatch_reviewed_str(ibs, imgsetid_list):
    fraction_annotmatch_reviewed_list = ibs.get_imageset_fraction_annotmatch_reviewed(
        imgsetid_list
    )
    percent_annotmach_reviewed_str_list = list(
        map(_percent_str, fraction_annotmatch_reviewed_list)
    )
    return percent_annotmach_reviewed_str_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/num/image/', methods=['GET'])
def get_imageset_num_gids(ibs, imgsetid_list):
    r"""
    Returns:
        nGids_list (list): number of images in each imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/num/image/
    """
    nGids_list = list(map(len, ibs.get_imageset_gids(imgsetid_list)))
    return nGids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/num/annot/', methods=['GET'])
def get_imageset_num_aids(ibs, imgsetid_list):
    r"""
    Returns:
        nGids_list (list): number of images in each imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/num/annot/
    """
    nAids_list = list(map(len, ibs.get_imageset_aids(imgsetid_list)))
    return nAids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/imageset/annot/rowid/', methods=['GET'])
def get_imageset_aids(ibs, imgsetid_list):
    r"""
    Returns:
        aids_list (list):  a list of list of aids in each imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/annot/rowid/

    Args:
        ibs (IBEISController):  wbia controller object
        imgsetid_list (list):

    Returns:
        list: aids_list

    CommandLine:
        python -m wbia.control.manual_imageset_funcs --test-get_imageset_aids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> ibs.delete_imagesets(ibs.get_valid_imgsetids())
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> imgsetid_list = ibs.get_valid_imgsetids()
        >>> aids_list = get_imageset_aids(ibs, imgsetid_list)
        >>> result = ('aids_list = %s' % (str(aids_list),))
        >>> print(result)
    """
    gids_list = ibs.get_imageset_gids(imgsetid_list)
    aids_list_ = ibs.unflat_map(ibs.get_image_aids, gids_list)
    aids_list = list(map(ut.flatten, aids_list_))
    # print('get_imageset_aids')
    # print('imgsetid_list = %r' % (imgsetid_list,))
    # print('gids_list = %r' % (gids_list,))
    # print('aids_list_ = %r' % (aids_list_,))
    # print('aids_list = %r' % (aids_list,))
    return aids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/imageset/annot/uuid/', methods=['GET'])
def get_imageset_uuids(ibs, imgsetid_list):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        imgsetid_list (list):

    Returns:
        list: annot_uuids_list

    RESTful:
        Method: GET
        URL:    /api/imageset/annot/uuid/

    CommandLine:
        python -m wbia.control.manual_imageset_funcs --test-get_imageset_aids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> ibs.delete_imagesets(ibs.get_valid_imgsetids())
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> imgsetid_list = ibs.get_valid_imgsetids()
        >>> aids_list = get_imageset_aids(ibs, imgsetid_list)
        >>> result = ('aids_list = %s' % (str(aids_list),))
        >>> print(result)
    """
    aids_list = ibs.get_imageset_aids(imgsetid_list)
    annot_uuids_list = [ibs.get_annot_uuids(aid_list) for aid_list in aids_list]
    return annot_uuids_list


@register_ibs_method
@accessor_decors.getter_1toM
@accessor_decors.cache_getter(const.IMAGESET_TABLE, 'image_rowids')
@register_api('/api/imageset/image/rowid/', methods=['GET'])
@profile
def get_imageset_gids(ibs, imgsetid_list):
    r"""
    Returns:
        gids_list (list):  a list of list of gids in each imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/image/rowid/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    NEW_INDEX_HACK = True
    if NEW_INDEX_HACK:
        # FIXME: This index should when the database is defined.
        # Ensure that an index exists on the image column of the annotation table
        ibs.db.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS gids_to_gs ON {GSG_RELATION_TABLE} (imageset_rowid);
            """.format(
                GSG_RELATION_TABLE=const.GSG_RELATION_TABLE
            )
        ).fetchall()
    gids_list = ibs.db.get(
        const.GSG_RELATION_TABLE,
        ('image_rowid',),
        imgsetid_list,
        id_colname='imageset_rowid',
        unpack_scalars=False,
    )
    # print('get_imageset_gids')
    # print('imgsetid_list = %r' % (imgsetid_list,))
    # print('gids_list = %r' % (gids_list,))
    return gids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/imageset/image/uuid/', methods=['GET'])
def get_imageset_image_uuids(ibs, imgsetid_list):
    r"""
    Returns:
        gids_list (list):  a list of list of gids in each imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/image/uuid/
    """
    gids_list = ibs.get_imageset_gids(imgsetid_list)
    image_uuid_list = [ibs.get_image_uuids(gid_list) for gid_list in gids_list]
    return image_uuid_list


@register_ibs_method
# @register_api('/api/imageset/gsgrids/', methods=['GET'])
def get_imageset_gsgrids(ibs, imgsetid_list=None, gid_list=None):
    r"""
    Returns:
        list_ (list):  a list of imageset-image-relationship rowids for each encouterid
    """
    # WEIRD FUNCTION FIXME
    assert (
        imgsetid_list is not None or gid_list is not None
    ), 'Either imgsetid_list or gid_list must be None'
    if imgsetid_list is not None and gid_list is None:
        # TODO: Group type
        params_iter = ((imgsetid,) for imgsetid in imgsetid_list)
        where_clause = 'imageset_rowid=?'
        # list of relationships for each imageset
        gsgrids_list = ibs.db.get_where(
            const.GSG_RELATION_TABLE,
            ('gsgr_rowid',),
            params_iter,
            where_clause,
            unpack_scalars=False,
        )
    elif gid_list is not None and imgsetid_list is None:
        # TODO: Group type
        params_iter = ((gid,) for gid in gid_list)
        where_clause = 'image_rowid=?'
        # list of relationships for each imageset
        gsgrids_list = ibs.db.get_where(
            const.GSG_RELATION_TABLE,
            ('gsgr_rowid',),
            params_iter,
            where_clause,
            unpack_scalars=False,
        )
    else:
        # TODO: Group type
        params_iter = ((imgsetid, gid,) for imgsetid, gid in zip(imgsetid_list, gid_list))
        where_clause = 'imageset_rowid=? AND image_rowid=?'
        # list of relationships for each imageset
        gsgrids_list = ibs.db.get_where(
            const.GSG_RELATION_TABLE,
            ('gsgr_rowid',),
            params_iter,
            where_clause,
            unpack_scalars=False,
        )
    return gsgrids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/imageset/name/rowid/', methods=['GET'])
def get_imageset_nids(ibs, imgsetid_list):
    r"""
    Returns:
        list_ (list):  a list of list of known nids in each imageset

    CommandLine:
        python -m wbia.control.manual_imageset_funcs --test-get_imageset_nids

    RESTful:
        Method: GET
        URL:    /api/imageset/name/rowid/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs.delete_imagesets(ibs.get_valid_imgsetids())
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> imgsetid_list = ibs.get_valid_imgsetids()
        >>> nids_list = ibs.get_imageset_nids(imgsetid_list)
        >>> result = nids_list
        >>> print(result)
        [[1, 2, 3], [4, 5, 6, 7]]
    """
    # FIXME: SLOW
    aids_list = ibs.get_imageset_aids(imgsetid_list)
    nids_list = ibs.unflat_map(ibs.get_annot_name_rowids, aids_list)
    # nids_list_ = [[nid[0] for nid in nids if len(nid) > 0] for nids in nids_list]
    # Remove any unknown anmes
    nids_list = [[nid for nid in nids if nid > 0] for nids in nids_list]

    nids_list = list(map(ut.unique_ordered, nids_list))
    # print('get_imageset_nids')
    # print('imgsetid_list = %r' % (imgsetid_list,))
    # print('aids_list = %r' % (aids_list,))
    # print('nids_list_ = %r' % (nids_list_,))
    # print('nids_list = %r' % (nids_list,))
    return nids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/imageset/name/uuid/', methods=['GET'])
def get_imageset_name_uuids(ibs, imgsetid_list):
    r"""
    Returns:
        name_uuid_list (list):  a list of list of known name uuids in each imageset

    CommandLine:
        python -m wbia.control.manual_imageset_funcs --test-get_imageset_name_uuids

    RESTful:
        Method: GET
        URL:    /api/imageset/name/uuid/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs.delete_imagesets(ibs.get_valid_imgsetids())
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> imgsetid_list = ibs.get_valid_imgsetids()
        >>> nids_list = ibs.get_imageset_nids(imgsetid_list)
        >>> result = nids_list
        >>> print(result)
        [[1, 2, 3], [4, 5, 6, 7]]
    """
    nids_list = ibs.get_imageset_nids(imgsetid_list)
    name_uuid_list = [ibs.get_name_uuids(nid_list) for nid_list in nids_list]
    return name_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/uuid/', methods=['GET'])
def get_imageset_uuid(ibs, imgsetid_list):
    r"""
    Returns:
        list_ (list): imageset_uuid of each imgsetid in imgsetid_list

    RESTful:
        Method: GET
        URL:    /api/imageset/uuid/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    encuuid_list = ibs.db.get(
        const.IMAGESET_TABLE,
        ('imageset_uuid',),
        imgsetid_list,
        id_colname='imageset_rowid',
    )
    return encuuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/text/', methods=['GET'])
def get_imageset_text(ibs, imgsetid_list):
    r"""
    Returns:
        list_ (list): imageset_text of each imgsetid in imgsetid_list

    RESTful:
        Method: GET
        URL:    /api/imageset/text/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    imagesettext_list = ibs.db.get(
        const.IMAGESET_TABLE,
        ('imageset_text',),
        imgsetid_list,
        id_colname='imageset_rowid',
    )
    return imagesettext_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/rowid/uuid/', methods=['GET'])
def get_imageset_imgsetids_from_uuid(ibs, uuid_list):
    r"""
    Returns:
        list_ (list): a list of imgsetids corresponding to each imageset imagesettext
    #FIXME: make new naming scheme for non-primary-key-getters
    get_imageset_imgsetids_from_text_from_text

    RESTful:
        Method: GET
        URL:    /api/imageset/rowid/uuid/
    """
    imgsetid_list = ibs.db.get(
        const.IMAGESET_TABLE, ('imageset_rowid',), uuid_list, id_colname='imageset_uuid'
    )
    return imgsetid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/rowid/text/', methods=['GET'])
def get_imageset_imgsetids_from_text(ibs, imagesettext_list, ensure=True):
    r"""
    Returns:
        list_ (list): a list of imgsetids corresponding to each imageset imagesettext
    #FIXME: make new naming scheme for non-primary-key-getters
    get_imageset_imgsetids_from_text_from_text

    RESTful:
        Method: GET
        URL:    /api/imageset/rowid/text/
    """
    if ensure:
        imgsetid_list = ibs.add_imagesets(imagesettext_list)
    else:
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        imgsetid_list = ibs.db.get(
            const.IMAGESET_TABLE,
            ('imageset_rowid',),
            imagesettext_list,
            id_colname='imageset_text',
        )
    return imgsetid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/note/', methods=['GET'])
def get_imageset_note(ibs, imgsetid_list):
    r"""
    Returns:
        list_ (list): imageset_note of each imgsetid in imgsetid_list

    RESTful:
        Method: GET
        URL:    /api/imageset/note/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    encnote_list = ibs.db.get(
        const.IMAGESET_TABLE,
        ('imageset_note',),
        imgsetid_list,
        id_colname='imageset_rowid',
    )
    return encnote_list


@register_ibs_method
@accessor_decors.deleter
@register_api('/api/imageset/', methods=['DELETE'])
def delete_imagesets(ibs, imgsetid_list):
    r"""
    Removes imagesets and thier relationships (images are not effected)

    RESTful:
        Method: DELETE
        URL:    /api/imageset/
    """
    # Optimization hack, less SQL calls
    # gsgrid_list = ut.flatten(ibs.get_imageset_gsgrids(imgsetid_list=imgsetid_list))
    # ibs.db.delete_rowids(const.GSG_RELATION_TABLE, gsgrid_list)
    # ibs.db.delete(const.GSG_RELATION_TABLE, imgsetid_list, id_colname='imageset_rowid')
    if ut.VERBOSE:
        print('[ibs] deleting %d imagesets' % len(imgsetid_list))
    ibs.delete_gsgr_imageset_relations(imgsetid_list)
    ibs.db.delete_rowids(const.IMAGESET_TABLE, imgsetid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/time/posix/end/', methods=['GET'])
def get_imageset_end_time_posix(ibs, imageset_rowid_list):
    r"""
    imageset_end_time_posix_list <- imageset.imageset_end_time_posix[imageset_rowid_list]

    gets data from the "native" column "imageset_end_time_posix" in the "imageset" table

    Args:
        imageset_rowid_list (list):

    Returns:
        list: imageset_end_time_posix_list

    TemplateInfo:
        Tgetter_table_column
        col = imageset_end_time_posix
        tbl = imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/time/posix/end/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> imageset_rowid_list = ibs._get_all_imageset_rowids()
        >>> imageset_end_time_posix_list = ibs.get_imageset_end_time_posix(imageset_rowid_list)
        >>> assert len(imageset_rowid_list) == len(imageset_end_time_posix_list)
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_END_TIME_POSIX,)
    imageset_end_time_posix_list = ibs.db.get(
        const.IMAGESET_TABLE, colnames, id_iter, id_colname='rowid'
    )
    return imageset_end_time_posix_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/gps/lat/', methods=['GET'], __api_plural_check__=False)
def get_imageset_gps_lats(ibs, imageset_rowid_list):
    r"""
    imageset_gps_lat_list <- imageset.imageset_gps_lat[imageset_rowid_list]

    gets data from the "native" column "imageset_gps_lat" in the "imageset" table

    Args:
        imageset_rowid_list (list):

    Returns:
        list: imageset_gps_lat_list

    TemplateInfo:
        Tgetter_table_column
        col = imageset_gps_lat
        tbl = imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/gps/lat/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> imageset_rowid_list = ibs._get_all_imageset_rowids()
        >>> imageset_gps_lat_list = ibs.get_imageset_gps_lats(imageset_rowid_list)
        >>> assert len(imageset_rowid_list) == len(imageset_gps_lat_list)
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_GPS_LAT,)
    imageset_gps_lat_list = ibs.db.get(
        const.IMAGESET_TABLE, colnames, id_iter, id_colname='rowid'
    )
    return imageset_gps_lat_list


@register_ibs_method
@register_api('/api/imageset/info/', methods=['PUT'])
def update_imageset_info(ibs, imageset_rowid_list, **kwargs):
    r"""
    sets start and end time for imagesets

    FIXME: should not need to bulk update, should be handled as it goes

    RESTful:
        Method: PUT
        URL:    /api/imageset/info/

    Example:
        >>> # DOCTEST_DISABLE
        >>> imageset_rowid_list = ibs.get_valid_imgsetids()
    """
    gids_list_ = ibs.get_imageset_gids(imageset_rowid_list)
    hasgids_list = [len(gids) > 0 for gids in gids_list_]
    gids_list = ut.compress(gids_list_, hasgids_list)
    imgsetid_list = ut.compress(imageset_rowid_list, hasgids_list)
    unixtimes_list = [
        ibs.get_image_unixtime(gid_list, **kwargs) for gid_list in gids_list
    ]
    # TODO: replace -1's with nans and do nanmin
    imageset_start_time_posix_list = [min(unixtimes) for unixtimes in unixtimes_list]
    imageset_end_time_posix_list = [max(unixtimes) for unixtimes in unixtimes_list]
    ibs.set_imageset_start_time_posix(imgsetid_list, imageset_start_time_posix_list)
    ibs.set_imageset_end_time_posix(imgsetid_list, imageset_end_time_posix_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/gps/lon/', methods=['GET'], __api_plural_check__=False)
def get_imageset_gps_lons(ibs, imageset_rowid_list):
    r"""
    imageset_gps_lon_list <- imageset.imageset_gps_lon[imageset_rowid_list]

    gets data from the "native" column "imageset_gps_lon" in the "imageset" table

    Args:
        imageset_rowid_list (list):

    Returns:
        list: imageset_gps_lon_list

    TemplateInfo:
        Tgetter_table_column
        col = imageset_gps_lon
        tbl = imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/gps/lon/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> imageset_rowid_list = ibs._get_all_imageset_rowids()
        >>> imageset_gps_lon_list = ibs.get_imageset_gps_lons(imageset_rowid_list)
        >>> assert len(imageset_rowid_list) == len(imageset_gps_lon_list)
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_GPS_LON,)
    imageset_gps_lon_list = ibs.db.get(
        const.IMAGESET_TABLE, colnames, id_iter, id_colname='rowid'
    )
    return imageset_gps_lon_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/imageset/note/', methods=['GET'])
def get_imageset_notes(ibs, imageset_rowid_list):
    r"""
    imageset_note_list <- imageset.imageset_note[imageset_rowid_list]

    gets data from the "native" column "imageset_note" in the "imageset" table

    Args:
        imageset_rowid_list (list):

    Returns:
        list: imageset_note_list

    TemplateInfo:
        Tgetter_table_column
        col = imageset_note
        tbl = imageset

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> imageset_rowid_list = ibs._get_all_imageset_rowids()
        >>> imageset_note_list = ibs.get_imageset_notes(imageset_rowid_list)
        >>> assert len(imageset_rowid_list) == len(imageset_note_list)
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_NOTE,)
    imageset_note_list = ibs.db.get(
        const.IMAGESET_TABLE, colnames, id_iter, id_colname='rowid'
    )
    return imageset_note_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/metadata/', methods=['GET'])
def get_imageset_metadata(ibs, imageset_rowid_list, return_raw=False):
    r"""
    Returns:
        list_ (list): imageset metadata dictionary

    RESTful:
        Method: GET
        URL:    /api/imageset/metadata/
    """
    metadata_str_list = ibs.db.get(
        const.IMAGESET_TABLE, ('imageset_metadata_json',), imageset_rowid_list
    )
    metadata_list = []
    for metadata_str in metadata_str_list:
        if metadata_str in [None, '']:
            metadata_dict = {}
        else:
            metadata_dict = metadata_str if return_raw else ut.from_json(metadata_str)
        metadata_list.append(metadata_dict)
    return metadata_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/occurrence/', methods=['GET'])
def get_imageset_occurrence_flags(ibs, imageset_rowid_list):
    r"""
    imageset_occurrence_flag_list <- imageset.imageset_occurrence_flag[imageset_rowid_list]

    gets data from the "native" column "imageset_occurrence_flag" in the "imageset" table

    Args:
        imageset_rowid_list (list):

    Returns:
        list: imageset_occurrence_flag_list

    TemplateInfo:
        Tgetter_table_column
        col = imageset_occurrence_flag
        tbl = imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/occurrence/

    CommandLine:
        python -m wbia.control.manual_imageset_funcs --test-get_imageset_occurrence_flags

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> imageset_rowid_list = ibs._get_all_imageset_rowids()
        >>> imageset_occurrence_flag_list = ibs.get_imageset_occurrence_flags(imageset_rowid_list)
        >>> assert len(imageset_rowid_list) == len(imageset_occurrence_flag_list)
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_OCCURRENCE_FLAG,)
    imageset_occurrence_flag_list = ibs.db.get(
        const.IMAGESET_TABLE, colnames, id_iter, id_colname='rowid'
    )
    return imageset_occurrence_flag_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/processed/', methods=['GET'])
def get_imageset_processed_flags(ibs, imageset_rowid_list):
    r"""
    imageset_processed_flag_list <- imageset.imageset_processed_flag[imageset_rowid_list]

    gets data from the "native" column "imageset_processed_flag" in the "imageset" table

    Args:
        imageset_rowid_list (list):

    Returns:
        list: imageset_processed_flag_list

    TemplateInfo:
        Tgetter_table_column
        col = imageset_processed_flag
        tbl = imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/processed/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> imageset_rowid_list = ibs._get_all_imageset_rowids()
        >>> imageset_processed_flag_list = ibs.get_imageset_processed_flags(imageset_rowid_list)
        >>> assert len(imageset_rowid_list) == len(imageset_processed_flag_list)
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_PROCESSED_FLAG,)
    imageset_processed_flag_list = ibs.db.get(
        const.IMAGESET_TABLE, colnames, id_iter, id_colname='rowid'
    )
    return imageset_processed_flag_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/shipped/', methods=['GET'])
def get_imageset_shipped_flags(ibs, imageset_rowid_list):
    r"""
    imageset_shipped_flag_list <- imageset.imageset_shipped_flag[imageset_rowid_list]

    gets data from the "native" column "imageset_shipped_flag" in the "imageset" table

    Args:
        imageset_rowid_list (list):

    Returns:
        list: imageset_shipped_flag_list

    TemplateInfo:
        Tgetter_table_column
        col = imageset_shipped_flag
        tbl = imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/shipped/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> imageset_rowid_list = ibs._get_all_imageset_rowids()
        >>> imageset_shipped_flag_list = ibs.get_imageset_shipped_flags(imageset_rowid_list)
        >>> assert len(imageset_rowid_list) == len(imageset_shipped_flag_list)
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_SHIPPED_FLAG,)
    imageset_shipped_flag_list = ibs.db.get(
        const.IMAGESET_TABLE, colnames, id_iter, id_colname='rowid'
    )
    return imageset_shipped_flag_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/imageset/time/posix/start/', methods=['GET'])
def get_imageset_start_time_posix(ibs, imageset_rowid_list):
    r"""
    imageset_start_time_posix_list <- imageset.imageset_start_time_posix[imageset_rowid_list]

    gets data from the "native" column "imageset_start_time_posix" in the "imageset" table

    Args:
        imageset_rowid_list (list):

    Returns:
        list: imageset_start_time_posix_list

    TemplateInfo:
        Tgetter_table_column
        col = imageset_start_time_posix
        tbl = imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/time/posix/start/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> imageset_rowid_list = ibs._get_all_imageset_rowids()
        >>> imageset_start_time_posix_list = ibs.get_imageset_start_time_posix(imageset_rowid_list)
        >>> assert len(imageset_rowid_list) == len(imageset_start_time_posix_list)
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_START_TIME_POSIX,)
    imageset_start_time_posix_list = ibs.db.get(
        const.IMAGESET_TABLE, colnames, id_iter, id_colname='rowid'
    )
    return imageset_start_time_posix_list


@register_ibs_method
@accessor_decors.getter
@register_api('/api/imageset/duration/', methods=['GET'])
def get_imageset_duration(ibs, imageset_rowid_list):
    r"""
    gets the imageset's duration

    Args:
        imageset_rowid_list (list):

    Returns:
        list: imageset_duration

    RESTful:
        Method: GET
        URL:    /api/imageset/duration/
    """

    def _process(start, end):
        if start is None or end is None:
            return 'None'
        seconds_in_day = 60 * 60 * 24
        days = 0
        duration = int(end - start)
        if duration >= seconds_in_day:
            days = duration // seconds_in_day
            duration = duration % seconds_in_day
        duration_str = time.strftime('%H:%M:%S', time.gmtime(duration))
        if days > 0:
            duration_str = '%d days, %s' % (days, duration_str,)
        return duration_str

    import time

    start_time_list = ibs.get_imageset_start_time_posix(imageset_rowid_list)
    end_time_list = ibs.get_imageset_end_time_posix(imageset_rowid_list)
    zipped = zip(start_time_list, end_time_list)
    duration_list = [_process(start, end) for start, end in zipped]
    return duration_list


@register_ibs_method
@accessor_decors.setter
@register_api('/api/imageset/time/posix/end/', methods=['PUT'])
def set_imageset_end_time_posix(ibs, imageset_rowid_list, imageset_end_time_posix_list):
    r"""
    imageset_end_time_posix_list -> imageset.imageset_end_time_posix[imageset_rowid_list]

    Args:
        imageset_rowid_list
        imageset_end_time_posix_list

    TemplateInfo:
        Tsetter_native_column
        tbl = imageset
        col = imageset_end_time_posix

    RESTful:
        Method: PUT
        URL:    /api/imageset/time/posix/end/
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_END_TIME_POSIX,)
    ibs.db.set(const.IMAGESET_TABLE, colnames, imageset_end_time_posix_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/imageset/gps/lat/', methods=['PUT'], __api_plural_check__=False)
def set_imageset_gps_lats(ibs, imageset_rowid_list, imageset_gps_lat_list):
    r"""
    imageset_gps_lat_list -> imageset.imageset_gps_lat[imageset_rowid_list]

    Args:
        imageset_rowid_list
        imageset_gps_lat_list

    TemplateInfo:
        Tsetter_native_column
        tbl = imageset
        col = imageset_gps_lat

    RESTful:
        Method: PUT
        URL:    /api/imageset/gps/lat/
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_GPS_LAT,)
    ibs.db.set(const.IMAGESET_TABLE, colnames, imageset_gps_lat_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/imageset/gps/lon/', methods=['PUT'], __api_plural_check__=False)
def set_imageset_gps_lons(ibs, imageset_rowid_list, imageset_gps_lon_list):
    r"""
    imageset_gps_lon_list -> imageset.imageset_gps_lon[imageset_rowid_list]

    Args:
        imageset_rowid_list
        imageset_gps_lon_list

    TemplateInfo:
        Tsetter_native_column
        tbl = imageset
        col = imageset_gps_lon

    RESTful:
        Method: PUT
        URL:    /api/imageset/gps/lon/
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_GPS_LON,)
    ibs.db.set(const.IMAGESET_TABLE, colnames, imageset_gps_lon_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/imageset/note/', methods=['PUT'])
def set_imageset_notes(ibs, imageset_rowid_list, imageset_note_list):
    r"""
    imageset_note_list -> imageset.imageset_note[imageset_rowid_list]

    Args:
        imageset_rowid_list
        imageset_note_list

    TemplateInfo:
        Tsetter_native_column
        tbl = imageset
        col = imageset_note
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_NOTE,)
    ibs.db.set(const.IMAGESET_TABLE, colnames, imageset_note_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/imageset/metadata/', methods=['PUT'])
def set_imageset_metadata(ibs, imageset_rowid_list, metadata_dict_list):
    r"""
    Sets the imageset's metadata using a metadata dictionary

    RESTful:
        Method: PUT
        URL:    /api/imageset/metadata/

    """
    id_iter = ((gid,) for gid in imageset_rowid_list)
    metadata_str_list = []
    for metadata_dict in metadata_dict_list:
        metadata_str = ut.to_json(metadata_dict)
        metadata_str_list.append(metadata_str)
    val_list = ((metadata_str,) for metadata_str in metadata_str_list)
    ibs.db.set(const.IMAGESET_TABLE, ('imageset_metadata_json',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/imageset/occurrence/', methods=['PUT'])
def set_imageset_occurrence_flags(
    ibs, imageset_rowid_list, imageset_occurrence_flag_list
):
    r"""
    imageset_occurrence_flag_list -> imageset.imageset_occurrence_flag[imageset_rowid_list]

    Args:
        imageset_rowid_list
        imageset_occurrence_flag_list

    TemplateInfo:
        Tsetter_native_column
        tbl = imageset
        col = imageset_occurrence_flag

    RESTful:
        Method: PUT
        URL:    /api/imageset/occurrence/
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_OCCURRENCE_FLAG,)
    val_iter = ((occurrence_flag,) for occurrence_flag in imageset_occurrence_flag_list)
    ibs.db.set(const.IMAGESET_TABLE, colnames, val_iter, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/imageset/processed/', methods=['PUT'])
def set_imageset_processed_flags(ibs, imageset_rowid_list, imageset_processed_flag_list):
    r"""
    imageset_processed_flag_list -> imageset.imageset_processed_flag[imageset_rowid_list]

    Args:
        imageset_rowid_list
        imageset_processed_flag_list

    TemplateInfo:
        Tsetter_native_column
        tbl = imageset
        col = imageset_processed_flag

    RESTful:
        Method: PUT
        URL:    /api/imageset/processed/
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_PROCESSED_FLAG,)
    val_iter = ((processed_flag,) for processed_flag in imageset_processed_flag_list)
    ibs.db.set(const.IMAGESET_TABLE, colnames, val_iter, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/imageset/shipped/', methods=['PUT'])
def set_imageset_shipped_flags(ibs, imageset_rowid_list, imageset_shipped_flag_list):
    r"""
    imageset_shipped_flag_list -> imageset.imageset_shipped_flag[imageset_rowid_list]

    Args:
        imageset_rowid_list
        imageset_shipped_flag_list

    TemplateInfo:
        Tsetter_native_column
        tbl = imageset
        col = imageset_shipped_flag

    RESTful:
        Method: PUT
        URL:    /api/imageset/shipped/
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_SHIPPED_FLAG,)
    val_iter = ((shipped_flag,) for shipped_flag in imageset_shipped_flag_list)
    ibs.db.set(const.IMAGESET_TABLE, colnames, val_iter, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/imageset/time/posix/start/', methods=['PUT'])
def set_imageset_start_time_posix(
    ibs, imageset_rowid_list, imageset_start_time_posix_list
):
    r"""
    imageset_start_time_posix_list -> imageset.imageset_start_time_posix[imageset_rowid_list]

    Args:
        imageset_rowid_list
        imageset_start_time_posix_list

    TemplateInfo:
        Tsetter_native_column
        tbl = imageset
        col = imageset_start_time_posix

    RESTful:
        Method: PUT
        URL:    /api/imageset/time/posix/start/
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_START_TIME_POSIX,)
    ibs.db.set(const.IMAGESET_TABLE, colnames, imageset_start_time_posix_list, id_iter)


@register_ibs_method
# @accessor_decors.cache_getter(const.IMAGESET_TABLE, IMAGESET_SMART_WAYPOINT_ID)
@register_api('/api/imageset/smart/waypoint/', methods=['GET'])
def get_imageset_smart_waypoint_ids(ibs, imageset_rowid_list):
    r"""
    imageset_smart_waypoint_id_list <- imageset.imageset_smart_waypoint_id[imageset_rowid_list]

    gets data from the "native" column "imageset_smart_waypoint_id" in the "imageset" table

    Args:
        imageset_rowid_list (list):

    Returns:
        list: imageset_smart_waypoint_id_list

    TemplateInfo:
        Tgetter_table_column
        col = imageset_smart_waypoint_id
        tbl = imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/smart/waypoint/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> imageset_rowid_list = ibs._get_all_imageset_rowids()
        >>> imageset_smart_waypoint_id_list = ibs.get_imageset_smart_waypoint_ids(imageset_rowid_list)
        >>> assert len(imageset_rowid_list) == len(imageset_smart_waypoint_id_list)
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_SMART_WAYPOINT_ID,)
    imageset_smart_waypoint_id_list = ibs.db.get(
        const.IMAGESET_TABLE, colnames, id_iter, id_colname='rowid'
    )
    return imageset_smart_waypoint_id_list


@register_ibs_method
# @accessor_decors.cache_getter(const.IMAGESET_TABLE, IMAGESET_SMART_XML_FNAME)
@register_api('/api/imageset/smart/xml/file/name/', methods=['GET'])
def get_imageset_smart_xml_fnames(ibs, imageset_rowid_list):
    r"""
    imageset_smart_xml_fname_list <- imageset.imageset_smart_xml_fname[imageset_rowid_list]

    gets data from the "native" column "imageset_smart_xml_fname" in the "imageset" table

    Args:
        imageset_rowid_list (list):

    Returns:
        list: imageset_smart_xml_fname_list

    TemplateInfo:
        Tgetter_table_column
        col = imageset_smart_xml_fname
        tbl = imageset

    RESTful:
        Method: GET
        URL:    /api/imageset/smart/xml/file/name/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_imageset_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> imageset_rowid_list = ibs._get_all_imageset_rowids()
        >>> imageset_smart_xml_fname_list = ibs.get_imageset_smart_xml_fnames(imageset_rowid_list)
        >>> assert len(imageset_rowid_list) == len(imageset_smart_xml_fname_list)
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_SMART_XML_FNAME,)
    imageset_smart_xml_fname_list = ibs.db.get(
        const.IMAGESET_TABLE, colnames, id_iter, id_colname='rowid'
    )
    return imageset_smart_xml_fname_list


@register_ibs_method
# @accessor_decors.cache_getter(const.IMAGESET_TABLE, IMAGESET_SMART_XML_FNAME)
@register_api('/api/imageset/smart/xml/file/content/', methods=['GET'])
def get_imageset_smart_xml_contents(ibs, imageset_rowid_list):
    from os.path import join, exists

    imageset_smart_xml_fname_list = ibs.get_imageset_smart_xml_fnames(imageset_rowid_list)
    content_list = []
    smart_patrol_dir = ibs.get_smart_patrol_dir()
    for imageset_smart_xml_fname in imageset_smart_xml_fname_list:
        if imageset_smart_xml_fname is None:
            content_list.append(None)
        else:
            imageset_smart_xml_fpath = join(smart_patrol_dir, imageset_smart_xml_fname)
            if exists(imageset_smart_xml_fpath):
                with open(imageset_smart_xml_fpath, 'r') as imageset_smart_xml:
                    content = imageset_smart_xml.read()
                    content_list.append(content)
            else:
                content_list.append(None)
    return content_list


@register_ibs_method
@register_api('/api/imageset/smart/waypoint/', methods=['PUT'])
def set_imageset_smart_waypoint_ids(
    ibs, imageset_rowid_list, imageset_smart_waypoint_id_list
):
    r"""
    imageset_smart_waypoint_id_list -> imageset.imageset_smart_waypoint_id[imageset_rowid_list]

    Args:
        imageset_rowid_list
        imageset_smart_waypoint_id_list

    TemplateInfo:
        Tsetter_native_column
        tbl = imageset
        col = imageset_smart_waypoint_id

    RESTful:
        Method: PUT
        URL:    /api/imageset/smart/waypoint/
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_SMART_WAYPOINT_ID,)
    ibs.db.set(const.IMAGESET_TABLE, colnames, imageset_smart_waypoint_id_list, id_iter)


@register_ibs_method
@register_api('/api/imageset/smart/xml/file/name/', methods=['PUT'])
def set_imageset_smart_xml_fnames(
    ibs, imageset_rowid_list, imageset_smart_xml_fname_list
):
    r"""
    imageset_smart_xml_fname_list -> imageset.imageset_smart_xml_fname[imageset_rowid_list]

    Args:
        imageset_rowid_list
        imageset_smart_xml_fname_list

    TemplateInfo:
        Tsetter_native_column
        tbl = imageset
        col = imageset_smart_xml_fname

    RESTful:
        Method: PUT
        URL:    /api/imageset/smart/xml/fname/
    """
    id_iter = imageset_rowid_list
    colnames = (IMAGESET_SMART_XML_FNAME,)
    ibs.db.set(const.IMAGESET_TABLE, colnames, imageset_smart_xml_fname_list, id_iter)


def testdata_ibs():
    r"""
    """
    import wbia

    ibs = wbia.opendb('testdb1')
    config2_ = None
    return ibs, config2_


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.control.manual_imageset_funcs
        python -m wbia.control.manual_imageset_funcs --allexamples
        python -m wbia.control.manual_imageset_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
