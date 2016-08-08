# -*- coding: utf-8 -*-
"""
Autogen:
    python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'ibeis.templates.template_generator')"  # NOQA
    sh Tgen.sh --key annot --invert --Tcfg with_getters=True with_setters=True --modfname manual_annot_funcs --funcname-filter=age_m  # NOQA
    sh Tgen.sh --key annot --invert --Tcfg with_getters=True with_setters=True --modfname manual_annot_funcs --funcname-filter=is_  # NOQA
    sh Tgen.sh --key annot --invert --Tcfg with_getters=True with_setters=True --modfname manual_annot_funcs --funcname-filter=is_ --diff  # NOQA
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import uuid
import numpy as np  # NOQA
from ibeis import constants as const
from ibeis.control import accessor_decors, controller_inject
import utool as ut
from ibeis.other import ibsfuncs
from ibeis.control.controller_inject import make_ibs_register_decorator
from collections import namedtuple
from ibeis.web import routes_ajax
print, rrr, profile = ut.inject2(__name__, '[manual_annot]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


ANNOT_AGE_MONTHS_EST_MAX = 'annot_age_months_est_max'
ANNOT_AGE_MONTHS_EST_MIN = 'annot_age_months_est_min'
ANNOT_NOTE               = 'annot_note'
ANNOT_NUM_VERTS          = 'annot_num_verts'
ANNOT_PARENT_ROWID       = 'annot_parent_rowid'
ANNOT_ROWID              = 'annot_rowid'
ANNOT_TAG_TEXT           = 'annot_tag_text'
ANNOT_SEMANTIC_UUID      = 'annot_semantic_uuid'
ANNOT_THETA              = 'annot_theta'
ANNOT_VERTS              = 'annot_verts'
ANNOT_UUID               = 'annot_uuid'
ANNOT_YAW                = 'annot_yaw'
ANNOT_VISUAL_UUID        = 'annot_visual_uuid'
CONFIG_ROWID             = 'config_rowid'
FEATWEIGHT_ROWID         = 'featweight_rowid'
IMAGE_ROWID              = 'image_rowid'
NAME_ROWID               = 'name_rowid'
SPECIES_ROWID            = 'species_rowid'
ANNOT_EXEMPLAR_FLAG      = 'annot_exemplar_flag'
ANNOT_QUALITY            = 'annot_quality'
ANNOT_ROWIDS             = 'annot_rowids'
GAR_ROWID                = 'gar_rowid'

SemanticInfoTup = namedtuple('SemanticInfoTup', ('image_uuid', 'verts',
                                                 'theta', 'yaw', 'name',
                                                 'species'))


# ==========
# IDERS
# ==========

# TODO CACHE THIS AND FIND WHAT IT SHOULD INVALIDATE IT
# ADD ANNOTS, DELETE ANNOTS ANYTHING ELSE?
@register_ibs_method
@accessor_decors.ider
def _get_all_aids(ibs):
    r"""
    Returns:
        list_ (list):  all unfiltered aids (annotation rowids)
    """
    all_aids = ibs.db.get_all_rowids(const.ANNOTATION_TABLE)
    return all_aids


@register_ibs_method
def get_num_annotations(ibs, **kwargs):
    r"""
    Number of valid annotations
    """
    aid_list = ibs.get_valid_aids(**kwargs)
    return len(aid_list)


@register_ibs_method
@accessor_decors.ider
@register_api('/api/annot/', methods=['GET'])
def get_valid_aids(ibs, imgsetid=None, include_only_gid_list=None,
                   yaw='no-filter',
                   is_exemplar=None,
                   species=None,
                   is_known=None,
                   hasgt=None,
                   minqual=None,
                   has_timestamp=None,
                   min_timedelta=None):
    r"""
    High level function for getting all annotation ids according a set of filters.

    Note: The yaw value cannot be None as a default because None is used as a
          filtering value

    Args:
        ibs (IBEISController):  ibeis controller object
        imgsetid (int): imageset id (default = None)
        include_only_gid_list (list): if specified filters annots not in these gids (default = None)
        yaw (str): (default = 'no-filter')
        is_exemplar (bool): if specified filters annots to either be or not be exemplars (default = None)
        species (str): (default = None)
        is_known (bool): (default = None)
        min_timedelta (int): minimum timedelta between annots of known individuals
        hasgt (bool): (default = None)

    Returns:
        list: aid_list - a list of valid ANNOTATION unique ids

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_valid_aids

    Ignore:
        ibs.print_annotation_table()

    RESTful:
        Method: GET
        URL:    /api/annot/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> ut.exec_funckw(get_valid_aids, globals())
        >>> imgsetid = 1
        >>> yaw = 'no-filter'
        >>> species = ibs.const.TEST_SPECIES.ZEB_PLAIN
        >>> is_known = False
        >>> ibs.delete_all_imagesets()
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> aid_list = get_valid_aids(ibs, imgsetid=imgsetid, species=species, is_known=is_known)
        >>> ut.assert_eq(ibs.get_annot_names(aid_list), [ibs.const.UNKNOWN] * 2, 'bad name')
        >>> ut.assert_eq(ibs.get_annot_species(aid_list), [species] * 2, 'bad species')
        >>> ut.assert_eq(ibs.get_annot_exemplar_flags(aid_list), [False] * 2, 'bad exemplar')
        >>> result = str(aid_list)
        >>> print(result)

        [1, 4]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list1 = get_valid_aids(ibs, is_exemplar=True)
        >>> aid_list2 = get_valid_aids(ibs, is_exemplar=False)
        >>> intersect_aids = set(aid_list1).intersection(aid_list2)
        >>> ut.assert_eq(len(aid_list1), 9)
        >>> ut.assert_eq(len(aid_list2), 4)
        >>> ut.assert_eq(len(intersect_aids), 0)

    Ignore:
        import utool as ut
        setup = ut.codeblock(
            '''
            import ibeis
            ibs = ibeis.opendb('PZ_Master1')
            '''
        )
        stmt_list = [
            ut.codeblock(
                '''
                ibs.db.get_all_rowids_where(ibs.const.ANNOTATION_TABLE, ibeis.control.DB_SCHEMA.ANNOT_PARENT_ROWID + " IS NULL", tuple())
                '''),
            ut.codeblock(
                '''
                ibs.db.get_all_rowids(ibs.const.ANNOTATION_TABLE)
                '''),
        ]
        iterations = 100
        verbose = True
        _ = ut.timeit_compare(stmt_list, setup=setup, iterations=iterations, verbose=verbose)

    """
    # getting imageset aid
    if imgsetid is None:
        if is_exemplar is not None:
            # Optimization Hack
            aid_list = ibs.db.get_all_rowids_where(
                const.ANNOTATION_TABLE, 'annot_exemplar_flag=?', (is_exemplar,))
        else:
            aid_list = ibs._get_all_aids()
    else:
        # HACK: Check to see if you want the
        # exemplar "imageset" (image group)
        imagesettext = ibs.get_imageset_text(imgsetid)
        if imagesettext == const.EXEMPLAR_IMAGESETTEXT:
            is_exemplar = True
        aid_list = ibs.get_imageset_aids(imgsetid)
        if is_exemplar is True:
            # corresponding unoptimized hack for is_exemplar
            flag_list = ibs.get_annot_exemplar_flags(aid_list)
            aid_list  = ut.compress(aid_list, flag_list)
        elif is_exemplar is False:
            flag_list = ibs.get_annot_exemplar_flags(aid_list)
            aid_list  = ut.filterfalse_items(aid_list, flag_list)
    aid_list = filter_annotation_set(
        ibs, aid_list, include_only_gid_list=include_only_gid_list, yaw=yaw,
        is_exemplar=is_exemplar, species=species, is_known=is_known,
        hasgt=hasgt, minqual=minqual, has_timestamp=has_timestamp,
        min_timedelta=min_timedelta)
    return aid_list


@register_ibs_method
@register_api('/api/annot/<rowid>/', methods=['GET'])
def annotation_src_api(rowid=None):
    r"""
    Returns the base64 encoded image of annotation <aid>

    RESTful:
        Method: GET
        URL:    /api/annot/<aid>/
    """
    return routes_ajax.annotation_src(rowid)


def filter_annotation_set(ibs, aid_list, include_only_gid_list=None,
                          yaw='no-filter', is_exemplar=None, species=None,
                          is_known=None, hasgt=None, minqual=None,
                          has_timestamp=None, min_timedelta=None):
    # -- valid aid filtering --
    if include_only_gid_list is not None:
        gid_list     = ibs.get_annot_gids(aid_list)
        is_valid_gid = [gid in include_only_gid_list for gid in gid_list]
        aid_list     = ut.compress(aid_list, is_valid_gid)
    if yaw != 'no-filter':
        yaw_list     = ibs.get_annot_yaws(aid_list)
        is_valid_yaw = [yaw == flag for flag in yaw_list]
        aid_list     = ut.compress(aid_list, is_valid_yaw)
    if species is not None:
        aid_list = ibs.filter_aids_to_species(aid_list, species)
    if is_known is not None:
        aid_list = ibs.filter_aids_without_name(aid_list, invert=not is_known)
    if minqual is not None:
        aid_list = ibs.filter_aids_to_quality(aid_list, minqual, unknown_ok=True)
    if has_timestamp is not None:
        aid_list = ibs.filter_aids_without_timestamps(aid_list, invert=not has_timestamp)
    if min_timedelta is not None:
        aid_list = ibs.filter_annots_using_minimum_timedelta(aid_list, min_timedelta)
    if hasgt:
        hasgt_list = ibs.get_annot_has_groundtruth(aid_list)
        aid_list = ut.compress(aid_list, hasgt_list)
    aid_list = sorted(aid_list)
    return aid_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_annot_aid(ibs, aid_list, eager=True, nInput=None):
    """ self verifier
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids() + [None, -1, 10434320432]
        >>> aid_list_ = ibs.get_annot_aid(aid_list)
        >>> assert [r is None for r in aid_list_[-3:]]
        >>> assert [r is not None for r in aid_list_[0:-3]]
    """
    id_iter = aid_list
    colnames = (ANNOT_ROWID,)
    aid_list = ibs.db.get(const.ANNOTATION_TABLE, colnames,
                          id_iter, id_colname='rowid', eager=eager, nInput=nInput)
    return aid_list


# ==========
# ADDERS
# ==========


@register_ibs_method
@accessor_decors.adder
@accessor_decors.cache_invalidator(const.IMAGE_TABLE, colnames=[ANNOT_ROWIDS], rowidx=None)
@accessor_decors.cache_invalidator(const.NAME_TABLE, colnames=[ANNOT_ROWIDS])
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['percent_names_with_exemplar_str'])
@register_api('/api/annot/', methods=['POST'])
def add_annots(ibs, gid_list, bbox_list=None, theta_list=None,
                species_list=None, nid_list=None, name_list=None,
                detect_confidence_list=None, notes_list=None,
                vert_list=None, annot_uuid_list=None, yaw_list=None,
                annot_visual_uuid_list=None, annot_semantic_uuid_list=None,
                species_rowid_list=None, quiet_delete_thumbs=False,
                prevent_visual_duplicates=True, skip_cleaning=False, **kwargs):
    r"""
    Adds an annotation to images

    # TODO:
        remove annot_visual_uuid_list and annot_semantic_uuid_list
        They are always infered

    Args:
        gid_list                 (list): image rowids to add annotation to
        bbox_list                (list): of [x, y, w, h] bounding boxes for each image (supply verts instead)
        theta_list               (list): orientations of annotations
        species_list             (list):
        nid_list                 (list):
        name_list                (list):
        detect_confidence_list   (list):
        notes_list               (list):
        vert_list                (list): alternative to bounding box
        annot_uuid_list          (list):
        yaw_list                 (list):
        annot_visual_uuid_list   (list):
        annot_semantic_uuid_list (list):
        quiet_delete_thumbs      (bool):

    Returns:
        list: aid_list

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-add_annots
        python -m ibeis.control.manual_annot_funcs --test-add_annots --verbose --print-caller

    Ignore:
       theta_list = None
       species_list = None
       nid_list = None
       name_list = None
       detect_confidence_list = None
       notes_list = None
       vert_list = None
       annot_uuid_list = None
       yaw_list = None
       quiet_delete_thumbs = False
       prevent_visual_duplicates = False

    RESTful:
        Method: POST
        URL:    /api/annot/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> prevalid = ibs.get_valid_aids()
        >>> num_add = 2
        >>> gid_list = ibs.get_valid_gids()[0:num_add]
        >>> bbox_list = [(int(w * .1), int(h * .6), int(w * .5), int(h *  .3))
        ...              for (w, h) in ibs.get_image_sizes(gid_list)]
        >>> # Add a test annotation
        >>> print('Testing add_annots')
        >>> aid_list = ibs.add_annots(gid_list, bbox_list=bbox_list)
        >>> bbox_list2 = ibs.get_annot_bboxes(aid_list)
        >>> vert_list2 = ibs.get_annot_verts(aid_list)
        >>> theta_list2 = ibs.get_annot_thetas(aid_list)
        >>> name_list2 = ibs.get_annot_names(aid_list)
        >>> print('Ensure=False. Should get back None chip fpaths')
        >>> chip_fpaths2 = ibs.get_annot_chip_fpath(aid_list, ensure=False)
        >>> assert [fpath is None for fpath in chip_fpaths2], 'should not have fpaths'
        >>> print('Ensure=True. Should get back None chip fpaths')
        >>> chip_fpaths = ibs.get_annot_chip_fpath(aid_list, ensure=True)
        >>> assert all([ut.checkpath(fpath, verbose=True) for fpath in chip_fpaths]), 'paths should exist'
        >>> ut.assert_eq(len(aid_list), num_add)
        >>> ut.assert_eq(len(vert_list2[0]), 4)
        >>> assert bbox_list2 == bbox_list, 'bboxes are unequal'
        >>> # Be sure to remove test annotation
        >>> # if this test fails a resetdbs might be nessary
        >>> result = ''
        >>> visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
        >>> semantic_uuid_list = ibs.get_annot_semantic_uuids(aid_list)
        >>> result += str(visual_uuid_list) + '\n'
        >>> result += str(semantic_uuid_list) + '\n'
        >>> print('Cleaning up. Removing added annotations')
        >>> ibs.delete_annots(aid_list)
        >>> assert not any([ut.checkpath(fpath, verbose=True) for fpath in chip_fpaths]), 'chip paths'
        >>> postvalid = ibs.get_valid_aids()
        >>> assert prevalid == postvalid, 'prevalid != postvalid'
        >>> result += str(postvalid)
        >>> print(result)
        [UUID('30f7639b-5161-a561-2c4f-41aed64e5b65'), UUID('5ccbb26d-104f-e655-cf2b-cf92e0ad2fd2')]
        [UUID('3e3e9c98-e47c-f153-7101-f3d4fdadfb90'), UUID('dbf3b1a2-2188-75b4-07d4-0ef7e4787d23')]
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        # semantic uuids change when hashing is different

        [UUID('30f7639b-5161-a561-2c4f-41aed64e5b65'), UUID('5ccbb26d-104f-e655-cf2b-cf92e0ad2fd2')]
        [UUID('68160c90-4b82-dc96-dafa-b12948739577'), UUID('03e74d19-1bf7-bc43-a291-8ee06a44da2e')]
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    Example2:
        >>> # Test with prevent_visual_duplicates on
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> prevalid = ibs.get_valid_aids()
        >>> num_add = 1
        >>> gid_list = ibs.get_valid_gids()[0:1] * num_add
        >>> bbox_list = [(int(w * .1), int(h * .6), int(w * .5), int(h *  .3))
        ...              for (w, h) in ibs.get_image_sizes(gid_list)]
        >>> bbox_list2 = [(int(w * .2), int(h * .6), int(w * .5), int(h *  .3))
        ...              for (w, h) in ibs.get_image_sizes(gid_list)]
        >>> # Add a test annotation
        >>> print('Testing add_annots')
        >>> aid_list1 = ibs.add_annots(gid_list, bbox_list=bbox_list, prevent_visual_duplicates=True)
        >>> aid_list2 = ibs.add_annots(gid_list, bbox_list=bbox_list, prevent_visual_duplicates=True)
        >>> aid_list3 = ibs.add_annots(gid_list, bbox_list=bbox_list2, prevent_visual_duplicates=True)
        >>> assert aid_list1 == aid_list2, 'aid_list1 == aid_list2'
        >>> assert aid_list1 != aid_list3, 'aid_list1 != aid_list3'
        >>> aid_list_new = aid_list1 + aid_list3
        >>> result = aid_list_new
        >>> print('Cleaning up. Removing added annotations')
        >>> ibs.delete_annots(aid_list_new)
        >>> print(result)
        [14, 15]
    """
    #ut.embed()
    from vtool import geometry
    if ut.VERBOSE:
        print('[ibs] adding annotations')
    # Prepare the SQL input
    assert name_list is None or nid_list is None, 'cannot specify both names and nids'
    # For import only, we can specify both by setting import_override to True
    assert bool(bbox_list is None) != bool(vert_list is None), (
        'must specify exactly one of bbox_list or vert_list')
    ut.assert_all_not_None(gid_list, 'gid_list')

    if theta_list is None:
        theta_list = [0.0 for _ in range(len(gid_list))]
    if name_list is not None:
        nid_list = ibs.add_names(name_list)
    else:
        if nid_list is None:
            nid_list = [const.UNKNOWN_NAME_ROWID for _ in range(len(gid_list))]
        name_list = ibs.get_name_texts(nid_list)

    if species_rowid_list is not None:
        assert species_list is None, 'cannot mix species_rowid and species'
        species_list = ibs.get_species_texts(species_rowid_list)
    else:
        if species_list is not None:
            species_rowid_list = ibs.add_species(species_list, skip_cleaning=skip_cleaning)
        else:
            species_rowid_list = [const.UNKNOWN_SPECIES_ROWID for _ in range(len(gid_list))]
            species_list = ibs.get_species_texts(species_rowid_list)
    if detect_confidence_list is None:
        detect_confidence_list = [0.0 for _ in range(len(gid_list))]
    if notes_list is None:
        notes_list = ['' for _ in range(len(gid_list))]

    if vert_list is None:
        vert_list = geometry.verts_list_from_bboxes_list(bbox_list)
    elif bbox_list is None:
        bbox_list = geometry.bboxes_from_vert_list(vert_list)

    len_bbox    = len(bbox_list)
    len_vert    = len(vert_list)
    len_gid     = len(gid_list)
    len_notes   = len(notes_list)
    len_theta   = len(theta_list)
    try:
        assert len_vert == len_bbox, 'bbox and verts are not of same size'
        assert len_gid  == len_bbox, 'bbox and gid are not of same size'
        assert len_gid  == len_theta, 'bbox and gid are not of same size'
        assert len_notes == len_gid, 'notes and gids are not of same size'
    except AssertionError as ex:
        ut.printex(ex, key_list=['len_vert', 'len_gid', 'len_bbox'
                                    'len_theta', 'len_notes'])
        raise

    if len(gid_list) == 0:
        # nothing is being added
        print('[ibs] WARNING: 0 annotations are beign added!')
        print(ut.dict_str(locals()))
        return []

    if yaw_list is None:
        yaw_list = [-1.0] * len(gid_list)
    nVert_list = [len(verts) for verts in vert_list]
    vertstr_list = [const.__STR__(verts) for verts in vert_list]
    xtl_list, ytl_list, width_list, height_list = list(zip(*bbox_list))
    assert len(nVert_list) == len(vertstr_list)

    # Build ~~deterministic?~~ random and unique ANNOTATION ids
    image_uuid_list = ibs.get_image_uuids(gid_list)
    if annot_uuid_list is None:
        annot_uuid_list = [uuid.uuid4() for _ in range(len(gid_list))]

    # FIXME
    # Careful this code is very fragile. It might go out of sync
    # with the updating of the determenistic uuids. Find a way to
    # integrate both pieces of code without too much reundancy.
    # Make sure these tuples are constructed correctly
    if annot_visual_uuid_list is None:
        visual_infotup = (image_uuid_list, vert_list, theta_list)
        assert len(visual_infotup) == 3, 'len=%r' % (len(visual_infotup),)
        annot_visual_uuid_list = [ut.augment_uuid(*tup) for tup in zip(*visual_infotup)]
    if annot_semantic_uuid_list is None:
        semantic_infotup = (image_uuid_list, vert_list, theta_list, yaw_list,
                            name_list, species_list)
        assert len(semantic_infotup) == 6, 'len=%r' % (len(semantic_infotup),)
        annot_semantic_uuid_list = [ut.augment_uuid(*tup) for tup in zip(*semantic_infotup)]

    # Define arguments to insert
    colnames = ('annot_uuid', 'image_rowid', 'annot_xtl', 'annot_ytl',
                'annot_width', 'annot_height', 'annot_theta', 'annot_num_verts',
                'annot_verts', ANNOT_YAW, 'annot_detect_confidence',
                'annot_note', 'name_rowid', 'species_rowid',
                'annot_visual_uuid', 'annot_semantic_uuid')

    check_uuid_flags = [not isinstance(auuid, uuid.UUID) for auuid in annot_uuid_list]
    if any(check_uuid_flags):
        pos = ut.list_where(check_uuid_flags)
        raise ValueError('positions %r have malformated UUIDS' % (pos,))

    params_iter = list(zip(annot_uuid_list, gid_list, xtl_list, ytl_list,
                            width_list, height_list, theta_list, nVert_list,
                            vertstr_list, yaw_list, detect_confidence_list,
                            notes_list, nid_list, species_rowid_list,
                           annot_visual_uuid_list, annot_semantic_uuid_list))

    # Execute add ANNOTATIONs SQL
    if prevent_visual_duplicates:
        superkey_paramx = (14,)
        get_rowid_from_superkey = ibs.get_annot_aids_from_visual_uuid
    else:
        superkey_paramx = (0,)
        get_rowid_from_superkey = ibs.get_annot_aids_from_uuid
    aid_list = ibs.db.add_cleanly(const.ANNOTATION_TABLE, colnames, params_iter,
                                  get_rowid_from_superkey, superkey_paramx)
    #ibs.update_annot_visual_uuids(aid_list)

    # Invalidate image thumbnails, quiet_delete_thumbs causes no output on deletion from ut
    config2_ = {'thumbsize': 221}
    ibs.delete_image_thumbs(gid_list, quiet=quiet_delete_thumbs, **config2_)
    return aid_list


@register_ibs_method
# @register_api('/api/annot/rows/', methods=['GET'])
def get_annot_rows(ibs, aid_list):
    r"""
    Auto-docstr for 'get_annot_rows'
    """
    colnames = ('annot_uuid', 'image_rowid', 'annot_xtl', 'annot_ytl',
                'annot_width', 'annot_height', 'annot_theta', 'annot_num_verts',
                'annot_verts', ANNOT_YAW, 'annot_detect_confidence',
                'annot_note', 'name_rowid', 'species_rowid',
                'annot_visual_uuid', 'annot_semantic_uuid')
    rows_list = ibs.db.get(const.ANNOTATION_TABLE, colnames, aid_list,
                           unpack_scalars=False)
    return rows_list


# ==========
# DELETERS
# ==========


@register_ibs_method
@accessor_decors.deleter
@register_api('/api/annot/name/rowid/', methods=['DELETE'])
def delete_annot_nids(ibs, aid_list):
    r"""
    Remove name assocation from the list of input aids.
    Does this by setting each annotations nid to the UNKNOWN name rowid

    RESTful:
        Method: DELETE
        URL:    /api/annot/name/rowid/
    """
    # FIXME: This should be implicit by setting the anotation name to the
    # unknown name
    #ibs.delete_annot_relations_oftype(aid_list, const.INDIVIDUAL_KEY)
    ibs.set_annot_name_rowids(aid_list, [const.UNKNOWN_NAME_ROWID] * len(aid_list))


@register_ibs_method
@accessor_decors.deleter
@register_api('/api/annot/species/rowid/', methods=['DELETE'], __api_plural_check__=False)
def delete_annot_speciesids(ibs, aid_list):
    r"""
    Deletes nids of a list of annotations

    RESTful:
        Method: DELETE
        URL:    /api/annot/species/rowid/
    """
    # FIXME: This should be implicit by setting the anotation name to the
    # unknown species
    #ibs.delete_annot_relations_oftype(aid_list, const.SPECIES_KEY)
    ibs.set_annot_species_rowids(aid_list, [const.UNKNOWN_SPECIES_ROWID] * len(aid_list))


@register_ibs_method
@accessor_decors.deleter
@accessor_decors.cache_invalidator(const.ANNOTATION_TABLE, rowidx=0)
@accessor_decors.cache_invalidator(const.IMAGE_TABLE, colnames=[ANNOT_ROWIDS], rowidx=None)
@accessor_decors.cache_invalidator(const.NAME_TABLE, colnames=[ANNOT_ROWIDS], rowidx=None)
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['percent_names_with_exemplar_str'])
@register_api('/api/annot/', methods=['DELETE'])
def delete_annots(ibs, aid_list):
    r"""
    deletes annotations from the database

    RESTful:
        Method: DELETE
        URL:    /api/annot/

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-delete_annots
        python -m ibeis.control.manual_annot_funcs --test-delete_annots --debug-api-cache
        python -m ibeis.control.manual_annot_funcs --test-delete_annots

    SeeAlso:
        back.delete_annot

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> from os.path import exists
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> ibs.delete_empty_nids()
        >>> # Add some annotations to delete
        >>> num_add = 2
        >>> gid_list = ibs.get_valid_gids()[0:num_add]
        >>> nid = ibs.make_next_nids(1)[0]
        >>> nid_list = [nid] * num_add
        >>> bbox_list = [(int(w * .1), int(h * .6), int(w * .5), int(h *  .3))
        ...              for (w, h) in ibs.get_image_sizes(gid_list)]
        >>> new_aid_list = ibs.add_annots(gid_list, bbox_list=bbox_list,
        >>>                               nid_list=nid_list)
        >>> ibs.get_annot_nids(new_aid_list)
        >>> ut.assert_lists_eq(ibs.get_annot_nids(new_aid_list), nid_list)
        >>> assert ibs.get_name_aids(nid) == new_aid_list, 'annots should all have same name'
        >>> assert new_aid_list == ibs.get_name_aids(nid), 'inverse name mapping should work'
        >>> #thumpaths = ibs.get_image_thumbpath(gid_list, ensure_paths=True, **{'thumbsize': 221})
        >>> #assert any(ut.lmap(exists, thumpaths)), 'thumbs should be there'
        >>> before_aids = ibs.get_image_aids(gid_list)
        >>> print('BEFORE gids: ' + str(before_aids))
        >>> result = ibs.delete_annots(new_aid_list)
        >>> assert ibs.get_name_aids(nid) == [], 'annots should be removed'
        >>> after_aids = ibs.get_image_aids(gid_list)
        >>> #thumpaths = ibs.get_image_thumbpath(gid_list, ensure_paths=False, **{'thumbsize': 221})
        >>> #assert not any(ut.lmap(exists, thumpaths)), 'thumbs should be gone'
        >>> assert after_aids != before_aids, 'the invalidators must have bugs'
        >>> print('AFTER gids: ' + str(after_aids))
        >>> valid_aids = ibs.get_valid_aids()
        >>> assert  [aid not in valid_aids for aid in new_aid_list], 'should no longer be valid aids'
        >>> print(result)
        >>> ibs.delete_empty_nids()

    """
    if ut.VERBOSE:
        print('[ibs] deleting %d annotations' % len(aid_list))
    # FIXME: Need to reliabely delete thumbnails
    # config2_ = {'draw_annots': True, 'thumbsize': 221}
    # MEGA HACK FOR QT
    ibs.delete_annot_imgthumbs(aid_list)
    # Delete chips and features first
    #ibs.delete_annot_relations(aid_list)
    ibs.delete_annot_chips(aid_list)
    ibs.depc_annot.delete_root(aid_list)
    # TODO:
    # delete parent rowid column if exists in annot table
    return ibs.db.delete_rowids(const.ANNOTATION_TABLE, aid_list)


@register_ibs_method
@accessor_decors.deleter
def delete_annot_imgthumbs(ibs, aid_list):
    # Less hacky
    gid_list_ = ibs.get_annot_gids(aid_list)
    table_config_filter = {
        'thumbnails': {
            'draw_annots': True,
        }
    }
    ibs.depc_image.delete_root(gid_list_, table_config_filter)

    # MEGA HACK FOR QT
    config2_ = {'thumbsize': 221}
    ibs.delete_image_thumbs(gid_list_, **config2_)
    # ibs.delete_image_thumbs(gid_list_)


# ==========
# GETTERS
# ==========


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annot/aids/uuid/semantic/', methods=['GET'])
def get_annot_aids_from_semantic_uuid(ibs, semantic_uuid_list):
    r"""
    Args:
        semantic_uuid_list (list):

    Returns:
        list: annot rowids
    """
    aids_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_ROWID,),
                           semantic_uuid_list, id_colname=ANNOT_SEMANTIC_UUID)
    return aids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/rowid/uuid/', methods=['GET'])
def get_annot_aids_from_uuid(ibs, uuid_list):
    r"""
    Returns:
        list_ (list): annot rowids

    RESTful:
        Method: GET
        URL:    /api/annot/rowid/uuid/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    aids_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_ROWID,), uuid_list,
                           id_colname=ANNOT_UUID)
    return aids_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annot/uuid/missing/', methods=['GET'])
def get_annot_missing_uuid(ibs, uuid_list):
    r"""
    Returns:
        list_ (list): a list of missing annot uuids
    """
    aid_list = ibs.get_annot_aids_from_uuid(uuid_list)
    zipped = zip(aid_list, uuid_list)
    missing_uuid_list = [ uuid for aid, uuid in zipped if aid is None ]
    return missing_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annot/aids/uuid/visual/', methods=['GET'])
def get_annot_aids_from_visual_uuid(ibs, visual_uuid_list):
    r"""
    Args:
        visual_uuid_list (list):

    Returns:
        list: annot rowids
    """
    aids_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_ROWID,),
                           visual_uuid_list, id_colname=ANNOT_VISUAL_UUID)
    return aids_list

get_annot_rowids_from_visual_uuid = get_annot_aids_from_visual_uuid


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1toM
@register_api('/api/annot/bbox/', methods=['GET'])
def get_annot_bboxes(ibs, aid_list):
    r"""
    Returns:
        bbox_list (list):  annotation bounding boxes in image space

    RESTful:
        Method: GET
        URL:    /api/annot/bbox/
    """
    colnames = ('annot_xtl', 'annot_ytl', 'annot_width', 'annot_height',)
    bbox_list = ibs.db.get(const.ANNOTATION_TABLE, colnames, aid_list)
    return bbox_list


@register_ibs_method
# @register_api('/api/annot/labels/', methods=['GET'])
def get_annot_class_labels(ibs, aid_list):
    r"""
    DEPRICATE?

    Returns:
        list of tuples: identifying animal name and view
    """
    name_list = ibs.get_annot_name_rowids(aid_list)
    # TODO: use yaw?
    yaw_list = [0 for _ in name_list]
    classlabel_list = list(zip(name_list, yaw_list))
    return classlabel_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/detect/confidence/', methods=['GET'])
def get_annot_detect_confidence(ibs, aid_list):
    r"""
    Returns:
        list_ (list): a list confidences that the annotations is a valid detection

    RESTful:
        Method: GET
        URL:    /api/annot/detect/confidence/
    """
    annot_detect_confidence_list = ibs.db.get(const.ANNOTATION_TABLE,
                                              ('annot_detect_confidence',),
                                              aid_list)
    return annot_detect_confidence_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/exemplar/', methods=['GET'])
def get_annot_exemplar_flags(ibs, aid_list):
    r"""
    returns if an annotation is an exemplar

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: annot_exemplar_flag_list - True if annotation is an exemplar

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_exemplar_flags

    RESTful:
        Method: GET
        URL:    /api/annot/exemplar/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> gid_list = get_annot_exemplar_flags(ibs, aid_list)
        >>> result = str(gid_list)
        >>> print(result)
    """
    annot_exemplar_flag_list = ibs.db.get(const.ANNOTATION_TABLE,
                                          ('annot_exemplar_flag',), aid_list)
    return annot_exemplar_flag_list


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1to1
#@cache_getter(const.ANNOTATION_TABLE, 'image_rowid')
@register_api('/api/annot/image/rowid/', methods=['GET'])
def get_annot_gids(ibs, aid_list):
    r"""
    Get parent image rowids of annotations

    Args:
        aid_list (list):

    Returns:
        gid_list (list):  image rowids

    RESTful:
        Method: GET
        URL:    /api/annot/image/rowid/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> result = get_annot_gids(ibs, aid_list)
        >>> print(result)
    """
    gid_list = ibs.db.get(const.ANNOTATION_TABLE, ('image_rowid',), aid_list)
    return gid_list


@register_ibs_method
def get_annot_image_rowids(ibs, aid_list):
    return ibs.get_annot_gids(aid_list)


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1to1
#@cache_getter(const.ANNOTATION_TABLE, 'image_rowid')
@register_api('/api/annot/imageset/rowid/', methods=['GET'])
def get_annot_imgsetids(ibs, aid_list):
    r"""
    Get parent image rowids of annotations

    Args:
        aid_list (list):

    Returns:
        imgsetid_list (list):  imageset rowids

    RESTful:
        Method: GET
        URL:    /api/annot/imageset/rowid/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> result = get_annot_gids(ibs, aid_list)
        >>> print(result)
    """
    gid_list = ibs.db.get(const.ANNOTATION_TABLE, ('image_rowid',), aid_list)
    imgsetids_list = ibs.get_image_imgsetids(gid_list)
    return imgsetids_list


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1to1
#@cache_getter(const.ANNOTATION_TABLE, 'image_rowid')
@register_api('/api/annot/imageset/uuid/', methods=['GET'])
def get_annot_imgset_uuids(ibs, aid_list):
    r"""
    Get parent image rowids of annotations

    Args:
        aid_list (list):

    Returns:
        imgset_uuid_list (list):  imageset uuids

    RESTful:
        Method: GET
        URL:    /api/annot/imageset/uuid/

    """
    imgsetids_list = ibs.get_annot_imgsetids(aid_list)
    imgset_uuid_list = ibs.get_imageset_uuid(imgsetids_list)
    return imgset_uuid_list


@register_ibs_method
@accessor_decors.getter_1toM
# @register_api('/api/annot/gar/rowid/', methods=['GET'])
def get_annot_gar_rowids(ibs, aid_list):
    r"""
    Auto-docstr for 'get_annot_gar_rowids'
    """
    colnames = (GAR_ROWID,)
    gar_rowid_list = ibs.db.get(const.GA_RELATION_TABLE, colnames, aid_list,
                                id_colname=ANNOT_ROWID, unpack_scalars=False)
    return gar_rowid_list


@register_ibs_method
@accessor_decors.getter_1toM
# @register_api('/api/annot/aids/otherimage/', methods=['GET'])
def get_annot_otherimage_aids(ibs, aid_list, daid_list=None):
    r"""
    Auto-docstr for 'get_annot_otherimage_aids'

    """
    gid_list = ibs.get_annot_gids(aid_list)
    if daid_list is None:
        image_aids_list = ibs.get_image_aids(gid_list)
        # Remove self from list
        other_aids_list = [list(set(aids) - {aid})
                           for aids, aid in zip(image_aids_list, aid_list)]
    else:
        daids = np.array(daid_list)
        internal_data_gids  = ibs.get_annot_gids(daids)
        other_aids_list = [
            daids.compress(internal_data_gids == gid)
            for gid in gid_list
        ]
    return other_aids_list


@register_ibs_method
@accessor_decors.getter_1toM
# @register_api('/api/annot/aids/contact/', methods=['GET'])
def get_annot_contact_aids(ibs, aid_list, daid_list=None, check_isect=False):
    r"""
    Returns the other aids that appear in the same image that this
    annotation is from.

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_contact_aids;1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> contact_aids = ibs.get_annot_contact_aids(aid_list)
        >>> contact_gids = ibs.unflat_map(ibs.get_annot_gids, contact_aids)
        >>> gid_list = ibs.get_annot_gids(aid_list)
        >>> for gids, gid, aids, aid in zip(contact_gids, gid_list, contact_aids, aid_list):
        ...     assert ut.allsame(gids), 'annots should be from same image'
        ...     assert len(gids) == 0 or gids[0] == gid, 'and same image as parent annot'
        ...     assert aid not in aids, 'should not include self'

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb2')
        >>> aid_list = ibs.get_valid_aids()
        >>> contact_aids = ibs.get_annot_contact_aids(aid_list)
        >>> contact_gids = ibs.unflat_map(ibs.get_annot_gids, contact_aids)
        >>> gid_list = ibs.get_annot_gids(aid_list)
        >>> print('contact_aids = %r' % (contact_aids,))
        >>> for gids, gid, aids, aid in zip(contact_gids, gid_list, contact_aids, aid_list):
        ...     assert ut.allsame(gids), 'annots should be from same image'
        ...     assert len(gids) == 0 or gids[0] == gid, 'and same image as parent annot'
        ...     assert aid not in aids, 'should not include self'
    """
    other_aids_list = ibs.get_annot_otherimage_aids(aid_list, daid_list=daid_list)
    if check_isect:
        import shapely.geometry
        # TODO: might not be accounting for rotated verticies
        verts_list = ibs.get_annot_verts(aid_list)
        other_verts_list = ibs.unflat_map(ibs.get_annot_verts, other_aids_list)
        poly_list = [shapely.geometry.Polygon(verts) for verts in verts_list]
        other_polys_list = [[shapely.geometry.Polygon(verts) for verts in _]
                            for _ in other_verts_list]
        flags_list = [[p1.intersects(p2) for p2 in p2_list]
                      for p1, p2_list in zip(poly_list, other_polys_list)]
        contact_aids = [ut.compress(other_aids, flags)
                        for other_aids, flags in zip(other_aids_list, flags_list)]
    else:
        contact_aids = other_aids_list
    return contact_aids


#@register_ibs_method
#@accessor_decors.getter_1toM
#def get_annot_intersecting_aids(ibs, aid_list):
#    pass


@register_ibs_method
@accessor_decors.getter_1to1
def get_annot_num_contact_aids(ibs, aid_list):
    r"""
    Auto-docstr for 'get_annot_num_contact_aids'
    """
    nOther_aids_list = list(map(len, ibs.get_annot_contact_aids(aid_list)))
    return nOther_aids_list


@register_ibs_method
@accessor_decors.getter_1toM
def get_annot_groundfalse(ibs, aid_list, is_exemplar=None, valid_aids=None,
                          filter_unknowns=True, daid_list=None):
    r"""
    gets all annotations with different names

    Returns:
        groundfalse_list (list): a list of aids which are known to be different for each

    #Example:
    #    >>> # ENABLE_DOCTEST
    #    >>> import ibeis
    #    >>> ibs = ibeis.opendb('PZ_MTEST')
    #    >>> aid_list = ibs.get_valid_aids()
    #    >>> is_exemplar, noself, daid_list = None, True, None
    #    >>> groundfalse_list = ibs.get_annot_groundfalse(aid_list)
    #    >>> result = str(groundtruth_list)
    #    >>> print(result)
    """
    if valid_aids is None:
        # get all valid aids if not specified
        # really the examplar flag should not be allowed and only daids
        # should be taken as input
        valid_aids = ibs.get_valid_aids(is_exemplar=is_exemplar)
    if daid_list is not None:
        valid_aids = list(set(daid_list).intersection(set(valid_aids)))
    if filter_unknowns:
        # Remove aids which do not have a name
        isunknown_list = ibs.is_aid_unknown(valid_aids)
        valid_aids_ = ut.filterfalse_items(valid_aids, isunknown_list)
    else:
        valid_aids_ = valid_aids
    # Build the set of groundfalse annotations
    nid_list = ibs.get_annot_name_rowids(aid_list)
    aids_list = ibs.get_name_aids(nid_list, enable_unknown_fix=True)
    aids_setlist = map(set, aids_list)
    valid_aids = set(valid_aids_)
    groundfalse_list = [list(valid_aids - aids) for aids in aids_setlist]
    return groundfalse_list


@register_ibs_method
@accessor_decors.getter_1toM
# @register_api('/api/annot/groundtruth/', methods=['GET'])
def get_annot_groundtruth(ibs, aid_list, is_exemplar=None, noself=True,
                          daid_list=None):
    r"""
    gets all annotations with the same names

    Args:
        aid_list    (list): list of annotation rowids to get groundtruth of
        is_exemplar (None):
        noself      (bool):
        daid_list   (list):

    Returns:
        groundtruth_list (list): a list of aids with the same name foreach
        aid in aid_list.  a set of aids belonging to the same name is called
        a groundtruth.  A list of these is called a groundtruth_list.

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_groundtruth:0
        python -m ibeis.control.manual_annot_funcs --test-get_annot_groundtruth:1
        python -m ibeis.control.manual_annot_funcs --test-get_annot_groundtruth:2
        python -m --tf get_annot_groundtruth:0 --db=PZ_Master0 --aids=97 --exec-mode

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ut.get_argval('--aids', list, ibs.get_valid_aids())
        >>> is_exemplar, noself, daid_list = None, True, None
        >>> groundtruth_list = ibs.get_annot_groundtruth(aid_list, is_exemplar, noself, daid_list)
        >>> result = 'groundtruth_list = ' + str(groundtruth_list)
        >>> print(result)
        groundtruth_list = [[], [3], [2], [], [6], [5], [], [], [], [], [], [], []]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> is_exemplar, noself, daid_list = True, True, None
        >>> groundtruth_list = ibs.get_annot_groundtruth(aid_list, is_exemplar, noself, daid_list)
        >>> result = str(groundtruth_list)
        >>> print(result)
        [[], [3], [2], [], [6], [5], [], [], [], [], [], [], []]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> is_exemplar, noself, daid_list = False, False, aid_list
        >>> groundtruth_list = ibs.get_annot_groundtruth(aid_list, is_exemplar, noself, daid_list)
        >>> result = str(groundtruth_list)
        >>> print(result)
        [[1], [], [], [4], [], [], [], [], [9], [], [11], [], []]

    """
    # TODO: Optimize
    nid_list = ibs.get_annot_name_rowids(aid_list)
    #if daid_list is not None:
    #    # when given a valid pool try to skip the get_name_aids call
    #    aids_list_, nid_list_ = ibs.group_annots_by_name(daid_list, distinguish_unknowns=True)
    #    nid2_aids = dict(zip(nid_list_, aids_list_))
    #    aids_list = ut.dict_take(nid2_aids, nid_list, [])
    #else:
    aids_list = ibs.get_name_aids(nid_list, enable_unknown_fix=True)
    if is_exemplar is None:
        groundtruth_list_ = aids_list
    else:
        # Filter out non-exemplars
        exemplar_flags_list = ibsfuncs.unflat_map(ibs.get_annot_exemplar_flags, aids_list)
        isvalids_list = [[flag == is_exemplar for flag in flags] for flags in exemplar_flags_list]
        groundtruth_list_ = [ut.compress(aids, isvalids)
                             for aids, isvalids in zip(aids_list, isvalids_list)]
    if noself:
        # Remove yourself from the set
        groundtruth_list = [list(set(aids) - {aid})
                            for aids, aid in zip(groundtruth_list_, aid_list)]
    else:
        groundtruth_list = groundtruth_list_

    if daid_list is not None:
        # filter out any groundtruth that isn't allowed
        daid_set = set(daid_list)
        groundtruth_list = [list(daid_set.intersection(set(aids))) for aids in groundtruth_list]
    return groundtruth_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annot/groundtruth/check/', methods=['GET'])
def get_annot_has_groundtruth(ibs, aid_list, is_exemplar=None, noself=True, daid_list=None):
    r"""
    Args:
        aid_list    (list):
        is_exemplar (None):
        noself      (bool):
        daid_list   (list):

    Returns:
        list: has_gt_list

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_has_groundtruth

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> is_exemplar = None
        >>> noself = True
        >>> daid_list = None
        >>> has_gt_list = get_annot_has_groundtruth(ibs, aid_list, is_exemplar, noself, daid_list)
        >>> result = str(has_gt_list)
        >>> print(result)
    """
    # TODO: Optimize
    numgts_list = ibs.get_annot_num_groundtruth(aid_list, is_exemplar=is_exemplar,
                                                noself=noself,
                                                daid_list=daid_list)
    has_gt_list = [num_gts > 0 for num_gts in numgts_list]
    return has_gt_list


#@register_ibs_method
#def get_annot_hashid_rowid(ibs, aid_list, prefix=''):
#    raise AssertionError('You probably want to call a uuid hash id method')
#    label = ''.join(('_', prefix, 'UUIDS'))
#    aids_hashid = ut.hashstr_arr(aid_list, label)
#    return aids_hashid


@register_ibs_method
@register_api('/api/annot/uuid/hashid/', methods=['GET'])
def get_annot_hashid_uuid(ibs, aid_list, prefix=''):
    r"""
    builds an aggregate random hash id for a list of aids

    RESTful:
        Method: GET
        URL:    /api/annot/uuid/hashid/
    """
    uuid_list    = ibs.get_annot_uuids(aid_list)
    label = ''.join(('_', prefix, 'UUIDS'))
    uuid_hashid  = ut.hashstr_arr(uuid_list, label)
    return uuid_hashid


@register_ibs_method
# @register_api('/api/annot/uuid/visual/hashid/', methods=['GET'])
def get_annot_hashid_visual_uuid(ibs, aid_list, prefix='', pathsafe=False):
    r"""
    builds an aggregate visual hash id for a list of aids

    Args:
        _new (bool): Eventually we will change the hashing scheme and all old
            data will be invalidated. (default=False)
    """
    visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
    label = ''.join(('_', prefix, 'VUUIDS'))
    #if _new:
    visual_uuid_hashid  = ut.hashstr_arr27(visual_uuid_list, label, pathsafe=pathsafe)
    #else:
    #visual_uuid_hashid  = ut.hashstr_arr(visual_uuid_list, label)
    return visual_uuid_hashid


@register_ibs_method
# @register_api('/api/annot/uuid/semantic/hashid/', methods=['GET'])
def get_annot_hashid_semantic_uuid(ibs, aid_list, prefix='', pathsafe=False):
    r"""
    builds an aggregate semantic hash id for a list of aids

    Args:
        ibs (ibeis.IBEISController):  ibeis controller object
        aid_list (list):  list of annotation rowids
        prefix (str): (default = '')
        _new (bool): Eventually we will change the hashing scheme and all old
            data will be invalidated. (default=False)

    Returns:
        str: semantic_uuid_hashid

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_hashid_semantic_uuid

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:2]
        >>> prefix = ''
        >>> semantic_uuid_hashid = get_annot_hashid_semantic_uuid(ibs, aid_list, prefix)
        >>> result = ut.list_str(ibs.get_annot_semantic_uuids(aid_list)) + '\n'
        >>> result += ('semantic_uuid_hashid = %s' % (str(semantic_uuid_hashid),))
        >>> print(result)
        [
            UUID('0a41bee7-1760-0995-c220-f4b5ddf189ba'),
            UUID('8150d84c-2ec3-c9ff-3e0f-30a8274a2392'),
        ]
        semantic_uuid_hashid = _SUUIDS((2)gqxovcbwzimfzovy)
    """
    semantic_uuid_list = ibs.get_annot_semantic_uuids(aid_list)
    label = ''.join(('_', prefix, 'SUUIDS'))
    #if _new:
    semantic_uuid_hashid  = ut.hashstr_arr27(semantic_uuid_list, label, pathsafe=pathsafe)
    #else:
    #semantic_uuid_hashid  = ut.hashstr_arr(semantic_uuid_list, label)
    return semantic_uuid_hashid


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/theta/', methods=['GET'])
def get_annot_thetas(ibs, aid_list):
    r"""
    Returns:
        theta_list (list): a list of floats describing the angles of each chip

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_thetas

    RESTful:
        Method: GET
        URL:    /api/annot/theta/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('NAUT_test')
        >>> aid_list = ibs.get_valid_aids()
        >>> result = get_annot_thetas(ibs, aid_list)
        >>> print(result)
        [2.75742, 0.792917, 2.53605, 2.67795, 0.946773, 2.56729]
    """
    theta_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_theta',), aid_list)
    return theta_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/uuid/', methods=['GET'])
def get_annot_uuids(ibs, aid_list):
    r"""
    Returns:
        list: annot_uuid_list a list of image uuids by aid

    RESTful:
        Method: GET
        URL:    /api/annot/uuid/
    """
    annot_uuid_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_uuid',), aid_list)
    return annot_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annot/uuid/valid/', methods=['GET'])
def get_valid_annot_uuids(ibs):
    r"""
    Returns:
        list: annot_uuid_list a list of image uuids for all valid aids
    """
    aid_list = ibs.get_valid_aids()
    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    return annot_uuid_list


# It is a good idea to have the cache on for the annot uuids, they are quite slow to load
@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.ANNOTATION_TABLE, ANNOT_SEMANTIC_UUID)
# @register_api('/api/annot/uuid/semantic/', methods=['GET'])
def get_annot_semantic_uuids(ibs, aid_list):
    r"""
    annot_semantic_uuid_list <- annot.annot_semantic_uuid[aid_list]

    gets data from the "native" column "annot_semantic_uuid" in the "annot" table

    Args:
        aid_list (list):

    Returns:
        list: annot_semantic_uuid_list

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_semantic_uuids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> ibs, qreq_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()[0:1]
        >>> annot_semantic_uuid_list = ibs.get_annot_semantic_uuids(aid_list)
        >>> assert len(aid_list) == len(annot_semantic_uuid_list)
        >>> result = annot_semantic_uuid_list
        [UUID('0a41bee7-1760-0995-c220-f4b5ddf189ba')]
    """
    id_iter = aid_list
    colnames = (ANNOT_SEMANTIC_UUID,)
    annot_semantic_uuid_list = ibs.db.get(
        const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid')
    return annot_semantic_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.ANNOTATION_TABLE, ANNOT_VISUAL_UUID)
# @register_api('/api/annot/uuid/visual/', methods=['GET'])
def get_annot_visual_uuids(ibs, aid_list):
    r"""
    The image uuid, annotation verticies, are theta is hashted together to
    compute the visual uuid.  The visual uuid does not include name or species
    information.

    annot_visual_uuid_list <- annot.annot_visual_uuid[aid_list]

    gets data from the "native" column "annot_visual_uuid" in the "annot" table

    Args:
        aid_list (list):

    Returns:
        list: annot_visual_uuid_list

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_visual_uuids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> ibs, qreq_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()[0:1]
        >>> annot_visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
        >>> assert len(aid_list) == len(annot_visual_uuid_list)
        >>> result = annot_visual_uuid_list
        [UUID('8687dcb6-1f1f-fdd3-8b72-8f36f9f41905')]

        [UUID('76de0416-7c92-e1b3-4a17-25df32e9c2b4')]
    """
    id_iter = aid_list
    colnames = (ANNOT_VISUAL_UUID,)
    annot_visual_uuid_list = ibs.db.get(
        const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid')
    return annot_visual_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/vert/', methods=['GET'])
def get_annot_verts(ibs, aid_list):
    r"""
    Returns:
        vert_list (list): the vertices that form the polygon of each chip

    RESTful:
        Method: GET
        URL:    /api/annot/vert/
    """
    from ibeis.algo.preproc import preproc_annot
    vertstr_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_verts',), aid_list)
    vert_list = preproc_annot.postget_annot_verts(vertstr_list)
    #vert_list = [eval(vertstr, {}, {}) for vertstr in vertstr_list]
    return vert_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/vert/rotated/', methods=['GET'])
def get_annot_rotated_verts(ibs, aid_list):
    r"""
    Returns:
        rotated_vert_list (list): verticies after rotation by theta.

    RESTful:
        Method: GET
        URL:    /api/annot/vert/rotated/
    """
    import vtool as vt
    vert_list = ibs.get_annot_verts(aid_list)
    theta_list = ibs.get_annot_thetas(aid_list)
    # Convex bounding boxes for verticies
    bbox_list = vt.geometry.bboxes_from_vert_list(vert_list)
    rot_list = [vt.rotation_around_bbox_mat3x3(theta, bbox)
                for theta, bbox in zip(theta_list, bbox_list)]
    rotated_vert_list = [vt.transform_points_with_homography(rot, np.array(verts).T).T.tolist()
                         for rot, verts in zip(rot_list, vert_list)]
    #vert_list = [eval(vertstr, {}, {}) for vertstr in vertstr_list]
    return rotated_vert_list


@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.ANNOTATION_TABLE, ANNOT_YAW)
@register_api('/api/annot/yaw/', methods=['GET'])
def get_annot_yaws(ibs, aid_list):
    r"""
    A yaw is the yaw of the annotation in radians
    Viewpoint yaw is inverted. Will be fixed soon.

    The following views have these angles of yaw:
        left side  - 0.50 tau radians
        front side - 0.25 tau radians
        right side - 0.00 tau radians
        back side  - 0.75 tau radians

        tau = 2 * pi

    SeeAlso:
        ibies.const.VIEWTEXT_TO_YAW_RADIANS

    Returns:
        yaw_list (list): the yaw (in radians) for the annotation

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_yaws

    RESTful:
        Method: GET
        URL:    /api/annot/yaw/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[::3]
        >>> result = get_annot_yaws(ibs, aid_list)
        >>> print(result)
        [3.141592653589793, 3.141592653589793, None, 3.141592653589793, None]
    """
    yaw_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_YAW,), aid_list)
    yaw_list = [yaw if yaw is not None and yaw >= 0.0 else None for yaw in yaw_list]
    return yaw_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_annot_yaws_asfloat(ibs, aid_list):
    r"""
    Ensures that Nones are returned as nans
    """
    yaw_list = ibs.get_annot_yaws(aid_list)
    yaw_list = np.array(ut.replace_nones(yaw_list, np.nan))
    return yaw_list


@register_ibs_method
@accessor_decors.setter
@accessor_decors.cache_invalidator(const.ANNOTATION_TABLE, [ANNOT_YAW], rowidx=0)
@register_api('/api/annot/yaw/', methods=['PUT'])
def set_annot_yaws(ibs, aid_list, yaw_list, input_is_degrees=False):
    r"""
    Sets the  yaw of a list of chips by aid

    A yaw is the yaw of the annotation in radians
    Viewpoint yaw is inverted. Will be fixed soon.

    Note:
        The following views have these angles of yaw:
            left side  - 0.00 tau radians
            front side - 0.25 tau radians
            right side - 0.50 tau radians
            back side  - 0.75 tau radians
            (tau = 2 * pi)

    SeeAlso:
        ibies.const.VIEWTEXT_TO_YAW_RADIANS

    References:
        http://upload.wikimedia.org/wikipedia/commons/7/7e/Rollpitchyawplain.png

    RESTful:
        Method: PUT
        URL:    /api/annot/yaw/
    """
    id_iter = ((aid,) for aid in aid_list)
    #yaw_list = [-1 if yaw is None else yaw for yaw in yaw_list]
    if input_is_degrees:
        yaw_list = [-1 if yaw is None else ut.deg_to_rad(yaw) for yaw in yaw_list]
    #assert all([0.0 <= yaw < 2 * np.pi or yaw == -1.0 for yaw in yaw_list])
    val_iter = ((yaw, ) for yaw in yaw_list)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_YAW,), val_iter, id_iter)
    ibs.update_annot_visual_uuids(aid_list)
    ibs.depc_annot.notify_root_changed(aid_list, 'yaws')


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/note/', methods=['GET'])
def get_annot_notes(ibs, aid_list):
    r"""
    Returns:
        annotation_notes_list (list): a list of annotation notes

    RESTful:
        Method: GET
        URL:    /api/annot/note/
    """
    annotation_notes_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_NOTE,), aid_list)
    return annotation_notes_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_annot_num_groundtruth(ibs, aid_list, is_exemplar=None, noself=True,
                              daid_list=None):
    r"""
    Returns:
        list_ (list): number of other chips with the same name

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_num_groundtruth
        python -m ibeis.control.manual_annot_funcs --test-get_annot_num_groundtruth:0
        python -m ibeis.control.manual_annot_funcs --test-get_annot_num_groundtruth:1

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> noself = True
        >>> result = get_annot_num_groundtruth(ibs, aid_list, noself=noself)
        >>> print(result)
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> noself = False
        >>> result = get_annot_num_groundtruth(ibs, aid_list, noself=noself)
        >>> print(result)
        [1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1]
    """
    # TODO: Optimize
    groundtruth_list = ibs.get_annot_groundtruth(aid_list,
                                                 is_exemplar=is_exemplar,
                                                 noself=noself,
                                                 daid_list=daid_list)
    nGt_list = list(map(len, groundtruth_list))
    return nGt_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/num/vert/', methods=['GET'])
def get_annot_num_verts(ibs, aid_list):
    r"""
    Returns:
        nVerts_list (list): the number of vertices that form the polygon of each chip

    RESTful:
        Method: GET
        URL:    /api/annot/num/vert/
    """
    nVerts_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_NUM_VERTS,), aid_list)
    return nVerts_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annot/parent/aid/', methods=['GET'])
def get_annot_parent_aid(ibs, aid_list):
    r"""
    Returns:
        list_ (list): a list of parent (in terms of parts) annotation rowids.
    """
    annot_parent_rowid_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_PARENT_ROWID,), aid_list)
    return annot_parent_rowid_list


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.ANNOTATION_TABLE, NAME_ROWID, cfgkeys=['distinguish_unknowns'])
# @register_api('/api/annot/name/rowid/', methods=['GET'])
def get_annot_name_rowids(ibs, aid_list, distinguish_unknowns=True):
    r"""
    Returns:
        list_ (list): the name id of each annotation.

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --exec-get_annot_name_rowids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> from ibeis import constants as const
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> distinguish_unknowns = True
        >>> nid_arr1 = np.array(ibs.get_annot_name_rowids(aid_list, distinguish_unknowns=distinguish_unknowns))
        >>> nid_arr2 = np.array(ibs.get_annot_name_rowids(aid_list, distinguish_unknowns=False))
        >>> nid_arr2 = np.array(ibs.get_annot_name_rowids(None, distinguish_unknowns=True))
        >>> assert const.UNKNOWN_LBLANNOT_ROWID == 0
        >>> assert np.all(nid_arr1[np.where(const.UNKNOWN_LBLANNOT_ROWID == nid_arr2)[0]] < 0)
    """
    id_iter = aid_list
    colnames = (NAME_ROWID,)
    nid_list_ = ibs.db.get(const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid')
    if distinguish_unknowns:
        nid_list = [(None if aid is None else -aid)
                    if nid == const.UNKNOWN_LBLANNOT_ROWID or nid is None else nid
                    for nid, aid in zip(nid_list_, aid_list)]
    else:
        nid_list = [const.UNKNOWN_LBLANNOT_ROWID if nid is None else nid for nid in nid_list_]
    return nid_list


@register_ibs_method
@register_api('/api/annot/name/rowid/', methods=['GET'])
def get_annot_nids(ibs, aid_list, distinguish_unknowns=True):
    r"""
    alias

    RESTful:
        Method: GET
        URL:    /api/annot/name/rowid/
    """
    return ibs.get_annot_name_rowids(aid_list, distinguish_unknowns=distinguish_unknowns)


@register_ibs_method
@register_api('/api/annot/name/uuid/', methods=['GET'])
def get_annot_name_uuids(ibs, aid_list, **kwargs):
    r"""
    alias

    RESTful:
        Method: GET
        URL:    /api/annot/name/uuid/
    """
    nid_list = ibs.get_annot_name_rowids(aid_list, **kwargs)
    name_uuid_list = ibs.get_name_uuids(nid_list)
    return name_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annot/name/', methods=['GET'])
def get_annot_names(ibs, aid_list):
    r"""
    alias
    """
    return ibs.get_annot_name_texts(aid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/name/text/', methods=['GET'])
def get_annot_name_texts(ibs, aid_list):
    r"""
    Args:
        aid_list (list):

    Returns:
        list or strs: name_list. e.g: ['fred', 'sue', ...]
             for each annotation identifying the individual

    RESTful:
        Method: GET
        URL:    /api/annot/name/text/

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_name_texts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[::2]
        >>> result = ut.list_str(get_annot_name_texts(ibs, aid_list), nl=False)
        >>> print(result)
        ['____', 'easy', 'hard', 'jeff', '____', '____', 'zebra']
    """
    nid_list = ibs.get_annot_name_rowids(aid_list)
    name_list = ibs.get_name_texts(nid_list)
    #name_list = ibs.get_annot_lblannot_value_of_lbltype(aid_list,
    #const.INDIVIDUAL_KEY, ibs.get_name_texts)
    return name_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/species/', methods=['GET'], __api_plural_check__=False)
def get_annot_species(ibs, aid_list):
    r"""
    alias

    RESTful:
        Method: GET
        URL:    /api/annot/species/
    """
    return ibs.get_annot_species_texts(aid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/species/text/', methods=['GET'], __api_plural_check__=False)
def get_annot_species_texts(ibs, aid_list):
    r"""

    Args:
        aid_list (list):

    Returns:
        list : species_list - a list of strings ['plains_zebra',
        'grevys_zebra', ...] for each annotation
        identifying the species

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_species_texts

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[1::3]
        >>> result = ut.list_str(get_annot_species_texts(ibs, aid_list), nl=False)
        >>> print(result)
        ['zebra_plains', 'zebra_plains', '____', 'bear_polar']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> species_list = get_annot_species_texts(ibs, aid_list)
        >>> result = ut.list_str(list(set(species_list)), nl=False)
        >>> print(result)
        ['zebra_plains']

    RESTful:
        Method: GET
        URL:    /api/annot/species/text/
    """
    species_rowid_list = ibs.get_annot_species_rowids(aid_list)
    speceis_text_list  = ibs.get_species_texts(species_rowid_list)
    #speceis_text_list = ibs.get_annot_lblannot_value_of_lbltype(
    #    aid_list, const.SPECIES_KEY, ibs.get_species)
    return speceis_text_list


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.ANNOTATION_TABLE, SPECIES_ROWID)
@register_api('/api/annot/species/rowid/', methods=['GET'], __api_plural_check__=False)
def get_annot_species_rowids(ibs, aid_list):
    r"""
    species_rowid_list <- annot.species_rowid[aid_list]

    gets data from the "native" column "species_rowid" in the "annot" table

    Args:
        aid_list (list):

    Returns:
        list: species_rowid_list

    RESTful:
        Method: GET
        URL:    /api/annot/species/rowid/
    """

    id_iter = aid_list
    colnames = (SPECIES_ROWID,)
    species_rowid_list = ibs.db.get(
        const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid')
    return species_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/species/uuid/', methods=['GET'], __api_plural_check__=False)
def get_annot_species_uuids(ibs, aid_list):
    r"""
    species_rowid_list <- annot.species_rowid[aid_list]

    Args:
        aid_list (list):

    Returns:
        list: species_uuid_list

    RESTful:
        Method: GET
        URL:    /api/annot/species/uuid/
    """
    species_rowid_list = ibs.get_annot_species_rowids(aid_list)
    species_uuid_list = ibs.get_species_uuids(species_rowid_list)
    return species_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/image/name/', methods=['GET'])
def get_annot_image_names(ibs, aid_list):
    r"""
    Args:
        aid_list (list):

    Returns:
        list of strs: gname_list the image names of each annotation

    RESTful:
        Method: GET
        URL:    /api/annot/image/name/
    """
    gid_list = ibs.get_annot_gids(aid_list)
    gname_list = ibs.get_image_gnames(gid_list)
    return gname_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/image/unixtime/', methods=['GET'])
def get_annot_image_unixtimes(ibs, aid_list):
    r"""
    Args:
        aid_list (list):

    Returns:
        list: unixtime_list

    RESTful:
        Method: GET
        URL:    /api/annot/image/unixtime/
    """
    gid_list = ibs.get_annot_gids(aid_list)
    unixtime_list = ibs.get_image_unixtime(gid_list)
    return unixtime_list


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1to1
# @register_api('/api/annot/image/unixtime/float/', methods=['GET'])
def get_annot_image_unixtimes_asfloat(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):  list of annotation rowids

    Returns:
        list: unixtime_list

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --exec-get_annot_image_unixtimes_asfloat --show --db PZ_MTEST

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> unixtime_list = get_annot_image_unixtimes_asfloat(ibs, aid_list)
        >>> result = ('unixtime_list = %s' % (str(unixtime_list),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    unixtime_list = np.array(ibs.get_annot_image_unixtimes(aid_list), dtype=np.float)
    unixtime_list[unixtime_list == -1] = np.nan
    return unixtime_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_annot_image_datetime_str(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: datetime_list

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_image_datetime_str

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> datetime_list = get_annot_image_datetime_str(ibs, aid_list)
        >>> result = str(datetime_list)
        >>> print(result)
    """
    gid_list = ibs.get_annot_gids(aid_list)
    datetime_list = ibs.get_image_datetime_str(gid_list)
    return datetime_list


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1to1
@register_api('/api/annot/image/gps/', methods=['GET'], __api_plural_check__=False)
def get_annot_image_gps(ibs, aid_list):
    r"""
    Args:
        aid_list (list):

    Returns:
        list: unixtime_list

    RESTful:
        Method: GET
        URL:    /api/annot/image/gps/
    """
    gid_list = ibs.get_annot_gids(aid_list)
    gps_list = ibs.get_image_gps(gid_list)
    return gps_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/image/file/path/', methods=['GET'])
def get_annot_image_paths(ibs, aid_list):
    r"""
    Args:
        aid_list (list):

    Returns:
        list of strs: gpath_list the image paths of each annotation

    RESTful:
        Method: GET
        URL:    /api/annot/image/file/path/
    """
    gid_list = ibs.get_annot_gids(aid_list)
    try:
        ut.assert_all_not_None(gid_list, 'gid_list')
    except AssertionError:
        print('[!get_annot_image_paths] ' + ut.list_dbgstr('aid_list'))
        print('[!get_annot_image_paths] ' + ut.list_dbgstr('gid_list'))
        raise
    gpath_list = ibs.get_image_paths(gid_list)
    ut.assert_all_not_None(gpath_list, 'gpath_list')
    return gpath_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/image/uuid/', methods=['GET'])
def get_annot_image_uuids(ibs, aid_list):
    r"""
    Args:
        aid_list (list):

    Returns:
        list: image_uuid_list

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_image_uuids --enableall

    RESTful:
        Method: GET
        URL:    /api/annot/image/uuid/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:1]
        >>> result = get_annot_image_uuids(ibs, aid_list)
        >>> print(result)
        [UUID('66ec193a-1619-b3b6-216d-1784b4833b61')]
    """
    gid_list = ibs.get_annot_gids(aid_list)
    ut.assert_all_not_None(gid_list, 'gid_list')
    image_uuid_list = ibs.get_image_uuids(gid_list)
    return image_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annot/images/', methods=['GET'])
def get_annot_images(ibs, aid_list):
    r"""
    Args:
        aid_list (list):

    Returns:
        list of ndarrays: the images of each annotation

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_images

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:1]
        >>> image_list = ibs.get_annot_images(aid_list)
        >>> result = str(map(np.shape, image_list))
        >>> print(result)
        [(715, 1047, 3)]
    """
    gid_list = ibs.get_annot_gids(aid_list)
    image_list = ibs.get_image_imgdata(gid_list)
    return image_list


@register_ibs_method
# @register_api('/api/annot/uuid/visual/info/', methods=['GET'])
def get_annot_visual_uuid_info(ibs, aid_list):
    r"""

    Returns information used to compute annotation UUID.
    The image uuid, annotation verticies, are theta is hashted together to
      compute the visual uuid.
     The visual uuid does not include name or species information.

    get_annot_visual_uuid_info

    Args:
        aid_list (list):

    Returns:
        tuple: visual_infotup (image_uuid_list, verts_list, theta_list)

    SeeAlso:
        get_annot_visual_uuids
        get_annot_semantic_uuid_info

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_visual_uuid_info

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:2]
        >>> visual_infotup = ibs.get_annot_visual_uuid_info(aid_list)
        >>> result = str(list(zip(*visual_infotup))[0])
        >>> print(result)
        (UUID('66ec193a-1619-b3b6-216d-1784b4833b61'), ((0, 0), (1047, 0), (1047, 715), (0, 715)), 0.0)
    """
    image_uuid_list = ibs.get_annot_image_uuids(aid_list)
    verts_list      = ibs.get_annot_verts(aid_list)
    theta_list      = ibs.get_annot_thetas(aid_list)
    #visual_info_iter = zip(image_uuid_list, verts_list, theta_list, yaw_list)
    #visual_info_list = list(visual_info_iter)
    visual_infotup = (image_uuid_list, verts_list, theta_list)
    return visual_infotup


@register_ibs_method
# @register_api('/api/annot/uuid/semantic/info/', methods=['GET'])
def get_annot_semantic_uuid_info(ibs, aid_list, _visual_infotup=None):
    r"""
    Semenatic uuids are made up of visual and semantic information. Semantic
    information is name, species, viewpoint.  Visual info is image uuid, verts,
    and theta

    Args:
        aid_list (list):
        _visual_infotup (tuple) : internal use only

    Returns:
        tuple:  semantic_infotup (image_uuid_list, verts_list, theta_list, yaw_list, name_list, species_list)

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_semantic_uuid_info

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:2]
        >>> semantic_infotup = ibs.get_annot_semantic_uuid_info(aid_list)
        >>> result = str(list(zip(*semantic_infotup))[1])
        >>> print(result)
        (UUID('d8903434-942f-e0f5-d6c2-0dcbe3137bf7'), ((0, 0), (1035, 0), (1035, 576), (0, 576)), 0.0, 3.141592653589793, u'easy', u'zebra_plains')

    """
    # Semantic info depends on visual info
    if _visual_infotup is None:
        visual_infotup = get_annot_visual_uuid_info(ibs, aid_list)
    else:
        visual_infotup = _visual_infotup
    image_uuid_list, verts_list, theta_list = visual_infotup
    # It is visual info augmented with name and species
    name_list       = ibs.get_annot_names(aid_list)
    species_list    = ibs.get_annot_species_texts(aid_list)

    # TODO: Remove yaw?
    yaw_list        = ibs.get_annot_yaws(aid_list)

    semantic_infotup = SemanticInfoTup(
        image_uuid_list, verts_list, theta_list, yaw_list, name_list,
        species_list)
    return semantic_infotup


@register_ibs_method
@accessor_decors.cache_invalidator(const.ANNOTATION_TABLE, [ANNOT_SEMANTIC_UUID], rowidx=0)
# @register_api('/api/annot/uuid/semantic/', methods=['PUT'])
def update_annot_semantic_uuids(ibs, aid_list, _visual_infotup=None):
    r"""
    Ensures that annots have the proper semantic uuids
    """
    semantic_infotup = ibs.get_annot_semantic_uuid_info(aid_list, _visual_infotup)
    assert len(semantic_infotup) == 6, 'len=%r' % (len(semantic_infotup),)
    annot_semantic_uuid_list = [ut.augment_uuid(*tup) for tup in zip(*semantic_infotup)]
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_SEMANTIC_UUID,), annot_semantic_uuid_list, aid_list)


@register_ibs_method
@accessor_decors.cache_invalidator(const.ANNOTATION_TABLE,
                                   [ANNOT_VISUAL_UUID, ANNOT_SEMANTIC_UUID], rowidx=0)
# @register_api('/api/annot/uuid/visual/', methods=['PUT'])
def update_annot_visual_uuids(ibs, aid_list):
    r"""
    Ensures that annots have the proper visual uuids

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):  list of annotation rowids

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --exec-update_annot_visual_uuids --db PZ_Master1
        python -m ibeis --tf update_annot_visual_uuids --db PZ_Master1
        python -m ibeis --tf update_annot_visual_uuids --db PZ_Master0
        python -m ibeis --tf update_annot_visual_uuids --db PZ_MTEST

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs._get_all_aids()[0:1]
        >>> update_annot_visual_uuids(ibs, aid_list)
        >>> result = ibs.get_annot_visual_uuids(aid_list)[0]
        8687dcb6-1f1f-fdd3-8b72-8f36f9f41905
    """
    visual_infotup = ibs.get_annot_visual_uuid_info(aid_list)
    assert len(visual_infotup) == 3, 'len=%r' % (len(visual_infotup),)
    annot_visual_uuid_list = [ut.augment_uuid(*tup) for tup in zip(*visual_infotup)]
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_VISUAL_UUID,), annot_visual_uuid_list, aid_list)
    # If visual uuids are changes semantic ones are also changed
    ibs.update_annot_semantic_uuids(aid_list, _visual_infotup=visual_infotup)


#### SETTERS ####  # NOQA


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/bbox/', methods=['PUT'])
def set_annot_bboxes(ibs, aid_list, bbox_list, delete_thumbs=True):
    r"""
    Sets bboxes of a list of annotations by aid,

    Args:
        aid_list (list of rowids): list of annotation rowids
        bbox_list (list of (x, y, w, h)): new bounding boxes for each aid

    Note:
        set_annot_bboxes is a proxy for set_annot_verts

    RESTful:
        Method: PUT
        URL:    /api/annot/bbox/
    """
    from vtool import geometry
    # changing the bboxes also changes the bounding polygon
    vert_list = geometry.verts_list_from_bboxes_list(bbox_list)
    # naively overwrite the bounding polygon with a rectangle - for now trust the user!
    ibs.set_annot_verts(aid_list, vert_list, delete_thumbs=delete_thumbs)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/detect/confidence/', methods=['PUT'])
def set_annot_detect_confidence(ibs, aid_list, confidence_list):
    r"""
    Sets annotation notes

    RESTful:
        Method: PUT
        URL:    /api/annot/detect/confidence/
    """
    id_iter = ((aid,) for aid in aid_list)
    val_iter = ((confidence,) for confidence in confidence_list)
    ibs.db.set(const.ANNOTATION_TABLE, ('annot_detect_confidence',), val_iter, id_iter)


@register_ibs_method
@accessor_decors.setter
@accessor_decors.cache_invalidator(const.ANNOTATION_TABLE, [ANNOT_EXEMPLAR_FLAG], rowidx=0)
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['percent_names_with_exemplar_str'])
@register_api('/api/annot/exemplar/', methods=['PUT'])
def set_annot_exemplar_flags(ibs, aid_list, flag_list):
    r"""
    Sets if an annotation is an exemplar

    RESTful:
        Method: PUT
        URL:    /api/annot/exemplar/
    """
    id_iter = ((aid,) for aid in aid_list)
    val_iter = ((flag,) for flag in flag_list)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_EXEMPLAR_FLAG,), val_iter, id_iter)


@register_ibs_method
@accessor_decors.setter
@accessor_decors.cache_invalidator(const.ANNOTATION_TABLE, [NAME_ROWID], rowidx=0)
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['percent_names_with_exemplar_str'])
# @register_api('/api/annot/name/rowid/', methods=['PUT'])
def set_annot_name_rowids(ibs, aid_list, name_rowid_list):
    r"""
    name_rowid_list -> annot.name_rowid[aid_list]

    Sets names/nids of a list of annotations.

    Args:
        aid_list (list):
        name_rowid_list (list):

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:2]
        >>> # check clean state
        >>> ut.assert_eq(ibs.get_annot_names(aid_list), ['____', 'easy'])
        >>> ut.assert_eq(ibs.get_annot_exemplar_flags(aid_list), [0, 1])
        >>> # run function
        >>> name_list = ['easy', '____']
        >>> name_rowid_list = ibs.get_name_rowids_from_text(name_list)
        >>> ibs.set_annot_name_rowids(aid_list, name_rowid_list)
        >>> # check results
        >>> ut.assert_eq(ibs.get_annot_names(aid_list), ['easy', '____'])
        >>> ut.assert_eq(ibs.get_annot_exemplar_flags(aid_list), [0, 0])
        >>> # restore database state
        >>> ibs.set_annot_names(aid_list, ['____', 'easy'])
        >>> ibs.set_annot_exemplar_flags(aid_list, [0, 1])
        >>> ut.assert_eq(ibs.get_annot_names(aid_list), ['____', 'easy'])
        >>> ut.assert_eq(ibs.get_annot_exemplar_flags(aid_list), [0, 1])
    """
    #ibsfuncs.assert_lblannot_rowids_are_type(ibs, name_rowid_list,
    #ibs.lbltype_ids[const.INDIVIDUAL_KEY])
    id_iter = aid_list
    colnames = (NAME_ROWID,)
    # WE NEED TO PERFORM A SPECIAL CHECK. ANY ANIMAL WHICH IS GIVEN AN UNKONWN
    # NAME MUST HAVE ITS EXEMPLAR FLAG SET TO FALSE
    will_be_unknown_flag_list = [nid == const.UNKNOWN_NAME_ROWID for nid in name_rowid_list]
    if any(will_be_unknown_flag_list):
        # remove exemplar status from any annotations that will become unknown
        will_be_unknown_aids = ut.compress(aid_list, will_be_unknown_flag_list)
        ibs.set_annot_exemplar_flags(will_be_unknown_aids, [False] *
                                     len(will_be_unknown_aids))
    ibs.db.set(const.ANNOTATION_TABLE, colnames, name_rowid_list, id_iter)
    # postset nids
    ibs.update_annot_semantic_uuids(aid_list)
    # TODO: flag name rowid update
    # TODO: flag when the actual name changes as well?
    ibs.depc_annot.notify_root_changed(aid_list, 'name')


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/name/', methods=['PUT'])
def set_annot_names(ibs, aid_list, name_list):
    r"""
    Sets the attrlbl_value of type(INDIVIDUAL_KEY) Sets names/nids of a
    list of annotations.

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-set_annot_names --enableall

    RESTful:
        Method: PUT
        URL:    /api/annot/name/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> name_list1 = get_annot_names(ibs, aid_list)
        >>> name_list2 = [name + '_TESTAUG' for name in name_list1]
        >>> set_annot_names(ibs, aid_list, name_list2)
        >>> name_list3 = get_annot_names(ibs, aid_list)
        >>> set_annot_names(ibs, aid_list, name_list1)
        >>> name_list4 = get_annot_names(ibs, aid_list)
        >>> assert name_list2 == name_list3
        >>> assert name_list4 == name_list1
        >>> assert name_list4 != name_list2
        >>> print(result)
    """
    #ibs.get_nids_from_text
    assert len(aid_list) == len(name_list)
    assert not isinstance(name_list, six.string_types)
    #name_rowid_list = ibs.get_name_rowids_from_text(name_list, ensure=True)
    assert not any([name == '' for name in name_list]), (
        'cannot change name to empty string use ____ for unknown.')
    name_rowid_list = ibs.add_names(name_list)
    ibs.set_annot_name_rowids(aid_list, name_rowid_list)


@register_ibs_method
@accessor_decors.getter_1to1
def set_annot_name_texts(ibs, aid_list, name_list):
    r"""
    alias

    RESTful:
        Method: GET
        URL:    /api/annot/name/
    """
    return ibs.set_annot_names(aid_list, name_list)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/species/', methods=['PUT'], __api_plural_check__=False)
def set_annot_species(ibs, aid_list, species_text_list):
    r"""
    Sets species/speciesids of a list of annotations.
    Convenience function for set_annot_lblannot_from_value

    RESTful:
        Method: PUT
        URL:    /api/annot/species/
    """
    #ibs.get_nids_from_text
    species_rowid_list = ibs.get_species_rowids_from_text(species_text_list)
    ibs.set_annot_species_rowids(aid_list, species_rowid_list)


@register_ibs_method
@accessor_decors.setter
def set_annot_species_and_notify(ibs, *args, **kwargs):
    # for gui
    ibs.set_annot_species(*args, **kwargs)
    ibs.notify_observers()


@register_ibs_method
@accessor_decors.setter
@accessor_decors.cache_invalidator(const.ANNOTATION_TABLE, [SPECIES_ROWID], rowidx=0)
@register_api('/api/annot/species/rowid/', methods=['PUT'], __api_plural_check__=False)
def set_annot_species_rowids(ibs, aid_list, species_rowid_list):
    r"""
    species_rowid_list -> annot.species_rowid[aid_list]

    Sets species/speciesids of a list of annotations.

    Args:
        aid_list
        species_rowid_list


    RESTful:
        Method: PUT
        URL:    /api/annot/species/rowid/
    """
    id_iter = aid_list
    colnames = (SPECIES_ROWID,)
    ibs.db.set(const.ANNOTATION_TABLE, colnames, species_rowid_list, id_iter)
    ibs.update_annot_semantic_uuids(aid_list)
    ibs.depc_annot.notify_root_changed(aid_list, 'species')


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/note/', methods=['PUT'])
def set_annot_notes(ibs, aid_list, notes_list):
    r"""
    Sets annotation notes

    RESTful:
        Method: PUT
        URL:    /api/annot/note/
    """
    id_iter = ((aid,) for aid in aid_list)
    val_iter = ((notes,) for notes in notes_list)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_NOTE,), val_iter, id_iter)


@register_ibs_method
@accessor_decors.setter
# @register_api('/api/annot/parent/rowid/', methods=['PUT'])
def set_annot_parent_rowid(ibs, aid_list, parent_aid_list):
    r"""
    Sets the annotation's parent aid.
    TODO DEPRICATE IN FAVOR OF SEPARATE PARTS TABLE

    RESTful:
        Method: PUT
        URL:    /api/annot/parent/rowid/
    """
    id_iter = ((aid,) for aid in aid_list)
    val_iter = ((parent_aid,) for parent_aid in parent_aid_list)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_PARENT_ROWID,), val_iter, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/theta/', methods=['PUT'])
def set_annot_thetas(ibs, aid_list, theta_list, delete_thumbs=True):
    r"""
    Sets thetas of a list of chips by aid

    RESTful:
        Method: PUT
        URL:    /api/annot/theta/
    """
    id_iter = ((aid,) for aid in aid_list)
    val_list = ((theta,) for theta in theta_list)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_THETA,), val_list, id_iter)
    if delete_thumbs:
        ibs.delete_annot_chips(aid_list)  # Changing theta redefines the chips
        ibs.delete_annot_imgthumbs(aid_list)
    ibs.update_annot_visual_uuids(aid_list)
    ibs.depc_annot.notify_root_changed(aid_list, 'theta', force_delete=True)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/vert/', methods=['PUT'])
def set_annot_verts(ibs, aid_list, verts_list, delete_thumbs=True):
    r"""
    Sets the vertices [(x, y), ...] of a list of chips by aid

    RESTful:
        Method: PUT
        URL:    /api/annot/vert/
    """
    from vtool import geometry
    nInput = len(aid_list)
    # Compute data to set
    num_verts_list   = list(map(len, verts_list))
    verts_as_strings = list(map(const.__STR__, verts_list))
    id_iter1 = ((aid,) for aid in aid_list)
    # also need to set the internal number of vertices
    val_iter1 = ((num_verts, verts) for (num_verts, verts)
                 in zip(num_verts_list, verts_as_strings))
    colnames = (ANNOT_NUM_VERTS, ANNOT_VERTS,)
    # SET VERTS in ANNOTATION_TABLE
    ibs.db.set(const.ANNOTATION_TABLE, colnames, val_iter1, id_iter1, nInput=nInput)
    # changing the vertices also changes the bounding boxes
    bbox_list = geometry.bboxes_from_vert_list(verts_list)      # new bboxes
    xtl_list, ytl_list, width_list, height_list = list(zip(*bbox_list))
    val_iter2 = zip(xtl_list, ytl_list, width_list, height_list)
    id_iter2 = ((aid,) for aid in aid_list)
    colnames = ('annot_xtl', 'annot_ytl', 'annot_width', 'annot_height',)
    # SET BBOX in ANNOTATION_TABLE
    ibs.db.set(const.ANNOTATION_TABLE, colnames, val_iter2, id_iter2, nInput=nInput)
    if delete_thumbs:
        ibs.delete_annot_chips(aid_list)  # INVALIDATE THUMBNAILS
        ibs.delete_annot_imgthumbs(aid_list)
    ibs.update_annot_visual_uuids(aid_list)
    ibs.depc_annot.notify_root_changed(aid_list, 'verts', force_delete=True)


# PROBCHIP

@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annot/probchip/fpath/', methods=['GET'])
def get_annot_probchip_fpath(ibs, aid_list, config2_=None):
    r"""
    Returns paths to probability images.

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):  list of annotation rowids
        config2_ (dict): (default = None)

    Returns:
        list: probchip_fpath_list

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --exec-get_annot_probchip_fpath --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()[0:10]
        >>> config2_ = ibs.new_query_params({'fw_detector': 'cnn'})
        >>> probchip_fpath_list = get_annot_probchip_fpath(ibs, aid_list, config2_)
        >>> result = ('probchip_fpath_list = %s' % (str(probchip_fpath_list),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(probchip_fpath_list, nPerPage=4)
        >>> iteract_obj.start()
        >>> ut.show_if_requested()
    """
    probchip_fpath_list = ibs.depc_annot.get('probchip', aid_list, 'img',
                                             config=config2_, read_extern=False)
    return probchip_fpath_list


# ---
# NEW
# ---

@register_ibs_method
@accessor_decors.getter_1to1
@accessor_decors.cache_getter(const.ANNOTATION_TABLE, ANNOT_QUALITY)
@register_api('/api/annot/quality/', methods=['GET'])
def get_annot_qualities(ibs, aid_list, eager=True):
    r"""
    annot_quality_list <- annot.annot_quality[aid_list]

    gets data from the "native" column "annot_quality" in the "annot" table

    Args:
        aid_list (list):

    Returns:
        list: annot_quality_list

    TemplateInfo:
        Tgetter_table_column
        col = annot_quality
        tbl = annot

    SeeAlso:
        ibeis.const.QUALITY_INT_TO_TEXT

    RESTful:
        Method: GET
        URL:    /api/annot/quality/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> ibs, qreq_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()
        >>> eager = True
        >>> annot_quality_list = ibs.get_annot_qualities(aid_list, eager=eager)
        >>> print('annot_quality_list = %r' % (annot_quality_list,))
        >>> assert len(aid_list) == len(annot_quality_list)
    """
    id_iter = aid_list
    colnames = (ANNOT_QUALITY,)
    annot_quality_list = ibs.db.get(
        const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid', eager=eager)
    return annot_quality_list


@register_ibs_method
@accessor_decors.setter
@accessor_decors.cache_invalidator(const.ANNOTATION_TABLE, [ANNOT_QUALITY], rowidx=0)
@register_api('/api/annot/quality/', methods=['PUT'])
def set_annot_qualities(ibs, aid_list, annot_quality_list):
    r"""
    annot_quality_list -> annot.annot_quality[aid_list]

    A quality is an integer representing the following types:

    Args:
        aid_list
        annot_quality_list

    SeeAlso:
        ibeis.const.QUALITY_INT_TO_TEXT

    RESTful:
        Method: PUT
        URL:    /api/annot/quality/
    """
    id_iter = aid_list
    colnames = (ANNOT_QUALITY,)
    ibs.db.set(const.ANNOTATION_TABLE, colnames, annot_quality_list, id_iter)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/quality/text/', methods=['GET'])
def get_annot_quality_texts(ibs, aid_list):
    r"""
    Auto-docstr for 'get_annot_quality_texts'

    RESTful:
        Method: GET
        URL:    /api/annot/quality/text/
    """
    quality_list = ibs.get_annot_qualities(aid_list)
    quality_text_list = ut.dict_take(const.QUALITY_INT_TO_TEXT, quality_list)
    return quality_text_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annot/isjunk/', methods=['GET'])
def get_annot_isjunk(ibs, aid_list):
    r"""
    Auto-docstr for 'get_annot_isjunk'
    """
    qual_list = ibs.get_annot_qualities(aid_list)
    #isjunk_list = [qual == const.QUALITY_TEXT_TO_INT['junk'] for qual in qual_list]
    isjunk_list = [qual in const.QUALITY_TEXT_TO_INTS['junk'] for qual in qual_list]
    return isjunk_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/yaw/text/', methods=['GET'])
def get_annot_yaw_texts(ibs, aid_list):
    r"""
    Auto-docstr for 'get_annot_yaw_texts'

    RESTful:
        Method: GET
        URL:    /api/annot/yaw/text/
    """
    yaw_list = ibs.get_annot_yaws(aid_list)
    yaw_text_list = ibsfuncs.get_yaw_viewtexts(yaw_list)
    return yaw_text_list


@register_ibs_method
@register_api('/api/annot/quality/text/', methods=['PUT'])
def set_annot_quality_texts(ibs, aid_list, quality_text_list):
    r"""
    Auto-docstr for 'set_annot_quality_texts'

    RESTful:
        Method: PUT
        URL:    /api/annot/quality/text/
    """
    if not ut.isiterable(aid_list):
        aid_list = [aid_list]
    if isinstance(quality_text_list, six.string_types):
        quality_text_list = [quality_text_list]
    quality_list = ut.dict_take(const.QUALITY_TEXT_TO_INT, quality_text_list)
    ibs.set_annot_qualities(aid_list, quality_list)


@register_ibs_method
@register_api('/api/annot/yaw/text/', methods=['PUT'])
def set_annot_yaw_texts(ibs, aid_list, yaw_text_list):
    r"""
    Auto-docstr for 'set_annot_yaw_texts'

    RESTful:
        Method: PUT
        URL:    /api/annot/yaw/text/
    """
    if not ut.isiterable(aid_list):
        aid_list = [aid_list]
    if isinstance(yaw_text_list, six.string_types):
        yaw_text_list = [yaw_text_list]
    yaw_list = ut.dict_take(const.VIEWTEXT_TO_YAW_RADIANS, yaw_text_list, None)
    ibs.set_annot_yaws(aid_list, yaw_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/sex/', methods=['GET'])
def get_annot_sex(ibs, aid_list, eager=True, nInput=None):
    r"""
    Auto-docstr for 'get_annot_sex'

    RESTful:
        Method: GET
        URL:    /api/annot/sex/
    """
    nid_list = ibs.get_annot_nids(aid_list)
    sex_list = ibs.get_name_sex(nid_list)
    return sex_list


@register_ibs_method
@register_api('/api/annot/sex/text/', methods=['GET'])
def get_annot_sex_texts(ibs, aid_list, eager=True, nInput=None):
    r"""
    Auto-docstr for 'get_annot_sex_texts'

    RESTful:
        Method: GET
        URL:    /api/annot/sex/text/
    """
    nid_list = ibs.get_annot_nids(aid_list)
    sex_text_list = ibs.get_name_sex_text(nid_list)
    return sex_text_list


@register_ibs_method
@register_api('/api/annot/sex/', methods=['PUT'])
def set_annot_sex(ibs, aid_list, name_sex_list, eager=True, nInput=None):
    r"""
    Auto-docstr for 'set_annot_sex'

    RESTful:
        Method: PUT
        URL:    /api/annot/sex/
    """
    nid_list = ibs.get_annot_nids(aid_list)
    flag_list = [ nid is not None for nid in nid_list ]
    nid_list = ut.compress(nid_list, flag_list)
    name_sex_list = ut.compress(name_sex_list, flag_list)
    ibs.set_name_sex(nid_list, name_sex_list)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/sex/text/', methods=['PUT'])
def set_annot_sex_texts(ibs, aid_list, name_sex_text_list, eager=True, nInput=None):
    r"""
    Auto-docstr for 'set_annot_sex_texts'

    RESTful:
        Method: PUT
        URL:    /api/annot/sex/text/
    """
    nid_list = ibs.get_annot_nids(aid_list)
    flag_list = [ nid is not None for nid in nid_list ]
    nid_list = ut.compress(nid_list, flag_list)
    name_sex_text_list = ut.compress(name_sex_text_list, flag_list)
    ibs.set_name_sex_text(nid_list, name_sex_text_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/age/months/min/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_min(ibs, aid_list, eager=True, nInput=None):
    r"""
    annot_age_months_est_min_list <- annot.annot_age_months_est_min[aid_list]

    gets data from the "native" column "annot_age_months_est_min" in the "annot" table

    Args:
        aid_list (list):

    Returns:
        list: annot_age_months_est_min_list

    RESTful:
        Method: GET
        URL:    /api/annot/age/months/min/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()
        >>> eager = True
        >>> annot_age_months_est_min_list = ibs.get_annot_age_months_est_min(aid_list, eager=eager)
        >>> assert len(aid_list) == len(annot_age_months_est_min_list)
    """
    id_iter = aid_list
    colnames = (ANNOT_AGE_MONTHS_EST_MIN,)
    annot_age_months_est_min_list = ibs.db.get(
        const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid', eager=eager, nInput=nInput)
    return annot_age_months_est_min_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/age/months/max/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_max(ibs, aid_list, eager=True, nInput=None):
    r"""
    annot_age_months_est_max_list <- annot.annot_age_months_est_max[aid_list]

    gets data from the "native" column "annot_age_months_est_max" in the "annot" table

    Args:
        aid_list (list):

    Returns:
        list: annot_age_months_est_max_list

    RESTful:
        Method: GET
        URL:    /api/annot/age/months/max/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()
        >>> eager = True
        >>> annot_age_months_est_max_list = ibs.get_annot_age_months_est_max(aid_list, eager=eager)
        >>> assert len(aid_list) == len(annot_age_months_est_max_list)
    """
    id_iter = aid_list
    colnames = (ANNOT_AGE_MONTHS_EST_MAX,)
    annot_age_months_est_max_list = ibs.db.get(
        const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid', eager=eager, nInput=nInput)
    return annot_age_months_est_max_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/age/months/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est(ibs, aid_list, eager=True, nInput=None):
    r"""
    annot_age_months_est_list <- annot.annot_age_months_est[aid_list]

    gets data from the annotation's native age in months

    Args:
        aid_list (list):

    Returns:
        list: annot_age_months_est_list

    RESTful:
        Method: GET
        URL:    /api/annot/age/months/
    """
    annot_age_months_est_min_list = ibs.get_annot_age_months_est_min(aid_list)
    annot_age_months_est_max_list = ibs.get_annot_age_months_est_max(aid_list)
    annot_age_months_est_list = zip(
        annot_age_months_est_min_list,
        annot_age_months_est_max_list
    )
    return annot_age_months_est_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/age/months/min/text/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_min_texts(ibs, aid_list, eager=True, nInput=None):
    r"""
    annot_age_months_est_min_texts_list <- annot.annot_age_months_est_min_texts[aid_list]

    gets string versions of the annotation's native min age in months

    Args:
        aid_list (list):

    Returns:
        list: annot_age_months_est_min_list

    RESTful:
        Method: GET
        URL:    /api/annot/age/months/min/text/
    """
    annot_age_months_est_min_list = ibs.get_annot_age_months_est_min(aid_list)
    annot_age_months_est_min_text_list = [
        '%d Months' % (age_min, )
        for age_min in annot_age_months_est_min_list
    ]
    return annot_age_months_est_min_text_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/age/months/max/text/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_max_texts(ibs, aid_list, eager=True, nInput=None):
    r"""
    annot_age_months_est_max_texts_list <- annot.annot_age_months_est_max_texts[aid_list]

    gets string versions of the annotation's native max age in months

    Args:
        aid_list (list):

    Returns:
        list: annot_age_months_est_max_list

    RESTful:
        Method: GET
        URL:    /api/annot/age/months/max/text/
    """
    annot_age_months_est_max_list = ibs.get_annot_age_months_est_max(aid_list)
    annot_age_months_est_max_text_list = [
        '%d Months' % (age_max, )
        for age_max in annot_age_months_est_max_list
    ]
    return annot_age_months_est_max_text_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/age/months/text/', methods=['GET'], __api_plural_check__=False)
def get_annot_age_months_est_texts(ibs, aid_list, eager=True, nInput=None):
    r"""
    annot_age_months_est_texts_list <- annot.annot_age_months_est_texts[aid_list]

    gets string versions of the annotation's native combined age in months

    Args:
        aid_list (list):

    Returns:
        list: annot_age_months_est_text_list

    RESTful:
        Method: GET
        URL:    /api/annot/age/months/text/
    """
    annot_age_months_est_min_text_list = ibs.get_annot_age_months_est_min_texts(aid_list)
    annot_age_months_est_max_text_list = ibs.get_annot_age_months_est_max_texts(aid_list)
    annot_age_months_est_text_list = [
        '%s to %s' % (age_min_text, age_max_text, )
        for age_min_text, age_max_text in zip(
            annot_age_months_est_min_text_list,
            annot_age_months_est_max_text_list
        )
    ]
    return annot_age_months_est_text_list


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/age/months/min/', methods=['PUT'], __api_plural_check__=False)
def set_annot_age_months_est_min(ibs, aid_list, annot_age_months_est_min_list,
                                 duplicate_behavior='error'):
    r"""
    annot_age_months_est_min_list -> annot.annot_age_months_est_min[aid_list]

    Args:
        aid_list
        annot_age_months_est_min_list

    TemplateInfo:
        Tsetter_native_column
        tbl = annot
        col = annot_age_months_est_min

    RESTful:
        Method: PUT
        URL:    /api/annot/age/months/min/
    """
    id_iter = aid_list
    colnames = (ANNOT_AGE_MONTHS_EST_MIN,)
    ibs.db.set(const.ANNOTATION_TABLE, colnames, annot_age_months_est_min_list,
               id_iter, duplicate_behavior=duplicate_behavior)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/age/months/max/', methods=['PUT'], __api_plural_check__=False)
def set_annot_age_months_est_max(ibs, aid_list, annot_age_months_est_max_list,
                                 duplicate_behavior='error'):
    r"""
    annot_age_months_est_max_list -> annot.annot_age_months_est_max[aid_list]

    Args:
        aid_list
        annot_age_months_est_max_list

    TemplateInfo:
        Tsetter_native_column
        tbl = annot
        col = annot_age_months_est_max

    RESTful:
        Method: PUT
        URL:    /api/annot/age/months/max/
    """
    id_iter = aid_list
    colnames = (ANNOT_AGE_MONTHS_EST_MAX,)
    ibs.db.set(const.ANNOTATION_TABLE, colnames, annot_age_months_est_max_list,
               id_iter, duplicate_behavior=duplicate_behavior)


@register_ibs_method
@accessor_decors.getter
@register_api('/api/annot/image/contributor/tag/', methods=['GET'])
def get_annot_image_contributor_tag(ibs, aid_list):
    r"""
    Auto-docstr for 'get_annot_image_contributor_tag'
    """
    gid_list = ibs.get_annot_gids(aid_list)
    contrib_tag_list = ibs.get_image_contributor_tag(gid_list)
    return contrib_tag_list


@register_ibs_method
@accessor_decors.getter
@register_api('/api/annot/imageset/text/', methods=['GET'])
def get_annot_image_set_texts(ibs, aid_list):
    r"""
    Auto-docstr for 'get_annot_image_contributor_tag'

    RESTful:
        Method: GET
        URL:    /api/annot/imageset/text/
    """
    gid_list = ibs.get_annot_gids(aid_list)
    imagesettext_list = ibs.get_image_imagesettext(gid_list)
    filter_imageset_set = set([
        '*Exemplars',
        '*All Images',
        '*Ungrouped Images',
    ])
    filtered_imagesettext_list = [
        [
            imageset
            for imageset in imagesettext
            if imageset not in filter_imageset_set
        ]
        for imagesettext in imagesettext_list
    ]
    imagesettext_list = [
        ','.join(map(str, imagesettext))
        for imagesettext in filtered_imagesettext_list
    ]
    return imagesettext_list


@register_ibs_method
@accessor_decors.getter
def get_annot_rowids_from_partial_vuuids(ibs, partial_vuuid_strs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        partial_uuid_list (list):

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annots_from_partial_uuids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[::2]
        >>> vuuids = ibs.get_annot_visual_uuids(aid_list)
        >>> partial_vuuid_strs = [u[0:4] for u in map(str, vuuids)]
        >>> aids_list = get_annot_rowids_from_partial_vuuids(ibs, partial_uuid_list)
        >>> print(result)
        [[1], [3], [5], [7], [9], [11], [13]]
    """
    # Hackyway because I can't figure out startswith in sqlite for UUID blobs
    # just need to figure out how to convert a string into its corresponding byte
    # value, then I can do it with the referecence provided
    # References:
    # # search like with blobs
    # http://sqlite.1065341.n5.nabble.com/LIKE-with-BLOB-td48050.html
    aid_list = ibs.get_valid_aids()
    vuuids = ibs.get_annot_visual_uuids(aid_list)
    vuuid_strs = [_.replace('-', '') for _ in map(str, vuuids)]
    aids_list = [
        [aid for aid, vuuid in zip(aid_list, vuuid_strs) if vuuid.startswith(partial_vuuid)]
        for partial_vuuid in partial_vuuid_strs
    ]
    return aids_list

    #partial_vuuid = partial_vuuid_strs[0]

    #res = ibs.db.cur.execute(
    #    '''
    #    SELECT annot_rowid from ANNOTATIONS
    #    WHERE annot_visual_uuid LIKE ? || '%'
    #    ''', (bytes(partial_vuuid),))
    #print(res.fetchall())

    #res = ibs.db.cur.execute(
    #    '''
    #    SELECT annot_rowid from ANNOTATIONS
    #    WHERE annot_visual_uuid LIKE ? || '%'
    #    ''', (partial_vuuid,))
    #print(res.fetchall())

    #res = ibs.db.cur.execute(
    #    '''
    #    SELECT annot_rowid from ANNOTATIONS
    #    WHERE annot_visual_uuid LIKE ? || '%'
    #    ''', (bytes(partial_vuuid),))
    #print(res.fetchall())

    # # || - is used to concat strings

    #res = ibs.db.cur.execute(
    #    '''
    #    SELECT annot_rowid from ANNOTATIONS
    #    WHERE annot_note LIKE ? || '%'
    #    ''', ('very',))
    #print(res.fetchall())
    #pass


@register_ibs_method
@accessor_decors.getter_1to1
def get_annot_tag_text(ibs, aid_list, eager=True, nInput=None):
    r""" annot_tags_list <- annot.annot_tags[aid_list]

    gets data from the "native" column "annot_tags" in the "annot" table

    Args:
        aid_list (list):

    Returns:
        list: annot_tags_list

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()
        >>> eager = True
        >>> annot_tags_list = ibs.get_annot_tag_text(aid_list, eager=eager)
        >>> assert len(aid_list) == len(annot_tags_list)
    """
    id_iter = aid_list
    colnames = (ANNOT_TAG_TEXT,)
    annot_tags_list = ibs.db.get(
        const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid', eager=eager, nInput=nInput)
    return annot_tags_list


@register_ibs_method
@accessor_decors.setter
def set_annot_tag_text(ibs, aid_list, annot_tags_list, duplicate_behavior='error'):
    r""" annot_tags_list -> annot.annot_tags[aid_list]

    Args:
        aid_list
        annot_tags_list

    """
    #print('[ibs] set_annot_tag_text of aids=%r to tags=%r' % (aid_list, annot_tags_list))
    id_iter = aid_list
    colnames = (ANNOT_TAG_TEXT,)
    ibs.db.set(const.ANNOTATION_TABLE, colnames, annot_tags_list,
               id_iter, duplicate_behavior=duplicate_behavior)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/reviewed/', methods=['GET'])
def get_annot_reviewed(ibs, aid_list):
    r"""
    Returns:
        list_ (list): "All Instances Found" flag, true if all objects of interest
    (animals) have an ANNOTATION in the annot

    RESTful:
        Method: GET
        URL:    /api/annot/reviewed/
    """
    reviewed_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_toggle_reviewed',), aid_list)
    return reviewed_list


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/reviewed/', methods=['PUT'])
def set_annot_reviewed(ibs, aid_list, reviewed_list):
    r"""
    Sets the annot all instances found bit

    RESTful:
        Method: PUT
        URL:    /api/annot/reviewed/
    """
    id_iter = ((aid,) for aid in aid_list)
    val_list = ((reviewed,) for reviewed in reviewed_list)
    ibs.db.set(const.ANNOTATION_TABLE, ('annot_toggle_reviewed',), val_list, id_iter)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot/multiple/', methods=['GET'])
def get_annot_multiple(ibs, aid_list):
    r"""
    RESTful:
        Method: GET
        URL:    /api/annot/multiple/
    """
    flag_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_toggle_multiple',), aid_list)
    return flag_list


@register_ibs_method
@accessor_decors.setter
@register_api('/api/annot/mutiple/', methods=['PUT'])
def set_annot_multiple(ibs, aid_list, flag_list):
    r"""
    Sets the annot all instances found bit

    RESTful:
        Method: PUT
        URL:    /api/annot/multiple/
    """
    id_iter = ((aid,) for aid in aid_list)
    val_list = ((flag,) for flag in flag_list)
    ibs.db.set(const.ANNOTATION_TABLE, ('annot_toggle_multiple',), val_list, id_iter)


#==========
# Testdata
#==========


def testdata_ibs():
    r"""
    Auto-docstr for 'testdata_ibs'
    """
    import ibeis
    ibs = ibeis.opendb('testdb1')
    qreq_ = None
    return ibs, qreq_


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.control.manual_annot_funcs
        python -m ibeis.control.manual_annot_funcs --allexamples
        python -m ibeis.control.manual_annot_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
