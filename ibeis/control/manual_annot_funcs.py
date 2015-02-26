from __future__ import absolute_import, division, print_function
import six  # NOQA
import uuid
import numpy as np  # NOQA
from ibeis import constants as const
from ibeis.control import accessor_decors
from ibeis.control.accessor_decors import (adder, ider, getter_1to1, getter_1toM, deleter, setter)
import utool as ut
from ibeis import ibsfuncs
from ibeis.control.controller_inject import make_ibs_register_decorator
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[manual_annot]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


ANNOT_NOTE          = 'annot_note'
ANNOT_NUM_VERTS     = 'annot_num_verts'
ANNOT_PARENT_ROWID  = 'annot_parent_rowid'
ANNOT_ROWID         = 'annot_rowid'
ANNOT_SEMANTIC_UUID = 'annot_semantic_uuid'
ANNOT_THETA         = 'annot_theta'
ANNOT_VERTS         = 'annot_verts'
ANNOT_UUID          = 'annot_uuid'
ANNOT_YAW           = 'annot_yaw'
ANNOT_VISUAL_UUID   = 'annot_visual_uuid'
CONFIG_ROWID        = 'config_rowid'
FEATWEIGHT_ROWID    = 'featweight_rowid'
IMAGE_ROWID         = 'image_rowid'
NAME_ROWID          = 'name_rowid'
SPECIES_ROWID       = 'species_rowid'
ANNOT_EXEMPLAR_FLAG = 'annot_exemplar_flag'
ANNOT_QUALITY       = 'annot_quality'


# ==========
# IDERS
# ==========

# TODO CACHE THIS AND FIND WHAT IT SHOULD INVALIDATE IT
# ADD ANNOTS, DELETE ANNOTS ANYTHING ELSE?
@register_ibs_method
@ider
def _get_all_aids(ibs):
    """
    Returns:
        list_ (list):  all unfiltered aids (annotation rowids)
    """
    all_aids = ibs.db.get_all_rowids(const.ANNOTATION_TABLE)
    return all_aids


@register_ibs_method
def get_num_annotations(ibs, **kwargs):
    """ Number of valid annotations """
    aid_list = ibs.get_valid_aids(**kwargs)
    return len(aid_list)


@register_ibs_method
@ider
def get_valid_aids(ibs, eid=None, include_only_gid_list=None,
                   yaw='no-filter', is_exemplar=None, species=None,
                   is_known=None, nojunk=False):
    """
    Note: The yaw value cannot be None as a default because None is used as a
          filtering value

    Args:
        ibs (IBEISController):  ibeis controller object
        eid (None):
        include_only_gid_list (list): if specified filters annots not in these gids
        yaw (str):
        is_exemplar (bool): if specified filters annots to either be or not be exemplars
        species (None):
        is_known (None):

    Returns:
        list: aid_list - a list of valid ANNOTATION unique ids

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_valid_aids

    Ignore:
        ibs.print_annotation_table()

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> from ibeis import constants as const
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> eid = 1
        >>> ibs.delete_all_encounters()
        >>> ibs.compute_encounters()
        >>> include_only_gid_list = None
        >>> yaw = 'no-filter'
        >>> is_exemplar = None
        >>> species = const.Species.ZEB_PLAIN
        >>> is_known = False
        >>> # execute function
        >>> aid_list = get_valid_aids(ibs, eid, include_only_gid_list, yaw, is_exemplar, species, is_known)
        >>> ut.assert_eq(ibs.get_annot_names(aid_list), [const.UNKNOWN] * 2, 'bad name')
        >>> ut.assert_eq(ibs.get_annot_species(aid_list), [const.Species.ZEB_PLAIN] * 2, 'bad species')
        >>> ut.assert_eq(ibs.get_annot_exemplar_flags(aid_list), [False] * 2, 'bad exemplar')
        >>> # verify results
        >>> result = str(aid_list)
        >>> print(result)
        [1, 4]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> from ibeis import constants as const
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> # execute function
        >>> aid_list1 = get_valid_aids(ibs, is_exemplar=True)
        >>> aid_list2 = get_valid_aids(ibs, is_exemplar=False)
        >>> intersect_aids = set(aid_list1).intersection(aid_list2)
        >>> ut.assert_eq(len(aid_list1), 9)
        >>> ut.assert_eq(len(aid_list2), 4)
        >>> ut.assert_eq(len(intersect_aids), 0)
    """
    # getting encounter aid
    if eid is None:
        if is_exemplar is not None:
            # Optimization Hack
            aid_list = ibs.db.get_all_rowids_where(const.ANNOTATION_TABLE, 'annot_exemplar_flag=?', (is_exemplar,))
        else:
            aid_list = ibs._get_all_aids()
    else:
        # HACK: Check to see if you want the
        # exemplar "encounter" (image group)
        enctext = ibs.get_encounter_enctext(eid)
        if enctext == const.EXEMPLAR_ENCTEXT:
            is_exemplar = True
        aid_list = ibs.get_encounter_aids(eid)
        if is_exemplar is True:
            # corresponding unoptimized hack for is_exemplar
            flag_list = ibs.get_annot_exemplar_flags(aid_list)
            aid_list  = ut.filter_items(aid_list, flag_list)
        elif is_exemplar is False:
            flag_list = ibs.get_annot_exemplar_flags(aid_list)
            aid_list  = ut.filterfalse_items(aid_list, flag_list)
    # -- valid aid filtering --
    if include_only_gid_list is not None:
        gid_list     = ibs.get_annot_gids(aid_list)
        is_valid_gid = [gid in include_only_gid_list for gid in gid_list]
        aid_list     = ut.filter_items(aid_list, is_valid_gid)
    if yaw != 'no-filter':
        yaw_list     = ibs.get_annot_yaws(aid_list)
        is_valid_yaw = [yaw == flag for flag in yaw_list]
        aid_list           = ut.filter_items(aid_list, is_valid_yaw)
    if species is not None:
        species_rowid      = ibs.get_species_rowids_from_text(species)
        species_rowid_list = ibs.get_annot_species_rowids(aid_list)
        is_valid_species   = [sid == species_rowid for sid in species_rowid_list]
        aid_list           = ut.filter_items(aid_list, is_valid_species)
    if is_known is not None:
        is_unknown_list = ibs.is_aid_unknown(aid_list)
        if is_known is True:
            aid_list = ut.filterfalse_items(aid_list, is_unknown_list)
        elif is_known is False:
            aid_list = ut.filter_items(aid_list, is_unknown_list)
    if nojunk is True:
        # remove junk annotations
        quality_list = ibs.get_annot_qualities(aid_list)
        isjunk_list = [quality == const.QUALITY_TEXT_TO_INT['junk'] for quality in quality_list]
        aid_list = ut.filterfalse_items(aid_list, isjunk_list)
    return aid_list


# ==========
# ADDERS
# ==========


@register_ibs_method
@adder
def add_annots(ibs, gid_list, bbox_list=None, theta_list=None,
                species_list=None, nid_list=None, name_list=None,
                detect_confidence_list=None, notes_list=None,
                vert_list=None, annot_uuid_list=None, yaw_list=None,
                annot_visual_uuid_list=None, annot_semantic_uuid_list=None,
                species_rowid_list=None, quiet_delete_thumbs=False,
                prevent_visual_duplicates=True):
    r"""
    Adds an annotation to images

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
        yaw_list           (list):
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
        >>> chip_fpaths2 = ibs.get_annot_chip_fpaths(aid_list, ensure=False)
        >>> assert [fpath is None for fpath in chip_fpaths2]
        >>> print('Ensure=True. Should get back None chip fpaths')
        >>> chip_fpaths = ibs.get_annot_chip_fpaths(aid_list, ensure=True)
        >>> assert all([ut.checkpath(fpath, verbose=True) for fpath in chip_fpaths])
        >>> assert len(aid_list) == num_add
        >>> assert len(vert_list2[0]) == 4
        >>> assert bbox_list2 == bbox_list
        >>> # Be sure to remove test annotation
        >>> # if this test fails a resetdbs might be nessary
        >>> result = ''
        >>> visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
        >>> semantic_uuid_list = ibs.get_annot_semantic_uuids(aid_list)
        >>> result += str(visual_uuid_list) + '\n'
        >>> result += str(semantic_uuid_list) + '\n'
        >>> print('Cleaning up. Removing added annotations')
        >>> ibs.delete_annots(aid_list)
        >>> assert not any([ut.checkpath(fpath, verbose=True) for fpath in chip_fpaths])
        >>> postvalid = ibs.get_valid_aids()
        >>> assert prevalid == postvalid
        >>> result += str(postvalid)
        >>> print(result)
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
        >>> assert aid_list1 == aid_list2
        >>> assert aid_list1 != aid_list3
        >>> aid_list_new = aid_list1 + aid_list3
        >>> result = aid_list_new
        >>> print('Cleaning up. Removing added annotations')
        >>> ibs.delete_annots(aid_list_new)
        >>> print(result)
        [14, 15]
    """
    #ut.embed()
    from ibeis.model.preproc import preproc_annot
    from vtool import geometry
    if ut.VERBOSE:
        print('[ibs] adding annotations')
    # Prepare the SQL input
    assert name_list is None or nid_list is None, 'cannot specify both names and nids'
    # For import only, we can specify both by setting import_override to True
    assert bool(bbox_list is None) != bool(vert_list is None), 'must specify exactly one of bbox_list or vert_list'
    ut.assert_all_not_None(gid_list, 'gid_list')

    if theta_list is None:
        theta_list = [0.0 for _ in range(len(gid_list))]
    if name_list is not None:
        nid_list = ibs.add_names(name_list)
    else:
        if nid_list is None:
            nid_list = [ibs.UNKNOWN_NAME_ROWID for _ in range(len(gid_list))]
        name_list = ibs.get_name_texts(nid_list)

    if species_rowid_list is not None:
        assert species_list is None, 'cannot mix species_rowid and species'
        species_list = ibs.get_species_texts(species_rowid_list)
    else:
        if species_list is not None:
            species_rowid_list = ibs.add_species(species_list)
        else:
            species_rowid_list = [ibs.UNKNOWN_SPECIES_ROWID for _ in range(len(gid_list))]
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

    # Careful this code is very fragile. It might go out of sync
    # with the updating of the determenistic uuids. Find a way to
    # integrate both pieces of code without too much reundancy.
    # Make sure these tuples are constructed correctly
    if annot_visual_uuid_list is None:
        visual_infotup = (image_uuid_list, vert_list, theta_list)
        annot_visual_uuid_list = preproc_annot.make_annot_visual_uuid(visual_infotup)
    if annot_semantic_uuid_list is None:
        semantic_infotup = (image_uuid_list, vert_list, theta_list, yaw_list,
                            name_list, species_list)
        annot_semantic_uuid_list = preproc_annot.make_annot_semantic_uuid(semantic_infotup)

    # Define arguments to insert
    colnames = ('annot_uuid', 'image_rowid', 'annot_xtl', 'annot_ytl',
                'annot_width', 'annot_height', 'annot_theta', 'annot_num_verts',
                'annot_verts', ANNOT_YAW, 'annot_detect_confidence',
                'annot_note', 'name_rowid', 'species_rowid',
                'annot_visual_uuid', 'annot_semantic_uuid')

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
    ibs.delete_image_thumbs(gid_list, quiet=quiet_delete_thumbs)
    return aid_list


@register_ibs_method
def get_annot_rows(ibs, aid_list):
    colnames = ('annot_uuid', 'image_rowid', 'annot_xtl', 'annot_ytl',
                'annot_width', 'annot_height', 'annot_theta', 'annot_num_verts',
                'annot_verts', ANNOT_YAW, 'annot_detect_confidence',
                'annot_note', 'name_rowid', 'species_rowid',
                'annot_visual_uuid', 'annot_semantic_uuid')
    rows_list = ibs.db.get(const.ANNOTATION_TABLE, colnames, aid_list, unpack_scalars=False)
    return rows_list


# ==========
# DELETERS
# ==========


@register_ibs_method
@deleter
def delete_annot_nids(ibs, aid_list):
    """
    Remove name assocation from the list of input aids.
    Does this by setting each annotations nid to the UNKNOWN name rowid
    """
    # FIXME: This should be implicit by setting the anotation name to the
    # unknown name
    #ibs.delete_annot_relations_oftype(aid_list, const.INDIVIDUAL_KEY)
    ibs.set_annot_name_rowids(aid_list, [ibs.UNKNOWN_NAME_ROWID] * len(aid_list))


@register_ibs_method
@deleter
def delete_annot_speciesids(ibs, aid_list):
    """ Deletes nids of a list of annotations """
    # FIXME: This should be implicit by setting the anotation name to the
    # unknown species
    #ibs.delete_annot_relations_oftype(aid_list, const.SPECIES_KEY)
    ibs.set_annot_species_rowids(aid_list, [ibs.UNKNOWN_SPECIES_ROWID] * len(aid_list))


@register_ibs_method
@deleter
@accessor_decors.cache_invalidator(const.ANNOTATION_TABLE)
def delete_annots(ibs, aid_list):
    """ deletes annotations from the database """
    if ut.VERBOSE:
        print('[ibs] deleting %d annotations' % len(aid_list))
    # Delete chips and features first
    from ibeis.model.preproc import preproc_annot
    preproc_annot.on_delete(ibs, aid_list)
    ibs.db.delete_rowids(const.ANNOTATION_TABLE, aid_list)


# ==========
# GETTERS
# ==========


@register_ibs_method
@getter_1to1
def get_annot_aids_from_semantic_uuid(ibs, semantic_uuid_list):
    """
    Args:
        semantic_uuid_list (list):

    Returns:
        list: annot rowids

    """
    aids_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_ROWID,),
                           semantic_uuid_list, id_colname=ANNOT_SEMANTIC_UUID)
    return aids_list


@register_ibs_method
@getter_1to1
def get_annot_aids_from_uuid(ibs, uuid_list):
    """
    Returns:
        list_ (list): annot rowids
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    aids_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_ROWID,), uuid_list, id_colname=ANNOT_UUID)
    return aids_list


@register_ibs_method
@getter_1to1
def get_annot_aids_from_visual_uuid(ibs, visual_uuid_list):
    """
    Args:
        visual_uuid_list (list):

    Returns:
        list: annot rowids

    """
    aids_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_ROWID,),
                           visual_uuid_list, id_colname=ANNOT_VISUAL_UUID)
    return aids_list


@register_ibs_method
@ut.accepts_numpy
@getter_1toM
def get_annot_bboxes(ibs, aid_list):
    """
    Returns:
        bbox_list (list):  annotation bounding boxes in image space
    """
    colnames = ('annot_xtl', 'annot_ytl', 'annot_width', 'annot_height',)
    bbox_list = ibs.db.get(const.ANNOTATION_TABLE, colnames, aid_list)
    return bbox_list


@register_ibs_method
def get_annot_class_labels(ibs, aid_list):
    """
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
@getter_1to1
def get_annot_detect_confidence(ibs, aid_list):
    """
    Returns:
        list_ (list): a list confidences that the annotations is a valid detection
    """
    annot_detect_confidence_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_detect_confidence',), aid_list)
    return annot_detect_confidence_list


@register_ibs_method
@getter_1to1
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> # execute function
        >>> gid_list = get_annot_exemplar_flags(ibs, aid_list)
        >>> # verify results
        >>> result = str(gid_list)
        >>> print(result)
    """
    annot_exemplar_flag_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_exemplar_flag',), aid_list)
    return annot_exemplar_flag_list


@register_ibs_method
@ut.accepts_numpy
@getter_1to1
#@cache_getter(const.ANNOTATION_TABLE, 'image_rowid')
def get_annot_gids(ibs, aid_list):
    """
    Get parent image rowids of annotations

    Args:
        aid_list (list):

    Returns:
        gid_list (list):  image rowids

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
@getter_1toM
def get_annot_contact_aids(ibs, aid_list):
    """
    Returns the other aids that appear in the same image that this
    annotation is from.

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_contact_aids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> # execute function
        >>> contact_aids = ibs.get_annot_contact_aids(aid_list)
        >>> # verify results
        >>> contact_gids = ibs.unflat_map(ibs.get_annot_gids, contact_aids)
        >>> gid_list = ibs.get_annot_gids(aid_list)
        >>> for gids, gid, aids, aid in zip(contact_gids, gid_list, contact_aids, aid_list):
        ...     assert ut.list_allsame(gids), 'annots should be from same image'
        ...     assert len(gids) == 0 or gids[0] == gid, 'and same image as parent annot'
        ...     assert aid not in aids, 'should not include self'
    """
    gid_list = ibs.get_annot_gids(aid_list)
    other_aids_list = ibs.get_image_aids(gid_list)
    # remove self
    contact_aids = [list(set(aids) - {aid})
                    for aids, aid in zip(other_aids_list, aid_list)]
    return contact_aids


@register_ibs_method
@getter_1to1
def get_annot_num_contact_aids(ibs, aid_list):
    nOther_aids_list = list(map(len, ibs.get_annot_contact_aids(aid_list)))
    return nOther_aids_list


@register_ibs_method
@getter_1toM
def get_annot_groundfalse(ibs, aid_list, is_exemplar=None, valid_aids=None,
                          filter_unknowns=True, daid_list=None):
    """
    gets all annotations with different names

    Returns:
        groundfalse_list (list): a list of aids which are known to be different for each

    input aid
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
@getter_1toM
def get_annot_groundtruth(ibs, aid_list, is_exemplar=None, noself=True,
                          daid_list=None):
    """
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> is_exemplar, noself, daid_list = None, True, None
        >>> groundtruth_list = ibs.get_annot_groundtruth(aid_list, is_exemplar, noself, daid_list)
        >>> result = str(groundtruth_list)
        >>> print(result)
        [[], [3], [2], [], [6], [5], [], [], [], [], [], [], []]

    Example1:
        >>> # ENABLE_DOCTEST
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
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> is_exemplar, noself, daid_list = False, False, None
        >>> groundtruth_list = ibs.get_annot_groundtruth(aid_list, is_exemplar, noself, daid_list)
        >>> result = str(groundtruth_list)
        >>> print(result)
        [[1], [], [], [4], [], [], [], [], [9], [], [11], [], []]

    """
    # TODO: Optimize
    nid_list = ibs.get_annot_name_rowids(aid_list)
    aids_list = ibs.get_name_aids(nid_list, enable_unknown_fix=True)
    if is_exemplar is None:
        groundtruth_list_ = aids_list
    else:
        # Filter out non-exemplars
        exemplar_flags_list = ibsfuncs.unflat_map(ibs.get_annot_exemplar_flags, aids_list)
        isvalids_list = [[flag == is_exemplar for flag in flags] for flags in exemplar_flags_list]
        groundtruth_list_ = [ut.filter_items(aids, isvalids)
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
@getter_1to1
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
def get_annot_hashid_uuid(ibs, aid_list, prefix=''):
    uuid_list    = ibs.get_annot_uuids(aid_list)
    label = ''.join(('_', prefix, 'UUIDS'))
    uuid_hashid  = ut.hashstr_arr(uuid_list, label)
    return uuid_hashid


@register_ibs_method
def get_annot_hashid_visual_uuid(ibs, aid_list, prefix=''):
    visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
    label = ''.join(('_', prefix, 'VUUIDS'))
    visual_uuid_hashid  = ut.hashstr_arr(visual_uuid_list, label)
    return visual_uuid_hashid


@register_ibs_method
def get_annot_hashid_semantic_uuid(ibs, aid_list, prefix=''):
    semantic_uuid_list = ibs.get_annot_semantic_uuids(aid_list)
    label = ''.join(('_', prefix, 'SUUIDS'))
    semantic_uuid_hashid  = ut.hashstr_arr(semantic_uuid_list, label)
    return semantic_uuid_hashid


@register_ibs_method
@getter_1to1
def get_annot_thetas(ibs, aid_list):
    """
    Returns:
        theta_list (list): a list of floats describing the angles of each chip

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_thetas

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
@getter_1to1
def get_annot_uuids(ibs, aid_list):
    """
    Returns:
        list: annot_uuid_list a list of image uuids by aid
    """
    annot_uuid_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_uuid',), aid_list)
    return annot_uuid_list


@register_ibs_method
@getter_1to1
#@accessor_decors.dev_cache_getter(const.ANNOTATION_TABLE, ANNOT_SEMANTIC_UUID)
def get_annot_semantic_uuids(ibs, aid_list):
    """ annot_semantic_uuid_list <- annot.annot_semantic_uuid[aid_list]

    gets data from the "native" column "annot_semantic_uuid" in the "annot" table

    Args:
        aid_list (list):

    Returns:
        list: annot_semantic_uuid_list

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_semantic_uuids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control._autogen_annot_funcs import *  # NOQA
        >>> ibs, qreq_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()[0:1]
        >>> annot_semantic_uuid_list = ibs.get_annot_semantic_uuids(aid_list)
        >>> assert len(aid_list) == len(annot_semantic_uuid_list)
        >>> result = annot_semantic_uuid_list
        [UUID('215ab5f9-fe53-d7d1-59b8-d6b5ce7e6ca6')]
    """
    id_iter = aid_list
    colnames = (ANNOT_SEMANTIC_UUID,)
    annot_semantic_uuid_list = ibs.db.get(
        const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid')
    return annot_semantic_uuid_list


@register_ibs_method
@getter_1to1
@accessor_decors.dev_cache_getter(const.ANNOTATION_TABLE, ANNOT_VISUAL_UUID, native_rowids=True)
def get_annot_visual_uuids(ibs, aid_list):
    """ annot_visual_uuid_list <- annot.annot_visual_uuid[aid_list]

    gets data from the "native" column "annot_visual_uuid" in the "annot" table

    Args:
        aid_list (list):

    Returns:
        list: annot_visual_uuid_list

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_visual_uuids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control._autogen_annot_funcs import *  # NOQA
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
@getter_1to1
def get_annot_verts(ibs, aid_list):
    """
    Returns:
        vert_list (list): the vertices that form the polygon of each chip
    """
    from ibeis.model.preproc import preproc_annot
    vertstr_list = ibs.db.get(const.ANNOTATION_TABLE, ('annot_verts',), aid_list)
    vert_list = preproc_annot.postget_annot_verts(vertstr_list)
    return vert_list


@register_ibs_method
@getter_1to1
def get_annot_yaws(ibs, aid_list):
    """
    A yaw is the yaw of the annotation in radians
    Viewpoint yaw is inverted. Will be fixed soon.

    The following views have these angles of yaw:
        left side  - 0.50 tau radians
        front side - 0.25 tau radians
        right side - 0.00 radians
        back side  - 0.75 tau radians

    SeeAlso:
        ibies.const.VIEWTEXT_TO_YAW_RADIANS

    Returns:
        yaw_list (list): the yaw (in radians) for the annotation

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_yaws

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[::3]
        >>> result = get_annot_yaws(ibs, aid_list)
        >>> print(result)
        [None, None, None, None, None]
    """
    #from ibeis.model.preproc import preproc_annot
    yaw_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_YAW,), aid_list)
    yaw_list = [yaw if yaw >= 0.0 else None for yaw in yaw_list]
    return yaw_list


@register_ibs_method
@setter
def set_annot_yaws(ibs, aid_list, yaw_list, input_is_degrees=False):
    """
    Sets the  yaw of a list of chips by aid

    A yaw is the yaw of the annotation in radians
    Viewpoint yaw is inverted. Will be fixed soon.

    The following views have these angles of yaw:
        left side  - 0.00 tau radians
        front side - 0.25 tau radians
        right side - 0.50 radians
        back side  - 0.75 tau radians

    SeeAlso:
        ibies.const.VIEWTEXT_TO_YAW_RADIANS

    References;
        http://upload.wikimedia.org/wikipedia/commons/7/7e/Rollpitchyawplain.png

    """
    id_iter = ((aid,) for aid in aid_list)
    #yaw_list = [-1 if yaw is None else yaw for yaw in yaw_list]
    if input_is_degrees:
        yaw_list = [-1 if yaw is None else ut.deg_to_rad(yaw) for yaw in yaw_list]
    #assert all([0.0 <= yaw < 2 * np.pi or yaw == -1.0 for yaw in yaw_list])
    val_iter = ((yaw, ) for yaw in yaw_list)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_YAW,), val_iter, id_iter)
    ibs.update_annot_visual_uuids(aid_list)


@register_ibs_method
@getter_1to1
def get_annot_notes(ibs, aid_list):
    """
    Returns:
        annotation_notes_list (list): a list of annotation notes
    """
    annotation_notes_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_NOTE,), aid_list)
    return annotation_notes_list


@register_ibs_method
@getter_1to1
def get_annot_num_groundtruth(ibs, aid_list, is_exemplar=None, noself=True,
                              daid_list=None):
    """
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
@getter_1to1
def get_annot_num_verts(ibs, aid_list):
    """
    Returns:
        nVerts_list (list): the number of vertices that form the polygon of each chip
    """
    nVerts_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_NUM_VERTS,), aid_list)
    return nVerts_list


@register_ibs_method
@getter_1to1
def get_annot_parent_aid(ibs, aid_list):
    """
    Returns:
        list_ (list): a list of parent (in terms of parts) annotation rowids.
    """
    annot_parent_rowid_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_PARENT_ROWID,), aid_list)
    return annot_parent_rowid_list


@register_ibs_method
@ut.accepts_numpy
@getter_1to1
def get_annot_name_rowids(ibs, aid_list, distinguish_unknowns=True):
    """
    Returns:
        list_ (list): the name id of each annotation.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> distinguish_unknowns = True
        >>> nid_arr1 = np.array(ibs.get_annot_name_rowids(aid_list, distinguish_unknowns))
        >>> nid_arr2 = np.array(ibs.get_annot_name_rowids(aid_list, False))
        >>> assert ibs.UNKNOWN_LBLANNOT_ROWID == 0
        >>> assert np.all(nid_arr1[np.where(ibs.UNKNOWN_LBLANNOT_ROWID == nid_arr2)[0]] < 0)
    """
    id_iter = aid_list
    colnames = (NAME_ROWID,)
    nid_list_ = ibs.db.get(
        const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid')

    ## OLD LBLANNOT WAY
    ## Get all the annotation lblannot relationships
    ## filter out only the ones which specify names
    #alrids_list = ibs.get_annot_alrids_oftype(aid_list, ibs.lbltype_ids[const.INDIVIDUAL_KEY])
    #lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
    ## Get a single nid from the list of lblannot_rowids of type INDIVIDUAL
    ## TODO: get index of highest confidence name
    #nid_list_ = [lblannot_rowids[0] if len(lblannot_rowids) > 0 else ibs.UNKNOWN_LBLANNOT_ROWID for
    #             lblannot_rowids in lblannot_rowids_list]

    if distinguish_unknowns:
        from ibeis.model.preproc import preproc_annot
        nid_list = preproc_annot.distinguish_unknown_nids(ibs, aid_list, nid_list_)
        #nid_list = [-aid if nid == ibs.UNKNOWN_LBLANNOT_ROWID else nid
        #            for nid, aid in zip(nid_list_, aid_list)]
    else:
        nid_list = nid_list_
    return nid_list


@register_ibs_method
def get_annot_nids(ibs, aid_list, distinguish_unknowns=True):
    """ alias """
    return ibs.get_annot_name_rowids(aid_list, distinguish_unknowns=distinguish_unknowns)


@register_ibs_method
@getter_1to1
def get_annot_names(ibs, aid_list):
    """ alias """
    return ibs.get_annot_name_texts(aid_list)


@register_ibs_method
@getter_1to1
def get_annot_name_texts(ibs, aid_list):
    """
    Args:
        aid_list (list):

    Returns:
        list or strs: name_list. e.g: ['fred', 'sue', ...]
             for each annotation identifying the individual

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[::2]
        >>> result = get_annot_name_texts(ibs, aid_list)
        >>> print(result)
        ['____', u'easy', u'hard', u'jeff', '____', '____', u'zebra']
    """
    nid_list = ibs.get_annot_name_rowids(aid_list)
    name_list = ibs.get_name_texts(nid_list)
    #name_list = ibs.get_annot_lblannot_value_of_lbltype(aid_list, const.INDIVIDUAL_KEY, ibs.get_name_texts)
    return name_list


@register_ibs_method
@getter_1to1
def get_annot_species(ibs, aid_list):
    """ alias"""
    return ibs.get_annot_species_texts(aid_list)


@register_ibs_method
@getter_1to1
def get_annot_species_texts(ibs, aid_list):
    """

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
        >>> result = get_annot_species_texts(ibs, aid_list)
        >>> print(result)
        [u'zebra_plains', u'zebra_plains', '____', u'bear_polar']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_annot_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> species_list = get_annot_species_texts(ibs, aid_list)
        >>> result = set(species_list)
        >>> print(result)
        set([u'zebra_plains'])
    """
    species_rowid_list = ibs.get_annot_species_rowids(aid_list)
    speceis_text_list  = ibs.get_species_texts(species_rowid_list)
    #speceis_text_list = ibs.get_annot_lblannot_value_of_lbltype(
    #    aid_list, const.SPECIES_KEY, ibs.get_species)
    return speceis_text_list


@register_ibs_method
@ut.accepts_numpy
@getter_1to1
@accessor_decors.dev_cache_getter(const.ANNOTATION_TABLE, SPECIES_ROWID, native_rowids=True)
def get_annot_species_rowids(ibs, aid_list):
    """
    species_rowid_list <- annot.species_rowid[aid_list]

    gets data from the "native" column "species_rowid" in the "annot" table

    Args:
        aid_list (list):

    Returns:
        list: species_rowid_list
    """

    id_iter = aid_list
    colnames = (SPECIES_ROWID,)
    species_rowid_list = ibs.db.get(
        const.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid')
    return species_rowid_list


@register_ibs_method
@getter_1to1
def get_annot_image_names(ibs, aid_list):
    """
    Args:
        aid_list (list):

    Returns:
        list of strs: gname_list the image names of each annotation
    """
    gid_list = ibs.get_annot_gids(aid_list)
    gname_list = ibs.get_image_gnames(gid_list)
    return gname_list


@register_ibs_method
@getter_1to1
def get_annot_image_unixtimes(ibs, aid_list):
    """
    Args:
        aid_list (list):

    Returns:
        list: unixtime_list
    """
    gid_list = ibs.get_annot_gids(aid_list)
    unixtime_list = ibs.get_image_unixtime(gid_list)
    return unixtime_list


@register_ibs_method
@getter_1to1
def get_annot_image_paths(ibs, aid_list):
    """
    Args:
        aid_list (list):

    Returns:
        list of strs: gpath_list the image paths of each annotation
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
@getter_1to1
def get_annot_image_uuids(ibs, aid_list):
    """
    Args:
        aid_list (list):

    Returns:
        list: image_uuid_list

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_image_uuids --enableall

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
@getter_1to1
def get_annot_images(ibs, aid_list):
    """
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
    image_list = ibs.get_images(gid_list)
    return image_list


@register_ibs_method
def get_annot_visual_uuid_info(ibs, aid_list):
    """
    Returns annotation UUID that is unique for the visual qualities
    of the annoation. does not include name ore species information.

    get_annot_visual_uuid_info

    Args:
        aid_list (list):

    Returns:
        tuple: visual_infotup (image_uuid_list, verts_list, theta_list)

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_visual_uuid_info

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_annot import *  # NOQA
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
def get_annot_semantic_uuid_info(ibs, aid_list, _visual_infotup=None):
    """
    Args:
        aid_list (list):
        _visual_infotup (tuple) : internal use only

    Returns:
        tuple:  semantic_infotup (image_uuid_list, verts_list, theta_list, yaw_list, name_list, species_list)

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-get_annot_semantic_uuid_info

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_annot import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:2]
        >>> semantic_infotup = ibs.get_annot_semantic_uuid_info(aid_list)
        >>> result = str(list(zip(*semantic_infotup))[1])
        >>> print(result)
        (UUID('d8903434-942f-e0f5-d6c2-0dcbe3137bf7'), ((0, 0), (1035, 0), (1035, 576), (0, 576)), 0.0, None, u'easy', u'zebra_plains')

    """
    # Semantic info depends on visual info
    if _visual_infotup is None:
        visual_infotup = get_annot_visual_uuid_info(ibs, aid_list)
    else:
        visual_infotup = _visual_infotup
    image_uuid_list, verts_list, theta_list = visual_infotup
    # It is visual info augmented with name and species
    yaw_list        = ibs.get_annot_yaws(aid_list)
    name_list       = ibs.get_annot_names(aid_list)
    species_list    = ibs.get_annot_species_texts(aid_list)
    semantic_infotup = (image_uuid_list, verts_list, theta_list, yaw_list,
                        name_list, species_list)
    return semantic_infotup


@register_ibs_method
@accessor_decors.dev_cache_invalidator(const.ANNOTATION_TABLE, ANNOT_SEMANTIC_UUID, native_rowids=True)
def update_annot_semantic_uuids(ibs, aid_list, _visual_infotup=None):
    """ Updater for semantic uuids """
    from ibeis.model.preproc import preproc_annot
    semantic_infotup = ibs.get_annot_semantic_uuid_info(aid_list, _visual_infotup)
    annot_semantic_uuid_list = preproc_annot.make_annot_semantic_uuid(semantic_infotup)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_SEMANTIC_UUID,), annot_semantic_uuid_list, aid_list)


@register_ibs_method
@accessor_decors.dev_cache_invalidator(const.ANNOTATION_TABLE, ANNOT_VISUAL_UUID, native_rowids=True)
def update_annot_visual_uuids(ibs, aid_list):
    """ Updater for visual uuids """
    from ibeis.model.preproc import preproc_annot
    visual_infotup = ibs.get_annot_visual_uuid_info(aid_list)
    annot_visual_uuid_list = preproc_annot.make_annot_visual_uuid(visual_infotup)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_VISUAL_UUID,), annot_visual_uuid_list, aid_list)
    # If visual uuids are changes semantic ones are also changed
    ibs.update_annot_semantic_uuids(aid_list, _visual_infotup=visual_infotup)


#### SETTERS ###

@register_ibs_method
@setter
def set_annot_bboxes(ibs, aid_list, bbox_list, delete_thumbs=True):
    """
    Sets bboxes of a list of annotations by aid,

    Args:
        aid_list (list of rowids): list of annotation rowids
        bbox_list (list of (x, y, w, h)): new bounding boxes for each aid

    Note:
        set_annot_bboxes is a proxy for set_annot_verts
    """
    from vtool import geometry
    # changing the bboxes also changes the bounding polygon
    vert_list = geometry.verts_list_from_bboxes_list(bbox_list)
    # naively overwrite the bounding polygon with a rectangle - for now trust the user!
    ibs.set_annot_verts(aid_list, vert_list, delete_thumbs=delete_thumbs)


@register_ibs_method
@setter
def set_annot_detect_confidence(ibs, aid_list, confidence_list):
    """ Sets annotation notes """
    id_iter = ((aid,) for aid in aid_list)
    val_iter = ((confidence,) for confidence in confidence_list)
    ibs.db.set(const.ANNOTATION_TABLE, ('annot_detect_confidence',), val_iter, id_iter)


@register_ibs_method
@setter
@accessor_decors.dev_cache_invalidator(const.ANNOTATION_TABLE, ANNOT_EXEMPLAR_FLAG, native_rowids=True)
def set_annot_exemplar_flags(ibs, aid_list, flag_list):
    """ Sets if an annotation is an exemplar """
    id_iter = ((aid,) for aid in aid_list)
    val_iter = ((flag,) for flag in flag_list)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_EXEMPLAR_FLAG,), val_iter, id_iter)


@register_ibs_method
@setter
#@cache_invalidator(const.NAME_TABLE)
@accessor_decors.dev_cache_invalidator(const.ANNOTATION_TABLE, NAME_ROWID, native_rowids=True)
def set_annot_name_rowids(ibs, aid_list, name_rowid_list):
    """ name_rowid_list -> annot.name_rowid[aid_list]

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
    #ibsfuncs.assert_lblannot_rowids_are_type(ibs, name_rowid_list, ibs.lbltype_ids[const.INDIVIDUAL_KEY])
    id_iter = aid_list
    colnames = (NAME_ROWID,)
    # WE NEED TO PERFORM A SPECIAL CHECK. ANY ANIMAL WHICH IS GIVEN AN UNKONWN
    # NAME MUST HAVE ITS EXEMPLAR FLAG SET TO FALSE
    will_be_unknown_flag_list = [nid == const.UNKNOWN_NAME_ROWID for nid in name_rowid_list]
    if any(will_be_unknown_flag_list):
        # remove exemplar status from any annotations that will become unknown
        will_be_unknown_aids = ut.filter_items(aid_list, will_be_unknown_flag_list)
        ibs.set_annot_exemplar_flags(will_be_unknown_aids, [False] * len(will_be_unknown_aids))
    ibs.db.set(const.ANNOTATION_TABLE, colnames, name_rowid_list, id_iter)
    # postset nids
    ibs.update_annot_semantic_uuids(aid_list)


@register_ibs_method
@setter
def set_annot_names(ibs, aid_list, name_list):
    """
    Sets the attrlbl_value of type(INDIVIDUAL_KEY) Sets names/nids of a
    list of annotations.

    CommandLine:
        python -m ibeis.control.manual_annot_funcs --test-set_annot_names --enableall

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
    name_rowid_list = ibs.add_names(name_list)
    ibs.set_annot_name_rowids(aid_list, name_rowid_list)
    #ibs.update_annot_semantic_uuids(aid_list) # set annot_name_rowids does this


@register_ibs_method
@setter
def set_annot_species(ibs, aid_list, species_text_list):
    """
    Sets species/speciesids of a list of annotations.
    Convenience function for set_annot_lblannot_from_value
    """
    #ibs.get_nids_from_text
    species_rowid_list = ibs.get_species_rowids_from_text(species_text_list)
    ibs.set_annot_species_rowids(aid_list, species_rowid_list)
    #ibs.set_annot_lblannot_from_value(aid_list, species_text_list, const.SPECIES_KEY)
    ibs.update_annot_semantic_uuids(aid_list)


@register_ibs_method
@setter
@accessor_decors.dev_cache_invalidator(const.ANNOTATION_TABLE, SPECIES_ROWID, native_rowids=True)
def set_annot_species_rowids(ibs, aid_list, species_rowid_list):
    """ species_rowid_list -> annot.species_rowid[aid_list]

    Sets species/speciesids of a list of annotations.

    Args:
        aid_list
        species_rowid_list

    """
    #ibs.set_annot_lblannot_from_rowid(aid_list, speciesid_list, const.SPECIES_KEY)
    #ibsfuncs.assert_lblannot_rowids_are_type(ibs, species_rowid_list, ibs.lbltype_ids[const.SPECIES_KEY])
    id_iter = aid_list
    colnames = (SPECIES_ROWID,)
    ibs.db.set(const.ANNOTATION_TABLE, colnames, species_rowid_list, id_iter)


@register_ibs_method
@setter
def set_annot_notes(ibs, aid_list, notes_list):
    """ Sets annotation notes """
    id_iter = ((aid,) for aid in aid_list)
    val_iter = ((notes,) for notes in notes_list)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_NOTE,), val_iter, id_iter)


@register_ibs_method
@setter
def set_annot_parent_rowid(ibs, aid_list, parent_aid_list):
    """ Sets the annotation's parent aid """
    id_iter = ((aid,) for aid in aid_list)
    val_iter = ((parent_aid,) for parent_aid in parent_aid_list)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_PARENT_ROWID,), val_iter, id_iter)


@register_ibs_method
@setter
def set_annot_thetas(ibs, aid_list, theta_list, delete_thumbs=True):
    """ Sets thetas of a list of chips by aid """
    id_iter = ((aid,) for aid in aid_list)
    val_list = ((theta,) for theta in theta_list)
    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_THETA,), val_list, id_iter)
    if delete_thumbs:
        ibs.delete_annot_chips(aid_list)  # Changing theta redefines the chips
    ibs.update_annot_visual_uuids(aid_list)


@register_ibs_method
@setter
def set_annot_verts(ibs, aid_list, verts_list, delete_thumbs=True):
    """ Sets the vertices [(x, y), ...] of a list of chips by aid """
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
    ibs.update_annot_visual_uuids(aid_list)


# PROBCHIP
# TODO: autogenerate probchip stuff

@register_ibs_method
@getter_1to1
def get_annot_probchip_fpaths(ibs, aid_list, qreq_=None):
    """
    Returns paths to probability images.
    """
    # FIXME: this is implemented very poorly. Caches not robust. IE they are
    # never invalidated. Not all config information is passed through
    from ibeis.model.preproc import preproc_probchip
    probchip_fpath_list = preproc_probchip.compute_and_write_probchip(ibs, aid_list, qreq_=qreq_)
    return probchip_fpath_list


# ---
# NEW
# ---

@register_ibs_method
#@accessor_decors.cache_getter(const.ANNOTATION_TABLE, ANNOT_QUALITY)
@getter_1to1
def get_annot_qualities(ibs, aid_list, eager=True):
    """ annot_quality_list <- annot.annot_quality[aid_list]

    gets data from the "native" column "annot_quality" in the "annot" table

    Args:
        aid_list (list):

    Returns:
        list: annot_quality_list

    TemplateInfo:
        Tgetter_table_column
        col = annot_quality
        tbl = annot

    SeeALso:
        ibeis.const.QUALITY_INT_TO_TEXT

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control._autogen_annot_funcs import *  # NOQA
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
#@accessor_decors.cache_invalidator(const.ANNOTATION_TABLE, ANNOT_QUALITY, native_rowids=True)
def set_annot_qualities(ibs, aid_list, annot_quality_list):
    """ annot_quality_list -> annot.annot_quality[aid_list]

    A quality is an integer representing the following types:

    Args:
        aid_list
        annot_quality_list

    SeeAlso:
        ibeis.const.QUALITY_INT_TO_TEXT

    TemplateInfo:
        Tsetter_native_column
        tbl = annot
        col = annot_quality
    """
    id_iter = aid_list
    colnames = (ANNOT_QUALITY,)
    ibs.db.set(const.ANNOTATION_TABLE, colnames, annot_quality_list, id_iter)


@register_ibs_method
@getter_1to1
def get_annot_quality_texts(ibs, aid_list):
    quality_list = ibs.get_annot_qualities(aid_list)
    quality_text_list = ut.dict_take(const.QUALITY_INT_TO_TEXT, quality_list)
    return quality_text_list


@register_ibs_method
@getter_1to1
def get_annot_isjunk(ibs, aid_list):
    qual_list = ibs.get_annot_qualities(aid_list)
    isjunk_list = [qual == const.QUALITY_TEXT_TO_INT['junk'] for qual in qual_list]
    return isjunk_list


@register_ibs_method
@getter_1to1
def get_annot_yaw_texts(ibs, aid_list):
    yaw_list = ibs.get_annot_yaws(aid_list)
    yaw_text_list = ibsfuncs.get_yaw_viewtexts(yaw_list)
    return yaw_text_list


@register_ibs_method
def set_annot_quality_texts(ibs, aid_list, quality_text_list):
    if not ut.isiterable(aid_list):
        aid_list = [aid_list]
    if isinstance(quality_text_list, six.string_types):
        quality_text_list = [quality_text_list]
    quality_list = ut.dict_take(const.QUALITY_TEXT_TO_INT, quality_text_list)
    ibs.set_annot_qualities(aid_list, quality_list)


@register_ibs_method
def set_annot_yaw_texts(ibs, aid_list, yaw_text_list):
    if not ut.isiterable(aid_list):
        aid_list = [aid_list]
    if isinstance(yaw_text_list, six.string_types):
        yaw_text_list = [yaw_text_list]
    yaw_list = ut.dict_take(const.VIEWTEXT_TO_YAW_RADIANS, yaw_text_list)
    ibs.set_annot_yaws(aid_list, yaw_list)


#==========
# Testdata
#==========

def testdata_ibs():
    import ibeis
    ibs = ibeis.opendb('testdb1')
    qreq_ = None
    return ibs, qreq_


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.manual_annot_funcs
        python -m ibeis.control.manual_annot_funcs --allexamples
        python -m ibeis.control.manual_annot_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
