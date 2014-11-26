"""
The goal of this module is to offload annotation work from the controller into a
single place.


CommandLine Help for manual controller functions

python ibeis/control/template_generator.py --tbls annotations --Tcfg with_getters:True with_native:True strip_docstr:True

python ibeis/control/template_generator.py --tbls annotations --Tcfg with_getters:True strip_docstr:True with_columns:False --quiet
python ibeis/control/template_generator.py --tbls annotations --Tcfg with_getters:True strip_docstr:False with_columns:False


# python -c "import utool; utool.write_to('agenhelp.bat', 'python ibeis/control/template_generator.py %*')

# Cross platform alias helper
python -c "import utool as ut; ut.write_to('Tgen.sh', 'python ibeis/control/template_generator.py $@')"

Tgen.sh --tbls annotations --Tcfg with_getters:True strip_docstr:False with_columns:False

"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, range, filter  # NOQA
import utool as ut  # NOQA
import uuid
from vtool import geometry
from ibeis import constants
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[preproc_annot]', DEBUG=False)


def generate_annot_properties(ibs, gid_list, bbox_list=None, theta_list=None,
                              species_list=None, nid_list=None, name_list=None,
                              detect_confidence_list=None, notes_list=None,
                              vert_list=None, annot_uuid_list=None,
                              viewpoint_list=None, quiet_delete_thumbs=False):
    #annot_uuid_list = ibsfuncs.make_annotation_uuids(image_uuid_list, bbox_list,
    #                                                      theta_list, deterministic=False)
    image_uuid_list = ibs.get_image_uuids(gid_list)
    if annot_uuid_list is None:
        annot_uuid_list = [uuid.uuid4() for _ in range(len(image_uuid_list))]
    # Prepare the SQL input
    assert name_list is None or nid_list is None, 'cannot specify both names and nids'
    # For import only, we can specify both by setting import_override to True
    assert bool(bbox_list is None) != bool(vert_list is None), 'must specify exactly one of bbox_list or vert_list'

    if theta_list is None:
        theta_list = [0.0 for _ in range(len(gid_list))]
    if name_list is not None:
        nid_list = ibs.add_names(name_list)
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

    # Build ~~deterministic?~~ random and unique ANNOTATION ids
    image_uuid_list = ibs.get_image_uuids(gid_list)
    #annot_uuid_list = ibsfuncs.make_annotation_uuids(image_uuid_list, bbox_list,
    #                                                      theta_list, deterministic=False)
    if annot_uuid_list is None:
        annot_uuid_list = [uuid.uuid4() for _ in range(len(image_uuid_list))]
    if viewpoint_list is None:
        viewpoint_list = [-1.0] * len(image_uuid_list)
    nVert_list = [len(verts) for verts in vert_list]
    vertstr_list = [constants.__STR__(verts) for verts in vert_list]
    xtl_list, ytl_list, width_list, height_list = list(zip(*bbox_list))
    assert len(nVert_list) == len(vertstr_list)
    # Define arguments to insert


def get_annot_visual_uuid_info(ibs, aid_list):
    """
    Returns annotation UUID that is unique for the visual qualities
    of the annoation. does not include name ore species information.

    get_annot_visual_uuid_info

    Args:
        ibs      (IBEISController):
        aid_list (list):

    Returns:
        ?: annot_appearance_determenistic_info_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_annot import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> visual_info_list = get_annot_visual_uuid_info(ibs, aid_list)
        >>> result = str(visual_info_list[0])
        >>> print(result)
        (UUID('66ec193a-1619-b3b6-216d-1784b4833b61'), ((0, 0), (1047, 0), (1047, 715), (0, 715)), 0.0, None)
    """
    image_uuid_list = ibs.get_annot_image_uuids(aid_list)
    verts_list      = ibs.get_annot_verts(aid_list)
    theta_list      = ibs.get_annot_thetas(aid_list)
    view_list       = ibs.get_annot_viewpoints(aid_list)
    visual_info_iter = zip(image_uuid_list, verts_list, theta_list, view_list)
    visual_info_list = list(visual_info_iter)
    return visual_info_list


def get_annot_semantic_uuid_info(ibs, aid_list):
    """
    Args:
        ibs      (IBEISController):
        aid_list (list):

    Returns:
        list: semantic_info_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_annot import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> semantic_info_list = get_annot_semantic_uuid_info(ibs, aid_list)
        >>> result = str(semantic_info_list[0])
        >>> print(result)
        (UUID('66ec193a-1619-b3b6-216d-1784b4833b61'), ((0, 0), (1047, 0), (1047, 715), (0, 715)), 0.0, None, '____', u'zebra_plains')

    """
    # Semantic info depends on visual info
    visual_info_list = get_annot_visual_uuid_info(ibs, aid_list)
    # It is visual info augmented with name and species
    name_list       = ibs.get_annot_names(aid_list)
    species_list    = ibs.get_annot_species(aid_list)
    aug_info_list = zip(name_list, species_list)
    # Perform augmentation
    semantic_info_iter = (visinfo + auginfo for visinfo, auginfo in zip(visual_info_list, aug_info_list))
    semantic_info_list = list(semantic_info_iter)
    return semantic_info_list


def make_annot_visual_uuid(ibs, aid_list):
    """
    make_annot_visual_uuid

    Args:
        ibs      (IBEISController):
        aid_list (list):

    Returns:
        list: annot_visual_uuid_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_annot import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> annot_visual_uuid_list = make_annot_visual_uuid(ibs, aid_list)
        >>> result = str(annot_visual_uuid_list[0])
        >>> print(result)
        76de0416-7c92-e1b3-4a17-25df32e9c2b4
    """
    get_annot_visual_uuid_info(ibs, aid_list)
    determenistic_info_list = get_annot_visual_uuid_info(ibs, aid_list)
    annot_visual_uuid_list = [ut.augment_uuid(*tup) for tup in determenistic_info_list]
    return annot_visual_uuid_list


def make_annot_semeantic_uuid(ibs, aid_list):
    """
    make_annot_semeantic_uuid

    Args:
        ibs      (IBEISController):
        aid_list (list):

    Returns:
        list: annot_semantic_uuid_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_annot import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> annot_semantic_uuid_list = make_annot_semeantic_uuid(ibs, aid_list)
        >>> result = str(annot_semantic_uuid_list[0])
        >>> print(result)
        215ab5f9-fe53-d7d1-59b8-d6b5ce7e6ca6
    """
    get_annot_visual_uuid_info(ibs, aid_list)
    determenistic_info_list = get_annot_semantic_uuid_info(ibs, aid_list)
    annot_semantic_uuid_list = [ut.augment_uuid(*tup) for tup in determenistic_info_list]
    return annot_semantic_uuid_list


def make_annotation_uuids(image_uuid_list, bbox_list, theta_list, deterministic=True):
    try:
        # Check to make sure bbox input is a tuple-list, not a list-list
        if len(bbox_list) > 0:
            try:
                assert isinstance(bbox_list[0], tuple), 'Bounding boxes must be tuples of ints!'
                assert isinstance(bbox_list[0][0], int), 'Bounding boxes must be tuples of ints!'
            except AssertionError as ex:
                ut.printex(ex)
                print('bbox_list = %r' % (bbox_list,))
                raise
        annotation_uuid_list = [ut.augment_uuid(img_uuid, bbox, theta)
                                for img_uuid, bbox, theta
                                in zip(image_uuid_list, bbox_list, theta_list)]
        if not deterministic:
            # Augment determenistic uuid with a random uuid to ensure randomness
            # (this should be ensured in all hardward situations)
            annotation_uuid_list = [ut.augment_uuid(ut.random_uuid(), _uuid)
                                    for _uuid in annotation_uuid_list]
    except Exception as ex:
        ut.printex(ex, 'Error building annotation_uuids', '[add_annot]',
                      key_list=['image_uuid_list'])
        raise
    return annotation_uuid_list


def test_annotation_uuid(ibs):
    """ Consistency test """
    aid_list        = ibs.get_valid_aids()
    bbox_list       = ibs.get_annot_bboxes(aid_list)
    theta_list      = ibs.get_annot_thetas(aid_list)
    image_uuid_list = ibs.get_annot_image_uuids(aid_list)

    annotation_uuid_list1 = ibs.get_annot_uuids(aid_list)
    annotation_uuid_list2 = make_annotation_uuids(image_uuid_list, bbox_list, theta_list)

    assert annotation_uuid_list1 == annotation_uuid_list2


def schema_1_2_0_postprocess_fixuuids(ibs):
    """
    schema_1_2_0_postprocess_fixuuids

    Args:
        ibs (IBEISController):

    Example:
        >>> from ibeis.model.preproc.preproc_annot import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> result = schema_1_2_0_postprocess_fixuuids(ibs)
        >>> print(result)
    """
    aid_list = ibs.get_valid_aids()
    #ibs.get_annot_nids(aid_list)
    #ANNOT_PARENT_ROWID      = 'annot_parent_rowid'
    #ANNOT_ROWID             = 'annot_rowid'
    ANNOT_SEMANTIC_UUID     = 'annot_semantic_uuid'
    ANNOT_VISUAL_UUID       = 'annot_visual_uuid'
    NAME_ROWID              = 'name_rowid'
    SPECIES_ROWID           = 'species_rowid'

    def get_nid_from_lblannot_relation(ibs, aid_list):
        from ibeis import ibsfuncs
        alrids_list = ibs.get_annot_alrids_oftype(aid_list, ibs.lbltype_ids[constants.INDIVIDUAL_KEY])
        lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
        # Get a single nid from the list of lblannot_rowids of type INDIVIDUAL
        # TODO: get index of highest confidence name
        nid_list_ = [lblannot_rowids[0] if len(lblannot_rowids) else ibs.UNKNOWN_LBLANNOT_ROWID for
                     lblannot_rowids in lblannot_rowids_list]
        return nid_list_

    def get_annot_name_rowid(ibs, aid_list):
        id_iter = aid_list
        colnames = (NAME_ROWID,)
        name_rowid_list = ibs.db.get(constants.ANNOTATION_TABLE, colnames, id_iter, id_colname='rowid')
        return name_rowid_list

    def set_annot_semantic_uuids(ibs, aid_list, annot_semantic_uuid_list):
        id_iter = aid_list
        colnames = (ANNOT_SEMANTIC_UUID,)
        ibs.db.set(constants.ANNOTATION_TABLE, colnames,
                   annot_semantic_uuid_list, id_iter)

    def set_annot_species_rowids(ibs, aid_list, species_rowid_list):
        id_iter = aid_list
        colnames = (SPECIES_ROWID,)
        ibs.db.set(
            constants.ANNOTATION_TABLE, colnames, species_rowid_list, id_iter)

    def set_annot_visual_uuids(ibs, aid_list, annot_visual_uuid_list):
        id_iter = aid_list
        colnames = (ANNOT_VISUAL_UUID,)
        ibs.db.set(constants.ANNOTATION_TABLE, colnames,
                   annot_visual_uuid_list, id_iter)

    def set_annot_name_rowids(ibs, aid_list, name_rowid_list):
        id_iter = aid_list
        colnames = (NAME_ROWID,)
        ibs.db.set(constants.ANNOTATION_TABLE, colnames, name_rowid_list, id_iter)

    ibs.print_annotation_table(verbosity=1)

    # Move old name rowids to the new columns
    nid_list = get_nid_from_lblannot_relation(ibs, aid_list)
    nid_list1_ = get_annot_name_rowid(ibs, aid_list)  # NOQA
    annot_semantic_uuid_list = make_annot_semeantic_uuid(ibs, aid_list)  # NOQA
    annot_visual_uuid_list = make_annot_visual_uuid(ibs, aid_list)  # NOQA

    set_annot_name_rowids(ibs, aid_list, nid_list)
    nid_list2_ = get_annot_name_rowid(ibs, aid_list)  # NOQA

    #ibs.db.set(constants.ANNOTATION_TABLE,)

    ibs.print_annotation_table(verbosity=1)


if __name__ == '__main__':
    """
    CommandLine:
        python ibeis/control/template_generator.py --tbls annotations --Tflags getters native

        python -c "import utool, ibeis.model.preproc.preproc_annot; utool.doctest_funcs(ibeis.model.preproc.preproc_annot, allexamples=True)"
        python -m ibeis.model.preproc.preproc_annot
        python -m ibeis.model.preproc.preproc_annot --allexamples
        python -m ibeis.model.preproc.preproc_annot --allexamples --noface --nosrc

    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
