"""
The goal of this module is to offload annotation work from the controller into a
single place.

CommandLine Help for manual controller functions

# Cross platform alias helper
python -c "import utool as ut; ut.write_to('Tgen.sh', 'python -m ibeis.control.template_generator $@')"

Tgen.sh --tbls annotations --Tcfg with_getters:True strip_docstr:False with_columns:False
Tgen.sh --tbls annotations --Tcfg with_getters:True with_native:True strip_docstr:True
Tgen.sh --tbls annotations --Tcfg with_getters:True strip_docstr:True with_columns:False --quiet
Tgen.sh --tbls annotations --Tcfg with_getters:True strip_docstr:False with_columns:False
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, range, filter  # NOQA
import utool as ut  # NOQA
import uuid
from vtool import geometry
from ibeis import constants as const
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[preproc_annot]')


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
    assert name_list is None or nid_list is None, (
        'cannot specify both names and nids')
    # For import only, we can specify both by setting import_override to True
    assert bool(bbox_list is None) != bool(vert_list is None), (
        'must specify exactly one of bbox_list or vert_list')

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
    vertstr_list = [const.__STR__(verts) for verts in vert_list]
    xtl_list, ytl_list, width_list, height_list = list(zip(*bbox_list))
    assert len(nVert_list) == len(vertstr_list)
    # Define arguments to insert


def make_annot_visual_uuid(visual_infotup):
    """
    Args:
        visual_infotup (tuple):  (image_uuid_list, verts_list, theta_list)

    Returns:
        list: annot_visual_uuid_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_annot import *  # NOQA
        >>> ibs, aid_list = testdata_preproc_annot()
        >>> visual_infotup = ibs.get_annot_visual_uuid_info(aid_list)
        >>> annot_visual_uuid_list = make_annot_visual_uuid(visual_infotup)
        >>> result = str(annot_visual_uuid_list[0])
        >>> print(result)
        8687dcb6-1f1f-fdd3-8b72-8f36f9f41905
    """
    assert len(visual_infotup) == 3, 'len=%r' % (len(visual_infotup),)
    annot_visual_uuid_list = [ut.augment_uuid(*tup) for tup in zip(*visual_infotup)]
    return annot_visual_uuid_list


def make_annot_semantic_uuid(semantic_infotup):
    """

    Args:
        semantic_infotup (tuple): (image_uuid_list, verts_list, theta_list, view_list, name_list, species_list)

    Returns:
        list: annot_semantic_uuid_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.preproc.preproc_annot import *  # NOQA
        >>> ibs, aid_list = testdata_preproc_annot()
        >>> semantic_infotup = ibs.get_annot_semantic_uuid_info(aid_list)
        >>> annot_semantic_uuid_list = make_annot_semantic_uuid(semantic_infotup)
        >>> result = str(annot_semantic_uuid_list[0])
        >>> print(result)
        215ab5f9-fe53-d7d1-59b8-d6b5ce7e6ca6
    """
    assert len(semantic_infotup) == 6, 'len=%r' % (len(semantic_infotup),)
    annot_semantic_uuid_list = [ut.augment_uuid(*tup) for tup in zip(*semantic_infotup)]
    return annot_semantic_uuid_list


def testdata_preproc_annot():
    import ibeis
    ibs = ibeis.opendb('testdb1')
    aid_list = ibs.get_valid_aids()
    return ibs, aid_list


def test_annotation_uuid(ibs):
    """ Consistency test """
    # DEPRICATE
    aid_list        = ibs.get_valid_aids()
    bbox_list       = ibs.get_annot_bboxes(aid_list)
    theta_list      = ibs.get_annot_thetas(aid_list)
    image_uuid_list = ibs.get_annot_image_uuids(aid_list)

    annotation_uuid_list1 = ibs.get_annot_uuids(aid_list)
    annotation_uuid_list2 = make_annotation_uuids(image_uuid_list, bbox_list, theta_list)

    assert annotation_uuid_list1 == annotation_uuid_list2


def distinguish_unknown_nids(ibs, aid_list, nid_list_):
    nid_list = [-aid if nid == ibs.UNKNOWN_LBLANNOT_ROWID else nid
                for nid, aid in zip(nid_list_, aid_list)]
    return nid_list


def postget_annot_verts(vertstr_list):
    # TODO: Sanatize input for eval
    #print('vertstr_list = %r' % (vertstr_list,))
    locals_ = {}
    globals_ = {}
    vert_list = [eval(vertstr, globals_, locals_) for vertstr in vertstr_list]
    return vert_list


def on_delete(ibs, aid_list):
    #ibs.delete_annot_relations(aid_list)
    # image thumbs are deleted in here too
    ibs.delete_annot_chips(aid_list)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.template_generator --tbls annotations --Tflags getters native

        python -m ibeis.model.preproc.preproc_annot
        python -m ibeis.model.preproc.preproc_annot --allexamples
        python -m ibeis.model.preproc.preproc_annot --allexamples --noface --nosrc

    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
