#!/usr/bin/env python2.7
"""
Exports subset of an IBEIS database to a new IBEIS database
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.io.export_subset))"
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from collections import namedtuple
import utool
from ibeis.constants import (IMAGE_TABLE, ANNOTATION_TABLE, LBLANNOT_TABLE,
                             AL_RELATION_TABLE, __STR__)
from vtool import geometry

# TODO: Write better doctests and ensure they pass


# TODO: Ensure that all relevant information from
# ibeis/ibeis/control/DB_Schema.py is accounted for in Transfer Data Structures

# Transfer data structures could become classes.

TransferData = namedtuple(
    'TransferData', (
        'img_td',
        'annot_td',
        'lblannot_td',
        'alr_td',
    ))


IMAGE_TransferData = namedtuple(
    'IMAGE_TransferData', (
        'img_uuid_list',
        'img_uri_list',
        'img_orig_name_list',
        'img_ext_list',
        'img_gsize_list',
        'img_unixtime_list',
        'img_latlon_list',
        'img_note_list',
    ))

ANNOTATION_TransferData = namedtuple(
    'ANNOTATION_TransferData', (
        'annot_image_uuid_list',
        'annot_uuid_list',
        'annot_vert_list',
        'annot_thetas_list',
        'annot_exemplar_flag_list',
        'annot_note_list',
    ))

LBLANNOT_TransferData = namedtuple(
    'LBLANNOT_TransferData', (
        'lblannot_uuid_list',
        'lblannot_value_list',
        'lblannot_note_list',
        #Label type
        'lblannot_lbltype_text_list',
    ))


AL_RELATION_TransferData = namedtuple(
    'AL_RELATION_TransferData', (
        'alr_annot_uuid_list',
        'alr_lblannot_uuid_list',
    ))


# TODO: Make sure the transfer data getters correspond with the namedtuple/class
# definitions

def get_annot_transfer_data(ibs, _aid_list, _annot_gid_list):
    # avoid sql call if possible
    #_annot_gid_list = ibs.get_annot_gids(_aid_list) if annot_gid_list is None else annot_gid_list
    annot_td = ANNOTATION_TransferData(
        ibs.get_image_uuids(_annot_gid_list),
        ibs.get_annot_uuids(_aid_list),
        ibs.get_annot_verts(_aid_list),
        ibs.get_annot_thetas(_aid_list),
        ibs.get_annot_exemplar_flag(_aid_list),
        ibs.get_annot_notes(_aid_list),
    )
    return annot_td


def get_image_transfer_data(ibs, _gid_list):
    img_td = IMAGE_TransferData(
        ibs.get_image_uuids(_gid_list),
        ibs.get_image_absolute_uri(_gid_list),
        ibs.get_image_gnames(_gid_list),
        ibs.get_image_exts(_gid_list),
        ibs.get_image_sizes(_gid_list),
        ibs.get_image_unixtime(_gid_list),
        ibs.get_image_gps(_gid_list),
        ibs.get_image_notes(_gid_list),
    )
    return img_td


def get_lblannot_transfer_data(ibs, _lblannot_rowid_list):
    _lblannot_lbltype_rowid_list = ibs.get_lblannot_lbltypes_rowids(_lblannot_rowid_list)
    lblannot_td = LBLANNOT_TransferData(
        ibs.get_lblannot_uuids(_lblannot_rowid_list),
        ibs.get_lblannot_values(_lblannot_rowid_list),
        ibs.get_lblannot_notes(_lblannot_rowid_list),
        ibs.get_lbltype_text(_lblannot_lbltype_rowid_list),
    )
    return lblannot_td


def get_alr_transfer_data(ibs, _alr_rowid_list):
    _alr_lblannot_rowid_list = ibs.get_alr_lblannot_rowids(_alr_rowid_list)
    _alr_annot_rowid_list = ibs.get_alr_annot_rowids(_alr_rowid_list)
    alr_td = AL_RELATION_TransferData(
        ibs.get_annot_uuids(_alr_annot_rowid_list),
        ibs.get_lblannot_uuids(_alr_lblannot_rowid_list),
    )
    return alr_td


def collect_transfer_data(ibs_src, gid_list, aid_list,
                          include_image_annots=True):
    """
    Packs all the data you are going to transfer from ibs_src
    info the transfer_data named tuple.

    >>> from ibeis.all_imports import *  # NOQA
    >>> from ibeis.io.export_subset import *  # NOQA
    >>> import utool
    >>> ibs = ibs_src = ibeis.opendb('testdb1')
    >>> aid_list = ibs.get_valid_aids()[-3:]
    >>> gid_list = ibs.get_valid_gids()[0:2]
    >>> include_image_annots = True
    >>> transfer_data = collect_transfer_data(ibs_src, gid_list, aid_list, include_image_annots)
    >>> print(utool.hashstr(transfer_data))
    rxirvfa08psrle&e
    """
    ibs = ibs_src
    if include_image_annots:
        # include all annotations from input images
        _img_aid_list = utool.flatten(ibs.get_image_aids(gid_list))
        _aid_list = list(set(aid_list + _img_aid_list))
    else:
        # do not include all annotations from input images
        _aid_list = aid_list
    # ALWAYS include images from annotations
    _annot_gid_list = ibs.get_annot_gids(_aid_list)
    _gid_list = sorted(list(set(gid_list + _annot_gid_list)))
    #_alr_rowid_list = ibs._get_all_alr_rowids()
    _alr_rowid_list = utool.flatten(ibs.get_annot_alrids(_aid_list))
    _lblannot_rowid_list = utool.flatten(ibs.get_annot_lblannot_rowids(_aid_list))

    # Get transfer data dicts
    img_td      = get_image_transfer_data(ibs, _gid_list)
    annot_td    = get_annot_transfer_data(ibs, _aid_list, _annot_gid_list)
    lblannot_td = get_lblannot_transfer_data(ibs, _lblannot_rowid_list)
    alr_td      = get_alr_transfer_data(ibs, _alr_rowid_list)
    transfer_data = TransferData(img_td, annot_td, lblannot_td, alr_td)
    return transfer_data


def test_conflicts():
    """
    just a function which has a
    standalone doctest

    >>> from ibeis.all_imports import *  # NOQA
    >>> from ibeis.io.export_subset import *  # NOQA
    >>> ibs1 = ibs_src = ibeis.opendb('testdb1')
    >>> _gid_list1 = gid_list1 = ibs1.get_valid_gids()[1:10:2]
    >>> _aid_list1 = aid_list1 = []
    >>> include_image_annots = True
    >>> ibs2 = ibs_dst = ibeis.opendb('testdb_dst', allow_newdir=True, delete_ibsdir=True)
    >>> print(ibs_src.get_infostr())  # doctest: +ELLIPSIS
    <...
    dbname = 'testdb1'
    num_images = 13
    num_annotations = 13
    num_names = 7
    >>> print(ibs_dst.get_infostr())  # doctest: +ELLIPSIS
    <...
    dbname = 'testdb_dst'
    num_images = 0
    num_annotations = 0
    num_names = 0
    """
    pass


def execute_transfer(ibs_src, ibs_dst, gid_list1=None, aid_list1=None,
                     include_image_annots=True):
    """
    Collects data specified for transfer from ibs_src and moves it into ibs_dst.
    UUIDS are checked for conflicts

    >>> from ibeis.all_imports import *  # NOQA
    >>> from ibeis.io.export_subset import *  # NOQA
    >>> ibs1 = ibs_src = ibeis.opendb('testdb1')
    >>> _gid_list1 = gid_list1 = ibs1.get_valid_gids()[1:10:2]
    >>> _aid_list1 = aid_list1 = []
    >>> include_image_annots = True
    >>> ibs2 = ibs_dst = ibeis.opendb('testdb_dst', allow_newdir=True, delete_ibsdir=True)
    >>> execute_transfer(ibs_src, ibs_dst, gid_list1=gid_list1, aid_list1=aid_list1)
    >>> _gid_list1 = gid_list1 = ibs1.get_valid_gids()[1:9]
    >>> execute_transfer(ibs_src, ibs_dst, gid_list1=gid_list1, aid_list1=aid_list1)
    >>> print(ibs_dst.get_infostr())
    """

    _aid_list1 = [] if aid_list1 is None else aid_list1
    _gid_list1 = [] if gid_list1 is None else gid_list1

    assert all([gid is not None for gid in _gid_list1]), 'bad gid input'
    assert all([aid is not None for aid in _aid_list1]), 'bad aid input'

    # Get information from source database
    """
    <TEST>
    ibs = ibs_src
    aid_list = _aid_list1
    gid_list = _gid_list1
    """
    transfer_data = collect_transfer_data(ibs_src, _gid_list1, _aid_list1, include_image_annots)
    img_td      = transfer_data.img_td
    annot_td    = transfer_data.annot_td
    lblannot_td = transfer_data.lblannot_td
    alr_td      = transfer_data.alr_td

    # Ensure no merge conflicts
    check_conflicts(ibs_src, ibs_dst, transfer_data)

    # Add information to destination database
    gid_list2            = internal_add_images(ibs_dst, img_td)
    aid_list2            = internal_add_annots(ibs_dst, annot_td)
    lblannot_rowid_list2 = internal_add_lblannot(ibs_dst, lblannot_td)
    alr_rowid_list2      = internal_add_alr(ibs_dst, alr_td)

    return (gid_list2, aid_list2, lblannot_rowid_list2, alr_rowid_list2)


def check_conflicts(ibs_src, ibs_dst, transfer_data):
    """
    Check to make sure the destination database does not have any conflicts
    with the incoming transfer.

    Currently only checks that images do not have conflicting annotations.

    Does not check label consistency.
    """

    # TODO: Check label consistency: ie check that labels with the
    # same (type, value) should also have the same UUID
    img_td      = transfer_data.img_td
    #annot_td    = transfer_data.annot_td
    #lblannot_td = transfer_data.lblannot_td
    #alr_td      = transfer_data.alr_td

    image_uuid_list1 = img_td.img_uuid_list
    sameimg_gid_list2_ = ibs_dst.get_image_gids_from_uuid(image_uuid_list1)
    issameimg = [gid is not None for gid in sameimg_gid_list2_]
    # Check if databases contain the same images
    if any(issameimg):
        sameimg_gid_list2 = utool.filter_items(sameimg_gid_list2_, issameimg)
        sameimg_image_uuids = utool.filter_items(image_uuid_list1, issameimg)
        print('! %d/%d images are duplicates' % (len(sameimg_gid_list2), len(image_uuid_list1)))
        # Check if sameimg images in dst has any annotations.
        sameimg_aids_list2 = ibs_dst.get_image_aids(sameimg_gid_list2)
        hasannots = [len(aids) > 0 for aids in sameimg_aids_list2]
        if any(hasannots):
            # TODO: Merge based on some merge stratagy parameter (like annotation timestamp)
            sameimg_gid_list1 = ibs_src.get_image_gids_from_uuid(sameimg_image_uuids)
            hasannot_gid_list2 = utool.filter_items(sameimg_gid_list2, hasannots)
            hasannot_gid_list1 = utool.filter_items(sameimg_gid_list1, hasannots)
            print('  !! %d/%d of those have annotations' % (len(hasannot_gid_list2), len(sameimg_gid_list2)))
            # They had better be the same annotations!
            assert_images_have_same_annnots(ibs_src, ibs_dst, hasannot_gid_list1, hasannot_gid_list2)
            print('  ...phew, all of the annotations were the same.')
        #raise AssertionError('dst dataset contains some of this data')


def assert_images_have_same_annnots(ibs_src, ibs_dst, hasannot_gid_list1, hasannot_gid_list2):
    """ Given a list of gids from each ibs, this function asserts that every
        annontation in gid1 is the same as every annontation in gid2
    """
    from ibeis.ibsfuncs import unflat_map
    hasannot_aids_list1 = ibs_src.get_image_aids(hasannot_gid_list1)
    hasannot_aids_list2 = ibs_dst.get_image_aids(hasannot_gid_list2)
    hasannot_auuids_list1 = unflat_map(ibs_src.get_annot_uuids, hasannot_aids_list1)
    hasannot_auuids_list2 = unflat_map(ibs_dst.get_annot_uuids, hasannot_aids_list2)
    hasannot_verts_list1 = unflat_map(ibs_src.get_annot_verts, hasannot_aids_list1)
    hasannot_verts_list2 = unflat_map(ibs_dst.get_annot_verts, hasannot_aids_list2)
    assert_same_annot_uuids(hasannot_auuids_list1, hasannot_auuids_list2)
    assert_same_annot_verts(hasannot_verts_list1, hasannot_verts_list2)  # hack, check verts as well


def assert_same_annot_uuids(hasannot_auuids_list1, hasannot_auuids_list2):
    uuids_pair_iter = zip(hasannot_auuids_list1, hasannot_auuids_list2)
    msg = ('The {count}-th image has inconsistent annotation:. '
           'auuids1={auuids1} auuids2={auuids2}')
    for count, (auuids1, auuids2) in enumerate(uuids_pair_iter):
        assert auuids1 == auuids2, msg.format(
            count=count, auuids1=auuids1, auuids2=auuids2,)


def assert_same_annot_verts(hasannot_verts_list1, hasannot_verts_list2):
    verts_pair_iter = zip(hasannot_verts_list1, hasannot_verts_list2)
    msg = ('The {count}-th image has inconsistent annotation:. '
           'averts1={averts1} averts2={averts2}')
    for count, (averts1, averts2) in enumerate(verts_pair_iter):
        assert averts1 == averts2, msg.format(
            count=count, averts1=averts1, averts2=averts2,)


# TODO: make sure the interal adders take into account all information specified
# in transferdata objects

#@adder
def internal_add_images(ibs, img_td):
    # Unpack transfer data
    (uuid_list,
     uri_list,
     original_name_list,
     ext_list,
     gsize_list,
     unixtime_list,
     latlon_list,
     notes_list) = img_td

    colnames = ('image_uuid', 'image_uri', 'image_original_name',
                'image_ext', 'image_width', 'image_height',
                'image_time_posix', 'image_gps_lat',
                'image_gps_lon', 'image_note',)

    params_list = [(uuid, uri, orig_name, ext, size[0], size[1], unixtime,
                    latlon[0], latlon[1], note)
                    for uuid, uri, orig_name, ext, size, unixtime, latlon, note in
                    zip(uuid_list, uri_list, original_name_list, ext_list,
                        gsize_list, unixtime_list, latlon_list, notes_list)]

    gid_list = ibs.db.add_cleanly(IMAGE_TABLE, colnames, params_list, ibs.get_image_gids_from_uuid)
    if ibs.cfg.other_cfg.auto_localize:
        # Move to ibeis database local cache
        ibs.localize_images(gid_list)
    return gid_list


#@adder
def internal_add_annots(ibs, annot_td):
    # Unpack transfer data
    (annot_img_uuid_list,
     annot_uuid_list,
     vert_list,
     theta_list,
     isexemplar_list,
     notes_list) = annot_td

    gid_list = ibs.get_image_gids_from_uuid(annot_img_uuid_list)
    bbox_list = geometry.bboxes_from_vert_list(vert_list)  # , castint=True)

    vertstr_list = map(__STR__, vert_list)
    nVert_list   = map(len, vert_list)
    xtl_list, ytl_list, width_list, height_list = list(zip(*bbox_list))
    # Define arguments to insert
    colnames = ('annot_uuid', 'image_rowid', 'annot_xtl', 'annot_ytl',
                'annot_width', 'annot_height', 'annot_theta', 'annot_num_verts',
                'annot_verts', 'annot_note',)

    params_iter = list(zip(annot_uuid_list, gid_list, xtl_list, ytl_list,
                            width_list, height_list, theta_list, nVert_list,
                            vertstr_list, notes_list))

    # Execute add ANNOTATIONs SQL
    aid_list = ibs.db.add_cleanly(ANNOTATION_TABLE, colnames, params_iter, ibs.get_annot_aids_from_uuid)

    # Invalidate image thumbnails
    ibs.delete_image_thumbs(gid_list)
    return aid_list


def internal_add_lblannot(ibs, lblannot_td):
    """
    Transfers annotation labels (like name, species, ...) by UUID (text)
    TODO: Check if values conflict
    """
    (lblannot_uuid_list,
     value_list,
     note_list,
     lbltype_text_list) = lblannot_td
    lbltype_rowid_list  = ibs.get_lbltype_rowid_from_text(lbltype_text_list)
    # Pack into params iter
    colnames = ('lblannot_uuid', 'lblannot_value', 'lblannot_note', 'lbltype_rowid',)
    params_iter = list(zip(lblannot_uuid_list, value_list, note_list, lbltype_rowid_list,))
    get_rowid_from_superkey = ibs.get_lblannot_rowid_from_typevaltup
    superkey_paramx = (3, 1)
    lblannot_rowid_list = ibs.db.add_cleanly(LBLANNOT_TABLE, colnames, params_iter,
                                             get_rowid_from_superkey, superkey_paramx)
    # FIXME: LabelAnnot UUIDS might not be copied properly if there is duplicate (value, type)
    # TODO: At least throw an assertion error when copying different label UUIDS
    # with the same (value, type), this should be done in the consistency checks.
    lblannot_uuid_list_test = ibs.get_lblannot_uuids(lblannot_rowid_list)
    assert all([uuid1 == uuid2 for uuid1, uuid2 in
                zip(lblannot_uuid_list, lblannot_uuid_list_test)]),\
            'issue occurred. alrtable will fail if FIXME not done. Same name with different uuid'
    return lblannot_rowid_list


def internal_add_alr(ibs, alr_td):
    """
    Transfers relationships between annotations and labels both by uuid.
    TODO: config-ids / confidences are not yet transfered


    """
    # Unpack transfer data
    (annot_uuid_list,
     lblannot_uuid_list) = alr_td
    # Convert to correct data
    annot_rowid_list    = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    lblannot_rowid_list = ibs.get_lblannot_rowid_from_uuid(lblannot_uuid_list)
    config_rowid_list   = [ibs.MANUAL_CONFIGID] * len(annot_rowid_list)  # FIXME
    alr_confidence_list = [0.0] * len(annot_rowid_list)  # FIXME
    # Pack into params iter
    colnames = ('annot_rowid', 'lblannot_rowid', 'config_rowid', 'alr_confidence',)
    params_iter = list(zip(annot_rowid_list, lblannot_rowid_list, config_rowid_list, alr_confidence_list))
    get_rowid_from_superkey = ibs.get_alrid_from_superkey
    #ibs.get_alrid_from_superkey(annot_rowid_list,
    superkey_paramx = (0, 1, 2)
    alrid_list = ibs.db.add_cleanly(AL_RELATION_TABLE, colnames, params_iter,
                                    get_rowid_from_superkey, superkey_paramx)
    return alrid_list
