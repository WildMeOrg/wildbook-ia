#!/usr/bin/env python2.7
"""
Exports subset of an IBEIS database to a new IBEIS database
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.export.export_subset))"
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from collections import namedtuple
import utool
from ibeis.constants import (IMAGE_TABLE, ANNOTATION_TABLE, LBLANNOT_TABLE,
                             AL_RELATION_TABLE, __STR__)
from vtool import geometry

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
        ibs.get_image_abs_uri(_gid_list),
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
    >>> from ibeis.all_imports import *
    >>> from ibeis.export.export_subset import *
    >>> import utool
    >>> ibs = ibs_src = ibeis.opendb('testdb1')
    >>> aid_list = ibs.get_valid_aids()[-3:]
    >>> gid_list = ibs.get_valid_gids()[0:2]
    >>> include_image_annots = True
    >>> transfer_data = collect_transfer_data(ibs_src, gid_list, aid_list, include_image_annots)
    >>> print(utool.hashstr(transfer_data))
    ad20ozek356das0m
    """
    ibs = ibs_src
    if include_image_annots:
        # include all annotations from input images
        _img_aid_list = utool.flatten(ibs.get_image_aids(gid_list))
        _aid_list = list(set(aid_list + _img_aid_list))
    else:
        # do not include all annotations from input images
        _aid_list = aid_list
    # always include images from annotations
    _annot_gid_list = ibs.get_annot_gids(_aid_list)
    _gid_list = list(set(gid_list + _annot_gid_list))
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


def transfer_data(ibs_src, ibs_dst, gid_list1=None, aid_list1=None,
                  include_image_annots=True):
    """
    >>> from ibeis.all_imports import *
    >>> from ibeis.export.export_subset import *
    >>> ibs1 = ibs_src = ibeis.opendb('testdb1')
    >>> _aid_list1 = gid_list1 = ibs1.get_valid_aids()
    >>> _gid_list1 = aid_list1 = ibs1.get_valid_gids()
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
    >>> transfer_data(ibs_src, ibs_dst, gid_list1=gid_list1, aid_list1=aid_list1)
    """
    ibs1, ibs2 = ibs_src, ibs_dst

    _aid_list1 = [] if aid_list1 is None else aid_list1
    _gid_list1 = [] if gid_list1 is None else gid_list1

    # Get information from source database
    transfer_data = collect_transfer_data(ibs1, _aid_list1, _gid_list1, include_image_annots)
    img_td      = transfer_data.img_td
    annot_td    = transfer_data.annot_td
    lblannot_td = transfer_data.lblannot_td
    alr_td      = transfer_data.alr_td

    # Add information to destination database
    gid_list2            = internal_add_images(ibs2, img_td)
    aid_list2            = internal_add_annots(ibs2, annot_td)
    lblannot_rowid_list2 = internal_add_lblannot(ibs2, lblannot_td)
    alr_rowid_list2      = internal_add_alr(ibs2, alr_td)

    return (gid_list2, aid_list2, lblannot_rowid_list2, alr_rowid_list2)


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
    (lblannot_uuid_list,
     value_list,
     note_list,
     lbltype_text_list) = lblannot_td
    lbltype_rowid_list  = ibs.get_lbltype_rowid_from_text(lbltype_text_list)
    # Pack into params iter
    colnames = ('lblannot_uuid', 'lblannot_value', 'lblannot_note', 'lbltype_rowid',)
    params_iter = list(zip(lblannot_uuid_list, value_list, note_list, lbltype_rowid_list,))
    get_rowid_from_superkey = ibs.get_lblannot_rowid_from_superkey
    superkey_paramx = (1, 2)
    lblannot_rowid_list = ibs.db.add_cleanly(LBLANNOT_TABLE, colnames, params_iter,
                                               get_rowid_from_superkey, superkey_paramx)
    return lblannot_rowid_list


def internal_add_alr(ibs, alr_td):
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
