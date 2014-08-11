#!/usr/bin/env python2.7
"""
Exports subset of an IBEIS database to a new IBEIS database
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.export.export_subset))"
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function


def get_annot_transfer_data(ibs, _aid_list):
    _annot_gid_list = ibs.get_annot_gids(_aid_list)
    return {
        'annot_image_uuid_list'    : ibs.get_image_uuids(_annot_gid_list),
        'annot_uuid_list'          : ibs.get_annot_uuids(_aid_list),
        'annot_vert_list'          : ibs.get_annot_verts(_aid_list),
        'annot_notes_list'         : ibs.get_annot_notes(_aid_list),
        'annot_thetas_list'        : ibs.get_annot_thetas(_aid_list),
        'annot_exemplar_flag_list' : ibs.get_annot_exemplar_flag(_aid_list),
    }, _annot_gid_list


def get_image_transfer_data(ibs, _gid_list):
    return {
        'img_uuid_list'      : ibs.get_image_uuids(_gid_list),
        'img_uri_list'       : ibs.get_image_uris(_gid_list),
        'img_orig_name_list' : ibs.get_image_gnames(_gid_list),
        'img_gsize_list'     : ibs.get_image_sizes(_gid_list),
        'img_ext_list'       : ibs.get_image_exts(_gid_list),
        'img_latlon_list'    : ibs.get_image_gps(_gid_list),
        'img_unixtime_list'  : ibs.get_image_unixtime(_gid_list),
        'img_note_list'      : ibs.get_image_notes(_gid_list),
    }


def get_lblannot_transfer_data(ibs, _lblannot_rowid_list):
    return {
        #Label
        'lblannot_uuid_list'         : ibs.get_lblannot_uuids(_lblannot_rowid_list),
        'lblannot_value_list'        : ibs.get_lblannot_values(_lblannot_rowid_list),
        'lblannot_note_list'         : ibs.get_lblannot_notes(_lblannot_rowid_list),
        #Label type
        'lblannot_lbltype_text_list' : ibs.get_lbltype_text(_lblannot_rowid_list),
    }


def get_alr_transfer_data(ibs, _alr_rowid_list):
    _alr_lblannot_rowid_list = ibs.get_alr_lblannot_rowids(_alr_rowid_list)
    _alr_annot_rowid_list = ibs.get_alr_annot_rowids(_alr_rowid_list)
    return {
        'alr_lblannot_uuid_list': ibs.get_lblannot_uuids(_alr_lblannot_rowid_list),
        'alr_annot_uuid_list':    ibs.get_annot_uuids(_alr_annot_rowid_list),
    }


def collect_transfer_data(ibs_src, gid_list1=None, aid_list1=None):
    """
    >>> from ibeis.all_imports import *
    >>> from ibeis.export.export_subset import *
    >>> ibs1 = ibs_src = ibeis.opendb('testdb1')
    >>> print(ibs_src.get_infostr())
    >>> transfer_data(ibs_src, ibs_dst, gid_list1=gid_list1, aid_list1=aid_list1)
    """
    ibs1 = ibs_src
    if gid_list1 is None:
        gid_list1 = ibs1.get_valid_gids()

    #Annotations
    if aid_list1 is None:
        aid_list1 = ibs1.get_valid_aids()

    _lblannot_rowid_list1 = ibs1._get_all_lblannot_rowids()
    _lblannot_lbltype_rowid_list1 = ibs1.get_lblannot_lbltypes_rowids(_lblannot_rowid_list1)

    #Annotation label relationships
    _alr_rowid_list1 = ibs1._get_all_alr_rowids()

    #Images
    _gid_list1 = list(set(gid_list1 + _annot_gid_list1))



def transfer_data(ibs_src, ibs_dst, gid_list1=None, aid_list1=None):
    """
    >>> from ibeis.all_imports import *
    >>> from ibeis.export.export_subset import *
    >>> ibs1 = ibs_src = ibeis.opendb('testdb1')
    >>> ibs2 = ibs_dst = ibeis.opendb('testdb_dst', allow_newdir=True)
    >>> print(ibs_src.get_infostr())
    >>> print(ibs_dst.get_infostr())
    >>> gid_list1 = None
    >>> aid_list1 = None
    >>> transfer_data(ibs_src, ibs_dst, gid_list1=gid_list1, aid_list1=aid_list1)
    """
    ibs1 = ibs_src
    ibs2 = ibs_dst

    # Step 1: Collect data to transfer

    # Add information to destination database
    ibs2._internal_add_images(img_uuid_list1, img_uri_list1,
                              img_orig_name_list1, img_ext_list1,
                              img_gsize_list1, img_unixtime_list1,
                              img_latlon_list1, img_note_list1)

    ibs2._internal_add_annots(annot_image_uuid_list1, annot_uuid_list1,
                              annot_thetas_list1, annot_notes_list1,
                              annot_vert_list1)
