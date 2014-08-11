#!/usr/bin/env python2.7
"""
Exports subset of an IBEIS database to a new IBEIS database
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.export.export_subset))"
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function


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

    #Annotations
    if aid_list1 is None:
        aid_list1 = ibs1.get_valid_aids()
    annot_uuid_list1 = ibs1.get_annot_uuids(aid_list1)
    annot_exemplar_flag_list1 = ibs1.get_annot_exemplar_flag(aid_list1)
    annot_vert_list1 = ibs1.get_annot_verts(aid_list1)
    annot_thetas_list1 = ibs1.get_annot_thetas(aid_list1)
    _annot_gid_list1 = ibs1.get_annot_gids(aid_list1)
    annot_image_uuid_list1 = ibs1.get_image_uuids(_annot_gid_list1)

    #Labels
    lblannot_rowid_list1 = ibs1._get_all_lblannot_rowids()
    lblannot_uuid_list1 = ibs1.get_lblannot_uuids(lblannot_rowid_list1)
    lblannot_value_list1 = ibs1.get_lblannot_values(lblannot_rowid_list1)
    lblannot_note_list1 = ibs1.get_lblannot_notes(lblannot_rowid_list1)
    #Label type
    lblannot_lbltype_rowid_list1 = ibs1.get_lblannot_lbltypes_rowids(lblannot_rowid_list1)
    lblannot_lbltype_text_list1 = ibs1.get_lbltype_text(lblannot_lbltype_rowid_list1)
    lblannot_lbltype_default_list1 = ibs1.get_lbltype_default(lblannot_lbltype_rowid_list1)

    #Annotation label relationships
    alr_rowid_list1 = ibs1._get_all_alr_rowids()
    _alr_lblannot_rowid_list1 = ibs1.get_alr_lblannot_rowids(alr_rowid_list1)
    alr_lblannot_uuid_list1 = ibs1.get_lblannot_uuids(_alr_lblannot_rowid_list1)
    _alr_annot_rowid_list1 = ibs1.get_alr_annot_rowids(alr_rowid_list1)
    alr_annot_uuid_list1 = ibs1.get_annot_uuids(_alr_annot_rowid_list1)

    #Images
    if gid_list1 is None:
        gid_list1 = ibs1.get_valid_gids()
    _gid_list1 = sorted(list(set(gid_list1 + _annot_gid_list1)))
    img_uuid_list1 = ibs1.get_image_uuids(_gid_list1)
    img_uri_list1 = ibs1.get_image_uris(_gid_list1)
    img_orig_name_list1 = ibs1.get_image_gnames(_gid_list1)
    img_gsize_list1 = ibs1.get_image_sizes(_gid_list1)
    img_ext_list1 = ibs1.get_image_exts(_gid_list1)
    img_latlon_list1 = ibs1.get_image_gps(_gid_list1)
    img_unixtime_list1 = ibs1.get_image_unixtime(_gid_list1)
    img_note_list1 = ibs1.get_image_notes(_gid_list1)

    ## Add information to destination database
    ibs2._internal_add_images(img_uuid_list1, img_uri_list1,
                              img_orig_name_list1, img_ext_list1, img_gsize_list1,
                              img_unixtime_list1, img_latlon_list1, img_note_list1)
    ibs2._internal_add_annots(annot_image_uuid_list1, annot_uuid_list1, annot_thetas_list1,
                             notes_list=None, annot_vert_list1)
