# -*- coding: utf-8 -*-
"""
developer convenience functions for ibs

TODO: need to split up into sub modules:
    consistency_checks
    feasibility_fixes
    move the export stuff to dbio

    python -m utool.util_inspect check_module_usage --pat="ibsfuncs.py"

    then there are also convineience functions that need to be ordered at least
    within this file
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import types
import functools
import re
from six.moves import zip, range, map, reduce
from os.path import split, join, exists
import numpy as np
import vtool as vt
import utool as ut
import ubelt as ub
from utool._internal.meta_util_six import get_funcname, set_funcname
import itertools as it
from wbia import constants as const
from wbia.control import accessor_decors
from wbia.control import controller_inject
from wbia import annotmatch_funcs  # NOQA
from skimage import io
import xml.etree.ElementTree as ET
import datetime
from PIL import Image
import cv2


# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[ibsfuncs]')


# Must import class before injection
CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)


register_api = controller_inject.get_wbia_flask_api(__name__)


@ut.make_class_postinject_decorator(CLASS_INJECT_KEY, __name__)
def postinject_func(ibs):
    r"""
    Args:
        ibs (IBEISController):

    CommandLine:
        python -m wbia.other.ibsfuncs --test-postinject_func

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs.delete_empty_nids()  # a test run before this forgot to do this
        >>> aids_list = ibs.get_name_aids(ibs.get_valid_nids())
        >>> # indirectly test postinject_func
        >>> thetas_list = ibs.get_unflat_annot_thetas(aids_list)
        >>> result = str(thetas_list)
        >>> print(result)
        [[0.0, 0.0], [0.0, 0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    """
    # List of getters to ut.unflatten2
    to_unflatten = [
        ibs.get_annot_uuids,
        ibs.get_image_uuids,
        ibs.get_name_texts,
        ibs.get_image_unixtime,
        ibs.get_annot_bboxes,
        ibs.get_annot_thetas,
    ]
    for flat_getter in to_unflatten:
        unflat_getter = _make_unflat_getter_func(flat_getter)
        ut.inject_func_as_method(ibs, unflat_getter, allow_override=ibs.allow_override)
    # very hacky, but useful
    ibs.unflat_map = unflat_map


@register_ibs_method
def export_to_hotspotter(ibs):
    from wbia.dbio import export_hsdb

    export_hsdb.export_wbia_to_hotspotter(ibs)


@register_ibs_method
def get_image_time_statstr(ibs, gid_list=None):
    if gid_list is None:
        gid_list = ibs.get_valid_gids()
    unixtime_list_ = ibs.get_image_unixtime(gid_list)
    utvalid_list = [time != -1 for time in unixtime_list_]
    unixtime_list = list(ub.compress(unixtime_list_, utvalid_list))
    unixtime_statstr = ut.get_timestats_str(unixtime_list, newlines=True)
    return unixtime_statstr


@register_ibs_method
def get_image_annotation_bboxes(ibs, gid_list):
    aids_list = ibs.get_image_aids(gid_list)
    bboxes_list = ibs.get_unflat_annotation_bboxes(aids_list)
    return bboxes_list


@register_ibs_method
def get_image_annotation_thetas(ibs, gid_list):
    aids_list = ibs.get_image_aids(gid_list)
    thetas_list = ibs.get_unflat_annotation_thetas(aids_list)
    return thetas_list


@register_ibs_method
def filter_junk_annotations(ibs, aid_list):
    r"""
    remove junk annotations from a list

    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: filtered_aid_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> filtered_aid_list = filter_junk_annotations(ibs, aid_list)
        >>> result = str(filtered_aid_list)
        >>> print(result)
    """
    isjunk_list = ibs.get_annot_isjunk(aid_list)
    filtered_aid_list = ut.filterfalse_items(aid_list, isjunk_list)
    return filtered_aid_list


@register_ibs_method
def compute_all_chips(ibs, aid_list=None, **kwargs):
    """ Executes lazy evaluation of all chips """
    print('[ibs] compute_all_chips')
    if aid_list is None:
        aid_list = ibs.get_valid_aids(**kwargs)
    cid_list = ibs.depc_annot.get_rowids('chips', aid_list)
    return cid_list


@register_ibs_method
def ensure_annotation_data(ibs, aid_list, chips=True, feats=True, featweights=False):
    if featweights:
        ibs.depc_annot.get_rowids('featweight', aid_list)
    elif feats:
        ibs.depc_annot.get_rowids('feat', aid_list)
    elif chips:
        ibs.depc_annot.get_rowids('chips', aid_list)


@register_ibs_method
def convert_empty_images_to_annotations(ibs):
    """ images without chips are given an ANNOTATION over the entire image """
    gid_list = ibs.get_empty_gids()
    aid_list = ibs.use_images_as_annotations(gid_list)
    return aid_list


@register_ibs_method
def add_trivial_annotations(ibs, *args, **kwargs):
    ibs.use_images_as_annotations(*args, **kwargs)


@register_ibs_method
def use_images_as_annotations(
    ibs,
    gid_list,
    name_list=None,
    nid_list=None,
    notes_list=None,
    adjust_percent=0.0,
    tags_list=None,
):
    """ Adds an annotation the size of the entire image to each image.
    adjust_percent - shrinks the ANNOTATION by percentage on each side
    """
    pct = adjust_percent  # Alias
    gsize_list = ibs.get_image_sizes(gid_list)
    # Build bounding boxes as images size minus padding
    bbox_list = [
        (
            int(0 + (gw * pct)),
            int(0 + (gh * pct)),
            int(gw - (gw * pct * 2)),
            int(gh - (gh * pct * 2)),
        )
        for (gw, gh) in gsize_list
    ]
    theta_list = [0.0 for _ in range(len(gsize_list))]
    aid_list = ibs.add_annots(
        gid_list,
        bbox_list,
        theta_list,
        name_list=name_list,
        nid_list=nid_list,
        notes_list=notes_list,
    )
    if tags_list is not None:
        ibs.append_annot_case_tags(aid_list, tags_list)
    return aid_list


@register_ibs_method
def get_annot_been_adjusted(ibs, aid_list):
    """
    Returns if a bounding box has been adjusted from defaults set in
    use_images_as_annotations Very hacky very heurstic.
    """
    bbox_list = ibs.get_annot_bboxes(aid_list)
    ori_list = np.array(ibs.get_annot_thetas(aid_list))
    size_list = ibs.get_image_sizes(ibs.get_annot_gids(aid_list))

    been_ori_adjusted = ori_list != 0

    adjusted_list = [
        (bbox[0] / gw, bbox[1] / gh, (1 - (bbox[2] / gw)) / 2, (1 - (bbox[3] / gh)) / 2,)
        for bbox, (gw, gh) in zip(bbox_list, size_list)
    ]

    # Has the bounding box been moved past the default value?
    been_bbox_adjusted = np.array(
        [
            np.abs(np.diff(np.array(list(ut.iprod(pcts, pcts))), axis=1)).max() > 1e-2
            for pcts in adjusted_list
        ]
    )

    been_adjusted = np.logical_or(been_ori_adjusted, been_bbox_adjusted)
    return been_adjusted


@register_ibs_method
def assert_valid_species_texts(ibs, species_list, iswarning=True):
    if ut.NO_ASSERTS:
        return
    try:
        valid_species = ibs.get_all_species_texts()
        isvalid_list = [
            species in valid_species  # or species == const.UNKNOWN
            for species in species_list
        ]
        assert all(isvalid_list), 'invalid species found in %r: %r' % (
            ut.get_caller_name(range(1, 3)),
            ut.compress(species_list, ut.not_list(isvalid_list)),
        )
    except AssertionError as ex:
        ut.printex(ex, iswarning=iswarning)
        if not iswarning:
            raise


@register_ibs_method
def assert_singleton_relationship(ibs, alrids_list):
    if ut.NO_ASSERTS:
        return
    try:
        assert all(
            [len(alrids) == 1 for alrids in alrids_list]
        ), 'must only have one relationship of a type'
    except AssertionError as ex:
        parent_locals = ut.get_parent_frame().f_locals
        ut.printex(
            ex, 'parent_locals=' + ut.repr2(parent_locals), key_list=['alrids_list']
        )
        raise


@register_ibs_method
def assert_valid_gids(ibs, gid_list, verbose=False, veryverbose=False):
    r"""
    """
    isinvalid_list = [gid is None for gid in ibs.get_image_gid(gid_list)]
    try:
        assert not any(isinvalid_list), 'invalid gids: %r' % (
            ut.compress(gid_list, isinvalid_list),
        )
        isinvalid_list = [not isinstance(gid, ut.VALID_INT_TYPES) for gid in gid_list]
        assert not any(isinvalid_list), 'invalidly typed gids: %r' % (
            ut.compress(gid_list, isinvalid_list),
        )
    except AssertionError as ex:
        print('dbname = %r' % (ibs.get_dbname()))
        ut.printex(ex)
        raise
    if veryverbose:
        print('passed assert_valid_gids')


@register_ibs_method
def assert_valid_aids(
    ibs, aid_list, verbose=False, veryverbose=False, msg='', auuid_list=None
):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (int):  list of annotation ids
        verbose (bool):  verbosity flag(default = False)
        veryverbose (bool): (default = False)

    CommandLine:
        python -m wbia.other.ibsfuncs --test-assert_valid_aids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> verbose = False
        >>> veryverbose = False
        >>> print('Asserting multiple')
        >>> result = assert_valid_aids(ibs, aid_list, verbose, veryverbose)
        >>> print('Asserting single')
        >>> result = assert_valid_aids(ibs, aid_list[0:1], verbose, veryverbose)
        >>> print('Asserting multiple incorrect')
        >>> auuid_list = ibs.get_annot_uuids(aid_list) + [None]
        >>> try:
        >>>    result = assert_valid_aids(ibs, aid_list + [0], verbose, veryverbose, auuid_list=auuid_list)
        >>> except AssertionError:
        >>>    print('Correctly got assertion')
        >>> else:
        >>>    assert False, 'should have failed'
        >>> print('Asserting single incorrect')
        >>> try:
        >>>    result = assert_valid_aids(ibs, [0], verbose, veryverbose)
        >>> except AssertionError:
        >>>    print('Correctly got assertion')
        >>> else:
        >>>    assert False, 'should have failed'
        >>> print(result)
        >>> print(result)
    """
    # if ut.NO_ASSERTS and not force:
    #    return
    # valid_aids = set(ibs.get_valid_aids())
    # invalid_aids = [aid for aid in aid_list if aid not in valid_aids]
    # isinvalid_list = [aid not in valid_aids for aid in aid_list]
    isinvalid_list = [aid is None for aid in ibs.get_annot_aid(aid_list)]
    # isinvalid_list = [aid not in valid_aids for aid in aid_list]
    invalid_aids = ut.compress(aid_list, isinvalid_list)
    try:
        assert not any(isinvalid_list), '%d/%d invalid aids: %r' % (
            sum(isinvalid_list),
            len(aid_list),
            invalid_aids,
        )
        isinvalid_list = [not ut.is_int(aid) for aid in aid_list]
        invalid_aids = ut.compress(aid_list, isinvalid_list)
        assert not any(isinvalid_list), '%d/%d invalidly typed aids: %r' % (
            sum(isinvalid_list),
            len(aid_list),
            invalid_aids,
        )
    except AssertionError as ex:
        if auuid_list is not None and len(auuid_list) == len(aid_list):
            invalid_auuids = ut.compress(auuid_list, isinvalid_list)  # NOQA
        else:
            invalid_auuids = 'not-available'
        dbname = ibs.get_dbname()  # NOQA
        locals_ = dict(
            dbname=dbname, invalid_auuids=invalid_auuids, invalid_aids=invalid_aids
        )
        ut.printex(
            ex,
            'assert_valid_aids: ' + msg,
            locals_=locals_,
            keys=['invalid_aids', 'invalid_auuids', 'dbname'],
        )
        raise
    if veryverbose:
        print('passed assert_valid_aids')


@register_ibs_method
def get_missing_gids(ibs, gid_list=None):
    r"""
    Finds gids with broken links to the original data.

    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list): (default = None)

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-get_missing_gids --db GZ_Master1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> #ibs = wbia.opendb('GZ_Master1')
        >>> gid_list = ibs.get_valid_gids()
        >>> bad_gids = ibs.get_missing_gids(gid_list)
        >>> print('#bad_gids = %r / %r' % (len(bad_gids), len(gid_list)))
    """
    if gid_list is None:
        gid_list = ibs.get_valid_gids()
    gpath_list = ibs.get_image_paths(gid_list)
    exists_list = list(map(exists, gpath_list))
    bad_gids = ut.compress(gid_list, ut.not_list(exists_list))
    return bad_gids


@register_ibs_method
def assert_images_exist(ibs, gid_list=None, verbose=True):
    if gid_list is None:
        gid_list = ibs.get_valid_gids()
    print('checking images exist')
    bad_gids = ibs.get_missing_gids()
    num_bad_gids = len(bad_gids)
    if verbose:
        bad_gpaths = ibs.get_image_paths(bad_gids)
        print('Bad Gpaths:')
        print(ut.truncate_str(ut.repr2(bad_gpaths), maxlen=500))
        print('Bad GIDs:')
        print(bad_gids)
    print('%d images dont exist' % (num_bad_gids,))
    print('[check] checked %d images exist' % len(gid_list))
    return num_bad_gids


@register_ibs_method
def assert_images_are_unique(ibs, gid_list=None, verbose=True):
    if gid_list is None:
        gid_list = ibs.get_valid_gids()

    gpath_list = ibs.get_image_paths(gid_list)
    hash_list = [ut.get_file_hash(gpath, hexdigest=True) for gpath in gpath_list]

    if len(hash_list) != len(set(hash_list)):
        hash_histogram = {}
        for gid, gpath, hash_ in zip(gid_list, gpath_list, hash_list):
            if hash_ not in hash_histogram:
                hash_histogram[hash_] = []
            vals = (gid, gpath)
            hash_histogram[hash_].append(vals)

        divergent = 0
        counter = 0
        global_delete_gid_list = []
        for key, gid_gpath_list_ in hash_histogram.items():
            if len(gid_gpath_list_) >= 2:
                gid_list = [_[0] for _ in gid_gpath_list_]
                gpath_list = [_[1] for _ in gid_gpath_list_]

                aids_list = ibs.get_image_aids(gid_list)
                aids_len_list = list(map(len, aids_list))
                max_aids = max(aids_len_list)

                filtered_gid_list = sorted(
                    [
                        gid
                        for gid, aids_len in zip(gid_list, aids_len_list)
                        if aids_len == max_aids
                    ]
                )

                survive_gid = filtered_gid_list[0]
                delete_gid_list = list(gid_list)
                delete_gid_list.remove(survive_gid)
                assert survive_gid not in delete_gid_list

                global_delete_gid_list += delete_gid_list
                counter += 1

                if len(set(aids_len_list)) > 1:
                    divergent += 1
                    # print('FOUND DIVERGENT')

                # print(gid_list)
                # print(aids_list)
                # print(aids_len_list)
                # print(max_aids)
                # print(filtered_gid_list)
                # print(survive_gid)
                # print(delete_gid_list)
                # print('-' * 40)

        total = len(hash_histogram.keys())
        print(
            'Found [%d / %d / %d] images that have duplicates...'
            % (divergent, counter, total,)
        )
        ibs.delete_images(global_delete_gid_list)
        print('Deleted %d images.' % (len(global_delete_gid_list),))


def assert_valid_names(name_list):
    """ Asserts that user specified names do not conflict with the standard unknown name """
    if ut.NO_ASSERTS:
        return

    def isconflict(name, other):
        return name.startswith(other) and len(name) > len(other)

    valid_namecheck = [not isconflict(name, const.UNKNOWN) for name in name_list]
    assert all(valid_namecheck), (
        'A name conflicts with UKNONWN Name. -- '
        'cannot start a name with four underscores'
    )


@ut.on_exception_report_input
def assert_lblannot_rowids_are_type(ibs, lblannot_rowid_list, valid_lbltype_rowid):
    if ut.NO_ASSERTS:
        return
    lbltype_rowid_list = ibs.get_lblannot_lbltypes_rowids(lblannot_rowid_list)
    try:
        # HACK: the unknown_lblannot_rowid will have a None type
        # the unknown lblannot_rowid should be handled more gracefully
        # this should just check the first condition (get rid of the or)
        ut.assert_same_len(lbltype_rowid_list, lbltype_rowid_list)
        ut.assert_scalar_list(lblannot_rowid_list)
        validtype_list = [
            (lbltype_rowid == valid_lbltype_rowid)
            or (lbltype_rowid is None and lblannot_rowid == const.UNKNOWN_LBLANNOT_ROWID)
            for lbltype_rowid, lblannot_rowid in zip(
                lbltype_rowid_list, lblannot_rowid_list
            )
        ]
        assert all(validtype_list), 'not all types match valid type'
    except AssertionError as ex:
        tup_list = list(map(str, list(zip(lbltype_rowid_list, lblannot_rowid_list))))
        print('[!!!] (lbltype_rowid, lblannot_rowid) = : ' + ut.indentjoin(tup_list))
        print('[!!!] valid_lbltype_rowid: %r' % (valid_lbltype_rowid,))

        ut.printex(
            ex,
            'not all types match valid type',
            keys=['valid_lbltype_rowid', 'lblannot_rowid_list'],
        )
        raise


@register_ibs_method
def check_for_unregistered_images(ibs):
    images = ibs.images()
    # Check if any images in the image directory are unregistered
    gpath_disk = set(ut.ls(ibs.imgdir))
    gpath_registered = set(images.paths)
    overlaps = ut.set_overlaps(gpath_disk, gpath_registered, 'ondisk', 'reg')
    print('overlaps' + ut.repr3(overlaps))
    gpath_unregistered = gpath_disk - gpath_registered
    return overlaps, gpath_unregistered


@register_ibs_method
def delete_unregistered_images(ibs, verbose=True):
    dst_fpath = ibs.trashdir
    ut.ensuredir(dst_fpath)
    _, gpath_unregistered = ibs.check_for_unregistered_images()
    gname_list = [ut.split(gpath)[1] for gpath in gpath_unregistered]
    dst_fpath_list = [join(dst_fpath, gname) for gname in gname_list]
    ut.copy_files_to(gpath_unregistered, dst_fpath_list=dst_fpath_list)
    ut.remove_file_list(gpath_unregistered)


@register_ibs_method
def check_image_consistency(ibs, gid_list=None):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list): (default = None)

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-check_image_consistency  --db=GZ_Master1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> gid_list = None
        >>> result = check_image_consistency(ibs, gid_list)
        >>> print(result)
    """
    # TODO: more consistency checks
    if gid_list is None:
        gid_list = ibs.get_valid_gids()
    print('check image consistency. len(gid_list)=%r' % len(gid_list))
    assert len(ut.debug_duplicate_items(gid_list)) == 0
    assert_images_exist(ibs, gid_list)
    image_uuid_list = ibs.get_image_uuids(gid_list)
    assert len(ut.debug_duplicate_items(image_uuid_list)) == 0
    # check_image_uuid_consistency(ibs, gid_list)


@register_ibs_method
def check_image_uuid_consistency(ibs, gid_list=None):
    """
    Checks to make sure image uuids are computed detemenistically
    by recomputing all guuids and checking that they are equal to
    what is already there.

    VERY SLOW

    CommandLine:
        python -m wbia.other.ibsfuncs --test-check_image_uuid_consistency --db=PZ_Master0
        python -m wbia.other.ibsfuncs --test-check_image_uuid_consistency --db=GZ_Master1
        python -m wbia.other.ibsfuncs --test-check_image_uuid_consistency
        python -m wbia.other.ibsfuncs --test-check_image_uuid_consistency --db lynx

    Example:
        >>> # SCRIPT
        >>> import wbia
        >>> import utool as ut
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> images = ibs.images()
        >>> # Check only very the largest files
        >>> #bytes_list_ = [
        >>> #    ut.get_file_nBytes(path)
        >>> #    for path in ut.ProgIter(images.paths, lbl='reading nbytes')]
        >>> #sortx = ut.list_argsort(bytes_list_, reverse=True)[0:10]
        >>> #images = images.take(sortx)
        >>> gid_list = list(images)
        >>> wbia.other.ibsfuncs.check_image_uuid_consistency(ibs, gid_list)
    """
    print('checking image uuid consistency')
    if gid_list is None:
        gid_list = ibs.get_valid_gids()
    uuid_list = ibs.get_image_uuids(gid_list)
    gpath_list = ibs.get_image_paths(gid_list)
    uuid_list_ = ibs.compute_image_uuids(gpath_list)

    bad_list = []
    for gid, uuid, uuid_ in zip(gid_list, uuid_list, uuid_list_):
        if uuid != uuid_:
            bad_list.append(gid)

    return bad_list


@register_ibs_method
def check_image_loadable(ibs, gid_list=None):
    print('checking image loadable')
    if gid_list is None:
        gid_list = ibs.get_valid_gids()

    gpath_list = ibs.get_image_paths(gid_list)
    orient_list = ibs.get_image_orientation(gid_list)

    arg_iter = list(zip(gpath_list, orient_list,))
    flag_list = ut.util_parallel.generate2(
        check_image_loadable_worker, arg_iter, futures_threaded=True
    )
    flag_list = list(flag_list)
    loadable_list = ut.take_column(flag_list, 0)
    exif_list = ut.take_column(flag_list, 1)

    bad_loadable_list = ut.filterfalse_items(gid_list, loadable_list)
    bad_exif_list = ut.filterfalse_items(gid_list, exif_list)
    return bad_loadable_list, bad_exif_list


def check_image_loadable_worker(gpath, orient):
    loadable, exif = True, True
    try:
        img = cv2.imread(gpath)
        assert img is not None
        # Sanitize weird behavior
        cv2.imwrite(gpath, img)
        img = Image.open(gpath, 'r')
        assert img is not None
        img = io.imread(gpath)
        assert img is not None
        img = vt.imread(gpath, orient=orient)
        assert img is not None
    except Exception:
        loadable = False
    try:
        img = vt.imread(gpath, orient='auto')
        assert img is not None
    except Exception:
        exif = False
    return loadable, exif


@register_ibs_method
def check_image_duplcates(ibs, gid_list=None):
    print('checking image duplcates')
    if gid_list is None:
        gid_list = ibs.get_valid_gids()

    gpath_list = ibs.get_image_paths(gid_list)
    uuid_list = ibs.compute_image_uuids(gpath_list)

    duplcate_dict = {}
    for gid, uuid in zip(gid_list, uuid_list):
        if uuid not in duplcate_dict:
            duplcate_dict[uuid] = []
        duplcate_dict[uuid].append(gid)

    delete_gid_list = []
    for uuid in duplcate_dict:
        duplcate_gid_list = duplcate_dict[uuid]
        length = len(duplcate_gid_list)

        assert length > 0
        if length == 1:
            continue

        print(uuid, duplcate_gid_list)
        keep_index = None

        uuid_list = ibs.get_image_uuids(duplcate_gid_list)
        if uuid in uuid_list:
            keep_index = uuid_list.index(uuid)
        else:
            reviewed_list = ibs.get_image_reviewed(duplcate_gid_list)
            aids_list = ibs.get_image_aids(duplcate_gid_list)
            length_list = list(map(len, aids_list))
            index_list = list(range(len(duplcate_gid_list)))
            zipped = sorted(list(zip(reviewed_list, length_list, index_list)))
            keep_index = zipped[-1][2]

        assert keep_index is not None

        delete_gid_list_ = []
        for index, duplcate_gid in enumerate(duplcate_gid_list):
            if index == keep_index:
                continue
            delete_gid_list_.append(duplcate_gid)
        print(keep_index, delete_gid_list_)

        delete_gid_list += delete_gid_list_

    return delete_gid_list


def check_annot_overlap(ibs, gid_list=None, PIXELS=100.0, IOU=0.1):
    from wbia.other.detectfuncs import general_overlap

    if gid_list is None:
        gid_list = ibs.get_valid_gids()

    reviewed_list = ibs.get_image_reviewed(gid_list)
    aid_list = ibs.get_image_aids(gid_list)
    zipped = list(zip(gid_list, reviewed_list, aid_list))

    bad_gid_list = []

    # Criteria 1 - Absolute Standard Deviation
    for gid, reviewed, aids in zipped:
        if len(aids) <= 1:
            continue
        if not reviewed:
            continue
        bbox_list = ibs.get_annot_bboxes(aids)
        bbox_list = np.array(bbox_list)
        mean = np.mean(bbox_list, axis=0)
        std = np.mean(np.abs(bbox_list - mean), axis=0)
        a = sum(std <= PIXELS)
        if a == 4:
            print(gid)
            print(bbox_list)
            print(std)
            bad_gid_list.append(gid)

    # Criteria 2 - IoU
    for gid, reviewed, aids in zip(gid_list, reviewed_list, aid_list):
        if len(aids) <= 1:
            continue
        if not reviewed:
            continue
        bbox_list = ibs.get_annot_bboxes(aids)
        bbox_dict_list = [
            {
                'xtl': xtl,
                'ytl': ytl,
                'xbr': xtl + width,
                'ybr': ytl + height,
                'width': width,
                'height': height,
            }
            for xtl, ytl, width, height in bbox_list
        ]
        if len(aids) > 2:
            overlap = general_overlap(bbox_dict_list, bbox_dict_list)
            triangle = np.tri(overlap.shape[0]).astype(np.bool)
            indices = np.where(triangle)
            overlap[indices] = 0.0
            overlap_flag = overlap >= IOU
            total = np.sum(overlap_flag)
            if total > 0:
                print(gid)
                print(overlap)
                bad_gid_list.append(gid)

    # ibs.set_image_reviewed(bad_gid_list, [0] * len(bad_gid_list))
    return bad_gid_list


@register_ibs_method
def check_annot_consistency(ibs, aid_list=None):
    r"""
    Args:
        ibs      (IBEISController):
        aid_list (list):

    CommandLine:
        python -m wbia.other.ibsfuncs --test-check_annot_consistency

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> result = check_annot_consistency(ibs, aid_list)
        >>> print(result)
    """
    # TODO: more consistency checks
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    print('check annot consistency. len(aid_list)=%r' % len(aid_list))
    annot_gid_list = ibs.get_annot_gids(aid_list)
    num_None_annot_gids = sum(ut.flag_None_items(annot_gid_list))
    print('num_None_annot_gids = %r ' % (num_None_annot_gids,))
    assert num_None_annot_gids == 0
    # print(ut.repr2(dict(ut.debug_duplicate_items(annot_gid_list))))
    assert_images_exist(ibs, annot_gid_list)
    unique_gids = list(set(annot_gid_list))
    print('num_unique_images=%r / %r' % (len(unique_gids), len(annot_gid_list)))
    cfpath_list = ibs.get_annot_chip_fpath(aid_list, ensure=False)
    valid_chip_list = [
        None if cfpath is None else exists(cfpath) for cfpath in cfpath_list
    ]
    invalid_list = [flag is False for flag in valid_chip_list]
    invalid_aids = ut.compress(aid_list, invalid_list)
    if len(invalid_aids) > 0:
        print('found %d inconsistent chips attempting to fix' % len(invalid_aids))
        ibs.delete_annot_chips(invalid_aids)
    ibs.check_chip_existence(aid_list=aid_list)
    visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
    exemplar_flag = ibs.get_annot_exemplar_flags(aid_list)
    is_unknown = ibs.is_aid_unknown(aid_list)
    # Exemplars should all be known
    unknown_exemplar_flags = ut.compress(exemplar_flag, is_unknown)
    is_error = [not flag for flag in unknown_exemplar_flags]
    assert all(is_error), 'Unknown annotations are set as exemplars'
    ut.debug_duplicate_items(visual_uuid_list)


@register_ibs_method
def check_annot_corrupt_uuids(ibs, aid_list=None):
    """
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> check_annot_corrupt_uuids(ibs, aid_list)
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    try:
        ibs.get_annot_uuids(aid_list)
    except Exception as ex:
        ut.printex(ex)
        failed_aids = []
        for aid in aid_list:
            try:
                ibs.get_annot_uuids(aid)
            except Exception:
                failed_aids.append(aid)
        print('failed_aids = %r' % (failed_aids,))
        return failed_aids
    else:
        print('uuids do not seem to be corrupt')


def check_name_consistency(ibs, nid_list):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        nid_list (list):

    CommandLine:
        python -m wbia.other.ibsfuncs --test-check_name_consistency

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_nids()
        >>> result = check_name_consistency(ibs, nid_list)
        >>> print(result)
    """
    # aids_list = ibs.get_name_aids(nid_list)
    print('check name consistency. len(nid_list)=%r' % len(nid_list))
    aids_list = ibs.get_name_aids(nid_list)
    print('Checking that all annotations of a name have the same species')
    species_rowids_list = ibs.unflat_map(ibs.get_annot_species_rowids, aids_list)
    error_list = []
    for aids, sids in zip(aids_list, species_rowids_list):
        if not ut.allsame(sids):
            error_msg = (
                'aids=%r have the same name, but belong to multiple species=%r'
                % (aids, ibs.get_species_texts(ut.unique_ordered(sids)))
            )
            print(error_msg)
            error_list.append(error_msg)
    if len(error_list) > 0:
        raise AssertionError(
            'A total of %d names failed check_name_consistency' % (len(error_list))
        )


@register_ibs_method
def check_name_mapping_consistency(ibs, nx2_aids):
    """ checks that all the aids grouped in a name ahave the same name """
    # DEBUGGING CODE
    try:
        from wbia import ibsfuncs

        _nids_list = ibsfuncs.unflat_map(ibs.get_annot_name_rowids, nx2_aids)
        assert all(map(ut.allsame, _nids_list))
    except Exception as ex:
        # THESE SHOULD BE CONSISTENT BUT THEY ARE NOT!!?
        # name_annots = [ibs.get_annot_name_rowids(aids) for aids in nx2_aids]
        bad = 0
        good = 0
        huh = 0
        for nx, aids in enumerate(nx2_aids):
            nids = ibs.get_annot_name_rowids(aids)
            if np.all(np.array(nids) > 0):
                print(nids)
                if ut.allsame(nids):
                    good += 1
                else:
                    huh += 1
            else:
                bad += 1
        ut.printex(ex, keys=['good', 'bad', 'huh'])


@register_ibs_method
def check_annot_size(ibs):
    print('Checking annot sizes')
    aid_list = ibs.get_valid_aids()
    uuid_list = ibs.get_annot_uuids(aid_list)
    desc_list = ibs.get_annot_vecs(aid_list, ensure=False)
    kpts_list = ibs.get_annot_kpts(aid_list, ensure=False)
    vert_list = ibs.get_annot_verts(aid_list)
    print('size(aid_list) = ' + ut.byte_str2(ut.get_object_nbytes(aid_list)))
    print('size(vert_list) = ' + ut.byte_str2(ut.get_object_nbytes(vert_list)))
    print('size(uuid_list) = ' + ut.byte_str2(ut.get_object_nbytes(uuid_list)))
    print('size(desc_list) = ' + ut.byte_str2(ut.get_object_nbytes(desc_list)))
    print('size(kpts_list) = ' + ut.byte_str2(ut.get_object_nbytes(kpts_list)))


def check_exif_data(ibs, gid_list):
    """ TODO CALL SCRIPT """
    import vtool.exif as exif
    from PIL import Image  # NOQA

    gpath_list = ibs.get_image_paths(gid_list)
    exif_dict_list = []
    for ix in ut.ProgIter(range(len(gpath_list)), lbl='checking exif: '):
        gpath = gpath_list[ix]
        pil_img = Image.open(gpath, 'r')  # NOQA
        exif_dict = exif.get_exif_dict(pil_img)
        exif_dict_list.append(exif_dict)
        # if len(exif_dict) > 0:
        #    break

    has_latlon = []
    for exif_dict in exif_dict_list:
        latlon = exif.get_lat_lon(exif_dict, None)
        if latlon is not None:
            has_latlon.append(True)
        else:
            has_latlon.append(False)

    print('%d / %d have gps info' % (sum(has_latlon), len(has_latlon),))

    key2_freq = ut.ddict(lambda: 0)
    num_tags_list = []
    for exif_dict in exif_dict_list:
        exif_dict2 = exif.make_exif_dict_human_readable(exif_dict)
        num_tags_list.append(len(exif_dict))
        for key in exif_dict2.keys():
            key2_freq[key] += 1

    print('Stats for num tags per image')
    print(ut.repr4(ut.get_stats(num_tags_list)))

    print('tag frequency')
    print(ut.repr2(key2_freq))


@register_ibs_method
def run_integrity_checks(ibs, embed=False):
    """ Function to run all database consistency checks """
    print('[ibsfuncs] Checking consistency')
    gid_list = ibs.get_valid_gids()
    aid_list = ibs.get_valid_aids()
    nid_list = ibs.get_valid_nids()
    check_annot_size(ibs)
    check_image_consistency(ibs, gid_list)
    check_annot_consistency(ibs, aid_list)
    check_name_consistency(ibs, nid_list)
    check_annotmatch_consistency(ibs)
    # Very slow check
    check_image_uuid_consistency(ibs, gid_list)
    if embed:
        ut.embed()
    print('[ibsfuncs] Finshed consistency check')


@register_ibs_method
def check_annotmatch_consistency(ibs):
    annomatch_rowids = ibs._get_all_annotmatch_rowids()
    aid1_list = ibs.get_annotmatch_aid1(annomatch_rowids)
    aid2_list = ibs.get_annotmatch_aid2(annomatch_rowids)
    exists1_list = ibs.db.check_rowid_exists(const.ANNOTATION_TABLE, aid1_list)
    exists2_list = ibs.db.check_rowid_exists(const.ANNOTATION_TABLE, aid2_list)
    invalid_list = ut.not_list(ut.and_lists(exists1_list, exists2_list))
    invalid_annotmatch_rowids = ut.compress(annomatch_rowids, invalid_list)
    print('There are %d invalid annotmatch rowids' % (len(invalid_annotmatch_rowids),))
    return invalid_annotmatch_rowids


@register_ibs_method
def fix_invalid_annotmatches(ibs):
    print('Fixing invalid annotmatches')
    invalid_annotmatch_rowids = ibs.check_annotmatch_consistency()
    ibs.delete_annotmatch(invalid_annotmatch_rowids)


def fix_remove_visual_dupliate_annotations(ibs):
    r"""
    depricate because duplicate visual_uuids
    are no longer allowed to be duplicates

    Add to clean database?

    removes visually duplicate annotations

    Args:
        ibs (IBEISController):

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('GZ_ALL')
        >>> fix_remove_visual_dupliate_annotations(ibs)
    """
    aid_list = ibs.get_valid_aids()
    visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
    ibs_dup_annots = ut.debug_duplicate_items(visual_uuid_list)
    dupaids_list = []
    if len(ibs_dup_annots):
        for key, dupxs in six.iteritems(ibs_dup_annots):
            aids = ut.take(aid_list, dupxs)
            dupaids_list.append(aids[1:])

        toremove_aids = ut.flatten(dupaids_list)
        print('About to delete toremove_aids=%r' % (toremove_aids,))
        if ut.are_you_sure('About to delete %r aids' % (len(toremove_aids))):
            ibs.delete_annots(toremove_aids)

            aid_list = ibs.get_valid_aids()
            visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
            ibs_dup_annots = ut.debug_duplicate_items(visual_uuid_list)
            assert len(ibs_dup_annots) == 0


def fix_zero_features(ibs):
    aid_list = ibs.get_valid_aids()
    nfeat_list = ibs.get_annot_num_feats(aid_list, ensure=False)
    haszero_list = [nfeat == 0 for nfeat in nfeat_list]
    haszero_aids = ut.compress(aid_list, haszero_list)
    ibs.delete_annot_chips(haszero_aids)


@register_ibs_method
def fix_and_clean_database(ibs):
    """ Function to run all database cleanup scripts

    Rename to run_cleanup_scripts

    Break into two funcs:
        run_cleanup_scripts
        run_fixit_scripts

    CONSITENCY CHECKS TODO:
        * check that annotmatches marked as False do not have the
          same name for similar viewpoints.
        * check that photobombs are have different names
        * warn if scenery matches have the same name

    """
    # TODO: Call more stuff, maybe rename to 'apply duct tape'
    with ut.Indenter('[FIX_AND_CLEAN]'):
        print('starting fixes and consistency checks')
        ibs.fix_unknown_exemplars()
        ibs.fix_invalid_name_texts()
        ibs.fix_invalid_nids()
        ibs.fix_invalid_annotmatches()
        fix_zero_features(ibs)
        ibs.update_annot_visual_uuids(ibs.get_valid_aids())
        ibs.delete_empty_nids()
        ibs.delete_empty_imgsetids()
        ibs.db.vacuum()
        print('finished fixes and consistency checks\n')


@register_ibs_method
def fix_exif_data(ibs, gid_list):
    """ TODO CALL SCRIPT

    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list): list of image ids

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-fix_exif_data

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='lynx')
        >>> gid_list = ibs.get_valid_gids()
        >>> result = fix_exif_data(ibs, gid_list)
        >>> print(result)
    """
    import vtool as vt
    from PIL import Image  # NOQA

    gpath_list = ibs.get_image_paths(gid_list)

    pil_img_gen = (Image.open(gpath, 'r') for gpath in gpath_list)  # NOQA

    exif_dict_list = [
        vt.get_exif_dict(pil_img)
        for pil_img in ut.ProgIter(
            pil_img_gen, length=len(gpath_list), lbl='reading exif: ', adjust=True
        )
    ]

    def fix_property(
        exif_getter, ibs_getter, ibs_setter, dirty_ibs_val, propname='property'
    ):
        exif_prop_list = [exif_getter(_dict, None) for _dict in exif_dict_list]
        hasprop_list = [prop is not None for prop in exif_prop_list]

        exif_prop_list_ = ut.compress(exif_prop_list, hasprop_list)
        gid_list_ = ut.compress(gid_list, hasprop_list)
        ibs_prop_list = ibs_getter(gid_list_)
        isdirty_list = [prop == dirty_ibs_val for prop in ibs_prop_list]

        print('%d / %d need %s update' % (sum(isdirty_list), len(isdirty_list), propname))

        if False and sum(isdirty_list) > 0:
            assert sum(isdirty_list) == len(
                isdirty_list
            ), 'safety. remove and evaluate if hit'
            # ibs.set_image_imagesettext(gid_list_, ['HASGPS'] * len(gid_list_))
            new_exif_prop_list = ut.compress(exif_prop_list_, isdirty_list)
            dirty_gid_list = ut.compress(gid_list_, isdirty_list)
            ibs_setter(dirty_gid_list, new_exif_prop_list)

    FIX_GPS = True
    if FIX_GPS:
        fix_property(
            vt.get_lat_lon, ibs.get_image_gps, ibs.set_image_gps, (-1, -1), 'gps'
        )
        # latlon_list = [vt.get_lat_lon(_dict, None) for _dict in exif_dict_list]
        # hasprop_list = [latlon is not None for latlon in latlon_list]

        # latlon_list_ = ut.compress(latlon_list, hasprop_list)
        # gid_list_    = ut.compress(gid_list, hasprop_list)
        # gps_list = ibs.get_image_gps(gid_list_)
        # isdirty_list = [gps == (-1, -1) for gps in gps_list]

        # print('%d / %d need gps update' % (sum(isdirty_list),
        #                                   len(isdirty_list)))

        # if False and sum(isdirty_list)  > 0:
        #    assert sum(isdirty_list) == len(isdirty_list), (
        #        'safety. remove and evaluate if hit')
        #    #ibs.set_image_imagesettext(gid_list_, ['HASGPS'] * len(gid_list_))
        #    latlon_list__ = ut.compress(latlon_list_, isdirty_list)
        #    gid_list__ = ut.compress(gid_list_, isdirty_list)
        #    ibs.set_image_gps(gid_list__, latlon_list__)

    FIX_UNIXTIME = True
    if FIX_UNIXTIME:
        dirty_ibs_val = -1
        propname = 'unixtime'
        exif_getter = vt.get_unixtime
        ibs_getter = ibs.get_image_unixtime
        ibs_setter = ibs.set_image_unixtime
        fix_property(exif_getter, ibs_getter, ibs_setter, dirty_ibs_val, propname)
        # exif_prop_list = [vt.get_unixtime(_dict, None) for _dict in exif_dict_list]
        # hasprop_list = [prop is not None for prop in exif_prop_list]

        # exif_prop_list_ = ut.compress(exif_prop_list, hasprop_list)
        # gid_list_       = ut.compress(gid_list, hasprop_list)
        # ibs_prop_list   = ibs.get_image_unixtime(gid_list_)
        # isdirty_list    = [prop == dirty_ibs_val for prop in ibs_prop_list]

        # print('%d / %d need time update' % (sum(isdirty_list),
        #                                    len(isdirty_list)))

        # if False and sum(isdirty_list)  > 0:
        #    assert sum(isdirty_list) == len(isdirty_list), 'safety. remove and evaluate if hit'
        #    #ibs.set_image_imagesettext(gid_list_, ['HASGPS'] * len(gid_list_))
        #    new_exif_prop_list = ut.compress(exif_prop_list_, isdirty_list)
        #    dirty_gid_list = ut.compress(gid_list_, isdirty_list)
        #    ibs.set_image_unixtime(dirty_gid_list, new_exif_prop_list)


@register_ibs_method
def fix_invalid_nids(ibs):
    r"""
    Make sure that all rowids are greater than 0

    We can only handle there being a name with rowid 0 if it is UNKNOWN. In this
    case we safely delete it, but anything more complicated needs to be handled
    anually

    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.ibsfuncs --test-fix_invalid_nids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> result = fix_invalid_nids(ibs)
        >>> print(result)
    """
    print('[ibs] fixing invalid nids (nids that are <= ibs.UKNOWN_NAME_ROWID)')
    # Get actual rowids from sql database (no postprocessing)
    nid_list = ibs._get_all_known_name_rowids()
    # Get actual names from sql database (no postprocessing)
    name_text_list = ibs.get_name_texts(nid_list, apply_fix=False)
    is_invalid_nid_list = [nid <= const.UNKNOWN_NAME_ROWID for nid in nid_list]
    if any(is_invalid_nid_list):
        invalid_nids = ut.compress(nid_list, is_invalid_nid_list)
        invalid_texts = ut.compress(name_text_list, is_invalid_nid_list)
        if (
            len(invalid_nids) == 0
            and invalid_nids[0] == const.UNKNOWN_NAME_ROWID
            and invalid_texts[0] == const.UNKNOWN
        ):
            print('[ibs] found bad name rowids = %r' % (invalid_nids,))
            print('[ibs] found bad name texts  = %r' % (invalid_texts,))
            ibs.delete_names([const.UNKNOWN_NAME_ROWID])
        else:
            errmsg = 'Unfixable error: Found invalid (nid, text) pairs: '
            errmsg += ut.repr2(list(zip(invalid_nids, invalid_texts)))
            raise AssertionError(errmsg)


@register_ibs_method
def fix_invalid_name_texts(ibs):
    r"""
    Ensure  that no name text is empty or '____'

    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.ibsfuncs --test-fix_invalid_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> result = fix_invalid_name_texts(ibs)
        >>> print(result)

    ibs.set_name_texts(nid_list[3], '____')
    ibs.set_name_texts(nid_list[2], '')
    """
    print('checking for invalid name texts')
    # Get actual rowids from sql database (no postprocessing)
    nid_list = ibs._get_all_known_name_rowids()
    # Get actual names from sql database (no postprocessing)
    name_text_list = ibs.get_name_texts(nid_list, apply_fix=False)
    invalid_name_set = {'', const.UNKNOWN}
    is_invalid_name_text_list = [
        name_text in invalid_name_set for name_text in name_text_list
    ]
    if any(is_invalid_name_text_list):
        invalid_nids = ut.compress(nid_list, is_invalid_name_text_list)
        invalid_texts = ut.compress(name_text_list, is_invalid_name_text_list)
        for count, (invalid_nid, invalid_text) in enumerate(
            zip(invalid_nids, invalid_texts)
        ):
            conflict_set = invalid_name_set.union(
                set(ibs.get_name_texts(nid_list, apply_fix=False))
            )
            base_str = 'fixedname%d' + invalid_text
            new_text = ut.get_nonconflicting_string(base_str, conflict_set, offset=count)
            print('Fixing name %r -> %r' % (invalid_text, new_text))
            ibs.set_name_texts((invalid_nid,), (new_text,))
        print('Fixed %d name texts' % (len(invalid_nids)))
    else:
        print('all names seem valid')


@register_ibs_method
def copy_imagesets(ibs, imgsetid_list):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        imgsetid_list (list):

    Returns:
        list: new_imgsetid_list

    CommandLine:
        python -m wbia.other.ibsfuncs --test-copy_imagesets

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs.delete_all_imagesets()
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> imgsetid_list = ibs.get_valid_imgsetids()
        >>> new_imgsetid_list = copy_imagesets(ibs, imgsetid_list)
        >>> result = str(ibs.get_imageset_text(new_imgsetid_list))
        >>> assert [2] == list(set(map(len, ibs.get_image_imgsetids(ibs.get_valid_gids()))))
        >>> print(result)
        >>> ibs.delete_all_imagesets()
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
    """
    all_imagesettext_list = ibs.get_imageset_text(ibs.get_valid_imgsetids())
    imagesettext_list = ibs.get_imageset_text(imgsetid_list)
    new_imagesettext_list = [
        ut.get_nonconflicting_string(
            imagesettext + '_Copy(%d)', set(all_imagesettext_list)
        )
        for imagesettext in imagesettext_list
    ]
    new_imgsetid_list = ibs.add_imagesets(new_imagesettext_list)
    gids_list = ibs.get_imageset_gids(imgsetid_list)
    # new_imgsetid_list =
    for gids, new_imgsetid in zip(gids_list, new_imgsetid_list):
        ibs.set_image_imgsetids(gids, [new_imgsetid] * len(gids))
    return new_imgsetid_list


@register_ibs_method
def fix_unknown_exemplars(ibs):
    """
    Goes through all of the annotations, and sets their exemplar flag to 0 if it
    is associated with an unknown annotation
    """
    aid_list = ibs.get_valid_aids()
    # nid_list = ibs.get_annot_nids(aid_list, distinguish_unknowns=False)
    flag_list = ibs.get_annot_exemplar_flags(aid_list)
    unknown_list = ibs.is_aid_unknown(aid_list)
    # Exemplars should all be known
    unknown_exemplar_flags = ut.compress(flag_list, unknown_list)
    unknown_aid_list = ut.compress(aid_list, unknown_list)
    print(
        'Fixing %d unknown annotations set as exemplars' % (sum(unknown_exemplar_flags),)
    )
    ibs.set_annot_exemplar_flags(unknown_aid_list, [False] * len(unknown_aid_list))
    # is_error = [not flag for flag in unknown_exemplar_flags]
    # new_annots = [flag if nid != const.UNKNOWN_NAME_ROWID else 0
    #              for nid, flag in
    #              zip(nid_list, flag_list)]
    # ibs.set_annot_exemplar_flags(aid_list, new_annots)
    pass


@register_ibs_method
def delete_all_recomputable_data(ibs):
    """ Delete all cached data including chips and imagesets """
    print('[ibs] delete_all_recomputable_data')
    ibs.delete_cachedir()
    ibs.delete_all_chips()
    ibs.delete_all_imagesets()
    print('[ibs] finished delete_all_recomputable_data')


@register_ibs_method
def delete_cache(ibs, delete_imagesets=False):
    """
    Deletes the cache directory in the database directory.
    Can specify to delete encoutners as well.

    CommandLine:
        python -m wbia delete_cache --db testdb1

    Example:
        >>> # SCRIPT
        >>> import wbia
        >>> ibs = wbia.opendb()
        >>> result = ibs.delete_cache()
    """
    ibs.ensure_directories()
    ibs.delete_cachedir()
    ibs.ensure_directories()
    if delete_imagesets:
        ibs.delete_all_imagesets()


@register_ibs_method
def delete_cachedir(ibs):
    """
    Deletes the cache directory in the database directory.

    CommandLine:
        python -m wbia.other.ibsfuncs delete_cachedir
        python -m wbia delete_cachedir --db testdb1

    Example:
        >>> # SCRIPT
        >>> import wbia
        >>> ibs = wbia.opendb()
        >>> result = ibs.delete_cachedir()
    """
    print('[ibs] delete_cachedir')
    # Need to close depc before restarting
    ibs._close_depcache()
    cachedir = ibs.get_cachedir()
    print('[ibs] cachedir=%r' % cachedir)
    ut.delete(cachedir)
    print('[ibs] finished delete cachedir')
    # Reinit cache
    ibs.ensure_directories()
    ibs._init_depcache()


@register_ibs_method
def delete_qres_cache(ibs):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia --tf delete_qres_cache
        python -m wbia --tf delete_qres_cache --db PZ_MTEST
        python -m wbia --tf delete_qres_cache --db PZ_Master1

    Example:
        >>> # SCRIPT
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> result = delete_qres_cache(ibs)
        >>> print(result)
    """
    print('[ibs] delete delete_qres_cache')
    qreq_cachedir = ibs.get_qres_cachedir()
    qreq_bigcachedir = ibs.get_big_cachedir()
    # Preliminary-ensure
    ut.ensuredir(qreq_bigcachedir)
    ut.ensuredir(qreq_cachedir)
    ut.delete(qreq_cachedir, verbose=ut.VERBOSE)
    ut.delete(qreq_bigcachedir, verbose=ut.VERBOSE)
    # Re-ensure
    ut.ensuredir(qreq_bigcachedir)
    ut.ensuredir(qreq_cachedir)
    print('[ibs] finished delete_qres_cache')


@register_ibs_method
def delete_neighbor_cache(ibs):
    print('[ibs] delete neighbor_cache')
    neighbor_cachedir = ibs.get_neighbor_cachedir()
    ut.delete(neighbor_cachedir)
    ut.ensuredir(neighbor_cachedir)
    print('[ibs] finished delete neighbor_cache')


@register_ibs_method
def delete_all_features(ibs):
    print('[ibs] delete_all_features')
    all_fids = ibs._get_all_fids()
    ibs.delete_features(all_fids)
    print('[ibs] finished delete_all_features')


@register_ibs_method
def delete_all_chips(ibs):
    print('[ibs] delete_all_chips')
    ut.ensuredir(ibs.chipdir)
    ibs.delete_annot_chipl(ibs.get_valid_aids())
    ut.delete(ibs.chipdir)
    ut.ensuredir(ibs.chipdir)
    print('[ibs] finished delete_all_chips')


@register_ibs_method
def delete_all_imagesets(ibs):
    print('[ibs] delete_all_imagesets')
    all_imgsetids = ibs._get_all_imgsetids()
    ibs.delete_imagesets(all_imgsetids)
    print('[ibs] finished delete_all_imagesets')


@register_ibs_method
def delete_all_annotations(ibs):
    """ Carefull with this function. Annotations are not recomputable """
    print('[ibs] delete_all_annotations')
    ans = six.moves.input('Are you sure you want to delete all annotations?')
    if ans != 'yes':
        return
    all_aids = ibs._get_all_aids()
    ibs.delete_annots(all_aids)
    print('[ibs] finished delete_all_annotations')


@register_ibs_method
def delete_thumbnails(ibs):
    gid_list = ibs.get_valid_gids()
    ibs.delete_image_thumbs(gid_list)
    ibs.delete_annot_imgthumbs(ibs.get_valid_aids())


@register_ibs_method
def delete_flann_cachedir(ibs):
    print('[ibs] delete_flann_cachedir')
    flann_cachedir = ibs.get_flann_cachedir()
    ut.remove_files_in_dir(flann_cachedir)


def delete_wbia_database(dbdir):
    _ibsdb = join(dbdir, const.PATH_NAMES._ibsdb)
    print('[ibsfuncs] DELETEING: _ibsdb=%r' % _ibsdb)
    if exists(_ibsdb):
        ut.delete(_ibsdb)


@register_ibs_method
def vd(ibs):
    ibs.view_dbdir()


@register_ibs_method
def view_dbdir(ibs):
    ut.view_directory(ibs.get_dbdir())


@register_ibs_method
@accessor_decors.getter_1to1
def is_nid_unknown(ibs, nid_list):
    return [nid <= 0 for nid in nid_list]


@register_ibs_method
def set_annot_names_to_next_name(ibs, aid_list):
    next_name = ibs.make_next_name()
    ibs.set_annot_names(aid_list, [next_name] * len(aid_list))


@register_ibs_method
def _overwrite_annot_species_to_plains(ibs, aid_list):
    species_list = ['zebra_plains'] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


@register_ibs_method
def _overwrite_annot_species_to_grevys(ibs, aid_list):
    species_list = ['zebra_grevys'] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


@register_ibs_method
def _overwrite_annot_species_to_giraffe(ibs, aid_list):
    species_list = ['giraffe_reticulated'] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


@register_ibs_method
def _overwrite_all_annot_species_to(ibs, species):
    """ THIS OVERWRITES A LOT OF INFO """
    valid_species = ibs.get_all_species_texts()
    assert species in valid_species, repr(species) + 'is not in ' + repr(valid_species)
    aid_list = ibs.get_valid_aids()
    species_list = [species] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


def unflat_map(method, unflat_rowids, **kwargs):
    """
    Uses an wbia lookup function with a non-flat rowid list.
    In essence this is equivilent to map(method, unflat_rowids).
    The utility of this function is that it only calls method once.
    This is more efficient for calls that can take a list of inputs

    Args:
        method        (method):  wbia controller method
        unflat_rowids (list): list of rowid lists

    Returns:
        list of values: unflat_vals

    CommandLine:
        python -m wbia.other.ibsfuncs --test-unflat_map

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> method = ibs.get_annot_name_rowids
        >>> unflat_rowids = ibs.get_name_aids(ibs.get_valid_nids())
        >>> unflat_vals = unflat_map(method, unflat_rowids)
        >>> result = str(unflat_vals)
        >>> print(result)
        [[1, 1], [2, 2], [3], [4], [5], [6], [7]]
    """
    # ut.assert_unflat_level(unflat_rowids, level=1, basetype=(int, uuid.UUID))
    # First flatten the list, and remember the original dimensions
    flat_rowids, reverse_list = ut.invertible_flatten2(unflat_rowids)
    # Then preform the lookup / implicit mapping
    flat_vals = method(flat_rowids, **kwargs)
    if True:
        assert len(flat_vals) == len(flat_rowids), (
            'flat lens not the same, len(flat_vals)=%d len(flat_rowids)=%d'
            % (len(flat_vals), len(flat_rowids),)
        )
    # Then ut.unflatten2 the results to the original input dimensions
    unflat_vals = ut.unflatten2(flat_vals, reverse_list)
    if True:
        assert len(unflat_vals) == len(unflat_rowids), (
            'unflat lens not the same, len(unflat_vals)=%d len(unflat_rowids)=%d'
            % (len(unflat_vals), len(unflat_rowids),)
        )
    return unflat_vals


def _make_unflat_getter_func(flat_getter):
    """ makes an unflat version of an wbia getter """
    if isinstance(flat_getter, types.MethodType):
        # Unwrap fmethods
        func = ut.get_method_func(flat_getter)
    else:
        func = flat_getter
    funcname = get_funcname(func)
    assert funcname.startswith('get_'), 'only works on getters, not: ' + funcname

    # Create new function
    def unflat_getter(self, unflat_rowids, *args, **kwargs):
        # First flatten the list
        flat_rowids, reverse_list = ut.invertible_flatten2(unflat_rowids)
        # Then preform the lookup
        flat_vals = func(self, flat_rowids, *args, **kwargs)
        # Then ut.unflatten2 the list
        unflat_vals = ut.unflatten2(flat_vals, reverse_list)
        return unflat_vals

    set_funcname(unflat_getter, funcname.replace('get_', 'get_unflat_'))
    return unflat_getter


def ensure_unix_gpaths(gpath_list):
    """
    Asserts that all paths are given with forward slashes.
    If not it fixes them
    """
    # if ut.NO_ASSERTS:
    #    return
    try:
        msg = (
            'gpath_list must be in unix format (no backslashes).'
            'Failed on %d-th gpath=%r'
        )
        for count, gpath in enumerate(gpath_list):
            assert gpath.find('\\') == -1, msg % (count, gpath)
    except AssertionError as ex:
        ut.printex(ex, iswarning=True)
        gpath_list = list(map(ut.unixpath, gpath_list))
    return gpath_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_annot_info(ibs, aid_list, default=False, reference_aid=None, **kwargs):
    r"""
    Args:
        ibs (wbia.IBEISController):  wbia controller object
        aid_list (list):  list of annotation rowids
        default (bool): (default = False)

    Returns:
        list: infodict_list

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-get_annot_info --tb

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:2]
        >>> default = True
        >>> infodict_list = ibs.get_annot_info(1, default)
        >>> result = ('infodict_list = %s' % (ut.repr2(infodict_list, nl=4),))
        >>> print(result)
    """
    # TODO rectify and combine with viz_helpers.get_annot_texts
    key_list = []
    vals_list = []

    # if len(aid_list) == 0:
    #    return []

    key = 'aid'
    if kwargs.get(key, default or True):
        vals_list += [aid_list]
        key_list += [key]

    key = 'notes'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_notes(aid_list)]
        key_list += [key]

    key = 'case_tags'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_case_tags(aid_list)]
        key_list += [key]

    key = 'match_tags'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_annotmatch_tags(aid_list)]
        key_list += [key]

    key = 'name'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_names(aid_list)]
        key_list += [key]

    key = 'nid'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_name_rowids(aid_list)]
        key_list += [key]

    key = 'num_gt'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_num_groundtruth(aid_list)]
        key_list += [key]

    key = 'gname'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_image_names(aid_list)]
        key_list += [key]

    key = 'bbox'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_bboxes(aid_list)]
        key_list += [key]

    key = 'viewpoint_code'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_viewpoints(aid_list)]
        key_list += [key]

    key = 'time'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_image_unixtimes(aid_list)]
        key_list += [key]

    key = 'timestr'
    if kwargs.get(key, default):
        unixtimes = ibs.get_annot_image_unixtimes(aid_list)
        vals_list += [list(map(ut.util_time.unixtime_to_datetimestr, unixtimes))]
        key_list += [key]

    key = 'timedelta'
    if kwargs.get(key, default) and reference_aid is not None:
        times = np.array(ibs.get_annot_image_unixtimes_asfloat(aid_list))
        ref_time = ibs.get_annot_image_unixtimes_asfloat(reference_aid)
        vals_list += [(times - ref_time)]
        key_list += [key]

    key = 'quality_text'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_quality_texts(aid_list)]
        key_list += [key]

    key = 'exemplar'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_exemplar_flags(aid_list)]
        key_list += [key]

    infodict_list = [
        {key_: val for key_, val in zip(key_list, vals)} for vals in zip(*vals_list)
    ]
    return infodict_list


def aidstr(aid, ibs=None, notes=False):
    """ Helper to make a string from an aid """
    if not notes:
        return 'aid%d' % (aid,)
    else:
        assert ibs is not None
        notes = ibs.get_annot_notes(aid)
        name = ibs.get_annot_names(aid)
        return 'aid%d-%r-%r' % (aid, str(name), str(notes))


@register_ibs_method
# @ut.time_func
# @profile
def update_exemplar_special_imageset(ibs):
    # FIXME SLOW
    exemplar_imgsetid = ibs.get_imageset_imgsetids_from_text(const.EXEMPLAR_IMAGESETTEXT)
    # ibs.delete_imagesets(exemplar_imgsetid)
    ibs.delete_gsgr_imageset_relations(exemplar_imgsetid)
    # aid_list = ibs.get_valid_aids(is_exemplar=True)
    # gid_list = ut.unique_ordered(ibs.get_annot_gids(aid_list))
    gid_list = list(set(_get_exemplar_gids(ibs)))
    # ibs.set_image_imagesettext(gid_list, [const.EXEMPLAR_IMAGESETTEXT] * len(gid_list))
    ibs.set_image_imgsetids(gid_list, [exemplar_imgsetid] * len(gid_list))


@register_ibs_method
# @ut.time_func
# @profile
def update_reviewed_unreviewed_image_special_imageset(
    ibs, reviewed=True, unreviewed=True
):
    """
    Creates imageset of images that have not been reviewed
    and that have been reviewed (wrt detection)
    """
    # FIXME SLOW
    if unreviewed:
        unreviewed_imgsetid = ibs.get_imageset_imgsetids_from_text(
            const.UNREVIEWED_IMAGE_IMAGESETTEXT
        )
        ibs.delete_gsgr_imageset_relations(unreviewed_imgsetid)
        unreviewed_gids = _get_unreviewed_gids(ibs)  # hack
        ibs.set_image_imgsetids(
            unreviewed_gids, [unreviewed_imgsetid] * len(unreviewed_gids)
        )
    if reviewed:
        reviewed_imgsetid = ibs.get_imageset_imgsetids_from_text(
            const.REVIEWED_IMAGE_IMAGESETTEXT
        )
        ibs.delete_gsgr_imageset_relations(reviewed_imgsetid)
        # gid_list = ibs.get_valid_gids(reviewed=False)
        # ibs.set_image_imagesettext(gid_list, [const.UNREVIEWED_IMAGE_IMAGESETTEXT] * len(gid_list))
        reviewed_gids = _get_reviewed_gids(ibs)  # hack
        ibs.set_image_imgsetids(reviewed_gids, [reviewed_imgsetid] * len(reviewed_gids))


@register_ibs_method
# @ut.time_func
# @profile
def update_all_image_special_imageset(ibs):
    # FIXME SLOW
    allimg_imgsetid = ibs.get_imageset_imgsetids_from_text(const.ALL_IMAGE_IMAGESETTEXT)
    # ibs.delete_imagesets(allimg_imgsetid)
    gid_list = ibs.get_valid_gids()
    # ibs.set_image_imagesettext(gid_list, [const.ALL_IMAGE_IMAGESETTEXT] * len(gid_list))
    ibs.set_image_imgsetids(gid_list, [allimg_imgsetid] * len(gid_list))


@register_ibs_method
def update_species_imagesets(ibs):
    gid_list = ibs.get_valid_gids()
    aids_list = ibs.get_image_aids(gid_list)
    species_list = map(ibs.get_annot_species_texts, aids_list)
    species_list = map(set, species_list)

    species_dict = {}
    for species_list_, gid in zip(species_list, gid_list):
        for species in species_list_:
            if species not in species_dict:
                species_dict[species] = []
            species_dict[species].append(gid)

    key_list = sorted(species_dict.keys())
    imageset_text_list = [
        'Species: %s' % (key.replace('_', ' ').title(),) for key in key_list
    ]

    imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(imageset_text_list)
    for key, imageset_rowid_list in zip(key_list, imageset_rowid_list):
        gid_list = species_dict[key]
        gid_list = list(set(gid_list))
        ibs.set_image_imgsetids(gid_list, [imageset_rowid_list] * len(gid_list))


@register_ibs_method
def get_special_imgsetids(ibs):
    get_imagesettext_imgsetid = ibs.get_imageset_imgsetids_from_text
    special_imagesettext_list = [
        const.UNGROUPED_IMAGES_IMAGESETTEXT,
        const.ALL_IMAGE_IMAGESETTEXT,
        const.UNREVIEWED_IMAGE_IMAGESETTEXT,
        const.REVIEWED_IMAGE_IMAGESETTEXT,
        const.EXEMPLAR_IMAGESETTEXT,
    ]
    special_imgsetids_ = [
        get_imagesettext_imgsetid(imagesettext, ensure=False)
        for imagesettext in special_imagesettext_list
    ]
    special_imgsetids = [i for i in special_imgsetids_ if i is not None]
    return special_imgsetids


@register_ibs_method
@profile
def get_ungrouped_gids(ibs):
    """
    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_ungrouped_gids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs.delete_all_imagesets()
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> ibs.update_special_imagesets()
        >>> # Now we want to remove some images from a non-special imageset
        >>> nonspecial_imgsetids = [i for i in ibs.get_valid_imgsetids() if i not in ibs.get_special_imgsetids()]
        >>> print("Nonspecial EIDs %r" % nonspecial_imgsetids)
        >>> images_to_remove = ibs.get_imageset_gids(nonspecial_imgsetids[0:1])[0][0:1]
        >>> print("Removing %r" % images_to_remove)
        >>> ibs.unrelate_images_and_imagesets(images_to_remove,nonspecial_imgsetids[0:1] * len(images_to_remove))
        >>> ibs.update_special_imagesets()
        >>> ungr_imgsetid = ibs.get_imageset_imgsetids_from_text(const.UNGROUPED_IMAGES_IMAGESETTEXT)
        >>> print("Ungrouped gids %r" % ibs.get_ungrouped_gids())
        >>> print("Ungrouped imgsetid %d contains %r" % (ungr_imgsetid, ibs.get_imageset_gids([ungr_imgsetid])))
        >>> ungr_gids = ibs.get_imageset_gids([ungr_imgsetid])[0]
        >>> assert(sorted(images_to_remove) == sorted(ungr_gids))
    """
    special_imgsetids = set(get_special_imgsetids(ibs))
    gid_list = ibs.get_valid_gids()
    imgsetids_list = ibs.get_image_imgsetids(gid_list)
    has_imgsetids = [
        special_imgsetids.issuperset(set(imgsetids)) for imgsetids in imgsetids_list
    ]
    ungrouped_gids = ut.compress(gid_list, has_imgsetids)
    return ungrouped_gids


@register_ibs_method
# @ut.time_func
# @profile
@profile
def update_ungrouped_special_imageset(ibs):
    """
    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.ibsfuncs --test-update_ungrouped_special_imageset

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb9')
        >>> result = update_ungrouped_special_imageset(ibs)
        >>> print(result)
    """
    # FIXME SLOW
    if ut.VERBOSE:
        print('[ibsfuncs] update_ungrouped_special_imageset.1')
    ungrouped_imgsetid = ibs.get_imageset_imgsetids_from_text(
        const.UNGROUPED_IMAGES_IMAGESETTEXT
    )
    if ut.VERBOSE:
        print('[ibsfuncs] update_ungrouped_special_imageset.2')
    ibs.delete_gsgr_imageset_relations(ungrouped_imgsetid)
    if ut.VERBOSE:
        print('[ibsfuncs] update_ungrouped_special_imageset.3')
    ungrouped_gids = ibs.get_ungrouped_gids()
    if ut.VERBOSE:
        print('[ibsfuncs] update_ungrouped_special_imageset.4')
    ibs.set_image_imgsetids(ungrouped_gids, [ungrouped_imgsetid] * len(ungrouped_gids))
    if ut.VERBOSE:
        print('[ibsfuncs] update_ungrouped_special_imageset.5')


@register_ibs_method
# @ut.time_func
@profile
def update_special_imagesets(ibs, use_more_special_imagesets=False):
    if ut.get_argflag('--readonly-mode'):
        # SUPER HACK
        return
    # FIXME SLOW
    if use_more_special_imagesets:
        # ibs.update_reviewed_unreviewed_image_special_imageset()
        ibs.update_reviewed_unreviewed_image_special_imageset(reviewed=False)
        ibs.update_exemplar_special_imageset()
        ibs.update_all_image_special_imageset()
    ibs.update_ungrouped_special_imageset()


# def _get_unreviewed_gids(ibs):
#     # hack
#     gid_list = ibs.db.executeone(
#         '''
#         SELECT image_rowid
#         FROM {IMAGE_TABLE}
#         WHERE
#         image_toggle_reviewed=0
#         '''.format(**const.__dict__))
#     return gid_list


# def _get_reviewed_gids(ibs):
#     # hack
#     gid_list = ibs.db.executeone(
#         '''
#         SELECT image_rowid
#         FROM {IMAGE_TABLE}
#         WHERE
#         image_toggle_reviewed=1
#         '''.format(**const.__dict__))
#     return gid_list


def _get_unreviewed_gids(ibs):
    """
    >>> import wbia  # NOQA
    >>> ibs = wbia.opendb('testdb1')
    """
    # hack
    gid_list = ibs.get_valid_gids()
    flag_list = ibs.detect_cnn_yolo_exists(gid_list)
    gid_list_ = ut.compress(gid_list, ut.not_list(flag_list))
    # unreviewed and unshipped
    imgsets_list = ibs.get_image_imgsetids(gid_list_)
    nonspecial_imgset_ids = [
        ut.compress(ids, ut.not_list(mask))
        for mask, ids in zip(
            ibs.unflat_map(ibs.is_special_imageset, imgsets_list), imgsets_list
        )
    ]
    flags_list = ibs.unflat_map(ibs.get_imageset_shipped_flags, nonspecial_imgset_ids)
    # Keep images that have at least one instance in an unshipped non-special set
    flag_list = [not all(flags) for flags in flags_list]
    gid_list_ = ut.compress(gid_list_, flag_list)
    # Filter by if the user has specified the image has been reviewed manually
    flag_list = ibs.get_image_reviewed(gid_list_)
    gid_list_ = ut.compress(gid_list_, ut.not_list(flag_list))
    return gid_list_


def _get_reviewed_gids(ibs):
    gid_list = ibs.get_valid_gids()
    OLD = False
    if OLD:
        # hack
        flag_list = ibs.detect_cnn_yolo_exists(gid_list)
        gid_list_ = ut.filter_items(gid_list, flag_list)
    else:
        flag_list = ibs.get_image_reviewed(gid_list)
        gid_list_ = ut.compress(gid_list, flag_list)
    return gid_list_


def _get_gids_in_imgsetid(ibs, imgsetid):
    gid_list = ibs.db.executeone(
        """
        SELECT image_rowid
        FROM {GSG_RELATION_TABLE}
        WHERE
            imageset_rowid==?
        """.format(
            **const.__dict__
        ),
        params=(imgsetid,),
    )
    return gid_list


def _get_dirty_reviewed_gids(ibs, imgsetid):
    gid_list = ibs.db.executeone(
        """
        SELECT image_rowid
        FROM {GSG_RELATION_TABLE}
        WHERE
            imageset_rowid==? AND
            image_rowid NOT IN (SELECT rowid FROM {IMAGE_TABLE} WHERE image_toggle_reviewed=1)
        """.format(
            **const.__dict__
        ),
        params=(imgsetid,),
    )
    return gid_list


def _get_exemplar_gids(ibs):
    gid_list = ibs.db.executeone(
        """
        SELECT image_rowid
        FROM {ANNOTATION_TABLE}
        WHERE annot_exemplar_flag=1
        """.format(
            **const.__dict__
        )
    )
    return gid_list


@register_ibs_method
def print_dbinfo(ibs, **kwargs):
    from wbia.other import dbinfo

    dbinfo.get_dbinfo(ibs, *kwargs)


@register_ibs_method
def print_infostr(ibs, **kwargs):
    print(ibs.get_infostr())


@register_ibs_method
def print_annotation_table(ibs, verbosity=1, exclude_columns=[], include_columns=[]):
    """
    Dumps annotation table to stdout

    Args:
        ibs (IBEISController):
        verbosity (int):

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> verbosity = 1
        >>> print_annotation_table(ibs, verbosity)
    """
    exclude_columns = exclude_columns[:]
    if verbosity < 5:
        exclude_columns += ['annot_uuid', 'annot_verts']
    if verbosity < 4:
        exclude_columns += [
            'annot_xtl',
            'annot_ytl',
            'annot_width',
            'annot_height',
            'annot_theta',
            'annot_yaw',
            'annot_detect_confidence',
            'annot_note',
            'annot_parent_rowid',
        ]
    for x in include_columns:
        try:
            exclude_columns.remove(x)
        except ValueError:
            pass
    print('\n')
    print(ibs.db.get_table_csv(const.ANNOTATION_TABLE, exclude_columns=exclude_columns))


@register_ibs_method
def print_annotmatch_table(ibs):
    """
    Dumps annotation match table to stdout

    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-print_annotmatch_table
        python -m wbia.other.ibsfuncs --exec-print_annotmatch_table --db PZ_Master1

    Example:
        >>> # SCRIPT
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> result = print_annotmatch_table(ibs)
        >>> print(result)
    """
    print('\n')
    exclude_columns = ['annotmatch_confidence']
    print(ibs.db.get_table_csv(const.ANNOTMATCH_TABLE, exclude_columns=exclude_columns))


@register_ibs_method
def print_chip_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.CHIP_TABLE))


@register_ibs_method
def print_feat_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(
        ibs.db.get_table_csv(
            const.FEATURE_TABLE, exclude_columns=['feature_keypoints', 'feature_vecs']
        )
    )


@register_ibs_method
def print_image_table(ibs, **kwargs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.IMAGE_TABLE, **kwargs))
    # , exclude_columns=['image_rowid']))


@register_ibs_method
def print_party_table(ibs, **kwargs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.PARTY_TABLE, **kwargs))
    # , exclude_columns=['image_rowid']))


@register_ibs_method
def print_lblannot_table(ibs, **kwargs):
    """ Dumps lblannot table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.LBLANNOT_TABLE, **kwargs))


@register_ibs_method
def print_name_table(ibs, **kwargs):
    """ Dumps name table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.NAME_TABLE, **kwargs))


@register_ibs_method
def print_species_table(ibs, **kwargs):
    """ Dumps species table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.SPECIES_TABLE, **kwargs))


@register_ibs_method
def print_alr_table(ibs, **kwargs):
    """ Dumps alr table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.AL_RELATION_TABLE, **kwargs))


@register_ibs_method
def print_config_table(ibs, **kwargs):
    """ Dumps config table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.CONFIG_TABLE, **kwargs))


@register_ibs_method
def print_imageset_table(ibs, **kwargs):
    """ Dumps imageset table to stdout

    Kwargs:
        exclude_columns (list):
    """
    print('\n')
    print(ibs.db.get_table_csv(const.IMAGESET_TABLE, **kwargs))


@register_ibs_method
def print_egpairs_table(ibs, **kwargs):
    """ Dumps egpairs table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.GSG_RELATION_TABLE, **kwargs))


@register_ibs_method
def print_tables(ibs, exclude_columns=None, exclude_tables=None):
    if exclude_columns is None:
        exclude_columns = [
            'annot_uuid',
            'lblannot_uuid',
            'annot_verts',
            'feature_keypoints',
            'feature_vecs',
            'image_uuid',
            'image_uri',
        ]
    if exclude_tables is None:
        exclude_tables = ['masks', 'recognitions', 'chips', 'features']
    for table_name in ibs.db.get_table_names():
        if table_name in exclude_tables:
            continue
        print('\n')
        print(ibs.db.get_table_csv(table_name, exclude_columns=exclude_columns))
    # ibs.print_image_table()
    # ibs.print_annotation_table()
    # ibs.print_lblannots_table()
    # ibs.print_alr_table()
    # ibs.print_config_table()
    # ibs.print_chip_table()
    # ibs.print_feat_table()
    print('\n')


@register_ibs_method
def print_contributor_table(ibs, verbosity=1, exclude_columns=[]):
    """
    Dumps annotation table to stdout

    Args:
        ibs (IBEISController):
        verbosity (int):

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> verbosity = 1
        >>> print_contributor_table(ibs, verbosity)
    """
    exclude_columns = exclude_columns[:]
    if verbosity < 5:
        exclude_columns += ['contributor_uuid']
        exclude_columns += ['contributor_location_city']
        exclude_columns += ['contributor_location_state']
        exclude_columns += ['contributor_location_country']
        exclude_columns += ['contributor_location_zip']
    print('\n')
    print(ibs.db.get_table_csv(const.CONTRIBUTOR_TABLE, exclude_columns=exclude_columns))


@register_ibs_method
def is_aid_unknown(ibs, aid_list):
    """ Returns if an annotation has been given a name (even if that name is temporary) """
    nid_list = ibs.get_annot_name_rowids(aid_list)
    return ibs.is_nid_unknown(nid_list)


@register_ibs_method
def batch_rename_consecutive_via_species(
    ibs, imgsetid=None, location_text=None, notify_wildbook=True, assert_wildbook=True
):
    import wbia

    wildbook_existing_name_list = []
    if notify_wildbook and wbia.ENABLE_WILDBOOK_SIGNAL:
        wildbook_existing_name_list = ibs.wildbook_get_existing_names()
        if wildbook_existing_name_list is None:
            wildbook_existing_name_list = []
    else:
        wildbook_existing_name_list = []

    """ actually sets the new consecutive names"""
    new_nid_list, new_name_list = ibs.get_consecutive_newname_list_via_species(
        imgsetid=imgsetid,
        location_text=location_text,
        wildbook_existing_name_list=wildbook_existing_name_list,
    )

    def get_conflict_names(ibs, new_nid_list, new_name_list):
        other_nid_list = list(set(ibs.get_valid_nids()) - set(new_nid_list))
        other_names = ibs.get_name_texts(other_nid_list)
        conflict_names = list(set(other_names).intersection(set(new_name_list)))
        return conflict_names

    def _assert_no_name_conflicts(ibs, new_nid_list, new_name_list):
        print('checking for conflicting names')
        conflit_names = get_conflict_names(ibs, new_nid_list, new_name_list)
        assert len(conflit_names) == 0, 'conflit_names=%r' % (conflit_names,)

    # Check to make sure new names dont conflict with other names
    _assert_no_name_conflicts(ibs, new_nid_list, new_name_list)
    ibs.set_name_texts(
        new_nid_list,
        new_name_list,
        verbose=ut.NOT_QUIET,
        notify_wildbook=notify_wildbook,
        assert_wildbook=assert_wildbook,
    )


def get_location_text(ibs, location_text, default_location_text):
    if location_text is None:
        # Check for Lewa server
        comp_name = ut.get_computer_name()
        db_name = ibs.dbname
        is_lewa = comp_name in ['wbia.cs.uic.edu'] or db_name in ['LEWA', 'lewa_grevys']
        if is_lewa:
            location_text = 'LWC'
        else:
            location_text = default_location_text
    return location_text


@register_ibs_method
def get_consecutive_newname_list_via_species(
    ibs, imgsetid=None, location_text=None, wildbook_existing_name_list=[]
):
    """
    Just creates the nams, but does not set them

    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_consecutive_newname_list_via_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs._clean_species()
        >>> imgsetid = None
        >>> new_nid_list, new_name_list = get_consecutive_newname_list_via_species(ibs, imgsetid=imgsetid)
        >>> result = ut.repr2((new_nid_list, new_name_list), nl=1)
        >>> print(result)
        (
            [1, 2, 3, 4, 5, 6, 7],
            ['WBIA_PZ_0001', 'WBIA_PZ_0002', 'WBIA_UNKNOWN_0001', 'WBIA_UNKNOWN_0002', 'WBIA_GZ_0001', 'WBIA_PB_0001', 'WBIA_UNKNOWN_0003'],
        )

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs._clean_species()
        >>> ibs.delete_all_imagesets()
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> imgsetid = ibs.get_valid_imgsetids()[1]
        >>> new_nid_list, new_name_list = get_consecutive_newname_list_via_species(ibs, imgsetid=imgsetid)
        >>> result = ut.repr2((new_nid_list, new_name_list), nl=1)
        >>> print(result)
        (
            [4, 5, 6, 7],
            ['WBIA_UNKNOWN_Occurrence_1_0001', 'WBIA_GZ_Occurrence_1_0001', 'WBIA_PB_Occurrence_1_0001', 'WBIA_UNKNOWN_Occurrence_1_0002'],
        )
    """
    wildbook_existing_name_set = set(wildbook_existing_name_list)
    args = (len(wildbook_existing_name_set),)
    print(
        '[ibs] get_consecutive_newname_list_via_species with %d existing WB names' % args
    )
    location_text = get_location_text(ibs, location_text, 'IBEIS')
    ibs.delete_empty_nids()
    nid_list = ibs.get_valid_nids(imgsetid=imgsetid)
    # name_list = ibs.get_name_texts(nid_list)
    aids_list = ibs.get_name_aids(nid_list)

    species_rowids_list = ibs.unflat_map(ibs.get_annot_species_rowids, aids_list)
    unique_species_rowids_list = list(map(ut.unique_ordered, species_rowids_list))
    species_rowid_list = ut.flatten(unique_species_rowids_list)
    try:
        assert len(species_rowid_list) == len(nid_list)
    except AssertionError:
        print('WARNING: Names assigned to annotations with inconsistent species')
        inconsistent_nid_list = []
        for nid, unique_species_rowid_list in zip(nid_list, unique_species_rowids_list):
            if len(unique_species_rowid_list) > 1:
                inconsistent_nid_list.append(nid)
        message = 'Inconsistent nid_list = %r' % (inconsistent_nid_list,)
        raise ValueError(message)
    code_list = ibs.get_species_codes(species_rowid_list)

    _code2_count = ut.ddict(lambda: 0)

    def get_next_index(code):
        _code2_count[code] += 1
        return _code2_count[code]

    def get_new_name(code):
        if imgsetid is not None:
            imgset_text = ibs.get_imageset_text(imgsetid)
            imgset_text = imgset_text.replace(' ', '_').replace("'", '').replace('"', '')
            args = (
                location_text,
                code,
                imgset_text,
                get_next_index(code),
            )
            new_name = '%s_%s_%s_%04d' % args
        else:
            args = (
                location_text,
                code,
                get_next_index(code),
            )
            new_name = '%s_%s_%04d' % args
        return new_name

    new_name_list = []
    for code in code_list:
        new_name = get_new_name(code)
        while new_name in wildbook_existing_name_set:
            new_name = get_new_name(code)
        new_name_list.append(new_name)

    new_nid_list = nid_list
    assert len(new_nid_list) == len(new_name_list)
    return new_nid_list, new_name_list


@register_ibs_method
def set_annot_names_to_same_new_name(ibs, aid_list):
    new_nid = ibs.make_next_nids(num=1)[0]
    if ut.VERBOSE:
        print(
            'Setting aid_list={aid_list} to have new_nid={new_nid}'.format(
                aid_list=aid_list, new_nid=new_nid
            )
        )
    ibs.set_annot_name_rowids(aid_list, [new_nid] * len(aid_list))


@register_ibs_method
def set_annot_names_to_different_new_names(ibs, aid_list, **kwargs):
    new_nid_list = ibs.make_next_nids(num=len(aid_list))
    if ut.VERBOSE:
        print(
            'Setting aid_list={aid_list} to have new_nid_list={new_nid_list}'.format(
                aid_list=aid_list, new_nid_list=new_nid_list
            )
        )
    ibs.set_annot_name_rowids(aid_list, new_nid_list, **kwargs)


@register_ibs_method
def make_next_nids(ibs, num=None, str_format=2, species_text=None, location_text=None):
    """
    makes name and adds it to the database returning the newly added name rowid(s)

    CAUTION; changes database state

    SeeAlso:
        make_next_name
    """
    next_names = ibs.make_next_name(
        num=num,
        str_format=str_format,
        species_text=species_text,
        location_text=location_text,
    )
    next_nids = ibs.add_names(next_names)
    return next_nids


@register_ibs_method
def make_next_name(ibs, num=None, str_format=2, species_text=None, location_text=None):
    """ Creates a number of names which are not in the database, but does not
    add them

    Args:
        ibs (IBEISController):  wbia controller object
        num (None):
        str_format (int): either 1 or 2

    Returns:
        str: next_name

    CommandLine:
        python -m wbia.other.ibsfuncs --test-make_next_name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs1 = wbia.opendb('testdb1')
        >>> ibs2 = wbia.opendb('PZ_MTEST')
        >>> ibs3 = wbia.opendb('NAUT_test')
        >>> ibs1._clean_species()
        >>> ibs2._clean_species()
        >>> ibs3._clean_species()
        >>> num = None
        >>> str_format = 2
        >>> next_name1 = make_next_name(ibs1, num, str_format)
        >>> next_name2 = make_next_name(ibs2, num, str_format)
        >>> next_name3 = make_next_name(ibs3, num, str_format)
        >>> next_name4 = make_next_name(ibs1, num, str_format, const.TEST_SPECIES.ZEB_GREVY)
        >>> name_list = [next_name1, next_name2, next_name3, next_name4]
        >>> next_name_list1 = make_next_name(ibs2, 5, str_format)
        >>> temp_nids = ibs2.add_names(['WBIA_PZ_0045', 'WBIA_PZ_0048'])
        >>> next_name_list2 = make_next_name(ibs2, 5, str_format)
        >>> ibs2.delete_names(temp_nids)
        >>> next_name_list3 = make_next_name(ibs2, 5, str_format)
        >>> # FIXME: nautiluses are not working right
        >>> names = (name_list, next_name_list1, next_name_list2, next_name_list3)
        >>> result = ut.repr4(names)
        >>> print(result)
        (
            ['WBIA_PZ_0008', 'WBIA_PZ_0042', 'WBIA_UNKNOWN_0004', 'WBIA_GZ_0008'],
            ['WBIA_PZ_0042', 'WBIA_PZ_0043', 'WBIA_PZ_0044', 'WBIA_PZ_0045', 'WBIA_PZ_0046'],
            ['WBIA_PZ_0044', 'WBIA_PZ_0046', 'WBIA_PZ_0047', 'WBIA_PZ_0049', 'WBIA_PZ_0050'],
            ['WBIA_PZ_0042', 'WBIA_PZ_0043', 'WBIA_PZ_0044', 'WBIA_PZ_0045', 'WBIA_PZ_0046'],
        )

    """
    # HACK TO FORCE TIMESTAMPS FOR NEW NAMES
    # str_format = 1
    if species_text is None:
        # TODO: optionally specify qreq_ or qparams?
        species_text = ibs.cfg.detect_cfg.species_text
    location_text = get_location_text(
        ibs, location_text, ibs.cfg.other_cfg.location_for_names
    )
    if num is None:
        num_ = 1
    else:
        num_ = num
    # Assign new names
    nid_list = ibs._get_all_known_name_rowids()
    names_used_list = set(ibs.get_name_texts(nid_list))
    base_index = len(nid_list)
    next_name_list = []
    while len(next_name_list) < num_:
        base_index += 1
        if str_format == 1:
            user_id = ut.get_user_name()
            timestamp = ut.get_timestamp('tag')
            # timestamp_suffix = '_TMP_'
            timestamp_suffix = '_'
            timestamp_prefix = ''
            name_prefix = timestamp_prefix + timestamp + timestamp_suffix + user_id + '_'
        elif str_format == 2:
            species_rowid = ibs.get_species_rowids_from_text(species_text)
            species_code = ibs.get_species_codes(species_rowid)
            name_prefix = location_text + '_' + species_code + '_'
        else:
            raise ValueError('Invalid str_format supplied')
        next_name = name_prefix + '%04d' % base_index
        if next_name not in names_used_list:
            # names_used_list.add(next_name)
            next_name_list.append(next_name)
    # Return a list or a string
    if num is None:
        return next_name_list[0]
    else:
        return next_name_list


@register_ibs_method
@profile
def group_annots_by_name(ibs, aid_list, distinguish_unknowns=True, assume_unique=False):
    r"""
    This function is probably the fastest of its siblings

    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (list):
        distinguish_unknowns (bool):

    Returns:
        tuple: grouped_aids, unique_nids

    CommandLine:
        python -m wbia.other.ibsfuncs --test-group_annots_by_name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> distinguish_unknowns = True
        >>> grouped_aids, unique_nids = group_annots_by_name(ibs, aid_list, distinguish_unknowns)
        >>> result = str([aids.tolist() for aids in grouped_aids])
        >>> result += '\n' + str(unique_nids.tolist())
        >>> print(result)
        [[11], [9], [4], [1], [2, 3], [5, 6], [7], [8], [10], [12], [13]]
        [-11, -9, -4, -1, 1, 2, 3, 4, 5, 6, 7]
    """
    nid_list = ibs.get_annot_name_rowids(
        aid_list, distinguish_unknowns=distinguish_unknowns, assume_unique=assume_unique
    )
    nid_list = np.array(nid_list)
    aid_list = np.array(aid_list)
    unique_nids, groupxs_list = vt.group_indices(nid_list)
    grouped_aids = vt.apply_grouping(aid_list, groupxs_list)
    return grouped_aids, unique_nids


# def group_annots_by_known_names_nochecks(ibs, aid_list):
#    nid_list = ibs.get_annot_name_rowids(aid_list)
#    nid2_aids = ut.group_items(aid_list, nid_list)
#    return list(nid2_aids.values())


@register_ibs_method
def group_annots_by_known_names(ibs, aid_list, checks=True):
    r"""
    FIXME; rectify this
    #>>> import wbia  # NOQA

    CommandLine:
        python -m wbia.other.ibsfuncs --test-group_annots_by_known_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(db='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        >>> known_aids_list, unknown_aids = group_annots_by_known_names(ibs, aid_list)
        >>> result = ut.repr2(sorted(known_aids_list)) + '\n'
        >>> result += ut.repr2(unknown_aids)
        >>> print(result)
        [[2, 3], [5, 6], [7], [8], [10], [12], [13]]
        [11, 9, 4, 1]
    """
    nid_list = ibs.get_annot_name_rowids(aid_list)
    nid2_aids = ut.group_items(aid_list, nid_list)

    def aid_gen():
        return six.itervalues(nid2_aids)

    isunknown_list = ibs.is_nid_unknown(six.iterkeys(nid2_aids))
    known_aids_list = list(ut.ifilterfalse_items(aid_gen(), isunknown_list))
    unknown_aids = list(ut.iflatten(ut.iter_compress(aid_gen(), isunknown_list)))
    if __debug__:
        # References:
        #     http://stackoverflow.com/questions/482014/how-would-you-do-the-equivalent-of-preprocessor-directives-in-python
        nidgroup_list = unflat_map(ibs.get_annot_name_rowids, known_aids_list)
        for nidgroup in nidgroup_list:
            assert ut.allsame(nidgroup), 'bad name grouping'
    return known_aids_list, unknown_aids


def get_primary_species_viewpoint(species, plus=0):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        species (?):

    Returns:
        str: primary_viewpoint

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-get_primary_species_viewpoint

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> species = wbia.const.TEST_SPECIES.ZEB_PLAIN
        >>> aid_subset = get_primary_species_viewpoint(species, 0)
        >>> result = ('aid_subset = %s' % (str(aid_subset),))
        >>> print(result)
        aid_subset = left
    """
    if species == 'zebra_plains':
        primary_viewpoint = 'left'
    elif species == 'zebra_grevys':
        primary_viewpoint = 'right'
    else:
        primary_viewpoint = 'left'
    if plus != 0:
        # return an augmented primary viewpoint
        primary_viewpoint = get_extended_viewpoints(
            primary_viewpoint, num1=1, num2=0, include_base=False
        )[0]
    return primary_viewpoint


def get_extended_viewpoints(
    base_yaw_text, towards='front', num1=0, num2=None, include_base=True
):
    """
    Given a viewpoint returns the acceptable viewpoints around it

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> yaw_text_list = ['left', 'right', 'back', 'front']
        >>> towards = 'front'
        >>> num1 = 1
        >>> num2 = 0
        >>> include_base = False
        >>> extended_yaws_list = [get_extended_viewpoints(base_yaw_text, towards, num1, num2, include_base)
        >>>                       for base_yaw_text in yaw_text_list]
        >>> result = ('extended_yaws_list = %s' % (ut.repr2(extended_yaws_list),))
        >>> print(result)
        extended_yaws_list = [['frontleft'], ['frontright'], ['backleft'], ['frontleft']]
    """
    import vtool as vt

    # DEPRICATE?
    ori1 = const.VIEWTEXT_TO_YAW_RADIANS[base_yaw_text]
    ori2 = const.VIEWTEXT_TO_YAW_RADIANS[towards]
    # Find which direction to go to get closer to `towards`
    yawdist = vt.signed_ori_distance(ori1, ori2)
    if yawdist == 0:
        # break ties
        print('warning extending viewpoint yaws from the same position as towards')
        yawdist += 1e-3
    if num1 is None:
        num1 = 0
    if num2 is None:
        num2 = num1
    assert num1 >= 0, 'must specify positive num'
    assert num2 >= 0, 'must specify positive num'
    yawtext_list = list(const.VIEW.CODE_TO_INT.keys())
    index = yawtext_list.index(base_yaw_text)
    other_index_list1 = [
        int((index + (np.sign(yawdist) * count)) % len(yawtext_list))
        for count in range(1, num1 + 1)
    ]
    other_index_list2 = [
        int((index - (np.sign(yawdist) * count)) % len(yawtext_list))
        for count in range(1, num2 + 1)
    ]
    if include_base:
        extended_index_list = sorted(
            list(set(other_index_list1 + other_index_list2 + [index]))
        )
    else:
        extended_index_list = sorted(list(set(other_index_list1 + other_index_list2)))
    extended_yaws = ut.take(yawtext_list, extended_index_list)
    return extended_yaws


def get_two_annots_per_name_and_singletons(ibs, onlygt=False):
    """
    makes controlled subset of data

    DEPRICATE

    CONTROLLED TEST DATA

    Build data for experiment that tries to rule out
    as much bad data as possible


    Returns a controlled set of annotations that conforms to
      * number of annots per name
      * uniform species
      * viewpoint restrictions
      * quality restrictions
      * time delta restrictions

    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_two_annots_per_name_and_singletons
        python -m wbia.other.ibsfuncs --test-get_two_annots_per_name_and_singletons --db GZ_ALL
        python -m wbia.other.ibsfuncs --test-get_two_annots_per_name_and_singletons --db PZ_Master0 --onlygt

    Ignore:
        sys.argv.extend(['--db', 'PZ_MTEST'])

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_Master0')
        >>> aid_subset = get_two_annots_per_name_and_singletons(ibs, onlygt=ut.get_argflag('--onlygt'))
        >>> wbia.other.dbinfo.get_dbinfo(ibs, aid_list=aid_subset, with_contrib=False)
        >>> result = str(aid_subset)
        >>> print(result)
    """
    species = get_primary_database_species(ibs, ibs.get_valid_aids())
    # aid_list = ibs.get_valid_aids(species='zebra_plains', is_known=True)
    aid_list = ibs.get_valid_aids(species=species, is_known=True)
    # FILTER OUT UNUSABLE ANNOTATIONS
    # Get annots with timestamps
    aid_list = filter_aids_without_timestamps(ibs, aid_list)
    minqual = 'ok'
    # valid_yaws = {'left', 'frontleft', 'backleft'}
    if species == 'zebra_plains':
        valid_yawtexts = {'left', 'frontleft'}
    elif species == 'zebra_grevys':
        valid_yawtexts = {'right', 'frontright'}
    else:
        valid_yawtexts = {'left', 'frontleft'}
    flags_list = ibs.get_quality_viewpoint_filterflags(aid_list, minqual, valid_yawtexts)
    aid_list = ut.compress(aid_list, flags_list)
    # print('print subset info')
    # print(ut.dict_hist(ibs.get_annot_viewpoints(aid_list)))
    # print(ut.dict_hist(ibs.get_annot_quality_texts(aid_list)))
    singletons, multitons = partition_annots_into_singleton_multiton(ibs, aid_list)
    # process multitons
    hourdists_list = ibs.get_unflat_annots_hourdists_list(multitons)
    # pairxs_list = [vt.pdist_argsort(x) for x in hourdists_list]
    # Get the pictures taken the furthest appart of each gt case
    best_pairx_list = [vt.pdist_argsort(x)[0] for x in hourdists_list]

    best_multitons = np.array(vt.ziptake(multitons, best_pairx_list))
    if onlygt:
        aid_subset = best_multitons.flatten()
    else:
        aid_subset = np.hstack([best_multitons.flatten(), np.array(singletons).flatten()])
    aid_subset.sort()
    return aid_subset


@register_ibs_method
def get_num_annots_per_name(ibs, aid_list):
    """
    Returns the number of annots per name (IN THIS LIST)

    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (int):  list of annotation ids

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-get_num_annots_per_name
        python -m wbia.other.ibsfuncs --exec-get_num_annots_per_name --db PZ_Master1

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids(is_known=True)
        >>> num_annots_per_name, unique_nids = get_num_annots_per_name(ibs, aid_list)
        >>> per_name_hist = ut.dict_hist(num_annots_per_name)
        >>> items = per_name_hist.items()
        >>> items = sorted(items)[::-1]
        >>> key_list = ut.get_list_column(items, 0)
        >>> val_list = ut.get_list_column(items, 1)
        >>> min_per_name = dict(zip(key_list, np.cumsum(val_list)))
        >>> result = ('per_name_hist = %s' % (ut.repr2(per_name_hist),))
        >>> print(result)
        >>> print('min_per_name = %s' % (ut.repr2(min_per_name),))
        per_name_hist = {
            1: 5,
            2: 2,
        }

    """
    aids_list, unique_nids = ibs.group_annots_by_name(aid_list)
    num_annots_per_name = list(map(len, aids_list))
    return num_annots_per_name, unique_nids


@register_ibs_method
def get_annots_per_name_stats(ibs, aid_list, **kwargs):
    stats_kw = dict(use_nan=True)
    stats_kw.update(kwargs)
    return ut.get_stats(ibs.get_num_annots_per_name(aid_list)[0], **stats_kw)


@register_ibs_method
def get_aids_with_groundtruth(ibs):
    """ returns aids with valid groundtruth """
    valid_aids = ibs.get_valid_aids()
    has_gt_list = ibs.get_annot_has_groundtruth(valid_aids)
    hasgt_aids = ut.compress(valid_aids, has_gt_list)
    return hasgt_aids


@register_ibs_method
def get_dbnotes_fpath(ibs, ensure=False):
    notes_fpath = join(ibs.get_ibsdir(), 'dbnotes.txt')
    if ensure and not exists(ibs.get_dbnotes_fpath()):
        ibs.set_dbnotes('None')
    return notes_fpath


@profile
def get_yaw_viewtexts(yaw_list):
    r"""
    Args:
        yaw_list (list of angles):

    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_yaw_viewtexts

    TODO:
        rhombicubeoctehedron

        https://en.wikipedia.org/wiki/Rhombicuboctahedron

        up,
        down,
        front,
        left,
        back,
        right,
        front-left,
        back-left,
        back-right,
        front-right,
        up-front,
        up-left,
        up-back,
        up-right,
        up-front-left,
        up-back-left,
        up-back-right,
        up-front-right,
        down-front,
        down-left,
        down-back,
        down-right,
        down-front-left,
        down-back-left,
        down-back-right,
        down-front-right,

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import numpy as np
        >>> yaw_list = [0.0, np.pi / 2, np.pi / 4, np.pi, 3.15, -.4, -8, .2, 4, 7, 20, None]
        >>> text_list = get_yaw_viewtexts(yaw_list)
        >>> result = ut.repr2(text_list, nl=False)
        >>> print(result)
        ['right', 'front', 'frontright', 'left', 'left', 'backright', 'back', 'right', 'backleft', 'frontright', 'frontright', None]

    """
    # import vtool as vt
    import numpy as np

    stdlblyaw_list = list(const.VIEWTEXT_TO_YAW_RADIANS.items())
    stdlbl_list = ut.get_list_column(stdlblyaw_list, 0)
    # ALTERNATE = False
    # if ALTERNATE:
    #    #with ut.Timer('fdsa'):
    #    TAU = np.pi * 2
    #    binsize = TAU / len(const.VIEWTEXT_TO_YAW_RADIANS)
    #    yaw_list_ = np.array([np.nan if yaw is None else yaw for yaw in yaw_list])
    #    index_list = np.floor(.5 + (yaw_list_ % TAU) / binsize)
    #    text_list = [None if np.isnan(index) else stdlbl_list[int(index)] for index in index_list]
    # else:
    # with ut.Timer('fdsa'):
    stdyaw_list = np.array(ut.take_column(stdlblyaw_list, 1))

    yaw_list

    is_not_none = ut.flag_not_None_items(yaw_list)
    has_nones = not all(is_not_none)
    if has_nones:
        yaw_list_ = ut.compress(yaw_list, is_not_none)
    else:
        yaw_list_ = yaw_list
    yaw_list_ = np.array(yaw_list_)
    textdists = vt.ori_distance(stdyaw_list, yaw_list_[:, None])
    index_list = textdists.argmin(axis=1)
    text_list_ = ut.take(stdlbl_list, index_list)
    if has_nones:
        text_list = ut.ungroup(
            [text_list_], [ut.where(is_not_none)], maxval=len(is_not_none) - 1
        )
    else:
        text_list = text_list_

    # textdists_list = [None if yaw is None else
    #                  vt.ori_distance(stdyaw_list, yaw)
    #                  for yaw in yaw_list]
    # index_list = [None if dists is None else dists.argmin()
    #              for dists in textdists_list]
    # text_list = [None if index is None else stdlbl_list[index] for index in index_list]
    # yaw_list_ / binsize
    # errors = ['%.2f' % dists[index] for dists, index in zip(textdists_list, index_list)]
    # return list(zip(yaw_list, errors, text_list))
    return text_list


def get_species_dbs(species_prefix):
    from wbia.init import sysres

    ibs_dblist = sysres.get_ibsdb_list()
    isvalid_list = [split(path)[1].startswith(species_prefix) for path in ibs_dblist]
    return ut.compress(ibs_dblist, isvalid_list)


@register_ibs_method
def get_annot_bbox_area(ibs, aid_list):
    bbox_list = ibs.get_annot_bboxes(aid_list)
    area_list = [bbox[2] * bbox[3] for bbox in bbox_list]
    return area_list


@register_ibs_method
def get_database_species(ibs, aid_list=None):
    r"""

    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_database_species

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> result = ut.repr2(ibs.get_database_species(), nl=False)
        >>> print(result)
        ['____', 'bear_polar', 'zebra_grevys', 'zebra_plains']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> result = ut.repr2(ibs.get_database_species(), nl=False)
        >>> print(result)
        ['zebra_plains']
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    species_rowids = set(ibs.get_annot_species_rowids(aid_list))
    unique_species = sorted(set(ibs.get_species_texts(species_rowids)))
    return unique_species


@register_ibs_method
def get_primary_database_species(ibs, aid_list=None, speedhack=True):
    r"""
    Args:
        aid_list (list):  list of annotation ids (default = None)

    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_primary_database_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = None
        >>> primary_species = get_primary_database_species(ibs, aid_list)
        >>> result = primary_species
        >>> print('primary_species = %r' % (primary_species,))
        >>> print(result)
        zebra_plains
    """
    if speedhack:
        # Use our conventions
        if ibs.get_dbname().startswith('PZ_'):
            return 'zebra_plains'
        elif ibs.get_dbname() == 'NNP_Master':
            return 'zebra_plains'
        elif ibs.get_dbname().startswith('GZ_'):
            return 'zebra_grevys'
    if aid_list is None:
        aid_list = ibs.get_valid_aids(is_staged=None)
    species_list = ibs.get_annot_species_texts(aid_list)
    species_hist = ut.dict_hist(species_list)
    if len(species_hist) == 0:
        primary_species = const.UNKNOWN
    else:
        frequent_species = sorted(
            species_hist.items(), key=lambda item: item[1], reverse=True
        )
        primary_species = frequent_species[0][0]
    return primary_species


@register_ibs_method
def get_dominant_species(ibs, aid_list):
    r"""
    Args:
        aid_list (int):  list of annotation ids

    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_dominant_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> result = get_dominant_species(ibs, aid_list)
        >>> print(result)
        zebra_plains
    """
    hist_ = ut.dict_hist(ibs.get_annot_species_texts(aid_list))
    keys = list(hist_.keys())
    vals = list(hist_.values())
    species_text = keys[ut.list_argmax(vals)]
    return species_text


@register_ibs_method
def get_database_species_count(ibs, aid_list=None):
    """

    CommandLine:
        python -m wbia.other.ibsfuncs --test-get_database_species_count

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> #print(ut.repr2(wbia.opendb('PZ_Master0').get_database_species_count()))
        >>> ibs = wbia.opendb('testdb1')
        >>> result = ut.repr2(ibs.get_database_species_count(), nl=False)
        >>> print(result)
        {'____': 3, 'bear_polar': 2, 'zebra_grevys': 2, 'zebra_plains': 6}

    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    species_list = ibs.get_annot_species_texts(aid_list)
    species_count_dict = ut.item_hist(species_list)
    return species_count_dict


@register_ibs_method
def get_dbinfo_str(ibs):
    from wbia.other import dbinfo

    return dbinfo.get_dbinfo(ibs, verbose=False)['info_str']


@register_ibs_method
def get_infostr(ibs):
    """ Returns sort printable database information

    Args:
        ibs (IBEISController):  wbia controller object

    Returns:
        str: infostr
    """
    from wbia.other import dbinfo

    return dbinfo.get_short_infostr(ibs)


@register_ibs_method
def get_dbnotes(ibs):
    """ sets notes for an entire database """
    notes = ut.read_from(ibs.get_dbnotes_fpath(), strict=False)
    if notes is None:
        ibs.set_dbnotes('None')
        notes = ut.read_from(ibs.get_dbnotes_fpath())
    return notes


@register_ibs_method
def set_dbnotes(ibs, notes):
    """ sets notes for an entire database """
    import wbia

    assert isinstance(ibs, wbia.control.IBEISControl.IBEISController)
    ut.write_to(ibs.get_dbnotes_fpath(), notes)


@register_ibs_method
def annotstr(ibs, aid):
    return 'aid=%d' % aid


@register_ibs_method
def merge_names(ibs, merge_name, other_names):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        merge_name (str):
        other_names (list):

    CommandLine:
        python -m wbia.other.ibsfuncs --test-merge_names

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> merge_name = 'zebra'
        >>> other_names = ['occl', 'jeff']
        >>> result = merge_names(ibs, merge_name, other_names)
        >>> print(result)
        >>> ibs.print_names_table()
    """
    print(
        '[ibsfuncs] merging other_names=%r into merge_name=%r' % (other_names, merge_name)
    )
    other_nid_list = ibs.get_name_rowids_from_text(other_names)
    ibs.set_name_alias_texts(other_nid_list, [merge_name] * len(other_nid_list))
    other_aids_list = ibs.get_name_aids(other_nid_list)
    other_aids = ut.flatten(other_aids_list)
    print(
        '[ibsfuncs] ... %r annotations are being merged into merge_name=%r'
        % (len(other_aids), merge_name)
    )
    ibs.set_annot_names(other_aids, [merge_name] * len(other_aids))


def inspect_nonzero_yaws(ibs):
    """ python dev.py --dbdir /raid/work2/Turk/PZ_Master --cmd --show """
    from wbia.viz import viz_chip
    import wbia.plottool as pt

    aids = ibs.get_valid_aids()
    yaws = ibs.get_annot_yaws(aids)
    isnone_list = [yaw is not None for yaw in yaws]
    aids = ut.compress(aids, isnone_list)
    yaws = ut.compress(yaws, isnone_list)
    for aid, yaw in zip(aids, yaws):
        print(yaw)
        # We seem to be storing FULL paths in
        # the probchip table
        ibs.delete_annot_chips(aid)
        viz_chip.show_chip(ibs, aid, annote=False)
        pt.show_if_requested()


@register_ibs_method
@register_api('/api/annot/exemplar/', methods=['POST'])
def set_exemplars_from_quality_and_viewpoint(
    ibs,
    aid_list=None,
    exemplars_per_view=None,
    imgsetid=None,
    dry_run=False,
    verbose=True,
    prog_hook=None,
):
    """
    Automatic exemplar selection algorithm based on viewpoint and quality

    References:
        # implement maximum diversity approximation instead
        http://www.csbio.unc.edu/mcmillan/pubs/ICDM07_Pan.pdf

    CommandLine:
        python -m wbia.other.ibsfuncs --test-set_exemplars_from_quality_and_viewpoint
        python -m wbia.other.ibsfuncs --test-set_exemplars_from_quality_and_viewpoint:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> #ibs = wbia.opendb('PZ_MUGU_19')
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> dry_run = True
        >>> verbose = False
        >>> old_sum = sum(ibs.get_annot_exemplar_flags(ibs.get_valid_aids()))
        >>> new_flag_list = ibs.set_exemplars_from_quality_and_viewpoint(dry_run=dry_run)
        >>> new_sum = sum(new_flag_list)
        >>> print('old_sum = %r' % (old_sum,))
        >>> print('new_sum = %r' % (new_sum,))
        >>> zero_flag_list = ibs.set_exemplars_from_quality_and_viewpoint(exemplars_per_view=0, dry_run=dry_run)
        >>> assert sum(zero_flag_list) == 0
        >>> result = new_sum

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> dry_run = True
        >>> verbose = False
        >>> old_sum = sum(ibs.get_annot_exemplar_flags(ibs.get_valid_aids()))
        >>> new_flag_list = ibs.set_exemplars_from_quality_and_viewpoint(dry_run=dry_run)
        >>> # 2 of the 11 annots are unknown and should not be exemplars
        >>> ut.assert_eq(sum(new_flag_list), 9)

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb2')
        >>> dry_run = True
        >>> verbose = False
        >>> imgsetid = None
        >>> aid_list = ibs.get_valid_aids(imgsetid=imgsetid)
        >>> new_flag_list = ibs.set_exemplars_from_quality_and_viewpoint(aid_list, dry_run=dry_run)
        >>> old_flag_list = ibs.get_annot_exemplar_flags(aid_list)
        >>> new_exemplar_aids = ut.compress(aid_list, new_flag_list)
        >>> new_exemplar_qualtexts = ibs.get_annot_quality_texts(new_exemplar_aids)
        >>> assert 'junk' not in new_exemplar_qualtexts, 'should not have junk exemplars'
        >>> assert 'poor' not in new_exemplar_qualtexts, 'should not have poor exemplars'
        >>> #assert len(new_aid_list) == len(new_flag_list)
        >>> # 2 of the 11 annots are unknown and should not be exemplars
        >>> #ut.assert_eq(len(new_aid_list), 9)
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids(imgsetid=imgsetid)
    if exemplars_per_view is None:
        exemplars_per_view = 2
    new_flag_list = get_annot_quality_viewpoint_subset(
        ibs,
        aid_list=aid_list,
        annots_per_view=exemplars_per_view,
        verbose=verbose,
        prog_hook=prog_hook,
    )

    # Hack ensure each name has at least 1 exemplar
    # if False:
    #    nids = ibs.get_annot_nids(new_aid_list)
    #    uniquenids, groupxs = ut.group_indices(nids)
    #    num_hacked = 0
    #    grouped_exemplars = ut.apply_grouping(new_flag_list, groupxs)
    #    for exflags, idxs in zip(grouped_exemplars, groupxs):
    #        if not any(exflags):
    #            num_hacked += 1
    #            if len(idxs) > 0:
    #                new_flag_list[idxs[0]] = True
    #            if len(idxs) > 1:
    #                new_flag_list[idxs[1]] = True
    #    print('(exemplars) num_hacked = %r' % (num_hacked,))

    if not dry_run:
        ibs.set_annot_exemplar_flags(aid_list, new_flag_list)
    return new_flag_list


@register_ibs_method
def get_annot_quality_viewpoint_subset(
    ibs,
    aid_list=None,
    annots_per_view=2,
    max_annots=None,
    verbose=False,
    prog_hook=None,
    allow_unknown=False,
):
    """
    CommandLine:
        python -m wbia.other.ibsfuncs --exec-get_annot_quality_viewpoint_subset --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ut.exec_funckw(get_annot_quality_viewpoint_subset, globals())
        >>> ibs = wbia.opendb('testdb2')
        >>> new_flag_list = get_annot_quality_viewpoint_subset(ibs)
        >>> result = sum(new_flag_list)
        >>> print(result)
        38

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ut.exec_funckw(get_annot_quality_viewpoint_subset, globals())
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = [1]
        >>> new_flag_list = get_annot_quality_viewpoint_subset(ibs, aid_list, allow_unknown=True)
        >>> result = sum(new_flag_list)
        >>> print(result)
        1
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()

    INF = 999999  # effectively infinite

    qual2_value = {
        const.QUAL_EXCELLENT: 30,
        const.QUAL_GOOD: 20,
        const.QUAL_OK: 10,
        const.QUAL_UNKNOWN: 0,
        const.QUAL_POOR: -30,
        const.QUAL_JUNK: -INF,
    }

    # Value of previously being an exemplar
    oldexemp2_value = {
        True: 0,
        False: 1,
        None: 10,
    }

    # Value of not having multiple annotations
    ismulti2_value = {
        True: 0,
        False: 10,
        None: 10,
    }

    def get_chosen_flags(aids):
        # The weight of each annotation is 1. The value is based off its properties
        # We like good more than ok, and junk is infeasible We prefer items that
        # had previously been exemplars
        qual_value = np.array(ut.take(qual2_value, ibs.get_annot_quality_texts(aids)))
        oldexemp_value = np.array(
            ut.take(oldexemp2_value, ibs.get_annot_exemplar_flags(aids))
        )
        ismulti_value = np.array(ut.take(ismulti2_value, ibs.get_annot_multiple(aids)))
        base_value = 1
        values = qual_value + oldexemp_value + ismulti_value + base_value

        # Build input for knapsack
        weights = [1] * len(values)
        indices = list(range(len(weights)))
        values = np.round(values, 3).tolist()
        items = list(zip(values, weights, indices))

        # Greedy version is fine if all weights are 1, just pick the N maximum values
        total_value, chosen_items = ut.knapsack_greedy(items, maxweight=annots_per_view)
        # try:
        #    total_value, chosen_items = ut.knapsack(items, annots_per_view, method='recursive')
        # except Exception:
        #    print('WARNING: iterative method does not work correctly, but stack too big for recrusive')
        #    total_value, chosen_items = ut.knapsack(items, annots_per_view, method='iterative')

        chosen_indices = ut.get_list_column(chosen_items, 2)
        flags = [False] * len(aids)
        for index in chosen_indices:
            flags[index] = True
        return flags

    nid_list = np.array(ibs.get_annot_name_rowids(aid_list, distinguish_unknowns=True))
    unique_nids, groupxs_list = vt.group_indices(nid_list)
    grouped_aids = vt.apply_grouping(np.array(aid_list), groupxs_list)
    # aids = grouped_aids[-6]
    # for final settings because I'm too lazy to write
    new_aid_list = []
    new_flag_list = []
    _iter = ut.ProgIter(
        zip(grouped_aids, unique_nids),
        length=len(unique_nids),
        # freq=100,
        lbl='Picking best annots per viewpoint',
        prog_hook=prog_hook,
    )
    for aids_, nid in _iter:
        if not allow_unknown and ibs.is_nid_unknown(nid):
            # do not change unknown animals
            new_aid_list.extend(aids_)
            new_flag_list.extend([False] * len(aids_))
        else:
            # subgroup the names by viewpoints
            yawtexts = ibs.get_annot_viewpoints(aids_)
            yawtext2_aids = ut.group_items(aids_, yawtexts)
            for yawtext, aids in six.iteritems(yawtext2_aids):
                flags = get_chosen_flags(aids)
                new_aid_list.extend(aids)
                new_flag_list.extend(flags)

    aid2_idx = ut.make_index_lookup(aid_list)
    new_idxs = ut.take(aid2_idx, new_aid_list)

    # Re-order flags to agree with the input
    flag_list = ut.ungroup([new_flag_list], [new_idxs])
    assert ut.sortedby(new_flag_list, new_aid_list) == ut.sortedby(flag_list, aid_list)

    if verbose:
        print('Found %d exemplars for %d names' % (sum(flag_list), len(unique_nids)))
    return flag_list


def _split_car_contributor_tag(contributor_tag, distinguish_invalids=True):
    if contributor_tag is not None and 'NNP GZC Car' in contributor_tag:
        contributor_tag_split = contributor_tag.strip().split(',')
        if len(contributor_tag_split) == 2:
            contributor_tag = contributor_tag_split[0].strip()
    elif distinguish_invalids:
        contributor_tag = None
    return contributor_tag


@register_ibs_method
def report_sightings(ibs, complete=True, include_images=False, kaia=False, **kwargs):
    def sanitize_list(data_list):
        data_list = [str(data).replace(',', ':COMMA:') for data in list(data_list)]
        return_str = ','.join(data_list)
        return_str = return_str.replace(',None,', ',NONE,')
        return_str = return_str.replace(',%s,' % (const.UNKNOWN,), ',UNKNOWN,')
        return_str = return_str.replace(',-1,', ',UNKNOWN,')
        return_str = return_str.replace(',-1,', ',UNKNOWN,')
        return_str = return_str.replace(',-1.0,', ',UNKNOWN,')
        return_str = return_str.replace(',-1.0,', ',UNKNOWN,')
        return return_str

    def construct():
        if complete:
            cols_list = [
                ('annotation_id', aid_list),
                ('annotation_xtl', xtl_list),
                ('annotation_ytl', ytl_list),
                ('annotation_width', width_list),
                ('annotation_height', height_list),
                ('annotation_species', species_list),
                ('annotation_viewpoint', viewpoint_list),
                ('annotation_qualities', quality_list),
                ('annotation_sex', sex_list),
                ('annotation_age_min', age_min_list),
                ('annotation_age_max', age_max_list),
                ('annotation_age', age_list),
                ('annotation_comment', comment_list),
                ('annotation_name', name_list),
                ('image_id', gid_list),
                ('image_contributor', contributor_list),
                ('image_car', car_list),
                ('image_filename', uri_list),
                ('image_unixtime', unixtime_list),
                ('image_time_str', time_list),
                ('image_date_str', date_list),
                ('image_lat', lat_list),
                ('image_lon', lon_list),
                ('flag_first_seen', seen_list),
                ('flag_marked', marked_list),
            ]
        else:
            cols_list = [
                ('annotation_id', aid_list),
                ('image_time_str', time_list),
                ('image_date_str', date_list),
                ('flag_first_seen', seen_list),
                ('image_lat', lat_list),
                ('image_lon', lon_list),
                ('image_car', car_list),
                ('annotation_age_min', age_min_list),
                ('annotation_age_max', age_max_list),
                ('annotation_sex', sex_list),
            ]
        header_list = [sanitize_list([cols[0] for cols in cols_list])]
        data_list = zip(*[cols[1] for cols in cols_list])
        line_list = [sanitize_list(data) for data in list(data_list)]
        return header_list, line_list

    # Grab primitives
    if complete:
        aid_list = ibs.get_valid_aids()
    else:
        aid_list = ibs.filter_aids_count(pre_unixtime_sort=False)

    gid_list = ibs.get_annot_gids(aid_list)
    bbox_list = ibs.get_annot_bboxes(aid_list)
    xtl_list = [bbox[0] for bbox in bbox_list]
    ytl_list = [bbox[1] for bbox in bbox_list]
    width_list = [bbox[2] for bbox in bbox_list]
    height_list = [bbox[3] for bbox in bbox_list]
    species_list = ibs.get_annot_species_texts(aid_list)
    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    # quality_list   = ibs.get_annot_quality_texts(aid_list)
    quality_list = ibs.get_annot_qualities(aid_list)
    metadata_list = ibs.get_annot_metadata(aid_list)
    comment_list = [
        metadata.get('turk', {}).get('match', {}).get('comment', '')
        for metadata in metadata_list
    ]
    contributor_list = ibs.get_image_contributor_tag(gid_list)
    car_list = [
        _split_car_contributor_tag(contributor_tag)
        for contributor_tag in contributor_list
    ]
    uri_list = ibs.get_image_uris(gid_list)
    sex_list = ibs.get_annot_sex_texts(aid_list)
    age_min_list = ibs.get_annot_age_months_est_min(aid_list)
    age_max_list = ibs.get_annot_age_months_est_max(aid_list)
    age_list = []
    for age_min, age_max in zip(age_min_list, age_max_list):
        if age_min is None and age_max == 2:
            age = '0-3 Months'
        elif age_min == 3 and age_max == 5:
            age = '3-6 Months'
        elif age_min == 6 and age_max == 11:
            age = '6-12 Months'
        elif age_min == 12 and age_max == 23:
            age = 'Yearling'
        elif age_min == 24 and age_max == 35:
            age = '2-Year-Old'
        elif age_min == 36 and age_max is None:
            age = 'Adult'
        elif age_min is None and age_max is None:
            age = 'Unknown'
        else:
            age = 'Unknown'
        age_list.append(age)

    name_list = ibs.get_annot_names(aid_list)
    unixtime_list = ibs.get_image_unixtime(gid_list)
    datetime_list = [
        ut.unixtime_to_datetimestr(unixtime) if unixtime is not None else 'UNKNOWN'
        for unixtime in unixtime_list
    ]
    datetime_split_list = [datetime.split(' ') for datetime in datetime_list]
    date_list = [
        datetime_split[0] if len(datetime_split) == 2 else 'UNKNOWN'
        for datetime_split in datetime_split_list
    ]
    time_list = [
        datetime_split[1] if len(datetime_split) == 2 else 'UNKNOWN'
        for datetime_split in datetime_split_list
    ]
    lat_list = ibs.get_image_lat(gid_list)
    lon_list = ibs.get_image_lon(gid_list)
    marked_list = ibs.flag_aids_count(aid_list)
    seen_list = []
    seen_set = set()
    for name in name_list:
        if name is not None and name != const.UNKNOWN and name not in seen_set:
            seen_list.append(True)
            seen_set.add(name)
            continue
        seen_list.append(False)

    return_list, line_list = construct()
    return_list.extend(line_list)

    if include_images:
        all_gid_set = set(ibs.get_valid_gids())
        gid_set = set(gid_list)
        missing_gid_list = sorted(list(all_gid_set - gid_set))
        filler = [''] * len(missing_gid_list)

        aid_list = filler
        species_list = filler
        viewpoint_list = filler
        quality_list = filler
        comment_list = filler
        sex_list = filler
        age_min_list = filler
        age_max_list = filler
        age_list = filler
        name_list = filler
        gid_list = missing_gid_list
        contributor_list = ibs.get_image_contributor_tag(missing_gid_list)
        car_list = [
            _split_car_contributor_tag(contributor_tag)
            for contributor_tag in contributor_list
        ]
        uri_list = ibs.get_image_uris(missing_gid_list)
        unixtime_list = ibs.get_image_unixtime(missing_gid_list)
        datetime_list = [
            ut.unixtime_to_datetimestr(unixtime) if unixtime is not None else 'UNKNOWN'
            for unixtime in unixtime_list
        ]
        datetime_split_list = [datetime.split(' ') for datetime in datetime_list]
        date_list = [
            datetime_split[0] if len(datetime_split) == 2 else 'UNKNOWN'
            for datetime_split in datetime_split_list
        ]
        time_list = [
            datetime_split[1] if len(datetime_split) == 2 else 'UNKNOWN'
            for datetime_split in datetime_split_list
        ]
        lat_list = ibs.get_image_lat(missing_gid_list)
        lon_list = ibs.get_image_lon(missing_gid_list)
        seen_list = filler
        marked_list = filler

        (
            header_list,
            line_list,
        ) = construct()  # NOTE: discard the header list returned here
        return_list.extend(line_list)

    return return_list


@register_ibs_method
def report_sightings_str(ibs, **kwargs):
    line_list = ibs.report_sightings(**kwargs)
    return '\n'.join(line_list)


@register_ibs_method
def check_chip_existence(ibs, aid_list=None):
    aid_list = ibs.get_valid_aids()
    chip_fpath_list = ibs.get_annot_chip_fpath(aid_list)
    flag_list = [
        True if chip_fpath is None else exists(chip_fpath)
        for chip_fpath in chip_fpath_list
    ]
    aid_kill_list = ut.filterfalse_items(aid_list, flag_list)
    if len(aid_kill_list) > 0:
        print('found %d inconsistent chips attempting to fix' % len(aid_kill_list))
    ibs.delete_annot_chips(aid_kill_list)


@register_ibs_method
def get_quality_filterflags(ibs, aid_list, minqual, unknown_ok=True):
    r"""
    DEPRICATE

    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (int):  list of annotation ids
        minqual (str): qualtext
        unknown_ok (bool): (default = False)

    Returns:
        iter: qual_flags

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-get_quality_filterflags

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:20]
        >>> minqual = 'junk'
        >>> unknown_ok = False
        >>> qual_flags = list(get_quality_filterflags(ibs, aid_list, minqual, unknown_ok))
        >>> result = ('qual_flags = %s' % (str(qual_flags),))
        >>> print(result)
    """
    minqual_int = const.QUALITY_TEXT_TO_INT[minqual]
    qual_int_list = ibs.get_annot_qualities(aid_list)
    # print('qual_int_list = %r' % (qual_int_list,))
    if unknown_ok:
        qual_flags = (
            (qual_int is None or qual_int == -1) or qual_int >= minqual_int
            for qual_int in qual_int_list
        )
    else:
        qual_flags = (
            (qual_int is not None) and qual_int >= minqual_int
            for qual_int in qual_int_list
        )
    qual_flags = list(qual_flags)
    return qual_flags


@register_ibs_method
@profile
def get_viewpoint_filterflags(
    ibs, aid_list, valid_yaws, unknown_ok=True, assume_unique=False
):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (int):  list of annotation ids
        valid_yaws (?):
        unknown_ok (bool): (default = True)

    Returns:
        int: aid_list -  list of annotation ids

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-get_viewpoint_filterflags
        python -m wbia.other.ibsfuncs --exec-get_viewpoint_filterflags --db NNP_Master3

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='Spotted_Dolfin_Master')
        >>> aid_list = ibs.get_valid_aids()[0:20]
        >>> valid_yaws = ['left']
        >>> unknown_ok = False
        >>> yaw_flags = list(get_viewpoint_filterflags(ibs, aid_list, valid_yaws, unknown_ok))
        >>> result = ('yaw_flags = %s' % (str(yaw_flags),))
        >>> print(result)
    """
    assert valid_yaws is None or isinstance(
        valid_yaws, (set, list, tuple)
    ), 'valid_yaws is not a container'
    yaw_list = ibs.get_annot_viewpoints(aid_list, assume_unique=assume_unique)
    if unknown_ok:
        yaw_flags = (
            yaw is None or (valid_yaws is None or yaw in valid_yaws) for yaw in yaw_list
        )
    else:
        yaw_flags = (
            yaw is not None and (valid_yaws is None or yaw in valid_yaws)
            for yaw in yaw_list
        )
    yaw_flags = list(yaw_flags)
    return yaw_flags


@register_ibs_method
def get_quality_viewpoint_filterflags(ibs, aid_list, minqual, valid_yaws):
    qual_flags = get_quality_filterflags(ibs, aid_list, minqual)
    yaw_flags = get_viewpoint_filterflags(ibs, aid_list, valid_yaws)
    # qual_list = ibs.get_annot_qualities(aid_list)
    # yaw_list = ibs.get_annot_viewpoints(aid_list)
    # qual_flags = (qual is None or qual > minqual for qual in qual_list)
    # yaw_flags  = (yaw is None or yaw in valid_yaws for yaw in yaw_list)
    flags_list = list(ut.and_iters(qual_flags, yaw_flags))
    return flags_list


@register_ibs_method
def flag_aids_count(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (int):  list of annotation ids
        pre_unixtime_sort (bool):

    Returns:
        list:

    CommandLine:
        python -m wbia.other.ibsfuncs --test-flag_aids_count

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> gzc_flag_list = flag_aids_count(ibs, aid_list)
        >>> result = gzc_flag_list
        >>> print(result)
        [False, True, False, False, True, False, True, True, False, True, False, True, True]

    """
    # Get primitives
    unixtime_list = ibs.get_annot_image_unixtimes(aid_list)
    index_list = ut.list_argsort(unixtime_list)
    aid_list = ut.sortedby(aid_list, unixtime_list)
    gid_list = ibs.get_annot_gids(aid_list)
    nid_list = ibs.get_annot_name_rowids(aid_list)
    contributor_list = ibs.get_image_contributor_tag(gid_list)
    # Get filter flags for aids
    isunknown_list = ibs.is_aid_unknown(aid_list)
    flag_list = [not unknown for unknown in isunknown_list]
    # Filter by seen and car
    flag_list_ = []
    seen_dict = ut.ddict(set)
    # Mark the first annotation (for each name) seen per car
    values_list = zip(aid_list, gid_list, nid_list, flag_list, contributor_list)
    for aid, gid, nid, flag, contrib in values_list:
        if flag:
            contributor_ = _split_car_contributor_tag(contrib, distinguish_invalids=False)
            if nid not in seen_dict[contributor_]:
                seen_dict[contributor_].add(nid)
                flag_list_.append(True)
                continue
        flag_list_.append(False)
    # Take the inverse of the sorted
    gzc_flag_list = ut.list_inverse_take(flag_list_, index_list)
    return gzc_flag_list


@register_ibs_method
def filter_aids_count(ibs, aid_list=None, pre_unixtime_sort=True):
    if aid_list is None:
        # Get all aids and pre-sort by unixtime
        aid_list = ibs.get_valid_aids()
        if pre_unixtime_sort:
            unixtime_list = ibs.get_image_unixtime(ibs.get_annot_gids(aid_list))
            aid_list = ut.sortedby(aid_list, unixtime_list)
    flags_list = ibs.flag_aids_count(aid_list)
    aid_list_ = list(ut.iter_compress(aid_list, flags_list))
    return aid_list_


@register_ibs_method
@profile
def get_unflat_annots_kmdists_list(ibs, aids_list):
    # ibs.check_name_mapping_consistency(aids_list)
    latlons_list = ibs.unflat_map(ibs.get_annot_image_gps, aids_list)
    latlon_arrs = [np.array(latlons) for latlons in latlons_list]
    for arrs in latlon_arrs:
        # our database encodes -1 as invalid.
        # Silly, but its in the middle of the atlantic ocean
        arrs[arrs == -1] = np.nan
    km_dists_list = [
        ut.safe_pdist(latlon_arr, metric=vt.haversine) for latlon_arr in latlon_arrs
    ]
    return km_dists_list


@register_ibs_method
@profile
def get_unflat_annots_hourdists_list(ibs, aids_list):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> ibs = testdata_ibs('testdb1')
        >>> nid_list = get_valid_multiton_nids_custom(ibs)
        >>> aids_list_ = ibs.get_name_aids(nid_list)
        >>> aids_list = [(aids) for aids in aids_list_]
        >>> ibs.get_unflat_annots_hourdists_list(aids_list)
    """
    # assert all(list(map(ut.isunique, aids_list)))
    unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, aids_list)
    # assert all(list(map(ut.isunique, unixtimes_list)))
    unixtime_arrs = [np.array(unixtimes)[:, None] for unixtimes in unixtimes_list]
    hour_dists_list = [
        ut.safe_pdist(unixtime_arr, metric=ut.unixtime_hourdiff)
        for unixtime_arr in unixtime_arrs
    ]
    return hour_dists_list


@register_ibs_method
@profile
def get_unflat_annots_timedelta_list(ibs, aids_list):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> ibs = testdata_ibs('NNP_Master3')
        >>> nid_list = get_valid_multiton_nids_custom(ibs)
        >>> aids_list_ = ibs.get_name_aids(nid_list)
        >>> aids_list = [(aids) for aids in aids_list_]

    """
    # assert all(list(map(ut.isunique, aids_list)))
    unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, aids_list)
    # assert all(list(map(ut.isunique, unixtimes_list)))
    unixtime_arrs = [np.array(unixtimes)[:, None] for unixtimes in unixtimes_list]
    timedelta_list = [
        ut.safe_pdist(unixtime_arr, metric=ut.absdiff) for unixtime_arr in unixtime_arrs
    ]
    return timedelta_list


@register_ibs_method
@profile
def get_unflat_annots_speeds_list2(ibs, aids_list):
    """
    much faster than original version

    _ = ibs.get_unflat_annots_speeds_list2(aids_list)

    %timeit ibs.get_unflat_annots_speeds_list(aids_list)
    3.44 s per loop

    %timeit ibs.get_unflat_annots_speeds_list2(aids_list)
    665 ms per loop

    %timeit ibs.get_unflat_annots_speeds_list(aids_list[0:1])
    12.8 ms
    %timeit ibs.get_unflat_annots_speeds_list2(aids_list[0:1])
    6.51 ms

    assert ibs.get_unflat_annots_speeds_list([]) == ibs.get_unflat_annots_speeds_list2([])

    ibs.get_unflat_annots_speeds_list([[]])
    ibs.get_unflat_annots_speeds_list2([[]])

    """
    if True:
        unique_aids = sorted(list(set(ut.flatten(aids_list))))
        aid_pairs_list = [list(it.combinations(aids, 2)) for aids in aids_list]
        aid_pairs, cumsum = ut.invertible_flatten2(aid_pairs_list)
        speeds = ibs.get_annotpair_speeds(aid_pairs, unique_aids=unique_aids)
        speeds_list = ut.unflatten2(speeds, cumsum)
    else:
        # Use indexing for lookup efficiency
        unique_aids = sorted(list(set(ut.flatten(aids_list))))
        aid_to_idx = ut.make_index_lookup(unique_aids)
        idx_list = [ut.take(aid_to_idx, aids) for aids in aids_list]
        # Lookup values in SQL only once
        unique_unixtimes = ibs.get_annot_image_unixtimes_asfloat(unique_aids)
        unique_gps = ibs.get_annot_image_gps(unique_aids)
        unique_gps = np.array(
            [
                (np.nan if lat == -1 else lat, np.nan if lon == -1 else lon)
                for (lat, lon) in unique_gps
            ]
        )
        unique_gps = vt.atleast_nd(unique_gps, 2)
        if len(unique_gps) == 0:
            unique_gps.shape = (0, 2)
        # Find pairs that need comparison
        idx_pairs_list = [list(it.combinations(idxs, 2)) for idxs in idx_list]
        idx_pairs, cumsum = ut.invertible_flatten2(idx_pairs_list)
        idxs1 = ut.take_column(idx_pairs, 0)
        idxs2 = ut.take_column(idx_pairs, 1)
        # Find differences in time and space
        hour_dists = ut.unixtime_hourdiff(
            unique_unixtimes[idxs1], unique_unixtimes[idxs2]
        )
        km_dists = vt.haversine(unique_gps[idxs1].T, unique_gps[idxs2].T)
        # Zero km-distances less than a small epsilon if the timediff is zero to
        # prevent infinity problems
        idxs = np.where(hour_dists == 0)[0]
        under_eps = km_dists[idxs] < 0.5
        km_dists[idxs[under_eps]] = 0
        # Normal speed calculation
        speeds = km_dists / hour_dists
        # No movement over no time
        flags = np.logical_and(km_dists == 0, hour_dists == 0)
        speeds[flags] = 0
        speeds_list = ut.unflatten2(speeds, cumsum)
    return speeds_list


@register_ibs_method
@profile
def get_annotpair_speeds(ibs, aid_pairs, unique_aids=None):
    aids1 = ut.take_column(aid_pairs, 0)
    aids2 = ut.take_column(aid_pairs, 1)
    if unique_aids is None:
        unique_aids = sorted(list(set(ut.flatten([aids1, aids2]))))
    # Use indexing for lookup efficiency
    aid_to_idx = ut.make_index_lookup(unique_aids)
    idxs1 = ut.take(aid_to_idx, aids1)
    idxs2 = ut.take(aid_to_idx, aids2)
    # idx_list = [ut.take(aid_to_idx, aids) for aids in aids_list]
    # Lookup values in SQL only once
    unique_unixtimes = ibs.get_annot_image_unixtimes_asfloat(unique_aids)
    unique_gps = ibs.get_annot_image_gps(unique_aids)
    unique_gps = np.array(
        [
            (np.nan if lat == -1 else lat, np.nan if lon == -1 else lon)
            for (lat, lon) in unique_gps
        ]
    )
    unique_gps = vt.atleast_nd(unique_gps, 2)
    if len(unique_gps) == 0:
        unique_gps.shape = (0, 2)
    # Find differences in time and space
    hour_dists = ut.unixtime_hourdiff(unique_unixtimes[idxs1], unique_unixtimes[idxs2])
    km_dists = vt.haversine(unique_gps[idxs1].T, unique_gps[idxs2].T)
    # Zero km-distances less than a small epsilon if the timediff is zero to
    # prevent infinity problems
    idxs = np.where(hour_dists == 0)[0]
    under_eps = km_dists[idxs] < 0.5
    km_dists[idxs[under_eps]] = 0
    # Normal speed calculation
    speeds = km_dists / hour_dists
    # No movement over no time
    flags = np.logical_and(km_dists == 0, hour_dists == 0)
    speeds[flags] = 0
    return speeds


@register_ibs_method
@profile
def get_unflat_am_rowids(ibs, aids_list):
    aid_pairs = [list(it.combinations(aids, 2)) for aids in aids_list]
    flat_pairs, cumsum = ut.invertible_flatten2(aid_pairs)
    flat_aids1 = ut.take_column(flat_pairs, 0)
    flat_aids2 = ut.take_column(flat_pairs, 1)
    flat_ams_ = ibs.get_annotmatch_rowid_from_undirected_superkey(flat_aids1, flat_aids2)
    ams_ = ut.unflatten2(flat_ams_, cumsum)
    ams_list = [ut.filter_Nones(a) for a in ams_]
    return ams_list
    # flat_ams = ut.filter_Nones(ams)


@register_ibs_method
@profile
def get_unflat_am_aidpairs(ibs, aids_list):
    """ Gets only aid pairs that have some reviewed/matched status """
    ams_list = ibs.get_unflat_am_rowids(aids_list)
    flat_ams, cumsum = ut.invertible_flatten2(ams_list)
    flat_aids1 = ibs.get_annotmatch_aid1(flat_ams)
    flat_aids2 = ibs.get_annotmatch_aid2(flat_ams)
    flat_aid_pairs = list(zip(flat_aids1, flat_aids2))
    aid_pairs = ut.unflatten2(flat_aid_pairs, cumsum)
    return aid_pairs


@register_ibs_method
@profile
def get_unflat_case_tags(ibs, aids_list):
    """ Gets only aid pairs that have some reviewed/matched status """
    ams_list = ibs.get_unflat_am_rowids(aids_list)
    tags = ibs.unflat_map(ibs.get_annotmatch_case_tags, ams_list)
    return tags


@register_ibs_method
@profile
def get_unflat_annots_speeds_list(ibs, aids_list):
    """ DEPRICATE. SLOWER """
    km_dists_list = ibs.get_unflat_annots_kmdists_list(aids_list)
    hour_dists_list = ibs.get_unflat_annots_hourdists_list(aids_list)

    # hour_dists_list = ut.replace_nones(hour_dists_list, [])
    # km_dists_list = ut.replace_nones(km_dists_list, [])
    # flat_hours, cumsum1 = np.array(ut.invertible_flatten2(hour_dists_list))
    # flat_hours = np.array(flat_hours)
    # flat_kms = np.array(ut.flatten(km_dists_list))

    def compute_speed(km_dists, hours_dists):
        if km_dists is None or hours_dists is None:
            return None
        # Zero km-distances less than a small epsilon if the timediff is zero to
        # prevent infinity problems
        idxs = np.where(hours_dists == 0)[0]
        under_eps = km_dists[idxs] < 0.5
        km_dists[idxs[under_eps]] = 0
        # Normal speed calculation
        speeds = km_dists / hours_dists
        # No movement over no time
        flags = np.logical_and(km_dists == 0, hours_dists == 0)
        speeds[flags] = 0
        return speeds

    speeds_list = [
        compute_speed(km_dists, hours_dists)
        # vt.safe_div(km_dists, hours_dists)
        for km_dists, hours_dists in zip(km_dists_list, hour_dists_list)
    ]
    return speeds_list


def testdata_ibs(defaultdb='testdb1'):
    import wbia

    ibs = wbia.opendb(defaultdb=defaultdb)
    return ibs


def get_valid_multiton_nids_custom(ibs):
    nid_list_ = ibs._get_all_known_nids()
    ismultiton_list = [len((aids)) > 1 for aids in ibs.get_name_aids(nid_list_)]
    nid_list = ut.compress(nid_list_, ismultiton_list)
    return nid_list


@register_ibs_method
def make_next_imageset_text(ibs):
    """
    Creates what the next imageset name would be but does not add it to the database

    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.ibsfuncs --test-make_next_imageset_text

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> new_imagesettext = make_next_imageset_text(ibs)
        >>> result = new_imagesettext
        >>> print(result)
        New ImageSet 0
    """
    imgsetid_list = ibs.get_valid_imgsetids()
    old_imagesettext_list = ibs.get_imageset_text(imgsetid_list)
    new_imagesettext = ut.get_nonconflicting_string(
        'New ImageSet %d', old_imagesettext_list
    )
    return new_imagesettext


@register_ibs_method
def add_next_imageset(ibs):
    """ Adds a new imageset to the database """
    new_imagesettext = ibs.make_next_imageset_text()
    (new_imgsetid,) = ibs.add_imagesets([new_imagesettext])
    return new_imgsetid


@register_ibs_method
def create_new_imageset_from_images(ibs, gid_list, new_imgsetid=None):
    r"""
    Args:
        gid_list (list):

    CommandLine:
        python -m wbia.other.ibsfuncs --test-create_new_imageset_from_images

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[::2]
        >>> new_imgsetid = create_new_imageset_from_images(ibs, gid_list)
        >>> result = new_imgsetid
        >>> print(result)
    """
    if new_imgsetid is None:
        new_imgsetid = ibs.add_next_imageset()
    imgsetid_list = [new_imgsetid] * len(gid_list)
    ibs.set_image_imgsetids(gid_list, imgsetid_list)
    return new_imgsetid


@register_ibs_method
def new_imagesets_from_images(ibs, gids_list):
    r"""
    Args:
        gids_list (list):
    """
    imgsetid_list = [ibs.create_new_imageset_from_images(gids) for gids in gids_list]
    return imgsetid_list


@register_ibs_method
def create_new_imageset_from_names(ibs, nid_list):
    r"""
    Args:
        nid_list (list):

    CommandLine:
        python -m wbia.other.ibsfuncs --test-create_new_imageset_from_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_nids()[0:2]
        >>> new_imgsetid = ibs.create_new_imageset_from_names(nid_list)
        >>> # clean up
        >>> ibs.delete_imagesets(new_imgsetid)
        >>> result = new_imgsetid
        >>> print(result)
    """
    aids_list = ibs.get_name_aids(nid_list)
    gids_list = ibs.unflat_map(ibs.get_annot_gids, aids_list)
    gid_list = ut.flatten(gids_list)
    new_imgsetid = ibs.create_new_imageset_from_images(gid_list)
    return new_imgsetid


@register_ibs_method
def prepare_annotgroup_review(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (int):  list of annotation ids

    Returns:
        tuple: (src_ag_rowid, dst_ag_rowid) - source and dest annot groups

    CommandLine:
        python -m wbia.other.ibsfuncs --test-prepare_annotgroup_review

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> result = prepare_annotgroup_review(ibs, aid_list)
        >>> print(result)
    """
    # Build new names for source and dest annot groups
    all_annotgroup_rowid_list = ibs._get_all_annotgroup_rowids()
    all_annotgroup_text_list = ibs.get_annotgroup_text(all_annotgroup_rowid_list)
    new_grouptext_src = ut.get_nonconflicting_string(
        'Source Group %d', all_annotgroup_text_list
    )
    all_annotgroup_text_list += [new_grouptext_src]
    new_grouptext_dst = ut.get_nonconflicting_string(
        'Dest Group %d', all_annotgroup_text_list
    )
    # Add new empty groups
    annotgroup_text_list = [new_grouptext_src, new_grouptext_dst]
    annotgroup_uuid_list = list(map(ut.hashable_to_uuid, annotgroup_text_list))
    annotgroup_note_list = ['', '']
    src_ag_rowid, dst_ag_rowid = ibs.add_annotgroup(
        annotgroup_uuid_list, annotgroup_text_list, annotgroup_note_list
    )
    # Relate the annotations with the source group
    ibs.add_gar([src_ag_rowid] * len(aid_list), aid_list)
    return src_ag_rowid, dst_ag_rowid


@register_ibs_method
def remove_groundtrue_aids(ibs, aid_list, ref_aid_list):
    """ removes any aids that are known to match """
    ref_nids = set(ibs.get_annot_name_rowids(ref_aid_list))
    nid_list = ibs.get_annot_name_rowids(aid_list)
    flag_list = [nid not in ref_nids for nid in nid_list]
    aid_list_ = ut.compress(aid_list, flag_list)
    return aid_list_


@register_ibs_method
def search_annot_notes(ibs, pattern, aid_list=None):
    """

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_Master0')
        >>> pattern = ['gash', 'injury', 'scar', 'wound']
        >>> valid_aid_list = ibs.search_annot_notes(pattern)
        >>> print(valid_aid_list)
        >>> print(ibs.get_annot_notes(valid_aid_list))
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    notes_list = ibs.get_annot_notes(aid_list)
    # convert a list of patterns into an or statement
    if isinstance(pattern, (list, tuple)):
        pattern = '|'.join(['(%s)' % pat for pat in pattern])
    valid_index_list, valid_match_list = ut.search_list(
        notes_list, pattern, flags=re.IGNORECASE
    )
    # [match.group() for match in valid_match_list]
    valid_aid_list = ut.take(aid_list, valid_index_list)
    return valid_aid_list


@register_ibs_method
def filter_aids_to_quality(ibs, aid_list, minqual, unknown_ok=True, speedhack=True):
    """
    DEPRICATE

        >>> import wbia
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> ibs = wbia.opendb(defaultdb='PZ_Master1')
        >>> aid_list = ibs.get_valid_aids()
        >>> minqual = 'good'
        >>> x1 = filter_aids_to_quality(ibs, aid_list, 'good', True, speedhack=True)
        >>> x2 = filter_aids_to_quality(ibs, aid_list, 'good', True, speedhack=False)
    """
    if speedhack:
        list_repr = ','.join(map(str, aid_list))
        minqual_int = const.QUALITY_TEXT_TO_INT[minqual]
        if unknown_ok:
            operation = 'SELECT rowid from annotations WHERE (annot_quality ISNULL OR annot_quality==-1 OR annot_quality>={minqual_int}) AND rowid IN ({aids})'
        else:
            operation = 'SELECT rowid from annotations WHERE annot_quality NOTNULL AND annot_quality>={minqual_int} AND rowid IN ({aids})'
        operation = operation.format(aids=list_repr, minqual_int=minqual_int)
        aid_list_ = ut.take_column(ibs.db.cur.execute(operation).fetchall(), 0)
    else:
        qual_flags = list(
            ibs.get_quality_filterflags(aid_list, minqual, unknown_ok=unknown_ok)
        )
        aid_list_ = ut.compress(aid_list, qual_flags)
    return aid_list_


@register_ibs_method
def filter_aids_to_viewpoint(ibs, aid_list, valid_yaws, unknown_ok=True):
    """
    Removes aids that do not have a valid yaw

    TODO: rename to valid_viewpoint because this func uses category labels

    valid_yaws = ['primary', 'primary1', 'primary-1']
    """

    def rectify_view_category(view):
        @ut.memoize
        def _primary_species():
            return ibs.get_primary_database_species()

        if view == 'primary':
            view = get_primary_species_viewpoint(_primary_species())
        if view == 'primary1':
            view = get_primary_species_viewpoint(_primary_species(), 1)
        if view == 'primary-1':
            view = get_primary_species_viewpoint(_primary_species(), -1)
        return view

    valid_yaws = [rectify_view_category(view) for view in valid_yaws]

    yaw_flags = list(
        ibs.get_viewpoint_filterflags(aid_list, valid_yaws, unknown_ok=unknown_ok)
    )
    aid_list_ = ut.compress(aid_list, yaw_flags)
    return aid_list_


@register_ibs_method
def remove_aids_of_viewpoint(ibs, aid_list, invalid_yaws):
    """
    Removes aids that do not have a valid yaw

    TODO; rename to valid_viewpoint because this func uses category labels
    """
    notyaw_flags = list(
        ibs.get_viewpoint_filterflags(aid_list, invalid_yaws, unknown_ok=False)
    )
    yaw_flags = ut.not_list(notyaw_flags)
    aid_list_ = ut.compress(aid_list, yaw_flags)
    return aid_list_


@register_ibs_method
def filter_aids_without_name(ibs, aid_list, invert=False, speedhack=True):
    r"""
    Remove aids without names

    Example:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> annots = ibs.annots(aid_list)
        >>> aid_list1_ = ibs.filter_aids_without_name(aid_list)
        >>> aid_list2_ = ibs.filter_aids_without_name(aid_list, invert=True)
        >>> annots1_ = ibs.annots(aid_list1_)
        >>> annots2_ = ibs.annots(aid_list2_)
        >>> assert len(annots1_) + len(annots2_) == len(annots)
        >>> assert np.all(np.array(annots1_.nids) > 0)
        >>> assert len(annots1_) == 9
        >>> assert np.all(np.array(annots2_.nids) < 0)
        >>> assert len(annots2_) == 4
    """
    if speedhack:
        list_repr = ','.join(map(str, aid_list))
        if invert:
            operation = (
                'SELECT rowid from annotations WHERE name_rowid<=0 AND rowid IN (%s)'
                % (list_repr,)
            )
        else:
            operation = (
                'SELECT rowid from annotations WHERE name_rowid>0 AND rowid IN (%s)'
                % (list_repr,)
            )
        aid_list_ = ut.take_column(ibs.db.cur.execute(operation).fetchall(), 0)
    else:
        flag_list = ibs.is_aid_unknown(aid_list)
        if not invert:
            flag_list = ut.not_list(flag_list)
        aid_list_ = ut.compress(aid_list, flag_list)
    return aid_list_


@register_ibs_method
def filter_annots_using_minimum_timedelta(ibs, aid_list, min_timedelta):
    r"""
    Uses a dynamic program to find the maximum number of annotations that are
    above the minimum timedelta requirement.

    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (?):
        min_timedelta (?):

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-filter_annots_using_minimum_timedelta
        python -m wbia.other.ibsfuncs --exec-filter_annots_using_minimum_timedelta --db PZ_Master1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> aid_list = ibs.filter_aids_without_timestamps(aid_list)
        >>> print('Before')
        >>> ibs.print_annot_stats(aid_list, min_name_hourdist=True)
        >>> min_timedelta = 60 * 60 * 24
        >>> filtered_aids = filter_annots_using_minimum_timedelta(ibs, aid_list, min_timedelta)
        >>> print('After')
        >>> ibs.print_annot_stats(filtered_aids, min_name_hourdist=True)
        >>> ut.quit_if_noshow()
        >>> wbia.other.dbinfo.hackshow_names(ibs, aid_list)
        >>> wbia.other.dbinfo.hackshow_names(ibs, filtered_aids)
        >>> ut.show_if_requested()
    """
    import vtool as vt

    # min_timedelta = 60 * 60 * 24
    # min_timedelta = 60 * 10
    grouped_aids = ibs.group_annots_by_name(aid_list)[0]
    unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, grouped_aids)
    # Find the maximum size subset such that all timedeltas are less than a given value
    r"""
    Given a set of annotations V (all of the same name).
    Let $E = V \times V$ be the the set of all pairs of annotations.

    We will now indicate which annotations are included as to separate them by
    a minimum timedelta while maximizing the number of annotations taken.

    Let t[u, v] be the absolute difference in time deltas between u and v

    Let x[u, v] = 1 if the annotation pair (u, v) is included.

    Let y[u] = 1 if the annotation u is included.

    maximize sum(y[u] for u in V)
    subject to:

        # Annotations pairs are only included if their timedelta is less than
        # the threshold.
        x[u, v] = 0 if t[u, v] > thresh

        # If a pair is excluded than at least one annotation in that pair must
        # be excluded.
        y[u] + y[v] - x[u, v] < 2
    """
    chosen_idxs_list = [
        ut.maximin_distance_subset1d(unixtimes, min_thresh=min_timedelta)[0]
        for unixtimes in unixtimes_list
    ]
    filtered_groups = vt.ziptake(grouped_aids, chosen_idxs_list)
    filtered_aids = ut.flatten(filtered_groups)
    if ut.DEBUG2:
        timedeltas = ibs.get_unflat_annots_timedelta_list(filtered_groups)
        min_timedeltas = np.array(
            [np.nan if dists is None else np.nanmin(dists) for dists in timedeltas]
        )
        min_name_timedelta_stats = ut.get_stats(min_timedeltas, use_nan=True)
        print('min_name_timedelta_stats = %s' % (ut.repr2(min_name_timedelta_stats),))
    return filtered_aids


@register_ibs_method
def filter_aids_without_timestamps(ibs, aid_list, invert=False):
    """
    Removes aids without timestamps
    aid_list = ibs.get_valid_aids()
    """
    unixtime_list = ibs.get_annot_image_unixtimes(aid_list)
    flag_list = [unixtime != -1 for unixtime in unixtime_list]
    if invert:
        flag_list = ut.not_list(flag_list)
    aid_list_ = ut.compress(aid_list, flag_list)
    return aid_list_


@register_ibs_method
def filter_aids_to_species(ibs, aid_list, species, speedhack=True):
    """
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (int):  list of annotation ids
        species (?):

    Returns:
        list: aid_list_

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-filter_aids_to_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> species = wbia.const.TEST_SPECIES.ZEB_GREVY
        >>> aid_list_ = filter_aids_to_species(ibs, aid_list, species)
        >>> result = 'aid_list_ = %r' % (aid_list_,)
        >>> print(result)
        aid_list_ = [9, 10]
    """
    species_rowid = ibs.get_species_rowids_from_text(species)
    if speedhack:
        list_repr = ','.join(map(str, aid_list))
        operation = 'SELECT rowid from annotations WHERE (species_rowid == {species_rowid}) AND rowid IN ({aids})'
        operation = operation.format(aids=list_repr, species_rowid=species_rowid)
        aid_list_ = ut.take_column(ibs.db.cur.execute(operation).fetchall(), 0)
    else:
        species_rowid_list = ibs.get_annot_species_rowids(aid_list)
        is_valid_species = [sid == species_rowid for sid in species_rowid_list]
        aid_list_ = ut.compress(aid_list, is_valid_species)
        # flag_list = [species == species_text for species_text in ibs.get_annot_species(aid_list)]
        # aid_list_ = ut.compress(aid_list, flag_list)
    return aid_list_


@register_ibs_method
def partition_annots_into_singleton_multiton(ibs, aid_list):
    """ aid_list = aid_list_ """
    aids_list = ibs.group_annots_by_name(aid_list)[0]
    singletons = [aids for aids in aids_list if len(aids) == 1]
    multitons = [aids for aids in aids_list if len(aids) > 1]
    return singletons, multitons


@register_ibs_method
def partition_annots_into_corresponding_groups(ibs, aid_list1, aid_list2):
    """
    Used for grouping one-vs-one training pairs and corerspondence filtering

    Args:
        ibs (wbia.control.IBEISControl.IBEISController):  wbia controller object
        aid_list1 (int):  list of annotation ids
        aid_list2 (int):  list of annotation ids

    Returns:
        tuple: 4 lists of lists. In the first two each list is a list of aids
            grouped by names and the names correspond with each other. In the
            last two are the annots that did not correspond with anything in
            the other list.

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-partition_annots_into_corresponding_groups

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> grouped_aids = list(map(list, ibs.group_annots_by_name(ibs.get_valid_aids())[0]))
        >>> grouped_aids = [aids for aids in grouped_aids if len(aids) > 3]
        >>> # Get some overlapping groups
        >>> import copy
        >>> aids_group1 = copy.deepcopy((ut.get_list_column_slice(grouped_aids[0:5], slice(0, 2))))
        >>> aids_group2 = copy.deepcopy((ut.get_list_column_slice(grouped_aids[2:7], slice(2, None))))
        >>> # Ensure there is a singleton in each
        >>> ut.delete_items_by_index(aids_group1[0], [0])
        >>> ut.delete_items_by_index(aids_group2[-1], [0])
        >>> aid_list1 = ut.flatten(aids_group1)
        >>> aid_list2 = ut.flatten(aids_group2)
        >>> #aid_list1 = [1, 2, 8, 9, 60]
        >>> #aid_list2 = [3, 7, 20]
        >>> groups = partition_annots_into_corresponding_groups(ibs, aid_list1, aid_list2)
        >>> result = ut.repr2(groups)
        >>> print(result)
        [[10, 11], [17, 18], [22, 23]]
        [[12, 13, 14, 15], [19, 20, 21], [24, 25, 26]]
        [[2], [5, 6]]
        [[29, 30, 31, 32], [49]]
    """
    # ibs.
    import wbia.control.IBEISControl

    assert isinstance(ibs, wbia.control.IBEISControl.IBEISController)
    # ibs
    # ibs.get_ann

    grouped_aids1 = ibs.group_annots_by_name(aid_list1)[0]
    grouped_aids1 = [aids.tolist() for aids in grouped_aids1]

    # Get the group of available aids that a reference aid could match
    gropued_aids2 = ibs.get_annot_groundtruth(
        ut.get_list_column(grouped_aids1, 0), daid_list=aid_list2, noself=False
    )

    # Flag if there is a correspondence
    flag_list = [x > 0 for x in map(len, gropued_aids2)]

    # Corresonding lists of aids groups
    gt_grouped_aids1 = ut.compress(grouped_aids1, flag_list)
    gt_grouped_aids2 = ut.compress(gropued_aids2, flag_list)

    # Non-corresponding lists of aids groups
    gf_grouped_aids1 = ut.compress(grouped_aids1, ut.not_list(flag_list))
    # gf_aids1 = ut.flatten(gf_grouped_aids1)
    gf_aids2 = ut.setdiff_ordered(aid_list2, ut.flatten(gt_grouped_aids2))
    gf_grouped_aids2 = [aids.tolist() for aids in ibs.group_annots_by_name(gf_aids2)[0]]

    gt_grouped_aids1 = list(map(sorted, gt_grouped_aids1))
    gt_grouped_aids2 = list(map(sorted, gt_grouped_aids2))
    gf_grouped_aids1 = list(map(sorted, gf_grouped_aids1))
    gf_grouped_aids2 = list(map(sorted, gf_grouped_aids2))

    return gt_grouped_aids1, gt_grouped_aids2, gf_grouped_aids1, gf_grouped_aids2


@register_ibs_method
def dans_lists(ibs, positives=10, negatives=10, verbose=False):
    from random import shuffle

    aid_list = ibs.get_valid_aids()
    yaw_list = ibs.get_annot_viewpoints(aid_list)
    qua_list = ibs.get_annot_quality_texts(aid_list)
    sex_list = ibs.get_annot_sex_texts(aid_list)
    age_list = ibs.get_annot_age_months_est(aid_list)

    positive_list = [
        aid
        for aid, yaw, qua, sex, (start, end) in zip(
            aid_list, yaw_list, qua_list, sex_list, age_list
        )
        if (
            yaw.upper() == 'LEFT'
            and qua.upper() in ['OK', 'GOOD', 'EXCELLENT']
            and sex.upper() in ['MALE', 'FEMALE']
            and start != -1
            and end != -1
        )
    ]

    negative_list = [
        aid
        for aid, yaw, qua, sex, (start, end) in zip(
            aid_list, yaw_list, qua_list, sex_list, age_list
        )
        if (
            yaw.upper() == 'LEFT'
            and qua.upper() in ['OK', 'GOOD', 'EXCELLENT']
            and sex.upper() == 'UNKNOWN SEX'
            and start == -1
            and end == -1
        )
    ]

    shuffle(positive_list)
    shuffle(negative_list)

    positive_list = sorted(positive_list[:10])
    negative_list = sorted(negative_list[:10])

    if verbose:
        pos_yaw_list = ibs.get_annot_viewpoints(positive_list)
        pos_qua_list = ibs.get_annot_quality_texts(positive_list)
        pos_sex_list = ibs.get_annot_sex_texts(positive_list)
        pos_age_list = ibs.get_annot_age_months_est(positive_list)
        pos_chip_list = ibs.get_annot_chip_fpath(positive_list)

        neg_yaw_list = ibs.get_annot_viewpoints(negative_list)
        neg_qua_list = ibs.get_annot_quality_texts(negative_list)
        neg_sex_list = ibs.get_annot_sex_texts(negative_list)
        neg_age_list = ibs.get_annot_age_months_est(negative_list)
        neg_chip_list = ibs.get_annot_chip_fpath(negative_list)

        print('positive_aid_list = %s\n' % (positive_list,))
        print('positive_yaw_list = %s\n' % (pos_yaw_list,))
        print('positive_qua_list = %s\n' % (pos_qua_list,))
        print('positive_sex_list = %s\n' % (pos_sex_list,))
        print('positive_age_list = %s\n' % (pos_age_list,))
        print('positive_chip_list = %s\n' % (pos_chip_list,))

        print('-' * 90, '\n')

        print('negative_aid_list = %s\n' % (negative_list,))
        print('negative_yaw_list = %s\n' % (neg_yaw_list,))
        print('negative_qua_list = %s\n' % (neg_qua_list,))
        print('negative_sex_list = %s\n' % (neg_sex_list,))
        print('negative_age_list = %s\n' % (neg_age_list,))
        print('negative_chip_list = %s\n' % (neg_chip_list,))

        print('mkdir ~/Desktop/chips')
        for pos_chip in pos_chip_list:
            print('cp "%s" ~/Desktop/chips/' % (pos_chip,))
        for neg_chip in neg_chip_list:
            print('cp "%s" ~/Desktop/chips/' % (neg_chip,))

    return positive_list, negative_list


def _stat_str(dict_, multi=False, precision=2, **kwargs):
    import utool as ut

    dict_ = dict_.copy()
    if dict_.get('num_nan', None) == 0:
        del dict_['num_nan']
    exclude_keys = []  # ['std', 'nMin', 'nMax']
    if multi is True:
        str_ = ut.repr2(dict_, precision=precision, nl=2, strvals=True)
    else:
        str_ = ut.get_stats_str(
            stat_dict=dict_, precision=precision, exclude_keys=exclude_keys, **kwargs
        )
    str_ = str_.replace("'", '')
    str_ = str_.replace('num_nan: 0, ', '')
    return str_


@register_ibs_method
def group_annots_by_prop(ibs, aids, getter_func):
    # Make a dictionary that maps props into a dictionary of names to aids
    annot_prop_list = getter_func(aids)
    prop_to_aids = ut.group_items(aids, annot_prop_list)
    return prop_to_aids


@register_ibs_method
def get_annot_intermediate_viewpoint_stats(ibs, aids, size=2):
    """
    >>> from wbia.other.ibsfuncs import *  # NOQA
    >>> aids = available_aids
    """
    getter_func = ibs.get_annot_viewpoints
    prop_basis = list(const.VIEW.CODE_TO_INT.keys())

    group_annots_by_view_and_name = functools.partial(
        ibs.group_annots_by_prop_and_name, getter_func=getter_func
    )
    group_annots_by_view = functools.partial(
        ibs.group_annots_by_prop, getter_func=getter_func
    )

    prop2_nid2_aids = group_annots_by_view_and_name(aids)

    edge2_nid2_aids = group_prop_edges(prop2_nid2_aids, prop_basis, size=size, wrap=True)
    # Total number of names that have two viewpoints
    # yawtext_edge_nid_hist = ut.map_dict_vals(len, edge2_nid2_aids)
    edge2_grouped_aids = ut.map_dict_vals(
        lambda dict_: list(dict_.values()), edge2_nid2_aids
    )
    edge2_aids = ut.map_dict_vals(ut.flatten, edge2_grouped_aids)
    # Num annots of each type of viewpoint

    # Regroup by view and name
    edge2_vp2_pername_stats = {}
    edge2_vp2_aids = ut.map_dict_vals(group_annots_by_view, edge2_aids)
    for edge, vp2_aids in edge2_vp2_aids.items():
        vp2_pernam_stats = ut.map_dict_vals(
            functools.partial(ibs.get_annots_per_name_stats, use_sum=True), vp2_aids
        )
        edge2_vp2_pername_stats[edge] = vp2_pernam_stats

    return edge2_vp2_pername_stats


@register_ibs_method
def group_annots_by_name_dict(ibs, aids):
    grouped_aids, nids = ibs.group_annots_by_name(aids)
    return dict(zip(nids, map(list, grouped_aids)))


@register_ibs_method
def group_annots_by_prop_and_name(ibs, aids, getter_func):
    # Make a dictionary that maps props into a dictionary of names to aids
    prop2_aids = group_annots_by_prop(ibs, aids, getter_func)
    prop2_nid2_aids = ut.map_dict_vals(ibs.group_annots_by_name_dict, prop2_aids)
    return prop2_nid2_aids


@register_ibs_method
def group_annots_by_multi_prop(ibs, aids, getter_list):
    r"""

    Performs heirachical grouping of annotations based on properties

    Args:
        ibs (IBEISController):  wbia controller object
        aids (list):  list of annotation rowids
        getter_list (list):

    Returns:
        dict: multiprop2_aids

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-group_annots_by_multi_prop --db PZ_Master1 --props=viewpoint_code,name_rowids --keys1 frontleft
        python -m wbia.other.ibsfuncs --exec-group_annots_by_multi_prop

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids(is_known=True)
        >>> #getter_list = [ibs.get_annot_name_rowids, ibs.get_annot_viewpoints]
        >>> props = ut.get_argval('--props', type_=list, default=['viewpoint_code', 'name_rowids'])
        >>> getter_list = [getattr(ibs, 'get_annot_' + prop) for prop in props]
        >>> print('getter_list = %r' % (getter_list,))
        >>> #getter_list = [ibs.get_annot_viewpoints, ibs.get_annot_name_rowids]
        >>> multiprop2_aids = group_annots_by_multi_prop(ibs, aids, getter_list)
        >>> get_dict_values = lambda x: list(x.values())
        >>> # a bit convoluted
        >>> keys1 = ut.get_argval('--keys1', type_=list, default=list(multiprop2_aids.keys()))
        >>> multiprop2_num_aids = ut.hmap_vals(len, multiprop2_aids)
        >>> prop2_num_aids = ut.hmap_vals(get_dict_values, multiprop2_num_aids, max_depth=len(props) - 2)
        >>> #prop2_num_aids_stats = ut.hmap_vals(ut.get_stats, prop2_num_aids)
        >>> prop2_num_aids_hist = ut.hmap_vals(ut.dict_hist, prop2_num_aids)
        >>> prop2_num_aids_cumhist = ut.map_dict_vals(ut.dict_hist_cumsum, prop2_num_aids_hist)
        >>> print('prop2_num_aids_hist[%s] = %s' % (keys1,  ut.repr2(ut.dict_subset(prop2_num_aids_hist, keys1))))
        >>> print('prop2_num_aids_cumhist[%s] = %s' % (keys1,  ut.repr2(ut.dict_subset(prop2_num_aids_cumhist, keys1))))

    """
    aid_prop_list = [getter(aids) for getter in getter_list]
    # %timeit multiprop2_aids = ut.hierarchical_group_items(aids, aid_prop_list)
    # %timeit ut.group_items(aids, list(zip(*aid_prop_list)))
    multiprop2_aids = ut.hierarchical_group_items(aids, aid_prop_list)
    # multiprop2_aids = ut.group_items(aids, list(zip(*aid_prop_list)))
    return multiprop2_aids


def group_prop_edges(prop2_nid2_aids, prop_basis, size=2, wrap=True):
    """
    from wbia.other.ibsfuncs import *  # NOQA
    getter_func = ibs.get_annot_viewpoints
    prop_basis = list(const.VIEWTEXT_TO_YAW_RADIANS.keys())
    size = 2
    wrap = True
    """
    # Get intermediate viewpoints
    # prop = yawtext

    # Build a list of property edges (TODO: mabye include option for all pairwise combinations)
    prop_edges = list(ut.iter_window(prop_basis, size=size, step=size - 1, wrap=wrap))
    edge2_nid2_aids = ut.odict()

    for edge in prop_edges:
        edge_nid2_aids_list = [prop2_nid2_aids.get(prop, {}) for prop in edge]
        isect_nid2_aids = reduce(
            # functools.partial(ut.dict_intersection, combine=True),
            ut.dict_isect_combine,
            edge_nid2_aids_list,
        )
        edge2_nid2_aids[edge] = isect_nid2_aids
        # common_nids = list(isect_nid2_aids.keys())
        # common_num_prop1 = np.array([len(prop2_nid2_aids[prop1][nid]) for nid in common_nids])
        # common_num_prop2 = np.array([len(prop2_nid2_aids[prop2][nid]) for nid in common_nids])
    return edge2_nid2_aids


@register_ibs_method
def parse_annot_stats_filter_kws(ibs):
    kwkeys = ut.parse_func_kwarg_keys(ibs.get_annot_stats_dict)
    return kwkeys


# Indepdentent query / database stats
@register_ibs_method
def get_annot_stats_dict(
    ibs, aids, prefix='', forceall=False, old=True, use_hist=False, **kwargs
):
    """ stats for a set of annots

    Args:
        ibs (wbia.IBEISController):  wbia controller object
        aids (list):  list of annotation rowids
        prefix (str): (default = '')

    Kwargs:
        hashid, per_name, per_qual, per_vp, per_name_vpedge, per_image, min_name_hourdist

    Returns:
        dict: aid_stats_dict

    CommandLine:
        python -m wbia get_annot_stats_dict --db WWF_Lynx --all
        python -m wbia get_annot_stats_dict --db EWT_Cheetahs --all
        python -m wbia get_annot_stats_dict --db PZ_PB_RF_TRAIN --all
        python -m wbia get_annot_stats_dict --db PZ_Master1 --all

        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict
        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_PB_RF_TRAIN --use-hist=True --old=False --per_name_vpedge=False
        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_PB_RF_TRAIN --use-hist=False --old=False --per_name_vpedge=False

        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_MTEST --use-hist --per_name_vpedge=False
        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_MTEST --use-hist --per_name_vpedge=False

        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_Master1 --per_name_vpedge=True
        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_Master1 --min_name_hourdist=True
        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db GZ_ALL --min_name_hourdist=True --all
        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db GZ_Master1 --all
        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_Master1 --min_name_hourdist=True --all
        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db NNP_MasterGIRM_core --min_name_hourdist=True --all

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aids = ibs.annots().aids
        >>> stats = ibs.get_annot_stats_dict(aids)
        >>> import ubelt as ub
        >>> print('annot_stats = {}'.format(ub.repr2(stats, nl=1)))

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aids = wbia.testdata_aids(ibs=ibs)
        >>> prefix = ''
        >>> kwkeys = ut.parse_func_kwarg_keys(get_annot_stats_dict)
        >>> #default = True if ut.get_argflag('--all') else None
        >>> default = None if ut.get_argflag('--notall') else True
        >>> kwargs = ut.argparse_dict(dict(zip(kwkeys, [default] * len(kwkeys))))
        >>> #ut.argparse_funckw(ibs.get_annot_stats_dict)
        >>> print('kwargs = %r' % (kwargs,))
        >>> old = ut.get_argval('--old', default=True)
        >>> use_hist = ut.get_argval('--use_hist', default=True)
        >>> aid_stats_dict = get_annot_stats_dict(ibs, aids, prefix, use_hist=use_hist, old=old, **kwargs)
        >>> result = ('aid_stats_dict = %s' % (ub.repr2(aid_stats_dict, strkeys=True, strvals=True, nl=2, precision=2),))
        >>> print(result)
    """
    kwargs = kwargs.copy()

    class HackStatsDict(ut.DictLike):
        # def __repr__(self):
        #     return repr(self)
        def __init__(self, dict_, **kwargs):
            self.dict_ = dict_
            self.keys = dict_.keys
            self.kwargs = kwargs
            self.getitem = dict_.__getitem__
            self.delitem = dict_.__delitem__
            self.setitem = dict_.__setitem__

        def __str__(self):
            return _stat_str(self, **self.kwargs)

        def __repr__(self):
            return repr(self.dict_)

    statwrap = HackStatsDict

    annots = ibs.annots(aids)

    def stat_func(data, num_None=None):
        if use_hist:
            stats = ut.dict_hist(data)
        else:
            stats = ut.get_stats(data, use_nan=True, use_median=True)
            if stats.get('num_nan', None) == 0:
                del stats['num_nan']
            if num_None is not None:
                stats['num_None'] = num_None
            stats = HackStatsDict(stats)
        return stats

    def get_per_prop_stats(ibs, aids, getter_func):
        prop2_aids = ibs.group_annots_by_prop(aids, getter_func=getter_func)
        num_None = 0
        if None in prop2_aids:
            # Dont count invalid properties
            num_None += len(prop2_aids[None])
            del prop2_aids[None]
        if ibs.const.UNKNOWN in prop2_aids:
            num_None += len(prop2_aids[ibs.const.UNKNOWN])
            del prop2_aids[ibs.const.UNKNOWN]
        num_aids_list = list(map(len, prop2_aids.values()))
        # represent that Nones were removed
        # num_aids_list += ([np.nan] * num_None)
        # if num_None:
        num_aids_stats = stat_func(num_aids_list, num_None)
        return num_aids_stats

    keyval_list = [
        ('num_' + prefix + 'aids', len(aids)),
    ]

    if kwargs.pop('bigstr', False or forceall):
        bigstr = functools.partial(ut.truncate_str, maxlen=64, truncmsg=' ~TRUNC~ ')
        keyval_list += [(prefix + 'bigstr', bigstr(str(aids)))]

    if kwargs.pop('hashid', True or forceall):
        # TODO: depricate semantic hashid
        keyval_list += [
            (
                prefix + 'hashid',
                ibs.get_annot_hashid_semantic_uuid(aids, prefix=prefix.upper()),
            )
        ]

    if kwargs.pop('hashid_visual', False or forceall):
        keyval_list += [
            (
                prefix + 'hashid_visual',
                ibs.get_annot_hashid_visual_uuid(aids, prefix=prefix.upper()),
            )
        ]

    if kwargs.pop('hashid_uuid', False or forceall):
        keyval_list += [
            (
                prefix + 'hashid_uuid',
                ibs.get_annot_hashid_uuid(aids, prefix=prefix.upper()),
            )
        ]

    if kwargs.pop('per_name', True or forceall):
        data = ibs.get_num_annots_per_name(aids)[0]
        stats = stat_func(data)
        keyval_list += [(prefix + 'per_name', stats)]
        if not use_hist:
            a = ibs.annots(aids)
            pername = ut.dict_hist(ut.lmap(len, a.group_items(a.nids).values()))
            pername_bins = ut.odict(
                [
                    ('1', sum(v for k, v in pername.items() if k == 1)),
                    ('2-3', sum(v for k, v in pername.items() if k >= 2 and k < 4)),
                    ('4-5', sum(v for k, v in pername.items() if k >= 4 and k < 6)),
                    ('>=6', sum(v for k, v in pername.items() if k >= 6)),
                ]
            )
            keyval_list += [(prefix + 'per_name_bins', pername_bins)]

    # if kwargs.pop('per_name_dict', True or forceall):
    #     keyval_list += [
    #         (prefix + 'per_name_dict',
    #          ut.get_stats(ibs.get_num_annots_per_name(aids)[0],
    #                                 use_nan=True, use_median=True))]

    if kwargs.pop('per_qual', False or forceall):
        qualtext2_nAnnots = ut.order_dict_by(
            ut.map_vals(len, annots.group_items(annots.quality_texts)),
            list(ibs.const.QUALITY_TEXT_TO_INT.keys()),
        )
        keyval_list += [(prefix + 'per_qual', statwrap(qualtext2_nAnnots))]

    # if kwargs.pop('per_vp', False):
    if kwargs.pop('per_vp', True or forceall):
        yawtext2_nAnnots = ut.order_dict_by(
            ut.map_vals(len, annots.group_items(annots.viewpoint_code)),
            list(const.VIEW.CODE_TO_INT.keys()),
        )
        keyval_list += [(prefix + 'per_vp', statwrap(yawtext2_nAnnots))]

    if kwargs.pop('per_multiple', True or forceall):
        keyval_list += [
            (
                prefix + 'per_multiple',
                statwrap(ut.map_vals(len, annots.group_items(annots.multiple))),
            )
        ]

    # information about overlapping viewpoints
    if kwargs.pop('per_name_vpedge', False or forceall):
        keyval_list += [
            (
                prefix + 'per_name_vpedge',
                statwrap(ibs.get_annot_intermediate_viewpoint_stats(aids), multi=True),
            )
        ]

    if kwargs.pop('per_enc', True or forceall):
        keyval_list += [
            (
                prefix + 'per_enc',
                statwrap(get_per_prop_stats(ibs, aids, ibs.get_annot_encounter_text)),
            )
        ]

    if kwargs.pop('per_image', False or forceall):
        keyval_list += [
            (
                prefix + 'aid_per_image',
                statwrap(get_per_prop_stats(ibs, aids, ibs.get_annot_image_rowids)),
            )
        ]

    if kwargs.pop('enc_per_name', True or forceall):
        # Does not handle None encounters. They show up as just another encounter
        name_to_enc_ = ut.group_items(annots.encounter_text, annots.names)
        name_to_enc = ut.map_vals(set, name_to_enc_)
        name_to_num_enc = ut.map_vals(len, name_to_enc)
        num_enc_per_name = list(name_to_num_enc.values())
        stats = stat_func(num_enc_per_name)
        keyval_list += [(prefix + 'enc_per_name', stats)]

    if kwargs.pop('species_hist', False or forceall):
        keyval_list += [
            (prefix + 'species_hist', ut.dict_hist(ibs.get_annot_species_texts(aids)))
        ]

    if kwargs.pop('case_tag_hist', False or forceall):
        keyval_list += [
            (
                prefix + 'case_tags',
                ut.dict_hist(ut.flatten(ibs.get_annot_case_tags(aids))),
            )
        ]

    if kwargs.pop('match_tag_hist', False or forceall):
        keyval_list += [
            (
                prefix + 'match_tags',
                ut.dict_hist(ut.flatten(ibs.get_annot_annotmatch_tags(aids))),
            )
        ]

    if kwargs.pop('match_state', False or forceall):
        am_rowids = annots.get_am_rowids(internal=True)
        truths = ibs.get_annotmatch_evidence_decision(am_rowids)
        truths = np.array(ut.replace_nones(truths, np.nan))
        match_state = ut.odict(
            [
                ('None', np.isnan(truths).sum()),
                ('unknown', (truths == ibs.const.EVIDENCE_DECISION.UNKNOWN).sum()),
                ('incomp', (truths == ibs.const.EVIDENCE_DECISION.INCOMPARABLE).sum()),
                ('nomatch', (truths == ibs.const.EVIDENCE_DECISION.NEGATIVE).sum()),
                ('match', (truths == ibs.const.EVIDENCE_DECISION.POSITIVE).sum()),
            ]
        )
        keyval_list += [(prefix + 'match_state', match_state)]

    if kwargs.pop('min_name_hourdist', False or forceall):
        grouped_aids = ibs.group_annots_by_name(aids)[0]
        # ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, grouped_aids)
        timedeltas = ibs.get_unflat_annots_timedelta_list(grouped_aids)
        # timedeltas = [dists for dists in timedeltas if dists is not np.nan and
        # dists is not None]
        # timedeltas = [np.nan if dists is None else dists for dists in timedeltas]
        # min_timedelta_list = [np.nan if dists is None else dists.min() / (60 *
        # 60 * 24) for dists in timedeltas]
        # convert to hours
        min_timedelta_list = [
            np.nan if dists is None else dists.min() / (60 * 60) for dists in timedeltas
        ]
        # min_timedelta_list = [np.nan if dists is None else dists.min() for dists in timedeltas]
        min_name_timedelta_stats = ut.get_stats(min_timedelta_list, use_nan=True)
        keyval_list += [
            (
                prefix + 'min_name_hourdist',
                _stat_str(min_name_timedelta_stats, precision=4),
            )
        ]

    aid_stats_dict = ut.odict(keyval_list)
    return aid_stats_dict


@register_ibs_method
def print_annot_stats(ibs, aids, prefix='', label='', **kwargs):
    aid_stats_dict = ibs.get_annot_stats_dict(aids, prefix=prefix, **kwargs)
    print(label + ut.repr4(aid_stats_dict, strkeys=True, strvals=True))


@register_ibs_method
def compare_nested_props(ibs, aids1_list, aids2_list, getter_func, cmp_func):
    """
    Compares properties of query vs database annotations

    grouped_qaids = aids1_list
    grouped_groundtruth_list = aids2_list

    getter_func = ibs.get_annot_yaws
    cmp_func = vt.ori_distance

    getter_func = ibs.get_annot_image_unixtimes_asfloat
    cmp_func = ut.unixtime_hourdiff

    ExpandNestedComparisions:
        import itertools
        list(map(list, itertools.starmap(ut.iprod, zip(aids1_list, aids2_list))))

    Args:
        ibs (IBEISController):  wbia controller object
        aids1_list (list):
        aids2_list (list):
        getter_func (?):
        cmp_func (?):

    Returns:
        list of ndarrays:

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-compare_nested_props --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> aids1_list = [ibs.get_valid_aids()[8:11]]
        >>> aids2_list = [ibs.get_valid_aids()[8:11]]
        >>> getter_func = ibs.get_annot_image_unixtimes_asfloat
        >>> cmp_func = ut.unixtime_hourdiff
        >>> result = compare_nested_props(ibs, aids1_list, aids2_list, getter_func, cmp_func)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """

    def replace_none_with_nan(x):
        import utool as ut

        return np.array(ut.replace_nones(x, np.nan))

    props1_list = ibs.unflat_map(getter_func, aids1_list)
    props1_list = ibs.unflat_map(replace_none_with_nan, props1_list)

    props2_list = ibs.unflat_map(getter_func, aids2_list)
    props2_list = ibs.unflat_map(replace_none_with_nan, props2_list)
    # Compare the query yaws to the yaws of its correct matches in the database
    propdist_list = [
        cmp_func(np.array(qprops), np.array(gt_props)[:, None])
        for qprops, gt_props in zip(props1_list, props2_list)
    ]
    return propdist_list


def viewpoint_diff(ori1, ori2):
    """ convert distance in radians to distance in viewpoint category """
    # TODO: lookup distance
    TAU = np.pi * 2
    ori_diff = vt.ori_distance(ori1, ori2)
    viewpoint_diff = len(const.VIEWTEXT_TO_YAW_RADIANS) * ori_diff / TAU
    return viewpoint_diff


@register_ibs_method
def parse_annot_config_stats_filter_kws(ibs):
    # kwkeys = ibs.parse_annot_stats_filter_kws() + ['combined', 'combo_gt_info', 'combo_enc_info', 'combo_dists']
    kwkeys1 = ibs.parse_annot_stats_filter_kws()
    kwkeys2 = list(ut.get_func_kwargs(ibs.get_annotconfig_stats).keys())
    if 'verbose' in kwkeys2:
        kwkeys2.remove('verbose')
    kwkeys = ut.unique(kwkeys1 + kwkeys2)
    return kwkeys


@register_ibs_method
def print_annotconfig_stats(ibs, qaids, daids, **kwargs):
    """
    SeeAlso:
        ibs.get_annotconfig_stats
    """
    annotconfig_stats = ibs.get_annotconfig_stats(qaids, daids, verbose=False, **kwargs)
    stats_str2 = ut.repr4(
        annotconfig_stats, strvals=True, strkeys=True, nl=True, explicit=False, nobr=False
    )
    print(stats_str2)


@register_ibs_method
def get_annotconfig_stats(
    ibs,
    qaids,
    daids,
    verbose=False,
    combined=False,
    combo_gt_info=True,
    combo_enc_info=False,
    combo_dists=True,
    split_matchable_data=True,
    **kwargs
):
    r"""
    Gets statistics about a query / database set of annotations

    USEFUL DEVELOPER FUNCTION

    TODO: this function should return non-string values in dictionaries.
    The print function should do string conversions

    Args:
        ibs (IBEISController):  wbia controller object
        qaids (list):  query annotation ids
        daids (list):  database annotation ids

    SeeAlso:
        wbia.dbinfo.print_qd_info
        ibs.get_annot_stats_dict
        ibs.print_annotconfig_stats(qaid_list, daid_list)

    CommandLine:
        python -m wbia.other.ibsfuncs get_annotconfig_stats --db PZ_MTEST -a default
        python -m wbia.other.ibsfuncs get_annotconfig_stats --db testdb1  -a default
        python -m wbia.other.ibsfuncs get_annotconfig_stats --db PZ_MTEST -a controlled
        python -m wbia.other.ibsfuncs get_annotconfig_stats --db PZ_FlankHack -a default:qaids=allgt
        python -m wbia.other.ibsfuncs get_annotconfig_stats --db PZ_MTEST -a controlled:per_name=2,min_gt=4

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> from wbia.init import main_helpers
        >>> kwargs = {'per_enc': True, 'enc_per_name': True}
        >>> ibs, qaids, daids = main_helpers.testdata_expanded_aids(
        ...    defaultdb='testdb1', a='default:qsize=3')
        >>> stat_dict = get_annotconfig_stats(ibs, qaids, daids, **kwargs)
        >>> stats_str2 = ut.repr2(stat_dict, si=True, nl=True, nobr=False)
        >>> print(stats_str2)
    """
    import numpy as np
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice.')

        # The aids that should be matched by a query
        grouped_qaids = ibs.group_annots_by_name(qaids)[0]
        grouped_groundtruth_list = ibs.get_annot_groundtruth(
            ut.get_list_column(grouped_qaids, 0), daid_list=daids
        )
        # groundtruth_daids = ut.unique(ut.flatten(grouped_groundtruth_list))
        query_hasgt_list = ibs.get_annot_has_groundtruth(qaids, daid_list=daids)
        # The aids that should not match any query
        # nonquery_daids = np.setdiff1d(np.setdiff1d(daids, qaids), groundtruth_daids)
        # The query aids that should not get any match
        unmatchable_queries = ut.compress(qaids, ut.not_list(query_hasgt_list))
        # The query aids that should not have a match
        matchable_queries = ut.compress(qaids, query_hasgt_list)

        # Find the daids that are in the same occurrence as the qaids
        if combo_enc_info:
            query_encs = set(ibs.get_annot_encounter_text(qaids))
            data_encs = set(ibs.get_annot_encounter_text(daids))
            enc_intersect = query_encs.intersection(data_encs)
            enc_intersect.difference_update({None})

        # Compare the query yaws to the yaws of its correct matches in the database
        # For each name there will be nQaids:nid x nDaids:nid comparisons
        gt_viewdist_list = compare_nested_props(
            ibs,
            grouped_qaids,
            grouped_groundtruth_list,
            ibs.get_annot_yaws,
            viewpoint_diff,
        )

        # Compare the query qualities to the qualities of its correct matches in the database
        # gt_qualdists_list = compare_nested_props(
        #     ibs, grouped_qaids, grouped_groundtruth_list, ibs.get_annot_qualities, ut.absdiff)

        # Compare timedelta differences
        gt_hourdelta_list = compare_nested_props(
            ibs,
            grouped_qaids,
            grouped_groundtruth_list,
            ibs.get_annot_image_unixtimes_asfloat,
            ut.unixtime_hourdiff,
        )

        def super_flatten(arr_list):
            import utool as ut

            return ut.flatten([arr.ravel() for arr in arr_list])

        gt_viewdist_stats = ut.get_stats(super_flatten(gt_viewdist_list), use_nan=True)
        # gt_qualdist_stats  = ut.get_stats(super_flatten(gt_qualdists_list), use_nan=True)
        gt_hourdelta_stats = ut.get_stats(super_flatten(gt_hourdelta_list), use_nan=True)

        qaids2 = np.array(qaids).copy()
        daids2 = np.array(daids).copy()
        qaids2.sort()
        daids2.sort()
        if not np.all(qaids2 == qaids):
            print('WARNING: qaids are not sorted')
            # raise AssertionError('WARNING: qaids are not sorted')
        if not np.all(daids2 == daids):
            print('WARNING: daids are not sorted')
            # raise AssertionError('WARNING: qaids are not sorted')

        qaid_stats_dict = ibs.get_annot_stats_dict(qaids, 'q', **kwargs)
        daid_stats_dict = ibs.get_annot_stats_dict(daids, 'd', **kwargs)

        if split_matchable_data:
            # The aids that should not be matched by any query
            data_hasgt_list = ibs.get_annot_has_groundtruth(daids, daid_list=qaids)
            matchable_daids = ut.compress(daids, data_hasgt_list)
            confusor_daids = ut.compress(daids, ut.not_list(data_hasgt_list))

            matchable_daid_stats_dict = ibs.get_annot_stats_dict(
                matchable_daids, 'd', **kwargs
            )
            confusor_daid_stats_dict = ibs.get_annot_stats_dict(
                confusor_daids, 'd', **kwargs
            )

            daid_stats_dict = ut.dict_subset(daid_stats_dict, ['num_daids', 'dhashid'])

        # Intersections between qaids and daids
        common_aids = np.intersect1d(daids, qaids)

        qnids = ut.unique(ibs.get_annot_name_rowids(qaids))
        dnids = ut.unique(ibs.get_annot_name_rowids(daids))
        common_nids = np.intersect1d(qnids, dnids)

        annotconfig_stats_strs_list1 = []
        annotconfig_stats_strs_list2 = []
        annotconfig_stats_strs_list1 += [
            ('dbname', ibs.get_dbname()),
            ('num_qaids', (len(qaids))),
            ('num_daids', (len(daids))),
            ('num_annot_intersect', (len(common_aids))),
        ]

        annotconfig_stats_strs_list1 += [
            ('qaid_stats', qaid_stats_dict),
        ]
        annotconfig_stats_strs_list1 += [
            ('daid_stats', daid_stats_dict),
        ]
        if split_matchable_data:
            annotconfig_stats_strs_list1 += [
                ('matchable_daid_stats', matchable_daid_stats_dict),
                ('confusor_daid_stats', confusor_daid_stats_dict),
            ]
            matchable_daid_stats_dict = ibs.get_annot_stats_dict(
                matchable_daids, 'd', **kwargs
            )
            confusor_daid_stats_dict = ibs.get_annot_stats_dict(
                confusor_daids, 'd', **kwargs
            )
        if combined:
            combined_aids = np.unique((np.hstack((qaids, daids))))
            combined_aids.sort()
            annotconfig_stats_strs_list1 += [
                ('combined_aids', ibs.get_annot_stats_dict(combined_aids, **kwargs)),
            ]

        if combo_gt_info:
            annotconfig_stats_strs_list1 += [
                ('num_unmatchable_queries', len(unmatchable_queries)),
                ('num_matchable_queries', len(matchable_queries)),
            ]

        if combo_enc_info:
            annotconfig_stats_strs_list1 += [
                # ('num_qnids', (len(qnids))),
                # ('num_dnids', (len(dnids))),
                ('num_enc_intersect', (len(enc_intersect))),
                ('num_name_intersect', (len(common_nids))),
            ]

        if combo_dists:
            annotconfig_stats_strs_list2 += [
                # Distances between a query and its groundtruth
                ('viewdist', _stat_str(gt_viewdist_stats)),
                # ('qualdist', _stat_str(gt_qualdist_stats)),
                ('hourdist', _stat_str(gt_hourdelta_stats, precision=4)),
            ]

        annotconfig_stats_strs1 = ut.odict(annotconfig_stats_strs_list1)
        annotconfig_stats_strs2 = ut.odict(annotconfig_stats_strs_list2)

        annotconfig_stats = ut.odict(
            list(annotconfig_stats_strs1.items()) + list(annotconfig_stats_strs2.items())
        )
        if verbose:
            stats_str2 = ut.repr2(
                annotconfig_stats,
                strvals=True,
                newlines=True,
                explicit=False,
                nobraces=False,
            )
            print('annot_config_stats = ' + stats_str2)

        return annotconfig_stats


@register_ibs_method
def get_dbname_alias(ibs):
    """ convinience for plots """
    dbname = ibs.get_dbname()

    return const.DBNAME_ALIAS.get(dbname, dbname)


@register_ibs_method
def find_unlabeled_name_members(ibs, **kwargs):
    r"""
    Find annots where some members of a name have information but others do not.

    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-find_unlabeled_name_members --qual

    Example:
        >>> # SCRIPT
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_Master1')
        >>> defaultdict = dict(ut.parse_func_kwarg_keys(find_unlabeled_name_members, with_vals=True))
        >>> kwargs = ut.argparse_dict(defaultdict)
        >>> result = find_unlabeled_name_members(ibs, **kwargs)
        >>> print(result)
    """
    aid_list = ibs.get_valid_aids()
    aids_list, nids = ibs.group_annots_by_name(aid_list)
    aids_list = ut.compress(aids_list, [len(aids) > 1 for aids in aids_list])

    def find_missing(props_list, flags_list):
        missing_idx_list = ut.list_where(
            [any(flags) and not all(flags) for flags in flags_list]
        )
        missing_flag_list = ut.take(flags_list, missing_idx_list)
        missing_aids_list = ut.take(aids_list, missing_idx_list)
        # missing_prop_list = ut.take(props_list, missing_idx_list)
        missing_aid_list = vt.zipcompress(missing_aids_list, missing_flag_list)

        if False:
            missing_percent_list = [
                sum(flags) / len(flags) for flags in missing_flag_list
            ]
            print('Missing per name stats')
            print(ut.repr2(ut.get_stats(missing_percent_list, use_median=True)))
        return missing_aid_list

    selected_aids_list = []

    if kwargs.get('time', False):
        props_list = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, aids_list)
        flags_list = [np.isnan(props) for props in props_list]
        missing_time_aid_list = find_missing(props_list, flags_list)
        print('missing_time_aid_list = %r' % (len(missing_time_aid_list),))
        selected_aids_list.append(missing_time_aid_list)

    if kwargs.get('yaw', False):
        props_list = ibs.unflat_map(ibs.get_annot_yaws, aids_list)
        flags_list = [ut.flag_None_items(props) for props in props_list]
        missing_yaw_aid_list = find_missing(props_list, flags_list)
        print('num_names_missing_yaw = %r' % (len(missing_yaw_aid_list),))
        selected_aids_list.append(missing_yaw_aid_list)

    if kwargs.get('qual', False):
        props_list = ibs.unflat_map(ibs.get_annot_qualities, aids_list)
        flags_list = [[p is None or p == -1 for p in props] for props in props_list]
        missing_qual_aid_list = find_missing(props_list, flags_list)
        print('num_names_missing_qual = %r' % (len(missing_qual_aid_list),))
        selected_aids_list.append(missing_qual_aid_list)

    if kwargs.get('suspect_yaws', False):
        yaws_list = ibs.unflat_map(ibs.get_annot_yaws, aids_list)
        time_list = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, aids_list)
        max_timedelta_list = np.array(
            [
                np.nanmax(ut.safe_pdist(unixtime_arr[:, None], metric=ut.absdiff))
                for unixtime_arr in time_list
            ]
        )
        flags = max_timedelta_list > 60 * 60 * 1

        aids1 = ut.compress(aids_list, flags)
        max_yawdiff_list = np.array(
            [
                np.nanmax(ut.safe_pdist(np.array(yaws)[:, None], metric=vt.ori_distance))
                for yaws in ut.compress(yaws_list, flags)
            ]
        )

        # Find annots with large timedeltas but 0 viewpoint difference
        flags2 = max_yawdiff_list == 0
        selected_aids_list.append(ut.compress(aids1, flags2))

    x = ut.flatten(selected_aids_list)
    y = ut.sortedby2(x, list(map(len, x)))
    selected_aids = ut.unique_ordered(ut.flatten(y))
    return selected_aids

    # ibs.unflat_map(ibs.get_annot_quality_texts, aids_list)
    # ibs.unflat_map(ibs.get_annot_viewpoints, aids_list)


@register_ibs_method
def get_annot_pair_lazy_dict(ibs, qaid, daid, qconfig2_=None, dconfig2_=None):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        qaid (int):  query annotation id
        daid (?):
        qconfig2_ (dict): (default = None)
        dconfig2_ (dict): (default = None)

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-get_annot_pair_lazy_dict

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> qaid, daid = ibs.get_valid_aids()[0:2]
        >>> qconfig2_ = None
        >>> dconfig2_ = None
        >>> result = get_annot_pair_lazy_dict(ibs, qaid, daid, qconfig2_, dconfig2_)
        >>> print(result)
    """
    metadata = ut.LazyDict(
        {
            'annot1': get_annot_lazy_dict(ibs, qaid, config2_=qconfig2_),
            'annot2': get_annot_lazy_dict(ibs, daid, config2_=dconfig2_),
        },
        reprkw=dict(truncate=True),
    )
    return metadata


@register_ibs_method
def get_annot_lazy_dict(ibs, aid, config2_=None):
    r"""
    Args:
        ibs (wbia.IBEISController):  image analysis api
        aid (int):  annotation id
        config2_ (dict): (default = None)

    Returns:
        ut.LazyDict: metadata

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-get_annot_lazy_dict --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid = 1
        >>> config2_ = None
        >>> metadata = get_annot_lazy_dict(ibs, aid, config2_)
        >>> result = ('metadata = %s' % (ut.repr3(metadata),))
        >>> print(result)
    """
    # if False:
    #     metadata1 = ut.LazyDict({
    #         'aid': aid,
    #         'name': lambda: ibs.get_annot_names([aid])[0],
    #         'nid': lambda: ibs.get_annot_name_rowids([aid])[0],
    #         'rchip_fpath': lambda: ibs.get_annot_chip_fpath([aid], config2_=config2_)[0],
    #         'rchip': lambda: ibs.get_annot_chips([aid], config2_=config2_)[0],
    #         'vecs': lambda:  ibs.get_annot_vecs([aid], config2_=config2_)[0],
    #         'kpts': lambda:  ibs.get_annot_kpts([aid], config2_=config2_)[0],
    #         'chip_size': lambda: ibs.get_annot_chip_sizes([aid], config2_=config2_)[0],
    #         'dlen_sqrd': lambda: ibs.get_annot_chip_dlensqrd([aid], config2_=config2_)[0],
    #         # global measures
    #         'yaw': lambda: ibs.get_annot_yaws_asfloat(aid),
    #         'qual': lambda: ibs.get_annot_qualities(aid),
    #         'gps': lambda: ibs.get_annot_image_gps2(aid),
    #         'time': lambda: ibs.get_annot_image_unixtimes_asfloat(aid),
    #         'annot_context_options': lambda: interact_chip.build_annot_context_options(ibs, aid),
    #     }, reprkw=dict(truncate=True))
    annot = ibs.annots([aid], config=config2_)[0]
    metadata = annot._make_lazy_dict()
    # metadata['rchip'] = metadata.getitem('chips', is_eager=False)
    # metadata['dlen_sqrd'] = metadata.getitem('chip_dlensqrd', is_eager=False)
    # metadata['rchip_fpath'] = metadata.getitem('chip_fpath', is_eager=False)
    try:
        from wbia.viz.interact import interact_chip

        metadata[
            'annot_context_options'
        ] = lambda: interact_chip.build_annot_context_options(ibs, aid)
    except ImportError:
        pass
    return metadata


@register_ibs_method
def get_image_lazydict(ibs, gid, config=None):
    r"""
    Args:
        ibs (wbia.IBEISController):  image analysis api
        aid (int):  annotation id
        config (dict): (default = None)

    Returns:
        ut.LazyDict: metadata

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-get_annot_lazy_dict2 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> gid = 1
    """
    # from wbia.viz.interact import interact_chip
    metadata = ut.LazyDict(
        {
            'gid': gid,
            'unixtime': lambda: ibs.get_image_unixtime(gid),
            'datetime': lambda: ibs.get_image_datetime(gid),
            'aids': lambda: ibs.get_image_aids(gid),
            'size': lambda: ibs.get_image_sizes(gid),
            'uri': lambda: ibs.get_image_uris(gid),
            'uuid': lambda: ibs.get_image_uuids(gid),
            'gps': lambda: ibs.get_image_gps(gid),
            'orientation': lambda: ibs.get_image_orientation(gid),
            # 'annot_context_options': lambda: interact_chip.build_annot_context_options(ibs, aid),
        },
        reprkw=dict(truncate=True),
    )
    return metadata


@register_ibs_method
def get_image_instancelist(ibs, gid_list):
    # DEPRICATE
    obj_list = [ibs.get_image_lazydict(gid) for gid in gid_list]
    image_list = ut.instancelist(obj_list, check=False)
    return image_list


@register_ibs_method
def get_annot_instancelist(ibs, aid_list):
    # DEPRICATE
    obj_list = [ibs.get_annot_lazydict(aid) for aid in aid_list]
    annot_list = ut.instancelist(obj_list, check=False)
    return annot_list


@register_ibs_method
def get_annot_lazy_dict2(ibs, aid, config=None):
    r"""
    DEPRICATE FOR ibs.annots

    Args:
        ibs (wbia.IBEISController):  image analysis api
        aid (int):  annotation id
        config (dict): (default = None)

    Returns:
        ut.LazyDict: metadata

    CommandLine:
        python -m wbia.other.ibsfuncs --exec-get_annot_lazy_dict2 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aid = 1
        >>> config = {'dim_size': 450}
        >>> metadata = get_annot_lazy_dict2(ibs, aid, config)
        >>> result = ('metadata = %s' % (ut.repr3(metadata),))
        >>> print(result)
    """
    defaults = {
        'aid': aid,
        'name': lambda: ibs.get_annot_names(aid),
        'rchip_fpath': lambda: ibs.depc_annot.get(
            'chips', aid, 'img', config, read_extern=False
        ),
        'rchip': lambda: ibs.depc_annot.get('chips', aid, 'img', config),
        'vecs': lambda: ibs.depc_annot.get('feat', aid, 'vecs', config),
        'kpts': lambda: ibs.depc_annot.get('feat', aid, 'kpts', config),
        'dlen_sqrd': lambda: ibs.depc_annot['chips'].subproperties['dlen_sqrd'](
            ibs.depc_annot, [aid], config
        )[0],
    }
    try:
        from wbia.viz.interact import interact_chip
    except ImportError:
        pass
    else:
        # get_annot_chip_dlensqrd([aid], config=config)[0],
        defaults[
            'annot_context_options'
        ] = lambda: interact_chip.build_annot_context_options(ibs, aid)
    metadata = ut.LazyDict(defaults)
    return metadata


@register_ibs_method
def get_annot_primary_imageset(ibs, aid_list=None):
    # TODO: make it better
    imgsetids_list = ibs.get_annot_imgsetids(aid_list)
    flags_list = ibs.unflat_map(ibs.is_special_imageset, imgsetids_list)
    # GET IMAGESET QUERY STRUCTURE DATA
    flags_list = ibs.unflat_map(ut.not_list, flags_list)
    imgsetids_list = ut.list_zipcompress(imgsetids_list, flags_list)
    imgsetid_list = ut.get_list_column(imgsetids_list, 0)
    return imgsetid_list


@register_ibs_method
@profile
def lookup_annot_vecs_subset(ibs, unflat_aids, unflat_fxs, annots=None, config2_=None):
    """
    unflat_aids = naids_list
    unflat_fxs = nfxs_list
    annots = data_annots
    config2_ = data_config2_

    unflat_aids = cm.filtnorm_aids[0]
    unflat_fxs  = cm.filtnorm_fxs[0]
    """
    aids = np.unique(ut.flatten(unflat_aids))
    if annots is None:
        annots = {}
        annots = {aid: ibs.get_annot_lazy_dict(aid, config2_=config2_) for aid in aids}
    else:
        for aid in set(aids) - set(annots.keys()):
            annots[aid] = ibs.get_annot_lazy_dict(aid, config2_=config2_)

    # for annot in annots.values():
    #    annot.eager_eval('vecs')

    def extract_vecs(annots, aid, fxs):
        """ custom_func(lazydict, key, subkeys) for multigroup_lookup """
        vecs = annots[aid]['vecs'].take(fxs, axis=0)
        return vecs

    # unflat_vecs1 = vt.multigroup_lookup(annots, unflat_aids, unflat_fxs, extract_vecs)
    # HACK
    # FIXME: naive and regular multigroup still arnt equivalent
    # unflat_vecs = unflat_vecs1 = [[] if len(x) == 1 and x[0] is None else x  for x in unflat_vecs1]
    unflat_vecs = vt.multigroup_lookup_naive(
        annots, unflat_aids, unflat_fxs, extract_vecs
    )  # NOQA
    return unflat_vecs


@register_ibs_method
def get_annot_vecs_subset(ibs, aid_list, fxs_list, config2_=None):
    vecs_list = ibs.get_annot_vecs(aid_list, config2_=config2_)
    vecs_list = vt.ziptake(vecs_list, fxs_list, axis=0)
    return vecs_list


@register_ibs_method
def get_annot_fgweights_subset(ibs, aid_list, fxs_list, config2_=None):
    fgweight_list = ibs.get_annot_fgweights(aid_list, config2_=config2_)
    vecs_list = vt.ziptake(fgweight_list, fxs_list, axis=0)
    return vecs_list


@register_ibs_method
def _clean_species(ibs):
    if ut.VERBOSE:
        print('[_clean_species] Cleaning...')
    if ibs.readonly:
        # SUPER HACK
        return
    species_mapping_dict = {}
    if ibs is not None:
        flag = '--allow-keyboard-database-update'
        from six.moves import input as raw_input_
        from wbia.control.manual_species_funcs import _convert_species_nice_to_code

        species_rowid_list = ibs._get_all_species_rowids()
        species_text_list = ibs.get_species_texts(species_rowid_list)
        species_nice_list = ibs.get_species_nice(species_rowid_list)
        species_code_list = ibs.get_species_codes(species_rowid_list)
        for rowid, text, nice, code in zip(
            species_rowid_list, species_text_list, species_nice_list, species_code_list
        ):
            alias = None
            if text in const.SPECIES_MAPPING:
                species_code, species_nice = const.SPECIES_MAPPING[text]
                while species_code is None:
                    alias = species_nice
                    species_code, species_nice = const.SPECIES_MAPPING[species_nice]
            elif text is None or text.strip() in ['_', const.UNKNOWN, 'none', 'None', '']:
                print('[_clean_species] deleting species: %r' % (text,))
                ibs.delete_species(rowid)
                continue
            elif len(nice) == 0:
                if not ut.get_argflag(flag):
                    species_nice = text
                    species_code = _convert_species_nice_to_code([species_nice])[0]
                else:
                    print('Found an unknown species: %r' % (text,))
                    species_nice = raw_input_('Input a NICE name for %r: ' % (text,))
                    species_code = raw_input_('Input a CODE name for %r: ' % (text,))
                    assert len(species_code) > 0 and len(species_nice) > 0
            else:
                continue
            if nice != species_nice or code != species_code:
                ibs._set_species_nice([rowid], [species_nice])
                ibs._set_species_code([rowid], [species_code])
            if alias is not None:
                alias_rowid = ibs.get_species_rowids_from_text(alias, skip_cleaning=True)
                aid_list = ibs._get_all_aids()
                species_rowid_list = ibs.get_annot_species_rowids(aid_list)
                aid_list_ = [
                    aid
                    for aid, species_rowid in zip(aid_list, species_rowid_list)
                    if species_rowid == rowid
                ]
                species_mapping_dict[rowid] = alias_rowid
                ibs.set_annot_species_rowids(aid_list_, [alias_rowid] * len(aid_list_))
                ibs.delete_species([rowid])
    return species_mapping_dict


@register_ibs_method
def get_annot_encounter_text(ibs, aids):
    """ Encounter identifier for annotations """
    occur_texts = ibs.get_annot_occurrence_text(aids)
    name_texts = ibs.get_annot_names(aids)
    enc_texts = [
        ot + '_' + nt if ot is not None and nt is not None else None
        for ot, nt in zip(occur_texts, name_texts)
    ]
    return enc_texts


@register_ibs_method
def get_annot_occurrence_text(ibs, aids):
    """ Occurrence identifier for annotations

    Args:
        ibs (wbia.IBEISController):  image analysis api
        aids (list):  list of annotation rowids

    Returns:
        list: occur_texts

    CommandLine:
        python -m wbia.other.ibsfuncs get_annot_occurrence_text --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids()
        >>> occur_texts = get_annot_occurrence_text(ibs, aids)
        >>> result = ('occur_texts = %s' % (ut.repr2(occur_texts),))
        >>> print(result)
    """
    imgset_ids = ibs.get_annot_imgsetids(aids)
    flags = ibs.unflat_map(ibs.get_imageset_occurrence_flags, imgset_ids)
    # flags = [[text.lower().startswith('occurrence') for text in texts]
    #         for texts in imgset_texts]
    imgset_ids = ut.zipcompress(imgset_ids, flags)
    _occur_texts = ibs.unflat_map(ibs.get_imageset_text, imgset_ids)
    _occur_texts = [t if len(t) > 0 else [None] for t in _occur_texts]
    if not all([len(t) == 1 for t in _occur_texts]):
        print(
            '[%s] WARNING: annot must be in exactly one occurrence'
            % (ut.get_caller_name(),)
        )
    occur_texts = ut.take_column(_occur_texts, 0)
    return occur_texts


@register_ibs_method
def _parse_smart_xml(back, xml_path, nTotal, offset=1):
    # Storage for the patrol imagesets
    xml_dir, xml_name = split(xml_path)
    imageset_info_list = []
    last_photo_number = None
    last_imageset_info = None
    # Parse the XML file for the information
    patrol_tree = ET.parse(xml_path)
    namespace = '{http://www.smartconservationsoftware.org/xml/1.1/patrol}'
    # Load all waypoint elements
    element = './/%swaypoints' % (namespace,)
    waypoint_list = patrol_tree.findall(element)
    if len(waypoint_list) == 0:
        # raise IOError('There are no observations (waypoints) in this
        # Patrol XML file: %r' % (xml_path, ))
        print(
            'There are no observations (waypoints) in this Patrol XML file: %r'
            % (xml_path,)
        )
    for waypoint in waypoint_list:
        # Get the relevant information about the waypoint
        waypoint_id = int(waypoint.get('id'))
        waypoint_lat = float(waypoint.get('y'))
        waypoint_lon = float(waypoint.get('x'))
        waypoint_time = waypoint.get('time')
        waypoint_info = [
            xml_name,
            waypoint_id,
            (waypoint_lat, waypoint_lon),
            waypoint_time,
        ]
        if None in waypoint_info:
            raise IOError(
                'The observation (waypoint) is missing information: %r' % (waypoint_info,)
            )
        # Get all of the waypoint's observations (we expect only one
        # normally)
        element = './/%sobservations' % (namespace,)
        observation_list = waypoint.findall(element)
        # if len(observation_list) == 0:
        #     raise IOError('There are no observations in this waypoint,
        #     waypoint_id: %r' % (waypoint_id, ))
        for observation in observation_list:
            # Filter the observations based on type, we only care
            # about certain types
            categoryKey = observation.attrib['categoryKey']
            if categoryKey.startswith('animals.liveanimals') or categoryKey.startswith(
                'animals.problemanimal'
            ):
                # Get the photonumber attribute for the waypoint's
                # observation
                element = './/%sattributes[@attributeKey="photonumber"]' % (namespace,)
                photonumber = observation.find(element)
                if photonumber is not None:
                    element = './/%ssValue' % (namespace,)
                    # Get the value for photonumber
                    sValue = photonumber.find(element)
                    if sValue is None:
                        raise IOError(
                            (
                                'The photonumber sValue is missing from '
                                'photonumber, waypoint_id: %r'
                            )
                            % (waypoint_id,)
                        )
                    # Python cast the value
                    try:
                        photo_number = int(float(sValue.text)) - offset
                    except ValueError:
                        # raise IOError('The photonumber sValue is invalid,
                        # waypoint_id: %r' % (waypoint_id, ))
                        print(
                            (
                                '[ibs]     '
                                'Skipped Invalid Observation with '
                                'photonumber: %r, waypoint_id: %r'
                            )
                            % (sValue.text, waypoint_id,)
                        )
                        continue
                    # Check that the photo_number is within the acceptable bounds
                    if photo_number >= nTotal:
                        raise IOError(
                            'The Patrol XML file is looking for images '
                            'that do not exist (too few images given)'
                        )
                    # Keep track of the last waypoint that was processed
                    # becuase we only have photono, which indicates start
                    # indices and doesn't specify the end index.  The
                    # ending index is extracted as the next waypoint's
                    # photonum minus 1.
                    if last_photo_number is not None and last_imageset_info is not None:
                        imageset_info = last_imageset_info + [
                            (last_photo_number, photo_number)
                        ]
                        imageset_info_list.append(imageset_info)
                    last_photo_number = photo_number
                    last_imageset_info = waypoint_info
                else:
                    # raise IOError('The photonumber value is missing from
                    # waypoint, waypoint_id: %r' % (waypoint_id, ))
                    print(
                        (
                            '[ibs]     Skipped Empty Observation with'
                            '"categoryKey": %r, waypoint_id: %r'
                        )
                        % (categoryKey, waypoint_id,)
                    )
            else:
                print(
                    (
                        '[ibs]     '
                        'Skipped Incompatible Observation with '
                        '"categoryKey": %r, waypoint_id: %r'
                    )
                    % (categoryKey, waypoint_id,)
                )
    # Append the last photo_number
    if last_photo_number is not None and last_imageset_info is not None:
        imageset_info = last_imageset_info + [(last_photo_number, nTotal)]
        imageset_info_list.append(imageset_info)
    return imageset_info_list


@register_ibs_method
def compute_occurrences_smart(ibs, gid_list, smart_xml_fpath):
    """ Function to load and process a SMART patrol XML file """
    # Get file and copy to wbia database folder
    xml_dir, xml_name = split(smart_xml_fpath)
    dst_xml_path = join(ibs.get_smart_patrol_dir(), xml_name)
    ut.copy(smart_xml_fpath, dst_xml_path, overwrite=True)
    # Process the XML File
    print('[ibs] Processing Patrol XML file: %r' % (dst_xml_path,))
    try:
        imageset_info_list = ibs._parse_smart_xml(dst_xml_path, len(gid_list))
    except Exception as e:
        ibs.delete_images(gid_list)
        print(
            (
                '[ibs] ERROR: Parsing Patrol XML file failed, '
                'rolling back by deleting %d images...'
            )
            % (len(gid_list,))
        )
        raise e
    if len(gid_list) > 0:
        # Sanity check
        assert len(imageset_info_list) > 0, (
            'Trying to added %d images, but the Patrol  ' 'XML file has no observations'
        ) % (len(gid_list),)
    # Display the patrol imagesets
    for index, imageset_info in enumerate(imageset_info_list):
        smart_xml_fname, smart_waypoint_id, gps, local_time, range_ = imageset_info
        start, end = range_
        gid_list_ = gid_list[start:end]
        print('[ibs]     Found Patrol ImageSet: %r' % (imageset_info,))
        print('[ibs]         GIDs: %r' % (gid_list_,))
        if len(gid_list_) == 0:
            print('[ibs]         SKIPPING EMPTY IMAGESET')
            continue
        # Add the GPS data to the images
        gps_list = [gps] * len(gid_list_)
        ibs.set_image_gps(gid_list_, gps_list)
        # Create a new imageset
        imagesettext = '%s Waypoint %03d' % (xml_name.replace('.xml', ''), index + 1,)
        imgsetid = ibs.add_imagesets(imagesettext)
        # Add images to the imagesets
        imgsetid_list = [imgsetid] * len(gid_list_)
        ibs.set_image_imgsetids(gid_list_, imgsetid_list)
        # Set the imageset's smart fields
        ibs.set_imageset_smart_xml_fnames([imgsetid], [smart_xml_fname])
        ibs.set_imageset_smart_waypoint_ids([imgsetid], [smart_waypoint_id])
        # Set the imageset's time based on the images
        unixtime_list = sorted(ibs.get_image_unixtime(gid_list_))
        start_time = unixtime_list[0]
        end_time = unixtime_list[-1]
        ibs.set_imageset_start_time_posix([imgsetid], [start_time])
        ibs.set_imageset_end_time_posix([imgsetid], [end_time])
    # Complete
    print('[ibs] ...Done processing Patrol XML file')


@register_ibs_method
def compute_occurrences(ibs, config=None):
    """
    Clusters ungrouped images into imagesets representing occurrences

    CommandLine:
        python -m wbia.control.IBEISControl --test-compute_occurrences

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.IBEISControl import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> ibs.update_special_imagesets()
        >>> # Remove some images from a non-special imageset
        >>> nonspecial_imgsetids = [i for i in ibs.get_valid_imgsetids() if i not in ibs.get_special_imgsetids()]
        >>> images_to_remove = ibs.get_imageset_gids(nonspecial_imgsetids[0:1])[0][0:1]
        >>> ibs.unrelate_images_and_imagesets(images_to_remove,nonspecial_imgsetids[0:1] * len(images_to_remove))
        >>> ibs.update_special_imagesets()
        >>> ungr_imgsetid = ibs.get_imageset_imgsetids_from_text(const.UNGROUPED_IMAGES_IMAGESETTEXT)
        >>> ungr_gids = ibs.get_imageset_gids([ungr_imgsetid])[0]
        >>> #Now let's make sure that when we recompute imagesets, our non-special imgsetid remains the same
        >>> print('PRE COMPUTE: ImageSets are %r' % ibs.get_valid_imgsetids())
        >>> print('Containing: %r' % ibs.get_imageset_gids(ibs.get_valid_imgsetids()))
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> print('COMPUTE: New imagesets are %r' % ibs.get_valid_imgsetids())
        >>> print('Containing: %r' % ibs.get_imageset_gids(ibs.get_valid_imgsetids()))
        >>> ibs.update_special_imagesets()
        >>> print('UPDATE SPECIAL: New imagesets are %r' % ibs.get_valid_imgsetids())
        >>> print('Containing: %r' % ibs.get_imageset_gids(ibs.get_valid_imgsetids()))
        >>> assert(images_to_remove[0] not in ibs.get_imageset_gids(nonspecial_imgsetids[0:1])[0])
    """
    from wbia.algo.preproc import preproc_occurrence

    print('[ibs] Computing and adding imagesets.')
    # Only ungrouped images are clustered
    gid_list = ibs.get_ungrouped_gids()
    # gid_list = ibs.get_valid_gids(require_unixtime=False, reviewed=False)
    with ut.Timer('computing imagesets'):
        flat_imgsetids, flat_gids = preproc_occurrence.wbia_compute_occurrences(
            ibs, gid_list, config=config
        )
        sortx = ut.argsort(flat_imgsetids)
        flat_imgsetids = ut.take(flat_imgsetids, sortx)
        flat_gids = ut.take(flat_gids, sortx)

    valid_imgsetids = ibs.get_valid_imgsetids()
    imgsetid_offset = 0 if len(valid_imgsetids) == 0 else max(valid_imgsetids)

    # This way we can make sure that manually separated imagesets
    # remain untouched, and ensure that new imagesets are created
    flat_imgsetids_offset = [imgsetid + imgsetid_offset for imgsetid in flat_imgsetids]
    imagesettext_list = [
        'Occurrence ' + str(imgsetid) for imgsetid in flat_imgsetids_offset
    ]
    print('[ibs] Finished computing, about to add imageset.')
    ibs.set_image_imagesettext(flat_gids, imagesettext_list)
    # HACK TO UPDATE IMAGESET POSIX TIMES
    # CAREFUL THIS BLOWS AWAY SMART DATA
    ibs.update_imageset_info(ibs.get_valid_imgsetids())
    print('[ibs] Finished computing and adding imagesets.')


@register_ibs_method
def compute_ggr_path_dict(ibs):
    from matplotlib.path import Path
    import shapefile

    path_dict = {}

    # ADD ZONES
    point_dict = {
        1: [-0.829843, 35.732721],
        2: [-0.829843, 37.165353],
        3: [-0.829843, 38.566150],
        4: [0.405015, 37.165353],
        5: [0.405015, 38.566150],
        6: [1.292767, 35.732721],
        7: [1.292767, 36.701444],
        8: [1.292767, 37.029463],
        9: [1.292767, 37.415937],
        10: [1.292767, 38.566150],
        11: [2.641838, 35.732721],
        12: [2.641838, 37.029463],
        13: [2.641838, 37.415937],
        14: [2.641838, 38.566150],
    }

    zone_dict = {
        '1': [1, 2, 4, 7, 6],
        '2': [2, 3, 5, 4],
        '3': [4, 5, 10, 7],
        '4': [6, 8, 12, 11],
        '5': [8, 9, 13, 12],
        '6': [9, 10, 14, 13],
        'North': [6, 10, 14, 11],
        'Core': [1, 3, 14, 11],
    }

    for zone in zone_dict:
        point_list = [point_dict[vertex] for vertex in zone_dict[zone]]
        name = 'Zone %s' % (zone,)
        path_dict[name] = Path(np.array(point_list))

    # ADD COUNTIES
    name_list = [
        'Laikipia',
        'Samburu',
        'Isiolo',
        'Marsabit',
        'Meru',
    ]
    county_file_url = 'https://wildbookiarepository.azureedge.net/data/kenyan_counties_boundary_gps_coordinates.zip'
    unzipped_path = ut.grab_zipped_url(county_file_url)
    county_path = join(unzipped_path, 'County')
    counties = shapefile.Reader(county_path)
    for record, shape in zip(counties.records(), counties.shapes()):
        name = record[5]
        if name not in name_list:
            continue
        point_list = shape.points
        point_list = [list(point)[::-1] for point in point_list]
        name = 'County %s' % (name,)
        path_dict[name] = Path(np.array(point_list))

    # ADD LAND TENURES
    land_tenure_file_url = 'https://wildbookiarepository.azureedge.net/data/kenyan_land_tenures_boundary_gps_coordinates.zip'
    unzipped_path = ut.grab_zipped_url(land_tenure_file_url)
    land_tenure_path = join(unzipped_path, 'LandTenure')
    land_tenures = shapefile.Reader(land_tenure_path)
    for record, shape in zip(land_tenures.records(), land_tenures.shapes()):
        name = record[0]
        if len(name) == 0:
            continue
        point_list = shape.points
        point_list = [list(point)[::-1] for point in point_list]
        name = 'Land Tenure %s' % (name,)
        path_dict[name] = Path(np.array(point_list))

    return path_dict


@register_ibs_method
def compute_ggr_imagesets(
    ibs, gid_list=None, min_diff=86400, individual=False, purge_all_old=False
):

    if purge_all_old:
        imageset_rowid_list_all = ibs.get_valid_imgsetids()
        imageset_text_list_all = ibs.get_imageset_text(imageset_rowid_list_all)
        zipped = list(zip(imageset_rowid_list_all, imageset_text_list_all))
        imageset_rowid_list_delete = [
            imageset_rowid_all
            for imageset_rowid_all, imageset_text_all in zipped
            if 'GGR Special' in imageset_text_all
        ]
        for imageset_rowid_delete in imageset_rowid_list_delete:
            ibs.delete_gsgr_imageset_relations(imageset_rowid_delete)
        ibs.delete_imagesets(imageset_rowid_list_delete)

    # GET DATA
    path_dict = ibs.compute_ggr_path_dict()
    zone_list = sorted(path_dict.keys()) + ['Zone 7']
    imageset_dict = {zone: [] for zone in zone_list}

    if gid_list is None:
        gid_list = ibs.get_valid_gids()

    gps_list = ibs.get_image_gps(gid_list)
    note_list = ibs.get_image_notes(gid_list)
    temp = -1 if individual else -2
    note_list_ = [','.join(note.strip().split(',')[:temp]) for note in note_list]
    note_list = [','.join(note.strip().split(',')[:-1]) for note in note_list]

    special_zone_map = {  # NOQA
        'GGR,3,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,8,A': 'Zone 3,County Isiolo,Zone Core',
        'GGR,10,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,13,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,14,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,15,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,19,A': 'Zone 2,County Samburu,Zone Core',
        'GGR,23,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,24,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,25,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,27,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,29,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,37,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,37,B': 'Zone 1,County Laikipia,Zone Core',
        'GGR,38,C': 'Zone 1,County Laikipia,Zone Core',
        'GGR,40,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,41,B': 'Zone 1,County Laikipia,Zone Core',
        'GGR,44,A': None,
        'GGR,45,A': 'Zone 2,County Isiolo,Zone Core',
        'GGR,46,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,62,B': 'Zone 1,County Laikipia,Zone Core',
        'GGR,86,A': 'Zone 3,County Isiolo,Zone Core',
        'GGR,96,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,97,B': 'Zone 2,County Samburu,Zone Core',
        'GGR,108,A': 'Zone 1,County Laikipia,Zone Core',
        'GGR,118,C': 'Zone 6,Zone North,County Marsabit,Zone Core',
        'GGR2,8,D': 'Zone 1,County Laikipia,Zone Core,Land Tenure Mpala',
        'GGR2,39,A': 'Zone 1,County Laikipia,Zone Core,Land Tenure Colcheccio - Franscombe',
        'GGR2,40,A': 'Zone 2,County Samburu,Zone Core,Land Tenure Kalama',
        'GGR2,54,B': 'Zone 3,County Isiolo,Zone Core,Land Tenure Nasuulu',
        'GGR2,92,D': 'Zone 2,County Samburu,Zone Core,Land Tenure Westgate',
        'GGR2,94,C': 'Zone 1,County Laikipia,Zone Core,Land Tenure Mukogodo',
        'GGR2,103,A': None,
        'GGR2,106,A': None,
        'GGR2,107,A': None,
        'GGR2,107,B': None,
        'GGR2,126,B': 'Zone 1,County Laikipia,Zone Core',
        'GGR2,126,C': 'Zone 1,County Laikipia,Zone Core',
        'GGR2,137,B': 'Zone 2,County Samburu,Zone Core,Land Tenure Sera',
        'GGR2,160,E': 'Zone 1,County Laikipia,Zone Core,Land Tenure Mpala',
        'GGR2,192,A': 'County Laikipia,Zone Core,Land Tenure Melako',
        'GGR2,200,B': 'Zone 1,County Laikipia,Zone Core,Land Tenure Mpala',
        'GGR2,200,F': 'Zone 1,County Laikipia,Zone Core,Land Tenure Mpala',
        'GGR2,201,E': 'Zone 1,County Laikipia,Zone Core,Land Tenure Mpala',
        'GGR2,201,F': 'Zone 1,County Laikipia,Zone Core,Land Tenure Mpala',
        'GGR2,210,A': 'Zone 1,County Laikipia,Zone Core,Land Tenure Mugie',
        'GGR2,220,A': None,
        'GGR2,222,B': 'Zone 1,County Laikipia,Zone Core,Land Tenure Ol Jogi',
        'GGR2,224,A': 'Zone 1,County Laikipia,Zone Core,Land Tenure Mpala',
        'GGR2,224,B': 'Zone 1,County Laikipia,Zone Core,Land Tenure Mpala',
        'GGR2,225,A': 'Zone 1,County Laikipia,Zone Core,Land Tenure Ol Jogi',
        'GGR2,230,A': 'Zone 1,County Laikipia,Zone Core,Land Tenure Elkarama',
        'GGR2,230,C': 'Zone 1,County Laikipia,Zone Core,Land Tenure Elkarama',
        'GGR2,231,B': None,
        'GGR2,232,B': None,
    }

    skipped_gid_list = []
    skipped_note_list = []
    skipped = 0
    zipped = list(enumerate(zip(gid_list, gps_list)))
    for index, (gid, point) in ut.ProgIter(zipped, lbl='assigning zones'):
        if point == (-1, -1):
            unixtime = ibs.get_image_unixtime(gid)
            note = note_list_[index]

            # Find siblings in the same car
            sibling_gid_list = []
            for gid_, note_ in zip(gid_list, note_list_):
                if note_ == note:
                    sibling_gid_list.append(gid_)

            # Get valid GPS
            gps_list = ibs.get_image_gps(sibling_gid_list)
            flag_list = [gps != (-1, -1) for gps in gps_list]
            gid_list_ = ut.compress(sibling_gid_list, flag_list)

            # If found, get closest image
            if len(gid_list_) > 0:
                gps_list_ = ibs.get_image_gps(gid_list_)
                unixtime_list_ = ibs.get_image_unixtime(gid_list_)
                # Find closest
                closest_diff, closest_gps = np.inf, None
                for unixtime_, gps_ in zip(unixtime_list_, gps_list_):
                    diff = abs(unixtime - unixtime_)
                    if diff < closest_diff and gps_ != (-1, -1):
                        closest_diff = diff
                        closest_gps = gps_
                # Assign closest
                if closest_gps is not None and closest_diff <= min_diff:
                    point = closest_gps

        # if point == (-1, -1):
        #     note = note_list[index]
        #     if note in special_zone_map:
        #         zone_str = special_zone_map[note]
        #         if zone_str is not None:
        #             zone_list = zone_str.strip().split(',')
        #             for zone in zone_list:
        #                 imageset_dict[zone].append(gid)
        #             continue

        if point == (-1, -1):
            skipped_gid_list.append(gid)
            skipped_note_list.append(note)
            skipped += 1
            continue

        found = False
        for zone in sorted(path_dict.keys()):
            path = path_dict[zone]
            if path.contains_point(point):
                found = True  # NOQA
                imageset_dict[zone].append(gid)
        # if not found:
        #     imageset_dict['Zone 7'].append(gid)

    imageset_id_list = []
    for zone, gid_list in sorted(imageset_dict.items()):
        imageset_str = 'GGR Special Zone - %s' % (zone,)
        imageset_id = ibs.add_imagesets(imageset_str)
        imageset_id_list.append(imageset_id)
        args = (
            imageset_str,
            imageset_id,
            len(gid_list),
        )
        print('Creating new GGR imageset: %r (ID %d) with %d images' % args)
        ibs.delete_gsgr_imageset_relations(imageset_id)
        ibs.set_image_imgsetids(gid_list, [imageset_id] * len(gid_list))

    print('SKIPPED %d IMAGES' % (skipped,))
    skipped_note_list = sorted(list(set(skipped_note_list)))
    print('skipped_note_list = %r' % (skipped_note_list,))
    print('skipped_gid_list = %r' % (skipped_gid_list,))

    return imageset_id_list, skipped_note_list, skipped_gid_list


@register_ibs_method
def compute_ggr_fix_gps_names(ibs, min_diff=1800):  # 86,400 = 60 sec x 60 min X 24 hours
    # Get all aids
    aid_list = ibs.get_valid_aids()
    num_all = len(aid_list)
    gps_list = ibs.get_annot_image_gps(aid_list)
    flag_list = [gps == (-1, -1) for gps in gps_list]
    # Get bad GPS aids
    aid_list = ut.filter_items(aid_list, flag_list)
    num_bad = len(aid_list)
    nid_list = ibs.get_annot_name_rowids(aid_list)
    flag_list = [nid != const.UNKNOWN_NAME_ROWID for nid in nid_list]
    # Get KNOWN and bad GPS aids
    aid_list = ut.filter_items(aid_list, flag_list)
    num_known = len(aid_list)
    # Find close GPS
    num_found = 0
    recovered_aid_list = []
    recovered_gps_list = []
    recovered_dist_list = []
    for aid in aid_list:
        # Get annotation information
        unixtime = ibs.get_annot_image_unixtimes(aid)
        nid = ibs.get_annot_name_rowids(aid)
        # Get other sightings
        aid_list_ = ibs.get_name_aids(nid)
        aid_list_.remove(aid)
        unixtime_list = ibs.get_annot_image_unixtimes(aid_list_)
        gps_list = ibs.get_annot_image_gps(aid_list_)
        # Find closest
        closest_diff, closest_gps = np.inf, None
        for unixtime_, gps_ in zip(unixtime_list, gps_list):
            diff = abs(unixtime - unixtime_)
            if diff < closest_diff and gps_ != (-1, -1):
                closest_diff = diff
                closest_gps = gps_
        # Assign closest
        if closest_gps is not None and closest_diff <= min_diff:
            recovered_aid_list.append(aid)
            recovered_gps_list.append(closest_gps)
            recovered_dist_list.append(closest_diff)
            num_found += 1
            h = closest_diff // 3600
            closest_diff %= 3600
            m = closest_diff // 60
            closest_diff %= 60
            s = closest_diff
            print('FOUND LOCATION FOR AID %d' % (aid,))
            print('\tDIFF   : %d H, %d M, %d S' % (h, m, s,))
            print('\tNEW GPS: %s' % (closest_gps,))
    print(r'%d \ %d \ %d \ %d' % (num_all, num_bad, num_known, num_found,))
    return recovered_aid_list, recovered_gps_list, recovered_dist_list


@register_ibs_method
def parse_ggr_name(
    ibs, imageset_text, verbose=False, allow_short=False, require_short=False
):
    imageset_text = imageset_text.strip()

    if verbose:
        print('Processing %r' % (imageset_text,))

    imageset_text_ = imageset_text.split(',')

    valid_lengths = [3]
    if allow_short:
        valid_lengths += [2]

    if require_short:
        valid_lengths = [2]

    if len(imageset_text_) not in valid_lengths:
        return None

    try:
        dataset, number, letter = imageset_text_
    except Exception:
        assert allow_short or require_short
        dataset, number = imageset_text_
        letter = None

    if dataset != 'GGR2':
        return None

    number = int(number)
    if letter not in ['A', 'B', 'C', 'D', 'E', 'F', None]:
        return None

    if verbose:
        print('\tDataset: %r' % (dataset,))
        print('\tLetter : %r' % (letter,))
        print('\tNumber : %r' % (number,))

    return dataset, letter, number


def search_ggr_qr_codes_worker(
    imageset_rowid, imageset_text, values, gid_list, filepath_list, note_list, timeout
):
    import pyzbar.pyzbar as pyzbar
    import cv2

    if values is None:
        print(imageset_text)
    assert values is not None
    dataset, letter, number = values

    ret_list = []
    match = False
    for index, (gid, filepath, note) in enumerate(
        zip(gid_list, filepath_list, note_list)
    ):
        if timeout is not None and index > timeout:
            print('\tTimeout exceeded')
            break

        if match:
            print('\tMatch was found')
            break

        print('\tProcessing %r (%s)' % (filepath, note,))

        image = cv2.imread(filepath, 0)
        qr_list = pyzbar.decode(image, [pyzbar.ZBarSymbol.QRCODE])

        if len(qr_list) > 0:
            print('\t\tFound...')
            qr = qr_list[0]
            data = qr.data.decode('utf-8')

            try:
                data = data.split('/')[-1].strip('?')
                data = data.split('&')
                data = sorted(data)
                print('\t\t%r' % (data,))

                assert data[0] == 'car=%d' % (number,)
                assert data[1] == 'event=ggr2018'
                assert data[2] == 'person=%s' % (letter.lower(),)

                match = True
                print('\t\tPassed!')
            except Exception:
                pass
                print('\t\tFailed!')

            ret = (
                imageset_text,
                gid,
                match,
                data,
            )
            ret_list.append(ret)

    return imageset_rowid, ret_list


@register_ibs_method
def search_ggr_qr_codes(ibs, imageset_rowid_list=None, timeout=None, **kwargs):
    r"""
    Search for QR codes in each imageset.

    Args:
        ibs (IBEISController):  wbia controller object
        imageset_rowid_list (list):  imageset rowid list

    CommandLine:
        python -m wbia.other.ibsfuncs search_ggr_qr_codes

    Reference:
        https://www.learnopencv.com/barcode-and-qr-code-scanner-using-zbar-and-opencv/

        macOS:
            brew install zbar

            or

            curl -O https://ayera.dl.sourceforge.net/project/zbar/zbar/0.10/zbar-0.10.tar.bz2
            tar -xvjf zbar-0.10.tar.bz2
            cd zbar-0.10/
            CPPFLAGS="-I/opt/local/include" LDFLAGS="-L/opt/local/lib" ./configure --disable-video --without-qt --without-python --without-gtk --with-libiconv-prefix=/opt/local --with-jpeg=yes --prefix=$VIRTUAL_ENV
            make
            make install
            sudo ln $VIRTUAL_ENV/lib/libzbar.dylib /opt/local/lib/libzbar.dylib
            sudo ln $VIRTUAL_ENV/include/zbar.h /opt/local/include/zbar.h

        Ubuntu:
            sudo apt-get install libzbar-dev libzbar0

        pip install pyzbar

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> default_dbdir = join('/', 'data', 'wbia', 'GGR2-IBEIS')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> ibs.search_ggr_qr_codes()
    """
    if imageset_rowid_list is None:
        ibs.delete_empty_imgsetids()
        imageset_rowid_list = ibs.get_valid_imgsetids(is_special=False)

    imageset_text_list = ibs.get_imageset_text(imageset_rowid_list)
    values_list = [
        ibs.parse_ggr_name(imageset_text) for imageset_text in imageset_text_list
    ]
    gids_list = [
        sorted(ibs.get_imageset_gids(imageset_rowid))
        for imageset_rowid in imageset_rowid_list
    ]
    filepaths_list = [ibs.get_image_paths(gid_list) for gid_list in gids_list]
    notes_list = [ibs.get_image_notes(gid_list) for gid_list in gids_list]
    timeouts_list = [timeout] * len(imageset_rowid_list)

    arg_iter = list(
        zip(
            imageset_rowid_list,
            imageset_text_list,
            values_list,
            gids_list,
            filepaths_list,
            notes_list,
            timeouts_list,
        )
    )
    result_list = ut.util_parallel.generate2(search_ggr_qr_codes_worker, arg_iter)
    result_list = list(result_list)

    imageset_dict = {}
    for imageset_rowid, qr_list in result_list:
        imageset_dict[imageset_rowid] = qr_list

    assert len(list(imageset_dict.keys())) == len(imageset_rowid_list)
    return imageset_dict


@register_ibs_method
def fix_ggr_qr_codes(ibs, imageset_qr_dict):

    qr_fix_dict = {
        '5B': 1179,
        '6A': 1359,
        '8B': 2886,
        '8C': 3072,
        '8D': 3843,
        '9A': 4128,
        '14B': 4599,
        '21A': 5027,
        '21B': 5228,
        '25A': 6291,
        '26A': 6467,
        '26B': 6655,
        '27B': 8587,  # Three separate GPS records, using most prevalent
        '33B': 10168,
        '40A': 11189,
        '42A': 13350,
        '45B': 14338,
        '45D': 14670,
        '45E': 15217,
        '49A': 16483,
        '54A': 16815,
        '54B': 18018,
        '56A': 18204,
        '59A': 18369,
        '63B': 19465,
        '76A': 21858,
        '76B': 22233,
        '76C': 22410,
        '78A': 22734,
        '82D': 24683,
        '85B': 25746,
        '87A': 26274,
        '90A': 27221,
        '91A': 27287,
        '92C': 28700,
        '92D': 29216,
        '94A': 29632,
        '94B': 30659,
        '95A': 31961,
        '100A': 32224,
        '100B': 32615,
        '100C': 33034,
        '100E': 33688,
        '108B': 34524,
        '114A': 34963,
        '115A': 34969,
        '116A': 35569,
        '122A': 35737,
        '122B': 36134,
        '126B': 37333,
        '126D': 37791,
        '130A': 37877,
        '133A': 38113,
        '136A': 38184,
        '136B': 38462,
        '137A': 38548,
        '137B': 38559,
        '137C': 38831,
        '138A': 38919,
        '138B': 39124,
        '149A': 41079,
        '155A': 41886,
        '159A': 43129,
        '160E': 46284,
        '160F': 46823,
        '163B': 49228,
        '163C': 49544,
        '164A': 49730,
        '169C': 50387,
        '189A': 50961,
        '190A': 51382,
        '191A': 51626,
        '192A': 51843,
        '201E': 52494,
        '202B': 52525,
        '222A': 52907,
        '222B': 54182,
        '223B': 55114,
        '225A': 55968,
        '226A': 56005,
    }

    for ggr_name in qr_fix_dict:
        number = ggr_name[:-1]
        letter = ggr_name[-1].upper()
        qr_gid = qr_fix_dict[ggr_name]

        number = int(number)
        assert letter in ['A', 'B', 'C', 'D', 'E', 'F']

        imageset_name = 'GGR2,%d,%s' % (number, letter,)
        imageset_id = ibs.get_imageset_imgsetids_from_text(imageset_name)
        gid_list = ibs.get_imageset_gids(imageset_id)
        assert qr_gid in gid_list

        assert imageset_id in imageset_qr_dict
        imageset_list = imageset_qr_dict[imageset_id]

        tag = 'GGR2,%d,%s' % (number, letter)
        imageset_list_ = []
        for imageset in imageset_list:
            if tag == imageset[0]:
                continue
            imageset_list_.append(imageset)

        imageset = [
            imageset_name,
            qr_gid,
            True,
            ['car=%s' % (number,), 'event=ggr2018', 'person=%s' % (letter.lower(),)],
        ]
        imageset_list_.append(imageset)

        imageset_qr_dict[imageset_id] = imageset_list_

    return imageset_qr_dict


@register_ibs_method
def inspect_ggr_qr_codes(ibs, *args, **kwargs):
    r"""
    Inspect QR codes in each imageset.

    Args:
        ibs (IBEISController):  wbia controller object
        imageset_rowid_list (list):  imageset rowid list

    CommandLine:
        python -m wbia.other.ibsfuncs inspect_ggr_qr_codes

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> default_dbdir = join('/', 'data', 'wbia', 'GGR2-IBEIS')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> ibs.inspect_ggr_qr_codes()
    """
    filename_qr_json = join(ibs.dbdir, 'imageset_qr_dict.json')

    if not exists(filename_qr_json):
        imageset_dict = ibs.search_ggr_qr_codes(*args, **kwargs)
        assert not exists(filename_qr_json)
        ut.save_json(filename_qr_json, imageset_dict)

    imageset_qr_dict = ut.load_json(filename_qr_json)

    for key in list(imageset_qr_dict.keys()):
        imageset_qr_dict[int(key)] = imageset_qr_dict.pop(key)

    imageset_qr_dict = ibs.fix_ggr_qr_codes(imageset_qr_dict)

    ggr_qr_dict = {}
    for imageset_rowid in imageset_qr_dict:
        imageset_text = ibs.get_imageset_text(imageset_rowid)
        values = ibs.parse_ggr_name(imageset_text)
        assert values is not None
        dataset, letter, number = values

        if number not in ggr_qr_dict:
            ggr_qr_dict[number] = {}

        assert letter not in ggr_qr_dict[number]
        ggr_qr_dict[number][letter] = (
            imageset_rowid,
            imageset_qr_dict[imageset_rowid],
        )

    cleared_imageset_rowid_list = [
        2,  # NO QR AT ALL
        63,  # INDIVIDUAL WITH NO QR (BUT HAS GPS)
        77,  # INDIVIDUAL WITH NO QR (BUT HAS GPS)
        107,  # NO QR AT ALL
        185,  # INDIVIDUAL WITH NO QR (BUT HAS GPS)
        216,  # NO QR AT ALL
        217,  # NO QR AT ALL
    ]

    sync_dict = {}

    # Find all others and run checks
    for number in sorted(list(ggr_qr_dict.keys())):
        qr_dict = ggr_qr_dict[number]
        letter_list = sorted(list(qr_dict.keys()))
        num_letters = len(letter_list)
        if num_letters == 0:
            print('Empty car: %r' % (number,))
            break
        elif num_letters == 1:
            letter = letter_list[0]
            imageset_rowid, qr_list = qr_dict[letter]

            match_gid = None

            if letter != 'A':
                print('Individual car missing A: %r (%r)' % (number, letter,))
                match_gid = None

            if len(qr_list) == 0:
                print(
                    'Individual car missing QR: %r %r (imageset_rowid = %r)'
                    % (number, letter, imageset_rowid,)
                )
            else:
                for qr in qr_list:
                    if qr[2]:
                        match_gid = qr[1]
                        break

                if match_gid in [None, -1]:
                    print(
                        'Individual car incorrect QR: %r %r (imageset_rowid = %r)'
                        % (number, letter, imageset_rowid,)
                    )
                    print('\t%r' % (qr_list,))

            if imageset_rowid in cleared_imageset_rowid_list:
                print('\tCleared Imageset: %d %r' % (number, letter,))

            sync_dict[imageset_rowid] = match_gid
        else:
            failed_list = []
            missing_list = []
            for letter in letter_list:
                imageset_rowid, qr_list = qr_dict[letter]

                match_gid = None

                if imageset_rowid in sync_dict:
                    if sync_dict[imageset_rowid] is not None:
                        continue

                if len(qr_list) == 0:
                    missing_list.append((letter, imageset_rowid))
                else:
                    for qr in qr_list:
                        if qr[2]:
                            match_gid = qr[1]
                            break

                    if match_gid is None:
                        failed_list.append((letter, imageset_rowid))

                if imageset_rowid in cleared_imageset_rowid_list:
                    print('\tCleared Imageset: %d %r' % (number, letter,))

                sync_dict[imageset_rowid] = match_gid

            if len(missing_list) > 0:
                print('Group car missing QR: %r (%r)' % (number, letter_list,))
                for missing, imageset_rowid in missing_list:
                    print(
                        '\tNo QR for %r %r (imageset_rowid = %r)'
                        % (number, missing, imageset_rowid,)
                    )

            if len(failed_list) > 0:
                print('Group car incorrect QR: %r (%r)' % (number, letter_list,))
                for failed, imageset_rowid in failed_list:
                    print(
                        '\tBad QR for %r %r (imageset_rowid = %r)'
                        % (number, failed, imageset_rowid,)
                    )

    filename_qr_json = join(ibs.dbdir, 'imageset_qr_dict.final.json')
    ut.save_json(filename_qr_json, imageset_qr_dict)

    return sync_dict


@register_ibs_method
def overwrite_ggr_unixtimes_from_gps(ibs, gmt_offset=3.0, *args, **kwargs):
    r"""
    Sync image time offsets using QR codes sync data

    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.ibsfuncs overwrite_ggr_unixtimes_from_gps

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> default_dbdir = join('/', 'data', 'wbia', 'GGR2-IBEIS')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> ibs.overwrite_ggr_unixtimes_from_gps()
    """
    sync_dict = ibs.inspect_ggr_qr_codes(*args, **kwargs)
    imageset_rowid_list = sorted(sync_dict.keys())

    car_dict = {}
    for imageset_rowid in imageset_rowid_list:
        imageset_text = ibs.get_imageset_text(imageset_rowid)
        if imageset_text is None:
            continue
        values = ibs.parse_ggr_name(imageset_text)
        assert values is not None
        dataset, letter, number = values

        qr_gid = sync_dict[imageset_rowid]
        if letter == 'A':
            if qr_gid is not None:
                car_dict[number] = qr_gid

            gid_list = ibs.get_imageset_gids(imageset_rowid)
            count = overwrite_unixtimes_from_gps(ibs, gid_list, gmt_offset=gmt_offset)

            if count > 0:
                print('Overwrote %d image unixtimes from GPS for %r' % (count, values))


def overwrite_unixtimes_from_gps_worker(path):
    from vtool.exif import parse_exif_unixtime_gps

    unixtime_gps = parse_exif_unixtime_gps(path)
    return unixtime_gps


@register_ibs_method
def overwrite_unixtimes_from_gps(ibs, gid_list, gmt_offset=3.0):
    r"""
    Sync image time offsets using QR codes sync data

    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.ibsfuncs overwrite_unixtimes_from_gps

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> default_dbdir = join('/', 'data', 'wbia', 'GGR2-IBEIS')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> ibs.overwrite_unixtimes_from_gps()
    """
    # Check for GPS dates and use if available
    path_list = ibs.get_image_paths(gid_list)
    arg_iter = list(zip(path_list))
    unixtime_gps_list = list(
        ut.util_parallel.generate2(
            overwrite_unixtimes_from_gps_worker, arg_iter, ordered=True
        )
    )
    unixtime_list = ibs.get_image_unixtime(gid_list)

    zipped_list = list(zip(gid_list, path_list, unixtime_list, unixtime_gps_list))

    gid_list_ = []
    offset_list = []
    for gid, path, unixtime, unixtime_gps in zipped_list:
        if unixtime_gps != -1:
            unixtime_gps += gmt_offset * 60 * 60

            offset = unixtime_gps - unixtime
            if offset != 0:
                current_offset = ibs.get_image_timedelta_posix([gid])[0]
                offset += current_offset
                gid_list_.append(gid)
                offset_list.append(offset)

    ibs.set_image_timedelta_posix(gid_list_, offset_list)
    return len(gid_list_)


@register_ibs_method
def sync_ggr_with_qr_codes(ibs, local_offset=-8.0, gmt_offset=3.0, *args, **kwargs):
    r"""
    Sync image time offsets using QR codes sync data

    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.ibsfuncs sync_ggr_with_qr_codes

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia  # NOQA
        >>> default_dbdir = join('/', 'data', 'wbia', 'GGR2-IBEIS')
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> ibs.sync_ggr_with_qr_codes()
    """
    import datetime

    lower_posix = ut.datetime_to_posixtime(
        ut.date_to_datetime(datetime.date(2018, 1, 24))
    )
    upper_posix = ut.datetime_to_posixtime(ut.date_to_datetime(datetime.date(2018, 2, 1)))

    lower_posix += local_offset * 60 * 60
    upper_posix += local_offset * 60 * 60

    lower_posix -= gmt_offset * 60 * 60
    upper_posix -= gmt_offset * 60 * 60

    sync_dict = ibs.inspect_ggr_qr_codes(*args, **kwargs)
    imageset_rowid_list = sorted(sync_dict.keys())
    delete_gid_list = []

    car_dict = {}
    for imageset_rowid in imageset_rowid_list:
        imageset_text = ibs.get_imageset_text(imageset_rowid)
        if imageset_text is None:
            continue
        values = ibs.parse_ggr_name(imageset_text)
        assert values is not None
        dataset, letter, number = values

        qr_gid = sync_dict[imageset_rowid]
        if letter == 'A':
            if qr_gid is not None:
                car_dict[number] = qr_gid

            count = 0
            gid_list = ibs.get_imageset_gids(imageset_rowid)
            unixtime_list = ibs.get_image_unixtime(gid_list)
            for gid, unixtime in zip(gid_list, unixtime_list):
                if unixtime is None or unixtime < lower_posix or upper_posix < unixtime:
                    delete_gid_list.append(gid)
                    count += 1

            if count > 0:
                print('Found %d images to delete for %r' % (count, values))

    cleared_imageset_rowid_list = [
        187,  # Images from GGR2 but from 2015 with valid GPS coordinates
        188,  # Images from GGR2 but from 2015 with valid GPS coordinates
        189,  # Images from GGR2 but from 2015 with valid GPS coordinates
    ]
    for imageset_rowid in imageset_rowid_list:
        imageset_text = ibs.get_imageset_text(imageset_rowid)
        if imageset_text is None:
            continue

        values = ibs.parse_ggr_name(imageset_text)
        assert values is not None
        dataset, letter, number = values

        qr_gid = sync_dict[imageset_rowid]
        gid_list = ibs.get_imageset_gids(imageset_rowid)

        if letter == 'A':
            continue

        if qr_gid is None:
            print('Skipping None QR %r' % (values,))
            continue

        assert qr_gid in gid_list

        anchor_gid = car_dict.get(number, None)
        assert anchor_gid != qr_gid

        if anchor_gid is None:
            print('Skipping None Anchor %r' % (values,))
            continue

        qr_time = ibs.get_image_unixtime(qr_gid)
        anchor_time = ibs.get_image_unixtime(anchor_gid)
        offset = anchor_time - qr_time
        if offset != 0:
            current_offset = ibs.get_image_timedelta_posix([qr_gid])[0]
            offset += current_offset
            ibs.set_image_timedelta_posix(gid_list, [offset] * len(gid_list))
            print('Correcting offset for %r: %d' % (values, offset,))

        try:
            qr_time = ibs.get_image_unixtime(qr_gid)
            anchor_time = ibs.get_image_unixtime(anchor_gid)
            offset = anchor_time - qr_time
            assert offset == 0
        except AssertionError:
            print('\tFailed to correct offset for %r: %d ' % (values, offset,))

        if imageset_rowid not in cleared_imageset_rowid_list:
            assert lower_posix <= qr_time and qr_time <= upper_posix
            assert lower_posix <= anchor_time and anchor_time <= upper_posix

        count = 0
        unixtime_list = ibs.get_image_unixtime(gid_list)
        for gid, unixtime in zip(gid_list, unixtime_list):
            if unixtime is None or unixtime < lower_posix or upper_posix < unixtime:
                delete_gid_list.append(gid)

                count += 1

        if count > 0:
            print('Found %d images to delete from %r' % (count, values))

    delete_gid_list = sorted(list(set(delete_gid_list)))
    return delete_gid_list


@register_ibs_method
def query_ggr_gids_between_dates(
    ibs,
    gid_list=None,
    date1=(2018, 1, 27),
    date2=(2018, 1, 29),
    local_offset=-8.0,
    gmt_offset=3.0,
):
    import datetime

    date1y, date1m, date1d = date1
    date2y, date2m, date2d = date2

    lower_posix = ut.datetime_to_posixtime(
        ut.date_to_datetime(datetime.date(date1y, date1m, date1d))
    )
    upper_posix = ut.datetime_to_posixtime(
        ut.date_to_datetime(datetime.date(date2y, date2m, date2d))
    )

    lower_posix += local_offset * 60 * 60
    upper_posix += local_offset * 60 * 60

    lower_posix -= gmt_offset * 60 * 60
    upper_posix -= gmt_offset * 60 * 60

    if gid_list is None:
        gid_list = ibs.get_valid_gids()

    unixtime_list = ibs.get_image_unixtime(gid_list)
    gid_list_ = []
    for gid, unixtime in zip(gid_list, unixtime_list):
        if unixtime is not None and lower_posix <= unixtime and unixtime <= upper_posix:
            gid_list_.append(gid)

    return gid_list_


@register_ibs_method
def purge_ggr_unixtime_out_of_bounds(ibs, *args, **kwargs):
    delete_gid_list = ibs.sync_ggr_with_qr_codes(ibs, *args, **kwargs)
    print('Deleting %d gids' % (len(delete_gid_list),))
    ibs.delete_images(delete_gid_list)
    # ibs.delete_empty_imgsetids()


@register_ibs_method
def compute_ggr_fix_gps_contributors_gids(ibs, min_diff=600, individual=False):
    # Get all gids
    gid_list = ibs.get_valid_gids()
    num_all = len(gid_list)
    gps_list = ibs.get_image_gps(gid_list)
    flag_list = [gps == (-1, -1) for gps in gps_list]
    # Get bad GPS gids
    gid_list = ut.filter_items(gid_list, flag_list)
    num_bad = len(gid_list)

    recovered_gid_list = []
    recovered_gps_list = []
    recovered_dist_list = []

    unrecovered_gid_list = sorted(list(set(gid_list)))
    num_unrecovered = len(unrecovered_gid_list)

    gid_list = ibs.get_valid_gids()
    note_list = ibs.get_image_notes(gid_list)
    temp = -1 if individual else -2
    note_list = [','.join(note.strip().split(',')[:temp]) for note in note_list]

    not_found = set([])
    num_found = 0
    for gid in unrecovered_gid_list:
        unixtime = ibs.get_image_unixtime(gid)
        index = gid_list.index(gid)
        note = note_list[index]

        # Find siblings in the same car
        sibling_gid_list = [
            gid_ for gid_, note_ in zip(gid_list, note_list) if note_ == note
        ]

        # Get valid GPS
        gps_list = ibs.get_image_gps(sibling_gid_list)
        flag_list = [gps != (-1, -1) for gps in gps_list]
        gid_list_ = ut.compress(sibling_gid_list, flag_list)

        # If found, get closest image
        if len(gid_list_) > 0:
            gps_list_ = ibs.get_image_gps(gid_list_)
            unixtime_list_ = ibs.get_image_unixtime(gid_list_)
            # Find closest
            closest_diff, closest_gps = np.inf, None
            for unixtime_, gps_ in zip(unixtime_list_, gps_list_):
                diff = abs(unixtime - unixtime_)
                if diff < closest_diff and gps_ != (-1, -1):
                    closest_diff = diff
                    closest_gps = gps_
            # Assign closest
            if closest_gps is not None and closest_diff <= min_diff:
                recovered_gid_list.append(gid)
                recovered_gps_list.append(closest_gps)
                recovered_dist_list.append(closest_diff)
                num_found += 1
                h = closest_diff // 3600
                closest_diff %= 3600
                m = closest_diff // 60
                closest_diff %= 60
                s = closest_diff
                print('FOUND LOCATION FOR GID %d' % (gid,))
                print('\tDIFF   : %d H, %d M, %d S' % (h, m, s,))
                print('\tNEW GPS: %s' % (closest_gps,))
            else:
                not_found.add(note)
        else:
            not_found.add(note)
    print(r'%d \ %d \ %d \ %d' % (num_all, num_bad, num_unrecovered, num_found,))
    num_recovered = len(recovered_gid_list)
    num_unrecovered = num_bad - len(recovered_gid_list)
    print('Missing GPS: %d' % (num_bad,))
    print('Recovered  : %d' % (num_recovered,))
    print('Unrecovered: %d' % (num_unrecovered,))
    print('Not Found  : %r' % (not_found,))
    return recovered_gid_list, recovered_gps_list, recovered_dist_list


@register_ibs_method
def compute_ggr_fix_gps_contributors_aids(ibs, min_diff=600, individual=False):
    # Get all aids
    aid_list = ibs.get_valid_aids()
    num_all = len(aid_list)
    gps_list = ibs.get_annot_image_gps(aid_list)
    flag_list = [gps == (-1, -1) for gps in gps_list]
    # Get bad GPS aids
    aid_list = ut.filter_items(aid_list, flag_list)
    num_bad = len(aid_list)
    # Get found GPS list via naming
    vals = ibs.compute_ggr_fix_gps_names(min_diff=min_diff)
    recovered_aid_list, recovered_gps_list, recovered_dist_list = vals
    unrecovered_aid_list = list(set(aid_list) - set(recovered_aid_list))
    num_unrecovered = len(unrecovered_aid_list)

    gid_list = ibs.get_valid_gids()
    note_list = ibs.get_image_notes(gid_list)
    temp = -1 if individual else -2
    note_list = [','.join(note.strip().split(',')[:temp]) for note in note_list]

    not_found = set([])
    num_found = 0
    for aid in unrecovered_aid_list:
        gid = ibs.get_annot_gids(aid)
        unixtime = ibs.get_image_unixtime(gid)
        index = gid_list.index(gid)
        note = note_list[index]

        # Find siblings in the same car
        sibling_gid_list = [
            gid_ for gid_, note_ in zip(gid_list, note_list) if note_ == note
        ]

        # Get valid GPS
        gps_list = ibs.get_image_gps(sibling_gid_list)
        flag_list = [gps != (-1, -1) for gps in gps_list]
        gid_list_ = ut.compress(sibling_gid_list, flag_list)

        # If found, get closest image
        if len(gid_list_) > 0:
            gps_list_ = ibs.get_image_gps(gid_list_)
            unixtime_list_ = ibs.get_image_unixtime(gid_list_)
            # Find closest
            closest_diff, closest_gps = np.inf, None
            for unixtime_, gps_ in zip(unixtime_list_, gps_list_):
                diff = abs(unixtime - unixtime_)
                if diff < closest_diff and gps_ != (-1, -1):
                    closest_diff = diff
                    closest_gps = gps_
            # Assign closest
            if closest_gps is not None and closest_diff <= min_diff:
                recovered_aid_list.append(aid)
                recovered_gps_list.append(closest_gps)
                recovered_dist_list.append(closest_diff)
                num_found += 1
                h = closest_diff // 3600
                closest_diff %= 3600
                m = closest_diff // 60
                closest_diff %= 60
                s = closest_diff
                print('FOUND LOCATION FOR AID %d' % (aid,))
                print('\tDIFF   : %d H, %d M, %d S' % (h, m, s,))
                print('\tNEW GPS: %s' % (closest_gps,))
            else:
                not_found.add(note)
        else:
            not_found.add(note)
    print(r'%d \ %d \ %d \ %d' % (num_all, num_bad, num_unrecovered, num_found,))
    num_recovered = len(recovered_aid_list)
    num_unrecovered = num_bad - len(recovered_aid_list)
    print('Missing GPS: %d' % (num_bad,))
    print('Recovered  : %d' % (num_recovered,))
    print('Unrecovered: %d' % (num_unrecovered,))
    print('Not Found  : %r' % (not_found,))
    return recovered_aid_list, recovered_gps_list, recovered_dist_list


@register_ibs_method
def commit_ggr_fix_gps(ibs, **kwargs):
    if False:
        vals = ibs.compute_ggr_fix_gps_contributors_aids(**kwargs)
        recovered_aid_list, recovered_gps_list, recovered_dist_list = vals
        recovered_gid_list = ibs.get_annot_gids(recovered_aid_list)
    else:
        vals = ibs.compute_ggr_fix_gps_contributors_gids(**kwargs)
        recovered_gid_list, recovered_gps_list, recovered_dist_list = vals

    zipped = zip(recovered_gid_list, recovered_gps_list, recovered_dist_list)
    assignment_dict = {}
    for gid, gps, dist in zipped:
        if gid not in assignment_dict:
            assignment_dict[gid] = []
        assignment_dict[gid].append((dist, gps))

    assignment_gid_list = []
    assignment_gps_list = []
    for assignment_gid in assignment_dict:
        assignment_list = sorted(assignment_dict[assignment_gid])
        assignment_gps = assignment_list[0][1]
        assignment_gid_list.append(assignment_gid)
        assignment_gps_list.append(assignment_gps)

    ibs.set_image_gps(assignment_gid_list, assignment_gps_list)


def merge_ggr_staged_annots_marriage(
    ibs, user_id_list, user_dict, aid_list, index_list, min_overlap=0.10
):
    import itertools
    from wbia.other.detectfuncs import general_parse_gt_annots, general_overlap

    gt_dict = {}
    for aid in aid_list:
        gt = general_parse_gt_annots(ibs, [aid])[0][0]
        gt_dict[aid] = gt

    marriage_user_id_list = []
    for length in range(len(user_id_list), 0, -1):
        padding = len(user_id_list) - length
        combination_list = list(itertools.combinations(user_id_list, length))
        for combination in combination_list:
            combination = sorted(combination)
            combination += [None] * padding
            marriage_user_id_list.append(tuple(combination))

    marriage_aid_list = []
    for user_id1, user_id2, user_id3 in marriage_user_id_list:
        aid_list1 = user_dict.get(user_id1, [None])
        aid_list2 = user_dict.get(user_id2, [None])
        aid_list3 = user_dict.get(user_id3, [None])

        for aid1 in aid_list1:
            for aid2 in aid_list2:
                for aid3 in aid_list3:
                    marriage = [aid1, aid2, aid3]
                    padding = len(marriage)
                    marriage = [aid for aid in marriage if aid is not None]
                    marriage = sorted(marriage)
                    padding -= len(marriage)
                    marriage = sorted(marriage)
                    marriage += [None] * padding
                    marriage_aid_list.append(tuple(marriage))

    marriage_list = []
    for marriage_aids in marriage_aid_list:
        aid1, aid2, aid3 = marriage_aids
        assert aid1 in aid_list and aid1 is not None
        assert aid2 in aid_list or aid2 is None
        assert aid3 in aid_list or aid3 is None

        aid_list_ = [aid1, aid2, aid3]
        missing = len(aid_list_) - aid_list_.count(None)

        gt1 = gt_dict.get(aid1, None)
        gt2 = gt_dict.get(aid2, None)
        gt3 = gt_dict.get(aid3, None)

        overlap1 = 0.0 if None in [gt1, gt2] else general_overlap([gt1], [gt2])[0][0]
        overlap2 = 0.0 if None in [gt1, gt3] else general_overlap([gt1], [gt3])[0][0]
        overlap3 = 0.0 if None in [gt2, gt3] else general_overlap([gt2], [gt3])[0][0]

        # Assert min_overlap conditions
        overlap1 = 0.0 if overlap1 < min_overlap else overlap1
        overlap2 = 0.0 if overlap2 < min_overlap else overlap2
        overlap3 = 0.0 if overlap3 < min_overlap else overlap3

        score = np.sqrt(overlap1 + overlap2 + overlap3)

        marriage = [missing, score, set(marriage_aids)]
        marriage_list.append(marriage)

    marriage_list = sorted(marriage_list, reverse=True)

    segment_list = []
    married_aid_set = set([])
    for missing, score, marriage_aids in marriage_list:
        polygamy = len(married_aid_set & marriage_aids) > 0
        if not polygamy:
            marriage_aids = marriage_aids - {None}
            married_aid_set = married_aid_set | marriage_aids
            segment_list.append(marriage_aids)

    return segment_list


def merge_ggr_staged_annots_cluster(
    ibs, user_id_list, user_dict, aid_list, index_list, min_overlap=0.25
):
    from wbia.other.detectfuncs import general_parse_gt_annots, general_overlap
    from sklearn import cluster
    from scipy import sparse

    num_clusters = int(np.around(len(aid_list) / len(user_id_list)))

    ##############

    gt_list, species_set = general_parse_gt_annots(ibs, aid_list)
    connectivity = general_overlap(gt_list, gt_list)
    # Cannot match with little overlap
    connectivity[connectivity < min_overlap] = 0.0
    # Cannot match against your own aid
    np.fill_diagonal(connectivity, 0.0)
    # Cannot match against your own reviewer
    for start, finish in index_list:
        connectivity[start:finish, start:finish] = 0.0
    # Ensure that the matrix is symmetric, which it should always be
    connectivity = sparse.csr_matrix(connectivity)

    algorithm = cluster.AgglomerativeClustering(
        n_clusters=num_clusters,
        linkage='average',
        affinity='cityblock',
        connectivity=connectivity,
    )

    bbox_list = ibs.get_annot_bboxes(aid_list)
    X = np.vstack(bbox_list)
    algorithm.fit(X)
    if hasattr(algorithm, 'labels_'):
        prediction_list = algorithm.labels_.astype(np.int)
    else:
        prediction_list = algorithm.predict(X)

    ##############

    segment_dict = {}
    for aid, prediction in zip(aid_list, prediction_list):
        if prediction not in segment_dict:
            segment_dict[prediction] = set([])
        segment_dict[prediction].add(aid)
    segment_list = list(segment_dict.values())

    return segment_list


@register_ibs_method
def merge_ggr_staged_annots(ibs, min_overlap=0.25, reviews_required=3, liberal_aoi=False):
    r"""
    Merge the staged annotations into a single set of actual annotations (with AoI)

    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.ibsfuncs merge_ggr_staged_annots

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> from os.path import expanduser
        >>> import wbia  # NOQA
        >>> # default_dbdir = join('/', 'data', 'wbia', 'GGR2-IBEIS')
        >>> default_dbdir = expanduser(join('~', 'data', 'GGR2-IBEIS'))
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> new_aid_list, broken_gid_list = ibs.merge_ggr_staged_annots()
        >>> print('Encountered %d invalid gids: %r' % (len(broken_gid_list), broken_gid_list, ))
    """

    def _normalize_confidences(gt_list):
        for index in range(len(gt_list)):
            gt = gt_list[index]
            area = gt['width'] * gt['height']
            gt['confidence'] = area
            gt_list[index] = gt
        return gt_list

    imageset_rowid = ibs.get_imageset_imgsetids_from_text('DETECT')
    gid_list = sorted(ibs.get_imageset_gids(imageset_rowid))

    # reviewed_list = ibs.get_image_reviewed(gid_list)
    # gid_list = ut.filter_items(gid_list, reviewed_list)

    metadata_dict_list = ibs.get_image_metadata(gid_list)
    aids_list = ibs.get_image_aids(gid_list, is_staged=True)

    existing_aid_list = []
    new_gid_list = []
    new_bbox_list = []
    new_interest_list = []

    zipped = list(zip(gid_list, metadata_dict_list, aids_list))
    # zipped = zipped[:10]

    broken_gid_list = []
    for gid, metadata_dict, aid_list in zipped:
        print('Processing gid = %r with %d annots' % (gid, len(aid_list),))
        if len(aid_list) < 2:
            continue

        staged = metadata_dict.get('staged', {})
        sessions = staged.get('sessions', {})
        user_id_list = sessions.get('user_ids', [])
        user_id_list = list(set(user_id_list))
        user_dict = {user_id: [] for user_id in user_id_list}

        user_id_list = ibs.get_annot_staged_user_ids(aid_list)

        try:
            for aid, user_id in zip(aid_list, user_id_list):
                assert user_id in user_dict
                user_dict[user_id].append(aid)
        except AssertionError:
            print('\tBad GID')
            broken_gid_list.append(gid)
            continue

        user_id_list = sorted(list(user_dict.keys()))
        if len(user_id_list) < reviews_required:
            continue

        ##############
        index_list = []
        aid_list = []
        for user_id in user_id_list:
            aid_list_ = user_dict[user_id]
            start = len(aid_list)
            aid_list.extend(aid_list_)
            finish = len(aid_list)
            index_list.append((start, finish))

        try:
            if False:
                segment_list = merge_ggr_staged_annots_cluster(
                    ibs,
                    user_id_list,
                    user_dict,
                    aid_list,
                    index_list,
                    min_overlap=min_overlap,
                )
            else:
                segment_list = merge_ggr_staged_annots_marriage(
                    ibs,
                    user_id_list,
                    user_dict,
                    aid_list,
                    index_list,
                    min_overlap=min_overlap,
                )
        except Exception:
            print('\tInvalid GID')
            broken_gid_list.append(gid)
            continue

        ##############

        existing_aid_list_ = ibs.get_image_aids(gid, is_staged=False)
        existing_aid_list.extend(existing_aid_list_)

        for aid_set in segment_list:
            aid_list = list(aid_set)
            bbox_list = ibs.get_annot_bboxes(aid_list)
            interest_list = ibs.get_annot_interest(aid_list)

            xtl_list = []
            ytl_list = []
            w_list = []
            h_list = []
            aoi_list = []
            for (xtl, ytl, w, h), interest in zip(bbox_list, interest_list):
                xtl_list.append(xtl)
                ytl_list.append(ytl)
                w_list.append(w)
                h_list.append(h)
                aoi_list.append(interest)

            xtl = sum(xtl_list) / len(xtl_list)
            ytl = sum(ytl_list) / len(ytl_list)
            w = sum(w_list) / len(w_list)
            h = sum(h_list) / len(h_list)
            bbox = (xtl, ytl, w, h)
            aoi = aoi_list.count(True)

            majority = 1 if liberal_aoi else ((len(h_list) + 1) // 2)
            interest = 1 if aoi >= majority else 0

            new_gid_list.append(gid)
            new_bbox_list.append(bbox)
            new_interest_list.append(interest)

        print('\tSegments = %r' % (segment_list,))
        print('\tAIDS = %d' % (len(segment_list),))

    print('Performing delete of %d existing non-staged AIDS' % (len(existing_aid_list),))
    ibs.delete_annots(existing_aid_list)
    print('Adding %d new non-staged AIDS' % (len(new_gid_list),))
    new_aid_list = ibs.add_annots(
        new_gid_list, bbox_list=new_bbox_list, interest_list=new_interest_list
    )

    return new_aid_list, broken_gid_list


@register_ibs_method
def check_ggr_valid_aids(
    ibs, aid_list, species='zebra_grevys', threshold=0.75, enable_grid=True, verbose=True
):
    num_start = len(aid_list)

    # Filter by species
    aid_list = ibs.filter_annotation_set(aid_list, species=species)

    # Filter by viewpoint
    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    aid_list = [
        aid
        for aid, viewpoint in zip(aid_list, viewpoint_list)
        if viewpoint is not None and 'right' in viewpoint
    ]

    # Filter by confidence or AoI
    interest_list = ibs.get_annot_interest(aid_list)
    metadata_list = ibs.get_annot_metadata(aid_list)
    confidence_list = [
        metadata.get('confidence', {}).get('localization', 1.0)
        for metadata in metadata_list
    ]
    excluded_list = [metadata.get('excluded', False) for metadata in metadata_list]

    grid_list = [
        metadata.get('turk', {}).get('grid', False) for metadata in metadata_list
    ]
    if not enable_grid:
        grid_list = [False] * len(grid_list)

    zipped = list(zip(aid_list, interest_list, confidence_list, excluded_list, grid_list))
    aid_list = [
        aid
        for aid, interest, confidence, excluded, grid in zipped
        if not excluded and not grid and (interest or confidence > threshold)
    ]

    num_finish = len(aid_list)
    num_difference = num_start - num_finish
    if verbose:
        print(
            'Filtered out %d annotations from %d / %d'
            % (num_difference, num_finish, num_start,)
        )

    return aid_list


@register_ibs_method
def create_ggr_match_trees(ibs):
    r"""
    CommandLine:
        python -m wbia.other.ibsfuncs create_ggr_match_trees

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> from os.path import expanduser
        >>> import wbia  # NOQA
        >>> default_dbdir = join('/', 'data', 'wbia', 'GGR2-IBEIS')
        >>> # default_dbdir = expanduser(join('~', 'data', 'GGR2-IBEIS'))
        >>> dbdir = ut.get_argval('--dbdir', type_=str, default=default_dbdir)
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> imageset_rowid_list = ibs.create_ggr_match_trees()
    """
    imageset_rowid_list = ibs.get_valid_imgsetids()
    imageset_text_list = ibs.get_imageset_text(imageset_rowid_list)
    tag_list = [ibs.parse_ggr_name(imageset_text) for imageset_text in imageset_text_list]

    # Check that merged cars exist
    value_list = [
        (tag[2], tag[1], imageset_rowid)
        for imageset_rowid, tag in zip(imageset_rowid_list, tag_list)
        if tag is not None
    ]
    car_dict = {}
    for number, letter, imageset_rowid in value_list:
        if number not in car_dict:
            car_dict[number] = []
        car_dict[number].append(imageset_rowid)

    car_key_list = sorted(list(car_dict.keys()))
    for car_key in car_key_list:
        imageset_rowid_list_ = car_dict[car_key]
        gids_list = ibs.get_imageset_gids(imageset_rowid_list_)
        gid_list = ut.flatten(gids_list)
        imageset_text_ = 'GGR2,%d' % (car_key,)
        if imageset_text_ not in imageset_text_list:
            ibs.set_image_imagesettext(gid_list, [imageset_text_] * len(gid_list))

    # Partition Cars
    imageset_rowid_list = ibs.get_valid_imgsetids()
    imageset_text_list = ibs.get_imageset_text(imageset_rowid_list)
    tag_list = [
        ibs.parse_ggr_name(imageset_text, require_short=True)
        for imageset_text in imageset_text_list
    ]
    value_list = [
        (tag[2], imageset_rowid)
        for imageset_rowid, tag in zip(imageset_rowid_list, tag_list)
        if tag is not None
    ]
    value_list = sorted(value_list)

    k = 2
    species_list = [
        ('Zebra High', 'zebra_grevys', 0.75, 5),
        ('Zebra Low', 'zebra_grevys', 0.0, 6),
        ('Giraffe High', 'giraffe_reticulated', 0.75, 4),
        ('Giraffe Low', 'giraffe_reticulated', 0.0, 5),
    ]
    for tag, species, threshold, levels in species_list:
        print('Processing Tag: %r' % (tag,))
        len_list = []
        imageset_rowid_list = []
        for number, imageset_rowid in value_list:
            # print('Processing car %r (ImageSet ID: %r)' % (number, imageset_rowid, ))
            aid_list = ibs.get_imageset_aids(imageset_rowid)
            aid_list_ = ibs.check_ggr_valid_aids(
                aid_list, species=species, threshold=threshold, verbose=False
            )

            if len(aid_list_) > 0:
                imageset_rowid_list.append(imageset_rowid)
                len_list.append(len(aid_list_))

        args = partition_ordered_list_equal_sum_recursive(
            len_list, imageset_rowid_list, k, levels
        )
        len_list_, imageset_rowid_list_ = args
        print_partition_sizes_recursive(len_list_, k)
        create_ggr_match_leaves_recursive(ibs, tag, imageset_rowid_list_, k)


def print_partition_sizes_recursive(vals, k, level=0, index=0):
    if isinstance(vals, int):
        return 0

    if len(vals) == 0:
        return 0

    val = vals[0]
    if isinstance(val, int):
        return sum(vals)

    length = 0
    for idx, val in enumerate(vals):
        length += print_partition_sizes_recursive(
            val, k=k, level=level + 1, index=(k * index) + idx
        )

    prefix = '\t' * level
    print('%sLevel %d, %d - %d' % (prefix, level, index, length,))

    return length


def create_ggr_match_leaves_recursive(ibs, tag, imageset_rowid_list, k, level=0, index=0):
    assert not isinstance(imageset_rowid_list, int)
    assert len(imageset_rowid_list) > 0

    imageset_rowid = imageset_rowid_list[0]
    if isinstance(imageset_rowid, int):
        return imageset_rowid_list

    imageset_rowid_list_ = []
    for idx, val in enumerate(imageset_rowid_list):
        imageset_rowid_list_ += create_ggr_match_leaves_recursive(
            ibs, tag, val, k=k, level=level + 1, index=(k * index) + idx
        )

    gid_list = ut.flatten(ibs.get_imageset_gids(imageset_rowid_list_))
    imageset_text_ = 'Leaf - %s - %d - %d' % (tag, level, index,)

    print(
        'Setting %d for %d to %r'
        % (len(gid_list), len(imageset_rowid_list_), imageset_text_,)
    )
    imageset_text_list = ibs.get_imageset_text(ibs.get_valid_imgsetids())
    if imageset_text_ not in imageset_text_list:
        ibs.set_image_imagesettext(gid_list, [imageset_text_] * len(gid_list))
        imageset_rowid = ibs.get_imageset_imgsetids_from_text(imageset_text_)
        metadata = ibs.get_imageset_metadata(imageset_rowid)
        assert 'leaf' not in metadata
        metadata['leaf'] = {
            'imageset_rowid_list': imageset_rowid_list_,
        }
        ibs.set_imageset_metadata([imageset_rowid], [metadata])

    return imageset_rowid_list_


def partition_ordered_list_equal_sum_recursive(vals, ids, k, level):
    if level <= 0:
        return vals, ids

    vals_ = partition_ordered_list_equal_sum(vals, k)

    ids_ = []
    start = 0
    for val_ in vals_:
        end = start + len(val_)
        id_ = ids[start:end]
        ids_.append(id_)
        start = end

    assert len(vals_) == len(ids_)
    for index in range(len(vals_)):
        temp_vals_ = vals_[index]
        temp_ids_ = ids_[index]

        assert len(temp_vals_) == len(temp_ids_)
        temp_vals_, temp_ids_ = partition_ordered_list_equal_sum_recursive(
            temp_vals_, temp_ids_, k, level - 1
        )

        vals_[index] = temp_vals_
        ids_[index] = temp_ids_

    return vals_, ids_


def partition_ordered_list_equal_sum(a, k):
    r"""
    Partition a sorted list a into k partitions

    Reference:
        https://stackoverflow.com/a/35518205
        https://gist.github.com/laowantong/ee675108eee64640e5f94f00d8edbcb4

    CommandLine:
        python -m wbia.other.ibsfuncs partition_ordered_list_equal_sum

    Example:
        >>> # DISABLE_DOCTEST
        >>> import random
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> a = [random.randint(0,20) for x in range(50)]
        >>> k = 10
        >>> print('Partitioning {0} into {1} partitions'.format(a, k))
        >>> b = partition_ordered_list_equal_sum(a, k)
        >>> print('The best partitioning is {0}\n    With heights {1}\n'.format(b, list(map(sum, b))))
    """
    if k <= 1:
        return [a]
    if k >= len(a):
        return [[x] for x in a]
    partition_between = [(i + 1) * len(a) // k for i in range(k - 1)]
    average_height = float(sum(a)) / k
    best_score = None
    best_partitions = None
    count = 0

    while True:
        starts = [0] + partition_between
        ends = partition_between + [len(a)]
        partitions = [a[starts[i] : ends[i]] for i in range(k)]
        heights = list(map(sum, partitions))

        abs_height_diffs = list(map(lambda x: abs(average_height - x), heights))
        worst_partition_index = abs_height_diffs.index(max(abs_height_diffs))
        worst_height_diff = average_height - heights[worst_partition_index]

        if best_score is None or abs(worst_height_diff) < best_score:
            best_score = abs(worst_height_diff)
            best_partitions = partitions
            no_improvements_count = 0
        else:
            no_improvements_count += 1

        if worst_height_diff == 0 or no_improvements_count > 5 or count > 100:
            return best_partitions
        count += 1

        move = -1 if worst_height_diff < 0 else 1
        bound_to_move = (
            0
            if worst_partition_index == 0
            else k - 2
            if worst_partition_index == k - 1
            else worst_partition_index - 1
            if (worst_height_diff < 0)
            ^ (heights[worst_partition_index - 1] > heights[worst_partition_index + 1])
            else worst_partition_index
        )
        direction = -1 if bound_to_move < worst_partition_index else 1
        partition_between[bound_to_move] += move * direction


@register_ibs_method
def alias_common_coco_species(ibs, **kwargs):
    aid_list = ibs.get_valid_aids()
    species_text_list = ibs.get_annot_species_texts(aid_list)
    alias_dict = {
        'cat': 'cat_domestic',
        'cow': 'cow_domestic',
        'dog': 'dog_domestic',
        'horse': 'horse_domestic',
        'sheep': 'sheep_domestic',
        'elephant': 'elephant_savannah',
        'giraffe': 'giraffe_reticulated',
        'turtles': 'turtle_sea',
        'wild_dog': 'dog_wild',
        'whale_sharkbodyhalf': 'whale_shark',
        'whale_sharkpart': 'whale_shark',
    }
    species_fixed_text_list = [
        alias_dict.get(species_text, species_text) for species_text in species_text_list
    ]
    ibs.set_annot_species(aid_list, species_fixed_text_list)


@register_ibs_method
def fix_coco_species(ibs, **kwargs):
    ibs.alias_common_coco_species()
    aid_list = ibs.get_valid_aids()
    species_text_list = ibs.get_annot_species_texts(aid_list)
    depc = ibs.depc_annot
    species_text_list_ = depc.get_property('labeler', aid_list, 'species')
    species_set = set(['zebra_grevys', 'zebra_plains'])
    zipped = zip(species_text_list, species_text_list_)
    species_fixed_text_list = [
        species_text_
        if species_text == 'zebra' and species_text_ in species_set
        else species_text
        for species_text, species_text_ in zipped
    ]
    species_fixed_text_list = [
        'zebra_plains' if species_fixed_text == 'zebra' else species_fixed_text
        for species_fixed_text in species_fixed_text_list
    ]
    ibs.set_annot_species(aid_list, species_fixed_text_list)


@register_ibs_method
def princeton_process_encounters(ibs, input_file_path, assert_valid=True, **kwargs):
    assert exists(input_file_path)

    with open(input_file_path, 'r') as input_file:
        header_line = input_file.readline()
        header_list = header_line.strip().split(',')
        line_list = input_file.readlines()
        lines_list = [line.strip().split(',') for line in line_list]

    header_list = [_.strip() for _ in header_list]
    imageset_text_set = set(
        ibs.get_imageset_text(ibs.get_valid_imgsetids(is_special=False))
    )

    seen_set = set([])
    invalid_list = []
    duplicate_list = []

    imageset_rowid_list = []
    metadata_list = []
    for index, line_list in enumerate(lines_list):
        metadata_dict = dict(zip(header_list, line_list))
        for key in metadata_dict:
            if metadata_dict[key] == '':
                metadata_dict[key] = None
        # Get primitives
        imageset_text = metadata_dict.pop('Image_Set')
        found = imageset_text in imageset_text_set
        # print('Processing %r (Found %s)' % (imageset_text, found, ))
        if not found:
            invalid_list.append(imageset_text)
            continue
        if imageset_text in seen_set:
            duplicate_list.append(imageset_text)
            continue
        imageset_rowid = ibs.get_imageset_imgsetids_from_text(imageset_text)
        # Check ImageSetIDs
        imageset_rowid_ = int(metadata_dict.pop('ImageSetID'))
        if imageset_rowid != imageset_rowid_:
            args = (
                imageset_text,
                imageset_rowid,
                imageset_rowid_,
            )
            print('Invalid ImageSetID for %r - WANTED: %r, GAVE: %r' % args)
        # Check #Imgs
        imageset_num_images = len(ibs.get_imageset_gids(imageset_rowid))
        imageset_num_images_ = int(metadata_dict.pop('#Imgs'))
        if imageset_num_images != imageset_num_images_:
            args = (
                imageset_text,
                imageset_num_images,
                imageset_num_images_,
            )
            print('Invalid #Imgs for %r - WANTED: %r, GAVE: %r' % args)
        # ADD TO TRACKER
        seen_set.add(imageset_text)
        imageset_rowid_list.append(imageset_rowid)
        metadata_list.append(metadata_dict)
    valid_list = list(seen_set)
    missing_list = list(imageset_text_set - seen_set)

    invalid = len(invalid_list) + len(duplicate_list) + len(missing_list)
    if invalid > 0:
        print('VALID:     %r' % (valid_list,))
        print('INVALID:   %r' % (invalid_list,))
        print('DUPLICATE: %r' % (duplicate_list,))
        print('MISSING:   %r' % (missing_list,))
    else:
        ibs.set_imageset_metadata(imageset_rowid_list, metadata_list)


@register_ibs_method
def princeton_process_individuals(ibs, input_file_path, **kwargs):
    assert exists(input_file_path)

    with open(input_file_path, 'r') as input_file:
        header_line = input_file.readline()
        header_list = header_line.strip().split(',')
        line_list = input_file.readlines()
        lines_list = [line.strip().split(',') for line in line_list]

    header_list = [_.strip() for _ in header_list]
    aid_list = ibs.get_valid_aids()
    # aid_set = set(aid_list)
    gid_list = ibs.get_valid_gids()
    gname_list = ibs.get_image_gnames(gid_list)

    seen_aid_set = set([])
    # seen_nid_set = set([])
    invalid_list = []
    duplicate_list = []

    annot_rowid_list = []
    metadata_list = []
    for index, line_list in enumerate(lines_list):
        primary_header_list = header_list[:8]
        primary_line_list = line_list[:8]
        secondary_line_list = line_list[8:]
        metadata_dict = dict(zip(primary_header_list, primary_line_list))
        for key in metadata_dict:
            if metadata_dict[key] == '':
                metadata_dict[key] = None
        # Get primitives
        aid = int(metadata_dict.pop('AnnotationID'))
        nid = ibs.get_annot_nids(aid)
        # gid = ibs.get_annot_gids(aid)
        gname = metadata_dict.pop('Photo#')
        # Check if found
        found1 = aid in aid_list
        found2 = gname in gname_list
        if not (found1 and found2):
            args = (
                gname,
                aid,
                found1,
                found2,
            )
            print('Invalid Gname %r AID %r (aid %s, gname %s)' % args)
            if found2:
                zip_list = [
                    value
                    for value in list(zip(gname_list, gid_list))
                    if value[0] == gname
                ]
                assert len(zip_list) == 1
                gid_ = zip_list[0][1]
                aid_list_ = ibs.get_image_aids(gid_)
                print('\t AID_LIST: %r' % (aid_list_,))
            invalid_list.append(aid)
            continue
        if aid in seen_aid_set:
            duplicate_list.append(aid)
            continue
        # if nid in seen_nid_set:
        #     args = (nid, gname, aid, )
        #     print('Duplicate NID %r for Gname %r AID %r' % args)
        #     continue
        # if nid is not None:
        #     seen_nid_set.add(nid)
        # Check gname
        gname_ = ibs.get_annot_image_names(aid)
        if gname != gname_:
            args = (
                gname,
                aid,
                gname_,
                gname,
            )
            print(
                'Invalid Photo# %r for AnnotationID for %r - WANTED: %r, GAVE: %r' % args
            )
            zip_list = [
                value for value in list(zip(gname_list, gid_list)) if value[0] == gname
            ]
            assert len(zip_list) == 1
            gid_ = zip_list[0][1]
            aid_list_ = ibs.get_image_aids(gid_)
            print('\t AID_LIST: %r' % (aid_list_,))
        seen_aid_set.add(aid)
        annot_rowid_list.append(aid)
        metadata_list.append(metadata_dict)

        # Check associated aids
        aid2_list = ut.flatten(ibs.get_name_aids([nid]))
        try:
            for index in range(len(secondary_line_list) // 2):
                gname2 = secondary_line_list[(index * 2) + 0]
                if gname2 == '':
                    continue
                aid2 = int(secondary_line_list[(index * 2) + 1])
                nid2 = ibs.get_annot_nids(aid2)
                if nid != nid2:
                    args = (
                        nid2,
                        aid2,
                        found1,
                        found2,
                    )
                    print(
                        'Invalid NID %r for Secondary AID2 %r (aid %s, gname %s)' % args
                    )
                    if nid > 0 and nid is not None:
                        print('\tfixing to %r...' % (nid,))
                        ibs.set_annot_name_rowids([aid2], [nid])
                        aid2_list = ut.flatten(ibs.get_name_aids([nid]))
                # Check if found
                found1 = aid2 in aid2_list
                found2 = gname2 in gname_list
                if not (found1 and found2):
                    args = (
                        aid2,
                        found1,
                        found2,
                    )
                    print('Invalid Secondary AID2 %r (aid %s, gname %s)' % args)
                    args = (
                        gname,
                        aid,
                    )
                    print('\tGname %r AID %r' % args)
                    args = (
                        nid2,
                        nid,
                    )
                    print('\tNIDs - WANTED: %r, GAVE: %r' % args)
                    args = (aid2_list,)
                    print('\tAID2_LIST: %r' % args)
                    invalid_list.append(aid2)
                    continue
                seen_aid_set.add(aid2)
                annot_rowid_list.append(aid2)
                metadata_list.append(metadata_dict)
        except ValueError:
            args = (
                gname,
                aid,
            )
            print('Invalid secondary list for Gname %r AID %r' % args)
            raise ValueError

    valid_list = list(seen_aid_set)
    # missing_list = list(aid_set - seen_aid_set)

    invalid = len(invalid_list) + len(duplicate_list)  # + len(missing_list)
    if invalid > 0:
        print('VALID:     %r' % (valid_list,))
        print('INVALID:   %r' % (invalid_list,))
        print('DUPLICATE: %r' % (duplicate_list,))
        # print('MISSING:   %r' % (missing_list, ))
    else:
        ibs.set_annot_metadata(annot_rowid_list, metadata_list)

        # Set demographics to names
        name_sex_dict = {}
        name_species_dict = {}
        name_age_dict = {}
        for annot_rowid, metadata_dict in zip(annot_rowid_list, metadata_list):
            name_rowid = ibs.get_annot_nids(annot_rowid)
            if name_rowid < 0:
                new_nid = ibs.make_next_nids()
                ibs.set_annot_name_rowids([annot_rowid], [new_nid])
            sex_symbol = metadata_dict['Sex']
            if sex_symbol is not None:
                sex_symbol = sex_symbol.upper()
            species_symbol = metadata_dict['Species']
            if species_symbol is not None:
                species_symbol = species_symbol.upper()
            age_symbol = metadata_dict['Age']
            if age_symbol is not None:
                age_symbol = age_symbol.upper()
            # Get names
            if name_rowid not in name_sex_dict:
                name_sex_dict[name_rowid] = []
            if sex_symbol in ['M', 'F']:
                name_sex_dict[name_rowid].append(sex_symbol)
            elif sex_symbol not in [None]:
                raise ValueError('INVALID SEX: %r' % (sex_symbol,))
            # Get species
            if name_rowid not in name_species_dict:
                name_species_dict[name_rowid] = []
            if species_symbol in ['PLAINS', 'GREVY_S', 'GREVYS', 'HYBRID']:
                name_species_dict[name_rowid].append(species_symbol)
            elif sex_symbol not in [None]:
                raise ValueError('INVALID SPECIES: %r' % (species_symbol,))
            # Get age
            if name_rowid not in name_age_dict:
                name_age_dict[name_rowid] = []
            if age_symbol in [
                'ADULT',
                'JUVENILE',
                'JUVENILE (6-9MO)',
                'JUVENILE (1-1.5YR)',
            ]:
                name_age_dict[name_rowid].append(age_symbol)
            elif age_symbol not in [None]:
                raise ValueError('INVALID AGE: %r' % (age_symbol,))

        aid_list_ = []
        species_text_ = []
        for name_rowid in name_sex_dict:
            name_sex_list = name_sex_dict[name_rowid]
            if len(name_sex_list) == 0:
                sex_text = 'UNKNOWN SEX'
            else:
                sex_mode = max(set(name_sex_list), key=name_sex_list.count)
                assert sex_mode in ['M', 'F']
                sex_text = 'Male' if sex_mode == 'M' else 'Female'
            ibs.set_name_sex_text([name_rowid], [sex_text])

            name_species_list = name_species_dict[name_rowid]
            if len(name_species_list) == 0:
                species_text = ibs.const.UNKNOWN
            else:
                species_mode = max(set(name_species_list), key=name_species_list.count)
                assert species_mode in ['PLAINS', 'GREVY_S', 'GREVYS', 'HYBRID']
                if species_mode == 'PLAINS':
                    species_text = 'zebra_plains'
                elif species_mode == 'HYBRID':
                    species_text = 'zebra_hybrid'
                else:
                    species_text = 'zebra_grevys'
            aid_list = ibs.get_name_aids(name_rowid)
            aid_list_ += aid_list
            species_text_ += [species_text] * len(aid_list)

            name_age_list = name_age_dict[name_rowid]
            if len(name_age_list) == 0:
                age_tuple = (
                    -1,
                    -1,
                )
            else:
                if 'JUVENILE (6-9MO)' in name_age_list:
                    assert 'JUVENILE (1-1.5YR)' not in name_age_list
                    age_tuple = (
                        6,
                        12,
                    )
                elif 'JUVENILE (1-1.5YR)' in name_age_list:
                    assert 'JUVENILE (6-9MO)' not in name_age_list
                    age_tuple = (
                        12,
                        24,
                    )
                else:
                    age_tag = max(set(name_age_list), key=name_age_list.count)
                    assert age_tag in ['ADULT', 'JUVENILE']
                    if age_tag == 'ADULT':
                        age_tuple = (36, None)
                    else:
                        age_tuple = (
                            -1,
                            -1,
                        )
            aid_list_ = ibs.get_name_aids(name_rowid)
            ibs.set_annot_age_months_est_min(aid_list_, [age_tuple[0]] * len(aid_list_))
            ibs.set_annot_age_months_est_max(aid_list_, [age_tuple[1]] * len(aid_list_))
        ibs.set_annot_species(aid_list_, species_text_)


@register_ibs_method
def princeton_cameratrap_ocr_bottom_bar_csv(
    ibs, prefix='/data/raw/unprocessed/horses/', threshold=0.39
):
    gid_list = ibs.get_valid_gids()
    gid_list = sorted(gid_list)

    uuid_list = ibs.get_image_uuids(gid_list)
    uri_original_list = ibs.get_image_uris_original(gid_list)
    value_dict_list = ibs.princeton_cameratrap_ocr_bottom_bar(gid_list=gid_list)

    groundtruth_list_ = ibs.get_image_cameratrap(gid_list)
    groundtruth_list = []
    for groundtruth in groundtruth_list_:
        if groundtruth == 1:
            label = 'positive'
        elif groundtruth == 0:
            label = 'negative'
        else:
            label = ''
        groundtruth_list.append(label)

    config = {
        'classifier_algo': 'densenet',
        'classifier_weight_filepath': 'ryan.densenet.v2',
    }
    prediction_list = ibs.depc_image.get_property(
        'classifier', gid_list, 'class', config=config
    )
    confidence_list = ibs.depc_image.get_property(
        'classifier', gid_list, 'score', config=config
    )
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence in zip(prediction_list, confidence_list)
    ]
    prediction_list = [
        'positive' if confidence >= threshold else 'negative'
        for confidence in confidence_list
    ]

    same_list = []
    for gt, pred in zip(groundtruth_list, prediction_list):
        if len(gt) > 0:
            same = 'true' if gt == pred else 'false'
        else:
            same = ''
        same_list.append(same)

    header_list = [
        'IMAGE_UUID',
        'FILEPATH',
        'TEMP_CELSIUS',
        'TEMP_FAHRENHEIT',
        'DATE_MONTH',
        'DATE_DAY',
        'DATE_YEAR',
        'TIME_HOUR',
        'TIME_MINUTE',
        'TIME_SECOND',
        'SEQUENCE_NUMBER',
        'CLASSIFY_HUMAN_LABEL',
        'CLASSIFY_COMPUTER_LABEL',
        'CLASSIFY_AGREEANCE',
    ]
    line_list = [','.join(header_list)]

    zipped = zip(
        uuid_list,
        uri_original_list,
        value_dict_list,
        groundtruth_list,
        prediction_list,
        same_list,
    )
    for uuid_, uri_original, value_dict, groundtruth, prediction, same in zipped:
        datetime = value_dict.get('datetime')
        line = [
            uuid_,
            uri_original.replace(prefix, ''),
            value_dict.get('temp').get('c'),
            value_dict.get('temp').get('f'),
            datetime.month,
            datetime.day,
            datetime.year,
            datetime.hour,
            datetime.minute,
            datetime.second,
            value_dict.get('sequence'),
            groundtruth,
            prediction,
            same,
        ]
        line = ','.join(map(str, line))
        line_list.append(line)

    with open('export.csv', 'w') as export_file:
        export_file.write('\n'.join(line_list))


@register_ibs_method
def princeton_cameratrap_ocr_bottom_bar_accuracy(ibs, offset=61200, **kwargs):
    # status_list = ibs.princeton_cameratrap_ocr_bottom_bar_accuracy()
    gid_list = ibs.get_valid_gids()
    value_dict_list = ibs.princeton_cameratrap_ocr_bottom_bar(gid_list=gid_list)
    value_dict_list = list(value_dict_list)
    datetime_list = ibs.get_image_datetime(gid_list)

    key_list = ['temp', 'date', 'time', 'datetime', 'sequence']
    status_dict = {
        'success': 0,
    }
    status_dict['failure'] = {key: 0 for key in key_list}
    status_dict['sanity'] = {key: 0 for key in key_list}
    status_dict['failure']['tempc'] = 0
    status_dict['failure']['tempf'] = 0
    status_dict['sanity']['tempc'] = 0
    status_dict['sanity']['tempf'] = 0

    difference_list = []
    for value_dict, datetime_ in zip(value_dict_list, datetime_list):
        success = True
        printing = False
        for key in key_list:
            if key not in value_dict:
                success = False
                status_dict['failure'][key] += 1
            else:
                if key == 'temp':
                    temp = value_dict[key]
                    if 'c' not in temp:
                        success = False
                        status_dict['failure']['tempc'] += 1
                    if 'f' not in temp:
                        success = False
                        status_dict['failure']['tempf'] += 1
        if success:
            status_dict['success'] += 1

            temp = value_dict['temp']
            tempc = temp.get('c')
            tempf = temp.get('f')

            tempc_ = (tempf - 32) * 5.0 / 9.0
            if abs(tempc_ - tempc) > 1:
                status_dict['sanity']['tempc'] += 1
                printing = True

            tempf_ = ((9.0 / 5.0) * tempc) + 32
            if abs(tempf_ - tempf) > 1:
                status_dict['sanity']['tempf'] += 1
                printing = True

            date = datetime.date(*value_dict['date'])
            date_ = datetime_.date()
            delta = date_ - date
            if delta.days > 0:
                printing = True
                status_dict['sanity']['date'] += 1

            time = datetime.time(*value_dict['time'])
            time_ = datetime_.time()
            time = datetime.datetime.combine(datetime.date.today(), time)
            time_ = datetime.datetime.combine(datetime.date.today(), time_)
            delta = time_ - time
            if (delta.seconds - offset) > 1:
                printing = True
                status_dict['sanity']['time'] += 1
                difference_list.append('%0.02f' % (delta.seconds,))

            delta = datetime_ - value_dict['datetime']
            if (delta.seconds - offset) > 1:
                printing = True
                status_dict['sanity']['datetime'] += 1
                difference_list.append('%0.02f' % (delta.seconds,))

            sequence = value_dict['sequence']
            if sequence < 0 or 10000 < sequence:
                printing = True
                status_dict['sanity']['sequence'] += 1
        else:
            printing = True

        if printing:
            print('Failed: %r' % (value_dict.get('split', None),))
    print(ut.repr3(status_dict))
    return status_dict


@register_ibs_method
def princeton_cameratrap_ocr_bottom_bar(ibs, gid_list=None):
    print('OCR Camera Trap Bottom Bar')
    if gid_list is None:
        gid_list = ibs.get_valid_gids()

    gid_list = ibs.get_valid_gids()
    raw_list = ibs.depc_image.get_property('cameratrap_exif', gid_list, 'raw')

    value_dict_list = []
    for raw in raw_list:
        value_dict = princeton_cameratrap_ocr_bottom_bar_parser(raw)
        value_dict_list.append(value_dict)

    return value_dict_list


def princeton_cameratrap_ocr_bottom_bar_parser(raw):
    value_dict = {}
    try:
        values = raw
        value_dict['raw'] = values
        assert len(values) > 0
        value_list = values.split(' ')
        value_dict['split'] = value_list
        assert len(values) > 0
        value_list = [
            value.strip().replace('C', '').replace('F', '').replace('°', '')
            for value in value_list
        ]
        if value_list[-2] == '0000':
            value_list[-2] = ''
        value_list = [value for value in value_list if len(value) > 0]
        if len(value_list[-2]) == 18:
            temp = value_list[-2]
            temp_date = temp[:10]
            temp_time = temp[10:]
            value_list = value_list[:-2] + [temp_date] + [temp_time] + value_list[-1:]
        assert len(value_list) >= 5
        value_list_ = value_list[-5:]
        assert len(value_list_) == 5
        value_dict['parsed'] = value_list_
        # print('Parsed: %s' % (value_list_, ))
        tempc, tempf, date, time, sequence = value_list_

        try:
            assert len(tempc) > 0
            if len(tempc) > 2:
                tempc = tempc[-2:]
            tempc = int(tempc)
            if 'temp' not in value_dict:
                value_dict['temp'] = {}
            value_dict['temp']['c'] = tempc
        except Exception:
            pass
        try:
            assert len(tempf) > 0
            tempf = int(tempf)
            if 'temp' not in value_dict:
                value_dict['temp'] = {}
            value_dict['temp']['f'] = tempf
        except Exception:
            pass
        try:
            date = date.strip().replace('/', '')
            assert len(date) == 8
            month = date[0:2]
            day = date[2:4]
            year = date[4:8]
            month = int(month)
            day = int(day)
            year = int(year)
            value_dict['date'] = (
                year,
                month,
                day,
            )
        except Exception:
            month, day, year = None, None, None
            pass
        try:
            time = time.strip().replace(':', '')
            if len(time) == 8 and time[2] in ['1', '3'] and time[5] in ['1', '3']:
                time = time[0:2] + time[3:5] + time[6:8]
            assert len(time) == 6
            hour = time[0:2]
            minute = time[2:4]
            second = time[4:6]
            hour = int(hour)
            minute = int(minute)
            second = int(second)
            value_dict['time'] = (
                hour,
                minute,
                second,
            )
        except Exception:
            hour, minute, second = None, None, None
            pass
        try:
            assert None not in [month, day, year, hour, minute, second]
            value_dict['datetime'] = datetime.datetime(
                year, month, day, hour, minute, second
            )
        except Exception:
            pass
        try:
            assert len(sequence) == 4
            sequence = int(sequence)
            value_dict['sequence'] = sequence
        except Exception:
            pass
    except Exception:
        pass

    return value_dict


@register_ibs_method
def import_folder(ibs, path, recursive=True, **kwargs):
    from wbia.detecttools.directory import Directory

    direct = Directory(path, recursive=recursive, images=True)
    gid_list = ibs.add_images(direct.files(), **kwargs)
    return gid_list


@register_ibs_method
def export_ggr_folders(ibs, output_path=None):
    from os.path import exists, join
    import pytz
    import tqdm
    import math

    if output_path is None:
        output_path_clean = '/media/jason.parham/princeton/GGR-2016-CLEAN/'
        output_path_census = '/media/jason.parham/princeton/GGR-2016-CENSUS/'
        prefix = '/media/extend/GGR-COMBINED/'
        needle = 'GGR Special'

        output_path_clean = '/media/jason.parham/princeton/GGR-2018-CLEAN/'
        output_path_census = '/media/jason.parham/princeton/GGR-2018-CENSUS/'
        prefix = '/data/wbia/GGR2/GGR2018data/'
        needle = 'GGR Special Zone -'

    gid_list = ibs.get_valid_gids()

    uuid_list = ibs.get_image_uuids(gid_list)
    gpath_list = ibs.get_image_paths(gid_list)
    datetime_list = ibs.get_image_datetime(gid_list)
    gps_list = ibs.get_image_gps2(gid_list)
    uri_original_list = ibs.get_image_uris_original(gid_list)
    aids_list = ibs.get_image_aids(gid_list)
    note_list = ibs.get_image_notes(gid_list)

    clean_line_list = []
    census_line_list = []

    imageset_dict = {}
    imageset_rowid_list = ibs.get_valid_imgsetids()
    imageset_text_list = ibs.get_imageset_text(imageset_rowid_list)
    for imageset_rowid, imageset_text in zip(imageset_rowid_list, imageset_text_list):
        if needle in imageset_text:
            imageset_text = imageset_text.replace(needle, '').strip()
            print(imageset_text)
            imageset_dict[imageset_text] = set(ibs.get_imageset_gids(imageset_rowid))

    header = [
        'IMAGE_UUID',
        'FILEPATH',
        'GGR_CAR_NUMBER',
        'GGR_PERSON_LETTER',
        'GGR_IMAGE_INDEX',
        'USED IN GGR CENSUS?',
        'DATE',
        'TIME',
        'GPS_LATITUDE',
        'GPS_LONGITUDE',
        'ZONES',
        'ZEBRA NAMES',
        'GIRAFFE NAMES',
        'ORIGINAL FILEPATH',
    ]
    header_str = ','.join(map(str, header))
    clean_line_list.append(header_str)
    census_line_list.append(header_str)

    VERIFY = True

    zipped = list(
        zip(
            gid_list,
            uuid_list,
            gpath_list,
            datetime_list,
            gps_list,
            uri_original_list,
            aids_list,
            note_list,
        )
    )

    tz = pytz.timezone('Africa/Nairobi')

    errors = []
    census = 0
    cleaned = 0
    for values in tqdm.tqdm(zipped):
        gid, uuid, gpath, datetime_, gps, uri_original, aid_list, note = values

        if prefix not in uri_original:
            errors.append(uri_original)
            continue
        uri_original = uri_original.replace(prefix, '')

        datetime_ = datetime_.astimezone(tz)
        date = datetime_.strftime('%x')
        time = datetime_.strftime('%X')

        lat, lon = gps

        if math.isnan(lat):
            lat = ''
        if math.isnan(lon):
            lon = ''

        note = note.split(',')
        if len(note) != 4:
            errors.append(note)
            continue
        ggr, number, letter, image = note
        image = int(image)
        assert image < 10000

        output_path = join(output_path_clean, number, letter)
        ut.ensuredir(output_path)

        output_filename = '%04d.jpg' % (image,)
        output_filepath = join(output_path, output_filename)
        assert '.jpg' in gpath
        # print(output_filepath)
        if VERIFY:
            assert exists(output_filepath)
        else:
            ut.copy(gpath, output_filepath)

        named = False
        nid_list = ibs.get_annot_nids(aid_list)
        for nid in nid_list:
            if nid > 0:
                named = True
                break

        zebra_aids = ibs.filter_annotation_set(aid_list, species='zebra_grevys')
        giraffe_aids = ibs.filter_annotation_set(aid_list, species='giraffe_reticulated')
        zebra_nids = ibs.get_annot_nids(zebra_aids)
        giraffe_nids = ibs.get_annot_nids(giraffe_aids)
        zebra_nids = [zebra_nid for zebra_nid in zebra_nids if zebra_nid > 0]
        giraffe_nids = [giraffe_nid for giraffe_nid in giraffe_nids if giraffe_nid > 0]
        zebra_names = ibs.get_name_texts(zebra_nids)
        giraffe_names = ibs.get_name_texts(giraffe_nids)
        zebra_name_str = ';'.join(map(str, zebra_names))
        giraffe_name_str = ';'.join(map(str, giraffe_names))

        zone_list = []
        for imageset_text in imageset_dict:
            if gid in imageset_dict[imageset_text]:
                zone_list.append(imageset_text)
        zone_list = sorted(zone_list)
        zone_str = ';'.join(zone_list)

        filepath_ = output_filepath.replace(output_path_clean, '')
        line = [
            uuid,
            filepath_,
            number,
            letter,
            image,
            'YES' if named else 'NO',
            date,
            time,
            lat,
            lon,
            zone_str,
            zebra_name_str,
            giraffe_name_str,
            uri_original,
        ]
        line_str = ','.join(map(str, line))
        clean_line_list.append(line_str)

        cleaned += 1
        if named:
            census += 1
            output_path = join(output_path_census, number, letter)
            ut.ensuredir(output_path)
            output_filepath = join(output_path, output_filename)
            # print(output_filepath)
            if VERIFY:
                assert exists(output_filepath)
            else:
                ut.copy(gpath, output_filepath)
            filepath_ = output_filepath.replace(output_path_census, '')
            line[1] = filepath_
            assert line[5] == 'YES'
            line_str = ','.join(map(str, line))
            census_line_list.append(line_str)

    assert cleaned + 1 == len(clean_line_list)
    assert census + 1 == len(census_line_list)

    clean_line_str = '\n'.join(clean_line_list)
    census_line_str = '\n'.join(census_line_list)

    clean_filepath = join(output_path_clean, 'manifest.csv')
    with open(clean_filepath, 'w') as clean_file:
        clean_file.write(clean_line_str)

    census_filepath = join(output_path_census, 'manifest.csv')
    with open(census_filepath, 'w') as census_file:
        census_file.write(census_line_str)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.other.ibsfuncs
        python -m wbia.other.ibsfuncs --allexamples
        python -m wbia.other.ibsfuncs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
