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
from six.moves import zip, range, map
from os.path import split, join, exists
import numpy as np
import vtool as vt
import utool as ut
from utool._internal.meta_util_six import get_funcname, get_imfunc, set_funcname
import itertools as it
from ibeis import constants as const
from ibeis.control import accessor_decors
from ibeis.control import controller_inject
from ibeis import annotmatch_funcs  # NOQA
import xml.etree.ElementTree as ET

# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[ibsfuncs]')


# Must import class before injection
CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))


@ut.make_class_postinject_decorator(CLASS_INJECT_KEY, __name__)
def postinject_func(ibs):
    r"""
    Args:
        ibs (IBEISController):

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-postinject_func

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
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
        ut.inject_func_as_method(ibs, unflat_getter,
                                 allow_override=ibs.allow_override)
    # very hacky, but useful
    ibs.unflat_map = unflat_map


@register_ibs_method
def refresh(ibs):
    """
    DEPRICATE
    """
    from ibeis import ibsfuncs
    from ibeis import all_imports
    ibsfuncs.rrr()
    all_imports.reload_all()
    ibs.rrr()


@register_ibs_method
def export_to_hotspotter(ibs):
    from ibeis.dbio import export_hsdb
    export_hsdb.export_ibeis_to_hotspotter(ibs)


#def export_image_subset(ibs, gid_list, dst_fpath=None):
#    dst_fpath = ut.truepath('~')
#    #gid_list = [692, 693, 680, 781, 751, 753, 754, 755, 756]
#    gpath_list = ibs.get_image_paths(gid_list)
#    gname_list = [join(dst_fpath, gname) for gname in
#    ibs.get_image_gnames(gid_list)]
#    ut.copy_files_to(gpath_list, dst_fpath_list=gname_list)


@register_ibs_method
def get_image_time_statstr(ibs, gid_list=None):
    if gid_list is None:
        gid_list = ibs.get_valid_gids()
    unixtime_list_ = ibs.get_image_unixtime(gid_list)
    utvalid_list   = [time != -1 for time in unixtime_list_]
    unixtime_list  = ut.compress(unixtime_list_, utvalid_list)
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
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: filtered_aid_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> filtered_aid_list = filter_junk_annotations(ibs, aid_list)
        >>> result = str(filtered_aid_list)
        >>> print(result)
    """
    isjunk_list = ibs.get_annot_isjunk(aid_list)
    filtered_aid_list = ut.filterfalse_items(aid_list, isjunk_list)
    return filtered_aid_list


@register_ibs_method
def compute_all_chips(ibs, **kwargs):
    """
    Executes lazy evaluation of all chips
    """
    print('[ibs] compute_all_chips')
    aid_list = ibs.get_valid_aids(**kwargs)
    cid_list = ibs.depc_annot.get_rowids('chips', aid_list)
    return cid_list


@register_ibs_method
def ensure_annotation_data(ibs, aid_list, chips=True, feats=True,
                           featweights=False):
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
def use_images_as_annotations(ibs, gid_list, name_list=None, nid_list=None,
                              notes_list=None, adjust_percent=0.0,
                              tags_list=None):
    """ Adds an annotation the size of the entire image to each image.
    adjust_percent - shrinks the ANNOTATION by percentage on each side
    """
    pct = adjust_percent  # Alias
    gsize_list = ibs.get_image_sizes(gid_list)
    # Build bounding boxes as images size minus padding
    bbox_list  = [(int( 0 + (gw * pct)),
                   int( 0 + (gh * pct)),
                   int(gw - (gw * pct * 2)),
                   int(gh - (gh * pct * 2)))
                  for (gw, gh) in gsize_list]
    theta_list = [0.0 for _ in range(len(gsize_list))]
    aid_list = ibs.add_annots(gid_list, bbox_list, theta_list,
                              name_list=name_list, nid_list=nid_list,
                              notes_list=notes_list)
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
        (bbox[0] / gw,
         bbox[1] / gh,
         (1 - (bbox[2] / gw)) / 2,
         (1 - (bbox[3] / gh)) / 2,)
        for bbox, (gw, gh) in zip(bbox_list, size_list)
    ]

    # Has the bounding box been moved past the default value?
    been_bbox_adjusted = np.array([
        np.abs(np.diff(np.array(list(ut.iprod(pcts, pcts))), axis=1)).max() > 1e-2
        for pcts in adjusted_list
    ])

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
            ut.get_caller_name(range(1, 3)), ut.compress(
                species_list, ut.not_list(isvalid_list)),)
    except AssertionError as ex:
        ut.printex(ex, iswarning=iswarning)
        if not iswarning:
            raise


@register_ibs_method
def assert_singleton_relationship(ibs, alrids_list):
    if ut.NO_ASSERTS:
        return
    try:
        assert all([len(alrids) == 1 for alrids in alrids_list]), (
            'must only have one relationship of a type')
    except AssertionError as ex:
        parent_locals = ut.get_parent_locals()
        ut.printex(ex, 'parent_locals=' + ut.dict_str(parent_locals),
                   key_list=['alrids_list', ])
        raise


@register_ibs_method
def assert_valid_gids(ibs, gid_list, verbose=False, veryverbose=False):
    r"""
    """
    isinvalid_list = [gid is None for gid in ibs.get_image_gid(gid_list)]
    try:
        assert not any(isinvalid_list), 'invalid gids: %r' % (
            ut.compress(gid_list, isinvalid_list),)
        isinvalid_list = [not isinstance(gid, ut.VALID_INT_TYPES)
                          for gid in gid_list]
        assert not any(isinvalid_list), 'invalidly typed gids: %r' % (
            ut.compress(gid_list, isinvalid_list),)
    except AssertionError as ex:
        print('dbname = %r' % (ibs.get_dbname()))
        ut.printex(ex)
        raise
    if veryverbose:
        print('passed assert_valid_gids')


@register_ibs_method
def assert_valid_aids(ibs, aid_list, verbose=False, veryverbose=False, msg='', auuid_list=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        verbose (bool):  verbosity flag(default = False)
        veryverbose (bool): (default = False)

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-assert_valid_aids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
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
    #if ut.NO_ASSERTS and not force:
    #    return
    #valid_aids = set(ibs.get_valid_aids())
    #invalid_aids = [aid for aid in aid_list if aid not in valid_aids]
    #isinvalid_list = [aid not in valid_aids for aid in aid_list]
    isinvalid_list = [aid is None for aid in ibs.get_annot_aid(aid_list)]
    #isinvalid_list = [aid not in valid_aids for aid in aid_list]
    invalid_aids = ut.compress(aid_list, isinvalid_list)
    try:
        assert not any(isinvalid_list), '%d/%d invalid aids: %r' % (
            sum(isinvalid_list), len(aid_list), invalid_aids,)
        isinvalid_list = [
            not ut.is_int(aid) for aid in aid_list]
        invalid_aids = ut.compress(aid_list, isinvalid_list)
        assert not any(isinvalid_list), '%d/%d invalidly typed aids: %r' % (
            sum(isinvalid_list), len(aid_list), invalid_aids,)
    except AssertionError as ex:
        if auuid_list is not None and len(auuid_list) == len(aid_list):
            invalid_auuids = ut.compress(auuid_list, isinvalid_list)  # NOQA
        else:
            invalid_auuids = 'not-available'
        dbname = ibs.get_dbname()  # NOQA
        locals_ = dict(dbname=dbname, invalid_auuids=invalid_auuids, invalid_aids=invalid_aids)
        ut.printex(
            ex, 'assert_valid_aids: ' + msg,
            locals_=locals_,
            keys=['invalid_aids', 'invalid_auuids', 'dbname']
        )
        raise
    if veryverbose:
        print('passed assert_valid_aids')


@register_ibs_method
def get_missing_gids(ibs, gid_list=None):
    r"""
    Finds gids with broken links to the original data.

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list): (default = None)

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_missing_gids --db GZ_Master1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> #ibs = ibeis.opendb('GZ_Master1')
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
        print(ut.truncate_str(ut.list_str(bad_gpaths), maxlen=500))
    assert num_bad_gids == 0, '%d images dont exist' % (num_bad_gids,)
    print('[check] checked %d images exist' % len(gid_list))


def assert_valid_names(name_list):
    """ Asserts that user specified names do not conflict with
    the standard unknown name """
    if ut.NO_ASSERTS:
        return
    def isconflict(name, other):
        return name.startswith(other) and len(name) > len(other)
    valid_namecheck = [not isconflict(name, const.UNKNOWN)
                       for name in name_list]
    assert all(valid_namecheck), ('A name conflicts with UKNONWN Name. -- '
                                  'cannot start a name with four underscores')


@ut.on_exception_report_input
def assert_lblannot_rowids_are_type(ibs, lblannot_rowid_list,
                                    valid_lbltype_rowid):
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
            (lbltype_rowid == valid_lbltype_rowid) or
            (lbltype_rowid is None and
             lblannot_rowid == const.UNKNOWN_LBLANNOT_ROWID)
            for lbltype_rowid, lblannot_rowid in
            zip(lbltype_rowid_list, lblannot_rowid_list)
        ]
        assert all(validtype_list), 'not all types match valid type'
    except AssertionError as ex:
        tup_list = list(map(str, list(
            zip(lbltype_rowid_list, lblannot_rowid_list))))
        print('[!!!] (lbltype_rowid, lblannot_rowid) = : ' +
              ut.indentjoin(tup_list))
        print('[!!!] valid_lbltype_rowid: %r' %
              (valid_lbltype_rowid,))

        ut.printex(ex, 'not all types match valid type',
                      keys=['valid_lbltype_rowid', 'lblannot_rowid_list'])
        raise


@register_ibs_method
def check_for_unregistered_images(ibs):
    images = ibs.images()
    # Check if any images in the image directory are unregistered
    gpath_disk = set(ut.ls(ibs.imgdir))
    gpath_registered = set(images.paths)
    overlaps = ut.set_overlaps(gpath_disk, gpath_registered, 'ondisk', 'reg')
    print('overlaps' + ut.repr3(overlaps))
    # gpath_unregistered = gpath_disk - gpath_registered
    # ut.remove_file_list(gpath_unregistered)
    return overlaps


@register_ibs_method
def check_image_consistency(ibs, gid_list=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list): (default = None)

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-check_image_consistency  --db=GZ_Master1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
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
    #check_image_uuid_consistency(ibs, gid_list)


def check_image_uuid_consistency(ibs, gid_list):
    """
    Checks to make sure image uuids are computed detemenistically
    by recomputing all guuids and checking that they are equal to
    what is already there.

    VERY SLOW

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-check_image_uuid_consistency --db=PZ_Master0
        python -m ibeis.other.ibsfuncs --test-check_image_uuid_consistency --db=GZ_Master1
        python -m ibeis.other.ibsfuncs --test-check_image_uuid_consistency
        python -m ibeis.other.ibsfuncs --test-check_image_uuid_consistency --db lynx

    Example:
        >>> # DISABLE_DOCTEST
        >>> # Check for very large files
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> gid_list_ = ibs.get_valid_gids()
        >>> gpath_list_ = ibs.get_image_paths(gid_list_)
        >>> bytes_list_ = [ut.get_file_nBytes(path) for path in gpath_list_]
        >>> sortx = ut.list_argsort(bytes_list_, reverse=True)[0:10]
        >>> gpath_list = ut.take(gpath_list_, sortx)
        >>> bytes_list = ut.take(bytes_list_, sortx)
        >>> gid_list   = ut.take(gid_list_, sortx)
        >>> ibeis.other.ibsfuncs.check_image_uuid_consistency(ibs, gid_list)
    """
    print('checking image uuid consistency')
    import ibeis.algo.preproc.preproc_image as preproc_image
    gpath_list = ibs.get_image_paths(gid_list)
    guuid_list = ibs.get_image_uuids(gid_list)
    for ix in ut.ProgressIter(range(len(gpath_list))):
        gpath = gpath_list[ix]
        guuid_stored = guuid_list[ix]
        param_tup = preproc_image.parse_imageinfo(gpath)
        guuid_computed = param_tup[0]
        assert guuid_stored == guuid_computed, 'image ix=%d had a bad uuid' % ix


@register_ibs_method
def check_annot_consistency(ibs, aid_list=None):
    r"""
    Args:
        ibs      (IBEISController):
        aid_list (list):

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-check_annot_consistency

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
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
    #print(ut.dict_str(dict(ut.debug_duplicate_items(annot_gid_list))))
    assert_images_exist(ibs, annot_gid_list)
    unique_gids = list(set(annot_gid_list))
    print('num_unique_images=%r / %r' % (len(unique_gids), len(annot_gid_list)))
    cfpath_list = ibs.get_annot_chip_fpath(aid_list, ensure=False)
    valid_chip_list = [None if cfpath is None else exists(cfpath) for cfpath in cfpath_list]
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
    # del dtool.__SQLITE__.converters['UUID']
    # import uuid
    # del dtool.__SQLITE__.adapters[(uuid.UUID, dtool.__SQLITE__.PrepareProtocol)]

        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('PZ_MTEST')
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
            except Exception as ex:
                failed_aids.append(aid)
        print('failed_aids = %r' % (failed_aids,))
        return failed_aids
    else:
        print('uuids do not seem to be corrupt')


def check_name_consistency(ibs, nid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        nid_list (list):

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-check_name_consistency

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_nids()
        >>> # execute function
        >>> result = check_name_consistency(ibs, nid_list)
        >>> # verify results
        >>> print(result)
    """
    #aids_list = ibs.get_name_aids(nid_list)
    print('check name consistency. len(nid_list)=%r' % len(nid_list))
    aids_list = ibs.get_name_aids(nid_list)
    print('Checking that all annotations of a name have the same species')
    species_rowids_list = ibs.unflat_map(ibs.get_annot_species_rowids, aids_list)
    error_list = []
    for aids, sids in zip(aids_list, species_rowids_list):
        if not ut.allsame(sids):
            error_msg = 'aids=%r have the same name, but belong to multiple species=%r' % (
                aids, ibs.get_species_texts(ut.unique_ordered(sids)))
            print(error_msg)
            error_list.append(error_msg)
    if len(error_list) > 0:
        raise AssertionError('A total of %d names failed check_name_consistency' % (
            len(error_list)))


@register_ibs_method
def check_name_mapping_consistency(ibs, nx2_aids):
    """
    checks that all the aids grouped in a name ahave the same name
    """
    # DEBUGGING CODE
    try:
        from ibeis import ibsfuncs
        _nids_list = ibsfuncs.unflat_map(ibs.get_annot_name_rowids, nx2_aids)
        assert all(map(ut.allsame, _nids_list))
    except Exception as ex:
        # THESE SHOULD BE CONSISTENT BUT THEY ARE NOT!!?
        #name_annots = [ibs.get_annot_name_rowids(aids) for aids in nx2_aids]
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
    print('size(aid_list) = ' + ut.byte_str2(ut.get_object_size(aid_list)))
    print('size(vert_list) = ' + ut.byte_str2(ut.get_object_size(vert_list)))
    print('size(uuid_list) = ' + ut.byte_str2(ut.get_object_size(uuid_list)))
    print('size(desc_list) = ' + ut.byte_str2(ut.get_object_size(desc_list)))
    print('size(kpts_list) = ' + ut.byte_str2(ut.get_object_size(kpts_list)))


def check_exif_data(ibs, gid_list):
    """ TODO CALL SCRIPT """
    import vtool.exif as exif
    from PIL import Image  # NOQA
    gpath_list = ibs.get_image_paths(gid_list)
    exif_dict_list = []
    for ix in ut.ProgressIter(range(len(gpath_list)), lbl='checking exif: '):
        gpath = gpath_list[ix]
        pil_img = Image.open(gpath, 'r')  # NOQA
        exif_dict = exif.get_exif_dict(pil_img)
        exif_dict_list.append(exif_dict)
        #if len(exif_dict) > 0:
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

    ut.print_stats(num_tags_list, 'num tags per image')

    print('tag frequency')
    print(ut.dict_str(key2_freq))


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
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('GZ_ALL')
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
    #TODO: Call more stuff, maybe rename to 'apply duct tape'
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
        ibs (IBEISController):  ibeis controller object
        gid_list (list): list of image ids

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-fix_exif_data

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='lynx')
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
        for pil_img in ut.ProgressIter(
            pil_img_gen, nTotal=len(gpath_list), lbl='reading exif: ',
            adjust=True)
    ]

    def fix_property(exif_getter, ibs_getter, ibs_setter, dirty_ibs_val, propname='property'):
        exif_prop_list = [exif_getter(_dict, None) for _dict in exif_dict_list]
        hasprop_list = [prop is not None for prop in exif_prop_list]

        exif_prop_list_ = ut.compress(exif_prop_list, hasprop_list)
        gid_list_       = ut.compress(gid_list, hasprop_list)
        ibs_prop_list   = ibs_getter(gid_list_)
        isdirty_list    = [prop == dirty_ibs_val for prop in ibs_prop_list]

        print('%d / %d need %s update' % (sum(isdirty_list),
                                          len(isdirty_list), propname))

        if False and sum(isdirty_list)  > 0:
            assert sum(isdirty_list) == len(isdirty_list), 'safety. remove and evaluate if hit'
            #ibs.set_image_imagesettext(gid_list_, ['HASGPS'] * len(gid_list_))
            new_exif_prop_list = ut.compress(exif_prop_list_, isdirty_list)
            dirty_gid_list = ut.compress(gid_list_, isdirty_list)
            ibs_setter(dirty_gid_list, new_exif_prop_list)

    FIX_GPS = True
    if FIX_GPS:
        fix_property(vt.get_lat_lon, ibs.get_image_gps, ibs.set_image_gps, (-1, -1), 'gps')
        #latlon_list = [vt.get_lat_lon(_dict, None) for _dict in exif_dict_list]
        #hasprop_list = [latlon is not None for latlon in latlon_list]

        #latlon_list_ = ut.compress(latlon_list, hasprop_list)
        #gid_list_    = ut.compress(gid_list, hasprop_list)
        #gps_list = ibs.get_image_gps(gid_list_)
        #isdirty_list = [gps == (-1, -1) for gps in gps_list]

        #print('%d / %d need gps update' % (sum(isdirty_list),
        #                                   len(isdirty_list)))

        #if False and sum(isdirty_list)  > 0:
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
        #exif_prop_list = [vt.get_unixtime(_dict, None) for _dict in exif_dict_list]
        #hasprop_list = [prop is not None for prop in exif_prop_list]

        #exif_prop_list_ = ut.compress(exif_prop_list, hasprop_list)
        #gid_list_       = ut.compress(gid_list, hasprop_list)
        #ibs_prop_list   = ibs.get_image_unixtime(gid_list_)
        #isdirty_list    = [prop == dirty_ibs_val for prop in ibs_prop_list]

        #print('%d / %d need time update' % (sum(isdirty_list),
        #                                    len(isdirty_list)))

        #if False and sum(isdirty_list)  > 0:
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
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-fix_invalid_nids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> # execute function
        >>> result = fix_invalid_nids(ibs)
        >>> # verify results
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
        if (len(invalid_nids) == 0 and
              invalid_nids[0] == const.UNKNOWN_NAME_ROWID and
              invalid_texts[0] == const.UNKNOWN):
            print('[ibs] found bad name rowids = %r' % (invalid_nids,))
            print('[ibs] found bad name texts  = %r' % (invalid_texts,))
            ibs.delete_names([const.UNKNOWN_NAME_ROWID])
        else:
            errmsg = 'Unfixable error: Found invalid (nid, text) pairs: '
            errmsg += ut.list_str(list(zip(invalid_nids, invalid_texts)))
            raise AssertionError(errmsg)


@register_ibs_method
def fix_invalid_name_texts(ibs):
    r"""
    Ensure  that no name text is empty or '____'

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-fix_invalid_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> # execute function
        >>> result = fix_invalid_name_texts(ibs)
        >>> # verify results
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
    is_invalid_name_text_list = [name_text in invalid_name_set
                                 for name_text in name_text_list]
    if any(is_invalid_name_text_list):
        invalid_nids = ut.compress(nid_list, is_invalid_name_text_list)
        invalid_texts = ut.compress(name_text_list, is_invalid_name_text_list)
        for count, (invalid_nid, invalid_text) in enumerate(zip(invalid_nids, invalid_texts)):
            conflict_set = invalid_name_set.union(
                set(ibs.get_name_texts(nid_list, apply_fix=False)))
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
        ibs (IBEISController):  ibeis controller object
        imgsetid_list (list):

    Returns:
        list: new_imgsetid_list

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-copy_imagesets

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> ibs.delete_all_imagesets()
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> imgsetid_list = ibs.get_valid_imgsetids()
        >>> # execute function
        >>> new_imgsetid_list = copy_imagesets(ibs, imgsetid_list)
        >>> # verify results
        >>> result = str(ibs.get_imageset_text(new_imgsetid_list))
        >>> assert [2] == list(set(map(len, ibs.get_image_imgsetids(ibs.get_valid_gids()))))
        >>> print(result)
        >>> ibs.delete_all_imagesets()
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
    """
    all_imagesettext_list = ibs.get_imageset_text(ibs.get_valid_imgsetids())
    imagesettext_list = ibs.get_imageset_text(imgsetid_list)
    new_imagesettext_list = [
        ut.get_nonconflicting_string(imagesettext + '_Copy(%d)', set(all_imagesettext_list))
        for imagesettext in imagesettext_list
    ]
    new_imgsetid_list = ibs.add_imagesets(new_imagesettext_list)
    gids_list = ibs.get_imageset_gids(imgsetid_list)
    #new_imgsetid_list =
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
    #nid_list = ibs.get_annot_nids(aid_list, distinguish_unknowns=False)
    flag_list = ibs.get_annot_exemplar_flags(aid_list)
    unknown_list = ibs.is_aid_unknown(aid_list)
    # Exemplars should all be known
    unknown_exemplar_flags = ut.compress(flag_list, unknown_list)
    unknown_aid_list = ut.compress(aid_list, unknown_list)
    print('Fixing %d unknown annotations set as exemplars' % (sum(unknown_exemplar_flags),))
    ibs.set_annot_exemplar_flags(unknown_aid_list, [False] * len(unknown_aid_list))
    #is_error = [not flag for flag in unknown_exemplar_flags]
    #new_annots = [flag if nid != const.UNKNOWN_NAME_ROWID else 0
    #              for nid, flag in
    #              zip(nid_list, flag_list)]
    #ibs.set_annot_exemplar_flags(aid_list, new_annots)
    pass


@register_ibs_method
def delete_all_recomputable_data(ibs):
    """
    Delete all cached data including chips and imagesets
    """
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
        python -m ibeis delete_cache --db testdb1

    Example:
        >>> # SCRIPT
        >>> import ibeis
        >>> ibs = ibeis.opendb()
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
        python -m ibeis.other.ibsfuncs delete_cachedir
        python -m ibeis delete_cachedir --db testdb1

    Example:
        >>> # SCRIPT
        >>> import ibeis
        >>> ibs = ibeis.opendb()
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
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis --tf delete_qres_cache
        python -m ibeis --tf delete_qres_cache --db PZ_MTEST
        python -m ibeis --tf delete_qres_cache --db PZ_Master1

    Example:
        >>> # SCRIPT
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
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


def delete_ibeis_database(dbdir):
    _ibsdb = join(dbdir, const.PATH_NAMES._ibsdb)
    print('[ibsfuncs] DELETEING: _ibsdb=%r' % _ibsdb)
    if exists(_ibsdb):
        ut.delete(_ibsdb)


def print_flann_cachedir(ibs):
    flann_cachedir = ibs.get_flann_cachedir()
    print(ut.list_str(ut.ls(flann_cachedir)))


@register_ibs_method
def vd(ibs):
    ibs.view_dbdir()


@register_ibs_method
def view_dbdir(ibs):
    ut.view_directory(ibs.get_dbdir())


@register_ibs_method
def get_empty_gids(ibs, imgsetid=None):
    """ returns gid list without any chips """
    gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
    nRois_list = ibs.get_image_num_annotations(gid_list)
    empty_gids = [gid for gid, nRois in zip(gid_list, nRois_list) if nRois == 0]
    return empty_gids


@register_ibs_method
def get_annot_vecs_cache(ibs, aids):
    """
    When you have a list with duplicates and you dont want to copy data
    creates a reference to each data object indexed by a dict
    """
    unique_aids = list(set(aids))
    unique_desc = ibs.get_annot_vecs(unique_aids)
    desc_cache = dict(list(zip(unique_aids, unique_desc)))
    return desc_cache


# TODO: move to const

@register_ibs_method
def get_annot_is_hard(ibs, aid_list):
    """
    CmdLine:
        ./dev.py --cmd --db PZ_Mothers

    Args:
        ibs (IBEISController):
        aid_list (list):

    Returns:
        list: is_hard_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0::2]
        >>> is_hard_list = get_annot_is_hard(ibs, aid_list)
        >>> result = str(is_hard_list)
        >>> print(result)
        [False, False, False, False, False, False, False]
    """
    notes_list = ibs.get_annot_notes(aid_list)
    is_hard_list = [const.HARD_NOTE_TAG in notes.upper().split() for (notes)
                    in notes_list]
    return is_hard_list


@register_ibs_method
def get_hard_annot_rowids(ibs):
    valid_aids = ibs.get_valid_aids()
    hard_aids = ut.compress(valid_aids, ibs.get_annot_is_hard(valid_aids))
    return hard_aids


@register_ibs_method
def get_easy_annot_rowids(ibs):
    hard_aids = ibs.get_hard_annot_rowids()
    easy_aids = ut.setdiff_ordered(ibs.get_valid_aids(), hard_aids)
    easy_aids = ut.compress(easy_aids, ibs.get_annot_has_groundtruth(easy_aids))
    return easy_aids


@register_ibs_method
def set_annot_is_hard(ibs, aid_list, flag_list):
    """
    Hack to mark hard cases in the notes column

    Example:
        >>> pz_mothers_hard_aids = [27, 43, 44, 49, 50, 51, 54, 66, 89, 97]
        >>> aid_list = pz_mothers_hard_aids
        >>> flag_list = [True] * len(aid_list)
    """
    notes_list = ibs.get_annot_notes(aid_list)
    is_hard_list = [const.HARD_NOTE_TAG in notes.lower().split() for (notes) in notes_list]
    def hack_notes(notes, is_hard, flag):
        """ Adds or removes hard tag if needed """
        if flag and is_hard or not (flag or is_hard):
            # do nothing
            return notes
        elif not is_hard and flag:
            # need to add flag
            return const.HARD_NOTE_TAG + ' '  + notes
        elif is_hard and not flag:
            # need to remove flag
            return notes.replace(const.HARD_NOTE_TAG, '').strip()
        else:
            raise AssertionError('impossible state')

    new_notes_list = [
        hack_notes(notes, is_hard, flag)
        for notes, is_hard, flag in zip(notes_list, is_hard_list, flag_list)]
    ibs.set_annot_notes(aid_list, new_notes_list)
    return is_hard_list


@register_ibs_method
@accessor_decors.getter_1to1
def is_nid_unknown(ibs, nid_list):
    return [ nid <= 0 for nid in nid_list]


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
    Uses an ibeis lookup function with a non-flat rowid list.
    In essence this is equivilent to map(method, unflat_rowids).
    The utility of this function is that it only calls method once.
    This is more efficient for calls that can take a list of inputs

    Args:
        method        (method):  ibeis controller method
        unflat_rowids (list): list of rowid lists

    Returns:
        list of values: unflat_vals

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-unflat_map

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> method = ibs.get_annot_name_rowids
        >>> unflat_rowids = ibs.get_name_aids(ibs.get_valid_nids())
        >>> unflat_vals = unflat_map(method, unflat_rowids)
        >>> result = str(unflat_vals)
        >>> print(result)
        [[1, 1], [2, 2], [3], [4], [5], [6], [7]]
    """
    #ut.assert_unflat_level(unflat_rowids, level=1, basetype=(int, uuid.UUID))
    # First flatten the list, and remember the original dimensions
    flat_rowids, reverse_list = ut.invertible_flatten2(unflat_rowids)
    # Then preform the lookup / implicit mapping
    flat_vals = method(flat_rowids, **kwargs)
    if True:
        assert len(flat_vals) == len(flat_rowids), (
            'flat lens not the same, len(flat_vals)=%d len(flat_rowids)=%d' %
            (len(flat_vals), len(flat_rowids),))
    # Then ut.unflatten2 the results to the original input dimensions
    unflat_vals = ut.unflatten2(flat_vals, reverse_list)
    if True:
        assert len(unflat_vals) == len(unflat_rowids), (
            'unflat lens not the same, len(unflat_vals)=%d len(unflat_rowids)=%d' %
            (len(unflat_vals), len(unflat_rowids),))
    return unflat_vals


#def unflat_filter(method, unflat_rowids, **kwargs):
    # does not seem possible with this input


def unflat_dict_map(method, dict_rowids, **kwargs):
    """ maps dictionaries of rowids to a function """
    key_list = list(dict_rowids.keys())
    unflat_rowids = list(dict_rowids.values())
    unflat_vals = unflat_map(method, unflat_rowids, **kwargs)
    keyval_iter = zip(key_list, unflat_vals)
    #_dict_cls = type(dict_rowids)
    #if isinstance(dict_rowids, ut.ddict):
    #    _dict_cls_args = (dict_rowids.default_factory, keyval_iter)
    #else:
    #    _dict_cls_args = (keyval_iter,)
    #dict_vals = _dict_cls(*_dict_cls_args)
    dict_vals = dict(keyval_iter)
    return dict_vals


def unflat_multimap(method_list, unflat_rowids, **kwargs):
    """ unflat_map, but allows multiple methods
    """
    # First flatten the list, and remember the original dimensions
    flat_rowids, reverse_list = ut.invertible_flatten2(unflat_rowids)
    # Then preform the lookup / implicit mapping
    flat_vals_list = [method(flat_rowids, **kwargs) for method in method_list]
    # Then ut.unflatten2 the results to the original input dimensions
    unflat_vals_list = [ut.unflatten2(flat_vals, reverse_list)
                        for flat_vals in flat_vals_list]
    return unflat_vals_list


def _make_unflat_getter_func(flat_getter):
    """
    makes an unflat version of an ibeis getter
    """
    if isinstance(flat_getter, types.MethodType):
        # Unwrap fmethods
        func = get_imfunc(flat_getter)
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
    #if ut.NO_ASSERTS:
    #    return
    try:
        msg = ('gpath_list must be in unix format (no backslashes).'
               'Failed on %d-th gpath=%r')
        for count, gpath in enumerate(gpath_list):
            assert gpath.find('\\') == -1, (msg % (count, gpath))
    except AssertionError as ex:
        ut.printex(ex, iswarning=True)
        gpath_list = list(map(ut.unixpath, gpath_list))
    return gpath_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_annot_info(ibs, aid_list, default=False, reference_aid=None, **kwargs):
    r"""
    Args:
        ibs (ibeis.IBEISController):  ibeis controller object
        aid_list (list):  list of annotation rowids
        default (bool): (default = False)

    Returns:
        list: infodict_list

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_annot_info --tb

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:2]
        >>> default = True
        >>> infodict_list = ibs.get_annot_info(1, default)
        >>> result = ('infodict_list = %s' % (ut.obj_str(infodict_list, nl=4),))
        >>> print(result)
    """
    # TODO rectify and combine with viz_helpers.get_annot_texts
    key_list = []
    vals_list = []

    #if len(aid_list) == 0:
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

    key = 'yawtext'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_yaw_texts(aid_list)]
        key_list += [key]

    key = 'time'
    if kwargs.get(key, default):
        vals_list += [ibs.get_annot_image_unixtimes(aid_list)]
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
        {key_: val for key_, val in zip(key_list, vals)}
        for vals in zip(*vals_list)
    ]
    return infodict_list


def aidstr(aid, ibs=None, notes=False):
    """ Helper to make a string from an aid """
    if not notes:
        return 'aid%d' % (aid,)
    else:
        assert ibs is not None
        notes = ibs.get_annot_notes(aid)
        name  = ibs.get_annot_names(aid)
        return 'aid%d-%r-%r' % (aid, str(name), str(notes))


@register_ibs_method
#@ut.time_func
#@profile
def update_exemplar_special_imageset(ibs):
    # FIXME SLOW
    exemplar_imgsetid = ibs.get_imageset_imgsetids_from_text(const.EXEMPLAR_IMAGESETTEXT)
    #ibs.delete_imagesets(exemplar_imgsetid)
    ibs.delete_gsgr_imageset_relations(exemplar_imgsetid)
    #aid_list = ibs.get_valid_aids(is_exemplar=True)
    #gid_list = ut.unique_ordered(ibs.get_annot_gids(aid_list))
    gid_list = list(set(_get_exemplar_gids(ibs)))
    #ibs.set_image_imagesettext(gid_list, [const.EXEMPLAR_IMAGESETTEXT] * len(gid_list))
    ibs.set_image_imgsetids(gid_list, [exemplar_imgsetid] * len(gid_list))


@register_ibs_method
#@ut.time_func
#@profile
def update_reviewed_unreviewed_image_special_imageset(ibs, reviewed=True, unreviewed=True):
    """
    Creates imageset of images that have not been reviewed
    and that have been reviewed (wrt detection)
    """
    # FIXME SLOW
    if unreviewed:
        unreviewed_imgsetid = ibs.get_imageset_imgsetids_from_text(const.UNREVIEWED_IMAGE_IMAGESETTEXT)
        ibs.delete_gsgr_imageset_relations(unreviewed_imgsetid)
        unreviewed_gids = _get_unreviewed_gids(ibs)  # hack
        ibs.set_image_imgsetids(unreviewed_gids, [unreviewed_imgsetid] * len(unreviewed_gids))
    if reviewed:
        reviewed_imgsetid = ibs.get_imageset_imgsetids_from_text(const.REVIEWED_IMAGE_IMAGESETTEXT)
        ibs.delete_gsgr_imageset_relations(reviewed_imgsetid)
        #gid_list = ibs.get_valid_gids(reviewed=False)
        #ibs.set_image_imagesettext(gid_list, [const.UNREVIEWED_IMAGE_IMAGESETTEXT] * len(gid_list))
        reviewed_gids   = _get_reviewed_gids(ibs)  # hack
        ibs.set_image_imgsetids(reviewed_gids, [reviewed_imgsetid] * len(reviewed_gids))


@register_ibs_method
#@ut.time_func
#@profile
def update_all_image_special_imageset(ibs):
    # FIXME SLOW
    allimg_imgsetid = ibs.get_imageset_imgsetids_from_text(const.ALL_IMAGE_IMAGESETTEXT)
    #ibs.delete_imagesets(allimg_imgsetid)
    gid_list = ibs.get_valid_gids()
    #ibs.set_image_imagesettext(gid_list, [const.ALL_IMAGE_IMAGESETTEXT] * len(gid_list))
    ibs.set_image_imgsetids(gid_list, [allimg_imgsetid] * len(gid_list))


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
    special_imgsetids_ = [get_imagesettext_imgsetid(imagesettext, ensure=False)
                          for imagesettext in special_imagesettext_list]
    special_imgsetids = [i for i in special_imgsetids_ if i is not None]
    return special_imgsetids


@register_ibs_method
@profile
def get_ungrouped_gids(ibs):
    """
    CommandLine:
        python -m ibeis.other.ibsfuncs --test-get_ungrouped_gids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
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
    has_imgsetids = [special_imgsetids.issuperset(set(imgsetids)) for imgsetids in imgsetids_list]
    ungrouped_gids = ut.compress(gid_list, has_imgsetids)
    return ungrouped_gids


@register_ibs_method
#@ut.time_func
#@profile
@profile
def update_ungrouped_special_imageset(ibs):
    """
    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-update_ungrouped_special_imageset

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb9')
        >>> # execute function
        >>> result = update_ungrouped_special_imageset(ibs)
        >>> # verify results
        >>> print(result)
    """
    # FIXME SLOW
    if ut.VERBOSE:
        print('[ibsfuncs] update_ungrouped_special_imageset.1')
    ungrouped_imgsetid = ibs.get_imageset_imgsetids_from_text(const.UNGROUPED_IMAGES_IMAGESETTEXT)
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
#@ut.time_func
@profile
def update_special_imagesets(ibs):
    if ut.get_argflag('--readonly-mode'):
        # SUPER HACK
        return
    # FIXME SLOW
    USE_MORE_SPECIAL_IMAGESETS = ibs.cfg.other_cfg.ensure_attr(
        'use_more_special_imagesets', False)
    if USE_MORE_SPECIAL_IMAGESETS:
        #ibs.update_reviewed_unreviewed_image_special_imageset()
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
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
    """
    # hack
    gid_list = ibs.get_valid_gids()
    flag_list = ibs.detect_cnn_yolo_exists(gid_list)
    gid_list_ = ut.compress(gid_list, ut.not_list(flag_list))
    # unreviewed and unshipped
    imgsets_list = ibs.get_image_imgsetids(gid_list_)
    nonspecial_imgset_ids = [ut.compress(ids, ut.not_list(mask))
                             for mask, ids in zip(ibs.unflat_map(ibs.is_special_imageset, imgsets_list), imgsets_list)]
    flags_list = ibs.unflat_map(ibs.get_imageset_shipped_flags, nonspecial_imgset_ids)
    # Keep images that have at least one instance in an unshipped non-special set
    flag_list = [not all(flags) for flags in flags_list]
    gid_list_ = ut.compress(gid_list_, flag_list)
    # Filter by if the user has specified the image has been reviewed manually
    flag_list = ibs.get_image_reviewed(gid_list_)
    gid_list_ = ut.compress(gid_list_, ut.not_list(flag_list))
    return gid_list_


def _get_reviewed_gids(ibs):
    # hack
    gid_list = ibs.get_valid_gids()
    flag_list = ibs.detect_cnn_yolo_exists(gid_list)
    gid_list_ = ut.filter_items(gid_list, flag_list)
    return gid_list_


def _get_gids_in_imgsetid(ibs, imgsetid):
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {GSG_RELATION_TABLE}
        WHERE
            imageset_rowid==?
        '''.format(**const.__dict__),
        params=(imgsetid,))
    return gid_list


def _get_dirty_reviewed_gids(ibs, imgsetid):
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {GSG_RELATION_TABLE}
        WHERE
            imageset_rowid==? AND
            image_rowid NOT IN (SELECT rowid FROM {IMAGE_TABLE} WHERE image_toggle_reviewed=1)
        '''.format(**const.__dict__),
        params=(imgsetid,))
    return gid_list


def _get_exemplar_gids(ibs):
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {ANNOTATION_TABLE}
        WHERE annot_exemplar_flag=1
        '''.format(**const.__dict__))
    return gid_list


#@register_ibs_method
#def print_stats(ibs):
    #from ibeis.other import dbinfo
    #dbinfo.dbstats(ibs)


@register_ibs_method
def print_dbinfo(ibs, **kwargs):
    from ibeis.other import dbinfo
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
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> verbosity = 1
        >>> print_annotation_table(ibs, verbosity)
    """
    exclude_columns = exclude_columns[:]
    if verbosity < 5:
        exclude_columns += ['annot_uuid', 'annot_verts']
    if verbosity < 4:
        exclude_columns += [
            'annot_xtl', 'annot_ytl', 'annot_width', 'annot_height',
            'annot_theta', 'annot_yaw', 'annot_detect_confidence',
            'annot_note', 'annot_parent_rowid']
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
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-print_annotmatch_table
        python -m ibeis.other.ibsfuncs --exec-print_annotmatch_table --db PZ_Master1

    Example:
        >>> # SCRIPT
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
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
    print(ibs.db.get_table_csv(const.FEATURE_TABLE, exclude_columns=[
        'feature_keypoints', 'feature_vecs']))


@register_ibs_method
def print_image_table(ibs, **kwargs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.IMAGE_TABLE, **kwargs))
    #, exclude_columns=['image_rowid']))


@register_ibs_method
def print_party_table(ibs, **kwargs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.PARTY_TABLE, **kwargs))
    #, exclude_columns=['image_rowid']))


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
        exclude_columns = ['annot_uuid', 'lblannot_uuid', 'annot_verts', 'feature_keypoints',
                           'feature_vecs', 'image_uuid', 'image_uri']
    if exclude_tables is None:
        exclude_tables = ['masks', 'recognitions', 'chips', 'features']
    for table_name in ibs.db.get_table_names():
        if table_name in exclude_tables:
            continue
        print('\n')
        print(ibs.db.get_table_csv(table_name, exclude_columns=exclude_columns))
    #ibs.print_image_table()
    #ibs.print_annotation_table()
    #ibs.print_lblannots_table()
    #ibs.print_alr_table()
    #ibs.print_config_table()
    #ibs.print_chip_table()
    #ibs.print_feat_table()
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
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
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
    """
    Returns if an annotation has been given a name (even if that name is temporary)
    """
    nid_list = ibs.get_annot_name_rowids(aid_list)
    return ibs.is_nid_unknown(nid_list)


def make_imagesettext_list(imgsetid_list, occur_cfgstr):
    # DEPRICATE
    imagesettext_list = [str(imgsetid) + occur_cfgstr for imgsetid in imgsetid_list]
    return imagesettext_list


@register_ibs_method
def batch_rename_consecutive_via_species(ibs, imgsetid=None):
    """ actually sets the new consectuive names"""
    new_nid_list, new_name_list = ibs.get_consecutive_newname_list_via_species(imgsetid=imgsetid)

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
    ibs.set_name_texts(new_nid_list, new_name_list, verbose=ut.NOT_QUIET)


@register_ibs_method
def get_consecutive_newname_list_via_species(ibs, imgsetid=None):
    """
    Just creates the nams, but does not set them

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-get_consecutive_newname_list_via_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> ibs._clean_species()
        >>> # execute function
        >>> imgsetid = None
        >>> new_nid_list, new_name_list = get_consecutive_newname_list_via_species(ibs, imgsetid=imgsetid)
        >>> result = ut.list_str((new_nid_list, new_name_list))
        >>> # verify results
        >>> print(result)
        (
            [1, 2, 3, 4, 5, 6, 7],
            ['IBEIS_PZ_0001', 'IBEIS_PZ_0002', 'IBEIS_UNKNOWN_0001', 'IBEIS_UNKNOWN_0002', 'IBEIS_GZ_0001', 'IBEIS_PB_0001', 'IBEIS_UNKNOWN_0003'],
        )

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> ibs._clean_species()
        >>> ibs.delete_all_imagesets()
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> # execute function
        >>> imgsetid = ibs.get_valid_imgsetids()[1]
        >>> new_nid_list, new_name_list = get_consecutive_newname_list_via_species(ibs, imgsetid=imgsetid)
        >>> result = ut.list_str((new_nid_list, new_name_list))
        >>> # verify results
        >>> print(result)
        (
            [4, 5, 6, 7],
            ['IBEIS_UNKNOWN_Occurrence_1_0001', 'IBEIS_GZ_Occurrence_1_0001', 'IBEIS_PB_Occurrence_1_0001', 'IBEIS_UNKNOWN_Occurrence_1_0002'],
        )
    """
    print('[ibs] get_consecutive_newname_list_via_species')
    ibs.delete_empty_nids()
    nid_list = ibs.get_valid_nids(imgsetid=imgsetid)
    #name_list = ibs.get_name_texts(nid_list)
    aids_list = ibs.get_name_aids(nid_list)
    species_rowids_list = ibs.unflat_map(ibs.get_annot_species_rowids, aids_list)
    unique_species_rowids_list = list(map(ut.unique_ordered, species_rowids_list))
    code_list = ibs.get_species_codes(ut.flatten(unique_species_rowids_list))

    _code2_count = ut.ddict(lambda: 0)
    def get_next_index(code):
        _code2_count[code] += 1
        return _code2_count[code]

    location_text = ibs.cfg.other_cfg.location_for_names
    if imgsetid is not None:
        imgset_text = ibs.get_imageset_text(imgsetid)
        imgset_text = imgset_text.replace(' ', '_').replace('\'', '').replace('"', '')
        new_name_list = [
            '%s_%s_%s_%04d' % (location_text, code, imgset_text, get_next_index(code))
            for code in code_list]
    else:
        new_name_list = [
            '%s_%s_%04d' % (location_text, code, get_next_index(code))
            for code in code_list]
    new_nid_list = nid_list
    assert len(new_nid_list) == len(new_name_list)
    return new_nid_list, new_name_list


@register_ibs_method
def set_annot_names_to_same_new_name(ibs, aid_list):
    new_nid = ibs.make_next_nids(num=1)[0]
    if ut.VERBOSE:
        print('Setting aid_list={aid_list} to have new_nid={new_nid}'.format(
            aid_list=aid_list, new_nid=new_nid))
    ibs.set_annot_name_rowids(aid_list, [new_nid] * len(aid_list))


@register_ibs_method
def set_annot_names_to_different_new_names(ibs, aid_list):
    new_nid_list = ibs.make_next_nids(num=len(aid_list))
    if ut.VERBOSE:
        print('Setting aid_list={aid_list} to have new_nid_list={new_nid_list}'.format(
            aid_list=aid_list, new_nid_list=new_nid_list))
    ibs.set_annot_name_rowids(aid_list, new_nid_list)


@register_ibs_method
def make_next_nids(ibs, *args, **kwargs):
    """
    makes name and adds it to the database returning the newly added name rowid(s)

    CAUTION; changes database state

    SeeAlso:
        make_next_name
    """
    next_names = ibs.make_next_name(*args, **kwargs)
    next_nids  = ibs.add_names(next_names)
    return next_nids


@register_ibs_method
def make_next_name(ibs, num=None, str_format=2, species_text=None, location_text=None):
    """ Creates a number of names which are not in the database, but does not
    add them

    Args:
        ibs (IBEISController):  ibeis controller object
        num (None):
        str_format (int): either 1 or 2

    Returns:
        str: next_name

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-make_next_name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs1 = ibeis.opendb('testdb1')
        >>> ibs2 = ibeis.opendb('PZ_MTEST')
        >>> ibs3 = ibeis.opendb('NAUT_test')
        >>> #ibs5 = ibeis.opendb('GIR_Tanya')
        >>> ibs1._clean_species()
        >>> ibs2._clean_species()
        >>> ibs3._clean_species()
        >>> num = None
        >>> str_format = 2
        >>> # execute function
        >>> next_name1 = make_next_name(ibs1, num, str_format)
        >>> next_name2 = make_next_name(ibs2, num, str_format)
        >>> next_name3 = make_next_name(ibs3, num, str_format)
        >>> next_name4 = make_next_name(ibs1, num, str_format, const.TEST_SPECIES.ZEB_GREVY)
        >>> name_list = [next_name1, next_name2, next_name3, next_name4]
        >>> next_name_list1 = make_next_name(ibs2, 5, str_format)
        >>> temp_nids = ibs2.add_names(['IBEIS_PZ_0045', 'IBEIS_PZ_0048'])
        >>> next_name_list2 = make_next_name(ibs2, 5, str_format)
        >>> ibs2.delete_names(temp_nids)
        >>> next_name_list3 = make_next_name(ibs2, 5, str_format)
        >>> # verify results
        >>> # FIXME: nautiluses are not working right
        >>> result = ut.list_str((name_list, next_name_list1, next_name_list2, next_name_list3))
        >>> print(result)
        (
            ['IBEIS_UNKNOWN_0008', 'IBEIS_PZ_0042', 'IBEIS_UNKNOWN_0004', 'IBEIS_GZ_0008'],
            ['IBEIS_PZ_0042', 'IBEIS_PZ_0043', 'IBEIS_PZ_0044', 'IBEIS_PZ_0045', 'IBEIS_PZ_0046'],
            ['IBEIS_PZ_0044', 'IBEIS_PZ_0046', 'IBEIS_PZ_0047', 'IBEIS_PZ_0049', 'IBEIS_PZ_0050'],
            ['IBEIS_PZ_0042', 'IBEIS_PZ_0043', 'IBEIS_PZ_0044', 'IBEIS_PZ_0045', 'IBEIS_PZ_0046'],
        )

    """
    # HACK TO FORCE TIMESTAMPS FOR NEW NAMES
    #str_format = 1
    if species_text is None:
        # TODO: optionally specify qreq_ or qparams?
        species_text  = ibs.cfg.detect_cfg.species_text
    if location_text is None:
        location_text = ibs.cfg.other_cfg.location_for_names
    if num is None:
        num_ = 1
    else:
        num_ = num
    nid_list = ibs._get_all_known_name_rowids()
    names_used_list = set(ibs.get_name_texts(nid_list))
    base_index = len(nid_list)
    next_name_list = []
    while len(next_name_list) < num_:
        base_index += 1
        if str_format == 1:
            userid = ut.get_user_name()
            timestamp = ut.get_timestamp('tag')
            #timestamp_suffix = '_TMP_'
            timestamp_suffix = '_'
            timestamp_prefix = ''
            name_prefix = timestamp_prefix + timestamp + timestamp_suffix + userid + '_'
        elif str_format == 2:
            species_rowid = ibs.get_species_rowids_from_text(species_text)
            species_code = ibs.get_species_codes(species_rowid)
            name_prefix = location_text + '_' + species_code + '_'
        else:
            raise ValueError('Invalid str_format supplied')
        next_name = name_prefix + '%04d' % base_index
        if next_name not in names_used_list:
            #names_used_list.add(next_name)
            next_name_list.append(next_name)
    # Return a list or a string
    if num is None:
        return next_name_list[0]
    else:
        return next_name_list


@register_ibs_method
def group_annots_by_name(ibs, aid_list, distinguish_unknowns=True):
    r"""
    This function is probably the fastest of its siblings

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):
        distinguish_unknowns (bool):

    Returns:
        tuple: grouped_aids_, unique_nids

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-group_annots_by_name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> distinguish_unknowns = True
        >>> # execute function
        >>> grouped_aids_, unique_nids = group_annots_by_name(ibs, aid_list, distinguish_unknowns)
        >>> result = str([aids.tolist() for aids in grouped_aids_])
        >>> result += '\n' + str(unique_nids.tolist())
        >>> # verify results
        >>> print(result)
        [[11], [9], [4], [1], [2, 3], [5, 6], [7], [8], [10], [12], [13]]
        [-11, -9, -4, -1, 1, 2, 3, 4, 5, 6, 7]
    """
    import vtool as vt
    nid_list = np.array(
        ibs.get_annot_name_rowids(
            aid_list, distinguish_unknowns=distinguish_unknowns))
    unique_nids, groupxs_list = vt.group_indices(nid_list)
    grouped_aids_ = vt.apply_grouping(np.array(aid_list), groupxs_list)
    return grouped_aids_, unique_nids


def group_annots_by_known_names_nochecks(ibs, aid_list):
    nid_list = ibs.get_annot_name_rowids(aid_list)
    nid2_aids = ut.group_items(aid_list, nid_list)
    return list(nid2_aids.values())


@register_ibs_method
def group_annots_by_known_names(ibs, aid_list, checks=True):
    r"""
    FIXME; rectify this
    #>>> import ibeis  # NOQA

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-group_annots_by_known_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(db='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        >>> known_aids_list, unknown_aids = group_annots_by_known_names(ibs, aid_list)
        >>> result = str(known_aids_list) + '\n'
        >>> result += str(unknown_aids)
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
        ibs (IBEISController):  ibeis controller object
        species (?):

    Returns:
        str: primary_viewpoint

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_primary_species_viewpoint

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> species = ibeis.const.TEST_SPECIES.ZEB_PLAIN
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
        primary_viewpoint = get_extended_viewpoints(primary_viewpoint, num1=1,
                                                    num2=0,
                                                    include_base=False)[0]
    return primary_viewpoint


def get_extended_viewpoints(base_yaw_text, towards='front', num1=0,
                            num2=None, include_base=True):
    """
    Given a viewpoint returns the acceptable viewpoints around it

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
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
    ori1 = const.VIEWTEXT_TO_YAW_RADIANS[base_yaw_text]
    ori2 = const.VIEWTEXT_TO_YAW_RADIANS[towards]
    # Find which direction to go to get closer to `towards`
    yawdist = vt.signed_ori_distance(ori1, ori2)
    if yawdist == 0:
        # break ties
        print('warning extending viewpoint yaws from the same position as towards')
        yawdist += 1E-3
    if num1 is None:
        num1 = 0
    if num2 is None:
        num2 = num1
    assert num1 >= 0, 'must specify positive num'
    assert num2 >= 0, 'must specify positive num'
    yawtext_list = list(const.VIEWTEXT_TO_YAW_RADIANS.keys())
    index = yawtext_list.index(base_yaw_text)
    other_index_list1 = [int((index + (np.sign(yawdist) * count)) %
                             len(yawtext_list))
                         for count in range(1, num1 + 1)]
    other_index_list2 = [int((index - (np.sign(yawdist) * count)) %
                             len(yawtext_list))
                         for count in range(1, num2 + 1)]
    if include_base:
        extended_index_list = sorted(list(set(other_index_list1 + other_index_list2 + [index])))
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
        python -m ibeis.other.ibsfuncs --test-get_two_annots_per_name_and_singletons
        python -m ibeis.other.ibsfuncs --test-get_two_annots_per_name_and_singletons --db GZ_ALL
        python -m ibeis.other.ibsfuncs --test-get_two_annots_per_name_and_singletons --db PZ_Master0 --onlygt

    Ignore:
        sys.argv.extend(['--db', 'PZ_MTEST'])

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master0')
        >>> aid_subset = get_two_annots_per_name_and_singletons(ibs, onlygt=ut.get_argflag('--onlygt'))
        >>> ibeis.other.dbinfo.get_dbinfo(ibs, aid_list=aid_subset, with_contrib=False)
        >>> result = str(aid_subset)
        >>> print(result)
    """
    species = get_primary_database_species(ibs, ibs.get_valid_aids())
    #aid_list = ibs.get_valid_aids(species='zebra_plains', is_known=True)
    aid_list = ibs.get_valid_aids(species=species, is_known=True)
    # FILTER OUT UNUSABLE ANNOTATIONS
    # Get annots with timestamps
    aid_list = filter_aids_without_timestamps(ibs, aid_list)
    minqual = 'ok'
    #valid_yaws = {'left', 'frontleft', 'backleft'}
    if species == 'zebra_plains':
        valid_yawtexts = {'left', 'frontleft'}
    elif species == 'zebra_grevys':
        valid_yawtexts = {'right', 'frontright'}
    else:
        valid_yawtexts = {'left', 'frontleft'}
    flags_list = ibs.get_quality_viewpoint_filterflags(aid_list, minqual, valid_yawtexts)
    aid_list = ut.compress(aid_list, flags_list)
    #print('print subset info')
    #print(ut.dict_hist(ibs.get_annot_yaw_texts(aid_list)))
    #print(ut.dict_hist(ibs.get_annot_quality_texts(aid_list)))
    singletons, multitons = partition_annots_into_singleton_multiton(ibs, aid_list)
    # process multitons
    hourdists_list = ibs.get_unflat_annots_hourdists_list(multitons)
    #pairxs_list = [vt.pdist_argsort(x) for x in hourdists_list]
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
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_num_annots_per_name
        python -m ibeis.other.ibsfuncs --exec-get_num_annots_per_name --db PZ_Master1

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids(is_known=True)
        >>> num_annots_per_name, unique_nids = get_num_annots_per_name(ibs, aid_list)
        >>> per_name_hist = ut.dict_hist(num_annots_per_name)
        >>> items = per_name_hist.items()
        >>> items = sorted(items)[::-1]
        >>> key_list = ut.get_list_column(items, 0)
        >>> val_list = ut.get_list_column(items, 1)
        >>> min_per_name = dict(zip(key_list, np.cumsum(val_list)))
        >>> result = ('per_name_hist = %s' % (ut.dict_str(per_name_hist),))
        >>> print(result)
        >>> print('min_per_name = %s' % (ut.dict_str(min_per_name),))
        per_name_hist = {
            1: 5,
            2: 2,
        }

    """
    aids_list, unique_nids  = ibs.group_annots_by_name(aid_list)
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
        python -m ibeis.other.ibsfuncs --test-get_yaw_viewtexts

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
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import numpy as np
        >>> # build test data
        >>> yaw_list = [0.0, np.pi / 2, np.pi / 4, np.pi, 3.15, -.4, -8, .2, 4, 7, 20, None]
        >>> # execute function
        >>> text_list = get_yaw_viewtexts(yaw_list)
        >>> result = ut.list_str(text_list, nl=False)
        >>> # verify results
        >>> print(result)
        ['right', 'front', 'frontright', 'left', 'left', 'backright', 'back', 'right', 'backleft', 'frontright', 'frontright', None]

    """
    #import vtool as vt
    import numpy as np
    import six
    stdlblyaw_list = list(six.iteritems(const.VIEWTEXT_TO_YAW_RADIANS))
    stdlbl_list = ut.get_list_column(stdlblyaw_list, 0)
    ALTERNATE = False
    if ALTERNATE:
        #with ut.Timer('fdsa'):
        TAU = np.pi * 2
        binsize = TAU / len(const.VIEWTEXT_TO_YAW_RADIANS)
        yaw_list_ = np.array([np.nan if yaw is None else yaw for yaw in yaw_list])
        index_list = np.floor(.5 + (yaw_list_ % TAU) / binsize)
        text_list = [None if np.isnan(index) else stdlbl_list[int(index)] for index in index_list]
    else:
        #with ut.Timer('fdsa'):
        stdyaw_list = np.array(ut.get_list_column(stdlblyaw_list, 1))
        textdists_list = [None if yaw is None else
                          vt.ori_distance(stdyaw_list, yaw)
                          for yaw in yaw_list]
        index_list = [None if dists is None else dists.argmin()
                      for dists in textdists_list]
        text_list = [None if index is None else stdlbl_list[index] for index in index_list]
        #yaw_list_ / binsize
    #errors = ['%.2f' % dists[index] for dists, index in zip(textdists_list, index_list)]
    #return list(zip(yaw_list, errors, text_list))
    return text_list


def get_species_dbs(species_prefix):
    from ibeis.init import sysres
    ibs_dblist = sysres.get_ibsdb_list()
    isvalid_list = [split(path)[1].startswith(species_prefix) for path in ibs_dblist]
    return ut.compress(ibs_dblist, isvalid_list)


@register_ibs_method
def get_annot_bbox_area(ibs, aid_list):
    bbox_list = ibs.get_annot_bboxes(aid_list)
    area_list = [bbox[2] * bbox[3] for bbox in bbox_list]
    return area_list


@register_ibs_method
def get_match_text(ibs, aid1, aid2):
    truth = ibs.get_match_truth(aid1, aid2)
    text = const.TRUTH_INT_TO_TEXT.get(truth, None)
    return text


@register_ibs_method
def get_database_species(ibs, aid_list=None):
    r"""

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-get_database_species

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> result = ut.list_str(ibs.get_database_species(), nl=False)
        >>> print(result)
        ['____', 'bear_polar', 'zebra_grevys', 'zebra_plains']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> result = ut.list_str(ibs.get_database_species(), nl=False)
        >>> print(result)
        ['zebra_plains']
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    species_list = ibs.get_annot_species_texts(aid_list)
    unique_species = sorted(list(set(species_list)))
    return unique_species


@register_ibs_method
def get_primary_database_species(ibs, aid_list=None):
    r"""
    Args:
        aid_list (list):  list of annotation ids (default = None)

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-get_primary_database_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = None
        >>> primary_species = get_primary_database_species(ibs, aid_list)
        >>> result = primary_species
        >>> print('primary_species = %r' % (primary_species,))
        >>> print(result)
        zebra_plains
    """
    SPEED_HACK = True
    if SPEED_HACK:
        if ibs.get_dbname() == 'PZ_MTEST':
            return 'zebra_plains'
        elif ibs.get_dbname() == 'PZ_Master0':
            return 'zebra_plains'
        elif ibs.get_dbname() == 'NNP_Master':
            return 'zebra_plains'
        elif ibs.get_dbname() == 'GZ_ALL':
            return 'zebra_grevys'
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    species_list = ibs.get_annot_species_texts(aid_list)
    species_hist = ut.dict_hist(species_list)
    if len(species_hist) == 0:
        primary_species = const.UNKNOWN
    else:
        frequent_species = sorted(species_hist.items(), key=lambda item: item[1], reverse=True)
        primary_species = frequent_species[0][0]
    return primary_species


@register_ibs_method
def get_dominant_species(ibs, aid_list):
    r"""
    Args:
        aid_list (int):  list of annotation ids

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-get_dominant_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> result = get_dominant_species(ibs, aid_list)
        >>> print(result)
        zebra_plains
    """
    hist_ = ut.dict_hist(ibs.get_annot_species_texts(aid_list))
    keys = hist_.keys()
    vals = hist_.values()
    species_text = keys[ut.list_argmax(vals)]
    return species_text


@register_ibs_method
def get_database_species_count(ibs, aid_list=None):
    """

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-get_database_species_count

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> #print(ut.dict_str(ibeis.opendb('PZ_Master0').get_database_species_count()))
        >>> ibs = ibeis.opendb('testdb1')
        >>> result = ut.dict_str(ibs.get_database_species_count(), nl=False)
        >>> print(result)
        {'____': 3, 'bear_polar': 2, 'zebra_grevys': 2, 'zebra_plains': 6}

    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    species_list = ibs.get_annot_species_texts(aid_list)
    species_count_dict = ut.item_hist(species_list)
    return species_count_dict


@register_ibs_method
def get_match_truth(ibs, aid1, aid2):
    nid1, nid2 = ibs.get_annot_name_rowids((aid1, aid2))
    isunknown_list = ibs.is_nid_unknown((nid1, nid2))
    if any(isunknown_list):
        truth = 2  # Unknown
    elif nid1 == nid2:
        truth = 1  # True
    elif nid1 != nid2:
        truth = 0  # False
    else:
        raise AssertionError('invalid_unknown_truth_state')
    return truth


@register_ibs_method
def get_aidpair_truths(ibs, aid1_list, aid2_list):
    r"""
    Uses NIDS to verify truth

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1_list (list):
        aid2_list (list):

    Returns:
        list[bool]: truth

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-get_aidpair_truths

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid1_list = ibs.get_valid_aids()
        >>> aid2_list = ut.list_roll(ibs.get_valid_aids(), -1)
        >>> # execute function
        >>> truth = get_aidpair_truths(ibs, aid1_list, aid2_list)
        >>> # verify results
        >>> result = str(truth)
        >>> print(result)
    """
    nid1_list = np.array(ibs.get_annot_name_rowids(aid1_list))
    nid2_list = np.array(ibs.get_annot_name_rowids(aid2_list))
    isunknown1_list = np.array(ibs.is_nid_unknown(nid1_list))
    isunknown2_list = np.array(ibs.is_nid_unknown(nid2_list))
    any_unknown = np.logical_or(isunknown1_list, isunknown2_list)
    truth_list = np.array((nid1_list == nid2_list), dtype=np.int32)
    truth_list[any_unknown] = const.TRUTH_UNKNOWN
    return truth_list


def get_title(ibs):
    if ibs is None:
        title = 'IBEIS - No Database Directory Open'
    elif ibs.dbdir is None:
        title = 'IBEIS - !! INVALID DATABASE !!'
    else:
        dbdir = ibs.get_dbdir()
        dbname = ibs.get_dbname()
        title = 'IBEIS - %r - Database Directory = %s' % (dbname, dbdir)
        wb_target = ibs.const.WILDBOOK_TARGET
        #params.args.wildbook_target
        if wb_target is not None:
            title = '%s - Wildbook Target = %s' % (title, wb_target)
    return title


@register_ibs_method
def get_dbinfo_str(ibs):
    from ibeis.other import dbinfo
    return dbinfo.get_dbinfo(ibs, verbose=False)['info_str']


@register_ibs_method
def get_infostr(ibs):
    """ Returns sort printable database information

    Args:
        ibs (IBEISController):  ibeis controller object

    Returns:
        str: infostr
    """
    from ibeis.other import dbinfo
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
    import ibeis
    assert isinstance(ibs, ibeis.control.IBEISControl.IBEISController)
    ut.write_to(ibs.get_dbnotes_fpath(), notes)


@register_ibs_method
def annotstr(ibs, aid):
    return 'aid=%d' % aid


@register_ibs_method
def merge_names(ibs, merge_name, other_names):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        merge_name (str):
        other_names (list):

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-merge_names

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> merge_name = 'zebra'
        >>> other_names = ['occl', 'jeff']
        >>> # execute function
        >>> result = merge_names(ibs, merge_name, other_names)
        >>> # verify results
        >>> print(result)
        >>> ibs.print_names_table()
    """
    print('[ibsfuncs] merging other_names=%r into merge_name=%r' %
            (other_names, merge_name))
    other_nid_list = ibs.get_name_rowids_from_text(other_names)
    ibs.set_name_alias_texts(other_nid_list, [merge_name] * len(other_nid_list))
    other_aids_list = ibs.get_name_aids(other_nid_list)
    other_aids = ut.flatten(other_aids_list)
    print('[ibsfuncs] ... %r annotations are being merged into merge_name=%r' %
            (len(other_aids), merge_name))
    ibs.set_annot_names(other_aids, [merge_name] * len(other_aids))


def inspect_nonzero_yaws(ibs):
    """
    python dev.py --dbdir /raid/work2/Turk/PZ_Master --cmd --show
    """
    from ibeis.viz import viz_chip
    import plottool as pt
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
def set_exemplars_from_quality_and_viewpoint(ibs, aid_list=None,
                                             exemplars_per_view=None, imgsetid=None,
                                             dry_run=False, verbose=True, prog_hook=None):
    """
    Automatic exemplar selection algorithm based on viewpoint and quality

    Ignore:
        # We want to choose the minimum per-item weight w such that
        # we can't pack more than N w's into the knapsack
        w * (N + 1) > N
        # and w < 1.0, so we can have wiggle room for preferences
        # so
        w * (N + 1) > N
        w > N / (N + 1)
        EPS = 1E-9
        w = N / (N + 1) + EPS

        # Preference denomiantor should not make any choice of
        # feasible items infeasible, but give more weight to a few.
        # delta_w is the wiggle room we have, but we need to choose a number
        # much less than it.
        prefdenom = N ** 2
        maybe its just N + EPS?
        N ** 2 should work though. Figure out correct value later
        delta_w = (1 - w)
        prefdenom = delta_w / N
        N - (w * N)

        N = 3
        EPS = 1E-9
        w = N / (N + 1) + EPS
        pref_decimator = N ** 2
        num_teir1_levels = 3
        pref_teir1 = w / (num_teir1_levels * pref_decimator)
        pref_teir2 = pref_teir1 / pref_decimator
        pref_teir3 = pref_teir2 / pref_decimator

    References:
        # implement maximum diversity approximation instead
        http://www.csbio.unc.edu/mcmillan/pubs/ICDM07_Pan.pdf

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-set_exemplars_from_quality_and_viewpoint
        python -m ibeis.other.ibsfuncs --test-set_exemplars_from_quality_and_viewpoint:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> #ibs = ibeis.opendb('PZ_MUGU_19')
        >>> ibs = ibeis.opendb('PZ_MTEST')
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
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> dry_run = True
        >>> verbose = False
        >>> old_sum = sum(ibs.get_annot_exemplar_flags(ibs.get_valid_aids()))
        >>> new_flag_list = ibs.set_exemplars_from_quality_and_viewpoint(dry_run=dry_run)
        >>> # 2 of the 11 annots are unknown and should not be exemplars
        >>> ut.assert_eq(sum(new_flag_list), 9)

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb2')
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
    if exemplars_per_view is None:
        exemplars_per_view = ibs.cfg.other_cfg.exemplars_per_view
    if aid_list is None:
        aid_list = ibs.get_valid_aids(imgsetid=imgsetid)
    HACK = ibs.cfg.other_cfg.enable_custom_filter
    assert not HACK, 'enable_custom_filter is no longer supported'
    new_flag_list = get_annot_quality_viewpoint_subset(
        ibs, aid_list=aid_list, annots_per_view=exemplars_per_view,
        verbose=verbose, prog_hook=prog_hook)

    # Hack ensure each name has at least 1 exemplar
    #if False:
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
def get_annot_quality_viewpoint_subset(ibs, aid_list=None, annots_per_view=2,
                                       max_annots=None, verbose=False,
                                       prog_hook=None, allow_unknown=False):
    """
    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_annot_quality_viewpoint_subset --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ut.exec_funckw(get_annot_quality_viewpoint_subset, globals())
        >>> ibs = ibeis.opendb('testdb2')
        >>> new_flag_list = get_annot_quality_viewpoint_subset(ibs)
        >>> result = sum(new_flag_list)
        >>> print(result)
        38

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ut.exec_funckw(get_annot_quality_viewpoint_subset, globals())
        >>> ibs = ibeis.opendb('testdb1')
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
        const.QUAL_EXCELLENT : 30,
        const.QUAL_GOOD      : 20,
        const.QUAL_OK        : 10,
        const.QUAL_UNKNOWN   : 0,
        const.QUAL_POOR      : -30,
        const.QUAL_JUNK      : -INF,
    }

    # Value of previously being an exemplar
    oldexemp2_value = {
        True:  0,
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
        qual_value     = np.array(ut.take(qual2_value, ibs.get_annot_quality_texts(aids)))
        oldexemp_value = np.array(ut.take(oldexemp2_value, ibs.get_annot_exemplar_flags(aids)))
        ismulti_value  = np.array(ut.take(ismulti2_value, ibs.get_annot_multiple(aids)))
        base_value = 1
        values = qual_value + oldexemp_value + ismulti_value + base_value

        # Build input for knapsack
        weights = [1] * len(values)
        indices = list(range(len(weights)))
        values = np.round(values, 3).tolist()
        items = list(zip(values, weights, indices))

        # Greedy version is fine if all weights are 1, just pick the N maximum values
        total_value, chosen_items = ut.knapsack_greedy(items, maxweight=annots_per_view)
        #try:
        #    total_value, chosen_items = ut.knapsack(items, annots_per_view, method='recursive')
        #except Exception:
        #    print('WARNING: iterative method does not work correctly, but stack too big for recrusive')
        #    total_value, chosen_items = ut.knapsack(items, annots_per_view, method='iterative')

        chosen_indices = ut.get_list_column(chosen_items, 2)
        flags = [False] * len(aids)
        for index in chosen_indices:
            flags[index] = True
        return flags

    nid_list = np.array(ibs.get_annot_name_rowids(aid_list, distinguish_unknowns=True))
    unique_nids, groupxs_list = vt.group_indices(nid_list)
    grouped_aids_ = vt.apply_grouping(np.array(aid_list), groupxs_list)
    #aids = grouped_aids_[-6]
    # for final settings because I'm too lazy to write
    new_aid_list = []
    new_flag_list = []
    _iter = ut.ProgIter(zip(grouped_aids_, unique_nids),
                        nTotal=len(unique_nids),
                        #freq=100,
                        lbl='Picking best annots per viewpoint',
                        prog_hook=prog_hook)
    for aids_, nid in _iter:
        if not allow_unknown and ibs.is_nid_unknown(nid):
            # do not change unknown animals
            new_aid_list.extend(aids_)
            new_flag_list.extend([False] * len(aids_))
        else:
            # subgroup the names by viewpoints
            yawtexts  = ibs.get_annot_yaw_texts(aids_)
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


def detect_join_cases(ibs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    Returns:
        QueryResult: qres_list -  object of feature correspondences and scores

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-detect_join_cases --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> # execute function
        >>> cm_list = detect_join_cases(ibs)
        >>> # verify results
        >>> #result = str(qres_list)
        >>> #print(result)
        >>> ut.quit_if_noshow()
        >>> import guitool
        >>> from ibeis.gui import inspect_gui
        >>> guitool.ensure_qapp()
        >>> qres_wgt = inspect_gui.QueryResultsWidget(ibs, cm_list, qreq_=qreq_, review_cfg=dict(filter_reviewed=False))
        >>> qres_wgt.show()
        >>> qres_wgt.raise_()
        >>> guitool.qtapp_loop(qres_wgt)
    """
    qaids = ibs.get_valid_aids(is_exemplar=None, minqual='poor')
    daids = ibs.get_valid_aids(is_exemplar=None, minqual='poor')
    cfgdict = dict(can_match_samename=False, use_k_padding=True)
    qreq_ = ibs.new_query_request(qaids, daids, cfgdict)
    cm_list = qreq_.execute()
    return cm_list
    #return qres_list


def _split_car_contrib_tag(contrib_tag, distinguish_invalids=True):
        if contrib_tag is not None and 'NNP GZC Car' in contrib_tag:
            contrib_tag_split = contrib_tag.strip().split(',')
            if len(contrib_tag_split) == 2:
                contrib_tag = contrib_tag_split[0].strip()
        elif distinguish_invalids:
            contrib_tag = None
        return contrib_tag


@register_ibs_method
def report_sightings(ibs, complete=True, include_images=False, **kwargs):
    def sanitize_list(data_list):
        data_list = [ str(data).replace(',', '<COMMA>') for data in list(data_list) ]
        return_str = (','.join(data_list))
        return_str = return_str.replace(',None,', ',NONE,')
        return_str = return_str.replace(',%s,' % (const.UNKNOWN, ) , ',UNKNOWN,')
        return_str = return_str.replace(',-1,', ',UNKNOWN,')
        return_str = return_str.replace(',-1,', ',UNKNOWN,')
        return_str = return_str.replace(',-1.0,', ',UNKNOWN,')
        return_str = return_str.replace(',-1.0,', ',UNKNOWN,')
        return return_str

    def construct():
        if complete:
            cols_list      = [
                ('annotation_id',        aid_list),
                ('annotation_xtl',       xtl_list),
                ('annotation_ytl',       ytl_list),
                ('annotation_width',     width_list),
                ('annotation_height',    height_list),
                ('annotation_species',   species_list),
                ('annotation_viewpoint', viewpoint_list),
                ('annotation_qualities', quality_list),
                ('annotation_sex',       sex_list),
                ('annotation_age_min',   age_min_list),
                ('annotation_age_max',   age_max_list),
                ('annotation_name',      name_list),
                ('image_id',             gid_list),
                ('image_contributor',    contrib_list),
                ('image_car',            car_list),
                ('image_filename',       uri_list),
                ('image_unixtime',       unixtime_list),
                ('image_time_str',       time_list),
                ('image_date_str',       date_list),
                ('image_lat',            lat_list),
                ('image_lon',            lon_list),
                ('flag_first_seen',      seen_list),
                ('flag_marked',          marked_list),
            ]
        else:
            cols_list      = [
                ('annotation_id',        aid_list),
                ('image_time_str',       time_list),
                ('image_date_str',       date_list),
                ('flag_first_seen',      seen_list),
                ('image_lat',            lat_list),
                ('image_lon',            lon_list),
                ('image_car',            car_list),
                ('annotation_age_min',   age_min_list),
                ('annotation_age_max',   age_max_list),
                ('annotation_sex',       sex_list),
            ]
        header_list    = [ sanitize_list([ cols[0] for cols in cols_list ]) ]
        data_list      = zip(*[ cols[1] for cols in cols_list ])
        line_list      = [ sanitize_list(data) for data in list(data_list) ]
        return header_list, line_list

    # Grab primitives
    if complete:
        aid_list   = ibs.get_valid_aids()
    else:
        aid_list   = ibs.filter_aids_count(pre_unixtime_sort=False)
    gid_list       = ibs.get_annot_gids(aid_list)
    bbox_list      = ibs.get_annot_bboxes(aid_list)
    xtl_list       = [ bbox[0] for bbox in bbox_list ]
    ytl_list       = [ bbox[1] for bbox in bbox_list ]
    width_list     = [ bbox[2] for bbox in bbox_list ]
    height_list    = [ bbox[3] for bbox in bbox_list ]
    species_list   = ibs.get_annot_species_texts(aid_list)
    viewpoint_list = ibs.get_annot_yaw_texts(aid_list)
    quality_list   = ibs.get_annot_quality_texts(aid_list)
    contrib_list   = ibs.get_image_contributor_tag(gid_list)
    car_list       = [ _split_car_contrib_tag(contrib_tag) for contrib_tag in contrib_list ]
    uri_list       = ibs.get_image_uris(gid_list)
    sex_list       = ibs.get_annot_sex_texts(aid_list)
    age_min_list   = ibs.get_annot_age_months_est_min(aid_list)
    age_max_list   = ibs.get_annot_age_months_est_max(aid_list)
    name_list      = ibs.get_annot_names(aid_list)
    unixtime_list  = ibs.get_image_unixtime(gid_list)
    datetime_list = [
        ut.unixtime_to_datetimestr(unixtime)
        if unixtime is not None else
        'UNKNOWN'
        for unixtime in unixtime_list
    ]
    datetime_split_list = [ datetime.split(' ') for datetime in datetime_list ]
    date_list      = [
        datetime_split[0] if len(datetime_split) == 2 else 'UNKNOWN'
        for datetime_split in datetime_split_list ]
    time_list      = [
        datetime_split[1] if len(datetime_split) == 2 else 'UNKNOWN'
        for datetime_split in datetime_split_list ]
    lat_list       = ibs.get_image_lat(gid_list)
    lon_list       = ibs.get_image_lon(gid_list)
    marked_list    = ibs.flag_aids_count(aid_list)
    seen_list      = []
    seen_set       = set()
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

        aid_list       = filler
        species_list   = filler
        viewpoint_list = filler
        quality_list   = filler
        sex_list       = filler
        age_min_list   = filler
        age_max_list   = filler
        name_list      = filler
        gid_list       = missing_gid_list
        contrib_list   = ibs.get_image_contributor_tag(missing_gid_list)
        car_list       = [ _split_car_contrib_tag(contrib_tag) for contrib_tag in contrib_list ]
        uri_list       = ibs.get_image_uris(missing_gid_list)
        unixtime_list  = ibs.get_image_unixtime(missing_gid_list)
        datetime_list = [
            ut.unixtime_to_datetimestr(unixtime)
            if unixtime is not None else
            'UNKNOWN'
            for unixtime in unixtime_list
        ]
        datetime_split_list = [ datetime.split(' ') for datetime in datetime_list ]
        date_list      = [
            datetime_split[0] if len(datetime_split) == 2 else 'UNKNOWN'
            for datetime_split in datetime_split_list ]
        time_list      = [
            datetime_split[1] if len(datetime_split) == 2 else 'UNKNOWN'
            for datetime_split in datetime_split_list ]
        lat_list       = ibs.get_image_lat(missing_gid_list)
        lon_list       = ibs.get_image_lon(missing_gid_list)
        seen_list      = filler
        marked_list    = filler

        header_list, line_list = construct()  # NOTE: discard the header list returned here
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
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        minqual (str): qualtext
        unknown_ok (bool): (default = False)

    Returns:
        iter: qual_flags

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_quality_filterflags

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:20]
        >>> minqual = 'junk'
        >>> unknown_ok = False
        >>> qual_flags = list(get_quality_filterflags(ibs, aid_list, minqual, unknown_ok))
        >>> result = ('qual_flags = %s' % (str(qual_flags),))
        >>> print(result)
    """
    minqual_int = const.QUALITY_TEXT_TO_INT[minqual]
    qual_int_list = ibs.get_annot_qualities(aid_list)
    #print('qual_int_list = %r' % (qual_int_list,))
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
def get_viewpoint_filterflags(ibs, aid_list, valid_yaws, unknown_ok=True):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        valid_yaws (?):
        unknown_ok (bool): (default = True)

    Returns:
        int: aid_list -  list of annotation ids

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_viewpoint_filterflags
        python -m ibeis.other.ibsfuncs --exec-get_viewpoint_filterflags --db NNP_Master3

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:20]
        >>> valid_yaws = None
        >>> unknown_ok = False
        >>> yaw_flags = list(get_viewpoint_filterflags(ibs, aid_list, valid_yaws, unknown_ok))
        >>> result = ('yaw_flags = %s' % (str(yaw_flags),))
        >>> print(result)
    """
    assert valid_yaws is None or isinstance(valid_yaws, (set, list, tuple)), 'valid_yaws is not a container'
    yaw_list = ibs.get_annot_yaw_texts(aid_list)
    if unknown_ok:
        yaw_flags  = (yaw is None or (valid_yaws is None or yaw in valid_yaws)
                      for yaw in yaw_list)
    else:
        yaw_flags  = (yaw is not None and (valid_yaws is None or yaw in valid_yaws)
                      for yaw in yaw_list)
    yaw_flags = list(yaw_flags)
    return yaw_flags


@register_ibs_method
def get_quality_viewpoint_filterflags(ibs, aid_list, minqual, valid_yaws):
    qual_flags = get_quality_filterflags(ibs, aid_list, minqual)
    yaw_flags = get_viewpoint_filterflags(ibs, aid_list, valid_yaws)
    #qual_list = ibs.get_annot_qualities(aid_list)
    #yaw_list = ibs.get_annot_yaw_texts(aid_list)
    #qual_flags = (qual is None or qual > minqual for qual in qual_list)
    #yaw_flags  = (yaw is None or yaw in valid_yaws for yaw in yaw_list)
    flags_list = list(ut.and_iters(qual_flags, yaw_flags))
    return flags_list


@register_ibs_method
def flag_aids_count(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        pre_unixtime_sort (bool):

    Returns:
        list:

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-flag_aids_count

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> # execute function
        >>> gzc_flag_list = flag_aids_count(ibs, aid_list)
        >>> result = gzc_flag_list
        >>> # verify results
        >>> print(result)
        [False, True, False, False, True, False, True, True, False, True, False, True, True]

    """
    # Get primitives
    unixtime_list  = ibs.get_annot_image_unixtimes(aid_list)
    index_list     = ut.list_argsort(unixtime_list)
    aid_list       = ut.sortedby(aid_list, unixtime_list)
    gid_list       = ibs.get_annot_gids(aid_list)
    nid_list       = ibs.get_annot_name_rowids(aid_list)
    contrib_list   = ibs.get_image_contributor_tag(gid_list)
    # Get filter flags for aids
    isunknown_list = ibs.is_aid_unknown(aid_list)
    flag_list      = [not unknown  for unknown in isunknown_list]
    # Filter by seen and car
    flag_list_     = []
    seen_dict      = ut.ddict(set)
    # Mark the first annotation (for each name) seen per car
    values_list    = zip(aid_list, gid_list, nid_list, flag_list, contrib_list)
    for aid, gid, nid, flag, contrib in values_list:
        if flag:
            contrib_ = _split_car_contrib_tag(contrib, distinguish_invalids=False)
            if nid not in seen_dict[contrib_]:
                seen_dict[contrib_].add(nid)
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
            aid_list      = ut.sortedby(aid_list, unixtime_list)
    flags_list = ibs.flag_aids_count(aid_list)
    aid_list_  = list(ut.iter_compress(aid_list, flags_list))
    return aid_list_


@register_ibs_method
@profile
def get_unflat_annots_kmdists_list(ibs, aids_list):
    #ibs.check_name_mapping_consistency(aids_list)
    latlons_list = ibs.unflat_map(ibs.get_annot_image_gps, aids_list)
    latlon_arrs   = [np.array(latlons) for latlons in latlons_list]
    for arrs in latlon_arrs:
        # our database encodes -1 as invalid.
        # Silly, but its in the middle of the atlantic ocean
        arrs[arrs == -1] = np.nan
    km_dists_list   = [ut.safe_pdist(latlon_arr, metric=vt.haversine) for latlon_arr in latlon_arrs]
    return km_dists_list


@register_ibs_method
@profile
def get_unflat_annots_hourdists_list(ibs, aids_list):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> ibs = testdata_ibs('testdb1')
        >>> nid_list = get_valid_multiton_nids_custom(ibs)
        >>> aids_list_ = ibs.get_name_aids(nid_list)
        >>> aids_list = [(aids) for aids in aids_list_]
        >>> ibs.get_unflat_annots_hourdists_list(aids_list)
    """
    #assert all(list(map(ut.isunique, aids_list)))
    unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, aids_list)
    #assert all(list(map(ut.isunique, unixtimes_list)))
    unixtime_arrs = [np.array(unixtimes)[:, None] for unixtimes in unixtimes_list]
    hour_dists_list = [ut.safe_pdist(unixtime_arr, metric=ut.unixtime_hourdiff)
                       for unixtime_arr in unixtime_arrs]
    return hour_dists_list


@register_ibs_method
@profile
def get_unflat_annots_timedelta_list(ibs, aids_list):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> ibs = testdata_ibs('NNP_Master3')
        >>> nid_list = get_valid_multiton_nids_custom(ibs)
        >>> aids_list_ = ibs.get_name_aids(nid_list)
        >>> aids_list = [(aids) for aids in aids_list_]

    """
    #assert all(list(map(ut.isunique, aids_list)))
    unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, aids_list)
    #assert all(list(map(ut.isunique, unixtimes_list)))
    unixtime_arrs = [np.array(unixtimes)[:, None] for unixtimes in unixtimes_list]
    timedelta_list = [ut.safe_pdist(unixtime_arr, metric=ut.absdiff) for
                      unixtime_arr in unixtime_arrs]
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
        unique_gps = np.array([
            (np.nan if lat == -1 else lat, np.nan if lon == -1 else lon)
            for (lat, lon) in unique_gps])
        unique_gps = vt.atleast_nd(unique_gps, 2)
        if len(unique_gps) == 0:
            unique_gps.shape = (0, 2)
        # Find pairs that need comparison
        idx_pairs_list = [list(it.combinations(idxs, 2)) for idxs in idx_list]
        idx_pairs, cumsum = ut.invertible_flatten2(idx_pairs_list)
        idxs1 = ut.take_column(idx_pairs, 0)
        idxs2 = ut.take_column(idx_pairs, 1)
        # Find differences in time and space
        hour_dists = ut.unixtime_hourdiff(unique_unixtimes[idxs1], unique_unixtimes[idxs2])
        km_dists = vt.haversine(unique_gps[idxs1].T, unique_gps[idxs2].T)
        # Zero km-distances less than a small epsilon if the timediff is zero to
        # prevent infinity problems
        idxs = np.where(hour_dists == 0)[0]
        under_eps = km_dists[idxs] < .5
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
    #idx_list = [ut.take(aid_to_idx, aids) for aids in aids_list]
    # Lookup values in SQL only once
    unique_unixtimes = ibs.get_annot_image_unixtimes_asfloat(unique_aids)
    unique_gps = ibs.get_annot_image_gps(unique_aids)
    unique_gps = np.array([
        (np.nan if lat == -1 else lat, np.nan if lon == -1 else lon)
        for (lat, lon) in unique_gps])
    unique_gps = vt.atleast_nd(unique_gps, 2)
    if len(unique_gps) == 0:
        unique_gps.shape = (0, 2)
    # Find differences in time and space
    hour_dists = ut.unixtime_hourdiff(unique_unixtimes[idxs1], unique_unixtimes[idxs2])
    km_dists = vt.haversine(unique_gps[idxs1].T, unique_gps[idxs2].T)
    # Zero km-distances less than a small epsilon if the timediff is zero to
    # prevent infinity problems
    idxs = np.where(hour_dists == 0)[0]
    under_eps = km_dists[idxs] < .5
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
    #flat_ams = ut.filter_Nones(ams)


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
    km_dists_list   = ibs.get_unflat_annots_kmdists_list(aids_list)
    hour_dists_list = ibs.get_unflat_annots_hourdists_list(aids_list)

    #hour_dists_list = ut.replace_nones(hour_dists_list, [])
    #km_dists_list = ut.replace_nones(km_dists_list, [])
    #flat_hours, cumsum1 = np.array(ut.invertible_flatten2(hour_dists_list))
    #flat_hours = np.array(flat_hours)
    #flat_kms = np.array(ut.flatten(km_dists_list))

    def compute_speed(km_dists, hours_dists):
        if km_dists is None or hours_dists is None:
            return None
        # Zero km-distances less than a small epsilon if the timediff is zero to
        # prevent infinity problems
        idxs = np.where(hours_dists == 0)[0]
        under_eps = km_dists[idxs] < .5
        km_dists[idxs[under_eps]] = 0
        # Normal speed calculation
        speeds = km_dists / hours_dists
        # No movement over no time
        flags = np.logical_and(km_dists == 0, hours_dists == 0)
        speeds[flags] = 0
        return speeds

    speeds_list     = [
        compute_speed(km_dists, hours_dists)
        #vt.safe_div(km_dists, hours_dists)
        for km_dists, hours_dists in
        zip(km_dists_list, hour_dists_list)]
    return speeds_list


def testdata_ibs(defaultdb='testdb1'):
    import ibeis
    ibs = ibeis.opendb(defaultdb=defaultdb)
    return ibs


def get_valid_multiton_nids_custom(ibs):
    nid_list_ = ibs._get_all_known_nids()
    ismultiton_list = [len((aids)) > 1
                       for aids in ibs.get_name_aids(nid_list_)]
    nid_list = ut.compress(nid_list_, ismultiton_list)
    return nid_list


@register_ibs_method
def make_next_imageset_text(ibs):
    """
    Creates what the next imageset name would be but does not add it to the database

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-make_next_imageset_text

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> new_imagesettext = make_next_imageset_text(ibs)
        >>> result = new_imagesettext
        >>> print(result)
        New ImageSet 0
    """
    imgsetid_list = ibs.get_valid_imgsetids()
    old_imagesettext_list = ibs.get_imageset_text(imgsetid_list)
    new_imagesettext = ut.get_nonconflicting_string('New ImageSet %d', old_imagesettext_list)
    return new_imagesettext


@register_ibs_method
def add_next_imageset(ibs):
    """
    Adds a new imageset to the database
    """
    new_imagesettext = ibs.make_next_imageset_text()
    (new_imgsetid,) = ibs.add_imagesets([new_imagesettext])
    return new_imgsetid


@register_ibs_method
def create_new_imageset_from_images(ibs, gid_list, new_imgsetid=None):
    r"""
    Args:
        gid_list (list):

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-create_new_imageset_from_images

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[::2]
        >>> # execute function
        >>> new_imgsetid = create_new_imageset_from_images(ibs, gid_list)
        >>> # verify results
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
        python -m ibeis.other.ibsfuncs --test-create_new_imageset_from_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_nids()[0:2]
        >>> # execute function
        >>> new_imgsetid = ibs.create_new_imageset_from_names(nid_list)
        >>> # clean up
        >>> ibs.delete_imagesets(new_imgsetid)
        >>> # verify results
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
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        tuple: (src_ag_rowid, dst_ag_rowid) - source and dest annot groups

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-prepare_annotgroup_review

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> result = prepare_annotgroup_review(ibs, aid_list)
        >>> print(result)
    """
    # Build new names for source and dest annot groups
    all_annotgroup_rowid_list = ibs._get_all_annotgroup_rowids()
    all_annotgroup_text_list = ibs.get_annotgroup_text(all_annotgroup_rowid_list)
    new_grouptext_src = ut.get_nonconflicting_string('Source Group %d', all_annotgroup_text_list)
    all_annotgroup_text_list += [new_grouptext_src]
    new_grouptext_dst = ut.get_nonconflicting_string('Dest Group %d', all_annotgroup_text_list)
    # Add new empty groups
    annotgroup_text_list = [new_grouptext_src, new_grouptext_dst]
    annotgroup_uuid_list = list(map(ut.hashable_to_uuid, annotgroup_text_list))
    annotgroup_note_list = ['', '']
    src_ag_rowid, dst_ag_rowid = ibs.add_annotgroup(annotgroup_uuid_list,
                                                    annotgroup_text_list,
                                                    annotgroup_note_list)
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
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_Master0')
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
    valid_index_list, valid_match_list = ut.search_list(notes_list, pattern, flags=re.IGNORECASE)
    #[match.group() for match in valid_match_list]
    valid_aid_list = ut.take(aid_list, valid_index_list)
    return valid_aid_list


@register_ibs_method
def filter_aids_to_quality(ibs, aid_list, minqual, unknown_ok=True):
    qual_flags = list(ibs.get_quality_filterflags(aid_list, minqual, unknown_ok=unknown_ok))
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

    yaw_flags = list(ibs.get_viewpoint_filterflags(aid_list, valid_yaws, unknown_ok=unknown_ok))
    aid_list_ = ut.compress(aid_list, yaw_flags)
    return aid_list_


@register_ibs_method
def remove_aids_of_viewpoint(ibs, aid_list, invalid_yaws):
    """
    Removes aids that do not have a valid yaw

    TODO; rename to valid_viewpoint because this func uses category labels
    """
    notyaw_flags = list(ibs.get_viewpoint_filterflags(aid_list, invalid_yaws, unknown_ok=False))
    yaw_flags = ut.not_list(notyaw_flags)
    aid_list_ = ut.compress(aid_list, yaw_flags)
    return aid_list_


@register_ibs_method
def filter_aids_without_name(ibs, aid_list, invert=False):
    """
    Remove aids without names
    """
    if invert:
        flag_list = ibs.is_aid_unknown(aid_list)
    else:
        flag_list = ut.not_list(ibs.is_aid_unknown(aid_list))
    aid_list_ = ut.compress(aid_list, flag_list)
    return aid_list_


@register_ibs_method
def filter_annots_using_minimum_timedelta(ibs, aid_list, min_timedelta):
    r"""
    Uses a dynamic program to find the maximum number of annotations that are
    above the minimum timedelta requirement.

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (?):
        min_timedelta (?):

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-filter_annots_using_minimum_timedelta
        python -m ibeis.other.ibsfuncs --exec-filter_annots_using_minimum_timedelta --db PZ_Master1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> aid_list = ibs.filter_aids_without_timestamps(aid_list)
        >>> print('Before')
        >>> ibs.print_annot_stats(aid_list, min_name_hourdist=True)
        >>> min_timedelta = 60 * 60 * 24
        >>> filtered_aids = filter_annots_using_minimum_timedelta(ibs, aid_list, min_timedelta)
        >>> print('After')
        >>> ibs.print_annot_stats(filtered_aids, min_name_hourdist=True)
        >>> ut.quit_if_noshow()
        >>> ibeis.other.dbinfo.hackshow_names(ibs, aid_list)
        >>> ibeis.other.dbinfo.hackshow_names(ibs, filtered_aids)
        >>> ut.show_if_requested()
    """
    import vtool as vt
    #min_timedelta = 60 * 60 * 24
    #min_timedelta = 60 * 10
    grouped_aids = ibs.group_annots_by_name(aid_list)[0]
    unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, grouped_aids)
    # Find the maximum size subset such that all timedeltas are less than a given value
    chosen_idxs_list = [
        ut.maximin_distance_subset1d(unixtimes, min_thresh=min_timedelta)[0]
        for unixtimes in unixtimes_list]
    filtered_groups = vt.ziptake(grouped_aids, chosen_idxs_list)
    filtered_aids = ut.flatten(filtered_groups)
    if ut.DEBUG2:
        timedeltas = ibs.get_unflat_annots_timedelta_list(filtered_groups)
        min_timedeltas = np.array([np.nan if dists is None else
                                   np.nanmin(dists) for dists in timedeltas])
        min_name_timedelta_stats = ut.get_stats(min_timedeltas, use_nan=True)
        print('min_name_timedelta_stats = %s' % (ut.dict_str(min_name_timedelta_stats),))
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
def filter_aids_to_species(ibs, aid_list, species):
    """
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        species (?):

    Returns:
        list: aid_list_

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-filter_aids_to_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> species = ibeis.const.TEST_SPECIES.ZEB_GREVY
        >>> aid_list_ = filter_aids_to_species(ibs, aid_list, species)
        >>> result = 'aid_list_ = %r' % (aid_list_,)
        >>> print(result)
        aid_list_ = [9, 10]
    """
    species_rowid      = ibs.get_species_rowids_from_text(species)
    species_rowid_list = ibs.get_annot_species_rowids(aid_list)
    is_valid_species   = [sid == species_rowid for sid in species_rowid_list]
    aid_list_           = ut.compress(aid_list, is_valid_species)
    #flag_list = [species == species_text for species_text in ibs.get_annot_species(aid_list)]
    #aid_list_ = ut.compress(aid_list, flag_list)
    return aid_list_


@register_ibs_method
def partition_annots_into_singleton_multiton(ibs, aid_list):
    """
    aid_list = aid_list_
    """
    aids_list = ibs.group_annots_by_name(aid_list)[0]
    singletons = [aids for aids in aids_list if len(aids) == 1]
    multitons = [aids for aids in aids_list if len(aids) > 1]
    return singletons, multitons


@register_ibs_method
def partition_annots_into_corresponding_groups(ibs, aid_list1, aid_list2):
    """
    Used for grouping one-vs-one training pairs and corerspondence filtering

    Args:
        ibs (ibeis.control.IBEISControl.IBEISController):  ibeis controller object
        aid_list1 (int):  list of annotation ids
        aid_list2 (int):  list of annotation ids

    Returns:
        tuple: 4 lists of lists. In the first two each list is a list of aids
            grouped by names and the names correspond with each other. In the
            last two are the annots that did not correspond with anything in
            the other list.

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-partition_annots_into_corresponding_groups

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
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
        >>> result = ut.list_str(groups, label_list=['gt_grouped_aids1', 'gt_grouped_aids2', 'gf_grouped_aids1', 'gf_gropued_aids2'])
        >>> print(result)
        gt_grouped_aids1 = [[10, 11], [17, 18], [22, 23]]
        gt_grouped_aids2 = [[12, 13, 14, 15], [19, 20, 21], [24, 25, 26]]
        gf_grouped_aids1 = [[2], [5, 6]]
        gf_gropued_aids2 = [[29, 30, 31, 32], [49]]
    """
    #ibs.
    import ibeis.control.IBEISControl
    assert isinstance(ibs, ibeis.control.IBEISControl.IBEISController)
    #ibs
    #ibs.get_ann

    grouped_aids1 = ibs.group_annots_by_name(aid_list1)[0]
    grouped_aids1 = [aids.tolist() for aids in grouped_aids1]

    # Get the group of available aids that a reference aid could match
    gropued_aids2 = ibs.get_annot_groundtruth(
        ut.get_list_column(grouped_aids1, 0), daid_list=aid_list2,
        noself=False)

    # Flag if there is a correspondence
    flag_list = [x > 0 for x in map(len, gropued_aids2)]

    # Corresonding lists of aids groups
    gt_grouped_aids1 = ut.compress(grouped_aids1, flag_list)
    gt_grouped_aids2 = ut.compress(gropued_aids2, flag_list)

    # Non-corresponding lists of aids groups
    gf_grouped_aids1 = ut.compress(grouped_aids1, ut.not_list(flag_list))
    #gf_aids1 = ut.flatten(gf_grouped_aids1)
    gf_aids2 = ut.setdiff_ordered(aid_list2, ut.flatten(gt_grouped_aids2))
    gf_grouped_aids2 = [aids.tolist() for aids in
                        ibs.group_annots_by_name(gf_aids2)[0]]

    gt_grouped_aids1 = list(map(sorted, gt_grouped_aids1))
    gt_grouped_aids2 = list(map(sorted, gt_grouped_aids2))
    gf_grouped_aids1 = list(map(sorted, gf_grouped_aids1))
    gf_grouped_aids2 = list(map(sorted, gf_grouped_aids2))

    return gt_grouped_aids1, gt_grouped_aids2, gf_grouped_aids1, gf_grouped_aids2


@register_ibs_method
def dans_lists(ibs, positives=10, negatives=10, verbose=False):
    from random import shuffle

    aid_list = ibs.get_valid_aids()
    yaw_list = ibs.get_annot_yaw_texts(aid_list)
    qua_list = ibs.get_annot_quality_texts(aid_list)
    sex_list = ibs.get_annot_sex_texts(aid_list)
    age_list = ibs.get_annot_age_months_est(aid_list)

    positive_list = [
        aid
        for aid, yaw, qua, sex, (start, end) in
        zip(aid_list, yaw_list, qua_list, sex_list, age_list)
        if (
            yaw.upper() == 'LEFT' and
            qua.upper() in ['OK', 'GOOD', 'EXCELLENT'] and
            sex.upper() in ['MALE', 'FEMALE'] and
            start != -1 and end != -1
        )
    ]

    negative_list = [
        aid
        for aid, yaw, qua, sex, (start, end) in zip(aid_list, yaw_list,
                                                    qua_list, sex_list,
                                                    age_list)
        if (
            yaw.upper() == 'LEFT' and
            qua.upper() in ['OK', 'GOOD', 'EXCELLENT'] and
            sex.upper() == 'UNKNOWN SEX' and
            start == -1 and end == -1
        )
    ]

    shuffle(positive_list)
    shuffle(negative_list)

    positive_list = sorted(positive_list[:10])
    negative_list = sorted(negative_list[:10])

    if verbose:
        pos_yaw_list = ibs.get_annot_yaw_texts(positive_list)
        pos_qua_list = ibs.get_annot_quality_texts(positive_list)
        pos_sex_list = ibs.get_annot_sex_texts(positive_list)
        pos_age_list = ibs.get_annot_age_months_est(positive_list)
        pos_chip_list = ibs.get_annot_chip_fpath(positive_list)

        neg_yaw_list = ibs.get_annot_yaw_texts(negative_list)
        neg_qua_list = ibs.get_annot_quality_texts(negative_list)
        neg_sex_list = ibs.get_annot_sex_texts(negative_list)
        neg_age_list = ibs.get_annot_age_months_est(negative_list)
        neg_chip_list = ibs.get_annot_chip_fpath(negative_list)

        print('positive_aid_list = %s\n' % (positive_list, ))
        print('positive_yaw_list = %s\n' % (pos_yaw_list, ))
        print('positive_qua_list = %s\n' % (pos_qua_list, ))
        print('positive_sex_list = %s\n' % (pos_sex_list, ))
        print('positive_age_list = %s\n' % (pos_age_list, ))
        print('positive_chip_list = %s\n' % (pos_chip_list, ))

        print('-' * 90, '\n')

        print('negative_aid_list = %s\n' % (negative_list, ))
        print('negative_yaw_list = %s\n' % (neg_yaw_list, ))
        print('negative_qua_list = %s\n' % (neg_qua_list, ))
        print('negative_sex_list = %s\n' % (neg_sex_list, ))
        print('negative_age_list = %s\n' % (neg_age_list, ))
        print('negative_chip_list = %s\n' % (neg_chip_list, ))

        print('mkdir ~/Desktop/chips')
        for pos_chip in pos_chip_list:
            print('cp "%s" ~/Desktop/chips/' % (pos_chip, ))
        for neg_chip in neg_chip_list:
            print('cp "%s" ~/Desktop/chips/' % (neg_chip, ))

    return positive_list, negative_list


def _stat_str(dict_, multi=False, precision=2, **kwargs):
    import utool as ut
    dict_ = dict_.copy()
    if dict_.get('num_nan', None) == 0:
        del dict_['num_nan']
    exclude_keys = []  # ['std', 'nMin', 'nMax']
    if multi is True:
        str_ = ut.dict_str(dict_, precision=precision, nl=2, strvals=True)
    else:
        str_ =  ut.get_stats_str(stat_dict=dict_, precision=precision,
                                 exclude_keys=exclude_keys, **kwargs)
    str_ = str_.replace('\'', '')
    str_ = str_.replace('num_nan: 0, ', '')
    return str_


@register_ibs_method
def make_property_stats_dict(ibs, aid_list, prop_getter, key_order=None):
    annot_prop_list = prop_getter(aid_list)
    prop2_aids = ut.group_items(aid_list, annot_prop_list)
    if key_order is None:
        key_order = set(annot_prop_list)
    assert set(key_order) >= set(annot_prop_list), (
        'bad keys: ' + str(set(annot_prop_list) - set(key_order)))
    prop2_nAnnots = ut.odict([
        (key, len(prop2_aids.get(key, []))) for key in key_order])
    # Filter 0's
    prop2_nAnnots = ut.odict([
        (key, val) for key, val in prop2_nAnnots.items() if val != 0])
    return prop2_nAnnots


# Quality and Viewpoint Stats
@register_ibs_method
def get_annot_qual_stats(ibs, aid_list):
    key_order = list(const.QUALITY_TEXT_TO_INT.keys())
    prop_getter = ibs.get_annot_quality_texts
    qualtext2_nAnnots = ibs.make_property_stats_dict(aid_list, prop_getter, key_order)
    # annot_qualtext_list = ibs.get_annot_quality_texts(aid_list)
    # qualtext2_aids = ut.group_items(aid_list, annot_qualtext_list)
    # qual_keys = list(const.QUALITY_TEXT_TO_INT.keys())
    # assert set(qual_keys) >= set(qualtext2_aids), (
    #     'bad keys: ' + str(set(qualtext2_aids) - set(qual_keys)))
    # qualtext2_nAnnots = ut.odict([(key, len(qualtext2_aids.get(key, []))) for key in qual_keys])
    # # Filter 0's
    # qualtext2_nAnnots = {key: val for key, val in six.iteritems(qualtext2_nAnnots) if val != 0}
    return qualtext2_nAnnots


@register_ibs_method
def get_annot_yaw_stats(ibs, aid_list):
    key_order = list(const.VIEWTEXT_TO_YAW_RADIANS.keys()) + [None]
    prop_getter = ibs.get_annot_yaw_texts
    yawtext2_nAnnots = ibs.make_property_stats_dict(aid_list, prop_getter, key_order)
    # annot_yawtext_list = ibs.get_annot_yaw_texts(aid_list)
    # yawtext2_aids = ut.group_items(aid_list, annot_yawtext_list)
    #  Order keys
    # yaw_keys = list(const.VIEWTEXT_TO_YAW_RADIANS.keys()) + [None]
    # assert set(yaw_keys) >= set(annot_yawtext_list), (
    #     'bad keys: ' + str(set(annot_yawtext_list) - set(yaw_keys)))
    # yawtext2_nAnnots = ut.odict(
    #     [(key, len(yawtext2_aids.get(key, []))) for key in yaw_keys])
    # # Filter 0's
    # yawtext2_nAnnots = {
    #     const.YAWALIAS.get(key, key): val
    #     for key, val in six.iteritems(yawtext2_nAnnots) if val != 0
    # }
    # yawtext2_nAnnots = {key: val for key, val in six.iteritems(yawtext2_nAnnots) if val != 0}
    return yawtext2_nAnnots


@register_ibs_method
def get_annot_intermediate_viewpoint_stats(ibs, aids, size=2):
    """
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> aids = available_aids
    """
    getter_func = ibs.get_annot_yaw_texts
    prop_basis = list(const.VIEWTEXT_TO_YAW_RADIANS.keys())

    group_annots_by_view_and_name = functools.partial(
        ibs.group_annots_by_prop_and_name, getter_func=getter_func)
    group_annots_by_view = functools.partial(ibs.group_annots_by_prop,
                                             getter_func=getter_func)

    prop2_nid2_aids = group_annots_by_view_and_name(aids)

    edge2_nid2_aids = group_prop_edges(prop2_nid2_aids, prop_basis, size=size, wrap=True)
    # Total number of names that have two viewpoints
    #yawtext_edge_nid_hist = ut.map_dict_vals(len, edge2_nid2_aids)
    edge2_grouped_aids = ut.map_dict_vals(lambda dict_: list(dict_.values()), edge2_nid2_aids)
    edge2_aids = ut.map_dict_vals(ut.flatten, edge2_grouped_aids)
    # Num annots of each type of viewpoint
    #yawtext_edge_aidyawtext_hist = ut.map_dict_vals(ut.dict_hist,
    #ut.map_dict_vals(getter_func, yawtext_edge_aid))

    # Regroup by view and name
    edge2_vp2_pername_stats = {}
    edge2_vp2_aids = ut.map_dict_vals(group_annots_by_view, edge2_aids)
    for edge, vp2_aids in edge2_vp2_aids.items():
        vp2_pernam_stats = ut.map_dict_vals(
            functools.partial(ibs.get_annots_per_name_stats, use_sum=True), vp2_aids)
        edge2_vp2_pername_stats[edge] = vp2_pernam_stats

    #yawtext_edge_numaids_hist = ut.map_dict_vals(len, yawtext_edge_aid)
    #yawtext_edge_aid_viewpoint_stats_hist =
    #ut.map_dict_vals(ibs.get_annot_yaw_stats, yawtext_edge_aid)
    return edge2_vp2_pername_stats


@register_ibs_method
def group_annots_by_name_dict(ibs, aids):
    grouped_aids, nids = ibs.group_annots_by_name(aids)
    return dict(zip(nids, map(list, grouped_aids)))


@register_ibs_method
def group_annots_by_prop(ibs, aids, getter_func):
    # Make a dictionary that maps props into a dictionary of names to aids
    annot_prop_list = getter_func(aids)
    prop2_aids = ut.group_items(aids, annot_prop_list)
    return prop2_aids


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
        ibs (IBEISController):  ibeis controller object
        aids (list):  list of annotation rowids
        getter_list (list):

    Returns:
        dict: multiprop2_aids

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-group_annots_by_multi_prop --db PZ_Master1 --props=yaw_texts,name_rowids --keys1 frontleft
        python -m ibeis.other.ibsfuncs --exec-group_annots_by_multi_prop

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids(is_known=True)
        >>> #getter_list = [ibs.get_annot_name_rowids, ibs.get_annot_yaw_texts]
        >>> props = ut.get_argval('--props', type_=list, default=['yaw_texts', 'name_rowids'])
        >>> getter_list = [getattr(ibs, 'get_annot_' + prop) for prop in props]
        >>> print('getter_list = %r' % (getter_list,))
        >>> #getter_list = [ibs.get_annot_yaw_texts, ibs.get_annot_name_rowids]
        >>> multiprop2_aids = group_annots_by_multi_prop(ibs, aids, getter_list)
        >>> get_dict_values = lambda x: list(x.values())
        >>> # a bit convoluted
        >>> keys1 = ut.get_argval('--keys1', type_=list, default=list(multiprop2_aids.keys()))
        >>> multiprop2_num_aids = ut.hmap_vals(len, multiprop2_aids)
        >>> prop2_num_aids = ut.hmap_vals(get_dict_values, multiprop2_num_aids, max_depth=len(props) - 2)
        >>> #prop2_num_aids_stats = ut.hmap_vals(ut.get_stats, prop2_num_aids)
        >>> prop2_num_aids_hist = ut.hmap_vals(ut.dict_hist, prop2_num_aids)
        >>> prop2_num_aids_cumhist = ut.map_dict_vals(ut.dict_hist_cumsum, prop2_num_aids_hist)
        >>> print('prop2_num_aids_hist[%s] = %s' % (keys1,  ut.dict_str(ut.dict_subset(prop2_num_aids_hist, keys1))))
        >>> print('prop2_num_aids_cumhist[%s] = %s' % (keys1,  ut.dict_str(ut.dict_subset(prop2_num_aids_cumhist, keys1))))

    """
    aid_prop_list = [getter(aids) for getter in getter_list]
    #%timeit multiprop2_aids = ut.hierarchical_group_items(aids, aid_prop_list)
    #%timeit ut.group_items(aids, list(zip(*aid_prop_list)))
    multiprop2_aids = ut.hierarchical_group_items(aids, aid_prop_list)
    #multiprop2_aids = ut.group_items(aids, list(zip(*aid_prop_list)))
    return multiprop2_aids


def group_prop_edges(prop2_nid2_aids, prop_basis, size=2, wrap=True):
    """
    from ibeis.other.ibsfuncs import *  # NOQA
    getter_func = ibs.get_annot_yaw_texts
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
            #functools.partial(ut.dict_intersection, combine=True),
            ut.dict_isect_combine,
            edge_nid2_aids_list)
        edge2_nid2_aids[edge] = isect_nid2_aids
        #common_nids = list(isect_nid2_aids.keys())
        #common_num_prop1 = np.array([len(prop2_nid2_aids[prop1][nid]) for nid in common_nids])
        #common_num_prop2 = np.array([len(prop2_nid2_aids[prop2][nid]) for nid in common_nids])
    return edge2_nid2_aids


@register_ibs_method
def parse_annot_stats_filter_kws(ibs):
    kwkeys = ut.parse_func_kwarg_keys(ibs.get_annot_stats_dict)
    return kwkeys


# Indepdentent query / database stats
@register_ibs_method
def get_annot_stats_dict(ibs, aids, prefix='', forceall=False, old=True, **kwargs):
    """ stats for a set of annots

    Args:
        ibs (ibeis.IBEISController):  ibeis controller object
        aids (list):  list of annotation rowids
        prefix (str): (default = '')

    Kwargs:
        hashid, per_name, per_qual, per_vp, per_name_vpedge, per_image, min_name_hourdist

    Returns:
        dict: aid_stats_dict

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_annot_stats_dict
        python -m ibeis.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_Master1 --per_name_vpedge=True
        python -m ibeis.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_Master1 --min_name_hourdist=True
        python -m ibeis.other.ibsfuncs --exec-get_annot_stats_dict --db GZ_ALL --min_name_hourdist=True --all
        python -m ibeis.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_Master1 --min_name_hourdist=True --all
        python -m ibeis.other.ibsfuncs --exec-get_annot_stats_dict --db NNP_MasterGIRM_core --min_name_hourdist=True --all

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids()
        >>> prefix = ''
        >>> kwkeys = ut.parse_func_kwarg_keys(get_annot_stats_dict)
        >>> #default = True if ut.get_argflag('--all') else None
        >>> default = None if ut.get_argflag('--notall') else True
        >>> kwargs = ut.argparse_dict(dict(zip(kwkeys, [default] * len(kwkeys))))
        >>> print('kwargs = %r' % (kwargs,))
        >>> aid_stats_dict = get_annot_stats_dict(ibs, aids, prefix, **kwargs)
        >>> result = ('aid_stats_dict = %s' % (ut.dict_str(aid_stats_dict, strvals=True, nl=True),))
        >>> print(result)
    """
    kwargs = kwargs.copy()

    statwrap = _stat_str if old else ut.identity

    def get_per_prop_stats(ibs, aids, getter_func):
        prop2_aids = ibs.group_annots_by_prop(aids, getter_func=getter_func)
        num_None = 0
        if None in prop2_aids:
            # Dont count invalid properties
            num_None += len(prop2_aids[None])
            del prop2_aids[None]
        if '____' in prop2_aids:
            num_None += len(prop2_aids['____'])
            del prop2_aids['____']
        num_aids_list = list(map(len, prop2_aids.values()))
        num_aids_stats = ut.get_stats(num_aids_list, use_nan=True, use_median=True)
        # represent that Nones were removed
        # num_aids_list += ([np.nan] * num_None)
        # if num_None:
        num_aids_stats['num_None'] = num_None
        return num_aids_stats

    keyval_list = [
        ('num_' + prefix + 'aids', len(aids)),
    ]

    if kwargs.pop('bigstr', False or forceall):
        bigstr = functools.partial(ut.truncate_str, maxlen=64, truncmsg=' ~TRUNC~ ')
        keyval_list += [
            (prefix + 'bigstr',
             bigstr(str(aids)))]

    if kwargs.pop('hashid', True or forceall):
        keyval_list += [
            (prefix + 'hashid',
             ibs.get_annot_hashid_semantic_uuid(aids, prefix=prefix.upper()))]

    if kwargs.pop('hashid_visual', False or forceall):
        keyval_list += [
            (prefix + 'hashid_visual',
             ibs.get_annot_hashid_visual_uuid(aids, prefix=prefix.upper()))]

    if kwargs.pop('hashid_uuid', False or forceall):
        keyval_list += [
            (prefix + 'hashid_uuid',
             ibs.get_annot_hashid_uuid(aids, prefix=prefix.upper()))]

    if kwargs.pop('per_name', True or forceall):
        keyval_list += [
            (prefix + 'per_name',
             statwrap(ut.get_stats(ibs.get_num_annots_per_name(aids)[0],
                                    use_nan=True, use_median=True)))]

    # if kwargs.pop('per_name_dict', True or forceall):
    #     keyval_list += [
    #         (prefix + 'per_name_dict',
    #          ut.get_stats(ibs.get_num_annots_per_name(aids)[0],
    #                                 use_nan=True, use_median=True))]

    if kwargs.pop('per_qual', False or forceall):
        keyval_list += [(prefix + 'per_qual',
                         statwrap(ibs.get_annot_qual_stats(aids)))]

    #if kwargs.pop('per_vp', False):
    if kwargs.pop('per_vp', True or forceall):
        keyval_list += [(prefix + 'per_vp',
                         statwrap(ibs.get_annot_yaw_stats(aids)))]

    if kwargs.pop('per_multiple', True or forceall):
        keyval_list += [(prefix + 'per_multiple',
                         statwrap(
                             ibs.make_property_stats_dict(aids, ibs.get_annot_multiple)))]

    # information about overlapping viewpoints
    if kwargs.pop('per_name_vpedge', False or forceall):
        keyval_list += [
            (prefix + 'per_name_vpedge',
             statwrap(ibs.get_annot_intermediate_viewpoint_stats(aids), multi=True))]

    if kwargs.pop('enc_per_name', True or forceall):
        # Does not handle None encounters. They show up as just another encounter
        name2_enctexts = ut.group_items(ibs.get_annot_encounter_text(aids), ibs.get_annot_names(aids))
        encsets = ut.lmap(set, name2_enctexts.values())
        nEncs_per_name = list(map(len, encsets))
        keyval_list += [
            (prefix + 'enc_per_name',
             statwrap(ut.get_stats(nEncs_per_name, use_nan=True, use_median=True))
             )]

    if kwargs.pop('per_enc', True or forceall):
        keyval_list += [
            (prefix + 'per_enc',
             statwrap(get_per_prop_stats(ibs, aids, ibs.get_annot_encounter_text)))]

    if kwargs.pop('per_image', False or forceall):
        keyval_list += [
            (prefix + 'aid_per_image',
             statwrap(get_per_prop_stats(ibs, aids, ibs.get_annot_image_rowids)))]

    if kwargs.pop('species_hist', False or forceall):
        keyval_list += [
            (prefix + 'species_hist',
             ut.dict_hist(ibs.get_annot_species_texts(aids)))
        ]

    if kwargs.pop('case_tag_hist', False or forceall):
        keyval_list += [
            (prefix + 'case_tags', ut.dict_hist(ut.flatten(ibs.get_annot_case_tags(aids))))]

    if kwargs.pop('match_tag_hist', False or forceall):
        keyval_list += [
            (prefix + 'match_tags', ut.dict_hist(ut.flatten(ibs.get_annot_annotmatch_tags(aids))))]

    if kwargs.pop('min_name_hourdist', False or forceall):
        grouped_aids = ibs.group_annots_by_name(aids)[0]
        #ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, grouped_aids)
        timedeltas = ibs.get_unflat_annots_timedelta_list(grouped_aids)
        #timedeltas = [dists for dists in timedeltas if dists is not np.nan and
        #dists is not None]
        #timedeltas = [np.nan if dists is None else dists for dists in timedeltas]
        #min_timedelta_list = [np.nan if dists is None else dists.min() / (60 *
        #60 * 24) for dists in timedeltas]
        # convert to hours
        min_timedelta_list = [
            np.nan if dists is None else dists.min() / (60 * 60)
            for dists in timedeltas]
        #min_timedelta_list = [np.nan if dists is None else dists.min() for dists in timedeltas]
        min_name_timedelta_stats = ut.get_stats(min_timedelta_list, use_nan=True)
        keyval_list += [(prefix + 'min_name_hourdist', _stat_str(min_name_timedelta_stats, precision=4))]

    aid_stats_dict = ut.odict(keyval_list)
    return aid_stats_dict


@register_ibs_method
def print_annot_stats(ibs, aids, prefix='', label='', **kwargs):
    aid_stats_dict = ibs.get_annot_stats_dict(aids, prefix=prefix, **kwargs)
    print(label + ut.dict_str(aid_stats_dict, strvals=True))


@register_ibs_method
def compare_nested_props(ibs, aids1_list,
                         aids2_list, getter_func,
                         cmp_func):
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
        ibs (IBEISController):  ibeis controller object
        aids1_list (list):
        aids2_list (list):
        getter_func (?):
        cmp_func (?):

    Returns:
        list of ndarrays:

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-compare_nested_props --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aids1_list = [ibs.get_valid_aids()[8:11]]
        >>> aids2_list = [ibs.get_valid_aids()[8:11]]
        >>> getter_func = ibs.get_annot_image_unixtimes_asfloat
        >>> cmp_func = ut.unixtime_hourdiff
        >>> result = compare_nested_props(ibs, aids1_list, aids2_list, getter_func, cmp_func)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
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
    TAU = np.pi * 2
    ori_diff = vt.ori_distance(ori1, ori2)
    viewpoint_diff = len(const.VIEWTEXT_TO_YAW_RADIANS) * ori_diff / TAU
    return viewpoint_diff


@register_ibs_method
def get_annot_per_name_stats(ibs, aid_list):
    """  stats about this set of aids """
    return ut.get_stats(ibs.get_num_annots_per_name(aid_list)[0], use_nan=True)


@register_ibs_method
def print_annotconfig_stats(ibs, qaids, daids, **kwargs):
    """
    SeeAlso:
        ibs.get_annotconfig_stats
    """
    ibs.get_annotconfig_stats(qaids, daids, verbose=True, **kwargs)


@register_ibs_method
def parse_annot_config_stats_filter_kws(ibs):
    #kwkeys = ibs.parse_annot_stats_filter_kws() + ['combined', 'combo_gt_info', 'combo_enc_info', 'combo_dists']
    kwkeys1 = ibs.parse_annot_stats_filter_kws()
    kwkeys2 = list(ut.get_func_kwargs(ibs.get_annotconfig_stats).keys())
    if 'verbose' in kwkeys2:
        kwkeys2.remove('verbose')
    kwkeys = ut.unique(kwkeys1 + kwkeys2)
    return kwkeys


@register_ibs_method
def get_annotconfig_stats(ibs, qaids, daids, verbose=True, combined=False,
                          combo_gt_info=True, combo_enc_info=True,
                          combo_dists=True,
                          **kwargs):
    r"""
    Gets statistics about a query / database set of annotations

    USEFUL DEVELOPER FUNCTION

    TODO: this function should return non-string values in dictionaries.
    The print function should do string conversions

    Args:
        ibs (IBEISController):  ibeis controller object
        qaids (list):  query annotation ids
        daids (list):  database annotation ids

    SeeAlso:
        ibeis.dbinfo.print_qd_info
        ibs.get_annot_stats_dict
        ibs.print_annotconfig_stats(qaid_list, daid_list)

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_annotconfig_stats --db PZ_MTEST -a default
        python -m ibeis.other.ibsfuncs --exec-get_annotconfig_stats --db testdb1  -a default
        python -m ibeis.other.ibsfuncs --exec-get_annotconfig_stats --db PZ_MTEST -a controlled
        python -m ibeis.other.ibsfuncs --exec-get_annotconfig_stats --db PZ_FlankHack -a default:qaids=allgt
        python -m ibeis.other.ibsfuncs --exec-get_annotconfig_stats --db PZ_MTEST -a controlled:per_name=2,min_gt=4

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> kwargs = {'per_enc': True, 'enc_per_name': True}
        >>> ibs, qaids, daids = main_helpers.testdata_expanded_aids(defaultdb='testdb1')
        >>> _ = get_annotconfig_stats(ibs, qaids, daids, **kwargs)
    """
    kwargs = kwargs.copy()
    import numpy as np
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) imageseted')
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice.')

        # The aids that should be matched by a query
        grouped_qaids = ibs.group_annots_by_name(qaids)[0]
        grouped_groundtruth_list = ibs.get_annot_groundtruth(
            ut.get_list_column(grouped_qaids, 0), daid_list=daids)
        groundtruth_daids = ut.unique_unordered(ut.flatten(grouped_groundtruth_list))
        hasgt_list = ibs.get_annot_has_groundtruth(qaids, daid_list=daids)
        # The aids that should not match any query
        nonquery_daids = np.setdiff1d(np.setdiff1d(daids, qaids), groundtruth_daids)
        # The query aids that should not get any match
        unmatchable_queries = ut.compress(qaids, ut.not_list(hasgt_list))
        # The query aids that should not have a match
        matchable_queries = ut.compress(qaids, hasgt_list)

        # Find the daids that are in the same occurrence as the qaids
        # if False:
        query_encs = set(ibs.get_annot_encounter_text(qaids))
        data_encs = set(ibs.get_annot_encounter_text(daids))
        enc_intersect = query_encs.intersection(data_encs)
        enc_intersect.difference_update({None})
        # import operator as op
        # compare_nested_props(
        #     ibs, grouped_qaids, grouped_groundtruth_list, ibs.get_annot_encounter_text, op.eq)

        # Intersection on a per name basis
        imposter_daid_per_name_stats = ibs.get_annot_per_name_stats(nonquery_daids)
        genuine_daid_per_name_stats  = ibs.get_annot_per_name_stats(groundtruth_daids)
        all_daid_per_name_stats = ibs.get_annot_per_name_stats(daids)
        #imposter_daid_per_name_stats =
        #ut.get_stats(ibs.get_num_annots_per_name(nonquery_daids)[0],
        #             use_nan=True)
        #genuine_daid_per_name_stats  =
        #ut.get_stats(ibs.get_num_annots_per_name(groundtruth_daids)[0],
        #             use_nan=True)
        #all_daid_per_name_stats = ut.get_stats(ibs.get_num_annots_per_name(daids)[0], use_nan=True)

        # Compare the query yaws to the yaws of its correct matches in the database
        # For each name there will be nQaids:nid x nDaids:nid comparisons
        gt_viewdist_list = compare_nested_props(
            ibs, grouped_qaids, grouped_groundtruth_list, ibs.get_annot_yaws, viewpoint_diff)

        # Compare the query qualities to the qualities of its correct matches in the database
        gt_qualdists_list = compare_nested_props(
            ibs, grouped_qaids, grouped_groundtruth_list, ibs.get_annot_qualities, ut.absdiff)

        # Compare timedelta differences
        gt_hourdelta_list = compare_nested_props(
            ibs, grouped_qaids, grouped_groundtruth_list,
            ibs.get_annot_image_unixtimes_asfloat, ut.unixtime_hourdiff)

        def super_flatten(arr_list):
            import utool as ut
            return ut.flatten([arr.ravel() for arr in arr_list])

        gt_viewdist_stats   = ut.get_stats(super_flatten(gt_viewdist_list), use_nan=True)
        gt_qualdist_stats  = ut.get_stats(super_flatten(gt_qualdists_list), use_nan=True)
        gt_hourdelta_stats = ut.get_stats(super_flatten(gt_hourdelta_list), use_nan=True)

        qaids2 = np.array(qaids).copy()
        daids2 = np.array(daids).copy()
        qaids2.sort()
        daids2.sort()
        if not np.all(qaids2 == qaids):
            print('WARNING: qaids are not sorted')
            #raise AssertionError('WARNING: qaids are not sorted')
        if not np.all(daids2 == daids):
            print('WARNING: daids are not sorted')
            #raise AssertionError('WARNING: qaids are not sorted')

        qaid_stats_dict = ibs.get_annot_stats_dict(qaids, 'q', **kwargs)
        daid_stats_dict = ibs.get_annot_stats_dict(daids, 'd', **kwargs)

        # Intersections between qaids and daids
        common_aids = np.intersect1d(daids, qaids)

        qnids = ut.unique_unordered(ibs.get_annot_name_rowids(qaids))
        dnids = ut.unique_unordered(ibs.get_annot_name_rowids(daids))
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
            ('daid_stats', daid_stats_dict),
        ]
        if combined:
            combined_aids = np.unique((np.hstack((qaids, daids))))
            combined_aids.sort()
            annotconfig_stats_strs_list1 += [
                ('combined_aids', ibs.get_annot_stats_dict(combined_aids,
                                                           **kwargs)),
            ]

        if combo_gt_info:
            annotconfig_stats_strs_list1 += [
                ('num_unmatchable_queries', len(unmatchable_queries)),
                ('num_matchable_queries', len(matchable_queries)),
            ]

        if combo_enc_info:
            annotconfig_stats_strs_list1 += [
                #('num_qnids', (len(qnids))),
                #('num_dnids', (len(dnids))),
                ('num_enc_intersect', (len(enc_intersect))),
                ('num_name_intersect', (len(common_nids))),
            ]

        if combo_gt_info:
            annotconfig_stats_strs_list2 += [
                # Number of aids per name for everything in the database
                #('per_name_', _stat_str(all_daid_per_name_stats)),
                # Number of aids in each name that should match to a query
                # (not quite sure how to phrase what this is)
                ('per_name_genuine', _stat_str(genuine_daid_per_name_stats)),
                # Number of aids in each name that should not match to any query
                # (not quite sure how to phrase what this is)
                ('per_name_imposter', _stat_str(imposter_daid_per_name_stats)),
            ]

        if combo_dists:
            annotconfig_stats_strs_list2 += [
                # Distances between a query and its groundtruth
                ('viewdist', _stat_str(gt_viewdist_stats)),
                #('qualdist', _stat_str(gt_qualdist_stats)),
                ('hourdist', _stat_str(gt_hourdelta_stats, precision=4)),
            ]

        annotconfig_stats_strs1 = ut.odict(annotconfig_stats_strs_list1)
        annotconfig_stats_strs2 = ut.odict(annotconfig_stats_strs_list2)

        annotconfig_stats_strs = ut.odict(list(annotconfig_stats_strs1.items()) +
                                          list(annotconfig_stats_strs2.items()))
        stats_str = ut.dict_str(annotconfig_stats_strs1, strvals=True,
                                newlines=False, explicit=True, nobraces=True)
        stats_str +=  '\n' + ut.dict_str(annotconfig_stats_strs2, strvals=True,
                                         newlines=True, explicit=True,
                                         nobraces=True)
        #stats_str = ut.align(stats_str, ':')
        stats_str2 = ut.dict_str(annotconfig_stats_strs, strvals=True,
                                 newlines=True, explicit=False, nobraces=False)
        if verbose:
            print('annot_config_stats = ' + stats_str2)

        return annotconfig_stats_strs, locals()


@register_ibs_method
def get_dbname_alias(ibs):
    """
    convinience for plots
    """
    dbname = ibs.get_dbname()
    return const.DBNAME_ALIAS.get(dbname, dbname)


@register_ibs_method
def find_unlabeled_name_members(ibs, **kwargs):
    r"""
    Find annots where some members of a name have information but others do not.

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-find_unlabeled_name_members --qual

    Example:
        >>> # SCRIPT
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> defaultdict = dict(ut.parse_func_kwarg_keys(find_unlabeled_name_members, with_vals=True))
        >>> kwargs = ut.argparse_dict(defaultdict)
        >>> result = find_unlabeled_name_members(ibs, **kwargs)
        >>> print(result)
    """
    aid_list = ibs.get_valid_aids()
    aids_list, nids = ibs.group_annots_by_name(aid_list)
    aids_list = ut.compress(aids_list, [len(aids) > 1 for aids in aids_list])

    def find_missing(props_list, flags_list):
        missing_idx_list = ut.list_where([any(flags) and not all(flags) for flags in flags_list])
        missing_flag_list = ut.take(flags_list, missing_idx_list)
        missing_aids_list = ut.take(aids_list, missing_idx_list)
        #missing_prop_list = ut.take(props_list, missing_idx_list)
        missing_aid_list = vt.zipcompress(missing_aids_list, missing_flag_list)

        if False:
            missing_percent_list = [sum(flags) / len(flags) for flags in missing_flag_list]
            print('Missing per name stats')
            print(ut.dict_str(ut.get_stats(missing_percent_list, use_median=True)))
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
        max_timedelta_list = np.array([
            np.nanmax(ut.safe_pdist(unixtime_arr[:, None], metric=ut.absdiff))
            for unixtime_arr in time_list])
        flags = max_timedelta_list > 60 * 60 * 1

        aids1 = ut.compress(aids_list, flags)
        max_yawdiff_list = np.array([
            np.nanmax(ut.safe_pdist(np.array(yaws)[:, None], metric=vt.ori_distance))
            for yaws in ut.compress(yaws_list, flags)
        ])

        # Find annots with large timedeltas but 0 viewpoint difference
        flags2 = max_yawdiff_list == 0
        selected_aids_list.append(ut.compress(aids1, flags2))

    x = ut.flatten(selected_aids_list)
    y = ut.sortedby2(x, list(map(len, x)))
    selected_aids = ut.unique_ordered(ut.flatten(y))
    return selected_aids

    #ibs.unflat_map(ibs.get_annot_quality_texts, aids_list)
    #ibs.unflat_map(ibs.get_annot_yaw_texts, aids_list)


@register_ibs_method
def get_annot_pair_lazy_dict(ibs, qaid, daid, qconfig2_=None, dconfig2_=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        qaid (int):  query annotation id
        daid (?):
        qconfig2_ (dict): (default = None)
        dconfig2_ (dict): (default = None)

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_annot_pair_lazy_dict

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> qaid, daid = ibs.get_valid_aids()[0:2]
        >>> qconfig2_ = None
        >>> dconfig2_ = None
        >>> result = get_annot_pair_lazy_dict(ibs, qaid, daid, qconfig2_, dconfig2_)
        >>> print(result)
    """
    metadata = ut.LazyDict({
        'annot1': get_annot_lazy_dict(ibs, qaid, config2_=qconfig2_),
        'annot2': get_annot_lazy_dict(ibs, daid, config2_=dconfig2_),
    }, reprkw=dict(truncate=True))
    return metadata


@register_ibs_method
def get_annot_lazy_dict(ibs, aid, config2_=None):
    r"""
    Args:
        ibs (ibeis.IBEISController):  image analysis api
        aid (int):  annotation id
        config2_ (dict): (default = None)

    Returns:
        ut.LazyDict: metadata

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_annot_lazy_dict --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid = 1
        >>> config2_ = None
        >>> metadata = get_annot_lazy_dict(ibs, aid, config2_)
        >>> result = ('metadata = %s' % (ut.repr3(metadata),))
        >>> print(result)
    """
    from ibeis.viz.interact import interact_chip
    metadata = ut.LazyDict({
        'aid': aid,
        'name': lambda: ibs.get_annot_names([aid])[0],
        'nid': lambda: ibs.get_annot_name_rowids([aid])[0],
        'rchip_fpath': lambda: ibs.get_annot_chip_fpath([aid], config2_=config2_)[0],
        'rchip': lambda: ibs.get_annot_chips([aid], config2_=config2_)[0],
        'vecs': lambda:  ibs.get_annot_vecs([aid], config2_=config2_)[0],
        'kpts': lambda:  ibs.get_annot_kpts([aid], config2_=config2_)[0],
        'dlen_sqrd': lambda: ibs.get_annot_chip_dlensqrd([aid], config2_=config2_)[0],
        'annot_context_options': lambda: interact_chip.build_annot_context_options(ibs, aid),
    }, reprkw=dict(truncate=True))
    return metadata


@register_ibs_method
def get_image_lazydict(ibs, gid, config=None):
    r"""
    Args:
        ibs (ibeis.IBEISController):  image analysis api
        aid (int):  annotation id
        config (dict): (default = None)

    Returns:
        ut.LazyDict: metadata

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_annot_lazy_dict2 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> gid = 1
    """
    #from ibeis.viz.interact import interact_chip
    metadata = ut.LazyDict({
        'gid': gid,
        'unixtime': lambda: ibs.get_image_unixtime(gid),
        'datetime': lambda: ibs.get_image_datetime(gid),
        'aids': lambda: ibs.get_image_aids(gid),
        'size': lambda: ibs.get_image_sizes(gid),
        'uri': lambda: ibs.get_image_uris(gid),
        'uuid': lambda: ibs.get_image_uuids(gid),
        'gps': lambda: ibs.get_image_gps(gid),
        'orientation': lambda: ibs.get_image_orientation(gid),
        #'annot_context_options': lambda: interact_chip.build_annot_context_options(ibs, aid),
    }, reprkw=dict(truncate=True))
    return metadata


@register_ibs_method
def get_image_instancelist(ibs, gid_list):
    obj_list = [ibs.get_image_lazydict(gid) for gid in gid_list]
    image_list = ut.make_instancelist(obj_list, check=False)
    return image_list


@register_ibs_method
def get_annot_instancelist(ibs, aid_list):
    obj_list = [ibs.get_annot_lazydict(aid) for aid in aid_list]
    annot_list = ut.make_instancelist(obj_list, check=False)
    return annot_list


@register_ibs_method
def get_annot_lazy_dict2(ibs, aid, config=None):
    r"""
    Args:
        ibs (ibeis.IBEISController):  image analysis api
        aid (int):  annotation id
        config (dict): (default = None)

    Returns:
        ut.LazyDict: metadata

    CommandLine:
        python -m ibeis.other.ibsfuncs --exec-get_annot_lazy_dict2 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid = 1
        >>> config = {'dim_size': 450}
        >>> metadata = get_annot_lazy_dict2(ibs, aid, config)
        >>> result = ('metadata = %s' % (ut.repr3(metadata),))
        >>> print(result)
    """
    from ibeis.viz.interact import interact_chip
    metadata = ut.LazyDict({
        'aid': aid,
        'name': lambda: ibs.get_annot_names(aid),
        'rchip_fpath': lambda: ibs.depc_annot.get('chips', aid, 'img', config, read_extern=False),
        'rchip': lambda: ibs.depc_annot.get('chips', aid, 'img', config),
        'vecs': lambda:  ibs.depc_annot.get('feat', aid, 'vecs', config),
        'kpts': lambda:  ibs.depc_annot.get('feat', aid, 'kpts', config),
        'dlen_sqrd': lambda: ibs.depc_annot['chips'].subproperties['dlen_sqrd'](
            ibs.depc_annot, [aid], config)[0],
        #get_annot_chip_dlensqrd([aid], config=config)[0],
        'annot_context_options': lambda: interact_chip.build_annot_context_options(ibs, aid),
    })
    return metadata


#@register_ibs_method
#def execute_pipeline_test(ibs, qaids, daids, pipecfg_name_list=['default']):
#    from ibeis.expt import harness, experiment_helpers
#    experiment_helpers
#    testnameid = ibs.get_dbname() + ' ' + str(pipecfg_name_list)
#    lbl = '[harn] TEST_CFG ' + str(pipecfg_name_list)

#    # Generate list of query pipeline param configs
#    cfgdict_list, pipecfg_list = experiment_helpers.get_pipecfg_list(
#        pipecfg_name_list, ibs=ibs)

#    cfgx2_lbl = experiment_helpers.get_varied_pipecfg_lbls(cfgdict_list)
#    testres = harness.run_test_configurations(
#        ibs, qaids, daids, pipecfg_list, cfgx2_lbl, cfgdict_list, lbl,
#        testnameid, use_cache=False)
#    return testres


#@register_ibs_method
#def get_imageset_expanded_aids(ibs, aid_list=None):
#    """
#    Example:
#        >>> import ibeis
#        >>> from ibeis.other.ibsfuncs import *  # NOQA
#        >>> ibs = ibeis.opendb(defaultdb='lynx')
#        >>> a = ['default:hack_imageset=True', ]
#        >>> from ibeis.expt import experiment_helpers
#        >>> acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, [a[0]], use_cache=False)
#        >>> aid_list = ibs.get_valid_aids()
#        >>> filter_kw = dict(been_adjusted=True)
#        >>> aid_list = ibs.filter_annots_general(aid_list, filter_kw)
#        >>> qaid_list, daid_list = ibs.get_imageset_expanded_aids()
#        >>> #ibs.query_chips(qaid_list, daid_list)
#        >>> testres = ibs.execute_pipeline_test(qaid_list, daid_list)
#        >>> testres.print_perent_identification_success()
#    """
#    if aid_list is None:
#        filter_kw = dict(been_adjusted=True)
#        aid_list = ibs.filter_annots_general(ibs.get_valid_aids(), filter_kw)
#    imgsetid_list = ibs.get_annot_primary_imageset(aid_list)
#    nid_list = ibs.get_annot_nids(aid_list)
#    multiprop2_aids = ut.hierarchical_group_items(aid_list, [nid_list, imgsetid_list])
#    daid_list = []
#    qaid_list = []
#    for imgsetid, nid2_aids in multiprop2_aids.iteritems():
#        if len(nid2_aids) == 1:
#            daid_list.extend(ut.flatten(list(nid2_aids.values())))
#        else:
#            aids_list = list(nid2_aids.values())
#            idx = ut.list_argmax(list(map(len, aids_list)))
#            qaids = aids_list[idx]
#            del aids_list[idx]
#            daids = ut.flatten(aids_list)
#            daid_list.extend(daids)
#            qaid_list.extend(qaids)
#    return qaid_list, daid_list


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

    #for annot in annots.values():
    #    annot.eager_eval('vecs')

    def extract_vecs(annots, aid, fxs):
        """ custom_func(lazydict, key, subkeys) for multigroup_lookup """
        vecs = annots[aid]['vecs'].take(fxs, axis=0)
        return vecs
    #unflat_vecs1 = vt.multigroup_lookup(annots, unflat_aids, unflat_fxs, extract_vecs)
    # HACK
    # FIXME: naive and regular multigroup still arnt equivalent
    #unflat_vecs = unflat_vecs1 = [[] if len(x) == 1 and x[0] is None else x  for x in unflat_vecs1]
    unflat_vecs =  unflat_vecs2 = vt.multigroup_lookup_naive(annots, unflat_aids, unflat_fxs, extract_vecs)  # NOQA
    # import utool
    # vt.sver_c_wrapper.asserteq(unflat_vecs1, unflat_vecs2)
    # unflat_vecs = unflat_vecs2
    # unflat_vecs = unflat_vecs1
    # import utool
    # utool.embed()

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
    species_mapping = {
        'bear_polar'          :       ('PB', 'Polar Bear'),
        'building'            : ('BUILDING', 'Building'),
        'cheetah'             :     ('CHTH', 'Cheetah'),
        'elephant_savanna'    :     ('ELEP', 'Elephant (Savanna)'),
        'frog'                :     ('FROG', 'Frog'),
        'giraffe_masai'       :     ('GIRM', 'Giraffe (Masai)'),
        'giraffe_reticulated' :      ('GIR', 'Giraffe (Reticulated)'),
        'hyena'               :    ('HYENA', 'Hyena'),
        'jaguar'              :      ('JAG', 'Jaguar'),
        'leopard'             :     ('LOEP', 'Leopard'),
        'lion'                :     ('LION', 'Lion'),
        'lionfish'            :       ('LF', 'Lionfish'),
        'lynx'                :     ('LYNX', 'Lynx'),
        'nautilus'            :     ('NAUT', 'Nautilus'),
        'other'               :    ('OTHER', 'Other'),
        'rhino_black'         :   ('BRHINO', 'Rhino (Black)'),
        'rhino_white'         :   ('WRHINO', 'Rhino (White)'),
        'seal_saimma_ringed'  :    ('SEAL2', 'Seal (Siamaa Ringed)'),
        'seal_spotted'        :    ('SEAL1', 'Seal (Spotted)'),
        'snail'               :    ('SNAIL', 'Snail'),
        'snow_leopard'        :    ('SLEOP', 'Snow Leopard'),
        'tiger'               :    ('TIGER', 'Tiger'),
        'toads_wyoming'       :   ('WYTOAD', 'Toad (Wyoming)'),
        'water_buffalo'       :     ('BUFF', 'Water Buffalo'),
        'wildebeest'          :       ('WB', 'Wildebeest'),
        'wild_dog'            :       ('WD', 'Wild Dog'),
        'whale_fluke'         :       ('WF', 'Whale Fluke'),
        'whale_humpback'      :       ('HW', 'Humpback Whale'),
        'whale_shark'         :       ('WS', 'Whale Shark'),
        'zebra_grevys'        :       ('GZ', 'Zebra (Grevy\'s)'),
        'zebra_hybrid'        :       ('HZ', 'Zebra (Hybrid)'),
        'zebra_plains'        :       ('PZ', 'Zebra (Plains)'),
        const.UNKNOWN         :  ('UNKNOWN', 'Unknown'),
    }
    if ibs.readonly:
        # SUPER HACK
        return
    if ibs is not None:
        flag = '--allow-keyboard-database-update'
        from six.moves import input as raw_input_
        from ibeis.control.manual_species_funcs import _convert_species_nice_to_code
        species_rowid_list = ibs._get_all_species_rowids()
        species_text_list = ibs.get_species_texts(species_rowid_list)
        species_nice_list = ibs.get_species_nice(species_rowid_list)
        species_code_list = ibs.get_species_codes(species_rowid_list)
        for rowid, text, nice, code in zip(species_rowid_list, species_text_list, species_nice_list, species_code_list):
            if text in species_mapping:
                species_code, species_nice = species_mapping[text]
            elif text is None or text.strip() in ['_', const.UNKNOWN, 'none', 'None', '']:
                print('[_clean_species] deleting species: %r' % (text, ))
                ibs.delete_species(rowid)
                continue
            elif len(nice) == 0:
                if not ut.get_argflag(flag):
                    species_nice = text
                    species_code = _convert_species_nice_to_code([species_nice])[0]
                else:
                    print('Found an unknown species: %r' % (text, ))
                    species_nice = raw_input_('Input a NICE name for %r: ' % (text, ))
                    species_code = raw_input_('Input a CODE name for %r: ' % (text, ))
                    assert len(species_code) > 0 and len(species_nice) > 0
            else:
                continue
            if nice != species_nice or code != species_code:
                ibs._set_species_nice([rowid], [species_nice])
                ibs._set_species_code([rowid], [species_code])


@register_ibs_method
def get_annot_encounter_text(ibs, aids):
    """ Encounter identifier for annotations """
    occur_texts = ibs.get_annot_occurrence_text(aids)
    name_texts = ibs.get_annot_names(aids)
    enc_texts = [ot + '_' + nt if ot is not None and nt is not None else None for ot, nt in zip(occur_texts, name_texts)]
    return enc_texts


@register_ibs_method
def get_annot_occurrence_text(ibs, aids):
    """ Occurrence identifier for annotations

    Args:
        ibs (ibeis.IBEISController):  image analysis api
        aids (list):  list of annotation rowids

    Returns:
        list: occur_texts

    CommandLine:
        python -m ibeis.other.ibsfuncs get_annot_occurrence_text --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids()
        >>> occur_texts = get_annot_occurrence_text(ibs, aids)
        >>> result = ('occur_texts = %s' % (ut.repr2(occur_texts),))
        >>> print(result)
    """
    imgset_ids = ibs.get_annot_imgsetids(aids)
    flags = ibs.unflat_map(ibs.get_imageset_isoccurrence, imgset_ids)
    #flags = [[text.lower().startswith('occurrence') for text in texts]
    #         for texts in imgset_texts]
    imgset_ids = ut.zipcompress(imgset_ids, flags)
    _occur_texts = ibs.unflat_map(ibs.get_imageset_text, imgset_ids)
    _occur_texts = [t if len(t) > 0 else [None] for t in _occur_texts]
    if not all([len(t) == 1 for t in _occur_texts]):
        print('WARNING: annot must be in exactly one occurrence')
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
    element = './/%swaypoints' % (namespace, )
    waypoint_list = patrol_tree.findall(element)
    if len(waypoint_list) == 0:
        # raise IOError('There are no observations (waypoints) in this
        # Patrol XML file: %r' % (xml_path, ))
        print('There are no observations (waypoints) in this Patrol XML file: %r' %
              (xml_path, ))
    for waypoint in waypoint_list:
        # Get the relevant information about the waypoint
        waypoint_id   = int(waypoint.get('id'))
        waypoint_lat  = float(waypoint.get('y'))
        waypoint_lon  = float(waypoint.get('x'))
        waypoint_time = waypoint.get('time')
        waypoint_info = [
            xml_name,
            waypoint_id,
            (waypoint_lat, waypoint_lon),
            waypoint_time,
        ]
        if None in waypoint_info:
            raise IOError(
                'The observation (waypoint) is missing information: %r' %
                (waypoint_info, ))
        # Get all of the waypoint's observations (we expect only one
        # normally)
        element = './/%sobservations' % (namespace, )
        observation_list = waypoint.findall(element)
        # if len(observation_list) == 0:
        #     raise IOError('There are no observations in this waypoint,
        #     waypoint_id: %r' % (waypoint_id, ))
        for observation in observation_list:
            # Filter the observations based on type, we only care
            # about certain types
            categoryKey = observation.attrib['categoryKey']
            if (categoryKey.startswith('animals.liveanimals') or
                  categoryKey.startswith('animals.problemanimal')):
                # Get the photonumber attribute for the waypoint's
                # observation
                element = './/%sattributes[@attributeKey="photonumber"]' % (
                    namespace, )
                photonumber = observation.find(element)
                if photonumber is not None:
                    element = './/%ssValue' % (namespace, )
                    # Get the value for photonumber
                    sValue  = photonumber.find(element)
                    if sValue is None:
                        raise IOError(
                            ('The photonumber sValue is missing from '
                             'photonumber, waypoint_id: %r') %
                            (waypoint_id, ))
                    # Python cast the value
                    try:
                        photo_number = int(float(sValue.text)) - offset
                    except ValueError:
                        # raise IOError('The photonumber sValue is invalid,
                        # waypoint_id: %r' % (waypoint_id, ))
                        print(('[ibs]     '
                               'Skipped Invalid Observation with '
                               'photonumber: %r, waypoint_id: %r')
                              % (sValue.text, waypoint_id, ))
                        continue
                    # Check that the photo_number is within the acceptable bounds
                    if photo_number >= nTotal:
                        raise IOError(
                            'The Patrol XML file is looking for images '
                            'that do not exist (too few images given)')
                    # Keep track of the last waypoint that was processed
                    # becuase we only have photono, which indicates start
                    # indices and doesn't specify the end index.  The
                    # ending index is extracted as the next waypoint's
                    # photonum minus 1.
                    if (last_photo_number is not None and
                         last_imageset_info is not None):
                        imageset_info = (
                            last_imageset_info + [(last_photo_number,
                                                    photo_number)])
                        imageset_info_list.append(imageset_info)
                    last_photo_number = photo_number
                    last_imageset_info = waypoint_info
                else:
                    # raise IOError('The photonumber value is missing from
                    # waypoint, waypoint_id: %r' % (waypoint_id, ))
                    print(('[ibs]     Skipped Empty Observation with'
                           '"categoryKey": %r, waypoint_id: %r') %
                          (categoryKey, waypoint_id, ))
            else:
                print(('[ibs]     '
                       'Skipped Incompatible Observation with '
                       '"categoryKey": %r, waypoint_id: %r') %
                      (categoryKey, waypoint_id, ))
    # Append the last photo_number
    if last_photo_number is not None and last_imageset_info is not None:
        imageset_info = last_imageset_info + [(last_photo_number, nTotal)]
        imageset_info_list.append(imageset_info)
    return imageset_info_list


@register_ibs_method
def compute_occurrences_smart(ibs, gid_list, smart_xml_fpath):
    """
    Function to load and process a SMART patrol XML file
    """
    # Get file and copy to ibeis database folder
    xml_dir, xml_name = split(smart_xml_fpath)
    dst_xml_path = join(ibs.get_smart_patrol_dir(), xml_name)
    ut.copy(smart_xml_fpath, dst_xml_path, overwrite=True)
    # Process the XML File
    print('[ibs] Processing Patrol XML file: %r' % (dst_xml_path, ))
    try:
        imageset_info_list = ibs._parse_smart_xml(dst_xml_path, len(gid_list))
    except Exception as e:
        ibs.delete_images(gid_list)
        print(('[ibs] ERROR: Parsing Patrol XML file failed, '
               'rolling back by deleting %d images...') %
              (len(gid_list, )))
        raise e
    if len(gid_list) > 0:
        # Sanity check
        assert len(imageset_info_list) > 0, (
            ('Trying to added %d images, but the Patrol  '
             'XML file has no observations') % (len(gid_list), ))
    # Display the patrol imagesets
    for index, imageset_info in enumerate(imageset_info_list):
        smart_xml_fname, smart_waypoint_id, gps, local_time, range_ = imageset_info
        start, end = range_
        gid_list_ = gid_list[start:end]
        print('[ibs]     Found Patrol ImageSet: %r' % (imageset_info, ))
        print('[ibs]         GIDs: %r' % (gid_list_, ))
        if len(gid_list_) == 0:
            print('[ibs]         SKIPPING EMPTY IMAGESET')
            continue
        # Add the GPS data to the iamges
        gps_list  = [ gps ] * len(gid_list_)
        ibs.set_image_gps(gid_list_, gps_list)
        # Create a new imageset
        imagesettext = '%s Waypoint %03d' % (xml_name.replace('.xml', ''), index + 1, )
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
        python -m ibeis.control.IBEISControl --test-compute_occurrences

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
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
    from ibeis.algo.preproc import preproc_occurrence
    print('[ibs] Computing and adding imagesets.')
    # Only ungrouped images are clustered
    gid_list = ibs.get_ungrouped_gids()
    #gid_list = ibs.get_valid_gids(require_unixtime=False, reviewed=False)
    with ut.Timer('computing imagesets'):
        flat_imgsetids, flat_gids = preproc_occurrence.ibeis_compute_occurrences(
            ibs, gid_list, config=config)
        sortx = ut.argsort(flat_imgsetids)
        flat_imgsetids = ut.take(flat_imgsetids, sortx)
        flat_gids = ut.take(flat_gids, sortx)

    valid_imgsetids = ibs.get_valid_imgsetids()
    imgsetid_offset = 0 if len(valid_imgsetids) == 0 else max(valid_imgsetids)

    # This way we can make sure that manually separated imagesets
    # remain untouched, and ensure that new imagesets are created
    flat_imgsetids_offset = [imgsetid + imgsetid_offset
                             for imgsetid in flat_imgsetids]
    imagesettext_list = ['Occurrence ' + str(imgsetid)
                         for imgsetid in flat_imgsetids_offset]
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

    point_dict = {
        1  : [-0.829843, 35.732721],
        2  : [-0.829843, 37.165353],
        3  : [-0.829843, 38.566150],
        4  : [ 0.405015, 37.165353],
        5  : [ 0.405015, 38.566150],
        6  : [ 1.292767, 35.732721],
        7  : [ 1.292767, 36.701444],
        8  : [ 1.292767, 37.029463],
        9  : [ 1.292767, 37.415937],
        10 : [ 1.292767, 38.566150],
        11 : [ 2.641838, 35.732721],
        12 : [ 2.641838, 37.029463],
        13 : [ 2.641838, 37.415937],
        14 : [ 2.641838, 38.566150],
    }

    zone_dict = {
        '1' : [1, 2, 4, 7, 6],
        '2' : [2, 3, 5, 4],
        '3' : [4, 5, 10, 7],
        '4' : [6, 8, 12, 11],
        '5' : [8, 9, 13, 12],
        '6' : [9, 10, 14, 13],
        'North' : [6, 10, 14, 11],
        'Core' : [1, 3, 14, 11],
    }

    path_dict = {
        zone : Path(np.array(
            [ point_dict[vertex] for vertex in zone_dict[zone] ]
        ))
        for zone in zone_dict
    }

    # ADD COUNTIES
    name_list = [
        'Laikipia',
        'Samburu',
        'Isiolo',
        'Marsabit',
        'Meru',
    ]
    county_file_url = 'https://lev.cs.rpi.edu/public/data/kenyan_counties_boundary_gps_coordinates.zip'
    unzipped_path = ut.grab_zipped_url(county_file_url)
    county_path = join(unzipped_path, 'County')
    counties = shapefile.Reader(county_path)
    for record, shape in zip(counties.records(), counties.shapes()):
        name = record[5]
        if name not in name_list:
            continue
        point_list = shape.points
        point_list = [ list(point)[::-1] for point in point_list ]
        path_dict[name] = Path(np.array(
            point_list
        ))

    return path_dict


@register_ibs_method
def compute_ggr_imagesets(ibs, gid_list=None, min_diff=86400, individual=False):
    # GET DATA
    path_dict = ibs.compute_ggr_path_dict()
    zone_list = sorted(path_dict.keys()) + ['7']
    imageset_dict = { zone : [] for zone in zone_list }

    if gid_list is None:
        gid_list = ibs.get_valid_gids()

    gps_list = ibs.get_image_gps(gid_list)
    note_list = ibs.get_image_notes(gid_list)
    temp = -1 if individual else -2
    note_list_ = [
        ','.join(note.strip().split(',')[:temp])
        for note in note_list
    ]
    note_list = [
        ','.join(note.strip().split(',')[:-1])
        for note in note_list
    ]

    special_zone_map = {
        'GGR,3,A' : '1,Laikipia,Core',
        'GGR,8,A' : '3,Isiolo,Core',
        'GGR,10,A' : '1,Laikipia,Core',
        'GGR,13,A' : '1,Laikipia,Core',
        'GGR,14,A' : '1,Laikipia,Core',
        'GGR,15,A' : '1,Laikipia,Core',
        'GGR,19,A' : '2,Samburu,Core',
        'GGR,23,A' : '1,Laikipia,Core',
        'GGR,24,A' : '1,Laikipia,Core',
        'GGR,25,A' : '1,Laikipia,Core',
        'GGR,27,A' : '1,Laikipia,Core',
        'GGR,29,A' : '1,Laikipia,Core',
        'GGR,37,A' : '1,Laikipia,Core',
        'GGR,37,B' : '1,Laikipia,Core',
        'GGR,38,C' : '1,Laikipia,Core',
        'GGR,40,A' : '1,Laikipia,Core',
        'GGR,41,B' : '1,Laikipia,Core',
        'GGR,44,A' : None,
        'GGR,45,A' : '2,Isiolo,Core',
        'GGR,46,A' : '1,Laikipia,Core',
        'GGR,62,B' : '1,Laikipia,Core',
        'GGR,86,A' : '3,Isiolo,Core',
        'GGR,96,A' : '1,Laikipia,Core',
        'GGR,97,B' : '2,Samburu,Core',
        'GGR,108,A' : '1,Laikipia,Core',
        'GGR,118,C' : '6,North,Marsabit,Core',
    }

    skipped_gid_list = []
    skipped_note_list = []
    skipped = 0
    for gid, point in zip(gid_list, gps_list):
        if point == (-1, -1):
            unixtime = ibs.get_image_unixtime(gid)
            index = gid_list.index(gid)
            note = note_list_[index]

            # Find siblings in the same car
            sibling_gid_list = [
                gid_
                for gid_, note_ in zip(gid_list, note_list_)
                if note_ == note
            ]

            # Get valid GPS
            gps_list = ibs.get_image_gps(sibling_gid_list)
            flag_list = [ gps != (-1, -1) for gps in gps_list ]
            gid_list_  = ut.compress(sibling_gid_list, flag_list)

            # If found, get closest image
            if len(gid_list_) > 0:
                gps_list_  = ibs.get_image_gps(gid_list_)
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

        if point == (-1, -1):
            note = note_list[index]
            if note in special_zone_map:
                zone_str = special_zone_map[note]
                if zone_str is not None:
                    zone_list = zone_str.strip().split(',')
                    for zone in zone_list:
                        imageset_dict[zone].append(gid)
                    continue

        if point == (-1, -1):
            skipped_gid_list.append(gid)
            skipped_note_list.append(note)
            skipped += 1
            continue

        found = False
        for zone in sorted(path_dict.keys()):
            path = path_dict[zone]
            if path.contains_point(point):
                found = True
                imageset_dict[zone].append(gid)
        if not found:
            imageset_dict['7'].append(gid)

    for zone, gid_list in sorted(imageset_dict.iteritems()):
        imageset_str = 'GGR Special Zone %s' % (zone, )
        imageset_id = ibs.add_imagesets(imageset_str)
        args = (imageset_str, imageset_id, len(gid_list), )
        print('Creating new GGR imageset: %r (ID %d) with %d images' % args)
        ibs.delete_gsgr_imageset_relations(imageset_id)
        ibs.set_image_imgsetids(gid_list, [imageset_id] * len(gid_list))
    print('SKIPPED %d IMAGES' % (skipped, ))
    skipped_note_list = sorted(list(set(skipped_note_list)))
    print('skipped_note_list = %r' % (skipped_note_list, ))
    print('skipped_gid_list = %r' % (skipped_gid_list, ))


@register_ibs_method
def compute_ggr_fix_gps_names(ibs, min_diff=1800):  # 86,400 = 60 sec x 60 min X 24 hours
    # Get all aids
    aid_list = ibs.get_valid_aids()
    num_all = len(aid_list)
    gps_list = ibs.get_annot_image_gps(aid_list)
    flag_list = [ gps == (-1, -1) for gps in gps_list ]
    # Get bad GPS aids
    aid_list = ut.filter_items(aid_list, flag_list)
    num_bad = len(aid_list)
    nid_list = ibs.get_annot_name_rowids(aid_list)
    flag_list = [ nid != const.UNKNOWN_NAME_ROWID for nid in nid_list ]
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
            print('FOUND LOCATION FOR AID %d' % (aid, ))
            print('\tDIFF   : %d H, %d M, %d S' % (h, m, s, ))
            print('\tNEW GPS: %s' % (closest_gps, ))
    print('%d \ %d \ %d \ %d' % (num_all, num_bad, num_known, num_found, ))
    return recovered_aid_list, recovered_gps_list, recovered_dist_list


@register_ibs_method
def compute_ggr_fix_gps_contributors(ibs, min_diff=1800, individual=True):
    # Get all aids
    aid_list = ibs.get_valid_aids()
    num_all = len(aid_list)
    gps_list = ibs.get_annot_image_gps(aid_list)
    flag_list = [ gps == (-1, -1) for gps in gps_list ]
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
    note_list = [
        ','.join(note.strip().split(',')[:temp])
        for note in note_list
    ]

    not_found = set([])
    num_found = 0
    for aid in unrecovered_aid_list:
        gid = ibs.get_annot_gids(aid)
        unixtime = ibs.get_image_unixtime(gid)
        index = gid_list.index(gid)
        note = note_list[index]

        # Find siblings in the same car
        sibling_gid_list = [
            gid_
            for gid_, note_ in zip(gid_list, note_list)
            if note_ == note
        ]

        # Get valid GPS
        gps_list = ibs.get_image_gps(sibling_gid_list)
        flag_list = [ gps != (-1, -1) for gps in gps_list ]
        gid_list_  = ut.compress(sibling_gid_list, flag_list)

        # If found, get closest image
        if len(gid_list_) > 0:
            gps_list_  = ibs.get_image_gps(gid_list_)
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
                print('FOUND LOCATION FOR AID %d' % (aid, ))
                print('\tDIFF   : %d H, %d M, %d S' % (h, m, s, ))
                print('\tNEW GPS: %s' % (closest_gps, ))
            else:
                not_found.add(note)
        else:
            not_found.add(note)
    print('%d \ %d \ %d \ %d' % (num_all, num_bad, num_unrecovered, num_found, ))
    num_recovered = len(recovered_aid_list)
    num_unrecovered = num_bad - len(recovered_aid_list)
    print('Missing GPS: %d' % (num_bad, ))
    print('Recovered  : %d' % (num_recovered, ))
    print('Unrecovered: %d' % (num_unrecovered, ))
    print('Not Found  : %r' % (not_found, ))
    return recovered_aid_list, recovered_gps_list, recovered_dist_list


@register_ibs_method
def commit_ggr_fix_gps(ibs, **kwargs):
    vals = ibs.compute_ggr_fix_gps_contributors(**kwargs)
    recovered_aid_list, recovered_gps_list, recovered_dist_list = vals
    recovered_gid_list = ibs.get_annot_gids(recovered_aid_list)

    zipped = zip(recovered_gid_list, recovered_gps_list, recovered_dist_list)
    assignment_dict = {}
    for gid, gps, dist in zipped:
        if gid not in assignment_dict:
            assignment_dict[gid] = []
        assignment_dict[gid].append((dist, gps))

    for assignment_gid in assignment_dict:
        assignment_list = sorted(assignment_dict[assignment_gid])
        assignment_gps = assignment_list[0][1]
        ibs.set_image_gps([assignment_gid], [assignment_gps])


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.other.ibsfuncs
        python -m ibeis.other.ibsfuncs --allexamples
        python -m ibeis.other.ibsfuncs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
