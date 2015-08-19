# -*- coding: utf-8 -*-
"""
developer convenience functions for ibs

TODO: need to split up into sub modules:
    consistency_checks
    feasibility_fixes
    move the export stuff to dbio

    then there are also convineience functions that need to be ordered at least
    within this file
"""
from __future__ import absolute_import, division, print_function
import six
import types
import re
from six.moves import zip, range, map
from os.path import split, join, exists
#import vtool.image as gtool
import numpy as np
from utool._internal.meta_util_six import get_funcname, get_imfunc, set_funcname
#from vtool import linalg, geometry, image
import vtool as vt
import utool as ut
import ibeis
#from ibeis import params
from ibeis import constants as const
try:
    from detecttools.pypascalmarkup import PascalVOC_Markup_Annotation
except ImportError as ex:
    ut.printex('COMMIT TO DETECTTOOLS')
    pass
from ibeis.control import accessor_decors

# Inject utool functions
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[ibsfuncs]', DEBUG=False)


# Try to work around circular import
#from ibeis.control.IBEISControl import IBEISController  # Must import class before injection
CLASS_INJECT_KEY = ('IBEISController', 'ibsfuncs')
__injectable = ut.make_class_method_decorator(CLASS_INJECT_KEY, __name__)


def fix_zero_features(ibs):
    aid_list = ibs.get_valid_aids()
    nfeat_list = ibs.get_annot_num_feats(aid_list, ensure=False)
    haszero_list = [nfeat == 0 for nfeat in nfeat_list]
    haszero_aids = ut.filter_items(aid_list, haszero_list)
    ibs.delete_annot_chips(haszero_aids)


@ut.make_class_postinject_decorator(CLASS_INJECT_KEY, __name__)
def postinject_func(ibs):
    r"""
    Args:
        ibs (IBEISController):

    CommandLine:
        python -m ibeis.ibsfuncs --test-postinject_func

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
    # List of getters to _unflatten
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


@__injectable
def refresh(ibs):
    from ibeis import ibsfuncs
    from ibeis import all_imports
    ibsfuncs.rrr()
    all_imports.reload_all()
    ibs.rrr()


def export_to_xml(ibs, offset=0, enforce_yaw=False):
    target_size = 900
    information = {
        'database_name' : ibs.get_dbname()
    }
    datadir = ibs._ibsdb + '/LearningData/'
    imagedir = datadir + 'JPEGImages/'
    annotdir = datadir + 'Annotations/'
    ut.ensuredir(datadir)
    ut.ensuredir(imagedir)
    ut.ensuredir(annotdir)
    gid_list = ibs.get_valid_gids()
    print('Exporting %d images' % (len(gid_list),))
    for gid in gid_list:
        yawed = True
        aid_list = ibs.get_image_aids(gid)
        image_uri = ibs.get_image_uris(gid)
        image_path = ibs.get_image_paths(gid)
        if len(aid_list) > 0:
            fulldir = image_path.split('/')
            filename = fulldir.pop()
            extension = filename.split('.')[-1]  # NOQA
            out_name = "2015_%06d" % offset
            out_img = out_name + ".jpg"
            folder = "IBEIS"

            _image = vt.imread(image_path)
            height, width, channels = _image.shape
            if width > height:
                ratio = height / width
                decrease = target_size / width
                width = target_size
                height = int(target_size * ratio)
            else:
                ratio = width / height
                decrease = target_size / height
                height = target_size
                width = int(target_size * ratio)

            dst_img = imagedir + out_img
            _image = vt.resize(_image, (width, height))
            vt.imwrite(dst_img, _image)

            annotation = PascalVOC_Markup_Annotation(dst_img, folder, out_img, source=image_uri, **information)
            bbox_list = ibs.get_annot_bboxes(aid_list)
            theta_list = ibs.get_annot_thetas(aid_list)
            for aid, bbox, theta in zip(aid_list, bbox_list, theta_list):
                # Transformation matrix
                R = vt.rotation_around_bbox_mat3x3(theta, bbox)
                # Get verticies of the annotation polygon
                verts = vt.verts_from_bbox(bbox, close=True)
                # Rotate and transform vertices
                xyz_pts = vt.add_homogenous_coordinate(np.array(verts).T)
                trans_pts = vt.remove_homogenous_coordinate(R.dot(xyz_pts))
                new_verts = np.round(trans_pts).astype(np.int).T.tolist()
                x_points = [pt[0] for pt in new_verts]
                y_points = [pt[1] for pt in new_verts]
                xmin = int(min(x_points) * decrease)
                xmax = int(max(x_points) * decrease)
                ymin = int(min(y_points) * decrease)
                ymax = int(max(y_points) * decrease)
                #TODO: Change species_name to getter in IBEISControl once implemented
                #species_name = 'grevys_zebra'
                species_name = ibs.get_annot_species_texts(aid)
                yaw = ibs.get_annot_yaws(aid)
                info = {}
                if yaw != -1 and yaw is not None:
                    info['pose'] = "%0.6f" % yaw
                else:
                    yawed = False
                    print("UNVIEWPOINTED: %d " % gid)
                annotation.add_object(species_name, (xmax, xmin, ymax, ymin), **info)
            dst_annot = annotdir + out_name  + '.xml'
            # Write XML
            if not enforce_yaw or yawed:
                print("Copying:\n%r\n%r\n%r\n\n" % (image_path, dst_img, (width, height), ))
                xml_data = open(dst_annot, 'w')
                xml_data.write(annotation.xml())
                xml_data.close()
                offset += 1
        else:
            print("Skipping:\n%r\n\n" % (image_path, ))


@__injectable
def export_to_hotspotter(ibs):
    from ibeis.dbio import export_hsdb
    export_hsdb.export_ibeis_to_hotspotter(ibs)


#def export_image_subset(ibs, gid_list, dst_fpath=None):
#    dst_fpath = ut.truepath('~')
#    #gid_list = [692, 693, 680, 781, 751, 753, 754, 755, 756]
#    gpath_list = ibs.get_image_paths(gid_list)
#    gname_list = [join(dst_fpath, gname) for gname in ibs.get_image_gnames(gid_list)]
#    ut.copy_files_to(gpath_list, dst_fpath_list=gname_list)


@__injectable
def mark_annot_pair_as_reviewed(ibs, aid1, aid2):
    """ denote that this match was reviewed and keep whatever status it is given """
    isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
    if isunknown1 or isunknown2:
        truth = const.TRUTH_UNKNOWN
    else:
        nid1, nid2 = ibs.get_annot_name_rowids((aid1, aid2))
        truth = const.TRUTH_UNKNOWN if (nid1 == nid2) else const.TRUTH_NOT_MATCH
    ibs.add_or_update_annotmatch(aid1, aid2, truth, [1.0])


@__injectable
def add_or_update_annotmatch(ibs, aid1, aid2, truth, confidence):
    annotmatch_rowid = ibs.get_annotmatch_rowid_from_superkey([aid1], [aid2])[0]
    # TODO: sql add or update?
    if annotmatch_rowid is not None:
        ibs.set_annotmatch_truth([annotmatch_rowid], [1])
        ibs.set_annotmatch_confidence([annotmatch_rowid], [1.0])
    else:
        ibs.add_annotmatch([aid1], [aid2], annotmatch_truth_list=[truth], annotmatch_confidence_list=[1.0])


# AUTOGENED CONSTANTS:


@__injectable
def get_annot_has_reviewed_matching_aids(ibs, aid_list, eager=True, nInput=None):
    num_reviewed_list = ibs.get_annot_num_reviewed_matching_aids(aid_list)
    has_reviewed_list = [num_reviewed > 0 for num_reviewed in num_reviewed_list]
    return has_reviewed_list


@__injectable
def get_annot_num_reviewed_matching_aids(ibs, aid1_list, eager=True, nInput=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        eager (bool):
        nInput (None):

    Returns:
        list: num_annot_reviewed_list

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_annot_num_reviewed_matching_aids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb2')
        >>> aid1_list = ibs.get_valid_aids()
        >>> eager = True
        >>> nInput = None
        >>> # execute function
        >>> num_annot_reviewed_list = get_annot_num_reviewed_matching_aids(ibs, aid_list, eager, nInput)
        >>> # verify results
        >>> result = str(num_annot_reviewed_list)
        >>> print(result)
    """
    aids_list = ibs.get_annot_reviewed_matching_aids(aid1_list, eager=eager, nInput=nInput)
    num_annot_reviewed_list = list(map(len, aids_list))
    return num_annot_reviewed_list


def get_annotmatch_rowids_from_aid1(ibs, aid1_list, eager=True, nInput=None):
    """
    Returns a list of the aids that were reviewed as candidate matches to the input aid

    aid_list = ibs.get_valid_aids()
    """
    ANNOT_ROWID1 = 'annot_rowid1'
    params_iter = [(aid1,) for aid1 in aid1_list]
    colnames = ('annotmatch_rowid',)
    andwhere_colnames = (ANNOT_ROWID1,)
    annotmach_rowid_list = ibs.db.get_where2(
        const.ANNOTMATCH_TABLE, colnames, params_iter,
        andwhere_colnames=andwhere_colnames, eager=eager, unpack_scalars=False, nInput=nInput)
    return annotmach_rowid_list


@__injectable
def get_annot_reviewed_matching_aids(ibs, aid_list, eager=True, nInput=None):
    """
    Returns a list of the aids that were reviewed as candidate matches to the input aid

    aid_list = ibs.get_valid_aids()
    """
    ANNOT_ROWID1 = 'annot_rowid1'
    ANNOT_ROWID2 = 'annot_rowid2'
    #params_iter = [(aid, aid) for aid in aid_list]
    #[(aid, aid) for aid in aid_list]
    #colnames = (ANNOT_ROWID1, ANNOT_ROWID2)
    #where_colnames = (ANNOT_ROWID1, ANNOT_ROWID2)
    params_iter = [(aid,) for aid in aid_list]
    colnames = (ANNOT_ROWID2,)
    andwhere_colnames = (ANNOT_ROWID1,)
    aids_list = ibs.db.get_where2(const.ANNOTMATCH_TABLE, colnames,
                                  params_iter,
                                  andwhere_colnames=andwhere_colnames,
                                  eager=eager, unpack_scalars=False,
                                  nInput=nInput)
    #logicop = 'OR'
    #aids_list = ibs.db.get_where3(
    #    const.ANNOTMATCH_TABLE, colnames, params_iter,
    #    where_colnames=where_colnames, logicop=logicop, eager=eager,
    #    unpack_scalars=False, nInput=nInput)
    return aids_list


@__injectable
def get_annot_pair_truth(ibs, aid1_list, aid2_list):
    """
    CAREFUL: uses annot match table for truth, so only works if reviews have happend
    """
    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
    annotmatch_truth_list = ibs.get_annotmatch_truth(annotmatch_rowid_list)
    return annotmatch_truth_list


@__injectable
def get_annot_pair_is_reviewed(ibs, aid1_list, aid2_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid1_list (list):
        aid2_list (list):

    Returns:
        list: annotmatch_reviewed_list

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_annot_pair_is_reviewed

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb2')
        >>> aid_list = ibs.get_valid_aids()
        >>> pairs = list(ut.product(aid_list, aid_list))
        >>> aid1_list = ut.get_list_column(pairs, 0)
        >>> aid2_list = ut.get_list_column(pairs, 1)
        >>> # execute function
        >>> annotmatch_reviewed_list = get_annot_pair_is_reviewed(ibs, aid1_list, aid2_list)
        >>> # verify results
        >>> reviewed_pairs = ut.list_compress(pairs, annotmatch_reviewed_list)
        >>> result = len(reviewed_pairs)
        >>> print(result)
        104
    """
    flag_non_None_items = lambda list_: (item_ is not None for item_ in list_)
    annotmatch_truth_list1 = ibs.get_annot_pair_truth(aid1_list, aid2_list)
    annotmatch_truth_list2 = ibs.get_annot_pair_truth(aid2_list, aid1_list)
    annotmatch_truth_list = ut.or_lists(
        flag_non_None_items(annotmatch_truth_list1),
        flag_non_None_items(annotmatch_truth_list2))
    #annotmatch_reviewed_list = [truth is not None for truth in annotmatch_truth_list]
    return annotmatch_truth_list


@__injectable
def get_image_time_statstr(ibs, gid_list=None):
    if gid_list is None:
        gid_list = ibs.get_valid_gids()
    unixtime_list_ = ibs.get_image_unixtime(gid_list)
    utvalid_list   = [time != -1 for time in unixtime_list_]
    unixtime_list  = ut.filter_items(unixtime_list_, utvalid_list)
    unixtime_statstr = ut.get_timestats_str(unixtime_list, newlines=True)
    return unixtime_statstr


@__injectable
def get_image_annotation_bboxes(ibs, gid_list):
    aids_list = ibs.get_image_aids(gid_list)
    bboxes_list = ibs.get_unflat_annotation_bboxes(aids_list)
    return bboxes_list


@__injectable
def get_image_annotation_thetas(ibs, gid_list):
    aids_list = ibs.get_image_aids(gid_list)
    thetas_list = ibs.get_unflat_annotation_thetas(aids_list)
    return thetas_list


@__injectable
def filter_junk_annotations(ibs, aid_list):
    r"""
    remove junk annotations from a list

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: filtered_aid_list

    CommandLine:
        python -m ibeis.ibsfuncs --test-filter_junk_annotations

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> # execute function
        >>> filtered_aid_list = filter_junk_annotations(ibs, aid_list)
        >>> # verify results
        >>> result = str(filtered_aid_list)
        >>> print(result)
    """
    isjunk_list = ibs.get_annot_isjunk(aid_list)
    filtered_aid_list = ut.filterfalse_items(aid_list, isjunk_list)
    return filtered_aid_list


@__injectable
def compute_all_chips(ibs, **kwargs):
    """
    Executes lazy evaluation of all chips
    """
    print('[ibs] compute_all_chips')
    aid_list = ibs.get_valid_aids(**kwargs)
    cid_list = ibs.add_annot_chips(aid_list)
    return cid_list


@__injectable
def compute_all_features(ibs, **kwargs):
    """
    Executes lazy evaluation of all chips and features
    """
    cid_list = ibs.compute_all_chips(**kwargs)
    print('[ibs] compute_all_features')
    fid_list = ibs.add_chip_feat(cid_list)
    return fid_list


@__injectable
def compute_all_featweights(ibs, **kwargs):
    """
    Executes lazy evaluation of all chips and features
    """
    fid_list = ibs.compute_all_features(**kwargs)
    print('[ibs] compute_all_featweights')
    featweight_rowid_list = ibs.add_feat_featweights(fid_list)
    return featweight_rowid_list


@__injectable
def precompute_all_annot_dependants(ibs, **kwargs):
    ibs.compute_all_featweights(**kwargs)


@__injectable
def recompute_fgweights(ibs, aid_list=None):
    """ delete all feature weights and then recompute them """
    # Delete all featureweights
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
        featweight_rowid_list = ibs._get_all_featweight_rowids()
    else:
        featweight_rowid_list = ibs.get_annot_featweight_rowids(aid_list)
    ibs.delete_featweight(featweight_rowid_list)
    #ibs.delete_annot_featweight(aid_list)
    # Recompute current featureweights
    ibs.get_annot_fgweights(aid_list, ensure=True)


@__injectable
def ensure_annotation_data(ibs, aid_list, chips=True, feats=True, featweights=False):
    if chips or feats or featweights:
        cid_list = ibs.add_annot_chips(aid_list)
    if feats or featweights:
        fid_list = ibs.add_chip_feat(cid_list)
    if featweights:
        featweight_rowid_list = ibs.add_feat_featweights(fid_list)  # NOQA


@__injectable
def convert_empty_images_to_annotations(ibs):
    """ images without chips are given an ANNOTATION over the entire image """
    gid_list = ibs.get_empty_gids()
    aid_list = ibs.use_images_as_annotations(gid_list)
    return aid_list


@__injectable
def use_images_as_annotations(ibs, gid_list, name_list=None, nid_list=None,
                              notes_list=None, adjust_percent=0.0):
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
                                   name_list=name_list, nid_list=nid_list, notes_list=notes_list)
    return aid_list


@__injectable
def assert_valid_species_texts(ibs, species_list, iswarning=True):
    if ut.NO_ASSERTS:
        return
    try:
        isvalid_list = [
            species in const.VALID_SPECIES  # or species == const.UNKNOWN
            for species in species_list
        ]
        assert all(isvalid_list), 'invalid species found in %r: %r' % (
            ut.get_caller_name(range(1, 3)), ut.filterfalse_items(species_list, isvalid_list),)
    except AssertionError as ex:
        ut.printex(ex, iswarning=iswarning)
        if not iswarning:
            raise


@__injectable
def assert_singleton_relationship(ibs, alrids_list):
    if ut.NO_ASSERTS:
        return
    try:
        assert all([len(alrids) == 1 for alrids in alrids_list]),\
            'must only have one relationship of a type'
    except AssertionError as ex:
        parent_locals = ut.get_parent_locals()
        ut.printex(ex, 'parent_locals=' + ut.dict_str(parent_locals), key_list=['alrids_list', ])
        raise


@__injectable
def assert_valid_aids(ibs, aid_list, verbose=False, veryverbose=False):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        verbose (bool):  verbosity flag(default = False)
        veryverbose (bool): (default = False)

    Returns:
        ?:

    CommandLine:
        python -m ibeis.ibsfuncs --test-assert_valid_aids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> verbose = False
        >>> veryverbose = False
        >>> result = assert_valid_aids(ibs, aid_list, verbose, veryverbose)
        >>> try:
        >>>    result = assert_valid_aids(ibs, aid_list + [0], verbose, veryverbose)
        >>> except AssertionError:
        >>>    print('Correctly got assertion')
        >>> else:
        >>>    assert False, 'should have failed'
        >>> print(result)
    """
    #if ut.NO_ASSERTS and not force:
    #    return
    #valid_aids = set(ibs.get_valid_aids())
    #invalid_aids = [aid for aid in aid_list if aid not in valid_aids]
    #isinvalid_list = [aid not in valid_aids for aid in aid_list]
    isinvalid_list = [aid is None for aid in ibs.get_annot_aid(aid_list)]
    #isinvalid_list = [aid not in valid_aids for aid in aid_list]
    try:
        assert not any(isinvalid_list), 'invalid aids: %r' % (ut.filter_items(aid_list, isinvalid_list),)
        isinvalid_list = [not isinstance(aid, ut.VALID_INT_TYPES) for aid in aid_list]
        assert not any(isinvalid_list), 'invalidly typed aids: %r' % (ut.filter_items(aid_list, isinvalid_list),)
    except AssertionError as ex:
        print('dbname = %r' % (ibs.get_dbname()))
        ut.printex(ex)
        #ut.embed()
        raise
    if veryverbose:
        print('passed assert_valid_aids')


@__injectable
def assert_images_exist(ibs, gid_list=None, verbose=True):
    if gid_list is None:
        gid_list = ibs.get_valid_gids()
    print('checking images exist')
    gpath_list = ibs.get_image_paths(gid_list)
    exists_list = list(map(exists, gpath_list))
    bad_gids = ut.filterfalse_items(gid_list, exists_list)
    num_bad_gids = len(bad_gids)
    if verbose:
        bad_gpaths = ut.filterfalse_items(gpath_list, exists_list)
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
    valid_namecheck = [not isconflict(name, const.UNKNOWN) for name in name_list]
    assert all(valid_namecheck), ('A name conflicts with UKNONWN Name. -- '
                                  'cannot start a name with four underscores')


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
            (lbltype_rowid == valid_lbltype_rowid) or
            (lbltype_rowid is None and lblannot_rowid == const.UNKNOWN_LBLANNOT_ROWID)
            for lbltype_rowid, lblannot_rowid in
            zip(lbltype_rowid_list, lblannot_rowid_list)]
        assert all(validtype_list), 'not all types match valid type'
    except AssertionError as ex:
        tup_list = list(map(str, list(zip(lbltype_rowid_list, lblannot_rowid_list))))
        print('[!!!] (lbltype_rowid, lblannot_rowid) = : ' + ut.indentjoin(tup_list))
        print('[!!!] valid_lbltype_rowid: %r' % (valid_lbltype_rowid,))

        ut.printex(ex, 'not all types match valid type',
                      keys=['valid_lbltype_rowid', 'lblannot_rowid_list'])
        raise


@__injectable
def check_image_consistency(ibs, gid_list=None):
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
        python -m ibeis.ibsfuncs --test-check_image_uuid_consistency --db=PZ_Master0
        python -m ibeis.ibsfuncs --test-check_image_uuid_consistency

    Example:
        >>> # DISABLE_DOCTEST
        >>> # Check for very large files
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> gid_list_ = ibs.get_valid_gids()
        >>> gpath_list_ = ibs.get_image_paths(gid_list_)
        >>> bytes_list_ = [ut.get_file_nBytes(path) for path in gpath_list_]
        >>> sortx = ut.list_argsort(bytes_list_, reverse=True)[0:10]
        >>> gpath_list = ut.list_take(gpath_list_, sortx)
        >>> bytes_list = ut.list_take(bytes_list_, sortx)
        >>> gid_list   = ut.list_take(gid_list_, sortx)
        >>> ibeis.ibsfuncs.check_image_uuid_consistency(ibs, gid_list)
    """
    print('checking image uuid consistency')
    import ibeis.model.preproc.preproc_image as preproc_image
    gpath_list = ibs.get_image_paths(gid_list)
    guuid_list = ibs.get_image_uuids(gid_list)
    for ix in ut.ProgressIter(range(len(gpath_list))):
        gpath = gpath_list[ix]
        guuid_stored = guuid_list[ix]
        param_tup = preproc_image.parse_imageinfo(gpath)
        guuid_computed = param_tup[0]
        assert guuid_stored == guuid_computed, 'image ix=%d had a bad uuid' % ix


@__injectable
def check_annot_consistency(ibs, aid_list=None):
    r"""
    Args:
        ibs      (IBEISController):
        aid_list (list):

    CommandLine:
        python -m ibeis.ibsfuncs --test-check_annot_consistency

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
    cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=False)
    cfpath_list = ibs.get_chip_fpath(cid_list)
    valid_chip_list = [None if cfpath is None else exists(cfpath) for cfpath in cfpath_list]
    invalid_list = [flag is False for flag in valid_chip_list]
    invalid_cids = ut.filter_items(cid_list, invalid_list)
    if len(invalid_cids) > 0:
        print('found %d inconsistent chips attempting to fix' % len(invalid_cids))
        ibs.delete_chips(invalid_cids, verbose=True)
    ibs.check_chip_existence(aid_list=aid_list)
    visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
    exemplar_flag = ibs.get_annot_exemplar_flags(aid_list)
    is_unknown = ibs.is_aid_unknown(aid_list)
    # Exemplars should all be known
    unknown_exemplar_flags = ut.filter_items(exemplar_flag, is_unknown)
    is_error = [not flag for flag in unknown_exemplar_flags]
    assert all(is_error), 'Unknown annotations are set as exemplars'
    ut.debug_duplicate_items(visual_uuid_list)


def check_name_consistency(ibs, nid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        nid_list (list):

    CommandLine:
        python -m ibeis.ibsfuncs --test-check_name_consistency

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
        if not ut.list_allsame(sids):
            error_msg = 'aids=%r have the same name, but belong to multiple species=%r' % (aids, ibs.get_species_texts(ut.unique_keep_order2(sids)))
            print(error_msg)
            error_list.append(error_msg)
    if len(error_list) > 0:
        raise AssertionError('A total of %d names failed check_name_consistency' % (len(error_list)))


@__injectable
def check_name_mapping_consistency(ibs, nx2_aids):
    """
    checks that all the aids grouped in a name ahave the same name
    """
    # DEBUGGING CODE
    try:
        from ibeis import ibsfuncs
        _nids_list = ibsfuncs.unflat_map(ibs.get_annot_name_rowids, nx2_aids)
        assert all(map(ut.list_allsame, _nids_list))
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
                if ut.list_allsame(nids):
                    good += 1
                else:
                    huh += 1
            else:
                bad += 1
        ut.printex(ex, keys=['good', 'bad', 'huh'])


@__injectable
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
    from PIL import Image
    gpath_list = ibs.get_image_paths(gid_list)
    mark_, end_ = ut.log_progress('checking exif: ', len(gpath_list))
    exif_dict_list = []
    for ix in range(len(gpath_list)):
        mark_(ix)
        gpath = gpath_list[ix]
        pil_img = Image.open(gpath, 'r')
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

    end_()


@__injectable
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


@__injectable
def check_annotmatch_consistency(ibs):
    annomatch_rowids = ibs._get_all_annotmatch_rowids()
    aid1_list = ibs.get_annotmatch_aid1(annomatch_rowids)
    aid2_list = ibs.get_annotmatch_aid2(annomatch_rowids)
    exists1_list = ibs.db.check_rowid_exists(const.ANNOTATION_TABLE, aid1_list)
    exists2_list = ibs.db.check_rowid_exists(const.ANNOTATION_TABLE, aid2_list)
    invalid_list = ut.not_list(ut.and_lists(exists1_list, exists2_list))
    invalid_annotmatch_rowids = ut.filter_items(annomatch_rowids, invalid_list)
    print('There are %d invalid annotmatch rowids' % (len(invalid_annotmatch_rowids),))
    return invalid_annotmatch_rowids


@__injectable
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
        >>> from ibeis.ibsfuncs import *  # NOQA
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
            aids = ut.list_take(aid_list, dupxs)
            dupaids_list.append(aids[1:])

        toremove_aids = ut.flatten(dupaids_list)
        print('About to delete toremove_aids=%r' % (toremove_aids,))
        if ut.are_you_sure():
            ibs.delete_annots(toremove_aids)

            aid_list = ibs.get_valid_aids()
            visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
            ibs_dup_annots = ut.debug_duplicate_items(visual_uuid_list)
            assert len(ibs_dup_annots) == 0


#@__injectable
#def vacuum_and_clean_databases(ibs):
#    # Add to duct tape? or DEPRICATE
#    #ibs.vdd()
#    print(ibs.db.get_table_names())
#    # Removes all lblannots and lblannot relations as we are not using them
#    if False:
#        print(ibs.db.get_table_csv(const.NAME_TABLE))
#        print(ibs.db.get_table_csv(const.ANNOTATION_TABLE))
#        print(ibs.db.get_table_csv(const.LBLTYPE_TABLE))
#        print(ibs.db.get_table_csv(const.LBLANNOT_TABLE))
#        print(ibs.db.get_table_csv(const.AL_RELATION_TABLE))
#    if False:
#        # We deleted these all at one point, but its not a good operation to
#        # repeat
#        # Get old table indexes
#        #lbltype_rowids = ibs.db.get_all_rowids(const.LBLTYPE_TABLE)
#        lblannot_rowids = ibs.db.get_all_rowids(const.LBLANNOT_TABLE)
#        alr_rowids = ibs.db.get_all_rowids(const.AL_RELATION_TABLE)
#        # delete those tables
#        #ibs.db.delete_rowids(const.LBLTYPE_TABLE, lbltype_rowids)
#        ibs.db.delete_rowids(const.LBLANNOT_TABLE, lblannot_rowids)
#        ibs.db.delete_rowids(const.AL_RELATION_TABLE, alr_rowids)
#    ibs.db.vacuum()


@__injectable
def fix_and_clean_database(ibs):
    """ Function to run all database cleanup scripts

    Rename to run_cleanup_scripts

    Break into two funcs:
        run_cleanup_scripts
        run_fixit_scripts

    CONSITENCY CHECKS TODO:
        * check that annotmatches marked as False do not have the same name for similar viewpoints.
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
        ibs.delete_empty_eids()
        ibs.db.vacuum()
        print('finished fixes and consistency checks\n')


@__injectable
def fix_exif_data(ibs, gid_list):
    """ TODO CALL SCRIPT """
    import vtool.exif as exif
    from PIL import Image
    gpath_list = ibs.get_image_paths(gid_list)
    mark_, end_ = ut.log_progress('checking exif: ', len(gpath_list))
    exif_dict_list = []
    for ix in range(len(gpath_list)):
        mark_(ix)
        gpath = gpath_list[ix]
        pil_img = Image.open(gpath, 'r')
        exif_dict = exif.get_exif_dict(pil_img)
        exif_dict_list.append(exif_dict)
        #if len(exif_dict) > 0:
        #    break
    end_()

    latlon_list = [exif.get_lat_lon(_dict, None) for _dict in exif_dict_list]
    haslatlon_list = [latlon is not None for latlon in latlon_list]

    latlon_list_ = ut.filter_items(latlon_list, haslatlon_list)
    gid_list_    = ut.filter_items(gid_list, haslatlon_list)

    gps_list = ibs.get_image_gps(gid_list_)
    needsupdate_list = [gps == (-1, -1) for gps in gps_list]

    print('%d / %d need gps update' % (sum(needsupdate_list),
                                       len(needsupdate_list)))

    if sum(needsupdate_list)  > 0:
        assert sum(needsupdate_list) == len(needsupdate_list), 'safety. remove and evaluate if hit'
        #ibs.set_image_enctext(gid_list_, ['HASGPS'] * len(gid_list_))
        latlon_list__ = ut.filter_items(latlon_list_, needsupdate_list)
        gid_list__ = ut.filter_items(gid_list_, needsupdate_list)
        ibs.set_image_gps(gid_list__, latlon_list__)


@__injectable
def fix_invalid_nids(ibs):
    r"""
    Make sure that all rowids are greater than 0

    We can only handle there being a name with rowid 0 if it is UNKNOWN. In this
    case we safely delete it, but anything more complicated needs to be handled
    anually

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-fix_invalid_nids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
    is_invalid_nid_list = [nid <= ibs.UNKNOWN_NAME_ROWID for nid in nid_list]
    if any(is_invalid_nid_list):
        invalid_nids = ut.filter_items(nid_list, is_invalid_nid_list)
        invalid_texts = ut.filter_items(name_text_list, is_invalid_nid_list)
        if (len(invalid_nids) == 0 and
              invalid_nids[0] == ibs.UNKNOWN_NAME_ROWID and
              invalid_texts[0] == const.UNKNOWN):
            print('[ibs] found bad name rowids = %r' % (invalid_nids,))
            print('[ibs] found bad name texts  = %r' % (invalid_texts,))
            ibs.delete_names([ibs.UNKNOWN_NAME_ROWID])
        else:
            errmsg = 'Unfixable error: Found invalid (nid, text) pairs: '
            errmsg += ut.list_str(list(zip(invalid_nids, invalid_texts)))
            raise AssertionError(errmsg)


@__injectable
def fix_invalid_name_texts(ibs):
    r"""
    Ensure  that no name text is empty or '____'

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-fix_invalid_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
        invalid_nids = ut.filter_items(nid_list, is_invalid_name_text_list)
        invalid_texts = ut.filter_items(name_text_list, is_invalid_name_text_list)
        for count, (invalid_nid, invalid_text) in enumerate(zip(invalid_nids, invalid_texts)):
            conflict_set = invalid_name_set.union(set(ibs.get_name_texts(nid_list, apply_fix=False)))
            base_str = 'fixedname%d' + invalid_text
            new_text = ut.get_nonconflicting_string(base_str, conflict_set, offset=count)
            print('Fixing name %r -> %r' % (invalid_text, new_text))
            ibs.set_name_texts((invalid_nid,), (new_text,))
        print('Fixed %d name texts' % (len(invalid_nids)))
    else:
        print('all names seem valid')


@__injectable
def copy_encounters(ibs, eid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        eid_list (list):

    Returns:
        list: new_eid_list

    CommandLine:
        python -m ibeis.ibsfuncs --test-copy_encounters

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> ibs.delete_all_encounters()
        >>> ibs.compute_encounters()
        >>> eid_list = ibs.get_valid_eids()
        >>> # execute function
        >>> new_eid_list = copy_encounters(ibs, eid_list)
        >>> # verify results
        >>> result = str(ibs.get_encounter_text(new_eid_list))
        >>> assert [2] == list(set(map(len, ibs.get_image_eids(ibs.get_valid_gids()))))
        >>> print(result)
        >>> ibs.delete_all_encounters()
        >>> ibs.compute_encounters()
    """
    all_enctext_list = ibs.get_encounter_text(ibs.get_valid_eids())
    enctext_list = ibs.get_encounter_text(eid_list)
    new_enctext_list = [
        ut.get_nonconflicting_string(enctext + '_Copy(%d)', set(all_enctext_list))
        for enctext in enctext_list
    ]
    new_eid_list = ibs.add_encounters(new_enctext_list)
    gids_list = ibs.get_encounter_gids(eid_list)
    #new_eid_list =
    for gids, new_eid in zip(gids_list, new_eid_list):
        ibs.set_image_eids(gids, [new_eid] * len(gids))
    return new_eid_list


@__injectable
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
    unknown_exemplar_flags = ut.filter_items(flag_list, unknown_list)
    unknown_aid_list = ut.filter_items(aid_list, unknown_list)
    print('Fixing %d unknown annotations set as exemplars' % (sum(unknown_exemplar_flags),))
    ibs.set_annot_exemplar_flags(unknown_aid_list, [False] * len(unknown_aid_list))
    #is_error = [not flag for flag in unknown_exemplar_flags]
    #new_annots = [flag if nid != const.UNKNOWN_NAME_ROWID else 0
    #              for nid, flag in
    #              zip(nid_list, flag_list)]
    #ibs.set_annot_exemplar_flags(aid_list, new_annots)


@__injectable
def delete_all_recomputable_data(ibs):
    """
    Delete all cached data including chips and encounters
    """
    print('[ibs] delete_all_recomputable_data')
    ibs.delete_cachedir()
    ibs.delete_all_chips()
    ibs.delete_all_encounters()
    print('[ibs] finished delete_all_recomputable_data')


@__injectable
def delete_cache(ibs, delete_chips=False, delete_encounters=False):
    """
    Deletes the cache directory in the database directory.
    Can specify to delete encoutners and chips as well.
    """
    ibs.ensure_directories()
    ibs.delete_cachedir()
    ibs.ensure_directories()
    if delete_chips:
        ibs.delete_all_chips()
    if delete_encounters:
        ibs.delete_all_encounters()


@__injectable
def delete_cachedir(ibs):
    """
    Deletes the cache directory in the database directory.

    (does not remove chips)
    """
    print('[ibs] delete_cachedir')
    # Need to close dbcache before restarting
    ibs._close_sqldbcache()
    cachedir = ibs.get_cachedir()
    print('[ibs] cachedir=%r' % cachedir)
    ut.delete(cachedir)
    print('[ibs] finished delete cachedir')
    # Reinit cache
    ibs.ensure_directories()
    ibs._init_sqldbcache()


@__injectable
def delete_qres_cache(ibs):
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


@__injectable
def delete_neighbor_cache(ibs):
    print('[ibs] delete neighbor_cache')
    neighbor_cachedir = ibs.get_neighbor_cachedir()
    ut.delete(neighbor_cachedir)
    ut.ensuredir(neighbor_cachedir)
    print('[ibs] finished delete neighbor_cache')


@__injectable
def delete_all_features(ibs):
    print('[ibs] delete_all_features')
    all_fids = ibs._get_all_fids()
    ibs.delete_features(all_fids)
    print('[ibs] finished delete_all_features')


@__injectable
def delete_all_chips(ibs):
    print('[ibs] delete_all_chips')
    ut.ensuredir(ibs.chipdir)
    all_cids = ibs._get_all_chip_rowids()
    ibs.delete_chips(all_cids)
    ut.delete(ibs.chipdir)
    ut.ensuredir(ibs.chipdir)
    print('[ibs] finished delete_all_chips')


@__injectable
def delete_all_encounters(ibs):
    print('[ibs] delete_all_encounters')
    all_eids = ibs._get_all_eids()
    ibs.delete_encounters(all_eids)
    print('[ibs] finished delete_all_encounters')


@__injectable
def delete_all_annotations(ibs):
    """ Carefull with this function. Annotations are not recomputable """
    print('[ibs] delete_all_annotations')
    ans = six.moves.input('Are you sure you want to delete all annotations?')
    if ans != 'yes':
        return
    all_aids = ibs._get_all_aids()
    ibs.delete_annots(all_aids)
    print('[ibs] finished delete_all_annotations')


@__injectable
def delete_thumbnails(ibs):
    ut.remove_files_in_dir(ibs.get_thumbdir())


@__injectable
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


@__injectable
def vd(ibs):
    ibs.view_dbdir()


@__injectable
def view_dbdir(ibs):
    ut.view_directory(ibs.get_dbdir())


@__injectable
def get_empty_gids(ibs, eid=None):
    """ returns gid list without any chips """
    gid_list = ibs.get_valid_gids(eid=eid)
    nRois_list = ibs.get_image_num_annotations(gid_list)
    empty_gids = [gid for gid, nRois in zip(gid_list, nRois_list) if nRois == 0]
    return empty_gids


@__injectable
def get_annot_vecs_cache(ibs, aids):
    """ When you have a list with duplicates and you dont want to copy data
    creates a reference to each data object indexed by a dict """
    unique_aids = list(set(aids))
    unique_desc = ibs.get_annot_vecs(unique_aids)
    desc_cache = dict(list(zip(unique_aids, unique_desc)))
    return desc_cache


# TODO: move to const

@__injectable
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
        >>> from ibeis.ibsfuncs import *  # NOQA
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


@__injectable
def get_hard_annot_rowids(ibs):
    valid_aids = ibs.get_valid_aids()
    hard_aids = ut.filter_items(valid_aids, ibs.get_annot_is_hard(valid_aids))
    return hard_aids


@__injectable
def get_easy_annot_rowids(ibs):
    hard_aids = ibs.get_hard_annot_rowids()
    easy_aids = ut.setdiff_ordered(ibs.get_valid_aids(), hard_aids)
    easy_aids = ut.filter_items(easy_aids, ibs.get_annot_has_groundtruth(easy_aids))
    return easy_aids


@__injectable
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
        " Adds or removes hard tag if needed "
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

    new_notes_list = [hack_notes(notes, is_hard, flag) for notes, is_hard, flag in zip(notes_list, is_hard_list, flag_list)]
    ibs.set_annot_notes(aid_list, new_notes_list)
    return is_hard_list


@__injectable
@accessor_decors.getter_1to1
def is_nid_unknown(ibs, nid_list):
    return [ nid <= 0 for nid in nid_list]


@__injectable
def set_annot_names_to_next_name(ibs, aid_list):
    next_name = ibs.make_next_name()
    ibs.set_annot_names(aid_list, [next_name] * len(aid_list))


@__injectable
def _overwrite_annot_species_to_plains(ibs, aid_list):
    species_list = [const.Species.ZEB_PLAIN] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


@__injectable
def _overwrite_annot_species_to_grevys(ibs, aid_list):
    species_list = [const.Species.ZEB_GREVY] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


@__injectable
def _overwrite_annot_species_to_giraffe(ibs, aid_list):
    species_list = [const.Species.GIR] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


@__injectable
def _overwrite_all_annot_species_to(ibs, species):
    """ THIS OVERWRITES A LOT OF INFO """
    assert species in const.VALID_SPECIES, repr(species) + 'is not in ' + repr(const.VALID_SPECIES)
    aid_list = ibs.get_valid_aids()
    species_list = [species] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


# Use new invertible flatten functions
_invertible_flatten = ut.invertible_flatten2
_unflatten = ut.unflatten2


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
        python -m ibeis.ibsfuncs --test-unflat_map

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
    flat_rowids, reverse_list = _invertible_flatten(unflat_rowids)
    # Then preform the lookup / implicit mapping
    flat_vals = method(flat_rowids, **kwargs)

    if True:
        assert len(flat_vals) == len(flat_rowids), (
            'flat lens not the same, len(flat_vals)=%d len(flat_rowids)=%d' %
            (len(flat_vals), len(flat_rowids),))

    # Then _unflatten the results to the original input dimensions
    unflat_vals = _unflatten(flat_vals, reverse_list)

    if True:
        assert len(unflat_vals) == len(unflat_rowids), (
            'unflat lens not the same, len(unflat_vals)=%d len(unflat_rowids)=%d' %
            (len(unflat_vals), len(unflat_rowids),))

    return unflat_vals


def unflat_multimap(method_list, unflat_rowids, **kwargs):
    """ unflat_map, but allows multiple methods
    """
    # First flatten the list, and remember the original dimensions
    flat_rowids, reverse_list = _invertible_flatten(unflat_rowids)
    # Then preform the lookup / implicit mapping
    flat_vals_list = [method(flat_rowids, **kwargs) for method in method_list]
    # Then _unflatten the results to the original input dimensions
    unflat_vals_list = [_unflatten(flat_vals, reverse_list)
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
        flat_rowids, reverse_list = _invertible_flatten(unflat_rowids)
        # Then preform the lookup
        flat_vals = func(self, flat_rowids, *args, **kwargs)
        # Then _unflatten the list
        unflat_vals = _unflatten(flat_vals, reverse_list)
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


def aidstr(aid, ibs=None, notes=False):
    """ Helper to make a string from an RID """
    if not notes:
        return 'aid%d' % (aid,)
    else:
        assert ibs is not None
        notes = ibs.get_annot_notes(aid)
        name  = ibs.get_annot_names(aid)
        return 'aid%d-%r-%r' % (aid, str(name), str(notes))


def vsstr(qaid, aid, lite=False):
    if lite:
        return '%d-vs-%d' % (qaid, aid)
    else:
        return 'qaid%d-vs-aid%d' % (qaid, aid)


@__injectable
@ut.time_func
#@profile
def update_exemplar_special_encounter(ibs):
    # FIXME SLOW
    exemplar_eid = ibs.get_encounter_eids_from_text(const.EXEMPLAR_ENCTEXT)
    #ibs.delete_encounters(exemplar_eid)
    ibs.delete_egr_encounter_relations(exemplar_eid)
    #aid_list = ibs.get_valid_aids(is_exemplar=True)
    #gid_list = ut.unique_ordered(ibs.get_annot_gids(aid_list))
    gid_list = list(set(_get_exemplar_gids(ibs)))
    #ibs.set_image_enctext(gid_list, [const.EXEMPLAR_ENCTEXT] * len(gid_list))
    ibs.set_image_eids(gid_list, [exemplar_eid] * len(gid_list))


@__injectable
@ut.time_func
#@profile
def update_reviewed_unreviewed_image_special_encounter(ibs):
    """
    Creates encounter of images that have not been reviewed
    and that have been reviewed
    """
    # FIXME SLOW
    unreviewed_eid = ibs.get_encounter_eids_from_text(const.UNREVIEWED_IMAGE_ENCTEXT)
    reviewed_eid = ibs.get_encounter_eids_from_text(const.REVIEWED_IMAGE_ENCTEXT)
    #ibs.delete_encounters(eid)
    ibs.delete_egr_encounter_relations(unreviewed_eid)
    ibs.delete_egr_encounter_relations(reviewed_eid)
    #gid_list = ibs.get_valid_gids(reviewed=False)
    #ibs.set_image_enctext(gid_list, [const.UNREVIEWED_IMAGE_ENCTEXT] * len(gid_list))
    unreviewed_gids = _get_unreviewed_gids(ibs)  # hack
    reviewed_gids   = _get_reviewed_gids(ibs)  # hack
    ibs.set_image_eids(unreviewed_gids, [unreviewed_eid] * len(unreviewed_gids))
    ibs.set_image_eids(reviewed_gids, [reviewed_eid] * len(reviewed_gids))


@__injectable
@ut.time_func
#@profile
def update_all_image_special_encounter(ibs):
    # FIXME SLOW
    allimg_eid = ibs.get_encounter_eids_from_text(const.ALL_IMAGE_ENCTEXT)
    #ibs.delete_encounters(allimg_eid)
    gid_list = ibs.get_valid_gids()
    #ibs.set_image_enctext(gid_list, [const.ALL_IMAGE_ENCTEXT] * len(gid_list))
    ibs.set_image_eids(gid_list, [allimg_eid] * len(gid_list))


@__injectable
def get_special_eids(ibs):
    get_enctext_eid = ibs.get_encounter_eids_from_text
    special_enctext_list = [
        const.UNGROUPED_IMAGES_ENCTEXT,
        const.ALL_IMAGE_ENCTEXT,
        const.UNREVIEWED_IMAGE_ENCTEXT,
        const.REVIEWED_IMAGE_ENCTEXT,
        const.EXEMPLAR_ENCTEXT,
    ]
    special_eids_ = [get_enctext_eid(enctext, ensure=False)
                     for enctext in special_enctext_list]
    special_eids = [i for i in special_eids_ if i is not None]
    return special_eids


@__injectable
def get_ungrouped_gids(ibs):
    """
    CommandLine:
        python -m ibeis.ibsfuncs --test-get_ungrouped_gids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> ibs.delete_all_encounters()
        >>> ibs.compute_encounters()
        >>> ibs.update_special_encounters()
        >>> # Now we want to remove some images from a non-special encounter
        >>> nonspecial_eids = [i for i in ibs.get_valid_eids() if i not in ibs.get_special_eids()]
        >>> print("Nonspecial EIDs %r" % nonspecial_eids)
        >>> images_to_remove = ibs.get_encounter_gids(nonspecial_eids[0:1])[0][0:1]
        >>> print("Removing %r" % images_to_remove)
        >>> ibs.unrelate_images_and_encounters(images_to_remove,nonspecial_eids[0:1] * len(images_to_remove))
        >>> ibs.update_special_encounters()
        >>> ungr_eid = ibs.get_encounter_eids_from_text(const.UNGROUPED_IMAGES_ENCTEXT)
        >>> print("Ungrouped gids %r" % ibs.get_ungrouped_gids())
        >>> print("Ungrouped eid %d contains %r" % (ungr_eid, ibs.get_encounter_gids([ungr_eid])))
        >>> ungr_gids = ibs.get_encounter_gids([ungr_eid])[0]
        >>> assert(sorted(images_to_remove) == sorted(ungr_gids))
    """
    special_eids = set(get_special_eids(ibs))
    gid_list = ibs.get_valid_gids()
    eids_list = ibs.get_image_eids(gid_list)
    has_eids = [special_eids.issuperset(set(eids)) for eids in eids_list]
    ungrouped_gids = ut.filter_items(gid_list, has_eids)
    return ungrouped_gids


@__injectable
#@ut.time_func
#@profile
def update_ungrouped_special_encounter(ibs):
    """
    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-update_ungrouped_special_encounter

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb9')
        >>> # execute function
        >>> result = update_ungrouped_special_encounter(ibs)
        >>> # verify results
        >>> print(result)
    """
    # FIXME SLOW
    if ut.VERBOSE:
        print('[ibsfuncs] update_ungrouped_special_encounter.1')
    ungrouped_eid = ibs.get_encounter_eids_from_text(const.UNGROUPED_IMAGES_ENCTEXT)
    if ut.VERBOSE:
        print('[ibsfuncs] update_ungrouped_special_encounter.2')
    ibs.delete_egr_encounter_relations(ungrouped_eid)
    if ut.VERBOSE:
        print('[ibsfuncs] update_ungrouped_special_encounter.3')
    ungrouped_gids = ibs.get_ungrouped_gids()
    if ut.VERBOSE:
        print('[ibsfuncs] update_ungrouped_special_encounter.4')
    ibs.set_image_eids(ungrouped_gids, [ungrouped_eid] * len(ungrouped_gids))
    if ut.VERBOSE:
        print('[ibsfuncs] update_ungrouped_special_encounter.5')


@__injectable
@ut.time_func
#@profile
def update_special_encounters(ibs):
    # FIXME SLOW
    USE_MORE_SPECIAL_ENCOUNTERS = ibs.cfg.other_cfg.ensure_attr('use_more_special_encounters', False)
    if USE_MORE_SPECIAL_ENCOUNTERS:
        #ibs.update_reviewed_unreviewed_image_special_encounter()
        ibs.update_exemplar_special_encounter()
        ibs.update_all_image_special_encounter()
    ibs.update_ungrouped_special_encounter()


def _get_unreviewed_gids(ibs):
    # hack
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {IMAGE_TABLE}
        WHERE
        image_toggle_reviewed=0
        '''.format(**const.__dict__))
    return gid_list


def _get_reviewed_gids(ibs):
    # hack
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {IMAGE_TABLE}
        WHERE
        image_toggle_reviewed=1
        '''.format(**const.__dict__))
    return gid_list


def _get_gids_in_eid(ibs, eid):
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {EG_RELATION_TABLE}
        WHERE
            encounter_rowid==?
        '''.format(**const.__dict__),
        params=(eid,))
    return gid_list


def _get_dirty_reviewed_gids(ibs, eid):
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {EG_RELATION_TABLE}
        WHERE
            encounter_rowid==? AND
            image_rowid NOT IN (SELECT rowid FROM {IMAGE_TABLE} WHERE image_toggle_reviewed=1)
        '''.format(**const.__dict__),
        params=(eid,))
    return gid_list


def _get_exemplar_gids(ibs):
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {ANNOTATION_TABLE}
        WHERE annot_exemplar_flag=1
        '''.format(**const.__dict__))
    return gid_list


#@__injectable
#def print_stats(ibs):
    #from ibeis.other import dbinfo
    #dbinfo.dbstats(ibs)


@__injectable
def print_dbinfo(ibs, **kwargs):
    from ibeis.other import dbinfo
    dbinfo.get_dbinfo(ibs, *kwargs)


@__injectable
def print_infostr(ibs, **kwargs):
    print(ibs.get_infostr())


@__injectable
def print_annotation_table(ibs, verbosity=1, exclude_columns=[]):
    """
    Dumps annotation table to stdout

    Args:
        ibs (IBEISController):
        verbosity (int):

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
    print('\n')
    print(ibs.db.get_table_csv(const.ANNOTATION_TABLE, exclude_columns=exclude_columns))


@__injectable
def print_chip_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.CHIP_TABLE))


@__injectable
def print_feat_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.FEATURE_TABLE, exclude_columns=[
        'feature_keypoints', 'feature_vecs']))


@__injectable
def print_image_table(ibs, **kwargs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.IMAGE_TABLE, **kwargs))
    #, exclude_columns=['image_rowid']))


@__injectable
def print_party_table(ibs, **kwargs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.PARTY_TABLE, **kwargs))
    #, exclude_columns=['image_rowid']))


@__injectable
def print_lblannot_table(ibs, **kwargs):
    """ Dumps lblannot table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.LBLANNOT_TABLE, **kwargs))


@__injectable
def print_name_table(ibs, **kwargs):
    """ Dumps name table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.NAME_TABLE, **kwargs))


@__injectable
def print_species_table(ibs, **kwargs):
    """ Dumps species table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.SPECIES_TABLE, **kwargs))


@__injectable
def print_alr_table(ibs, **kwargs):
    """ Dumps alr table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.AL_RELATION_TABLE, **kwargs))


@__injectable
def print_config_table(ibs, **kwargs):
    """ Dumps config table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.CONFIG_TABLE, **kwargs))


@__injectable
def print_encounter_table(ibs, **kwargs):
    """ Dumps encounter table to stdout

    Kwargs:
        exclude_columns (list):
    """
    print('\n')
    print(ibs.db.get_table_csv(const.ENCOUNTER_TABLE, **kwargs))


@__injectable
def print_egpairs_table(ibs, **kwargs):
    """ Dumps egpairs table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(const.EG_RELATION_TABLE, **kwargs))


@__injectable
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


@__injectable
def print_contributor_table(ibs, verbosity=1, exclude_columns=[]):
    """
    Dumps annotation table to stdout

    Args:
        ibs (IBEISController):
        verbosity (int):

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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


@__injectable
def is_aid_unknown(ibs, aid_list):
    """
    Returns if an annotation has been given a name (even if that name is temporary)
    """
    nid_list = ibs.get_annot_name_rowids(aid_list)
    return ibs.is_nid_unknown(nid_list)


def make_enctext_list(eid_list, enc_cfgstr):
    # DEPRICATE
    enctext_list = [str(eid) + enc_cfgstr for eid in eid_list]
    return enctext_list


@__injectable
def batch_rename_consecutive_via_species(ibs, eid=None):
    """ actually sets the new consectuive names"""
    new_nid_list, new_name_list = ibs.get_consecutive_newname_list_via_species(eid=eid)

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


@__injectable
def get_consecutive_newname_list_via_species(ibs, eid=None):
    """
    Just creates the nams, but does not set them

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_consecutive_newname_list_via_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> # execute function
        >>> eid = None
        >>> new_nid_list, new_name_list = get_consecutive_newname_list_via_species(ibs, eid=eid)
        >>> result = ut.list_str((new_nid_list, new_name_list))
        >>> # verify results
        >>> print(result)
        (
            [1, 2, 3, 4, 5, 6, 7],
            ['IBEIS_PZ_0001', 'IBEIS_PZ_0002', 'IBEIS_UNKNOWN_0001', 'IBEIS_UNKNOWN_0002', 'IBEIS_GZ_0001', 'IBEIS_PB_0001', 'IBEIS_UNKNOWN_0003'],
        )

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> ibs.delete_all_encounters()
        >>> ibs.compute_encounters()
        >>> # execute function
        >>> eid = ibs.get_valid_eids()[1]
        >>> new_nid_list, new_name_list = get_consecutive_newname_list_via_species(ibs, eid=eid)
        >>> result = ut.list_str((new_nid_list, new_name_list))
        >>> # verify results
        >>> print(result)
        (
            [4, 5, 6, 7],
            [u'IBEIS_UNKNOWN_Encounter_1_0001', u'IBEIS_GZ_Encounter_1_0001', u'IBEIS_PB_Encounter_1_0001', u'IBEIS_UNKNOWN_Encounter_1_0002'],
        )
    """
    print('[ibs] get_consecutive_newname_list_via_species')
    ibs.delete_empty_nids()
    nid_list = ibs.get_valid_nids(eid=eid)
    #name_list = ibs.get_name_texts(nid_list)
    aids_list = ibs.get_name_aids(nid_list)
    species_rowids_list = ibs.unflat_map(ibs.get_annot_species_rowids, aids_list)
    unique_species_rowids_list = list(map(ut.unique_keep_order2, species_rowids_list))
    # TODO: ibs.duplicate_map
    unique_species_texts_list = ibs.unflat_map(ibs.get_species_texts, unique_species_rowids_list)
    species_codes = [list(map(const.get_species_code, texts)) for texts in unique_species_texts_list]
    code_list = ['_'.join(codes) for codes in species_codes]

    _code2_count = ut.ddict(lambda: 0)
    def get_next_index(code):
        _code2_count[code] += 1
        return _code2_count[code]

    location_text = ibs.cfg.other_cfg.location_for_names
    if eid is not None:
        enc_text = ibs.get_encounter_text(eid)
        enc_text = enc_text.replace(' ', '_').replace('\'', '').replace('"', '')
        new_name_list = ['%s_%s_%s_%04d' % (location_text, code, enc_text, get_next_index(code)) for code in code_list]
    else:
        new_name_list = ['%s_%s_%04d' % (location_text, code, get_next_index(code)) for code in code_list]
    new_nid_list = nid_list
    return new_nid_list, new_name_list


@__injectable
def set_annot_names_to_same_new_name(ibs, aid_list):
    new_nid = ibs.make_next_nids(num=1)[0]
    if ut.VERBOSE:
        print('Setting aid_list={aid_list} to have new_nid={new_nid}'.format(
            aid_list=aid_list, new_nid=new_nid))
    ibs.set_annot_name_rowids(aid_list, [new_nid] * len(aid_list))


@__injectable
def set_annot_names_to_different_new_names(ibs, aid_list):
    new_nid_list = ibs.make_next_nids(num=len(aid_list))
    if ut.VERBOSE:
        print('Setting aid_list={aid_list} to have new_nid_list={new_nid_list}'.format(
            aid_list=aid_list, new_nid_list=new_nid_list))
    ibs.set_annot_name_rowids(aid_list, new_nid_list)


@__injectable
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


@__injectable
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
        python -m ibeis.ibsfuncs --test-make_next_name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs1 = ibeis.opendb('testdb1')
        >>> ibs2 = ibeis.opendb('PZ_MTEST')
        >>> ibs3 = ibeis.opendb('NAUT_test')
        >>> #ibs5 = ibeis.opendb('GIR_Tanya')
        >>> num = None
        >>> str_format = 2
        >>> # execute function
        >>> next_name1 = make_next_name(ibs1, num, str_format)
        >>> next_name2 = make_next_name(ibs2, num, str_format)
        >>> next_name3 = make_next_name(ibs3, num, str_format)
        >>> next_name4 = make_next_name(ibs1, num, str_format, const.Species.ZEB_GREVY)
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
            species_code = const.get_species_code(species_text)
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


def hack(ibs):
    #ibs.get_encounter_text(eid_list)
    #eid = ibs.get_encounter_eids_from_text("NNP GZC Car '1PURPLE'")

    def get_name_linked_encounters_by_eid(ibs, eid):
        import utool as ut
        #gid_list = ibs.get_encounter_gids(eid)
        aid_list_ = ibs.get_encounter_aids(eid)
        aid_list = ut.filterfalse_items(aid_list_, ibs.is_aid_unknown(aid_list_))

        #all(ibs.db.check_rowid_exists(const.ANNOTATION_TABLE, aid_list))
        #aids_list2 = ibs.get_image_aids(gid_list)
        #assert ut.flatten(aids_list2) == aids_list1
        nid_list = list(set(ibs.get_annot_nids(aid_list, distinguish_unknowns=False)))
        # remove unknown annots
        name_eids = ibs.get_name_eids(nid_list)
        name_enctexts = ibs.get_encounter_text(name_eids)
        return name_enctexts

    eid_list = ibs.get_valid_eids()
    linked_enctexts = [get_name_linked_encounters_by_eid(ibs, eid) for eid in eid_list]
    enctext_list = ibs.get_encounter_text(eid_list)
    print(ut.dict_str(dict(zip(eid_list, linked_enctexts))))
    print(ut.align(ut.dict_str(dict(zip(enctext_list, linked_enctexts))), ':'))
    print(ut.align(ut.dict_str(dict(zip(enctext_list, eid_list)), sorted_=True), ':'))

    #if False:
    #    eids_with_bad_names = [6, 7, 16]
    #    bad_nids = ut.unique_keep_order2(ut.flatten(ibs.get_encounter_nids(eids_with_bad_names)))


def draw_thumb_helper(tup):
    thumb_path, thumbsize, gpath, bbox_list, theta_list = tup
    img = vt.imread(gpath)  # time consuming
    (gh, gw) = img.shape[0:2]
    img_size = (gw, gh)
    max_dsize = (thumbsize, thumbsize)
    dsize, sx, sy = vt.resized_clamped_thumb_dims(img_size, max_dsize)
    new_verts_list = list(vt.scaled_verts_from_bbox_gen(bbox_list, theta_list, sx, sy))
    #thumb = vt.resize_thumb(img, max_dsize)
    # -----------------
    # Actual computation
    thumb = vt.resize(img, dsize)
    orange_bgr = (0, 128, 255)
    for new_verts in new_verts_list:
        thumb = vt.draw_verts(thumb, new_verts, color=orange_bgr, thickness=2)
    vt.imwrite(thumb_path, thumb)
    return True
    #return (thumb_path, thumb)


@__injectable
def preprocess_image_thumbs(ibs, gid_list=None, use_cache=True, chunksize=8,
                            draw_annots=True, thumbsize=None, **kwargs):
    """ Computes thumbs of images in parallel based on kwargs """
    if gid_list is None:
        gid_list = ibs.get_valid_gids(**kwargs)
    thumbsize = ibs.get_image_thumbsize(thumbsize, draw_annots)
    print('[ibsfuncs] preprocess_image_thumbs(use_cache=%r, thumbsize=%r)' % (use_cache, thumbsize))
    thumbpath_list = ibs.get_image_thumbpath_(gid_list,
                                              draw_annots=draw_annots,
                                              thumbsize=thumbsize)
    if use_cache:
        # Filter out paths gids that already have existing thumb paths
        exists_list = list(map(exists, thumbpath_list))
        gid_list_ = ut.filterfalse_items(gid_list, exists_list)
        thumbpath_list_ = ut.filterfalse_items(thumbpath_list, exists_list)
    else:
        gid_list_ = gid_list
        thumbpath_list_ = thumbpath_list

    compute_image_thumbs(ibs, gid_list_, thumbpath_list_, chunksize, draw_annots, thumbsize)
    return thumbpath_list


def compute_image_thumbs(ibs, gid_list_, thumbpath_list_, chunksize, draw_annots, thumbsize):
    """
    Parallel work function. Computes image thumbnails. Overwrites anything that exists.
    Does not use any caching
    """
    gpath_list = ibs.get_image_paths(gid_list_)
    aids_list = ibs.get_image_aids(gid_list_)
    if draw_annots:
        bboxes_list = unflat_map(ibs.get_annot_bboxes, aids_list)
        thetas_list = unflat_map(ibs.get_annot_thetas, aids_list)
    else:
        bboxes_list = [ [] for aids in aids_list ]
        thetas_list = [ [] for aids in aids_list ]
    args_list = [(thumb_path, thumbsize, gpath, bbox_list, theta_list)
                 for thumb_path, gpath, bbox_list, theta_list in
                 zip(thumbpath_list_, gpath_list, bboxes_list, thetas_list)]

    # Execute all tasks in parallel
    genkw = {
        'ordered': False,
        'chunksize': chunksize,
        #'force_serial': True,
    }
    gen = ut.generate(draw_thumb_helper, args_list, nTasks=len(args_list), **genkw)
    ut.evaluate_generator(gen)


@__injectable
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
        python -m ibeis.ibsfuncs --test-group_annots_by_name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
    nid_list = np.array(ibs.get_annot_name_rowids(aid_list, distinguish_unknowns=distinguish_unknowns))
    unique_nids, groupxs_list = vt.group_indices(nid_list)
    grouped_aids_ = vt.apply_grouping(np.array(aid_list), groupxs_list)
    return grouped_aids_, unique_nids


def group_annots_by_known_names_nochecks(ibs, aid_list):
    nid_list = ibs.get_annot_name_rowids(aid_list)
    nid2_aids = ut.group_items(aid_list, nid_list)
    return list(nid2_aids.values())


@__injectable
def group_annots_by_known_names(ibs, aid_list, checks=True):
    r"""
    FIXME; rectify this
    #>>> import ibeis  # NOQA

    CommandLine:
        python -m ibeis.ibsfuncs --test-group_annots_by_known_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
    aid_gen = lambda: six.itervalues(nid2_aids)
    isunknown_list = ibs.is_nid_unknown(six.iterkeys(nid2_aids))
    known_aids_list = list(ut.ifilterfalse_items(aid_gen(), isunknown_list))
    unknown_aids = list(ut.iflatten(ut.ifilter_items(aid_gen(), isunknown_list)))
    if __debug__:
        # References:
        #     http://stackoverflow.com/questions/482014/how-would-you-do-the-equivalent-of-preprocessor-directives-in-python
        nidgroup_list = unflat_map(ibs.get_annot_name_rowids, known_aids_list)
        for nidgroup in nidgroup_list:
            assert ut.list_allsame(nidgroup), 'bad name grouping'
    return known_aids_list, unknown_aids


@__injectable
def get_upsize_data(ibs, qaid_list, daid_list=None, num_samp=5, clamp_gt=1,
                    clamp_gf=1, seed=False):
    """
    Returns qaids and a corresponding list of lists for true matches and false
    matches to try.

    each item in the zip(*upsizetup) is qaid, true_aids, false_aids_samples
    which corresponds to a query aid and a list of true aids and false aids
    to try it as a query against.

    get_upsize_data
    DEPRICATE

    Args:
        ibs (IBEISController):
        qaid_list (int): query annotation id
        daid_list (list):
        num_samp (int):
        clamp_gt (int):
        clamp_gf (int):
        seed (int): if False seed is random else seeds numpy random num gen

    Returns:
        tuple: upsizetup

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_upsize_data

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()
        >>> daid_list = None
        >>> num_samp = 5
        >>> clamp_gt = 1
        >>> clamp_gf = 1
        >>> seed = 143039
        >>> upsizetup = get_upsize_data(ibs, qaid_list, daid_list, num_samp, clamp_gt, clamp_gf, seed)
        >>> (qaid_list, qaid_trues_list, qaid_false_samples_list, nTotal) = upsizetup
        >>> ut.assert_eq(len(qaid_list), 119, var1_name='len(qaid_list)')
        >>> ut.assert_eq(len(qaid_trues_list), 119, var1_name='len(qaid_trues_list)')
        >>> ut.assert_eq(len(qaid_false_samples_list), 119, var1_name='len(qaid_false_samples_list)')
        >>> ut.assert_eq(nTotal, 525, var1_name='nTotal')
        >>> qaid, true_aids, false_aids_samples = six.next(zip(qaid_list, qaid_trues_list, qaid_false_samples_list))
        >>> result = ut.hashstr(str(upsizetup))
        >>> print(result)
        objl8qnhyics@0cr

    b9lvi3nz&ld9u8rg
    """
    if seed is not False:
        # Determanism
        np.random.seed(seed)
    if daid_list is None:
        daid_list = ibs.get_valid_aids()
    # List of database sizes to test
    samp_min, samp_max = (2, ibs.get_num_names())
    dbsamplesize_list = ut.sample_domain(samp_min, samp_max, num_samp)
    #
    # Sample true and false matches for every query annotation
    qaid_trues_list = ibs.get_annot_groundtruth_sample(qaid_list, per_name=clamp_gt, isexemplar=None)
    qaid_falses_list = ibs.get_annot_groundfalse_sample(qaid_list, per_name=clamp_gf)
    #
    # Vary the size of the falses
    def generate_varied_falses():
        for false_aids in qaid_falses_list:
            false_sample_list = []
            for dbsize in dbsamplesize_list:
                if dbsize > len(false_aids):
                    continue
                false_sample = np.random.choice(false_aids, dbsize, replace=False).tolist()
                false_sample_list.append(false_sample)
            yield false_sample_list
    qaid_false_samples_list = list(generate_varied_falses())

    #
    # Get a rough idea of how many queries will be run
    nTotal = sum([len(false_aids_samples) * len(true_aids)
                  for true_aids, false_aids_samples
                  in zip(qaid_false_samples_list, qaid_trues_list)])
    upsizetup = (qaid_list, qaid_trues_list, qaid_false_samples_list, nTotal)
    return upsizetup


@__injectable
def get_annot_groundfalse_sample(ibs, aid_list, per_name=1, seed=False):
    """
    get_annot_groundfalse_sample

    FIXME
    DEPRICATE

    Args:
        ibs (IBEISController):
        aid_list (list):
        per_name (int): number of groundfalse per name
        seed (bool or int): if False no seed, otherwise seeds numpy randgen

    Returns:
        list: gf_aids_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[::4]
        >>> per_name = 1
        >>> seed = 42
        >>> sample_trues_list = get_annot_groundfalse_sample(ibs, aid_list, per_name, seed)
        >>> #result = str(sample_trues_list)
        >>> #print(result)

    [[3, 5, 7, 8, 10, 12, 13], [3, 7, 8, 10, 12, 13], [3, 6, 7, 8, 10, 12, 13], [2, 6, 7, 8, 10, 12]]
    [[2, 6, 7, 8, 10, 12, 13], [2, 7, 8, 10, 12, 13], [2, 5, 7, 8, 10, 12, 13], [2, 6, 7, 8, 10, 12]]
    [[2, 5, 7, 8, 10, 12, 13], [3, 7, 8, 10, 12, 13], [2, 5, 7, 8, 10, 12, 13], [3, 5, 7, 8, 10, 12]]
    """
    if seed is not False:
        # Determanism
        np.random.seed(seed)
    # Get valid names
    valid_aids = ibs.get_valid_aids()
    valid_nids = ibs.get_annot_name_rowids(valid_aids)
    nid2_aids = ut.group_items(valid_aids, valid_nids)
    for nid in list(nid2_aids.keys()):
        if ibs.is_nid_unknown(nid):
            # Remove unknown
            del nid2_aids[nid]
            continue
        # Cast into numpy arrays
        aids =  np.array(nid2_aids[nid])
        if len(aids) == 0:
            # Remove empties
            print('[ibsfuncs] name with 0 aids. need to clean database')
            del nid2_aids[nid]
            continue
        nid2_aids[nid] = aids
        # Shuffle known annotations in each name
        #np.random.shuffle(aids)
    # Get not beloning to input names
    nid_list = ibs.get_annot_name_rowids(aid_list)
    def _sample(nid_):
        aids_iter = (aids for nid, aids in six.iteritems(nid2_aids) if nid != nid_)
        sample_gf_aids = np.hstack([np.random.choice(aids, per_name, replace=False) for aids in aids_iter])
        return sample_gf_aids.tolist()
    gf_aids_list = [_sample(nid_) for nid_ in nid_list]
    return gf_aids_list


@__injectable
def get_annot_groundtruth_sample(ibs, aid_list, per_name=1, isexemplar=True):
    r"""
    get_annot_groundtruth_sample

    DEPRICATE

    Args:
        ibs (IBEISController):
        aid_list (list):
        per_name (int):

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_annot_groundtruth_sample --verbose-class
        python -m ibeis.ibsfuncs --test-get_annot_groundtruth_sample:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[::2]
        >>> per_name = 1
        >>> result = get_annot_groundtruth_sample(ibs, aid_list, per_name)
        >>> print(result)
        [[], [2], [6], [], [], [], []]

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb(ut.get_argval('--db', str, 'testdb1'))
        >>> aid_list = ibs.get_valid_aids()
        >>> per_name = 1
        >>> result = get_annot_groundtruth_sample(ibs, aid_list, per_name)
        >>> print(result)
    """
    all_trues_list = ibs.get_annot_groundtruth(aid_list, noself=True, is_exemplar=isexemplar)
    def random_choice(aids):
        size = min(len(aids), per_name)
        return np.random.choice(aids, size, replace=False).tolist()
    sample_trues_list = [random_choice(aids) if len(aids) > 0 else [] for aids in all_trues_list]
    return sample_trues_list


@__injectable
def get_one_annot_per_name(ibs, col='rand'):
    r"""

    DEPRICATE

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_one_annot_per_name --db PZ_Master0
        python -m ibeis.ibsfuncs --test-get_one_annot_per_name --db PZ_MTEST
        python -m ibeis.ibsfuncs --test-get_one_annot_per_name --dbdir /raid/work2/Turk/GIR_Master

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> result = get_one_annot_per_name(ibs)
        >>> # verify results
        >>> print(result)
    """
    #nid_list = ibs.get_valid_nids()
    #aids_list = ibs.get_name_aids(nid_list)
    #num_annots_list = list(map(len, aids_list))
    #aids_list = ut.sortedby(aids_list, num_annots_list, reverse=True)
    #aid_list = ut.get_list_column(aids_list, 0)
    # Keep only a certain number of annots for distinctiveness mapping
    #aid_list_ = ut.listclip(aid_list, max_annots)
    aid_list_ = ibs.get_valid_aids()
    aids_list, nid_list = ibs.group_annots_by_name(aid_list_, distinguish_unknowns=True)
    if col == 'rand':
        def random_choice(aids):
            size = min(len(aids), 1)
            return np.random.choice(aids, size, replace=False).tolist()
        aid_list = [random_choice(aids)[0] if len(aids) > 0 else [] for aids in aids_list]
    else:
        aid_list = ut.get_list_column(aids_list, 0)
    allow_unnamed = True
    if not allow_unnamed:
        raise NotImplementedError('fixme')
    if col == 'rand':
        import random
        random.shuffle(aid_list)
    return aid_list


@__injectable
def get_annot_rowid_sample(ibs, aid_list=None, per_name=1, min_gt=1,
                           method='random', seed=0,
                           offset=0, stagger_names=False, distinguish_unknowns=True, grouped_aids=None):
    r"""
    Gets a sampling of annotations

    DEPRICATE

    Args:
        per_name (int): number of annotations per name
        min_ngt (int): filters any name with less than this number of annotations
        seed (int): random seed
        aid_list (list): base aid_list to start with. If None
        get_valid_aids(nojunk=True) is used stagger_names (bool): if True
        staggers the order of the returned sample

    Returns:
        list: sample_aids

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_annot_rowid_sample

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> per_name = 3
        >>> min_gt = 1
        >>> seed = 0
        >>> # execute function
        >>> sample_aid_list = ibs.get_annot_rowid_sample(None, per_name, min_gt, seed)
        >>> result = ut.hashstr_arr(sample_aid_list)
        arr((66)crj9l5jde@@hdmlp)
    """
    #qaids = ibs.get_easy_annot_rowids()
    if grouped_aids is None:
        if aid_list is None:
            aid_list = np.array(ibs.get_valid_aids(nojunk=True))
        grouped_aids_, unique_nids = ibs.group_annots_by_name(aid_list, distinguish_unknowns=distinguish_unknowns)
        if min_gt is None:
            grouped_aids = grouped_aids_
        else:
            grouped_aids = list(filter(lambda x: len(x) >= min_gt, grouped_aids_))
    else:
        # grouped aids was precomputed
        pass
    if method == 'random2':
        # always returns per_name when available
        sample_aids_list = ut.sample_lists(grouped_aids, num=per_name, seed=seed)
    elif method == 'random':
        # Random that allows for offset.
        # may return less than per_name when available if offset > 0
        rng = np.random.RandomState(seed)
        for aids in grouped_aids:
            rng.shuffle(aids)
        sample_aids_list = ut.get_list_column_slice(grouped_aids, offset, offset + per_name)
    elif method == 'simple':
        sample_aids_list = ut.get_list_column_slice(grouped_aids, offset, offset + per_name)
    else:
        raise NotImplementedError('method = %r' % (method,))
    if stagger_names:
        from six.moves import zip_longest
        sample_aid_list = ut.filter_Nones(ut.iflatten(zip_longest(*sample_aids_list)))
    else:
        sample_aid_list = ut.flatten(sample_aids_list)

    return sample_aid_list


def get_primary_species_viewpoint(species, plus=0):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        species (?):

    Returns:
        str: primary_viewpoint

    CommandLine:
        python -m ibeis.ibsfuncs --exec-get_primary_species_viewpoint

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> species = ibeis.const.Species.ZEB_PLAIN
        >>> aid_subset = get_primary_species_viewpoint(species, 0)
        >>> result = ('aid_subset = %s' % (str(aid_subset),))
        >>> print(result)
        aid_subset = left
    """
    if species == const.Species.ZEB_PLAIN:
        primary_viewpoint = 'left'
    elif species == const.Species.ZEB_GREVY:
        primary_viewpoint = 'right'
    else:
        primary_viewpoint = 'left'
    if plus != 0:
        # return an augmented primary viewpoint
        primary_viewpoint = get_extended_viewpoints(primary_viewpoint, num1=1, num2=0, include_base=False)[0]
    return primary_viewpoint


def get_extended_viewpoints(base_yaw_text, towards='front', num1=0, num2=None, include_base=True):
    """
    Given a viewpoint returns the acceptable viewpoints around it

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> yaw_text_list = ['left', 'right', 'back', 'front']
        >>> towards = 'front'
        >>> num1 = 1
        >>> num2 = 0
        >>> include_base = False
        >>> extended_yaws_list = [get_extended_viewpoints(base_yaw_text, towards, num1, num2, include_base)
        >>>                       for base_yaw_text in yaw_text_list]
        >>> result = ('extended_yaws_list = %s' % (ut.list_str(extended_yaws_list),))
        >>> print(result)
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
    if num2 is None:
        num2 = num1
    assert num1 >= 0, 'must specify positive num'
    assert num2 >= 0, 'must specify positive num'
    yawtext_list = list(const.VIEWTEXT_TO_YAW_RADIANS.keys())
    index = yawtext_list.index(base_yaw_text)
    other_index_list1 = [int((index + (np.sign(yawdist) * count)) % len(yawtext_list)) for count in range(1, num1 + 1)]
    other_index_list2 = [int((index - (np.sign(yawdist) * count)) % len(yawtext_list)) for count in range(1, num2 + 1)]
    if include_base:
        extended_index_list = sorted(list(set(other_index_list1 + other_index_list2 + [index])))
    else:
        extended_index_list = sorted(list(set(other_index_list1 + other_index_list2)))
    extended_yaws = ut.list_take(yawtext_list, extended_index_list)
    return extended_yaws


def get_two_annots_per_name_and_singletons(ibs, onlygt=False):
    """
    makes controlled subset of data

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
        python -m ibeis.ibsfuncs --test-get_two_annots_per_name_and_singletons
        python -m ibeis.ibsfuncs --test-get_two_annots_per_name_and_singletons --db GZ_ALL
        python -m ibeis.ibsfuncs --test-get_two_annots_per_name_and_singletons --db PZ_Master0 --onlygt

    Ignore:
        sys.argv.extend(['--db', 'PZ_MTEST'])

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master0')
        >>> aid_subset = get_two_annots_per_name_and_singletons(ibs, onlygt=ut.get_argflag('--onlygt'))
        >>> ibeis.other.dbinfo.get_dbinfo(ibs, aid_list=aid_subset, with_contrib=False)
        >>> result = str(aid_subset)
        >>> print(result)
    """
    species = get_primary_database_species(ibs, ibs.get_valid_aids())
    #aid_list = ibs.get_valid_aids(species=ibs.const.Species.ZEB_PLAIN, is_known=True)
    aid_list = ibs.get_valid_aids(species=species, is_known=True)
    # FILTER OUT UNUSABLE ANNOTATIONS
    # Get annots with timestamps
    aid_list = filter_aids_without_timestamps(ibs, aid_list)
    minqual = 'ok'
    #valid_yaws = {'left', 'frontleft', 'backleft'}
    if species == ibs.const.Species.ZEB_PLAIN:
        valid_yawtexts = {'left', 'frontleft'}
    elif species == ibs.const.Species.ZEB_GREVY:
        valid_yawtexts = {'right', 'frontright'}
    else:
        valid_yawtexts = {'left', 'frontleft'}
    flags_list = ibs.get_quality_viewpoint_filterflags(aid_list, minqual, valid_yawtexts)
    aid_list = ut.list_compress(aid_list, flags_list)
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

    """
    if False:
        qaids = aid_subset
        daids = aid_list
        def get_close_groundfalse_pairs(qaids, daids):
            valid_daids_list = [ibs.get_annot_groundfalse(qaid, daid_list=daids) for qaid in qaids]
            for qaid, valid_daids in zip(qaids, valid_daids_list):
                query_unixtime = ibs.get_annot_image_unixtimes(qaid)
                valid_data_unixtimes = np.array(ibs.get_annot_image_unixtimes(valid_daids))
                hourdiffs = ut.unixtime_hourdiff(query_unixtime, valid_data_unixtimes)
                valid_daids[hourdiffs.argmin()]
    """

    #best_hourdists_list = ut.flatten(ibs.get_unflat_annots_hourdists_list(best_multitons))
    #assert len(best_hourdists_list) == len(best_multitons)
    #best_multitons_sortx = np.array(best_hourdists_list).argsort()[::-1]
    #best_pairs = ut.list_take(best_multitons, best_multitons_sortx)
    #best_multis = ut.flatten(best_pairs)

    # process singletons
    #[aids for aids in zip(aids, hour_dists_list]
    return aid_subset


@__injectable
def get_num_annots_per_name(ibs, aid_list):
    """
    Returns the number of annots per name (IN THIS LIST)

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    CommandLine:
        python -m ibeis.ibsfuncs --exec-get_num_annots_per_name

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> num_annots_per_name, unique_nids = get_num_annots_per_name(ibs, aid_list)
        >>> result = ('num_annots_per_name = %r' % (num_annots_per_name,))
        >>> print(result)
        num_annots_per_name = [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1]
    """
    aids_list, unique_nids  = ibs.group_annots_by_name(aid_list)
    num_annots_per_name = list(map(len, aids_list))
    return num_annots_per_name, unique_nids


@__injectable
def get_aids_with_groundtruth(ibs):
    """ returns aids with valid groundtruth """
    valid_aids = ibs.get_valid_aids()
    has_gt_list = ibs.get_annot_has_groundtruth(valid_aids)
    hasgt_aids = ut.filter_items(valid_aids, has_gt_list)
    return hasgt_aids


@__injectable
def get_dbnotes_fpath(ibs, ensure=False):
    notes_fpath = join(ibs.get_ibsdir(), 'dbnotes.txt')
    if ensure and not exists(ibs.get_dbnotes_fpath()):
        ibs.set_dbnotes('None')
    return notes_fpath


@profile
def get_yaw_viewtexts(yaw_list):
    r"""
    Args:
        yaw (?):

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_yaw_viewtexts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import numpy as np
        >>> # build test data
        >>> yaw_list = [0.0, np.pi / 2, np.pi / 4, np.pi, 3.15, -.4, -8, .2, 4, 7, 20, None]
        >>> # execute function
        >>> text_list = get_yaw_viewtexts(yaw_list)
        >>> result = str(text_list)
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
        textdists_list = [None if yaw is None else vt.ori_distance(stdyaw_list, yaw) for yaw in yaw_list]
        index_list = [None if dists is None else dists.argmin() for dists in textdists_list]
        text_list = [None if index is None else stdlbl_list[index] for index in index_list]
        #yaw_list_ / binsize
    #errors = ['%.2f' % dists[index] for dists, index in zip(textdists_list, index_list)]
    #return list(zip(yaw_list, errors, text_list))
    return text_list


def get_species_dbs(species_prefix):
    from ibeis.init import sysres
    ibs_dblist = sysres.get_ibsdb_list()
    isvalid_list = [split(path)[1].startswith(species_prefix) for path in ibs_dblist]
    return ut.filter_items(ibs_dblist, isvalid_list)


@__injectable
def get_annot_bbox_area(ibs, aid_list):
    bbox_list = ibs.get_annot_bboxes(aid_list)
    area_list = [bbox[2] * bbox[3] for bbox in bbox_list]
    return area_list


@__injectable
def get_match_text(ibs, aid1, aid2):
    truth = ibs.get_match_truth(aid1, aid2)
    text = const.TRUTH_INT_TO_TEXT.get(truth, None)
    return text


@__injectable
def get_database_species(ibs, aid_list=None):
    r"""

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_database_species

    Example1:
        >>> # ENABLE_DOCTEST
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> result = ibs.get_database_species()
        >>> print(result)
        ['____', u'bear_polar', u'zebra_grevys', u'zebra_plains']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> result = ibs.get_database_species()
        >>> print(result)
        [u'zebra_plains']
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    species_list = ibs.get_annot_species_texts(aid_list)
    unique_species = sorted(list(set(species_list)))
    return unique_species


@__injectable
def get_primary_database_species(ibs, aid_list=None):
    r"""
    Args:
        aid_list (list):  list of annotation ids (default = None)

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_primary_database_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
            return const.Species.ZEB_PLAIN
        elif ibs.get_dbname() == 'PZ_Master0':
            return const.Species.ZEB_PLAIN
        elif ibs.get_dbname() == 'NNP_Master':
            return const.Species.ZEB_PLAIN
        elif ibs.get_dbname() == 'GZ_ALL':
            return const.Species.ZEB_GREVY
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    species_list = ibs.get_annot_species_texts(aid_list)
    species_hist = ut.dict_hist(species_list)
    if len(species_hist) == 0:
        primary_species = const.Species.UNKNOWN
    else:
        frequent_species = sorted(species_hist.items(), key=lambda item: item[1], reverse=True)
        primary_species = frequent_species[0][0]
    return primary_species


#def get_dominant_species(ibs, aid_list):
#    r"""
#    Args:
#        aid_list (int):  list of annotation ids

#    CommandLine:
#        python -m ibeis.ibsfuncs --test-get_dominant_species

#    Example:
#        >>> # ENABLE_DOCTEST
#        >>> from ibeis.ibsfuncs import *  # NOQA
#        >>> import ibeis
#        >>> ibs = ibeis.opendb('testdb1')
#        >>> aid_list = ibs.get_valid_aids()
#        >>> result = get_dominant_species(ibs, aid_list)
#        >>> print(result)
#        zebra_plains
#    """
#    hist_ = ut.dict_hist(ibs.get_annot_species_texts(aid_list))
#    keys = hist_.keys()
#    vals = hist_.values()
#    species_text = keys[ut.list_argmax(vals)]
#    return species_text


@__injectable
def get_database_species_count(ibs, aid_list=None):
    """

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_database_species_count

    Example:
        >>> # ENABLE_DOCTEST
        >>> import ibeis  # NOQA
        >>> #print(ut.dict_str(ibeis.opendb('PZ_Master0').get_database_species_count()))
        >>> ibs = ibeis.opendb('testdb1')
        >>> result = ibs.get_database_species_count()
        >>> print(result)
        {u'zebra_plains': 6, '____': 3, u'zebra_grevys': 2, u'bear_polar': 2}
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    species_list = ibs.get_annot_species_texts(aid_list)
    species_count_dict = ut.item_hist(species_list)
    return species_count_dict


@__injectable
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


@__injectable
def get_aidpair_truths(ibs, aid1_list, aid2_list):
    r"""
    Uses NIDS to verify truth

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1_list (list):
        aid2_list (list):

    Returns:
        ?: truth

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_aidpair_truths

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
        wb_target = ibs.get_wildbook_target()
        #params.args.wildbook_target
        if wb_target is not None:
            title = '%s - Wildbook Target = %s' % (title, wb_target)
    return title


@__injectable
def get_dbinfo_str(ibs):
    from ibeis.other import dbinfo
    return dbinfo.get_dbinfo(ibs, verbose=False)['info_str']


@__injectable
def get_infostr(ibs):
    """ Returns sort printable database information

    Args:
        ibs (IBEISController):  ibeis controller object

    Returns:
        str: infostr
    """
    from ibeis.other import dbinfo
    return dbinfo.get_short_infostr(ibs)


@__injectable
def get_dbnotes(ibs):
    """ sets notes for an entire database """
    notes = ut.read_from(ibs.get_dbnotes_fpath(), strict=False)
    if notes is None:
        ibs.set_dbnotes('None')
        notes = ut.read_from(ibs.get_dbnotes_fpath())
    return notes


@__injectable
def set_dbnotes(ibs, notes):
    """ sets notes for an entire database """
    assert isinstance(ibs, ibeis.control.IBEISControl.IBEISController)
    ut.write_to(ibs.get_dbnotes_fpath(), notes)


@__injectable
def annotstr(ibs, aid):
    return 'aid=%d' % aid


@__injectable
def redownload_detection_models(ibs):
    r"""
    Args:
        ibs (IBEISController):

    CommandLine:
        python -c "from ibeis.model.detect import grabmodels; grabmodels.redownload_models()"
        python -c "import utool, ibeis.model; utool.view_directory(ibeis.model.detect.grabmodels._expand_modeldir())"

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> result = redownload_detection_models(ibs)
        >>> print(result)
    """
    print('[ibsfuncs] redownload_detection_models')
    from ibeis.model.detect import grabmodels
    modeldir = ibs.get_detect_modeldir()
    grabmodels.redownload_models(modeldir=modeldir)


@__injectable
def view_model_dir(ibs):
    print('[ibsfuncs] redownload_detection_models')
    modeldir = ibs.get_detect_modeldir()
    ut.view_directory(modeldir)
    #grabmodels.redownload_models(modeldir=modeldir)


@__injectable
def merge_names(ibs, merge_name, other_names):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        merge_name (str):
        other_names (list):

    CommandLine:
        python -m ibeis.ibsfuncs --test-merge_names

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
    aids = ut.filter_items(aids, isnone_list)
    yaws = ut.filter_items(yaws, isnone_list)
    for aid, yaw in zip(aids, yaws):
        print(yaw)
        # We seem to be storing FULL paths in
        # the probchip table
        ibs.delete_annot_chips(aid)
        viz_chip.show_chip(ibs, aid, annote=False)
        pt.show_if_requested()


def detect_false_positives(ibs):
    """
    TODO: this function should detect problems in the database
    It should execute queries for annotations with groundtruth

    then if the groundtruth is no in the top results it is given to the user who
    should try and rectify the problem.

    If the top ranked match is not a groundtruth then it is a true error or hard
    case for the system. To prevent having to review this "hard case" again an
    explicit negative link should be made between the offending annotation pair.

    """
    pass
    #qaid_list = ibs.get_valid_aids(nojunk=True, isknown=True)
    #qres_list = ibs.query_annots(qaid_list)
    #for qres in qres_list:
    #    top_aids = qres.get_top_aids(num=2)


@__injectable
def set_exemplars_from_quality_and_viewpoint(ibs, aid_list=None, exemplars_per_view=None, eid=None, dry_run=False, verbose=False):
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
        python -m ibeis.ibsfuncs --test-set_exemplars_from_quality_and_viewpoint
        python -m ibeis.ibsfuncs --test-set_exemplars_from_quality_and_viewpoint:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> #ibs = ibeis.opendb('PZ_MUGU_19')
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> dry_run = True
        >>> verbose = False
        >>> old_sum = sum(ibs.get_annot_exemplar_flags(ibs.get_valid_aids()))
        >>> new_aid_list, new_flag_list = ibs.set_exemplars_from_quality_and_viewpoint(dry_run=dry_run)
        >>> new_sum = sum(new_flag_list)
        >>> print('old_sum = %r' % (old_sum,))
        >>> print('new_sum = %r' % (new_sum,))
        >>> zero_aid_list, zero_flag_list = ibs.set_exemplars_from_quality_and_viewpoint(exemplars_per_view=0, dry_run=dry_run)
        >>> assert sum(zero_flag_list) == 0
        >>> result = new_sum

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> dry_run = True
        >>> verbose = False
        >>> old_sum = sum(ibs.get_annot_exemplar_flags(ibs.get_valid_aids()))
        >>> new_aid_list, new_flag_list = ibs.set_exemplars_from_quality_and_viewpoint(dry_run=dry_run)
        >>> assert len(new_aid_list) == len(new_flag_list)
        >>> # 2 of the 11 annots are unknown and should not be exemplars
        >>> ut.assert_eq(len(new_aid_list), 9)

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb2')
        >>> dry_run = True
        >>> verbose = False
        >>> eid = None
        >>> new_aid_list, new_flag_list = ibs.set_exemplars_from_quality_and_viewpoint(dry_run=dry_run)
        >>> old_flag_list = ibs.get_annot_exemplar_flags(new_aid_list)
        >>> new_exemplar_aids = ut.filter_items(new_aid_list, new_flag_list)
        >>> new_exemplar_qualtexts = ibs.get_annot_quality_texts(new_exemplar_aids)
        >>> assert 'junk' not in new_exemplar_qualtexts, 'should not have junk exemplars'
        >>> assert 'poor' not in new_exemplar_qualtexts, 'should not have poor exemplars'
        >>> #assert len(new_aid_list) == len(new_flag_list)
        >>> # 2 of the 11 annots are unknown and should not be exemplars
        >>> #ut.assert_eq(len(new_aid_list), 9)
    """
    # General params
    #PREFER_GOOD_OVER_OLDFLAG = True
    #verbose = False
    #dry_run = False
    ##
    ## Params for knapsack
    #def make_knapsack_params(N, levels_per_tier_list):
    #    """
    #    Args:
    #        N (int): the integral maximum number of items
    #        levels_per_tier_list (list): list of number of distinctions possible
    #        per tier.

    #    Returns:
    #        per-item weights, weights go group items into several tiers, and an
    #        infeasible weight
    #    """
    #    EPS = 1E-9
    #    # Solve for the minimum per-item weight
    #    # to allow for preference wiggle room
    #    w = N / (N + 1) + EPS
    #    # level1 perference augmentation
    #    # TODO: figure out mathematically ellegant value
    #    pref_decimator = max(1, (N + EPS)) ** 2  # max is a hack for N = 0
    #    # we want space to specify two levels of tier1 preference
    #    tier_w_list = []
    #    last_w = w
    #    for num_levels in levels_per_tier_list:
    #        last_w = tier_w = last_w / (num_levels * pref_decimator)
    #        tier_w_list.append(tier_w)
    #    infeasible_w = max(9001, N + 1)
    #    return w, tier_w_list, infeasible_w
    #N = exemplars_per_view
    #levels_per_tier_list = [3, 1, 1]
    #w, tier_w_list, infeasible_w = make_knapsack_params(N, levels_per_tier_list)

    #qual2_weight = {
    #    const.QUAL_EXCELLENT : w + tier_w_list[0] + tier_w_list[1],
    #    const.QUAL_GOOD      : w + tier_w_list[0],
    #    const.QUAL_OK        : w + tier_w_list[1],
    #    const.QUAL_UNKNOWN   : w + tier_w_list[1],
    #    const.QUAL_POOR      : w + tier_w_list[2],
    #    const.QUAL_JUNK      : infeasible_w,
    #}
    ## this probably broke with the introduction of 2 more tiers
    #oldflag_offset = (
    #    # always prefer good over ok
    #    tier_w_list[0] - tier_w_list[1]
    #    if PREFER_GOOD_OVER_OLDFLAG else
    #    # prefer ok over good when ok has oldflag
    #    tier_w_list[0] + tier_w_list[1]
    #)

    #def choose_exemplars(aids):
    #    qualtexts = ibs.get_annot_quality_texts(aids)
    #    oldflags = ibs.get_annot_exemplar_flags(aids)
    #    # We like good more than ok, and junk is infeasible We prefer items that
    #    # had previously been exemplars Build input for knapsack
    #    weights = [qual2_weight[qual] + oldflag_offset * oldflag
    #               for qual, oldflag in zip(qualtexts, oldflags)]
    #    #values = [1] * len(weights)
    #    values = weights
    #    indices = list(range(len(weights)))
    #    items = list(zip(values, weights, indices))
    #    total_value, chosen_items = ut.knapsack(items, N)
    #    chosen_indices = ut.get_list_column(chosen_items, 2)
    #    new_flags = [False] * len(aids)
    #    for index in chosen_indices:
    #        new_flags[index] = True
    #    return new_flags

    #def get_changed_infostr(yawtext, aids, new_flags):
    #    old_flags = ibs.get_annot_exemplar_flags(aids)
    #    quals = ibs.get_annot_quality_texts(aids)
    #    ischanged = ut.xor_lists(old_flags, new_flags)
    #    changed_list = ['***' if flag else ''
    #                    for flag in ischanged]
    #    infolist = list(zip(aids, quals, old_flags, new_flags, changed_list))
    #    infostr = ('yawtext=%r:\n' % (yawtext,)) + ut.list_str(infolist)
    #    return infostr

    #aid_list = ibs.get_valid_aids()
    #aids_list, unique_nids  = ibs.group_annots_by_name(aid_list)
    ## for final settings because I'm too lazy to write
    ## this correctly using group_indicies instead of group_items
    #new_aid_list = []
    #new_flag_list = []
    #_iter = ut.ProgressIter(zip(aids_list, unique_nids), nTotal=len(aids_list), lbl='Optimizing name exemplars')
    #for aids_, nid in _iter:
    #    if ibs.is_nid_unknown(nid):
    #        # do not change unknown animals
    #        continue
    #    yawtexts  = ibs.get_annot_yaw_texts(aids_)
    #    yawtext2_aids = ut.group_items(aids_, yawtexts)
    #    if verbose:
    #        print('+ ---')
    #        print('  nid=%r' % (nid))
    #    for yawtext, aids in six.iteritems(yawtext2_aids):
    #        new_flags = choose_exemplars(aids)
    #        if verbose:
    #            print(ut.indent(get_changed_infostr(yawtext, aids, new_flags)))
    #        new_aid_list.extend(aids)
    #        new_flag_list.extend(new_flags)
    #    if verbose:
    #        print('L ___')
    if exemplars_per_view is None:
        exemplars_per_view = ibs.cfg.other_cfg.exemplars_per_view
    if aid_list is None:
        aid_list = ibs.get_valid_aids(eid=eid)
    HACK = ibs.cfg.other_cfg.enable_custom_filter
    #True
    if not HACK:
        new_aid_list, new_flag_list = get_annot_quality_viewpoint_subset(
            ibs, aid_list=aid_list, annots_per_view=exemplars_per_view, verbose=verbose)
    else:
        # HACK
        new_exemplar_aids = ibs.get_prioritized_name_subset(aid_list, exemplars_per_view)
        new_nonexemplar_aids = list(set(aid_list) - set(new_exemplar_aids))
        new_aid_list = new_nonexemplar_aids + new_exemplar_aids
        new_flag_list = [0] * len(new_nonexemplar_aids) + [1] * len(new_exemplar_aids)

    if not dry_run:
        ibs.set_annot_exemplar_flags(new_aid_list, new_flag_list)
    return new_aid_list, new_flag_list


@__injectable
def get_prioritized_name_subset(ibs, aid_list=None, annots_per_name=None):
    """
    TODO: this needs to be integrated more cleanly with a nonhacky way of
    getting a subset of exemplars. Currently ther is duplicate code in guiback
    and here to use left side only when custom filter is on.

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_prioritized_name_subset

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb2')
        >>> aid_list = ibs.get_valid_aids()
        >>> annots_per_name = 2
        >>> aid_subset = get_prioritized_name_subset(ibs, aid_list, annots_per_name)
        >>> qualtexts = ibs.get_annot_quality_texts(aid_subset)
        >>> yawtexts = ibs.get_annot_yaw_texts(aid_subset)
        >>> assert 'junk' not in qualtexts
        >>> assert 'right' not in yawtexts
        >>> result = len(aid_subset)
        >>> print(result)
        28

    Exeample:
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb2')
        >>> aid_list = ibs.get_valid_aids()
        >>> aid_list = ut.list_compress(aid_list, ibs.is_aid_unknown(aid_list))
        >>> annots_per_name = 2
        >>> aid_subset = get_prioritized_name_subset(ibs, aid_list, annots_per_name)
        >>> qualtexts = ibs.get_annot_quality_texts(aid_list)
        >>> yawtexts = ibs.get_annot_yaw_texts(aid_list)
    """
    if annots_per_name is None:
        annots_per_name = ibs.cfg.other_cfg.prioritized_subset_annots_per_name
    if aid_list is None:
        aid_list = ibs.get_valid_aids()

    # Paramaterize?
    qualtext2_weight = {
        const.QUAL_EXCELLENT : 7,
        const.QUAL_GOOD      : 6,
        const.QUAL_OK        : 5,
        const.QUAL_POOR      : 0,
        const.QUAL_UNKNOWN   : 0,
        const.QUAL_JUNK      : 0,
    }

    yawtext2_weight = {
        'right'      : 0,
        'frontright' : 0,
        'front'      : 0,
        'frontleft'  : 3,
        'left'       : 6,
        'backleft'   : 0,
        'back'       : 0,
        'backright'  : 0,
        None         : 0,
    }

    weight_thresh = 7

    qualtext_list = ibs.get_annot_quality_texts(aid_list)
    yawtext_list = ibs.get_annot_yaw_texts(aid_list)

    nid_list = np.array(ibs.get_annot_name_rowids(aid_list, distinguish_unknowns=True))
    unique_nids, groupxs_list = vt.group_indices(nid_list)
    grouped_aids_     = vt.apply_grouping(np.array(aid_list), groupxs_list)
    grouped_qualtexts = vt.apply_grouping(np.array(qualtext_list), groupxs_list)
    grouped_yawtexts  = vt.apply_grouping(np.array(yawtext_list), groupxs_list)
    yaw_weights_list = [
        np.array(ut.dict_take(yawtext2_weight, yawtexts))
        for yawtexts in grouped_yawtexts
    ]
    qual_weights_list = [
        np.array(ut.dict_take(qualtext2_weight, yawtexts))
        for yawtexts in grouped_qualtexts
    ]
    weights_list = [
        yaw_weights + qual_weights
        for yaw_weights, qual_weights in zip(yaw_weights_list, qual_weights_list)
    ]

    sortx_list = [
        weights.argsort()[::-1]
        for weights in weights_list
    ]

    sorted_weight_list = [
        weights.take(order)
        for weights, order in zip(weights_list, sortx_list)
    ]

    sorted_aids_list = [
        aids.take(order)
        for aids, order in zip(grouped_aids_, sortx_list)
    ]

    passed_thresh_list = [
        weights > weight_thresh
        for weights in sorted_weight_list
    ]

    valid_ordered_aids_list = [
        ut.listclip(aids.compress(passed), annots_per_name)
        for aids, passed in zip(sorted_aids_list, passed_thresh_list)
    ]

    aid_subset = ut.flatten(valid_ordered_aids_list)
    return aid_subset


@__injectable
def get_annot_quality_viewpoint_subset(ibs, aid_list=None, annots_per_view=2, verbose=False):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb2')
        >>> aid_list = ibs.get_valid_aids()
        >>> annots_per_view = 2
        >>> new_aid_list, new_flag_list = get_annot_quality_viewpoint_subset(ibs)
        >>> result = sum(new_flag_list)
        >>> print(result)
        38
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()

    PREFER_GOOD_EXEMPLAR_OVER_EXCELLENT = True

    # Params for knapsack
    def make_knapsack_params(N, levels_per_tier_list):
        """
        Args:
            N (int): the integral maximum number of items
            levels_per_tier_list (list): list of number of distinctions possible
            per tier.

        Returns:
            tuple: (w, tier_w_list, infeasible_w)
                w            - is the base weight of all items
                tier_w_list  - is a list of w offsets per tier that does not bring it over 1
                                but suggest a preference for that item.
                infeasible_w - weight of impossible items
        """
        EPS = 1E-9
        # Solve for the minimum per-item weight
        # to allow for preference wiggle room
        w = N / (N + 1) + EPS
        # level1 perference augmentation
        # TODO: figure out mathematically ellegant value
        pref_decimator = max(1, (N + EPS)) ** 2  # max is a hack for N = 0
        # we want space to specify two levels of tier1 preference
        tier_w_list = []
        last_w = w
        for num_levels in levels_per_tier_list:
            last_w = tier_w = last_w / (num_levels * pref_decimator)
            tier_w_list.append(tier_w)
        infeasible_w = max(9001, N + 1)
        return w, tier_w_list, infeasible_w
    levels_per_tier_list = [4, 1, 1, 1]
    w, tier_w_list, infeasible_w = make_knapsack_params(annots_per_view, levels_per_tier_list)

    qual2_weight = {
        const.QUAL_EXCELLENT : tier_w_list[0] * 3,
        const.QUAL_GOOD      : tier_w_list[0] * 2,
        const.QUAL_OK        : tier_w_list[0] * 1,
        const.QUAL_UNKNOWN   : tier_w_list[2],
        const.QUAL_POOR      : tier_w_list[3],
        const.QUAL_JUNK      : infeasible_w,
    }

    exemplar_offset = (
        # always prefer good over ok
        tier_w_list[0] - tier_w_list[1]
        if PREFER_GOOD_EXEMPLAR_OVER_EXCELLENT else
        # prefer ok over good when ok has oldflag
        tier_w_list[0] + tier_w_list[1]
    )
    # this probably broke with the introduction of 2 more tiers

    def get_knapsack_flags(weights, N):
        #values = [1] * len(weights)
        values = weights
        indices = list(range(len(weights)))
        items = list(zip(values, weights, indices))
        total_value, chosen_items = ut.knapsack(items, annots_per_view)
        chosen_indices = ut.get_list_column(chosen_items, 2)
        flags = [False] * len(aids)
        for index in chosen_indices:
            flags[index] = True
        return flags

    def get_chosen_flags(aids, annots_per_view, w, qual2_weight, exemplar_offset):
        qualtexts = ibs.get_annot_quality_texts(aids)
        isexemplar_flags = ibs.get_annot_exemplar_flags(aids)
        # base weight plug preference offsets
        weights = [w + qual2_weight[qual] + exemplar_offset * isexemplar
                   for qual, isexemplar in zip(qualtexts, isexemplar_flags)]
        N = annots_per_view
        flags = get_knapsack_flags(weights, N)
        # We like good more than ok, and junk is infeasible We prefer items that
        # had previously been exemplars Build input for knapsack
        return flags

    nid_list = np.array(ibs.get_annot_name_rowids(aid_list, distinguish_unknowns=True))
    unique_nids, groupxs_list = vt.group_indices(nid_list)
    grouped_aids_ = vt.apply_grouping(np.array(aid_list), groupxs_list)
    #aids = grouped_aids_[-6]
    # for final settings because I'm too lazy to write
    new_aid_list = []
    new_flag_list = []
    _iter = ut.ProgressIter(zip(grouped_aids_, unique_nids), nTotal=len(unique_nids), lbl='Picking best annots per viewpoint')
    for aids_, nid in _iter:
        if ibs.is_nid_unknown(nid):
            # do not change unknown animals
            continue
        # subgroup the names by viewpoints
        yawtexts  = ibs.get_annot_yaw_texts(aids_)
        yawtext2_aids = ut.group_items(aids_, yawtexts)
        for yawtext, aids in six.iteritems(yawtext2_aids):
            flags = get_chosen_flags(aids, annots_per_view, w, qual2_weight, exemplar_offset)
            new_aid_list.extend(aids)
            new_flag_list.extend(flags)
        if verbose:
            print('L ___')
    return new_aid_list, new_flag_list


#@__injectable
#def query_enc_names_vs_exemplars(ibs, exemplars_per_view=2, eid=None):
#    """

#    """
#    aid_list = ibs.get_valid_aids(eid=eid)
#    new_aid_list, new_flag_list = get_annot_quality_viewpoint_subset(
#        ibs, aid_list=aid_list, annots_per_view=exemplars_per_view)
#    qaids = ut.filter_items(new_aid_list, new_flag_list)
#    daids = ibs.get_valid_aids(is_exemplar=True, nojunk=True)
#    cfgdict = dict(can_match_samename=False)
#    #, use_k_padding=True)
#    qreq_ = ibs.new_query_request(qaids, daids, cfgdict)
#    qres_list = ibs.query_chips(qreq_=qreq_)
#    return qres_list


def detect_join_cases(ibs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    Returns:
        QueryResult: qres_list -  object of feature correspondences and scores

    CommandLine:
        python -m ibeis.ibsfuncs --test-detect_join_cases --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> # execute function
        >>> qres_list = detect_join_cases(ibs)
        >>> # verify results
        >>> #result = str(qres_list)
        >>> #print(result)
        >>> ut.quit_if_noshow()
        >>> import guitool
        >>> from ibeis.gui import inspect_gui
        >>> guitool.ensure_qapp()
        >>> qaid2_qres = {qres.qaid: qres for qres in qres_list}
        >>> qres_wgt = inspect_gui.QueryResultsWidget(ibs, qaid2_qres, filter_reviewed=False)
        >>> qres_wgt.show()
        >>> qres_wgt.raise_()
        >>> guitool.qtapp_loop(qres_wgt)
    """
    qaids = ibs.get_valid_aids(is_exemplar=None, nojunk=True)
    daids = ibs.get_valid_aids(is_exemplar=None, nojunk=True)
    cfgdict = dict(can_match_samename=False, use_k_padding=True)
    qreq_ = ibs.new_query_request(qaids, daids, cfgdict)
    qres_list = ibs.query_chips(qreq_=qreq_)
    return qres_list
    #return qres_list


def _split_car_contrib_tag(contrib_tag, distinguish_invalids=True):
        if contrib_tag is not None and 'NNP GZC Car' in contrib_tag:
            contrib_tag_split = contrib_tag.strip().split(',')
            if len(contrib_tag_split) == 2:
                contrib_tag = contrib_tag_split[0].strip()
        elif distinguish_invalids:
            contrib_tag = None
        return contrib_tag


@__injectable
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
        ut.unixtime_to_datetime(unixtime)
        if unixtime is not None else
        'UNKNOWN'
        for unixtime in unixtime_list
    ]
    datetime_split_list = [ datetime.split(' ') for datetime in datetime_list ]
    date_list      = [ datetime_split[0] if len(datetime_split) == 2 else 'UNKNOWN' for datetime_split in datetime_split_list ]
    time_list      = [ datetime_split[1] if len(datetime_split) == 2 else 'UNKNOWN' for datetime_split in datetime_split_list ]
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
            ut.unixtime_to_datetime(unixtime)
            if unixtime is not None else
            'UNKNOWN'
            for unixtime in unixtime_list
        ]
        datetime_split_list = [ datetime.split(' ') for datetime in datetime_list ]
        date_list      = [ datetime_split[0] if len(datetime_split) == 2 else 'UNKNOWN' for datetime_split in datetime_split_list ]
        time_list      = [ datetime_split[1] if len(datetime_split) == 2 else 'UNKNOWN' for datetime_split in datetime_split_list ]
        lat_list       = ibs.get_image_lat(missing_gid_list)
        lon_list       = ibs.get_image_lon(missing_gid_list)
        seen_list      = filler
        marked_list    = filler

        header_list, line_list = construct()  # NOTE: discard the header list returned here
        return_list.extend(line_list)

    return return_list


@__injectable
def report_sightings_str(ibs, **kwargs):
    line_list = ibs.report_sightings(**kwargs)
    return '\n'.join(line_list)


@__injectable
def check_chip_existence(ibs, aid_list=None):
    aid_list = ibs.get_valid_aids()
    cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=False)
    chip_fpath_list = ibs.get_chip_fpath(cid_list)
    flag_list = [
        True if chip_fpath is None else exists(chip_fpath)
        for chip_fpath in chip_fpath_list
    ]
    cid_kill_list = ut.filterfalse_items(cid_list, flag_list)
    if len(cid_kill_list) > 0:
        print('found %d inconsistent chips attempting to fix' % len(cid_kill_list))
    ibs.delete_chips(cid_kill_list)


@__injectable
def is_special_encounter(ibs, eid_list):
    enctext_list = ibs.get_encounter_text(eid_list)
    isspecial_list = [str(enctext) in set(const.SPECIAL_ENCOUNTER_LABELS)
                      for enctext in enctext_list]
    return isspecial_list


@__injectable
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
        python -m ibeis.ibsfuncs --exec-get_quality_filterflags

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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


@__injectable
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
        python -m ibeis.ibsfuncs --exec-get_viewpoint_filterflags
        python -m ibeis.ibsfuncs --exec-get_viewpoint_filterflags --db NNP_Master3

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:20]
        >>> valid_yaws = None
        >>> unknown_ok = False
        >>> yaw_flags = list(get_viewpoint_filterflags(ibs, aid_list, valid_yaws, unknown_ok))
        >>> result = ('yaw_flags = %s' % (str(yaw_flags),))
        >>> print(result)
    """
    assert isinstance(valid_yaws, (set, list, tuple)), 'valid_yaws is not a container'
    yaw_list = ibs.get_annot_yaw_texts(aid_list)
    if unknown_ok:
        yaw_flags  = (yaw is None or yaw in valid_yaws for yaw in yaw_list)
    else:
        yaw_flags  = (yaw is not None and yaw in valid_yaws for yaw in yaw_list)
    yaw_flags = list(yaw_flags)
    return yaw_flags


@__injectable
def get_quality_viewpoint_filterflags(ibs, aid_list, minqual, valid_yaws):
    qual_flags = get_quality_filterflags(aid_list, minqual)
    yaw_flags = get_viewpoint_filterflags(aid_list, valid_yaws)
    #qual_list = ibs.get_annot_qualities(aid_list)
    #yaw_list = ibs.get_annot_yaw_texts(aid_list)
    #qual_flags = (qual is None or qual > minqual for qual in qual_list)
    #yaw_flags  = (yaw is None or yaw in valid_yaws for yaw in yaw_list)
    flags_list = list(ut.and_iters(qual_flags, yaw_flags))
    return flags_list


@__injectable
def get_annot_custom_filterflags(ibs, aid_list):
    if not ibs.cfg.other_cfg.enable_custom_filter:
        return [True] * len(aid_list)
    #minqual = const.QUALITY_TEXT_TO_INT['poor']
    minqual = 'ok'
    #valid_yaws = {'left', 'frontleft', 'backleft'}
    valid_yawtexts = {'left', 'frontleft'}
    flags_list = ibs.get_quality_viewpoint_filterflags(aid_list, minqual, valid_yawtexts)
    return flags_list


@__injectable
def filter_aids_custom(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: aid_list_

    CommandLine:
        python -m ibeis.ibsfuncs --test-filter_aids_custom

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb2')
        >>> aid_list = ibs.get_valid_aids()
        >>> # execute function
        >>> aid_list_ = filter_aids_custom(ibs, aid_list)
        >>> # verify results
        >>> result = str(aid_list_)
        >>> print(result)
    """
    if not ibs.cfg.other_cfg.enable_custom_filter:
        return aid_list
    flags_list = ibs.get_annot_custom_filterflags(aid_list)
    aid_list_ = list(ut.ifilter_items(aid_list, flags_list))
    #aid_list_ = list(ut.list_compress(aid_list, flags_list))
    return aid_list_


@__injectable
def flag_aids_count(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        pre_unixtime_sort (bool):

    Returns:
        ?:

    CommandLine:
        python -m ibeis.ibsfuncs --test-flag_aids_count

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
    flag_list      = ibs.get_annot_custom_filterflags(aid_list)
    isunknown_list = ibs.is_aid_unknown(aid_list)
    flag_list      = [ not unknown and flag for unknown, flag in zip(isunknown_list, flag_list) ]
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


@__injectable
def filter_aids_count(ibs, aid_list=None, pre_unixtime_sort=True):
    if aid_list is None:
        # Get all aids and pre-sort by unixtime
        aid_list = ibs.get_valid_aids()
        if pre_unixtime_sort:
            unixtime_list = ibs.get_image_unixtime(ibs.get_annot_gids(aid_list))
            aid_list      = ut.sortedby(aid_list, unixtime_list)
    flags_list = ibs.flag_aids_count(aid_list)
    aid_list_  = list(ut.ifilter_items(aid_list, flags_list))
    return aid_list_


@__injectable
def filterflags_unflat_aids_custom(ibs, aids_list):
    def some(flags):
        """ like any, but some at least one must be True """
        return len(flags) != 0 and any(flags)
    filtered_aids_list = ibs.unflat_map(ibs.get_annot_custom_filterflags, aids_list)
    isvalid_list = list(map(some, filtered_aids_list))
    return isvalid_list


@__injectable
def filter_nids_custom(ibs, nid_list):
    aids_list = ibs.get_name_aids(nid_list)
    isvalid_list = ibs.filterflags_unflat_aids_custom(aids_list)
    filtered_nid_list = ut.filter_items(nid_list, isvalid_list)
    return filtered_nid_list


@__injectable
def filter_gids_custom(ibs, gid_list):
    aids_list = ibs.get_image_aids(gid_list)
    isvalid_list = ibs.filterflags_unflat_aids_custom(aids_list)
    filtered_gid_list = ut.filter_items(gid_list, isvalid_list)
    return filtered_gid_list


@__injectable
def get_name_gps_tracks(ibs, nid_list=None, aid_list=None):
    """
    CommandLine:
        python -m ibeis.ibsfuncs --test-get_name_gps_tracks

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> #ibs = ibeis.opendb('PZ_Master0')
        >>> ibs = ibeis.opendb('testdb1')
        >>> #nid_list = ibs.get_valid_nids()
        >>> aid_list = ibs.get_valid_aids()
        >>> nid_list, gps_track_list, aid_track_list = ibs.get_name_gps_tracks(aid_list=aid_list)
        >>> nonempty_list = list(map(lambda x: len(x) > 0, gps_track_list))
        >>> ut.list_compress(nid_list, nonempty_list)
        >>> ut.list_compress(gps_track_list, nonempty_list)
        >>> ut.list_compress(aid_track_list, nonempty_list)
        >>> result = str(aid_track_list)
        >>> print(result)
        [[11], [], [4], [1], [2, 3], [5, 6], [7], [8], [10], [12], [13]]
    """
    assert aid_list is None or nid_list is None, 'only specify one please'
    if aid_list is None:
        aids_list_ = ibs.get_name_aids(nid_list)
    else:
        aids_list_, nid_list = ibs.group_annots_by_name(aid_list)
    aids_list = [ut.sortedby(aids, ibs.get_annot_image_unixtimes(aids)) for aids in aids_list_]
    gids_list = ibs.unflat_map(ibs.get_annot_gids, aids_list)
    gpss_list = ibs.unflat_map(ibs.get_image_gps, gids_list)

    isvalids_list = [[gps[0] != -1.0 or gps[1] != -1.0 for gps in gpss] for gpss in gpss_list]
    gps_track_list = [ut.list_compress(gpss, isvalids) for gpss, isvalids in zip(gpss_list, isvalids_list)]
    aid_track_list  = [ut.list_compress(aids, isvalids) for aids, isvalids in zip(aids_list, isvalids_list)]
    return nid_list, gps_track_list, aid_track_list


@__injectable
def get_unflat_annots_kmdists_list(ibs, aids_list):
    #ibs.check_name_mapping_consistency(aids_list)
    latlons_list = ibs.unflat_map(ibs.get_annot_image_gps, aids_list)
    latlon_arrs   = [np.array(latlons) for latlons in latlons_list]
    km_dists_list   = [ut.safe_pdist(latlon_arr, metric=ut.haversine) for latlon_arr in latlon_arrs]
    return km_dists_list


@__injectable
def get_unflat_annots_hourdists_list(ibs, aids_list):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> ibs = testdata_ibs('NNP_Master3')
        >>> nid_list = get_valid_multiton_nids_custom(ibs)
        >>> aids_list_ = ibs.get_name_aids(nid_list)
        >>> aids_list = [ibs.filter_aids_custom(aids) for aids in aids_list_]

    """
    assert all(list(map(ut.isunique, aids_list)))
    unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes, aids_list)
    #assert all(list(map(ut.isunique, unixtimes_list)))
    unixtime_arrs = [np.array(unixtimes)[:, None] for unixtimes in unixtimes_list]
    hour_dists_list = [ut.safe_pdist(unixtime_arr, metric=ut.unixtime_hourdiff) for unixtime_arr in unixtime_arrs]
    return hour_dists_list


@__injectable
def get_unflat_annots_speeds_list(ibs, aids_list):
    km_dists_list   = ibs.get_unflat_annots_kmdists_list(aids_list)
    hour_dists_list = ibs.get_unflat_annots_hourdists_list(aids_list)
    speeds_list     = [ut.safe_div(km_dists, hours_dists) for km_dists, hours_dists in zip(km_dists_list, hour_dists_list)]
    return speeds_list


def testdata_ibs(defaultdb='testdb1'):
    import ibeis
    ibs = ibeis.opendb(defaultdb=defaultdb)
    return ibs


def get_valid_multiton_nids_custom(ibs):
    nid_list_ = ibs._get_all_known_nids()
    ismultiton_list = [len(ibs.filter_aids_custom(aids)) > 1 for aids in ibs.get_name_aids(nid_list_)]
    nid_list = ut.list_compress(nid_list_, ismultiton_list)
    return nid_list


@__injectable
def get_name_speeds(ibs, nid_list):
    r"""
    CommandLine:
        python -m ibeis.ibsfuncs --test-get_name_speeds

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> ibs = testdata_ibs('NNP_Master3')
        >>> nid_list = get_valid_multiton_nids_custom(ibs)
        >>> speeds_list = get_name_speeds(ibs, nid_list)
        >>> result = str(speeds_list)
        >>> print(result)
    """
    aids_list_ = ibs.get_name_aids(nid_list)
    #ibs.check_name_mapping_consistency(aids_list_)
    aids_list = [ibs.filter_aids_custom(aids) for aids in aids_list_]
    speeds_list = ibs.get_unflat_annots_speeds_list(aids_list)
    return speeds_list


@__injectable
@accessor_decors.getter
def get_name_hourdiffs(ibs, nid_list):
    """
    CommandLine:
        python -m ibeis.ibsfuncs --test-get_name_hourdiffs

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> ibs = testdata_ibs('NNP_Master3')
        >>> nid_list = ibs.filter_nids_custom(ibs._get_all_known_nids())
        >>> hourdiffs_list = ibs.get_name_hourdiffs(nid_list)
        >>> result = hourdiffs_list
        >>> print(hourdiffs_list)
    """
    aids_list_ = ibs.get_name_aids(nid_list)
    #ibs.check_name_mapping_consistency(aids_list_)
    aids_list = [ibs.filter_aids_custom(aids) for aids in aids_list_]
    hourdiffs_list = ibs.get_unflat_annots_hourdists_list(aids_list)
    return hourdiffs_list


@__injectable
@accessor_decors.getter
def get_name_max_hourdiff(ibs, nid_list):
    hourdiffs_list = ibs.get_name_hourdiffs(nid_list)
    maxhourdiff_list = np.array(list(map(ut.safe_max, hourdiffs_list)))
    return maxhourdiff_list


@__injectable
@accessor_decors.getter
def get_name_max_speed(ibs, nid_list):
    """
    CommandLine:
        python -m ibeis.ibsfuncs --test-get_name_max_speed

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> ibs = testdata_ibs('NNP_Master3')
        >>> nid_list = ibs.filter_nids_custom(ibs._get_all_known_nids())
        >>> maxspeed_list = ibs.get_name_max_speed(nid_list)
        >>> result = maxspeed_list
        >>> print(maxspeed_list)
    """
    speeds_list = ibs.get_name_speeds(nid_list)
    maxspeed_list = np.array(list(map(ut.safe_max, speeds_list)))
    return maxspeed_list


def get_fastest_names(ibs, nid_list=None):
    r"""
    CommandLine:
        python -m ibeis.ibsfuncs --test-get_fastest_names

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> ibs = testdata_ibs('NNP_Master3')
        >>> nid_list = None
        >>> # execute function
        >>> nid_list_, maxspeed_list_ = get_fastest_names(ibs, nid_list)
        >>> # verify results
        >>> result = str(list(zip(nid_list_, maxspeed_list_))[0:10])
        >>> print(result)
    """
    if nid_list is None:
        nid_list = ibs._get_all_known_nids()
    maxspeed_list = ibs.get_name_max_speed(nid_list)
    # filter nans
    notnan_flags = ~np.isnan(maxspeed_list)
    maxspeed_list__ = maxspeed_list.compress(notnan_flags)
    nid_list__      = np.array(nid_list).compress(notnan_flags)
    # sort by speed
    sortx = maxspeed_list__.argsort()[::-1]
    nid_list_      = nid_list__.take(sortx)
    maxspeed_list_ = maxspeed_list__.take(sortx)
    return nid_list_, maxspeed_list_


def find_location_disparate_splits(ibs):
    """
    CommandLine:
        python -m ibeis.ibsfuncs --test-find_location_disparate_splits

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('NNP_Master3')
        >>> # execute function
        >>> offending_nids = find_location_disparate_splits(ibs)
        >>> # verify results
        >>> print('offending_nids = %r' % (offending_nids,))

    """
    import scipy.spatial.distance as spdist
    import functools
    #aid_list_count = ibs.get_valid_aids()
    aid_list_count = ibs.filter_aids_count()
    nid_list, gps_track_list, aid_track_list = ibs.get_name_gps_tracks(aid_list=aid_list_count)

    # Filter to only multitons
    has_multiple_list = [len(gps_track) > 1 for gps_track in gps_track_list]
    gps_track_list_ = ut.list_compress(gps_track_list, has_multiple_list)
    aid_track_list_ = ut.list_compress(aid_track_list, has_multiple_list)
    nid_list_ = ut.list_compress(nid_list, has_multiple_list)

    # Other properties
    unixtime_track_list_ = ibs.unflat_map(ibs.get_annot_image_unixtimes, aid_track_list_)

    # Move into arrays
    gpsarr_track_list_ = list(map(np.array, gps_track_list_))
    unixtimearr_track_list_ = [np.array(unixtimes)[:, None] for unixtimes in unixtime_track_list_]

    def unixtime_hourdiff(x, y):
        return np.abs(np.subtract(x, y)) / (60 ** 2)

    haversin_pdist = functools.partial(spdist.pdist, metric=ut.haversine)
    unixtime_pdist = functools.partial(spdist.pdist, metric=unixtime_hourdiff)
    # Get distances
    gpsdist_vector_list = list(map(haversin_pdist, gpsarr_track_list_))
    hourdist_vector_list = list(map(unixtime_pdist, unixtimearr_track_list_))

    # Get the speed in kilometers per hour for each animal
    speed_vector_list = [gpsdist / hourdist for gpsdist, hourdist in
                         zip(gpsdist_vector_list, hourdist_vector_list)]

    #maxhourdist_list = np.array([hourdist_vector.max() for hourdist_vector in hourdist_vector_list])
    maxgpsdist_list  = np.array([gpsdist_vector.max() for gpsdist_vector in gpsdist_vector_list])
    maxspeed_list = np.array([speed_vector.max() for speed_vector in speed_vector_list])
    sortx  = maxspeed_list.argsort()
    sorted_maxspeed_list = maxspeed_list[sortx]
    #sorted_nid_list = np.array(ut.list_take(nid_list_, sortx))

    if False:
        import plottool as pt
        pt.plot(sorted_maxspeed_list)
        allgpsdist_list = np.array(ut.flatten(gpsdist_vector_list))
        alltimedist_list = np.array(ut.flatten(hourdist_vector_list))

        pt.figure(fnum1=1, doclf=True, docla=True)
        alltime_sortx = alltimedist_list.argsort()
        pt.plot(allgpsdist_list[alltime_sortx])
        pt.plot(alltimedist_list[alltime_sortx])
        pt.iup()

        pt.figure(fnum1=2, doclf=True, docla=True)
        allgps_sortx = allgpsdist_list.argsort()
        pt.plot(allgpsdist_list[allgps_sortx])
        pt.plot(alltimedist_list[allgps_sortx])
        pt.iup()

        #maxgps_sortx = maxgpsdist_list.argsort()
        #pt.plot(maxgpsdist_list[maxgps_sortx])
        pt.iup()

    maxgps_sortx = maxgpsdist_list.argsort()
    gpsdist_thresh = 15
    sorted_maxgps_list = maxgpsdist_list[maxgps_sortx]
    offending_sortx = maxgps_sortx.compress(sorted_maxgps_list > gpsdist_thresh)

    speed_thresh_kph = 6  # kilometers per hour
    offending_sortx = sortx.compress(sorted_maxspeed_list > speed_thresh_kph)
    #sorted_isoffending = sorted_maxspeed_list > speed_thresh_kph
    #offending_nids = sorted_nid_list.compress(sorted_isoffending)
    offending_nids = ut.list_take(nid_list_, offending_sortx)
    #offending_speeds = ut.list_take(maxspeed_list, offending_sortx)
    print('offending_nids = %r' % (offending_nids,))

    for index in offending_sortx:
        print('\n\n--- Offender index=%d ---' % (index,))
        # Inspect a specific index
        aids = aid_track_list_[index]
        nid = nid_list_[index]
        assert np.all(np.array(ibs.get_annot_name_rowids(aids)) == nid)

        aid1_list, aid2_list = zip(*list(ut.product(aids, aids)))
        annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
        annotmatch_truth_list = ibs.get_annotmatch_truth(annotmatch_rowid_list)
        annotmatch_truth_list = ut.replace_nones(annotmatch_truth_list, -1)
        truth_mat = np.array(annotmatch_truth_list).reshape((len(aids), len(aids)))

        contrib_rowids = ibs.get_image_contributor_rowid(ibs.get_annot_gids(aids))
        contrib_tags = ibs.get_contributor_tag(contrib_rowids)

        print('nid = %r' % (nid,))
        print('maxspeed = %.2f km/h' % (maxspeed_list[index],))
        print('aids = %r' % (aids,))
        print('gpss = %s' % (ut.list_str(gps_track_list_[index]),))
        print('contribs = %s' % (ut.list_str(contrib_tags),))

        print('speedist_mat = \n' + ut.numpy_str(spdist.squareform(speed_vector_list[index]), precision=2))
        truth_mat_str = ut.numpy_str(truth_mat, precision=2)
        truth_mat_str = truth_mat_str.replace('-1' , ' _')

        print('truth_mat = \n' + truth_mat_str)
        print('gpsdist_mat  = \n' + ut.numpy_str(spdist.squareform(gpsdist_vector_list[index]), precision=2))
        print('hourdist_mat = \n' + ut.numpy_str(spdist.squareform(hourdist_vector_list[index]), precision=2))

    return offending_nids

    #gpsdist_matrix_list = list(map(spdist.squareform, gpsdist_vector_list))


@__injectable
def find_offending_contributors(ibs):
    lat_min, lon_min = (-1.340726, 36.792234)
    lat_max, lon_max = (-1.341633, 36.793340)
    gid_list = ibs.get_valid_gids()
    gps_list = ibs.get_image_gps(gid_list)

    gid_list_filtered = [
        gid
        for gid, (lat, lon) in zip(gid_list, gps_list)
        if lat_min >= lat and lat >= lat_max  and lon_min <= lon and lon <= lon_max
    ]
    contrib_list_filtered = ibs.get_image_contributor_tag(gid_list_filtered)

    contribs = {}
    for gid, contrib in zip(gid_list_filtered, contrib_list_filtered):
        if contrib not in contribs:
            contribs[contrib] = []
        contribs[contrib].append(gid)

    lengths_list = list(zip(contribs.keys(), [ len(contribs[k]) for k in contribs.keys() ]))
    print(lengths_list)


def export_nnp_master3_subset(ibs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-export_nnp_master3_subset

    Example:
        >>> # SCRIPT
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('NNP_Master3')
        >>> # execute function
        >>> result = export_nnp_master3_subset(ibs)
        >>> # verify results
        >>> print(result)
    """
    # Get the subset of the dataset marked as difficult
    annotmatch_rowid_list = ibs._get_all_annotmatch_rowids()
    ishard_list         = ibs.get_annotmatch_is_hard(annotmatch_rowid_list)
    isphotobomb_list    = ibs.get_annotmatch_is_photobomb(annotmatch_rowid_list)
    isscenerymatch_list = ibs.get_annotmatch_is_scenerymatch(annotmatch_rowid_list)
    isnondistinct_list  = ibs.get_annotmatch_is_nondistinct(annotmatch_rowid_list)
    hards        = np.array(ut.replace_nones(ishard_list, False))
    photobombs   = np.array(ut.replace_nones(isphotobomb_list, False))
    scenerys     = np.array(ut.replace_nones(isscenerymatch_list, False))
    nondistincts = np.array(ut.replace_nones(isnondistinct_list, False))
    flags = vt.and_lists(vt.or_lists(hards, nondistincts), ~photobombs, ~scenerys)
    annotmatch_rowid_list_ = ut.list_compress(annotmatch_rowid_list, flags)

    aid1_list = ibs.get_annotmatch_aid1(annotmatch_rowid_list_)
    aid2_list = ibs.get_annotmatch_aid2(annotmatch_rowid_list_)
    aid_list = sorted(list(set(aid1_list + aid2_list)))
    from ibeis import dbio
    gid_list = sorted(list(set(ibs.get_annot_gids(aid_list))))
    dbio.export_subset.export_images(ibs, gid_list, new_dbpath=join(ibs.get_workdir(), 'testdb3'))


@__injectable
def mark_annot_pair_as_positive_match(ibs, aid1, aid2, dryrun=False, on_nontrivial_merge=None):
    """
    TODO: ELEVATE THIS FUNCTION
    Change into make_task_mark_annot_pair_as_positive_match and it returns what
    needs to be done.

    Need to test several cases:
        uknown, unknown
        knownA, knownA
        knownB, knownA
        unknown, knownA
        knownA, unknown

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1 (int):  query annotation id
        aid2 (int):  matching annotation id

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-mark_annot_pair_as_positive_match

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid1, aid2 = ibs.get_valid_aids()[0:2]
        >>> dryrun = True
        >>> status = mark_annot_pair_as_positive_match(ibs, aid1, aid2, dryrun)
        >>> # verify results
        >>> print(status)
    """
    def _set_annot_name_rowids(aid_list, nid_list):
        if not ut.QUIET:
            print('... _set_annot_name_rowids(aids=%r, nids=%r)' % (aid_list, nid_list))
            print('... names = %r' % (ibs.get_name_texts(nid_list)))
        assert len(aid_list) == len(nid_list), 'list must correspond'
        if not dryrun:
            ibs.set_annot_name_rowids(aid_list, nid_list)
            ibs.mark_annot_pair_as_reviewed(aid1, aid2)
            #ibs.add_or_update_annotmatch(aid1, aid2, const.TRUTH_MATCH, [1.0])
        # Return the new annots in this name
        _aids_list = ibs.get_name_aids(nid_list)
        _combo_aids_list = [_aids + [aid] for _aids, aid, in zip(_aids_list, aid_list)]
        status = _combo_aids_list
        return status
    print('[marking_match] aid1 = %r, aid2 = %r' % (aid1, aid2))

    nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2])
    if nid1 == nid2:
        print('...images already matched')
        status = None
        ibs.mark_annot_pair_as_reviewed(aid1, aid2)
    else:
        isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
        if isunknown1 and isunknown2:
            print('...match unknown1 to unknown2 into 1 new name')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1, aid2], next_nids * 2)
        elif not isunknown1 and not isunknown2:
            print('...merge known1 into known2')
            aid1_and_groundtruth = ibs.get_annot_groundtruth(aid1, noself=False)
            aid2_and_groundtruth = ibs.get_annot_groundtruth(aid2, noself=False)
            trivial_merge = len(aid1_and_groundtruth) == 1 and len(aid2_and_groundtruth) == 1
            if not trivial_merge:
                if on_nontrivial_merge is None:
                    raise Exception('no function is set up to handle nontrivial merges!')
                else:
                    on_nontrivial_merge(ibs, aid1, aid2)
            status =  _set_annot_name_rowids(aid1_and_groundtruth, [nid2] * len(aid1_and_groundtruth))
        elif isunknown2 and not isunknown1:
            print('...match unknown2 into known1')
            status =  _set_annot_name_rowids([aid2], [nid1])
        elif isunknown1 and not isunknown2:
            print('...match unknown1 into known2')
            status =  _set_annot_name_rowids([aid1], [nid2])
        else:
            raise AssertionError('impossible state')
    return status


@__injectable
def mark_annot_pair_as_negative_match(ibs, aid1, aid2, dryrun=False, on_nontrivial_split=None):
    """
    TODO: ELEVATE THIS FUNCTION

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1 (int):  annotation id
        aid2 (int):  annotation id
        dryrun (bool):

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-mark_annot_pair_as_negative_match

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid1, aid2 = ibs.get_valid_aids()[0:2]
        >>> dryrun = True
        >>> # execute function
        >>> result = mark_annot_pair_as_negative_match(ibs, aid1, aid2, dryrun)
        >>> # verify results
        >>> print(result)
    """
    def _set_annot_name_rowids(aid_list, nid_list):
        print('... _set_annot_name_rowids(%r, %r)' % (aid_list, nid_list))
        if not dryrun:
            ibs.set_annot_name_rowids(aid_list, nid_list)
            ibs.mark_annot_pair_as_reviewed(aid1, aid2)
            #ibs.add_or_update_annotmatch(aid1, aid2, const.TRUTH_NOT_MATCH, [1.0])
    nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2])
    if nid1 == nid2:
        print('images are marked as having the same name... we must tread carefully')
        aid1_groundtruth = ibs.get_annot_groundtruth(aid1, noself=True)
        if len(aid1_groundtruth) == 1 and aid1_groundtruth == [aid2]:
            # this is the only safe case for same name split
            # Change so the names are not the same
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1], next_nids)
        else:
            if on_nontrivial_split is None:
                raise Exception('no function is set up to handle nontrivial splits!')
            else:
                on_nontrivial_split(ibs, aid1, aid2)
    else:
        isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
        if isunknown1 and isunknown2:
            print('...nonmatch unknown1 and unknown2 into 2 new names')
            next_nids = ibs.make_next_nids(num=2)
            status =  _set_annot_name_rowids([aid1, aid2], next_nids)
        elif not isunknown1 and not isunknown2:
            print('...nonmatch known1 and known2... nothing to do (yet)')
            ibs.mark_annot_pair_as_reviewed(aid1, aid2)
            status = None
        elif isunknown2 and not isunknown1:
            print('...nonmatch unknown2 -> newname and known1')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid2], next_nids)
        elif isunknown1 and not isunknown2:
            print('...nonmatch unknown1 -> newname and known2')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1], next_nids)
        else:
            raise AssertionError('impossible state')
    return status


@__injectable
def make_next_encounter_text(ibs):
    """
    Creates what the next encounter name would be but does not add it to the database

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-make_next_encounter_text

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> new_enctext = make_next_encounter_text(ibs)
        >>> result = new_enctext
        >>> print(result)
        New Encounter 0
    """
    eid_list = ibs.get_valid_eids()
    old_enctext_list = ibs.get_encounter_text(eid_list)
    new_enctext = ut.get_nonconflicting_string('New Encounter %d', old_enctext_list)
    return new_enctext


@__injectable
def add_next_encounter(ibs):
    """
    Adds a new encounter to the database
    """
    new_enctext = ibs.make_next_encounter_text()
    (new_eid,) = ibs.add_encounters([new_enctext])
    return new_eid


@__injectable
def create_new_encounter_from_images(ibs, gid_list, new_eid=None):
    r"""
    Args:
        gid_list (list):

    CommandLine:
        python -m ibeis.ibsfuncs --test-create_new_encounter_from_images

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[::2]
        >>> # execute function
        >>> new_eid = create_new_encounter_from_images(ibs, gid_list)
        >>> # verify results
        >>> result = new_eid
        >>> print(result)
    """
    if new_eid is None:
        new_eid = ibs.add_next_encounter()
    eid_list = [new_eid] * len(gid_list)
    ibs.set_image_eids(gid_list, eid_list)
    return new_eid


@__injectable
def new_encounters_from_images(ibs, gids_list):
    r"""
    Args:
        gids_list (list):
    """
    eid_list = [ibs.create_new_encounter_from_images(gids) for gids in gids_list]
    return eid_list


@__injectable
def create_new_encounter_from_names(ibs, nid_list):
    r"""
    Args:
        nid_list (list):

    CommandLine:
        python -m ibeis.ibsfuncs --test-create_new_encounter_from_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> nid_list = ibs._get_all_known_nids()[0:2]
        >>> # execute function
        >>> new_eid = ibs.create_new_encounter_from_names(nid_list)
        >>> # clean up
        >>> ibs.delete_encounters(new_eid)
        >>> # verify results
        >>> result = new_eid
        >>> print(result)
    """
    aids_list = ibs.get_name_aids(nid_list)
    gids_list = ibs.unflat_map(ibs.get_annot_gids, aids_list)
    gid_list = ut.flatten(gids_list)
    new_eid = ibs.create_new_encounter_from_images(gid_list)
    return new_eid


@__injectable
def prepare_annotgroup_review(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        tuple: (src_ag_rowid, dst_ag_rowid) - source and dest annot groups

    CommandLine:
        python -m ibeis.ibsfuncs --test-prepare_annotgroup_review

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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


def remove_rfdetect(ibs):
    aids = ibs.search_annot_notes('rfdetect')
    notes = ibs.get_annot_notes(aids)
    newnotes = [note.replace('rfdetect', '') for note in notes]
    ibs.set_annot_notes(aids, newnotes)


@__injectable
def search_annot_notes(ibs, pattern, aid_list=None):
    """

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
    valid_index_list, valid_match_list = search_list(notes_list, pattern, flags=re.IGNORECASE)
    #[match.group() for match in valid_match_list]
    valid_aid_list = ut.list_take(aid_list, valid_index_list)
    return valid_aid_list


def search_list(text_list, pattern, flags=0):
    """
    CommandLine:
        python -m ibeis.ibsfuncs --test-search_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> text_list = ['ham', 'jam', 'eggs', 'spam']
        >>> pattern = '.am'
        >>> flags = 0
        >>> (valid_index_list, valid_match_list) = search_list(text_list, pattern, flags)
        >>> result = str(valid_index_list)
        >>> print(result)
        [0, 1, 3]
    """
    match_list = [re.search(pattern, text, flags=flags) for text in text_list]
    valid_index_list = [index for index, match in enumerate(match_list) if match is not None]
    valid_match_list = ut.list_take(match_list, valid_index_list)
    return valid_index_list, valid_match_list


@__injectable
def get_annot_pair_timdelta(ibs, aid_list1, aid_list2):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list1 (int):  list of annotation ids
        aid_list2 (int):  list of annotation ids

    Returns:
        list: timedelta_list

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_annot_pair_timdelta

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> from six.moves import filter
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids(hasgt=True)
        >>> unixtimes = ibs.get_annot_image_unixtimes(aid_list)
        >>> aid_list = ut.list_compress(aid_list, np.array(unixtimes) != -1)
        >>> gt_aids_list = ibs.get_annot_groundtruth(aid_list, daid_list=aid_list)
        >>> aid_list1 = [aid for aid, gt_aids in zip(aid_list, gt_aids_list) if len(gt_aids) > 0][0:5]
        >>> aid_list2 = [gt_aids[0] for gt_aids in gt_aids_list if len(gt_aids) > 0][0:5]
        >>> timedelta_list = ibs.get_annot_pair_timdelta(aid_list1, aid_list2)
        >>> result = ut.numpy_str(timedelta_list, precision=2)
        >>> print(result)
        np.array([  7.57e+07,   7.57e+07,   2.41e+06,   1.98e+08,   9.69e+07], dtype=np.float64)

    """
    unixtime_list1 = np.array(ibs.get_annot_image_unixtimes(aid_list1), dtype=np.float)
    unixtime_list2 = np.array(ibs.get_annot_image_unixtimes(aid_list2), dtype=np.float)
    unixtime_list1[unixtime_list1 == -1] = np.nan
    unixtime_list2[unixtime_list2 == -1] = np.nan
    timedelta_list = np.abs(unixtime_list1 - unixtime_list2)
    return timedelta_list


@__injectable
def filter_aids_to_quality(ibs, aid_list, minqual, unknown_ok=True):
    qual_flags = list(ibs.get_quality_filterflags(aid_list, minqual, unknown_ok=unknown_ok))
    aid_list_ = ut.list_compress(aid_list, qual_flags)
    return aid_list_


@__injectable
def filter_aids_to_viewpoint(ibs, aid_list, valid_yaws, unknown_ok=True):
    """
    Removes aids that do not have a valid yaw

    TODO; rename to valid_viewpoint because this func uses category labels
    """
    yaw_flags = list(ibs.get_viewpoint_filterflags(aid_list, valid_yaws, unknown_ok=unknown_ok))
    aid_list_ = ut.list_compress(aid_list, yaw_flags)
    return aid_list_


@__injectable
def filter_aids_without_name(ibs, aid_list):
    """
    Remove aids without names
    """
    flag_list = ut.not_list(ibs.is_aid_unknown(aid_list))
    aid_list_ = ut.list_compress(aid_list, flag_list)
    return aid_list_


@__injectable
def filter_aids_without_timestamps(ibs, aid_list):
    """
    Removes aids without timestamps
    aid_list = ibs.get_valid_aids()
    """
    unixtime_list = ibs.get_annot_image_unixtimes(aid_list)
    flag_list = [unixtime != -1 for unixtime in unixtime_list]
    aid_list_ = ut.list_compress(aid_list, flag_list)
    return aid_list_


@__injectable
def filter_aids_to_species(ibs, aid_list, species):
    """
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        species (?):

    Returns:
        list: aid_list_

    CommandLine:
        python -m ibeis.ibsfuncs --exec-filter_aids_to_species

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> species = ibeis.const.Species.ZEB_GREVY
        >>> aid_list_ = filter_aids_to_species(ibs, aid_list, species)
        >>> print(aid_list_)
        [9, 10]
    """
    flag_list = [species == species_text for species_text in ibs.get_annot_species(aid_list)]
    aid_list_ = ut.list_compress(aid_list, flag_list)
    return aid_list_


@__injectable
def partition_annots_into_singleton_multiton(ibs, aid_list):
    """
    aid_list = aid_list_
    """
    aids_list = ibs.group_annots_by_name(aid_list)[0]
    singletons = [aids for aids in aids_list if len(aids) == 1]
    multitons = [aids for aids in aids_list if len(aids) > 1]
    return singletons, multitons


@__injectable
def partition_annots_into_corresponding_groups(ibs, aid_list1, aid_list2):
    """
    Returns 4 lists of lists. In the first two each list is a list of aids
    grouped by names and the names correspond with each other. In the last two
    are the annots that did not correspond with anything in the other list.

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list1 (int):  list of annotation ids
        aid_list2 (int):  list of annotation ids

    CommandLine:
        python -m ibeis.ibsfuncs --exec-partition_annots_into_corresponding_groups

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aid_list1 = [1, 2, 8, 9, 60]
        >>> aid_list2 = [3, 7, 20]
        >>> result = partition_annots_into_corresponding_groups(ibs, aid_list1, aid_list2)
        >>> print(result)
        ([[1, 2], [8]], [[3], [7]], [9, 60], [20])

    """
    grouped_aids1 = ibs.group_annots_by_name(aid_list1)[0]
    grouped_aids1 = [aids.tolist() for aids in grouped_aids1]
    # Get the group of available aids that a reference aid could match
    gropued_aids2 = ibs.get_annot_groundtruth(ut.get_list_column(grouped_aids1, 0), daid_list=aid_list2)

    flag_list = [x > 0 for x in map(len, gropued_aids2)]

    gt_grouped_aids1 = ut.list_compress(grouped_aids1, flag_list)
    gt_grouped_aids2 = ut.list_compress(gropued_aids2, flag_list)

    gf_grouped_aids1 = ut.flatten(ut.list_compress(grouped_aids1, ut.not_list(flag_list)))
    gf_gropued_aids2 = ut.setdiff_ordered(aid_list2, ut.flatten(gt_grouped_aids2))
    return gt_grouped_aids1, gt_grouped_aids2, gf_grouped_aids1, gf_gropued_aids2


def make_temporally_distinct_blind_test(ibs, challenge_num=None):

    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=1
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=2
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=3
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=4
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=5
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=6
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=7

    Ignore:
        challenge_pdfs = ut.glob('.', 'pair_*_compressed.pdf', recursive=True)
        dname = 'localdata/all_challenges'
        ut.ensuredir(dname)
        ut.copy(challenge_pdfs, dname)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('Elephants_drop1')
        >>> result = make_temporally_distinct_blind_test(ibs)
        >>> print(result)
    """
    # Parameters
    if challenge_num is None:
        challenge_num = ut.get_argval('--challenge-num', type_=int, default=1)
    num_repeats = 2
    num_singles = 4

    # Filter out ones from previous challenges
    all_aid_list = ibs.get_valid_aids()

    def get_challenge_aids(aid_list):
        # Get annots with timestamps
        aid_list_ = filter_aids_without_timestamps(ibs, aid_list)
        singletons, multitons = partition_annots_into_singleton_multiton(ibs, aid_list_)
        # process multitons
        hourdists_list = ibs.get_unflat_annots_hourdists_list(multitons)
        best_pairx_list = [vt.pdist_argsort(x)[-1] for x in hourdists_list]
        best_multitons = vt.ziptake(multitons, best_pairx_list)
        best_hourdists_list = ut.flatten(ibs.get_unflat_annots_hourdists_list(best_multitons))
        assert len(best_hourdists_list) == len(best_multitons)
        best_multitons_sortx = np.array(best_hourdists_list).argsort()[::-1]
        best_pairs = ut.list_take(best_multitons, best_multitons_sortx[0:num_repeats])
        best_multis = ut.flatten(best_pairs)

        # process singletons
        best_singles = ut.list_take(singletons, ut.random_indexes(len(singletons), num_singles, seed=0))
        best_singles = ut.flatten(best_singles)

        chosen_aids_ = best_multis + best_singles
        assert len(best_multis) == num_repeats * 2, 'len(best_multis)=%r' % (len(best_multis),)
        assert len(best_singles) == num_singles, 'len(best_singles)=%r' % (len(best_singles),)
        return chosen_aids_
        #return best_multis, best_singles

    aid_list = all_aid_list
    # Define invalid aids recusrively, so challenges are dijsoint
    invalid_aids = []
    for _ in range(1, challenge_num + 1):
        prev_chosen_aids_ = get_challenge_aids(aid_list)
        invalid_aids += prev_chosen_aids_
        aid_list = ut.setdiff_ordered(all_aid_list, invalid_aids)

    chosen_aids_ = get_challenge_aids(aid_list)
    chosen_nids_ = ibs.get_annot_nids(chosen_aids_)

    # Randomize order of chosen aids and nids
    shufflx = ut.random_indexes(len(chosen_aids_), seed=432)
    chosen_aids = ut.list_take(chosen_aids_, shufflx)
    chosen_nids = ut.list_take(chosen_nids_, shufflx)
    choosex_list = np.arange(len(chosen_aids))

    # ----------------------------------
    # SAMPLE A SET OF PAIRS TO PRESNET
    import itertools
    #nid_pair_list   = list(itertools.combinations(chosen_nids, 2))

    def index2d_take(list_, indexes_list):
        return [[list_[x] for x in xs] for xs in indexes_list]

    choosex_pair_list = list(itertools.combinations(choosex_list, 2))
    nid_pair_list = index2d_take(chosen_nids, choosex_pair_list)
    # Do not choose any correct pairs
    is_correct = [nid1 == nid2 for nid1, nid2 in nid_pair_list]
    p = 1. - np.array(is_correct)
    p /= p.sum()
    random_pair_sample = 15
    # Randomly remove nonmatch pairs
    pairx_list = np.arange(len(choosex_pair_list))
    num_remove = len(pairx_list) - random_pair_sample
    randstate = np.random.RandomState(seed=10)
    remove_xs = randstate.choice(pairx_list, num_remove, replace=False, p=p)
    keep_mask = np.logical_not(vt.index_to_boolmask(remove_xs, len(pairx_list)))
    sample_choosex_list = np.array(choosex_pair_list)[keep_mask]

    # One last shuffle of pairs
    #ut.random_indexes(len(chosen_aids), seed=10)
    randstate = np.random.RandomState(seed=10)
    randstate.shuffle(sample_choosex_list)

    # Print out info
    avuuids_list = ibs.get_annot_visual_uuids(chosen_aids)
    print('avuuids_list=\n' + ut.list_str(avuuids_list))
    chosen_avuuids_list = index2d_take(avuuids_list, sample_choosex_list)
    for count, (avuuid1, avuuid2)  in enumerate(chosen_avuuids_list, start=1):
        print('pair %3d: %s - vs - %s' % (count, avuuid1, avuuid2))
    # Print
    best_times = ibs.get_unflat_annots_hourdists_list(ibs.group_annots_by_name(chosen_aids_)[0])
    print('best_times = %r' % (best_times,))

    # -------------------------
    # WRITE CHOSEN AIDS TO DISK
    import ibeis.viz
    #fname = 'aidchallenge_' + ibs.get_dbname() + '_' + str(challenge_num)
    challengename = 'pair_challenge'  '_' + str(challenge_num) + '_dbname=' + ibs.get_dbname()
    output_dirname = ut.truepath(join('localdata', challengename))
    ut.ensuredir(output_dirname)
    #aid = chosen_aids[0]
    #ibeis.viz.viz_chip.show_many_chips(ibs, chosen_aids)

    def dump_challenge_fig(ibs, aid, output_dirname):
        import plottool as pt
        pt.clf()
        title_suffix = str(ibs.get_annot_visual_uuids(aid))
        fig, ax = ibeis.viz.show_chip(ibs, aid, annote=False, show_aidstr=False, show_name=False, show_num_gt=False, fnum=1, title_suffix=title_suffix)
        #fig.show()
        #pt.iup()
        fpath = ut.truepath(join(output_dirname, 'avuuid%s.png' % (title_suffix.replace('-', ''),)))
        pt.save_figure(fpath_strict=fpath, fig=fig, fnum=1, verbose=False)
        vt.clipwhite_ondisk(fpath, fpath)
        return fpath

    orig_fpath_list = [dump_challenge_fig(ibs, aid, output_dirname) for aid in chosen_aids]
    #fpath_pair_list = index2d_take(orig_fpath_list, choosex_pair_list)
    #fpath_pair_list_ = ut.list_compress(fpath_pair_list, keep_mask)
    fpath_pair_list_ = index2d_take(orig_fpath_list, sample_choosex_list)
    ut.vd(output_dirname)

    # ----- DUMP TO LATEX ---

    # Randomize pair ordering
    #randstate = np.random.RandomState(seed=10)
    ##ut.random_indexes(len(chosen_aids))
    #randstate.shuffle(fpath_pair_list_)

    #latex_blocks = [ut.get_latex_figure_str(fpath_pair, width_str='2.4in', nCols=2, dpath=output_dirname) for fpath_pair in fpath_pair_list_]
    latex_blocks = [ut.get_latex_figure_str(fpath_pair, width_str='.5\\textwidth', nCols=2, dpath=output_dirname, caption_str='pair %d' % (count,))
                    for count, fpath_pair in enumerate(fpath_pair_list_, start=1)]
    latex_text = '\n\n'.join(latex_blocks)
    #print(latex_text)
    title = challengename.replace('_', ' ')
    pdf_fpath = ut.compile_latex_text(latex_text, dpath=output_dirname, verbose=True, fname=challengename, silence=True, title=title)
    output_pdf_fpath = ut.compress_pdf(pdf_fpath)
    ut.startfile(output_pdf_fpath)
    """
    import ibeis
    ibs = ibeis.opendb('Elephants_drop1')

    Guesses:
        def verify_guesses(partial_pairs):
            aids_lists = ibs.unflat_map(ibs.get_annot_rowids_from_partial_vuuids, partial_pairs)
            aid_pairs = list(map(ut.flatten, aids_lists))
            aids1 = ut.get_list_column(aid_pairs, 0)
            aids2 = ut.get_list_column(aid_pairs, 1)
            truth = ibs.get_aidpair_truths(aids1, aids2)
            return truth

        # Chuck: 2, 6   --
        partial_pairs = [('fc4fcc', '47622'), ('981ef476', '73721a95')]
        print(verify_guesses(partial_pairs))
        # Mine: 1, 14   --
        partial_pairs = [('fc4f', 'f0d32'), ('73721', '47622')]
        print(verify_guesses(partial_pairs))
        # Hendrik: 3, X --
        partial_pairs = [('476225', '4c383f'), ('48c1', '981ef')]
        # Hendrik2: 3, 13 --
        partial_pairs = [('476225', '4c383f'), ('fc4fcc', '24a50bb')]
        print(verify_guesses(partial_pairs))
        # Tanya:
        partial_pairs = [('476225', 'fc4fcc48'), ('73721a', '981ef476')]

        partial_vuuid_strs = ut.flatten(partial_pairs)

        get_annots_from_partial_vuuids

    """
    #else:
    #    def format_challenge_pairs():
    #        # -------------------------
    #        # EMBED INTO A PDF
    #        #from utool.util_latex import *  # NOQA
    #        import utool as ut
    #        import vtool as vt
    #        #verbose = True
    #        dpath = '/home/joncrall/code/ibeis/aidchallenge'
    #        # Read images
    #        orig_fpath_list = ut.list_images(dpath, fullpath=True)
    #        # Clip white out
    #        clipped_fpath_list = [vt.clipwhite_ondisk(fpath) for fpath in orig_fpath_list]
    #        # move into temporary figure dir with hashed names
    #        fpath_list = []
    #        figdpath = join(dpath, 'figures')
    #        ut.ensuredir(figdpath)
    #        from os.path import splitext, basename, dirname
    #        for fpath in clipped_fpath_list:
    #            fname, ext = splitext(basename(fpath))
    #            fname_ = ut.hashstr(fname, alphabet=ut.ALPHABET_16) + ext
    #            fpath_ = join(figdpath, fname_)
    #            ut.move(fpath, fpath_)
    #            fpath_list.append(fpath_)

    #            figure_str = ut.get_latex_figure_str(fpath_list, width_str='2.4in', nCols=2, dpath=dirname(figdpath))
    #            input_text = figure_str
    #            pdf_fpath = ut.compile_latex_text(input_text, dpath=dpath, verbose=False, quiet=True)  # NOQA

    #        #output_pdf_fpath = ut.compress_pdf(pdf_fpath)

    #        #fpath_list
    #        ## Weirdness
    #        #new_rel_fpath_list = [ut.relpath_unix(fpath_, dpath) for fpath_ in new_fpath_list]

    #    #for fpath_pair in fpath_pair_list:
    #    #    print(fpath_pair)

    #    # -------------------------
    #    # EMBED INTO A PDF
    #if False:
    #    # TODO: quadratic programmming optimization
    #    # Actually its a quadratically constrained quadratic integer program
    #    def get_assignment_mat(ibs, aid_list):
    #        #import scipy.spatial.distance as spdist
    #        #nid_list = ibs.get_annot_nids(aid_list)
    #        aid1_list, aid2_list = list(zip(*ut.iprod(aid_list, aid_list)))
    #        truths = ibs.get_aidpair_truths(aid1_list, aid2_list)
    #        assignment_mat = truths.reshape(len(aid_list), len(aid_list))
    #        assignment_mat[np.diag_indices(assignment_mat.shape[0])] = 0
    #        return assignment_mat
    #    import scipy.spatial.distance as spdist
    #    pairwise_times = ibs.get_unflat_annots_hourdists_list([aid_list])[0]
    #    ptime_utri = np.triu(spdist.squareform(pairwise_times))  # NOQA
    #    assignment_utri = np.triu(get_assignment_mat(ibs, aid_list))  # NOQA
    #    nassign_utri = np.triu((1 - assignment_utri) - np.eye(len(assignment_utri)))  # NOQA

    #    r"""
    #    Let d be number of annots
    #    Let x \in {0, 1}^d be an assignment vector

    #    let l = num matching pairs = 2
    #    let m = num nonmatching pairs = 2

    #    # maximize total time delta
    #    min 1/2 x^T (-ptime_utri) x
    #    s.t.
    #    # match constraint
    #     .5 * x.T @ A @ x - l <= 0
    #    -.5 * x.T @ A @ x + l <= 0
    #    # nonmatch constraint
    #     .5 * x.T @ \bar{A} @ x - l <= 0
    #    -.5 * x.T @ \bar{A} @ x + l <= 0
    #    """

    #maxhourdists = np.array(list(map(ut.safe_max, hourdists_list)))
    #top_multi_indexes = maxhourdists.argsort()[::-1][:num_repeats]
    #top_multitons = ut.list_take(multitons, top_multi_indexes)

    #
    #scipy.optimize.linprog


@__injectable
def dans_lists(ibs, positives=10, negatives=10, verbose=False):
    from random import shuffle

    aid_list = ibs.get_valid_aids()
    yaw_list = ibs.get_annot_yaw_texts(aid_list)
    qua_list = ibs.get_annot_quality_texts(aid_list)
    sex_list = ibs.get_annot_sex_texts(aid_list)
    age_list = ibs.get_annot_age_months_est(aid_list)

    positive_list = [
        aid
        for aid, yaw, qua, sex, (start, end) in zip(aid_list, yaw_list, qua_list, sex_list, age_list)
        if yaw.upper() == 'LEFT' and qua.upper() in ['OK', 'GOOD', 'EXCELLENT'] and sex.upper() in ['MALE', 'FEMALE'] and start != -1 and end != -1
    ]

    negative_list = [
        aid
        for aid, yaw, qua, sex, (start, end) in zip(aid_list, yaw_list, qua_list, sex_list, age_list)
        if yaw.upper() == 'LEFT' and qua.upper() in ['OK', 'GOOD', 'EXCELLENT'] and sex.upper() == 'UNKNOWN SEX' and start == -1 and end == -1
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


def _stat_str(dict_):
    import utool as ut
    dict_ = dict_.copy()
    if dict_.get('num_nan', None) == 0:
        del dict_['num_nan']
    exclude_keys = []  # ['std', 'nMin', 'nMax']
    str_ =  ut.get_stats_str(stat_dict=dict_, precision=2, exclude_keys=exclude_keys)
    str_ = str_.replace('\'', '')
    return str_


# Quality and Viewpoint Stats
@__injectable
def get_annot_qual_stats(ibs, aid_list):
    annot_qualtext_list = ibs.get_annot_quality_texts(aid_list)
    qualtext2_aids = ut.group_items(aid_list, annot_qualtext_list)
    qual_keys = list(const.QUALITY_TEXT_TO_INT.keys())
    assert set(qual_keys) >= set(qualtext2_aids), 'bad keys: ' + str(set(qualtext2_aids) - set(qual_keys))
    qualtext2_nAnnots = ut.odict([(key, len(qualtext2_aids.get(key, []))) for key in qual_keys])
    # Filter 0's
    qualtext2_nAnnots = {key: val for key, val in six.iteritems(qualtext2_nAnnots) if val != 0}
    return qualtext2_nAnnots


@__injectable
def get_annot_yaw_stats(ibs, aid_list):
    annot_yawtext_list = ibs.get_annot_yaw_texts(aid_list)
    yawtext2_aids = ut.group_items(aid_list, annot_yawtext_list)
    yaw_keys = list(const.VIEWTEXT_TO_YAW_RADIANS.keys()) + [None]
    assert set(yaw_keys) >= set(annot_yawtext_list), 'bad keys: ' + str(set(annot_yawtext_list) - set(yaw_keys))
    yawtext2_nAnnots = ut.odict([(key, len(yawtext2_aids.get(key, []))) for key in yaw_keys])
    # Filter 0's
    yawtext2_nAnnots = {key: val for key, val in six.iteritems(yawtext2_nAnnots) if val != 0}
    return yawtext2_nAnnots


# Indepdentent query / database stats
@__injectable
def get_annot_stats_dict(ibs, aids, prefix=''):
    """ stats for a set of annots """
    aid_per_name_stats = ut.get_stats(ibs.get_num_annots_per_name(aids)[0], use_nan=True)
    qual_stats = ibs.get_annot_qual_stats(aids)
    yaw_stats  = ibs.get_annot_yaw_stats(aids)
    aid_stats_dict = ut.odict([
        ('num_' + prefix + 'aids', len(aids)),
        (prefix + 'aid_per_name', _stat_str(aid_per_name_stats)),
        (prefix + 'aid_quals', _stat_str(qual_stats)),
        (prefix + 'aid_viewpoints', _stat_str(yaw_stats)),
    ])
    return aid_stats_dict


# Compares properties of query vs database annotations
def compare_correct_match_properties(ibs, grouped_qaids, grouped_groundtruth_list, getter_func, cmp_func):
    """
    getter_func = ibs.get_annot_yaws
    cmp_func = vt.ori_distance

    getter_func = ibs.get_annot_image_unixtimes_asfloat
    cmp_func = ut.unixtime_hourdiff
    """
    def replace_none_with_nan(x):
        import utool as ut
        return np.array(ut.replace_nones(x, np.nan))

    query_props = ibs.unflat_map(getter_func, grouped_qaids)
    query_props = ibs.unflat_map(replace_none_with_nan, query_props)

    match_props = ibs.unflat_map(getter_func, grouped_groundtruth_list)
    match_props = ibs.unflat_map(replace_none_with_nan, match_props)
    # Compare the query yaws to the yaws of its correct matches in the database
    gt_propdists_list = [cmp_func(np.array(qprops), np.array(gt_props)[:, None]) for qprops, gt_props in zip(query_props, match_props)]
    return gt_propdists_list


def viewpoint_diff(ori1, ori2):
    """ convert distance in radians to distance in viewpoint category """
    TAU = np.pi * 2
    ori_diff = vt.ori_distance(ori1, ori2)
    viewpoint_diff = len(const.VIEWTEXT_TO_YAW_RADIANS) * ori_diff / TAU
    return viewpoint_diff


@__injectable
def get_annot_per_name_stats(ibs, aid_list):
    """  stats about this set of aids """
    return ut.get_stats(ibs.get_num_annots_per_name(aid_list)[0], use_nan=True)


@__injectable
def get_annotconfig_stats(ibs, qaids, daids, verbose=True):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        qaids (list):  query annotation ids
        daids (list):  database annotation ids

    CommandLine:
        python -m ibeis.ibsfuncs --exec-get_annotconfig_stats --db PZ_MTEST
        python -m ibeis.ibsfuncs --exec-get_annotconfig_stats --db PZ_MTEST -a controlled
        python -m ibeis.ibsfuncs --exec-get_annotconfig_stats --db PZ_FlankHack -a default:qaids=allgt
        python -m ibeis.ibsfuncs --exec-get_annotconfig_stats --db PZ_MTEST -a controlled:per_name=2,min_gt=4

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, qaids, daids = main_helpers.testdata_ibeis()
        >>> get_annotconfig_stats(ibs, qaids, daids)
    """
    import numpy as np
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice.')

        # The aids that should be matched by a query
        grouped_qaids = ibs.group_annots_by_name(qaids)[0]
        grouped_groundtruth_list = ibs.get_annot_groundtruth(ut.get_list_column(grouped_qaids, 0), daid_list=daids)
        groundtruth_daids = ut.unique_unordered(ut.flatten(grouped_groundtruth_list))
        hasgt_list = ibs.get_annot_has_groundtruth(qaids, daid_list=daids)
        # The aids that should not match any query
        nonquery_daids = np.setdiff1d(np.setdiff1d(daids, qaids), groundtruth_daids)
        # The query aids that should not get any match
        unmatchable_queries = ut.list_compress(qaids, ut.not_list(hasgt_list))
        # The query aids that should not have a match
        matchable_queries = ut.list_compress(qaids, hasgt_list)

        # Intersection on a per name basis
        imposter_daid_per_name_stats = ibs.get_annot_per_name_stats(nonquery_daids)
        genuine_daid_per_name_stats  = ibs.get_annot_per_name_stats(groundtruth_daids)
        all_daid_per_name_stats = ibs.get_annot_per_name_stats(daids)
        #imposter_daid_per_name_stats = ut.get_stats(ibs.get_num_annots_per_name(nonquery_daids)[0], use_nan=True)
        #genuine_daid_per_name_stats  = ut.get_stats(ibs.get_num_annots_per_name(groundtruth_daids)[0], use_nan=True)
        #all_daid_per_name_stats = ut.get_stats(ibs.get_num_annots_per_name(daids)[0], use_nan=True)

        # Compare the query yaws to the yaws of its correct matches in the database
        # For each name there will be nQaids:nid x nDaids:nid comparisons
        gt_viewdist_list = compare_correct_match_properties(
            ibs, grouped_qaids, grouped_groundtruth_list, ibs.get_annot_yaws, viewpoint_diff)

        # Compare the query qualities to the qualities of its correct matches in the database
        gt_qualdists_list = compare_correct_match_properties(
            ibs, grouped_qaids, grouped_groundtruth_list, ibs.get_annot_qualities, ut.absdiff)

        # Compare timedelta differences
        gt_hourdelta_list = compare_correct_match_properties(
            ibs, grouped_qaids, grouped_groundtruth_list, ibs.get_annot_image_unixtimes_asfloat, ut.unixtime_hourdiff)

        def super_flatten(arr_list):
            import utool as ut
            return ut.flatten([arr.ravel() for arr in arr_list])

        gt_viewdist_stats   = ut.get_stats(super_flatten(gt_viewdist_list), use_nan=True)
        gt_qualdist_stats  = ut.get_stats(super_flatten(gt_qualdists_list), use_nan=True)
        gt_hourdelta_stats = ut.get_stats(super_flatten(gt_hourdelta_list), use_nan=True)

        qaid_stats_dict = ibs.get_annot_stats_dict(qaids, 'q')
        daid_stats_dict = ibs.get_annot_stats_dict(daids, 'd')

        # Intersections between qaids and daids
        common_aids = np.intersect1d(daids, qaids)

        qnids = ut.unique_unordered(ibs.get_annot_name_rowids(qaids))
        dnids = ut.unique_unordered(ibs.get_annot_name_rowids(daids))
        common_nids = np.intersect1d(qnids, dnids)

        annotconfig_stats_strs1 = ut.odict([
            ('dbname', ibs.get_dbname()),
            ('num_qaids', (len(qaids))),
            ('num_daids', (len(daids))),
            ('num_annot_intersect', (len(common_aids))),
            ('qaid_stats', qaid_stats_dict),
            ('daid_stats', daid_stats_dict),
            ('num_unmatchable_queries', len(unmatchable_queries)),
            ('num_matchable_queries', len(matchable_queries)),
            ('num_qnids', (len(qnids))),
            ('num_dnids', (len(dnids))),
            ('num_name_intersect', (len(common_nids))),
        ])
        annotconfig_stats_strs2 = ut.odict([
            # Number of aids per name for everything in the database
            ('aids_per_database_name', _stat_str(all_daid_per_name_stats)),
            # Number of aids in each name that should match to a query
            # (not quite sure how to phrase what this is)
            ('aids_per_genuine_name', _stat_str(genuine_daid_per_name_stats)),
            # Number of aids in each name that should not match to any query
            # (not quite sure how to phrase what this is)
            ('aids_per_imposter_name', _stat_str(imposter_daid_per_name_stats)),
            # Distances between a query and its groundtruth
            ('viewdist', _stat_str(gt_viewdist_stats)),
            ('qualdist', _stat_str(gt_qualdist_stats)),
            ('hourdist', _stat_str(gt_hourdelta_stats)),
        ])
        annotconfig_stats_strs = ut.odict(annotconfig_stats_strs1.items() + annotconfig_stats_strs2.items())
        stats_str = ut.dict_str(annotconfig_stats_strs1, strvals=True, newlines=False, explicit=True, nobraces=True)
        stats_str +=  '\n' + ut.dict_str(annotconfig_stats_strs2, strvals=True, newlines=True, explicit=True, nobraces=True)
        #stats_str = ut.align(stats_str, ':')
        stats_str2 = ut.dict_str(annotconfig_stats_strs, strvals=True, newlines=True, explicit=False, nobraces=False)
        if verbose:
            print('annot_config_stats = ' + stats_str2)

        return annotconfig_stats_strs, locals()


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.ibsfuncs
        python -m ibeis.ibsfuncs --allexamples
        python -m ibeis.ibsfuncs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
