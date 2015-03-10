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
from six.moves import zip, range, map
from os.path import split, join, exists, commonprefix
import vtool.image as gtool
import numpy as np
from utool._internal.meta_util_six import get_funcname, get_imfunc, set_funcname
from vtool import linalg, geometry, image
import vtool as vt
import utool as ut
import ibeis
from ibeis import params
from ibeis import constants as const
try:
    from detecttools.pypascalmarkup import PascalVOC_Markup_Annotation
except ImportError as ex:
    ut.printex('COMMIT TO DETECTTOOLS')
    pass
from ibeis.control.accessor_decors import getter_1to1

# Inject utool functions
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[ibsfuncs]', DEBUG=False)


# Try to work around circular import
#from ibeis.control.IBEISControl import IBEISController  # Must import class before injection
CLASS_INJECT_KEY = ('IBEISController', 'ibsfuncs')
__injectable = ut.make_class_method_decorator(CLASS_INJECT_KEY, __name__)


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


def export_to_xml(ibs, offset=2829, enforce_yaw=True):
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
    gid_list = ibs.get_valid_gids(reviewed=1)
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
            out_name = "2014_%06d" % offset
            out_img = out_name + ".jpg"
            folder = "IBEIS"

            _image = image.imread(image_path)
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
            _image = image.resize(_image, (width, height))
            image.imwrite(dst_img, _image)

            annotation = PascalVOC_Markup_Annotation(dst_img, folder, out_img, source=image_uri, **information)
            bbox_list = ibs.get_annot_bboxes(aid_list)
            theta_list = ibs.get_annot_thetas(aid_list)
            for aid, bbox, theta in zip(aid_list, bbox_list, theta_list):
                # Transformation matrix
                R = linalg.rotation_around_bbox_mat3x3(theta, bbox)
                # Get verticies of the annotation polygon
                verts = geometry.verts_from_bbox(bbox, close=True)
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
    fid_list = ibs.add_chip_feats(cid_list)
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
        fid_list = ibs.add_chip_feats(cid_list)
    if featweights:
        featweight_rowid_list = ibs.add_feat_featweights(fid_list)  # NOQA


@__injectable
def get_empty_gids(ibs, eid=None):
    """ returns gid list without any chips """
    gid_list = ibs.get_valid_gids(eid=eid)
    nRois_list = ibs.get_image_num_annotations(gid_list)
    empty_gids = [gid for gid, nRois in zip(gid_list, nRois_list) if nRois == 0]
    return empty_gids


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
    if ut.NO_ASSERTS:
        return
    valid_aids = set(ibs.get_valid_aids())
    #invalid_aids = [aid for aid in aid_list if aid not in valid_aids]
    isinvalid_list = [aid not in valid_aids for aid in aid_list]
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
        aid_list = ibs.get_valid_gids()
    print('check annot consistency. len(aid_list)=%r' % len(aid_list))
    annot_gid_list = ibs.get_annot_gids(aid_list)
    assert_images_exist(ibs, annot_gid_list)
    unique_gids = list(set(annot_gid_list))
    print('num_unique_images=%r / %r' % (len(unique_gids), len(annot_gid_list)))
    cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=False)
    cfpath_list = ibs.get_chip_uris(cid_list)
    valid_chip_list = [None if cfpath is None else exists(cfpath) for cfpath in cfpath_list]
    invalid_list = [flag is False for flag in valid_chip_list]
    invalid_cids = ut.filter_items(cid_list, invalid_list)
    if len(invalid_cids) > 0:
        print('found %d inconsistent chips attempting to fix' % len(invalid_cids))
        ibs.delete_chips(invalid_cids, verbose=True)

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
    for aids, sids in zip(aids_list, species_rowids_list):
        assert ut.list_allsame(sids), \
            'aids=%r have the same name, but belong to multiple species=%r' % (aids, ibs.get_species_texts(ut.unique_keep_order2(sids)))


@__injectable
def check_annot_size(ibs):
    print('Checking annot sizes')
    aid_list = ibs.get_valid_aids()
    uuid_list = ibs.get_annot_uuids(aid_list)
    desc_list = ibs.get_annot_vecs(aid_list)
    kpts_list = ibs.get_annot_kpts(aid_list)
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
def check_consistency(ibs, embed=False):
    """ Function to run all database consistency checks

    Rename to run_check_consistency_scripts

    """
    print('[ibsfuncs] Checking consistency')
    gid_list = ibs.get_valid_gids()
    aid_list = ibs.get_valid_aids()
    nid_list = ibs.get_valid_nids()
    check_annot_size(ibs)
    check_image_consistency(ibs, gid_list)
    check_annot_consistency(ibs, aid_list)
    check_name_consistency(ibs, nid_list)
    # Very slow check
    check_image_uuid_consistency(ibs, gid_list)
    if embed:
        ut.embed()
    print('[ibsfuncs] Finshed consistency check')


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

    """
    #TODO: Call more stuff, maybe rename to 'apply duct tape'
    ibs.fix_unknown_exemplars()
    ibs.fix_invalid_name_texts()
    ibs.fix_invalid_nids()
    ibs.update_annot_visual_uuids(ibs.get_valid_aids())
    ibs.delete_empty_nids()
    ibs.delete_empty_eids()
    ibs.db.vacuum()


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
    def fix_notes(notes, is_hard, flag):
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

    new_notes_list = [fix_notes(notes, is_hard, flag) for notes, is_hard, flag in zip(notes_list, is_hard_list, flag_list)]
    ibs.set_annot_notes(aid_list, new_notes_list)
    return is_hard_list


@__injectable
def get_annot_bbox_area(ibs, aid_list):
    bbox_list = ibs.get_annot_bboxes(aid_list)
    area_list = [bbox[2] * bbox[3] for bbox in bbox_list]
    return area_list


#@__injectable
def DEPRICATE_rebase_images(ibs, new_path, gid_list=None):
    """
    DEPRICATE

    Moves the images into the ibeis image cache.
    Images are renamed to img_uuid.ext
    """
    if gid_list is None:
        gid_list  = ibs.get_valid_gids()
        #new_path = 'G:\PZ_Ol_Pejeta_All'
    gpath_list = ibs.get_image_paths(gid_list)
    len_prefix = len(commonprefix(gpath_list))
    #if common_prefix.rfind('/')
    # assum
    gname_list = [gpath[len_prefix:] for gpath in gpath_list]
    new_gpath_list = [join(new_path, gname) for gname in gname_list]
    new_gpath_list = list(map(ut.unixpath, new_gpath_list))
    #orig_exists = map(exists, gpath_list)
    new_exists = list(map(exists, new_gpath_list))
    assert all(new_exists), 'some rebased images do not exist'
    ibs.set_image_uris(gid_list, new_gpath_list)
    assert  'not all images copied'


@__injectable
@getter_1to1
def is_nid_unknown(ibs, nid_list):
    return [ nid <= 0 for nid in nid_list]


@__injectable
def get_match_text(ibs, aid1, aid2):
    truth = ibs.get_match_truth(aid1, aid2)
    text = {
        2: 'NEW Match ',
        0: 'JOIN Match ',
        1: 'SPLIT Match ',
    }.get(truth, None)
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
    # Then _unflatten the results to the original input dimensions
    unflat_vals = _unflatten(flat_vals, reverse_list)
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


def delete_ibeis_database(dbdir):
    _ibsdb = join(dbdir, const.PATH_NAMES._ibsdb)
    print('[ibsfuncs] DELETEING: _ibsdb=%r' % _ibsdb)
    if exists(_ibsdb):
        ut.delete(_ibsdb)


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


def list_images(img_dir, fullpath=True, recursive=True):
    """ lists images that are not in an internal cache """
    ignore_list = ['_hsdb', '.hs_internals', '_ibeis_cache', '_ibsdb']
    gpath_list = ut.list_images(img_dir,
                                   fullpath=fullpath,
                                   recursive=recursive,
                                   ignore_list=ignore_list)
    return gpath_list


def get_species_dbs(species_prefix):
    from ibeis.dev import sysres
    ibs_dblist = sysres.get_ibsdb_list()
    isvalid_list = [split(path)[1].startswith(species_prefix) for path in ibs_dblist]
    return ut.filter_items(ibs_dblist, isvalid_list)


#def delete_non_exemplars(ibs):
#    """ deletes images without exemplars """
#    gid_list = ibs.get_valid_gids
#    aids_list = ibs.get_image_aids(gid_list)
#    flags_list = unflat_map(ibs.get_annot_exemplar_flags, aids_list)
#    delete_gid_flag_list = [not any(flags) for flags in flags_list]
#    delete_gid_list = ut.filter_items(gid_list, delete_gid_flag_list)
#    ibs.delete_images(delete_gid_list)
#    delete_empty_eids(ibs)
#    delete_empty_nids(ibs)


#@__injectable
#@ut.time_func
#@profile
#def update_reviewed_image_encounter(ibs):
#    # FIXME SLOW
#    #ibs.delete_encounters(eid)
#    ibs.delete_egr_encounter_relations(eid)
#    #gid_list = ibs.get_valid_gids(reviewed=True)
#    gid_list = _get_reviewed_gids(ibs)  # hack
#    #ibs.set_image_enctext(gid_list, [const.REVIEWED_IMAGE_ENCTEXT] * len(gid_list))
#    ibs.set_image_eids(gid_list, [eid] * len(gid_list))


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
@ut.time_func
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
    ungrouped_eid = ibs.get_encounter_eids_from_text(const.UNGROUPED_IMAGES_ENCTEXT)
    ibs.delete_egr_encounter_relations(ungrouped_eid)
    ungrouped_gids = ibs.get_ungrouped_gids()
    ibs.set_image_eids(ungrouped_gids, [ungrouped_eid] * len(ungrouped_gids))


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


def get_title(ibs):
    if ibs is None:
        title = 'IBEIS - No Database Directory Open'
    elif ibs.dbdir is None:
        title = 'IBEIS - !! INVALID DATABASE !!'
    else:
        dbdir = ibs.get_dbdir()
        dbname = ibs.get_dbname()
        title = 'IBEIS - %r - Database Directory = %s' % (dbname, dbdir)
        wb_target = params.args.wildbook_target
        if wb_target is not None:
            title = '%s - Wildbook Target = %s' % (title, wb_target)
    return title


@__injectable
def print_stats(ibs):
    from ibeis.dev import dbinfo
    dbinfo.dbstats(ibs)


@__injectable
def print_dbinfo(ibs, **kwargs):
    from ibeis.dev import dbinfo
    dbinfo.get_dbinfo(ibs, *kwargs)


@__injectable
def print_infostr(ibs, **kwargs):
    print(ibs.get_infostr())


@__injectable
def get_dbinfo_str(ibs):
    from ibeis.dev import dbinfo
    return dbinfo.get_dbinfo(ibs, verbose=False)['info_str']


@__injectable
def get_infostr(ibs):
    """ Returns printable database information

    Args:
        ibs (IBEISController):  ibeis controller object

    Returns:
        str: infostr

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_infostr

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> # execute function
        >>> infostr = get_infostr(ibs)
        >>> # verify results
        >>> result = str(infostr)
        >>> print(result)
        dbname = 'testdb1'
        num_images = 13
        num_annotations = 13
        num_names = 7
    """
    dbname = ibs.get_dbname()
    #workdir = ut.unixpath(ibs.get_workdir())
    num_images = ibs.get_num_images()
    num_annotations = ibs.get_num_annotations()
    num_names = ibs.get_num_names()
    #workdir = %r
    infostr = ut.codeblock('''
    dbname = %r
    num_images = %r
    num_annotations = %r
    num_names = %r
    ''' % (dbname, num_images, num_annotations, num_names))
    return infostr


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
    nid_list = ibs.get_annot_name_rowids(aid_list)
    return ibs.is_nid_unknown(nid_list)


def make_enctext_list(eid_list, enc_cfgstr):
    # DEPRICATE
    enctext_list = [str(eid) + enc_cfgstr for eid in eid_list]
    return enctext_list


@__injectable
def batch_rename_consecutive_via_species(ibs):
    """ actually sets the new consectuive names"""
    new_nid_list, new_name_list = ibs.get_consecutive_newname_list_via_species()
    ibs.set_name_texts(new_nid_list, new_name_list, verbose=ut.NOT_QUIET)


@__injectable
def get_consecutive_newname_list_via_species(ibs):
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
        >>> new_nid_list, new_name_list = get_consecutive_newname_list_via_species(ibs)
        >>> result = ut.list_str((new_nid_list, new_name_list))
        >>> # verify results
        >>> print(result)
        (
            [1, 2, 3, 4, 5, 6, 7],
            ['IBEIS_PZ_0001', 'IBEIS_PZ_0002', 'IBEIS_UNKNOWN_0001', 'IBEIS_UNKNOWN_0002', 'IBEIS_GZ_0001', 'IBEIS_PB_0001', 'IBEIS_UNKNOWN_0003'],
        )
    """
    print('[ibs] get_consecutive_newname_list_via_species')
    ibs.delete_empty_nids()
    nid_list  = ibs.get_valid_nids()
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
    new_name_list = ['%s_%s_%04d' % (location_text, code, get_next_index(code)) for code in code_list]
    new_nid_list = nid_list
    return new_nid_list, new_name_list


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
        >>> # verify results
        >>> # FIXME: nautiluses are not working right
        >>> result = str(name_list)
        >>> print(result)
        ['IBEIS_UNKNOWN_0008', 'IBEIS_PZ_0042', 'IBEIS_GZ_0004', 'IBEIS_GZ_0008']

    """
    if species_text is None:
        # TODO: optionally specify qreq_ or qparams?
        species_text  = ibs.cfg.detect_cfg.species_text
    if location_text is None:
        location_text = ibs.cfg.other_cfg.location_for_names
    base_index = len(ibs._get_all_known_name_rowids()) + 1  # ibs.get_num_names()
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
    if num is None:
        next_name = name_prefix + '%04d' % base_index
        return next_name
    else:
        next_names = [name_prefix + '%04d' % (base_index + x) for x in range(num)]
        return next_names


@__injectable
def prune_exemplars(ibs):
    r"""
    Prunes exemplars from names with too many exemplars.

    Args:
        ibs (IBEISController):
    """
    nid_list = ibs.get_valid_nids()
    aids_list = ibs.get_name_exemplar_aids(nid_list)
    max_exemplars = ibs.cfg.other_cfg.max_exemplars
    problem_aids = [np.array(aids) for aids in aids_list if len(aids) > max_exemplars]
    problem_bboxes = unflat_map(ibs.get_annot_bboxes, problem_aids)
    #problem_gids   = unflat_map(ibs.get_annot_gids, problem_aids)
    #problem_sizes  = unflat_map(ibs.get_image_sizes, problem_gids)
    def bbox_area(bbox):
        return bbox[-2] * bbox[-1]
    def bboxes_area(bbox_list):
        return list(map(bbox_area, bbox_list))

    # Get area of annotations, area of parent images, and the ratio

    problem_annot_areas = list(map(np.array, list(map(bboxes_area, problem_bboxes))))

    #problem_img_areas = list(map(np.array, list(map(bboxes_area, problem_sizes))))

    #problem_ratios = [(annot_areas / img_areas) for annot_areas, img_areas in
    #                  zip(problem_annot_areas, problem_img_areas)]

    problem_sortx = [areas.argsort() for areas in problem_annot_areas]
    # Get aids with the smallest bounding boxes to unexemplar
    small_aids_list = [aids[sortx][:-max_exemplars] for aids, sortx in zip(problem_aids, problem_sortx)]
    small_aids = ut.flatten(small_aids_list)
    ibs.set_annot_exemplar_flags(small_aids, [False] * len(small_aids))


def draw_thumb_helper(tup):
    thumb_path, thumbsize, gpath, bbox_list, theta_list = tup
    img = gtool.imread(gpath)  # time consuming
    (gh, gw) = img.shape[0:2]
    img_size = (gw, gh)
    max_dsize = (thumbsize, thumbsize)
    dsize, sx, sy = gtool.resized_clamped_thumb_dims(img_size, max_dsize)
    new_verts_list = list(gtool.scaled_verts_from_bbox_gen(bbox_list, theta_list, sx, sy))
    #thumb = gtool.resize_thumb(img, max_dsize)
    # -----------------
    # Actual computation
    thumb = gtool.resize(img, dsize)
    orange_bgr = (0, 128, 255)
    for new_verts in new_verts_list:
        thumb = geometry.draw_verts(thumb, new_verts, color=orange_bgr, thickness=2)
    gtool.imwrite(thumb_path, thumb)
    return True
    #return (thumb_path, thumb)


@__injectable
def preprocess_image_thumbs(
        ibs, gid_list=None, use_cache=True, chunksize=8, **kwargs):
    """ Computes thumbs of images in parallel based on kwargs """
    print('[ibsfuncs] preprocess_image_thumbs')
    if gid_list is None:
        gid_list = ibs.get_valid_gids(**kwargs)
    thumbsize = 128
    thumbpath_list = ibs.get_image_thumbpath(gid_list, thumbsize=thumbsize)
    #use_cache = False
    if use_cache:
        exists_list = list(map(exists, thumbpath_list))
        gid_list_ = ut.filterfalse_items(gid_list, exists_list)
        thumbpath_list_ = ut.filterfalse_items(thumbpath_list, exists_list)
    else:
        gid_list_ = gid_list
        thumbpath_list_ = thumbpath_list
    gpath_list = ibs.get_image_paths(gid_list_)

    aids_list = ibs.get_image_aids(gid_list_)
    bboxes_list = unflat_map(ibs.get_annot_bboxes, aids_list)
    thetas_list = unflat_map(ibs.get_annot_thetas, aids_list)

    args_list = [(thumb_path, thumbsize, gpath, bbox_list, theta_list)
                 for thumb_path, gpath, bbox_list, theta_list in
                 zip(thumbpath_list_, gpath_list, bboxes_list, thetas_list)]

    # Execute all tasks in parallel
    genkw = {
        'ordered': False,
        'chunksize': chunksize,
        #'force_serial': True,
    }
    #genkw['force_serial'] = True
    #genkw['chunksize'] = max(len(gid_list_) // 16, 1)
    gen = ut.generate(draw_thumb_helper, args_list, nTasks=len(args_list), **genkw)
    #for output in gen:
    #    #with ut.Timer('gentime'):
    #    gtool.imwrite(output[0], output[1])
    try:
        while True:
            six.next(gen)
    except StopIteration:
        pass


@__injectable
def compute_all_thumbs(ibs, **kwargs):
    preprocess_image_thumbs(ibs, **kwargs)


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
        # http://stackoverflow.com/questions/482014/how-would-you-do-the-equivalent-of-preprocessor-directives-in-python
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
def get_annot_rowid_sample(ibs, per_name=1, min_ngt=1, seed=0, aid_list=None,
                           stagger_names=False):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        per_name (int):
        min_ngt (int):
        seed (int):
        aid_list (list):

    Returns:
        ?: sample_aids

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_annot_rowid_sample

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> per_name = 100
        >>> min_ngt = 1
        >>> seed = 0
        >>> # execute function
        >>> sample_aids = ibs.get_annot_rowid_sample(per_name, min_ngt, seed)
        >>> # verify results
        >>> # FIXME
        >>> #result = str(sample_aids)
        >>> #print(result)
    """
    #qaids = ibs.get_easy_annot_rowids()
    if aid_list is None:
        aid_list = np.array(ibs.get_valid_aids())
    grouped_aids_, unique_nids = ibs.group_annots_by_name(aid_list, distinguish_unknowns=False)
    grouped_aids = list(filter(lambda x: len(x) > min_ngt, grouped_aids_))
    if stagger_names:
        from six.moves import zip_longest
        sample_aids = ut.filter_Nones(ut.iflatten(zip_longest(*ut.sample_lists(grouped_aids, num=per_name, seed=seed))))
    else:
        sample_aids = ut.flatten(ut.sample_lists(grouped_aids, num=per_name, seed=seed))

    return sample_aids


@__injectable
def get_annot_groundfalse_sample(ibs, aid_list, per_name=1, seed=False):
    """
    get_annot_groundfalse_sample

    FIXME

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
def get_aids_with_groundtruth(ibs):
    """ returns aids with valid groundtruth """
    valid_aids = ibs.get_valid_aids()
    has_gt_list = ibs.get_annot_has_groundtruth(valid_aids)
    hasgt_aids = ut.filter_items(valid_aids, has_gt_list)
    return hasgt_aids


@__injectable
def export_encounters(ibs, eid_list, new_dbdir=None):
    gid_list = list(set(ut.flatten(ibs.get_encounter_gids(eid_list))))
    if new_dbdir is None:
        from ibeis.dev import sysres
        dbname = ibs.get_dbname()
        enc_texts = ', '.join(ibs.get_encounter_enctext(eid_list)).replace(' ', '-')
        nimg_text = 'nImg=%r' % len(gid_list)
        new_dbname = dbname + '_' + enc_texts + '_' + nimg_text
        workdir = sysres.get_workdir()
        new_dbdir_ = join(workdir, new_dbname)
    else:
        new_dbdir_ = new_dbdir
    ibs.export_images(gid_list, new_dbdir_=new_dbdir_)


@__injectable
def export_images(ibs, gid_list, new_dbdir_):
    """ See ibeis/tests/test_ibs_export.py """
    from ibeis.dbio import export_subset
    print('[ibsfuncs] exporting to new_dbdir_=%r' % new_dbdir_)
    print('[ibsfuncs] opening database')
    ibs_dst = ibeis.opendb(dbdir=new_dbdir_, allow_newdir=True)
    print('[ibsfuncs] begining transfer')
    ibs_src = ibs
    gid_list1 = gid_list
    aid_list1 = None
    include_image_annots = True
    status = export_subset.execute_transfer(ibs_src, ibs_dst, gid_list1,
                                            aid_list1, include_image_annots)
    return status


@__injectable
def set_dbnotes(ibs, notes):
    """ sets notes for an entire database """
    assert isinstance(ibs, ibeis.control.IBEISControl.IBEISController)
    ut.write_to(ibs.get_dbnotes_fpath(), notes)


@__injectable
def get_dbnotes_fpath(ibs, ensure=False):
    notes_fpath = join(ibs.get_ibsdir(), 'dbnotes.txt')
    if ensure and not exists(ibs.get_dbnotes_fpath()):
        ibs.set_dbnotes('None')
    return notes_fpath


@__injectable
def get_dbnotes(ibs):
    """ sets notes for an entire database """
    notes = ut.read_from(ibs.get_dbnotes_fpath(), strict=False)
    if notes is None:
        ibs.set_dbnotes('None')
        notes = ut.read_from(ibs.get_dbnotes_fpath())
    return notes


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
            print('[ibs] fuond bad name rowids = %r' % (invalid_nids,))
            print('[ibs] fuond bad name texts  = %r' % (invalid_texts,))
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


def export_testset_for_chuck(ibs, min_num_annots):
    """
    Exports a set with some number of annotations that has good demo examples.
    multiple annotations per name and large time variation within names.

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-export_testset_for_chuck --dbdir /raid/work2/Turk/PZ_Master --min-num-annots 100
        python -m ibeis.ibsfuncs --test-export_testset_for_chuck --dbdir /raid/work2/Turk/PZ_Master --min-num-annots 500

        python -m ibeis.ibsfuncs --test-export_testset_for_chuck --dbdir /raid/work2/Turk/GZ_Master --min-num-annots 100
        python -m ibeis.ibsfuncs --test-export_testset_for_chuck --dbdir
        /raid/work2/Turk/GZ_Master --min-num-annots 500_DOCTEST


        python -m ibeis.ibsfuncs --test-export_testset_for_chuck --db GIR_Tanya --min-num-annots 100

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> #dbdir = ut.get_argval(('--dbdir',), type_=str, default='testdb1')
        >>> min_num_annots = ut.get_argval(('--min-num-annots',), type_=int, default=500)
        >>> #ibs = ibeis.opendb('testdb1')
        >>> #ibs = ibeis.opendb(dbdir='/raid/work2/Turk/PZ_Master')
        >>> ibs = ibeis.opendb()  # dbdir=dbdir)
        >>> #ibs = ibeis.opendb(dbdir='/raid/work2/Turk/GZ_Master')
        >>> print(ibs.get_dbinfo_str())
        >>> #ibs = ibeis.opendb('testdb1')
        >>> # execute function
        >>> result = export_testset_for_chuck(ibs, min_num_annots)
        >>> # verify results
        >>> print(result)

    min_num_annots = 500
    """
    import numpy as np

    min_num_annots_per_name = 3
    max_annot_per_image = 5
    #3

    #min_num_annots_per_name = 1
    #max_annot_per_image = 3000

    nid_list = ibs.get_valid_nids()
    aids_list = ibs.get_name_aids(nid_list)
    nAids_list = list(map(len, aids_list))
    nOther_aids_list = ibs.unflat_map(ibs.get_annot_num_contact_aids, aids_list)

    invalid_by_num_annots = [num < min_num_annots_per_name for num in nAids_list]
    invalid_by_num_others = [any([num > max_annot_per_image for num in nums])
                             for nums in nOther_aids_list]
    invalid_list = ut.or_lists(invalid_by_num_annots, invalid_by_num_others)

    valid_nids = ut.filterfalse_items(nid_list, invalid_list)

    def get_name_time_variation(ibs, nid_list):
        aids_list      = ibs.get_name_aids(nid_list)
        unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes, aids_list)
        unixtimes_arrs = list(map(np.array, unixtimes_list))
        fixtimes_list  = [arr[arr > 0] for arr in unixtimes_arrs]
        std_list       = [np.std(arr) if len(arr) > 1 else 0 for arr in fixtimes_list]
        return std_list

    std_list = get_name_time_variation(ibs, valid_nids)
    sorted_nids = ut.sortedby(valid_nids, std_list, reverse=True)

    # Find which names to include
    num_annot_cumsum = np.cumsum(ibs.get_name_num_annotations(sorted_nids))
    pos_list = np.where(num_annot_cumsum >= min_num_annots)[0]
    assert len(pos_list) > 0

    nid_list_chosen = sorted_nids[:pos_list[0] + 1]
    print('using names:')
    print(ibs.get_name_texts(nid_list_chosen))
    aids_list_chosen = ibs.get_name_aids(nid_list_chosen)
    aid_list_chosen = ut.flatten(aids_list_chosen)
    gid_list_chosen = ibs.get_annot_gids(aid_list_chosen)
    #ut.debug_duplicate_items(gid_list_chosen)

    # make sure not too many other annots are along for the ride
    other_aids = ibs.get_annot_contact_aids(aid_list_chosen)
    unexpected_aids = list(set(ut.flatten(other_aids)).difference(set(aid_list_chosen)))
    print('got %d unexpected_aids' % (len(unexpected_aids),))

    from ibeis.dbio import export_subset

    def new_nonconflicting_dbpath(ibs):
        dpath, dbname = split(ibs.get_dbdir())
        base_fmtstr = dbname + '_demo' + str(min_num_annots) + '_export%d'
        new_dbpath = ut.get_nonconflicting_path(base_fmtstr, dpath)
        return new_dbpath

    #ut.embed()

    dbpath = new_nonconflicting_dbpath(ibs)
    ibs_dst = ibeis.opendb(dbdir=dbpath, allow_newdir=True)
    ibs_src = ibs
    gid_list = gid_list_chosen
    export_subset.merge_databases(ibs_src, ibs_dst, gid_list=gid_list)

    DEBUG_NAME = False
    if DEBUG_NAME:
        ibs.get_name_num_annotations(sorted_nids[0:10])
        import plottool as pt
        ibeis.viz.viz_name.show_name(ibs, sorted_nids[0])
        pt.update()


def export_subdatabase_all_annots_new(ibs):
    """
    Exports a set with some number of annotations that has good demo examples.
    multiple annotations per name and large time variation within names.

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-export_subdatabase_all_annots_new --db GIR_Tanya

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb()
        >>> print(ibs.get_dbinfo_str())
        >>> result = export_subdatabase_all_annots_new(ibs)
        >>> # verify results
        >>> print(result)

    min_num_annots = 500
    """
    aid_list = ibs.get_valid_aids()
    gid_list = list(set(ibs.get_annot_gids(aid_list)))

    from ibeis.dbio import export_subset

    def new_nonconflicting_dbpath(ibs):
        dpath, dbname = split(ibs.get_dbdir())
        base_fmtstr = dbname + '_demo' + 'all_annots' + '_export%d'
        new_dbpath = ut.get_nonconflicting_path(base_fmtstr, dpath)
        return new_dbpath

    #ut.embed()

    dbpath = new_nonconflicting_dbpath(ibs)
    ibs_dst = ibeis.opendb(dbdir=dbpath, allow_newdir=True)
    ibs_src = ibs
    export_subset.merge_databases(ibs_src, ibs_dst, gid_list=gid_list)


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


def get_yaw_viewtexts(yaw_list):
    r"""
    Args:
        yaw (?):

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_yaw_viewtexts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> # build test data
        >>> yaw_list = [0.0, 3.15, -.4, -8, .2, 4, 7, 20, None]
        >>> # execute function
        >>> text_list = get_yaw_viewtexts(yaw_list)
        >>> result = str(text_list)
        >>> # verify results
        >>> print(result)
        ['right', 'left', 'backright', 'back', 'right', 'backleft', 'frontright', 'frontright', None]
    """
    import vtool as vt
    import numpy as np
    import six
    stdlblyaw_list = list(six.iteritems(const.VIEWTEXT_TO_YAW_RADIANS))
    stdlbl_list = ut.get_list_column(stdlblyaw_list, 0)
    stdyaw_list = np.array(ut.get_list_column(stdlblyaw_list, 1))
    textdists_list = [None if yaw is None else vt.ori_distance(stdyaw_list, yaw) for yaw in yaw_list]
    index_list = [None if dists is None else dists.argmin() for dists in textdists_list]
    text_list = [None if index is None else stdlbl_list[index] for index in index_list]
    #errors = ['%.2f' % dists[index] for dists, index in zip(textdists_list, index_list)]
    #return list(zip(yaw_list, errors, text_list))
    return text_list


def GreatZebraCount_batch_rename(ibs):
    """
    python dev.py --db PZ_MUGU_19 --cmd
    python dev.py --db PZ_MUGU_20 --cmd
    python dev.py --db PZ_MUGU_18 --cmd
    python dev.py --db GIRM_MUGU_20 --cmd
    """
    nid_list = ibs.get_valid_nids()
    name_list = ibs.get_name_texts(nid_list)
    new_name_list = [name.replace('IBEIS', 'MUGU') for name in name_list]
    ibs.set_name_texts(nid_list, new_name_list)
    print(ibs.get_name_texts(nid_list))
    ibs.cfg.other_cfg.location_for_names = 'MUGU'
    ibs.cfg.save()


def GreatZebraCount_mergedbs():
    """ docstr """

    import ibeis
    #ibeis.opendb('PZ_MUGU_18')
    #ibeis.opendb('GIRM_MUGU_20')

    ibs_pz_mugu_18 = ibeis.opendb('PZ_MUGU_18')
    ibs_pz_mugu_19 = ibeis.opendb('PZ_MUGU_19')
    ibs_pz_mugu_20 = ibeis.opendb('PZ_MUGU_20')

    ibs_pz_mugu_all  = ibeis.opendb('PZ_MUGU_ALL', allow_newdir=True)
    #ibs_mugu_all = ibeis.opendb('PZ_MUGU_ALL', allow_newdir=True, delete_ibsdir=True)

    def merge_rename(ibs, num):
        nid_list = ibs.get_valid_nids()
        name_list = ibs.get_name_texts(nid_list)
        assert all(['IBEIS' not in name for name in name_list])
        repl = 'MUGU_%d' % num
        new_name_list = [
            name if repl in name else name.replace('MUGU', repl)
            for name in name_list
        ]
        ibs.set_name_texts(nid_list, new_name_list)

    merge_rename(ibs_pz_mugu_18, 18)
    merge_rename(ibs_pz_mugu_19, 19)
    merge_rename(ibs_pz_mugu_20, 20)

    ibs_pz_mugu_18.print_name_table()
    ibs_pz_mugu_20.print_name_table()
    ibs_pz_mugu_19.print_name_table()
    ibs_pz_mugu_all.print_name_table()

    #ibeis.ibsfuncs.GreatZebraCount_batch_rename(ibs_pz_mugu_19)
    #ibeis.ibsfuncs.GreatZebraCount_batch_rename(ibs_pz_mugu_20)

    ibs_pz_mugu_18.fix_and_clean_database()
    ibs_pz_mugu_19.fix_and_clean_database()
    ibs_pz_mugu_20.fix_and_clean_database()
    ibs_pz_mugu_all.fix_and_clean_database()

    ibs_pz_mugu_18.print_dbinfo()
    ibs_pz_mugu_19.print_dbinfo()
    ibs_pz_mugu_20.print_dbinfo()
    ibs_pz_mugu_all.print_dbinfo()

    ibs_pz_mugu_19.print_infostr()
    ibs_pz_mugu_20.print_infostr()
    ibs_pz_mugu_all.print_infostr()

    # Merging works from {19}->{}
    ibeis.dbio.export_subset.merge_databases(ibs_pz_mugu_18, ibs_pz_mugu_all)
    ibeis.dbio.export_subset.merge_databases(ibs_pz_mugu_19, ibs_pz_mugu_all)
    # Remerging ignores due to the contributor from {19}->{19} This should be
    # changed. Images may be added and annots may be added also things might be
    # changed.
    #ibeis.dbio.export_subset.merge_databases(ibs_pz_mugu_19, ibs_mugu_all)
    ibeis.dbio.export_subset.merge_databases(ibs_pz_mugu_20, ibs_pz_mugu_all)

    len(ibs_pz_mugu_all.get_name_texts(ibs_pz_mugu_all.get_valid_nids()))
    ibs_pz_mugu_all.get_name_texts(ibs_pz_mugu_all.get_valid_nids())

    ####

    ibs_girm_mugu_20 = ibeis.opendb('GIRM_MUGU_20', allow_newdir=True)
    ibs_girm_mugu_20.fix_and_clean_database()

    # Master multi-species database
    ibs_pz_mugu_all = ibeis.opendb('PZ_MUGU_ALL')
    ibs_mugu_master = ibeis.opendb('MUGU_MASTER', allow_newdir=True)
    # Merge all PZ into Master
    ibeis.dbio.export_subset.merge_databases(ibs_pz_mugu_all, ibs_mugu_master)
    # Merge all GIRM into Master
    ibeis.dbio.export_subset.merge_databases(ibs_girm_mugu_20, ibs_mugu_master)
    ibs_mugu_master.print_dbinfo()
    ibs_mugu_master.print_dbinfo()


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
def set_exemplars_from_quality_and_viewpoint(ibs, exemplars_per_view=None, dry_run=False, verbose=False):
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
    """
    # General params
    if exemplars_per_view is None:
        exemplars_per_view = ibs.cfg.other_cfg.exemplars_per_view
    PREFER_GOOD_OVER_OLDFLAG = True
    verbose = False
    dry_run = False
    #
    # Params for knapsack
    def make_knapsack_params(N, levels_per_tier_list):
        """
        Args:
            N (int): the integral maximum number of items
            levels_per_tier_list (list): list of number of distinctions possible
            per tier.

        Returns:
            per-item weights, weights go group items into several tiers, and an
            infeasible weight
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
    N = exemplars_per_view
    levels_per_tier_list = [3, 1, 1]
    w, tier_w_list, infeasible_w = make_knapsack_params(N, levels_per_tier_list)

    qual2_weight = {
        const.QUAL_EXCELLENT : w + tier_w_list[0] + tier_w_list[1],
        const.QUAL_GOOD      : w + tier_w_list[0],
        const.QUAL_OK        : w + tier_w_list[1],
        const.QUAL_UNKNOWN   : w + tier_w_list[1],
        const.QUAL_POOR      : w + tier_w_list[2],
        const.QUAL_JUNK      : infeasible_w,
    }
    # this probably broke with the introduction of 2 more tiers
    oldflag_offset = (
        # always prefer good over ok
        tier_w_list[0] - tier_w_list[1]
        if PREFER_GOOD_OVER_OLDFLAG else
        # prefer ok over good when ok has oldflag
        tier_w_list[0] + tier_w_list[1]
    )

    def choose_exemplars(aids):
        qualtexts = ibs.get_annot_quality_texts(aids)
        oldflags = ibs.get_annot_exemplar_flags(aids)
        # We like good more than ok, and junk is infeasible We prefer items that
        # had previously been exemplars Build input for knapsack
        weights = [qual2_weight[qual] + oldflag_offset * oldflag
                   for qual, oldflag in zip(qualtexts, oldflags)]
        values = [1] * len(weights)
        indices = list(range(len(weights)))
        items = list(zip(values, weights, indices))
        total_value, chosen_items = ut.knapsack(items, N)
        chosen_indices = ut.get_list_column(chosen_items, 2)
        new_flags = [False] * len(aids)
        for index in chosen_indices:
            new_flags[index] = True
        return new_flags

    def get_changed_infostr(yawtext, aids, new_flags):
        old_flags = ibs.get_annot_exemplar_flags(aids)
        quals = ibs.get_annot_quality_texts(aids)
        ischanged = ut.xor_lists(old_flags, new_flags)
        changed_list = ['***' if flag else ''
                        for flag in ischanged]
        infolist = list(zip(aids, quals, old_flags, new_flags, changed_list))
        infostr = ('yawtext=%r:\n' % (yawtext,)) + ut.list_str(infolist)
        return infostr

    aid_list = ibs.get_valid_aids()
    aids_list, unique_nids  = ibs.group_annots_by_name(aid_list)
    # for final settings because I'm too lazy to write
    # this correctly using group_indicies instead of group_items
    new_aid_list = []
    new_flag_list = []
    _iter = ut.ProgressIter(zip(aids_list, unique_nids), nTotal=len(aids_list), lbl='Optimizing name exemplars')
    for aids_, nid in _iter:
        if ibs.is_nid_unknown(nid):
            # do not change unknown animals
            continue
        yawtexts  = ibs.get_annot_yaw_texts(aids_)
        yawtext2_aids = ut.group_items(aids_, yawtexts)
        if verbose:
            print('+ ---')
            print('  nid=%r' % (nid))
        for yawtext, aids in six.iteritems(yawtext2_aids):
            new_flags = choose_exemplars(aids)
            if verbose:
                print(ut.indent(get_changed_infostr(yawtext, aids, new_flags)))
            new_aid_list.extend(aids)
            new_flag_list.extend(new_flags)
        if verbose:
            print('L ___')

    if not dry_run:
        ibs.set_annot_exemplar_flags(new_aid_list, new_flag_list)
    return new_aid_list, new_flag_list


def detect_join_cases(ibs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    Returns:
        QueryResult: qres_list -  object of feature correspondences and scores

    CommandLine:
        python -m ibeis.ibsfuncs --test-detect_join_cases

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> # execute function
        >>> qres_list = detect_join_cases(ibs)
        >>> # verify results
        >>> result = str(qres_list)
        >>> print(result)
    """
    qaids = ibs.get_valid_aids(is_exemplar=None, nojunk=True)
    daids = ibs.get_valid_aids(is_exemplar=None, nojunk=True)
    cfgdict = dict(can_match_samename=False)
    qreq_ = ibs.new_query_request(qaids, daids, cfgdict)
    qres_list = ibs.query_chips(qreq_=qreq_)

    from ibeis.gui import inspect_gui
    qaid2_qres = {qres.qaid: qres for qres in qres_list}
    qres_wgt = inspect_gui.QueryResultsWidget(ibs, qaid2_qres)
    qres_wgt.show()
    qres_wgt.raise_()
    #return qres_list


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
