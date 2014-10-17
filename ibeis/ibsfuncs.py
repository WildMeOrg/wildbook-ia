"""
python -c "import doctest, ibeis; print(doctest.testmod(ibeis.ibsfuncs))"
python -m doctest -v ibeis/model/hots/hots_nn_index.py
python -m doctest ibeis/model/hots/hots_nn_index.py
"""
# developer convenience functions for ibs
from __future__ import absolute_import, division, print_function
#import uuid
import six
import types
from six.moves import zip, range, map
from utool._internal.meta_util_six import get_funcname, get_imfunc, set_funcname
from functools import partial
from os.path import split, join, exists, commonprefix
import utool
import ibeis
from ibeis import constants
from ibeis.dev import sysres
from ibeis.io import export_hsdb
try:
    from detecttools.pypascalmarkup import PascalVOC_Markup_Annotation
except ImportError as ex:
    utool.printex('COMMIT TO DETECTTOOLS')
    pass
#from ibeis import constants
from ibeis.control.accessor_decors import getter_1to1
from vtool import linalg, geometry, image
import vtool.image as gtool
import numpy as np

# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[ibsfuncs]', DEBUG=False)


__INJECTABLE_FUNCS__ = []


def __injectable(input_):
    def closure_injectable(func, indent=True):
        global __INJECTABLE_FUNCS__
        if indent:
            func_ = utool.indent_func(func)
        else:
            func_ = func
        __INJECTABLE_FUNCS__.append(func_)
        return func_
    if utool.is_funclike(input_):
        return closure_injectable(input_)
    else:
        return partial(closure_injectable, indent=input_)


def inject_ibeis(ibs):
    """ Injects custom functions into an IBEISController """
    # Give the ibeis object the inject_func_as_method
    utool.inject_func_as_method(ibs, utool.inject_func_as_method)
    # List of getters to unflatten
    to_unflatten = [
        ibs.get_annot_uuids,
        ibs.get_image_uuids,
        ibs.get_name_text,
        ibs.get_image_unixtime,
        ibs.get_annot_bboxes,
        ibs.get_annot_thetas,
    ]
    for flat_getter in to_unflatten:
        unflat_getter = _make_unflat_getter_func(flat_getter)
        ibs.inject_func_as_method(unflat_getter)
    for func in __INJECTABLE_FUNCS__:
        ibs.inject_func_as_method(func)


@__injectable
def refresh(ibs):
    from ibeis import ibsfuncs
    from ibeis import all_imports
    ibsfuncs.rrr()
    all_imports.reload_all()
    ibsfuncs.inject_ibeis(ibs)


def export_to_xml(ibs):
    offset = 9921
    target_size = 900
    information = {
        'database_name' : ibs.get_dbname()
    }
    datadir = ibs._ibsdb + "/LearningData/"
    imagedir = datadir + 'JPEGImages/'
    annotdir = datadir + 'Annotations/'
    utool.ensuredir(datadir)
    utool.ensuredir(imagedir)
    utool.ensuredir(annotdir)
    gid_list = ibs.get_valid_gids()
    for gid in gid_list:
        aid_list = ibs.get_image_aids(gid)
        image_uri = ibs.get_image_uris(gid)
        image_path = ibs.get_image_paths(gid)
        if len(aid_list) > 0:
            fulldir = image_path.split('/')
            filename = fulldir.pop()
            extension = filename.split('.')[-1]
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
            print("Copying:\n%r\n%r\n%r\n\n" % (image_path, dst_img, (width, height), ))

            annotation = PascalVOC_Markup_Annotation(dst_img, folder, out_img, source=image_uri, **information)
            bbox_list = ibs.get_annot_bboxes(aid_list)
            theta_list = ibs.get_annot_thetas(aid_list)
            for aid, bbox, theta in zip(aid_list, bbox_list, theta_list):
                # Transformation matrix
                R = linalg.rotation_around_bbox_mat3x3(theta, bbox)
                # Get verticies of the annotation polygon
                verts = geometry.verts_from_bbox(bbox, close=True)
                # Rotate and transform vertices
                xyz_pts = geometry.homogonize(np.array(verts).T)
                trans_pts = geometry.unhomogonize(R.dot(xyz_pts))
                new_verts = np.round(trans_pts).astype(np.int).T.tolist()
                x_points = [pt[0] for pt in new_verts]
                y_points = [pt[1] for pt in new_verts]
                xmin = int(min(x_points) * decrease)
                xmax = int(max(x_points) * decrease)
                ymin = int(min(y_points) * decrease)
                ymax = int(max(y_points) * decrease)
                #TODO: Change species_name to getter in IBEISControl once implemented
                #species_name = 'grevys_zebra'
                species_name = ibs.get_annot_species(aid)
                annotation.add_object(species_name, (xmax, xmin, ymax, ymin))
            dst_annot = annotdir + out_name  + '.xml'
            # Write XML
            xml_data = open(dst_annot, 'w')
            xml_data.write(annotation.xml())
            xml_data.close()
            offset += 1
        else:
            print("Skipping:\n%r\n\n" % (image_path, ))


@__injectable
def export_to_hotspotter(ibs):
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
def compute_all_chips(ibs):
    """
    Executes lazy evaluation of all chips
    """
    print('[ibs] compute_all_chips')
    aid_list = ibs.get_valid_aids()
    cid_list = ibs.add_chips(aid_list)
    return cid_list


@__injectable
def compute_all_features(ibs, **kwargs):
    """
    Executes lazy evaluation of all chips and features
    """
    print('[ibs] compute_all_features')
    aid_list = ibs.get_valid_aids(**kwargs)
    cid_list = ibs.get_annot_cids(aid_list, ensure=True)
    fid_list = ibs.add_feats(cid_list)
    return fid_list


@__injectable
def ensure_annotation_data(ibs, aid_list, chips=True, feats=True):
    if chips or feats:
        cid_list = ibs.add_chips(aid_list)
    if feats:
        ibs.add_feats(cid_list)


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
def assert_valid_species(ibs, species_list, iswarning=True):
    if utool.NO_ASSERTS:
        return
    try:
        assert all([species in constants.VALID_SPECIES
                    for species in species_list]), 'invalid species added'
    except AssertionError as ex:
        utool.printex(ex, iswarning=iswarning)
        if not iswarning:
            raise


@__injectable
def assert_singleton_relationship(ibs, alrids_list):
    if utool.NO_ASSERTS:
        return
    try:
        assert all([len(alrids) == 1 for alrids in alrids_list]),\
            'must only have one relationship of a type'
    except AssertionError as ex:
        parent_locals = utool.get_parent_locals()
        utool.printex(ex, 'parent_locals=' + utool.dict_str(parent_locals), key_list=['alrids_list', ])
        raise


@__injectable
def assert_valid_aids(ibs, aid_list):
    if utool.NO_ASSERTS:
        return
    valid_aids = set(ibs.get_valid_aids())
    #invalid_aids = [aid for aid in aid_list if aid not in valid_aids]
    isinvalid_list = [aid not in valid_aids for aid in aid_list]
    assert not any(isinvalid_list), 'invalid aids: %r' % (utool.filter_items(aid_list, isinvalid_list),)
    isinvalid_list = [not isinstance(aid, utool.VALID_INT_TYPES) for aid in aid_list]
    assert not any(isinvalid_list), 'invalidly typed aids: %r' % (utool.filter_items(aid_list, isinvalid_list),)


def assert_images_exist(ibs, gid_list):
    gpath_list = ibs.get_image_paths(gid_list)
    exists_list = list(map(exists, gpath_list))
    bad_gids = utool.filterfalse_items(gid_list, exists_list)
    assert len(bad_gids) == 0, 'some images dont exist'
    print('[check] checked %d images exist' % len(gid_list))


@__injectable
def check_image_consistency(ibs, gid_list):
    # TODO: more consistency checks
    print('check image consistency. len(gid_list)=%r' % len(gid_list))
    assert_images_exist(ibs, gid_list)


def check_image_uuid_consistency(ibs, gid_list):
    """ VERY SLOW """
    gpath_list = ibs.get_image_paths(gid_list)
    guuid_list = ibs.get_image_uuids(gid_list)
    import ibeis.model.preproc.preproc_image as preproc_image
    mark_, end_ = utool.log_progress('checking uuids', len(gpath_list))
    for ix in range(len(gpath_list)):
        mark_(ix)
        gpath = gpath_list[ix]
        guuid = guuid_list[ix]
        # Test params
        param_tup = preproc_image.parse_imageinfo(gpath)
        guuid_target = param_tup[0]
        assert guuid == guuid_target, 'image ix=%d had a bad uuid' % ix
    end_()


@__injectable
def check_annot_consistency(ibs, aid_list):
    # TODO: more consistency checks
    print('check annot consistency. len(aid_list)=%r' % len(aid_list))
    annot_gid_list = ibs.get_annot_gids(aid_list)
    assert_images_exist(ibs, annot_gid_list)
    unique_gids = list(set(annot_gid_list))
    print('num_unique_images=%r / %r' % (len(unique_gids), len(annot_gid_list)))
    cid_list = ibs.get_annot_cids(aid_list, ensure=False)
    cfpath_list = ibs.get_chip_paths(cid_list)
    valid_chip_list = [None if cfpath is None else exists(cfpath) for cfpath in cfpath_list]
    invalid_list = [flag is False for flag in valid_chip_list]
    invalid_cids = utool.filter_items(cid_list, invalid_list)
    if len(invalid_cids) > 0:
        print('found %d inconsistent chips attempting to fix' % len(invalid_cids))
        ibs.delete_chips(invalid_cids, verbose=True)


def check_name_consistency(ibs, nid_list):
    #aids_list = ibs.get_name_aids(nid_list)
    print('check name consistency. len(nid_list)=%r' % len(nid_list))
    lbltype_rowid_list = ibs.get_lblannot_lbltypes_rowids(nid_list)
    individual_lbltype_rowid = ibs.lbltype_ids[constants.INDIVIDUAL_KEY]
    for lbltype_rowid in lbltype_rowid_list:
        assert lbltype_rowid == individual_lbltype_rowid, 'non individual lbltype'


@__injectable
def check_annot_size(ibs):
    aid_list = ibs.get_valid_aids()
    uuid_list = ibs.get_annot_uuids(aid_list)
    desc_list = ibs.get_annot_desc(aid_list)
    kpts_list = ibs.get_annot_kpts(aid_list)
    vert_list = ibs.get_annot_verts(aid_list)
    print('size(aid_list) = ' + utool.byte_str2(utool.get_object_size(aid_list)))
    print('size(vert_list) = ' + utool.byte_str2(utool.get_object_size(vert_list)))
    print('size(uuid_list) = ' + utool.byte_str2(utool.get_object_size(uuid_list)))
    print('size(desc_list) = ' + utool.byte_str2(utool.get_object_size(desc_list)))
    print('size(kpts_list) = ' + utool.byte_str2(utool.get_object_size(kpts_list)))


@__injectable
def check_consistency(ibs, embed=False):
    print('[ibsfuncs] Checking consistency')
    gid_list = ibs.get_valid_gids()
    aid_list = ibs.get_valid_aids()
    nid_list = ibs.get_valid_nids()
    check_image_consistency(ibs, gid_list)
    check_annot_consistency(ibs, aid_list)
    check_name_consistency(ibs, nid_list)
    # Very slow check
    check_image_uuid_consistency(ibs, gid_list)
    if embed:
        utool.embed()
    print('[ibsfuncs] Finshed consistency check')


@__injectable
def fix_exif_data(ibs, gid_list):
    import vtool.exif as exif
    from PIL import Image
    import utool
    gpath_list = ibs.get_image_paths(gid_list)
    mark_, end_ = utool.log_progress('checking exif: ', len(gpath_list))
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

    latlon_list_ = utool.filter_items(latlon_list, haslatlon_list)
    gid_list_    = utool.filter_items(gid_list, haslatlon_list)

    gps_list = ibs.get_image_gps(gid_list_)
    needsupdate_list = [gps == (-1, -1) for gps in gps_list]

    print('%d / %d need gps update' % (sum(needsupdate_list),
                                       len(needsupdate_list)))

    if sum(needsupdate_list)  > 0:
        assert sum(needsupdate_list) == len(needsupdate_list), 'safety. remove and evaluate if hit'
        #ibs.set_image_enctext(gid_list_, ['HASGPS'] * len(gid_list_))
        latlon_list__ = utool.filter_items(latlon_list_, needsupdate_list)
        gid_list__ = utool.filter_items(gid_list_, needsupdate_list)
        ibs.set_image_gps(gid_list__, latlon_list__)


def check_exif_data(ibs, gid_list):
    import vtool.exif as exif
    from PIL import Image
    import utool
    gpath_list = ibs.get_image_paths(gid_list)
    mark_, end_ = utool.log_progress('checking exif: ', len(gpath_list))
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

    key2_freq = utool.ddict(lambda: 0)
    num_tags_list = []
    for exif_dict in exif_dict_list:
        exif_dict2 = exif.make_exif_dict_human_readable(exif_dict)
        num_tags_list.append(len(exif_dict))
        for key in exif_dict2.keys():
            key2_freq[key] += 1

    utool.print_stats(num_tags_list, 'num tags per image')

    print('tag frequency')
    print(utool.dict_str(key2_freq))

    end_()


@__injectable
def delete_all_features(ibs):
    print('[ibs] delete_all_features')
    all_fids = ibs._get_all_fids()
    ibs.delete_features(all_fids)
    print('[ibs] finished delete_all_features')


@__injectable
def delete_all_annotations(ibs):
    print('[ibs] delete_all_annotations')
    all_aids = ibs._get_all_aids()
    ibs.delete_annots(all_aids)
    print('[ibs] finished delete_all_annotations')


@__injectable
def delete_all_chips(ibs):
    print('[ibs] delete_all_chips')
    all_cids = ibs._get_all_cids()
    ibs.delete_chips(all_cids)
    print('[ibs] finished delete_all_chips')


@__injectable
def delete_all_encounters(ibs):
    print('[ibs] delete_all_encounters')
    all_eids = ibs._get_all_eids()
    ibs.delete_encounters(all_eids)
    print('[ibs] finished delete_all_encounters')


@__injectable
def vd(ibs):
    utool.view_directory(ibs.get_dbdir())


@__injectable
def get_annot_desc_cache(ibs, aids):
    """ When you have a list with duplicates and you dont want to copy data
    creates a reference to each data object idnexed by a dict """
    unique_aids = list(set(aids))
    unique_desc = ibs.get_annot_desc(unique_aids)
    desc_cache = dict(list(zip(unique_aids, unique_desc)))
    return desc_cache


@__injectable
def get_annot_is_hard(ibs, aid_list):
    notes_list = ibs.get_annot_notes(aid_list)
    is_hard_list = ['hard' in notes.lower().split() for (notes)
                    in notes_list]
    return is_hard_list


@__injectable
def get_annot_bbox_area(ibs, aid_list):
    bbox_list = ibs.get_annot_bboxes(aid_list)
    area_list = [bbox[2] * bbox[3] for bbox in bbox_list]
    return area_list


@__injectable
def localize_images(ibs, gid_list_=None):
    """
    Moves the images into the ibeis image cache.
    Images are renamed to img_uuid.ext
    """
    if gid_list_ is None:
        gid_list_  = ibs.get_valid_gids()
    isnone_list = [gid is None for gid in gid_list_]
    gid_list = utool.filterfalse_items(gid_list_, isnone_list)
    gpath_list = ibs.get_image_paths(gid_list)
    guuid_list = ibs.get_image_uuids(gid_list)
    gext_list  = ibs.get_image_exts(gid_list)
    # Build list of image names based on uuid in the ibeis imgdir
    guuid_strs = (str(guuid) for guuid in guuid_list)
    loc_gname_list = [guuid + ext for (guuid, ext) in zip(guuid_strs, gext_list)]
    loc_gpath_list = [join(ibs.imgdir, gname) for gname in loc_gname_list]
    utool.copy_list(gpath_list, loc_gpath_list, lbl='Localizing Images: ')
    ibs.set_image_uris(gid_list, loc_gname_list)
    assert all(map(exists, loc_gpath_list)), 'not all images copied'


@__injectable
def rebase_images(ibs, new_path, gid_list=None):
    """
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
    new_gpath_list = list(map(utool.unixpath, new_gpath_list))
    #orig_exists = map(exists, gpath_list)
    new_exists = list(map(exists, new_gpath_list))
    assert all(new_exists), 'some rebased images do not exist'
    ibs.set_image_uris(gid_list, new_gpath_list)
    assert  'not all images copied'


@__injectable
def delete_invalid_nids(ibs):
    """ Removes names that have no Rois from the database """
    invalid_nids = ibs.get_invalid_nids()
    ibs.delete_names(invalid_nids)


@__injectable
def delete_invalid_eids(ibs):
    """ Removes encounters without images """
    eid_list = ibs.get_valid_eids(min_num_gids=0)
    nGids_list = ibs.get_encounter_num_gids(eid_list)
    is_invalid = [nGids == 0 for nGids in nGids_list]
    invalid_eids = utool.filter_items(eid_list, is_invalid)
    ibs.delete_encounters(invalid_eids)


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
def set_annot_names_to_next_name(ibs, aid_list):
    next_name = ibs.make_next_name()
    ibs.set_annot_names(aid_list, [next_name] * len(aid_list))


@__injectable
def _overwrite_annot_species_to_plains(ibs, aid_list):
    species_list = [constants.Species.ZEB_PLAIN] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


@__injectable
def _overwrite_annot_species_to_grevys(ibs, aid_list):
    species_list = [constants.Species.ZEB_GREVY] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


@__injectable
def _overwrite_annot_species_to_giraffe(ibs, aid_list):
    species_list = [constants.Species.GIR] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


@__injectable
def _overwrite_all_annot_species_to(ibs, species):
    """ THIS OVERWRITES A LOT OF INFO """
    assert species in constants.VALID_SPECIES, repr(species) + 'is not in ' + repr(constants.VALID_SPECIES)
    aid_list = ibs.get_valid_aids()
    species_list = [species] * len(aid_list)
    ibs.set_annot_species(aid_list, species_list)


@__injectable
def get_match_truth(ibs, aid1, aid2):
    nid1, nid2 = ibs.get_annot_nids((aid1, aid2))
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


def unflat_map(method, unflat_rowids, **kwargs):
    """
    Uses an ibeis lookup function with a non-flat rowid list.
    In essence this is equivilent to map(method, unflat_rowids).
    The utility of this function is that it only calls method once.
    This is more efficient for calls that can take a list of inputs
    """
    #utool.assert_unflat_level(unflat_rowids, level=1, basetype=(int, uuid.UUID))
    # First flatten the list, and remember the original dimensions
    flat_rowids, reverse_list = utool.invertable_flatten(unflat_rowids)
    # Then preform the lookup / implicit mapping
    flat_vals = method(flat_rowids, **kwargs)
    # Then unflatten the results to the original input dimensions
    unflat_vals = utool.util_list.unflatten(flat_vals, reverse_list)
    return unflat_vals


def unflat_multimap(method_list, unflat_rowids, **kwargs):
    """ unflat_map, but allows multiple methods
    """
    # First flatten the list, and remember the original dimensions
    flat_rowids, reverse_list = utool.invertable_flatten(unflat_rowids)
    # Then preform the lookup / implicit mapping
    flat_vals_list = [method(flat_rowids, **kwargs) for method in method_list]
    # Then unflatten the results to the original input dimensions
    unflat_vals_list = [utool.util_list.unflatten(flat_vals, reverse_list)
                        for flat_vals in flat_vals_list]
    return unflat_vals_list

# TODO: Depricate the lookup names
unflat_lookup = unflat_map


def _make_unflat_getter_func(flat_getter):
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
        flat_rowids, reverse_list = utool.invertable_flatten(unflat_rowids)
        # Then preform the lookup
        flat_vals = func(self, flat_rowids, *args, **kwargs)
        # Then unflatten the list
        unflat_vals = utool.util_list.unflatten(flat_vals, reverse_list)
        return unflat_vals
    set_funcname(unflat_getter, funcname.replace('get_', 'get_unflat_'))
    return unflat_getter


def delete_ibeis_database(dbdir):
    _ibsdb = join(dbdir, constants.PATH_NAMES._ibsdb)
    print('[ibsfuncs] DELETEING: _ibsdb=%r' % _ibsdb)
    if exists(_ibsdb):
        utool.delete(_ibsdb)


def assert_valid_names(name_list):
    """ Asserts that user specified names do not conflict with
    the standard unknown name """
    if utool.NO_ASSERTS:
        return
    def isconflict(name, other):
        return name.startswith(other) and len(name) > len(other)
    valid_namecheck = [not isconflict(name, constants.UNKNOWN) for name in name_list]
    assert all(valid_namecheck), ('A name conflicts with UKNONWN Name. -- '
                                  'cannot start a name with four underscores')


def assert_lblannot_rowids_are_type(ibs, lblannot_rowid_list, valid_lbltype_rowid):
    if utool.NO_ASSERTS:
        return
    lbltype_rowid_list = ibs.get_lblannot_lbltypes_rowids(lblannot_rowid_list)
    try:
        # HACK: the unknown_lblannot_rowid will have a None type
        # the unknown lblannot_rowid should be handled more gracefully
        # this should just check the first condition (get rid of the or)
        assert len(lbltype_rowid_list) == len(lbltype_rowid_list), 'lens dont match'
        validtype_list = [
            (lbltype_rowid == valid_lbltype_rowid) or
            (lbltype_rowid is None and lblannot_rowid == constants.UNKNOWN_LBLANNOT_ROWID)
            for lbltype_rowid, lblannot_rowid in
            zip(lbltype_rowid_list, lblannot_rowid_list)]
        assert all(validtype_list), 'not all types match valid type'
    except AssertionError as ex:
        tup_list = list(map(str, list(zip(lbltype_rowid_list, lblannot_rowid_list))))
        print('[!!!] (lbltype_rowid, lblannot_rowid) = : ' + utool.indentjoin(tup_list))
        print('[!!!] valid_lbltype_rowid: %r' % (valid_lbltype_rowid,))
        utool.printex(ex, 'not all types match valid type',
                      key_list=['valid_lbltype_rowid'])
        raise


def ensure_unix_gpaths(gpath_list):
    """
    Asserts that all paths are given with forward slashes.
    If not it fixes them
    """
    #if utool.NO_ASSERTS:
    #    return
    try:
        msg = ('gpath_list must be in unix format (no backslashes).'
               'Failed on %d-th gpath=%r')
        for count, gpath in enumerate(gpath_list):
            assert gpath.find('\\') == -1, (msg % (count, gpath))
    except AssertionError as ex:
        utool.printex(ex, iswarning=True)
        gpath_list = list(map(utool.unixpath, gpath_list))
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
    gpath_list = utool.list_images(img_dir,
                                   fullpath=fullpath,
                                   recursive=recursive,
                                   ignore_list=ignore_list)
    return gpath_list


def make_annotation_uuids(image_uuid_list, bbox_list, theta_list, deterministic=True):
    augment_uuid = utool.util_hash.augment_uuid
    random_uuid = utool.util_hash.random_uuid
    try:
        # Check to make sure bbox input is a tuple-list, not a list-list
        if len(bbox_list) > 0:
            try:
                assert isinstance(bbox_list[0], tuple), 'Bounding boxes must be tuples of ints!'
                assert isinstance(bbox_list[0][0], int), 'Bounding boxes must be tuples of ints!'
            except AssertionError as ex:
                utool.printex(ex)
                print('bbox_list = %r' % (bbox_list,))
                raise
        annotation_uuid_list = [augment_uuid(img_uuid, bbox, theta)
                                for img_uuid, bbox, theta
                                in zip(image_uuid_list, bbox_list, theta_list)]
        if not deterministic:
            # Augment determenistic uuid with a random uuid to ensure randomness
            # (this should be ensured in all hardward situations)
            annotation_uuid_list = [augment_uuid(random_uuid(), _uuid)
                                    for _uuid in annotation_uuid_list]
    except Exception as ex:
        utool.printex(ex, 'Error building annotation_uuids', '[add_annot]',
                      key_list=['image_uuid_list'])
        raise
    return annotation_uuid_list


def get_species_dbs(species_prefix):
    ibs_dblist = sysres.get_ibsdb_list()
    isvalid_list = [split(path)[1].startswith(species_prefix) for path in ibs_dblist]
    return utool.filter_items(ibs_dblist, isvalid_list)


def merge_species_databases(species_prefix):
    """ Build a merged database """
    from ibeis.control import IBEISControl
    print('[ibsfuncs] Merging species with prefix: %r' % species_prefix)
    utool.util_parallel.ensure_pool(warn=False)
    with utool.Indenter('    '):
        # Build / get target database
        all_db = '__ALL_' + species_prefix + '_'
        all_dbdir = sysres.db_to_dbdir(all_db, allow_newdir=True)
        ibs_target = IBEISControl.IBEISController(all_dbdir)
        # Build list of databases to merge
        species_dbdir_list = get_species_dbs(species_prefix)
        ibs_source_list = []
        for dbdir in species_dbdir_list:
            ibs_source = IBEISControl.IBEISController(dbdir)
            ibs_source_list.append(ibs_source)
    print('[ibsfuncs] Destination database: %r' % all_db)
    print('[ibsfuncs] Source databases:' +
          utool.indentjoin(species_dbdir_list, '\n *   '))
    #Merge the databases into ibs_target
    merge_databases(ibs_target, ibs_source_list)
    return ibs_target


def merge_databases(ibs_target, ibs_source_list):
    """ Merges a list of databases into a target
    This is OLD. use export_subset instead
    """
    raise AssertionError('Use transfer_subset instead')

    def merge_images(ibs_target, ibs_source):
        """ merge image helper """
        gid_list1   = ibs_source.get_valid_gids()
        uuid_list1  = ibs_source.get_image_uuids(gid_list1)
        gpath_list1 = ibs_source.get_image_paths(gid_list1)
        reviewed_list1 = ibs_source.get_image_reviewed(gid_list1)
        # Add images to target
        ibs_target.add_images(gpath_list1)
        # Merge properties
        gid_list2  = ibs_target.get_image_gids_from_uuid(uuid_list1)
        ibs_target.set_image_reviewed(gid_list2, reviewed_list1)

    def merge_annotations(ibs_target, ibs_source):
        """ merge annotations helper """
        aid_list1   = ibs_source.get_valid_aids()
        uuid_list1  = ibs_source.get_annot_uuids(aid_list1)
        # Get the images in target_db
        gid_list1   = ibs_source.get_annot_gids(aid_list1)
        bbox_list1  = ibs_source.get_annot_bboxes(aid_list1)
        theta_list1 = ibs_source.get_annot_thetas(aid_list1)
        name_list1  = ibs_source.get_annot_names(aid_list1)
        notes_list1 = ibs_source.get_annot_notes(aid_list1)

        image_uuid_list1 = ibs_source.get_image_uuids(gid_list1)
        gid_list2  = ibs_target.get_image_gids_from_uuid(image_uuid_list1)
        image_uuid_list2 = ibs_target.get_image_uuids(gid_list2)
        # Assert that the image uuids have not changed
        assert image_uuid_list1 == image_uuid_list2, 'error merging annotation image uuids'
        aid_list2 = ibs_target.add_annots(gid_list2,
                                          bbox_list1,
                                          theta_list=theta_list1,
                                          name_list=name_list1,
                                          notes_list=notes_list1)
        uuid_list2 = ibs_target.get_annot_uuids(aid_list2)
        assert uuid_list2 == uuid_list1, 'error merging annotation uuids'

    # Do the merging
    for ibs_source in ibs_source_list:
        try:
            print('Merging ' + ibs_source.get_dbname() +
                  ' into ' + ibs_target.get_dbname())
            merge_images(ibs_target, ibs_source)
            merge_annotations(ibs_target, ibs_source)
        except Exception as ex:
            utool.printex(ex, 'error merging ' + ibs_source.get_dbname() +
                          ' into ' + ibs_target.get_dbname())


@__injectable
@utool.time_func
#@profile
def delete_non_exemplars(ibs):
    gid_list = ibs.get_valid_gids
    aids_list = ibs.get_image_aids(gid_list)
    flags_list = unflat_map(ibs.get_annot_exemplar_flag, aids_list)
    delete_gid_flag_list = [not any(flags) for flags in flags_list]
    delete_gid_list = utool.filter_items(gid_list, delete_gid_flag_list)
    ibs.delete_images(delete_gid_list)
    delete_invalid_eids(ibs)
    delete_invalid_nids(ibs)


@__injectable
@utool.time_func
#@profile
def update_exemplar_encounter(ibs):
    # FIXME SLOW
    exemplar_eid = ibs.get_encounter_eids_from_text(constants.EXEMPLAR_ENCTEXT)
    #ibs.delete_encounters(exemplar_eid)
    ibs.unrelate_encounter_from_images(exemplar_eid)
    #aid_list = ibs.get_valid_aids(is_exemplar=True)
    #gid_list = utool.unique_ordered(ibs.get_annot_gids(aid_list))
    gid_list = list(set(_get_exemplar_gids(ibs)))
    #ibs.set_image_enctext(gid_list, [constants.EXEMPLAR_ENCTEXT] * len(gid_list))
    ibs.set_image_eids(gid_list, [exemplar_eid] * len(gid_list))


@__injectable
@utool.time_func
#@profile
def update_reviewed_unreviewed_image_encounter(ibs):
    # FIXME SLOW
    unreviewed_eid = ibs.get_encounter_eids_from_text(constants.UNREVIEWED_IMAGE_ENCTEXT)
    reviewed_eid = ibs.get_encounter_eids_from_text(constants.REVIEWED_IMAGE_ENCTEXT)
    #ibs.delete_encounters(eid)
    ibs.unrelate_encounter_from_images(unreviewed_eid)
    ibs.unrelate_encounter_from_images(reviewed_eid)
    #gid_list = ibs.get_valid_gids(reviewed=False)
    #ibs.set_image_enctext(gid_list, [constants.UNREVIEWED_IMAGE_ENCTEXT] * len(gid_list))
    unreviewed_gids = _get_unreviewed_gids(ibs)  # hack
    reviewed_gids   = _get_reviewed_gids(ibs)  # hack
    ibs.set_image_eids(unreviewed_gids, [unreviewed_eid] * len(unreviewed_gids))
    ibs.set_image_eids(reviewed_gids, [reviewed_eid] * len(reviewed_gids))


#@__injectable
#@utool.time_func
#@profile
#def update_reviewed_image_encounter(ibs):
#    # FIXME SLOW
#    #ibs.delete_encounters(eid)
#    ibs.unrelate_encounter_from_images(eid)
#    #gid_list = ibs.get_valid_gids(reviewed=True)
#    gid_list = _get_reviewed_gids(ibs)  # hack
#    #ibs.set_image_enctext(gid_list, [constants.REVIEWED_IMAGE_ENCTEXT] * len(gid_list))
#    ibs.set_image_eids(gid_list, [eid] * len(gid_list))


@__injectable
@utool.time_func
#@profile
def update_all_image_encounter(ibs):
    # FIXME SLOW
    eid = ibs.get_encounter_eids_from_text(constants.ALL_IMAGE_ENCTEXT)
    #ibs.delete_encounters(eid)
    gid_list = ibs.get_valid_gids()
    #ibs.set_image_enctext(gid_list, [constants.ALL_IMAGE_ENCTEXT] * len(gid_list))
    ibs.set_image_eids(gid_list, [eid] * len(gid_list))


@__injectable(False)
@utool.time_func
#@profile
def update_special_encounters(ibs):
    # FIXME SLOW
    ibs.update_exemplar_encounter()
    ibs.update_reviewed_unreviewed_image_encounter()
    ibs.update_all_image_encounter()


def _get_unreviewed_gids(ibs):
    # hack
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {IMAGE_TABLE}
        WHERE
        image_toggle_reviewed=0
        '''.format(**constants.__dict__))
    return gid_list


def _get_reviewed_gids(ibs):
    # hack
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {IMAGE_TABLE}
        WHERE
        image_toggle_reviewed=1
        '''.format(**constants.__dict__))
    return gid_list


def _get_gids_in_eid(ibs, eid):
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {EG_RELATION_TABLE}
        WHERE
            encounter_rowid==?
        '''.format(**constants.__dict__),
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
        '''.format(**constants.__dict__),
        params=(eid,))
    return gid_list


def _get_exemplar_gids(ibs):
    gid_list = ibs.db.executeone(
        '''
        SELECT image_rowid
        FROM {ANNOTATION_TABLE}
        WHERE annot_exemplar_flag=1
        '''.format(**constants.__dict__))
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
    return title


@__injectable
def print_stats(ibs):
    from ibeis.dev import dbinfo
    dbinfo.dbstats(ibs)


@__injectable
def print_dbinfo(ibs):
    from ibeis.dev import dbinfo
    dbinfo.get_dbinfo(ibs)


@__injectable
def get_infostr(ibs):
    """ Returns printable database information """
    dbname = ibs.get_dbname()
    workdir = utool.unixpath(ibs.get_workdir())
    num_images = ibs.get_num_images()
    num_annotations = ibs.get_num_annotations()
    num_names = ibs.get_num_names()
    infostr = '''
    workdir = %r
    dbname = %r
    num_images = %r
    num_annotations = %r
    num_names = %r
    ''' % (workdir, dbname, num_images, num_annotations, num_names)
    return infostr


@__injectable(False)
def print_annotation_table(ibs):
    """ Dumps annotation table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.ANNOTATION_TABLE, exclude_columns=['annotation_uuid', 'annotation_verts']))


@__injectable(False)
def print_chip_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.CHIP_TABLE))


@__injectable(False)
def print_feat_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.FEATURE_TABLE, exclude_columns=[
        'feature_keypoints', 'feature_sifts']))


@__injectable(False)
def print_image_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.IMAGE_TABLE))
    #, exclude_columns=['image_rowid']))


@__injectable(False)
def print_lblannot_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.LBLANNOT_TABLE))


@__injectable(False)
def print_alr_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.AL_RELATION_TABLE))


@__injectable(False)
def print_config_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.CONFIG_TABLE))


@__injectable(False)
def print_encounter_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.ENCOUNTER_TABLE))


@__injectable(False)
def print_egpairs_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.EG_RELATION_TABLE))


@__injectable(False)
def print_tables(ibs, exclude_columns=None, exclude_tables=None):
    if exclude_columns is None:
        exclude_columns = ['annot_uuid', 'lblannot_uuid', 'annot_verts', 'feature_keypoints',
                           'feature_sifts', 'image_uuid', 'image_uri']
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


#@getter_1to1
@__injectable
def is_aid_unknown(ibs, aid_list):
    nid_list = ibs.get_annot_nids(aid_list)
    return ibs.is_nid_unknown(nid_list)


def make_enctext_list(eid_list, enc_cfgstr):
    # DEPRICATE
    enctext_list = [str(eid) + enc_cfgstr for eid in eid_list]
    return enctext_list


@__injectable
def make_next_name(ibs, num=None):
    """ Creates a number of names which are not in the database, but does not
    add them """
    num_names = ibs.get_num_names()
    userid = utool.get_user_name()
    timestamp = utool.get_timestamp('tag')
    name_prefix = timestamp + '_TMP_' + userid + '_'
    if num is None:
        next_name = name_prefix + '%04d' % num_names
        return next_name
    else:
        next_names = [name_prefix + '%04d' % (num_names + x) for x in range(num)]
        return next_names


@__injectable
def prune_exemplars(ibs):
    nid_list = ibs.get_valid_nids()
    aids_list = ibs.get_name_exemplar_aids(nid_list)
    MAX_EXEMPLAR = 6
    problem_aids = [np.array(aids) for aids in aids_list if len(aids) > MAX_EXEMPLAR]
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
    small_aids_list = [aids[sortx][:-MAX_EXEMPLAR] for aids, sortx in zip(problem_aids, problem_sortx)]
    small_aids = utool.flatten(small_aids_list)
    ibs.set_annot_exemplar_flag(small_aids, [False] * len(small_aids))


@__injectable
def delete_cachedir(ibs, removefeat=True):
    """
    Deletes the cache directory in the database directory.

    Also removes features from the sql controller if removefeat is True
    """
    print('[ibs] delete_cachedir')
    cachedir = ibs.get_cachedir()
    print('[ibs] cachedir=%r' % cachedir)
    utool.delete(cachedir)
    print('[ibs] finished delete cachedir')
    # TODO: features really need to not be in SQL or in a separate SQLDB
    if removefeat:
        ibs.delete_all_features()


def draw_thumb_helper(tup):
    thumb_path, thumbsize, gpath, bbox_list, theta_list = tup
    img = gtool.imread(gpath)  # time consuming
    (gh, gw) = img.shape[0:2]
    img_size = (gw, gh)
    max_dsize = (thumbsize, thumbsize)
    dsize, sx, sy = gtool.resized_clamped_thumb_dims(img_size, max_dsize)
    new_verts_list = list(gtool.scale_bbox_to_verts_gen(bbox_list, theta_list, sx, sy))
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
def preprocess_image_thumbs(ibs, gid_list=None, use_cache=True, chunksize=8, **kwargs):
    """ Computes thumbs of images in parallel based on kwargs """
    if gid_list is None:
        gid_list = ibs.get_valid_gids(**kwargs)
    thumbsize = 128
    thumbpath_list = ibs.get_image_thumbpath(gid_list, thumbsize=thumbsize)
    #use_cache = False
    if use_cache:
        exists_list = list(map(exists, thumbpath_list))
        gid_list_ = utool.filterfalse_items(gid_list, exists_list)
        thumbpath_list_ = utool.filterfalse_items(thumbpath_list, exists_list)
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
    gen = utool.generate(draw_thumb_helper, args_list, **genkw)
    #for output in gen:
    #    #with utool.Timer('gentime'):
    #    gtool.imwrite(output[0], output[1])
    try:
        while True:
            six.next(gen)
    except StopIteration:
        pass


@__injectable
def compute_all_thumbs(ibs, **kwargs):
    preprocess_image_thumbs(ibs, **kwargs)


def group_annots_by_known_names_nochecks(ibs, aid_list):
    nid_list = ibs.get_annot_nids(aid_list)
    nid2_aids = utool.group_items(aid_list, nid_list)
    return list(nid2_aids.values())


def group_annots_by_known_names(ibs, aid_list, checks=True):
    """
    #>>> import ibeis

    Example:
        >>> from ibeis import ibsfuncst *  # NOQA
        >>> ibs = ibeis.opendb(db='testdb1') #doctest: +ELLIPSIS
        [main]...
        >>> aid_list = ibs.get_valid_aids()
        >>> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        >>> known_aids_list, unknown_aids = group_annots_by_known_names(ibs, aid_list)
        >>> print(known_aids_list)
        [[2, 3], [5, 6], [7], [8], [10], [12], [13]]
        >>> print(unknown_aids)
        [11, 9, 4, 1]
    """
    nid_list = ibs.get_annot_nids(aid_list)
    nid2_aids = utool.group_items(aid_list, nid_list)
    aid_gen = lambda: six.itervalues(nid2_aids)
    isunknown_list = ibs.is_nid_unknown(six.iterkeys(nid2_aids))
    known_aids_list = list(utool.ifilterfalse_items(aid_gen(), isunknown_list))
    unknown_aids = list(utool.iflatten(utool.ifilter_items(aid_gen(), isunknown_list)))
    if __debug__:
        # http://stackoverflow.com/questions/482014/how-would-you-do-the-equivalent-of-preprocessor-directives-in-python
        nidgroup_list = unflat_map(ibs.get_annot_nids, known_aids_list)
        for nidgroup in nidgroup_list:
            assert utool.list_allsame(nidgroup), 'bad name grouping'
    return known_aids_list, unknown_aids


@__injectable
def get_annot_rowid_hashid(ibs, aid_list, label='_AIDS'):
    aids_hashid = utool.hashstr_arr(aid_list, label)
    return aids_hashid


@__injectable
def get_annot_uuid_hashid(ibs, aid_list, label='_UUIDS'):
    uuid_list    = ibs.get_annot_uuids(aid_list)
    uuui_hashid  = utool.hashstr_arr(uuid_list, label)
    return uuui_hashid


@__injectable
def get_annot_groundfalse_sample(ibs, aid_list, per_name=1):
    """
    >>> per_name = 1
    """
    # Get valid names
    valid_aids = ibs.get_valid_aids()
    valid_nids = ibs.get_annot_nids(valid_aids)
    nid2_aids = utool.group_items(valid_aids, valid_nids)
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
    nid_list = ibs.get_annot_nids(aid_list)
    def _sample(nid_):
        aids_iter = (aids for nid, aids in six.iteritems(nid2_aids) if nid != nid_)
        sample_gf_aids = np.hstack([np.random.choice(aids, per_name, replace=False) for aids in aids_iter])
        return sample_gf_aids.tolist()
    gf_aids_list = [_sample(nid_) for nid_ in nid_list]
    return gf_aids_list


@__injectable
def get_annot_groundtruth_sample(ibs, aid_list, per_name=1):
    """
    Example:
        >>> from ibeis.all_imports import *  # NOQA
        >>> ibs = ibeis.test_main(db='testdb1')
        >>> per_name = 1
        >>> aid_list = ibs.get_valid_aids()
    """
    all_trues_list = ibs.get_annot_groundtruth(aid_list, noself=True, is_exemplar=True)
    def random_choice(aids):
        size = min(len(aids), per_name)
        return np.random.choice(aids, size, replace=False).tolist()
    sample_trues_list = [random_choice(aids) if len(aids) > 0 else [] for aids in all_trues_list]
    return sample_trues_list


@__injectable
def get_aids_with_groundtruth(ibs):
    """ returns aids with valid groundtruth """
    valid_aids = ibs.get_valid_aids()
    has_gt_list = ibs.get_annot_has_groundtruth(valid_aids)
    hasgt_aids = utool.filter_items(valid_aids, has_gt_list)
    return hasgt_aids


@__injectable
def export_encounters(ibs, eid_list, new_dbdir=None):
    gid_list = list(set(utool.flatten(ibs.get_encounter_gids(eid_list))))
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
    from ibeis.io import export_subset
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
    utool.write_to(ibs.get_dbnotes_fpath(), notes)


@__injectable
def get_dbnotes_fpath(ibs, ensure=False):
    notes_fpath = join(ibs.get_ibsdir(), 'dbnotes.txt')
    if ensure and not exists(ibs.get_dbnotes_fpath()):
        ibs.set_dbnotes('None')
    return notes_fpath


@__injectable
def get_dbnotes(ibs):
    """ sets notes for an entire database """
    notes = utool.read_from(ibs.get_dbnotes_fpath(), strict=False)
    if notes is None:
        ibs.set_dbnotes('None')
        notes = utool.read_from(ibs.get_dbnotes_fpath())
    return notes


@__injectable
def annotstr(ibs, aid):
    return 'aid=%d' % aid
