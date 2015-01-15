"""
Functions for images and encoutners that will be injected into an
IBEISController instance.


CommandLine:
    # Autogenerate Encounter Functions
    # key should be the table name
    # the write flag makes a file, but dont use that
    python -m ibeis.control.template_generator --key featweight
    python -m ibeis.control.template_generator --key encounter

"""
from __future__ import absolute_import, division, print_function
import six  # NOQA
import functools
import vtool as vt
import uuid
from ibeis import constants as const
from ibeis.control.accessor_decors import (ider, adder, getter_1to1, getter_1toM, deleter, setter)
import utool as ut
from os.path import join, exists
from ibeis import ibsfuncs
import numpy as np
from ibeis.control.controller_inject import make_ibs_register_decorator
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[manual_image]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


ENCOUNTER_END_TIME_POSIX   = 'encounter_end_time_posix'
ENCOUNTER_GPS_LAT          = 'encounter_gps_lat'
ENCOUNTER_GPS_LON          = 'encounter_gps_lon'
ENCOUNTER_NOTE             = 'encounter_note'
ENCOUNTER_PROCESSED_FLAG   = 'encounter_processed_flag'
ENCOUNTER_ROWID            = 'encounter_rowid'
ENCOUNTER_SHIPPED_FLAG     = 'encounter_shipped_flag'
ENCOUNTER_START_TIME_POSIX = 'encounter_start_time_posix'

ENCOUNTER_SMART_WAYPOINT_ID = 'encounter_smart_waypoint_id'
ENCOUNTER_SMART_XML_FNAME   = 'encounter_smart_xml_fname'


@register_ibs_method
@ider
def _get_all_gids(ibs):
    """
    Returns:
        list_ (list):  all unfiltered gids (image rowids) """
    all_gids = ibs.db.get_all_rowids(const.IMAGE_TABLE)
    return all_gids


@register_ibs_method
@ider
def _get_all_encounter_rowids(ibs):
    """
    Returns:
        list_ (list):  all unfiltered eids (encounter rowids)
    """
    all_eids = ibs.db.get_all_rowids(const.ENCOUNTER_TABLE)
    return all_eids


@register_ibs_method
@ider
def _get_all_eids(ibs):
    """ alias """
    return _get_all_encounter_rowids(ibs)


@register_ibs_method
@ider
def get_valid_gids(ibs, eid=None, require_unixtime=False, reviewed=None):
    if eid is None:
        gid_list = ibs._get_all_gids()
    else:
        gid_list = ibs.get_encounter_gids(eid)
    if require_unixtime:
        # Remove images without timestamps
        unixtime_list = ibs.get_image_unixtime(gid_list)
        isvalid_list = [unixtime != -1 for unixtime in unixtime_list]
        gid_list = ut.filter_items(gid_list, isvalid_list)
    if reviewed is not None:
        reviewed_list = ibs.get_image_reviewed(gid_list)
        isvalid_list = [reviewed == flag for flag in reviewed_list]
        gid_list = ut.filter_items(gid_list, isvalid_list)
    return gid_list


@register_ibs_method
@ider
def get_valid_eids(ibs, min_num_gids=0, processed=None, shipped=None):
    """
    Returns:
        list_ (list):  list of all encounter ids """
    eid_list = ibs._get_all_eids()
    if min_num_gids > 0:
        num_gids_list = ibs.get_encounter_num_gids(eid_list)
        flag_list = [num_gids >= min_num_gids for num_gids in num_gids_list]
        eid_list  = ut.filter_items(eid_list, flag_list)
    if processed is not None:
        flag_list = ibs.get_encounter_processed_flags(eid_list)
        isvalid_list = [ flag == 1 if processed else flag == 0 for flag in flag_list]
        eid_list  = ut.filter_items(eid_list, isvalid_list)
    if shipped is not None:
        flag_list = ibs.get_encounter_shipped_flags(eid_list)
        isvalid_list = [ flag == 1 if shipped else flag == 0 for flag in flag_list]
        eid_list  = ut.filter_items(eid_list, isvalid_list)

    return eid_list


@register_ibs_method
def get_num_images(ibs, **kwargs):
    """ Number of valid images """
    gid_list = ibs.get_valid_gids(**kwargs)
    return len(gid_list)


@register_ibs_method
@adder
def add_images(ibs, gpath_list, params_list=None, as_annots=False, auto_localize=None):
    """
    Adds a list of image paths to the database.

    Initially we set the image_uri to exactely the given gpath.
    Later we change the uri, but keeping it the same here lets
    us process images asychronously.

    Args:
        gpath_list (list): list of image paths to add
        params_list (list): metadata list for corresponding images that can either be
            specified outright or can be parsed from the image data directly if None
        as_annots (bool): if True, an annotation is automatically added for the entire
            image
        auto_localize (bool): if None uses the default specified in ibs.cfg

    Returns:
        gid_list (list of rowids): gids are image rowids

    Examples:
        >>> from ibeis.all_imports import *  # NOQA  # doctest.SKIP
        >>> gpath_list = grabdata.get_test_gpaths(ndata=7) + ['doesnotexist.jpg']
        >>> ibs.add_images(gpath_list)
    """
    from ibeis.model.preproc import preproc_image
    print('[ibs] add_images')
    print('[ibs] len(gpath_list) = %d' % len(gpath_list))
    #print('[ibs] gpath_list = %r' % (gpath_list,))
    # Processing an image might fail, yeilding a None instead of a tup
    gpath_list = ibsfuncs.ensure_unix_gpaths(gpath_list)
    if params_list is None:
        # Create param_iter
        params_list  = list(preproc_image.add_images_params_gen(gpath_list))
    # Error reporting
    print('\n'.join(
        [' ! Failed reading gpath=%r' % (gpath,) for (gpath, params_)
         in zip(gpath_list, params_list) if not params_]))
    # Add any unadded images
    colnames = ('image_uuid', 'image_uri', 'image_original_name',
                'image_ext', 'image_width', 'image_height',
                'image_time_posix', 'image_gps_lat',
                'image_gps_lon', 'image_note',)
    # <DEBUG>
    if ut.VERBOSE:
        uuid_list = [None if params_ is None else params_[0] for params_ in params_list]
        gid_list_ = ibs.get_image_gids_from_uuid(uuid_list)
        valid_gids = ibs.get_valid_gids()
        valid_uuids = ibs.get_image_uuids(valid_gids)
        print('[preadd] uuid / gid_ = ' + ut.indentjoin(zip(uuid_list, gid_list_)))
        print('[preadd] valid uuid / gid = ' + ut.indentjoin(zip(valid_uuids, valid_gids)))
    # </DEBUG>
    # Execute SQL Add
    gid_list = ibs.db.add_cleanly(const.IMAGE_TABLE, colnames, params_list, ibs.get_image_gids_from_uuid)

    if ut.duplicates_exist(gid_list):
        gpath_list = ibs.get_image_paths(gid_list)
        guuid_list = ibs.get_image_uuids(gid_list)
        gext_list  = ibs.get_image_exts(gid_list)
        ut.debug_duplicate_items(gid_list, gpath_list, guuid_list, gext_list)

    if ut.VERBOSE:
        uuid_list = [None if params_ is None else params_[0] for params_ in params_list]
        gid_list_ = ibs.get_image_gids_from_uuid(uuid_list)
        valid_gids = ibs.get_valid_gids()
        valid_uuids = ibs.get_image_uuids(valid_gids)
        print('[postadd] uuid / gid_ = ' + ut.indentjoin(zip(uuid_list, gid_list_)))
        print('[postadd] uuid / gid = ' + ut.indentjoin(zip(uuid_list, gid_list)))
        print('[postadd] valid uuid / gid = ' + ut.indentjoin(zip(valid_uuids, valid_gids)))

    #ibs.cfg.other_cfg.ensure_attr('auto_localize', True)
    if auto_localize is None:
        auto_localize = ibs.cfg.other_cfg.auto_localize
    if auto_localize:
        # Move to ibeis database local cache
        ibs.localize_images(gid_list)

    if as_annots:
        # Add succesfull imports as annotations
        isnone_list = [gid is None for gid in gid_list]
        gid_list_ = ut.filterfalse_items(gid_list, isnone_list)
        aid_list = ibs.use_images_as_annotations(gid_list)
        print('[ibs] added %d annotations' % (len(aid_list),))
    return gid_list


@register_ibs_method
@adder
def add_encounters(ibs, enctext_list, encounter_uuid_list=None, config_rowid_list=None,
                   notes_list=None):
    """
    Adds a list of encounters.

    Returns:
        eid_list (list): added encounter rowids
    """
    if ut.VERBOSE:
        print('[ibs] adding %d encounters' % len(enctext_list))
    # Add encounter text names to database
    if notes_list is None:
        notes_list = [''] * len(enctext_list)
    if encounter_uuid_list is None:
        encounter_uuid_list = [uuid.uuid4() for _ in range(len(enctext_list))]
    if config_rowid_list is None:
        config_rowid_list = [ibs.MANUAL_CONFIGID] * len(enctext_list)
    colnames = ['encounter_text', 'encounter_uuid', 'config_rowid', 'encounter_note']
    params_iter = zip(enctext_list, encounter_uuid_list, config_rowid_list, notes_list)
    get_rowid_from_superkey = functools.partial(ibs.get_encounter_eids_from_text, ensure=False)
    eid_list = ibs.db.add_cleanly(const.ENCOUNTER_TABLE, colnames, params_iter, get_rowid_from_superkey)
    return eid_list


@register_ibs_method
def localize_images(ibs, gid_list_=None):
    """
    Moves the images into the ibeis image cache.
    Images are renamed to img_uuid.ext
    """
    if gid_list_ is None:
        print('WARNING: you are localizing all gids')
        gid_list_  = ibs.get_valid_gids()
    isnone_list = [gid is None for gid in gid_list_]
    gid_list = ut.unique_keep_order2(ut.filterfalse_items(gid_list_, isnone_list))
    gpath_list = ibs.get_image_paths(gid_list)
    guuid_list = ibs.get_image_uuids(gid_list)
    gext_list  = ibs.get_image_exts(gid_list)
    # Build list of image names based on uuid in the ibeis imgdir
    guuid_strs = (str(guuid) for guuid in guuid_list)
    loc_gname_list = [guuid + ext for (guuid, ext) in zip(guuid_strs, gext_list)]
    loc_gpath_list = [join(ibs.imgdir, gname) for gname in loc_gname_list]
    # Copy images to local directory
    ut.copy_list(gpath_list, loc_gpath_list, lbl='Localizing Images: ')
    # Update database uris
    ibs.set_image_uris(gid_list, loc_gname_list)
    assert all(map(exists, loc_gpath_list)), 'not all images copied'


# SETTERS::IMAGE


@register_ibs_method
@setter
def set_image_uris(ibs, gid_list, new_gpath_list):
    """ Sets the image URIs to a new local path.
    This is used when localizing or unlocalizing images.
    An absolute path can either be on this machine or on the cloud
    A relative path is relative to the ibeis image cache on this machine.
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((new_gpath,) for new_gpath in new_gpath_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_uri',), val_list, id_iter)


@register_ibs_method
@setter
def set_image_contributor_rowid(ibs, gid_list, contributor_rowid_list):
    """ Sets the image contributor rowid """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((contrib_rowid,) for contrib_rowid in contributor_rowid_list)
    ibs.db.set(const.IMAGE_TABLE, ('contributor_rowid',), val_list, id_iter)


@register_ibs_method
@setter
def set_image_reviewed(ibs, gid_list, reviewed_list):
    """ Sets the image all instances found bit """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((reviewed,) for reviewed in reviewed_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_toggle_reviewed',), val_list, id_iter)


@register_ibs_method
@setter
def set_image_enabled(ibs, gid_list, enabled_list):
    """ Sets the image all instances found bit """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((enabled,) for enabled in enabled_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_toggle_enabled',), val_list, id_iter)


@register_ibs_method
@setter
def set_image_notes(ibs, gid_list, notes_list):
    """ Sets the image all instances found bit """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((notes,) for notes in notes_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_note',), val_list, id_iter)


@register_ibs_method
@setter
def set_image_unixtime(ibs, gid_list, unixtime_list):
    """ Sets the image unixtime (does not modify exif yet) """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((unixtime,) for unixtime in unixtime_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_time_posix',), val_list, id_iter)


@register_ibs_method
@setter
def set_image_enctext(ibs, gid_list, enctext_list):
    """ Sets the encoutertext of each image """
    # FIXME: Slow and weird
    if ut.VERBOSE:
        print('[ibs] setting %r image encounter ids (from text)' % len(gid_list))
    eid_list = ibs.add_encounters(enctext_list)
    ibs.set_image_eids(gid_list, eid_list)


@register_ibs_method
@setter
def set_image_eids(ibs, gid_list, eid_list):
    """ Sets the encoutertext of each image """
    if ut.VERBOSE:
        print('[ibs] setting %r image encounter ids' % len(gid_list))
    egrid_list = ibs.add_image_relationship(gid_list, eid_list)
    del egrid_list


@register_ibs_method
@setter
def set_image_gps(ibs, gid_list, gps_list=None, lat_list=None, lon_list=None):
    """ see get_image_gps for how the gps_list should look.
        lat and lon should be given in degrees """
    if gps_list is not None:
        assert lat_list is None
        assert lon_list is None
        lat_list = [tup[0] for tup in gps_list]
        lon_list = [tup[1] for tup in gps_list]
    colnames = ('image_gps_lat', 'image_gps_lon',)
    val_list = zip(lat_list, lon_list)
    id_iter = ((gid,) for gid in gid_list)
    ibs.db.set(const.IMAGE_TABLE, colnames, val_list, id_iter)


#
# GETTERS::IMAGE_TABLE


@register_ibs_method
@getter_1to1
def get_images(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of images in numpy matrix form by gid
    """
    from vtool import image as gtool
    gpath_list = ibs.get_image_paths(gid_list)
    image_list = [gtool.imread(gpath) for gpath in gpath_list]
    return image_list


@register_ibs_method
@getter_1to1
def get_image_thumbtup(ibs, gid_list, thumbsize=None):
    """
    Returns:
        list: thumbtup_list - [(thumb_path, img_path, imgsize, bboxes, thetas)]
    """
    if thumbsize is None:
        thumbsize = ibs.cfg.other_cfg.thumb_size
    # print('gid_list = %r' % (gid_list,))
    aids_list = ibs.get_image_aids(gid_list)
    bboxes_list = ibsfuncs.unflat_map(ibs.get_annot_bboxes, aids_list)
    thetas_list = ibsfuncs.unflat_map(ibs.get_annot_thetas, aids_list)
    thumb_gpaths = ibs.get_image_thumbpath(gid_list, thumbsize=thumbsize)
    image_paths = ibs.get_image_paths(gid_list)
    gsize_list = ibs.get_image_sizes(gid_list)
    thumbtup_list = [
        (thumb_path, img_path, img_size, bboxes, thetas)
        for thumb_path, img_path, img_size, bboxes, thetas in
        zip(thumb_gpaths, image_paths, gsize_list, bboxes_list, thetas_list)
    ]
    return thumbtup_list


@register_ibs_method
@getter_1to1
def get_image_thumbpath(ibs, gid_list, thumbsize=None):
    """
    Returns:
        list_ (list): the thumbnail path of each gid """
    if thumbsize is None:
        thumbsize = ibs.cfg.other_cfg.thumb_size
    thumb_dpath = ibs.thumb_dpath
    img_uuid_list = ibs.get_image_uuids(gid_list)
    thumb_suffix = '_' + str(thumbsize) + const.IMAGE_THUMB_SUFFIX
    thumbpath_list = [join(thumb_dpath, const.__STR__(uuid) + thumb_suffix)
                      for uuid in img_uuid_list]
    return thumbpath_list


@register_ibs_method
@getter_1to1
def get_image_uuids(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of image uuids by gid """
    image_uuid_list = ibs.db.get(const.IMAGE_TABLE, ('image_uuid',), gid_list)
    return image_uuid_list


@register_ibs_method
@getter_1to1
def get_image_contributor_rowid(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of image contributor rowids by gid """
    contrib_rowid_list = ibs.db.get(const.IMAGE_TABLE, ('contributor_rowid',), gid_list)
    return contrib_rowid_list


@register_ibs_method
@getter_1to1
def get_image_exts(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of image uuids by gid """
    image_uuid_list = ibs.db.get(const.IMAGE_TABLE, ('image_ext',), gid_list)
    return image_uuid_list


@register_ibs_method
@getter_1to1
def get_image_uris(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of image uris relative to the image dir by gid """
    uri_list = ibs.db.get(const.IMAGE_TABLE, ('image_uri',), gid_list)
    return uri_list


@register_ibs_method
@getter_1to1
def get_image_gids_from_uuid(ibs, uuid_list):
    """
    Returns:
        list_ (list): a list of original image names """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    gid_list = ibs.db.get(const.IMAGE_TABLE, ('image_rowid',), uuid_list, id_colname='image_uuid')
    return gid_list

#get_image_rowid_from_uuid = get_image_gids_from_uuid


@register_ibs_method
@getter_1to1
def get_image_paths(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of image absolute paths to img_dir """
    ut.assert_all_not_None(gid_list, 'gid_list', key_list=['gid_list'])
    uri_list = ibs.get_image_uris(gid_list)
    # Images should never have null uris
    ut.assert_all_not_None(uri_list, 'uri_list', key_list=['uri_list', 'gid_list'])
    gpath_list = [join(ibs.imgdir, uri) for uri in uri_list]
    return gpath_list

# TODO make this actually return a uri format
#get_image_absolute_uri = get_image_paths


@register_ibs_method
@getter_1to1
def get_image_detectpaths(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of image paths resized to a constant area for detection
    """
    from ibeis.model.preproc import preproc_detectimg
    new_gfpath_list = preproc_detectimg.compute_and_write_detectimg_lazy(ibs, gid_list)
    return new_gfpath_list


@register_ibs_method
@getter_1to1
def get_image_gnames(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of original image names """
    gname_list = ibs.db.get(const.IMAGE_TABLE, ('image_original_name',), gid_list)
    return gname_list


@register_ibs_method
@getter_1to1
def get_image_sizes(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of (width, height) tuples """
    gsize_list = ibs.db.get(const.IMAGE_TABLE, ('image_width', 'image_height'), gid_list)
    return gsize_list


@register_ibs_method
@ut.accepts_numpy
@getter_1to1
def get_image_unixtime(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of times that the images were taken by gid.

    Returns:
        list_ (list): -1 if no timedata exists for a given gid
    """
    return ibs.db.get(const.IMAGE_TABLE, ('image_time_posix',), gid_list)


@register_ibs_method
@getter_1to1
def get_image_gps(ibs, gid_list):
    """
    Returns:
        gps_list (list): -1 if no timedata exists for a given gid
    """
    gps_list = ibs.db.get(const.IMAGE_TABLE, ('image_gps_lat', 'image_gps_lon'), gid_list)
    return gps_list


@register_ibs_method
@getter_1to1
def get_image_lat(ibs, gid_list):
    lat_list = ibs.db.get(const.IMAGE_TABLE, ('image_gps_lat',), gid_list)
    return lat_list


@register_ibs_method
@getter_1to1
def get_image_lon(ibs, gid_list):
    lon_list = ibs.db.get(const.IMAGE_TABLE, ('image_gps_lon',), gid_list)
    return lon_list


@register_ibs_method
@getter_1to1
def get_image_enabled(ibs, gid_list):
    """
    Returns:
        list_ (list): "Image Enabled" flag, true if the image is enabled """
    enabled_list = ibs.db.get(const.IMAGE_TABLE, ('image_toggle_enabled',), gid_list)
    return enabled_list


@register_ibs_method
@getter_1to1
def get_image_reviewed(ibs, gid_list):
    """
    Returns:
        list_ (list): "All Instances Found" flag, true if all objects of interest
    (animals) have an ANNOTATION in the image """
    reviewed_list = ibs.db.get(const.IMAGE_TABLE, ('image_toggle_reviewed',), gid_list)
    return reviewed_list


@register_ibs_method
@getter_1to1
def get_image_detect_confidence(ibs, gid_list):
    """
    Returns:
        list_ (list): image detection confidence as the max of ANNOTATION confidences """
    aids_list = ibs.get_image_aids(gid_list)
    confs_list = ibsfuncs.unflat_map(ibs.get_annot_detect_confidence, aids_list)
    maxconf_list = [max(confs) if len(confs) > 0 else -1 for confs in confs_list]
    return maxconf_list


@register_ibs_method
@getter_1to1
def get_image_notes(ibs, gid_list):
    """
    Returns:
        list_ (list): image notes """
    notes_list = ibs.db.get(const.IMAGE_TABLE, ('image_note',), gid_list)
    return notes_list


@register_ibs_method
@getter_1to1
def get_image_nids(ibs, gid_list):
    """

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):

    Returns:
        list: nids_list - the name ids associated with an image id

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-get_image_nids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> # execute function
        >>> nids_list = ibs.get_image_nids(gid_list)
        >>> # verify results
        >>> result = str(nids_list)
        >>> print(result)

    """
    aids_list = ibs.get_image_aids(gid_list)
    nids_list = ibs.unflat_map(ibs.get_annot_name_rowids, aids_list)
    return nids_list


@register_ibs_method
@getter_1to1
def get_image_species_rowids(ibs, gid_list):
    """
    Returns:
        list_ (list): the name ids associated with an image id """
    aids_list = ibs.get_image_aids(gid_list)
    species_rowid_list = ibs.get_annot_species_rowids(aids_list)
    return species_rowid_list


@register_ibs_method
@getter_1toM
def get_image_eids(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of encounter ids for each image by gid """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    colnames = ('encounter_rowid',)
    eids_list = ibs.db.get(const.EG_RELATION_TABLE, colnames, gid_list,
                           id_colname='image_rowid', unpack_scalars=False)
    return eids_list


@register_ibs_method
@getter_1toM
def get_image_enctext(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of enctexts for each image by gid """
    eids_list = ibs.get_image_eids(gid_list)
    enctext_list = ibsfuncs.unflat_map(ibs.get_encounter_enctext, eids_list)
    return enctext_list


ANNOT_ROWID = 'annot_rowid'
IMAGE_ROWID = 'image_rowid'


@register_ibs_method
@getter_1toM
#@cache_getter(const.IMAGE_TABLE)
#@profile
def get_image_aids(ibs, gid_list):
    """
    Returns:
        list_ (list): a list of aids for each image by gid

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):

    Returns:
        list: aids_list

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-get_image_aids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_annot_gids(ibs.get_valid_aids())
        >>> gid_list = gid_list + gid_list[::5]
        >>> # execute function
        >>> aids_list = get_image_aids(ibs, gid_list)
        >>> # verify results
        >>> result = str(aids_list)
        >>> print(result)
        [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [1], [6], [11]]


    Ignore:
        print('len(gid_list) = %r' % (len(gid_list),))
        print('len(input_list) = %r' % (len(input_list),))
        print('len(pair_list) = %r' % (len(pair_list),))
        print('len(aidscol) = %r' % (len(aidscol),))
        print('len(gidscol) = %r' % (len(gidscol),))
        print('len(unique_gids) = %r' % (len(unique_gids),))
    """

    # FIXME: SLOW JUST LIKE GET_NAME_AIDS
    # print('gid_list = %r' % (gid_list,))
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    USE_GROUPING_HACK = False
    if USE_GROUPING_HACK:
        input_list, inverse_unique = np.unique(gid_list, return_inverse=True)
        # This code doesn't work because it doesn't respect empty names
        input_str = ', '.join(list(map(str, input_list)))
        opstr = '''
        SELECT annot_rowid, image_rowid
        FROM {ANNOTATION_TABLE}
        WHERE image_rowid IN
            ({input_str})
            ORDER BY image_rowid ASC, annot_rowid ASC
        '''.format(input_str=input_str, ANNOTATION_TABLE=const.ANNOTATION_TABLE)
        pair_list = ibs.db.connection.execute(opstr).fetchall()
        aidscol = np.array(ut.get_list_column(pair_list, 0))
        gidscol = np.array(ut.get_list_column(pair_list, 1))
        unique_gids, groupx = vt.group_indicies(gidscol)
        grouped_aids_ = vt.apply_grouping(aidscol, groupx)
        #aids_list = [sorted(arr.tolist()) for arr in grouped_aids_]
        structured_aids_list = [arr.tolist() for arr in grouped_aids_]
        with ut.EmbedOnException():
            aids_list = np.array(structured_aids_list)[inverse_unique].tolist()
    else:
        USE_NUMPY_IMPL = True
        # Use qt if getting one at a time otherwise perform bulk operation
        USE_NUMPY_IMPL = len(gid_list) > 1
        if USE_NUMPY_IMPL:
            # This seems to be 30x faster for bigger inputs
            valid_aids = np.array(ibs._get_all_aids())
            valid_gids = np.array(ibs.db.get_all_col_rows(const.ANNOTATION_TABLE, IMAGE_ROWID))
            #np.array(ibs.get_annot_name_rowids(valid_aids, distinguish_unknowns=False))
            aids_list = [valid_aids.take(np.flatnonzero(np.equal(valid_gids, gid))).tolist() for gid in gid_list]
        else:
            # SQL IMPL
            aids_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_ROWID,), gid_list,
                                       id_colname=IMAGE_ROWID, unpack_scalars=False)
    #print('aids_list = %r' % (aids_list,))
    return aids_list


@register_ibs_method
@getter_1toM
#@cache_getter(const.IMAGE_TABLE)
def get_image_aids_of_species(ibs, gid_list, species=None):
    """
    Returns:
        list_ (list): a list of aids for each image by gid filtered by species """
    def _filter(aid_list):
        species_list = ibs.get_annot_species(aid_list)
        isvalid_list = [ species_ == species for species_ in species_list ]
        aid_list = ut.filter_items(aid_list, isvalid_list)
        return aid_list
    # Get and filter aids_list
    aids_list = ibs.get_image_aids(gid_list)
    if species is None:
        # We do this so that the species flag behaves nicely with the getter_1toM
        print('[get_image_aids_of_species] WARNING! Use get_image_aids() instead.')
        return aids_list
    aids_list = [ _filter(aid_list) for aid_list in aids_list]
    return aids_list


@register_ibs_method
@getter_1to1
#@profile
def get_image_num_annotations(ibs, gid_list):
    """
    Returns:
        list_ (list): the number of chips in each image """
    return list(map(len, ibs.get_image_aids(gid_list)))


@register_ibs_method
@getter_1to1
def get_image_egrids(ibs, gid_list):
    """
    Returns:
        list_ (list):  a list of encounter-image-relationship rowids for each imageid """
    # TODO: Group type
    params_iter = ((gid,) for gid in gid_list)
    where_clause = 'image_rowid=?'
    # list of relationships for each image
    egrids_list = ibs.db.get_where(const.EG_RELATION_TABLE, ('egr_rowid',), params_iter, where_clause, unpack_scalars=False)
    return egrids_list


@register_ibs_method
@deleter
def delete_images(ibs, gid_list):
    """ deletes images from the database that belong to gids"""
    if ut.VERBOSE:
        print('[ibs] deleting %d images' % len(gid_list))
    # Move images to trash before deleting them. #
    # TODO: only move localized images
    # TODO: ensure there are no name conflicts when using the original names
    gpath_list = ibs.get_image_paths(gid_list)
    gname_list = ibs.get_image_gnames(gid_list)
    ext_list   = ibs.get_image_exts(gid_list)
    trash_dir  = ibs.get_trashdir()
    ut.ensuredir(trash_dir)
    gpath_list2 = [join(trash_dir, gname + ext) for (gname, ext) in
                   zip(gname_list, ext_list)]
    ut.copy_list(gpath_list, gpath_list2)
    #ut.view_directory(trash_dir)

    # Delete annotations first
    aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    ibs.delete_annots(aid_list)
    ibs.delete_image_thumbs(gid_list)  # delete thumbs in case an annot doesnt delete them TODO: pass flag to not delete them in delete_annots
    ibs.db.delete_rowids(const.IMAGE_TABLE, gid_list)
    #egrid_list = ut.flatten(ibs.get_image_egrids(gid_list))
    #ibs.db.delete_rowids(const.EG_RELATION_TABLE, egrid_list)
    ibs.db.delete(const.EG_RELATION_TABLE, gid_list, id_colname='image_rowid')


@register_ibs_method
@deleter
def delete_image_thumbs(ibs, gid_list, quiet=False):
    """ Removes image thumbnails from disk """
    # print('gid_list = %r' % (gid_list,))
    thumbpath_list = ibs.get_image_thumbpath(gid_list)
    #ut.remove_fpaths(thumbpath_list, quiet=quiet, lbl='image_thumbs')
    ut.remove_existing_fpaths(thumbpath_list, quiet=quiet, lbl='image_thumbs')


@register_ibs_method
@deleter
def delete_egr_encounter_relations(ibs, eid_list):
    """ Removes relationship between input encounters and all images """
    ibs.db.delete(const.EG_RELATION_TABLE, eid_list, id_colname='encounter_rowid')


@register_ibs_method
@deleter
def delete_egr_image_relations(ibs, gid_list):
    """ Removes relationship between input images and all encounters """
    ibs.db.delete(const.EG_RELATION_TABLE, gid_list, id_colname='image_rowid')


@register_ibs_method
@deleter
def unrelate_images_and_encounters(ibs, gid_list, eid_list):
    """
    Seems to unrelate specific image encounter pairs

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):
        eid_list (list):

    Returns:
        list: gids_list

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-unrelate_images_and_encounters
        python -c "import utool; print(utool.auto_docstr('ibeis.control.manual_image_funcs', 'delete_egr_image_relations'))"

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> # Reset and compute encounters
        >>> ibs.delete_all_encounters()
        >>> ibs.compute_encounters()
        >>> eid_list = ibs.get_valid_eids()
        >>> gids_list = ibs.get_encounter_gids(eid_list)
        >>> assert len(eid_list) == 2
        >>> assert len(gids_list) == 2
        >>> assert len(gids_list[0]) == 7
        >>> assert len(gids_list[1]) == 6
        >>> # Add encounter 2 gids to encounter 1 so an image belongs to multiple encounters
        >>> enc2_gids = gids_list[1][0:1]
        >>> enc1_eids = eid_list[0:1]
        >>> ibs.add_image_relationship(enc2_gids, enc1_eids)
        >>> # Now delete the image from the encounter 2
        >>> enc2_eids = eid_list[1:2]
        >>> # execute function
        >>> ibs.unrelate_images_and_encounters(enc2_gids, enc2_eids)
        >>> # verify results
        >>> ibs.print_egpairs_table()
        >>> eid_list_ = ibs.get_valid_eids()
        >>> gids_list_ = ibs.get_encounter_gids(eid_list_)
        >>> result = str(gids_list_)
        >>> print(result)
        >>> # enc2_gids should now only be in encounter1
        >>> assert enc2_gids[0] in gids_list_[0]
        >>> assert enc2_gids[0] not in gids_list_[1]
    """
    # WHAT IS THIS FUNCTION? FIXME CALLS WEIRD FUNCTION
    if ut.VERBOSE:
        print('[ibs] deleting %r image\'s encounter ids' % len(gid_list))
    egrid_list = ut.flatten(ibs.get_encounter_egrids(eid_list=eid_list, gid_list=gid_list))
    ibs.db.delete_rowids(const.EG_RELATION_TABLE, egrid_list)


# SETTERS::ENCOUNTER


@register_ibs_method
@setter
def set_encounter_enctext(ibs, eid_list, names_list):
    """ Sets names of encounters (groups of animals) """
    id_iter = ((eid,) for eid in eid_list)
    val_list = ((names,) for names in names_list)
    ibs.db.set(const.ENCOUNTER_TABLE, ('encounter_text',), val_list, id_iter)

#
# GETTERS::ENCOUNTER


@register_ibs_method
@getter_1to1
def get_encounter_num_gids(ibs, eid_list):
    """
    Returns:
        nGids_list (list): number of images in each encounter """
    nGids_list = list(map(len, ibs.get_encounter_gids(eid_list)))
    return nGids_list


@register_ibs_method
@getter_1toM
def get_encounter_aids(ibs, eid_list):
    """
    Returns:
        aids_list (list):  a list of list of aids in each encounter """
    gids_list = ibs.get_encounter_gids(eid_list)
    aids_list_ = ibsfuncs.unflat_map(ibs.get_image_aids, gids_list)
    aids_list = list(map(ut.flatten, aids_list_))
    #print('get_encounter_aids')
    #print('eid_list = %r' % (eid_list,))
    #print('gids_list = %r' % (gids_list,))
    #print('aids_list_ = %r' % (aids_list_,))
    #print('aids_list = %r' % (aids_list,))
    return aids_list


@register_ibs_method
@getter_1toM
def get_encounter_gids(ibs, eid_list):
    """
    Returns:
        gids_list (list):  a list of list of gids in each encounter """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    gids_list = ibs.db.get(const.EG_RELATION_TABLE, ('image_rowid',), eid_list, id_colname='encounter_rowid', unpack_scalars=False)
    #print('get_encounter_gids')
    #print('eid_list = %r' % (eid_list,))
    #print('gids_list = %r' % (gids_list,))
    return gids_list


@register_ibs_method
def get_encounter_egrids(ibs, eid_list=None, gid_list=None):
    """
    Returns:
        list_ (list):  a list of encounter-image-relationship rowids for each encouterid
    """
    # WEIRD FUNCTION FIXME
    assert eid_list is not None or gid_list is not None, "Either eid_list or gid_list must be None"
    if eid_list is not None and gid_list is None:
        # TODO: Group type
        params_iter = ((eid,) for eid in eid_list)
        where_clause = 'encounter_rowid=?'
        # list of relationships for each encounter
        egrids_list = ibs.db.get_where(const.EG_RELATION_TABLE, ('egr_rowid',),
                                       params_iter, where_clause, unpack_scalars=False)
    elif gid_list is not None and eid_list is None:
        # TODO: Group type
        params_iter = ((gid,) for gid in gid_list)
        where_clause = 'image_rowid=?'
        # list of relationships for each encounter
        egrids_list = ibs.db.get_where(const.EG_RELATION_TABLE, ('egr_rowid',),
                                       params_iter, where_clause, unpack_scalars=False)
    else:
        # TODO: Group type
        params_iter = ((eid, gid,) for eid, gid in zip(eid_list, gid_list))
        where_clause = 'encounter_rowid=? AND image_rowid=?'
        # list of relationships for each encounter
        egrids_list = ibs.db.get_where(const.EG_RELATION_TABLE, ('egr_rowid',),
                                       params_iter, where_clause, unpack_scalars=False)
    return egrids_list


@register_ibs_method
@getter_1toM
def get_encounter_nids(ibs, eid_list):
    """
    Returns:
        list_ (list):  a list of list of nids in each encounter

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-get_encounter_nids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> ibs.delete_encounters(ibs.get_valid_eids())
        >>> ibs.compute_encounters()
        >>> eid_list = ibs.get_valid_eids()
        >>> nids_list = ibs.get_encounter_nids(eid_list)
        >>> result = nids_list
        >>> print(result)
        [[1, 2, 3], [4, 5, 6, 7]]
    """
    aids_list = ibs.get_encounter_aids(eid_list)
    nids_list = ibsfuncs.unflat_map(ibs.get_annot_name_rowids, aids_list)
    #nids_list_ = [[nid[0] for nid in nids if len(nid) > 0] for nids in nids_list]
    # Remove any unknown anmes
    nids_list = [[nid for nid in nids if nid > 0] for nids in nids_list]

    nids_list = list(map(ut.unique_ordered, nids_list))
    #print('get_encounter_nids')
    #print('eid_list = %r' % (eid_list,))
    #print('aids_list = %r' % (aids_list,))
    #print('nids_list_ = %r' % (nids_list_,))
    #print('nids_list = %r' % (nids_list,))
    return nids_list


@register_ibs_method
@getter_1to1
def get_encounter_uuid(ibs, eid_list):
    """
    Returns:
        list_ (list): encounter_uuid of each eid in eid_list """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    encuuid_list = ibs.db.get(const.ENCOUNTER_TABLE, ('encounter_uuid',), eid_list, id_colname='encounter_rowid')
    return encuuid_list


@register_ibs_method
@getter_1to1
def get_encounter_configid(ibs, eid_list):
    """
    Returns:
        list_ (list): config_rowid of each eid in eid_list """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    config_rowid_list = ibs.db.get(const.ENCOUNTER_TABLE, ('config_rowid',), eid_list, id_colname='encounter_rowid')
    return config_rowid_list


@register_ibs_method
@getter_1to1
def get_encounter_enctext(ibs, eid_list):
    """
    Returns:
        list_ (list): encounter_text of each eid in eid_list """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    enctext_list = ibs.db.get(const.ENCOUNTER_TABLE, ('encounter_text',), eid_list, id_colname='encounter_rowid')
    return enctext_list


@register_ibs_method
@getter_1to1
def get_encounter_eids_from_text(ibs, enctext_list, ensure=True):
    """
    Returns:
        list_ (list): a list of eids corresponding to each encounter enctext
    #FIXME: make new naming scheme for non-primary-key-getters
    get_encounter_eids_from_text_from_text
    """
    if ensure:
        eid_list = ibs.add_encounters(enctext_list)
    else:
        # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
        eid_list = ibs.db.get(const.ENCOUNTER_TABLE, ('encounter_rowid',), enctext_list, id_colname='encounter_text')
    return eid_list


@register_ibs_method
@getter_1to1
def get_encounter_note(ibs, eid_list):
    """
    Returns:
        list_ (list): encounter_note of each eid in eid_list """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    encnote_list = ibs.db.get(const.ENCOUNTER_TABLE, ('encounter_note',), eid_list, id_colname='encounter_rowid')
    return encnote_list


@register_ibs_method
@deleter
def delete_encounters(ibs, eid_list):
    """ Removes encounters and thier relationships (images are not effected) """
    # Optimization hack, less SQL calls
    #egrid_list = ut.flatten(ibs.get_encounter_egrids(eid_list=eid_list))
    #ibs.db.delete_rowids(const.EG_RELATION_TABLE, egrid_list)
    #ibs.db.delete(const.EG_RELATION_TABLE, eid_list, id_colname='encounter_rowid')
    if ut.VERBOSE:
        print('[ibs] deleting %d encounters' % len(eid_list))
    ibs.delete_egr_encounter_relations(eid_list)
    ibs.db.delete_rowids(const.ENCOUNTER_TABLE, eid_list)


@register_ibs_method
@getter_1to1
def get_encounter_end_time_posix(ibs, encounter_rowid_list):
    """ encounter_end_time_posix_list <- encounter.encounter_end_time_posix[encounter_rowid_list]

    gets data from the "native" column "encounter_end_time_posix" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_end_time_posix_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_end_time_posix
        tbl = encounter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, qreq_ = get_autogen_testdata()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_end_time_posix_list = ibs.get_encounter_end_time_posix(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_end_time_posix_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_END_TIME_POSIX,)
    encounter_end_time_posix_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_end_time_posix_list


@register_ibs_method
@getter_1to1
def get_encounter_gps_lats(ibs, encounter_rowid_list):
    """ encounter_gps_lat_list <- encounter.encounter_gps_lat[encounter_rowid_list]

    gets data from the "native" column "encounter_gps_lat" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_gps_lat_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_gps_lat
        tbl = encounter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, qreq_ = get_autogen_testdata()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_gps_lat_list = ibs.get_encounter_gps_lats(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_gps_lat_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_GPS_LAT,)
    encounter_gps_lat_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_gps_lat_list


@register_ibs_method
def update_encounter_info(ibs, encounter_rowid_list):
    """ sets start and end time for encounters

    FIXME: should not need to bulk update, should be handled as it goes

    Example:
        >>> # DOCTEST_DISABLE
        >>> encounter_rowid_list = ibs.get_valid_eids()
    """
    gids_list_ = ibs.get_encounter_gids(encounter_rowid_list)
    hasgids_list = [len(gids) > 0 for gids in gids_list_]
    gids_list = ut.filter_items(gids_list_, hasgids_list)
    eid_list = ut.filter_items(encounter_rowid_list, hasgids_list)
    unixtimes_list = ibs.unflat_map(ibs.get_image_unixtime, gids_list)
    encounter_end_time_posix_list = [max(unixtimes) for unixtimes in unixtimes_list]
    encounter_start_time_posix_list = [min(unixtimes) for unixtimes in unixtimes_list]
    ibs.set_encounter_end_time_posix(eid_list, encounter_end_time_posix_list)
    ibs.set_encounter_start_time_posix(eid_list, encounter_start_time_posix_list)


@register_ibs_method
@getter_1to1
def get_encounter_gps_lons(ibs, encounter_rowid_list):
    """ encounter_gps_lon_list <- encounter.encounter_gps_lon[encounter_rowid_list]

    gets data from the "native" column "encounter_gps_lon" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_gps_lon_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_gps_lon
        tbl = encounter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, qreq_ = get_autogen_testdata()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_gps_lon_list = ibs.get_encounter_gps_lons(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_gps_lon_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_GPS_LON,)
    encounter_gps_lon_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_gps_lon_list


@register_ibs_method
@getter_1to1
def get_encounter_notes(ibs, encounter_rowid_list):
    """ encounter_note_list <- encounter.encounter_note[encounter_rowid_list]

    gets data from the "native" column "encounter_note" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_note_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_note
        tbl = encounter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, qreq_ = get_autogen_testdata()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_note_list = ibs.get_encounter_notes(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_note_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_NOTE,)
    encounter_note_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_note_list


@register_ibs_method
@getter_1to1
def get_encounter_processed_flags(ibs, encounter_rowid_list):
    """ encounter_processed_flag_list <- encounter.encounter_processed_flag[encounter_rowid_list]

    gets data from the "native" column "encounter_processed_flag" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_processed_flag_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_processed_flag
        tbl = encounter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, qreq_ = get_autogen_testdata()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_processed_flag_list = ibs.get_encounter_processed_flags(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_processed_flag_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_PROCESSED_FLAG,)
    encounter_processed_flag_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_processed_flag_list


@register_ibs_method
@getter_1to1
def get_encounter_shipped_flags(ibs, encounter_rowid_list):
    """ encounter_shipped_flag_list <- encounter.encounter_shipped_flag[encounter_rowid_list]

    gets data from the "native" column "encounter_shipped_flag" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_shipped_flag_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_shipped_flag
        tbl = encounter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, qreq_ = get_autogen_testdata()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_shipped_flag_list = ibs.get_encounter_shipped_flags(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_shipped_flag_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SHIPPED_FLAG,)
    encounter_shipped_flag_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_shipped_flag_list


@register_ibs_method
@getter_1to1
def get_encounter_start_time_posix(ibs, encounter_rowid_list):
    """ encounter_start_time_posix_list <- encounter.encounter_start_time_posix[encounter_rowid_list]

    gets data from the "native" column "encounter_start_time_posix" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_start_time_posix_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_start_time_posix
        tbl = encounter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, qreq_ = get_autogen_testdata()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_start_time_posix_list = ibs.get_encounter_start_time_posix(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_start_time_posix_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_START_TIME_POSIX,)
    encounter_start_time_posix_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_start_time_posix_list


@register_ibs_method
@setter
def set_encounter_end_time_posix(ibs, encounter_rowid_list, encounter_end_time_posix_list):
    """ encounter_end_time_posix_list -> encounter.encounter_end_time_posix[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_end_time_posix_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_end_time_posix
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_END_TIME_POSIX,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames,
               encounter_end_time_posix_list, id_iter)


@register_ibs_method
@setter
def set_encounter_gps_lats(ibs, encounter_rowid_list, encounter_gps_lat_list):
    """ encounter_gps_lat_list -> encounter.encounter_gps_lat[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_gps_lat_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_gps_lat
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_GPS_LAT,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames, encounter_gps_lat_list, id_iter)


@register_ibs_method
@setter
def set_encounter_gps_lons(ibs, encounter_rowid_list, encounter_gps_lon_list):
    """ encounter_gps_lon_list -> encounter.encounter_gps_lon[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_gps_lon_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_gps_lon
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_GPS_LON,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames, encounter_gps_lon_list, id_iter)


@register_ibs_method
@setter
def set_encounter_notes(ibs, encounter_rowid_list, encounter_note_list):
    """ encounter_note_list -> encounter.encounter_note[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_note_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_note
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_NOTE,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames, encounter_note_list, id_iter)


@register_ibs_method
@setter
def set_encounter_processed_flags(ibs, encounter_rowid_list, encounter_processed_flag_list):
    """ encounter_processed_flag_list -> encounter.encounter_processed_flag[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_processed_flag_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_processed_flag
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_PROCESSED_FLAG,)
    val_iter = ((processed_flag,) for processed_flag in encounter_processed_flag_list)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames, val_iter, id_iter)


@register_ibs_method
@setter
def set_encounter_shipped_flags(ibs, encounter_rowid_list, encounter_shipped_flag_list):
    """ encounter_shipped_flag_list -> encounter.encounter_shipped_flag[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_shipped_flag_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_shipped_flag
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SHIPPED_FLAG,)
    val_iter = ((shipped_flag,) for shipped_flag in encounter_shipped_flag_list)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames,
               val_iter, id_iter)


@register_ibs_method
@setter
def set_encounter_start_time_posix(ibs, encounter_rowid_list, encounter_start_time_posix_list):
    """ encounter_start_time_posix_list -> encounter.encounter_start_time_posix[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_start_time_posix_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_start_time_posix
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_START_TIME_POSIX,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames,
               encounter_start_time_posix_list, id_iter)


# GETTERS::EG_RELATION_TABLE


@register_ibs_method
@getter_1to1
def get_egr_rowid_from_superkey(ibs, gid_list, eid_list):
    """
    Returns:
        egrid_list (list):  eg-relate-ids from info constrained to be unique (eid, gid) """
    colnames = ('image_rowid',)
    params_iter = zip(gid_list, eid_list)
    where_clause = 'image_rowid=? AND encounter_rowid=?'
    egrid_list = ibs.db.get_where(const.EG_RELATION_TABLE, colnames, params_iter, where_clause)
    return egrid_list


# NEW ENCOUNTER SMART SETTER GETTERS


@register_ibs_method
#@accessor_decors.cache_getter(const.ENCOUNTER_TABLE, ENCOUNTER_SMART_WAYPOINT_ID)
def get_encounter_smart_waypoint_ids(ibs, encounter_rowid_list):
    """ encounter_smart_waypoint_id_list <- encounter.encounter_smart_waypoint_id[encounter_rowid_list]

    gets data from the "native" column "encounter_smart_waypoint_id" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_smart_waypoint_id_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_smart_waypoint_id
        tbl = encounter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, qreq_ = get_autogen_testdata()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_smart_waypoint_id_list = ibs.get_encounter_smart_waypoint_ids(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_smart_waypoint_id_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SMART_WAYPOINT_ID,)
    encounter_smart_waypoint_id_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_smart_waypoint_id_list


@register_ibs_method
#@accessor_decors.cache_getter(const.ENCOUNTER_TABLE, ENCOUNTER_SMART_XML_FNAME)
def get_encounter_smart_xml_fnames(ibs, encounter_rowid_list):
    """ encounter_smart_xml_fname_list <- encounter.encounter_smart_xml_fname[encounter_rowid_list]

    gets data from the "native" column "encounter_smart_xml_fname" in the "encounter" table

    Args:
        encounter_rowid_list (list):

    Returns:
        list: encounter_smart_xml_fname_list

    TemplateInfo:
        Tgetter_table_column
        col = encounter_smart_xml_fname
        tbl = encounter

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, qreq_ = get_autogen_testdata()
        >>> encounter_rowid_list = ibs._get_all_encounter_rowids()
        >>> encounter_smart_xml_fname_list = ibs.get_encounter_smart_xml_fnames(encounter_rowid_list)
        >>> assert len(encounter_rowid_list) == len(encounter_smart_xml_fname_list)
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SMART_XML_FNAME,)
    encounter_smart_xml_fname_list = ibs.db.get(
        const.ENCOUNTER_TABLE, colnames, id_iter, id_colname='rowid')
    return encounter_smart_xml_fname_list


@register_ibs_method
#@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ENCOUNTER_SMART_WAYPOINT_ID, native_rowids=True)
def set_encounter_smart_waypoint_ids(ibs, encounter_rowid_list, encounter_smart_waypoint_id_list):
    """ encounter_smart_waypoint_id_list -> encounter.encounter_smart_waypoint_id[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_smart_waypoint_id_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_smart_waypoint_id
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SMART_WAYPOINT_ID,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames,
               encounter_smart_waypoint_id_list, id_iter)


@register_ibs_method
#@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ENCOUNTER_SMART_XML_FNAME, native_rowids=True)
def set_encounter_smart_xml_fnames(ibs, encounter_rowid_list, encounter_smart_xml_fname_list):
    """ encounter_smart_xml_fname_list -> encounter.encounter_smart_xml_fname[encounter_rowid_list]

    Args:
        encounter_rowid_list
        encounter_smart_xml_fname_list

    TemplateInfo:
        Tsetter_native_column
        tbl = encounter
        col = encounter_smart_xml_fname
    """
    id_iter = encounter_rowid_list
    colnames = (ENCOUNTER_SMART_XML_FNAME,)
    ibs.db.set(const.ENCOUNTER_TABLE, colnames,
               encounter_smart_xml_fname_list, id_iter)


def get_autogen_testdata():
    import ibeis
    ibs = ibeis.opendb('testdb1')
    qreq_ = None
    return ibs, qreq_


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.manual_image_funcs
        python -m ibeis.control.manual_image_funcs --allexamples
        python -m ibeis.control.manual_image_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
