"""
Functions for images and encoutners that will be injected into an
IBEISController instance.


CommandLine:
    # Autogenerate Encounter Functions
    # key should be the table name
    # the write flag makes a file, but dont use that
    python -m ibeis.templates.template_generator --key image --onlyfn
    python -m ibeis.templates.template_generator --key image --fnfilt timedelta_posix --modfname manual_image_funcs
    python -m ibeis.templates.template_generator --key image --fnfilt location --modfname manual_image_funcs
    python -m ibeis.templates.template_generator --key image --fnfilt set_.*time --modfname manual_image_funcs

    image_timedelta_posix

"""
from __future__ import absolute_import, division, print_function
from ibeis import constants as const
from ibeis.control import accessor_decors, controller_inject
from ibeis.control.controller_inject import make_ibs_register_decorator
from os.path import join, exists
import numpy as np
import utool as ut
import vtool as vt
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[manual_image]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


IMAGE_TIME_POSIX      = 'image_time_posix'
IMAGE_LOCATION_CODE   = 'image_location_code'
IMAGE_TIMEDELTA_POSIX = 'image_timedelta_posix'
PARTY_ROWID           = 'party_rowid'
CONTRIBUTOR_ROWID     = 'contributor_rowid'


@register_ibs_method
@accessor_decors.ider
def _get_all_gids(ibs):
    r"""
    alias

    Returns:
        list_ (list):  all unfiltered gids (image rowids)
    """
    all_gids = ibs._get_all_image_rowids()
    return all_gids


@register_ibs_method
def _get_all_image_rowids(ibs):
    r"""
    all_image_rowids <- image.get_all_rowids()

    Returns:
        list_ (list): unfiltered image_rowids

    TemplateInfo:
        Tider_all_rowids
        tbl = image

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-_get_all_image_rowids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> all_image_rowids = ibs._get_all_image_rowids()
        >>> result = str(all_image_rowids)
        >>> print(result)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    """
    all_image_rowids = ibs.db.get_all_rowids(const.IMAGE_TABLE)
    return all_image_rowids


@register_ibs_method
@accessor_decors.ider
@register_api('/api/image/', methods=['GET'])
def get_valid_gids(ibs, eid=None, require_unixtime=False, reviewed=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        eid (None):
        require_unixtime (bool):
        reviewed (None):

    Returns:
        list: gid_list

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-get_valid_gids

    RESTful:
        Method: GET
        URL:    /api/image/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> eid = None
        >>> require_unixtime = False
        >>> reviewed = None
        >>> # execute function
        >>> gid_list = get_valid_gids(ibs, eid, require_unixtime, reviewed)
        >>> # verify results
        >>> result = str(gid_list)
        >>> print(result)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    """
    if eid is None:
        gid_list = ibs._get_all_gids()
    else:
        assert not ut.isiterable(eid)
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
@accessor_decors.getter_1to1
def get_image_gid(ibs, gid_list, eager=True, nInput=None):
    """ self verifier

    CommandLine:
        python -m ibeis.control.manual_image_funcs --exec-get_image_gid

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids() + [None, -1, 10434320432]
        >>> gid_list_ = ibs.get_image_gid(gid_list)
        >>> assert [r is None for r in gid_list_[-3:]]
        >>> assert [r is not None for r in gid_list_[0:-3]]
        >>> print('gid_list_ = %r' % (gid_list_,))
    """
    id_iter = gid_list
    colnames = (IMAGE_ROWID,)
    gid_list = ibs.db.get(const.IMAGE_TABLE, colnames,
                          id_iter, id_colname='rowid', eager=eager, nInput=nInput)
    return gid_list


@register_ibs_method
@accessor_decors.ider
@register_api('/api/image/valid_rowids/', methods=['GET'])
def get_valid_image_rowids(ibs, eid=None, require_unixtime=False, reviewed=None):
    r"""
    alias

    RESTful:
        Method: GET
        URL:    /api/image/valid_rowids/
    """
    return get_valid_gids(ibs, eid, require_unixtime, reviewed)


@register_ibs_method
@register_api('/api/image/num/', methods=['GET'])
def get_num_images(ibs, **kwargs):
    r"""
    Number of valid images

    RESTful:
        Method: GET
        URL:    /api/image/num/
    """
    gid_list = ibs.get_valid_gids(**kwargs)
    return len(gid_list)


@register_ibs_method
@accessor_decors.adder
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_imgs_reviewed_str'])
@register_api('/api/image/path', methods=['POST'])
def add_images(ibs, gpath_list, params_list=None, as_annots=False, auto_localize=None):
    r"""
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

    RESTful:
        Method: POST
        URL:    /api/image/path

    Example0:
        >>> # ENABLE_DOCTEST
        >>> # Test returns None on fail to add
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> gpath_list = ['doesnotexist.jpg']
        >>> assert not ut.checkpath(gpath_list[0])
        >>> gid_list = ibs.add_images(gpath_list)
        >>> assert len(gid_list) == len(gpath_list)
        >>> assert gid_list[0] is None

    Example1:
        >>> # ENABLE_DOCTSET
        >>> # test double add
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> new_gpath_list = [ut.grab_test_imgpath('carl.jpg')]
        >>> new_gids1 = ibs.add_images(new_gpath_list, auto_localize=False)
        >>> new_gids2 = ibs.add_images(new_gpath_list, auto_localize=False)
        >>> #new_gids2 = ibs.add_images(new_gpath_list, auto_localize=True)
        >>> assert new_gids1 == new_gids2, 'should be the same'
        >>> new_gpath_list2 = ibs.get_image_paths(new_gids1)
        >>> assert new_gpath_list == new_gpath_list2, 'should not move when autolocalize is False'
        >>> # Clean things up
        >>> ibs.delete_images(new_gids1)
    """
    from ibeis.model.preproc import preproc_image
    from ibeis import ibsfuncs
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
        uuid_colx = colnames.index('image_uuid')
        uuid_list = [None if params_ is None else params_[uuid_colx] for params_ in params_list]
        gid_list_ = ibs.get_image_gids_from_uuid(uuid_list)
        valid_gids = ibs.get_valid_gids()
        valid_uuids = ibs.get_image_uuids(valid_gids)
        print('[preadd] uuid / gid_ = ' + ut.indentjoin(zip(uuid_list, gid_list_)))
        print('[preadd] valid uuid / gid = ' + ut.indentjoin(zip(valid_uuids, valid_gids)))
    # </DEBUG>
    # Execute SQL Add
    from distutils.version import LooseVersion

    if LooseVersion(ibs.db.get_db_version()) >= LooseVersion('1.3.4'):
        colnames = colnames + ('image_original_path', 'image_location_code')
        params_list = [tuple(params) + (gpath, ibs.cfg.other_cfg.location_for_names)
                        if params is not None else None
                        for params, gpath in zip(params_list, gpath_list)]

    gid_list = ibs.db.add_cleanly(const.IMAGE_TABLE, colnames, params_list, ibs.get_image_gids_from_uuid)

    if ut.duplicates_exist(gid_list):
        gpath_list = ibs.get_image_paths(gid_list)
        guuid_list = ibs.get_image_uuids(gid_list)
        gext_list  = ibs.get_image_exts(gid_list)
        if ut.VERBOSE:
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
        # grab value from config
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
def localize_images(ibs, gid_list_=None):
    r"""
    Moves the images into the ibeis image cache.
    Images are renamed to img_uuid.ext

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list_ (list):

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-localize_images

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> gpath_list  = [ut.unixpath(ut.grab_test_imgpath('carl.jpg'))]
        >>> gid_list_   = ibs.add_images(gpath_list, auto_localize=False)
        >>> gpath_list2 = ibs.get_image_paths(gid_list_)
        >>> ut.assert_eq(gpath_list, gpath_list2, 'should not move when autolocalize is False')
        >>> # execute function
        >>> result = localize_images(ibs, gid_list_)
        >>> gpath_list3 = ibs.get_image_paths(gid_list_)
        >>> assert gpath_list3 != gpath_list2, 'should now be different'
        >>> gpath3 = gpath_list3[0]
        >>> rel_gpath3 = ut.relpath_unix(gpath3, ibs.get_workdir())
        >>> result = rel_gpath3
        >>> print(result)
        >>> # Clean things up
        >>> ibs.delete_images(gid_list_)
        testdb1/_ibsdb/images/f498fa6f-6b24-b4fa-7932-2612144fedd5.jpg

    Ignore:
        ibs.vd()

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
@accessor_decors.setter
@register_api('/api/image/uris/', methods=['PUT'])
def set_image_uris(ibs, gid_list, new_gpath_list):
    r"""
    Sets the image URIs to a new local path.
    This is used when localizing or unlocalizing images.
    An absolute path can either be on this machine or on the cloud
    A relative path is relative to the ibeis image cache on this machine.

    RESTful:
        Method: PUT
        URL:    /api/image/uris/
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((new_gpath,) for new_gpath in new_gpath_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_uri',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/contributor_rowid/', methods=['PUT'])
def set_image_contributor_rowid(ibs, gid_list, contributor_rowid_list, **kwargs):
    r"""
    Sets the image contributor rowid

    RESTful:
        Method: PUT
        URL:    /api/image/contributor_rowid/
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((contrib_rowid,) for contrib_rowid in contributor_rowid_list)
    ibs.db.set(const.IMAGE_TABLE, ('contributor_rowid',), val_list, id_iter, **kwargs)


@register_ibs_method
@accessor_decors.setter
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_imgs_reviewed_str'])
@register_api('/api/image/reviewed/', methods=['PUT'])
def set_image_reviewed(ibs, gid_list, reviewed_list):
    r"""
    Sets the image all instances found bit

    RESTful:
        Method: PUT
        URL:    /api/image/reviewed/
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((reviewed,) for reviewed in reviewed_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_toggle_reviewed',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/enabled/', methods=['PUT'])
def set_image_enabled(ibs, gid_list, enabled_list):
    r"""
    Sets the image all instances found bit

    RESTful:
        Method: PUT
        URL:    /api/image/enabled/
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((enabled,) for enabled in enabled_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_toggle_enabled',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/notes/', methods=['PUT'])
def set_image_notes(ibs, gid_list, notes_list):
    r"""
    Sets the image all instances found bit

    RESTful:
        Method: PUT
        URL:    /api/image/notes/
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((notes,) for notes in notes_list)
    ibs.db.set(const.IMAGE_TABLE, ('image_note',), val_list, id_iter)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/unixtime/', methods=['PUT'])
def set_image_unixtime(ibs, gid_list, unixtime_list, duplicate_behavior='error'):
    r"""
    Sets the image unixtime (does not modify exif yet)
        alias for set_image_time_posix

    RESTful:
        Method: PUT
        URL:    /api/image/unixtime/
    """
    id_iter = ((gid,) for gid in gid_list)
    val_list = ((unixtime,) for unixtime in unixtime_list)
    ibs.db.set(const.IMAGE_TABLE, (IMAGE_TIME_POSIX,), val_list, id_iter, duplicate_behavior=duplicate_behavior)


@register_ibs_method
@register_api('/api/image/time_posix/', methods=['PUT'])
def set_image_time_posix(ibs, image_rowid_list, image_time_posix_list, duplicate_behavior='error'):
    r"""
    image_time_posix_list -> image.image_time_posix[image_rowid_list]

    SeeAlso:
        set_image_unixtime

    Args:
        image_rowid_list
        image_time_posix_list

    TemplateInfo:
        Tsetter_native_column
        tbl = image
        col = image_time_posix

    RESTful:
        Method: PUT
        URL:    /api/image/time_posix/
    """
    id_iter = image_rowid_list
    colnames = (IMAGE_TIME_POSIX,)
    ibs.db.set(const.IMAGE_TABLE, colnames, image_time_posix_list,
               id_iter, duplicate_behavior=duplicate_behavior)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/enctext/', methods=['PUT'])
def set_image_enctext(ibs, gid_list, enctext_list):
    r"""
    Sets the encoutertext of each image

    RESTful:
        Method: PUT
        URL:    /api/image/enctext/
    """
    # FIXME: Slow and weird
    if ut.VERBOSE:
        print('[ibs] setting %r image encounter ids (from text)' % len(gid_list))
    eid_list = ibs.add_encounters(enctext_list)
    ibs.set_image_eids(gid_list, eid_list)


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/eids/', methods=['PUT'])
def set_image_eids(ibs, gid_list, eid_list):
    r"""
    Sets the encoutertext of each image

    RESTful:
        Method: PUT
        URL:    /api/image/eids/
    """
    if ut.VERBOSE:
        print('[ibs] setting %r image encounter ids' % len(gid_list))
    egrid_list = ibs.add_image_relationship(gid_list, eid_list)
    del egrid_list


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/gps/', methods=['PUT'])
def set_image_gps(ibs, gid_list, gps_list=None, lat_list=None, lon_list=None):
    r"""
    see get_image_gps for how the gps_list should look.
        lat and lon should be given in degrees

    RESTful:
        Method: PUT
        URL:    /api/image/gps/
    """
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
@accessor_decors.getter_1to1
def get_images(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of images in numpy matrix form by gid

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):

    Returns:
        list: image_list

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-get_images

    RESTful:
        Returns the base64 encoded image of image <gid>  # Documented and routed in ibeis.web app.py
        Method: GET
        URL:    /api/image/<gid>

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> # execute function
        >>> image_list = get_images(ibs, gid_list)
        >>> # verify results
        >>> result = str(image_list[0].shape)
        >>> print(result)
        (715, 1047, 3)
    """
    from vtool import image as gtool
    gpath_list = ibs.get_image_paths(gid_list)
    image_list = [gtool.imread(gpath) for gpath in gpath_list]
    return image_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/thumbtup/', methods=['GET'])
def get_image_thumbtup(ibs, gid_list, draw_annots=True, thumbsize=None):
    r"""
    Returns:
        list: thumbtup_list - [(thumb_path, img_path, imgsize, bboxes, thetas)]

    RESTful:
        Method: GET
        URL:    /api/image/thumbtup/
    """
    thumbsize = ibs.get_image_thumbsize(thumbsize, draw_annots)
    # print('gid_list = %r' % (gid_list,))
    aids_list = ibs.get_image_aids(gid_list)
    bboxes_list = ibs.unflat_map(ibs.get_annot_bboxes, aids_list)
    thetas_list = ibs.unflat_map(ibs.get_annot_thetas, aids_list)
    thumb_gpaths = ibs.get_image_thumbpath_(gid_list, draw_annots=draw_annots,
                                            thumbsize=thumbsize)
    image_paths = ibs.get_image_paths(gid_list)
    gsize_list = ibs.get_image_sizes(gid_list)
    thumbtup_list = [
        (thumb_path, img_path, img_size, bboxes, thetas)
        for thumb_path, img_path, img_size, bboxes, thetas in
        zip(thumb_gpaths, image_paths, gsize_list, bboxes_list, thetas_list)
    ]
    return thumbtup_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/thumbpath/', methods=['GET'])
def get_image_thumbpath(ibs, gid_list, ensure_paths=False, draw_annots=True,
                        thumbsize=None):
    r"""
    Returns:
        list_ (list): the thumbnail path of each gid

    RESTful:
        Method: GET
        URL:    /api/image/thumbpath/
    """
    if ensure_paths:
        thumbpath_list = ibs.preprocess_image_thumbs(gid_list,
                                                     draw_annots=draw_annots,
                                                     thumbsize=thumbsize)
    else:
        thumbpath_list = get_image_thumbpath_(ibs, gid_list, draw_annots=True, thumbsize=thumbsize)
    return thumbpath_list


@register_ibs_method
def get_image_thumbsize(ibs, thumbsize=None, draw_annots=True):
    if thumbsize is None:
        if draw_annots:
            thumbsize = ibs.cfg.other_cfg.thumb_size
        else:
            thumbsize = ibs.cfg.other_cfg.thumb_bare_size
    return thumbsize


@register_ibs_method
@accessor_decors.getter_1to1
def get_image_thumbpath_(ibs, gid_list, draw_annots=True, thumbsize=None):
    """ get_image_thumbpath, but never will ensure existence """
    thumbsize = ibs.get_image_thumbsize(thumbsize, draw_annots)
    thumb_dpath = ibs.get_thumbdir()
    img_uuid_list = ibs.get_image_uuids(gid_list)
    if draw_annots:
        thumb_suffix = '_' + str(thumbsize) + const.IMAGE_THUMB_SUFFIX
    else:
        thumb_suffix = '_' + str(thumbsize) + const.IMAGE_BARE_THUMB_SUFFIX
    thumbpath_list = [join(thumb_dpath, const.__STR__(uuid) + thumb_suffix)
                      for uuid in img_uuid_list]
    return thumbpath_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/uuids/', methods=['GET'])
def get_image_uuids(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of image uuids by gid

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):

    Returns:
        list: image_uuid_list

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-get_image_uuids

    RESTful:
        Method: GET
        URL:    /api/image/uuids/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> # execute function
        >>> image_uuid_list = ibs.get_image_uuids(gid_list)
        >>> # verify results
        >>> result = ut.list_str(image_uuid_list)
        >>> print(result)
        [
            UUID('66ec193a-1619-b3b6-216d-1784b4833b61'),
            UUID('d8903434-942f-e0f5-d6c2-0dcbe3137bf7'),
            UUID('b73b72f4-4acb-c445-e72c-05ce02719d3d'),
            UUID('0cd05978-3d83-b2ee-2ac9-798dd571c3b3'),
            UUID('0a9bc03d-a75e-8d14-0153-e2949502aba7'),
            UUID('2deeff06-5546-c752-15dc-2bd0fdb1198a'),
            UUID('a9b70278-a936-c1dd-8a3b-bc1e9a998bf0'),
            UUID('42fdad98-369a-2cbc-67b1-983d6d6a3a60'),
            UUID('c459d381-fd74-1d99-6215-e42e3f432ea9'),
            UUID('33fd9813-3a2b-774b-3fcc-4360d1ae151b'),
            UUID('97e8ea74-873f-2092-b372-f928a7be30fa'),
            UUID('588bc218-83a5-d400-21aa-d499832632b0'),
            UUID('163a890c-36f2-981e-3529-c552b6d668a3'),
        ]
        """
    image_uuid_list = ibs.db.get(const.IMAGE_TABLE, ('image_uuid',), gid_list)
    return image_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/contributor_rowid/', methods=['GET'])
def get_image_contributor_rowid(ibs, image_rowid_list, eager=True, nInput=None):
    r"""
    contributor_rowid_list <- image.contributor_rowid[image_rowid_list]

    gets data from the "native" column "contributor_rowid" in the "image" table

    Args:
        image_rowid_list (list):

    Returns:
        list: contributor_rowid_list - list of image contributor rowids by gid

    TemplateInfo:
        Tgetter_table_column
        col = contributor_rowid
        tbl = image

    RESTful:
        Method: GET
        URL:    /api/image/contributor_rowid/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> image_rowid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> contributor_rowid_list = ibs.get_image_contributor_rowid(image_rowid_list, eager=eager)
        >>> assert len(image_rowid_list) == len(contributor_rowid_list)
    """
    id_iter = image_rowid_list
    colnames = (CONTRIBUTOR_ROWID,)
    contributor_rowid_list = ibs.db.get(
        const.IMAGE_TABLE, colnames, id_iter, id_colname='rowid', eager=eager, nInput=nInput)
    return contributor_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/exts/', methods=['GET'])
def get_image_exts(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of image uuids by gid

    RESTful:
        Method: GET
        URL:    /api/image/exts/
    """
    image_uuid_list = ibs.db.get(const.IMAGE_TABLE, ('image_ext',), gid_list)
    return image_uuid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/uris/', methods=['GET'])
def get_image_uris(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of image uris relative to the image dir by gid

    RESTful:
        Method: GET
        URL:    /api/image/uris/
    """
    uri_list = ibs.db.get(const.IMAGE_TABLE, ('image_uri',), gid_list)
    return uri_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/gids_from_uuid/', methods=['GET'])
def get_image_gids_from_uuid(ibs, uuid_list):
    r"""
    Returns:
        list_ (list): a list of original image names

    RESTful:
        Method: GET
        URL:    /api/image/gids_from_uuid/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    gid_list = ibs.db.get(const.IMAGE_TABLE, ('image_rowid',), uuid_list, id_colname='image_uuid')
    return gid_list

#get_image_rowid_from_uuid = get_image_gids_from_uuid


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/paths/', methods=['GET'])
def get_image_paths(ibs, gid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list): a list of image absolute paths to img_dir

    Returns:
        list: gpath_list

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-get_image_paths

    RESTful:
        Method: GET
        URL:    /api/image/paths/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> #gid_list = ibs.get_valid_gids()
        >>> # execute function
        >>> #gpath_list = get_image_paths(ibs, gid_list)
        >>> new_gpath = ut.unixpath(ut.grab_test_imgpath('carl.jpg'))
        >>> new_gids = ibs.add_images([new_gpath], auto_localize=False)
        >>> new_gpath_list = get_image_paths(ibs, new_gids)
        >>> # verify results
        >>> ut.assert_eq(new_gpath, new_gpath_list[0])
        >>> result = str(new_gpath_list)
        >>> # clean up the database!
        >>> ibs.delete_images(new_gids)
        >>> print(result)
        """
    ut.assert_all_not_None(gid_list, 'gid_list', key_list=['gid_list'])
    uri_list = ibs.get_image_uris(gid_list)
    # Images should never have null uris
    # If the uri is not absolute then it is infered to be relative to ibs.imgdir
    ut.assert_all_not_None(uri_list, 'uri_list', key_list=['uri_list', 'gid_list'])
    gpath_list = [join(ibs.imgdir, uri) for uri in uri_list]
    return gpath_list

# TODO make this actually return a uri format
#get_image_absolute_uri = get_image_paths


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/detectpaths/', methods=['GET'])
def get_image_detectpaths(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of image paths resized to a constant area for detection

    RESTful:
        Method: GET
        URL:    /api/image/detectpaths/
    """
    from ibeis.model.preproc import preproc_detectimg
    new_gfpath_list = preproc_detectimg.compute_and_write_detectimg_lazy(ibs, gid_list)
    return new_gfpath_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/gnames/', methods=['GET'])
def get_image_gnames(ibs, gid_list):
    r"""
    Args:
        gid_list (list):

    Returns:
        list: gname_list - a list of original image names

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-get_image_gnames

    RESTful:
        Method: GET
        URL:    /api/image/gnames/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> # execute function
        >>> gname_list = get_image_gnames(ibs, gid_list)
        >>> # verify results
        >>> result = ut.list_str(gname_list)
        >>> print(result)
        [
            u'easy1.JPG',
            u'easy2.JPG',
            u'easy3.JPG',
            u'hard1.JPG',
            u'hard2.JPG',
            u'hard3.JPG',
            u'jeff.png',
            u'lena.jpg',
            u'occl1.JPG',
            u'occl2.JPG',
            u'polar1.jpg',
            u'polar2.jpg',
            u'zebra.jpg',
        ]
    """
    gname_list = ibs.db.get(const.IMAGE_TABLE, ('image_original_name',), gid_list)
    return gname_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/sizes/', methods=['GET'])
def get_image_sizes(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of (width, height) tuples

    RESTful:
        Method: GET
        URL:    /api/image/sizes/
    """
    gsize_list = ibs.db.get(const.IMAGE_TABLE, ('image_width', 'image_height'), gid_list)
    return gsize_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/widths/', methods=['GET'])
def get_image_widths(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of (width, height) tuples

    RESTful:
        Method: GET
        URL:    /api/image/widths/
    """
    gwidth_list = ibs.db.get(const.IMAGE_TABLE, ('image_width',), gid_list)
    return gwidth_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/heights/', methods=['GET'])
def get_image_heights(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of (width, height) tuples

    RESTful:
        Method: GET
        URL:    /api/image/heights/
    """
    gheight_list = ibs.db.get(const.IMAGE_TABLE, ('image_height',), gid_list)
    return gheight_list


@register_ibs_method
@ut.accepts_numpy
@accessor_decors.getter_1to1
@register_api('/api/image/unixtime/', methods=['GET'])
def get_image_unixtime(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of times that the images were taken by gid.

    Returns:
        list_ (list): -1 if no timedata exists for a given gid

    RESTful:
        Method: GET
        URL:    /api/image/unixtime/
    """
    return ibs.db.get(const.IMAGE_TABLE, ('image_time_posix',), gid_list)


@register_ibs_method
@accessor_decors.getter_1to1
def get_image_datetime(ibs, gid_list):
    unixtime_list = ibs.get_image_unixtime(gid_list)
    datetime_list = list(map(ut.unixtime_to_datetimestr, unixtime_list))
    return datetime_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/gps/', methods=['GET'])
def get_image_gps(ibs, gid_list):
    r"""
    Returns:
        gps_list (list): -1 if no timedata exists for a given gid

    RESTful:
        Method: GET
        URL:    /api/image/gps/
    """
    gps_list = ibs.db.get(const.IMAGE_TABLE, ('image_gps_lat', 'image_gps_lon'), gid_list)
    return gps_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/lat/', methods=['GET'])
def get_image_lat(ibs, gid_list):
    r"""
    Auto-docstr for 'get_image_lat'

    RESTful:
        Method: GET
        URL:    /api/image/lat/
    """
    lat_list = ibs.db.get(const.IMAGE_TABLE, ('image_gps_lat',), gid_list)
    return lat_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/lon/', methods=['GET'])
def get_image_lon(ibs, gid_list):
    r"""
    Auto-docstr for 'get_image_lon'

    RESTful:
        Method: GET
        URL:    /api/image/lon/
    """
    lon_list = ibs.db.get(const.IMAGE_TABLE, ('image_gps_lon',), gid_list)
    return lon_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/enabled/', methods=['GET'])
def get_image_enabled(ibs, gid_list):
    r"""
    Returns:
        list_ (list): "Image Enabled" flag, true if the image is enabled

    RESTful:
        Method: GET
        URL:    /api/image/enabled/
    """
    enabled_list = ibs.db.get(const.IMAGE_TABLE, ('image_toggle_enabled',), gid_list)
    return enabled_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/reviewed/', methods=['GET'])
def get_image_reviewed(ibs, gid_list):
    r"""
    Returns:
        list_ (list): "All Instances Found" flag, true if all objects of interest
    (animals) have an ANNOTATION in the image

    RESTful:
        Method: GET
        URL:    /api/image/reviewed/
    """
    reviewed_list = ibs.db.get(const.IMAGE_TABLE, ('image_toggle_reviewed',), gid_list)
    return reviewed_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/detect_confidence/', methods=['GET'])
def get_image_detect_confidence(ibs, gid_list):
    r"""
    Returns:
        list_ (list): image detection confidence as the max of ANNOTATION confidences

    RESTful:
        Method: GET
        URL:    /api/image/detect_confidence/
    """
    aids_list = ibs.get_image_aids(gid_list)
    confs_list = ibs.unflat_map(ibs.get_annot_detect_confidence, aids_list)
    maxconf_list = [max(confs) if len(confs) > 0 else -1 for confs in confs_list]
    return maxconf_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/notes/', methods=['GET'])
def get_image_notes(ibs, gid_list):
    r"""
    Returns:
        list_ (list): image notes

    RESTful:
        Method: GET
        URL:    /api/image/notes/
    """
    notes_list = ibs.db.get(const.IMAGE_TABLE, ('image_note',), gid_list)
    return notes_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/nids/', methods=['GET'])
def get_image_nids(ibs, gid_list):
    r"""

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):

    Returns:
        list: nids_list - the name ids associated with an image id

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-get_image_nids

    RESTful:
        Method: GET
        URL:    /api/image/nids/

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
@accessor_decors.getter_1to1
@register_api('/api/image/species_rowids/', methods=['GET'])
def get_image_species_rowids(ibs, gid_list):
    r"""
    Returns:
        list_ (list): the name ids associated with an image id

    RESTful:
        Method: GET
        URL:    /api/image/species_rowids/
    """
    aids_list = ibs.get_image_aids(gid_list)
    species_rowid_list = ibs.get_annot_species_rowids(aids_list)
    return species_rowid_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/image/eids/', methods=['GET'])
def get_image_eids(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of encounter ids for each image by gid

    RESTful:
        Method: GET
        URL:    /api/image/eids/
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    colnames = ('encounter_rowid',)
    eids_list = ibs.db.get(const.EG_RELATION_TABLE, colnames, gid_list,
                           id_colname='image_rowid', unpack_scalars=False)
    return eids_list


@register_ibs_method
@accessor_decors.getter_1toM
@register_api('/api/image/enctext/', methods=['GET'])
def get_image_enctext(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of enctexts for each image by gid

    RESTful:
        Method: GET
        URL:    /api/image/enctext/
    """
    eids_list = ibs.get_image_eids(gid_list)
    enctext_list = ibs.unflat_map(ibs.get_encounter_text, eids_list)
    return enctext_list


ANNOT_ROWID = 'annot_rowid'
ANNOT_ROWIDS = 'annot_rowids'
IMAGE_ROWID = 'image_rowid'


@register_ibs_method
@accessor_decors.getter_1toM
@accessor_decors.cache_getter(const.IMAGE_TABLE, ANNOT_ROWIDS)
#@profile
@register_api('/api/image/aids/', methods=['GET'])
def get_image_aids(ibs, gid_list):
    r"""
    Returns:
        list_ (list): a list of aids for each image by gid

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):

    Returns:
        list: aids_list

    CommandLine:
        python -m ibeis.control.manual_image_funcs --test-get_image_aids

    RESTful:
        Method: GET
        URL:    /api/image/aids/

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
        unique_gids, groupx = vt.group_indices(gidscol)
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
@accessor_decors.getter_1toM
#@cache_getter(const.IMAGE_TABLE)
@register_api('/api/image/aids_of_species/', methods=['GET'])
def get_image_aids_of_species(ibs, gid_list, species=None):
    r"""
    Returns:
        list_ (list): a list of aids for each image by gid filtered by species

    RESTful:
        Method: GET
        URL:    /api/image/aids_of_species/
    """
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
@accessor_decors.getter_1to1
#@profile
@register_api('/api/image/num_annotations/', methods=['GET'])
def get_image_num_annotations(ibs, gid_list):
    r"""
    Returns:
        list_ (list): the number of chips in each image

    RESTful:
        Method: GET
        URL:    /api/image/num_annotations/
    """
    return list(map(len, ibs.get_image_aids(gid_list)))


@register_ibs_method
@accessor_decors.deleter
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_imgs_reviewed_str'])
@register_api('/api/image/', methods=['DELETE'])
def delete_images(ibs, gid_list, trash_images=True):
    r"""
    deletes images from the database that belong to gids

    RESTful:
        Method: DELETE
        URL:    /api/image/
    """
    if ut.NOT_QUIET:
        print('[ibs] deleting %d images' % len(gid_list))
    # Move images to trash before deleting them. #
    # TODO: only move localized images
    # TODO: ensure there are no name conflicts when using the original names
    gpath_list = ibs.get_image_paths(gid_list)
    gname_list = ibs.get_image_gnames(gid_list)
    ext_list   = ibs.get_image_exts(gid_list)
    if trash_images:
        trash_dir  = ibs.get_trashdir()
        ut.ensuredir(trash_dir)
        gpath_list2 = [join(trash_dir, gname + ext) for (gname, ext) in
                       zip(gname_list, ext_list)]
        ut.copy_list(gpath_list, gpath_list2, ioerr_ok=True, oserror_ok=True, lbl='Trashing Images')
    else:
        raise NotImplementedError('must trash images for now')
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
@accessor_decors.deleter
@register_api('/api/image/thumbs/', methods=['DELETE'])
def delete_image_thumbs(ibs, gid_list, quiet=False):
    r"""
    Removes image thumbnails from disk

    RESTful:
        Method: DELETE
        URL:    /api/image/thumbs/
    """
    # print('gid_list = %r' % (gid_list,))
    thumbpath_list = ibs.get_image_thumbpath(gid_list)
    #ut.remove_fpaths(thumbpath_list, quiet=quiet, lbl='image_thumbs')
    ut.remove_existing_fpaths(thumbpath_list, quiet=quiet, lbl='image_thumbs')


@register_ibs_method
#@accessor_decors.cache_getter(const.IMAGE_TABLE, IMAGE_TIMEDELTA_POSIX)
@register_api('/api/image/timedelta_posix/', methods=['GET'])
def get_image_timedelta_posix(ibs, image_rowid_list, eager=True):
    r"""
    image_timedelta_posix_list <- image.image_timedelta_posix[image_rowid_list]

    # TODO: INTEGRATE THIS FUNCTION. CURRENTLY OFFSETS ARE ENCODIED DIRECTLY IN UNIXTIME

    gets data from the "native" column "image_timedelta_posix" in the "image" table

    Args:
        image_rowid_list (list):

    Returns:
        list: image_timedelta_posix_list

    TemplateInfo:
        Tgetter_table_column
        col = image_timedelta_posix
        tbl = image

    RESTful:
        Method: GET
        URL:    /api/image/timedelta_posix/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> image_rowid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> image_timedelta_posix_list = ibs.get_image_timedelta_posix(image_rowid_list, eager=eager)
        >>> assert len(image_rowid_list) == len(image_timedelta_posix_list)
    """
    id_iter = image_rowid_list
    colnames = (IMAGE_TIMEDELTA_POSIX,)
    image_timedelta_posix_list = ibs.db.get(
        const.IMAGE_TABLE, colnames, id_iter, id_colname='rowid', eager=eager)
    return image_timedelta_posix_list


@register_ibs_method
@register_api('/api/image/timedelta_posix/', methods=['PUT'])
def set_image_timedelta_posix(ibs, image_rowid_list, image_timedelta_posix_list, duplicate_behavior='error'):
    r"""
    image_timedelta_posix_list -> image.image_timedelta_posix[image_rowid_list]

    Args:
        image_rowid_list
        image_timedelta_posix_list

    TemplateInfo:
        Tsetter_native_column
        tbl = image
        col = image_timedelta_posix

    RESTful:
        Method: PUT
        URL:    /api/image/timedelta_posix/
    """
    id_iter = image_rowid_list
    colnames = (IMAGE_TIMEDELTA_POSIX,)
    ibs.db.set(const.IMAGE_TABLE, colnames, image_timedelta_posix_list,
               id_iter, duplicate_behavior=duplicate_behavior)


@register_ibs_method
#@accessor_decors.cache_getter(const.IMAGE_TABLE, IMAGE_LOCATION_CODE)
@register_api('/api/image/location_codes/', methods=['GET'])
def get_image_location_codes(ibs, image_rowid_list, eager=True):
    r"""
    image_location_code_list <- image.image_location_code[image_rowid_list]

    gets data from the "native" column "image_location_code" in the "image" table

    Args:
        image_rowid_list (list):

    Returns:
        list: image_location_code_list

    TemplateInfo:
        Tgetter_table_column
        col = image_location_code
        tbl = image

    RESTful:
        Method: GET
        URL:    /api/image/location_codes/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> image_rowid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> image_location_code_list = ibs.get_image_location_codes(image_rowid_list, eager=eager)
        >>> assert len(image_rowid_list) == len(image_location_code_list)
    """
    id_iter = image_rowid_list
    colnames = (IMAGE_LOCATION_CODE,)
    image_location_code_list = ibs.db.get(
        const.IMAGE_TABLE, colnames, id_iter, id_colname='rowid', eager=eager)
    return image_location_code_list


@register_ibs_method
@register_api('/api/image/location_codes/', methods=['PUT'])
def set_image_location_codes(ibs, image_rowid_list, image_location_code_list, duplicate_behavior='error'):
    r"""
    image_location_code_list -> image.image_location_code[image_rowid_list]

    Args:
        image_rowid_list
        image_location_code_list

    TemplateInfo:
        Tsetter_native_column
        tbl = image
        col = image_location_code

    RESTful:
        Method: PUT
        URL:    /api/image/location_codes/
    """
    id_iter = image_rowid_list
    colnames = (IMAGE_LOCATION_CODE,)
    ibs.db.set(const.IMAGE_TABLE, colnames, image_location_code_list,
               id_iter, duplicate_behavior=duplicate_behavior)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/party_rowids/', methods=['GET'])
def get_image_party_rowids(ibs, image_rowid_list, eager=True, nInput=None):
    r"""
    party_rowid_list <- image.party_rowid[image_rowid_list]

    gets data from the "native" column "party_rowid" in the "image" table

    Args:
        image_rowid_list (list):

    Returns:
        list: party_rowid_list

    TemplateInfo:
        Tgetter_table_column
        col = party_rowid
        tbl = image

    RESTful:
        Method: GET
        URL:    /api/image/party_rowids/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> image_rowid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> party_rowid_list = ibs.get_image_party_rowids(image_rowid_list, eager=eager)
        >>> assert len(image_rowid_list) == len(party_rowid_list)
    """
    id_iter = image_rowid_list
    colnames = (PARTY_ROWID,)
    party_rowid_list = ibs.db.get(
        const.IMAGE_TABLE, colnames, id_iter, id_colname='rowid', eager=eager, nInput=nInput)
    return party_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/party_tag/', methods=['GET'])
def get_image_party_tag(ibs, image_rowid_list, eager=True, nInput=None):
    r"""
    party_tag_list <- image.party_tag[image_rowid_list]

    Args:
        image_rowid_list (list):

    Returns:
        list: party_tag_list

    TemplateInfo:
        Tgetter_extern
        tbl = image
        externtbl = party
        externcol = party_tag

    RESTful:
        Method: GET
        URL:    /api/image/party_tag/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> image_rowid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> party_tag_list = ibs.get_image_party_tag(image_rowid_list, eager=eager)
        >>> assert len(image_rowid_list) == len(party_tag_list)
    """
    party_rowid_list = ibs.get_image_party_rowids(
        image_rowid_list, eager=eager, nInput=nInput)
    party_tag_list = ibs.get_party_tag(
        party_rowid_list, eager=eager, nInput=nInput)
    return party_tag_list


@register_ibs_method
@accessor_decors.setter
@register_api('/api/image/party_rowids/', methods=['PUT'])
def set_image_party_rowids(ibs, image_rowid_list, party_rowid_list, duplicate_behavior='error'):
    r"""
    party_rowid_list -> image.party_rowid[image_rowid_list]

    Args:
        image_rowid_list
        party_rowid_list

    TemplateInfo:
        Tsetter_native_column
        tbl = image
        col = party_rowid

    RESTful:
        Method: PUT
        URL:    /api/image/party_rowids/
    """
    id_iter = image_rowid_list
    colnames = (PARTY_ROWID,)
    ibs.db.set(const.IMAGE_TABLE, colnames, party_rowid_list,
               id_iter, duplicate_behavior=duplicate_behavior)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/image/contributor_tag/', methods=['GET'])
def get_image_contributor_tag(ibs, image_rowid_list, eager=True, nInput=None):
    r"""
    contributor_tag_list <- image.contributor_tag[image_rowid_list]

    Args:
        image_rowid_list (list):

    Returns:
        list: contributor_tag_list

    TemplateInfo:
        Tgetter_extern
        tbl = image
        externtbl = contributor
        externcol = contributor_tag

    RESTful:
        Method: GET
        URL:    /api/image/contributor_tag/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_image_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> image_rowid_list = ibs._get_all_image_rowids()
        >>> eager = True
        >>> contributor_tag_list = ibs.get_image_contributor_tag(image_rowid_list, eager=eager)
        >>> assert len(image_rowid_list) == len(contributor_tag_list)
    """
    contributor_rowid_list = ibs.get_image_contributor_rowid(
        image_rowid_list, eager=eager, nInput=nInput)
    contributor_tag_list = ibs.get_contributor_tag(
        contributor_rowid_list, eager=eager, nInput=nInput)
    return contributor_tag_list


def testdata_ibs():
    r"""
    Auto-docstr for 'testdata_ibs'
    """
    import ibeis
    ibs = ibeis.opendb('testdb1')
    config2_ = None
    return ibs, config2_


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.control.manual_image_funcs
        python -m ibeis.control.manual_image_funcs --allexamples
        python -m ibeis.control.manual_image_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
