"""
Functions for chips:
    to work on autogeneration

python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'ibeis.templates.template_generator')"
sh Tgen.sh --key chip --Tcfg with_setters=False with_getters=True  with_adders=True --modfname manual_chip_funcs
sh Tgen.sh --key chip

"""
from __future__ import absolute_import, division, print_function
import numpy as np  # NOQA
from six.moves import zip, range
import six  # NOQA
import functools
from os.path import join
from ibeis import constants as const
#from ibeis.control import accessor_decors
from ibeis.control import accessor_decors, controller_inject
import utool as ut
from os.path import exists
from ibeis.control.controller_inject import make_ibs_register_decorator
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[manual_chips]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


ANNOT_ROWID   = 'annot_rowid'
CHIP_ROWID    = 'chip_rowid'
FEAT_VECS     = 'feature_vecs'
FEAT_KPTS     = 'feature_keypoints'
FEAT_NUM_FEAT = 'feature_num_feats'
CONFIG_ROWID  = 'config_rowid'

# ---------------------
# ROOT LEAF FUNCTIONS
# ---------------------


@register_ibs_method
@accessor_decors.adder
@register_api('/api/annot_chip/', methods=['POST'])
def add_annot_chips(ibs, aid_list, config2_=None, verbose=not ut.QUIET, return_num_dirty=False):
    r"""
    annot.chip.add(aid_list)

    CRITICAL FUNCTION MUST EXIST FOR ALL DEPENDANTS
    Adds / ensures / computes a dependant property

    Args:
         aid_list

    Returns:
        returns chip_rowid_list of added (or already existing chips)

    TemplateInfo:
        Tadder_pl_dependant
        parent = annot
        leaf = chip

    CommandLine:
        python -m ibeis.control.manual_chip_funcs --test-add_annot_chips

    RESTful:
        Method: POST
        URL:    /api/annot_chip/

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()[::3]
        >>> chip_rowid_list = ibs.add_annot_chips(aid_list, config2_=config2_)
        >>> assert len(chip_rowid_list) == len(aid_list)
        >>> ut.assert_all_not_None(chip_rowid_list)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()[0:10]
        >>> sub_aid_list1 = aid_list[0:6]
        >>> sub_aid_list2 = aid_list[5:7]
        >>> sub_aid_list3 = aid_list[0:7]
        >>> sub_chip_rowid_list1 = ibs.get_annot_chip_rowids(sub_aid_list1, config2_=config2_, ensure=True)
        >>> ibs.get_annot_chip_rowids(sub_aid_list1, config2_=config2_, ensure=True)
        >>> sub_chip_rowid_list1, num_dirty0 = ibs.add_annot_chips(sub_aid_list1, config2_=config2_, return_num_dirty=True)
        >>> assert num_dirty0 == 0
        >>> ut.assert_all_not_None(sub_chip_rowid_list1)
        >>> ibs.delete_annot_chips(sub_aid_list2)
        >>> #ibs.delete_annot_chip(sub_aid_list2)?
        >>> sub_chip_rowid_list3 = ibs.get_annot_chip_rowids(sub_aid_list3, config2_=config2_, ensure=False)
        >>> # Only the last two should be None
        >>> ut.assert_all_not_None(sub_chip_rowid_list3[0:5], 'sub_chip_rowid_list3[0:5])')
        >>> assert sub_chip_rowid_list3[5:7] == [None, None]
        >>> sub_chip_rowid_list3_ensured, num_dirty1 = ibs.add_annot_chips(sub_aid_list3, config2_=config2_, return_num_dirty=True)
        >>> assert num_dirty1 == 2, 'Only two params should have been computed here'
        >>> ut.assert_all_not_None(sub_chip_rowid_list3_ensured)
    """
    from ibeis.model.preproc import preproc_chip
    ut.assert_all_not_None(aid_list, ' annot_rowid_list')
    # Get requested configuration id
    config_rowid = ibs.get_chip_config_rowid(config2_=config2_)
    # Find leaf rowids that need to be computed
    initial_chip_rowid_list = get_annot_chip_rowids_(ibs, aid_list, config2_=config2_)
    # Get corresponding "dirty" parent rowids
    isdirty_list = ut.flag_None_items(initial_chip_rowid_list)
    dirty_aid_list = ut.filter_items(aid_list, isdirty_list)
    num_dirty = len(dirty_aid_list)
    num_total = len(aid_list)
    if num_dirty > 0:
        if verbose:
            fmtstr = '[add_annot_chips] adding %d / %d new chip for config_rowid=%r'
            print(fmtstr % (num_dirty, num_total, config_rowid))
        # Dependant columns do not need true from_superkey getters.
        # We can use the Tgetter_pl_dependant_rowids_ instead
        get_rowid_from_superkey = functools.partial(
            ibs.get_annot_chip_rowids_, config2_=config2_)
        proptup_gen = preproc_chip.generate_chip_properties(ibs, dirty_aid_list, config2_=config2_)
        dirty_params_iter = (
            (aid, config_rowid, chip_uri, chip_width, chip_height)
            for aid, (chip_uri, chip_width, chip_height,) in
            zip(dirty_aid_list, proptup_gen)
        )
        colnames = ['annot_rowid', 'config_rowid',
                    'chip_uri', 'chip_width', 'chip_height']
        #chip_rowid_list = ibs.dbcache.add_cleanly(const.CHIP_TABLE, colnames, dirty_params_iter, get_rowid_from_superkey)
        ibs.dbcache._add(const.CHIP_TABLE, colnames, dirty_params_iter)
        # Now that the dirty params are added get the correct order of rowids
        chip_rowid_list = get_rowid_from_superkey(aid_list)
    else:
        chip_rowid_list = initial_chip_rowid_list
    if return_num_dirty:
        return chip_rowid_list, num_dirty
    return chip_rowid_list


@register_ibs_method
@register_api('/api/annot_chip/rowids/', methods=['GET'])
def get_annot_chip_rowids(ibs, aid_list, config2_=None, ensure=True, eager=True, nInput=None, extra_tries=1, check_external_storage=False):
    r"""
    chip_rowid_list <- annot.chip.rowids[aid_list]

    get chip rowids of annot under the current state configuration
    if ensure is True, this function is equivalent to add_annot_chips

    Args:
        aid_list (list):
        ensure (bool): default false

    Returns:
        list: chip_rowid_list

    TemplateInfo:
        Tgetter_pl_dependant_rowids
        parent = annot
        leaf = chip

    Timeit:
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> # Test to see if there is any overhead to injected vs native functions
        >>> %timeit get_annot_chip_rowids(ibs, aid_list)
        >>> %timeit ibs.get_annot_chip_rowids(aid_list)

    RESTful:
        Method: GET
        URL:    /api/annot_chip/rowids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()
        >>> ensure = False
        >>> chip_rowid_list = ibs.get_annot_chip_rowids(aid_list, config2_, ensure)
        >>> assert len(chip_rowid_list) == len(aid_list)
    """
    if ensure:
        for try_num in range(extra_tries + 1):
            try:
                chip_rowid_list = add_annot_chips(ibs, aid_list, config2_=config2_)
                if check_external_storage:
                    # Chips store data externally on disk. Ensure that they are there.
                    # If this fails it will try to add again.
                    check_chip_external_storage(ibs, chip_rowid_list)
            except controller_inject.ExternalStorageException as ex:
                try_again = try_num < extra_tries
                msg = ('WILL TRY AT MOST %d MORE TIME(S)'  % (extra_tries - try_num,) if try_again else
                       'EXCEDED MAXIMUM NUMBER OF TRIES extra_tries=%d. RAISING ERROR' % (extra_tries,))
                ut.printex(ex, msg, iswarning=try_again)
                if not try_again:
                    raise
            else:
                break
    else:
        chip_rowid_list = get_annot_chip_rowids_(
            ibs, aid_list, config2_=config2_, eager=eager, nInput=nInput)
    return chip_rowid_list


@register_ibs_method
@register_api('/api/annot_chip/rowids_/', methods=['GET'])
def get_annot_chip_rowids_(ibs, aid_list, config2_=None, eager=True, nInput=None):
    r"""
    equivalent to get_annot_chip_rowids_ except ensure is constrained
    to be False.

    Also you save a stack frame because get_annot_chip_rowids just
    calls this function if ensure is False

    TemplateInfo:
        Tgetter_pl_dependant_rowids_

    RESTful:
        Method: GET
        URL:    /api/annot_chip/rowids_/
    """
    colnames = (CHIP_ROWID,)
    config_rowid = ibs.get_chip_config_rowid(config2_=config2_)
    andwhere_colnames = (ANNOT_ROWID, CONFIG_ROWID,)
    params_iter = ((aid, config_rowid,) for aid in aid_list)
    chip_rowid_list = ibs.dbcache.get_where2(
        const.CHIP_TABLE, colnames, params_iter, andwhere_colnames, eager=eager, nInput=nInput)
    return chip_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot_chip/fpath/', methods=['GET'])
def get_annot_chip_fpath(ibs, aid_list, ensure=True, config2_=None, check_external_storage=False, extra_tries=0):
    r"""
    Returns the cached chip uri based off of the current
    configuration.

    Returns:
        chip_fpath_list (list): cfpaths defined by ANNOTATIONs

    TODO:
        template this as an external storage getter

    RESTful:
        Method: GET
        URL:    /api/annot_chip/fpath/
    """
    cid_list  = ibs.get_annot_chip_rowids(aid_list, ensure=ensure, config2_=config2_, check_external_storage=check_external_storage, extra_tries=extra_tries)
    chip_fpath_list = ibs.get_chip_fpath(cid_list, check_external_storage=check_external_storage)
    return chip_fpath_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot_chip/', methods=['GET'])
def get_annot_chips(ibs, aid_list, ensure=True, config2_=None, verbose=False):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        ensure (bool):  eager evaluation if True
        config2_ (QueryRequest):  query request object with hyper-parameters

    Returns:
        list: chip_list

    CommandLine:
        python -m ibeis.control.manual_chip_funcs --test-get_annot_chips

    RESTful:
        Method: GET
        URL:    /api/annot_chip/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:5]
        >>> ensure = True
        >>> config2_ = None
        >>> chip_list = get_annot_chips(ibs, aid_list, ensure, config2_)
        >>> chip_sum_list = list(map(np.sum, chip_list))
        >>> ut.assert_almost_eq(chip_sum_list, [96053500, 65152954, 67223241, 109358624, 73995960], 2000)
        >>> print(chip_sum_list)
    """
    ut.assert_all_not_None(aid_list, 'aid_list')
    cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=ensure, config2_=config2_)
    chip_list = ibs.get_chips(cid_list, ensure=ensure, verbose=verbose)
    return chip_list


@register_ibs_method
@accessor_decors.getter_1to1
#@cache_getter(const.ANNOTATION_TABLE, 'chipsizes')
@register_api('/api/annot_chip/sizes/', methods=['GET'])
def get_annot_chip_sizes(ibs, aid_list, ensure=True, config2_=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        ensure (bool):  eager evaluation if True

    Returns:
        list: chipsz_list - the (width, height) of computed annotation chips.

    CommandLine:
        python -m ibeis.control.manual_chip_funcs --test-get_annot_chip_sizes

    RESTful:
        Method: GET
        URL:    /api/annot_chip/sizes/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:3]
        >>> ensure = True
        >>> # execute function
        >>> chipsz_list = get_annot_chip_sizes(ibs, aid_list, ensure)
        >>> # verify results
        >>> result = str(chipsz_list)
        >>> print(result)
        [(545, 372), (603, 336), (520, 390)]
    """
    cid_list  = ibs.get_annot_chip_rowids(aid_list, ensure=ensure, config2_=config2_)
    chipsz_list = ibs.get_chip_sizes(cid_list)
    return chipsz_list


@register_ibs_method
@register_api('/api/annot_chip/dlensqrd/', methods=['GET'])
def get_annot_chip_dlensqrd(ibs, aid_list, config2_=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):

    Returns:
        list: topx2_dlen_sqrd

    CommandLine:
        python -m ibeis.control.manual_chip_funcs --test-get_annot_chip_dlensqrd

    RESTful:
        Method: GET
        URL:    /api/annot_chip/dlensqrd/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> config2_ = None
        >>> # execute function
        >>> topx2_dlen_sqrd = ibs.get_annot_chip_dlensqrd(aid_list, config2_=config2_)
        >>> # verify results
        >>> result = str(topx2_dlen_sqrd)
        >>> print(result)
        [435409, 476505, 422500, 422500, 422500, 437924, 405000, 405000, 447805, 420953, 405008, 406265, 512674]
    """
    topx2_dlen_sqrd = [
        ((w ** 2) + (h ** 2))
        for (w, h) in ibs.get_annot_chip_sizes(aid_list, config2_=config2_)
    ]
    return topx2_dlen_sqrd


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot_chip/thumbpath/', methods=['GET'])
def get_annot_chip_thumbpath(ibs, aid_list, thumbsize=None, config2_=None):
    r"""
    just constructs the path. does not compute it. that is done by
    api_thumb_delegate

    RESTful:
        Method: GET
        URL:    /api/annot_chip/thumbpath/
    """
    if thumbsize is None:
        thumbsize = ibs.cfg.other_cfg.thumb_size
    thumb_dpath = ibs.thumb_dpath
    thumb_suffix = '_' + str(thumbsize) + const.CHIP_THUMB_SUFFIX
    annot_uuid_list = ibs.get_annot_visual_uuids(aid_list)
    thumbpath_list = [join(thumb_dpath, const.__STR__(uuid) + thumb_suffix)
                      for uuid in annot_uuid_list]
    return thumbpath_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/annot_chip/thumbtup/', methods=['GET'])
def get_annot_chip_thumbtup(ibs, aid_list, thumbsize=None, config2_=None):
    r"""
    get chip thumb info

    Args:
        aid_list  (list):
        thumbsize (int):

    Returns:
        list: thumbtup_list - [(thumb_path, img_path, imgsize, bboxes, thetas)]

    CommandLine:
        python -m ibeis.control.manual_chip_funcs --test-get_annot_chip_thumbtup

    RESTful:
        Method: GET
        URL:    /api/annot_chip/thumbtup/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> thumbsize = 128
        >>> result = get_annot_chip_thumbtup(ibs, aid_list, thumbsize)
        >>> print(result)
    """
    #isiterable = isinstance(aid_list, (list, tuple, np.ndarray))
    #if not isiterable:
    #   aid_list = [aid_list]
    # HACK TO MAKE CHIPS COMPUTE
    #cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=True)  # NOQA
    #thumbsize = 256
    if thumbsize is None:
        thumbsize = ibs.cfg.other_cfg.thumb_size
    thumb_gpaths = ibs.get_annot_chip_thumbpath(aid_list, thumbsize=thumbsize, config2_=config2_)
    #print(thumb_gpaths)
    chip_paths = ibs.get_annot_chip_fpath(aid_list, ensure=True, config2_=config2_)
    chipsize_list = ibs.get_annot_chip_sizes(aid_list, ensure=False, config2_=config2_)
    thumbtup_list = [
        (thumb_path, chip_path, chipsize, [], [])
        for (thumb_path, chip_path, chipsize) in
        zip(thumb_gpaths, chip_paths, chipsize_list,)
    ]
    #if not isiterable:
    #    return thumbtup_list[0]
    return thumbtup_list


@register_ibs_method
@accessor_decors.deleter
@register_api('/api/annot_chip/thumbs/', methods=['DELETE'])
def delete_annot_chip_thumbs(ibs, aid_list, quiet=False):
    r"""
    Removes chip thumbnails from disk

    RESTful:
        Method: DELETE
        URL:    /api/annot_chip/thumbs/
    """
    thumbpath_list = ibs.get_annot_chip_thumbpath(aid_list)
    #ut.remove_fpaths(thumbpath_list, quiet=quiet, lbl='chip_thumbs')
    ut.remove_existing_fpaths(thumbpath_list, quiet=quiet, lbl='chip_thumbs')


@register_ibs_method
@accessor_decors.deleter
@register_api('/api/annot_chip/', methods=['DELETE'])
def delete_annot_chips(ibs, aid_list, config2_=None):
    r"""
    Clears annotation data (does not remove the annotation)

    RESTful:
        Method: DELETE
        URL:    /api/annot_chip/
    """
    _cid_list = ibs.get_annot_chip_rowids(aid_list, ensure=False, config2_=config2_)
    cid_list = ut.filter_Nones(_cid_list)
    ibs.delete_chips(cid_list)
    # HACK FIX: if annot chips are None then the image thumbnail
    # will not be invalidated
    if len(_cid_list) != len(cid_list):
        aid_list_ = [aid for aid, _cid in zip(aid_list, _cid_list) if _cid is None]
        gid_list_ = ibs.get_annot_gids(aid_list_)
        ibs.delete_image_thumbs(gid_list_)


# ---------------------
# NATIVE CHIP FUNCTIONS
# ---------------------


@register_ibs_method
def _get_all_chip_rowids(ibs):
    r"""
    all_chip_rowids <- chip.get_all_rowids()

    Returns:
        list_ (list): unfiltered chip_rowids

    TemplateInfo:
        Tider_all_rowids
        tbl = chip

    Example:
        >>> # ENABLE_DOCTEST
        >>> ibs, config2_ = testdata_ibs()
        >>> ibs._get_all_chip_rowids()
    """
    all_chip_rowids = ibs.dbcache.get_all_rowids(const.CHIP_TABLE)
    return all_chip_rowids


@register_ibs_method
@accessor_decors.ider
@register_api('/api/chip/', methods=['GET'])
def get_valid_cids(ibs, config2_=None):
    r"""
    Valid chip rowids of the current configuration

    RESTful:
        Method: GET
        URL:    /api/chip/
    """
    # FIXME: configids need reworking
    chip_config_rowid = ibs.get_chip_config_rowid(config2_=config2_)
    #cid_list = ibs.dbcache.get_all_rowids_where(const.FEATURE_TABLE, 'config_rowid=?', (chip_config_rowid,))  # big big I belive
    cid_list = ibs.dbcache.get_all_rowids_where(const.CHIP_TABLE, 'config_rowid=?', (chip_config_rowid,))
    return cid_list


@register_ibs_method
@accessor_decors.deleter
#@cache_invalidator(const.CHIP_TABLE)
@register_api('/api/chip/', methods=['DELETE'])
def delete_chips(ibs, cid_list, verbose=ut.VERBOSE, config2_=None):
    r"""
    deletes images from the database that belong to gids

    RESTful:
        Method: DELETE
        URL:    /api/chip/
    """
    from ibeis.model.preproc import preproc_chip
    if verbose:
        print('[ibs] deleting %d annotation-chips' % len(cid_list))
    # Delete sql-external (on-disk) information
    preproc_chip.on_delete(ibs, cid_list)
    # Delete sql-dependencies
    fid_list = ut.filter_Nones(ibs.get_chip_feat_rowid(cid_list, config2_=config2_, ensure=False))
    aid_list = ibs.get_chip_aids(cid_list)
    gid_list = ibs.get_annot_gids(aid_list)
    ibs.delete_image_thumbs(gid_list)
    ibs.delete_annot_chip_thumbs(aid_list)
    ibs.delete_features(fid_list, config2_=config2_)
    # Delete chips from sql
    ibs.dbcache.delete_rowids(const.CHIP_TABLE, cid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/chip/aids/', methods=['GET'])
def get_chip_aids(ibs, cid_list):
    r"""
    Auto-docstr for 'get_chip_aids'

    RESTful:
        Method: GET
        URL:    /api/chip/aids/

    Args:
        ibs (IBEISController):  ibeis controller object
        cid_list (list):

    Returns:
        int: aid_list -  list of annotation ids

    CommandLine:
        python -m ibeis.control.manual_chip_funcs --test-get_chip_aids

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> import ibeis
        >>> from ibeis.model import Config
        >>> from ibeis.model.hots import query_params
        >>> #chip_config2_ = Config.ChipConfig(chip_sqrt_area=450)
        >>> #chip_config3_ = Config.ChipConfig(chip_sqrt_area=200)
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> config2_ = query_params.QueryParams(cfgdict=dict(chip_sqrt_area=450))
        >>> config3_ = query_params.QueryParams(cfgdict=dict(chip_sqrt_area=200))
        >>> aid_list = ibs.get_valid_aids()[0:2]
        >>> cid_list2 = ibs.get_annot_chip_rowids(aid_list, config2_=config2_)
        >>> cid_list3 = ibs.get_annot_chip_rowids(aid_list, config2_=config3_)
        >>> aid_list2 = get_chip_aids(ibs, cid_list2)
        >>> aid_list3 = get_chip_aids(ibs, cid_list3)
        >>> assert aid_list2 == aid_list3
        >>> assert cid_list2 != cid_list3
        >>> result  = ('cid_list2 = %s\n' % (str(cid_list2),))
        >>> result += ('cid_list3 = %s' % (str(cid_list3),))
        >>> ibs.get_chip_config_rowid(config2_)
        >>> ibs.get_chip_config_rowid(config3_)
        >>> # Extra testing
        >>> # Delete the fpath
        >>> cfpath_lcfpath_list2ist2 = ibs.get_chip_fpath(cid_list2)
        >>> cfpath_list2 = ibs.get_chip_fpath(cid_list2)
        >>> ut.remove_file_list(cfpath_list2)
        >>> assert not any([ut.checkpath(cfpath) for cfpath in cfpath_list2])
        >>> try:
        >>>     vecs1 = ibs.get_annot_vecs(aid_list, config2_=config2_)
        >>> except controller_inject.ExternalStorageException:
        >>>     vecs1 = ibs.get_annot_vecs(aid_list, config2_=config2_)
        >>> else:
        >>>     assert False, 'Should have gotten external storage execpetion'
        >>> print(result)
    """
    aid_list = ibs.dbcache.get(const.CHIP_TABLE, (ANNOT_ROWID,), cid_list)
    return aid_list


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/chip/config_rowid/', methods=['GET'])
def get_chip_config_rowid(ibs, config2_=None):
    r"""
    # FIXME: Configs are still handled poorly

    This method deviates from the rest of the controller methods because it
    always returns a scalar instead of a list. I'm still not sure how to
    make it more ibeisy

    TemplateInfo:
        python -m ibeis.templates.template_generator --key chip --funcname-filter '\<get_chip_config_rowid\>'
        Tcfg_rowid_getter
        leaf = chip

    RESTful:
        Method: GET
        URL:    /api/chip/config_rowid/
    """
    if config2_ is not None:
        # TODO store config_rowid in qparams
        # Or find better way to do this in general
        #chip_cfg_suffix = config2_.qparams.chip_cfgstr
        chip_cfg_suffix = config2_.get('chip_cfgstr')
        assert chip_cfg_suffix is not None
    else:
        chip_cfg_suffix = ibs.cfg.chip_cfg.get_cfgstr()
    chip_cfg_rowid = ibs.ensure_config_rowid_from_suffix(chip_cfg_suffix)
    return chip_cfg_rowid


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/chip/detectpaths/', methods=['GET'])
def get_chip_detectpaths(ibs, cid_list):
    r"""
    Returns:
        new_gfpath_list (list): a list of image paths resized to a constant area for detection

    RESTful:
        Method: GET
        URL:    /api/chip/detectpaths/
    """
    from ibeis.model.preproc import preproc_detectimg
    new_gfpath_list = preproc_detectimg.compute_and_write_detectchip_lazy(ibs, cid_list)
    return new_gfpath_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/chip/uris/', methods=['GET'])
def get_chip_uris(ibs, cid_list):
    r"""

    Returns:
        chip_uri_list (list): a list of chip paths by their aid

    RESTful:
        Method: GET
        URL:    /api/chip/uris/
    """
    chip_uri_list = ibs.dbcache.get(const.CHIP_TABLE, ('chip_uri',), cid_list)
    return chip_uri_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/chip/fpath/', methods=['GET'])
def get_chip_fpath(ibs, cid_list, check_external_storage=False):
    r"""
    Combines the uri with the expected chip directory.
    config2_ is only needed if ensure_external_storage=True

    Returns:
        chip_fpath_list (list): a list of chip paths by their aid

    RESTful:
        Method: GET
        URL:    /api/chip/fpath/
    """
    if check_external_storage:
        chip_fpath_list = check_chip_external_storage(ibs, cid_list)
    else:
        chip_uri_list = ibs.get_chip_uris(cid_list)
        chipdir = ibs.get_chipdir()
        chip_fpath_list = [
            None if chip_uri is None else ut.unixjoin(chipdir, chip_uri)
            for chip_uri in chip_uri_list
        ]
    return chip_fpath_list


@register_ibs_method
@accessor_decors.getter_1to1
def check_chip_external_storage(ibs, cid_list):
    chip_fpath_list = get_chip_fpath(ibs, cid_list, check_external_storage=False)
    notexists_flags = [not exists(cfpath) for cfpath in chip_fpath_list]
    if any(notexists_flags):
        invalid_cids = ut.list_compress(cid_list, notexists_flags)
        print('ERROR: %d CHIPS DO NOT EXIST' % (len(invalid_cids)))
        print('ATTEMPING TO FIX %d / %d non-existing chip paths' % (len(invalid_cids), len(cid_list)))
        ibs.delete_chips(invalid_cids)
        raise controller_inject.ExternalStorageException('NON-EXISTING EXTRENAL STORAGE ERROR. REQUIRES RECOMPUTE. TRY AGAIN')
    return chip_fpath_list


@register_ibs_method
@accessor_decors.getter_1to1
#@cache_getter('const.CHIP_TABLE', 'chip_size')
@register_api('/api/chip/sizes/', methods=['GET'])
def get_chip_sizes(ibs, cid_list):
    r"""
    Auto-docstr for 'get_chip_sizes'

    RESTful:
        Method: GET
        URL:    /api/chip/sizes/
    """
    chipsz_list  = ibs.dbcache.get(const.CHIP_TABLE, ('chip_width', 'chip_height',), cid_list)
    return chipsz_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_chips(ibs, cid_list, ensure=True, verbose=False):
    r"""
    Returns:
        chip_list (list): a list cropped images in numpy array form by their cid

    Args:
        cid_list (list):
        ensure (bool):  eager evaluation if True

    RESTful:
        Returns the base64 encoded image of annotation (chip) <aid>  # Documented and routed in ibeis.web app.py
        Method: GET
        URL:    /api/annot/<aid>

    Returns:
        list: chip_list
    """
    from ibeis.model.preproc import preproc_chip
    if ensure:
        try:
            ut.assert_all_not_None(cid_list, 'cid_list')
        except AssertionError as ex:
            ut.printex(ex, 'Invalid cid_list', key_list=[
                'ensure', 'cid_list'])
            raise
    aid_list = ibs.get_chip_aids(cid_list)
    chip_list = preproc_chip.compute_or_read_annotation_chips(ibs, aid_list, ensure=ensure, verbose=verbose)
    return chip_list


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
        python -m ibeis.control.manual_chip_funcs
        python -m ibeis.control.manual_chip_funcs --allexamples
        python -m ibeis.control.manual_chip_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
