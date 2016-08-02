# -*- coding: utf-8 -*-
"""
python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'ibeis.templates.template_generator')"
sh Tgen.sh --key chip --Tcfg with_setters=False with_getters=True  with_adders=True --modfname manual_chip_funcs
sh Tgen.sh --key chip

python -m utool.util_inspect --exec-check_module_usage --pat="manual_chip_funcs.py"
"""
from __future__ import absolute_import, division, print_function
import numpy as np  # NOQA
from six.moves import zip
import six  # NOQA
from os.path import join
from ibeis import constants as const
#from ibeis.control import accessor_decors
from ibeis.control import accessor_decors, controller_inject
import utool as ut
from ibeis.control.controller_inject import make_ibs_register_decorator
print, rrr, profile = ut.inject2(__name__, '[manual_chips]')


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

NEW_DEPC = True


@register_ibs_method
@accessor_decors.getter_1to1
# register_api('/api/chip/fpath/', methods=['GET'])
def get_annot_chip_fpath(ibs, aid_list, ensure=True, config2_=None,
                         check_external_storage=False, num_retries=1):
    r"""
    Returns the cached chip uri based off of the current
    configuration.

    Returns:
        chip_fpath_list (list): cfpaths defined by ANNOTATIONs

    RESTful:
        Method: GET
        URL:    /api/chip/fpath/
    """
    #import dtool
    #try:
    return ibs.depc_annot.get('chips', aid_list, 'img', config=config2_,
                              ensure=ensure, read_extern=False)
    #except dtool.ExternalStorageException:
    #    # TODO; this check might go in dtool itself
    #    return ibs.depc_annot.get('chips', aid_list, 'img', config=config2_,
    #                              ensure=ensure, read_extern=False)


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/chip/', methods=['GET'])
def get_annot_chips(ibs, aid_list, config2_=None, ensure=True, verbose=False, eager=True):
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
        python -m ibeis.templates.template_generator --key chip --funcname-filter '\<get_annot_chips\>'

    RESTful:
        Method: GET
        URL:    /api/chip/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:5]
        >>> config2_ = {'dim_size': 450, 'resize_dim': 'area'}
        >>> chip_list = get_annot_chips(ibs, aid_list, config2_)
        >>> chip_sum_list = list(map(np.sum, chip_list))
        >>> ut.assert_almost_eq(chip_sum_list, [96053500, 65152954, 67223241, 109358624, 73995960], 2000)
        >>> print(chip_sum_list)
    """
    return ibs.depc_annot.get('chips', aid_list, 'img', config=config2_,
                              ensure=ensure)


@register_ibs_method
@accessor_decors.getter_1to1
#@cache_getter(const.ANNOTATION_TABLE, 'chipsizes')
# @register_api('/api/chip/sizes/', methods=['GET'])
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:3]
        >>> ensure = True
        >>> config2_ = {'dim_size': 450, 'resize_dim': 'area'}
        >>> # execute function
        >>> chipsz_list = get_annot_chip_sizes(ibs, aid_list, ensure, config2_=config2_)
        >>> # verify results
        >>> result = str(chipsz_list)
        >>> print(result)
        [(545, 372), (603, 336), (520, 390)]
    """
    return ibs.depc_annot.get('chips', aid_list, ('width', 'height'), config=config2_, ensure=ensure)


@register_ibs_method
# @register_api('/api/chip/dlensqrd/', methods=['GET'])
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
        URL:    /api/chip/dlensqrd/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> config2_ = {'dim_size': 450, 'resize_dim': 'area'}
        >>> topx2_dlen_sqrd = ibs.get_annot_chip_dlensqrd(aid_list, config2_=config2_)
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
# @register_api('/api/chip/thumbpath/', methods=['GET'])
def get_annot_chip_thumbpath(ibs, aid_list, thumbsize=None, config2_=None):
    r"""
    just constructs the path. does not compute it. that is done by
    api_thumb_delegate

    RESTful:
        Method: GET
        URL:    /api/chip/thumbpath/
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
# @register_api('/api/chip/thumbtup/', methods=['GET'])
def get_annot_chip_thumbtup(ibs, aid_list, thumbsize=None, config2_=None):
    r"""
    get chip thumb info
    The return type of this is interpreted and computed in
    ~/code/guitool/guitool/api_thumb_delegate.py

    Args:
        aid_list  (list):
        thumbsize (int):

    Returns:
        list: thumbtup_list - [(thumb_path, img_path, imgsize, bboxes, thetas)]

    CommandLine:
        python -m ibeis.control.manual_chip_funcs --test-get_annot_chip_thumbtup

    RESTful:
        Method: GET
        URL:    /api/chip/thumbtup/

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
    #cid_list = ibs.get _annot_chip_rowids(aid_list, ensure=True)  # NOQA
    #thumbsize = 256
    if thumbsize is None:
        thumbsize = ibs.cfg.other_cfg.thumb_size
    thumb_gpaths = ibs.get_annot_chip_thumbpath(aid_list, thumbsize=thumbsize,
                                                config2_=config2_)
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
# @register_api('/api/chip/', methods=['DELETE'])
def delete_annot_chips(ibs, aid_list, config2_=None):
    r"""
    Clears annotation data (does not remove the annotation)

    RESTful:
        Method: DELETE
        URL:    /api/chip/
    """
    thumbpath_list = ibs.get_annot_chip_thumbpath(aid_list)
    #print(thumbpath_list)
    #ut.remove_fpaths(thumbpath_list, quiet=quiet, lbl='chip_thumbs')
    ut.remove_existing_fpaths(thumbpath_list, quiet=False, lbl='chip_thumbs')
    ibs.depc_annot.delete_property('chips', aid_list, config=config2_)
    return


# ---------------------
# NATIVE CHIP FUNCTIONS
# ---------------------


#@register_ibs_method
#@accessor_decors.getter_1to1
#@register_api('/api/chip/aids/', methods=['GET'])
#def get_chip_aids(ibs, cid_list):
#    r"""

#    RESTful:
#        Method: GET
#        URL:    /api/chip/aids/

#    Args:
#        ibs (IBEISController):  ibeis controller object
#        cid_list (list):

#    Returns:
#        int: aid_list -  list of annotation ids

#    CommandLine:
#        python -m ibeis.control.manual_chip_funcs --test-get_chip_aids

#    Example:
#        >>> # ENABLE_DOCTEST
#        >>> from ibeis.control.manual_chip_funcs import *  # NOQA
#        >>> import ibeis
#        >>> from ibeis.algo import Config
#        >>> from ibeis.algo.hots import query_params
#        >>> #chip_config2_ = Config.ChipConfig(dim_size=450)
#        >>> #chip_config3_ = Config.ChipConfig(dim_size=200)
#        >>> ibs = ibeis.opendb(defaultdb='testdb1')
#        >>> config2_ = query_params.QueryParams(cfgdict=dict(dim_size=450))
#        >>> config3_ = query_params.QueryParams(cfgdict=dict(dim_size=200))
#        >>> aid_list = ibs.get_valid_aids()[0:2]
#        >>> cid_list2 = ibs.get_annot_chip_rowids(aid_list, config2_=config2_)
#        >>> cid_list3 = ibs.get_annot_chip_rowids(aid_list, config2_=config3_)
#        >>> aid_list2 = get_chip_aids(ibs, cid_list2)
#        >>> aid_list3 = get_chip_aids(ibs, cid_list3)
#        >>> assert aid_list2 == aid_list3
#        >>> assert cid_list2 != cid_list3
#        >>> result  = ('cid_list2 = %s\n' % (str(cid_list2),))
#        >>> result += ('cid_list3 = %s' % (str(cid_list3),))
#        >>> ibs.get_chip_config_rowid(config2_)
#        >>> ibs.get_chip_config_rowid(config3_)
#        >>> # Extra testing
#        >>> # Delete the fpath and the annotations
#        >>> cfpath_list2 = ibs.get_chip_fpath(cid_list2)
#        >>> ut.remove_file_list(cfpath_list2)
#        >>> ibs.delete_annot_feats(aid_list)
#        >>> # Trying to get the vecs should fail because the chip is not there.
#        >>> # But trying it again will work.
#        >>> assert not any([ut.checkpath(cfpath) for cfpath in cfpath_list2])
#        >>> try:
#        >>>     fids1 = ibs.get_annot_feat_rowids(aid_list, config2_=config2_, num_retries=0)
#        >>> except controller_inject.ExternalStorageException:
#        >>>     fids1 = ibs.get_annot_feat_rowids(aid_list, config2_=config2_, num_retries=0)
#        >>> else:
#        >>>     assert False, 'Should have gotten external storage execpetion'
#        >>> print(result)
#    """
#    if NEW_DEPC:
#        raise NotImplementedError('')
#        #ibs.depc_annot['chips'].delete_rows('chips', cid_list)
#    aid_list = ibs.dbcache.get(const.CHIP_TABLE, (ANNOT_ROWID,), cid_list)
#    return aid_list


#@register_ibs_method
#@accessor_decors.getter_1to1
#def check_chip_external_storage(ibs, cid_list):
#    if NEW_DEPC:
#        raise NotImplementedError('')
#    chip_fpath_list = get_chip_fpath(ibs, cid_list, check_external_storage=False)
#    notexists_flags = [not exists(cfpath) for cfpath in chip_fpath_list]
#    if any(notexists_flags):
#        invalid_cids = ut.compress(cid_list, notexists_flags)
#        print('ERROR: %d CHIPS DO NOT EXIST' % (len(invalid_cids)))
#        print('ATTEMPING TO FIX %d / %d non-existing chip paths' % (len(invalid_cids), len(cid_list)))
#        ibs.delete_chips(invalid_cids)
#        raise controller_inject.ExternalStorageException('NON-EXISTING EXTRENAL STORAGE ERROR. REQUIRES RECOMPUTE. TRY AGAIN')
#    return chip_fpath_list


def testdata_ibs():
    r"""
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
