# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import six
import utool as ut
from six.moves import zip
from os.path import join
from wbia import constants as const
from wbia.control import accessor_decors, controller_inject
from wbia.control.controller_inject import make_ibs_register_decorator

print, rrr, profile = ut.inject2(__name__)


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api = controller_inject.get_wbia_flask_api(__name__)


ANNOT_ROWID = 'annot_rowid'
CHIP_ROWID = 'chip_rowid'
FEAT_VECS = 'feature_vecs'
FEAT_KPTS = 'feature_keypoints'
FEAT_NUM_FEAT = 'feature_num_feats'
CONFIG_ROWID = 'config_rowid'

# ---------------------
# ROOT LEAF FUNCTIONS
# ---------------------

NEW_DEPC = True


@register_ibs_method
@accessor_decors.getter_1to1
# register_api('/api/chip/fpath/', methods=['GET'])
def get_annot_chip_fpath(
    ibs,
    aid_list,
    ensure=True,
    config2_=None,
    check_external_storage=False,
    num_retries=1,
):
    r"""
    Returns the cached chip uri based off of the current
    configuration.

    Returns:
        chip_fpath_list (list): cfpaths defined by ANNOTATIONs

    RESTful:
        Method: GET
        URL:    /api/chip/fpath/
    """
    return ibs.depc_annot.get(
        'chips', aid_list, 'img', config=config2_, ensure=ensure, read_extern=False
    )


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/chip/', methods=['GET'])
def get_annot_chips(ibs, aid_list, config2_=None, ensure=True, verbose=False, eager=True):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (int):  list of annotation ids
        ensure (bool):  eager evaluation if True
        config2_ (QueryRequest):  query request object with hyper-parameters

    Returns:
        list: chip_list


    CommandLine:
        python -m wbia.control.manual_chip_funcs get_annot_chips

    RESTful:
        Method: GET
        URL:    /api/chip/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_chip_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:5]
        >>> config2_ = {'dim_size': 450, 'resize_dim': 'area'}
        >>> chip_list = get_annot_chips(ibs, aid_list, config2_)
        >>> chip_sum_list = [chip.sum() for chip in chip_list]
        >>> target = [96053500, 65152954, 67223241, 109358624, 73995960]
        >>> ut.assert_almost_eq(chip_sum_list, target, 2000)
        >>> print(chip_sum_list)
    """
    return ibs.depc_annot.get('chips', aid_list, 'img', config=config2_, ensure=ensure)


@register_ibs_method
@accessor_decors.getter_1to1
# @cache_getter(const.ANNOTATION_TABLE, 'chipsizes')
# @register_api('/api/chip/sizes/', methods=['GET'])
def get_annot_chip_sizes(ibs, aid_list, ensure=True, config2_=None):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (int):  list of annotation ids
        ensure (bool):  eager evaluation if True

    Returns:
        list: chipsz_list - the (width, height) of computed annotation chips.

    CommandLine:
        python -m wbia.control.manual_chip_funcs get_annot_chip_sizes

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_chip_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
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
    return ibs.depc_annot.get(
        'chips', aid_list, ('width', 'height'), config=config2_, ensure=ensure
    )


@register_ibs_method
def get_annot_chip_dlensqrd(ibs, aid_list, config2_=None):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (list):

    Returns:
        list: topx2_dlen_sqrd

    CommandLine:
        python -m wbia.control.manual_chip_funcs get_annot_chip_dlensqrd

    RESTful:
        Method: GET
        URL:    /api/chip/dlensqrd/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_chip_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
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
    thumbpath_list = [
        join(thumb_dpath, six.text_type(uuid) + thumb_suffix) for uuid in annot_uuid_list
    ]
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
        python -m wbia.control.manual_chip_funcs --test-get_annot_chip_thumbtup

    RESTful:
        Method: GET
        URL:    /api/chip/thumbtup/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_chip_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[1:2]
        >>> thumbsize = 128
        >>> result = get_annot_chip_thumbtup(ibs, aid_list, thumbsize)
        >>> print(result)
    """
    # isiterable = isinstance(aid_list, (list, tuple, np.ndarray))
    # if not isiterable:
    #   aid_list = [aid_list]
    # HACK TO MAKE CHIPS COMPUTE
    # cid_list = ibs.get _annot_chip_rowids(aid_list, ensure=True)  # NOQA
    # thumbsize = 256
    if thumbsize is None:
        thumbsize = ibs.cfg.other_cfg.thumb_size
    thumb_gpaths = ibs.get_annot_chip_thumbpath(
        aid_list, thumbsize=thumbsize, config2_=config2_
    )
    # print(thumb_gpaths)
    chip_paths = ibs.get_annot_chip_fpath(aid_list, ensure=True, config2_=config2_)
    chipsize_list = ibs.get_annot_chip_sizes(aid_list, ensure=False, config2_=config2_)
    thumbtup_list = [
        (thumb_path, chip_path, chipsize, [], [], [])
        for (thumb_path, chip_path, chipsize) in zip(
            thumb_gpaths, chip_paths, chipsize_list,
        )
    ]
    # if not isiterable:
    #    return thumbtup_list[0]
    return thumbtup_list


@register_ibs_method
@accessor_decors.getter_1to1
def get_annot_chip_thumb_path2(ibs, aid_list, thumbsize=None, config=None):
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
        python -m wbia.control.manual_chip_funcs --test-get_annot_chip_thumbtup

    RESTful:
        Method: GET
        URL:    /api/chip/thumbtup/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_chip_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[1:2]
        >>> thumbsize = 128
        >>> result = get_annot_chip_thumbtup(ibs, aid_list, thumbsize)
        >>> print(result)
    """
    if thumbsize is not None:
        config = {} if config is None else config.copy()
        config['thumbsize'] = thumbsize
    imgpath_list = ibs.depc_annot.get(
        'chipthumb', aid_list, 'img', config=config, read_extern=False
    )
    return imgpath_list


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
    # FIXME: Should config2_ be passed down?
    # Not sure why it isn't currently
    thumbpath_list = ibs.get_annot_chip_thumbpath(aid_list)
    # print(thumbpath_list)
    # ut.remove_fpaths(thumbpath_list, quiet=quiet, lbl='chip_thumbs')
    ut.remove_existing_fpaths(thumbpath_list, quiet=False, lbl='chip_thumbs')
    ibs.depc_annot.delete_property('chips', aid_list, config=config2_)
    return


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/pchip/', methods=['GET'])
def get_part_chips(
    ibs, part_rowid_list, config2_=None, ensure=True, verbose=False, eager=True
):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        part_rowid_list (int):  list of part ids
        ensure (bool):  eager evaluation if True
        config2_ (QueryRequest):  query request object with hyper-parameters

    Returns:
        list: chip_list


    CommandLine:
        python -m wbia.control.manual_chip_funcs get_part_chips

    RESTful:
        Method: GET
        URL:    /api/pchip/

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_chip_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> aid_list = aid_list[:10]
        >>> bbox_list = ibs.get_annot_bboxes(aid_list)
        >>> bbox_list = [
        >>>     (xtl + 100, ytl + 100, w - 100, h - 100)
        >>>     for xtl, ytl, w, h in bbox_list
        >>> ]
        >>> part_rowid_list = ibs.add_parts(aid_list, bbox_list=bbox_list)
        >>> config2_ = {'dim_size': 450, 'resize_dim': 'area'}
        >>> chip_list = get_part_chips(ibs, part_rowid_list, config2_)
        >>> chip_sum_list = [chip.sum() for chip in chip_list]
        >>> target = [86763970, 62020065, 61333964, 111418156, 63593594, 51404427, 139395045, 84060806, 41257586, 89658838]
        >>> ut.assert_almost_eq(chip_sum_list, target, 2000)
        >>> print(chip_sum_list)
    """
    return ibs.depc_part.get(
        'pchips', part_rowid_list, 'img', config=config2_, ensure=ensure
    )


@register_ibs_method
@accessor_decors.deleter
# @register_api('/api/chip/', methods=['DELETE'])
def delete_part_chips(ibs, part_rowid_list, config2_=None):
    r"""
    Clears part data

    RESTful:
        Method: DELETE
        URL:    /api/pchip/
    """
    ibs.depc_part.delete_property('pchips', part_rowid_list, config=config2_)
    return


def testdata_ibs():
    r"""
    """
    import wbia

    ibs = wbia.opendb('testdb1')
    config2_ = None
    return ibs, config2_


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.control.manual_chip_funcs
        python -m wbia.control.manual_chip_funcs --allexamples
        python -m wbia.control.manual_chip_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
