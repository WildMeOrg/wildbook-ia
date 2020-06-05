# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import utool as ut  # NOQA
import numpy as np
import vtool as vt

# from wbia import constants as const
from wbia.control import accessor_decors  # NOQA
from wbia.control.controller_inject import make_ibs_register_decorator

print, rrr, profile = ut.inject2(__name__)

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


@register_ibs_method
def new_query_request(
    ibs, qaid_list, daid_list, cfgdict=None, verbose=ut.NOT_QUIET, **kwargs
):
    """
    alias for wbia.algo.hots.query_request.new_wbia_query_request

    Args:
        qaid_list (list):
        daid_list (list):
        cfgdict (None):
        verbose (bool):

    Returns:
        wbia.QueryRequest: qreq_ -  hyper-parameters
    """
    from wbia.algo.hots import query_request

    qreq_ = query_request.new_wbia_query_request(
        ibs, qaid_list, daid_list, cfgdict=cfgdict, verbose=verbose, **kwargs
    )
    return qreq_


@register_ibs_method
def get_annot_kpts_distinctiveness(ibs, aid_list, config2_=None, **kwargs):
    """
    very hacky, but cute way to cache keypoint distinctivness

    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (list):
        dstncvs_normer (None):

    Returns:
        list: dstncvs_list

    CommandLine:
        python -m wbia.control.manual_wbiacontrol_funcs --test-get_annot_kpts_distinctiveness

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from wbia.control.manual_wbiacontrol_funcs import *  # NOQA
        >>> from wbia.algo.hots import distinctiveness_normalizer
        >>> import wbia
        >>> import numpy as np
        >>> config2_ = None
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids(species=const.TEST_SPECIES.ZEB_PLAIN)
        >>> # execute function
        >>> aid_list1 = aid_list[::2]
        >>> aid_list2 = aid_list[1::3]
        >>> dstncvs_list1 = get_annot_kpts_distinctiveness(ibs, aid_list1)
        >>> dstncvs_list2 = get_annot_kpts_distinctiveness(ibs, aid_list2)
        >>> dstncvs_list = get_annot_kpts_distinctiveness(ibs, aid_list)
        >>> print(ut.depth_profile(dstncvs_list1))
        >>> stats_dict = ut.dict_stack([ut.get_stats(dstncvs) for dstncvs in dstncvs_list])
        >>> print(ut.repr2(stats_dict))
        >>> assert np.all(np.array(stats_dict['min']) >= 0), 'distinctiveness was out of bounds'
        >>> assert np.all(np.array(stats_dict['max']) <= 1), 'distinctiveness was out of bounds'
    """
    from wbia.algo.hots import distinctiveness_normalizer as dcvs_normer

    # per-species disinctivness wrapper around wbia cached function
    # get feature rowids
    aid_list = np.array(aid_list)
    fid_list = np.array(
        ibs.get_annot_feat_rowids(
            aid_list, ensure=True, eager=True, nInput=None, config2_=config2_
        )
    )
    species_rowid_list = np.array(ibs.get_annot_species_rowids(aid_list))
    # Compute distinctivness separately for each species
    unique_sids, groupxs = vt.group_indices(species_rowid_list)
    fids_groups = vt.apply_grouping(fid_list, groupxs)
    species_text_list = ibs.get_species_texts(unique_sids)
    # Map distinctivness computation
    normer_list = [
        dcvs_normer.request_species_distinctiveness_normalizer(species)
        for species in species_text_list
    ]
    # Reduce to get results
    dstncvs_groups = [
        get_feat_kpts_distinctiveness(
            ibs, fids, dstncvs_normer=dstncvs_normer, species_rowid=sid, **kwargs
        )
        for dstncvs_normer, fids, sid in zip(normer_list, fids_groups, unique_sids)
    ]
    dstncvs_list = vt.invert_apply_grouping(dstncvs_groups, groupxs)
    return dstncvs_list


# dcvs_cfgkeys = dcvs_normer.DCVS_DEFAULT.get_varnames() + ['species_rowid']
# dcvs_colname = dcvs_normer.DCVS_DEFAULT.name

# @accessor_decors.cache_getter(const.FEATURE_TABLE, dcvs_colname, cfgkeys=dcvs_cfgkeys, debug=None)
def get_feat_kpts_distinctiveness(
    ibs, fid_list, dstncvs_normer=None, species_rowid=None, **kwargs
):
    # print('[ibs] get_feat_kpts_distinctiveness fid_list=%r' % (fid_list,))
    vecs_list = ibs.get_feat_vecs(fid_list, eager=True, nInput=None)
    dstncvs_list = [
        None if vecs is None else dstncvs_normer.get_distinctiveness(vecs, **kwargs)
        for vecs in vecs_list
    ]
    return dstncvs_list


@register_ibs_method
def show_annot(ibs, aid, *args, **kwargs):
    """ viz helper see wbia.viz.viz_chip.show_chip """
    from wbia.viz import viz_chip

    return viz_chip.show_chip(ibs, aid, *args, **kwargs)


@register_ibs_method
def show_annot_image(ibs, aid, *args, **kwargs):
    """ viz helper see wbia.viz.viz_chip.show_chip """
    from wbia.viz import viz_image

    gid = ibs.get_annot_gids(aid)
    return viz_image.show_image(ibs, gid, *args, **kwargs)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.control.manual_wbiacontrol_funcs
        python -m wbia.control.manual_wbiacontrol_funcs --allexamples
        python -m wbia.control.manual_wbiacontrol_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
