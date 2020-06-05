# -*- coding: utf-8 -*-
"""
python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'wbia.templates.template_generator')"
sh Tgen.sh --key feat --Tcfg with_setters=False with_getters=True  with_adders=True --modfname manual_feat_funcs
sh Tgen.sh --key feat --Tcfg with_deleters=True --autogen_modname manual_feat_funcs
"""
from __future__ import absolute_import, division, print_function
import six  # NOQA
from wbia.control.accessor_decors import getter_1to1, getter_1toM, deleter
import utool as ut
from wbia.control import controller_inject

print, rrr, profile = ut.inject2(__name__)


CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)

NEW_DEPC = True

ANNOT_ROWID = 'annot_rowid'
CHIP_ROWID = 'chip_rowid'
FEAT_VECS = 'feature_vecs'
FEAT_KPTS = 'feature_keypoints'
FEAT_NUM_FEAT = 'feature_num_feats'


# ----------------
# ROOT LEAF FUNCTIONS
# ----------------


@register_ibs_method
@deleter
def delete_annot_feats(ibs, aid_list, config2_=None):
    """ annot.feat.delete(aid_list)

    Args:
        aid_list

    TemplateInfo:
        Tdeleter_rl_depenant
        root = annot
        leaf = feat

    CommandLine:
        python -m wbia.control.manual_feat_funcs --test-delete_annot_feats
        python -m wbia.control.manual_feat_funcs --test-delete_annot_feats --verb-control

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_feat_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> aid_list = ibs._get_all_aids()[:1]
        >>> fids_list = ibs.get_annot_feat_rowids(aid_list, config2_=config2_, ensure=True)
        >>> num_deleted1 = ibs.delete_annot_feats(aid_list, config2_=config2_)
        >>> ut.assert_eq(num_deleted1, len(fids_list))
        >>> num_deleted2 = ibs.delete_annot_feats(aid_list, config2_=config2_)
        >>> ut.assert_eq(num_deleted2, 0)
    """
    if ut.VERBOSE:
        print('[ibs] deleting %d annots leaf nodes' % len(aid_list))
    return ibs.depc_annot.delete_property('feat', aid_list, config=config2_)


@register_ibs_method
@getter_1to1
def get_annot_feat_rowids(
    ibs, aid_list, ensure=True, eager=True, nInput=None, config2_=None, num_retries=1
):
    """
    CommandLine:
        python -m wbia.control.manual_feat_funcs get_annot_feat_rowids --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.query_request import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids()[0:3]
        >>> config2_ = {}
        >>> ibs.delete_annot_feats(aids, config2_=config2_)  # Remove the chips
        >>> ut.remove_file_list(ibs.get_annot_chip_fpath(aids, config2_=config2_))
        >>> qfids = ibs.get_annot_feat_rowids(aids, ensure=True, config2_=config2_)
    """
    return ibs.depc_annot.get_rowids(
        'feat',
        aid_list,
        config=config2_,
        ensure=ensure,
        eager=eager,
        num_retries=num_retries,
    )


@register_ibs_method
@ut.accepts_numpy
@getter_1toM
# @cache_getter(const.ANNOTATION_TABLE, 'kpts')
def get_annot_kpts(ibs, aid_list, ensure=True, eager=True, nInput=None, config2_=None):
    """
    Args:
        aid_list (int):  list of annotation ids
        ensure (bool):  eager evaluation if True
        eager (bool):
        nInput (None):
        config2_ (QueryRequest):  query request object with hyper-parameters

    Returns:
        kpts_list (list): annotation descriptor keypoints

    CommandLine:
        python -m wbia.control.manual_feat_funcs --test-get_annot_kpts --show
        python -m wbia.control.manual_feat_funcs --test-get_annot_kpts --show --darken .9
        python -m wbia.control.manual_feat_funcs --test-get_annot_kpts --show --darken .9 --verbose
        python -m wbia.control.manual_feat_funcs --test-get_annot_kpts --show --darken .9 --verbose --no-affine-invariance
        python -m wbia.control.manual_feat_funcs --test-get_annot_kpts --show --darken .9 --verbose --no-affine-invariance --scale_max=20
        python -m wbia.control.manual_feat_funcs --test-get_annot_kpts --show --feat_type=hesaff+siam128
        ipython -i -- --show --feat_type=hesaff+siam128

    Example:
        >>> # SLOW_DOCTEST
        >>> # xdoctest: +SKIP
        >>> from wbia.control.manual_feat_funcs import *  # NOQA
        >>> import vtool as vt
        >>> import numpy as np
        >>> import wbia
        >>> import wbia.viz.interact
        >>> # build test data
        >>> qreq1_ = wbia.testdata_qreq_(defaultdb='testdb1', p=['default:RI=True'])
        >>> qreq2_ = wbia.testdata_qreq_(defaultdb='testdb1', p=['default:RI=False'])
        >>> ibs = qreq1_.ibs
        >>> aid_list = qreq1_.get_external_qaids()
        >>> with ut.Indenter('[TEST_GET_ANNOT_KPTS]'):
        ...     print('qreq1 params: ' + qreq1_.qparams.feat_cfgstr)
        ...     print('qreq2 params: ' + qreq2_.qparams.feat_cfgstr)
        ...     print('id(qreq1): ' + str(id(qreq1_)))
        ...     print('id(qreq2): ' + str(id(qreq2_)))
        ...     #print('feat_config_rowid1 = %r' % (ibs.get_feat_config_rowid(config2_=qreq1_.extern_query_config2),))
        ...     #print('feat_config_rowid2 = %r' % (ibs.get_feat_config_rowid(config2_=qreq2_.extern_query_config2),))
        >>> # Force recomputation of features
        >>> with ut.Indenter('[DELETE1]'):
        ...     ibs.delete_annot_feats(aid_list, config2_=qreq1_.extern_query_config2)
        >>> with ut.Indenter('[DELETE2]'):
        ...     ibs.delete_annot_feats(aid_list, config2_=qreq2_.extern_query_config2)
        >>> eager, ensure, nInput = True, True, None
        >>> # execute function
        >>> with ut.Indenter('[GET1]'):
        ...     kpts1_list = get_annot_kpts(ibs, aid_list, ensure, eager, nInput, qreq1_.extern_query_config2)
        >>> with ut.Indenter('[GET2]'):
        ...     kpts2_list = get_annot_kpts(ibs, aid_list, ensure, eager, nInput, qreq2_.extern_query_config2)
        >>> # verify results
        >>> assert not np.all(vt.get_oris(kpts1_list[0]) == 0)
        >>> assert np.all(vt.get_oris(kpts2_list[0]) == 0)
        >>> ut.quit_if_noshow()
        >>> #wbia.viz.viz_chip.show_chip(ibs, aid_list[0], config2_=qreq1_, ori=True)
        >>> wbia.viz.interact.interact_chip.ishow_chip(ibs, aid_list[0], config2_=qreq1_.extern_query_config2, ori=True, fnum=1)
        >>> wbia.viz.interact.interact_chip.ishow_chip(ibs, aid_list[0], config2_=qreq2_.extern_query_config2, ori=True, fnum=2)
        >>> ut.show_if_requested()
    """
    return ibs.depc_annot.get(
        'feat', aid_list, 'kpts', config=config2_, ensure=ensure, eager=eager
    )


@register_ibs_method
@getter_1toM
def get_annot_vecs(ibs, aid_list, ensure=True, eager=True, nInput=None, config2_=None):
    """
    Returns:
        vecs_list (list): annotation descriptor vectors
    """
    return ibs.depc_annot.get(
        'feat', aid_list, 'vecs', config=config2_, ensure=ensure, eager=eager
    )


@register_ibs_method
@getter_1to1
def get_annot_num_feats(
    ibs, aid_list, ensure=True, eager=True, nInput=None, config2_=None, _debug=False
):
    """
    Args:
        aid_list (list):

    Returns:
        nFeats_list (list): num descriptors per annotation

    CommandLine:
        python -m wbia.control.manual_feat_funcs --test-get_annot_num_feats

    Example:
        >>> # ENABLE_DOCTEST
        >>> # this test might fail on different machines due to
        >>> # determenism bugs in hesaff maybe? or maybe jpeg...
        >>> # in which case its hopeless
        >>> from wbia.control.manual_feat_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[0:3]
        >>> config2_ = {'dim_size': 450, 'resize_dim': 'area'}
        >>> nFeats_list = get_annot_num_feats(ibs, aid_list, ensure=True, config2_=config2_, _debug=True)
        >>> print('nFeats_list = %r' % (nFeats_list,))
        >>> assert len(nFeats_list) == 3
        >>> ut.assert_inbounds(nFeats_list[0], 1200, 1259)
        >>> ut.assert_inbounds(nFeats_list[1],  900,  922)
        >>> ut.assert_inbounds(nFeats_list[2], 1300, 1343)

    Ignore:
        depc = ibs.depc_annot
        tablename = 'feat'
        input_rowids = aid_list
        colnames = 'num_feats'
        config = config2_

    """
    return ibs.depc_annot.get(
        'feat',
        aid_list,
        'num_feats',
        config=config2_,
        ensure=ensure,
        eager=eager,
        _debug=_debug,
    )


def testdata_ibs():
    import wbia

    ibs = wbia.opendb('testdb1')
    config2_ = None
    return ibs, config2_


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.control.manual_feat_funcs
        python -m wbia.control.manual_feat_funcs --allexamples
        python -m wbia.control.manual_feat_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
