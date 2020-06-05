# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
from wbia.control import controller_inject

print, rrr, profile = ut.inject2(__name__)

# Create dectorator to inject functions in this module into the IBEISController
CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)


@register_ibs_method
def get_annot_fgweights(ibs, aid_list, config2_=None, ensure=True):
    r"""
    Args:
        ibs (wbia.IBEISController):  image analysis api
        aid_list (list):  list of annotation rowids
        config2_ (dict): (default = None)
        ensure (bool):  eager evaluation if True(default = True)

    CommandLine:
        python -m wbia.control.manual_featweight_funcs get_annot_fgweights

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_featweight_funcs import *  # NOQA
        >>> import wbia
        >>> import numpy as np
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> aid_list = [1, 2]
        >>> config2_ = None
        >>> ensure = True
        >>> fgws_list = get_annot_fgweights(ibs, aid_list, config2_, ensure)
        >>> depth = ut.depth_profile(fgws_list)
        >>> assert np.all(np.array(depth) > [1200, 1400])
        >>> percent_ = (fgws_list[0] > .5).sum() / len(fgws_list[0])
        >>> print('Calculated percent = %0.04f' % (percent_, ))
        >>> assert percent_ > .6 and percent_ < .8, 'should be around 0.7472'
    """
    fgws_list = ibs.depc_annot.get('featweight', aid_list, 'fwg', config=config2_)
    return fgws_list


@register_ibs_method
def get_annot_fgweight_rowids(ibs, aid_list, config2_=None, ensure=True):
    r"""
    Args:
        ibs (wbia.IBEISController):  image analysis api
        aid_list (list):  list of annotation rowids
        config2_ (dict): (default = None)
        ensure (bool):  eager evaluation if True(default = True)

    CommandLine:
        python -m wbia.control.manual_featweight_funcs get_annot_fgweight_rowids
    """
    fgw_rowid_list = ibs.depc_annot.get_rowids('featweight', aid_list, config=config2_)
    return fgw_rowid_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.control.manual_featweight_funcs
        python -m wbia.control.manual_featweight_funcs --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()
    import utool as ut

    ut.doctest_funcs()
