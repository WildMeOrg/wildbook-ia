"""
module that specified how we choose paramaters based on current search database
properties
"""
from __future__ import absolute_import, division, print_function
#import six
import utool as ut
#import numpy as np
#import vtool as vt
#from ibeis.model.hots import hstypes
#from ibeis.model.hots import match_chips4 as mc4
#from ibeis.model.hots import distinctiveness_normalizer
#from six.moves import filter
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[autoparams]')


@profile
def choose_vsmany_K(num_names, qaids, daids):
    """
    TODO: Should also scale up the number of checks as well

    method for choosing K in the initial vsmany queries

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> # Shows plot for K vs number of names
        >>> from ibeis.model.hots.automated_params import *  # NOQA
        >>> import ibeis
        >>> from ibeis import constants as const
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> valid_aids = ibs.get_valid_aids(species=const.Species.ZEB_PLAIN)
        >>> num_names = np.arange(0, 1000)
        >>> num_names_slope = .1
        >>> K_max = 10
        >>> K_min = 1
        >>> K_list = np.floor(num_names_slope * num_names)
        >>> K_list[K_list > K_max] = K_max
        >>> K_list[K_list < K_min] = K_min
        >>> clip_index_list = np.where(K_list >= K_max)[0]
        >>> clip_index = clip_index_list[min(len(clip_index_list) - 1, 10)]
        >>> K_list = K_list[0:clip_index]
        >>> num_names = num_names[0:clip_index]
        >>> pt.plot2(num_names, K_list, x_label='num_names', y_label='K',
        ...          equal_aspect=False, marker='g-', pad=1, dark=True)
        >>> pt.update()
    """
    #K = ibs.cfg.query_cfg.nn_cfg.K
    # TODO: paramaterize in config
    num_names_slope = .1  # increase K every fifty names
    K_max = 10
    K_min = 1
    num_names_lower = K_min / num_names_slope
    num_names_upper = K_max / num_names_slope
    if num_names < num_names_lower:
        K = K_min
    elif num_names < num_names_upper:
        K = num_names_slope * num_names
    else:
        K  = K_max

    with ut.embed_on_exception_context:
        if len(ut.intersect_ordered(qaids, daids)) > 0:
            # if self is in query bump k
            K += 1
    return K


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.automated_params
        python -m ibeis.model.hots.automated_params --allexamples
        python -m ibeis.model.hots.automated_params --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
