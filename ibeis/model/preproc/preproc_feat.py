from __future__ import division, print_function
# Python
from itertools import izip
# Science
import pyhesaff
# UTool
import utool
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[preproc_feat]', DEBUG=False)


def add_feat_params_gen(ibs, cid_list):
    """ Computes features and yeilds results asynchronously """
    # TODO: Actually make this compute in parallel
    cfpath_list = ibs.get_chip_paths(cid_list)
    feat_cfg  = ibs.cfg.feat_cfg
    dict_args = feat_cfg.get_dict_args()
    # TODO: make this an async process
    for cid, cpath in izip(cid_list, cfpath_list):
        print_('.')
        kpts, desc = pyhesaff.detect_kpts(cpath, **dict_args)
        yield cid, kpts, desc
