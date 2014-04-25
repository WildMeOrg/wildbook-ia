from __future__ import absolute_import, division, print_function
# Python
from itertools import izip
# Science
import pyhesaff
# UTool
import utool
import sys
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[preproc_feat]', DEBUG=False)


def add_feat_params_gen(ibs, cid_list):
    """ Computes features and yeilds results asynchronously """
    # TODO: Actually make this compute in parallel
    cfpath_list = ibs.get_chip_paths(cid_list)
    feat_cfg  = ibs.cfg.feat_cfg
    dict_args = feat_cfg.get_dict_args()
    num_feats = len(cid_list)
    feat_config_uid = ibs.get_feat_config_uid()
    mark_prog, end_prog = utool.progress_func(num_feats, lbl='hesaff: ',
                                              flush_after=1, mark_start=True)
    # TODO: make this an async process
    sys.stdout.flush()
    for count, (cid, cpath) in enumerate(izip(cid_list, cfpath_list)):
        kpts, desc = pyhesaff.detect_kpts(cpath, **dict_args)
        mark_prog(count)
        yield cid, len(kpts), kpts, desc, feat_config_uid
    end_prog()
