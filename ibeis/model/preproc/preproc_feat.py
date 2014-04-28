from __future__ import absolute_import, division, print_function
# Python
from itertools import izip
# Science
import pyhesaff
# UTool
import utool
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[preproc_feat]', DEBUG=False)


def gen_feat(cid, cpath, dict_args=None, feat_config_uid=None):
    kpts, desc = pyhesaff.detect_kpts(cpath, **dict_args)
    return cid, len(kpts), kpts, desc, feat_config_uid


def gen_feat2(tup):
    cid, cpath, dict_args, feat_config_uid = tup
    kpts, desc = pyhesaff.detect_kpts(cpath, **dict_args)
    return cid, len(kpts), kpts, desc, feat_config_uid


def add_feat_params_gen(ibs, cid_list):
    """ Computes features and yields results asynchronously """
    # TODO: Actually make this compute in parallel
    feat_cfg  = ibs.cfg.feat_cfg
    dict_args = feat_cfg.get_dict_args()
    feat_config_uid = ibs.get_feat_config_uid()
    cfpath_list = ibs.get_chip_paths(cid_list)
    #num_feats = len(cid_list)
    #mark_prog, end_prog = utool.progress_func(num_feats, lbl='hesaff: ',
                                              #flush_after=1, mark_start=True)
    # TODO: make this an async process
    #sys.stdout.flush()
    #for count, (cid, cpath) in enumerate(izip(cid_list, cfpath_list)):
        #kpts, desc = pyhesaff.detect_kpts(cpath, **dict_args)
        #mark_prog(count)
        #yield cid, len(kpts), kpts, desc, feat_config_uid
    #end_prog()
    util_parallel = utool.util_parallel
    #arg_list = list(izip(cid_list, cfpath_list))
    #args_dict = {'feat_config_uid': feat_config_uid,
                 #'dict_args': dict_args}
    arg_list = [(list(tup) + [dict_args, feat_config_uid]) for tup in izip(cid_list, cfpath_list)]
    return util_parallel.generate(gen_feat2, arg_list)
