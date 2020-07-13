# -*- coding: utf-8 -*-
# TODO: find unused functions and kill them
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import copy
import operator
import utool as ut
import vtool as vt
import numpy as np
import itertools as it
from functools import partial
from six import next
from six.moves import zip, range, map, reduce
from wbia.expt import cfghelpers
from wbia.expt import experiment_helpers

print, rrr, profile = ut.inject2(__name__)


def build_cmsinfo(cm_list, qreq_):
    r"""
    Helper function to report results over multiple queries (chip matches).
    Basically given a group of queries of the same name, we only care if one of
    them is correct.  This emulates encounters.

    Runs queries of a specific configuration returns the best rank of each
    query.

    Args:
        cm_list (list): list of chip matches
        qreq_ (QueryRequest): request that computed the chip matches.

    Returns:
        dict: cmsinfo - info about multiple chip matches cm_list

    CommandLine:
        python -m wbia get_query_result_info
        python -m wbia get_query_result_info:0 --db lynx \
            -a :qsame_imageset=True,been_adjusted=True,excluderef=True -t :K=1
        python -m wbia get_query_result_info:0 --db lynx \
            -a :qsame_imageset=True,been_adjusted=True,excluderef=True -t :K=1 --cmd

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.expt.test_result import *  # NOQA
        >>> import wbia
        >>> qreq_ = wbia.main_helpers.testdata_qreq_(a=[':qindex=0:3,dindex=0:5'])
        >>> cm_list = qreq_.execute()
        >>> cmsinfo = build_cmsinfo(cm_list, qreq_)
        >>> print(ut.repr2(cmsinfo))

    Ignore:
        wbia -e rank_cmc --db humpbacks -a :has_any=hasnotch,mingt=2 \
                -t :proot=BC_DTW --show --nocache-big

        wbia -e rank_cmc --db humpbacks -a :is_known=True,mingt=2 \
                -t :pipeline_root=BC_DTW

        wbia -e rank_cmc --db humpbacks -a :is_known=True \
                -t :pipeline_root=BC_DTW \
                --qaid=1,9,15,16,18 --daid-override=1,9,15,16,18,21,22 \
                --show --debug-depc

        --clear-all-depcache
    """
    ibs = qreq_.ibs

    qaids = qreq_.qaids
    daids = qreq_.daids
    # Get the groundtruth ranks and accuracy measures
    qx2_cminfo = []
    for cm in cm_list:
        if hasattr(cm, 'extend_results'):
            cminfo = cm.extend_results(qreq_).summarize(qreq_)
        else:
            cminfo = cm.summarize(qreq_)
        qx2_cminfo.append(cminfo)
    cmsinfo = ut.dict_stack(qx2_cminfo, 'qx2_')
    cmsinfo['qx2_gt_rank'] = ut.replace_nones(cmsinfo['qx2_gt_rank'], -1)

    if False:
        qx2_gtaids = ibs.get_annot_groundtruth(qaids, daid_list=daids)
        qx2_avepercision = np.array(
            [
                cm.get_average_percision(ibs=ibs, gt_aids=gt_aids)
                for (cm, gt_aids) in zip(cm_list, qx2_gtaids)
            ]
        )
        cmsinfo['qx2_avepercision'] = qx2_avepercision

    # Compute mAP score  # TODO: use mAP score
    # (Actually map score doesn't make much sense if using name scoring
    # mAP = qx2_avepercision[~np.isnan(qx2_avepercision)].mean()  # NOQA

    qaids = qreq_.qaids
    # qaids2 = [cm.qaid for cm in cm_list]
    # qnids = qreq_.get_qreq_annot_nids(qaids)  # TODO: use new nid getter
    qnids = ibs.get_annot_nids(qaids)

    unique_dnids = np.unique(ibs.get_annot_nids(qreq_.daids))
    unique_qnids, groupxs = ut.group_indices(qnids)
    cm_group_list = ut.apply_grouping(cm_list, groupxs)
    qnid2_aggnamescores = {}

    qnx2_nameres_info = []

    # Ranked list aggregation over groups of query annots
    nameres_info_list = []
    for qnid, cm_group in zip(unique_qnids, cm_group_list):
        nid2_name_score_group = [
            dict([(nid, cm.name_score_list[nidx]) for nid, nidx in cm.nid2_nidx.items()])
            for cm in cm_group
        ]
        aligned_name_scores = np.array(
            [
                ut.dict_take(nid_to_name_score, unique_dnids.tolist(), -np.inf)
                for nid_to_name_score in nid2_name_score_group
            ]
        ).T
        name_score_list = np.nanmax(aligned_name_scores, axis=1)
        qnid2_aggnamescores[qnid] = name_score_list
        # sort
        sortx = name_score_list.argsort()[::-1]
        sorted_namescores = name_score_list[sortx]
        sorted_dnids = unique_dnids[sortx]

        # infer agg name results
        success = sorted_dnids == qnid
        failure = np.logical_and(~success, sorted_dnids > 0)
        gt_name_rank = None if not np.any(success) else np.where(success)[0][0]
        gf_name_rank = None if not np.any(failure) else np.nonzero(failure)[0][0]

        gt_nid = sorted_dnids[gt_name_rank]
        gf_nid = sorted_dnids[gf_name_rank]
        gt_name_score = sorted_namescores[gt_name_rank]
        gf_name_score = sorted_namescores[gf_name_rank]

        if gt_name_score <= 0:
            # ensure failure cases are loud give them the worst possible rank
            # instead of a random one.
            if hasattr(qreq_, 'dnids'):
                gt_name_rank = len(qreq_.dnids) + 1
            else:
                dnids = list(set(ibs.get_annot_nids(qreq_.daids)))
                gt_name_rank = len(dnids) + 1
        qnx2_nameres_info = {}
        qnx2_nameres_info['qnid'] = qnid
        qnx2_nameres_info['gt_nid'] = gt_nid
        qnx2_nameres_info['gf_nid'] = gf_nid
        qnx2_nameres_info['gt_name_rank'] = gt_name_rank
        qnx2_nameres_info['gf_name_rank'] = gf_name_rank
        qnx2_nameres_info['gt_name_score'] = gt_name_score
        qnx2_nameres_info['gf_name_score'] = gf_name_score

        nameres_info_list.append(qnx2_nameres_info)
        nameres_info = ut.dict_stack(nameres_info_list, 'qnx2_')
        cmsinfo.update(nameres_info)

    return cmsinfo


def combine_testres_list(ibs, testres_list):
    """
    combine test results over multiple annot configs

    The combination of pipeline and annotation config is indexed by cfgx.
    A cfgx corresponds to a unique query request

    CommandLine:
        python -m wbia --tf combine_testres_list

        python -m wbia --tf -draw_rank_cmc --db PZ_MTEST --show
        python -m wbia --tf -draw_rank_cmc --db PZ_Master1 --show
        python -m wbia --tf -draw_rank_cmc --db PZ_MTEST --show -a varysize -t default
        python -m wbia --tf -draw_rank_cmc --db PZ_MTEST --show -a varysize -t default

    >>> # DISABLE_DOCTEST
    >>> from wbia.expt.test_result import *  # NOQA
    >>> from wbia.expt import harness
    >>> ibs, testres = harness.testdata_expts('PZ_MTEST', ['varysize'])
    """
    import copy
    from wbia.expt import annotation_configs

    acfg_list = [tr.acfg for tr in testres_list]
    acfg_lbl_list = annotation_configs.get_varied_acfg_labels(acfg_list)

    flat_acfg_list = annotation_configs.flatten_acfg_list(acfg_list)
    nonvaried_acfg, varied_acfg_list = ut.partition_varied_cfg_list(flat_acfg_list)

    def combine_lbls(lbl, acfg_lbl):
        if len(lbl) == 0:
            return acfg_lbl
        if len(acfg_lbl) == 0:
            return lbl
        return lbl + '+' + acfg_lbl

    # TODO: depcirate cfg_dict list for pcfg_list (I think)

    agg_cfg_list = ut.flatten([tr.cfg_list for tr in testres_list])
    agg_cfgx2_qreq_ = ut.flatten([tr.cfgx2_qreq_ for tr in testres_list])
    agg_cfgdict_list = ut.flatten([tr.cfgdict_list for tr in testres_list])
    agg_cfgx2_cmsinfo = ut.flatten([tr.cfgx2_cmsinfo for tr in testres_list])
    agg_varied_acfg_list = ut.flatten(
        [[acfg] * len(tr.cfg_list) for tr, acfg in zip(testres_list, varied_acfg_list)]
    )
    agg_cfgx2_lbls = ut.flatten(
        [
            [combine_lbls(lbl, acfg_lbl) for lbl in tr.cfgx2_lbl]
            for tr, acfg_lbl in zip(testres_list, acfg_lbl_list)
        ]
    )

    agg_cfgx2_acfg = ut.flatten(
        [
            [copy.deepcopy(acfg)] * len(tr.cfg_list)
            for tr, acfg in zip(testres_list, acfg_list)
        ]
    )

    big_testres = TestResult(
        agg_cfg_list, agg_cfgx2_lbls, agg_cfgx2_cmsinfo, agg_cfgx2_qreq_
    )

    # Give the big test result an acfg that is common between everything
    big_testres.acfg = annotation_configs.unflatten_acfgdict(nonvaried_acfg)
    # TODO: cfgdict_list -> pcfg_list
    big_testres.cfgdict_list = agg_cfgdict_list  # TODO: depricate

    big_testres.common_acfg = annotation_configs.compress_aidcfg(big_testres.acfg)
    big_testres.common_cfgdict = reduce(ut.dict_intersection, big_testres.cfgdict_list)
    big_testres.varied_acfg_list = agg_varied_acfg_list
    big_testres.nonvaried_acfg = nonvaried_acfg
    big_testres.varied_cfg_list = [
        ut.delete_dict_keys(cfgdict.copy(), list(big_testres.common_cfgdict.keys()))
        for cfgdict in big_testres.cfgdict_list
    ]
    big_testres.acfg_list = acfg_list
    big_testres.cfgx2_acfg = agg_cfgx2_acfg
    big_testres.cfgx2_pcfg = agg_cfgdict_list

    assert len(agg_cfgdict_list) == len(agg_cfgx2_acfg)

    # big_testres.acfg
    testres = big_testres
    # big_testres = testres
    return testres


@six.add_metaclass(ut.ReloadingMetaclass)
class TestResult(ut.NiceRepr):
    """
    CommandLine:
        export SMK_PIPE="smk:nwords=[64000],sv=[False]"
        wbia TestResult --db PZ_MTEST -a ctrl -p $SMK_PIPE
        wbia TestResult --db Oxford   -a oxford -p $SMK_PIPE

    Example:
        >>> # Script
        >>> from wbia.init import main_helpers
        >>> import utool as ut
        >>> ibs, testres = main_helpers.testdata_expts()
        >>> testres.help()
        >>> actions = testres.get_actions()
        >>> testres.map_score()
        >>> ut.qtensure()
        >>> prompt = ut.InteractivePrompt(actions)
        >>> prompt.loop()
    """

    def __init__(testres, cfg_list, cfgx2_lbl, cfgx2_cmsinfo, cfgx2_qreq_):
        assert len(cfg_list) == len(cfgx2_lbl), 'bad lengths1: %r != %r' % (
            len(cfg_list),
            len(cfgx2_lbl),
        )
        assert len(cfgx2_qreq_) == len(cfgx2_lbl), 'bad lengths2: %r != %r' % (
            len(cfgx2_qreq_),
            len(cfgx2_lbl),
        )
        assert len(cfgx2_cmsinfo) == len(cfgx2_lbl), 'bad lengths3: %r != %r' % (
            len(cfgx2_cmsinfo),
            len(cfgx2_lbl),
        )
        # TODO rename cfg_list to pcfg_list
        testres.cfg_list = cfg_list
        testres.cfgx2_lbl = cfgx2_lbl
        testres.cfgx2_cmsinfo = cfgx2_cmsinfo
        testres.cfgx2_qreq_ = cfgx2_qreq_
        # TODO: uncomment
        # testres.cfgx2_acfg
        # testres.cfgx2_qcfg
        # testres.acfg_list        = None  #
        testres.lbl = None
        testres.testnameid = None

    @classmethod
    def from_cms(cls, cm_list, qreq_):
        cfg_list = [qreq_.qparams]  # should actually be the specified dict
        cfgx2_lbl = ['unspecified']
        cmsinfo = build_cmsinfo(cm_list, qreq_)
        cfgx2_cmsinfo = [cmsinfo]
        cfgx2_qreq_ = [qreq_]
        testres = cls(cfg_list, cfgx2_lbl, cfgx2_cmsinfo, cfgx2_qreq_)
        return testres

    def __str__(testres):
        return testres.reconstruct_test_flags()

    # def __repr__(testres):
    #    return testres._custom_str()

    def __nice__(testres):
        dbname = None if testres.ibs is None else testres.ibs.get_dbname()
        # hashkw = dict(_new=True, pathsafe=False)
        infostr_ = 'nCfg=%s' % testres.nConfig
        if testres.nConfig == 1:
            qreq_ = testres.cfgx2_qreq_[0]
            infostr_ += ' nQ=%s, nD=%s %s' % (
                len(qreq_.qaids),
                len(qreq_.daids),
                qreq_.get_pipe_hashid(),
            )
        # nD=%s %s' % (, len(testres.daids), testres.get_pipe_hashid())
        nice = '%s %s' % (dbname, infostr_)
        return nice

    @property
    def ibs(testres):
        ibs_list = []
        for qreq_ in testres.cfgx2_qreq_:
            try:
                ibs_list.append(qreq_.ibs)
            except AttributeError:
                ibs_list.append(qreq_.depc.controller)
        ibs = ibs_list[0]
        for ibs_ in ibs_list:
            assert (
                ibs.get_dbdir() == ibs_.get_dbdir()
            ), 'all requests must use the same database'
        return ibs

    @property
    def qaids(testres):
        assert (
            testres.has_constant_qaids()
        ), 'must have constant qaids to use this property'
        return testres.cfgx2_qaids[0]
        # return testres._qaids

    @property
    def nConfig(testres):
        # FIXME: this is the number of requests not the number of
        # pipeline configurations
        return len(testres.cfg_list)

    @property
    def unique_pcfgs(testres):
        unique_idxs = ut.unique_indices(map(id, testres.cfgx2_pcfg))
        return ut.take(testres.cfgx2_pcfg, unique_idxs)

    @property
    def nQuery(testres):
        return len(testres.qaids)

    @property
    def rank_mat(testres):
        return testres.get_rank_mat()

    @property
    def cfgx2_daids(testres):
        daids_list = [qreq_.daids for qreq_ in testres.cfgx2_qreq_]
        return daids_list

    @property
    def cfgx2_qaids(testres):
        qaids_list = [qreq_.qaids for qreq_ in testres.cfgx2_qreq_]
        return qaids_list

    def has_constant_daids(testres):
        return ut.allsame(testres.cfgx2_daids)

    def has_constant_qaids(testres):
        return ut.allsame(testres.cfgx2_qaids)

    def has_constant_length_daids(testres):
        return ut.allsame(list(map(len, testres.cfgx2_daids)))

    def has_constant_length_qaids(testres):
        return ut.allsame(list(map(len, testres.cfgx2_qaids)))

    def get_infoprop_list(testres, key, qaids=None):
        """
        key = 'qx2_gt_rank'
        key = 'qx2_gt_rank'
        qaids = testres.get_test_qaids()
        """
        if key == 'participant':
            # Get if qaids are part of the config
            cfgx2_infoprop = [np.in1d(qaids, aids_) for aids_ in testres.cfgx2_qaids]
        else:
            _tmp1_cfgx2_infoprop = ut.get_list_column(testres.cfgx2_cmsinfo, key)
            _tmp2_cfgx2_infoprop = list(
                map(np.array, ut.util_list.replace_nones(_tmp1_cfgx2_infoprop, np.nan))
            )
            if qaids is None:
                cfgx2_infoprop = _tmp2_cfgx2_infoprop
            else:
                # Use nan if the aid doesnt exist
                cfgx2_qaid2_qx = [
                    dict(zip(aids_, range(len(aids_)))) for aids_ in testres.cfgx2_qaids
                ]
                qxs_list = [
                    ut.dict_take(qaid2_qx, qaids, None) for qaid2_qx in cfgx2_qaid2_qx
                ]
                cfgx2_infoprop = [
                    [np.nan if x is None else props[x] for x in qxs]
                    for props, qxs in zip(_tmp2_cfgx2_infoprop, qxs_list)
                ]
            if key == 'qx2_gt_rank' or key.endswith('_rank'):
                # hack
                wpr = testres.get_worst_possible_rank()
                cfgx2_infoprop = [
                    np.array([wpr if rank == -1 else rank for rank in infoprop])
                    for infoprop in cfgx2_infoprop
                ]
        return cfgx2_infoprop

    def get_infoprop_mat(testres, key, qaids=None):
        """
        key = 'qx2_gf_raw_score'
        key = 'qx2_gt_raw_score'
        """
        cfgx2_infoprop = testres.get_infoprop_list(key, qaids)
        # concatenate each query rank across configs
        infoprop_mat = np.vstack(cfgx2_infoprop).T
        return infoprop_mat

    @ut.memoize
    def get_rank_mat(testres, qaids=None):
        # Ranks of Best Results
        rank_mat = testres.get_infoprop_mat(key='qx2_gt_rank', qaids=qaids)
        return rank_mat

    def get_worst_possible_rank(testres):
        # worst_possible_rank = max(9001, len(testres.daids) + 1)
        worst_possible_rank = max([len(qreq_.daids) for qreq_ in testres.cfgx2_qreq_]) + 1
        # worst_possible_rank = len(testres.daids) + 1
        return worst_possible_rank

    def get_rank_histograms(testres, bins=None, key=None, join_acfgs=False):
        """
        Ignore:
            testres.get_infoprop_mat('qnx2_gt_name_rank')
            testres.get_infoprop_mat('qnx2_gf_name_rank')
            testres.get_infoprop_mat('qnx2_qnid')

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> from wbia.init import main_helpers
            >>> ibs, testres = main_helpers.testdata_expts('testdb1', a=['default'])
            >>> bins = 'dense'
            >>> key = 'qnx2_gt_name_rank'
            >>> config_hists = testres.get_rank_histograms(bins, key=key)
        """
        if key is None:
            key = 'qx2_gt_rank'
            # key = 'qnx2_gt_name_rank'
        if bins is None:
            bins = testres.get_rank_histogram_bins()
        elif bins == 'dense':
            bins = np.arange(testres.get_worst_possible_rank() + 1)

        cfgx2_ranks = testres.get_infoprop_list(key=key)

        # Use numpy histogram repr
        cfgx2_hist = np.zeros((len(cfgx2_ranks), len(bins) - 1), dtype=np.int32)

        for cfgx, ranks in enumerate(cfgx2_ranks):
            freq = np.histogram(ranks, bins=bins)[0]
            cfgx2_hist[cfgx] = freq

        if join_acfgs:
            # Hack for turtles / general way of doing cross validation
            # however, we need to change the name
            groupxs = testres.get_cfgx_groupxs()
            cfgx2_hist = np.array(
                [
                    np.sum(group, axis=0)
                    for group in ut.apply_grouping(cfgx2_hist, groupxs)
                ]
            )

        return cfgx2_hist, bins

    def get_rank_percentage_cumhist(testres, bins='dense', key=None, join_acfgs=False):
        r"""
        Args:
            bins (unicode): (default = u'dense')
            key (None): (default = None)
            join_acfgs (bool): (default = False)

        Returns:
            tuple: (config_cdfs, edges)

        CommandLine:
            python -m wbia --tf TestResult.get_rank_percentage_cumhist
            python -m wbia --tf TestResult.get_rank_percentage_cumhist \
                -t baseline -a unctrl ctrl

            python -m wbia --tf TestResult.get_rank_percentage_cumhist \
                --db lynx \
                -a default:qsame_imageset=True,been_adjusted=True,excluderef=True \
                -t default:K=1 --show --cmd

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> from wbia.init import main_helpers
            >>> ibs, testres = main_helpers.testdata_expts(
            >>>     'testdb1', a=['default:num_names=1,name_offset=[0,1]'])
            >>> bins = u'dense'
            >>> key = None
            >>> (config_cdfs, edges) = testres.get_rank_percentage_cumhist(bins)
            >>> result = ('(config_cdfs, edges) = %s' % (str((config_cdfs, edges)),))
            >>> print(result)
        """
        cfgx2_hist, edges = testres.get_rank_histograms(
            bins, key=key, join_acfgs=join_acfgs
        )
        cfgx2_cumhist = np.cumsum(cfgx2_hist, axis=1)
        cfgx2_cumhist_percent = 100 * cfgx2_cumhist / cfgx2_cumhist.T[-1].T[:, None]
        return cfgx2_cumhist_percent, edges

    def get_cfgx_groupxs(testres):
        r"""
        Returns the group indices of configurations specified to be joined.

        Ignore:
            a = [
            'default:minqual=good,require_timestamp=True,view=left,crossval_enc=True,joinme=1',
            'default:minqual=good,require_timestamp=True,view=right,crossval_enc=True,joinme=1',
            'default:minqual=ok,require_timestamp=True,view=left,crossval_enc=True,joinme=2',
            'default:minqual=ok,require_timestamp=True,view=right,crossval_enc=True,joinme=2',
            ]
            >>> a = [
            >>>     'default:minqual=good,require_timestamp=True,view=left,crossval_enc=True,joinme=1',
            >>>     'default:minqual=good,require_timestamp=True,view=right,crossval_enc=True,joinme=1',
            >>>     'default:minqual=ok,require_timestamp=True,view=left,crossval_enc=True,joinme=2',
            >>>     'default:minqual=ok,require_timestamp=True,view=right,crossval_enc=True,joinme=2',
            >>> ]
            >>> from wbia.init import main_helpers
            >>> #a = 'default:minqual=good,require_timestamp=True,crossval_enc=True,view=[right,left]'
            >>> t = 'default:K=[1]'
            >>> ibs, testres = main_helpers.testdata_expts('WWF_Lynx_Copy', a=a, t=t)
            >>> testres.get_cfgx_groupxs()

            ut.lmap(sum, ut.apply_grouping([len(ut.unique(ibs.annots(aids).nids)) for aids in testres.cfgx2_qaids], testres.get_cfgx_groupxs()))
            ut.lmap(sum, ut.apply_grouping([len(ut.unique(ibs.annots(aids))) for aids in testres.cfgx2_qaids], testres.get_cfgx_groupxs()))

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> from wbia.init import main_helpers
            >>> ibs, testres = main_helpers.testdata_expts(
            >>>    'PZ_MTEST',
            >>>     a=['default:qnum_names=1,qname_offset=[0,1],joinme=1,dpername=1',
            >>>        'default:qsize=1,dpername=[1,2]'],
            >>>     t=['default:K=[1,2]'])
            >>> groupxs = testres.get_cfgx_groupxs()
            >>> result = groupxs
            >>> print(result)
            [[6], [4], [0, 2], [7], [5], [1, 3]]
        """
        # Group-ids for annotations are determined by joinme labels
        # (used primarilly in cross-validation)
        acfg_joinid = [acfg['qcfg']['joinme'] for acfg in testres.cfgx2_acfg]
        # Anything that does not have a joinme groupid is standalone and must
        # be given a unique groupid
        gen_groupid = it.count(-1, step=-1)
        acfg_groupids = [
            next(gen_groupid) if grpid is None else grpid for grpid in acfg_joinid
        ]
        # Ensure that different pipeline configs are in different groups
        pcfg_groupids = ut.get_varied_cfg_lbls(testres.cfgx2_pcfg)
        cfg_groupids = list(zip(pcfg_groupids, acfg_groupids))
        groupxs = ut.group_indices(cfg_groupids)[1]
        return groupxs

    def get_rank_histogram_bins(testres):
        """ easy to see histogram bins """
        worst_possible_rank = testres.get_worst_possible_rank()
        if worst_possible_rank > 50:
            bins = [0, 1, 5, 50, worst_possible_rank, worst_possible_rank + 1]
        elif worst_possible_rank > 5:
            bins = [0, 1, 5, worst_possible_rank, worst_possible_rank + 1]
        else:
            bins = [0, 1, 5]
        return bins

    def get_X_LIST(testres):
        """ DEPRICATE or refactor """
        # X_LIST = ut.get_argval('--rank-lt-list', type_=list, default=[1])
        X_LIST = ut.get_argval('--rank-lt-list', type_=list, default=[1, 5])
        return X_LIST

    def get_nLessX_dict(testres):
        """
        Build a (histogram) dictionary mapping X (as in #ranks < X) to a list
        of cfg scores
        """
        X_LIST = testres.get_X_LIST()
        nLessX_dict = {int(X): np.zeros(testres.nConfig) for X in X_LIST}
        cfgx2_qx2_gt_rank = testres.get_infoprop_list('qx2_gt_rank')
        for X in X_LIST:
            cfgx2_lessX_mask = [
                np.logical_and(0 <= qx2_gt_ranks, qx2_gt_ranks < X)
                for qx2_gt_ranks in cfgx2_qx2_gt_rank
            ]
            cfgx2_nLessX = np.array([lessX_.sum(axis=0) for lessX_ in cfgx2_lessX_mask])
            nLessX_dict[int(X)] = cfgx2_nLessX
        return nLessX_dict

    def get_all_varied_params(testres):
        r"""
        Returns the parameters that were varied between different
        configurations in this test

        Returns:
            list: varied_params

        CommandLine:
            python -m wbia TestResult.get_all_varied_params

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> import wbia
            >>> testres = wbia.testdata_expts(
            >>>     'PZ_MTEST', t='default:K=[1,2]')[1]
            >>> varied_params = sorted(testres.get_all_varied_params())
            >>> result = ('varied_params = %s' % (ut.repr2(varied_params),))
            >>> print(result)
            varied_params = ['K', '_cfgindex']
        """
        # only for big results
        varied_cfg_params = list(
            set(ut.flatten([cfgdict.keys() for cfgdict in testres.varied_cfg_list]))
        )
        varied_acfg_params = list(
            set(ut.flatten([acfg.keys() for acfg in testres.varied_acfg_list]))
        )
        varied_params = varied_acfg_params + varied_cfg_params
        return varied_params

    def get_total_num_varied_params(testres):
        return len(testres.get_all_varied_params())

    def get_param_basis(testres, key):
        """
        Returns what a param was varied between over all tests
        key = 'K'
        key = 'dcfg_sample_size'
        """
        if key == 'len(daids)':
            basis = sorted(list(set([len(daids) for daids in testres.cfgx2_daids])))
        elif any([key in cfgdict for cfgdict in testres.varied_cfg_list]):
            basis = sorted(
                list(set([cfgdict[key] for cfgdict in testres.varied_cfg_list]))
            )
        elif any([key in cfgdict for cfgdict in testres.varied_acfg_list]):
            basis = sorted(list(set([acfg[key] for acfg in testres.varied_acfg_list])))
        else:
            # assert False, 'param is not varied'
            if key in testres.common_cfgdict:
                basis = [testres.common_cfgdict[key]]
            elif key in testres.nonvaried_acfg:
                basis = [testres.nonvaried_acfg[key]]
            else:
                assert False, 'param=%r doesnt exist' % (key,)
        return basis

    def get_param_val_from_cfgx(testres, cfgx, key):
        if key == 'len(daids)':
            return len(testres.cfgx2_daids[cfgx])
        # --- HACK - the keys are different in varied dict for some reason ---
        elif any([key in cfgdict for cfgdict in testres.varied_cfg_list]):
            return testres.varied_cfg_list[cfgx][key]
        elif any([key in cfgdict for cfgdict in testres.varied_acfg_list]):
            return testres.varied_acfg_list[cfgx][key]
        # --- / Hack
        elif any([key in cfgdict for cfgdict in testres.cfgx2_pcfg]):
            return testres.cfgx2_pcfg[cfgx][key]
        elif any([key in cfgdict for cfgdict in testres.cfgx2_acfg]):
            return testres.cfgx2_acfg[cfgx][key]
        else:
            assert False, 'param=%r doesnt exist' % (key,)

    def get_cfgx_with_param(testres, key, val):
        """
        Gets configs where the given parameter is held constant
        """
        if key == 'len(daids)':
            cfgx_list = [
                cfgx
                for cfgx, daids in enumerate(testres.cfgx2_daids)
                if len(daids) == val
            ]
        elif any([key in cfgdict for cfgdict in testres.varied_cfg_list]):
            cfgx_list = [
                cfgx
                for cfgx, cfgdict in enumerate(testres.varied_cfg_list)
                if cfgdict[key] == val
            ]
        elif any([key in cfgdict for cfgdict in testres.varied_acfg_list]):
            cfgx_list = [
                cfgx
                for cfgx, acfg in enumerate(testres.varied_acfg_list)
                if acfg[key] == val
            ]
        else:
            if key in testres.common_cfgdict:
                cfgx_list = list(range(testres.nConfig))
            elif key in testres.nonvaried_acfg:
                cfgx_list = list(range(testres.nConfig))
            else:
                assert False, 'param=%r doesnt exist' % (key,)
            # assert False, 'param is not varied'
        return cfgx_list

    def get_pipecfg_args(testres):
        if '_cfgstr' in testres.common_cfgdict:
            pipecfg_args = [testres.common_cfgdict['_cfgstr']]
        else:
            pipecfg_args = ut.unique_ordered(
                [cfg['_cfgstr'] for cfg in testres.varied_cfg_list]
            )
        return ' '.join(pipecfg_args)

    def get_annotcfg_args(testres):
        """
        CommandLine:
            # TODO: More robust fix
            # To reproduce the error
            wbia -e rank_cmc --db humpbacks_fb -a default:mingt=2,qsize=10,dsize=100 default:qmingt=2,qsize=10,dsize=100 -t default:proot=BC_DTW,decision=max,crop_dim_size=500,crop_enabled=True,manual_extract=False,use_te_scorer=True,ignore_notch=True,te_score_weight=0.5 --show
        """
        if '_cfgstr' in testres.common_acfg['common']:
            annotcfg_args = [testres.common_acfg['common']['_cfgstr']]
        else:
            try:
                annotcfg_args = ut.unique_ordered(
                    [acfg['common']['_cfgstr'] for acfg in testres.varied_acfg_list]
                )
            except KeyError:
                # HACK FIX
                try:
                    annotcfg_args = ut.unique_ordered(
                        [acfg['_cfgstr'] for acfg in testres.varied_acfg_list]
                    )
                except KeyError:
                    annotcfg_args = ut.unique_ordered(
                        [acfg['qcfg__cfgstr'] for acfg in testres.varied_acfg_list]
                    )
        return ' '.join(annotcfg_args)

    def reconstruct_test_flags(testres):
        flagstr = ' '.join(
            [
                '-a ' + testres.get_annotcfg_args(),
                '-t ' + testres.get_pipecfg_args(),
                '--db ' + testres.ibs.get_dbname(),
            ]
        )
        return flagstr

    def get_full_cfgstr(testres, cfgx):
        """ both qannots and dannots included """
        full_cfgstr = testres.cfgx2_qreq_[cfgx].get_full_cfgstr()
        return full_cfgstr

    @ut.memoize
    def get_cfgstr(testres, cfgx):
        """ just dannots and config_str """
        cfgstr = testres.cfgx2_qreq_[cfgx].get_cfgstr()
        return cfgstr

    def _shorten_lbls(testres, lbl):
        """
        hacky function
        """
        import re

        repl_list = [
            ('candidacy_', ''),
            ('viewpoint_compare', 'viewpoint'),
            # ('custom', 'default'),
            # ('fg_on', 'FG'),
            # ('fg_on=True', 'FG'),
            # ('fg_on=False,?', ''),
            ('fg_on=True', 'FG=True'),
            ('fg_on=False,?', 'FG=False'),
            ('lnbnn_on=True', 'LNBNN'),
            ('lnbnn_on=False,?', ''),
            ('normonly_on=True', 'normonly'),
            ('normonly_on=False,?', ''),
            ('bar_l2_on=True', 'dist'),
            ('bar_l2_on=False,?', ''),
            (r'joinme=\d+,?', ''),
            ('dcrossval_enc', 'denc_per_name'),
            ('sv_on', 'SV'),
            ('rotation_invariance', 'RI'),
            ('affine_invariance', 'AI'),
            ('query_rotation_heuristic', 'QRH'),
            ('nNameShortlistSVER', 'nRR'),
            #
            # ('sample_per_ref_name', 'per_ref_name'),
            ('sample_per_ref_name', 'per_gt_name'),
            ('require_timestamp=True', 'require_timestamp'),
            ('require_timestamp=False,?', ''),
            ('require_timestamp=None,?', ''),
            ('[_A-Za-z]*=None,?', ''),
            ('dpername=None,?', ''),
            # ???
            # ('sample_per_ref_name', 'per_gt_name'),
            # ('per_name', 'per_gf_name'),   # Try to make labels clearer for paper
            # ----
            # ('prescore_method=\'?csum\'?,score_method=\'?csum\'?,?', 'amech'),
            # ('prescore_method=\'?nsum\'?,score_method=\'?nsum\'?,?', 'fmech'),
            ("prescore_method='?csum'?,score_method='?csum'?,?", 'mech=annot'),
            ("prescore_method='?nsum'?,score_method='?nsum'?,?", 'mech=name'),
            ('force_const_size=[^,]+,?', ''),
            (r'[dq]?_true_size=\d+,?', ''),
            (r'[dq]?_orig_size=[^,]+,?', ''),
            # Hack
            (
                '[qd]?exclude_reference='
                + ut.regex_or(['True', 'False', 'None'])
                + r'\,?',
                '',
            ),
            # ('=True', '=On'),
            # ('=False', '=Off'),
            ('=True', '=T'),
            ('=False', '=F'),
            (',$', ''),
        ]
        for ser, rep in repl_list:
            lbl = re.sub(ser, rep, lbl)
        return lbl

    def get_short_cfglbls(testres, join_acfgs=False):
        """
        Labels for published tables

        cfg_lbls = ['baseline:nRR=200+default:', 'baseline:+default:']

        CommandLine:
            python -m wbia --tf TestResult.get_short_cfglbls

        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> import wbia
            >>> ibs, testres = wbia.testdata_expts('PZ_MTEST', a=['ctrl:size=10'],
            >>>                                     t=['default:dim_size=[450,550]'])
            >>> cfg_lbls = testres.get_short_cfglbls()
            >>> result = ('cfg_lbls = %s' % (ut.repr2(cfg_lbls),))
            >>> print(result)
            cfg_lbls = [
                'default:dim_size=450+ctrl',
                'default:dim_size=550+ctrl',
            ]
        """
        from wbia.expt import annotation_configs

        if False:
            acfg_names = [acfg['qcfg']['_cfgstr'] for acfg in testres.cfgx2_acfg]
            pcfg_names = [pcfg['_cfgstr'] for pcfg in testres.cfgx2_pcfg]
            # Only vary the label settings within the cfgname
            acfg_hashes = np.array(list(map(hash, acfg_names)))
            unique_hashes, a_groupxs = vt.group_indices(acfg_hashes)
            a_label_groups = []
            for groupx in a_groupxs:
                acfg_list = ut.take(testres.cfgx2_acfg, groupx)
                varied_lbls = annotation_configs.get_varied_acfg_labels(
                    acfg_list, mainkey='_cfgstr'
                )
                a_label_groups.append(varied_lbls)
            acfg_lbls = vt.invert_apply_grouping(a_label_groups, a_groupxs)

            pcfg_hashes = np.array(list(map(hash, pcfg_names)))
            unique_hashes, p_groupxs = vt.group_indices(pcfg_hashes)
            p_label_groups = []
            for groupx in p_groupxs:
                pcfg_list = ut.take(testres.cfgx2_pcfg, groupx)
                varied_lbls = ut.get_varied_cfg_lbls(pcfg_list, mainkey='_cfgstr')
                p_label_groups.append(varied_lbls)
            pcfg_lbls = vt.invert_apply_grouping(p_label_groups, p_groupxs)

            cfg_lbls = [albl + '+' + plbl for albl, plbl in zip(acfg_lbls, pcfg_lbls)]
        else:
            cfg_lbls_ = testres.cfgx2_lbl[:]
            cfg_lbls_ = [testres._shorten_lbls(lbl) for lbl in cfg_lbls_]
            # split configs up by param and annots
            pa_tups = [lbl.split('+') for lbl in cfg_lbls_]
            cfg_lbls = []
            for pa in pa_tups:
                new_parts = []
                for part in pa:
                    _tup = part.split(ut.NAMEVARSEP)
                    name, settings = _tup if len(_tup) > 1 else (_tup[0], '')
                    new_parts.append(part if settings else name)
                if len(new_parts) == 2 and new_parts[1] == 'default':
                    newlbl = new_parts[0]
                else:
                    newlbl = '+'.join(new_parts)
                cfg_lbls.append(newlbl)
        if join_acfgs:
            groupxs = testres.get_cfgx_groupxs()
            group_lbls = []
            for group in ut.apply_grouping(cfg_lbls, groupxs):
                num_parts = 0
                part_dicts = []
                for lbl in group:
                    parts = []
                    for count, pa in enumerate(lbl.split('+')):
                        num_parts = max(num_parts, count + 1)
                        cfgdict = cfghelpers.parse_cfgstr_list2([pa], strict=False)[0][0]
                        parts.append(cfgdict)
                    part_dicts.append(parts)
                group_lbl_parts = []
                for px in range(num_parts):
                    cfgs = ut.take_column(part_dicts, px)
                    nonvaried_cfg = ut.partition_varied_cfg_list(cfgs)[0]
                    group_lbl_parts.append(ut.get_cfg_lbl(nonvaried_cfg))
                    # print('nonvaried_lbl = %r' % (nonvaried_lbl,))
                group_lbl = '+'.join(group_lbl_parts)
                group_lbls.append(group_lbl)
            cfg_lbls = group_lbls
        return cfg_lbls

    def get_varied_labels(testres, shorten=False, join_acfgs=False, sep=''):
        """
        Returns labels indicating only the parameters that have been varied between
        different annot/pipeline configurations.

        Helper for consistent figure titles

        CommandLine:
            python -m wbia --tf TestResult.make_figtitle  --prefix "Seperability " --db GIRM_Master1   -a timectrl -t Ell:K=2     --hargv=scores
            python -m wbia --tf TestResult.make_figtitle
            python -m wbia TestResult.get_varied_labels

        Example:
            >>> # SLOW_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> import wbia
            >>> ibs, testres = wbia.testdata_expts(
            >>>     'PZ_MTEST', t='default:K=[1,2]',
            >>>     #a=['timectrl:qsize=[1,2],dsize=[3,4]']
            >>>     a=[
            >>>        'default:qsize=[1,2],dsize=2,joinme=1,view=left',
            >>>        'default:qsize=2,dsize=3,joinme=1,view=primary',
            >>>        'default:qsize=[3,2],dsize=4,joinme=2,view=left',
            >>>        'default:qsize=4,dsize=5,joinme=2,view=primary',
            >>>       ]
            >>> )
            >>> # >>> ibs, testres = wbia.testdata_expts(
            >>> # >>>     'WWF_Lynx_Copy', t='default:K=1',
            >>> # >>>     a=[
            >>> # >>>         'default:minqual=good,require_timestamp=True,view=left,dcrossval_enc=1,joinme=1',
            >>> # >>>         'default:minqual=good,require_timestamp=True,view=left,dcrossval_enc=2,joinme=2',
            >>> # >>>         #'default:minqual=good,require_timestamp=True,view=left,dcrossval_enc=3,joinme=3',
            >>> # >>>         'default:minqual=good,require_timestamp=True,view=right,dcrossval_enc=1,joinme=1',
            >>> # >>>         'default:minqual=good,require_timestamp=True,view=right,dcrossval_enc=2,joinme=2',
            >>> # >>>         #'default:minqual=good,require_timestamp=True,view=right,dcrossval_enc=3,joinme=3',
            >>> # >>>       ]
            >>> # >>> )
            >>> varied_lbls = testres.get_varied_labels(shorten=False, join_acfgs=True)
            >>> result = ('varied_lbls = %s' % (ut.repr2(varied_lbls, strvals=True, nl=2),))
            >>> print(result)

            varied_lbls = [u'K=1+qsize=1', u'K=2+qsize=1', u'K=1+qsize=2', u'K=2+qsize=2']
        """
        from wbia.expt import annotation_configs

        varied_acfgs = annotation_configs.get_varied_acfg_labels(
            testres.cfgx2_acfg, checkname=True
        )
        # print('varied_acfgs = %s' % (ut.repr2(varied_acfgs, nl=2),))
        # print('testres.cfgx2_acfg = %s' % (ut.repr3(testres.cfgx2_acfg),))
        varied_pcfgs = ut.get_varied_cfg_lbls(testres.cfgx2_pcfg, checkname=True)
        # varied_acfgs = ut.get_varied_cfg_lbls(testres.cfgx2_acfg, checkname=True)
        name_sep = ':'
        cfg_sep = '+'

        if join_acfgs:
            # Hack for the grouped config problem
            new_varied_acfgs = []
            groupxs = testres.get_cfgx_groupxs()
            grouped_acfgs = ut.apply_grouping(varied_acfgs, groupxs)
            grouped_pcfgs = ut.apply_grouping(varied_pcfgs, groupxs)
            for group in grouped_acfgs:
                group = [p if name_sep in p else name_sep + p for p in group]
                # Re-parse given back into dictionary form
                cfgdicts_ = cfghelpers.parse_cfgstr_list2(group, strict=False)
                # I forget why these are stored in a 2d-list
                cfgdicts = ut.take_column(cfgdicts_, 0)
                new_acfgs = ut.partition_varied_cfg_list(cfgdicts)
                # Hack, just taking the first one that has agreement between
                # joinme / crossvalidation runs
                new_acfg = new_acfgs[0]
                if True:
                    # look at internal variance within xval runs
                    internal_cfgs = new_acfgs[1]
                    import pandas as pd

                    intern_variations = pd.DataFrame.from_dict(internal_cfgs).to_dict(
                        orient='list'
                    )

                    op_prefixes = {
                        'sum': (np.sum, 'Σ-', ''),
                        'mean': (np.mean, 'µ-', ''),
                        'set': (lambda x: '&'.join(set(map(six.text_type, x))), '', 's'),
                    }
                    known_modes = {
                        'dsize': 'mean',
                        'qsize': 'sum',
                        'view': 'set',
                    }
                    for key in intern_variations.keys():
                        if key.startswith('_'):
                            continue
                        mode = known_modes.get(key, None)
                        vals = intern_variations[key]
                        if mode is None:
                            mode = 'set'
                        if key == 'crossval_idx':
                            new_acfg['folds'] = len(intern_variations['crossval_idx'])
                        else:
                            op, pref, suff = op_prefixes[mode]
                            c = op(vals)
                            if isinstance(c, six.string_types):
                                new_acfg[pref + key + suff] = c
                            else:
                                new_acfg[pref + key + suff] = ut.repr2(c, precision=2)
                    # if 'dsize' in intern_variations:
                    #     new_acfg['µ-dsize'] = np.sum(intern_variations['dsize'])
                    # if 'qsize' in intern_variations:
                    #     new_acfg['Σ-qsize'] = np.sum(intern_variations['qsize'])
                    # if 'view' in intern_variations:
                    #     new_acfg['views'] = '&'.join(set(intern_variations['view']))
                    # if 'crossval_idx' in intern_variations:
                    #     new_acfg['folds'] = len(intern_variations['crossval_idx'])
                new_varied_acfgs.append(new_acfg)

            # Do one more dup check to remove the duplicate summaries
            common_new_acfg = ut.partition_varied_cfg_list(new_varied_acfgs)[0]
            for key in common_new_acfg.keys():
                if not key.startswith('_'):
                    for new_acfg in new_varied_acfgs:
                        del new_acfg[key]

            varied_pcfgs = ut.take_column(grouped_pcfgs, 0)
            varied_acfgs = [
                ut.get_cfg_lbl(new_acfg_, with_name=False, sep=sep)
                for new_acfg_ in new_varied_acfgs
            ]

        def combo_lbls(lbla, lblp):
            parts = []
            if lbla != name_sep and lbla:
                parts.append(lbla)
            if lblp != name_sep and lblp:
                parts.append(lblp)
            return (sep + cfg_sep).join(parts)

        varied_lbls = [
            combo_lbls(lbla, lblp) for lblp, lbla in zip(varied_acfgs, varied_pcfgs)
        ]
        if shorten:
            varied_lbls = [testres._shorten_lbls(lbl) for lbl in varied_lbls]

        return varied_lbls

    def get_sorted_config_labels(testres):
        """
        helper
        """
        key = 'qx2_gt_rank'
        cfgx2_cumhist_percent, edges = testres.get_rank_percentage_cumhist(
            bins='dense', key=key
        )
        label_list = testres.get_short_cfglbls()
        label_list = [
            ('%6.2f%%' % (percent,))
            # ut.scalar_str(percent, precision=2)
            + ' - ' + label
            for percent, label in zip(cfgx2_cumhist_percent.T[0], label_list)
        ]
        sortx = cfgx2_cumhist_percent.T[0].argsort()[::-1]
        label_list = ut.take(label_list, sortx)
        return label_list

    def make_figtitle(testres, plotname='', filt_cfg=None):
        """
        Helper for consistent figure titles

        CommandLine:
            python -m wbia --tf TestResult.make_figtitle  --prefix "Seperability " --db GIRM_Master1   -a timectrl -t Ell:K=2     --hargv=scores
            python -m wbia --tf TestResult.make_figtitle

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> import wbia
            >>> ibs, testres = wbia.testdata_expts('PZ_MTEST')
            >>> plotname = ''
            >>> figtitle = testres.make_figtitle(plotname)
            >>> result = ('figtitle = %r' % (figtitle,))
            >>> print(result)
        """
        figtitle_prefix = ut.get_argval('--prefix', type_=str, default='')
        if figtitle_prefix != '':
            figtitle_prefix = figtitle_prefix.rstrip() + ' '
        figtitle = figtitle_prefix + plotname
        hasprefix = figtitle_prefix == ''
        if hasprefix:
            figtitle += '\n'

        title_aug = testres.get_title_aug(friendly=True, with_cfg=hasprefix)
        figtitle += ' ' + title_aug

        if filt_cfg is not None:
            filt_cfgstr = ut.get_cfg_lbl(filt_cfg)
            if filt_cfgstr.strip() != ':':
                figtitle += ' ' + filt_cfgstr
        return figtitle

    def get_title_aug(
        testres, with_size=True, with_db=True, with_cfg=True, friendly=False
    ):
        r"""
        Args:
            with_size (bool): (default = True)

        Returns:
            str: title_aug

        CommandLine:
            python -m wbia --tf TestResult.get_title_aug --db PZ_Master1 -a timequalctrl::timectrl

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> import wbia
            >>> ibs, testres = wbia.testdata_expts('PZ_MTEST')
            >>> with_size = True
            >>> title_aug = testres.get_title_aug(with_size)
            >>> res = u'title_aug = %s' % (title_aug,)
            >>> print(res)
        """
        ibs = testres.ibs
        title_aug = ''
        if with_db:
            title_aug += 'db=' + (ibs.get_dbname())
        if with_cfg:
            try:
                if '_cfgname' in testres.common_acfg['common']:
                    try:
                        annot_cfgname = testres.common_acfg['common']['_cfgstr']
                    except KeyError:
                        annot_cfgname = testres.common_acfg['common']['_cfgname']
                else:
                    cfgname_list = [
                        cfg['dcfg__cfgname'] for cfg in testres.varied_acfg_list
                    ]
                    cfgname_list = ut.unique_ordered(cfgname_list)
                    annot_cfgname = '[' + ','.join(cfgname_list) + ']'
                try:
                    pipeline_cfgname = testres.common_cfgdict['_cfgstr']
                except KeyError:
                    # pipeline_cfgname = testres.common_cfgdict['_cfgname']
                    cfgstr_list = [cfg['_cfgstr'] for cfg in testres.varied_cfg_list]
                    uniuqe_cfgstrs = ut.unique_ordered(cfgstr_list)
                    pipeline_cfgname = '[' + ','.join(uniuqe_cfgstrs) + ']'

                annot_cfgname = testres._shorten_lbls(annot_cfgname)
                pipeline_cfgname = testres._shorten_lbls(pipeline_cfgname)
                # hack turn these off if too long
                if len(annot_cfgname) < 64:
                    title_aug += ' a=' + annot_cfgname
                if len(pipeline_cfgname) < 64:
                    title_aug += ' t=' + pipeline_cfgname
            except Exception as ex:
                print(ut.repr2(testres.common_acfg))
                print(ut.repr2(testres.common_cfgdict))
                ut.printex(ex)
                raise
        if with_size:
            if ut.get_argflag('--hack_size_nl'):
                title_aug += '\n'
            if testres.has_constant_qaids():
                title_aug += ' #qaids=%r' % (len(testres.qaids),)
            elif testres.has_constant_length_qaids():
                title_aug += ' #qaids=%r*' % (len(testres.cfgx2_qaids[0]),)
            if testres.has_constant_daids():
                daids = testres.cfgx2_daids[0]
                title_aug += ' #daids=%r' % (len(testres.cfgx2_daids[0]),)
                if testres.has_constant_qaids():
                    all_daid_per_name_stats = ut.get_stats(
                        ibs.get_num_annots_per_name(daids)[0], use_nan=True
                    )
                    if all_daid_per_name_stats['std'] == 0:
                        title_aug += ' dper_name=%s' % (
                            ut.scalar_str(
                                all_daid_per_name_stats['mean'], max_precision=2
                            ),
                        )
                    else:
                        title_aug += ' dper_name=%s±%s' % (
                            ut.scalar_str(all_daid_per_name_stats['mean'], precision=2),
                            ut.scalar_str(all_daid_per_name_stats['std'], precision=2),
                        )
            elif testres.has_constant_length_daids():
                daids = testres.cfgx2_daids[0]
                title_aug += ' #daids=%r*' % (len(testres.cfgx2_daids[0]),)

        if friendly:
            # Hackiness for friendliness
            # title_aug = title_aug.replace('db=PZ_Master1', 'Plains Zebras')
            # title_aug = title_aug.replace('db=NNP_MasterGIRM_core', 'Masai Giraffes')
            # title_aug = title_aug.replace('db=GZ_ALL', 'Grevy\'s Zebras')
            title_aug = ut.multi_replace(
                title_aug,
                list(ibs.const.DBNAME_ALIAS.keys()),
                list(ibs.const.DBNAME_ALIAS.values()),
            )
            # title_aug = title_aug.replace('db=PZ_Master1', 'db=PZ')
            # title_aug = title_aug.replace('db=NNP_MasterGIRM_core', 'Masai Giraffes')
            # title_aug = title_aug.replace('db=GZ_ALL', 'Grevy\'s Zebras')
        return title_aug

    def get_fname_aug(testres, **kwargs):
        import re

        title_aug = testres.get_title_aug(**kwargs)
        valid_regex = '-a-zA-Z0-9_.() '
        valid_extra = '=,'
        valid_regex += valid_extra
        title_aug = title_aug.replace(' ', '_')  # spaces suck
        fname_aug = re.sub('[^' + valid_regex + ']+', '', title_aug)
        fname_aug = fname_aug.strip('_')
        return fname_aug

    def print_pcfg_info(testres):
        """
        Prints verbose information about each pipeline configuration

            >>> from wbia.expt.test_result import *  # NOQA
        """
        # TODO: Rectify with other printers
        # for pcfgx, (pipecfg, lbl) in enumerate(zip(pipecfg_list, pipecfg_lbls)):
        #     print('+--- %d / %d ===' % (pcfgx, (len(pipecfg_list))))
        #     ut.colorprint(lbl, 'white')
        #     print(pipecfg.get_cfgstr())
        #     print('L___')
        # for qreq_ in testres.cfgx2_qreq_:
        #     print(qreq_.get_full_cfgstr())
        # cfgdict_list = [qreq_.qparams for qreq_ in testres.cfgx2_qreq_]
        experiment_helpers.print_pipe_configs(testres.cfgx2_pcfg, testres.cfgx2_qreq_)

    def print_acfg_info(testres, **kwargs):
        """
        Prints verbose information about the annotations used in each test
        configuration

        CommandLine:
            python -m wbia --tf TestResult.print_acfg_info

        Kwargs:
            see ibs.get_annot_stats_dict
            hashid, per_name, per_qual, per_vp, per_name_vpedge, per_image,
            min_name_hourdist

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> import wbia
            >>> ibs, testres = wbia.testdata_expts('PZ_MTEST',
            >>>                                     a=['ctrl::unctrl_comp'],
            >>>                                     t=['candk:K=[1,2]'])
            >>> ibs = None
            >>> result = testres.print_acfg_info()
            >>> print(result)
        """
        from wbia.expt import annotation_configs

        ibs = testres.ibs
        # Get unique annotation configs
        cfgx2_acfg_label = annotation_configs.get_varied_acfg_labels(testres.cfgx2_acfg)
        flags = ut.flag_unique_items(cfgx2_acfg_label)
        qreq_list = ut.compress(testres.cfgx2_qreq_, flags)
        acfg_list = ut.compress(testres.cfgx2_acfg, flags)
        expanded_aids_list = [(qreq_.qaids, qreq_.daids) for qreq_ in qreq_list]
        annotation_configs.print_acfg_list(acfg_list, expanded_aids_list, ibs, **kwargs)

    def print_unique_annot_config_stats(testres, ibs=None):
        r"""
        Args:
            ibs (IBEISController): wbia controller object(default = None)

        CommandLine:
            python -m wbia TestResult.print_unique_annot_config_stats

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> import wbia
            >>> testres = wbia.testdata_expts('PZ_MTEST', a=['ctrl::unctrl_comp'])
            >>> ibs = None
            >>> result = testres.print_unique_annot_config_stats(ibs)
            >>> print(result)
        """
        if ibs is None:
            ibs = testres.ibs
        cfx2_dannot_hashid = [
            ibs.get_annot_hashid_visual_uuid(daids) for daids in testres.cfgx2_daids
        ]
        unique_daids = ut.compress(
            testres.cfgx2_daids, ut.flag_unique_items(cfx2_dannot_hashid)
        )
        with ut.Indenter('[acfgstats]'):
            print('+====')
            print('Printing %d unique annotconfig stats' % (len(unique_daids)))
            common_acfg = testres.common_acfg
            common_acfg['common'] = ut.dict_filter_nones(common_acfg['common'])
            print('testres.common_acfg = ' + ut.repr2(common_acfg))
            print(
                'param_basis(len(daids)) = %r' % (testres.get_param_basis('len(daids)'),)
            )
            for count, daids in enumerate(unique_daids):
                print('+---')
                print('acfgx = %r/%r' % (count, len(unique_daids)))
                if testres.has_constant_qaids():
                    ibs.print_annotconfig_stats(testres.qaids, daids)
                else:
                    ibs.print_annot_stats(daids, prefix='d')
                print('L___')

    def report(testres):
        testres.print_results()

    def print_results(testres, **kwargs):
        r"""
        CommandLine:
            python -m wbia --tf TestResult.print_results

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> from wbia.expt import harness
            >>> ibs, testres = harness.testdata_expts('PZ_MTEST')
            >>> result = testres.print_results()
            >>> print(result)
        """
        from wbia.expt import experiment_printres

        ibs = testres.ibs
        experiment_printres.print_results(ibs, testres, **kwargs)

    def get_common_qaids(testres):
        if not testres.has_constant_qaids():
            # Get only cases the tests share for now
            common_qaids = reduce(np.intersect1d, testres.cfgx2_qaids)
            return common_qaids
        else:
            return testres.qaids

    def get_all_qaids(testres):
        all_qaids = np.array(ut.unique(ut.flatten(testres.cfgx2_qaids)))
        return all_qaids

    def get_test_qaids(testres):
        # Transition fucntion
        return testres.get_all_qaids()
        # return testres.get_common_qaids()
        # all_qaids = ut.unique(ut.flatten(testres.cfgx2_qaids))
        # return all_qaids

    def get_all_tags(testres):
        r"""
        CommandLine:
            python -m wbia --tf TestResult.get_all_tags --db PZ_Master1 --show --filt :
            python -m wbia --tf TestResult.get_all_tags --db PZ_Master1 --show --filt :min_gf_timedelta=24h
            python -m wbia --tf TestResult.get_all_tags --db PZ_Master1 --show --filt :min_gf_timedelta=24h,max_gt_rank=5

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> from wbia.init import main_helpers
            >>> ibs, testres = main_helpers.testdata_expts('PZ_Master1', a=['timectrl'])
            >>> filt_cfg = main_helpers.testdata_filtcfg()
            >>> case_pos_list = testres.case_sample2(filt_cfg)
            >>> all_tags = testres.get_all_tags()
            >>> selected_tags = ut.take(all_tags, case_pos_list.T[0])
            >>> flat_tags = list(map(str, ut.flatten(ut.flatten(selected_tags))))
            >>> print(ut.repr2(ut.dict_hist(flat_tags), key_order_metric='val'))
            >>> ut.quit_if_noshow()
            >>> import wbia.plottool as pt
            >>> pt.word_histogram2(flat_tags, fnum=1, pnum=(1, 2, 1))
            >>> pt.wordcloud(' '.join(flat_tags), fnum=1, pnum=(1, 2, 2))
            >>> pt.set_figtitle(ut.get_cfg_lbl(filt_cfg))
            >>> ut.show_if_requested()
        """
        gt_tags = testres.get_gt_tags()
        gf_tags = testres.get_gf_tags()
        all_tags = [ut.list_zipflatten(*item) for item in zip(gf_tags, gt_tags)]
        return all_tags

    def get_gf_tags(testres):
        r"""
        Returns:
            list: case_pos_list

        CommandLine:
            python -m wbia --tf TestResult.get_gf_tags --db PZ_Master1 --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> from wbia.init import main_helpers
            >>> ibs, testres = main_helpers.testdata_expts('PZ_Master1', a=['timectrl'])
            >>> filt_cfg = main_helpers.testdata_filtcfg()
            >>> case_pos_list = testres.case_sample2(filt_cfg)
            >>> gf_tags = testres.get_gf_tags()
        """
        ibs = testres.ibs
        truth2_prop, prop2_mat = testres.get_truth2_prop()
        gf_annotmatch_rowids = truth2_prop['gf']['annotmatch_rowid']
        gf_tags = ibs.unflat_map(ibs.get_annotmatch_case_tags, gf_annotmatch_rowids)
        return gf_tags

    def get_gt_tags(testres):
        ibs = testres.ibs
        truth2_prop, prop2_mat = testres.get_truth2_prop()
        gt_annotmatch_rowids = truth2_prop['gt']['annotmatch_rowid']
        gt_tags = ibs.unflat_map(ibs.get_annotmatch_case_tags, gt_annotmatch_rowids)
        return gt_tags

    def get_gt_annot_tags(testres):
        ibs = testres.ibs
        truth2_prop, prop2_mat = testres.get_truth2_prop()
        gt_annot_tags = ibs.unflat_map(ibs.get_annot_case_tags, truth2_prop['gt']['aid'])
        return gt_annot_tags

    def get_query_annot_tags(testres):
        # FIXME: will break with new config structure
        ibs = testres.ibs
        truth2_prop, prop2_mat = testres.get_truth2_prop()
        unflat_qids = np.tile(testres.qaids[:, None], (len(testres.cfgx2_qaids)))
        query_annot_tags = ibs.unflat_map(ibs.get_annot_case_tags, unflat_qids)
        return query_annot_tags

    def get_gtquery_annot_tags(testres):
        gt_annot_tags = testres.get_gt_annot_tags()
        query_annot_tags = testres.get_query_annot_tags()
        both_tags = [
            [ut.flatten(t) for t in zip(*item)]
            for item in zip(query_annot_tags, gt_annot_tags)
        ]
        return both_tags

    def case_sample2(testres, filt_cfg, qaids=None, return_mask=False, verbose=None):
        r"""
        Filters individual test result cases based on how they performed, what
        tags they had, and various other things.

        Args:
            filt_cfg (dict):

        Returns:
            list: case_pos_list (list of (qx, cfgx)) or isvalid mask

        CommandLine:
            python -m wbia TestResult.case_sample2
            python -m wbia TestResult.case_sample2:0
            python -m wbia TestResult.case_sample2:1 --db GZ_ALL --filt :min_tags=1
            python -m wbia TestResult.case_sample2:1 --db PZ_Master1 --filt :min_gf_tags=1

            python -m wbia TestResult.case_sample2:2 --db PZ_Master1

        Example0:
            >>> # ENABLE_DOCTEST
            >>> # The same results is achievable with different filter config settings
            >>> from wbia.expt.test_result import *  # NOQA
            >>> from wbia.init import main_helpers
            >>> verbose = True
            >>> ibs, testres = main_helpers.testdata_expts('PZ_MTEST', a=['ctrl'])
            >>> filt_cfg1 = {'fail': True}
            >>> case_pos_list1 = testres.case_sample2(filt_cfg1)
            >>> filt_cfg2 = {'min_gtrank': 1}
            >>> case_pos_list2 = testres.case_sample2(filt_cfg2)
            >>> filt_cfg3 = {'min_gtrank': 0}
            >>> case_pos_list3 = testres.case_sample2(filt_cfg3)
            >>> filt_cfg4 = {}
            >>> case_pos_list4 = testres.case_sample2(filt_cfg4)
            >>> assert np.all(case_pos_list1 == case_pos_list2), 'should be equiv configs'
            >>> assert np.any(case_pos_list2 != case_pos_list3), 'should be diff configs'
            >>> assert np.all(case_pos_list3 == case_pos_list4), 'should be equiv configs'
            >>> ibs, testres = main_helpers.testdata_expts('PZ_MTEST', a=['ctrl'], t=['default:sv_on=[True,False]'])
            >>> filt_cfg5 = filt_cfg1.copy()
            >>> mask5 = testres.case_sample2(filt_cfg5, return_mask=True)
            >>> case_pos_list5 = testres.case_sample2(filt_cfg5, return_mask=False)
            >>> assert len(mask5.shape) == 2
            >>> assert np.all(mask5.T[0] == mask5.T[1])
            >>> filt_cfg6 = {'fail': True, 'allcfg': True}
            >>> mask6 = testres.case_sample2(filt_cfg6, return_mask=True)
            >>> assert np.all(mask6.T[0] == mask6.T[1])
            >>> print(mask5)
            >>> print(case_pos_list5)
            >>> filt_cfg = filt_cfg7 = {'disagree': True}
            >>> case_pos_list7 = testres.case_sample2(filt_cfg7, verbose=verbose)
            >>> print(case_pos_list7)

        Example1:
            >>> # SCRIPT
            >>> from wbia.expt.test_result import *  # NOQA
            >>> from wbia.init import main_helpers
            >>> ibs, testres = main_helpers.testdata_expts('PZ_MTEST', a=['ctrl'])
            >>> filt_cfg = main_helpers.testdata_filtcfg()
            >>> case_pos_list = testres.case_sample2(filt_cfg)
            >>> result = ('case_pos_list = %s' % (str(case_pos_list),))
            >>> print(result)
            >>> # Extra stuff
            >>> all_tags = testres.get_all_tags()
            >>> selcted_tags = ut.take(all_tags, case_pos_list.T[0])
            >>> print('selcted_tags = %r' % (selcted_tags,))

        Example1:
            >>> # DISABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> from wbia.init import main_helpers
            >>> ibs, testres = main_helpers.testdata_expts('PZ_MTEST', a=['ctrl'], t=['default:K=[1,2,3]'])
            >>> ut.exec_funckw(testres.case_sample2, globals())
            >>> filt_cfg = {'fail': True, 'min_gtrank': 1, 'max_gtrank': None, 'min_gf_timedelta': '24h'}
            >>> ibs, testres = main_helpers.testdata_expts('humpbacks_fb', a=['default:has_any=hasnotch,mingt=2,qindex=0:300,dindex=0:300'], t=['default:proot=BC_DTW,decision=max,crop_dim_size=500,crop_enabled=True,manual_extract=False,use_te_scorer=True,ignore_notch=True,te_net=annot_simple', 'default:proot=vsmany'], qaid_override=[12])
            >>> filt_cfg = ':disagree=True,index=0:8,min_gtscore=.00001,require_all_cfg=True'
            >>> #filt_cfg = cfghelpers.parse_argv_cfg('--filt')[0]
            >>> case_pos_list = testres.case_sample2(filt_cfg, verbose=True)
            >>> result = ('case_pos_list = %s' % (str(case_pos_list),))
            >>> print(result)
            >>> # Extra stuff
            >>> all_tags = testres.get_all_tags()
            >>> selcted_tags = ut.take(all_tags, case_pos_list.T[0])
            >>> print('selcted_tags = %r' % (selcted_tags,))


            print('qaid = %r' % (qaid,))
            print('qx = %r' % (qx,))
            print('cfgxs = %r' % (cfgxs,))
            # print testres info about this item
            take_cfgs = ut.partial(ut.take, index_list=cfgxs)
            take_qx = ut.partial(ut.take, index_list=qx)
            truth_cfgs = ut.hmap_vals(take_qx, truth2_prop)
            truth_item = ut.hmap_vals(take_cfgs, truth_cfgs, max_depth=1)
            prop_cfgs = ut.hmap_vals(take_qx, prop2_mat)
            prop_item = ut.hmap_vals(take_cfgs, prop_cfgs, max_depth=0)
            print('truth2_prop[item] = ' + ut.repr3(truth_item, nl=2))
            print('prop2_mat[item] = ' + ut.repr3(prop_item, nl=1))
        """
        from wbia.expt import cfghelpers

        if verbose is None:
            verbose = ut.NOT_QUIET
        if verbose:
            print('[testres] case_sample2')

        if isinstance(filt_cfg, six.string_types):
            filt_cfg = [filt_cfg]
        if isinstance(filt_cfg, list):
            _combos = cfghelpers.parse_cfgstr_list2(filt_cfg, strict=False)
            filt_cfg = ut.flatten(_combos)[0]
        if isinstance(filt_cfg, six.string_types):
            _combos = cfghelpers.parse_cfgstr_list2([filt_cfg], strict=False)
            filt_cfg = ut.flatten(_combos)[0]
        if filt_cfg is None:
            filt_cfg = {}

        qaids = testres.get_test_qaids() if qaids is None else qaids
        truth2_prop, prop2_mat = testres.get_truth2_prop(qaids)
        ibs = testres.ibs

        # Initialize isvalid flags to all true
        # np.ones(prop2_mat['is_success'].shape, dtype=np.bool)
        participates = prop2_mat['participates']
        is_valid = participates.copy()

        def unflat_tag_filterflags(tags_list, **kwargs):
            from wbia import tag_funcs

            flat_tags, cumsum = ut.invertible_flatten2(tags_list)
            flat_flags = tag_funcs.filterflags_general_tags(flat_tags, **kwargs)
            flags = np.array(ut.unflatten2(flat_flags, cumsum))
            return flags

        UTFF = unflat_tag_filterflags

        def cols_disagree(mat, val):
            """
            is_success = prop2_mat['is_success']
            """
            nCols = mat.shape[1]
            sums = mat.sum(axis=1)
            # Find out which rows have different values
            disagree_flags1d = np.logical_and(sums > 0, sums < nCols)
            disagree_flags2d = np.tile(disagree_flags1d[:, None], (1, nCols))
            if not val:
                # User asked for rows that agree
                flags = np.logical_not(disagree_flags2d)
            else:
                flags = disagree_flags2d
            return flags

        def cfg_scoresep(mat, val, op):
            """
            Compares scores between different configs

            op = operator.ge
            is_success = prop2_mat['is_success']
            """
            # import scipy.spatial.distance as spdist
            nCols = mat.shape[1]
            pdistx = vt.pdist_indicies(nCols)
            pdist_list = np.array([vt.safe_pdist(row) for row in mat])
            flags_list = op(pdist_list, val)
            colx_list = [
                np.unique(ut.flatten(ut.compress(pdistx, flags))) for flags in flags_list
            ]
            offsets = np.arange(0, nCols * len(mat), step=nCols)
            idx_list = ut.flatten(
                [colx + offset for colx, offset in zip(colx_list, offsets)]
            )
            mask = vt.index_to_boolmask(idx_list, maxval=offsets[-1] + nCols)
            flags = mask.reshape(mat.shape)
            return flags

        # List of rules that can filter results
        rule_list = [
            ('disagree', lambda val: cols_disagree(prop2_mat['is_failure'], val)),
            (
                'min_gt_cfg_scoresep',
                lambda val: cfg_scoresep(truth2_prop['gt']['score'], val, operator.ge),
            ),
            ('fail', prop2_mat['is_failure']),
            ('success', prop2_mat['is_success']),
            ('min_gtrank', partial(operator.ge, truth2_prop['gt']['rank'])),
            ('max_gtrank', partial(operator.le, truth2_prop['gt']['rank'])),
            ('max_gtscore', partial(operator.le, truth2_prop['gt']['score'])),
            ('min_gtscore', partial(operator.ge, truth2_prop['gt']['score'])),
            ('min_gf_timedelta', partial(operator.ge, truth2_prop['gf']['timedelta'])),
            ('max_gf_timedelta', partial(operator.le, truth2_prop['gf']['timedelta'])),
            # Tag filtering
            # FIXME: will break with new config structure
            ('min_tags', lambda val: UTFF(testres.get_all_tags(), min_num=val)),
            ('max_tags', lambda val: UTFF(testres.get_all_tags(), max_num=val)),
            ('min_gf_tags', lambda val: UTFF(testres.get_gf_tags(), min_num=val)),
            ('max_gf_tags', lambda val: UTFF(testres.get_gf_tags(), max_num=val)),
            ('min_gt_tags', lambda val: UTFF(testres.get_gt_tags(), min_num=val)),
            ('max_gt_tags', lambda val: UTFF(testres.get_gt_tags(), max_num=val)),
            (
                'min_query_annot_tags',
                lambda val: UTFF(testres.get_query_annot_tags(), min_num=val),
            ),
            (
                'min_gt_annot_tags',
                lambda val: UTFF(testres.get_gt_annot_tags(), min_num=val),
            ),
            (
                'min_gtq_tags',
                lambda val: UTFF(testres.get_gtquery_annot_tags(), min_num=val),
            ),
            (
                'max_gtq_tags',
                lambda val: UTFF(testres.get_gtquery_annot_tags(), max_num=val),
            ),
            ('without_gf_tag', lambda val: UTFF(testres.get_gf_tags(), has_none=val)),
            ('without_gt_tag', lambda val: UTFF(testres.get_gt_tags(), has_none=val)),
            ('with_gf_tag', lambda val: UTFF(testres.get_gf_tags(), has_any=val)),
            ('with_gt_tag', lambda val: UTFF(testres.get_gt_tags(), has_any=val)),
            ('with_tag', lambda val: UTFF(testres.get_all_tags(), has_any=val)),
            ('without_tag', lambda val: UTFF(testres.get_all_tags(), has_none=val)),
        ]
        rule_dict = ut.odict(rule_list)
        rule_list.append(('max_gf_td', rule_dict['max_gf_timedelta']))
        rule_list.append(('min_gf_td', rule_dict['min_gf_timedelta']))

        filt_cfg_ = copy.deepcopy(filt_cfg)

        # hack to convert to seconds
        for tdkey in filt_cfg_.keys():
            # timedelta_keys = ['min_gf_timedelta', 'max_gf_timedelta']
            # for tdkey in timedelta_keys:
            if tdkey.endswith('_timedelta'):
                filt_cfg_[tdkey] = ut.ensure_timedelta(filt_cfg_[tdkey])

        class VerbFilterInfo(object):
            def __init__(self):
                self.prev_num_valid = None

            def print_pre(self, is_valid, filt_cfg_):
                num_valid = is_valid.sum()
                print(
                    '[testres] Sampling from is_valid.size=%r with filt=%r'
                    % (is_valid.size, ut.get_cfg_lbl(filt_cfg_))
                )
                print('  * is_valid.shape = %r' % (is_valid.shape,))
                print('  * num_valid = %r' % (num_valid,))
                self.prev_num_valid = num_valid

            def print_post(self, is_valid, flags, msg):
                if flags is not None:
                    num_passed = flags.sum()
                num_valid = is_valid.sum()
                num_invalidated = self.prev_num_valid - num_valid
                print(msg)
                if num_invalidated == 0:
                    if flags is not None:
                        print('  * num_passed = %r' % (num_passed,))
                    print('  * num_invalided = %r' % (num_invalidated,))
                else:
                    print('  * prev_num_valid = %r' % (self.prev_num_valid,))
                    print('  * num_valid = %r' % (num_valid,))
                    # print('  * is_valid.shape = %r' % (is_valid.shape,))
                self.prev_num_valid = num_valid

        verbinfo = VerbFilterInfo()

        if verbose:
            verbinfo.print_pre(is_valid, filt_cfg_)

        # Pop irrelevant info
        ut.delete_keys(filt_cfg_, ['_cfgstr', '_cfgindex', '_cfgname', '_cfgtype'])
        # Pop other non-rule config options
        valid_rules = []

        def poprule(rulename, default):
            # register other rule names for debuging
            valid_rules.append(rulename)
            return filt_cfg_.pop(rulename, default)

        allcfg = poprule('allcfg', None)
        orderby = poprule('orderby', None)
        reverse = poprule('reverse', None)
        sortasc = poprule('sortasc', None)
        sortdsc = poprule('sortdsc', poprule('sortdesc', None))
        max_pername = poprule('max_pername', None)
        require_all_cfg = poprule('require_all_cfg', None)
        index = poprule('index', None)
        # Pop all chosen rules
        rule_value_list = [poprule(key, None) for key, rule in rule_list]

        # Assert that only valid configurations were given
        if len(filt_cfg_) > 0:
            print('ERROR')
            print('filtcfg valid rules are = %s' % (ut.repr2(valid_rules, nl=1),))
            for key in filt_cfg_.keys():
                print(
                    'did you mean %r instead of %r?'
                    % (ut.closet_words(key, valid_rules)[0], key)
                )
            raise NotImplementedError(
                'Unhandled filt_cfg.keys() = %r' % (filt_cfg_.keys())
            )

        # Remove test cases that do not satisfy chosen rules
        chosen_rule_idxs = ut.where([val is not None for val in rule_value_list])
        chosen_rules = ut.take(rule_list, chosen_rule_idxs)
        chosen_vals = ut.take(rule_value_list, chosen_rule_idxs)
        for (key, rule), val in zip(chosen_rules, chosen_vals):
            if isinstance(rule, np.ndarray):
                # When a rule is an ndarray it must have boolean values
                flags = rule == val
            else:
                flags = rule(val)
            # HACK: flags are forced to be false for non-participating cases
            flags = np.logical_and(flags, participates)
            # conjunctive normal form of satisfiability
            is_valid = np.logical_and(is_valid, flags)
            if verbose:
                verbinfo.print_post(is_valid, flags, 'SampleRule: %s = %r' % (key, val))

        # HACK:
        # If one config for a row passes the filter then all configs should pass
        if allcfg:
            is_valid = np.logical_or(np.logical_or.reduce(is_valid.T)[:, None], is_valid)
            is_valid = np.logical_and(is_valid, participates)

        qx_list, cfgx_list = np.nonzero(is_valid)

        # Determine a good ordering of the test cases
        if sortdsc is not None:
            assert orderby is None, 'use orderby or sortasc'
            assert reverse is None, 'reverse does not work with sortdsc'
            orderby = sortdsc
            reverse = True
        elif sortasc is not None:
            assert reverse is None, 'reverse does not work with sortasc'
            assert orderby is None, 'use orderby or sortasc'
            orderby = sortasc
            reverse = False
        else:
            reverse = False
        if orderby is not None:
            # if orderby == 'gtscore':
            #    order_values = truth2_prop['gt']['score']
            # elif orderby == 'gfscore':
            #    order_values = truth2_prop['gf']['score']
            # else:
            import re

            order_values = None
            for prefix_pattern in ['^gt_?', '^gf_?']:
                prefix_match = re.match(prefix_pattern, orderby)
                if prefix_match is not None:
                    truth = prefix_pattern[1:3]
                    propname = orderby[prefix_match.end() :]
                    if verbose:
                        print('Ordering by truth=%s propname=%s' % (truth, propname))
                    order_values = truth2_prop[truth][propname]
                    break
            if order_values is None:
                raise NotImplementedError('Unknown orerby=%r' % (orderby,))
        else:
            order_values = np.arange(is_valid.size).reshape(is_valid.shape)

        # Convert mask into indicies
        flat_order = order_values[is_valid]
        # Flat sorting indeices in a matrix
        if verbose:
            if verbose:
                print('Reversing ordering (descending)')
            else:
                print('Normal ordering (ascending)')
        if reverse:
            sortx = flat_order.argsort()[::-1]
        else:
            sortx = flat_order.argsort()
        qx_list = qx_list.take(sortx, axis=0)
        cfgx_list = cfgx_list.take(sortx, axis=0)

        # Return at most ``max_pername`` annotation examples per name
        if max_pername is not None:
            if verbose:
                print('Returning at most %d cases per name ' % (max_pername,))
            # FIXME: multiple configs
            _qaid_list = np.take(qaids, qx_list)
            _qnid_list = ibs.get_annot_nids(_qaid_list)
            _valid_idxs = []
            seen_ = ut.ddict(lambda: 0)
            for idx, _qnid in enumerate(_qnid_list):
                if seen_[_qnid] < max_pername:
                    seen_[_qnid] += 1
                    _valid_idxs.append(idx)
            _qx_list = qx_list[_valid_idxs]
            _cfgx_list = cfgx_list[_valid_idxs]
            _valid_index = np.vstack((_qx_list, _cfgx_list)).T
            is_valid = vt.index_to_boolmask(_valid_index, is_valid.shape, isflat=False)
            qx_list = _qx_list
            cfgx_list = _cfgx_list

        if require_all_cfg:
            if verbose:
                prev_num_valid = is_valid.sum()
                print('Enforcing that all configs must pass filters')
                print('  * prev_num_valid = %r' % (prev_num_valid,))
            qx2_valid_cfgs = ut.group_items(cfgx_list, qx_list)
            hasall_cfg = [len(qx2_valid_cfgs[qx]) == testres.nConfig for qx in qx_list]
            _qx_list = qx_list.compress(hasall_cfg)
            _cfgx_list = cfgx_list.compress(hasall_cfg)
            _valid_index = np.vstack((_qx_list, _cfgx_list)).T
            is_valid = vt.index_to_boolmask(_valid_index, is_valid.shape, isflat=False)
            qx_list = _qx_list
            cfgx_list = _cfgx_list
            if verbose:
                verbinfo.print_post(
                    is_valid, None, 'Enforcing that all configs must pass filters'
                )

        if index is not None:
            if isinstance(index, six.string_types):
                index = ut.smart_cast(index, slice)
            _qx_list = ut.take(qx_list, index)
            _cfgx_list = ut.take(cfgx_list, index)
            _valid_index = np.vstack((_qx_list, _cfgx_list)).T
            is_valid = vt.index_to_boolmask(_valid_index, is_valid.shape, isflat=False)
            qx_list = _qx_list
            cfgx_list = _cfgx_list
            if verbose:
                verbinfo.print_post(
                    is_valid,
                    None,
                    'Taking index=%r sample from len(qx_list) = %r'
                    % (index, len(qx_list),),
                )

        if not return_mask:
            case_pos_list = np.vstack((qx_list, cfgx_list)).T
            case_identifier = case_pos_list
        else:
            if verbose:
                print('Converting cases indicies to a 2d-mask')
            case_identifier = is_valid
        if verbose:
            print('Finished case filtering')
            print('Final case stats:')
            qx_hist = ut.dict_hist(qx_list)
            print('config per query stats: %r' % (ut.get_stats_str(qx_hist.values()),))
            print(
                'query per config stats: %r'
                % (ut.get_stats_str(ut.dict_hist(cfgx_list).values()),)
            )

        return case_identifier

    def get_truth2_prop(testres, qaids=None, join_acfg=False):
        r"""
        Returns:
            tuple: (truth2_prop, prop2_mat)

        CommandLine:
            python -m wbia.expt.test_result --exec-get_truth2_prop --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> import wbia
            >>> ibs, testres = wbia.testdata_expts('PZ_MTEST', a=['ctrl'])
            >>> (truth2_prop, prop2_mat) = testres.get_truth2_prop()
            >>> result = '(truth2_prop, prop2_mat) = %s' % str((truth2_prop, prop2_mat))
            >>> print(result)
            >>> ut.quit_if_noshow()
            >>> import wbia.plottool as pt
            >>> ut.show_if_requested()
        """
        ibs = testres.ibs
        test_qaids = testres.get_test_qaids() if qaids is None else qaids

        # test_qaids = ut.random_sample(test_qaids, 20)
        truth2_prop = ut.ddict(ut.odict)

        # TODO: have this function take in a case_pos_list as input instead
        participates = testres.get_infoprop_mat('participant', test_qaids)

        truth2_prop['gt']['aid'] = testres.get_infoprop_mat('qx2_gt_aid', test_qaids)
        truth2_prop['gf']['aid'] = testres.get_infoprop_mat('qx2_gf_aid', test_qaids)
        truth2_prop['gt']['rank'] = testres.get_infoprop_mat('qx2_gt_rank', test_qaids)
        truth2_prop['gf']['rank'] = testres.get_infoprop_mat('qx2_gf_rank', test_qaids)

        truth2_prop['gt']['score'] = testres.get_infoprop_mat(
            'qx2_gt_raw_score', test_qaids
        )
        truth2_prop['gf']['score'] = testres.get_infoprop_mat(
            'qx2_gf_raw_score', test_qaids
        )
        truth2_prop['gt']['score'] = np.nan_to_num(truth2_prop['gt']['score'])
        truth2_prop['gf']['score'] = np.nan_to_num(truth2_prop['gf']['score'])

        # Cast nans to ints (that are participants)
        # if False:
        for truth in ['gt', 'gf']:
            rank_mat = truth2_prop[truth]['rank']
            flags = np.logical_and(np.isnan(rank_mat), participates)
            rank_mat[flags] = testres.get_worst_possible_rank()
            # truth2_prop[truth]['rank'] = rank_mat.astype(np.int)

        is_success = truth2_prop['gt']['rank'] == 0
        is_failure = np.logical_not(is_success)

        # THIS IS NOT THE CASE IF THERE ARE UNKNOWN INDIVIDUALS IN THE DATABASE
        assert np.all(is_success == (truth2_prop['gt']['rank'] == 0))

        # WEIRD THINGS HAPPEN WHEN UNKNOWNS ARE HERE
        # hardness_degree_rank[is_success]
        # These probably just completely failure spatial verification
        # is_weird = hardness_degree_rank == 0

        # Get timedelta and annotmatch rowid
        for truth in ['gt', 'gf']:
            aid_mat = truth2_prop[truth]['aid']
            timedelta_mat = np.vstack(
                [ibs.get_annot_pair_timedelta(test_qaids, aids) for aids in aid_mat.T]
            ).T
            annotmatch_rowid_mat = np.vstack(
                [
                    ibs.get_annotmatch_rowid_from_undirected_superkey(test_qaids, aids)
                    for aids in aid_mat.T
                ]
            ).T
            truth2_prop[truth]['annotmatch_rowid'] = annotmatch_rowid_mat
            truth2_prop[truth]['timedelta'] = timedelta_mat
        prop2_mat = {}

        prop2_mat['is_success'] = is_success
        prop2_mat['is_failure'] = is_failure
        prop2_mat['participates'] = participates

        groupxs = testres.get_cfgx_groupxs()

        def group_prop(val, grouped_flags, groupxs):
            nRows = len(val)
            # Allocate space for new val
            new_shape = (nRows, len(groupxs))
            if val.dtype == object or val.dtype.type == object:
                new_val = np.full(new_shape, None, dtype=val.dtype)
            elif ut.is_float(val):
                new_val = np.full(new_shape, np.nan, dtype=val.dtype)
            else:
                new_val = np.zeros(new_shape, dtype=val.dtype)
            # Populate new val
            grouped_vals = vt.apply_grouping(val.T, groupxs)
            _iter = enumerate(zip(grouped_flags, grouped_vals))
            for new_col, (flags, group) in _iter:
                rows, cols = np.where(flags.T)
                new_val[rows, new_col] = group.T[(rows, cols)]
            return new_val

        if join_acfg:
            assert ut.allsame(participates.sum(axis=1))
            grouped_flags = vt.apply_grouping(participates.T, groupxs)

            # new_prop2_mat = {key: group_prop(val)
            #                 for key, val in prop2_mat.items()}
            # new_truth2_prop = {
            #    truth: {key: group_prop(val)
            #            for key, val in props.items()}
            #    for truth, props in truth2_prop.items()}

            new_prop2_mat = {}
            for key, val in prop2_mat.items():
                new_prop2_mat[key] = group_prop(val, grouped_flags, groupxs)

            new_truth2_prop = {}
            for truth, props in truth2_prop.items():
                new_props = {}
                for key, val in props.items():
                    new_props[key] = group_prop(val, grouped_flags, groupxs)
                new_truth2_prop[truth] = new_props

            prop2_mat_ = new_prop2_mat
            truth2_prop_ = new_truth2_prop
        else:
            prop2_mat_ = prop2_mat
            truth2_prop_ = truth2_prop
        return truth2_prop_, prop2_mat_

    def interact_individual_result(testres, qaid, cfgx=0):
        ibs = testres.ibs
        cfgx_list = ut.ensure_iterable(cfgx)
        qreq_list = ut.take(testres.cfgx2_qreq_, cfgx_list)
        # Preload any requested configs
        cm_list = [qreq_.execute(qaids=[qaid]) for qreq_ in qreq_list]
        cfgx2_shortlbl = testres.get_short_cfglbls()
        show_kwargs = {
            'N': 3,
            'ori': True,
            'ell_alpha': 0.9,
        }
        # SHOW ANALYSIS
        show_kwargs['show_query'] = False
        show_kwargs['viz_name_score'] = True
        show_kwargs['show_timedelta'] = True
        show_kwargs['show_gf'] = True
        show_kwargs['with_figtitle'] = False
        for cfgx, cm, qreq_ in zip(cfgx_list, cm_list, qreq_list):
            query_lbl = cfgx2_shortlbl[cfgx]
            fnum = cfgx
            cm.ishow_analysis(
                ibs,
                figtitle=query_lbl,
                fnum=fnum,
                annot_mode=1,
                qreq_=qreq_,
                **show_kwargs
            )

    def draw_score_diff_disti(testres):
        r"""

        CommandLine:
            python -m wbia --tf TestResult.draw_score_diff_disti --show -a varynannots_td -t best --db PZ_Master1
            python -m wbia --tf TestResult.draw_score_diff_disti --show -a varynannots_td -t best --db GZ_Master1
            python -m wbia --tf TestResult.draw_score_diff_disti --show -a varynannots_td1h -t best --db GIRM_Master1

            python -m wbia --tf TestResult.draw_score_diff_disti --show -a varynannots_td:qmin_pername=3,dpername=2 -t best --db PZ_Master1

            python -m wbia --tf get_annotcfg_list -a varynannots_td -t best --db PZ_Master1
            13502
            python -m wbia --tf draw_match_cases --db PZ_Master1 -a varynannots_td:dsample_size=.01 -t best  --show --qaid 13502
            python -m wbia --tf draw_match_cases --db PZ_Master1 -a varynannots_td -t best  --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
            >>> import wbia
            >>> ibs, testres = wbia.testdata_expts('PZ_Master1', a=['varynannots_td'], t=['best'])
            >>> result = testres.draw_score_diff_disti()
            >>> print(result)
            >>> ut.show_if_requested()
        """
        import wbia.plottool as pt
        import vtool as vt

        # dont look at filtered cases
        ibs = testres.ibs
        qaids = testres.get_test_qaids()
        qaids = ibs.get_annot_tag_filterflags(qaids, {'has_none': 'timedeltaerror'})

        gt_rawscore = testres.get_infoprop_mat('qx2_gt_raw_score', qaids=qaids)
        gf_rawscore = testres.get_infoprop_mat('qx2_gf_raw_score', qaids=qaids)

        gt_valid_flags_list = np.isfinite(gt_rawscore).T
        gf_valid_flags_list = np.isfinite(gf_rawscore).T

        cfgx2_gt_scores = vt.zipcompress(gt_rawscore.T, gt_valid_flags_list)
        cfgx2_gf_scores = vt.zipcompress(gf_rawscore.T, gf_valid_flags_list)

        # partition by rank
        gt_rank = testres.get_infoprop_mat('qx2_gt_rank', qaids=qaids)
        gf_ranks = testres.get_infoprop_mat('qx2_gf_rank', qaids=qaids)
        cfgx2_gt_ranks = vt.zipcompress(gt_rank.T, gt_valid_flags_list)
        cfgx2_rank0_gt_scores = vt.zipcompress(
            cfgx2_gt_scores, [ranks == 0 for ranks in cfgx2_gt_ranks]
        )
        cfgx2_rankX_gt_scores = vt.zipcompress(
            cfgx2_gt_scores, [ranks > 0 for ranks in cfgx2_gt_ranks]
        )
        cfgx2_gf_ranks = vt.zipcompress(gf_ranks.T, gf_valid_flags_list)
        cfgx2_rank0_gf_scores = vt.zipcompress(
            cfgx2_gf_scores, [ranks == 0 for ranks in cfgx2_gf_ranks]
        )

        # valid_gtranks = gt_rank[isvalid]
        # valid_qaids = qaids[isvalid]
        # Hack remove timedelta error
        # valid_qaids = valid_qaids[flags]
        # valid_gt_rawscore = valid_gt_rawscore[flags]
        # valid_gtranks = valid_gtranks[flags]

        xdata = list(map(len, testres.cfgx2_daids))

        USE_MEDIAN = True  # not ut.get_argflag('--use-mean')
        # USE_LOG = True
        USE_LOG = False
        if USE_MEDIAN:
            ave = np.median
            dev = vt.median_abs_dev
        else:
            ave = np.mean
            dev = np.std

        def make_interval_args(arr_list, ave=ave, dev=dev, **kwargs):
            # if not USE_MEDIAN:
            #    # maybe approximate median by removing the most extreme values
            #    arr_list = [np.array(sorted(arr))[5:-5] for arr in arr_list]
            import utool as ut

            if USE_LOG:
                arr_list = list(map(lambda x: np.log(x + 1), arr_list))
            sizes_ = list(map(len, arr_list))
            ydata_ = list(map(ave, arr_list))
            spread_ = list(map(dev, arr_list))
            # ut.get_stats(arr_list, axis=0)
            label = kwargs.get('label', '')
            label += ' ' + ut.get_funcname(ave)
            kwargs['label'] = label
            print(
                label
                + 'score stats : '
                + ut.repr2(
                    ut.get_jagged_stats(arr_list, use_median=True), nl=1, precision=1
                )
            )
            return ydata_, spread_, kwargs, sizes_

        args_list1 = [
            make_interval_args(cfgx2_gt_scores, label='GT', color=pt.TRUE_BLUE),
            make_interval_args(cfgx2_gf_scores, label='GF', color=pt.FALSE_RED),
        ]

        args_list2 = [
            make_interval_args(
                cfgx2_rank0_gt_scores, label='GT-rank = 0', color=pt.LIGHT_GREEN
            ),
            make_interval_args(
                cfgx2_rankX_gt_scores, label='GT-rank > 0', color=pt.YELLOW
            ),
            make_interval_args(cfgx2_rank0_gf_scores, label='GF-rank = 0', color=pt.PINK),
            # make_interval_args(cfgx2_rank2_gt_scores, label='gtrank < 2'),
        ]

        plotargs_list = [args_list1, args_list2]
        # plotargs_list = [args_list1]
        ymax = -np.inf
        ymin = np.inf
        for args_list in plotargs_list:
            ydata_list = np.array(ut.get_list_column(args_list, 0))
            spread = np.array(ut.get_list_column(args_list, 1))
            ymax = max(ymax, np.array(ydata_list + spread).max())
            ymin = min(ymax, np.array(ydata_list - spread).min())

        ylabel = 'log name score' if USE_LOG else 'name score'

        statickw = dict(
            # title='scores vs dbsize',
            xlabel='database size (number of annotations)',
            ylabel=ylabel,
            # xscale='log', ymin=0, ymax=10,
            linewidth=2,
            spread_alpha=0.5,
            lightbg=True,
            marker='o',
            # xmax='data',
            ymax=ymax,
            ymin=ymin,
            xmax='data',
            xmin='data',
        )

        fnum = pt.ensure_fnum(None)
        pnum_ = pt.make_pnum_nextgen(len(plotargs_list), 1)

        for args_list in plotargs_list:
            ydata_list = ut.get_list_column(args_list, 0)
            spread_list = ut.get_list_column(args_list, 1)
            kwargs_list = ut.get_list_column(args_list, 2)
            sizes_list = ut.get_list_column(args_list, 3)
            print('sizes_list = %s' % (ut.repr2(sizes_list, nl=1),))

            # Pack kwargs list for multi_plot
            plotkw = ut.dict_stack2(kwargs_list, '_list')
            plotkw2 = ut.merge_dicts(statickw, plotkw)

            pt.multi_plot(
                xdata,
                ydata_list,
                spread_list=spread_list,
                fnum=fnum,
                pnum=pnum_(),
                **plotkw2
            )

        # pt.adjust_subplots(hspace=.3)
        figtitle = 'Score vs DBSize: %s' % (testres.get_title_aug())
        pt.set_figtitle(figtitle)

    def draw_rank_cmc(testres):
        """
        Wrapper
        """
        from wbia.expt import experiment_drawing

        experiment_drawing.draw_rank_cmc(testres.ibs, testres)

    def draw_match_cases(testres, **kwargs):
        """
        Wrapper
        """
        from wbia.expt import experiment_drawing

        experiment_drawing.draw_match_cases(testres.ibs, testres, **kwargs)

    def draw_failure_cases(testres, **kwargs):
        """
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> ibs, testres = wbia.testdata_expts(defaultdb='PZ_MTEST', a='timectrl:qsize=2', t='invar:AI=[False],RI=False', use_cache=False)
        """
        from wbia.expt import experiment_drawing

        # kwargs = kwargs.copy()
        orig_filter = ':'
        kwargs['f'] = orig_filter + 'fail'
        case_pos_list = testres.case_sample2(':fail=True,index=0:5')
        experiment_drawing.draw_match_cases(
            testres.ibs,
            testres,
            case_pos_list=case_pos_list,
            annot_modes=[1],
            interact=True,
        )

    def find_score_thresh_cutoff(testres):
        """
        FIXME
        DUPLICATE CODE
        rectify with experiment_drawing
        """
        # import wbia.plottool as pt
        import vtool as vt

        if ut.VERBOSE:
            print('[dev] FIX DUPLICATE CODE find_thresh_cutoff')
        # from wbia.expt import cfghelpers

        assert len(testres.cfgx2_qreq_) == 1, 'can only specify one config here'
        cfgx = 0
        # qreq_ = testres.cfgx2_qreq_[cfgx]
        test_qaids = testres.get_test_qaids()
        gt_rawscore = testres.get_infoprop_mat('qx2_gt_raw_score', qaids=test_qaids).T[
            cfgx
        ]
        gf_rawscore = testres.get_infoprop_mat('qx2_gf_raw_score', qaids=test_qaids).T[
            cfgx
        ]

        # FIXME: may need to specify which cfg is used in the future
        # isvalid = testres.case_sample2(filt_cfg, return_mask=True).T[cfgx]

        tp_nscores = gt_rawscore
        tn_nscores = gf_rawscore
        tn_qaids = tp_qaids = test_qaids
        # encoder = vt.ScoreNormalizer(target_tpr=.7)
        # print(qreq_.get_cfgstr())
        part_attrs = {1: {'qaid': tp_qaids}, 0: {'qaid': tn_qaids}}

        fpr = None
        tpr = 0.85
        encoder = vt.ScoreNormalizer(adjust=8, fpr=fpr, tpr=tpr, monotonize=True)
        # tp_scores = tp_nscores
        # tn_scores = tn_nscores
        name_scores, labels, attrs = encoder._to_xy(tp_nscores, tn_nscores, part_attrs)
        encoder.fit(name_scores, labels, attrs)
        score_thresh = encoder.learn_threshold2()

        # Find intersection point
        # TODO: add to score normalizer.
        # Improve robustness
        # pt.figure()
        # pt.plot(xdata, curve)
        # pt.plot(x_submax, y_submax, 'o')
        return score_thresh

    def print_percent_identification_success(testres):
        """
        Prints names identified (at rank 1) / names queried.
        This combines results over multiple queries of a particular name using
        max

        OLD, MAYBE DEPRIATE

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.expt.test_result import *  # NOQA
        """
        ibs = testres.ibs
        qaids = testres.get_test_qaids()
        unique_nids, groupxs = ut.group_indices(ibs.get_annot_nids(qaids))

        qx2_gt_raw_score = testres.get_infoprop_mat('qx2_gt_raw_score', qaids=qaids)
        qx2_gf_raw_score = testres.get_infoprop_mat('qx2_gf_raw_score', qaids=qaids)

        nx2_gt_raw_score = np.array(
            [
                np.nanmax(scores, axis=0)
                for scores in vt.apply_grouping(qx2_gt_raw_score, groupxs)
            ]
        )

        nx2_gf_raw_score = np.array(
            [
                np.nanmax(scores, axis=0)
                for scores in vt.apply_grouping(qx2_gf_raw_score, groupxs)
            ]
        )

        cfgx2_success = (nx2_gt_raw_score > nx2_gf_raw_score).T
        print('Identification success (names identified / names queried)')
        for cfgx, success in enumerate(cfgx2_success):
            pipelbl = testres.cfgx2_lbl[cfgx]
            percent = 100 * success.sum() / len(success)
            print(
                '%2d) success = %r/%r = %.2f%% -- %s'
                % (cfgx, success.sum(), len(success), percent, pipelbl)
            )

    def print_config_overlap(testres, with_plot=True):
        truth2_prop, prop2_mat = testres.get_truth2_prop()
        qx2_gt_ranks = truth2_prop['gt']['rank']
        qx2_success = qx2_gt_ranks == 0
        cfgx2_num_correct = np.nansum(qx2_success, axis=0)
        best_cfgx = cfgx2_num_correct.argmax()

        print('Config Overlap')

        # Matrix version
        # disjoint_mat = np.zeros((testres.nConfig, testres.nConfig), dtype=np.int32)
        # improves_mat = np.zeros((testres.nConfig, testres.nConfig), dtype=np.int32)
        isect_mat = np.zeros((testres.nConfig, testres.nConfig), dtype=np.int32)
        union_mat = np.zeros((testres.nConfig, testres.nConfig), dtype=np.int32)
        for cfgx1 in range(testres.nConfig):
            for cfgx2 in range(testres.nConfig):
                if cfgx1 == cfgx2:
                    success_qx1 = np.where(qx2_success.T[cfgx1])[0]
                    isect_mat[cfgx1][cfgx2] = len(success_qx1)
                    union_mat[cfgx1][cfgx2] = len(success_qx1)
                    continue
                success_qx1 = np.where(qx2_success.T[cfgx1])[0]
                success_qx2 = np.where(qx2_success.T[cfgx2])[0]
                union_ = np.union1d(success_qx1, success_qx2)
                isect_ = np.intersect1d(success_qx1, success_qx2)
                # disjoints = np.setdiff1d(union_, isect_)
                # disjoint_mat[cfgx1][cfgx2] = len(disjoints)
                isect_mat[cfgx1][cfgx2] = len(isect_)
                union_mat[cfgx1][cfgx2] = len(union_)
                # improves = np.setdiff1d(success_qx2, isect_)
                # improves_mat[cfgx2][cfgx1] = len(improves)

        n_success_list = np.array(
            [qx2_success.T[cfgx1].sum() for cfgx1 in range(testres.nConfig)]
        )
        improves_mat = n_success_list[:, None] - isect_mat

        disjoint_mat = union_mat - isect_mat
        print('n_success_list = %r' % (n_success_list,))
        print('union_mat =\n%s' % (union_mat,))
        print('isect_mat =\n%s' % (isect_mat,))
        print('cfgx1 and cfgx2 have <x> not in common')
        print('disjoint_mat =\n%s' % (disjoint_mat,))
        print('cfgx1 helps cfgx2 by <x>')
        print('improves_mat =\n%s' % (improves_mat,))
        print('improves_mat.sum(axis=1) = \n%s' % (improves_mat.sum(axis=1),))
        bestx_by_improves = improves_mat.sum(axis=1).argmax()
        print('bestx_by_improves = %r' % (bestx_by_improves,))

        # Numbered version
        print('best_cfgx = %r' % (best_cfgx,))
        for cfgx in range(testres.nConfig):
            if cfgx == best_cfgx:
                continue
            pipelbl = testres.cfgx2_lbl[cfgx]
            qx2_anysuccess = np.logical_or(qx2_success.T[cfgx], qx2_success.T[best_cfgx])
            # Queries that other got right that best did not get right
            qx2_othersuccess = np.logical_and(
                qx2_anysuccess, np.logical_not(qx2_success.T[best_cfgx])
            )
            print(
                'cfgx %d) has %d success cases that that the best config does not have -- %s'
                % (cfgx, qx2_othersuccess.sum(), pipelbl)
            )

        qx2_success.T[cfgx]

        if with_plot:
            # y = None
            # for x in qx2_gt_ranks:
            #    x = np.minimum(x, 3)
            #    z =  (x.T - x[:, None])
            #    if np.any(z):
            #        print(z)
            #    if y is None:
            #        y = z
            #    else:
            #        y += z

            if False:
                # Chip size stats
                ave_dlen = [  # NOQA
                    np.sqrt(
                        np.array(
                            testres.ibs.get_annot_chip_dlensqrd(
                                testres.qaids, config2_=qreq_.query_config2_
                            )
                        )
                    ).mean()
                    for qreq_ in testres.cfgx2_qreq_
                ]

                ave_width_inimg = [  # NOQA
                    np.array(
                        testres.ibs.get_annot_bboxes(
                            testres.qaids, config2_=qreq_.query_config2_
                        )
                    )[:, 2 + 0].mean()
                    for qreq_ in testres.cfgx2_qreq_
                ]

                ave_width = [  # NOQA
                    np.array(
                        testres.ibs.get_annot_chip_sizes(
                            testres.qaids, config2_=qreq_.query_config2_
                        )
                    )[:, 0].mean()
                    for qreq_ in testres.cfgx2_qreq_
                ]

            import wbia.plottool as pt

            # pt.plt.imshow(-y, interpolation='none', cmap='hot')
            # pt.plt.colorbar()

            def label_ticks():
                import wbia.plottool as pt

                ax = pt.gca()
                labels = testres.get_varied_labels()
                ax.set_xticks(list(range(len(labels))))
                ax.set_xticklabels([lbl[0:100] for lbl in labels])
                [lbl.set_rotation(-25) for lbl in ax.get_xticklabels()]
                [lbl.set_horizontalalignment('left') for lbl in ax.get_xticklabels()]

                # xgrid, ygrid = np.meshgrid(range(len(labels)), range(len(labels)))
                # pt.plot_surface3d(xgrid, ygrid, disjoint_mat)
                ax.set_yticks(list(range(len(labels))))
                ax.set_yticklabels([lbl[0:100] for lbl in labels])
                [lbl.set_horizontalalignment('right') for lbl in ax.get_yticklabels()]
                [lbl.set_verticalalignment('center') for lbl in ax.get_yticklabels()]
                # [lbl.set_rotation(20) for lbl in ax.get_yticklabels()]

            pt.figure(fnum=pt.next_fnum())
            pt.plt.imshow(union_mat, interpolation='none', cmap='hot')
            pt.plt.colorbar()
            pt.set_title(
                'union mat: cfg<x> and cfg<y> have <z> success cases in in total'
            )
            label_ticks()
            label_ticks()

            pt.figure(fnum=pt.next_fnum())
            pt.plt.imshow(isect_mat, interpolation='none', cmap='hot')
            pt.plt.colorbar()
            pt.set_title('isect mat: cfg<x> and cfg<y> have <z> success cases in common')
            label_ticks()

            pt.figure(fnum=pt.next_fnum())
            pt.plt.imshow(disjoint_mat, interpolation='none', cmap='hot')
            pt.plt.colorbar()
            pt.set_title(
                'disjoint mat (union - isect): cfg<x> and cfg<y> have <z> success cases not in common'
            )

            # xgrid, ygrid = np.meshgrid(range(len(labels)), range(len(labels)))
            # pt.plot_surface3d(xgrid, ygrid, improves_mat)

            pt.figure(fnum=pt.next_fnum())
            pt.plt.imshow(improves_mat, interpolation='none', cmap='hot')
            pt.plt.colorbar()
            pt.set_title(
                'improves mat (diag.T - isect): cfg<x> got <z> qaids that cfg <y> missed'
            )
            label_ticks()
            # pt.colorbar(np.unique(y))

    def map_score(testres):
        """
        For each query compute a precision recall curve.
        Then, for each query compute the average precision.
        Then take the mean of all average precisions to obtain the mAP.

        Script:
            >>> #ibs = wbia.opendb('Oxford')
            >>> #ibs, testres = wbia.testdata_expts('Oxford', a='oxford', p='smk:nWords=[64000],nAssign=[1],SV=[False,True]')
            >>> import wbia
            >>> ibs, testres = wbia.testdata_expts('Oxford', a='oxford', p='smk:nWords=[64000],nAssign=[1],SV=[False,True],can_match_sameimg=True')
            >>> import wbia
            >>> ibs, testres = wbia.testdata_expts('Oxford', a='oxford', p='smk:nWords=[64000],nAssign=[1],SV=[False],can_match_sameimg=True')
        """
        import sklearn.metrics

        qaids = testres.get_test_qaids()
        ibs = testres.ibs

        PLOT = True
        PLOT = False

        cfgx2_cms = []
        for qreq_ in testres.cfgx2_qreq_:
            cm_list = qreq_.execute(qaids)
            cm_list = [cm.extend_results(qreq_) for cm in cm_list]
            for cm in cm_list:
                cm.score_annot_csum(qreq_)
            #    #cm.sortself()
            cfgx2_cms.append(cm_list)

        map_list = []
        unique_names, groupxs = ut.group_indices(ibs.annots(qaids).names)

        for cm_list, qreq_ in zip(cfgx2_cms, testres.cfgx2_qreq_):
            if PLOT:
                import wbia.plottool as pt

                pt.qt4ensure()
                fnum = pt.ensure_fnum(None)
                pt.figure(fnum=fnum)
            avep_list = []
            # fnum = pt.ensure_fnum(None)
            # pt.figure(fnum=fnum)
            for cm in cm_list:
                # Ignore junk images
                flags = np.array(ibs.annots(cm.daid_list).quality_texts) != 'junk'
                assert np.all(flags)
                daid_list = cm.daid_list
                dnid_list = cm.dnid_list
                y_true = (cm.qnid == dnid_list).compress(flags).astype(np.int)
                y_score = cm.annot_score_list.compress(flags)

                y_score[~np.isfinite(y_score)] = 0
                y_score = np.nan_to_num(y_score)
                sortx = np.argsort(y_score)[::-1]
                daid_list = daid_list.take(sortx)
                dnid_list = dnid_list.take(sortx)
                y_true = y_true.take(sortx)
                y_score = y_score.take(sortx)
                # print(cm.get_annot_ranks(cm.get_top_gt_aids(ibs)))

                precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
                    y_true, y_score
                )

                if PLOT:
                    pt.plot2(
                        recall,
                        precision,
                        marker='',
                        linestyle='-',
                        x_label='recall',
                        y_label='precision',
                    )

                avep = sklearn.metrics.average_precision_score(y_true, y_score)
                # avep = [
                #    sklearn.metrics.average_precision_score(y_true, y_score, average=average)
                #    for average in ['micro', 'macro', 'samples', 'weighted']
                # ]
                # if np.any(np.isnan(avep)):
                #    break
                # if np.isnan(avep):
                #     break
                avep_list.append(avep)
            # mean_ave_precision = np.mean(avep_list, axis=0)
            name_to_ave = [np.mean(a) for a in ut.apply_grouping(avep_list, groupxs)]
            name_to_ave_ = dict(zip(unique_names, name_to_ave))
            print(
                'name_to_ave_ = %s' % (ut.align(ut.repr3(name_to_ave_, precision=3), ':'))
            )
            mean_ave_precision = np.mean(name_to_ave)
            print('mean_ave_precision = %r' % (mean_ave_precision,))
            map_list.append(mean_ave_precision)
        return map_list

    def embed_testres(testres):
        """
        CommandLine:
            python -m wbia TestResults.embed_testres

        Example:
            >>> # SCRIPT
            >>> from wbia.expt.test_result import *  # NOQA
            >>> from wbia.init import main_helpers
            >>> ibs, testres = main_helpers.testdata_expts(defaultdb='PZ_MTEST')
            >>> embed_testres(testres)
        """
        ut.embed()

    def get_options(testres):
        func_list = [
            testres.print_results,
            testres.draw_rank_cmc,
            testres.draw_match_cases,
            testres.embed_testres,
        ]
        return func_list

    def get_actions(testres):
        actions = ut.odict(
            [
                (testres.print_results, (['print', 'p'], '')),
                (testres.draw_rank_cmc, (['cmc'], '')),
                (testres.draw_match_cases, (['case'], '')),
                (testres.embed_testres, (['embed', 'ipy'], '')),
            ]
        )
        return actions

    def help(testres):
        # list functions that accept the standard interface
        prefix = 'wbia'
        suffix = testres.reconstruct_test_flags()
        func_list = testres.get_options()
        funcname_list = [ut.get_funcname(func) for func in func_list]
        cmdstr_list = [' '.join([prefix, funcname, suffix]) for funcname in funcname_list]
        ut.cprint('Available Functions:', 'blue')
        print(', '.join(funcname_list))
        ut.cprint('Available Commandline:', 'blue')
        print('\n'.join(cmdstr_list))


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.expt.test_result
        python -m wbia.expt.test_result --allexamples
        python -m wbia.expt.test_result --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
