# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
from six.moves import zip, range, map
import vtool as vt
import copy
import utool as ut
from ibeis.expt import cfghelpers
from ibeis.expt import experiment_helpers  # NOQA
import operator
from functools import partial
print, rrr, profile = ut.inject2(__name__, '[testres]')


def combine_testres_list(ibs, testres_list):
    """
    combine test results over multiple annot configs

    CommandLine:
        python -m ibeis --tf combine_testres_list

        python -m ibeis --tf -draw_rank_cdf --db PZ_MTEST --show
        python -m ibeis --tf -draw_rank_cdf --db PZ_Master1 --show
        python -m ibeis --tf -draw_rank_cdf --db PZ_MTEST --show -a varysize -t default
        python -m ibeis --tf -draw_rank_cdf --db PZ_MTEST --show -a varysize -t default

    >>> # DISABLE_DOCTEST
    >>> from ibeis.expt.test_result import *  # NOQA
    >>> from ibeis.expt import harness
    >>> ibs, testres_list = harness.testdata_expts('PZ_MTEST', ['varysize'])
    >>> combine_testres_list(ibs, testres_list)
    """
    import copy
    from ibeis.expt import annotation_configs

    acfg_list = [testres.acfg for testres in testres_list]
    acfg_lbl_list = annotation_configs.get_varied_acfg_labels(acfg_list)

    flat_acfg_list = annotation_configs.flatten_acfg_list(acfg_list)
    nonvaried_acfg, varied_acfg_list = cfghelpers.partition_varied_cfg_list(flat_acfg_list)

    def combine_lbls(lbl, acfg_lbl):
        if len(lbl) == 0:
            return acfg_lbl
        if len(acfg_lbl) == 0:
            return lbl
        return lbl + '+' + acfg_lbl

    agg_cfg_list        = ut.flatten(
        [testres.cfg_list for testres in testres_list])
    agg_cfgx2_cfgreinfo = ut.flatten(
        [testres.cfgx2_cfgresinfo for testres in testres_list])
    agg_cfgx2_qreq_     = ut.flatten(
        [testres.cfgx2_qreq_ for testres in testres_list])
    agg_cfgdict_list    = ut.flatten(
        [testres.cfgdict_list for testres in testres_list])
    agg_varied_acfg_list = ut.flatten([
        [acfg] * len(testres.cfg_list)
        for testres, acfg in zip(testres_list, varied_acfg_list)
    ])
    agg_cfgx2_lbls      = ut.flatten(
        [[combine_lbls(lbl, acfg_lbl) for lbl in testres.cfgx2_lbl]
         for testres, acfg_lbl in zip(testres_list, acfg_lbl_list)])

    agg_cfgx2_acfg = ut.flatten(
        [[copy.deepcopy(acfg)] * len(testres.cfg_list) for
         testres, acfg in zip(testres_list, acfg_list)])

    big_testres = TestResult(agg_cfg_list, agg_cfgx2_lbls,
                             agg_cfgx2_cfgreinfo, agg_cfgx2_qreq_)

    # Give the big test result an acfg that is common between everything
    big_testres.acfg = annotation_configs.unflatten_acfgdict(nonvaried_acfg)
    big_testres.cfgdict_list = agg_cfgdict_list  # TODO: depricate

    big_testres.common_acfg = annotation_configs.compress_aidcfg(big_testres.acfg)
    big_testres.common_cfgdict = reduce(ut.dict_intersection, big_testres.cfgdict_list)
    big_testres.varied_acfg_list = agg_varied_acfg_list
    big_testres.nonvaried_acfg = nonvaried_acfg
    big_testres.varied_cfg_list = [
        ut.delete_dict_keys(cfgdict.copy(), list(big_testres.common_cfgdict.keys()))
        for cfgdict in big_testres.cfgdict_list]
    big_testres.acfg_list = acfg_list
    big_testres.cfgx2_acfg = agg_cfgx2_acfg
    big_testres.cfgx2_pcfg = agg_cfgdict_list

    assert len(agg_cfgdict_list) == len(agg_cfgx2_acfg)

    #big_testres.acfg
    testres = big_testres
    # big_testres = testres
    return testres


@six.add_metaclass(ut.ReloadingMetaclass)
class TestResult(object):
    def __init__(testres, cfg_list, cfgx2_lbl, cfgx2_cfgresinfo, cfgx2_qreq_):
        assert len(cfg_list) == len(cfgx2_lbl), (
            'bad lengths1: %r != %r' % (len(cfg_list), len(cfgx2_lbl)))
        assert len(cfgx2_qreq_) == len(cfgx2_lbl), (
            'bad lengths2: %r != %r' % (len(cfgx2_qreq_), len(cfgx2_lbl)))
        assert len(cfgx2_cfgresinfo) == len(cfgx2_lbl), (
            'bad lengths3: %r != %r' % (len(cfgx2_cfgresinfo), len(cfgx2_lbl)))
        testres.cfg_list         = cfg_list
        testres.cfgx2_lbl        = cfgx2_lbl
        testres.cfgx2_cfgresinfo = cfgx2_cfgresinfo
        testres.cfgx2_qreq_      = cfgx2_qreq_
        testres.lbl              = None
        testres.testnameid       = None

    def __str__(testres):
        return testres.reconstruct_test_flags()

    def __repr__(testres):
        return testres._custom_str()

    def _custom_str(testres):
        typestr = testres.__class__.__name__
        dbname = None if testres.ibs is None else testres.ibs.get_dbname()
        # hashkw = dict(_new=True, pathsafe=False)
        infostr_ = 'nCfg=%s'  % testres.nConfig
        if testres.nConfig ==  1:
            qreq_ = testres.cfgx2_qreq_[0]
            infostr_ += ' nQ=%s, nD=%s %s' % (len(qreq_.qaids), len(qreq_.daids), qreq_.get_pipe_hashid())
        # nD=%s %s' % (, len(testres.daids), testres.get_pipe_hashid())
        custom_str = '<%s(%s) %s at %s>' % (typestr, dbname, infostr_, hex(id(testres)))
        return custom_str

    @property
    def ibs(testres):
        def tryget_ibs(qreq_):
            try:
                return qreq_.ibs
            except AttributeError:
                return qreq_.depc.controller

        ibs_list = [tryget_ibs(qreq_) for qreq_ in testres.cfgx2_qreq_]
        ibs = ibs_list[0]
        for ibs_ in ibs_list:
            assert ibs is ibs_, 'not all query requests are using the same controller'
        return ibs

    @property
    def qaids(testres):
        assert testres.has_constant_qaids(), 'must have constant qaids to use this property'
        return testres.cfgx2_qaids[0]
        #return testres._qaids

    @property
    def nConfig(testres):
        return len(testres.cfg_list)

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
        return ut.list_allsame(testres.cfgx2_daids)

    def has_constant_qaids(testres):
        return ut.list_allsame(testres.cfgx2_qaids)

    def has_constant_length_daids(testres):
        return ut.list_allsame(list(map(len, testres.cfgx2_daids)))

    def has_constant_length_qaids(testres):
        return ut.list_allsame(list(map(len, testres.cfgx2_qaids)))

    def get_infoprop_list(testres, key, qaids=None):
        """
        key = 'qx2_bestranks'
        key = 'qx2_gt_rank'
        qaids = testres.get_test_qaids()
        """
        if key == 'participant':
            # Get if qaids are part of the config
            cfgx2_infoprop = [np.in1d(qaids, aids_) for aids_ in testres.cfgx2_qaids]
        else:
            _tmp1_cfgx2_infoprop = ut.get_list_column(testres.cfgx2_cfgresinfo, key)
            _tmp2_cfgx2_infoprop = list(map(
                np.array,
                ut.util_list.replace_nones(_tmp1_cfgx2_infoprop, np.nan)))
            if qaids is None:
                cfgx2_infoprop = _tmp2_cfgx2_infoprop
            else:
                old = False
                if old:
                    # Old way forcing common qaids
                    flags_list = [np.in1d(aids_, qaids) for aids_ in testres.cfgx2_qaids]
                    cfgx2_infoprop = vt.zipcompress(_tmp2_cfgx2_infoprop, flags_list)
                else:
                    # New way with nans
                    cfgx2_qaid2_qx = [dict(zip(aids_, range(len(aids_)))) for aids_ in testres.cfgx2_qaids]
                    qxs_list = [ut.dict_take(qaid2_qx , qaids, None) for qaid2_qx  in cfgx2_qaid2_qx]
                    cfgx2_infoprop = [[np.nan if x is None else props[x] for x in qxs]
                                      for props, qxs in zip(_tmp2_cfgx2_infoprop, qxs_list)]
            if key == 'qx2_bestranks' or key.endswith('_rank'):
                # hack
                wpr = testres.get_worst_possible_rank()
                cfgx2_infoprop = [np.array([wpr if rank == -1 else rank
                                            for rank in infoprop])
                                  for infoprop in cfgx2_infoprop]
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
        rank_mat = testres.get_infoprop_mat(key='qx2_bestranks', qaids=qaids)
        return rank_mat

    def get_worst_possible_rank(testres):
        #worst_possible_rank = max(9001, len(testres.daids) + 1)
        worst_possible_rank = max([len(qreq_.daids) for qreq_ in testres.cfgx2_qreq_]) + 1
        #worst_possible_rank = len(testres.daids) + 1
        return worst_possible_rank

    def get_rank_histograms(testres, bins=None, asdict=True, jagged=False, key=None):
        """
        Ignore:
            testres.get_infoprop_mat('qnx2_gt_name_rank')
            testres.get_infoprop_mat('qnx2_gf_name_rank')
            testres.get_infoprop_mat('qnx2_qnid')

        """
        if key is None:
            key = 'qx2_bestranks'
            #key = 'qnx2_gt_name_rank'
        if bins is None:
            bins = testres.get_rank_histogram_bins()
        elif bins == 'dense':
            bins = np.arange(testres.get_worst_possible_rank() + 1)
        if jagged:
            assert not asdict
            cfgx2_bestranks = testres.get_infoprop_list(key)
            cfgx2_bestranks = [
                ut.list_replace(bestranks, -1, testres.get_worst_possible_rank())
                for bestranks in cfgx2_bestranks]
            cfgx2_hist = np.zeros((len(cfgx2_bestranks), len(bins) - 1), dtype=np.int32)
            for cfgx, ranks in enumerate(cfgx2_bestranks):
                bin_values, bin_edges  = np.histogram(ranks, bins=bins)
                assert len(ranks) == bin_values.sum(), 'should sum to be equal'
                cfgx2_hist[cfgx] = bin_values
            return cfgx2_hist, bin_edges

        #rank_mat = testres.get_rank_mat()
        rank_mat = testres.get_infoprop_mat(key=key)

        if not asdict:
            # Use numpy histogram repr
            config_hists = np.zeros((len(rank_mat.T), len(bins) - 1), dtype=np.int32)
        else:
            config_hists = []
            pass
        bin_sum = None
        for cfgx, ranks in enumerate(rank_mat.T):
            bin_values, bin_edges  = np.histogram(ranks, bins=bins)
            if bin_sum is None:
                bin_sum = bin_values.sum()
            else:
                assert bin_sum == bin_values.sum(), 'should sum to be equal'
            if asdict:
                # Use dictionary histogram repr
                bin_keys = list(zip(bin_edges[:-1], bin_edges[1:]))
                hist_dict = dict(zip(bin_keys, bin_values))
                config_hists.append(hist_dict)
            else:
                config_hists[cfgx] = bin_values
        if not asdict:
            return config_hists, bin_edges
        else:
            return config_hists

    def get_rank_percentage_cumhist(testres, bins='dense', key=None, join_acfgs=False):
        r"""
        Args:
            bins (unicode): (default = u'dense')

        Returns:
            tuple: (config_cdfs, edges)

        CommandLine:
            python -m ibeis --tf TestResult.get_rank_percentage_cumhist
            python -m ibeis --tf TestResult.get_rank_percentage_cumhist -t baseline -a uncontrolled ctrl

            python -m ibeis --tf TestResult.get_rank_percentage_cumhist \
                --db lynx \
                -a default:qsame_imageset=True,been_adjusted=True,excluderef=True \
                -t default:K=1 --show --cmd

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> from ibeis.init import main_helpers
            >>> #ibs, testres = main_helpers.testdata_expts('testdb1')
            >>> ibs, testres = main_helpers.testdata_expts('testdb1', a=['default:num_names=1,name_offset=[0,1]'])
            >>> bins = u'dense'
            >>> key = None
            >>> (config_cdfs, edges) = testres.get_rank_percentage_cumhist(bins)
            >>> result = ('(config_cdfs, edges) = %s' % (str((config_cdfs, edges)),))
            >>> print(result)
        """
        #testres.rrr()
        cfgx2_hist, edges = testres.get_rank_histograms(bins, asdict=False, jagged=True, key=key)
        cfgx2_cumhist = np.cumsum(cfgx2_hist, axis=1)
        if join_acfgs:
            # Hack for turtles
            groupxs = testres.get_pcfg_groupxs()
            cfgx2_cumhist = np.array([np.sum(group, axis=0) for group in ut.apply_grouping(cfgx2_cumhist, groupxs)])
        cfgx2_cumhist_percent = 100 * cfgx2_cumhist / cfgx2_cumhist.T[-1].T[:, None]
        return cfgx2_cumhist_percent, edges

    def get_pcfg_groupxs(testres):
        """ Groups by common pipe configs """
        phashid_list = [qreq_.get_pipe_hashid() for qreq_ in testres.cfgx2_qreq_]
        groupxs = ut.group_indices(phashid_list)[1]
        return groupxs

    def get_rank_cumhist(testres, bins='dense'):
        #testres.rrr()
        hist_list, edges = testres.get_rank_histograms(bins, asdict=False)
        #hist_list, edges = testres.get_rank_histograms(bins, asdict=False, jagged=True)
        config_cdfs = np.cumsum(hist_list, axis=1)
        return config_cdfs, edges

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

    def get_rank_histogram_bin_edges(testres):
        bins = testres.get_rank_histogram_bins()
        bin_keys = list(zip(bins[:-1], bins[1:]))
        return bin_keys

    def get_rank_histogram_qx_binxs(testres):
        rank_mat = testres.get_rank_mat()
        config_hists = testres.get_rank_histograms()
        config_binxs = []
        bin_keys = testres.get_rank_histogram_bin_edges()
        for hist_dict, ranks in zip(config_hists, rank_mat.T):
            bin_qxs = [np.where(np.logical_and(low <= ranks, ranks < high))[0]
                       for low, high in bin_keys]
            qx2_binx = -np.ones(len(ranks))
            for binx, qxs in enumerate(bin_qxs):
                qx2_binx[qxs] = binx
            config_binxs.append(qx2_binx)
        return config_binxs

    def get_rank_histogram_qx_sample(testres, size=10):
        size = 10
        rank_mat = testres.get_rank_mat()
        config_hists = testres.get_rank_histograms()
        config_rand_bin_qxs = []
        bins = testres.get_rank_histogram_bins()
        bin_keys = list(zip(bins[:-1], bins[1:]))
        randstate = np.random.RandomState(seed=0)
        for hist_dict, ranks in zip(config_hists, rank_mat.T):
            bin_qxs = [np.where(np.logical_and(low <= ranks, ranks < high))[0]
                       for low, high in bin_keys]
            rand_bin_qxs = [qxs if len(qxs) <= size else
                            randstate.choice(qxs, size=size, replace=False)
                            for qxs in bin_qxs]
            config_rand_bin_qxs.append(rand_bin_qxs)
        return config_rand_bin_qxs

    def get_X_LIST(testres):
        #X_LIST = ut.get_argval('--rank-lt-list', type_=list, default=[1])
        X_LIST = ut.get_argval('--rank-lt-list', type_=list, default=[1, 5])
        return X_LIST

    def get_nLessX_dict(testres):
        # Build a (histogram) dictionary mapping X (as in #ranks < X) to a list of cfg scores
        X_LIST = testres.get_X_LIST()
        nLessX_dict = {int(X): np.zeros(testres.nConfig) for X in X_LIST}
        cfgx2_qx2_bestrank = testres.get_infoprop_list('qx2_bestranks')
        #rank_mat = testres.rank_mat  # HACK
        for X in X_LIST:
            cfgx2_lessX_mask = [
                np.logical_and(0 <= qx2_ranks, qx2_ranks < X)
                for qx2_ranks in cfgx2_qx2_bestrank]
            #lessX_ = np.logical_and(np.less(rank_mat, X), np.greater_equal(rank_mat, 0))
            cfgx2_nLessX = np.array([lessX_.sum(axis=0) for lessX_ in cfgx2_lessX_mask])
            nLessX_dict[int(X)] = cfgx2_nLessX
        return nLessX_dict

    def get_all_varied_params(testres):
        r"""
        Returns the parameters that were varied between different configurations in this test

        Returns:
            list: varied_params

        CommandLine:
            python -m ibeis --tf -get_all_varied_params

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> import ibeis
            >>> testres = ibeis.testdata_expts('PZ_MTEST', t='default:K=[1,2]')[1]
            >>> varied_params = testres.get_all_varied_params()
            >>> result = ('varied_params = %s' % (ut.repr2(varied_params),))
            >>> print(result)
            varied_params = ['K', '_cfgindex']
        """
        # only for big results
        varied_cfg_params = list(set(ut.flatten(
            [cfgdict.keys()
             for cfgdict in testres.varied_cfg_list])))
        varied_acfg_params = list(set(ut.flatten([
            acfg.keys()
            for acfg in testres.varied_acfg_list])))
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
            basis = sorted(list(set([
                cfgdict[key]
                for cfgdict in testres.varied_cfg_list])))
        elif any([key in cfgdict for cfgdict in testres.varied_acfg_list]):
            basis = sorted(list(set([
                acfg[key]
                for acfg in testres.varied_acfg_list])))
        else:
            #assert False, 'param is not varied'
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
            cfgx_list = [cfgx for cfgx, daids in enumerate(testres.cfgx2_daids)
                         if len(daids) == val]
        elif any([key in cfgdict for cfgdict in testres.varied_cfg_list]):
            cfgx_list = [cfgx for cfgx, cfgdict in enumerate(testres.varied_cfg_list)
                         if cfgdict[key] == val]
        elif any([key in cfgdict for cfgdict in testres.varied_acfg_list]):
            cfgx_list = [cfgx for cfgx, acfg in enumerate(testres.varied_acfg_list)
                         if acfg[key] == val]
        else:
            if key in testres.common_cfgdict:
                cfgx_list = list(range(testres.nConfig))
            elif key in testres.nonvaried_acfg:
                cfgx_list = list(range(testres.nConfig))
            else:
                assert False, 'param=%r doesnt exist' % (key,)
            #assert False, 'param is not varied'
        return cfgx_list

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
            #('custom', 'default'),
            #('fg_on', 'FG'),
            #('fg_on=True', 'FG'),
            #('fg_on=False,?', ''),
            ('fg_on=True', 'FG=True'),
            ('fg_on=False,?', 'FG=False'),

            ('lnbnn_on=True', 'LNBNN'),
            ('lnbnn_on=False,?', ''),

            ('normonly_on=True', 'normonly'),
            ('normonly_on=False,?', ''),

            ('bar_l2_on=True', 'dist'),
            ('bar_l2_on=False,?', ''),

            ('sv_on', 'SV'),
            ('rotation_invariance', 'RI'),
            ('affine_invariance', 'AI'),
            ('augment_queryside_hack', 'QRH'),
            ('nNameShortlistSVER', 'nRR'),
            #
            #('sample_per_ref_name', 'per_ref_name'),
            ('sample_per_ref_name', 'per_gt_name'),
            ('require_timestamp=True', 'require_timestamp'),
            ('require_timestamp=False,?', ''),
            ('require_timestamp=None,?', ''),
            ('[_A-Za-z]*=None,?', ''),
            ('dpername=None,?', ''),
            #???
            #('sample_per_ref_name', 'per_gt_name'),
            #('per_name', 'per_gf_name'),   # Try to make labels clearer for paper
            #----
            ('prescore_method=\'?csum\'?,score_method=\'?csum\'?,?', 'amech'),
            ('prescore_method=\'?nsum\'?,score_method=\'?nsum\'?,?', 'fmech'),
            ('force_const_size=[^,]+,?', ''),
            (r'[dq]_true_size=\d+,?', ''),
            (r'_orig_size=[^,]+,?', ''),
            # Hack
            ('[qd]?exclude_reference=' + ut.regex_or(['True', 'False', 'None']) + '\,?', ''),
            #('=True', '=On'),
            #('=False', '=Off'),
            ('=True', '=T'),
            ('=False', '=F'),
        ]
        for ser, rep in repl_list:
            lbl = re.sub(ser, rep, lbl)
        return lbl

    #def _friendly_shorten_lbls(testres, lbl):
    #    import re
    #    repl_list = [
    #        ('dmingt=None,?', ''),
    #        ('qpername=None,?', ''),
    #    ]
    #    for ser, rep in repl_list:
    #        lbl = re.sub(ser, rep, lbl)
    #    return lbl

    def get_short_cfglbls(testres, join_acfgs=False):
        """
        Labels for published tables

        cfg_lbls = ['baseline:nRR=200+default:', 'baseline:+default:']

        CommandLine:
            python -m ibeis --tf TestResult.get_short_cfglbls

        Example:
            >>> # SLOW_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> import ibeis
            >>> ibs, testres = ibeis.testdata_expts('PZ_MTEST', a=['unctrl', 'ctrl::unctrl_comp'])
            >>> cfg_lbls = testres.get_short_cfglbls()
            >>> result = ('cfg_lbls = %s' % (ut.list_str(cfg_lbls),))
            >>> print(result)
        """
        from ibeis.expt import cfghelpers
        from ibeis.expt import annotation_configs
        if False:
            acfg_names = [acfg['qcfg']['_cfgstr'] for acfg in testres.cfgx2_acfg]
            pcfg_names = [pcfg['_cfgstr'] for pcfg in testres.cfgx2_pcfg]
            # Only vary the label settings within the cfgname
            acfg_hashes = np.array(list(map(hash, acfg_names)))
            unique_hashes, a_groupxs = vt.group_indices(acfg_hashes)
            a_label_groups = []
            for groupx in a_groupxs:
                acfg_list = ut.take(testres.cfgx2_acfg, groupx)
                #varied_lbls = cfghelpers.get_varied_cfg_lbls(acfg_list)
                varied_lbls = annotation_configs.get_varied_acfg_labels(
                    acfg_list, mainkey='_cfgstr')
                a_label_groups.append(varied_lbls)
            acfg_lbls = vt.invert_apply_grouping(a_label_groups, a_groupxs)

            pcfg_hashes = np.array(list(map(hash, pcfg_names)))
            unique_hashes, p_groupxs = vt.group_indices(pcfg_hashes)
            p_label_groups = []
            for groupx in p_groupxs:
                pcfg_list = ut.take(testres.cfgx2_pcfg, groupx)
                varied_lbls = cfghelpers.get_varied_cfg_lbls(pcfg_list, mainkey='_cfgstr')
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
                    _tup = part.split(cfghelpers.NAMEVARSEP)
                    name, settings = _tup if len(_tup) > 1 else (_tup[0], '')
                    new_parts.append(part if settings else name)
                if len(new_parts) == 2 and new_parts[1] == 'default':
                    newlbl = new_parts[0]
                else:
                    newlbl = '+'.join(new_parts)
                cfg_lbls.append(newlbl)
        if join_acfgs:
            groupxs = testres.get_pcfg_groupxs()
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
                    nonvaried_cfg = cfghelpers.partition_varied_cfg_list(cfgs)[0]
                    group_lbl_parts.append(cfghelpers.get_cfg_lbl(nonvaried_cfg))
                    # print('nonvaried_lbl = %r' % (nonvaried_lbl,))
                group_lbl = '+'.join(group_lbl_parts)
                group_lbls.append(group_lbl)
            cfg_lbls = group_lbls
        return cfg_lbls

    def make_figtitle(testres, plotname='', filt_cfg=None):
        """
        Helper for consistent figure titles

        CommandLine:
            python -m ibeis --tf TestResult.make_figtitle  --prefix "Seperability " --db GIRM_Master1   -a timectrl -t Ell:K=2     --hargv=scores
            python -m ibeis --tf TestResult.make_figtitle

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> import ibeis
            >>> ibs, testres = ibeis.testdata_expts('PZ_MTEST')
            >>> plotname = ''
            >>> figtitle = testres.make_figtitle(plotname)
            >>> result = ('figtitle = %r' % (figtitle,))
            >>> print(result)
        """
        figtitle_prefix = ut.get_argval('--prefix', type_=str, default='')
        if figtitle_prefix != '':
            figtitle_prefix = figtitle_prefix.rstrip() + ' '
        figtitle = (figtitle_prefix + plotname)
        hasprefix = figtitle_prefix == ''
        if hasprefix:
            figtitle += '\n'

        title_aug = testres.get_title_aug(friendly=True, with_cfg=hasprefix)
        figtitle += ' ' + title_aug

        if filt_cfg is not None:
            filt_cfgstr = cfghelpers.get_cfg_lbl(filt_cfg)
            if filt_cfgstr.strip() != ':':
                figtitle += ' ' + filt_cfgstr
        return figtitle

    def get_title_aug(testres, with_size=True, with_db=True, with_cfg=True,
                      friendly=False):
        r"""
        Args:
            with_size (bool): (default = True)

        Returns:
            str: title_aug

        CommandLine:
            python -m ibeis --tf TestResult.get_title_aug --db PZ_Master1 -a timequalctrl::timectrl

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> import ibeis
            >>> ibs, testres = ibeis.testdata_expts('PZ_MTEST')
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
                    cfgname_list = [cfg['dcfg__cfgname']
                                    for cfg in testres.varied_acfg_list]
                    cfgname_list = ut.unique_ordered(cfgname_list)
                    annot_cfgname = '[' + ','.join(cfgname_list) + ']'
                try:
                    pipeline_cfgname = testres.common_cfgdict['_cfgstr']
                except KeyError:
                    #pipeline_cfgname = testres.common_cfgdict['_cfgname']
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
                print(ut.dict_str(testres.common_acfg))
                print(ut.dict_str(testres.common_cfgdict))
                ut.printex(ex)
                raise
        if with_size:
            if testres.has_constant_qaids():
                title_aug += ' #qaids=%r' % (len(testres.qaids),)
            elif testres.has_constant_length_qaids():
                title_aug += ' #qaids=%r*' % (len(testres.cfgx2_qaids[0]),)
            if testres.has_constant_daids():
                daids = testres.cfgx2_daids[0]
                title_aug += ' #daids=%r' % (len(testres.cfgx2_daids[0]),)
                if testres.has_constant_qaids():
                    locals_ = ibs.get_annotconfig_stats(
                        testres.qaids, daids, verbose=False)[1]
                    all_daid_per_name_stats = locals_['all_daid_per_name_stats']
                    if all_daid_per_name_stats['std'] == 0:
                        title_aug += ' dper_name=%s' % (
                            ut.scalar_str(all_daid_per_name_stats['mean'],
                                          max_precision=2),)
                    else:
                        title_aug += ' dper_name=%s±%s' % (
                            ut.scalar_str(all_daid_per_name_stats['mean'], precision=2),
                            ut.scalar_str(all_daid_per_name_stats['std'], precision=2),)
            elif testres.has_constant_length_daids():
                daids = testres.cfgx2_daids[0]
                title_aug += ' #daids=%r*' % (len(testres.cfgx2_daids[0]),)

        if friendly:
            # Hackiness for friendliness
            #title_aug = title_aug.replace('db=PZ_Master1', 'Plains Zebras')
            #title_aug = title_aug.replace('db=NNP_MasterGIRM_core', 'Masai Giraffes')
            #title_aug = title_aug.replace('db=GZ_ALL', 'Grevy\'s Zebras')
            title_aug = ut.multi_replace(
                title_aug,
                list(ibs.const.DBNAME_ALIAS.keys()),
                list(ibs.const.DBNAME_ALIAS.values()))
            #title_aug = title_aug.replace('db=PZ_Master1', 'db=PZ')
            #title_aug = title_aug.replace('db=NNP_MasterGIRM_core', 'Masai Giraffes')
            #title_aug = title_aug.replace('db=GZ_ALL', 'Grevy\'s Zebras')
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

            >>> from ibeis.expt.test_result import *  # NOQA
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
        experiment_helpers.print_pipe_configs(testres.cfgx2_pcfg,  testres.cfgx2_qreq_)

    def print_acfg_info(testres, **kwargs):
        """
        Prints verbose information about the annotations used in each test
        configuration

        CommandLine:
            python -m ibeis --tf TestResult.print_acfg_info

        Kwargs;
            see ibs.get_annot_stats_dict
            hashid, per_name, per_qual, per_vp, per_name_vpedge, per_image,
            min_name_hourdist

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> import ibeis
            >>> ibs, testres = ibeis.testdata_expts('PZ_MTEST', a=['ctrl::unctrl_comp'], t=['candk:K=[1,2]'])
            >>> ibs = None
            >>> result = testres.print_acfg_info()
            >>> print(result)
        """
        from ibeis.expt import annotation_configs
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
            ibs (IBEISController):  ibeis controller object(default = None)

        CommandLine:
            python -m ibeis --tf TestResult.print_unique_annot_config_stats

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> import ibeis
            >>> testres = ibeis.testdata_expts('PZ_MTEST', a=['ctrl::unctrl_comp'])
            >>> ibs = None
            >>> result = testres.print_unique_annot_config_stats(ibs)
            >>> print(result)
        """
        if ibs is None:
            ibs = testres.ibs
        cfx2_dannot_hashid = [ibs.get_annot_hashid_visual_uuid(daids)
                              for daids in testres.cfgx2_daids]
        unique_daids = ut.compress(testres.cfgx2_daids,
                                        ut.flag_unique_items(cfx2_dannot_hashid))
        with ut.Indenter('[acfgstats]'):
            print('+====')
            print('Printing %d unique annotconfig stats' % (len(unique_daids)))
            common_acfg = testres.common_acfg
            common_acfg['common'] = ut.dict_filter_nones(common_acfg['common'])
            print('testres.common_acfg = ' + ut.dict_str(common_acfg))
            print('param_basis(len(daids)) = %r' % (
                testres.get_param_basis('len(daids)'),))
            for count, daids in enumerate(unique_daids):
                print('+---')
                print('acfgx = %r/%r' % (count, len(unique_daids)))
                if testres.has_constant_qaids():
                    ibs.print_annotconfig_stats(testres.qaids, daids)
                else:
                    ibs.print_annot_stats(daids, prefix='d')
                print('L___')

    def print_results(testres):
        r"""
        CommandLine:
            python -m ibeis --tf TestResult.print_results

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> from ibeis.expt import harness
            >>> ibs, testres = harness.testdata_expts('PZ_MTEST')
            >>> result = testres.print_results()
            >>> print(result)
        """
        from ibeis.expt import experiment_printres
        ibs = testres.ibs
        experiment_printres.print_results(ibs, testres)

    @ut.memoize
    def get_new_hard_qx_list(testres):
        """ Mark any query as hard if it didnt get everything correct """
        rank_mat = testres.get_rank_mat()
        is_new_hard_list = rank_mat.max(axis=1) > 0
        new_hard_qx_list = np.where(is_new_hard_list)[0]
        return new_hard_qx_list

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
            python -m ibeis --tf TestResult.get_all_tags --db PZ_Master1 --show --filt :
            python -m ibeis --tf TestResult.get_all_tags --db PZ_Master1 --show --filt :min_gf_timedelta=24h
            python -m ibeis --tf TestResult.get_all_tags --db PZ_Master1 --show --filt :min_gf_timedelta=24h,max_gt_rank=5

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> from ibeis.init import main_helpers
            >>> ibs, testres = main_helpers.testdata_expts('PZ_Master1', a=['timectrl'])
            >>> filt_cfg = main_helpers.testdata_filtcfg()
            >>> case_pos_list = testres.case_sample2(filt_cfg)
            >>> all_tags = testres.get_all_tags()
            >>> selected_tags = ut.take(all_tags, case_pos_list.T[0])
            >>> flat_tags = list(map(str, ut.flatten(ut.flatten(selected_tags))))
            >>> print(ut.dict_str(ut.dict_hist(flat_tags), key_order_metric='val'))
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> pt.word_histogram2(flat_tags, fnum=1, pnum=(1, 2, 1))
            >>> pt.wordcloud(' '.join(flat_tags), fnum=1, pnum=(1, 2, 2))
            >>> pt.set_figtitle(cfghelpers.get_cfg_lbl(filt_cfg))
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
            python -m ibeis --tf TestResult.get_gf_tags --db PZ_Master1 --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> from ibeis.init import main_helpers
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
        both_tags = [[ut.flatten(t) for t in zip(*item)]
                     for item in zip(query_annot_tags, gt_annot_tags)]
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
            python -m ibeis --tf TestResult.case_sample2
            python -m ibeis --tf TestResult.case_sample2:0
            python -m ibeis --tf TestResult.case_sample2:1 --db PZ_Master1 --filt :min_tags=1
            python -m ibeis --tf TestResult.case_sample2:1 --db PZ_Master1 --filt :min_gf_tags=1

            python -m ibeis --tf TestResult.case_sample2:2 --db PZ_Master1

        Example0:
            >>> # ENABLE_DOCTEST
            >>> # The same results is achievable with different filter config settings
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> from ibeis.init import main_helpers
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
            >>> assert not np.all(mask5.T[0] == mask5.T[1])
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
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> from ibeis.init import main_helpers
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
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> from ibeis.init import main_helpers
            >>> ibs, testres = main_helpers.testdata_expts('PZ_MTEST', a=['ctrl'])
            >>> filt_cfg = {'fail': True, 'min_gtrank': 1, 'max_gtrank': None, 'min_gf_timedelta': '24h'}
            >>> #filt_cfg = cfghelpers.parse_argv_cfg('--filt')[0]
            >>> case_pos_list = testres.case_sample2(filt_cfg)
            >>> result = ('case_pos_list = %s' % (str(case_pos_list),))
            >>> print(result)
            >>> # Extra stuff
            >>> all_tags = testres.get_all_tags()
            >>> selcted_tags = ut.take(all_tags, case_pos_list.T[0])
            >>> print('selcted_tags = %r' % (selcted_tags,))
        """
        from ibeis.expt import cfghelpers
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

        qaids = testres.get_test_qaids() if qaids is None else qaids
        truth2_prop, prop2_mat = testres.get_truth2_prop(qaids)
        ibs = testres.ibs

        # Initialize isvalid flags to all true
        # np.ones(prop2_mat['is_success'].shape, dtype=np.bool)
        participates = prop2_mat['participates']
        is_valid = participates.copy()

        def unflat_tag_filterflags(tags_list, **kwargs):
            from ibeis import tag_funcs
            flat_tags, cumsum = ut.invertible_flatten2(tags_list)
            flat_flags = tag_funcs.filterflags_general_tags(flat_tags, **kwargs)
            flags = np.array(ut.unflatten2(flat_flags, cumsum))
            return flags

        UTFF = unflat_tag_filterflags

        def cols_disagree(mat, val):
            """
            is_success = prop2_mat['is_success']
            """
            sums = mat.sum(axis=1)
            nCols = mat.shape[1]
            flags1d = np.logical_and(sums  > 0, sums < nCols)
            flags = np.tile(flags1d[:, None], (1, 2))
            if not val:
                flags = np.logical_not(flags)
            return flags

        rule_list = [
            ('disagree',  lambda val: cols_disagree(prop2_mat['is_failure'], val)),
            ('fail',     prop2_mat['is_failure']),
            ('success',  prop2_mat['is_success']),
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

            ('min_query_annot_tags', lambda val: UTFF(testres.get_query_annot_tags(), min_num=val)),
            ('min_gt_annot_tags', lambda val: UTFF(testres.get_gt_annot_tags(), min_num=val)),
            ('min_gtq_tags', lambda val: UTFF(testres.get_gtquery_annot_tags(), min_num=val)),
            ('max_gtq_tags', lambda val: UTFF(testres.get_gtquery_annot_tags(), max_num=val)),

            ('without_gf_tag', lambda val: UTFF(testres.get_gf_tags(), has_none=val)),
            ('without_gt_tag', lambda val: UTFF(testres.get_gt_tags(), has_none=val)),
            ('with_gf_tag', lambda val: UTFF(testres.get_gf_tags(), has_any=val)),
            ('with_gt_tag', lambda val: UTFF(testres.get_gt_tags(), has_any=val)),
            ('with_tag',    lambda val: UTFF(testres.get_all_tags(), has_any=val)),
            ('without_tag', lambda val: UTFF(testres.get_all_tags(), has_none=val)),
        ]

        filt_cfg = filt_cfg.copy()

        #timedelta_keys = [
        #    'min_gf_timedelta',
        #    'max_gf_timedelta',
        #]
        #for tdkey in timedelta_keys:

        # hack to convert to seconds
        for tdkey in filt_cfg.keys():
            if tdkey.endswith('_timedelta'):
                filt_cfg[tdkey] = ut.ensure_timedelta(filt_cfg[tdkey])

        if verbose:
            print('[testres] Sampling from is_valid.size=%r with filt=%r' %
                  (is_valid.size, cfghelpers.get_cfg_lbl(filt_cfg)))
            print('  * is_valid.shape = %r' % (is_valid.shape,))
            num_valid = is_valid.sum()
            print('  * num_valid = %r' % (num_valid,))

        filt_cfg = copy.deepcopy(filt_cfg)

        # Remove test cases that do not meet the criteria
        for key, rule in rule_list:
            val = filt_cfg.pop(key, None)
            if val is not None:
                if isinstance(rule, np.ndarray):
                    # When a rule is an ndarray it must have boolean values
                    flags = rule == val
                else:
                    flags = rule(val)
                # HACK: flags are force to be false for non-participating cases
                flags = np.logical_and(flags, participates)
                if verbose:
                    prev_num_valid = is_valid.sum()
                # TODO: generalize to satisfiablility formula
                is_valid = np.logical_and(is_valid, flags)
                if verbose:
                    print('  * is_valid.shape = %r' % (is_valid.shape,))
                    print('SampleRule: %s = %r' % (key, val))
                    num_passed = flags.sum()
                    num_valid = is_valid.sum()
                    print('  * num_passed = %r' % (num_passed,))
                    print('  * prev_num_valid = %r' % (prev_num_valid,))
                    print('  * num_invalided = %r' % (prev_num_valid - num_valid,))
                    print('  * num_valid = %r' % (num_valid,))

        # HACK:
        # If one config for a row passes the filter then all configs should pass
        allcfg = filt_cfg.pop('allcfg', None)
        if allcfg:
            is_valid = np.logical_or(np.logical_or.reduce(is_valid.T)[:, None], is_valid)
            is_valid = np.logical_and(is_valid, participates)

        qx_list, cfgx_list = np.nonzero(is_valid)

        # Determine a good ordering of the test cases
        orderby = filt_cfg.pop('orderby', None)
        reverse = filt_cfg.pop('reverse', None)
        sortasc = filt_cfg.pop('sortasc', None)
        sortdsc = filt_cfg.pop('sortdsc', filt_cfg.pop('sortdesc', None))
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
            if orderby == 'gtscore':
                order_values = truth2_prop['gt']['score']
            elif orderby == 'gfscore':
                order_values = truth2_prop['gf']['score']
            if orderby == 'gtscore':
                order_values = truth2_prop['gt']['score']
            elif orderby == 'gfscore':
                order_values = truth2_prop['gf']['score']
            else:
                if orderby.startswith('gt_'):
                    order_values = truth2_prop['gt'][orderby[3:]]
                elif orderby.startswith('gt'):
                    order_values = truth2_prop['gt'][orderby[2:]]
                elif orderby.startswith('gf_'):
                    order_values = truth2_prop['gt'][orderby[3:]]
                elif orderby.startswith('gf'):
                    order_values = truth2_prop['gt'][orderby[2:]]
                else:
                    raise NotImplementedError('Unknown orerby=%r' % (orderby,))
        else:
            order_values = np.arange(is_valid.size).reshape(is_valid.shape)

        # Convert mask into indicies
        flat_order = order_values[is_valid]
        # Flat sorting indeices in a matrix
        if reverse:
            sortx = flat_order.argsort()[::-1]
        else:
            sortx = flat_order.argsort()
        qx_list = qx_list.take(sortx, axis=0)
        cfgx_list = cfgx_list.take(sortx, axis=0)

        max_pername = filt_cfg.pop('max_pername', None)
        # Return at most ``max_pername`` annotation examples per name
        if max_pername is not None:
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
            is_valid = vt.index_to_boolmask(_valid_index, is_valid.shape, hack=True)
            qx_list = _qx_list
            cfgx_list = _cfgx_list

        if return_mask:
            return is_valid
        else:
            index = filt_cfg.pop('index', None)
            if index is not None:
                if verbose:
                    print('Taking index sample from len(qx_list) = %r' % (len(qx_list),))
                if isinstance(index, six.string_types):
                    index = ut.smart_cast(index, slice)
                qx_list = ut.take(qx_list, index)
                cfgx_list = ut.take(cfgx_list, index)

            ut.delete_keys(filt_cfg, ['_cfgstr', '_cfgindex', '_cfgname', '_cfgtype'])

            if len(filt_cfg) > 0:
                raise NotImplementedError('Unhandled filt_cfg.keys() = %r' % (filt_cfg.keys()))

            case_pos_list = np.vstack((qx_list, cfgx_list)).T
            return case_pos_list

    def get_truth2_prop(testres, qaids=None):
        r"""
        Returns:
            tuple: (truth2_prop, prop2_mat)

        CommandLine:
            python -m ibeis.expt.test_result --exec-get_truth2_prop --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> import ibeis
            >>> ibs, testres = ibeis.testdata_expts('PZ_MTEST', a=['ctrl'])
            >>> (truth2_prop, prop2_mat) = testres.get_truth2_prop()
            >>> result = ('(truth2_prop, prop2_mat) = %s' % (str((truth2_prop, prop2_mat)),))
            >>> print(result)
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> ut.show_if_requested()
        """
        ibs = testres.ibs
        test_qaids = testres.get_test_qaids() if qaids is None else qaids

        #test_qaids = ut.random_sample(test_qaids, 20)
        truth2_prop = ut.ddict(ut.odict)

        # TODO: have this function take in a case_pos_list as input instead
        participates = testres.get_infoprop_mat('participant', test_qaids)

        truth2_prop['gt']['aid'] = testres.get_infoprop_mat('qx2_gt_aid', test_qaids)
        truth2_prop['gf']['aid'] = testres.get_infoprop_mat('qx2_gf_aid', test_qaids)
        truth2_prop['gt']['rank'] = testres.get_infoprop_mat('qx2_gt_rank', test_qaids)
        truth2_prop['gf']['rank'] = testres.get_infoprop_mat('qx2_gf_rank', test_qaids)

        truth2_prop['gt']['score'] = np.nan_to_num(testres.get_infoprop_mat('qx2_gt_raw_score', test_qaids))
        truth2_prop['gf']['score'] = np.nan_to_num(testres.get_infoprop_mat('qx2_gf_raw_score', test_qaids))

        # Cast nans to ints (that are participants)
        # if False:
        for truth in ['gt', 'gf']:
            rank_mat = truth2_prop[truth]['rank']
            rank_mat[np.logical_and(np.isnan(rank_mat), participates)] = testres.get_worst_possible_rank()
            # truth2_prop[truth]['rank'] = rank_mat.astype(np.int)

        # Rank difference
        #hardness_degree_rank = truth2_prop['gt']['rank'] - truth2_prop['gf']['rank']
        #is_failure = hardness_degree_rank >= 0
        #is_success = hardness_degree_rank < 0
        is_success = truth2_prop['gt']['rank'] == 0
        is_failure = np.logical_not(is_success)

        # THIS IS NOT THE CASE IF THERE ARE UNKNOWN INDIVIDUALS IN THE DATABASE
        assert np.all(is_success == (truth2_prop['gt']['rank'] == 0))

        # WEIRD THINGS HAPPEN WHEN UNKNOWNS ARE HERE
        #hardness_degree_rank[is_success]
        #is_weird = hardness_degree_rank == 0  # These probably just completely failure spatial verification

        # Get timedelta and annotmatch rowid
        for truth in ['gt', 'gf']:
            aid_mat = truth2_prop[truth]['aid']
            timedelta_mat = np.vstack([
                ibs.get_annot_pair_timdelta(test_qaids, aids)
                for aids in aid_mat.T
            ]).T
            annotmatch_rowid_mat = np.vstack([
                ibs.get_annotmatch_rowid_from_superkey(test_qaids, aids)
                for aids in aid_mat.T
            ]).T
            truth2_prop[truth]['annotmatch_rowid']  = annotmatch_rowid_mat
            truth2_prop[truth]['timedelta'] = timedelta_mat
        prop2_mat = {}

        prop2_mat['is_success'] = is_success
        prop2_mat['is_failure'] = is_failure
        prop2_mat['participates'] = participates
        return truth2_prop, prop2_mat

    def interact_individual_result(testres, qaid, cfgx=0):
        ibs = testres.ibs
        cfgx_list = ut.ensure_iterable(cfgx)
        qreq_list = ut.take(testres.cfgx2_qreq_, cfgx_list)
        # Preload any requested configs
        qres_list = [qreq_.load_cached_qres(qaid) for qreq_ in qreq_list]
        cfgx2_shortlbl = testres.get_short_cfglbls()
        show_kwargs = {
            'N': 3,
            'ori': True,
            'ell_alpha': .9,
        }
        # SHOW ANALYSIS
        show_kwargs['show_query'] = False
        show_kwargs['viz_name_score'] = True
        show_kwargs['show_timedelta'] = True
        show_kwargs['show_gf'] = True
        show_kwargs['with_figtitle'] = False
        for cfgx, qres, qreq_ in zip(cfgx_list, qres_list, qreq_list):
            query_lbl = cfgx2_shortlbl[cfgx]
            fnum = cfgx
            qres.ishow_analysis(
                ibs, figtitle=query_lbl, fnum=fnum, annot_mode=1, qreq_=qreq_,
                **show_kwargs)

    def get_pipecfg_args(testres):
        if '_cfgstr' in testres.common_cfgdict:
            pipecfg_args = [testres.common_cfgdict['_cfgstr']]
        else:
            pipecfg_args = ut.unique_ordered(
                [cfg['_cfgstr'] for cfg in testres.varied_cfg_list])
        return ' ' .join(pipecfg_args)

    def get_annotcfg_args(testres):
        """
        CommandLine:
            # TODO: More robust fix
            # To reproduce the error
            ibeis -e rank_cdf --db humpbacks_fb -a default:mingt=2,qsize=10,dsize=100 default:qmingt=2,qsize=10,dsize=100 -t default:proot=BC_DTW,decision=max,crop_dim_size=500,crop_enabled=True,manual_extract=False,use_te_scorer=True,ignore_notch=True,te_score_weight=0.5 --show
        """
        if '_cfgstr' in testres.common_acfg['common']:
            annotcfg_args = [testres.common_acfg['common']['_cfgstr']]
        else:
            try:
                annotcfg_args = ut.unique_ordered([
                    acfg['common']['_cfgstr']
                    for acfg in testres.varied_acfg_list])
            except KeyError:
                # HACK FIX
                try:
                    annotcfg_args = ut.unique_ordered([
                        acfg['_cfgstr']
                        for acfg in testres.varied_acfg_list])
                except KeyError:
                    annotcfg_args = ut.unique_ordered([
                        acfg['qcfg__cfgstr']
                        for acfg in testres.varied_acfg_list])
        return ' ' .join(annotcfg_args)

    def reconstruct_test_flags(testres):
        flagstr =  ' '.join([
            '-a ' + testres.get_annotcfg_args(),
            '-t ' + testres.get_pipecfg_args(),
            '--db ' + testres.ibs.get_dbname()
        ])
        return flagstr

    def draw_score_diff_disti(testres):
        r"""

        CommandLine:
            python -m ibeis --tf TestResult.draw_score_diff_disti --show -a varynannots_td -t best --db PZ_Master1
            python -m ibeis --tf TestResult.draw_score_diff_disti --show -a varynannots_td -t best --db GZ_Master1
            python -m ibeis --tf TestResult.draw_score_diff_disti --show -a varynannots_td1h -t best --db GIRM_Master1

            python -m ibeis --tf TestResult.draw_score_diff_disti --show -a varynannots_td:qmin_pername=3,dpername=2 -t best --db PZ_Master1

            python -m ibeis --tf get_annotcfg_list -a varynannots_td -t best --db PZ_Master1
            13502
            python -m ibeis --tf draw_match_cases --db PZ_Master1 -a varynannots_td:dsample_size=.01 -t best  --show --qaid 13502
            python -m ibeis --tf draw_match_cases --db PZ_Master1 -a varynannots_td -t best  --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> import ibeis
            >>> ibs, testres = ibeis.testdata_expts('PZ_Master1', a=['varynannots_td'], t=['best'])
            >>> result = testres.draw_score_diff_disti()
            >>> print(result)
            >>> ut.show_if_requested()
        """
        import plottool as pt
        import vtool as vt

        # dont look at filtered cases
        ibs = testres.ibs
        qaids = testres.get_test_qaids()
        qaids = ibs.get_annot_tag_filterflags(qaids, {'has_none': 'timedeltaerror'})

        gt_rawscore = testres.get_infoprop_mat('qx2_gt_raw_score')
        gf_rawscore = testres.get_infoprop_mat('qx2_gf_raw_score')

        gt_valid_flags_list = np.isfinite(gt_rawscore).T
        gf_valid_flags_list = np.isfinite(gf_rawscore).T

        cfgx2_gt_scores = vt.zipcompress(gt_rawscore.T, gt_valid_flags_list)
        cfgx2_gf_scores = vt.zipcompress(gf_rawscore.T, gf_valid_flags_list)

        # partition by rank
        gt_rank     = testres.get_infoprop_mat('qx2_gt_rank')
        gf_ranks    = testres.get_infoprop_mat('qx2_gf_rank')
        cfgx2_gt_ranks  = vt.zipcompress(gt_rank.T,     gt_valid_flags_list)
        cfgx2_rank0_gt_scores = vt.zipcompress(cfgx2_gt_scores, [ranks == 0 for ranks in cfgx2_gt_ranks])
        cfgx2_rankX_gt_scores = vt.zipcompress(cfgx2_gt_scores, [ranks > 0 for ranks in cfgx2_gt_ranks])
        cfgx2_gf_ranks  = vt.zipcompress(gf_ranks.T,    gf_valid_flags_list)
        cfgx2_rank0_gf_scores = vt.zipcompress(cfgx2_gf_scores, [ranks == 0 for ranks in cfgx2_gf_ranks])

        #valid_gtranks = gt_rank[isvalid]
        #valid_qaids = qaids[isvalid]
        # Hack remove timdelta error
        #valid_qaids = valid_qaids[flags]
        #valid_gt_rawscore = valid_gt_rawscore[flags]
        #valid_gtranks = valid_gtranks[flags]

        xdata = list(map(len, testres.cfgx2_daids))

        USE_MEDIAN = True  # not ut.get_argflag('--use-mean')
        #USE_LOG = True
        USE_LOG = False
        if USE_MEDIAN:
            ave = np.median
            dev = vt.median_abs_dev
        else:
            ave = np.mean
            dev = np.std

        def make_interval_args(arr_list, ave=ave, dev=dev, **kwargs):
            #if not USE_MEDIAN:
            #    # maybe approximate median by removing the most extreme values
            #    arr_list = [np.array(sorted(arr))[5:-5] for arr in arr_list]
            import utool as ut
            if USE_LOG:
                arr_list = list(map(lambda x: np.log(x + 1), arr_list))
            sizes_ = list(map(len, arr_list))
            ydata_ = list(map(ave, arr_list))
            spread_ = list(map(dev, arr_list))
            #ut.get_stats(arr_list, axis=0)
            label = kwargs.get('label', '')
            label += ' ' + ut.get_funcname(ave)
            kwargs['label'] = label
            print(label + 'score stats : ' +
                  ut.repr2(ut.get_jagged_stats(arr_list, use_median=True), nl=1, precision=1))
            return ydata_, spread_, kwargs, sizes_

        args_list1 = [
            make_interval_args(cfgx2_gt_scores, label='GT', color=pt.TRUE_BLUE),
            make_interval_args(cfgx2_gf_scores, label='GF', color=pt.FALSE_RED),
        ]

        args_list2 = [
            make_interval_args(cfgx2_rank0_gt_scores, label='GT-rank = 0', color=pt.LIGHT_GREEN),
            make_interval_args(cfgx2_rankX_gt_scores, label='GT-rank > 0', color=pt.YELLOW),
            make_interval_args(cfgx2_rank0_gf_scores, label='GF-rank = 0', color=pt.PINK),
            #make_interval_args(cfgx2_rank2_gt_scores, label='gtrank < 2'),
        ]

        plotargs_list = [args_list1, args_list2]
        #plotargs_list = [args_list1]
        ymax = -np.inf
        ymin = np.inf
        for args_list in plotargs_list:
            ydata_list = np.array(ut.get_list_column(args_list, 0))
            spread = np.array(ut.get_list_column(args_list, 1))
            ymax = max(ymax, np.array(ydata_list + spread).max())
            ymin = min(ymax, np.array(ydata_list - spread).min())

        ylabel = 'log name score' if USE_LOG else 'name score'

        statickw = dict(
            #title='scores vs dbsize',
            xlabel='database size (number of annotations)',
            ylabel=ylabel,
            #xscale='log', ymin=0, ymax=10,
            linewidth=2, spread_alpha=.5, lightbg=True, marker='o',
            #xmax='data',
            ymax=ymax, ymin=ymin, xmax='data', xmin='data',
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

            pt.multi_plot(xdata, ydata_list, spread_list=spread_list,
                          fnum=fnum, pnum=pnum_(), **plotkw2)

        #pt.adjust_subplots2(hspace=.3)
        figtitle = 'Score vs DBSize: %s' % (testres.get_title_aug())
        pt.set_figtitle(figtitle)

    def draw_rank_cdf(testres):
        """
        Wrapper
        """
        from ibeis.expt import experiment_drawing
        experiment_drawing.draw_rank_cdf(testres.ibs, testres)

    def get_sorted_config_labels(testres):
        """
        helper
        """
        key = 'qx2_bestranks'
        cfgx2_cumhist_percent, edges = testres.get_rank_percentage_cumhist(bins='dense', key=key)
        label_list = testres.get_short_cfglbls()
        label_list = [
            ('%6.2f%%' % (percent,)) +
            #ut.scalar_str(percent, precision=2)
            ' - ' + label
            for percent, label in zip(cfgx2_cumhist_percent.T[0], label_list)]
        sortx = cfgx2_cumhist_percent.T[0].argsort()[::-1]
        label_list = ut.take(label_list, sortx)
        return label_list

    def find_score_thresh_cutoff(testres):
        """
        FIXME
        DUPLICATE CODE
        rectify with experiment_drawing
        """
        #import plottool as pt
        import vtool as vt
        if ut.VERBOSE:
            print('[dev] FIX DUPLICATE CODE find_thresh_cutoff')
        #from ibeis.expt import cfghelpers

        assert len(testres.cfgx2_qreq_) == 1, 'can only specify one config here'
        cfgx = 0
        #qreq_ = testres.cfgx2_qreq_[cfgx]
        test_qaids = testres.get_test_qaids()
        gt_rawscore = testres.get_infoprop_mat('qx2_gt_raw_score').T[cfgx]
        gf_rawscore = testres.get_infoprop_mat('qx2_gf_raw_score').T[cfgx]

        # FIXME: may need to specify which cfg is used in the future
        #isvalid = testres.case_sample2(filt_cfg, return_mask=True).T[cfgx]

        tp_nscores = gt_rawscore
        tn_nscores = gf_rawscore
        tn_qaids = tp_qaids = test_qaids
        #encoder = vt.ScoreNormalizer(target_tpr=.7)
        #print(qreq_.get_cfgstr())
        part_attrs = {1: {'qaid': tp_qaids},
                      0: {'qaid': tn_qaids}}

        fpr = None
        tpr = .85
        encoder = vt.ScoreNormalizer(adjust=8, fpr=fpr, tpr=tpr, monotonize=True)
        #tp_scores = tp_nscores
        #tn_scores = tn_nscores
        name_scores, labels, attrs = encoder._to_xy(tp_nscores, tn_nscores, part_attrs)
        encoder.fit(name_scores, labels, attrs)
        #score_thresh = encoder.learn_threshold()
        score_thresh = encoder.learn_threshold2()

        # Find intersection point
        # TODO: add to score normalizer.
        # Improve robustness
        #pt.figure()
        #pt.plot(xdata, curve)
        #pt.plot(x_submax, y_submax, 'o')
        return score_thresh

    def print_percent_identification_success(testres):
        """
        Prints names identified (at rank 1) / names queried.
        This combines results over multiple queries of a particular name using
        max

        OLD, MAYBE DEPRIATE

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
        """
        ibs = testres.ibs
        qaids = testres.get_test_qaids()
        unique_nids, groupxs = vt.group_indices(ibs.get_annot_nids(qaids))

        qx2_gt_raw_score = testres.get_infoprop_mat('qx2_gt_raw_score')
        qx2_gf_raw_score = testres.get_infoprop_mat('qx2_gf_raw_score')

        nx2_gt_raw_score = np.array([
            np.nanmax(scores, axis=0)
            for scores in vt.apply_grouping(qx2_gt_raw_score, groupxs)])

        nx2_gf_raw_score = np.array([
            np.nanmax(scores, axis=0)
            for scores in vt.apply_grouping(qx2_gf_raw_score, groupxs)])

        cfgx2_success = (nx2_gt_raw_score > nx2_gf_raw_score).T
        print('Identification success (names identified / names queried)')
        for cfgx, success in enumerate(cfgx2_success):
            pipelbl = testres.cfgx2_lbl[cfgx]
            percent = 100 * success.sum() / len(success)
            print('%2d) success = %r/%r = %.2f%% -- %s' % (
                cfgx, success.sum(), len(success), percent, pipelbl))

    def print_config_overlap(testres):
        truth2_prop, prop2_mat = testres.get_truth2_prop()
        qx2_gt_ranks = truth2_prop['gt']['rank']
        qx2_success = (qx2_gt_ranks == 0)
        cfgx2_num_correct = np.nansum(qx2_success, axis=0)
        best_cfgx = cfgx2_num_correct.argmax()

        print('Config Overlap')

        # Matrix version
        disjoint_mat = np.zeros((testres.nConfig, testres.nConfig), dtype=np.int32)
        disjoint_mat2 = np.zeros((testres.nConfig, testres.nConfig), dtype=np.int32)
        for cfgx1 in range(testres.nConfig):
            for cfgx2 in range(testres.nConfig):
                if cfgx1 == cfgx2:
                    continue
                success_qx1 = np.where(qx2_success.T[cfgx1])[0]
                success_qx2 = np.where(qx2_success.T[cfgx2])[0]
                union_ = np.union1d(success_qx1, success_qx2),
                isect_ = np.intersect1d(success_qx1, success_qx2)
                disjoints = np.setdiff1d(union_, isect_)
                disjoint_mat[cfgx1][cfgx2] = len(disjoints)
                disjoint2 = np.setdiff1d(success_qx2, isect_)
                disjoint_mat2[cfgx2][cfgx1] = len(disjoint2)
        print('cfgx1 and cfgx2 have <x> not in common')
        print('disjoint_mat =\n%s' % (disjoint_mat,))
        print('cfgx1 helps cfgx2 by <x>')
        print('disjoint_mat2 =\n%s' % (disjoint_mat2,))

        # Numbered version
        print('best_cfgx = %r' % (best_cfgx,))
        for cfgx in range(testres.nConfig):
            if cfgx == best_cfgx:
                continue
            pipelbl = testres.cfgx2_lbl[cfgx]
            qx2_anysuccess = np.logical_or(qx2_success.T[cfgx], qx2_success.T[best_cfgx])
            # Queries that other got right that best did not get right
            qx2_othersuccess = np.logical_and(qx2_anysuccess, np.logical_not(qx2_success.T[best_cfgx]))
            print('cfgx %d) has %d success cases that that the best config does not have -- %s' % (cfgx, qx2_othersuccess.sum(), pipelbl))

        qx2_success.T[cfgx]

    def load_full_chipmatch_results(testres):
        #cfgx2_qres
        pass

    def compare_score_pdfs(testres):
        """
        CommandLine:
            python -m ibeis.expt.test_result --exec-compare_score_pdfs --show --present
            python -m ibeis.expt.test_result --exec-compare_score_pdfs --show --present --nocache
            python -m ibeis.expt.test_result --exec-compare_score_pdfs --show --present -a timectrl:qindex=0:50

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> import ibeis
            >>> defaultdb = 'PZ_MTEST'
            >>> defaultdb = 'PZ_Master1'
            >>> ibs, testres = ibeis.testdata_expts(
            >>>     defaultdb=defaultdb, a=['timectrl'], t=['best'])
            >>> testres.compare_score_pdfs()
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> ut.show_if_requested()
        """
        #from ibeis.init import main_helpers
        import utool as ut
        #import plottool as pt
        ut.ensure_pylab_qt4()

        testres.draw_annot_scoresep(f='fail=False')
        #pt.adjust_subplots(bottom=.25, top=.8)
        encoder = testres.draw_feat_scoresep(f='fail=False', disttype=None)
        #pt.adjust_subplots(bottom=.25, top=.8)
        #encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['lnbnn'])
        #encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['ratio'])
        #encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['L2_sift'])
        encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['lnbnn', 'fg'])
        #pt.adjust_subplots(bottom=.25, top=.8)

        #ibs, testres = main_helpers.testdata_expts(
        #    defaultdb=defaultdb, a=['timectrl'], t=['best:lnbnn_on=False,ratio_thresh=1.0'])
        #encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['ratio'])
        #encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['lnbnn'])
        #encoder = testres.draw_feat_scoresep(f='fail=False', disttype=['L2_sift'])
        # TODO:
        return encoder

    def draw_annot_scoresep(testres, f=None):
        from ibeis.expt import experiment_drawing
        experiment_drawing.draw_annot_scoresep(testres.ibs, testres, f=f)

    def draw_feat_scoresep(testres, f=None, disttype=None):
        r"""
        SeeAlso:
            ibeis.algo.hots.scorenorm.train_featscore_normalizer

        CommandLine:
            python -m ibeis --tf TestResult.draw_feat_scoresep --show
            python -m ibeis --tf TestResult.draw_feat_scoresep --show -t default:sv_on=[True,False]
            python -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_Master1
            python -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_Master1 --disttype=L2_sift,fg
            python -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_Master1 --disttype=L2_sift
            python -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST -t best:lnbnn_on=True --namemode=True
            python -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST -t best:lnbnn_on=True --namemode=False

            python -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST --disttype=L2_sift
            python -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST --disttype=L2_sift -t best:SV=False

            utprof.py -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_Master1
            utprof.py -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_Master1 --fsvx=1:2
            utprof.py -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_Master1 --fsvx=0:1

            utprof.py -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_Master1 -t best:lnbnn_on=False,bar_l2_on=True  --fsvx=0:1

            # We want to query the oxford annots taged query
            # and we want the database to contain
            # K correct images per query, as well as the distractors

            python -m ibeis --tf TestResult.draw_feat_scoresep  --show --db Oxford -a default:qhas_any=\(query,\),dpername=1,exclude_reference=True,minqual=ok
            python -m ibeis --tf TestResult.draw_feat_scoresep  --show --db Oxford -a default:qhas_any=\(query,\),dpername=1,exclude_reference=True,minqual=good

            python -m ibeis --tf get_annotcfg_list  --db PZ_Master1 -a timectrl --acfginfo --verbtd  --veryverbtd --nocache-aid

            python -m ibeis --tf TestResult.draw_feat_scoresep --show --db PZ_MTEST --disttype=ratio

        Example:
            >>> # SCRIPT
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> from ibeis.init import main_helpers
            >>> disttype = ut.get_argval('--disttype', type_=list, default=None)
            >>> ibs, testres = main_helpers.testdata_expts(
            >>>     defaultdb='PZ_MTEST', a=['timectrl'], t=['best'])
            >>> f = ut.get_argval(('--filt', '-f'), type_=list, default=[''])
            >>> testres.draw_feat_scoresep(f=f)
            >>> ut.show_if_requested()
        """
        print('[testres] draw_feat_scoresep')
        import plottool as pt

        def load_feat_scores(qreq_, qaids):
            import ibeis  # NOQA
            from os.path import dirname, join  # NOQA
            # HACKY CACHE
            cfgstr = qreq_.get_cfgstr(with_input=True)
            cache_dir = join(dirname(dirname(ibeis.__file__)), 'TMP_FEATSCORE_CACHE')
            namemode = ut.get_argval('--namemode', default=True)
            fsvx = ut.get_argval('--fsvx', type_='fuzzy_subset',
                                 default=slice(None, None, None))
            threshx = ut.get_argval('--threshx', type_=int, default=None)
            thresh = ut.get_argval('--thresh', type_=float, default=.9)
            num = ut.get_argval('--num', type_=int, default=1)
            cfg_components = [cfgstr, disttype, namemode, fsvx, threshx, thresh, f, num]
            cache_cfgstr = ','.join(ut.lmap(six.text_type, cfg_components))
            cache_hashid = ut.hashstr27(cache_cfgstr + '_v1')
            cache_name = ('get_cfgx_feat_scores_' + cache_hashid)
            @ut.cached_func(cache_name, cache_dir=cache_dir, key_argx=[],
                            use_cache=True)
            def get_cfgx_feat_scores(qreq_, qaids):
                from ibeis.algo.hots import scorenorm
                cm_list = qreq_.execute_subset(qaids)
                # print('Done loading cached chipmatches')
                tup = scorenorm.get_training_featscores(qreq_, cm_list, disttype,
                                                        namemode, fsvx, threshx,
                                                        thresh, num=num)
                # print(ut.depth_profile(tup))
                tp_scores, tn_scores, scorecfg = tup
                return tp_scores, tn_scores, scorecfg

            tp_scores, tn_scores, scorecfg = get_cfgx_feat_scores(qreq_, qaids)
            return tp_scores, tn_scores, scorecfg

        valid_case_pos = testres.case_sample2(filt_cfg=f, return_mask=False)
        cfgx2_valid_qxs = ut.group_items(valid_case_pos.T[0], valid_case_pos.T[1])
        test_qaids = testres.get_test_qaids()
        cfgx2_valid_qaids = ut.map_dict_vals(ut.partial(ut.take, test_qaids), cfgx2_valid_qxs)

        join_acfgs = True

        # TODO: option to average over pipeline configurations
        if join_acfgs:
            groupxs = testres.get_pcfg_groupxs()
        else:
            groupxs = list(zip(range(len(testres.cfgx2_qreq_))))
        grouped_qreqs = ut.apply_grouping(testres.cfgx2_qreq_, groupxs)

        grouped_scores = []
        for cfgxs, qreq_group in zip(groupxs, grouped_qreqs):
            # testres.print_pcfg_info()
            score_group = []
            for cfgx, qreq_ in zip(cfgxs, testres.cfgx2_qreq_):
                print('Loading cached chipmatches')
                qaids = cfgx2_valid_qaids[cfgx]
                tp_scores, tn_scores, scorecfg = load_feat_scores(qreq_, qaids)
                score_group.append((tp_scores, tn_scores, scorecfg))
            grouped_scores.append(score_group)

        cfgx2_shortlbl = testres.get_short_cfglbls(join_acfgs=join_acfgs)
        for score_group, lbl in zip(grouped_scores, cfgx2_shortlbl):
            tp_scores = np.hstack(ut.take_column(score_group, 0))
            tn_scores = np.hstack(ut.take_column(score_group, 1))
            scorecfg = '+++'.join(ut.unique(ut.take_column(score_group, 2)))
            score_group
            # TODO: learn this score normalizer as a model
            # encoder = vt.ScoreNormalizer(adjust=4, monotonize=False)
            encoder = vt.ScoreNormalizer(adjust=2, monotonize=True)
            encoder.fit_partitioned(tp_scores, tn_scores, verbose=False)
            figtitle = 'Feature Scores: %s, %s' % (scorecfg, lbl)
            fnum = None

            vizkw = {}
            sephack = ut.get_argflag('--sephack')
            if not sephack:
                vizkw['target_tpr'] = .95
                vizkw['score_range'] = (0, 1.0)

            encoder.visualize(
                figtitle=figtitle, fnum=fnum,
                with_scores=False,
                #with_prebayes=True,
                with_prebayes=False,
                with_roc=True,
                with_postbayes=False,
                #with_postbayes=True,
                **vizkw
            )
            icon = testres.ibs.get_database_icon()
            if icon is not None:
                pt.overlay_icon(icon, coords=(1, 0), bbox_alignment=(1, 0))

            if ut.get_argflag('--contextadjust'):
                pt.adjust_subplots(left=.1, bottom=.25, wspace=.2, hspace=.2)
                pt.adjust_subplots2(use_argv=True)
        return encoder


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.expt.test_result
        python -m ibeis.expt.test_result --allexamples
        python -m ibeis.expt.test_result --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
