# -*- coding: utf-8 -*-
"""
TODO:
    save and load TestResult classes
"""
from __future__ import absolute_import, division, print_function
import six
import numpy as np
#import six
import utool
import utool as ut
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[expt_harn]')


def combine_test_results(ibs, test_result_list):
    """
    CommandLine:
        python -m ibeis.experiments.experiment_storage --exec-combine_test_results

        python -m ibeis.experiments.experiment_drawing --exec-draw_rank_cdf --db PZ_MTEST --show
        python -m ibeis.experiments.experiment_drawing --exec-draw_rank_cdf --db PZ_Master0 --show
        python -m ibeis.experiments.experiment_drawing --exec-draw_rank_cdf --db PZ_MTEST --show -a varysize -t default
        python -m ibeis.experiments.experiment_drawing --exec-draw_rank_cdf --db PZ_MTEST --show -a varysize -t default

    >>> # DISABLE_DOCTEST
    >>> from ibeis.experiments.experiment_storage import *  # NOQA
    >>> from ibeis.experiments import experiment_harness
    >>> ibs, test_result_list = experiment_harness.testdata_expts('PZ_MTEST', ['varysize'])
    >>> combine_test_results(ibs, test_result_list)
    """
    try:
        assert ut.list_allsame([test_result.qaids for test_result in test_result_list]), ' cannot handle non-same qaids right now'
    except AssertionError as ex:
        ut.printex(ex)
        ut.embed()
        raise

    from ibeis.experiments import annotation_configs
    acfg_list = [test_result.acfg for test_result in test_result_list]
    acfg_lbl_list = annotation_configs.get_varied_labels(acfg_list)

    qaids = test_result.qaids
    agg_cfg_list = ut.flatten([test_result.cfg_list for test_result in test_result_list])
    agg_cfgx2_lbls = ut.flatten([[lbl + acfg_lbl for lbl in test_result.cfgx2_lbl] for test_result, acfg_lbl in zip(test_result_list, acfg_lbl_list)])
    agg_cfgx2_cfgreinfo = ut.flatten([test_result.cfgx2_cfgresinfo for test_result in test_result_list])
    agg_cfgx2_qreq_ = ut.flatten([test_result.cfgx2_qreq_ for test_result in test_result_list])
    big_test_result = TestResult(agg_cfg_list, agg_cfgx2_lbls, 'foo', 'foo2', agg_cfgx2_cfgreinfo, agg_cfgx2_qreq_, qaids)
    test_result = big_test_result
    return test_result


@six.add_metaclass(ut.ReloadingMetaclass)
class TestResult(object):
    def __init__(test_result, cfg_list, cfgx2_lbl, lbl, testnameid, cfgx2_cfgresinfo, cfgx2_qreq_, qaids):
        assert len(cfg_list) == len(cfgx2_lbl), 'bad lengths: %r != %r' % (len(cfg_list), len(cfgx2_lbl))
        assert len(cfgx2_qreq_) == len(cfgx2_lbl), 'bad lengths: %r != %r' % (len(cfgx2_qreq_), len(cfgx2_lbl))
        assert len(cfgx2_cfgresinfo) == len(cfgx2_lbl), 'bad lengths: %r != %r' % (len(cfgx2_cfgresinfo), len(cfgx2_lbl))
        test_result._qaids = qaids
        #test_result.daids = daids
        test_result.cfg_list         = cfg_list
        test_result.cfgx2_lbl        = cfgx2_lbl
        test_result.lbl              = lbl
        test_result.testnameid       = testnameid
        test_result.cfgx2_cfgresinfo = cfgx2_cfgresinfo
        test_result.cfgx2_qreq_      = cfgx2_qreq_

    @property
    def qaids(test_result):
        return test_result._qaids

    @ut.memoize
    def get_rank_mat(test_result):
        # Ranks of Best Results
        cfgx2_bestranks = ut.get_list_column(test_result.cfgx2_cfgresinfo, 'qx2_bestranks')
        rank_mat = np.vstack(cfgx2_bestranks).T  # concatenate each query rank across configs
        # Set invalid ranks to the worse possible rank
        worst_possible_rank = test_result.get_worst_possible_rank()
        rank_mat[rank_mat == -1] =  worst_possible_rank
        return rank_mat

    def get_worst_possible_rank(test_result):
        #worst_possible_rank = max(9001, len(test_result.daids) + 1)
        worst_possible_rank = max([len(qreq_.get_external_daids()) for qreq_ in test_result.cfgx2_qreq_]) + 1
        #worst_possible_rank = len(test_result.daids) + 1
        return worst_possible_rank

    @ut.memoize
    def get_new_hard_qx_list(test_result):
        """ Mark any query as hard if it didnt get everything correct """
        rank_mat = test_result.get_rank_mat()
        is_new_hard_list = rank_mat.max(axis=1) > 0
        new_hard_qx_list = np.where(is_new_hard_list)[0]
        return new_hard_qx_list

    def get_rank_histograms(test_result, bins=None, asdict=True):
        rank_mat = test_result.get_rank_mat()
        if bins is None:
            bins = test_result.get_rank_histogram_bins()
        elif bins == 'dense':
            bins = np.arange(test_result.get_worst_possible_rank() + 1)
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

    def get_rank_cumhist(test_result, bins='dense'):
        #test_result.rrr()
        hist_list, edges = test_result.get_rank_histograms(bins, asdict=False)
        config_cdfs = np.cumsum(hist_list, axis=1)
        return config_cdfs, edges

    def get_rank_histogram_bins(test_result):
        """ easy to see histogram bins """
        worst_possible_rank = test_result.get_worst_possible_rank()
        if worst_possible_rank > 50:
            bins = [0, 1, 5, 50, worst_possible_rank, worst_possible_rank + 1]
        elif worst_possible_rank > 5:
            bins = [0, 1, 5, worst_possible_rank, worst_possible_rank + 1]
        else:
            bins = [0, 1, 5]
        return bins

    def get_rank_histogram_bin_edges(test_result):
        bins = test_result.get_rank_histogram_bins()
        bin_keys = list(zip(bins[:-1], bins[1:]))
        return bin_keys

    def get_rank_histogram_qx_binxs(test_result):
        rank_mat = test_result.get_rank_mat()
        config_hists = test_result.get_rank_histograms()
        config_binxs = []
        bin_keys = test_result.get_rank_histogram_bin_edges()
        for hist_dict, ranks in zip(config_hists, rank_mat.T):
            bin_qxs = [np.where(np.logical_and(low <= ranks, ranks < high))[0]
                       for low, high in bin_keys]
            qx2_binx = -np.ones(len(ranks))
            for binx, qxs in enumerate(bin_qxs):
                qx2_binx[qxs] = binx
            config_binxs.append(qx2_binx)
        return config_binxs

    def get_rank_histogram_qx_sample(test_result, size=10):
        size = 10
        rank_mat = test_result.get_rank_mat()
        config_hists = test_result.get_rank_histograms()
        config_rand_bin_qxs = []
        bins = test_result.get_rank_histogram_bins()
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

    def get_full_cfgstr(test_result, cfgx):
        """ both qannots and dannots included """
        full_cfgstr = test_result.cfgx2_qreq_[cfgx].get_full_cfgstr()
        return full_cfgstr

    def get_cfgstr(test_result, cfgx):
        """ just dannots and config_str """
        cfgstr = test_result.cfgx2_qreq_[cfgx].get_cfgstr()
        return cfgstr

    def get_short_cfglbls(test_result):
        """
        Labels for published tables
        """
        repl_list = [
            ('custom', 'default'),
            ('fg_on', 'FG'),
            ('sv_on', 'SV'),
            ('rotation_invariance', 'RI'),
            ('affine_invariance', 'AI'),
            ('augment_queryside_hack', 'AQH'),
            ('nNameShortlistSVER', 'nRR'),
        ]
        import re
        cfg_lbls = test_result.cfgx2_lbl[:]
        for ser, rep in repl_list:
            cfg_lbls = [re.sub(ser, rep, lbl) for lbl in cfg_lbls]
            cfg_lbls = [':'.join(tup) if len(tup) != 2 else tup[1] if len(tup[1]) > 0 else 'BASELINE' for tup in [lbl.split(':') for lbl in cfg_lbls]]

        #from ibeis.experiments import annotation_configs
        #lblaug = annotation_configs.compress_aidcfg(test_result.acfg)['common']['_cfgstr']

        #cfg_lbls = [lbl + ':' + lblaug for lbl in cfg_lbls]

        return cfg_lbls

    @property
    def nConfig(test_result):
        return len(test_result.cfg_list)

    @property
    def nQuery(test_result):
        return len(test_result.qaids)

    @property
    def rank_mat(test_result):
        return test_result.get_rank_mat()

    def get_X_LIST(test_result):
        X_LIST = ut.get_argval('--rank-lt-list', type_=list, default=[1])
        return X_LIST

    def get_nLessX_dict(test_result):
        # Build a (histogram) dictionary mapping X (as in #ranks < X) to a list of cfg scores
        X_LIST = test_result.get_X_LIST()
        nLessX_dict = {int(X): np.zeros(test_result.nConfig) for X in X_LIST}
        for X in X_LIST:
            rank_mat = test_result.rank_mat
            lessX_ = np.logical_and(np.less(rank_mat, X), np.greater_equal(rank_mat, 0))
            nLessX_dict[int(X)] = lessX_.sum(axis=0)
        return nLessX_dict


@six.add_metaclass(ut.ReloadingMetaclass)
class ResultMetadata(object):
    def __init__(metadata, fpath, autoconnect=False):
        """
        metadata_fpath = join(figdir, 'result_metadata.shelf')
        metadata = ResultMetadata(metadata_fpath)
        """
        metadata.fpath = fpath
        metadata.dimensions = ['query_cfgstr', 'qaid']
        metadata._shelf = None
        metadata.dictstore = None
        if autoconnect:
            metadata.connect()

    def connect(metadata):
        import shelve
        metadata._shelf = shelve.open(metadata.fpath)
        if 'dictstore' not in metadata._shelf:
            dictstore = ut.AutoVivification()
            metadata._shelf['dictstore'] = dictstore
        metadata.dictstore = metadata._shelf['dictstore']

    def clear(metadata):
        dictstore = ut.AutoVivification()
        metadata._shelf['dictstore'] = dictstore
        metadata.dictstore = metadata._shelf['dictstore']

    def __del__(metadata):
        metadata.close()

    def close(metadata):
        metadata._shelf.close()

    def write(metadata):
        metadata._shelf['dictstore'] = metadata.dictstore
        metadata._shelf.sync()

    def set_global_data(metadata, cfgstr, qaid, key, val):
        metadata.dictstore[cfgstr][qaid][key] = val

    def sync_test_results(metadata, test_result):
        """ store all test results in the shelf """
        for cfgx in range(len(test_result.cfgx2_qreq_)):
            cfgstr = test_result.get_cfgstr(cfgx)
            qaids = test_result.qaids
            cfgresinfo = test_result.cfgx2_cfgresinfo[cfgx]
            for key, val_list in six.iteritems(cfgresinfo):
                for qaid, val in zip(qaids, val_list):
                    metadata.set_global_data(cfgstr, qaid, key, val)
        metadata.write()

    def get_cfgstr_list(metadata):
        cfgstr_list = list(metadata.dictstore.keys())
        return cfgstr_list

    def get_column_keys(metadata):
        unflat_colname_list = [
            [cols.keys() for cols in qaid2_cols.values()]
            for qaid2_cols in six.itervalues(metadata.dictstore)
        ]
        colname_list = ut.unique_keep_order2(ut.flatten(ut.flatten(unflat_colname_list)))
        return colname_list

    def get_square_data(metadata, cfgstr=None):
        # can only support one config at a time right now
        if cfgstr is None:
            cfgstr = metadata.get_cfgstr_list()[0]
        qaid2_cols = metadata.dictstore[cfgstr]
        qaids = list(qaid2_cols.keys())
        col_name_list = ut.unique_keep_order2(ut.flatten([cols.keys() for cols in qaid2_cols.values()]))
        #col_name_list = ['qx2_scoreexpdiff', 'qx2_gt_aid']
        #colname2_colvals = [None for colname in col_name_list]
        column_list = [
            [colvals.get(colname, None) for qaid, colvals in six.iteritems(qaid2_cols)]
            for colname in col_name_list]
        col_name_list = ['qaids'] + col_name_list
        column_list = [qaids] + column_list
        print('depth_profile(column_list) = %r' % (ut.depth_profile(column_list),))
        return col_name_list, column_list
