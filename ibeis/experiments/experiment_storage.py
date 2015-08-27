# -*- coding: utf-8 -*-
"""
TODO:
    save and load TestResult classes
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
#import six
import vtool as vt
import utool as ut
print, print_, printDBG, rrr, profile = ut.inject(
    __name__, '[expt_harn]')

from ibeis.experiments.old_storage import ResultMetadata  # NOQA


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
    #try:
    #    assert ut.list_allsame([test_result.qaids for test_result in test_result_list]), ' cannot handle non-same qaids right now'
    #except AssertionError as ex:
    #    ut.printex(ex)
    #    raise

    from ibeis.experiments import annotation_configs
    from ibeis.experiments import cfghelpers

    acfg_list = [test_result.acfg for test_result in test_result_list]
    acfg_lbl_list = annotation_configs.get_varied_labels(acfg_list)

    flat_acfg_list = annotation_configs.flatten_acfg_list(acfg_list)
    nonvaried_acfg, varied_acfg_list = cfghelpers.partition_varied_cfg_list(flat_acfg_list)

    def combine_lbls(lbl, acfg_lbl):
        if len(lbl) == 0:
            return acfg_lbl
        if len(acfg_lbl) == 0:
            return lbl
        return lbl + '+' + acfg_lbl

    #qaids = test_result.qaids
    agg_cfg_list        = ut.flatten(
        [test_result.cfg_list
         for test_result in test_result_list])
    agg_cfgx2_cfgreinfo = ut.flatten(
        [test_result.cfgx2_cfgresinfo
         for test_result in test_result_list])
    agg_cfgx2_qreq_     = ut.flatten(
        [test_result.cfgx2_qreq_
         for test_result in test_result_list])
    agg_cfgdict_list    = ut.flatten(
        [test_result.cfgdict_list
         for test_result in test_result_list])
    agg_varied_acfg_list = ut.flatten([
        [acfg] * len(test_result.cfg_list)
        for test_result, acfg in zip(test_result_list, varied_acfg_list)
    ])
    agg_cfgx2_lbls      = ut.flatten(
        [[combine_lbls(lbl, acfg_lbl) for lbl in test_result.cfgx2_lbl]
         for test_result, acfg_lbl in zip(test_result_list, acfg_lbl_list)])

    big_test_result = TestResult(agg_cfg_list, agg_cfgx2_lbls,
                                 agg_cfgx2_cfgreinfo, agg_cfgx2_qreq_)

    # Give the big test result an acfg that is common between everything
    big_test_result.acfg = annotation_configs.unflatten_acfgdict(nonvaried_acfg)
    big_test_result.cfgdict_list = agg_cfgdict_list

    big_test_result.common_acfg = annotation_configs.compress_aidcfg(big_test_result.acfg)
    big_test_result.common_cfgdict = reduce(ut.dict_intersection, big_test_result.cfgdict_list)
    big_test_result.varied_acfg_list = agg_varied_acfg_list
    big_test_result.varied_cfg_list = [ut.delete_dict_keys(cfgdict.copy(), list(big_test_result.common_cfgdict.keys()))
                                       for cfgdict in big_test_result.cfgdict_list]
    big_test_result.acfg_list = acfg_list

    #big_test_result.acfg
    test_result = big_test_result
    # big_test_result = test_result
    return test_result


@six.add_metaclass(ut.ReloadingMetaclass)
class TestResult(object):
    def __init__(test_result, cfg_list, cfgx2_lbl, cfgx2_cfgresinfo, cfgx2_qreq_):
        assert len(cfg_list) == len(cfgx2_lbl), 'bad lengths1: %r != %r' % (len(cfg_list), len(cfgx2_lbl))
        assert len(cfgx2_qreq_) == len(cfgx2_lbl), 'bad lengths2: %r != %r' % (len(cfgx2_qreq_), len(cfgx2_lbl))
        assert len(cfgx2_cfgresinfo) == len(cfgx2_lbl), 'bad lengths3: %r != %r' % (len(cfgx2_cfgresinfo), len(cfgx2_lbl))
        #test_result._qaids = qaids
        #test_result.daids = daids
        test_result.cfg_list         = cfg_list
        test_result.cfgx2_lbl        = cfgx2_lbl
        test_result.cfgx2_cfgresinfo = cfgx2_cfgresinfo
        test_result.cfgx2_qreq_      = cfgx2_qreq_
        test_result.lbl              = None
        test_result.testnameid       = None

    @property
    def ibs(test_result):
        ibs_list = [qreq_.ibs for qreq_ in test_result.cfgx2_qreq_]
        ibs = ibs_list[0]
        for ibs_ in ibs_list:
            assert ibs is ibs_, 'not all query requests are using the same controller'
        return ibs

    @property
    def qaids(test_result):
        assert test_result.has_constant_qaids(), 'must have constant qaids to use this property'
        return test_result.cfgx2_qaids[0]
        #return test_result._qaids

    @property
    def nConfig(test_result):
        return len(test_result.cfg_list)

    @property
    def nQuery(test_result):
        return len(test_result.qaids)

    @property
    def rank_mat(test_result):
        return test_result.get_rank_mat()

    @property
    def cfgx2_daids(test_result):
        daids_list = [qreq_.get_external_daids() for qreq_ in test_result.cfgx2_qreq_]
        return daids_list

    @property
    def cfgx2_qaids(test_result):
        qaids_list = [qreq_.get_external_qaids() for qreq_ in test_result.cfgx2_qreq_]
        return qaids_list

    def has_constant_daids(test_result):
        return ut.list_allsame(test_result.cfgx2_daids)

    def has_constant_qaids(test_result):
        return ut.list_allsame(test_result.cfgx2_qaids)

    def has_constant_length_daids(test_result):
        return ut.list_allsame(list(map(len, test_result.cfgx2_daids)))

    def get_infoprop_list(test_result, key, qaids=None):
        _tmp1_cfgx2_infoprop = ut.get_list_column(test_result.cfgx2_cfgresinfo, key)
        _tmp2_cfgx2_infoprop = list(map(np.array, ut.util_list.replace_nones(_tmp1_cfgx2_infoprop, np.nan)))
        if qaids is not None:
            flags_list = [np.in1d(aids_, qaids) for aids_ in test_result.cfgx2_qaids]
            cfgx2_infoprop = vt.zipcompress(_tmp2_cfgx2_infoprop, flags_list)
        else:
            cfgx2_infoprop = _tmp2_cfgx2_infoprop
        if key == 'qx2_bestranks':
            # hack
            for infoprop in cfgx2_infoprop:
                infoprop[infoprop == -1] = test_result.get_worst_possible_rank()
        return cfgx2_infoprop

    def get_infoprop_mat(test_result, key, qaids=None):
        """
        key = 'qx2_gf_raw_score'
        key = 'qx2_gt_raw_score'
        """
        cfgx2_infoprop = test_result.get_infoprop_list(key, qaids)
        infoprop_mat = np.vstack(cfgx2_infoprop).T  # concatenate each query rank across configs
        return infoprop_mat

    @ut.memoize
    def get_rank_mat(test_result, qaids=None):
        # Ranks of Best Results
        #get_infoprop_mat(test_result, 'qx2_bestranks')
        rank_mat = test_result.get_infoprop_mat('qx2_bestranks', qaids=qaids)
        #cfgx2_bestranks = ut.get_list_column(test_result.cfgx2_cfgresinfo, 'qx2_bestranks')
        #rank_mat = np.vstack(cfgx2_bestranks).T  # concatenate each query rank across configs
        # Set invalid ranks to the worse possible rank
        #worst_possible_rank = test_result.get_worst_possible_rank()
        #rank_mat[rank_mat == -1] =  worst_possible_rank
        return rank_mat

    def get_worst_possible_rank(test_result):
        #worst_possible_rank = max(9001, len(test_result.daids) + 1)
        worst_possible_rank = max([len(qreq_.get_external_daids()) for qreq_ in test_result.cfgx2_qreq_]) + 1
        #worst_possible_rank = len(test_result.daids) + 1
        return worst_possible_rank

    def get_rank_histograms(test_result, bins=None, asdict=True, jagged=False):
        if bins is None:
            bins = test_result.get_rank_histogram_bins()
        elif bins == 'dense':
            bins = np.arange(test_result.get_worst_possible_rank() + 1)
        if jagged:
            assert not asdict
            cfgx2_bestranks = test_result.get_infoprop_list('qx2_bestranks')
            cfgx2_bestranks = [ut.list_replace(bestranks, -1, test_result.get_worst_possible_rank()) for bestranks in cfgx2_bestranks]
            cfgx2_hist = np.zeros((len(cfgx2_bestranks), len(bins) - 1), dtype=np.int32)
            for cfgx, ranks in enumerate(cfgx2_bestranks):
                bin_values, bin_edges  = np.histogram(ranks, bins=bins)
                assert len(ranks) == bin_values.sum(), 'should sum to be equal'
                cfgx2_hist[cfgx] = bin_values
            return cfgx2_hist, bin_edges

        rank_mat = test_result.get_rank_mat()
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

    def get_rank_percentage_cumhist(test_result, bins='dense'):
        r"""
        Args:
            bins (unicode): (default = u'dense')

        Returns:
            tuple: (config_cdfs, edges)

        CommandLine:
            python -m ibeis.experiments.experiment_storage --exec-get_rank_percentage_cumhist
            python -m ibeis.experiments.experiment_storage --exec-get_rank_percentage_cumhist -t baseline -a uncontrolled controlled

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.experiments.experiment_drawing import *  # NOQA
            >>> from ibeis.init import main_helpers
            >>> ibs, test_result = main_helpers.testdata_expts('PZ_MTEST')
            >>> bins = u'dense'
            >>> (config_cdfs, edges) = test_result.get_rank_percentage_cumhist(bins)
            >>> result = ('(config_cdfs, edges) = %s' % (str((config_cdfs, edges)),))
            >>> print(result)
        """
        #test_result.rrr()
        cfgx2_hist, edges = test_result.get_rank_histograms(bins, asdict=False, jagged=True)
        cfgx2_cumhist = np.cumsum(cfgx2_hist, axis=1)
        cfgx2_cumhist_percent = 100 * cfgx2_cumhist / cfgx2_cumhist.T[-1].T[:, None]
        return cfgx2_cumhist_percent, edges

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

    @ut.memoize
    def get_cfgstr(test_result, cfgx):
        """ just dannots and config_str """
        cfgstr = test_result.cfgx2_qreq_[cfgx].get_cfgstr()
        return cfgstr

    def get_short_cfglbls(test_result):
        """
        Labels for published tables

        cfg_lbls = ['baseline:nRR=200+default:', 'baseline:+default:']
        """
        cfg_lbls = test_result.cfgx2_lbl[:]
        import re
        repl_list = [
            ('candidacy_', ''),
            #('custom', 'default'),
            ('fg_on', 'FG'),
            ('sv_on', 'SV'),
            ('rotation_invariance', 'RI'),
            ('affine_invariance', 'AI'),
            ('augment_queryside_hack', 'AQH'),
            ('nNameShortlistSVER', 'nRR'),
            ('sample_per_ref_name', 'per_ref_name'),
            ('prescore_method=\'csum\',score_method=\'csum\'', 'csum'),
            ('prescore_method=\'nsum\',score_method=\'nsum\'', 'nsum'),
            # Hack
            ('[qd]?exclude_reference=' + ut.regex_or(['True', 'False', 'None']) + '\,?', ''),
            ('=True', '=On'),
            ('=False', '=Off'),
        ]
        for ser, rep in repl_list:
            cfg_lbls = [re.sub(ser, rep, lbl) for lbl in cfg_lbls]

        # split configs up by param and annots
        pa_tups = [lbl.split('+') for lbl in cfg_lbls]
        cfg_lbls2 = []
        for pa in pa_tups:
            new_parts = []
            for part in pa:
                name, settings = part.split(':')
                if len(settings) == 0:
                    new_parts.append(name)
                else:
                    new_parts.append(part)
            if len(new_parts) == 2 and new_parts[1] == 'default':
                newlbl = new_parts[0]
            else:
                newlbl = '+'.join(new_parts)
            cfg_lbls2.append(newlbl)
        #cfgtups = [lbl.split(':') for lbl in cfg_lbls]
        #cfg_lbls = [':'.join(tup) if len(tup) != 2 else tup[1] if len(tup[1]) > 0 else 'BASELINE' for tup in cfgtups]
        cfg_lbls = cfg_lbls2

        #from ibeis.experiments import annotation_configs
        #lblaug = annotation_configs.compress_aidcfg(test_result.acfg)['common']['_cfgstr']

        #cfg_lbls = [lbl + ':' + lblaug for lbl in cfg_lbls]

        return cfg_lbls

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

    def get_all_varied_params(test_result):
        # only for big results
        varied_cfg_params = list(set(ut.flatten([cfgdict.keys() for cfgdict in test_result.varied_cfg_list])))
        varied_acfg_params = list(set(ut.flatten([acfg.keys() for acfg in test_result.varied_acfg_list])))
        varied_params = varied_acfg_params + varied_cfg_params
        return varied_params

    def get_total_num_varied_params(test_result):
        return len(test_result.get_all_varied_params())

    def get_param_basis(test_result, key):
        """
        Returns what a param was varied between over all tests
        key = 'K'
        key = 'dcfg_sample_size'
        """
        if key == 'len(daids)':
            basis = sorted(list(set([len(daids) for daids in test_result.cfgx2_daids])))
        elif any([key in cfgdict for cfgdict in test_result.varied_cfg_list]):
            basis = sorted(list(set([cfgdict[key] for cfgdict in test_result.varied_cfg_list])))
        elif any([key in cfgdict for cfgdict in test_result.varied_acfg_list]):
            basis = sorted(list(set([acfg[key] for acfg in test_result.varied_acfg_list])))
        else:
            assert False
        return basis

    def get_param_val_from_cfgx(test_result, cfgx, key):
        if key == 'len(daids)':
            return len(test_result.cfgx2_daids[cfgx])
        elif any([key in cfgdict for cfgdict in test_result.varied_cfg_list]):
            return test_result.varied_cfg_list[cfgx][key]
        elif any([key in cfgdict for cfgdict in test_result.varied_acfg_list]):
            return test_result.varied_acfg_list[cfgx][key]
        else:
            assert False

    def get_cfgx_with_param(test_result, key, val):
        """
        Gets configs where the given parameter is held constant
        """
        if key == 'len(daids)':
            cfgx_list = [cfgx for cfgx, daids in enumerate(test_result.cfgx2_daids) if len(daids) == val]
        elif any([key in cfgdict for cfgdict in test_result.varied_cfg_list]):
            cfgx_list = [cfgx for cfgx, cfgdict in enumerate(test_result.varied_cfg_list) if cfgdict[key] == val]
        elif any([key in cfgdict for cfgdict in test_result.varied_acfg_list]):
            cfgx_list = [cfgx for cfgx, acfg in enumerate(test_result.varied_acfg_list) if acfg[key] == val]
        else:
            assert False
        return cfgx_list

    def get_title_aug(test_result):
        ibs = test_result.ibs
        title_aug = ''
        title_aug += ' db=' + (ibs.get_dbname())
        try:
            if '_cfgname' in test_result.common_acfg['common']:
                title_aug += ' a=' + test_result.common_acfg['common']['_cfgname']
            else:
                title_aug += ' a=[' + ','.join([cfg['dcfg__cfgname'] for cfg in test_result.varied_acfg_list]) + ']'
            title_aug += ' t=' + test_result.common_cfgdict['_cfgname']
        except Exception as ex:
            print(ut.dict_str(test_result.common_acfg))
            print(ut.dict_str(test_result.common_cfgdict))
            ut.printex(ex)
            raise
        if test_result.has_constant_qaids():
            title_aug += ' #qaids=%r' % (len(test_result.qaids),)
        if test_result.has_constant_daids():
            daids = test_result.cfgx2_daids[0]
            title_aug += ' #daids=%r' % (len(test_result.cfgx2_daids[0]),)
            if test_result.has_constant_qaids():
                locals_ = ibs.get_annotconfig_stats(test_result.qaids, daids, verbose=False)[1]
                all_daid_per_name_stats = locals_['all_daid_per_name_stats']
                if all_daid_per_name_stats['std'] == 0:
                    title_aug += ' dper_name=%s' % (ut.scalar_str(all_daid_per_name_stats['mean'], precision=2),)
                else:
                    title_aug += ' dper_name=%s±%s' % (ut.scalar_str(all_daid_per_name_stats['mean'], precision=2), ut.scalar_str(all_daid_per_name_stats['std'], precision=2),)
        elif test_result.has_constant_length_daids():
            daids = test_result.cfgx2_daids[0]
            title_aug += ' #daids=%r' % (len(test_result.cfgx2_daids[0]),)

        return title_aug

    def get_fname_aug(test_result):
        import re
        title_aug = test_result.get_title_aug()
        valid_regex = '-a-zA-Z0-9_.() '
        valid_extra = '=,'
        valid_regex += valid_extra
        title_aug = title_aug.replace(' ', '_')  # spaces suck
        fname_aug = re.sub('[^' + valid_regex + ']+', '', title_aug)
        return fname_aug

    def print_unique_annot_config_stats(test_result, ibs=None):
        if ibs is None:
            ibs = test_result.ibs
        cfx2_dannot_hashid = [ibs.get_annot_hashid_visual_uuid(daids) for daids in test_result.cfgx2_daids]
        unique_daids = ut.list_compress(test_result.cfgx2_daids, ut.flag_unique_items(cfx2_dannot_hashid))
        print('+====')
        print('Printing %d unique annotconfig stats' % (len(unique_daids)))
        print('test_result.common_acfg = ' + ut.dict_str(test_result.common_acfg))
        print('param_basis(len(daids)) = %r' % (test_result.get_param_basis('len(daids)'),))
        for count, daids in enumerate(unique_daids):
            print('+---')
            print('count = %r/%r' % (count, len(unique_daids)))
            if test_result.has_constant_qaids():
                annotconfig_stats_strs, locals_ = ibs.get_annotconfig_stats(test_result.qaids, daids)
            print('L___')

    def print_results(test_result):
        r"""
        CommandLine:
            python -m ibeis.experiments.experiment_storage --exec-print_results

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.experiments.experiment_storage import *  # NOQA
            >>> from ibeis.experiments import experiment_harness
            >>> ibs, test_result = experiment_harness.testdata_expts('PZ_MTEST')
            >>> result = test_result.print_results()
            >>> print(result)
        """
        from ibeis.experiments import experiment_printres
        ibs = test_result.ibs
        experiment_printres.print_results(ibs, test_result)

    @ut.memoize
    def get_new_hard_qx_list(test_result):
        """ Mark any query as hard if it didnt get everything correct """
        rank_mat = test_result.get_rank_mat()
        is_new_hard_list = rank_mat.max(axis=1) > 0
        new_hard_qx_list = np.where(is_new_hard_list)[0]
        return new_hard_qx_list

    def get_common_qaids(test_result):
        if not test_result.has_constant_qaids():
            # Get only cases the tests share for now
            common_qaids = reduce(np.intersect1d, test_result.cfgx2_qaids)
            return common_qaids
        else:
            return test_result.qaids

    def get_case_positions(test_result, mode='failure', disagree_first=True, samplekw=None):
        """
        Helps get failure and sucess cases

        Args:
            pass

        Returns:
            list: new_hard_qx_list

        CommandLine:
            python -m ibeis.experiments.experiment_storage --exec-get_case_positions

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.experiments.experiment_storage import *  # NOQA
            >>> from ibeis.init import main_helpers
            >>> ibs, test_result = main_helpers.testdata_expts('PZ_MTEST', a=['uncontrolled'], t=['default:K=[1,2]'])
            >>> mode = 'failure'
            >>> new_hard_qx_list = test_result.get_case_positions(mode)
            >>> result = ('new_hard_qx_list = %s' % (str(new_hard_qx_list),))
            >>> print(result)
        """
        common_qaids = test_result.get_common_qaids()
        # look at scores of the best gt and gf
        gf_score_mat = test_result.get_infoprop_mat('qx2_gf_raw_score', common_qaids)
        gt_score_mat = test_result.get_infoprop_mat('qx2_gt_raw_score', common_qaids)
        #gf_score_mat[np.isnan(gf_score_mat)]
        #gt_score_mat[np.isnan(gf_score_mat)]
        # Nan gf scores are easier, Nan gt scores are harder
        gf_score_mat[np.isnan(gf_score_mat)] = 0
        gt_score_mat[np.isnan(gt_score_mat)] = -np.inf

        # Make a degree of hardness
        # TODO: come up with a better measure of hardness
        hardness_degree_mat = gf_score_mat - gt_score_mat

        if False:
            for cfgx in range(len(gt_score_mat.T)):
                encoder = vt.ScoreNormalizer()
                tp_scores = gt_score_mat.T[cfgx]
                tn_scores = gf_score_mat.T[cfgx]
                encoder.fit_partitioned(tp_scores, tn_scores, finite_only=True)
                encoder.visualize()

        qx_list, cfgx_list = np.unravel_index(hardness_degree_mat.ravel().argsort()[::-1], hardness_degree_mat.shape)
        case_pos_list = np.vstack((qx_list, cfgx_list)).T

        ONLY_FINITE = True
        if ONLY_FINITE:
            flags = np.isfinite(hardness_degree_mat[tuple(case_pos_list.T)])
            case_pos_list = case_pos_list.compress(flags, axis=0)

        # Get list sorted by the easiest hard cases, so we can fix the
        # non-pathological cases first
        if mode == 'failure':
            flags = hardness_degree_mat[tuple(case_pos_list.T)] > 0
            case_pos_list = case_pos_list.compress(flags, axis=0)
        elif mode == 'success':
            flags = hardness_degree_mat[tuple(case_pos_list.T)] < 0
            case_pos_list = case_pos_list.compress(flags, axis=0)
        else:
            raise NotImplementedError('Unknown mode')

        # Group by configuration
        #case_hardness = hardness_degree_mat[tuple(case_pos_list.T)]
        #case_gt_score = gt_score_mat[tuple(case_pos_list.T)]
        #case_gf_score = gf_score_mat[tuple(case_pos_list.T)]

        #hard_degree_list = hardness_degree_mat[tuple(case_pos_list.T)]
        #groupids, groupxs = vt.group_indices(case_pos_list.T[0])
        #groupid2_score = [
        #case_qx_list = ut.unique_keep_order2(case_pos_list.T[0])

        #talk about convoluted
        _qx2_casegroup = ut.group_items(case_pos_list, case_pos_list.T[0], sorted_=False)
        qx2_casegroup = ut.order_dict_by(_qx2_casegroup, ut.unique_keep_order2(case_pos_list.T[0]))
        grouppos_list = list(qx2_casegroup.values())
        grouppos_len_list = list(map(len, grouppos_list))
        _len2_groupedpos = ut.group_items(grouppos_list, grouppos_len_list, sorted_=False)
        if samplekw is not None:
            #samplekw_default = {
            #    'per_group': 10,
            #    #'min_intersecting_cfgs': 1,
            #}
            _len2_groupedpos_keys = list(_len2_groupedpos.keys())
            _len2_groupedpos_values = [
                groupedpos[::max(1, len(groupedpos) // samplekw['per_group'])]
                for groupedpos in six.itervalues(_len2_groupedpos)
            ]
            _len2_groupedpos = dict(zip(_len2_groupedpos_keys, _len2_groupedpos_values))
        len2_groupedpos = ut.map_dict_vals(np.vstack, _len2_groupedpos)

        #percentile_stratify
        #def percentile_stratified_sample(x, num, rng=np.random):
        #    hardness = hardness_degree_mat[tuple(x.T)]
        #    percentiles = np.percentile(hardness, [0, 25, 50, 75, 100])
        #    percentiles[-1] += 1
        #    groups = [x.compress(np.logical_and(low <= hardness, hardness < high), axis=0) for low, high in ut.iter_window(percentiles)]
        #    [ut.random_sample(group, num, rng=rng) for group in groups]

        #ut.print_dict(len2_groupedpos, nl=2)
        if disagree_first:
            unflat_pos_list = list(len2_groupedpos.values())
        else:
            unflat_pos_list = list(len2_groupedpos.values()[::-1])
        case_pos_list = np.vstack(unflat_pos_list)
        return case_pos_list

    def get_interesting_ranks(test_results):
        """ find the rows that vary greatest with the parameter settings """
        rank_mat = test_results.get_rank_mat()
        # Find rows which scored differently over the various configs FIXME: duplicated
        isdiff_flags = [not np.all(row == row[0]) for row in rank_mat]
        #diff_aids    = ut.list_compress(test_results.qaids, isdiff_flags)
        diff_rank    = rank_mat.compress(isdiff_flags, axis=0)
        diff_qxs     = np.where(isdiff_flags)[0]
        if False:
            rankcategory = np.log(diff_rank + 1)
        else:
            rankcategory = diff_rank.copy()
            rankcategory[diff_rank == 0]  = 0
            rankcategory[diff_rank > 0]   = 1
            rankcategory[diff_rank > 2]   = 2
            rankcategory[diff_rank > 5]   = 3
            rankcategory[diff_rank > 50]  = 4
            rankcategory[diff_rank > 100] = 5
        row_rankcategory_std = np.std(rankcategory, axis=1)
        row_rankcategory_mean = np.mean(rankcategory, axis=1)
        import vtool as vt
        row_sortx = vt.argsort_multiarray([row_rankcategory_std, row_rankcategory_mean], reverse=True)
        interesting_qx_list = diff_qxs.take(row_sortx).tolist()
        #print("INTERSETING MEASURE")
        #print(interesting_qx_list)
        #print(row_rankcategory_std)
        #print(ut.list_take(qaids, row_sortx))
        #print(diff_rank.take(row_sortx, axis=0))
        return interesting_qx_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.experiments.experiment_storage
        python -m ibeis.experiments.experiment_storage --allexamples
        python -m ibeis.experiments.experiment_storage --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
