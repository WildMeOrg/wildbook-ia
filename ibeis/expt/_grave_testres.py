    def get_case_positions(testres, mode='failure', disagree_first=True,
                           samplekw=None):
        """
        Helps get failure and success cases

        DEPRICATE

        Args:
            pass

        Returns:
            list: new_hard_qx_list

        CommandLine:
            python -m ibeis --tf TestResult.get_case_positions

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.expt.test_result import *  # NOQA
            >>> from ibeis.init import main_helpers
            >>> ibs, testres = main_helpers.testdata_expts('PZ_MTEST', a=['uncontrolled'], t=['default:K=[1,2]'])
            >>> mode = 'failure'
            >>> new_hard_qx_list = testres.get_case_positions(mode)
            >>> result = ('new_hard_qx_list = %s' % (str(new_hard_qx_list),))
            >>> print(result)
        """
        common_qaids = testres.get_common_qaids()
        # look at scores of the best gt and gf
        gf_score_mat = testres.get_infoprop_mat('qx2_gf_raw_score', common_qaids)
        gt_score_mat = testres.get_infoprop_mat('qx2_gt_raw_score', common_qaids)
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

        qx_list, cfgx_list = np.unravel_index(
            hardness_degree_mat.ravel().argsort()[::-1],
            hardness_degree_mat.shape)
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

        #talk about convoluted
        _qx2_casegroup = ut.group_items(case_pos_list, case_pos_list.T[0], sorted_=False)
        qx2_casegroup = ut.order_dict_by(
            _qx2_casegroup, ut.unique_ordered(case_pos_list.T[0]))
        grouppos_list = list(qx2_casegroup.values())
        grouppos_len_list = list(map(len, grouppos_list))
        _len2_groupedpos = ut.group_items(grouppos_list, grouppos_len_list, sorted_=False)
        if samplekw is not None:
            #samplekw_default = {
            #    'per_group': 10,
            #    #'min_intersecting_cfgs': 1,
            #}
            per_group = samplekw['per_group']
            if per_group is not None:
                _len2_groupedpos_keys = list(_len2_groupedpos.keys())
                _len2_groupedpos_values = [
                    groupedpos[::max(1, len(groupedpos) // per_group)]
                    for groupedpos in six.itervalues(_len2_groupedpos)
                ]
                _len2_groupedpos = dict(zip(_len2_groupedpos_keys, _len2_groupedpos_values))
        len2_groupedpos = ut.map_dict_vals(np.vstack, _len2_groupedpos)

        #ut.print_dict(len2_groupedpos, nl=2)
        if disagree_first:
            unflat_pos_list = list(len2_groupedpos.values())
        else:
            unflat_pos_list = list(len2_groupedpos.values()[::-1])
        case_pos_list = vt.safe_vstack(unflat_pos_list, (0, 2), np.int)
        return case_pos_list


def get_individual_result_sample(testres, filt_cfg=None, **kwargs):
    """
    The selected rows are the query annotation you are interested in viewing
    The selected cols are the parameter configuration you are interested in viewing

    Args:
        testres (TestResult):  test result object
        filt_cfg (dict): config dict

    Kwargs:
        all, hard, hard2, easy, interesting, hist

    Returns:
        tuple: (sel_rows, sel_cols, flat_case_labels)

    CommandLine:
        python -m ibeis --tf -get_individual_result_sample --db PZ_Master1 -a ctrl
        python -m ibeis --tf -get_individual_result_sample --db PZ_Master1 -a ctrl --filt :fail=True,min_gtrank=5,gtrank_lt=20


    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.expt.experiment_drawing import *  # NOQA
        >>> from ibeis.init import main_helpers
        >>> ibs, testres = main_helpers.testdata_expts('PZ_MTEST')
        >>> filt_cfg = {'fail': True, 'success': True, 'min_gtrank': 5, 'max_gtrank': 40}
        >>> sel_rows, sel_cols, flat_case_labels = get_individual_result_sample(testres, filt_cfg)
        >>> result = ('(sel_rows, sel_cols, flat_case_labels) = %s' % (str((sel_rows, sel_cols, flat_case_labels)),))
        >>> print(result)
    """
    #from ibeis.expt import cfghelpers
    #sample_cfgstr_list = ut.get_argval('--filt', type_=list, default=None)
    #from ibeis.expt import cfghelpers

    #if sample_cfgstr_list is None:
    print('filt_cfg = %r' % (filt_cfg,))
    if filt_cfg is None or isinstance(filt_cfg, list):
        # Hack to check if specified on command line
        #if not show_in_notebook:
        #    from ibeis.init import main_helpers
        #    filt_cfg = main_helpers.testdata_filtcfg(default=filt_cfg)
        #else:
            from ibeis.expt import cfghelpers
            if filt_cfg is None:
                filt_cfg = ['']
            filt_cfg = ut.flatten(cfghelpers.parse_cfgstr_list2(filt_cfg, strict=False))[0]

    cfg_list = testres.cfg_list
    #qaids = testres.qaids
    qaids = testres.get_common_qaids()

    view_all          = kwargs.get('all', ut.get_argflag(('--view-all', '--va')))
    view_hard         = kwargs.get('hard', ut.get_argflag(('--view-hard', '--vh')))
    view_hard2        = kwargs.get('hard2', ut.get_argflag(('--view-hard2', '--vh2')))
    view_easy         = kwargs.get('easy', ut.get_argflag(('--view-easy', '--vz')))
    view_interesting  = kwargs.get('interesting', ut.get_argflag(('--view-interesting', '--vn')))
    hist_sample       = kwargs.get('hist', ut.get_argflag(('--hs', '--hist-sample')))
    view_differ_cases = kwargs.get('differcases', ut.get_argflag(('--diff-cases', '--dc')))
    view_cases        = kwargs.get('cases', ut.get_argflag(('--view-cases', '--vc')))

    if ut.get_argval('--qaid', type_=str, default=None) is not None:
        # hack
        view_all = True

    #sel_cols = params.args.sel_cols  # FIXME
    #sel_rows = params.args.sel_rows  # FIXME
    #sel_cols = [] if sel_cols is None else sel_cols
    #sel_rows = [] if sel_rows is None else sel_rows
    sel_rows = []
    sel_cols = []
    flat_case_labels = None
    if ut.NOT_QUIET:
        print('remember to inspect with --show --sel-rows (-r) and --sel-cols (-c) ')
        print('other options:')
        print('   --vf - view figure dir')
        print('   --va - view all (--filt :)')
        print('   --vh - view hard (--filt :fail=True)')
        print('   --ve - view easy (--filt :success=True)')
        print('   --vn - view iNteresting')
        print('   --hs - hist sample')
        print(' --filt - result filtering config (new way to do this func)')
        print('   --gv, --guiview - gui result inspection')
    if len(sel_rows) > 0 and len(sel_cols) == 0:
        sel_cols = list(range(len(cfg_list)))
    if len(sel_cols) > 0 and len(sel_rows) == 0:
        sel_rows = list(range(len(qaids)))
    if view_all:
        sel_rows = list(range(len(qaids)))
        sel_cols = list(range(len(cfg_list)))
    if view_hard:
        new_hard_qx_list = testres.get_new_hard_qx_list()
        sel_rows.extend(np.array(new_hard_qx_list).tolist())
        sel_cols.extend(list(range(len(cfg_list))))
    # sample-cases

    def convert_case_pos_to_cfgx(case_pos_list, case_labels_list):
        # Convert to all cfgx format
        qx_list = ut.unique_ordered(np.array(case_pos_list).T[0])
        ut.dict_take(ut.group_items(case_pos_list, case_pos_list.T[0]), qx_list)
        if case_labels_list is not None:
            grouped_labels = ut.dict_take(
                ut.group_items(case_labels_list, case_pos_list.T[0]),
                qx_list)
            flat_case_labels = list(map(ut.unique_ordered, map(ut.flatten, grouped_labels)))
        else:
            flat_case_labels = None
        new_rows = np.array(qx_list).tolist()
        new_cols = list(range(len(cfg_list)))
        return new_rows, new_cols, flat_case_labels

    if view_differ_cases:
        # Cases that passed on config but failed another
        case_pos_list, case_labels_list = testres.case_type_sample(
            1, with_success=True, min_success_diff=1)
        new_rows, new_cols, flat_case_labels = convert_case_pos_to_cfgx(
            case_pos_list, case_labels_list)
        sel_rows.extend(new_rows)
        sel_cols.extend(new_cols)

    if view_cases:
        case_pos_list, case_labels_list = testres.case_type_sample(1, with_success=False)
        new_rows, new_cols, flat_case_labels = convert_case_pos_to_cfgx(
            case_pos_list, case_labels_list)
        sel_rows.extend(new_rows)
        sel_cols.extend(new_cols)

    if view_hard2:
        # TODO handle returning case_pos_list
        #samplekw = ut.argparse_dict(dict(per_group=5))
        samplekw = ut.argparse_dict(dict(per_group=None))
        case_pos_list = testres.get_case_positions(mode='failure', samplekw=samplekw)
        failure_qx_list = ut.unique_ordered(case_pos_list.T[0])
        sel_rows.extend(np.array(failure_qx_list).tolist())
        sel_cols.extend(list(range(len(cfg_list))))

    if view_easy:
        new_hard_qx_list = testres.get_new_hard_qx_list()
        new_easy_qx_list = np.setdiff1d(np.arange(len(qaids)), new_hard_qx_list).tolist()
        sel_rows.extend(new_easy_qx_list)
        sel_cols.extend(list(range(len(cfg_list))))
    if view_interesting:
        interesting_qx_list = testres.get_interesting_ranks()
        sel_rows.extend(interesting_qx_list)
        # TODO: grab the best scoring and most interesting configs
        if len(sel_cols) == 0:
            sel_cols.extend(list(range(len(cfg_list))))
    if hist_sample:
        # Careful if there is more than one config
        config_rand_bin_qxs = testres.get_rank_histogram_qx_sample(size=10)
        sel_rows = np.hstack(ut.flatten(config_rand_bin_qxs))
        # TODO: grab the best scoring and most interesting configs
        if len(sel_cols) == 0:
            sel_cols.extend(list(range(len(cfg_list))))

    if filt_cfg is not None:
        # NEW WAY OF SAMPLING
        verbose = kwargs.get('verbose', None)
        case_pos_list = testres.case_sample2(filt_cfg, verbose=verbose)
        new_rows, new_cols, flat_case_labels = convert_case_pos_to_cfgx(case_pos_list, None)
        sel_rows.extend(new_rows)
        sel_cols.extend(new_cols)
        pass

    sel_rows = ut.unique_ordered(sel_rows)
    sel_cols = ut.unique_ordered(sel_cols)
    sel_cols = list(sel_cols)
    sel_rows = list(sel_rows)

    sel_rowxs = ut.get_argval('-r', type_=list, default=None)
    sel_colxs = ut.get_argval('-c', type_=list, default=None)

    if sel_rowxs is not None:
        sel_rows = ut.take(sel_rows, sel_rowxs)
        print('sel_rows = %r' % (sel_rows,))

    if sel_colxs is not None:
        sel_cols = ut.take(sel_cols, sel_colxs)

    if ut.NOT_QUIET:
        print('Returning Case Selection')
        print('len(sel_rows) = %r/%r' % (len(sel_rows), len(qaids)))
        print('len(sel_cols) = %r/%r' % (len(sel_cols), len(cfg_list)))

    return sel_rows, sel_cols, flat_case_labels

def get_interesting_ranks(test_results):
    """ find the rows that vary greatest with the parameter settings """
    rank_mat = test_results.get_rank_mat()
    # Find rows which scored differently over the various configs FIXME: duplicated
    isdiff_flags = [not np.all(row == row[0]) for row in rank_mat]
    #diff_aids    = ut.compress(test_results.qaids, isdiff_flags)
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
    row_sortx = vt.argsort_multiarray(
        [row_rankcategory_std, row_rankcategory_mean], reverse=True)
    interesting_qx_list = diff_qxs.take(row_sortx).tolist()
    #print("INTERSETING MEASURE")
    #print(interesting_qx_list)
    #print(row_rankcategory_std)
    #print(ut.take(qaids, row_sortx))
    #print(diff_rank.take(row_sortx, axis=0))
    return interesting_qx_list


def case_type_sample(testres, num_per_group=1, with_success=True,
                     with_failure=True, min_success_diff=0):
    category_poses = testres.partition_case_types(min_success_diff=min_success_diff)
    # STRATIFIED SAMPLE OF CASES FROM GROUPS
    #mode = 'failure'
    rng = np.random.RandomState(0)
    ignore_keys = ['total_failure', 'total_success']
    #ignore_keys = []
    #sample_keys = []
    #sample_vals = []
    flat_sample_dict = ut.ddict(list)

    #num_per_group = 1
    modes = []
    if with_success:
        modes += ['success']
    if with_failure:
        modes += ['failure']

    for mode in modes:
        for truth in ['gt', 'gf']:
            type2_poses = category_poses[mode + '_' + truth]
            for key, posses in six.iteritems(type2_poses):
                if key not in ignore_keys:
                    if num_per_group is not None:
                        sample_posses = ut.random_sample(posses, num_per_group, rng=rng)
                    else:
                        sample_posses = posses

                    flat_sample_dict[mode + '_' + truth + '_' + key].append(sample_posses)

    #list(map(np.vstack, flat_sample_dict.values()))
    sample_keys = flat_sample_dict.keys()
    sample_vals = list(map(np.vstack, flat_sample_dict.values()))

    has_sample = np.array(list(map(len, sample_vals))) > 0
    has_sample_idx = np.nonzero(has_sample)[0]

    print('Unsampled categories = %s' % (
        ut.list_str(ut.compress(sample_keys, ~has_sample))))
    print('Sampled categories = %s' % (
        ut.list_str(ut.compress(sample_keys, has_sample))))

    sampled_type_list = ut.take(sample_keys, has_sample_idx)
    sampled_cases_list = ut.take(sample_vals, has_sample_idx)

    sampled_lbl_list = ut.flatten([[lbl] * len(cases)
                                   for lbl, cases in zip(sampled_type_list, sampled_cases_list)])
    if len(sampled_cases_list) == 0:
        return [], []
    sampled_case_list = np.vstack(sampled_cases_list)

    # Computes unique test cases and groups them with all case labels
    caseid_list = vt.compute_unique_data_ids(sampled_case_list)
    unique_case_ids = ut.unique_ordered(caseid_list)
    labels_list = ut.dict_take(ut.group_items(sampled_lbl_list, caseid_list), unique_case_ids)
    cases_list = np.vstack(ut.get_list_column(ut.dict_take(ut.group_items(sampled_case_list, caseid_list), unique_case_ids), 0))

    #sampled_case_list = np.vstack(ut.flatten(sample_vals))
    #sampled_case_list = sampled_case_list[vt.unique_row_indexes(case_pos_list)]
    case_pos_list = cases_list
    case_labels_list = labels_list
    #case_pos_list.shape
    #vt.unique_row_indexes(case_pos_list).shape
    return case_pos_list, case_labels_list


def partition_case_types(testres, min_success_diff=0):
    """
    Category Definitions
       * Potential nondistinct cases: (probably more a failure to match query keypoints)
           false negatives with rank < 5 with false positives  that have medium score
    """
    # TODO: Make this function divide the failure cases into several types
    # * scenery failure, photobomb failure, matching failure.
    # TODO: Make this function divide success cases into several types
    # * easy success, difficult success, incidental success

    # Matching labels from annotmatch rowid
    truth2_prop, prop2_mat = testres.get_truth2_prop()
    is_success = prop2_mat['is_success']
    is_failure = prop2_mat['is_failure']

    # Which queries differ in success
    min_success_ratio = min_success_diff / (testres.nConfig)
    #qx2_cfgdiffratio = np.array([np.sum(flags) / len(flags) for flags in is_success])
    #qx2_isvalid = np.logical_and((1 - qx2_cfgdiffratio) >= min_success_ratio, min_success_ratio <= min_success_ratio)
    qx2_cfgdiffratio = np.array([
        min(np.sum(flags), len(flags) - np.sum(flags)) / len(flags)
        for flags in is_success])
    qx2_isvalid = qx2_cfgdiffratio >= min_success_ratio
    #qx2_configs_differed = np.array([len(np.unique(flags)) > min_success_diff for flags in is_success])
    #qx2_isvalid = qx2_configs_differed

    ibs = testres.ibs
    type_getters = [
        ibs.get_annotmatch_is_photobomb,
        ibs.get_annotmatch_is_scenerymatch,
        ibs.get_annotmatch_is_hard,
        ibs.get_annotmatch_is_nondistinct,
    ]
    ignore_gt_flags = set(['nondistinct'])
    truth2_is_type = ut.ddict(ut.odict)
    for truth in ['gt', 'gf']:
        annotmatch_rowid_mat = truth2_prop[truth]['annotmatch_rowid']
        # Check which annotmatch rowids are None, they have not been labeled with matching type
        is_unreviewed = np.isnan(annotmatch_rowid_mat.astype(np.float))
        truth2_is_type[truth]['unreviewed'] = is_unreviewed
        for getter_method in type_getters:
            funcname = ut.get_funcname(getter_method)
            key = funcname.replace('get_annotmatch_is_', '')
            if not (truth == 'gt' and key in ignore_gt_flags):
                is_type = ut.accepts_numpy(getter_method.im_func)(
                    ibs, annotmatch_rowid_mat).astype(np.bool)
                truth2_is_type[truth][key] = is_type

    truth2_is_type['gt']['cfgxdiffers'] = np.tile(
        (qx2_cfgdiffratio > 0), (testres.nConfig, 1)).T
    truth2_is_type['gt']['cfgxsame']    = ~truth2_is_type['gt']['cfgxdiffers']

    # Make other category information
    gt_rank_ranges = [(5, 50), (50, None), (None, 5)]
    gt_rank_range_keys = []
    for low, high in gt_rank_ranges:
        if low is None:
            rank_range_key = 'rank_under_' + str(high)
            truth2_is_type['gt'][rank_range_key] = truth2_prop['gt']['rank'] < high
        elif high is None:
            rank_range_key = 'rank_above_' + str(low)
            truth2_is_type['gt'][rank_range_key] = truth2_prop['gt']['rank'] >= low
        else:
            rank_range_key = 'rank_between_' + str(low) + '_' + str(high)
            truth2_is_type['gt'][rank_range_key] = np.logical_and(
                truth2_prop['gt']['rank'] >= low,
                truth2_prop['gt']['rank'] < high)
        gt_rank_range_keys.append(rank_range_key)

    # Large timedelta ground false cases
    for truth in ['gt', 'gf']:
        truth2_is_type[truth]['large_timedelta'] = truth2_prop[truth]['timedelta'] > 60 * 60
        truth2_is_type[truth]['small_timedelta'] = truth2_prop[truth]['timedelta'] <= 60 * 60

    # Group the positions of the cases into the appropriate categories
    # Success always means that the groundtruth was rank 0
    category_poses = ut.odict()
    for truth in ['gt', 'gf']:
        success_poses = ut.odict()
        failure_poses = ut.odict()
        for key, is_type_ in truth2_is_type[truth].items():
            success_pos_flags = np.logical_and(is_type_, is_success)
            failure_pos_flags = np.logical_and(is_type_, is_failure)
            success_pos_flags = np.logical_and(success_pos_flags, qx2_isvalid[:, None])
            failure_pos_flags = np.logical_and(failure_pos_flags, qx2_isvalid[:, None])
            is_success_pos = np.vstack(np.nonzero(success_pos_flags)).T
            is_failure_pos = np.vstack(np.nonzero(failure_pos_flags)).T
            success_poses[key] = is_success_pos
            failure_poses[key] = is_failure_pos
        # Record totals
        success_poses['total_success'] = np.vstack(np.nonzero(is_success)).T
        failure_poses['total_failure'] = np.vstack(np.nonzero(is_failure)).T
        # Append to parent dict
        category_poses['success_' + truth] = success_poses
        category_poses['failure_' + truth] = failure_poses

    # Remove categories that dont matter
    for rank_range_key in gt_rank_range_keys:
        if not rank_range_key.startswith('rank_under'):
            assert len(category_poses['success_gt'][rank_range_key]) == 0, (
                'category_poses[\'success_gt\'][%s] = %r' % (
                    rank_range_key,
                    category_poses['success_gt'][rank_range_key],))
        del (category_poses['success_gt'][rank_range_key])

    # Convert to histogram
    #category_hists = ut.odict()
    #for key, pos_dict in category_poses.items():
        #category_hists[key] = ut.map_dict_vals(len, pos_dict)
    #ut.print_dict(category_hists)

    # Split up between different configurations
    if False:
        cfgx2_category_poses = ut.odict()
        for cfgx in range(testres.nConfig):
            cfg_category_poses = ut.odict()
            for key, pos_dict in category_poses.items():
                cfg_pos_dict = ut.odict()
                for type_, pos_list in pos_dict.items():
                    #if False:
                    #    _qx2_casegroup = ut.group_items(pos_list, pos_list.T[0], sorted_=False)
                    #    qx2_casegroup = ut.order_dict_by(_qx2_casegroup, ut.unique_ordered(pos_list.T[0]))
                    #    grouppos_list = list(qx2_casegroup.values())
                    #    grouppos_len_list = list(map(len, grouppos_list))
                    #    _len2_groupedpos = ut.group_items(grouppos_list, grouppos_len_list, sorted_=False)
                    cfg_pos_list = pos_list[pos_list.T[1] == cfgx]
                    cfg_pos_dict[type_] = cfg_pos_list
                cfg_category_poses[key] = cfg_pos_dict
            cfgx2_category_poses[cfgx] = cfg_category_poses
        cfgx2_category_hist = ut.hmap_vals(len, cfgx2_category_poses)
        ut.print_dict(cfgx2_category_hist)

    # Print histogram
    # Split up between different configurations
    category_hists = ut.hmap_vals(len, category_poses)
    if ut.NOT_QUIET:
        ut.print_dict(category_hists)

    return category_poses
    #return cfgx2_category_poses
    #% pylab qt4
    #X = gf_timedelta_list[is_failure]
    ##ut.get_stats(X, use_nan=True)
    #X = X[X < 60 * 60 * 24]
    #encoder = vt.ScoreNormalizerUnsupervised(X)
    #encoder.visualize()

    #X = gf_timedelta_list
    #X = X[X < 60 * 60 * 24]
    #encoder = vt.ScoreNormalizerUnsupervised(X)
    #encoder.visualize()

    #X = gt_timedelta_list
    #X = X[X < 60 * 60 * 24]
    #encoder = vt.ScoreNormalizerUnsupervised(X)
    #encoder.visualize()

    #for key, val in key2_gf_is_type.items():
    #    print(val.sum())
