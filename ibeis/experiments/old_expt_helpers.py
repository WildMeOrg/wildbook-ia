def get_testcfg_varydicts(test_cfg_name_list):
    """
    build varydicts from experiment_configs.
    recomputes test_cfg_name_list_out in case there are any nested lists specified in it

    CommandLine:
        python -m ibeis.experiments.experiment_helpers --test-get_testcfg_varydicts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> test_cfg_name_list = ['lnbnn2']
        >>> vary_dicts, test_cfg_name_list_out = get_testcfg_varydicts(test_cfg_name_list)
        >>> result = ut.list_str(vary_dicts)
        >>> print(result)
        [
            {'lnbnn_weight': [0.0], 'loglnbnn_weight': [0.0, 1.0], 'normonly_weight': [0.0], 'pipeline_root': ['vsmany'], 'sv_on': [True],},
            {'lnbnn_weight': [0.0], 'loglnbnn_weight': [0.0], 'normonly_weight': [0.0, 1.0], 'pipeline_root': ['vsmany'], 'sv_on': [True],},
            {'lnbnn_weight': [0.0, 1.0], 'loglnbnn_weight': [0.0], 'normonly_weight': [0.0], 'pipeline_root': ['vsmany'], 'sv_on': [True],},
        ]

        [
            {'sv_on': [True], 'logdist_weight': [0.0, 1.0], 'lnbnn_weight': [0.0], 'pipeline_root': ['vsmany'], 'normonly_weight': [0.0]},
            {'sv_on': [True], 'logdist_weight': [0.0], 'lnbnn_weight': [0.0], 'pipeline_root': ['vsmany'], 'normonly_weight': [0.0, 1.0]},
            {'sv_on': [True], 'logdist_weight': [0.0], 'lnbnn_weight': [0.0, 1.0], 'pipeline_root': ['vsmany'], 'normonly_weight': [0.0]},
        ]

    Ignore:
        print(ut.indent(ut.list_str(vary_dicts), ' ' * 8))
    """

    vary_dicts = []
    test_cfg_name_list_out = []
    for cfg_name in test_cfg_name_list:
        # Find if the name exists in the experiment configs
        test_cfg = experiment_configs.__dict__[cfg_name]
        # does that name correspond with a dict or list of dicts?
        if isinstance(test_cfg, dict):
            vary_dicts.append(test_cfg)
            test_cfg_name_list_out.append(cfg_name)
        elif isinstance(test_cfg, list):
            vary_dicts.extend(test_cfg)
            # make sure len(test_cfg_names) still corespond with len(vary_dicts)
            #test_cfg_name_list_out.extend([cfg_name + '_%d' % (count,) for count in range(len(test_cfg))])
            test_cfg_name_list_out.extend([cfg_name for count in range(len(test_cfg))])
    if len(vary_dicts) == 0:
        valid_cfg_names = experiment_configs.TEST_NAMES
        raise Exception('Choose a valid testcfg:\n' + valid_cfg_names)
    for dict_ in vary_dicts:
        for key, val in six.iteritems(dict_):
            assert not isinstance(val, six.string_types), 'val should be list not string: not %r' % (type(val),)
            #assert not isinstance(val, (list, tuple)), 'val should be list or tuple: not %r' % (type(val),)
    return vary_dicts, test_cfg_name_list_out


get varied_params_list_old():
        pass
        #vary_dicts, test_cfg_name_list_out = get_testcfg_varydicts(test_cfg_name_list)

        #dict_comb_list = [ut.all_dict_combinations(dict_)
        #                  for dict_ in vary_dicts]

        #unflat_param_lbls = [ut.all_dict_combinations_lbls(dict_, allow_lone_singles=True, remove_singles=False)
        #                     for dict_ in vary_dicts]

        #unflat_name_lbls = [[name_lbl for lbl in comb_lbls]
        #                    for name_lbl, comb_lbls in
        #                    zip(test_cfg_name_list_out, unflat_param_lbls)]

        #param_lbl_list     = ut.flatten(unflat_param_lbls)
        #name_lbl_list      = ut.flatten(unflat_name_lbls)

        #varied_param_lbls = [name + ':' + lbl for name, lbl in zip(name_lbl_list, param_lbl_list)]

