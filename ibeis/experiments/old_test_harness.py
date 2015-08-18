

#-----------
#@utool.indent_func('[harn]')
@profile
def test_configurations(ibs, acfgstr_name_list, test_cfg_name_list):
    r"""
    Test harness driver function

    CommandLine:
        python -m ibeis.experiments.experiment_harness --exec-test_configurations --verbtd
        python -m ibeis.experiments.experiment_harness --exec-test_configurations --verbtd --draw-rank-cdf --show

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.experiments.experiment_harness import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> acfgstr_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=['candidacy:qsize=20,dper_name=1,dsize=10', 'candidacy:qsize=20,dper_name=10,dsize=100'])
        >>> test_cfg_name_list = ut.get_argval('-t', type_=list, default=['custom', 'custom:fg_on=False'])
        >>> test_configurations(ibs, acfgstr_name_list, test_cfg_name_list)
        >>> ut.show_if_requested()
    """

    test_result_list = run_test_configurations2(ibs, acfgstr_name_list, test_cfg_name_list)

    for test_result in test_result_list:
        if test_result is None:
            return
        else:
            experiment_printres.print_results(ibs, test_result)
            experiment_drawing.draw_results(ibs, test_result)
    return test_result_list



#def get_cmdline_test_result():
#    ibs, qaids, daids = main_helpers.testdata_ibeis(verbose=False)
#    test_cfg_name_list = ut.get_argval('-t', type_=list, default=['custom', 'custom:fg_on=False'])
#    test_result = run_test_configurations(ibs, qaids, daids, test_cfg_name_list)
#    return ibs, test_result
