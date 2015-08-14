# -*- coding: utf-8 -*-
"""
Helper module that helps expand parameters for grid search
"""
from __future__ import absolute_import, division, print_function
import utool as ut  # NOQA
import six
from six.moves import zip, map
import re
import itertools
from ibeis.experiments import experiment_configs
from ibeis.model import Config
print, print_, printDBG, rrr, profile = ut.inject(
    __name__, '[expt_helpers]', DEBUG=False)

QUIET = ut.QUIET


def get_vary_dicts(test_cfg_name_list):
    """
    build varydicts from experiment_configs.
    recomputes test_cfg_name_list_out in case there are any nested lists specified in it

    CommandLine:
        python -m ibeis.experiments.experiment_helpers --test-get_vary_dicts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> test_cfg_name_list = ['lnbnn2']
        >>> vary_dicts, test_cfg_name_list_out = get_vary_dicts(test_cfg_name_list)
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
            test_cfg_name_list_out.extend([cfg_name + '_%d' % (count,) for count in range(len(test_cfg))])
    if len(vary_dicts) == 0:
        valid_cfg_names = experiment_configs.TEST_NAMES
        raise Exception('Choose a valid testcfg:\n' + valid_cfg_names)
    for dict_ in vary_dicts:
        for key, val in six.iteritems(dict_):
            assert not isinstance(val, six.string_types), 'val should be list not string: not %r' % (type(val),)
            #assert not isinstance(val, (list, tuple)), 'val should be list or tuple: not %r' % (type(val),)
    return vary_dicts, test_cfg_name_list_out


def rankscore_str(thresh, nLess, total, withlbl=True):
    #helper to print rank scores of configs
    percent = 100 * nLess / total
    fmtsf = '%' + str(ut.num2_sigfig(total)) + 'd'
    if withlbl:
        fmtstr = ':#ranks < %d = ' + fmtsf + '/%d = (%.1f%%) (err=' + fmtsf + ')'
        rankscore_str = fmtstr % (thresh, nLess, total, percent, (total - nLess))
    else:
        fmtstr = fmtsf + '/%d = (%.1f%%) (err=' + fmtsf + ')'
        rankscore_str = fmtstr % (nLess, total, percent, (total - nLess))
    return rankscore_str


def wrap_cfgstr(cfgstr):
    # REGEX to locate _XXXX(
    cfg_regex = r'_[A-Z][A-Z]*\('
    cfgstrmarker_list = re.findall(cfg_regex, cfgstr)
    cfgstrconfig_list = re.split(cfg_regex, cfgstr)
    args = [cfgstrconfig_list, cfgstrmarker_list]
    interleave_iter = ut.interleave(args)
    new_cfgstr_list = []
    total_len = 0
    prefix_str = ''
    # If unbalanced there is a prefix before a marker
    if len(cfgstrmarker_list) < len(cfgstrconfig_list):
        frag = interleave_iter.next()
        new_cfgstr_list += [frag]
        total_len = len(frag)
        prefix_str = ' ' * len(frag)
    # Iterate through markers and config strings
    while True:
        try:
            marker_str = interleave_iter.next()
            config_str = interleave_iter.next()
            frag = marker_str + config_str
        except StopIteration:
            break
        total_len += len(frag)
        new_cfgstr_list += [frag]
        # Go to newline if past 80 chars
        if total_len > 80:
            total_len = 0
            new_cfgstr_list += ['\n' + prefix_str]
    wrapped_cfgstr = ''.join(new_cfgstr_list)
    return wrapped_cfgstr


def format_cfgstr_list(cfgstr_list):
    indented_list = ut.indent_list('    ', cfgstr_list)
    wrapped_list = list(map(wrap_cfgstr, indented_list))
    return ut.joins('\n', wrapped_list)


#---------------
# Big Test Cache
#-----------

def get_varied_params_list(test_cfg_name_list):
    """
    builds all combinations from dicts defined in experiment_configs


    CommandLine:
        python -m ibeis.experiments.experiment_helpers --test-get_varied_params_list

    Example:
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> test_cfg_name_list = ['lnbnn2']
        >>> test_cfg_name_list = ['elph']
        >>> varied_params_list, varied_param_lbls = get_varied_params_list(test_cfg_name_list)
        >>> print(ut.list_str(varied_params_list))
        >>> print(ut.list_str(varied_param_lbls))

    Example:
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> test_cfg_name_list = ['candidacy']
        >>> varied_params_list, varied_param_lbls = get_varied_params_list(test_cfg_name_list)
        >>> print(ut.list_str(varied_params_list))
        >>> print(ut.list_str(varied_param_lbls))
    """
    vary_dicts, test_cfg_name_list_out = get_vary_dicts(test_cfg_name_list)
    dict_comb_list = [ut.all_dict_combinations(dict_)
                      for dict_ in vary_dicts]
    varied_params_list = ut.flatten(dict_comb_list)
    OLD_ = True
    if OLD_:
        dict_comb_lbls = [ut.all_dict_combinations_lbls(dict_, allow_lone_singles=True, remove_singles=False)
                          for dict_ in vary_dicts]
        # Append testname
        dict_comb_lbls = [[name_lbl + ':' + lbl for lbl in comb_lbls]
                          for name_lbl, comb_lbls in
                          zip(test_cfg_name_list_out, dict_comb_lbls)]
    #else:
    #    #varied_params_list =
    #    pass
    varied_param_lbls = ut.flatten(dict_comb_lbls)
    return varied_params_list, varied_param_lbls


def get_cfg_list_helper(test_cfg_name_list):
    """

    Example:
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> test_cfg_name_list = ['lnbnn2']
        >>> cfg_list, cfgx2_lbl = get_cfg_list_helper(test_cfg_name_list)
        >>> cfgstr_list = [cfg.get_cfgstr() for cfg in cfg_list]
        >>> print('\n'.join(cfgstr_list))
        _vsmany_NN(K4+1,last,cks1024)_FILT()_SV(50,0.01_2_1.57,csum)_AGG(csum)_FLANN(4_kdtrees)_FEAT(hesaff+sift,0_9001)_CHIP(sz450)
        _vsmany_NN(K4+1,last,cks1024)_FILT(logdist_1.0)_SV(50,0.01_2_1.57,csum)_AGG(csum)_FLANN(4_kdtrees)_FEAT(hesaff+sift,0_9001)_CHIP(sz450)
        _vsmany_NN(K4+1,last,cks1024)_FILT(normonly_1.0)_SV(50,0.01_2_1.57,csum)_AGG(csum)_FLANN(4_kdtrees)_FEAT(hesaff+sift,0_9001)_CHIP(sz450)
        _vsmany_NN(K4+1,last,cks1024)_FILT(lnbnn_1.0)_SV(50,0.01_2_1.57,csum)_AGG(csum)_FLANN(4_kdtrees)_FEAT(hesaff+sift,0_9001)_CHIP(sz450)

     Ignore:
        >>> for cfg in cfg_list:
        ...     print('____')
        ...     cfg.printme3()

    CommandLine:
        python dev.py --allgt -t lnbnn2 --db PZ_Mothers --noqcache

    """
    # Get varied params (there may be duplicates)
    varied_params_list, varied_param_lbls = get_varied_params_list(test_cfg_name_list)
    # Enforce rule that removes duplicate configs
    # by using feasiblity from ibeis.model.Config
    cfg_list = []
    cfgx2_lbl = []
    cfg_set = set([])
    # Add unique configs to the list
    for dict_, lbl in zip(varied_params_list, varied_param_lbls):
        # TODO: Move this unique finding code to its own function
        # and then move it up one function level so even the custom
        # configs can be uniquified
        #cfg = Config.QueryConfig(**dict_)
        #
        # FIXME: There are tuned query configs for specific species.
        # Maybe those should be hacked in here? maybe not though.
        cfg = Config.QueryConfig(**dict_)
        if cfg not in cfg_set:
            cfgx2_lbl.append(lbl)
            cfg_list.append(cfg)
            cfg_set.add(cfg)
    if not QUIET:
        print('[harn.help] return %d / %d unique configs' % (len(cfg_list), len(varied_params_list)))
    return cfg_list, cfgx2_lbl


def parse_cfgstr_list2(cfgstr_list, named_defaults_dict, cfgtype=None, alias_keys=None):
    """
    Parse a genetic cfgstr --flag name1:custom_args1 name2:custom_args2
    """
    cfg_combo_list = []
    for cfgstr in cfgstr_list:
        cfgstr_split = cfgstr.split(':')
        cfgname = cfgstr_split[0]
        cfg = named_defaults_dict[cfgname].copy()
        # Parse dict out of a string
        if len(cfgstr_split) > 1:
            cfgstr_options =  ':'.join(cfgstr_split[1:]).split(',')
            cfg_options = ut.parse_cfgstr_list(cfgstr_options, smartcast=True, oldmode=False)
        else:
            cfg_options = {}
        # Hack for q/d-prefix specific configs
        if cfgtype is not None:
            for key in list(cfg_options.keys()):
                # check if key is nonstandard
                if not (key in cfg or key in alias_keys):
                    # does removing prefix make it stanard?
                    prefix = cfgtype[0]
                    if key.startswith(prefix):
                        key_ = key[len(prefix):]
                        if key_ in cfg or key_ in alias_keys:
                            # remove prefix
                            cfg_options[key_] = cfg_options[key]
                    try:
                        assert key[1:] in cfg or key[1:] in alias_keys, 'key=%r, key[1:] =%r' % (key, key[1:] )
                    except AssertionError as ex:
                        ut.printex(ex, 'error', keys=['key', 'cfg', 'alias_keys'])
                        raise
                    del cfg_options[key]
        # Remap keynames based on aliases
        if alias_keys is not None:
            for key in alias_keys.keys():
                if key in cfg_options:
                    # use standard new key
                    cfg_options[alias_keys[key]] = cfg_options[key]
                    # remove old alised key
                    del cfg_options[key]
        # Finalize configuration dict
        cfg = ut.update_existing(cfg, cfg_options, copy=True, assert_exists=True)
        cfg['_cfgtype'] = cfgtype
        cfg['_cfgname'] = cfgname
        cfg['_cfgstr'] = cfgstr

        cfg_combo = ut.all_dict_combinations(cfg)
        cfg_combo_list.append(cfg_combo)
    return cfg_combo_list


def get_annotcfg_list(acfgstr_list):
    r"""
    For now can only specify one acfg name list

    Args:
        annot_cfg_name_list (list):

    CommandLine:
        python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list:0
        python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list:1

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> acfgstr_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=['default:qsize=10'])
        >>> acfg_list = get_annotcfg_list(acfgstr_list)
        >>> result = ut.list_str(acfg_list, nl=3)
        >>> print(result)

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> acfgstr_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=['default:qsize=10', 'varysize', 'candidacy'])
        >>> acfg_list = get_annotcfg_list(acfgstr_list)
        >>> result = ut.list_str(acfg_list)
        >>> print(result)
    """
    from ibeis.experiments import annotation_configs
    named_defaults_dict = ut.dict_take(annotation_configs.__dict__, annotation_configs.TEST_NAMES)
    named_qcfg_defaults = dict(zip(annotation_configs.TEST_NAMES, ut.get_list_column(named_defaults_dict, 'qcfg')))
    named_dcfg_defaults = dict(zip(annotation_configs.TEST_NAMES, ut.get_list_column(named_defaults_dict, 'dcfg')))
    alias_keys = annotation_configs.alias_keys
    # need to have the cfgstr_lists be the same for query and database so they can be combined properly for now
    qcfg_combo_list = parse_cfgstr_list2(acfgstr_list, named_qcfg_defaults, 'qcfg', alias_keys=alias_keys)
    dcfg_combo_list = parse_cfgstr_list2(acfgstr_list, named_dcfg_defaults, 'dcfg', alias_keys=alias_keys)

    acfg_combo_list = []
    for qcfg_combo, dcfg_combo in zip(qcfg_combo_list, dcfg_combo_list):
        acfg_combo = [dict([('qcfg', qcfg), ('dcfg', dcfg)]) for qcfg, dcfg in list(itertools.product(qcfg_combo, dcfg_combo))]
        acfg_combo_list.append(acfg_combo)
    acfg_list = ut.flatten(acfg_combo_list)

    return acfg_list


def get_cfg_list_and_lbls(test_cfg_name_list, ibs=None):
    r"""
    Driver function

    Returns a list of varied query configurations. Only custom configs depend on
    IBEIS. The order of the output is not gaurenteed to aggree with input order.

    Args:
        test_cfg_name_list (list):
        ibs (IBEISController):  ibeis controller object

    Returns:
        tuple: (cfg_list, cfgx2_lbl) -
            cfg_list (list): list of config objects
            cfgx2_lbl (list): denotes which parameters are being varied.
                If there is just one config then nothing is varied

    CommandLine:
        python -m ibeis.experiments.experiment_helpers --test-get_cfg_list_and_lbls
        python -m ibeis.experiments.experiment_helpers --test-get_cfg_list_and_lbls:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> test_cfg_name_list = ['best', 'custom', 'custom:sv_on=False']
        >>> # execute function
        >>> (cfg_list, cfgx2_lbl) = get_cfg_list_and_lbls(test_cfg_name_list, ibs)
        >>> # verify results
        >>> query_cfg0 = cfg_list[0]
        >>> query_cfg1 = cfg_list[1]
        >>> assert query_cfg0.sv_cfg.sv_on is True
        >>> assert query_cfg1.sv_cfg.sv_on is False
        >>> print('cfg_list = '+ ut.list_str(cfg_list))
        >>> print('cfgx2_lbl = '+ ut.list_str(cfgx2_lbl))

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> test_cfg_name_list = ['lnbnn2']
        >>> ibs = None
        >>> (cfg_list, cfgx2_lbl) = get_cfg_list_and_lbls(test_cfg_name_list, ibs)
        >>> print('cfg_list = '+ ut.list_str(cfg_list))
        >>> print('cfgx2_lbl = '+ ut.list_str(cfgx2_lbl))

    """
    print('[harn.help] building cfg_list using: %s' % test_cfg_name_list)
    if 'custom' == test_cfg_name_list:
        # Use the ibeis config as a custom config
        # this can be modified with the --cfg command line option
        # eg --cfg xy_thresh=.01 score_method=csum
        print('   * custom cfg_list')
        cfg_list = [ibs.cfg.query_cfg.deepcopy()]
        cfgx2_lbl = [test_cfg_name_list]
    #elif 'custom' in test_cfg_name_list:
    #    test_cfg_name_list.remove('custom')
    #    if len(test_cfg_name_list) > 0:
    #        cfg_list, cfgx2_lbl = get_cfg_list_helper(test_cfg_name_list)
    #    else:
    #        cfg_list, cfgx2_lbl = [], []
    #    cfg_list += [ibs.cfg.query_cfg.deepcopy()]
    #    cfgx2_lbl += ['custom()']
    #    test_cfg_name_list.append('custom')
    else:
        #cfg_list, cfgx2_lbl = get_cfg_list_helper(test_cfg_name_list)
        cfg_list = []
        cfgx2_lbl = []
        test_cfg_name_list2 = []
        for test_cfg_name in test_cfg_name_list:
            if test_cfg_name == 'custom':
                cfg_list.append(ibs.cfg.query_cfg.deepcopy())
                cfgx2_lbl.append(test_cfg_name)
            elif test_cfg_name.startswith('custom:'):
                cfgstr_list = ':'.join(test_cfg_name.split(':')[1:]).split(',')
                # parse out modifications to custom
                cfgdict = ut.parse_cfgstr_list(cfgstr_list, smartcast=True)
                #ut.embed()
                query_cfg = ibs.cfg.query_cfg.deepcopy()
                query_cfg.update_query_cfg(**cfgdict)
                cfg_list.append(query_cfg)
                cfgx2_lbl.append(test_cfg_name)
            else:
                test_cfg_name_list2.append(test_cfg_name)
        if len(test_cfg_name_list2) > 0:
            cfg_list2, cfgx2_lbl2 = get_cfg_list_helper(test_cfg_name_list2)
            cfg_list.extend(cfg_list2)
            cfgx2_lbl.extend(cfgx2_lbl2)
    return (cfg_list, cfgx2_lbl)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.experiments.experiment_helpers
        python -m ibeis.experiments.experiment_helpers --allexamples
        python -m ibeis.experiments.experiment_helpers --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
