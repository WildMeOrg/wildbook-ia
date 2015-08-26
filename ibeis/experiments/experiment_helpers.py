# -*- coding: utf-8 -*-
"""
Helper module that helps expand parameters for grid search
TODO: move into custom pipe_cfg and annot_cfg modules
"""
from __future__ import absolute_import, division, print_function
import utool as ut  # NOQA
import six
import itertools
from ibeis.experiments import experiment_configs
from ibeis.experiments import cfghelpers
from ibeis.model import Config
from ibeis.init import filter_annots
print, print_, printDBG, rrr, profile = ut.inject(
    __name__, '[expt_helpers]', DEBUG=False)

QUIET = ut.QUIET


#---------------
# Big Test Cache
#-----------

def get_varied_params_list(test_cfg_name_list):
    """
    builds all combinations from dicts defined in experiment_configs


    CommandLine:
        python -m ibeis.experiments.experiment_helpers --test-get_varied_params_list:0
        python -m ibeis.experiments.experiment_helpers --test-get_varied_params_list:1

    Example:
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> test_cfg_name_list = ['lnbnn2']
        >>> test_cfg_name_list = ['candidacy_k']
        >>> test_cfg_name_list = ['candidacy_k', 'candidacy_k:fg_on=True']
        >>> cfgdict_list, varied_param_lbls = get_varied_params_list(test_cfg_name_list)
        >>> print(ut.list_str(cfgdict_list))
        >>> print(ut.list_str(varied_param_lbls))

    Example:
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> test_cfg_name_list = ['candidacy_baseline:fg_on=False']
        >>> cfgdict_list, cfg_lbl_list = get_varied_params_list(test_cfg_name_list)
        >>> print(ut.list_str(cfgdict_list))
        >>> print(ut.list_str(cfg_lbl_list))
    """
    # TODO: alias mumbojumbo and whatnot. Rectify duplicate code
    cfg_default_dict = dict(Config.QueryConfig().parse_items())
    valid_keys = list(cfg_default_dict.keys())
    cfgstr_list = test_cfg_name_list
    named_defaults_dict = ut.dict_subset(
        experiment_configs.__dict__, experiment_configs.TEST_NAMES)
    dict_comb_list = cfghelpers.parse_cfgstr_list2(
        cfgstr_list, named_defaults_dict, cfgtype=None, alias_keys=None,
        valid_keys=valid_keys)

    cfgdict_list = ut.flatten(dict_comb_list)

    cfglbl_list = cfghelpers.get_varied_cfg_lbls(cfgdict_list, cfg_default_dict)

    return cfgdict_list, cfglbl_list


def get_cfg_list_helper(test_cfg_name_list):
    """

    Example:
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> test_cfg_name_list = ['lnbnn2']
        >>> cfg_list, cfgx2_lbl, cfgdict_list = get_cfg_list_helper(test_cfg_name_list)
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
    _cfgdict_list, _cfglbl_list = get_varied_params_list(test_cfg_name_list)
    # Enforce rule that removes duplicate configs
    # by using feasiblity from ibeis.model.Config
    cfg_list = []
    cfgx2_lbl = []
    cfgdict_list = []
    # Add unique configs to the list
    cfg_set = set([])
    for _cfgdict, lbl in zip(_cfgdict_list, _cfglbl_list):
        # TODO: Move this unique finding code to its own function
        # and then move it up one function level so even the custom
        # configs can be uniquified
        #cfg = Config.QueryConfig(**dict_)
        cfg = Config.QueryConfig(**_cfgdict)
        if cfg not in cfg_set:
            cfg_set.add(cfg)
            cfgx2_lbl.append(lbl)
            cfg_list.append(cfg)
            cfgdict_list.append(_cfgdict)
    if not QUIET:
        print('[harn.help] return %d / %d unique configs' % (len(cfgdict_list), len(_cfgdict_list)))
    return cfg_list, cfgx2_lbl, cfgdict_list


def parse_acfg_combo_list(acfg_name_list):
    r"""
    Args:
        acfg_name_list (list):

    Returns:
        list: acfg_combo_list

    CommandLine:
        python -m ibeis.experiments.experiment_helpers --exec-parse_acfg_combo_list

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> import ibeis
        >>> from ibeis.experiments import annotation_configs
        >>> acfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=['default:qsize=10'])
        >>> acfg_combo_list = parse_acfg_combo_list(acfg_name_list)
        >>> acfg_list = ut.flatten(acfg_combo_list)
        >>> printkw = dict()
        >>> annotation_configs.print_acfg_list(acfg_list, **printkw)
    """
    from ibeis.experiments import annotation_configs
    named_defaults_dict = ut.dict_take(annotation_configs.__dict__, annotation_configs.TEST_NAMES)
    named_qcfg_defaults = dict(zip(annotation_configs.TEST_NAMES, ut.get_list_column(named_defaults_dict, 'qcfg')))
    named_dcfg_defaults = dict(zip(annotation_configs.TEST_NAMES, ut.get_list_column(named_defaults_dict, 'dcfg')))
    alias_keys = annotation_configs.ALIAS_KEYS
    # need to have the cfgstr_lists be the same for query and database so they can be combined properly for now
    qcfg_combo_list = cfghelpers.parse_cfgstr_list2(cfgstr_list=acfg_name_list,
                                                    named_defaults_dict=named_qcfg_defaults,
                                                    cfgtype='qcfg', alias_keys=alias_keys)
    dcfg_combo_list = cfghelpers.parse_cfgstr_list2(acfg_name_list, named_dcfg_defaults,
                                                    'dcfg', alias_keys=alias_keys)

    acfg_combo_list = []
    for qcfg_combo, dcfg_combo in zip(qcfg_combo_list, dcfg_combo_list):
        acfg_combo = [
            dict([('qcfg', qcfg), ('dcfg', dcfg)])
            for qcfg, dcfg in list(itertools.product(qcfg_combo, dcfg_combo))
        ]
        acfg_combo_list.append(acfg_combo)
    return acfg_combo_list


def get_annotcfg_list(ibs, acfg_name_list, filter_dups=True):
    r"""
    For now can only specify one acfg name list

    TODO: move to filter_annots

    Args:
        annot_cfg_name_list (list):

    CommandLine:
        python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list:0
        python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list:1
        python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list:2

        python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list:0 --db NNP_Master3 -a viewpoint_compare --nocache-aid --verbtd
        python -m ibeis.experiments.experiment_helpers --exec-get_annotcfg_list:0 --db PZ_ViewPoints -a viewpoint_compare --nocache-aid --verbtd

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.experiment_helpers import *  # NOQA
        >>> import ibeis
        >>> from ibeis.experiments import annotation_configs
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> filter_dups = not ut.get_argflag('--nofilter-dups')
        >>> #acfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=['default:qsize=10'])
        >>> acfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=['default'])
        >>> acfg_list, expanded_aids_list = get_annotcfg_list(ibs, acfg_name_list, filter_dups)
        >>> print('\n PRINTING TEST RESULTS')
        >>> result = ut.list_str(acfg_list, nl=3)
        >>> print('\n')
        >>> printkw = dict(combined=True, per_name_vpedge=None, per_qual=False, per_vp=False)
        >>> annotation_configs.print_acfg_list(acfg_list, expanded_aids_list, ibs, **printkw)
    """
    print('[harn.help] building acfg_list using %r' % (acfg_name_list,))
    from ibeis.experiments import annotation_configs
    acfg_combo_list = parse_acfg_combo_list(acfg_name_list)

    #acfg_slice = ut.get_argval('--acfg_slice', type_=slice, default=None)
    combo_slice = ut.get_argval('--combo_slice', type_=slice, default=slice(None))
    acfg_combo_list = [acfg_combo_[combo_slice] for acfg_combo_ in acfg_combo_list]

    #expanded_aids_list = [filter_annots.expand_acfgs(ibs, acfg) for acfg in acfg_list]
    expanded_aids_combo_list = [filter_annots.expand_acfgs_consistently(ibs, acfg_combo_) for acfg_combo_ in acfg_combo_list]
    expanded_aids_combo_flag_list = ut.flatten(expanded_aids_combo_list)
    acfg_list = ut.get_list_column(expanded_aids_combo_flag_list, 0)
    expanded_aids_list = ut.get_list_column(expanded_aids_combo_flag_list, 1)

    if filter_dups:
        acfg_list_ = []
        expanded_aids_list_ = []
        seen_ = ut.ddict(list)
        for acfg, (qaids, daids) in zip(acfg_list, expanded_aids_list):
            key = (ut.hashstr_arr27(qaids, 'qaids'), ut.hashstr_arr27(daids, 'daids'))
            if key in seen_:
                seen_[key].append(acfg)
                continue
            else:
                seen_[key].append(acfg)
                expanded_aids_list_.append((qaids, daids))
                acfg_list_.append(acfg)
        if ut.NOT_QUIET:
            duplicate_configs = dict([(key_, val_) for key_, val_ in seen_.items() if len(val_) > 1])
            if len(duplicate_configs) > 0:
                print('The following configs produced duplicate annnotation configs')
                for key, val in duplicate_configs.items():
                    nonvaried_compressed_dict, varied_compressed_dict_list = annotation_configs.compress_acfg_list_for_printing(val)
                    print('+--')
                    print('key = %r' % (key,))
                    print('varied_compressed_dict_list = %s' % (ut.list_str(varied_compressed_dict_list),))
                    print('nonvaried_compressed_dict = %s' % (ut.dict_str(nonvaried_compressed_dict),))
                    print('L__')

            print('[harn.help] return %d / %d unique annot configs' % (len(acfg_list_), len(acfg_list)))
        acfg_list = acfg_list_
        expanded_aids_list = expanded_aids_list_
    return acfg_list, expanded_aids_list


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
        >>> (cfg_list, cfgx2_lbl, cfgdict_list) = get_cfg_list_and_lbls(test_cfg_name_list, ibs)
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
        >>> (cfg_list, cfgx2_lbl, cfgdict_list) = get_cfg_list_and_lbls(test_cfg_name_list, ibs)
        >>> print('cfg_list = '+ ut.list_str(cfg_list))
        >>> print('cfgx2_lbl = '+ ut.list_str(cfgx2_lbl))

    """
    print('[harn.help] building cfg_list using: %s' % test_cfg_name_list)
    if isinstance(test_cfg_name_list, six.string_types):
        test_cfg_name_list = [test_cfg_name_list]
    cfg_list = []
    cfgx2_lbl = []
    cfgdict_list = []
    test_cfg_name_list2 = []
    for test_cfg_name in test_cfg_name_list:
        if test_cfg_name == 'custom':
            query_cfg = ibs.cfg.query_cfg.deepcopy()
            cfgdict = dict(query_cfg.parse_items())
            cfg_list.append(query_cfg)
            cfgx2_lbl.append(test_cfg_name)
            cfgdict_list.append(cfgdict)
            cfgdict['_cfgname'] = 'custom'
        elif test_cfg_name.startswith('custom:'):
            cfgstr_list = ':'.join(test_cfg_name.split(':')[1:]).split(',')
            # parse out modifications to custom
            cfgdict = ut.parse_cfgstr_list(cfgstr_list, smartcast=True)
            #ut.embed()
            query_cfg = ibs.cfg.query_cfg.deepcopy()
            query_cfg.update_query_cfg(**cfgdict)
            cfg_list.append(query_cfg)
            cfgx2_lbl.append(test_cfg_name)
            cfgdict['_cfgname'] = 'custom'
            cfgdict_list.append(cfgdict)
        else:
            test_cfg_name_list2.append(test_cfg_name)
    if len(test_cfg_name_list2) > 0:
        cfg_list2, cfgx2_lbl2, cfgdict_list2 = get_cfg_list_helper(test_cfg_name_list2)
        cfg_list.extend(cfg_list2)
        cfgx2_lbl.extend(cfgx2_lbl2)
        cfgdict_list.extend(cfgdict_list2)
    #cfgdict_list = [dict(cfg.parse_items()) for cfg in cfg_list]
    return (cfg_list, cfgx2_lbl, cfgdict_list)


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
