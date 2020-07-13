# -*- coding: utf-8 -*-
"""
Helper module that helps expand parameters for grid search
TODO: move into custom pipe_cfg and annot_cfg modules
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import sys
import six
import itertools
from wbia.expt import experiment_configs
from wbia.expt import cfghelpers
from wbia.algo import Config
from wbia.init import filter_annots

print, rrr, profile = ut.inject2(__name__)


def get_varied_pipecfg_lbls(cfgdict_list, pipecfg_list=None):
    if pipecfg_list is None:
        from wbia.algo import Config

        cfg_default_dict = dict(Config.QueryConfig().parse_items())
        cfgx2_lbl = ut.get_varied_cfg_lbls(cfgdict_list, cfg_default_dict)
    else:
        # TODO: group cfgdict by config type and then get varied labels
        cfg_default_dict = None
        cfgx2_lbl = ut.get_varied_cfg_lbls(cfgdict_list, cfg_default_dict)
    return cfgx2_lbl


def get_pipecfg_list(test_cfg_name_list, ibs=None, verbose=None):
    r"""
    Builds a list of varied query configurations. Only custom configs depend on
    an ibs object. The order of the output is not gaurenteed to aggree with
    input order.

    FIXME:
        This breaks if you proot=BC_DTW and ibs is None

    Args:
        test_cfg_name_list (list): list of strs
        ibs (wbia.IBEISController): wbia controller object (optional)

    Returns:
        tuple: (cfg_list, cfgx2_lbl) -
            cfg_list (list): list of config objects
            cfgx2_lbl (list): denotes which parameters are being varied.
                If there is just one config then nothing is varied

    CommandLine:
        python -m wbia get_pipecfg_list:0
        python -m wbia get_pipecfg_list:1 --db humpbacks
        python -m wbia get_pipecfg_list:2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.expt.experiment_helpers import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> #test_cfg_name_list = ['best', 'custom', 'custom:sv_on=False']
        >>> #test_cfg_name_list = ['default', 'default:sv_on=False', 'best']
        >>> test_cfg_name_list = ['default', 'default:sv_on=False', 'best']
        >>> # execute function
        >>> (pcfgdict_list, pipecfg_list) = get_pipecfg_list(test_cfg_name_list, ibs)
        >>> # verify results
        >>> assert pipecfg_list[0].sv_cfg.sv_on is True
        >>> assert pipecfg_list[1].sv_cfg.sv_on is False
        >>> pipecfg_lbls = get_varied_pipecfg_lbls(pcfgdict_list)
        >>> result = ('pipecfg_lbls = '+ ut.repr2(pipecfg_lbls))
        >>> print(result)
        pipecfg_lbls = ['default:', 'default:sv_on=False']

    Example1:
        >>> # DISABLE_DOCTEST
        >>> import ibeis_flukematch.plugin
        >>> from wbia.expt.experiment_helpers import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='humpbacks')
        >>> test_cfg_name_list = ['default:pipeline_root=BC_DTW,decision=average,crop_dim_size=[960,500]', 'default:K=[1,4]']
        >>> (pcfgdict_list, pipecfg_list) = get_pipecfg_list(test_cfg_name_list, ibs)
        >>> pipecfg_lbls = get_varied_pipecfg_lbls(pcfgdict_list)
        >>> result = ('pipecfg_lbls = '+ ut.repr2(pipecfg_lbls))
        >>> print(result)
        >>> print_pipe_configs(pcfgdict_list, pipecfg_list)
    """
    if verbose is None:
        verbose = ut.VERBOSE
    if ut.VERBOSE:
        print(
            '[expt_help.get_pipecfg_list] building pipecfg_list using: %s'
            % test_cfg_name_list
        )
    if isinstance(test_cfg_name_list, six.string_types):
        test_cfg_name_list = [test_cfg_name_list]
    _standard_cfg_names = []
    _pcfgdict_list = []

    # HACK: Parse out custom configs first
    for test_cfg_name in test_cfg_name_list:
        if test_cfg_name.startswith('custom:') or test_cfg_name == 'custom':
            print('[expthelpers] Parsing nonstandard custom config')
            assert False, 'custom is no longer supported'
            # if test_cfg_name.startswith('custom:'):
            #    # parse out modifications to custom
            #    cfgstr_list = ':'.join(test_cfg_name.split(':')[1:]).split(',')
            #    augcfgdict = ut.parse_cfgstr_list(cfgstr_list, smartcast=True)
            # else:
            #    augcfgdict = {}
            # # Take the configuration from the wbia object
            # pipe_cfg = ibs.--cfg.query_cfg.deepcopy()
            # # Update with augmented params
            # pipe_cfg.update_query_cfg(**augcfgdict)
            # # Parse out a standard cfgdict
            # cfgdict = dict(pipe_cfg.parse_items())
            # cfgdict['_cfgname'] = 'custom'
            # cfgdict['_cfgstr'] = test_cfg_name
            # _pcfgdict_list.append(cfgdict)
        else:
            _standard_cfg_names.append(test_cfg_name)
    # Handle stanndard configs next
    if len(_standard_cfg_names) > 0:
        # Get parsing information
        # cfg_default_dict = dict(Config.QueryConfig().parse_items())
        # valid_keys = list(cfg_default_dict.keys())
        cfgstr_list = _standard_cfg_names
        named_defaults_dict = ut.dict_subset(
            experiment_configs.__dict__, experiment_configs.TEST_NAMES
        )
        alias_keys = experiment_configs.ALIAS_KEYS
        # Parse standard pipeline cfgstrings
        metadata = {'ibs': ibs}
        dict_comb_list = cfghelpers.parse_cfgstr_list2(
            cfgstr_list,
            named_defaults_dict,
            cfgtype=None,
            alias_keys=alias_keys,
            # Hack out valid keys for humpbacks
            # valid_keys=valid_keys,
            strict=False,
            metadata=metadata,
        )
        # Get varied params (there may be duplicates)
        _pcfgdict_list.extend(ut.flatten(dict_comb_list))

    # Expand cfgdicts into PipelineConfig config objects
    # TODO: respsect different algorithm parameters like flukes
    if ibs is None:
        configclass_list = [Config.QueryConfig] * len(_pcfgdict_list)
    else:
        root_to_config = ibs.depc_annot.configclass_dict.copy()
        from wbia.algo.smk import smk_pipeline

        root_to_config['smk'] = smk_pipeline.SMKRequestConfig
        configclass_list = [
            root_to_config.get(
                _cfgdict.get('pipeline_root', _cfgdict.get('proot', 'vsmany')),
                Config.QueryConfig,
            )
            for _cfgdict in _pcfgdict_list
        ]
    _pipecfg_list = [
        cls(**_cfgdict) for cls, _cfgdict in zip(configclass_list, _pcfgdict_list)
    ]

    # Enforce rule that removes duplicate configs
    # by using feasiblity from wbia.algo.Config
    # TODO: Move this unique finding code to its own function
    # and then move it up one function level so even the custom
    # configs can be uniquified
    _flag_list = ut.flag_unique_items(_pipecfg_list)
    cfgdict_list = ut.compress(_pcfgdict_list, _flag_list)
    pipecfg_list = ut.compress(_pipecfg_list, _flag_list)
    if verbose:
        # for cfg in _pipecfg_list:
        #    print(cfg.get_cfgstr())
        #    print(cfg)
        print(
            '[harn.help] return %d / %d unique pipeline configs from: %r'
            % (len(cfgdict_list), len(_pcfgdict_list), test_cfg_name_list)
        )

    if ut.get_argflag(('--pcfginfo', '--pinfo', '--pipecfginfo')):
        ut.colorprint('Requested PcfgInfo for tests... ', 'red')
        print_pipe_configs(cfgdict_list, pipecfg_list)
        ut.colorprint('Finished Reporting PcfgInfo. Exiting', 'red')
        sys.exit(0)
    return (cfgdict_list, pipecfg_list)


def print_pipe_configs(cfgdict_list, pipecfg_list):
    pipecfg_lbls = get_varied_pipecfg_lbls(cfgdict_list, pipecfg_list)
    # pipecfg_lbls = pipecfg_list
    # assert len(pipecfg_lbls) == len(pipecfg_lbls), 'unequal lens'
    for pcfgx, (pipecfg, lbl) in enumerate(zip(pipecfg_list, pipecfg_lbls)):
        print('+--- %d / %d ===' % (pcfgx, (len(pipecfg_list))))
        ut.colorprint(lbl, 'white')
        print(pipecfg.get_cfgstr())
        print('L___')


def testdata_acfg_names(default_acfg_name_list=['default']):
    flags = ('--aidcfg', '--acfg', '-a')
    acfg_name_list = ut.get_argval(flags, type_=list, default=default_acfg_name_list)
    return acfg_name_list


def parse_acfg_combo_list(acfg_name_list):
    r"""
    Parses the name list into a list of config dicts

    Args:
        acfg_name_list (list): a list of annotation config strings

    Returns:
        list: acfg_combo_list

    CommandLine:
        python -m wbia parse_acfg_combo_list:0
        python -m wbia parse_acfg_combo_list:1
        python -m wbia parse_acfg_combo_list:2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.expt.experiment_helpers import *  # NOQA
        >>> import wbia
        >>> from wbia.expt import annotation_configs
        >>> acfg_name_list = testdata_acfg_names(['default', 'uncontrolled'])
        >>> acfg_combo_list = parse_acfg_combo_list(acfg_name_list)
        >>> acfg_list = ut.flatten(acfg_combo_list)
        >>> printkw = dict()
        >>> annotation_configs.print_acfg_list(acfg_list, **printkw)
        >>> result = ut.repr2(sorted(acfg_list[0].keys()))
        >>> print(result)
        ['dcfg', 'qcfg']

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.expt.experiment_helpers import *  # NOQA
        >>> import wbia
        >>> from wbia.expt import annotation_configs
        >>> # double colon :: means expand consistently and force const size
        >>> acfg_name_list = testdata_acfg_names(['unctrl', 'ctrl::unctrl'])
        >>> acfg_name_list = testdata_acfg_names(['unctrl', 'varysize', 'ctrl::unctrl'])
        >>> acfg_name_list = testdata_acfg_names(['unctrl', 'varysize', 'ctrl::varysize', 'ctrl::unctrl'])
        >>> acfg_combo_list = parse_acfg_combo_list(acfg_name_list)
        >>> acfg_list = ut.flatten(acfg_combo_list)
        >>> printkw = dict()
        >>> annotation_configs.print_acfg_list(acfg_list, **printkw)
    """
    from wbia.expt import annotation_configs

    named_defaults_dict = ut.dict_take(
        annotation_configs.__dict__, annotation_configs.TEST_NAMES
    )
    named_qcfg_defaults = dict(
        zip(
            annotation_configs.TEST_NAMES,
            ut.get_list_column(named_defaults_dict, 'qcfg'),
        )
    )
    named_dcfg_defaults = dict(
        zip(
            annotation_configs.TEST_NAMES,
            ut.get_list_column(named_defaults_dict, 'dcfg'),
        )
    )
    alias_keys = annotation_configs.ALIAS_KEYS
    # need to have the cfgstr_lists be the same for query and database so they
    # can be combined properly for now

    # Apply this flag to any case joined with ::
    special_join_dict = {'force_const_size': True}

    # Parse Query Annot Config
    nested_qcfg_combo_list = cfghelpers.parse_cfgstr_list2(
        cfgstr_list=acfg_name_list,
        named_defaults_dict=named_qcfg_defaults,
        cfgtype='qcfg',
        alias_keys=alias_keys,
        expand_nested=False,
        special_join_dict=special_join_dict,
        is_nestedcfgtype=True,
    )

    # Parse Data Annot Config
    nested_dcfg_combo_list = cfghelpers.parse_cfgstr_list2(
        cfgstr_list=acfg_name_list,
        named_defaults_dict=named_dcfg_defaults,
        cfgtype='dcfg',
        alias_keys=alias_keys,
        expand_nested=False,
        special_join_dict=special_join_dict,
        is_nestedcfgtype=True,
    )

    acfg_combo_list = []
    for nested_qcfg_combo, nested_dcfg_combo in zip(
        nested_qcfg_combo_list, nested_dcfg_combo_list
    ):
        acfg_combo = []
        # Only the inner nested combos are combinatorial
        for qcfg_combo, dcfg_combo in zip(nested_qcfg_combo, nested_dcfg_combo):
            _combo = [
                dict([('qcfg', qcfg), ('dcfg', dcfg)])
                for qcfg, dcfg in list(itertools.product(qcfg_combo, dcfg_combo))
            ]
            acfg_combo.extend(_combo)
        acfg_combo_list.append(acfg_combo)
    return acfg_combo_list


def filter_duplicate_acfgs(expanded_aids_list, acfg_list, acfg_name_list, verbose=None):
    """
    Removes configs with the same expanded aids list

    CommandLine:
        # The following will trigger this function:
        wbia -m wbia get_annotcfg_list:0 -a timectrl timectrl:view=left --db PZ_MTEST

    """
    from wbia.expt import annotation_configs

    if verbose is None:
        verbose = ut.VERBOSE
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
    if verbose:
        duplicate_configs = dict(
            [(key_, val_) for key_, val_ in seen_.items() if len(val_) > 1]
        )
        if len(duplicate_configs) > 0:
            print('The following configs produced duplicate annnotation configs')
            for key, val in duplicate_configs.items():
                # Print the difference between the duplicate configs
                _tup = annotation_configs.compress_acfg_list_for_printing(val)
                nonvaried_compressed_dict, varied_compressed_dict_list = _tup
                print('+--')
                print('key = %r' % (key,))
                print(
                    'duplicate_varied_cfgs = %s'
                    % (ut.repr2(varied_compressed_dict_list),)
                )
                print(
                    'duplicate_nonvaried_cfgs = %s'
                    % (ut.repr2(nonvaried_compressed_dict),)
                )
                print('L__')

        if verbose >= 1:
            print(
                '[harn.help] parsed %d / %d unique annot configs'
                % (len(acfg_list_), len(acfg_list),)
            )
        if verbose > 2:
            print('[harn.help] parsed from: %r' % (acfg_name_list,))
    return expanded_aids_list_, acfg_list_


def get_annotcfg_list(
    ibs,
    acfg_name_list,
    filter_dups=True,
    qaid_override=None,
    daid_override=None,
    initial_aids=None,
    use_cache=None,
    verbose=None,
):
    r"""
    For now can only specify one acfg name list

    TODO: move to filter_annots

    Args:
        annot_cfg_name_list (list):

    CommandLine:
        python -m wbia get_annotcfg_list:0
        python -m wbia get_annotcfg_list:1
        python -m wbia get_annotcfg_list:2

        wbia get_annotcfg_list:0 --ainfo
        wbia get_annotcfg_list:0 --db NNP_Master3 -a viewpoint_compare --nocache-aid --verbtd
        wbia get_annotcfg_list:0 --db PZ_ViewPoints -a viewpoint_compare --nocache-aid --verbtd
        wbia get_annotcfg_list:0 --db PZ_MTEST -a unctrl ctrl::unctrl --ainfo --nocache-aid
        wbia get_annotcfg_list:0 --db testdb1 -a : --ainfo --nocache-aid
        wbia get_annotcfg_list:0 --db Oxford -a :qhas_any=query --ainfo --nocache-aid
        wbia get_annotcfg_list:0 --db Oxford -a :qhas_any=query,dhas_any=distractor --ainfo --nocache-aid

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from wbia.expt.experiment_helpers import *  # NOQA
        >>> import wbia
        >>> from wbia.expt import annotation_configs
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> filter_dups = not ut.get_argflag('--nofilter-dups')
        >>> acfg_name_list = testdata_acfg_names()
        >>> _tup = get_annotcfg_list(ibs, acfg_name_list, filter_dups)
        >>> acfg_list, expanded_aids_list = _tup
        >>> print('\n PRINTING TEST RESULTS')
        >>> result = ut.repr2(acfg_list, nl=3)
        >>> print('\n')
        >>> #statskw = ut.parse_func_kwarg_keys(ibs.get_annot_stats_dict, with_vals=False)
        >>> printkw = dict(combined=True, per_name_vpedge=None,
        >>>                per_qual=False, per_vp=False, case_tag_hist=False)
        >>> annotation_configs.print_acfg_list(
        >>>     acfg_list, expanded_aids_list, ibs, **printkw)


    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.expt.experiment_helpers import *  # NOQA
        >>> import wbia
        >>> from wbia.init import main_helpers
        >>> from wbia.expt import annotation_configs
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> aids = ibs.get_valid_aids()
        >>> main_helpers.monkeypatch_encounters(ibs, aids, days=50)
        >>> a = ['default:crossval_enc=True,require_timestamp=True']
        >>> acfg_name_list = testdata_acfg_names(a)
        >>> acfg_list, expanded_aids_list = get_annotcfg_list(ibs, acfg_name_list)
        >>> annotation_configs.print_acfg_list(acfg_list, expanded_aids_list)
        >>> # Restore state
        >>> main_helpers.unmonkeypatch_encounters(ibs)
    """
    if ut.VERBOSE:
        print('[harn.help] building acfg_list using %r' % (acfg_name_list,))
    from wbia.expt import annotation_configs

    acfg_combo_list = parse_acfg_combo_list(acfg_name_list)

    # acfg_slice = ut.get_argval('--acfg_slice', type_=slice, default=None)
    # HACK: Sliceing happens before expansion (dependenceis get)
    combo_slice = ut.get_argval(
        '--combo_slice', type_='fuzzy_subset', default=slice(None)
    )
    acfg_combo_list = [
        ut.take(acfg_combo_, combo_slice) for acfg_combo_ in acfg_combo_list
    ]

    if ut.get_argflag('--consistent'):
        # Expand everything as one consistent annot list
        acfg_combo_list = [ut.flatten(acfg_combo_list)]

    # + --- Do Parsing ---
    expanded_aids_combo_list = [
        filter_annots.expand_acfgs_consistently(
            ibs,
            acfg_combo_,
            initial_aids=initial_aids,
            use_cache=use_cache,
            verbose=verbose,
            base=base,
        )
        for base, acfg_combo_ in enumerate(acfg_combo_list)
    ]
    expanded_aids_combo_flag_list = ut.flatten(expanded_aids_combo_list)
    acfg_list = ut.get_list_column(expanded_aids_combo_flag_list, 0)
    expanded_aids_list = ut.get_list_column(expanded_aids_combo_flag_list, 1)
    # L___

    # Slicing happens after expansion (but the labels get screwed up)
    acfg_slice = ut.get_argval('--acfg_slice', type_='fuzzy_subset', default=None)
    if acfg_slice is not None:
        acfg_list = ut.take(acfg_list, acfg_slice)
        expanded_aids_list = ut.take(expanded_aids_list, acfg_slice)

    # + --- Hack: Override qaids ---
    _qaids = ut.get_argval(
        ('--qaid', '--qaid-override'), type_=list, default=qaid_override
    )
    if _qaids is not None:
        expanded_aids_list = [(_qaids, daids) for qaids, daids in expanded_aids_list]
    # more hack for daids
    _daids = ut.get_argval(
        ('--daids-override', '--daid-override'), type_=list, default=daid_override
    )
    if _daids is not None:
        expanded_aids_list = [(qaids, _daids) for qaids, daids in expanded_aids_list]
    # L___

    if filter_dups:
        expanded_aids_list, acfg_list = filter_duplicate_acfgs(
            expanded_aids_list, acfg_list, acfg_name_list
        )

    if ut.get_argflag(
        ('--acfginfo', '--ainfo', '--aidcfginfo', '--print-acfg', '--printacfg')
    ):
        ut.colorprint('[experiment_helpers] Requested AcfgInfo ... ', 'red')
        print('combo_slice = %r' % (combo_slice,))
        print('acfg_slice = %r' % (acfg_slice,))
        annotation_configs.print_acfg_list(acfg_list, expanded_aids_list, ibs)
        ut.colorprint('[experiment_helpers] exiting due to AcfgInfo info request', 'red')
        sys.exit(0)

    return acfg_list, expanded_aids_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.expt.experiment_helpers
        python -m wbia.expt.experiment_helpers --allexamples
        python -m wbia.expt.experiment_helpers --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
