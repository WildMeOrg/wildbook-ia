# -*- coding: utf-8 -*-
"""
Helper module that helps expand parameters for grid search
"""
from __future__ import absolute_import, division, print_function
import utool as ut  # NOQA
from six.moves import zip, map  # NOQA
import re
print, rrr, profile = ut.inject2(__name__, '[cfghelpers]')


INTERNAL_CFGKEYS = ['_cfgstr', '_cfgname', '_cfgtype', '_cfgindex']


def remove_prefix_hack(cfg, cfgtype, cfg_options, alias_keys):
    if cfgtype is not None and cfgtype in ['qcfg', 'dcfg']:
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
                    assert key[1:] in cfg or key[1:] in alias_keys, (
                        'key=%r, key[1:] =%r' % (key, key[1:] ))
                except AssertionError as ex:
                    ut.printex(ex, 'Parse Error Customize Cfg Base ',
                               keys=['key', 'cfg', 'alias_keys',
                                     'cfgstr_options', 'cfgtype'])
                    raise
                del cfg_options[key]


def get_varied_cfg_lbls(cfg_list, default_cfg=None):
    r"""
    Args:
        cfg_list (list):
        default_cfg (None): (default = None)

    Returns:
        list: cfglbl_list

    CommandLine:
        python -m ibeis.experiments.cfghelpers --exec-get_varied_cfg_lbls

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.cfghelpers import *  # NOQA
        >>> cfg_list = [{'_cfgname': 'test', 'f': 1, 'b': 1},
        >>>             {'_cfgname': 'test', 'f': 2, 'b': 1},
        >>>             {'_cfgname': 'test', 'f': 3, 'b': 1, 'z': 4}]
        >>> default_cfg = None
        >>> cfglbl_list = get_varied_cfg_lbls(cfg_list, default_cfg)
        >>> result = ('cfglbl_list = %s' % (str(cfglbl_list),))
        >>> print(result)
        cfglbl_list = ['test:f=1', 'test:f=2', 'test:f=3,z=4']
    """
    cfgname_list = [cfg['_cfgname'] for cfg in cfg_list]
    nonvaried_cfg, varied_cfg_list = partition_varied_cfg_list(cfg_list, default_cfg)
    cfglbl_list = [
        get_cfg_lbl(cfg, name)
        for cfg, name in zip(varied_cfg_list, cfgname_list)]
    return cfglbl_list


def partition_varied_cfg_list(cfg_list, default_cfg=None):
    r"""

    CommandLine:
        python -m ibeis.experiments.cfghelpers --exec-partition_varied_cfg_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.annotation_configs import *  # NOQA
        >>> cfg_list = [{'f': 1, 'b': 1}, {'f': 2, 'b': 1}, {'f': 3, 'b': 1, 'z': 4}]
        >>> nonvaried_cfg, varied_cfg_list = partition_varied_cfg_list(cfg_list)
        >>> result = ut.list_str((nonvaried_cfg, varied_cfg_list), label_list=['nonvaried_cfg', 'varied_cfg_list'])
        >>> print(result)
        nonvaried_cfg = {'b': 1}
        varied_cfg_list = [{'f': 1}, {'f': 2}, {'f': 3, 'z': 4}]
    """
    if default_cfg is None:
        nonvaried_cfg = reduce(ut.dict_intersection, cfg_list)
    else:
        nonvaried_cfg = reduce(ut.dict_intersection, [default_cfg] + cfg_list)
    nonvaried_keys = list(nonvaried_cfg.keys())
    varied_cfg_list = [
        ut.delete_dict_keys(cfg.copy(), nonvaried_keys)
        for cfg in cfg_list]
    return nonvaried_cfg, varied_cfg_list


def get_cfg_lbl(cfg, name=None, nonlbl_keys=INTERNAL_CFGKEYS):
    """
    Formats a flat configuration dict into a short string label

    Args:
        cfg (dict):
        name (str): (default = None)
        nonlbl_keys (list): (default = INTERNAL_CFGKEYS)

    Returns:
        str: cfg_lbl

    CommandLine:
        python -m ibeis.experiments.cfghelpers --exec-get_cfg_lbl_list

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.cfghelpers import *  # NOQA
        >>> cfg = {'_cfgname': 'test', 'var1': 'val1', 'var2': 'val2'}
        >>> name = None
        >>> nonlbl_keys = ['_cfgstr', '_cfgname', '_cfgtype', '_cfgindex']
        >>> cfg_lbl = get_cfg_lbl_list(cfg, name, nonlbl_keys)
        >>> result = ('cfg_lbl = %s' % (str(cfg_lbl),))
        >>> print(result)
        cfg_lbl = test:var1=val1,var2=val2
    """
    if name is None:
        name = cfg['_cfgname']
    _search = ['dict(', ')', ' ']
    _repl = [''] * len(_search)
    # remove keys that should not belong to the label
    _clean_cfg = ut.delete_keys(cfg.copy(), nonlbl_keys)
    _lbl = ut.dict_str(_clean_cfg, explicit=True, nl=False, strvals=True)
    _lbl = ut.multi_replace(_lbl, _search, _repl).rstrip(',')
    cfg_lbl = name + ':' + _lbl
    return cfg_lbl


def customize_base_cfg(cfgname, cfgstr_options, base_cfg, cfgtype,
                       alias_keys=None, valid_keys=None):
    """
    cfgstr_options = 'dsize=1000,per_name=[1,2]'
    """
    cfg = base_cfg.copy()
    # Parse dict out of a string
    cfgstr_options_list = re.split(
        r',\s*' + ut.negative_lookahead(r'[^\[\]]*\]'), cfgstr_options)
    #cfgstr_options_list = cfgstr_options.split(',')
    cfg_options = ut.parse_cfgstr_list(
        cfgstr_options_list, smartcast=True, oldmode=False)
    # Hack for q/d-prefix specific configs
    remove_prefix_hack(cfg, cfgtype, cfg_options, alias_keys)
    # Remap keynames based on aliases
    if alias_keys is not None:
        for key in set(alias_keys.keys()):
            if key in cfg_options:
                # use standard new key
                cfg_options[alias_keys[key]] = cfg_options[key]
                # remove old alised key
                del cfg_options[key]
    # Ensure that nothing bad is being updated
    if valid_keys is not None:
        ut.assert_all_in(cfg_options.keys(), valid_keys, 'keys specified not in valid set')
    else:
        ut.assert_all_in(cfg_options.keys(), cfg.keys(), 'keys specified not in default options')
    # Finalize configuration dict
    #cfg = ut.update_existing(cfg, cfg_options, copy=True, assert_exists=False)
    cfg.update(cfg_options)
    cfg['_cfgtype'] = cfgtype
    cfg['_cfgname'] = cfgname
    cfg_combo = ut.all_dict_combinations(cfg)
    #if len(cfg_combo) > 1:
    for combox, cfg_ in enumerate(cfg_combo):
        #cfg_['_cfgname'] += ';' + str(combox)
        cfg_['_cfgindex'] = combox
    for cfg_ in cfg_combo:
        if len(cfgstr_options) > 0:
            cfg_['_cfgstr'] = cfg_['_cfgname'] + ':' + cfgstr_options
        else:
            cfg_['_cfgstr'] = cfg_['_cfgname']
    return cfg_combo


def parse_cfgstr_list2(cfgstr_list, named_defaults_dict, cfgtype=None, alias_keys=None, valid_keys=None):
    """
    Parse a genetic cfgstr --flag name1:custom_args1 name2:custom_args2
    """
    #OLD = True
    OLD = False
    cfg_combos_list = []
    for cfgstr in cfgstr_list:
        cfgstr_split = cfgstr.split(':')
        cfgname = cfgstr_split[0]
        base_cfg_list = named_defaults_dict[cfgname]
        if not isinstance(base_cfg_list, list):
            base_cfg_list = [base_cfg_list]
        cfg_combos = []
        for base_cfg in base_cfg_list:
            if not OLD:
                cfgstr_options =  ':'.join(cfgstr_split[1:])
                try:
                    cfg_combo = customize_base_cfg(cfgname, cfgstr_options, base_cfg, cfgtype, alias_keys=alias_keys, valid_keys=valid_keys)
                except Exception as ex:
                    ut.printex(ex, 'Parse Error CfgstrList2', keys=['cfgname', 'cfgstr_options', 'base_cfg', 'cfgtype', 'alias_keys', 'valid_keys'])
                    raise
            else:
                pass
                #cfg = base_cfg.copy()
                ## Parse dict out of a string
                #if len(cfgstr_split) > 1:
                #    cfgstr_options =  ':'.join(cfgstr_split[1:]).split(',')
                #    cfg_options = ut.parse_cfgstr_list(cfgstr_options, smartcast=True, oldmode=False)
                #else:
                #    cfgstr_options = ''
                #    cfg_options = {}
                ## Hack for q/d-prefix specific configs
                #if cfgtype is not None:
                #    for key in list(cfg_options.keys()):
                #        # check if key is nonstandard
                #        if not (key in cfg or key in alias_keys):
                #            # does removing prefix make it stanard?
                #            prefix = cfgtype[0]
                #            if key.startswith(prefix):
                #                key_ = key[len(prefix):]
                #                if key_ in cfg or key_ in alias_keys:
                #                    # remove prefix
                #                    cfg_options[key_] = cfg_options[key]
                #            try:
                #                assert key[1:] in cfg or key[1:] in alias_keys, 'key=%r, key[1:] =%r' % (key, key[1:] )
                #            except AssertionError as ex:
                #                ut.printex(ex, 'error', keys=['key', 'cfg', 'alias_keys'])
                #                raise
                #            del cfg_options[key]
                ## Remap keynames based on aliases
                #if alias_keys is not None:
                #    for key in set(alias_keys.keys()):
                #        if key in cfg_options:
                #            # use standard new key
                #            cfg_options[alias_keys[key]] = cfg_options[key]
                #            # remove old alised key
                #            del cfg_options[key]
                ## Ensure that nothing bad is being updated
                #if valid_keys is not None:
                #    ut.assert_all_in(cfg_options.keys(), valid_keys, 'keys specified not in valid set')
                #else:
                #    ut.assert_all_in(cfg_options.keys(), cfg.keys(), 'keys specified not in default options')
                ## Finalize configuration dict
                ##cfg = ut.update_existing(cfg, cfg_options, copy=True, assert_exists=False)
                #cfg.update(cfg_options)
                #cfg['_cfgtype'] = cfgtype
                #cfg['_cfgname'] = cfgname
                #cfg['_cfgstr'] = cfgstr
                #cfg_combo = ut.all_dict_combinations(cfg)
            cfg_combos.extend(cfg_combo)
        cfg_combos_list.append(cfg_combos)
    return cfg_combos_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.experiments.cfghelpers
        python -m ibeis.experiments.cfghelpers --allexamples
        python -m ibeis.experiments.cfghelpers --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
