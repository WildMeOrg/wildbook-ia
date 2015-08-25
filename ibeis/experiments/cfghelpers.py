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


def customize_base_cfg(cfgname, cfgstr_options, base_cfg, cfgtype, alias_keys=None, valid_keys=None):
    """
    cfgstr_options = 'dsize=1000,per_name=[1,2]'
    """
    cfg = base_cfg.copy()
    # Parse dict out of a string
    cfgstr_options_list = re.split(r',\s*' + ut.negative_lookahead(r'[^\[\]]*\]'), cfgstr_options)
    #cfgstr_options_list = cfgstr_options.split(',')
    cfg_options = ut.parse_cfgstr_list(cfgstr_options_list, smartcast=True, oldmode=False)
    # Hack for q/d-prefix specific configs
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
                    assert key[1:] in cfg or key[1:] in alias_keys, 'key=%r, key[1:] =%r' % (key, key[1:] )
                except AssertionError as ex:
                    ut.printex(ex, 'Parse Error Customize Cfg Base ', keys=['key', 'cfg', 'alias_keys', 'cfgstr_options', 'cfgtype'])
                    raise
                del cfg_options[key]
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
