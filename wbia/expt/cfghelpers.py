# -*- coding: utf-8 -*-
"""
Helper module that helps expand parameters for grid search

DEPRICATE: Most of this can likely be replaced by util_gridsearch
TODO: rectify with versions in util_gridsearch

It turns out a lot of the commandlines made possible here can be generatd by
using bash brace expansion.
http://www.linuxjournal.com/content/bash-brace-expansion
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut

print, rrr, profile = ut.inject2(__name__)


def remove_prefix_hack(cfg, cfgtype, cfg_options, alias_keys):
    if cfgtype is not None and cfgtype in ['qcfg', 'dcfg']:
        for key in list(cfg_options.keys()):
            # check if key is nonstandard
            if not (key in cfg or key in alias_keys):
                # does removing prefix make it stanard?
                prefix = cfgtype[0]
                if key.startswith(prefix):
                    key_ = key[len(prefix) :]
                    if key_ in cfg or key_ in alias_keys:
                        # remove prefix
                        cfg_options[key_] = cfg_options[key]
                try:
                    assert (
                        key[1:] in cfg or key[1:] in alias_keys
                    ), 'key=%r, key[1:] =%r' % (key, key[1:])
                except AssertionError as ex:
                    ut.printex(
                        ex,
                        'Parse Error Customize Cfg Base ',
                        keys=['key', 'cfg', 'alias_keys', 'cfgstr_options', 'cfgtype'],
                    )
                    raise
                del cfg_options[key]


@ut.on_exception_report_input(
    keys=['cfgname', 'cfgopt_strs', 'base_cfg', 'cfgtype', 'alias_keys', 'valid_keys'],
    force=True,
)
def customize_base_cfg(
    cfgname,
    cfgopt_strs,
    base_cfg,
    cfgtype,
    alias_keys=None,
    valid_keys=None,
    offset=0,
    strict=True,
):
    """
    DEPRICATE
    """
    import re

    cfg = base_cfg.copy()
    # Parse dict out of a string
    # ANYTHING_NOT_BRACE = r'[^\[\]]*\]'
    ANYTHING_NOT_PAREN_OR_BRACE = r'[^()\[\]]*[\]\)]'
    cfgstr_options_list = re.split(
        r',\s*' + ut.negative_lookahead(ANYTHING_NOT_PAREN_OR_BRACE), cfgopt_strs
    )
    # cfgstr_options_list = cfgopt_strs.split(',')
    cfg_options = ut.parse_cfgstr_list(
        cfgstr_list=cfgstr_options_list, smartcast=True, oldmode=False
    )
    # Hack for q/d-prefix specific configs
    remove_prefix_hack(cfg, cfgtype, cfg_options, alias_keys)
    # Remap keynames based on aliases
    if alias_keys is not None:
        # Use new standard keys and remove old aliased keys
        for key in set(alias_keys.keys()):
            if key in cfg_options:
                cfg_options[alias_keys[key]] = cfg_options[key]
                del cfg_options[key]
    # Ensure that nothing bad is being updated
    if strict:
        parsed_keys = cfg_options.keys()
        if valid_keys is not None:
            ut.assert_all_in(parsed_keys, valid_keys, 'keys specified not in valid set')
        else:
            ut.assert_all_in(
                parsed_keys, cfg.keys(), 'keys specified not in default options'
            )
    # Finalize configuration dict
    cfg.update(cfg_options)
    cfg['_cfgtype'] = cfgtype
    cfg['_cfgname'] = cfgname
    cfg_combo = ut.all_dict_combinations(cfg)
    # if len(cfg_combo) > 1:
    for combox, cfg_ in enumerate(cfg_combo, start=offset):
        # cfg_['_cfgname'] += ';' + str(combox)
        cfg_['_cfgindex'] = combox
    for cfg_ in cfg_combo:
        if len(cfgopt_strs) > 0:
            cfg_['_cfgstr'] = cfg_['_cfgname'] + ut.NAMEVARSEP + cfgopt_strs
        else:
            cfg_['_cfgstr'] = cfg_['_cfgname']
    return cfg_combo


def parse_cfgstr_list2(
    cfgstr_list,
    named_defaults_dict=None,
    cfgtype=None,
    alias_keys=None,
    valid_keys=None,
    expand_nested=True,
    strict=True,
    special_join_dict=None,
    is_nestedcfgtype=False,
    metadata=None,
):
    r"""
    Parses config strings. By looking up name in a dict of configs

    DEPRICATE

    Args:
        cfgstr_list (list):
        named_defaults_dict (dict): (default = None)
        cfgtype (None): (default = None)
        alias_keys (None): (default = None)
        valid_keys (None): (default = None)
        expand_nested (bool): (default = True)
        strict (bool): (default = True)
        is_nestedcfgtype - used for annot configs so special joins arent geometrically combined

    Note:
        Normal Case:
            --flag name

        Custom Arugment Cases:
            --flag name:custom_key1=custom_val1,custom_key2=custom_val2

        Multiple Config Case:
            --flag name1:custom_args1 name2:custom_args2

        Multiple Config (special join) Case:
            (here name2 and name3 have some special interaction)
            --flag name1:custom_args1 name2:custom_args2::name3:custom_args3

        Varied Argument Case:
            --flag name:key1=[val1,val2]

    Returns:
        list: cfg_combos_list

    CommandLine:
        python -m wbia.expt.cfghelpers --exec-parse_cfgstr_list2
        python -m wbia.expt.cfghelpers --test-parse_cfgstr_list2

    Example:
        >>> # ENABLE_DOCTET
        >>> from wbia.expt.cfghelpers import *  # NOQA
        >>> cfgstr_list = ['name', 'name:f=1', 'name:b=[1,2]', 'name1:f=1::name2:f=1,b=2']
        >>> #cfgstr_list = ['name', 'name1:f=1::name2:f=1,b=2']
        >>> named_defaults_dict = None
        >>> cfgtype = None
        >>> alias_keys = None
        >>> valid_keys = None
        >>> expand_nested = True
        >>> strict = False
        >>> special_join_dict = {'joined': True}
        >>> cfg_combos_list = parse_cfgstr_list2(cfgstr_list, named_defaults_dict,
        >>>                                      cfgtype, alias_keys, valid_keys,
        >>>                                      expand_nested, strict,
        >>>                                      special_join_dict)
        >>> print('cfg_combos_list = %s' % (ut.repr2(cfg_combos_list, nl=2),))
        >>> print(ut.depth_profile(cfg_combos_list))
        >>> cfg_list = ut.flatten(cfg_combos_list)
        >>> cfg_list = ut.flatten([cfg if isinstance(cfg, list) else [cfg] for cfg in cfg_list])
        >>> result = ut.repr2(ut.get_varied_cfg_lbls(cfg_list))
        >>> print(result)
        ['name:', 'name:f=1', 'name:b=1', 'name:b=2', 'name1:f=1,joined=True', 'name2:b=2,f=1,joined=True']
    """
    with ut.Indenter('    '):
        cfg_combos_list = []
        for cfgstr in cfgstr_list:
            cfg_combos = []
            # Parse special joined cfg case
            if cfgstr.find('::') > -1:
                special_cfgstr_list = cfgstr.split('::')
                special_combo_list = parse_cfgstr_list2(
                    special_cfgstr_list,
                    named_defaults_dict=named_defaults_dict,
                    cfgtype=cfgtype,
                    alias_keys=alias_keys,
                    valid_keys=valid_keys,
                    strict=strict,
                    expand_nested=expand_nested,
                    is_nestedcfgtype=False,
                    metadata=metadata,
                )
                OLD = False
                if OLD:
                    special_combo = ut.flatten(special_combo_list)
                    if special_join_dict is not None:
                        for cfg in special_combo:
                            cfg.update(special_join_dict)
                else:
                    if special_join_dict is not None:
                        for special_combo in special_combo_list:
                            for cfg in special_combo:
                                cfg.update(special_join_dict)
                if is_nestedcfgtype:
                    cfg_combo = tuple([combo for combo in special_combo_list])
                else:
                    # not sure if this is right
                    cfg_combo = special_combo_list
                # FIXME DUPLICATE CODE
                if expand_nested:
                    cfg_combos.extend(cfg_combo)
                else:
                    # print('Appending: ' + str(ut.depth_profile(cfg_combo)))
                    # if ut.depth_profile(cfg_combo) == [1, 9]:
                    #    ut.embed()
                    cfg_combos_list.append(cfg_combo)
            else:
                cfgname, cfgopt_strs, subx = ut.parse_cfgstr_name_options(cfgstr)
                # --
                # Lookup named default settings
                try:
                    base_cfg_list = ut.lookup_base_cfg_list(
                        cfgname, named_defaults_dict, metadata=metadata
                    )
                except Exception as ex:
                    ut.printex(ex, keys=['cfgstr_list'])
                    raise
                # --
                for base_cfg in base_cfg_list:
                    cfg_combo = customize_base_cfg(
                        cfgname,
                        cfgopt_strs,
                        base_cfg,
                        cfgtype,
                        alias_keys,
                        valid_keys,
                        strict=strict,
                        offset=len(cfg_combos),
                    )
                    if is_nestedcfgtype:
                        cfg_combo = [cfg_combo]
                    if expand_nested:
                        cfg_combos.extend(cfg_combo)
                    else:
                        cfg_combos_list.append(cfg_combo)
            # SUBX Cannot work here because of acfg hackiness
            # if subx is not None:
            #    cfg_combo = ut.take(cfg_combo, subx)
            if expand_nested:
                cfg_combos_list.append(cfg_combos)
        #    print('Updated to: ' + str(ut.depth_profile(cfg_combos_list)))
        # print('Returning len(cfg_combos_list) = %r' % (len(cfg_combos_list),))
    return cfg_combos_list


def parse_argv_cfg(argname, default=[''], named_defaults_dict=None, valid_keys=None):
    """ simple configs

    Args:
        argname (?):
        default (list): (default = [])
        named_defaults_dict (dict): (default = None)
        valid_keys (None): (default = None)

    Returns:
        list: cfg_list

    CommandLine:
        python -m wbia.expt.cfghelpers --exec-parse_argv_cfg --filt :foo=bar
        python -m wbia.expt.cfghelpers --test-parse_argv_cfg

    Example:
        >>> # ENABLE_DOCTET
        >>> from wbia.expt.cfghelpers import *  # NOQA
        >>> argname = '--filt'
        >>> cfg_list = parse_argv_cfg(argname)
        >>> result = ('cfg_list = %s' % (str(cfg_list),))
        >>> print(result)
    """
    if ut.in_jupyter_notebook():
        # dont parse argv in ipython notebook
        cfgstr_list = default
    else:
        cfgstr_list = ut.get_argval(argname, type_=list, default=default)
    if cfgstr_list is None:
        return None
    cfg_combos_list = parse_cfgstr_list2(
        cfgstr_list,
        named_defaults_dict=named_defaults_dict,
        valid_keys=valid_keys,
        strict=False,
    )
    cfg_list = ut.flatten(cfg_combos_list)
    return cfg_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.expt.cfghelpers
        python -m wbia.expt.cfghelpers --allexamples
        python -m wbia.expt.cfghelpers --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
