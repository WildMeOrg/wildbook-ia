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

#NAMEVARSEP = '^'
NAMEVARSEP = ':'


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


def get_varied_cfg_lbls(cfg_list, default_cfg=None, mainkey='_cfgname'):
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
    cfgname_list = [cfg[mainkey] for cfg in cfg_list]
    nonvaried_cfg, varied_cfg_list = partition_varied_cfg_list(cfg_list, default_cfg)
    cfglbl_list = [
        get_cfg_lbl(cfg, name)
        for cfg, name in zip(varied_cfg_list, cfgname_list)]
    return cfglbl_list


def partition_varied_cfg_list(cfg_list, default_cfg=None, recursive=False):
    r"""
    TODO: partition nested configs

    CommandLine:
        python -m ibeis.experiments.cfghelpers --exec-partition_varied_cfg_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.cfghelpers import *  # NOQA
        >>> cfg_list = [{'f': 1, 'b': 1}, {'f': 2, 'b': 1}, {'f': 3, 'b': 1, 'z': 4}]
        >>> nonvaried_cfg, varied_cfg_list = partition_varied_cfg_list(cfg_list)
        >>> result = ut.list_str((nonvaried_cfg, varied_cfg_list), label_list=['nonvaried_cfg', 'varied_cfg_list'])
        >>> print(result)
        nonvaried_cfg = {'b': 1}
        varied_cfg_list = [{'f': 1}, {'f': 2}, {'f': 3, 'z': 4}]

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.cfghelpers import *  # NOQA
        >>> cfg_list = [{'q1': 1, 'f1': {'a2': {'x3': 1, 'y3': 2}, 'b2': 1}}, {'q1': 1, 'f1': {'a2': {'x3': 1, 'y3':1}, 'b2': 1}, 'e1': 1}]
        >>> print(ut.list_str(cfg_list, nl=True))
        >>> nonvaried_cfg, varied_cfg_list = partition_varied_cfg_list(cfg_list, recursive=True)
        >>> result = ut.list_str((nonvaried_cfg, varied_cfg_list), label_list=['nonvaried_cfg', 'varied_cfg_list'])
        >>> print(result)
        nonvaried_cfg = {'f1': {'a2': {'x3': 1}, 'b2': 1}, 'q1': 1}
        varied_cfg_list = [{'f1': {'a2': {'y3': 2}}}, {'e1': 1, 'f1': {'a2': {'y3': 1}}}]
    """
    if default_cfg is None:
        nonvaried_cfg = reduce(ut.dict_intersection, cfg_list)
    else:
        nonvaried_cfg = reduce(ut.dict_intersection, [default_cfg] + cfg_list)
    nonvaried_keys = list(nonvaried_cfg.keys())
    varied_cfg_list = [
        ut.delete_dict_keys(cfg.copy(), nonvaried_keys)
        for cfg in cfg_list]
    if recursive:
        # Find which varied keys have dict values
        varied_keys = list(set([key for cfg in varied_cfg_list for key in cfg]))
        varied_vals_list = [[cfg[key] for cfg in varied_cfg_list if key in cfg] for key in varied_keys]
        for key, varied_vals in zip(varied_keys, varied_vals_list):
            if len(varied_vals) == len(cfg_list):
                if all([isinstance(val, dict) for val in varied_vals]):
                    nonvaried_subdict, varied_subdicts = partition_varied_cfg_list(varied_vals, recursive=recursive)
                    nonvaried_cfg[key] = nonvaried_subdict
                    for cfg, subdict in zip(varied_cfg_list, varied_subdicts):
                        cfg[key] = subdict
    return nonvaried_cfg, varied_cfg_list


def get_cfg_lbl(cfg, name=None, nonlbl_keys=INTERNAL_CFGKEYS, key_order=None):
    r"""
    Formats a flat configuration dict into a short string label

    Args:
        cfg (dict):
        name (str): (default = None)
        nonlbl_keys (list): (default = INTERNAL_CFGKEYS)

    Returns:
        str: cfg_lbl

    CommandLine:
        python -m ibeis.experiments.cfghelpers --exec-get_cfg_lbl

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.cfghelpers import *  # NOQA
        >>> cfg = {'_cfgname': 'test', 'var1': 'val1', 'var2': 'val2'}
        >>> name = None
        >>> nonlbl_keys = ['_cfgstr', '_cfgname', '_cfgtype', '_cfgindex']
        >>> cfg_lbl = get_cfg_lbl(cfg, name, nonlbl_keys)
        >>> result = ('cfg_lbl = %s' % (str(cfg_lbl),))
        >>> print(result)
        cfg_lbl = test:var1=val1,var2=val2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.cfghelpers import *  # NOQA
        >>> cfg = {'_cfgname': 'test:K=[1,2,3]', 'K': '1'}
        >>> name = None
        >>> nonlbl_keys = ['_cfgstr', '_cfgname', '_cfgtype', '_cfgindex']
        >>> cfg_lbl = get_cfg_lbl(cfg, name, nonlbl_keys)
        >>> result = ('cfg_lbl = %s' % (str(cfg_lbl),))
        >>> print(result)
        cfg_lbl = test:K=1
    """
    if name is None:
        name = cfg.get('_cfgname', '')
    _search = ['dict(', ')', ' ']
    _repl = [''] * len(_search)

    # remove keys that should not belong to the label
    _clean_cfg = ut.delete_keys(cfg.copy(), nonlbl_keys)
    _lbl = ut.dict_str(_clean_cfg, explicit=True, nl=False, strvals=True, key_order=key_order)
    _lbl = ut.multi_replace(_lbl, _search, _repl).rstrip(',')
    if NAMEVARSEP in name:
        # hack for when name contains a little bit of the _lbl
        # VERY HACKY TO PARSE OUT PARTS OF THE GIVEN NAME.
        hacked_name, _cfgstr, _ = parse_cfgstr_name_options(name)
        _cfgstr_options_list = re.split(
            r',\s*' + ut.negative_lookahead(r'[^\[\]]*\]'), _cfgstr)
        #cfgstr_options_list = cfgopt_strs.split(',')
        _cfg_options = ut.parse_cfgstr_list(
            _cfgstr_options_list, smartcast=False, oldmode=False)
        #
        ut.delete_keys(_cfg_options, cfg.keys())
        _preflbl = ut.dict_str(_cfg_options, explicit=True, nl=False, strvals=True)
        _preflbl = ut.multi_replace(_preflbl, _search, _repl).rstrip(',')
        hacked_name += NAMEVARSEP + _preflbl
        ###
        cfg_lbl = hacked_name + _lbl
    else:
        cfg_lbl = name + NAMEVARSEP + _lbl
    return cfg_lbl


def customize_base_cfg(cfgname, cfgopt_strs, base_cfg, cfgtype,
                       alias_keys=None, valid_keys=None, offset=0, strict=True):
    """
    cfgopt_strs = 'dsize=1000,per_name=[1,2]'
    """
    cfg = base_cfg.copy()
    # Parse dict out of a string
    cfgstr_options_list = re.split(
        r',\s*' + ut.negative_lookahead(r'[^\[\]]*\]'), cfgopt_strs)
    #cfgstr_options_list = cfgopt_strs.split(',')
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
    if strict:
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
    for combox, cfg_ in enumerate(cfg_combo, start=offset):
        #cfg_['_cfgname'] += ';' + str(combox)
        cfg_['_cfgindex'] = combox
    for cfg_ in cfg_combo:
        if len(cfgopt_strs) > 0:
            cfg_['_cfgstr'] = cfg_['_cfgname'] + NAMEVARSEP + cfgopt_strs
        else:
            cfg_['_cfgstr'] = cfg_['_cfgname']
    return cfg_combo


def try_customize_base(cfgname, cfgopt_strs, base_cfg, cfgtype, alias_keys, valid_keys, cfg_combos, strict):
    try:
        cfg_combo = customize_base_cfg(
            cfgname, cfgopt_strs, base_cfg, cfgtype,
            alias_keys=alias_keys, valid_keys=valid_keys,
            offset=len(cfg_combos), strict=strict)
        return cfg_combo
    except Exception as ex:
        ut.printex(ex, 'Parse Error CfgstrList2',
                   keys=['cfgname', 'cfgopt_strs', 'base_cfg',
                         'cfgtype', 'alias_keys', 'valid_keys'])
        raise


def parse_cfgstr_name_options(cfgstr):
    r"""
    Args:
        cfgstr (str):

    Returns:
        tuple: (cfgname, cfgopt_strs, subx)

    CommandLine:
        python -m ibeis.experiments.cfghelpers --test-parse_cfgstr_name_options

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.cfghelpers import *  # NOQA
        >>> cfgstr = 'default' + NAMEVARSEP + 'myvar1=myval1,myvar2=myval2'
        >>> (cfgname, cfgopt_strs, subx) = parse_cfgstr_name_options(cfgstr)
        >>> result = ('(cfgname, cfg_optstrs, subx) = %s' % (str((cfgname, cfgopt_strs, subx)),))
        >>> print(result)
        (cfgname, cfg_optstrs, subx) = ('default', 'myvar1=myval1,myvar2=myval2', None)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.cfghelpers import *  # NOQA
        >>> cfgstr = 'default[0:1]' + NAMEVARSEP + 'myvar1=myval1,myvar2=myval2'
        >>> (cfgname, cfgopt_strs, subx) = parse_cfgstr_name_options(cfgstr)
        >>> result = ('(cfgname, cfg_optstrs, subx) = %s' % (str((cfgname, cfgopt_strs, subx)),))
        >>> print(result)
        (cfgname, cfg_optstrs, subx) = ('default', 'myvar1=myval1,myvar2=myval2', slice(0, 1, None))

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.experiments.cfghelpers import *  # NOQA
        >>> cfgstr = 'default[0]' + NAMEVARSEP + 'myvar1=myval1,myvar2=myval2'
        >>> (cfgname, cfgopt_strs, subx) = parse_cfgstr_name_options(cfgstr)
        >>> result = ('(cfgname, cfg_optstrs, subx) = %s' % (str((cfgname, cfgopt_strs, subx)),))
        >>> print(result)
        (cfgname, cfg_optstrs, subx) = ('default', 'myvar1=myval1,myvar2=myval2', [0])

    """
    import re
    #cfgname_regex = ut.named_field('cfgname', r'[^\[:]+')  # name is not optional
    cfgname_regex = ut.named_field('cfgname', r'[^\[:]*')  # name is optional
    subx_regex = r'\[' + ut.named_field('subx', r'[^\]]*') + r'\]'
    cfgopt_regex = re.escape(NAMEVARSEP) + ut.named_field('cfgopt', '.*')
    regex_str = cfgname_regex + ('(%s)?' % (subx_regex,)) + ('(%s)?' % (cfgopt_regex,))
    match = re.match(regex_str, cfgstr)
    assert match is not None, 'parsing of cfgstr failed'
    groupdict = match.groupdict()
    cfgname = groupdict['cfgname']
    cfgopt_strs = groupdict.get('cfgopt', None)
    subx_str = groupdict.get('subx', None)
    if cfgopt_strs is None:
        cfgopt_strs = ''
    subx = ut.fuzzy_subset(subx_str)
    ##regex_str = ut.negative_lookbehind(r'\:')
    #cfgstr_split = cfgstr.split(NAMEVARSEP)
    #cfgname_fmt = cfgstr_split[0]
    #cfgopt_strs =  NAMEVARSEP.join(cfgstr_split[1:])
    #if cfgname_fmt.endswith(']'):
    #    # Parse out subindexes
    #    cfgname, subx_str = cfgname_fmt.split('[')
    #else:
    #    cfgname = cfgname_fmt
    #subx = None
    return cfgname, cfgopt_strs, subx


def lookup_base_cfg_list(cfgname, named_defaults_dict, metadata=None):
    if named_defaults_dict is None:
        base_cfg_list = [{}]
    else:
        try:
            base_cfg_list = named_defaults_dict[cfgname]
        except KeyError as ex:
            ut.printex(ex, 'Unknown configuration name', keys=['cfgname'])
            raise
    if ut.is_funclike(base_cfg_list):
        # make callable function
        base_cfg_list = base_cfg_list(metadata)
    if not isinstance(base_cfg_list, list):
        base_cfg_list = [base_cfg_list]
    return base_cfg_list


def parse_cfgstr_list2(cfgstr_list, named_defaults_dict=None, cfgtype=None,
                       alias_keys=None, valid_keys=None, expand_nested=True,
                       strict=True, special_join_dict=None, is_nestedcfgtype=False,
                       metadata=None):
    r"""
    Parses config strings. By looking up name in a dict of configs

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
        python -m ibeis.experiments.cfghelpers --exec-parse_cfgstr_list2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.cfghelpers import *  # NOQA
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
        >>> print('cfg_combos_list = %s' % (ut.list_str(cfg_combos_list, nl=2),))
        >>> print(ut.depth_profile(cfg_combos_list))
        >>> result = (get_varied_cfg_lbls(ut.flatten(cfg_combos_list)))
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
                    named_defaults_dict=named_defaults_dict, cfgtype=cfgtype,
                    alias_keys=alias_keys, valid_keys=valid_keys,
                    strict=strict, expand_nested=expand_nested,
                    is_nestedcfgtype=False, metadata=metadata)
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
                    #print('Appending: ' + str(ut.depth_profile(cfg_combo)))
                    #if ut.depth_profile(cfg_combo) == [1, 9]:
                    #    ut.embed()
                    cfg_combos_list.append(cfg_combo)
            else:
                cfgname, cfgopt_strs, subx = parse_cfgstr_name_options(cfgstr)
                # --
                # Lookup named default settings
                try:
                    base_cfg_list = lookup_base_cfg_list(cfgname, named_defaults_dict, metadata=metadata)
                except Exception as ex:
                    ut.printex(ex, keys=['cfgstr_list'])
                    raise
                # --
                for base_cfg in base_cfg_list:
                    cfg_combo = try_customize_base(cfgname, cfgopt_strs, base_cfg,
                                                   cfgtype, alias_keys, valid_keys,
                                                   cfg_combos, strict)
                    if is_nestedcfgtype:
                        cfg_combo = [cfg_combo]
                    if expand_nested:
                        cfg_combos.extend(cfg_combo)
                    else:
                        cfg_combos_list.append(cfg_combo)
            # SUBX Cannot work here because of acfg hackiness
            #if subx is not None:
            #    cfg_combo = ut.list_take(cfg_combo, subx)
            if expand_nested:
                cfg_combos_list.append(cfg_combos)
        #    print('Updated to: ' + str(ut.depth_profile(cfg_combos_list)))
        #print('Returning len(cfg_combos_list) = %r' % (len(cfg_combos_list),))
    return cfg_combos_list


def parse_argv_cfg(argname, default=[''], named_defaults_dict=None,
                   valid_keys=None):
    """ simple configs

    Args:
        argname (?):
        default (list): (default = [])
        named_defaults_dict (dict): (default = None)
        valid_keys (None): (default = None)

    Returns:
        list: cfg_list

    CommandLine:
        python -m ibeis.experiments.cfghelpers --exec-parse_argv_cfg --filt :foo=bar
        python -m ibeis.experiments.cfghelpers --exec-parse_argv_cfg

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.experiments.cfghelpers import *  # NOQA
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
    cfg_combos_list = parse_cfgstr_list2(cfgstr_list,
                                         named_defaults_dict=named_defaults_dict,
                                         valid_keys=valid_keys,
                                         strict=False)
    cfg_list = ut.flatten(cfg_combos_list)
    return cfg_list


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
