from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import copy
import six
from os.path import splitext
(print, rrr, profile) = ut.inject2(__name__, '[depbase]')


def parse_config_items(cfg):
    """
    Recursively extracts key, val pairs from Config objects
    into a flat list. (there must not be name conflicts)
    """
    param_list = []
    seen = set([])
    for item in cfg.items():
        key, val = item
        if isinstance(val, TableConfig):
            child_cfg = val
            param_list.extend(parse_config_items(child_cfg))
        elif key.startswith('_'):
            pass
        else:
            if key in seen:
                print('[Config] WARNING: key=%r appears more than once' %
                      (key,))
            seen.add(key)
            param_list.append(item)
    return param_list


def make_config_metaclass():
    """
    Hacked over from ibeis.Config
    """
    from utool._internal.meta_util_six import get_funcname
    methods_list = ut.get_comparison_methods()

    # Decorator for functions that we will inject into our metaclass
    def _register(func):
        methods_list.append(func)
        return func

    @_register
    def get_varnames(self):
        return [pi.varname for pi in self.get_param_info_list()] + self._subconfig_names

    @_register
    def get_cfgstr_list(cfg, ignore_keys=None, **kwargs):
        """ default get_cfgstr_list, can be overrided by a config object """
        if ignore_keys is not None:
            itemstr_list = [pi.get_itemstr(cfg)
                            for pi in cfg.get_param_info_list()
                            if pi.varname not in ignore_keys]
        else:
            itemstr_list = [pi.get_itemstr(cfg)
                            for pi in cfg.get_param_info_list()]
        filtered_itemstr_list = list(filter(len, itemstr_list))
        config_name = cfg.get_config_name()
        body = ','.join(filtered_itemstr_list)
        cfgstr = ''.join([config_name, '(', body, ')'])
        return cfgstr

    @_register
    def update(cfg, **kwargs):
        self_keys = set(cfg.__dict__.keys())
        for key, val in six.iteritems(kwargs):
            if key in self_keys:
                setattr(cfg, key, val)

    @_register
    def initialize_params(cfg, **kwargs):
        """ Initializes config class attributes based on params info list """
        for pi in cfg.get_param_info_list():
            setattr(cfg, pi.varname, pi.default)

        # SO HACKY
        cfg._subconfig_names = []
        if hasattr(cfg, 'get_sub_config_list'):
            for subclass in cfg.get_sub_config_list():
                subcfg = subclass()
                subcfg_name = ut.to_underscore_case(subcfg.get_config_name()) + '_cfg'
                setattr(cfg, subcfg_name, subcfg)
                cfg._subconfig_names.append(subcfg_name)
                subcfg.update(**kwargs)
        cfg.update(**kwargs)

    @_register
    def parse_items(cfg, **kwargs):
        param_list = parse_config_items(cfg)
        duplicate_keys = ut.find_duplicate_items(ut.get_list_column(param_list, 0))
        assert len(duplicate_keys) == 0, 'Configs have duplicate names: %r' % duplicate_keys
        return param_list

    @_register
    def get_config_name(cfg, **kwargs):
        """ the user might want to overwrite this function """
        class_str = str(cfg.__class__)
        full_class_str = class_str.replace('<class \'', '').replace('\'>', '')
        config_name = splitext(full_class_str)[1][1:].replace('Config', '')
        return config_name

    @_register
    def __hash__(cfg):
        """ Needed for comparison operators """
        return hash(cfg.get_cfgstr())

    @_register
    def get_cfgstr(cfg, **kwargs):
        str_ = ''.join(cfg.get_cfgstr_list(**kwargs))
        return '_'.join([str_] + [cfg[subcfg_name].get_cfgstr() for subcfg_name in cfg._subconfig_names])

    class ConfigMetaclass(type):
        """
        Defines extra methods for Configs
        """

        def __new__(cls, name, bases, dct):
            """
            cls - meta
            name - classname
            supers - bases
            dct - class dictionary
            """
            #assert 'get_cfgstr_list' in dct, (
            #  'must have defined get_cfgstr_list.  name=%r' % (name,))
            # Inject registered function
            for func in methods_list:
                if get_funcname(func) not in dct:
                    funcname = get_funcname(func)
                    dct[funcname] = func
                else:
                    funcname = get_funcname(func)
                    dct['meta_' + funcname] = func
                #ut.inject_func_as_method(metaself, func)
            return type.__new__(cls, name, bases, dct)

    return ConfigMetaclass

ConfigMetaclass = make_config_metaclass()


class AlgoRequest(object):
    """ Base class for algo request objects """
    pass


class AlgoResult(object):
    """ Base class for algo result objects """

    @classmethod
    def load_from_fpath(cls, fpath, verbose=None):
        state_dict = ut.load_cPkl(fpath, verbose=verbose)
        self = cls()
        self.__setstate__(state_dict)
        return self

    def save_to_fpath(cm, fpath, verbose=None):
        ut.save_cPkl(fpath, cm.__getstate__(), verbose=verbose, n=4)

    def __getstate__(self):
        state_dict = self.__dict__
        return state_dict

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)

    def copy(self):
        cls = self.__class__
        out = cls()
        state_dict = copy.deepcopy(self.__getstate__())
        out.__setstate__(state_dict)
        return out


@six.add_metaclass(ConfigMetaclass)
class TableConfig(ut.DictLike):
    """ Base class for heirarchical params """

    def __init__(cfg, **kwargs):
        cfg.initialize_params(**kwargs)

    def __repr__(cfg):
        return '<%s %s at %s>' % (cfg.__class__.__name__, cfg.get_cfgstr(), hex(id(cfg)))

    def __str__(cfg):
        return '<' + cfg.get_cfgstr() + '>'

    def keys(self):
        return self.get_varnames()

    def getitem(self, key):
        return getattr(self, key)

    def setitem(self, key, value):
        return getattr(self, key, value)


class AlgoConfig(TableConfig):
    """ Base class for heirarchical params """
    pass


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m dtool.base
        python -m dtool.base --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
