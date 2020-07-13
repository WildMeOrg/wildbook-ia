# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import re
import functools
import operator as op
import utool as ut
import numpy as np
import copy
import six

(print, rrr, profile) = ut.inject2(__name__, '[depbase]')


class StackedConfig(ut.DictLike, ut.HashComparable):
    """
    Manages a list of configurations
    """

    def __init__(self, config_list):
        self._orig_config_list = config_list
        # Cast all inputs to config classes
        self._new_config_list = [
            cfg if hasattr(cfg, 'get_cfgstr') else make_configclass(cfg, '')
            for cfg in self._orig_config_list
        ]
        # Parse out items
        self._items = ut.flatten(
            [
                list(cfg.parse_items())
                if hasattr(cfg, 'parse_items')
                else list(cfg.items())
                for cfg in self._orig_config_list
            ]
        )
        for key, val in self._items:
            setattr(self, key, val)
        # self.keys = ut.flatten(list(cfg.keys()) for cfg in self.config_list)

    def get_cfgstr(self):
        cfgstr_list = [cfg.get_cfgstr() for cfg in self._new_config_list]
        cfgstr = '_'.join(cfgstr_list)
        return cfgstr

    def keys(self):
        return ut.take_column(self._items, 0)

    def __hash__(cfg):
        """ Needed for comparison operators """
        return hash(cfg.get_cfgstr())

    def getitem(self, key):
        try:
            return getattr(self, key)
        except AttributeError as ex:
            raise KeyError(ex)


# @six.add_metaclass(ut.HashComparableMetaclass)
@functools.total_ordering
class Config(ut.NiceRepr, ut.DictLike):
    r"""
    Base class for heirarchical config
    need to overwrite get_param_info_list

    CommandLine:
        python -m dtool.base Config

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dtool.base import *  # NOQA
        >>> cfg1 = Config.from_dict({'a': 1, 'b': 2})
        >>> cfg2 = Config.from_dict({'a': 2, 'b': 2})
        >>> # Must be hashable and orderable
        >>> hash(cfg1)
        >>> cfg1 > cfg2

    """

    def __init__(cfg, **kwargs):
        cfg._parent = None
        cfg.initialize_params(**kwargs)

    def deepcopy(cfg):
        cfg2 = copy.deepcopy(cfg)
        cfg2._subconfig_attrs = copy.deepcopy(cfg._subconfig_attrs)
        cfg2._subconfig_names = copy.deepcopy(cfg._subconfig_names)
        try:
            cfg2._param_info_list = copy.deepcopy(cfg._param_info_list)
        except AttributeError:
            pass
        return cfg2

    def __nice__(cfg):
        return cfg.get_cfgstr(with_name=False)

    def __lt__(self, other):
        """ hash comparable broke in python3 """
        return ut.compare_instance(op.lt, self, other)

    def __eq__(self, other):
        """ hash comparable broke in python3 """
        return ut.compare_instance(op.eq, self, other)

    def __hash__(cfg):
        """ Needed for comparison operators """
        return hash(cfg.get_cfgstr())

    def get_config_name(cfg, **kwargs):
        """ the user might want to overwrite this function """
        # VERY HACKY
        config_name = cfg.__class__.__name__.replace('Config', '')
        config_name = re.sub('_$', '', config_name)
        return config_name

    def get_varnames(cfg):
        return [pi.varname for pi in cfg.get_param_info_list()] + cfg._subconfig_attrs

    def update(cfg, **kwargs):
        """
        Overwrites default DictLike update for only keys that exist.
        Non-existing key are ignored.

        Note:
            prefixed keys in the form <classname>_<key> will be just be
            interpreted as <key>

        CommandLine:
            python -m dtool.base update --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.base import *  # NOQA
            >>> from wbia.dtool.example_depcache import DummyVsManyConfig
            >>> cfg = DummyVsManyConfig()
            >>> cfg.update(DummyAlgo_version=4)
            >>> print(cfg)
        """
        # FIXME: currently can't update subconfigs based on namespaces
        # and non-namespaced vars are in the context of the root level.
        # self_keys = set(cfg.__dict__.keys())
        # self_keys.append(cfg.get_varnames())
        _aliases = cfg._make_key_alias_checker()
        self_keys = set(cfg.keys())
        for key, val in six.iteritems(kwargs):
            # update only existing keys or namespace prefixed keys
            for k in _aliases(key):
                if k in self_keys:
                    cfg.setitem(k, val)
                    break

    def pop_update(cfg, other):
        """
        Updates based on other, while popping off used arguments.
        (useful for testing if a parameter was unused or misspelled)

        Doctest:
            >>> from wbia.dtool.base import *  # NOQA
            >>> from wbia import dtool as dt
            >>> cfg = dt.Config.from_dict({'a': 1, 'b': 2, 'c': 3})
            >>> other = {'a': 5, 'e': 2}
            >>> cfg.pop_update(other)
            >>> assert cfg['a'] == 5
            >>> assert len(other) == 1 and 'a' not in other
        """
        _aliases = cfg._make_key_alias_checker()
        self_keys = set(cfg.keys())
        for key in list(other.keys()):
            # update only existing keys or namespace prefixed keys
            for k in _aliases(key):
                if k in self_keys:
                    val = other.pop(key)
                    cfg.setitem(k, val)
                    break

    def _make_key_alias_checker(cfg):
        prefixes = (cfg.get_config_name(), cfg.__class__.__name__)

        def _aliases(key):
            yield key
            for part in prefixes:
                prefix = part + '_'
                if key.startswith(prefix):
                    key_alias = key[len(prefix) :]
                    yield key_alias

        return _aliases

    def update2(cfg, *args, **kwargs):
        """
        Overwrites default DictLike update for only keys that exist.
        Non-existing key are ignored.
        Also updates nested configs.

        Note:
            prefixed keys in the form <classname>_<key> will be just be
            interpreted as <key>

        CommandLine:
            python -m dtool.base update --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.base import *  # NOQA
            >>> from wbia import dtool as dt
            >>> cfg = dt.Config.from_dict({
            >>>     'a': 1,
            >>>     'b': 2,
            >>>     'c': 3,
            >>>     'sub1': dt.Config.from_dict({
            >>>         'x': 'x',
            >>>         'y': {'z', 'x'},
            >>>         'c': 33,
            >>>     }),
            >>>     'sub2': dt.Config.from_dict({
            >>>         's': [1, 2, 3],
            >>>         't': (1, 2, 3),
            >>>         'c': 42,
            >>>         'sub3': dt.Config.from_dict({
            >>>             'b': 99,
            >>>             'c': 88,
            >>>         }),
            >>>     }),
            >>> })
            >>> kwargs = {'c': 10}
            >>> cfg.update2(c=10, y={1,2})
            >>> assert cfg.c == 10
            >>> assert cfg.sub1.c == 10
            >>> assert cfg.sub2.c == 10
            >>> assert cfg.sub2.sub3.c == 10
            >>> assert cfg.sub1.y == {1, 2}
        """
        if len(args) > 1:
            raise ValueError('only specify one arg')
        elif len(args) == 1:
            kwargs.update(args[0])
        return list(cfg._update2(kwargs))

    def _update2(cfg, kwargs):
        # yields a list of keys updated as they happen
        _aliases = cfg._make_key_alias_checker()
        for key, val in cfg.native_items():
            for k in _aliases(key):
                if k in kwargs:
                    cfg.setitem(k, kwargs[k])
                    yield k
                    break
        for key, val in cfg.nested_items():
            val = cfg[key]
            if isinstance(val, Config):
                for k in val._update2(kwargs):
                    yield k

    def nested_items(cfg):
        for key in cfg.keys():
            val = cfg[key]
            if isinstance(val, Config):
                yield key, val

    def native_items(cfg):
        for key in cfg.keys():
            val = cfg[key]
            if not isinstance(val, Config):
                yield key, val

    def initialize_params(cfg, **kwargs):
        """ Initializes config class attributes based on params info list """
        # print("INIT PARAMS")
        for pi in cfg.get_param_info_list():
            setattr(cfg, pi.varname, pi.default)

        # SO HACKY
        # Hacks in implicit edges from nodes to the algorithm
        # using their subconfigurations
        cfg._subconfig_attrs = []
        cfg._subconfig_names = []
        _sub_config_list = cfg.get_sub_config_list()
        if _sub_config_list:
            for subclass in _sub_config_list:
                # subclass.static_config_name()
                subcfg = subclass()
                subcfg_name = subcfg.get_config_name()
                subcfg_attr = ut.to_underscore_case(subcfg_name) + '_cfg'
                setattr(cfg, subcfg_attr, subcfg)
                cfg._subconfig_names.append(subcfg_name)
                cfg._subconfig_attrs.append(subcfg_attr)
                subcfg.update(**kwargs)
        cfg.update(**kwargs)

    def get_sub_config_list(cfg):
        if hasattr(cfg, '_sub_config_list'):
            return cfg._sub_config_list
        else:
            return []

    def parse_namespace_config_items(cfg):
        """
        Recursively extracts key, val pairs from Config objects
        into a flat list. (there must not be name conflicts)
        """
        param_list = []
        seen = set([])
        for item in cfg.items():
            key, val = item
            if hasattr(val, 'parse_namespace_config_items'):
                child_cfg = val
                child_params = child_cfg.parse_namespace_config_items()
                param_list.extend(child_params)
            elif hasattr(val, 'parse_items'):
                # hack for ut.Pref configs
                name = val.get_config_name()
                for key, val in val.parse_items():
                    if key in seen:
                        print('[Config] WARNING: key=%r appears more than once' % (key,))
                    seen.add(key)
                    # Incorporate namespace
                    param_list.append((name, key, val))
            elif key.startswith('_'):
                pass
            else:
                if key in seen:
                    print('[Config] WARNING: key=%r appears more than once' % (key,))
                seen.add(key)
                # Incorporate namespace
                name = cfg.get_config_name()
                param_list.append((name, key, val))
        return param_list

    def parse_items(cfg):
        r"""
        Returns:
            list: param_list

        CommandLine:
            python -m dtool.base --exec-parse_items

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.base import *  # NOQA
            >>> from wbia.dtool.example_depcache import DummyVsManyConfig
            >>> cfg = DummyVsManyConfig()
            >>> param_list = cfg.parse_items()
            >>> result = ('param_list = %s' % (ut.repr2(param_list, nl=1),))
            >>> print(result)
        """
        namespace_param_list = cfg.parse_namespace_config_items()
        param_names = ut.get_list_column(namespace_param_list, 1)
        needs_namespace_keys = ut.find_duplicate_items(param_names)
        param_list = ut.get_list_column(namespace_param_list, [1, 2])
        # prepend namespaces to variables that need it
        for idx in ut.flatten(needs_namespace_keys.values()):
            name = namespace_param_list[idx][0]
            param_list[idx][0] = name + '_' + param_list[idx][0]
        duplicate_keys = ut.find_duplicate_items(ut.get_list_column(param_list, 0))
        # hack to let version through
        # import utool
        # with utool.embed_on_exception_context:
        assert len(duplicate_keys) == 0, (
            'Configs have duplicate names: %r' % duplicate_keys
        )
        return param_list

    def get_cfgstr_list(cfg, ignore_keys=None, with_name=True, **kwargs):
        """ default get_cfgstr_list, can be overrided by a config object """
        if ignore_keys is not None:
            itemstr_list = [
                pi.get_itemstr(cfg)
                for pi in cfg.get_param_info_list()
                if pi.varname not in ignore_keys
            ]
        else:
            itemstr_list = [pi.get_itemstr(cfg) for pi in cfg.get_param_info_list()]
        filtered_itemstr_list = list(filter(len, itemstr_list))
        if with_name:
            config_name = cfg.get_config_name()
        else:
            config_name = ''
        body = ','.join(filtered_itemstr_list)
        cfgstr = ''.join([config_name, '(', body, ')'])
        return cfgstr

    def get_cfgstr(cfg, **kwargs):
        str_ = ''.join(cfg.get_cfgstr_list(**kwargs))
        return '_'.join(
            [str_]
            + [cfg[subcfg_attr].get_cfgstr() for subcfg_attr in cfg._subconfig_attrs]
        )

    def get_param_info_dict(cfg):
        param_info_list = cfg.get_param_info_list()
        param_info_dict = {pi.varname: pi for pi in param_info_list}
        return param_info_dict

    def assert_self_types(cfg, verbose=True):
        if verbose:
            print('Assert self types of cfg=%r' % (cfg,))
        pi_dict = cfg.get_param_info_dict()
        for key in cfg.keys():
            pi = pi_dict[key]
            value = cfg[key]
            pi.error_if_invalid_value(value)
        if verbose:
            print('... checks passed')

    def getinfo(cfg, key):
        pass

    def get_hashid(cfg):
        return ut.hashstr27(cfg.get_cfgstr())

    def keys(cfg):
        """ Required for DictLike interface """
        return cfg.get_varnames()

    def getitem(cfg, key):
        """ Required for DictLike interface """
        try:
            return getattr(cfg, key)
        except AttributeError as ex:
            raise KeyError(ex)

    def get(qparams, key, *d):
        """ get a paramater value by string """
        ERROR_ON_DEFAULT = False
        if ERROR_ON_DEFAULT:
            return getattr(qparams, key)
        else:
            return getattr(qparams, key, *d)

    def setitem(cfg, key, value):
        """ Required for DictLike interface """
        # TODO; check for valid config setting
        pi_dict = cfg.get_param_info_dict()
        pi = pi_dict[key]
        pi.error_if_invalid_value(value)
        return setattr(cfg, key, value)

    def get_param_info_list(cfg):
        try:
            return cfg._param_info_list
        except AttributeError:
            raise NotImplementedError(
                'Need to define _param_info_list or get_param_info_list'
            )

    @classmethod
    def from_argv_dict(cls, **kwargs):
        """
        handy command line tool
        ut.parse_argv_cfg
        """
        cfg = cls(**kwargs)
        new_vals = ut.parse_dict_from_argv(cfg)
        cfg.update(**new_vals)
        return cfg

    @classmethod
    def from_argv_cfgs(cls):
        """
        handy command line tool
        """
        cfg = cls()
        name = cfg.get_config_name()
        # name = cls.static_config_name()
        argname = '--' + name
        if hasattr(cfg, '_alias'):
            argname = (argname, '--' + cfg._alias)
        # if hasattr(cls, '_alias'):
        #    argname = (argname, '--' + cls._alias)
        new_vals_list = ut.parse_argv_cfg(argname)
        self_list = [cls(**new_vals) for new_vals in new_vals_list]
        return self_list

    @classmethod
    def from_dict(cls, dict_, tablename=None):
        r"""
        Args:
            dict_ (dict_):  a dictionary
            tablename (None): (default = None)

        Returns:
            list: param_info_list

        CommandLine:
            python -m dtool.base Config.from_dict --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.dtool.base import *  # NOQA
            >>> cls = Config
            >>> dict_ = {'K': 1, 'Knorm': 5, 'min_pername': 1, 'max_pername': 1,}
            >>> tablename = None
            >>> config = cls.from_dict(dict_, tablename)
            >>> print(config)
            >>> # xdoctest: +REQUIRES(--show)
            >>> ut.quit_if_noshow()
            >>> dlg = config.make_qt_dialog(
            >>>     title='Confirm Merge Query',
            >>>     msg='Confirm')
            >>> dlg.resize(700, 500)
            >>> dlg.show()
            >>> import wbia.plottool as pt
            >>> self = dlg.widget
            >>> guitool.qtapp_loop(qwin=dlg)
            >>> updated_config = self.config  # NOQA
            >>> print('updated_config = %r' % (updated_config,))
        """
        UnnamedConfig = cls.class_from_dict(dict_, tablename)
        config = UnnamedConfig()
        return config

    @classmethod
    def class_from_dict(cls, dict_, tablename=None):
        if tablename is None:
            tablename = 'Unnamed'
        UnnamedConfig = make_configclass(dict_, tablename)
        return UnnamedConfig

    def make_qt_dialog(cfg, parent=None, title='Edit Config', msg='Confim'):
        import wbia.guitool as gt

        gt.ensure_qapp()  # must be ensured before any embeding
        dlg = gt.ConfigConfirmWidget.as_dialog(title=title, msg=msg, config=cfg)
        dlg.resize(700, 500)
        dlg.show()
        return dlg

    def getstate_todict_recursive(cfg):
        from wbia import dtool

        _dict = cfg.asdict()
        _dict2 = {}
        for key, val in _dict.items():
            if isinstance(val, dtool.Config):
                # val = val.asdict()
                try:
                    val = val.getstate_todict_recursive()
                except Exception:
                    val = getstate_todict_recursive(val)  # NOQA
                _dict2[key] = val
            else:
                _dict2[key] = val
        return _dict2

    def __getstate__(cfg):
        """
        FIXME

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.base import *  # NOQA
            >>> from wbia.dtool.example_depcache import DummyKptsConfig
            >>> from six.moves import cPickle as pickle
            >>> cfg = DummyKptsConfig()
            >>> ser = pickle.dumps(cfg)
            >>> cfg2 = pickle.loads(ser)
            >>> assert cfg == cfg2
            >>> assert cfg is not cfg2

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.base import *  # NOQA
            >>> from wbia.dtool.example_depcache import DummyVsManyConfig
            >>> from six.moves import cPickle as pickle
            >>> cfg = DummyVsManyConfig()
            >>> state = cfg.__getstate__()
            >>> cfg2 = DummyVsManyConfig()
            >>> serialized = pickle.dumps(cfg)
            >>> unserialized = pickle.loads(serialized)
            >>> assert cfg == unserialized
            >>> assert cfg is not unserialized
        """
        # from wbia import dtool
        # _dict = cfg.asdict()
        # _dict2 = {}
        # for key, val in _dict.items():
        #    if isinstance(val, dtool.Config):
        #        val = val.asdict()
        #    _dict2[key] = val
        # return {'dtool.Config': _dict2}
        return cfg.__dict__

    def __setstate__(cfg, state):
        cfg.__dict__.update(**state)
        # cfg.initialize_params()
        # cfg.update(**state)

    # @classmethod
    # def static_config_name(cls):
    #    class_str = str(cls)
    #    full_class_str = class_str.replace('<class \'', '').replace('\'>', '')
    #    config_name = splitext(full_class_str)[1][1:].replace('Config', '')
    #    return config_name


def make_configclass(dict_, tablename):
    """ Creates a custom config class from a dict """

    def rectify_item(key, val):
        if val is None:
            return ut.ParamInfo(key, val)
        elif isinstance(val, ut.ParamInfo):
            if val.varname is None:
                # Copy and assign a new varname
                pi = copy.deepcopy(val)
                pi.varname = key
            else:
                pi = val
                assert pi.varname == key, 'Given varname=%r does not match key=%r' % (
                    pi.varname,
                    key,
                )
            return pi
        else:
            if isinstance(val, Config):
                # Set table name from key when doing nested from dicts
                if val.__class__.__name__ == 'UnnamedConfig':
                    val.__class__.__name__ = str(key + 'Config')
            return ut.ParamInfo(key, val, type_=type(val))

    param_info_list = [rectify_item(key, val) for key, val in dict_.items()]
    return from_param_info_list(param_info_list, tablename)


def from_param_info_list(param_info_list, tablename='Unnamed'):
    from wbia import dtool

    class UnnamedConfig(dtool.Config):
        _param_info_list = param_info_list

    UnnamedConfig.__name__ = str(tablename + 'Config')
    return UnnamedConfig


class IBEISRequestHacks(object):
    _isnewreq = True

    @property
    def ibs(request):
        """ HACK specific to wbia """
        if request.depc is None:
            return None
        return request.depc.controller

    @property
    def qannots(self):
        return self.ibs.annots(self.qaids, self.params)

    @property
    def dannots(self):
        return self.ibs.annots(self.daids, self.params)

    def get_qreq_annot_nids(self, aids):
        # VERY HACKY. To be just hacky it should store
        # the nids as a state, but whatever...
        # devleopment time constraints and whatnot
        return self.ibs.get_annot_nids(aids)
        # return self.ibs.annots(self.daids, self.params)

    @property
    def extern_query_config2(request):
        return request.params

    @property
    def extern_data_config2(request):
        return request.params

    #

    # def get_external_data_config2(request):
    #    # HACK
    #    #return None
    #    #print('[d] request.params = %r' % (request.params,))
    #    return request.params

    # def get_external_query_config2(request):
    #    # HACK
    #    #return None
    #    #print('[q] request.params = %r' % (request.params,))
    #    return request.params


def config_graph_subattrs(cfg, depc):
    # TODO: if this hack is fully completed need a way of getting the
    # full config belonging to both chip + feat
    # cfg = request.config.feat_cfg
    import networkx as netx

    tablename = ut.invert_dict(depc.configclass_dict)[cfg.__class__]
    # tablename = cfg.get_config_name()
    ancestors = netx.dag.ancestors(depc.graph, tablename)
    subconfigs_ = ut.dict_take(depc.configclass_dict, ancestors, None)
    subconfigs = ut.filter_Nones(subconfigs_)  # NOQA


@ut.reloadable_class
class BaseRequest(IBEISRequestHacks, ut.NiceRepr):
    r"""
    Class that maintains both an algorithm, inputs, and a config.
    """

    @staticmethod
    def static_new(cls, depc, parent_rowids, cfgdict=None, tablename=None):
        """ hack for autoreload """
        request = cls()
        if tablename is None:
            try:
                if hasattr(cls, '_tablename'):
                    tablename = cls._tablename
                else:
                    tablename = ut.invert_dict(depc.requestclass_dict)[cls]
            except Exception as ex:
                ut.printex(ex, 'tablename must be given')
                raise
        request.tablename = tablename
        request.parent_rowids = parent_rowids
        request.depc = depc
        if cfgdict is None:
            cfgdict = {}
        configclass = depc.configclass_dict[tablename]
        config = configclass(**cfgdict)
        request.config = config
        # HACK FOR IBEIS
        request.params = dict(config.parse_items())
        # HACK-ier FOR BACKWARDS COMPATABILITY
        if True:
            # params.featweight_cfgstr = query_cfg._featweight_cfg.get_cfgstr()
            # TODO: if this hack is fully completed need a way of getting the
            # full config belonging to both chip + feat
            try:
                request.params['chip_cfgstr'] = config.chip_cfg.get_cfgstr()
                request.params['chip_cfg_dict'] = config.chip_cfg.asdict()
                request.params['feat_cfgstr'] = config.feat_cfg.get_cfgstr()
                request.params['hesaff_params'] = config.feat_cfg.get_hesaff_params()
                request.params['featweight_cfgstr'] = config.feat_weight_cfg.get_cfgstr()
            except AttributeError:
                pass
        request.qparams = ut.DynStruct()
        for key, val in request.params.items():
            setattr(request.qparams, key, val)

        return request

    @classmethod
    def new(cls, depc, parent_rowids, cfgdict=None, tablename=None):
        return cls.static_new(cls, depc, parent_rowids, cfgdict, tablename)

    def _get_rootset_hashid(request, root_rowids, prefix):
        uuid_type = 'V'
        label = ''.join((prefix, uuid_type, 'UUIDS'))
        # Hack: allow general specification of uuid types
        uuid_list = request.depc.get_root_uuid(root_rowids)
        # uuid_hashid = ut.hashstr_arr27(uuid_list, label, pathsafe=True)
        uuid_hashid = ut.hashstr_arr27(uuid_list, label, pathsafe=False)
        # TODO: uuid_hashid = ut.hashid_arr(uuid_list, label=label)
        return uuid_hashid

    def get_cfgstr(request, with_input=False, with_pipe=True, **kwargs):
        r"""
        main cfgstring used to identify the 'querytype'
        """
        cfgstr_list = []
        if with_input:
            cfgstr_list.append(request.get_input_hashid())
        if with_pipe:
            cfgstr_list.append(request.get_pipe_cfgstr())
        cfgstr = '_'.join(cfgstr_list)
        return cfgstr

    def get_input_hashid(request):
        raise NotImplementedError('abstract class method')

    def get_pipe_cfgstr(request):
        return request.config.get_cfgstr()

    def get_pipe_hashid(request):
        return ut.hashstr27(request.get_pipe_cfgstr())

    def ensure_dependencies(request):
        r"""
        CommandLine:
            python -m dtool.base --exec-BaseRequest.ensure_dependencies

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.base import *  # NOQA
            >>> from wbia.dtool.example_depcache import testdata_depc
            >>> depc = testdata_depc()
            >>> request = depc.new_request('vsmany', [1, 2], [2, 3, 4])
            >>> request.ensure_dependencies()
        """
        import networkx as nx

        depc = request.depc
        if False:
            dependencies = nx.ancestors(depc.graph, request.tablename)
            subgraph = depc.graph.subgraph(set.union(dependencies, {request.tablename}))
            dependency_order = nx.topological_sort(subgraph)
            root = dependency_order[0]
            [
                nx.algorithms.dijkstra_path(subgraph, root, start)[:-1]
                + nx.algorithms.dijkstra_path(subgraph, start, request.tablename)
                for start in dependency_order
            ]
        graph = depc.graph
        root = list(nx.topological_sort(graph))[0]
        edges = graph.edges()
        # parent_to_children = ut.edges_to_adjacency_list(edges)
        child_to_parents = ut.edges_to_adjacency_list([t[::-1] for t in edges])
        to_root = {
            request.tablename: ut.paths_to_root(request.tablename, root, child_to_parents)
        }
        from_root = ut.reverse_path(to_root, root, child_to_parents)
        dependency_levels_ = ut.get_levels(from_root)
        dependency_levels = ut.longest_levels(dependency_levels_)

        true_order = ut.flatten(dependency_levels)[1:-1]
        # print('[req] Ensuring %s request dependencies: %r' % (request, true_order,))
        ut.colorprint(
            '[req] Ensuring request %s dependencies: %r' % (request, true_order,),
            'yellow',
        )
        for tablename in true_order:
            table = depc[tablename]
            if table.ismulti:
                pass
            else:
                # HACK FOR IBEIS
                all_aids = ut.flat_unique(request.qaids, request.daids)
                depc.get_rowids(tablename, all_aids)
                pass
            pass

        # zip(depc.get_implicit_edges())
        # zip(depc.get_implicit_edges())

        # raise NotImplementedError('todo')
        # depc = request.depc
        # parent_rowids = request.parent_rowids
        # config = request.config
        # rowid_dict = depc.get_all_descendant_rowids(
        #    request.tablename, root_rowids, config=config)
        pass

    def execute(request, parent_rowids=None, use_cache=None, postprocess=True):
        ut.colorprint('[req] Executing request %s' % (request,), 'yellow')
        table = request.depc[request.tablename]
        if use_cache is None:
            use_cache = not ut.get_argflag('--nocache')
        if parent_rowids is None:
            parent_rowids = request.parent_rowids
        # Compute and cache any uncomputed results
        rowids = table.get_rowid(parent_rowids, config=request, recompute=not use_cache)
        # Load all results
        result_list = table.get_row_data(rowids)
        if postprocess and hasattr(request, 'postprocess_execute'):
            print('Converting results')
            result_list = request.postprocess_execute(parent_rowids, result_list)
            pass
        return result_list

    def __getstate__(request):
        state_dict = request.__dict__.copy()
        # SUPER HACK
        state_dict['dbdir'] = request.depc.controller.get_dbdir()
        del state_dict['depc']
        del state_dict['config']
        return state_dict

    def __setstate__(request, state_dict):
        import wbia

        dbdir = state_dict['dbdir']
        del state_dict['dbdir']
        params = state_dict['params']
        depc = wbia.opendb(dbdir=dbdir, web=False).depc
        configclass = depc.configclass_dict[state_dict['tablename']]
        config = configclass(**params)
        state_dict['depc'] = depc
        state_dict['config'] = config
        request.__dict__.update(state_dict)


class AnnotSimiliarity(object):
    def get_query_hashid(request):
        return request._get_rootset_hashid(request.qaids, 'Q')

    def get_data_hashid(request):
        return request._get_rootset_hashid(request.daids, 'D')


@ut.reloadable_class
class VsOneSimilarityRequest(BaseRequest, AnnotSimiliarity):
    r"""
    Similarity request for pairwise scores

    References:
        https://thingspython.wordpress.com/2010/09/27/
        another-super-wrinkle-raising-typeerror/

    CommandLine:
        python -m dtool.base --exec-VsOneSimilarityRequest

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dtool.base import *  # NOQA
        >>> from wbia.dtool.example_depcache import testdata_depc
        >>> qaid_list = [1, 2, 3, 5]
        >>> daid_list = [2, 3, 4]
        >>> depc = testdata_depc()
        >>> request = depc.new_request('vsone', qaid_list, daid_list)
        >>> results = request.execute()
        >>> # Test that adding a query / data id only recomputes necessary items
        >>> request2 = depc.new_request('vsone', qaid_list + [4], daid_list + [5])
        >>> results2 = request2.execute()
        >>> print('results = %r' % (results,))
        >>> print('results2 = %r' % (results2,))
        >>> ut.assert_eq(len(results), 10, 'incorrect num output')
        >>> ut.assert_eq(len(results2), 16, 'incorrect num output')
    """

    _symmetric = False

    @classmethod
    def new(cls, depc, qaid_list, daid_list, cfgdict=None, tablename=None):
        parent_rowids = cls.make_parent_rowids(qaid_list, daid_list)
        parent_rowids = list(ut.product_nonsame(qaid_list, daid_list))
        request = cls.static_new(cls, depc, parent_rowids, cfgdict, tablename)
        request.qaids = safeop(np.array, qaid_list)
        request.daids = safeop(np.array, daid_list)
        return request

    @staticmethod
    def make_parent_rowids(qaid_list, daid_list):
        return list(ut.product_nonsame(qaid_list, daid_list))

    @property
    def parent_rowids_T(request):
        return ut.list_transpose(request.parent_rowids)

    def execute(request, parent_rowids=None, use_cache=None, postprocess=True):
        """ HACKY REIMPLEMENTATION """
        ut.colorprint('[req] Executing request %s' % (request,), 'yellow')
        table = request.depc[request.tablename]
        if use_cache is None:
            use_cache = not ut.get_argflag('--nocache')
        if parent_rowids is None:
            parent_rowids = request.parent_rowids
        else:
            # previously defined in execute subset
            # subparent_rowids = request.make_parent_rowids(
            # qaids, request.daids)
            print('given %d specific parent_rowids' % (len(parent_rowids),))

        # vsone hack (i,j) same as (j,i)
        if request._symmetric:
            import vtool as vt

            directed_edges = np.array(parent_rowids)
            undirected_edges = vt.to_undirected_edges(directed_edges)
            edge_ids = vt.compute_unique_data_ids(undirected_edges)
            unique_rows, unique_rowx, inverse_idx = np.unique(
                edge_ids, return_index=True, return_inverse=True
            )
            parent_rowids_ = ut.take(parent_rowids, unique_rowx)
        else:
            parent_rowids_ = parent_rowids

        # Compute and cache any uncomputed results
        rowids = table.get_rowid(parent_rowids_, config=request, recompute=not use_cache)
        # Load all results
        result_list = table.get_row_data(rowids)

        if request._symmetric:
            result_list = ut.take(result_list, inverse_idx)

        if postprocess and hasattr(request, 'postprocess_execute'):
            print('Converting results')
            result_list = request.postprocess_execute(parent_rowids, result_list)
            pass
        return result_list

    def get_input_hashid(request):
        return '_'.join([request.get_query_hashid(), request.get_data_hashid()])

    def __nice__(request):
        dbname = (
            None
            if request.depc is None or request.depc.controller is None
            else request.depc.controller.get_dbname()
        )
        infostr_ = 'nQ=%s, nD=%s, nP=%d %s' % (
            len(request.qaids),
            len(request.daids),
            len(request.parent_rowids),
            request.get_pipe_hashid(),
        )
        return '(%s) %s' % (dbname, infostr_)


@ut.reloadable_class
class VsManySimilarityRequest(BaseRequest, AnnotSimiliarity):
    r"""
    Request for one-vs-many simlarity

    CommandLine:
        python -m dtool.base --exec-VsManySimilarityRequest

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dtool.base import *  # NOQA
        >>> from wbia.dtool.example_depcache import testdata_depc
        >>> qaid_list = [1, 2]
        >>> daid_list = [2, 3, 4]
        >>> depc = testdata_depc()
        >>> request = depc.new_request('vsmany', qaid_list, daid_list)
        >>> request.ensure_dependencies()
        >>> results = request.execute()
        >>> # Test dependence on data
        >>> request2 = depc.new_request('vsmany', qaid_list + [3], daid_list + [5])
        >>> results2 = request2.execute()
        >>> print('results = %r' % (results,))
        >>> print('results2 = %r' % (results2,))
        >>> assert len(results) == 2, 'incorrect num output'
        >>> assert len(results2) == 3, 'incorrect num output'
    """

    @classmethod
    def new(cls, depc, qaid_list, daid_list, cfgdict=None, tablename=None):
        parent_rowids = list(zip(qaid_list))
        request = cls.static_new(cls, depc, parent_rowids, cfgdict, tablename)
        request.qaids = safeop(np.array, qaid_list)
        request.daids = safeop(np.array, daid_list)
        # HACK
        request.config.daids = request.daids
        return request

    def get_input_hashid(request):
        # return '_'.join([request.get_query_hashid(), request.get_data_hashid()])
        return '_'.join([request.get_query_hashid()])

    def get_cfgstr(
        request, with_input=False, with_data=True, with_pipe=True, hash_pipe=False
    ):
        r"""
        Override default get_cfgstr to show reliance on data
        """
        cfgstr_list = []
        if with_input:
            cfgstr_list.append(request.get_query_hashid())
        if with_data:
            cfgstr_list.append(request.get_data_hashid())
        if with_pipe:
            if hash_pipe:
                cfgstr_list.append(request.get_pipe_hashid())
            else:
                cfgstr_list.append(request.get_pipe_cfgstr())
        cfgstr = '_'.join(cfgstr_list)
        return cfgstr

    def __nice__(request):
        dbname = (
            None
            if request.depc is None or request.depc.controller is None
            else request.depc.controller.get_dbname()
        )
        infostr_ = 'nQ=%s, nD=%s %s' % (
            len(request.qaids),
            len(request.daids),
            request.get_pipe_hashid(),
        )
        return '(%s) %s' % (dbname, infostr_)


class ClassVsClassSimilarityRequest(BaseRequest):
    pass


class AlgoResult(object):
    """ Base class for algo result objects """

    @classmethod
    def load_from_fpath(cls, fpath, verbose=ut.VERBOSE):
        state_dict = ut.load_cPkl(fpath, verbose=verbose)
        self = cls()
        self.__setstate__(state_dict)
        return self

    def save_to_fpath(cm, fpath, verbose=ut.VERBOSE):
        ut.save_cPkl(fpath, cm.__getstate__(), verbose=verbose, n=2)

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


def safeop(op_, xs, *args, **kwargs):
    return None if xs is None else op_(xs, *args, **kwargs)


class MatchResult(AlgoResult, ut.NiceRepr):
    def __init__(
        self,
        qaid=None,
        daids=None,
        qnid=None,
        dnid_list=None,
        annot_score_list=None,
        unique_nids=None,
        name_score_list=None,
    ):
        self.qaid = qaid
        self.daid_list = safeop(np.array, daids)
        self.dnid_list = safeop(np.array, dnid_list)
        self.annot_score_list = safeop(np.array, annot_score_list)
        self.name_score_list = safeop(np.array, name_score_list)

    @property
    def num_daids(cm):
        return None if cm.daid_list is None else len(cm.daid_list)

    @property
    def daids(cm):
        return cm.daid_list

    @property
    def qaids(cm):
        return cm.qaid

    def __nice__(cm):
        return ' qaid=%s nD=%s' % (cm.qaid, cm.num_daids)


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
