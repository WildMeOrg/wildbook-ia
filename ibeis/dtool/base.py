# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import copy
import six
from os.path import splitext
(print, rrr, profile) = ut.inject2(__name__, '[depbase]')


class Config(ut.NiceRepr, ut.DictLike, ut.HashComparable):
    r"""
    Base class for heirarchical config
    need to overwrite get_param_info_list
    """
    def __init__(self, **kwargs):
        self.initialize_params(**kwargs)

    def __nice__(self):
        return self.get_cfgstr(with_name=False)

    def __hash__(cfg):
        """ Needed for comparison operators """
        return hash(cfg.get_cfgstr())

    def get_config_name(cfg, **kwargs):
        """ the user might want to overwrite this function """
        class_str = str(cfg.__class__)
        full_class_str = class_str.replace('<class \'', '').replace('\'>', '')
        config_name = splitext(full_class_str)[1][1:].replace('Config', '')
        return config_name

    def get_varnames(self):
        return ([pi.varname for pi in self.get_param_info_list()] +
                self._subconfig_attrs)

    def update(cfg, **kwargs):
        """
        Overwrites default DictLike update for only keys that exist.

        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.base import *  # NOQA
            >>> from dtool.example_depcache import DummyAlgoConfig
            >>> cfg = DummyAlgoConfig()
            >>> cfg.update(DummyAlgo_version=4)
            >>> print(cfg)
        """
        # FIXME: currently can't update subconfigs based on namespaces
        # and non-namespaced vars are in the context of the root level.
        self_keys = set(cfg.__dict__.keys())
        name = cfg.get_config_name()
        prefix = name + '_'
        for key, val in six.iteritems(kwargs):
            # update only existing keys or namespace prefixed keys
            if key.startswith(prefix):
                key = key[len(prefix):]
            if key in self_keys:
                setattr(cfg, key, val)

    def initialize_params(cfg, **kwargs):
        """ Initializes config class attributes based on params info list """
        for pi in cfg.get_param_info_list():
            setattr(cfg, pi.varname, pi.default)

        # SO HACKY
        # Hacks in implicit edges from nodes to the algorithm
        # using their subconfigurations
        cfg._subconfig_attrs = []
        cfg._subconfig_names = []
        _sub_config_list = None
        if hasattr(cfg, 'get_sub_config_list'):
            _sub_config_list = cfg.get_sub_config_list()
        if hasattr(cfg, '_sub_config_list'):
            _sub_config_list = cfg._sub_config_list
        if _sub_config_list:
            for subclass in _sub_config_list:
                #subclass.static_config_name()
                subcfg = subclass()
                subcfg_name = subcfg.get_config_name()
                subcfg_attr = ut.to_underscore_case(subcfg_name) + '_cfg'
                setattr(cfg, subcfg_attr, subcfg)
                cfg._subconfig_names.append(subcfg_name)
                cfg._subconfig_attrs.append(subcfg_attr)
                subcfg.update(**kwargs)
        cfg.update(**kwargs)

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
            elif key.startswith('_'):
                pass
            else:
                if key in seen:
                    print('[Config] WARNING: key=%r appears more than once' %
                          (key,))
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
            >>> from dtool.base import *  # NOQA
            >>> from dtool.example_depcache import DummyAlgoConfig
            >>> cfg = DummyAlgoConfig()
            >>> param_list = cfg.parse_items()
            >>> result = ('param_list = %s' % (ut.repr2(param_list, nl=1),))
            >>> print(result)
        """
        namespace_param_list = Config.parse_namespace_config_items(cfg)
        needs_namespace_keys = ut.find_duplicate_items(ut.get_list_column(namespace_param_list, 1))
        param_list = ut.get_list_column(namespace_param_list, [1, 2])
        # prepend namespaces to variables that need it
        for idx in ut.flatten(needs_namespace_keys.values()):
            name = namespace_param_list[idx][0]
            param_list[idx][0] = name + '_' + param_list[idx][0]
        duplicate_keys = ut.find_duplicate_items(ut.get_list_column(param_list, 0))
        # hack to let version through
        assert len(duplicate_keys) == 0, (
            'Configs have duplicate names: %r' % duplicate_keys)
        return param_list

    def get_cfgstr_list(cfg, ignore_keys=None, with_name=True, **kwargs):
        """ default get_cfgstr_list, can be overrided by a config object """
        if ignore_keys is not None:
            itemstr_list = [pi.get_itemstr(cfg)
                            for pi in cfg.get_param_info_list()
                            if pi.varname not in ignore_keys]
        else:
            itemstr_list = [pi.get_itemstr(cfg)
                            for pi in cfg.get_param_info_list()]
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
        return '_'.join([str_] + [cfg[subcfg_attr].get_cfgstr()
                                  for subcfg_attr in cfg._subconfig_attrs])

    def get_hashid(cfg):
        return ut.hashstr27(cfg.get_cfgstr())

    def keys(self):
        """ Required for DictLike interface """
        return self.get_varnames()

    def getitem(self, key):
        """ Required for DictLike interface """
        try:
            return getattr(self, key)
        except AttributeError as ex:
            raise KeyError(ex)

    def setitem(self, key, value):
        """ Required for DictLike interface """
        return getattr(self, key, value)

    def get_param_info_list(self):
        try:
            return self._param_info_list
        except AttributeError:
            raise NotImplementedError(
                'Need to define _param_info_list or get_param_info_list')

    @classmethod
    def from_argv_dict(cls, **kwargs):
        """
        handy command line tool
        ut.parse_argv_cfg
        """
        self = cls(**kwargs)
        new_vals = ut.parse_dict_from_argv(self)
        self.update(**new_vals)
        return self

    @classmethod
    def from_argv_cfgs(cls):
        """
        handy command line tool
        """
        self = cls()
        name = self.get_config_name()
        #name = cls.static_config_name()
        argname = '--' + name
        if hasattr(self, '_alias'):
            argname = (argname, '--' + self._alias)
        #if hasattr(cls, '_alias'):
        #    argname = (argname, '--' + cls._alias)
        new_vals_list = ut.parse_argv_cfg(argname)
        self_list = [cls(**new_vals) for new_vals in new_vals_list]
        return self_list

    def __getstate__(self):
        return self.asdict()

    def __setstate__(self, state):
        self.update(**state)

    #@classmethod
    #def static_config_name(cls):
    #    class_str = str(cls)
    #    full_class_str = class_str.replace('<class \'', '').replace('\'>', '')
    #    config_name = splitext(full_class_str)[1][1:].replace('Config', '')
    #    return config_name


class TableConfig(Config):
    pass


class AlgoConfig(TableConfig):
    pass


def dict_as_config(default_cfgdict, tablename):
    import dtool
    class UnnamedConfig(dtool.TableConfig):
        def get_param_info_list(self):
            #print('default_cfgdict = %r' % (default_cfgdict,))
            return [ut.ParamInfo(key, val)
                    for key, val in default_cfgdict.items()]
    UnnamedConfig.__name__ = str(tablename + 'Config')
    return UnnamedConfig


class AlgoRequest(ut.NiceRepr):
    """
    Base class for algo request objects
    Need this for TestResult Integration

    This class might not be need, and is being added for
    compatibility support.
    The problem it solve is having daids as part of a config.  A config should
    be used to specify algorithm parameters, but a referense set of matchable
    annotations seems to go beyond that.  Therefore, AlgoRequest.

    Ignore:
        cls = dtool.AlgoRequest

    Example:
        >>> # ENABLE_DOCTEST
        >>> from dtool.base import *  # NOQA
        >>> from dtool.example_depcache import testdata_depc
        >>> depc = testdata_depc()
        >>> request1 = depc.new_algo_request('vsone', [1, 2], [1, 2])
        >>> request2 = depc.new_algo_request('dumbalgo', [1, 2], [1, 2])
    """
    _isnewreq = True
    _qaids_independent = True
    _daids_independent = False

    @classmethod
    def new_algo_request(cls, depc, algoname, qaids, daids, cfgdict=None):
        request = cls()
        request._qaids = None
        request._daids = None

        request.depc = depc
        request.qaids = qaids
        request.daids = daids
        if cfgdict is None:
            cfgdict = {}
        configclass = depc.configclass_dict[algoname]
        config = configclass(**cfgdict)

        request.config = config
        request.algoname = algoname

        # hack
        request.params = dict(config.parse_items())
        return request

    @property
    def ibs(request):
        """ HACK specific to ibeis """
        if request.depc is None:
            return None
        return request.depc.controller

    def get_external_data_config2(request):
        # HACK
        #return None
        #print('[d] request.params = %r' % (request.params,))
        return request.params

    def get_external_query_config2(request):
        # HACK
        #return None
        #print('[q] request.params = %r' % (request.params,))
        return request.params

    @property
    def qaids(request):
        return request._qaids

    @qaids.setter
    def qaids(request, qaids):
        request._qaids = safeop(np.array, qaids)

    @property
    def daids(request):
        return request._daids

    @property
    def cfgstr(request):
        return request.get_cfgstr()

    @daids.setter
    def daids(request, daids):
        request._daids = safeop(np.array, daids)

    def get_parent_rowids(request):
        if request._daids_independent:
            parent_rowids = list(ut.product(request.qaids, request.daids))
        else:
            parent_rowids = list(zip(request.qaids))
        return parent_rowids

    def execute(request, qaids=None, use_cache=None):
        if qaids is not None:
            qaids = [qaids] if not ut.isiterable(qaids) else qaids
            subreq = request.shallowcopy(qaids=qaids)
            return subreq.execute(use_cache=True)
        else:
            tablename = request.algoname
            table = request.depc[tablename]
            if use_cache is None:
                use_cache = not ut.get_argflag('--nocache')

            parent_rowids = request.get_parent_rowids()
            rowids = table.get_rowid(parent_rowids, config=request,
                                     recompute=not use_cache)
            result_list = table.get_row_data(rowids)
            return ut.get_list_column(result_list, 0)

    def get_query_hashid(request):
        return request._get_rootset_hashid(request.qaids, 'Q')

    def get_data_hashid(request):
        return request._get_rootset_hashid(request.daids, 'D')

    def _get_rootset_hashid(request, root_rowids, preffix):
        uuid_type = 'V'
        label = ''.join((preffix, uuid_type, 'UUIDS'))
        uuid_list = request.depc.get_root_uuid(root_rowids)
        #uuid_hashid = ut.hashstr_arr27(uuid_list, label, pathsafe=True)
        uuid_hashid = ut.hashstr_arr27(uuid_list, label, pathsafe=False)
        return uuid_hashid

    def get_pipe_cfgstr(request):
        return request.config.get_cfgstr()

    def get_pipe_hashid(request):
        return ut.hashstr27(request.get_pipe_cfgstr())

    def get_cfgstr(request, with_query=None, with_data=None, with_pipe=True,
                   hash_pipe=False):
        r"""
        main cfgstring used to identify the 'querytype'
        """
        if with_query is None:
            #with_query = False
            with_query = not request._qaids_independent

        if with_data is None:
            #with_data = True
            # non-independent aids must be in config string
            with_data = not request._daids_independent

        cfgstr_list = []
        if with_query:
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

    def get_full_cfgstr(request):
        """ main cfgstring used to identify the algo hash id """
        full_cfgstr = request.get_cfgstr(with_query=True)
        return full_cfgstr

    def __nice__(request):
        dbname = (None if request.depc is None or request.depc.controller is None
                  else request.depc.controller.get_dbname())
        infostr_ = 'nQ=%s, nD=%s %s' % (len(request.qaids), len(request.daids),
                                        request.get_pipe_hashid())
        return '(%s) %s' % (dbname, infostr_)

    def __getstate__(request):
        state_dict = request.__dict__.copy()
        # SUPER HACK
        state_dict['dbdir'] = request.depc.controller.get_dbdir()
        del state_dict['depc']
        del state_dict['config']
        return state_dict

    def __setstate__(request, state_dict):
        import ibeis
        dbdir = state_dict['dbdir']
        del state_dict['dbdir']
        params = state_dict['params']
        depc = ibeis.opendb(dbdir=dbdir, web=False).depc
        configclass = depc.configclass_dict[state_dict['algoname'] ]
        config = configclass(**params)
        state_dict['depc'] = depc
        state_dict['config'] = config
        request.__dict__.update(state_dict)

    def shallowcopy(request, qmask=None, qaids=None):
        """
        Creates a copy of qreq with the same qparams object and a subset of the
        qx and dx objects.  used to generate chunks of vsone and vsmany queries
        """
        #subreq = copy.copy(request)  # copy calls setstate and getstate
        subreq = request.__class__()
        subreq.__dict__.update(request.__dict__)
        if qmask is not None:
            assert qaids is None, 'cannot specify both'
            qaid_list  = subreq.qaids
            subreq.qaids = ut.compress(qaid_list, qmask)
        elif qaids is not None:
            subreq.qaids = qaids
        return subreq


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
    def __init__(self, qaid=None, daids=None, qnid=None, dnid_list=None,
                 annot_score_list=None, unique_nids=None,
                 name_score_list=None):
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
