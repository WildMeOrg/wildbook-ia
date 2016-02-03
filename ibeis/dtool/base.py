from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
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
        return [pi.varname for pi in self.get_param_info_list()] + self._subconfig_attrs

    @_register
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
        return '_'.join([str_] + [cfg[subcfg_attr].get_cfgstr()
                                  for subcfg_attr in cfg._subconfig_attrs])

    @_register
    def get_hashid(cfg):
        return ut.hashstr27(cfg.get_cfgstr())

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

    """
    _isnewreq = True

    @classmethod
    def new_algo_request(cls, depc, algoname, qaids, daids, cfgdict=None):
        self = cls()
        self._qaids = None
        self._daids = None
        self.depc = depc
        self.qaids = qaids
        self.daids = daids
        if cfgdict is None:
            cfgdict = {}
        configclass = depc.configclass_dict[algoname]
        config = configclass(**cfgdict)

        self.config = config
        self.algoname = algoname

        # hack
        self.params = dict(config.parse_items())
        return self

    @property
    def ibs(self):
        if self.depc is None:
            return None
        return self.depc.controller

    def get_external_data_config2(self):
        # HACK
        #return None
        #print('[d] self.params = %r' % (self.params,))
        return self.params

    def get_external_query_config2(self):
        # HACK
        #return None
        #print('[q] self.params = %r' % (self.params,))
        return self.params

    @property
    def qaids(self):
        return self._qaids

    @qaids.setter
    def qaids(self, qaids):
        self._qaids = safeop(np.array, qaids)

    @property
    def daids(self):
        return self._daids

    @daids.setter
    def daids(self, daids):
        self._daids = safeop(np.array, daids)

    def execute(req, qaids=None, use_cache=None):
        if qaids is not None:
            qaids = [qaids] if not ut.isiterable(qaids) else qaids
            subreq = req.shallowcopy(qaids=qaids)
            return subreq.execute(use_cache=True)
        else:
            tablename = req.algoname
            table = req.depc[tablename]
            if use_cache is None:
                use_cache = not ut.get_argflag('--nocache')

            rowids = table.get_rowid(list(zip(req.qaids)), req, recompute=not use_cache)

            result_list = table.get_row_data(rowids)
            return ut.get_list_column(result_list, 0)

    def get_query_hashid(self):
        return self._get_rootset_hashid(self.qaids, 'Q')

    def get_data_hashid(self):
        return self._get_rootset_hashid(self.daids, 'D')

    def _get_rootset_hashid(self, root_rowids, preffix):
        uuid_type = 'V'
        label = ''.join((preffix, uuid_type, 'UUIDS'))
        uuid_list = self.depc.get_root_uuid(root_rowids)
        #uuid_hashid = ut.hashstr_arr27(uuid_list, label, pathsafe=True)
        uuid_hashid = ut.hashstr_arr27(uuid_list, label, pathsafe=False)
        return uuid_hashid

    def get_pipe_cfgstr(self):
        return self.config.get_cfgstr()

    def get_pipe_hashid(self):
        return ut.hashstr27(self.get_pipe_cfgstr())

    def get_cfgstr(req, with_query=False, with_data=True, with_pipe=True, hash_pipe=False):
        r"""
        main cfgstring used to identify the 'querytype'
        """
        cfgstr_list = []
        if with_query:
            cfgstr_list.append(req.get_query_hashid())
        if with_data:
            cfgstr_list.append(req.get_data_hashid())
        if with_pipe:
            if hash_pipe:
                cfgstr_list.append(req.get_pipe_hashid())
            else:
                cfgstr_list.append(req.get_pipe_cfgstr())
        cfgstr = '_'.join(cfgstr_list)
        return cfgstr

    def get_full_cfgstr(req):
        """ main cfgstring used to identify the algo hash id """
        full_cfgstr = req.get_cfgstr(with_query=True)
        return full_cfgstr

    def _custom_str(req):
        # typestr = ut.type_str(type(ibs)).split('.')[-1]
        typestr = req.__class__.__name__
        dbname = None if req.depc is None or req.depc.controller is None else req.depc.controller.get_dbname()
        # hashkw = dict(_new=True, pathsafe=False)
        # infostr_ = req.get_cfgstr(with_query=True, with_pipe=True, hash_pipe=True, hashkw=hashkw)
        infostr_ = 'nQ=%s, nD=%s %s' % (len(req.qaids), len(req.daids), req.get_pipe_hashid())
        custom_str = '<%s(%s) %s at %s>' % (typestr, dbname, infostr_, hex(id(req)))
        return custom_str

    def __repr__(req):
        return req._custom_str()

    def __str__(req):
        return req._custom_str()

    def __getstate__(req):
        state_dict = req.__dict__.copy()
        # SUPER HACK
        state_dict['dbdir'] = req.depc.controller.get_dbdir()
        del state_dict['depc']
        del state_dict['config']
        return state_dict

    def __setstate__(req, state_dict):
        import ibeis
        dbdir = state_dict['dbdir']
        del state_dict['dbdir']
        params = state_dict['params']
        depc = ibeis.opendb(dbdir=dbdir, web=False).depc
        configclass = depc.configclass_dict[state_dict['algoname'] ]
        config = configclass(**params)
        state_dict['depc'] = depc
        state_dict['config'] = config
        req.__dict__.update(state_dict)

    def shallowcopy(req, qmask=None, qaids=None):
        """
        Creates a copy of qreq with the same qparams object and a subset of the
        qx and dx objects.  used to generate chunks of vsone and vsmany queries
        """
        #subreq = copy.copy(req)  # copy calls setstate and getstate
        subreq = req.__class__()
        subreq.__dict__.update(req.__dict__)
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


class MatchResult(AlgoResult):
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

    def __repr__(cm):
        typestr = cm.__class__.__name__
        infostr_ = 'qaid=%s nD=%s' % (cm.qaid, cm.num_daids)
        return '<%s %s at %s>' % (typestr, infostr_, hex(id(cm)))

    def __str__(cm):
        typestr = cm.__class__.__name__
        infostr_ = 'qaid=%s nD=%s' % (cm.qaid, cm.num_daids)
        return '<%s %s>' % (typestr, infostr_)


@six.add_metaclass(ConfigMetaclass)
class Config(ut.NiceRepr, ut.DictLike):
    """ Base class for heirarchical config

    need to overwrite get_param_info_list

    """

    def __init__(self, **kwargs):
        self.initialize_params(**kwargs)

    def __nice__(self):
        return self.get_cfgstr(with_name=False)

    def keys(self):
        return self.get_varnames()

    def getitem(self, key):
        try:
            return getattr(self, key)
        except AttributeError as ex:
            raise KeyError(ex)

    def setitem(self, key, value):
        return getattr(self, key, value)

    def get_param_info_list(self):
        try:
            return self._param_info_list
        except AttributeError:
            raise NotImplementedError(
                'Need to define _param_info_list or get_param_info_list')

    #@classmethod
    #def static_config_name(cls):
    #    class_str = str(cls)
    #    full_class_str = class_str.replace('<class \'', '').replace('\'>', '')
    #    config_name = splitext(full_class_str)[1][1:].replace('Config', '')
    #    return config_name

    @classmethod
    def from_argv_dict(cls):
        """
        ut.parse_argv_cfg
        """
        self = cls()
        new_vals = ut.parse_dict_from_argv(self)
        self.update(**new_vals)
        return self

    @classmethod
    def from_argv_cfgs(cls):
        """
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

    def update(cfg, **kwargs):
        self_keys = set(cfg.__dict__.keys())
        for key, val in six.iteritems(kwargs):
            if key in self_keys:
                setattr(cfg, key, val)

    # @classmethod
    # def register_func


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
