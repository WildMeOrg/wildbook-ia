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
        return '_'.join([str_] + [cfg[subcfg_name].get_cfgstr()
                                  for subcfg_name in cfg._subconfig_names])

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
    @classmethod
    def new_algo_request(cls, depc, algoname, qaids, daids, cfgdict=None):
        self = cls()
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

    def execute(self):
        tablename = self.algoname
        table = self.depc[tablename]
        rowids = table.get_rowid(list(zip(self.qaids)), self)
        result_list = table.get_row_data(rowids)
        return ut.get_list_column(result_list, 0)

    def get_query_hashid(self):
        return self._get_rootset_hashid(self.qaids, 'Q')

    def get_data_hashid(self):
        return self._get_rootset_hashid(self.daids, 'D')

    def _get_rootset_hashid(self, root_rowids, preffix):
        uuid_type = 'S'
        label = ''.join((preffix, uuid_type, 'UUIDS'))
        uuid_list = self.depc.get_root_uuid(root_rowids)
        #uuid_hashid = ut.hashstr_arr27(uuid_list, label, pathsafe=True)
        uuid_hashid = ut.hashstr_arr27(uuid_list, label, pathsafe=False)
        return uuid_hashid

    def get_pipe_cfgstr(self):
        return self.config.get_cfgstr()

    def get_pipe_hashid(self):
        return ut.hashstr27(self.get_pipe_cfgstr())

    def get_cfgstr(qreq_, with_query=False, with_data=True, with_pipe=True, hash_pipe=False):
        r"""
        main cfgstring used to identify the 'querytype'
        """
        cfgstr_list = []
        if with_query:
            cfgstr_list.append(qreq_.get_query_hashid())
        if with_data:
            cfgstr_list.append(qreq_.get_data_hashid())
        if with_pipe:
            if hash_pipe:
                cfgstr_list.append(qreq_.get_pipe_hashstr())
            else:
                cfgstr_list.append(qreq_.get_pipe_cfgstr())
        cfgstr = '_'.join(cfgstr_list)
        return cfgstr

    def get_full_cfgstr(qreq_):
        """ main cfgstring used to identify the algo hash id """
        full_cfgstr = qreq_.get_cfgstr(with_query=True)
        return full_cfgstr


class AlgoResult(object):
    """ Base class for algo result objects """

    @classmethod
    def load_from_fpath(cls, fpath, verbose=None):
        state_dict = ut.load_cPkl(fpath, verbose=verbose)
        self = cls()
        self.__setstate__(state_dict)
        return self

    def save_to_fpath(cm, fpath, verbose=None):
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


class MatchResult(AlgoResult):
    def __init__(self, qaid=None, daids=None, qnid=None, dnid_list=None,
                 annot_score_list=None, unique_nids=None,
                 name_score_list=None):
        self.qaid = qaid
        self.daid_list = daids
        self.dnid_list = dnid_list
        self.annot_score_list = annot_score_list
        self.name_score_list = name_score_list

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
class TableConfig(ut.DictLike):
    """ Base class for heirarchical config """

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
    """ Base class for heirarchical config """
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
