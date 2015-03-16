"""
Algorithm and behavior configurations are stored here.  These classes are based
off of the utool.Preference.Pref class which really needs a good overhaul
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import six
import copy
from os.path import join
from os.path import splitext
from six.moves import zip, map, range, filter  # NOQA
from ibeis import constants as const
from utool._internal.meta_util_six import get_funcname
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[cfg]')

#ConfigBase = ut.DynStruct
#ConfigBase = object
ConfigBase = ut.Pref


def parse_config_items(cfg):
    """
    Recursively extracts key, val pairs from Config objects
    into a flat list. (there must not be name conflicts)

    Example:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> cfg = ibs.cfg.query_cfg
        >>> param_list = parse_config_items(cfg)
    """
    import ibeis
    param_list = []
    seen = set([])
    for item in cfg.items():
        key, val = item
        if isinstance(val, ibeis.model.Config.ConfigBase):
            child_cfg = val
            param_list.extend(parse_config_items(child_cfg))
            #print(key)
            pass
        elif key.startswith('_'):
            #print(key)
            pass
        else:
            if key in seen:
                print('[Config] WARNING: key=%r appears more than once' % (key,))
            seen.add(key)
            param_list.append(item)
            #print(key)
    return param_list


def make_config_metaclass():
    """
    Creates a metaclass for Config objects that automates some of the more
    tedious functions to write

    Like:
        get_cfgstr
        and the comparison methods
    """
    methods_list = ut.get_comparison_methods()

    # Decorator for functions that we will inject into our metaclass
    def _register(func):
        methods_list.append(func)
        return func

    @_register
    def get_cfgstr_list(cfg, **kwargs):
        """ default get_cfgstr_list, can be overrided by a config object """
        if hasattr(cfg, 'get_param_info_list'):
            itemstr_list = [pi.get_itemstr(cfg) for pi in cfg.get_param_info_list()]
        else:
            item_list = parse_config_items(cfg)
            itemstr_list = [key + '=' + str(val) for key, val in item_list]
        filtered_itemstr_list = list(filter(len, itemstr_list))
        config_name = cfg.get_config_name()
        return ['_' + config_name , '(' + ','.join(filtered_itemstr_list) + ')']

    @_register
    def parse_items(cfg, **kwargs):
        return parse_config_items(cfg)

    @_register
    def get_config_name(cfg, **kwargs):
        """ the user might want to overwrite this function """
        class_str = str(cfg.__class__)
        full_class_str = class_str.replace('<class \'', '').replace('\'>', '')
        config_name = splitext(full_class_str)[1][1:].replace('Config', '')
        return config_name
        #return 'METACONFIG'

    @_register
    def __hash__(cfg):
        """ Needed for comparison operators """
        return hash(cfg.get_cfgstr())

    @_register
    def get_cfgstr(cfg, **kwargs):
        return ''.join(cfg.get_cfgstr_list(**kwargs))

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
            #assert 'get_cfgstr_list' in dct, 'must have defined get_cfgstr_list.  name=%r' % (name,)
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


@six.add_metaclass(ConfigMetaclass)
class GenericConfig(ConfigBase):
    def __init__(cfg, *args, **kwargs):
        super(GenericConfig, cfg).__init__(*args, **kwargs)

    #def get_cfgstr_list(cfg, **kwargs):
    #    #raise NotImplementedError('abstract')
    #    item_list = parse_config_items(cfg)
    #    return ['GENERIC(' + ','.join([key + '=' + str(val) for key, val in item_list]) + ')']
    #    #return ['unimplemented']
    #    #pass

    #@abstract():
    #def get_cfgstr_list(cfg, **kwargs):


@six.add_metaclass(ConfigMetaclass)
class NNConfig(ConfigBase):
    def __init__(nn_cfg, **kwargs):
        super(NNConfig, nn_cfg).__init__()
        nn_cfg.K = 4
        #nn_cfg.min_reindex_thresh = 3  # 200  # number of annots before a new multi-indexer is built
        nn_cfg.min_reindex_thresh = 200  # number of annots before a new multi-indexer is built
        nn_cfg.max_subindexers = 2  # number of annots before a new multi-indexer is built
        nn_cfg.valid_index_methods = ['single', 'multi', 'name']
        nn_cfg.index_method = 'multi'
        nn_cfg.index_method = 'single'
        nn_cfg.Knorm = 1
        nn_cfg.checks = 800
        nn_cfg.normalizer_rule = ['last', 'name'][0]
        nn_cfg.update(**kwargs)

    def make_feasible(nn_cfg):
        # normalizer rule depends on Knorm
        assert nn_cfg.index_method in nn_cfg.valid_index_methods
        if isinstance(nn_cfg.Knorm, int) and nn_cfg.Knorm == 1:
            nn_cfg.normalizer_rule = 'last'

    def get_cfgstr_list(nn_cfg, **kwargs):
        nn_cfgstr  = ['_NN(',
                      nn_cfg.index_method,
                      ',K', str(nn_cfg.K),
                      '+', str(nn_cfg.Knorm),
                      ',', nn_cfg.normalizer_rule,
                      ',cks', str(nn_cfg.checks),
                      ')']
        return nn_cfgstr


@six.add_metaclass(ConfigMetaclass)
class SpatialVerifyConfig(ConfigBase):
    """
    Spatial verification
    """
    def __init__(sv_cfg, **kwargs):
        super(SpatialVerifyConfig, sv_cfg).__init__(name='sv_cfg')
        tau = 6.28  # 318530
        sv_cfg.sv_on = True
        sv_cfg.xy_thresh = .01
        sv_cfg.scale_thresh = 2.0
        sv_cfg.ori_thresh   = tau / 4.0
        sv_cfg.min_nInliers = 4
        sv_cfg.nNameShortlistSVER = 50
        sv_cfg.nAnnotPerNameSVER = 6
        #sv_cfg.prescore_method = 'csum'
        sv_cfg.prescore_method = 'nsum'
        sv_cfg.use_chip_extent = False  # BAD CONFIG?
        sv_cfg.sver_weighting = False  # weight feature scores with sver errors
        sv_cfg.update(**kwargs)

    def get_cfgstr_list(sv_cfg, **kwargs):
        if not sv_cfg.sv_on or sv_cfg.xy_thresh is None:
            return ['_SV(OFF)']
        thresh_tup = (sv_cfg.xy_thresh, sv_cfg.scale_thresh, sv_cfg.ori_thresh)
        thresh_str = ut.remove_chars(str(thresh_tup), ' ()').replace(',', ';')
        sv_cfgstr = [
            '_SV(',
            thresh_str,
            'minIn=%d,' % (sv_cfg.min_nInliers,),
            'nRR=%d,' % (sv_cfg.nNameShortlistSVER,),
            'nRR=%d,' % (sv_cfg.nNameShortlistSVER,),
            sv_cfg.prescore_method, ',',
            'cdl,' * sv_cfg.use_chip_extent,  # chip diag len
            '+w,' * sv_cfg.sver_weighting,  # chip diag len
            ')',
        ]
        return sv_cfgstr


@six.add_metaclass(ConfigMetaclass)
class AggregateConfig(ConfigBase):
    """
    Old Agg Cfg
    """
    def __init__(agg_cfg, **kwargs):
        super(AggregateConfig, agg_cfg).__init__(name='agg_cfg')
        # chipsum, namesum, placketluce
        #agg_cfg.score_method = 'csum'
        agg_cfg.score_method = 'nsum'
        agg_cfg.score_normalization = False
        #agg_cfg.score_normalization = True
        alt_methods = {
            'topk': 'topk',
            'borda': 'borda',
            'placketluce': 'pl',
            'chipsum': 'csum',
            'namesum': 'nsum',
            'coverage': 'coverage',
        }
        # For Placket-Luce
        agg_cfg.max_alts = 50
        #-----
        # User update
        agg_cfg.update(**kwargs)
        # ---
        key = agg_cfg.score_method.lower()
        # Use w as a toggle for weighted mode
        # Sanatize the scoring method
        if key in alt_methods:
            agg_cfg.score_method = alt_methods[key]

    def get_cfgstr_list(agg_cfg, **kwargs):
        agg_cfgstr = []
        agg_cfgstr.append('_AGG(')
        agg_cfgstr.append(agg_cfg.score_method)
        if agg_cfg.score_method  == 'pl':
            agg_cfgstr.append(',%d' % (agg_cfg.max_alts,))
        if agg_cfg.score_normalization:
            agg_cfgstr.append(',norm')
        agg_cfgstr.append(')')
        return agg_cfgstr


@six.add_metaclass(ConfigMetaclass)
class FlannConfig(ConfigBase):
    """

    this flann is only for neareset neighbors in vsone/many
    TODO: this might not need to be here

    References:
        http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_pami2014.pdf
        http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf
        http://docs.opencv.org/trunk/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html
    """
    def __init__(flann_cfg, **kwargs):
        super(FlannConfig, flann_cfg).__init__(name='flann_cfg')
        #General Params
        flann_cfg.algorithm = 'kdtree'  # linear
        flann_cfg.flann_cores = 0  # doesnt change config, just speed
        # KDTree params
        flann_cfg.trees = 8
        # KMeansTree params
        flann_cfg.iterations = 11
        flann_cfg.centers_init = 'random'
        flann_cfg.cb_index = .4
        flann_cfg.branching = 64
        flann_cfg.update(**kwargs)

    def get_flann_params(flann_cfg):
        flann_params = dict(
            algorithm=flann_cfg.algorithm,
            trees=flann_cfg.trees,
            cores=flann_cfg.flann_cores,
        )
        return flann_params

    def get_cfgstr_list(flann_cfg, **kwargs):
        flann_cfgstrs = ['_FLANN(']
        if flann_cfg.algorithm == 'kdtree':
            flann_cfgstrs += ['%d_kdtrees' % flann_cfg.trees]
        elif flann_cfg.algorithm == 'linear':
            flann_cfgstrs += ['%s' % flann_cfg.algorithm]
        elif flann_cfg.algorithm == 'kdtree':
            flann_cfgstrs += [
                '%s_' % flann_cfg.algorithm,
                'iter=%s_' % flann_cfg.iterations,
                'cb=%s_' % flann_cfg.cb_index,
                'branch=%s' % flann_cfg.branching,
            ]
        else:
            flann_cfgstrs += ['%s' % flann_cfg.algorithm]
        #flann_cfgstrs += ['checks=%r' % flann_cfg.checks]
        flann_cfgstrs += [')']
        return flann_cfgstrs


@six.add_metaclass(ConfigMetaclass)
class SMKConfig(ConfigBase):
    """

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.Config import *  # NOQA
        >>> smk_cfg = SMKConfig()
        >>> result1 = smk_cfg.get_cfgstr()
        >>> print(result1)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> smk_cfg = ibs.cfg.query_cfg.smk_cfg
        >>> smk_cfg.printme3()
    """
    def __init__(smk_cfg, **kwargs):
        super(SMKConfig, smk_cfg).__init__(name='smk_cfg')
        smk_cfg.smk_thresh = 0.0  # tau in the paper
        smk_cfg.smk_alpha  = 3.0
        smk_cfg.smk_aggregate  = False
        # TODO Separate into vocab config
        smk_cfg._valid_vocab_weighting = ['idf', 'negentropy']
        smk_cfg.vocab_weighting = 'idf'
        smk_cfg.allow_self_match = False
        smk_cfg.vocabtrain_cfg = VocabTrainConfig(**kwargs)
        smk_cfg.vocabassign_cfg = VocabAssignConfig(**kwargs)
        smk_cfg.update(**kwargs)

    def make_feasible(smk_cfg):

        hasvalid_weighting = any([
            smk_cfg.vocab_weighting == x
            for x in smk_cfg._valid_vocab_weighting])
        assert hasvalid_weighting, 'invalid vocab weighting %r' % smk_cfg.vocab_weighting

    def get_cfgstr_list(smk_cfg, **kwargs):
        smk_cfgstr_list = [
            '_SMK(',
            'agg=', str(smk_cfg.smk_aggregate),
            ',t=', str(smk_cfg.smk_thresh),
            ',a=', str(smk_cfg.smk_alpha),
            ',SelfOk' if smk_cfg.allow_self_match else '',
            ',%s' % smk_cfg.vocab_weighting,
            ')',
        ]
        smk_cfgstr_list.extend(smk_cfg.vocabassign_cfg.get_cfgstr_list())
        smk_cfgstr_list.extend(smk_cfg.vocabtrain_cfg.get_cfgstr_list())
        return smk_cfgstr_list


@six.add_metaclass(ConfigMetaclass)
class VocabTrainConfig(ConfigBase):
    """

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.Config import *  # NOQA
        >>> vocabtrain_cfg = VocabTrainConfig()
        >>> result = vocabtrain_cfg.get_cfgstr()
        >>> print(result)

    """
    def __init__(vocabtrain_cfg, **kwargs):
        super(VocabTrainConfig, vocabtrain_cfg).__init__(name='vocabtrain_cfg')
        vocabtrain_cfg.override_vocab = 'default'  # Vocab
        vocabtrain_cfg.vocab_taids = 'all'  # Vocab
        vocabtrain_cfg.nWords = int(8E3)  #
        vocabtrain_cfg.vocab_init_method = 'akmeans++'
        vocabtrain_cfg.vocab_nIters = 128
        vocabtrain_cfg.vocab_flann_params = dict(cores=0)  # TODO: easy flann params cfgstr
        vocabtrain_cfg.update(**kwargs)

    def get_cfgstr_list(vocabtrain_cfg, **kwargs):
        if vocabtrain_cfg.override_vocab == 'default':
            if isinstance(vocabtrain_cfg.vocab_taids, six.string_types):
                taids_cfgstr = 'taids=%s' % vocabtrain_cfg.vocab_taids
            else:
                taids_cfgstr = ut.hashstr_arr(vocabtrain_cfg.vocab_taids, 'taids', hashlen=8)
            vocabtrain_cfg_list = [
                '_VocabTrain(',
                'nWords=%d' % (vocabtrain_cfg.nWords,),
                ',init=', str(vocabtrain_cfg.vocab_init_method),
                ',nIters=%d,' % int(vocabtrain_cfg.vocab_nIters),
                taids_cfgstr,
                ')',
            ]
        else:
            vocabtrain_cfg_list = ['_VocabTrain(override=%s)' %
                                   (vocabtrain_cfg.override_vocab,)]
        return vocabtrain_cfg_list


@six.add_metaclass(ConfigMetaclass)
class VocabAssignConfig(ConfigBase):
    """

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.Config import *  # NOQA
        >>> vocabassign_cfg = VocabAssignConfig()
        >>> result = vocabassign_cfg.get_cfgstr()
        >>> print(result)
    """
    def __init__(vocabassign_cfg, **kwargs):
        super(VocabAssignConfig, vocabassign_cfg).__init__(name='vocabassign_cfg')
        vocabassign_cfg.nAssign = 10  # MultiAssignment
        vocabassign_cfg.massign_equal_weights = True
        vocabassign_cfg.massign_alpha = 1.2
        vocabassign_cfg.massign_sigma = 80.0
        vocabassign_cfg.update(**kwargs)

    def make_feasible(vocabassign_cfg):
        assert vocabassign_cfg.nAssign > 0, 'cannot assign to nothing'
        if vocabassign_cfg.nAssign == 1:
            # No point to multiassign weights if nAssign is 1
            vocabassign_cfg.massign_equal_weights = True

        if vocabassign_cfg.massign_equal_weights:
            # massign sigma makes no difference if there are equal weights
            vocabassign_cfg.massign_sigma = None

    def get_cfgstr_list(vocabassign_cfg, **kwargs):
        vocabassign_cfg_list = [
            '_VocabAssign(',
            'nAssign=', str(vocabassign_cfg.nAssign),
            ',a=', str(vocabassign_cfg.massign_alpha),
            ',s=', str(vocabassign_cfg.massign_sigma) if vocabassign_cfg.massign_equal_weights else '',
            ',eqw=T' if vocabassign_cfg.massign_equal_weights else ',eqw=F',
            ')',
        ]
        return vocabassign_cfg_list


@six.add_metaclass(ConfigMetaclass)
class NNWeightConfig(ConfigBase):
    r"""
    CommandLine:
        python -m ibeis.model.Config --test-NNWeightConfig

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.Config import *  # NOQA
        >>> cfg_list = [
        ...     NNWeightConfig(),
        ...     NNWeightConfig(can_match_sameimg=True, can_match_samename=False),
        ...     NNWeightConfig(ratio_thresh=.625, lnbnn_on=False)
        ... ]
        >>> result = '\n'.join([cfg.get_cfgstr() for cfg in cfg_list])
        >>> print(result)
        NNWeight(lnbnn,fg)
        NNWeight(lnbnn,fg,sameimg,nosamename)
        NNWeight(ratio_thresh=0.625,fg)
    """
    def __init__(nnweight_cfg, **kwargs):
        super(NNWeightConfig, nnweight_cfg).__init__(name='nnweight_cfg')
        for pi in nnweight_cfg.get_param_info_list():
            setattr(nnweight_cfg, pi.varname, pi.default)
        nnweight_cfg.update(**kwargs)

    #def get_config_name(nnweight_cfg):
    #    return 'NNWeight'

    def get_param_info_list(rrvsone_cfg):
        # new way to try and specify config options.
        # not sure if i like it yet
        param_info_list = ut.flatten([
            [
                ut.ParamInfo('ratio_thresh', None, type_=float, hideif=None),
                ut.ParamInfoBool('lnbnn_on', True,  hideif=False),
                #ut.ParamInfoBool('lograt_on', False, hideif=False),
                #ut.ParamInfoBool('logdist_on', False,  hideif=False),
                #ut.ParamInfoBool('dist_on', False,  hideif=False),
                #ut.ParamInfoBool('normonly_on', False,  hideif=False),
                #ut.ParamInfoBool('loglnbnn_on', False,  hideif=False),
                ut.ParamInfoBool('cos_on', False,  hideif=False),
                ut.ParamInfoBool('fg_on', True, hideif=False),
                #
                ut.ParamInfoBool('can_match_sameimg', False,  'sameimg', hideif=False),
                ut.ParamInfoBool('can_match_samename', True, 'samename', hideif=True),
            ],
        ])
        return param_info_list


@six.add_metaclass(ConfigMetaclass)
class RerankVsOneConfig(ConfigBase):
    """
    CommandLine:
        python -m ibeis.model.Config --test-RerankVsOneConfig

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.Config import *  # NOQA
        >>> rrvsone_cfg = RerankVsOneConfig(rrvsone_on=True)
        >>> result = rrvsone_cfg.get_cfgstr()
        >>> assert result.startswith('_RRVsOne(True,')

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.Config import *  # NOQA
        >>> rrvsone_cfg = RerankVsOneConfig(rrvsone_on=False)
        >>> result = rrvsone_cfg.get_cfgstr()
        >>> print(result)
        _RRVsOne(False)

    """
    def __init__(rrvsone_cfg, **kwargs):
        super(RerankVsOneConfig, rrvsone_cfg).__init__(name='rrvsone_cfg')
        for pi in rrvsone_cfg.get_param_info_list():
            setattr(rrvsone_cfg, pi.varname, pi.default)
        rrvsone_cfg.update(**kwargs)

    def get_config_name(rrvsone_cfg):
        return 'RRVsOne'

    def get_param_info_list(rrvsone_cfg):
        from ibeis.model.hots import distinctiveness_normalizer
        from ibeis.model.hots import vsone_pipeline
        # new way to try and specify config options.
        # not sure if i like it yet
        param_info_list = ut.flatten([
            [
                ut.ParamInfo('rrvsone_on', False, ''),
            ],
            vsone_pipeline.OTHER_RRVSONE_PARAMS.aslist(),
            vsone_pipeline.SHORTLIST_DEFAULTS.aslist(),
            vsone_pipeline.COEFF_DEFAULTS.aslist(),
            vsone_pipeline.UNC_DEFAULTS.aslist(),
            vsone_pipeline.SCR_DEFAULTS.aslist(),
            vsone_pipeline.COVKPTS_DEFAULT.aslist(
                hideif=lambda cfg: cfg['covscore_on'] and cfg['maskscore_mode'] != 'kpts'),
            vsone_pipeline.COVGRID_DEFAULT.aslist(
                hideif=lambda cfg: cfg['covscore_on'] and cfg['maskscore_mode'] != 'grid'),
            distinctiveness_normalizer.DCVS_DEFAULT.aslist(),
        ])
        return param_info_list

    def get_constraint_func():
        # TODO:
        def constraint_func(cfg):
            if cfg['rrvsone_on']:
                return False
            if cfg['use_gridcov_scoring'] and cfg['use_kptscov_scoring']:
                return False

    def get_cfgstr_list(rrvsone_cfg, **kwargs):
        if rrvsone_cfg.rrvsone_on:
            rrvsone_cfg_list = rrvsone_cfg.meta_get_cfgstr_list(**kwargs)
        else:
            rrvsone_cfg_list = [
                '_RRVsOne(',
                str(rrvsone_cfg.rrvsone_on),
                ')'
            ]
        return rrvsone_cfg_list


@six.add_metaclass(ConfigMetaclass)
class QueryConfig(ConfigBase):
    """
    query configuration parameters

    Example:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> cfg = ibs.cfg.query_cfg
        >>> cfgstr = ibs.cfg.query_cfg.get_cfgstr()
        >>> print(cfgstr)

    """
    def __init__(query_cfg, **kwargs):
        super(QueryConfig, query_cfg).__init__(name='query_cfg')
        query_cfg.nn_cfg         = NNConfig(**kwargs)
        query_cfg.nnweight_cfg   = NNWeightConfig(**kwargs)
        query_cfg.sv_cfg         = SpatialVerifyConfig(**kwargs)
        query_cfg.agg_cfg        = AggregateConfig(**kwargs)
        query_cfg.flann_cfg      = FlannConfig(**kwargs)
        query_cfg.smk_cfg        = SMKConfig(**kwargs)
        query_cfg.rrvsone_cfg    = RerankVsOneConfig(**kwargs)
        # causes some bug in Preference widget if these don't have underscore
        query_cfg._featweight_cfg = FeatureWeightConfig(**kwargs)
        query_cfg.use_cache = False
        # Start of pipeline
        query_cfg._valid_pipeline_roots = ['vsmany', 'vsone', 'smk']
        query_cfg.pipeline_root = 'vsmany'
        # <Hack Paramaters>
        query_cfg.with_metadata = False
        query_cfg.augment_queryside_hack = False
        query_cfg.return_expanded_nns = False  # for hacky distinctivness
        query_cfg.use_external_distinctiveness = False  # for distinctivness model
        query_cfg.codename = 'None'
        query_cfg.species_code = '____'  # TODO: make use of this
        # </Hack Paramaters>
        #if ut.is_developer():
        #    query_cfg.pipeline_root = 'smk'
        # Depends on feature config
        query_cfg.update_query_cfg(**kwargs)
        if ut.VERYVERBOSE:
            print('[config] NEW QueryConfig')

    def get_cfgstr_list(query_cfg, **kwargs):
        # Ensure feasibility of the configuration
        query_cfg.make_feasible()

        # Build cfgstr
        cfgstr_list = ['_' + query_cfg.pipeline_root ]
        if str(query_cfg.pipeline_root) == 'smk':
            # SMK Parameters
            if kwargs.get('use_smk', True):
                cfgstr_list += query_cfg.smk_cfg.get_cfgstr_list(**kwargs)
            if kwargs.get('use_sv', True):
                cfgstr_list += query_cfg.sv_cfg.get_cfgstr_list(**kwargs)
        elif str(query_cfg.pipeline_root) == 'vsmany' or str(query_cfg.pipeline_root) == 'vsone':
            # Naive Bayes Parameters
            if kwargs.get('use_nn', True):
                cfgstr_list += query_cfg.nn_cfg.get_cfgstr_list(**kwargs)
            if kwargs.get('use_nnweight', True):
                cfgstr_list += query_cfg.nnweight_cfg.get_cfgstr_list(**kwargs)
            if kwargs.get('use_sv', True):
                cfgstr_list += query_cfg.sv_cfg.get_cfgstr_list(**kwargs)
            if kwargs.get('use_agg', True):
                cfgstr_list += query_cfg.agg_cfg.get_cfgstr_list(**kwargs)
            if kwargs.get('use_flann', True):
                cfgstr_list += query_cfg.flann_cfg.get_cfgstr_list(**kwargs)
            if kwargs.get('use_rrvsone', True):
                cfgstr_list += query_cfg.rrvsone_cfg.get_cfgstr_list(**kwargs)
        else:
            raise AssertionError('bad pipeline root: ' + str(query_cfg.pipeline_root))
        if kwargs.get('use_featweight', True):
            cfgstr_list += query_cfg._featweight_cfg.get_cfgstr_list(**kwargs)

        if query_cfg.augment_queryside_hack:
            # HACK
            cfgstr_list += ['_HACK(augment_queryside)']
        return cfgstr_list

    def update_query_cfg(query_cfg, **cfgdict):
        # Each config paramater should be unique
        # So updating them all should not cause conflicts
        # FIXME: Should be able to infer all the children that need updates
        #
        # apply codename before updating subconfigs
        query_cfg.apply_codename(cfgdict.get('codename', None))
        # update subconfigs
        query_cfg.nn_cfg.update(**cfgdict)
        query_cfg.nnweight_cfg.update(**cfgdict)
        query_cfg.sv_cfg.update(**cfgdict)
        query_cfg.agg_cfg.update(**cfgdict)
        query_cfg.flann_cfg.update(**cfgdict)
        query_cfg.smk_cfg.update(**cfgdict)
        query_cfg.smk_cfg.vocabassign_cfg.update(**cfgdict)
        query_cfg.smk_cfg.vocabtrain_cfg.update(**cfgdict)
        query_cfg.rrvsone_cfg.update(**cfgdict)
        query_cfg._featweight_cfg.update(**cfgdict)
        query_cfg._featweight_cfg._feat_cfg.update(**cfgdict)
        query_cfg._featweight_cfg._feat_cfg._chip_cfg.update(**cfgdict)
        query_cfg.update(**cfgdict)
        # Ensure feasibility of the configuration
        try:
            query_cfg.make_feasible()
        except AssertionError as ex:
            print(ut.dict_str(cfgdict, sorted_=True))
            ut.printex(ex)
            raise

    def apply_codename(query_cfg, codename=None):
        """
        codenames denote mass changes to configurations
        it is a hacky solution to setting different parameter
        values all at once.
        """
        if codename is None:
            codename = query_cfg.codename

        nnweight_cfg = query_cfg.nnweight_cfg
        nn_cfg   = query_cfg.nn_cfg
        agg_cfg = query_cfg.agg_cfg

        if codename.startswith('csum') or codename.endswith('_csum'):
            raise NotImplementedError('codename nsum')
        if codename.startswith('nsum'):
            raise NotImplementedError('codename nsum')
        if codename.startswith('vsmany'):
            query_cfg.pipeline_root = 'vsmany'
        elif codename.startswith('vsone'):
            query_cfg.pipeline_root = 'vsone'
            nn_cfg.K = 1
            nn_cfg.Knorm = 1
            nnweight_cfg.lnbnn_on = False
            #nnweight_cfg.ratio_thresh = 1.6
            if codename.endswith('_dist') or '_dist_' in codename:
                # no ratio use distance
                nnweight_cfg.ratio_thresh = None
                nnweight_cfg.dist_on = True
            else:
                nnweight_cfg.ratio_thresh = .625
            if '_ratio' in codename:
                nnweight_cfg.ratio_thresh = .625
            if '_extern_distinctiveness' in codename:
                query_cfg.use_external_distinctiveness = True
            if codename.startswith('vsone_unnorm'):
                agg_cfg.score_normalization = False
            elif codename.startswith('vsone_norm'):
                agg_cfg.score_normalization = True
        elif codename.startswith('asmk'):
            query_cfg.pipeline_root = 'asmk'
        elif codename.startswith('smk'):
            query_cfg.pipeline_root = 'smk'
        elif codename == 'None':
            pass

    def make_feasible(query_cfg):
        try:
            query_cfg.make_feasible_()
        except AssertionError as ex:
            if ut.NOT_QUIET:
                query_cfg.printme3()
            ut.printex(ex, 'failed ot make feasible')
            raise

    def make_feasible_(query_cfg):
        """
        removes invalid parameter settings over all cfgs (move to QueryConfig)
        """
        nnweight_cfg = query_cfg.nnweight_cfg
        nn_cfg   = query_cfg.nn_cfg
        featweight_cfg = query_cfg._featweight_cfg
        #feat_cfg = query_cfg._featweight_cfg._feat_cfg
        smk_cfg = query_cfg.smk_cfg
        vocabassign_cfg = query_cfg.smk_cfg.vocabassign_cfg
        agg_cfg = query_cfg.agg_cfg
        sv_cfg = query_cfg.sv_cfg

        assert sv_cfg.prescore_method == agg_cfg.score_method, 'cannot be different yet.'

        if agg_cfg.score_normalization and query_cfg.pipeline_root == 'vsmany':
            assert agg_cfg.score_method == 'nsum'

        if query_cfg.pipeline_root == 'asmk':
            query_cfg.pipeline_root = 'smk'
            smk_cfg.smk_aggregate = True

        hasvalid_root = any([
            query_cfg.pipeline_root == root
            for root in query_cfg._valid_pipeline_roots])
        try:
            assert hasvalid_root, 'invalid pipeline root %r' % query_cfg.pipeline_root
        except AssertionError as ex:
            ut.printex(ex)
            if ut.SUPER_STRICT:
                raise
            else:
                query_cfg.pipeline_root = query_cfg._valid_pipeline_roots[0]
                pass

        if nnweight_cfg.fg_on is not True:
            featweight_cfg.featweight_enabled = False
        if featweight_cfg.featweight_enabled is not True:
            nnweight_cfg.fg_on = False

        vocabassign_cfg.make_feasible()
        smk_cfg.make_feasible()
        #nnweight_cfg.make_feasible()
        nn_cfg.make_feasible()

    def deepcopy(query_cfg, **kwargs):
        copy_ = copy.deepcopy(query_cfg)
        copy_.update_query_cfg(**kwargs)
        return copy_


@six.add_metaclass(ConfigMetaclass)
class FeatureWeightConfig(ConfigBase):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.Config import *  # NOQA
        >>> featweight_cfg = FeatureWeightConfig()
        >>> result = featweight_cfg.get_cfgstr()
        >>> print(result)
        _FEATWEIGHT(ON,uselabel,rf)_FEAT(hesaff+sift_)_CHIP(sz450)
    """

    def __init__(featweight_cfg, **kwargs):
        super(FeatureWeightConfig, featweight_cfg).__init__(name='featweight_cfg')
        # Featweights depend on features
        featweight_cfg._feat_cfg = FeatureConfig(**kwargs)
        # Feature weights depend on the detector, but we only need to mirror
        # some parameters because featweight_cfg should not use the detect_cfg
        # object
        featweight_cfg.featweight_enabled = True
        featweight_cfg.featweight_species  = 'uselabel'
        featweight_cfg.featweight_detector = 'rf'
        featweight_cfg.update(**kwargs)

    def make_feasible(featweight_cfg):
        #featweight_cfg.featweight_enabled = False
        pass

    def get_cfgstr_list(featweight_cfg, **kwargs):
        featweight_cfg.make_feasible()
        featweight_cfgstrs = []
        if kwargs.get('use_featweight', True):
            if featweight_cfg.featweight_enabled is not True:
                if featweight_cfg.featweight_enabled == 'ERR':
                    featweight_cfgstrs.extend(['_FEATWEIGHT(ERR)'])
                else:
                    featweight_cfgstrs.extend(['_FEATWEIGHT(OFF)'])
            else:
                featweight_cfgstrs.extend([
                    '_FEATWEIGHT(ON',
                    ',' + featweight_cfg.featweight_species,
                    ',' + featweight_cfg.featweight_detector,
                    ')'])
        featweight_cfgstrs.extend(featweight_cfg._feat_cfg.get_cfgstr_list(**kwargs))
        return featweight_cfgstrs


@six.add_metaclass(ConfigMetaclass)
class FeatureConfig(ConfigBase):
    """
    Feature configuration object.

    CommandLine:
        python -m ibeis.model.Config --test-FeatureConfig

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model import Config  # NOQA
        >>> from ibeis.model.Config import *  # NOQA
        >>> feat_cfg = Config.FeatureConfig()
        >>> result = (feat_cfg.get_cfgstr())
        >>> print(result)
        _FEAT(hesaff+sift_)_CHIP(sz450)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model import Config  # NOQA
        >>> from ibeis.model.Config import *  # NOQA
        >>> feat_cfg = Config.FeatureConfig(rotation_invariance=True)
        >>> result = (feat_cfg.get_cfgstr())
        >>> print(result)
        _FEAT(hesaff+sift_rotation_invariance=True)_CHIP(sz450)

    _FEAT(hesaff+sift_nScal=3,thrsh=5.33,edggn=10.00,nIter=16,cnvrg=0.05,intlS=1.60)_CHIP(sz450)
    """
    def __init__(feat_cfg, **kwargs):
        # Features depend on chips
        import pyhesaff
        super(FeatureConfig, feat_cfg).__init__(name='feat_cfg')
        feat_cfg._chip_cfg = ChipConfig(**kwargs)
        feat_cfg.feat_type = 'hesaff+sift'
        feat_cfg._param_list = list(six.iteritems(pyhesaff.get_hesaff_default_params()))

        for type_, name, default, doc in feat_cfg._iterparams():
            setattr(feat_cfg, name, default)

        #feat_cfg.affine_invariance = False  # 9001 # 80
        #feat_cfg.rotation_invariance = True  # 9001 # 80

        feat_cfg.use_adaptive_scale = False  # 9001 # 80
        feat_cfg.nogravity_hack = False  # 9001 # 80
        feat_cfg.update(**kwargs)

    def _iterparams(feat_cfg):
        """ DEPRICATE """
        for name, default in feat_cfg._param_list:
            type_ = None
            doc = None
            yield (type_, name, default, doc)

    def get_hesaff_params(feat_cfg):
        dict_args = {
            name: feat_cfg[name]
            for type_, name, default, doc in feat_cfg._iterparams()
        }
        return dict_args

    def get_cfgstr_list(feat_cfg, **kwargs):
        if kwargs.get('use_feat', True):
            import pyhesaff
            feat_cfgstrs = ['_FEAT(']
            feat_cfgstrs += [feat_cfg.feat_type]
            #feat_cfgstrs += [',%r_%r' % (feat_cfg.scale_min, feat_cfg.scale_max)]
            feat_cfgstrs += [',adaptive'] * feat_cfg.use_adaptive_scale
            feat_cfgstrs += [',nogravity'] * feat_cfg.nogravity_hack
            # TODO: Named Tuple
            alias = {
                'numberOfScales': 'nScales',
                'edgeEigValRat': 'EdgeEigvRat',
                'maxIterations': 'nIter',
            }
            ignore = []
            ignore_if_default = set(pyhesaff.get_hesaff_default_params().keys())
            def _gen():
                for param in feat_cfg._iterparams():
                    # a parameter is a type, name, default value, and docstring
                    (type_, name, default, doc) = param
                    if name in ignore:
                        continue
                    val = feat_cfg[name]
                    if name in ignore_if_default and val == default:
                        continue
                    if isinstance(val, float):
                        valstr = '%.2f' % val
                    else:
                        valstr = str(val)
                    #namestr = ut.hashstr(alias.get(name, name), hashlen=6,
                    #                        alphabet=ut.util_hash.ALPHABET_27)
                    #namestr = ut.clipstr(alias.get(name, name), 5)
                    namestr = alias.get(name, name)
                    str_ = namestr + '=' + valstr
                    yield str_
            feat_cfgstrs.append('_' + ',' .join(list(_gen())))
            feat_cfgstrs.append(')')
        else:
            feat_cfgstrs = []
        feat_cfgstrs.extend(feat_cfg._chip_cfg.get_cfgstr_list(**kwargs))
        return feat_cfgstrs


@six.add_metaclass(ConfigMetaclass)
class ChipConfig(ConfigBase):
    """ ChipConfig """
    def __init__(cc_cfg, **kwargs):
        super(ChipConfig, cc_cfg).__init__(name='chip_cfg')
        cc_cfg.chip_sqrt_area = 450
        cc_cfg.grabcut         = False
        cc_cfg.histeq          = False
        cc_cfg.adapteq         = False
        cc_cfg.region_norm     = False
        cc_cfg.rank_eq         = False
        cc_cfg.local_eq        = False
        cc_cfg.maxcontrast     = False
        cc_cfg.chipfmt         = '.png'
        cc_cfg.update(**kwargs)

    def get_cfgstr_list(cc_cfg, **kwargs):
        if kwargs.get('use_chip', True):
            chip_cfgstr = []
            #assert cc_cfg.chipfmt[0] == '.'
            chip_cfgstr += [cc_cfg.chipfmt[1:].lower()] * (cc_cfg.chipfmt != '.png')
            chip_cfgstr += ['histeq']  * cc_cfg.histeq
            chip_cfgstr += ['adapteq'] * cc_cfg.adapteq
            chip_cfgstr += ['grabcut'] * cc_cfg.grabcut
            chip_cfgstr += ['regnorm'] * cc_cfg.region_norm
            chip_cfgstr += ['rankeq']  * cc_cfg.rank_eq
            chip_cfgstr += ['localeq'] * cc_cfg.local_eq
            chip_cfgstr += ['maxcont'] * cc_cfg.maxcontrast
            isOrig = cc_cfg.chip_sqrt_area is None or cc_cfg.chip_sqrt_area  <= 0
            chip_cfgstr += ['szorig'] if isOrig else ['sz%r' % cc_cfg.chip_sqrt_area]
            chip_cfgstr_list = ['_CHIP(', (','.join(chip_cfgstr)), ')']
        else:
            chip_cfgstr_list = []
        return chip_cfgstr_list


@six.add_metaclass(ConfigMetaclass)
class DetectionConfig(ConfigBase):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.Config import *  # NOQA
        >>> detect_cfg = DetectionConfig()
        >>> result = (detect_cfg.get_cfgstr())
        >>> print(result)
        _DETECT(rf,zebra_grevys,sz=800)
    """
    def __init__(detect_cfg, **kwargs):
        super(DetectionConfig, detect_cfg).__init__(name='detect_cfg')
        detect_cfg.species_text     = const.Species.ZEB_GREVY
        detect_cfg.detector    = 'rf'
        # detect_cfg.scale_list  = '1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1'
        # detect_cfg.scale_list  = '1.15, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.10'
        detect_cfg.scale_list  = '1.25, 1.0, 0.80, 0.65, 0.50, 0.40, 0.30, 0.20, 0.10'
        detect_cfg.trees_path  = ''
        detect_cfg.detectimg_sqrt_area = 800
        detect_cfg.update(**kwargs)

    def get_cfgstr_list(detect_cfg):
        cfgstrs = ['_DETECT(',
                   detect_cfg.detector,
                   ',', detect_cfg.species_text,
                   ',sz=%d' % (detect_cfg.detectimg_sqrt_area,),
                   ')']
        return cfgstrs


@six.add_metaclass(ConfigMetaclass)
class EncounterConfig(ConfigBase):
    """ EncounterConfig """
    valid_cluster_algos = ['meanshift', 'agglomerative']

    def __init__(enc_cfg, **kwargs):
        super(EncounterConfig, enc_cfg).__init__(name='enc_cfg')
        enc_cfg.min_imgs_per_encounter = 1
        #enc_cfg.cluster_algo = 'meanshift'  # [agglomerative]
        enc_cfg.cluster_algo = 'agglomerative'
        enc_cfg.quantile = .01  # depends meanshift
        enc_cfg.seconds_thresh = 60    # depends agglomerative
        enc_cfg.use_gps = False

    def get_cfgstr_list(enc_cfg):
        enc_cfgstrs = []
        if enc_cfg.cluster_algo == 'meanshift':
            enc_cfgstrs.append('ms')
            enc_cfgstrs.append('quant_%r' % enc_cfg.quantile)
        elif enc_cfg.cluster_algo == 'agglomerative':
            enc_cfgstrs.append('agg')
            enc_cfgstrs.append('sec_%r' % enc_cfg.seconds_thresh)
        if enc_cfg.use_gps:
            enc_cfgstrs.append('gps')

        enc_cfgstrs.append(str(enc_cfg.min_imgs_per_encounter))
        return ['_ENC(', ','.join(enc_cfgstrs), ')']


@six.add_metaclass(ConfigMetaclass)
class DisplayConfig(ConfigBase):
    """ DisplayConfig """
    def __init__(display_cfg, **kwargs):
        super(DisplayConfig, display_cfg).__init__(name='display_cfg')
        display_cfg.N = 6
        display_cfg.name_scoring = False
        display_cfg.showanalysis = False
        display_cfg.annotations  = True
        display_cfg.vert = True  # None
        display_cfg.show_results_in_image = False  # None

    def get_cfgstr_list(nn_cfg):
        raise NotImplementedError('abstract')
        return ['unimplemented']


@six.add_metaclass(ConfigMetaclass)
class OtherConfig(ConfigBase):
    def __init__(other_cfg, **kwargs):
        super(OtherConfig, other_cfg).__init__(name='other_cfg')
        #other_cfg.thumb_size     = 128
        other_cfg.thumb_size     = 221
        other_cfg.ranks_lt       = 2
        other_cfg.auto_localize  = True
        # maximum number of exemplars per name
        other_cfg.max_exemplars  = 6
        other_cfg.exemplars_per_view  = 2
        other_cfg.exemplar_distinctiveness_thresh  = .95
        other_cfg.detect_add_after = 1
        # other_cfg.detect_use_chunks = True
        other_cfg.use_more_special_encounters = False
        other_cfg.location_for_names = 'IBEIS'
        #other_cfg.location_for_names = 'MUGU'
        other_cfg.smart_enabled = True
        other_cfg.update(**kwargs)

    #def get_cfgstr_list(nn_cfg):
    #    raise NotImplementedError('abstract')
    #    return ['unimplemented']


# Convinience
def __dict_default_func(dict_):
    # Sets keys only if they dont exist
    def set_key(key, val):
        if key not in dict_:
            dict_[key] = val
    return set_key


def default_vsone_cfg(ibs, **kwargs):
    # DEPRICATE
    kwargs['pipeline_root'] = 'vsone'
    ut.dict_update_newkeys(kwargs, {
        'lnbnn_on': False,
        'checks': 256, 'K': 1,
        'Knorm': 1,
        'ratio_thresh': .6666  # 1.5,
    })
    query_cfg = QueryConfig(**kwargs)
    return query_cfg


def set_query_cfg(cfg, query_cfg):
    """ hack 12-30-2014 """
    cfg.query_cfg = query_cfg
    cfg.featweight_cfg = cfg.query_cfg._featweight_cfg
    cfg.feat_cfg       = cfg.query_cfg._featweight_cfg._feat_cfg
    cfg.chip_cfg       = cfg.query_cfg._featweight_cfg._feat_cfg._chip_cfg


def update_query_config(cfg, **kwargs):
    """ hack 12-30-2014 """
    cfg.query_cfg.update_query_cfg(**kwargs)
    cfg.featweight_cfg = cfg.query_cfg._featweight_cfg
    cfg.feat_cfg       = cfg.query_cfg._featweight_cfg._feat_cfg
    cfg.chip_cfg       = cfg.query_cfg._featweight_cfg._feat_cfg._chip_cfg


def load_named_config(cfgname, dpath, use_config_cache=False):
    """ hack 12-30-2014

    Args:
        cfgname (str):
        dpath (str):
        use_config_cache (bool):

    Returns:
        Config: cfg

    CommandLine:
        python -m ibeis.model.Config --test-load_named_config

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.Config import *  # NOQA
        >>> from ibeis.model.Config import _default_config  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_Master0')
        >>> #ibs.cfg.save()
        >>> # build test data
        >>> cfgname = 'zebra_plains'
        >>> dpath = ibs.get_dbdir()
        >>> use_config_cache = True
        >>> # execute function
        >>> cfg = load_named_config(cfgname, dpath, use_config_cache)
        >>> #
        >>> keys1 = ut.get_list_column(cfg.parse_items(), 0)
        >>> keys2 = ut.get_list_column(ibs.cfg.parse_items(), 0)
        >>> symdiff = set(keys1) ^ set(keys2)
        >>> # verify results
        >>> result = str(cfg)
        >>> print(result)
    """
    if cfgname is None:
        # TODO: find last cfgname
        cfgname = 'cfg'
    fpath = join(dpath, cfgname) + '.cPkl'
    if not ut.QUIET:
        print('[Config] loading named config fpath=%r' % (fpath,))
    # Always a fresh object
    cfg = GenericConfig(cfgname, fpath=fpath)
    try:
        # Use pref cache
        if not use_config_cache:
            raise Exception('force config cache miss')
        # Get current "schema"
        #tmp = _default_config(cfg, cfgname, new=True)
        #current_itemset = tmp.parse_items()
        #current_keyset = list(ut.get_list_column(current_itemset, 0))
        # load saved preferences
        cfg.load()
        # Check if loaded schema has changed
        #loaded_keyset = list(ut.get_list_column(cfg.parse_items(), 0))
        #missing_keys = set(current_keyset) - set(loaded_keyset)
        #if len(missing_keys) != 0:
        #    missing_vals = ut.dict_take(dict(current_itemset), missing_keys)
        #    update_items = list(zip(missing_keys, missing_vals))
        #    # TODO: Finishme update the out of data preferences
        #    pass
        if ut.NOT_QUIET:
            print('[Config] successfully loaded config cfgname=%r' % (cfgname,))
    except Exception:
        # Totally new completely default preferences
        cfg = _default_config(cfg, cfgname)
    # Hack in cfgname
    cfg.z_cfgname = cfgname
    return cfg


def _default_config(cfg, cfgname=None, new=True):
    """ hack 12-30-2014 """
    if ut.VERBOSE:
        print('[Config] building default config')
    if cfgname is None:
        cfgname = cfg.z_cfgname
    if new:
        fpath = cfg.get_fpath()
        cfg = GenericConfig(cfgname, fpath=fpath)
        cfg.z_cfgname = cfgname
    query_cfg = QueryConfig(pipeline_root='vsmany')
    set_query_cfg(cfg, query_cfg)
    cfg.enc_cfg     = EncounterConfig()
    cfg.detect_cfg  = DetectionConfig()
    cfg.other_cfg   = OtherConfig()
    _default_named_config(cfg, cfgname)
    #if len(species_list) == 1:
    #    # try to be intelligent about the default speceis
    #    cfg.detect_cfg.species_text = species_list[0]
    return cfg


def _default_named_config(cfg, cfgname):
    """ hack 12-30-2014

    list default parameters per species

    """
    Species = const.Species
    if cfgname == 'cfg':
        cfg.detect_cfg.species_text = 'none'
    elif cfgname == Species.ZEB_PLAIN:
        cfg.detect_cfg.species_text = cfgname
        #speedup': 46.90769958496094,
        cfg.query_cfg.flann_cfg.algorithm = 'kdtree'
        cfg.query_cfg.flann_cfg.trees = 8
        cfg.query_cfg.nn_cfg.checks = 704
        #'algorithm': 'kdtree',
        #[dev.tune_flann]    'checks': 6656,
        #[dev.tune_flann]    'trees': 4,

        # Kmeans seems a bit more accurate
        #'algorithm': 'kmeans',
        #'branching': 16,
        #'cb_index': 0.6000000238418579,
        #'centers_init': 'random',
        #'checks': 18432,
        #'iterations': 1,
        #'leaf_max_size': 4,
        #'speedup': 65.54280090332031,
        #'target_precision': 0.9800000190734863,

    elif cfgname == Species.ZEB_GREVY:
        cfg.detect_cfg.species_text = cfgname
        #speedup': 224.7425994873047,
        cfg.query_cfg.flann_cfg.algorithm = 'kdtree'
        cfg.query_cfg.flann_cfg.trees = 4
        cfg.query_cfg.nn_cfg.checks = 896
    elif cfgname == Species.GIRAFFE:
        cfg.detect_cfg.species_text = cfgname
        cfg.query_cfg.flann_cfg.algorithm = 'kdtree'
        cfg.query_cfg.flann_cfg.trees = 8
        cfg.query_cfg.nn_cfg.checks = 316
    else:
        print('WARNING: UNKNOWN CFGNAME=%r' % (cfgname,))


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, ibeis.model.Config; utool.doctest_funcs(ibeis.model.Config, allexamples=True)"
        python -c "import utool, ibeis.model.Config; utool.doctest_funcs(ibeis.model.Config)"
        python -m ibeis.model.Config
        python -m ibeis.model.Config --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
