from __future__ import absolute_import, division, print_function
import utool
import utool as ut
import six
import copy
from utool._internal.meta_util_six import get_funcname
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[cfg]')

#ConfigBase = utool.DynStruct
#ConfigBase = object
ConfigBase = utool.Pref


def make_config_metaclass():
    methods_list = utool.get_comparison_methods()

    def _register(func):
        methods_list.append(func)
        return func

    @_register
    def get_cfgstr_list(cfg, **kwargs):
        item_list = parse_config_items(cfg)
        return ['METACONFIG(' + ','.join([key + '=' + str(val) for key, val in item_list]) + ')']
        #return ['cfg']

    # Needed for comparison operators
    @_register
    def __hash__(cfg):
        return hash(cfg.get_cfgstr())

    @_register
    def get_cfgstr(cfg, **kwargs):
        return ''.join(cfg.get_cfgstr_list(**kwargs))

    class ConfigMetaclass(type):
        """ Defines extra methods for Configs
        """
        #print(dct)

        def __new__(cls, name, bases, dct):
            """cls - meta
            name - classname
            supers - bases
            dct - class dictionary
            """
            #assert 'get_cfgstr_list' in dct, 'must have defined get_cfgstr_list.  name=%r' % (name,)
            for func in methods_list:
                if get_funcname(func) not in dct:
                    funcname = get_funcname(func)
                    dct[funcname] = func
                #utool.inject_func_as_method(metaself, func)

            #return type(name, bases, dct)
            return type.__new__(cls, name, bases, dct)

    #def __init__(metaself, name, bases, dct):
    #    super(ConfigMetaclass, metaself).__init__(name, bases, dct)

    #    # Give the new class the registered methods
    #    #funcname = get_funcname(func)
    #    #setattr(metaself, funcname, func)

    #    #metaself.__cfgmetaclass__ = True
    return ConfigMetaclass

ConfigMetaclass = make_config_metaclass()


def parse_config_items(cfg):
    """
    Recursively extracts key, val pairs from Config objects
    into a flat list. (there must not be name conflicts)

    Example:
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
        nn_cfg.Knorm = 1
        nn_cfg.normalizer_rule = ['last', 'name'][0]
        nn_cfg.checks  = 1024  # 512#128
        nn_cfg.update(**kwargs)

    def make_feasible(nn_cfg):
        # normalizer rule depends on Knorm
        if isinstance(nn_cfg.Knorm, int) and nn_cfg.Knorm == 1:
            nn_cfg.normalizer_rule = 'last'

    def get_cfgstr_list(nn_cfg, **kwargs):
        nn_cfgstr  = ['_NN(',
                      'K', str(nn_cfg.K),
                      '+', str(nn_cfg.Knorm),
                      ',', nn_cfg.normalizer_rule,
                      ',cks', str(nn_cfg.checks),
                      ')']
        return nn_cfgstr


@six.add_metaclass(ConfigMetaclass)
class FilterConfig(ConfigBase):
    """
    Rename to scoring mechanism

    Example:
        >>> from ibeis.model.Config import *  # NOQA
        >>> filt_cfg = FilterConfig()
        >>> print(filt_cfg.get_cfgstr())
        _FILT(lnbnn_1.0)
        >>> filt_cfg.fg_weight = 1
        >>> print(filt_cfg.get_cfgstr())
    """

    def __init__(filt_cfg, **kwargs):
        super(FilterConfig, filt_cfg).__init__(name='filt_cfg')
        filt_cfg = filt_cfg
        filt_cfg.filt_on = True
        filt_cfg.Krecip = 0  # 0 := off
        filt_cfg.can_match_sameimg = False
        filt_cfg.can_match_samename = True
        filt_cfg.gravity_weighting = False
        filt_cfg._valid_filters = []
        def addfilt(sign, filt, thresh, weight, depends=None):
            """
            dynamically adds filters
            (sign, filt, thresh, weight)
            """
            printDBG('[addfilt] %r %r %r %r' % (sign, filt, thresh, weight))
            filt_cfg._valid_filters.append(filt)
            filt_cfg[filt + '_thresh'] = None if thresh is None else float(thresh)
            filt_cfg[filt + '_weight'] = None if weight is None else float(weight)
            filt_cfg['_' + filt + '_depends'] = depends
            filt_cfg['_' + filt + '_sign'] = sign
        # thresh test is: sign * score <= sign * thresh
        # sign +1 --> Lower scores are better
        # sign -1 --> Higher scores are better
        #tup( Sign,        Filt, ValidSignThresh, ScoreMetaWeight)
        #    (sign,        filt, thresh, weight,  depends)
        addfilt(+1,  'bboxdist',   None,    0.0)
        addfilt(-1,     'recip',    0.0,    0.0, 'filt_cfg.Krecip > 0')
        addfilt(+1,    'bursty',   None,    0.0)
        addfilt(-1,     'ratio',   None,    0.0)
        addfilt(-1,     'lnbnn',   None,    1.0)
        addfilt(-1,   'dupvote',   None,    0.0)
        addfilt(-1,    'lograt',   None,    0.0)
        addfilt(-1,  'normonly',   None,    0.0)
        addfilt(-1,   'logdist',   None,    0.0)
        addfilt(-1,  'loglnbnn',   None,    0.0)
        addfilt(-1,        'fg',   None,    1.0)
        #addfilt(+1, 'scale' )
        filt_cfg.update(**kwargs)

    def make_feasible(filt_cfg):
        # Ensure the list of on filters is valid given the weight and thresh
        if filt_cfg.ratio_thresh is None or filt_cfg.ratio_thresh <= 1:
            filt_cfg.ratio_thresh = None
        if filt_cfg.bboxdist_thresh is None or filt_cfg.bboxdist_thresh >= 1:
            filt_cfg.bboxdist_thresh = None
        if filt_cfg.bursty_thresh  is None or filt_cfg.bursty_thresh <= 1:
            filt_cfg.bursty_thresh = None

    def get_stw(filt_cfg, filt):
        # stw = sign, thresh, weight
        if not isinstance(filt, six.string_types):
            raise AssertionError('Global cache seems corrupted')
        sign   = filt_cfg['_' + filt + '_sign']
        thresh = filt_cfg[filt + '_thresh']
        weight = filt_cfg[filt + '_weight']
        if weight == 1.0:
            weight = 1.0
        return sign, thresh, weight

    def get_active_filters(filt_cfg):
        active_filters = []
        for filt in filt_cfg._valid_filters:
            sign, thresh, weight = filt_cfg.get_stw(filt)
            depends = filt_cfg['_' + filt + '_depends']
            # Check to make sure dependencies are satisfied
            # RCOS TODO FIXME: Possible security flaw.
            # This eval needs to be removed.
            # Need to find a better way of encoding dependencies
            assert depends is None or depends.find('(') == -1, 'unsafe dependency'
            depends_ok = depends is None or eval(depends)
            conditions_ok = thresh is not None or weight != 0
            if conditions_ok and depends_ok:
                active_filters.append(filt)
        return active_filters

    def get_cfgstr_list(filt_cfg, **kwargs):
        if not filt_cfg.filt_on:
            return ['_FILT()']
        on_filters = filt_cfg.get_active_filters()
        filt_cfgstr = ['_FILT(']
        stw_list = []
        # Create a cfgstr for each filter
        for filt in on_filters:
            sign, thresh, weight = filt_cfg.get_stw(filt)
            stw_str = filt
            if thresh is None and weight == 0:
                continue
            if thresh is not None:
                sstr = '>' if sign == -1 else '<'  # actually <=, >=
                stw_str += sstr + str(thresh)
            if weight != 0:
                stw_str += ';' + str(weight)
            stw_list.append(stw_str)
        stw_str = ','.join(stw_list)
        if filt_cfg.Krecip != 0 and 'recip' in on_filters:
            filt_cfgstr += ['Kr' + str(filt_cfg.Krecip)]
            if len(stw_str) > 0:
                filt_cfgstr += [',']
        if len(stw_str) > 0:
            filt_cfgstr += [stw_str]
        if filt_cfg.can_match_sameimg:
            filt_cfgstr += 'sameimg'
        if not filt_cfg.can_match_samename:
            filt_cfgstr += 'notsamename'
        if filt_cfg.gravity_weighting:
            filt_cfgstr += [',gvweight']
        filt_cfgstr += [')']
        return filt_cfgstr


@six.add_metaclass(ConfigMetaclass)
class SpatialVerifyConfig(ConfigBase):
    """
    Spatial verification
    """
    def __init__(sv_cfg, **kwargs):
        super(SpatialVerifyConfig, sv_cfg).__init__(name='sv_cfg')
        tau = 6.28  # 318530
        sv_cfg.ori_thresh   = tau / 4.0
        sv_cfg.scale_thresh = 2
        sv_cfg.xy_thresh = .01
        sv_cfg.nShortlist = 50
        sv_cfg.prescore_method = 'csum'
        sv_cfg.use_chip_extent = False  # BAD CONFIG?
        sv_cfg.min_nInliers = 4
        sv_cfg.sv_on = True
        sv_cfg.update(**kwargs)

    def get_cfgstr_list(sv_cfg, **kwargs):
        if not sv_cfg.sv_on or sv_cfg.xy_thresh is None:
            return ['_SV()']
        sv_cfgstr = ['_SV(']
        sv_cfgstr += [str(sv_cfg.nShortlist)]
        thresh_tup = (sv_cfg.xy_thresh, sv_cfg.scale_thresh, sv_cfg.ori_thresh)
        thresh_str = utool.remove_chars(str(thresh_tup), ' ()').replace(',', ';')
        sv_cfgstr += [',' + thresh_str]
        sv_cfgstr += [',cdl' * sv_cfg.use_chip_extent]  # chip diag len
        sv_cfgstr += [',' + sv_cfg.prescore_method]
        sv_cfgstr += [')']
        return sv_cfgstr


@six.add_metaclass(ConfigMetaclass)
class AggregateConfig(ConfigBase):
    """
    Old Agg Cfg
    """
    def __init__(agg_cfg, **kwargs):
        super(AggregateConfig, agg_cfg).__init__(name='agg_cfg')
        # chipsum, namesum, placketluce
        agg_cfg.isWeighted = False  # nsum, pl
        agg_cfg.score_method = 'csum'  # nsum, pl
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
        if key.find('w') == len(key) - 1:
            agg_cfg.isWeighted = True
            key = key[:-1]
            agg_cfg.score_method = key
        # Sanatize the scoring method
        if key in alt_methods:
            agg_cfg.score_method = alt_methods[key]

    def get_cfgstr_list(agg_cfg, **kwargs):
        agg_cfgstr = []
        agg_cfgstr += ['_AGG(']
        agg_cfgstr += [agg_cfg.score_method]
        if agg_cfg.isWeighted:
            agg_cfgstr += ['w']
        if agg_cfg.score_method  == 'pl':
            agg_cfgstr += [',%d' % (agg_cfg.max_alts,)]
        agg_cfgstr += [')']
        return agg_cfgstr


@six.add_metaclass(ConfigMetaclass)
class FlannConfig(ConfigBase):
    """ FlannConfig

    this flann is only for neareset neighbors in vsone/many
    TODO: this might not need to be here
    """
    def __init__(flann_cfg, **kwargs):
        super(FlannConfig, flann_cfg).__init__(name='flann_cfg')
        flann_cfg.algorithm = 'kdtree'
        #flann_cfg.algorithm = 'linear'
        flann_cfg.trees = 4
        #flann_cfg.trees = 16
        flann_cfg.update(**kwargs)

    def get_dict_args(flann_cfg):
        return {
            'algorithm' : flann_cfg.algorithm,
            'trees' : flann_cfg.trees,
        }

    def get_cfgstr_list(flann_cfg, **kwargs):
        flann_cfgstrs = ['_FLANN(']
        if flann_cfg.algorithm == 'kdtree':
            flann_cfgstrs += ['%d_kdtrees' % flann_cfg.trees]
        elif flann_cfg.algorithm == 'linear':
            flann_cfgstrs += ['%s' % flann_cfg.algorithm]
        else:
            flann_cfgstrs += ['%s' % flann_cfg.algorithm]
        flann_cfgstrs += [')']
        return flann_cfgstrs


@six.add_metaclass(ConfigMetaclass)
class SMKConfig(ConfigBase):
    """
    SMKConfig

    Example:
        >>> from ibeis.model.Config import *  # NOQA
        >>> smk_cfg = SMKConfig()
        >>> result = smk_cfg.get_cfgstr()
        >>> print(result)

    Example2:
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> smk_cfg = ibs.cfg.query_cfg.smk_cfg
        >>> smk_cfg.printme3()
    """
    def __init__(smk_cfg, **kwargs):
        super(SMKConfig, smk_cfg).__init__(name='smk_cfg')
        smk_cfg.smk_thresh = 0.0  # tau in the paper
        smk_cfg.smk_alpha  = 3.0
        smk_cfg.aggregate  = False
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
            'agg=', str(smk_cfg.aggregate),
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
    """ VocabTrainConfig

    Example:
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
        vocabtrain_cfg.vocab_flann_params = {}  # TODO: easy flann params cfgstr
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
    """ VocabAssignConfig

    Example:
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
class QueryConfig(ConfigBase):
    """ query configuration parameters

        Example:
            >>> import ibeis
            >>> ibs = ibeis.opendb('testdb1')
            >>> cfg = ibs.cfg.query_cfg
            >>> cfgstr = ibs.cfg.query_cfg.get_cfgstr()
            >>> print(cfgstr)

    """
    def __init__(query_cfg, **kwargs):
        super(QueryConfig, query_cfg).__init__(name='query_cfg')
        query_cfg.nn_cfg         = NNConfig(**kwargs)
        query_cfg.filt_cfg       = FilterConfig(**kwargs)
        query_cfg.sv_cfg         = SpatialVerifyConfig(**kwargs)
        query_cfg.agg_cfg        = AggregateConfig(**kwargs)
        query_cfg.flann_cfg      = FlannConfig(**kwargs)
        query_cfg.smk_cfg        = SMKConfig(**kwargs)
        # causes some bug in Preference widget if these don't have underscore
        query_cfg._featweight_cfg = FeatureWeightConfig(**kwargs)
        query_cfg.use_cache = False
        query_cfg.num_results = 6
        # Start of pipeline
        query_cfg._valid_pipeline_roots = ['vsmany', 'vsone', 'smk']
        query_cfg.pipeline_root = 'vsmany'
        query_cfg.with_metadata = False
        query_cfg.codename = 'None'
        query_cfg.species_code = '____'
        #if utool.is_developer():
        #    query_cfg.pipeline_root = 'smk'
        # Depends on feature config
        query_cfg.update_query_cfg(**kwargs)
        if utool.VERYVERBOSE:
            print('[config] NEW QueryConfig')

    def update_query_cfg(query_cfg, **kwargs):
        # Each config paramater should be unique
        # So updating them all should not cause conflicts
        # FIXME: Should be able to infer all the children that need updates
        query_cfg.nn_cfg.update(**kwargs)
        query_cfg.filt_cfg.update(**kwargs)
        query_cfg.sv_cfg.update(**kwargs)
        query_cfg.agg_cfg.update(**kwargs)
        query_cfg.flann_cfg.update(**kwargs)
        query_cfg.smk_cfg.update(**kwargs)
        query_cfg.smk_cfg.vocabassign_cfg.update(**kwargs)
        query_cfg.smk_cfg.vocabtrain_cfg.update(**kwargs)
        query_cfg._featweight_cfg.update(**kwargs)
        query_cfg._featweight_cfg._feat_cfg.update(**kwargs)
        query_cfg._featweight_cfg._feat_cfg._chip_cfg.update(**kwargs)
        query_cfg.update(**kwargs)
        # Ensure feasibility of the configuration
        query_cfg.make_feasible()

    def make_feasible(query_cfg):
        """
        removes invalid parameter settings over all cfgs (move to QueryConfig)
        """
        codename = query_cfg.codename
        filt_cfg = query_cfg.filt_cfg
        nn_cfg   = query_cfg.nn_cfg
        featweight_cfg = query_cfg._featweight_cfg
        feat_cfg = query_cfg._featweight_cfg._feat_cfg
        smk_cfg = query_cfg.smk_cfg
        vocabassign_cfg = query_cfg.smk_cfg.vocabassign_cfg
        agg_cfg = query_cfg.agg_cfg
        sv_cfg = query_cfg.sv_cfg
        # TODO:
        if codename == 'nsum':
            filt_cfg.dupvote_weight = 1.0
            agg_cfg.score_method = 'nsum'
            sv_cfg.prescore_method = 'nsum'
        elif codename == 'vsmany':
            query_cfg.pipeline_root = 'vsmany'
        elif codename == 'vsone':
            query_cfg.pipeline_root = 'vsone'
            nn_cfg.K = 2
            nn_cfg.Knorm = 1
            filt_cfg.ratio_thresh = 1.6
            filt_cfg.ratio_weight = 1.0
        elif codename == 'asmk':
            query_cfg.pipeline_root = 'asmk'
        elif codename == 'smk':
            query_cfg.pipeline_root = 'smk'
        elif codename == 'None':
            pass
        #if query_cfg.species_code == '____':
        #    query_cfg.fg_weight = 0.0
        #if query_cfg.species_code == 'zebra_plains':
        #    query_cfg.fg_weight = 1.0
        #if query_cfg.species_code == 'zebra_grevys':
        #    query_cfg.fg_weight = 1.0
            # TODO:

        if query_cfg.pipeline_root == 'asmk':
            query_cfg.pipeline_root = 'smk'
            smk_cfg.aggregate = True

        hasvalid_root = any([
            query_cfg.pipeline_root == root
            for root in query_cfg._valid_pipeline_roots])
        try:
            assert hasvalid_root, 'invalid pipeline root %r' % query_cfg.pipeline_root
        except AssertionError as ex:
            utool.printex(ex)
            if utool.SUPER_STRICT:
                raise
            else:
                query_cfg.pipeline_root = query_cfg._valid_pipeline_roots[0]
                pass

        if feat_cfg.nogravity_hack is False:
            filt_cfg.gravity_weighting = False

        if featweight_cfg.featweight_on is not True or filt_cfg.fg_weight == 0.0:
            filt_cfg.fg_weight = 0.0
            featweight_cfg.featweight_on = False

        vocabassign_cfg.make_feasible()
        smk_cfg.make_feasible()
        filt_cfg.make_feasible()
        nn_cfg.make_feasible()

    def deepcopy(query_cfg, **kwargs):
        copy_ = copy.deepcopy(query_cfg)
        copy_.update_query_cfg(**kwargs)
        return copy_

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
            if kwargs.get('use_filt', True):
                cfgstr_list += query_cfg.filt_cfg.get_cfgstr_list(**kwargs)
            if kwargs.get('use_sv', True):
                cfgstr_list += query_cfg.sv_cfg.get_cfgstr_list(**kwargs)
            if kwargs.get('use_agg', True):
                cfgstr_list += query_cfg.agg_cfg.get_cfgstr_list(**kwargs)
            if kwargs.get('use_flann', True):
                cfgstr_list += query_cfg.flann_cfg.get_cfgstr_list(**kwargs)
        else:
            raise AssertionError('bad pipeline root: ' + str(query_cfg.pipeline_root))
        if kwargs.get('use_featweight', True):
            cfgstr_list += query_cfg._featweight_cfg.get_cfgstr_list(**kwargs)
        return cfgstr_list


@six.add_metaclass(ConfigMetaclass)
class FeatureWeightConfig(ConfigBase):
    """
    Example:
        >>> from ibeis.model.Config import *  # NOQA
        >>> featweight_cfg = FeatureWeightConfig()
        >>> print(featweight_cfg.get_cfgstr())
        _FEATWEIGHT(OFF)
    """

    def __init__(featweight_cfg, **kwargs):
        super(FeatureWeightConfig, featweight_cfg).__init__(name='featweight_cfg')
        # Featweights depend on features
        featweight_cfg._feat_cfg = FeatureConfig(**kwargs)
        # Feature weights depend on the detector, but we only need to mirror
        # some parameters because featweight_cfg should not use the detect_cfg
        # object
        featweight_cfg.featweight_on = True
        featweight_cfg.featweight_species  = 'uselabel'
        featweight_cfg.featweight_detector = 'rf'
        featweight_cfg.update(**kwargs)

    def make_feasible(featweight_cfg):
        #featweight_cfg.featweight_on = False
        pass

    def get_cfgstr_list(featweight_cfg, **kwargs):
        featweight_cfg.make_feasible()
        featweight_cfgstrs = []
        if kwargs.get('use_featweight', True):
            if featweight_cfg.featweight_on is not True:
                if featweight_cfg.featweight_on == 'ERR':
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

    Example:
        >>> import ibeis
        >>> from ibeis.model import Config  # NOQA
        >>> from ibeis.model.Config import *  # NOQA
        >>> feat_cfg = Config.FeatureConfig()
        >>> print(feat_cfg.get_cfgstr())
        _FEAT(hesaff+sift_nScal=3,thrsh=5.33,edggn=10.00,nIter=16,cnvrg=0.05,intlS=1.60)_CHIP(sz450)
    """
    def __init__(feat_cfg, **kwargs):
        import numpy as np
        import ctypes as C
        # Features depend on chips
        super(FeatureConfig, feat_cfg).__init__(name='feat_cfg')
        feat_cfg._chip_cfg = ChipConfig(**kwargs)
        feat_cfg.feat_type = 'hesaff+sift'
        feat_cfg.whiten = False
        #feat_cfg.scale_min = 0  # 0  # 30 # TODO: Put in pref types here
        #feat_cfg.scale_max = 9001  # 9001 # 80
        # Inlineish copy of pyhesaff.hesaff_types_parms
        PY2 = True
        if PY2:
            int_t     = C.c_int
        else:
            raise NotImplementedError('PY3')
        bool_t    = C.c_bool
        float_t   = C.c_float
        feat_cfg._param_list = [
            (int_t,   'numberOfScales', 3, 'number of scale per octave'),
            (float_t, 'threshold', 16.0 / 3.0, 'noise dependent threshold on the response (sensitivity)'),
            (float_t, 'edgeEigenValueRatio', 10.0, 'ratio of the eigenvalues'),
            (int_t,   'border', 5, 'number of pixels ignored at the border of image'),
            # Affine Shape Params
            (int_t,   'maxIterations', 16, 'number of affine shape interations'),
            (float_t, 'convergenceThreshold', 0.05, 'maximum deviation from isotropic shape at convergence'),
            (int_t,   'smmWindowSize', 19, 'width and height of the SMM (second moment matrix) mask'),
            (float_t, 'mrSize', 3.0 * np.sqrt(3.0), 'size of the measurement region (as multiple of the feature scale)'),
            # SIFT params
            (int_t,   'spatialBins', 4),
            (int_t,   'orientationBins', 8),
            (float_t, 'maxBinValue', 0.2),
            # Shared params
            (float_t, 'initialSigma', 1.6, 'amount of smoothing applied to the initial level of first octave'),
            (int_t,   'patchSize', 41, 'width and height of the patch'),
            # My params
            (float_t, 'scale_min', -1.0),
            (float_t, 'scale_max', -1.0),
            (bool_t,  'rotation_invariance', False),
        ]

        for type_, name, default, doc in feat_cfg._iterparams():
            setattr(feat_cfg, name, default)

        feat_cfg.use_adaptive_scale = False  # 9001 # 80
        feat_cfg.nogravity_hack = False  # 9001 # 80
        feat_cfg.update(**kwargs)

    def _iterparams(feat_cfg):
        for tup in feat_cfg._param_list:
            if len(tup) == 4:
                type_, name, default, doc = tup
            else:
                type_, name, default = tup
                doc = None
            yield (type_, name, default, doc)

    def get_dict_args(feat_cfg):
        dict_args = {
            name: feat_cfg[name]
            for type_, name, default, doc in feat_cfg._iterparams()
        }
        #dict_args = {
        #    'scale_min': feat_cfg.scale_min,
        #    'scale_max': feat_cfg.scale_max,
        #    'use_adaptive_scale': feat_cfg.use_adaptive_scale,
        #    'nogravity_hack': feat_cfg.nogravity_hack,
        #}
        return dict_args

    def get_cfgstr_list(feat_cfg, **kwargs):
        #if feat_cfg._chip_cfg is None:
        #    raise Exception('Chip config is required')
        #if feat_cfg.scale_min < 0:
        #    feat_cfg.scale_min = None
        #if feat_cfg.scale_max < 0:
        #    feat_cfg.scale_max = None
        if kwargs.get('use_feat', True):
            feat_cfgstrs = ['_FEAT(']
            feat_cfgstrs += [feat_cfg.feat_type]
            feat_cfgstrs += [',white'] * feat_cfg.whiten
            #feat_cfgstrs += [',%r_%r' % (feat_cfg.scale_min, feat_cfg.scale_max)]
            feat_cfgstrs += [',adaptive'] * feat_cfg.use_adaptive_scale
            feat_cfgstrs += [',nogravity'] * feat_cfg.nogravity_hack
            # TODO: Named Tuple
            alias = {
                'numberOfScales': 'nScales',
                'edgeEigValRat': 'EdgeEigvRat',
                'maxIterations': 'nIter',
                #'whiten':
            }
            ignore = []
            #ignore = set(['whiten', 'scale_min', 'scale_max', 'use_adaptive_scale',
            #              'nogravity_hack', 'feat_type'])
            ignore_if_default = set([
                'numberOfScales',
                'threshold',
                'edgeEigenValueRatio',
                'border',
                'maxIterations',
                'convergenceThreshold',

                'smmWindowSize',
                'mrSize',

                'spatialBins',
                'orientationBins',

                'maxBinValue',
                'initialSigma',
                'patchSize',

                'scale_min',
                'scale_max',
                'rotation_invariance',
            ])

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
                    #namestr = utool.hashstr(alias.get(name, name), hashlen=6,
                    #                        alphabet=utool.util_hash.ALPHABET_27)
                    namestr = utool.clipstr(alias.get(name, name), 5)
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
        chip_cfgstr = []
        if kwargs.get('use_chip', True):
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
        return ['_CHIP(', (','.join(chip_cfgstr)), ')']


@six.add_metaclass(ConfigMetaclass)
class DetectionConfig(ConfigBase):
    """
    Example:
        >>> from ibeis.model.Config import *  # NOQA
        >>> detect_cfg = DetectionConfig()
        >>> print(detect_cfg.get_cfgstr())
        _DETECT(rf,zebra_grevys)
    """
    def __init__(detect_cfg, **kwargs):
        super(DetectionConfig, detect_cfg).__init__(name='detect_cfg')
        detect_cfg.species     = 'zebra_grevys'
        detect_cfg.detector    = 'rf'
        detect_cfg.detectimg_sqrt_area = 800
        detect_cfg.update(**kwargs)

    def get_cfgstr_list(detect_cfg):
        cfgstrs = ['_DETECT(',
                   detect_cfg.detector,
                   ',', detect_cfg.species,
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
    def __init__(othercfg, **kwargs):
        super(OtherConfig, othercfg).__init__(name='othercfg')
        othercfg.thumb_size     = 64
        othercfg.ranks_lt       = 2
        othercfg.auto_localize  = True
        othercfg.detect_add_after = 1
        othercfg.detect_use_chunks = True
        othercfg.update(**kwargs)

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


def default_query_cfg(**kwargs):
    if utool.VERYVERBOSE:
        print('[config] default_query_cfg()')
    kwargs['pipeline_root'] = 'vsmany'
    query_cfg = QueryConfig(**kwargs)
    return query_cfg


def default_vsone_cfg(ibs, **kwargs):
    kwargs['pipeline_root'] = 'vsone'
    utool.dict_update_newkeys(kwargs, {
        'lnbnn_weight': 0.0,
        'checks': 256,
        'K': 1,
        'Knorm': 1,
        'ratio_weight': 1.0,
        'ratio_thresh': 1.5,
    })
    query_cfg = QueryConfig(**kwargs)
    return query_cfg


if __name__ == '__main__':
    print('[Config] main()')
    utool.VERYVERBOSE = True
    query_cfg = default_query_cfg()
    query_cfg.printme3()
    print(query_cfg.get_cfgstr())
    print('[Config] end()')
