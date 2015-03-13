from __future__ import absolute_import, division, print_function
import six  # NOQA
import utool as ut  # NOQA
import numpy as np
import vtool as vt
from ibeis.control.controller_inject import make_ibs_register_decorator
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[manual_newfuncs]')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


@register_ibs_method
def new_query_request(ibs, qaid_list, daid_list, cfgdict=None,
                      verbose=ut.NOT_QUIET, **kwargs):
    """
    alias for ibeis.model.hots.query_request.new_ibeis_query_request

    Args:
        qaid_list (list):
        daid_list (list):
        cfgdict (None):
        verbose (bool):

    Returns:
        QueryRequest: qreq_ -  hyper-parameters

    CommandLine:
        python -m ibeis.control.manual_ibeiscontrol_funcs --test-new_query_request

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_ibeiscontrol_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> daid_list = ibs.get_valid_aids()
        >>> qaid_list = daid_list[0:2]
        >>> cfgdict = {}
        >>> verbose = True
        >>> qreq_ = new_query_request(ibs, qaid_list, daid_list, cfgdict, verbose)
        >>> qreq_.set_external_qaid_mask(qaid_list[1:2])
        >>> print(qreq_.get_external_qaids())
        >>> result = str(qreq_.get_query_hashid())
        >>> print(result)
        _QSUUIDS((1)nztoqb6&7apjltd1)
    """
    from ibeis.model.hots import query_request
    qreq_ = query_request.new_ibeis_query_request(
        ibs, qaid_list, daid_list, cfgdict=cfgdict, verbose=verbose, **kwargs)
    return qreq_


@register_ibs_method
def get_vocab_cfgstr(ibs, taids=None, qreq_=None):
    # TODO: change into config_rowid
    if qreq_ is not None:
        cfg = qreq_.qparams
        vocab_cfgstr_ = qreq_.qparams.vocabtrain_cfgstr
        feat_cfgstr_ = qreq_.qparams.feat_cfgstr
    else:
        cfg = ibs.cfg.query_cfg.smk_cfg.vocabtrain_cfg
        vocab_cfgstr_ = ibs.cfg.query_cfg.smk_cfg.vocabtrain_cfg.get_cfgstr()
        feat_cfgstr_ = ibs.cfg.feat_cfg.get_cfgstr()

    if taids is None:
        if cfg.vocab_taids == 'all':
            taids = ibs.get_valid_aids()
        else:
            # FIXME Preferences cannot currently handle lists
            # TODO: Incorporated taids (vocab training ids) into qreq
            taids = cfg.vocab_taids

    tannot_hashid = ibs.get_annot_hashid_visual_uuid(taids, prefix='T')
    vocab_cfgstr = vocab_cfgstr_ + tannot_hashid + feat_cfgstr_
    return vocab_cfgstr


@register_ibs_method
def get_vocab_words(ibs, taids=None, qreq_=None):
    """
    Hackyish way of putting vocab generation into the controller.
    Ideally there would be a preproc_vocab in ibeis.model.preproc
    and sql would store this under some config

    Example:
        >>> from ibeis.control.manual_ibeiscontrol_funcs import *   # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
    """
    from vtool import clustering2 as clustertool
    import numpy as np
    if qreq_ is not None:
        cfg = qreq_.qparams
    else:
        cfg = ibs.cfg.query_cfg.smk_cfg.vocabtrain_cfg

    if taids is None:
        if cfg.vocab_taids == 'all':
            taids = ibs.get_valid_aids()
        else:
            # FIXME Preferences cannot currently handle lists
            # TODO: Incorporated taids (vocab training ids) into qreq
            taids = cfg.vocab_taids

    vocab_cfgstr = get_vocab_cfgstr(ibs, taids=taids, qreq_=qreq_)
    raise NotImplementedError('no temp state!')

    if vocab_cfgstr not in ibs.temporary_state:
        nWords = cfg.nWords
        initmethod   = cfg.vocab_init_method
        max_iters    = cfg.vocab_nIters
        flann_params = cfg.vocab_flann_params

        train_vecs_list = ibs.get_annot_vecs(taids, eager=True)
        # Stack vectors
        train_vecs = np.vstack(train_vecs_list)
        del train_vecs_list
        print('[get_vocab_words] Train Vocab(nWords=%d) using %d annots and %d descriptors' %
              (nWords, len(taids), len(train_vecs)))
        kwds = dict(max_iters=max_iters, initmethod=initmethod,
                    appname='smk', flann_params=flann_params)
        words = clustertool.cached_akmeans(train_vecs, nWords, **kwds)
        # Cache words in temporary state
        ibs.temporary_state[vocab_cfgstr] = words
        del train_vecs
    else:
        words = ibs.temporary_state[vocab_cfgstr]
    return words

#@register_ibs_method
#def get_vocab_assignments(ibs, qreq_=None):
#    pass


from ibeis import constants as const
from ibeis.control import accessor_decors
from ibeis.model.hots import distinctiveness_normalizer as dcvs_normer


#@ut.time_func
@register_ibs_method
def get_annot_kpts_distinctiveness(ibs, aid_list, config2_=None, **kwargs):
    """
    very hacky, but cute way to cache keypoint distinctivness

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):
        dstncvs_normer (None):

    Returns:
        list: dstncvs_list

    CommandLine:
        python -m ibeis.control.manual_ibeiscontrol_funcs --test-get_annot_kpts_distinctiveness

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_ibeiscontrol_funcs import *  # NOQA
        >>> from ibeis.model.hots import distinctiveness_normalizer
        >>> import ibeis
        >>> import numpy as np
        >>> config2_ = None
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids(species=const.Species.ZEB_PLAIN)
        >>> # execute function
        >>> aid_list1 = aid_list[::2]
        >>> aid_list2 = aid_list[1::3]
        >>> dstncvs_list1 = get_annot_kpts_distinctiveness(ibs, aid_list1)
        >>> dstncvs_list2 = get_annot_kpts_distinctiveness(ibs, aid_list2)
        >>> dstncvs_list = get_annot_kpts_distinctiveness(ibs, aid_list)
        >>> print(ut.depth_profile(dstncvs_list1))
        >>> stats_dict = ut.dict_stack([ut.get_stats(dstncvs) for dstncvs in dstncvs_list])
        >>> print(ut.dict_str(stats_dict))
        >>> assert np.all(np.array(stats_dict['min']) >= 0), 'distinctiveness was out of bounds'
        >>> assert np.all(np.array(stats_dict['max']) <= 1), 'distinctiveness was out of bounds'
    """
    # per-species disinctivness wrapper around ibeis cached function
    # get feature rowids
    aid_list = np.array(aid_list)
    fid_list = np.array(ibs.get_annot_feat_rowids(aid_list, ensure=True, eager=True, nInput=None, config2_=config2_))
    species_rowid_list = np.array(ibs.get_annot_species_rowids(aid_list))
    # Compute distinctivness separately for each species
    unique_sids, groupxs = vt.group_indices(species_rowid_list)
    fids_groups          = vt.apply_grouping(fid_list, groupxs)
    species_text_list    = ibs.get_species_texts(unique_sids)
    # Map distinctivness computation
    normer_list = [dcvs_normer.request_species_distinctiveness_normalizer(species)
                   for species in species_text_list]
    # Reduce to get results
    dstncvs_groups = [
        get_feat_kpts_distinctiveness(ibs, fids, dstncvs_normer=dstncvs_normer, species_rowid=sid, **kwargs)
        for dstncvs_normer, fids, sid in zip(normer_list, fids_groups, unique_sids)
    ]
    dstncvs_list = vt.invert_apply_grouping(dstncvs_groups, groupxs)
    return dstncvs_list


dcvs_cfgkeys = dcvs_normer.DCVS_DEFAULT.get_varnames() + ['species_rowid']
dcvs_colname = dcvs_normer.DCVS_DEFAULT.name


@accessor_decors.cache_getter(const.FEATURE_TABLE, dcvs_colname, cfgkeys=dcvs_cfgkeys, debug=None)
def get_feat_kpts_distinctiveness(ibs, fid_list, dstncvs_normer=None, species_rowid=None, **kwargs):
    #print('[ibs] get_feat_kpts_distinctiveness fid_list=%r' % (fid_list,))
    vecs_list = ibs.get_feat_vecs(fid_list, eager=True, nInput=None)
    dstncvs_list = [None if vecs is None else dstncvs_normer.get_distinctiveness(vecs, **kwargs) for vecs in vecs_list]
    return dstncvs_list


@register_ibs_method
def show_annot(ibs, aid, *args, **kwargs):
    """ viz helper see ibeis.viz.viz_chip.show_chip """
    from ibeis.viz import viz_chip
    return viz_chip.show_chip(ibs, aid, *args, **kwargs)


@register_ibs_method
def show_annot_image(ibs, aid, *args, **kwargs):
    """ viz helper see ibeis.viz.viz_chip.show_chip """
    from ibeis.viz import viz_image
    gid = ibs.get_annot_gids(aid)
    return viz_image.show_image(ibs, gid, *args, **kwargs)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.manual_ibeiscontrol_funcs
        python -m ibeis.control.manual_ibeiscontrol_funcs --allexamples
        python -m ibeis.control.manual_ibeiscontrol_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
