from __future__ import absolute_import, division, print_function
import functools  # NOQA
import six  # NOQA
from six.moves import map, range  # NOQA
from ibeis import constants  # NOQA
from ibeis.control.IBEISControl import IBEISController
import utool  # NOQA
import utool as ut  # NOQA
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[autogen_ibsfuncs]')

# Create dectorator to inject these functions into the IBEISController
register_ibs_aliased_method   = ut.make_class_method_decorator((IBEISController, 'manual'))
register_ibs_unaliased_method = ut.make_class_method_decorator((IBEISController, 'manual'))


def register_ibs_method(func):
    aliastup = (func, 'manual_' + ut.get_funcname(func))
    register_ibs_unaliased_method(func)
    register_ibs_aliased_method(aliastup)
    return func


@register_ibs_method
def new_query_request(ibs, qaid_list, daid_list, custom_qparams):
    """
    daid_list = ibs.get_valid_aids()
    qaid_list = daid_list[0:1]
    custom_qparams = {}
    """
    from ibeis.model.hots import query_request
    qreq_ = query_request.new_ibeis_query_request(ibs, qaid_list, daid_list, custom_qparams)
    return qreq_


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

    tannot_hashid = ibs.get_annot_uuid_hashid(taids, '_TAIDS')
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


def get_vocab_assignments(ibs, qreq_=None):
    pass
