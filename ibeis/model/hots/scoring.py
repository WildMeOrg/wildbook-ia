"""
TODO:
optional symetric and asymmetric search

"""

from __future__ import absolute_import, division, print_function
import six  # NOQA
from itertools import chain
import numpy as np
import vtool as vt
import utool as ut
from vtool import coverage_kpts
from vtool import coverage_grid
from ibeis.model.hots import hstypes
from ibeis.model.hots import name_scoring
from ibeis.model.hots import distinctiveness_normalizer
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
from six.moves import zip, range  # NOQA
#profile = ut.profile
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[scoring]', DEBUG=False)


def get_chipmatch_testdata(**kwargs):
    cfgdict = {}
    ibs, qreq_ = plh.get_pipeline_testdata('testdb1', cfgdict)
    locals_ = plh.testrun_pipeline_upto(qreq_, 'spatial_verification')
    cm_list = locals_['cm_list_FILT']
    # Get a single cmtup_old
    return ibs, qreq_, cm_list


@profile
def score_chipmatch_list(qreq_, cm_list, score_method):
    """
    CommandLine:
        python -m ibeis.model.hots.scoring --test-score_chipmatch_list
        python -m ibeis.model.hots.scoring --test-score_chipmatch_list:0 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver()
        >>> score_method = qreq_.qparams.prescore_method
        >>> score_chipmatch_list(qreq_, cm_list, score_method)
        >>> cm = cm_list[0]
        >>> assert cm.score_list.argmax() == 0
        >>> cm.testshow(qreq_)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_post_sver()
        >>> qaid = qreq_.get_external_qaids()[0]
        >>> cm = cm_list[0]
        >>> score_method = qreq_.qparams.score_method
        >>> score_chipmatch_list(qreq_, cm_list, score_method)
        >>> assert cm.score_list.argmax() == 0
        >>> cm.testshow(qreq_)
    """
    # Choose the appropriate scoring mechanism
    if score_method == 'csum':
        evaluate_chipmatch_annot_scores(cm_list, qreq_)
        for cm in cm_list:
            cm.score_list = cm.annot_score_list
    elif score_method == 'nsum':
        with ut.embed_on_exception_context:
            evaluate_chipmatch_annot_scores(cm_list, qreq_)
            evaluate_chipmatch_name_scores(cm_list, qreq_)
            for cm in cm_list:
                expand_name_scores_to_annot(cm, qreq_)
            #cm.score_list = cm.annot_score_list
    else:
        raise Exception('[hs] unknown scoring method:' + score_method)


@profile
def make_chipmatch_shortlists(qreq_, cm_list, nNameShortList, nAnnotPerName):
    """
    Makes shortlists for reranking

    CommandLine:
        python -m ibeis.model.hots.scoring --test-make_chipmatch_shortlists --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list=[5])
        >>> score_method    = 'nsum'
        >>> nNameShortList  = 5
        >>> nAnnotPerName   = 2
        >>> score_chipmatch_list(qreq_, cm_list, score_method)
        >>> assert cm_list[0].dnid_list.take(cm_list[0].argsort())[0] == cm_list[0].qnid
        >>> # execute function
        >>> cm_shortlist = make_chipmatch_shortlists(qreq_, cm_list, nNameShortList, nAnnotPerName)
        >>> cm_list[0].print_rawinfostr()
        >>> cm = cm_shortlist[0]
        >>> cm.print_rawinfostr()
        >>> # should be sorted already from the shortlist take
        >>> top_nid_list = cm.dnid_list
        >>> top_aid_list = cm.daid_list
        >>> qnid = cm.qnid
        >>> print('top_aid_list = %r' % (top_aid_list,))
        >>> print('top_nid_list = %r' % (top_nid_list,))
        >>> print('qnid = %r' % (qnid,))
        >>> rankx = top_nid_list.tolist().index(qnid)
        >>> assert rankx == 0, 'qnid=%r should be first rank, not rankx=%r' % (qnid, rankx)
        >>> max_num_rerank = nNameShortList * nAnnotPerName
        >>> min_num_rerank = nNameShortList
        >>> ut.assert_inbounds(len(top_nid_list), min_num_rerank, max_num_rerank, 'incorrect number in shortlist', eq=True)
        >>> cm.testshow(qreq_, daid=top_aid_list[0])
    """
    cm_shortlist = []
    for cm in cm_list:
        nscore_tup = name_scoring.group_scores_by_name(qreq_.ibs, cm.daid_list, cm.score_list)
        (sorted_nids, sorted_nscore, sorted_aids, sorted_scores) = nscore_tup
        # Clip number of names
        _top_aids_list  = ut.listclip(sorted_aids, nNameShortList)
        # Clip number of annots per name
        _top_clipped_aids_list = [ut.listclip(aids, nAnnotPerName) for aids in _top_aids_list]
        top_aids = ut.flatten(_top_clipped_aids_list)
        cm_subset = cm.shortlist_subset(top_aids)
        cm_shortlist.append(cm_subset)
    return cm_shortlist


def evaluate_chipmatch_annot_scores(cm_list, qreq_=None):
    """
    Evaluates chimmatch csum score in place

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = get_chipmatch_testdata()
        >>> cm = cm_list[0]
        >>> assert cm.annot_score_list is None
        >>> evaluate_chipmatch_annot_scores(cm_list)
        >>> assert cm.annot_score_list is not None
        >>> cm.print_rawinfostr()
    """
    for cm in cm_list:
        fs_list = (fsv.prod(axis=1) for fsv in cm.fsv_list)
        cm.annot_score_list = np.array([np.sum(fs) for fs in fs_list])


def evaluate_chipmatch_name_scores(cm_list, qreq_):
    """
    Evaluates chimmatch nsum score in place

    CommandLine:
        python dev.py -t custom:score_method=csum,prescore_method=csum --db GZ_ALL --show --va -w --qaid 1032 --noqcache
        python dev.py -t nsum_nosv --db GZ_ALL --allgt --noqcache
        python dev.py -t nsum --db GZ_ALL --show --va -w --qaid 1032 --noqcache
        python dev.py -t nsum_nosv --db GZ_ALL --show --va -w --qaid 1032 --noqcache
        python -m ibeis.model.hots.scoring --test-evaluate_chipmatch_name_scores
        python -m ibeis.model.hots.scoring --test-evaluate_chipmatch_name_scores --show
        utprof.py -m ibeis.model.hots.scoring --test-evaluate_chipmatch_name_scores

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.name_scoring import *   # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver()
        >>> evaluate_chipmatch_name_scores(cm_list, qreq_)
        >>> cm = cm_list[0]
        >>> nid = cm.qnid
        >>> ut.assert_eq(ut.list_argmax(cm.name_score_list), cm.nid2_idx[nid])
        >>> cm.testshow(qreq_)
    """
    for cm in cm_list:
        cm.evaluate_dnids(qreq_.ibs)
        unique_nids, name_score_list = name_scoring.name_scoring_sparse(cm)
        cm.assign_name_scores(unique_nids, name_score_list)


def expand_name_scores_to_annot(cm, qreq_):
    """
    weird name, fix that, takes name scores and gives them to the best annotation

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> ibs, qreq_, cm_list = plh.testdata_pre_sver()
        >>> evaluate_chipmatch_annot_scores(cm_list, qreq_)
        >>> evaluate_chipmatch_name_scores(cm_list, qreq_)
        >>> cm = cm_list[0]
        >>> assert cm.score_list is None
        >>> expand_name_scores_to_annot(cm, qreq_)
        >>> assert cm.score_list is not None
        >>> assert ut.isunique(cm.dnid_list[cm.score_list > 0]), 'bad name score'

    """
    try:
        annot_score_list = cm.annot_score_list
        annot_nid_list = cm.dnid_list
        nid_list, groupxs = vt.group_indices(annot_nid_list)
        grouped_scores = vt.apply_grouping(annot_score_list, groupxs)
        # Find the position of the highest scoring annotation for each name
        offset_list = np.array([annot_scores.argmax() for annot_scores in grouped_scores])
        # use chain to start offsets with 0
        sizeoffset_list = np.array([len(annot_scores) for annot_scores in chain([[]], grouped_scores[:-1])])
        baseindex_list = sizeoffset_list.cumsum()
        idx_list = baseindex_list + offset_list
        # give the annotation domain a name score
        name_score_list = cm.name_score_list
        score_list = np.zeros(len(annot_score_list), dtype=name_score_list.dtype)
        score_list[idx_list] = name_score_list
        cm.score_list = score_list
    except Exception as ex:
        cm.print_rawinfostr()
        cm.print_csv()
        ut.printex(ex)
        raise


#### FEATURE WEIGHTS ####
# TODO: qreq_


def sift_selectivity_score(vecs1_m, vecs2_m, cos_power=3.0, dtype=np.float64):
    """
    applies selectivity score from SMK paper
    Take componentwise dot produt and divide by 512**2 because of the
    sift descriptor uint8 trick
    """
    # Compute dot product (cosine of angle between sift descriptors)
    cosangle = vt.componentwise_dot(vecs1_m.astype(dtype), vecs2_m.astype(dtype))
    # Adjust for uint8 trick
    cosangle /= hstypes.PSEUDO_UINT8_MAX_SQRD
    # apply selectivity functiodictin
    selectivity_score = np.power(cosangle, cos_power)
    # If cosine can be less than 0 replace previous line with next line
    # or just write an rvec_selectivity_score function
    #selectivity_score = np.multiply(np.sign(cosangle), np.power(cosangle, cos_power))
    return selectivity_score


@profile
def get_kpts_distinctiveness(qreq_, aid_list, config={}):
    """
    per-species disinctivness wrapper around ibeis cached function
    """
    dcvs_kw = distinctiveness_normalizer.DCVS_DEFAULT.updated_cfgdict(config)
    dstncvs_list = qreq_.ibs.get_annot_kpts_distinctiveness(aid_list, qreq_=qreq_, **dcvs_kw)
    return dstncvs_list


def get_annot_kpts_basline_weights(qreq_, aid_list, config={}):
    # TODO: clip the fgweights? (dilation?)
    # TODO; normalize and paramatarize and clean
    qdstncvs_list  = get_kpts_distinctiveness(qreq_, aid_list, config)
    qfgweight_list = qreq_.ibs.get_annot_fgweights(aid_list, ensure=True, qreq_=qreq_)
    weights_list = [(qfgweight * qdstncvs) for (qfgweight,  qdstncvs) in zip(qdstncvs_list, qfgweight_list)]
    #weights_list = [(qfgweight * qdstncvs) ** .5 for (qfgweight,  qdstncvs) in zip(qdstncvs_list, qfgweight_list)]
    #weights_list = [(qfgweight + qdstncvs) / 2 for (qfgweight,  qdstncvs) in zip(qdstncvs_list, qfgweight_list)]
    return weights_list


def get_mask_func(config):
    maskscore_mode = config.get('maskscore_mode', 'grid')
    #print(maskscore_mode)
    FUNC_ARGS_DICT = {
        'grid': (coverage_grid.make_grid_coverage_mask, coverage_grid.COVGRID_DEFAULT),
        'kpts': (coverage_kpts.make_kpts_coverage_mask, coverage_kpts.COVKPTS_DEFAULT),
    }
    make_mask_func, func_defaults = FUNC_ARGS_DICT[maskscore_mode]
    cov_cfg = func_defaults.updated_cfgdict(config)
    # Hack to make kwargs happy
    #cov_cfg = ut.filter_valid_kwargs(make_mask_func, config)
    return make_mask_func, cov_cfg


#### MASK COVERAGE SCORING ####


def compute_coverage_score(qreq_, unscored_cm, config={}):
    """
    CommandLine:
        python -m ibeis.model.hots.scoring --test-compute_coverage_score:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> qreq_, unscored_cm = plh.testdata_scoring()
        >>> config = qreq_.qparams
        >>> score_list = compute_coverage_score(qreq_, unscored_cm)
        >>> ut.assert_inbounds(np.array(score_list), 0, 1, eq=True)
        >>> result = ut.list_str(score_list, precision=3)
        >>> print(result)
    """
    make_mask_func, cov_cfg = get_mask_func(config)
    masks_iter = general_coverage_mask_generator(make_mask_func, qreq_, unscored_cm, config, cov_cfg)
    score_list = list(score_masks(masks_iter))
    return score_list


def score_masks(masks_iter):
    for weight_mask_m, weight_mask in masks_iter:
        coverage_score = weight_mask_m.sum() / weight_mask.sum()
        yield coverage_score


def general_coverage_mask_generator(make_mask_func, qreq_, unscored_cm, config, cov_cfg):
    if ut.VERYVERBOSE:
        print('make_mask_func = %r' % (make_mask_func,))
        print('cov_cfg = %s' % (ut.dict_str(cov_cfg),))
    qaid = unscored_cm.qaid
    fm_list = unscored_cm.fm_list
    fs_list = unscored_cm.fs_list
    # Distinctivness and foreground weight
    weights = get_annot_kpts_basline_weights(qreq_, [qaid], config)[0]
    # Denominator weight mask
    chipsize    = qreq_.ibs.get_annot_chip_sizes(qaid, qreq_=qreq_)
    qkpts       = qreq_.ibs.get_annot_kpts(qaid, qreq_=qreq_)
    weight_mask = make_mask_func(qkpts, chipsize, weights, resize=False, **cov_cfg)
    # Prealloc data for loop
    weight_mask_m = weight_mask.copy()
    SYMMETRIC = False
    # Apply weighted scoring to matches
    for fm, fs in zip(fm_list, fs_list):
        # Get matching query keypoints
        if SYMMETRIC:
            get_annot_kpts_basline_weights()

        qkpts_m    = qkpts.take(fm.T[0], axis=0)
        weights_m  = fs * weights.take(fm.T[0], axis=0)
        # hacky buisness
        #weights_m  = weights.take(fm.T[0], axis=0)
        #weights_m = fs
        weight_mask_m = make_mask_func(qkpts_m, chipsize, weights_m,
                                       out=weight_mask_m, resize=False,
                                       **cov_cfg)
        # CAREFUL weight_mask_m is overriden on every iteration
        yield weight_mask_m, weight_mask


def get_masks(qreq_, cm, config={}):
    r"""
    testing function

    CommandLine:
        # SHOW THE BASELINE AND MATCHING MASKS
        python -m ibeis.model.hots.scoring --test-get_masks
        python -m ibeis.model.hots.scoring --test-get_masks \
            --maskscore_mode=kpts --show --prior_coeff=.5 --unconstrained_coeff=.3 --constrained_coeff=.2
        python -m ibeis.model.hots.scoring --test-get_masks \
            --maskscore_mode=grid --show --prior_coeff=.5 --unconstrained_coeff=0 --constrained_coeff=.5
        python -m ibeis.model.hots.scoring --test-get_masks --qaid 4\
            --maskscore_mode=grid --show --prior_coeff=.5 --unconstrained_coeff=0 --constrained_coeff=.5
        python -m ibeis.model.hots.scoring --test-get_masks --qaid 86\
            --maskscore_mode=grid --show --prior_coeff=.5 --unconstrained_coeff=0 --constrained_coeff=.5 --grid_scale_factor=.5

        python -m ibeis.model.hots.scoring --test-get_masks --show --db PZ_MTEST --aid 1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> qreq_, unscored_cm = plh.testdata_scoring()
        >>> cm = unscored_cm
        >>> config = qreq_.qparams
        >>> # execute function
        >>> masks_list = get_masks(qreq_, cm, config)
        >>> if ut.show_was_requested():
        ...     import plottool as pt
        ...     show_coverage_mask(qreq_, cm, masks_list)
        ...     pt.show_if_requested()
    """
    make_mask_func, cov_cfg = get_mask_func(config)
    masks_iter = general_coverage_mask_generator(make_mask_func, qreq_, cm, config, cov_cfg)
    # copy weight mask as it comes back if you want to see them
    masks_list = [(weight_mask_m.copy(), weight_mask) for weight_mask_m, weight_mask  in masks_iter]
    return masks_list


def show_coverage_mask(qreq_, cm, masks_list, index=0, fnum=None):
    import plottool as pt
    from ibeis import viz
    fnum = pt.ensure_fnum(fnum)
    weight_mask_m, weight_mask = masks_list[index]
    stacked_weights, woff, hoff = pt.stack_images(weight_mask_m, weight_mask)
    #pt.imshow(255 * ut.norm_zero_one(stacked_weights), pnum=(1, 2, 1))
    wh1 = weight_mask_m.shape[0:2][::-1]
    wh2 = weight_mask.shape[0:2][::-1]
    score_list = list(score_masks(masks_list))
    daid, fm, fs = cm.daid_list[index], cm.fm_list[index], cm.fs_list[index]
    qaid = cm.qaid
    pt.figure(fnum=fnum, pnum=(1, 2, 1))
    # Draw image with bbox
    pt.imshow(255 * (stacked_weights), fnum=fnum, pnum=(1, 2, 1))
    pt.draw_bbox((   0,    0) + wh1, bbox_color=(0, 0, 1))
    pt.draw_bbox((woff, hoff) + wh2, bbox_color=(0, 0, 1))
    # Draw matches
    viz.viz_matches.show_matches2(qreq_.ibs, qaid, daid, fm, fs, draw_pts=False, draw_lines=True, draw_ell=False, fnum=fnum, pnum=(1, 2, 2))
    pt.set_figtitle('score=%.4f' % (score_list[index],))


def show_annot_weights(qreq_, aid, config={}):
    r"""
    DEMO FUNC

    CommandLine:
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db GZ_ALL --aid 1 --maskscore_mode='grid'
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db GZ_ALL --aid 1 --maskscore_mode='kpts'
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db PZ_Master0 --aid 1
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db PZ_MTEST --aid 1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> from ibeis.model.hots import _pipeline_helpers as plh
        >>> import plottool as pt
        >>> ibs, qreq_ = plh.get_pipeline_testdata(qaid_list=[1], defaultdb='testdb1', cmdline_ok=True)
        >>> aid = qreq_.get_external_qaids()[0]
        >>> config = qreq_.qparams
        >>> show_annot_weights(qreq_, aid, config)
        >>> pt.show_if_requested()
    """
    #import plottool as pt
    fnum = 1
    chipsize = qreq_.ibs.get_annot_chip_sizes(aid, qreq_=qreq_)
    chip  = qreq_.ibs.get_annot_chips(aid, qreq_=qreq_)
    qkpts = qreq_.ibs.get_annot_kpts(aid, qreq_=qreq_)
    weights = get_annot_kpts_basline_weights(qreq_, [aid], config)[0]
    make_mask_func, cov_cfg = get_mask_func(config)
    mask = make_mask_func(qkpts, chipsize, weights, resize=True, **cov_cfg)
    coverage_kpts.show_coverage_map(chip, mask, None, qkpts, fnum, ell_alpha=.2, show_mask_kpts=False)
    #pt.set_figtitle(mode)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.scoring
        python -m ibeis.model.hots.scoring --allexamples
        python -m ibeis.model.hots.scoring --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
