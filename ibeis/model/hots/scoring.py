from __future__ import absolute_import, division, print_function
import numpy as np
import vtool as vt
import utool as ut
from vtool import coverage_kpts
from vtool import coverage_grid
from ibeis.model.hots import hstypes
from ibeis.model.hots import distinctiveness_normalizer
from six.moves import zip, range  # NOQA
#profile = ut.profile
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[scoring]', DEBUG=False)


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
def get_kpts_distinctiveness(ibs, aid_list, config={}):
    """
    per-species disinctivness wrapper around ibeis cached function
    """
    aid_list = np.array(aid_list)
    sid_list = np.array(ibs.get_annot_species_rowids(aid_list))
    # Compute distinctivness separately for each species
    unique_sids, groupxs = vt.group_indicies(sid_list)
    aids_groups = vt.apply_grouping(aid_list, groupxs)
    species_text_list = ibs.get_species_texts(unique_sids)
    # Map distinctivness computation
    normer_list = [distinctiveness_normalizer.request_species_distinctiveness_normalizer(species)
                   for species in species_text_list]

    dcvs_kw = {
        'dcvs_K'        : config.get('dcvs_K', 5),
        'dcvs_power'    : config.get('dcvs_power', 1),
        'dcvs_min_clip' : config.get('dcvs_min_clip', .2),
        'dcvs_max_clip' : config.get('dcvs_max_clip', .3),
    }

    # Reduce to get results
    dstncvs_groups = [
        # uses ibeis non-persistant cache
        # code lives in manual_ibeiscontrol_funcs
        ibs.get_annot_kpts_distinctiveness(aids, dstncvs_normer=dstncvs_normer, **dcvs_kw)
        for dstncvs_normer, aids in zip(normer_list, aids_groups)
    ]
    dstncvs_list = vt.invert_apply_grouping(dstncvs_groups, groupxs)
    return dstncvs_list


def get_annot_kpts_basline_weights(ibs, qaid, config={}):
    # TODO: clip the fgweights
    # TODO; normalize and paramatarize and clean
    qdstncvs  = get_kpts_distinctiveness(ibs, [qaid])[0]
    qfgweight = ibs.get_annot_fgweights([qaid], ensure=True)[0]
    #print(ut.get_stats_str(qfgweight))
    #print(ut.get_stats_str(qdstncvs))
    #featweight_min = .98
    #featweight_min = .9
    #featweight_max = 1.0
    #qfgweight = vt.clipnorm(qfgweight, featweight_min, featweight_max)
    #print(ut.get_stats_str(qfgweight))

    #weights = (qfgweight * qdstncvs) ** .5
    weights = (qfgweight + qdstncvs) / 2
    weights = (qfgweight * qdstncvs)
    return weights


#### MASK COVERAGE SCORING ####


@profile
def compute_kpts_coverage_score(ibs, qaid, daid_list, fm_list, fs_list, config={}):
    cov_cfg_keys = [
        'cov_agg_mode',
        'cov_blur_ksize',
        'cov_blur_on',
        'cov_blur_sigma',
        'cov_remove_scale',
        'cov_remove_shape',
        'cov_scale_factor',
        'cov_size_penalty_frac',
        'cov_size_penalty_on',
        'cov_size_penalty_power',
    ]
    cov_cfg = {key: config.get(key) for key in cov_cfg_keys}
    make_mask_func = coverage_kpts.make_kpts_coverage_mask
    score_list = list(general_coverage_score_generator(make_mask_func, ibs, qaid, fm_list, fs_list, config, cov_cfg))
    return score_list


@profile
def compute_grid_coverage_score(ibs, qaid, daid_list, fm_list, fs_list, config={}):
    """ Ignore: qaid = 6 daid_list = [41] fm_list = [[]] fs_list = [[]] """
    cov_cfg = {
        'grid_scale_factor' : config.get('grid_scale_factor', .2),
        'grid_steps'        : config.get('grid_steps', 2),
        'grid_sigma'        : config.get('grid_sigma', 1.6),
    }
    make_mask_func = coverage_grid.make_grid_coverage_mask
    score_list = list(general_coverage_score_generator(make_mask_func, ibs, qaid, fm_list, fs_list, config, cov_cfg))
    return score_list


def general_coverage_score_generator(*args, **kwargs):
    masks_iter = general_coverage_mask_generator(*args, **kwargs)
    return score_masks(masks_iter)


def score_masks(masks_iter):
    for weight_mask_m, weight_mask in masks_iter:
        coverage_score = weight_mask_m.sum() / weight_mask.sum()
        yield coverage_score


def general_coverage_mask_generator(make_mask_func, ibs, qaid, fm_list, fs_list,
                                    config, cov_cfg):
    """
    CommandLine:
        python -m ibeis.model.hots.scoring --test-general_coverage_mask_generator:0
        python -m ibeis.model.hots.scoring --test-general_coverage_mask_generator:1

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> qreq_, unscored_cm = testdata_scoring()
        >>> qaid, fm_list, fs_list = ut.dict_take(unscored_cm.__dict__, 'qaid, fm_list, fs_list'.split(', '))
        >>> ibs = qreq_.ibs
        >>> ibs = qreq_.ibs
        >>> score_list = compute_grid_coverage_score(ibs, qaid, daid_list, fm_list, fs_list)
        >>> result = ut.list_str(score_list, precision=3)
        >>> print(result)

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> qreq_, unscored_cm = testdata_scoring()
        >>> qaid, fm_list, fs_list = ut.dict_take(unscored_cm.__dict__, 'qaid, fm_list, fs_list'.split(', '))
        >>> ibs = qreq_.ibs
        >>> score_list = compute_kpts_coverage_score(ibs, qaid, daid_list, fm_list, fs_list)
        >>> result = ut.list_str(score_list, precision=3)
        >>> print(result)
    """
    print('make_mask_func = %r' % (make_mask_func,))
    print('cov_cfg = %s' % (ut.dict_str(cov_cfg),))
    # Distinctivness and foreground weight
    weights = get_annot_kpts_basline_weights(ibs, qaid, config)
    # Denominator weight mask
    chipsize    = ibs.get_annot_chipsizes(qaid)
    qkpts       = ibs.get_annot_kpts(qaid)
    weight_mask = make_mask_func(qkpts, chipsize, weights, resize=False, **cov_cfg)
    # Prealloc data for loop
    weight_mask_m = weight_mask.copy()
    # Apply weighted scoring to matches
    for fm, fs in zip(fm_list, fs_list):
        # Get matching query keypoints
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


def show_coverage_mask(ibs, cm, masks_list, fnum=None):
    import plottool as pt
    from ibeis import viz
    if fnum is None:
        fnum = pt.next_fnum()
    index = 0
    weight_mask_m, weight_mask = masks_list[0]
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
    viz.viz_matches.show_matches2(ibs, qaid, daid, fm, fs, draw_pts=False, draw_lines=True, draw_ell=False, fnum=fnum, pnum=(1, 2, 2))
    pt.set_figtitle('score=%.4f' % (score_list[index],))


def get_masks(ibs, cm, config={}):
    r"""

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

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> qreq_, unscored_cm = testdata_scoring()
        >>> cm = unscored_cm
        >>> ibs = qreq_.ibs
        >>> config = qreq_.qparams
        >>> # execute function
        >>> masks_list = get_masks(ibs, cm, config)
        >>> if ut.show_was_requested():
        ...     import plottool as pt
        ...     show_coverage_mask(ibs, cm, masks_list)
        ...     pt.show_if_requested()
    """
    #if config is None:
    #    config =
    #print(config.maskscore_mode)
    qaid = cm.qaid
    fm_list = cm.fm_list
    fs_list = cm.fs_list
    maskscore_mode = config.get('maskscore_mode', 'grid')
    #print(maskscore_mode)
    make_mask_func = {
        'grid': coverage_grid.make_grid_coverage_mask,
        'kpts': coverage_kpts.make_kpts_coverage_mask
    }[maskscore_mode]

    cov_cfg = ut.filter_valid_kwargs(make_mask_func, config)

    masks_iter = general_coverage_mask_generator(make_mask_func, ibs, qaid, fm_list, fs_list, config, cov_cfg)
    masks_list = [(weight_mask_m.copy(), weight_mask) for weight_mask_m, weight_mask  in masks_iter]
    #if ut.DEBUG2:
    #    score_list = list(score_masks(masks_list))
    #    assert score_list == list(general_coverage_score_generator(make_mask_func, ibs, qaid, fm_list, fs_list, config, cov_cfg))
    return masks_list


def testdata_scoring():
    from ibeis.model.hots import vsone_pipeline
    ibs, qreq_, prior_cm = vsone_pipeline.testdata_matching()
    config = qreq_.qparams
    unscored_cm = vsone_pipeline.refine_matches(ibs, prior_cm, config)
    return qreq_, unscored_cm
    #qaid      = unscored_cm.qaid
    #daid_list = unscored_cm.daid_list
    #fm_list   = unscored_cm.fm_list
    #fs_list   = unscored_cm.fs_list
    #H_list    = prior_cm.H_list
    #ibs, qreq_, qaid, daid_list, H_list, prior_chipmatch, prior_filtkey_list = vsone_pipeline.testdata_matching()
    # run vsone
    #fm_list, fs_list = vsone_pipeline.compute_query_matches(ibs, qaid, daid_list, H_list)  # 35.8
    #qaid = qaid_list[0]
    #chipmatch = qaid2_chipmatch[qaid]
    #fm = chipmatch.aid2_fm[daid]
    #fsv = chipmatch.aid2_fsv[daid]


def show_annot_weights(ibs, aid, mode='dstncvs'):
    r"""
    DEMO FUNC

    Args:
        ibs (IBEISController):  ibeis controller object
        aid (int):  annotation id
        mode (str):

    CommandLine:
        alias show_annot_weights='python -m ibeis.model.hots.scoring --test-show_annot_weights --show'
        show_annot_weights
        show_annot_weights --db PZ_MTEST --aid 1 --mode 'dstncvs'
        show_annot_weights --db PZ_MTEST --aid 1 --mode 'fgweight'&
        show_annot_weights --db GZ_ALL --aid 1 --mode 'dstncvs'
        show_annot_weights --db GZ_ALL --aid 1 --mode 'fgweight'&


        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db GZ_ALL --aid 1 --mode 'dstncvs'
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db PZ_MTEST --aid 1 --mode 'dstncvs'
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db GZ_ALL --aid 1 --mode 'fgweight'
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db PZ_MTEST --aid 1 --mode 'fgweight'

        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db PZ_MTEST --aid 1 --mode 'fgweight' --kptscov

        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db GZ_ALL --aid 1 --mode 'dstncvs*fgweight'
        python -m ibeis.model.hots.scoring --test-show_annot_weights --show --db PZ_MTEST --aid 1 --mode 'dstncvs*fgweight'

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> import plottool as pt
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(ut.get_argval('--db', type_=str, default='testdb1'))
        >>> aid = ut.get_argval('--aid', type_=int, default=1)
        >>> mode = ut.get_argval('--mode', type_=str, default='dstncvs')
        >>> # execute function
        >>> show_annot_weights(ibs, aid, mode)
        >>> pt.show_if_requested()
    """
    import plottool as pt
    fnum = 1
    chipsize = ibs.get_annot_chipsizes(aid)
    chip = ibs.get_annot_chips(aid)
    qkpts = ibs.get_annot_kpts(aid)
    mode = mode.strip('\'')  # win32 hack
    config = {}
    weights = get_annot_kpts_basline_weights(ibs, aid, config)
    if ut.get_flag('--kptscov'):
        make_mask_func = coverage_kpts.make_kpts_coverage_mask
        mask = make_mask_func(qkpts, chipsize, weights, mode='max', resize=True)
    else:
        make_mask_func = coverage_grid.make_grid_coverage_mask
        mask = make_mask_func(qkpts, chipsize, weights, grid_scale_factor=.5,
                              grid_steps=2, grid_sigma=3.0, resize=True)
    #mask = (mask / mask.max()) ** 2
    coverage_kpts.show_coverage_map(chip, mask, None, qkpts, fnum, ell_alpha=.2, show_mask_kpts=False)
    pt.set_figtitle(mode)


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
