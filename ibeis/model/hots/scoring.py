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
    # apply selectivity function
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


@profile
#@ut.memprof
def compute_kpts_coverage_score(ibs, qaid, daid_list, fm_list, fs_list, config={}):
    """
    Returns a grayscale chip match which represents which pixels
    should be matches in order for a candidate to be considered a match.

    CommandLine:
        python -m ibeis.model.hots.scoring --test-compute_kpts_coverage_score

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> # build test data
        >>> ibs, qaid, daid_list, fm_list, fs_list = testdata_scoring()
        >>> # execute function
        >>> score_list = compute_kpts_coverage_score(ibs, qaid, daid_list, fm_list, fs_list)
        >>> # verify results
        >>> result = str(score_list)
        >>> print(result)
    """
    # Distinctivness and foreground weight
    qdstncvs  = get_kpts_distinctiveness(ibs, [qaid])[0]
    qfgweight = ibs.get_annot_fgweights([qaid], ensure=True)[0]
    # TODO: clip the fgweights
    #
    # Denominator weight mask
    qchipsize = ibs.get_annot_chipsizes(qaid)
    qkpts     = ibs.get_annot_kpts(qaid)
    kptscov_cfg = {
        'cov_agg_mode'           : 'max',
        'cov_blur_ksize'         : (19, 19),
        'cov_blur_on'            : True,
        'cov_blur_sigma'         : 5.0,
        'cov_remove_scale'       : True,
        'cov_remove_shape'       : True,
        'cov_scale_factor'       : .2,
        'cov_size_penalty_frac'  : .1,
        'cov_size_penalty_on'    : True,
        'cov_size_penalty_power' : .5,
    }
    weights = (qfgweight * qdstncvs) ** .5
    weight_mask = coverage_kpts.make_kpts_coverage_mask(
        qkpts, qchipsize, weights, resize=False,
        return_patch=False, **kptscov_cfg)
    # Apply weighted scoring to matches
    score_list = []
    for fm, fs in zip(fm_list, fs_list):
        # Get matching query keypoints
        qkpts_m     = qkpts.take(fm.T[0], axis=0)
        qdstncvs_m  = qdstncvs.take(fm.T[0], axis=0)
        qfgweight_m = qfgweight.take(fm.T[0], axis=0)
        weights_m   = fs * qdstncvs_m * qfgweight_m
        weight_mask_m = coverage_kpts.make_kpts_coverage_mask(
            qkpts_m, qchipsize, weights_m, resize=False, **kptscov_cfg)
        coverage_score = weight_mask_m.sum() / weight_mask.sum()
        score_list.append(coverage_score)
    del weight_mask
    return score_list


@profile
def compute_grid_coverage_score(ibs, qaid, daid_list, fm_list, fs_list, config={}):
    """

    Ignore:
        qaid = 6
        daid_list = [41]
        fm_list = [[]]
        fs_list = [[]]

    CommandLine:
        python -m ibeis.model.hots.scoring --test-compute_grid_coverage_score

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.scoring import *  # NOQA
        >>> # build test data
        >>> ibs, qaid, daid_list, fm_list, fs_list = testdata_scoring()
        >>> score_list = compute_grid_coverage_score(ibs, qaid, daid_list, fm_list, fs_list)
        >>> print(score_list)
    """
    qdstncvs  = get_kpts_distinctiveness(ibs, [qaid])[0]
    qfgweight = ibs.get_annot_fgweights([qaid], ensure=True)[0]
    # Make weight mask
    chipsize = qchipsize = ibs.get_annot_chipsizes(qaid)  # NOQA
    kpts = qkpts = ibs.get_annot_kpts(qaid)  # NOQA
    #mode = 'max'
    # Foregroundness*Distinctiveness weight mask
    weights = (qfgweight * qdstncvs) ** .5
    gridcov_cfg = {
        'grid_scale_factor' : config.get('grid_scale_factor', .2),
        'grid_steps'        : config.get('grid_steps', 2),
        'grid_sigma'        : config.get('grid_sigma', 1.6),
    }
    weight_mask = coverage_grid.make_grid_coverage_mask(kpts, chipsize, weights, resize=False, **gridcov_cfg)
    # Prealloc data for loop
    weight_mask_m = weight_mask.copy()
    # Apply weighted scoring to matches
    score_list = []
    for fm, fs in zip(fm_list, fs_list):
        # Get matching query keypoints
        qkpts_m     = qkpts.take(fm.T[0], axis=0)
        qdstncvs_m  = qdstncvs.take(fm.T[0], axis=0)
        qfgweight_m = qfgweight.take(fm.T[0], axis=0)
        weights_m   = fs * qdstncvs_m * qfgweight_m
        weight_mask_m = coverage_grid.make_grid_coverage_mask(
            qkpts_m, chipsize, weights_m, out=weight_mask_m, resize=False, **gridcov_cfg)
        coverage_score = weight_mask_m.sum() / weight_mask.sum()
        score_list.append(coverage_score)
    return score_list


def testdata_scoring():
    from ibeis.model.hots import vsone_pipeline
    ibs, qreq_, qaid, daid_list, H_list, prior_chipmatch, prior_filtkey_list = vsone_pipeline.testdata_matching()
    # run vsone
    fm_list, fs_list = vsone_pipeline.compute_query_matches(ibs, qaid, daid_list, H_list)  # 35.8
    return ibs, qaid, daid_list, fm_list, fs_list
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
    import functools
    import plottool as pt
    fnum = 1
    chipsize = ibs.get_annot_chipsizes(aid)
    #chipshape = chipsize[::-1]
    chip = ibs.get_annot_chips(aid)
    kpts = ibs.get_annot_kpts(aid)
    mode = mode.strip('\'')  # win32 hack
    fx2_score = 1.0
    weight_fn_dict = {
        'dstncvs': functools.partial(get_kpts_distinctiveness, ibs),
        'fgweight': ibs.get_annot_fgweights,
    }
    key_list = mode.split('*')
    for key in key_list:
        #print(key)
        get_weight = weight_fn_dict[key]
        fx2_weight = get_weight([aid])[0]
        #print(fx2_weight)
        fx2_score = fx2_score * fx2_weight
    fx2_score **= 1.0 / len(key_list)  # geometric average
    #mask, patch = coverage_kpts.make_kpts_coverage_mask(
    #    kpts, chipshape, fx2_score=fx2_score, mode='max')
    mask = coverage_grid.make_grid_coverage_mask(
        kpts, chipsize, fx2_score, grid_scale_factor=.5, grid_steps=2, grid_sigma=3.0,
        resize=True)
    #mask = (mask / mask.max()) ** 2
    coverage_kpts.show_coverage_map(chip, mask, None, kpts, fnum, ell_alpha=.2, show_mask_kpts=False)
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
