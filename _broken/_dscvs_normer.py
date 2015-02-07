def old_test_single_annot_distinctiveness_params(ibs, aid):
    r"""

    CommandLine:
        python -m ibeis.model.hots.distinctiveness_normalizer --test-old_test_single_annot_distinctiveness_params --show
        python -m ibeis.model.hots.distinctiveness_normalizer --test-old_test_single_annot_distinctiveness_params --show --db GZ_ALL

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.distinctiveness_normalizer import *  # NOQA
        >>> import plottool as pt
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(ut.get_argval('--db', type_=str, default='PZ_MTEST'))
        >>> aid = ut.get_argval('--aid', type_=int, default=1)
        >>> # execute function
        >>> old_test_single_annot_distinctiveness_params(ibs, aid)
        >>> pt.show_if_requested()
    """
    ####
    # TODO: Also paramatarize the downweighting based on the keypoint size
    ####
    # HACK IN ABILITY TO SET CONFIG
    from ibeis.dev.main_commands import postload_commands
    postload_commands(ibs, None)

    from vtool import coverage_image
    import plottool as pt
    from plottool import interact_impaint

    #cfglbl_list = cfgdict_list
    #ut.all_dict_combinations_lbls(varied_dict)

    # Get info to find distinctivness of
    species_text = ibs.get_annot_species(aid)
    vecs = ibs.get_annot_vecs(aid)
    kpts = ibs.get_annot_kpts(aid)
    print(kpts)
    chip = ibs.get_annot_chips(aid)
    chipsize = ibs.get_annot_chipsizes(aid)

    # Paramater space to search
    # TODO: use slicing to control the params being varied
    # Use GridSearch class to modify paramaters as you go.

    gauss_patch_varydict = {
        'gauss_shape': [(7, 7), (19, 19), (41, 41), (5, 5), (3, 3)],
        'gauss_sigma_frac': [.2, .5, .7, .95],
    }
    cov_blur_varydict = {
        'cov_blur_on': [True, False],
        'cov_blur_ksize': [(5, 5,),  (7, 7), (17, 17)],
        'cov_blur_sigma': [5.0, 1.2],
    }
    dstncvs_varydict = {
        'dcvs_power': [.01, .1, .5, 1.0],
        'dcvs_clip_max': [.05, .1, .2, .5],
        'dcvs_K': [2, 3, 5],
    }
    size_penalty_varydict = {
        'remove_affine_information': [False, True],
        'constant_scaling': [False, True],
        'size_penalty_on': [True, False],
        'size_penalty_power': [.5, .1, 1.0],
        'size_penalty_scale': [.1, 1.0],
    }
    keyval_iter = ut.iflatten([
        dstncvs_varydict.items(),
        gauss_patch_varydict.items(),
        cov_blur_varydict.items(),
        size_penalty_varydict.items(),
    ])

    # Dont vary most paramaters, specify how much of their list can be used
    param_slice_dict = {
        'dcvs_power'                  : slice(0, 2),
        'dcvs_K'                  : slice(0, 2),
        'dcvs_clip_max'      : slice(0, 2),
        'dcvs_clip_max'      : slice(0, 2),
        #'gauss_shape'        : slice(0, 3),
        'gauss_sigma_frac'          : slice(0, 2),
        'remove_affine_information' : slice(0, 2),
        'constant_scaling'          : slice(0, 2),
        'size_penalty_on'           : slice(0, 2),
        #'cov_blur_on'        : slice(0, 2),
        #'cov_blur_ksize'     : slice(0, 2),
        #'cov_blur_sigma'     : slice(0, 1),
        #'size_penalty_power' : slice(0, 2),
        #'size_penalty_scale' : slice(0, 2),
    }
    varied_dict = {
        key: val[param_slice_dict.get(key, slice(0, 1))]
        for key, val in keyval_iter
    }

    def constrain_config(cfg):
        """ encode what makes a configuration feasible """
        if cfg['cov_blur_on'] is False:
            cfg['cov_blur_ksize'] = None
            cfg['cov_blur_sigma'] = None
        if cfg['constant_scaling'] is True:
            cfg['remove_affine_information'] = True
            cfg['size_penalty_on'] = False
        if cfg['remove_affine_information'] is True:
            cfg['gauss_shape'] = (41, 41)
        if cfg['size_penalty_on'] is False:
            cfg['size_penalty_power'] = None
            cfg['size_penalty_scale'] = None

    print('Varied Dict: ')
    print(ut.dict_str(varied_dict))

    cfgdict_list, cfglbl_list = ut.make_constrained_cfg_and_lbl_list(varied_dict, constrain_config)

    # Get groundtruthish distinctivness map
    # for objective function
    GT_IS_DSTNCVS = 255
    GT_NOT_DSTNCVS = 100
    GT_UNKNOWN = 0
    label_colors = [GT_IS_DSTNCVS, GT_NOT_DSTNCVS, GT_UNKNOWN]
    gtmask = interact_impaint.cached_impaint(chip, 'dstncvnss',
                                             label_colors=label_colors,
                                             aug=True, refine=ut.get_argflag('--refine'))
    true_dstncvs_mask = gtmask == GT_IS_DSTNCVS
    false_dstncvs_mask = gtmask == GT_NOT_DSTNCVS

    true_dstncvs_mask_sum = true_dstncvs_mask.sum()
    false_dstncvs_mask_sum = false_dstncvs_mask.sum()

    def distinctiveness_objective_function(dstncvs_mask):
        true_mask  = true_dstncvs_mask * dstncvs_mask
        false_mask = false_dstncvs_mask * dstncvs_mask
        true_score = true_mask.sum() / true_dstncvs_mask_sum
        false_score = false_mask.sum() / false_dstncvs_mask_sum
        score = true_score * (1 - false_score)
        return score

    # Load distinctivness normalizer
    with ut.Timer('Loading Distinctivness Normalizer for %s' % (species_text)):
        dstcvnss_normer = request_species_distinctiveness_normalizer(species_text)

    # Get distinctivness over all params
    dstncvs_list = [dstcvnss_normer.get_distinctiveness(vecs, **cfgdict)
                    for cfgdict in ut.ProgressIter(cfgdict_list, lbl='get dstcvns')]

    # Then compute the distinctinvess coverage map
    #gauss_shape = kwargs.get('gauss_shape', (19, 19))
    #sigma_frac = kwargs.get('sigma_frac', .3)
    dstncvs_mask_list = [
        coverage_image.make_coverage_mask(
            kpts, chipsize, fx2_score=dstncvs, mode='max', return_patch=False, **cfg)
        for cfg, dstncvs in ut.ProgressIter(zip(cfgdict_list, dstncvs_list), lbl='Warping Image')
    ]
    score_list = [distinctiveness_objective_function(dstncvs_mask) for dstncvs_mask in dstncvs_mask_list]

    fnum = 1

    def show_covimg_result(img, fnum=None, pnum=None):
        pt.imshow(255 * img, fnum=fnum, pnum=pnum)

    ut.interact_gridsearch_result_images(
        show_covimg_result, cfgdict_list, cfglbl_list, dstncvs_mask_list,
        score_list=score_list, fnum=fnum, figtitle='dstncvs gridsearch')

    # Show subcomponents of grid search
    gauss_patch_cfgdict_list, gauss_patch_cfglbl_list = ut.get_cfgdict_lbl_list_subset(cfgdict_list, gauss_patch_varydict)
    patch_list = [coverage_image.get_gaussian_weight_patch(**cfgdict)
                  for cfgdict in ut.ProgressIter(gauss_patch_cfgdict_list, lbl='patch cfg')]

    ut.interact_gridsearch_result_images(
        show_covimg_result, gauss_patch_cfgdict_list, gauss_patch_cfglbl_list,
        patch_list, fnum=fnum + 1, figtitle='gaussian patches')

    patch = patch_list[0]

    # Show the first mask in more depth
    dstncvs = dstncvs_list[0]
    dstncvs_mask = dstncvs_mask_list[0]
    coverage_image.show_coverage_map(chip, dstncvs_mask, patch, kpts, fnum=fnum + 2, ell_alpha=.2, show_mask_kpts=False)

    pt.imshow(gtmask, fnum=fnum + 3, pnum=(1, 2, 1), title='ground truth distinctiveness')
    pt.imshow(chip, fnum=fnum + 3, pnum=(1, 2, 2))
    pt.present()
    #pt.iup()

    #ut.print_resource_usage()
    #pt.set_figtitle(mode)
    #pass


#def test_example():
#    import scipy.linalg as spl
#    M = np.array([
#        [1.0, 0.6, 0. , 0. , 0. ],
#        [0.6, 1.0, 0.5, 0.2, 0. ],
#        [0. , 0.5, 1.0, 0. , 0. ],
#        [0. , 0.2, 0. , 1.0, 0.8],
#        [0. , 0. , 0. , 0.8, 1.0],
#    ])
#    M_ = M / M.sum(axis=0)[:, None]
#    #eigvals, eigvecs = np.linalg.eigh(M_)
#    #, left=True, right=False)
#    eigvals, eigvecs = spl.eig(M_, left=True, right=False)
#    index = np.where(np.isclose(eigvals, 1))[0]
#    pi = stationary_vector = eigvecs.T[index]
#    pi_test = pi.dot(M_)
#    pi / pi.sum()
#    print(pi / np.linalg.norm(pi))
#    print(pi_test / np.linalg.norm(pi_test))

#    M = np.array([
#        [1.0, 0.6],
#        [0.6, 1.0],
#    ])
#    M_ = M / M.sum(axis=0)[:, None]
#    #eigvals, eigvecs = np.linalg.eigh(M_)
#    #, left=True, right=False)
#    eigvals, eigvecs = spl.eig(M_, left=True, right=False)
#    index = np.where(np.isclose(eigvals, 1))[0]
#    pi = stationary_vector = eigvecs.T[index]
#    pi_test = pi.dot(M_)
#    pi / pi.sum()
#    print(pi / np.linalg.norm(pi))
#    print(pi_test / np.linalg.norm(pi_test))
#    #pi = pi / pi.sum()



