# -*- coding: utf-8 -*-
"""
Interface to pyrf random forest object detection.
"""
from __future__ import absolute_import, division, print_function
from os.path import exists, join
from wbia.algo.detect import grabmodels
import utool as ut
import vtool as vt
from six.moves import zip, map
import cv2
import random

(print, rrr, profile) = ut.inject2(__name__, '[randomforest]')

if not ut.get_argflag('--no-pyrf'):
    try:
        import pyrf
    except ImportError:
        print('WARNING Failed to import pyrf. ' 'Randomforest detection is unavailable')
        if ut.SUPER_STRICT:
            raise

VERBOSE_RF = ut.get_argflag('--verbrf') or ut.VERBOSE


def train_gid_list(
    ibs, gid_list, trees_path=None, species=None, setup=True, teardown=False, **kwargs
):
    """
    Args:
        gid_list (list of int): the list of IBEIS image_rowids that need detection
        trees_path (str): the path that the trees will be saved into (along
            with temporary training inventory folders that are deleted once
            training is finished)
        species (str): the species that should be used to assign to the newly
            trained trees

    Kwargs (optional): refer to the PyRF documentation for configuration settings

    Returns:
        None
    """
    print(
        '[randomforest.train()] training with %d gids and species=%r'
        % (len(gid_list), species,)
    )
    if trees_path is None and species is not None:
        trees_path = join(ibs.get_cachedir(), 'trees', species)

    # Get positive chip paths
    if species is None:
        aids_list = ibs.get_image_aids(gid_list)
    else:
        aids_list = ibs.get_image_aids_of_species(gid_list, species)

    # ##### TEMP #####
    # gid_list_ = []
    # aids_list_ = []
    # for gid, aid_list in zip(gid_list, aids_list):
    #     if len(aid_list) > 1:
    #         gid_list_.append(gid)
    #         aids_list_.append(aid_list)
    #     elif len(aid_list) == 1:
    #         (xtl, ytl, width, height) = ibs.get_annot_bboxes(aid_list)[0]
    #         if xtl > 5 and ytl > 5:
    #             gid_list_.append(gid)
    #             aids_list_.append(aid_list)
    # gid_list = gid_list_
    # aids_list = aids_list_
    # kwargs['trees_max_patches'] = 100000
    # ##### TEMP #####

    aid_list = ut.flatten(aids_list)
    train_pos_cpath_list = ibs.get_annot_chip_fpath(aid_list)

    # Ensure directories for negatives
    negatives_cache = join(ibs.get_cachedir(), 'pyrf_train_negatives')
    if (setup and not exists(negatives_cache)) or setup == 'force':
        # Force Check
        if exists(negatives_cache):
            ut.remove_dirs(negatives_cache)
        ut.ensuredir(negatives_cache)
        # Get negative chip paths
        print(
            '[randomforest.train()] Mining %d negative patches'
            % (len(train_pos_cpath_list),)
        )
        train_neg_cpath_list = []
        while len(train_neg_cpath_list) < len(train_pos_cpath_list):
            sample = random.randint(0, len(gid_list) - 1)
            gid = gid_list[sample]
            img_width, img_height = ibs.get_image_sizes(gid)
            size = min(img_width, img_height)
            if species is None:
                aid_list = ibs.get_image_aids(gid)
            else:
                aid_list = ibs.get_image_aids_of_species(gid, species)
            annot_bbox_list = ibs.get_annot_bboxes(aid_list)
            # Find square patches
            square = random.randint(int(size / 4), int(size / 2))
            xmin = random.randint(0, img_width - square)
            xmax = xmin + square
            ymin = random.randint(0, img_height - square)
            ymax = ymin + square
            if _valid_candidate((xmin, xmax, ymin, ymax), annot_bbox_list):
                if VERBOSE_RF:
                    print(
                        '[%d / %d] MINING NEGATIVE PATCH (%04d, %04d, %04d, %04d) FROM GID %d'
                        % (
                            len(train_neg_cpath_list),
                            len(train_pos_cpath_list),
                            xmin,
                            xmax,
                            ymin,
                            ymax,
                            gid,
                        )
                    )
                img = ibs.get_images(gid)
                img_path = join(
                    negatives_cache, 'neg_%07d.JPEG' % (len(train_neg_cpath_list),)
                )
                img = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(img_path, img)
                train_neg_cpath_list.append(img_path)
    else:
        train_neg_cpath_list = ut.ls(negatives_cache, '*.JPEG')
        # direct = Directory(negatives_cache, include_extensions=['JPEG'])
        # train_neg_cpath_list = direct.files()

    # Train trees
    train_gpath_list(
        ibs,
        train_pos_cpath_list,
        train_neg_cpath_list,
        trees_path=trees_path,
        species=species,
        **kwargs,
    )

    # Remove cached negatives directory
    if teardown:
        ut.remove_dirs(negatives_cache)


def train_gpath_list(
    ibs, train_pos_cpath_list, train_neg_cpath_list, trees_path=None, **kwargs
):
    """
    Args:
        train_pos_cpath_list (list of str): the list of positive image paths
            for training
        train_neg_cpath_list (list of str): the list of negative image paths
            for training
        trees_path (str): the path that the trees will be saved into (along
            with temporary training inventory folders that are deleted once
            training is finished)
        species (str, optional): the species that should be used to assign to
            the newly trained trees

    Kwargs (optional): refer to the PyRF documentation for configuration settings

    Returns:
        None
    """
    if trees_path is None:
        trees_path = join(ibs.get_treesdir(), 'generic')
    # Train trees
    detector = pyrf.Random_Forest_Detector()
    detector.train(train_pos_cpath_list, train_neg_cpath_list, trees_path, **kwargs)


def detect_gpath_list_with_species(ibs, gpath_list, species, **kwargs):
    """
    Args:
        gpath_list (list of str): the list of image paths that need detection
        species (str): the species that should be used to select the pre-trained
            random forest model
        downsample (bool, optional): a flag to indicate if the original image
            sizes should be used; defaults to True

            True:  ibs.get_image_detectpaths() is used
            False: ibs.get_image_paths() is used

    Kwargs (optional): refer to the PyRF documentation for configuration settings

    Yields:
        iter
    """
    tree_path_list = _get_models(ibs, species)
    results_iter = detect(ibs, gpath_list, tree_path_list, **kwargs)
    return results_iter


def detect_gid_list_with_species(ibs, gid_list, species, downsample=True, **kwargs):
    """
    Args:
        gid_list (list of int): the list of IBEIS image_rowids that need detection
        species (str): the species that should be used to select the pre-trained
            random forest model
        downsample (bool, optional): a flag to indicate if the original image
            sizes should be used; defaults to True

            True:  ibs.get_image_detectpaths() is used
            False: ibs.get_image_paths() is used

    Kwargs (optional): refer to the PyRF documentation for configuration settings

    Returns:
        iter

    CommandLine:
        python -m wbia.algo.detect.randomforest --test-detect_gid_list_with_species

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.detect.randomforest import *  # NOQA
        >>> from wbia.algo.detect.randomforest import _get_models  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> species = wbia.const.TEST_SPECIES.ZEB_PLAIN
        >>> gid_list = ibs.get_valid_gids()
        >>> downsample = True
        >>> kwargs = {}
        >>> # execute function
        >>> result = detect_gid_list_with_species(ibs, gid_list, species, downsample)
        >>> # verify results
        >>> print(result)
    """
    tree_path_list = _get_models(ibs, species)
    results_iter = detect_gid_list(
        ibs, gid_list, tree_path_list, downsample=downsample, verbose=False, **kwargs
    )
    return results_iter


def detect_gid_list(ibs, gid_list, tree_path_list, downsample=True, **kwargs):
    """
    Args:
        gid_list (list of int): the list of IBEIS image_rowids that need detection
        tree_path_list (list of str): the list of trees to load for detection
        downsample (bool, optional): a flag to indicate if the original image
            sizes should be used; defaults to True

            True:  ibs.get_image_detectpaths() is used
            False: ibs.get_image_paths() is used

    Kwargs (optional): refer to the PyRF documentation for configuration settings

    Yields:
        results (list of dict)
    """
    # Get new gpaths if downsampling
    if downsample:
        gpath_list = ibs.get_image_detectpaths(gid_list)
        neww_list = [vt.open_image_size(gpath)[0] for gpath in gpath_list]
        oldw_list = [oldw for (oldw, oldh) in ibs.get_image_sizes(gid_list)]
        downsample_list = [oldw / neww for oldw, neww in zip(oldw_list, neww_list)]
    else:
        gpath_list = ibs.get_image_paths(gid_list)
        downsample_list = [None] * len(gpath_list)
    # Run detection
    results_iter = detect(ibs, gpath_list, tree_path_list, **kwargs)
    # Upscale the results
    for gid, downsample, (gpath, result_list) in zip(
        gid_list, downsample_list, results_iter
    ):
        # Upscale the results back up to the original image size
        if downsample is not None and downsample != 1.0:
            for result in result_list:
                for key in ['centerx', 'centery', 'xtl', 'ytl', 'width', 'height']:
                    result[key] = int(result[key] * downsample)
        yield gid, gpath, result_list


def detect(ibs, gpath_list, tree_path_list, **kwargs):
    """
    Args:
        gpath_list (list of str): the list of image paths that need detection
        tree_path_list (list of str): the list of trees to load for detection

    Kwargs (optional): refer to the PyRF documentation for configuration settings

    Returns:
        iter
    """
    # Get scales from detect config, if not specified
    if 'scale_list' not in kwargs.keys():
        kwargs['scale_list'] = list(map(float, ibs.cfg.detect_cfg.scale_list.split(',')))
        assert all([isinstance(scale, float) for scale in kwargs['scale_list']])

    verbose = kwargs.get('verbose', ut.VERBOSE)
    if verbose:
        print(
            '[randomforest.detect()] Detecting with %d trees with scale_list=%r'
            % (len(tree_path_list), kwargs['scale_list'],)
        )

    # Run detection
    detector = pyrf.Random_Forest_Detector(verbose=verbose)
    forest = detector.forest(tree_path_list)
    results_iter = detector.detect(forest, gpath_list, **kwargs)
    return results_iter


########################


def _overlap_percentage(minmax_tup1, minmax_tup2):
    (xmin1, xmax1, ymin1, ymax1) = minmax_tup1
    (xmin2, xmax2, ymin2, ymax2) = minmax_tup2
    width1, height1 = xmax1 - xmin1, ymax1 - ymin1
    width2, height2 = xmax2 - xmin2, ymax2 - ymin2
    x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    area_overlap = float(x_overlap * y_overlap)
    area_total = min(width1 * height1, width2 * height2)
    percentage = area_overlap / area_total
    return percentage


def _valid_candidate(candidate, annot_bbox_list, overlap=0.0, tries=10):
    for i in range(tries):
        valid = True
        for annot_bbox in annot_bbox_list:
            xtl, ytl, width, height = annot_bbox
            xmin, xmax, ymin, ymax = xtl, xtl + width, ytl, ytl + height
            if _overlap_percentage(candidate, (xmin, xmax, ymin, ymax)) > overlap:
                valid = False
                break  # break inner loop
        if valid:
            return True
    return False


def _get_models(ibs, species, modeldir='default', cfg_override=True, verbose=VERBOSE_RF):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        species (?):
        modeldir (str): (default = 'default')
        cfg_override (bool): (default = True)
        verbose (bool):  verbosity flag(default = False)

    Returns:
        ?: fpath_list

    CommandLine:
        python -m wbia.algo.detect.randomforest --test-_get_models

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.detect.randomforest import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> species = wbia.const.TEST_SPECIES.ZEB_PLAIN
        >>> modeldir = 'default'
        >>> cfg_override = True
        >>> verbose = False
        >>> fpath_list = _get_models(ibs, species, modeldir, cfg_override, verbose)
        >>> result = ('fpath_list = %s' % (str(fpath_list),))
        >>> print(result)
    """
    # with ut.embed_on_exception_context:
    if cfg_override and len(ibs.cfg.detect_cfg.trees_path) > 0:
        trees_path = ibs.cfg.detect_cfg.trees_path
    else:
        # Ensure all models downloaded and accounted for
        assert (
            species is not None
        ), '[_get_models] Cannot detect without specifying a species'
        grabmodels.ensure_models(modeldir=modeldir, verbose=verbose)
        trees_path = grabmodels.get_species_trees_paths(species, modeldir=modeldir)
    # Load tree paths
    if ut.checkpath(trees_path, verbose=verbose):
        fpath_list = ut.ls(trees_path, '*.txt')
        # direct = Directory(trees_path, include_extensions=['txt'])
        # files = direct.files()
    else:
        # If the models do not exist, return None
        fpath_list = None
    if fpath_list is None or len(fpath_list) == 0:
        msg = (
            ut.codeblock(
                """
            [_get_models] Error loading trees, either directory or fpath_list not found
              * trees_path = %r
              * fpath_list = %r
              * species = %r
              * model_dir = %r
              * cfg_override = %r
            """
            )
            % (trees_path, fpath_list, species, modeldir, cfg_override)
        )
        raise AssertionError(msg)
    return fpath_list
