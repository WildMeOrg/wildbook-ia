# -*- coding: utf-8 -*-
"""Developer convenience functions for ibs (detections).

TODO: need to split up into sub modules:
    consistency_checks
    feasibility_fixes
    move the export stuff to dbio

    then there are also convineience functions that need to be ordered at least
    within this file
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, range
from os.path import exists, expanduser, join, abspath
import numpy as np
import utool as ut
import cv2
from wbia.control import controller_inject
from wbia.other.detectfuncs import (
    general_parse_gt,
    general_get_imageset_gids,
    localizer_parse_pred,
    general_overlap,
)
from wbia.other.detectcore import (
    nms,
    classifier_visualize_training_localizations,
    _bootstrap_mine,
)


# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[other.detectgrave]')


CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)


@register_ibs_method
def bootstrap_pca_train(
    ibs, dims=64, pca_limit=500000, ann_batch=50, output_path=None, **kwargs
):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import IncrementalPCA
    from annoy import AnnoyIndex
    import numpy as np
    import random

    def _get_data(depc, gid_list, limit=None, shuffle=False):
        gid_list_ = gid_list[:]
        if shuffle:
            random.shuffle(gid_list_)
        config = {
            'algo': '_COMBINED',
            'features': True,
            'feature2_algo': 'resnet',
        }
        total = 0
        features_list = []
        index_list = []
        gid_iter = ut.ProgIter(gid_list_, lbl='collect feature vectors', bs=True)
        for gid in gid_iter:
            if limit is not None and total >= limit:
                break
            feature_list = depc.get_property(
                'localizations_features', gid, 'vector', config=config
            )
            total += len(feature_list)
            index_list += [(gid, offset,) for offset in range(len(feature_list))]
            features_list.append(feature_list)
        print('\nUsed %d images to mine %d features' % (len(features_list), total,))
        data_list = np.vstack(features_list)
        if len(data_list) > limit:
            data_list = data_list[:limit]
            index_list = index_list[:limit]
        assert len(data_list) == len(index_list)
        features_list = None
        return total, data_list, index_list

    # gid_list = ibs.get_valid_gids()
    gid_list = general_get_imageset_gids(ibs, 'TRAIN_SET', **kwargs)
    # gid_list = gid_list[:200]

    # Get data
    depc = ibs.depc_image
    total, data_list, index_list = _get_data(depc, gid_list, pca_limit, True)
    print(data_list.shape)

    # Normalize data
    print('Fit Scaler')
    scaler = StandardScaler()
    scaler.fit(data_list)
    data_list = scaler.transform(data_list)

    # Fit PCA
    print('Fit PCA')
    pca_model = IncrementalPCA(n_components=dims)
    pca_model.fit(data_list)

    pca_quality = pca_model.explained_variance_ratio_.sum() * 100.0
    print('PCA Variance Quality: %0.04f %%' % (pca_quality,))

    # Fit ANN for PCA's vectors
    index = 0
    ann_model = AnnoyIndex(dims)  # Length of item vector that will be indexed
    ann_rounds = int(np.ceil(float(len(gid_list)) / ann_batch))
    manifest_dict = {}
    for ann_round in range(ann_rounds):
        start_index = ann_round * ann_batch
        stop_index = (ann_round + 1) * ann_batch
        assert start_index < len(gid_list)
        stop_index = min(stop_index, len(gid_list))
        print('Slicing index range: [%r, %r)' % (start_index, stop_index,))

        # Slice gids and get feature data
        gid_list_ = gid_list[start_index:stop_index]
        total, data_list, index_list = _get_data(depc, gid_list_)

        # Scaler
        data_list = scaler.transform(data_list)

        # Transform data to smaller vectors
        data_list_ = pca_model.transform(data_list)

        zipped = zip(index_list, data_list_)
        data_iter = ut.ProgIter(zipped, lbl='add vectors to ANN model', bs=True)
        for (gid, offset), feature in data_iter:
            ann_model.add_item(index, feature)
            manifest_dict[index] = (
                gid,
                offset,
            )
            index += 1

    # Build forest
    trees = index // 100000
    print('Build ANN model using %d feature vectors and %d trees' % (index, trees,))
    ann_model.build(trees)

    # Save forest
    if output_path is None:
        output_path = abspath(expanduser(join('~', 'code', 'wbia', 'models')))

    scaler_filename = 'forest.pca'
    scaler_filepath = join(output_path, scaler_filename)
    print('Saving scaler model to: %r' % (scaler_filepath,))
    model_tup = (
        pca_model,
        scaler,
        manifest_dict,
    )
    ut.save_cPkl(scaler_filepath, model_tup)

    forest_filename = 'forest.ann'
    forest_filepath = join(output_path, forest_filename)
    print('Saving ANN model to: %r' % (forest_filepath,))
    ann_model.save(forest_filepath)

    # ibs.bootstrap_pca_test(model_path=output_path)
    return output_path


@register_ibs_method
def bootstrap_pca_test(
    ibs,
    dims=64,
    pca_limit=500000,
    ann_batch=50,
    model_path=None,
    output_path=None,
    neighbors=1000,
    nms_thresh=0.5,
    min_confidence=0.3,
    **kwargs,
):
    from annoy import AnnoyIndex
    import random

    if output_path is None:
        output_path = abspath(expanduser(join('~', 'Desktop', 'output-ann')))
    ut.ensuredir(output_path)

    # gid_list = ibs.get_valid_gids()
    gid_list = general_get_imageset_gids(ibs, 'TRAIN_SET', **kwargs)
    random.shuffle(gid_list)
    # gid_list = gid_list[:100]

    # Load forest
    if model_path is None:
        model_path = abspath(expanduser(join('~', 'code', 'wbia', 'models')))

    scaler_filename = 'forest.pca'
    scaler_filepath = join(model_path, scaler_filename)
    print('Loading scaler model from: %r' % (scaler_filepath,))
    model_tup = ut.load_cPkl(scaler_filepath)
    pca_model, scaler, manifest_dict = model_tup

    forest_filename = 'forest.ann'
    forest_filepath = join(model_path, forest_filename)
    print('Loading ANN model from: %r' % (forest_filepath,))
    ann_model = AnnoyIndex(dims)
    ann_model.load(forest_filepath)

    config = {
        'algo': '_COMBINED',
        'features': True,
        'feature2_algo': 'resnet',
        'classify': True,
        'classifier_algo': 'svm',
        'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.image.zebra.pkl',
    }

    print('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=gid_list, **config)

    print('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=gid_list, **config)

    for image_uuid in gt_dict:
        # Get the gt and prediction list
        gt_list = gt_dict[image_uuid]
        pred_list = pred_dict[image_uuid]

        # Calculate overlap
        overlap = general_overlap(gt_list, pred_list)
        num_gt, num_pred = overlap.shape

        max_overlap = np.max(overlap, axis=0)
        index_list = np.argsort(max_overlap)

        example_limit = 1
        worst_idx_list = index_list[:example_limit]
        best_idx_list = index_list[-1 * example_limit :]

        print('Worst ovelap: %r' % (overlap[:, worst_idx_list],))
        print('Best ovelap:  %r' % (overlap[:, best_idx_list],))

        for idx_list in [best_idx_list, worst_idx_list]:
            example_list = ut.take(pred_list, idx_list)

            interpolation = cv2.INTER_LANCZOS4
            warpkw = dict(interpolation=interpolation)

            for example, offset in zip(example_list, idx_list):
                gid = example['gid']
                feature_list = np.array([example['feature']])
                data_list = scaler.transform(feature_list)
                data_list_ = pca_model.transform(data_list)[0]

                neighbor_index_list = ann_model.get_nns_by_vector(data_list_, neighbors)
                neighbor_manifest_list = list(
                    set(
                        [
                            manifest_dict[neighbor_index]
                            for neighbor_index in neighbor_index_list
                        ]
                    )
                )
                neighbor_gid_list_ = ut.take_column(neighbor_manifest_list, 0)
                neighbor_gid_list_ = [gid] + neighbor_gid_list_
                neighbor_uuid_list_ = ibs.get_image_uuids(neighbor_gid_list_)
                neighbor_offset_list_ = ut.take_column(neighbor_manifest_list, 1)
                neighbor_offset_list_ = [offset] + neighbor_offset_list_

                neighbor_gid_set_ = list(set(neighbor_gid_list_))
                neighbor_image_list = ibs.get_images(neighbor_gid_set_)
                neighbor_image_dict = {
                    gid: image
                    for gid, image in zip(neighbor_gid_set_, neighbor_image_list)
                }

                neighbor_pred_dict = localizer_parse_pred(
                    ibs, test_gid_list=neighbor_gid_set_, **config
                )

                neighbor_dict = {}
                zipped = zip(
                    neighbor_gid_list_, neighbor_uuid_list_, neighbor_offset_list_
                )
                for neighbor_gid, neighbor_uuid, neighbor_offset in zipped:
                    if neighbor_gid not in neighbor_dict:
                        neighbor_dict[neighbor_gid] = []
                    neighbor_pred = neighbor_pred_dict[neighbor_uuid][neighbor_offset]
                    neighbor_dict[neighbor_gid].append(neighbor_pred)

                # Perform NMS
                chip_list = []
                query_image = ibs.get_images(gid)
                xbr = example['xbr']
                ybr = example['ybr']
                xtl = example['xtl']
                ytl = example['ytl']

                height, width = query_image.shape[:2]
                xbr = int(xbr * width)
                ybr = int(ybr * height)
                xtl = int(xtl * width)
                ytl = int(ytl * height)
                # Get chips
                try:
                    chip = query_image[ytl:ybr, xtl:xbr, :]
                    chip = cv2.resize(chip, (192, 192), **warpkw)
                    chip_list.append(chip)
                except Exception:
                    pass
                chip_list.append(np.zeros((192, 10, 3)))

                for neighbor_gid in neighbor_dict:
                    neighbor_list = neighbor_dict[neighbor_gid]
                    # Compile coordinate list of (xtl, ytl, xbr, ybr) instead of (xtl, ytl, w, h)
                    coord_list = []
                    confs_list = []
                    for neighbor in neighbor_list:
                        xbr = neighbor['xbr']
                        ybr = neighbor['ybr']
                        xtl = neighbor['xtl']
                        ytl = neighbor['ytl']
                        conf = neighbor['confidence']
                        coord_list.append([xtl, ytl, xbr, ybr])
                        confs_list.append(conf)
                    coord_list = np.vstack(coord_list)
                    confs_list = np.array(confs_list)
                    # Perform NMS
                    keep_indices_list = nms(coord_list, confs_list, nms_thresh)
                    keep_indices_set = set(keep_indices_list)
                    neighbor_list_ = [
                        neighbor
                        for index, neighbor in enumerate(neighbor_list)
                        if index in keep_indices_set
                    ]

                    neighbor_image = neighbor_image_dict[neighbor_gid]
                    for neightbor_ in neighbor_list_:
                        xbr = neightbor_['xbr']
                        ybr = neightbor_['ybr']
                        xtl = neightbor_['xtl']
                        ytl = neightbor_['ytl']
                        conf = neighbor['confidence']

                        height, width = neighbor_image.shape[:2]
                        xbr = int(xbr * width)
                        ybr = int(ybr * height)
                        xtl = int(xtl * width)
                        ytl = int(ytl * height)
                        # Get chips
                        try:
                            chip = neighbor_image[ytl:ybr, xtl:xbr, :]
                            chip = cv2.resize(chip, (192, 192), **warpkw)
                            color = (0, 255, 0) if conf >= min_confidence else (0, 0, 255)
                            cv2.rectangle(chip, (0, 0), (192, 192), color, 10)
                            chip_list.append(chip)
                        except Exception:
                            pass

                min_chips = 16
                if len(chip_list) < min_chips:
                    continue

                chip_list = chip_list[:min_chips]
                canvas = np.hstack(chip_list)
                output_filename = 'neighbors_%d_%d.png' % (gid, offset,)
                output_filepath = join(output_path, output_filename)
                cv2.imwrite(output_filepath, canvas)


@register_ibs_method
def bootstrap(
    ibs,
    species_list=['zebra'],
    N=10,
    rounds=20,
    scheme=2,
    ensemble=9,
    output_path=None,
    precompute=True,
    precompute_test=True,
    recompute=False,
    visualize=True,
    C=1.0,
    kernel='rbf',
    **kwargs,
):
    from sklearn import svm, preprocessing

    # Establish variables

    kernel = str(kernel.lower())
    species_list = [species.lower() for species in species_list]
    species_list_str = '.'.join(species_list)
    assert scheme in [1, 2], 'Invalid scheme'
    if output_path is None:
        # species_list_str = '+'.join(species_list)
        # args = (N, rounds, scheme, species_list_str, )
        # output_path_ = 'models-bootstrap-%s-%s-%s-%s' % args
        output_path_ = 'models-bootstrap'
        output_path = abspath(expanduser(join('~', 'code', 'wbia', output_path_)))
    print('Using output_path = %r' % (output_path,))
    if recompute:
        ut.delete(output_path)
    ut.ensuredir(output_path)

    # Get the test images for later
    depc = ibs.depc_image
    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)

    wic_model_filepath = ibs.classifier_train_image_svm(
        species_list, output_path=output_path, dryrun=True
    )
    is_wic_model_trained = exists(wic_model_filepath)
    ######################################################################################
    # Step 1: train whole-image classifier
    #         this will compute and cache any ResNet features that
    #         haven't been computed
    if not is_wic_model_trained:
        wic_model_filepath = ibs.classifier_train_image_svm(
            species_list, output_path=output_path
        )

    # Load model pickle
    model_tup = ut.load_cPkl(wic_model_filepath)
    model, scaler = model_tup

    ######################################################################################
    # Step 2: sort all test images based on whole image classifier
    #         establish a review ordering based on classification probability

    # Get scores
    vals = get_classifier_svm_data_labels(ibs, 'TRAIN_SET', species_list)
    train_gid_set, data_list, label_list = vals
    # Normalize data
    data_list = scaler.transform(data_list)
    # score_list_ = model.decision_function(data_list)  # NOQA
    score_list_ = model.predict_proba(data_list)
    score_list_ = score_list_[:, 1]

    # Sort gids by scores (initial ranking)
    comb_list = sorted(list(zip(score_list_, train_gid_set)), reverse=True)
    sorted_gid_list = [comb[1] for comb in comb_list]

    config = {
        'algo': '_COMBINED',
        'species_set': set(species_list),
        'features': True,
        'feature2_algo': 'resnet',
        'classify': True,
        'classifier_algo': 'svm',
        'classifier_weight_filepath': wic_model_filepath,
        'nms': True,
        'nms_thresh': 0.50,
        # 'thresh'       : True,
        # 'index_thresh' : 0.25,
    }
    config_list = [config.copy()]

    ######################################################################################
    # Step 2.5: pre-compute localizations and ResNet features (without loading to memory)
    #
    if precompute:
        needed = N * rounds
        needed = min(needed, len(sorted_gid_list))
        sorted_gid_list_ = sorted_gid_list[:needed]
        depc.get_rowids('localizations_features', sorted_gid_list_, config=config)

    # Precompute test features
    if precompute and precompute_test:
        # depc.get_rowids('localizations_features', test_gid_list, config=config)
        if not is_wic_model_trained:
            depc.delete_property('localizations_classifier', test_gid_list, config=config)
        depc.get_rowids('localizations_classifier', test_gid_list, config=config)

    # return

    ######################################################################################
    # Step 3: for each bootstrapping round, ask user for input
    # The initial classifier is the whole image classifier

    reviewed_gid_dict = {}
    for current_round in range(rounds):
        print('------------------------------------------------------')
        print('Current Round %r' % (current_round,))

        ##################################################################################
        # Step 4: gather the (unreviewed) images to review for this round
        round_gid_list = []
        temp_index = 0
        while len(round_gid_list) < N and temp_index < len(sorted_gid_list):
            temp_gid = sorted_gid_list[temp_index]
            if temp_gid not in reviewed_gid_dict:
                round_gid_list.append(temp_gid)
            temp_index += 1

        args = (
            len(round_gid_list),
            round_gid_list,
        )
        print('Found %d unreviewed gids: %r' % args)

        ##################################################################################
        # Step 5: add any images reviewed from a previous round

        reviewed_gid_list = reviewed_gid_dict.keys()
        args = (
            len(reviewed_gid_list),
            reviewed_gid_list,
        )
        print('Adding %d previously reviewed gids: %r' % args)

        # All gids that have been reviewed
        round_gid_list = reviewed_gid_list + round_gid_list

        # Get model ensemble path
        limit = len(round_gid_list)
        args = (
            species_list_str,
            limit,
            kernel,
            C,
        )
        output_filename = 'classifier.svm.localization.%s.%d.%s.%s' % args
        svm_model_path = join(output_path, output_filename)
        is_svm_model_trained = exists(svm_model_path)

        ut.ensuredir(svm_model_path)

        ##################################################################################
        # Step 6: gather gt (simulate user interaction)

        print('\tGather Ground-Truth')
        gt_dict = general_parse_gt(ibs, test_gid_list=round_gid_list, **config)

        ##################################################################################
        # Step 7: gather predictions from all algorithms combined

        if not is_svm_model_trained:
            print('\tDelete Old Classifications')
            depc.delete_property(
                'localizations_classifier', round_gid_list, config=config
            )

        print('\tGather Predictions')
        pred_dict = localizer_parse_pred(ibs, test_gid_list=round_gid_list, **config)

        ##################################################################################
        # Step 8: train SVM ensemble using fresh mined data for each ensemble

        # Train models, one-by-one
        for current_ensemble in range(1, ensemble + 1):
            # Mine for a new set of (static) positives and (random) negatives
            values = _bootstrap_mine(
                ibs, gt_dict, pred_dict, scheme, reviewed_gid_dict, **kwargs
            )
            mined_gid_list, mined_gt_list, mined_pos_list, mined_neg_list = values

            if visualize:
                output_visualize_path = join(svm_model_path, 'visualize')
                ut.ensuredir(output_visualize_path)
                output_visualize_path = join(
                    output_visualize_path, '%s' % (current_ensemble,)
                )
                ut.ensuredir(output_visualize_path)
                classifier_visualize_training_localizations(
                    ibs, None, output_path=output_visualize_path, values=values
                )

            # Get the confidences of the selected positives and negatives
            pos_conf_list = []
            neg_conf_list = []
            for pos in mined_pos_list:
                pos_conf_list.append(pos['confidence'])
            for neg in mined_neg_list:
                neg_conf_list.append(neg['confidence'])

            pos_conf_list = np.array(pos_conf_list)
            args = (
                np.min(pos_conf_list),
                np.mean(pos_conf_list),
                np.std(pos_conf_list),
                np.max(pos_conf_list),
            )
            print(
                'Positive Confidences: %0.02f min, %0.02f avg, %0.02f std, %0.02f max'
                % args
            )
            neg_conf_list = np.array(neg_conf_list)
            args = (
                np.min(neg_conf_list),
                np.mean(neg_conf_list),
                np.std(neg_conf_list),
                np.max(neg_conf_list),
            )
            print(
                'Negative Confidences: %0.02f min, %0.02f avg, %0.02f std, %0.02f max'
                % args
            )

            # Train new models
            if not is_svm_model_trained:
                # Compile feature data and label list
                data_list = []
                label_list = []
                for pos in mined_pos_list:
                    data_list.append(pos['feature'])
                    label_list.append(1)
                for neg in mined_neg_list:
                    data_list.append(neg['feature'])
                    label_list.append(0)

                data_list = np.array(data_list)
                label_list = np.array(label_list)

                print('Train Ensemble SVM (%d)' % (current_ensemble,))
                # Train scaler
                scaler = preprocessing.StandardScaler().fit(data_list)
                data_list = scaler.transform(data_list)
                # Train model
                model = svm.SVC(C=C, kernel=kernel, probability=True)
                model.fit(data_list, label_list)

                # Save model pickle
                args = (
                    species_list_str,
                    limit,
                    current_ensemble,
                )
                svm_model_filename = 'classifier.svm.localization.%s.%d.%d.pkl' % args
                svm_model_filepath = join(svm_model_path, svm_model_filename)
                model_tup = (
                    model,
                    scaler,
                )
                ut.save_cPkl(svm_model_filepath, model_tup)

        ##################################################################################
        # Step 8: update the bootstrapping algorithm to use the new ensemble during
        #         the next round
        config['classifier_weight_filepath'] = svm_model_path
        config_list.append(config.copy())

        ##################################################################################
        # Step 9: get the test images and classify (cache) their proposals using
        #         the new model ensemble
        if precompute and precompute_test:
            if not is_svm_model_trained:
                depc.delete_property(
                    'localizations_classifier', test_gid_list, config=config
                )
            depc.get_rowids('localizations_classifier', test_gid_list, config=config)

    # Return the list of used configs
    return config_list


@register_ibs_method
def bootstrap2(
    ibs,
    species_list=['zebra'],
    alpha=10,
    gamma=16,
    epsilon=0.3,
    rounds=20,
    ensemble=3,
    dims=64,
    pca_limit=1000000,
    nms_thresh_pos=0.5,
    nms_thresh_neg=0.90,
    C=1.0,
    kernel='rbf',
    theta=1.0,
    output_path=None,
    precompute=True,
    precompute_test=True,
    recompute=False,
    recompute_classifications=True,
    overlap_thresh_cat_1=0.75,
    overlap_thresh_cat_2=0.25,
    overlap_thresh_cat_3=0.0,
    **kwargs,
):
    from sklearn import svm, preprocessing
    from annoy import AnnoyIndex

    # Establish variables
    kernel = str(kernel.lower())
    species_list = [species.lower() for species in species_list]
    species_list_str = '.'.join(species_list)

    if output_path is None:
        output_path_ = 'models-bootstrap'
        output_path = abspath(expanduser(join('~', 'code', 'wbia', output_path_)))
    print('Using output_path = %r' % (output_path,))

    if recompute:
        ut.delete(output_path)
    ut.ensuredir(output_path)

    scaler_filename = 'forest.pca'
    scaler_filepath = join(output_path, scaler_filename)
    forest_filename = 'forest.ann'
    forest_filepath = join(output_path, forest_filename)

    is_ann_model_trained = exists(scaler_filepath) and exists(forest_filepath)

    # Train forest
    if not is_ann_model_trained:
        ibs.bootstrap_pca_train(dims=dims, pca_limit=pca_limit, output_path=output_path)

    print('Loading scaler model from: %r' % (scaler_filepath,))
    model_tup = ut.load_cPkl(scaler_filepath)
    pca_model, scaler, manifest_dict = model_tup

    print('Loading ANN model from: %r' % (forest_filepath,))
    ann_model = AnnoyIndex(dims)
    ann_model.load(forest_filepath)

    # Get the test images for later
    depc = ibs.depc_image
    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', species_list, **kwargs)

    wic_model_filepath = ibs.classifier_train_image_svm(
        species_list, output_path=output_path, dryrun=True
    )
    is_wic_model_trained = exists(wic_model_filepath)
    ######################################################################################
    # Step 1: train whole-image classifier
    #         this will compute and cache any ResNet features that
    #         haven't been computed
    if not is_wic_model_trained:
        wic_model_filepath = ibs.classifier_train_image_svm(
            species_list, output_path=output_path
        )

    # Load model pickle
    model_tup = ut.load_cPkl(wic_model_filepath)
    model, scaler = model_tup

    ######################################################################################
    # Step 2: sort all test images based on whole image classifier
    #         establish a review ordering based on classification probability

    # Get scores
    vals = get_classifier_svm_data_labels(ibs, 'TRAIN_SET', species_list)
    train_gid_set, data_list, label_list = vals
    # Normalize data
    data_list = scaler.transform(data_list)
    # score_list_ = model.decision_function(data_list)  # NOQA
    score_list_ = model.predict_proba(data_list)
    score_list_ = score_list_[:, 1]

    # Sort gids by scores (initial ranking)
    comb_list = sorted(list(zip(score_list_, train_gid_set)), reverse=True)
    sorted_gid_list = [comb[1] for comb in comb_list]

    config = {
        'algo': '_COMBINED',
        'species_set': set(species_list),
        # 'features'       : True,
        'features_lazy': True,
        'feature2_algo': 'resnet',
        'classify': True,
        'classifier_algo': 'svm',
        'classifier_weight_filepath': wic_model_filepath,
        # 'nms'          : True,
        # 'nms_thresh'   : nms_thresh,
        # 'thresh'       : True,
        # 'index_thresh' : 0.25,
    }
    config_list = [config.copy()]

    ######################################################################################
    # Step 2.5: pre-compute localizations and ResNet features (without loading to memory)
    #
    if precompute:
        depc.get_rowids('localizations_features', sorted_gid_list, config=config)

    # Precompute test features
    if precompute and precompute_test:
        # depc.get_rowids('localizations_features', test_gid_list, config=config)
        if not is_wic_model_trained:
            depc.delete_property('localizations_classifier', test_gid_list, config=config)
        depc.get_rowids('localizations_classifier', test_gid_list, config=config)

    ######################################################################################
    # Step 3: for each bootstrapping round, ask user for input
    # The initial classifier is the whole image classifier

    sorted_gid_list_ = sorted_gid_list[:]
    reviewed_gid_list = []
    for current_round in range(rounds):
        print('------------------------------------------------------')
        print('Current Round %r' % (current_round,))

        ##################################################################################
        # Step 4: gather the (unreviewed) images to review for this round
        round_gid_list = []
        temp_index = 0
        while len(round_gid_list) < alpha and temp_index < len(sorted_gid_list_):
            temp_gid = sorted_gid_list_[temp_index]
            if temp_gid not in reviewed_gid_list:
                round_gid_list.append(temp_gid)
            temp_index += 1

        args = (
            len(round_gid_list),
            round_gid_list,
        )
        print('Found %d unreviewed gids: %r' % args)

        ##################################################################################
        # Step 5: add any images reviewed from a previous round

        args = (
            len(reviewed_gid_list),
            reviewed_gid_list,
        )
        print('Adding %d previously reviewed gids: %r' % args)

        # All gids that have been reviewed
        round_gid_list = reviewed_gid_list + round_gid_list
        reviewed_gid_list = round_gid_list

        # Get model ensemble path
        limit = len(round_gid_list)
        args = (
            species_list_str,
            limit,
            kernel,
            C,
        )
        output_filename = 'classifier.svm.localization.%s.%d.%s.%s' % args
        svm_model_path = join(output_path, output_filename)
        is_svm_model_trained = exists(svm_model_path)

        round_neighbor_gid_hist = {}
        if not is_svm_model_trained:
            ##################################################################################
            # Step 6: gather gt (simulate user interaction)

            print('\tGather Ground-Truth')
            gt_dict = general_parse_gt(ibs, test_gid_list=round_gid_list, **config)

            ##################################################################################
            # Step 7: gather predictions from all algorithms combined
            if recompute_classifications:
                print('\tDelete Old Classifications')
                depc.delete_property(
                    'localizations_classifier', round_gid_list, config=config
                )

            print('\tGather Predictions')
            pred_dict = localizer_parse_pred(ibs, test_gid_list=round_gid_list, **config)

            category_dict = {}
            for image_index, image_uuid in enumerate(gt_dict.keys()):
                image_gid = ibs.get_image_gids_from_uuid(image_uuid)
                args = (
                    image_gid,
                    image_uuid,
                    image_index + 1,
                    len(round_gid_list),
                )
                print('Processing neighbors for image %r, %r (%d / %d)' % args)

                # Get the gt and prediction list
                gt_list = gt_dict[image_uuid]
                pred_list = pred_dict[image_uuid]

                # Calculate overlap
                overlap = general_overlap(gt_list, pred_list)
                num_gt, num_pred = overlap.shape
                max_overlap = np.max(overlap, axis=0)

                # Find overlap category bins
                cat1_idx_list = max_overlap >= overlap_thresh_cat_1
                # cat2_idx_list = np.logical_and(overlap_thresh_cat_1 > max_overlap, max_overlap >= overlap_thresh_cat_2)
                cat3_idx_list = np.logical_and(
                    overlap_thresh_cat_2 > max_overlap,
                    max_overlap > overlap_thresh_cat_3,
                )
                cat4_idx_list = overlap_thresh_cat_3 >= max_overlap

                # Mine for prediction neighbors in category 1
                cat_config_list = [
                    ('cat1', cat1_idx_list),
                    # ('cat2', cat2_idx_list),
                    ('cat3', cat3_idx_list),
                    ('cat4', cat4_idx_list),
                ]
                for cat_tag, cat_idx_list in cat_config_list:
                    if cat_tag not in category_dict:
                        category_dict[cat_tag] = {}

                    # Take the predictions for this category
                    cat_pred_list = ut.compress(pred_list, list(cat_idx_list))
                    args = (
                        cat_tag,
                        len(cat_pred_list),
                    )
                    print('\t Working on category %r with %d predictions' % args)

                    # Add raw predictions
                    if image_gid not in category_dict[cat_tag]:
                        category_dict[cat_tag][image_gid] = []
                    category_dict[cat_tag][image_gid] += cat_pred_list

                    if cat_tag == 'cat1':
                        # Go over predictions and find neighbors, sorting into either cat1 or cat3
                        neighbor_manifest_list = []
                        cat_pred_iter = ut.ProgIter(
                            cat_pred_list, lbl='find neighbors', bs=True
                        )
                        for cat_pred in cat_pred_iter:
                            feature = cat_pred.get('feature', None)
                            if feature is None:
                                feature_func = cat_pred.get('feature_lazy', None)
                                # print('Lazy loading neighbor feature with %r' % (feature_func, ))
                                assert feature_func is not None
                                feature = feature_func()
                                # cat_pred['feature'] = feature
                            feature_list = np.array([feature])
                            data_list = scaler.transform(feature_list)
                            data_list_ = pca_model.transform(data_list)[0]

                            neighbor_index_list = ann_model.get_nns_by_vector(
                                data_list_, gamma
                            )
                            neighbor_manifest_list += [
                                manifest_dict[neighbor_index]
                                for neighbor_index in neighbor_index_list
                            ]

                        neighbor_manifest_list = list(set(neighbor_manifest_list))
                        neighbor_gid_list_ = ut.take_column(neighbor_manifest_list, 0)
                        neighbor_gid_set_ = set(neighbor_gid_list_)
                        neighbor_uuid_list_ = ibs.get_image_uuids(neighbor_gid_list_)
                        neighbor_idx_list_ = ut.take_column(neighbor_manifest_list, 1)

                        # Keep track of the round's results
                        for neighbor_gid_ in neighbor_gid_list_:
                            if neighbor_gid_ not in round_neighbor_gid_hist:
                                round_neighbor_gid_hist[neighbor_gid_] = 0
                            round_neighbor_gid_hist[neighbor_gid_] += 1

                        args = (
                            len(neighbor_gid_set_),
                            len(neighbor_manifest_list),
                        )
                        print('\t\tGetting %d images for %d neighbors' % args)
                        neighbor_pred_dict = localizer_parse_pred(
                            ibs, test_gid_list=list(neighbor_gid_set_), **config
                        )

                        zipped = zip(
                            neighbor_gid_list_, neighbor_uuid_list_, neighbor_idx_list_
                        )
                        for neighbor_gid, neighbor_uuid, neighbor_idx in zipped:
                            neighbor_pred = neighbor_pred_dict[neighbor_uuid][
                                neighbor_idx
                            ]
                            cat_tag_ = (
                                'cat1'
                                if neighbor_pred['confidence'] >= epsilon
                                else 'cat3'
                            )
                            if cat_tag_ not in category_dict:
                                category_dict[cat_tag_] = {}
                            if neighbor_gid not in category_dict[cat_tag_]:
                                category_dict[cat_tag_][neighbor_gid] = []
                            category_dict[cat_tag_][neighbor_gid].append(neighbor_pred)

            # Perform NMS on each category
            for cat_tag in sorted(category_dict.keys()):
                cat_pred_dict = category_dict[cat_tag]
                cat_pred_list = []
                cat_pred_total = 0
                for cat_gid in cat_pred_dict:
                    pred_list = cat_pred_dict[cat_gid]
                    cat_pred_total += len(pred_list)
                    # Compile coordinate list of (xtl, ytl, xbr, ybr) instead of (xtl, ytl, w, h)
                    coord_list = []
                    confs_list = []
                    for pred in pred_list:
                        xbr = pred['xbr']
                        ybr = pred['ybr']
                        xtl = pred['xtl']
                        ytl = pred['ytl']
                        conf = pred['confidence']
                        coord_list.append([xtl, ytl, xbr, ybr])
                        confs_list.append(conf)
                    coord_list = np.vstack(coord_list)
                    confs_list = np.array(confs_list)
                    # Perform NMS
                    nms_thresh = (
                        nms_thresh_pos if cat_tag in ['cat1', 'cat3'] else nms_thresh_neg
                    )
                    keep_indices_list = nms(coord_list, confs_list, nms_thresh)
                    keep_indices_set = set(keep_indices_list)
                    pred_list_ = [
                        pred
                        for index, pred in enumerate(pred_list)
                        if index in keep_indices_set
                    ]
                    cat_pred_list += pred_list_
                print(
                    'NMS Proposals (start) for category %r: %d'
                    % (cat_tag, cat_pred_total,)
                )
                # Print stats
                conf_list = []
                for cat_pred in cat_pred_list:
                    conf_list.append(cat_pred['confidence'])
                conf_list = np.array(conf_list)
                args = (
                    cat_tag,
                    np.min(conf_list),
                    np.mean(conf_list),
                    np.std(conf_list),
                    np.max(conf_list),
                )
                print(
                    'Category %r Confidences: %0.02f min, %0.02f avg, %0.02f std, %0.02f max'
                    % args
                )
                # Overwrite GID dictionary with a list of predictions
                category_dict[cat_tag] = cat_pred_list
                cat_total = len(cat_pred_list)
                print('NMS Proposals (end) for category %r: %d' % (cat_tag, cat_total,))

            ##################################################################################
            # Step 8: train SVM ensemble using fresh mined data for each ensemble

            ut.ensuredir(svm_model_path)

            # Train models, one-by-one
            for current_ensemble in range(1, ensemble + 1):
                # Compile feature data and label list
                mined_pos_list = category_dict['cat1']
                mined_hard_list = category_dict['cat3']
                mined_neg_list = category_dict['cat4']

                num_pos = len(mined_pos_list)
                num_target = int(num_pos / theta)
                print('Mining %d target negatives' % (num_target,))

                if len(mined_hard_list) > num_target:
                    print('Sampling Hard')
                    np.random.shuffle(mined_hard_list)
                    mined_hard_list = mined_hard_list[:num_target]

                if len(mined_neg_list) > num_target:
                    print('Sampling Negatives')
                    np.random.shuffle(mined_neg_list)
                    mined_neg_list = mined_neg_list[:num_target]

                num_pos = len(mined_pos_list)
                num_hard = len(mined_hard_list)
                num_neg = len(mined_neg_list)
                num_total = num_pos + num_hard + num_neg
                args = (
                    num_pos,
                    num_hard + num_neg,
                    num_hard,
                    num_neg,
                    num_pos / num_total,
                )
                print(
                    'Training with %d positives and %d (%d + %d) negatives (%0.02f split)'
                    % args
                )

                temp_list = [
                    ('pos', 1, mined_pos_list),
                    ('hard', 0, mined_hard_list),
                    ('neg', 0, mined_neg_list),
                ]

                # data_list = []
                index = 0
                data_list = None
                label_list = []
                for label_tag, label, mined_data_list in temp_list:
                    lbl = 'gathering training features for %s' % (label_tag,)
                    mined_data_iter = ut.ProgIter(mined_data_list, lbl=lbl, bs=True)
                    for data in mined_data_iter:
                        feature = data.get('feature', None)
                        if feature is None:
                            feature_func = data.get('feature_lazy', None)
                            # print('Lazy loading ensemble feature with %r' % (feature_func, ))
                            assert feature_func is not None
                            feature = feature_func()
                            # data['feature'] = feature
                        if data_list is None:
                            num_dims = len(feature)
                            data_shape = (
                                num_total,
                                num_dims,
                            )
                            data_list = np.zeros(data_shape, dtype=feature.dtype)
                        # Add feature and label to list
                        # data_list.append(feature)
                        data_list[index] = feature
                        index += 1
                        label_list.append(label)

                # data_list = np.array(data_list)
                label_list = np.array(label_list)

                print('Train Ensemble SVM (%d)' % (current_ensemble,))
                # Train scaler
                scaler = preprocessing.StandardScaler().fit(data_list)
                data_list = scaler.transform(data_list)
                # Train model
                model = svm.SVC(C=C, kernel=kernel, probability=True)
                model.fit(data_list, label_list)

                # Save model pickle
                args = (current_ensemble,)
                svm_model_filename = 'classifier.svm.localization.%d.pkl' % args
                svm_model_filepath = join(svm_model_path, svm_model_filename)
                model_tup = (
                    model,
                    scaler,
                )
                ut.save_cPkl(svm_model_filepath, model_tup)

        ##################################################################################
        # Step 8: update the sorted_gid_list based on what neighbors were samples
        if len(round_neighbor_gid_hist) >= alpha:
            vals_list = [
                (round_neighbor_gid_hist[neighbor_gid_], neighbor_gid_,)
                for neighbor_gid_ in round_neighbor_gid_hist
            ]
            vals_list = sorted(vals_list, reverse=True)
            vals_list = vals_list[:alpha]
            print('Reference Histogram: %r' % (vals_list,))
            top_referenced_neighbor_gid_list = [_[1] for _ in vals_list]
            round_neighbor_gid_set = set(top_referenced_neighbor_gid_list)

            # Partition set
            lower_sorted_gid_list = [
                sorted_gid
                for sorted_gid in sorted_gid_list
                if sorted_gid in round_neighbor_gid_set
            ]
            higher_sorted_gid_list = [
                sorted_gid
                for sorted_gid in sorted_gid_list
                if sorted_gid not in lower_sorted_gid_list
            ]
            sorted_gid_list_ = higher_sorted_gid_list + lower_sorted_gid_list

            assert len(sorted_gid_list_) == len(higher_sorted_gid_list) + len(
                lower_sorted_gid_list
            )
            assert len(sorted_gid_list_) == len(sorted_gid_list)
            args = (
                len(higher_sorted_gid_list),
                len(lower_sorted_gid_list),
            )
            print('Round Sorted Image Re-index: %d Above + %d Below' % args)
        else:
            print('NO IMAGE RE-INDEXING: NOT ENOUGH NEIGHBOR IMAGES SEEN')

        ##################################################################################
        # Step 9: update the bootstrapping algorithm to use the new ensemble during
        #         the next round
        config['classifier_weight_filepath'] = svm_model_path
        config_list.append(config.copy())

        ##################################################################################
        # Step 10: get the test images and classify (cache) their proposals using
        #          the new model ensemble
        if precompute and precompute_test:
            if not is_svm_model_trained:
                depc.delete_property(
                    'localizations_classifier', test_gid_list, config=config
                )
            depc.get_rowids('localizations_classifier', test_gid_list, config=config)

    # Return the list of used configs
    return config_list


def remove_rfdetect(ibs):
    aids = ibs.search_annot_notes('rfdetect')
    notes = ibs.get_annot_notes(aids)
    newnotes = [note.replace('rfdetect', '') for note in notes]
    ibs.set_annot_notes(aids, newnotes)


@register_ibs_method
def set_reviewed_from_target_species_count(ibs, species_set=None, target=1000):
    import random

    if species_set is None:
        species_set = set(
            [
                'giraffe_masai',
                'giraffe_reticulated',
                'turtle_green',
                'turtle_hawksbill',
                'whale_fluke',
                'zebra_grevys',
                'zebra_plains',
            ]
        )

    gid_list = ibs.get_valid_gids()
    ibs.set_image_reviewed(gid_list, [0] * len(gid_list))

    aids_list = ibs.get_image_aids(gid_list)
    species_list = map(ibs.get_annot_species_texts, aids_list)
    species_list = map(set, species_list)

    species_dict = {}
    for species_list_, gid in zip(species_list, gid_list):
        for species in species_list_:
            if species not in species_dict:
                species_dict[species] = []
            species_dict[species].append(gid)

    recover_dict = {}
    while True:
        candidate_list = []
        for species in species_set:
            gid_list = species_dict.get(species, [])
            if len(gid_list) > target:
                candidate_list += gid_list

        if len(candidate_list) == 0:
            break

        candidate = random.choice(candidate_list)
        # print('Purging %d' % (candidate, ))

        aid_list_ = ibs.get_image_aids(candidate)
        species_list_ = ibs.get_annot_species_texts(aid_list_)
        species_set_ = list(set(species_list_) & species_set)
        if len(species_set_) == 1:
            species_ = species_set_[0]
            if species_ not in recover_dict:
                recover_dict[species_] = []
            recover_dict[species_].append(candidate)

        flag = True
        for species in species_dict:
            if candidate in species_dict[species]:
                species_dict[species].remove(candidate)
            if species in species_set and len(species_dict[species]) > target:
                flag = False

        if flag:
            break

    for species in recover_dict:
        random.shuffle(recover_dict[species])

    for species in species_set:
        gid_list = species_dict.get(species, [])

        if species in recover_dict:
            while len(gid_list) < target and len(recover_dict[species]) > 0:
                recover = recover_dict[species].pop(0)
                # print('Recovering %d' % (recover, ))
                gid_list.append(recover)

        print('%r: %d' % (species, len(gid_list),))

    redo = input('Redo? [enter to continue] ')
    redo = redo.strip()
    if len(redo) == 0:
        ibs.set_reviewed_from_target_species_count(species_set=species_set, target=target)
    else:
        gid_list = []
        for species in species_set:
            gid_list += species_dict.get(species, [])
        gid_list = list(set(gid_list))
        ibs.set_image_reviewed(gid_list, [1] * len(gid_list))
        ibs.update_reviewed_unreviewed_image_special_imageset()


def get_classifier2_rf_data_labels(ibs, dataset_tag, category_list):
    depc = ibs.depc_image
    train_gid_set = general_get_imageset_gids(ibs, dataset_tag)
    config = {
        'algo': 'resnet',
    }
    data_list = depc.get_property('features', train_gid_set, 'vector', config=config)
    data_list = np.array(data_list)

    print('Loading labels for images')
    # Load targets
    aids_list = ibs.get_image_aids(train_gid_set)
    species_set_list = [
        set(ibs.get_annot_species_texts(aid_list_)) for aid_list_ in aids_list
    ]
    label_list = [
        [1.0 if category in species_set else 0.0 for category in category_list]
        for species_set in species_set_list
    ]
    label_list = np.array(label_list)

    # Return values
    return train_gid_set, data_list, label_list


def get_classifier_svm_data_labels(ibs, dataset_tag, species_list):
    depc = ibs.depc_image
    train_gid_set = general_get_imageset_gids(ibs, dataset_tag)
    config = {
        'algo': 'resnet',
    }
    data_list = depc.get_property('features', train_gid_set, 'vector', config=config)
    data_list = np.array(data_list)

    print('Loading labels for images')
    # Load targets
    aids_list = ibs.get_image_aids(train_gid_set)
    category_set = set(species_list)
    species_set_list = [
        set(ibs.get_annot_species_texts(aid_list_)) for aid_list_ in aids_list
    ]
    label_list = [
        1 if len(species_set & category_set) else 0 for species_set in species_set_list
    ]
    label_list = np.array(label_list)

    # Return values
    return train_gid_set, data_list, label_list


@register_ibs_method
def classifier_train_image_svm(
    ibs, species_list, output_path=None, dryrun=False, C=1.0, kernel='rbf'
):
    from sklearn import svm, preprocessing

    # Load data
    print('Loading pre-trained features for images')

    # Save model pickle
    if output_path is None:
        output_path = abspath(expanduser(join('~', 'code', 'wbia', 'models')))
    ut.ensuredir(output_path)
    species_list = [species.lower() for species in species_list]
    species_list_str = '.'.join(species_list)
    kernel = str(kernel.lower())

    args = (
        species_list_str,
        kernel,
        C,
    )
    output_filename = 'classifier.svm.image.%s.%s.%s.pkl' % args
    output_filepath = join(output_path, output_filename)
    if not dryrun:
        vals = get_classifier_svm_data_labels(ibs, 'TRAIN_SET', species_list)
        train_gid_set, data_list, label_list = vals

        print('Train SVM scaler using features')
        # Train new scaler and model using data and labels
        scaler = preprocessing.StandardScaler().fit(data_list)
        data_list = scaler.transform(data_list)
        print('Train SVM model using features and target labels')
        model = svm.SVC(C=C, kernel=kernel, probability=True)
        model.fit(data_list, label_list)

        model_tup = (
            model,
            scaler,
        )
        ut.save_cPkl(output_filepath, model_tup)

        # Load model pickle
        model_tup_ = ut.load_cPkl(output_filepath)
        model_, scaler_ = model_tup_

        # Test accuracy
        vals = get_classifier_svm_data_labels(ibs, 'TEST_SET', species_list)
        train_gid_set, data_list, label_list = vals
        # Normalize data
        data_list = scaler_.transform(data_list)
        label_list_ = model_.predict(data_list)
        # score_list_ = model_.decision_function(data_list)  # NOQA
        score_list_ = model_.predict_proba(data_list)  # NOQA
        tp, tn, fp, fn = 0, 0, 0, 0
        for label_, label in zip(label_list_, label_list):
            if label == 1 and label == label_:
                tp += 1
            elif label == 0 and label == label_:
                tn += 1
            elif label == 1 and label != label_:
                fn += 1
            elif label == 0 and label != label_:
                fp += 1
            else:
                raise ValueError

        pos, neg = tp + fn, tn + fp
        correct = tp + tn
        total = tp + tn + fp + fn
        accuracy = correct / total
        print('Accuracy: %0.02f' % (accuracy,))
        print('\t TP: % 4d (%0.02f %%)' % (tp, tp / pos,))
        print('\t FN: % 4d (%0.02f %%)' % (fn, fn / neg,))
        print('\t TN: % 4d (%0.02f %%)' % (tn, tn / neg,))
        print('\t FP: % 4d (%0.02f %%)' % (fp, fp / pos,))

    return output_filepath


@register_ibs_method
def classifier_train_image_svm_sweep(ibs, species_list, precompute=True, **kwargs):

    depc = ibs.depc_image
    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', species_list)

    config_list = [
        (0.5, 'rbf'),
        (1.0, 'rbf'),
        (2.0, 'rbf'),
        (0.5, 'linear'),
        (1.0, 'linear'),
        (2.0, 'linear'),
    ]
    output_filepath_list = []
    for C, kernel in config_list:
        output_filepath = ibs.classifier_train_image_svm(
            species_list, C=C, kernel=kernel, **kwargs
        )
        output_filepath_list.append(output_filepath)

        if precompute:
            config = {
                'algo': '_COMBINED',
                'features': True,
                'feature2_algo': 'resnet',
                'feature2_chip_masking': False,
                'classify': True,
                'classifier_algo': 'svm',
                'classifier_masking': False,
                'classifier_weight_filepath': output_filepath,
            }
            depc.get_rowids('localizations_features', test_gid_list, config=config)
            depc.get_rowids('localizations_classifier', test_gid_list, config=config)
            # config['feature2_chip_masking'] = True
            # config['classifier_masking'] = True
            # depc.get_rowids('localizations_features', test_gid_list, config=config)
            # depc.get_rowids('localizations_classifier', test_gid_list, config=config)

    return output_filepath_list


@register_ibs_method
def classifier2_train_image_rf(
    ibs, species_list, output_path=None, dryrun=False, n_estimators=100
):
    from sklearn import ensemble, preprocessing

    # Load data
    print('Loading pre-trained features for images')

    # Save model pickle
    if output_path is None:
        output_path = abspath(expanduser(join('~', 'code', 'wbia', 'models')))
    ut.ensuredir(output_path)
    species_list = [species.lower() for species in species_list]
    species_list_str = '.'.join(species_list)

    args = (
        species_list_str,
        n_estimators,
    )
    output_filename = 'classifier2.rf.image.%s.%s.pkl' % args
    output_filepath = join(output_path, output_filename)
    if not dryrun:
        vals = get_classifier2_rf_data_labels(ibs, 'TRAIN_SET', species_list)
        train_gid_set, data_list, label_list = vals

        print('Train data scaler using features')
        # Train new scaler and model using data and labels
        scaler = preprocessing.StandardScaler().fit(data_list)
        data_list = scaler.transform(data_list)
        print('Train RF model using features and target labels')
        model = ensemble.RandomForestClassifier(
            n_estimators=n_estimators, max_features=None
        )
        model.fit(data_list, label_list)

        model_tup = (
            model,
            scaler,
        )
        ut.save_cPkl(output_filepath, model_tup)

        # Load model pickle
        model_tup_ = ut.load_cPkl(output_filepath)
        model_, scaler_ = model_tup_

        # Test accuracy
        vals = get_classifier2_rf_data_labels(ibs, 'TEST_SET', species_list)
        train_gid_set, data_list, label_list = vals
        # Normalize data
        data_list = scaler_.transform(data_list)
        label_list_ = model_.predict(data_list)
        # score_list_ = model_.decision_function(data_list)  # NOQA
        score_list_ = model_.predict_proba(data_list)  # NOQA
        tp, tn, fp, fn = 0, 0, 0, 0
        for label_, label in zip(label_list_, label_list):
            if label == 1 and label == label_:
                tp += 1
            elif label == 0 and label == label_:
                tn += 1
            elif label == 1 and label != label_:
                fn += 1
            elif label == 0 and label != label_:
                fp += 1
            else:
                raise ValueError

        pos, neg = tp + fn, tn + fp
        correct = tp + tn
        total = tp + tn + fp + fn
        accuracy = correct / total
        print('Accuracy: %0.02f' % (accuracy,))
        print('\t TP: % 4d (%0.02f %%)' % (tp, tp / pos,))
        print('\t FN: % 4d (%0.02f %%)' % (fn, fn / neg,))
        print('\t TN: % 4d (%0.02f %%)' % (tn, tn / neg,))
        print('\t FP: % 4d (%0.02f %%)' % (fp, fp / pos,))

    return output_filepath


@register_ibs_method
def classifier2_train_image_rf_sweep(ibs, species_list, precompute=True, **kwargs):

    depc = ibs.depc_image
    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', species_list)

    config_list = [
        10,
    ]
    output_filepath_list = []
    for n_estimators in config_list:
        output_filepath = ibs.classifier2_train_image_rf(
            species_list, n_estimators=n_estimators, **kwargs
        )
        output_filepath_list.append(output_filepath)

        if precompute:
            config = {
                'classifier_two_algo': 'rf',
                'classifier_two_weight_filepath': output_filepath,
            }
            depc.get_rowids('classifier_two', test_gid_list, config=config)

    return output_filepath_list


config_list = [
    # {'label': 'All Species',         'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'species_set' : species_set},
    # {'label': 'Masai Giraffe',       'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'species_set' : [ species_set[0] ]},
    # {'label': 'Reticulated Giraffe', 'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'species_set' : [ species_set[1] ]},
    # {'label': 'Sea Turtle',          'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'species_set' : [ species_set[2] ]},
    # {'label': 'Whale Fluke',         'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'species_set' : [ species_set[3] ]},
    # {'label': 'Grevy\'s Zebra',      'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'species_set' : [ species_set[4] ]},
    # {'label': 'Plains Zebra',        'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'species_set' : [ species_set[5] ]},
    # {'label': 'V1',             'grid' : False, 'config_filepath' : 'v1', 'weight_filepath' : 'v1'},
    # {'label': 'V1 (GRID)',      'grid' : True,  'config_filepath' : 'v1', 'weight_filepath' : 'v1'},
    # {'label': 'V2',             'grid' : False, 'config_filepath' : 'v2', 'weight_filepath' : 'v2'},
    # {'label': 'V2 (GRID)',      'grid' : True,  'config_filepath' : 'v2', 'weight_filepath' : 'v2'},
    # {'label': 'V3',             'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3'},
    # {'label': 'V3 (GRID)',      'grid' : True,  'config_filepath' : 'v3', 'weight_filepath' : 'v3'},
    # {'label': 'V3 Whale Shark', 'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set' : set(['whale_shark'])},
    # {'label': 'V3 Whale Fluke', 'grid' : True,  'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set' : set(['whale_fluke'])},
    # {'label': 'V3',                 'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set' : set(['whale_fluke'])},
    # {'label': 'Whale Fluke V1',     'grid' : False, 'config_filepath' : 'whalefluke', 'weight_filepath' : 'whalefluke', 'species_set' : set(['whale_fluke'])},
    # {'label': 'Whale Fluke V2',     'grid' : False, 'config_filepath' : 'whalefluke_v2', 'weight_filepath' : 'whalefluke_v2', 'species_set' : set(['whale_fluke'])},
    # {'label': 'Green',             'grid' : False, 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'species_set' : set(['turtle_green']), 'check_species': False},
    # {'label': 'Hawksbill',         'grid' : False, 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'species_set' : set(['turtle_hawksbill']), 'check_species': False},
    # {'label': 'Sea Turtle',        'grid' : False, 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'species_set' : set(['turtle_green', 'turtle_hawksbill']), 'check_species': False},
    # {'label': 'Green (Head)',      'grid' : False, 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'species_set' : set(['turtle_green+head']), 'check_species': False},
    # {'label': 'Hawksbill (Head)',  'grid' : False, 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'species_set' : set(['turtle_hawksbill+head']), 'check_species': False},
    # {'label': 'Sand Tiger',        'grid' : False, 'config_filepath' : 'sandtiger', 'weight_filepath' : 'sandtiger'},
    # {'label': 'Sand Tiger (Grid)', 'grid' : True,  'config_filepath' : 'sandtiger', 'weight_filepath' : 'sandtiger'},
    # {'label': 'Hammerhead',        'grid' : False, 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead'},
    # {'label': 'Hammerhead (Grid)', 'grid' : True,  'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead'},
    # {'label': 'Sea Turtle',       'grid' : False, 'config_filepath' : 'sea', 'weight_filepath' : 'sea', 'species_set' : set(['turtle_general'])},
    # {'label': 'Shark',            'grid' : False, 'config_filepath' : 'sea', 'weight_filepath' : 'sea', 'species_set' : set(['shark_general'])},
    # {'label': 'Whaleshark',       'grid' : False, 'config_filepath' : 'sea', 'weight_filepath' : 'sea', 'species_set' : set(['whaleshark'])},
    # {'label': 'Sea Turtle (Green)',       'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['turtle_green'])},
    {
        'label': 'Hawksbill 00',
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'seaturtle',
        'weight_filepath': 'seaturtle',
        'include_parts': True,
        'sensitivity': 0.01,
        'nms': True,
        'nms_thresh': 0.00,
        'species_set': set(['turtle_hawksbill']),
    },
    {
        'label': 'Hawksbill 10',
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'seaturtle',
        'weight_filepath': 'seaturtle',
        'include_parts': True,
        'sensitivity': 0.01,
        'nms': True,
        'nms_thresh': 0.10,
        'species_set': set(['turtle_hawksbill']),
    },
    {
        'label': 'Hawksbill 20',
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'seaturtle',
        'weight_filepath': 'seaturtle',
        'include_parts': True,
        'sensitivity': 0.01,
        'nms': True,
        'nms_thresh': 0.20,
        'species_set': set(['turtle_hawksbill']),
    },
    {
        'label': 'Hawksbill 30',
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'seaturtle',
        'weight_filepath': 'seaturtle',
        'include_parts': True,
        'sensitivity': 0.01,
        'nms': True,
        'nms_thresh': 0.30,
        'species_set': set(['turtle_hawksbill']),
    },
    {
        'label': 'Hawksbill 40',
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'seaturtle',
        'weight_filepath': 'seaturtle',
        'include_parts': True,
        'sensitivity': 0.01,
        'nms': True,
        'nms_thresh': 0.40,
        'species_set': set(['turtle_hawksbill']),
    },
    {
        'label': 'Hawksbill 50',
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'seaturtle',
        'weight_filepath': 'seaturtle',
        'include_parts': True,
        'sensitivity': 0.01,
        'nms': True,
        'nms_thresh': 0.50,
        'species_set': set(['turtle_hawksbill']),
    },
    {
        'label': 'Hawksbill 60',
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'seaturtle',
        'weight_filepath': 'seaturtle',
        'include_parts': True,
        'sensitivity': 0.01,
        'nms': True,
        'nms_thresh': 0.60,
        'species_set': set(['turtle_hawksbill']),
    },
    {
        'label': 'Hawksbill 70',
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'seaturtle',
        'weight_filepath': 'seaturtle',
        'include_parts': True,
        'sensitivity': 0.01,
        'nms': True,
        'nms_thresh': 0.70,
        'species_set': set(['turtle_hawksbill']),
    },
    {
        'label': 'Hawksbill 80',
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'seaturtle',
        'weight_filepath': 'seaturtle',
        'include_parts': True,
        'sensitivity': 0.01,
        'nms': True,
        'nms_thresh': 0.80,
        'species_set': set(['turtle_hawksbill']),
    },
    {
        'label': 'Hawksbill 90',
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'seaturtle',
        'weight_filepath': 'seaturtle',
        'include_parts': True,
        'sensitivity': 0.01,
        'nms': True,
        'nms_thresh': 0.90,
        'species_set': set(['turtle_hawksbill']),
    },
    {
        'label': 'Hawksbill 100',
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'seaturtle',
        'weight_filepath': 'seaturtle',
        'include_parts': True,
        'sensitivity': 0.01,
        'nms': True,
        'nms_thresh': 1.00,
        'species_set': set(['turtle_hawksbill']),
    },
    # {'label': 'Hawksbill Heads 00',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['turtle_hawksbill+head'])},
    # {'label': 'Hawksbill Heads 10',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['turtle_hawksbill+head'])},
    # {'label': 'Hawksbill Heads 20',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['turtle_hawksbill+head'])},
    # {'label': 'Hawksbill Heads 30',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['turtle_hawksbill+head'])},
    # {'label': 'Hawksbill Heads 40',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['turtle_hawksbill+head'])},
    # {'label': 'Hawksbill Heads 50',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['turtle_hawksbill+head'])},
    # {'label': 'Hawksbill Heads 60',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['turtle_hawksbill+head'])},
    # {'label': 'Hawksbill Heads 70',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['turtle_hawksbill+head'])},
    # {'label': 'Hawksbill Heads 80',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['turtle_hawksbill+head'])},
    # {'label': 'Hawksbill Heads 90',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['turtle_hawksbill+head'])},
    # {'label': 'Hawksbill Heads 100',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'include_parts': True, 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['turtle_hawksbill+head'])},
    # {'label': 'Sea Turtle 00%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['turtle_green', 'turtle_hawksbill'])},
    # {'label': 'Sea Turtle 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['turtle_green', 'turtle_hawksbill'])},
    # {'label': 'Sea Turtle 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['turtle_green', 'turtle_hawksbill'])},
    # {'label': 'Sea Turtle 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['turtle_green', 'turtle_hawksbill'])},
    # {'label': 'Sea Turtle 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['turtle_green', 'turtle_hawksbill'])},
    # {'label': 'Sea Turtle 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['turtle_green', 'turtle_hawksbill'])},
    # {'label': 'Hammerhead Shark 00%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['shark_hammerhead'])},
    # {'label': 'Hammerhead Shark 10%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['shark_hammerhead'])},
    # {'label': 'Hammerhead Shark 20%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['shark_hammerhead'])},
    # {'label': 'Hammerhead Shark 30%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['shark_hammerhead'])},
    # {'label': 'Hammerhead Shark 40%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['shark_hammerhead'])},
    # {'label': 'Hammerhead Shark 50%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['shark_hammerhead'])},
    # {'label': 'Hammerhead Shark 60%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['shark_hammerhead'])},
    # {'label': 'Hammerhead Shark 70%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['shark_hammerhead'])},
    # {'label': 'Hammerhead Shark 80%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['shark_hammerhead'])},
    # {'label': 'Hammerhead Shark 90%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['shark_hammerhead'])},
    # {'label': 'Hammerhead Shark 100%',        'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'sensitivity': 0.01, 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['shark_hammerhead'])},
    # {'label': 'LYNX',           'grid' : False, 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx'},
    # {'label': 'LYNX (GRID)',    'grid' : True,  'config_filepath' : 'lynx', 'weight_filepath' : 'lynx'},
    # {'label': 'V3',          'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3'},
    # {'label': 'V3 PZ',       'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set': set(['zebra_plains'])},
    # {'label': 'V3 GZ',       'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set': set(['zebra_grevys'])},
    # {'label': 'V3 KENYA',    'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set': set(['zebra_plains', 'zebra_grevys', 'giraffe_reticulated', 'giraffe_masai', 'elephant_savannah', 'antelope', 'dog_wild', 'lion', 'hippopotamus'])},
    # {'label': 'V3 DOMESTIC', 'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set': set(['bird', 'horse_domestic', 'cow_domestic', 'sheep_domestic', 'dog_domestic', 'cat_domestic', 'unspecified_animal'])},
    # {'label': 'V3 OCEAN',    'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set': set(['lionfish', 'turtle_sea', 'whale_shark', 'whale_fluke'])},
    # {'label': 'V3 PERSON',   'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set': set(['person'])},
    # {'label': 'V3 VEHICLE',  'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set': set(['car', 'bicycle', 'motorcycle', 'truck', 'boat', 'bus', 'train', 'airplane'])},
    # {'label': 'SS2', 'algo': 'selective-search-rcnn', 'grid': False, 'species_set' : species_set},
    # {'label': 'YOLO1', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set},
    # {'label': 'YOLO1*', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set, 'classify': True},
    # {'label': 'YOLO1^', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set, 'classify': True, 'classifier_masking': True},
    # {'label': 'YOLO1^ 0.0', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set, 'classify': True, 'p': 0.0, 'classifier_masking': True},
    # {'label': 'YOLO1^ 0.1', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set, 'classify': True, 'p': 0.1, 'classifier_masking': True},
    # {'label': 'YOLO1^ 0.3', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set, 'classify': True, 'p': 0.3, 'classifier_masking': True},
    # {'label': 'YOLO1^ 0.5', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set, 'classify': True, 'p': 0.5, 'classifier_masking': True},
    # {'label': 'YOLO1^ 0.7', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set, 'classify': True, 'p': 0.7, 'classifier_masking': True},
    # {'label': 'YOLO1^ 0.9', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set, 'classify': True, 'p': 0.9, 'classifier_masking': True},
    # {'label': 'YOLO1^ 1.0', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set, 'classify': True, 'p': 1.0, 'classifier_masking': True},
    # {'label': 'SS1', 'algo': 'selective-search', 'species_set' : species_set},
    # {'label': 'YOLO1', 'algo': 'darknet', 'config_filepath': 'pretrained-tiny-pascal', 'species_set' : species_set},
    # {'label': 'YOLO2', 'algo': 'darknet', 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set},
    # {'label': 'FRCNN1', 'algo': 'faster-rcnn', 'config_filepath': 'pretrained-zf-pascal', 'species_set' : species_set},
    # {'label': 'FRCNN2', 'algo': 'faster-rcnn', 'config_filepath': 'pretrained-vgg-pascal', 'species_set' : species_set},
    # {'label': 'SSD1', 'algo': 'ssd', 'config_filepath': 'pretrained-300-pascal', 'species_set' : species_set},
    # {'label': 'SSD2', 'algo': 'ssd', 'config_filepath': 'pretrained-512-pascal', 'species_set' : species_set},
    # {'label': 'SSD3', 'algo': 'ssd', 'config_filepath': 'pretrained-300-pascal-plus', 'species_set' : species_set},
    # {'label': 'SSD4', 'algo': 'ssd', 'config_filepath': 'pretrained-512-pascal-plus', 'species_set' : species_set},
    # {'label': 'COMBINED', 'algo': '_COMBINED', 'species_set' : species_set},
    # {'label': 'COMBINED~0.5', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.50, 'line_dotted': True},
    # {'label': 'COMBINED` 0.5', 'algo': '_COMBINED', 'species_set' : species_set, 'thresh': True, 'index_thresh': 0.5},
    # {'label': 'COMBINED` 0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'thresh': True, 'index_thresh': 0.1},
    # {'label': 'COMBINED` 0.05', 'algo': '_COMBINED', 'species_set' : species_set, 'thresh': True, 'index_thresh': 0.05},
    # {'label': 'COMBINED` 0.01', 'algo': '_COMBINED', 'species_set' : species_set, 'thresh': True, 'index_thresh': 0.01},
    # {'label': 'COMBINED', 'algo': '_COMBINED', 'species_set' : species_set},
    # {'label': 'COMBINED 0', 'algo': '_COMBINED', 'species_set' : species_set},
    # {'label': 'COMBINED 2 None', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.25, 'thresh': True, 'index_thresh': 0.25, 'classify': True, 'p': None, 'classifier_algo': 'svm', 'classifier_weight_filepath': None},
    # {'label': 'COMBINED 3 None', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.25, 'thresh': True, 'index_thresh': 0.25, 'classify': True, 'p': None, 'classifier_algo': 'svm', 'classifier_weight_filepath': 'localizer-zebra-10'},
    # {'label': 'COMBINED 4 None', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.25, 'thresh': True, 'index_thresh': 0.25, 'classify': True, 'p': None, 'classifier_algo': 'svm', 'classifier_weight_filepath': 'localizer-zebra-50'},
    # {'label': 'COMBINED 2 0.5', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.25, 'thresh': True, 'index_thresh': 0.25, 'classify': True, 'p': 'mult', 'classifier_algo': 'svm', 'classifier_weight_filepath': None},
    # {'label': 'COMBINED 3 0.5', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.25, 'thresh': True, 'index_thresh': 0.25, 'classify': True, 'p': 'mult', 'classifier_algo': 'svm', 'classifier_weight_filepath': 'localizer-zebra-10'},
    # {'label': 'COMBINED 4 0.5', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.25, 'thresh': True, 'index_thresh': 0.25, 'classify': True, 'p': 'mult', 'classifier_algo': 'svm', 'classifier_weight_filepath': 'localizer-zebra-50'},
    # {'label': 'COMBINED 4', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.1, 'thresh': True, 'index_thresh': 0.10, 'classify': True, 'classifier_algo': 'svm', 'classifier_weight_filepath': 'localizer-zebra-100'},
    # {
    #     'label'        : 'C_0',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.image.zebra.rbf.1.0.pkl',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'C_1',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.10.rbf.1.0',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'C_2',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.20.rbf.1.0',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'C_3',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.30.rbf.1.0',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'C_4',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.40.rbf.1.0',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'C_5',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.50.rbf.1.0',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'C_6',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.60.rbf.1.0',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'C_7',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.70.rbf.1.0',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'C_8',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.80.rbf.1.0',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'C_9',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.90.rbf.1.0',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'C_10',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.100.rbf.1.0',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'LINEAR,0.5',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models/classifier.svm.image.zebra.linear.0.5.pkl',
    # },
    # {
    #     'label'        : 'LINEAR,1.0',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models/classifier.svm.image.zebra.linear.1.0.pkl',
    # },
    # {
    #     'label'        : 'LINEAR,2.0',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models/classifier.svm.image.zebra.linear.2.0.pkl',
    # },
    # {
    #     'label'        : 'RBF,0.5',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models/classifier.svm.image.zebra.rbf.0.5.pkl',
    # },
    # {
    #     'label'        : 'RBF,1.0',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models/classifier.svm.image.zebra.rbf.1.0.pkl',
    # },
    # {
    #     'label'        : 'RBF,2.0',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models/classifier.svm.image.zebra.rbf.2.0.pkl',
    # },
    # {
    #     'label'        : 'LINEAR,0.5~0.5',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models/classifier.svm.image.zebra.linear.0.5.pkl',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'LINEAR,1.0~0.5',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models/classifier.svm.image.zebra.linear.1.0.pkl',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'LINEAR,2.0~0.5',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models/classifier.svm.image.zebra.linear.2.0.pkl',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'RBF,0.5~0.5',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models/classifier.svm.image.zebra.rbf.0.5.pkl',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'RBF,1.0~0.5',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.image.zebra.rbf.1.0.pkl',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.30,
    #     # 'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'RBF,2.0~0.5',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models/classifier.svm.image.zebra.rbf.2.0.pkl',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'WIC',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.image.zebra.pkl',
    # },
    # {
    #     'label'        : 'COMBINED ~0.75',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'nms'          : True,
    #     'nms_thresh'   : 0.75,
    # },
    # {
    #     'label'        : 'COMBINED ~0.50',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     'line_dotted'  : True,
    # },
    # {
    #     'label'        : 'COMBINED ~0.25',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    # },
    # {
    #     'label'        : 'WIC',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.image.zebra.pkl',
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'WIC ~0.25',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.image.zebra.pkl',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'WIC ~0.5',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.image.zebra.pkl',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.50,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'WIC ~0.75',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.image.zebra.pkl',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.75,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    ###################
    # {
    #     'label'        : 'LOC-E 1',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.10',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'LOC-E 2',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.20',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'LOC-E 3',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.30',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'LOC-E 4',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.40',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'LOC-E 5',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.50',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'LOC-E 6',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.60',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'LOC-E 7',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.70',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'LOC-E 8',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.80',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'LOC-E 9',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.90',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {
    #     'label'        : 'LOC-E 10',
    #     'algo'         : '_COMBINED',
    #     'species_set'  : species_set,
    #     'classify'     : True,
    #     'classifier_algo': 'svm',
    #     'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.localization.zebra.100',
    #     'nms'          : True,
    #     'nms_thresh'   : 0.25,
    #     # 'thresh'       : True,
    #     # 'index_thresh' : 0.25,
    # },
    # {'label': 'COMBINED`* 0.5', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'thresh': True, 'index_thresh': 0.5},
    # {'label': 'COMBINED`* 0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'thresh': True, 'index_thresh': 0.1},
    # {'label': 'COMBINED`* 0.05', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'thresh': True, 'index_thresh': 0.05},
    # {'label': 'COMBINED`* 0.01', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'thresh': True, 'index_thresh': 0.01},
    # {'label': 'COMBINED*', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True},
    # {'label': 'COMBINED`0.1* ~0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'nms': True, 'nms_thresh': 0.1, 'thresh': True, 'index_thresh': 0.1},
    # {'label': 'COMBINED`0.5* ~0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'nms': True, 'nms_thresh': 0.1, 'thresh': True, 'index_thresh': 0.5},
    # {'label': 'COMBINED` ~0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.1, 'thresh': True, 'index_thresh': 0.1},
    # {'label': 'COMBINED`*', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'thresh': True, 'index_thresh': 0.1},
    # {'label': 'COMBINED`', 'algo': '_COMBINED', 'species_set' : species_set, 'thresh': True, 'index_thresh': 0.1},
    # {'label': 'COMBINED* ~0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'nms': True, 'nms_thresh': 0.1},
    # {'label': 'COMBINED ~0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.1},
    # {'label': 'COMBINED*', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True},
    # {'label': 'COMBINED', 'algo': '_COMBINED', 'species_set' : species_set},
    # {'label': 'COMBINED`', 'algo': '_COMBINED', 'species_set' : species_set, 'limited': True},
    # {'label': 'COMBINED`* ~0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'nms': True, 'nms_thresh': 0.1, 'limited': True},
    # {'label': 'COMBINED !0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'conf_thresh': 0.1},
    # {'label': 'COMBINED !0.5', 'algo': '_COMBINED', 'species_set' : species_set, 'conf_thresh': 0.5},
    # {'label': 'COMBINED !0.9', 'algo': '_COMBINED', 'species_set' : species_set, 'conf_thresh': 0.9},
    # {'label': 'COMBINED ~0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.1},
    # {'label': 'COMBINED ~0.5', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.5},
    # {'label': 'COMBINED ~0.9', 'algo': '_COMBINED', 'species_set' : species_set, 'nms': True, 'nms_thresh': 0.9},
    # # {'label': 'YOLO1*', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-tiny-pascal', 'species_set' : species_set, 'classify': True},
    # # {'label': 'YOLO2*', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : species_set, 'classify': True},
    # # {'label': 'FRCNN1*', 'algo': 'faster-rcnn', 'grid': False, 'config_filepath': 'pretrained-zf-pascal', 'species_set' : species_set, 'classify': True},
    # {'label': 'FRCNN2*', 'algo': 'faster-rcnn', 'grid': False, 'config_filepath': 'pretrained-vgg-pascal', 'species_set' : species_set, 'classify': True},
    # # {'label': 'SSD1*', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-300-pascal', 'species_set' : species_set, 'classify': True},
    # # {'label': 'SSD2*', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-512-pascal', 'species_set' : species_set, 'classify': True},
    # # {'label': 'SSD3*', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-300-pascal-plus', 'species_set' : species_set, 'classify': True},
    # {'label': 'SSD4*', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-512-pascal-plus', 'species_set' : species_set, 'classify': True},
    # {'label': 'COMBINED*', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True},
    # {'label': 'COMBINED* !0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'conf_thresh': 0.1},
    # {'label': 'COMBINED* !0.5', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'conf_thresh': 0.5},
    # {'label': 'COMBINED* !0.9', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'conf_thresh': 0.9},
    # {'label': 'COMBINED* ~0.01', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'nms': True, 'nms_thresh': 0.01},
    # {'label': 'COMBINED* ~0.05', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'nms': True, 'nms_thresh': 0.05},
    # {'label': 'COMBINED* ~0.5', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'nms': True, 'nms_thresh': 0.5},
    # {'label': 'COMBINED* ~0.9', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'nms': True, 'nms_thresh': 0.9},
    # {'label': 'COMBINED 0.0', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'p': 0.0},
    # {'label': 'COMBINED 0.1', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'p': 0.1},
    # {'label': 'COMBINED 0.2', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'p': 0.2},
    # {'label': 'COMBINED 0.3', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'p': 0.3},
    # {'label': 'COMBINED 0.4', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'p': 0.4},
    # {'label': 'COMBINED 0.5', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'p': 0.5},
    # {'label': 'COMBINED 0.6', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'p': 0.6},
    # {'label': 'COMBINED 0.7', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'p': 0.7},
    # {'label': 'COMBINED 0.8', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'p': 0.8},
    # {'label': 'COMBINED 0.9', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'p': 0.9},
    # {'label': 'COMBINED 1.0', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True, 'p': 1.0},
    # {'label': 'COMBINED MUL', 'algo': '_COMBINED', 'species_set' : species_set, 'classify': True},
]


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.other.detectgrave
        python -m wbia.other.detectgrave --allexamples
        python -m wbia.other.detectgrave --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
