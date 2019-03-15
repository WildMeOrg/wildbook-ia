from __future__ import absolute_import, division, print_function, unicode_literals
from ibeis_cnn.ingest_ibeis import get_cnn_classifier_cameratrap_binary_training_images_pytorch
from ibeis.control import controller_inject
from ibeis.algo.detect import densenet
from os.path import expanduser, join, abspath, exists
import numpy as np
import utool as ut
import random
import tqdm
import cv2


PYTORCH = True


# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[vulcanfuncs]')


# Must import class before injection
CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))


register_api = controller_inject.get_ibeis_flask_api(__name__)


@register_ibs_method
def vulcan_subsample_imageset(ibs, ratio=10, target_species='elephant_savanna'):
    gid_all_list = ibs.get_valid_gids(is_tile=None)

    imageset_text_list = [
        '20161108_Nikon_Left',
        '20161108_Nikon_Right',
    ]
    imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(imageset_text_list)
    gids_list = ibs.get_imageset_gids(imageset_rowid_list)

    zipped = zip(imageset_text_list, imageset_rowid_list, gids_list)
    for imageset_text, imageset_rowid, gid_list in zipped:
        imageset_text_subsample = '%s_Subsample' % (imageset_text, )
        aids_list = ibs.get_image_aids(gid_list)

        positive_list = []
        negative_list = []
        for gid, aid_list in zip(gid_list, aids_list):
            aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
            if len(aid_list) > 0:
                positive_list.append(gid)
            else:
                negative_list.append(gid)

        num_positive = len(positive_list)
        num_negative = min(len(negative_list), num_positive * ratio)
        random.shuffle(negative_list)
        negative_list = negative_list[:num_negative]
        subsample_list = sorted(positive_list + negative_list)

        args = (imageset_text, imageset_text_subsample, num_positive, num_negative, )
        print('Subsampling %s into %s (%d positive, %d negative)' % args)

        imageset_rowid, = ibs.get_imageset_imgsetids_from_text([imageset_text_subsample])
        ibs.unrelate_images_and_imagesets(gid_all_list, [imageset_rowid] * len(gid_all_list))
        ibs.set_image_imagesettext(
            subsample_list,
            [imageset_text_subsample] * len(subsample_list)
        )
        num_total = len(ibs.get_imageset_gids(imageset_rowid))
        print('...%d images added' % (num_total, ))


@register_ibs_method
def vulcan_get_valid_tile_rowids(ibs, imageset_text_list=None, return_gids=False,
                                 return_configs=False, limit=None, **kwargs):
    if imageset_text_list is None:
        imageset_text_list = [
            'elephant',
            'RR18_BIG_2015_09_23_R_AM',
            'TA24_TPM_L_2016-10-30-A',
            'TA24_TPM_R_2016-10-30-A',
            '2012-08-16_AM_L_Azohi',
            '2012-08-15_AM_R_Marealle',
            '2012-08-14_PM_R_Chediel',
            # '20161108_Nikon_Left',
            # '20161108_Nikon_Right',
            '20161108_Nikon_Left_Sample',
            '20161108_Nikon_Right_Sample',
        ]

    imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(imageset_text_list)
    gids_list = ibs.get_imageset_gids(imageset_rowid_list)
    gid_list = ut.flatten(gids_list)
    gid_list = sorted(gid_list)

    if limit is not None:
        gid_list = gid_list[:limit]

    tile_size = 256
    tile_overlap = 64
    config1 = {
        'tile_width':   tile_size,
        'tile_height':  tile_size,
        'tile_overlap': tile_overlap,
    }
    tiles1_list = ibs.compute_tiles(gid_list=gid_list, **config1)
    tile1_list = ut.flatten(tiles1_list)
    config1_list = [1] * len(tile1_list)

    tile_offset = (tile_size - tile_overlap) // 2
    config2 = {
        'tile_width':    tile_size,
        'tile_height':   tile_size,
        'tile_overlap':  tile_overlap,
        'tile_offset':   tile_offset,
        'allow_borders': False,
    }
    tiles2_list = ibs.compute_tiles(gid_list=gid_list, **config2)
    tile2_list = ut.flatten(tiles2_list)
    config2_list = [2] * len(tile2_list)

    tile_list_ = tile1_list + tile2_list
    config_list_ = config1_list + config2_list
    tile_list = []
    config_list = []

    seen_set = set([])
    for tile, config in sorted(zip(tile_list_, config_list_)):
        if tile not in seen_set:
            tile_list.append(tile)
            config_list.append(config)
        seen_set.add(tile)

    if return_configs:
        value_list = list(zip(tile_list, config_list))
    else:
        value_list = tile_list

    return (gid_list, value_list) if return_gids else value_list


@register_ibs_method
def vulcan_visualize_tiles(ibs, target_species='elephant_savanna', margin=32,
                           **kwargs):
    RANDOM_VISUALIZATION_OFFSET = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5

    value_list = ibs.vulcan_get_valid_tile_rowids(return_configs=True, **kwargs)
    tile_list = ut.take_column(value_list, 0)
    config_list = ut.take_column(value_list, 1)

    ancestor_gid_list = ibs.get_vulcan_image_tile_ancestor_gids(tile_list)
    values = ibs.vulcan_tile_positive_cumulative_area(tile_list, target_species=target_species,
                                                      margin=margin)
    cumulative_area_list, total_area_list, flag_list = values

    ancestor_gid_dict = {}
    zipped = zip(tile_list, config_list, cumulative_area_list, total_area_list, flag_list, ancestor_gid_list)
    for tile, config, cumulative_area, total_area, flag, ancestor_gid in zipped:
        if ancestor_gid not in ancestor_gid_dict:
            ancestor_gid_dict[ancestor_gid] = []
        ancestor_gid_dict[ancestor_gid].append((cumulative_area, total_area, tile, flag, config, ))
    ancestor_key_list = sorted(ancestor_gid_dict.keys())

    canvas_path = abspath(expanduser(join('~', 'Desktop', 'visualize_tiles')))
    ut.delete(canvas_path)
    ut.ensuredir(canvas_path)

    for ancestor_key in ancestor_key_list:
        values_list = ancestor_gid_dict[ancestor_key]
        values_list = sorted(values_list, reverse=True)
        total_positive_tiles = 0

        config_dict = {}
        for values in values_list:
            cumulative_area, total_area, tile, flag, config = values
            config = str(config)
            border = ibs.get_vulcan_image_tile_border_flag(tile)
            if border:
                config = 'border'
            if flag:
                total_positive_tiles += 1
            assert config in ['1', '2', 'border']
            if config not in config_dict:
                config_dict[config] = []
            value = (cumulative_area / total_area, tile, flag, )
            config_dict[config].append(value)

        print('Processing %s (%d positive tiles)' % (ancestor_key, total_positive_tiles, ))
        if total_positive_tiles == 0:
            print('\tContinue')
            continue

        canvas = ibs.get_images(ancestor_key)
        canvas_h, canvas_w = canvas.shape[:2]

        vpadding = np.zeros((10, canvas_w, 3), dtype=np.uint8)
        hpadding = np.zeros((256, 5, 3), dtype=np.uint8)

        config_color_dict = {
            '1'      : (111, 155, 231),
            '2'      : (231, 155, 111),
            'green'  : (111, 231, 155),
            'red'    : (126, 122, 202),
            'gold'   : (  0, 215, 255),
            'border' : (111, 111, 111),
        }

        tile_seen_dict = {}
        tile_canvas_list_bottom = []
        config_key_list = sorted(config_dict.keys())
        for config_key in config_key_list:
            border = config_key == 'border'

            value_list = config_dict[config_key]
            value_list = sorted(value_list, reverse=True)

            area_list_ = ut.take_column(value_list, 0)
            tile_list_ = ut.take_column(value_list, 1)
            flag_list_ = ut.take_column(value_list, 2)
            # Visualize positive tiles

            tile_canvas_list_list = []
            tile_canvas_list = []
            seen_tracker = 0
            w_tracker = 0
            for area, tile, flag in zip(area_list_, tile_list_, flag_list_):
                if not flag:
                    continue

                seen_tracker += 1
                tile_canvas = ibs.get_images(tile)
                h_, w_ = tile_canvas.shape[:2]

                cv2.rectangle(tile_canvas, (margin, margin), (w_ - margin, h_ - margin), config_color_dict['red'], 1)

                aid_list = ibs.get_image_aids(tile)
                aid_list = sorted(aid_list)
                aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
                bbox_list = ibs.get_annot_bboxes(aid_list, reference_tile_gid=tile)
                for aid, bbox in zip(aid_list, bbox_list):
                    if aid not in tile_seen_dict:
                        tile_seen_dict[aid] = 0
                    tile_seen_dict[aid] += 1
                    xtl, ytl, w, h = bbox
                    cv2.rectangle(tile_canvas, (xtl, ytl), (xtl + w, ytl + h), config_color_dict['gold'], 2)

                thickness = 2
                color = config_color_dict[config_key]
                cv2.rectangle(tile_canvas, (1, 1), (256 - 1, 256 - 1), color, 2)

                text = '%0.04f' % (area, )
                text_width, text_height = cv2.getTextSize(text, font, scale, -1)[0]
                cv2.rectangle(tile_canvas, (3, 3), (text_width + 3, text_height + 3), color, -1)
                cv2.putText(tile_canvas, text, (5 + 3, text_height + 3), font, 0.4, (255, 255, 255))

                tile_canvas = np.hstack((hpadding, tile_canvas, hpadding))
                h_, w_ = tile_canvas.shape[:2]
                assert h_ == 256
                if w_tracker + w_ > canvas_w:
                    tile_canvas_list_list.append(tile_canvas_list)
                    tile_canvas_list = []
                    w_tracker = 0
                tile_canvas_list.append(tile_canvas)
                w_tracker += w_
            tile_canvas_list_list.append(tile_canvas_list)

            if seen_tracker > 0:
                for index in range(len(tile_canvas_list_list)):
                    tile_canvas_list = tile_canvas_list_list[index]
                    if len(tile_canvas_list) == 0:
                        continue
                    tile_canvas = np.hstack(tile_canvas_list)
                    h_, w_ = tile_canvas.shape[:2]
                    missing = canvas_w - w_
                    assert h_ == 256 and missing >= 0
                    missing_left = missing // 2
                    missing_right = (missing + 1) // 2
                    assert missing_left + missing_right == missing
                    hpadding_left = np.zeros((256, missing_left, 3), dtype=np.uint8)
                    hpadding_right = np.zeros((256, missing_right, 3), dtype=np.uint8)
                    tile_canvas = np.hstack((hpadding_left, tile_canvas, hpadding_right, ))
                    h_, w_ = tile_canvas.shape[:2]
                    tile_canvas_list_bottom.append(vpadding)
                    tile_canvas_list_bottom.append(tile_canvas)
                    assert h_ == 256 and w_ == canvas_w

            # Visualize location of all tiles
            value_list_ = sorted(zip(tile_list_, flag_list_))
            tile_list = ut.take_column(value_list_, 0)
            flag_list = ut.take_column(value_list_, 1)

            bbox_list = ibs.get_vulcan_image_tile_bboxes(tile_list)
            for bbox, flag in zip(bbox_list, flag_list):
                xtl, ytl, w, h = bbox
                xtl += int(np.around(random.uniform(-RANDOM_VISUALIZATION_OFFSET, RANDOM_VISUALIZATION_OFFSET)))
                ytl += int(np.around(random.uniform(-RANDOM_VISUALIZATION_OFFSET, RANDOM_VISUALIZATION_OFFSET)))
                w   += int(np.around(random.uniform(-RANDOM_VISUALIZATION_OFFSET, RANDOM_VISUALIZATION_OFFSET)))
                h   += int(np.around(random.uniform(-RANDOM_VISUALIZATION_OFFSET, RANDOM_VISUALIZATION_OFFSET)))

                thickness = 2
                color_key = config_key

                if border:
                    thickness = 4
                if flag:
                    color_key = 'green'

                color = config_color_dict[color_key]
                cv2.rectangle(canvas, (xtl, ytl), (xtl + w, ytl + h), color, thickness)

        aid_list = ibs.get_image_aids(ancestor_key)
        aid_list = sorted(aid_list)
        aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
        bbox_list = ibs.get_annot_bboxes(aid_list)
        num_missed = 0
        for aid, bbox in zip(aid_list, bbox_list):
            xtl, ytl, w, h = bbox
            if tile_seen_dict.get(aid, 0) == 0:
                num_missed += 1
            cv2.rectangle(canvas, (xtl, ytl), (xtl + w, ytl + h), config_color_dict['gold'], 2)

        canvas = np.vstack([canvas] + tile_canvas_list_bottom + [vpadding])

        args = (ancestor_key, total_positive_tiles, num_missed, )
        canvas_filename = 'vulcan-tile-gid-%s-num-patches-%d-num-missed-%d.png' % args
        canvas_filepath = join(canvas_path, canvas_filename)
        cv2.imwrite(canvas_filepath, canvas)


@register_ibs_method
def vulcan_tile_positive_cumulative_area(ibs, tile_list, target_species='elephant_savanna',
                                         min_cumulative_percentage=0.025, margin=32,
                                         margin_discount=0.5):
    tile_bbox_list = ibs.get_vulcan_image_tile_bboxes(tile_list)
    aids_list = ibs.get_vulcan_image_tile_aids(tile_list)
    species_set_list = list(map(set, map(ibs.get_annot_species_texts, aids_list)))

    cumulative_area_list = []
    total_area_list = []
    for tile_id, tile_bbox, aid_list, species_set in zip(tile_list, tile_bbox_list, aids_list, species_set_list):
        tile_xtl, tile_ytl, tile_w, tile_h = tile_bbox
        canvas = np.zeros((tile_h, tile_w), dtype=np.float32)
        if target_species in species_set:
            bbox_list = ibs.get_annot_bboxes(aid_list, reference_tile_gid=tile_id)
            for bbox in bbox_list:
                xtl, ytl, w, h = bbox
                xbr = xtl + w
                ybr = ytl + h
                xtl = max(xtl, 0)
                ytl = max(ytl, 0)
                xbr = min(xbr, tile_w)
                ybr = min(ybr, tile_h)
                canvas[ytl: ybr, xtl: xbr] = 1
        canvas[:margin, :]  *= margin_discount
        canvas[:, :margin]  *= margin_discount
        canvas[-margin:, :] *= margin_discount
        canvas[:, -margin:] *= margin_discount
        cumulative_area = int(np.sum(canvas))
        total_area = tile_w * tile_h
        cumulative_area_list.append(cumulative_area)
        total_area_list.append(total_area)

    flag_list = [
        cumulative_area_ >= int(np.floor(total_area_ * min_cumulative_percentage))
        for cumulative_area_, total_area_ in zip(cumulative_area_list, total_area_list)
    ]
    return cumulative_area_list, total_area_list, flag_list


@register_ibs_method
def vulcan_imageset_train_test_split(ibs, target_species='elephant_savanna',
                                     recompute_split=False, **kwargs):
    tile_list = ibs.vulcan_get_valid_tile_rowids(**kwargs)

    values = ibs.vulcan_tile_positive_cumulative_area(tile_list, target_species=target_species)
    cumulative_area_list, total_area_list, flag_list = values

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])
    gid_all_list = ibs.get_valid_gids(is_tile=None)
    ibs.unrelate_images_and_imagesets(gid_all_list, [pid] * len(gid_all_list))
    ibs.unrelate_images_and_imagesets(gid_all_list, [nid] * len(gid_all_list))

    gids = [ gid for gid, flag in zip(tile_list, flag_list) if flag == 1 ]
    print(len(gids))
    ibs.set_image_imagesettext(gids, ['POSITIVE'] * len(gids))

    gids = [ gid for gid, flag in zip(tile_list, flag_list) if flag == 0 ]
    print(len(gids))
    ibs.set_image_imagesettext(gids, ['NEGATIVE'] * len(gids))

    if recompute_split:
        ibs.imageset_train_test_split(is_tile=False)

    train_imgsetid = ibs.add_imagesets('TRAIN_SET')
    test_imgsetid = ibs.add_imagesets('TEST_SET')

    train_gid_list = ibs.get_imageset_gids(train_imgsetid)
    test_gid_list = ibs.get_imageset_gids(test_imgsetid)

    train_gid_set = set(train_gid_list)
    test_gid_set = set(test_gid_list)

    ancestor_gid_list = ibs.get_vulcan_image_tile_ancestor_gids(tile_list)

    tile_train_list = []
    tile_test_list = []

    for tile_id, ancestor_gid in zip(tile_list, ancestor_gid_list):
        if ancestor_gid in train_gid_set:
            tile_train_list.append(tile_id)
        elif ancestor_gid in test_gid_set:
            tile_test_list.append(tile_id)
        else:
            raise ValueError()

    tid_all_list = ibs.get_valid_gids(is_tile=True)
    ibs.unrelate_images_and_imagesets(tid_all_list, [train_imgsetid] * len(tid_all_list))
    ibs.unrelate_images_and_imagesets(tid_all_list, [test_imgsetid]  * len(tid_all_list))

    ibs.set_image_imgsetids(tile_train_list, [train_imgsetid] * len(tile_train_list))
    ibs.set_image_imgsetids(tile_test_list, [test_imgsetid] * len(tile_test_list))

    return tile_list


@register_ibs_method
def vulcan_compute_visual_clusters(ibs, num_clusters=80, n_neighbors=10,
                                   max_images=None,
                                   min_pca_variance=0.9,
                                   cleanup_memory=True,
                                   all_tile_list=None, **kwargs):
    from sklearn.decomposition import PCA
    import numpy as np
    try:
        import hdbscan
        import umap
    except Exception as ex:
        print('Install required dependencies with: \n\tpip install --upgrade numpy pip scikit-image\n\tpip install hdbscan umap-learn')
        raise ex

    if all_tile_list is None:
        all_tile_list = ibs.vulcan_get_valid_tile_rowids(**kwargs)
        if max_images is not None:
            all_tile_list = all_tile_list[:max_images]

    all_tile_list = sorted(all_tile_list)
    hashstr = ut.hash_data(all_tile_list)
    hashstr = hashstr[:16]
    cache_path = ibs.cachedir
    cluster_cache_path = join(cache_path, 'vulcan', 'clusters')
    ut.ensuredir(cluster_cache_path)

    umap_cache_filename = 'umap.%s.%s.pkl' % (hashstr, n_neighbors, )
    umap_cache_filepath = join(cluster_cache_path, umap_cache_filename)

    cluster_cache_filename = 'cluster.%s.%s.%s.pkl' % (hashstr, num_clusters, n_neighbors, )
    cluster_cache_filepath = join(cluster_cache_path, cluster_cache_filename)

    if not exists(cluster_cache_filepath):
        print('Computing clusters for tile list hash %s' % (hashstr, ))

        if not exists(umap_cache_filepath):
            with ut.Timer('Load DenseNet features'):
                config = {
                    'framework': 'torch',
                    'model':     'densenet',
                }
                feature_list = ibs.depc_image.get_property('features', all_tile_list, 'vector', config=config)
                feature_list = np.vstack(feature_list)

            # Whiten
            with ut.Timer('Whiten features'):
                mean = np.mean(feature_list, axis=1).reshape(-1, 1)
                std = np.std(feature_list, axis=1).reshape(-1, 1)
                normalized_feature_list = (feature_list - mean) / std
                if cleanup_memory:
                    feature_list = None

            # Perform PCA
            with ut.Timer('Reduce features with PCA'):
                for pca_index in range(10, 50):
                    pca_ = PCA(n_components=pca_index, whiten=False)
                    pca_feature_list = pca_.fit_transform(normalized_feature_list)
                    variance = sum(pca_.explained_variance_ratio_)
                    print('PCA %d captured %0.04f of the variance' % (pca_index, variance * 100.0, ))

                    if variance >= min_pca_variance:
                        break
                assert variance >= min_pca_variance
                if cleanup_memory:
                    normalized_feature_list = None

            # Further reduce with learned embedding
            with ut.Timer('Reduce features with UMAP'):
                umap_ = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0.001,
                    n_components=2,
                    metric='correlation'
                )
                umap_feature_list = umap_.fit_transform(pca_feature_list)
                if cleanup_memory:
                    pca_feature_list = None

            ut.save_cPkl(umap_cache_filepath, umap_feature_list)
        else:
            umap_feature_list = ut.load_cPkl(umap_cache_filepath)

        # Cluster with HDBSCAN
        with ut.Timer('Cluster features with HDBSCAN'):
            exclude_set = set([-1])

            best_distance = np.inf
            best_samples = None
            best_unclassified = np.inf
            best_prediction_list = None

            found = False
            for min_cluster_size in range(50, 1001, 50):
                if found:
                    break
                for min_samples in list(range(1, 50, 1)):
                    if found:
                        break
                    hdbscan_ = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                    )
                    hdbscan_prediction_list = hdbscan_.fit_predict(umap_feature_list)

                    hdbscan_prediction_list = list(hdbscan_prediction_list)
                    num_unclassified = hdbscan_prediction_list.count(-1)
                    num_found_clusters = len(set(hdbscan_prediction_list) - exclude_set)
                    print('%d, %d Unclassified: %d / %d' % (min_cluster_size, min_samples, num_unclassified, len(hdbscan_prediction_list), ))
                    print('%d, %d Clusters:     %d' % (min_cluster_size, min_samples, num_found_clusters, ))

                    distance_clusters = abs(num_clusters - num_found_clusters)
                    if distance_clusters < best_distance or (distance_clusters == best_distance and num_unclassified < best_unclassified):
                        best_distance = distance_clusters
                        best_unclassified = num_unclassified

                        best_samples = min_samples
                        best_cluster_size = min_cluster_size
                        best_prediction_list = hdbscan_prediction_list[:]
                        print('Found Better')

                    if best_distance == 0 and best_unclassified == 0:
                        print('Found Desired, stopping early')
                        found = True

            num_unclassified = best_prediction_list.count(-1)
            num_found_clusters = len(set(best_prediction_list) - exclude_set)
            print('Best %d, %d Unclassified: %d / %d' % (best_cluster_size, best_samples, num_unclassified, len(best_prediction_list), ))
            print('Best %d, %d Clusters:     %d' % (best_cluster_size, best_samples, num_found_clusters, ))
            print(ut.repr3(ut.dict_hist(best_prediction_list)))

        assignment_zip = list(zip(best_prediction_list, map(list, umap_feature_list)))
        assignment_dict = dict(list(zip(all_tile_list, assignment_zip)))
        ut.save_cPkl(cluster_cache_filepath, assignment_dict)
    else:
        assignment_dict = ut.load_cPkl(cluster_cache_filepath)

    return hashstr, assignment_dict


@register_ibs_method
def vulcan_visualize_clusters(ibs, num_clusters=50, n_neighbors=50,
                              examples=50, **kwargs):
    """
    for n_neighbors in range(5, 61, 5):
        for num_clusters in range(10, 51, 10):
            ibs.vulcan_visualize_clusters(num_clusters, n_neighbors)
    """
    import matplotlib.pyplot as plt
    import plottool as pt
    import random

    values = ibs.vulcan_compute_visual_clusters(num_clusters=num_clusters,
                                                n_neighbors=n_neighbors,
                                                **kwargs)
    hashstr, assignment_dict = values

    cluster_dict = {}
    values_list = list(assignment_dict.items())
    minx, miny = np.inf, np.inf
    maxx, maxy = -np.inf, -np.inf
    for tile_id, (cluster, embedding) in values_list:
        x, y = embedding
        minx = min(minx, x)
        miny = min(miny, y)
        maxx = max(maxx, x)
        maxy = max(maxy, y)
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        value = (tile_id, embedding)
        cluster_dict[cluster].append(value)

    cluster_list = sorted(cluster_dict.keys())
    color_list_ = [(0.2, 0.2, 0.2)]
    color_list = pt.distinct_colors(len(cluster_list) - len(color_list_), randomize=False)
    color_list = color_list_ + color_list

    fig_ = plt.figure(figsize=(15, 15), dpi=400)  # NOQA
    axes = plt.subplot(111)

    axes.grid(False, which='major')
    axes.grid(False, which='minor')
    axes.set_autoscalex_on(False)
    axes.set_autoscaley_on(False)
    axes.set_ylabel('')
    axes.set_ylabel('')
    axes.set_xlim([minx, maxx])
    axes.set_ylim([miny, maxy])

    x_list = []
    x_list_ = []
    y_list = []
    y_list_ = []
    c_list = []
    c_list_ = []
    a_list = []
    a_list_ = []
    m_list = []
    m_list_ = []

    canvas_list = []
    for cluster, color in zip(cluster_list, color_list):
        print('Processing cluster %d' % (cluster, ))
        all_value_list = cluster_dict[cluster]
        random.shuffle(all_value_list)
        num_tiles = min(len(all_value_list), examples)
        value_list = all_value_list[:num_tiles]
        tile_id_list = ut.take_column(value_list, 0)
        tile_id_set = set(tile_id_list)

        for tile_id, embedding in all_value_list:
            x, y = embedding
            c = color
            if tile_id in tile_id_set:
                a = 1.0
                m = '*'
                x_list_.append(x)
                y_list_.append(y)
                c_list_.append(c)
                a_list_.append(a)
                m_list_.append(m)
            else:
                a = 0.2
                m = 'o'
                x_list.append(x)
                y_list.append(y)
                c_list.append(c)
                a_list.append(a)
                m_list.append(m)

        config_ = {
            'draw_annots' : False,
            'thumbsize'   : (densenet.INPUT_SIZE, densenet.INPUT_SIZE),
        }
        thumbnail_list = ibs.depc_image.get_property('thumbnails', tile_id_list, 'img', config=config_)

        color = np.array(color[::-1], dtype=np.float32)
        color = np.around(color * 255.0).astype(np.uint8)
        vertical_color = np.zeros((densenet.INPUT_SIZE, 10, 3), dtype=np.uint8)
        vertical_color += color
        canvas_ = np.hstack([vertical_color] + thumbnail_list + [vertical_color])

        horizontal_color = np.zeros((10, canvas_.shape[1], 3), dtype=np.uint8)
        horizontal_color += color
        canvas = np.vstack([horizontal_color, canvas_, horizontal_color])
        canvas_list.append(canvas)

    zipped = list(zip([True] * len(x_list), x_list, y_list, c_list, a_list, m_list))
    zipped_ = list(zip([False] * len(x_list_), x_list_, y_list_, c_list_, a_list_, m_list_))
    random.shuffle(zipped)
    random.shuffle(zipped_)
    skip_rate = max(0.0, 1.0 - (20000 / len(zipped)))
    print('Using skiprate = %0.02f' % (skip_rate, ))
    zipped_combined = zipped + zipped_
    for flag, x, y, c, a, m in tqdm.tqdm(zipped_combined):
        if flag and random.uniform(0.0, 1.0) < skip_rate:
            continue
        plt.plot([x], [y], color=c, alpha=a, marker=m, linestyle='None')

    canvas = np.vstack(canvas_list)

    args = (hashstr, num_clusters, n_neighbors, )
    canvas_filename = 'vulcan-wic-clusters-%s-%s-%s-examples.png' % args
    canvas_filepath = abspath(expanduser(join('~', 'Desktop', canvas_filename)))
    cv2.imwrite(canvas_filepath, canvas)

    fig_filename = 'vulcan-wic-clusters-%s-%s-%s-plot.png' % args
    fig_filepath = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_filepath, bbox_inches='tight')


@register_ibs_method
def vulcan_wic_train(ibs, ensembles=5, rounds=10,
                     boost_confidence_thresh=0.2,
                     boost_round_ratio=2,
                     num_clusters=80,
                     n_neighbors=10,
                     use_clusters=False,
                     hashstr=None,
                     restart_config_dict=None,
                     **kwargs):
    """
    Example:
       >>> restart_config_dict = {
       >>>     'vulcan-d3e8bf43-boost0': 'https://kaiju.dyn.wildme.io/public/models/classifier2.vulcan.d3e8bf43.0.zip',
       >>>     'vulcan-d3e8bf43-boost1': 'https://kaiju.dyn.wildme.io/public/models/classifier2.vulcan.d3e8bf43.1.zip',
       >>>     'vulcan-d3e8bf43-boost2': 'https://kaiju.dyn.wildme.io/public/models/classifier2.vulcan.d3e8bf43.2.zip',
       >>>     'vulcan-d3e8bf43-boost3': 'https://kaiju.dyn.wildme.io/public/models/classifier2.vulcan.d3e8bf43.3.zip',
       >>>     'vulcan-d3e8bf43-boost4': 'https://kaiju.dyn.wildme.io/public/models/classifier2.vulcan.d3e8bf43.4.zip',
       >>> }
       >>> ibs.vulcan_wic_train(restart_config_dict=restart_config_dict)
       >>>

    """
    import random

    latest_model_tag = None
    config_list = []
    restart_round_num = 0

    if restart_config_dict:
        latest_round = 0
        restart_config_key_list = sorted(restart_config_dict.keys())
        for restart_config_key in restart_config_key_list:
            restart_config_url = restart_config_dict[restart_config_key]

            restart_config_key_ = restart_config_key.strip().split('-')
            assert len(restart_config_key_) == 3
            namespace, hashstr_, round_ = restart_config_key_
            if hashstr is None:
                hashstr = hashstr_
            assert hashstr == hashstr_, 'Cannot mix hash strings in a single restart'
            round_ = int(round_.replace('boost', ''))
            assert latest_round == round_, 'Boosting rounds cannot be skipped, please include'
            latest_round += 1

            densenet.ARCHIVE_URL_DICT[restart_config_key] = restart_config_url
            latest_model_tag = restart_config_key
            config_list.append(
                {'label': 'WIC %s rRound %d' % (hashstr, round_, ), 'classifier_algo': 'densenet', 'classifier_weight_filepath': restart_config_key},
            )

        restart_round_num = latest_round
        ibs.vulcan_wic_validate(config_list)

    # Start training
    if hashstr is None:
        hashstr = ut.random_nonce()[:8]
    print('Using hashstr=%r' % (hashstr, ))

    gid_all_list = ibs.get_valid_gids(is_tile=None)
    all_tile_set = set(ibs.vulcan_get_valid_tile_rowids(**kwargs))

    if use_clusters:
        values = ibs.vulcan_compute_visual_clusters(num_clusters, n_neighbors, **kwargs)
        _, assignment_dict = values

        cluster_dict = {}
        values_list = list(assignment_dict.items())
        for tile_id, (cluster, embedding) in values_list:
            if cluster not in cluster_dict:
                cluster_dict[cluster] = []
            cluster_dict[cluster].append(tile_id)
    else:
        cluster_dict = {
            -1: all_tile_set
        }
    cluster_list = sorted(cluster_dict.keys())

    train_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET')))
    train_gid_set = all_tile_set & train_gid_set

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])
    negative_gid_set = set(ibs.get_imageset_gids(nid))
    negative_gid_set = negative_gid_set & train_gid_set

    num_total = len(train_gid_set)
    num_negative = len(negative_gid_set)
    num_positive = num_total - num_negative

    test_tile_list = list(negative_gid_set)

    class_weights = {
        'positive': 2.0,
        'negative': 1.0,
    }

    for round_num in range(rounds):
        if round_num < restart_round_num:
            continue

        if round_num == 0:
            assert latest_model_tag is None
            # Skip ensemble hard negative mining, save for model mining
            round_confidence_list = [-1.0] * len(test_tile_list)
        else:
            assert latest_model_tag is not None
            round_confidence_list = ibs.vulcan_wic_test(test_tile_list, model_tag=latest_model_tag)

        flag_list = [confidence >= boost_confidence_thresh for confidence in round_confidence_list]
        round_hard_neg_test_tile_list = ut.compress(test_tile_list, flag_list)
        round_hard_neg_confidence_list = ut.compress(round_confidence_list, flag_list)

        weights_path_list = []
        for ensemble_num in range(ensembles):
            boost_imageset_text = 'NEGATIVE-BOOST-%s-%d-%d' % (hashstr, round_num, ensemble_num, )
            boost_id, = ibs.get_imageset_imgsetids_from_text([boost_imageset_text])
            ibs.unrelate_images_and_imagesets(gid_all_list, [boost_id] * len(gid_all_list))

            # Get new images for current round
            if round_num == 0:
                # Add a random confidence that randomizes the first sample
                kickstart_examples = num_positive * boost_round_ratio
                num_cluster_examples = int(kickstart_examples / len(cluster_list))

                ensemble_confidence_dict = {}
                for cluster in cluster_list:
                    cluster_tile_list = list(cluster_dict[cluster])
                    random.shuffle(cluster_tile_list)

                    num_cluster_neg = len(cluster_tile_list)
                    cluster_num_negative = min(num_cluster_examples, num_cluster_neg)

                    for cluster_index, cluster_tile in enumerate(cluster_tile_list):
                        if cluster_index < cluster_num_negative:
                            cluster_confidence = boost_confidence_thresh + random.uniform(0.0, 0.001)
                        else:
                            cluster_confidence = -1.0
                        assert cluster_tile not in ensemble_confidence_dict
                        ensemble_confidence_dict[cluster_tile] = cluster_confidence

                ensemble_confidence_list = [
                    ensemble_confidence_dict[test_tile]
                    for test_tile in test_tile_list
                ]
            else:
                ensemble_latest_model_tag = '%s:%d' % (latest_model_tag, ensemble_num, )
                ensemble_confidence_list = ibs.vulcan_wic_test(test_tile_list, model_tag=ensemble_latest_model_tag)

            flag_list = [confidence >= boost_confidence_thresh for confidence in ensemble_confidence_list]
            ensemble_hard_neg_test_tile_list = ut.compress(test_tile_list, flag_list)
            ensemble_hard_neg_confidence_list = ut.compress(ensemble_confidence_list, flag_list)

            message = 'Found %d ENSEMBLE hard negatives for round %d (boost_confidence_thresh=%0.02f)'
            args = (len(round_hard_neg_test_tile_list), round_num, boost_confidence_thresh, )
            print(message % args)

            message = 'Found %d MODEL hard negatives for round %d model %d (boost_confidence_thresh=%0.02f)'
            args = (len(ensemble_hard_neg_test_tile_list), round_num, ensemble_num, boost_confidence_thresh, )
            print(message % args)

            # Combine the round's ensemble hard negatives with the specific model's hard negatives
            hard_neg_test_tile_list = round_hard_neg_test_tile_list + ensemble_hard_neg_test_tile_list
            hard_neg_confidence_list = round_hard_neg_confidence_list + ensemble_hard_neg_confidence_list
            hard_neg_test_tuple_list_ = sorted(zip(hard_neg_confidence_list, hard_neg_test_tile_list), reverse=True)

            # Remove duplicates with different confidences
            seen_set = set([])
            hard_neg_test_tuple_list = []
            for hard_neg_test_tuple in hard_neg_test_tuple_list_:
                hard_neg_test_confidence, hard_neg_test_tile = hard_neg_test_tuple
                if hard_neg_test_tile not in seen_set:
                    hard_neg_test_tuple_list.append(hard_neg_test_tuple)
                seen_set.add(hard_neg_test_tile)

            num_hard_neg = len(hard_neg_test_tuple_list)
            ensemble_num_negative = min(num_positive * boost_round_ratio, num_hard_neg)
            ensemble_test_tuple_list = hard_neg_test_tuple_list[:ensemble_num_negative]
            ensemble_test_confidence_list = ut.take_column(ensemble_test_tuple_list, 0)
            ensemble_test_tile_list = ut.take_column(ensemble_test_tuple_list, 1)
            args = (
                np.min(ensemble_test_confidence_list),
                np.max(ensemble_test_confidence_list),
                np.mean(ensemble_test_confidence_list),
                np.std(ensemble_test_confidence_list),
            )
            print('Mined negatives with confidences in range [%0.04f, %0.04f] (avg %0.04f +/- %0.04f)' % args)

            # Add previous negative boosting rounds
            last_ensemble_test_tile_list = []
            for previous_round_num in range(0, round_num):
                previous_boost_imageset_text = 'NEGATIVE-BOOST-%s-%d-%d' % (hashstr, previous_round_num, ensemble_num, )
                print('Searching previous boosting rounds for %r: %r' % (boost_imageset_text, previous_boost_imageset_text, ))
                previous_boost_id, = ibs.get_imageset_imgsetids_from_text([previous_boost_imageset_text])
                previous_ensemble_test_tile_list = ibs.get_imageset_gids(previous_boost_id)
                last_ensemble_test_tile_list = previous_ensemble_test_tile_list[:]
                print('\tFound %d images' % (len(previous_ensemble_test_tile_list), ))
                ensemble_test_tile_list += previous_ensemble_test_tile_list

            ensemble_test_tile_list = list(set(ensemble_test_tile_list))

            num_new = len(list(set(ensemble_test_tile_list) - set(last_ensemble_test_tile_list)))
            message = 'Found %d TOTAL hard negatives for round %d model %d (%d new this round)'
            args = (len(ensemble_test_tile_list), round_num, ensemble_num, num_new, )
            print(message % args)

            # Set combined image set to current pool of negatives
            ibs.set_image_imagesettext(ensemble_test_tile_list, [boost_imageset_text] * len(ensemble_test_tile_list))

            args = (hashstr, round_num, ensemble_num, )
            data_path = join(ibs.get_cachedir(), 'extracted-%s-%d-%d' % args)
            output_path = join(ibs.get_cachedir(), 'training', 'classifier-cameratrap-%s-%d-%d' % args)

            # Extract training data
            extracted_path = get_cnn_classifier_cameratrap_binary_training_images_pytorch(
                ibs,
                pid,
                boost_id,
                dest_path=data_path,
                skip_rate_neg=0.0,
            )
            weights_path = densenet.train(extracted_path, output_path, flip=True, rotate=20,
                                          shear=20, class_weights=class_weights,
                                          sample_multiplier=2.0)
            weights_path_list.append(weights_path)

        latest_model_tag, _ = ibs.vulcan_wic_deploy(weights_path_list, hashstr, round_num)
        config_list.append(
            {'label': 'WIC %s Round %d' % (hashstr, round_num, ), 'classifier_algo': 'densenet', 'classifier_weight_filepath': latest_model_tag},
        )
        ibs.vulcan_wic_validate(config_list)

    models = densenet.ARCHIVE_URL_DICT
    print(ut.repr3(models))
    return models


@register_ibs_method
def vulcan_wic_deploy(ibs, weights_path_list, hashstr, round_num=0, temporary=True):
    args = (hashstr, round_num, )
    output_name = 'classifier2.vulcan.%s.%d' % args
    ensemble_path = join(ibs.get_cachedir(), 'training', output_name)
    ut.ensuredir(ensemble_path)

    archive_path = '%s.zip' % (ensemble_path)
    ensemble_weights_path_list = []

    for index, weights_path in enumerate(sorted(weights_path_list)):
        assert exists(weights_path)
        ensemble_weights_path = join(ensemble_path, 'classifier.%d.weights' % (index, ))
        ut.copy(weights_path, ensemble_weights_path)
        ensemble_weights_path_list.append(ensemble_weights_path)

    ensemble_weights_path_list = [ensemble_path] + ensemble_weights_path_list
    ut.archive_files(archive_path, ensemble_weights_path_list, overwrite=True, common_prefix=True)

    output_path = '/data/public/models/%s.zip' % (output_name, )
    ut.copy(archive_path, output_path)

    model_key = 'vulcan-%s-boost%s' % (hashstr, round_num, )
    densenet.ARCHIVE_URL_DICT[model_key] = 'https://kaiju.dyn.wildme.io/public/models/%s.zip' % (output_name, )
    print(ut.repr3(densenet.ARCHIVE_URL_DICT))

    return model_key, output_name


@register_ibs_method
def vulcan_wic_test(ibs, test_tile_list, model_tag=None):
    config = {
        'classifier_algo': 'densenet',
        'classifier_weight_filepath': model_tag,
    }
    prediction_list = ibs.depc_image.get_property('classifier', test_tile_list, 'class', config=config)
    confidence_list = ibs.depc_image.get_property('classifier', test_tile_list, 'score', config=config)
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence in zip(prediction_list, confidence_list)
    ]

    return confidence_list


@register_ibs_method
def vulcan_wic_validate(ibs, config_list=None, offset_black=0, **kwargs):
    all_tile_set = set(ibs.vulcan_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
    test_gid_set = all_tile_set & test_gid_set
    test_tile_list = list(test_gid_set)

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])

    if config_list is None:
        # config_list = [
        #     {'label': 'ELPH WIC B0 Ensemble', 'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost0'},
        #     {'label': 'ELPH WIC B0 0',        'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost0:0'},
        #     {'label': 'ELPH WIC B0 1',        'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost0:1'},
        #     {'label': 'ELPH WIC B0 2',        'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost0:2'},
        #     {'label': 'ELPH WIC B0 3',        'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost0:3'},
        #     {'label': 'ELPH WIC B0 4',        'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost0:4'},
        # ]
        # offset_black = 1

        # config_list = [
        #     {'label': 'ELPH WIC B1 Ensemble', 'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost1'},
        #     {'label': 'ELPH WIC B1 0',        'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost1:0'},
        #     {'label': 'ELPH WIC B1 1',        'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost1:1'},
        #     {'label': 'ELPH WIC B1 2',        'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost1:2'},
        #     {'label': 'ELPH WIC B1 3',        'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost1:3'},
        #     {'label': 'ELPH WIC B1 4',        'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost1:4'},
        # ]
        # offset_black = 1

        # config_list = [
        #     {'label': 'ELPH WIC B0 Ensemble', 'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost0'},
        #     {'label': 'ELPH WIC B1 Ensemble', 'classifier_algo': 'densenet', 'classifier_weight_filepath': 'vulcan-boost1'},
        # ]
        pass

    for target_recall in [None, 0.8, 0.85, 0.9, 0.95, 0.98]:
        ibs.classifier_cameratrap_precision_recall_algo_display(pid, nid, test_gid_list=test_tile_list,
                                                                config_list=config_list,
                                                                offset_black=offset_black,
                                                                target_recall=target_recall,
                                                                force_target_recall=True)


# @register_ibs_method
# def vulcan_wic_validate_image(ibs, strategy='avg', target_species='elephant_savanna', **kwargs):
#     strategy = strategy.lower()
#     assert strategy in ['avg', 'min', 'max', 'thresh']

#     canvas_path = abspath(expanduser(join('~', 'Desktop')))

#     all_tile_set = set(ibs.vulcan_get_valid_tile_rowids(**kwargs))
#     test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
#     test_gid_set = all_tile_set & test_gid_set
#     test_gid_list = list(test_gid_set)
#     ancestor_gid_list = ibs.get_vulcan_image_tile_ancestor_gids(test_gid_list)

#     values = ibs.vulcan_tile_positive_cumulative_area(test_gid_list, target_species=target_species)
#     cumulative_area_list, total_area_list, flag_list = values

#     model_tag = 'vulcan-d3e8bf43-boost3'
#     densenet.ARCHIVE_URL_DICT[model_tag] = 'https://kaiju.dyn.wildme.io/public/models/classifier2.vulcan.d3e8bf43.3.zip'
#     confidence_list = ibs.vulcan_wic_test(test_gid_list, model_tag=model_tag)
#     confidence_list = []

#     ancestor_gt_dict = {}
#     for ancestor_gid, flag in zip(ancestor_gid_list, flag_list):
#         flag_ = ancestor_gt_dict.get(ancestor_gid, False)
#         ancestor_gt_dict[ancestor_gid] = flag_ or flag

#     confidence_list = []
#     for

#     gid_list, tile_list = ibs.vulcan_get_valid_tile_rowids(return_gids=True)

#     test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
#     gid_list = list(set(gid_list) & test_gid_set)

#     config = {
#         'tile_width':   256,
#         'tile_height':  256,
#         'tile_overlap': 64,
#     }
#     tiles_list = ibs.compute_tiles(gid_list=gid_list, **config)
#     tile_list = ut.flatten(tiles_list)

#     aids_list = ibs.get_image_aids(gid_list)
#     length_list = list(map(len, aids_list))
#     flag_list = [0 < length for length in length_list]

#     confidences_list = []
#     for tile_list in tiles_list:
#         confidence_list = ibs.vulcan_wic_test(tile_list, model_tag=model_tag)
#         confidences_list.append(confidence_list)

#     best_accuracy = 0.0
#     best_thresh = None
#     for index in range(100):
#         confidence_thresh = index / 100.0

#         correct = 0
#         for flag, confidence_list in zip(flag_list, confidences_list):
#             # confidence = sum(confidence_list) / len(confidence_list)
#             confidence = np.max(confidence_list)
#             flag_ = confidence >= confidence_thresh
#             correct += 1 if flag == flag_ else 0

#         accuracy = correct / len(flag_list)
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_thresh = confidence_thresh

#     return best_thresh, best_accuracy


@register_ibs_method
def vulcan_wic_visualize_errors_location(ibs, target_species='elephant_savanna',
                                         thresh=0.02, **kwargs):
    def _render(gid_list, flag_list, invert=False):
        if invert:
            flag_list = [
                not flag
                for flag in flag_list
            ]
        tile_list = ut.compress(gid_list, flag_list)

        num_bboxes = 0
        canvas = np.zeros((256, 256), dtype=np.float32)
        for tile in tqdm.tqdm(tile_list):
            aid_list = ibs.get_image_aids(tile)
            aid_list = sorted(aid_list)
            aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
            bbox_list = ibs.get_annot_bboxes(aid_list, reference_tile_gid=tile)
            for bbox in bbox_list:
                xtl, ytl, w, h = bbox
                xbr = xtl + w
                ybr = ytl + h
                canvas[ytl: ybr, xtl: xbr] += 1
                num_bboxes += 1

        max_canvas = np.max(canvas)
        canvas = canvas / max_canvas
        canvas = np.sqrt(canvas)
        canvas = np.around(canvas * 255.0).astype(np.uint8)
        return canvas, max_canvas, num_bboxes

    canvas_path = abspath(expanduser(join('~', 'Desktop')))

    all_tile_set = set(ibs.vulcan_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
    test_gid_set = all_tile_set & test_gid_set
    test_gid_list = list(test_gid_set)

    values = ibs.vulcan_tile_positive_cumulative_area(test_gid_list, target_species=target_species)
    cumulative_area_list, total_area_list, flag_list = values
    gt_positive_test_gid_list = sorted(ut.compress(test_gid_list, flag_list))
    gt_negative_test_gid_list = sorted(set(test_gid_list) - set(gt_positive_test_gid_list))

    model_tag = 'vulcan-d3e8bf43-boost3'
    densenet.ARCHIVE_URL_DICT[model_tag] = 'https://kaiju.dyn.wildme.io/public/models/classifier2.vulcan.d3e8bf43.3.zip'
    gt_positive_confidence_list = ibs.vulcan_wic_test(gt_positive_test_gid_list, model_tag=model_tag)
    gt_negative_confidence_list = ibs.vulcan_wic_test(gt_negative_test_gid_list, model_tag=model_tag)
    gt_positive_flag_list = [
        gt_positive_confidence >= thresh
        for gt_positive_confidence in gt_positive_confidence_list
    ]
    gt_negative_flag_list = [
        gt_negative_confidence >= thresh
        for gt_negative_confidence in gt_negative_confidence_list
    ]

    canvas, max_canvas, num_bboxes = _render(gt_positive_test_gid_list, gt_positive_flag_list)
    canvas_filename = 'visualize_errors_location_gt_positive_pred_positive_max_%d_bboxes_%d.png' % (int(max_canvas), num_bboxes, )
    canvas_filepath = join(canvas_path, canvas_filename)
    cv2.imwrite(canvas_filepath, canvas)

    canvas, max_canvas, num_bboxes = _render(gt_positive_test_gid_list, gt_positive_flag_list, invert=True)
    canvas_filename = 'visualize_errors_location_gt_positive_pred_negative_max_%d_bboxes_%d.png' % (int(max_canvas), num_bboxes, )
    canvas_filepath = join(canvas_path, canvas_filename)
    cv2.imwrite(canvas_filepath, canvas)

    canvas, max_canvas, num_bboxes = _render(gt_negative_test_gid_list, gt_negative_flag_list)
    canvas_filename = 'visualize_errors_location_gt_negative_pred_positive_max_%d_bboxes_%d.png' % (int(max_canvas), num_bboxes, )
    canvas_filepath = join(canvas_path, canvas_filename)
    cv2.imwrite(canvas_filepath, canvas)

    canvas, max_canvas, num_bboxes = _render(gt_negative_test_gid_list, gt_negative_flag_list, invert=True)
    canvas_filename = 'visualize_errors_location_gt_negative_pred_negative_max_%d_bboxes_%d.png' % (int(max_canvas), num_bboxes, )
    canvas_filepath = join(canvas_path, canvas_filename)
    cv2.imwrite(canvas_filepath, canvas)


@register_ibs_method
def vulcan_background_train(ibs, target_species='elephant_savanna'):
    from ibeis_cnn.ingest_ibeis import get_background_training_patches2
    from ibeis_cnn.process import numpy_processed_directory2
    from ibeis_cnn.models.background import train_background
    from ibeis_cnn.utils import save_model
    import numpy as np

    pid, = ibs.get_imageset_imgsetids_from_text(['POSITIVE'])
    train_tile_set = set(ibs.get_imageset_gids(pid))
    ancestor_gid_list = ibs.get_vulcan_image_tile_ancestor_gids(train_tile_set)
    train_gid_set = list(set(ancestor_gid_list))

    aid_list = ut.flatten(ibs.get_image_aids(train_gid_set))
    bbox_list = ibs.get_annot_bboxes(aid_list)
    w_list = ut.take_column(bbox_list, 2)
    annot_size = int(np.around(np.mean(w_list)))

    data_path = join(ibs.get_cachedir(), 'extracted')
    output_path = join(ibs.get_cachedir(), 'training', 'background')

    extracted_path = get_background_training_patches2(ibs, target_species, data_path,
                                                      patch_size=50,
                                                      annot_size=annot_size,
                                                      patch_size_min=0.9,
                                                      patch_size_max=1.1,
                                                      patches_per_annotation=10,
                                                      train_gid_set=train_gid_set,
                                                      visualize=True,
                                                      inside_boundary=False,
                                                      purge=True,
                                                      supercharge_negative_multiplier=10.0)

    # rm -rf /data/ibeis/ELPH_Vulcan/_ibsdb/_ibeis_cache/training/background/

    id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
    model_path = train_background(output_path, X_file, y_file)
    model_state = ut.load_cPkl(model_path)
    assert 'species' not in model_state
    model_state['species'] = target_species
    save_model(model_state, model_path)

    return model_path


@register_ibs_method
def vulcan_background_deploy(ibs, model_path):
    ut.copy(model_path, '/data/public/models/background.vulcan.pkl')
    return model_path


@register_ibs_method
def vulcan_background_compute(ibs, tile_rowid_list, smooth_thresh=20,
                              smooth_ksize=20, model_tag='vulcan'):
    """ Computes tile probability masks."""
    from ibeis.core_annots import postprocess_mask

    tilemask_dir = join(ibs.get_cachedir(), 'tilemasks')
    ut.ensuredir(tilemask_dir)

    # dont use extrmargin here (for now)
    for chunk in ut.ichunks(tile_rowid_list, 256):
        output_path_list = [
            join(tilemask_dir, 'tilemask_tile_id_%d_model_%s.png' % (tile_id, model_tag, ))
            for tile_id in chunk
        ]
        dirty_list = [
            not exists(output_path)
            for output_path in output_path_list
        ]
        if len(dirty_list) > 0:
            chunk_ =  ut.compress(chunk, dirty_list)
            output_path_list_ = ut.compress(output_path_list, dirty_list)

            tile_path_list = ibs.get_image_paths(chunk_)
            mask_gen = ibs.generate_species_background_mask(tile_path_list, model_tag)

            args_list = list(zip(list(mask_gen), output_path_list_))
            for mask, output_path in args_list:
                if smooth_thresh is not None and smooth_ksize is not None:
                    tilemask = postprocess_mask(mask, smooth_thresh, smooth_ksize)
                else:
                    tilemask = mask
                cv2.imwrite(output_path, tilemask)

        for output_path in output_path_list:
            assert exists(output_path)
            tilemask = cv2.imread(output_path)
            yield tilemask


@register_ibs_method
def vulcan_background_validate(ibs, output_path=None, model_tag='vulcan', **kwargs):
    if output_path is None:
        output_path = join(ibs.get_cachedir(), 'tilemasks_combined')
        ut.ensuredir(output_path)

    all_tile_set = set(ibs.vulcan_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
    test_gid_set = all_tile_set & test_gid_set
    test_tile_list = list(test_gid_set)

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])
    positive_gid_set = set(ibs.get_imageset_gids(pid))
    negative_gid_set = set(ibs.get_imageset_gids(nid))

    test_label_list = []
    for test_tile in test_tile_list:
        if test_tile in positive_gid_set:
            test_label_list.append('positive')
        elif test_tile in negative_gid_set:
            test_label_list.append('negative')
        else:
            raise ValueError()

    flag_list = [test_label == 'positive' for test_label in test_label_list]
    test_tile_list = ut.compress(test_tile_list, flag_list)

    masks = ibs.vulcan_background_compute(test_tile_list, model_tag=model_tag)
    masks = list(masks)

    images = ibs.get_images(test_tile_list)
    for test_tile, image, mask in zip(test_tile_list, images, masks):
        output_filename = 'tilemask_combined_tile_id_%d_model_%s.png' % (test_tile, model_tag, )
        output_filepath = join(output_path, output_filename)

        combined = np.around(image.astype(np.float32) * mask.astype(np.float32) / 255.0).astype(np.uint8)
        canvas = np.hstack((image, mask, combined))

        cv2.imwrite(output_filepath, canvas)


@register_ibs_method
def vulcan_localizer_train(ibs, target_species='elephant_savanna', ratio=3.0, **kwargs):
    all_tile_set = set(ibs.vulcan_get_valid_tile_rowids(**kwargs))
    train_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET')))
    train_gid_set = all_tile_set & train_gid_set

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])
    positive_gid_set = set(ibs.get_imageset_gids(pid))
    negative_gid_set = set(ibs.get_imageset_gids(nid))
    positive_gid_set = positive_gid_set & train_gid_set
    negative_gid_set = negative_gid_set & train_gid_set
    negative_gid_list = list(negative_gid_set)

    model_tag = 'vulcan-d3e8bf43-boost3'
    densenet.ARCHIVE_URL_DICT[model_tag] = 'https://kaiju.dyn.wildme.io/public/models/classifier2.vulcan.d3e8bf43.3.zip'
    confidence_list = ibs.vulcan_wic_test(negative_gid_list, model_tag=model_tag)
    zipped = sorted(list(zip(confidence_list, negative_gid_list)), reverse=True)

    num_positive = len(positive_gid_set)
    num_negative = min(len(negative_gid_set), int(ratio * num_positive))
    zipped = zipped[:num_negative]

    negative_gid_set = set(ut.take_column(zipped, 1))
    tid_list = sorted(list(positive_gid_set | negative_gid_set))

    species_list = [target_species]
    values = ibs.localizer_lightnet_train(species_list, gid_list=tid_list, cuda_device='1', target_size=256, **kwargs)
    model_weight_filepath, model_config_filepath = values

    return model_weight_filepath, model_config_filepath


@register_ibs_method
def vulcan_localizer_validate(ibs, target_species='elephant_savanna',
                              thresh=0.02, **kwargs):
    species_set = set([target_species])
    template = (
        [
            {'label': 'Elephant NMS 0%',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 0.00, 'species_set' : species_set},
            {'label': 'Elephant NMS 10%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 0.10, 'species_set' : species_set},
            {'label': 'Elephant NMS 20%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
            {'label': 'Elephant NMS 30%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 0.30, 'species_set' : species_set},
            {'label': 'Elephant NMS 40%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
            {'label': 'Elephant NMS 50%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 0.50, 'species_set' : species_set},
            {'label': 'Elephant NMS 60%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            {'label': 'Elephant NMS 70%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            {'label': 'Elephant NMS 80%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            {'label': 'Elephant NMS 90%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            {'label': 'Elephant NMS 100%', 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 1.00, 'species_set' : species_set},
        ],
        {},
    )

    all_tile_set = set(ibs.vulcan_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
    test_gid_set = all_tile_set & test_gid_set
    all_test_gid_list = list(test_gid_set)

    values = ibs.vulcan_tile_positive_cumulative_area(all_test_gid_list, target_species=target_species)
    cumulative_area_list, total_area_list, flag_list = values
    gt_positive_test_gid_list = sorted(ut.compress(all_test_gid_list, flag_list))
    gt_negative_test_gid_list = sorted(set(all_test_gid_list) - set(gt_positive_test_gid_list))

    model_tag = 'vulcan-d3e8bf43-boost3'
    densenet.ARCHIVE_URL_DICT[model_tag] = 'https://kaiju.dyn.wildme.io/public/models/classifier2.vulcan.d3e8bf43.3.zip'
    all_test_confidence_list = ibs.vulcan_wic_test(all_test_gid_list, model_tag=model_tag)
    all_test_flag_list = [
        all_test_confidence >= thresh
        for all_test_confidence in all_test_confidence_list
    ]
    wic_positive_test_gid_list = sorted(ut.compress(all_test_gid_list, all_test_flag_list))
    wic_negative_test_gid_list = sorted(set(all_test_gid_list) - set(wic_positive_test_gid_list))

    # All Positive Tiles
    config_dict = {'vulcan-gt-positive': template}
    ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=gt_positive_test_gid_list)

    # All WIC-Passing Tiles
    config_dict = {'vulcan-wic-passing': template}
    ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=wic_positive_test_gid_list)

    # All Negative Tiles
    config_dict = {'vulcan-gt-negative': template}
    ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=gt_negative_test_gid_list)

    # All WIC-Failed Tiles
    config_dict = {'vulcan-wic-failing': template}
    ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=wic_negative_test_gid_list)

    # All Test Tiles
    config_dict = {'vulcan-all': template}
    ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=all_test_gid_list)

    config = {'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'vulcan_v0', 'weight_filepath' : 'vulcan_v0', 'nms': True, 'nms_thresh': 0.40, 'sensitivity': 0.2}
    ibs.visualize_predictions(config, gid_list=all_test_gid_list)
    ibs.visualize_ground_truth(config, gid_list=all_test_gid_list)


# def __delete_old_tiles(ibs, ):
#     tid_all_list = ibs.get_valid_gids(is_tile=True)

#     imageset_text_list = [
#         'elephant',
#         'RR18_BIG_2015_09_23_R_AM',
#         'TA24_TPM_L_2016-10-30-A',
#         'TA24_TPM_R_2016-10-30-A',
#         '2012-08-16_AM_L_Azohi',
#         '2012-08-15_AM_R_Marealle',
#         '2012-08-14_PM_R_Chediel',
#         # '20161108_Nikon_Left',
#         # '20161108_Nikon_Right',
#         '20161108_Nikon_Left_Sample',
#         '20161108_Nikon_Right_Sample',
#     ]

#     imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(imageset_text_list)
#     gids_list = ibs.get_imageset_gids(imageset_rowid_list)
#     gid_list = ut.flatten(gids_list)
#     gid_list = sorted(gid_list)

#     tile_size = 256
#     tile_overlap = 64
#     config1 = {
#         'tile_width':   tile_size,
#         'tile_height':  tile_size,
#         'tile_overlap': tile_overlap,
#     }
#     tiles1_list = ibs.compute_tiles(gid_list=gid_list, **config1)
#     tile1_list = ut.flatten(tiles1_list)
#     remaining = set(tid_all_list) - set(tile1_list)
#     ibs.delete_images(remaining, trash_images=True)

#     table = ibs.depc_image['tiles']
#     depc_all_rowid_list = table._get_all_rowids()
#     depc_rowid_list = table.get_rowids_from_root(gid_list, config=config1)
#     remaining = set(depc_all_rowid_list) - set(depc_rowid_list)
#     table.delete_rows(remaining, delete_extern=True)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.other.vulcanfuncs
        python -m ibeis.other.vulcanfuncs --allexamples
        python -m ibeis.other.vulcanfuncs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
