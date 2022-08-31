# -*- coding: utf-8 -*-
import logging
import random
from functools import partial  # NOQA
from os.path import abspath, exists, expanduser, join

import cv2
import numpy as np
import tqdm
import utool as ut

from wbia.algo.detect import densenet
from wbia.control import controller_inject
from wbia.other.detectexport import (
    get_cnn_classifier_cameratrap_binary_training_images_pytorch,
)

logger = logging.getLogger('wbia')


PYTORCH = True


# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[scoutfuncs]')


# Must import class before injection
CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)


register_api = controller_inject.get_wbia_flask_api(__name__)


# @register_ibs_method
# def scout_subsample_imageset(ibs, ratio=10, target_species='elephant_savanna'):
#     gid_all_list = ibs.get_valid_gids(is_tile=None)

#     imageset_text_list = [
#         '20161106_Nikon_Left',
#         '20161106_Nikon_Right',
#         '20161108_Nikon_Left',
#         '20161108_Nikon_Right',
#     ]
#     imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(imageset_text_list)
#     gids_list = ibs.get_imageset_gids(imageset_rowid_list)

#     zipped = zip(imageset_text_list, imageset_rowid_list, gids_list)
#     for imageset_text, imageset_rowid, gid_list in zipped:
#         imageset_text_subsample = '%s_Sample' % (imageset_text, )
#         aids_list = ibs.get_image_aids(gid_list)

#         positive_list = []
#         negative_list = []
#         for gid, aid_list in zip(gid_list, aids_list):
#             aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
#             if len(aid_list) > 0:
#                 positive_list.append(gid)
#             else:
#                 negative_list.append(gid)

#         num_positive = len(positive_list)
#         num_negative = min(len(negative_list), num_positive * ratio)
#         random.shuffle(negative_list)
#         negative_list = negative_list[:num_negative]
#         subsample_list = sorted(positive_list + negative_list)

#         args = (imageset_text, imageset_text_subsample, num_positive, num_negative, )
#         logger.info('Subsampling %s into %s (%d positive, %d negative)' % args)

#         imageset_rowid, = ibs.get_imageset_imgsetids_from_text([imageset_text_subsample])
#         ibs.unrelate_images_and_imagesets(gid_all_list, [imageset_rowid] * len(gid_all_list))
#         ibs.set_image_imagesettext(
#             subsample_list,
#             [imageset_text_subsample] * len(subsample_list)
#         )
#         num_total = len(ibs.get_imageset_gids(imageset_rowid))
#         logger.info('...%d images added' % (num_total, ))


@register_ibs_method
def scout_print_database_stats(ibs, target_species='elephant_savanna'):
    tid_list = ibs.scout_get_valid_tile_rowids()
    aids_list = ibs.get_image_aids(tid_list)

    tile_has_annots = 0
    tile_has_elephs = 0
    for tid, aid_list in zip(tid_list, aids_list):
        if len(aid_list) > 0:
            tile_has_annots += 1
            aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
            if len(aid_list) > 0:
                tile_has_elephs += 1

    gid_list = ibs.get_tile_ancestor_gids(tid_list)
    gid_list = list(set(gid_list))
    aids_list = ibs.get_image_aids(gid_list)

    image_has_annots = 0
    image_has_elephs = 0
    for gid, aid_list in zip(gid_list, aids_list):
        if len(aid_list) > 0:
            image_has_annots += 1
            aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
            if len(aid_list) > 0:
                image_has_elephs += 1

    aid_list = ut.flatten(aids_list)
    num_annots = len(aid_list)
    aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
    num_elephs = len(aid_list)

    logger.info('\nAnnotations: %d' % (num_annots,))
    logger.info('\t%d are elephants' % (num_elephs,))

    logger.info('\nTiles: %d' % (len(tid_list),))
    logger.info('\tfrom %d images' % (len(gid_list),))
    logger.info('\t%d has annots' % (tile_has_annots,))
    logger.info('\t%d has elephants' % (tile_has_elephs,))

    logger.info('\nImages: %d' % (len(gid_list),))
    logger.info('\t%d has annots' % (image_has_annots,))
    logger.info('\t%d has elephants' % (image_has_elephs,))


# def __delete_old_tiles(ibs, **kwargs):
#     tid_all_list = ibs.get_valid_gids(is_tile=True)

#     imageset_text_list = [
#         'elephant',
#         'RR18_BIG_2015_09_23_R_AM',
#         'TA24_TPM_L_2016-10-30-A',
#         'TA24_TPM_R_2016-10-30-A',
#         '2012-08-16_AM_L_Azohi',
#         '2012-08-15_AM_R_Marealle',
#         '2012-08-14_PM_R_Chediel',
#         '20161108_Nikon_Left',
#         '20161108_Nikon_Right',
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


# def __export_test_images(ibs, **kwargs):
#     all_tid_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
#     test_tid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
#     test_tid_set = all_tid_set & test_tid_set
#     test_tid_list = list(test_tid_set)
#     test_gid_list = ibs.get_tile_ancestor_gids(test_tid_list)
#     test_gid_set = list(set(test_gid_list))
#     image_path_list = ibs.get_image_paths(test_gid_set)

#     source_path = ibs.imgdir
#     output_path = abspath(expanduser(join('~', 'Downloads', 'export')))
#     ut.ensuredir(output_path)
#     for image_path_src in image_path_list:
#         image_path_dst = image_path_src.replace(source_path, output_path)
#         logger.info(image_path_src, image_path_dst)
#         ut.copy(image_path_src, image_path_dst)


@register_ibs_method
def export_dataset(ibs, **kwargs):
    gid_list = ibs.get_valid_gids(is_tile=False)
    ibs.export_to_coco(
        ['elephant_savanna'], target_size=None, gid_list=gid_list, require_named=False
    )


@register_ibs_method
def recompute_tiles(ibs, gid_list):
    tid_list = ibs.scout_get_valid_tile_rowids(gid_list=gid_list)
    ibs.delete_images(tid_list, trash_images=False)
    ibs.depc_image.delete_property_all('tiles', gid_list)


@register_ibs_method
def scout_get_valid_tile_rowids(
    ibs,
    imageset_text_list=None,
    return_gids=False,
    return_configs=False,
    limit=None,
    gid_list=None,
    include_grid2=True,
    **kwargs
):
    if gid_list is None:
        # if imageset_text_list is None:
        #     imageset_text_list = [
        #         'elephant',
        #         'RR18_BIG_2015_09_23_R_AM',
        #         'TA24_TPM_L_2016-10-30-A',
        #         'TA24_TPM_R_2016-10-30-A',
        #         '2012-08-16_AM_L_Azohi',
        #         '2012-08-15_AM_R_Marealle',
        #         '2012-08-14_PM_R_Chediel',
        #         # '20161106_Nikon_Left',
        #         # '20161106_Nikon_Right',
        #         '20161106_Nikon_Left_Sample',
        #         '20161106_Nikon_Right_Sample',
        #         # '20161108_Nikon_Left',
        #         # '20161108_Nikon_Right',
        #         '20161108_Nikon_Left_Sample',
        #         '20161108_Nikon_Right_Sample',
        #     ]

        # imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(imageset_text_list)
        # gids_list = ibs.get_imageset_gids(imageset_rowid_list)
        # gid_list = ut.flatten(gids_list)
        gid_list = ibs.get_valid_gids()

    gid_list = sorted(gid_list)

    if limit is not None:
        gid_list = gid_list[:limit]

    tile_size = 256
    tile_overlap = 64
    config1 = {
        'tile_width': tile_size,
        'tile_height': tile_size,
        'tile_overlap': tile_overlap,
    }
    tiles1_list = ibs.compute_tiles(gid_list=gid_list, **config1)
    tile1_list = ut.flatten(tiles1_list)
    config1_list = [1] * len(tile1_list)

    if include_grid2:
        tile_offset = (tile_size - tile_overlap) // 2
        config2 = {
            'tile_width': tile_size,
            'tile_height': tile_size,
            'tile_overlap': tile_overlap,
            'tile_offset': tile_offset,
            'allow_borders': False,
        }
        tiles2_list = ibs.compute_tiles(gid_list=gid_list, **config2)
        tile2_list = ut.flatten(tiles2_list)
        config2_list = [2] * len(tile2_list)
    else:
        tile2_list = []
        config2_list = []

    tile_list_ = tile1_list + tile2_list
    config_list_ = config1_list + config2_list
    tile_list = []
    config_list = []

    seen_set = set()
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
def scout_visualize_tiles(ibs, target_species='elephant_savanna', margin=32, **kwargs):
    RANDOM_VISUALIZATION_OFFSET = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5

    value_list = ibs.scout_get_valid_tile_rowids(return_configs=True, **kwargs)
    tile_list = ut.take_column(value_list, 0)
    config_list = ut.take_column(value_list, 1)

    ancestor_gid_list = ibs.get_tile_ancestor_gids(tile_list)
    values = ibs.scout_tile_positive_cumulative_area(
        tile_list, target_species=target_species, margin=margin
    )
    cumulative_area_list, total_area_list, flag_list = values

    ancestor_gid_dict = {}
    zipped = zip(
        tile_list,
        config_list,
        cumulative_area_list,
        total_area_list,
        flag_list,
        ancestor_gid_list,
    )
    for tile, config, cumulative_area, total_area, flag, ancestor_gid in zipped:
        if ancestor_gid not in ancestor_gid_dict:
            ancestor_gid_dict[ancestor_gid] = []
        ancestor_gid_dict[ancestor_gid].append(
            (
                cumulative_area,
                total_area,
                tile,
                flag,
                config,
            )
        )
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
            border = ibs.get_tile_border_flag(tile)
            if border:
                config = 'border'
            if flag:
                total_positive_tiles += 1
            assert config in ['1', '2', 'border']
            if config not in config_dict:
                config_dict[config] = []
            value = (
                cumulative_area / total_area,
                tile,
                flag,
            )
            config_dict[config].append(value)

        logger.info(
            'Processing %s (%d positive tiles)'
            % (
                ancestor_key,
                total_positive_tiles,
            )
        )
        if total_positive_tiles == 0:
            logger.info('\tContinue')
            continue

        canvas = ibs.get_images(ancestor_key)
        canvas_h, canvas_w = canvas.shape[:2]

        vpadding = np.zeros((10, canvas_w, 3), dtype=np.uint8)
        hpadding = np.zeros((256, 5, 3), dtype=np.uint8)

        config_color_dict = {
            '1': (111, 155, 231),
            '2': (231, 155, 111),
            'green': (111, 231, 155),
            'red': (126, 122, 202),
            'gold': (0, 215, 255),
            'border': (111, 111, 111),
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

                cv2.rectangle(
                    tile_canvas,
                    (margin, margin),
                    (w_ - margin, h_ - margin),
                    config_color_dict['red'],
                    1,
                )

                aid_list = ibs.get_image_aids(tile)
                aid_list = sorted(aid_list)
                aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
                bbox_list = ibs.get_annot_bboxes(aid_list, reference_tile_gid=tile)
                for aid, bbox in zip(aid_list, bbox_list):
                    if aid not in tile_seen_dict:
                        tile_seen_dict[aid] = 0
                    tile_seen_dict[aid] += 1
                    xtl, ytl, w, h = bbox
                    cv2.rectangle(
                        tile_canvas,
                        (xtl, ytl),
                        (xtl + w, ytl + h),
                        config_color_dict['gold'],
                        2,
                    )

                thickness = 2
                color = config_color_dict[config_key]
                cv2.rectangle(tile_canvas, (1, 1), (256 - 1, 256 - 1), color, 2)

                text = '{:0.04f}'.format(area)
                text_width, text_height = cv2.getTextSize(text, font, scale, -1)[0]
                cv2.rectangle(
                    tile_canvas, (3, 3), (text_width + 3, text_height + 3), color, -1
                )
                cv2.putText(
                    tile_canvas,
                    text,
                    (5 + 3, text_height + 3),
                    font,
                    0.4,
                    (255, 255, 255),
                )

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
                    tile_canvas = np.hstack(
                        (
                            hpadding_left,
                            tile_canvas,
                            hpadding_right,
                        )
                    )
                    h_, w_ = tile_canvas.shape[:2]
                    tile_canvas_list_bottom.append(vpadding)
                    tile_canvas_list_bottom.append(tile_canvas)
                    assert h_ == 256 and w_ == canvas_w

            # Visualize location of all tiles
            value_list_ = sorted(zip(tile_list_, flag_list_))
            tile_list = ut.take_column(value_list_, 0)
            flag_list = ut.take_column(value_list_, 1)

            bbox_list = ibs.get_tile_bboxes(tile_list)
            for bbox, flag in zip(bbox_list, flag_list):
                xtl, ytl, w, h = bbox
                xtl += int(
                    np.around(
                        random.uniform(
                            -RANDOM_VISUALIZATION_OFFSET, RANDOM_VISUALIZATION_OFFSET
                        )
                    )
                )
                ytl += int(
                    np.around(
                        random.uniform(
                            -RANDOM_VISUALIZATION_OFFSET, RANDOM_VISUALIZATION_OFFSET
                        )
                    )
                )
                w += int(
                    np.around(
                        random.uniform(
                            -RANDOM_VISUALIZATION_OFFSET, RANDOM_VISUALIZATION_OFFSET
                        )
                    )
                )
                h += int(
                    np.around(
                        random.uniform(
                            -RANDOM_VISUALIZATION_OFFSET, RANDOM_VISUALIZATION_OFFSET
                        )
                    )
                )

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
            cv2.rectangle(
                canvas, (xtl, ytl), (xtl + w, ytl + h), config_color_dict['gold'], 2
            )

        canvas = np.vstack([canvas] + tile_canvas_list_bottom + [vpadding])

        args = (
            ancestor_key,
            total_positive_tiles,
            num_missed,
        )
        canvas_filename = 'scout-tile-gid-%s-num-patches-%d-num-missed-%d.png' % args
        canvas_filepath = join(canvas_path, canvas_filename)
        cv2.imwrite(canvas_filepath, canvas)


@register_ibs_method
def scout_tile_positive_cumulative_area(
    ibs,
    tile_list,
    target_species='elephant_savanna',
    min_cumulative_percentage=0.025,
    margin=32,
    margin_discount=0.5,
):
    tile_bbox_list = ibs.get_tile_bboxes(tile_list)
    aids_list = ibs.get_tile_aids(tile_list)
    species_set_list = list(map(set, map(ibs.get_annot_species_texts, aids_list)))

    cumulative_area_list = []
    total_area_list = []
    for tile_id, tile_bbox, aid_list, species_set in zip(
        tile_list, tile_bbox_list, aids_list, species_set_list
    ):
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
                canvas[ytl:ybr, xtl:xbr] = 1
        canvas[:margin, :] *= margin_discount
        canvas[:, :margin] *= margin_discount
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
def scout_imageset_train_test_split(ibs, recompute_split=False, **kwargs):
    tile_list = ibs.scout_get_valid_tile_rowids(**kwargs)
    values = ibs.scout_tile_positive_cumulative_area(tile_list, **kwargs)
    cumulative_area_list, total_area_list, flag_list = values

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])
    pidi, nidi = ibs.get_imageset_imgsetids_from_text(
        ['POSITIVE_IMAGE', 'NEGATIVE_IMAGE']
    )
    gid_all_list = ibs.get_valid_gids(is_tile=None)
    ibs.unrelate_images_and_imagesets(gid_all_list, [pid] * len(gid_all_list))
    ibs.unrelate_images_and_imagesets(gid_all_list, [nid] * len(gid_all_list))
    ibs.unrelate_images_and_imagesets(gid_all_list, [pidi] * len(gid_all_list))
    ibs.unrelate_images_and_imagesets(gid_all_list, [nidi] * len(gid_all_list))

    gids = [gid for gid, flag in zip(tile_list, flag_list) if flag == 1]
    logger.info(len(gids))
    ibs.set_image_imgsetids(gids, [pid] * len(gids))
    pgidsi = list(set(ibs.get_tile_ancestor_gids(gids)))
    ibs.set_image_imgsetids(pgidsi, [pidi] * len(pgidsi))

    gids = [gid for gid, flag in zip(tile_list, flag_list) if flag == 0]
    logger.info(len(gids))
    ibs.set_image_imgsetids(gids, [nid] * len(gids))
    ngidsi = list(set(ibs.get_tile_ancestor_gids(gids)))
    ngidsi = list(set(ngidsi) - set(pgidsi))
    ibs.set_image_imgsetids(ngidsi, [nidi] * len(ngidsi))

    if recompute_split:
        ibs.imageset_train_test_split(is_tile=False)

    train_imgsetid = ibs.add_imagesets('TRAIN_SET')
    test_imgsetid = ibs.add_imagesets('TEST_SET')

    train_gid_list = ibs.get_imageset_gids(train_imgsetid)
    test_gid_list = ibs.get_imageset_gids(test_imgsetid)

    train_gid_set = set(train_gid_list)
    test_gid_set = set(test_gid_list)
    assert len(test_gid_set & train_gid_set) == 0
    assert len(test_gid_set) + len(train_gid_set) == len(
        ibs.get_valid_gids(is_tile=False)
    )

    ancestor_gid_list = ibs.get_tile_ancestor_gids(tile_list)

    tile_train_list = []
    tile_test_list = []

    for tile_id, ancestor_gid in zip(tile_list, ancestor_gid_list):
        if ancestor_gid in train_gid_set:
            tile_train_list.append(tile_id)
        elif ancestor_gid in test_gid_set:
            tile_test_list.append(tile_id)
        else:
            raise ValueError()

    # Set tiles
    tid_all_list = ibs.get_valid_gids(is_tile=True)
    ibs.unrelate_images_and_imagesets(tid_all_list, [train_imgsetid] * len(tid_all_list))
    ibs.unrelate_images_and_imagesets(tid_all_list, [test_imgsetid] * len(tid_all_list))

    ibs.set_image_imgsetids(tile_train_list, [train_imgsetid] * len(tile_train_list))
    ibs.set_image_imgsetids(tile_test_list, [test_imgsetid] * len(tile_test_list))

    return tile_list


@register_ibs_method
def scout_compute_visual_clusters(
    ibs,
    num_clusters=80,
    n_neighbors=10,
    max_images=None,
    min_pca_variance=0.9,
    cleanup_memory=True,
    all_tile_list=None,
    reclassify_outliers=True,
    **kwargs
):
    import numpy as np
    import scipy
    from sklearn.decomposition import PCA

    try:
        import hdbscan
        import umap
    except Exception as ex:
        logger.info(
            'Install required dependencies with: \n\tpip install --upgrade numpy pip scikit-image\n\tpip install hdbscan umap-learn'
        )
        raise ex

    if all_tile_list is None:
        all_tile_list = ibs.scout_get_valid_tile_rowids(**kwargs)
        if max_images is not None:
            all_tile_list = all_tile_list[:max_images]

    all_tile_list = sorted(all_tile_list)
    hashstr = ut.hash_data(all_tile_list)
    hashstr = hashstr[:16]
    cache_path = ibs.cachedir
    cluster_cache_path = join(cache_path, 'scout', 'clusters')
    ut.ensuredir(cluster_cache_path)

    umap_cache_filename = 'umap.{}.{}.pkl'.format(
        hashstr,
        n_neighbors,
    )
    umap_cache_filepath = join(cluster_cache_path, umap_cache_filename)

    cluster_cache_filename = 'cluster.{}.{}.{}.pkl'.format(
        hashstr,
        num_clusters,
        n_neighbors,
    )
    cluster_cache_filepath = join(cluster_cache_path, cluster_cache_filename)

    if not exists(cluster_cache_filepath):
        logger.info('Computing clusters for tile list hash {}'.format(hashstr))

        if not exists(umap_cache_filepath):
            with ut.Timer('Load DenseNet features'):
                config = {
                    'framework': 'torch',
                    'model': 'densenet',
                }
                feature_list = ibs.depc_image.get_property(
                    'features', all_tile_list, 'vector', config=config
                )
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
                    logger.info(
                        'PCA %d captured %0.04f of the variance'
                        % (
                            pca_index,
                            variance * 100.0,
                        )
                    )

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
                    metric='correlation',
                )
                umap_feature_list = umap_.fit_transform(pca_feature_list)
                if cleanup_memory:
                    pca_feature_list = None

            ut.save_cPkl(umap_cache_filepath, umap_feature_list)
        else:
            umap_feature_list = ut.load_cPkl(umap_cache_filepath)

        # Cluster with HDBSCAN
        with ut.Timer('Cluster features with HDBSCAN'):
            exclude_set = {-1}

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
                    logger.info(
                        '%d, %d Unclassified: %d / %d'
                        % (
                            min_cluster_size,
                            min_samples,
                            num_unclassified,
                            len(hdbscan_prediction_list),
                        )
                    )
                    logger.info(
                        '%d, %d Clusters:     %d'
                        % (
                            min_cluster_size,
                            min_samples,
                            num_found_clusters,
                        )
                    )

                    distance_clusters = abs(num_clusters - num_found_clusters)
                    if distance_clusters < best_distance or (
                        distance_clusters == best_distance
                        and num_unclassified < best_unclassified
                    ):
                        best_distance = distance_clusters
                        best_unclassified = num_unclassified

                        best_samples = min_samples
                        best_cluster_size = min_cluster_size
                        best_prediction_list = hdbscan_prediction_list[:]
                        logger.info('Found Better')

                    if best_distance == 0 and best_unclassified == 0:
                        logger.info('Found Desired, stopping early')
                        found = True

            num_unclassified = best_prediction_list.count(-1)
            num_found_clusters = len(set(best_prediction_list) - exclude_set)
            logger.info(
                'Best %d, %d Unclassified: %d / %d'
                % (
                    best_cluster_size,
                    best_samples,
                    num_unclassified,
                    len(best_prediction_list),
                )
            )
            logger.info(
                'Best %d, %d Clusters:     %d'
                % (
                    best_cluster_size,
                    best_samples,
                    num_found_clusters,
                )
            )
            logger.info(ut.repr3(ut.dict_hist(best_prediction_list)))

        assignment_zip = list(zip(best_prediction_list, map(list, umap_feature_list)))
        assignment_dict = dict(list(zip(all_tile_list, assignment_zip)))
        ut.save_cPkl(cluster_cache_filepath, assignment_dict)
    else:
        assignment_dict = ut.load_cPkl(cluster_cache_filepath)

    cluster_center_dict = {}
    cluster_dict = {}
    values_list = list(assignment_dict.items())
    minx, miny = np.inf, np.inf
    maxx, maxy = -np.inf, -np.inf
    for tile_id, (cluster, embedding) in values_list:
        cluster = int(cluster)
        x, y = embedding
        minx = min(minx, x)
        miny = min(miny, y)
        maxx = max(maxx, x)
        maxy = max(maxy, y)

        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        if cluster not in cluster_center_dict:
            cluster_center_dict[cluster] = [0.0, 0.0, 0.0]

        value = (tile_id, embedding)
        cluster_dict[cluster].append(value)
        cluster_center_dict[cluster][0] += x
        cluster_center_dict[cluster][1] += y
        cluster_center_dict[cluster][2] += 1

    for cluster in cluster_center_dict:
        total = cluster_center_dict[cluster][2]
        cluster_center_dict[cluster][0] = cluster_center_dict[cluster][0] / total
        cluster_center_dict[cluster][1] = cluster_center_dict[cluster][1] / total

    if reclassify_outliers:
        centers = []
        for cluster in sorted(cluster_center_dict.keys()):
            if cluster >= 0:
                center = cluster_center_dict[cluster]
                centers.append(center[:2])
        centers = np.vstack(centers)

        outliers_list = cluster_dict.pop(-1)
        tile_list = []
        embeddings = []
        for outlier in outliers_list:
            tile_id, embedding = outlier
            tile_list.append(tile_id)
            embeddings.append(embedding)
        embeddings = np.vstack(embeddings)

        distances = scipy.spatial.distance.cdist(embeddings, centers)
        clusters = np.argmin(distances, axis=1)
        clusters = list(map(int, clusters))
        zipped = list(zip(tile_list, embeddings, clusters))
        assert len(zipped) == len(embeddings)
        for tile_id, embedding, cluster in tqdm.tqdm(zipped):
            assert cluster >= -1
            assert cluster in cluster_dict
            value = (tile_id, embedding)
            previous = len(cluster_dict[cluster])
            cluster_dict[cluster].append(value)
            assignment_dict[tile_id] = (cluster, embedding)
            assert len(cluster_dict[cluster]) == previous + 1

    limits = minx, maxx, miny, maxy
    return hashstr, assignment_dict, cluster_dict, cluster_center_dict, limits


@register_ibs_method
def scout_visualize_visual_clusters(
    ibs, num_clusters=80, n_neighbors=10, examples=50, reclassify_outliers=True, **kwargs
):
    """
    for n_neighbors in range(5, 61, 5):
        for num_clusters in range(10, 51, 10):
            ibs.scout_visualize_visual_clusters(num_clusters, n_neighbors)
    """
    import random

    import cv2
    import matplotlib.pyplot as plt

    import wbia.plottool as pt

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.5
    thickness = 5

    values = ibs.scout_compute_visual_clusters(
        num_clusters=num_clusters,
        n_neighbors=n_neighbors,
        reclassify_outliers=reclassify_outliers,
        **kwargs
    )
    hashstr, assignment_dict, cluster_dict, cluster_center_dict, limits = values
    minx, maxx, miny, maxy = limits

    cluster_list = sorted(cluster_dict.keys())
    if reclassify_outliers:
        color_list_ = []
    else:
        color_list_ = [(0.2, 0.2, 0.2)]
    color_list = pt.distinct_colors(len(cluster_list) - len(color_list_), randomize=False)
    color_list = color_list_ + color_list

    fig_ = plt.figure(figsize=(30, 15), dpi=400)  # NOQA
    axes = plt.subplot(121)
    plt.title('Visualization of Embedding Space - UMAP + HDBSCAN Unsupervised Labeling')

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

    text_width_ = 120
    max_text_width = 0.0
    canvas_list = []
    for cluster, color in zip(cluster_list, color_list):
        logger.info('Processing cluster %d' % (cluster,))
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
            'draw_annots': False,
            'thumbsize': (densenet.INPUT_SIZE, densenet.INPUT_SIZE),
        }
        thumbnail_list = ibs.depc_image.get_property(
            'thumbnails', tile_id_list, 'img', config=config_
        )

        color = np.array(color[::-1], dtype=np.float32)
        color = np.around(color * 255.0).astype(np.uint8)
        vertical_color = np.zeros((densenet.INPUT_SIZE, 10, 3), dtype=np.uint8)
        vertical_color += color

        text = '%d' % (cluster,)
        text_width, text_height = cv2.getTextSize(text, font, scale, thickness)[0]
        max_text_width = max(text_width, max_text_width)
        prefix_vertical_color = np.zeros(
            (densenet.INPUT_SIZE, text_width_, 3), dtype=np.uint8
        )
        prefix_vertical_color += color

        canvas_ = np.hstack(
            [prefix_vertical_color, vertical_color] + thumbnail_list + [vertical_color]
        )
        hoffset = (text_width_ - text_width) // 2 + 5
        voffset = (densenet.INPUT_SIZE + text_height) // 2 + 5
        cv2.putText(
            canvas_,
            text,
            (hoffset, voffset),
            font,
            scale,
            (255, 255, 255),
            thickness=thickness,
        )

        horizontal_color = np.zeros((10, canvas_.shape[1], 3), dtype=np.uint8)
        horizontal_color += color
        canvas = np.vstack([horizontal_color, canvas_, horizontal_color])

        canvas_list.append(canvas)

    logger.info('Suggest use max_text_width = %d' % (max_text_width,))
    args = (
        hashstr,
        num_clusters,
        n_neighbors,
        reclassify_outliers,
    )

    canvas = np.vstack(canvas_list)
    canvas_filename = 'scout-wic-clusters-%s-%s-%s-outliers-%s-examples.png' % args
    canvas_filepath = abspath(expanduser(join('~', 'Desktop', canvas_filename)))
    cv2.imwrite(canvas_filepath, canvas)

    zipped = list(zip([True] * len(x_list), x_list, y_list, c_list, a_list, m_list))
    zipped_ = list(
        zip([False] * len(x_list_), x_list_, y_list_, c_list_, a_list_, m_list_)
    )
    random.shuffle(zipped)
    random.shuffle(zipped_)
    skip_rate = max(0.0, 1.0 - (20000 / len(zipped)))
    logger.info('Using skiprate = {:0.02f}'.format(skip_rate))
    zipped_combined = zipped + zipped_
    for flag, x, y, c, a, m in tqdm.tqdm(zipped_combined):
        if flag and random.uniform(0.0, 1.0) < skip_rate:
            continue
        plt.plot([x], [y], color=c, alpha=a, marker=m, linestyle='None')

    for cluster, color in zip(cluster_list, color_list):
        center = cluster_center_dict[cluster]
        x, y = center[:2]
        plt.plot(
            [x],
            [y],
            color=color,
            marker='*',
            linestyle='None',
            markersize=10,
            markerfacecolor=color,
            markeredgewidth=1.0,
            markeredgecolor=(1.0, 1.0, 1.0),
        )
        label = '%d' % (cluster,)
        plt.text(
            x,
            y,
            label,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=3,
        )

    axes = plt.subplot(122)

    plt.xscale('log')
    plt.ylabel('Cluster Number')
    plt.title(
        'Number of tiles by cluster\nHashed bars indicate reclassified tiles using NN'
    )

    width = 0.35
    maxx = 0
    for version in [0, 1]:
        value_list = []
        for index, (cluster, color) in enumerate(zip(cluster_list, color_list)):
            original = cluster_center_dict[cluster][2]
            used = len(cluster_dict.get(cluster, []))
            reclassified = used - original
            if version == 0:
                value = used
                left = 0
            else:
                value = reclassified
                left = used
                maxx = max(value + left, maxx)
            bar_ = plt.barh([index], [value], width, color=color, left=left)
            if version == 1:
                for temp in bar_:
                    temp.set_hatch('////')

    axes.set_autoscalex_on(False)
    axes.set_autoscaley_on(False)
    axes.set_xlim([0, int(maxx * 1.1)])
    axes.set_ylim([-1, len(cluster_list)])

    fig_filename = 'scout-wic-clusters-%s-%s-%s-outliers-%s-plot.png' % args
    fig_filepath = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_filepath, bbox_inches='tight')


@register_ibs_method
def scout_wic_train(
    ibs,
    ensembles=5,
    rounds=10,
    boost_confidence_thresh=0.2,
    boost_round_ratio=2,
    num_clusters=80,
    n_neighbors=10,
    use_clusters=False,
    hashstr=None,
    restart_config_dict=None,
    **kwargs
):
    """
    Ignore:
        >>> from wbia.control.manual_annot_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> restart_config_dict = {
        >>>     'scout-d3e8bf43-boost0': 'https://wildbookiarepository.azureedge.net/models/classifier2.scout.d3e8bf43.0.zip',
        >>>     'scout-d3e8bf43-boost1': 'https://wildbookiarepository.azureedge.net/models/classifier2.scout.d3e8bf43.1.zip',
        >>>     'scout-d3e8bf43-boost2': 'https://wildbookiarepository.azureedge.net/models/classifier2.scout.d3e8bf43.2.zip',
        >>>     'scout-d3e8bf43-boost3': 'https://wildbookiarepository.azureedge.net/models/classifier2.scout.d3e8bf43.3.zip',
        >>>     'scout-d3e8bf43-boost4': 'https://wildbookiarepository.azureedge.net/models/classifier2.scout.d3e8bf43.4.zip',
        >>>     'scout-d3e8bf43-boost5': 'https://wildbookiarepository.azureedge.net/models/classifier2.scout.d3e8bf43.5.zip',
        >>>     'scout-d3e8bf43-boost6': 'https://wildbookiarepository.azureedge.net/models/classifier2.scout.d3e8bf43.6.zip',
        >>>     'scout-d3e8bf43-boost7': 'https://wildbookiarepository.azureedge.net/models/classifier2.scout.d3e8bf43.7.zip',
        >>> }
        >>> ibs.scout_wic_train(restart_config_dict=restart_config_dict)
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
            assert (
                latest_round == round_
            ), 'Boosting rounds cannot be skipped, please include'
            latest_round += 1

            densenet.ARCHIVE_URL_DICT[restart_config_key] = restart_config_url
            latest_model_tag = restart_config_key
            config_list.append(
                {
                    'label': 'WIC %s rRound %d'
                    % (
                        hashstr,
                        round_,
                    ),
                    'classifier_algo': 'densenet',
                    'classifier_weight_filepath': restart_config_key,
                },
            )

        restart_round_num = latest_round
        ibs.scout_wic_validate(config_list)

    # Start training
    if hashstr is None:
        hashstr = ut.random_nonce()[:8]
    logger.info('Using hashstr={!r}'.format(hashstr))

    gid_all_list = ibs.get_valid_gids(is_tile=None)
    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))

    if use_clusters:
        values = ibs.scout_compute_visual_clusters(num_clusters, n_neighbors, **kwargs)
        _, assignment_dict = values

        cluster_dict = {}
        values_list = list(assignment_dict.items())
        for tile_id, (cluster, embedding) in values_list:
            if cluster not in cluster_dict:
                cluster_dict[cluster] = []
            cluster_dict[cluster].append(tile_id)
    else:
        cluster_dict = {-1: all_tile_set}
    cluster_list = sorted(cluster_dict.keys())

    train_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET'))
    )
    train_gid_set = all_tile_set & train_gid_set

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])
    negative_gid_set = set(ibs.get_imageset_gids(nid))
    negative_gid_set = negative_gid_set & train_gid_set

    num_total = len(train_gid_set)
    num_negative = len(negative_gid_set)
    num_positive = num_total - num_negative

    test_tile_list = list(negative_gid_set)

    class_weights = {
        'positive': 1.0,
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
            round_confidence_list = ibs.scout_wic_test(
                test_tile_list, model_tag=latest_model_tag
            )

        flag_list = [
            confidence >= boost_confidence_thresh for confidence in round_confidence_list
        ]
        round_hard_neg_test_tile_list = ut.compress(test_tile_list, flag_list)
        round_hard_neg_confidence_list = ut.compress(round_confidence_list, flag_list)

        weights_path_list = []
        for ensemble_num in range(ensembles):
            boost_imageset_text = 'NEGATIVE-BOOST-%s-%d-%d' % (
                hashstr,
                round_num,
                ensemble_num,
            )
            (boost_id,) = ibs.get_imageset_imgsetids_from_text([boost_imageset_text])
            ibs.unrelate_images_and_imagesets(
                gid_all_list, [boost_id] * len(gid_all_list)
            )

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
                            cluster_confidence = boost_confidence_thresh + random.uniform(
                                0.0, 0.001
                            )
                        else:
                            cluster_confidence = -1.0
                        assert cluster_tile not in ensemble_confidence_dict
                        ensemble_confidence_dict[cluster_tile] = cluster_confidence

                ensemble_confidence_list = [
                    ensemble_confidence_dict[test_tile] for test_tile in test_tile_list
                ]
            else:
                ensemble_latest_model_tag = '%s:%d' % (
                    latest_model_tag,
                    ensemble_num,
                )
                ensemble_confidence_list = ibs.scout_wic_test(
                    test_tile_list, model_tag=ensemble_latest_model_tag
                )

            flag_list = [
                confidence >= boost_confidence_thresh
                for confidence in ensemble_confidence_list
            ]
            ensemble_hard_neg_test_tile_list = ut.compress(test_tile_list, flag_list)
            ensemble_hard_neg_confidence_list = ut.compress(
                ensemble_confidence_list, flag_list
            )

            message = 'Found %d ENSEMBLE hard negatives for round %d (boost_confidence_thresh=%0.02f)'
            args = (
                len(round_hard_neg_test_tile_list),
                round_num,
                boost_confidence_thresh,
            )
            logger.info(message % args)

            message = 'Found %d MODEL hard negatives for round %d model %d (boost_confidence_thresh=%0.02f)'
            args = (
                len(ensemble_hard_neg_test_tile_list),
                round_num,
                ensemble_num,
                boost_confidence_thresh,
            )
            logger.info(message % args)

            # Combine the round's ensemble hard negatives with the specific model's hard negatives
            hard_neg_test_tile_list = (
                round_hard_neg_test_tile_list + ensemble_hard_neg_test_tile_list
            )
            hard_neg_confidence_list = (
                round_hard_neg_confidence_list + ensemble_hard_neg_confidence_list
            )
            hard_neg_test_tuple_list_ = sorted(
                zip(hard_neg_confidence_list, hard_neg_test_tile_list), reverse=True
            )

            # Remove duplicates with different confidences
            seen_set = set()
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
            logger.info(
                'Mined negatives with confidences in range [%0.04f, %0.04f] (avg %0.04f +/- %0.04f)'
                % args
            )

            # Add previous negative boosting rounds
            last_ensemble_test_tile_list = []
            for previous_round_num in range(0, round_num):
                previous_boost_imageset_text = 'NEGATIVE-BOOST-%s-%d-%d' % (
                    hashstr,
                    previous_round_num,
                    ensemble_num,
                )
                logger.info(
                    'Searching previous boosting rounds for %r: %r'
                    % (
                        boost_imageset_text,
                        previous_boost_imageset_text,
                    )
                )
                (previous_boost_id,) = ibs.get_imageset_imgsetids_from_text(
                    [previous_boost_imageset_text]
                )
                previous_ensemble_test_tile_list = ibs.get_imageset_gids(
                    previous_boost_id
                )
                last_ensemble_test_tile_list = previous_ensemble_test_tile_list[:]
                logger.info(
                    '\tFound %d images' % (len(previous_ensemble_test_tile_list),)
                )
                ensemble_test_tile_list += previous_ensemble_test_tile_list

            ensemble_test_tile_list = list(set(ensemble_test_tile_list))

            num_new = len(
                list(set(ensemble_test_tile_list) - set(last_ensemble_test_tile_list))
            )
            message = (
                'Found %d TOTAL hard negatives for round %d model %d (%d new this round)'
            )
            args = (
                len(ensemble_test_tile_list),
                round_num,
                ensemble_num,
                num_new,
            )
            logger.info(message % args)

            # Set combined image set to current pool of negatives
            ibs.set_image_imagesettext(
                ensemble_test_tile_list,
                [boost_imageset_text] * len(ensemble_test_tile_list),
            )

            args = (
                hashstr,
                round_num,
                ensemble_num,
            )
            data_path = join(ibs.get_cachedir(), 'extracted-%s-%d-%d' % args)
            output_path = join(
                ibs.get_cachedir(), 'training', 'classifier-cameratrap-%s-%d-%d' % args
            )

            # Extract training data
            extracted_path = get_cnn_classifier_cameratrap_binary_training_images_pytorch(
                ibs,
                pid,
                boost_id,
                dest_path=data_path,
                skip_rate_neg=0.0,
            )
            weights_path = densenet.train(
                extracted_path,
                output_path,
                flip=True,
                rotate=20,
                shear=20,
                class_weights=class_weights,
                sample_multiplier=1.0,
            )
            weights_path_list.append(weights_path)

        latest_model_tag, _ = ibs.scout_wic_deploy(weights_path_list, hashstr, round_num)
        config_list.append(
            {
                'label': 'WIC %s Round %d'
                % (
                    hashstr,
                    round_num,
                ),
                'classifier_algo': 'densenet',
                'classifier_weight_filepath': latest_model_tag,
            },
        )
        ibs.scout_wic_validate(config_list)

    models = densenet.ARCHIVE_URL_DICT
    logger.info(ut.repr3(models))
    return models


@register_ibs_method
def scout_wic_deploy(ibs, weights_path_list, hashstr, round_num=0, temporary=True):
    args = (
        hashstr,
        round_num,
    )
    output_name = 'classifier2.scout.%s.%d' % args
    ensemble_path = join(ibs.get_cachedir(), 'training', output_name)
    ut.ensuredir(ensemble_path)

    archive_path = '%s.zip' % (ensemble_path)
    ensemble_weights_path_list = []

    for index, weights_path in enumerate(sorted(weights_path_list)):
        assert exists(weights_path)
        ensemble_weights_path = join(ensemble_path, 'classifier.%d.weights' % (index,))
        ut.copy(weights_path, ensemble_weights_path)
        ensemble_weights_path_list.append(ensemble_weights_path)

    ensemble_weights_path_list = [ensemble_path] + ensemble_weights_path_list
    ut.archive_files(
        archive_path, ensemble_weights_path_list, overwrite=True, common_prefix=True
    )

    output_path = '/data/public/models/{}.zip'.format(output_name)
    ut.copy(archive_path, output_path)

    model_key = 'scout-{}-boost{}'.format(
        hashstr,
        round_num,
    )
    densenet.ARCHIVE_URL_DICT[
        model_key
    ] = 'https://wildbookiarepository.azureedge.net/models/{}.zip'.format(output_name)
    logger.info(ut.repr3(densenet.ARCHIVE_URL_DICT))

    return model_key, output_name


@register_ibs_method
def scout_wic_test(
    ibs, test_tile_list, classifier_algo='densenet', model_tag=None, testing=False
):
    config = {
        'classifier_algo': classifier_algo,
        'classifier_weight_filepath': model_tag,
    }
    prediction_list = ibs.depc_image.get_property(
        'classifier',
        test_tile_list,
        'class',
        config=config,
        recompute=testing,
        recompute_all=testing,
    )
    confidence_list = ibs.depc_image.get_property(
        'classifier',
        test_tile_list,
        'score',
        config=config,
        recompute=testing,
        recompute_all=testing,
    )
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence in zip(prediction_list, confidence_list)
    ]

    return confidence_list


@register_ibs_method
def scout_wic_validate(
    ibs,
    config_list,
    offset_black=0,
    target_recall_list=None,
    recompute=False,
    desired_index=None,
    fn_recovery=False,
    use_ancestors=False,
    quick=False,
    **kwargs
):
    """
    Ignore:
        >>> config_list = [
        >>>     {'label': 'WIC d3e8bf43 R0', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost0'},
        >>>     {'label': 'WIC d3e8bf43 R1', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost1'},
        >>>     {'label': 'WIC d3e8bf43 R2', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost2'},
        >>>     {'label': 'WIC d3e8bf43 R3', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost3'},
        >>>     {'label': 'WIC d3e8bf43 R4', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost4'},
        >>>     {'label': 'WIC d3e8bf43 R5', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost5'},
        >>>     {'label': 'WIC d3e8bf43 R6', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost6'},
        >>>     {'label': 'WIC d3e8bf43 R7', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost7'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list)
        >>> ibs.scout_wic_validate(config_list, fn_recovery=True)
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC d3e8bf43 R4',   'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost4'},
        >>>     {'label': 'WIC d3e8bf43 R4:0', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost4:0'},
        >>>     {'label': 'WIC d3e8bf43 R4:1', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost4:1'},
        >>>     {'label': 'WIC d3e8bf43 R4:2', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost4:2'},
        >>>     {'label': 'WIC d3e8bf43 R4:3', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost4:3'},
        >>>     {'label': 'WIC d3e8bf43 R4:4', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost4:4'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list, offset_black=1, desired_index=0, target_recall_list=[None])
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC R4',            'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost4'},
        >>>     {'label': 'LOC V0',            'classifier_algo': 'lightnet',           'classifier_weight_filepath': 'scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC PR  R4+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost4,0.708,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC ROC R0+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost0,0.271,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 80% R4+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost4,0.238,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 85% R6+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost6,0.034,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 90% R0+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost0,0.706,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 95% R0+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost0,0.209,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 98% R0+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost0,0.026,scout_v0,0.50'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None])
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC+R R6',            'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-d3e8bf43-boost6'},
        >>>     {'label': 'LOC+R V0',            'classifier_algo': 'lightnet',           'classifier_weight_filepath': 'scout_v0,0.50'},
        >>>     {'label': 'WIC+R+LOC PR  R6+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost6,0.607,scout_v0,0.50'},
        >>>     {'label': 'WIC+R+LOC ROC R2+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost2,0.027,scout_v0,0.50'},
        >>>     {'label': 'WIC+R+LOC 80% R4+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost4,0.645,scout_v0,0.50'},
        >>>     {'label': 'WIC+R+LOC 85% R2+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost2,0.347,scout_v0,0.50'},
        >>>     {'label': 'WIC+R+LOC 90% R4+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost4,0.193,scout_v0,0.50'},
        >>>     {'label': 'WIC+R+LOC 95% R6+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost6,0.025,scout_v0,0.50'},
        >>>     {'label': 'WIC+R+LOC 98% R4+V0', 'classifier_algo': 'densenet+lightnet',  'classifier_weight_filepath': 'scout-d3e8bf43-boost4,0.005,scout_v0,0.50'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list, fn_recovery=True, target_recall_list=[None])
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC R6',            'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-d3e8bf43-boost6'},
        >>>     {'label': 'LOC V0',            'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'lightnet;scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC PR  R4+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost4,0.708,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC ROC R0+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost0,0.271,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 80% R4+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost4,0.238,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 85% R6+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost6,0.034,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 90% R0+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost0,0.706,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 95% R0+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost0,0.209,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 98% R0+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost0,0.026,scout_v0,0.50'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True)
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True, quick=True)
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC R6',            'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-d3e8bf43-boost6'},
        >>>     {'label': 'LOC V0',            'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'lightnet;scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC PR  R6+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost6,0.607,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC ROC R2+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost2,0.027,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 80% R4+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost4,0.645,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 85% R2+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost2,0.347,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 90% R4+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost4,0.193,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 95% R6+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost6,0.025,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 98% R4+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost4,0.005,scout_v0,0.50'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True)
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True, quick=True)
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC+LOC 85% R2+V0',   'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost2,0.347,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 85% R2:0+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost2:0,0.347,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 85% R2:1+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost2:1,0.347,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 85% R2:2+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost2:2,0.347,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 85% R2:3+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost2:3,0.347,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 85% R2:4+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost2:4,0.347,scout_v0,0.50'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True, offset_black=1)
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True, offset_black=1, quick=True)
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC+LOC 85% R2+V0',   'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost2,0.347,scout_v0,0.50'},
        >>>     {'label': 'WIC+LOC 85% R2:3+V0', 'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet+lightnet;scout-d3e8bf43-boost2:3,0.347,scout_v0,0.50'},
        >>>     {'label': 'Scout DetectNet',    'classifier_algo': 'scout_detectnet',  'classifier_weight_filepath': 'annotations_scout_model.json'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True, quick=True)
        >>>
        >>> # config_list = [
        >>> #     {'label': 'WIC d3e8bf43 R4', 'classifier_algo': 'densenet+neighbors',     'classifier_weight_filepath': 'scout-d3e8bf43-boost4'},
        >>> # ]
        >>> # ibs.scout_wic_validate(config_list)
        >>>
        >>>
        >>> ###############################################################################################################################################################################
        >>>
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC 5fbfff26 R0', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-5fbfff26-boost0'},
        >>>     {'label': 'WIC 5fbfff26 R1', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-5fbfff26-boost1'},
        >>>     {'label': 'WIC 5fbfff26 R2', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-5fbfff26-boost2'},
        >>>     {'label': 'WIC 5fbfff26 R3', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-5fbfff26-boost3'},
        >>> ]
        >>> # ibs.scout_wic_validate(config_list)
        >>> # ibs.scout_wic_validate(config_list, target_recall_list=[None], fn_recovery=True)
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], fn_recovery=True, quick=True)
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC 5fbfff26 R3',   'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-5fbfff26-boost3'},
        >>>     {'label': 'WIC 5fbfff26 R3:0', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-5fbfff26-boost3:0'},
        >>>     {'label': 'WIC 5fbfff26 R3:1', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-5fbfff26-boost3:1'},
        >>>     {'label': 'WIC 5fbfff26 R3:2', 'classifier_algo': 'densenet',           'classifier_weight_filepath': 'scout-5fbfff26-boost3:2'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list, offset_black=1, desired_index=0, target_recall_list=[None])
        >>> ibs.scout_wic_validate(config_list, offset_black=1, desired_index=0, target_recall_list=[None], fn_recovery=True)
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC 5fbfff26 R0',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost0'},
        >>>     {'label': 'WIC 5fbfff26 R1',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost1'},
        >>>     {'label': 'WIC 5fbfff26 R2',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost2'},
        >>>     {'label': 'WIC 5fbfff26 R3',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost3'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True)
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True, quick=True)
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC 5fbfff26 R0',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost0'},
        >>>     {'label': 'WIC 5fbfff26 R0:0',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost0:0'},
        >>>     {'label': 'WIC 5fbfff26 R0:1',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost0:1'},
        >>>     {'label': 'WIC 5fbfff26 R0:2',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost0:2'},
        >>>     {'label': 'WIC 5fbfff26 R1',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost1'},
        >>>     {'label': 'WIC 5fbfff26 R1:0',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost1:0'},
        >>>     {'label': 'WIC 5fbfff26 R1:1',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost1:1'},
        >>>     {'label': 'WIC 5fbfff26 R1:2',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost1:2'},
        >>>     {'label': 'WIC 5fbfff26 R2',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost2'},
        >>>     {'label': 'WIC 5fbfff26 R2:0',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost2:0'},
        >>>     {'label': 'WIC 5fbfff26 R2:1',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost2:1'},
        >>>     {'label': 'WIC 5fbfff26 R2:2',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost2:2'},
        >>>     {'label': 'WIC 5fbfff26 R3',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost3'},
        >>>     {'label': 'WIC 5fbfff26 R3:0',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost3:0'},
        >>>     {'label': 'WIC 5fbfff26 R3:1',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost3:1'},
        >>>     {'label': 'WIC 5fbfff26 R3:2',    'classifier_algo': 'tile_aggregation',  'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost3:2'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True)
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True, quick=True)
        >>>
        >>> config_list = [
        >>>     {'label': 'WIC 5fbfff26 R3',     'classifier_algo': 'tile_aggregation',       'classifier_weight_filepath': 'densenet;scout-5fbfff26-boost3'},
        >>>     {'label': 'Scout DetectNet',    'classifier_algo': 'scout_detectnet_csv',   'classifier_weight_filepath': 'WIC_detectnet_output.csv'},
        >>>     {'label': 'Scout Faster R-CNN', 'classifier_algo': 'scout_faster_rcnn_csv', 'classifier_weight_filepath': 'WIC_fasterRCNN_output.csv'},
        >>> ]
        >>> ibs.scout_wic_validate(config_list, target_recall_list=[None], use_ancestors=True, quick=True)
    """

    def _filter_fn_func(ibs, version, values, gid_aids_mapping):
        positive_tile_set = set()
        if version == 1:
            tile_id, label, confidence, category, conf, zipped = values
            for label_, confidence_, tile_id_ in zipped:
                if label_ == category and conf <= confidence_:
                    positive_tile_set.add(tile_id_)
        else:
            tile_id, label, prediction, zipped = values
            for tile_id_, label_, prediction_ in zipped:
                if label_ == 'positive' and prediction_ == 'positive':
                    positive_tile_set.add(tile_id_)

        assert tile_id not in positive_tile_set
        positive_tile_list = list(positive_tile_set)
        positive_aids_list = ut.take(gid_aids_mapping, positive_tile_list)
        positive_aid_list = list(set(ut.flatten(positive_aids_list)))

        flag = False
        aid_list = gid_aids_mapping.get(tile_id, [])
        for aid in aid_list:
            if aid not in positive_aid_list:
                flag = True
                break
        return flag

    if quick:
        kwargs['include_grid2'] = False
        for index in range(len(config_list)):
            classifier_algo = config_list[index].get('classifier_algo', None)
            if classifier_algo in ['tile_aggregation']:
                classifier_algo = '{}_quick'.format(classifier_algo)
                config_list[index]['classifier_algo'] = classifier_algo

    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    test_gid_set = all_tile_set & test_gid_set
    test_tile_list = list(test_gid_set)

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])

    if use_ancestors:
        ancestor_gid_list = ibs.get_tile_ancestor_gids(test_tile_list)
        test_tile_list = list(set(ancestor_gid_list))
        pid, nid = ibs.get_imageset_imgsetids_from_text(
            ['POSITIVE_IMAGE', 'NEGATIVE_IMAGE']
        )

    if target_recall_list is None:
        target_recall_list = [None, 0.8, 0.85, 0.9, 0.95, 0.98]

    filter_fn_func = _filter_fn_func if fn_recovery else None
    for target_recall in target_recall_list:
        ibs.classifier_cameratrap_precision_recall_algo_display(
            pid,
            nid,
            test_gid_list=test_tile_list,
            config_list=config_list,
            offset_black=offset_black,
            target_recall=target_recall,
            force_target_recall=True,
            desired_index=desired_index,
            filter_fn_func=filter_fn_func,
        )


@register_ibs_method
def scout_wic_visualize_errors_location(
    ibs, target_species='elephant_savanna', thresh=0.024, **kwargs
):
    def _render(gid_list, flag_list, invert=False):
        if invert:
            flag_list = [not flag for flag in flag_list]
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
                canvas[ytl:ybr, xtl:xbr] += 1
                num_bboxes += 1

        max_canvas = np.max(canvas)
        canvas = canvas / max_canvas
        canvas = np.sqrt(canvas)
        canvas = np.around(canvas * 255.0).astype(np.uint8)
        return canvas, max_canvas, num_bboxes

    canvas_path = abspath(expanduser(join('~', 'Desktop')))

    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    test_gid_set = all_tile_set & test_gid_set
    test_gid_list = list(test_gid_set)

    values = ibs.scout_tile_positive_cumulative_area(
        test_gid_list, target_species=target_species
    )
    cumulative_area_list, total_area_list, flag_list = values
    gt_positive_test_gid_list = sorted(ut.compress(test_gid_list, flag_list))
    gt_negative_test_gid_list = sorted(
        set(test_gid_list) - set(gt_positive_test_gid_list)
    )

    model_tag = 'scout-d3e8bf43-boost4'
    gt_positive_confidence_list = ibs.scout_wic_test(
        gt_positive_test_gid_list, model_tag=model_tag
    )
    gt_negative_confidence_list = ibs.scout_wic_test(
        gt_negative_test_gid_list, model_tag=model_tag
    )
    gt_positive_flag_list = [
        gt_positive_confidence >= thresh
        for gt_positive_confidence in gt_positive_confidence_list
    ]
    gt_negative_flag_list = [
        gt_negative_confidence >= thresh
        for gt_negative_confidence in gt_negative_confidence_list
    ]

    canvas, max_canvas, num_bboxes = _render(
        gt_positive_test_gid_list, gt_positive_flag_list
    )
    canvas_filename = (
        'visualize_errors_location_gt_positive_pred_positive_max_%d_bboxes_%d.png'
        % (
            int(max_canvas),
            num_bboxes,
        )
    )
    canvas_filepath = join(canvas_path, canvas_filename)
    cv2.imwrite(canvas_filepath, canvas)

    canvas, max_canvas, num_bboxes = _render(
        gt_positive_test_gid_list, gt_positive_flag_list, invert=True
    )
    canvas_filename = (
        'visualize_errors_location_gt_positive_pred_negative_max_%d_bboxes_%d.png'
        % (
            int(max_canvas),
            num_bboxes,
        )
    )
    canvas_filepath = join(canvas_path, canvas_filename)
    cv2.imwrite(canvas_filepath, canvas)

    canvas, max_canvas, num_bboxes = _render(
        gt_negative_test_gid_list, gt_negative_flag_list
    )
    canvas_filename = (
        'visualize_errors_location_gt_negative_pred_positive_max_%d_bboxes_%d.png'
        % (
            int(max_canvas),
            num_bboxes,
        )
    )
    canvas_filepath = join(canvas_path, canvas_filename)
    cv2.imwrite(canvas_filepath, canvas)

    canvas, max_canvas, num_bboxes = _render(
        gt_negative_test_gid_list, gt_negative_flag_list, invert=True
    )
    canvas_filename = (
        'visualize_errors_location_gt_negative_pred_negative_max_%d_bboxes_%d.png'
        % (
            int(max_canvas),
            num_bboxes,
        )
    )
    canvas_filepath = join(canvas_path, canvas_filename)
    cv2.imwrite(canvas_filepath, canvas)


@register_ibs_method
def scout_wic_visualize_errors_annots(
    ibs,
    target_species='elephant_savanna',
    min_cumulative_percentage=0.025,
    thresh=0.024,
    errors_only=False,
    **kwargs
):
    import matplotlib.pyplot as plt

    import wbia.plottool as pt

    fig_ = plt.figure(figsize=(12, 20), dpi=400)  # NOQA

    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    test_gid_set = all_tile_set & test_gid_set
    test_gid_list = list(test_gid_set)

    values = ibs.scout_tile_positive_cumulative_area(
        test_gid_list,
        target_species=target_species,
        min_cumulative_percentage=min_cumulative_percentage,
    )
    cumulative_area_list, total_area_list, flag_list = values
    area_percentage_list = [
        cumulative_area / total_area
        for cumulative_area, total_area in zip(cumulative_area_list, total_area_list)
    ]

    model_tag = 'scout-d3e8bf43-boost4'
    confidence_list = ibs.scout_wic_test(test_gid_list, model_tag=model_tag)

    color_list = pt.distinct_colors(4, randomize=False)

    # Coverage
    logger.info('Plotting coverage')
    plt.subplot(211)

    bucket_size = 5.0
    percentage_dict = {}
    for test_gid, flag, percentage, confidence in zip(
        test_gid_list, flag_list, area_percentage_list, confidence_list
    ):
        if percentage < min_cumulative_percentage:
            bucket = -1
        else:
            bucket = int((percentage * 100.0) / bucket_size)

        if bucket not in percentage_dict:
            percentage_dict[bucket] = [0, 0, 0, 0]

        flag_ = confidence >= thresh
        if flag:
            # GT Positive
            if flag_:
                # Pred Positive
                if not errors_only:
                    percentage_dict[bucket][0] += 1
            else:
                # Pred Negative
                percentage_dict[bucket][1] += 1
        else:
            # GT Negative
            if flag_:
                # Pred Positive
                percentage_dict[bucket][2] += 1
            else:
                # Pred Negative
                if not errors_only:
                    percentage_dict[bucket][3] += 1

    num_tn = percentage_dict[-1][3]
    percentage_dict[-1][3] = 0

    include_negatives = True

    width = 0.35
    percentage_list = sorted(percentage_dict.keys())
    index_list = np.arange(len(percentage_list))

    if not include_negatives:
        index_list = index_list[1:]

    bottom = None
    bar_list = []
    for index, color in enumerate(color_list):
        value_list = []
        for percentage in percentage_list:
            if not include_negatives and percentage < 0:
                continue
            value = percentage_dict[percentage][index]
            value_list.append(value)
        value_list = np.array(value_list)
        logger.info(value_list)
        if bottom is None:
            bottom = np.zeros(value_list.shape, dtype=value_list.dtype)
        bar_ = plt.bar(index_list, value_list, width, color=color, bottom=bottom)
        bar_list.append(bar_)
        bottom += value_list

    label_list = ['TP', 'FN', 'FP', 'TN']
    plt.legend(bar_list, label_list)

    plt.ylabel('Number of Tiles')
    plt.yscale('log')
    if errors_only:
        plt.title('WIC Performance by Area of Coverage (Errors only)')
    else:
        plt.title('WIC Performance by Area of Coverage\nGT Neg TN - %d' % (num_tn,))

    tick_list = ['[0, 2.5)', '[2.5, 5)']
    for percentage in percentage_list:
        if percentage <= 0:
            continue
        bucket_min = int(bucket_size * percentage)
        bucket_max = int(bucket_size * (percentage + 1))
        tick = '[%d, %d)' % (
            bucket_min,
            bucket_max,
        )
        tick_list.append(tick)

    if not include_negatives:
        tick_list = tick_list[1:]

    plt.xticks(index_list, tick_list)

    # Number of annotations
    logger.info('Plotting num annotations')
    plt.subplot(212)

    aids_list = ibs.get_image_aids(test_gid_list)

    percentage_dict = {}
    for test_gid, flag, confidence, aid_list in zip(
        test_gid_list, flag_list, confidence_list, aids_list
    ):
        aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
        bucket = len(aid_list)

        if bucket not in percentage_dict:
            percentage_dict[bucket] = [0, 0, 0, 0]

        flag_ = confidence >= thresh
        if flag:
            # GT Positive
            if flag_:
                # Pred Positive
                if not errors_only:
                    percentage_dict[bucket][0] += 1
            else:
                # Pred Negative
                percentage_dict[bucket][1] += 1
        else:
            # GT Negative
            if flag_:
                # Pred Positive
                percentage_dict[bucket][2] += 1
            else:
                # Pred Negative
                if not errors_only:
                    percentage_dict[bucket][3] += 1

    num_tn = percentage_dict[0][3]
    percentage_dict[0][3] = 0

    include_negatives = True

    width = 0.35
    percentage_list = sorted(percentage_dict.keys())
    index_list = np.arange(len(percentage_list))

    if not include_negatives:
        index_list = index_list[1:]

    bottom = None
    bar_list = []
    for index, color in enumerate(color_list):
        value_list = []
        for percentage in percentage_list:
            if not include_negatives and percentage < 0:
                continue
            value = percentage_dict[percentage][index]
            value_list.append(value)
        value_list = np.array(value_list)
        logger.info(value_list)
        if bottom is None:
            bottom = np.zeros(value_list.shape, dtype=value_list.dtype)
        bar_ = plt.bar(index_list, value_list, width, color=color, bottom=bottom)
        bar_list.append(bar_)
        bottom += value_list

    label_list = ['TP', 'FN', 'FP', 'TN']
    plt.legend(bar_list, label_list)

    plt.ylabel('Number of Tiles')
    plt.yscale('log')
    if errors_only:
        plt.title('WIC Performance by Number of Annotations (Errors only)')
    else:
        plt.title('WIC Performance by Number of Annotations\nGT Neg TN - %d' % (num_tn,))

    tick_list = []
    for percentage in percentage_list:
        tick = '%d' % (percentage,)
        tick_list.append(tick)

    if not include_negatives:
        tick_list = tick_list[1:]

    plt.xticks(index_list, tick_list)

    if errors_only:
        fig_filename = 'scout-wic-errors-annots-plot-errors.png'
    else:
        fig_filename = 'scout-wic-errors-annots-plot.png'

    fig_filepath = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_filepath, bbox_inches='tight')


@register_ibs_method
def scout_wic_visualize_errors_clusters(
    ibs, target_species='elephant_savanna', thresh=0.024, errors_only=False, **kwargs
):
    import matplotlib.pyplot as plt

    import wbia.plottool as pt

    fig_ = plt.figure(figsize=(40, 12), dpi=400)  # NOQA

    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    test_gid_set = all_tile_set & test_gid_set
    test_gid_list = list(test_gid_set)

    values = ibs.scout_tile_positive_cumulative_area(
        test_gid_list, target_species=target_species
    )
    cumulative_area_list, total_area_list, flag_list = values

    model_tag = 'scout-d3e8bf43-boost4'
    confidence_list = ibs.scout_wic_test(test_gid_list, model_tag=model_tag)

    values = ibs.scout_compute_visual_clusters(80, 10, **kwargs)
    hashstr, assignment_dict, cluster_dict, cluster_center_dict, limits = values

    color_list = pt.distinct_colors(4, randomize=False)

    # Coverage
    logger.info('Plotting clusters')
    plt.subplot(111)

    percentage_dict = {}
    for test_gid, flag, confidence in zip(test_gid_list, flag_list, confidence_list):
        cluster, embedding = assignment_dict[test_gid]
        bucket = int(cluster)
        if bucket not in percentage_dict:
            percentage_dict[bucket] = [0, 0, 0, 0]

        flag_ = confidence >= thresh
        if flag:
            # GT Positive
            if flag_:
                # Pred Positive
                if not errors_only:
                    percentage_dict[bucket][0] += 1
            else:
                # Pred Negative
                percentage_dict[bucket][1] += 1
        else:
            # GT Negative
            if flag_:
                # Pred Positive
                percentage_dict[bucket][2] += 1
            else:
                # Pred Negative
                if not errors_only:
                    percentage_dict[bucket][3] += 1

    width = 0.35
    percentage_list = sorted(percentage_dict.keys())
    index_list = np.arange(len(percentage_list))

    bottom = None
    bar_list = []
    for index, color in enumerate(color_list):
        value_list = []
        for percentage in percentage_list:
            value = percentage_dict[percentage][index]
            value_list.append(value)
        value_list = np.array(value_list)
        logger.info(value_list)
        if bottom is None:
            bottom = np.zeros(value_list.shape, dtype=value_list.dtype)
        bar_ = plt.bar(index_list, value_list, width, color=color, bottom=bottom)
        bar_list.append(bar_)
        bottom += value_list

    label_list = ['TP', 'FN', 'FP', 'TN']
    plt.legend(bar_list, label_list)

    plt.ylabel('Number of Tiles')
    if errors_only:
        plt.title('WIC Performance by Visual Cluster (Errors only)')
    else:
        plt.yscale('log')
        plt.title('WIC Performance by Visual Cluster')
    tick_list = []
    for percentage in percentage_list:
        tick = '%d' % (percentage,)
        tick_list.append(tick)
    plt.xticks(index_list, tick_list)

    if errors_only:
        fig_filename = 'scout-wic-errors-clusters-plot-errors.png'
    else:
        fig_filename = 'scout-wic-errors-clusters-plot.png'
    fig_filepath = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_filepath, bbox_inches='tight')


@register_ibs_method
def scout_background_train(ibs, target_species='elephant_savanna'):
    import numpy as np
    from wbia_cnn.ingest_wbia import get_background_training_patches2
    from wbia_cnn.models.background import train_background
    from wbia_cnn.process import numpy_processed_directory2
    from wbia_cnn.utils import save_model

    (pid,) = ibs.get_imageset_imgsetids_from_text(['POSITIVE'])
    train_tile_set = set(ibs.get_imageset_gids(pid))
    ancestor_gid_list = ibs.get_tile_ancestor_gids(train_tile_set)
    train_gid_set = list(set(ancestor_gid_list))

    aid_list = ut.flatten(ibs.get_image_aids(train_gid_set))
    bbox_list = ibs.get_annot_bboxes(aid_list)
    w_list = ut.take_column(bbox_list, 2)
    annot_size = int(np.around(np.mean(w_list)))

    data_path = join(ibs.get_cachedir(), 'extracted')
    output_path = join(ibs.get_cachedir(), 'training', 'background')

    extracted_path = get_background_training_patches2(
        ibs,
        target_species,
        data_path,
        patch_size=50,
        annot_size=annot_size,
        patch_size_min=0.9,
        patch_size_max=1.1,
        patches_per_annotation=10,
        train_gid_set=train_gid_set,
        visualize=True,
        inside_boundary=False,
        purge=True,
        supercharge_negative_multiplier=10.0,
    )

    # rm -rf /data/wbia/ELPH_Scout/_ibsdb/_wbia_cache/training/background/

    id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
    model_path = train_background(output_path, X_file, y_file)
    model_state = ut.load_cPkl(model_path)
    assert 'species' not in model_state
    model_state['species'] = target_species
    save_model(model_state, model_path)

    return model_path


@register_ibs_method
def scout_background_deploy(ibs, model_path):
    ut.copy(model_path, '/data/public/models/background.scout.v0.pkl')
    return model_path


@register_ibs_method
def scout_background_compute(
    ibs, tile_rowid_list, smooth_thresh=20, smooth_ksize=20, model_tag='scout'
):
    """Computes tile probability masks."""
    from wbia.core_annots import postprocess_mask

    tilemask_dir = join(ibs.get_cachedir(), 'tilemasks')
    ut.ensuredir(tilemask_dir)

    # dont use extrmargin here (for now)
    for chunk in ut.ichunks(tile_rowid_list, 256):
        output_path_list = [
            join(
                tilemask_dir,
                'tilemask_tile_id_%d_model_%s.png'
                % (
                    tile_id,
                    model_tag,
                ),
            )
            for tile_id in chunk
        ]
        dirty_list = [not exists(output_path) for output_path in output_path_list]
        if len(dirty_list) > 0:
            chunk_ = ut.compress(chunk, dirty_list)
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
def scout_background_validate(ibs, output_path=None, model_tag='scout', **kwargs):
    if output_path is None:
        output_path = join(ibs.get_cachedir(), 'tilemasks_combined')
        ut.ensuredir(output_path)

    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
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

    masks = ibs.scout_background_compute(test_tile_list, model_tag=model_tag)
    masks = list(masks)

    images = ibs.get_images(test_tile_list)
    for test_tile, image, mask in zip(test_tile_list, images, masks):
        output_filename = 'tilemask_combined_tile_id_%d_model_%s.png' % (
            test_tile,
            model_tag,
        )
        output_filepath = join(output_path, output_filename)

        combined = np.around(
            image.astype(np.float32) * mask.astype(np.float32) / 255.0
        ).astype(np.uint8)
        canvas = np.hstack((image, mask, combined))

        cv2.imwrite(output_filepath, canvas)


@register_ibs_method
def scout_localizer_train(
    ibs, target_species='elephant_savanna', ratio=2.0, config=None, **kwargs
):
    """
    config = {'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.40, 'sensitivity': 0.5675}
    ibs.scout_localizer_train(config=config)

    ###

    from wbia.other.detectfuncs import localizer_parse_pred
    target_species='elephant_savanna'
    ratio=2.0
    kwargs = {}

    config = {'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.40, 'sensitivity': 0.5675}

    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    train_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET')))
    train_gid_set = all_tile_set & train_gid_set

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])
    positive_gid_set = set(ibs.get_imageset_gids(pid))
    negative_gid_set = set(ibs.get_imageset_gids(nid))
    positive_gid_set = positive_gid_set & train_gid_set
    negative_gid_set = negative_gid_set & train_gid_set
    negative_gid_list = list(negative_gid_set)

    num = len(negative_gid_list) // 2

    negative_gid_list = negative_gid_list[:num]
    negative_gid_list = negative_gid_list[num:]
    negative_pred_list = localizer_parse_pred(  ibs, test_gid_list=list(negative_gid_list), **config)


    """
    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    train_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET'))
    )
    train_gid_set = all_tile_set & train_gid_set

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])
    positive_gid_set = set(ibs.get_imageset_gids(pid))
    negative_gid_set = set(ibs.get_imageset_gids(nid))
    positive_gid_set = positive_gid_set & train_gid_set
    negative_gid_set = negative_gid_set & train_gid_set
    negative_gid_list = list(negative_gid_set)

    # model_tag = 'scout-d3e8bf43-boost4'
    model_tag = 'scout-5fbfff26-boost3'
    confidence_list = ibs.scout_wic_test(negative_gid_list, model_tag=model_tag)
    zipped = sorted(list(zip(confidence_list, negative_gid_list)), reverse=True)

    num_positive = len(positive_gid_set)
    num_negative = min(len(negative_gid_set), int(ratio * num_positive))
    zipped = zipped[:num_negative]
    negative_gid_set = set(ut.take_column(zipped, 1))

    if config is not None:
        from wbia.other.detectfuncs import localizer_parse_pred

        negative_pred_list = localizer_parse_pred(
            ibs, test_gid_list=list(negative_gid_list), **config
        )

        value_list = []
        for negative_uuid in negative_pred_list:
            negative_pred = negative_pred_list[negative_uuid]
            area = 0
            for pred in negative_pred:
                w, h = pred['width'], pred['height']
                area += w * h
            value = (
                len(negative_pred),
                area,
                negative_uuid,
            )
            value_list.append(value)

        negative_zipped = sorted(value_list, reverse=True)
        num_negative = min(len(negative_zipped), int(ratio * num_positive))
        negative_zipped = negative_zipped[:num_negative]
        negative_uuid_list = set(ut.take_column(negative_zipped, 2))
        negative_gid_list_ = ibs.get_image_gids_from_uuid(negative_uuid_list)
        assert None not in negative_gid_list_
        negative_gid_set = negative_gid_set | set(negative_gid_list_)

    logger.info(
        'Using %d positives, %d negatives'
        % (
            len(positive_gid_set),
            len(negative_gid_set),
        )
    )
    tid_list = sorted(list(positive_gid_set | negative_gid_set))

    species_list = [target_species]
    values = ibs.localizer_lightnet_train(
        species_list, gid_list=tid_list, cuda_device='0,1', target_size=256, **kwargs
    )
    model_weight_filepath, model_config_filepath = values

    return model_weight_filepath, model_config_filepath


def _scout_localizer_ignore_filter_func(
    ibs, annot, margin, min_bbox_coverage, *args, **kwargs
):
    from wbia.other.detectfuncs import general_intersection_over_union

    margin = float(margin)
    gid = annot.get('gid', None)
    assert gid is not None
    w, h = ibs.get_image_sizes(gid)
    margin_percent_w = margin / w
    margin_percent_h = margin / h
    xtl = margin_percent_w
    ytl = margin_percent_h
    xbr = 1.0 - margin_percent_w
    ybr = 1.0 - margin_percent_h
    width = xbr - xtl
    height = ybr - ytl
    center = {
        'xtl': xtl,
        'ytl': ytl,
        'xbr': xbr,
        'ybr': ybr,
        'width': width,
        'height': height,
    }
    intersection, union = general_intersection_over_union(
        annot, center, return_components=True
    )
    area = annot['width'] * annot['height']
    if area <= 0:
        return True
    overlap = intersection / area
    flag = overlap < min_bbox_coverage
    return flag


@register_ibs_method
def scout_localizer_validate(
    ibs,
    target_species='elephant_savanna',
    thresh=0.024,
    margin=32,
    min_bbox_coverage=0.5,
    offset_color=0,
    **kwargs
):
    species_set = {target_species}
    template_v0 = (
        [
            # {'label': 'Elephant V0 NMS 0%',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.00, 'species_set' : species_set},
            # {'label': 'Elephant V0 NMS 10%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.10, 'species_set' : species_set},
            # {'label': 'Elephant V0 NMS 20%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
            # {'label': 'Elephant V0 NMS 30%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.30, 'species_set' : species_set},
            # {'label': 'Elephant V0 NMS 40%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
            # {'label': 'Elephant V0 NMS 50%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.50, 'species_set' : species_set},
            # {'label': 'Elephant V0 NMS 60%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            # {'label': 'Elephant V0 NMS 70%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': 'Elephant V0 NMS 80%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': 'Elephant V0 NMS 90%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': 'Elephant V0 NMS 100%', 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 1.00, 'species_set' : species_set},
            #############################################
            # {'label': 'Elephant V1 NMS 0%',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.00, 'species_set' : species_set},
            # {'label': 'Elephant V1 NMS 10%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.10, 'species_set' : species_set},
            # {'label': 'Elephant V1 NMS 20%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
            # {'label': 'Elephant V1 NMS 30%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.30, 'species_set' : species_set},
            # {'label': 'Elephant V1 NMS 40%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
            # {'label': 'Elephant V1 NMS 50%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.50, 'species_set' : species_set},
            # {'label': 'Elephant V1 NMS 60%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            # {'label': 'Elephant V1 NMS 70%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': 'Elephant V1 NMS 80%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': 'Elephant V1 NMS 90%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': 'Elephant V1 NMS 100%', 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 1.00, 'species_set' : species_set},
            #############################################
            # {'label': 'd3e8bf43 V0 NMS 60%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            # {'label': '5fbfff26 V0 NMS 0%',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.00, 'species_set' : species_set},
            # {'label': '5fbfff26 V0 NMS 10%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.10, 'species_set' : species_set},
            # {'label': '5fbfff26 V0 NMS 20%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
            # {'label': '5fbfff26 V0 NMS 30%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.30, 'species_set' : species_set},
            # {'label': '5fbfff26 V0 NMS 40%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
            # {'label': '5fbfff26 V0 NMS 50%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.50, 'species_set' : species_set},
            # {'label': '5fbfff26 V0 NMS 60%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            # {'label': '5fbfff26 V0 NMS 70%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': '5fbfff26 V0 NMS 80%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 V0 NMS 90%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 V0 NMS 100%', 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v0', 'weight_filepath' : 'scout_5fbfff26_v0', 'nms': True, 'nms_thresh': 1.00, 'species_set' : species_set},
            # ibs.scout_localizer_validate(offset_color=1)
            #############################################
            # {'label': 'd3e8bf43 V1 NMS 60%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            # {'label': '5fbfff26 V1 NMS 0%',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v1', 'weight_filepath' : 'scout_5fbfff26_v1', 'nms': True, 'nms_thresh': 0.00, 'species_set' : species_set},
            # {'label': '5fbfff26 V1 NMS 10%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v1', 'weight_filepath' : 'scout_5fbfff26_v1', 'nms': True, 'nms_thresh': 0.10, 'species_set' : species_set},
            # {'label': '5fbfff26 V1 NMS 20%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v1', 'weight_filepath' : 'scout_5fbfff26_v1', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
            # {'label': '5fbfff26 V1 NMS 30%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v1', 'weight_filepath' : 'scout_5fbfff26_v1', 'nms': True, 'nms_thresh': 0.30, 'species_set' : species_set},
            # {'label': '5fbfff26 V1 NMS 40%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v1', 'weight_filepath' : 'scout_5fbfff26_v1', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
            # {'label': '5fbfff26 V1 NMS 50%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v1', 'weight_filepath' : 'scout_5fbfff26_v1', 'nms': True, 'nms_thresh': 0.50, 'species_set' : species_set},
            # {'label': '5fbfff26 V1 NMS 60%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v1', 'weight_filepath' : 'scout_5fbfff26_v1', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            # {'label': '5fbfff26 V1 NMS 70%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v1', 'weight_filepath' : 'scout_5fbfff26_v1', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': '5fbfff26 V1 NMS 80%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v1', 'weight_filepath' : 'scout_5fbfff26_v1', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 V1 NMS 90%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v1', 'weight_filepath' : 'scout_5fbfff26_v1', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 V1 NMS 100%', 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_5fbfff26_v1', 'weight_filepath' : 'scout_5fbfff26_v1', 'nms': True, 'nms_thresh': 1.00, 'species_set' : species_set},
            # ibs.scout_localizer_validate(offset_color=1)
            #############################################
            {
                'label': 'd3e8bf43 V0 NMS 60%',
                'grid': False,
                'algo': 'lightnet',
                'config_filepath': 'scout_v0',
                'weight_filepath': 'scout_v0',
                'nms': True,
                'nms_thresh': 0.60,
                'species_set': species_set,
            },
            {
                'label': 'd3e8bf43 V1 NMS 60%',
                'grid': False,
                'algo': 'lightnet',
                'config_filepath': 'scout_v1',
                'weight_filepath': 'scout_v1',
                'nms': True,
                'nms_thresh': 0.60,
                'species_set': species_set,
            },
            {
                'label': '5fbfff26 V0 NMS 50%',
                'grid': False,
                'algo': 'lightnet',
                'config_filepath': 'scout_5fbfff26_v0',
                'weight_filepath': 'scout_5fbfff26_v0',
                'nms': True,
                'nms_thresh': 0.50,
                'species_set': species_set,
            },
            {
                'label': '5fbfff26 V1 NMS 50%',
                'grid': False,
                'algo': 'lightnet',
                'config_filepath': 'scout_5fbfff26_v1',
                'weight_filepath': 'scout_5fbfff26_v1',
                'nms': True,
                'nms_thresh': 0.50,
                'species_set': species_set,
            },
        ],
        {},
    )

    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    test_gid_set = all_tile_set & test_gid_set
    all_test_gid_list = list(test_gid_set)

    values = ibs.scout_tile_positive_cumulative_area(
        all_test_gid_list, target_species=target_species
    )
    cumulative_area_list, total_area_list, flag_list = values
    gt_positive_test_gid_list = sorted(ut.compress(all_test_gid_list, flag_list))

    ignore_filter_func_ = partial(
        _scout_localizer_ignore_filter_func,
        margin=margin,
        min_bbox_coverage=min_bbox_coverage,
    )

    # All Positive Tiles (All)
    config_dict = {
        'scout-gt-positive-all-v0-v1': template_v0,
    }
    ibs.localizer_precision_recall(
        config_dict=config_dict,
        test_gid_list=gt_positive_test_gid_list,
        overwrite_config_keys=True,
        offset_color=offset_color,
    )

    # All Positive Tiles (Margin)
    config_dict = {
        'scout-gt-positive-margin-{}-v0-v1'.format(margin): template_v0,
    }
    ibs.localizer_precision_recall(
        config_dict=config_dict,
        test_gid_list=gt_positive_test_gid_list,
        overwrite_config_keys=True,
        ignore_filter_func=ignore_filter_func_,
        offset_color=offset_color,
    )


@register_ibs_method
def scout_localizer_image_validate(
    ibs, target_species='elephant_savanna', quick=False, offset_color=0, **kwargs
):

    algo = 'tile_aggregation_quick' if quick else 'tile_aggregation'

    species_set = {target_species}
    template_v0 = (
        [
            # {'label': '5fbfff26 R3+V0 NMS 0%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.00, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 NMS 10%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.10, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 NMS 20%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 NMS 30%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.30, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 NMS 40%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 NMS 50%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.50, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 NMS 60%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 NMS 70%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 NMS 80%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 NMS 90%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 NMS 100%', 'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 1.00, 'species_set' : species_set},
            ########################################################################################################
            # {'label': '5fbfff26 R3+V0 703 NMS 90%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 750 NMS 70%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.750,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 750 NMS 80%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.750,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 750 NMS 90%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.750,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 650 NMS 70%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.650,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 650 NMS 80%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.650,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 650 NMS 90%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.650,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 600 NMS 70%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.600,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 600 NMS 80%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.600,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 600 NMS 90%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.600,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 550 NMS 70%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.550,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 550 NMS 80%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.550,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 550 NMS 90%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.550,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 500 NMS 70%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.500,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 500 NMS 80%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.500,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 500 NMS 90%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.500,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 450 NMS 70%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.450,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 450 NMS 80%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.450,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 450 NMS 90%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.450,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 400 NMS 70%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 400 NMS 80%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 400 NMS 90%',  'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # ibs.scout_localizer_image_validate(offset_color=1)
            ########################################################################################################
            # {'label': '5fbfff26 R3+V0 703+0%   NMS 90%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+20%  NMS 20%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.2', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+40%  NMS 20%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+60%  NMS 20%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.6', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+80%  NMS 20%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.8', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+20%  NMS 40%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.2', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+40%  NMS 40%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+60%  NMS 40%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.6', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+80%  NMS 40%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.8', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+20%  NMS 60%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.2', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+40%  NMS 60%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+60%  NMS 60%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.6', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+80%  NMS 60%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.8', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+20%  NMS 80%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.2', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+40%  NMS 80%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+60%  NMS 80%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.6', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+80%  NMS 80%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.8', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+20%  NMS 90%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.2', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+40%  NMS 90%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+60%  NMS 90%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.6', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+80%  NMS 90%',   'grid' : False, 'algo': algo, 'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.8', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # ibs.scout_localizer_image_validate(offset_color=1)
            ########################################################################################################
            # {'label': '5fbfff26 R3+V0 703+0%  NMS 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 400+40% NMS 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': 'Scout DetectNet',                'grid' : False, 'algo': 'scout_detectnet_json',   'config_filepath' : 'variant1', 'weight_filepath' : 'annotations_detectnet_COCO.json',   'nms': False, 'species_set' : species_set},
            # {'label': 'Scout Faster R-CNN',             'grid' : False, 'algo': 'scout_faster_rcnn_json', 'config_filepath' : 'variant1', 'weight_filepath' : 'annotations_faster_rcnn_COCO.json', 'nms': False, 'species_set' : species_set},
            # ibs.scout_localizer_image_validate()
            ########################################################################################################
            # {'label': '5fbfff26 R3+V0 796+0% NMS 90%',   'grid' : False, 'algo': algo,                      'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 400+40% NMS 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant1', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': 'Scout DetectNet',                'grid' : False, 'algo': 'scout_detectnet_json',   'config_filepath' : 'variant1', 'weight_filepath' : 'annotations_detectnet_COCO.json',   'nms': False, 'species_set' : species_set},
            # {'label': 'Scout Faster R-CNN',             'grid' : False, 'algo': 'scout_faster_rcnn_json', 'config_filepath' : 'variant1', 'weight_filepath' : 'annotations_faster_rcnn_COCO.json', 'nms': False, 'species_set' : species_set},
            # ibs.scout_localizer_image_validate(quick=True)
            ########################################################################################################
            # {'label': '5fbfff26 R3+V0 703+0%  Var1    NMS 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant1',    'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+0%  Var2-32 NMS 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant2-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+0%  Var3-32 NMS 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+0%  Var4-32 NMS 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant4-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+0%  Var2-64 NMS 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant2-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+0%  Var3-64 NMS 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbfff26 R3+V0 703+0%  Var4-64 NMS 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant4-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.703,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # ibs.scout_localizer_image_validate()
            ########################################################################################################
            # {'label': '5fbf R3+V0 400+40% V1    80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant1',    'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V2-32 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant2-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V3-32 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V4-32 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant4-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V2-64 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant2-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V3-64 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V4-64 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant4-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # ibs.scout_localizer_image_validate()
            ########################################################################################################
            # {'label': '5fbf R3+V0 796+0% Var1    90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant1',    'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.796,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 796+0% Var2-32 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant2-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.796,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 796+0% Var3-32 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.796,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 796+0% Var4-32 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant4-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.796,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 796+0% Var2-64 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant2-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.796,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 796+0% Var3-64 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.796,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 796+0% Var4-64 90%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant4-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.796,scout_5fbfff26_v0,0.0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
            # ibs.scout_localizer_image_validate(quick=True)
            ########################################################################################################
            # {'label': '5fbf R3+V0 400+40% V1    80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant1',    'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V2-32 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant2-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V3-32 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V4-32 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant4-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V2-64 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant2-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V3-64 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3+V0 400+40% V4-64 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant4-64', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # ibs.scout_localizer_image_validate(quick=True)
            ########################################################################################################
            # {'label': '5fbf R3  +V0 400+40% V3-32 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3:0+V0 400+40% V3-32 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3:0,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3:1+V0 400+40% V3-32 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3:1,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': '5fbf R3:2+V0 400+40% V3-32 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3:2,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # ibs.scout_localizer_image_validate(quick=True, offset_color=1)
            ########################################################################################################
            # {'label': '5fbf R3:1+V0 400+40% V3-32 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3:1,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # {'label': 'Scout DetectNet',                'grid' : False, 'algo': 'scout_detectnet_json',   'config_filepath' : 'variant1', 'weight_filepath' : 'annotations_detectnet_COCO.json',   'nms': False, 'species_set' : species_set},
            # {'label': 'Scout Faster R-CNN',             'grid' : False, 'algo': 'scout_faster_rcnn_json', 'config_filepath' : 'variant1', 'weight_filepath' : 'annotations_faster_rcnn_COCO.json', 'nms': False, 'species_set' : species_set},
            # ibs.scout_localizer_image_validate(quick=True)
            ########################################################################################################
            # {'label': '5fbf R3+V0 400+40% V3-32 80%',     'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
            # # {'label': '5fbf R3+V0 400+40% V3-32S 80%',     'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set, 'squared': True},
            # {'label': 'Scout Faster R-CNN',               'grid' : False, 'algo': 'scout_faster_rcnn_json', 'config_filepath' : 'variant1',    'weight_filepath' : 'annotations_faster_rcnn_COCO.json', 'nms': False, 'species_set' : species_set},
            # {'label': 'Scout DetectNet',                  'grid' : False, 'algo': 'scout_detectnet_json',   'config_filepath' : 'variant1',    'weight_filepath' : 'annotations_detectnet_COCO.json',   'nms': False, 'species_set' : species_set},
            # ibs.scout_localizer_image_validate()
            ########################################################################################################
            {
                'label': '5fbf R3:1+V0 400+40% V3-32  80%',
                'grid': False,
                'algo': algo,
                'config_filepath': 'variant3-32',
                'weight_filepath': 'densenet+lightnet;scout-5fbfff26-boost3:1,0.400,scout_5fbfff26_v0,0.4',
                'nms': True,
                'nms_thresh': 0.80,
                'species_set': species_set,
            },
            # {'label': '5fbf R3:1+V0 400+40% V3-32S 80%',  'grid' : False, 'algo': algo,                      'config_filepath' : 'variant3-32', 'weight_filepath' : 'densenet+lightnet;scout-5fbfff26-boost3:1,0.400,scout_5fbfff26_v0,0.4', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set, 'squared': True},
            {
                'label': 'Scout Faster R-CNN',
                'grid': False,
                'algo': 'scout_faster_rcnn_json',
                'config_filepath': 'variant1',
                'weight_filepath': 'annotations_faster_rcnn_COCO.json',
                'nms': False,
                'species_set': species_set,
            },
            {
                'label': 'Scout DetectNet',
                'grid': False,
                'algo': 'scout_detectnet_json',
                'config_filepath': 'variant1',
                'weight_filepath': 'annotations_detectnet_COCO.json',
                'nms': False,
                'species_set': species_set,
            },
            # ibs.scout_localizer_image_validate(quick=True)
        ],
        {},
    )

    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    test_gid_set = all_tile_set & test_gid_set
    test_tile_list = list(test_gid_set)

    ancestor_gid_list = ibs.get_tile_ancestor_gids(test_tile_list)
    test_gid_list = list(set(ancestor_gid_list))

    key = 'scout-localizer-image-{}-comparison-final'.format(algo)
    config_dict = {
        key: template_v0,
    }
    ibs.localizer_precision_recall(
        config_dict=config_dict,
        test_gid_list=test_gid_list,
        overwrite_config_keys=True,
        offset_color=offset_color,
        min_overlap=0.2,
        target_recall=0.9,
    )

    if False:
        detection_config = ibs.scout_detect_config(quick=False)
        test_gid_list = [
            1,
            2,
            5,
            6,
            7,
            14,
            20,
            23,
            24,
            27,
            29,
            33,
            34,
            35,
            41,
            50,
            53,
            56,
            61,
            67,
        ]
        ibs.visualize_predictions(detection_config, gid_list=test_gid_list, t_width=2000)
        ibs.visualize_ground_truth(detection_config, gid_list=test_gid_list, t_width=2000)


# @register_ibs_method
# def scout_localizer_validate(ibs, target_species='elephant_savanna',
#                               thresh=0.024, margin=32, min_bbox_coverage=0.5, **kwargs):
#     species_set = set([target_species])
#     template_v0 = (
#         [
#             {'label': 'Elephant V0 NMS 0%',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.00, 'species_set' : species_set},
#             {'label': 'Elephant V0 NMS 10%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.10, 'species_set' : species_set},
#             {'label': 'Elephant V0 NMS 20%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
#             {'label': 'Elephant V0 NMS 30%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.30, 'species_set' : species_set},
#             {'label': 'Elephant V0 NMS 40%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
#             {'label': 'Elephant V0 NMS 50%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.50, 'species_set' : species_set},
#             {'label': 'Elephant V0 NMS 60%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
#             {'label': 'Elephant V0 NMS 70%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
#             {'label': 'Elephant V0 NMS 80%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
#             {'label': 'Elephant V0 NMS 90%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
#             {'label': 'Elephant V0 NMS 100%', 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 1.00, 'species_set' : species_set},
#         ],
#         {},
#     )
#     template_v1 = (
#         [
#             {'label': 'Elephant V1 NMS 0%',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.00, 'species_set' : species_set},
#             {'label': 'Elephant V1 NMS 10%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.10, 'species_set' : species_set},
#             {'label': 'Elephant V1 NMS 20%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.20, 'species_set' : species_set},
#             {'label': 'Elephant V1 NMS 30%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.30, 'species_set' : species_set},
#             {'label': 'Elephant V1 NMS 40%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.40, 'species_set' : species_set},
#             {'label': 'Elephant V1 NMS 50%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.50, 'species_set' : species_set},
#             {'label': 'Elephant V1 NMS 60%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.60, 'species_set' : species_set},
#             {'label': 'Elephant V1 NMS 70%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.70, 'species_set' : species_set},
#             {'label': 'Elephant V1 NMS 80%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.80, 'species_set' : species_set},
#             {'label': 'Elephant V1 NMS 90%',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 0.90, 'species_set' : species_set},
#             {'label': 'Elephant V1 NMS 100%', 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v1', 'weight_filepath' : 'scout_v1', 'nms': True, 'nms_thresh': 1.00, 'species_set' : species_set},
#         ],
#         {},
#     )

#     all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
#     test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
#     test_gid_set = all_tile_set & test_gid_set
#     all_test_gid_list = list(test_gid_set)

#     values = ibs.scout_tile_positive_cumulative_area(all_test_gid_list, target_species=target_species)
#     cumulative_area_list, total_area_list, flag_list = values
#     gt_positive_test_gid_list = sorted(ut.compress(all_test_gid_list, flag_list))
#     # gt_negative_test_gid_list = sorted(set(all_test_gid_list) - set(gt_positive_test_gid_list))

#     model_tag = 'scout-d3e8bf43-boost4'
#     all_test_confidence_list = ibs.scout_wic_test(all_test_gid_list, model_tag=model_tag)
#     all_test_flag_list = [
#         all_test_confidence >= thresh
#         for all_test_confidence in all_test_confidence_list
#     ]
#     wic_positive_test_gid_list = sorted(ut.compress(all_test_gid_list, all_test_flag_list))
#     # wic_negative_test_gid_list = sorted(set(all_test_gid_list) - set(wic_positive_test_gid_list))

#     config = {'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'scout_v0', 'weight_filepath' : 'scout_v0', 'nms': True, 'nms_thresh': 0.50, 'sensitivity': 0.4425}

#     # Visualize
#     ibs.visualize_predictions(config, gid_list=gt_positive_test_gid_list)
#     ibs.visualize_ground_truth(config, gid_list=gt_positive_test_gid_list)

#     ignore_filter_func_ = partial(_scout_localizer_ignore_filter_func, margin=margin, min_bbox_coverage=min_bbox_coverage)

#     # All Positive Tiles (All)
#     config_dict = {
#         'scout-gt-positive-all-v0': template_v0,
#         'scout-gt-positive-all-v1': template_v1,
#     }
#     ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=gt_positive_test_gid_list, overwrite_config_keys=True)

#     # All Positive Tiles (Margin)
#     config_dict = {
#         'scout-gt-positive-margin-%s-v0' % (margin, ): template_v0,
#         'scout-gt-positive-margin-%s-v1' % (margin, ): template_v1,
#     }
#     ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=gt_positive_test_gid_list, overwrite_config_keys=True, ignore_filter_func=ignore_filter_func_)

#     # All WIC-Passing Tiles (All)
#     config_dict = {
#         'scout-wic-passing-all-v0': template_v0,
#         'scout-wic-passing-all-v1': template_v1,
#     }
#     ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=wic_positive_test_gid_list, overwrite_config_keys=True)

#     # All WIC-Passing Tiles (Margin)
#     config_dict = {
#         'scout-wic-passing-margin-%s-v0' % (margin, ): template_v0,
#         'scout-wic-passing-margin-%s-v1' % (margin, ): template_v1,
#     }
#     ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=wic_positive_test_gid_list, overwrite_config_keys=True, ignore_filter_func=ignore_filter_func_)

#     # # All Negative Tiles
#     # config_dict = {'scout-gt-negative': template}
#     # ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=gt_negative_test_gid_list, overwrite_config_keys=True)

#     # # All WIC-Failed Tiles
#     # config_dict = {'scout-wic-failing': template}
#     # ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=wic_negative_test_gid_list, overwrite_config_keys=True)

#     # # All Test Tiles
#     # config_dict = {'scout-all': template}
#     # ibs.localizer_precision_recall(config_dict=config_dict, test_gid_list=all_test_gid_list, overwrite_config_keys=True)


@register_ibs_method
def scout_verify_negative_gt_suggestsions(
    ibs, min_confidence=0.5, max_examples=None, use_wic=True, use_loc=False, **kwargs
):
    from wbia.other.detectfuncs import localizer_parse_pred

    tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    tile_gid_list = list(tile_set)

    values = ibs.scout_tile_positive_cumulative_area(tile_gid_list)
    cumulative_area_list, total_area_list, flag_list = values
    gt_positive_gid_list = sorted(ut.compress(tile_gid_list, flag_list))
    gt_negative_gid_list = sorted(set(tile_gid_list) - set(gt_positive_gid_list))

    # WIC
    if use_wic:
        model_tag = 'scout-5fbfff26-boost3'
        confidence_list = ibs.scout_wic_test(gt_negative_gid_list, model_tag=model_tag)

        zipped = sorted(list(zip(confidence_list, gt_negative_gid_list)), reverse=True)
        if max_examples is None:
            num_negative = min(len(gt_negative_gid_list), max_examples)
        else:
            for index, value in enumerate(zipped):
                if value[0] < min_confidence:
                    break
            num_negative = index

        zipped = zipped[:num_negative]
        wic_confidence_list = ut.take_column(zipped, 0)
        wic_verify_set = set(ut.take_column(zipped, 1))
        logger.info('Num negative: %d' % (num_negative,))
        logger.info(
            'Confidence: Min %0.02f, Max %0.02f'
            % (
                min(wic_confidence_list),
                max(wic_confidence_list),
            )
        )
    else:
        wic_verify_set = set()

    # Localizer
    if use_loc:
        config = {
            'grid': False,
            'algo': 'lightnet',
            'config_filepath': 'scout_v0',
            'weight_filepath': 'scout_v0',
            'nms': True,
            'nms_thresh': 0.5,
            'sensitivity': 0.4425,
        }
        prediction_list = localizer_parse_pred(
            ibs, test_gid_list=gt_negative_gid_list, **config
        )

        value_list = []
        for negative_uuid in prediction_list:
            negative_pred = prediction_list[negative_uuid]
            area = 0
            for pred in negative_pred:
                w, h = pred['width'], pred['height']
                area += w * h
            value = (
                len(negative_pred),
                area,
                negative_uuid,
            )
            value_list.append(value)

        zipped = sorted(value_list, reverse=True)
        zipped = zipped[:num_negative]
        loc_verify_set = ut.take_column(zipped, 2)
        loc_verify_set = set(ibs.get_image_gids_from_uuid(loc_verify_set))
    else:
        loc_verify_set = set()

    logger.info('Suggested WIC Verify: %d' % (len(wic_verify_set),))
    logger.info('Suggested LOC Verify: %d' % (len(loc_verify_set),))
    verify_list = list(wic_verify_set | loc_verify_set)

    verify_gid_list = ibs.get_tile_ancestor_gids(verify_list)
    verify_gid_set = list(set(verify_gid_list))
    logger.info(
        'Suggested TOTAL Verify: %d from %d unique images'
        % (
            len(verify_list),
            len(verify_gid_set),
        )
    )

    output_path = abspath(expanduser(join('~', 'Desktop', 'verify')))
    ut.delete(output_path)
    ut.ensuredir(output_path)
    ibs.visualize_ground_truth(config, gid_list=verify_list, output_path=output_path)

    bbox_list = ibs.get_tile_bboxes(verify_list)
    zipped = sorted(zip(verify_list, verify_gid_list, bbox_list))

    verify_filepath = join(output_path, 'verify.csv')
    with open(verify_filepath, 'w') as verify_file:
        verify_file.write('TILE_UUID,TILE_ROWID,FILENAME,XTL,YTL,XBR,YBR\n')
        for tid, gid, bbox in zipped:
            xtl, ytl, w, h = bbox
            xbr = xtl + w
            ybr = ytl + h
            # image_uuid = ibs.get_image_uuids(gid)
            tile_uuid = ibs.get_image_uuids(tid)
            original_filepath = ibs.get_image_uris_original(gid)
            original_filepath = original_filepath.replace('/Users/jason.parham/raw/', '')
            args = (
                tile_uuid,
                tid,
                original_filepath,
                xtl,
                ytl,
                xbr,
                ybr,
            )
            args = map(str, args)
            args_str = ','.join(args)
            verify_file.write('{}\n'.format(args_str))


@register_ibs_method
def scout_localizer_visualize_errors_annots(
    ibs,
    target_species='elephant_savanna',
    min_cumulative_percentage=0.025,
    sensitivity=0.4425,
    errors_only=False,
    **kwargs
):
    import matplotlib.pyplot as plt

    import wbia.plottool as pt
    from wbia.other.detectfuncs import (
        general_parse_gt,
        localizer_parse_pred,
        localizer_tp_fp,
    )

    fig_ = plt.figure(figsize=(12, 20), dpi=400)  # NOQA

    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    test_gid_set = all_tile_set & test_gid_set
    test_gid_list = list(test_gid_set)

    values = ibs.scout_tile_positive_cumulative_area(
        test_gid_list,
        target_species=target_species,
        min_cumulative_percentage=min_cumulative_percentage,
    )
    cumulative_area_list, total_area_list, flag_list = values
    area_percentage_list = [
        cumulative_area / total_area
        for cumulative_area, total_area in zip(cumulative_area_list, total_area_list)
    ]

    config = {
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'scout_v0',
        'weight_filepath': 'scout_v0',
        'nms': True,
        'nms_thresh': 0.5,
        'sensitivity': sensitivity,
    }

    test_uuid_list = ibs.get_image_uuids(test_gid_list)
    logger.info('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **config)

    logger.info('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **config)

    # Filter for speices
    dict_list = [
        (gt_dict, 'Ground-Truth'),
        (pred_dict, 'Predictions'),
    ]
    for dict_, dict_tag in dict_list:
        for image_uuid in dict_:
            temp = []
            for val in dict_[image_uuid]:
                if val.get('class', None) != target_species:
                    continue
                temp.append(val)
            dict_[image_uuid] = temp

    values = localizer_tp_fp(
        test_uuid_list,
        gt_dict,
        pred_dict,
        return_match_dict=True,
        min_overlap=0.2,
        **kwargs
    )
    conf_list, tp_list, fp_list, total, match_dict = values

    color_list = pt.distinct_colors(4, randomize=False)

    # Coverage
    logger.info('Plotting coverage')
    plt.subplot(211)

    bucket_size = 5.0
    percentage_dict = {}
    for test_uuid, percentage in zip(test_uuid_list, area_percentage_list):
        if percentage < min_cumulative_percentage:
            bucket = -1
        else:
            bucket = int((percentage * 100.0) / bucket_size)

        if bucket not in percentage_dict:
            percentage_dict[bucket] = [0, 0, 0, 0]

        match_list, total = match_dict[test_uuid]
        tp = 0
        for match in match_list:
            conf, flag, gt, overlap = match

            if flag:
                tp += 1
                if not errors_only:
                    percentage_dict[bucket][0] += 1
            else:
                percentage_dict[bucket][2] += 1
        percentage_dict[bucket][1] += total - tp

    width = 0.35
    percentage_list = sorted(percentage_dict.keys())
    index_list = np.arange(len(percentage_list))

    bottom = None
    bar_list = []
    for index, color in enumerate(color_list):
        value_list = []
        for percentage in percentage_list:
            value = percentage_dict[percentage][index]
            value_list.append(value)
        value_list = np.array(value_list)
        logger.info(value_list)
        if bottom is None:
            bottom = np.zeros(value_list.shape, dtype=value_list.dtype)
        bar_ = plt.bar(index_list, value_list, width, color=color, bottom=bottom)
        bar_list.append(bar_)
        bottom += value_list

    label_list = ['TP', 'FN', 'FP', 'TN']
    plt.legend(bar_list, label_list)

    plt.ylabel('Number of Tiles')
    plt.yscale('log')
    if errors_only:
        plt.title('WIC Performance by Area of Coverage (Errors only)')
    else:
        plt.title('WIC Performance by Area of Coverage')

    tick_list = ['[0, 2.5)', '[2.5, 5)']
    for percentage in percentage_list:
        if percentage <= 0:
            continue
        bucket_min = int(bucket_size * percentage)
        bucket_max = int(bucket_size * (percentage + 1))
        tick = '[%d, %d)' % (
            bucket_min,
            bucket_max,
        )
        tick_list.append(tick)

    plt.xticks(index_list, tick_list)

    # Number of annotations
    logger.info('Plotting num annotations')
    plt.subplot(212)

    percentage_dict = {}
    for test_uuid in test_uuid_list:
        bucket = len(pred_dict[test_uuid])

        if bucket not in percentage_dict:
            percentage_dict[bucket] = [0, 0, 0, 0]

        match_list, total = match_dict[test_uuid]
        tp = 0
        for match in match_list:
            conf, flag, index, overlap = match

            if flag:
                tp += 1
                if not errors_only:
                    percentage_dict[bucket][0] += 1
            else:
                percentage_dict[bucket][2] += 1
        percentage_dict[bucket][1] += total - tp

    width = 0.35
    percentage_list = sorted(percentage_dict.keys())
    index_list = np.arange(len(percentage_list))

    bottom = None
    bar_list = []
    for index, color in enumerate(color_list):
        value_list = []
        for percentage in percentage_list:
            value = percentage_dict[percentage][index]
            value_list.append(value)
        value_list = np.array(value_list)
        logger.info(value_list)
        if bottom is None:
            bottom = np.zeros(value_list.shape, dtype=value_list.dtype)
        bar_ = plt.bar(index_list, value_list, width, color=color, bottom=bottom)
        bar_list.append(bar_)
        bottom += value_list

    label_list = ['TP', 'FN', 'FP', 'TN']
    plt.legend(bar_list, label_list)

    plt.ylabel('Number of Tiles')
    plt.yscale('log')
    if errors_only:
        plt.title('Localization Performance by Number of Annotations (Errors only)')
    else:
        plt.title('Localization Performance by Number of Annotations')

    tick_list = []
    for percentage in percentage_list:
        tick = '%d' % (percentage,)
        tick_list.append(tick)

    plt.xticks(index_list, tick_list)

    if errors_only:
        fig_filename = 'scout-loc-errors-annots-plot-errors.png'
    else:
        fig_filename = 'scout-loc-errors-annots-plot.png'

    fig_filepath = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_filepath, bbox_inches='tight')


@register_ibs_method
def scout_localizer_visualize_errors_clusters(
    ibs,
    target_species='elephant_savanna',
    sensitivity=0.4425,
    thresh=0.024,
    errors_only=False,
    **kwargs
):
    import matplotlib.pyplot as plt

    import wbia.plottool as pt
    from wbia.other.detectfuncs import (
        general_parse_gt,
        localizer_parse_pred,
        localizer_tp_fp,
    )

    fig_ = plt.figure(figsize=(40, 12), dpi=400)  # NOQA

    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    test_gid_set = all_tile_set & test_gid_set
    test_gid_list = list(test_gid_set)

    values = ibs.scout_tile_positive_cumulative_area(
        test_gid_list, target_species=target_species
    )
    cumulative_area_list, total_area_list, flag_list = values

    config = {
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': 'scout_v0',
        'weight_filepath': 'scout_v0',
        'nms': True,
        'nms_thresh': 0.5,
        'sensitivity': sensitivity,
    }

    test_uuid_list = ibs.get_image_uuids(test_gid_list)
    logger.info('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **config)

    logger.info('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **config)

    # Filter for speices
    dict_list = [
        (gt_dict, 'Ground-Truth'),
        (pred_dict, 'Predictions'),
    ]
    for dict_, dict_tag in dict_list:
        for image_uuid in dict_:
            temp = []
            for val in dict_[image_uuid]:
                if val.get('class', None) != target_species:
                    continue
                temp.append(val)
            dict_[image_uuid] = temp

    values = localizer_tp_fp(
        test_uuid_list,
        gt_dict,
        pred_dict,
        return_match_dict=True,
        min_overlap=0.2,
        **kwargs
    )
    conf_list, tp_list, fp_list, total, match_dict = values

    values = ibs.scout_compute_visual_clusters(80, 10, **kwargs)
    hashstr, assignment_dict, cluster_dict, cluster_center_dict, limits = values

    color_list = pt.distinct_colors(4, randomize=False)

    # Coverage
    logger.info('Plotting clusters')
    plt.subplot(111)

    percentage_dict = {}
    for test_gid, test_uuid in zip(test_gid_list, test_uuid_list):
        cluster, embedding = assignment_dict[test_gid]
        bucket = int(cluster)

        if bucket not in percentage_dict:
            percentage_dict[bucket] = [0, 0, 0, 0]

        match_list, total = match_dict[test_uuid]
        tp = 0
        for match in match_list:
            conf, flag, gt, overlap = match

            if flag:
                tp += 1
                if not errors_only:
                    percentage_dict[bucket][0] += 1
            else:
                percentage_dict[bucket][2] += 1
        percentage_dict[bucket][1] += total - tp

    width = 0.35
    percentage_list = sorted(percentage_dict.keys())
    index_list = np.arange(len(percentage_list))

    bottom = None
    bar_list = []
    for index, color in enumerate(color_list):
        value_list = []
        for percentage in percentage_list:
            value = percentage_dict[percentage][index]
            value_list.append(value)
        value_list = np.array(value_list)
        logger.info(value_list)
        if bottom is None:
            bottom = np.zeros(value_list.shape, dtype=value_list.dtype)
        bar_ = plt.bar(index_list, value_list, width, color=color, bottom=bottom)
        bar_list.append(bar_)
        bottom += value_list

    label_list = ['TP', 'FN', 'FP', 'TN']
    plt.legend(bar_list, label_list)

    plt.ylabel('Number of Tiles')
    if errors_only:
        plt.title('Localization Performance by Visual Cluster (Errors only)')
    else:
        plt.yscale('log')
        plt.title('Localization Performance by Visual Cluster')
    tick_list = []
    for percentage in percentage_list:
        tick = '%d' % (percentage,)
        tick_list.append(tick)
    plt.xticks(index_list, tick_list)

    if errors_only:
        fig_filename = 'scout-loc-errors-clusters-plot-errors.png'
    else:
        fig_filename = 'scout-loc-errors-clusters-plot.png'
    fig_filepath = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_filepath, bbox_inches='tight')


def _scout_compute_annotation_clusters_metric_func(value1, value2):
    cx1, cy1, radius1 = value1
    cx2, cy2, radius2 = value2
    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    dist = max(0.0, dist - radius1 - radius2)
    return dist


@register_ibs_method
def _scout_compute_annotation_clusters(ibs, bbox_list, distance=128):
    import numpy as np
    from scipy.cluster.hierarchy import fclusterdata

    if len(bbox_list) == 0:
        return [], []

    centers = []
    radii = []
    for bbox in bbox_list:
        xtl, ytl, w, h = bbox
        cx = int(np.around(xtl + (w // 2)))
        cy = int(np.around(ytl + (h // 2)))
        center = (
            cx,
            cy,
        )
        radius = int(np.around(max(w, h) * 0.5))
        centers.append(center)
        radii.append(radius)

    centers = np.array(centers)
    radii = np.array(radii).reshape(-1, 1)
    value_list = np.hstack((centers, radii))

    if len(value_list) > 1:
        prediction_list = fclusterdata(
            value_list,
            t=distance,
            criterion='distance',
            metric=_scout_compute_annotation_clusters_metric_func,
        )
    else:
        prediction_list = [1]

    return prediction_list, value_list


@register_ibs_method
def scout_visualize_annotation_clusters(
    ibs, assignment_image_dict, tag, output_path=None, **kwargs
):
    import cv2

    import wbia.plottool as pt

    if output_path is None:
        folder_name = 'bboxes_circles_{}'.format(tag)
        output_path = abspath(expanduser(join('~', 'Desktop', folder_name)))
        ut.ensuredir(output_path)

    for gid in tqdm.tqdm(assignment_image_dict):
        assignment_annot_dict, value_annot_dict = assignment_image_dict[gid]

        index_list = sorted(assignment_annot_dict.keys())
        globals().update(locals())
        prediction_list = [assignment_annot_dict[index] for index in index_list]
        cluster_list = sorted(set(prediction_list))
        color_list = pt.distinct_colors(len(cluster_list), randomize=False)
        color_list = [
            tuple(map(int, np.around(np.array(color, dtype=np.float32) * 255.0)))
            for color in color_list
        ]
        color_dict = dict(zip(cluster_list, color_list))

        image = ibs.get_images(gid)
        for index in index_list:
            cluster = assignment_annot_dict[index]
            value = value_annot_dict[index]
            color = color_dict[cluster]
            cx, cy, radius = value
            cv2.circle(image, (cx, cy), 3, color, 5)
            cv2.circle(image, (cx, cy), radius, color, 5)

        write_filename = 'bboxes_%d_circle_annots_%d_clusters_%d.png' % (
            gid,
            len(index_list),
            len(cluster_list),
        )
        write_filepath = join(output_path, write_filename)
        logger.info(write_filepath)
        cv2.imwrite(write_filepath, image)


@register_ibs_method
def scout_compute_gt_annotation_clusters(
    ibs, target_species='elephant_savanna', use_ancestors=True, **kwargs
):

    all_tile_set = set(ibs.scout_get_valid_tile_rowids(**kwargs))
    all_tile_list = list(all_tile_set)
    values = ibs.scout_tile_positive_cumulative_area(all_tile_list)
    cumulative_area_list, total_area_list, flag_list = values
    gt_positive_gid_list = sorted(ut.compress(all_tile_list, flag_list))

    if use_ancestors:
        gt_positive_gid_list_ = ibs.get_tile_ancestor_gids(gt_positive_gid_list)
        gt_positive_gid_list = list(set(gt_positive_gid_list_))

    assignment_image_dict = {}
    aids_list = ibs.get_image_aids(gt_positive_gid_list)
    zipped = zip(gt_positive_gid_list, aids_list)
    for gid, aid_list in tqdm.tqdm(zipped):
        aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
        bbox_list = ibs.get_annot_bboxes(aid_list, reference_tile_gid=gid)

        prediction_list, value_list = ibs._scout_compute_annotation_clusters(
            bbox_list, **kwargs
        )
        assignment_annot_dict = dict(zip(aid_list, prediction_list))
        value_annot_dict = dict(zip(aid_list, value_list))
        assignment_image_dict[gid] = (
            assignment_annot_dict,
            value_annot_dict,
        )

    return assignment_image_dict


@register_ibs_method
def scout_visualize_gt_annotation_clusters(ibs, use_ancestors=True, **kwargs):
    logger.info('Computing Assignments')
    tag = 'images_gt' if use_ancestors else 'tiles_gt'
    assignment_image_dict = ibs.scout_compute_gt_annotation_clusters(
        ibs, use_ancestors=use_ancestors, **kwargs
    )
    ibs.scout_visualize_annotation_clusters(assignment_image_dict, tag, **kwargs)


@register_ibs_method
def scout_compute_pred_annotation_clusters(
    ibs, target_species='elephant_savanna', use_ancestors=True, quick=True, **kwargs
):
    assert use_ancestors is True, 'Tile cluster predictions are not supported yet'

    gt_positive_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('POSITIVE_IMAGE'))
    )
    gt_test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    pred_gid_list = list(gt_positive_gid_set & gt_test_gid_set)

    result_list = ibs.scout_detect(
        pred_gid_list, quick=quick, return_clustering_plot_values=True, **kwargs
    )

    assignment_image_dict = {}
    zipped = zip(pred_gid_list, result_list)
    for gid, result in tqdm.tqdm(zipped):
        bboxes, classes, confs, prediction_list, value_list = result
        aid_list = list(range(len(bboxes)))

        assignment_annot_dict = dict(zip(aid_list, prediction_list))
        value_annot_dict = dict(zip(aid_list, value_list))
        assignment_image_dict[gid] = (
            assignment_annot_dict,
            value_annot_dict,
        )

    return assignment_image_dict


@register_ibs_method
def scout_visualize_pred_annotation_clusters(
    ibs, use_ancestors=True, quick=True, **kwargs
):
    logger.info('Computing Assignments')
    tag = 'images_pred_quick_%s' if use_ancestors else 'tiles_pred_quick_%s'
    tag = tag % (quick,)
    assignment_image_dict = ibs.scout_compute_pred_annotation_clusters(
        ibs, use_ancestors=use_ancestors, quick=quick, **kwargs
    )
    ibs.scout_visualize_annotation_clusters(assignment_image_dict, tag, **kwargs)


@register_ibs_method
def scout_visualize_annotation_clusters_distribution(
    ibs, target_species='elephant_savanna', min_cumulative_percentage=0.025, **kwargs
):
    import matplotlib.pyplot as plt

    import wbia.plottool as pt

    fig_ = plt.figure(figsize=(12, 20), dpi=400)  # NOQA

    assignment_image_dict = ibs.scout_compute_annotation_clusters(ibs, **kwargs)
    test_gid_list = sorted(assignment_image_dict.keys())

    values = ibs.scout_tile_positive_cumulative_area(
        test_gid_list,
        target_species=target_species,
        min_cumulative_percentage=min_cumulative_percentage,
    )
    cumulative_area_list, total_area_list, flag_list = values
    area_percentage_list = [
        cumulative_area / total_area
        for cumulative_area, total_area in zip(cumulative_area_list, total_area_list)
    ]

    num_clusters = 0
    for gid in assignment_image_dict:
        assignment_annot_dict, value_annot_dict = assignment_image_dict[gid]
        clusters = []
        for aid in assignment_annot_dict:
            cluster = assignment_annot_dict[aid]
            clusters.append(cluster)
        clusters = set(clusters)
        num_clusters = max(num_clusters, len(clusters))

    color_list = pt.distinct_colors(num_clusters, randomize=False)

    # Coverage
    logger.info('Plotting coverage')
    plt.subplot(211)

    bucket_size = 5.0
    percentage_dict = {}
    for test_gid, percentage in zip(test_gid_list, area_percentage_list):
        if percentage < min_cumulative_percentage:
            bucket = -1
        else:
            bucket = int((percentage * 100.0) / bucket_size)

        if bucket not in percentage_dict:
            percentage_dict[bucket] = [0] * num_clusters

        assignment_annot_dict, value_annot_dict = assignment_image_dict[test_gid]
        clusters = []
        for aid in assignment_annot_dict:
            cluster = assignment_annot_dict[aid]
            clusters.append(cluster)
        clusters = set(clusters)
        num_clusters_ = len(clusters)

        cluster = num_clusters_ - 1
        percentage_dict[bucket][cluster] += 1

    width = 0.35
    percentage_list = sorted(percentage_dict.keys())
    index_list = np.arange(len(percentage_list))

    bottom = None
    bar_list = []
    for index, color in enumerate(color_list):
        value_list = []
        for percentage in percentage_list:
            value = percentage_dict[percentage][index]
            value_list.append(value)
        value_list = np.array(value_list)
        logger.info(value_list)
        if bottom is None:
            bottom = np.zeros(value_list.shape, dtype=value_list.dtype)
        bar_ = plt.bar(index_list, value_list, width, color=color, bottom=bottom)
        bar_list.append(bar_)
        bottom += value_list

    label_list = list(map(str, range(1, num_clusters + 1)))
    plt.legend(bar_list, label_list)

    plt.ylabel('Number of Tiles')
    # plt.yscale('log')
    plt.title('Number of Annotation Clusters by Area of Coverage')

    tick_list = ['[0, 2.5)', '[2.5, 5)']
    for percentage in percentage_list:
        if percentage <= 0:
            continue
        bucket_min = int(bucket_size * percentage)
        bucket_max = int(bucket_size * (percentage + 1))
        tick = '[%d, %d)' % (
            bucket_min,
            bucket_max,
        )
        tick_list.append(tick)

    plt.xticks(index_list, tick_list)

    # Number of annotations
    logger.info('Plotting num annotations')
    plt.subplot(212)

    percentage_dict = {}
    aids_list = ibs.get_image_aids(test_gid_list)
    for test_gid, aid_list in zip(test_gid_list, aids_list):
        aid_list = ibs.filter_annotation_set(aid_list, species=target_species)
        bucket = len(aid_list)

        if bucket not in percentage_dict:
            percentage_dict[bucket] = [0] * num_clusters

        assignment_annot_dict, value_annot_dict = assignment_image_dict[test_gid]
        clusters = []
        for aid in assignment_annot_dict:
            cluster = assignment_annot_dict[aid]
            clusters.append(cluster)
        clusters = set(clusters)
        num_clusters_ = len(clusters)

        cluster = num_clusters_ - 1
        percentage_dict[bucket][cluster] += 1

    width = 0.35
    percentage_list = sorted(percentage_dict.keys())
    index_list = np.arange(len(percentage_list))

    bottom = None
    bar_list = []
    for index, color in enumerate(color_list):
        value_list = []
        for percentage in percentage_list:
            value = percentage_dict[percentage][index]
            value_list.append(value)
        value_list = np.array(value_list)
        logger.info(value_list)
        if bottom is None:
            bottom = np.zeros(value_list.shape, dtype=value_list.dtype)
        bar_ = plt.bar(index_list, value_list, width, color=color, bottom=bottom)
        bar_list.append(bar_)
        bottom += value_list

    label_list = list(map(str, range(1, num_clusters + 1)))
    plt.legend(bar_list, label_list)

    plt.ylabel('Number of Tiles')
    # plt.yscale('log')
    plt.title('Number of Annotation Clusters by Number of Annotations')

    tick_list = []
    for percentage in percentage_list:
        tick = '%d' % (percentage,)
        tick_list.append(tick)

    plt.xticks(index_list, tick_list)

    fig_filename = 'scout-annot-clusters-annots-plot.png'
    fig_filepath = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_filepath, bbox_inches='tight')


# @register_ibs_method
# def _export_test_daatset(ibs):
#     import wbia
#     from wbia.dbio.export_subset import merge_databases
#     import wbia.constants as const
#     import random

#     gid_list = ibs.get_valid_gids()
#     random.shuffle(gid_list)
#     gid_list_ = gid_list[:50]

#     ibs_src = ibs
#     ibs_dst = wbia.opendb(dbdir='/data/wbia/TEST_Scout/')
#     rowid_subsets = {
#         const.IMAGE_TABLE: gid_list_
#     }
#     merge_databases(ibs_src, ibs_dst, rowid_subsets=rowid_subsets, localize_images=True)


@register_ibs_method
def scout_localizer_count_residuals_exploration(
    ibs, target_species='elephant_savanna', **kwargs
):
    from wbia.other.detectfuncs import (
        general_parse_gt,
        localizer_parse_pred,
        localizer_tp_fp,
    )

    gid_list = ibs.get_valid_gids()
    test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    test_gid_set = set(gid_list) & test_gid_set
    test_gid_list = list(test_gid_set)

    test_uuid_list = ibs.get_image_uuids(test_gid_list)
    logger.info('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list)

    # Filter for speices
    dict_list = [
        (gt_dict, 'Ground-Truth'),
        # (pred_dict, 'Predictions'),
    ]
    for dict_, dict_tag in dict_list:
        for image_uuid in dict_:
            temp = []
            for val in dict_[image_uuid]:
                if val.get('class', None) != target_species:
                    continue
                temp.append(val)
            dict_[image_uuid] = temp

    detection_config_list = []
    for agg_algo in ['tile_aggregation_quick', 'tile_aggregation']:
        for agg_variant in tqdm.tqdm(
            [
                'variant1',
                'variant2-32',
                'variant2-64',
                'variant3-32',
                'variant4-32',
                'variant4-64',
            ]
        ):
            for wic_model in [
                'scout-5fbfff26-boost0',
                'scout-5fbfff26-boost1',
                'scout-5fbfff26-boost2',
                'scout-5fbfff26-boost3',
            ]:
                for wic_sensitivity in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    for loc_model in ['scout_5fbfff26_v0']:
                        for loc_nms in [0.1, 0.3, 0.5, 0.7, 0.9]:
                            for agg_nms in [0.1, 0.3, 0.5, 0.7, 0.9]:
                                for agg_sensitivity in [0.1, 0.3, 0.5, 0.7, 0.9]:
                                    detection_config = {
                                        'algo': agg_algo,
                                        'config_filepath': agg_variant,
                                        'weight_filepath': 'densenet+lightnet;%s,%s,%s,%s'
                                        % (
                                            wic_model,
                                            wic_sensitivity,
                                            loc_model,
                                            loc_nms,
                                        ),
                                        'nms_thresh': agg_nms,
                                        'sensitivity': agg_sensitivity,
                                    }
                                    detection_config_list.append(detection_config)

    random.shuffle(detection_config_list)

    keep = 20
    snapshot = 50
    running_list = []
    for detection_config in tqdm.tqdm(detection_config_list):
        pred_dict = localizer_parse_pred(
            ibs, test_gid_list=test_gid_list, **detection_config
        )

        # Filter for speices
        dict_list = [
            # (gt_dict, 'Ground-Truth'),
            (pred_dict, 'Predictions'),
        ]
        for dict_, dict_tag in dict_list:
            for image_uuid in dict_:
                temp = []
                for val in dict_[image_uuid]:
                    if val.get('class', None) != target_species:
                        continue
                    temp.append(val)
                dict_[image_uuid] = temp

        values = localizer_tp_fp(
            test_uuid_list,
            gt_dict,
            pred_dict,
            return_match_dict=True,
            min_overlap=0.2,
            **kwargs
        )
        conf_list, tp_list, fp_list, total, match_dict = values
        correct = 0
        net_bias = 0
        for test_gid, test_uuid in zip(test_gid_list, test_uuid_list):
            gt_list = gt_dict[test_uuid]
            pred_list = pred_dict[test_uuid]

            bias = len(pred_list) - len(gt_list)
            if bias == 0:
                correct += 1
            else:
                net_bias += bias
        net_bias_ = -1 * abs(net_bias)
        result = (correct, net_bias_, net_bias, ut.to_json(detection_config))
        logger.info(ut.repr3(result))
        running_list.append(result)

        if len(running_list) >= snapshot:
            running_list = sorted(running_list, reverse=True)
            running_list = running_list[:keep]
            logger.info(ut.repr3(running_list))

    return running_list


@register_ibs_method
def scout_localizer_visualize_annotation_clusters_residuals(
    ibs, quick=False, target_species='elephant_savanna', distance=128, **kwargs
):
    import matplotlib.pyplot as plt

    from wbia.other.detectfuncs import (
        general_parse_gt,
        localizer_parse_pred,
        localizer_tp_fp,
    )

    gid_list = ibs.get_valid_gids()
    test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    test_gid_set = set(gid_list) & test_gid_set
    test_gid_list = list(test_gid_set)

    detection_config = ibs.scout_detect_config(quick=quick)

    test_uuid_list = ibs.get_image_uuids(test_gid_list)
    logger.info('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **detection_config)

    logger.info('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **detection_config)

    # Filter for speices
    dict_list = [
        (gt_dict, 'Ground-Truth'),
        (pred_dict, 'Predictions'),
    ]
    for dict_, dict_tag in dict_list:
        for image_uuid in dict_:
            temp = []
            for val in dict_[image_uuid]:
                if val.get('class', None) != target_species:
                    continue
                temp.append(val)
            dict_[image_uuid] = temp

    values = localizer_tp_fp(
        test_uuid_list,
        gt_dict,
        pred_dict,
        return_match_dict=True,
        min_overlap=0.2,
        **kwargs
    )
    conf_list, tp_list, fp_list, total, match_dict = values

    fig_ = plt.figure(figsize=(60, 12), dpi=400)  # NOQA

    # Annots / Image
    logger.info('Plotting Annots / Image')
    plt.subplot(151)

    bias_label_list = [
        (-10, '<=-10'),
        (-5, '-9 to -5'),
        (-4, '-4'),
        (-3, '-3'),
        (-2, '-2'),
        (-1, '-1'),
        (0, '0'),
        (1, '1'),
        (2, '2'),
        (3, '3'),
        (4, '4'),
        (5, '5 to 9'),
        (10, '>=10'),
    ]

    bias_dict = {bias_index: 0 for bias_index, bias_label in bias_label_list}
    totalx, undercountx, correctx, overcountx = 0, 0, 0, 0
    for test_gid, test_uuid in zip(test_gid_list, test_uuid_list):
        gt_list = gt_dict[test_uuid]
        pred_list = pred_dict[test_uuid]

        bias = len(pred_list) - len(gt_list)

        totalx += len(gt_list)
        if bias == 0:
            correctx += len(gt_list)
        elif bias < 0:
            undercountx += -1 * bias
        elif bias > 0:
            overcountx += bias

        if bias <= -10:
            bias = -10
        elif bias <= -5:
            bias = -5
        elif bias >= 10:
            bias = 10
        elif bias >= 5:
            bias = 5

        assert bias in bias_dict
        bias_dict[bias] += 1

    width = 0.50
    index_list = np.arange(len(bias_dict))
    key_list = ut.take_column(bias_label_list, 0)
    label_list = ut.take_column(bias_label_list, 1)
    value_list = ut.take(bias_dict, key_list)

    undercounti, correcti, overcounti = 0, 0, 0
    color_list = []
    for key, value in zip(key_list, value_list):
        if key < 0:
            undercounti += value
            color = (0.8078, 0.2039, 0.1647)
        elif key == 0:
            correcti += value
            color = (0.4118, 0.8588, 0.2824)
        else:
            overcounti += value
            color = (0.2824, 0.619, 0.8549)
        color_list.append(color)

    for index, value, color in zip(index_list, value_list, color_list):
        plt.bar([index], [value], width, color=color)

    logger.info(totalx)
    plt.ylabel('Number of Images')
    plt.xlabel('Bias for Detections (PRED - GT)')
    plt.yscale('log')
    args = (
        correcti,
        correctx,
        totalx,
        undercounti,
        undercountx,
        overcounti,
        overcountx,
    )
    plt.title(
        'Count Bias for Detections / Image\nCorrect (%d Img, %d Det / %d Total)\n(%d Img, %d Det) Under | Over (%d Img, %d Det)'
        % args
    )
    plt.xticks(index_list, label_list)

    # Clusters / Image
    logger.info('Plotting Clusters / Image')
    plt.subplot(152)

    bias_label_list = [
        (-5, '<=-5'),
        (-4, '-4'),
        (-3, '-3'),
        (-2, '-2'),
        (-1, '-1'),
        (0, '0'),
        (1, '1'),
        (2, '2'),
        (3, '3'),
        (4, '4'),
        (5, '>=5'),
    ]

    bias_dict = {bias_index: 0 for bias_index, bias_label in bias_label_list}
    totalx, undercountx, correctx, overcountx = 0, 0, 0, 0
    for test_uuid in test_uuid_list:
        gt_list = gt_dict[test_uuid]
        pred_list = pred_dict[test_uuid]

        image_height, image_width = ibs.get_image_sizes(test_gid)
        globals().update(locals())
        gt_bbox_list = [
            (
                int(np.around(gt['xtl'] * image_width)),
                int(np.around(gt['ytl'] * image_height)),
                int(np.around(gt['width'] * image_width)),
                int(np.around(gt['height'] * image_height)),
            )
            for gt in gt_list
        ]
        pred_bbox_list = [
            (
                int(np.around(pred['xtl'] * image_width)),
                int(np.around(pred['ytl'] * image_height)),
                int(np.around(pred['width'] * image_width)),
                int(np.around(pred['height'] * image_height)),
            )
            for pred in pred_list
        ]
        gt_prediction_list, gt_value_list = _scout_compute_annotation_clusters(
            ibs, gt_bbox_list, distance=distance
        )
        pred_prediction_list, pred_value_list = _scout_compute_annotation_clusters(
            ibs, pred_bbox_list, distance=distance
        )
        bias = len(set(pred_prediction_list)) - len(set(gt_prediction_list))

        totalx += len(set(gt_prediction_list))
        if bias == 0:
            correctx += len(set(gt_prediction_list))
        elif bias < 0:
            undercountx += -1 * bias
        elif bias > 0:
            overcountx += bias

        if bias <= -5:
            bias = -5
        elif bias >= 5:
            bias = 5

        assert bias in bias_dict
        bias_dict[bias] += 1

    width = 0.50
    index_list = np.arange(len(bias_dict))
    key_list = ut.take_column(bias_label_list, 0)
    label_list = ut.take_column(bias_label_list, 1)
    value_list = ut.take(bias_dict, key_list)

    undercounti, correcti, overcounti = 0, 0, 0
    color_list = []
    for key, value in zip(key_list, value_list):
        if key < 0:
            undercounti += value
            color = (0.8078, 0.2039, 0.1647)
        elif key == 0:
            correcti += value
            color = (0.4118, 0.8588, 0.2824)
        else:
            overcounti += value
            color = (0.2824, 0.619, 0.8549)
        color_list.append(color)

    for index, value, color in zip(index_list, value_list, color_list):
        plt.bar([index], [value], width, color=color)

    logger.info(totalx)
    plt.ylabel('Number of Images')
    plt.xlabel('Bias in Clusters (PRED - GT)')
    plt.yscale('log')
    args = (
        correcti,
        correctx,
        totalx,
        undercounti,
        undercountx,
        overcounti,
        overcountx,
    )
    plt.title(
        'Count Bias for Clusters / Image\nCorrect (%d Img, %d Clust / %d Total)\n(%d Img, %d Clust) Under | Over (%d Img, %d Clust)'
        % args
    )
    plt.xticks(index_list, label_list)

    # Annot / Cluster
    logger.info('Plotting Annot / Cluster')
    plt.subplot(153)

    bias_label_list = [
        (-5, '<=-5'),
        (-4, '-4'),
        (-3, '-3'),
        (-2, '-2'),
        (-1, '-1'),
        (0, '0'),
        (1, '1'),
        (2, '2'),
        (3, '3'),
        (4, '4'),
        (5, '>=5'),
    ]

    cluster_bias_label_list = [
        (1, '1'),
        (2, '2'),
        (3, '3-4'),
        (5, '5-9'),
        (10, '10-14'),
        (15, '15+'),
    ]

    unassigned = 0
    bias_dict = {bias_index: 0 for bias_index, bias_label in bias_label_list}
    cluster_bias_dict = {
        bias_index: [[0, 0, 0], [0, 0, 0]]
        for bias_index, bias_label in cluster_bias_label_list
    }

    totalx, undercountx, correctx, overcountx = 0, 0, 0, 0
    for test_uuid in test_uuid_list:
        gt_list = gt_dict[test_uuid]
        pred_list = pred_dict[test_uuid]

        image_height, image_width = ibs.get_image_sizes(test_gid)
        globals().update(locals())
        gt_bbox_list = [
            (
                int(np.around(gt['xtl'] * image_width)),
                int(np.around(gt['ytl'] * image_height)),
                int(np.around(gt['width'] * image_width)),
                int(np.around(gt['height'] * image_height)),
            )
            for gt in gt_list
        ]
        pred_bbox_list = [
            (
                int(np.around(pred['xtl'] * image_width)),
                int(np.around(pred['ytl'] * image_height)),
                int(np.around(pred['width'] * image_width)),
                int(np.around(pred['height'] * image_height)),
            )
            for pred in pred_list
        ]
        gt_prediction_list, gt_value_list = _scout_compute_annotation_clusters(
            ibs, gt_bbox_list, distance=distance
        )
        pred_prediction_list, pred_value_list = _scout_compute_annotation_clusters(
            ibs, pred_bbox_list, distance=distance
        )

        cluster_tabulation = {}
        for pred_index, pred_value in enumerate(pred_value_list):
            best_distance = np.inf
            best_prediction = None
            for gt_index, (gt_prediction, gt_value) in enumerate(
                zip(gt_prediction_list, gt_value_list)
            ):
                dist = _scout_compute_annotation_clusters_metric_func(
                    pred_value, gt_value
                )
                if dist <= distance:
                    if dist < best_distance:
                        best_distance = dist
                        best_prediction = gt_prediction

            if best_prediction is not None:
                if best_prediction not in cluster_tabulation:
                    cluster_tabulation[best_prediction] = [0, 0]
                cluster_tabulation[best_prediction][0] += 1
            else:
                unassigned += 1

        for gt_prediction in gt_prediction_list:
            if gt_prediction not in cluster_tabulation:
                cluster_tabulation[gt_prediction] = [0, 0]
            cluster_tabulation[gt_prediction][0] -= 1
            cluster_tabulation[gt_prediction][1] += 1

        for key in cluster_tabulation:
            bias, total = cluster_tabulation[key]

            totalx += total
            if bias == 0:
                correctx += total
            elif bias < 0:
                undercountx += -1 * bias
            elif bias > 0:
                overcountx += bias

            if bias <= -5:
                bias = -5
            elif bias >= 5:
                bias = 5

            assert bias in bias_dict
            bias_dict[bias] += 1

            cluster_total = total
            if cluster_total <= 2:
                pass
            elif cluster_total <= 4:
                cluster_total = 3
            elif cluster_total <= 9:
                cluster_total = 5
            elif cluster_total <= 14:
                cluster_total = 10
            else:
                cluster_total = 15

            assert cluster_total in cluster_bias_dict

            if bias == 0:
                cluster_bias_dict[cluster_total][0][1] += 1
                cluster_bias_dict[cluster_total][1][1] += total
            elif bias < 0:
                cluster_bias_dict[cluster_total][0][0] += 1
                cluster_bias_dict[cluster_total][1][0] += -1 * bias
            elif bias > 0:
                cluster_bias_dict[cluster_total][0][2] += 1
                cluster_bias_dict[cluster_total][1][2] += bias

    width = 0.50
    index_list = np.arange(len(bias_dict))
    key_list = ut.take_column(bias_label_list, 0)
    label_list = ut.take_column(bias_label_list, 1)
    value_list = ut.take(bias_dict, key_list)

    undercountc, correctc, overcountc = 0, 0, 0
    color_list = []
    for key, value in zip(key_list, value_list):
        if key < 0:
            undercountc += value
            color = (0.8078, 0.2039, 0.1647)
        elif key == 0:
            correctc += value
            color = (0.4118, 0.8588, 0.2824)
        else:
            overcountc += value
            color = (0.2824, 0.619, 0.8549)
        color_list.append(color)

    for index, value, color in zip(index_list, value_list, color_list):
        plt.bar([index], [value], width, color=color)

    logger.info(totalx)
    plt.ylabel('Number of GT Clusters')
    plt.xlabel('Bias in Detections (PRED - GT)')
    plt.yscale('log')
    args = (
        correctc,
        correctx,
        totalx,
        unassigned,
        undercountc,
        undercountx,
        overcountc,
        overcountx,
    )
    plt.title(
        'Count Bias for Detections / Cluster\nCorrect (%d Clust, %d Det / %d Total), Unassigned (%d Det)\n(%d Clust, %d Det) Under | Over (%d Clust, %d Det)'
        % args
    )
    plt.xticks(index_list, label_list)

    # Over, Under, Correct Per Cluster / Cluster Size
    logger.info('Plotting Over, Under, Correct Per Cluster / Cluster Size')
    plt.subplot(154)

    width = 0.50
    index_list = np.arange(len(cluster_bias_dict))
    key_list = ut.take_column(cluster_bias_label_list, 0)
    label_list = ut.take_column(cluster_bias_label_list, 1)
    value_list = ut.take_column(ut.take(cluster_bias_dict, key_list), 0)
    color_list = [
        (0.8078, 0.2039, 0.1647),
        (0.4118, 0.8588, 0.2824),
        (0.2824, 0.619, 0.8549),
    ]

    sum_list = []
    bar_list = []
    bottom = np.array([0] * len(index_list))
    for index in range(3):
        color = color_list[index]
        value_list_ = ut.take_column(value_list, index)
        bar = plt.bar(index_list, value_list_, width, color=color, bottom=bottom)
        bar_list.append(bar)
        sum_list.append(sum(value_list_))
        bottom += np.array(value_list_)

    label_list_ = ['Undercounted', 'Correct', 'Overcounted']
    plt.legend(bar_list, label_list_)

    plt.ylabel('Number of GT Clusters')
    plt.xlabel('Cluster Size')
    # plt.yscale('log')
    args = (
        undercountc,
        undercountx,
        correctc,
        correctx,
        totalx,
        unassigned,
        overcountc,
        overcountx,
    )
    plt.title('Detection Accuracy for Clusters by Cluster Size')
    plt.xticks(index_list, label_list)

    # Over, Under, Correct Per Cluster / Cluster Size
    logger.info('Plotting Over, Under, Correct Per Detection / Cluster Size')
    plt.subplot(155)

    width = 0.50
    index_list = np.arange(len(cluster_bias_dict))
    key_list = ut.take_column(cluster_bias_label_list, 0)
    label_list = ut.take_column(cluster_bias_label_list, 1)
    value_list = ut.take_column(ut.take(cluster_bias_dict, key_list), 1)
    color_list = [
        (0.8078, 0.2039, 0.1647),
        (0.4118, 0.8588, 0.2824),
        (0.2824, 0.619, 0.8549),
    ]

    sum_list = []
    bar_list = []
    bottom = np.array([0] * len(index_list))
    for index in range(3):
        color = color_list[index]
        value_list_ = ut.take_column(value_list, index)
        bar = plt.bar(index_list, value_list_, width, color=color, bottom=bottom)
        bar_list.append(bar)
        sum_list.append(sum(value_list_))
        bottom += np.array(value_list_)

    label_list_ = ['Undercounted', 'Correct', 'Overcounted']
    plt.legend(bar_list, label_list_)

    plt.ylabel('Number of Detections')
    plt.xlabel('Cluster Size')
    # plt.yscale('log')
    args = (
        undercountc,
        undercountx,
        correctc,
        correctx,
        totalx,
        unassigned,
        overcountc,
        overcountx,
    )
    plt.title('Detection Accuracy by Cluster Size')
    plt.xticks(index_list, label_list)

    fig_filename = 'scout-errors-residuals-plot-quick-{}.png'.format(quick)
    fig_filepath = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_filepath, bbox_inches='tight')

    return


@register_ibs_method
def scout_localizer_test(
    ibs,
    test_tile_list,
    algo='lightnet',
    model_tag=None,
    sensitivity=0.0,
    nms=True,
    nms_thresh=0.2,
    invalid=True,
    invalid_margin=0.25,
    boundary=True,
    testing=False,
):
    assert model_tag is not None
    config = {
        'algo': algo,
        'config_filepath': model_tag,
        'weight_filepath': model_tag,
        'sensitivity': sensitivity,
        'nms': nms,
        'nms_thresh': nms_thresh,
        # 'grid'            : False,
        # 'invalid'         : invalid,
        # 'invalid_magin'   : invalid_margin,
        # 'boundary'        : boundary,
    }

    detections_list = ibs.depc_image.get_property(
        'localizations',
        test_tile_list,
        config=config,
        recompute=testing,
        recompute_all=testing,
    )
    return detections_list


@register_ibs_method
def scout_detect_config(ibs, quick=True):
    if quick:
        detection_config = {
            'algo': 'tile_aggregation_quick',
            'config_filepath': 'variant3-32',
            'weight_filepath': 'densenet+lightnet;scout-5fbfff26-boost3:1,0.400,scout_5fbfff26_v0,0.4',
            'nms_thresh': 0.8,
            'sensitivity': 0.4215,
        }
    else:
        detection_config = {
            'algo': 'tile_aggregation',
            'config_filepath': 'variant3-32',
            'weight_filepath': 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4',
            'nms_thresh': 0.8,
            'sensitivity': 0.5077,
        }
    return detection_config


@register_ibs_method
def scout_detect(
    ibs,
    gid_list,
    quick=True,
    testing=False,
    detection_config=None,
    return_times=False,
    return_clustering_plot_values=False,
    **kwargs
):
    if detection_config is None:
        detection_config = ibs.scout_detect_config(quick=quick)

    detections_list = ibs.depc_image.get_property(
        'localizations',
        gid_list,
        config=detection_config,
        recompute=testing,
        recompute_all=testing,
    )

    with ut.Timer('Clustering') as time_loc_cluster:
        result_list = []
        for detections in detections_list:
            score, bboxes, thetas, confs, classes = detections
            clusters, values = ibs._scout_compute_annotation_clusters(bboxes, **kwargs)
            if return_clustering_plot_values:
                result = (bboxes, classes, confs, clusters, values)
            else:
                result = (
                    bboxes,
                    classes,
                    confs,
                    clusters,
                )
            result_list.append(result)

    if return_times:
        return result_list, time_loc_cluster
    else:
        return result_list


@register_ibs_method
def _scout_localizer_visualize_tp_fp_canvas(
    ibs, value_list, line_length=10, target_size=150
):
    from wbia.web import appfuncs as appf

    image_dict = {}
    line_list = []
    line = []
    for value in tqdm.tqdm(value_list):
        gid = value['gid']
        image = ibs.get_images(gid)
        if gid not in image_dict:
            image_dict[gid] = image
        else:
            image = image_dict[gid]
        width, height = ibs.get_image_sizes(gid)
        xtl = int(np.around(value['xtl'] * width))
        ytl = int(np.around(value['ytl'] * height))
        xbr = int(np.around(value['xbr'] * width))
        ybr = int(np.around(value['ybr'] * height))
        chip = image[ytl:ybr, xtl:xbr]

        h, w, c = chip.shape
        if w <= h:
            chip = appf._resize(chip, t_height=target_size)
        else:
            chip = appf._resize(chip, t_width=target_size)
        h, w, c = chip.shape

        while chip.shape[0] < target_size:
            pad = np.zeros((1, chip.shape[1], c), dtype=chip.dtype)
            chip = np.vstack((pad, chip))
            if chip.shape[0] == target_size:
                break
            chip = np.vstack((chip, pad))

        while chip.shape[1] < target_size:
            pad = np.zeros((chip.shape[0], 1, c), dtype=chip.dtype)
            chip = np.hstack((pad, chip))
            if chip.shape[1] == target_size:
                break
            chip = np.hstack((chip, pad))

        assert h <= target_size and w <= target_size

        line.append(chip)
        if len(line) >= line_length:
            line_list.append(np.hstack(line))
            line = []

    if len(line) > 0:
        while len(line) < line_length:
            # borrow dtype and c from last chip, a bit hacky, but whatever
            chip = np.zeros((target_size, target_size, c), dtype=chip.dtype)
            line.append(chip)
        line_list.append(np.hstack(line))
    canvas = np.vstack(line_list)

    # Release images in memory
    image_dict = None

    return canvas


@register_ibs_method
def scout_localizer_visualize_tp_fp_examples(
    ibs, target_species='elephant_savanna', quick=True, **kwargs
):
    from wbia.other.detectfuncs import (
        general_parse_gt,
        localizer_parse_pred,
        localizer_tp_fp,
    )

    canvas_path = abspath(expanduser(join('~', 'Desktop')))

    detection_config = ibs.scout_detect_config(quick=quick)

    gt_positive_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('POSITIVE_IMAGE'))
    )
    gt_test_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    )
    test_gid_list = list(gt_positive_gid_set & gt_test_gid_set)

    test_uuid_list = ibs.get_image_uuids(test_gid_list)
    logger.info('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **detection_config)

    logger.info('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **detection_config)

    # Filter for speices
    dict_list = [
        (gt_dict, 'Ground-Truth'),
        (pred_dict, 'Predictions'),
    ]
    for dict_, dict_tag in dict_list:
        for image_uuid in dict_:
            temp = []
            for val in dict_[image_uuid]:
                if val.get('class', None) != target_species:
                    continue
                temp.append(val)
            dict_[image_uuid] = temp

    values = localizer_tp_fp(
        test_uuid_list,
        gt_dict,
        pred_dict,
        return_match_dict=True,
        min_overlap=0.2,
        **kwargs
    )
    conf_list, tp_list, fp_list, total, match_dict = values

    fn_det_list = []
    fp_det_list = []
    for test_uuid in test_uuid_list:
        match_list, total = match_dict[test_uuid]
        gt_list = gt_dict[test_uuid]
        pred_list = pred_dict[test_uuid]
        assert len(pred_list) == len(match_list)
        assert len(gt_list) == total

        matched_gt_index_set = set()
        matched_pred_index_set = set()
        for pred_index, match in enumerate(match_list):
            match_confidence, match_flag, match_gt, match_overlap = match
            if match_flag:
                match_index = None
                for gt_index, gt in enumerate(gt_list):
                    if gt['aid'] == match_gt['aid']:
                        match_index = gt_index
                assert match_index is not None
                matched_gt_index_set.add(match_index)
                matched_pred_index_set.add(pred_index)

        remaining_gt_index_set = set(list(range(len(gt_list)))) - matched_gt_index_set
        remaining_pred_index_set = (
            set(list(range(len(pred_list)))) - matched_pred_index_set
        )
        remaining_gt_index_list = sorted(list(remaining_gt_index_set))
        remaining_pred_index_list = sorted(list(remaining_pred_index_set))

        fn_det_list += ut.take(gt_list, remaining_gt_index_list)
        fp_det_list_preds = ut.take(pred_list, remaining_pred_index_list)
        fp_det_list_confs = ut.take(
            ut.take_column(match_list, 0), remaining_pred_index_list
        )
        fp_det_list += list(zip(fp_det_list_confs, fp_det_list_preds))

    fp_det_list = sorted(fp_det_list, reverse=True)
    fp_det_list = ut.take_column(fp_det_list, 1)

    fn_canvas = ibs._scout_localizer_visualize_tp_fp_canvas(fn_det_list, **kwargs)
    fp_canvas = ibs._scout_localizer_visualize_tp_fp_canvas(fp_det_list, **kwargs)

    canvas_filename = 'scout-detection-errors-fn.png'
    canvas_filepath = join(canvas_path, canvas_filename)
    cv2.imwrite(canvas_filepath, fn_canvas)

    canvas_filename = 'scout-detection-errors-fp.png'
    canvas_filepath = join(canvas_path, canvas_filename)
    cv2.imwrite(canvas_filepath, fp_canvas)
