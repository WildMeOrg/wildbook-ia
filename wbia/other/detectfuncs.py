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
from os.path import expanduser, join, abspath
import numpy as np
import vtool as vt
import utool as ut
import cv2
from wbia.control import controller_inject
import tqdm

# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[other.detectfuncs]')

SAMPLES = 1000
AP_SAMPLE_POINTS = [_ / float(SAMPLES) for _ in range(0, SAMPLES + 1)]

# Must import class before injection
CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)


def _resize(image, t_width=None, t_height=None, verbose=False):
    if verbose:
        print('RESIZING WITH t_width = %r and t_height = %r' % (t_width, t_height,))
    height, width = image.shape[:2]
    if t_width is None and t_height is None:
        return image
    elif t_width is not None and t_height is not None:
        pass
    elif t_width is None:
        t_width = (width / height) * float(t_height)
    elif t_height is None:
        t_height = (height / width) * float(t_width)
    t_width, t_height = float(t_width), float(t_height)
    t_width, t_height = int(np.around(t_width)), int(np.around(t_height))
    assert t_width > 0 and t_height > 0, 'target size too small'
    assert (
        t_width <= width * 10 and t_height <= height * 10
    ), 'target size too large (capped at 1000%)'
    # interpolation = cv2.INTER_LANCZOS4
    interpolation = cv2.INTER_LINEAR
    return cv2.resize(image, (t_width, t_height), interpolation=interpolation)


def simple_code(label):
    from wbia.constants import YAWALIAS, SPECIES_MAPPING

    if label == 'ignore':
        return 'IGNORE'

    for key in SPECIES_MAPPING:
        if key in label:
            species_code, species_nice = SPECIES_MAPPING[key]
            while species_code is None:
                species_code, species_nice = SPECIES_MAPPING[species_nice]
            assert species_code is not None
            label = label.replace(key, species_code)

    for key in sorted(YAWALIAS.keys(), key=len, reverse=True):
        value = YAWALIAS[key]
        label = label.replace(key, value)

    return label


##########################################################################################


def general_precision_recall_algo(
    ibs, label_list, confidence_list, category='positive', samples=SAMPLES, **kwargs
):
    def errors(zipped, conf, category):
        tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
        for index, (label, confidence) in enumerate(zipped):
            if label == category:
                if conf <= confidence:
                    tp += 1
                else:
                    fn += 1
            else:
                if conf <= confidence:
                    fp += 1
                else:
                    tn += 1
        return tp, tn, fp, fn

    zipped = list(zip(label_list, confidence_list))
    conf_list = [_ / float(samples) for _ in range(0, int(samples) + 1)]
    conf_dict = {}
    for conf in conf_list:
        conf_dict[conf] = errors(zipped, conf, category)

    conf_list_ = [-1.0, -1.0]
    pr_list = [1.0, 0.0]
    re_list = [0.0, 1.0]
    tpr_list = [0.0, 1.0]
    fpr_list = [0.0, 1.0]
    # conf_list_ = []
    # pr_list = []
    # re_list = []
    # tpr_list = []
    # fpr_list = []
    for conf in sorted(conf_dict.keys(), reverse=True):
        error_list = conf_dict[conf]
        tp, tn, fp, fn = error_list
        try:
            pr = tp / (tp + fp)
            re = tp / (tp + fn)
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            conf_list_.append(conf)
            pr_list.append(pr)
            re_list.append(re)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        except ZeroDivisionError:
            print(
                'Zero division error (%r) - tp: %r tn: %r fp: %r fn: %r'
                % (conf, tp, tn, fp, fn,)
            )

    return conf_list_, pr_list, re_list, tpr_list, fpr_list


def general_interpolate_precision_recall(conf_list, re_list, pr_list):
    conf_list_, re_list_, pr_list_ = [], [], []
    zipped = zip(re_list, conf_list, pr_list)
    zipped = sorted(zipped, reverse=True)
    max_pr = None
    for re, conf, pr in zipped:
        if max_pr is None or pr > max_pr:
            if max_pr is not None:
                conf_list_.append(np.nan)
                re_list_.append(re)
                pr_list_.append(max_pr)
            max_pr = pr
        if pr < max_pr:
            pr = max_pr
        conf_list_.append(conf)
        re_list_.append(re)
        pr_list_.append(pr)
    return conf_list_, re_list_, pr_list_


def general_identify_operating_point(conf_list, x_list, y_list, target=(1.0, 1.0)):
    best_length = np.inf
    best_conf_list = []
    best_x_list = []
    best_y_list = []
    tx, ty = target
    for conf, x, y in sorted(zip(conf_list, x_list, y_list)):
        x_ = x
        y_ = y
        x_ = x_ - tx
        y_ = y_ - ty
        length = np.sqrt(x_ * x_ + y_ * y_)
        if length < best_length:
            best_length = length
            best_conf_list = [conf]
            best_x_list = [x]
            best_y_list = [y]
        elif length == best_length:
            flag_list = [abs(best_conf - conf) > 0.01 for best_conf in best_conf_list]
            if False in flag_list:
                continue
            best_conf_list.append(conf)
            best_x_list.append(x)
            best_y_list.append(y)

    return best_conf_list, best_x_list, best_y_list, best_length


def general_area_best_conf(
    conf_list,
    x_list,
    y_list,
    label='Unknown',
    color='b',
    marker='o',
    plot_point=True,
    interpolate=True,
    target=(1.0, 1.0),
    target_recall=None,
    **kwargs,
):
    import matplotlib.pyplot as plt

    zipped = list(sorted(zip(x_list, y_list, conf_list)))
    x_list = [_[0] for _ in zipped]
    y_list = [_[1] for _ in zipped]
    conf_list = [_[2] for _ in zipped]

    if interpolate:
        conf_list, x_list, y_list = general_interpolate_precision_recall(
            conf_list, x_list, y_list
        )

    if interpolate:
        ap_list = []
        for AP_POINT in AP_SAMPLE_POINTS:
            for re, pr in sorted(zip(x_list, y_list)):
                if AP_POINT <= re:
                    ap_list.append(pr)
                    break
        ap = sum(ap_list) / len(ap_list)
    else:
        ap = np.trapz(y_list, x=x_list)

    tup1 = general_identify_operating_point(conf_list, x_list, y_list, target=target)
    best_conf_list, best_x_list, best_y_list, best_length = tup1

    tup2 = None
    if target_recall is not None:
        for x, y, conf in sorted(zip(x_list, y_list, conf_list)):
            if target_recall <= x and not np.isnan(conf):
                tup2 = [conf], [x], [y], None
                break

    if len(best_conf_list) > 1:
        print('WARNING: Multiple best operating points found %r' % (best_conf_list,))

    assert len(best_conf_list) > 0
    best_conf = best_conf_list[0]

    if interpolate:
        # label = '%s [AP = %0.02f, OP = %0.02f]' % (label, ap * 100.0, best_conf)
        label = '%s [AP = %0.02f]' % (label, ap * 100.0)
    else:
        label = '%s [AUC = %0.02f]' % (label, ap * 100.0,)

    linestyle = '--' if kwargs.get('line_dotted', False) else '-'
    plt.plot(x_list, y_list, color=color, linestyle=linestyle, label=label)

    if plot_point:
        plt.plot(best_x_list, best_y_list, color=color, marker=marker)

    return ap, best_conf, tup1, tup2


def general_confusion_matrix_algo(
    label_correct_list,
    label_predict_list,
    category_list,
    category_mapping,
    fig_,
    axes_,
    fuzzy_dict=None,
    conf=None,
    conf_list=None,
    size=10,
    **kwargs,
):
    # import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    suppressed_label = 'SUP'
    if conf is not None:
        assert conf_list is not None
        category_list.append(suppressed_label)
        index = len(category_list) - 1
        category_mapping[suppressed_label] = index
        if fuzzy_dict is not None:
            fuzzy_dict[index] = set([])

    if category_mapping is not None:
        index_list = [category_mapping[category] for category in category_list]
        zipped = list(sorted(zip(index_list, category_list)))
        category_list = [_[1] for _ in zipped]

    # Get the number of categories
    num_categories = len(category_list)

    # Build the confusion matrix
    confusion_matrix = np.zeros((num_categories, num_categories))
    zipped = zip(label_correct_list, label_predict_list)
    suppressed = 0.0
    suppressed_correct = 0.0
    suppressed_fuzzy = 0.0
    for index, (label_correct, label_predict) in enumerate(zipped):
        if conf is not None:
            conf_ = conf_list[index]
            if conf_ < conf:
                if label_correct != label_predict:
                    suppressed_correct += 1
                if fuzzy_dict is not None:
                    x = category_mapping[label_correct]
                    y = category_mapping[label_predict]
                    if not (y in fuzzy_dict[x] or x in fuzzy_dict[y]):
                        suppressed_fuzzy += 1
                label_predict = suppressed_label
                suppressed += 1
        # Perform any mapping that needs to be done
        correct_ = category_mapping[label_correct]
        predict_ = category_mapping[label_predict]
        # Add to the confidence matrix
        confusion_matrix[correct_][predict_] += 1

    # Normalize the confusion matrix using the rows
    row_normalizer = np.sum(confusion_matrix, axis=1)
    confusion_normalized = np.array((confusion_matrix.T / row_normalizer).T)

    # Draw the confusion matrix
    res = axes_.imshow(confusion_normalized, cmap=plt.cm.jet, interpolation='nearest')

    correct = suppressed_correct
    fuzzy = suppressed_fuzzy
    total = 0.0
    for x in range(num_categories):
        for y in range(num_categories):
            number = int(confusion_matrix[x][y])
            if x == y:
                correct += number
            if fuzzy_dict is not None and (y in fuzzy_dict[x] or x in fuzzy_dict[y]):
                fuzzy += number
            total += number
            axes_.annotate(
                str(number),
                xy=(y, x),
                horizontalalignment='center',
                verticalalignment='center',
                size=size,
            )

    cb = fig_.colorbar(res)  # NOQA
    cb.set_clim(0.0, 1.0)
    plt.xticks(np.arange(num_categories), category_list, rotation=90)
    plt.yticks(np.arange(num_categories), category_list)
    margin_small = 0.1
    margin_large = 0.9
    plt.subplots_adjust(
        left=margin_small, right=margin_large, bottom=margin_small, top=margin_large
    )

    correct_rate = correct / total
    fuzzy_rate = fuzzy / total
    return correct_rate, fuzzy_rate


def general_intersection_over_union(bbox1, bbox2):
    intersection_xtl = max(bbox1['xtl'], bbox2['xtl'])
    intersection_ytl = max(bbox1['ytl'], bbox2['ytl'])
    intersection_xbr = min(bbox1['xbr'], bbox2['xbr'])
    intersection_ybr = min(bbox1['ybr'], bbox2['ybr'])

    intersection_w = intersection_xbr - intersection_xtl
    intersection_h = intersection_ybr - intersection_ytl

    if intersection_w <= 0 or intersection_h <= 0:
        return 0.0

    intersection = intersection_w * intersection_h
    union = (
        (bbox1['width'] * bbox1['height'])
        + (bbox2['width'] * bbox2['height'])
        - intersection
    )

    return intersection / union


def general_overlap(gt_list, pred_list):
    overlap = np.zeros((len(gt_list), len(pred_list)), dtype=np.float32)
    for i, gt in enumerate(gt_list):
        for j, pred in enumerate(pred_list):
            overlap[i, j] = general_intersection_over_union(gt, pred)
    return overlap


def general_tp_fp_fn(gt_list, pred_list, min_overlap, **kwargs):
    overlap = general_overlap(gt_list, pred_list)
    num_gt, num_pred = overlap.shape
    if num_gt == 0:
        tp = 0.0
        fp = num_pred
        fn = 0.0
    elif num_pred == 0:
        tp = 0.0
        fp = 0.0
        fn = num_gt
    else:
        pred_index_list = range(num_pred)
        gt_index_list = np.argmax(overlap, axis=0)
        max_overlap_list = np.max(overlap, axis=0)
        confidence_list = [pred.get('confidence', None) for pred in pred_list]
        assert None not in confidence_list
        zipped = zip(confidence_list, max_overlap_list, pred_index_list, gt_index_list)
        pred_conf_list = [
            (confidence, max_overlap, pred_index, gt_index,)
            for confidence, max_overlap, pred_index, gt_index in zipped
        ]
        pred_conf_list = sorted(pred_conf_list, reverse=True)

        assignment_dict = {}
        for pred_conf, max_overlap, pred_index, gt_index in pred_conf_list:
            if max_overlap > min_overlap:
                if gt_index not in assignment_dict:
                    assignment_dict[gt_index] = pred_index

        tp = len(assignment_dict.keys())
        fp = num_pred - tp
        fn = num_gt - tp
        assert tp >= 0
        assert fp >= 0
        assert fn >= 0

    return tp, fp, fn


def general_get_imageset_gids(ibs, imageset_text, unique=True, **kwargs):
    imageset_id = ibs.get_imageset_imgsetids_from_text(imageset_text)
    test_gid_list = ibs.get_imageset_gids(imageset_id)
    if unique:
        test_gid_list = list(set(test_gid_list))
    return test_gid_list


def general_parse_gt_annots(
    ibs, aid_list, include_parts=True, species_mapping={}, **kwargs
):
    gid_list = ibs.get_annot_gids(aid_list)

    species_set = set([])
    gt_list = []
    for gid, aid in zip(gid_list, aid_list):
        width, height = ibs.get_image_sizes(gid)

        bbox = ibs.get_annot_bboxes(aid)
        theta = ibs.get_annot_thetas(aid)

        # Transformation matrix
        R = vt.rotation_around_bbox_mat3x3(theta, bbox)
        # Get verticies of the annotation polygon
        verts = vt.verts_from_bbox(bbox, close=True)
        # Rotate and transform vertices
        xyz_pts = vt.add_homogenous_coordinate(np.array(verts).T)
        trans_pts = vt.remove_homogenous_coordinate(R.dot(xyz_pts))
        new_verts = np.round(trans_pts).astype(np.int).T.tolist()
        x_points = [pt[0] for pt in new_verts]
        y_points = [pt[1] for pt in new_verts]
        xtl = int(min(x_points))
        xbr = int(max(x_points))
        ytl = int(min(y_points))
        ybr = int(max(y_points))
        bbox = (xtl, ytl, xbr - xtl, ybr - ytl)

        species = ibs.get_annot_species_texts(aid)
        viewpoint = ibs.get_annot_viewpoints(aid)
        interest = ibs.get_annot_interest(aid)
        temp = {
            'gid': gid,
            'aid': aid,
            'xtl': bbox[0] / width,
            'ytl': bbox[1] / height,
            'xbr': (bbox[0] + bbox[2]) / width,
            'ybr': (bbox[1] + bbox[3]) / height,
            'width': bbox[2] / width,
            'height': bbox[3] / height,
            'class': species_mapping.get(species, species),
            'viewpoint': viewpoint,
            'interest': interest,
            'confidence': 1.0,
        }
        species_set.add(temp['class'])
        gt_list.append(temp)

        part_rowid_list = ibs.get_annot_part_rowids(aid)
        if include_parts:
            for part_rowid in part_rowid_list:
                bbox = ibs.get_part_bboxes(part_rowid)
                theta = ibs.get_part_thetas(part_rowid)

                # Transformation matrix
                R = vt.rotation_around_bbox_mat3x3(theta, bbox)
                # Get verticies of the annotation polygon
                verts = vt.verts_from_bbox(bbox, close=True)
                # Rotate and transform vertices
                xyz_pts = vt.add_homogenous_coordinate(np.array(verts).T)
                trans_pts = vt.remove_homogenous_coordinate(R.dot(xyz_pts))
                new_verts = np.round(trans_pts).astype(np.int).T.tolist()
                x_points = [pt[0] for pt in new_verts]
                y_points = [pt[1] for pt in new_verts]
                xtl = int(min(x_points))
                xbr = int(max(x_points))
                ytl = int(min(y_points))
                ybr = int(max(y_points))
                bbox = (xtl, ytl, xbr - xtl, ybr - ytl)

                tag = ibs.get_part_tag_text(part_rowid)

                if tag is None:
                    tag = species
                else:
                    tag = '%s+%s' % (species, tag,)

                temp = {
                    'gid': gid,
                    'aid': aid,
                    'part_id': part_rowid,
                    'xtl': bbox[0] / width,
                    'ytl': bbox[1] / height,
                    'xbr': (bbox[0] + bbox[2]) / width,
                    'ybr': (bbox[1] + bbox[3]) / height,
                    'width': bbox[2] / width,
                    'height': bbox[3] / height,
                    'class': tag,
                    'viewpoint': viewpoint,
                    'interest': interest,
                    'confidence': 1.0,
                }
                species_set.add(temp['class'])
                gt_list.append(temp)

    return gt_list, species_set


def general_parse_gt(ibs, test_gid_list=None, **kwargs):
    if test_gid_list is None:
        test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)

    uuid_list = ibs.get_image_uuids(test_gid_list)
    gid_list = ibs.get_image_gids_from_uuid(uuid_list)

    species_set = set([])
    gt_dict = {}
    for gid, uuid in zip(gid_list, uuid_list):
        aid_list = ibs.get_image_aids(gid)
        gt_list, species_set = general_parse_gt_annots(ibs, aid_list, **kwargs)
        species_set = species_set | species_set
        gt_dict[uuid] = gt_list

    # print('General Parse GT species_set = %r' % (species_set, ))
    return gt_dict


##########################################################################################


def localizer_parse_pred(ibs, test_gid_list=None, species_mapping={}, **kwargs):
    depc = ibs.depc_image

    if 'feature2_algo' not in kwargs:
        kwargs['feature2_algo'] = 'resnet'

    if test_gid_list is None:
        test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
    uuid_list = ibs.get_image_uuids(test_gid_list)

    size_list = ibs.get_image_sizes(test_gid_list)

    # Unsure, but we need to call this multiple times?  Lazy loading bug?
    bboxes_list = depc.get_property(
        'localizations', test_gid_list, 'bboxes', config=kwargs
    )
    # Get actual data
    bboxes_list = depc.get_property(
        'localizations', test_gid_list, 'bboxes', config=kwargs
    )
    thetas_list = depc.get_property(
        'localizations', test_gid_list, 'thetas', config=kwargs
    )
    confss_list = depc.get_property(
        'localizations', test_gid_list, 'confs', config=kwargs
    )
    classs_list = depc.get_property(
        'localizations', test_gid_list, 'classes', config=kwargs
    )

    length_list = [len(bbox_list) for bbox_list in bboxes_list]

    # Establish primitives
    test_gids_list = [
        [test_gid] * length for test_gid, length in zip(test_gid_list, length_list)
    ]
    sizes_list = [[size] * length for size, length in zip(size_list, length_list)]
    keeps_list = [[True] * length for length in length_list]
    features_list = [[None] * length for length in length_list]
    features_lazy_list = [[None] * length for length in length_list]
    viewpoints_list = [[None] * length for length in length_list]
    interests_list = [[None] * length for length in length_list]

    # Get features
    if kwargs.get('features', False):
        features_list = depc.get_property(
            'localizations_features', test_gid_list, 'vector', config=kwargs
        )

    if kwargs.get('features_lazy', False):
        from functools import partial

        def features_lazy_func(gid, offset):
            vector_list = depc.get_property(
                'localizations_features', gid, 'vector', config=kwargs
            )
            vector = vector_list[offset]
            return vector

        features_lazy_list = [
            [
                partial(features_lazy_func, test_gid, test_offset)
                for test_offset in range(length)
            ]
            for test_gid, length in zip(test_gid_list, length_list)
        ]

    # Get species and viewpoints labels
    if kwargs.get('labels', False):
        classs_list = depc.get_property(
            'localizations_labeler', test_gid_list, 'species', config=kwargs
        )
        viewpoints_list = depc.get_property(
            'localizations_labeler', test_gid_list, 'viewpoint', config=kwargs
        )

    # Get updated confidences for boxes
    if kwargs.get('classify', False):
        print('Using alternate classifications')
        # depc.delete_property('localizations_classifier', test_gid_list, config=kwargs)
        confss_list = depc.get_property(
            'localizations_classifier', test_gid_list, 'score', config=kwargs
        )

    # Get updated confidences for boxes
    if kwargs.get('interest', False):
        print('Using alternate AoI interest flags')
        interests_list = depc.get_property(
            'localizations_classifier', test_gid_list, 'score', config=kwargs
        )

    # Reformat results for json
    zipped_list_list = zip(
        keeps_list,
        test_gids_list,
        sizes_list,
        bboxes_list,
        thetas_list,
        confss_list,
        classs_list,
        viewpoints_list,
        interests_list,
        features_list,
        features_lazy_list,
    )
    results_list = [
        [
            {
                'gid': test_gid,
                'xtl': bbox[0] / width,
                'ytl': bbox[1] / height,
                'xbr': (bbox[0] + bbox[2]) / width,
                'ybr': (bbox[1] + bbox[3]) / height,
                'width': bbox[2] / width,
                'height': bbox[3] / height,
                'theta': theta,
                'confidence': conf,
                'class': species_mapping.get(class_, class_),
                'viewpoint': viewpoint,
                'interest': None if interest is None else interest >= 0.84,
                'feature': feature,
                'feature_lazy': feature_lazy,
            }
            for keep_, test_gid, (
                width,
                height,
            ), bbox, theta, conf, class_, viewpoint, interest, feature, feature_lazy in zip(
                *zipped_list
            )
            if keep_
        ]
        for zipped_list in zipped_list_list
    ]

    pred_dict = {
        uuid_: result_list for uuid_, result_list in zip(uuid_list, results_list)
    }
    return pred_dict


def localizer_precision_recall_algo(ibs, samples=SAMPLES, test_gid_list=None, **kwargs):
    if test_gid_list is None:
        test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)

    test_uuid_list = ibs.get_image_uuids(test_gid_list)

    print('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **kwargs)

    print('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

    species_set = kwargs.get('species_set', None)
    if species_set is not None:
        # filter out any prefix ! to denote interest only
        species_set_ = set([species.lstrip('!') for species in species_set])

        dict_list = [
            (gt_dict, 'Ground-Truth'),
            (pred_dict, 'Predictions'),
        ]
        for dict_, dict_tag in dict_list:
            for image_uuid in dict_:
                dict_[image_uuid] = [
                    val
                    for val in dict_[image_uuid]
                    if val.get('class', None) in species_set_
                ]

    values = localizer_tp_fp(test_uuid_list, gt_dict, pred_dict, **kwargs)
    conf_list, tp_list, fp_list, total = values

    conf_list_ = [-1.0, -1.0]
    pr_list = [1.0, 0.0]
    re_list = [0.0, 1.0]
    for conf, tp, fp in zip(conf_list, tp_list, fp_list):
        try:
            pr = tp / (tp + fp)
            re = tp / total
        except ZeroDivisionError:
            continue
        conf_list_.append(conf)
        pr_list.append(pr)
        re_list.append(re)

    return conf_list_, pr_list, re_list


def localizer_assign(gt_list, pred, min_overlap):
    best_overlap = min_overlap
    best_index = None
    for index, gt in enumerate(gt_list):
        if gt['class'] != pred['class']:
            continue

        overlap = general_intersection_over_union(gt, pred)
        if overlap < best_overlap:
            continue

        best_overlap = overlap
        best_index = index

    if best_index is None:
        best_overlap = None

    return best_index, best_overlap


def localizer_assignments(pred_list, gt_list, gt_list_=[], min_overlap=0.5):
    pred_list = sorted(pred_list, key=lambda pred: pred['confidence'], reverse=True)

    match_list = []
    for pred in pred_list:
        flag = False

        match_index, best_overlap = localizer_assign(gt_list, pred, min_overlap)
        match_index_, best_overlap_ = localizer_assign(gt_list_, pred, min_overlap)

        if match_index is not None:
            flag = True
            del gt_list[match_index]
        elif match_index_ is not None:
            flag = None

        if flag is not None:
            match_list += [(pred['confidence'], flag, match_index, best_overlap)]

    return match_list


def localizer_tp_fp(uuid_list, gt_dict, pred_dict, min_overlap=0.5, **kwargs):
    total = 0.0

    interest_species_set = set([])
    species_set = kwargs.get('species_set', None)
    if species_set is not None:
        for species in species_set:
            if species.startswith('!'):
                species = species.lstrip('!')
                interest_species_set.add(species)

    match_list = []
    for image_uuid in uuid_list:
        gt_list = []
        gt_list_ = []
        pred_list = pred_dict[image_uuid]

        for gt in gt_dict[image_uuid]:
            species = gt['class']
            interest = gt['interest']
            if species in interest_species_set and not interest:
                gt_list_.append(gt)
            else:
                gt_list.append(gt)

        total += len(gt_list)

        # Match predictions
        match_list_ = localizer_assignments(pred_list, gt_list, gt_list_, min_overlap)
        for match_ in match_list_:
            match_list.append(match_)

    # sort matches by confidence from high to low
    match_list = sorted(match_list, key=lambda match: match[0], reverse=True)

    conf_list = []
    tp_list = []
    fp_list = []

    tp_counter = 0
    fp_counter = 0
    for conf, flag, index, overlap in match_list:
        if flag:
            tp_counter += 1
        else:
            fp_counter += 1
        conf_list.append(conf)
        tp_list.append(tp_counter)
        fp_list.append(fp_counter)

    # print('\t tps  [:10]     : %r' % (tp_list[:10], ))
    # print('\t fps  [:10]     : %r' % (fp_list[:10], ))
    # print('\t con  [:10]     : %r' % (conf_list[:10], ))
    # print('\t tps [-10:]     : %r' % (tp_list[-10:], ))
    # print('\t fps [-10:]     : %r' % (fp_list[-10:], ))
    # print('\t con [-10:]     : %r' % (conf_list[-10:], ))
    # print('\t num_annotations: %r' % (total, ))

    return conf_list, tp_list, fp_list, total


def localizer_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing Precision-Recall for: %r' % (label,))
    conf_list, pr_list, re_list = localizer_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def _ignore_filter_identity_func(*args, **kwargs):
    return False


def localizer_iou_recall_algo(
    ibs, samples=100, test_gid_list=None, ignore_filter_func=None, **kwargs
):

    assert 'min_overlap' not in kwargs

    if test_gid_list is None:
        test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)

    test_uuid_list = ibs.get_image_uuids(test_gid_list)

    if ignore_filter_func is None:
        ignore_filter_func = _ignore_filter_identity_func

    print('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **kwargs)

    print('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

    species_set = kwargs.get('species_set', None)
    if species_set is not None:
        # filter out any prefix ! to denote interest only
        species_set_ = set([species.lstrip('!') for species in species_set])

        dict_list = [
            (gt_dict, 'Ground-Truth'),
            (pred_dict, 'Predictions'),
        ]
        for dict_, dict_tag in dict_list:
            for image_uuid in dict_:
                temp = []
                for val in dict_[image_uuid]:
                    if val.get('class', None) not in species_set_:
                        continue
                    if ignore_filter_func(ibs, val):
                        continue
                    temp.append(val)
                dict_[image_uuid] = temp

    target = (1.0, 1.0)
    iou_list = [_ / float(samples) for _ in range(0, int(samples) + 1)]

    conf_list_ = []
    iou_list_ = []
    recall_list = []
    for iou in tqdm.tqdm(iou_list):
        values = localizer_tp_fp(
            test_uuid_list, gt_dict, pred_dict, min_overlap=iou, **kwargs
        )
        conf_list, tp_list, fp_list, total = values

        conf_list_ = []
        pr_list = []
        re_list = []
        for conf, tp, fp in zip(conf_list, tp_list, fp_list):
            try:
                pr = tp / (tp + fp)
                re = tp / total
            except ZeroDivisionError:
                continue
            conf_list_.append(conf)
            pr_list.append(pr)
            re_list.append(re)

        best_tup = general_identify_operating_point(
            conf_list, re_list, pr_list, target=target
        )
        best_conf_list, best_re_list, best_pr_list, best_length = best_tup
        if len(best_conf_list) > 1:
            print('WARNING: Multiple best operating points found %r' % (best_conf_list,))
        assert len(best_conf_list) > 0

        best_re_index = np.argmax(best_re_list)
        best_re = best_re_list[best_re_index]
        best_conf = best_conf_list[best_re_index]

        conf_list_.append(best_conf)
        iou_list_.append(iou)
        recall_list.append(best_re)

    return conf_list_, iou_list_, recall_list


def localizer_iou_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing IoU-Recall for: %r' % (label,))
    conf_list, iou_list, recall_list = localizer_iou_recall_algo(ibs, **kwargs)
    return general_area_best_conf(
        conf_list, iou_list, recall_list, interpolate=False, **kwargs
    )


# def localizer_iou_precision_algo_plot(ibs, **kwargs):
#     label = kwargs['label']
#     print('Processing Precision-Recall for: %r' % (label, ))
#     conf_list, iou_list, pr_list, re_list = localizer_iou_precision_recall_algo(ibs, **kwargs)
# return general_area_best_conf(conf_list, iou_list, re_list, **kwargs)


def localizer_confusion_matrix_algo_plot(
    ibs, label=None, target_conf=None, test_gid_list=None, **kwargs
):
    if test_gid_list is None:
        test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)

    test_uuid_list = ibs.get_image_uuids(test_gid_list)

    print('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **kwargs)

    print('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

    species_set = kwargs.get('species_set', None)
    if species_set is not None:
        # filter out any prefix ! to denote interest only
        species_set_ = set([species.lstrip('!') for species in species_set])

        dict_list = [
            (gt_dict, 'Ground-Truth'),
            (pred_dict, 'Predictions'),
        ]
        for dict_, dict_tag in dict_list:
            for image_uuid in dict_:
                dict_[image_uuid] = [
                    val
                    for val in dict_[image_uuid]
                    if val.get('class', None) in species_set_
                ]

    values = localizer_tp_fp(test_uuid_list, gt_dict, pred_dict, **kwargs)
    conf_list, tp_list, fp_list, total = values

    best_conf = None
    best_accuracy = None
    best_args = None
    for conf, tp, fp in sorted(zip(conf_list, tp_list, fp_list)):
        fn = total - tp
        accuracy = tp / (tp + fp + fn)

        if target_conf is None:
            if best_accuracy is None or accuracy > best_accuracy:
                best_conf = conf
                best_accuracy = accuracy
                best_args = (tp, fp, fn)
        else:
            if target_conf <= conf:
                best_conf = conf
                best_accuracy = accuracy
                best_args = (tp, fp, fn)
                break

    try:
        assert None not in [best_conf, best_accuracy, best_args]
    except AssertionError:
        ut.embed()
        return np.nan, (np.nan, None)

    print(
        'Processing Confusion Matrix for: %r (Conf = %0.02f, Accuracy = %0.02f)'
        % (label, best_conf, best_accuracy,)
    )
    tp, fp, fn = best_args

    label_list = []
    prediction_list = []
    for _ in range(int(tp)):
        label_list.append('positive')
        prediction_list.append('positive')
    for _ in range(int(fp)):
        label_list.append('negative')
        prediction_list.append('positive')
    for _ in range(int(fn)):
        label_list.append('positive')
        prediction_list.append('negative')

    category_list = ['positive', 'negative']
    category_mapping = {
        'positive': 0,
        'negative': 1,
    }
    values = general_confusion_matrix_algo(
        label_list, prediction_list, category_list, category_mapping, size=20, **kwargs
    )
    return best_conf, values


@register_ibs_method
def localizer_precision_recall(
    ibs, config_dict=None, output_path=None, test_gid_list=None, **kwargs
):
    if config_dict is None:
        if test_gid_list is not None:
            print('Using %d test gids' % (len(test_gid_list),))

        # species_mapping = {  # NOQA
        #     'giraffe_masai'       : 'giraffe',
        #     'giraffe_reticulated' : 'giraffe',
        #     'zebra_grevys'        : 'zebra',
        #     'zebra_plains'        : 'zebra',
        # }

        config_dict = {
            # 'seaturtle': (
            #     [
            #         {'label': 'Sea Turtle',        'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['turtle_green', 'turtle_hawksbill'])},
            #         {'label': 'Sea Turtle Heads',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['turtle_green+head', 'turtle_hawksbill+head'])},
            #         {'label': 'Green',             'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['turtle_green'])},
            #         {'label': 'Green Heads',       'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['turtle_green+head'])},
            #         {'label': 'Hawksbill',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['turtle_hawksbill'])},
            #         {'label': 'Hawksbill Heads',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['turtle_hawksbill+head'])},
            #     ],
            #     {'BEST_INDEX': 0},
            # ),
            # '!seaturtle': (
            #     [
            #         {'label': '! Sea Turtle',        'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['!turtle_green', '!turtle_hawksbill'])},
            #         {'label': '! Sea Turtle Heads',  'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['!turtle_green+head', '!turtle_hawksbill+head'])},
            #         {'label': '! Green',             'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['!turtle_green'])},
            #         {'label': '! Green Heads',       'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['!turtle_green+head'])},
            #         {'label': '! Hawksbill',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['!turtle_hawksbill'])},
            #         {'label': '! Hawksbill Heads',   'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.2, 'species_set' : set(['!turtle_hawksbill+head'])},
            #     ],
            #     {'BEST_INDEX': 0},
            # ),
            # 'hawksbills': (
            #     [
            #         {'label': 'Hawksbill NMS 0%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['turtle_hawksbill'])},
            #         {'label': 'Hawksbill NMS 10%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['turtle_hawksbill'])},
            #         {'label': 'Hawksbill NMS 20%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['turtle_hawksbill'])},
            #         {'label': 'Hawksbill NMS 30%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['turtle_hawksbill'])},
            #         {'label': 'Hawksbill NMS 40%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['turtle_hawksbill'])},
            #         {'label': 'Hawksbill NMS 50%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['turtle_hawksbill'])},
            #         {'label': 'Hawksbill NMS 60%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['turtle_hawksbill'])},
            #         {'label': 'Hawksbill NMS 70%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['turtle_hawksbill'])},
            #         {'label': 'Hawksbill NMS 80%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['turtle_hawksbill'])},
            #         {'label': 'Hawksbill NMS 90%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['turtle_hawksbill'])},
            #         {'label': 'Hawksbill NMS 100%',        'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['turtle_hawksbill'])},
            #     ],
            #     {},
            # ),
            # 'hawsbills+heads': (
            #     [
            #         {'label': 'Hawksbill Head NMS 0%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['turtle_hawksbill+head'])},
            #         {'label': 'Hawksbill Head NMS 10%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['turtle_hawksbill+head'])},
            #         {'label': 'Hawksbill Head NMS 20%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['turtle_hawksbill+head'])},
            #         {'label': 'Hawksbill Head NMS 30%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['turtle_hawksbill+head'])},
            #         {'label': 'Hawksbill Head NMS 40%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['turtle_hawksbill+head'])},
            #         {'label': 'Hawksbill Head NMS 50%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['turtle_hawksbill+head'])},
            #         {'label': 'Hawksbill Head NMS 60%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['turtle_hawksbill+head'])},
            #         {'label': 'Hawksbill Head NMS 70%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['turtle_hawksbill+head'])},
            #         {'label': 'Hawksbill Head NMS 80%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['turtle_hawksbill+head'])},
            #         {'label': 'Hawksbill Head NMS 90%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['turtle_hawksbill+head'])},
            #         {'label': 'Hawksbill Head NMS 100%',        'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'seaturtle', 'weight_filepath' : 'seaturtle', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['turtle_hawksbill+head'])},
            #     ],
            #     {},
            # ),
            # 'hammerhead': (
            #     [
            #         {'label': 'Hammerhead NMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['shark_hammerhead'])},
            #         {'label': 'Hammerhead NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['shark_hammerhead'])},
            #         {'label': 'Hammerhead NMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['shark_hammerhead'])},
            #         {'label': 'Hammerhead NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['shark_hammerhead'])},
            #         {'label': 'Hammerhead NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['shark_hammerhead'])},
            #         {'label': 'Hammerhead NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['shark_hammerhead'])},
            #         {'label': 'Hammerhead NMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['shark_hammerhead'])},
            #         {'label': 'Hammerhead NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['shark_hammerhead'])},
            #         {'label': 'Hammerhead NMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['shark_hammerhead'])},
            #         {'label': 'Hammerhead NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['shark_hammerhead'])},
            #         {'label': 'Hammerhead NMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['shark_hammerhead'])},
            #     ],
            #     {},
            # ),
            # '!hammerhead': (
            #     [
            #         {'label': 'Hammerhead NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['shark_hammerhead'])},
            #         {'label': 'Hammerhead ! NMS 0%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['!shark_hammerhead'])},
            #         {'label': 'Hammerhead ! NMS 10%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['!shark_hammerhead'])},
            #         {'label': 'Hammerhead ! NMS 20%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['!shark_hammerhead'])},
            #         {'label': 'Hammerhead ! NMS 30%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['!shark_hammerhead'])},
            #         {'label': 'Hammerhead ! NMS 40%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['!shark_hammerhead'])},
            #         {'label': 'Hammerhead ! NMS 50%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['!shark_hammerhead'])},
            #         {'label': 'Hammerhead ! NMS 60%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['!shark_hammerhead'])},
            #         {'label': 'Hammerhead ! NMS 70%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['!shark_hammerhead'])},
            #         {'label': 'Hammerhead ! NMS 80%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['!shark_hammerhead'])},
            #         {'label': 'Hammerhead ! NMS 90%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['!shark_hammerhead'])},
            #         {'label': 'Hammerhead ! NMS 100%',        'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'hammerhead', 'weight_filepath' : 'hammerhead', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['!shark_hammerhead'])},
            #     ],
            #     {'offset_color': 1},
            # ),
            # 'ggr2-giraffe-lightnet': (
            #     [
            #         {'label': 'Giraffe NMS 0%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 10%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.10, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 20%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.20, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 30%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.30, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 40%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.40, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 50%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.50, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 60%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.60, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 70%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.70, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 80%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.80, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 90%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.90, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 100%',        'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 1.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #     ],
            #     {},
            # ),
            # 'ggr2-zebra-lightnet': (
            #     [
            #         {'label': 'Zebra NMS 0%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 10%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.10, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 20%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.20, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 30%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.30, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 40%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.40, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 50%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.50, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 60%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.60, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 70%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.70, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 80%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.80, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 90%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.90, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 100%',        'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 1.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #     ],
            #     {},
            # ),
            # 'ggr2-!giraffe-lightnet': (
            #     [
            #         {'label': 'Giraffe ! NMS 0%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 10%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.10, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 20%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.20, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 30%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.30, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 40%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.40, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 50%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.50, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 60%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.60, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 70%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.70, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 80%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.80, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 90%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.90, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 100%',        'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 1.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #     ],
            #     {},
            # ),
            # 'ggr2-!zebra-lightnet': (
            #     [
            #         {'label': 'Zebra ! NMS 0%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 10%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.10, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 20%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.20, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 30%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.30, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 40%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.40, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 50%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.50, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 60%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.60, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 70%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.70, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 80%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.80, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 90%',         'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 0.90, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 100%',        'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'ggr2', 'weight_filepath' : 'ggr2', 'nms': True, 'nms_thresh': 1.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #     ],
            #     {},
            # ),
            # 'ggr2-giraffe-azure': (
            #     [
            #         {'label': 'Giraffe NMS 0%',          'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 10%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.10, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 20%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.20, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 30%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.30, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 40%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.40, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 50%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.50, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 60%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.60, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 70%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.70, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 80%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.80, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 90%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.90, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #         {'label': 'Giraffe NMS 100%',        'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 1.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['giraffe'])},
            #     ],
            #     {},
            # ),
            # 'ggr2-zebra-azure': (
            #     [
            #         {'label': 'Zebra NMS 0%',          'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 10%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.10, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 20%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.20, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 30%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.30, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 40%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.40, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 50%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.50, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 60%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.60, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 70%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.70, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 80%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.80, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 90%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.90, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #         {'label': 'Zebra NMS 100%',        'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 1.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['zebra'])},
            #     ],
            #     {},
            # ),
            # 'ggr2-!giraffe-azure': (
            #     [
            #         {'label': 'Giraffe ! NMS 0%',          'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 10%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.10, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 20%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.20, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 30%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.30, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 40%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.40, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 50%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.50, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 60%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.60, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 70%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.70, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 80%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.80, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 90%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.90, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #         {'label': 'Giraffe ! NMS 100%',        'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 1.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!giraffe'])},
            #     ],
            #     {},
            # ),
            # 'ggr2-!zebra-azure': (
            #     [
            #         {'label': 'Zebra ! NMS 0%',          'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 10%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.10, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 20%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.20, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 30%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.30, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 40%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.40, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 50%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.50, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 60%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.60, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 70%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.70, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 80%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.80, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 90%',         'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 0.90, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #         {'label': 'Zebra ! NMS 100%',        'grid' : False, 'algo': 'azure', 'config_filepath' : None, 'weight_filepath' : None, 'nms': True, 'nms_thresh': 1.00, 'test_gid_list': test_gid_list, 'species_mapping': species_mapping, 'species_set': set(['!zebra'])},
            #     ],
            #     {},
            # ),
            # 'lynx': (
            #     [
            #         {'label': 'Lynx NMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['lynx'])},
            #         {'label': 'Lynx NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['lynx'])},
            #         {'label': 'Lynx NMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['lynx'])},
            #         {'label': 'Lynx NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['lynx'])},
            #         {'label': 'Lynx NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['lynx'])},
            #         {'label': 'Lynx NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['lynx'])},
            #         {'label': 'Lynx NMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['lynx'])},
            #         {'label': 'Lynx NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['lynx'])},
            #         {'label': 'Lynx NMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['lynx'])},
            #         {'label': 'Lynx NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['lynx'])},
            #         {'label': 'Lynx NMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['lynx'])},
            #     ],
            #     {},
            # ),
            # 'jaguar': (
            #     [
            #         {'label': 'Jaguar NMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['jaguar'])},
            #         {'label': 'Jaguar NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['jaguar'])},
            #         {'label': 'Jaguar NMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['jaguar'])},
            #         {'label': 'Jaguar NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['jaguar'])},
            #         {'label': 'Jaguar NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['jaguar'])},
            #         {'label': 'Jaguar NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['jaguar'])},
            #         {'label': 'Jaguar NMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['jaguar'])},
            #         {'label': 'Jaguar NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['jaguar'])},
            #         {'label': 'Jaguar NMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['jaguar'])},
            #         {'label': 'Jaguar NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['jaguar'])},
            #         {'label': 'Jaguar NMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['jaguar'])},
            #     ],
            #     {},
            # ),
            # '!jaguar': (
            #     [
            #         {'label': 'Jaguar NMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['!jaguar'])},
            #         {'label': 'Jaguar NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['!jaguar'])},
            #         {'label': 'Jaguar NMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['!jaguar'])},
            #         {'label': 'Jaguar NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['!jaguar'])},
            #         {'label': 'Jaguar NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['!jaguar'])},
            #         {'label': 'Jaguar NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['!jaguar'])},
            #         {'label': 'Jaguar NMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['!jaguar'])},
            #         {'label': 'Jaguar NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['!jaguar'])},
            #         {'label': 'Jaguar NMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['!jaguar'])},
            #         {'label': 'Jaguar NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['!jaguar'])},
            #         {'label': 'Jaguar NMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'jaguar_v2', 'weight_filepath' : 'jaguar_v2', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['!jaguar'])},
            #     ],
            #     {},
            # ),
            # 'manta': (
            #     [
            #         {'label': 'Manta NMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['manta_ray_giant'])},
            #         {'label': 'Manta NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['manta_ray_giant'])},
            #         {'label': 'Manta NMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['manta_ray_giant'])},
            #         {'label': 'Manta NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['manta_ray_giant'])},
            #         {'label': 'Manta NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['manta_ray_giant'])},
            #         {'label': 'Manta NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['manta_ray_giant'])},
            #         {'label': 'Manta NMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['manta_ray_giant'])},
            #         {'label': 'Manta NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['manta_ray_giant'])},
            #         {'label': 'Manta NMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['manta_ray_giant'])},
            #         {'label': 'Manta NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['manta_ray_giant'])},
            #         {'label': 'Manta NMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['manta_ray_giant'])},
            #     ],
            #     {},
            # ),
            # '!manta': (
            #     [
            #         {'label': 'Manta NMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['!manta_ray_giant'])},
            #         {'label': 'Manta NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['!manta_ray_giant'])},
            #         {'label': 'Manta NMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['!manta_ray_giant'])},
            #         {'label': 'Manta NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['!manta_ray_giant'])},
            #         {'label': 'Manta NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['!manta_ray_giant'])},
            #         {'label': 'Manta NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['!manta_ray_giant'])},
            #         {'label': 'Manta NMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['!manta_ray_giant'])},
            #         {'label': 'Manta NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['!manta_ray_giant'])},
            #         {'label': 'Manta NMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['!manta_ray_giant'])},
            #         {'label': 'Manta NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['!manta_ray_giant'])},
            #         {'label': 'Manta NMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'manta', 'weight_filepath' : 'manta', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['!manta_ray_giant'])},
            #     ],
            #     {},
            # ),
            # 'giraffe': (
            #     [
            #         {'label': 'Giraffe NMS 10%',                 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['giraffe_masai', 'giraffe_reticulated'])},
            #         {'label': 'Giraffe NMS 30%',                 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['giraffe_masai', 'giraffe_reticulated'])},
            #         {'label': 'Giraffe NMS 50%',                 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['giraffe_masai', 'giraffe_reticulated'])},
            #         {'label': 'Giraffe NMS 70%',                 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['giraffe_masai', 'giraffe_reticulated'])},
            #         {'label': 'Giraffe NMS 90%',                 'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['giraffe_masai', 'giraffe_reticulated'])},
            #         {'label': 'Masai Giraffe NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['giraffe_masai'])},
            #         {'label': 'Masai Giraffe NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['giraffe_masai'])},
            #         {'label': 'Masai Giraffe NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['giraffe_masai'])},
            #         {'label': 'Masai Giraffe NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['giraffe_masai'])},
            #         {'label': 'Masai Giraffe NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['giraffe_masai'])},
            #         {'label': 'Reticulated Giraffe NMS 10%',     'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['giraffe_reticulated'])},
            #         {'label': 'Reticulated Giraffe NMS 30%',     'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['giraffe_reticulated'])},
            #         {'label': 'Reticulated Giraffe NMS 50%',     'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['giraffe_reticulated'])},
            #         {'label': 'Reticulated Giraffe NMS 70%',     'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['giraffe_reticulated'])},
            #         {'label': 'Reticulated Giraffe NMS 90%',     'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'giraffe_v1', 'weight_filepath' : 'giraffe_v1', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['giraffe_reticulated'])},
            #     ],
            #     {},
            # ),
            # 'spotted_skunk_v0': (
            #     [
            #         {'label': 'Spotted Skunk NMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['skunk_spotted'])},
            #     ],
            #     {},
            # ),
            # '!spotted_skunk_v0': (
            #     [
            #         {'label': 'Spotted Skunk NMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['!skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['!skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['!skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['!skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['!skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['!skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['!skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['!skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['!skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['!skunk_spotted'])},
            #         {'label': 'Spotted Skunk NMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_skunk_v0', 'weight_filepath' : 'spotted_skunk_v0', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['!skunk_spotted'])},
            #     ],
            #     {},
            # ),
            # 'nassau_grouper_v0': (
            #     [
            #         {'label': 'Nassau Grouper NMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['grouper_nassau'])},
            #         {'label': 'Nassau Grouper NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['grouper_nassau'])},
            #         {'label': 'Nassau Grouper NMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['grouper_nassau'])},
            #         {'label': 'Nassau Grouper NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['grouper_nassau'])},
            #         {'label': 'Nassau Grouper NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['grouper_nassau'])},
            #         {'label': 'Nassau Grouper NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['grouper_nassau'])},
            #         {'label': 'Nassau Grouper NMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['grouper_nassau'])},
            #         {'label': 'Nassau Grouper NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['grouper_nassau'])},
            #         {'label': 'Nassau Grouper NMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['grouper_nassau'])},
            #         {'label': 'Nassau Grouper NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['grouper_nassau'])},
            #         {'label': 'Nassau Grouper NMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['grouper_nassau'])},
            #     ],
            #     {},
            # ),
            # '!nassau_grouper_v0': (
            #     [
            #         {'label': 'Nassau Grouper! NMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['!grouper_nassau'])},
            #         {'label': 'Nassau Grouper! NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['!grouper_nassau'])},
            #         {'label': 'Nassau Grouper! NMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['!grouper_nassau'])},
            #         {'label': 'Nassau Grouper! NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['!grouper_nassau'])},
            #         {'label': 'Nassau Grouper! NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['!grouper_nassau'])},
            #         {'label': 'Nassau Grouper! NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['!grouper_nassau'])},
            #         {'label': 'Nassau Grouper! NMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['!grouper_nassau'])},
            #         {'label': 'Nassau Grouper! NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['!grouper_nassau'])},
            #         {'label': 'Nassau Grouper! NMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['!grouper_nassau'])},
            #         {'label': 'Nassau Grouper! NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['!grouper_nassau'])},
            #         {'label': 'Nassau Grouper! NMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'nassau_grouper_v0', 'weight_filepath' : 'nassau_grouper_v0', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['!grouper_nassau'])},
            #     ],
            #     {},
            # ),
            # 'spotted_dolphin_v0': (
            #     [
            #         {'label': 'Spotted DolphinNMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['dolphin_spotted'])},
            #         {'label': 'Spotted DolphinNMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['dolphin_spotted'])},
            #         {'label': 'Spotted DolphinNMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['dolphin_spotted'])},
            #         {'label': 'Spotted DolphinNMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['dolphin_spotted'])},
            #         {'label': 'Spotted DolphinNMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['dolphin_spotted'])},
            #         {'label': 'Spotted DolphinNMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['dolphin_spotted'])},
            #         {'label': 'Spotted DolphinNMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['dolphin_spotted'])},
            #         {'label': 'Spotted DolphinNMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['dolphin_spotted'])},
            #         {'label': 'Spotted DolphinNMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['dolphin_spotted'])},
            #         {'label': 'Spotted DolphinNMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['dolphin_spotted'])},
            #         {'label': 'Spotted DolphinNMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['dolphin_spotted'])},
            #     ],
            #     {},
            # ),
            # '!spotted_dolphin_v0': (
            #     [
            #         {'label': 'Spotted Dolphin! NMS 0%',            'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.00, 'species_set' : set(['!dolphin_spotted'])},
            #         {'label': 'Spotted Dolphin! NMS 10%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.10, 'species_set' : set(['!dolphin_spotted'])},
            #         {'label': 'Spotted Dolphin! NMS 20%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.20, 'species_set' : set(['!dolphin_spotted'])},
            #         {'label': 'Spotted Dolphin! NMS 30%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.30, 'species_set' : set(['!dolphin_spotted'])},
            #         {'label': 'Spotted Dolphin! NMS 40%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.40, 'species_set' : set(['!dolphin_spotted'])},
            #         {'label': 'Spotted Dolphin! NMS 50%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.50, 'species_set' : set(['!dolphin_spotted'])},
            #         {'label': 'Spotted Dolphin! NMS 60%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.60, 'species_set' : set(['!dolphin_spotted'])},
            #         {'label': 'Spotted Dolphin! NMS 70%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.70, 'species_set' : set(['!dolphin_spotted'])},
            #         {'label': 'Spotted Dolphin! NMS 80%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.80, 'species_set' : set(['!dolphin_spotted'])},
            #         {'label': 'Spotted Dolphin! NMS 90%',           'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 0.90, 'species_set' : set(['!dolphin_spotted'])},
            #         {'label': 'Spotted Dolphin! NMS 100%',          'grid' : False, 'algo': 'lightnet', 'config_filepath' : 'spotted_dolphin_v0', 'weight_filepath' : 'spotted_dolphin_v0', 'nms': True, 'nms_thresh': 1.00, 'species_set' : set(['!dolphin_spotted'])},
            #     ],
            #     {},
            # ),
            'seadragon_weedy_v1': (
                [
                    {
                        'label': 'Weedy Body NMS 0%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.00,
                        'species_set': set(['seadragon_leafy']),
                    },
                    {
                        'label': 'Weedy Body NMS 10%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.10,
                        'species_set': set(['seadragon_leafy']),
                    },
                    {
                        'label': 'Weedy Body NMS 20%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.20,
                        'species_set': set(['seadragon_leafy']),
                    },
                    {
                        'label': 'Weedy Body NMS 30%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.30,
                        'species_set': set(['seadragon_leafy']),
                    },
                    {
                        'label': 'Weedy Body NMS 40%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.40,
                        'species_set': set(['seadragon_leafy']),
                    },
                    {
                        'label': 'Weedy Body NMS 50%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.50,
                        'species_set': set(['seadragon_leafy']),
                    },
                    {
                        'label': 'Weedy Body NMS 60%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.60,
                        'species_set': set(['seadragon_leafy']),
                    },
                    {
                        'label': 'Weedy Body NMS 70%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.70,
                        'species_set': set(['seadragon_leafy']),
                    },
                    {
                        'label': 'Weedy Body NMS 80%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.80,
                        'species_set': set(['seadragon_leafy']),
                    },
                    {
                        'label': 'Weedy Body NMS 90%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.90,
                        'species_set': set(['seadragon_leafy']),
                    },
                    {
                        'label': 'Weedy Body NMS 100%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 1.00,
                        'species_set': set(['seadragon_leafy']),
                    },
                ],
                {},
            ),
            'seadragon_leafy_v1': (
                [
                    {
                        'label': 'Leafy Body NMS 0%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.00,
                        'species_set': set(['seadragon_weedy']),
                    },
                    {
                        'label': 'Leafy Body NMS 10%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.10,
                        'species_set': set(['seadragon_weedy']),
                    },
                    {
                        'label': 'Leafy Body NMS 20%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.20,
                        'species_set': set(['seadragon_weedy']),
                    },
                    {
                        'label': 'Leafy Body NMS 30%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.30,
                        'species_set': set(['seadragon_weedy']),
                    },
                    {
                        'label': 'Leafy Body NMS 40%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.40,
                        'species_set': set(['seadragon_weedy']),
                    },
                    {
                        'label': 'Leafy Body NMS 50%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.50,
                        'species_set': set(['seadragon_weedy']),
                    },
                    {
                        'label': 'Leafy Body NMS 60%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.60,
                        'species_set': set(['seadragon_weedy']),
                    },
                    {
                        'label': 'Leafy Body NMS 70%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.70,
                        'species_set': set(['seadragon_weedy']),
                    },
                    {
                        'label': 'Leafy Body NMS 80%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.80,
                        'species_set': set(['seadragon_weedy']),
                    },
                    {
                        'label': 'Leafy Body NMS 90%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.90,
                        'species_set': set(['seadragon_weedy']),
                    },
                    {
                        'label': 'Leafy Body NMS 100%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 1.00,
                        'species_set': set(['seadragon_weedy']),
                    },
                ],
                {},
            ),
            'seadragon_weedy_head_v1': (
                [
                    {
                        'label': 'Weedy Head NMS 0%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.00,
                        'species_set': set(['seadragon_leafy+head']),
                    },
                    {
                        'label': 'Weedy Head NMS 10%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.10,
                        'species_set': set(['seadragon_leafy+head']),
                    },
                    {
                        'label': 'Weedy Head NMS 20%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.20,
                        'species_set': set(['seadragon_leafy+head']),
                    },
                    {
                        'label': 'Weedy Head NMS 30%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.30,
                        'species_set': set(['seadragon_leafy+head']),
                    },
                    {
                        'label': 'Weedy Head NMS 40%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.40,
                        'species_set': set(['seadragon_leafy+head']),
                    },
                    {
                        'label': 'Weedy Head NMS 50%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.50,
                        'species_set': set(['seadragon_leafy+head']),
                    },
                    {
                        'label': 'Weedy Head NMS 60%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.60,
                        'species_set': set(['seadragon_leafy+head']),
                    },
                    {
                        'label': 'Weedy Head NMS 70%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.70,
                        'species_set': set(['seadragon_leafy+head']),
                    },
                    {
                        'label': 'Weedy Head NMS 80%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.80,
                        'species_set': set(['seadragon_leafy+head']),
                    },
                    {
                        'label': 'Weedy Head NMS 90%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.90,
                        'species_set': set(['seadragon_leafy+head']),
                    },
                    {
                        'label': 'Weedy Head NMS 100%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 1.00,
                        'species_set': set(['seadragon_leafy+head']),
                    },
                ],
                {},
            ),
            'seadragon_leafy_head_v1': (
                [
                    {
                        'label': 'Leafy Head NMS 0%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.00,
                        'species_set': set(['seadragon_weedy+head']),
                    },
                    {
                        'label': 'Leafy Head NMS 10%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.10,
                        'species_set': set(['seadragon_weedy+head']),
                    },
                    {
                        'label': 'Leafy Head NMS 20%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.20,
                        'species_set': set(['seadragon_weedy+head']),
                    },
                    {
                        'label': 'Leafy Head NMS 30%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.30,
                        'species_set': set(['seadragon_weedy+head']),
                    },
                    {
                        'label': 'Leafy Head NMS 40%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.40,
                        'species_set': set(['seadragon_weedy+head']),
                    },
                    {
                        'label': 'Leafy Head NMS 50%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.50,
                        'species_set': set(['seadragon_weedy+head']),
                    },
                    {
                        'label': 'Leafy Head NMS 60%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.60,
                        'species_set': set(['seadragon_weedy+head']),
                    },
                    {
                        'label': 'Leafy Head NMS 70%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.70,
                        'species_set': set(['seadragon_weedy+head']),
                    },
                    {
                        'label': 'Leafy Head NMS 80%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.80,
                        'species_set': set(['seadragon_weedy+head']),
                    },
                    {
                        'label': 'Leafy Head NMS 90%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 0.90,
                        'species_set': set(['seadragon_weedy+head']),
                    },
                    {
                        'label': 'Leafy Head NMS 100%',
                        'grid': False,
                        'algo': 'lightnet',
                        'config_filepath': 'seadragon_v1',
                        'weight_filepath': 'seadragon_v1',
                        'nms': True,
                        'nms_thresh': 1.00,
                        'species_set': set(['seadragon_weedy+head']),
                    },
                ],
                {},
            ),
        }

    for config_key in config_dict:
        config_list, config = config_dict[config_key]

        for key in kwargs:
            config[key] = kwargs[key]

        # Backwards compatibility hack
        if test_gid_list is not None:
            for config_ in config_list:
                if 'test_gid_list' not in config_:
                    config_['test_gid_list'] = test_gid_list

        ibs.localizer_precision_recall_algo_display(
            config_list, config_tag=config_key, output_path=output_path, **config
        )


@register_ibs_method
def localizer_precision_recall_algo_display(
    ibs,
    config_list,
    config_tag='',
    min_overlap=0.5,
    figsize=(40, 9),
    target_recall=0.8,
    BEST_INDEX=None,
    offset_color=0,
    write_images=False,
    plot_point=True,
    output_path=None,
    plot_iou_recall=True,
    **kwargs,
):
    import matplotlib.pyplot as plt
    import wbia.plottool as pt

    if output_path is None:
        output_path = abspath(expanduser(join('~', 'Desktop')))

    color_list_ = []
    for _ in range(offset_color):
        color_list_ += [(0.2, 0.2, 0.2)]

    color_list = pt.distinct_colors(len(config_list) - len(color_list_), randomize=False)
    color_list = color_list_ + color_list

    fig_ = plt.figure(figsize=figsize, dpi=400)

    ######################################################################################

    axes_ = plt.subplot(141)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall (Ground-Truth IOU >= %0.02f)' % (min_overlap,))
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])

    ret_list = [
        localizer_precision_recall_algo_plot(
            ibs,
            color=color,
            min_overlap=min_overlap,
            plot_point=plot_point,
            target_recall=target_recall,
            **config,
        )
        for color, config in zip(color_list, config_list)
    ]

    area_list = [ret[0] for ret in ret_list]
    tup2_list = [ret[3] for ret in ret_list]

    best_index = (
        None if BEST_INDEX is None else BEST_INDEX
    )  # Match formatting of below, this is a silly conditional

    best_y = 0.0
    best_index_ = None
    valid_best_index = []
    for index, tup2 in enumerate(tup2_list):
        if tup2 is None:
            continue

        conf_list, x_list, y_list, length = tup2
        y = y_list[0]
        if best_y < y:
            valid_best_index.append(index)
            best_index_ = index
            best_y = y

    # If user defined best_index is invalid, don't use it
    if best_index is None:
        best_index = best_index_
    else:
        if best_index not in valid_best_index:
            best_index = None

    if best_index is not None:
        best_conf_list, best_x_list, best_y_list, best_length = tup2_list[best_index]
        color = 'xkcd:gold'
        marker = 'D'
        plt.plot(best_x_list, best_y_list, color=color, marker=marker)

    plt.title('Precision-Recall Curves', y=1.19)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    ######################################################################################

    if plot_iou_recall:
        axes_ = plt.subplot(142)
        axes_.set_autoscalex_on(False)
        axes_.set_autoscaley_on(False)
        axes_.set_xlabel('IOU (Intersection / Union)')
        axes_.set_ylabel('Recall')
        axes_.set_xlim([0.0, 1.01])
        axes_.set_ylim([0.0, 1.01])

        ret_list = [
            localizer_iou_recall_algo_plot(ibs, color=color_, plot_point=False, **config_)
            for color_, config_ in zip(color_list, config_list)
        ]

        # area_list = [ ret[0] for ret in ret_list ]
        # tup2_list = [ ret[3] for ret in ret_list ]

        # best_index = None if BEST_INDEX is None else BEST_INDEX  # Match formatting of below, this is a silly conditional

        # best_y = 0.0
        # best_index_ = None
        # valid_best_index = []
        # for index, tup2 in enumerate(tup2_list):
        #     if tup2 is None:
        #         continue

        #     conf_list, x_list, y_list, length = tup2
        #     y = y_list[0]
        #     if best_y < y:
        #         valid_best_index.append(index)
        #         best_index_ = index
        #         best_y = y

        # # If user defined best_index is invalid, don't use it
        # if best_index is None:
        #     best_index = best_index_
        # else:
        #     if best_index not in valid_best_index:
        #         best_index = None

        # if best_index is not None:
        #     best_conf_list, best_x_list, best_y_list, best_length = tup2_list[best_index]
        #     color = 'xkcd:gold'
        #     marker = 'D'
        #     plt.plot(best_x_list, best_y_list, color=color, marker=marker)

        plt.title('Recall-IOU Curves', y=1.19)
        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=3,
            ncol=2,
            mode='expand',
            borderaxespad=0.0,
        )

    ######################################################################################

    # axes_ = plt.subplot(153)
    # axes_.set_autoscalex_on(False)
    # axes_.set_autoscaley_on(False)
    # axes_.set_xlabel('IOU (Intersection / Union)')
    # axes_.set_ylabel('Precision')
    # axes_.set_xlim([0.0, 1.01])
    # axes_.set_ylim([0.0, 1.01])

    # ret_list = [
    #     localizer_iou_precision_algo_plot(ibs, color=color_, plot_point=False, **config_)
    #     for color_, config_ in zip(color_list, config_list)
    # ]

    # plt.title('Precision-IOU Curves', y=1.19)
    # plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
    #            borderaxespad=0.0)

    ######################################################################################

    if best_index is not None:
        axes_ = plt.subplot(144)
        axes_.set_aspect(1)
        gca_ = plt.gca()
        gca_.grid(False)

        target_conf = best_conf_list[0]
        best_config = config_list[best_index]
        best_label = config_list[best_index]['label']
        best_area = area_list[best_index]

        values = localizer_confusion_matrix_algo_plot(
            ibs,
            min_overlap=min_overlap,
            fig_=fig_,
            axes_=axes_,
            target_conf=target_conf,
            **best_config,
        )
        best_conf, (correct_rate, _) = values

        axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0,))
        axes_.set_ylabel('Ground-Truth')
        args = (
            target_recall,
            best_label,
            best_area,
            best_conf,
        )
        plt.title(
            'Confusion Matrix for Recall >= %0.02f\n(Algo: %s, mAP = %0.02f, OP = %0.02f)'
            % args,
            y=1.26,
        )

    ######################################################################################
    axes_ = plt.subplot(143)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)

    best_index = np.argmax(area_list) if BEST_INDEX is None else BEST_INDEX

    best_config = config_list[best_index]
    best_label = config_list[best_index]['label']
    best_area = area_list[best_index]

    values = localizer_confusion_matrix_algo_plot(
        ibs, min_overlap=min_overlap, fig_=fig_, axes_=axes_, **best_config
    )
    best_conf, (correct_rate, _) = values

    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0,))
    axes_.set_ylabel('Ground-Truth')
    args = (
        best_label,
        best_area,
        best_conf,
    )
    plt.title('Confusion Matrix\n(Algo: %s, mAP = %0.02f, OP = %0.02f)' % args, y=1.26)

    ######################################################################################
    if len(config_tag) > 0:
        config_tag = '%s-' % (config_tag,)

    fig_filename = '%slocalizer-precision-recall-%0.2f.png' % (config_tag, min_overlap,)
    fig_path = join(output_path, fig_filename)
    plt.savefig(fig_path, bbox_inches='tight')

    return fig_path


@register_ibs_method
def localizer_precision_recall_algo_display_animate(ibs, config_list, **kwargs):
    for value in range(10):
        min_overlap = value / 10.0
        print('Processing: %r' % (min_overlap,))
        ibs.localizer_precision_recall_algo_display(
            config_list, min_overlap=min_overlap, **kwargs
        )


# def localizer_classification_tp_tn_fp_fn(gt_list, pred_list, conf, min_overlap,
#                                          check_species=False,
#                                          check_viewpoint=False, **kwargs):
#     overlap = general_overlap(gt_list, pred_list)
#     num_gt, num_pred = overlap.shape

#     # Get confidences
#     conf_list = [pred['confidence'] for pred in pred_list]
#     pred_flag_list = [conf <= conf_ for conf_ in conf_list]

#     if num_gt == 0:
#         tp_list = [False] * len(pred_list)
#         tn_list = [not pred_flag for pred_flag in pred_flag_list]
#         fp_list = [    pred_flag for pred_flag in pred_flag_list]
#         fn_list = [False] * len(pred_list)
#     elif num_pred == 0:
#         tp_list = []
#         tn_list = []
#         fp_list = []
#         fn_list = []
#     else:
#         max_overlap = np.max(overlap, axis=0)
#         gt_flag_list = min_overlap < max_overlap

#         status_list = []
#         for gt_flag, pred_flag in zip(gt_flag_list, pred_flag_list):
#             if gt_flag and pred_flag:
#                 status_list.append('tp')
#             elif gt_flag and not pred_flag:
#                 status_list.append('fn')
#             elif not gt_flag and pred_flag:
#                 status_list.append('fp')
#             elif not gt_flag and not pred_flag:
#                 status_list.append('tn')
#             else:
#                 raise ValueError

#         tp_list = [status == 'tp' for status in status_list]
#         tn_list = [status == 'tn' for status in status_list]
#         fp_list = [status == 'fp' for status in status_list]
#         fn_list = [status == 'fn' for status in status_list]

#     return tp_list, tn_list, fp_list, fn_list


# def localizer_classification_confusion_matrix_algo_plot(ibs, color, conf,
#                                                         label=None,
#                                                         min_overlap=0.25,
#                                                         write_images=False,
#                                                         **kwargs):
#     print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf, ))

#     test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
#     test_uuid_list = ibs.get_image_uuids(test_gid_list)

#     print('\tGather Ground-Truth')
#     gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **kwargs)

#     print('\tGather Predictions')
#     pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

#     if write_images:
#         output_folder = 'localizer-classification-confusion-matrix-%0.2f-%0.2f-images' % (min_overlap, conf, )
#         output_path = abspath(expanduser(join('~', 'Desktop', output_folder)))
#         ut.ensuredir(output_path)

#     label_list = []
#     prediction_list = []
#     for index, (test_gid, test_uuid) in enumerate(zip(test_gid_list, test_uuid_list)):
#         if test_uuid in pred_dict:
#             gt_list = gt_dict[test_uuid]
#             pred_list = pred_dict[test_uuid]
#             values = localizer_classification_tp_tn_fp_fn(gt_list, pred_list, conf,
#                                                           min_overlap=min_overlap,
#                                                           **kwargs)
#             tp_list, tn_list, fp_list, fn_list = values
#             tp = tp_list.count(True)
#             tn = tn_list.count(True)
#             fp = fp_list.count(True)
#             fn = fn_list.count(True)

#             for _ in range(int(tp)):
#                 label_list.append('positive')
#                 prediction_list.append('positive')
#             for _ in range(int(tn)):
#                 label_list.append('negative')
#                 prediction_list.append('negative')
#             for _ in range(int(fp)):
#                 label_list.append('negative')
#                 prediction_list.append('positive')
#             for _ in range(int(fn)):
#                 label_list.append('positive')
#                 prediction_list.append('negative')

#             if write_images:
#                 test_image = ibs.get_images(test_gid)
#                 test_image = _resize(test_image, t_width=600, verbose=False)
#                 height_, width_, channels_ = test_image.shape

#                 for gt in gt_list:
#                     xtl = int(gt['xtl'] * width_)
#                     ytl = int(gt['ytl'] * height_)
#                     xbr = int(gt['xbr'] * width_)
#                     ybr = int(gt['ybr'] * height_)
#                     cv2.rectangle(test_image, (xtl, ytl), (xbr, ybr), (0, 0, 255))

#                 zipped = zip(pred_list, tp_list, tn_list, fp_list, fn_list)
#                 for pred, tp_, tn_, fp_, fn_ in zipped:
#                     if tp_:
#                         color = (0, 255, 0)
#                     elif fp_:
#                         continue
#                         # color = (255, 0, 0)
#                     elif fn_:
#                         color = (255, 0, 0)
#                     elif tn_:
#                         continue
#                     else:
#                         continue

#                     xtl = int(pred['xtl'] * width_)
#                     ytl = int(pred['ytl'] * height_)
#                     xbr = int(pred['xbr'] * width_)
#                     ybr = int(pred['ybr'] * height_)
#                     cv2.rectangle(test_image, (xtl, ytl), (xbr, ybr), color)

#                 status_str = 'success' if (fp + fn) == 0 else 'failure'
#                 status_val = tp - fp - fn
#                 args = (status_str, status_val, test_gid, tp, fp, fn, )
#                 output_filename = 'test_%s_%d_gid_%d_tp_%d_fp_%d_fn_%d.png' % args
#                 output_filepath = join(output_path, output_filename)
#                 cv2.imwrite(output_filepath, test_image)

#     category_list = ['positive', 'negative']
#     category_mapping = {
#         'positive': 0,
#         'negative': 1,
#     }
#     return general_confusion_matrix_algo(label_list, prediction_list, category_list,
#                                          category_mapping, size=20, **kwargs)


# @register_ibs_method
# def localizer_classifications_confusion_matrix_algo_display(ibs, conf,
#                                                             min_overlap=0.25,
#                                                             figsize=(24, 7),
#                                                             write_images=False,
#                                                             target_recall=0.9,
#                                                             plot_point=True,
#                                                             masking=False,
#                                                             **kwargs):
#     import matplotlib.pyplot as plt

#     fig_ = plt.figure(figsize=figsize)

#     config = {
#         'label'        : 'WIC',
#         'algo'         : '_COMBINED',
#         'species_set'  : set(['zebra']),
#         'classify'     : True,
#         'classifier_algo': 'svm',
#         'classifier_masking': masking,
#         'classifier_weight_filepath': '/home/jason/code/wbia/models-bootstrap/classifier.svm.image.zebra.pkl',
#     }

#     axes_ = plt.subplot(111)
#     axes_.set_aspect(1)
#     gca_ = plt.gca()
#     gca_.grid(False)

#     correct_rate, _ = localizer_classification_confusion_matrix_algo_plot(ibs, None, conf,
#                                                                           min_overlap=min_overlap,
#                                                                           write_images=write_images,
#                                                                           fig_=fig_, axes_=axes_,
#                                                                           **config)
#     axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
#     axes_.set_ylabel('Ground-Truth')
#     args = (min_overlap, conf, )
#     plt.title('Confusion Matrix (IoU %0.02f, Conf %0.02f)' % args, y=1.13)

#     # plt.show()
#     args = (min_overlap, conf, )
#     fig_filename = 'localizer-classification-confusion-matrix-%0.2f-%0.2f.png' % args
#     fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
#     plt.savefig(fig_path, bbox_inches='tight')


# @register_ibs_method
# def localizer_classifications_confusion_matrix_algo_display_animate(ibs, total=10, **kwargs):
#     for index in range(0, total + 1):
#         conf = index / total
#         ibs.localizer_classifications_confusion_matrix_algo_display(conf, **kwargs)


def classifier_cameratrap_precision_recall_algo(
    ibs, positive_imageset_id, negative_imageset_id, **kwargs
):
    depc = ibs.depc_image
    test_gid_set_ = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_set_ = list(test_gid_set_)

    positive_gid_set = set(ibs.get_imageset_gids(positive_imageset_id))
    negative_gid_set = set(ibs.get_imageset_gids(negative_imageset_id))

    test_gid_set = []
    label_list = []
    for gid in test_gid_set_:
        if gid in positive_gid_set:
            label = 'positive'
        elif gid in negative_gid_set:
            label = 'negative'
        else:
            # label = 'unknown'
            continue
        test_gid_set.append(gid)
        label_list.append(label)

    prediction_list = depc.get_property(
        'classifier', test_gid_set, 'class', config=kwargs
    )
    confidence_list = depc.get_property(
        'classifier', test_gid_set, 'score', config=kwargs
    )
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence in zip(prediction_list, confidence_list)
    ]
    return general_precision_recall_algo(ibs, label_list, confidence_list)


def classifier_cameratrap_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing Precision-Recall for: %r' % (label,))
    (
        conf_list,
        pr_list,
        re_list,
        tpr_list,
        fpr_list,
    ) = classifier_cameratrap_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def classifier_cameratrap_roc_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing ROC for: %r' % (label,))
    (
        conf_list,
        pr_list,
        re_list,
        tpr_list,
        fpr_list,
    ) = classifier_cameratrap_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(
        conf_list, fpr_list, tpr_list, interpolate=False, target=(0.0, 1.0), **kwargs
    )


def classifier_cameratrap_confusion_matrix_algo_plot(
    ibs,
    label,
    color,
    conf,
    positive_imageset_id,
    negative_imageset_id,
    output_cases=False,
    **kwargs,
):
    print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf,))
    depc = ibs.depc_image
    test_gid_set_ = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_set_ = list(test_gid_set_)

    positive_gid_set = set(ibs.get_imageset_gids(positive_imageset_id))
    negative_gid_set = set(ibs.get_imageset_gids(negative_imageset_id))

    test_gid_set = []
    label_list = []
    for gid in test_gid_set_:
        if gid in positive_gid_set:
            label = 'positive'
        elif gid in negative_gid_set:
            label = 'negative'
        else:
            # label = 'unknown'
            continue
        test_gid_set.append(gid)
        label_list.append(label)

    prediction_list = depc.get_property(
        'classifier', test_gid_set, 'class', config=kwargs
    )
    confidence_list = depc.get_property(
        'classifier', test_gid_set, 'score', config=kwargs
    )
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence in zip(prediction_list, confidence_list)
    ]
    prediction_list = [
        'positive' if confidence >= conf else 'negative' for confidence in confidence_list
    ]

    if output_cases:
        output_path = 'cameratrap-confusion-incorrect'
        output_path = abspath(expanduser(join('~', 'Desktop', output_path)))
        positive_path = join(output_path, 'positive')
        negative_path = join(output_path, 'negative')
        ut.delete(output_path)
        ut.ensuredir(output_path)
        ut.ensuredir(positive_path)
        ut.ensuredir(negative_path)

        interpolation = cv2.INTER_LANCZOS4
        warpkw = dict(interpolation=interpolation)
        for gid, label, prediction in zip(test_gid_set, label_list, prediction_list):
            if label == prediction:
                continue
            image = ibs.get_images(gid)
            image = cv2.resize(image, (192, 192), **warpkw)
            # Get path
            image_path = positive_path if label == 'positive' else negative_path
            image_filename = 'hardidx_%d_pred_%s_case_fail.jpg' % (gid, prediction,)
            image_filepath = join(image_path, image_filename)
            # Save path
            cv2.imwrite(image_filepath, image)

    category_list = ['positive', 'negative']
    category_mapping = {
        'positive': 0,
        'negative': 1,
    }
    return general_confusion_matrix_algo(
        label_list, prediction_list, category_list, category_mapping, **kwargs
    )


@register_ibs_method
def classifier_cameratrap_precision_recall_algo_display(
    ibs, positive_imageset_id, negative_imageset_id, config_list=None, figsize=(20, 20)
):
    import matplotlib.pyplot as plt
    import wbia.plottool as pt

    fig_ = plt.figure(figsize=figsize, dpi=400)

    if config_list is None:
        config_list = [
            # {'label': 'Initial Model (5%)  - IBEIS_CNN',  'classifier_algo': 'cnn',      'classifier_weight_filepath': 'ryan.wbia_cnn.v1'},
            # {'label': 'Initial Model (5%)  - DenseNet',   'classifier_algo': 'densenet', 'classifier_weight_filepath': 'ryan_densenet_v1'},
            # {'label': 'Initial Model (5%)  - DenseNet 0', 'classifier_algo': 'densenet', 'classifier_weight_filepath': 'ryan_densenet_v1:0'},
            # {'label': 'Initial Model (5%)  - DenseNet 1', 'classifier_algo': 'densenet', 'classifier_weight_filepath': 'ryan_densenet_v1:1'},
            # {'label': 'Initial Model (5%)  - DenseNet 2', 'classifier_algo': 'densenet', 'classifier_weight_filepath': 'ryan_densenet_v1:2'},
            {
                'label': 'Initial Model (10%) - DenseNet',
                'classifier_algo': 'densenet',
                'classifier_weight_filepath': 'ryan_densenet_v2',
            },
            # {'label': 'Initial Model (10%) - DenseNet 0', 'classifier_algo': 'densenet', 'classifier_weight_filepath': 'ryan_densenet_v2:0'},
            # {'label': 'Initial Model (10%) - DenseNet 1', 'classifier_algo': 'densenet', 'classifier_weight_filepath': 'ryan_densenet_v2:1'},
            # {'label': 'Initial Model (10%) - DenseNet 2', 'classifier_algo': 'densenet', 'classifier_weight_filepath': 'ryan_densenet_v2:2'},
            # {'label': 'Initial Model   (0%)', 'classifier_algo': 'cnn', 'classifier_weight_filepath': 'megan2.1'},
            # {'label': 'Retrained Model (1%)', 'classifier_algo': 'cnn', 'classifier_weight_filepath': 'megan2.2'},
            # {'label': 'Retrained Model (2%)', 'classifier_algo': 'cnn', 'classifier_weight_filepath': 'megan2.3'},
            # {'label': 'Retrained Model (3%)', 'classifier_algo': 'cnn', 'classifier_weight_filepath': 'megan2.4'},
            # {'label': 'Retrained Model (4%)', 'classifier_algo': 'cnn', 'classifier_weight_filepath': 'megan2.5'},
            # {'label': 'Retrained Model (5%)', 'classifier_algo': 'cnn', 'classifier_weight_filepath': 'megan2.6'},
            # {'label': 'Initial Model   (0%)',   'classifier_weight_filepath': 'megan1.1'},
            # {'label': 'Retrained Model (1%)',   'classifier_weight_filepath': 'megan1.2'},
            # {'label': 'Retrained Model (2%)',   'classifier_weight_filepath': 'megan1.3'},
            # {'label': 'Retrained Model (3%)',   'classifier_weight_filepath': 'megan1.4'},
            # {'label': 'Retrained Model (3.5%)', 'classifier_weight_filepath': 'megan1.5'},
            # {'label': 'Retrained Model (5%)',   'classifier_weight_filepath': 'megan1.6'},
        ]
    color_list = pt.distinct_colors(len(config_list), randomize=False)

    axes_ = plt.subplot(221)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall')
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    ret_list = [
        classifier_cameratrap_precision_recall_algo_plot(
            ibs,
            color=color,
            positive_imageset_id=positive_imageset_id,
            negative_imageset_id=negative_imageset_id,
            **config,
        )
        for color, config in zip(color_list, config_list)
    ]
    area_list = [ret[0] for ret in ret_list]
    conf_list = [ret[1] for ret in ret_list]
    index = np.argmax(area_list)
    # index = 0
    best_label1 = config_list[index]['label']
    best_config1 = config_list[index]
    best_color1 = color_list[index]
    best_area1 = area_list[index]
    best_conf1 = conf_list[index]
    plt.title(
        'Precision-Recall Curve (Best: %s, AP = %0.02f)' % (best_label1, best_area1,),
        y=1.10,
    )
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(222)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    ret_list = [
        classifier_cameratrap_roc_algo_plot(
            ibs,
            color=color,
            positive_imageset_id=positive_imageset_id,
            negative_imageset_id=negative_imageset_id,
            **config,
        )
        for color, config in zip(color_list, config_list)
    ]
    area_list = [ret[0] for ret in ret_list]
    conf_list = [ret[1] for ret in ret_list]
    index = np.argmax(area_list)
    # index = 0
    best_label2 = config_list[index]['label']
    best_config2 = config_list[index]
    best_color2 = color_list[index]
    best_area2 = area_list[index]
    best_conf2 = conf_list[index]
    plt.title('ROC Curve (Best: %s, AP = %0.02f)' % (best_label2, best_area2,), y=1.10)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(223)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = classifier_cameratrap_confusion_matrix_algo_plot(
        ibs,
        color=best_color1,
        conf=best_conf1,
        fig_=fig_,
        axes_=axes_,
        positive_imageset_id=positive_imageset_id,
        negative_imageset_id=negative_imageset_id,
        output_cases=True,
        **best_config1,
    )
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0,))
    axes_.set_ylabel('Ground-Truth')
    plt.title('P-R Confusion Matrix (OP = %0.02f)' % (best_conf1,), y=1.12)

    axes_ = plt.subplot(224)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = classifier_cameratrap_confusion_matrix_algo_plot(
        ibs,
        color=best_color2,
        conf=best_conf2,
        fig_=fig_,
        axes_=axes_,
        positive_imageset_id=positive_imageset_id,
        negative_imageset_id=negative_imageset_id,
        **best_config2,
    )
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0,))
    axes_.set_ylabel('Ground-Truth')
    plt.title('ROC Confusion Matrix (OP = %0.02f)' % (best_conf2,), y=1.12)

    fig_filename = 'classifier-cameratrap-precision-recall-roc.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


# def classifier_binary_precision_recall_algo(ibs, category_set, **kwargs):
#     depc = ibs.depc_image
#     test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
#     test_gid_set = list(test_gid_set)
#     aids_list = ibs.get_image_aids(test_gid_set)
#     species_set_list = [
#         set(ibs.get_annot_species_texts(aid_list))
#         for aid_list in aids_list
#     ]
#     label_list = [
#         'negative' if len(species_set & category_set) == 0 else 'positive'
#         for species_set in species_set_list
#     ]
#     prediction_list = depc.get_property('classifier', test_gid_set, 'class', config=kwargs)
#     confidence_list = depc.get_property('classifier', test_gid_set, 'score', config=kwargs)
#     confidence_list = [
#         confidence if prediction == 'positive' else 1.0 - confidence
#         for prediction, confidence  in zip(prediction_list, confidence_list)
#     ]
#     return general_precision_recall_algo(ibs, label_list, confidence_list)


# def classifier_binary_precision_recall_algo_plot(ibs, **kwargs):
#     label = kwargs['label']
#     print('Processing Precision-Recall for: %r' % (label, ))
#     conf_list, pr_list, re_list, tpr_list, fpr_list = classifier_binary_precision_recall_algo(ibs, **kwargs)
#     return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


# def classifier_binary_roc_algo_plot(ibs, **kwargs):
#     label = kwargs['label']
#     print('Processing ROC for: %r' % (label, ))
#     conf_list, pr_list, re_list, tpr_list, fpr_list = classifier_binary_precision_recall_algo(ibs, **kwargs)
#     return general_area_best_conf(conf_list, fpr_list, tpr_list, interpolate=False,
#                                   target=(0.0, 1.0), **kwargs)


# def classifier_binary_confusion_matrix_algo_plot(ibs, label, color, conf, category_set, **kwargs):
#     print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf, ))
#     depc = ibs.depc_image
#     test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
#     test_gid_set = list(test_gid_set)
#     aids_list = ibs.get_image_aids(test_gid_set)
#     species_set_list = [
#         set(ibs.get_annot_species_texts(aid_list))
#         for aid_list in aids_list
#     ]
#     label_list = [
#         'negative' if len(species_set & category_set) == 0 else 'positive'
#         for species_set in species_set_list
#     ]
#     prediction_list = depc.get_property('classifier', test_gid_set, 'class', config=kwargs)
#     confidence_list = depc.get_property('classifier', test_gid_set, 'score', config=kwargs)
#     confidence_list = [
#         confidence if prediction == 'positive' else 1.0 - confidence
#         for prediction, confidence  in zip(prediction_list, confidence_list)
#     ]
#     prediction_list = [
#         'positive' if confidence >= conf else 'negative'
#         for confidence in confidence_list
#     ]

#     category_list = ['positive', 'negative']
#     category_mapping = {
#         'positive': 0,
#         'negative': 1,
#     }
#     return general_confusion_matrix_algo(label_list, prediction_list, category_list,
#                                          category_mapping, **kwargs)


# @register_ibs_method
# def classifier_binary_precision_recall_algo_display(ibs, figsize=(16, 16), **kwargs):
#     import matplotlib.pyplot as plt

#     fig_ = plt.figure(figsize=figsize)

#     # label = 'V1'
#     # species_list = ['zebra']
#     # kwargs['classifier_weight_filepath'] = 'coco_zebra'

#     label = 'V3'
#     species_list = ['zebra_plains', 'zebra_grevys']
#     kwargs['classifier_weight_filepath'] = 'v3_zebra'

#     category_set = set(species_list)

#     axes_ = plt.subplot(221)
#     axes_.set_autoscalex_on(False)
#     axes_.set_autoscaley_on(False)
#     axes_.set_xlabel('Recall')
#     axes_.set_ylabel('Precision')
#     axes_.set_xlim([0.0, 1.01])
#     axes_.set_ylim([0.0, 1.01])
#     area, best_conf1, _ = classifier_binary_precision_recall_algo_plot(ibs, label=label, color='r', category_set=category_set, **kwargs)
#     plt.title('Precision-Recall Curve (AP = %0.02f)' % (area, ), y=1.10)
#     plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
#                borderaxespad=0.0)

#     axes_ = plt.subplot(222)
#     axes_.set_autoscalex_on(False)
#     axes_.set_autoscaley_on(False)
#     axes_.set_xlabel('False-Positive Rate')
#     axes_.set_ylabel('True-Positive Rate')
#     axes_.set_xlim([0.0, 1.01])
#     axes_.set_ylim([0.0, 1.01])
#     area, best_conf2, _ = classifier_binary_roc_algo_plot(ibs, label=label, color='r', category_set=category_set, **kwargs)
#     plt.title('ROC Curve (AP = %0.02f)' % (area, ), y=1.10)
#     plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
#                borderaxespad=0.0)

#     axes_ = plt.subplot(223)
#     axes_.set_aspect(1)
#     gca_ = plt.gca()
#     gca_.grid(False)
#     correct_rate, _ = classifier_binary_confusion_matrix_algo_plot(ibs, label, 'r', conf=best_conf1, fig_=fig_, axes_=axes_, category_set=category_set, **kwargs)
#     axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
#     axes_.set_ylabel('Ground-Truth')
#     plt.title('P-R Confusion Matrix (OP = %0.02f)' % (best_conf1, ), y=1.12)

#     axes_ = plt.subplot(224)
#     axes_.set_aspect(1)
#     gca_ = plt.gca()
#     gca_.grid(False)
#     correct_rate, _ = classifier_binary_confusion_matrix_algo_plot(ibs, label, 'r', conf=best_conf2, fig_=fig_, axes_=axes_, category_set=category_set, **kwargs)
#     axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
#     axes_.set_ylabel('Ground-Truth')
#     plt.title('ROC Confusion Matrix (OP = %0.02f)' % (best_conf2, ), y=1.12)

#     fig_filename = 'classifier-precision-recall-roc.png'
#     fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
#     plt.savefig(fig_path, bbox_inches='tight')


def classifier2_precision_recall_algo(
    ibs,
    category,
    species_mapping={},
    output_path=None,
    test_gid_list=None,
    test_label_list=None,
    **kwargs,
):
    depc = ibs.depc_image
    if test_gid_list is None:
        test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
        test_gid_list = list(test_gid_set)

    if test_label_list is None:
        aids_list = ibs.get_image_aids(test_gid_list)
        species_list_list = list(map(ibs.get_annot_species_texts, aids_list))
        species_set_list = []
        for species_list in species_list_list:
            species_set = set([])
            for species in species_list:
                species = species_mapping.get(species, species)
                species_set.add(species)
            species_set_list.append(species_set)
    else:
        species_set_list = [set([label]) for label in test_label_list]

    label_list = [
        'positive' if category in species_set_ else 'negative'
        for species_set_ in species_set_list
    ]

    confidence_dict_list = depc.get_property(
        'classifier_two', test_gid_list, 'scores', config=kwargs
    )
    confidence_list = [
        confidence_dict[category] for confidence_dict in confidence_dict_list
    ]

    if output_path is not None:
        ut.ensuredir(output_path)
        config_ = {
            'draw_annots': False,
            'thumbsize': (192, 192),
        }
        thumbnail_list = depc.get_property(
            'thumbnails', test_gid_list, 'img', config=config_
        )
        zipped = zip(
            test_gid_list, thumbnail_list, species_set_list, confidence_dict_list
        )
        for index, (test_gid, thumbnail, species_set, confidence_dict) in enumerate(
            zipped
        ):
            print(index)
            x = ';'.join(species_set)
            y = []
            for key in confidence_dict:
                y.append('%s-%0.04f' % (key, confidence_dict[key],))
            y = ';'.join(y)
            output_filename = 'image-index-%s-gid-%s-gt-%s-pred-%s.png' % (
                index,
                test_gid,
                x,
                y,
            )
            output_filepath = join(output_path, output_filename)
            cv2.imwrite(output_filepath, thumbnail)

    kwargs.pop('category', None)
    return general_precision_recall_algo(ibs, label_list, confidence_list)


def classifier2_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing Precision-Recall for: %r' % (label,))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier2_precision_recall_algo(
        ibs, **kwargs
    )
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def classifier2_roc_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing ROC for: %r' % (label,))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier2_precision_recall_algo(
        ibs, **kwargs
    )
    return general_area_best_conf(
        conf_list, fpr_list, tpr_list, interpolate=False, target=(0.0, 1.0), **kwargs
    )


@register_ibs_method
def classifier2_precision_recall_algo_display(
    ibs,
    species_list=None,
    species_mapping={},
    nice_mapping={},
    test_gid_list=None,
    test_label_list=None,
    figsize=(20, 9),
    **kwargs,
):
    import matplotlib.pyplot as plt
    import wbia.plottool as pt

    depc = ibs.depc_image
    fig_ = plt.figure(figsize=figsize, dpi=400)  # NOQA

    # kwargs['classifier_two_weight_filepath'] = 'v3'
    # kwargs['classifier_two_weight_filepath'] = 'candidacy'
    # kwargs['classifier_two_weight_filepath'] = 'ggr2'

    is_labeled = test_label_list is not None

    kwargs['classifier_two_algo'] = 'densenet'
    kwargs['classifier_two_weight_filepath'] = 'flukebook_v1'

    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_list_ = list(test_gid_set) if test_gid_list is None else test_gid_list
    test_label_list_ = test_label_list if is_labeled else [None] * len(test_gid_list_)

    zipped = list(zip(test_gid_list_, test_label_list_))
    test_gid_list_ = []
    test_label_list_ = []
    for test_gid_, test_label_ in zipped:
        if test_gid_ in test_gid_set:
            test_gid_list_.append(test_gid_)
            test_label_list_.append(test_label_)

    test_gid_list = test_gid_list_
    test_label_list = test_label_list_ if is_labeled else None

    # depc.delete_property('classifier_two', test_gid_list, config=kwargs)

    if species_list is None:
        test_gid = test_gid_list[0]
        confidence_dict = depc.get_property(
            'classifier_two', test_gid, 'scores', config=kwargs
        )
        species_list = confidence_dict.keys()

    category_set = sorted(species_list)

    config_list = []
    for category in category_set:
        category_nice = nice_mapping.get(category, category)
        config_dict = {
            'label': category_nice,
            'category': category,
        }
        config_dict.update(kwargs)
        config_list.append(config_dict)

    color_list_ = []
    color_list = pt.distinct_colors(len(config_list) - len(color_list_), randomize=False)
    color_list = color_list_ + color_list

    axes_ = plt.subplot(121)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall')
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])

    for color, config in zip(color_list, config_list):
        classifier2_precision_recall_algo_plot(
            ibs,
            color=color,
            test_gid_list=test_gid_list,
            test_label_list=test_label_list,
            species_mapping=species_mapping,
            **config,
        )
    plt.title('Precision-Recall Curves', y=1.19)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(122)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])

    op_dict = {}
    for color, config in zip(color_list, config_list):
        values = classifier2_roc_algo_plot(
            ibs,
            color=color,
            test_gid_list=test_gid_list,
            test_label_list=test_label_list,
            species_mapping=species_mapping,
            **config,
        )
        ap, best_conf, tup1, tup2 = values
        op_dict[config['category']] = best_conf

    plt.title('ROC Curves', y=1.19)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    if is_labeled:
        species_set_list = [set([label]) for label in test_label_list]
    else:
        aids_list = ibs.get_image_aids(test_gid_list)
        species_list_list = list(map(ibs.get_annot_species_texts, aids_list))
        species_set_list = []
        for species_list in species_list_list:
            species_set = set([])
            for species in species_list:
                species = species_mapping.get(species, species)
                species_set.add(species)
            species_set_list.append(species_set)
    confidence_dict_list = depc.get_property(
        'classifier_two', test_gid_list, 'scores', config=kwargs
    )

    correct = 0
    for test_gid, confidence_dict, species_set in zip(
        test_gid_list, confidence_dict_list, species_set_list
    ):
        species_set_ = set([])
        for key in confidence_dict:
            if op_dict[key] <= confidence_dict[key]:
                species_set_.add(key)
        if len(species_set ^ species_set_) == 0:
            correct += 1
        else:
            print(test_gid, confidence_dict, species_set)
    print('Accuracy: %0.04f' % (100.0 * correct / len(test_gid_list)))
    print('\t using op_dict = %r' % (op_dict,))

    fig_filename = 'classifier2-precision-recall-roc.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


def labeler_tp_tn_fp_fn(
    ibs,
    category_list,
    species_mapping={},
    viewpoint_mapping={},
    samples=SAMPLES,
    test_gid_set=None,
    **kwargs,
):
    def errors(zipped, conf, category):
        tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
        for index, (label, confidence) in enumerate(zipped):
            if label == category:
                if conf <= confidence:
                    tp += 1
                else:
                    fn += 1
            else:
                if conf <= confidence:
                    fp += 1
                else:
                    tn += 1
        return tp, tn, fp, fn

    depc = ibs.depc_annot

    if test_gid_set is None:
        test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
        test_gid_set = list(test_gid_set)

    aids_list = ibs.get_image_aids(test_gid_set)
    aid_list = ut.flatten(aids_list)
    # Get annot species and viewpoints
    species_list = ibs.get_annot_species_texts(aid_list)
    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    # Filter aids with species of interest and undefined viewpoints

    species_list = [species_mapping.get(species, species) for species in species_list]
    viewpoint_list = [
        viewpoint_mapping.get(species, {}).get(viewpoint, viewpoint)
        for species, viewpoint in zip(species_list, viewpoint_list)
    ]

    flag_list = [
        species in category_list and viewpoint is not None
        for species, viewpoint in zip(species_list, viewpoint_list)
    ]

    if False in flag_list:
        aid_list = ut.compress(aid_list, flag_list)
        species_list = ut.compress(species_list, flag_list)
        viewpoint_list = ut.compress(viewpoint_list, flag_list)

    # Make ground-truth
    label_list = [
        '%s:%s' % (species, viewpoint_,)
        for species, viewpoint_ in zip(species_list, viewpoint_list)
    ]
    # Get predictions
    # depc.delete_property('labeler', aid_list, config=kwargs)
    probability_dict_list = depc.get_property('labeler', aid_list, 'probs', config=kwargs)

    value1_list = set(label_list)
    value2_list = set(probability_dict_list[0].keys())
    assert len(value1_list - value2_list) == 0
    assert len(value2_list - value1_list) == 0

    conf_list = [_ / float(samples) for _ in range(0, int(samples) + 1)]
    label_dict = {}
    for key in value1_list:
        print('\t%r' % (key,))
        conf_dict = {}
        confidence_list = [
            probability_dict[key] for probability_dict in probability_dict_list
        ]
        zipped = list(zip(label_list, confidence_list))
        for conf in conf_list:
            conf_dict[conf] = errors(zipped, conf, key)
        label_dict[key] = conf_dict
    return label_dict


def labeler_precision_recall_algo(ibs, category_list, label_dict, **kwargs):

    if category_list is None:
        category_list_ = label_dict.keys()
    else:
        category_list_ = []
        for category in category_list:
            for key in label_dict:
                if category in key or category is None:
                    category_list_.append(key)

    global_conf_dict = {}
    for category in category_list_:
        conf_dict = label_dict[category]
        for conf in conf_dict:
            new_list = conf_dict[conf]
            if conf not in global_conf_dict:
                global_conf_dict[conf] = new_list
            else:
                cur_list = global_conf_dict[conf]
                zipped_ = zip(cur_list, new_list)
                global_conf_dict[conf] = [cur + new for cur, new in zipped_]

    conf_list_ = [-1.0, -1.0]
    pr_list = [1.0, 0.0]
    re_list = [0.0, 1.0]
    tpr_list = [0.0, 1.0]
    fpr_list = [0.0, 1.0]
    # conf_list_ = []
    # pr_list = []
    # re_list = []
    # tpr_list = []
    # fpr_list = []
    for conf in sorted(global_conf_dict.keys(), reverse=True):
        error_list = global_conf_dict[conf]
        tp, tn, fp, fn = error_list
        try:
            pr = tp / (tp + fp)
            re = tp / (tp + fn)
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            conf_list_.append(conf)
            pr_list.append(pr)
            re_list.append(re)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        except ZeroDivisionError:
            print(
                'Zero division error (%r) - tp: %r tn: %r fp: %r fn: %r'
                % (conf, tp, tn, fp, fn,)
            )

    return conf_list_, pr_list, re_list, tpr_list, fpr_list


def labeler_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    category_list = kwargs['category_list']
    print(
        'Processing Precision-Recall for: %r (category_list = %r)'
        % (label, category_list,)
    )
    conf_list, pr_list, re_list, tpr_list, fpr_list = labeler_precision_recall_algo(
        ibs, **kwargs
    )
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def labeler_roc_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    category_list = kwargs['category_list']
    print('Processing ROC for: %r (category_list = %r)' % (label, category_list,))
    conf_list, pr_list, re_list, tpr_list, fpr_list = labeler_precision_recall_algo(
        ibs, **kwargs
    )
    return general_area_best_conf(
        conf_list, fpr_list, tpr_list, interpolate=False, target=(0.0, 1.0), **kwargs
    )


def labeler_confusion_matrix_algo_plot(
    ibs,
    category_list,
    species_mapping={},
    viewpoint_mapping={},
    category_mapping=None,
    test_gid_set=None,
    **kwargs,
):
    print('Processing Confusion Matrix')
    depc = ibs.depc_annot

    if test_gid_set is None:
        test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
        test_gid_set = list(test_gid_set)

    aids_list = ibs.get_image_aids(test_gid_set)
    aid_list = ut.flatten(aids_list)
    species_list = ibs.get_annot_species_texts(aid_list)
    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    label_list = [
        '%s:%s'
        % (
            species_mapping.get(species, species),
            viewpoint_mapping.get(species, {}).get(viewpoint, viewpoint),
        )
        for species, viewpoint in zip(species_list, viewpoint_list)
    ]
    temp_list = [
        (aid, label) for aid, label in zip(aid_list, label_list) if label in category_list
    ]
    aid_list = [_[0] for _ in temp_list]
    label_list = [_[1] for _ in temp_list]
    conf_list = depc.get_property('labeler', aid_list, 'score', config=kwargs)
    species_list = depc.get_property('labeler', aid_list, 'species', config=kwargs)
    viewpoint_list = depc.get_property('labeler', aid_list, 'viewpoint', config=kwargs)
    prediction_list = [
        '%s:%s' % (species, viewpoint,)
        for species, viewpoint in zip(species_list, viewpoint_list)
    ]

    category_list = list(map(simple_code, category_list))
    label_list = list(map(simple_code, label_list))
    prediction_list = list(map(simple_code, prediction_list))

    if category_mapping is None:
        category_mapping = {key: index for index, key in enumerate(category_list)}

    category_mapping = {
        simple_code(key): category_mapping[key] for key in category_mapping
    }
    return general_confusion_matrix_algo(
        label_list,
        prediction_list,
        category_list,
        category_mapping,
        conf_list=conf_list,
        size=8,
        **kwargs,
    )


@register_ibs_method
def labeler_precision_recall_algo_display(
    ibs,
    category_list=None,
    species_mapping={},
    viewpoint_mapping={},
    category_mapping=None,
    fuzzy_dict=None,
    figsize=(30, 9),
    test_gid_set=None,
    use_axis_aligned_chips=False,
    labeler_weight_filepath=None,
    config_list=None,
    **kwargs,
):
    import matplotlib.pyplot as plt
    import plottool as pt

    if category_list is None:

        if test_gid_set is None:
            test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
            test_gid_set = list(test_gid_set)

        aids_list = ibs.get_image_aids(test_gid_set)
        aid_list = ut.flatten(aids_list)
        species_list = ibs.get_annot_species_texts(aid_list)
        species_list = [species_mapping.get(species, species) for species in species_list]
        category_list = sorted(list(set(species_list)))

    print('Compiling raw numbers...')
    kwargs['labeler_algo'] = 'densenet'
    if labeler_weight_filepath is None:
        # kwargs['labeler_weight_filepath'] = 'zebra_v1'
        # kwargs['labeler_weight_filepath'] = 'seaturtle'
        # kwargs['labeler_weight_filepath'] = 'giraffe_v1'
        # kwargs['labeler_weight_filepath'] = 'lynx_v3'
        # kwargs['labeler_weight_filepath'] = 'seaturtle_v3'
        # kwargs['labeler_weight_filepath'] = 'jaguar_v3'
        # kwargs['labeler_weight_filepath'] = 'hendrik_dorsal_v2'
        # kwargs['labeler_weight_filepath'] = 'spotted_skunk_v0'
        # kwargs['labeler_weight_filepath'] = 'nassau_grouper_v0'
        # kwargs['labeler_weight_filepath'] = 'spotted_dolphin_v0'
        # kwargs['labeler_weight_filepath'] = 'seadragon_v1'
        kwargs['labeler_weight_filepath'] = 'seadragon_v2'
    else:
        kwargs['labeler_weight_filepath'] = labeler_weight_filepath
    kwargs['labeler_axis_aligned'] = use_axis_aligned_chips

    label_dict = labeler_tp_tn_fp_fn(
        ibs,
        category_list,
        species_mapping=species_mapping,
        viewpoint_mapping=viewpoint_mapping,
        test_gid_set=test_gid_set,
        **kwargs,
    )

    if config_list is None:
        config_list = [
            # {'label': 'Giraffe',                'category_list': None},
            # {'label': 'Masai Giraffe',          'category_list': ['giraffe_masai']},
            # {'label': 'Reticulated Giraffe',    'category_list': ['giraffe_reticulated']},
            # {'label': 'Lynx',                   'category_list': ['lynx_pardinus']},
            # {'label': 'Sea Turtle',             'category_list': ['turtle_sea']},
            # {'label': 'Sea Turtle Head',        'category_list': ['turtle_sea+head']},
            # {'label': 'Manta',                  'category_list': ['manta_ray_giant']},
            # {'label': 'Jaguar',                 'category_list': ['jaguar']},
            # {'label': 'Dorsal Fin',             'category_list': ['dolphin_bottlenose_fin']},
            # {'label': 'Reticulated Giraffe',    'category_list': ['giraffe_reticulated']},
            # {'label': 'Sea Turtle',             'category_list': ['turtle_sea']},
            # {'label': 'Whale Fluke',            'category_list': ['whale_fluke']},
            # {'label': 'Grevy\'s Zebra',         'category_list': ['zebra_grevys']},
            # {'label': 'Plains Zebra',           'category_list': ['zebra_plains']},
            # {'label': 'Spotted Skunk',          'category_list': ['skunk_spotted']},
            # {'label': 'Nassau Grouper',           'category_list': ['grouper_nassau']},
            # {'label': 'Spotted Dolphin',           'category_list': ['dolphin_spotted']},
            # {'label': 'Spotted Dolphin',           'category_list': ['dolphin_spotted']},
            {'label': 'Weedy SD ', 'category_list': ['seadragon_weedy']},
            {'label': 'Weedy Head', 'category_list': ['seadragon_weedy+head']},
            {'label': 'Leafy SD ', 'category_list': ['seadragon_leafy']},
            {'label': 'Leafy Head', 'category_list': ['seadragon_leafy+head']},
        ]

    color_list = [(0.0, 0.0, 0.0)]
    color_list += pt.distinct_colors(len(config_list) - len(color_list), randomize=False)

    fig_ = plt.figure(figsize=figsize, dpi=400)  # NOQA

    axes_ = plt.subplot(131)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall')
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    area_list = []
    for color, config in zip(color_list, config_list):
        ret = labeler_precision_recall_algo_plot(
            ibs, label_dict=label_dict, color=color, **config
        )
        area = ret[0]
        area_list.append(area)
    plt.title('Precision-Recall Curve', y=1.19)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(132)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    for color, config in zip(color_list, config_list):
        labeler_roc_algo_plot(ibs, label_dict=label_dict, color=color, **config)
    plt.title('ROC Curve', y=1.19)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    key_list = sorted(label_dict.keys())

    fuzzy = fuzzy_dict is not None
    if not fuzzy:
        fuzzy_dict = {}
        for index1, label1 in enumerate(key_list):
            if label1 == 'ignore':
                fuzzy_list = []
            else:
                species, viewpoint = label1.strip().split(':')
                fuzzy_list = []
                for index2, label2 in enumerate(key_list):
                    if species in label2:
                        fuzzy_list.append(index2)
            fuzzy_dict[index1] = set(fuzzy_list)

    axes_ = plt.subplot(133)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, fuzzy_rate = labeler_confusion_matrix_algo_plot(
        ibs,
        key_list,
        species_mapping=species_mapping,
        viewpoint_mapping=viewpoint_mapping,
        category_mapping=category_mapping,
        fig_=fig_,
        axes_=axes_,
        fuzzy_dict=fuzzy_dict,
        test_gid_set=test_gid_set,
        **kwargs,
    )

    if fuzzy:
        axes_.set_xlabel(
            'Predicted (Correct = %0.02f%%, Fuzzy = %0.02f%%)'
            % (correct_rate * 100.0, fuzzy_rate * 100.0,)
        )
    else:
        axes_.set_xlabel(
            'Predicted (Correct = %0.02f%%, Species = %0.02f%%)'
            % (correct_rate * 100.0, fuzzy_rate * 100.0,)
        )

    axes_.set_ylabel('Ground-Truth')
    # area_list_ = area_list[1:]
    area_list_ = area_list
    mAP = sum(area_list_) / len(area_list_)
    args = (mAP * 100.0,)
    plt.title('Confusion Matrix\nmAP = %0.02f' % args, y=1.19)

    fig_filename = 'labeler-precision-recall-roc.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


def canonical_precision_recall_algo(ibs, species, **kwargs):
    depc = ibs.depc_annot

    test_gid_set_ = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_list_ = list(test_gid_set_)
    test_aid_list_ = ut.flatten(ibs.get_image_aids(test_gid_list_))
    test_aid_list_ = ibs.filter_annotation_set(test_aid_list_, species=species)
    test_flag_list_ = ibs.get_annot_canonical(test_aid_list_)

    test_aid_set = []
    label_list = []
    for aid, flag in zip(test_aid_list_, test_flag_list_):
        if flag:
            label = 'positive'
        else:
            label = 'negative'
        test_aid_set.append(aid)
        label_list.append(label)

    prediction_list = depc.get_property(
        'classifier', test_aid_set, 'class', config=kwargs
    )
    confidence_list = depc.get_property(
        'classifier', test_aid_set, 'score', config=kwargs
    )
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence in zip(prediction_list, confidence_list)
    ]
    return general_precision_recall_algo(ibs, label_list, confidence_list)


def canonical_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing Precision-Recall for: %r' % (label,))
    conf_list, pr_list, re_list, tpr_list, fpr_list = canonical_precision_recall_algo(
        ibs, **kwargs
    )
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def canonical_roc_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing ROC for: %r' % (label,))
    conf_list, pr_list, re_list, tpr_list, fpr_list = canonical_precision_recall_algo(
        ibs, **kwargs
    )
    return general_area_best_conf(
        conf_list, fpr_list, tpr_list, interpolate=False, target=(0.0, 1.0), **kwargs
    )


def canonical_confusion_matrix_algo_plot(
    ibs, label, color, conf, species, output_cases=False, **kwargs
):
    print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf,))
    depc = ibs.depc_annot

    test_gid_set_ = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_list_ = list(test_gid_set_)
    test_aid_list_ = ut.flatten(ibs.get_image_aids(test_gid_list_))
    test_aid_list_ = ibs.filter_annotation_set(test_aid_list_, species=species)
    test_flag_list_ = ibs.get_annot_canonical(test_aid_list_)

    test_aid_set = []
    label_list = []
    for aid, flag in zip(test_aid_list_, test_flag_list_):
        if flag:
            label = 'positive'
        else:
            label = 'negative'
        test_aid_set.append(aid)
        label_list.append(label)

    prediction_list = depc.get_property(
        'classifier', test_aid_set, 'class', config=kwargs
    )
    confidence_list = depc.get_property(
        'classifier', test_aid_set, 'score', config=kwargs
    )
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence in zip(prediction_list, confidence_list)
    ]
    prediction_list = [
        'positive' if confidence >= conf else 'negative' for confidence in confidence_list
    ]

    if output_cases:
        output_path = 'canonical-confusion-incorrect'
        output_path = abspath(expanduser(join('~', 'Desktop', output_path)))
        positive_path = join(output_path, 'positive')
        negative_path = join(output_path, 'negative')
        ut.delete(output_path)
        ut.ensuredir(output_path)
        ut.ensuredir(positive_path)
        ut.ensuredir(negative_path)

        config = {
            'dim_size': (192, 192),
            'resize_dim': 'wh',
        }
        chip_list = ibs.depc_annot.get_property(
            'chips', test_aid_set, 'img', config=config
        )

        zipped = zip(test_aid_set, chip_list, label_list, prediction_list)
        for aid, chip, label, prediction in zipped:
            if label == prediction:
                continue
            # Get path
            image_path = positive_path if label == 'positive' else negative_path
            image_filename = 'hardidx_%d_pred_%s_case_fail.jpg' % (aid, prediction,)
            image_filepath = join(image_path, image_filename)
            # Save path
            cv2.imwrite(image_filepath, chip)

    category_list = ['positive', 'negative']
    category_mapping = {
        'positive': 0,
        'negative': 1,
    }
    return general_confusion_matrix_algo(
        label_list, prediction_list, category_list, category_mapping, **kwargs
    )


@register_ibs_method
def canonical_precision_recall_algo_display(ibs, figsize=(20, 20)):
    import matplotlib.pyplot as plt
    import plottool as pt

    fig_ = plt.figure(figsize=figsize, dpi=400)

    config_list = [
        {
            'label': 'CA V1 Ensemble',
            'classifier_algo': 'densenet',
            'classifier_weight_filepath': 'canonical_zebra_grevys_v1',
            'species': 'zebra_grevys',
        },  # SMALLER DATASET
        {
            'label': 'CA V2 Ensemble',
            'classifier_algo': 'densenet',
            'classifier_weight_filepath': 'canonical_zebra_grevys_v2',
            'species': 'zebra_grevys',
        },  # BROKEN L/R AUGMENTATION
        {
            'label': 'CA V3 Ensemble',
            'classifier_algo': 'densenet',
            'classifier_weight_filepath': 'canonical_zebra_grevys_v3',
            'species': 'zebra_grevys',
        },  # LARGER DATASET, TOO HARSH AUGMENTATION
        {
            'label': 'CA V4 Ensemble',
            'classifier_algo': 'densenet',
            'classifier_weight_filepath': 'canonical_zebra_grevys_v4',
            'species': 'zebra_grevys',
        },  # BETTER AUGMENTATION
        # {'label': 'CA V4 Model 0',  'classifier_algo': 'densenet', 'classifier_weight_filepath': 'canonical_zebra_grevys_v4:0', 'species': 'zebra_grevys'},
        # {'label': 'CA V4 Model 1',  'classifier_algo': 'densenet', 'classifier_weight_filepath': 'canonical_zebra_grevys_v4:1', 'species': 'zebra_grevys'},
        # {'label': 'CA V4 Model 2',  'classifier_algo': 'densenet', 'classifier_weight_filepath': 'canonical_zebra_grevys_v4:2', 'species': 'zebra_grevys'},
    ]
    color_list = []
    # color_list = [(0, 0, 0)]
    color_list += pt.distinct_colors(len(config_list) - len(color_list), randomize=False)

    axes_ = plt.subplot(221)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall')
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    ret_list = [
        canonical_precision_recall_algo_plot(ibs, color=color, **config)
        for color, config in zip(color_list, config_list)
    ]
    area_list = [ret[0] for ret in ret_list]
    conf_list = [ret[1] for ret in ret_list]
    # index = np.argmax(area_list)
    index = -1
    best_label1 = config_list[index]['label']
    best_config1 = config_list[index]
    best_color1 = color_list[index]
    best_area1 = area_list[index]
    best_conf1 = conf_list[index]
    plt.title(
        'Precision-Recall Curve (Best: %s, AP = %0.02f)' % (best_label1, best_area1,),
        y=1.10,
    )
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(222)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    ret_list = [
        canonical_roc_algo_plot(ibs, color=color, **config)
        for color, config in zip(color_list, config_list)
    ]
    area_list = [ret[0] for ret in ret_list]
    conf_list = [ret[1] for ret in ret_list]
    # index = np.argmax(area_list)
    index = -1
    best_label2 = config_list[index]['label']
    best_config2 = config_list[index]
    best_color2 = color_list[index]
    best_area2 = area_list[index]
    best_conf2 = conf_list[index]
    plt.title('ROC Curve (Best: %s, AP = %0.02f)' % (best_label2, best_area2,), y=1.10)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(223)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = canonical_confusion_matrix_algo_plot(
        ibs,
        color=best_color1,
        conf=best_conf1,
        fig_=fig_,
        axes_=axes_,
        output_cases=True,
        **best_config1,
    )
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0,))
    axes_.set_ylabel('Ground-Truth')
    plt.title('P-R Confusion Matrix (OP = %0.02f)' % (best_conf1,), y=1.12)

    axes_ = plt.subplot(224)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = canonical_confusion_matrix_algo_plot(
        ibs, color=best_color2, conf=best_conf2, fig_=fig_, axes_=axes_, **best_config2
    )
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0,))
    axes_.set_ylabel('Ground-Truth')
    plt.title('ROC Confusion Matrix (OP = %0.02f)' % (best_conf2,), y=1.12)

    fig_filename = 'canonical-precision-recall-roc.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


def _canonical_get_boxes(ibs, gid_list, species):
    from wbia.web.appfuncs import CANONICAL_PART_TYPE

    aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    aid_list = ibs.filter_annotation_set(aid_list, species=species)
    flag_list = ibs.get_annot_canonical(aid_list)
    part_rowids_list = ibs.get_annot_part_rowids(aid_list)
    part_types_list = list(map(ibs.get_part_types, part_rowids_list))

    aid_set = []
    bbox_set = []
    zipped = zip(aid_list, flag_list, part_rowids_list, part_types_list)
    for aid, flag, part_rowid_list, part_type_list in zipped:
        part_rowid_ = None
        if flag:
            for part_rowid, part_type in zip(part_rowid_list, part_type_list):
                if part_type == CANONICAL_PART_TYPE:
                    assert part_rowid_ is None, 'Cannot have multiple CA for one image'
                    part_rowid_ = part_rowid

        if part_rowid_ is not None:
            axtl, aytl, aw, ah = ibs.get_annot_bboxes(aid)
            axbr, aybr = axtl + aw, aytl + ah
            pxtl, pytl, pw, ph = ibs.get_part_bboxes(part_rowid_)
            pxbr, pybr = pxtl + pw, pytl + ph
            x0 = pxtl - axtl
            y0 = pytl - aytl
            x1 = axbr - pxbr
            y1 = aybr - pybr
            x0 = max(x0 / aw, 0.0)
            y0 = max(y0 / ah, 0.0)
            x1 = max(x1 / aw, 0.0)
            y1 = max(y1 / ah, 0.0)
            assert x0 + x1 < 0.99
            assert y0 + y1 < 0.99
            bbox = (x0, y0, x1, y1)
            aid_set.append(aid)
            bbox_set.append(bbox)

    return aid_set, bbox_set


def canonical_localization_deviation_plot(
    ibs, attribute, color, index, label=None, species=None, marker='o', **kwargs
):
    import random
    import matplotlib.pyplot as plt

    assert None not in [label, species]
    print('Processing Deviation for: %r' % (label,))
    depc = ibs.depc_annot

    if attribute == 'x0':
        take_index = 0
    elif attribute == 'y0':
        take_index = 1
    elif attribute == 'x1':
        take_index = 2
    elif attribute == 'y1':
        take_index = 3
    else:
        raise ValueError('attribute not valid')

    test_gid_set_ = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_list_ = list(test_gid_set_)
    test_aid_set, test_bbox_set = _canonical_get_boxes(ibs, test_gid_list_, species)

    value_list = ut.take_column(test_bbox_set, take_index)
    prediction_list = depc.get_property(
        'canonical', test_aid_set, attribute, config=kwargs
    )

    x_list = []
    y_list = []
    overshoot = 0.0
    for value, prediction in zip(value_list, prediction_list):
        x = random.uniform(index, index + 1)
        y = value - prediction
        if y < 0:
            overshoot += 1
        x_list.append(x)
        y_list.append(y)
    mean = np.mean(y_list)
    std = np.std(y_list)
    overshoot /= len(y_list)

    label = '%s (Over: %0.02f, %0.02f+/-%0.02f)' % (label, overshoot, mean, std,)
    plt.plot(
        x_list,
        y_list,
        color=color,
        linestyle='None',
        marker=marker,
        label=label,
        alpha=0.5,
    )

    plt.plot(
        [index, index + 1], [0.0, 0.0], color=(0.2, 0.2, 0.2), linestyle='-', alpha=0.3
    )
    if index % 4 == 3:
        plt.plot(
            [index + 1, index + 1],
            [-1.0, 1.0],
            color=(0.2, 0.2, 0.2),
            linestyle='--',
            alpha=0.1,
        )

    color = 'xkcd:gold'
    marker = 'D'
    plt.errorbar(
        [index + 0.5],
        [mean],
        [std],
        linestyle='None',
        color=color,
        marker=marker,
        zorder=999,
        barsabove=True,
    )
    # plt.plot([index + 0.5], [mean], color=color, marker=marker)


def canonical_localization_iou_plot(
    ibs, color, index, label=None, species=None, marker='o', threshold=0.75, **kwargs
):
    import random
    import matplotlib.pyplot as plt

    def _convert(bbox):
        x0, y0, x1, y1 = bbox
        retval = {
            'xtl': x0,
            'ytl': y0,
            'xbr': 1.0 - x1,
            'ybr': 1.0 - y1,
        }
        retval['width'] = retval['xbr'] - retval['xtl']
        retval['height'] = retval['ybr'] - retval['ytl']
        return retval

    assert None not in [label, species]
    print('Processing IoU for: %r' % (label,))
    depc = ibs.depc_annot

    test_gid_set_ = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_list_ = list(test_gid_set_)
    test_aid_set, test_bbox_set = _canonical_get_boxes(ibs, test_gid_list_, species)

    prediction_list = depc.get_property('canonical', test_aid_set, None, config=kwargs)

    gt_list = [_convert(test_bbox) for test_bbox in test_bbox_set]
    pred_list = [_convert(prediction) for prediction in prediction_list]

    correct = 0.0
    x_list = []
    y_list = []
    for gt, pred in zip(gt_list, pred_list):
        overlap = general_overlap([gt], [pred])
        x = random.uniform(index, index + 1)
        y = overlap[0][0]
        if y >= threshold:
            correct += 1.0
        x_list.append(x)
        y_list.append(y)
    accuracy = correct / len(y_list)
    mean = np.mean(y_list)
    std = np.std(y_list)

    label = '%s (Acc: %0.02f, %0.02f+/-%0.02f)' % (label, accuracy, mean, std,)
    plt.plot(
        x_list,
        y_list,
        color=color,
        linestyle='None',
        marker=marker,
        label=label,
        alpha=0.5,
    )

    for y_value in [0.5, 0.75, 0.9]:
        plt.plot(
            [index, index + 1],
            [y_value, y_value],
            color=(0.2, 0.2, 0.2),
            linestyle='-',
            alpha=0.3,
        )

    if index % 4 == 3:
        plt.plot(
            [index + 1, index + 1],
            [0.0, 1.0],
            color=(0.2, 0.2, 0.2),
            linestyle='--',
            alpha=0.1,
        )

    color = 'xkcd:gold'
    marker = 'D'
    plt.errorbar(
        [index + 0.5],
        [mean],
        [std],
        linestyle='None',
        color=color,
        marker=marker,
        zorder=999,
        barsabove=True,
    )
    # plt.plot([index + 0.5], [mean], color=color, marker=marker)

    return test_aid_set, test_bbox_set, prediction_list, y_list, accuracy


@register_ibs_method
def canonical_localization_iou_visualize(
    ibs,
    index,
    test_aid_set,
    test_bbox_set,
    prediction_list,
    overlap_list,
    color_list,
    label=None,
    species=None,
    **kwargs,
):
    assert None not in [label, species]
    assert len(color_list) == 4
    print('Processing Renderings for: %r' % (label,))

    color_list_ = []
    for color in color_list:
        color_ = []
        for value in color:
            value_ = int(np.around(255.0 * value))
            color_ = [value_] + color_
        color_ = tuple(color_)
        color_list_.append(color_)
    color_list = color_list_

    output_path = expanduser(join('~', 'Desktop', 'canonical-regression-%d' % (index,)))
    ut.delete(output_path)
    ut.ensuredir(output_path)

    config = {
        'dim_size': 600,
        'resize_dim': 'maxwh',
    }
    chip_list = ibs.depc_annot.get_property('chips', test_aid_set, 'img', config=config)
    zipped = list(
        zip(test_aid_set, chip_list, test_bbox_set, prediction_list, overlap_list)
    )

    for test_aid, chip, test_bbox, prediction, overlap in zipped:
        h, w = chip.shape[:2]

        chipa = chip.copy()
        chipb = chip.copy()

        x0a, y0a, x1a, y1a = test_bbox
        x0b, y0b, x1b, y1b = prediction

        x0a = int(np.around(x0a * w))
        y0a = int(np.around(y0a * h))
        x1a = int(np.around(x1a * w))
        y1a = int(np.around(y1a * h))

        x0b = int(np.around(x0b * w))
        y0b = int(np.around(y0b * h))
        x1b = int(np.around(x1b * w))
        y1b = int(np.around(y1b * h))

        x1a = w - x1a
        x1b = w - x1b
        y1a = h - y1a
        y1b = h - y1b

        chipa = cv2.line(chipa, (x0a, y0a), (x0a, y1a), color_list[0], 3)
        chipa = cv2.line(chipa, (x0a, y0a), (x1a, y0a), color_list[1], 3)
        chipa = cv2.line(chipa, (x1a, y0a), (x1a, y1a), color_list[2], 3)
        chipa = cv2.line(chipa, (x0a, y1a), (x1a, y1a), color_list[3], 3)

        chipb = cv2.line(chipb, (x0b, y0b), (x0b, y1b), color_list[0], 3)
        chipb = cv2.line(chipb, (x0b, y0b), (x1b, y0b), color_list[1], 3)
        chipb = cv2.line(chipb, (x1b, y0b), (x1b, y1b), color_list[2], 3)
        chipb = cv2.line(chipb, (x0b, y1b), (x1b, y1b), color_list[3], 3)

        canvas = np.hstack((chipa, chipb))

        canvas_filepath = join(
            output_path,
            'canonical-regression-iou-%0.02f-aid-%s.jpg' % (overlap, test_aid,),
        )
        cv2.imwrite(canvas_filepath, canvas)


@register_ibs_method
def canonical_localization_precision_recall_algo_display(ibs, figsize=(20, 40)):
    import matplotlib.pyplot as plt
    import plottool as pt

    fig_ = plt.figure(figsize=figsize, dpi=400)  # NOQA

    config_list = [
        # {'label': 'CA V1 Ensemble', 'canonical_weight_filepath': 'canonical_zebra_grevys_v1',   'species': 'zebra_grevys'}, # OVER = 1.0, small dataset
        # {'label': 'CA V1 Model 0',  'canonical_weight_filepath': 'canonical_zebra_grevys_v1:0', 'species': 'zebra_grevys'}, # OVER = 1.0, small dataset
        # {'label': 'CA V1 Model 1',  'canonical_weight_filepath': 'canonical_zebra_grevys_v1:1', 'species': 'zebra_grevys'}, # OVER = 1.0, small dataset
        # {'label': 'CA V1 Model 2',  'canonical_weight_filepath': 'canonical_zebra_grevys_v1:2', 'species': 'zebra_grevys'}, # OVER = 1.0, small dataset
        # {'label': 'CA V2 Ensemble', 'canonical_weight_filepath': 'canonical_zebra_grevys_v2',   'species': 'zebra_grevys'}, # OVER = 1.0, large dataset
        # {'label': 'CA V2 Model 0',  'canonical_weight_filepath': 'canonical_zebra_grevys_v2:0', 'species': 'zebra_grevys'}, # OVER = 1.0, large dataset
        # {'label': 'CA V2 Model 1',  'canonical_weight_filepath': 'canonical_zebra_grevys_v2:1', 'species': 'zebra_grevys'}, # OVER = 1.0, large dataset
        # {'label': 'CA V2 Model 2',  'canonical_weight_filepath': 'canonical_zebra_grevys_v2:2', 'species': 'zebra_grevys'}, # OVER = 1.0, large dataset
        # {'label': 'CA V3 Ensemble', 'canonical_weight_filepath': 'canonical_zebra_grevys_v3',   'species': 'zebra_grevys'},  # OVER = 2.0
        # {'label': 'CA V3 Model 0',  'canonical_weight_filepath': 'canonical_zebra_grevys_v3:0', 'species': 'zebra_grevys'},  # OVER = 2.0
        # {'label': 'CA V3 Model 1',  'canonical_weight_filepath': 'canonical_zebra_grevys_v3:1', 'species': 'zebra_grevys'},  # OVER = 2.0
        # {'label': 'CA V3 Model 2',  'canonical_weight_filepath': 'canonical_zebra_grevys_v3:2', 'species': 'zebra_grevys'},  # OVER = 2.0
        {
            'label': 'CA V5-1.0 Ens.',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v5',
            'species': 'zebra_grevys',
        },  # OVER = 1.0
        {
            'label': 'CA V5-1.0 M0',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v5:0',
            'species': 'zebra_grevys',
        },  # OVER = 1.0
        {
            'label': 'CA V5-1.0 M1',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v5:1',
            'species': 'zebra_grevys',
        },  # OVER = 1.0
        {
            'label': 'CA V5-1.0 M2',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v5:2',
            'species': 'zebra_grevys',
        },  # OVER = 1.0
        {
            'label': 'CA V6-2.0 Ens.',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v6',
            'species': 'zebra_grevys',
        },  # OVER = 2.0
        {
            'label': 'CA V6-2.0 M0',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v6:0',
            'species': 'zebra_grevys',
        },  # OVER = 2.0
        {
            'label': 'CA V6-2.0 M1',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v6:1',
            'species': 'zebra_grevys',
        },  # OVER = 2.0
        {
            'label': 'CA V6-2.0 M2',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v6:2',
            'species': 'zebra_grevys',
        },  # OVER = 2.0
        {
            'label': 'CA V4-4.0 Ens.',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v4',
            'species': 'zebra_grevys',
        },  # OVER = 4.0
        {
            'label': 'CA V4-4.0 M0',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v4:0',
            'species': 'zebra_grevys',
        },  # OVER = 4.0
        {
            'label': 'CA V4-4.0 M1',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v4:1',
            'species': 'zebra_grevys',
        },  # OVER = 4.0
        {
            'label': 'CA V4-4.0 M2',
            'canonical_weight_filepath': 'canonical_zebra_grevys_v4:2',
            'species': 'zebra_grevys',
        },  # OVER = 4.0
    ]
    color_list = []
    # color_list = [(0, 0, 0)]
    color_list += pt.distinct_colors(len(config_list) - len(color_list), randomize=False)

    min_, max_ = -1.0, 1.0

    axes_ = plt.subplot(321)
    axes_.grid(True, which='major')
    axes_.grid(False, which='minor')
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.get_xaxis().set_ticks([])
    axes_.set_ylabel('GT - Pred Deviation (in percentages)')
    axes_.set_xlim([0.0, len(config_list)])
    axes_.set_ylim([min_, max_])
    axes_.fill_between([0.0, len(config_list)], -1, 0, facecolor='red', alpha=0.1)
    for index, (color, config) in enumerate(zip(color_list, config_list)):
        canonical_localization_deviation_plot(
            ibs, 'x0', color=color, index=index, **config
        )

    plt.title('X0 Deviation Scatter Plot')
    plt.legend(
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(322)
    axes_.grid(True, which='major')
    axes_.grid(False, which='minor')
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.get_xaxis().set_ticks([])
    axes_.set_ylabel('GT - Pred Deviation (in percentages)')
    axes_.set_xlim([0.0, len(config_list)])
    axes_.set_ylim([min_, max_])
    axes_.fill_between([0.0, len(config_list)], -1, 0, facecolor='red', alpha=0.1)
    for index, (color, config) in enumerate(zip(color_list, config_list)):
        canonical_localization_deviation_plot(
            ibs, 'x1', color=color, index=index, **config
        )

    plt.title('Y0 Deviation Scatter Plot')
    plt.legend(
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(323)
    axes_.grid(True, which='major')
    axes_.grid(False, which='minor')
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.get_xaxis().set_ticks([])
    axes_.set_ylabel('GT - Pred Deviation (in percentages)')
    axes_.set_xlim([0.0, len(config_list)])
    axes_.set_ylim([min_, max_])
    axes_.fill_between([0.0, len(config_list)], -1, 0, facecolor='red', alpha=0.1)
    for index, (color, config) in enumerate(zip(color_list, config_list)):
        canonical_localization_deviation_plot(
            ibs, 'y0', color=color, index=index, **config
        )

    plt.title('X1 Deviation Scatter Plot')
    plt.legend(
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(324)
    axes_.grid(True, which='major')
    axes_.grid(False, which='minor')
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.get_xaxis().set_ticks([])
    axes_.set_ylabel('GT - Pred Deviation (in percentages)')
    axes_.set_xlim([0.0, len(config_list)])
    axes_.set_ylim([min_, max_])
    axes_.fill_between([0.0, len(config_list)], -1, 0, facecolor='red', alpha=0.1)
    for index, (color, config) in enumerate(zip(color_list, config_list)):
        canonical_localization_deviation_plot(
            ibs, 'y1', color=color, index=index, **config
        )

    plt.title('Y1 Deviation Scatter Plot')
    plt.legend(
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(325)
    axes_.grid(True, which='major')
    axes_.grid(False, which='minor')
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.get_xaxis().set_ticks([])
    axes_.set_ylabel('GT - Pred Deviation (in percentages)')
    axes_.set_xlim([0.0, len(config_list)])
    axes_.set_ylim([min_, max_])
    axes_.fill_between([0.0, len(config_list)], -1, 0, facecolor='red', alpha=0.1)

    assert len(config_list) % 4 == 0
    rounds = len(config_list) // 4
    colors = pt.distinct_colors(4, randomize=False)

    attribute_list = []
    color_list_ = []
    for _ in range(rounds):
        attribute_list += ['x0', 'y0', 'x1', 'y1']
        color_list_ += colors

    for index, (attribute, color_) in enumerate(zip(attribute_list, color_list_)):
        index_ = (index // 4) * 4
        config_ = config_list[index_].copy()
        config_['label'] = '%s %s' % (config_['label'], attribute,)
        canonical_localization_deviation_plot(
            ibs, attribute, color=color_, index=index, **config_
        )

    plt.title('Ensemble Deviation Scatter Plot')
    plt.legend(
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(326)
    axes_.grid(True, which='major')
    axes_.grid(False, which='minor')
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.get_xaxis().set_ticks([])
    axes_.set_ylabel('IoU')
    axes_.set_xlim([0.0, len(config_list)])
    axes_.set_ylim([0.0, 1.0])

    for index, (color, config) in enumerate(zip(color_list, config_list)):
        values_ = canonical_localization_iou_plot(ibs, color=color, index=index, **config)
        if index % 4 == 0:
            config_ = config_list[index]
            test_aid_set, test_bbox_set, prediction_list, y_list, accuracy = values_
            ibs.canonical_localization_iou_visualize(
                index,
                test_aid_set,
                test_bbox_set,
                prediction_list,
                y_list,
                colors,
                **config_,
            )

    plt.title('IoU Scatter Plot')
    plt.legend(
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    fig_filename = 'canonical-localization-deviance.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


@register_ibs_method
def background_accuracy_display(ibs, category_list, test_gid_set=None, output_path=None):

    if output_path is None:
        output_path = abspath(expanduser(join('~', 'Desktop', 'background')))
        ut.ensuredir(output_path)

    if test_gid_set is None:
        test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
        test_gid_set = list(test_gid_set)
    aids_list = ibs.get_image_aids(test_gid_set)
    aid_list = ut.flatten(aids_list)
    species_list = ibs.get_annot_species_texts(aid_list)

    aid_list = [
        aid for aid, species in zip(aid_list, species_list) if species in category_list
    ]
    species_list = ibs.get_annot_species_texts(aid_list)
    gid_list = ibs.get_annot_gids(aid_list)

    config2_ = {'fw_detector': 'cnn'}
    hough_cpath_list = ibs.get_annot_probchip_fpath(aid_list, config2_=config2_)
    image_list = [vt.imread(hough_cpath) for hough_cpath in hough_cpath_list]
    chip_list = ibs.get_annot_chips(aid_list, config2_=config2_)
    zipped = zip(aid_list, gid_list, species_list, image_list, chip_list)
    for index, (aid, gid, species, image, chip) in enumerate(zipped):
        print(index)
        mask = vt.resize_mask(image, chip)
        blended = vt.blend_images_multiply(chip, mask)
        blended *= 255.0
        blended = np.around(blended)
        blended[blended < 0] = 0
        blended[blended > 255] = 255
        blended = blended.astype(np.uint8)

        canvas = np.hstack((chip, mask, blended))
        output_filepath = join(
            output_path, 'background.%s.%d.%d.png' % (species, gid, aid,)
        )
        cv2.imwrite(output_filepath, canvas)


def aoi2_precision_recall_algo(ibs, category_list=None, test_gid_set_=None, **kwargs):
    depc = ibs.depc_annot
    if test_gid_set_ is None:
        test_gid_set_ = general_get_imageset_gids(ibs, 'TEST_SET')
    test_aid_list_ = list(set(ut.flatten(ibs.get_image_aids(test_gid_set_))))
    species_list = ibs.get_annot_species_texts(test_aid_list_)
    interest_list = ibs.get_annot_interest(test_aid_list_)

    test_aid_list = []
    label_list = []
    for test_aid, species, interest in zip(test_aid_list_, species_list, interest_list):
        if category_list is not None:
            if species not in category_list:
                continue
        if interest is None:
            continue
        label = 'positive' if interest else 'negative'
        test_aid_list.append(test_aid)
        label_list.append(label)

    prediction_list = depc.get_property('aoi_two', test_aid_list, 'class', config=kwargs)
    confidence_list = depc.get_property('aoi_two', test_aid_list, 'score', config=kwargs)
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence in zip(prediction_list, confidence_list)
    ]
    return general_precision_recall_algo(ibs, label_list, confidence_list, **kwargs)


def aoi2_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing Precision-Recall for: %r' % (label,))
    conf_list, pr_list, re_list, tpr_list, fpr_list = aoi2_precision_recall_algo(
        ibs, **kwargs
    )
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def aoi2_roc_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing ROC for: %r' % (label,))
    conf_list, pr_list, re_list, tpr_list, fpr_list = aoi2_precision_recall_algo(
        ibs, **kwargs
    )
    return general_area_best_conf(
        conf_list, fpr_list, tpr_list, interpolate=False, target=(0.0, 1.0), **kwargs
    )


def aoi2_confusion_matrix_algo_plot(
    ibs,
    label,
    color,
    conf,
    output_cases=False,
    category_list=None,
    test_gid_set_=None,
    **kwargs,
):
    print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf,))
    depc = ibs.depc_annot
    if test_gid_set_ is None:
        test_gid_set_ = general_get_imageset_gids(ibs, 'TEST_SET')
    test_aid_list_ = list(set(ut.flatten(ibs.get_image_aids(test_gid_set_))))
    species_list = ibs.get_annot_species_texts(test_aid_list_)
    interest_list = ibs.get_annot_interest(test_aid_list_)

    test_aid_list = []
    label_list = []
    for test_aid, species, interest in zip(test_aid_list_, species_list, interest_list):
        if category_list is not None:
            if species not in category_list:
                continue
        if interest is None:
            continue
        label = 'positive' if interest else 'negative'
        test_aid_list.append(test_aid)
        label_list.append(label)

    prediction_list = depc.get_property('aoi_two', test_aid_list, 'class', config=kwargs)
    confidence_list = depc.get_property('aoi_two', test_aid_list, 'score', config=kwargs)
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence in zip(prediction_list, confidence_list)
    ]
    prediction_list = [
        'positive' if confidence >= conf else 'negative' for confidence in confidence_list
    ]

    if output_cases:
        output_path = 'aoi2-confusion-incorrect'
        output_path = abspath(expanduser(join('~', 'Desktop', output_path)))
        ut.delete(output_path)
        ut.ensuredir(output_path)

        manifest_dict = {}
        test_gid_list = ibs.get_annot_gids(test_aid_list)
        zipped = zip(test_gid_list, test_aid_list, label_list, prediction_list)
        for test_gid, test_aid, label, prediction in zipped:
            if test_gid not in manifest_dict:
                manifest_dict[test_gid] = {}
            assert test_aid not in manifest_dict[test_gid]
            manifest_dict[test_gid][test_aid] = (
                label,
                prediction,
            )

        for test_gid in manifest_dict:
            image = ibs.get_images(test_gid)
            w, h = ibs.get_image_sizes(test_gid)
            image = _resize(image, t_width=600, verbose=False)
            height_, width_, channels_ = image.shape

            for test_aid in manifest_dict[test_gid]:
                label, prediction = manifest_dict[test_gid][test_aid]
                bbox = ibs.get_annot_bboxes(test_aid)
                xtl, ytl, width, height = bbox
                xbr = xtl + width
                ybr = ytl + height

                xtl = int(np.round((xtl / w) * width_))
                ytl = int(np.round((ytl / h) * height_))
                xbr = int(np.round((xbr / w) * width_))
                ybr = int(np.round((ybr / h) * height_))
                if label == 'positive':
                    color = (255, 99, 46)
                else:
                    color = (127, 255, 127)
                cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, 4)
                if prediction == 'positive':
                    color = (255, 99, 46)
                else:
                    color = (127, 255, 127)
                cv2.rectangle(image, (xtl - 4, ytl - 4), (xbr + 4, ybr + 4), color, 4)

            image_filename = 'image_%d.png' % (test_gid,)
            image_filepath = join(output_path, image_filename)
            cv2.imwrite(image_filepath, image)

    category_list = ['positive', 'negative']
    category_mapping = {
        'positive': 0,
        'negative': 1,
    }
    return general_confusion_matrix_algo(
        label_list, prediction_list, category_list, category_mapping, size=20, **kwargs
    )


@register_ibs_method
def aoi2_precision_recall_algo_display(
    ibs, test_gid_list=None, output_cases=False, figsize=(20, 20)
):
    import matplotlib.pyplot as plt
    import wbia.plottool as pt

    fig_ = plt.figure(figsize=figsize)

    test_gid_set = None if test_gid_list is None else sorted(set(test_gid_list))
    config_list = [
        # {'label': 'All Species',         'aoi_two_weight_filepath': 'ggr2', 'category_list': None},
        # {'label': 'Masai Giraffe',       'aoi_two_weight_filepath': 'ggr2', 'category_list': ['giraffe_masai']},
        # {'label': 'Reticulated Giraffe', 'aoi_two_weight_filepath': 'ggr2', 'category_list': ['giraffe_reticulated']},
        # {'label': 'Sea Turtle',          'aoi_two_weight_filepath': 'ggr2', 'category_list': ['turtle_sea']},
        # {'label': 'Whale Fluke',         'aoi_two_weight_filepath': 'ggr2', 'category_list': ['whale_fluke']},
        # {'label': 'Grevy\'s Zebra',      'aoi_two_weight_filepath': 'ggr2', 'category_list': ['zebra_grevys']},
        # {'label': 'Plains Zebra',        'aoi_two_weight_filepath': 'ggr2', 'category_list': ['zebra_plains']},
        # {'label': 'Hammerhead',        'aoi_two_weight_filepath': 'hammerhead', 'category_list': ['shark_hammerhead']},
        {
            'label': 'Jaguar',
            'aoi_two_weight_filepath': 'jaguar',
            'category_list': ['jaguar'],
        },
    ]
    color_list = [(0, 0, 0)]
    color_list += pt.distinct_colors(len(config_list) - len(color_list), randomize=False)

    axes_ = plt.subplot(221)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall')
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    ret_list = [
        aoi2_precision_recall_algo_plot(
            ibs, color=color, test_gid_set_=test_gid_set, **config
        )
        for color, config in zip(color_list, config_list)
    ]
    area_list = [ret[0] for ret in ret_list]
    conf_list = [ret[1] for ret in ret_list]
    # index = np.argmax(area_list)
    index = 0
    best_label1 = config_list[index]['label']
    best_config1 = config_list[index]
    best_color1 = color_list[index]
    best_area1 = area_list[index]
    best_conf1 = conf_list[index]
    plt.title(
        'Precision-Recall Curve (Best: %s, AP = %0.02f)' % (best_label1, best_area1,),
        y=1.10,
    )
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )

    axes_ = plt.subplot(222)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    ret_list = [
        aoi2_roc_algo_plot(ibs, color=color, **config)
        for color, config in zip(color_list, config_list)
    ]
    area_list = [ret[0] for ret in ret_list]
    conf_list = [ret[1] for ret in ret_list]
    # index = np.argmax(area_list)
    index = 0
    best_label2 = config_list[index]['label']
    best_config2 = config_list[index]
    best_color2 = color_list[index]
    best_area2 = area_list[index]
    best_conf2 = conf_list[index]
    plt.title('ROC Curve (Best: %s, AP = %0.02f)' % (best_label2, best_area2,), y=1.10)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.0,
    )
    plt.plot([0.0, 1.0], [0.0, 1.0], color=(0.5, 0.5, 0.5), linestyle='--')
    axes_ = plt.subplot(223)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = aoi2_confusion_matrix_algo_plot(
        ibs,
        color=best_color1,
        conf=best_conf1,
        fig_=fig_,
        axes_=axes_,
        output_cases=output_cases,
        test_gid_set_=test_gid_set,
        **best_config1,
    )
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0,))
    axes_.set_ylabel('Ground-Truth')
    plt.title('P-R Confusion Matrix (OP = %0.02f)' % (best_conf1,), y=1.12)

    axes_ = plt.subplot(224)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = aoi2_confusion_matrix_algo_plot(
        ibs,
        color=best_color2,
        conf=best_conf2,
        fig_=fig_,
        axes_=axes_,
        test_gid_set_=test_gid_set,
        **best_config2,
    )
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0,))
    axes_.set_ylabel('Ground-Truth')
    plt.title('ROC Confusion Matrix (OP = %0.02f)' % (best_conf2,), y=1.12)

    fig_filename = 'aoi2-precision-recall-roc.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


def detector_parse_gt(ibs, test_gid_list=None, **kwargs):
    if test_gid_list is None:
        test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
    uuid_list = ibs.get_image_uuids(test_gid_list)
    gid_list = ibs.get_image_gids_from_uuid(uuid_list)

    gt_dict = {}
    for gid, uuid in zip(gid_list, uuid_list):
        width, height = ibs.get_image_sizes(gid)
        aid_list = ibs.get_image_aids(gid)
        gt_list = []
        for aid in aid_list:
            bbox = ibs.get_annot_bboxes(aid)
            temp = {
                'gid': gid,
                'xtl': bbox[0] / width,
                'ytl': bbox[1] / height,
                'xbr': (bbox[0] + bbox[2]) / width,
                'ybr': (bbox[1] + bbox[3]) / height,
                'width': bbox[2] / width,
                'height': bbox[3] / height,
                'class': ibs.get_annot_species_texts(aid),
                'viewpoint': ibs.get_annot_viewpoints(aid),
                'confidence': 1.0,
            }
            gt_list.append(temp)
        gt_dict[uuid] = gt_list
    return gt_dict


# def detector_parse_pred(ibs, test_gid_list=None, **kwargs):
#     depc = ibs.depc_image

#     if test_gid_list is None:
#         test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
#     uuid_list = ibs.get_image_uuids(test_gid_list)

#     # depc.delete_property('detections', test_gid_list, config=kwargs)
#     results_list = depc.get_property('detections', test_gid_list, None, config=kwargs)
#     size_list = ibs.get_image_sizes(test_gid_list)
#     zipped_list = zip(results_list)
#     # Reformat results for json
#     results_list = [
#         [
#             {
#                 'gid'        : test_gid,
#                 'xtl'        : bbox[0] / width,
#                 'ytl'        : bbox[1] / height,
#                 'width'      : bbox[2] / width,
#                 'height'     : bbox[3] / height,
#                 'theta'      : theta,  # round(theta, 4),
#                 'confidence' : conf,   # round(conf, 4),
#                 'class'      : class_,
#                 'viewpoint'  : viewpoint,
#             }
#             for bbox, theta, class_, viewpoint, conf in zip(*zipped[0][1:])
#         ]
#         for zipped, (width, height), test_gid in zip(zipped_list, size_list, test_gid_list)
#     ]

#     pred_dict = {
#         uuid_ : result_list
#         for uuid_, result_list in zip(uuid_list, results_list)
#     }
#     # print(pred_dict)
#     return pred_dict


# def detector_precision_recall_algo(ibs, samples=SAMPLES, force_serial=FORCE_SERIAL, **kwargs):
#     test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
#     uuid_list = ibs.get_image_uuids(test_gid_list)

#     print('\tGather Ground-Truth')
#     gt_dict = detector_parse_gt(ibs, test_gid_list=test_gid_list)

#     print('\tGather Predictions')
#     pred_dict = detector_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

#     print('\tGenerate Curves...')
#     conf_list = [ _ / float(samples) for _ in range(0, int(samples) + 1) ]
#     conf_list = sorted(conf_list, reverse=True)

#     uuid_list_list = [ uuid_list for _ in conf_list ]
#     gt_dict_list   = [ gt_dict   for _ in conf_list ]
#     pred_dict_list = [ pred_dict for _ in conf_list ]
#     kwargs_list    = [ kwargs    for _ in conf_list ]
#     arg_iter = zip(conf_list, uuid_list_list, gt_dict_list, pred_dict_list, kwargs_list)
#     pr_re_gen = ut.generate2(detector_precision_recall_algo_worker, arg_iter,
#                              nTasks=len(conf_list), ordered=True,
#                              chunksize=CHUNK_SIZE, force_serial=force_serial)

#     conf_list_ = [-1.0, -1.0]
#     pr_list = [1.0, 0.0]
#     re_list = [0.0, 1.0]
#     # conf_list_ = []
#     # pr_list = []
#     # re_list = []
#     for conf, pr, re in pr_re_gen:
#         conf_list_.append(conf)
#         pr_list.append(pr)
#         re_list.append(re)

#     print('...complete')
#     return conf_list_, pr_list, re_list


# def detector_precision_recall_algo_worker(conf, uuid_list, gt_dict, pred_dict,
#                                           kwargs):
#     tp, fp, fn = 0.0, 0.0, 0.0
#     for index, uuid_ in enumerate(uuid_list):
#         if uuid_ in pred_dict:
#             pred_list = [
#                 pred
#                 for pred in pred_dict[uuid_]
#                 if pred['confidence'] >= conf
#             ]
#             tp_, fp_, fn_ = general_tp_fp_fn(gt_dict[uuid_], pred_list, **kwargs)
#             tp += tp_
#             fp += fp_
#             fn += fn_
#     pr = tp / (tp + fp)
#     re = tp / (tp + fn)
#     return (conf, pr, re)


# def detector_precision_recall_algo_plot(ibs, **kwargs):
#     label = kwargs['label']
#     print('Processing Precision-Recall for: %r' % (label, ))
#     conf_list, pr_list, re_list = detector_precision_recall_algo(ibs, **kwargs)
#     return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


# def detector_confusion_matrix_algo_plot(ibs, label, color, conf, **kwargs):
#     print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf, ))

#     test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
#     uuid_list = ibs.get_image_uuids(test_gid_list)

#     print('\tGather Ground-Truth')
#     gt_dict = detector_parse_gt(ibs, test_gid_list=test_gid_list)

#     print('\tGather Predictions')
#     pred_dict = detector_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

#     label_list = []
#     prediction_list = []
#     for index, uuid_ in enumerate(uuid_list):
#         if uuid_ in pred_dict:
#             gt_list = gt_dict[uuid_]
#             pred_list = [
#                 pred
#                 for pred in pred_dict[uuid_]
#                 if pred['confidence'] >= conf
#             ]
#             tp, fp, fn = general_tp_fp_fn(gt_list, pred_list, **kwargs)
#             for _ in range(int(tp)):
#                 label_list.append('positive')
#                 prediction_list.append('positive')
#             for _ in range(int(fp)):
#                 label_list.append('negative')
#                 prediction_list.append('positive')
#             for _ in range(int(fn)):
#                 label_list.append('positive')
#                 prediction_list.append('negative')

#     category_list = ['positive', 'negative']
#     category_mapping = {
#         'positive': 0,
#         'negative': 1,
#     }
#     return general_confusion_matrix_algo(label_list, prediction_list, category_list,
#                                          category_mapping, **kwargs)


# @register_ibs_method
# def detector_precision_recall_algo_display(ibs, min_overlap=0.5, figsize=(24, 7), **kwargs):
#     import matplotlib.pyplot as plt

#     fig_ = plt.figure(figsize=figsize)

#     axes_ = plt.subplot(131)
#     axes_.set_autoscalex_on(False)
#     axes_.set_autoscaley_on(False)
#     axes_.set_xlabel('Recall (Ground-Truth IOU >= %0.02f)' % (min_overlap, ))
#     axes_.set_ylabel('Precision')
#     axes_.set_xlim([0.0, 1.01])
#     axes_.set_ylim([0.0, 1.01])

#     kwargs_list = [
#         {
#             'min_overlap'            : min_overlap,
#             'classifier_sensitivity' : 0.64,
#             'localizer_grid'         : False,
#             'localizer_sensitivity'  : 0.16,
#             'labeler_sensitivity'    : 0.42,
#         },
#         {
#             'min_overlap'            : min_overlap,
#             'classifier_sensitivity' : 0.64,
#             'localizer_grid'         : False,
#             'localizer_sensitivity'  : 0.16,
#             'labeler_sensitivity'    : 0.42,
#             'check_species'          : True,
#         },
#         {
#             'min_overlap'            : min_overlap,
#             'classifier_sensitivity' : 0.64,
#             'localizer_grid'         : False,
#             'localizer_sensitivity'  : 0.16,
#             'labeler_sensitivity'    : 0.42,
#             'check_viewpoint'        : True,
#         },
#         {
#             'min_overlap'            : min_overlap,
#             'classifier_sensitivity' : 0.04,
#             'localizer_grid'         : True,
#             'localizer_sensitivity'  : 0.05,
#             'labeler_sensitivity'    : 0.39,
#         },
#         {
#             'min_overlap'            : min_overlap,
#             'classifier_sensitivity' : 0.04,
#             'localizer_grid'         : True,
#             'localizer_sensitivity'  : 0.05,
#             'labeler_sensitivity'    : 0.39,
#             'check_species'          : True,
#         },
#         {
#             'min_overlap'            : min_overlap,
#             'classifier_sensitivity' : 0.04,
#             'localizer_grid'         : True,
#             'localizer_sensitivity'  : 0.05,
#             'labeler_sensitivity'    : 0.39,
#             'check_viewpoint'        : True,
#         },
#     ]
#     label_list = [
#         'Opt L',
#         'Opt L+S',
#         'Opt L+S+V',
#         'Rec L',
#         'Rec L+S',
#         'Rec L+S+V',
#     ]
#     color_list = [
#         'r',
#         'b',
#         'g',
#         'k',
#         'y',
#         'c',
#     ]
#     ret_list = [
#         detector_precision_recall_algo_plot(ibs, label=label, color=color, **kwargs_)
#         for label, color, kwargs_ in zip(label_list, color_list, kwargs_list)
#     ]

#     area_list = [ ret[0] for ret in ret_list ]
#     conf_list = [ ret[1] for ret in ret_list ]
#     index = np.argmax(area_list)
#     best_label = label_list[index]
#     best_kwargs = kwargs_list[index]
#     best_area = area_list[index]
#     best_conf = conf_list[index]
#     plt.title('Precision-Recall Curve (Best: %s, AP = %0.02f)' % (best_label, best_area, ), y=1.20)
#     # Display graph
#     plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
#                borderaxespad=0.0)

#     axes_ = plt.subplot(132)
#     axes_.set_aspect(1)
#     gca_ = plt.gca()
#     gca_.grid(False)
#     correct_rate, _ = detector_confusion_matrix_algo_plot(ibs, 'V1', 'r', conf=best_conf, fig_=fig_, axes_=axes_, **best_kwargs)
#     axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
#     axes_.set_ylabel('Ground-Truth')
#     plt.title('P-R Confusion Matrix (Algo: %s, OP = %0.02f)' % (best_label, best_conf, ), y=1.26)

#     best_index = None
#     best_conf = None
#     best_pr = 0.0
#     best_re = 0.0
#     tup_list  = [ ret[2] for ret in ret_list ]
#     for index, tup in enumerate(tup_list):
#         for conf, re, pr in zip(*tup):
#             if pr > best_pr:
#                 best_index = index
#                 best_conf = conf
#                 best_pr = pr
#                 best_re = re

#     if best_index is not None:
#         axes_ = plt.subplot(131)
#         plt.plot([best_re], [best_pr], 'yo')

#         best_label = label_list[best_index]
#         best_kwargs = kwargs_list[best_index]

#         axes_ = plt.subplot(133)
#         axes_.set_aspect(1)
#         gca_ = plt.gca()
#         gca_.grid(False)
#         correct_rate, _ = detector_confusion_matrix_algo_plot(ibs, 'V1', 'r', conf=best_conf, fig_=fig_, axes_=axes_, **best_kwargs)
#         axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
#         axes_.set_ylabel('Ground-Truth')
#         plt.title('P-R Confusion Matrix (Algo: %s, OP = %0.02f)' % (best_label, best_conf, ), y=1.26)

#     # plt.show()
#     fig_filename = 'detector-precision-recall-%0.2f.png' % (min_overlap, )
#     fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
#     plt.savefig(fig_path, bbox_inches='tight')


# @register_ibs_method
# def detector_metric_graphs(ibs, species_list=[]):
#     ibs.classifier_precision_recall_algo_display(species_list)
#     ibs.localizer_precision_recall_algo_display()
#     ibs.labeler_precision_recall_algo_display()
#     ibs.detector_precision_recall_algo_display()


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.other.detectfuncs
        python -m wbia.other.detectfuncs --allexamples
        python -m wbia.other.detectfuncs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
