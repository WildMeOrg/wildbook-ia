# -*- coding: utf-8 -*-
"""
developer convenience functions for ibs

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
import vtool as vt
import utool as ut
import cv2
try:
    from detecttools.pypascalmarkup import PascalVOC_Markup_Annotation
except ImportError as ex:
    ut.printex('COMMIT TO DETECTTOOLS')
    pass
from ibeis.control import controller_inject
from ibeis import annotmatch_funcs  # NOQA

# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[other.detectfuncs]')


# Must import class before injection
CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))


@register_ibs_method
def export_to_xml(ibs, offset='auto', enforce_yaw=False, target_size=500, purge=False):
    import random
    from datetime import date

    current_year = int(date.today().year)
    # target_size = 900
    information = {
        'database_name' : ibs.get_dbname()
    }
    datadir = ibs._ibsdb + '/LearningData/'
    imagedir = datadir + 'JPEGImages/'
    annotdir = datadir + 'Annotations/'
    setsdir = datadir + 'ImageSets/'
    mainsetsdir = setsdir + 'Main/'
    if purge:
        ut.delete(datadir)
    ut.ensuredir(datadir)
    ut.ensuredir(imagedir)
    ut.ensuredir(annotdir)
    ut.ensuredir(setsdir)
    ut.ensuredir(mainsetsdir)
    gid_list = ibs.get_valid_gids()
    sets_dict = {
        'test'     : [],
        'train'    : [],
        'trainval' : [],
        'val'      : [],
    }
    index = 1 if offset == 'auto' else offset
    try:
        train_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET')))
        test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
    except:
        train_gid_set = set([])
        test_gid_set = set([])

    print('Exporting %d images' % (len(gid_list),))
    for gid in gid_list:
        yawed = True
        aid_list = ibs.get_image_aids(gid)
        image_uri = ibs.get_image_uris(gid)
        image_path = ibs.get_image_paths(gid)
        if len(aid_list) > -1:
            fulldir = image_path.split('/')
            filename = fulldir.pop()
            extension = filename.split('.')[-1]  # NOQA
            out_name = "%d_%06d" % (current_year, index, )
            out_img = out_name + ".jpg"
            folder = "IBEIS"

            _image = ibs.get_images(gid)
            height, width, channels = _image.shape

            if width > height:
                ratio = height / width
                decrease = target_size / width
                width = target_size
                height = int(target_size * ratio)
            else:
                ratio = width / height
                decrease = target_size / height
                height = target_size
                width = int(target_size * ratio)

            dst_img = imagedir + out_img
            _image = vt.resize(_image, (width, height))
            vt.imwrite(dst_img, _image)

            annotation = PascalVOC_Markup_Annotation(dst_img, folder, out_img,
                                                     source=image_uri,
                                                     **information)
            bbox_list = ibs.get_annot_bboxes(aid_list)
            theta_list = ibs.get_annot_thetas(aid_list)
            for aid, bbox, theta in zip(aid_list, bbox_list, theta_list):
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
                xmin = int(min(x_points) * decrease)
                xmax = int(max(x_points) * decrease)
                ymin = int(min(y_points) * decrease)
                ymax = int(max(y_points) * decrease)
                # Bounds check
                xmin = max(xmin, 0)
                ymin = max(ymin, 0)
                xmax = min(xmax, width - 1)
                ymax = min(ymax, height - 1)
                #TODO: Change species_name to getter in IBEISControl once
                #implemented
                #species_name = 'grevys_zebra'
                info = {}
                species_name = ibs.get_annot_species_texts(aid)
                # if species_name not in ['zebra_plains', 'zebra_grevys']:
                #     species_name = 'unspecified'
                # yaw = ibs.get_annot_yaw_texts(aid)
                # if yaw != '' and yaw is not None:
                #     info['pose'] = yaw
                yaw = ibs.get_annot_yaws(aid)
                if yaw != -1 and yaw is not None:
                    info['pose'] = '%0.06f' % (yaw, )
                else:
                    yawed = False
                    print("UNVIEWPOINTED: %d " % gid)
                annotation.add_object(
                    species_name, (xmax, xmin, ymax, ymin), **info)
            dst_annot = annotdir + out_name  + '.xml'

            # # Update sets
            # state = random.uniform(0.0, 1.0)
            # if state <= 0.50:
            #     sets_dict['test'].append(out_name)
            # elif state <= 0.75:
            #     sets_dict['train'].append(out_name)
            #     sets_dict['trainval'].append(out_name)
            # else:
            #     sets_dict['val'].append(out_name)
            #     sets_dict['trainval'].append(out_name)

            if gid in test_gid_set:
                sets_dict['test'].append(out_name)
            elif True or gid in train_gid_set:
                state = random.uniform(0.0, 1.0)
                if state <= 0.75:
                    sets_dict['train'].append(out_name)
                    sets_dict['trainval'].append(out_name)
                else:
                    sets_dict['val'].append(out_name)
                    sets_dict['trainval'].append(out_name)
            else:
                raise NotImplementedError()

            # Write XML
            if True or not enforce_yaw or yawed:
                print("Copying:\n%r\n%r\n%r\n\n" % (
                    image_path, dst_img, (width, height), ))
                xml_data = open(dst_annot, 'w')
                xml_data.write(annotation.xml())
                xml_data.close()
                while exists(dst_annot):
                    index += 1
                    if offset != 'auto':
                        break
                    out_name = "%d_%06d" % (current_year, index, )
                    dst_annot = annotdir + out_name  + '.xml'
        else:
            print("Skipping:\n%r\n\n" % (image_path, ))

    for key in sets_dict.keys():
        with open(mainsetsdir + key + '.txt', 'w') as file_:
            sets_dict[key].append('')
            content = sets_dict[key]
            content = '\n'.join(content)
            file_.write(content)

    print('...completed')
    return datadir


@register_ibs_method
def imageset_train_test_split(ibs, train_split=0.8):
    from random import shuffle
    gid_list = ibs.get_valid_gids()
    aids_list = ibs.get_image_aids(gid_list)
    distro_dict = {}
    for gid, aid_list in zip(gid_list, aids_list):
        total = len(aid_list)
        if total not in distro_dict:
            distro_dict[total] = []
        distro_dict[total].append(gid)

    print('Processing train/test imagesets...')
    global_train_list = []
    global_test_list = []
    for distro, gid_list_ in distro_dict.iteritems():
        total = len(gid_list_)
        shuffle(gid_list_)
        split_index = total * (1.0 - train_split) + 1E-9  # weird
        if split_index < 1.0:
            split_index = total / 2
        else:
            split_index = np.around(split_index)
        split_index = int(split_index)
        args = (distro, total, split_index, )
        print('\tdistro: %d - total: %d - split_index: %d' % args)
        train_list = gid_list_[split_index:]
        test_list = gid_list_[:split_index]
        args = (len(test_list), len(train_list), len(train_list) / total, )
        print('\ttest: %d\n\ttrain: %d\n\tsplit: %0.04f' % args)
        global_train_list.extend(train_list)
        global_test_list.extend(test_list)

    args = (
        len(global_train_list),
        len(global_test_list),
        len(global_train_list) + len(global_test_list),
        len(global_train_list) / len(gid_list),
        train_split,
    )

    train_imgsetid = ibs.add_imagesets('TRAIN_SET')
    test_imgsetid = ibs.add_imagesets('TEST_SET')

    temp_list = ibs.get_imageset_gids(train_imgsetid)
    ibs.unrelate_images_and_imagesets(temp_list, [train_imgsetid] * len(temp_list))
    temp_list = ibs.get_imageset_gids(test_imgsetid)
    ibs.unrelate_images_and_imagesets(temp_list, [test_imgsetid] * len(temp_list))

    ibs.set_image_imgsetids(global_train_list, [train_imgsetid] * len(global_train_list))
    ibs.set_image_imgsetids(global_test_list, [test_imgsetid] * len(global_test_list))

    print('Complete... %d train + %d test = %d [%0.04f, %0.04f]' % args)


@register_ibs_method
def detect_distributions(ibs):
    # Process distributions of densities
    gid_list = ibs.get_valid_gids()
    aids_list = ibs.get_image_aids(gid_list)
    distro_dict = {}
    species_dict = {}
    for gid, aid_list in zip(gid_list, aids_list):
        total = len(aid_list)
        if total >= 7:
            total = 7
        if total not in distro_dict:
            distro_dict[total] = 0
        distro_dict[total] += 1
        for aid in aid_list:
            species = ibs.get_annot_species_texts(aid)
            if species not in ['zebra_plains', 'zebra_grevys']:
                species = 'unspecified'
            viewpoint = ibs.get_annot_yaw_texts(aid)
            if species not in species_dict:
                species_dict[species] = {}
            if viewpoint not in species_dict[species]:
                species_dict[species][viewpoint] = 0
            species_dict[species][viewpoint] += 1

    print('Density distribution')
    for distro in sorted(distro_dict.keys()):
        print('%d,%d' % (distro, distro_dict[distro]))
    print('\n')

    for species in sorted(species_dict.keys()):
        print('Species viewpoint distribution: %r' % (species, ))
        viewpoint_dict = species_dict[species]
        total = 0
        for viewpoint in sorted(viewpoint_dict.keys()):
            print('%s: %d' % (viewpoint, viewpoint_dict[viewpoint]))
            total += viewpoint_dict[viewpoint]
        print('Total: %d\n' % (total, ))

    plot_distrobutions(distro_dict)


def plot_distrobutions(distro_dict):
    import matplotlib.pyplot as plt
    key_list = sorted(distro_dict.keys())
    label_list = [ '7+' if key == 7 else str(key) for key in key_list ]
    size_list = [ distro_dict[key] for key in key_list ]
    color_list = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = [0.0] + [0.0] * (len(size_list) - 1)

    plt.pie(size_list, explode=explode, labels=label_list, colors=color_list,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.show()


def detect_intersection_over_union(bbox1, bbox2):
    x_overlap = max(0.0,
                    min(bbox1['xtl'] + bbox1['width'], bbox2['xtl'] + bbox2['width']) -
                    max(bbox1['xtl'], bbox2['xtl']))
    y_overlap = max(0.0, min(bbox1['ytl'] + bbox1['height'], bbox2['ytl'] + bbox2['height']) -
                    max(bbox1['ytl'], bbox2['ytl']))
    intersection = float(x_overlap * y_overlap)
    union = (bbox1['width'] * bbox1['height']) + (bbox2['width'] * bbox2['height']) - intersection
    return intersection / union


def detect_overlap(gt_list, pred_list):
    overlap = np.zeros((len(gt_list), len(pred_list)), dtype=np.float32)
    for i, gt in enumerate(gt_list):
        for j, pred in enumerate(pred_list):
            overlap[i, j] = detect_intersection_over_union(gt, pred)
    return overlap


def detect_tp_fp_fn(gt_list, pred_list, min_overlap, **kwargs):
    overlap = detect_overlap(gt_list, pred_list)
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
        index_list = np.argmax(overlap, axis=1)
        max_overlap = np.max(overlap, axis=1)
        max_overlap[max_overlap < min_overlap] = 0.0
        assignment_dict = {
            i : index_list[i] for i in range(num_gt) if max_overlap[i] != 0
        }
        tp = len(assignment_dict.keys())
        # Fix where multiple GT claim the same prediction
        if tp > num_pred:
            index_list_ = np.argmax(overlap, axis=0)
            key_list = sorted(assignment_dict.keys())
            for key in key_list:
                if key not in index_list_:
                    del assignment_dict[key]
            tp = len(assignment_dict.keys())
        fp = num_pred - tp
        fn = num_gt - tp
    return tp, fp, fn


def detect_parse_gt(ibs, test_gid_set=None):
    if test_gid_set is None:
        test_gid_set = ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    uuid_list = ibs.get_image_uuids(test_gid_set)
    gid_list = ibs.get_image_gids_from_uuid(uuid_list)

    gt_dict = {}
    for gid, uuid in zip(gid_list, uuid_list):
        width, height = ibs.get_image_sizes(gid)
        aid_list = ibs.get_image_aids(gid)
        temp_list = []
        for aid in aid_list:
            bbox = ibs.get_annot_bboxes(aid)
            temp = {
                'xtl'        : bbox[0] / width,
                'ytl'        : bbox[1] / height,
                'width'      : bbox[2] / width,
                'height'     : bbox[3] / height,
                'species'    : ibs.get_annot_species_texts(aid),
                'viewpoint'  : ibs.get_annot_yaw_texts(aid),
                'confidence' : 1.0,
            }
            temp_list.append(temp)
        gt_dict[uuid] = temp_list
    return gt_dict


def detect_parse_pred(ibs, test_gid_set=None, **kwargs):
    depc = ibs.depc_image

    if test_gid_set is None:
        test_gid_set = ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    uuid_list = ibs.get_image_uuids(test_gid_set)

    config = {
        'algo'            : 'yolo',
        'sensitivity'     : 0.0,
        'config_filepath' : None,
        'weight_filepath' : None,
        'grid'            : False,
    }
    config = ut.update_existing(config, kwargs)
    if config.get('algo', None) == 'rf':
        species = kwargs.get('species', None)
        if species is not None:
            config['species'] = species
    results_list = depc.get_property('detections', test_gid_set, None, config=config)
    size_list = ibs.get_image_sizes(test_gid_set)
    zipped_list = zip(results_list)
    # Reformat results for json
    results_list = [
        [
            {
                'xtl'        : bbox[0] / width,
                'ytl'        : bbox[1] / height,
                'width'      : bbox[2] / width,
                'height'     : bbox[3] / height,
                'theta'      : round(theta, 4),
                'confidence' : round(conf, 4),
                # 'class'      : class_,
                'species'    : class_,
            }
            for bbox, theta, conf, class_ in zip(*zipped[0][1:])
        ]
        for zipped, (width, height) in zip(zipped_list, size_list)
    ]

    # pred_dict_dict = {}
    # for gid, uuid_, result_list in zip(test_gid_set, uuid_list, results_list):
    #     # Get detections from depc
    #     pred_dict = {}
    #     for current_index in range(0, 10001):
    #         conf = current_index / 10000.0
    #         temp_list = []
    #         for result in result_list:
    #             confidence = result['confidence']
    #             if confidence < conf:
    #                 continue
    #             temp_list.append(result)
    #         alpha = 10000 - current_index  # Invert sensitivity for YOLO
    #         pred_dict[alpha] = temp_list
    #     pred_dict_dict[uuid_] = pred_dict
    # return pred_dict_dict

    pred_dict = {
        uuid_ : result_list
        for uuid_, result_list in zip(uuid_list, results_list)
    }
    return pred_dict


def detect_precision_recall_algo(ibs, samples=1000, **kwargs):
    # test_gid_set = ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    test_gid_set = ibs.get_valid_gids()
    uuid_list = ibs.get_image_uuids(test_gid_set)

    print('\tGather Ground-Truth')
    gt_dict = detect_parse_gt(ibs, test_gid_set=test_gid_set)

    print('\tGather Predictions')
    pred_dict = detect_parse_pred(ibs, test_gid_set=test_gid_set, **kwargs)

    print('\tGenerate Curves...')
    conf_list = [ _ / float(samples) for _ in range(0, int(samples) + 1) ]

    uuid_list_list = [ uuid_list for _ in conf_list ]
    gt_dict_list   = [ gt_dict   for _ in conf_list ]
    pred_dict_list = [ pred_dict for _ in conf_list ]
    kwargs_list    = [ kwargs    for _ in conf_list ]
    arg_iter = zip(conf_list, uuid_list_list, gt_dict_list, pred_dict_list, kwargs_list)
    arg_list = list(arg_iter)
    pr_re_gen = ut.generate(detect_precision_recall_algo_worker, arg_list,
                            nTasks=len(arg_list), ordered=True, verbose=False,
                            quiet=True, chunksize=64, force_serial=False)

    conf_list_, pr_list, re_list = [-1.0], [1.0], [0.0]
    for conf, pr, re in pr_re_gen:
        conf_list_.append(conf)
        pr_list.append(pr)
        re_list.append(re)

    print('...complete')
    return conf_list_, pr_list, re_list


def detect_precision_recall_algo_worker(tup):
    conf, uuid_list, gt_dict, pred_dict, kwargs = tup
    tp, fp, fn = 0.0, 0.0, 0.0
    for index, uuid_ in enumerate(uuid_list):
        if uuid_ in pred_dict:
            temp_list = [
                pred
                for pred in pred_dict[uuid_]
                if pred['confidence'] >= conf
            ]
            tp_, fp_, fn_ = detect_tp_fp_fn(gt_dict[uuid_], temp_list, **kwargs)
            tp += tp_
            fp += fp_
            fn += fn_
    pr = tp / (tp + fp)
    re = tp / (tp + fn)
    return (conf, pr, re)


def detect_precision_recall_algo_plot(ibs, label, color, **kwargs):
    import matplotlib.pyplot as plt
    print('Processing Precision-Recall for: %r' % (label, ))
    conf_list, pr_list, re_list = detect_precision_recall_algo(ibs, **kwargs)
    best_conf_list, best_x_list, best_y_list = identify_operating_point(conf_list, re_list, pr_list)
    best_conf_list = ','.join([ '%0.02f' % (conf, ) for conf in best_conf_list ])
    label = '%s [OP = %s]' % (label, best_conf_list, )
    plt.plot(re_list, pr_list, '%s-' % (color, ), label=label)
    plt.plot(best_x_list, best_y_list, '%so' % (color, ))
    area = np.trapz(pr_list, x=re_list)
    return area


@register_ibs_method
def detect_precision_recall_algo_display(ibs, min_overlap=0.5, figsize=(10, 9), **kwargs):
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    axes_ = plt.subplot(111)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall (Ground-truth IOU >= %0.02f)' % (min_overlap, ))
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])

    name_list = [
        'Original',
        'Retrained',
        'Original (GRID)',
        'Retrained (GRID)',
        # 'Hough Forests',
    ]
    area_list = []
    area_list.append(detect_precision_recall_algo_plot(ibs, name_list[0], 'r', min_overlap=min_overlap, grid=False, config_filepath='v1', weight_filepath='v1'))
    area_list.append(detect_precision_recall_algo_plot(ibs, name_list[1], 'b', min_overlap=min_overlap, grid=False, config_filepath='v2', weight_filepath='v2'))
    area_list.append(detect_precision_recall_algo_plot(ibs, name_list[2], 'k', min_overlap=min_overlap, grid=True, config_filepath='v1', weight_filepath='v1'))
    area_list.append(detect_precision_recall_algo_plot(ibs, name_list[3], 'g', min_overlap=min_overlap, grid=True, config_filepath='v2', weight_filepath='v2'))
    # area_list.append(detect_precision_recall_algo_plot(ibs, name_list[4], 'p', min_overlap=min_overlap, algo='pyrf', species='zebra_grevys'))

    index = np.argmax(area_list)
    best_name = name_list[index]
    best_area = area_list[index]
    plt.title('Precision-Recall Curve (Best: %s, mAP = %0.02f )' % (best_name, best_area, ), y=1.13)

    # Display graph
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)
    # plt.show()
    fig_filename = 'detection-precision-recall-%0.2f.png' % (min_overlap, )
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


@register_ibs_method
def classifier_precision_recall_algo(ibs, samples=10000, **kwargs):
    def errors(zipped, conf):
        error_list = [0, 0, 0, 0]
        for index, (label, confidence) in enumerate(zipped):
            if label == 'positive' and conf <= confidence:
                error_list[0] += 1
            elif label == 'negative' and conf <= confidence:
                error_list[2] += 1
            elif label == 'positive':
                error_list[3] += 1
            elif label == 'negative':
                error_list[1] += 1
        return error_list

    depc = ibs.depc_image
    category_set = set(['zebra_plains', 'zebra_grevys'])
    gid_list = ibs.get_valid_gids()
    aids_list = ibs.get_image_aids(gid_list)
    species_set_list = [
        set(ibs.get_annot_species_texts(aid_list))
        for aid_list in aids_list
    ]
    label_list = [
        'negative' if len(species_set & category_set) == 0 else 'positive'
        for species_set in species_set_list
    ]
    prediction_list = depc.get_property('classifier', gid_list, 'class')
    confidence_list = depc.get_property('classifier', gid_list, 'score')
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence  in zip(prediction_list, confidence_list)
    ]

    zipped = list(zip(label_list, confidence_list))
    conf_list = [ _ / float(samples) for _ in range(0, int(samples) + 1) ]
    conf_dict = {}
    for conf in conf_list:
        conf_dict[conf] = errors(zipped, conf)

    conf_list = [-1.0]
    pr_list = [1.0]
    re_list = [0.0]
    tpr_list = [0.0]
    fpr_list = [0.0]
    for conf in sorted(conf_dict.keys(), reverse=True):
        error_list = conf_dict[conf]
        tp, tn, fp, fn = error_list
        pr = tp / (tp + fp)
        re = tp / (tp + fn)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        conf_list.append(conf)
        pr_list.append(pr)
        re_list.append(re)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return conf_list, pr_list, re_list, tpr_list, fpr_list


# @register_ibs_method
# def classifier_precision_recall_algo2(ibs, **kwargs):
#     def errors(zipped, conf):
#         error_list = [0, 0, 0, 0]
#         for index, (label, confidence) in enumerate(zipped):
#             if label == 'positive' and conf <= confidence:
#                 error_list[0] += 1
#             elif label == 'negative' and conf <= confidence:
#                 error_list[2] += 1
#             elif label == 'positive':
#                 error_list[3] += 1
#             elif label == 'negative':
#                 error_list[1] += 1
#         return error_list

#     depc = ibs.depc_image
#     category_set = set(['zebra_plains', 'zebra_grevys'])
#     gid_list = ibs.get_valid_gids()
#     aids_list = ibs.get_image_aids(gid_list)
#     species_set_list = [
#         set(ibs.get_annot_species_texts(aid_list))
#         for aid_list in aids_list
#     ]
#     label_list = [
#         'negative' if len(species_set & category_set) == 0 else 'positive'
#         for species_set in species_set_list
#     ]
#     prediction_list = depc.get_property('classifier', gid_list, 'class')
#     confidence_list = depc.get_property('classifier', gid_list, 'score')
#     confidence_list = [
#         confidence if prediction == 'positive' else 1.0 - confidence
#         for prediction, confidence  in zip(prediction_list, confidence_list)
#     ]

#     from sklearn.metrics import precision_recall_curve
#     y_true = np.array([
#         0 if label == 'negative' else 1
#         for label in label_list
#     ])
#     y_scores = np.array(confidence_list)
#     pr_list, re_list, thresholds = precision_recall_curve(y_true, y_scores)
#     return pr_list, re_list


def identify_operating_point(conf_list, x_list, y_list, invert=False):
    best_length = np.inf
    best_conf_list = []
    best_x_list = []
    best_y_list = []
    for conf, x, y in zip(conf_list, x_list, y_list):
        x_ = 1.0 - x if not invert else x
        y_ = 1.0 - y
        length = np.sqrt(x_ * x_ + y_ * y_)
        if length < best_length:
            best_length = length
            best_conf_list = [conf]
            best_x_list = [x]
            best_y_list = [y]
        elif length == best_length:
            flag_list = [ abs(best_conf - conf) > 0.01 for best_conf in best_conf_list ]
            if False in flag_list:
                continue
            best_conf_list.append(conf)
            best_x_list.append(x)
            best_y_list.append(y)

    return best_conf_list, best_x_list, best_y_list


def classifier_precision_recall_algo_plot(ibs, label, color, **kwargs):
    import matplotlib.pyplot as plt
    print('Processing Precision-Recall for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier_precision_recall_algo(ibs, **kwargs)
    best_conf_list, best_x_list, best_y_list = identify_operating_point(conf_list, re_list, pr_list)
    best_conf_list = ','.join([ '%0.02f' % (conf, ) for conf in best_conf_list ])
    label = '%s [%s]' % (label, best_conf_list, )
    plt.plot(re_list, pr_list, '%s-' % (color, ), label=label)
    plt.plot(best_x_list, best_y_list, '%so' % (color, ))
    area = np.trapz(pr_list, x=re_list)
    return area


# def classifier_precision_recall_algo_plot2(ibs, label, color, **kwargs):
#     import matplotlib.pyplot as plt
#     print('Processing Precision-Recall for: %r' % (label, ))
#     pr_list, re_list, = classifier_precision_recall_algo2(ibs, **kwargs)
#     plt.plot(re_list, pr_list, '%s-' % (color, ), label=label)


def classifier_roc_algo_plot(ibs, label, color, **kwargs):
    import matplotlib.pyplot as plt
    print('Processing Precision-Recall for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier_precision_recall_algo(ibs, **kwargs)
    best_conf_list, best_x_list, best_y_list = identify_operating_point(conf_list, fpr_list, tpr_list, invert=True)
    best_conf_list = ','.join([ '%0.02f' % (conf, ) for conf in best_conf_list ])
    label = '%s [%s]' % (label, best_conf_list, )
    plt.plot(fpr_list, tpr_list, '%s-' % (color, ), label=label)
    plt.plot(best_x_list, best_y_list, '%so' % (color, ))
    area = np.trapz(tpr_list, x=fpr_list)
    return area


def classifier_confusion_matrix_algo(label_correct_list, label_predict_list,
                                     category_list, category_mapping,
                                     fig_, axes_):
    # import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    # Get the number of categories
    num_categories = len(category_list)

    # Build the confusion matrix
    confusion_matrix = np.zeros((num_categories, num_categories))
    zipped = zip(label_correct_list, label_predict_list)
    for label_correct, label_predict, in zipped:
        # Perform any mapping that needs to be done
        correct_ = category_mapping[label_correct]
        predict_ = category_mapping[label_predict]
        # Add to the confidence matrix
        confusion_matrix[correct_][predict_] += 1

    # Normalize the confusion matrix using the rows
    row_normalizer = np.sum(confusion_matrix, axis=1)
    confusion_normalized = np.array((confusion_matrix.T / row_normalizer).T)

    # Draw the confusion matrix
    res = axes_.imshow(confusion_normalized, cmap=plt.cm.jet,
                       interpolation='nearest')

    for x in range(num_categories):
        for y in range(num_categories):
            axes_.annotate(
                str(int(confusion_matrix[x][y])), xy=(y, x),
                horizontalalignment='center',
                verticalalignment='center'
            )

    cb = fig_.colorbar(res)  # NOQA
    cb.set_clim(0.0, 1.0)
    plt.xticks(np.arange(num_categories), category_list, rotation=90)
    plt.yticks(np.arange(num_categories), category_list)


def classifier_confusion_matrix_algo_plot(ibs, label, color, **kwargs):
    print('Processing Precision-Recall for: %r' % (label, ))
    depc = ibs.depc_image
    category_set = set(['zebra_plains', 'zebra_grevys'])
    gid_list = ibs.get_valid_gids()
    aids_list = ibs.get_image_aids(gid_list)
    species_set_list = [
        set(ibs.get_annot_species_texts(aid_list))
        for aid_list in aids_list
    ]
    label_list = [
        'negative' if len(species_set & category_set) == 0 else 'positive'
        for species_set in species_set_list
    ]
    prediction_list = depc.get_property('classifier', gid_list, 'class')
    category_list = ['positive', 'negative']
    category_mapping = {
        'positive': 0,
        'negative': 1,
    }
    classifier_confusion_matrix_algo(label_list, prediction_list, category_list,
                                     category_mapping, **kwargs)


@register_ibs_method
def classifier_precision_recall_algo_display(ibs, figsize=(21, 6), **kwargs):
    import matplotlib.pyplot as plt

    fig_ = plt.figure(figsize=figsize)

    axes_ = plt.subplot(131)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall')
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    area = classifier_precision_recall_algo_plot(ibs, 'V1', 'r')
    plt.title('Precision-Recall Curve (mAP = %0.02f)' % (area, ), y=1.10)
    # classifier_precision_recall_algo_plot2(ibs, 'V1', 'g')
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(132)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    area = classifier_roc_algo_plot(ibs, 'V1', 'r')
    plt.title('ROC Curve (mAP = %0.02f)' % (area, ), y=1.10)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(133)
    plt.title('Confusion Matrix', y=1.15)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Ground-Truth')
    axes_.set_ylabel('Predictions')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    classifier_confusion_matrix_algo_plot(ibs, 'V1', 'r', fig_=fig_, axes_=axes_)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    fig_filename = 'classifier-precision-recall-roc.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


@register_ibs_method
def detect_precision_recall_algo_display_animate(ibs, **kwargs):
    for value in range(10):
        min_overlap = value / 10.0
        print('Processing: %r' % (min_overlap, ))
        ibs.detect_precision_recall_algo_display(min_overlap=min_overlap)


def _resize(image, t_width=None, t_height=None):
    print('RESIZING WITH t_width = %r and t_height = %r' % (t_width, t_height, ))
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
    assert t_width <= width * 10 and t_height <= height * 10, 'target size too large (capped at 1000%)'
    # interpolation = cv2.INTER_LANCZOS4
    interpolation = cv2.INTER_LINEAR
    return cv2.resize(image, (t_width, t_height), interpolation=interpolation)


@register_ibs_method
def detect_write_detection_all(ibs):
    test_gid_list = ibs.get_valid_gids()
    test_image_list = ibs.get_images(test_gid_list)
    test_uuid_list = ibs.get_image_uuids(test_gid_list)

    write_path = abspath(expanduser(join('~', 'Desktop')))
    # gt_dict = detect_parse_gt(ibs_, test_gid_set=test_gid_list)
    for index, (test_gid, test_uuid, test_image) in enumerate(zip(test_gid_list, test_uuid_list, test_image_list)):
        height_old, width_old, channels_old = test_image.shape
        test_image = _resize(test_image, t_width=600)
        height_, width_, channels_ = test_image.shape
        rescale = width_ / width_old
        aid_list = ibs.get_image_aids(test_gid)
        annot_list = ibs.get_annot_bboxes(aid_list)
        for xtl, ytl, width, height in annot_list:
            xbr = int((xtl + width) * rescale)
            ybr = int((ytl + height) * rescale)
            xtl = int(xtl * rescale)
            ytl = int(ytl * rescale)
            cv2.rectangle(test_image, (xtl, ytl), (xbr, ybr), (0, 140, 255), 4)
        write_filepath = join(write_path, '%d.jpg' % (index, ))
        print(write_filepath)
        cv2.imwrite(write_filepath, test_image)


@register_ibs_method
def redownload_detection_models(ibs):
    r"""
    Args:
        ibs (IBEISController):

    CommandLine:
        python -c "from ibeis.algo.detect import grabmodels; grabmodels.redownload_models()"
        python -c "import utool, ibeis.algo; utool.view_directory(ibeis.algo.detect.grabmodels._expand_modeldir())"

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.detectfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> result = redownload_detection_models(ibs)
        >>> print(result)
    """
    print('[other.detectfuncs] redownload_detection_models')
    from ibeis.algo.detect import grabmodels
    modeldir = ibs.get_detect_modeldir()
    grabmodels.redownload_models(modeldir=modeldir)


@register_ibs_method
def view_model_dir(ibs):
    print('[other.detectfuncs] redownload_detection_models')
    modeldir = ibs.get_detect_modeldir()
    ut.view_directory(modeldir)
    #grabmodels.redownload_models(modeldir=modeldir)


def remove_rfdetect(ibs):
    aids = ibs.search_annot_notes('rfdetect')
    notes = ibs.get_annot_notes(aids)
    newnotes = [note.replace('rfdetect', '') for note in notes]
    ibs.set_annot_notes(aids, newnotes)


@register_ibs_method
def fix_horizontal_bounding_boxes_to_orient(ibs, gid, bbox_list):
    orient = ibs.get_image_orientation(gid)
    bbox_list_ = []
    for bbox in bbox_list:
        (xtl, ytl, width, height) = bbox
        if orient == 6:
            full_w, full_h = ibs.get_image_sizes(gid)
            xtl, ytl = full_w - ytl - height, xtl
            width, height = height, width
        elif orient == 8:
            full_w, full_h = ibs.get_image_sizes(gid)
            xtl, ytl = ytl, full_h - xtl - width
            width, height = height, width
        else:
            pass
        bbox_ = (xtl, ytl, width, height)
        bbox_list_.append(bbox_)
    return bbox_list_


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.other.detectfuncs
        python -m ibeis.other.detectfuncs --allexamples
        python -m ibeis.other.detectfuncs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
