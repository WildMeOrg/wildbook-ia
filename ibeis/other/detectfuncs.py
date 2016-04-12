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
from os.path import exists
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


# def detect_parse_sweep(sweep_filepath):
#     from os.path import abspath, expanduser, join, exists
#     with open(sweep_filepath, 'r') as sweep_file:
#         line_list = sweep_file.readlines()
#     sweep_dict = {}
#     temp_list = []
#     current_gpath = None
#     current_index = None
#     current_width, current_height = None, None
#     # print('Parsing: %r' % (sweep_filepath, ))
#     for line in line_list:
#         if not line.startswith(' '):
#             if current_index is not None:
#                 sweep_dict[current_index] = temp_list
#                 temp_list = []
#             line = line.strip().split()
#             assert len(line) == 2
#             current_gpath = line[0]
#             if exists(current_gpath):
#                 current_width, current_height = vt.open_image_size(current_gpath)
#             else:
#                 current_gpath = current_gpath.replace(
#                     '/media/extend/jason/data',
#                     abspath(expanduser(join('~', 'Desktop', 'data')))
#                 )
#                 current_width, current_height = vt.open_image_size(current_gpath)
#             current_index = int(line[1])
#         else:
#             assert current_gpath is not None and current_index is not None
#             assert current_width is not None and current_height is not None
#             line = line.strip().split()
#             assert len(line) == 8
#             temp = {
#                 'xtl'        : float(line[2]) / current_width,
#                 'ytl'        : float(line[3]) / current_height,
#                 'width'      : float(line[4]) / current_width,
#                 'height'     : float(line[5]) / current_height,
#                 'species'    : None if line[6] == '0' else line[6],
#                 'confidence' : float(line[7]),
#             }
#             temp_list.append(temp)
#     sweep_dict[current_index] = temp_list
#     if 100 not in sweep_dict:
#         raise ValueError('Error parsing (incomplete): %r' % (sweep_filepath, ))
#     return sweep_dict


# def detect_precision_recall_algo(ibs, algo, **kwargs):
#     import uuid
#     from os.path import abspath, expanduser, join, split
#     test_path = abspath(expanduser(join('~', 'Desktop', 'results', algo)))
#     sweep_filepath_list = ut.ls(test_path, '*_sweep.txt')

#     test_gid_set = ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
#     uuid_list = ibs.get_image_uuids(test_gid_set)

#     sweep_dict_dict = {}
#     processed, skipped, errored = 0, 0, 0
#     for sweep_filepath in sweep_filepath_list:
#         sweep_path, sweep_filename = split(sweep_filepath)
#         uuid_ = uuid.UUID(sweep_filename.split('.JPG')[0])
#         if uuid_ not in uuid_list:
#             # print('Skipping: image UUID not in this dataset')
#             skipped += 1
#             continue
#         try:
#             sweep_dict = detect_parse_sweep(sweep_filepath)
#             sweep_dict_dict[uuid_] = sweep_dict
#             processed += 1
#         except ValueError:
#             # print('Skipping: Incomplete sweep file')
#             errored += 1
#             continue
#         except AssertionError:
#             # print('Skipping: Corrupt sweep file')
#             errored += 1
#             continue
#         # print('Parsed: %r' % (sweep_filepath, ))
#         # print(sweep_dict)
#     print('Processed: %d' % (processed, ))
#     print('Skipped: %d' % (skipped, ))
#     print('Errored: %d' % (errored, ))
#     assert errored == 0
#     return sweep_dict_dict


def detect_parse_pred(ibs, test_gid_set=None, **kwargs):
    depc = ibs.depc_image

    if test_gid_set is None:
        test_gid_set = ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    uuid_list = ibs.get_image_uuids(test_gid_set)

    config = {
        'algo'            : 'yolo',
        'sensitivity'     : 0.0,
        'config_filepath' : '/media/hdd/jason/yolo/weights/detect.yolo.12.cfg',
        'weight_filepath' : '/media/hdd/jason/yolo/weights/detect.yolo.12.weights',
        'grid'            : False,
    }
    config = ut.update_existing(config, kwargs)
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
    result_list = results_list

    pred_dict_dict = {}
    for gid, uuid_, result_list in zip(test_gid_set, uuid_list, results_list):
        # Get detections from depc
        pred_dict = {}
        for current_index in range(0, 101):
            conf = current_index / 100.0
            temp_list = []
            for result in result_list:
                confidence = result['confidence']
                if confidence < conf:
                    continue
                temp_list.append(result)
            alpha = 100 - current_index  # Invert sensitivity for YOLO
            pred_dict[alpha] = temp_list
        pred_dict_dict[uuid_] = pred_dict
    return pred_dict_dict


def detect_precision_recall_algo(ibs, **kwargs):
    # test_gid_set = ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET'))
    test_gid_set = ibs.get_valid_gids()
    uuid_list = ibs.get_image_uuids(test_gid_set)

    print('\tGather Ground-Truth')
    gt_dict = detect_parse_gt(ibs, test_gid_set=test_gid_set)

    print('\tGather Predictions')
    pred_dict_dict = detect_parse_pred(ibs, test_gid_set=test_gid_set, **kwargs)

    print('\tGenerate Curves...')
    global_dict = {
        'tp' : {},
        'fp' : {},
        'fn' : {},
    }
    for gid, uuid in zip(test_gid_set, uuid_list):
        gt_list = gt_dict[uuid]
        if uuid in pred_dict_dict:
            pred_dict = pred_dict_dict[uuid]
            for alpha in pred_dict:
                for key in global_dict.keys():
                    if alpha not in global_dict[key]:
                        global_dict[key][alpha] = 0.0
                pred_list = pred_dict[alpha]
                tp, fp, fn = detect_tp_fp_fn(gt_list, pred_list, **kwargs)
                global_dict['tp'][alpha] += tp
                global_dict['fp'][alpha] += fp
                global_dict['fn'][alpha] += fn
    print('...complete')

    pr_list, re_list = [], []
    alpha_list = sorted(global_dict['tp'].keys())
    for alpha in alpha_list:
        tp = global_dict['tp'][alpha]
        fp = global_dict['fp'][alpha]
        fn = global_dict['fn'][alpha]
        pr = tp / (tp + fp)
        re = tp / (tp + fn)
        pr_list.append(pr)
        re_list.append(re)

    return pr_list, re_list


def detect_precision_recall_algo_plot(ibs, label, color, **kwargs):
    import matplotlib.pyplot as plt
    print('Processing Precision-Recall for: %r' % (label, ))
    pr_list, re_list = detect_precision_recall_algo(ibs, **kwargs)
    plt.plot(re_list, pr_list, '%s-' % (color, ), label=label)


@register_ibs_method
def detect_precision_recall_algo_display(ibs, min_overlap=0.7, figsize=(10, 6), **kwargs):
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    axes_ = plt.subplot(111)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall (Ground-truth IOU >= %0.02f)' % (min_overlap, ))
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])

    detect_precision_recall_algo_plot(ibs, 'Original', 'r', min_overlap=min_overlap, grid=False, config_filepath='v1', weight_filepath='v1')
    detect_precision_recall_algo_plot(ibs, 'Original (GRID)', 'k', min_overlap=min_overlap, grid=True, config_filepath='v1', weight_filepath='v1')
    detect_precision_recall_algo_plot(ibs, 'Retrained', 'b', min_overlap=min_overlap, grid=False)
    detect_precision_recall_algo_plot(ibs, 'Retrained (GRID)', 'g', min_overlap=min_overlap, grid=True)

    # Display graph
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)
    # plt.show()
    fig_path = '/Users/bluemellophone/Desktop/precision-recall-%0.2f.png' % (min_overlap, )
    plt.savefig(fig_path, bbox_inches='tight')


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


# @register_ibs_method
# def detect_write_detection_exmaples(ibs, SEED=23170):
#     from os.path import abspath, expanduser, join
#     import ibeis
#     import random

#     operating_dict = {
#         'rcnn': 80,
#         'rf' : 60,
#         'yolo': 80,
#     }

#     random.seed(SEED)
#     for database, tag in [ ('PZ_Paper', 'pz'), ('GZ_Paper', 'gz') ]:
#         ibs_ = ibeis.opendb(dbdir=abspath(expanduser(join('~', 'Desktop', 'data', database))))
#         # Get random set of test images
#         test_gid_set = ibs_.get_imageset_gids(ibs_.get_imageset_imgsetids_from_text('TEST_SET'))
#         random.shuffle(test_gid_set)
#         test_gid_list = test_gid_set[:10]

#         if tag == 'gz':
#             test_gid_list[0] = test_gid_set[10]

#         test_image_list = ibs_.get_images(test_gid_list)
#         test_uuid_list = ibs_.get_image_uuids(test_gid_list)

#         write_path = abspath('/Users/bluemellophone/Dropbox/Shared/Chuck WACV/resources/detections/')
#         print(write_path)
#         # gt_dict = detect_parse_gt(ibs_, test_gid_set=test_gid_list)
#         for algo, conf in operating_dict.iteritems():
#             test_path = abspath(expanduser(join('~', 'Desktop', 'results', algo,)))
#             for index, (test_uuid, test_image) in enumerate(zip(test_uuid_list, test_image_list)):
#                 test_image = _resize(test_image, t_width=600)
#                 height, width, channels = test_image.shape
#                 sweep_filepath = join(test_path, '%s.JPG_sweep.txt' % (test_uuid, ))
#                 sweep_dict = detect_parse_sweep(sweep_filepath)
#                 annot_list = sweep_dict[conf]
#                 for annot in annot_list:
#                     xtl = int(annot['xtl'] * width)
#                     ytl = int(annot['ytl'] * height)
#                     xbr = int((annot['xtl'] + annot['width']) * width)
#                     ybr = int((annot['ytl'] + annot['height']) * height)
#                     cv2.rectangle(test_image, (xtl, ytl), (xbr, ybr), (0, 140, 255), 4)
#                 write_filepath = join(write_path, algo, '%s%d.jpg' % (tag, index, ))
#                 print(write_filepath)
#                 cv2.imwrite(write_filepath, test_image)


@register_ibs_method
def detect_write_detection_all(ibs):
    from os.path import abspath, join

    test_gid_list = ibs.get_valid_gids()
    test_image_list = ibs.get_images(test_gid_list)
    test_uuid_list = ibs.get_image_uuids(test_gid_list)

    write_path = abspath('/Users/bluemellophone/Desktop/')
    print(write_path)
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
