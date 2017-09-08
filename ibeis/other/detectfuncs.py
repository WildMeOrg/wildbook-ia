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
import ibeis.constants as const
from ibeis.control import controller_inject
from ibeis import annotmatch_funcs  # NOQA

# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[other.detectfuncs]')


SAMPLES = 1000
CHUNK_SIZE = SAMPLES // ut.num_cpus()
FORCE_SERIAL = True

AP_SAMPLE_POINTS = [_ / 100.0 for _ in range(0, 101)]
# AP_SAMPLE_POINTS = AP_SAMPLE_POINTS[1:-1]
# print('USING AP_SAMPLE_POINTS = %r' % (AP_SAMPLE_POINTS, ))

# Must import class before injection
CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))


def simple_code(label):
    if label == 'ignore':
        return 'IGNORE'
    label = label.replace('lion',                'LN')
    label = label.replace('zebra_plains',        'PZ')
    label = label.replace('hippopotamus',        'HIPPO')
    label = label.replace('antelope',            'ANTEL')
    label = label.replace('elephant_savannah',   'ELEPH')
    label = label.replace('person',              'PERSON')
    # label = label.replace('giraffe_reticulated', 'GIR')
    label = label.replace('giraffe_reticulated', 'RG')
    label = label.replace('zebra_grevys',        'GZ')
    # label = label.replace('giraffe_masai',       'GIRM')
    label = label.replace('giraffe_masai',       'MG')
    label = label.replace('unspecified_animal',  'UNSPEC')
    label = label.replace('car',                 'CAR')
    label = label.replace('bird',                'B')
    label = label.replace('whale_shark',         'WS')
    label = label.replace('whale_fluke',         'WF')
    label = label.replace('lionfish',            'LF')
    label = label.replace('turtle_sea',          'ST')
    label = label.replace('dog_wild',            'WD')
    label = label.replace('cow_domestic',        'DOMW')
    label = label.replace('sheep_domestic',      'DOMS')
    label = label.replace('dog_domestic',        'DOMD')
    label = label.replace('bicycle',             'CYCLE')
    label = label.replace('motorcycle',          'MCYCLE')
    label = label.replace('bus',                 'BUS')
    label = label.replace('truck',               'TRUCK')
    label = label.replace('horse_domestic',      'DOMH')
    label = label.replace('boat',                'BOAT')
    label = label.replace('train',               'TRAIN')
    label = label.replace('cat_domestic',        'DOCM')
    label = label.replace('airplane',            'PLANE')

    label = label.replace('frontleft',           'FL')
    label = label.replace('frontright',          'FR')
    label = label.replace('backleft',            'BL')
    label = label.replace('backright',           'BR')
    label = label.replace('front',               'F')
    label = label.replace('back',                'B')
    label = label.replace('left',                'L')
    label = label.replace('right',               'R')
    label = label.replace('up',                  'U')
    label = label.replace('down',                'D')
    return label


@register_ibs_method
def export_to_xml(ibs, species_list=None, offset='auto', enforce_viewpoint=False,
                  target_size=900, purge=False, use_maximum_linear_dimension=True,
                  use_existing_train_test=True, **kwargs):
    """
    Creates training XML for training models
    """
    import random
    from datetime import date
    from detecttools.pypascalmarkup import PascalVOC_Markup_Annotation

    current_year = int(date.today().year)
    information = {
        'database_name' : ibs.get_dbname()
    }
    datadir = ibs.get_cachedir() + '/LearningData/'
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

    # Get all gids and process them
    gid_list = ibs.get_valid_gids()
    sets_dict = {
        'test'     : [],
        'train'    : [],
        'trainval' : [],
        'val'      : [],
    }
    index = 1 if offset == 'auto' else offset

    # Make a preliminary train / test split as imagesets or use the existing ones
    if not use_existing_train_test:
        ibs.imageset_train_test_split(**kwargs)

    train_gid_set = set(general_get_imageset_gids(ibs, 'TRAIN_SET', **kwargs))
    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET', **kwargs))

    print('Exporting %d images' % (len(gid_list),))
    for gid in gid_list:
        viewpointed = True
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

            _image = ibs.get_image_imgdata(gid)
            height, width, channels = _image.shape

            condition = width > height if use_maximum_linear_dimension else width < height
            if condition:
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
                species_name = ibs.get_annot_species_texts(aid)
                if species_list is not None:
                    if species_name not in species_list:
                        continue
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
                # Get info
                info = {}
                viewpoint = ibs.get_annot_viewpoints(aid)
                if viewpoint != -1 and viewpoint is not None:
                    info['pose'] = viewpoint
                else:
                    viewpointed = False
                    print("UNVIEWPOINTED: %d " % gid)
                annotation.add_object(
                    species_name,
                    (xmax, xmin, ymax, ymin),
                    **info
                )
            dst_annot = annotdir + out_name  + '.xml'

            if gid in test_gid_set:
                sets_dict['test'].append(out_name)
            elif gid in train_gid_set:
                state = random.uniform(0.0, 1.0)
                if state <= 0.75:
                    sets_dict['train'].append(out_name)
                    sets_dict['trainval'].append(out_name)
                else:
                    sets_dict['val'].append(out_name)
                    sets_dict['trainval'].append(out_name)
            else:
                raise AssertionError('All gids must be either in the TRAIN_SET or TEST_SET imagesets')

            # Write XML
            if True or not enforce_viewpoint or viewpointed:
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
def imageset_train_test_split(ibs, train_split=0.8, **kwargs):
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
    for distro, gid_list_ in distro_dict.items():
        total = len(gid_list_)
        shuffle(gid_list_)
        split_index = total * (1.0 - train_split) + 1E-9  # weird
        if split_index < 1.0:
            split_index = total / 2
        else:
            split_index = np.around(split_index)
        split_index = int(split_index)
        args = (distro, total, split_index, )
        print('\tnum aids distro: %d - total: %d - split_index: %d' % args)
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

    print('Complete... %d train + %d test = %d (%0.04f %0.04f)' % args)


@register_ibs_method
def localizer_distributions(ibs, threshold=10, dataset=None):
    # Process distributions of densities
    if dataset is None:
        gid_list = ibs.get_valid_gids()
    else:
        assert dataset in ['TRAIN_SET', 'TEST_SET']
        imageset_id = ibs.get_imageset_imgsetids_from_text(dataset)
        gid_list = list(set(ibs.get_imageset_gids(imageset_id)))
    aids_list = ibs.get_image_aids(gid_list)
    distro_dict = {}
    species_dict = {}
    for gid, aid_list in zip(gid_list, aids_list):
        total = len(aid_list)
        if total >= threshold:
            total = threshold
        if total not in distro_dict:
            distro_dict[total] = 0
        total = '%s' % (total, ) if total < threshold else '%s+' % (total, )
        distro_dict[total] += 1
        for aid in aid_list:
            species = ibs.get_annot_species_texts(aid)
            viewpoint = ibs.get_annot_viewpoints(aid)
            if species not in species_dict:
                species_dict[species] = {}
            if viewpoint not in species_dict[species]:
                species_dict[species][viewpoint] = 0
            species_dict[species][viewpoint] += 1

    print('Annotation density distribution (annotations per image)')
    for distro in sorted(distro_dict.keys()):
        print('{:>6} annot(s): {:>5}'.format(distro, distro_dict[distro]))
    print('')

    for species in sorted(species_dict.keys()):
        print('Species viewpoint distribution: %r' % (species, ))
        viewpoint_dict = species_dict[species]
        total = 0
        for viewpoint in const.VIEWTEXT_TO_VIEWPOINT_RADIANS:
            count = viewpoint_dict.get(viewpoint, 0)
            print('{:>15}: {:>5}'.format(viewpoint, count))
            total += count
        print('TOTAL: %d\n' % (total, ))

    # plot_distrobutions(distro_dict, threshold=threshold)


# def plot_distrobutions(distro_dict, threshold=10):
#     import matplotlib.pyplot as plt
#     key_list = sorted(distro_dict.keys())
#     threshold_str = '%d+' % (threshold, )
#     label_list = [
#         threshold_str if key == threshold else str(key)
#         for key in key_list
#     ]
#     size_list = [ distro_dict[key] for key in key_list ]
#     color_list = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
#     explode = [0.0] + [0.0] * (len(size_list) - 1)

#     plt.pie(size_list, explode=explode, labels=label_list, colors=color_list,
#             autopct='%1.1f%%', shadow=True, startangle=90)
#     plt.axis('equal')
#     plt.show()


def general_precision_recall_algo(ibs, label_list, confidence_list, category='positive', samples=SAMPLES, **kwargs):
    def errors(zipped, conf):
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
    conf_list = [ _ / float(samples) for _ in range(0, int(samples) + 1) ]
    conf_dict = {}
    for conf in conf_list:
        conf_dict[conf] = errors(zipped, conf)

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
            print('Zero division error (%r) - tp: %r tn: %r fp: %r fn: %r' % (conf, tp, tn, fp, fn, ))

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


def general_identify_operating_point(conf_list, x_list, y_list, x_norm=None,
                                     target=(1.0, 1.0)):
    best_length = np.inf
    best_conf_list = []
    best_x_list = []
    best_y_list = []
    tx, ty = target
    for conf, x, y in zip(conf_list, x_list, y_list):
        x_ = x
        y_ = y
        if x_norm is not None:
            x_ /= x_norm
        x_ = (x_ - tx)
        y_ = (y_ - ty)
        length = np.sqrt(x_ * x_ + y_ * y_)
        if length < best_length:
            best_length = length
            best_conf_list = [conf]
            best_x_list = [x]
            best_y_list = [y]
        elif length == best_length:
            flag_list = [
                abs(best_conf - conf) > 0.01
                for best_conf in best_conf_list
            ]
            if False in flag_list:
                continue
            best_conf_list.append(conf)
            best_x_list.append(x)
            best_y_list.append(y)

    return best_conf_list, best_x_list, best_y_list


def general_area_best_conf(conf_list, x_list, y_list, label='Unknown', color='b',
                           plot_point=True, interpolate=True, target=(1.0, 1.0),
                           version=1, **kwargs):
    import matplotlib.pyplot as plt
    zipped = list(sorted(zip(x_list, y_list, conf_list)))
    x_list = [_[0] for _ in zipped]
    y_list = [_[1] for _ in zipped]
    conf_list = [_[2] for _ in zipped]
    if interpolate:
        conf_list, x_list, y_list = general_interpolate_precision_recall(
            conf_list,
            x_list,
            y_list
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
        # y_list = y_list[::-1]
        # x_list = x_list[::-1]
        ap = np.trapz(y_list, x=x_list)
    tup = general_identify_operating_point(conf_list, x_list, y_list, target=target)
    best_conf_list, best_x_list, best_y_list = tup
    best_conf = best_conf_list[0] if len(best_conf_list) > 0 else np.nan
    # best_conf_list_ = ','.join([ '%0.02f' % (conf, ) for conf in best_conf_list ])
    # label = '%s [OP = %s]' % (label, best_conf_list_, )
    # label = '%s [OP = %0.02f]' % (label, best_conf, )
    if interpolate:
        label = '%s [AP = %0.02f]' % (label, ap * 100.0, )
    else:
        label = '%s [AUC = %0.02f]' % (label, ap * 100.0, )
    linestyle = '--' if kwargs.get('line_dotted', False) else '-'
    plt.plot(x_list, y_list, color=color, linestyle=linestyle, label=label)
    if plot_point:
        plt.plot(best_x_list, best_y_list, color=color, marker='o')
    if len(best_conf_list) > 1:
        print('WARNING: %r' % (best_conf_list, ))
    return ap, best_conf, tup


def general_confusion_matrix_algo(label_correct_list, label_predict_list,
                                  category_list, category_mapping,
                                  fig_, axes_, fuzzy_dict=None, conf=None,
                                  conf_list=None, size=10, **kwargs):
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
    res = axes_.imshow(confusion_normalized, cmap=plt.cm.jet,
                       interpolation='nearest')

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
                str(number), xy=(y, x),
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
        left=margin_small,
        right=margin_large,
        bottom=margin_small,
        top=margin_large
    )

    correct_rate = correct / total
    fuzzy_rate = fuzzy / total
    return correct_rate, fuzzy_rate


def general_intersection_over_union(bbox1, bbox2):
    x_overlap = max(
        0.0,
        min(bbox1['xbr'], bbox2['xbr']) - max(bbox1['xtl'], bbox2['xtl'])
    )
    y_overlap = max(
        0.0,
        min(bbox1['ybr'], bbox2['ybr']) - max(bbox1['ytl'], bbox2['ytl'])
    )
    intersection = float(x_overlap * y_overlap)
    union = (bbox1['width'] * bbox1['height']) + (bbox2['width'] * bbox2['height']) - intersection
    return intersection / union


def general_overlap(gt_list, pred_list):
    overlap = np.zeros((len(gt_list), len(pred_list)), dtype=np.float32)
    for i, gt in enumerate(gt_list):
        for j, pred in enumerate(pred_list):
            overlap[i, j] = general_intersection_over_union(gt, pred)
    return overlap


def general_tp_fp_fn(gt_list, pred_list, min_overlap,
                     check_species=True, check_viewpoint=False,
                     check_intereset=True, **kwargs):
    OLD = False
    if OLD:
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
            if check_species or check_viewpoint:
                for gt, pred in assignment_dict.items():
                    # print(gt_list[gt]['class'], pred_list[pred]['class'])
                    # print(gt_list[gt]['viewpoint'], pred_list[pred]['viewpoint'])
                    if gt_list[gt]['class'] != pred_list[pred]['class']:
                        tp -= 1
                    elif check_viewpoint and gt_list[gt]['viewpoint'] != pred_list[pred]['viewpoint']:
                        tp -= 1
            fp = num_pred - tp
            fn = num_gt - tp
        return tp, fp, fn
    else:
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
            confidence_list = [
                pred.get('confidence', None)
                for pred in pred_list
            ]
            assert None not in confidence_list
            zipped = zip(
                confidence_list,
                max_overlap_list,
                pred_index_list,
                gt_index_list
            )
            pred_conf_list = [
                (
                    confidence,
                    max_overlap,
                    pred_index,
                    gt_index,
                )
                for confidence, max_overlap, pred_index, gt_index in zipped
            ]
            pred_conf_list = sorted(pred_conf_list, reverse=True)

            assignment_dict = {}
            for pred_conf, max_overlap, pred_index, gt_index in pred_conf_list:
                if max_overlap > min_overlap:
                    if gt_index not in assignment_dict:
                        assignment_dict[gt_index] = pred_index

            assign_mod_set = set([])
            gt_mod_set = set([])
            pred_mod_set = set([])
            if check_species:
                species_set = kwargs.get('species_set', None)
                if species_set is not None:
                    for gt_index in assignment_dict:
                        pred_index = assignment_dict[gt_index]
                        if gt_list[gt_index]['class'] not in species_set:
                            assign_mod_set.add(gt_index)
                            pred_mod_set.add(pred_index)

                    for gt_index in range(num_gt):
                        if gt_list[gt_index]['class'] not in species_set:
                            gt_mod_set.add(gt_index)

            if check_intereset:
                for gt_index in assignment_dict:
                    pred_index = assignment_dict[gt_index]
                    if not gt_list[gt_index]['interest']:
                        assign_mod_set.add(gt_index)
                        pred_mod_set.add(pred_index)

                for gt_index in range(num_gt):
                    if not gt_list[gt_index]['interest']:
                        gt_mod_set.add(gt_index)

            tp = len(assignment_dict.keys()) - len(assign_mod_set)
            fp = num_pred - len(pred_mod_set) - tp
            fn = num_gt - len(gt_mod_set) - tp
            assert tp >= 0
            assert fp >= 0
            assert fn >= 0

        return tp, fp, fn


def general_get_imageset_gids(ibs, imageset_text, species_set=None,
                              filter_images=True, unique=False,
                              **kwargs):
    imageset_id = ibs.get_imageset_imgsetids_from_text(imageset_text)
    test_gid_list = ibs.get_imageset_gids(imageset_id)
    if filter_images and species_set is not None:
        species_set = set(species_set)
        args = (len(test_gid_list), species_set, )
        print('Filtering GIDs (%d) on species set: %r' % args)
        aids_list = ibs.get_image_aids(test_gid_list)
        species_list_list = ibs.unflat_map(ibs.get_annot_species_texts, aids_list)
        species_set_list = map(set, species_list_list)
        zipped = zip(test_gid_list, species_set_list)
        test_gid_list = [
            test_gid
            for test_gid, species_set_ in zipped
            if len(species_set_ & species_set) > 0
        ]
        print('    %d GIDs survived' % (len(test_gid_list), ))
    if unique:
        test_gid_list = list(set(test_gid_list))
    return test_gid_list


def general_parse_gt(ibs, test_gid_list=None, **kwargs):
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
                'gid'        : gid,
                'xtl'        : bbox[0] / width,
                'ytl'        : bbox[1] / height,
                'xbr'        : (bbox[0] + bbox[2]) / width,
                'ybr'        : (bbox[1] + bbox[3]) / height,
                'width'      : bbox[2] / width,
                'height'     : bbox[3] / height,
                'class'      : ibs.get_annot_species_texts(aid),
                'viewpoint'  : ibs.get_annot_viewpoints(aid),
                'interest'   : ibs.get_annot_interest(aid),
                'confidence' : 1.0,
            }
            gt_list.append(temp)
        gt_dict[uuid] = gt_list
    return gt_dict


def localizer_parse_pred(ibs, test_gid_list=None, **kwargs):
    depc = ibs.depc_image

    if 'feature2_algo' not in kwargs:
        kwargs['feature2_algo'] = 'resnet'

    if test_gid_list is None:
        test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
    uuid_list = ibs.get_image_uuids(test_gid_list)

    size_list = ibs.get_image_sizes(test_gid_list)
    bboxes_list = depc.get_property('localizations', test_gid_list, 'bboxes',  config=kwargs)
    thetas_list = depc.get_property('localizations', test_gid_list, 'thetas',  config=kwargs)
    confss_list = depc.get_property('localizations', test_gid_list, 'confs',   config=kwargs)
    classs_list = depc.get_property('localizations', test_gid_list, 'classes', config=kwargs)

    length_list = [ len(bbox_list) for bbox_list in bboxes_list ]

    # Establish primitives
    test_gids_list = [ [test_gid] * length for test_gid, length in zip(test_gid_list, length_list) ]
    sizes_list = [ [size] * length for size, length in zip(size_list, length_list) ]
    keeps_list = [ [True] * length for length in length_list ]
    features_list = [ [None] * length for length in length_list ]
    features_lazy_list = [ [None] * length for length in length_list ]

    # Get features
    if kwargs.get('features', False):
        features_list = depc.get_property('localizations_features', test_gid_list,
                                          'vector', config=kwargs)

    if kwargs.get('features_lazy', False):
        from functools import partial

        def features_lazy_func(gid, offset):
            vector_list = depc.get_property('localizations_features', gid,
                                            'vector', config=kwargs)
            vector = vector_list[offset]
            return vector

        features_lazy_list = [
            [
                partial(features_lazy_func, test_gid, test_offset)
                for test_offset in range(length)
            ]
            for test_gid, length in zip(test_gid_list, length_list)
        ]

    # Get updated confidences for boxes
    if kwargs.get('classify', False):
        print('Using alternate classifications')
        # depc.delete_property('localizations_classifier', test_gid_list, config=kwargs)
        confss_list = depc.get_property('localizations_classifier', test_gid_list,
                                        'score', config=kwargs)

    # Apply NMS
    if kwargs.get('nms', False):
        nms_thresh = kwargs.get('nms_thresh', 0.2)
        print('Filtering with nms_thresh = %0.02f' % (nms_thresh, ))
        count_old_list = []
        count_new_list = []
        keeps_list = []
        for bbox_list, confs_list in zip(bboxes_list, confss_list):
            # Compile coordinate list of (xtl, ytl, xbr, ybr) instead of (xtl, ytl, w, h)
            coord_list = []
            for xtl, ytl, width, height in bbox_list:
                xbr = xtl + width
                ybr = ytl + height
                coord_list.append([xtl, ytl, xbr, ybr])
            coord_list = np.vstack(coord_list)
            # Perform NMS
            keep_indices_list = nms(coord_list, confs_list, nms_thresh)
            # Analytics
            count_old_list.append(len(coord_list))
            count_new_list.append(len(keep_indices_list))
            keep_indices_set = set(keep_indices_list)
            # Keep track of which indices to keep
            keep_list = [ index in keep_indices_set for index in range(len(coord_list)) ]
            keeps_list.append(keep_list)
        # Print analytics
        count_old = sum(count_old_list)
        count_old_avg = count_old / len(count_old_list)
        count_new = sum(count_new_list)
        count_new_avg = count_new / len(count_new_list)
        count_diff = count_old - count_new
        args = (count_old, count_new, count_diff, 100.0 * count_diff / count_old, )
        print('[nms] %d old -> %d new (%d, %0.02f%% suppressed)' % args)
        args = (count_old_avg, count_new_avg, )
        print('[nms] %0.02f old avg. -> %0.02f new avg.' % args)

    # Filter by confidence or index
    if kwargs.get('thresh', False):
        conf_thresh = kwargs.get('conf_thresh', 0.0)
        index_thresh = kwargs.get('index_thresh', 1.0)
        print('Filtering with conf_thresh = %0.02f' % (conf_thresh, ))
        print('Filtering with index_thresh = %s' % (index_thresh, ))
        count_old_list = []
        count_new_list = []
        keeps_list_ = []
        for confs_list, keep_list in zip(confss_list, keeps_list):
            # Find percentage threshold for this image
            index_thresh_ = int(len(keep_list) * index_thresh)
            # Analytics
            count_old_list.append(keep_list.count(True))
            temp_list = []
            zipped = list(zip(confs_list, keep_list))
            for index, (conf, keep) in enumerate(sorted(zipped, reverse=True)):
                keep = keep and conf >= conf_thresh and index < index_thresh_
                temp_list.append(keep)
            keeps_list_.append(temp_list)
            # Analytics
            count_new_list.append(temp_list.count(True))
        keeps_list = keeps_list_
        # Print analytics
        count_old = sum(count_old_list)
        count_old_avg = count_old / len(count_old_list)
        count_new = sum(count_new_list)
        count_new_avg = count_new / len(count_new_list)
        count_diff = count_old - count_new
        args = (count_old, count_new, count_diff, 100.0 * count_diff / count_old, )
        print('[thresh] %d old -> %d new (%d, %0.02f%% suppressed)' % args)
        args = (count_old_avg, count_new_avg, )
        print('[thresh] %0.02f old avg. -> %0.02f new avg.' % args)
        # Alias

    # Reformat results for json
    zipped_list_list = zip(
        keeps_list,
        test_gids_list,
        sizes_list,
        bboxes_list,
        thetas_list,
        confss_list,
        classs_list,
        features_list,
        features_lazy_list,
    )
    results_list = [
        [
            {
                'gid'          : test_gid,
                'xtl'          : bbox[0] / width,
                'ytl'          : bbox[1] / height,
                'xbr'          : (bbox[0] + bbox[2]) / width,
                'ybr'          : (bbox[1] + bbox[3]) / height,
                'width'        : bbox[2] / width,
                'height'       : bbox[3] / height,
                'theta'        : theta,
                'confidence'   : conf,
                'class'        : class_,
                'feature'      : feature,
                'feature_lazy' : feature_lazy,
            }
            for keep_, test_gid, (width, height), bbox, theta, conf, class_, feature, feature_lazy in zip(*zipped_list)
            if keep_
        ]
        for zipped_list in zipped_list_list
    ]

    pred_dict = {
        uuid_ : result_list
        for uuid_, result_list in zip(uuid_list, results_list)
    }
    return pred_dict


def localizer_precision_recall_algo(ibs, samples=SAMPLES, force_serial=FORCE_SERIAL,
                                    **kwargs):
    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
    uuid_list = ibs.get_image_uuids(test_gid_list)

    print('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **kwargs)

    print('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

    # species_set = kwargs.get('species_set', None)
    # if filter_annots and species_set is not None:
    #     dict_list = [
    #         (gt_dict, 'Ground-Truth'),
    #         # (pred_dict, 'Predictions'),
    #     ]
    #     for dict_, dict_tag in dict_list:
    #         total = 0
    #         survived = 0
    #         for image_uuid in dict_:
    #             annot_list = dict_[image_uuid]
    #             total += len(annot_list)
    #             annot_list = [
    #                 annot
    #                 for annot in annot_list
    #                 if annot.get('class', None) in species_set
    #             ]
    #             survived += len(annot_list)
    #             dict_[image_uuid] = annot_list
    #         args = (dict_tag, total, species_set)
    #         print('Filtering %s AIDs (%d) on species set: %r' % args)
    #         print('    %d AIDs survived' % (survived , ))

    print('\tGenerate Curves...')
    conf_list = [ _ / float(samples) for _ in range(0, int(samples) + 1) ]
    conf_list = sorted(conf_list, reverse=True)

    uuid_list_list = [ uuid_list for _ in conf_list ]
    gt_dict_list   = [ gt_dict   for _ in conf_list ]
    pred_dict_list = [ pred_dict for _ in conf_list ]
    kwargs_list    = [ kwargs    for _ in conf_list ]
    arg_iter = zip(conf_list, uuid_list_list, gt_dict_list, pred_dict_list, kwargs_list)
    pr_re_gen = ut.generate2(localizer_precision_recall_algo_worker, arg_iter,
                             nTasks=len(conf_list), ordered=True,
                             chunksize=CHUNK_SIZE, force_serial=force_serial)

    conf_list_ = [-1.0, -1.0]
    pr_list = [1.0, 0.0]
    re_list = [0.0, 1.0]
    # conf_list_ = []
    # pr_list = []
    # re_list = []
    for values in pr_re_gen:
        if values is None:
            continue
        conf, pr, re = values
        conf_list_.append(conf)
        pr_list.append(pr)
        re_list.append(re)

    print('...complete')
    return conf_list_, pr_list, re_list


def localizer_precision_recall_algo_worker(conf, uuid_list, gt_dict, pred_dict,
                                           kwargs):
    # print('Started %s' % (conf, ))
    tp, fp, fn = 0.0, 0.0, 0.0
    for index, uuid_ in enumerate(uuid_list):
        if uuid_ in pred_dict:
            gt_list = gt_dict[uuid_]
            pred_list = [
                pred
                for pred in pred_dict[uuid_]
                if pred['confidence'] >= conf
            ]
            tp_, fp_, fn_ = general_tp_fp_fn(gt_list, pred_list, **kwargs)
            tp += tp_
            fp += fp_
            fn += fn_
    try:
        pr = tp / (tp + fp)
        re = tp / (tp + fn)
    except ZeroDivisionError:
        return None
    return (conf, pr, re)


def nms(dets, scores, thresh, use_cpu=True):
    # Interface into Faster R-CNN's Python native NMS algorithm by Girshick et al.
    from ibeis.algo.detect.nms.py_cpu_nms import py_cpu_nms
    return py_cpu_nms(dets, scores, thresh)


def localizer_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing Precision-Recall for: %r' % (label, ))
    conf_list, pr_list, re_list = localizer_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def localizer_confusion_matrix_algo_plot(ibs, color, conf, label=None, min_overlap=0.5,
                                         write_images=False, **kwargs):
    print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf, ))

    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
    test_uuid_list = ibs.get_image_uuids(test_gid_list)

    print('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **kwargs)

    print('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

    # species_set = kwargs.get('species_set', None)
    # if filter_annots and species_set is not None:
    #     dict_list = [
    #         (gt_dict, 'Ground-Truth'),
    #         # (pred_dict, 'Predictions'),
    #     ]
    #     for dict_, dict_tag in dict_list:
    #         total = 0
    #         survived = 0
    #         for image_uuid in dict_:
    #             annot_list = dict_[image_uuid]
    #             total += len(annot_list)
    #             annot_list = [
    #                 annot
    #                 for annot in annot_list
    #                 if annot.get('class', None) in species_set
    #             ]
    #             survived += len(annot_list)
    #             dict_[image_uuid] = annot_list
    #         args = (dict_tag, total, species_set)
    #         print('Filtering %s AIDs (%d) on species set: %r' % args)
    #         print('    %d AIDs survived' % (survived , ))

    if write_images:
        output_folder = 'localizer-precision-recall-%0.2f-images' % (min_overlap, )
        output_path = abspath(expanduser(join('~', 'Desktop', output_folder)))
        ut.ensuredir(output_path)

    label_list = []
    prediction_list = []
    for index, (test_gid, test_uuid) in enumerate(zip(test_gid_list, test_uuid_list)):
        if test_uuid in pred_dict:
            gt_list = gt_dict[test_uuid]
            pred_list = [
                pred
                for pred in pred_dict[test_uuid]
                if pred['confidence'] >= conf
            ]
            tp, fp, fn = general_tp_fp_fn(gt_list, pred_list, min_overlap=min_overlap,
                                          **kwargs)
            for _ in range(int(tp)):
                label_list.append('positive')
                prediction_list.append('positive')
            for _ in range(int(fp)):
                label_list.append('negative')
                prediction_list.append('positive')
            for _ in range(int(fn)):
                label_list.append('positive')
                prediction_list.append('negative')

            if write_images:
                test_image = ibs.get_image_imgdata(test_gid)
                test_image = _resize(test_image, t_width=600, verbose=False)
                height_, width_, channels_ = test_image.shape

                for gt in gt_list:
                    xtl = int(gt['xtl'] * width_)
                    ytl = int(gt['ytl'] * height_)
                    xbr = int(gt['xbr'] * width_)
                    ybr = int(gt['ybr'] * height_)
                    cv2.rectangle(test_image, (xtl, ytl), (xbr, ybr), (0, 255, 0))

                for pred in pred_list:
                    xtl = int(pred['xtl'] * width_)
                    ytl = int(pred['ytl'] * height_)
                    xbr = int(pred['xbr'] * width_)
                    ybr = int(pred['ybr'] * height_)
                    cv2.rectangle(test_image, (xtl, ytl), (xbr, ybr), (0, 0, 255))

                status_str = 'success' if (fp + fn) == 0 else 'failure'
                status_val = tp - fp - fn
                args = (status_str, status_val, test_gid, tp, fp, fn, )
                output_filename = 'test_%s_%d_gid_%d_tp_%d_fp_%d_fn_%d.png' % args
                output_filepath = join(output_path, output_filename)
                cv2.imwrite(output_filepath, test_image)

    category_list = ['positive', 'negative']
    category_mapping = {
        'positive': 0,
        'negative': 1,
    }
    return general_confusion_matrix_algo(label_list, prediction_list, category_list,
                                         category_mapping, size=20, **kwargs)


@register_ibs_method
def localizer_precision_recall_algo_display(ibs, min_overlap=0.5, figsize=(30, 9),
                                            write_images=False, min_recall=0.9,
                                            plot_point=True, **kwargs):
    import matplotlib.pyplot as plt
    import plottool as pt

    fig_ = plt.figure(figsize=figsize, dpi=400)

    axes_ = plt.subplot(131)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall (Ground-Truth IOU >= %0.02f)' % (min_overlap, ))
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])

    # species_set = set(['zebra'])
    # species_set = set(['giraffe'])
    # species_set = set(['elephant'])
    # species_set = None

    species_set = [
        'giraffe_masai',
        'giraffe_reticulated',
        'turtle_sea',
        'whale_fluke',
        'zebra_grevys',
        'zebra_plains',
    ]

    config_list = [


        {'label': 'All Species',         'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'filter_annots' : True, 'species_set' : species_set},
        {'label': 'Masai Giraffe',       'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'filter_annots' : True, 'species_set' : [ species_set[0] ]},
        {'label': 'Reticulated Giraffe', 'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'filter_annots' : True, 'species_set' : [ species_set[1] ]},
        {'label': 'Sea Turtle',          'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'filter_annots' : True, 'species_set' : [ species_set[2] ]},
        {'label': 'Whale Fluke',         'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'filter_annots' : True, 'species_set' : [ species_set[3] ]},
        {'label': 'Grevy\'s Zebra',      'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'filter_annots' : True, 'species_set' : [ species_set[4] ]},
        {'label': 'Plains Zebra',        'grid' : False, 'config_filepath' : 'candidacy', 'weight_filepath' : 'candidacy', 'filter_annots' : True, 'species_set' : [ species_set[5] ]},


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

        # {'label': 'Sand Tiger',        'grid' : False, 'config_filepath' : 'sandtiger', 'weight_filepath' : 'sandtiger'},
        # {'label': 'Sand Tiger (Grid)', 'grid' : True,  'config_filepath' : 'sandtiger', 'weight_filepath' : 'sandtiger'},

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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.image.zebra.rbf.1.0.pkl',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.10.rbf.1.0',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.20.rbf.1.0',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.30.rbf.1.0',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.40.rbf.1.0',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.50.rbf.1.0',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.60.rbf.1.0',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.70.rbf.1.0',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.80.rbf.1.0',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.90.rbf.1.0',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.100.rbf.1.0',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models/classifier.svm.image.zebra.linear.0.5.pkl',
        # },
        # {
        #     'label'        : 'LINEAR,1.0',
        #     'algo'         : '_COMBINED',
        #     'species_set'  : species_set,
        #     'classify'     : True,
        #     'classifier_algo': 'svm',
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models/classifier.svm.image.zebra.linear.1.0.pkl',
        # },
        # {
        #     'label'        : 'LINEAR,2.0',
        #     'algo'         : '_COMBINED',
        #     'species_set'  : species_set,
        #     'classify'     : True,
        #     'classifier_algo': 'svm',
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models/classifier.svm.image.zebra.linear.2.0.pkl',
        # },
        # {
        #     'label'        : 'RBF,0.5',
        #     'algo'         : '_COMBINED',
        #     'species_set'  : species_set,
        #     'classify'     : True,
        #     'classifier_algo': 'svm',
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models/classifier.svm.image.zebra.rbf.0.5.pkl',
        # },
        # {
        #     'label'        : 'RBF,1.0',
        #     'algo'         : '_COMBINED',
        #     'species_set'  : species_set,
        #     'classify'     : True,
        #     'classifier_algo': 'svm',
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models/classifier.svm.image.zebra.rbf.1.0.pkl',
        # },
        # {
        #     'label'        : 'RBF,2.0',
        #     'algo'         : '_COMBINED',
        #     'species_set'  : species_set,
        #     'classify'     : True,
        #     'classifier_algo': 'svm',
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models/classifier.svm.image.zebra.rbf.2.0.pkl',
        # },
        # {
        #     'label'        : 'LINEAR,0.5~0.5',
        #     'algo'         : '_COMBINED',
        #     'species_set'  : species_set,
        #     'classify'     : True,
        #     'classifier_algo': 'svm',
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models/classifier.svm.image.zebra.linear.0.5.pkl',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models/classifier.svm.image.zebra.linear.1.0.pkl',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models/classifier.svm.image.zebra.linear.2.0.pkl',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models/classifier.svm.image.zebra.rbf.0.5.pkl',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.image.zebra.rbf.1.0.pkl',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models/classifier.svm.image.zebra.rbf.2.0.pkl',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.image.zebra.pkl',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.image.zebra.pkl',
        #     # 'thresh'       : True,
        #     # 'index_thresh' : 0.25,
        # },

        # {
        #     'label'        : 'WIC ~0.25',
        #     'algo'         : '_COMBINED',
        #     'species_set'  : species_set,
        #     'classify'     : True,
        #     'classifier_algo': 'svm',
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.image.zebra.pkl',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.image.zebra.pkl',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.image.zebra.pkl',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.10',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.20',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.30',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.40',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.50',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.60',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.70',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.80',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.90',
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
        #     'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.localization.zebra.100',
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

    # color_list = pt.distinct_colors(len(config_list), randomize=False)

    # color_list = pt.distinct_colors(len(config_list) - 2, randomize=False)
    # color_list += [(0.2, 0.2, 0.2)]
    # color_list += [(0.2, 0.2, 0.2)]

    # color_list_ = []
    color_list_ = [(0.2, 0.2, 0.2)]
    # color_list_ = [(0.2, 0.2, 0.2), (0.2, 0.2, 0.2)]
    color_list = pt.distinct_colors(len(config_list) - len(color_list_), randomize=False)
    color_list = color_list_ + color_list

    # color_list = pt.distinct_colors(len(config_list) // 2, randomize=False)
    # color_list = color_list + color_list

    ret_list = [
        localizer_precision_recall_algo_plot(ibs, color=color, min_overlap=min_overlap,
                                             plot_point=plot_point, **config)
        for color, config in zip(color_list, config_list)
    ]

    area_list = [ ret[0] for ret in ret_list ]
    conf_list = [ ret[1] for ret in ret_list ]
    index = np.argmax(area_list)
    # index = 0
    best_label = config_list[index]['label']  # NOQA
    best_color = color_list[index]
    best_config = config_list[index]
    best_area = area_list[index]  # NOQA
    best_conf = conf_list[index]
    print('BEST OPERATING POINT %0.04f' % (best_conf[0], ))
    # plt.title('Precision-Recall Curve (Best: %s, AP = %0.02f)' % (best_label, best_area, ), y=1.13)
    plt.title('Precision-Recall Curves', y=1.19)

    # Display graph of the overall highest area
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(132)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = localizer_confusion_matrix_algo_plot(ibs, best_color, best_conf,
                                                           min_overlap=min_overlap,
                                                           write_images=write_images,
                                                           fig_=fig_, axes_=axes_,
                                                           **best_config)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    # args = (best_area, best_label, best_conf, )
    # plt.title('Confusion Matrix for Highest AP %0.02f\n(Algo: %s, OP = %0.02f)' % args, y=1.26)

    # area_list_ = area_list[1:]
    # mAP = sum(area_list_) / len(area_list_)
    # args = (mAP * 100.0, )
    # plt.title('Confusion Matrix\nmAP = %0.02f' % args, y=1.26)

    plt.title('Confusion Matrix', y=1.26)

    # # Show best that is greater than the best_pr
    # best_index = None
    # best_conf = None
    # best_pr = 0.0
    # best_re = 0.0
    # tup_list  = [ ret[2] for ret in ret_list ]
    # for index, tup in enumerate(tup_list):
    #     for conf, re, pr in zip(*tup):
    #         # if pr > best_pr:
    #         #     best_index = index
    #         #     best_conf = conf
    #         #     best_pr = pr
    #         #     best_re = re
    #         if re > best_re:
    #             best_index = index
    #             best_conf = conf
    #             best_pr = pr
    #             best_re = re

    # if best_index is not None:
    #     axes_ = plt.subplot(131)
    #     plt.plot([best_re], [best_pr], 'yo')

    #     best_label = config_list[best_index]['label']
    #     best_color = color_list[index]
    #     best_config = config_list[best_index]

    #     axes_ = plt.subplot(133)
    #     axes_.set_aspect(1)
    #     gca_ = plt.gca()
    #     gca_.grid(False)
    #     correct_rate, _ = localizer_confusion_matrix_algo_plot(ibs, best_color, best_conf,
    #                                                            min_overlap=min_overlap,
    #                                                            fig_=fig_, axes_=axes_,
    #                                                            **best_config)
    #     axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    #     axes_.set_ylabel('Ground-Truth')
    #     # args = (min_recall, best_label, best_conf, )
    #     # plt.title('P-R Confusion Matrix for Highest Precision with Recall >= %0.02f\n(Algo: %s, OP = %0.02f)' % args, y=1.26)
    #     args = (best_re, best_label, best_conf, )
    #     plt.title('Confusion Matrix for Highest Recall %0.02f\n(Algo: %s, OP = %0.02f)' % args, y=1.26)

    # plt.show()
    fig_filename = 'localizer-precision-recall-%0.2f.png' % (min_overlap, )
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


@register_ibs_method
def localizer_precision_recall_algo_display_animate(ibs, **kwargs):
    for value in range(10):
        min_overlap = value / 10.0
        print('Processing: %r' % (min_overlap, ))
        ibs.localizer_precision_recall_algo_display(min_overlap=min_overlap, **kwargs)


def localizer_classification_tp_tn_fp_fn(gt_list, pred_list, conf, min_overlap,
                                         check_species=False,
                                         check_viewpoint=False, **kwargs):
    overlap = general_overlap(gt_list, pred_list)
    num_gt, num_pred = overlap.shape

    # Get confidences
    conf_list = [pred['confidence'] for pred in pred_list]
    pred_flag_list = [conf <= conf_ for conf_ in conf_list]

    if num_gt == 0:
        tp_list = [False] * len(pred_list)
        tn_list = [not pred_flag for pred_flag in pred_flag_list]
        fp_list = [    pred_flag for pred_flag in pred_flag_list]
        fn_list = [False] * len(pred_list)
    elif num_pred == 0:
        tp_list = []
        tn_list = []
        fp_list = []
        fn_list = []
    else:
        max_overlap = np.max(overlap, axis=0)
        gt_flag_list = min_overlap < max_overlap

        status_list = []
        for gt_flag, pred_flag in zip(gt_flag_list, pred_flag_list):
            if gt_flag and pred_flag:
                status_list.append('tp')
            elif gt_flag and not pred_flag:
                status_list.append('fn')
            elif not gt_flag and pred_flag:
                status_list.append('fp')
            elif not gt_flag and not pred_flag:
                status_list.append('tn')
            else:
                raise ValueError

        tp_list = [status == 'tp' for status in status_list]
        tn_list = [status == 'tn' for status in status_list]
        fp_list = [status == 'fp' for status in status_list]
        fn_list = [status == 'fn' for status in status_list]

    return tp_list, tn_list, fp_list, fn_list


def localizer_classification_confusion_matrix_algo_plot(ibs, color, conf,
                                                        label=None,
                                                        min_overlap=0.25,
                                                        write_images=False,
                                                        **kwargs):
    print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf, ))

    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
    test_uuid_list = ibs.get_image_uuids(test_gid_list)

    print('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **kwargs)

    print('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

    if write_images:
        output_folder = 'localizer-classification-confusion-matrix-%0.2f-%0.2f-images' % (min_overlap, conf, )
        output_path = abspath(expanduser(join('~', 'Desktop', output_folder)))
        ut.ensuredir(output_path)

    label_list = []
    prediction_list = []
    for index, (test_gid, test_uuid) in enumerate(zip(test_gid_list, test_uuid_list)):
        if test_uuid in pred_dict:
            gt_list = gt_dict[test_uuid]
            pred_list = pred_dict[test_uuid]
            values = localizer_classification_tp_tn_fp_fn(gt_list, pred_list, conf,
                                                          min_overlap=min_overlap,
                                                          **kwargs)
            tp_list, tn_list, fp_list, fn_list = values
            tp = tp_list.count(True)
            tn = tn_list.count(True)
            fp = fp_list.count(True)
            fn = fn_list.count(True)

            for _ in range(int(tp)):
                label_list.append('positive')
                prediction_list.append('positive')
            for _ in range(int(tn)):
                label_list.append('negative')
                prediction_list.append('negative')
            for _ in range(int(fp)):
                label_list.append('negative')
                prediction_list.append('positive')
            for _ in range(int(fn)):
                label_list.append('positive')
                prediction_list.append('negative')

            if write_images:
                test_image = ibs.get_image_imgdata(test_gid)
                test_image = _resize(test_image, t_width=600, verbose=False)
                height_, width_, channels_ = test_image.shape

                for gt in gt_list:
                    xtl = int(gt['xtl'] * width_)
                    ytl = int(gt['ytl'] * height_)
                    xbr = int(gt['xbr'] * width_)
                    ybr = int(gt['ybr'] * height_)
                    cv2.rectangle(test_image, (xtl, ytl), (xbr, ybr), (0, 0, 255))

                zipped = zip(pred_list, tp_list, tn_list, fp_list, fn_list)
                for pred, tp_, tn_, fp_, fn_ in zipped:
                    if tp_:
                        color = (0, 255, 0)
                    elif fp_:
                        continue
                        # color = (255, 0, 0)
                    elif fn_:
                        color = (255, 0, 0)
                    elif tn_:
                        continue
                    else:
                        continue

                    xtl = int(pred['xtl'] * width_)
                    ytl = int(pred['ytl'] * height_)
                    xbr = int(pred['xbr'] * width_)
                    ybr = int(pred['ybr'] * height_)
                    cv2.rectangle(test_image, (xtl, ytl), (xbr, ybr), color)

                status_str = 'success' if (fp + fn) == 0 else 'failure'
                status_val = tp - fp - fn
                args = (status_str, status_val, test_gid, tp, fp, fn, )
                output_filename = 'test_%s_%d_gid_%d_tp_%d_fp_%d_fn_%d.png' % args
                output_filepath = join(output_path, output_filename)
                cv2.imwrite(output_filepath, test_image)

    category_list = ['positive', 'negative']
    category_mapping = {
        'positive': 0,
        'negative': 1,
    }
    return general_confusion_matrix_algo(label_list, prediction_list, category_list,
                                         category_mapping, size=20, **kwargs)


@register_ibs_method
def localizer_classifications_confusion_matrix_algo_display(ibs, conf,
                                                            min_overlap=0.25,
                                                            figsize=(24, 7),
                                                            write_images=False,
                                                            min_recall=0.9,
                                                            plot_point=True,
                                                            masking=False,
                                                            **kwargs):
    import matplotlib.pyplot as plt

    fig_ = plt.figure(figsize=figsize)

    config = {
        'label'        : 'WIC',
        'algo'         : '_COMBINED',
        'species_set'  : set(['zebra']),
        'classify'     : True,
        'classifier_algo': 'svm',
        'classifier_masking': masking,
        'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.image.zebra.pkl',
    }

    axes_ = plt.subplot(111)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)

    correct_rate, _ = localizer_classification_confusion_matrix_algo_plot(ibs, None, conf,
                                                                          min_overlap=min_overlap,
                                                                          write_images=write_images,
                                                                          fig_=fig_, axes_=axes_,
                                                                          **config)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    args = (min_overlap, conf, )
    plt.title('Confusion Matrix (IoU %0.02f, Conf %0.02f)' % args, y=1.13)

    # plt.show()
    args = (min_overlap, conf, )
    fig_filename = 'localizer-classification-confusion-matrix-%0.2f-%0.2f.png' % args
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


@register_ibs_method
def localizer_classifications_confusion_matrix_algo_display_animate(ibs, total=10, **kwargs):
    for index in range(0, total + 1):
        conf = index / total
        ibs.localizer_classifications_confusion_matrix_algo_display(conf, **kwargs)


def classifier_cameratrap_precision_recall_algo(ibs, positive_imageset_id, negative_imageset_id, **kwargs):
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

    prediction_list = depc.get_property('classifier', test_gid_set, 'class', config=kwargs)
    confidence_list = depc.get_property('classifier', test_gid_set, 'score', config=kwargs)
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence in zip(prediction_list, confidence_list)
    ]
    return general_precision_recall_algo(ibs, label_list, confidence_list, **kwargs)


def classifier_cameratrap_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing Precision-Recall for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier_cameratrap_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def classifier_cameratrap_roc_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing ROC for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier_cameratrap_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, fpr_list, tpr_list, interpolate=False,
                                  target=(0.0, 1.0), **kwargs)


def classifier_cameratrap_confusion_matrix_algo_plot(ibs, label, color, conf, positive_imageset_id, negative_imageset_id, output_cases=False, **kwargs):
    print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf, ))
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

    prediction_list = depc.get_property('classifier', test_gid_set, 'class', config=kwargs)
    confidence_list = depc.get_property('classifier', test_gid_set, 'score', config=kwargs)
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence  in zip(prediction_list, confidence_list)
    ]
    prediction_list = [
        'positive' if confidence >= conf else 'negative'
        for confidence in confidence_list
    ]

    if output_cases:
        output_path = 'cameratrap-confusion-incorrect'
        output_path = abspath(expanduser(join('~', 'Desktop', output_path)))
        positive_path = join(output_path, 'positive')
        negative_path = join(output_path, 'negative')
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
            image_filename = 'hardidx_%d_pred_%s_case_fail.jpg' % (gid, prediction, )
            image_filepath = join(image_path, image_filename)
            # Save path
            cv2.imwrite(image_filepath, image)

    category_list = ['positive', 'negative']
    category_mapping = {
        'positive': 0,
        'negative': 1,
    }
    return general_confusion_matrix_algo(label_list, prediction_list, category_list,
                                         category_mapping, **kwargs)


@register_ibs_method
def classifier_cameratrap_precision_recall_algo_display(ibs, positive_imageset_id, negative_imageset_id, figsize=(16, 16)):
    import matplotlib.pyplot as plt
    import plottool as pt

    fig_ = plt.figure(figsize=figsize)

    config_list = [
        {'label': 'Initial Model   (0%)', 'classifier_weight_filepath': 'megan2.1'},
        {'label': 'Retrained Model (1%)', 'classifier_weight_filepath': 'megan2.2'},

        # {'label': 'Initial Model   (0%)', 'classifier_weight_filepath': 'megan1.1'},
        # {'label': 'Retrained Model (1%)', 'classifier_weight_filepath': 'megan1.2'},
        # {'label': 'Retrained Model (2%)', 'classifier_weight_filepath': 'megan1.3'},
        # {'label': 'Retrained Model (3%)', 'classifier_weight_filepath': 'megan1.4'},
        # {'label': 'Retrained Model (3.5%)', 'classifier_weight_filepath': 'megan1.5'},
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
        classifier_cameratrap_precision_recall_algo_plot(ibs, color=color,
                                                         positive_imageset_id=positive_imageset_id,
                                                         negative_imageset_id=negative_imageset_id,
                                                         **config)
        for color, config in zip(color_list, config_list)
    ]
    area_list = [ ret[0] for ret in ret_list ]
    conf_list = [ ret[1] for ret in ret_list ]
    index = np.argmax(area_list)
    # index = 0
    best_label1 = config_list[index]['label']
    best_config1 = config_list[index]
    best_color1 = color_list[index]
    best_area1 = area_list[index]
    best_conf1 = conf_list[index]
    plt.title('Precision-Recall Curve (Best: %s, AP = %0.02f)' % (best_label1, best_area1, ), y=1.10)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(222)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    ret_list = [
        classifier_cameratrap_roc_algo_plot(ibs, color=color,
                                            positive_imageset_id=positive_imageset_id,
                                            negative_imageset_id=negative_imageset_id,
                                            **config)
        for color, config in zip(color_list, config_list)
    ]
    area_list = [ ret[0] for ret in ret_list ]
    conf_list = [ ret[1] for ret in ret_list ]
    index = np.argmax(area_list)
    # index = 0
    best_label2 = config_list[index]['label']
    best_config2 = config_list[index]
    best_color2 = color_list[index]
    best_area2 = area_list[index]
    best_conf2 = conf_list[index]
    plt.title('ROC Curve (Best: %s, AP = %0.02f)' % (best_label2, best_area2, ), y=1.10)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(223)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = classifier_cameratrap_confusion_matrix_algo_plot(ibs, color=best_color1,
                                                                       conf=best_conf1, fig_=fig_, axes_=axes_,
                                                                       positive_imageset_id=positive_imageset_id,
                                                                       negative_imageset_id=negative_imageset_id,
                                                                       output_cases=True, **best_config1)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    plt.title('P-R Confusion Matrix (OP = %0.02f)' % (best_conf1, ), y=1.12)

    axes_ = plt.subplot(224)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = classifier_cameratrap_confusion_matrix_algo_plot(ibs, color=best_color2,
                                                                       conf=best_conf2, fig_=fig_, axes_=axes_,
                                                                       positive_imageset_id=positive_imageset_id,
                                                                       negative_imageset_id=negative_imageset_id,
                                                                       **best_config2)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    plt.title('ROC Confusion Matrix (OP = %0.02f)' % (best_conf2, ), y=1.12)

    fig_filename = 'classifier-precision-recall-roc.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


def classifier_binary_precision_recall_algo(ibs, category_set, **kwargs):
    depc = ibs.depc_image
    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_set = list(test_gid_set)
    aids_list = ibs.get_image_aids(test_gid_set)
    species_set_list = [
        set(ibs.get_annot_species_texts(aid_list))
        for aid_list in aids_list
    ]
    label_list = [
        'negative' if len(species_set & category_set) == 0 else 'positive'
        for species_set in species_set_list
    ]
    prediction_list = depc.get_property('classifier', test_gid_set, 'class', config=kwargs)
    confidence_list = depc.get_property('classifier', test_gid_set, 'score', config=kwargs)
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence  in zip(prediction_list, confidence_list)
    ]
    return general_precision_recall_algo(ibs, label_list, confidence_list, **kwargs)


def classifier_binary_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing Precision-Recall for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier_binary_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def classifier_binary_roc_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing ROC for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier_binary_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, fpr_list, tpr_list, interpolate=False,
                                  target=(0.0, 1.0), **kwargs)


def classifier_binary_confusion_matrix_algo_plot(ibs, label, color, conf, category_set, **kwargs):
    print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf, ))
    depc = ibs.depc_image
    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_set = list(test_gid_set)
    aids_list = ibs.get_image_aids(test_gid_set)
    species_set_list = [
        set(ibs.get_annot_species_texts(aid_list))
        for aid_list in aids_list
    ]
    label_list = [
        'negative' if len(species_set & category_set) == 0 else 'positive'
        for species_set in species_set_list
    ]
    prediction_list = depc.get_property('classifier', test_gid_set, 'class', config=kwargs)
    confidence_list = depc.get_property('classifier', test_gid_set, 'score', config=kwargs)
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence  in zip(prediction_list, confidence_list)
    ]
    prediction_list = [
        'positive' if confidence >= conf else 'negative'
        for confidence in confidence_list
    ]

    category_list = ['positive', 'negative']
    category_mapping = {
        'positive': 0,
        'negative': 1,
    }
    return general_confusion_matrix_algo(label_list, prediction_list, category_list,
                                         category_mapping, **kwargs)


@register_ibs_method
def classifier_binary_precision_recall_algo_display(ibs, figsize=(16, 16), **kwargs):
    import matplotlib.pyplot as plt

    fig_ = plt.figure(figsize=figsize)

    # label = 'V1'
    # species_list = ['zebra']
    # kwargs['classifier_weight_filepath'] = 'coco_zebra'

    label = 'V3'
    species_list = ['zebra_plains', 'zebra_grevys']
    kwargs['classifier_weight_filepath'] = 'v3_zebra'

    category_set = set(species_list)

    axes_ = plt.subplot(221)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall')
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    area, best_conf1, _ = classifier_binary_precision_recall_algo_plot(ibs, label=label, color='r', category_set=category_set, **kwargs)
    plt.title('Precision-Recall Curve (AP = %0.02f)' % (area, ), y=1.10)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(222)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    area, best_conf2, _ = classifier_binary_roc_algo_plot(ibs, label=label, color='r', category_set=category_set, **kwargs)
    plt.title('ROC Curve (AP = %0.02f)' % (area, ), y=1.10)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(223)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = classifier_binary_confusion_matrix_algo_plot(ibs, label, 'r', conf=best_conf1, fig_=fig_, axes_=axes_, category_set=category_set, **kwargs)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    plt.title('P-R Confusion Matrix (OP = %0.02f)' % (best_conf1, ), y=1.12)

    axes_ = plt.subplot(224)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = classifier_binary_confusion_matrix_algo_plot(ibs, label, 'r', conf=best_conf2, fig_=fig_, axes_=axes_, category_set=category_set, **kwargs)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    plt.title('ROC Confusion Matrix (OP = %0.02f)' % (best_conf2, ), y=1.12)

    fig_filename = 'classifier-precision-recall-roc.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


def classifier2_precision_recall_algo(ibs, category, **kwargs):
    depc = ibs.depc_image
    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_set = list(test_gid_set)
    aids_list = ibs.get_image_aids(test_gid_set)
    species_set_list = list(map(ibs.get_annot_species_texts, aids_list))

    label_list = [
        'positive' if category in species_set else 'negative'
        for species_set in species_set_list
    ]

    confidence_dict_list = depc.get_property('classifier_two', test_gid_set, 'scores', config=kwargs)
    confidence_list = [
        confidence_dict[category]
        for confidence_dict in confidence_dict_list
    ]

    # config_ = {
    #     'draw_annots' : False,
    #     'thumbsize'   : (192, 192),
    # }
    # thumbnail_list = depc.get_property('thumbnails', test_gid_set, 'img', config=config_)
    # zipped = zip(test_gid_set, thumbnail_list, species_set_list, confidence_dict_list)
    # for index, (test_gid, thumbnail, species_set, confidence_dict) in enumerate(zipped):
    #     print(index)
    #     x = ';'.join(species_set)
    #     y = []
    #     for key in confidence_dict:
    #         y.append('%s-%0.04f' % (key, confidence_dict[key], ))
    #     y = ';'.join(y)
    #     image_path = '/home/jason/Desktop/batch3/image----%s----%s----%s----%s.png'
    #     cv2.imwrite(image_path % (index, test_gid, x, y), thumbnail)

    return general_precision_recall_algo(ibs, label_list, confidence_list, **kwargs)


def classifier2_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing Precision-Recall for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier2_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def classifier2_roc_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing ROC for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier2_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, fpr_list, tpr_list, interpolate=False,
                                  target=(0.0, 1.0), **kwargs)


def classifier2_confusion_matrix_algo_plot(ibs, category_set, samples=SAMPLES, **kwargs):

    def _get_prediction_list(conf, confidence_dict_list):
        predictions_list = [
            [
                category
                for category, confidence in confidence_dict.items()
                if conf <= confidence
            ]
            for confidence_dict in confidence_dict_list
        ]
        prediction_list = [
            ','.join(sorted(prediction_list_))
            for prediction_list_ in predictions_list
        ]
        return prediction_list

    def _get_accuracy(label_list, prediction_list):
        assert len(label_list) == len(prediction_list)
        correct = 0
        for label, prediction in zip(label_list, prediction_list):
            if label == prediction:
                correct += 1
        return correct / len(label_list)

    print('Processing Confusion Matrix')
    depc = ibs.depc_image
    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_set = list(test_gid_set)
    aids_list = ibs.get_image_aids(test_gid_set)
    species_set_list = [
        set(ibs.get_annot_species_texts(aid_list))
        for aid_list in aids_list
    ]

    label_list = [
        ','.join(sorted(list(species_set)))
        for species_set in species_set_list
    ]
    confidence_dict_list = depc.get_property('classifier_two', test_gid_set, 'scores', config=kwargs)

    # Find the best confidence
    conf_list = [ _ / float(samples) for _ in range(0, int(samples) + 1) ]

    print('Processing best prediction_list...')
    best_conf = None
    best_accuracy = 0.0
    for conf in conf_list:
        prediction_list = _get_prediction_list(conf, confidence_dict_list)
        accuracy = _get_accuracy(label_list, prediction_list)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_conf = conf

    label_list_ = ['positive'] * len(label_list)
    prediction_list = _get_prediction_list(best_conf, confidence_dict_list)
    prediction_list_ = [
        'positive' if label == prediction else 'negative'
        for label, prediction in zip(label_list, prediction_list)
    ]

    category_list = ['positive', 'negative']
    category_mapping = {
        'positive': 0,
        'negative': 1,
    }
    return best_conf, general_confusion_matrix_algo(label_list_, prediction_list_,
                                                    category_list, category_mapping,
                                                    size=20, **kwargs)


@register_ibs_method
def classifier2_precision_recall_algo_display(ibs, species_list=None,
                                              figsize=(20, 9), **kwargs):
    import matplotlib.pyplot as plt
    import plottool as pt

    fig_ = plt.figure(figsize=figsize, dpi=400)  # NOQA

    # kwargs['classifier_two_weight_filepath'] = 'v3'
    kwargs['classifier_two_weight_filepath'] = 'candidacy'

    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_set = list(test_gid_set)
    depc = ibs.depc_image
    depc.delete_property('classifier_two', test_gid_set, config=kwargs)

    if species_list is None:
        test_gid = test_gid_set[0]
        confidence_dict = depc.get_property('classifier_two', test_gid, 'scores', config=kwargs)
        species_list = confidence_dict.keys()

    category_set = sorted(species_list)

    nice_mapping = {
        'giraffe_masai'       : 'Masai Giraffe',
        'giraffe_reticulated' : 'Reticulated Giraffe',
        'turtle_sea'          : 'Sea Turtle',
        'whale_fluke'         : 'Whale Fluke',
        'zebra_grevys'        : 'Grevy\'s Zebra',
        'zebra_plains'        : 'Plains Zebra',
    }

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
        classifier2_precision_recall_algo_plot(ibs, color=color, **config)
    plt.title('Precision-Recall Curves', y=1.19)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(122)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])

    op_dict = {}
    for color, config in zip(color_list, config_list):
        values = classifier2_roc_algo_plot(ibs, color=color, **config)
        ap, best_conf, tup = values
        op_dict[config['category']] = best_conf

    plt.title('ROC Curves', y=1.19)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    ut.embed()
    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_set = list(test_gid_set)
    aids_list = ibs.get_image_aids(test_gid_set)
    species_set_list = list(map(set, map(ibs.get_annot_species_texts, aids_list)))
    confidence_dict_list = depc.get_property('classifier_two', test_gid_set, 'scores', config=kwargs)

    correct = 0
    for confidence_dict, species_set in zip(confidence_dict_list, species_set_list):
        species_set_ = set([])
        for key in confidence_dict:
            if op_dict[key] <= confidence_dict[key]:
                species_set_.add(key)
        if len(species_set ^ species_set_) == 0:
            correct += 1
    print('Accuracy: %0.04f' % (100.0 * correct / len(test_gid_set)))

    skipped_gid_list = []
    for test_gid, confidence_dict in zip(test_gid_set, confidence_dict_list):
        species_set_ = set([])
        for key in confidence_dict:
            if op_dict[key] <= confidence_dict[key]:
                species_set_.add(key)
        if len(species_set_) == 0:
            skipped_gid_list.append(test_gid)

    # from ibeis.ibeis.scripts.sklearn_utils import classification_report2

    # axes_ = plt.subplot(325)
    # axes_.set_aspect(1)
    # gca_ = plt.gca()
    # gca_.grid(False)
    # best_conf, (correct_rate, _) = classifier2_confusion_matrix_algo_plot(ibs, category_set=category_set, fig_=fig_, axes_=axes_, **kwargs)
    # axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    # axes_.set_ylabel('Ground-Truth')
    # plt.title('Confusion Matrix (OP = %0.02f)' % (best_conf, ), y=1.12)

    fig_filename = 'classifier2-precision-recall-roc.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


def labeler_tp_tn_fp_fn(ibs, category_list, viewpoint_mapping=None,
                        samples=SAMPLES, **kwargs):

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
    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_set = list(test_gid_set)
    aids_list = ibs.get_image_aids(test_gid_set)
    aid_list = ut.flatten(aids_list)
    # Get annot species and viewpoints
    species_list = ibs.get_annot_species_texts(aid_list)
    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    # Filter aids with species of interest and undefined viewpoints
    flag_list = [
        species in category_list
        for species, viewpoint in zip(species_list, viewpoint_list)
    ]
    if False in flag_list:
        aid_list = ut.compress(aid_list, flag_list)
        # Get new species and viewpoints
        viewpoint_list = ibs.get_annot_viewpoints(aid_list)
        species_list = ibs.get_annot_species_texts(aid_list)

    # Make ground-truth
    label_list = [
        '%s:%s' % (
            species,
            viewpoint_mapping.get(species, {}).get(viewpoint, viewpoint),
        )
        for species, viewpoint in zip(species_list, viewpoint_list)
    ]
    # Get predictions
    # depc.delete_property('labeler', aid_list, config=kwargs)
    probability_dict_list = depc.get_property('labeler', aid_list, 'probs', config=kwargs)

    value1_list = set(label_list)
    value2_list = set(probability_dict_list[0].keys())
    assert len(value1_list - value2_list) == 0
    assert len(value2_list - value1_list) == 0

    conf_list = [ _ / float(samples) for _ in range(0, int(samples) + 1) ]
    label_dict = {}
    for key in value1_list:
        print('\t%r' % (key, ))
        conf_dict = {}
        confidence_list = [
            probability_dict[key]
            for probability_dict in probability_dict_list
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
            print('Zero division error (%r) - tp: %r tn: %r fp: %r fn: %r' % (conf, tp, tn, fp, fn, ))

    return conf_list_, pr_list, re_list, tpr_list, fpr_list


def labeler_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    category_list = kwargs['category_list']
    print('Processing Precision-Recall for: %r (category_list = %r)' % (label, category_list, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = labeler_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def labeler_roc_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    category_list = kwargs['category_list']
    print('Processing ROC for: %r (category_list = %r)' % (label, category_list, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = labeler_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, fpr_list, tpr_list, interpolate=False,
                                  target=(0.0, 1.0), **kwargs)


def labeler_confusion_matrix_algo_plot(ibs, category_list, viewpoint_mapping, category_mapping=None, **kwargs):
    print('Processing Confusion Matrix')
    depc = ibs.depc_annot
    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
    test_gid_set = list(test_gid_set)
    aids_list = ibs.get_image_aids(test_gid_set)
    aid_list = ut.flatten(aids_list)
    species_list = ibs.get_annot_species_texts(aid_list)
    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    label_list = [
        '%s:%s' % (
            species,
            viewpoint_mapping.get(species, {}).get(viewpoint, viewpoint),
        )
        for species, viewpoint in zip(species_list, viewpoint_list)
    ]
    temp_list = [
        (aid, label)
        for aid, label in zip(aid_list, label_list)
        if label in category_list
    ]
    aid_list = [_[0] for _ in temp_list]
    label_list = [_[1] for _ in temp_list]
    conf_list = depc.get_property('labeler', aid_list, 'score', config=kwargs)
    species_list = depc.get_property('labeler', aid_list, 'species', config=kwargs)
    viewpoint_list = depc.get_property('labeler', aid_list, 'viewpoint', config=kwargs)
    prediction_list = [
        '%s:%s' % (species, viewpoint, )
        for species, viewpoint in zip(species_list, viewpoint_list)
    ]

    category_list = list(map(simple_code, category_list))
    label_list = list(map(simple_code, label_list))
    prediction_list = list(map(simple_code, prediction_list))
    if category_mapping is None:
        category_mapping = { key: index for index, key in enumerate(category_list) }
    category_mapping = {
        simple_code(key): category_mapping[key]
        for key in category_mapping
    }
    return general_confusion_matrix_algo(label_list, prediction_list, category_list,
                                                 category_mapping, conf_list=conf_list,
                                                 size=8, **kwargs)


@register_ibs_method
def labeler_precision_recall_algo_display(ibs, category_list=None, viewpoint_mapping=None,
                                          category_mapping=None, figsize=(30, 9), **kwargs):
    import matplotlib.pyplot as plt
    import plottool as pt

    if category_list is None:
        test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
        test_gid_set = list(test_gid_set)
        aids_list = ibs.get_image_aids(test_gid_set)
        aid_list = ut.flatten(aids_list)
        species_list = ibs.get_annot_species_texts(aid_list)
        category_list = sorted(list(set(species_list)))

    print('Compiling raw numbers...')
    kwargs['labeler_weight_filepath'] = 'candidacy'
    label_dict = labeler_tp_tn_fp_fn(ibs, category_list, viewpoint_mapping=viewpoint_mapping,
                                     **kwargs)

    config_list = [
        {'label': 'All Species',         'category_list': None},
        {'label': 'Masai Giraffe',       'category_list': ['giraffe_masai']},
        {'label': 'Reticulated Giraffe', 'category_list': ['giraffe_reticulated']},
        {'label': 'Sea Turtle',          'category_list': ['turtle_sea']},
        {'label': 'Whale Fluke',         'category_list': ['whale_fluke']},
        {'label': 'Grevy\'s Zebra',      'category_list': ['zebra_grevys']},
        {'label': 'Plains Zebra',        'category_list': ['zebra_plains']},
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
        area, conf, _ = labeler_precision_recall_algo_plot(ibs, label_dict=label_dict,
                                                           color=color, **config)
        area_list.append(area)
    plt.title('Precision-Recall Curve', y=1.19)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(132)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    for color, config in zip(color_list, config_list):
        labeler_roc_algo_plot(ibs, label_dict=label_dict,
                              color=color, **config)
    plt.title('ROC Curve', y=1.19)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    key_list = sorted(label_dict.keys())
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
    correct_rate, fuzzy_rate = labeler_confusion_matrix_algo_plot(ibs, key_list, viewpoint_mapping=viewpoint_mapping, category_mapping=category_mapping, fig_=fig_, axes_=axes_, fuzzy_dict=fuzzy_dict, **kwargs)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%, Species = %0.02f%%)' % (correct_rate * 100.0, fuzzy_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    area_list_ = area_list[1:]
    mAP = sum(area_list_) / len(area_list_)
    args = (mAP * 100.0, )
    plt.title('Confusion Matrix\nmAP = %0.02f' % args, y=1.19)

    fig_filename = 'labeler-precision-recall-roc.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


@register_ibs_method
def background_accuracy_display(ibs, category_list, test_gid_set=None):
    if test_gid_set is None:
        test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'))
        test_gid_set = list(test_gid_set)
    aids_list = ibs.get_image_aids(test_gid_set)
    aid_list = ut.flatten(aids_list)
    species_list = ibs.get_annot_species_texts(aid_list)

    aid_list = [
        aid
        for aid, species in zip(aid_list, species_list)
        if species in category_list
    ]
    species_list = ibs.get_annot_species_texts(aid_list)
    gid_list = ibs.get_annot_gids(aid_list)

    config2_ = {
        'fw_detector': 'cnn'
    }
    hough_cpath_list = ibs.get_annot_probchip_fpath(aid_list, config2_=config2_)
    image_list = [vt.imread(hough_cpath) for hough_cpath in hough_cpath_list]
    chip_list = ibs.get_annot_chips(aid_list, config2_=config2_)
    zipped = zip(aid_list, gid_list, species_list, image_list, chip_list)
    for index, (aid, gid, species, image, chip) in enumerate(zipped):
        print(index)
        canvas = vt.blend_images_multiply(chip, vt.resize_mask(image, chip))
        canvas *= 255.0
        canvas = np.around(canvas)
        canvas[canvas < 0] = 0
        canvas[canvas > 255] = 255
        canvas = canvas.astype(np.uint8)
        cv2.imwrite('/home/jason/Desktop/background/background.%s.%d.%d.png' % (species, gid, aid, ), canvas)


def aoi2_precision_recall_algo(ibs, category_list=None, **kwargs):
    depc = ibs.depc_annot
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
    print('Processing Precision-Recall for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = aoi2_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def aoi2_roc_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing ROC for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = aoi2_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, fpr_list, tpr_list, interpolate=False,
                                  target=(0.0, 1.0), **kwargs)


def aoi2_confusion_matrix_algo_plot(ibs, label, color, conf, output_cases=False, category_list=None, **kwargs):
    print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf, ))
    depc = ibs.depc_annot
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
        'positive' if confidence >= conf else 'negative'
        for confidence in confidence_list
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
            manifest_dict[test_gid][test_aid] = (label, prediction, )

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

            image_filename = 'image_%d.png' % (test_gid, )
            image_filepath = join(output_path, image_filename)
            cv2.imwrite(image_filepath, image)

    category_list = ['positive', 'negative']
    category_mapping = {
        'positive': 0,
        'negative': 1,
    }
    return general_confusion_matrix_algo(label_list, prediction_list, category_list,
                                         category_mapping, size=20, **kwargs)


@register_ibs_method
def aoi2_precision_recall_algo_display(ibs, output_cases=False, figsize=(20, 20)):
    import matplotlib.pyplot as plt
    import plottool as pt

    fig_ = plt.figure(figsize=figsize)

    config_list = [
        {'label': 'All Species',         'category_list': None},
        {'label': 'Masai Giraffe',       'category_list': ['giraffe_masai']},
        {'label': 'Reticulated Giraffe', 'category_list': ['giraffe_reticulated']},
        {'label': 'Sea Turtle',          'category_list': ['turtle_sea']},
        {'label': 'Whale Fluke',         'category_list': ['whale_fluke']},
        {'label': 'Grevy\'s Zebra',      'category_list': ['zebra_grevys']},
        {'label': 'Plains Zebra',        'category_list': ['zebra_plains']},
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
        aoi2_precision_recall_algo_plot(ibs, color=color, **config)
        for color, config in zip(color_list, config_list)
    ]
    area_list = [ ret[0] for ret in ret_list ]
    conf_list = [ ret[1] for ret in ret_list ]
    # index = np.argmax(area_list)
    index = 0
    best_label1 = config_list[index]['label']
    best_config1 = config_list[index]
    best_color1 = color_list[index]
    best_area1 = area_list[index]
    best_conf1 = conf_list[index]
    plt.title('Precision-Recall Curve (Best: %s, AP = %0.02f)' % (best_label1, best_area1, ), y=1.10)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

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
    area_list = [ ret[0] for ret in ret_list ]
    conf_list = [ ret[1] for ret in ret_list ]
    # index = np.argmax(area_list)
    index = 0
    best_label2 = config_list[index]['label']
    best_config2 = config_list[index]
    best_color2 = color_list[index]
    best_area2 = area_list[index]
    best_conf2 = conf_list[index]
    plt.title('ROC Curve (Best: %s, AP = %0.02f)' % (best_label2, best_area2, ), y=1.10)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)
    plt.plot([0.0, 1.0], [0.0, 1.0], color=(0.5, 0.5, 0.5), linestyle='--')
    axes_ = plt.subplot(223)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = aoi2_confusion_matrix_algo_plot(ibs, color=best_color1,
                                                      conf=best_conf1, fig_=fig_, axes_=axes_,
                                                      output_cases=output_cases, **best_config1)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    plt.title('P-R Confusion Matrix (OP = %0.02f)' % (best_conf1, ), y=1.12)

    axes_ = plt.subplot(224)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = aoi2_confusion_matrix_algo_plot(ibs, color=best_color2,
                                                      conf=best_conf2, fig_=fig_, axes_=axes_,
                                                      **best_config2)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    plt.title('ROC Confusion Matrix (OP = %0.02f)' % (best_conf2, ), y=1.12)

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
                'gid'        : gid,
                'xtl'        : bbox[0] / width,
                'ytl'        : bbox[1] / height,
                'xbr'        : (bbox[0] + bbox[2]) / width,
                'ybr'        : (bbox[1] + bbox[3]) / height,
                'width'      : bbox[2] / width,
                'height'     : bbox[3] / height,
                'class'      : ibs.get_annot_species_texts(aid),
                'viewpoint'  : ibs.get_annot_viewpoints(aid),
                'confidence' : 1.0,
            }
            gt_list.append(temp)
        gt_dict[uuid] = gt_list
    return gt_dict


def detector_parse_pred(ibs, test_gid_list=None, **kwargs):
    depc = ibs.depc_image

    if test_gid_list is None:
        test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
    uuid_list = ibs.get_image_uuids(test_gid_list)

    # depc.delete_property('detections', test_gid_list, config=kwargs)
    results_list = depc.get_property('detections', test_gid_list, None, config=kwargs)
    size_list = ibs.get_image_sizes(test_gid_list)
    zipped_list = zip(results_list)
    # Reformat results for json
    results_list = [
        [
            {
                'gid'        : test_gid,
                'xtl'        : bbox[0] / width,
                'ytl'        : bbox[1] / height,
                'width'      : bbox[2] / width,
                'height'     : bbox[3] / height,
                'theta'      : theta,  # round(theta, 4),
                'confidence' : conf,   # round(conf, 4),
                'class'      : class_,
                'viewpoint'  : viewpoint,
            }
            for bbox, theta, class_, viewpoint, conf in zip(*zipped[0][1:])
        ]
        for zipped, (width, height), test_gid in zip(zipped_list, size_list, test_gid_list)
    ]

    pred_dict = {
        uuid_ : result_list
        for uuid_, result_list in zip(uuid_list, results_list)
    }
    # print(pred_dict)
    return pred_dict


def detector_precision_recall_algo(ibs, samples=SAMPLES, force_serial=FORCE_SERIAL, **kwargs):
    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
    uuid_list = ibs.get_image_uuids(test_gid_list)

    print('\tGather Ground-Truth')
    gt_dict = detector_parse_gt(ibs, test_gid_list=test_gid_list)

    print('\tGather Predictions')
    pred_dict = detector_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

    print('\tGenerate Curves...')
    conf_list = [ _ / float(samples) for _ in range(0, int(samples) + 1) ]
    conf_list = sorted(conf_list, reverse=True)

    uuid_list_list = [ uuid_list for _ in conf_list ]
    gt_dict_list   = [ gt_dict   for _ in conf_list ]
    pred_dict_list = [ pred_dict for _ in conf_list ]
    kwargs_list    = [ kwargs    for _ in conf_list ]
    arg_iter = zip(conf_list, uuid_list_list, gt_dict_list, pred_dict_list, kwargs_list)
    pr_re_gen = ut.generate2(detector_precision_recall_algo_worker, arg_iter,
                             nTasks=len(conf_list), ordered=True,
                             chunksize=CHUNK_SIZE, force_serial=force_serial)

    conf_list_ = [-1.0, -1.0]
    pr_list = [1.0, 0.0]
    re_list = [0.0, 1.0]
    # conf_list_ = []
    # pr_list = []
    # re_list = []
    for conf, pr, re in pr_re_gen:
        conf_list_.append(conf)
        pr_list.append(pr)
        re_list.append(re)

    print('...complete')
    return conf_list_, pr_list, re_list


def detector_precision_recall_algo_worker(conf, uuid_list, gt_dict, pred_dict,
                                          kwargs):
    tp, fp, fn = 0.0, 0.0, 0.0
    for index, uuid_ in enumerate(uuid_list):
        if uuid_ in pred_dict:
            pred_list = [
                pred
                for pred in pred_dict[uuid_]
                if pred['confidence'] >= conf
            ]
            tp_, fp_, fn_ = general_tp_fp_fn(gt_dict[uuid_], pred_list, **kwargs)
            tp += tp_
            fp += fp_
            fn += fn_
    pr = tp / (tp + fp)
    re = tp / (tp + fn)
    return (conf, pr, re)


def detector_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing Precision-Recall for: %r' % (label, ))
    conf_list, pr_list, re_list = detector_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def detector_confusion_matrix_algo_plot(ibs, label, color, conf, **kwargs):
    print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf, ))

    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
    uuid_list = ibs.get_image_uuids(test_gid_list)

    print('\tGather Ground-Truth')
    gt_dict = detector_parse_gt(ibs, test_gid_list=test_gid_list)

    print('\tGather Predictions')
    pred_dict = detector_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

    label_list = []
    prediction_list = []
    for index, uuid_ in enumerate(uuid_list):
        if uuid_ in pred_dict:
            gt_list = gt_dict[uuid_]
            pred_list = [
                pred
                for pred in pred_dict[uuid_]
                if pred['confidence'] >= conf
            ]
            tp, fp, fn = general_tp_fp_fn(gt_list, pred_list, **kwargs)
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
    return general_confusion_matrix_algo(label_list, prediction_list, category_list,
                                         category_mapping, **kwargs)


@register_ibs_method
def detector_precision_recall_algo_display(ibs, min_overlap=0.5, figsize=(24, 7), **kwargs):
    import matplotlib.pyplot as plt

    fig_ = plt.figure(figsize=figsize)

    axes_ = plt.subplot(131)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall (Ground-Truth IOU >= %0.02f)' % (min_overlap, ))
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])

    kwargs_list = [
        {
            'min_overlap'            : min_overlap,
            'classifier_sensitivity' : 0.64,
            'localizer_grid'         : False,
            'localizer_sensitivity'  : 0.16,
            'labeler_sensitivity'    : 0.42,
        },
        {
            'min_overlap'            : min_overlap,
            'classifier_sensitivity' : 0.64,
            'localizer_grid'         : False,
            'localizer_sensitivity'  : 0.16,
            'labeler_sensitivity'    : 0.42,
            'check_species'          : True,
        },
        {
            'min_overlap'            : min_overlap,
            'classifier_sensitivity' : 0.64,
            'localizer_grid'         : False,
            'localizer_sensitivity'  : 0.16,
            'labeler_sensitivity'    : 0.42,
            'check_viewpoint'        : True,
        },
        {
            'min_overlap'            : min_overlap,
            'classifier_sensitivity' : 0.04,
            'localizer_grid'         : True,
            'localizer_sensitivity'  : 0.05,
            'labeler_sensitivity'    : 0.39,
        },
        {
            'min_overlap'            : min_overlap,
            'classifier_sensitivity' : 0.04,
            'localizer_grid'         : True,
            'localizer_sensitivity'  : 0.05,
            'labeler_sensitivity'    : 0.39,
            'check_species'          : True,
        },
        {
            'min_overlap'            : min_overlap,
            'classifier_sensitivity' : 0.04,
            'localizer_grid'         : True,
            'localizer_sensitivity'  : 0.05,
            'labeler_sensitivity'    : 0.39,
            'check_viewpoint'        : True,
        },
    ]
    label_list = [
        'Opt L',
        'Opt L+S',
        'Opt L+S+V',
        'Rec L',
        'Rec L+S',
        'Rec L+S+V',
    ]
    color_list = [
        'r',
        'b',
        'g',
        'k',
        'y',
        'c',
    ]
    ret_list = [
        detector_precision_recall_algo_plot(ibs, label=label, color=color, **kwargs_)
        for label, color, kwargs_ in zip(label_list, color_list, kwargs_list)
    ]

    area_list = [ ret[0] for ret in ret_list ]
    conf_list = [ ret[1] for ret in ret_list ]
    index = np.argmax(area_list)
    best_label = label_list[index]
    best_kwargs = kwargs_list[index]
    best_area = area_list[index]
    best_conf = conf_list[index]
    plt.title('Precision-Recall Curve (Best: %s, AP = %0.02f)' % (best_label, best_area, ), y=1.20)
    # Display graph
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(132)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = detector_confusion_matrix_algo_plot(ibs, 'V1', 'r', conf=best_conf, fig_=fig_, axes_=axes_, **best_kwargs)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    plt.title('P-R Confusion Matrix (Algo: %s, OP = %0.02f)' % (best_label, best_conf, ), y=1.26)

    best_index = None
    best_conf = None
    best_pr = 0.0
    best_re = 0.0
    tup_list  = [ ret[2] for ret in ret_list ]
    for index, tup in enumerate(tup_list):
        for conf, re, pr in zip(*tup):
            if pr > best_pr:
                best_index = index
                best_conf = conf
                best_pr = pr
                best_re = re

    if best_index is not None:
        axes_ = plt.subplot(131)
        plt.plot([best_re], [best_pr], 'yo')

        best_label = label_list[best_index]
        best_kwargs = kwargs_list[best_index]

        axes_ = plt.subplot(133)
        axes_.set_aspect(1)
        gca_ = plt.gca()
        gca_.grid(False)
        correct_rate, _ = detector_confusion_matrix_algo_plot(ibs, 'V1', 'r', conf=best_conf, fig_=fig_, axes_=axes_, **best_kwargs)
        axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
        axes_.set_ylabel('Ground-Truth')
        plt.title('P-R Confusion Matrix (Algo: %s, OP = %0.02f)' % (best_label, best_conf, ), y=1.26)

    # plt.show()
    fig_filename = 'detector-precision-recall-%0.2f.png' % (min_overlap, )
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


@register_ibs_method
def detector_metric_graphs(ibs, species_list=[]):
    ibs.classifier_precision_recall_algo_display(species_list)
    ibs.localizer_precision_recall_algo_display()
    ibs.labeler_precision_recall_algo_display()
    ibs.detector_precision_recall_algo_display()


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
        set(ibs.get_annot_species_texts(aid_list_))
        for aid_list_ in aids_list
    ]
    label_list = [
        [
            1.0 if category in species_set else 0.0
            for category in category_list
        ]
        for species_set in species_set_list
    ]
    label_list = np.array(label_list)

    # Return values
    return train_gid_set, data_list, label_list


@register_ibs_method
def classifier2_train_image_rf(ibs, species_list, output_path=None, dryrun=False,
                               n_estimators=100):
    from sklearn import ensemble, preprocessing

    # Load data
    print('Loading pre-trained features for images')

    # Save model pickle
    if output_path is None:
        output_path = abspath(expanduser(join('~', 'code', 'ibeis', 'models')))
    ut.ensuredir(output_path)
    species_list = [species.lower() for species in species_list]
    species_list_str = '.'.join(species_list)

    args = (species_list_str, n_estimators, )
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
        model = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                max_features=None)
        model.fit(data_list, label_list)

        model_tup = (model, scaler, )
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
        print('Accuracy: %0.02f' % (accuracy, ))
        print('\t TP: % 4d (%0.02f %%)' % (tp, tp / pos, ))
        print('\t FN: % 4d (%0.02f %%)' % (fn, fn / neg, ))
        print('\t TN: % 4d (%0.02f %%)' % (tn, tn / neg, ))
        print('\t FP: % 4d (%0.02f %%)' % (fp, fp / pos, ))

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
        output_filepath = ibs.classifier2_train_image_rf(species_list,
                                                         n_estimators=n_estimators,
                                                         **kwargs)
        output_filepath_list.append(output_filepath)

        if precompute:
            config = {
                'classifier_two_algo'            : 'rf',
                'classifier_two_weight_filepath' : output_filepath,
            }
            depc.get_rowids('classifier_two', test_gid_list, config=config)

    return output_filepath_list


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
        set(ibs.get_annot_species_texts(aid_list_))
        for aid_list_ in aids_list
    ]
    label_list = [
        1 if len(species_set & category_set) else 0
        for species_set in species_set_list
    ]
    label_list = np.array(label_list)

    # Return values
    return train_gid_set, data_list, label_list


@register_ibs_method
def classifier_train_image_svm(ibs, species_list, output_path=None, dryrun=False,
                               C=1.0, kernel='rbf'):
    from sklearn import svm, preprocessing

    # Load data
    print('Loading pre-trained features for images')

    # Save model pickle
    if output_path is None:
        output_path = abspath(expanduser(join('~', 'code', 'ibeis', 'models')))
    ut.ensuredir(output_path)
    species_list = [species.lower() for species in species_list]
    species_list_str = '.'.join(species_list)
    kernel = str(kernel.lower())

    args = (species_list_str, kernel, C, )
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

        model_tup = (model, scaler, )
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
        print('Accuracy: %0.02f' % (accuracy, ))
        print('\t TP: % 4d (%0.02f %%)' % (tp, tp / pos, ))
        print('\t FN: % 4d (%0.02f %%)' % (fn, fn / neg, ))
        print('\t TN: % 4d (%0.02f %%)' % (tn, tn / neg, ))
        print('\t FP: % 4d (%0.02f %%)' % (fp, fp / pos, ))

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
        output_filepath = ibs.classifier_train_image_svm(species_list, C=C,
                                                         kernel=kernel,
                                                         **kwargs)
        output_filepath_list.append(output_filepath)

        if precompute:
            config = {
                'algo'                       : '_COMBINED',
                'features'                   : True,
                'feature2_algo'              : 'resnet',
                'feature2_chip_masking'      : False,
                'classify'                   : True,
                'classifier_algo'            : 'svm',
                'classifier_masking'         : False,
                'classifier_weight_filepath' : output_filepath,
            }
            depc.get_rowids('localizations_features', test_gid_list, config=config)
            depc.get_rowids('localizations_classifier', test_gid_list, config=config)
            # config['feature2_chip_masking'] = True
            # config['classifier_masking'] = True
            # depc.get_rowids('localizations_features', test_gid_list, config=config)
            # depc.get_rowids('localizations_classifier', test_gid_list, config=config)

    return output_filepath_list


@register_ibs_method
def bootstrap_pca_train(ibs, dims=64, pca_limit=500000, ann_batch=50,
                        output_path=None, **kwargs):
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
            'algo'         : '_COMBINED',
            'features'     : True,
            'feature2_algo': 'resnet',
        }
        total = 0
        features_list = []
        index_list = []
        gid_iter = ut.ProgIter(gid_list_, lbl='collect feature vectors', bs=True)
        for gid in gid_iter:
            if limit is not None and total >= limit:
                break
            feature_list = depc.get_property('localizations_features', gid,
                                             'vector', config=config)
            total += len(feature_list)
            index_list += [
                (gid, offset, )
                for offset in range(len(feature_list))
            ]
            features_list.append(feature_list)
        print('\nUsed %d images to mine %d features' % (len(features_list), total, ))
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
    print('PCA Variance Quality: %0.04f %%' % (pca_quality, ))

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
        print('Slicing index range: [%r, %r)' % (start_index, stop_index, ))

        # Slice gids and get feature data
        gid_list_ = gid_list[start_index: stop_index]
        total, data_list, index_list = _get_data(depc, gid_list_)

        # Scaler
        data_list = scaler.transform(data_list)

        # Transform data to smaller vectors
        data_list_ = pca_model.transform(data_list)

        zipped = zip(index_list, data_list_)
        data_iter = ut.ProgIter(zipped, lbl='add vectors to ANN model', bs=True)
        for (gid, offset), feature in data_iter:
            ann_model.add_item(index, feature)
            manifest_dict[index] = (gid, offset, )
            index += 1

    # Build forest
    trees = index // 100000
    print('Build ANN model using %d feature vectors and %d trees' % (index, trees, ))
    ann_model.build(trees)

    # Save forest
    if output_path is None:
        output_path = abspath(expanduser(join('~', 'code', 'ibeis', 'models')))

    scaler_filename = 'forest.pca'
    scaler_filepath = join(output_path, scaler_filename)
    print('Saving scaler model to: %r' % (scaler_filepath, ))
    model_tup = (pca_model, scaler, manifest_dict, )
    ut.save_cPkl(scaler_filepath, model_tup)

    forest_filename = 'forest.ann'
    forest_filepath = join(output_path, forest_filename)
    print('Saving ANN model to: %r' % (forest_filepath, ))
    ann_model.save(forest_filepath)

    # ibs.bootstrap_pca_test(model_path=output_path)
    return output_path


@register_ibs_method
def bootstrap_pca_test(ibs, dims=64, pca_limit=500000, ann_batch=50,
                       model_path=None, output_path=None, neighbors=1000,
                       nms_thresh=0.5, min_confidence=0.3, **kwargs):
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
        model_path = abspath(expanduser(join('~', 'code', 'ibeis', 'models')))

    scaler_filename = 'forest.pca'
    scaler_filepath = join(model_path, scaler_filename)
    print('Loading scaler model from: %r' % (scaler_filepath, ))
    model_tup = ut.load_cPkl(scaler_filepath)
    pca_model, scaler, manifest_dict = model_tup

    forest_filename = 'forest.ann'
    forest_filepath = join(model_path, forest_filename)
    print('Loading ANN model from: %r' % (forest_filepath, ))
    ann_model = AnnoyIndex(dims)
    ann_model.load(forest_filepath)

    config = {
        'algo'         : '_COMBINED',
        'features'     : True,
        'feature2_algo': 'resnet',
        'classify'     : True,
        'classifier_algo': 'svm',
        'classifier_weight_filepath': '/home/jason/code/ibeis/models-bootstrap/classifier.svm.image.zebra.pkl',
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
        best_idx_list = index_list[-1 * example_limit:]

        print('Worst ovelap: %r' % (overlap[:, worst_idx_list], ))
        print('Best ovelap:  %r' % (overlap[:, best_idx_list], ))

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
                neighbor_manifest_list = list(set([
                    manifest_dict[neighbor_index]
                    for neighbor_index in neighbor_index_list
                ]))
                neighbor_gid_list_ = ut.take_column(neighbor_manifest_list, 0)
                neighbor_gid_list_ = [gid] + neighbor_gid_list_
                neighbor_uuid_list_ = ibs.get_image_uuids(neighbor_gid_list_)
                neighbor_offset_list_ = ut.take_column(neighbor_manifest_list, 1)
                neighbor_offset_list_ = [offset] + neighbor_offset_list_

                neighbor_gid_set_ = list(set(neighbor_gid_list_))
                neighbor_image_list = ibs.get_image_imgdata(neighbor_gid_set_)
                neighbor_image_dict = {
                    gid: image
                    for gid, image in zip(neighbor_gid_set_, neighbor_image_list)
                }

                neighbor_pred_dict = localizer_parse_pred(ibs, test_gid_list=neighbor_gid_set_,
                                                          **config)

                neighbor_dict = {}
                zipped = zip(neighbor_gid_list_, neighbor_uuid_list_, neighbor_offset_list_)
                for neighbor_gid, neighbor_uuid, neighbor_offset in zipped:
                    if neighbor_gid not in neighbor_dict:
                        neighbor_dict[neighbor_gid] = []
                    neighbor_pred = neighbor_pred_dict[neighbor_uuid][neighbor_offset]
                    neighbor_dict[neighbor_gid].append(neighbor_pred)

                # Perform NMS
                chip_list = []
                query_image = ibs.get_image_imgdata(gid)
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
                    chip = query_image[ytl: ybr, xtl: xbr, :]
                    chip = cv2.resize(chip, (192, 192), **warpkw)
                    chip_list.append(chip)
                except:
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
                            chip = neighbor_image[ytl: ybr, xtl: xbr, :]
                            chip = cv2.resize(chip, (192, 192), **warpkw)
                            color = (0, 255, 0) if conf >= min_confidence else (0, 0, 255)
                            cv2.rectangle(chip, (0, 0), (192, 192), color, 10)
                            chip_list.append(chip)
                        except:
                            pass

                min_chips = 16
                if len(chip_list) < min_chips:
                    continue

                chip_list = chip_list[:min_chips]
                canvas = np.hstack(chip_list)
                output_filename = 'neighbors_%d_%d.png' % (gid, offset, )
                output_filepath = join(output_path, output_filename)
                cv2.imwrite(output_filepath, canvas)


def _bootstrap_mine(ibs, gt_dict, pred_dict, scheme, reviewed_gid_dict,
                    min_overlap=0.75, max_overlap=0.25):
    import random
    ##################################################################################
    # Step 7.5: gather SVM training data from overlap images
    #           note that this step randomly subsamples new negatives, so
    #           each SVM in the ensemble is given a different set of negatives
    mined_gid_list = []
    mined_gt_list = []
    mined_pos_list = []
    mined_neg_list = []
    for image_uuid in gt_dict:
        # print('--- Processing user interaction for image %r' % (image_uuid, ))
        # Get the gt and prediction list
        gt_list = gt_dict[image_uuid]
        pred_list = pred_dict[image_uuid]

        # If never seen this image before, pick a new selection of GT bboxes
        image_gid = ibs.get_image_gids_from_uuid(image_uuid)
        if image_gid not in reviewed_gid_dict:
            # Simulate the user selecting the gt bounding box(es)
            index_list = list(range(len(gt_list)))
            if scheme == 1:
                # Pick a random bbox
                index_list_ = [random.choice(index_list)]
            elif scheme == 2:
                # Pick all gt boxes
                index_list_ = index_list[:]
            else:
                raise ValueError
            reviewed_gid_dict[image_gid] = index_list_

        # Filter based on picked bboxes for gt
        picked_index_list = reviewed_gid_dict[image_gid]
        gt_list_ = [
            gt_list[picked_index]
            for picked_index in picked_index_list
        ]

        # Calculate overlap
        overlap = general_overlap(gt_list_, pred_list)
        num_gt, num_pred = overlap.shape

        if num_gt == 0 or num_pred == 0:
            continue
        else:
            pos_idx_list = np.where(overlap >= min_overlap)[1]
            neg_idx_list = np.where(overlap <= max_overlap)[1]

            num_pos = len(pos_idx_list)
            num_neg = len(neg_idx_list)

            # Randomly sample negative chips to get new candidates
            # Most of the time (like almost always will happen)
            if num_neg > num_pos:
                np.random.shuffle(neg_idx_list)
                neg_idx_list = neg_idx_list[:num_pos]

            mined_gid_list.append(image_gid)
            mined_gt_list += gt_list_
            mined_pos_list += [pred_list[idx] for idx in pos_idx_list]
            mined_neg_list += [pred_list[idx] for idx in neg_idx_list]

    args = (len(mined_pos_list), len(mined_neg_list), len(mined_gid_list), )
    print('Mined %d positive, %d negative from %d images' % args)

    return mined_gid_list, mined_gt_list, mined_pos_list, mined_neg_list


@register_ibs_method
def bootstrap(ibs, species_list=['zebra'], N=10, rounds=20, scheme=2, ensemble=9,
              output_path=None, precompute=True, precompute_test=True,
              recompute=False, visualize=True, C=1.0, kernel='rbf', **kwargs):
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
        output_path = abspath(expanduser(join('~', 'code', 'ibeis', output_path_)))
    print('Using output_path = %r' % (output_path, ))
    if recompute:
        ut.delete(output_path)
    ut.ensuredir(output_path)

    # Get the test images for later
    depc = ibs.depc_image
    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)

    wic_model_filepath = ibs.classifier_train_image_svm(species_list, output_path=output_path, dryrun=True)
    is_wic_model_trained = exists(wic_model_filepath)
    ######################################################################################
    # Step 1: train whole-image classifier
    #         this will compute and cache any ResNet features that
    #         haven't been computed
    if not is_wic_model_trained:
        wic_model_filepath = ibs.classifier_train_image_svm(species_list, output_path=output_path)

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
        'algo'         : '_COMBINED',
        'species_set'  : set(species_list),
        'features'     : True,
        'feature2_algo': 'resnet',
        'classify'     : True,
        'classifier_algo': 'svm',
        'classifier_weight_filepath': wic_model_filepath,
        'nms'          : True,
        'nms_thresh'   : 0.50,
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
        print('Current Round %r' % (current_round, ))

        ##################################################################################
        # Step 4: gather the (unreviewed) images to review for this round
        round_gid_list = []
        temp_index = 0
        while len(round_gid_list) < N and temp_index < len(sorted_gid_list):
            temp_gid = sorted_gid_list[temp_index]
            if temp_gid not in reviewed_gid_dict:
                round_gid_list.append(temp_gid)
            temp_index += 1

        args = (len(round_gid_list), round_gid_list, )
        print('Found %d unreviewed gids: %r' % args)

        ##################################################################################
        # Step 5: add any images reviewed from a previous round

        reviewed_gid_list = reviewed_gid_dict.keys()
        args = (len(reviewed_gid_list), reviewed_gid_list, )
        print('Adding %d previously reviewed gids: %r' % args)

        # All gids that have been reviewed
        round_gid_list = reviewed_gid_list + round_gid_list

        # Get model ensemble path
        limit = len(round_gid_list)
        args = (species_list_str, limit, kernel, C, )
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
            depc.delete_property('localizations_classifier', round_gid_list, config=config)

        print('\tGather Predictions')
        pred_dict = localizer_parse_pred(ibs, test_gid_list=round_gid_list, **config)

        ##################################################################################
        # Step 8: train SVM ensemble using fresh mined data for each ensemble

        # Train models, one-by-one
        for current_ensemble in range(1, ensemble + 1):
            # Mine for a new set of (static) positives and (random) negatives
            values = _bootstrap_mine(ibs, gt_dict, pred_dict, scheme,
                                     reviewed_gid_dict, **kwargs)
            mined_gid_list, mined_gt_list, mined_pos_list, mined_neg_list = values

            if visualize:
                output_visualize_path = join(svm_model_path, 'visualize')
                ut.ensuredir(output_visualize_path)
                output_visualize_path = join(output_visualize_path, '%s' % (current_ensemble, ))
                ut.ensuredir(output_visualize_path)
                classifier_visualize_training_localizations(ibs, None,
                                                            output_path=output_visualize_path,
                                                            values=values)

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
            print('Positive Confidences: %0.02f min, %0.02f avg, %0.02f std, %0.02f max' % args)
            neg_conf_list = np.array(neg_conf_list)
            args = (
                np.min(neg_conf_list),
                np.mean(neg_conf_list),
                np.std(neg_conf_list),
                np.max(neg_conf_list),
            )
            print('Negative Confidences: %0.02f min, %0.02f avg, %0.02f std, %0.02f max' % args)

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

                print('Train Ensemble SVM (%d)' % (current_ensemble, ))
                # Train scaler
                scaler = preprocessing.StandardScaler().fit(data_list)
                data_list = scaler.transform(data_list)
                # Train model
                model = svm.SVC(C=C, kernel=kernel, probability=True)
                model.fit(data_list, label_list)

                # Save model pickle
                args = (species_list_str, limit, current_ensemble, )
                svm_model_filename = 'classifier.svm.localization.%s.%d.%d.pkl' % args
                svm_model_filepath = join(svm_model_path, svm_model_filename)
                model_tup = (model, scaler, )
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
                depc.delete_property('localizations_classifier', test_gid_list, config=config)
            depc.get_rowids('localizations_classifier', test_gid_list, config=config)

    # Return the list of used configs
    return config_list


@register_ibs_method
def bootstrap2(ibs, species_list=['zebra'],
               alpha=10, gamma=16, epsilon=0.3, rounds=20, ensemble=3, dims=64, pca_limit=1000000,
               nms_thresh_pos=0.5, nms_thresh_neg=0.90, C=1.0, kernel='rbf', theta=1.0,
               output_path=None,
               precompute=True, precompute_test=True, recompute=False, recompute_classifications=True,
               overlap_thresh_cat_1=0.75, overlap_thresh_cat_2=0.25, overlap_thresh_cat_3=0.0,
               **kwargs):
    from sklearn import svm, preprocessing
    from annoy import AnnoyIndex

    # Establish variables
    kernel = str(kernel.lower())
    species_list = [species.lower() for species in species_list]
    species_list_str = '.'.join(species_list)

    if output_path is None:
        output_path_ = 'models-bootstrap'
        output_path = abspath(expanduser(join('~', 'code', 'ibeis', output_path_)))
    print('Using output_path = %r' % (output_path, ))

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

    print('Loading scaler model from: %r' % (scaler_filepath, ))
    model_tup = ut.load_cPkl(scaler_filepath)
    pca_model, scaler, manifest_dict = model_tup

    print('Loading ANN model from: %r' % (forest_filepath, ))
    ann_model = AnnoyIndex(dims)
    ann_model.load(forest_filepath)

    # Get the test images for later
    depc = ibs.depc_image
    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', species_list, **kwargs)

    wic_model_filepath = ibs.classifier_train_image_svm(species_list, output_path=output_path, dryrun=True)
    is_wic_model_trained = exists(wic_model_filepath)
    ######################################################################################
    # Step 1: train whole-image classifier
    #         this will compute and cache any ResNet features that
    #         haven't been computed
    if not is_wic_model_trained:
        wic_model_filepath = ibs.classifier_train_image_svm(species_list, output_path=output_path)

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
        'algo'           : '_COMBINED',
        'species_set'    : set(species_list),
        # 'features'       : True,
        'features_lazy'  : True,
        'feature2_algo'  : 'resnet',
        'classify'       : True,
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
        print('Current Round %r' % (current_round, ))

        ##################################################################################
        # Step 4: gather the (unreviewed) images to review for this round
        round_gid_list = []
        temp_index = 0
        while len(round_gid_list) < alpha and temp_index < len(sorted_gid_list_):
            temp_gid = sorted_gid_list_[temp_index]
            if temp_gid not in reviewed_gid_list:
                round_gid_list.append(temp_gid)
            temp_index += 1

        args = (len(round_gid_list), round_gid_list, )
        print('Found %d unreviewed gids: %r' % args)

        ##################################################################################
        # Step 5: add any images reviewed from a previous round

        args = (len(reviewed_gid_list), reviewed_gid_list, )
        print('Adding %d previously reviewed gids: %r' % args)

        # All gids that have been reviewed
        round_gid_list = reviewed_gid_list + round_gid_list
        reviewed_gid_list = round_gid_list

        # Get model ensemble path
        limit = len(round_gid_list)
        args = (species_list_str, limit, kernel, C, )
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
                depc.delete_property('localizations_classifier', round_gid_list, config=config)

            print('\tGather Predictions')
            pred_dict = localizer_parse_pred(ibs, test_gid_list=round_gid_list, **config)

            category_dict = {}
            for image_index, image_uuid in enumerate(gt_dict.keys()):
                image_gid = ibs.get_image_gids_from_uuid(image_uuid)
                args = (image_gid, image_uuid, image_index + 1, len(round_gid_list), )
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
                cat3_idx_list = np.logical_and(overlap_thresh_cat_2 > max_overlap, max_overlap > overlap_thresh_cat_3)
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
                    args = (cat_tag, len(cat_pred_list), )
                    print('\t Working on category %r with %d predictions' % args)

                    # Add raw predictions
                    if image_gid not in category_dict[cat_tag]:
                        category_dict[cat_tag][image_gid] = []
                    category_dict[cat_tag][image_gid] += cat_pred_list

                    if cat_tag == 'cat1':
                        # Go over predictions and find neighbors, sorting into either cat1 or cat3
                        neighbor_manifest_list = []
                        cat_pred_iter = ut.ProgIter(cat_pred_list, lbl='find neighbors', bs=True)
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

                            neighbor_index_list = ann_model.get_nns_by_vector(data_list_, gamma)
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

                        args = (len(neighbor_gid_set_), len(neighbor_manifest_list), )
                        print('\t\tGetting %d images for %d neighbors' % args)
                        neighbor_pred_dict = localizer_parse_pred(ibs, test_gid_list=list(neighbor_gid_set_),
                                                                  **config)

                        zipped = zip(neighbor_gid_list_, neighbor_uuid_list_, neighbor_idx_list_)
                        for neighbor_gid, neighbor_uuid, neighbor_idx in zipped:
                            neighbor_pred = neighbor_pred_dict[neighbor_uuid][neighbor_idx]
                            cat_tag_ = 'cat1' if neighbor_pred['confidence'] >= epsilon else 'cat3'
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
                    nms_thresh = nms_thresh_pos if cat_tag in ['cat1', 'cat3'] else nms_thresh_neg
                    keep_indices_list = nms(coord_list, confs_list, nms_thresh)
                    keep_indices_set = set(keep_indices_list)
                    pred_list_ = [
                        pred
                        for index, pred in enumerate(pred_list)
                        if index in keep_indices_set
                    ]
                    cat_pred_list += pred_list_
                print('NMS Proposals (start) for category %r: %d' % (cat_tag, cat_pred_total, ))
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
                print('Category %r Confidences: %0.02f min, %0.02f avg, %0.02f std, %0.02f max' % args)
                # Overwrite GID dictionary with a list of predictions
                category_dict[cat_tag] = cat_pred_list
                cat_total = len(cat_pred_list)
                print('NMS Proposals (end) for category %r: %d' % (cat_tag, cat_total, ))

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
                print('Mining %d target negatives' % (num_target, ))

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
                print('Training with %d positives and %d (%d + %d) negatives (%0.02f split)' % args)

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
                    lbl = 'gathering training features for %s' % (label_tag, )
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
                            data_shape = (num_total, num_dims, )
                            data_list = np.zeros(data_shape, dtype=feature.dtype)
                        # Add feature and label to list
                        # data_list.append(feature)
                        data_list[index] = feature
                        index += 1
                        label_list.append(label)

                # data_list = np.array(data_list)
                label_list = np.array(label_list)

                print('Train Ensemble SVM (%d)' % (current_ensemble, ))
                # Train scaler
                scaler = preprocessing.StandardScaler().fit(data_list)
                data_list = scaler.transform(data_list)
                # Train model
                model = svm.SVC(C=C, kernel=kernel, probability=True)
                model.fit(data_list, label_list)

                # Save model pickle
                args = (current_ensemble, )
                svm_model_filename = 'classifier.svm.localization.%d.pkl' % args
                svm_model_filepath = join(svm_model_path, svm_model_filename)
                model_tup = (model, scaler, )
                ut.save_cPkl(svm_model_filepath, model_tup)

        ##################################################################################
        # Step 8: update the sorted_gid_list based on what neighbors were samples
        if len(round_neighbor_gid_hist) >= alpha:
            vals_list = [
                (
                    round_neighbor_gid_hist[neighbor_gid_],
                    neighbor_gid_,
                )
                for neighbor_gid_ in round_neighbor_gid_hist
            ]
            vals_list = sorted(vals_list, reverse=True)
            vals_list = vals_list[:alpha]
            print('Reference Histogram: %r' % (vals_list, ))
            top_referenced_neighbor_gid_list = [ _[1] for _ in vals_list ]
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

            assert len(sorted_gid_list_) == len(higher_sorted_gid_list) + len(lower_sorted_gid_list)
            assert len(sorted_gid_list_) == len(sorted_gid_list)
            args = (len(higher_sorted_gid_list), len(lower_sorted_gid_list), )
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
                depc.delete_property('localizations_classifier', test_gid_list, config=config)
            depc.get_rowids('localizations_classifier', test_gid_list, config=config)

    # Return the list of used configs
    return config_list


@register_ibs_method
def classifier_visualize_training_localizations(ibs, classifier_weight_filepath,
                                                species_list=['zebra'], scheme=2,
                                                output_path=None, values=None,
                                                **kwargs):
    def _draw(image_dict, list_, color):
        for _ in list_:
            vals = _['gid'], _['xbr'], _['ybr'], _['xtl'], _['ytl']
            gid, xbr, ybr, xtl, ytl = vals
            height, width = image_dict[gid].shape[:2]
            xbr = int(xbr * width)
            ybr = int(ybr * height)
            xtl = int(xtl * width)
            ytl = int(ytl * height)
            image = image_dict[gid]
            cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, 4)

    def _write_chips(chip_list, output_path_fmt_str):
        interpolation = cv2.INTER_LANCZOS4
        warpkw = dict(interpolation=interpolation)
        chip_list = []
        for _ in list_:
            vals = _['gid'], _['xbr'], _['ybr'], _['xtl'], _['ytl']
            gid, xbr, ybr, xtl, ytl = vals
            height, width = image_dict[gid].shape[:2]
            xbr = int(xbr * width)
            ybr = int(ybr * height)
            xtl = int(xtl * width)
            ytl = int(ytl * height)
            image = image_dict[gid]
            # Get chips
            chip = image[ytl: ybr, xtl: xbr, :]
            chip = cv2.resize(chip, (192, 192), **warpkw)
            chip_list.append(chip)
        return chip_list

    # Get output path
    if output_path is None:
        output_path = abspath(expanduser(join('~', 'Desktop', 'output-bootstrap')))
    ut.delete(output_path)
    ut.ensuredir(output_path)

    if values is None:
        # Load data
        print('Loading pre-trained features for filtered localizations')
        train_gid_list = general_get_imageset_gids(ibs, 'TRAIN_SET', **kwargs)
        train_gid_list = train_gid_list[:10]

        config = {
            'algo'         : '_COMBINED',
            'species_set'  : set(species_list),
            'features'     : True,
            'feature2_algo': 'resnet',
            'classify'     : True,
            'classifier_algo': 'svm',
            'classifier_weight_filepath': classifier_weight_filepath,
            'nms'          : True,
            'nms_thresh'   : 0.50,
            # 'thresh'       : True,
            # 'index_thresh' : 0.25,
        }

        print('\tGather Ground-Truth')
        gt_dict = general_parse_gt(ibs, test_gid_list=train_gid_list, **config)

        print('\tGather Predictions')
        pred_dict = localizer_parse_pred(ibs, test_gid_list=train_gid_list, **config)

        print('Mine proposals')
        reviewed_gid_dict = {}
        values = _bootstrap_mine(ibs, gt_dict, pred_dict, scheme,
                                 reviewed_gid_dict, **kwargs)

    mined_gid_list, mined_gt_list, mined_pos_list, mined_neg_list = values

    print('Prepare images')
    # Get images and a dictionary based on their gids
    image_list = ibs.get_image_imgdata(mined_gid_list)
    image_dict = { gid: image for gid, image in zip(mined_gid_list, image_list) }

    # Draw positives
    list_ = mined_pos_list
    color = (0, 255, 0)
    chip_list = _draw(image_dict, list_, color)
    pos_path = join(output_path, 'positives')
    ut.ensuredir(pos_path)
    _write_chips(chip_list, join(pos_path, 'chips_pos_%05d.png'))

    # Draw negatives
    list_ = mined_neg_list
    color = (0, 0, 255)
    chip_list = _draw(image_dict, list_, color)
    neg_path = join(output_path, 'negatives')
    ut.ensuredir(neg_path)
    _write_chips(chip_list, join(neg_path, 'chips_neg_%05d.png'))

    # Draw positives
    list_ = mined_gt_list
    color = (255, 0, 0)
    _draw(image_dict, list_, color)

    print('Write images to %r' % (output_path, ))
    # Write images to disk
    for gid in image_dict:
        output_filename = 'localizations_gid_%d.png' % (gid, )
        output_filepath = join(output_path, output_filename)
        cv2.imwrite(output_filepath, image_dict[gid])


@register_ibs_method
def classifier_cameratrap_train(ibs, positive_imageset_id, negative_imageset_id, **kwargs):
    from ibeis_cnn.ingest_ibeis import get_cnn_classifier_cameratrap_binary_training_images
    from ibeis_cnn.process import numpy_processed_directory2
    from ibeis_cnn.models.classifier import train_classifier
    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_cnn_classifier_cameratrap_binary_training_images(ibs,
                                                                          positive_imageset_id,
                                                                          negative_imageset_id,
                                                                          dest_path=data_path,
                                                                          **kwargs)
    id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'classifier-cameratrap')
    model_path = train_classifier(output_path, X_file, y_file)
    # Return model path
    return model_path


@register_ibs_method
def classifier_binary_train(ibs, species_list, **kwargs):
    from ibeis_cnn.ingest_ibeis import get_cnn_classifier_binary_training_images
    from ibeis_cnn.process import numpy_processed_directory2
    from ibeis_cnn.models.classifier import train_classifier
    from ibeis_cnn.utils import save_model
    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_cnn_classifier_binary_training_images(ibs, species_list,
                                                               dest_path=data_path,
                                                               **kwargs)
    id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'classifier-binary')
    model_path = train_classifier(output_path, X_file, y_file)
    # Add the species_list to the model
    model_state = ut.load_cPkl(model_path)
    assert 'species_list' not in model_state
    model_state['species_list'] = species_list
    save_model(model_state, model_path)
    # Return model path
    return model_path


@register_ibs_method
def classifier2_train(ibs, species_list=None, **kwargs):
    from ibeis_cnn.ingest_ibeis import get_cnn_classifier2_training_images
    from ibeis_cnn.process import numpy_processed_directory3
    from ibeis_cnn.models.classifier2 import train_classifier2
    from ibeis_cnn.utils import save_model
    if species_list is not None:
        species_list = sorted(species_list)
    data_path = join(ibs.get_cachedir(), 'extracted')
    values = get_cnn_classifier2_training_images(ibs, species_list,
                                                 dest_path=data_path,
                                                 **kwargs)
    extracted_path, category_list = values
    id_file, X_file, y_file = numpy_processed_directory3(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'classifier2')
    model_path = train_classifier2(output_path, X_file, y_file, purge=True)
    # Add the species_list to the model
    model_state = ut.load_cPkl(model_path)
    assert 'category_list' not in model_state
    model_state['category_list'] = category_list
    save_model(model_state, model_path)
    # Return model path
    return model_path


@register_ibs_method
def classifier_train(ibs, **kwargs):
    return ibs.classifier2_train(**kwargs)


@register_ibs_method
def localizer_train(ibs, species_list=None, **kwargs):
    from pydarknet import Darknet_YOLO_Detector
    data_path = ibs.export_to_xml(species_list=species_list, **kwargs)
    output_path = join(ibs.get_cachedir(), 'training', 'localizer')
    ut.ensuredir(output_path)
    dark = Darknet_YOLO_Detector()
    model_path = dark.train(data_path, output_path)
    del dark
    return model_path


@register_ibs_method
def labeler_train(ibs, species_list=None, viewpoint_mapping=None, **kwargs):
    from ibeis_cnn.ingest_ibeis import get_cnn_labeler_training_images
    from ibeis_cnn.process import numpy_processed_directory2
    from ibeis_cnn.models.labeler import train_labeler
    from ibeis_cnn.utils import save_model
    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_cnn_labeler_training_images(ibs, data_path,
                                                     category_list=species_list,
                                                     viewpoint_mapping=viewpoint_mapping,
                                                     **kwargs)
    id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'labeler')
    model_path = train_labeler(output_path, X_file, y_file)
    # Add the species_list to the model
    model_state = ut.load_cPkl(model_path)
    assert 'category_list' not in model_state
    model_state['category_list'] = species_list
    assert 'viewpoint_mapping' not in model_state
    model_state['viewpoint_mapping'] = viewpoint_mapping
    save_model(model_state, model_path)
    return model_path


# @register_ibs_method
# def qualifier_train(ibs, **kwargs):
#     from ibeis_cnn.ingest_ibeis import get_cnn_qualifier_training_images
#     from ibeis_cnn.process import numpy_processed_directory2
#     from ibeis_cnn.models.qualifier import train_qualifier
#     data_path = join(ibs.get_cachedir(), 'extracted')
#     extracted_path = get_cnn_qualifier_training_images(ibs, data_path, **kwargs)
#     id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
#     output_path = join(ibs.get_cachedir(), 'training', 'qualifier')
#     model_path = train_qualifier(output_path, X_file, y_file)
#     return model_path


@register_ibs_method
def detector_train(ibs):
    results = ibs.localizer_train()
    localizer_weight_path, localizer_config_path, localizer_class_path = results
    classifier_model_path = ibs.classifier_binary_train()
    labeler_model_path = ibs.labeler_train()
    output_path = join(ibs.get_cachedir(), 'training', 'detector')
    ut.ensuredir(output_path)
    ut.copy(localizer_weight_path, join(output_path, 'localizer.weights'))
    ut.copy(localizer_config_path, join(output_path, 'localizer.config'))
    ut.copy(localizer_class_path,  join(output_path, 'localizer.classes'))
    ut.copy(classifier_model_path, join(output_path, 'classifier.npy'))
    ut.copy(labeler_model_path,    join(output_path, 'labeler.npy'))


@register_ibs_method
def background_train(ibs, species):
    from ibeis_cnn.ingest_ibeis import get_background_training_patches2
    from ibeis_cnn.process import numpy_processed_directory2
    from ibeis_cnn.models.background import train_background
    from ibeis_cnn.utils import save_model
    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_background_training_patches2(ibs, species, data_path,
                                                      patch_size=50,
                                                      global_limit=500000)
    id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'background')
    model_path = train_background(output_path, X_file, y_file)
    model_state = ut.load_cPkl(model_path)
    assert 'species' not in model_state
    model_state['species'] = species
    save_model(model_state, model_path)
    return model_path


@register_ibs_method
def aoi_train(ibs, species_list=None):
    from ibeis_cnn.ingest_ibeis import get_aoi_training_data
    from ibeis_cnn.process import numpy_processed_directory4
    from ibeis_cnn.models.aoi import train_aoi
    from ibeis_cnn.utils import save_model
    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_aoi_training_data(ibs, data_path, target_species_list=species_list)
    id_file, X_file, y_file = numpy_processed_directory4(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'aoi')
    model_path = train_aoi(output_path, X_file, y_file)
    model_state = ut.load_cPkl(model_path)
    assert 'species_list' not in model_state
    model_state['species_list'] = species_list
    save_model(model_state, model_path)
    return model_path


@register_ibs_method
def aoi2_train(ibs, species_list=None):
    from ibeis_cnn.ingest_ibeis import get_aoi2_training_data
    from ibeis_cnn.process import numpy_processed_directory5
    from ibeis_cnn.models.aoi2 import train_aoi2
    from ibeis_cnn.utils import save_model
    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_aoi2_training_data(ibs, dest_path=data_path, target_species_list=species_list)
    id_file, X_file, y_file = numpy_processed_directory5(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'aoi2')
    model_path = train_aoi2(output_path, X_file, y_file)
    model_state = ut.load_cPkl(model_path)
    assert 'species_list' not in model_state
    model_state['species_list'] = species_list
    save_model(model_state, model_path)
    return model_path


def _resize(image, t_width=None, t_height=None, verbose=False):
    if verbose:
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
    test_image_list = ibs.get_image_imgdata(test_gid_list)
    test_uuid_list = ibs.get_image_uuids(test_gid_list)

    write_path = abspath(expanduser(join('~', 'Desktop')))
    # gt_dict = detect_parse_gt(ibs_, test_gid_list=test_gid_list)
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


@register_ibs_method
def set_reviewed_from_target_species_count(ibs, species_set=None, target=1000):
    import random

    if species_set is None:
        species_set = set([
            'giraffe_masai',
            'giraffe_reticulated',
            'turtle_green',
            'turtle_hawksbill',
            'whale_fluke',
            'zebra_grevys',
            'zebra_plains'
        ])

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

        print('%r: %d' % (species, len(gid_list), ))

    redo = raw_input('Redo? [enter to continue] ')
    redo = redo.strip()
    if len(redo) == 0:
        ibs.set_reviewed_from_target_species_count(species_set=species_set,
                                                   target=target)
    else:
        gid_list = []
        for species in species_set:
            gid_list += species_dict.get(species, [])
        gid_list = list(set(gid_list))
        ibs.set_image_reviewed(gid_list, [1] * len(gid_list))
        ibs.update_reviewed_unreviewed_image_special_imageset()


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
